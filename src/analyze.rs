//! Instruction-trace analysis built on top of the line index: a shared parser
//! for the GumTrace/Frida-style ARM64 format this server already serves, plus
//! the analysis tools that read it (runtime strings today; taint, memory and
//! call-tree reuse the same parser).
//!
//! The parser trusts the *value annotations* the tracer already emits on each
//! line (`reg=0x..` before and after `=>`, `mem_r[0x..]`, `mem_w[0x..]=0x..`,
//! `str:"..."`, `nzcv:`), so no instruction-set model is needed: every read and
//! write a line performed is stated in the text.

use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::Command,
};

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::index::{self, LineIndex};

/// A normalized ARM64 register. `w`/`x` views collapse to the same 64-bit
/// register, and `b`/`h`/`s`/`d`/`q` views collapse to one vector register, so
/// taint over a sub-register view covers the whole register (an over-estimate
/// that never loses a dependency).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reg {
    /// General-purpose x0..x30 (w0..w30 alias here).
    Gp(u8),
    /// Stack pointer (sp/wsp).
    Sp,
    /// Vector/FP register v0..v31 (b/h/s/d/q views alias here).
    Vec(u8),
    /// The NZCV condition flags, treated as one pseudo-register so a compare
    /// that sets them and a conditional that reads them form a normal data edge.
    Flags,
}

impl Reg {
    pub fn normalize(name: &str) -> Option<Reg> {
        match name {
            "sp" | "wsp" => Some(Reg::Sp),
            "xzr" | "wzr" | "pc" | "lr" | "fp" => None,
            _ => {
                let (head, rest) = name.split_at(1);
                let num: u8 = rest.parse().ok()?;
                match head {
                    "x" | "w" if num <= 30 => Some(Reg::Gp(num)),
                    "q" | "v" | "d" | "s" | "h" | "b" if num <= 31 => Some(Reg::Vec(num)),
                    _ => None,
                }
            }
        }
    }
}

/// What kind of line this is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LineKind {
    /// A decoded CPU instruction (`module+0xoff 0xpc: mnemonic ...`).
    Instruction,
    /// A high-level JNI/JavaVM call (`>>> ...`).
    Jni,
    /// A libc/API call (`[CALL] name(args) => ret`).
    Call,
    /// A syscall (`[SVC] name(args)`).
    Svc,
    /// Blank or unrecognized.
    Other,
}

/// One parsed line. Borrows from the input line; nothing is copied until a tool
/// decides to keep it.
#[derive(Clone, Debug)]
pub struct ParsedLine<'a> {
    pub kind: LineKind,
    pub module: Option<&'a str>,
    pub offset: Option<u64>,
    pub mnemonic: Option<&'a str>,
    pub reg_reads: Vec<Reg>,
    pub reg_writes: Vec<Reg>,
    pub mem_reads: Vec<u64>,
    pub mem_writes: Vec<(u64, Option<u64>)>,
    pub writes_flags: bool,
    /// A conditional branch (b.cond / cbz / cbnz / tbz / tbnz).
    pub is_cond_branch: bool,
    pub decoded_str: Option<&'a str>,
    /// For Jni/Call/Svc lines, the text after the marker.
    pub marker: Option<&'a str>,
}

impl<'a> ParsedLine<'a> {
    fn empty(kind: LineKind) -> Self {
        ParsedLine {
            kind,
            module: None,
            offset: None,
            mnemonic: None,
            reg_reads: Vec::new(),
            reg_writes: Vec::new(),
            mem_reads: Vec::new(),
            mem_writes: Vec::new(),
            writes_flags: false,
            is_cond_branch: false,
            decoded_str: None,
            marker: None,
        }
    }
}

struct Patterns {
    header: Regex,
    str_tok: Regex,
    reg_tok: Regex,
    mem_read: Regex,
    mem_write: Regex,
}

fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(|| Patterns {
        // module+0xoffset 0xpc: <rest>
        header: Regex::new(r"^([\w.\-]+)\+0x([0-9a-fA-F]+) 0x[0-9a-fA-F]+:\s*(.*)$").unwrap(),
        // str:"...."  (greedy to the final quote on the line is wrong; take up to the closing quote)
        str_tok: Regex::new(r#"str:"((?:[^"\\]|\\.)*)""#).unwrap(),
        // a register value token: name=0xVALUE  (name is lowercase letters+digits)
        reg_tok: Regex::new(r"\b([a-z][a-z0-9]*)=0x([0-9a-fA-F]+)").unwrap(),
        mem_read: Regex::new(r"mem_r\[0x([0-9a-fA-F]+)").unwrap(),
        // address, then (anything, incl. one nested [..]) up to ]=0xVALUE
        mem_write: Regex::new(r"mem_w\[0x([0-9a-fA-F]+).*?\]=0x([0-9a-fA-F]+)").unwrap(),
    })
}

/// Parse one raw line (without needing the trailing newline stripped).
pub fn parse_line(line: &str) -> ParsedLine<'_> {
    let p = patterns();
    // Lines arrive with their trailing newline; Rust regex `$` does not skip it.
    let line = line.trim_end_matches(['\n', '\r']);
    let trimmed = line.trim_start();

    if let Some(rest) = trimmed.strip_prefix(">>>") {
        let mut pl = ParsedLine::empty(LineKind::Jni);
        pl.marker = Some(rest.trim());
        return pl;
    }
    if let Some(rest) = trimmed.strip_prefix("[CALL]") {
        let mut pl = ParsedLine::empty(LineKind::Call);
        pl.marker = Some(rest.trim());
        return pl;
    }
    if let Some(rest) = trimmed.strip_prefix("[SVC]") {
        let mut pl = ParsedLine::empty(LineKind::Svc);
        pl.marker = Some(rest.trim());
        return pl;
    }

    let caps = match p.header.captures(trimmed) {
        Some(caps) => caps,
        None => return ParsedLine::empty(LineKind::Other),
    };
    let module = caps.get(1).unwrap().as_str();
    let offset = u64::from_str_radix(caps.get(2).unwrap().as_str(), 16).ok();
    let body = caps.get(3).unwrap().as_str();

    let mut pl = ParsedLine::empty(LineKind::Instruction);
    pl.module = Some(module);
    pl.offset = offset;
    pl.mnemonic = body.split_whitespace().next();
    pl.is_cond_branch = pl
        .mnemonic
        .is_some_and(|m| m.starts_with("b.") || matches!(m, "cbz" | "cbnz" | "tbz" | "tbnz"));

    // Pull the decoded string out first so its contents can't be mis-scanned.
    let body_wo_str: std::borrow::Cow<str> = if let Some(m) = p.str_tok.captures(body) {
        pl.decoded_str = m.get(1).map(|g| g.as_str());
        p.str_tok.replace(body, "").into_owned().into()
    } else {
        body.into()
    };
    let scan = body_wo_str.as_ref();

    // Memory is classified by the mem_r / mem_w keyword, not by which side of
    // `=>` it sits on.
    for m in p.mem_read.captures_iter(scan) {
        if let Ok(addr) = u64::from_str_radix(&m[1], 16) {
            pl.mem_reads.push(addr);
        }
    }
    for m in p.mem_write.captures_iter(scan) {
        if let Ok(addr) = u64::from_str_radix(&m[1], 16) {
            let value = u64::from_str_radix(&m[2], 16).ok();
            pl.mem_writes.push((addr, value));
        }
    }

    // Registers read are value tokens before `=>`; registers written are after.
    let (left, right) = match scan.split_once(" => ") {
        Some((l, r)) => (l, r),
        None => (scan, ""),
    };
    for m in p.reg_tok.captures_iter(left) {
        if let Some(reg) = Reg::normalize(&m[1])
            && !pl.reg_reads.contains(&reg)
        {
            pl.reg_reads.push(reg);
        }
    }
    for m in p.reg_tok.captures_iter(right) {
        if let Some(reg) = Reg::normalize(&m[1])
            && !pl.reg_writes.contains(&reg)
        {
            pl.reg_writes.push(reg);
        }
    }
    // Flags as a pseudo-register: a compare/op that sets NZCV (nzcv after `=>`)
    // writes it; a conditional branch or select that reads NZCV (nzcv before
    // `=>`, or on a branch line with no `=>`) reads it. This makes the
    // compare → conditional link a normal data edge.
    pl.writes_flags = right.contains("nzcv");
    if pl.writes_flags {
        pl.reg_writes.push(Reg::Flags);
    }
    if left.contains("nzcv") {
        pl.reg_reads.push(Reg::Flags);
    }

    pl
}

// ---------------------------------------------------------------------------
// get_strings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct StringsRequest {
    /// One-based first line to scan. Defaults to 1.
    pub start_line: Option<u64>,
    /// Maximum lines to scan. Defaults to the whole file.
    pub max_scan_lines: Option<u64>,
    /// Minimum string length to keep. Defaults to 4.
    pub min_len: Option<usize>,
    /// Only keep strings containing this substring (case-sensitive).
    pub contains: Option<String>,
    /// Maximum distinct strings to return. Defaults to 1000.
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct RuntimeString {
    pub text: String,
    pub count: u64,
    pub first_line: u64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct StringsResponse {
    pub scanned_lines: u64,
    pub distinct: usize,
    pub returned: usize,
    pub truncated: bool,
    pub strings: Vec<RuntimeString>,
}

/// Hard ceiling on distinct strings held in memory, so a hostile file cannot
/// blow up the scan.
const MAX_DISTINCT_TRACKED: usize = 200_000;

/// Scan for the runtime strings the tracer already decoded (`str:"..."`),
/// deduplicated with a count and first-seen line.
pub fn extract_strings(
    path: &Path,
    index: &LineIndex,
    request: StringsRequest,
) -> index::Result<StringsResponse> {
    use std::collections::HashMap;

    let start_line = request.start_line.unwrap_or(1).max(1);
    let min_len = request.min_len.unwrap_or(4);
    let max_results = request.max_results.unwrap_or(1000);
    let scan_budget = request.max_scan_lines.unwrap_or(u64::MAX);
    let contains = request.contains.as_deref();

    let mut reader = open_from_line(path, index, start_line)?;
    let mut line_no = start_line;
    let mut scanned = 0_u64;
    let mut buf = Vec::with_capacity(256);
    // text -> (count, first_line), insertion order kept separately for stable output.
    let mut seen: HashMap<String, (u64, u64)> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut truncated = false;

    while scanned < scan_budget {
        buf.clear();
        let read = reader.read_until(b'\n', &mut buf)?;
        if read == 0 {
            break;
        }
        scanned += 1;
        let line = String::from_utf8_lossy(&buf);
        let parsed = parse_line(&line);
        if let Some(text) = parsed.decoded_str
            && text.len() >= min_len
            && contains.is_none_or(|needle| text.contains(needle))
        {
            match seen.get_mut(text) {
                Some(entry) => entry.0 += 1,
                None => {
                    if seen.len() < MAX_DISTINCT_TRACKED {
                        seen.insert(text.to_owned(), (1, line_no));
                        order.push(text.to_owned());
                    } else {
                        truncated = true;
                    }
                }
            }
        }
        line_no += 1;
    }

    let distinct = order.len();
    let mut strings: Vec<RuntimeString> = order
        .into_iter()
        .map(|text| {
            let (count, first_line) = seen[&text];
            RuntimeString {
                text,
                count,
                first_line,
            }
        })
        .collect();
    // Most frequent first, then by first appearance.
    strings.sort_by(|a, b| b.count.cmp(&a.count).then(a.first_line.cmp(&b.first_line)));
    if strings.len() > max_results {
        truncated = true;
        strings.truncate(max_results);
    }

    Ok(StringsResponse {
        scanned_lines: scanned,
        distinct,
        returned: strings.len(),
        truncated,
        strings,
    })
}

/// A buffered sequential reader positioned at the first byte of `start_line`.
pub(crate) fn open_from_line(
    path: &Path,
    index: &LineIndex,
    start_line: u64,
) -> index::Result<BufReader<std::fs::File>> {
    use std::io::{Seek, SeekFrom};
    let mut file = index::open_sequential(path)?;
    if start_line > 1 {
        let offset = index::locate_line_offset(&mut file, index, start_line)?;
        file.seek(SeekFrom::Start(offset))?;
    }
    Ok(BufReader::with_capacity(1 << 20, file))
}

// ---------------------------------------------------------------------------
// get_memory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct MemoryRequest {
    /// Target address, hex (with or without 0x).
    pub address_hex: String,
    /// Match accesses whose address is in [address, address+length). Defaults to 1.
    pub length: Option<u64>,
    /// One-based first line to scan. Defaults to 1.
    pub start_line: Option<u64>,
    /// Maximum lines to scan. Defaults to the whole file.
    pub max_scan_lines: Option<u64>,
    /// value_at is the most recent write at or before this line. Defaults to the
    /// last write seen.
    pub up_to_line: Option<u64>,
    /// Maximum access events to return. Defaults to 500.
    pub max_events: Option<usize>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MemoryAccess {
    pub line: u64,
    pub kind: String,
    pub address: String,
    pub value: Option<String>,
    /// The complete original trace line, verbatim.
    pub line_text: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MemoryResponse {
    pub address: String,
    pub length: u64,
    pub scanned_lines: u64,
    pub reads: u64,
    pub writes: u64,
    /// The value that was live at the queried address as of up_to_line.
    pub value_at: Option<MemoryAccess>,
    pub events: Vec<MemoryAccess>,
    pub truncated: bool,
}

/// Scan the trace for every memory access to [address, address+length): their
/// read/write history and the value last written at or before `up_to_line`.
pub fn query_memory(
    path: &Path,
    index: &LineIndex,
    request: MemoryRequest,
) -> index::Result<MemoryResponse> {
    let base = parse_hex_u64(&request.address_hex).ok_or_else(|| {
        index::IndexError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("address_hex is not hex: {}", request.address_hex),
        ))
    })?;
    let length = request.length.unwrap_or(1).max(1);
    let end = base.saturating_add(length);
    let in_range = |addr: u64| addr >= base && addr < end;
    let start_line = request.start_line.unwrap_or(1).max(1);
    let scan_budget = request.max_scan_lines.unwrap_or(u64::MAX);
    let up_to_line = request.up_to_line.unwrap_or(u64::MAX);
    let max_events = request.max_events.unwrap_or(500);

    let mut reader = open_from_line(path, index, start_line)?;
    let mut line_no = start_line;
    let mut scanned = 0_u64;
    let mut buf = Vec::with_capacity(256);
    let (mut reads, mut writes) = (0_u64, 0_u64);
    let mut events: Vec<MemoryAccess> = Vec::new();
    let mut truncated = false;
    let mut value_at: Option<MemoryAccess> = None;

    while scanned < scan_budget {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        scanned += 1;
        let line = String::from_utf8_lossy(&buf);
        let parsed = parse_line(&line);
        let text = line.trim_end_matches(['\n', '\r']);

        for &addr in &parsed.mem_reads {
            if in_range(addr) {
                reads += 1;
                push_event(
                    &mut events,
                    &mut truncated,
                    max_events,
                    MemoryAccess {
                        line: line_no,
                        kind: "read".to_owned(),
                        address: format!("0x{addr:x}"),
                        value: None,
                        line_text: text.to_owned(),
                    },
                );
            }
        }
        for &(addr, value) in &parsed.mem_writes {
            if in_range(addr) {
                writes += 1;
                let access = MemoryAccess {
                    line: line_no,
                    kind: "write".to_owned(),
                    address: format!("0x{addr:x}"),
                    value: value.map(|v| format!("0x{v:x}")),
                    line_text: text.to_owned(),
                };
                if line_no <= up_to_line {
                    value_at = Some(access.clone());
                }
                push_event(&mut events, &mut truncated, max_events, access);
            }
        }
        line_no += 1;
    }

    Ok(MemoryResponse {
        address: format!("0x{base:x}"),
        length,
        scanned_lines: scanned,
        reads,
        writes,
        value_at,
        events,
        truncated,
    })
}

fn push_event(
    events: &mut Vec<MemoryAccess>,
    truncated: &mut bool,
    max: usize,
    event: MemoryAccess,
) {
    if events.len() < max {
        events.push(event);
    } else {
        *truncated = true;
    }
}

fn parse_hex_u64(text: &str) -> Option<u64> {
    let hex = text
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    u64::from_str_radix(hex, 16).ok()
}

// ---------------------------------------------------------------------------
// get_call_tree (reuses the bundled build_calltree.py unchanged)
// ---------------------------------------------------------------------------

const BUILD_CALLTREE_PY: &str = include_str!("../scripts/build_calltree.py");
const CALLTREE_RUN_PY: &str = include_str!("../scripts/calltree_run.py");

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CallTreeRequest {
    /// Only count calls within this module (e.g. "libloader.so"). Defaults to
    /// the first module seen.
    pub so: Option<String>,
    /// Tree root as a hex module offset (e.g. "321cb0"). Defaults to the first
    /// instruction's offset.
    pub entry_hex: Option<String>,
    /// Maximum trace lines to parse. Defaults to the script's own 50,000,000.
    pub max_lines: Option<u64>,
    /// Output base name under the export root. Defaults to the trace file stem.
    pub output_name: Option<String>,
    /// Lines of the generated call tree to return inline. Defaults to 60.
    pub preview_lines: Option<usize>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TopFunction {
    pub func: String,
    pub calls: u64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct CallTreeResponse {
    pub scanned_lines: u64,
    pub total_calls: u64,
    pub unique_functions: u64,
    pub entry: String,
    pub so: String,
    pub calltree_path: String,
    pub callgraph_path: String,
    pub top_functions: Vec<TopFunction>,
    pub preview: String,
}

fn find_python() -> Option<String> {
    for candidate in [
        "python3",
        "/opt/homebrew/bin/python3",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "python",
    ] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate.to_owned());
        }
    }
    None
}

/// Build the call tree/graph by running the bundled build_calltree.py, writing
/// its two output files under `export_root` and returning a summary + preview.
pub fn build_call_tree(
    trace_path: &Path,
    export_root: &Path,
    request: CallTreeRequest,
) -> Result<CallTreeResponse, String> {
    let python = find_python().ok_or_else(|| {
        "python3 was not found on PATH; install Python 3 to use get_call_tree".to_owned()
    })?;

    // Keep the output under the export root: only a bare file name is allowed.
    let base_name = match &request.output_name {
        Some(name) => Path::new(name)
            .file_name()
            .ok_or_else(|| "output_name must be a plain file name".to_owned())?
            .to_owned(),
        None => trace_path
            .file_stem()
            .ok_or_else(|| "trace path has no file name".to_owned())?
            .to_owned(),
    };
    std::fs::create_dir_all(export_root).map_err(|e| format!("cannot create export dir: {e}"))?;
    let out_base = export_root.join(&base_name);

    if let Some(entry) = &request.entry_hex {
        let hex = entry.trim_start_matches("0x");
        u64::from_str_radix(hex, 16).map_err(|_| format!("entry_hex is not hex: {entry}"))?;
    }

    // Write the bundled scripts to a throwaway dir so Python can import the
    // unchanged build_calltree module.
    let dir = tempfile::tempdir().map_err(|e| format!("cannot create temp dir: {e}"))?;
    for (name, body) in [
        ("build_calltree.py", BUILD_CALLTREE_PY),
        ("calltree_run.py", CALLTREE_RUN_PY),
    ] {
        let mut f = std::fs::File::create(dir.path().join(name))
            .map_err(|e| format!("write script: {e}"))?;
        f.write_all(body.as_bytes())
            .map_err(|e| format!("write script: {e}"))?;
    }

    let mut cmd = Command::new(&python);
    cmd.arg(dir.path().join("calltree_run.py"))
        .arg(trace_path)
        .arg(&out_base)
        .arg(request.so.as_deref().unwrap_or(""))
        .arg(request.entry_hex.as_deref().unwrap_or(""))
        .arg(request.max_lines.map(|n| n.to_string()).unwrap_or_default());
    let output = cmd
        .output()
        .map_err(|e| format!("failed to run python: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "build_calltree.py failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let summary: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("could not parse script output: {e}; output was: {stdout}"))?;
    if summary.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(summary
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("call tree build produced no calls")
            .to_owned());
    }

    let calltree_path = summary["calltree_path"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let preview_lines = request.preview_lines.unwrap_or(60).min(2000);
    let preview = std::fs::read_to_string(&calltree_path)
        .map(|text| {
            text.lines()
                .take(preview_lines)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    let top_functions = summary
        .get("top_functions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    Some(TopFunction {
                        func: item.get("func")?.as_str()?.to_owned(),
                        calls: item.get("calls")?.as_u64()?,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(CallTreeResponse {
        scanned_lines: summary["scanned_lines"].as_u64().unwrap_or(0),
        total_calls: summary["total_calls"].as_u64().unwrap_or(0),
        unique_functions: summary["unique_functions"].as_u64().unwrap_or(0),
        entry: summary["entry"].as_str().unwrap_or_default().to_owned(),
        so: summary["so"].as_str().unwrap_or_default().to_owned(),
        calltree_path,
        callgraph_path: summary["callgraph_path"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        top_functions,
        preview,
    })
}

// ---------------------------------------------------------------------------
// taint (backward dynamic data-dependency slice)
// ---------------------------------------------------------------------------

/// Above this many lines, the in-memory backward slice is refused (streaming
/// mode is future work). The real analysis targets are far smaller.
const TAINT_MAX_LINES: u64 = 5_000_000;
const TAINT_MARK: &str = "\u{bb} ";

/// Which way to follow the data flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    /// What computed the source at target_line (walk earlier lines).
    #[default]
    Backward,
    /// What the source at target_line goes on to affect (walk later lines).
    Forward,
}

/// How to write the slice file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum SliceMode {
    /// Write only the tainted lines, verbatim.
    #[default]
    Filter,
    /// Write the whole trace, prefixing tainted lines with "» ".
    Highlight,
}

impl SliceMode {
    fn emit_full(self) -> bool {
        matches!(self, SliceMode::Highlight)
    }
    fn as_str(self) -> &'static str {
        match self {
            SliceMode::Filter => "filter",
            SliceMode::Highlight => "highlight",
        }
    }
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TaintRequest {
    /// Direction of the slice. Defaults to backward.
    pub direction: Option<Direction>,
    /// One-based anchor line. backward: slice uses lines ≤ it. forward: the
    /// source is live here and propagation starts at the next line.
    pub target_line: u64,
    /// Registers that are the taint source, e.g. ["x0"]. Combine with addresses/regions.
    pub registers: Option<Vec<String>>,
    /// Single memory addresses (hex) that are the taint source.
    pub addresses: Option<Vec<String>>,
    /// Memory regions [address, address+length) that are the taint source —
    /// e.g. a whole input array or allocated buffer from list_sources.
    pub regions: Option<Vec<MemRegion>>,
    /// backward only: also follow control dependencies (the branch decisions
    /// that governed each kept line). Defaults to false (data flow only).
    pub include_control_deps: Option<bool>,
    /// How to write the output file. Defaults to filter.
    pub mode: Option<SliceMode>,
    /// Output base name under the export root. Defaults to the trace stem.
    pub output_name: Option<String>,
    /// Maximum tainted line numbers to return inline. Defaults to 1000.
    pub max_line_numbers: Option<usize>,
    /// Maximum complete tainted rows (verbatim) to return inline. Defaults to 50.
    pub max_rows: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct MemRegion {
    /// Start address, hex (with or without 0x).
    pub address_hex: String,
    /// Length in bytes; every byte address in the range becomes a source.
    pub length: u64,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TaintedRow {
    pub line: u64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TaintResponse {
    pub direction: String,
    pub control_deps: bool,
    pub target_line: u64,
    pub source: String,
    pub mode: String,
    pub total_lines: u64,
    pub scanned_lines: u64,
    pub tainted_lines: u64,
    pub first_tainted_line: Option<u64>,
    pub output_path: String,
    pub line_numbers: Vec<u64>,
    pub line_numbers_truncated: bool,
    /// The first tainted rows, complete and verbatim (no marker), for inspection.
    pub rows: Vec<TaintedRow>,
}

#[derive(Default)]
struct LineDeps {
    reg_reads: Vec<Reg>,
    reg_writes: Vec<Reg>,
    mem_reads: Vec<u64>,
    mem_writes: Vec<u64>,
    /// 0-based index of the conditional branch that governed this line, if any.
    control_branch: Option<u32>,
}

/// Cap on the number of byte addresses a region source may expand to.
const MAX_SEED_ADDRS: usize = 4_194_304;

struct SliceWrite {
    output_path: String,
    line_numbers: Vec<u64>,
    line_numbers_truncated: bool,
    rows: Vec<TaintedRow>,
    first_tainted_line: Option<u64>,
}

/// Resolve the output path `<stem>_taint.txt` under the export root.
fn slice_out_path(
    path: &Path,
    export_root: &Path,
    output_name: &Option<String>,
) -> Result<std::path::PathBuf, String> {
    let base_name = match output_name {
        Some(name) => Path::new(name)
            .file_name()
            .ok_or_else(|| "output_name must be a plain file name".to_owned())?
            .to_owned(),
        None => path
            .file_stem()
            .ok_or_else(|| "trace path has no file name".to_owned())?
            .to_owned(),
    };
    std::fs::create_dir_all(export_root).map_err(|e| format!("cannot create export dir: {e}"))?;
    // Append rather than set_extension: trace stems contain dots (libfoo.so+..).
    Ok(export_root.join(format!("{}_taint.txt", base_name.to_string_lossy())))
}

/// Stream the whole trace once and write the slice: filter mode emits only the
/// tainted lines verbatim, highlight mode emits every line with tainted ones
/// prefixed by the marker. Collects the inline previews along the way.
#[allow(clippy::too_many_arguments)]
fn write_slice(
    path: &Path,
    index: &LineIndex,
    out_path: &Path,
    emit_full: bool,
    tainted: &HashSet<u64>,
    total_lines: u64,
    max_numbers: usize,
    max_rows: usize,
) -> Result<SliceWrite, String> {
    use std::io::{BufWriter, Write as _};
    let mut line_numbers: Vec<u64> = Vec::new();
    let mut rows: Vec<TaintedRow> = Vec::new();
    let mut line_numbers_truncated = false;
    let mut first_tainted_line = None;

    let out = std::fs::File::create(out_path).map_err(|e| format!("cannot write output: {e}"))?;
    let mut writer = BufWriter::new(out);
    let mut reader = open_from_line(path, index, 1).map_err(|e| e.to_string())?;
    let mut buf = Vec::with_capacity(256);
    for n in 1..=total_lines {
        buf.clear();
        if reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        let is_tainted = tainted.contains(&n);
        if is_tainted {
            if first_tainted_line.is_none() {
                first_tainted_line = Some(n);
            }
            if line_numbers.len() < max_numbers {
                line_numbers.push(n);
            } else {
                line_numbers_truncated = true;
            }
            if rows.len() < max_rows {
                let text = String::from_utf8_lossy(&buf);
                rows.push(TaintedRow {
                    line: n,
                    text: text.trim_end_matches(['\n', '\r']).to_owned(),
                });
            }
        }
        if emit_full {
            if is_tainted {
                writer
                    .write_all(TAINT_MARK.as_bytes())
                    .map_err(|e| e.to_string())?;
            }
            writer.write_all(&buf).map_err(|e| e.to_string())?;
        } else if is_tainted {
            writer.write_all(&buf).map_err(|e| e.to_string())?;
        }
    }
    writer.flush().map_err(|e| e.to_string())?;

    Ok(SliceWrite {
        output_path: out_path.to_string_lossy().into_owned(),
        line_numbers,
        line_numbers_truncated,
        rows,
        first_tainted_line,
    })
}

/// Parse the request's registers/addresses into location sets, returning a
/// human-readable description of the source.
fn parse_source(request: &TaintRequest) -> Result<(HashSet<Reg>, HashSet<u64>, String), String> {
    let mut regs: HashSet<Reg> = HashSet::new();
    let mut mem: HashSet<u64> = HashSet::new();
    let mut parts: Vec<String> = Vec::new();
    for name in request.registers.iter().flatten() {
        let reg = Reg::normalize(&name.to_lowercase())
            .ok_or_else(|| format!("not a register name: {name}"))?;
        regs.insert(reg);
        parts.push(name.to_lowercase());
    }
    for addr in request.addresses.iter().flatten() {
        let value = parse_hex_u64(addr).ok_or_else(|| format!("address is not hex: {addr}"))?;
        mem.insert(value);
        parts.push(format!("mem[0x{value:x}]"));
    }
    for region in request.regions.iter().flatten() {
        let base = parse_hex_u64(&region.address_hex)
            .ok_or_else(|| format!("region address is not hex: {}", region.address_hex))?;
        if mem.len().saturating_add(region.length as usize) > MAX_SEED_ADDRS {
            return Err(format!(
                "taint source region is too large (over {MAX_SEED_ADDRS} bytes total)"
            ));
        }
        for offset in 0..region.length {
            mem.insert(base + offset);
        }
        parts.push(format!("mem[0x{base:x}..+0x{:x}]", region.length));
    }
    if regs.is_empty() && mem.is_empty() {
        return Err(
            "provide at least one register, address, or region as the taint source".to_owned(),
        );
    }
    Ok((regs, mem, parts.join(",")))
}

/// Dispatch to the backward or forward slice based on `direction`.
pub fn taint(
    path: &Path,
    index: &LineIndex,
    export_root: &Path,
    request: TaintRequest,
) -> Result<TaintResponse, String> {
    match request.direction.unwrap_or_default() {
        Direction::Backward => taint_backward(path, index, export_root, request),
        Direction::Forward => taint_forward(path, index, export_root, request),
    }
}

/// Core backward dynamic slice: returns the set of one-based line numbers (into
/// `deps`) that helped define the needed locations. Walks from the last record
/// to the first; a line that defines any needed location is kept, its defined
/// locations are satisfied, and its own inputs become newly needed.
fn backward_slice(
    deps: &[LineDeps],
    mut needed_regs: HashSet<Reg>,
    mut needed_mem: HashSet<u64>,
    include_control: bool,
) -> HashSet<u64> {
    let mut tainted: HashSet<u64> = HashSet::new();
    // Lines pulled in because they are the controlling branch of a kept line.
    let mut forced: HashSet<usize> = HashSet::new();
    for i in (0..deps.len()).rev() {
        let rec = &deps[i];
        let defines = rec.reg_writes.iter().any(|r| needed_regs.contains(r))
            || rec.mem_writes.iter().any(|a| needed_mem.contains(a));
        if !defines && !(include_control && forced.contains(&i)) {
            continue;
        }
        tainted.insert((i + 1) as u64);
        for r in &rec.reg_writes {
            needed_regs.remove(r);
        }
        for a in &rec.mem_writes {
            needed_mem.remove(a);
        }
        for r in &rec.reg_reads {
            needed_regs.insert(*r);
        }
        for a in &rec.mem_reads {
            needed_mem.insert(*a);
        }
        // This line was governed by its controlling branch; pull the branch in
        // (and, via the branch's own reads of the flags/tested register, the
        // compare that set the condition).
        if include_control && let Some(cb) = rec.control_branch {
            forced.insert(cb as usize);
        }
    }
    tainted
}

/// Backward dynamic data-dependency slice: starting from the registers/memory
/// named at `target_line`, keep every earlier line that helped compute them.
/// Control dependencies are not included yet (data flow only).
pub fn taint_backward(
    path: &Path,
    index: &LineIndex,
    export_root: &Path,
    request: TaintRequest,
) -> Result<TaintResponse, String> {
    let total_lines = index.line_count;
    let target_line = request.target_line;
    if target_line == 0 {
        return Err("target_line is one-based and must be at least 1".to_owned());
    }
    if target_line > total_lines {
        return Err(format!(
            "target_line {target_line} is past the end of the file ({total_lines} lines)"
        ));
    }
    if target_line > TAINT_MAX_LINES {
        return Err(format!(
            "target_line {target_line} exceeds the in-memory backward-taint limit {TAINT_MAX_LINES}; \
             use direction=forward (streaming) for very large traces"
        ));
    }
    let mode = request.mode.unwrap_or_default();
    let include_control = request.include_control_deps.unwrap_or(false);
    let (needed_regs, needed_mem, source_desc) = parse_source(&request)?;

    // Pass 1: record the reads/writes of lines 1..=target_line, and the
    // conditional branch that governs each line (for control dependencies).
    let mut deps: Vec<LineDeps> = Vec::with_capacity(target_line as usize);
    {
        let mut reader = open_from_line(path, index, 1).map_err(|e| e.to_string())?;
        let mut buf = Vec::with_capacity(256);
        let mut last_cond_branch: Option<u32> = None;
        for idx in 0..target_line {
            buf.clear();
            if reader
                .read_until(b'\n', &mut buf)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            let line = String::from_utf8_lossy(&buf);
            let parsed = parse_line(&line);
            deps.push(LineDeps {
                reg_reads: parsed.reg_reads,
                reg_writes: parsed.reg_writes,
                mem_reads: parsed.mem_reads,
                mem_writes: parsed.mem_writes.into_iter().map(|(a, _)| a).collect(),
                control_branch: last_cond_branch,
            });
            if parsed.is_cond_branch {
                last_cond_branch = Some(idx as u32);
            }
        }
    }

    // Pass 2: walk backward, matching each needed location to its definition.
    let tainted = backward_slice(&deps, needed_regs, needed_mem, include_control);

    // Pass 3: write the slice file.
    let out_path = slice_out_path(path, export_root, &request.output_name)?;
    let sw = write_slice(
        path,
        index,
        &out_path,
        mode.emit_full(),
        &tainted,
        total_lines,
        request.max_line_numbers.unwrap_or(1000),
        request.max_rows.unwrap_or(50),
    )?;

    Ok(TaintResponse {
        direction: "backward".to_owned(),
        control_deps: include_control,
        target_line,
        source: format!("{source_desc} @ L{target_line}"),
        mode: mode.as_str().to_owned(),
        total_lines,
        scanned_lines: target_line,
        tainted_lines: tainted.len() as u64,
        first_tainted_line: sw.first_tainted_line,
        output_path: sw.output_path,
        line_numbers: sw.line_numbers,
        line_numbers_truncated: sw.line_numbers_truncated,
        rows: sw.rows,
    })
}

/// Forward dynamic data-dependency taint: the source at `target_line` is live;
/// scanning forward, every line that reads a tainted location is kept and its
/// written locations become tainted, while a write with no tainted input clears
/// (kills) that location. Single streaming pass — no whole-trace state, so it
/// scales to very large traces.
pub fn taint_forward(
    path: &Path,
    index: &LineIndex,
    export_root: &Path,
    request: TaintRequest,
) -> Result<TaintResponse, String> {
    let total_lines = index.line_count;
    let target_line = request.target_line;
    if target_line == 0 {
        return Err("target_line is one-based and must be at least 1".to_owned());
    }
    if target_line > total_lines {
        return Err(format!(
            "target_line {target_line} is past the end of the file ({total_lines} lines)"
        ));
    }
    let mode = request.mode.unwrap_or_default();
    let (mut tainted_regs, mut tainted_mem, source_desc) = parse_source(&request)?;

    // Propagate forward, starting just after the source line.
    let mut tainted: HashSet<u64> = HashSet::new();
    let start = target_line + 1;
    if start <= total_lines {
        let mut reader = open_from_line(path, index, start).map_err(|e| e.to_string())?;
        let mut buf = Vec::with_capacity(256);
        let mut n = start;
        while n <= total_lines {
            buf.clear();
            if reader
                .read_until(b'\n', &mut buf)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            let line = String::from_utf8_lossy(&buf);
            let parsed = parse_line(&line);
            let reads_tainted = parsed.reg_reads.iter().any(|r| tainted_regs.contains(r))
                || parsed.mem_reads.iter().any(|a| tainted_mem.contains(a));
            if reads_tainted {
                tainted.insert(n);
                for r in &parsed.reg_writes {
                    tainted_regs.insert(*r);
                }
                for (a, _) in &parsed.mem_writes {
                    tainted_mem.insert(*a);
                }
            } else {
                // The written locations are overwritten with untainted data.
                for r in &parsed.reg_writes {
                    tainted_regs.remove(r);
                }
                for (a, _) in &parsed.mem_writes {
                    tainted_mem.remove(a);
                }
            }
            n += 1;
        }
    }

    let out_path = slice_out_path(path, export_root, &request.output_name)?;
    let sw = write_slice(
        path,
        index,
        &out_path,
        mode.emit_full(),
        &tainted,
        total_lines,
        request.max_line_numbers.unwrap_or(1000),
        request.max_rows.unwrap_or(50),
    )?;

    Ok(TaintResponse {
        direction: "forward".to_owned(),
        control_deps: false,
        target_line,
        source: format!("{source_desc} from L{target_line}"),
        mode: mode.as_str().to_owned(),
        total_lines,
        scanned_lines: total_lines.saturating_sub(target_line),
        tainted_lines: tainted.len() as u64,
        first_tainted_line: sw.first_tainted_line,
        output_path: sw.output_path,
        line_numbers: sw.line_numbers,
        line_numbers_truncated: sw.line_numbers_truncated,
        rows: sw.rows,
    })
}

// ---------------------------------------------------------------------------
// list_sources (auto-detect natural taint sources from JNI / [CALL] markers)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SourcesRequest {
    /// One-based first line to scan. Defaults to 1.
    pub start_line: Option<u64>,
    /// Maximum lines to scan. Defaults to the whole file.
    pub max_scan_lines: Option<u64>,
    /// Maximum candidates to return. Defaults to 200.
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SourceCandidate {
    pub line: u64,
    pub api: String,
    /// "alloc" | "input" | "output" | "copy".
    pub kind: String,
    pub pointer: Option<String>,
    pub size: Option<u64>,
    /// Which taint direction this source suggests.
    pub suggest_direction: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SourcesResponse {
    pub scanned_lines: u64,
    pub count: usize,
    pub truncated: bool,
    pub sources: Vec<SourceCandidate>,
}

fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_owned());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

fn parse_num(s: &str) -> Option<u64> {
    let s = s.trim();
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse::<u64>().ok(),
    }
}

/// Classify one marker line into a taint-source candidate, if it is one.
fn classify_marker(line: u64, marker: &str) -> Option<SourceCandidate> {
    static ALLOC: OnceLock<Regex> = OnceLock::new();
    static COPY: OnceLock<Regex> = OnceLock::new();
    static JNI_IN: OnceLock<Regex> = OnceLock::new();
    static JNI_OUT: OnceLock<Regex> = OnceLock::new();
    let alloc = ALLOC.get_or_init(|| {
        Regex::new(r"^(malloc|calloc|realloc|valloc|aligned_alloc)\((.*)\)\s*=>\s*(0x[0-9a-fA-F]+)")
            .unwrap()
    });
    let copy = COPY.get_or_init(|| Regex::new(r"^(memcpy|memmove)\((.*)\)").unwrap());
    let jni_in = JNI_IN.get_or_init(|| {
        Regex::new(r"->(Get\w*ArrayElements|GetStringUTFChars|GetStringChars|GetPrimitiveArrayCritical|GetDirectBufferAddress)\(.*?\)\s*=>\s*(0x[0-9a-fA-F]+)").unwrap()
    });
    let jni_out = JNI_OUT.get_or_init(|| Regex::new(r"->(Set\w*ArrayRegion)\((.*)\)").unwrap());

    if let Some(c) = alloc.captures(marker) {
        let api = c[1].to_owned();
        let args = split_args(&c[2]);
        let size = match api.as_str() {
            "calloc" if args.len() >= 2 => parse_num(&args[0])
                .zip(parse_num(&args[1]))
                .map(|(n, s)| n * s),
            "realloc" if args.len() >= 2 => parse_num(&args[1]),
            _ => args.first().and_then(|a| parse_num(a)),
        };
        let ptr = c[3].to_owned();
        return Some(SourceCandidate {
            line,
            api,
            kind: "alloc".to_owned(),
            pointer: Some(ptr.clone()),
            size,
            suggest_direction: "forward".to_owned(),
            detail: format!("allocated buffer at {ptr}"),
        });
    }
    if let Some(c) = copy.captures(marker) {
        let api = c[1].to_owned();
        let args = split_args(&c[2]);
        let src = args.get(1).cloned();
        let size = args.get(2).and_then(|a| parse_num(a));
        return Some(SourceCandidate {
            line,
            api,
            kind: "copy".to_owned(),
            pointer: src.clone(),
            size,
            suggest_direction: "forward".to_owned(),
            detail: format!(
                "copy src={} dst={}",
                src.as_deref().unwrap_or("?"),
                args.first().map(|s| s.as_str()).unwrap_or("?")
            ),
        });
    }
    if let Some(c) = jni_in.captures(marker) {
        let api = c[1].to_owned();
        let ptr = c[2].to_owned();
        return Some(SourceCandidate {
            line,
            api,
            kind: "input".to_owned(),
            pointer: Some(ptr.clone()),
            size: None,
            suggest_direction: "forward".to_owned(),
            detail: format!("JNI input buffer at {ptr} (track forward)"),
        });
    }
    if let Some(c) = jni_out.captures(marker) {
        let api = c[1].to_owned();
        let args = split_args(&c[2]);
        // Set<T>ArrayRegion(array, start, len, srcPtr)
        let len = args.get(2).and_then(|a| parse_num(a));
        let src = args.get(3).cloned();
        return Some(SourceCandidate {
            line,
            api,
            kind: "output".to_owned(),
            pointer: src.clone(),
            size: len,
            suggest_direction: "backward".to_owned(),
            detail: format!(
                "JNI output written from {} (track backward)",
                src.as_deref().unwrap_or("?")
            ),
        });
    }
    None
}

/// Scan the trace's JNI / [CALL] markers and list natural taint sources
/// (allocated buffers, JNI input arrays/strings, JNI output regions, copies).
pub fn list_sources(
    path: &Path,
    index: &LineIndex,
    request: SourcesRequest,
) -> index::Result<SourcesResponse> {
    let start_line = request.start_line.unwrap_or(1).max(1);
    let scan_budget = request.max_scan_lines.unwrap_or(u64::MAX);
    let max_results = request.max_results.unwrap_or(200);

    let mut reader = open_from_line(path, index, start_line)?;
    let mut line_no = start_line;
    let mut scanned = 0_u64;
    let mut buf = Vec::with_capacity(256);
    let mut sources: Vec<SourceCandidate> = Vec::new();
    let mut truncated = false;

    while scanned < scan_budget {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        scanned += 1;
        let line = String::from_utf8_lossy(&buf);
        let parsed = parse_line(&line);
        if matches!(parsed.kind, LineKind::Call | LineKind::Jni)
            && let Some(marker) = parsed.marker
            && let Some(candidate) = classify_marker(line_no, marker)
        {
            if sources.len() < max_results {
                sources.push(candidate);
            } else {
                truncated = true;
            }
        }
        line_no += 1;
    }

    Ok(SourcesResponse {
        scanned_lines: scanned,
        count: sources.len(),
        truncated,
        sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_store_pair_with_memory_write() {
        let line = "libloader.so+0x321cb0 0x773db52cb0: stp x29, x30, [sp, #-0x20]! x29=0x7fe4f253c0 x30=0x7af354d30c sp=0x7fe4f252a0 => sp=0x7fe4f25280  mem_w[0x7fe4f25280 [stack]]=0x7fe4f253c0\n";
        let p = parse_line(line);
        assert_eq!(p.kind, LineKind::Instruction);
        assert_eq!(p.module, Some("libloader.so"));
        assert_eq!(p.offset, Some(0x321cb0));
        assert_eq!(p.mnemonic, Some("stp"));
        assert!(p.reg_reads.contains(&Reg::Gp(29)));
        assert!(p.reg_reads.contains(&Reg::Gp(30)));
        assert!(p.reg_reads.contains(&Reg::Sp));
        assert_eq!(p.reg_writes, vec![Reg::Sp]);
        assert_eq!(p.mem_writes, vec![(0x7fe4f25280, Some(0x7fe4f253c0))]);
        assert!(p.mem_reads.is_empty());
        assert!(!p.writes_flags);
    }

    #[test]
    fn parses_load_with_memory_read_and_reg_alias() {
        let line = "libloader.so+0x320188 0x773db51188: ldr x8, [x8, #0x28] x8=0x7af8222a40 => x8=0x508eeb27e266a175  mem_r[0x7af8222a68 [anon:stack_and_tls:main]]";
        let p = parse_line(line);
        assert_eq!(p.reg_reads, vec![Reg::Gp(8)]);
        assert_eq!(p.reg_writes, vec![Reg::Gp(8)]);
        assert_eq!(p.mem_reads, vec![0x7af8222a68]);
        assert!(p.mem_writes.is_empty());
    }

    #[test]
    fn parses_flag_write_and_file_backed_read() {
        let line = "libloader.so+0x3201a4 0x773db511a4: subs x9, x0, x9 x0=0x0 x9=0x773d831000 => nzcv: N=1, Z=0, C=0, V=0 x9=0xffffff88c27cf000";
        let p = parse_line(line);
        assert!(p.reg_reads.contains(&Reg::Gp(0)));
        assert!(p.reg_reads.contains(&Reg::Gp(9)));
        // subs writes x9 and, because it sets NZCV, the Flags pseudo-register.
        assert!(p.reg_writes.contains(&Reg::Gp(9)));
        assert!(p.reg_writes.contains(&Reg::Flags));
        assert!(p.writes_flags);
        assert!(!p.is_cond_branch);

        let file_backed = "libloader.so+0x3201d0 0x773db511d0: ldr x20, [x20, #0x2e0] x20=0x773df1b000 => x20=0x773df25858  mem_r[0x773df1b2e0 split_config.arm64_v8a.apk+0x6ea2e0:ro]";
        let p2 = parse_line(file_backed);
        assert_eq!(p2.mem_reads, vec![0x773df1b2e0]);
    }

    #[test]
    fn parses_decoded_string_and_width_alias() {
        let line = "libloader.so+0x233700 0x773da64700: ldp x7, x8, [x1, #-0x10] x1=0x7fe4f24000 => x7=0x2f656d69746e7572 x8=0x616f6c2f45534142  mem_r[0x7fe4f23ff0 [stack]] str:\"runtime/BASE/loading/Setup\"";
        let p = parse_line(line);
        assert_eq!(p.decoded_str, Some("runtime/BASE/loading/Setup"));
        assert_eq!(p.reg_reads, vec![Reg::Gp(1)]);
        assert!(p.reg_writes.contains(&Reg::Gp(7)));
        assert!(p.reg_writes.contains(&Reg::Gp(8)));

        let wreg = "libloader.so+0x3201b0 0x773db511b0: sub w8, w11, w10 w11=0x3deebec0 w10=0x3d831000 => w8=0x6baec0";
        let p2 = parse_line(wreg);
        assert!(p2.reg_reads.contains(&Reg::Gp(11)));
        assert!(p2.reg_reads.contains(&Reg::Gp(10)));
        assert_eq!(p2.reg_writes, vec![Reg::Gp(8)]);
    }

    fn deps(reads: &[Reg], writes: &[Reg], mem_r: &[u64], mem_w: &[u64]) -> LineDeps {
        LineDeps {
            reg_reads: reads.to_vec(),
            reg_writes: writes.to_vec(),
            mem_reads: mem_r.to_vec(),
            mem_writes: mem_w.to_vec(),
            control_branch: None,
        }
    }

    #[test]
    fn backward_slice_follows_data_flow_through_registers_and_memory() {
        // L1: mov x1, x0        ; x1 <- x0
        // L2: add x2, x1, x3    ; x2 <- x1, x3
        // L3: mov x5, x9        ; unrelated
        // L4: str x2 -> mem[A]  ; mem[A] <- x2
        let program = [
            deps(&[Reg::Gp(0)], &[Reg::Gp(1)], &[], &[]),
            deps(&[Reg::Gp(1), Reg::Gp(3)], &[Reg::Gp(2)], &[], &[]),
            deps(&[Reg::Gp(9)], &[Reg::Gp(5)], &[], &[]),
            deps(&[Reg::Gp(2)], &[], &[], &[0xA]),
        ];
        // Source: the value stored at mem[A].
        let tainted = backward_slice(&program, HashSet::new(), HashSet::from([0xA]), false);
        // L1, L2, L4 contributed; L3 (x5<-x9) did not.
        assert_eq!(tainted, HashSet::from([1, 2, 4]));
    }

    #[test]
    fn backward_slice_stops_at_most_recent_definition() {
        // L1: mov x0, x7   (earlier def of x0, should be ignored)
        // L2: mov x0, x8   (the def that reaches the sink)
        // sink: x0
        let program = [
            deps(&[Reg::Gp(7)], &[Reg::Gp(0)], &[], &[]),
            deps(&[Reg::Gp(8)], &[Reg::Gp(0)], &[], &[]),
        ];
        let tainted = backward_slice(&program, HashSet::from([Reg::Gp(0)]), HashSet::new(), false);
        assert_eq!(tainted, HashSet::from([2]));
    }

    #[test]
    fn control_dependency_pulls_in_the_governing_branch() {
        // L1: cmp x8, x9      => writes Flags
        // L2: b.eq ...         reads Flags, governed by no earlier branch
        // L3: mov x0, #5       writes x0, governed by the b.eq at L2 (index 1)
        let flags = Reg::Flags;
        let program = [
            LineDeps {
                reg_reads: vec![Reg::Gp(8), Reg::Gp(9)],
                reg_writes: vec![flags],
                mem_reads: vec![],
                mem_writes: vec![],
                control_branch: None,
            },
            LineDeps {
                reg_reads: vec![flags],
                reg_writes: vec![],
                mem_reads: vec![],
                mem_writes: vec![],
                control_branch: None,
            },
            LineDeps {
                reg_reads: vec![],
                reg_writes: vec![Reg::Gp(0)],
                mem_reads: vec![],
                mem_writes: vec![],
                control_branch: Some(1),
            },
        ];
        // Data only: x0 does not read the flags, so only its own def is kept.
        let data = backward_slice(&program, HashSet::from([Reg::Gp(0)]), HashSet::new(), false);
        assert_eq!(data, HashSet::from([3]));
        // With control deps: the governing b.eq (L2) and the cmp (L1) it reads
        // come in too.
        let control = backward_slice(&program, HashSet::from([Reg::Gp(0)]), HashSet::new(), true);
        assert_eq!(control, HashSet::from([1, 2, 3]));
    }

    #[test]
    fn recognizes_markers() {
        assert_eq!(
            parse_line("    >>> JNIEnv->FindClass(\"qwerty/asdfgh/zxcvbn\") => 0x7af822e025 @ [libloader.so]0x340260").kind,
            LineKind::Jni
        );
        assert_eq!(
            parse_line("  [CALL] malloc(40)    => 0xb4000078db9dfd10").kind,
            LineKind::Call
        );
        assert_eq!(
            parse_line("  [SVC] getpid(0x813c7350, 0x77813c7350, 0x1, 0x0)").kind,
            LineKind::Svc
        );
        assert_eq!(parse_line("").kind, LineKind::Other);
    }

    #[test]
    fn parses_conditional_branches() {
        // b.cond: reads the flags, is a conditional branch.
        let bcond = parse_line(
            "libloader.so+0x3202c0 0x773db512c0: b.le 0x773db5159c nzcv: N=1, Z=0, C=0, V=0",
        );
        assert!(bcond.is_cond_branch);
        assert!(bcond.reg_reads.contains(&Reg::Flags));
        assert!(!bcond.writes_flags);
        // cbz: reads the tested register, is a conditional branch, no flags.
        let cbz = parse_line("libloader.so+0x9df3c 0x773d8cef3c: cbz w0, 0x773d8cef6c w0=0x0");
        assert!(cbz.is_cond_branch);
        assert!(cbz.reg_reads.contains(&Reg::Gp(0)));
        assert!(!cbz.reg_reads.contains(&Reg::Flags));
        // an unconditional branch is not a conditional branch.
        assert!(!parse_line("libloader.so+0x320284 0x773db51284: b 0x773db512a0").is_cond_branch);
    }
}
