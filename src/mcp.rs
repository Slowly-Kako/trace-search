use std::sync::Arc;

use rmcp::{
    Json, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};

use crate::engine::{
    ExportLinesRequest, ExportLinesResponse, FileEngine, FileStatusResponse, OpenFileRequest,
    OpenFileResponse, ReadLinesRequest, ReadLinesResponse, SearchLinesRequest, SearchLinesResponse,
};

#[derive(Clone)]
pub struct TraceSearchServer {
    engine: Arc<FileEngine>,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl TraceSearchServer {
    pub fn new(engine: Arc<FileEngine>) -> Self {
        Self {
            engine,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "open_file",
        description = "Open the single active file. A valid persistent index is reused; otherwise indexing starts asynchronously. Poll get_file_status until state is ready."
    )]
    async fn open_file(
        &self,
        Parameters(request): Parameters<OpenFileRequest>,
    ) -> Result<Json<OpenFileResponse>, String> {
        self.engine
            .open_file(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "get_file_status",
        description = "Return the active file and index state, including first-pass indexing progress."
    )]
    async fn get_file_status(&self) -> Json<FileStatusResponse> {
        Json(self.engine.status().await)
    }

    #[tool(
        name = "close_file",
        description = "Close the active file or cancel an in-progress index build. Existing persistent indexes are retained."
    )]
    async fn close_file(&self) -> Json<FileStatusResponse> {
        Json(self.engine.close_file().await)
    }

    #[tool(
        name = "read_lines",
        description = "Read at most line_count lines beginning at one-based start_line. Line endings are preserved."
    )]
    async fn read_lines(
        &self,
        Parameters(request): Parameters<ReadLinesRequest>,
    ) -> Result<Json<ReadLinesResponse>, String> {
        self.engine
            .read_lines(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "search_lines",
        description = "Search the caller-provided max_scan_lines range beginning at one-based start_line using parallel indexed chunks. regex=false uses a precompiled literal search; regex=true uses Rust regex syntax."
    )]
    async fn search_lines(
        &self,
        Parameters(request): Parameters<SearchLinesRequest>,
    ) -> Result<Json<SearchLinesResponse>, String> {
        self.engine
            .search_lines(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "export_lines",
        description = "Copy at most line_count lines beginning at one-based start_line to output_path under the configured export root. Data is streamed and the destination is published atomically."
    )]
    async fn export_lines(
        &self,
        Parameters(request): Parameters<ExportLinesRequest>,
    ) -> Result<Json<ExportLinesResponse>, String> {
        self.engine
            .export_lines(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "get_strings",
        description = "Extract the runtime strings the tracer decoded inline (str:\"...\"), deduplicated with occurrence counts and the first line each appeared on. Scans max_scan_lines from start_line; filter with min_len and contains."
    )]
    async fn get_strings(
        &self,
        Parameters(request): Parameters<crate::analyze::StringsRequest>,
    ) -> Result<Json<crate::analyze::StringsResponse>, String> {
        self.engine
            .extract_strings(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "taint_analysis",
        description = "Dynamic taint slice over the active trace. The source is any combination of registers (e.g. [\"x0\"]), single addresses (hex), or regions ([{address_hex,length}] — a whole input array/buffer, e.g. straight from list_sources), anchored at target_line. direction=backward (default) keeps every earlier line that helped compute the source; direction=forward keeps every later line the source goes on to affect (single streaming pass, scales to huge traces). include_control_deps=true (backward only) additionally pulls in the branch decisions that governed each kept line (default false = data flow only). mode=filter (default) writes only the tainted lines verbatim; mode=highlight writes the whole trace with tainted lines prefixed by \"» \". The output file goes under the export root; the response returns the tainted line numbers and the first complete tainted rows verbatim."
    )]
    async fn taint_analysis(
        &self,
        Parameters(request): Parameters<crate::analyze::TaintRequest>,
    ) -> Result<Json<crate::analyze::TaintResponse>, String> {
        self.engine
            .taint_backward(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "list_sources",
        description = "Scan the active trace's JNI and [CALL] markers for natural taint sources: allocated buffers (malloc/calloc/realloc), JNI input arrays/strings (Get*ArrayElements, GetStringUTFChars), JNI output regions (Set*ArrayRegion), and memory copies. Each candidate carries its line, pointer, size when known, and the suggested taint direction, so you can feed one straight into taint_analysis."
    )]
    async fn list_sources(
        &self,
        Parameters(request): Parameters<crate::analyze::SourcesRequest>,
    ) -> Result<Json<crate::analyze::SourcesResponse>, String> {
        self.engine
            .list_sources(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "get_memory",
        description = "Scan the active trace for every read/write to [address, address+length): the access history (line, read/write, value, and the complete original line) plus value_at, the value last written at or before up_to_line. address_hex accepts 0x-prefixed or bare hex."
    )]
    async fn get_memory(
        &self,
        Parameters(request): Parameters<crate::analyze::MemoryRequest>,
    ) -> Result<Json<crate::analyze::MemoryResponse>, String> {
        self.engine
            .query_memory(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        name = "get_call_tree",
        description = "Rebuild the function call tree and call graph of the active trace from bl/blr/ret, writing <name>_calltree.txt and <name>_callgraph.txt under the export root and returning a summary plus a preview. Optional so filters by module, entry_hex sets the root offset."
    )]
    async fn get_call_tree(
        &self,
        Parameters(request): Parameters<crate::analyze::CallTreeRequest>,
    ) -> Result<Json<crate::analyze::CallTreeResponse>, String> {
        self.engine
            .build_call_tree(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TraceSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "trace-search-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Call open_file first. If indexing starts, poll get_file_status until ready. Lines are one-based and the active source is treated as an immutable snapshot.",
            )
    }
}
