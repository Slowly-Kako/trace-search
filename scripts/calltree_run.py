#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin runner around build_calltree.py so the MCP server can direct output to a
chosen base path (under the export root) and read back a JSON summary. The
original build_calltree.py is used unchanged.

Usage: calltree_run.py <trace_path> <out_base> [so] [entry_hex] [max_lines]
Prints one JSON object to stdout.
"""
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_calltree as bt


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "reason": "usage: calltree_run.py <trace> <out_base> ..."}))
        return
    trace = sys.argv[1]
    out_base = sys.argv[2]
    so = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    entry = int(sys.argv[4], 16) if len(sys.argv) > 4 and sys.argv[4] else None
    maxn = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 50_000_000

    edges, total, fcount, ent, dso, n = bt.parse_file(trace, so, entry, maxn)
    if total == 0:
        print(json.dumps({"ok": False, "reason": "no calls parsed (format mismatch or empty)",
                          "scanned_lines": n}))
        return
    bt.write_outputs(out_base, edges, total, fcount, ent, so or dso)
    top = sorted(fcount.items(), key=lambda kv: -kv[1])[:20]
    print(json.dumps({
        "ok": True,
        "scanned_lines": n,
        "total_calls": total,
        "unique_functions": len(fcount),
        "entry": "sub_%X" % ent,
        "so": so or dso,
        "calltree_path": out_base + "_calltree.txt",
        "callgraph_path": out_base + "_callgraph.txt",
        "top_functions": [{"func": "sub_%X" % off, "calls": cnt} for off, cnt in top],
    }))


if __name__ == '__main__':
    main()
