# search

## Intent

The `glob`, `list_files` and `grep` model-facing tools, split one concern per file: ports `openjiuwen/harness/tools/filesystem.py` and the English descriptions from `prompts/tools/`.
`glob` delegates to `FsOperation::search_files`, `list_files` to `FsOperation::list_files`/`list_directories`, and `grep` runs an in-process `walkdir` + `regex` scanner that emits ripgrep-shaped output (no `rg` binary, no shell).

## API Surface

- `mod.rs` — module declarations, the `//!` port/deviations docs, and the re-exports `GlobArgs`/`GlobTool`, `ListDirArgs`/`ListDirTool`, `GrepArgs`/`GrepOutputMode`/`GrepTool` consumed by `crate::tools::mod` and `ToolRegistry`; shared helpers `relative_name`, `string_list`, `deserialize_flag`, `json_flag` live here (`pub(super)`).
- `glob.rs` — `GlobArgs` + `GlobTool` (`new`, `card_for`, `expand_braces`, `Tool` impl); 100-result cap, absolute-path de-duplication, brace expansion.
- `list.rs` — `ListDirArgs` + `ListDirTool`; one level, sorted names, `show_hidden` filter.
- `grep.rs` — `GrepArgs`, `GrepOutputMode`, `GrepTool` and the full scanner/context/glob-filter helpers; VCS directories (`.git`, `.svn`, `.hg`, `.bzr`, `.jj`, `.sl`) are always skipped; `apply_head_limit` is `pub(super)` because the tests call it.
- `tests.rs` — the tool-level test table moved verbatim from the old single file (glob cap/brace expansion, list hidden filter, grep modes/context/head_limit/VCS skip).

## Constraints

- Tool names, JSON schemas, payload shapes, message text and test assertions are unchanged from the pre-split module.
- `type` and `multiline` grep options still report the Python "requires ripgrep" failure; binary files are skipped by NUL sniffing.
- No cross-module dependency on `prompts`, `subagents`, `agent`, `rails` or `task_loop`.
- Files ≤1000 lines; shared helpers live only in `mod.rs`, never duplicated per tool.

## Routing Table

- `./mod.rs` → declarations, re-exports, shared path/string/serde flag helpers
- `./glob.rs` → `GlobTool`, `GlobArgs`, brace expansion
- `./list.rs` → `ListDirTool`, `ListDirArgs`, entry-name filtering
- `./grep.rs` → `GrepTool`, `GrepArgs`, `GrepOutputMode`, scanner and glob filters
- `./tests.rs` → `#[cfg(test)] mod tests` for the three tools
