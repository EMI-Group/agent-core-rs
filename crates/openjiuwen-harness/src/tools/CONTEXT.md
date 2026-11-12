# tools

## Intent

The model-facing coding tool set, implemented on `openjiuwen_sysop` and the foundation `Tool` contract, with Python-compatible tool names and JSON schemas.
Ports `openjiuwen/harness/tools/` (`shell/`, `filesystem.py`, `todo.py`, `ask_user.py`, `memory.py`) and the card/description half of `prompts/tools/`.

## API Surface

- `mod.rs`: `BuiltinTool` enum listing implemented tools; `ToolRegistry` (`new(workspace, sysop)`, `with_permissions`, `with_memory`, `build() -> Vec<Arc<dyn Tool>>`, `cards(language) -> Vec<ToolCard>`); `tool_names()`.
- `shell.rs`: `BashTool` — name `bash`; inputs `command` (required), `timeout` (300; clamped 1..=3600), `description`, `run_in_background`, `workdir`, `max_output_chars` (20000, min 200, 0 = unlimited), `shell_type` (`auto|cmd|powershell|bash|sh`); output `{pid,status}` for background or `{stdout, stderr, exit_code, content}`; timeout keeps partial output with `success=false`.
- `filesystem.rs`: `ReadFileTool` (`read_file`: `file_path` required, `offset`, `limit`), `WriteFileTool` (`write_file`: `file_path`, `content`; read-before-write enforcement), `EditFileTool` (`edit_file`: `file_path`, `old_string`, `new_string`, `replace_all`), `GlobTool` (`glob`: `pattern`, `path`; max 100 results), `ListDirTool` (`list_files`: `path`, `show_hidden`), `GrepTool` (`grep`: `pattern`, `path`, `ignore_case`, `glob`, `output_mode` content/files_with_matches/count, context flags, `head_limit`).
- `todo.rs`: `todo_create`, `todo_list`, `todo_get`, `todo_modify` backed by `<workspace>/<session_id>/todo.json`; first created item becomes `in_progress`; invalid requests return `success=false` with Python message text.
- `ask_user.rs`: `AskUserTool` (name `ask_user`): `questions[1..=4]{header, question, options[{label, description, preview}]}`, `multi_select`; raises an interaction request through the session and returns `{}` on answer.
- `memory.rs`: `memory_search`, `memory_get`, `write_memory`, `read_memory` over `openjiuwen_memory`.
- `support.rs`: shared path resolution via `Workspace`/permission, output truncation, `tool_output(...)` JSON helper (`{success, data, error, extracted_content}`), `render_fields`.

## Constraints

- Every tool implements `openjiuwen_foundation::Tool`; `ToolCard.input_params` is a JSON Schema generated from the same typed params (`schemars`), and `render_for_llm` output matches Python strings.
- Expected failures return a `success=false` payload (never `Err`); protocol violations (unknown tool/unsupported input type) return `Error::tool`; cancellation returns `Error::Cancelled`.
- Paths are always resolved through the workspace and the permission engine; no raw `Path::new(model_input)`.
- File writes are atomic (temp file + rename) and honour the read-before-write rule and version conflicts.
- No stdout logging; use `tracing`.
- No dependency on `crate::subagents` or `crate::agent`; the task/subagent tools live in `crate::subagents`.

## Routing Table

- `./mod.rs` → `ToolRegistry`, `BuiltinTool`, re-exports and tool-name list
- `./shell.rs` → `bash`
- `./filesystem.rs` → read/write/edit/glob/list/grep
- `./todo.rs` → todo tools and the shared todo store
- `./ask_user.rs` → `ask_user`
- `./memory.rs` → memory tools
- `./support.rs` → shared helpers (path resolution, truncation, output rendering)

## Test Strategy

- Unit tests: exact tool names and schema key sets, read-before-write rejection, edit `old_string` uniqueness/`replace_all`, glob truncation at 100, grep output modes, bash timeout/exit-code semantics, todo store round-trip, ask_user returns `{}`, memory tool error payloads, `render_for_llm` strings.
- Parity references: `tests/unit_tests/harness/tools/test_filesystem_tools.py`, `test_shell_tools.py`, `test_todo.py`, `test_tool_render_for_llm.py`, `test_memory.py`.
