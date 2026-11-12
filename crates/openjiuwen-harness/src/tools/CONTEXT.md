# tools

## Intent

The model-facing coding tool set of the `DeepAgent`, implemented on `openjiuwen_sysop` and the foundation `Tool`/`ToolCard` contract, with Python-compatible tool names, JSON schemas, payload shapes and message text.
Ports `openjiuwen/harness/tools/` (`shell/`, `filesystem.py`, `todo.py`, `ask_user.py`, `memory.py`, `coding_memory.py`) and the English tool descriptions from `openjiuwen/harness/prompts/tools/`.
Upper layers (agent, rails, protocol) only see `ToolRegistry`, `BuiltinTool` and the two seam traits (`UserAsker`, `MemoryToolBackend`).

## API Surface

### Registry — `mod.rs`

- `BuiltinTool` enum: `Bash`, `ReadFile`, `WriteFile`, `EditFile`, `Glob`, `ListFiles`, `Grep`, `TodoCreate`, `TodoList`, `TodoGet`, `TodoModify`, `AskUser`, `MemorySearch`, `MemoryGet`, `WriteMemory`, `ReadMemory`, `EditMemory`, `CodingMemoryRead`, `CodingMemoryWrite`, `CodingMemoryEdit`; `ALL`, `name()`, `from_name()`, `parallel_safe()`.
- `tool_names() -> &'static [&'static str]` — the model-facing names in registry order.
- `ToolRegistry` — `new(Workspace)` (builds a local `SysOperation` rooted at the workspace), `with_sys_operation(Arc<SysOperation>)`, `with_fs_operation`, `with_shell_operation`, `with_permissions(Arc<PermissionEngine>)`, `with_user_asker(Arc<dyn UserAsker>)`, `with_memory(Arc<dyn MemoryToolBackend>)`, `with_language(impl Into<String>)`, `with_todo_dir(impl Into<PathBuf>)`, `with_tools(impl IntoIterator<Item = BuiltinTool>)`, `build() -> Vec<Arc<dyn Tool>>`, `cards(language: &str) -> Vec<ToolCard>`, `card(BuiltinTool, language: &str) -> ToolCard`; accessors `env()`, `read_state()`, `sys_operation()`, `fs_operation()`, `shell_operation()`, `todo_dir()`, `tools()`.
- `build`, `cards` and the tool modules `shell`/`read`/`write`/`search`/`todo`/`ask_user`/`memory` are doc-comment-only skeletons: they return empty results until the tool implementations land.
- The registry never builds `task`/subagent tools (they live in `crate::subagents`).

### Shared helpers — `support.rs`

- `ToolPayload` — the Python `ToolOutput` envelope as JSON: `success`, `data`, `error`, `extracted_content`, `include_extracted_content_only_once`, `long_term_memory`; `ok(data)`, `ok_empty()`, `fail(message)`, `fail_with_data(message, data)`, `with_extracted_content(text)`, `to_json()` (always emits all six keys).
- `render_fields(&[(&str, Value)]) -> String` — Python `render_fields`: `key: text` pairs joined by `\n`, skipping `null` and `""`, non-strings as compact JSON.
- `truncate_chars(text, max_chars) -> Truncated { text, truncated, omitted_chars }` — `0` means unlimited, never splits a UTF-8 character, appends `TRUNCATION_MARKER`.
- `schema_for::<T: JsonSchema>() -> Value` — object JSON Schema with `$schema`/`title`/`definitions` stripped so it matches the Python `{"type","properties","required"}` shape.
- `tool_card(name, description, input_params, parallel_safe) -> ToolCard` — id is a fresh hex uuid, `exposure = Direct`, `execution = Server`, `idempotent = false`.
- `ToolEnv` — the workspace plus the optional permission gate and language: `new(Workspace)`, `with_permission`, `with_language`, `workspace()`, `language()`, `is_english()`, `deny_reason(tool, args) -> Option<String>` (only a `deny` decision blocks; `ask` is the interrupt rail's job), `resolve_path(tool, raw, args) -> Result<PathBuf, ToolPayload>`, `resolve_dir(...)`.
- `ModelArgs` — serde-deserialization of `invoke` inputs with Python-friendly errors: `parse::<T>(value) -> Result<T, ToolPayload>` where a non-object value yields `tool arguments must be an object`.

### Read-before-write state — `read_state.rs`

- `FileFingerprint { mtime_ns, size_bytes }` + `file_fingerprint(&Metadata)`.
- `ReadCoverage { start, end }` — a 1-based inclusive line window reported by `read_file`.
- `FileReadState { fingerprint, total_lines, read_ranges: Option<Vec<(usize, usize)>>, content: Option<String> }`.
- `ReadStateRegistry` (`new() -> Arc<Self>`): `record_read(path, fingerprint, coverage, content)`, `record_written(path, fingerprint, content)`, `write_gate(path, fingerprint, current) -> WriteGate`, `edit_gate(path, fingerprint, current) -> EditGate`, `lock(path) -> OwnedMutexGuard`, `invalidate(path)`, `clear()`.
- `WriteGate { Ok, NotRead, Partial { lines_read, total_lines, resume_offset }, Stale }`; `EditGate { Ok, NotRead, Stale }`.
- `merge_range`, `ranges_cover`, `first_unread_offset` are pure helpers with unit tests.

### Shell — `shell.rs`

- `BashTool` — card name `bash`, `parallel_safe = true`.
- Inputs (`BashArgs`): `command` (required), `timeout` (300, clamped `1..=3600`), `description`, `run_in_background`, `workdir`, `max_output_chars` (20000, minimum 200, `0` = unlimited), `shell_type` (`auto|cmd|powershell|bash|sh`, lenient parse).
- Foreground success `data = {content}` where content is `Command: {cmd}\nStdout: {stdout}\n\nStderr: {stderr}\nExit Code: {code}` with `(empty)` for an empty stream.
- Non-zero exit, timeout or backend failure → `success=false` with the same rendered content as `error` (partial output is never lost); background → `{pid, status: "started"}`.
- A missing explicit `workdir` → `workdir does not exist: {path}`; blank command → `command cannot be empty`.

### Filesystem — `read.rs`, `write.rs`, `search.rs`

- `ReadFileTool` (`read_file`): `file_path` (required), `offset` (0-based lines to skip), `limit` (max 2000, default 2000), `pages` and `caption` accepted and ignored.
  Output is line-numbered `{idx:>6}\t{line}`; `data = {content, file_path, line_count}`; 2000-line and 256 KiB defaults; a read registers coverage in the read-state registry.
- `WriteFileTool` (`write_file`, `parallel_safe = false`): `file_path`, `content`; read-before-write requires full coverage; `data = {file_path, bytes_written, type: "create"|"update", created, original_file}`; atomic write through `FsOperation::write_file` with `prepend_newline = false`.
- `EditFileTool` (`edit_file`, `parallel_safe = false`): `file_path`, `old_string`, `new_string`, `replace_all`; rejects identical strings and a missing/non-unique `old_string`; empty `old_string` creates a missing (or whitespace-only) file; every write carries `expected_content_sha256` (compare-and-swap); `data = {file_path, replacements}`.
- `GlobTool` (`glob`): `pattern` (required, `{a,b}` braces expanded), `path`; caps at 100 results; `data = {durationMs, numFiles, filenames, matching_files, truncated, count}`.
- `ListDirTool` (`list_files`): `path`, `show_hidden`; `data = {files, dirs}` — names only, sorted, one level.
- `GrepTool` (`grep`): `pattern` (required), `path`, `ignore_case`/`-i`, `glob`, `output_mode` (`content|files_with_matches|count`), `-B`/`-A`/`-C`/`context`, `-n`, `head_limit` (250, `0` = unlimited), `offset`, `type`, `multiline`; VCS directories (`.git`, `.svn`, `.hg`, `.bzr`, `.jj`, `.sl`) are always skipped; results are produced by an in-process scanner that emits ripgrep-shaped output.

### Todos — `todo.rs`

- `TodoStore` (`new(dir) -> Arc<Self>`, `for_workspace(&Workspace) -> Arc<Self>`, `path_for(session)`, `load`, `save`, `lock`) persisting a JSON array of `crate::schema::TodoItem` at `<todo dir>/<session_id>/todo.json` with per-session async locks.
- Tools `todo_create`, `todo_list`, `todo_get`, `todo_modify`; the session id comes from `ToolContext::session_id`, falling back to `default` when absent or unsafe.
- `todo_create` inputs `todos[{id, content, activeForm, description, status?, selected_model_id?}]`; the first item becomes `in_progress`, the rest `pending`.
- `todo_modify` inputs `action` (`update|delete|cancel|append|insert_after|insert_before`), `ids`, `todos`, `todo_data{target_id, items}`.
- Payloads: create/modify `{message}`, list `{todos[{id, content, status, depends_on}]}` (active items only), get `{todo}`; validation failures are `success=false` payloads carrying the Python message text.

### User interaction — `ask_user.rs`

- `UserAsker` trait: `async fn ask(&self, questions: Vec<AskQuestion>) -> Result<Value>`.
- `AskQuestion { header, question, options: Vec<AskOption>, multi_select }`, `AskOption { label, description, preview: Option<String> }` (all serde + `JsonSchema`).
- `AskUserTool` (`ask_user`): 1..=4 questions; returns `{}` on success; without a wired asker returns `success=false` with `ask_user is not available: no user interaction handler is configured`.

### Memory — `memory.rs`

- `MemoryScope { Memory, CodingMemory }`.
- `MemoryToolBackend` trait: `search(MemorySearchRequest) -> Result<Value>`, `read(scope, path, offset, limit) -> Result<MemoryReadData>`, `write(scope, path, content, append) -> Result<MemoryWriteData>`, `edit(scope, path, old_text, new_text) -> Result<MemoryEditData>`.
- Data types `MemorySearchRequest { query, max_results, min_score, session_key, scope }`, `MemoryReadData { path, text, total_lines, start_line, end_line }`, `MemoryWriteData { path, full_path, appended, file_existed }`, `MemoryEditData { path, replacements }`.
- `MemoryTool { kind: MemoryToolKind, backend: Option<Arc<dyn MemoryToolBackend>> }` renders eight cards: `memory_search`, `memory_get`, `write_memory`, `read_memory`, `edit_memory` (scope `Memory`) and `coding_memory_read`, `coding_memory_write`, `coding_memory_edit` (scope `CodingMemory`).
- No backend configured → `success=false` with `memory tool backend is not configured`.

### Descriptions — `descriptions.rs`

- `description(name, language) -> &'static str` and `SUPPORTED_LANGUAGES`; English text is verbatim from the Python `prompts/tools/` package, Chinese text is a faithful translation, unknown language falls back to `cn` (Python behavior).
- Descriptions are built here rather than read from `crate::prompts::tool_descriptions`, so this module has no dependency on the prompt builder.

## Constraints

- Every tool implements `openjiuwen_foundation::Tool`; `ToolCard::input_params` is a `schemars` schema of the same typed args struct.
- Expected failures return a `success=false` payload, never `Err`; `Err` is reserved for protocol violations (unknown tool, malformed input type) and cancellation (`Error::Cancelled` checked via `ToolContext::check_cancelled`).
- Model-supplied paths always go through `ToolEnv::resolve_path` → `crate::workspace::Workspace::resolve`; absolute paths inside the workspace are accepted, anything outside is refused. No raw `Path::new(model_input)`.
- Mutations go through `FsOperation` (atomic rename, compare-and-swap); reads of resolved paths may use `tokio::fs`.
- No panics, no `unwrap`/`expect` outside tests, no `unsafe`; `openjiuwen_common::error::{Error, Result}` only.
- No dependency on `crate::prompts`, `crate::subagents`, `crate::agent`, `crate::rails`, `crate::task_loop`; upper layers depend on this module, never the reverse.
- `tracing` only, no stdout/stderr writes.
- Files ≤1000 lines; shared helpers live in `support.rs`, never duplicated per tool.
- Process state: the read-state registry and todo store are per-`ToolRegistry` instances (no process globals) so tests are isolated; the todo file location is the only shared on-disk state.

## Design decisions

- **Deviation — tool-layer permission**: Python confines paths only through the rail (`FileGuard`); here `ToolEnv` resolves against the workspace and an optional `PermissionEngine` short-circuits a `deny` decision into a `success=false` payload. An `ask` decision is *not* enforced inside a tool because only the interrupt rail can ask a human.
- **Deviation — read-state scope**: Python keeps a module-global `_FILE_READ_REGISTRY`; here it is an `Arc<ReadStateRegistry>` owned by the registry (testability, no global mutable state). A successful write records the written content, so writing the same file twice in a row is allowed when the content is unchanged; Python rejects a second write until the file is re-read.
- **Deviation — payload envelope**: Python returns a `ToolOutput` object (filesystem/memory) or raises (todo) or returns `{}` (ask_user). Everything here returns the six-key envelope, so todo validation errors become `success=false` payloads with the Python message text, and `ask_user` returns `{}` on success.
- **Deviation — grep implementation**: results come from an in-process walker (`regex` + `walkdir`) instead of shelling out to `rg`/`grep`, giving deterministic offline behavior and no shell-quoting surface. Output shapes (content lines `path:line:text`, `-` separators for context lines, `path`, `path:count`) match ripgrep. `type` and `multiline` report the Python "requires ripgrep" failure because no rg binary is used.
- **Deviation — bash truncation**: Python persists oversized output to a temp file and shows a preview; here `max_output_chars` truncates in place with `TRUNCATION_MARKER` and reports `truncated` in `data`.
- **Deviation — bash exit codes**: a non-zero exit is reported as `success=false`; Python's `interpret_exit_code` neutrality for `grep`/`diff`/`test`/`find` is not ported.
- **Deviation — todo path**: `<todo node>/<session_id>/todo.json` (the workspace `todo` node, which is the Python `TaskPlanningRail` wiring), overridable through `ToolRegistry::with_todo_dir`; the session id falls back to `default` instead of raising `Session ID is required`.
- **Deviation — descriptions**: each tool carries the Python English description text verbatim (Chinese translated) inside this module; `crate::prompts` may re-export them but never owns them.
- **Deviation — edit_file fidelity**: LF/CRLF round-trip, quote-style and read-state rules are ported; `html.unescape`, the `_DESANITIZATIONS` table, `.ipynb` protection and operation-history files are not.
- Never resolved here (owned elsewhere): skill/subagent/web/LSP/MCP tools, `task`/`subagent_*`, tool discovery, mode switching, cron, goal, compression recall, worktree and multimodal tools.

## Deferred Python behavior

- `read_file`: PDF/Office/notebook/image readers, the token budget estimate, `multimodal` data, device-file refusal and the strict-decode retry; only text files are read.
- `write_file`: the 5 MiB content / 1 GiB file limits, conditional-write registration by the agent, and the `.agent_history` operation log.
- `bash`: `OPENJIUWEN_BASH_STRICT` injection detection, destructive-command warnings, rm-target history, sudo rewriting, `PowerShellTool`, and persisted oversized output.
- `grep`: ripgrep binary invocation, `--type` filters, multiline search, Windows `Select-String`, and `.gitignore` awareness.
- `todo`: `TodoLockManager` per-session process registry (replaced by `TodoStore` locks) and the Python code-numbered `build_error` envelope.
- `memory`: SQLite/embedding hybrid retrieval lives in `openjiuwen_memory` behind `MemoryToolBackend`; `coding_memory` front-matter validation is the backend's job.
- `ask_user`: the `AskUserRail` interrupt wiring lives in `crate::rails`.

## Routing Table

- `./mod.rs` → `ToolRegistry`, `BuiltinTool`, `tool_names`, module declarations
- `./support.rs` → `ToolPayload`, `render_fields`, truncation, `schema_for`, `tool_card`, `ToolEnv` (path resolution + permission gate), arg parsing
- `./read_state.rs` → read-before-write registry, fingerprints, range coverage, per-path locks
- `./descriptions.rs` → model-facing description text (en/cn) and language fallback
- `./shell.rs` → `BashTool` and the `bash` card
- `./read.rs` → `ReadFileTool`
- `./write.rs` → `WriteFileTool`, `EditFileTool`
- `./search.rs` → `GlobTool`, `ListDirTool`, `GrepTool`
- `./todo.rs` → `TodoStore` and the four todo tools
- `./ask_user.rs` → `UserAsker` seam and `AskUserTool`
- `./memory.rs` → `MemoryToolBackend` seam and the eight memory tools
- `./tests.rs` → cross-module parity tests (names, cards, registry, render strings)
- `./tests/` → offline integration suite driving the registry over a temp workspace (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests live beside each module: `support` (truncation, render_fields, schema normalization, path confinement, deny gate), `read_state` (coverage merges, partial/stale gates), `shell` (clamping, rendering, timeout, background, blank command), `read` (line numbering, offset/limit, 256 KiB limit, empty file), `write` (read-before-write, create/update, edit uniqueness, `replace_all`, identical strings, CAS conflict), `search` (glob cap 100, brace patterns, list hidden filter, grep modes/context/head_limit/VCS skip), `todo` (create→list→get→modify round-trip, first item in_progress, validation messages, session fallback), `ask_user` (no asker), `memory` (no backend, payload shapes for each of the eight tools).
- `tests.rs` asserts the exact model-facing names, per-tool card key sets (`properties`/`required`), `parallel_safe` flags, and `render_for_llm` strings.
- All tests are offline and deterministic: `tempfile` workspaces, fake `UserAsker`/`MemoryToolBackend` implementations, no network, no reliance on `rg`/`git` binaries.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
