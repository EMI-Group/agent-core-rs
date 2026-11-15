# tools

## Intent

The model-facing coding tool set of the `DeepAgent`, implemented on `openjiuwen_sysop` and the foundation `Tool`/`ToolCard` contract, with Python-compatible tool names, JSON schemas, payload shapes and message text.
Ports `openjiuwen/harness/tools/` (`shell/`, `filesystem.py`, `todo.py`, `ask_user.py`, `memory.py`, `coding_memory.py`) and the English tool descriptions from `openjiuwen/harness/prompts/tools/`.
The language-server navigation (`lsp`), git-worktree (`worktree`) and the worktree-session pair (`enter_worktree`/`exit_worktree`) tools are owned by the sibling `crate::lsp` and `crate::worktree` modules and are mounted, configured and described here as extended tools.
Upper layers (agent, rails, protocol) only see `ToolRegistry`, `BuiltinTool`, the `ToolIndex` and the seam traits (`UserAsker`, `MemoryToolBackend`, `HttpFetcher`, `SearchBackend`, `McpResourceProvider`, `McpResourceSubscriptionProvider`, `WorktreeSessionStore`).

## API Surface

### Registry — `mod.rs` + `registry.rs`

- `BuiltinTool` enum: the core twenty `Bash`, `ReadFile`, `WriteFile`, `EditFile`, `Glob`, `ListFiles`, `Grep`, `TodoCreate`, `TodoList`, `TodoGet`, `TodoModify`, `AskUser`, `MemorySearch`, `MemoryGet`, `WriteMemory`, `ReadMemory`, `EditMemory`, `CodingMemoryRead`, `CodingMemoryWrite`, `CodingMemoryEdit`, then the fifteen extended `FetchWebpage`, `WebSearch`, `ImageMetadata`, `AudioMetadata`, `Cron`, `ListMcpResources`, `ReadMcpResource`, `McpResourceSubscription`, `ToolSearch`, `GetCurrentGoal`, `SubmitGoalReport`, `Lsp`, `Worktree`, `EnterWorktree`, `ExitWorktree`; `DEFAULT: [Self; 20]`, `EXTENDED: [Self; 35]`, `ALL: [Self; 35]` (`= EXTENDED`), `name()`, `from_name()`, `parallel_safe()`.
- `tool_names() -> &'static [&'static str]` — the core twenty names in registry order (what `ToolRegistry::new` mounts); `extended_tool_names() -> &'static [&'static str]` — all thirty-five names in `EXTENDED` order.
- `parallel_safe()` is false for `WriteFile`, `EditFile`, `Cron`, `SubmitGoalReport`, `McpResourceSubscription`, `Worktree`, `EnterWorktree` and `ExitWorktree` (they mutate shared on-disk or session state) and true for every other variant; each tool's card reports the same flag.
- `ToolRegistry` — `new(Workspace)` (builds a local `SysOperation` rooted at the workspace and mounts `BuiltinTool::DEFAULT`), `with_sys_operation(Arc<SysOperation>)`, `with_fs_operation`, `with_shell_operation`, `with_permissions(Arc<PermissionEngine>)`, `with_user_asker(Arc<dyn UserAsker>)`, `with_memory(Arc<dyn MemoryToolBackend>)`, `with_http_fetcher(Arc<dyn HttpFetcher>)`, `with_search_backend(Arc<dyn SearchBackend>)`, `with_mcp_resources(Arc<dyn McpResourceProvider>)`, `with_mcp_subscriptions(Arc<McpResourceSubscriptions>)`, `with_cron_scheduler(Arc<CronScheduler>)`, `with_goal_store(Arc<GoalStore>)`, `with_lsp_config(impl Into<Option<LspToolConfig>>)` (a bare `LspToolConfig` or an `Option`), `with_worktree_manager(Arc<WorktreeManager>)`, `with_worktree_root(impl Into<Option<PathBuf>>)`, `with_worktree_session_store(Arc<dyn WorktreeSessionStore>)`, `with_language(impl Into<String>)`, `with_todo_dir(impl Into<PathBuf>)`, `with_tools(impl IntoIterator<Item = BuiltinTool>)`, `with_deferred_tools(impl IntoIterator<Item = BuiltinTool>)`, `build() -> Vec<Arc<dyn Tool>>`, `cards(language: &str) -> Vec<ToolCard>`, `card(BuiltinTool, language: &str) -> ToolCard`; discovery API `discover(query, limit) -> Vec<ToolSearchHit>`, `reveal(names) -> Vec<ToolCard>`, `deferred_cards(language)`, `deferred_tools()`; accessors `env()`, `read_state()`, `sys_operation()`, `fs_operation()`, `shell_operation()`, `user_asker()`, `memory_backend()`, `http_fetcher()`, `search_backend()`, `mcp_provider()`, `mcp_subscriptions()`, `lsp_config()`, `worktree_manager()`, `worktree_root()`, `worktree_session_store()`, `todo_dir()`, `tools()`.
- Defaults: the `fetch_webpage` fetcher is a `ReqwestFetcher` so the tool works out of the box; `web_search` has no default backend (it reports the typed `not_configured_reason()` until `with_search_backend` is wired, `HttpSearchBackend::from_env()` reads `OPENJIUWEN_WEB_SEARCH_ENDPOINT`, and `WebSearchTool::with_endpoint(..)` opens the same gap in one call); the cron scheduler and goal store default to their workspace-backed instances, created once per `build()` call; MCP resources have no default provider.
- `lsp` is always mounted when selected: with no `with_lsp_config` it is built from `LspToolConfig::default()` and every call answers `success=false` with the LSP module's "no LSP server configured" message, naming the missing `server` setting instead of the tool disappearing; a configured server descriptor is used as given (a command that cannot be spawned is the same kind of typed payload). The registry fills the config workspace root from the workspace when unset and always renders the built tool in the registry language; `cards(language)` always uses the requested language.
- The three worktree tools share one `WorktreeManager` — the wired one, else one rooted at `with_worktree_root` or the workspace root — and forward the registry's `PermissionEngine`; nothing is checked at construction time, so a non-repository workspace reports the manager's typed git failure as a `success=false` payload.
- `enter_worktree` and `exit_worktree` also share one `WorktreeSessionStore` per registry, `with_worktree_session_store` when the host keeps the session elsewhere and a `MemoryWorktreeSessionStore` by default; they never change the process cwd and report `worktree_path`/`original_cwd` in the payload instead.
- Deferred tools are held by `with_deferred_tools` but NOT mounted by `build()` and NOT listed by `cards()`: the model finds them through the `tool_search` tool (or `discover`) and the host unions `reveal(names)` into the next model call's tool list. The mounted set always wins, so a deferred tool can never shadow an active one.
- With nothing deferred (`deferred` empty, the default) `discovery_set()` is `BuiltinTool::EXTENDED`, so `discover`/`reveal`/`deferred_cards` cover all thirty-five extended tools even though only twenty are mounted; a configured deferred list narrows discovery to that list.
- `build()` yields one `Arc<dyn Tool>` per selected variant in `self.tools` order; the four todo tools share one `Arc<TodoStore>` created lazily once per call (`todo_dir` when set, else `TodoStore::for_workspace`), and the filesystem tools share the registry's `ReadStateRegistry`.
- `cards(language)` maps `self.tools` through `card(tool, language)`, which returns the owning module's real `card_for`, never a placeholder.
- Card schemas are Python-shaped: `schemars` output with `$schema`/`title`/`$defs` stripped, every `$ref` inlined (`todo` and `ask_user` post-process theirs), and `required` narrowed to Python's list; no card contains a dangling reference.
- The public API of every tool module is re-exported here (`AskUserTool`/`UserAsker`, `MemoryTool`/`MemoryToolBackend`/`MemoryToolKind`, `BashTool`, the read/write/edit/glob/list/grep tools and their arg structs, the todo store/tools); `support` and `read_state` helpers are re-exported as well, so upper layers import from `crate::tools` only.
- The registry never builds `task`/subagent tools (they live in `crate::subagents`).

### Shared helpers — `support.rs`

- `ToolPayload` — the Python `ToolOutput` envelope as JSON: `success`, `data`, `error`, `extracted_content`, `include_extracted_content_only_once`, `long_term_memory`; `ok(data)`, `ok_empty()`, `fail(message)`, `fail_with_data(message, data)`, `with_extracted_content(text)`, `to_json()` (always emits all six keys).
- `render_fields(&[(&str, Value)]) -> String` — Python `render_fields`: `key: text` pairs joined by `\n`, skipping `null` and `""`, non-strings as compact JSON.
- `truncate_chars(text, max_chars) -> Truncated { text, truncated, omitted_chars }` — `0` means unlimited, never splits a UTF-8 character, appends `TRUNCATION_MARKER`.
- `schema_for::<T: JsonSchema>() -> Value` — object JSON Schema with `$schema`/`title`/`definitions` stripped so it matches the Python `{"type","properties","required"}` shape.
- `tool_card(name, description, input_params, parallel_safe) -> ToolCard` — id is a fresh hex uuid, `exposure = Direct`, `execution = Server`, `idempotent = false`.
- `ToolEnv` — the workspace plus the optional permission gate and language: `new(Workspace)`, `with_permission`, `with_language`, `workspace()`, `language()`, `permission() -> Option<&Arc<PermissionEngine>>`, `is_english()`, `deny_reason(tool, args) -> Option<String>` (only a `deny` decision blocks; `ask` is the interrupt rail's job), `resolve_path(tool, raw, args) -> Result<PathBuf, ToolPayload>`, `resolve_dir(...)`.
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

### Web — `web/`

- `HttpFetcher` trait (`async fn fetch(&self, FetchRequest) -> Result<FetchResponse>`), `FetchRequest { url, timeout, max_bytes, headers }`, `FetchResponse { status, final_url, headers, body }` with `header()`/`content_type()`; `ReqwestFetcher` (redirect cap 10, byte-capped streaming read, no gzip/deflate, Chrome UA) is the default.
- `FetchWebpageTool` (`fetch_webpage`) inputs `url` (required), `max_chars`, `max_bytes`, `timeout_seconds`, `prompt`, `caption`; consults the permission net guard before the request; detects the charset from `Content-Type` then `<meta charset>`; renders HTML through `html::extract`, passes JSON/text through; payload `{url, final_url, status, content_type, charset, content, truncated, title?, prompt?, caption?}`.
- `SearchBackend` trait (`search(&self, query, max_results)`), `SearchResult { title, url, snippet }`, `HttpSearchBackend` (configurable endpoint, injectable fetcher), `FakeSearchBackend` (documented test double with canned results, failure injection and call recording), `WebSearchTool` (`web_search`) with `{success, results:[{title,url,snippet}]}`.
- `web/html.rs` — `extract(&str) -> ExtractedHtml { title, text }`, `normalize_url(base, href)`, `decode_entities`, `collapse_whitespace`; `web/decode.rs` — `detect_charset`, `charset_from_content_type`, `sniff_meta_charset`, `decode_body` (UTF-8/ASCII, Latin-1/CP1252, UTF-16, BOM precedence, lossy fallback).

### Media — `media/`

- `ImageMetadataTool` (`image_metadata`) and `AudioMetadataTool` (`audio_metadata`), input `{image_path}` / `{audio_path}` (aliases `path`, `file_path`, `image_path_or_url` / `audio_path_or_url`), resolved through `ToolEnv::resolve_path` and read with `tokio::fs::read`.
- Pure header parsing in `parse.rs` (bounds-checked `Cursor`, endian readers, `fourcc`, `Failure`, `MediaKind`) feeding `image/{png,jpeg,gif,webp,bmp}.rs` and `audio/{wav,flac,mp3,ogg}.rs`; `parse_image_header(&[u8])` / `parse_audio_header(&[u8]) -> ParseResult<…>` are public.
- Payload `{kind, format, byte_size, …}`: PNG width/height/bit depth/color type/alpha/interlace/frames, JPEG dimensions/components/precision/progressive/EXIF orientation, GIF version/frames/animated/alpha, WebP variant/alpha/animated, BMP bit count/compression/top-down; WAV sample rate/channels/bits/format tag/duration, FLAC sample rate/channels/bits/total samples/duration, MP3 version/layer/bitrate/sample rate/channels/duration, OGG codec/sample rate/channels/duration.

### Cron — `cron/`

- `CronSchedule::parse(&str)` and `next_after(DateTime<Utc>) -> Option<DateTime<Utc>>` (strictly after, bounded horizon); the accepted expression grammar is documented in `cron/CONTEXT.md`.
- `CronScheduler::new(dir)`, `for_workspace(&Workspace)` (`<cron node>/jobs.json`), `create`, `create_with_description`, `list`, `cancel`, `due(now)` (returns the fired jobs and advances/persists their `next_run_at`); `CronJob { id, cron, next_run_at, created_at, enabled, prompt, description }`.
- `CronTool` (`cron`) dispatching its actions to the shared scheduler, with `invoke_at(inputs, now)` as the injected-clock entry point; the action surface is documented in `cron/CONTEXT.md`.

### MCP resources — `mcp_resources.rs`

- `McpResourceProvider` trait (`list_resources`, `read_resource`), `McpResource { uri, name, mime_type, description }`, `McpResourceContent { uri, mime_type, text, blob_base64, byte_size }`, `FakeMcpResourceProvider` (documented test double).
- `ListMcpResourcesTool` (`list_mcp_resources`) and `ReadMcpResourceTool` (`read_mcp_resource`, requires `uri`); payloads wrap the Python bare list/object in `{resources: [...]}` / `{resource: {...}}`; no provider wired yields a typed `success=false` reason.

### MCP resource subscriptions — `mcp_subscriptions.rs` + `mcp_subscription_tool.rs`

- `McpResourceSubscriptionProvider` trait: `subscribe(uri, &McpResourceSink)`, `unsubscribe(uri)`, `supports_subscriptions()`; both methods default to `Error::Unsupported` with `SUBSCRIBE_UNSUPPORTED`/`UNSUBSCRIBE_UNSUPPORTED`, so a list/read-only provider still coerces into the trait.
- `McpResourceSubscriptions` (`new`, `with_provider`, `with_capacity`, `with_max_subscriptions`, `subscribe`, `unsubscribe`, `reap`, `list`, `info`, `poll`, `publish`, `sink`, `events`, `recv`, `is_subscribed`, `capacity`, `max_subscriptions`, `provider`): one bounded ring buffer plus one broadcast channel per URI, sized by `DEFAULT_BUFFER_CAPACITY` (32, clamped 1..=1024); the registry tracks at most `DEFAULT_MAX_SUBSCRIPTIONS` (64) subscriptions and refuses one more with `Error::Validation` (a failed open registers nothing, and `unsubscribe`/`reap` free a slot); `McpSubscriptionInfo { uri, capacity, buffered, received, dropped, lagged, last_sequence }`, `McpResourceEvent { uri, sequence, reason, content }`, `McpResourceSink` (`uri`, `updated`, `notified`, `info`), `NO_PROVIDER_REASON`.
- `recv` never makes concurrent callers queue: the caller that acquires the shared receiver awaits it (keeping the `lagged` accounting) and every other concurrent caller receives from its own `broadcast::Receiver` (the `events` view), so two `recv` calls on one URI both observe the events published while they wait; per-call receivers never feed `lagged`.
- `unsubscribe` reaps the local entry when the provider answers `Error::NotFound` (the server already closed the subscription) and returns the stale snapshot instead of pinning the ring buffer; `reap(uri)` drops one entry without calling the provider, for a host that saw the server close it.
- `FakeMcpResourceSubscriptionProvider` — documented test double (`new`, `unsupported`, `subscribe_calls`, `unsubscribe_calls`, `is_subscribed`, `fail_subscribe`, `fail_unsubscribe`, `drop_subscription`, `push_content`, `push_notification`).
- `McpResourceSubscriptionTool` (`mcp_resource_subscription`, Rust-only extension — Python has no subscription surface): `action` = `subscribe|unsubscribe|list|poll`, plus `uri` and `limit` (`poll` default 10, clamped 1..=50); payloads `{subscription, present, message}` for subscribe, `{subscription, removed, present, message}` (success) / `{subscription, removed: false, present, reason}` (failure, `success=false`) for unsubscribe, `{subscriptions, count, capacity, max_subscriptions}` for list and `{uri, events, count, subscription, present, reason}` for poll; a snapshot that a raced lookup or a reaped entry cannot produce is never a bare `null` — `subscription: null` always comes with `present: false` and the `reason`; no registry wired yields the typed `no subscription registry is configured (ToolRegistry::with_mcp_subscriptions)` payload, a provider without `resources/subscribe` yields `NO_PROVIDER_REASON`, and a subscribe past the registry cap yields the cap's validation message.

### Worktree enter/exit — `crate::worktree`

- `EnterWorktreeTool` (`enter_worktree`, inputs `{name?}`) creates or recovers the worktree through the shared `WorktreeManager`, records a `WorktreeSessionState` in the `WorktreeSessionStore` and reports `{worktree_path, worktree_branch, message, existed, sparse}`; `ExitWorktreeTool` (`exit_worktree`, inputs `{action: keep|remove, discard_changes?}`) clears the session and reports `{action, original_cwd, worktree_path, worktree_branch, worktree_name, message, discarded_files?, discarded_commits?}`.
- Both tools consult the permission engine through the worktree module's `write_file` deny gate, refuse an already active session, and return `success=false` payloads for every expected failure; neither changes the process cwd.

### Discovery — `discovery.rs`

- `ToolIndex::from_cards(&[ToolCard])` with an in-crate BM25 scorer (`k1 = 1.2`, `b = 0.75`), `search(query, limit) -> Vec<ToolSearchHit { name, score, description }>` sorted by descending score then name.
- `ToolSearchTool` (`tool_search`) inputs `{query, limit}` (default 5, clamped 1..=20) and returns `{query, results, count}`.

### Goal — `goal/`

- `GoalStore`, `GoalRecord`, `GoalStatus`, `GoalAssessmentStatus`, `GoalAssessment`, `TokenUsage` (see `goal/CONTEXT.md`), `GetCurrentGoalTool` (`get_current_goal`) and `SubmitGoalReportTool` (`submit_goal_report`), plus `GoalRail`/`GoalRailConfig`/`GOAL_EVENT_EXTRA_KEY`.

### Descriptions — `descriptions.rs`

- `description(name, language) -> &'static str` and `SUPPORTED_LANGUAGES`; English text is verbatim from the Python `prompts/tools/` package, Chinese text is a faithful translation, unknown language falls back to `cn` (Python behavior).
- Descriptions are built here rather than read from `crate::prompts::tool_descriptions`, so this module has no dependency on the prompt builder.

## Constraints

- Every tool implements `openjiuwen_foundation::Tool`; `ToolCard::input_params` is a `schemars` schema of the same typed args struct.
- Expected failures return a `success=false` payload, never `Err`; `Err` is reserved for protocol violations (unknown tool, malformed input type) and cancellation (`Error::Cancelled` checked via `ToolContext::check_cancelled`).
- Model-supplied paths always go through `ToolEnv::resolve_path` → `crate::workspace::Workspace::resolve`; absolute paths inside the workspace are accepted, anything outside is refused. No raw `Path::new(model_input)`.
- Mutations go through `FsOperation` (atomic rename, compare-and-swap); reads of resolved paths may use `tokio::fs`.
- No panics, no `unwrap`/`expect` outside tests, no `unsafe`; `openjiuwen_common::error::{Error, Result}` only.
- No dependency on `crate::prompts`, `crate::subagents`, `crate::agent`, `crate::rails`, `crate::task_loop`; upper layers depend on this module, never the reverse. One documented exception: `goal::GoalRail` implements `crate::rails::HarnessRail` and builds `crate::prompts::PromptSection` values, because it is owned by this node; it must only use those modules' public APIs.
- `tracing` only, no stdout/stderr writes.
- Files ≤1000 lines; shared helpers live in `support.rs`, never duplicated per tool.
- Process state: the read-state registry, todo store, cron scheduler, goal store, discovery index, MCP subscription registry and worktree session store are per-`ToolRegistry` instances (no process globals) so tests are isolated; the todo file, cron job file and goal file are the only shared on-disk state (plus the worktree index when a worktree tool runs).
- Every new tool was added with `BuiltinTool::DEFAULT` untouched: `ToolRegistry::new` still mounts exactly the core twenty, so a host opts into the extended surface with `with_tools(BuiltinTool::EXTENDED)` or per tool.

## Design decisions

- **Deviation — tool-layer permission**: Python confines paths only through the rail (`FileGuard`); here `ToolEnv` resolves against the workspace and an optional `PermissionEngine` short-circuits a `deny` decision into a `success=false` payload. An `ask` decision is *not* enforced inside a tool because only the interrupt rail can ask a human.
- **Deviation — read-state scope**: Python keeps a module-global `_FILE_READ_REGISTRY`; here it is an `Arc<ReadStateRegistry>` owned by the registry (testability, no global mutable state). A successful write records the written content, so writing the same file twice in a row is allowed when the content is unchanged; Python rejects a second write until the file is re-read.
- **Deviation — payload envelope**: Python returns a `ToolOutput` object (filesystem/memory) or raises (todo) or returns `{}` (ask_user). Everything here returns the six-key envelope, so todo validation errors become `success=false` payloads with the Python message text, and `ask_user` returns `{}` on success.
- **Deviation — grep implementation**: results come from an in-process walker (`regex` + `walkdir`) instead of shelling out to `rg`/`grep`, giving deterministic offline behavior and no shell-quoting surface. Output shapes (content lines `path:line:text`, `-` separators for context lines, `path`, `path:count`) match ripgrep. `type` and `multiline` report the Python "requires ripgrep" failure because no rg binary is used.
- **Deviation — bash truncation**: Python persists oversized output to a temp file and shows a preview; here `max_output_chars` truncates in place with `TRUNCATION_MARKER` and reports `truncated` in `data`.
- **Deviation — bash exit codes**: a non-zero exit is reported as `success=false`; Python's `interpret_exit_code` neutrality for `grep`/`diff`/`test`/`find` is not ported.
- **Deviation — todo path**: `<todo node>/<session_id>/todo.json` (the workspace `todo` node, which is the Python `TaskPlanningRail` wiring), overridable through `ToolRegistry::with_todo_dir`; the session id falls back to `default` instead of raising `Session ID is required`.
- **Deviation — web**: no JS rendering, no Jina/vendor reader fallback, no paid search SDKs (Jina/Bocha/Serper/Perplexity env-key clients) and no LLM quality/assessment/budget pass; fetch and search both sit behind async seams with a `ReqwestFetcher` default and a documented `FakeSearchBackend` test double.
- **Deviation — media**: metadata only, pure header parsing with no decoding crates; no OCR, transcription or visual/audio question answering (those need a model), and no URL inputs — a `*_path_or_url` value is resolved as a workspace path.
- **Deviation — cron**: no background daemon — the host polls `due(now)`; the expression grammar and the action surface are documented in `cron/CONTEXT.md`.
- **Deviation — MCP resources**: the tools work through the narrow `McpResourceProvider` seam, which `McpClientResourceProvider` adapts over any `openjiuwen_foundation::McpClient`; Python's `server_id` argument is tolerated but ignored.
- **Deviation — discovery**: in-crate BM25 (`k1 = 1.2`) with an exact-name boost folded into the score instead of Python's separate boolean sort key; `tool_call` (relaying a call to a discovered tool) is not ported because `reveal` hands the card to the host instead.
- **Deviation — goal**: no `GoalManager`/`GoalReportSink`; the tools write the store directly and the loop-side role lives in `goal::GoalRail`, which queues `goal.updated` events under `harness.goal_events` for the host to drain (rails have no event sink).
- **Deviation — descriptions**: each tool carries the Python English description text verbatim (Chinese translated) in `descriptions.rs`; the table covers the core twenty plus the fifteen extended names, and `description()` still returns `""` for an unknown name.
- **Deviation — lsp/worktree mounting**: Python mounts `lsp_tool/` and `worktree/` from the same `tools/` package; here they live in the sibling `crate::lsp` and `crate::worktree` modules with their own `Tool` implementations, and `registry.rs` only constructs them (config, manager, session store, language, permission engine) and renders their cards. None of them is mounted by `BuiltinTool::DEFAULT`; a host opts in with `with_tools(BuiltinTool::EXTENDED)` or by naming the variant.
- **Deviation — edit_file fidelity**: LF/CRLF round-trip, quote-style and read-state rules are ported; `html.unescape`, the `_DESANITIZATIONS` table, `.ipynb` protection and operation-history files are not.
- Still owned elsewhere: skill/subagent tools, `task`/`subagent_*`, mode switching, compression recall, the worktree rails integration and the cross-process worktree lock (the three worktree tools are mounted here), browser/CUA/mobile GUI tools and the model-backed OCR/transcription/QA tools.
- Deferred within the ported modules: `tool_call` (relay a call to a discovered tool).

## Notes for Agents

- Upper layers (`agent`, `rails`, `protocol`, `cli`) construct tools only through `ToolRegistry::build`; live seams are wired with `with_user_asker`, `with_memory`, `with_permissions` and the sys-operation setters.
- A registry's `ReadStateRegistry` is per-instance, so build the tool set once per agent (or per session) rather than per call, otherwise the read-before-write gate resets.
- `ask_user` reports `ask_user is not available: no user interaction handler is configured` until an asker is wired; the eight memory tools report `memory tool backend is not configured` until a backend is wired.
- The todo store defaults to `<workspace>/todo/<session>/todo.json`; the session id comes from `ToolContext::session_id` and falls back to `default`, so a session-less caller shares one list.
- `grep` reports `success=true` with empty output on no matches (exit-code 1 lives in `data`), which is what makes it render `No matches found.`.
- The extended tools are opt-in: `ToolRegistry::new(..).with_tools(BuiltinTool::EXTENDED)` mounts all thirty-five; wire `with_search_backend`, `with_mcp_resources`, `with_mcp_subscriptions`, `with_http_fetcher`, `with_goal_store`/`with_cron_scheduler`, `with_lsp_config` and `with_worktree_manager`/`with_worktree_root`/`with_worktree_session_store` only when a live seam is wanted.
- `lsp` needs an `LspToolConfig` with a `LspServerConfig` (or an injected transport through `crate::lsp::LspTool::with_transport`) before it can answer; the three worktree tools need a git repository at the configured root.
- `enter_worktree` and `exit_worktree` share the registry's `WorktreeSessionStore`, so a second registry (or a rebuilt tool set) only sees the session when the host wires the same store.
- `goal::GoalRail` is the one module here that implements a `crate::rails` trait; `DeepAgentBuilder::build` registers it automatically when a mounted tool card is `get_current_goal` or `submit_goal_report`, and a host drains `GOAL_EVENT_EXTRA_KEY` (`harness.goal_events`) after the task-iteration hooks, because the `DeepAgent` loop has no drain seam of its own.

## Deferred Python behavior

- `read_file`: PDF/Office/notebook/image readers, the token budget estimate, `multimodal` data, device-file refusal and the strict-decode retry; only text files are read.
- `write_file`: the 5 MiB content / 1 GiB file limits, conditional-write registration by the agent, and the `.agent_history` operation log.
- `bash`: `OPENJIUWEN_BASH_STRICT` injection detection, destructive-command warnings, rm-target history, sudo rewriting, `PowerShellTool`, and persisted oversized output.
- `grep`: ripgrep binary invocation, `--type` filters, multiline search, Windows `Select-String`, and `.gitignore` awareness.
- `todo`: `TodoLockManager` per-session process registry (replaced by `TodoStore` locks) and the Python code-numbered `build_error` envelope.
- `memory`: SQLite/embedding hybrid retrieval lives in `openjiuwen_memory` behind `MemoryToolBackend`, but no production `MemoryToolBackend` implementation ships in this crate (only the `FakeBackend` in `memory/tests.rs`; `rails::memory::LongTermMemoryAdapter` implements `MemoryRecall`/`MemoryWriteBack` instead) — the host supplies one via `ToolRegistry::with_memory`; `coding_memory` front-matter validation is the backend's job.
- `ask_user`: the `AskUserRail` interrupt wiring lives in `crate::rails`.

## Routing Table

- `./mod.rs` → `BuiltinTool` and its `DEFAULT`/`EXTENDED`/`ALL` tables, `tool_names`/`extended_tool_names`, flat re-exports of every tool module, module declarations and the `pub use registry::ToolRegistry`
- `./registry.rs` → `ToolRegistry` (builder and configuration setters, `build`, `cards`, `card`, discovery/deferred API, accessors, the shared worktree manager/session-store helpers) and the `BuildCaches`/`memory_kind` arm helpers
- `./support.rs` → `ToolPayload`, `render_fields`, truncation, `schema_for`, `tool_card`, `ToolEnv` (path resolution + permission gate), arg parsing
- `./read_state.rs` → read-before-write registry, fingerprints, range coverage, per-path locks
- `./descriptions.rs` → model-facing description text (en/cn) and language fallback
- `./shell.rs` → `BashTool` and the `bash` card
- `./read.rs` → `ReadFileTool`
- `./write.rs` → `WriteFileTool`, `EditFileTool`
- `./search/` → `glob`, `list_files` and `grep` tools, one per file, plus the in-process grep scanner (`search/CONTEXT.md`)
- `./todo/` → todo store, card-schema inlining, shared validation and the four todo tools, one per file (`todo/CONTEXT.md`)
- `./ask_user.rs` → `UserAsker` seam and `AskUserTool`
- `./web/` → `fetch_webpage`, `web_search`, `HttpFetcher`/`SearchBackend` seams, HTML→text extraction, charset detection (`web/CONTEXT.md`)
- `./media/` → `image_metadata`, `audio_metadata`, pure header parsers for PNG/JPEG/GIF/WebP/BMP and WAV/FLAC/MP3/OGG (`media/CONTEXT.md`)
- `./cron/` → cron expression parser, next-occurrence computation, persisted job list, the `cron` tool (`cron/CONTEXT.md`)
- `./goal/` → goal record/store, `get_current_goal`, `submit_goal_report`, `GoalRail` (`goal/CONTEXT.md`)
- `./discovery.rs` → BM25 `ToolIndex` and the `tool_search` tool used by the registry's deferred-tool mechanism
- `./mcp_resources.rs` → `McpResourceProvider` seam, `McpClientResourceProvider`, `list_mcp_resources`, `read_mcp_resource`, the fake provider
- `./mcp_subscriptions.rs` → `McpResourceSubscriptions` registry, `McpResourceSubscriptionProvider`, sink/event/info types, the fake provider
- `./mcp_subscription_tool.rs` → the `mcp_resource_subscription` tool and its four actions
- `./memory/` → memory backend seam, tool kinds and the eight memory tools (`memory/CONTEXT.md`)
- `./tests.rs` → cross-module parity suite: name order, card schemas/languages, `required` lists, `parallel_safe`, end-to-end registry calls over a temp workspace, `render_for_llm` strings, the `lsp`/`worktree`/`enter_worktree`/`exit_worktree` cards and registry wiring

## Test Strategy

- Unit tests live beside each module: `support` (truncation, render_fields, schema normalization, path confinement, deny gate), `read_state` (coverage merges, partial/stale gates), `shell` (clamping, rendering, timeout, background, blank command), `read` (line numbering, offset/limit, 256 KiB limit, empty file), `write` (read-before-write, create/update, edit uniqueness, `replace_all`, identical strings, CAS conflict), `search` (glob cap 100, brace patterns, list hidden filter, grep modes/context/head_limit/VCS skip), `todo` (create→list→get→modify round-trip, first item in_progress, validation messages, session fallback), `ask_user` (no asker), `memory` (no backend, payload shapes for each of the eight tools).
- `tests.rs` drives one `build()` per temp workspace so the filesystem tools share read state: `read_file`→`edit_file`, `write_file` create, `glob`/`list_files`/`grep` discovery, a `todo_create`→`todo_list` round-trip read back through a second registry over the same workspace, `bash echo`, the missing-asker and missing-memory-backend failures, and the exact `render_for_llm` strings.
- `tests.rs` also asserts the exact model-facing names, per-tool `required` lists, `parallel_safe` flags and `input_params` shape.
- Extended coverage in `tests.rs`: all thirty-five cards and their `required` lists, bilingual descriptions for every card, `DEFAULT` vs `EXTENDED` mounting, deferred-tool hide/discover/reveal, and end-to-end calls for `fetch_webpage` (injected fake fetcher), `web_search` (`FakeSearchBackend`), `image_metadata`/`audio_metadata` (inline header bytes), `cron` create/list/cancel, the goal tools, the MCP resource tools and `tool_search`.
- `lsp`/`worktree` coverage in `tests.rs`: card name/uuid/schema/`required`/`parallel_safe`, `extended_tool_names()` ↔ `BuiltinTool::EXTENDED` (all thirty-five, `from_name` round-trip), the not-configured `lsp` payload naming the missing server, a configured LSP server descriptor reached by the built tool (an impossible command proves the config was used), `LspTool::with_transport` driven by an in-process fake JSON-RPC server over `tokio::io::duplex` (no language-server binary), the `worktree` tool over a scripted `GitRunner` manager (`list` payload), the default workspace-rooted manager's typed git failure, and the permission engine forwarded into `worktree create` (a `deny` becomes a `success=false` payload).
- `enter_worktree`/`exit_worktree` coverage in `tests.rs`: card name/uuid/schema (`enter_worktree` takes an optional `name`, `exit_worktree` requires `action` plus optional `discard_changes`)/`parallel_safe`, mounting in `EXTENDED` and absence from the default set, the default memory session store and `with_worktree_session_store` replacement, the typed git failure without a manager, the permission deny short-circuit, and a scripted-`GitRunner` happy path where `enter_worktree` records the session that `exit_worktree keep` then clears while keeping the default name.
- Module suites: `web` (HTML extraction, charset, fetch over a loopback `TcpListener`, redirects, truncation, timeouts, net-guard denial, search parsing, endpoint configuration), `media` (per-format happy path/truncated/unknown-magic, path confinement), `cron` (grammar, next-occurrence, persistence, tool actions), `goal` (schema round-trip, store, tools, rail heartbeat/events), `discovery`, `mcp_resources`, `mcp_subscriptions` (buffer/lag/drop counters, provider defaults, sink fan-out, two concurrent `recv` callers both observing an event under a bounded await, subscription-cap default/configurability/refusal and slot freeing, reaping on a provider `NotFound` and through `reap`) and `mcp_subscription_tool` (four actions, validation, not-configured payload, `removed: false` + `reason` on a double unsubscribe and on a provider failure, `max_subscriptions` in the list payload, the missing-snapshot fields).
- All tests are offline and deterministic: `tempfile` workspaces, fake `UserAsker`/`MemoryToolBackend`/`HttpFetcher`/`SearchBackend`/`McpResourceProvider`/`GitRunner` implementations, an in-process fake language server over a duplex stream, loopback sockets only, injected clocks, no network, no reliance on `rg`/`git` binaries or a language-server binary.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean, `cargo doc -p openjiuwen-harness --no-deps` warning-free.
- Current counts: 2794 unit tests plus 84 integration and 19 doc tests crate-wide; `mod.rs` (variant tables, re-exports, `registry_tests`) is 497 lines, `registry.rs` 811 and `descriptions.rs` 888, so every source file in this subtree is under the 1000-line baseline and the 1662-line `tests.rs` is the size-noted parity suite.
