# cli
## Intent
Command-line logic for the `openjiuwen` binary: a clap-based `run` command and an interactive REPL that build a `DeepAgent`, stream output, and answer permission prompts.
Ports the agent-facing parts of `openjiuwen/harness/cli/` (`cli.py`, `agent/factory.py`, `ui/repl.py`, `ui/runner.py`, `ui/renderer.py`); the binary itself lives in `openjiuwen-cli` and only calls `Cli::parse` + `run_cli`.
This module owns stdout/stderr for the product surface: it is the only place in the crate allowed to write to the terminal, and it writes through the injectable [`io`] handles so tests stay deterministic.
## API Surface
Module root (`./mod.rs`), every item also re-exported flat from `crate::cli`:
- `Cli` — clap `Parser` (`name = "openjiuwen"`, `version`): global `--model/-m`, `--provider`, `--api-key`, `--api-base`, `--workspace/-w`, `--verbose/-v`, optional `#[command(subcommand)] command: Option<Command>`.
- `Command::{Run(RunArgs), Chat(ChatArgs)}`; `RunArgs { prompt: Option<String>, output_format: OutputFormat }`; `ChatArgs` (no options yet); `OutputFormat::{Text, Json, StreamJson}` (`ValueEnum`, wire names `text`/`json`/`stream-json`, `as_str()`), `Default = Text`.
- `Dispatch::{Run(RunArgs), Chat}` + `Cli::dispatch(&self, stdin_is_tty: bool) -> Dispatch` (pure: explicit subcommand wins; no subcommand → `Chat` on a TTY, else `Run` with no prompt so the prompt is read from stdin).
- `pub fn stdin_is_tty() -> bool` (`std::io::IsTerminal`) — the only process-level probe, isolated so `dispatch` stays testable.
- `pub async fn run_cli(cli: Cli) -> Result<()>` — canonical entry point: resolve `CliOptions`, dispatch, then `run::run_once` or `Repl::run`; the binary uses `pub use run_cli as run`.
- Deprecated-free alias `pub async fn run(cli: Cli) -> Result<()>` = `run_cli` (the binary-facing spelling used by `openjiuwen-cli`).
`./config.rs`:
- `CliOptions { model, provider, api_key: Option<SecretString>, api_base: Option<String>, workspace: PathBuf, max_iterations: u32, max_tokens: u32, verbose: bool, home: OpenJiuwenHome }` + redacted `Debug`; `CliOptions::resolve(&Cli) -> Result<Self>`, `resolve_with(&Cli, &dyn EnvSource, Option<&Settings>, cwd: &Path) -> Result<Self>` (pure precedence core), `ensure_workspace(&self) -> Result<()>`, `async fn build_model(&self) -> Result<Arc<dyn Model>>`, `sessions(&self) -> SessionStore`, `permissions(&self) -> PermissionsSection`.
- `EnvSource` trait (`get(name) -> Option<String>`) with `ProcessEnv` (via `openjiuwen_common::config::env`) and `MapEnv` (test double); unset/blank → `None`.
- `Settings` — `~/.openjiuwen/settings.json`, serde alias-tolerant (`api_key`/`apiKey`, `api_base`/`apiBase`, `max_iterations`/`maxIterations`, `max_tokens`/`maxTokens`); `Settings::load(path)`, `Settings::from_json`, unknown keys ignored, malformed JSON logs a warning and degrades to `Settings::default()`.
- Constants: `DEFAULT_PROVIDER = "openai"`, `DEFAULT_MODEL = "gpt-4o"`, `DEFAULT_MAX_ITERATIONS = 30`, `DEFAULT_MAX_TOKENS = 8192`, `DEFAULT_LANGUAGE = "en"`, `MIN_MAX_TOKENS = 256`, `SESSION_PREFIX = "cli-"`, and env names `OPENJIUWEN_API_BASE`/`OPENJIUWEN_MODEL`/`OPENJIUWEN_PROVIDER`/`OPENJIUWEN_WORKSPACE`/`OPENJIUWEN_MAX_ITERATIONS`/`OPENJIUWEN_MAX_TOKENS` (not yet in `openjiuwen_common::config::env`).
`./io.rs`:
- `#[async_trait] trait LineReader: Send { async fn read_line(&mut self) -> Result<Option<String>> }` (`None` = EOF) and `#[async_trait] trait Writer: Send { async fn write(&mut self, text: &str) -> Result<()>; async fn flush(&mut self) -> Result<()> }`.
- `#[derive(Clone)] CliIo` — `Arc<tokio::sync::Mutex<(Box<dyn LineReader>, Box<dyn Writer>)>>`; `CliIo::new(reader, writer)` (non-interactive), `CliIo::with_interactive(reader, writer)` (same but interactive; the injectable seam for tests and embedding), `CliIo::stdio()`, `prompt_line(&self, prompt) -> Result<Option<String>>` (writes prompt, flushes, reads one trimmed line), `write/write_line/write_blank/flush`, `is_interactive(&self) -> bool` (true for the stdio handle and `with_interactive`; drives fail-closed permission behavior).
- `StdioReader` (`tokio::task::spawn_blocking` around `std::io::stdin().read_line`), `StdoutWriter` (locked stdout, flushed after each write).
`./render.rs`:
- `RenderState { text: String, chunk_types: Vec<String>, usage: Option<Value>, interactions: Vec<Value>, result_type: Option<String> }`, `RenderState::new`, `async fn render(&mut self, chunk: &AgentChunk, out: Option<&CliIo>, verbose: bool) -> Result<()>`, `fn answer(&self) -> &str`, `fn finish(&mut self, out: Option<&CliIo>) -> Result<()>`.
- Per-kind output: `llm_output` raw text + trailing newline per burst; `answer` captured (printed only when no `llm_output` was streamed); `llm_reasoning` printed dim only when `verbose`; `llm_usage` captured into `usage`; `tool_call` → `● {name}({args})`; `tool_result` → `  ⎿  {summary}` + blank line; `__interaction__` recorded (id + value) and never printed; every other kind is recorded in `chunk_types` and ignored.
`./run.rs`:
- `RunOutcome { answer: String, result_type: String, usage: Option<Value>, session_id: String, chunks: usize }` + `RunOutcome::to_json(&self) -> Value` (`{answer, result_type, usage, session_id}`) and `stream_json_line(&AgentChunk) -> Value` (`{type, index, payload}`), `result_json_line(&RunOutcome) -> Value` (`{"type":"result", …to_json}`).
- `pub async fn run_once(options: &CliOptions, args: &RunArgs, io: &CliIo) -> Result<RunOutcome>` — resolves the prompt (`args.prompt` or stdin when `None`/`-`, trimmed; empty → `Error::validation("A prompt argument is required, or pipe via stdin.")`), builds the model, delegates to `run_with_model`.
- `pub async fn run_with_model(options: &CliOptions, args: &RunArgs, io: &CliIo, model: Arc<dyn Model>) -> Result<RunOutcome>` — the test seam: builds the agent, runs one streamed invocation, renders per `output_format` (`Text` renders as it streams, `Json`/`StreamJson` collect and print exactly one object/lines at the end), writes the session record.
- `pub async fn build_agent(model: Arc<dyn Model>, options: &CliOptions, io: &CliIo) -> Result<Arc<DeepAgent>>` and `pub fn new_session(session_id: &str) -> Arc<dyn Session>`; `pub fn session_id() -> String` (`cli-<uuid8>`).
`./repl.rs`:
- `Repl::new(agent: Arc<DeepAgent>, options: CliOptions) -> Self` (stdio IO + real Ctrl-C watcher), `Repl::with_io_and_signals(agent, options, io: CliIo) -> Self` (terminal-backed over an existing stdio handle, watcher enabled; the binary path that shares one handle with the permission host), and `Repl::with_io(agent, options, io: CliIo) -> Self` (injectable, no signal watcher — the test path).
- `async fn run(&mut self) -> Result<()>`; `async fn run_turn(&mut self, input: &str) -> Result<TurnOutcome>`; `fn handle_input(&self, line: &str) -> ReplCommand` (pure: `Prompt(String)`, `Exit`, `New`, `Mode(AgentMode)`, `Help`, `Status`, `Usage(&'static str)`, `Unknown(String)`, `Empty`); `fn on_interrupt(&self) -> InterruptAction`.
- Slash commands: `/exit` (alias `/quit`) → `Goodbye!`; `/new` → new `cli-<uuid8>` session, clears agent state, `Started a new session: <id>`; `/help` → the fixed command table; `/mode <normal|plan>` → `agent.switch_mode`, `Mode: <mode>` (bad/missing arg → usage line); `/status` → model/provider/session id; unknown → `Unknown command: <cmd>` + `Type /help to see available commands.`; empty line ignored.
- `InterruptCounter` (`AtomicUsize`) + `InterruptAction::{Abort, Exit}`: first SIGINT calls `agent.abort()` and prints `⏹ Interrupted`, second requests exit; counter resets per turn.
- Per-turn flow: render the prompt `You> `, stream via `agent.stream({"query": input}, session)`, render chunks through `RenderState`, persist the user and assistant messages in `SessionStore`, reuse the same session across turns.
`./permissions.rs`:
- `PermissionAnswer::{Approve, Always, Reject, Feedback(String)}` + `PermissionAnswer::parse(&str)` (trim + lowercase; empty/`y`/`yes`/`ok`/`approve`/`allow`/`allow_once`/`true`/`1` → `Approve`; `a`/`always`/`allow_always` → `Always`; `n`/`no`/`deny`/`reject` → `Reject`; anything else → `Feedback(text)`) + `fn to_response(&self) -> PermissionConfirmResponse` (`Approve` → approved, no auto-confirm; `Always` → approved + `auto_confirm = true`; `Reject` → rejected, empty feedback; `Feedback(text)` → rejected with `text`).
- `fn render_prompt(tool: &str, arguments: &Value) -> String` — `⚠ Approve {tool}?` + up to three `  {key}: {value}` arg lines (values truncated at 200 chars + `...`) + `(y/yes, a/always, n/no, or feedback)` + `Approve? (y/a/n)`.
- `CliPermissionHost { io, session_id, workspace, permissions: RwLock<Option<PermissionsSection>> }` (`new(io, session_id, workspace, permissions)`), impl `PermissionHost`: `request_confirmation` renders and parses (EOF or non-interactive → `ConfirmationOutcome::Failed`, i.e. fail closed), `session_id`, `resolve_workspace_dir`, `permissions_snapshot`, `persist_session_allow_rule` (store the merged section, return `true`), `persist_allow_rule` → `false` (no permanent YAML write yet).
`./session_store.rs`:
- `SessionRecord { session_id, model, created_at, messages: Vec<MessageRecord> }`, `MessageRecord { role, content, timestamp }`; `SessionStore::new(&OpenJiuwenHome)`, `path_for(&self, session_id)`, `new_session(&self, session_id, model) -> Result<SessionRecord>` (writes the file), `append_message(&self, session_id, role, content) -> Result<()>` (load-or-create, push, atomic rewrite via a `.tmp` + rename), `load(&self, session_id) -> Result<Option<SessionRecord>>`.
- JSON shape is Python's: `{"session_id", "model", "created_at" (RFC-3339 UTC), "messages": [{"role", "content", "timestamp"}]}`; `role` is `user`/`assistant`.
## Constraints
- The CLI owns stdout/stderr; this module is the only writer. Libraries log through `tracing`, never `println!`.
- Errors are `openjiuwen_common::error::{Error, Result}` (no `anyhow`); user-facing failures are `Error::validation`/`Error::config`/`Error::session` with the Python message text; no panics on user input (`unwrap`/`expect` only under `#[cfg(test)]`).
- `--api-key` is read into `SecretString`; `CliOptions`'s `Debug` redacts key and never logs secrets; permission prompts never echo the key.
- Async-first: `LineReader`/`Writer` are async traits; blocking stdin happens only inside `StdioReader` via `spawn_blocking`; no `parking_lot`/`std` lock is held across `.await` (IO state sits behind `tokio::sync::Mutex`).
- Workspace and sessions directories are created lazily (`ensure_workspace` / first `SessionStore` write), so `--help` and parse errors never touch the disk.
- Files ≤1000 lines; the renderer, session store and IO handles are separate files so `run.rs` and `repl.rs` share them instead of duplicating.
- No dependency on `crate::protocol`, `crate::subagents` or upper-layer crates; only `crate::agent`, `crate::permission`, `crate::schema`, `crate::workspace` and kernel crates.
- Prompt texts and command output match the Python CLI where ported (`You> `, `/help` table, `⚠ Approve …`, `⏹ Interrupted`, `Goodbye!`, `Unknown command: …`).
## Design decisions
- **Deviation from Python dispatch text**: the Python group has no `/new` or `/mode`; the Rust REPL adds them (session reset and plan-mode switching are first-class in `DeepAgent`). Documented addition, not a parity bug.
- **`/mode` usage line**: a missing or invalid `/mode` argument yields `ReplCommand::Usage("Usage: /mode <normal|plan>")`, so bad input prints usage instead of the generic unknown-command hint.
- **Permission host session id**: `build_agent` gives `CliPermissionHost` a freshly generated `cli-<uuid8>` id because the one-shot/REPL session id is created after the agent; approvals live only in the host's in-memory section (`persist_allow_rule` returns `false`), so the ids never need to match.
- **JSON output shapes** follow the product spec, not Python's `{result, chunks, model}`: `json` prints `{answer, result_type, usage, session_id}`; `stream-json` prints one `{type, index, payload}` object per chunk plus a terminal `{"type":"result", …}` line so consumers learn the session id (Python emits chunks only). Documented addition.
- **IO is a cloneable handle, not a `&mut` borrow**: the permission host is called from inside `agent.stream()` while the REPL renders the same stream, so both sides clone `CliIo` and lock per call — no lock is ever held across the stream await, which avoids a self-deadlock between renderer and confirm prompt.
- **Ctrl-C is cooperative**: the watcher task (stdio REPL only) calls `Repl::on_interrupt`, which calls `DeepAgent::abort()` (cancellation token) and records the press; the stream ends with an abort outcome, so no `select!` over the chunk stream is needed and tests drive `on_interrupt` directly.
- **Fail closed on permissions**: `run` (non-interactive) never answers a prompt — EOF yields `ConfirmationOutcome::Failed`, so the tool call is rejected; only an interactive TTY with empty Enter approves once (Python parity).
- **Model construction is a seam**: `CliOptions::build_model` is the only place `init_model` is called; `run_with_model` and `Repl::with_io` accept a prebuilt model/agent so tests use `ScriptedModel` and never touch the network.
- **Session persistence is append-only and best-effort**: a failed session write logs `tracing::warn!` and never loses the answer; there is no history-restore path (Python has none either).
- **`run.rs` module vs `run` fn**: legal in Rust (module in the type namespace, function in the value namespace); `mod.rs` keeps `use run::...` for items and defines the public alias `pub async fn run`.
- **`OPENJIUWEN_*` gaps**: `OPENJIUWEN_API_BASE`, `OPENJIUWEN_MODEL`, `OPENJIUWEN_PROVIDER`, `OPENJIUWEN_WORKSPACE`, `OPENJIUWEN_MAX_*` are defined here because `openjiuwen_common::config::env` does not carry them; they belong in common once another crate needs them.
## Deferred Python areas
- `auto-harness` subcommands (`run`, `experience`, `verify-ext`, `gap-analyze`, `history`) and their Chinese UI.
- Remote mode (`--remote`, `OPENJIUWEN_SERVER_URL`, `RemoteBackend`).
- Rich/prompt_toolkit rendering, ANSI colouring, `SlashCompleter`, `~/.openjiuwen_history` file history, multiline input.
- First-run onboarding wizard that writes `settings.json`.
- `/clear`, `/cost`, `/compact`, `/sessions` and the skill-slash (`/<skill>`) and `!cmd` shell-passthrough commands.
- MCP loading from `~/.openjiuwen/mcp.json`, web tools, CLI `IDENTITY.md` override, `OPENJIUWEN.md` memory prompt, plugin hot-loading.
- Permanent allow-rule YAML write-back (`persist_allow_rule` returns `false`), file-based session history restore.
## Routing Table
- `./mod.rs` → `Cli`, `Command`, `RunArgs`/`ChatArgs`, `OutputFormat`, `Dispatch`, `run_cli`/`run`, flat re-exports
- `./config.rs` → `CliOptions` resolution (flags/env/settings/defaults), `EnvSource`, `Settings`, paths, model construction, permission section
- `./io.rs` → `LineReader`/`Writer` traits, `CliIo` handle, stdio implementations
- `./render.rs` → `RenderState`: chunk-to-terminal rendering shared by `run` and the REPL
- `./run.rs` → one-shot execution, agent/session construction, output formats, `RunOutcome`
- `./repl.rs` → `Repl`, slash commands, Ctrl-C handling, per-turn streaming
- `./permissions.rs` → approval prompt text, answer mapping, `CliPermissionHost`
- `./session_store.rs` → `~/.openjiuwen/sessions/<id>.json` records (append-only)
- `./tests.rs` → node unit/integration suite (parsing, precedence, prompt mapping, scripted REPL, output shapes)
## Test Strategy
- `./tests.rs` (offline, `ScriptedModel`, `tempfile`, injected `MapEnv`/`CliIo`):
  clap parsing (defaults, `-m/-w/-v`, `run` output-format variants, no-subcommand); `dispatch` with `stdin_is_tty` true/false and explicit subcommands;
  option precedence for model/provider/key/base/workspace across flag → `OPENJIUWEN_*` → generic env → settings → defaults, plus redacted `Debug`;
  settings JSON alias tolerance and malformed-JSON degradation; `max_tokens`/`max_iterations` validation messages;
  `PermissionAnswer::parse` table + response mapping (`Always` sets `auto_confirm`), prompt text with arg truncation, `CliPermissionHost` answer/EOF-fail-closed;
  `SessionStore` write/append/load round-trip and file naming; renderer per-kind output and verbose reasoning;
  `run_with_model` text/json/stream-json shapes with a scripted agent (answer text, JSON keys, one line per chunk + result line) and the empty-prompt usage error;
  scripted REPL transcript (`/help`, `/exit`, one streamed turn, `/new`, `/mode plan`, unknown command) with a scripted terminal.
- Keep process-env access out of tests by using `MapEnv`; where a test must touch process env, hold the same static `ENV_LOCK` guard style used by `openjiuwen_common::config` tests.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` clean.
- `./tests.rs` is the node's single mandated test file and is intentionally ~1k lines (52 tests across all child modules); its file-level doc comment records that role.
- Parity references (read-only): `openjiuwen/harness/cli/cli.py`, `agent/factory.py`, `ui/repl.py`, `ui/runner.py`, `ui/renderer.py`, `tests/unit_tests/cli/test_agent_factory.py`, `tests/cli/unit/test_renderer.py`.
