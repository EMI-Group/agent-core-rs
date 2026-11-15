# cli

## Intent

Command-line logic for the `openjiuwen` binary: a clap grammar, a one-shot runner, an interactive REPL with slash commands, first-run onboarding, permission prompts, history, colour and file-backed configuration.
Ports the agent-facing parts of `openjiuwen/harness/cli/` (`cli.py`, `ui/repl.py`, `ui/renderer.py`, `ui/runner.py`, `ui/tool_display.py`, `ui/todo_render.py`, `storage/session_store.py`, `agent/{config,factory}.py`); the binary itself lives in `openjiuwen-cli` and only calls `Cli::parse` + `run`.
This module owns stdout/stderr for the product surface: it is the only place in the crate allowed to write to the terminal, and it writes through the injectable [`io`] handles so tests stay deterministic.
Every behaviour is exercisable offline with injected IO, env (`EnvSource`), home directory, shell runner and model; no process-global state, no real terminal, no sleeps.

## API Surface

Module root (`./mod.rs`), every item also re-exported flat from `crate::cli`:
- `Cli` — clap `Parser` (`name = "openjiuwen"`, `version`): global `--model/-m`, `--provider`, `--api-key`, `--api-base`, `--workspace/-w`, `--verbose/-v`, `--no-color`, optional `#[command(subcommand)] command: Option<Command>`.
- `Command::{Run(RunArgs), Chat(ChatArgs)}`; `RunArgs { prompt: Option<String>, output_format: OutputFormat }`; `ChatArgs` (no options yet); `OutputFormat::{Text, Json, StreamJson}` (wire names `text`/`json`/`stream-json`, `as_str()`), `Default = Text`.
- `Dispatch::{Run(RunArgs), Chat}` + `Cli::dispatch(&self, stdin_is_tty: bool) -> Dispatch` (pure: explicit subcommand wins; no subcommand → `Chat` on a TTY, else `Run` with the prompt read from stdin).
- `pub fn stdin_is_tty()` / `pub fn stdout_is_tty()` (`std::io::IsTerminal`) — the only process-level probes, isolated so `dispatch` and colour resolution stay testable.
- `pub async fn run_cli(cli: Cli) -> Result<()>` — resolve options, then: `run` → `ensure_workspace` + `run::run_once` over `CliIo::stdio_non_interactive()` (uncoloured, prompts fail closed); `chat` → `ensure_workspace`, `paths::color_enabled` → `CliIo::stdio().with_color(..)`, `onboarding::maybe_run`, `build_model`, `run::build_agent`, `Repl::with_io_and_signals(..).run()`.
- `pub async fn run(cli: Cli) -> Result<()>` = `run_cli` (the binary-facing spelling used by `openjiuwen-cli`).

`./config.rs` (the module is **private**: `mod config;` — its items are reachable only through the flat `pub use config::{…}` in `./mod.rs` and the crate-root `pub use cli::{Cli, CliOptions, …}`, never as `cli::config::…`):
- `CliOptions { model: String, provider: String, api_key: Option<SecretString>, api_base: Option<String>, workspace: PathBuf, max_iterations: u32, max_tokens: u32, verbose: bool, home: OpenJiuwenHome }` — all fields `pub`; derives `Clone` only, plus a hand-written `Debug` that redacts `api_key`; no `Default`, no `PartialEq`.
- `resolve(&Cli) -> Result<Self>` (sync, reads process env + `<home>/settings.json` + `current_dir`), `resolve_with(&Cli, &dyn EnvSource, Option<&Settings>, cwd: &Path) -> Result<Self>` (pure; relative workspace joins `cwd`), `ensure_workspace() -> Result<()>`, `async build_model() -> Result<Arc<dyn Model>>`, `sessions() -> SessionStore`, `permissions() -> PermissionsSection` (forces `write_file`/`edit_file` to `Ask`).
- `EnvSource` (`fn get(&self, name: &str) -> Option<String>`, **not** `Send + Sync`) with `ProcessEnv` (unit struct, `Debug + Clone + Copy + Default`) and `MapEnv` (`new`, `from_pairs`, `insert`; trims and treats blank as absent); `Settings` (`Debug + Clone + Default + Serialize + Deserialize`, `load(&Path) -> Result<Option<Self>>`, `from_json(&str) -> Result<Self>`, `~/.openjiuwen/settings.json`, alias-tolerant, malformed JSON degrades to defaults); constants `DEFAULT_PROVIDER`/`DEFAULT_MODEL`/`DEFAULT_MAX_*`/`DEFAULT_LANGUAGE`/`MIN_MAX_TOKENS`/`SESSION_PREFIX` and the `OPENJIUWEN_*` env names.
- `home` is the kernel `openjiuwen_common::config::OpenJiuwenHome` (`Debug + Clone + PartialEq + Eq`; `root()`, `join()`, `sessions_dir()`, `logs_dir()`), resolved from `OPENJIUWEN_HOME` or `$HOME/.openjiuwen`.

`./paths.rs`:
- Filenames `HISTORY_FILENAME` (`.openjiuwen_history`), `ALLOW_RULES_FILENAME` (`permissions.yaml`), `MCP_CONFIG_FILENAME` (`mcp.json`), `MCP_CONFIG_DIRNAME` (`.openjiuwen`), `SETTINGS_FILENAME`; `MAX_HISTORY_ENTRIES = 1000`.
- Env switches `NO_COLOR`, `OPENJIUWEN_HISTORY`, `OPENJIUWEN_HISTORY_REPLAY`, `OPENJIUWEN_MCP_CONFIG`, `OPENJIUWEN_NO_ONBOARDING`.
- `settings_path`, `history_path`, `allow_rules_path`, `mcp_config_paths`, `color_enabled(flag, env, is_terminal)`, `onboarding_disabled(env)`, `history_replay_enabled(env)` — the single place every CLI file path is resolved.

`./io.rs`:
- `trait LineReader` / `trait Writer` (`async`, `Send`), `CliIo` handle (`new`, `with_interactive`, `stdio`, `stdio_non_interactive`, `prompt_line`, `write`, `write_line`, `write_blank`, `flush`, `is_interactive`).
- Colour and seeding: `is_color()`, `with_color(bool)` (default off), `seed_lines(impl IntoIterator<Item = String>)`, `pending_lines()` — queued seeds are returned by `prompt_line` before the reader is consulted (history replay).
- `StdioReader` (blocking stdin via `spawn_blocking`) and `StdoutWriter` (flushed after each write).

`./style.rs`: `Style` (`plain`, `new`, `enabled`, `paint`, `cyan|green|yellow|red|dim|bold|bold_cyan|bold_yellow`) plus the ANSI code constants; empty text is never wrapped.

`./render.rs`: `RenderState` (`new`, `styled(bool)`, `style()`, `render(&AgentChunk, Option<&CliIo>, verbose)`, `answer`, `result_type`, `finish`) — tool calls/reasoning/results are coloured only when styling is on, streamed answer text never is.

`./run.rs`:
- `RunOutcome` (`to_json`, `stream_json_line`, `result_json_line`), `session_id()`, `new_session()`, `run_once`, `run_with_model` (plain/text shapes unchanged).
- `stream_run(options, args, io, agent, session, session_id, prompt)` is the shared tail: streaming, per-format rendering/output, `RunOutcome` and the warn-only transcript append; only a caller starting a fresh run may create the record (`SessionStore::new_session` truncates).
- `resolve_prompt(args, io)` is `pub(crate)` so the resumed runner resolves prompts by the same rules.
- `build_agent(model, options, io)` and `build_agent_with(model, options, io, &dyn McpToolProvider)`: load `InstructionFiles` into the system prompt, mount the thirteen extended built-in tools on top of the builder's default twenty, mount MCP tools, merge persisted allow rules, install `CliPermissionHost`.
- `CLI_EXTENDED_TOOLS` / `cli_extended_tools(workspace, permissions)`: the mounted extra set — `fetch_webpage`, `web_search`, `image_metadata`, `audio_metadata`, `cron`, `list_mcp_resources`, `read_mcp_resource`, `tool_search`, `get_current_goal`, `submit_goal_report`, `worktree`, `enter_worktree`, `exit_worktree` — built from a `ToolRegistry` restricted to it and wired with the agent's permission engine.
- `CLI_EXTENDED_TOOLS` is `BuiltinTool::EXTENDED` minus the default twenty minus exactly two documented exclusions: `lsp` (no per-workspace server configuration is resolved) and `mcp_resource_subscription` (no subscription registry is wired).
- `trait McpToolProvider` + `FoundationMcpToolProvider` (real `mcp` connect/list, a failing server is warned and skipped).

`./repl.rs`: `Repl` (`new`, `with_io_and_signals`, `with_io`, plus builders `with_history`, `with_history_replay`, `with_shell`, `with_skills`), `run`, `run_turn`, `seed_history(history: Vec<BaseMessage>)`, `handle_input`, `on_interrupt`, `session_id`; `ReplCommand::{Prompt, Exit, New, Clear, Help, Status, Cost, Compact, Sessions, Mode, Shell, Skill, Usage, Unknown, Empty}`; `TurnOutcome`, `InterruptCounter`, `InterruptAction`.
- `Repl::seed_history` seeds the current kernel session (user/assistant/system/tool messages) and mirrors the user/assistant text into the `/compact` transcript; the private input-history replay helper is `replay_input_history`.

`./resume.rs`: `ResumedTurn` (`new`, `with_history`, `with_state`; fields `session_id`, `history: Vec<BaseMessage>`, `state: Option<Value>`), `run_with_model_resumed(options, args, io, model, resumed)`, `run_once_resumed(options, args, io, resumed)` — plain-data resume surface: seeds the kernel session id, restores the `deepagent` state lane, continues the CLI record and finishes through `run::stream_run`.

`./permissions.rs`: `PermissionAnswer` (`parse`, `to_response`), `render_prompt`, `render_prompt_styled`, `CliPermissionHost` (`new`, `with_allow_rules`, `with_style`, `permission_yaml_path`, `PermissionHost` impl).

`./allow_rules.rs`: `AllowRules` (`new`, `for_home`, `path`, `load`, `persist`, `load_merged`, `yaml`) — `~/.openjiuwen/permissions.yaml`, full `PermissionsSection` YAML, atomic rewrite, malformed file warns and is ignored.

`./session_store.rs`: `SessionRecord`, `MessageRecord`, `SessionSummary`, `SessionStore` (`path_for`, `new_session`, `append_message`, `load`, `list`, `replace_messages`) — `<home>/sessions/<id>.json`, atomic rewrite.

`./history.rs`: `History` (`new`, `with_capacity`, `load`, `record`, `entries`, `clear`) — one entry per line, blanks and repeats skipped, newest 1000 retained, corrupt bytes tolerated.

`./compact.rs`: `Limits`, `HistoryMessage`, `CompactionReport`, `ConversationHistory` (`push`, `messages`, `compact`, `clear`), `truncate_message` — the bounded CLI-side history truncation behind `/compact`.

`./usage.rs`: `UsageTotals` (`record(&AgentChunk)`, `record_turn`, `merge`, `reset`, `render`, `render_line`) — per-session input/output/total tokens, model calls and tool calls.

`./instruction.rs`: `InstructionFiles` (`load(workspace, home)`, `system_prompt`, `documents`, `is_empty`), `InstructionDoc`, `InstructionSource`, `IDENTITY_FILENAME`, `MEMORY_FILENAME`, `MAX_INSTRUCTION_CHARS` (40 000), the memory headers and `TRUNCATION_MARKER`.

`./mcp_config.rs`: `McpConfiguration` (`parse`, `load`, `is_empty`, `servers`), `McpServerSpec`, `load_servers`, `load_for(home, workspace, env)` — `mcpServers` map → `openjiuwen_foundation::tool::mcp::McpServerConfig`.

`./shell.rs`: `ShellRunner` trait, `ProcessShellRunner` (`sh -c`, `kill_on_drop`, no timeout), `ShellOutput` (`success`, `status_line`, `is_empty`).

`./skill_command.rs`: `default_skill_roots`, `parse_skill_command`, `build_skill_query` (byte-exact Python wrapping), `load_skills`, `render_skill_prompt`, `help_rows`.

`./onboarding.rs`: `OnboardingWizard` (`new`, `needed`, `needed_with`, `run`), `OnboardingOutcome::{Skipped, Declined, Configured}`, `maybe_run(io, home, env)` — first-run setup that writes `settings.json`.

`./tests/` → `mod.rs` (original parity suite + shared doubles), `commands.rs` (slash-command/history/shell/skill behaviour), `features.rs` (cross-feature public-API coverage), `resume.rs` (one-shot continuation, `Repl::seed_history`, output parity) (`tests/CONTEXT.md` not present; this section documents them).

## Constraints

- The CLI owns stdout/stderr; this module is the only writer. Libraries log through `tracing`, never `println!`.
- Errors are `openjiuwen_common::error::{Error, Result}`; user-facing failures use `Error::validation`/`config`/`session`/`external`; no panics on user input (`unwrap`/`expect` only under `#[cfg(test)]`).
- `--api-key` is read into `SecretString`; `CliOptions`'s `Debug` redacts it and no prompt echoes it.
- Async-first: IO traits are async; blocking stdin only inside `StdioReader`; no std/parking_lot lock held across `.await` (IO state is behind `tokio::sync::Mutex`).
- `CliOptions` fields are frozen: `crates/openjiuwen-e2e/tests/cli_binary.rs` and `tests/cli.rs` construct it with a struct literal, so new knobs go into `paths`/`Cli`/`CliIo`, never into `CliOptions`.
- Colour is opt-in per handle (`CliIo::with_color`); plain output must stay byte-identical so piped `text`, `json` and `stream-json` runs contain no escape bytes.
- Files ≤ ~1000 lines; `repl.rs` (786) is the largest logic file and stays single-responsibility; the test tree is split per concern.
- No dependency on `crate::protocol`, `crate::subagents` or upper-layer crates; only `crate::agent`, `crate::permission`, `crate::schema`, `crate::skills`, `crate::workspace` and kernel crates.
- Precedence everywhere is explicit and documented: flags → `OPENJIUWEN_*` → generic SDK env → settings file → default for options; workspace identity over home identity, home memory before workspace memory for instructions; home `mcp.json` before workspace `.openjiuwen/mcp.json` (workspace wins by server name).

## Design decisions

- **`CliOptions` stays field-frozen** (external struct literals): colour lives on `CliIo`, onboarding on `run_cli` + `paths`, and every file path in `paths`.
- **REPL self-wires** history (`paths::history_path`), replay (`history_replay_enabled`), skills (`default_skill_roots`) and the shell runner from the injected options, so `run_cli` is just `Repl::with_io_and_signals(..).run()`; the `with_*` builders exist for tests.
- **`/clear` differs from `/new`** (documented Rust addition): `/clear` resets session, conversation mirror, usage counters and agent state; `/new` only starts a new persisted session.
- **`/compact` is bounded truncation, not summarisation**: the kernel `ContextEngine` is not reachable from `DeepAgent`, so `ConversationHistory` keeps the newest 8 messages within a 12 000-character budget and the trimmed transcript is written back through `SessionStore::replace_messages`.
- **`/cost` counts what the kernel actually emits**: tokens from every `llm_usage` frame (`usage_metadata`), tool calls from the `answer` frame's `observations` (the `tool_call`/`tool_result` frame kinds never appear on `DeepAgent::stream`).
- **Fail closed on permissions**: a non-interactive handle never prompts; EOF rejects the call; only an interactive terminal answers, and `Always` persists the whole merged section to `permissions.yaml`.
- **Resuming is a seeded turn on an existing session id**: `ResumedTurn` carries the id, the prior message window and the optional `deepagent` state; `run_with_model_resumed`/`run_once_resumed` build the agent, seed it (`DeepAgent::seed_history`, then `agent::seed_state`), continue the record and finish through `run::stream_run`. Seed errors propagate; `run_once`/`run_with_model` and a bare `Repl` still start from `new_session(id)` and send `json!({"query": …})`.
- **A resumed run continues the CLI record, never truncates it**: the record is created only when `SessionStore::load` reports `None`; an existing `<home>/sessions/<id>.json` keeps its earlier messages and only gains the new user/assistant pair.
- `SessionStore` transcripts stay write-only from the agent's point of view: they are read back by `/sessions` (summaries), rewritten by `/compact` from the CLI-side `ConversationHistory`, and existence-checked by the resumed runner; `SessionStore::load` is never fed to `build_agent`/`DeepAgent`.
- The `resume` module is the host seam for the crate's `persistent_agent` module (`PersistentSessionRecord`, `ResumePlan`, `WindowAssembly`, `resume`): `src/cli/` imports nothing from it — a host extracts a `ResumePlan` (or any other store's data) into `ResumedTurn` itself, because `cli` must never depend on `persistent_agent`.
- **Persistent allow rules are a Rust addition**: Python keeps "always" in memory for the session only; here `CliPermissionHost::with_allow_rules` writes YAML and the next run reloads it via `AllowRules::load_merged(&options.permissions())`.
- **Instruction files**: `IDENTITY.md` (workspace wins over home) is a Rust addition injected through `DeepAgentConfig::with_system_prompt` (which replaces the built-in identity preamble); `OPENJIUWEN.md` layers keep the Python `### User-level memory` / `### Project-level memory` headers and the 40 000-character cap.
- **MCP config is typed and loud**: malformed `mcp.json` returns `Error::config` naming the path (Python silently drops every server); a server that fails to connect is warned and skipped at build time so one bad entry never blocks the CLI.
- **Onboarding is a Rust addition** with a decline question: it runs only for interactive `chat`, only when `<home>/settings.json` is absent, and is disabled by `OPENJIUWEN_NO_ONBOARDING`; non-interactive runs never prompt and never write.
- **Colour resolution**: `paths::color_enabled(flag, env, stdout_is_tty())` gates `CliIo::with_color`; `NO_COLOR` and `--no-color` both opt out, non-terminals are always plain.
- **History replay is opt-in**: `OPENJIUWEN_HISTORY_REPLAY` seeds `CliIo` with stored lines (the offline analogue of a pre-filled line editor); submitted lines are always appended best-effort and never fail a turn.
- **`run_with_model`/`build_agent` signatures are frozen** (an out-of-crate e2e suite calls them); richer wiring goes through `build_agent_with` and the builder methods.
- **Agent root vs workspace**: `build_agent_with` sets `DeepAgentConfig::cwd`/`project_root` to the process directory (`std::env::current_dir()`, falling back to `options.workspace`), while the tool `Workspace` tree is `Workspace::for_language(&options.workspace, "en")`.
- **CLI tool set**: the CLI mounts `CLI_EXTENDED_TOOLS` (the fifteen `BuiltinTool::EXTENDED` entries beyond the default twenty, minus `lsp` and `mcp_resource_subscription`) through `ToolRegistry::with_tools`, so the builder baseline stays the default twenty for library users.
- **Two extended tools stay unmounted**: `lsp` needs a per-workspace language-server configuration the CLI does not resolve, and `mcp_resource_subscription` needs a subscription registry the CLI does not wire; mounting either would only offer the model a tool that always answers "not configured".
- **The three worktree tools are mounted together**: `worktree`, `enter_worktree` and `exit_worktree` share the registry's in-process `MemoryWorktreeSessionStore` and its manager rooted at the workspace, so they work inside a git repository and degrade to typed `success=false` payloads outside one (`exit_worktree` reports no active session without touching git).
- **No backend defaults for the online tools**: `web_search` and the two MCP resource tools stay backend-less and answer with their documented `success=false` payload, `fetch_webpage` keeps the workspace `reqwest` fetcher, and no search key or endpoint is configured by the CLI.
- **Goal tools register the goal rail**: mounting `get_current_goal`/`submit_goal_report` makes the factory register `tools::goal::GoalRail`, so a CLI agent publishes the goal prompt section and its `goal.updated` events.

## Known Issues

- Multi-turn REPL sessions go silent: the task loop's `NoProgressAnswer` guard (`max_consecutive_empty_answers = 3`, `min_answer_chars = 20`, enabled by default) keeps its streak in the reused session state, so after three consecutive short answers the next `agent.stream` returns one empty `answer` frame without calling the model. Fix belongs to the task-loop/agent area, not this module.
- `ReplCommand`'s `Unknown skill:` branch is unreachable from `handle_input` (unknown names are rejected by `parse_skill_command` first and surface as `Unknown command: /name`).
- `crates/openjiuwen-harness/tests/cli.rs` covers the pre-extension surface only; the new features are covered by `src/cli/tests/{commands,features,resume}.rs` instead (that file is outside this node).
- A resumed REPL turn is not interruptible in a new way: `Repl::seed_history` seeds a built REPL only, so a host that resumes into `chat` must call it before the first turn and after any `agent.configure`.
- `Repl` cannot adopt a persisted session id: `Repl::build` mints a fresh id (`session_id()`), no constructor or `with_*` setter accepts one, and the field is reassigned only by `/new` and `/clear`.
Consequently a host that resumes into `chat` (e.g. `openjiuwen-cli`'s `chat --resume`) seeds the history into a newly minted `cli-*` session, whose `SessionStore` record starts empty rather than continuing the persisted transcript.
- `run_once_resumed` has no behavioural test: it resolves a real provider through `CliOptions::build_model`, so every offline suite drives `run_with_model_resumed` (which differs only by that call); `crates/openjiuwen-harness/tests/public_surface.rs` only signature-pins it.
- `build_agent_with` installs `CliPermissionHost` with a freshly generated `run::session_id()`, not the caller's run/REPL session id, so `PermissionHost::session_id()` never matches the id in `<home>/sessions/<id>.json`.
- The one-shot `run` path never emits colour, even on a terminal: `run_cli` resolves colour only in the `chat` branch and hands `run` a `CliIo::stdio_non_interactive()` whose colour flag stays off.

## Deferred Python areas

- `auto-harness` subcommands (`run`, `experience`, `verify-ext`, `gap-analyze`, `history`), their Chinese UI and `/auto-harness`.
- Remote mode (`--remote`, `OPENJIUWEN_SERVER_URL`, `RemoteBackend`).
- Rich/prompt_toolkit TUI: panels/logo banner, `SlashCompleter`, multiline editing, markdown rendering, diff rendering in tool results (plain ANSI styling only).
- Dollar-price `/cost` reporting (Python has none either: token/call counts only).
- LLM-summarising `/compact` (bounded truncation instead).
- The `lsp` built-in tool is not mounted by the CLI: a language server needs a per-workspace configuration the CLI does not resolve yet.
- Plugin hot-loading (templates are loaded by `crate::spec`, not by the CLI) and a `--config` command.
- No automatic resume in the binary: there is no `--resume`/`--session` flag and `run_cli` never restores a past transcript by itself; the `resume` module is the library seam a host drives.
- Multi-skill roots beyond `<workspace>/skills` and `<home>/skills` (`~/.claude/skills`, `~/.codex/skills` are not scanned).

## Routing Table

- `./mod.rs` → `Cli`, `Command`, `RunArgs`/`ChatArgs`, `OutputFormat`, `Dispatch`, `stdin_is_tty`/`stdout_is_tty`, `run_cli`/`run`, flat re-exports
- `./config.rs` → `CliOptions` resolution (flags/env/settings/defaults), `EnvSource`, `Settings`, model construction, permission section
- `./paths.rs` → every CLI file path and env switch
- `./io.rs` → `LineReader`/`Writer` traits, `CliIo` (colour flag, seed lines), stdio implementations
- `./style.rs` → ANSI `Style` helpers
- `./render.rs` → `RenderState`: chunk-to-terminal rendering shared by `run` and the REPL
- `./run.rs` → one-shot execution, agent construction (instructions, extended built-in tools, MCP, permissions), output formats, `RunOutcome`
- `./repl.rs` → `Repl`, slash commands, history integration, skill/shell dispatch, Ctrl-C handling, per-turn streaming, `seed_history`
- `./resume.rs` → `ResumedTurn` plus the resumed one-shot entry points (`run_with_model_resumed`, `run_once_resumed`): seed history/state on an existing session id and continue through `run::stream_run`
- `./permissions.rs` → approval prompt text/styling, answer mapping, `CliPermissionHost` with persistent rules
- `./allow_rules.rs` → `permissions.yaml` persistence and merge
- `./session_store.rs` → `~/.openjiuwen/sessions/<id>.json` records (append, list, replace)
- `./history.rs` → capped input history file
- `./compact.rs` → bounded conversation truncation behind `/compact`
- `./usage.rs` → `/cost` token and tool-call accounting
- `./instruction.rs` → `IDENTITY.md`/`OPENJIUWEN.md` discovery and prompt composition
- `./mcp_config.rs` → `mcp.json` parsing/merging into foundation server configs
- `./shell.rs` → injectable `!cmd` runner
- `./skill_command.rs` → `/<skill>` resolution and prompt wrapping
- `./onboarding.rs` → first-run wizard writing `settings.json`
- `./tests/mod.rs` → original parsing/precedence/permission/runner/REPL parity suite and shared doubles
- `./tests/commands.rs` → slash-command, history, shell and skill behaviour
- `./tests/features.rs` → cross-feature public-API coverage (colour, history, onboarding, instructions, MCP, allow rules, `/cost`, `/compact`, skills)
- `./tests/resume.rs` → `ResumedTurn` builders, record continuation, seeded-window ordering, `Repl::seed_history` over a turn, resumed-vs-fresh output parity

## Test Strategy

- All suites are offline, deterministic and injected: scripted `LineReader`/`Writer` doubles, `MapEnv`, `tempfile` homes/workspaces, `ScriptedModel`, fake shell runners and fake MCP providers; no real terminal, network, sleeps or process-env reads.
- `tests/mod.rs` keeps the parity guards: clap parsing/dispatch, option precedence, settings aliases, permission answer mapping, session store round-trip, renderer per-kind output, `run_with_model` text/json/stream-json shapes, scripted REPL transcript.
- `tests/commands.rs` drives each slash command end to end: `handle_input` table, `/help` (with and without skills), `/cost`/`/status` usage, `/clear`, `/compact` (no-op and truncating), `/sessions` (empty and populated), `!cmd` (stdout/stderr/exit line, model untouched, failure keeps the loop alive), skill turn prompt wrapping, history recording and replay seeding.
- `tests/features.rs` proves the wiring from the public API: `--no-color` parsing + `color_enabled`, styled-vs-plain prompt bytes, history round-trip/cap, onboarding interactive vs non-interactive, instruction precedence, `mcp.json` loading and malformed-file error, allow-rule persistence through the host, and escape-free `text`/single-object `json` output.
- `tests/resume.rs` proves the resume surface: a pre-created record keeps its earlier message (three messages in order, roles `user`/`user`/`assistant`), the seeded window reaches `ScriptedModel::last_call()` ahead of the new query, `Repl::seed_history` survives `run_turn` (model window plus rendered answer), the builders expose their fields, and a resumed `text` run writes byte-identical output to `run_with_model`.
- `run.rs`/`permissions.rs` carry inline tests for `build_agent_with` (no MCP config → offline turn, recording provider, instruction prompt, persisted rules) and the styled/fail-closed permission host.
- `run.rs` also pins the CLI tool set: the `CLI_EXTENDED_TOOLS` shape (extended minus the default twenty minus the two documented exclusions, disjoint from the default twenty), the mounted card names in the rendered prompt, the absence of both excluded cards, the automatic `goal` rail, the typed `success=false` payloads of the backend-less `web_search`/`list_mcp_resources`/`fetch_webpage` tools and of `enter_worktree`/`exit_worktree` outside a repository, and a scripted `web_search` run whose tool result carries "web search backend is not configured".
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings` all clean.
- Parity references (read-only): `openjiuwen/harness/cli/{cli.py,ui/repl.py,ui/renderer.py,ui/runner.py,ui/tool_display.py,storage/session_store.py,prompts/builder.py,rails/{token_tracker,tool_tracker}.py,agent/{config,factory}.py}`.
