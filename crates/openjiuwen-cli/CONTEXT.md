# openjiuwen-cli

## Intent

The `openjiuwen` executable: the product entry point a user runs.
It is a thin process boundary that owns the outer clap grammar and dispatches every command group.
`run`/`chat` are forwarded to `openjiuwen_harness::cli::run_cli`; `auto-harness`, `lsp`, `cron`, `team`, `watch` and `sessions` are dispatched here.
The harness-surface commands are terminal ends of ports the harness already owns: its language-server client, its cron scheduler, the `openjiuwen-agent-teams` attach protocol, its workspace watcher, and its persistent-agent stores.
`run`/`chat` additionally accept the flags the harness grammar has no words for (`--worktree`, `--watch` on `chat`, and `--resume <ID>`/`--continue` on both).
`--resume`/`--continue` are resolved by the session catalogue in `src/sessions/catalog.rs`, which merges the CLI session records, the persistent-agent execution journals and the persistent-agent session records; the resume note goes to stderr so stdout keeps the run's output format.
`sessions` exposes that same catalogue: list, show and prune.
The operating-system concerns also live here: `.env` loading, `tracing` installation on stderr, clap's exit codes, and the stderr diagnostic for a failed run.
Everything about `run`/`chat` behaviour — option resolution, the one-shot runner, the REPL, permission prompts, session storage — belongs to the harness.
Python parity: the `openjiuwen` console script and `openjiuwen/harness/cli/cli.py`'s `main` plus its `auto-harness` Click group; `lsp`, `cron`, `team`, `watch`, the two `run`/`chat` flags and the `--resume`/`--continue` pair are Rust-side surfaces over harness ports.

## API Surface

Binary `openjiuwen` (`src/main.rs`, `[[bin]] name = "openjiuwen"`); the crate has no library target.
- Entry point: `#[tokio::main] async fn main() -> ExitCode`.
- Startup order: `dotenvy::dotenv()` (errors ignored, walks the working directory and its ancestors) → `init_logging()` → `cli::Cli::parse()` → `cli::run(cli).await`.
- `init_logging()` installs `openjiuwen_common::logging` with `LogWriter::Stderr`; `RUST_LOG` (trimmed, ignored when blank) wins over `OPENJIUWEN_LOG_LEVEL`/`OPENJIUWEN_LOG_FORMAT`, and an invalid filter prints a warning and falls back to the env/default substrate so logging never blocks startup.

### Grammar (`src/cli.rs`)

- `Cli { global: GlobalArgs, command: Option<Command> }`, `#[command(name = "openjiuwen", version, about = "openJiuwen agent CLI")]`.
- `GlobalArgs` mirrors the harness globals flag-for-flag and all are `global = true`: `-m/--model`, `--provider`, `--api-key`, `--api-base`, `-w/--workspace`, `-v/--verbose`, `--no-color`.
- `Command::{Run(RunCliArgs), Chat(ChatCliArgs), Lsp(lsp::LspCli), Cron(cron::CronCli), Team(team::TeamCli), Watch(watch::WatchCli), Sessions(sessions::SessionsCli), AutoHarness(AutoHarnessArgs)}`.
- `RunCliArgs { base: RunArgs, worktree: Option<String>, resume: Option<String>, continue_: bool }` and `ChatCliArgs { base: ChatArgs, watch: bool, worktree: Option<String>, resume: Option<String>, continue_: bool }`: the harness types are flattened verbatim, the extra fields are this crate's (`--worktree` takes `value_name = "NAME"`, `--resume` takes `value_name = "ID"`, and the `continue_` field parses `--continue`).
- `--resume` and `--continue` are `conflicts_with` each other (the clap ids are `resume` and `continue_`), so the pair is a usage error (exit `2`).
- `Cli::to_harness()` forwards every global (including `--no-color`) and maps `Run`/`Chat` onto the harness `Command`.
- `to_harness` drops `--worktree`/`--watch`/`--resume`/`--continue`; `Lsp`, `Cron`, `Team`, `Watch`, `Sessions`, `AutoHarness` and "no subcommand" become `command: None`, so the harness only reads the globals there.
- `cli::run` dispatches `auto-harness`, `lsp`, `cron`, `team`, `watch` and `sessions` to their modules, `run`/`chat` to `extras::run_run`/`extras::run_chat`, and "no subcommand" to `openjiuwen_harness::cli::run_cli`.
- Nine unit tests guard the wiring: `--no-color` before and after a subcommand, global forwarding plus `run`/`chat` mapping, `--worktree` and `chat --watch` staying in this crate, every harness-surface command mapping to `command: None`, "no subcommand" staying `None`, and the drift checks that every harness global flag and subcommand exists in this grammar.
- The resume flags are asserted where they are resolved: the catalogue's own inline tests, not the grammar tests.

### `auto-harness` (`src/auto_harness/`)

- `AutoHarnessArgs { json: bool (group-global), command: AutoHarnessCommand }`; `--json` is a Rust addition that switches every text command to JSON.
- `AutoHarnessCommand::{Run, Experience, VerifyExt, GapAnalyze, History}`.
- `run`: flags `--task <TEXT>`, `--task-file <PATH>`, `--dry-run`, `--stage <assess|plan|implement|verify>`, `--no-push` (parity only, no effect), `--budget <SECS>`, `--goal <TEXT>`, `--pipeline <meta|extended|auto>` (absent means `meta`).
- `run` loads `<data-dir>/config.yaml` via `AutoHarnessConfig::load_or_default`, applies the overrides, validates, resolves the tasks, and either writes/echoes the `--dry-run` list or returns `Error::Unsupported` naming the missing harness evaluation runner.
- `--budget` also lowers `task_timeout_secs` to `min(current, budget * 0.95)`, `--pipeline`'s absence means `meta`, and `--no-push` reaches only a `tracing::debug!`.
- `--dry-run` writes `<data-dir>/runs/tasks.json`, prints the same pretty JSON array on stdout, and prints `[dry-run] task list -> <path>` on stderr.
- `--task` wins over `--task-file`; a task file may be a JSON array or a single object, needs `topic` per entry, defaults `description` to `""` and `files` to `[]`, and ignores unknown keys.
- `experience search <QUERY>` searches a lowercased `topic`/`summary`/`details` blob (`outcome` is never searched), drops query words shorter than two characters, scores hits plus a 30-day linear recency bonus, and prints the best ten.
- `experience list [--type <TYPE>] [--limit <N>]` (default limit 10) and `history [--limit <N>]` (default 20) sort newest-first, truncate to the limit, and only then apply the exact, case-sensitive type filter.
- Both print `[<type>] <topic>: <summary or outcome>` per row, the documented `no experience records` / `no matching experiences` message when empty, and a pretty JSON array with `--json`.
- `verify-ext [--ext-path <DIR>]` always returns `Error::Unsupported` from `openjiuwen_rsi::deferred::RuntimeExtensionLoader` and does not validate the path.
- `gap-analyze` prints the placeholder line, or with `--json` the object `{"gaps": [], "placeholder": true, "note": …}`, and exits `0`.
- `data_dir` reproduces Python's asymmetry: `<-w workspace>/auto_harness` when `-w` was given, otherwise `<home>/auto_harness`; the store commands always read `<resolved workspace>/auto_harness/experience/experiences.jsonl`.

### `lsp` (`src/lsp.rs`)

- `LspCli { action: LspActionArg, file: Option<PathBuf>, line: Option<u32>, column: Option<u32> (alias `--col`), query: Option<String>, context_lines: Option<u32>, server: Option<String>, server_command: Option<String>, server_args: Vec<String>, language_id: Option<String>, json: bool }`.
- `LspActionArg` is a kebab-case `ValueEnum` over the tool's eleven actions (`hover`, `definition`, `references`, `completion`, `symbols`, `workspace-symbols`, `diagnostics`, `implementation`, `prepare-call-hierarchy`, `incoming-calls`, `outgoing-calls`), mapped one-to-one by `action()`.
- `run` resolves `CliOptions`, pre-validates `--file` (`tokio::fs::metadata`; a miss is `Error::not_found`, "file not found: <path>"), builds the `LspToolConfig` and invokes the harness `LspTool` exactly as the model would.
- Server configuration: `--server-command` wins over `--server`, which wins over all built-ins; without either, `with_builtin_servers()` uses the harness table (`pyright`, `rust`, `typescript`, `gopls`, `jdtls`), whose programs are never probed against `PATH`.
- An unknown `--server` name is `Error::validation` naming those five names; `--language-id` becomes a server field with `--server-command` and a tool-level hint otherwise.
- `LspArgs.path` is the spelling the user typed (`--file` as text); the tool resolves it against the configured workspace root.
- `shutdown()` runs on both the success and the failure path, so no language-server child outlives the command.
- Rendering: `--json` prints the whole payload document (pretty); text mode prints `data.result` when it is a non-blank string, else the pretty `data` JSON.
- A payload with `success: false` becomes `Error::tool(payload.error)` — `error: tool: …`, exit `1`, empty stdout.

### `cron` (`src/cron.rs`)

- `CronCli { json: bool (global), command: CronCommand }`; `CronCommand::{List, Status, Get, Create, Preview, Toggle, Run, Runs, Cancel}`.
- `Create { cron, prompt (required `--prompt`), description (optional `--description`) }`, `Preview { job_id, count (default 5, clamped to 1..=20) }`, `Runs { job_id (optional `--job-id`), limit (default 10, `0` = all) }`, `Get`/`Toggle`/`Run`/`Cancel` take a positional job id.
- `run` resolves `CliOptions`, creates the workspace directory, builds `CronScheduler::for_workspace(&Workspace::new(workspace))` and takes one `chrono::Utc::now()` per invocation; the OS scheduler files are `<workspace>/cron/jobs.json` and `<workspace>/cron/runs.json`.
- Text output: `list`/`get` print `<id>\t<cron>\tnext <ts>\t<enabled|disabled>` (or `no cron jobs`); `status` prints `N cron job(s): N enabled, N disabled, N due; N run record(s)` plus `next run at <ts>` when an enabled job is pending; `create` prints `created <job line>`; `toggle` prints the state word then the job line; `preview` prints one occurrence per line (or `no upcoming occurrences`); `run` prints `ran <id>; next run at <ts>`; `runs` prints `<recorded>\t<job-id>\t<manual|scheduled>\toccurrence <ts>` (or `no cron runs`); `cancel` prints `cancelled <id>`.
- `--json` prints one pretty document: the job array (`list`), the `CronStatus` object (`status`), one `CronJob` (`get`/`create`/`toggle`/`cancel`), one `CronRun` (`run`), the occurrence strings (`preview`), the run array (`runs`).
- Timestamps use the cron tool's own style, `to_rfc3339_opts(SecondsFormat::AutoSi, true)`.
- An unknown id is `Error::not_found` with `cron job not found: <id>` from every id-taking command.

### `team` (`src/team/mod.rs`)

- `TeamCli { socket: Option<PathBuf> (global, `value_name = "PATH"`), json: bool (global), command: TeamCommand }`; both flags may therefore follow the subcommand.
- `TeamCommand::{Members, Member, Inbox, Send, Broadcast, CreateTask, Task, Claim, Complete, Update, Snapshot, Stop}`; `CreateTask { title, content, task_id (optional `--task-id`), dependencies (repeatable `--dep`) }`, `Update { task_id, title, content }`, and `Task { command: TeamTaskCommand::{List { status }, Claimable, Get { task_id } } }`.
- `run` resolves the socket (`--socket`, else `OPENJIUWEN_TEAM_SOCKET` from `TEAM_SOCKET_ENV`, else `Error::validation` naming both sources), dials it with `AttachClient::connect`, and performs exactly one operation — no reconnect and no retry.
- `AttachError::into_error` preserves the wire kind, so a remote `not_found` reaches the exit path as `error: not_found: …` instead of a transport failure.
- `task get` on a host that answers with no task detail is this crate's own `Error::not_found` (`task not found: <id>`), not an attach error.
- `--status` is parsed by `parse_status` into the schema `TaskStatus` after the connection but before any request; an unknown value is `Error::validation` naming the seven statuses.
- Text rows: members/member are `<name>\t<role>\t<status>\t<execution>` plus a fifth display-name column when it differs (empty roster: `no team members`); inbox is the host text verbatim or `inbox is empty`; send/broadcast are `<message_id> from <from> to <to|*>`; create-task/update are `<task_id>\t<status>\t<title>`; task list/claimable are `<task_id>\t<status>\t<title>[\t<assignee>]` with `no tasks` / `no claimable tasks`; task get prints `task_id`, `title`, `status`, `assignee` (`-` when unowned) rows then the content; claim/complete print `claimed <id>` / `completed <id>`; snapshot prints `team <name> (<state>): N member(s), M task(s)` followed by the member and task rows; stop prints `stop requested`.
- A payload with `ok: false` is `Error::tool` carrying the host's `reason` in both output modes — never a successful exit with JSON on stdout; a payload without `ok` (or without the fields a row needs) is printed as compact JSON.
- `--json` prints one pretty document: member array, `MemberSnapshot`, inbox text, operation payload, task array, `TaskDetail`, or the whole `TeamSnapshot`.

### `watch` (`src/watch.rs`)

- `WatchCli { json: bool, list: bool, debounce_ms: u64 (default 250), max_events: Option<u64> }`.
- `run` resolves `CliOptions` and `WatchPlan::workspace(&workspace, &home, &ProcessEnv).existing_only()`; `--list` renders that plan and returns without starting a watcher.
- Text plan lines are `<kind>\t<path>` with ` (recursive)` for directory targets, and `skipped\t<path>` for every candidate root that does not exist; `--json` prints `{"targets": [{kind, path, label, recursive}], "skipped": [...]}`.
- Watching registers the instruction, MCP-config and skill-library reload handlers, starts `WorkspaceWatcher`, subscribes, announces `watching N target(s)` on stderr (flushed, a readiness signal), then selects on the subscription and one `ctrl_c` future until the stream closes, Ctrl-C, or `--max-events`.
- Text updates: `event <kind> <change> <path>` (with ` <- <previous>` after a rename), `reload failed [<handler>] <path>: <message>`, `dropped <n> <stage>`, `watcher error: <message>`; `--json` prints the `WatchUpdate` serde form, tagged `{"type": "event"|"reload_failed"|"dropped"|"watcher_failed"}`.
- `WatchSink`/`WatchSpec`/`WatchSession`/`start` are the non-blocking session `chat --watch` uses: text lines on stderr, no event limit, stopped by `WatchSession::stop`.
- The handle is shut down before `run` returns, so nothing is printed after the last rendered line.
- An empty plan is a typed `Error::config` ("… requires at least one watch target") from the backend instead of a permanently idle process.

### `sessions` (`src/sessions/`)

- `SessionsCli { data_dir: Option<PathBuf> (global, `value_name = "DIR"`), json: bool (global), command: SessionsCommand }`; `--data-dir` replaces the default `<workspace>/persistent_agent` base with `PersistentAgentLayout::new(dir)`.
- `SessionsCommand::{List { kind: Option<KindArg>, limit: usize (default 20, 0 = all) }, Show { id }, Prune { older_than: Option<f64> (`--older-than DAYS`, conflicts with `--all`), all: bool, dry_run: bool }}`.
- `KindArg` is a kebab-case `ValueEnum` over `cli`, `execution` and `state`; it is serde-free because the rows carry the tag string.
- `catalog::Catalog::for_options(&CliOptions, data_dir)` is the one value holding `OpenJiuwenHome` plus `PersistentAgentLayout`; it never creates a directory, so `sessions list` works on a home and workspace that were never materialised (`CliOptions::ensure_workspace` is never called here).
- `Catalog::list` reads the three sources into `Row { id, kind, created_at, detail, data, sources }` (`data` and `sources` are `#[serde(skip)]`, so the JSON shape stays `{id, kind, created_at, detail}`): the CLI records of `<home>/sessions` (`<n> messages, model <model>`), the journals of `ExecutionCatalog` (`<entries> entries, session <id>, usable <bool>`), and the records of `FileSessionStore` (`agent <id>, waiting <bool>, ending <bool>`, `created_at` from the `state.json` mtime).
- One `run` writes to all three stores, so the rows are then folded by `(agent id, session id)` — CLI records under the local `CLI_AGENT_ID = "openjiuwen-cli"` (it mirrors the harness's private pinned card id, which `openjiuwen_harness::cli` does not export), a session record under `binding.agent_id`/`binding.session_id`, a journal under `agent_id` and its `session_id` (or its own `execution_id` when that is empty, so session-less executions keep their identity) — into **one row per session**.
- The kept member is the richest source (`state` 3 > `cli` 2 > `execution` 1), then the newest, then the smallest id; its own `created_at` is kept and a timestamp-less row inherits the newest timestamp of the group, so ordering and `--older-than` still work. Every other member goes to `Row::sources`, richest first then by id.
- `Row::has_kind(kind)` matches the kept `data` **or** any source, so `--kind cli`/`--kind execution` still find a session whose row is reported from `state`; `Row::is_resumable()` is false only when every source is an `Execution` with `usable == false`.
- Ordering is newest-first by parsed RFC 3339 `created_at`, rows without a timestamp last, ties by id ascending; `--kind` filters before `--limit` truncates, and one row is one session.
- `Catalog::resolve` takes `ResumeRequest::{Id(&str), Continue}` and yields `Resolved { id, kind, summary, history, state, plan, record }`; both resolve through the same precedence — a persistent state record wins over a CLI record, which wins over an execution journal — and differ only in how the id is chosen.
- A state record resumes through `openjiuwen_harness::persistent_agent::resume` when its `ending`/`checkpoint`/`continuation` names an execution, seeding `plan.seed_history()` plus the plan's `deepagent_state` object; without an execution ref it seeds `record.messages` and leaves `plan: None`.
- A CLI record maps `user`/`assistant`/`system` messages onto `BaseMessage` (other roles are skipped) and keeps the `SessionRecord`.
- An execution journal is opened, read, and reassembled with `WindowAssembler::new(Capacity::default()).assemble(&snapshot.entries)`; the resumed id is the journal's `session_id`, falling back to the execution id.
- `ResumeRequest::Continue` takes the first resumable row of `list()` (newest-first, so ties break by id) and resolves that row's id through the documented precedence.
- The summary strings are fixed: `resuming session <id> from the CLI session record (<n> messages)`, `… from the persistent session record via execution <exec> (<n> messages restored)`, `… from the persistent session record (<n> messages restored)`, `… from execution <exec> (<n> messages restored)`; `Continue` says `resuming the most recent session <id>` instead of `resuming session <id>`.
- `Catalog::show` finds the kept row by `id` alone (after the merge an id can resolve from a different store than the row was reported from, e.g. a journal reached through its own id), falling back to a synthesized row for an id that vanished between the two scans.
- `Catalog::prune` selects by `--all` or by age (`created_at < now - days`, timestamp-less rows only via `--all`), deletes **every** source of each selected session (the CLI file, each journal directory, the `state.json` directory) so a pruned session cannot reappear from a leftover store, and without either selector is `Error::validation("prune needs --older-than <DAYS> or --all")`.
- `list` text is `<id>\t<created_at or ->\t<kind>\t<detail>` (or `no sessions`); `show` text is `session <id>`, `kind <kind>`, `created <ts or ->`, the detail, `messages <n>`, one `<role>: <content>` line per seeded message (newlines flattened), plus `plan execution <ref or -> resume-seq <n>` when a plan resolved; `prune` text is `would prune <kind>\t<id>` / `pruned <kind>\t<id>` / `nothing to prune`.
- `--json` prints one pretty document: the row array, the `show` object (`id`, `kind`, `created_at`, `detail`, `messages: [{role, content}]`, `plan`, `record`), or `{"dry_run", "pruned", "entries"}`.
- A window rebuilt from a journal renders its `execution_started` entry as `【本次执行】<text>`, so the first restored message carries that prefix in `show` and in the resumed turn.

### `extras` (`src/extras.rs`)

- `run_run(args, base)` applies `--worktree` and hands the resulting harness `Cli` to `run_cli`; with `--resume`/`--continue` it instead resolves the catalogue, prints the one-line summary on stderr, and runs `run_once_resumed(&options, &args.base, &CliIo::stdio_non_interactive(), ResumedTurn::new(id).with_history(history) [.with_state(state)])`, mirroring `run_cli`'s `Dispatch::Run` branch (uncoloured and non-interactive, so permission prompts fail closed).
- `run_chat(args, base)` applies `--worktree`, then `--watch` around one `chat_entry` call: start the stderr watch session, run the REPL (fresh or resumed), stop the session; the REPL's own error wins over a stop failure.
- The resumed chat path mirrors `run_cli`'s chat branch (`CliOptions::resolve`, `paths::color_enabled`, `CliIo::stdio().with_color`, `onboarding::maybe_run`, `build_model`, `build_agent`), then seeds `Repl::seed_history`, prints `<summary>; the REPL continues as <new id>` on stderr, and runs the REPL.
- `prepare_worktree` resolves `CliOptions`, ensures the workspace, discovers the repository with `find_git_root`/`ProcessGitRunner`, and creates (or reuses) the worktree through `WorktreeManager::new(repo)`.
- The worktree lives at `<repo-root>/.worktrees/<slug>` on branch `worktree-<slug>`; the announcement `worktree: <slug> -> <path> (branch <branch>)` goes to stderr and the returned harness `Cli` points `-w` at the worktree path.
- A workspace outside a git repository (or a host without git) is a typed `Error::config` before any model is built.

### Exit codes

- `0` success, `--help`/`--version`, and a `watch` that stopped at `--max-events`.
- `1` a runtime failure from any command, printed by `main` as `error: <kind>: <cause>` on stderr (for example `error: unsupported: …`, `error: not_found: cron job not found: …`).
- `2` usage errors from clap's `parse()`: unknown subcommand or flag, missing value, invalid enum value (`lsp bogus`, `--pipeline bogus`), and `--resume` together with `--continue`.
- stdout is the result stream; `run -f json`/`-f stream-json`, `auto-harness --json`, `lsp --json`, `cron --json`, `team --json`, `watch --json`, `watch --list`, `sessions list|show|prune --json` and the dry-run task array write there.
- Human notices go to stderr: logs, permission prompts, the `error:` diagnostic, the `[dry-run]` note, the `worktree:` announcement, the `resuming …` note of `--resume`/`--continue`, `watching N target(s)` and the `chat --watch` reload notices.

## Constraints

- Only the harness defines `run`/`chat` behaviour; this crate must not re-implement parsing, option precedence, rendering or exit-code classification it already owns.
- `lsp`, `cron`, `team`, `watch` and `sessions` are translations, not re-implementations: the harness owns the LSP client and server table, the cron scheduler and its JSON types, the attach protocol and its schema, the watcher with its reload handlers, and the persistent-agent layout, catalogues and `resume` plan.
- The session catalogue is read-only apart from `prune`: it never creates a directory and never calls `CliOptions::ensure_workspace` (resume does, because it is about to run).
- Resume never re-implements the harness: `run --resume` goes through `run_once_resumed` and `chat --resume` through `Repl::seed_history`, so rendering and session-record writes stay in the harness.
- The duplicated globals are held in sync by the drift unit tests in `src/cli.rs` against `openjiuwen_harness::cli::Cli::command()`, not by convention.
- stdout stays machine-readable in `--json`/`stream-json` modes; logs, permission prompts, the `error:` diagnostic, the `[dry-run]` note, the `worktree:` line, the `resuming …` note, the `watching` readiness line and the `chat --watch` notices go to stderr.
- No panics on user input — parsing is delegated to clap and command functions return `openjiuwen_common::error::Error`; `unwrap`/`expect`/indexing on input are forbidden in `src/`.
- `println!`/`eprintln!` are allowed here (binary crate) for user-facing results and diagnostics only; libraries log through `tracing`.
- No new crates.io dependencies: these surfaces need only `chrono` (the cron clock and timestamp rendering) and the `openjiuwen-harness` `notify` feature (the watcher), both already in the manifest; the crate builds offline with the workspace-pinned set, and `openjiuwen-test-support` is a dev-dependency only.
- Declared dependencies: `openjiuwen` (the umbrella, for the `agent_teams` attach types), `openjiuwen-harness`, `openjiuwen-core` (`foundation::tool` in `lsp`), `openjiuwen-common`, `openjiuwen-rsi`, `clap`, `chrono`, `tokio`, `tracing`, `tracing-subscriber`, `dotenvy`, `serde` (only `auto_harness/experience.rs`) and `serde_json`; `anyhow` and `openjiuwen-harness-protocol` are declared but referenced by no source file.
- Files stay around ≤1000 lines (`src/sessions/catalog.rs` is the largest at ~864 lines); sibling binaries are not added here.
- The `[[bin]]` target declares `doc = false` so rustdoc never collides with the `openjiuwen` umbrella crate's `target/doc/openjiuwen/`.
- `team` is Unix-only at runtime: the grammar and wire types compile everywhere, but the attach transport answers `Error::Unsupported` ("cross-process attach requires Unix domain sockets") on non-Unix targets.

## Gotchas

- `-h` and `--help` differ: `-h` prints the short `about`, `--help` prints the long help synthesized from the `Cli` struct doc comment; assert on `Usage:`, `Commands:` and row names instead of the banner.
- `version` is set only on the top-level `Cli` (no `propagate_version`), so `openjiuwen --version`/`-V` prints `openjiuwen <workspace version>` while `openjiuwen run --version` is a usage error (exit `2`).
- `dotenvy::dotenv()` in `main` loads `.env` from the working directory and its ancestors, so every process test must run the child in a temporary cwd (or clear inherited `API_KEY`/`OPENAI_*`) or ambient state leaks in.
- `env!("CARGO_BIN_EXE_openjiuwen")` exists only for integration tests of this package; tests in other crates cannot use it, and this crate has no lib target to depend on.
- `--no-color` is forwarded through `to_harness`, so harness option resolution sees the flag exactly as typed; it is also accepted after a subcommand because clap marks it global.
- clap interleaves globals into every subcommand's help page, so help tests assert on the documented rows and flags rather than on exact layout or ordering.
- `LspArgs.path` is a string resolved against the configured workspace root; the CLI pre-checks the same resolution (absolute stays absolute, a relative `--file` is joined to the workspace) before the tool is built.
- `lsp` rejects a missing `--file` as `not_found` before any server is selected, so a typo cannot launch (or probe for) a language server, and `--json` does not turn that error into a document.
- cron jobs live at `<workspace>/cron` (`jobs.json` for the schedule, `runs.json` for the bounded history); `cron` creates the `-w` workspace first, so a workspace that was never materialised still gets its `cron` directory instead of a store error.
- team attach is Unix-only and needs `--socket` or `OPENJIUWEN_TEAM_SOCKET`; the group flags are global, so `team members --socket <path>` parses, and an unknown `--status` is rejected after the connect but before any request frame is written.
- `watch --list` is the deterministic plan view: it starts no watcher, needs no existing file, and prints the skipped roots instead of dropping them silently.
- `--max-events` exists so scripts and tests can stop the watcher; it counts rendered updates (of any kind) and exits `0`.
- Worktree creation prints to stderr (`worktree: <slug> -> <path> (branch <branch>)`), so `run -f json --worktree …` keeps a clean JSON document on stdout.
- `chat --watch` reload notices are diagnostics only: the running REPL keeps the instructions, skills and MCP configuration it started with, and a restart applies the change.
- `--resume` and `--continue` conflict in clap (the ids are `resume` and `continue_`), so the pair is a usage error (exit `2`) before any store is read.
- `sessions list` never creates the home or the workspace, so it is the safe way to inspect a fresh install; `sessions show` runs the same resolution `--resume` does, so its `messages <n>` is exactly what the resumed turn seeds (a CLI row's detail can count more records than that: `tool` rows are skipped when the transcript is mapped).
- `sessions prune --all` deletes the CLI record, the journal directory or the `state.json` directory of every row it lists, so it is the one destructive command here; `--dry-run` prints the same selection without deleting.
- `--resume`/`--continue` require the resolved workspace to exist (the catalogue reads the workspace's `persistent_agent` node), so a run in a path that does not exist yet fails in `ensure_workspace`, not in the lookup.
- The watcher needs the `openjiuwen-harness` `notify` feature, which this crate enables unconditionally, so `watch` and `chat --watch` are always compiled in.
- The `auto-harness` dry run writes before it prints: a failed write is a `Config` error and nothing reaches stdout.
- A non-dry `auto-harness run` with resolved tasks prints `using N manually specified task(s)` on stdout before the `unsupported` error on stderr (exit `1`).

## Known Issues

- The harness `Cli` has no `auto-harness`, `lsp`, `cron`, `team` or `watch` command, which is why the outer grammar and its globals are duplicated here; adding harness command groups would let this crate drop the copies.
- The harness pins the agent id of a CLI run privately (`CLI_AGENT_ID` in `crates/openjiuwen-harness/src/cli/run.rs`), so the catalogue carries its own `CLI_AGENT_ID` in `src/sessions/catalog.rs` to fold CLI records with the persistent-agent records of the same run; a harness export would let it drop the copy.
- `openjiuwen-rsi` ships no harness evaluation runner and no experience store, so a non-dry `auto-harness run` and `verify-ext` return typed `Error::Unsupported` and the experience commands read the documented JSONL directly instead of a Rust store.
- `chat --resume` does not adopt the persisted session id: `run_chat_resumed` builds the REPL with `Repl::with_io_and_signals` and never calls the available `Repl::with_session_id`, so the restored transcript is seeded into a fresh `cli-*` session and the stderr note names both ids. `run --resume` does continue the id (`run_once_resumed` creates a kernel session with it and appends to that record).
- Nothing in this binary writes persistent-agent journals, session records or links — `src/cli/resume.rs` and the whole `cli` module of the harness keep `persistent_agent` out of the CLI path — so `sessions`/`--resume` only discover state that another host persisted; `tests/resume.rs` seeds those stores through the harness API to exercise them.
- The resume summary uses `openjiuwen_harness::cli::session_id()` as the request id for `resume`, so a state-record resume is a *new* request id rather than the persisted one; the plan settles only its own waiting scene.
- `lsp` needs an installed language server: the built-in table is not probed against `PATH`, so a missing program surfaces as a typed launch failure (`error: tool: …`) on first use rather than a fallback to another server.
- `team`'s `invoke_tool` op is not exposed as a command: the client supports it, but the in-repo attach host always answers `unsupported`, so a command would only surface that error.
- `openjiuwen-e2e/tests/cli_binary.rs` (outside this node) builds the harness `Cli` as a struct literal including `no_color`, so any new harness global must be added there as well; the crate currently compiles.
- Deferred harness features reachable from these commands (browser/CUA/mobile presets, MCP resource tools needing a host backend, OTel export) are documented in `../openjiuwen-harness/` and `../../docs/PARITY.md`, not here.

## Routing Table

- `./src/main.rs` → process boundary: startup order, logging installation, clap parse, `cli::run` dispatch, exit-code mapping.
- `./src/cli.rs` → outer grammar (`Cli`, `GlobalArgs`, `Command`, `RunCliArgs`, `ChatCliArgs`), `to_harness`, dispatch into the harness or the binary's own commands, and the grammar/drift unit tests.
- `./src/extras.rs` → `run`/`chat` wiring for `--worktree` (worktree creation and workspace redirection), `chat --watch`, and `--resume`/`--continue` (the resumed run and the seeded REPL).
- `./src/sessions/mod.rs` → `sessions`: the grammar, dispatch, and the list/show/prune text and JSON renderings.
- `./src/sessions/catalog.rs` → the catalogue behind both `sessions` and `--resume`: the three-store read, the one-row-per-session fold and ordering, id/`Continue` resolution, the resume-plan probing, and prune selection/deletion of every source.
- `./src/sessions/catalog/tests.rs` → the catalogue's own tests: row ordering, the fold (richness, timestamp inheritance, session-less execution keys, sources, `has_kind`, `is_resumable`), role mapping, the resume-note prefixes, and three async cases against real stores.
- `./src/lsp.rs` → `lsp`: action enum, server selection, `--file` pre-check, and payload rendering.
- `./src/cron.rs` → `cron`: the `CronScheduler` call per subcommand and its text/JSON output.
- `./src/team/mod.rs` → `team`: socket resolution, the attach client calls, and the roster/task/message renderings.
- `./src/watch.rs` → `watch` and the reusable watch session behind `chat --watch` (plan rendering, readiness line, update rendering, `--max-events`).
- `./src/auto_harness/mod.rs` → `AutoHarnessArgs`/`AutoHarnessCommand`, the dispatch, `data_dir`.
- `./src/auto_harness/run.rs` → `auto-harness run`: config loading and overrides, task resolution, the dry run, the deferred session error.
- `./src/auto_harness/experience.rs` → the read-only JSONL store behind `experience search`, `experience list` and `history` (scoring, ordering, type filter, output forms).
- `./src/auto_harness/deferred.rs` → `verify-ext` (typed deferred loader) and the `gap-analyze` placeholder.
- `./tests/support/mod.rs` → the shared `Sandbox` (temp cwd/home/workspace, ambient-env scrubbing, bounded spawn) and the exit/JSON/ANSI assertion helpers.
- `./tests/cli_smoke.rs` → process-level grammar and help suite (`--version`, every help page including `lsp`/`cron`/`team`/`watch`, `--no-color` placement, unknown subcommand).
- `./tests/cli_run.rs` → offline scripted `run` through the real process against the loopback mock model server, in all three output formats.
- `./tests/cli_features.rs` → `run --worktree`, the non-repository failure, and `chat --watch` through the real process.
- `./tests/cron.rs` → process-level `cron` suite: store, mutations, reads, and the typed failures.
- `./tests/sessions.rs` → process-level `sessions` suite: hand-written CLI records and a damaged journal tree behind the empty listing, ordering, `--kind`/`--limit`, `show`, prune, the typed failures and the help pages.
- `./tests/resume.rs` → process-level `run --resume`/`--continue`/`chat --resume` suite against the mock model, with persistent-agent journals and state records seeded through the harness API.
- `./tests/lsp.rs` → process-level `lsp` suite: help, unknown action, missing file, failing/unspawnable server programs.
- `./tests/team.rs` → process-level `team` suite against a stub attach host on a Unix socket (grammar, wire arguments, renderings, failures).
- `./tests/watch.rs` → process-level `watch` suite: `--list` text/JSON, the empty plan, and a live event with `--max-events`.
- `./tests/auto_harness.rs` → process-level `auto-harness` suite: task resolution, dry run, deferred surfaces, experience store, history, gap-analyze.
- `./README.md` → user-facing build/run instructions, the full command reference, exit codes, environment variables, data layout, and the Rust-port differences.
- `../openjiuwen-harness/src/cli/` → all `run`/`chat` CLI logic (`CliOptions`, `run_once`, `Repl`, permissions, session store, `RunOutcome::to_json`).
- `../openjiuwen-harness/src/lsp/` → the LSP client, server table, actions and payload rendering `src/lsp.rs` drives.
- `../openjiuwen-harness/src/tools/cron/` → `CronJob`, `CronStatus`, `CronRun`, `CronSchedule` and `CronScheduler` behind `src/cron.rs`.
- `../openjiuwen-agent-teams/src/attach/` → `AttachClient`, the request/response frames and `AttachError` behind `src/team/mod.rs`.
- `../openjiuwen-harness/src/watcher/` → `WatchPlan`, `WorkspaceWatcher`, the reload handlers and `WatchUpdate` behind `src/watch.rs`.
- `../openjiuwen-harness/src/worktree/` → `WorktreeManager`, `find_git_root` and the registry `--worktree` uses.
- `../openjiuwen-rsi/src/harness_rsi/auto_harness/` → `AutoHarnessConfig`, `OptimizationTask`, `PipelinePreference` and the `deferred::RuntimeExtensionLoader` this crate calls.

## Test Strategy

- 95 process tests in 10 binaries (`auto_harness` 20, `cli_features` 3, `cli_run` 5, `cli_smoke` 13, `cron` 9, `lsp` 6, `resume` 7, `sessions` 10, `team` 18, `watch` 4) plus 56 in-source unit tests (`cli.rs` 9, `team/mod.rs` 7, `sessions/catalog/tests.rs` 16, `sessions/mod.rs` 1, `auto_harness/run.rs` 10, `auto_harness/experience.rs` 10, `auto_harness/deferred.rs` 3) — 151 total.
- `main.rs`, `extras.rs`, `cron.rs`, `lsp.rs`, `watch.rs`, `run.rs` and `chat.rs` have no unit tests; only the process suites cover them.
- `tests/cli_smoke.rs` (13 tests) covers the clap layer only — `--version`, the eight command rows (`run`, `chat`, `lsp`, `cron`, `team`, `watch`, `sessions`, `auto-harness`), `auto-harness`'s five subcommands and its `run`/`experience` flags, `lsp`'s eleven actions, `cron`'s nine subcommands, `team`'s twelve, `watch`'s four flags, `run --worktree`/`--resume`/`--continue` and `chat --watch`/`--worktree`/`--resume` in their help, `--no-color` before and after a subcommand, and the unknown-subcommand usage error (exit `2`, diagnostic on stderr, empty stdout).
- `tests/cli_run.rs` (5 tests) drives a real `run` through `MockLlmServer`: a reusable scripted fallback answers any number of agent-loop calls, the connection is passed as flags (`--provider openai --api-base <mock> --api-key … -m mock-model`), and the cases assert the `json` object keys (`answer`, `result_type`, `usage`, `session_id`), the `text` stream, one JSON object per `stream-json` line with a terminal `{"type":"result"}` line, escape-free stdout, `--no-color` through a real run, and `chat --help`.
- `tests/auto_harness.rs` (20 tests) pins task resolution (`--task`, array/object task files, unknown keys, missing `topic`, `--task` precedence, the empty list), the dry run and its `runs/tasks.json`, the deferred non-dry run (exit exactly `1`, `unsupported` + `harness evaluation runner`), `--stage` in the message, `--pipeline bogus` as exit `2`, the seeded experience store (ordering, malformed line skipped, limit before type filter, search fields, short tokens, `--json`, empty-store messages), `history` (limit and JSON), and `gap-analyze` text/JSON.
- `tests/cli_features.rs` (3 tests) covers `run --worktree` end to end (worktree at `<workspace>/.worktrees/demo`, the `worktree: demo -> … (branch worktree-demo)` stderr line, the scripted answer, escape-free stdout), `--worktree` outside a repository as `error: config:` naming the repository with nothing created, and `chat --watch` (readiness line on stderr, EOF on closed stdin, exit `0`); the git cases skip when `git --version` cannot run.
- `tests/cron.rs` (9 tests) pins the empty store, `create` persisting `<workspace>/cron/jobs.json` with the job fields, `list --json`, the `get` row, `toggle` both ways, `preview --count` (RFC 3339, strictly increasing), `run` plus the manual `runs` record, `status` counts and `next run at`, `cancel`, and the `not_found`/`validation` failures (exit `1`, empty stdout).
- `tests/lsp.rs` (6 tests) pins the action/flag help, an unknown action as exit `2` with empty stdout, a missing `--file` as `error: not_found:` naming the path (also under `--json`), `/bin/false` as `error: tool:` (Unix only), and an unspawnable `--server-command` as `error: tool:`.
- `tests/team.rs` (18 tests, `#![cfg(unix)]`) drives the real binary against a stub attach host: the grammar, exactly one request per command with the documented args (`list_tasks` status, `task_detail` id, `send_member`, `create_task`, `complete_task`), the member/task/message/outcome renderings and empty markers, one pretty JSON document per command, a remote `not_found`, an unknown `--status` reaching no request, and the missing-socket validation error.
- `tests/watch.rs` (4 tests) pins `--list` text (instruction/MCP/skill targets plus `skipped` rows) and JSON (`targets`/`skipped`), an empty plan as `error: config:` naming the missing targets, and a live change after the readiness line reported as exactly one JSON `event` with `--max-events 1` exiting `0`.
- `tests/sessions.rs` (10 tests) drives the catalogue through the real process on hand-written `<home>/sessions/<id>.json` records, a deliberately damaged `persistent_agent/logs` tree and a harness-seeded state record: the empty listing in text and JSON, newest-first rows with `--limit 1` and `--kind execution`, `show` text (header lines, `messages 2`, one `<role>: <content>` line per message) and `show --json` (`messages` order, `record.session_id`, null `plan`), a malformed `meta.json` plus a metadata-less directory being skipped instead of failing the listing, the unknown-id `error: not_found: unknown session:` (exit `1`, empty stdout), `prune --all --dry-run` reporting newest-first and deleting nothing, `prune --all` then `no sessions`, the `{dry_run, pruned, entries}` JSON, `--older-than 7` on a record dated now as `nothing to prune`, the selector-less `validation` error, `--all` with `--older-than` as clap exit `2`, and the `sessions`/`list`/`prune`/`show` help pages.
- `tests/sessions.rs`'s two dedupe cases: `a_session_in_all_three_stores_is_listed_and_pruned_once` writes one session id into the CLI store, a harness-seeded journal and a harness-seeded `state.json` under agent `openjiuwen-cli` and asserts one text line and one JSON row (`kind: "state"`), `--kind cli`/`execution`/`state` each matching it exactly once, a `show --json` with non-empty `messages` carrying the journalled question, a one-entry `prune --all --dry-run`, and a `prune --all` that leaves `no sessions` with the journal directory, the `state.json` directory and the CLI record file all absent; `a_real_run_lists_one_session_and_continue_continues_it` runs the binary against `MockLlmServer` (`run "first question" -f json`), then asserts exactly one row whose id and `state` kind match the run's `session_id`, a `show` with non-empty `messages`, a `run --continue` that reports the same id with `resuming the most recent session <id>` on stderr, and still exactly one row afterwards.
- `tests/resume.rs` (7 tests) covers the resume surface against the loopback mock model server and persistent-agent fixtures seeded with the harness API in the test process: `run --resume <cli-id>` (note on stderr, one JSON object on stdout, the record growing 2 → 4 messages, the restored question before the new one in the recorded model request), `--continue` twice (the `resuming the most recent session cli-…` note and a 6-message record), the unknown-id `error: not_found: unknown session:` and the empty-catalogue `no resumable session found` (both exit `1` with empty stdout), a state record resuming `via execution exec-1` with a non-null `plan` in `sessions show --json`, `--resume exec-1` reporting `from execution exec-1 (2 messages restored)` and leaving a `cli` row for the session it continues, and `chat --resume` printing the note plus `the REPL continues as cli-` on EOF.
- The `tests/resume.rs` ordering assertions match the restored text as a substring, because a window rebuilt from a journal renders `execution_started` as `【本次执行】<text>`.
- The `sessions` module's inline tests cover the pure helpers (sort order with mixed timestamps, role mapping, resume-note prefixes, kind tags) plus three async tests against real stores in a temp home/workspace: a persistent state record listing/resolving/pruning, a CLI record resolving with its transcript, and the typed `not_found`/`validation` failures.
- Hermeticity is enforced by `tests/support/mod.rs`: every child gets a fresh temp cwd, a temp `OPENJIUWEN_HOME` and a temp `-w` workspace; ambient `OPENJIUWEN_*`/`OPENAI_*`/`API_KEY`/`API_BASE`/`MODEL_*`/`NO_COLOR`/`RUST_LOG` are removed; stdin is `/dev/null`; and every spawn is bounded by a 120 s async timeout that kills the child instead of hanging the run.
- Nothing here needs the network, credentials or a terminal: the mock server binds loopback, the team suite binds its own Unix socket, and the `run`/`cli_features`/`resume` suites are the only ones that build a model.
- In-process CLI behaviour (streaming render, REPL transcripts, permissions, onboarding) is covered by `crates/openjiuwen-harness/tests/cli.rs` and `src/cli/tests.rs`; do not duplicate it here.
- Definition of done: `cargo fmt -p openjiuwen-cli`, `cargo test -p openjiuwen-cli --offline` (repeated to prove the process suites are not flaky), `cargo clippy -p openjiuwen-cli --all-targets --offline -- -D warnings`, `cargo metadata --offline --no-deps`, and a manual `cargo run -p openjiuwen-cli -- --help`.
