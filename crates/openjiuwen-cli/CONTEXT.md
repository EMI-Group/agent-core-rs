# openjiuwen-cli

## Intent

The `openjiuwen` executable: the product entry point a user runs.
It is a thin process boundary that owns the outer clap grammar and dispatches every command group.
`run`/`chat` are forwarded to `openjiuwen_harness::cli::run_cli`; `auto-harness`, `lsp`, `cron`, `team`, `watch` and `sessions` are dispatched here.
The harness-surface commands are terminal ends of ports the harness already owns: its language-server client, its cron scheduler, the `openjiuwen-agent-teams` attach protocol, its workspace watcher, and its persistent-agent stores.
`run`/`chat` additionally accept the flags the harness grammar has no words for (`--worktree`, `--watch` on `chat`, and `--resume <ID>`/`--continue` on both).
`--resume`/`--continue` are resolved by the session catalogue in `src/sessions/catalog.rs`, which folds the CLI session records, the persistent-agent execution journals and the persistent-agent session records into one row per session; both commands then **continue the resolved session id** (`run` through the harness `ResumedTurn` seam, `chat` through `Repl::with_session_id`), and the resume note goes to stderr so stdout keeps the run's output format.
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
- The resume flags are asserted where they are resolved: the catalogue's own inline tests and the process suites, not the grammar tests.

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
- **One row per session.** `Catalog::list` reads the three stores into rows and folds the entries sharing an `(agent id, session id)` pair into one row, reported from its **richest** source (`state` 3 > `cli` 2 > `execution` 1). The grouping key is `Detail::group_key()`: a CLI record has no agent id and is keyed under `CLI_AGENT_ID` (`"openjiuwen-cli"`, the id the harness records CLI runs under), a state record under its binding, and an execution under its `agent_id` plus its `session_id` (or its own execution id when the journal names no session).
- The kept row is the richest member, then the newest, then the smallest id; it keeps its own timestamp and inherits the newest parseable group timestamp when it has none (so ordering and `--older-than` still work). Every other member is kept in `Row::sources` (`#[serde(skip)]`, richest first then by id) and the JSON shape stays `{id, kind, created_at, detail}` (`data` and `sources` are skipped).
- Rows are ordered newest-first by parsed RFC 3339 `created_at`, rows without a timestamp last, ties by id ascending; `--kind` filters before `--limit` truncates.
- `--kind <KIND>` matches a row when **any** source has that kind (`Row::has_kind`), so a CLI-run session reported from `state` is still found by `--kind cli` and by `--kind execution`; `limit` counts sessions.
- Row details: a `cli` row reads `<n> messages, model <model>`, an `execution` row `<entries> entries, session <id>, usable <bool>`, a `state` row `agent <id>, waiting <bool>, ending <bool>`.
- `Catalog::resolve` takes `ResumeRequest::{Id(&str), Continue}` and yields `Resolved { id, kind, summary, history, state, plan, record }`. `Id` looks for a persistent session record, then a CLI record, then an execution journal, all by their own identifier.
- `Continue` takes the first **resumable** row of the merged listing (newest session; a row is unresumable only when every source is an unusable execution journal), then resolves that id through the same precedence — so a CLI-run session continues from its rich state record, not from a bare journal.
- A state record resumes through `openjiuwen_harness::persistent_agent::resume` when its `ending`/`checkpoint`/`continuation` names an execution, seeding `plan.seed_history()` plus the plan's `deepagent_state` object; without an execution ref it seeds `record.messages` and leaves `plan: None`.
- A CLI record maps `user`/`assistant`/`system` messages onto `BaseMessage` (other roles are skipped) and keeps the `SessionRecord`.
- An execution journal is opened, read, and reassembled with `WindowAssembler::new(Capacity::default()).assemble(&snapshot.entries)`; the resumed id is the journal's `session_id`, falling back to the execution id.
- The summary strings are fixed: `resuming session <id> from the CLI session record (<n> messages)`, `… from the persistent session record via execution <exec> (<n> messages restored)`, `… from the persistent session record (<n> messages restored)`, `… from execution <exec> (<n> messages restored)`; `Continue` says `resuming the most recent session <id>` instead of `resuming session <id>`.
- `Catalog::resolve_detail(id)` probes each store by the id it owns — `FileSessionStore::load` for a persistent record, `SessionStore::load` for a CLI record, `ExecutionCatalog::open` for a journal — so `show`, `run --resume` and `--continue` never list the home; `Catalog::list` is the only whole-store read.
- `Catalog::show` uses that one `resolve_detail` pass and builds its row from the entry the resolution read: the entry's kind, its creation timestamp where the store has one (a journal reports `-`, since the reader exposes none without a scan) and its detail line, with the id and kind taken from the resolution (so an execution id reports the session it belongs to).
- `Catalog::prune` selects by `--all` or by age (`created_at < now - days`, timestamp-less rows only via `--all`), then deletes **every source** of each selected session — the CLI record file, each journal directory, each `state.json` directory — and without either selector is `Error::validation("prune needs --older-than <DAYS> or --all")`.
- Deletion is best effort: one unremovable entry never stops the next, and `PruneReport { pruned, failures }` returns every selected row plus one `<kind> <id>: <error>` line per failed deletion; `sessions prune` prints the report first and then returns an `Error::session` naming the failures, so the command exits non-zero instead of claiming success.
- `list` text is `<id>\t<created_at or ->\t<kind>\t<detail>` (or `no sessions`); `show` text is `session <id>`, `kind <kind>`, `created <ts or ->`, the detail, `messages <n>`, one `<role>: <content>` line per seeded message (newlines flattened), plus `plan execution <ref or -> resume-seq <n>` when a plan resolved; `prune` text is `would prune <kind>\t<id>` / `pruned <kind>\t<id>` / `nothing to prune`.
- `--json` prints one pretty document: the row array, the `show` object (`id`, `kind`, `created_at`, `detail`, `messages: [{role, content}]`, `plan`, `record`), or `{"dry_run", "pruned", "entries", "failures"}`.
- A window rebuilt from a journal renders its `execution_started` entry as `【本次执行】<text>` and may open with a `【窗口摘要】…` frame, so the first restored message carries those decorations in `show` and in the resumed turn.

### `extras` (`src/extras.rs`)

- `run_run(args, base)` applies `--worktree` and hands the resulting harness `Cli` to `run_cli`; with `--resume`/`--continue` it resolves the catalogue, prints the one-line summary on stderr, builds the model (`CliOptions::build_model`) and runs `run_with_model_resumed(&options, &args.base, &CliIo::stdio_non_interactive(), model, ResumedTurn::new(id).with_history(history) [.with_state(state)])`, mirroring `run_cli`'s `Dispatch::Run` branch (uncoloured and non-interactive, so permission prompts fail closed). The restored window and the persisted `deepagent` state are seeded into a kernel session with that id, and the CLI record at `<home>/sessions/<id>.json` is continued in place.
- `run_chat(args, base)` applies `--worktree`, then `--watch` around one `chat_entry` call: start the stderr watch session, run the REPL (fresh or resumed), stop the session; the REPL's own error wins over a stop failure.
- The resumed chat path mirrors `run_cli`'s chat branch (`CliOptions::resolve`, `paths::color_enabled`, `CliIo::stdio().with_color`, `onboarding::maybe_run`, `build_model`, `build_agent`), then builds the REPL as `Repl::with_io_and_signals(agent, options, io).with_session_id(id)` **before** `Repl::seed_history(history)` (the setter rebuilds the kernel session, so an earlier seed would be dropped), prints the catalogue summary alone on stderr, and runs the REPL.
- Because the REPL adopts the persisted id, every following turn appends to that session's own CLI record: `chat --resume` no longer mints a `cli-*` session, and the stderr note names one id.
- `prepare_worktree` resolves `CliOptions`, ensures the workspace, discovers the repository with `find_git_root`/`ProcessGitRunner`, and creates (or reuses) the worktree through `WorktreeManager::new(repo)`.
- The worktree lives at `<repo-root>/.worktrees/<slug>` on branch `worktree-<slug>`; the announcement `worktree: <slug> -> <path> (branch <branch>)` goes to stderr and the returned harness `Cli` points `-w` at the worktree path.
- A workspace outside a git repository (or a host without git) is a typed `Error::config` before any model is built.

### Exit codes

- `0` success, `--help`/`--version`, and a `watch` that stopped at `--max-events`.
- `1` a runtime failure from any command, printed by `main` as `error: <kind>: <cause>` on stderr (for example `error: unsupported: …`, `error: not_found: cron job not found: …`, `error: not_found: unknown session: …`).
- `2` usage errors from clap's `parse()`: unknown subcommand or flag, missing value, invalid enum value (`lsp bogus`, `--pipeline bogus`), and `--resume` together with `--continue`.
- stdout is the result stream; `run -f json`/`-f stream-json`, `auto-harness --json`, `lsp --json`, `cron --json`, `team --json`, `watch --json`, `watch --list`, `sessions list|show|prune --json` and the dry-run task array write there.
- Human notices go to stderr: logs, permission prompts, the `error:` diagnostic, the `[dry-run]` note, the `worktree:` announcement, the `resuming …` note of `--resume`/`--continue`, `watching N target(s)` and the `chat --watch` reload notices.

## Constraints

- Only the harness defines `run`/`chat` behaviour; this crate must not re-implement parsing, option precedence, rendering or exit-code classification it already owns.
- `lsp`, `cron`, `team`, `watch` and `sessions` are translations, not re-implementations: the harness owns the LSP client and server table, the cron scheduler and its JSON types, the attach protocol and its schema, the watcher with its reload handlers, and the persistent-agent layout, catalogues and `resume` plan.
- The session catalogue is read-only apart from `prune`: it never creates a directory and never calls `CliOptions::ensure_workspace` (resume does, because it is about to run).
- Resume never re-implements the harness: `run --resume` goes through `run_with_model_resumed`/`ResumedTurn` and `chat --resume` through `Repl::with_session_id` + `Repl::seed_history`, so rendering and session-record writes stay in the harness.
- `sessions` folds the three stores in `src/sessions/catalog.rs` (merge rule, rank, timestamp inheritance) rather than filtering duplicates at render time, so `list`, `show`, `--kind`, `prune` and `--continue` all agree on what one session is.
- `CLI_AGENT_ID` in `src/sessions/catalog.rs` mirrors the harness's `crates/openjiuwen-harness/src/cli/run.rs` constant (private and not re-exported), because a CLI session record carries no agent id; keep the two in sync if the harness id ever changes.
- The duplicated globals are held in sync by the drift unit tests in `src/cli.rs` against `openjiuwen_harness::cli::Cli::command()`, not by convention.
- stdout stays machine-readable in `--json`/`stream-json` modes; logs, permission prompts, the `error:` diagnostic, the `[dry-run]` note, the `worktree:` line, the `resuming …` note, the `watching` readiness line and the `chat --watch` notices go to stderr.
- No panics on user input — parsing is delegated to clap and command functions return `openjiuwen_common::error::Error`; `unwrap`/`expect`/indexing on input are forbidden in `src/`.
- `println!`/`eprintln!` are allowed here (binary crate) for user-facing results and diagnostics only; libraries log through `tracing`.
- No new crates.io dependencies: these surfaces need only `chrono` (the cron clock, the catalogue timestamps) and the `openjiuwen-harness` `notify` feature (the watcher), both already in the manifest; the crate builds offline with the workspace-pinned set, and `openjiuwen-test-support` is a dev-dependency only.
- Declared dependencies: `openjiuwen` (the umbrella, for the `agent_teams` attach types), `openjiuwen-harness`, `openjiuwen-core` (`foundation::tool` in `lsp`), `openjiuwen-common`, `openjiuwen-rsi`, `clap`, `chrono`, `tokio`, `tracing`, `tracing-subscriber`, `dotenvy`, `serde` (only `auto_harness/experience.rs`) and `serde_json`; `anyhow` and `openjiuwen-harness-protocol` are declared but referenced by no source file.
- Files stay around ≤1000 lines (`src/sessions/catalog.rs` is the largest at ~816, with its inline tests split into `src/sessions/catalog/tests.rs`); sibling binaries are not added here.
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
- Worktree creation prints to stderr (`worktree: <slug> -> <path> (branch <branch>)`), so `run -f json --worktree …` keeps a clean JSON document on stdout, and `chat --watch` reload notices are diagnostics only (a restart applies the changed instructions, skills or MCP configuration).
- `--resume` and `--continue` conflict in clap (the ids are `resume` and `continue_`), so the pair is a usage error (exit `2`) before any store is read.
- Every CLI run is journal-backed: `run`/`chat` build the agent with `persistent_agent: true` and the pinned card id `openjiuwen-cli`, so one turn leaves a CLI record, a persistent `state.json` record and an execution journal. Nothing writes an execution link, so a state-record resume's chain trace stops with a stop note (the window still rebuilds from the recovery execution).
- Consequently `run --resume <cli-id>` and `--continue` resolve through the **persistent state record** (`… from the persistent session record via execution exec_… (N messages restored)`), and the restored window opens with the harness's `【窗口摘要】…` frame and `【本次执行】<question>` marker while the CLI record file stays the clean `user`/`assistant` transcript.
- `sessions list` counts sessions, not store entries: after any `run` the same conversation would otherwise appear three times. The `kind` column names the richest kept source (normally `state` for a CLI run), while `--kind cli`/`--kind execution` still find it.
- `sessions prune --all` is the one destructive command here and now deletes **every** source of each listed session (CLI record file, journal directories, `state.json` directories), so a pruned session cannot reappear from a leftover store; `--dry-run` prints the same selection without deleting.
- `sessions list` never creates the home or the workspace, so it is the safe way to inspect a fresh install; `sessions show` runs the same resolution `--resume` does, so its `messages <n>` is exactly what the resumed turn seeds (a CLI row's detail can count more records than that: `tool` rows are skipped when the transcript is mapped).
- `--resume`/`--continue` require the resolved workspace to exist (the catalogue reads the workspace's `persistent_agent` node), so a run in a path that does not exist yet fails in `ensure_workspace`, not in the lookup.
- The watcher needs the `openjiuwen-harness` `notify` feature, which this crate enables unconditionally, so `watch` and `chat --watch` are always compiled in.
- The `auto-harness` dry run writes before it prints: a failed write is a `Config` error and nothing reaches stdout.
- A non-dry `auto-harness run` with resolved tasks prints `using N manually specified task(s)` on stdout before the `unsupported` error on stderr (exit `1`).
- A piped-stdin REPL session needs `<home>/settings.json` to exist (an interactive first run otherwise launches the onboarding wizard and consumes the piped line); the resumed-chat process test seeds it with `{}`.

## Known Issues

- The harness `Cli` has no `auto-harness`, `lsp`, `cron`, `team`, `watch` or `sessions` command, which is why the outer grammar and its globals are duplicated here; adding harness command groups would let this crate drop the copies.
- `openjiuwen-harness` keeps its CLI agent id private (`crates/openjiuwen-harness/src/cli/run.rs`'s `CLI_AGENT_ID`), so `src/sessions/catalog.rs` mirrors the string to key CLI records against the state and journal entries of the same run. Exporting `pub const CLI_AGENT_ID: &str = "openjiuwen-cli"` from `openjiuwen_harness::cli` would let this crate drop the duplicate.
- Nothing in this binary writes execution links, so the CLI's own runs resume with a "cannot confirm whether an earlier execution exists" stop note in the plan's background; the window and `deepagent` state are still restored. Writing a `First` link per run inside the harness would remove the note.
- `openjiuwen-rsi` ships no harness evaluation runner and no experience store, so a non-dry `auto-harness run` and `verify-ext` return typed `Error::Unsupported` and the experience commands read the documented JSONL directly instead of a Rust store.
- The resume summary uses `openjiuwen_harness::cli::session_id()` as the request id for `resume`, so a state-record resume is a *new* request id rather than the persisted one; the plan settles only its own waiting scene.
- `lsp` needs an installed language server: the built-in table is not probed against `PATH`, so a missing program surfaces as a typed launch failure (`error: tool: …`) on first use rather than a fallback to another server.
- `team`'s `invoke_tool` op is not exposed as a command: the client supports it, but the in-repo attach host always answers `unsupported`, so a command would only surface that error.
- `openjiuwen-e2e/tests/cli_binary.rs` (outside this node) builds the harness `Cli` as a struct literal including `no_color`, so any new harness global must be added there as well; the crate currently compiles.
- Deferred harness features reachable from these commands (browser/CUA/mobile presets, MCP resource tools needing a host backend, OTel export) are documented in `../openjiuwen-harness/` and `../../docs/PARITY.md`, not here.

## Routing Table

- `./src/main.rs` → process boundary: startup order, logging installation, clap parse, `cli::run` dispatch, exit-code mapping.
- `./src/cli.rs` → outer grammar (`Cli`, `GlobalArgs`, `Command`, `RunCliArgs`, `ChatCliArgs`), `to_harness`, dispatch into the harness or the binary's own commands, and the grammar/drift unit tests.
- `./src/extras.rs` → `run`/`chat` wiring for `--worktree` (worktree creation and workspace redirection), `chat --watch`, and `--resume`/`--continue` (the resumed run through `run_with_model_resumed`, the REPL adopting the persisted id).
- `./src/sessions/mod.rs` → `sessions`: the grammar, dispatch, and the list/show/prune text and JSON renderings.
- `./src/sessions/catalog.rs` → the catalogue behind both `sessions` and `--resume`: reading the three stores, the one-row-per-session merge (group key, richness, timestamp inheritance, `sources`), id/`Continue` resolution, the resume-plan probing, and prune selection/deletion.
- `./src/sessions/catalog/tests.rs` → the catalogue's unit tests: merge ranking, group keys, timestamp inheritance, `has_kind`/`is_resumable`, row ordering, role mapping, note prefixes, and the async store cases.
- `./src/lsp.rs` → `lsp`: action enum, server selection, `--file` pre-check, and payload rendering.
- `./src/cron.rs` → `cron`: the `CronScheduler` call per subcommand and its text/JSON output.
- `./src/team/mod.rs` → `team`: socket resolution, the attach client calls, and the roster/task/message renderings.
- `./src/watch.rs` → `watch` and the reusable watch session behind `chat --watch` (plan rendering, readiness line, update rendering, `--max-events`).
- `./src/auto_harness/mod.rs` → `AutoHarnessArgs`/`AutoHarnessCommand`, the dispatch, `data_dir`.
- `./src/auto_harness/run.rs` → `auto-harness run`: config loading and overrides, task resolution, the dry run, the deferred session error.
- `./src/auto_harness/experience.rs` → the read-only JSONL store behind `experience search`, `experience list` and `history` (scoring, ordering, type filter, output forms).
- `./src/auto_harness/deferred.rs` → `verify-ext` (typed deferred loader) and the `gap-analyze` placeholder.
- `./tests/support/mod.rs` → the shared `Sandbox` (temp cwd/home/workspace, ambient-env scrubbing, bounded spawn, `run`/`run_with_stdin`/`spawn`) and the exit/JSON/ANSI assertion helpers.
- `./tests/cli_smoke.rs` → process-level grammar and help suite (`--version`, every help page including `lsp`/`cron`/`team`/`watch`/`sessions`, `--no-color` placement, unknown subcommand).
- `./tests/cli_run.rs` → offline scripted `run` through the real process against the loopback mock model server, in all three output formats.
- `./tests/cli_features.rs` → `run --worktree`, the non-repository failure, and `chat --watch` through the real process.
- `./tests/cron.rs` → process-level `cron` suite: store, mutations, reads, and the typed failures.
- `./tests/sessions.rs` → process-level `sessions` suite: hand-written CLI records, a damaged journal tree, ordering/`--kind`/`--limit`, `show`, prune, the three-store dedupe fixture test, a real scripted run listed once with `--continue` following it, the typed failures and the help pages.
- `./tests/resume.rs` → process-level `run --resume`/`--continue`/`chat --resume` suite against the mock model, with the CLI's own journal-backed runs, persistent-agent journals and state records seeded through the harness API.
- `./tests/lsp.rs` → process-level `lsp` suite: help, unknown action, missing file, failing/unspawnable server programs.
- `./tests/team.rs` → process-level `team` suite against a stub attach host on a Unix socket (grammar, wire arguments, renderings, failures).
- `./tests/watch.rs` → process-level `watch` suite: `--list` text/JSON, the empty plan, and a live event with `--max-events`.
- `./tests/auto_harness.rs` → process-level `auto-harness` suite: task resolution, dry run, deferred surfaces, experience store, history, gap-analyze.
- `./README.md` → user-facing build/run instructions, the full command reference, exit codes, environment variables, data layout, and the Rust-port differences.
- `../openjiuwen-harness/src/cli/` → all `run`/`chat` CLI logic (`CliOptions`, `run_once`, `run_with_model_resumed`/`ResumedTurn`, `Repl` incl. `with_session_id`, permissions, session store, `RunOutcome::to_json`) and the recording a CLI run performs.
- `../openjiuwen-harness/src/persistent_agent/` → the layout, execution journals, session records, links and `resume` plan the catalogue reads.
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
- `tests/sessions.rs` (10 tests) drives the catalogue through the real process on hand-written `<home>/sessions/<id>.json` records and a deliberately damaged `persistent_agent/logs` tree: the empty listing in text and JSON, newest-first rows with `--limit 1` and `--kind execution`, `show` text (header lines, `messages 2`, one `<role>: <content>` line per message) and `show --json` (`messages` order, `record.session_id`, null `plan`), a malformed `meta.json` plus a metadata-less directory being skipped instead of failing the listing, the unknown-id `error: not_found: unknown session:` (exit `1`, empty stdout), `prune --all --dry-run` reporting newest-first and deleting nothing, `prune --all` then `no sessions`, the `{dry_run, pruned, entries}` JSON, `--older-than 7` on a record dated now as `nothing to prune`, the selector-less `validation` error, `--all` with `--older-than` as clap exit `2`, and the `sessions`/`list`/`prune`/`show` help pages.
- `tests/sessions.rs` also seeds one session in all three stores (`persistent_agent/logs` journal, `sessions/openjiuwen-cli/<id>/state.json`, hand-written CLI record): the listing has one text line and one JSON row (`kind: "state"`), `--kind cli|execution|state` each find it exactly once, `show --json` seeds non-empty messages carrying the journalled question, `prune --all --dry-run` reports one entry, and a real `prune --all` leaves `no sessions` with the journal directory, the `state.json` directory and the CLI record all gone.
- `tests/sessions.rs` additionally runs the real binary against `MockLlmServer`: one scripted `run` produces exactly one `sessions list --json` row for its `session_id` (reported from `state`), `show` seeds messages, `run --continue` keeps that id with `resuming the most recent session <id>` on stderr, and the listing still has one row for it.
- `tests/resume.rs` (7 tests) covers the resume surface against the loopback mock model server: `run --resume <cli-id>` (the note says `from the persistent session record via execution …`, the JSON `session_id` is unchanged, the CLI record grows to four clean messages in order, and the restored question precedes the new one in the recorded model request), `--continue` twice (the same-id note plus a six-message record), the unknown-id `error: not_found: unknown session:` and the empty-catalogue `no resumable session found` (both exit `1` with empty stdout), a hand-seeded state record whose `sessions show --json` reports a non-null `plan` and a window carrying the journalled question **before** the resumed run rewrites the record's `ending`, `--resume exec-1` reporting `from execution exec-1 (2 messages restored)` and leaving a CLI record for the session it continues, and `chat --resume` with one piped line: the stderr note is the catalogue summary and names no second `cli-` id, the record for that id grows to four messages with the piped question, and `SessionStore::list` still reports exactly one session.
- The `tests/resume.rs` window assertions match restored text as substrings, because a window rebuilt from a journal renders `execution_started` as `【本次执行】<text>` and may open with a `【窗口摘要】…` frame; after a resumed run the record's `ending` names the new execution, so `show` reports the resumed turn's window rather than the seeded one.
- `tests/resume.rs` reads every CLI transcript through `openjiuwen_harness::cli::SessionStore` (`load` for one record, `list` for the catalogue) over the sandbox home, never by opening a file at a hand-built path or parsing the record document, so the store owns the on-disk encoding and a format or extension change cannot silently break the assertions.
- The catalogue's inline tests cover the pure helpers (merge ranking, group keys incl. a session-less execution, timestamp inheritance, `has_kind`, `is_resumable`, row ordering with mixed timestamps, role mapping, resume-note prefixes, kind tags) plus three async tests against real stores in a temp home/workspace: a persistent state record listing/resolving/pruning, a CLI record resolving with its transcript, and the typed `not_found`/`validation` failures.
- Hermeticity is enforced by `tests/support/mod.rs`: every child gets a fresh temp cwd, a temp `OPENJIUWEN_HOME` and a temp `-w` workspace; ambient `OPENJIUWEN_*`/`OPENAI_*`/`API_KEY`/`API_BASE`/`MODEL_*`/`NO_COLOR`/`RUST_LOG` are removed; stdin is `/dev/null` unless `run_with_stdin` pipes scripted input; and every spawn is bounded by a 120 s async timeout that kills the child instead of hanging the run.
- Nothing here needs the network, credentials or a terminal: the mock server binds loopback, the team suite binds its own Unix socket, and the `run`/`cli_features`/`resume`/`sessions` suites are the only ones that build a model.
- In-process CLI behaviour (streaming render, REPL transcripts, permissions, onboarding) is covered by `crates/openjiuwen-harness/tests/cli.rs` and `src/cli/tests.rs`; do not duplicate it here.
- Definition of done: `cargo fmt -p openjiuwen-cli`, `cargo test -p openjiuwen-cli --offline` (repeated to prove the process suites are not flaky), `cargo clippy -p openjiuwen-cli --all-targets --offline -- -D warnings`, `cargo metadata --offline --no-deps`, and a manual `cargo run -p openjiuwen-cli -- --help`.
