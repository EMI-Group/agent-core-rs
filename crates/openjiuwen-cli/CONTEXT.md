# openjiuwen-cli

## Intent

The `openjiuwen` executable: the product entry point a user runs.
It is a thin process boundary that owns the outer clap grammar, forwards `run`/`chat` to `openjiuwen_harness::cli::run_cli`, and dispatches the `auto-harness` command group itself.
The operating-system concerns also live here: `.env` loading, `tracing` installation on stderr, clap's exit codes, and the stderr diagnostic for a failed run.
Everything about `run`/`chat` behaviour — option resolution, the one-shot runner, the REPL, permission prompts, session storage — belongs to the harness.
Python parity: the `openjiuwen` console script and `openjiuwen/harness/cli/cli.py`'s `main` plus its `auto-harness` Click group.

## API Surface

Binary `openjiuwen` (`src/main.rs`, `[[bin]] name = "openjiuwen"`); the crate has no library target.
- Entry point: `#[tokio::main] async fn main() -> ExitCode`.
- Startup order: `dotenvy::dotenv()` (errors ignored, walks the working directory and its ancestors) → `init_logging()` → `cli::Cli::parse()` → `cli::run(cli).await`.
- `init_logging()` installs `openjiuwen_common::logging` with `LogWriter::Stderr`; `RUST_LOG` (trimmed, ignored when blank) wins over `OPENJIUWEN_LOG_LEVEL`/`OPENJIUWEN_LOG_FORMAT`, and an invalid filter prints a warning and falls back to the env/default substrate so logging never blocks startup.

### Grammar (`src/cli.rs`)

- `Cli { global: GlobalArgs, command: Option<Command> }`, `#[command(name = "openjiuwen", version, about = "openJiuwen agent CLI")]`.
- `GlobalArgs` mirrors the harness globals flag-for-flag and all are `global = true`: `-m/--model`, `--provider`, `--api-key`, `--api-base`, `-w/--workspace`, `-v/--verbose`, `--no-color`.
- `Command::{Run(RunArgs), Chat(ChatArgs), AutoHarness(AutoHarnessArgs)}`: `RunArgs`, `ChatArgs` and `OutputFormat` are the harness types reused verbatim, `AutoHarnessArgs` is this crate's own.
- `Cli::to_harness()` forwards every global (including `--no-color`) and maps `Run`/`Chat` onto the harness `Command`; `AutoHarness` and "no subcommand" become `command: None`, so the harness only reads the globals and keeps its TTY-based dispatch.
- `cli::run` sends `auto-harness` to `auto_harness::run(args, &cli.to_harness())` and everything else to `openjiuwen_harness::cli::run_cli`.
- Two unit tests guard the duplication: every long flag of `openjiuwen_harness::cli::Cli::command()` and every harness subcommand must exist here.

### `auto-harness` (`src/auto_harness/`)

- `AutoHarnessArgs { json: bool (group-global), command: AutoHarnessCommand }`; `--json` is a Rust addition that switches every text command to JSON.
- `AutoHarnessCommand::{Run, Experience, VerifyExt, GapAnalyze, History}`.
- `run`: flags `--task <TEXT>`, `--task-file <PATH>`, `--dry-run`, `--stage <assess|plan|implement|verify>`, `--no-push` (parity only, no effect), `--budget <SECS>`, `--goal <TEXT>`, `--pipeline <meta|extended|auto>` (absent means `meta`).
- `run` loads `<data-dir>/config.yaml` via `AutoHarnessConfig::load_or_default`, applies the overrides, validates, resolves the tasks, and either writes/echoes the `--dry-run` list or returns `Error::Unsupported` naming the missing harness evaluation runner.
- `--dry-run` writes `<data-dir>/runs/tasks.json`, prints the same pretty JSON array on stdout, and prints `[dry-run] task list -> <path>` on stderr.
- `--task` wins over `--task-file`; a task file may be a JSON array or a single object, needs `topic` per entry, defaults `description` to `""` and `files` to `[]`, and ignores unknown keys.
- `experience search <QUERY>` searches a lowercased `topic`/`summary`/`details` blob (`outcome` is never searched), drops query words shorter than two characters, scores hits plus a 30-day linear recency bonus, and prints the best ten.
- `experience list [--type <TYPE>] [--limit <N>]` (default limit 10) and `history [--limit <N>]` (default 20) sort newest-first, truncate to the limit, and only then apply the exact, case-sensitive type filter.
- Both print `[<type>] <topic>: <summary or outcome>` per row, the documented `no experience records` / `no matching experiences` message when empty, and a pretty JSON array with `--json`.
- `verify-ext [--ext-path <DIR>]` always returns `Error::Unsupported` from `openjiuwen_rsi::deferred::RuntimeExtensionLoader` and does not validate the path.
- `gap-analyze` prints the placeholder line, or with `--json` the object `{"gaps": [], "placeholder": true, "note": …}`, and exits `0`.
- `data_dir` reproduces Python's asymmetry: `<-w workspace>/auto_harness` when `-w` was given, otherwise `<home>/auto_harness`; the store commands always read `<resolved workspace>/auto_harness/experience/experiences.jsonl`.

### Exit codes

- `0` success, and `--help`/`--version`.
- `2` usage errors from clap's `parse()`: unknown subcommand or flag, missing value, invalid enum value.
- `1` a runtime failure from `run_cli` or `auto_harness::run`, printed as `error: <kind>: <cause>` on stderr (for example `error: unsupported: …`).

## Constraints

- Only the harness defines `run`/`chat` behaviour; this crate must not re-implement parsing, option precedence, rendering or exit-code classification it already owns.
- The duplicated globals are held in sync by the drift unit tests in `src/cli.rs` against `openjiuwen_harness::cli::Cli::command()`, not by convention.
- stdout stays machine-readable: `run -f json`/`-f stream-json`, `auto-harness --json` and the dry-run task array are the only stdout writers in those modes; logs, permission prompts, the `error:` diagnostic and the `[dry-run]` note go to stderr.
- No panics on user input — parsing is delegated to clap and command functions return `openjiuwen_common::error::Error`; `unwrap`/`expect`/indexing on input are forbidden in `src/`.
- `println!`/`eprintln!` are allowed here (binary crate) for user-facing results and diagnostics only; libraries log through `tracing`.
- No new crates.io dependencies: the crate builds offline with the workspace-pinned set, and `openjiuwen-test-support` is a dev-dependency only.
- Files stay around ≤1000 lines (`src/auto_harness/run.rs` is the largest at ~437 lines); sibling binaries are not added here.
- The `[[bin]]` target declares `doc = false` so rustdoc never collides with the `openjiuwen` umbrella crate's `target/doc/openjiuwen/`.

## Gotchas

- `-h` and `--help` differ: `-h` prints the short `about`, `--help` prints the long help synthesized from the `Cli` struct doc comment; assert on `Usage:`, `Commands:` and row names instead of the banner.
- `version` is set only on the top-level `Cli` (no `propagate_version`), so `openjiuwen --version`/`-V` prints `openjiuwen <workspace version>` while `openjiuwen run --version` is a usage error (exit `2`).
- `dotenvy::dotenv()` in `main` loads `.env` from the working directory and its ancestors, so every process test must run the child in a temporary cwd (or clear inherited `API_KEY`/`OPENAI_*`) or ambient state leaks in.
- `env!("CARGO_BIN_EXE_openjiuwen")` exists only for integration tests of this package; tests in other crates cannot use it, and this crate has no lib target to depend on.
- `--no-color` is forwarded through `to_harness`, so harness option resolution sees the flag exactly as typed; it is also accepted after a subcommand because clap marks it global.
- clap interleaves globals into every subcommand's help page, so help tests assert on the documented rows and flags rather than on exact layout or ordering.
- The `auto-harness` dry run writes before it prints: a failed write is a `Config` error and nothing reaches stdout.

## Known Issues

- The harness `Cli` has no `auto-harness` command, which is why the outer grammar and its globals are duplicated here; adding a harness command group would let this crate drop the copy.
- `openjiuwen-rsi` ships no harness evaluation runner and no experience store, so a non-dry `auto-harness run` and `verify-ext` return typed `Error::Unsupported` and the experience commands read the documented JSONL directly instead of a Rust store.
- `openjiuwen-e2e/tests/cli_binary.rs` (outside this node) builds the harness `Cli` as a struct literal including `no_color`, so any new harness global must be added there as well; the crate currently compiles.
- Deferred harness features reachable from `run`/`chat` (browser/CUA/mobile presets, LSP, MCP resource tools needing a host backend, OTel export) are documented in `../openjiuwen-harness/` and `../../docs/PARITY.md`, not here.

## Routing Table

- `./src/main.rs` → process boundary: startup order, logging installation, clap parse, `cli::run` dispatch, exit-code mapping.
- `./src/cli.rs` → outer grammar (`Cli`, `GlobalArgs`, `Command`), `to_harness`, dispatch into the harness or `auto_harness`, and the drift unit tests.
- `./src/auto_harness/mod.rs` → `AutoHarnessArgs`/`AutoHarnessCommand`, the dispatch, `data_dir`.
- `./src/auto_harness/run.rs` → `auto-harness run`: config loading and overrides, task resolution, the dry run, the deferred session error.
- `./src/auto_harness/experience.rs` → the read-only JSONL store behind `experience search`, `experience list` and `history` (scoring, ordering, type filter, output forms).
- `./src/auto_harness/deferred.rs` → `verify-ext` (typed deferred loader) and the `gap-analyze` placeholder.
- `./tests/support/mod.rs` → the shared `Sandbox` (temp cwd/home/workspace, ambient-env scrubbing, bounded spawn) and the exit/JSON/ANSI assertion helpers.
- `./tests/cli_smoke.rs` → process-level grammar and help suite (`--version`, every help page, `--no-color` placement, unknown subcommand).
- `./tests/cli_run.rs` → offline scripted `run` through the real process against the loopback mock model server, in all three output formats.
- `./tests/auto_harness.rs` → process-level `auto-harness` suite: task resolution, dry run, deferred surfaces, experience store, history, gap-analyze.
- `./README.md` → user-facing build/run instructions, the full command reference, exit codes, environment variables, data layout, and the Rust-port differences.
- `../openjiuwen-harness/src/cli/` → all `run`/`chat` CLI logic (`CliOptions`, `run_once`, `Repl`, permissions, session store, `RunOutcome::to_json`).
- `../openjiuwen-rsi/src/harness_rsi/auto_harness/` → `AutoHarnessConfig`, `OptimizationTask`, `PipelinePreference` and the `deferred::RuntimeExtensionLoader` this crate calls.

## Test Strategy

- `tests/cli_smoke.rs` (8 tests) covers the clap layer only — `--version`, the `run`/`chat`/`auto-harness` rows, `auto-harness`'s five subcommands, `run`'s seven flags, `experience`'s `search`/`list`, `--no-color` before and after a subcommand, and the unknown-subcommand usage error (exit `2`, diagnostic on stderr, empty stdout).
- `tests/cli_run.rs` (5 tests) drives a real `run` through `MockLlmServer`: a reusable scripted fallback answers any number of agent-loop calls, the connection is passed as flags (`--provider openai --api-base <mock> --api-key … -m mock-model`), and the cases assert the `json` object keys (`answer`, `result_type`, `usage`, `session_id`), the `text` stream, one JSON object per `stream-json` line with a terminal `{"type":"result"}` line, escape-free stdout, `--no-color` through a real run, and `chat --help`.
- `tests/auto_harness.rs` (20 tests) pins task resolution (`--task`, array/object task files, unknown keys, missing `topic`, `--task` precedence, the empty list), the dry run and its `runs/tasks.json`, the deferred non-dry run (exit exactly `1`, `unsupported` + `harness evaluation runner`), `--stage` in the message, `--pipeline bogus` as exit `2`, the seeded experience store (ordering, malformed line skipped, limit before type filter, search fields, short tokens, `--json`, empty-store messages), `history` (limit and JSON), and `gap-analyze` text/JSON.
- Hermeticity is enforced by `tests/support/mod.rs`: every child gets a fresh temp cwd, a temp `OPENJIUWEN_HOME` and a temp `-w` workspace; ambient `OPENJIUWEN_*`/`OPENAI_*`/`API_KEY`/`API_BASE`/`MODEL_*`/`NO_COLOR`/`RUST_LOG` are removed; stdin is `/dev/null`; and every spawn is bounded by a 120 s async timeout that kills the child instead of hanging the run.
- Nothing here needs the network, credentials or a terminal: the mock server binds loopback, and the `run` suite is the only one that builds a model.
- In-process CLI behaviour (streaming render, REPL transcripts, permissions, onboarding) is covered by `crates/openjiuwen-harness/tests/cli.rs` and `src/cli/tests.rs`; do not duplicate it here.
- Definition of done: `cargo fmt -p openjiuwen-cli`, `cargo test -p openjiuwen-cli --offline` (repeated to prove the process suites are not flaky), `cargo clippy -p openjiuwen-cli --all-targets --offline -- -D warnings`, `cargo metadata --offline --no-deps`, and a manual `cargo run -p openjiuwen-cli -- --help`.
