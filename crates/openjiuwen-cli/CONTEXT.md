# openjiuwen-cli

## Intent

The `openjiuwen` executable: the product entry point a user runs.
It is a thin process boundary over `openjiuwen_harness::cli` — command grammar, option resolution, the one-shot runner, the interactive REPL, permission prompts and session storage all live in the harness CLI, not here.
This crate owns only what the operating system observes: `.env` loading, logging installation, clap exit codes, and the stderr diagnostic for a failed run.
Python parity: `openjiuwen` console script and `openjiuwen/harness/cli/cli.py`'s `main`.

## API Surface

Binary `openjiuwen` (`src/main.rs`, `[[bin]] name = "openjiuwen"`); the crate has no library target and no public Rust API.
- Entry point: `#[tokio::main] async fn main() -> ExitCode`.
- Startup order: `dotenvy::dotenv()` (errors ignored) → `init_logging()` → `Cli::parse()` → `openjiuwen_harness::cli::run_cli(cli).await`.
- `init_logging()` installs `openjiuwen_common::logging` with `LogWriter::Stderr`; `RUST_LOG` (trimmed, ignored when blank) wins over `OPENJIUWEN_LOG_LEVEL` / `OPENJIUWEN_LOG_FORMAT`; an invalid filter prints a warning and falls back to the env/default substrate so logging never blocks startup.
- Subcommands are delegated verbatim to the harness: `run [PROMPT] [-f text|json|stream-json]` (one prompt, prompt read from stdin when omitted or `-`) and `chat` (interactive REPL); global options `-m/--model`, `--provider`, `--api-key`, `--api-base`, `-w/--workspace`, `-v/--verbose`; no subcommand → `chat` on a TTY, otherwise `run`.
- Exit codes: `0` success and `--help`/`--version`; `2` usage errors (clap's `parse()` exit code: unknown subcommand/flag, bad value); `1` a runtime failure from `run_cli`, reported as `error: <cause>` on stderr.
- `--version` is clap's `version` attribute on `Cli` (`<name> <version>` on stdout), backed by the shared workspace version, so the binary reports the version of the CLI logic it embeds.

## Constraints

- Only the harness CLI defines command behaviour; this crate must not re-implement parsing, option precedence, rendering or exit-code classification that `openjiuwen_harness::cli` already owns.
- Logs go to **stderr only**: stdout is a machine-readable result stream for `run -f json` / `run -f stream-json`.
- No panics on user input — parsing is delegated to clap and `run_cli` returns `openjiuwen_common::error::Error`; `unwrap`/`expect`/indexing on input are forbidden in `src/`.
- `println!`/`eprintln!` are allowed here (binary crate) but limited to the user-facing diagnostic and logging fallback; other binaries follow the same boundary rule.
- Async-first: the process runs on the tokio multi-thread runtime provided by `#[tokio::main]`; no blocking calls are added at the boundary.
- Do not add sibling binaries here.
- The `[[bin]]` target declares `doc = false`: the binary is named `openjiuwen`, so documenting it would write `target/doc/openjiuwen/index.html` and collide with the `openjiuwen` umbrella crate's lib docs.

## Gotchas

- `-h` and `--help` differ: `-h` prints the explicit `about = "openJiuwen agent CLI"`, `--help` prints the long help synthesized from the `Cli` struct doc comment; assert on `Usage:`, `Commands:`, `run`/`chat` rows instead of the banner.
- `version` is set only on the top-level `Cli` (no `propagate_version`), so `openjiuwen --version` / `-V` prints `openjiuwen <workspace version>` while `openjiuwen run --version` is a usage error (exit `2`).
- `dotenvy::dotenv()` in `main` loads `.env` from the working directory **and its ancestors**, so a spawned-process test must run the child in a temp cwd (or clear inherited `API_KEY`/`OPENAI_*`) or ambient state leaks in.
- `env!("CARGO_BIN_EXE_openjiuwen")` exists only for integration tests of this package (Cargo defines it per binary-owning package); tests in other crates cannot use it, and this crate has no lib target to depend on.
- Runtime failures print `error: <kind>: <cause>` (the `Error` display adds the kind prefix) and exit `1`; the run may already have created the workspace directory by then.

## Routing Table

- `./src/main.rs` → the whole binary: startup order, logging installation, clap parse, `run_cli` dispatch, exit-code mapping.
- `./tests/cli_smoke.rs` → process-level smoke suite for the compiled binary (see Test Strategy).
- `../openjiuwen-harness/src/cli/` → all CLI logic (`Cli`, `CliOptions`, `run_once`, `Repl`, permissions, session store); read its `CONTEXT.md` before changing CLI behaviour.
- Sibling binaries: `team-member` and `openjiuwen-team-mcp` ship from `openjiuwen-agent-teams` (`src/bin/`), not from this crate; `openjiuwen-team-mcp` is behind the `mcp` feature.

## Test Strategy

- `./tests/cli_smoke.rs` spawns `env!("CARGO_BIN_EXE_openjiuwen")` with `std::process::Command`; it asserts only on process-visible behaviour, with no network or model calls:
  `--version` exits 0 and prints the crate version on stdout; `--help` exits 0 and lists `run` and `chat` rows; `run --help` exits 0 and lists `--output-format`; an unknown subcommand exits nonzero with `unrecognized subcommand` on stderr and empty stdout.
- The help/version paths are clap-only, so they never touch the workspace directory, the home directory or settings resolution (the harness creates directories lazily) — the suite stays hermetic.
- End-to-end behaviour (streaming, output formats, permissions, REPL) is covered by the harness CLI suites (`crates/openjiuwen-harness/tests/cli.rs`, `src/cli/tests.rs`) against scripted agents; do not duplicate it here.
- Definition of done: `cargo fmt -p openjiuwen-cli`, `cargo test -p openjiuwen-cli`, `cargo clippy -p openjiuwen-cli --all-targets -- -D warnings`, and a manual `cargo run -p openjiuwen-cli -- --help`.
