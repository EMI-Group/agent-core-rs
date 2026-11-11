# shell — shell command execution backend

## Intent

The shell half of `openjiuwen-sysop`: frozen shell types, first-token allowlist and regex deny
filtering, per-`ShellType` spawn-plan resolution, the session cancellation registry, the shared
child-process execution engine, and the local backend ported from Python
`openjiuwen/core/sys_operation/local/shell_operation.py`.
The parent `../CONTEXT.md` "Behavior contract" section is the authoritative Python-parity checklist;
this file records only what is specific to this subtree.

## API Surface

- `mod.rs` — `pub fn local_shell(config: LocalWorkConfig) -> Result<Arc<dyn ShellOperation>>` and the
  module declarations; `pub use local::LocalShellOperation`.
- `local::LocalShellOperation` — `new(config)`, `with_cancellation(token)`, `config()`, `policy()`,
  `filter()`, and `impl ShellOperation` (`execute_cmd`, `execute_cmd_stream`,
  `execute_cmd_background`, `execute_argv`).
- `allowlist::CommandFilter` — `new(config)`, `from_parts(..)`, `allowlist()`, `allows_command`,
  `allows_program`, `dangerous_match`, `check`, `first_token`.
- `plan::{ExecutionPlan, resolve_plan, extract_absolute_paths, is_ignored_absolute_path}` — the
  interpreter invocation for one `ShellType` and the absolute-path literals a command mentions.
- `registry::{ShellProcessRegistry, Registration}` — `global()`, `register`, `cancel_session`,
  `cancel_all`, `active_sessions`, `is_registered`.
- `types::{ShellType, StreamKind, ShellExecParams, ShellArgvParams, ShellOutput, ShellChunk,
  ShellBackgroundOutput}` — Python field names, serde defaults, and `JsonSchema` derivation.
- `process` (`pub(crate)`) — `spawn_child`, `apply_environment`, `run_captured(child, RunOptions)`
  yielding `RunOutcome`, `stream_child(child, RunOptions)` yielding a `BoxStream<EngineEvent>`, the
  `CapturedText`/`RunOptions`/`RunOutcome`/`EngineEvent` types, and the background stderr capture
  helpers `drain_stderr`, `background_early_exit`, `StderrSink`.

## Constraints

- Engine contract: `spawn_child` hardens every child (stdin null, piped stdout/stderr,
  kill-on-drop, own process group on Unix); callers never spawn directly.
- Capture is bounded per stream by `RunOptions::max_output_bytes` and appends
  `constants::OUTPUT_TRUNCATION_MARKER` once when the budget is exceeded.
- `RunOptions::timeout_secs == 0` means no deadline; a timeout or any of the cancellation tokens
  kills the child exactly once, keeps partial output, and sets `timed_out`/`cancelled`.
- `stream_child` emits exactly one terminal event (`Exit`, `TimedOut`, or `Cancelled`) and joins its
  pipe readers with a grace period; reader failures end the affected pipe and never panic.
- Cancellation tokens come from `shell::registry` (per session) plus the optional per-instance token.
- Gate order is blank command → deny list → allowlist → cwd/sandbox → shell plan, so a rejected
  command never spawns a child.
- `execute_argv` never uses a shell and applies the allowlist to `argv[0]` only; arguments are data.
- `tracing` logging only, no `unwrap`/`expect` outside `#[cfg(test)]`, no `unsafe`, files ≤1000 lines.

## Routing Table

- `./types.rs` → `ShellType`, `StreamKind`, params and result types with Python defaults and schemas
- `./allowlist.rs` → `CommandFilter`: quote-aware first-token splitter, allowlist, regex deny list
- `./plan.rs` → `resolve_plan` per `ShellType`/OS and absolute-path literal extraction
- `./registry.rs` → `ShellProcessRegistry`: per-session cancellation tokens, RAII `Registration`
- `./process.rs` → shared child-process engine: hardened spawn, bounded capture, timeout, streaming, background stderr capture
- `./local.rs` → `LocalShellOperation`: policy gates, spawn plan, trait impls, facade tests
- `./mod.rs` → `local_shell` factory and module declarations

## Test Strategy

- Inline facade tests in `local.rs` cover the ported Python behavior (empty command, allowlist and
  deny list, output capture, bounded capture marker, timeout, env overlay/scrub, cwd resolution,
  sandbox rejection, argv form, background early exit and pid, stream chunks, session and instance
  cancellation) and skip with a message when `sh`/`echo` is missing.
- Inline engine tests in `process.rs` cover the incremental UTF-8 decoder (split characters, invalid
  and incomplete bytes).
- Every test is offline and deterministic (`tempfile` workspaces, no network).
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop --offline`,
  `cargo test -p openjiuwen-sysop --features sandbox --offline`, and
  `cargo clippy -p openjiuwen-sysop --all-targets -- -D warnings`.
