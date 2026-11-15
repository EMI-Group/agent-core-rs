# shell — shell command execution backend

## Intent

The shell half of `openjiuwen-sysop`: frozen shell types, first-token allowlist and regex deny filtering, per-`ShellType` spawn-plan resolution, the session cancellation registry, the shared child-process execution engine with process-group teardown, the line-buffering wrapper, PTY-backed streaming, and the local backend.
It is ported from Python `openjiuwen/core/sys_operation/local/shell_operation.py` (`_build_shell_command`, `shell_process_registry.py`, `utils._kill_process_tree`).
The parent `../CONTEXT.md` "Behavior contract" section is the authoritative Python-parity checklist; this file records only what is specific to this subtree.

## API Surface

- `mod.rs` — `pub fn local_shell(config: LocalWorkConfig) -> Result<Arc<dyn ShellOperation>>`, the module declarations, and `pub use local::LocalShellOperation`.
- `local::LocalShellOperation` — `new(config)`, `with_cancellation(token)`, `config()`, `policy()`, `filter()`, and `impl ShellOperation` (`execute_cmd`, `execute_cmd_stream`, `execute_cmd_background`, `execute_argv`).
- `allowlist::CommandFilter` — `new(config)`, `from_parts(..)`, `allowlist()`, `allows_command`, `allows_program`, `dangerous_match`, `check`, `first_token`.
- `plan::{ExecutionPlan, resolve_plan, extract_absolute_paths, is_ignored_absolute_path}` — the interpreter invocation for one `ShellType` and the absolute-path literals a command mentions.
- `registry::{ShellProcessRegistry, Registration, terminate_group}` — `global()`, `register`, `register_group`, `cancel_session`, `cancel_all`, `active_sessions`, `is_registered`; `terminate_group(pid)` is the pid-based stop path for a background command.
- `types::{ShellType, StreamKind, ShellExecParams, ShellArgvParams, ShellOutput, ShellChunk, ShellBackgroundOutput}` — Python field names, serde defaults, and `JsonSchema` derivation.
- `process` (`pub(crate)`) — `spawn_child`, `apply_environment`, `run_captured(child, RunOptions)` yielding `RunOutcome`, `stream_child(child, RunOptions)` yielding a `BoxStream<EngineEvent>`, and `drive_child` (the timeout/cancellation/terminal-event driver both the pipe and PTY readers use).
- `process` types — `CapturedText`, `RunOptions`, `RunOutcome`, `EngineEvent`, `ProcessGroup`.
- `process` process-group helpers — `kill_group(pid)` (`killpg(pid, SIGKILL)`, best effort) and `ProcessGroup::terminate(reaped)` (kill-once guard).
- `process` line buffering — `line_buffered_argv(argv, enabled)` prepends `stdbuf -oL -eL` when it resolves.
- `pty` (`pub(crate)`, Unix only) — `PtyChild`, `try_spawn(command)`, `stream_pty_child(child, master, RunOptions)`; a failed allocation returns `None` so the caller falls back to pipes.
- `process` background stderr capture — `drain_stderr`, `background_early_exit`, `StderrSink`.

## Constraints

- Engine contract: `spawn_child` hardens every child (stdin null, piped stdout/stderr, kill-on-drop, own process group on Unix); callers never spawn directly.
- Capture is bounded per stream by `RunOptions::max_output_bytes` and appends `constants::OUTPUT_TRUNCATION_MARKER` once when the budget is exceeded.
- `RunOptions::timeout_secs == 0` means no deadline.
- A timeout or any of the cancellation tokens kills the child exactly once, keeps partial output, and sets `timed_out`/`cancelled`.
- `stream_child` emits exactly one terminal event (`Exit`, `TimedOut`, or `Cancelled`) and joins its pipe readers with a grace period.
- Reader failures end the affected pipe and never panic; anything other than clean EOF / `EIO` / `BrokenPipe` is logged at `warn!` with the path or stream name, so truncated output is never silent.
- Best-effort steps log their outcome instead of discarding it: a failed `start_kill`, a failed reap, a failed group termination, a session-map poison, and the PTY→pipe fallback each emit `warn!`/`debug!` with the pid or stream and the underlying error.
- A child that was signalled (or whose status could not be read) still reports exit code `-1` (Python parity), with a `debug!`/`warn!` naming the case.
- Cancellation tokens come from `shell::registry` (per session) plus the optional per-instance token.
- Every child is spawned into a fresh process group (`Command::process_group(0)`, pgid = child pid). A timeout, a cancellation, or a session teardown kills **the group** with `killpg(SIGKILL)` before `Child::wait`, and `ProcessGroup` guarantees at most one kill and never a kill after the reap (pid reuse).
- `setsid()` is deliberately not used: it needs an `unsafe` `pre_exec` hook, which the crate forbids. `process_group(0)` gives the same killpg-able group without detaching a controlling terminal.
- Line buffering is applied at spawn time (never inside `plan::resolve_plan`), so `resolve_plan` still yields `/bin/sh` and the allowlist/sandbox gates see the user's command. Missing `stdbuf` or a non-Unix host spawns the plan unwrapped — never an error.
- `execute_argv` is never wrapped and never uses a shell.
- `pty_streaming` affects only `execute_cmd_stream`: a terminal merges stderr into stdout, so PTY chunks are always `StreamKind::Stdout`; captured runs keep pipes. The slave clears `ECHO`/`OPOST`/`ONLCR` through the safe nix termios API, and the reader drops `\r` so output is stable even when termios could not be applied.
- Gate order is blank command → deny list → allowlist → cwd/sandbox → shell plan, so a rejected command never spawns a child.
- `execute_argv` never uses a shell and applies the allowlist to `argv[0]` only; arguments are data.
- `tracing` logging only, no `unwrap`/`expect` outside `#[cfg(test)]`, no `unsafe`, files ≤1000 lines.

## Routing Table

- `./types.rs` → `ShellType`, `StreamKind`, params and result types with Python defaults and schemas
- `./allowlist.rs` → `CommandFilter`: quote-aware first-token splitter, allowlist, regex deny list
- `./plan.rs` → `resolve_plan` per `ShellType`/OS and absolute-path literal extraction
- `./registry.rs` → `ShellProcessRegistry`: per-session cancellation tokens, RAII `Registration`
- `./process.rs` → shared child-process engine: hardened spawn, process-group kill, line-buffering wrapper, bounded capture, timeout, streaming, background stderr capture
- `./pty.rs` → pseudo-terminal allocation, terminal configuration, and the PTY stream reader (Unix only)
- `./local.rs` → `LocalShellOperation`: policy gates, spawn plan, trait impls, facade tests
- `./mod.rs` → `local_shell` factory and module declarations

## Test Strategy

- Inline facade tests in `local.rs` cover the ported Python behavior: empty command, allowlist and deny list, output capture, bounded capture marker, timeout, env overlay/scrub, cwd resolution, sandbox rejection, argv form, background early exit and pid, stream chunks, and session/instance cancellation.
- Facade tests that need `sh` or `echo` skip with a message when the binary is missing.
- Inline engine tests in `process.rs` cover the incremental UTF-8 decoder (split characters, invalid and incomplete bytes), the line-buffering wrapper decision (applied, skipped without a wrapper, skipped when disabled, empty argv), and the `ProcessGroup` guard (never after the reap, kill-at-most-once, pid 0 rejected).
- `tests/process_group.rs` proves the tree kill with a shell that leaves a `sleep 30` grandchild: timeout, per-instance cancellation, session cancellation, and the background stop path all leave no live grandchild (bounded `/proc` polling, `kill -0` elsewhere), and `execute_argv` inherits no `stdbuf` wrapper.
- `tests/pty_streaming.rs` asserts the PTY path is really a terminal (`[ -t 1 ]`), that four lines arrive as several chunks with no `\r`, that `exit 3` is reported on the terminal chunk, and that pipes are still used when `pty_streaming` is off.
- Inline PTY tests in `pty.rs` cover `\r` stripping, the configured terminal attributes, and an end-to-end terminal run.
- Every test is offline and deterministic (`tempfile` workspaces, no network).
- Gates: `cargo fmt -p openjiuwen-sysop`, `cargo test -p openjiuwen-sysop --offline`, `cargo test -p openjiuwen-sysop --features sandbox --offline`, and `cargo clippy -p openjiuwen-sysop --all-targets -- -D warnings`.

## Notes for Agents

- `local.rs` re-exports the engine as `pub(crate) use crate::shell::process;` so the sibling `src/code/local.rs` keeps resolving `crate::shell::local::process`.
- Add new engine behavior to `process.rs`, never to `local.rs`, so the shell and code backends cannot diverge.
