# openjiuwen-sysop integration tests

## Intent

External integration suites for `openjiuwen-sysop`: they exercise the crate through its public
API only (`openjiuwen_sysop::...`), the way a downstream crate or the harness consumes it.
Each file ports behavior from the Python reference package `openjiuwen.core.sys_operation`
and is named for the surface it covers, so a Python-derived expectation can be checked here
without reading the unit tests beside the implementation.

## API Surface

Integration tests may use only items exported from `openjiuwen_sysop`:
`SysOperation::{local, new, sandbox, fs, shell, code, tool_specs, tool_specs_for}`,
`fs::local_fs`, `shell::local_shell`, `code::local_code`,
`card::SysOperationCard`, `config::{LocalWorkConfig, EnvPolicy, BinaryPolicy, EnvInheritMode}`,
`sandbox::config::*`, `mode::{OperationMode, OperationKind}`,
`operation::{FsOperation, FsExtOperation, ShellOperation, CodeOperation, OpStream}`,
`fs::types::*`, `shell::types::*`, `code::types::*`,
`result::OpResult`, `status::SysOpStatus`, `tool::OperationToolSpec`.
`pub(crate)` items are not reachable here; a test that needs one belongs in a `#[cfg(test)]`
module beside the code.

## Constraints

- Every file starts with a module doc comment naming the Python test it ports: a line
  beginning `//! Ports` followed by the backticked Python test path, e.g.
  ``tests/unit_tests/core/sys_operation/local/test_fs_operation.py``.
  The path must be backticked: a bare path trips `clippy::doc_markdown`, which is an error
  under `-D warnings`.
- Only files in this directory are test targets; helper code is duplicated per file rather
  than shared through a module, so each suite stays independently readable.
- Offline and deterministic: `tempfile` workspaces, no network, no fixed absolute paths, no
  reliance on the developer's home directory.
- Runtime-dependent cases (`sh`, `bash`, `python3`, `node`) first check availability
  (`which::which`), then `eprintln!` an explanatory skip message and return; a missing runtime
  is never a failure.
- Plain `assert!`/`assert_eq!` with descriptive messages; `unwrap`/`expect` are allowed under a
  file-level `#![allow(clippy::unwrap_used, clippy::expect_used)]`, because workspace lints are
  inherited by integration-test targets and CI runs clippy with `-D warnings`.
- Clippy runs with `all` + `pedantic`: fix warnings rather than widening allows (only
  `unwrap_used`/`expect_used` may be allowed file-wide).
- Async tests use `#[tokio::test]`; streams are drained with `futures::StreamExt`.
- No dependency may be added: `tempfile` (dev), `tokio`, `futures`, `serde_json`, `which` and
  `async-stream` are already available to test targets.
- Tests are additive: they must pass against the current implementation, and a failing
  expectation is either a test bug or a parity bug to report — never silence an assertion.

## Routing Table

- `./card_json.rs` → card/config JSON parity: field names, mode case-insensitivity, validation (199002), tool-id shape, defaults, unknown-field tolerance
- `./local_fs.rs` → local read/write/append/list/search/upload/download round-trips and options
- `./fs_edges.rs` → read slicing, byte-mode errors, chunking, size/encoding limits, conditional write (199011), streaming shape
- `./traversal.rs` → sandbox containment: `..`, absolute escapes, symlinks, root-prefix false positives
- `./shell_local.rs` → local shell: allowlist/denylist, capture, truncation, timeout, cwd, env, argv, streaming, background
- `./code_local.rs` → local code: python3/node execution, non-zero exit semantics, validation, temp-file strategy, timeout
- `./tool_specs.rs` → 13 generated tool specs, ids, names, JSON schemas, kind filtering
- `./sandbox_stub.rs` → `sandbox`-feature behavior without a remote provider: card validation, unsupported errors, forwarding shape

## Notes for Agents

- `WriteFileParams::prepend_newline` defaults to `true`; use `WriteFileParams::raw` when the file
  must contain exactly the content.
- Failure results carry `code`/`message`; match on `SysOpStatus::*.code()` and on the Python
  message templates from `status.rs` (e.g. `file version conflict`, `command can not be empty`).
- `SysOperation::local` builds all three backends; the direct constructors
  (`fs::local_fs`, `shell::local_shell`, `code::local_code`) avoid constructing unused ones.
- A sandbox build without a registered provider fails at `SysOperation::new`, not on first use;
  no test may register anything in the process-global provider registry.
