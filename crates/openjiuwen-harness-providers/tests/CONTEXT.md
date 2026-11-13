# Tests

## Intent

Offline integration suites for the provider registry, the native provider wiring, the IO adapter, the implemented vendor providers (`claudecode`, `codex`, `dsh`), the team-hosted `native_v2` provider, the skill installer, and the shared vendor CLI subprocess transport (NDJSON and JSON-RPC).
Behavior ports name their Python counterpart (`tests/unit_tests/harness_providers/<file>::<test>`) in a comment.

## API Surface

No public API; the suites exercise `openjiuwen_harness_providers` through its public surface and the public `openjiuwen_harness` / `openjiuwen_harness_protocol` types.

## Constraints

- Offline and deterministic: no network, no sleeps (bounded `tokio::time::timeout` awaits only), `tempfile::tempdir` for every filesystem case.
- The vendor providers are real subprocess adapters: the suites start `support::FakeCli`, a local POSIX shell script speaking the provider's wire protocol, and never a vendor CLI or SDK.
- `create` resolves the configured executable only; it returns an unstarted harness and every suite asserts that before any `start`.
- Test doubles (`FakeHarness`, `FakeProvider`, `FakeSink`, `FakeToolGateway`, `FakeDriver`) live inside the test files and double as usage examples.
- `support::FakeCli` is the one shared fixture: a temp-directory POSIX shell script with a stdin transcript, included per binary with `mod support;` (the module carries `#![allow(dead_code)]` and `#![allow(unreachable_pub)]`).
- `FakeCli` stages the script as `fake-cli.staged`, chmods it and renames it onto `fake-cli`, and the bytes are written by a helper `cat` process: a write descriptor opened in the test process is copied into every concurrently forked child and makes `exec` fail with `ETXTBSY` (`Text file busy`).
- Registry tests use globally unique provider names instead of a reset hook, because the registry is process-global.
- `#![allow(clippy::unwrap_used, clippy::expect_used)]` at the top of each suite (assert-by-panic is the intent).
- Vendor suites are `#![cfg(feature = "vendor")]` and additionally `#![cfg(unix)]` when they start a child process; the default and `--no-default-features` runs must stay green (those binaries compile to zero tests).
- Feature-dependent expectations are `#[cfg(feature = "native-v2")]` split: the same suite runs with and without the team-hosted provider.

## Routing Table

- `./registry.rs` → name resolution, registration semantics, model-config gates, context building, `create_harness`, built-in descriptors and their prerequisites (26 tests with `vendor`, 23 with the default features, 18 with `--no-default-features`)
- `./native.rs` → native card, create validation, unstarted state, start/send/events/stop over `ScriptedModel`, model control, checkpoints
- `./io_adapter.rs` → event→chunk projection, turn lifecycle/failure frames, interactions (host prompts, tool gateway, provider handlers), answer parsing, delivery modes, abort/pause/resume gating, lifecycle errors; four in-test doubles plus 20 tests make the file intentionally ~1550 lines
- `./claudecode.rs` → Claude Code card, configuration defaults/cross-field rules, binary resolution, stream-json frames, approvals, resume and turn endings over `FakeCli` (`#![cfg(unix)]`)
- `./codex.rs` → Codex card, configuration defaults, app-server JSON-RPC handshake, thread/turn events, approvals and checkpoints over `FakeCli` (`#![cfg(unix)]`)
- `./dsh.rs` → DSH card, launch contract (`DSH_HOME`, argv, patches), JSON-RPC initialize/prompt, chunk projection, turn endings, subagent items vs. `child.session.event` provider events, and the capability-gated commands over `FakeCli` (`#![cfg(unix)]`)
- `./native_v2.rs` → team-hosted provider: config/card, driver turn over `TeamAgent`, lifecycle and checkpoint deferral (feature `native-v2`)
- `./vendor.rs` → vendor cards (declared and deliberately undeclared capabilities), config decoding/round-trips, model fragments, and the `create` path (typed `Protocol` error for a missing binary, unstarted harness for a fake executable) (`vendor`; 37 tests)
- `./skills.rs` → skill normalization and conflict policies
- `./vendor_cli.rs` → real subprocess transport: NDJSON round trip, ordered replies, EOF, stderr tail, broken pipe, malformed/over-long lines, `kill` reaping, args/cwd, cleared environment (`#![cfg(unix)]`)
- `./vendor_rpc.rs` → real subprocess JSON-RPC path over `FakeCli`: request round trips and transcript, an unsolicited notification through `next_incoming`, a peer error with its code, notifications without an id, EOF failing the pending request, a skipped unreadable line (`#![cfg(unix)]`)
- `./vendor_supervisor.rs` → `VendorHarness` over an in-test `FakeDriver`: turn sequence, queueing, steering, abort gating, pause/resume, checkpoints, idempotent stop
- `./support/mod.rs` → shared `FakeCli` fixture (scripted POSIX shell CLI with a stdin transcript; helper-written, atomically installed script)
