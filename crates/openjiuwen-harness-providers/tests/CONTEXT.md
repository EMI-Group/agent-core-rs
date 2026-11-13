# Tests

## Intent

Offline integration suites for the provider registry, the native provider wiring, the IO adapter, the vendor stubs, the skill installer, and the shared vendor CLI subprocess transport.
Behavior ports name their Python counterpart (`tests/unit_tests/harness_providers/<file>::<test>`) in a comment.

## API Surface

No public API; the suites exercise `openjiuwen_harness_providers` through its public surface and the public `openjiuwen_harness` / `openjiuwen_harness_protocol` types.

## Constraints

- Offline and deterministic: no network, no external CLIs, no sleeps (bounded `tokio::time::timeout` awaits only), `tempfile::tempdir` for every filesystem case.
- Test doubles (`FakeHarness`, `FakeProvider`, `FakeSink`, `FakeToolGateway`) live inside the test files and double as usage examples.
- `support::FakeCli` is the one shared fixture: a temp-directory POSIX shell script with a stdin transcript, included per binary with `mod support;` (the module carries `#![allow(dead_code)]` and `#![allow(unreachable_pub)]`).
- Registry tests use globally unique provider names instead of a reset hook, because the registry is process-global.
- `#![allow(clippy::unwrap_used, clippy::expect_used)]` at the top of each suite (assert-by-panic is the intent).
- Vendor suites are `#![cfg(feature = "vendor")]`; the default suite must pass with `--no-default-features` too.

## Routing Table

- `./registry.rs` → name resolution, registration semantics, model-config gates, context building, `create_harness`
- `./native.rs` → native card, create validation, unstarted state, start/send/events/stop over `ScriptedModel`, model control, checkpoints
- `./io_adapter.rs` → event→chunk projection, turn lifecycle/failure frames, interactions (host prompts, tool gateway, provider handlers), answer parsing, delivery modes, abort/pause/resume gating, lifecycle errors; four in-test doubles plus 20 tests make the file intentionally ~1550 lines
- `./vendor.rs` → vendor cards/configs/model fragments/`Unsupported` create (feature `vendor`)
- `./skills.rs` → skill normalization and conflict policies
- `./vendor_cli.rs` → real subprocess transport: NDJSON round trip, ordered replies, EOF, stderr tail, broken pipe, malformed/over-long lines, `kill` reaping, args/cwd, cleared environment (`#![cfg(unix)]`)
- `./support/mod.rs` → shared `FakeCli` fixture (scripted POSIX shell CLI with a stdin transcript)
