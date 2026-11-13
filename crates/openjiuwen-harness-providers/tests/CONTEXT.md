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
- Test doubles (`FakeHarness`, `FakeProvider`, `FakeSink`, `FakeToolGateway`, `FakeDriver`, `FakeCli`-driven peers) live inside the test files and double as usage examples; `native_v2`'s `FakeHarnessFactory`/`ScriptedHarness` is a local restatement of the `openjiuwen-agent-teams` pattern because another crate's `tests/` module is not importable.
- `support::FakeCli` is the one shared fixture: a temp-directory POSIX shell script with a stdin transcript, included per binary with `mod support;` (the module carries `#![allow(dead_code)]` and `#![allow(unreachable_pub)]`).
- `FakeCli` stages the script as `fake-cli.staged`, chmods it and renames it onto `fake-cli`, and the bytes are written by a helper `cat` process: a write descriptor opened in the test process is copied into every concurrently forked child and makes `exec` fail with `ETXTBSY` (`Text file busy`).
- Registry tests use globally unique provider names instead of a reset hook, because the registry is process-global; the suites that also assert positional `available_providers()` order guard those cases with a per-binary mutex.
- `#![allow(clippy::unwrap_used, clippy::expect_used)]` at the top of each suite (assert-by-panic is the intent).
- Feature gating: `vendor.rs`, `claudecode.rs`, `codex.rs` are `#![cfg(feature = "vendor")]`; `dsh.rs` is `#![cfg(all(feature = "vendor", unix))]`; `vendor_cli.rs` and `vendor_rpc.rs` are `#![cfg(unix)]`; `native_v2.rs` is `#![cfg(feature = "native-v2")]`. The default and `--no-default-features` runs must stay green (gated binaries compile to zero tests).
- `claudecode.rs` and `codex.rs` spawn child processes without the `unix` gate, so they would compile on a non-Unix target and fail at runtime — a known gap recorded in the crate's `## Known Issues`.
- Feature-dependent expectations are `#[cfg(feature = "vendor")]` / `#[cfg(feature = "native-v2")]` split: the same suite runs with and without the vendors and the team-hosted provider.

## Suite Inventory

Counts as measured today (`cargo test -p openjiuwen-harness-providers --offline --no-fail-fast`).

| Suite | default | `vendor` | `vendor,native-v2` | `--no-default-features` |
|---|---|---|---|---|
| lib (`src/`) | 198 | 274 | 283 | 190 |
| `io_adapter.rs` | 20 | 20 | 20 | 20 |
| `native.rs` | 9 | 9 | 9 | 4 |
| `registry.rs` | 23 | 26 | 26 | 18 |
| `skills.rs` | 33 | 33 | 33 | 33 |
| `vendor.rs` | 0 | 37 | 37 | 0 |
| `claudecode.rs` | 0 | 12 | 12 | 0 |
| `codex.rs` | 0 | 11 | 11 | 0 |
| `dsh.rs` | 0 | 14 | 14 | 0 |
| `native_v2.rs` | 0 | 0 | 12 | 0 |
| `vendor_cli.rs` | 11 | 11 | 11 | 11 |
| `vendor_rpc.rs` | 6 | 6 | 6 | 6 |
| `vendor_supervisor.rs` | 14 | 14 | 14 | 14 |
| **total** | **314** | **467** | **488** | **296** |

## Routing Table

- `./registry.rs` → name resolution, registration semantics, model-config gates, context building, `create_harness`, built-in descriptors and their feature-dependent `implemented`/`requires`
- `./native.rs` → native card, create validation, unstarted state, start/send/events/stop over `ScriptedModel`, model control, checkpoints (group A needs `native`, group B drives `openjiuwen_harness::NativeHarness` in every feature set)
- `./io_adapter.rs` → event→chunk projection, turn lifecycle/failure frames, interactions (host prompts, tool gateway, provider handlers), answer parsing, delivery modes, abort/pause/resume gating, lifecycle errors; four in-test doubles plus 20 tests make the file intentionally ~1550 lines
- `./claudecode.rs` → Claude Code card, configuration defaults/cross-field rules, binary resolution, stream-json frames, permission round trip, interrupt, `--session-id`/`--resume` argv and turn endings over `FakeCli`
- `./codex.rs` → Codex card, configuration defaults, app-server handshake, `-c` overrides, thread/turn events, approvals, steering, interrupt, resumed thread and checkpoints over `FakeCli`
- `./dsh.rs` → DSH card, launch contract (`DSH_HOME`, argv, patches), JSON-RPC initialize/prompt, chunk projection, turn endings, subagent items vs. `child.session.event` provider events, and the capability-gated commands over `FakeCli`
- `./native_v2.rs` → team-hosted provider: config/card, driver turn over `TeamAgent`, steering/abort/pause while a run hangs, checkpoint deferral (feature `native-v2`)
- `./vendor.rs` → vendor cards (declared and deliberately undeclared capabilities), config decoding/round-trips/cross-field rules, model fragments, and the `create` path (typed `Protocol` error for a missing binary, unstarted harness for a fake executable)
- `./skills.rs` → skill normalization and conflict policies, fallback-name rules, per-vendor scan dirs, `SkillInstaller` name rules and the `dsh` flat-`*.md` normalization
- `./vendor_cli.rs` → real subprocess transport: NDJSON round trip, ordered replies, EOF, stderr tail, broken pipe, malformed/over-long lines, `kill` reaping, args/cwd, cleared environment
- `./vendor_rpc.rs` → real subprocess JSON-RPC path over `FakeCli`: request round trips and transcript, an unsolicited notification through `next_incoming`, a peer error with its code, notifications without an id, EOF failing the pending request, a skipped unreadable line
- `./vendor_supervisor.rs` → `VendorHarness` over an in-test `FakeDriver`: turn sequence, queueing, steering, abort gating, pause/resume, checkpoints, idempotent stop
- `./support/mod.rs` → shared `FakeCli` fixture (scripted POSIX shell CLI with a stdin transcript; helper-written, atomically installed script)
