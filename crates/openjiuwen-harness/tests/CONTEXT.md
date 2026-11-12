# harness integration tests

## Intent

Offline, deterministic integration suites for `openjiuwen-harness`: they drive the public API end to end with `openjiuwen_test_support::ScriptedModel`, temp workspaces, and fake permission/telemetry hosts.
Unit tests live beside their modules; these cover cross-module behavior.

## API Surface

Test files (one concern each):
- `deep_agent_loop.rs` — no-tool answer, one tool round, permission denial, interrupt/resume, max-iteration error, follow-up, abort.
- `workspace_safety.rs` — default schema seeding, traversal/absolute/drive rejection, `DirectoryBuilder` idempotency.
- `permissions.rs` — allow/ask/deny precedence, session-scoped approvals, file guard path folds.
- `state_persistence.rs` — `DeepAgentState` session round-trip, pending follow-ups, plan mode.
- `subagents_skills.rs` — subagent spawn/wait via a fake factory, skill discovery/mount/budget.
- `protocol.rs` — `NativeHarness` lifecycle, capability gating, cursors, checkpoint export.
- `cli.rs` — clap parsing, `run` with a scripted model, permission prompt answers.

## Constraints

- No network, no sleeps longer than a few milliseconds, no reliance on wall-clock ordering; use `ScriptedModel` and `tokio::test`.
- Every test creates its own `tempfile::TempDir` workspace; never writes outside it.
- Assert persisted JSON keys exactly where parity matters (`deepagent`, `todos`, `activeForm`).
- No `unwrap` in async tasks that could mask panics; prefer `expect("message")` in tests.
- Feature-gated areas (browser/cua/mobile) assert `Error::Unsupported` only.

## Routing Table

- `./deep_agent_loop.rs` → loop, tools, interrupts
- `./workspace_safety.rs` → workspace and path safety
- `./permissions.rs` → permission engine and rail
- `./state_persistence.rs` → session state round-trip
- `./subagents_skills.rs` → subagents and skills
- `./protocol.rs` → native harness protocol
- `./cli.rs` → CLI parsing and run

## Test Strategy

- Run with `cargo test -p openjiuwen-harness` (default features) and `--no-default-features`; both must pass.
- Tests must not depend on `notify`, `browser`, `cua`, or `mobile` features.
