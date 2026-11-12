# tests/

## Intent

Crate-level integration suites for `openjiuwen-agent`, driven by `openjiuwen-test-support::ScriptedModel` (offline, deterministic).

## API Surface

Suites (run with `cargo test -p openjiuwen-agent`):
- `agent_card.rs` (15) → `AgentCard` field set/`Card` trait/round-trip, `AgentOutput` result-type wire strings and `state` frames, absent-when-empty key shapes, Python-shaped result dicts, `AgentChunk` ↔ `StreamItem` bridge.
- `react_agent.rs` (12) → no-tool answer, single tool, parallel tools, tool error as tool message, max-iteration policies (`error`, `return_observations`, `final_answer`), empty-response retry, prompt/window contents, input/config errors.
- `streaming.rs` (4) → `llm_output`/`llm_reasoning`/`llm_usage`/`answer` frame assembly from a streamed scripted model, plus frame mirroring into a supplied session.
- `rails.rs` (7) → rails steering the running loop: priority ordering per event, force-finish before the model and before a tool, `ModelBackupRail` retry, `GuardrailRail` blocking (`ErrorKind::Security`), `MemoryRail` recall reaching the model + write-back, `skip_tool` replacement.
- `interrupt.rs` (7) → confirm pause shape and persisted state, approve resume, reject with and without feedback, session-seeded auto-confirm, ask-user answer as tool result, two pending calls pausing once and resuming individually.
- `llm_agent.rs` (7) → config mapping onto `ReActAgent`, provider fallback, invoke/stream output filtering, prompt-template override, interrupt pass-through.
- `workflow_agent.rs` (5) → single-workflow invoke/stream, model-driven selection with default-response fallback, workflow interrupt/resume, no-workflow error.

## Constraints

- Every test is offline and deterministic; no wall-clock assertions beyond short timeouts and no external services.
- Shared fixtures live in `common/mod.rs` (`SESSION_ID`, session/agent builders, recording and failing tools, the input log); suites that need bespoke rails, stores, or workflows keep those helpers local to the file.
- Tests assert Python-compatible serialized names (`result_type`, `termination_reason`, `interrupt_kind`/`interrupt_owner`, `__interaction__`, `llm_output`, `workflow_final`).

## Routing Table

- `./common/` → shared fixtures (tools, sessions, workflows)
