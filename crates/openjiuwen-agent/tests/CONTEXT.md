# tests/

## Intent

Crate-level integration suites for `openjiuwen-agent`, driven by `openjiuwen-test-support::ScriptedModel` (offline, deterministic).

## API Surface

Suites (run with `cargo test -p openjiuwen-agent`):
- `agent_card.rs` → `AgentCard`/`AgentOutput`/`AgentChunk` JSON round-trips and Python wire names.
- `react_agent.rs` → no-tool answer, single tool, parallel tools, tool error as tool message, max-iteration policies (`error`, `return_observations`, `final_answer`), empty-response retry, prompt/window contents, ability mounting.
- `streaming.rs` → `llm_output`/`llm_reasoning`/`llm_usage`/`answer` frame assembly from a streamed scripted model.
- `rails.rs` → priority ordering, force-finish on model/tool phases, `ModelBackupRail` retry, `GuardrailRail` blocking, `MemoryRail` injection, skip-tool.
- `interrupt.rs` → confirm pause/approve/reject, ask-user answer, auto-confirm, resume flows.
- `llm_agent.rs` → config mapping, invoke/stream output filtering.
- `workflow_agent.rs` → single-workflow invoke/stream, multi-workflow selection, default response, interrupt/resume.

## Constraints

- Every test is offline and deterministic; no wall-clock assertions beyond short timeouts.
- Shared fixtures live in `common/mod.rs` (scripted tools, session helpers, workflow builders); no external services.
- Tests assert Python-compatible serialized names (`result_type`, `termination_reason`, `__interaction__`, `llm_output`, `workflow_final`).

## Routing Table

- `./common/` → shared fixtures (tools, sessions, workflows)
