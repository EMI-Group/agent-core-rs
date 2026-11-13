# openjiuwen-agent source tree

## Intent

Module tree of the `openjiuwen-agent` crate: the `Agent` seam, the `ReAct` loop, abilities, rails, interrupts, the application agents (`LLMAgent`, `WorkflowAgent`), the tool policy layer, callbacks, steering, skills, the controller stack, KV-cache affinity, tunable operators, self-evolution, and YAML prompt packs.

## Module Layout

| Module | Role |
| --- | --- |
| `agent.rs` | the `Agent` trait |
| `card.rs` | `AgentCard` |
| `input.rs` | `AgentInput` parsing |
| `output.rs` | `AgentOutput`, `AgentChunk`, `Observation`, `OutputFrame`, frame-kind constants |
| `prompt.rs` | `SystemPromptBuilder` and `PromptSection` |
| `run.rs` | session acquisition/commit |
| `interrupt.rs` | interrupt data model and resume matching |
| `abilities/` | tool registry, argument repair, tool execution |
| `rails/` | rail trait/context/registry and built-in rails |
| `react/` | `ReActAgent`, config, iteration loop, model step, resume replay, loop hooks (callbacks/steering/final check), frame emission, tool-exposure projection, wiring behaviour tests |
| `llm_agent.rs` | `LlmAgent` over `ReActAgent` |
| `workflow_agent/` | `WorkflowAgent` and workflow selection |
| `callbacks/` | lifecycle callback events, payloads, `CallbackFramework` |
| `steering/` | `SteeringMessage`, `SteeringQueue` |
| `policy/` | tool exposure policy, MCP allowlist, client tools, resilience, `FinalRequestCheck` |
| `skills/` | skill definitions, discovery, prompt injection |
| `controller/` | controller stack routing events to agents/workflows |
| `kv_cache/` | cache identity, runtime, metadata, model-call affinity hook |
| `operator/` | tunable operators for LLM/tool/memory/context-evolve calls |
| `evolve/` | `ReActAgentEvolve` operator seam |
| `prompt_pack/` | YAML prompt packs |

## Constraints

- Each module owns its types; shared wire types live in `output.rs`, `input.rs`, `interrupt.rs`, or `openjiuwen-foundation`.
- The `ReAct` loop is the only place that sequences model and tool steps; policy/callback/steering modules expose pure seams the loop calls.
- Files stay around ≤1000 lines.

## Notes for Agents

- The seams are consumed, not duplicated: `react/hooks.rs` fires callbacks and drains steering, `react/exposure.rs` projects the model-visible tools, `react/frames.rs` builds the `context.usage`/`llm_route` payloads, `react/pause.rs` persists an intercepted turn, and `react/resume.rs` replays it (including client tools).
- Every seam defaults to the pre-wiring behavior (empty callback framework, empty steering queue, allow-all exposure, no MCP restriction, empty client registry, single-attempt resilience, no final check); a caller that configures nothing must keep observing today's loop, which the integration suites `react_agent`, `rails`, `interrupt`, `streaming`, `workflow_agent`, and `agent_card` enforce.
- `crate::callbacks`, `crate::steering`, and `crate::policy` own their types and are read-only from `react/`; new loop behavior goes through `react/hooks.rs`, `react/exposure.rs`, `react/frames.rs`, `react/pause.rs`, or `react/resume.rs`.

## Routing Table

- `./react/` → `ReActAgent`, config, loop, model step, hooks, frames, exposure, pause/resume (`react/CONTEXT.md`)
- `./abilities/` → ability manager and tool execution (`abilities/CONTEXT.md`)
- `./rails/` → rail seam and built-in rails (`rails/CONTEXT.md`)
- `./workflow_agent/` → `WorkflowAgent` (`workflow_agent/CONTEXT.md`)
- `./callbacks/` → callback framework (`callbacks/CONTEXT.md`)
- `./steering/` → steering queue (`steering/CONTEXT.md`)
- `./policy/` → tool policy, client tools, resilience, final request check (`policy/CONTEXT.md`)
- `./skills/` → skill utilities and prompt injection
- `./controller/` → controller stack (event → intent → task executor)
- `./kv_cache/` → KV-cache affinity types and in-process runtime
- `./operator/` → tunable operators
- `./evolve/` → `ReActAgentEvolve`
- `./prompt_pack/` → YAML prompt packs
