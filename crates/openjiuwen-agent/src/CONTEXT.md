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
| `react/` | `ReActAgent`, config, iteration loop, model step, resume replay |
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

## Routing Table

- `./react/` → `ReActAgent`, config, loop, model step, resume (`react/CONTEXT.md`)
- `./abilities/` → ability manager and tool execution (`abilities/CONTEXT.md`)
- `./rails/` → rail seam and built-in rails (`rails/CONTEXT.md`)
- `./workflow_agent/` → `WorkflowAgent` (`workflow_agent/CONTEXT.md`)
- `./callbacks/` → callback framework (`callbacks/CONTEXT.md`)
- `./steering/` → steering queue (`steering/CONTEXT.md`)
- `./policy/` → tool policy, client tools, resilience, final request check
- `./skills/` → skill utilities and prompt injection
- `./controller/` → controller stack (event → intent → task executor)
- `./kv_cache/` → KV-cache affinity types and in-process runtime
- `./operator/` → tunable operators
- `./evolve/` → `ReActAgentEvolve`
- `./prompt_pack/` → YAML prompt packs
