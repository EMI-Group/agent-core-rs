# openjiuwen-agent

## Intent

`BaseAgent`, `ReActAgent`, abilities, rails, interrupts, `LLMAgent`, and `WorkflowAgent`.

## API Surface

`openjiuwen-agent` exposes the `openjiuwen_agent` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
