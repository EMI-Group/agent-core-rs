# openjiuwen-harness

## Intent

DeepAgent task loop, rails, workspace, permissions, skills, subagents, prompts, and CLI logic.

## API Surface

`openjiuwen-harness` exposes the `openjiuwen_harness` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
