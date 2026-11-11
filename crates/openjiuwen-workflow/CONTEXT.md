# openjiuwen-workflow

## Intent

Workflow builder, components, and `` data flow.

## API Surface

`openjiuwen-workflow` exposes the `openjiuwen_workflow` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
