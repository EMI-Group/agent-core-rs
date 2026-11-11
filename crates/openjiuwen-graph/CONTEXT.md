# openjiuwen-graph

## Intent

Pregel/BSP graph execution engine.

## API Surface

`openjiuwen-graph` exposes the `openjiuwen_graph` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
