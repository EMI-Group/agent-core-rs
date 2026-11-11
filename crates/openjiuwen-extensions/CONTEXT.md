# openjiuwen-extensions

## Intent

A2A, OTel tracer, and checkpointer/store/MQ/sysop adapters behind feature gates.

## API Surface

`openjiuwen-extensions` exposes the `openjiuwen_extensions` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
