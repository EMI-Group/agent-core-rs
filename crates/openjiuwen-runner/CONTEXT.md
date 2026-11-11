# openjiuwen-runner

## Intent

`Runner`, `ResourceMgr`, callbacks, and the MQ abstraction.

## API Surface

`openjiuwen-runner` exposes the `openjiuwen_runner` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
