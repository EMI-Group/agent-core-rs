# openjiuwen-symphony

## Intent

Deferred scope placeholder for the capability-asset engine per `docs/PARITY.md`.

## API Surface

`openjiuwen-symphony` exposes the `openjiuwen_symphony` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
