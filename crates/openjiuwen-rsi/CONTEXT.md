# openjiuwen-rsi

## Intent

Deferred scope placeholder for recursive self-improvement per `docs/PARITY.md`.

## API Surface

`openjiuwen-rsi` exposes the `openjiuwen_rsi` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
