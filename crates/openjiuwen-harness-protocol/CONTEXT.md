# openjiuwen-harness-protocol

## Intent

Provider-neutral harness SPI with traits, events, results, and streams.

## API Surface

`openjiuwen-harness-protocol` exposes the `openjiuwen_harness_protocol` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
