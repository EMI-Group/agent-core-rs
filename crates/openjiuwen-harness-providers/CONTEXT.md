# openjiuwen-harness-providers

## Intent

Harness provider registry where `native` is implemented and vendor providers are deferred.

## API Surface

`openjiuwen-harness-providers` exposes the `openjiuwen_harness_providers` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
