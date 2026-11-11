# openjiuwen-macros

## Intent

Proc macros for `#[tool]` and schema derivation helpers.

## API Surface

`openjiuwen-macros` exposes the `openjiuwen_macros` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
