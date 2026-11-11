# openjiuwen-common

## Intent

Card/Config schema, typed errors, logging, env/config, HTTP helpers, task manager, and utilities.

## API Surface

`openjiuwen-common` exposes the `openjiuwen_common` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
