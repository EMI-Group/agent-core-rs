# openjiuwen-auto-harness

## Intent

Re-export shim over `rsi::auto_harness`.

## API Surface

`openjiuwen-auto-harness` exposes the `openjiuwen_auto_harness` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
