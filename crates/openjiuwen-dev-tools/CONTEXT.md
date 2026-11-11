# openjiuwen-dev-tools

## Intent

Agent builder, prompt builder, skill creator/evaluator, and tuning helpers.

## API Surface

`openjiuwen-dev-tools` exposes the `openjiuwen_dev_tools` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
