# openjiuwen-test-support

## Intent

Mock LLM server, fakes, and fixtures for tests.

## API Surface

`openjiuwen-test-support` exposes the `openjiuwen_test_support` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
