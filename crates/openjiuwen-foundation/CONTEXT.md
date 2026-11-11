# openjiuwen-foundation

## Intent

LLM clients and `Model`, prompts/messages, tools and MCP, and storage backends.

## API Surface

`openjiuwen-foundation` exposes the `openjiuwen_foundation` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
