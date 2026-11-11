# src — crate internals

## Intent

Source tree of `openjiuwen-foundation`, split into four independent provider seams.
Each module owns its own `CONTEXT.md` and its own submodules; cross-module dependencies point one way: `prompt` ← `llm`, `prompt` ← `tool`, and `store` stands alone.
`lib.rs` (owned by the crate architect) declares the modules and re-exports the crate-level seams.

## API Surface

`lib.rs` re-exports, for call sites that should not care about module paths: `BaseMessage`, `MessageContent`, `ContentPart`, `Role`, `ToolCall`, `Usage`, `PromptTemplate`, `Model`, `ModelClientConfig`, `ModelRequestConfig`, `ChatResponse`, `StreamChunk`, `FinishReason`, `init_model`, `Tool`, `ToolCard`, `ToolContext`, `FunctionTool`, `ToolOutput`, and the five storage traits.

## Constraints

- `prompt` depends only on `openjiuwen-common`; `llm`, `tool`, and `store` may depend on `prompt` but `prompt` never on them.
- No module re-exports another module's internals: each `mod.rs` (or module root) exposes its own `pub use` list.
- Every public item needs a doc comment (`missing_docs` is a workspace lint and clippy runs with `-D warnings`).
- Crate-level `#![allow(...)]` in `lib.rs` covers pedantic noise (`missing_errors_doc`, `module_name_repetitions`, cast lints, ...); module code must be clippy-clean otherwise.
- `#[cfg(test)]` modules may use `unwrap`/`expect` (allowed in `lib.rs` for the test cfg); integration tests in `tests/` add their own allow attribute.

## Routing Table

- `./prompt/` → roles, content parts, messages, tool calls, usage, prompt templates, token helpers
- `./llm/` → configs, `Model` trait, provider table/registry/factory, SSE decoder, OpenAI-compatible and Anthropic clients
- `./tool/` → tool cards, `Tool` contract, function tools, JSON-Schema helpers, REST tool, MCP
- `./store/` → kv, document, vector, graph, and object storage contracts and backends
