# openjiuwen-examples

## Intent

Runnable examples mirroring the Python `examples/` tree.

## API Surface

The crate is a scaffold stub: `openjiuwen_examples::crate_name() -> &'static str` is its only item.
`examples/` contains only `.gitkeep` — no runnable example targets exist yet.

## Constraints

- Depends on the `openjiuwen` umbrella (`openjiuwen.workspace = true`, default features → `native`), plus `tokio`, `serde_json`, `anyhow`, `dotenvy`, `futures`; `openjiuwen-harness-providers` is reachable only transitively.
- `default = []` — the crate declares no features of its own.
- Umbrella features that exist: `native` (default), `otel`, `a2a`, `sqlite`, `live-tests`; there is **no** `mcp` feature — `openjiuwen-foundation`'s `mcp` feature is default-on, so MCP is always reachable.
- Reaching an optional umbrella feature (e.g. `a2a`) needs both `openjiuwen = { workspace = true, features = ["a2a"] }` and a forwarding entry in this crate's `[features]`.
- `openjiuwen::prelude::*` exports `Agent`, `AgentCard`, `AgentSession`, `Session`, `BaseMessage`, `FunctionTool`, `Model`, `Tool`, `ContextEngine`, `Workflow`, `LongTermMemory`, `Runner`, `create_deep_agent`, `DeepAgent`, `DeepAgentConfig`, `HarnessCard`, `HarnessProtocol`, `create_harness`, `HarnessIOAdapter`, `TeamAgent`, `TeamAgentSpec`; it does **not** export `ReActAgent`, `ReActAgentConfig`, `ModelClientConfig`, `ModelRequestConfig`, or any retrieval type.
- The `#[tool]` macro is not re-exported anywhere: an example using it needs `openjiuwen-macros`, `schemars`, `serde`, `async-trait` and either `openjiuwen-foundation`/`openjiuwen-common` or `#[tool(crate = "::openjiuwen::core::foundation", common_crate = "::openjiuwen::core::common")]`.
- `openjiuwen-test-support` (`ScriptedModel`, `MockLlmServer`) is dev-layer only — library crates use it as `[dev-dependencies]`, and `openjiuwen-e2e` is the sole `[dependencies]` precedent; an example target may take it as a dev-dependency.
- `make clippy` runs `cargo clippy --workspace --all-targets -- -D warnings` and workspace lints warn on `unwrap_used`/`expect_used`, so an example needs `#![allow(clippy::unwrap_used, clippy::expect_used)]` or explicit error handling.
- Examples are outside the workspace `default-members`, so `cargo build` skips them while `--workspace`/`--all-targets` include them.
- Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging (examples may use `anyhow`).

## Routing Table

- `./src/` → `lib.rs` only (crate-name stub); no child routing.
- `./examples/` → empty except `.gitkeep`; examples added here become `cargo` example targets.
