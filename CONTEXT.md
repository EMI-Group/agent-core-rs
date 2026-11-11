# openJiuwen (Rust) — root

## Intent

Rust rewrite of the openJiuwen Python agent SDK (reference: read-only foreign repo `/home/bill/Source/agent-core`, v0.1.20).
The objective is functional and interface parity with the Python SDK's public surface: agent runtime, LLM clients, prompts, tools/MCP, sessions and checkpoints, graph/workflow engine, context engine, memory, retrieval, sys-ops, guardrails, multi-agent teams, the DeepAgent (harness) framework, and the `openjiuwen` CLI.
Internal implementation is idiomatic Rust (tokio, serde, traits, thiserror), not a transliteration of Python internals.
Python package `openjiuwen.<pkg>` maps to crate `openjiuwen-<pkg>`; `openjiuwen-core` and `openjiuwen` are umbrella crates preserving the module paths.

## API Surface

- `openjiuwen` — top-level umbrella: `core`, `harness`, `agent_teams`, `extensions`, `dev_tools`, `harness_protocol`, `harness_providers`, `symphony`, `rsi`, `agent_evolving`, `auto_harness`.
- `openjiuwen-core` — kernel umbrella: `common`, `foundation`, `session`, `graph`, `workflow`, `context_engine`, `memory`, `retrieval`, `sys_operation`, `security`, `single_agent`, `application`, `multi_agent`, `runner`.
- `openjiuwen-cli` — binary `openjiuwen` (run/repl front-end).
- `crates/` — every workspace crate, layering, and per-crate routing.
- `examples/` — runnable examples mirroring the Python `examples/` tree.

## Constraints

- **Compatibility contract**: preserve OpenAI-compatible and Anthropic wire formats; `*Card` JSON field names and shapes; workflow `${node.field}` references; `OPENJIUWEN_*`, `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_*` environment variables; and `openjiuwen` CLI commands. See `docs/PARITY.md`.
- **Async-first**: tokio runtime; no blocking calls on async tasks; public streams are `Send + 'static`; cancellation via `tokio_util::sync::CancellationToken`.
- **Errors**: `openjiuwen_common::error::{Error, Result}` across libraries; no panics on user input; `anyhow` only in binaries, examples, and tests.
- **Serialization**: every public data type is `serde`-serializable, snake_case, tolerant of unknown fields; Cards round-trip through JSON and expose JSON Schema via `schemars`.
- **Logging**: `tracing` through `openjiuwen_common::logging`; no `println!`/`eprintln!` in libraries.
- **Safety**: no `unsafe` (workspace lint `deny`); no undocumented global mutable state (the process-global `Runner` is the documented exception, with local instances supported for tests).
- **Size**: files around ≤1000 lines; split by responsibility; shared types live at the lowest common ancestor crate.
- **Tests**: unit tests beside modules and in `crates/<crate>/tests/`; cross-crate tests in `openjiuwen-e2e`; fake LLM server in `openjiuwen-test-support`; `cargo test --workspace` passes offline.
- **Foreign repo** is read-only; never modify it. Delegate investigation to read-only investigators.

## Routing Table

- `./crates/` → workspace crates, dependency layers, and per-crate responsibilities (`crates/CONTEXT.md`)
- `./docs/` → architecture and parity documentation (`docs/CONTEXT.md`)
- `./crates/openjiuwen-examples/` → runnable examples mirroring Python `examples/` (`crates/CONTEXT.md`)
- `./Cargo.toml` → workspace manifest, shared dependency versions, lints
- `./Makefile` → build/test/lint entry points
