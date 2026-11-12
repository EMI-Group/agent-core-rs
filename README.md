# openJiuwen — Rust

A Rust rewrite of the [openJiuwen](https://github.com/openjiuwen/openjiuwen) Python agent SDK.
The goal is behavioral parity with the Python SDK's public surface — agent runtime, LLM clients,
prompts, tools and MCP, sessions and checkpoints, the graph/workflow engine, context engine,
memory, retrieval, sys-ops, guardrails, multi-agent teams, the DeepAgent (harness) framework,
and the `openjiuwen` CLI — while using idiomatic Rust internally.

## Status

The kernel, DeepAgent harness, harness providers, in-process agent teams, A2A/OTel extensions, dev tools, the symphony offline engine, auto-harness/RSI, agent-evolving, the umbrella crate, and the `openjiuwen` binary are implemented as documented subsets.
Features that depend on Python research/ML stacks, vendor SDKs, external databases or brokers, or desktop/mobile transports are not reimplemented; their API-shaped entry points return `Error::Unsupported` with a reason.
`docs/PARITY.md` carries the per-package status matrix, wire-compatibility notes, intentional differences, and the complete deferred list.

## Quick start

```bash
cargo run -p openjiuwen-cli -- --version
cargo run -p openjiuwen-cli -- run "Summarize this repository" --model gpt-4o --provider openai
cargo run -p openjiuwen-examples --example 01_hello_agent   # offline, scripted model
cargo run -p openjiuwen-examples --example 08_deep_agent    # offline DeepAgent in a temp workspace
```

The CLI and examples run offline by default; a live provider is selected by the flags or by `OPENJIUWEN_API_KEY`, `OPENJIUWEN_API_BASE`, `OPENJIUWEN_MODEL`, and `OPENJIUWEN_PROVIDER` (the examples additionally require `OPENJIUWEN_LIVE=1`).
`cargo install --path crates/openjiuwen-cli` installs the `openjiuwen` binary; `team-member` and `openjiuwen-team-mcp` are built from `crates/openjiuwen-agent-teams`.

## Layout

The Cargo workspace mirrors the Python package tree:

| Python package | Rust crate |
|---|---|
| `openjiuwen.core.common` | `openjiuwen-common` |
| `openjiuwen.core.foundation` | `openjiuwen-foundation` |
| `openjiuwen.core.session` | `openjiuwen-session` |
| `openjiuwen.core.graph` | `openjiuwen-graph` |
| `openjiuwen.core.workflow` | `openjiuwen-workflow` |
| `openjiuwen.core.context_engine` | `openjiuwen-context` |
| `openjiuwen.core.memory` | `openjiuwen-memory` |
| `openjiuwen.core.retrieval` | `openjiuwen-retrieval` |
| `openjiuwen.core.sys_operation` | `openjiuwen-sysop` |
| `openjiuwen.core.security` | `openjiuwen-security` |
| `openjiuwen.core.single_agent` + `core.application` | `openjiuwen-agent` |
| `openjiuwen.core.multi_agent` | `openjiuwen-multiagent` |
| `openjiuwen.core.runner` | `openjiuwen-runner` |
| (kernel umbrella) | `openjiuwen-core` |
| `openjiuwen.harness_protocol` | `openjiuwen-harness-protocol` |
| `openjiuwen.harness` | `openjiuwen-harness` |
| `openjiuwen.harness_providers` | `openjiuwen-harness-providers` |
| `openjiuwen.agent_teams` | `openjiuwen-agent-teams` |
| `openjiuwen.extensions` | `openjiuwen-extensions` |
| `openjiuwen.dev_tools` | `openjiuwen-dev-tools` |
| `openjiuwen.symphony` | `openjiuwen-symphony` |
| `openjiuwen.rsi` | `openjiuwen-rsi` |
| `openjiuwen.agent_evolving` | `openjiuwen-agent-evolving` |
| `openjiuwen.auto_harness` | `openjiuwen-auto-harness` |
| `openjiuwen` (top level) | `openjiuwen` |

`openjiuwen-core` and `openjiuwen` are umbrella crates that re-export the module trees, so
`openjiuwen::core::foundation::llm::Model` corresponds to `openjiuwen.core.foundation.llm.Model`.

## Build and test

```bash
cargo build --workspace
cargo test --workspace          # no network or external services required
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
make verify                     # the full gate matrix, including feature-gated suites
```

`make test-all` runs the optional feature suites (A2A, OTel, SQLite, sandbox, vendor providers, no-default-features builds), and `make examples` compiles all runnable examples.
The cross-crate scenarios live in `openjiuwen-e2e`; the 12 runnable examples live in `crates/openjiuwen-examples/examples/`.
Optional integration tests that call real model providers are gated behind the `live-tests` feature and are never run by default.

## Documentation

- `docs/ARCHITECTURE.md` — crate layering, shared contracts, conventions.
- `docs/PARITY.md` — Python → Rust feature matrix, deviations, and explicitly deferred scope.
- Each crate and directory has a `CONTEXT.md` with intent, API surface, constraints, and routing.

## License

Apache-2.0. See `LICENSE`.
