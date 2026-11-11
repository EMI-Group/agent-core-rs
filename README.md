# openJiuwen — Rust

A Rust rewrite of the [openJiuwen](https://github.com/openjiuwen/openjiuwen) Python agent SDK.
The goal is behavioral parity with the Python SDK's public surface — agent runtime, LLM clients,
prompts, tools and MCP, sessions and checkpoints, the graph/workflow engine, context engine,
memory, retrieval, sys-ops, guardrails, multi-agent teams, the DeepAgent (harness) framework,
and the `openjiuwen` CLI — while using idiomatic Rust internally.

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
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
```

Optional integration tests that call real model providers are gated behind the `live-tests`
feature and are never run by default.

## Documentation

- `docs/ARCHITECTURE.md` — crate layering, shared contracts, conventions.
- `docs/PARITY.md` — Python → Rust feature matrix, deviations, and explicitly deferred scope.
- Each crate and directory has a `CONTEXT.md` with intent, API surface, constraints, and routing.

## License

Apache-2.0. See `LICENSE`.
