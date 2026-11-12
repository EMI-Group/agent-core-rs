# openjiuwen-core

## Intent

Kernel umbrella crate that mirrors the Python `openjiuwen.core` package tree with Rust module paths.
It is a pure re-export facade: it implements no behavior and owns no data types.
The concrete kernel crates remain the canonical home for every item's implementation, docs, and tests.
Consumers use this crate when they want Python-shaped paths; workspace-internal code should depend on the concrete crate instead so the dependency graph stays acyclic.

## Module Mapping

| Python path | Rust path | Source crate |
| --- | --- | --- |
| `openjiuwen.core.common` | `openjiuwen_core::common` | `openjiuwen-common` |
| `openjiuwen.core.foundation` | `openjiuwen_core::foundation` | `openjiuwen-foundation` |
| `openjiuwen.core.session` | `openjiuwen_core::session` | `openjiuwen-session` |
| `openjiuwen.core.graph` | `openjiuwen_core::graph` | `openjiuwen-graph` |
| `openjiuwen.core.workflow` | `openjiuwen_core::workflow` | `openjiuwen-workflow` |
| `openjiuwen.core.context_engine` | `openjiuwen_core::context_engine` | `openjiuwen-context` |
| `openjiuwen.core.memory` | `openjiuwen_core::memory` | `openjiuwen-memory` |
| `openjiuwen.core.retrieval` | `openjiuwen_core::retrieval` | `openjiuwen-retrieval` |
| `openjiuwen.core.sys_operation` | `openjiuwen_core::sys_operation` | `openjiuwen-sysop` |
| `openjiuwen.core.security` | `openjiuwen_core::security` | `openjiuwen-security` |
| `openjiuwen.core.single_agent` | `openjiuwen_core::single_agent` | `openjiuwen-agent` (glob re-export) |
| `openjiuwen.core.application` | `openjiuwen_core::application` | `openjiuwen-agent` (application-agent surface only) |
| `openjiuwen.core.multi_agent` | `openjiuwen_core::multi_agent` | `openjiuwen-multiagent` |
| `openjiuwen.core.runner` | `openjiuwen_core::runner` | `openjiuwen-runner` |

The `application` module re-exports `LLMAgent` (from `LlmAgent`), `LLMAgentConfig`, `WorkflowAgent`,
`WorkflowAgentConfig`, `DefaultResponse`, `ConstrainConfig`, `ModelConfig`, `ModelInfo`, and `create_llm_agent_config`;
it deliberately does not glob the whole agent crate.

## Constraints

- Re-exports only: no logic, no new public data types, and no `unsafe` (`#![forbid(unsafe_code)]`).
- `publish = false` until the SDK API stabilizes.
- Concrete crates must not depend on this umbrella, except the product/harness layers and above; the crate stays acyclic.
- Item identity is guaranteed: each alias is the same type as the concrete crate's item, verified by `TypeId` in tests.
- Naming follows the Python tree even where it diverges from the crate name (`context_engine` → `openjiuwen-context`, `sys_operation` → `openjiuwen-sysop`, `multi_agent` → `openjiuwen-multiagent`).

## Routing Table

- `./src/lib.rs` → module aliases, the `single_agent` and `application` shims, and the Python→Rust path table in the crate docs.
- `./tests/module_paths.rs` → resolve/identity checks for one representative item per module.

## Test Strategy

- `tests/module_paths.rs` proves every module path resolves and aliases the same type as the source crate via `TypeId`;
  `single_agent`, `application`, and the runner identity helper are covered explicitly.
- The crate-level doc example compiles as a doctest (`cargo test -p openjiuwen-core`), guarding the public paths.
- No unit tests here: the umbrella has no behavior; behavior is tested in the concrete crates.
- `cargo clippy -p openjiuwen-core --all-targets --no-deps -- -D warnings` must stay clean.
