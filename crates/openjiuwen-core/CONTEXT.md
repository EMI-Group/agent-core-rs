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
| `openjiuwen.core.controller` | `openjiuwen_core::controller` | `openjiuwen-agent` (`controller` module glob) |
| `openjiuwen.core.kv_cache` | `openjiuwen_core::kv_cache` | `openjiuwen-agent` (`kv_cache` module glob) |
| `openjiuwen.core.operator` | `openjiuwen_core::operator` | `openjiuwen-agent` (`operator` module glob) |
| `openjiuwen.core.multi_agent` | `openjiuwen_core::multi_agent` | `openjiuwen-multiagent` |
| `openjiuwen.core.runner` | `openjiuwen_core::runner` | `openjiuwen-runner` |

The `application` module re-exports `LLMAgent` (from `LlmAgent`), `LLMAgentConfig`, `WorkflowAgent`,
`WorkflowAgentConfig`, `DefaultResponse`, `ConstrainConfig`, `ModelConfig`, `ModelInfo`, and `create_llm_agent_config`;
it deliberately does not glob the whole agent crate.
`single_agent` is a blanket glob of `openjiuwen-agent`, while `controller`, `kv_cache`, and `operator` are narrower
per-module globs mirroring Python's top-level packages; nested paths such as `controller::schema`, `kv_cache::runtime`,
and `operator::base` stay reachable through them, and `single_agent` also exposes the same items as an alternative spelling.

## Constraints

- Re-exports only: no logic, no new public data types, and no `unsafe` (`#![forbid(unsafe_code)]`).
- `publish = false` until the SDK API stabilizes.
- Concrete crates must not depend on this umbrella, except the product/harness layers and above; the crate stays acyclic.
- Item identity is guaranteed: each alias is the same type as the concrete crate's item, verified by `TypeId` in tests.
- Naming follows the Python tree even where it diverges from the crate name (`context_engine` → `openjiuwen-context`, `sys_operation` → `openjiuwen-sysop`, `multi_agent` → `openjiuwen-multiagent`).
- Every dependency is taken with default features only: the crate declares `default = []` plus the single opt-in `otel` feature, which forwards `openjiuwen-session/otel` so `session::install_trace_sink` really installs the sink instead of being the documented no-op.

## Routing Table

- `./src/lib.rs` → module aliases, the `single_agent`, `application`, `controller`, `kv_cache`, and `operator` shims, and the Python→Rust path table in the crate docs.
- `./tests/module_paths.rs` → resolve/identity checks for one representative item per module, plus per-area extended tests for the kernel's wider surface.

## Test Strategy

- `tests/module_paths.rs` proves every module path resolves and aliases the same type as the source crate via `TypeId`;
  `single_agent`, `application`, and `runner` (its `Runner`, `RunOptions`, `ResourceMgr`, `CallbackEvent`, and `InMemoryMessageQueue` exports, plus a `MessageQueue` trait-object coercion) are covered explicitly.
- Per-area identity tests cover the extended kernel surface: `controller_*`, `kv_cache_*`, `operator_*`, `single_agent_extended_*`, `session_extended_*`, `runner_extended_*` (including nested module paths), `foundation_extended_*`, `memory_extended_*`, and `retrieval_extended_*`.
  Traits (`TaskExecutor`, `QueryRewriter`) are proved by generic bounds on the concrete trait object; function items (`abilities::parts::extract_parts`, `install_trace_sink`, `diff_snapshots`) by fn-pointer identity, and `load_mcp_abilities` by reference only.
- The crate-level doc example compiles as a doctest (`cargo test -p openjiuwen-core`), guarding the public paths.
- `cargo test -p openjiuwen-core --features otel` proves the forwarded session feature builds and keeps the same surface.
- No unit tests here: the umbrella has no behavior; behavior is tested in the concrete crates.
- `cargo clippy -p openjiuwen-core --all-targets --all-features -- -D warnings` must stay clean.
