# openjiuwen

## Intent

Top-level umbrella crate: the Rust counterpart of the Python package root `openjiuwen`.
It is a pure facade — module aliases onto concrete crates plus one curated `prelude` — and owns no logic or data types of its own.
Consumers use it for Python-shaped paths (`openjiuwen::harness::DeepAgent`); workspace-internal code depends on the concrete crate instead, which keeps the dependency graph acyclic.
The concrete crates stay canonical for documentation, semver, behavior and tests; this crate only re-exports.

## API Surface

### Module aliases

| Python | Rust | Source crate |
| --- | --- | --- |
| `openjiuwen.core` | `core` | `openjiuwen-core` (kernel umbrella, including its `single_agent` / `application` shims) |
| `openjiuwen.harness` | `harness` | `openjiuwen-harness` |
| `openjiuwen.harness_protocol` | `harness_protocol` | `openjiuwen-harness-protocol` |
| `openjiuwen.harness_providers` | `harness_providers` | `openjiuwen-harness-providers` |
| `openjiuwen.agent_teams` | `agent_teams` | `openjiuwen-agent-teams` |
| `openjiuwen.extensions` | `extensions` | `openjiuwen-extensions` |
| `openjiuwen.dev_tools` | `dev_tools` | `openjiuwen-dev-tools` |
| `openjiuwen.symphony` | `symphony` | `openjiuwen-symphony` |
| `openjiuwen.rsi` | `rsi` | `openjiuwen-rsi` |
| `openjiuwen.agent_evolving` | `agent_evolving` | `openjiuwen-agent-evolving` |
| `openjiuwen.auto_harness` | `auto_harness` | `openjiuwen-auto-harness` (shim over `rsi::auto_harness`) |

Every alias names the same item as the concrete crate; `tests/module_paths.rs` proves identity via `TypeId` and function-pointer equality.
`crate_name()` returns `"openjiuwen"` and is `#[must_use] pub const fn`.

### prelude

`openjiuwen::prelude` re-exports the most-used items for `use openjiuwen::prelude::*;`.
It is convenience only: no new types, no behavior, and no stability promise beyond the items it forwards.

| Item | Canonical path |
| --- | --- |
| `Error`, `Result` | `core::common::error` |
| `Agent`, `AgentCard` | `core::single_agent` |
| `Session`, `AgentSession` | `core::session` |
| `Model`, `BaseMessage` | `core::foundation` |
| `Tool`, `FunctionTool` | `core::foundation` |
| `CompiledGraph` | `core::graph` |
| `Workflow` | `core::workflow` |
| `ContextEngine` | `core::context_engine` |
| `LongTermMemory` | `core::memory` |
| `Runner` | `core::runner` |
| `DeepAgent`, `DeepAgentConfig`, `create_deep_agent` | `harness` |
| `HarnessCard`, `HarnessProtocol` | `harness_protocol` |
| `create_harness`, `HarnessIOAdapter` | `harness_providers` |
| `TeamAgent`, `TeamAgentSpec` | `agent_teams` |
| `A2aClient` (feature `a2a`) | `extensions::a2a::client` |
| `OtelTracerConfig` (feature `otel`) | `extensions::otel` |

`AgentSession` is the in-memory session implementation (`openjiuwen-session`); there is no separate `InMemorySession` type in the Rust API.

### Feature matrix

| Feature | Default | Enables | Effect |
| --- | --- | --- | --- |
| `native` | yes | `openjiuwen-harness/native`, `openjiuwen-harness-providers/native` | in-process native `DeepAgent` harness and provider |
| `otel` | no | `openjiuwen-extensions/otel` | live OpenTelemetry exporters and sink |
| `a2a` | no | `openjiuwen-extensions/a2a` | Agent-to-Agent client/server modules |
| `sqlite` | no | `openjiuwen-extensions/sqlite`, `openjiuwen-symphony/sqlite` | optional SQLite checkpointer, stores, and asset store |
| `live-tests` | no | `openjiuwen-harness/live-tests`, `openjiuwen-harness-providers/live-tests`, `openjiuwen-agent-teams/live-tests`, `openjiuwen-extensions/live-tests` | opt-in network/vendor test gates |

The default build needs no network and no external service: every feature above is additive and off by default except `native`.

## Constraints

- Re-exports only: no logic, no new public data types, no `unsafe` (`#![forbid(unsafe_code)]`); the `thiserror`/`serde` stack is not a dependency of this crate.
- `publish = false` until the SDK API stabilizes, and no Sibling or `docs/` edits are made from here.
- Feature names in this crate map to existing sibling features; adding a feature requires verifying the sibling's feature name first and keeping the default build offline.
- Python package spelling (`harness_protocol`, `agent_teams`, `sys_operation`) is mirrored even where it repeats the crate suffix, so `clippy::module_name_repetitions` is allowed crate-wide.
- The crate-level doc example is the "getting started" block: it must compile as a doctest and touch no network.
- Concrete crates are canonical: documentation lives there; this file documents only the facade.

## Routing Table

- `./src/lib.rs` → crate docs (Python→Rust table, feature matrix, getting-started doctest), module aliases, `crate_name()`
- `./src/prelude.rs` → curated convenience re-exports and the "prelude is convenience only" note
- `./Cargo.toml` → dependency list and the feature mapping table
- `./tests/module_paths.rs` → identity checks for one representative item per module alias, plus feature-gated `a2a` / `otel` checks

## Test Strategy

- `tests/module_paths.rs` asserts `TypeId` identity (and `std::ptr::eq` for functions) between each alias and the concrete crate, so a mis-pointed alias fails loudly; `a2a` and `otel` items are behind `#[cfg(feature = ...)]`.
- The crate-level getting-started example is the doctest that guards the primary public paths (`cargo test -p openjiuwen`).
- No unit tests: the facade has no behavior; behavior is tested in the concrete crates.
- Definition of done: `cargo fmt -p openjiuwen`, `cargo test -p openjiuwen`, `cargo test -p openjiuwen --features otel,a2a`, `cargo clippy -p openjiuwen --all-targets -- -D warnings` all clean.
