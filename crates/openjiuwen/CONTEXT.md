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
| `ReActAgent`, `ReActAgentConfig` | `core::single_agent` (`openjiuwen_agent::react`) |
| `AgentInput`, `AgentOutput`, `AgentChunk`, `ResultType` | `core::single_agent` (`openjiuwen_agent::{input, output}`) |
| `LlmAgent`, `WorkflowAgent` | `core::single_agent` (`openjiuwen_agent::{llm_agent, workflow_agent}`) |
| `Session`, `AgentSession` | `core::session` |
| `Model`, `BaseMessage` | `core::foundation` |
| `Tool`, `FunctionTool` | `core::foundation` |
| `CompiledGraph` | `core::graph` |
| `Workflow` | `core::workflow` |
| `ContextEngine` | `core::context_engine` |
| `LongTermMemory` | `core::memory` |
| `Runner` | `core::runner` |
| `DeepAgent`, `DeepAgentConfig`, `DeepAgentBuilder`, `create_deep_agent` | `harness` |
| `HarnessCard`, `HarnessProtocol` | `harness_protocol` |
| `create_harness`, `HarnessIOAdapter` | `harness_providers` |
| `TeamAgent`, `TeamAgentSpec` | `agent_teams` |
| `A2aClient` (feature `a2a`) | `extensions::a2a::client` |
| `OtelTracerConfig` (feature `otel`) | `extensions::otel` |

`AgentSession` is the in-memory session implementation (`openjiuwen-session`); there is no separate `InMemorySession` type in the Rust API.
The prelude is capped at ~35 items so `use openjiuwen::prelude::*;` stays unpolluted: agent-construction types and entry points only, no module globs, and the only config structs it forwards are `ReActAgentConfig`, `DeepAgentConfig` and `TeamAgentSpec`.

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
- `publish = false` until the SDK API stabilizes; sibling crates and `docs/` are never edited from here.
- Feature names in this crate map to existing sibling features; adding a feature requires verifying the sibling's feature name first and keeping the default build offline.
- `openjiuwen-harness-providers` is a non-optional dependency (`Cargo.toml:28`; workspace entry `../Cargo.toml:64` sets no `default-features = false`), so that crate's own `default = ["native"]` always applies and the umbrella's `native` feature cannot actually disable the providers' native provider (same for `openjiuwen-harness`, workspace `../Cargo.toml:63`).
- No feature of this crate enables `openjiuwen-harness-providers/vendor`, so the vendor providers (`claudecode`, `codex`, `dsh`) are unreachable through the umbrella; a consumer must depend on `openjiuwen-harness-providers` directly with `features = ["vendor"]`.
- Python package spelling (`harness_protocol`, `agent_teams`, `sys_operation`) is mirrored even where it repeats the crate suffix, so `clippy::module_name_repetitions` is allowed crate-wide.
- The crate-level doc example is the "getting started" block: it must compile as a doctest and touch no network.
- Concrete crates are canonical: documentation lives there; this file documents only the facade.

## Notes for Agents

- `harness_providers` (`src/lib.rs:67`) and `harness` (`src/lib.rs:65`) are unconditional `pub use` aliases with no `#[cfg]`; only the prelude's `a2a` / `otel` items and the corresponding `tests/module_paths.rs` cases are feature-gated.
- The provider registry is reached as `openjiuwen::harness_providers::{create_harness, resolve_provider, ...}`; `create_harness` is `openjiuwen_harness_providers::factory::create_harness` (proved by `tests/module_paths.rs:44-48`) and the prelude forwards it ungated (`src/prelude.rs:30`).
- `TeamAgent` is defined in `openjiuwen_agent_teams::runtime::team_agent` and re-exported from both `agent_teams::runtime` and the crate root, so the prelude forwards the crate-root spelling.
- Every prelude item has a verified canonical path; `Role` and `JsonObject` are intentionally not in the prelude and are addressed by their full paths in docs and tests.
- `ReActAgent`/`ReActAgentConfig` are forwarded from `core::single_agent` (`openjiuwen_agent::react`) because examples and construction code need them without spelling the kernel path; `tests/module_paths.rs::prelude_resolves_the_agent_construction_items` proves identity for the agent-construction items and the `ContextEngine`/`LongTermMemory`/`Runner` lifecycle items.
- `LlmAgent` is forwarded with the crate-root spelling (`openjiuwen_agent::LlmAgent`); `core::application` exposes the same type as `LLMAgent`, so the prelude deliberately carries only one of the two spellings.
- `openjiuwen::core::session` is the whole `openjiuwen-session` crate root (`openjiuwen-core/src/lib.rs:57` re-exports the crate as `session`), so `AgentSession`, `AgentSessionOptions`, `create_agent_session`, `Session`, `SessionType`, `SessionConfig`, `Checkpointer`, `MemoryCheckpointer`, `FileCheckpointer`, `Checkpoint`, `CheckpointMeta`, `CheckpointerConfig`, `create_checkpointer`, `StreamItem` and `StreamEvent` are all reachable directly at `openjiuwen::core::session::<Name>`; the nested `checkpoint::*` and inner `session::*` modules are alternative spellings of the same items.
- The `sqlite` feature is wired to `openjiuwen-extensions/sqlite` and `openjiuwen-symphony/sqlite`; `cargo check -p openjiuwen --features sqlite` succeeds.
- `core` is `pub use openjiuwen_core as core` (`src/lib.rs:62`) and that crate aliases the whole foundation crate (`pub use openjiuwen_foundation as foundation`, `../openjiuwen-core/src/lib.rs:52`), so `openjiuwen::core::foundation::…` is the full `openjiuwen_foundation` module tree, not a hand-written shim.
- Tools resolve at both depths: `openjiuwen::core::foundation::{Tool, ToolCard, ToolContext, ToolOutput, FunctionTool, FunctionToolBuilder, ToolHandler, schema_for, RestfulApiTool, openai_tools}` (flat, `../openjiuwen-foundation/src/lib.rs:102`) and `openjiuwen::core::foundation::tool::{…same items…}` (`../openjiuwen-foundation/src/tool/mod.rs:31`); `Tool`/`FunctionTool` are also in the prelude (`src/prelude.rs:19`).
- The `#[tool]` proc macro is reachable through no alias of this crate: `openjiuwen` does not depend on `openjiuwen-macros` (`Cargo.toml:11-35`) and `openjiuwen-foundation` neither uses nor re-exports it, so a consumer adds `openjiuwen-macros` as its own dependency and writes `#[openjiuwen_macros::tool(...)]` (generated code also needs `async-trait` in the consumer).

## Known Issues

- `cargo clippy -p openjiuwen --all-targets -- -D warnings` also lints path dependencies in this workspace and currently fails on pre-existing `clippy::must_use_candidate` errors in sibling `crate_name` functions (`openjiuwen-auto-harness`, `openjiuwen-memory`, `openjiuwen-e2e`, `openjiuwen-examples`), so the definition of done is run with `--no-deps` until those siblings add `#[must_use]`.

## Routing Table

- `./src/lib.rs` → crate docs (Python→Rust table, feature matrix, getting-started doctest), module aliases, `crate_name()`
- `./src/prelude.rs` → curated convenience re-exports and the "prelude is convenience only" note
- `./Cargo.toml` → dependency list and the feature mapping table
- `./tests/module_paths.rs` → identity checks for one representative item per module alias, prelude compile/identity checks for the agent-construction and lifecycle items, plus feature-gated `a2a` / `otel` checks

## Test Strategy

- `tests/module_paths.rs` asserts `TypeId` identity (and `std::ptr::eq` for functions) between each alias and the concrete crate, so a mis-pointed alias fails loudly; `a2a` and `otel` items are behind `#[cfg(feature = ...)]`.
- `prelude_resolves_the_common_items` and `prelude_resolves_the_agent_construction_items` import `openjiuwen::prelude::*` and assert both usability and `TypeId` identity against the concrete crates, so a dropped or mis-pointed prelude export fails loudly.
- The crate-level getting-started example is the doctest that guards the primary public paths (`cargo test -p openjiuwen`).
- No unit tests: the facade has no behavior; behavior is tested in the concrete crates.
- Definition of done: `cargo fmt -p openjiuwen`, `cargo test -p openjiuwen`, `cargo test -p openjiuwen --features otel,a2a`, and `cargo clippy -p openjiuwen --all-targets -- -D warnings` all clean (add `--no-deps` while the sibling `crate_name` lint in `## Known Issues` is open).
