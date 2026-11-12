//! Provider registry and adapters for the provider-neutral harness SPI of openJiuwen.
//!
//! The crate ports the Python package `openjiuwen.harness_providers` (v0.1.20). It owns *provider
//! selection and adaptation*: resolving a provider by name, creating a harness from it, describing
//! its capabilities and model-configuration hook, and projecting a harness onto the kernel IO
//! surface. It never reimplements a harness: the native openJiuwen `DeepAgent` lives in
//! `openjiuwen-harness`, and the vendor harnesses (Claude Code, Codex, DSH) are deferred — their
//! cards and configuration types exist, but creating one returns `Error::Unsupported`.
//!
//! # Python parity
//!
//! Registry names, provider card names, capability declarations, error message wording and the
//! environment-driven model selection follow Python. Where Python relies on duck typing or a single
//! `ValueError`, Rust uses a typed provider interface and distinct error categories; those
//! deviations are recorded in `CONTEXT.md` and in the affected items.
//!
//! # Features
//!
//! - `native` (default) — the in-process native provider backed by `openjiuwen-harness`.
//! - `vendor` — the deferred `claudecode`, `codex` and `dsh` provider modules and their
//!   registration.
//! - `live-tests` — opt-in, never enabled by default; gated end-to-end tests against real vendor
//!   harnesses.
//!
//! The crate builds with `--no-default-features` and with `--features vendor`.
//!
//! # Module map
//!
//! - [`error`] — translation from the harness SPI error type to the workspace error.
//! - [`factory`] — the provider registry: registration, resolution, harness creation, the
//!   model-configuration gates, provider descriptors and the harness-context builder.
//! - [`input`] — harness input conversion, text projection and chunk-kind constants.
//! - [`io_adapter`] — the harness IO adapter: lifecycle, the single-consumer event pump, event
//!   to chunk projection, interaction routing and host answer parsing.
//! - [`json`] — JSON projections (`to_json_safe`, `to_json_object`).
//! - [`mcp_naming`] — the `<mcp-tools>` naming preamble for provider system prompts.
//! - [`provider`] — the registered provider interface (`ProviderSpec`), its wrappers, the
//!   registry-name constants and the introspection descriptor.
//! - [`providers`] — the compile-time built-in provider table (`native`, plus the `vendor`-gated
//!   stubs) and its resolution helper.
//! - [`skills`] — portable skill-bundle normalization and installation.
//!
//! The crate-internal `decode` module holds the strict JSON mapping helpers (unknown-key
//! rejection, typed option decoding) shared by the factory and the provider configuration
//! decoders.

mod decode;

pub mod error;
pub mod factory;
pub mod input;
pub mod io_adapter;
pub mod json;
pub mod mcp_naming;
pub mod provider;
pub mod providers;
pub mod skills;

pub use error::providers_error_to_common;
pub use factory::{
    available_providers, build_harness_context, create_harness, describe_provider,
    provider_descriptors, provider_model_config, register_harness_provider, resolve_provider,
    unregister_harness_provider, HarnessContextOptions, ProviderFactory,
    HARNESS_PROVIDER_ENTRY_POINT_GROUP, PROVIDER_NAMES,
};
pub use input::{
    harness_input_text, to_harness_input, INTERACTION_KIND, INTERACTIVE_INPUT_KIND,
    TASK_FAILED_KIND, TURN_LIFECYCLE_KIND,
};
pub use io_adapter::{
    EventObserver, HarnessIOAdapter, HarnessIOAdapterConfig, ProviderInteractionHandler,
};
pub use json::{to_json_object, to_json_safe};
pub use mcp_naming::{mcp_tool_naming_preamble, TOOL_PLACEHOLDER};
pub use provider::{
    is_builtin_provider, model_configuring_provider, plain_provider, provider_names,
    ProviderDescriptor, ProviderSpec,
};
pub use skills::{
    install_skills, normalize_skills, skill_scan_dir, SkillConflict, SkillMode, SkillSource,
};
