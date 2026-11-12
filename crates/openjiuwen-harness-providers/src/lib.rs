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
//! - [`input`] — harness input conversion, text projection and chunk-kind constants.
//! - [`json`] — JSON projections (`to_json_safe`, `to_json_object`).
//! - [`mcp_naming`] — the `<mcp-tools>` naming preamble for provider system prompts.
//! - [`skills`] — portable skill-bundle normalization and installation.
//!
//! Later phases add the provider registry (`factory`), the provider table (`providers`) and the IO
//! adapter (`io_adapter`); the strict JSON mapping helpers in the crate-internal `decode` module
//! back their configuration decoders.

mod decode;

pub mod error;
pub mod input;
pub mod json;
pub mod mcp_naming;
pub mod skills;

pub use error::providers_error_to_common;
pub use input::{
    harness_input_text, to_harness_input, INTERACTION_KIND, INTERACTIVE_INPUT_KIND,
    TASK_FAILED_KIND, TURN_LIFECYCLE_KIND,
};
pub use json::{to_json_object, to_json_safe};
pub use mcp_naming::{mcp_tool_naming_preamble, TOOL_PLACEHOLDER};
pub use skills::{
    install_skills, normalize_skills, skill_scan_dir, SkillConflict, SkillMode, SkillSource,
};
