//! Symphony — the capability-asset engine of openJiuwen.
//!
//! Maps to the Python package `openjiuwen.symphony` (v0.1.20). Python's package is a
//! ~52k-line, host-driven domain engine: it fingerprints capability assets, indexes
//! them, builds a capability relationship graph, plans asset compositions with an LLM,
//! and distils observed execution flows into skills. This crate ports the offline,
//! deterministic core of that engine and wires it onto the Rust kernel.
//!
//! # Module map
//!
//! - [`model`] — the capability-asset cards (`CapabilityDescriptor`, `CapabilityIo`,
//!   `CapabilityFingerprint`), static fingerprint artifacts, data-type vocabulary, and
//!   name-normalization helpers.
//! - [`store`] — the [`store::AssetStore`] persistence seam and its foundation-KV
//!   backend (in-memory, local JSON files, optional SQLite).
//! - [`registry`] — [`registry::CapabilityRegistry`]: in-memory authoritative catalog
//!   with write-through persistence, CRUD, and asset search.
//! - [`search`] — BM25 keyword plus metadata search ([`search::LexicalIndex`]) and an
//!   embedding-backed semantic index ([`search::SemanticIndex`]) built on
//!   `openjiuwen-retrieval`.
//! - [`graph`] — the versioned capability relationship graph (`can_feed` edges, JGF v2
//!   JSON) and its deterministic mutations.
//! - [`artifact`] — [`artifact::GraphArtifactStore`]: immutable, atomically published
//!   graph revisions with a `current` pointer (in-memory and filesystem backends).
//! - [`orchestration`] — deterministic relation candidates, the bounded planner, the
//!   typed [`orchestration::OrchestrationPlan`], and an executor that maps a plan onto
//!   an `openjiuwen-graph` compiled graph.
//! - [`evaluation`] — fail-soft evaluation suite with deterministic static metrics.
//! - [`discovery`] — `SKILL.md` folder scanning ([`discovery::SkillFolderScanner`]) and
//!   the lexical [`discovery::SkillFs`] catalog over installed skills.
//! - [`experience`] — [`experience::ExperienceBank`] and the persisted,
//!   embedding-backed [`experience::ExperienceIndex`]: distilled capability combinations
//!   with deterministic ranking and persistence.
//! - [`flow`] — flow distillation ([`flow::FlowEngine`]) and reusable `SkillPack` artifacts.
//! - [`llm`] — LLM-guided stages (tree retrieval, ontology matching, semantic scoring,
//!   fingerprint extraction, judging) with deterministic offline fallbacks.
//! - [`object_artifact`] — S3/OBS, GCS and Azure Blob graph artifact persistence over
//!   foundation's object stores and the Google Cloud Storage and Azure Blob REST APIs.
//! - [`runtime`] — [`runtime::SymphonyRuntime`], the composition root: configuration,
//!   lifecycle (`start`/`stop`/`close`), graph build, plan, invoke, mutations, and
//!   evolution feedback.
//! - [`deferred`] — an API-shaped, documented `Error::Unsupported` stub for the only
//!   remaining deferred backend, which needs a native GPU inference stack: `vLLM` logit
//!   selection.
//!
//! # Contracts
//!
//! Every public data type is `serde`-serializable (`snake_case`, tolerant of unknown
//! fields), errors are [`openjiuwen_common::error::Error`], logging is `tracing`, and
//! no `unsafe` is used. Async I/O is tokio-based; the registry and plan APIs never
//! block on a lock across an `.await`.
#![forbid(unsafe_code)]

/// The crate name, useful in diagnostics and documentation.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-symphony"
}

pub use openjiuwen_common::error::{Error, Result};

/// Immutable, atomically published capability-graph revisions.
pub mod artifact;
/// API-shaped stubs for the Python areas that require an LLM/ML stack or external services.
pub mod deferred;
/// Skill-folder discovery: `SKILL.md` scanning and lexical skill search.
pub mod discovery;
/// Error constructors and helpers over [`openjiuwen_common::error`].
pub mod error;
/// Fail-soft capability evaluation with deterministic static metrics.
pub mod evaluation;
/// The experience bank: distilled capability-combination knowledge with ranking.
pub mod experience;
/// Flow distillation and reusable SkillPack artifacts.
pub mod flow;
/// The `can_feed` capability relationship graph (JGF v2 documents).
pub mod graph;
/// LLM-guided stages with deterministic offline fallbacks.
pub mod llm;
/// Capability cards, fingerprints, data-type vocabulary, and naming helpers.
pub mod model;
/// Object-store backed graph artifact persistence.
pub mod object_artifact;
/// Deterministic relation candidates, planning, and plan execution.
pub mod orchestration;
/// The in-memory capability registry with write-through persistence.
pub mod registry;
/// The symphony composition root: configuration, lifecycle, graph build, plan, invoke.
pub mod runtime;
/// Keyword/metadata and semantic capability retrieval.
pub mod search;
/// Capability-asset persistence over foundation KV stores.
pub mod store;
