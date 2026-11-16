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
//! - [`evaluation`] — fail-soft evaluation suite: static card metrics, trace metrics,
//!   event-time case windows, status-aware aggregation, and the LLM judge.
//! - [`discovery`] — `SKILL.md` folder scanning ([`discovery::SkillFolderScanner`]), the
//!   lexical [`discovery::SkillFs`] catalog over installed skills, its model-facing prompt
//!   snapshot, and the incremental skill-notice cursor.
//! - [`experience`] — [`experience::ExperienceBank`] and the persisted,
//!   embedding-backed [`experience::ExperienceIndex`], fed by session-trace ingestion,
//!   deterministic clustering, and the collector.
//! - [`flow`] — flow distillation ([`flow::FlowEngine`]) and reusable `SkillPack` artifacts.
//! - [`redaction`] — credential/PII redaction for model-facing and packaged text.
//! - [`llm`] — LLM-guided stages (tree retrieval, ontology matching, semantic scoring,
//!   fingerprint extraction, judging) with deterministic offline fallbacks, plus the
//!   model/metadata identity digests and the persistent relation-match cache.
//! - [`object_artifact`] — S3/OBS, GCS and Azure Blob graph artifact persistence over
//!   foundation's object stores and the Google Cloud Storage and Azure Blob REST APIs.
//! - [`runtime`] — [`runtime::SymphonyRuntime`], the composition root: configuration,
//!   lifecycle (`start`/`stop`/`close`), graph build, plan, invoke, mutations, and
//!   evolution feedback.
//! - [`observation`] — the caller-supplied evidence log, content-hashed observation and
//!   merged graph revisions, and the per-edge overlay weights the planner reads.
//! - [`retrieval`] — capability-tree index construction and progressive retrieval: the
//!   persisted index artifact set, field-weighted lexical scoring, bounded disclosure
//!   rendering, and embedding-backed branch selection.
//! - [`deferred`] — API-shaped, documented `Error::Unsupported` stubs for the backends
//!   that need a native GPU inference stack: `vLLM` logit selection.
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

/// Agent-facing browsing and search tools over the capability-tree index.
pub mod agent;
/// Immutable, atomically published capability-graph revisions.
pub mod artifact;
/// API-shaped stub for the one Python area that requires a GPU inference stack.
pub mod deferred;
/// Skill-folder discovery: `SKILL.md` scanning and lexical skill search.
pub mod discovery;
/// Error constructors and helpers over [`openjiuwen_common::error`].
pub mod error;
/// Fail-soft capability evaluation: static and trace metrics, windows, aggregation.
pub mod evaluation;
/// The experience layer: bank, persisted index, trace ingestion, clustering, collector.
pub mod experience;
/// Flow distillation and reusable SkillPack artifacts.
pub mod flow;
/// The `can_feed` capability relationship graph (JGF v2 documents).
pub mod graph;
/// LLM-guided stages with deterministic offline fallbacks, plus the match cache.
pub mod llm;
/// Capability cards, fingerprints, data-type vocabulary, and naming helpers.
pub mod model;
/// Object-store backed graph artifact persistence.
pub mod object_artifact;
/// Graph observation: the evidence log, derived graph revisions, and overlay weights.
pub mod observation;
/// Deterministic relation candidates, planning, and plan execution.
pub mod orchestration;
/// Credential and PII redaction for model-facing and packaged text.
pub mod redaction;
/// The in-memory capability registry with write-through persistence.
pub mod registry;
/// Capability-tree index build and progressive retrieval (lexical and embedding-backed).
pub mod retrieval;
/// The symphony composition root: configuration, lifecycle, graph build, plan, invoke.
pub mod runtime;
/// Keyword/metadata and semantic capability retrieval.
pub mod search;
/// Capability-asset persistence over foundation KV stores.
pub mod store;
