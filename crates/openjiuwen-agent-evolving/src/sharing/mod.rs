//! Sharing: packaging experience records for reuse across skills and users.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.sharing`: the wire
//! types (`sharing/types.py`), keyword extraction (`sharing/keyword_extractor.py`),
//! the local-file backend (`sharing/backends/local_file.py`) and the stager
//! (`sharing/share_stager.py`).
//!
//! - [`SharedExperience`] / [`SharedSkillBundle`] — the published records.
//! - [`KeywordExtractor`] — deterministic keyword/summary extraction.
//! - [`ShareStager`] — approval gating and bundle assembly.
//! - [`SharingBackend`] — the pluggable store seam, with a local-file backend,
//!   an in-memory backend and the remote hub client.
//!
//! The local backends are the offline port; [`RemoteHubSharingBackend`] and its
//! HTTP hub protocol are new design (the Python package only knows the local
//! filesystem), so a hub can be swapped in without touching the stager.

mod backend;
mod keywords;
mod remote;
mod stager;
mod types;

pub use backend::{LocalFileSharingBackend, MemorySharingBackend, SharingBackend};
pub use keywords::KeywordExtractor;
pub use remote::{RemoteHubOptions, RemoteHubSharingBackend, HUB_API_KEY_ENV, HUB_URL_ENV};
pub use stager::{ShareStager, StagedBundle};
pub use types::{
    QueryKeywords, SharedExperience, SharedSkillBundle, SharingMeta, SkillPackageMeta,
    SkillSearchResult,
};
