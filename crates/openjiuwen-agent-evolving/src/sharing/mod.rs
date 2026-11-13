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
//! - [`SharingBackend`] — the pluggable store seam with a local-file backend.
//!
//! The remote hub client is deferred; the backend seam is real and testable
//! offline.
