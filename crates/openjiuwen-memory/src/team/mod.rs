//! Team-scoped memory records, persistence, and projection.
//!
//! [`model`] holds the plain `serde` data of team memory: scopes, kinds, visibility,
//! records, create/update inputs, filters, and result types.
//! [`store`] is the persistence seam — the [`store::TeamMemoryStore`] trait, its in-memory
//! implementation, the [`store::TeamMemory`] facade, and the shared search ranking.
//! A [`store::TeamMemoryStore`] is the single source of truth for team memory: the facade
//! never caches, and deleted rows are gone from every view.
//! The `SQLite` backend in `team::sqlite` is an opt-in feature (`sqlite`): it implements
//! the same trait over a database file, so a facade can be made durable without changing
//! any caller.
//! [`projection`] converts a team record to and from the unified
//! [`crate::record::MemoryRecord`] so team memory can flow through the same recall and
//! prompt paths as long-term memory — a view derived from the store, never a second copy
//! of it.
//! [`document`] renders records to the shared `TEAM_MEMORY.md` Markdown format and parses
//! that file back losslessly, so a team summary survives a process restart.
//! [`toolkit`] exposes four model-facing tools (`remember`, `search`, `forget`, `stats`)
//! over a member-scoped store, and routes their JSON calls.
//! [`extractor`] turns a [`extractor::TeamTranscript`] into create requests, using a
//! deterministic rule-based default and a model-backed implementation.
//! [`prompts`] carries the extraction prompt, the JSON answer schema, the section tags,
//! and the bounded transcript renderer that the two extractors share.
//! [`hooks`] provides explicit, caller-driven lifecycle hooks (member join, session start,
//! turn sync, session end, member leave) that decide what to write without a background
//! task or a global registry.
//! [`observe`] is the optional observability seam: a [`observe::TeamMemoryObserver`]
//! receives one [`observe::TeamMemoryEvent`] per store operation, and starts unattached.
pub mod document;
pub mod extractor;
pub mod hooks;
pub mod model;
pub mod observe;
pub mod projection;
pub mod prompts;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod store;
pub mod toolkit;

pub use document::*;
pub use extractor::*;
pub use hooks::*;
pub use model::*;
pub use observe::*;
pub use projection::*;
// `prompts::EXTRACTION_SYSTEM_PROMPT` is not re-exported here: the crate root already
// binds `EXTRACTION_SYSTEM_PROMPT` from `graph` (restated explicitly in `lib.rs` because
// `analyzer` has one too), so a third glob would make the root name ambiguous.
// It stays reachable as `team::prompts::EXTRACTION_SYSTEM_PROMPT`.
pub use prompts::{
    render_transcript, EXTRACTION_RESPONSE_SCHEMA, MESSAGE_CONTENT_PREVIEW_MAX,
    TASK_CONTENT_PREVIEW_MAX, TEAM_MEMORY_TAGS, TRUNCATION_MARKER,
};
#[cfg(feature = "sqlite")]
pub use sqlite::*;
pub use store::*;
pub use toolkit::*;
