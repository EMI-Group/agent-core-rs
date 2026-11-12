//! Team-scoped memory model: scopes, kinds, visibility, records, inputs, filters.
//!
//! Everything here is plain `serde` data plus pure validation and normalization:
//! no storage, no async, and no clock reads apart from the documented "now" defaults.
//! JSON is `snake_case` and tolerant of unknown fields; a missing required field
//! yields a constructible but invalid value that `validate()` rejects instead of a
//! deserialization error.
//!
//! # JSON shape
//!
//! ```json
//! {
//!   "id": "018f1d2e3f4a5b6c7d8e9f01",
//!   "team_id": "review-team",
//!   "member_id": "alice",
//!   "kind": "decision",
//!   "content": "Use SQLite for the shared store.",
//!   "title": "Storage choice",
//!   "importance": 0.8,
//!   "tags": ["storage", "sqlite"],
//!   "metadata": {"issue": 42},
//!   "created_at": "2026-01-02T03:04:05Z",
//!   "updated_at": "2026-01-02T03:04:05Z",
//!   "source": "leader-extraction",
//!   "visibility": {"members": ["alice", "bob"]},
//!   "expires_at": null
//! }
//! ```
//!
//! `visibility` is externally tagged: `"team"`, `"private"`, or
//! `{"members": ["alice", "bob"]}`. `kind` is a plain string, so an unknown kind
//! round-trips as `TeamMemoryKind::Custom` instead of failing.
//!
//! # Layout
//!
//! - [`scope`] — `TeamMemoryScope` and the visibility/read rules.
//! - [`kind`] — `TeamMemoryKind` and `TeamVisibility`.
//! - [`record`] — `TeamMemoryRecord` and id generation.
//! - [`input`] — `NewTeamMemory` and `TeamMemoryUpdate`.
//! - [`filter`] — `TeamMemoryFilter`.
//! - [`results`] — `TeamMemoryStats` and `TeamSearchHit`.
//! - `validate` — the shared validation and normalization helpers.
mod filter;
mod input;
mod kind;
mod record;
mod results;
mod scope;
mod validate;
pub use filter::*;
pub use input::*;
pub use kind::*;
pub use record::*;
pub use results::*;
pub use scope::*;
pub use validate::normalize_tags;

/// Longest accepted team, member, or session id, in characters.
pub const MAX_SCOPE_ID_CHARS: usize = 128;
/// Longest accepted record id, in characters.
pub const MAX_RECORD_ID_CHARS: usize = 128;
/// Longest accepted record content, in characters.
pub const MAX_CONTENT_CHARS: usize = 16_384;
/// Longest accepted title, in characters.
pub const MAX_TITLE_CHARS: usize = 256;
/// Maximum number of tags on one record.
pub const MAX_TAGS: usize = 32;
/// Longest accepted single tag, in characters.
pub const MAX_TAG_CHARS: usize = 64;
/// Maximum number of members named by [`TeamVisibility::Members`].
pub const MAX_VISIBILITY_MEMBERS: usize = 256;
/// Importance used when neither the input nor the caller supplies one.
pub const DEFAULT_IMPORTANCE: f32 = 0.5;
/// Bucket key used by [`TeamMemoryStats::by_member`] for records without a member.
pub const TEAM_WIDE_MEMBER_KEY: &str = "__team__";

/// Returns an empty JSON object; the serde default of every `metadata` field.
pub(crate) fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

/// Returns [`DEFAULT_IMPORTANCE`]; the serde default of `importance`.
pub(crate) fn default_importance() -> f32 {
    DEFAULT_IMPORTANCE
}
