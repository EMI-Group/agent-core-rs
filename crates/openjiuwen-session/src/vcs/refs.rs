//! Local refs for a session's version history.
//!
//! Ports the ref/head subset of `openjiuwen/core/session/vcs/` (v0.1.20): the
//! `Head` pointer that Python keeps per session becomes a plain refs document.
//! A refs document lives at `<session_dir>/history/refs.json` and maps a
//! validated [`RefName`] to the version it points at:
//!
//! ```json
//! {"refs": {"HEAD": {"name": "HEAD", "target": 3, "created_at": "…", "updated_at": "…"}}}
//! ```
//!
//! Two refs are created automatically on the first commit:
//!
//! - [`HEAD_REF`] is the mutable tip: every commit advances it and a rewind
//!   resets it.
//! - [`MAIN_REF`] tracks the latest committed version: commits advance it and a
//!   hard rewind re-points it.
//!
//! Every other name ([`MAIN_REF`]-style, `before-refactor`, …) is an ordinary
//! entry, so callers can annotate versions themselves.
//! Refs are names, never git objects: names are flat (no path separators), and
//! there is no git/external backend behind them.
//! All refs are local metadata stored next to the versions, so a refs document
//! travels with the session directory and needs no network.
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, TimeDelta, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{format_iso8601, now_rfc3339, now_utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::version::VersionId;

/// The mutable tip ref, advanced by every commit and reset by a rewind.
pub const HEAD_REF: &str = "HEAD";

/// The ref tracking the latest committed version.
pub const MAIN_REF: &str = "main";

/// The longest accepted ref name, in characters.
pub const MAX_REF_NAME_LEN: usize = 128;

/// A validated ref name.
///
/// Names are non-empty, at most [`MAX_REF_NAME_LEN`] characters long, and free
/// of whitespace, control characters, path separators (`/`, `\`), and `..`, so a
/// name is never a path and never escapes the history directory.
/// Construction validates, so a [`RefName`] value is always usable as a refs
/// document key; serde keeps the string form and re-validates on read.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RefName(String);

impl RefName {
    /// Validates and wraps `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when the name is empty, too long, or contains whitespace, control
    /// characters, a path separator, or `..`.
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        validate_ref_name(&name)?;
        Ok(Self(name))
    }

    /// The [`HEAD_REF`] name.
    #[must_use]
    pub fn head() -> Self {
        Self(HEAD_REF.to_string())
    }

    /// The [`MAIN_REF`] name.
    #[must_use]
    pub fn main() -> Self {
        Self(MAIN_REF.to_string())
    }

    /// The name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether this is the [`HEAD_REF`] name.
    #[must_use]
    pub fn is_head(&self) -> bool {
        self.0 == HEAD_REF
    }
}

impl AsRef<str> for RefName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RefName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for RefName {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl Serialize for RefName {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for RefName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::new(name).map_err(serde::de::Error::custom)
    }
}

/// Rejects names that could not be used as a refs document key.
fn validate_ref_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::validation("ref name must not be empty"));
    }
    if name.chars().count() > MAX_REF_NAME_LEN {
        return Err(Error::validation(format!(
            "ref name must be at most {MAX_REF_NAME_LEN} characters: {name:?}"
        )));
    }
    if name.chars().any(char::is_whitespace) {
        return Err(Error::validation(format!(
            "ref name must not contain whitespace: {name:?}"
        )));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(Error::validation(format!(
            "ref name must not contain path separators: {name:?}"
        )));
    }
    if name.contains("..") {
        return Err(Error::validation(format!(
            "ref name must not contain `..`: {name:?}"
        )));
    }
    if name.chars().any(char::is_control) {
        return Err(Error::validation(format!(
            "ref name must not contain control characters: {name:?}"
        )));
    }
    Ok(())
}

/// One ref: the name, the version it points at, and its timestamps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RefEntry {
    /// The ref name; equal to the key it is stored under.
    pub name: RefName,
    /// The version the ref points at.
    pub target: VersionId,
    /// When the ref was first created, as an RFC 3339 string.
    pub created_at: String,
    /// When the ref last moved, as an RFC 3339 string (strictly monotonic).
    pub updated_at: String,
}

impl Default for RefEntry {
    /// The tolerant-deserialization placeholder: [`HEAD_REF`] pointing at the
    /// first version. A refs document always spells every field out, so this is
    /// only reached for a hand-written partial entry.
    fn default() -> Self {
        Self {
            name: RefName::head(),
            target: VersionId(1),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }
}

impl RefEntry {
    /// The version this ref points at.
    #[must_use]
    pub fn target(&self) -> VersionId {
        self.target
    }
}

/// The `refs.json` document: a name-ordered map of refs.
///
/// Unknown fields are tolerated and a missing `refs` key yields an empty
/// document, so an older or newer writer never breaks a reader.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Refs {
    /// The refs, keyed by name and iterated in name order.
    pub refs: BTreeMap<RefName, RefEntry>,
}

impl Refs {
    /// Creates an empty refs document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up one ref.
    #[must_use]
    pub fn get(&self, name: &RefName) -> Option<&RefEntry> {
        self.refs.get(name)
    }

    /// Creates or updates `name` to point at `target`, returning the entry.
    ///
    /// An existing ref keeps its `created_at` and gets a strictly later
    /// `updated_at`, so two sets in the same millisecond still order correctly.
    pub fn set(&mut self, name: &RefName, target: VersionId) -> RefEntry {
        let entry = if let Some(existing) = self.refs.get(name) {
            RefEntry {
                name: name.clone(),
                target,
                created_at: existing.created_at.clone(),
                updated_at: next_timestamp(&existing.updated_at),
            }
        } else {
            let now = now_rfc3339();
            RefEntry {
                name: name.clone(),
                target,
                created_at: now.clone(),
                updated_at: now,
            }
        };
        self.refs.insert(name.clone(), entry.clone());
        entry
    }

    /// Points [`HEAD_REF`] at `target`.
    pub fn set_head(&mut self, target: VersionId) -> RefEntry {
        self.set(&RefName::head(), target)
    }

    /// Points [`MAIN_REF`] at `target`.
    pub fn set_main(&mut self, target: VersionId) -> RefEntry {
        self.set(&RefName::main(), target)
    }

    /// Removes one ref, returning it when it existed.
    pub fn remove(&mut self, name: &RefName) -> Option<RefEntry> {
        self.refs.remove(name)
    }

    /// The refs in name order.
    #[must_use]
    pub fn list(&self) -> Vec<&RefEntry> {
        self.refs.values().collect()
    }

    /// The number of refs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Whether the document holds no refs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// The [`HEAD_REF`] entry, when it exists.
    #[must_use]
    pub fn head(&self) -> Option<&RefEntry> {
        self.refs.get(&RefName::head())
    }

    /// The [`MAIN_REF`] entry, when it exists.
    #[must_use]
    pub fn main(&self) -> Option<&RefEntry> {
        self.refs.get(&RefName::main())
    }
}

/// Returns a timestamp strictly after `previous`, formatted like
/// [`now_rfc3339`] (RFC 3339, millisecond precision, `Z` suffix).
///
/// The comparison happens at the serialized millisecond precision, so two calls
/// inside one millisecond still produce an increasing pair.
fn next_timestamp(previous: &str) -> String {
    let now = now_utc();
    if let Ok(previous) = DateTime::parse_from_rfc3339(previous) {
        if now.timestamp_millis() <= previous.timestamp_millis() {
            return format_iso8601(&(previous.with_timezone(&Utc) + TimeDelta::milliseconds(1)));
        }
    }
    format_iso8601(&now)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{next_timestamp, RefEntry, RefName, Refs, HEAD_REF, MAIN_REF, MAX_REF_NAME_LEN};
    use crate::vcs::VersionId;

    #[test]
    fn ref_names_accept_reserved_and_custom_names() {
        assert_eq!(RefName::head().as_str(), HEAD_REF);
        assert_eq!(RefName::main().as_str(), MAIN_REF);
        assert_eq!(
            "before-refactor".parse::<RefName>().unwrap().as_str(),
            "before-refactor"
        );
        assert!(RefName::new("v1.0").is_ok());
        assert!(RefName::head().is_head());
        assert!(!RefName::main().is_head());
        assert_eq!(RefName::head().to_string(), "HEAD");
        assert_eq!(RefName::head().as_ref(), "HEAD");
    }

    #[test]
    fn ref_names_reject_unsafe_forms() {
        for name in ["", "a b", "a\tb", "a/b", "a\\b", "..", "a..b", "a\u{7}b"] {
            let error = RefName::new(name).unwrap_err();
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation,
                "name {name:?} must be rejected"
            );
        }
        let long = "r".repeat(MAX_REF_NAME_LEN + 1);
        assert!(RefName::new(long).is_err());
        assert!(RefName::new("r".repeat(MAX_REF_NAME_LEN)).is_ok());
    }

    #[test]
    fn refs_document_round_trips_through_json() {
        let mut refs = Refs::new();
        refs.set(&RefName::head(), VersionId(3));
        refs.set(&RefName::new("release-1").unwrap(), VersionId(2));
        let json = serde_json::to_value(&refs).unwrap();
        assert_eq!(json["refs"]["HEAD"]["target"], serde_json::json!(3));
        assert_eq!(json["refs"]["HEAD"]["name"], serde_json::json!("HEAD"));
        let restored: Refs = serde_json::from_value(json).unwrap();
        assert_eq!(restored, refs);
        assert_eq!(restored.len(), 2);
        assert!(!restored.is_empty());
        assert_eq!(restored.head().unwrap().target(), VersionId(3));
        let names: Vec<&str> = restored.list().iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["HEAD", "release-1"]);
    }

    #[test]
    fn refs_document_tolerates_missing_and_unknown_fields() {
        let empty: Refs = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(empty.is_empty());
        assert!(empty.head().is_none());
        let unknown: Refs =
            serde_json::from_value(serde_json::json!({"schema": 2, "refs": {}})).unwrap();
        assert!(unknown.is_empty());
    }

    #[test]
    fn set_creates_then_updates_monotonically() {
        let mut refs = Refs::new();
        let created = refs.set(&RefName::main(), VersionId(1));
        assert_eq!(created.target, VersionId(1));
        assert_eq!(created.created_at, created.updated_at);
        assert!(created.created_at.ends_with('Z'));

        let moved = refs.set(&RefName::main(), VersionId(2));
        assert_eq!(moved.target, VersionId(2));
        assert_eq!(moved.created_at, created.created_at);
        assert!(moved.updated_at > created.updated_at);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs.main().unwrap().target(), VersionId(2));

        let back = refs.set_main(VersionId(1));
        assert!(back.updated_at > moved.updated_at);
        assert_eq!(refs.get(&RefName::main()).unwrap().target(), VersionId(1));
        assert_eq!(refs.set_head(VersionId(1)).target, VersionId(1));
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn remove_reports_whether_the_ref_existed() {
        let mut refs = Refs::new();
        refs.set_head(VersionId(1));
        assert!(refs.remove(&RefName::head()).is_some());
        assert!(refs.remove(&RefName::head()).is_none());
        assert!(refs.is_empty());
    }

    #[test]
    fn next_timestamp_moves_forward_for_equal_or_later_input() {
        let current = openjiuwen_common::utils::id::now_rfc3339();
        let advanced = next_timestamp(&current);
        assert!(advanced > current);
        // A stamp in the future is bumped by exactly one millisecond.
        assert_eq!(
            next_timestamp("2999-01-01T00:00:00.000Z"),
            "2999-01-01T00:00:00.001Z"
        );
        // A malformed stamp falls back to now.
        let fallback = next_timestamp("not-a-timestamp");
        assert!(fallback.ends_with('Z'));
    }

    #[test]
    fn ref_entry_default_is_the_tolerant_placeholder() {
        let entry = RefEntry::default();
        assert_eq!(entry.name.as_str(), HEAD_REF);
        assert_eq!(entry.target, VersionId(1));
    }
}
