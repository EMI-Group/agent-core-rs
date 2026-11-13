//! Version identifiers and the persisted version document.
//!
//! A version is one JSON document under `<session_dir>/history/versions/`:
//! `{"id": 1, "label": null, "created_at": "<RFC 3339>", "parent": null,
//! "snapshot": <SessionSnapshot>}`.
use std::fmt;
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::session::SessionSnapshot;

/// A sequential version number (`v000001`), starting at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VersionId(pub u64);

impl VersionId {
    /// The raw version number.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for VersionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{:06}", self.0)
    }
}

impl FromStr for VersionId {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let digits = value.strip_prefix('v').unwrap_or(value);
        let id: u64 = digits.parse().map_err(|_| {
            Error::validation(format!(
                "invalid version id {value:?}; expected `v000001` or `1`"
            ))
        })?;
        if id == 0 {
            return Err(Error::validation("version ids start at 1"));
        }
        Ok(Self(id))
    }
}

impl From<u64> for VersionId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<VersionId> for u64 {
    fn from(id: VersionId) -> Self {
        id.0
    }
}

/// One complete version document, including its snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Version {
    /// The sequential version number.
    pub id: u64,
    /// An optional human-readable label (e.g. `"before-refactor"`).
    pub label: Option<String>,
    /// Creation time as an RFC 3339 string.
    pub created_at: String,
    /// The version this one was committed on top of, when any.
    pub parent: Option<u64>,
    /// The session snapshot captured for this version.
    pub snapshot: SessionSnapshot,
}

impl Version {
    /// The version's metadata, without its snapshot.
    #[must_use]
    pub fn meta(&self) -> VersionMeta {
        VersionMeta::from(self)
    }
}

/// A version's metadata, without its (potentially large) snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VersionMeta {
    /// The sequential version number.
    pub id: u64,
    /// An optional human-readable label.
    pub label: Option<String>,
    /// Creation time as an RFC 3339 string.
    pub created_at: String,
    /// The version this one was committed on top of, when any.
    pub parent: Option<u64>,
}

impl From<&Version> for VersionMeta {
    fn from(version: &Version) -> Self {
        Self {
            id: version.id,
            label: version.label.clone(),
            created_at: version.created_at.clone(),
            parent: version.parent,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::VersionId;

    #[test]
    fn display_uses_six_padded_digits() {
        assert_eq!(VersionId(1).to_string(), "v000001");
        assert_eq!(VersionId(42).to_string(), "v000042");
        assert_eq!(VersionId(1_234_567).to_string(), "v1234567");
    }

    #[test]
    fn parse_accepts_prefixed_and_plain_forms() {
        assert_eq!("v000001".parse::<VersionId>().unwrap(), VersionId(1));
        assert_eq!("1".parse::<VersionId>().unwrap(), VersionId(1));
        assert_eq!("v42".parse::<VersionId>().unwrap(), VersionId(42));
    }

    #[test]
    fn parse_rejects_invalid_ids() {
        assert!("".parse::<VersionId>().is_err());
        assert!("abc".parse::<VersionId>().is_err());
        assert!("v0".parse::<VersionId>().is_err());
        assert!("-1".parse::<VersionId>().is_err());
    }
}
