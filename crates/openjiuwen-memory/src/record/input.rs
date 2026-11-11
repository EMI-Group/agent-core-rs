//! [`NewMemory`] and [`MemoryUpdate`]: the write inputs accepted by the engine.
//!
//! Both types are `snake_case` serde data and validate their own invariants with
//! `Error::Validation`; neither touches storage or the clock.
use super::kind::MemoryType;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// A memory the caller wants to store.
///
/// The engine fills in the id, scope, and timestamps; `importance` falls back to
/// `MemoryEngineConfig::importance_default` and `timestamp` to "now" when absent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NewMemory {
    /// The memory text; must contain a non-whitespace character.
    pub content: String,
    /// The memory kind; [`MemoryType::Variable`] is rejected.
    #[serde(alias = "mem_type")]
    pub kind: MemoryType,
    /// Importance override in `0.0..=1.0`; engine default when absent.
    pub importance: Option<f32>,
    /// Structured fields stored with the record.
    pub metadata: BTreeMap<String, Value>,
    /// Optional provenance marker.
    pub source: Option<String>,
    /// Creation time override; "now" when absent.
    pub timestamp: Option<DateTime<Utc>>,
}
impl NewMemory {
    /// Builds a new memory of `kind`.
    #[must_use]
    pub fn new(content: impl Into<String>, kind: MemoryType) -> Self {
        Self {
            content: content.into(),
            kind,
            ..Self::default()
        }
    }
    /// Builds a [`MemoryType::SemanticMemory`] input.
    #[must_use]
    pub fn semantic(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::SemanticMemory)
    }
    /// Builds a [`MemoryType::EpisodicMemory`] input.
    #[must_use]
    pub fn episodic(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::EpisodicMemory)
    }
    /// Builds a [`MemoryType::UserProfile`] input.
    #[must_use]
    pub fn user_profile(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::UserProfile)
    }
    /// Builds a [`MemoryType::Summary`] input.
    #[must_use]
    pub fn summary(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::Summary)
    }
    /// Builds a [`MemoryType::Coding`] input.
    #[must_use]
    pub fn coding(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::Coding)
    }
    /// Returns the input with `importance` set.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }
    /// Returns the input with one metadata field inserted (incoming value wins).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    /// Returns the input with `source` set.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
    /// Returns the input with an explicit `timestamp`.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }
    /// Validates the input.
    ///
    /// Rejects empty/whitespace-only content, the non-writable [`MemoryType::Variable`]
    /// kind, and an importance outside `0.0..=1.0` (or non-finite).
    pub fn validate(&self) -> Result<()> {
        if self.content.trim().is_empty() {
            return Err(Error::validation(
                "new memory `content` must not be empty or whitespace",
            ));
        }
        if self.kind == MemoryType::Variable {
            return Err(Error::validation(
                "new memory `kind` must not be `variable`; use the variable API instead",
            ));
        }
        if let Some(importance) = self.importance {
            validate_importance(importance)?;
        }
        Ok(())
    }
}
/// A partial update of a stored record.
///
/// Only set fields are applied: `content`/`kind`/`importance`/`source` replace the
/// stored value, and `metadata` merges key by key with the incoming value winning.
/// An empty update is a no-op and is reported by [`MemoryUpdate::is_empty`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryUpdate {
    /// Replacement content.
    pub content: Option<String>,
    /// Replacement kind; [`MemoryType::Variable`] is rejected.
    pub kind: Option<MemoryType>,
    /// Replacement importance in `0.0..=1.0`.
    pub importance: Option<f32>,
    /// Metadata fields to merge (incoming wins per key).
    pub metadata: BTreeMap<String, Value>,
    /// Replacement source.
    pub source: Option<String>,
}
impl MemoryUpdate {
    /// Builds an update replacing the content.
    #[must_use]
    pub fn content(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            ..Self::default()
        }
    }
    /// Returns the update with `importance` set.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }
    /// Returns the update with one metadata field to merge.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    /// Returns the update with `source` set.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
    /// Returns the update with `kind` set.
    #[must_use]
    pub fn with_kind(mut self, kind: MemoryType) -> Self {
        self.kind = Some(kind);
        self
    }
    /// Returns `true` when the update would not change anything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_none()
            && self.kind.is_none()
            && self.importance.is_none()
            && self.metadata.is_empty()
            && self.source.is_none()
    }
    /// Validates the patch.
    ///
    /// Rejects empty/whitespace-only content, the non-writable [`MemoryType::Variable`]
    /// kind, and an importance outside `0.0..=1.0` (or non-finite).
    pub fn validate(&self) -> Result<()> {
        if let Some(content) = &self.content {
            if content.trim().is_empty() {
                return Err(Error::validation(
                    "memory update `content` must not be empty or whitespace",
                ));
            }
        }
        if self.kind == Some(MemoryType::Variable) {
            return Err(Error::validation(
                "memory update `kind` must not be `variable`; use the variable API instead",
            ));
        }
        if let Some(importance) = self.importance {
            validate_importance(importance)?;
        }
        Ok(())
    }
}
/// Rejects a non-finite importance or one outside `0.0..=1.0`.
fn validate_importance(importance: f32) -> Result<()> {
    if !importance.is_finite() || !(0.0..=1.0).contains(&importance) {
        return Err(Error::validation(format!(
            "`importance` must be within 0.0..=1.0 (got {importance})"
        )));
    }
    Ok(())
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn constructors_set_kind_and_content() {
        assert_eq!(NewMemory::semantic("a").kind, MemoryType::SemanticMemory);
        assert_eq!(NewMemory::episodic("a").kind, MemoryType::EpisodicMemory);
        assert_eq!(NewMemory::user_profile("a").kind, MemoryType::UserProfile);
        assert_eq!(NewMemory::summary("a").kind, MemoryType::Summary);
        assert_eq!(NewMemory::coding("a").kind, MemoryType::Coding);
        assert_eq!(NewMemory::new("a", MemoryType::Unknown).content, "a");
        assert_eq!(NewMemory::default().kind, MemoryType::Unknown);
    }
    #[test]
    fn new_memory_builders_chain() {
        let memory = NewMemory::semantic("likes tea")
            .with_importance(0.4)
            .with_metadata("lang", serde_json::json!("en"))
            .with_source("chat")
            .with_timestamp(DateTime::from_timestamp(1_700_000_000, 0).unwrap());
        assert_eq!(memory.importance, Some(0.4));
        assert_eq!(memory.metadata.get("lang"), Some(&serde_json::json!("en")));
        assert_eq!(memory.source.as_deref(), Some("chat"));
        assert!(memory.timestamp.is_some());
        assert!(memory.validate().is_ok());
    }
    #[test]
    fn new_memory_validate_rejects_bad_input() {
        assert!(NewMemory::semantic("   ")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("content"));
        assert!(NewMemory::new("x", MemoryType::Variable)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("variable"));
        assert!(NewMemory::semantic("x")
            .with_importance(2.0)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("importance"));
        assert!(NewMemory::semantic("x")
            .with_importance(f32::NAN)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("importance"));
        assert!(NewMemory::semantic("x")
            .with_importance(0.0)
            .validate()
            .is_ok());
        assert!(NewMemory::semantic("x")
            .with_importance(1.0)
            .validate()
            .is_ok());
    }
    #[test]
    fn new_memory_serde_round_trip_and_mem_type_alias() {
        let memory = NewMemory::coding("uses rustfmt").with_importance(0.6);
        let json = serde_json::to_value(&memory).unwrap();
        assert_eq!(json["kind"], serde_json::json!("coding"));
        assert_eq!(json["content"], serde_json::json!("uses rustfmt"));
        let restored: NewMemory = serde_json::from_value(json).unwrap();
        assert_eq!(restored, memory);
        let aliased: NewMemory = serde_json::from_value(serde_json::json!({
            "content": "x",
            "mem_type": "semantic_memory"
        }))
        .unwrap();
        assert_eq!(aliased.kind, MemoryType::SemanticMemory);
        assert_eq!(aliased.importance, None);
    }
    #[test]
    fn update_constructors_and_builders() {
        let update = MemoryUpdate::content("new text")
            .with_importance(0.9)
            .with_metadata("a", serde_json::json!(1))
            .with_source("chat")
            .with_kind(MemoryType::Summary);
        assert_eq!(update.content.as_deref(), Some("new text"));
        assert_eq!(update.importance, Some(0.9));
        assert_eq!(update.metadata.len(), 1);
        assert_eq!(update.source.as_deref(), Some("chat"));
        assert_eq!(update.kind, Some(MemoryType::Summary));
        assert!(!update.is_empty());
        assert!(update.validate().is_ok());
    }
    #[test]
    fn empty_update_is_reported() {
        let update = MemoryUpdate::default();
        assert!(update.is_empty());
        assert!(update.validate().is_ok());
        assert!(!MemoryUpdate::content("x").is_empty());
        assert!(!MemoryUpdate::default()
            .with_metadata("a", serde_json::json!(1))
            .is_empty());
    }
    #[test]
    fn update_validate_rejects_bad_input() {
        assert!(MemoryUpdate::content("  ")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("content"));
        assert!(MemoryUpdate::default()
            .with_kind(MemoryType::Variable)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("variable"));
        assert!(MemoryUpdate::default()
            .with_importance(-0.1)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("importance"));
    }
    #[test]
    fn update_serde_round_trip() {
        let update = MemoryUpdate::content("x").with_importance(0.3);
        let json = serde_json::to_value(&update).unwrap();
        let restored: MemoryUpdate = serde_json::from_value(json).unwrap();
        assert_eq!(restored, update);
        let restored: MemoryUpdate = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, MemoryUpdate::default());
    }
}
