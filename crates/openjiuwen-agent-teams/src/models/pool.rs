//! The team's model pool: an ordered, validated list of LLM endpoints.
//!
//! Python parity source: `openjiuwen/agent_teams/models/pool.py` (`ModelPoolEntry`,
//! `model_group_label`, and the duplicate/blank-name validation `RouterAllocator` applies).
//!
//! A pool is *ordered data*, never a set: the position of an entry is its stable identity inside a
//! team (`MemberRecord::model_ref` stores `(model_name, index)`), and every routing strategy reads
//! the pool in declaration order. [`ModelPool::new`] therefore validates but never re-sorts.
//!
//! The pool deliberately knows nothing about allocation or routing: [`super::ModelRouter`] owns the
//! per-request choice and `crate::runtime::ModelAllocator` owns the per-member choice.

use std::collections::HashMap;
use std::fmt;

use openjiuwen_common::error::{Error, Result};

use crate::schema::ModelPoolEntry;

/// Provider key of a canonical model-group entry (Python `INTELLI_ROUTER_PROVIDER`).
const INTELLI_ROUTER_PROVIDER: &str = "intelli_router";

/// Logical model name of a canonical model-group entry (Python `INTELLI_ROUTER_UNIFIED_MODEL`).
const INTELLI_ROUTER_UNIFIED_MODEL: &str = "*";

/// An ordered, validated list of the LLM endpoints a team may use.
///
/// The pool keeps pool order verbatim and indexes entries by `model_name` for lookup; because
/// duplicate names are rejected at construction, a name maps to exactly one position.
/// `ModelPoolEntry::api_key` is a `SecretString`, so cloning a pool copies credentials while
/// [`fmt::Debug`] output only ever shows the model names.
#[derive(Clone, PartialEq)]
pub struct ModelPool {
    /// Entries in declaration order; the position is the entry's identity.
    entries: Vec<ModelPoolEntry>,
    /// `model_name` → pool position, so lookups never scan.
    by_name: HashMap<String, usize>,
}

impl ModelPool {
    /// Builds a pool from `entries`, validating every entry.
    ///
    /// Pool order is preserved exactly as given — entries are never sorted or de-duplicated
    /// silently, because the index of an entry is what a persisted member reference points at.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when an entry has a blank (empty or whitespace-only)
    /// `model_name` — the message names the pool position — or when two entries share a
    /// `model_name`, which would make by-name routing ambiguous.
    pub fn new(entries: Vec<ModelPoolEntry>) -> Result<Self> {
        let mut by_name = HashMap::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            if entry.model_name.trim().is_empty() {
                return Err(Error::validation(format!(
                    "model_pool entry at index {index} has a blank model_name"
                )));
            }
            if let Some(first) = by_name.insert(entry.model_name.clone(), index) {
                return Err(Error::validation(format!(
                    "model_pool declares duplicate model name {:?} at indices {first} and {index}",
                    entry.model_name
                )));
            }
        }
        Ok(Self { entries, by_name })
    }

    /// Builds a pool by copying `entries` (see [`ModelPool::new`] for validation).
    ///
    /// # Errors
    ///
    /// Same as [`ModelPool::new`].
    pub fn from_entries(entries: &[ModelPoolEntry]) -> Result<Self> {
        Self::new(entries.to_vec())
    }

    /// Returns a pool with no entries: the state of a team that declares no `model_pool`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            by_name: HashMap::new(),
        }
    }

    /// Returns `true` when the pool has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of pooled endpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the entries in pool order.
    #[must_use]
    pub fn entries(&self) -> &[ModelPoolEntry] {
        &self.entries
    }

    /// Returns the entry at pool position `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&ModelPoolEntry> {
        self.entries.get(index)
    }

    /// Returns the pool position and entry serving `model_name`.
    ///
    /// `None` means the pool does not serve that name; callers then fall back to the first entry
    /// rather than failing, so a stale member model still runs somewhere.
    #[must_use]
    pub fn find(&self, model_name: &str) -> Option<(usize, &ModelPoolEntry)> {
        let index = *self.by_name.get(model_name)?;
        self.entries.get(index).map(|entry| (index, entry))
    }

    /// Returns `true` when some entry serves `model_name`.
    #[must_use]
    pub fn contains(&self, model_name: &str) -> bool {
        self.by_name.contains_key(model_name)
    }

    /// Returns every model name in pool order.
    ///
    /// Names repeat only if the caller built the pool without [`ModelPool::new`]; a validated pool
    /// yields one name per entry.
    #[must_use]
    pub fn model_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.model_name.as_str())
            .collect()
    }

    /// Returns the human-readable name of the model group this pool describes.
    ///
    /// Python parity: `models.pool.model_group_label`. A *canonical model-group pool* is a single
    /// entry whose `model_name` is `"*"` and whose provider is `"intelli_router"`; its label is
    /// `metadata.model_group.display_name`, else
    /// `metadata.client.intelli_router.model_group_id`, else `"*"`.
    ///
    /// Deviation from Python: Python returns `None` for a pool that is not a model-group pool,
    /// because only hosts that show `IntelliRouter` groups need a label. Here the function returns
    /// `String` for every pool — the caller always has something to print — so a plain pool is
    /// labelled with its distinct model names joined by `", "` in pool order (empty string for an
    /// empty pool). `IntelliRouter` itself is deferred, but the label stays wire-compatible so a pool
    /// round-tripped through Python renders identically.
    #[must_use]
    pub fn model_group_label(&self) -> String {
        if let Some(label) = self.intelli_router_label() {
            return label;
        }
        let mut names: Vec<&str> = Vec::new();
        for entry in &self.entries {
            let name = entry.model_name.as_str();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        names.join(", ")
    }

    /// Returns the `IntelliRouter` model-group label of a canonical model-group pool, if this is one.
    fn intelli_router_label(&self) -> Option<String> {
        let [entry] = self.entries.as_slice() else {
            return None;
        };
        if entry.model_name != INTELLI_ROUTER_UNIFIED_MODEL
            || entry.api_provider != INTELLI_ROUTER_PROVIDER
        {
            return None;
        }
        let router = entry
            .metadata
            .get("client")
            .and_then(serde_json::Value::as_object)
            .and_then(|client| client.get("intelli_router"))
            .and_then(serde_json::Value::as_object)?;
        let display_name = entry
            .metadata
            .get("model_group")
            .and_then(serde_json::Value::as_object)
            .and_then(|group| group.get("display_name"))
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty());
        let group_id = router
            .get("model_group_id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty());
        Some(
            display_name
                .or(group_id)
                .unwrap_or(INTELLI_ROUTER_UNIFIED_MODEL)
                .to_owned(),
        )
    }
}

impl Default for ModelPool {
    /// An empty pool — the state of a team that declares no `model_pool`.
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for ModelPool {
    /// Prints the size and the model names; credentials stay out of the output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelPool")
            .field("entries", &self.entries.len())
            .field("model_names", &self.model_names())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    /// Builds `n` distinct entries named `m0`, `m1`, …
    fn entries(n: usize) -> Vec<ModelPoolEntry> {
        (0..n)
            .map(|i| {
                ModelPoolEntry::new(
                    format!("m{i}"),
                    "openai",
                    format!("https://endpoint-{i}.example/v1"),
                    format!("sk-{i}"),
                )
            })
            .collect()
    }

    #[test]
    fn pool_keeps_declaration_order_and_indexes_names() {
        let pool = ModelPool::new(entries(3)).expect("valid pool");
        assert_eq!(pool.len(), 3);
        assert!(!pool.is_empty());
        assert_eq!(pool.model_names(), vec!["m0", "m1", "m2"]);
        assert_eq!(
            pool.entries()
                .iter()
                .map(|e| e.model_name.as_str())
                .collect::<Vec<_>>(),
            vec!["m0", "m1", "m2"]
        );

        let (index, entry) = pool.find("m2").expect("m2 is pooled");
        assert_eq!(index, 2);
        assert_eq!(entry.model_name, "m2");
        assert!(pool.contains("m1"));
        assert!(!pool.contains("m9"));
        assert!(pool.find("m9").is_none());
        assert_eq!(pool.get(0).map(|e| e.model_name.as_str()), Some("m0"));
        assert!(pool.get(3).is_none());
    }

    #[test]
    fn duplicate_model_names_are_rejected() {
        let mut list = entries(2);
        list[1].model_name = "m0".to_owned();
        let error = ModelPool::new(list).expect_err("duplicate name");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.to_string();
        assert!(message.contains("duplicate"), "{message}");
        assert!(message.contains("m0"), "{message}");
        assert!(message.contains('0') && message.contains('1'), "{message}");
    }

    #[test]
    fn blank_model_name_is_rejected_with_the_pool_index() {
        for blank in ["", "   ", "\t"] {
            let mut list = entries(2);
            list[1].model_name = blank.to_owned();
            let error = ModelPool::new(list).expect_err("blank name");
            assert_eq!(error.kind(), ErrorKind::Validation);
            let message = error.to_string();
            assert!(message.contains("index 1"), "{message}");
            assert!(message.contains("blank"), "{message}");
        }
    }

    #[test]
    fn empty_pool_is_allowed() {
        let pool = ModelPool::new(Vec::new()).expect("empty pool is legal");
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.model_names(), Vec::<&str>::new());
        assert_eq!(pool.model_group_label(), "");
        assert_eq!(ModelPool::empty(), pool);
        assert_eq!(ModelPool::default(), pool);
        assert!(pool.find("m0").is_none());
    }

    #[test]
    fn from_entries_copies_the_slice() {
        let mut source = entries(2);
        let pool = ModelPool::from_entries(&source).expect("valid pool");
        source[0].model_name = "changed".to_owned();
        // The pool owns its entries: later edits to the caller's vector do not leak in.
        assert_eq!(pool.model_names(), vec!["m0", "m1"]);
    }

    #[test]
    fn model_group_label_joins_distinct_model_names_in_pool_order() {
        let pool = ModelPool::new(entries(2)).expect("valid pool");
        assert_eq!(pool.model_group_label(), "m0, m1");

        let single = ModelPool::new(entries(1)).expect("valid pool");
        assert_eq!(single.model_group_label(), "m0");
    }

    #[test]
    fn model_group_label_names_an_intelli_router_group() {
        let mut entry = ModelPoolEntry::new("*", "intelli_router", "https://router/v1", "sk");
        entry.metadata.insert(
            "client".to_owned(),
            json!({"intelli_router": {"model_group_id": "grp-42"}}),
        );
        let pool = ModelPool::new(vec![entry.clone()]).expect("valid pool");
        assert_eq!(pool.model_group_label(), "grp-42");

        // A host-supplied display name wins over the router's group id.
        entry.metadata.insert(
            "model_group".to_owned(),
            json!({"display_name": "Coding Team"}),
        );
        let named = ModelPool::new(vec![entry.clone()]).expect("valid pool");
        assert_eq!(named.model_group_label(), "Coding Team");

        // Without a router id the logical "*" name is the label.
        entry.metadata.insert("client".to_owned(), json!({}));
        let bare = ModelPool::new(vec![entry.clone()]).expect("valid pool");
        assert_eq!(bare.model_group_label(), "*");

        // A "*" name from another provider is not a model group.
        entry.api_provider = "openai".to_owned();
        let foreign = ModelPool::new(vec![entry]).expect("valid pool");
        assert_eq!(foreign.model_group_label(), "*");
    }

    #[test]
    fn model_group_label_needs_a_single_entry() {
        let mut first = ModelPoolEntry::new("*", "intelli_router", "https://router/v1", "sk");
        first.metadata.insert(
            "client".to_owned(),
            json!({"intelli_router": {"model_group_id": "grp-1"}}),
        );
        let second = ModelPoolEntry::new("m1", "openai", "https://endpoint/v1", "sk-1");
        let pool = ModelPool::new(vec![first, second]).expect("valid pool");
        assert_eq!(pool.model_group_label(), "*, m1");
    }

    #[test]
    fn debug_lists_names_but_not_credentials() {
        let pool = ModelPool::new(entries(2)).expect("valid pool");
        let debug = format!("{pool:?}");
        assert!(debug.contains("entries: 2"), "{debug}");
        assert!(debug.contains("m0"), "{debug}");
        assert!(!debug.contains("sk-0"), "{debug}");
        assert!(!debug.contains("endpoint-0"), "{debug}");
    }
}
