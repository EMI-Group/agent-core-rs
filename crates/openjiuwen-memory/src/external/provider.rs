//! The provider SPI: [`MemoryProvider`], capability flags, and health reports.
//!
//! A provider is the seam through which the memory surface is delegated to a backend
//! other than the in-process engine: a vendor service, a remote store, or the local
//! engine adapted by [`crate::external::EngineProvider`].
//!
//! # Contract
//!
//! - Every method takes `&self`: one `Arc<dyn MemoryProvider>` is shared freely across
//!   tasks, and a provider owns its own interior synchronization.
//! - Every method is cancel-safe. Dropping the returned future leaves the backend
//!   consistent: a cancelled write is either fully applied or not applied at all.
//! - [`MemoryProvider::initialize`] is idempotent; the second and later calls succeed
//!   without repeating backend setup.
//! - Results are never silently truncated: a capability a backend does not have is
//!   reported as [`openjiuwen_common::error::Error::Unsupported`], never as an empty
//!   list or a zero count.
//!
//! # Error kinds
//!
//! | Situation | Kind |
//! | --- | --- |
//! | invalid scope, record, filter, or search option | `Error::Validation` |
//! | unknown record id passed to [`MemoryProvider::update`] | `Error::NotFound` |
//! | capability the backend does not implement | `Error::Unsupported` |
//! | cancelled call | `Error::Cancelled` |
//! | elapsed deadline | `Error::Timeout` |
//! | backend or transport failure | the backend's own kind (`Error::Transport`, `Error::Memory`, …) |
//!
//! The capability-dependent `Unsupported` cases are:
//!
//! - [`MemoryProvider::initialize`] — the backend's SDK or transport is absent from this
//!   build (the vendor adapters in [`crate::external::vendor`]).
//! - [`MemoryProvider::search`] — the caller asks for a [`SearchMode`] that
//!   [`ProviderCapabilities`] does not declare, for example semantic recall without an
//!   embedding backend.
//! - [`MemoryProvider::add`] — the backend is read-only, or a batch larger than
//!   [`ProviderCapabilities::max_batch`] is submitted (a provider may prefer
//!   `Error::Validation` there).
//! - [`MemoryProvider::health`] — there is no reachable backend to probe. A backend that
//!   answers but is degraded reports `Ok(ProviderHealth { healthy: false, .. })` instead.
//!
//! A provider must never panic on user input and must never swallow an error into an
//! empty success.
//!
//! [`SearchMode`]: crate::record::SearchMode

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

/// The operations a backend actually implements.
///
/// The default is deliberately conservative — every flag `false` and no batch
/// ceiling — so a provider that forgets to declare a capability is never believed to
/// support it. A provider must report exactly what it can do: callers branch on these
/// flags before issuing a call that would otherwise fail with
/// [`openjiuwen_common::error::Error::Unsupported`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProviderCapabilities {
    /// Vector/semantic recall is available.
    pub semantic_search: bool,
    /// Keyword (lexical) recall is available.
    pub keyword_search: bool,
    /// Entity/relation graph memory is available.
    pub graph: bool,
    /// Scope variables (name → value) are available.
    pub variables: bool,
    /// Conversation message history is available.
    pub message_history: bool,
    /// `list` supports caller-driven paging (offset/limit) rather than full scans.
    pub pagination: bool,
    /// Maximum number of memories accepted by one [`MemoryProvider::add`] call.
    ///
    /// `None` means the provider does not impose a documented ceiling; it may still
    /// reject a batch with [`openjiuwen_common::error::Error::Validation`].
    pub max_batch: Option<usize>,
}

impl Default for ProviderCapabilities {
    /// No declared capability and no batch ceiling.
    fn default() -> Self {
        Self {
            semantic_search: false,
            keyword_search: false,
            graph: false,
            variables: false,
            message_history: false,
            pagination: false,
            max_batch: None,
        }
    }
}

impl ProviderCapabilities {
    /// The capabilities of a record backend with no optional feature at all.
    ///
    /// This is [`ProviderCapabilities::default`], exposed under a name that reads better
    /// at the call site.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// The capabilities of the in-process engine adapted by
    /// [`crate::external::EngineProvider`].
    ///
    /// Record CRUD, listing, and keyword recall are always available; semantic recall is
    /// available only when the engine has an embedding model. Graph memory, variables,
    /// message history, and paging are facade concerns outside this record SPI.
    #[must_use]
    pub fn local(has_embedding: bool) -> Self {
        Self {
            semantic_search: has_embedding,
            keyword_search: true,
            max_batch: None,
            ..Self::default()
        }
    }
}

/// The outcome of a [`MemoryProvider::health`] probe.
///
/// `healthy` is `true` only when the backend answered a cheap read. `detail` carries the
/// provider's own diagnostic (never a credential); `checked_at` is the probe time so a
/// caller can age the result out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProviderHealth {
    /// Whether the backend answered its probe.
    pub healthy: bool,
    /// Provider-supplied diagnostic; absent when there is nothing to add.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// When the probe ran.
    pub checked_at: DateTime<Utc>,
}

impl Default for ProviderHealth {
    /// An unchecked probe: not healthy, no detail, checked now.
    fn default() -> Self {
        Self {
            healthy: false,
            detail: None,
            checked_at: Utc::now(),
        }
    }
}

impl ProviderHealth {
    /// A healthy report checked now.
    #[must_use]
    pub fn healthy() -> Self {
        Self {
            healthy: true,
            ..Self::default()
        }
    }

    /// A degraded report checked now, carrying `detail`.
    #[must_use]
    pub fn degraded(detail: impl Into<String>) -> Self {
        Self {
            healthy: false,
            detail: Some(detail.into()),
            checked_at: Utc::now(),
        }
    }

    /// Returns the report with `detail` attached (healthy or not).
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// A memory backend behind one uniform, record-oriented surface.
///
/// Implementations are shared as `Arc<dyn MemoryProvider>`; see the module documentation
/// for the exact error contract, the cancel-safety guarantee, and the operations that may
/// report [`openjiuwen_common::error::Error::Unsupported`].
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// The stable provider id used as the registry key (`"local"`, `"mem0"`, …).
    fn name(&self) -> &str;

    /// The operations this backend implements.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Prepares the backend for use; idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Unsupported`] when the backend's SDK or
    /// transport is not part of this build, and the backend's own error kind when setup
    /// fails.
    async fn initialize(&self) -> Result<()>;

    /// Stores `memories` under `scope`, applying the dedupe policy in `options`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid scope,
    /// memory, or an oversized batch, [`openjiuwen_common::error::Error::Cancelled`] or
    /// [`openjiuwen_common::error::Error::Timeout`] for a cancelled or expired call, and
    /// the backend error when the write fails.
    async fn add(
        &self,
        scope: &MemoryScope,
        memories: Vec<NewMemory>,
        options: &AddOptions,
    ) -> Result<AddOutcome>;

    /// Recalls the memories of `scope` that match `query`, best hit first.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid scope,
    /// blank query, or invalid options, [`openjiuwen_common::error::Error::Unsupported`]
    /// when the requested mode is not declared by [`Self::capabilities`], and the backend
    /// error when the read fails.
    async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>>;

    /// Reads one record, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid scope and
    /// the backend error when the read fails.
    async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;

    /// Applies `update` to the record `id` and returns the stored record.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::NotFound`] when `id` does not exist,
    /// [`openjiuwen_common::error::Error::Validation`] for an invalid scope or patch, and
    /// the backend error when the write fails.
    async fn update(
        &self,
        scope: &MemoryScope,
        id: &str,
        update: MemoryUpdate,
        options: &CallOptions,
    ) -> Result<MemoryRecord>;

    /// Deletes one record, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid scope and
    /// the backend error when the delete fails.
    async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool>;

    /// Lists every record matching `filter`, without scoring.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid filter and
    /// the backend error when the listing fails.
    async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;

    /// Deletes every record matching `filter`, returning how many were removed.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid filter and
    /// the backend error when a delete fails.
    async fn clear(&self, filter: &MemoryFilter) -> Result<usize>;

    /// Probes the backend with a cheap read and reports its state.
    ///
    /// A backend that answers but is degraded returns `Ok(ProviderHealth { healthy:
    /// false, .. })`; [`openjiuwen_common::error::Error::Unsupported`] is reserved for a
    /// backend that cannot be probed at all.
    async fn health(&self) -> Result<ProviderHealth>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    #[test]
    fn capabilities_default_is_conservative() {
        let capabilities = ProviderCapabilities::default();
        assert_eq!(capabilities, ProviderCapabilities::none());
        assert!(!capabilities.semantic_search);
        assert!(!capabilities.keyword_search);
        assert!(!capabilities.graph);
        assert!(!capabilities.variables);
        assert!(!capabilities.message_history);
        assert!(!capabilities.pagination);
        assert_eq!(capabilities.max_batch, None);
    }

    #[test]
    fn local_capabilities_follow_the_embedding_presence() {
        let keyword_only = ProviderCapabilities::local(false);
        assert!(!keyword_only.semantic_search);
        assert!(keyword_only.keyword_search);
        assert_eq!(keyword_only.max_batch, None);
        let semantic = ProviderCapabilities::local(true);
        assert!(semantic.semantic_search);
        assert!(semantic.keyword_search);
    }

    #[test]
    fn capabilities_serde_round_trip_and_defaults() {
        let capabilities = ProviderCapabilities::local(true);
        let json = serde_json::to_value(capabilities).unwrap();
        assert_eq!(json["semantic_search"], serde_json::json!(true));
        assert_eq!(json["keyword_search"], serde_json::json!(true));
        assert_eq!(json["graph"], serde_json::json!(false));
        assert_eq!(json["max_batch"], serde_json::Value::Null);
        let restored: ProviderCapabilities = serde_json::from_value(json).unwrap();
        assert_eq!(restored, ProviderCapabilities::local(true));
        let restored: ProviderCapabilities = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, ProviderCapabilities::default());
    }

    #[test]
    fn healthy_and_degraded_health_reports() {
        let healthy = ProviderHealth::healthy();
        assert!(healthy.healthy);
        assert!(healthy.detail.is_none());
        let degraded = ProviderHealth::degraded("store unreachable");
        assert!(!degraded.healthy);
        assert_eq!(degraded.detail.as_deref(), Some("store unreachable"));
        let annotated = ProviderHealth::healthy().with_detail("probe ok");
        assert!(annotated.healthy);
        assert_eq!(annotated.detail.as_deref(), Some("probe ok"));
        assert!(!ProviderHealth::default().healthy);
    }

    #[test]
    fn provider_health_serializes_with_an_rfc_3339_timestamp() {
        let health = ProviderHealth {
            healthy: true,
            detail: Some("ok".to_owned()),
            checked_at: DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
        };
        let json = serde_json::to_value(&health).unwrap();
        assert_eq!(json["healthy"], serde_json::json!(true));
        assert_eq!(json["detail"], serde_json::json!("ok"));
        assert_eq!(
            json["checked_at"],
            serde_json::json!("2023-11-14T22:13:20Z")
        );
        let restored: ProviderHealth = serde_json::from_value(json).unwrap();
        assert_eq!(restored, health);
        let restored: ProviderHealth = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(!restored.healthy);
        assert!(restored.detail.is_none());
        assert!(restored.checked_at <= Utc::now());
    }

    #[test]
    fn the_trait_is_object_safe_and_send_sync() {
        assert_send_sync_static::<Arc<dyn MemoryProvider>>();

        fn takes_provider(_provider: &dyn MemoryProvider) {}

        struct NullProvider;

        #[async_trait]
        impl MemoryProvider for NullProvider {
            fn name(&self) -> &'static str {
                "null"
            }

            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::none()
            }

            async fn initialize(&self) -> Result<()> {
                Ok(())
            }

            async fn add(
                &self,
                _scope: &MemoryScope,
                _memories: Vec<NewMemory>,
                _options: &AddOptions,
            ) -> Result<AddOutcome> {
                Ok(AddOutcome::default())
            }

            async fn search(
                &self,
                _scope: &MemoryScope,
                _query: &str,
                _options: &SearchOptions,
            ) -> Result<Vec<RecallHit>> {
                Ok(Vec::new())
            }

            async fn get(&self, _scope: &MemoryScope, _id: &str) -> Result<Option<MemoryRecord>> {
                Ok(None)
            }

            async fn update(
                &self,
                _scope: &MemoryScope,
                _id: &str,
                _update: MemoryUpdate,
                _options: &CallOptions,
            ) -> Result<MemoryRecord> {
                Err(openjiuwen_common::error::Error::unsupported("read-only"))
            }

            async fn delete(&self, _scope: &MemoryScope, _id: &str) -> Result<bool> {
                Ok(false)
            }

            async fn list(&self, _filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
                Ok(Vec::new())
            }

            async fn clear(&self, _filter: &MemoryFilter) -> Result<usize> {
                Ok(0)
            }

            async fn health(&self) -> Result<ProviderHealth> {
                Ok(ProviderHealth::healthy())
            }
        }

        let provider: Arc<dyn MemoryProvider> = Arc::new(NullProvider);
        assert_eq!(provider.name(), "null");
        takes_provider(provider.as_ref());
    }
}
