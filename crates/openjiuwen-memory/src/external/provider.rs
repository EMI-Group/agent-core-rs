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
//! # Tool rail
//!
//! The Python SPI is tool-rail oriented, so the trait also carries the rail as
//! *provided* methods — a backend gets a working rail for free and overrides only what it
//! wants to change:
//!
//! | Method | Default |
//! | --- | --- |
//! | [`MemoryProvider::is_available`] | any declared capability ([`ProviderCapabilities::none`] means "not configured"); no network call |
//! | [`MemoryProvider::is_initialized`] | `false` |
//! | [`MemoryProvider::default_scope`] | [`MemoryScope::default`]; vendors derive it from their configuration |
//! | [`MemoryProvider::get_tool_schemas`] | the four vendor-neutral specs from [`crate::external::rail::default_tool_schemas`] |
//! | [`MemoryProvider::handle_tool_call`] | dispatches those four tools against [`MemoryProvider::default_scope`] |
//! | [`MemoryProvider::prefetch`] | one recall rendered by [`crate::prompt::render_snippets`] |
//! | [`MemoryProvider::sync_turn`] | one episodic memory holding the exchange |
//! | [`MemoryProvider::system_prompt_block`] | `""` |
//! | [`MemoryProvider::on_session_end`] | no-op |
//!
//! The rail's error policy differs from the record SPI on purpose, and matches Python:
//! [`MemoryProvider::handle_tool_call`] returns a `{"error": …}` object and logs the
//! cause at `debug` instead of failing the turn, and [`MemoryProvider::prefetch`] returns
//! an empty string, because a rail tool is model-facing and a raised error would abort the
//! conversation. [`MemoryProvider::sync_turn`] and [`MemoryProvider::on_session_end`] are
//! caller-facing and therefore keep returning [`openjiuwen_common::error::Error`].
//!
//! [`SearchMode`]: crate::record::SearchMode

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use super::rail;
use crate::analyzer::ConversationTurn;
use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

/// The tracing target of provider and rail diagnostics.
const TARGET: &str = "openjiuwen_memory::external";

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

    /// Whether the provider is configured and ready, evaluated without any network call.
    ///
    /// The default reports readiness as soon as the backend declares one capability, so a
    /// provider with no credential and no backend reports `false`. A vendor whose readiness
    /// depends on configuration overrides this with the exact check (Python's `mem0`
    /// provider answers "an API key is set").
    fn is_available(&self) -> bool {
        rail::is_available(&self.capabilities())
    }

    /// Whether [`Self::initialize`] has run on this provider.
    ///
    /// The SPI cannot observe that for an arbitrary backend, so the default is `false`; a
    /// provider that tracks its own setup state overrides it.
    fn is_initialized(&self) -> bool {
        false
    }

    /// The scope the tool rail reads and writes when a tool call carries no scope.
    ///
    /// Defaults to [`MemoryScope::default`] (the `__default__` user and scope). A vendor
    /// derives it from its configuration — the user id it was built for, its workspace —
    /// because a rail tool never receives a full [`MemoryScope`].
    fn default_scope(&self) -> MemoryScope {
        MemoryScope::default()
    }

    /// The JSON-Schema tool specs this provider advertises to a model.
    ///
    /// Defaults to the four vendor-neutral specs of
    /// [`crate::external::rail::default_tool_schemas`]; a provider that overrides this must
    /// override [`Self::handle_tool_call`] to accept the same names.
    fn get_tool_schemas(&self) -> Vec<Value> {
        rail::default_tool_schemas()
    }

    /// Executes one model-issued tool call and returns its JSON payload.
    ///
    /// The default dispatch handles [`rail::ADD_MEMORY_TOOL`], [`rail::SEARCH_MEMORY_TOOL`],
    /// [`rail::LIST_MEMORIES_TOOL`], and [`rail::DELETE_MEMORY_TOOL`] against
    /// [`Self::default_scope`]:
    ///
    /// - `add_memory` → `{"result": "stored", "id", "content"}`
    /// - `search_memory` → `{"results": [{"id", "content", "score", "type"}], "count"}`
    /// - `list_memories` → `{"memories": [{"id", "content", "type", "importance"}], "count"}`
    /// - `delete_memory` → `{"result": "deleted" | "not_found", "id"}`
    ///
    /// An unknown tool name answers `{"error": "unknown tool: <name>"}`. A validation or
    /// backend failure is **not** propagated — the payload is `{"error": <message>}` and
    /// the cause is logged at `debug`, mirroring the Python rail, which never raises out of
    /// a tool call because a raised error would abort the conversation. The error is never
    /// dropped silently: `tracing` always records it.
    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        let scope = self.default_scope();
        match tool {
            rail::ADD_MEMORY_TOOL => {
                let memory = match rail::parse_add_args(args) {
                    Ok(memory) => memory,
                    Err(error) => return rail::tool_error(error.to_string()),
                };
                match self.add(&scope, vec![memory], &AddOptions::default()).await {
                    Ok(outcome) => outcome
                        .added
                        .first()
                        .or_else(|| outcome.merged.first())
                        .map_or_else(
                            || rail::tool_error("the memory was neither added nor merged"),
                            rail::stored_result,
                        ),
                    Err(error) => return tool_failure(self.name(), tool, &error),
                }
            }
            rail::SEARCH_MEMORY_TOOL => {
                let (query, options) = match rail::parse_search_args(args) {
                    Ok(parsed) => parsed,
                    Err(error) => return rail::tool_error(error.to_string()),
                };
                match self.search(&scope, &query, &options).await {
                    Ok(hits) => rail::search_results(&hits),
                    Err(error) => return tool_failure(self.name(), tool, &error),
                }
            }
            rail::LIST_MEMORIES_TOOL => {
                let map = rail::args_object(args);
                let mut filter = MemoryFilter::for_scope(&scope);
                if let Some(raw) = rail::string_arg(map, "kind")
                    .map(str::trim)
                    .filter(|raw| !raw.is_empty())
                {
                    match rail::parse_kind(raw) {
                        Ok(kind) => filter.kinds.push(kind),
                        Err(error) => return rail::tool_error(error.to_string()),
                    }
                }
                let limit = rail::integer_arg(map, "limit")
                    .map_or(rail::DEFAULT_LIST_LIMIT, |limit| {
                        usize::try_from(limit)
                            .unwrap_or(rail::DEFAULT_LIST_LIMIT)
                            .max(1)
                    })
                    .min(rail::MAX_LIST_LIMIT);
                match self.list(&filter).await {
                    Ok(mut records) => {
                        records.truncate(limit);
                        rail::memory_listing(&records)
                    }
                    Err(error) => return tool_failure(self.name(), tool, &error),
                }
            }
            rail::DELETE_MEMORY_TOOL => {
                let id = match rail::string_arg(rail::args_object(args), "id")
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                {
                    Some(id) => id.to_owned(),
                    None => return rail::tool_error("missing required parameter: id"),
                };
                match self.delete(&scope, &id).await {
                    Ok(existed) => rail::deleted_result(&id, existed),
                    Err(error) => return tool_failure(self.name(), tool, &error),
                }
            }
            other => rail::tool_error(format!("unknown tool: {other}")),
        }
    }

    /// Renders the memories relevant to `query` as prompt text for the current turn.
    ///
    /// A blank query returns `""` without touching the backend. Otherwise one recall of
    /// [`rail::DEFAULT_PREFETCH_TOP_K`] hits runs against [`Self::default_scope`] and is
    /// rendered by [`crate::prompt::render_snippets`] within
    /// [`rail::DEFAULT_PREFETCH_MAX_TOKENS`]; no hits, or a failed recall, render to `""`
    /// so a degraded memory backend never breaks the turn (the failure is logged at
    /// `debug`).
    async fn prefetch(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }
        let options = SearchOptions::default().with_top_k(rail::DEFAULT_PREFETCH_TOP_K);
        match self.search(&self.default_scope(), query, &options).await {
            Ok(hits) => crate::prompt::render_snippets(&hits, rail::DEFAULT_PREFETCH_MAX_TOKENS),
            Err(error) => {
                tracing::debug!(
                    target: TARGET,
                    provider = %self.name(),
                    %error,
                    "memory rail prefetch failed; the turn continues without recalled memories"
                );
                String::new()
            }
        }
    }

    /// Records one exchange as an episodic memory.
    ///
    /// Both messages must hold a non-whitespace character; otherwise nothing is written and
    /// `Ok(())` is returned (Python's rail skips a half turn the same way). The stored
    /// content is `"User: <user>\nAssistant: <assistant>"` with source `"conversation"`, so
    /// a later recall can tell a synced exchange from an explicitly stored fact.
    ///
    /// # Errors
    ///
    /// Returns the backend's write error unchanged; a session that cannot persist its turn
    /// may decide to log it, but the SPI never hides it.
    async fn sync_turn(&self, user_msg: &str, assistant_msg: &str) -> Result<()> {
        if user_msg.trim().is_empty() || assistant_msg.trim().is_empty() {
            return Ok(());
        }
        let memory = NewMemory::episodic(format!("User: {user_msg}\nAssistant: {assistant_msg}"))
            .with_source("conversation");
        self.add(&self.default_scope(), vec![memory], &AddOptions::default())
            .await
            .map(|_outcome| ())
    }

    /// Provider guidance to splice into the agent's system prompt.
    ///
    /// The default is empty: a backend with nothing to say about its own tools stays out of
    /// the prompt. A provider that advertises tools overrides this to explain them, as
    /// Python's `mem0` provider does.
    fn system_prompt_block(&self) -> String {
        String::new()
    }

    /// Finalises a session with every turn it produced.
    ///
    /// The default is a no-op: a provider that consolidates at session end (one summary
    /// write, one remote flush) overrides it. A turn list may be empty.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the finalisation write fails.
    async fn on_session_end(&self, messages: &[ConversationTurn]) -> Result<()> {
        let _ = messages;
        Ok(())
    }
}

/// Turns a rail tool failure into the `{"error": …}` payload, logging the cause.
///
/// The Python rail returns the exception text to the model; the diagnostic stays on the
/// `debug` channel here so the message surface is unchanged while operators keep the cause.
fn tool_failure(provider: &str, tool: &str, error: &openjiuwen_common::error::Error) -> Value {
    tracing::debug!(
        target: TARGET,
        provider,
        tool,
        %error,
        "memory rail tool call failed"
    );
    rail::tool_error(error.to_string())
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

    /// A provider with no backend at all: it declares no capability and every record call
    /// fails, which is exactly the shape the rail must degrade from.
    struct UnbackedProvider;

    /// The error every [`UnbackedProvider`] call returns.
    fn no_backend() -> openjiuwen_common::error::Error {
        openjiuwen_common::error::Error::unsupported("no backend configured")
    }

    #[async_trait]
    impl MemoryProvider for UnbackedProvider {
        fn name(&self) -> &'static str {
            "unbacked"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::none()
        }

        async fn initialize(&self) -> Result<()> {
            Err(no_backend())
        }

        async fn add(
            &self,
            _scope: &MemoryScope,
            _memories: Vec<NewMemory>,
            _options: &AddOptions,
        ) -> Result<AddOutcome> {
            Err(no_backend())
        }

        async fn search(
            &self,
            _scope: &MemoryScope,
            _query: &str,
            _options: &SearchOptions,
        ) -> Result<Vec<RecallHit>> {
            Err(no_backend())
        }

        async fn get(&self, _scope: &MemoryScope, _id: &str) -> Result<Option<MemoryRecord>> {
            Err(no_backend())
        }

        async fn update(
            &self,
            _scope: &MemoryScope,
            _id: &str,
            _update: MemoryUpdate,
            _options: &CallOptions,
        ) -> Result<MemoryRecord> {
            Err(no_backend())
        }

        async fn delete(&self, _scope: &MemoryScope, _id: &str) -> Result<bool> {
            Err(no_backend())
        }

        async fn list(&self, _filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
            Err(no_backend())
        }

        async fn clear(&self, _filter: &MemoryFilter) -> Result<usize> {
            Err(no_backend())
        }

        async fn health(&self) -> Result<ProviderHealth> {
            Err(no_backend())
        }
    }

    /// A local rail provider over a fresh in-memory engine.
    async fn local_rail() -> crate::external::EngineProvider {
        let memory =
            crate::facade::LongTermMemory::in_memory(crate::config::MemoryEngineConfig::default())
                .await
                .expect("in-memory facade");
        crate::external::EngineProvider::new(Arc::clone(memory.engine()))
    }

    #[test]
    fn the_default_tool_set_is_advertised_without_a_backend() {
        let provider = UnbackedProvider;
        let schemas = provider.get_tool_schemas();
        assert_eq!(schemas.len(), 4);
        let mut names: Vec<&str> = schemas
            .iter()
            .map(|schema| schema["name"].as_str().expect("tool name"))
            .collect();
        assert_eq!(
            names,
            vec![
                rail::ADD_MEMORY_TOOL,
                rail::SEARCH_MEMORY_TOOL,
                rail::LIST_MEMORIES_TOOL,
                rail::DELETE_MEMORY_TOOL
            ]
        );
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 4, "tool names are unique");
        for schema in &schemas {
            assert_eq!(schema["parameters"]["type"], serde_json::json!("object"));
            assert!(schema["parameters"]["properties"].is_object());
            assert!(schema["parameters"]["required"].is_array());
        }

        assert!(!provider.is_available(), "no declared capability");
        assert!(!provider.is_initialized());
        assert_eq!(provider.default_scope(), MemoryScope::default());
        assert_eq!(provider.system_prompt_block(), "");
    }

    #[tokio::test]
    async fn handle_tool_call_round_trips_every_default_tool() {
        let provider = local_rail().await;

        let stored = provider
            .handle_tool_call(
                rail::ADD_MEMORY_TOOL,
                &serde_json::json!({ "content": "alice prefers dark mode" }),
            )
            .await;
        assert_eq!(stored["result"], serde_json::json!("stored"));
        assert_eq!(
            stored["content"],
            serde_json::json!("alice prefers dark mode")
        );
        let id = stored["id"].as_str().expect("stored id").to_owned();

        let listing = provider
            .handle_tool_call(rail::LIST_MEMORIES_TOOL, &serde_json::json!({}))
            .await;
        assert_eq!(listing["count"], serde_json::json!(1));
        assert_eq!(listing["memories"][0]["id"], serde_json::json!(id));
        assert_eq!(
            listing["memories"][0]["type"],
            serde_json::json!("semantic_memory")
        );

        let narrowed = provider
            .handle_tool_call(
                rail::LIST_MEMORIES_TOOL,
                &serde_json::json!({ "kind": "coding", "limit": 5 }),
            )
            .await;
        assert_eq!(narrowed["count"], serde_json::json!(0));
        let bad_kind = provider
            .handle_tool_call(
                rail::LIST_MEMORIES_TOOL,
                &serde_json::json!({ "kind": "nope" }),
            )
            .await;
        assert!(
            bad_kind["error"]
                .as_str()
                .expect("error payload")
                .contains("unknown memory kind"),
            "{bad_kind}"
        );

        let searched = provider
            .handle_tool_call(
                rail::SEARCH_MEMORY_TOOL,
                &serde_json::json!({ "query": "dark mode" }),
            )
            .await;
        assert_eq!(searched["count"], serde_json::json!(1));
        assert_eq!(
            searched["results"][0]["content"],
            serde_json::json!("alice prefers dark mode")
        );

        let deleted = provider
            .handle_tool_call(rail::DELETE_MEMORY_TOOL, &serde_json::json!({ "id": id }))
            .await;
        assert_eq!(
            deleted,
            serde_json::json!({ "result": "deleted", "id": id })
        );
        let again = provider
            .handle_tool_call(rail::DELETE_MEMORY_TOOL, &serde_json::json!({ "id": id }))
            .await;
        assert_eq!(
            again,
            serde_json::json!({ "result": "not_found", "id": id })
        );

        let unknown = provider
            .handle_tool_call("nope", &serde_json::json!({}))
            .await;
        assert_eq!(
            unknown,
            serde_json::json!({ "error": "unknown tool: nope" })
        );

        let missing = provider
            .handle_tool_call(rail::ADD_MEMORY_TOOL, &serde_json::json!({}))
            .await;
        assert!(
            missing["error"]
                .as_str()
                .expect("error payload")
                .contains("missing required parameter: content"),
            "{missing}"
        );
    }

    #[tokio::test]
    async fn prefetch_skips_blank_queries_and_renders_hits() {
        let provider = local_rail().await;
        assert_eq!(provider.prefetch("").await, "");
        assert_eq!(provider.prefetch("   ").await, "");
        assert_eq!(provider.prefetch("nothing is stored yet").await, "");

        provider
            .handle_tool_call(
                rail::ADD_MEMORY_TOOL,
                &serde_json::json!({ "content": "alice prefers dark mode" }),
            )
            .await;
        let rendered = provider.prefetch("dark mode").await;
        assert!(rendered.contains("alice prefers dark mode"), "{rendered}");
        assert!(rendered.starts_with("## "), "{rendered}");
    }

    #[tokio::test]
    async fn sync_turn_stores_one_episodic_memory_and_skips_blank_turns() {
        let memory =
            crate::facade::LongTermMemory::in_memory(crate::config::MemoryEngineConfig::default())
                .await
                .expect("in-memory facade");
        let provider = crate::external::EngineProvider::new(Arc::clone(memory.engine()));
        let scope = MemoryScope::default();

        provider
            .sync_turn("   ", "an answer")
            .await
            .expect("blank user");
        provider
            .sync_turn("a question", "  ")
            .await
            .expect("blank assistant");
        assert_eq!(
            memory.list(&scope, &[]).await.expect("list"),
            Vec::new(),
            "a half turn is not written"
        );

        provider
            .sync_turn("what theme?", "dark mode")
            .await
            .expect("sync");
        let listed = memory.list(&scope, &[]).await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].kind, crate::record::MemoryType::EpisodicMemory);
        assert_eq!(listed[0].content, "User: what theme?\nAssistant: dark mode");
        assert_eq!(listed[0].source.as_deref(), Some("conversation"));
    }

    #[tokio::test]
    async fn the_rail_degrades_instead_of_raising() {
        let provider = UnbackedProvider;
        let payload = provider
            .handle_tool_call(
                rail::ADD_MEMORY_TOOL,
                &serde_json::json!({ "content": "x" }),
            )
            .await;
        assert!(
            payload["error"]
                .as_str()
                .expect("error payload")
                .contains("no backend configured"),
            "{payload}"
        );
        let search = provider
            .handle_tool_call(
                rail::SEARCH_MEMORY_TOOL,
                &serde_json::json!({ "query": "x" }),
            )
            .await;
        assert_eq!(
            search,
            serde_json::json!({ "error": "unsupported: no backend configured" })
        );
        assert_eq!(provider.prefetch("query").await, "");
        provider
            .on_session_end(&[ConversationTurn::new("user", "hi")])
            .await
            .expect("no-op session end");
        provider
            .on_session_end(&[])
            .await
            .expect("empty session end");
    }
}
