//! The per-team team-memory wiring a `TeamAgent` drives.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/manager.py` (`TeamMemoryManager`), which
//! owns one team's memory, binds the coordination session, registers the member tools, and
//! extracts memories when the session pauses or stops. Rust keeps the same shape as one handle:
//!
//! - [`TeamMemoryWiring::open`] builds the [`TeamMemory`] facade, the
//!   [`TeamMemoryHooks`] lifecycle seam over it, and the `TEAM_MEMORY.md` document path;
//! - [`TeamMemoryWiring::toolkit`] hands each member its [`MemberMemoryToolkit`], cached per
//!   member, and [`TeamMemoryWiring::tools`] projects that toolkit onto
//!   [`Tool`] handles;
//! - the lifecycle methods forward to the engine with a fresh
//!   [`CallOptions`] (see "Cancellation" below);
//! - [`TeamMemoryWiring::export`] and [`TeamMemoryWiring::import`] move the whole team through
//!   the `TEAM_MEMORY.md` document.
//!
//! The engine is [`openjiuwen_memory::team`]; nothing here re-implements records, extraction,
//! search, or the document format.
//!
//! # Durability
//!
//! This crate does not enable the `openjiuwen-memory` `sqlite` feature, so no wiring can open a
//! SQLite-backed store. A [`MemoryStoreConfig::File`](crate::memory::MemoryStoreConfig::File) configuration therefore keeps the
//! in-memory facade and persists through the document: [`TeamMemoryWiring::export`] writes
//! `TEAM_MEMORY.md`, and the next [`TeamMemoryWiring::open`] imports it. The file store's `dir`
//! selects the document's directory; without one the document lives at
//! `<workspace root>/TEAM_MEMORY.md`.
//!
//! # Disabled
//!
//! When [`MemoryConfig::enabled`] is `false` the wiring touches nothing: no hook runs, no turn is
//! buffered, no session is installed, no toolkit write is accepted from this node, [`TeamMemoryWiring::flush`]
//! answers an empty [`TeamMemoryFlush`], and [`TeamMemoryWiring::export`] answers `Ok(None)`. The facade and
//! the hooks are still constructed, so a disabled wiring is a lawful value with no side effects.
//!
//! # Cancellation
//!
//! A runtime has no per-call memory cancellation token, so every lifecycle pass-through supplies
//! [`CallOptions::new`] (never cancels, never times out). A caller that needs a deadline drives
//! [`TeamMemoryWiring::hooks`] directly; the pass-throughs are the convenience path, not the only one.
//!
//! # Observability
//!
//! A traced configuration ([`MemoryConfig::trace`]) chains the crate's team-memory observer
//! ([`MemorySpanObserver`](crate::observability::MemorySpanObserver)) onto the facade where the
//! facade is built — [`TeamMemory::with_observer`] consumes it, so it cannot be added later. Every
//! observable operation then becomes one `memory.{operation}` span plus one structured `tracing`
//! event at target `openjiuwen_agent_teams::memory`, and on the `otel` build one
//! `SpanRecord` in [`MemoryConfig::otel_sink`]. The
//! facade owns the observer for the wiring's lifetime; [`TeamMemoryWiring::observer`] keeps a second
//! handle to the same object, and `trace: false` or a disabled configuration attaches none.
//!
//! Two caveats belong here because they are visible from this node:
//!
//! - The bridge mints its trace from the team name
//!   ([`TeamSpan::root_for`](crate::observability::TeamSpan::root_for)), so a wiring's `memory.*`
//!   spans are a *sibling* trace of the runtime's `team.{team_name}` trace, not children of it:
//!   the wiring never sees the runtime's root span. Joining the two needs a
//!   `MemorySpanObserver::from_root(TeamSpan)` constructor on the observability node.
//! - Real OTLP export stays deferred: the `opentelemetry` SDK and the exporter live in
//!   `openjiuwen-extensions`, which sits in this crate's dependency layer and cannot be depended on
//!   from here, which is why the `otel` feature only fills an in-memory sink.
//!
//! # Failure
//!
//! A failing hook is logged once with the team and the member and its error is returned
//! unchanged. The engine writes multi-record passes through one batch call, so a failed pass
//! leaves no partial record set; `on_session_start` installs the session only after the engine
//! returned `Ok`, and the toolkit cache is only filled after the context validated.
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::Tool;
use openjiuwen_memory::record::CallOptions;
use openjiuwen_memory::team::{
    LlmTeamExtractor, MemberMemoryContext, MemberMemoryToolkit, RuleTeamExtractor,
    TeamExtractConfig, TeamMemory, TeamMemoryExtractor, TeamMemoryFilter, TeamMemoryFlush,
    TeamMemoryHooks, TeamMemoryObserver, TeamMemoryRecord, TeamTurn, MAX_SCOPE_ID_CHARS,
};
use parking_lot::Mutex;
use tracing::{debug, warn};

use crate::memory::config::{MemoryConfig, MemoryExtractorKind};
use crate::memory::tools::team_memory_tools;
use crate::memory::trace;

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_agent_teams::memory";

/// The team-memory document Python keeps at the workspace root.
pub const TEAM_MEMORY_FILE: &str = "TEAM_MEMORY.md";

/// One team's memory wiring: the facade, its hooks, the document path, and the member toolkits.
///
/// Share it as `Arc<TeamMemoryWiring>`: the type is `Send + Sync`, and every accessor is either
/// immutable or behind a short `parking_lot` lock. One wiring belongs to exactly one team.
pub struct TeamMemoryWiring {
    /// The team whose memory this wiring addresses.
    team_id: String,
    /// The configuration in force.
    config: MemoryConfig,
    /// The engine facade shared by the hooks and every member toolkit.
    memory: Arc<TeamMemory>,
    /// The lifecycle hooks over `memory`.
    hooks: TeamMemoryHooks,
    /// Absolute path of the team's `TEAM_MEMORY.md` document.
    document_path: PathBuf,
    /// One toolkit per member, keyed by member name.
    toolkits: Mutex<BTreeMap<String, Arc<MemberMemoryToolkit>>>,
    /// The observer the facade reports to, when the configuration traces; the facade holds the
    /// same object, and this field is the handle a host can inspect without asking the engine.
    observer: Option<Arc<dyn TeamMemoryObserver>>,
}

impl fmt::Debug for TeamMemoryWiring {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TeamMemoryWiring")
            .field("team_id", &self.team_id)
            .field("config", &self.config)
            .field("document_path", &self.document_path)
            .field("extractor", &self.hooks.extractor_name())
            .field("trace", &self.trace_enabled())
            .field("toolkits", &self.toolkits.lock().len())
            .finish_non_exhaustive()
    }
}

impl TeamMemoryWiring {
    /// Opens the team-memory wiring of `team_id` over an in-memory facade.
    ///
    /// The extractor follows [`MemoryConfig::extractor`], except that the model-backed extractor
    /// needs a model: with no model supplied this builds the deterministic rule extractor and logs
    /// it at debug level. Use [`Self::open_with_model`] to wire the LLM extractor, or
    /// [`Self::with_extractor`] to supply any extractor.
    ///
    /// The existing document is imported when [`MemoryConfig::import_on_build`] is set and the
    /// file exists; an unreadable or malformed document is logged and the store stays empty, so
    /// opening never fails because of a stale file.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the team id breaks the engine's id rules or the configuration
    /// does not validate.
    pub async fn open(
        team_id: impl Into<String>,
        config: MemoryConfig,
        workspace_root: impl Into<PathBuf>,
    ) -> Result<Arc<Self>> {
        let extractor = default_extractor(&config);
        Self::build(team_id.into(), config, workspace_root.into(), extractor).await
    }

    /// Opens the wiring with the model-backed extractor reading `model`.
    ///
    /// [`MemoryExtractorKind`] is ignored on this path: the supplied model decides the extractor.
    /// The engine degrades to the rule extractor on a failing model call, so a model outage still
    /// yields the deterministic memories.
    ///
    /// # Errors
    ///
    /// The same errors as [`Self::open`].
    pub async fn open_with_model(
        team_id: impl Into<String>,
        config: MemoryConfig,
        workspace_root: impl Into<PathBuf>,
        model: Arc<dyn Model>,
    ) -> Result<Arc<Self>> {
        let extractor: Arc<dyn TeamMemoryExtractor> =
            Arc::new(LlmTeamExtractor::new(model, TeamExtractConfig::default()));
        Self::build(team_id.into(), config, workspace_root.into(), extractor).await
    }

    /// Builds the wiring, validating the ids, resolving the document, and importing it.
    async fn build(
        team_id: String,
        config: MemoryConfig,
        workspace_root: PathBuf,
        extractor: Arc<dyn TeamMemoryExtractor>,
    ) -> Result<Arc<Self>> {
        validate_id("team_id", &team_id)?;
        config.validate()?;
        let document_path = config.store.dir().map_or_else(
            || workspace_root.join(TEAM_MEMORY_FILE),
            |dir| dir.join(TEAM_MEMORY_FILE),
        );
        let (memory, observer) = trace::facade(&team_id, &config);
        let memory = Arc::new(memory);
        let hooks = TeamMemoryHooks::new(
            team_id.clone(),
            Arc::clone(&memory),
            extractor,
            config.hooks,
        );
        let wiring = Arc::new(Self {
            team_id,
            config,
            memory,
            hooks,
            document_path,
            toolkits: Mutex::new(BTreeMap::new()),
            observer,
        });
        if wiring.config.enabled && wiring.config.import_on_build {
            if let Err(error) = wiring.import().await {
                warn!(
                    target: LOG_TARGET,
                    team = %wiring.team_id,
                    document = %wiring.document_path.display(),
                    %error,
                    "ignoring an unreadable team memory document; starting from an empty store"
                );
            }
        }
        Ok(wiring)
    }

    /// Returns a new handle over the same facade whose hooks extract with `extractor`.
    ///
    /// The fresh hooks start with no session, no buffered turns, and no members, and the toolkit
    /// cache starts empty; the facade (and therefore every stored record) is shared. Call it right
    /// after [`Self::open`], before the first hook, when the configuration's extractor is not the
    /// one you want.
    #[must_use]
    pub fn with_extractor(&self, extractor: Arc<dyn TeamMemoryExtractor>) -> Arc<Self> {
        let hooks = TeamMemoryHooks::new(
            self.team_id.clone(),
            Arc::clone(&self.memory),
            extractor,
            self.config.hooks,
        );
        Arc::new(Self {
            team_id: self.team_id.clone(),
            config: self.config.clone(),
            memory: Arc::clone(&self.memory),
            hooks,
            document_path: self.document_path.clone(),
            toolkits: Mutex::new(BTreeMap::new()),
            observer: self.observer.clone(),
        })
    }

    /// The team this wiring belongs to.
    #[must_use]
    pub fn team_id(&self) -> &str {
        &self.team_id
    }

    /// The engine facade the hooks and toolkits read and write.
    #[must_use]
    pub fn memory(&self) -> &Arc<TeamMemory> {
        &self.memory
    }

    /// The lifecycle hooks, for a caller that wants its own [`CallOptions`].
    #[must_use]
    pub fn hooks(&self) -> &TeamMemoryHooks {
        &self.hooks
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// The path of the team's `TEAM_MEMORY.md` document.
    #[must_use]
    pub fn document_path(&self) -> &Path {
        &self.document_path
    }

    /// Whether team memory is enabled at all.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Whether this wiring observes its operations, i.e. it is enabled and
    /// [`MemoryConfig::trace`] is on.
    ///
    /// `false` means no `memory.*` span, no `tracing` event and no sink record is produced.
    #[must_use]
    pub fn trace_enabled(&self) -> bool {
        self.observer.is_some()
    }

    /// The observer the facade reports to, when [`Self::trace_enabled`].
    ///
    /// The facade holds the same object
    /// ([`TeamMemory::observer`](openjiuwen_memory::team::TeamMemory::observer)), so this is a
    /// convenience handle rather than a second registration; both are dropped with the wiring.
    #[must_use]
    pub fn observer(&self) -> Option<&Arc<dyn TeamMemoryObserver>> {
        self.observer.as_ref()
    }

    /// The toolkit of `member`, building and caching it on first use.
    ///
    /// `session` is stamped onto every record the toolkit writes and narrows its searches; it is
    /// read when the toolkit is built, so pass the session on the first call for a member. The
    /// cache key is the member name alone, and the same member always gets the same `Arc`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the member id or the session id breaks the engine's id rules.
    pub fn toolkit(&self, member: &str, session: Option<&str>) -> Result<Arc<MemberMemoryToolkit>> {
        if let Some(cached) = self.toolkits.lock().get(member) {
            return Ok(Arc::clone(cached));
        }
        let mut context = MemberMemoryContext::new(self.team_id.as_str()).with_member(member);
        if let Some(session) = session {
            context = context.with_session(session);
        }
        // Validate before caching, so a bad id never leaves a dead toolkit behind.
        context.scope()?;
        let toolkit = Arc::new(MemberMemoryToolkit::new(Arc::clone(&self.memory), context));
        let mut cache = self.toolkits.lock();
        Ok(Arc::clone(
            cache.entry(member.to_owned()).or_insert(toolkit),
        ))
    }

    /// The four model-facing team-memory tools of `member`, in toolkit order.
    ///
    /// # Errors
    ///
    /// The same errors as [`Self::toolkit`].
    pub fn tools(&self, member: &str, session: Option<&str>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(team_memory_tools(self.toolkit(member, session)?))
    }

    /// Binds the coordination session and discards the previous session's buffered turns.
    ///
    /// The session is installed only after the engine accepted it, so a rejected id leaves the
    /// wiring as it was.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unusable session id, and the engine's own errors.
    pub async fn on_session_start(&self, session_id: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        self.hooks
            .on_session_start(session_id, &CallOptions::new())
            .await
            .map_err(|error| self.failed("on_session_start", None, error))
    }

    /// Records that `member` joined the team.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unusable member id, and the engine's own errors.
    pub async fn on_member_join(&self, member: &str) -> Result<Option<TeamMemoryRecord>> {
        if !self.config.enabled {
            return Ok(None);
        }
        self.hooks
            .on_member_join(member, &CallOptions::new())
            .await
            .map_err(|error| self.failed("on_member_join", Some(member), error))
    }

    /// Records that `member` left the team.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unusable member id, and the engine's own errors.
    pub async fn on_member_leave(&self, member: &str) -> Result<Option<TeamMemoryRecord>> {
        if !self.config.enabled {
            return Ok(None);
        }
        self.hooks
            .on_member_leave(member, &CallOptions::new())
            .await
            .map_err(|error| self.failed("on_member_leave", Some(member), error))
    }

    /// Buffers one team turn for the next [`TeamMemoryWiring::flush`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid turn or team id.
    pub async fn on_turn(&self, turn: TeamTurn) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        self.hooks
            .on_turn(turn, &CallOptions::new())
            .await
            .map_err(|error| self.failed("on_turn", None, error))
    }

    /// Buffers one message as a team turn.
    ///
    /// `to` names the addressee; `None` means the message went to the whole team.
    ///
    /// # Errors
    ///
    /// The same errors as [`Self::on_turn`].
    pub async fn on_turn_message(
        &self,
        from: &str,
        to: Option<&str>,
        content: &str,
        at: DateTime<Utc>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let mut turn = TeamTurn::new(from, content, at);
        if let Some(to) = to {
            turn = turn.with_to(to);
        }
        self.on_turn(turn).await
    }

    /// Extracts the buffered turns and writes them as one batch.
    ///
    /// A disabled wiring answers an empty flush without touching the engine.
    ///
    /// # Errors
    ///
    /// The extractor's or the store's own error, with the buffered turns kept for a retry.
    pub async fn flush(&self) -> Result<TeamMemoryFlush> {
        if !self.config.enabled {
            return Ok(TeamMemoryFlush::empty());
        }
        self.hooks
            .on_session_end(&CallOptions::new())
            .await
            .map_err(|error| self.failed("flush", None, error))
    }

    /// Renders the whole team as `TEAM_MEMORY.md`, writes it, and returns the markdown.
    ///
    /// Answers `Ok(None)` without writing anything when the wiring is disabled, when
    /// [`MemoryConfig::export_on_stop`] is `false`, or when the rendered document has no content.
    ///
    /// # Errors
    ///
    /// The store's own error, and [`Error::Memory`] when the document cannot be written.
    pub async fn export(&self) -> Result<Option<String>> {
        if !self.config.enabled || !self.config.export_on_stop {
            return Ok(None);
        }
        let filter = TeamMemoryFilter::for_team(self.team_id.as_str());
        let markdown = self.memory.export_markdown(&filter).await?;
        if markdown.trim().is_empty() {
            return Ok(None);
        }
        self.write_document(&markdown).await?;
        Ok(Some(markdown))
    }

    /// Imports the team's `TEAM_MEMORY.md` and returns how many records were stored.
    ///
    /// An absent document answers `Ok(0)`. Import is explicit caller action, so it runs even when
    /// [`MemoryConfig::import_on_build`] is `false`; only the open-time import is gated on it.
    ///
    /// # Errors
    ///
    /// [`Error::Memory`] when the document cannot be read, [`Error::Validation`] for a malformed
    /// document (nothing is written), and the store's own error.
    pub async fn import(&self) -> Result<usize> {
        let content = match tokio::fs::read_to_string(&self.document_path).await {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => {
                return Err(Error::memory(format!(
                    "reading the team memory document `{}` failed: {error}",
                    self.document_path.display()
                )));
            }
        };
        self.import_from(&content).await
    }

    /// Imports `content` (a `TEAM_MEMORY.md` image) and returns how many records were stored.
    ///
    /// Re-importing the same document is idempotent: identical ids are upserted, so the stored
    /// count stays stable.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for a malformed document (nothing is written), and the store's own
    /// error.
    pub async fn import_from(&self, content: &str) -> Result<usize> {
        let records = self
            .memory
            .import_markdown(content, &CallOptions::new())
            .await?;
        debug!(
            target: LOG_TARGET,
            team = %self.team_id,
            imported = records.len(),
            "imported the team memory document"
        );
        Ok(records.len())
    }

    /// Writes `markdown` to [`Self::document_path`], creating the parent directories.
    async fn write_document(&self, markdown: &str) -> Result<()> {
        if let Some(parent) = self.document_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| {
                    Error::memory(format!(
                        "creating the team memory directory `{}` failed: {error}",
                        parent.display()
                    ))
                })?;
            }
        }
        tokio::fs::write(&self.document_path, markdown.as_bytes())
            .await
            .map_err(|error| {
                Error::memory(format!(
                    "writing the team memory document `{}` failed: {error}",
                    self.document_path.display()
                ))
            })
    }

    /// Logs one failed hook with the team and member, and returns `error` unchanged.
    fn failed(&self, hook: &'static str, member: Option<&str>, error: Error) -> Error {
        warn!(
            target: LOG_TARGET,
            team = %self.team_id,
            member = member.unwrap_or_default(),
            hook,
            %error,
            "team memory hook failed"
        );
        error
    }
}

/// The extractor a configuration selects without a model: the LLM extractor needs one, so it
/// degrades to the deterministic rules.
fn default_extractor(config: &MemoryConfig) -> Arc<dyn TeamMemoryExtractor> {
    if config.extractor == MemoryExtractorKind::Llm {
        debug!(
            target: LOG_TARGET,
            "the llm team memory extractor needs a model; using the rule extractor"
        );
    }
    Arc::new(RuleTeamExtractor::new(TeamExtractConfig::default()))
}

/// Validates a team, member, or session id against the engine's storage-key rules.
fn validate_id(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::validation(format!(
            "team memory `{field}` must not be empty"
        )));
    }
    if value.contains('\0') || value.contains('/') || value.contains('\\') {
        return Err(Error::validation(format!(
            "team memory `{field}` must not contain NUL or a path separator (got {value:?})"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::validation(format!(
            "team memory `{field}` must not contain control characters (got {value:?})"
        )));
    }
    let length = value.chars().count();
    if length > MAX_SCOPE_ID_CHARS {
        return Err(Error::validation(format!(
            "team memory `{field}` must be at most {MAX_SCOPE_ID_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::TimeZone as _;
    use openjiuwen_memory::team::{TeamMemoryKind, TeamMemoryScope};

    use crate::memory::config::MemoryStoreConfig;
    use serde_json::json;

    use super::*;

    /// A fixed timestamp, so nothing in these tests depends on the wall clock.
    fn at(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).unwrap()
    }

    /// Builds a wiring over a temp workspace with rules extraction.
    async fn opened(dir: &tempfile::TempDir, config: MemoryConfig) -> Arc<TeamMemoryWiring> {
        TeamMemoryWiring::open("team-1", config, dir.path())
            .await
            .unwrap()
    }

    /// Lists everything stored for the team.
    async fn stored(wiring: &TeamMemoryWiring) -> Vec<TeamMemoryRecord> {
        wiring
            .memory()
            .list(&TeamMemoryFilter::for_team("team-1"))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn open_builds_an_in_memory_facade_and_the_document_path() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        assert_eq!(wiring.team_id(), "team-1");
        assert!(wiring.is_enabled());
        assert_eq!(wiring.config(), &MemoryConfig::default());
        assert_eq!(wiring.document_path(), dir.path().join(TEAM_MEMORY_FILE));
        assert_eq!(wiring.hooks().extractor_name(), "rule");
        assert_eq!(stored(&wiring).await.len(), 0);
    }

    #[tokio::test]
    async fn a_file_store_moves_the_document_into_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("nested/store");
        let config = MemoryConfig {
            store: MemoryStoreConfig::File {
                dir: Some(store.clone()),
            },
            ..MemoryConfig::default()
        };
        let wiring = opened(&dir, config).await;
        assert_eq!(wiring.document_path(), store.join(TEAM_MEMORY_FILE));
        // The facade stays in memory: the document is the durable image here.
        wiring.on_member_join("dev-1").await.unwrap();
        assert_eq!(stored(&wiring).await.len(), 1);
        assert!(!store.join(TEAM_MEMORY_FILE).exists());
        wiring.export().await.unwrap().unwrap();
        assert!(store.join(TEAM_MEMORY_FILE).exists());
    }

    #[tokio::test]
    async fn open_rejects_unusable_ids_and_configurations() {
        let dir = tempfile::tempdir().unwrap();
        for team in ["", "   ", "a/b", "a\\b", "a\nb"] {
            let error = TeamMemoryWiring::open(team, MemoryConfig::default(), dir.path())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::Validation(_)), "{error}");
        }
        let long = "x".repeat(MAX_SCOPE_ID_CHARS + 1);
        let error = TeamMemoryWiring::open(long, MemoryConfig::default(), dir.path())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("at most"), "{error}");
        let broken = MemoryConfig {
            hooks: openjiuwen_memory::team::TeamMemoryHookConfig {
                max_pending_turns: 0,
                ..Default::default()
            },
            ..MemoryConfig::default()
        };
        let error = TeamMemoryWiring::open("team-1", broken, dir.path())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
    }

    #[tokio::test]
    async fn a_disabled_configuration_touches_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let config = MemoryConfig {
            enabled: false,
            ..MemoryConfig::default()
        };
        let wiring = opened(&dir, config).await;
        assert!(!wiring.is_enabled());
        wiring.on_session_start("s-1").await.unwrap();
        assert!(wiring.on_member_join("dev-1").await.unwrap().is_none());
        assert!(wiring.on_member_leave("dev-1").await.unwrap().is_none());
        wiring
            .on_turn_message("dev-1", None, "We decided to ship it.", at(1_700_000_000))
            .await
            .unwrap();
        assert!(wiring.flush().await.unwrap().is_empty());
        assert!(wiring.export().await.unwrap().is_none());
        assert!(!dir.path().join(TEAM_MEMORY_FILE).exists());
        assert_eq!(stored(&wiring).await.len(), 0);
        assert_eq!(wiring.hooks().pending_turns(), 0);
        assert_eq!(wiring.hooks().session_id(), None);
    }

    /// Builds a configuration the way the runtime does: from `spec.metadata["memory"]`.
    fn config_from_metadata(value: serde_json::Value) -> MemoryConfig {
        let mut spec = crate::schema::TeamAgentSpec {
            team_name: "team-1".to_owned(),
            ..crate::schema::TeamAgentSpec::default()
        };
        spec.metadata = BTreeMap::from([(crate::memory::MEMORY_METADATA_KEY.to_owned(), value)]);
        crate::memory::memory_config(&spec)
    }

    #[tokio::test]
    async fn the_default_configuration_traces_every_operation() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        assert!(wiring.trace_enabled());
        let observer = wiring.observer().expect("an observer");
        let held = wiring.memory().observer().expect("the facade holds it");
        assert!(
            Arc::ptr_eq(observer, held),
            "the wiring and the facade share one observer"
        );
        assert!(format!("{wiring:?}").contains("trace: true"));
    }

    #[tokio::test]
    async fn trace_false_in_the_metadata_attaches_no_observer() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_from_metadata(json!({"trace": false}));
        assert!(!config.trace);
        let wiring = opened(&dir, config).await;
        assert!(!wiring.trace_enabled());
        assert!(wiring.observer().is_none());
        assert!(wiring.memory().observer().is_none());
        assert!(format!("{wiring:?}").contains("trace: false"));
        // Nothing else changes: the wiring still records.
        wiring.on_member_join("dev-1").await.unwrap();
        assert_eq!(stored(&wiring).await.len(), 1);
    }

    #[tokio::test]
    async fn a_disabled_configuration_attaches_no_observer() {
        let dir = tempfile::tempdir().unwrap();
        let config = MemoryConfig {
            enabled: false,
            ..MemoryConfig::default()
        };
        let wiring = opened(&dir, config).await;
        assert!(wiring.config().trace);
        assert!(!wiring.trace_enabled());
        assert!(wiring.observer().is_none());
        assert!(wiring.memory().observer().is_none());
    }

    #[tokio::test]
    async fn with_extractor_keeps_the_observer() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        let rewired = wiring.with_extractor(Arc::new(RuleTeamExtractor::new(
            TeamExtractConfig::default(),
        )));
        let original = wiring.observer().expect("an observer");
        let kept = rewired.observer().expect("an observer");
        assert!(Arc::ptr_eq(original, kept));
        assert!(rewired.trace_enabled());
    }

    #[tokio::test]
    async fn the_session_reaches_the_written_records() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        wiring.on_session_start("s-1").await.unwrap();
        let record = wiring.on_member_join("dev-1").await.unwrap().unwrap();
        assert_eq!(record.session_id.as_deref(), Some("s-1"));
        assert_eq!(record.member_id.as_deref(), Some("dev-1"));
        assert_eq!(record.kind, TeamMemoryKind::MemberProfile);
        let left = wiring.on_member_leave("dev-1").await.unwrap().unwrap();
        assert_eq!(left.session_id.as_deref(), Some("s-1"));
        assert_eq!(left.kind, TeamMemoryKind::HandoffNote);
        assert_eq!(stored(&wiring).await.len(), 2);
    }

    #[tokio::test]
    async fn a_rejected_session_leaves_the_wiring_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        wiring.on_session_start("s-1").await.unwrap();
        let error = wiring.on_session_start("bad/session").await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert_eq!(wiring.hooks().session_id().as_deref(), Some("s-1"));
        let error = wiring.on_member_join("dev/1").await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
    }

    #[tokio::test]
    async fn turns_flush_into_one_batch_with_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        wiring.on_session_start("s-1").await.unwrap();
        wiring
            .on_turn_message(
                "dev-1",
                Some("dev-2"),
                "We decided to split the release train in two.",
                at(1_700_000_000),
            )
            .await
            .unwrap();
        wiring
            .on_turn_message(
                "dev-2",
                None,
                "Next time we should freeze the branch first.",
                at(1_700_000_060),
            )
            .await
            .unwrap();
        let flush = wiring.flush().await.unwrap();
        assert_eq!(flush.turns, 2);
        assert_eq!(flush.len(), 2);
        assert!(flush.degraded.is_none());
        assert!(flush
            .records
            .iter()
            .all(|record| record.session_id.as_deref() == Some("s-1")));
        // The buffer is consumed: a second flush stores nothing.
        assert!(wiring.flush().await.unwrap().is_empty());
        assert_eq!(stored(&wiring).await.len(), 2);
    }

    #[tokio::test]
    async fn an_invalid_turn_is_rejected_without_buffering() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        let error = wiring
            .on_turn_message("dev-1", None, "   ", at(1_700_000_000))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        let error = wiring
            .on_turn_message("dev/1", None, "We decided to ship.", at(1_700_000_000))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert_eq!(wiring.hooks().pending_turns(), 0);
    }

    #[tokio::test]
    async fn the_toolkit_cache_is_per_member() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        let first = wiring.toolkit("dev-1", Some("s-1")).unwrap();
        let again = wiring.toolkit("dev-1", None).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        let other = wiring.toolkit("dev-2", Some("s-1")).unwrap();
        assert!(!Arc::ptr_eq(&first, &other));
        assert_eq!(first.context().member_id.as_deref(), Some("dev-1"));
        assert_eq!(first.context().session_id.as_deref(), Some("s-1"));
        assert_eq!(other.context().member_id.as_deref(), Some("dev-2"));
    }

    #[tokio::test]
    async fn a_bad_member_id_never_enters_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        for member in ["", "dev/1", "dev\n1"] {
            let Err(error) = wiring.toolkit(member, None) else {
                panic!("member {member:?} was accepted");
            };
            assert!(matches!(error, Error::Validation(_)), "{error}");
        }
        assert_eq!(wiring.toolkit("dev-1", None).unwrap().tool_names().len(), 4);
        assert!(!format!("{wiring:?}").contains("dev/1"));
    }

    #[tokio::test]
    async fn a_member_toolkit_writes_records_with_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        let toolkit = wiring.toolkit("dev-1", Some("s-1")).unwrap();
        let stored = toolkit
            .handle_tool_call(
                openjiuwen_memory::team::TOOL_REMEMBER,
                json!({"content": "The build runs on nightly.", "kind": "shared_fact"}),
            )
            .await
            .unwrap();
        assert_eq!(stored["ok"], true);
        let records = wiring
            .memory()
            .list(&TeamMemoryFilter::for_team("team-1"))
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].session_id.as_deref(), Some("s-1"));
        assert_eq!(records[0].member_id.as_deref(), Some("dev-1"));
    }

    #[tokio::test]
    async fn export_writes_the_document_and_a_second_open_imports_it() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        wiring.on_session_start("s-1").await.unwrap();
        wiring.on_member_join("dev-1").await.unwrap();
        wiring
            .on_turn_message(
                "dev-1",
                None,
                "We decided to pin the toolchain for the release.",
                at(1_700_000_000),
            )
            .await
            .unwrap();
        wiring.flush().await.unwrap();
        let markdown = wiring.export().await.unwrap().unwrap();
        assert!(markdown.contains("openjiuwen-team-memory"));
        let path = dir.path().join(TEAM_MEMORY_FILE);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), markdown);

        let reopened = opened(&dir, MemoryConfig::default()).await;
        assert_eq!(stored(&reopened).await.len(), 2);
        // Importing again is idempotent.
        assert_eq!(reopened.import().await.unwrap(), 2);
        assert_eq!(stored(&reopened).await.len(), 2);
    }

    #[tokio::test]
    async fn export_writes_an_empty_document_and_respects_the_switch() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        // A rendered document always carries front matter, so it is written even when empty.
        let markdown = wiring.export().await.unwrap().unwrap();
        assert!(markdown.contains("team_id: team-1"));
        let disabled = MemoryConfig {
            export_on_stop: false,
            ..MemoryConfig::default()
        };
        let wiring = opened(&dir, disabled).await;
        assert!(wiring.export().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_absent_document_imports_zero_records() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        assert_eq!(wiring.import().await.unwrap(), 0);
        assert_eq!(stored(&wiring).await.len(), 0);
    }

    #[tokio::test]
    async fn a_malformed_document_is_reported_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(TEAM_MEMORY_FILE), "not a document").unwrap();
        let error = TeamMemoryWiring::open("team-1", MemoryConfig::default(), dir.path())
            .await
            .err();
        // Opening never fails on a stale document: the store just stays empty.
        assert!(error.is_none());
        let wiring = opened(&dir, MemoryConfig::default()).await;
        assert_eq!(stored(&wiring).await.len(), 0);
        assert!(wiring.import().await.is_err());
        // A second open with the import switched off says nothing at all.
        let quiet = MemoryConfig {
            import_on_build: false,
            ..MemoryConfig::default()
        };
        let wiring = opened(&dir, quiet).await;
        assert_eq!(stored(&wiring).await.len(), 0);
    }

    #[tokio::test]
    async fn import_from_reads_a_held_document() {
        let dir = tempfile::tempdir().unwrap();
        let source = opened(&dir, MemoryConfig::default()).await;
        source.on_member_join("dev-1").await.unwrap();
        let markdown = source.export().await.unwrap().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let quiet = MemoryConfig {
            import_on_build: false,
            ..MemoryConfig::default()
        };
        let target = opened(&target_dir, quiet).await;
        assert_eq!(target.import_from(&markdown).await.unwrap(), 1);
        assert_eq!(stored(&target).await.len(), 1);
    }

    #[tokio::test]
    async fn with_extractor_rebuilds_the_hooks_over_the_same_facade() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        wiring.on_member_join("dev-1").await.unwrap();
        let rewired = wiring.with_extractor(Arc::new(RuleTeamExtractor::new(
            TeamExtractConfig::default(),
        )));
        assert_eq!(rewired.hooks().extractor_name(), "rule");
        assert!(Arc::ptr_eq(wiring.memory(), rewired.memory()));
        assert_eq!(rewired.document_path(), wiring.document_path());
        assert_eq!(stored(&rewired).await.len(), 1);
        // The fresh hooks start with no session and no members.
        assert_eq!(rewired.hooks().session_id(), None);
    }

    #[tokio::test]
    async fn the_llm_extractor_path_builds_over_a_scripted_model() {
        let dir = tempfile::tempdir().unwrap();
        let model: Arc<dyn Model> = Arc::new(openjiuwen_test_support::ScriptedModel::text(
            "{\"memories\": []}",
        ));
        let config = MemoryConfig {
            extractor: MemoryExtractorKind::Llm,
            ..MemoryConfig::default()
        };
        let wiring = TeamMemoryWiring::open_with_model("team-1", config, dir.path(), model)
            .await
            .unwrap();
        assert_eq!(wiring.hooks().extractor_name(), "llm");
        // Without a model the same configuration degrades to the deterministic rules.
        let wiring = opened(
            &dir,
            MemoryConfig {
                extractor: MemoryExtractorKind::Llm,
                ..MemoryConfig::default()
            },
        )
        .await;
        assert_eq!(wiring.hooks().extractor_name(), "rule");
    }

    #[tokio::test]
    async fn the_scope_of_a_written_record_is_the_team() {
        let dir = tempfile::tempdir().unwrap();
        let wiring = opened(&dir, MemoryConfig::default()).await;
        wiring.on_session_start("s-1").await.unwrap();
        wiring.on_member_join("dev-1").await.unwrap();
        let scope = TeamMemoryScope::new("team-1").with_member("dev-1");
        let records = wiring
            .memory()
            .list(&TeamMemoryFilter::for_team("team-1"))
            .await
            .unwrap();
        assert!(records.iter().all(|record| record.team_id == scope.team_id));
    }

    #[test]
    fn the_wiring_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TeamMemoryWiring>();
        assert_send_sync::<Arc<TeamMemoryWiring>>();
    }

    #[cfg(not(feature = "otel"))]
    #[tokio::test]
    async fn a_traced_wiring_stores_no_span_records_without_the_feature() {
        // The same code compiles on every build: the sink slot exists but can never be filled.
        let dir = tempfile::tempdir().unwrap();
        let config = MemoryConfig::default();
        assert!(config.otel_sink.is_none());
        let wiring = opened(&dir, config).await;
        assert!(wiring.trace_enabled());
        assert!(wiring.config().otel_sink.is_none());
        wiring.on_member_join("dev-1").await.unwrap();
        assert_eq!(stored(&wiring).await.len(), 1);
    }

    #[cfg(feature = "otel")]
    mod otel_bridge {
        use super::*;
        use crate::observability::memory::{
            AT_MEMORY_MEMBER_ID, AT_MEMORY_OPERATION, AT_MEMORY_OUTCOME, AT_MEMORY_TEAM_ID,
        };
        use crate::observability::otel::OtelSpanSink;

        /// Opens a wiring over `dir` whose observer also writes into `sink`.
        async fn opened_with_sink(
            dir: &tempfile::TempDir,
            sink: &Arc<OtelSpanSink>,
        ) -> Arc<TeamMemoryWiring> {
            let config = MemoryConfig {
                otel_sink: Some(Arc::clone(sink)),
                ..MemoryConfig::default()
            };
            opened(dir, config).await
        }

        #[tokio::test]
        async fn the_sink_records_one_span_per_operation() {
            let dir = tempfile::tempdir().unwrap();
            let sink = Arc::new(OtelSpanSink::new(8));
            let wiring = opened_with_sink(&dir, &sink).await;
            assert!(wiring.trace_enabled());
            wiring.on_member_join("dev-1").await.unwrap();

            let records = sink.records();
            assert_eq!(records.len(), 1, "{records:?}");
            assert_eq!(records[0].name, "memory.write");
            assert!(
                records[0].parent_span_id.is_some(),
                "the span is a child of the observer's root"
            );
            assert_eq!(
                records[0].attributes.get(AT_MEMORY_TEAM_ID),
                Some(&json!("team-1"))
            );
            assert_eq!(
                records[0].attributes.get(AT_MEMORY_OPERATION),
                Some(&json!("write"))
            );
            assert_eq!(
                records[0].attributes.get(AT_MEMORY_OUTCOME),
                Some(&json!("ok"))
            );
            assert_eq!(
                records[0].attributes.get(AT_MEMORY_MEMBER_ID),
                Some(&json!("dev-1"))
            );
            assert_eq!(sink.evicted(), 0);
        }

        #[tokio::test]
        async fn every_operation_of_the_wiring_lands_in_the_sink() {
            let dir = tempfile::tempdir().unwrap();
            let sink = Arc::new(OtelSpanSink::new(16));
            let wiring = opened_with_sink(&dir, &sink).await;
            wiring.on_member_join("dev-1").await.unwrap();
            wiring.on_member_leave("dev-1").await.unwrap();
            let records = sink.records();
            assert_eq!(records.len(), 2, "{records:?}");
            assert_eq!(records[0].name, "memory.write");
            assert_eq!(records[1].name, "memory.write");
            // One wiring, one trace: every record shares the observer's trace id.
            assert_eq!(records[0].trace_id, records[1].trace_id);
        }

        #[tokio::test]
        async fn a_second_open_of_the_same_team_mints_a_second_trace() {
            let dir = tempfile::tempdir().unwrap();
            let sink = Arc::new(OtelSpanSink::new(16));
            let first = opened_with_sink(&dir, &sink).await;
            let second = opened_with_sink(&dir, &sink).await;
            first.on_member_join("dev-1").await.unwrap();
            second.on_member_join("dev-2").await.unwrap();

            let records = sink.records();
            assert_eq!(records.len(), 2, "{records:?}");
            assert_ne!(
                records[0].trace_id, records[1].trace_id,
                "each wiring mints its own trace; there is no global registry"
            );
            assert_eq!(
                records[1].attributes.get(AT_MEMORY_MEMBER_ID),
                Some(&json!("dev-2"))
            );
        }

        #[tokio::test]
        async fn an_untraced_wiring_never_touches_the_sink() {
            let dir = tempfile::tempdir().unwrap();
            let sink = Arc::new(OtelSpanSink::new(8));
            let config = MemoryConfig {
                trace: false,
                otel_sink: Some(Arc::clone(&sink)),
                ..MemoryConfig::default()
            };
            let wiring = opened(&dir, config).await;
            wiring.on_member_join("dev-1").await.unwrap();
            assert!(sink.is_empty());
            assert_eq!(sink.evicted(), 0);
        }
    }
}
