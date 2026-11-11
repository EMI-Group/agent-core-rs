//! Lite coding memory: typed coding facts and a rendered Markdown memory index.
//!
//! Ports the subset of Python `core/memory/lite/` the harness relies on: coding facts
//! with a fact type, a one-line description, tags, and an optional source path;
//! name-keyed upsert/read/edit/delete/list/search per scope; and a
//! `MEMORY.md`-shaped index with a line cap.
//!
//! # Record mapping
//!
//! A [`CodingFact`] is not a separate persistence concept: it is an ordinary
//! [`MemoryType::Coding`] [`MemoryRecord`](crate::record::MemoryRecord) on the shared
//! [`MemoryEngine`], so facts inherit scoping, dedupe/merge, recall, and every storage
//! backend. The fact-specific fields live in the record metadata (see
//! [`fact`](self)); [`CodingFact::from_record`] identifies coding records by their
//! `coding_fact` marker and metadata `name`.
//!
//! # Deviations from the Python lite stack
//!
//! - There is no Markdown file source of truth and no SQLite FTS5/`sqlite_vec` index:
//!   facts are engine records and keyword recall comes from the engine's BM25 path.
//! - There are no filesystem watchers, no cross-process rebuild lock, and no
//!   embedding cache: [`CodingMemory::render_index`] returns a string and the caller
//!   decides where to write it.
//! - The instance is an explicit `Arc<CodingMemory>` over an injected engine, not a
//!   process-global singleton.
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;

use crate::config::CodingMemoryConfig;
use crate::engine::MemoryEngine;
use crate::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryType, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

mod fact;
mod index;

pub use fact::{CodingFact, CodingFactType};

/// The tracing target of lite-memory diagnostics.
const TARGET: &str = "openjiuwen_memory::lite";

/// The lite coding-memory facade over one [`MemoryEngine`].
///
/// Clone it freely: every clone shares the engine, and all writes are serialized by
/// the engine's per-scope lock. `config` is a snapshot of
/// [`crate::config::MemoryEngineConfig::coding`] taken at construction.
#[derive(Clone)]
pub struct CodingMemory {
    /// The engine every fact is persisted through and recalled from.
    engine: Arc<MemoryEngine>,
    /// The coding-memory settings (index cap, accepted fact types).
    config: CodingMemoryConfig,
}

impl CodingMemory {
    /// Builds the facade over `engine`, using its coding-memory configuration.
    #[must_use]
    pub fn new(engine: Arc<MemoryEngine>) -> Self {
        Self {
            config: engine.config().coding.clone(),
            engine,
        }
    }

    /// The coding-memory configuration in effect.
    #[must_use]
    pub fn config(&self) -> &CodingMemoryConfig {
        &self.config
    }

    /// Writes a fact, updating the same-named fact of the scope when one exists.
    ///
    /// The stored fact type is resolved against
    /// [`CodingMemoryConfig::fact_types`]: the fact's own name is used when it is
    /// configured, otherwise [`CodingMemoryConfig::default_fact_type`] is stored (a
    /// type mismatch never fails a write). An existing fact keeps its id and creation
    /// time; only the content, metadata, and `updated_at` change. A cleared `path` is
    /// written as JSON `null` so the merged metadata really drops the old path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the fact or its scope is invalid (including
    /// blank content, which the engine rejects), and the engine's storage/embedding
    /// error when the write fails. Returns [`Error::NotFound`] for the degenerate case
    /// where the engine skipped the write as a duplicate but the record cannot be read
    /// back, and [`Error::Memory`] when the write produced no record at all.
    pub async fn write(&self, fact: CodingFact, options: &AddOptions) -> Result<MemoryRecord> {
        fact.validate()?;
        let fact_type = self.resolve_fact_type(fact.fact_type);
        let mut metadata = fact.metadata_map(&fact_type);
        if let Some(existing) = self.find_record(&fact.scope, &fact.name).await? {
            if fact.path.is_none() {
                metadata.insert(fact::METADATA_PATH.to_owned(), Value::Null);
            }
            let update = MemoryUpdate {
                content: Some(fact.content.clone()),
                metadata,
                ..MemoryUpdate::default()
            };
            return self
                .engine
                .update(&fact.scope, &existing.id, update, &options.call)
                .await;
        }
        let mut memory = NewMemory::coding(fact.content.clone()).with_timestamp(fact.updated_at);
        for (key, value) in metadata {
            memory = memory.with_metadata(key, value);
        }
        let outcome = self.engine.add(&fact.scope, memory, options).await?;
        if let Some(record) = outcome.added.into_iter().next() {
            return Ok(record);
        }
        if let Some(record) = outcome.merged.into_iter().next() {
            return Ok(record);
        }
        if let Some(id) = outcome.skipped.into_iter().next() {
            return self.engine.get(&fact.scope, &id).await?.ok_or_else(|| {
                Error::not_found(format!(
                    "coding fact `{}` was skipped as a duplicate but no stored record remains",
                    fact.name
                ))
            });
        }
        Err(Error::memory(format!(
            "coding fact `{}` produced no memory record",
            fact.name
        )))
    }

    /// Reads the fact named `name` in `scope`, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the scope cannot be listed.
    pub async fn read(&self, scope: &MemoryScope, name: &str) -> Result<Option<CodingFact>> {
        Ok(self
            .find_record(scope, name)
            .await?
            .and_then(|record| CodingFact::from_record(&record)))
    }

    /// Replaces the body of the fact named `name` and returns the updated fact.
    ///
    /// The fact's name, type, description, tags, and path are preserved; only the
    /// content and `updated_at` change.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when no fact of that name exists, [`Error::Validation`]
    /// when the content is blank or the scope is invalid, and the engine's
    /// storage/embedding error when the update fails.
    pub async fn edit(
        &self,
        scope: &MemoryScope,
        name: &str,
        content: impl Into<String>,
        options: &CallOptions,
    ) -> Result<CodingFact> {
        let existing = self
            .find_record(scope, name)
            .await?
            .ok_or_else(|| Error::not_found(format!("coding fact `{name}` not found in scope")))?;
        let updated = self
            .engine
            .update(scope, &existing.id, MemoryUpdate::content(content), options)
            .await?;
        CodingFact::from_record(&updated).ok_or_else(|| {
            Error::memory(format!(
                "coding fact `{name}` could not be decoded after the update"
            ))
        })
    }

    /// Deletes the fact named `name`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the delete fails.
    pub async fn delete(&self, scope: &MemoryScope, name: &str) -> Result<bool> {
        let Some(existing) = self.find_record(scope, name).await? else {
            return Ok(false);
        };
        self.engine.delete(scope, &existing.id).await
    }

    /// Lists every coding fact of `scope`, newest first and then by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the listing fails.
    pub async fn list(&self, scope: &MemoryScope) -> Result<Vec<CodingFact>> {
        let filter = MemoryFilter::for_scope(scope).kind(MemoryType::Coding);
        let records = self.engine.list(&filter).await?;
        let mut facts: Vec<CodingFact> =
            records.iter().filter_map(CodingFact::from_record).collect();
        facts.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| left.name.cmp(&right.name))
        });
        Ok(facts)
    }

    /// Recalls coding facts of `scope` matching `query`.
    ///
    /// Only [`MemoryType::Coding`] records are searched, regardless of
    /// [`SearchOptions::kinds`]; every other knob (top-k, threshold, mode, token
    /// budget, cancellation) is passed through to the engine.
    ///
    /// # Errors
    ///
    /// Returns the engine's [`MemoryEngine::search`] errors: validation for a blank
    /// query or bad options, [`Error::Unsupported`] for semantic mode without an
    /// embedding model, and the storage/embedding error when a read fails.
    pub async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        let mut options = options.clone();
        options.kinds = vec![MemoryType::Coding];
        self.engine.search(scope, query, &options).await
    }

    /// Renders the `MEMORY.md`-shaped index of `scope`.
    ///
    /// The index is capped at [`CodingMemoryConfig::max_index_lines`] lines including
    /// its title; nothing is written to disk.
    ///
    /// # Errors
    ///
    /// Returns the [`CodingMemory::list`] errors.
    pub async fn render_index(&self, scope: &MemoryScope) -> Result<String> {
        let facts = self.list(scope).await?;
        Ok(index::render_index(&facts, self.config.max_index_lines))
    }

    /// Renders the citation of a recalled fact: `path#L1-L<n>` or `coding:<name>`.
    #[must_use]
    pub fn citation(&self, hit: &RecallHit) -> String {
        index::citation(&hit.record)
    }

    /// Lists the scope's coding records and picks the one whose metadata name is
    /// exactly `name` (case-sensitive).
    ///
    /// Only the public engine API is used; no store is touched directly.
    async fn find_record(&self, scope: &MemoryScope, name: &str) -> Result<Option<MemoryRecord>> {
        let filter = MemoryFilter::for_scope(scope).kind(MemoryType::Coding);
        let records = self.engine.list(&filter).await?;
        Ok(records
            .into_iter()
            .find(|record| CodingFact::from_record(record).is_some_and(|fact| fact.name == name)))
    }

    /// Resolves the fact-type name to store.
    ///
    /// The canonical name is kept when the configured `fact_types` list contains it
    /// (trimmed, ASCII-case-insensitive); otherwise the configured default is used and
    /// a debug event records the fallback.
    fn resolve_fact_type(&self, fact_type: CodingFactType) -> String {
        let canonical = fact_type.as_str();
        let configured = self
            .config
            .fact_types
            .iter()
            .any(|configured| configured.trim().eq_ignore_ascii_case(canonical));
        if configured {
            return canonical.to_owned();
        }
        tracing::debug!(
            target: TARGET,
            fact_type = canonical,
            default = %self.config.default_fact_type,
            "coding fact type is not configured; storing the default fact type"
        );
        self.config.default_fact_type.clone()
    }
}
