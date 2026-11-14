//! Variables and per-scope configuration.
//!
//! Variables are plain text KV entries owned by a user scope, or by a session when the
//! scope carries a `session_id`; scope configuration is JSON stored under
//! `memory_scope_config/{scope_id}`. Both live in [`crate::store::MemoryStorage`], so
//! the facade only validates the ids and forwards the call.
//!
//! Every variable write additionally maintains the [`MemoryType::Variable`] record
//! projection owned by [`MemoryEngine::write_variable`](crate::engine::MemoryEngine::write_variable),
//! so a variable is readable through the Python-shaped KV API *and* listable as a
//! record. Rewriting one name updates the same projection in place; the deterministic
//! record id means the text-similarity dedupe scan is never involved.
//!
//! [`MemoryType::Variable`]: crate::record::MemoryType::Variable

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::analyzer::ExtractedFact;
use crate::config::MemoryScopeConfig;
use crate::engine::VariableWrite;
use crate::record::{AddOptions, MemoryRecord, MemoryScope, NewMemory, DEFAULT_ID};

use super::LongTermMemory;

/// The tag the analyzer puts on a variable-assignment fact.
const VARIABLE_TAG: &str = "variable";
/// The analyzer metadata key holding the variable name.
const VARIABLE_NAME_FIELD: &str = "variable";
/// The analyzer metadata key holding the variable value.
const VARIABLE_VALUE_FIELD: &str = "value";
/// The crate-wide metadata key for free-form tags (the `integrations` convention).
const TAGS_KEY: &str = "tags";

/// What one [`LongTermMemory::remember_variables`] call did.
///
/// `extracted` is the number of facts considered, `variables` the number written as
/// [`MemoryType::Variable`](crate::record::MemoryType::Variable) projections, and the
/// `records_*` counters aggregate the [`AddOutcome`](crate::record::AddOutcome) of every
/// remaining fact.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VariableAssignmentReport {
    /// Number of facts the caller supplied.
    pub extracted: usize,
    /// Number of variable-tagged facts written as variable records.
    pub variables: usize,
    /// Number of remaining facts stored as new records.
    pub records_added: usize,
    /// Number of remaining facts folded into (or replacing) an existing record.
    pub records_merged: usize,
    /// Number of remaining facts dropped by the dedupe policy.
    pub records_skipped: usize,
}

/// Validates a variable name: non-empty after trimming, without `/` or NUL.
pub(super) fn validate_variable_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(Error::validation(format!(
            "memory variable `name` must not be empty (got \"{name}\")"
        )));
    }
    if name.contains('/') {
        return Err(Error::validation(format!(
            "memory variable `name` must not contain '/' (got \"{name}\")"
        )));
    }
    if name.contains('\0') {
        return Err(Error::validation(format!(
            "memory variable `name` must not contain NUL (got \"{name}\")"
        )));
    }
    Ok(())
}

/// Validates a bare `scope_id` with the [`MemoryScope`] rules (non-empty, no `/`, no
/// backslash, no NUL, at most 128 characters).
pub(super) fn validate_scope_id(scope_id: &str) -> Result<()> {
    MemoryScope::new(DEFAULT_ID, scope_id).validate()
}

impl LongTermMemory {
    /// Stores `value` as the text variable `name` of `scope`.
    ///
    /// A scope with a `session_id` writes a session variable; otherwise the variable is
    /// user-scoped. The write keeps the Python-shaped KV entry and maintains the
    /// [`MemoryType::Variable`](crate::record::MemoryType::Variable) record projection:
    /// rewriting one name updates that same record in place, so a scope never
    /// accumulates one record per value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the scope or the variable name is invalid or
    /// the name is listed in
    /// [`forbidden_variables`](crate::config::MemoryEngineConfig::forbidden_variables)
    /// (all before any write) and the storage error when the write fails.
    pub async fn set_variable(&self, scope: &MemoryScope, name: &str, value: &str) -> Result<()> {
        self.assign_variable(scope, name, value, &AddOptions::default())
            .await?;
        Ok(())
    }

    /// Stores `name = value` of `scope`, returning the projection id and whether it was
    /// created.
    ///
    /// This is [`LongTermMemory::set_variable`] with explicit [`AddOptions`], so the
    /// call can be cancelled, bounded by a timeout, or skip the embedding of the
    /// projection record.
    ///
    /// # Errors
    ///
    /// A validation error from the scope, the name, or
    /// [`forbidden_variables`](crate::config::MemoryEngineConfig::forbidden_variables)
    /// (all before any I/O), [`Error::Cancelled`]/[`Error::Timeout`] when `options`
    /// fires, and the storage/embedding error when a write fails.
    pub async fn assign_variable(
        &self,
        scope: &MemoryScope,
        name: &str,
        value: &str,
        options: &AddOptions,
    ) -> Result<VariableWrite> {
        scope.validate()?;
        validate_variable_name(name)?;
        self.engine
            .write_variable(scope, name, value, options)
            .await
    }

    /// Lists the [`MemoryType::Variable`](crate::record::MemoryType::Variable) records
    /// of `scope`, ascending by `(timestamp, id)`.
    ///
    /// These are the record projections of the variables written through
    /// [`LongTermMemory::set_variable`]/[`LongTermMemory::assign_variable`], not the KV
    /// entries [`LongTermMemory::get_variables`] reads. A scope without a `session_id`
    /// selects the variables of every session of `(user_id, scope_id)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when the
    /// listing fails.
    pub async fn variable_records(&self, scope: &MemoryScope) -> Result<Vec<MemoryRecord>> {
        scope.validate()?;
        self.engine.variable_records(scope).await
    }

    /// Writes analyzer `facts` under `scope`, turning variable assignments into variable
    /// records.
    ///
    /// A fact is a variable assignment when it carries the `variable` tag, the `variable`
    /// metadata key (the name), and the `value` metadata key — the shape
    /// [`RuleExtractor`](crate::analyzer::RuleExtractor) emits. Such a fact becomes one
    /// [`MemoryType::Variable`](crate::record::MemoryType::Variable) projection (never a
    /// `SemanticMemory` record) and never enters the text-similarity dedupe scan; every
    /// other fact is stored as a record exactly like
    /// [`LongTermMemory::remember`] does.
    ///
    /// [`LongTermMemory::remember`] itself still stores a variable assignment as a
    /// `SemanticMemory` record: this entry point is the opt-in to the variable write
    /// path, so existing callers keep their behaviour.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope, for a variable name the write
    /// path rejects (a forbidden name is named in the message), or for a fact the
    /// analyzer/engine rejects; [`Error::Cancelled`] when `options.call` fires (earlier
    /// writes stay stored); and the storage/embedding error when a write fails. A failed
    /// write is never dropped silently: the error names the fact index and content.
    pub async fn remember_variables(
        &self,
        scope: &MemoryScope,
        facts: &[ExtractedFact],
        options: &AddOptions,
    ) -> Result<VariableAssignmentReport> {
        scope.validate()?;
        options.call.check()?;
        let mut report = VariableAssignmentReport {
            extracted: facts.len(),
            ..VariableAssignmentReport::default()
        };
        for (index, fact) in facts.iter().enumerate() {
            options.call.check()?;
            let Some((name, value)) = variable_assignment(fact) else {
                let outcome = self
                    .engine
                    .add(scope, record_input(fact), options)
                    .await
                    .map_err(|error| {
                        error.with_context(format!("writing fact {index} (`{}`)", fact.content))
                    })?;
                report.records_added += outcome.added.len();
                report.records_merged += outcome.merged.len();
                report.records_skipped += outcome.skipped.len();
                continue;
            };
            match self.assign_variable(scope, &name, &value, options).await {
                Ok(_) => report.variables += 1,
                // A forbidden or otherwise unacceptable *name* fails the call: the
                // caller asked for a variable write, so a silent drop would lose data.
                Err(error) => {
                    return Err(
                        error.with_context(format!("writing variable fact {index} (`{name}`)"))
                    );
                }
            }
        }
        Ok(report)
    }

    /// Reads variables of `scope`: every stored variable when `names` is `None`, or the
    /// named ones that exist.
    ///
    /// This reads the KV entries, so it keeps the Python-shaped semantics regardless of
    /// the record projection.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or variable name and the
    /// storage error when a read fails.
    pub async fn get_variables(
        &self,
        scope: &MemoryScope,
        names: Option<&[String]>,
    ) -> Result<BTreeMap<String, String>> {
        scope.validate()?;
        let storage = self.engine.storage();
        let session_id = scope.session_id.as_deref();
        let Some(names) = names else {
            return storage.list_variables(scope, session_id).await;
        };
        for name in names {
            validate_variable_name(name)?;
        }
        let mut variables = BTreeMap::new();
        for name in names {
            if let Some(value) = storage.get_variable(scope, session_id, name).await? {
                variables.insert(name.clone(), value);
            }
        }
        Ok(variables)
    }

    /// Deletes `names` from `scope`, returning how many variables existed.
    ///
    /// Every name is validated before anything is deleted, so an invalid name cannot
    /// leave a partial delete behind. An empty `names` slice deletes nothing and
    /// returns `0`. Both representations are removed: the KV entry and the
    /// [`MemoryType::Variable`](crate::record::MemoryType::Variable) projection (with
    /// its vector entry).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or variable name and the
    /// storage error when a delete fails.
    pub async fn delete_variables(&self, scope: &MemoryScope, names: &[String]) -> Result<usize> {
        scope.validate()?;
        for name in names {
            validate_variable_name(name)?;
        }
        let mut removed = 0;
        for name in names {
            if self.engine.delete_variable_everywhere(scope, name).await? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Stores the configuration of `scope_id`, returning whether it replaced one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and the storage error
    /// when the write fails.
    pub async fn set_scope_config(
        &self,
        scope_id: &str,
        config: MemoryScopeConfig,
    ) -> Result<bool> {
        validate_scope_id(scope_id)?;
        self.engine
            .storage()
            .set_scope_config(scope_id, &config)
            .await
    }

    /// Reads the configuration of `scope_id`, or `None` when it is absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` or malformed stored JSON
    /// and the storage error when the read fails.
    pub async fn get_scope_config(&self, scope_id: &str) -> Result<Option<MemoryScopeConfig>> {
        validate_scope_id(scope_id)?;
        self.engine.storage().get_scope_config(scope_id).await
    }

    /// Deletes the configuration of `scope_id`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and the storage error
    /// when the delete fails.
    pub async fn delete_scope_config(&self, scope_id: &str) -> Result<bool> {
        validate_scope_id(scope_id)?;
        self.engine.storage().delete_scope_config(scope_id).await
    }
}

/// Returns the `(name, value)` a variable-assignment fact carries, if it is one.
///
/// The shape is the one the rule extractor emits: the `variable` tag plus the
/// `variable`/`value` metadata keys. A fact with a non-scalar value or without the name
/// is not a variable assignment and stays a record.
fn variable_assignment(fact: &ExtractedFact) -> Option<(String, String)> {
    if !fact.tags.iter().any(|tag| tag == VARIABLE_TAG) {
        return None;
    }
    let name = fact.metadata.get(VARIABLE_NAME_FIELD)?.as_str()?;
    let value = scalar_text(fact.metadata.get(VARIABLE_VALUE_FIELD)?)?;
    Some((name.to_owned(), value))
}

/// Renders a scalar JSON value as variable text (strings, numbers, booleans).
fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Converts an extracted fact into an engine write input, keeping its tags.
///
/// [`ExtractedFact::to_new_memory`] already carries content, kind, importance, source,
/// and metadata. Tags are not a [`NewMemory`] field, so they go under the crate's `tags`
/// metadata key (the convention `crate::facade::integrations` and `lite` use) unless the
/// fact supplies that key itself.
fn record_input(fact: &ExtractedFact) -> NewMemory {
    let mut memory = fact.to_new_memory();
    if !fact.tags.is_empty() && !memory.metadata.contains_key(TAGS_KEY) {
        let tags = fact.tags.iter().cloned().map(Value::String);
        memory
            .metadata
            .insert(TAGS_KEY.to_owned(), Value::Array(tags.collect()));
    }
    memory
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio_util::sync::CancellationToken;

    use crate::analyzer::ExtractionInput;
    use crate::config::MemoryEngineConfig;
    use crate::record::MemoryType;

    async fn facade() -> std::sync::Arc<LongTermMemory> {
        facade_with(MemoryEngineConfig::default()).await
    }

    async fn facade_with(config: MemoryEngineConfig) -> std::sync::Arc<LongTermMemory> {
        LongTermMemory::in_memory(config).await.expect("facade")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    /// A config listing `name` as forbidden.
    fn forbidding(name: &str) -> MemoryEngineConfig {
        MemoryEngineConfig {
            forbidden_variables: vec![name.to_owned()],
            ..Default::default()
        }
    }

    /// The variable-assignment fact shape the rule extractor emits.
    fn variable_fact(name: &str, value: &str) -> ExtractedFact {
        let mut fact = ExtractedFact {
            content: format!("{name}={value}"),
            kind: MemoryType::SemanticMemory,
            tags: vec![VARIABLE_TAG.to_owned()],
            ..ExtractedFact::default()
        };
        fact.metadata.insert(
            VARIABLE_NAME_FIELD.to_owned(),
            Value::String(name.to_owned()),
        );
        fact.metadata.insert(
            VARIABLE_VALUE_FIELD.to_owned(),
            Value::String(value.to_owned()),
        );
        fact
    }

    #[test]
    fn variable_names_are_validated() {
        assert!(validate_variable_name("theme").is_ok());
        for name in ["", "   ", "a/b", "nul\0byte"] {
            let error = validate_variable_name(name).expect_err("invalid name");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
    }

    #[test]
    fn scope_ids_are_validated() {
        assert!(validate_scope_id("ws").is_ok());
        for scope_id in ["", "a/b", &"x".repeat(129)] {
            let error = validate_scope_id(scope_id).expect_err("invalid scope id");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
        assert!(validate_scope_id(&"x".repeat(128)).is_ok());
    }

    #[tokio::test]
    async fn variables_round_trip_through_the_facade() {
        let ltm = facade().await;
        ltm.set_variable(&scope(), "theme", "dark")
            .await
            .expect("set");
        ltm.set_variable(&scope(), "city", "Berlin")
            .await
            .expect("set");
        let all = ltm.get_variables(&scope(), None).await.expect("list");
        assert_eq!(all.len(), 2);
        assert_eq!(all.get("theme").map(String::as_str), Some("dark"));

        let named = ltm
            .get_variables(&scope(), Some(&["theme".to_owned(), "missing".to_owned()]))
            .await
            .expect("named");
        assert_eq!(named.len(), 1);
        assert_eq!(named.get("theme").map(String::as_str), Some("dark"));

        assert_eq!(
            ltm.delete_variables(&scope(), &["theme".to_owned()])
                .await
                .expect("delete"),
            1
        );
        assert_eq!(
            ltm.delete_variables(&scope(), &["theme".to_owned()])
                .await
                .expect("delete again"),
            0
        );
        assert_eq!(
            ltm.delete_variables(&scope(), &[])
                .await
                .expect("delete none"),
            0
        );
        assert_eq!(
            ltm.get_variables(&scope(), None).await.expect("list").len(),
            1
        );
    }

    #[tokio::test]
    async fn session_variables_are_separate_from_user_variables() {
        let ltm = facade().await;
        ltm.set_variable(&scope(), "theme", "dark")
            .await
            .expect("user variable");
        let session = scope().session("s1");
        ltm.set_variable(&session, "theme", "light")
            .await
            .expect("session variable");
        assert_eq!(
            ltm.get_variables(&scope(), None)
                .await
                .expect("user")
                .get("theme")
                .map(String::as_str),
            Some("dark")
        );
        assert_eq!(
            ltm.get_variables(&session, None)
                .await
                .expect("session")
                .get("theme")
                .map(String::as_str),
            Some("light")
        );
    }

    #[tokio::test]
    async fn invalid_variables_and_scopes_are_rejected() {
        let ltm = facade().await;
        let error = ltm
            .set_variable(&scope(), "a/b", "value")
            .await
            .expect_err("invalid name");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = ltm
            .set_variable(&MemoryScope::new("alice", "bad/scope"), "theme", "dark")
            .await
            .expect_err("invalid scope");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn variable_reads_and_deletes_validate_names_before_io() {
        let ltm = facade().await;
        let scope = scope();
        let error = ltm
            .get_variables(&scope, Some(&["a/b".to_owned()]))
            .await
            .expect_err("invalid name");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        ltm.set_variable(&scope, "ok", "value")
            .await
            .expect("set valid variable");
        let error = ltm
            .delete_variables(&scope, &["ok".to_owned(), "a/b".to_owned()])
            .await
            .expect_err("invalid name");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert_eq!(
            ltm.get_variables(&scope, None)
                .await
                .expect("variables")
                .get("ok")
                .map(String::as_str),
            Some("value"),
            "an invalid name must not cause a partial delete"
        );
    }

    #[tokio::test]
    async fn scope_config_round_trips_and_is_deleted() {
        let ltm = facade().await;
        let config = MemoryScopeConfig {
            user_profile_definition: Some("identity facts".to_owned()),
            semantic_memory_definition: Some("durable facts".to_owned()),
            episodic_memory_definition: None,
        };
        assert!(!ltm
            .set_scope_config("ws", config.clone())
            .await
            .expect("create"));
        assert_eq!(
            ltm.get_scope_config("ws").await.expect("get"),
            Some(config.clone())
        );
        assert!(ltm
            .set_scope_config("ws", config.clone())
            .await
            .expect("replace"));
        assert_eq!(ltm.get_scope_config("other").await.expect("absent"), None);
        assert!(ltm.delete_scope_config("ws").await.expect("delete"));
        assert!(!ltm.delete_scope_config("ws").await.expect("delete again"));
        assert_eq!(ltm.get_scope_config("ws").await.expect("absent"), None);
        let error = ltm
            .get_scope_config("bad/scope")
            .await
            .expect_err("invalid scope id");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_variable_write_keeps_the_entry_and_one_projection_record() {
        let ltm = facade().await;
        let scope = scope();
        ltm.set_variable(&scope, "theme", "dark")
            .await
            .expect("set");
        let records = ltm.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, MemoryType::Variable);
        assert_eq!(records[0].content, "theme=dark");
        assert_eq!(
            records[0].metadata.get("value").and_then(Value::as_str),
            Some("dark")
        );
        let created = records[0].timestamp;
        // The KV entry keeps the legacy Python semantics.
        assert_eq!(
            ltm.get_variables(&scope, None)
                .await
                .expect("kv")
                .get("theme")
                .map(String::as_str),
            Some("dark")
        );

        ltm.set_variable(&scope, "theme", "light")
            .await
            .expect("rewrite");
        let records = ltm.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1, "the same name updates in place");
        assert_eq!(records[0].content, "theme=light");
        assert_eq!(
            records[0].metadata.get("value").and_then(Value::as_str),
            Some("light")
        );
        assert_eq!(records[0].timestamp, created, "creation time is preserved");
        assert_eq!(
            ltm.get_variables(&scope, None)
                .await
                .expect("kv")
                .get("theme")
                .map(String::as_str),
            Some("light")
        );
    }

    #[tokio::test]
    async fn a_forbidden_variable_is_rejected_before_any_write() {
        let ltm = facade_with(forbidding("api_key")).await;
        let scope = scope();
        let error = ltm
            .set_variable(&scope, "API_KEY", "secret")
            .await
            .expect_err("forbidden");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let message = format!("{error:?}");
        assert!(message.contains("API_KEY"), "{message}");
        assert!(message.contains("forbidden"), "{message}");
        assert!(ltm
            .get_variables(&scope, None)
            .await
            .expect("kv")
            .is_empty());
        assert_eq!(
            ltm.variable_records(&scope).await.expect("records").len(),
            0
        );
    }

    #[tokio::test]
    async fn deleting_a_variable_removes_the_entry_and_the_projection() {
        let ltm = facade().await;
        let scope = scope();
        ltm.set_variable(&scope, "theme", "dark")
            .await
            .expect("set theme");
        ltm.set_variable(&scope, "city", "Berlin")
            .await
            .expect("set city");
        assert_eq!(
            ltm.variable_records(&scope).await.expect("records").len(),
            2
        );
        assert_eq!(
            ltm.delete_variables(&scope, &["theme".to_owned()])
                .await
                .expect("delete"),
            1
        );
        assert!(!ltm
            .get_variables(&scope, None)
            .await
            .expect("kv")
            .contains_key("theme"));
        let records = ltm.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "city=Berlin");
        assert_eq!(
            ltm.delete_variables(&scope, &["missing".to_owned()])
                .await
                .expect("unknown name"),
            0
        );
        assert_eq!(
            ltm.delete_variables(&scope, &[]).await.expect("no names"),
            0
        );
    }

    #[tokio::test]
    async fn assign_variable_reports_creation_and_honours_cancellation() {
        let ltm = facade().await;
        let scope = scope();
        let first = ltm
            .assign_variable(&scope, "theme", "dark", &AddOptions::default())
            .await
            .expect("assign");
        assert!(first.created);
        let second = ltm
            .assign_variable(&scope, "theme", "light", &AddOptions::default())
            .await
            .expect("assign again");
        assert!(!second.created);
        assert_eq!(first.record_id, second.record_id);

        let token = CancellationToken::new();
        token.cancel();
        let cancelled = AddOptions::default().with_cancellation(token);
        let error = ltm
            .assign_variable(&scope, "city", "Berlin", &cancelled)
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert!(!ltm
            .get_variables(&scope, None)
            .await
            .expect("kv")
            .contains_key("city"));
        assert_eq!(
            ltm.variable_records(&scope).await.expect("records").len(),
            1
        );
    }

    #[tokio::test]
    async fn remember_variables_projects_assignments_and_keeps_other_facts() {
        let ltm = facade().await;
        let scope = scope();
        let facts = vec![
            variable_fact("theme", "dark"),
            ExtractedFact {
                content: "Alice prefers dark mode".to_owned(),
                kind: MemoryType::UserProfile,
                tags: vec!["preference".to_owned()],
                ..ExtractedFact::default()
            },
        ];
        let report = ltm
            .remember_variables(&scope, &facts, &AddOptions::default())
            .await
            .expect("remember variables");
        assert_eq!(
            (
                report.extracted,
                report.variables,
                report.records_added,
                report.records_merged,
                report.records_skipped,
            ),
            (2, 1, 1, 0, 0),
            "{report:?}"
        );
        let records = ltm.variable_records(&scope).await.expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "theme=dark");
        assert_eq!(
            ltm.get_variables(&scope, None)
                .await
                .expect("kv")
                .get("theme")
                .map(String::as_str),
            Some("dark")
        );
        // The variable fact never becomes a semantic record.
        assert_eq!(
            ltm.list(&scope, &[MemoryType::SemanticMemory])
                .await
                .expect("semantic")
                .len(),
            0
        );
        let profiles = ltm
            .list(&scope, &[MemoryType::UserProfile])
            .await
            .expect("profiles");
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0]
            .metadata
            .get(TAGS_KEY)
            .and_then(Value::as_array)
            .is_some_and(|tags| !tags.is_empty()));
    }

    #[tokio::test]
    async fn a_forbidden_variable_fact_fails_the_call() {
        let ltm = facade_with(forbidding("api_key")).await;
        let scope = scope();
        let error = ltm
            .remember_variables(
                &scope,
                &[variable_fact("api_key", "secret")],
                &AddOptions::default(),
            )
            .await
            .expect_err("forbidden");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let message = format!("{error:?}");
        assert!(message.contains("api_key"), "{message}");
        assert_eq!(
            ltm.variable_records(&scope).await.expect("records").len(),
            0
        );
        assert!(ltm
            .get_variables(&scope, None)
            .await
            .expect("kv")
            .is_empty());
    }

    #[tokio::test]
    async fn remember_still_stores_an_assignment_as_a_semantic_record() {
        let ltm = facade().await;
        let scope = scope();
        let input = ExtractionInput::from_text("theme = dark");
        let report = ltm
            .remember(&scope, &input, &AddOptions::default())
            .await
            .expect("remember");
        assert_eq!((report.extracted, report.added), (1, 1), "{report:?}");
        assert_eq!(
            ltm.list(&scope, &[MemoryType::SemanticMemory])
                .await
                .expect("semantic")
                .len(),
            1
        );
        // Opting into `remember_variables` is what creates the variable record.
        assert_eq!(
            ltm.variable_records(&scope).await.expect("records").len(),
            0
        );
        assert!(ltm
            .get_variables(&scope, None)
            .await
            .expect("kv")
            .is_empty());
    }
}
