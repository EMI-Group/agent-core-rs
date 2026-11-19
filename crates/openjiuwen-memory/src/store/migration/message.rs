//! Message-family migrator: legacy message rows into the current `messages` document collection.
//!
//! Python keeps conversation history in a SQL table (`user_message`) whose rows are
//! `message_id`, `user_id`, `scope_id`, `content`, `session_id`, `role`, and an epoch
//! millisecond `timestamp`; the Rust store keeps the same flat rows in one document
//! collection (`MessageStore`). This migrator is the one-way import
//! between the two: it scans the [`LEGACY_MESSAGE_PREFIXES`] KV namespaces, decodes each
//! row tolerantly (flat rows and the nested backup shape), and upserts the equivalent
//! document through `stored_document` — the exact shape the append path writes — before
//! deleting the migrated legacy key.
//!
//! - **Never lose a row.** A row that cannot be decoded becomes a `MigrationIssue`
//!   (bounded by [`MigrationContext::max_issues`]) and is counted as `failed`; its legacy
//!   key is left in place so the data stays visible and an operator can repair it.
//! - **Idempotent.** Rows already stored identically are counted as `skipped` and their
//!   now-redundant legacy keys are removed; a rerun after a successful import scans
//!   nothing, because every migrated key is gone.
//! - **Deterministic ids.** A row without an id gets the store's own
//!   `derived_message_id`, so re-importing
//!   the same content, role, and timestamp replaces the same document.
//! - **Dry runs write nothing.** Planning and a dry run only read; counts report exactly
//!   what a real run would convert.
//!
//! Timestamps are never invented: a missing or unparseable one is a validation failure
//! naming the row, because a fabricated timestamp would silently reorder history.
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::KvValue;

use crate::store::keys;
use crate::store::messages::{
    derived_message_id, stored_document, DerivedMessageId, StoredMessage,
};

use super::operation::{MigrationOperationHandler, OperationReport};
use super::plan::{MigrationContext, MigrationCounts, MigrationStep, Migrator};

/// Legacy KV prefixes holding serialised message rows (Python's `user_message` table).
pub const LEGACY_MESSAGE_PREFIXES: &[&str] = &["user_message"];

/// Stable identifier of this migration family.
pub const MIGRATOR_ID: &str = "message";

/// The data schema version this migrator brings the stores to.
pub const MIGRATOR_VERSION: u32 = 1;

/// The single import operation.
pub const IMPORT_LEGACY_ROWS: &str = "message.import_legacy_rows";

/// One decoded legacy message row.
///
/// The flat form carries every field at the top level; the nested backup form (a
/// `{"message": {"content", "role"}, "metadata": {...}}` object) is accepted too, and its
/// values win over top-level ones. A blank or absent `message_id` decodes as an empty id,
/// which [`MessageMigrator`] replaces with the deterministic derived id; `user_id` and
/// `scope_id` are mandatory, while `role` may be empty and is preserved verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyMessageRow {
    /// Legacy message id (empty when the row carries none).
    pub message_id: String,
    /// Owning user id.
    pub user_id: String,
    /// Owning scope id.
    pub scope_id: String,
    /// Session the message belongs to, when the row names one.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Author role, preserved verbatim (may be empty).
    #[serde(default)]
    pub role: String,
    /// Plain-text message content.
    pub content: String,
    /// When the message was created.
    pub timestamp: DateTime<Utc>,
}

impl LegacyMessageRow {
    /// Tolerant parse of one stored row; unusable input is `Error::Validation` with a reason.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the text is not JSON, is not a JSON object, lacks a
    /// usable `content`/`timestamp`, or names an empty/mistyped `user_id` or `scope_id`.
    pub fn parse(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("invalid legacy message row JSON: {error}"))
        })?;
        Self::parse_value(&value)
    }

    /// Tolerant parse of one already-decoded row value.
    ///
    /// Nested `message`/`metadata` members win over their top-level counterparts, so both
    /// the flat row and the nested backup shape decode into the same struct.
    ///
    /// # Errors
    ///
    /// Returns a validation error when `value` is not an object or when a required field
    /// is missing, empty, or of an unsupported shape.
    pub fn parse_value(value: &Value) -> Result<Self> {
        let Value::Object(map) = value else {
            return Err(Error::validation(
                "legacy message row must be a JSON object",
            ));
        };
        let message = map.get("message").and_then(Value::as_object);
        let metadata = map.get("metadata").and_then(Value::as_object);

        let content = field(&[message, Some(map)], &["content"])
            .ok_or_else(|| Error::validation("legacy message row is missing its content"))?;
        let content = parse_content(content)?;

        let role = match field(&[message, Some(map)], &["role"]) {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(role)) => role.clone(),
            Some(other) => {
                return Err(Error::validation(format!(
                    "legacy message row role must be a string, got {}",
                    type_name(other)
                )))
            }
        };

        let message_id = match field(&[metadata, Some(map)], &["message_id", "id"]) {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(id)) => id.clone(),
            Some(other) => {
                return Err(Error::validation(format!(
                    "legacy message row message_id must be a string, got {}",
                    type_name(other)
                )))
            }
        };

        let user_id = required_string(&[metadata, Some(map)], &["user_id"], "user_id")?;
        let scope_id = required_string(&[metadata, Some(map)], &["scope_id"], "scope_id")?;

        let session_id = match field(&[metadata, Some(map)], &["session_id"]) {
            None | Some(Value::Null) => None,
            Some(Value::String(session_id)) if session_id.is_empty() => None,
            Some(Value::String(session_id)) => Some(session_id.clone()),
            Some(other) => {
                return Err(Error::validation(format!(
                    "legacy message row session_id must be a string, got {}",
                    type_name(other)
                )))
            }
        };

        let timestamp = field(&[metadata, Some(map)], &["timestamp"])
            .ok_or_else(|| Error::validation("legacy message row is missing its timestamp"))
            .and_then(parse_timestamp)?;

        Ok(Self {
            message_id,
            user_id,
            scope_id,
            session_id,
            role,
            content,
            timestamp,
        })
    }

    /// Projects the row into the current document shape, preserving id/timestamp/role.
    #[must_use]
    pub fn into_stored(self) -> StoredMessage {
        StoredMessage {
            id: self.message_id,
            user_id: self.user_id,
            scope_id: self.scope_id,
            session_id: self.session_id,
            role: self.role,
            content: self.content,
            timestamp: self.timestamp,
        }
    }
}

/// Imports legacy message rows into the current `messages` document collection.
///
/// The migrator holds no state: every store and option arrives through the
/// [`MigrationContext`], and the target collection name comes from the context's prefix
/// layout, so a renamed `messages` family is written exactly like the append path writes
/// it.
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageMigrator;

impl MessageMigrator {
    /// Builds the migrator (it has no options).
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The document collection legacy rows are imported into.
    #[must_use]
    pub fn target_collection(context: &MigrationContext) -> String {
        keys::messages_collection_in(&context.prefixes).to_owned()
    }

    /// Scans, decodes, and classifies every legacy row.
    async fn scan(&self, context: &MigrationContext) -> Result<ScannedMessages> {
        let target = Self::target_collection(context);
        let mut counts = MigrationCounts::default();
        let mut rows = Vec::new();
        let mut issues = Vec::new();
        for prefix in LEGACY_MESSAGE_PREFIXES {
            for (key, value) in context.kv().get_by_prefix(prefix).await? {
                counts.scanned += 1;
                match decode_row(&key, &value) {
                    Ok(row) => {
                        let mut message = row.into_stored();
                        if message.id.is_empty() {
                            // A legacy row carries its own scope; folding it into the id
                            // keeps two scopes' identical rows apart, and the import
                            // writes one row at a time, so the ordinal is always `0`.
                            message.id = derived_message_id(DerivedMessageId {
                                user_id: &message.user_id,
                                scope_id: &message.scope_id,
                                session_id: message.session_id.as_deref(),
                                content: &message.content,
                                role: &message.role,
                                timestamp: message.timestamp,
                                ordinal: 0,
                            });
                        }
                        match context.document().get(&target, &message.id).await? {
                            Some(existing) if existing == stored_document(&message) => {
                                counts.skipped += 1;
                            }
                            _ => counts.affected += 1,
                        }
                        rows.push((key, message));
                    }
                    Err(error) => {
                        counts.failed += 1;
                        issues.push((key, error.to_string()));
                    }
                }
            }
        }
        Ok(ScannedMessages {
            counts,
            rows,
            issues,
        })
    }
}

/// The result of one legacy scan: counters, decodable rows, and row-level failures.
struct ScannedMessages {
    /// Rows seen, converted, skipped, and failed.
    counts: MigrationCounts,
    /// Decoded rows with their legacy KV key, in scan order.
    rows: Vec<(String, StoredMessage)>,
    /// `(legacy key, reason)` for every undecodable row.
    issues: Vec<(String, String)>,
}

#[async_trait]
impl Migrator for MessageMigrator {
    fn id(&self) -> &'static str {
        MIGRATOR_ID
    }

    fn version(&self) -> u32 {
        MIGRATOR_VERSION
    }

    fn description(&self) -> &'static str {
        "Import legacy user_message rows into the messages document collection"
    }

    async fn is_applicable(&self, context: &MigrationContext) -> Result<bool> {
        Ok(!self.plan(context).await?.is_empty())
    }

    async fn plan(&self, context: &MigrationContext) -> Result<Vec<MigrationStep>> {
        let target = Self::target_collection(context);
        let scanned = self.scan(context).await?;
        if scanned.counts.scanned == 0 {
            return Ok(Vec::new());
        }
        let mut step = MigrationStep::new(
            IMPORT_LEGACY_ROWS,
            MIGRATOR_ID,
            MIGRATOR_VERSION,
            "import legacy user_message rows into the messages document collection",
        )
        .with_prefixes(
            LEGACY_MESSAGE_PREFIXES
                .first()
                .map(|prefix| (*prefix).to_owned()),
            Some(target.clone()),
        )
        .with_params(serde_json::json!({
            "sources": LEGACY_MESSAGE_PREFIXES,
            "target": target,
        }))
        .with_counts(scanned.counts);
        for (key, reason) in scanned.issues {
            step.push_issue(key, reason, context.max_issues);
        }
        Ok(vec![step])
    }

    fn operations(&self) -> Vec<Arc<dyn MigrationOperationHandler>> {
        vec![Arc::new(ImportLegacyRows)]
    }
}

/// The [`IMPORT_LEGACY_ROWS`] operation: upsert the migrated documents, delete their keys.
struct ImportLegacyRows;

#[async_trait]
impl MigrationOperationHandler for ImportLegacyRows {
    fn operation_id(&self) -> &'static str {
        IMPORT_LEGACY_ROWS
    }

    async fn execute(
        &self,
        context: &MigrationContext,
        _step: &MigrationStep,
    ) -> Result<OperationReport> {
        let collection = MessageMigrator::target_collection(context);
        let scanned = MessageMigrator.scan(context).await?;
        if !context.is_dry_run() {
            let documents: Vec<_> = scanned
                .rows
                .iter()
                .map(|(_, message)| stored_document(message))
                .collect();
            if !documents.is_empty() {
                context.document().upsert(&collection, documents).await?;
            }
            for (key, _) in &scanned.rows {
                context.kv().delete(key).await?;
            }
        }
        tracing::info!(
            target: "openjiuwen_memory::store",
            collection = %collection,
            affected = scanned.counts.affected,
            skipped = scanned.counts.skipped,
            failed = scanned.counts.failed,
            "imported legacy message rows"
        );
        Ok(OperationReport::succeeded(
            IMPORT_LEGACY_ROWS,
            scanned.counts.affected,
            scanned.counts.skipped,
            scanned.counts.failed,
        ))
    }
}

/// Decodes one stored legacy value, reporting the KV key on failure.
fn decode_row(key: &str, value: &KvValue) -> Result<LegacyMessageRow> {
    let text = match value {
        KvValue::Text(text) => text.clone(),
        KvValue::Bytes(bytes) => String::from_utf8(bytes.clone()).map_err(|_| {
            Error::validation(format!("legacy message row {key:?} is not valid UTF-8"))
        })?,
    };
    LegacyMessageRow::parse(&text)
        .map_err(|error| Error::validation(format!("legacy message row {key:?}: {error}")))
}

/// Looks `keys` up in `layers` first-to-last (nested layers before the top-level map).
fn field<'a>(layers: &[Option<&'a Map<String, Value>>], keys: &[&str]) -> Option<&'a Value> {
    for layer in layers {
        let Some(map) = layer else {
            continue;
        };
        for key in keys {
            if let Some(value) = map.get(*key) {
                return Some(value);
            }
        }
    }
    None
}

/// Reads a mandatory non-empty string field.
fn required_string(
    layers: &[Option<&Map<String, Value>>],
    keys: &[&str],
    name: &str,
) -> Result<String> {
    match field(layers, keys) {
        Some(Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(Error::validation(format!(
            "legacy message row {name} must not be empty"
        ))),
        Some(other) => Err(Error::validation(format!(
            "legacy message row {name} must be a string, got {}",
            type_name(other)
        ))),
        None => Err(Error::validation(format!(
            "legacy message row is missing its {name}"
        ))),
    }
}

/// Decodes the content of a row: a plain string, a `{"text": ...}` object, or a list of
/// string/`{"text": ...}` parts joined with no separator.
fn parse_content(value: &Value) -> Result<String> {
    match value {
        Value::String(content) => Ok(content.clone()),
        Value::Object(map) => match map.get("text") {
            Some(Value::String(text)) => Ok(text.clone()),
            _ => Err(Error::validation(
                "legacy message row content object must carry a string text member",
            )),
        },
        Value::Array(parts) => {
            let mut content = String::new();
            for part in parts {
                match part {
                    Value::String(text) => content.push_str(text),
                    Value::Object(map) => match map.get("text") {
                        Some(Value::String(text)) => content.push_str(text),
                        _ => {
                            return Err(Error::validation(
                                "legacy message row content part must be a string or carry a string text member",
                            ))
                        }
                    },
                    other => {
                        return Err(Error::validation(format!(
                            "legacy message row content part has an unsupported shape: {}",
                            type_name(other)
                        )))
                    }
                }
            }
            Ok(content)
        }
        other => Err(Error::validation(format!(
            "legacy message row content has an unsupported shape: {}",
            type_name(other)
        ))),
    }
}

/// Decodes an RFC 3339 timestamp or epoch milliseconds (number or numeric string).
fn parse_timestamp(value: &Value) -> Result<DateTime<Utc>> {
    match value {
        Value::Number(number) => {
            let millis = number.as_i64().ok_or_else(|| {
                Error::validation(
                    "legacy message row timestamp number must be an integer number of milliseconds",
                )
            })?;
            millis_to_datetime(millis)
        }
        Value::String(text) => {
            if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
                return Ok(parsed.with_timezone(&Utc));
            }
            let millis: i64 = text.trim().parse().map_err(|_| {
                Error::validation(format!(
                    "legacy message row timestamp {text:?} is neither RFC 3339 nor epoch milliseconds"
                ))
            })?;
            millis_to_datetime(millis)
        }
        other => Err(Error::validation(format!(
            "legacy message row timestamp must be RFC 3339 text or epoch milliseconds, got {}",
            type_name(other)
        ))),
    }
}

/// Converts epoch milliseconds into a UTC timestamp.
fn millis_to_datetime(millis: i64) -> Result<DateTime<Utc>> {
    DateTime::from_timestamp_millis(millis).ok_or_else(|| {
        Error::validation(format!(
            "legacy message row timestamp {millis} milliseconds is out of range"
        ))
    })
}

/// A short noun phrase naming a JSON value's kind, for error messages.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)] // `assert!(x.is_empty())` reads better in these tests
mod tests {
    use super::super::operation::{MigrationJournal, MigrationRun, OperationOutcome};
    use super::super::plan::MigrationPlan;
    use super::*;
    use crate::store::{MemoryStores, MessageStore};
    use openjiuwen_foundation::store::Query;

    /// An in-memory store bundle.
    fn stores() -> MemoryStores {
        MemoryStores::in_memory()
    }

    /// A real-run context over `stores`.
    fn context(stores: &MemoryStores) -> MigrationContext {
        MigrationContext::new(stores.clone())
    }

    /// Seeds one legacy row under `user_message/{key}`.
    async fn seed(stores: &MemoryStores, key: &str, document: &str) {
        stores
            .kv
            .set(&format!("user_message/{key}"), KvValue::text(document))
            .await
            .expect("seed legacy row");
    }

    /// Every document of the `messages` collection.
    async fn documents(stores: &MemoryStores) -> Vec<crate::store::StoredMessage> {
        stores
            .document
            .query(keys::MESSAGES_COLLECTION, &Query::default())
            .await
            .expect("query messages")
            .into_iter()
            .map(|document| {
                serde_json::from_value(document.to_value()).expect("stored message decodes")
            })
            .collect()
    }

    /// The legacy keys still present.
    async fn legacy_keys(stores: &MemoryStores) -> Vec<String> {
        stores
            .kv
            .get_by_prefix(LEGACY_MESSAGE_PREFIXES[0])
            .await
            .expect("scan legacy keys")
            .into_keys()
            .collect()
    }

    /// Applies `migrator` over a fresh plan.
    async fn apply(migrator: &MessageMigrator, context: &MigrationContext) -> MigrationRun {
        let plan = MigrationPlan::new(migrator.plan(context).await.expect("plan"));
        let mut journal = MigrationJournal::new();
        migrator
            .apply(context, &plan, &mut journal)
            .await
            .expect("apply")
    }

    fn millis(value: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(value).expect("valid millis")
    }

    #[tokio::test]
    async fn flat_rows_are_imported_with_ids_roles_and_timestamps_preserved() {
        let stores = stores();
        seed(
            &stores,
            "a",
            r#"{"message_id":"m1","user_id":"alice","scope_id":"ws","content":"hello","role":"user","timestamp":1700000000000}"#,
        )
        .await;
        seed(
            &stores,
            "b",
            r#"{"id":"m2","user_id":"alice","scope_id":"ws","session_id":"s1","content":"hi","role":"assistant","timestamp":"2026-01-02T03:04:05Z"}"#,
        )
        .await;
        seed(
            &stores,
            "c",
            r#"{"message_id":"m3","user_id":"bob","scope_id":"ws","content":[{"text":"part one "},"part two"],"role":"tool","timestamp":"1700000000001"}"#,
        )
        .await;

        let context = context(&stores);
        let migrator = MessageMigrator::new();
        let plan = MigrationPlan::new(migrator.plan(&context).await.expect("plan"));
        assert_eq!(plan.len(), 1);
        let step = &plan.steps[0];
        assert_eq!(step.operation_id, IMPORT_LEGACY_ROWS);
        assert_eq!(step.migrator, MIGRATOR_ID);
        assert_eq!(step.version, MIGRATOR_VERSION);
        assert_eq!(step.source_prefix.as_deref(), Some("user_message"));
        assert_eq!(
            step.target_prefix.as_deref(),
            Some(keys::MESSAGES_COLLECTION)
        );
        assert_eq!(step.counts.scanned, 3);
        assert_eq!(step.counts.affected, 3);
        assert_eq!(step.counts.skipped, 0);
        assert_eq!(step.counts.failed, 0);
        assert_eq!(step.params["sources"], serde_json::json!(["user_message"]));
        assert_eq!(step.params["target"], keys::MESSAGES_COLLECTION);

        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports.len(), 1);
        assert_eq!(run.reports[0].affected, 3);

        let rows = documents(&stores).await;
        assert_eq!(rows.len(), 3);
        let m1 = rows.iter().find(|row| row.id == "m1").expect("m1");
        assert_eq!(m1.user_id, "alice");
        assert_eq!(m1.scope_id, "ws");
        assert_eq!(m1.role, "user");
        assert_eq!(m1.content, "hello");
        assert_eq!(m1.timestamp, millis(1_700_000_000_000));
        assert_eq!(m1.session_id, None);
        let m2 = rows.iter().find(|row| row.id == "m2").expect("m2");
        assert_eq!(m2.session_id.as_deref(), Some("s1"));
        assert_eq!(
            m2.timestamp,
            DateTime::parse_from_rfc3339("2026-01-02T03:04:05Z")
                .expect("fixed")
                .with_timezone(&Utc)
        );
        let m3 = rows.iter().find(|row| row.id == "m3").expect("m3");
        assert_eq!(m3.content, "part one part two");
        assert_eq!(m3.timestamp, millis(1_700_000_000_001));

        assert!(legacy_keys(&stores).await.is_empty());
    }

    #[tokio::test]
    async fn nested_backup_rows_decode_and_missing_ids_are_derived() {
        let stores = stores();
        seed(
            &stores,
            "nested",
            r#"{"message":{"content":"nested","role":"assistant"},"metadata":{"id":"m9","user_id":"alice","scope_id":"ws","session_id":"s1","timestamp":"1700000000000"}}"#,
        )
        .await;
        seed(
            &stores,
            "anon",
            r#"{"user_id":"alice","scope_id":"ws","content":"anonymous","role":"user","timestamp":1700000000000}"#,
        )
        .await;

        let context = context(&stores);
        let migrator = MessageMigrator::new();
        let run = apply(&migrator, &context).await;
        assert_eq!(run.reports[0].affected, 2);

        let rows = documents(&stores).await;
        assert_eq!(rows.len(), 2);
        let nested = rows.iter().find(|row| row.id == "m9").expect("nested id");
        assert_eq!(nested.content, "nested");
        assert_eq!(nested.role, "assistant");
        assert_eq!(nested.session_id.as_deref(), Some("s1"));
        assert_eq!(nested.user_id, "alice");
        let expected = derived_message_id(DerivedMessageId {
            user_id: "alice",
            scope_id: "ws",
            session_id: None,
            content: "anonymous",
            role: "user",
            timestamp: millis(1_700_000_000_000),
            ordinal: 0,
        });
        assert!(rows.iter().any(|row| row.id == expected));
        assert!(legacy_keys(&stores).await.is_empty());

        // Re-seeding the id-less row is idempotent: the deterministic id already exists.
        seed(
            &stores,
            "anon",
            r#"{"user_id":"alice","scope_id":"ws","content":"anonymous","role":"user","timestamp":1700000000000}"#,
        )
        .await;
        let plan = MigrationPlan::new(migrator.plan(&context).await.expect("plan"));
        assert_eq!(plan.steps[0].counts.scanned, 1);
        assert_eq!(plan.steps[0].counts.skipped, 1);
        assert_eq!(plan.steps[0].counts.affected, 0);
        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports[0].affected, 0);
        assert_eq!(run.reports[0].skipped, 1);
        assert_eq!(documents(&stores).await.len(), 2);
        assert!(legacy_keys(&stores).await.is_empty());
    }

    #[tokio::test]
    async fn malformed_rows_are_reported_and_left_in_place() {
        let stores = stores();
        seed(&stores, "json", "not json").await;
        seed(
            &stores,
            "user",
            r#"{"message_id":"x","scope_id":"ws","content":"c","timestamp":1700000000000}"#,
        )
        .await;
        seed(
            &stores,
            "time",
            r#"{"message_id":"x","user_id":"a","scope_id":"ws","content":"c","timestamp":"yesterday"}"#,
        )
        .await;
        seed(
            &stores,
            "content",
            r#"{"message_id":"x","user_id":"a","scope_id":"ws","content":42,"timestamp":1700000000000}"#,
        )
        .await;
        stores
            .kv
            .set("user_message/bytes", KvValue::bytes(vec![0xff, 0xfe, 0x00]))
            .await
            .expect("seed bytes");
        seed(
            &stores,
            "good",
            r#"{"message_id":"good","user_id":"a","scope_id":"ws","content":"fine","role":"user","timestamp":1700000000000}"#,
        )
        .await;

        let context = context(&stores);
        let migrator = MessageMigrator::new();
        let plan = MigrationPlan::new(migrator.plan(&context).await.expect("plan"));
        let step = &plan.steps[0];
        assert_eq!(step.counts.scanned, 6);
        assert_eq!(step.counts.failed, 5);
        assert_eq!(step.counts.affected, 1);
        assert_eq!(step.issues.len(), 5);
        for issue in &step.issues {
            assert!(issue.key.starts_with("user_message/"), "{issue:?}");
            assert!(!issue.reason.is_empty(), "{issue:?}");
        }

        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports[0].affected, 1);
        assert_eq!(run.reports[0].failed, 5);
        assert_eq!(documents(&stores).await.len(), 1);
        let remaining = legacy_keys(&stores).await;
        assert_eq!(remaining.len(), 5);
        assert!(!remaining.iter().any(|key| key.ends_with("good")));
    }

    #[tokio::test]
    async fn dry_runs_report_the_plan_and_write_nothing() {
        let stores = stores();
        seed(
            &stores,
            "a",
            r#"{"message_id":"m1","user_id":"alice","scope_id":"ws","content":"hello","role":"user","timestamp":1700000000000}"#,
        )
        .await;
        seed(
            &stores,
            "b",
            r#"{"message_id":"m2","user_id":"alice","scope_id":"ws","content":"bye","role":"user","timestamp":1700000000001}"#,
        )
        .await;

        let real = context(&stores);
        let expected = MessageMigrator::new().plan(&real).await.expect("plan");
        let dry = context(&stores).with_dry_run(true);
        let migrator = MessageMigrator::new();
        let plan = MigrationPlan::new(migrator.plan(&dry).await.expect("plan"));
        assert_eq!(plan.counts(), expected[0].counts);

        let mut journal = MigrationJournal::new();
        let run = migrator
            .apply(&dry, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(run.reports[0].outcome, OperationOutcome::Skipped);
        assert_eq!(documents(&stores).await.len(), 0);
        assert_eq!(legacy_keys(&stores).await.len(), 2);

        // The handler itself also refuses to write when invoked on a dry-run context.
        let report = ImportLegacyRows
            .execute(&dry, &plan.steps[0])
            .await
            .expect("handler");
        assert_eq!(report.affected, 2);
        assert_eq!(documents(&stores).await.len(), 0);
        assert_eq!(legacy_keys(&stores).await.len(), 2);
    }

    #[tokio::test]
    async fn a_rerun_after_a_successful_import_plans_nothing() {
        let stores = stores();
        seed(
            &stores,
            "a",
            r#"{"message_id":"m1","user_id":"alice","scope_id":"ws","content":"hello","role":"user","timestamp":1700000000000}"#,
        )
        .await;
        let context = context(&stores);
        let migrator = MessageMigrator::new();
        apply(&migrator, &context).await;
        assert!(migrator.plan(&context).await.expect("plan").is_empty());
        assert!(!migrator.is_applicable(&context).await.expect("applicable"));
    }

    #[tokio::test]
    async fn rows_already_in_the_target_are_skipped_and_their_keys_removed() {
        let stores = stores();
        let timestamp = millis(1_700_000_000_000);
        // Write the target document through the ordinary append path first.
        let message_store = MessageStore::new(stores.document.clone());
        message_store
            .append(
                &crate::record::MemoryScope::new("alice", "ws"),
                &[openjiuwen_foundation::prompt::BaseMessage::user("hello")],
                timestamp,
            )
            .await
            .expect("append");
        let id = derived_message_id(DerivedMessageId {
            user_id: "alice",
            scope_id: "ws",
            session_id: None,
            content: "hello",
            role: "user",
            timestamp,
            ordinal: 0,
        });
        let row = format!(
            r#"{{"message_id":"{id}","user_id":"alice","scope_id":"ws","content":"hello","role":"user","timestamp":1700000000000}}"#
        );
        seed(&stores, "dup", &row).await;

        let context = context(&stores);
        let migrator = MessageMigrator::new();
        let plan = MigrationPlan::new(migrator.plan(&context).await.expect("plan"));
        assert_eq!(plan.steps[0].counts.scanned, 1);
        assert_eq!(plan.steps[0].counts.skipped, 1);
        assert_eq!(plan.steps[0].counts.affected, 0);
        let mut journal = MigrationJournal::new();
        migrator
            .apply(&context, &plan, &mut journal)
            .await
            .expect("apply");
        assert_eq!(documents(&stores).await.len(), 1);
        assert!(legacy_keys(&stores).await.is_empty());
    }

    #[test]
    fn parse_accepts_every_supported_shape() {
        let flat = LegacyMessageRow::parse(
            r#"{"message_id":"m","user_id":"u","scope_id":"s","content":"c","role":"assistant","timestamp":1700000000000}"#,
        )
        .expect("flat");
        assert_eq!(flat.message_id, "m");
        assert_eq!(flat.role, "assistant");
        assert_eq!(flat.session_id, None);
        assert_eq!(flat.timestamp, millis(1_700_000_000_000));

        let text_content = LegacyMessageRow::parse(
            r#"{"id":"i","user_id":"u","scope_id":"s","session_id":"sess","content":{"text":"object"},"timestamp":"2026-01-02T03:04:05Z"}"#,
        )
        .expect("object content");
        assert_eq!(text_content.message_id, "i");
        assert_eq!(text_content.content, "object");
        assert_eq!(text_content.session_id.as_deref(), Some("sess"));

        let parts = LegacyMessageRow::parse_value(&serde_json::json!({
            "user_id": "u",
            "scope_id": "s",
            "content": ["a", {"text": "b"}, "c"],
            "timestamp": "1700000000001",
        }))
        .expect("parts content");
        assert_eq!(parts.content, "abc");
        assert_eq!(parts.message_id, "");
        assert_eq!(parts.role, "");

        let nested = LegacyMessageRow::parse(
            r#"{"message":{"content":"nested","role":"tool"},"metadata":{"message_id":"m","user_id":"u","scope_id":"s","timestamp":1700000000000}}"#,
        )
        .expect("nested");
        assert_eq!(nested.content, "nested");
        assert_eq!(nested.role, "tool");

        let empty_session = LegacyMessageRow::parse(
            r#"{"message_id":"m","user_id":"u","scope_id":"s","session_id":"","content":"c","timestamp":1700000000000}"#,
        )
        .expect("empty session");
        assert_eq!(empty_session.session_id, None);
    }

    #[test]
    fn parse_rejects_every_unusable_shape() {
        let cases = [
            ("not json", "invalid legacy message row JSON"),
            ("[]", "must be a JSON object"),
            (
                r#"{"scope_id":"s","content":"c","timestamp":1}"#,
                "missing its user_id",
            ),
            (
                r#"{"user_id":"u","content":"c","timestamp":1}"#,
                "missing its scope_id",
            ),
            (
                r#"{"user_id":"","scope_id":"s","content":"c","timestamp":1}"#,
                "user_id must not be empty",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","timestamp":1}"#,
                "missing its content",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","content":1.5,"timestamp":1}"#,
                "content has an unsupported shape",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","content":"c"}"#,
                "missing its timestamp",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","content":"c","timestamp":"soon"}"#,
                "neither RFC 3339 nor epoch milliseconds",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","content":"c","timestamp":true}"#,
                "must be RFC 3339 text or epoch milliseconds",
            ),
            (
                r#"{"user_id":7,"scope_id":"s","content":"c","timestamp":1}"#,
                "user_id must be a string",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","content":"c","timestamp":1,"role":4}"#,
                "role must be a string",
            ),
            (
                r#"{"user_id":"u","scope_id":"s","content":"c","timestamp":1,"session_id":4}"#,
                "session_id must be a string",
            ),
        ];
        for (text, reason) in cases {
            let error = LegacyMessageRow::parse(text).expect_err(text);
            assert!(
                error.to_string().contains(reason),
                "{text}: {error} (expected {reason})"
            );
        }
    }
}
