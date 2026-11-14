//! Backup and restore of the memory key-value keyspace (records, variables, scope config).
//!
//! A [`MemoryBackup`] is a self-describing JSON document holding every entry of the four
//! key-value memory families — [`KvFamily::Records`], [`KvFamily::UserVariables`],
//! [`KvFamily::SessionVariables`], and [`KvFamily::ScopeConfig`] — with each value
//! preserved verbatim ([`KvValue::Text`] stays text, [`KvValue::Bytes`] stays bytes).
//!
//! Python snapshots every registered KV prefix before a migration
//! (`MEMORY_MIGRATION_KV_SCHEMA_VERSION_BACKUP_{ms}`) and restores the blob on failure.
//! This module keeps the same data shape as an explicit, caller-driven document instead:
//! [`dump`]/[`dump_with`] read, [`restore`]/[`restore_with`] write, and nothing runs
//! automatically.
//!
//! # Key normalisation
//!
//! [`dump_with`] scans the current prefix of each family *and* every alias of the
//! registry, then rewrites an alias key to the current-layout key
//! (`keys::current_equivalent_in`). The scan visits the current prefix first and never
//! overwrites an entry it already holds, so when a legacy key and its current-layout
//! counterpart both exist the **current value wins** and exactly one entry is produced.
//! A restore therefore always writes the current layout, and a dump/restore round trip
//! loses no value across a family rename.
//!
//! # Restore semantics
//!
//! [`restore_with`] maps each entry key through the *target* registry: a family that the
//! target renamed keeps the target's current prefix, an entry whose family the target
//! does not know (and which the backup's own recorded prefixes do not explain) is written
//! verbatim. Restoring is additive: entries whose target already holds an equal value are
//! counted `skipped`, every other entry is written, and target keys the backup does not
//! contain are never deleted. A dry run performs every read and no write, so it reports
//! exactly what a real restore would do.
//!
//! # Out of scope
//!
//! Conversation messages live in the document store, not in the KV keyspace, and
//! `scope_user_mapping` is rebuilt by the migration pipeline; neither family is dumped or
//! restored (and neither is counted in [`BackupCounts`]). Vector collections, graph data,
//! and document collections are not part of a backup.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::KvValue;
use serde::{Deserialize, Serialize};

use super::keys;
use super::prefix::{KvFamily, KvPrefixRegistry};
use super::MemoryStores;

/// Marker in the JSON document so a file cannot be restored by the wrong subsystem.
pub const BACKUP_FORMAT: &str = "openjiuwen.memory.backup";

/// Format version of the backup document.
pub const BACKUP_FORMAT_VERSION: u32 = 1;

/// The families a backup covers, in [`KvFamily::ALL`] order.
///
/// Only `StorageArea::Kv` families are dumped; [`KvFamily::Messages`] lives in the
/// document store and [`KvFamily::ScopeUserMapping`] is rebuilt by the migration
/// pipeline.
const DUMPED_FAMILIES: [KvFamily; 4] = [
    KvFamily::Records,
    KvFamily::UserVariables,
    KvFamily::SessionVariables,
    KvFamily::ScopeConfig,
];

/// Per-family entry counts.
///
/// [`BackupCounts::total`] counts only the entries that belong to one of the four dumped
/// families, so it can be smaller than the number of dumped entries when a backup also
/// carries keys of another family (for example a hand-written one).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackupCounts {
    /// Entries under the `records` family.
    #[serde(default)]
    pub records: usize,
    /// Entries under the `user_variables` family.
    #[serde(default)]
    pub user_variables: usize,
    /// Entries under the `session_variables` family.
    #[serde(default)]
    pub session_variables: usize,
    /// Entries under the `scope_config` family.
    #[serde(default)]
    pub scope_configs: usize,
    /// Every classified entry (the sum of the four fields above).
    #[serde(default)]
    pub total: usize,
}

impl BackupCounts {
    /// Classifies `entries` by [`KvPrefixRegistry::family_of`] under `prefixes`.
    ///
    /// Keys of the `messages` and `scope_user_mapping` families, and keys of unknown
    /// families, are counted in no field (and not in [`BackupCounts::total`]).
    #[must_use]
    pub fn from_entries(entries: &BTreeMap<String, KvValue>, prefixes: &KvPrefixRegistry) -> Self {
        let mut counts = Self::default();
        for key in entries.keys() {
            counts.add(prefixes.family_of(key));
        }
        counts
    }

    /// Adds `other` into `self`, field by field.
    pub fn merge(&mut self, other: &Self) {
        self.records += other.records;
        self.user_variables += other.user_variables;
        self.session_variables += other.session_variables;
        self.scope_configs += other.scope_configs;
        self.total += other.total;
    }

    /// Counts one entry of `family`; families outside the four dumped ones are ignored.
    fn add(&mut self, family: Option<KvFamily>) {
        match family {
            Some(KvFamily::Records) => self.records += 1,
            Some(KvFamily::UserVariables) => self.user_variables += 1,
            Some(KvFamily::SessionVariables) => self.session_variables += 1,
            Some(KvFamily::ScopeConfig) => self.scope_configs += 1,
            Some(KvFamily::ScopeUserMapping | KvFamily::Messages) | None => return,
        }
        self.total += 1;
    }
}

/// A full dump of the memory key-value keyspace.
///
/// `entries` is a [`BTreeMap`], so the JSON encoding is stable and diffable; only
/// `created_at` depends on the wall clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryBackup {
    /// Format marker, always [`BACKUP_FORMAT`] for a document this module produced.
    #[serde(default = "default_format")]
    pub format: String,
    /// Format version, [`BACKUP_FORMAT_VERSION`] for a document this module produced.
    #[serde(default = "default_version")]
    pub version: u32,
    /// When the dump was taken; the only non-deterministic member.
    pub created_at: DateTime<Utc>,
    /// Family label -> the current prefix in force when the dump was taken.
    #[serde(default)]
    pub prefixes: BTreeMap<String, String>,
    /// Every dumped entry, keyed by its **current-layout** key, value preserved verbatim.
    #[serde(default)]
    pub entries: BTreeMap<String, KvValue>,
    /// Per-family entry counts of `entries`.
    #[serde(default)]
    pub counts: BackupCounts,
}

impl MemoryBackup {
    /// Builds a backup document and computes its [`BackupCounts`] from `entries`.
    ///
    /// `prefixes` records the family layout the entries were normalised to; it is what
    /// [`MemoryBackup::verify`] classifies the entries with, so a backup stays
    /// self-consistent even when the restoring process uses another registry.
    #[must_use]
    pub fn new(
        prefixes: BTreeMap<String, String>,
        entries: BTreeMap<String, KvValue>,
        created_at: DateTime<Utc>,
    ) -> Self {
        let mut backup = Self {
            format: BACKUP_FORMAT.to_owned(),
            version: BACKUP_FORMAT_VERSION,
            created_at,
            prefixes,
            entries,
            counts: BackupCounts::default(),
        };
        backup.counts = backup.recompute_from_recorded();
        backup
    }

    /// Number of dumped entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dump holds no entry at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The stored per-family counts (recomputed by [`MemoryBackup::new`]).
    #[must_use]
    pub fn counts(&self) -> BackupCounts {
        self.counts
    }

    /// Recomputes [`BackupCounts`] for `entries`, classifying keys with `prefixes`.
    ///
    /// A key whose family `prefixes` does not know is classified with the backup's own
    /// recorded prefixes instead, so a dump taken under one layout can still be counted
    /// correctly after the layout was renamed.
    #[must_use]
    pub fn recompute_counts(&self, prefixes: &KvPrefixRegistry) -> BackupCounts {
        let mut counts = BackupCounts::default();
        for key in self.entries.keys() {
            counts.add(
                prefixes
                    .family_of(key)
                    .or_else(|| recorded_family(&self.prefixes, key)),
            );
        }
        counts
    }

    /// Checks the format marker, the format version, and the declared counts.
    ///
    /// The counts are validated against the backup's own recorded `prefixes`, never
    /// against a restoring registry: the two may legitimately differ.
    ///
    /// # Errors
    ///
    /// `Error::Validation` when `format` is not [`BACKUP_FORMAT`] (the message names the
    /// found value) or when the declared counts disagree with the entries;
    /// `Error::Unsupported` when `version` is newer than [`BACKUP_FORMAT_VERSION`],
    /// because this binary cannot restore a newer backup format.
    pub fn verify(&self) -> Result<()> {
        if self.format != BACKUP_FORMAT {
            return Err(Error::validation(format!(
                "memory backup has the format `{}`, expected `{BACKUP_FORMAT}`",
                self.format
            )));
        }
        if self.version > BACKUP_FORMAT_VERSION {
            return Err(Error::unsupported(format!(
                "memory backup uses format version {} but this binary supports format \
                 version {BACKUP_FORMAT_VERSION} at most; a newer backup format cannot be \
                 restored",
                self.version
            )));
        }
        let recomputed = self.recompute_from_recorded();
        if self.counts != recomputed {
            return Err(Error::validation(format!(
                "memory backup count mismatch: declared {} entries ({:?}) but the entries \
                 hold {} ({:?})",
                self.counts.total, self.counts, recomputed.total, recomputed
            )));
        }
        Ok(())
    }

    /// Serializes the document as pretty JSON, with stable entry ordering.
    ///
    /// # Errors
    ///
    /// Returns the `serde_json` error annotated with the serialization context.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| Error::external(error).with_context("serializing the memory backup"))
    }

    /// Decodes a document produced by [`MemoryBackup::to_json`] and verifies it.
    ///
    /// Unknown members are ignored and `format`/`version`/`prefixes`/`entries`/`counts`
    /// fall back to their defaults when absent, so a minimal document decodes to an empty
    /// backup.
    ///
    /// # Errors
    ///
    /// `Error::Validation` for malformed JSON, plus every [`MemoryBackup::verify`] error.
    pub fn from_json(text: &str) -> Result<Self> {
        let backup: Self = serde_json::from_str(text)
            .map_err(|error| Error::validation(format!("invalid memory backup JSON: {error}")))?;
        backup.verify()?;
        Ok(backup)
    }

    /// [`BackupCounts`] derived from the recorded `prefixes` map alone.
    fn recompute_from_recorded(&self) -> BackupCounts {
        let mut counts = BackupCounts::default();
        for key in self.entries.keys() {
            counts.add(recorded_family(&self.prefixes, key));
        }
        counts
    }
}

/// What a restore did (or, for a dry run, what it would do).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreReport {
    /// Whether the run only reported; a dry run writes nothing.
    pub dry_run: bool,
    /// Entries written (or, for a dry run, that would be written).
    #[serde(default)]
    pub written: usize,
    /// Entries skipped because the target already held an equal value.
    #[serde(default)]
    pub skipped: usize,
    /// Entries considered, that is [`MemoryBackup::len`] of the restored backup.
    #[serde(default)]
    pub total: usize,
    /// Per-family counts of the entries written (would be written).
    #[serde(default)]
    pub counts: BackupCounts,
}

impl RestoreReport {
    /// Number of entries written plus skipped, equal to [`RestoreReport::total`].
    #[must_use]
    pub fn processed(&self) -> usize {
        self.written + self.skipped
    }
}

/// Dumps the four key-value memory families of `stores` using the default prefix registry.
///
/// # Errors
///
/// Propagates any key-value store read error, annotated with the scanned prefix.
pub async fn dump(stores: &MemoryStores) -> Result<MemoryBackup> {
    dump_with(stores, &KvPrefixRegistry::default()).await
}

/// Dumps the four key-value memory families of `stores` under `prefixes`.
///
/// Every alias prefix is scanned as well as the current one, and each alias key is
/// rewritten to its current-layout key before it is stored; the current layout wins when
/// both layouts hold the same data.
///
/// # Errors
///
/// Propagates any key-value store read error, annotated with the scanned prefix.
pub async fn dump_with(stores: &MemoryStores, prefixes: &KvPrefixRegistry) -> Result<MemoryBackup> {
    let mut entries: BTreeMap<String, KvValue> = BTreeMap::new();
    for family in DUMPED_FAMILIES {
        collect_family(stores, prefixes, family, &mut entries).await?;
    }
    let backup = MemoryBackup::new(recorded_prefixes(prefixes), entries, Utc::now());
    tracing::info!(
        target: "openjiuwen_memory::store",
        entries = backup.len(),
        records = backup.counts.records,
        user_variables = backup.counts.user_variables,
        session_variables = backup.counts.session_variables,
        scope_configs = backup.counts.scope_configs,
        "dumped the memory key-value keyspace"
    );
    Ok(backup)
}

/// Restores `backup` into `stores`, writing every entry.
///
/// Uses the default prefix registry; see [`restore_with`] for an explicit registry and a
/// dry run.
///
/// # Errors
///
/// Every [`MemoryBackup::verify`] error, plus any key-value store read or write error,
/// annotated with the affected key.
pub async fn restore(stores: &MemoryStores, backup: &MemoryBackup) -> Result<RestoreReport> {
    restore_with(stores, backup, &KvPrefixRegistry::default(), false).await
}

/// Restores `backup` into `stores` under `prefixes`, optionally as a dry run.
///
/// Each entry key is mapped onto the target layout (an entry of a family `prefixes`
/// renamed keeps the new current prefix; an entry the backup's recorded prefixes cannot
/// explain is written verbatim), then read: an existing equal value counts `skipped`,
/// anything else is written (or, for `dry_run`, only counted as `written`). Target keys
/// absent from the backup are left untouched.
///
/// # Errors
///
/// Every [`MemoryBackup::verify`] error, plus any key-value store read or write error,
/// annotated with the affected key.
pub async fn restore_with(
    stores: &MemoryStores,
    backup: &MemoryBackup,
    prefixes: &KvPrefixRegistry,
    dry_run: bool,
) -> Result<RestoreReport> {
    backup.verify()?;
    let mut report = RestoreReport {
        dry_run,
        written: 0,
        skipped: 0,
        total: backup.len(),
        counts: BackupCounts::default(),
    };
    for (key, value) in &backup.entries {
        let target = target_key(backup, prefixes, key);
        let existing = stores.kv.get(&target).await.map_err(|error| {
            error.with_context(format!("reading the memory restore target `{target}`"))
        })?;
        if existing.as_ref() == Some(value) {
            report.skipped += 1;
            continue;
        }
        if !dry_run {
            stores
                .kv
                .set(&target, value.clone())
                .await
                .map_err(|error| {
                    error.with_context(format!("writing the memory restore target `{target}`"))
                })?;
        }
        report.written += 1;
        report.counts.add(prefixes.family_of(&target));
    }
    tracing::info!(
        target: "openjiuwen_memory::store",
        dry_run = report.dry_run,
        written = report.written,
        skipped = report.skipped,
        total = report.total,
        "restored the memory key-value keyspace"
    );
    Ok(report)
}

/// Reads every entry under the current prefix of `family` and then under each alias.
async fn collect_family(
    stores: &MemoryStores,
    prefixes: &KvPrefixRegistry,
    family: KvFamily,
    entries: &mut BTreeMap<String, KvValue>,
) -> Result<()> {
    for prefix in prefixes.prefixes(family) {
        let scan = scan_prefix(prefix);
        let found =
            stores.kv.get_by_prefix(&scan).await.map_err(|error| {
                error.with_context(format!("scanning the memory prefix `{scan}`"))
            })?;
        for (stored, value) in found {
            let rewritten = keys::current_equivalent_in(prefixes, &stored);
            if let Some(current) = &rewritten {
                tracing::debug!(
                    target: "openjiuwen_memory::store",
                    family = family.label(),
                    key = %stored,
                    current = %current,
                    "normalising a legacy memory key for the backup"
                );
            }
            let target = rewritten.unwrap_or(stored);
            // The current layout is scanned first, so an existing entry always wins.
            entries.entry(target).or_insert(value);
        }
    }
    Ok(())
}

/// The scan prefix of one family prefix: a key is always `<prefix>/<rest>`.
fn scan_prefix(prefix: &str) -> String {
    format!("{prefix}/")
}

/// The recorded family layout: one entry per [`KvFamily::ALL`] member.
fn recorded_prefixes(prefixes: &KvPrefixRegistry) -> BTreeMap<String, String> {
    KvFamily::ALL
        .into_iter()
        .map(|family| {
            (
                family.label().to_owned(),
                prefixes.current(family).to_owned(),
            )
        })
        .collect()
}

/// The family whose current prefix recorded in `recorded` matches the key's first segment.
fn recorded_family(recorded: &BTreeMap<String, String>, key: &str) -> Option<KvFamily> {
    let head = key.split_once('/').map_or(key, |(head, _)| head);
    if head.is_empty() {
        return None;
    }
    recorded
        .iter()
        .find(|(_, prefix)| prefix.as_str() == head)
        .and_then(|(label, _)| family_from_label(label))
}

/// The family carrying `label` in its serde (and backup) representation.
fn family_from_label(label: &str) -> Option<KvFamily> {
    KvFamily::ALL
        .into_iter()
        .find(|family| family.label() == label)
}

/// The family of a backup entry key: the target registry first, then the recorded layout.
fn resolve_family(
    backup: &MemoryBackup,
    prefixes: &KvPrefixRegistry,
    key: &str,
) -> Option<KvFamily> {
    prefixes
        .family_of(key)
        .or_else(|| recorded_family(&backup.prefixes, key))
}

/// The key an entry is restored to: the same key with the target's current family prefix.
fn target_key(backup: &MemoryBackup, prefixes: &KvPrefixRegistry, key: &str) -> String {
    let Some((head, rest)) = key.split_once('/') else {
        return key.to_owned();
    };
    let Some(family) = resolve_family(backup, prefixes, key) else {
        return key.to_owned();
    };
    let current = prefixes.current(family);
    if current == head {
        return key.to_owned();
    }
    format!("{current}/{rest}")
}

/// The serde default of [`MemoryBackup::format`].
fn default_format() -> String {
    BACKUP_FORMAT.to_owned()
}

/// The serde default of [`MemoryBackup::version`].
const fn default_version() -> u32 {
    BACKUP_FORMAT_VERSION
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::record::MemoryScope;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn family_counts(
        records: usize,
        user_variables: usize,
        session_variables: usize,
        scope_configs: usize,
    ) -> BackupCounts {
        BackupCounts {
            records,
            user_variables,
            session_variables,
            scope_configs,
            total: records + user_variables + session_variables + scope_configs,
        }
    }

    /// One seeded in-memory store holding exactly one record entry.
    async fn seeded_backup() -> (MemoryStores, MemoryBackup) {
        let stores = MemoryStores::in_memory();
        let scope = scope();
        stores
            .kv
            .set(
                &keys::record_key(&scope, "rec-1"),
                KvValue::text("{\"id\":\"rec-1\"}"),
            )
            .await
            .expect("seed record");
        let backup = dump(&stores).await.expect("dump");
        (stores, backup)
    }

    #[tokio::test]
    async fn empty_store_dumps_and_round_trips_through_json() {
        let stores = MemoryStores::in_memory();
        let backup = dump(&stores).await.expect("dump");
        assert!(backup.is_empty());
        assert_eq!(backup.len(), 0);
        assert_eq!(backup.counts, BackupCounts::default());
        assert_eq!(backup.counts(), BackupCounts::default());
        assert_eq!(backup.format, BACKUP_FORMAT);
        assert_eq!(backup.version, BACKUP_FORMAT_VERSION);
        backup.verify().expect("verify");
        assert_eq!(
            backup.prefixes.get("records").map(String::as_str),
            Some(keys::RECORD_PREFIX)
        );
        let text = backup.to_json().expect("serialize");
        let decoded = MemoryBackup::from_json(&text).expect("decode");
        assert_eq!(decoded, backup);
    }

    #[tokio::test]
    async fn round_trip_on_in_memory_stores_preserves_every_value() {
        let stores = MemoryStores::in_memory();
        let scope = scope();
        let record = keys::record_key(&scope, "rec-1");
        let variable = keys::variable_key(&scope, "theme");
        let session_variable = keys::session_variable_key(&scope, "s1", "cursor");
        let scope_config = keys::scope_config_key("ws");
        stores
            .kv
            .set(&record, KvValue::text("{\"id\":\"rec-1\"}"))
            .await
            .expect("record");
        stores
            .kv
            .set(&variable, KvValue::bytes(vec![0xff, 0x00, 0x80]))
            .await
            .expect("variable");
        stores
            .kv
            .set(&session_variable, KvValue::text("42"))
            .await
            .expect("session variable");
        stores
            .kv
            .set(&scope_config, KvValue::text("{\"enabled\":true}"))
            .await
            .expect("scope config");

        let backup = dump(&stores).await.expect("dump");
        assert_eq!(backup.len(), 4);
        assert_eq!(backup.counts, family_counts(1, 1, 1, 1));
        backup.verify().expect("verify");

        let fresh = MemoryStores::in_memory();
        let report = restore(&fresh, &backup).await.expect("restore");
        assert!(!report.dry_run);
        assert_eq!(report.written, 4);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.total, 4);
        assert_eq!(report.processed(), 4);
        assert_eq!(report.counts, backup.counts);
        for (key, value) in &backup.entries {
            assert_eq!(
                fresh.kv.get(key).await.expect("read restored key").as_ref(),
                Some(value),
                "key `{key}` is not restored verbatim"
            );
        }
        assert_eq!(
            fresh.kv.get(&variable).await.expect("binary value"),
            Some(KvValue::bytes(vec![0xff, 0x00, 0x80]))
        );

        let again = restore(&fresh, &backup).await.expect("restore again");
        assert_eq!(again.written, 0);
        assert_eq!(again.skipped, 4);
        assert_eq!(again.counts, BackupCounts::default());
    }

    #[tokio::test]
    async fn local_file_round_trip_reproduces_the_entries() {
        let source_dir = tempfile::tempdir().expect("source dir");
        let target_dir = tempfile::tempdir().expect("target dir");
        let source = MemoryStores::local_file(source_dir.path())
            .await
            .expect("source stores");
        let scope = scope();
        source
            .kv
            .set(
                &keys::record_key(&scope, "rec-1"),
                KvValue::text("{\"id\":\"rec-1\"}"),
            )
            .await
            .expect("record");
        source
            .kv
            .set(
                &keys::variable_key(&scope, "theme"),
                KvValue::bytes(b"binary\x00\xff".to_vec()),
            )
            .await
            .expect("variable");
        source
            .kv
            .set(&keys::scope_config_key("ws"), KvValue::text("{}"))
            .await
            .expect("scope config");

        let backup = dump(&source).await.expect("dump");
        let target = MemoryStores::local_file(target_dir.path())
            .await
            .expect("target stores");
        let report = restore(&target, &backup).await.expect("restore");
        assert_eq!(report.written, backup.len());
        assert_eq!(report.skipped, 0);

        let redumped = dump(&target).await.expect("re-dump");
        assert_eq!(redumped.entries, backup.entries);
        assert_eq!(redumped.counts, backup.counts);
        assert_eq!(redumped.prefixes, backup.prefixes);
    }

    #[tokio::test]
    async fn dump_normalises_legacy_keys_to_the_current_layout() {
        let stores = MemoryStores::in_memory();
        let scope = scope();
        let legacy_record = keys::legacy_record_key(&scope, "rec-1");
        let legacy_variable = keys::legacy_variable_key(&scope, "theme");
        stores
            .kv
            .set(&legacy_record, KvValue::text("{\"id\":\"rec-1\"}"))
            .await
            .expect("legacy record");
        stores
            .kv
            .set(&legacy_variable, KvValue::text("dark"))
            .await
            .expect("legacy variable");

        let backup = dump(&stores).await.expect("dump");
        assert_eq!(backup.len(), 2);
        assert_eq!(
            backup.entries.get(&keys::record_key(&scope, "rec-1")),
            Some(&KvValue::text("{\"id\":\"rec-1\"}"))
        );
        assert_eq!(
            backup.entries.get(&keys::variable_key(&scope, "theme")),
            Some(&KvValue::text("dark"))
        );
        assert!(!backup.entries.contains_key(&legacy_record));
        assert!(!backup.entries.contains_key(&legacy_variable));
        assert_eq!(backup.counts, family_counts(1, 1, 0, 0));
        backup.verify().expect("verify");

        let target = MemoryStores::in_memory();
        restore(&target, &backup).await.expect("restore");
        assert_eq!(
            target
                .kv
                .get(&keys::record_key(&scope, "rec-1"))
                .await
                .expect("current record"),
            Some(KvValue::text("{\"id\":\"rec-1\"}"))
        );
        assert!(target
            .kv
            .get(&legacy_record)
            .await
            .expect("legacy record")
            .is_none());
    }

    #[tokio::test]
    async fn dump_prefers_the_current_key_over_its_legacy_alternative() {
        let stores = MemoryStores::in_memory();
        let scope = scope();
        stores
            .kv
            .set(
                &keys::legacy_record_key(&scope, "rec-1"),
                KvValue::text("legacy"),
            )
            .await
            .expect("legacy record");
        stores
            .kv
            .set(&keys::record_key(&scope, "rec-1"), KvValue::text("current"))
            .await
            .expect("current record");

        let backup = dump(&stores).await.expect("dump");
        assert_eq!(backup.len(), 1);
        assert_eq!(
            backup.entries.get(&keys::record_key(&scope, "rec-1")),
            Some(&KvValue::text("current"))
        );
        assert_eq!(backup.counts, family_counts(1, 0, 0, 0));
    }

    #[tokio::test]
    async fn dry_run_reports_writes_without_touching_the_target() {
        let (_source, backup) = seeded_backup().await;
        let target = MemoryStores::in_memory();
        let report = restore_with(&target, &backup, &KvPrefixRegistry::default(), true)
            .await
            .expect("dry run");
        assert!(report.dry_run);
        assert_eq!(report.written, backup.len());
        assert_eq!(report.skipped, 0);
        assert_eq!(report.total, backup.len());
        assert_eq!(report.counts, backup.counts);
        for key in backup.entries.keys() {
            assert!(
                target.kv.get(key).await.expect("read target").is_none(),
                "the dry run wrote `{key}`"
            );
        }
    }

    #[tokio::test]
    async fn restore_maps_entry_keys_through_the_target_registry() {
        let renamed = KvPrefixRegistry::new()
            .with_current(KvFamily::UserVariables, "vars_v2")
            .expect("rename");
        let scope = scope();
        let renamed_key = keys::variable_key_in(&renamed, &scope, "theme");

        // Renamed source -> default target.
        let source = MemoryStores::in_memory();
        source
            .kv
            .set(&renamed_key, KvValue::text("dark"))
            .await
            .expect("seed renamed variable");
        let backup = dump_with(&source, &renamed).await.expect("dump");
        assert_eq!(backup.entries.len(), 1);
        assert!(backup.entries.contains_key(&renamed_key));
        assert_eq!(
            backup.prefixes.get("user_variables").map(String::as_str),
            Some("vars_v2")
        );

        let default_target = MemoryStores::in_memory();
        let report = restore(&default_target, &backup).await.expect("restore");
        assert_eq!(report.written, 1);
        assert_eq!(report.counts, family_counts(0, 1, 0, 0));
        assert_eq!(
            default_target
                .kv
                .get(&keys::variable_key(&scope, "theme"))
                .await
                .expect("default layout"),
            Some(KvValue::text("dark"))
        );
        assert!(default_target
            .kv
            .get(&renamed_key)
            .await
            .expect("renamed layout")
            .is_none());

        // Default source -> renamed target (the reverse direction).
        let default_source = MemoryStores::in_memory();
        default_source
            .kv
            .set(&keys::variable_key(&scope, "theme"), KvValue::text("light"))
            .await
            .expect("seed default variable");
        let backup = dump(&default_source).await.expect("dump default layout");

        let renamed_target = MemoryStores::in_memory();
        let report = restore_with(&renamed_target, &backup, &renamed, false)
            .await
            .expect("restore into the renamed layout");
        assert_eq!(report.written, 1);
        assert_eq!(report.counts, family_counts(0, 1, 0, 0));
        assert_eq!(
            renamed_target
                .kv
                .get(&renamed_key)
                .await
                .expect("renamed layout"),
            Some(KvValue::text("light"))
        );
        assert!(renamed_target
            .kv
            .get(&keys::variable_key(&scope, "theme"))
            .await
            .expect("default layout")
            .is_none());
    }

    #[tokio::test]
    async fn verification_rejects_a_foreign_format_a_newer_version_and_tampered_counts() {
        let (_stores, backup) = seeded_backup().await;
        let text = backup.to_json().expect("serialize");
        let mut document: serde_json::Value = serde_json::from_str(&text).expect("decode JSON");

        document["format"] = serde_json::json!("openjiuwen.other.backup");
        let error = MemoryBackup::from_json(&document.to_string()).expect_err("wrong format");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(
            error.to_string().contains("openjiuwen.other.backup"),
            "{error}"
        );

        document["format"] = serde_json::json!(BACKUP_FORMAT);
        document["version"] = serde_json::json!(BACKUP_FORMAT_VERSION + 1);
        let error = MemoryBackup::from_json(&document.to_string()).expect_err("newer version");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        assert!(error.to_string().contains("format version"), "{error}");
        assert!(
            error
                .to_string()
                .contains(&(BACKUP_FORMAT_VERSION + 1).to_string()),
            "{error}"
        );

        let mut tampered = backup.clone();
        tampered.counts.records = 9;
        let error = tampered.verify().expect_err("tampered counts");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("count mismatch"), "{error}");
        assert!(error.to_string().contains("records: 9"), "{error}");
        assert!(error.to_string().contains("records: 1"), "{error}");

        document["version"] = serde_json::json!(BACKUP_FORMAT_VERSION);
        document["counts"]["records"] = serde_json::json!(9);
        let error = MemoryBackup::from_json(&document.to_string()).expect_err("tampered JSON");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("count mismatch"), "{error}");
    }

    #[test]
    fn decode_is_tolerant_of_missing_members_and_ignores_unknown_ones() {
        let text = serde_json::json!({
            "created_at": "2026-01-02T03:04:05Z",
            "unexpected": {"nested": [1, 2, 3]},
        })
        .to_string();
        let backup = MemoryBackup::from_json(&text).expect("decode");
        assert_eq!(backup.format, BACKUP_FORMAT);
        assert_eq!(backup.version, BACKUP_FORMAT_VERSION);
        assert!(backup.prefixes.is_empty());
        assert!(backup.entries.is_empty());
        assert_eq!(backup.counts, BackupCounts::default());
        assert!(backup.is_empty());
        backup.verify().expect("verify");

        let error = MemoryBackup::from_json("{\"format\": \"x\"}").expect_err("missing created_at");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("JSON"), "{error}");
    }

    #[test]
    fn from_entries_classifies_the_four_dumped_families_only() {
        let registry = KvPrefixRegistry::new();
        let scope = scope();
        let mut entries = BTreeMap::new();
        entries.insert(keys::record_key(&scope, "rec-1"), KvValue::text("r"));
        entries.insert(keys::variable_key(&scope, "theme"), KvValue::text("u"));
        entries.insert(
            keys::session_variable_key(&scope, "s1", "cursor"),
            KvValue::text("s"),
        );
        entries.insert(keys::scope_config_key("ws"), KvValue::text("c"));
        entries.insert("messages/whatever".to_owned(), KvValue::text("m"));
        entries.insert(
            keys::scope_user_mapping_key_in(&registry, "ws", "alice"),
            KvValue::text("map"),
        );
        entries.insert("unknown/elsewhere".to_owned(), KvValue::text("x"));

        let counts = BackupCounts::from_entries(&entries, &registry);
        assert_eq!(counts, family_counts(1, 1, 1, 1));

        let mut merged = counts;
        merged.merge(&counts);
        assert_eq!(merged, family_counts(2, 2, 2, 2));

        // The recorded layout classifies the same keys without a registry.
        let backup = MemoryBackup::new(recorded_prefixes(&registry), entries, Utc::now());
        assert_eq!(backup.counts, counts);
        backup.verify().expect("verify");
    }
}
