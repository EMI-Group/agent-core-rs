//! Legacy key-layout read compatibility for the memory stores.
//!
//! Writes always use the current layout ([`super::keys`]). Reads fall back to the
//! legacy layout (the `legacy_alternatives` of a current key or prefix) only when the
//! current key is absent, and deletes remove both variants; a legacy hit is logged at
//! debug level and never migrated implicitly. See the `keys` module docs for the family
//! mapping.
//!
//! This module owns only the fallback mechanics, as free helpers over the KV trait
//! object: [`read_text`], [`list_records`], [`list_variables`], and
//! [`delete_with_legacy`], each with an `_in` variant taking the caller's
//! [`KvPrefixRegistry`] (the plain functions use the default registry). Ranking, record
//! validation, and every other storage responsibility stay in [`super::storage`], which is
//! the only caller.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::KvStore;

use crate::record::{MemoryFilter, MemoryRecord};

use super::json;
use super::keys;
use super::prefix::KvPrefixRegistry;

/// Reads the UTF-8 text stored under `key`, falling back to the alias key layout.
///
/// Returns the key that was actually found plus its text. The current key always
/// wins: an alias candidate is tried only when `key` is absent. A candidate holding
/// a non-text value is `Error::Validation` naming that key, because a point read
/// must not hide corruption behind a "missing" answer. `None` means no candidate
/// key exists at all.
#[cfg_attr(not(test), allow(dead_code))] // default-registry form; `storage` passes its own registry
pub(crate) async fn read_text(kv: &dyn KvStore, key: String) -> Result<Option<(String, String)>> {
    read_text_in(kv, key, &KvPrefixRegistry::default()).await
}

/// [`read_text`] over the alias layout of `prefixes`.
pub(crate) async fn read_text_in(
    kv: &dyn KvStore,
    key: String,
    prefixes: &KvPrefixRegistry,
) -> Result<Option<(String, String)>> {
    let mut candidates = vec![key.clone()];
    candidates.extend(keys::legacy_alternatives_in(prefixes, &key));
    for (position, candidate) in candidates.into_iter().enumerate() {
        let Some(value) = kv.get(&candidate).await? else {
            continue;
        };
        let Some(text) = value.as_str() else {
            return Err(Error::validation(format!(
                "stored memory value `{candidate}` is not UTF-8 text"
            )));
        };
        if position > 0 {
            log_read(&key, &candidate);
        }
        return Ok(Some((candidate, text.to_owned())));
    }
    Ok(None)
}

/// Lists the records under `prefix` plus the records of every alias prefix.
///
/// Records found only under an alias layout are appended after the current-layout
/// scan, and a record present in both layouts is reported once from the current one:
/// entries are deduplicated by the record's *current* key, so the current layout wins
/// even when only the alias row carries a given id. The result is sorted newest
/// first and by id ascending on ties.
#[cfg_attr(not(test), allow(dead_code))] // default-registry form; `storage` passes its own registry
pub(crate) async fn list_records(
    kv: &dyn KvStore,
    prefix: &str,
    filter: &MemoryFilter,
) -> Result<Vec<MemoryRecord>> {
    list_records_in(kv, prefix, filter, &KvPrefixRegistry::default()).await
}

/// [`list_records`] over the alias layout of `prefixes`.
pub(crate) async fn list_records_in(
    kv: &dyn KvStore,
    prefix: &str,
    filter: &MemoryFilter,
    prefixes: &KvPrefixRegistry,
) -> Result<Vec<MemoryRecord>> {
    let mut records = scan_records(kv, prefix, filter).await?;
    let mut seen: BTreeSet<String> = records
        .iter()
        .map(|record| keys::record_key_in(prefixes, &record.scope, &record.id))
        .collect();
    for legacy in keys::legacy_alternatives_in(prefixes, prefix) {
        for record in scan_records(kv, &legacy, filter).await? {
            let current = keys::record_key_in(prefixes, &record.scope, &record.id);
            if seen.insert(current.clone()) {
                log_read(&current, &legacy);
                records.push(record);
            }
        }
    }
    records.sort_by(|left, right| {
        right
            .timestamp
            .cmp(&left.timestamp)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(records)
}

/// Lists every variable under `prefix` plus the variables of every alias prefix.
///
/// Alias names are inserted only when the name is still absent, so a current-layout
/// variable always wins on a name clash. Non-text values are logged and skipped.
#[cfg_attr(not(test), allow(dead_code))] // default-registry form; `storage` passes its own registry
pub(crate) async fn list_variables(
    kv: &dyn KvStore,
    prefix: &str,
) -> Result<BTreeMap<String, String>> {
    list_variables_in(kv, prefix, &KvPrefixRegistry::default()).await
}

/// [`list_variables`] over the alias layout of `prefixes`.
pub(crate) async fn list_variables_in(
    kv: &dyn KvStore,
    prefix: &str,
    prefixes: &KvPrefixRegistry,
) -> Result<BTreeMap<String, String>> {
    let mut candidates = vec![(prefix.to_owned(), false)];
    candidates.extend(
        keys::legacy_alternatives_in(prefixes, prefix)
            .into_iter()
            .map(|legacy| (legacy, true)),
    );
    let mut variables = BTreeMap::new();
    for (candidate, from_legacy) in candidates {
        let entries = kv.get_by_prefix(&candidate).await?;
        for (key, value) in entries {
            let Some(name) = key.strip_prefix(&candidate) else {
                continue;
            };
            let Some(text) = value.as_str() else {
                tracing::warn!(
                    target: "openjiuwen_memory::store",
                    key = %key,
                    "skipping a memory variable that is not UTF-8 text"
                );
                continue;
            };
            if from_legacy {
                let current = keys::current_equivalent_in(prefixes, &key)
                    .unwrap_or_else(|| prefix.to_owned());
                log_read(&current, &key);
                variables
                    .entry(name.to_owned())
                    .or_insert_with(|| text.to_owned());
            } else {
                variables.insert(name.to_owned(), text.to_owned());
            }
        }
    }
    Ok(variables)
}

/// Deletes `key` and every alias alternative, reporting whether any of them existed.
#[cfg_attr(not(test), allow(dead_code))] // default-registry form; `storage` passes its own registry
pub(crate) async fn delete_with_legacy(kv: &dyn KvStore, key: &str) -> Result<bool> {
    delete_with_legacy_in(kv, key, &KvPrefixRegistry::default()).await
}

/// [`delete_with_legacy`] over the alias layout of `prefixes`.
pub(crate) async fn delete_with_legacy_in(
    kv: &dyn KvStore,
    key: &str,
    prefixes: &KvPrefixRegistry,
) -> Result<bool> {
    let mut candidates = vec![key.to_owned()];
    candidates.extend(keys::legacy_alternatives_in(prefixes, key));
    let mut existed = false;
    for candidate in candidates {
        if kv.exists(&candidate).await? {
            existed = true;
        }
        kv.delete(&candidate).await?;
    }
    Ok(existed)
}

/// Scans `prefix`, returning the records that match `filter` (unsorted).
///
/// Malformed or non-text entries are logged and skipped, never fatal.
async fn scan_records(
    kv: &dyn KvStore,
    prefix: &str,
    filter: &MemoryFilter,
) -> Result<Vec<MemoryRecord>> {
    let entries = kv.get_by_prefix(prefix).await?;
    let mut records = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let Some(text) = value.as_str() else {
            tracing::warn!(
                target: "openjiuwen_memory::store",
                key = %key,
                "skipping a stored memory record that is not UTF-8 text"
            );
            continue;
        };
        match json::record_from_text(text) {
            Ok(record) => {
                if filter.matches(&record) {
                    records.push(record);
                }
            }
            Err(error) => tracing::warn!(
                target: "openjiuwen_memory::store",
                key = %key,
                %error,
                "skipping a malformed stored memory record"
            ),
        }
    }
    Ok(records)
}

/// Logs a read that was served from the legacy key layout.
fn log_read(current_key: &str, legacy_key: &str) {
    tracing::debug!(
        target: "openjiuwen_memory::store",
        current_key = %current_key,
        legacy_key = %legacy_key,
        "read from a legacy memory key"
    );
}

#[cfg(test)]
mod tests {
    use super::super::prefix::KvFamily;
    use super::super::{MemoryStorage, MemoryStores};
    use super::*;

    use chrono::{DateTime, TimeZone, Utc};
    use openjiuwen_foundation::store::KvValue;

    use crate::config::MemoryScopeConfig;
    use crate::record::{MemoryRecord, MemoryScope, MemoryType};

    fn stores() -> MemoryStores {
        MemoryStores::in_memory()
    }

    fn storage(stores: &MemoryStores) -> MemoryStorage {
        MemoryStorage::new(stores.clone())
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws").session("s1")
    }

    fn timestamp(offset: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + offset, 0)
            .single()
            .expect("fixed time")
    }

    fn record(id: &str, content: &str, offset: i64) -> MemoryRecord {
        MemoryRecord::new(scope(), content, MemoryType::SemanticMemory)
            .with_id(id)
            .with_timestamp(timestamp(offset))
    }

    async fn put_legacy(stores: &MemoryStores, key: &str, value: &str) {
        stores
            .kv
            .set(key, KvValue::text(value))
            .await
            .expect("write legacy value");
    }

    async fn kv_text(stores: &MemoryStores, key: &str) -> Option<String> {
        stores
            .kv
            .get(key)
            .await
            .expect("read key")
            .and_then(|value| value.as_str().map(str::to_owned))
    }

    async fn assert_key_gone(stores: &MemoryStores, key: &str) {
        assert!(kv_text(stores, key).await.is_none(), "`{key}` still exists");
    }

    fn variables(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[tokio::test]
    async fn legacy_record_keys_are_read_without_outranking_the_current_key() {
        let stores = stores();
        let storage = storage(&stores);
        let scope = scope();
        let legacy_key = keys::legacy_record_key(&scope, "legacy");
        let legacy = record("legacy", "from the legacy layout", 0);
        let legacy_text = json::record_to_text(&legacy).expect("serialize record");
        put_legacy(&stores, &legacy_key, &legacy_text).await;
        let found = storage
            .get_record(&scope, "legacy")
            .await
            .expect("get legacy")
            .expect("legacy record readable");
        assert_eq!(found.content, "from the legacy layout");
        let listed = storage
            .list_records(&MemoryFilter::for_scope(&scope))
            .await
            .expect("list legacy");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "legacy");
        // A current-layout write of the same id wins for point reads and listings.
        let current = record("legacy", "from the current layout", 1);
        storage.put_record(&current).await.expect("put current");
        let found = storage
            .get_record(&scope, "legacy")
            .await
            .expect("get current")
            .expect("current record");
        assert_eq!(found.content, "from the current layout");
        let listed = storage
            .list_records(&MemoryFilter::for_scope(&scope))
            .await
            .expect("list both");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].content, "from the current layout");
        // Reads never migrate implicitly: the legacy row is still stored.
        assert!(kv_text(&stores, &legacy_key).await.is_some());
    }

    #[tokio::test]
    async fn legacy_user_variables_are_read_and_current_writes_win() {
        let stores = stores();
        let storage = storage(&stores);
        let scope = scope();
        let theme = keys::legacy_variable_key(&scope, "theme");
        put_legacy(&stores, &theme, "dark").await;
        put_legacy(&stores, &keys::legacy_variable_key(&scope, "lang"), "en").await;
        let value = storage
            .get_variable(&scope, None, "theme")
            .await
            .expect("get legacy");
        assert_eq!(value.as_deref(), Some("dark"));
        let listed = storage
            .list_variables(&scope, None)
            .await
            .expect("list legacy");
        assert_eq!(listed, variables(&[("theme", "dark"), ("lang", "en")]));
        storage
            .set_variable(&scope, None, "theme", "light")
            .await
            .expect("set theme");
        assert_eq!(
            kv_text(&stores, &theme).await.as_deref(),
            Some("dark"),
            "the write must land on the current key only"
        );
        let value = storage
            .get_variable(&scope, None, "theme")
            .await
            .expect("get current");
        assert_eq!(value.as_deref(), Some("light"));
        // The current-layout value wins over the legacy one, a legacy-only variable stays.
        let listed = storage
            .list_variables(&scope, None)
            .await
            .expect("list mixed");
        assert_eq!(listed, variables(&[("theme", "light"), ("lang", "en")]));
    }

    #[tokio::test]
    async fn legacy_session_variables_stay_in_their_session_namespace() {
        let stores = stores();
        let storage = storage(&stores);
        let scope = scope();
        put_legacy(
            &stores,
            &keys::legacy_session_variable_key(&scope, "s1", "theme"),
            "light",
        )
        .await;
        let value = storage
            .get_variable(&scope, Some("s1"), "theme")
            .await
            .expect("get legacy session");
        assert_eq!(value.as_deref(), Some("light"));
        let listed = storage
            .list_variables(&scope, Some("s1"))
            .await
            .expect("list legacy session");
        assert_eq!(listed, variables(&[("theme", "light")]));
        // A session variable never leaks into the user-scoped namespace.
        let value = storage
            .get_variable(&scope, None, "theme")
            .await
            .expect("get user");
        assert_eq!(value, None);
        assert!(storage
            .list_variables(&scope, None)
            .await
            .expect("list user")
            .is_empty());
        assert!(storage
            .list_variables(&scope, Some("s2"))
            .await
            .expect("list other session")
            .is_empty());
        storage
            .set_variable(&scope, Some("s1"), "theme", "blue")
            .await
            .expect("set session");
        let value = storage
            .get_variable(&scope, Some("s1"), "theme")
            .await
            .expect("get current session");
        assert_eq!(value.as_deref(), Some("blue"));
    }

    #[tokio::test]
    async fn legacy_scope_config_is_read_and_current_writes_win() {
        let stores = stores();
        let storage = storage(&stores);
        let legacy = MemoryScopeConfig {
            user_profile_definition: Some("legacy facts".to_owned()),
            semantic_memory_definition: None,
            episodic_memory_definition: None,
        };
        let legacy_json = serde_json::to_string(&legacy).expect("serialize config");
        put_legacy(&stores, &keys::legacy_scope_config_key("ws"), &legacy_json).await;
        let found = storage.get_scope_config("ws").await.expect("get legacy");
        assert_eq!(found, Some(legacy.clone()));
        let current = MemoryScopeConfig {
            user_profile_definition: Some("current facts".to_owned()),
            ..legacy
        };
        assert!(
            !storage
                .set_scope_config("ws", &current)
                .await
                .expect("set current"),
            "only the legacy key existed, so nothing was replaced"
        );
        let found = storage.get_scope_config("ws").await.expect("get current");
        assert_eq!(found, Some(current));
    }

    #[tokio::test]
    async fn deletes_remove_current_and_legacy_keys_and_report_existence() {
        let stores = stores();
        let storage = storage(&stores);
        let scope = scope();
        let shared = record("shared", "record", 0);
        let shared_json = json::record_to_text(&shared).expect("serialize record");
        storage.put_record(&shared).await.expect("put record");
        put_legacy(
            &stores,
            &keys::legacy_record_key(&scope, "shared"),
            &shared_json,
        )
        .await;
        assert!(storage.delete_record(&shared).await.expect("delete record"));
        assert_key_gone(&stores, &keys::record_key(&scope, "shared")).await;
        assert_key_gone(&stores, &keys::legacy_record_key(&scope, "shared")).await;
        assert!(!storage.delete_record(&shared).await.expect("delete again"));
        // A legacy-only row is reported `true` on its first delete and `false` afterwards.
        let orphan = record("orphan", "legacy only", 1);
        let orphan_json = json::record_to_text(&orphan).expect("serialize orphan");
        put_legacy(
            &stores,
            &keys::legacy_record_key(&scope, "orphan"),
            &orphan_json,
        )
        .await;
        assert!(storage.delete_record(&orphan).await.expect("delete orphan"));
        assert!(!storage
            .delete_record(&orphan)
            .await
            .expect("delete orphan again"));

        storage
            .set_variable(&scope, None, "theme", "dark")
            .await
            .expect("set variable");
        put_legacy(&stores, &keys::legacy_variable_key(&scope, "theme"), "grey").await;
        assert!(storage
            .delete_variable(&scope, None, "theme")
            .await
            .expect("delete variable"));
        assert_key_gone(&stores, &keys::variable_key(&scope, "theme")).await;
        assert_key_gone(&stores, &keys::legacy_variable_key(&scope, "theme")).await;
        assert!(!storage
            .delete_variable(&scope, None, "theme")
            .await
            .expect("delete variable again"));

        let config = MemoryScopeConfig {
            user_profile_definition: None,
            semantic_memory_definition: Some("facts".to_owned()),
            episodic_memory_definition: None,
        };
        let config_json = serde_json::to_string(&config).expect("serialize config");
        storage
            .set_scope_config("ws", &config)
            .await
            .expect("set config");
        put_legacy(&stores, &keys::legacy_scope_config_key("ws"), &config_json).await;
        assert!(storage
            .delete_scope_config("ws")
            .await
            .expect("delete config"));
        assert_key_gone(&stores, &keys::scope_config_key("ws")).await;
        assert_key_gone(&stores, &keys::legacy_scope_config_key("ws")).await;
        assert!(!storage
            .delete_scope_config("ws")
            .await
            .expect("delete config again"));
    }

    #[tokio::test]
    async fn direct_listings_prefer_the_current_layout_on_a_key_clash() {
        let stores = stores();
        let scope = scope();
        let current = record("shared", "from the current layout", 1);
        let legacy = record("shared", "from the legacy layout", 1);
        put_legacy(
            &stores,
            &keys::record_key(&scope, "shared"),
            &json::record_to_text(&current).expect("serialize current"),
        )
        .await;
        put_legacy(
            &stores,
            &keys::legacy_record_key(&scope, "shared"),
            &json::record_to_text(&legacy).expect("serialize legacy"),
        )
        .await;
        let listed = list_records(
            stores.kv.as_ref(),
            &keys::record_prefix(&scope),
            &MemoryFilter::for_scope(&scope),
        )
        .await
        .expect("list records");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].content, "from the current layout");

        let prefix = keys::variable_prefix(&scope);
        put_legacy(&stores, &keys::variable_key(&scope, "theme"), "dark").await;
        put_legacy(&stores, &keys::legacy_variable_key(&scope, "theme"), "grey").await;
        put_legacy(&stores, &keys::legacy_variable_key(&scope, "lang"), "en").await;
        let listed = list_variables(stores.kv.as_ref(), &prefix)
            .await
            .expect("list variables");
        assert_eq!(listed, variables(&[("theme", "dark"), ("lang", "en")]));
    }

    #[tokio::test]
    async fn default_registry_wrappers_read_and_delete_the_documented_layout() {
        let stores = stores();
        let scope = scope();
        let current_key = keys::variable_key(&scope, "theme");
        let legacy_key = keys::legacy_variable_key(&scope, "theme");
        put_legacy(&stores, &legacy_key, "grey").await;
        let found = read_text(stores.kv.as_ref(), current_key.clone())
            .await
            .expect("read")
            .expect("legacy hit");
        assert_eq!(found.0, legacy_key);
        assert_eq!(found.1, "grey");
        assert!(delete_with_legacy(stores.kv.as_ref(), &current_key)
            .await
            .expect("delete"));
        assert!(!delete_with_legacy(stores.kv.as_ref(), &current_key)
            .await
            .expect("delete again"));
        assert!(read_text(stores.kv.as_ref(), current_key)
            .await
            .expect("read")
            .is_none());
    }

    #[tokio::test]
    async fn custom_registry_reads_and_deletes_every_alias() {
        let stores = stores();
        let scope = scope();
        let prefixes = KvPrefixRegistry::new()
            .with_current(KvFamily::UserVariables, "vars_v2")
            .expect("rename user variables");
        let current = keys::variable_key_in(&prefixes, &scope, "theme");
        assert_eq!(current, "vars_v2/alice/ws/theme");
        put_legacy(&stores, &current, "dark").await;
        put_legacy(
            &stores,
            &keys::legacy_variable_key(&scope, "theme"),
            "older",
        )
        .await;
        put_legacy(&stores, &keys::variable_key(&scope, "theme"), "grey").await;
        let prefix = keys::variable_prefix_in(&prefixes, &scope);
        assert_eq!(prefix, "vars_v2/alice/ws/");
        let listed = list_variables_in(stores.kv.as_ref(), &prefix, &prefixes)
            .await
            .expect("list variables");
        assert_eq!(listed, variables(&[("theme", "dark")]));
        let found = read_text_in(stores.kv.as_ref(), current.clone(), &prefixes)
            .await
            .expect("read")
            .expect("current hit");
        assert_eq!(found.0, current);
        assert_eq!(found.1, "dark");
        assert!(
            delete_with_legacy_in(stores.kv.as_ref(), &current, &prefixes)
                .await
                .expect("delete")
        );
        assert_key_gone(&stores, &current).await;
        assert_key_gone(&stores, &keys::variable_key(&scope, "theme")).await;
        assert_key_gone(&stores, &keys::legacy_variable_key(&scope, "theme")).await;
    }

    #[tokio::test]
    async fn custom_registry_lists_records_of_the_renamed_family() {
        let stores = stores();
        let scope = scope();
        let prefixes = KvPrefixRegistry::new()
            .with_current(KvFamily::Records, "MEMV2")
            .expect("rename records");
        let current = record("current", "from MEMV2", 1);
        put_legacy(
            &stores,
            &keys::record_key_in(&prefixes, &scope, "current"),
            &json::record_to_text(&current).expect("serialize current"),
        )
        .await;
        let legacy = record("legacy", "from user_mem", 0);
        put_legacy(
            &stores,
            &keys::legacy_record_key(&scope, "legacy"),
            &json::record_to_text(&legacy).expect("serialize legacy"),
        )
        .await;
        let prefix = keys::record_prefix_in(&prefixes, &scope);
        assert_eq!(prefix, "MEMV2/alice/ws/");
        let listed = list_records_in(
            stores.kv.as_ref(),
            &prefix,
            &MemoryFilter::for_scope(&scope),
            &prefixes,
        )
        .await
        .expect("list records");
        assert_eq!(
            listed.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["current", "legacy"]
        );
    }
}
