//! Runtime key-prefix registry mapping logical memory families to current prefixes and historical aliases.
//!
//! Python keeps a process-global `kv_prefix_registry` singleton
//! (`core/memory/common/kv_prefix_registry.py`). Here the registry is a plain value that the
//! code needing it receives, so there is no hidden global mutable state and tests as well as
//! multi-tenant processes stay isolated.
//!
//! [`KvPrefixRegistry::default`] reproduces the Python key layout exactly, built from the
//! prefix constants and the `(current, legacy)` pair table in [`super::keys`]:
//!
//! | family | current prefix | aliases | storage area |
//! | --- | --- | --- | --- |
//! | `records` | `UMD` | `user_mem` | KV |
//! | `user_variables` | `user_var` | `user_variable` | KV |
//! | `session_variables` | `session_var` | `session_variable` | KV |
//! | `scope_config` | `memory_scope_config` | `scope_config` | KV |
//! | `scope_user_mapping` | `scope_user_mapping` | — | KV |
//! | `messages` | `messages` | — | document collection |
//!
//! [`KvPrefixRegistry::register_current`] renames a family and demotes the previous current
//! prefix into the alias list, so data already written under it stays readable through the
//! legacy fallback in [`super::legacy`]; [`KvPrefixRegistry::register_legacy`] records a
//! historical alias explicitly. The `*_in` builders of [`super::keys`] take the registry, and
//! the plain builders keep using the default one.
//!
//! The lock seam lives in [`super::kv_lock`] rather than in a `lock` module because
//! `crate::lite::lock` already owns the crate-root `lock` name.
//!
//! Every prefix is a single key/collection segment: non-empty, at most 128 characters, and
//! free of `/`, `\` and NUL, so appending `/{rest}` always keeps the namespace separable.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::keys;

/// Maximum accepted length of one prefix, in characters.
const MAX_PREFIX_LENGTH: usize = 128;

/// Where a memory family is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageArea {
    /// A key-value namespace (`KvStore`).
    Kv,
    /// A document collection (`DocumentStore`).
    Document,
}

/// Logical memory data families.
///
/// The declaration order is also the [`KvFamily::ALL`] order, the `BTreeMap` order of a
/// [`KvPrefixRegistry`], and therefore the order of [`KvPrefixRegistry::all_prefixes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvFamily {
    /// Memory records (`UMD/{user}/{scope}/{id}`).
    Records,
    /// User-scoped variables (`user_var/{user}/{scope}/{name}`).
    UserVariables,
    /// Session-scoped variables (`session_var/{user}/{scope}/{session}/{name}`).
    SessionVariables,
    /// Per-scope configuration (`memory_scope_config/{scope_id}`).
    ScopeConfig,
    /// Persisted scope↔user mapping (`scope_user_mapping/{scope_id}/{user_id}`).
    ScopeUserMapping,
    /// Conversation history, in a document collection.
    Messages,
}

impl KvFamily {
    /// Every family, in the documented order.
    pub const ALL: [KvFamily; 6] = [
        KvFamily::Records,
        KvFamily::UserVariables,
        KvFamily::SessionVariables,
        KvFamily::ScopeConfig,
        KvFamily::ScopeUserMapping,
        KvFamily::Messages,
    ];

    /// The stable lowercase name used in validation messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            KvFamily::Records => "records",
            KvFamily::UserVariables => "user_variables",
            KvFamily::SessionVariables => "session_variables",
            KvFamily::ScopeConfig => "scope_config",
            KvFamily::ScopeUserMapping => "scope_user_mapping",
            KvFamily::Messages => "messages",
        }
    }

    /// Where the family is stored; only [`StorageArea::Document`] for [`KvFamily::Messages`].
    #[must_use]
    pub const fn area(self) -> StorageArea {
        match self {
            KvFamily::Messages => StorageArea::Document,
            _ => StorageArea::Kv,
        }
    }
}

/// The current prefix of one family plus its historical aliases.
///
/// `aliases` is empty for a family that never had another prefix, and the order of the
/// aliases is preserved: [`KvPrefixRegistry::register_current`] pushes the demoted prefix to
/// the front, so the most recently retired prefix is tried first.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyPrefixes {
    /// The prefix every write of this family uses.
    pub current: String,
    /// Historical prefixes, consulted read-only by the legacy fallback.
    #[serde(default)]
    pub aliases: Vec<String>,
}

impl FamilyPrefixes {
    /// Whether `head` is this family's current prefix or one of its aliases.
    fn matches(&self, head: &str) -> bool {
        self.current == head || self.aliases.iter().any(|alias| alias == head)
    }
}

/// A mutable map from logical family to current prefix plus historical aliases.
///
/// Build it with [`KvPrefixRegistry::new`] (identical to [`KvPrefixRegistry::default`]) or
/// [`KvPrefixRegistry::from_json`]; both keep every family present. Passing the registry to
/// [`super::keys`]' `*_in` builders and to [`super::legacy`] is what makes a renamed keyspace
/// discoverable without changing any other code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvPrefixRegistry {
    /// One entry per [`KvFamily`]; complete after `new`/`Default`/`from_json`.
    #[serde(default = "default_families")]
    families: BTreeMap<KvFamily, FamilyPrefixes>,
}

impl Default for KvPrefixRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl KvPrefixRegistry {
    /// Builds the registry that reproduces the documented (Python) key layout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: default_families(),
        }
    }

    /// Installs `prefix` as the current prefix of `family`.
    ///
    /// `Ok(())` without changes when `prefix` already is the current one. Otherwise the
    /// previous current prefix is moved to the front of `aliases` (so data already written
    /// under it stays readable), duplicates are dropped, and `prefix` becomes the current
    /// prefix. An empty, whitespace-only, over-long, or `/`/`\`/NUL-containing `prefix` is
    /// `Error::Validation`.
    pub fn register_current(&mut self, family: KvFamily, prefix: impl Into<String>) -> Result<()> {
        let prefix = prefix.into();
        validate_prefix(family, &prefix)?;
        let entry = self.entry(family);
        if entry.current == prefix {
            return Ok(());
        }
        let retired = std::mem::replace(&mut entry.current, prefix);
        let mut aliases = Vec::with_capacity(entry.aliases.len() + 1);
        push_unique(&mut aliases, retired);
        for alias in entry.aliases.drain(..) {
            push_unique(&mut aliases, alias);
        }
        // A prefix that becomes the current one again stops being an alias.
        aliases.retain(|alias| alias != &entry.current);
        entry.aliases = aliases;
        Ok(())
    }

    /// Records `prefix` as a historical alias of `family`.
    ///
    /// `Ok(())` without changes when `prefix` is already the current prefix (it can never be
    /// an alias as well) or already an alias. An invalid prefix is `Error::Validation`.
    pub fn register_legacy(&mut self, family: KvFamily, prefix: impl Into<String>) -> Result<()> {
        let prefix = prefix.into();
        validate_prefix(family, &prefix)?;
        let entry = self.entry(family);
        if entry.current == prefix || entry.aliases.contains(&prefix) {
            return Ok(());
        }
        entry.aliases.push(prefix);
        Ok(())
    }

    /// Builder form of [`KvPrefixRegistry::register_current`].
    pub fn with_current(mut self, family: KvFamily, prefix: impl Into<String>) -> Result<Self> {
        self.register_current(family, prefix)?;
        Ok(self)
    }

    /// Builder form of [`KvPrefixRegistry::register_legacy`].
    pub fn with_legacy(mut self, family: KvFamily, prefix: impl Into<String>) -> Result<Self> {
        self.register_legacy(family, prefix)?;
        Ok(self)
    }

    /// The prefix every write of `family` uses.
    ///
    /// A registry decoded straight from JSON without [`KvPrefixRegistry::validate`] may lack
    /// a family; `current` then answers with that family's documented default prefix.
    #[must_use]
    pub fn current(&self, family: KvFamily) -> &str {
        self.families.get(&family).map_or_else(
            || default_prefix(family),
            |prefixes| prefixes.current.as_str(),
        )
    }

    /// The historical aliases of `family`, in registration order.
    #[must_use]
    pub fn aliases(&self, family: KvFamily) -> &[String] {
        &self.family_prefixes(family).aliases
    }

    /// The current prefix plus aliases of `family`.
    ///
    /// An incomplete registry (only reachable by decoding JSON without validation) answers
    /// with an empty entry; [`KvPrefixRegistry::validate`] reports the missing family.
    #[must_use]
    pub fn family_prefixes(&self, family: KvFamily) -> &FamilyPrefixes {
        self.families.get(&family).unwrap_or(&EMPTY_FAMILY_PREFIXES)
    }

    /// The current prefix of `family`, then its aliases in order.
    pub fn prefixes(&self, family: KvFamily) -> impl Iterator<Item = &str> + '_ {
        let prefixes = self.family_prefixes(family);
        std::iter::once(prefixes.current.as_str())
            .chain(prefixes.aliases.iter().map(String::as_str))
    }

    /// Every prefix of every family, in family order, de-duplicated.
    ///
    /// The order is [`KvFamily::ALL`], each family contributing its current prefix first and
    /// then its aliases; a prefix registered for two families appears once, at its first
    /// position.
    #[must_use]
    pub fn all_prefixes(&self) -> Vec<String> {
        let mut all: Vec<String> = Vec::new();
        for family in KvFamily::ALL {
            for prefix in self.prefixes(family) {
                if !all.iter().any(|existing| existing == prefix) {
                    all.push(prefix.to_owned());
                }
            }
        }
        all
    }

    /// Every family present in the registry, in [`KvFamily::ALL`] order.
    pub fn families(&self) -> impl Iterator<Item = KvFamily> + '_ {
        self.families.keys().copied()
    }

    /// The family whose current prefix or alias equals the first `/`-separated segment of `key`.
    ///
    /// Matching only the first segment keeps `user_variable/{rest}` from being mistaken for
    /// the `user_var` namespace. Families are probed in [`KvFamily::ALL`] order, so a
    /// deliberately duplicated prefix resolves to the earlier family. Returns `None` for an
    /// unknown key, the empty string, and a key whose first segment is empty.
    #[must_use]
    pub fn family_of(&self, key: &str) -> Option<KvFamily> {
        let head = key.split_once('/').map_or(key, |(head, _)| head);
        if head.is_empty() {
            return None;
        }
        KvFamily::ALL
            .into_iter()
            .find(|family| self.family_prefixes(*family).matches(head))
    }

    /// Whether `prefix` is the current prefix or an alias of `family`.
    #[must_use]
    pub fn contains_prefix(&self, family: KvFamily, prefix: &str) -> bool {
        self.family_prefixes(family).matches(prefix)
    }

    /// Re-checks every family: prefix rules, presence, and alias integrity.
    ///
    /// Rejects a missing family, an invalid current prefix or alias, an alias equal to the
    /// current prefix, and a duplicated alias, with `Error::Validation` naming the family and
    /// the offending value.
    pub fn validate(&self) -> Result<()> {
        for family in KvFamily::ALL {
            let Some(prefixes) = self.families.get(&family) else {
                return Err(Error::validation(format!(
                    "memory prefix registry has no `{}` family",
                    family.label()
                )));
            };
            validate_prefix(family, &prefixes.current)?;
            for (position, alias) in prefixes.aliases.iter().enumerate() {
                validate_prefix(family, alias)?;
                if alias == &prefixes.current {
                    return Err(Error::validation(format!(
                        "memory prefix family `{}` lists its current prefix `{alias}` as an alias",
                        family.label()
                    )));
                }
                if prefixes.aliases[..position].contains(alias) {
                    return Err(Error::validation(format!(
                        "memory prefix family `{}` lists the alias `{alias}` twice",
                        family.label()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Serializes the registry to JSON, preserving family and alias order.
    ///
    /// # Errors
    ///
    /// Returns the `serde_json` error (kind [`openjiuwen_common::error::ErrorKind::External`])
    /// annotated with the serialization context.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            Error::external(error).with_context("serializing the memory prefix registry")
        })
    }

    /// Decodes and validates a registry produced by [`KvPrefixRegistry::to_json`].
    ///
    /// A malformed document, a missing family, or any prefix/alias rule violation is
    /// `Error::Validation`, so a decoded registry is always usable for key derivation.
    pub fn from_json(text: &str) -> Result<Self> {
        let registry: Self = serde_json::from_str(text).map_err(|error| {
            Error::validation(format!("invalid memory prefix registry JSON: {error}"))
        })?;
        registry.validate()?;
        Ok(registry)
    }

    /// The mutable entry of `family`, created from the documented default when absent.
    fn entry(&mut self, family: KvFamily) -> &mut FamilyPrefixes {
        self.families
            .entry(family)
            .or_insert_with(|| FamilyPrefixes {
                current: default_prefix(family).to_owned(),
                aliases: Vec::new(),
            })
    }
}

/// Sentinel answered by [`KvPrefixRegistry::family_prefixes`] for an unvalidated registry.
static EMPTY_FAMILY_PREFIXES: FamilyPrefixes = FamilyPrefixes {
    current: String::new(),
    aliases: Vec::new(),
};

/// The documented default prefix of `family`, taken from [`super::keys`].
fn default_prefix(family: KvFamily) -> &'static str {
    match family {
        KvFamily::Records => keys::RECORD_PREFIX,
        KvFamily::UserVariables => keys::VARIABLE_PREFIX,
        KvFamily::SessionVariables => keys::SESSION_VARIABLE_PREFIX,
        KvFamily::ScopeConfig => keys::SCOPE_CONFIG_PREFIX,
        KvFamily::ScopeUserMapping => keys::SCOPE_USER_MAPPING_PREFIX,
        KvFamily::Messages => keys::MESSAGES_COLLECTION,
    }
}

/// The complete default family map, with the legacy aliases from the `keys` pair table.
fn default_families() -> BTreeMap<KvFamily, FamilyPrefixes> {
    let mut families = BTreeMap::new();
    for family in KvFamily::ALL {
        let current = default_prefix(family);
        let aliases: Vec<String> = keys::legacy_prefix(current)
            .map(str::to_owned)
            .into_iter()
            .collect();
        families.insert(
            family,
            FamilyPrefixes {
                current: current.to_owned(),
                aliases,
            },
        );
    }
    families
}

/// Appends `candidate` when it is non-empty and not present yet.
fn push_unique(aliases: &mut Vec<String>, candidate: String) {
    if !candidate.is_empty() && !aliases.contains(&candidate) {
        aliases.push(candidate);
    }
}

/// Rejects a prefix that is not exactly one safe key/collection segment.
fn validate_prefix(family: KvFamily, prefix: &str) -> Result<()> {
    if prefix.trim().is_empty() {
        return Err(Error::validation(format!(
            "memory prefix family `{}` has an empty prefix `{prefix}`",
            family.label()
        )));
    }
    if prefix.chars().any(|c| c == '/' || c == '\\' || c == '\0') {
        return Err(Error::validation(format!(
            "memory prefix family `{}` has the invalid prefix `{prefix}`: a prefix must be a \
             single segment without `/`, `\\` or NUL",
            family.label()
        )));
    }
    let length = prefix.chars().count();
    if length > MAX_PREFIX_LENGTH {
        return Err(Error::validation(format!(
            "memory prefix family `{}` has the invalid prefix `{prefix}`: {length} characters \
             exceed the {MAX_PREFIX_LENGTH}-character limit",
            family.label()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_reproduce_the_documented_layout() {
        let registry = KvPrefixRegistry::new();
        assert_eq!(registry, KvPrefixRegistry::default());
        for (family, current, aliases) in [
            (KvFamily::Records, "UMD", vec!["user_mem"]),
            (KvFamily::UserVariables, "user_var", vec!["user_variable"]),
            (
                KvFamily::SessionVariables,
                "session_var",
                vec!["session_variable"],
            ),
            (
                KvFamily::ScopeConfig,
                "memory_scope_config",
                vec!["scope_config"],
            ),
            (KvFamily::ScopeUserMapping, "scope_user_mapping", vec![]),
            (KvFamily::Messages, "messages", vec![]),
        ] {
            assert_eq!(registry.current(family), current, "{family:?}");
            assert_eq!(registry.aliases(family), aliases.as_slice(), "{family:?}");
            let mut expected = vec![current];
            expected.extend(aliases.iter().copied());
            assert_eq!(
                registry.prefixes(family).collect::<Vec<_>>(),
                expected,
                "{family:?}"
            );
        }
        assert_eq!(registry.families().collect::<Vec<_>>(), KvFamily::ALL);
        assert_eq!(
            KvFamily::ALL.map(KvFamily::label),
            [
                "records",
                "user_variables",
                "session_variables",
                "scope_config",
                "scope_user_mapping",
                "messages",
            ]
        );
        assert_eq!(KvFamily::Messages.area(), StorageArea::Document);
        assert!(KvFamily::ALL
            .into_iter()
            .filter(|family| *family != KvFamily::Messages)
            .all(|family| family.area() == StorageArea::Kv));
        assert_eq!(
            registry.all_prefixes(),
            vec![
                "UMD",
                "user_mem",
                "user_var",
                "user_variable",
                "session_var",
                "session_variable",
                "memory_scope_config",
                "scope_config",
                "scope_user_mapping",
                "messages",
            ]
        );
        assert!(registry.validate().is_ok());
    }

    #[test]
    fn register_current_demotes_the_old_current_into_the_aliases() {
        let registry = KvPrefixRegistry::new()
            .with_current(KvFamily::UserVariables, "vars_v2")
            .expect("valid prefix");
        assert_eq!(registry.current(KvFamily::UserVariables), "vars_v2");
        assert_eq!(
            registry.aliases(KvFamily::UserVariables),
            ["user_var", "user_variable"]
        );
        assert_eq!(
            registry.prefixes(KvFamily::UserVariables).next(),
            Some("vars_v2")
        );
        // Registering the current prefix again is an `Ok` no-op.
        let mut registry = registry;
        registry
            .register_current(KvFamily::UserVariables, "vars_v2")
            .expect("no-op");
        assert_eq!(
            registry.aliases(KvFamily::UserVariables),
            ["user_var", "user_variable"]
        );
        // A further rename stacks the retired prefix at the front, once.
        registry
            .register_current(KvFamily::UserVariables, "vars_v3")
            .expect("rename");
        assert_eq!(registry.current(KvFamily::UserVariables), "vars_v3");
        assert_eq!(
            registry.aliases(KvFamily::UserVariables),
            ["vars_v2", "user_var", "user_variable"]
        );
        // Renaming back to an alias reuses it instead of duplicating it.
        registry
            .register_current(KvFamily::UserVariables, "user_var")
            .expect("rename back");
        assert_eq!(
            registry.aliases(KvFamily::UserVariables),
            ["vars_v3", "vars_v2", "user_variable"]
        );
        assert!(registry.validate().is_ok());
    }

    #[test]
    fn register_legacy_dedupes_and_ignores_the_current_prefix() {
        let mut registry = KvPrefixRegistry::new();
        registry
            .register_legacy(KvFamily::Records, "UMD")
            .expect("current is not an alias");
        assert_eq!(registry.aliases(KvFamily::Records), ["user_mem"]);
        registry
            .register_legacy(KvFamily::Records, "legacy_umd")
            .expect("append");
        registry
            .register_legacy(KvFamily::Records, "legacy_umd")
            .expect("already an alias");
        assert_eq!(
            registry.aliases(KvFamily::Records),
            ["user_mem", "legacy_umd"]
        );
        assert!(registry.contains_prefix(KvFamily::Records, "legacy_umd"));
        assert!(registry.contains_prefix(KvFamily::Records, "UMD"));
        assert!(!registry.contains_prefix(KvFamily::Records, "other"));
        assert!(registry.validate().is_ok());
    }

    #[test]
    fn invalid_prefixes_are_validation_errors() {
        let long = "x".repeat(129);
        for invalid in ["", "  ", "a/b", "a\\b", "a\0b", long.as_str()] {
            let mut registry = KvPrefixRegistry::new();
            let error = registry
                .register_current(KvFamily::Records, invalid)
                .expect_err("invalid current prefix");
            assert!(
                matches!(error, Error::Validation(_)),
                "{invalid:?}: {error:?}"
            );
            let message = error.to_string();
            assert!(message.contains("records"), "{message}");
            assert!(message.contains(invalid), "{message}");
            let error = registry
                .register_legacy(KvFamily::SessionVariables, invalid)
                .expect_err("invalid alias");
            assert!(matches!(error, Error::Validation(_)), "{invalid:?}");
            let error = KvPrefixRegistry::new()
                .with_current(KvFamily::ScopeConfig, invalid)
                .expect_err("invalid builder prefix");
            assert!(matches!(error, Error::Validation(_)), "{invalid:?}");
        }
        // Exactly 128 characters is still accepted.
        let limit = "y".repeat(128);
        assert!(KvPrefixRegistry::new()
            .with_legacy(KvFamily::Records, limit.as_str())
            .is_ok());
    }

    #[test]
    fn validate_rejects_alias_duplicates_and_self_aliases() {
        let mut duplicate = KvPrefixRegistry::new();
        duplicate
            .families
            .get_mut(&KvFamily::Records)
            .expect("records")
            .aliases = vec!["user_mem".to_owned(), "user_mem".to_owned()];
        let error = duplicate.validate().expect_err("duplicate alias");
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("twice"), "{error}");

        let mut self_alias = KvPrefixRegistry::new();
        self_alias
            .families
            .get_mut(&KvFamily::ScopeConfig)
            .expect("scope config")
            .aliases
            .push("memory_scope_config".to_owned());
        let error = self_alias.validate().expect_err("self alias");
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("current prefix"), "{error}");

        let mut missing = KvPrefixRegistry::new();
        missing.families.remove(&KvFamily::Messages);
        let error = missing.validate().expect_err("missing family");
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("messages"), "{error}");
        assert_eq!(missing.current(KvFamily::Messages), "messages");
        assert!(missing.aliases(KvFamily::Messages).is_empty());
    }

    #[test]
    fn family_of_resolves_current_and_alias_keys_only() {
        let registry = KvPrefixRegistry::new();
        assert_eq!(
            registry.family_of("UMD/alice/ws/1"),
            Some(KvFamily::Records)
        );
        assert_eq!(
            registry.family_of("user_mem/alice/ws/1"),
            Some(KvFamily::Records)
        );
        assert_eq!(
            registry.family_of("user_variable/alice/ws/theme"),
            Some(KvFamily::UserVariables)
        );
        assert_eq!(
            registry.family_of("user_var/alice/ws/theme"),
            Some(KvFamily::UserVariables)
        );
        assert_eq!(
            registry.family_of("memory_scope_config/ws"),
            Some(KvFamily::ScopeConfig)
        );
        assert_eq!(
            registry.family_of("scope_user_mapping/ws/alice"),
            Some(KvFamily::ScopeUserMapping)
        );
        assert_eq!(registry.family_of("messages"), Some(KvFamily::Messages));
        assert_eq!(registry.family_of("other/alice"), None);
        assert_eq!(registry.family_of("user_varx/alice"), None);
        assert_eq!(registry.family_of(""), None);
        assert_eq!(registry.family_of("/alice"), None);
    }

    #[test]
    fn json_round_trip_preserves_aliases_and_order() {
        let registry = KvPrefixRegistry::new()
            .with_current(KvFamily::UserVariables, "vars_v2")
            .expect("rename")
            .with_legacy(KvFamily::UserVariables, "vars_v1")
            .expect("alias")
            .with_current(KvFamily::ScopeConfig, "scope_config_v2")
            .expect("rename config");
        let text = registry.to_json().expect("serialize");
        let decoded = KvPrefixRegistry::from_json(&text).expect("deserialize");
        assert_eq!(decoded, registry);
        assert_eq!(
            decoded.aliases(KvFamily::UserVariables),
            ["user_var", "user_variable", "vars_v1"]
        );
        assert_eq!(decoded.current(KvFamily::ScopeConfig), "scope_config_v2");
        assert_eq!(
            decoded.aliases(KvFamily::ScopeConfig),
            ["memory_scope_config", "scope_config"]
        );
    }

    #[test]
    fn from_json_rejects_invalid_prefixes_and_missing_families() {
        let mut invalid = KvPrefixRegistry::new();
        invalid
            .families
            .get_mut(&KvFamily::Records)
            .expect("records")
            .current = "a/b".to_owned();
        let text = invalid.to_json().expect("serialize");
        let error = KvPrefixRegistry::from_json(&text).expect_err("invalid prefix");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        let text = serde_json::json!({"families": {"records": {"current": "UMD", "aliases": []}}});
        let error = KvPrefixRegistry::from_json(&text.to_string()).expect_err("missing families");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        let error = KvPrefixRegistry::from_json("not json").expect_err("malformed");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(error.to_string().contains("JSON"), "{error}");
    }
}
