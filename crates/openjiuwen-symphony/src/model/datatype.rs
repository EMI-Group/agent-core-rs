//! The embedded data-type compatibility vocabulary used by capability graphs.
//!
//! Ports the deterministic part of Python `openjiuwen.symphony.shared.data_types`:
//! a small, closed vocabulary of canonical type names plus the aliases commonly seen
//! in capability cards. [`DataTypeVocabulary::can_feed`] answers whether an output of
//! one type can be wired into an input of another type, which is the predicate behind
//! every `can_feed` graph edge.

use std::collections::BTreeMap;

/// The canonical type names, in the order documented by the crate contract.
const CANONICAL_TYPES: &[&str] = &[
    "any", "string", "text", "number", "integer", "boolean", "object", "array", "null", "file",
    "image", "audio",
];

/// Alias → canonical name. Every key is lowercase.
///
/// `bytes` and `text` map to themselves: both are canonical spellings even though
/// `bytes` is not part of the primary [`CANONICAL_TYPES`] table.
const ALIASES: &[(&str, &str)] = &[
    ("str", "string"),
    ("int", "integer"),
    ("float", "number"),
    ("double", "number"),
    ("bool", "boolean"),
    ("dict", "object"),
    ("map", "object"),
    ("record", "object"),
    ("json", "object"),
    ("list", "array"),
    ("tuple", "array"),
    ("bytes", "bytes"),
    ("text", "text"),
];

/// The canonical name used for blank or absent types.
const ANY: &str = "any";

/// A normalized data-type vocabulary: canonical names plus a case-insensitive alias map.
///
/// The vocabulary is immutable and cheap enough to construct on demand; it holds no
/// interior mutability, so it is safe to share across tasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTypeVocabulary {
    /// Alias → canonical name, keyed by lowercase alias.
    aliases: BTreeMap<&'static str, &'static str>,
}

impl DataTypeVocabulary {
    /// Builds the standard vocabulary (the canonical names plus [`ALIASES`]).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            aliases: ALIASES.iter().copied().collect(),
        }
    }

    /// Normalizes `raw` to a canonical (or unknown-but-lowercased) type name.
    ///
    /// The input is trimmed; a blank input becomes `any`. Aliases are matched
    /// case-insensitively, and anything unrecognized is returned lowercased and
    /// trimmed so later comparisons stay deterministic.
    #[must_use]
    pub fn normalize(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return ANY.to_string();
        }
        let lower = trimmed.to_lowercase();
        match self.aliases.get(lower.as_str()) {
            Some(canonical) => (*canonical).to_string(),
            None => lower,
        }
    }

    /// Returns the canonical type names, in their documented order.
    #[must_use]
    pub fn canonical_types(&self) -> Vec<&'static str> {
        CANONICAL_TYPES.to_vec()
    }

    /// Reports whether an output of type `source` can feed an input of type `target`.
    ///
    /// It is true when the normalized types are equal, when the target is `any`, or
    /// when the source is a subtype of the target. In particular `any` only feeds
    /// `any`: an untyped output is not a promise of any concrete type.
    #[must_use]
    pub fn can_feed(&self, source: &str, target: &str) -> bool {
        let source = self.normalize(source);
        let target = self.normalize(target);
        source == target || self.is_subtype(&source, &target)
    }

    /// Reports whether the normalized type `candidate` is a subtype of `parent`.
    ///
    /// The relation is reflexive, every type is a subtype of `any`, and the two
    /// documented widenings hold: `integer` → `number` and `text` → `string`.
    #[must_use]
    pub fn is_subtype(&self, candidate: &str, parent: &str) -> bool {
        let candidate = self.normalize(candidate);
        let parent = self.normalize(parent);
        if candidate == parent || parent == ANY {
            return true;
        }
        matches!(
            (candidate.as_str(), parent.as_str()),
            ("integer", "number") | ("text", "string")
        )
    }
}

impl Default for DataTypeVocabulary {
    fn default() -> Self {
        Self::standard()
    }
}

/// Normalizes `raw` with the standard vocabulary.
///
/// A shorthand for [`DataTypeVocabulary::standard`] followed by
/// [`DataTypeVocabulary::normalize`].
#[must_use]
pub fn normalize_type(raw: &str) -> String {
    DataTypeVocabulary::standard().normalize(raw)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn blank_and_unknown_types_normalize_deterministically() {
        let vocabulary = DataTypeVocabulary::standard();
        assert_eq!(vocabulary.normalize(""), "any");
        assert_eq!(vocabulary.normalize("   "), "any");
        assert_eq!(vocabulary.normalize("  STR  "), "string");
        assert_eq!(vocabulary.normalize("Banana"), "banana");
    }

    #[test]
    fn canonical_types_match_the_contract_table() {
        assert_eq!(
            DataTypeVocabulary::standard().canonical_types(),
            CANONICAL_TYPES.to_vec()
        );
        // `bytes` is a canonical spelling outside the primary table.
        assert!(!CANONICAL_TYPES.contains(&"bytes"));
        assert_eq!(normalize_type("bytes"), "bytes");
    }

    #[test]
    fn aliases_map_to_canonical_names() {
        let vocabulary = DataTypeVocabulary::standard();
        for (alias, canonical) in ALIASES {
            assert_eq!(vocabulary.normalize(alias), *canonical);
        }
    }

    #[test]
    fn any_only_feeds_any() {
        let vocabulary = DataTypeVocabulary::standard();
        assert!(vocabulary.can_feed("any", "any"));
        assert!(!vocabulary.can_feed("any", "string"));
        assert!(vocabulary.can_feed("string", "any"));
        assert!(!vocabulary.can_feed("", "text"));
        assert!(vocabulary.can_feed("", "any"));
    }

    #[test]
    fn documented_widenings_hold() {
        let vocabulary = DataTypeVocabulary::standard();
        assert!(vocabulary.can_feed("integer", "number"));
        assert!(vocabulary.can_feed("text", "string"));
        assert!(vocabulary.is_subtype("int", "float"));
        assert!(!vocabulary.can_feed("number", "integer"));
        assert!(!vocabulary.can_feed("string", "text"));
    }

    #[test]
    fn is_subtype_is_reflexive_and_accepts_an_unknown_parent() {
        let vocabulary = DataTypeVocabulary::standard();
        assert!(vocabulary.is_subtype("image", "image"));
        assert!(vocabulary.is_subtype("banana", "any"));
        assert!(!vocabulary.is_subtype("any", "banana"));
    }

    #[test]
    fn normalize_type_uses_the_standard_vocabulary() {
        assert_eq!(normalize_type("BOOL"), "boolean");
        assert_eq!(normalize_type(""), "any");
    }
}
