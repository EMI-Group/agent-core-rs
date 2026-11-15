//! Deterministic I/O-name and I/O-spec normalization.
//!
//! Ports `openjiuwen.symphony.shared.fingerprint.normalization` (Python, ~990 lines), the
//! layer that turns whatever an extractor wrote about a capability's inputs and outputs
//! into typed [`CapabilityIo`](crate::model::CapabilityIo) values that the graph, the
//! planner, and the evaluator can compare — and that records *why* every value looks the
//! way it does.
//!
//! # The three steps
//!
//! 1. **Names** ([`normalize_io_name`]). A raw label is redacted, compatibility-folded,
//!    lowercased, reduced to alphanumeric runs joined by single underscores, and bounded.
//!    `Read File (v2)` and `read_file-v2` both become `read_file_v2`.
//! 2. **Vocabulary** ([`IONameVocabulary`]). The corpus's own names — a descriptor's
//!    inputs and outputs, then the raw specs an LLM extraction produced — form the
//!    vocabulary the next pass resolves against, so `summary` and `summaries` agree within
//!    one build. Only unambiguous variants are aliased, and the vocabulary's SHA-256 is
//!    order-independent.
//! 3. **Specs** ([`normalize_io_specs_for_direction`]). Each declaration accepts Python's
//!    alias keys (`name`/`id`, `type`/`data_type`/`datatype`/`format`/`mime_type`,
//!    `description`/`help`, `required`/`optional`, `default`, `metadata`), resolves its
//!    name and type, coerces its flag, redacts credentials, and emits the normalized card
//!    plus one [`NormalizationDecision`] per choice and one [`NormalizationIssue`] per
//!    cleanup.
//!
//! Nothing is dropped silently: a blank name falls back *with* an issue, an unknown type
//! keeps its spelling *and* records `raw_type`, a duplicate name gets an occurrence suffix
//! *with* an audit record, and a credential-shaped name, metadata key, or default is
//! redacted with the reason recorded.
//!
//! # Offline and deterministic
//!
//! No model, network, clock, RNG, or hash-map iteration order participates: terms,
//! directions, data types, aliases, metadata keys and issue lists are all sorted, so the
//! same input yields byte-identical JSON on every run.
//!
//! ```
//! use openjiuwen_symphony::model::normalization::{
//!     builtin_io_name_vocabulary, normalize_io_specs_with_audit,
//! };
//! use serde_json::json;
//!
//! let (items, issues, decisions) = normalize_io_specs_with_audit(
//!     &json!([{"id": "Query", "format": "str"}, {"name": "", "type": "banana"}]),
//!     &builtin_io_name_vocabulary(),
//!     true,
//! );
//! assert_eq!(items[0].name, "query");
//! assert_eq!(items[0].io_type, "string");
//! assert_eq!(items[1].name, "input");
//! assert!(issues.iter().any(|issue| issue.code == "missing_io_name"));
//! assert!(decisions.iter().any(|decision| decision.field == "type"));
//! ```
//!
//! # Deviations from Python
//!
//! Each is a documented decision, not an oversight:
//!
//! - **No Unicode NFKC.** This crate does not depend on a normalization table crate, so
//!   full-width ASCII and the Unicode space separators are folded by hand and other
//!   compatibility forms (ligatures, subscripts, circled numbers) are not decomposed; see
//!   [`names::normalize_io_name`] for the exact list.
//! - **Every non-alphanumeric run collapses, including `_`.** Python keeps existing
//!   underscores, so `read__file` survives there; here it becomes `read_file`, matching
//!   [`normalize_name_key`](crate::model::normalize_name_key).
//! - **Names are bounded** at [`MAX_IO_NAME_LEN`] characters; Python has no cap.
//! - **Sensitive names are redacted.** Python keeps an `api_key` parameter name and drops
//!   only its default; this port redacts the name as well and audits it.
//! - **Metadata is redacted on the way in.** Python redacts it only in the sanitized
//!   replay state.
//! - **Unknown data types keep their spelling** instead of collapsing to the Python
//!   `unknown` sentinel, because [`DataTypeVocabulary`](crate::model::DataTypeVocabulary)
//!   has none; blank types become `any`.
//! - **[`builtin_io_name_vocabulary`] exists.** Python's vocabulary is always
//!   corpus-derived; the crate ships a small documented default corpus so normalization
//!   works before a corpus exists.
//! - **`sensitive_io_name_redacted`, `sensitive_metadata_redacted` and `invalid_required`
//!   are Rust-only issue codes**; Python's own set is `invalid_io_collection`,
//!   `invalid_io_entry`, `missing_io_name`, `io_name_not_in_vocabulary`,
//!   `duplicate_io_name`, `unknown_data_type`, and `sensitive_default_redacted`.

/// I/O-name normalization and conservative aliasing.
pub mod names;
/// I/O-spec normalization, its audit trail, and the replay-state sanitizer.
pub mod specs;
/// The observed-corpus I/O-name vocabulary and the built-in default corpus.
pub mod vocabulary;

pub use names::{
    conservative_io_name_alias, normalize_io_name, IONameResolution, MAX_IO_NAME_LEN,
    METHOD_VOCAB_ALIAS, METHOD_VOCAB_EXACT, METHOD_VOCAB_UNSEEN,
};
pub use specs::{
    normalize_io_specs, normalize_io_specs_for_direction, normalize_io_specs_with_audit,
    normalize_io_specs_with_issues, resolve_data_type, sanitize_io_specs, DataTypeResolution,
    DataTypeSource, IONormalizationResult, NormalizationDecision, NormalizationIssue,
    DATA_TYPE_VOCABULARY_VERSION, FIELD_DEFAULT, FIELD_DESCRIPTION, FIELD_METADATA, FIELD_NAME,
    FIELD_REQUIRED, FIELD_TYPE, IO_SPEC_KEYS, ISSUE_CODE_DUPLICATE_IO_NAME,
    ISSUE_CODE_INVALID_IO_COLLECTION, ISSUE_CODE_INVALID_IO_ENTRY, ISSUE_CODE_INVALID_REQUIRED,
    ISSUE_CODE_IO_NAME_NOT_IN_VOCABULARY, ISSUE_CODE_MISSING_IO_NAME,
    ISSUE_CODE_SENSITIVE_DEFAULT_REDACTED, ISSUE_CODE_SENSITIVE_IO_NAME_REDACTED,
    ISSUE_CODE_SENSITIVE_METADATA_REDACTED, ISSUE_CODE_UNKNOWN_DATA_TYPE, MAX_JSON_DEPTH,
    METHOD_COERCED, METHOD_DEDUPLICATE, METHOD_DEFAULTED, METHOD_REDACTED, METHOD_SYNTAX,
    METHOD_VOCAB_FALLBACK, VOCABULARY_DATA_TYPE, VOCABULARY_IO_NAME, VOCABULARY_REDACTION,
};
pub use vocabulary::{
    builtin_io_name_vocabulary, IONameTerm, IONameVocabulary, IoDirection, IO_NAME_VOCAB_VERSION,
    MAX_TERM_VALUES, MAX_VOCABULARY_ALIASES, MAX_VOCABULARY_TERMS,
};
