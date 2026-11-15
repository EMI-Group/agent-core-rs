//! Shared validation and normalization for the team memory model.
//!
//! Every public constructor in this module tree accepts unnormalized input and every
//! public `validate` rejects malformed values with
//! [`openjiuwen_common::error::Error::Validation`]. The helpers here are the single
//! source of truth for the limits declared in [`super`]: ids are single storage-key
//! segments, content and titles must be non-blank, importance must be finite, and tags
//! are trimmed, lowercased, and deduplicated.
use super::{MAX_CONTENT_CHARS, MAX_TAGS, MAX_TAG_CHARS, MAX_TITLE_CHARS};
use openjiuwen_common::error::{Error, Result};
use std::collections::HashSet;
/// Normalizes tag input: trims each entry, lowercases it, drops empty entries, and
/// removes case-insensitive duplicates while preserving first-seen order.
#[must_use]
pub fn normalize_tags(tags: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(tags.len());
    let mut normalized = Vec::with_capacity(tags.len());
    for tag in tags {
        let trimmed = tag.trim().to_lowercase();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.clone()) {
            normalized.push(trimmed);
        }
    }
    normalized
}
/// Validates a single storage-key component (`team_id`, `member_id`, `session_id`,
/// record id, ...).
///
/// The value must be non-blank, must not contain NUL, `/`, or `\` (they would break a
/// flat storage key), must not contain control characters, and must be at most
/// `max_chars` characters long.
pub(super) fn validate_component(field: &str, value: &str, max_chars: usize) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::validation(format!(
            "team memory `{field}` must not be empty"
        )));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!(
            "team memory `{field}` must not contain NUL (got {value:?})"
        )));
    }
    if value.contains('/') {
        return Err(Error::validation(format!(
            "team memory `{field}` must not contain '/' (got {value:?})"
        )));
    }
    if value.contains('\\') {
        return Err(Error::validation(format!(
            "team memory `{field}` must not contain a backslash (got {value:?})"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::validation(format!(
            "team memory `{field}` must not contain control characters (got {value:?})"
        )));
    }
    let length = value.chars().count();
    if length > max_chars {
        return Err(Error::validation(format!(
            "team memory `{field}` must be at most {max_chars} characters (got {length})"
        )));
    }
    Ok(())
}
/// Validates record content: it must contain a non-whitespace character and be at most
/// [`MAX_CONTENT_CHARS`] characters long.
pub(super) fn validate_content(content: &str) -> Result<()> {
    if content.trim().is_empty() {
        return Err(Error::validation("team memory `content` must not be empty"));
    }
    let length = content.chars().count();
    if length > MAX_CONTENT_CHARS {
        return Err(Error::validation(format!(
            "team memory `content` must be at most {MAX_CONTENT_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}
/// Validates a record title: it must contain a non-whitespace character and be at most
/// [`MAX_TITLE_CHARS`] characters long.
pub(super) fn validate_title(title: &str) -> Result<()> {
    if title.trim().is_empty() {
        return Err(Error::validation("team memory `title` must not be empty"));
    }
    let length = title.chars().count();
    if length > MAX_TITLE_CHARS {
        return Err(Error::validation(format!(
            "team memory `title` must be at most {MAX_TITLE_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}
/// Validates importance: it must be finite and within `0.0..=1.0`.
pub(super) fn validate_importance(importance: f32) -> Result<()> {
    if !importance.is_finite() {
        return Err(Error::validation(format!(
            "team memory `importance` must be finite (got {importance})"
        )));
    }
    if !(0.0..=1.0).contains(&importance) {
        return Err(Error::validation(format!(
            "team memory `importance` must be within 0.0..=1.0 (got {importance})"
        )));
    }
    Ok(())
}
/// Validates a record embedding: a non-empty vector whose components are all finite.
///
/// An empty vector is rejected instead of being read as "no vector": a caller that
/// meant to clear the embedding clears the option, and an embedder that returned
/// nothing is a bug worth surfacing.
pub(crate) fn validate_embedding(embedding: &[f32]) -> Result<()> {
    if embedding.is_empty() {
        return Err(Error::validation(
            "team memory `embedding` must not be empty",
        ));
    }
    if let Some((index, value)) = embedding
        .iter()
        .enumerate()
        .find(|(_, component)| !component.is_finite())
    {
        return Err(Error::validation(format!(
            "team memory `embedding` component {index} must be finite (got {value})"
        )));
    }
    Ok(())
}
/// Validates tag entries as supplied by a caller: at most [`MAX_TAGS`] entries, each
/// trimmed non-blank, free of NUL, and at most [`MAX_TAG_CHARS`] characters long.
///
/// Duplicates are allowed here because [`super::normalize_tags`] collapses them before
/// a record is stored; use [`validate_tags`] for stored records instead.
pub(super) fn validate_tag_entries(tags: &[String]) -> Result<()> {
    if tags.len() > MAX_TAGS {
        return Err(Error::validation(format!(
            "team memory `tags` must contain at most {MAX_TAGS} entries (got {})",
            tags.len()
        )));
    }
    for tag in tags {
        let trimmed = tag.trim();
        if trimmed.is_empty() {
            return Err(Error::validation(
                "team memory `tags` must not contain empty entries",
            ));
        }
        if trimmed.contains('\0') {
            return Err(Error::validation(format!(
                "team memory `tags` entry must not contain NUL (got {tag:?})"
            )));
        }
        let length = trimmed.chars().count();
        if length > MAX_TAG_CHARS {
            return Err(Error::validation(format!(
                "team memory `tags` entries must be at most {MAX_TAG_CHARS} characters (got {length})"
            )));
        }
    }
    Ok(())
}
/// Validates the tags of a stored record: [`validate_tag_entries`] plus a rejection of
/// case-insensitive duplicates (stored records hold normalized, unique tags).
pub(super) fn validate_tags(tags: &[String]) -> Result<()> {
    validate_tag_entries(tags)?;
    let mut seen = HashSet::with_capacity(tags.len());
    for tag in tags {
        let key = tag.trim().to_lowercase();
        if !seen.insert(key) {
            return Err(Error::validation(format!(
                "team memory `tags` must not contain duplicate entries (got {tag:?})"
            )));
        }
    }
    Ok(())
}
/// Normalizes a member list: trims each id, drops empty entries, and removes exact
/// duplicates while preserving first-seen order.
pub(super) fn normalize_members(members: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(members.len());
    let mut normalized = Vec::with_capacity(members.len());
    for member in members {
        let trimmed = member.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_owned()) {
            normalized.push(trimmed.to_owned());
        }
    }
    normalized
}
/// Returns `true` when every tag in `wanted` is present in `tags`, compared
/// case-insensitively. An empty `wanted` list is always satisfied.
pub(super) fn contains_all_tags(tags: &[String], wanted: &[String]) -> bool {
    if wanted.is_empty() {
        return true;
    }
    let present: HashSet<String> = tags.iter().map(|tag| tag.trim().to_lowercase()).collect();
    wanted
        .iter()
        .all(|tag| present.contains(&tag.trim().to_lowercase()))
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn normalize_tags_trims_lowercases_and_dedupes() {
        let tags = normalize_tags(vec![
            "  Storage ".to_owned(),
            "storage".to_owned(),
            String::new(),
            "   ".to_owned(),
            "SQLite".to_owned(),
        ]);
        assert_eq!(tags, vec!["storage".to_owned(), "sqlite".to_owned()]);
        assert_eq!(normalize_tags(Vec::new()).len(), 0);
    }
    #[test]
    fn validate_component_accepts_regular_ids() {
        assert!(validate_component("team_id", "review-team", 128).is_ok());
        assert!(validate_component("team_id", "a", 1).is_ok());
        assert!(validate_component("team_id", "ünïcode", 128).is_ok());
    }
    #[test]
    fn validate_component_rejects_empty_and_blank() {
        assert!(validate_component("team_id", "", 128)
            .unwrap_err()
            .to_string()
            .contains("must not be empty"));
        assert!(validate_component("team_id", "   ", 128)
            .unwrap_err()
            .to_string()
            .contains("must not be empty"));
    }
    #[test]
    fn validate_component_rejects_forbidden_characters() {
        let err = validate_component("member_id", "nul\0id", 128).unwrap_err();
        assert!(err.to_string().contains("NUL"));
        assert!(err.to_string().contains("member_id"));
        assert!(validate_component("team_id", "a/b", 128)
            .unwrap_err()
            .to_string()
            .contains("'/'"));
        assert!(validate_component("team_id", "a\\b", 128)
            .unwrap_err()
            .to_string()
            .contains("backslash"));
        assert!(validate_component("team_id", "a\tb", 128)
            .unwrap_err()
            .to_string()
            .contains("control characters"));
        assert!(validate_component("session_id", "a\u{7}b", 128)
            .unwrap_err()
            .to_string()
            .contains("control characters"));
    }
    #[test]
    fn validate_component_enforces_the_length_limit() {
        let at_limit = "x".repeat(8);
        assert!(validate_component("local_id", &at_limit, 8).is_ok());
        let over = "x".repeat(9);
        let err = validate_component("local_id", &over, 8).unwrap_err();
        assert!(err.to_string().contains("at most 8 characters"));
        assert!(err.to_string().contains("got 9"));
    }
    #[test]
    fn validate_content_requires_non_blank_and_respects_the_limit() {
        assert!(validate_content("meaningful text").is_ok());
        assert!(validate_content(" \n\t ")
            .unwrap_err()
            .to_string()
            .contains("content"));
        let at_limit = "x".repeat(MAX_CONTENT_CHARS);
        assert!(validate_content(&at_limit).is_ok());
        let over = "x".repeat(MAX_CONTENT_CHARS + 1);
        assert!(validate_content(&over)
            .unwrap_err()
            .to_string()
            .contains("at most"));
    }
    #[test]
    fn validate_title_requires_non_blank_and_respects_the_limit() {
        assert!(validate_title("Storage choice").is_ok());
        assert!(validate_title("  ")
            .unwrap_err()
            .to_string()
            .contains("title"));
        let at_limit = "x".repeat(MAX_TITLE_CHARS);
        assert!(validate_title(&at_limit).is_ok());
        let over = "x".repeat(MAX_TITLE_CHARS + 1);
        assert!(validate_title(&over)
            .unwrap_err()
            .to_string()
            .contains("at most"));
    }
    #[test]
    fn validate_importance_requires_a_finite_unit_interval_value() {
        for value in [0.0_f32, 0.5, 1.0] {
            assert!(validate_importance(value).is_ok());
        }
        assert!(validate_importance(1.5)
            .unwrap_err()
            .to_string()
            .contains("0.0..=1.0"));
        assert!(validate_importance(-0.1)
            .unwrap_err()
            .to_string()
            .contains("0.0..=1.0"));
        assert!(validate_importance(f32::NAN)
            .unwrap_err()
            .to_string()
            .contains("finite"));
        assert!(validate_importance(f32::INFINITY)
            .unwrap_err()
            .to_string()
            .contains("finite"));
    }
    #[test]
    fn validate_tag_entries_allows_duplicates_but_bounds_entries() {
        assert!(validate_tag_entries(&["a".to_owned(), "A".to_owned()]).is_ok());
        assert!(validate_tag_entries(&[String::new()])
            .unwrap_err()
            .to_string()
            .contains("empty entries"));
        assert!(validate_tag_entries(&["  ".to_owned()])
            .unwrap_err()
            .to_string()
            .contains("empty entries"));
        assert!(validate_tag_entries(&["nul\0".to_owned()])
            .unwrap_err()
            .to_string()
            .contains("NUL"));
        let over = vec!["x".repeat(MAX_TAG_CHARS + 1)];
        assert!(validate_tag_entries(&over)
            .unwrap_err()
            .to_string()
            .contains("at most"));
        let too_many = vec!["x".to_owned(); MAX_TAGS + 1];
        assert!(validate_tag_entries(&too_many)
            .unwrap_err()
            .to_string()
            .contains("at most 32 entries"));
    }
    #[test]
    fn validate_tags_rejects_case_insensitive_duplicates() {
        assert!(validate_tags(&["storage".to_owned(), "sqlite".to_owned()]).is_ok());
        let err = validate_tags(&["Storage".to_owned(), "storage".to_owned()]).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }
    #[test]
    fn normalize_members_trims_and_dedupes() {
        let members = normalize_members(vec![
            " alice ".to_owned(),
            "bob".to_owned(),
            String::new(),
            "alice".to_owned(),
        ]);
        assert_eq!(members, vec!["alice".to_owned(), "bob".to_owned()]);
    }
    #[test]
    fn contains_all_tags_is_case_insensitive() {
        let tags = vec!["Storage".to_owned(), "sqlite".to_owned()];
        assert!(contains_all_tags(&tags, &[]));
        assert!(contains_all_tags(&tags, &["storage".to_owned()]));
        assert!(contains_all_tags(
            &tags,
            &["STORAGE".to_owned(), "SQLite".to_owned()]
        ));
        assert!(!contains_all_tags(
            &tags,
            &["storage".to_owned(), "redis".to_owned()]
        ));
        assert!(!contains_all_tags(&tags, &["redis".to_owned()]));
    }
}
