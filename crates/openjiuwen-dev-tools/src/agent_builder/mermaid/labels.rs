//! Identifier and label helpers for the Mermaid renderer.
//!
//! The renderer in [`super`] owns the grammar (header, node lines, edge lines);
//! this module owns the pure text work around it: sanitizing and deduplicating
//! node ids, resolving edge endpoints, escaping labels, and extracting the label
//! a `When condition met[...]` edge description carries.

use std::collections::{BTreeMap, BTreeSet};

/// The literal prefix of Python's `When condition met[...]` edge descriptions.
const WHEN_CONDITION_PREFIX: &str = "When condition met";

/// Sanitizes a node id into a Mermaid-safe identifier.
///
/// ASCII alphanumerics and `_` are kept; every other character (CJK, `-`,
/// spaces, punctuation) becomes `_` one by one, without collapsing. An empty
/// result falls back to `node_<index>` (1-based document position), and a result
/// starting with a digit is prefixed with `n_`.
pub(super) fn sanitize_id(raw: &str, index: usize) -> String {
    let mut sanitized = String::with_capacity(raw.len().max(1));
    for character in raw.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            sanitized.push(character);
        } else {
            sanitized.push('_');
        }
    }

    if sanitized.is_empty() {
        return format!("node_{index}");
    }
    if sanitized.starts_with(|character: char| character.is_ascii_digit()) {
        return format!("n_{sanitized}");
    }
    sanitized
}

/// Reserves `base` in `used`, appending `_2`, `_3`, ... until it is unique.
pub(super) fn unique_id(base: &str, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }

    let mut suffix = 2_u32;
    loop {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Resolves an edge endpoint to the id it is rendered with.
///
/// Known endpoints use their deduplicated rendered id; a dangling endpoint keeps
/// its sanitized form so the edge still points somewhere parseable.
pub(super) fn resolve_endpoint(ids: &BTreeMap<&str, String>, endpoint: &str) -> String {
    ids.get(endpoint)
        .cloned()
        .unwrap_or_else(|| sanitize_id(endpoint, 0))
}

/// Escapes free text into a label, exactly like Python's `to_mermaid`.
///
/// Backticks and double quotes become `'`; every other character is kept
/// verbatim, so whitespace, brackets, pipes, and angle brackets survive.
pub(super) fn escape_label(raw: &str) -> String {
    raw.chars()
        .map(|character| {
            if character == '`' || character == '"' {
                '\''
            } else {
                character
            }
        })
        .collect()
}

/// Extracts the label Python's `When condition met[...]` descriptions carry.
///
/// Returns `None` when `condition` is not such a description, so the caller keeps
/// rendering the whole (escaped) condition as the edge label.
///
/// The prefix match is case-sensitive and ignores surrounding whitespace, like
/// Python's anchored `^When condition met` regex. A remainder starting with `[`
/// yields the text up to its matching `]` (bracket-depth aware, so nested
/// brackets stay inside the label); any other remainder yields itself with one
/// leading `:`/`=`/`-` and its surrounding whitespace stripped.
///
/// Deviation: Python's `^When condition met[(?P<cond>.+?)]` regex is broken — the
/// `[` opens a character class, so the named group never exists, and even as
/// intended the lazy `.+?` would capture a single character. This port extracts
/// the whole label instead of reproducing that capture, so diagrams show
/// `order valid` rather than the wrapper text.
#[must_use]
pub fn when_condition_label(condition: &str) -> Option<String> {
    let remainder = condition.trim().strip_prefix(WHEN_CONDITION_PREFIX)?.trim();

    let label = match bracketed(remainder) {
        Some(inner) => inner.trim(),
        None => remainder
            .strip_prefix(&[':', '=', '-'][..])
            .unwrap_or(remainder)
            .trim(),
    };

    if label.is_empty() {
        None
    } else {
        Some(label.to_string())
    }
}

/// The text between the first `[` of `rest` and its matching `]`.
///
/// Bracket depth is tracked, so nested brackets belong to the label; an
/// unbalanced remainder has no match. Never panics.
fn bracketed(rest: &str) -> Option<&str> {
    let inner = rest.strip_prefix('[')?;
    let mut depth = 1_usize;

    for (offset, character) in inner.char_indices() {
        match character {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&inner[..offset]);
                }
            }
            _ => {}
        }
    }

    None
}

/// The escaped label of one edge.
///
/// A `When condition met[...]` condition renders the extracted label so the
/// wrapper never leaks into the diagram; every other condition renders escaped
/// verbatim, exactly as before.
pub(super) fn edge_label(condition: &str) -> String {
    match when_condition_label(condition) {
        Some(label) => escape_label(&label),
        None => escape_label(condition),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracketed_condition_yields_the_inner_text() {
        assert_eq!(
            when_condition_label("When condition met[order valid]").as_deref(),
            Some("order valid")
        );
    }

    #[test]
    fn unbracketed_condition_yields_the_remainder() {
        assert_eq!(
            when_condition_label("When condition met order valid").as_deref(),
            Some("order valid")
        );
    }

    #[test]
    fn one_leading_separator_is_stripped() {
        assert_eq!(
            when_condition_label("When condition met: order valid").as_deref(),
            Some("order valid")
        );
        assert_eq!(
            when_condition_label("When condition met = order valid").as_deref(),
            Some("order valid")
        );
        assert_eq!(
            when_condition_label("When condition met - order valid").as_deref(),
            Some("order valid")
        );
        assert_eq!(
            when_condition_label("When condition met:- order valid").as_deref(),
            Some("- order valid")
        );
    }

    #[test]
    fn bare_prefix_has_no_label() {
        assert_eq!(when_condition_label("When condition met"), None);
        assert_eq!(when_condition_label("When condition met   "), None);
        assert_eq!(when_condition_label("When condition met:"), None);
        assert_eq!(when_condition_label("When condition met[]"), None);
        assert_eq!(when_condition_label("When condition met[   ]"), None);
    }

    #[test]
    fn empty_and_blank_conditions_have_no_label() {
        assert_eq!(when_condition_label(""), None);
        assert_eq!(when_condition_label("   "), None);
        assert_eq!(when_condition_label("\n\t"), None);
    }

    #[test]
    fn nested_and_unbalanced_brackets_do_not_panic() {
        assert_eq!(
            when_condition_label("When condition met[a[b]c]").as_deref(),
            Some("a[b]c")
        );
        assert_eq!(
            when_condition_label("When condition met[order").as_deref(),
            Some("[order")
        );
        assert_eq!(
            when_condition_label("When condition met]").as_deref(),
            Some("]")
        );
        assert_eq!(
            when_condition_label("When condition met[[order]]").as_deref(),
            Some("[order]")
        );
    }

    #[test]
    fn other_conditions_have_no_label() {
        assert_eq!(
            when_condition_label("when condition met[order valid]"),
            None
        );
        assert_eq!(
            when_condition_label("When conditions met[order valid]"),
            None
        );
        assert_eq!(
            when_condition_label("Before When condition met[order]"),
            None
        );
        assert_eq!(when_condition_label("score > 0.5"), None);
        assert_eq!(when_condition_label("otherwise"), None);
    }

    #[test]
    fn only_the_literal_prefix_is_required() {
        assert_eq!(
            when_condition_label("When condition metx[order]").as_deref(),
            Some("x[order]")
        );
        assert_eq!(
            when_condition_label("When condition met_when[order]").as_deref(),
            Some("_when[order]")
        );
    }

    #[test]
    fn surrounding_and_internal_whitespace_is_trimmed() {
        assert_eq!(
            when_condition_label("  When condition met  [  order valid  ]  ").as_deref(),
            Some("order valid")
        );
        assert_eq!(
            when_condition_label("When condition met    order valid   ").as_deref(),
            Some("order valid")
        );
        assert_eq!(
            when_condition_label("When condition met\t\t[\torder valid\t]").as_deref(),
            Some("order valid")
        );
    }

    #[test]
    fn edge_label_extracts_then_escapes() {
        assert_eq!(
            edge_label("When condition met[say \"hi\" and `bye`]"),
            "say 'hi' and 'bye'"
        );
        assert_eq!(edge_label("`tick` and \"quote\""), "'tick' and 'quote'");
        assert_eq!(edge_label("When condition met"), "When condition met");
        assert_eq!(edge_label("   "), "   ");
    }

    #[test]
    fn ids_are_sanitized_prefixed_and_deduplicated() {
        assert_eq!(sanitize_id("my node 1", 1), "my_node_1");
        assert_eq!(sanitize_id("1", 2), "n_1");
        assert_eq!(sanitize_id("中文", 3), "__");
        assert_eq!(sanitize_id("", 6), "node_6");

        let mut used = BTreeSet::new();
        assert_eq!(unique_id("a", &mut used), "a");
        assert_eq!(unique_id("a", &mut used), "a_2");
        assert_eq!(unique_id("a", &mut used), "a_3");
        assert_eq!(unique_id("a_2", &mut used), "a_2_2");
    }

    #[test]
    fn endpoints_resolve_known_ids_and_sanitize_dangling_ones() {
        let mut ids: BTreeMap<&str, String> = BTreeMap::new();
        ids.insert("a b", "a_b".to_string());

        assert_eq!(resolve_endpoint(&ids, "a b"), "a_b");
        assert_eq!(resolve_endpoint(&ids, "ghost node"), "ghost_node");
        assert_eq!(resolve_endpoint(&ids, ""), "node_0");
    }
}
