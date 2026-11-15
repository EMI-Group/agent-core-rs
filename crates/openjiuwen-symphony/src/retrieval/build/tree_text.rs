//! `cid` and token helpers shared by the index-build stages.
//!
//! Ports Python `openjiuwen.symphony.retrieval.build.workflows.tree_text` (`slug_term`,
//! `text_tokens`, `join_cid`, `parent_cid`, `unique_child_cid`) and the `cid_term` /
//! `sort_tree_nodes` helpers of `build/tree/preset_writer.py` and `build/io/tree.py`.
//!
//! Everything here is hand rolled on `char` iterators — no `regex` — so the crate stays
//! free of an extra dependency while producing byte-identical results for the ASCII input
//! the build stages use.

use std::collections::BTreeSet;

use crate::retrieval::types::IndexNode;

/// Turns `value` into a lowercase slug usable as one `cid` segment.
///
/// Mirrors Python `slug_term`: every alphanumeric character is kept (lowercased), every
/// other character becomes `-`, runs of `-` collapse, leading and trailing `-` are
/// trimmed, and an empty result falls back to `fallback`.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::tree_text::slug_term;
///
/// assert_eq!(slug_term("My Skill_v2!", "skill"), "my-skill-v2");
/// assert_eq!(slug_term("***", "skill"), "skill");
/// ```
#[must_use]
pub fn slug_term(value: &str, fallback: &str) -> String {
    let mut dashed = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_alphanumeric() {
            dashed.extend(character.to_lowercase());
        } else {
            dashed.push('-');
        }
    }
    let compact = dashed
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if compact.is_empty() {
        fallback.to_string()
    } else {
        compact
    }
}

/// Turns `value` into a `PascalCase` `cid` segment.
///
/// Mirrors Python `cid_term`: the value splits on every non-ASCII-alphanumeric character,
/// each part keeps its tail behind an uppercased first character, and the parts are
/// concatenated. An input with no parts (or a blank result) falls back to `fallback`, and
/// a result that does not start with an alphabetic character is prefixed with `N` so a
/// `cid` segment never starts with a digit.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::tree_text::cid_term;
///
/// assert_eq!(cid_term("my skill", "Node"), "MySkill");
/// assert_eq!(cid_term("2 fast", "Node"), "N2Fast");
/// assert_eq!(cid_term("***", "node"), "Node");
/// ```
#[must_use]
pub fn cid_term(value: &str, fallback: &str) -> String {
    let parts: Vec<&str> = value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect();
    let parts = if parts.is_empty() {
        vec![fallback]
    } else {
        parts
    };
    let mut token = String::new();
    for part in parts {
        let mut characters = part.chars();
        if let Some(first) = characters.next() {
            token.extend(first.to_uppercase());
            token.push_str(characters.as_str());
        }
    }
    if token.is_empty() {
        token = fallback.to_string();
    }
    if token.is_empty() {
        // An empty fallback leaves nothing to prefix; a blank segment is more useful to a
        // caller than a panic, and `unique_child_cid` still produces a unique cid.
        return token;
    }
    if token.chars().next().is_some_and(char::is_alphabetic) {
        token
    } else {
        format!("N{token}")
    }
}

/// Joins a parent `cid` and a child segment with `.`, tolerating an empty parent.
#[must_use]
pub fn join_cid(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}.{child}")
    }
}

/// The parent `cid` of `cid`, or `""` when the `cid` has no `.` separator.
#[must_use]
pub fn parent_cid(cid: &str) -> &str {
    cid.rfind('.').map_or("", |index| &cid[..index])
}

/// How many `.`-separated segments `cid` has; `0` for a blank `cid`.
///
/// Python's `sort_tree_nodes` counts `len(cid.split("."))`, which is `1` for an empty
/// string; this port reports `0` so a blank `cid` sorts with the other blanks instead of
/// claiming to be a root-level segment.
#[must_use]
pub fn cid_depth(cid: &str) -> usize {
    if cid.trim().is_empty() {
        0
    } else {
        cid.split('.').count()
    }
}

/// Returns `join_cid(parent, segment)`, suffixing `-2`, `-3`, … until it is unused.
///
/// The returned cid is inserted into `used`, so a caller that calls this once per item
/// never has to remember to record the result itself.
///
/// ```
/// use std::collections::BTreeSet;
/// use openjiuwen_symphony::retrieval::build::tree_text::unique_child_cid;
///
/// let mut used = BTreeSet::new();
/// assert_eq!(unique_child_cid("Skills", "tool", &mut used), "Skills.tool");
/// assert_eq!(unique_child_cid("Skills", "tool", &mut used), "Skills.tool-2");
/// assert_eq!(unique_child_cid("Skills", "tool", &mut used), "Skills.tool-3");
/// ```
#[must_use]
pub fn unique_child_cid(parent: &str, segment: &str, used: &mut BTreeSet<String>) -> String {
    let base = join_cid(parent, segment);
    let candidate = if used.contains(&base) {
        let mut index = 2_usize;
        loop {
            let attempt = join_cid(parent, &format!("{segment}-{index}"));
            if !used.contains(&attempt) {
                break attempt;
            }
            index += 1;
        }
    } else {
        base
    };
    used.insert(candidate.clone());
    candidate
}

/// Splits `values` into lowercase alphanumeric tokens.
///
/// Mirrors Python `text_tokens`: `-`, `_` and `.` become separators, the remaining
/// whitespace-separated tokens are lowercased, and every character that is not
/// alphanumeric is dropped. The result is a [`BTreeSet`], so token order is deterministic.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::tree_text::text_tokens;
///
/// let tokens = text_tokens(&["Web-Search_v2", "skills"]);
/// assert_eq!(
///     tokens.into_iter().collect::<Vec<_>>(),
///     vec!["search", "skills", "v2", "web"]
/// );
/// ```
#[must_use]
pub fn text_tokens(values: &[&str]) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for value in values {
        for token in value.replace(['-', '_', '.'], " ").split_whitespace() {
            let cleaned: String = token
                .trim()
                .to_lowercase()
                .chars()
                .filter(|character| character.is_alphanumeric())
                .collect();
            if !cleaned.is_empty() {
                tokens.insert(cleaned);
            }
        }
    }
    tokens
}

/// Sorts `nodes` by `(depth, cid)` — the order every writer emits.
///
/// Mirrors Python `sort_tree_nodes`, so a parent always precedes its descendants and
/// siblings stay in `cid` order.
///
/// ```
/// use openjiuwen_symphony::retrieval::build::tree_text::sort_nodes;
/// use openjiuwen_symphony::retrieval::types::IndexNode;
///
/// let mut nodes = vec![IndexNode::branch("A.B"), IndexNode::branch("A")];
/// sort_nodes(&mut nodes);
/// assert_eq!(nodes[0].cid, "A");
/// ```
pub fn sort_nodes(nodes: &mut [IndexNode]) {
    nodes.sort_by(|left, right| {
        (left.depth(), left.cid.as_str()).cmp(&(right.depth(), right.cid.as_str()))
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn slug_term_lowercases_and_collapses_separators() {
        assert_eq!(slug_term("My Skill_v2!", "skill"), "my-skill-v2");
        assert_eq!(slug_term("--a--b--", "skill"), "a-b");
        assert_eq!(slug_term("", "skill"), "skill");
        assert_eq!(slug_term("***", "skill"), "skill");
        assert_eq!(slug_term("能力 Skill", "skill"), "能力-skill");
    }

    #[test]
    fn cid_term_pascal_cases_and_prefixes_digits() {
        assert_eq!(cid_term("my skill", "Node"), "MySkill");
        assert_eq!(cid_term("web.Search-v2", "Node"), "WebSearchV2");
        assert_eq!(cid_term("***", "node"), "Node");
        assert_eq!(cid_term("2 fast", "Node"), "N2Fast");
        assert_eq!(cid_term("", "Node"), "Node");
        assert_eq!(cid_term("", ""), "");
        assert_eq!(cid_term("能力", "Node"), "Node");
    }

    #[test]
    fn cid_helpers_join_split_and_measure() {
        assert_eq!(join_cid("", "A"), "A");
        assert_eq!(join_cid("A", "B"), "A.B");
        assert_eq!(parent_cid("A.B.C"), "A.B");
        assert_eq!(parent_cid("A"), "");
        assert_eq!(parent_cid(""), "");
        assert_eq!(cid_depth("A.B.C"), 3);
        assert_eq!(cid_depth("A"), 1);
        assert_eq!(cid_depth(""), 0);
    }

    #[test]
    fn unique_child_cid_suffixes_until_free_and_records_the_result() {
        let mut used = BTreeSet::new();
        assert_eq!(unique_child_cid("Skills", "tool", &mut used), "Skills.tool");
        assert!(used.contains("Skills.tool"));
        used.insert("Skills.tool-3".to_string());
        assert_eq!(
            unique_child_cid("Skills", "tool", &mut used),
            "Skills.tool-2"
        );
        assert_eq!(
            unique_child_cid("Skills", "tool", &mut used),
            "Skills.tool-4"
        );
        assert_eq!(unique_child_cid("", "top", &mut used), "top");
    }

    #[test]
    fn text_tokens_splits_lowercases_and_drops_punctuation() {
        let tokens = text_tokens(&["Web-Search_v2", "Skills.foo", "  "]);
        assert_eq!(
            tokens.into_iter().collect::<Vec<_>>(),
            vec!["foo", "search", "skills", "v2", "web"]
        );
    }

    #[test]
    fn text_tokens_keeps_alphanumeric_only() {
        let tokens = text_tokens(&["路径 (beta)!"]);
        assert!(tokens.contains("路径"));
        assert!(tokens.contains("beta"));
        assert!(!tokens.contains("("));
        assert_eq!(text_tokens(&["", "--"]), BTreeSet::new());
    }

    #[test]
    fn sort_nodes_orders_by_depth_then_cid() {
        let mut nodes = vec![
            IndexNode::branch("A.B"),
            IndexNode::branch("B"),
            IndexNode::branch("A"),
            IndexNode::leaf("A.B.C", "w"),
            IndexNode::branch(""),
        ];
        sort_nodes(&mut nodes);
        let cids: Vec<&str> = nodes.iter().map(|node| node.cid.as_str()).collect();
        assert_eq!(cids, vec!["", "A", "B", "A.B", "A.B.C"]);
    }
}
