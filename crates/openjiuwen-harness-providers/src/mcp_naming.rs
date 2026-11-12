//! The `<mcp-tools>` naming preamble injected into provider system prompts.
//!
//! Providers differ in how they expose MCP tools; the harness tells the model how to address them
//! through a single XML block. For every mapping entry (server name → tool-name pattern) the
//! preamble contains one self-closing element, in server-name order (a
//! [`BTreeMap`](std::collections::BTreeMap) iterates sorted), with no trailing newline:
//!
//! ```text
//! <mcp-tools>
//! <server name="{name}" tool-name="{pattern}"/>
//! </mcp-tools>
//! ```
//!
//! The pattern may contain [`TOOL_PLACEHOLDER`] (`{tool}`), which the provider substitutes with
//! the concrete tool name. An empty mapping yields an empty string (nothing is injected).

use std::collections::BTreeMap;

/// Placeholder a tool-name pattern uses for the concrete tool name.
pub const TOOL_PLACEHOLDER: &str = "{tool}";

/// Renders the `<mcp-tools>` naming preamble for `patterns` (server name → tool-name pattern).
///
/// Returns an empty string when `patterns` is empty; otherwise one `<server .../>` line per entry
/// in server-name order, wrapped in `<mcp-tools>`/`</mcp-tools>` and without a trailing newline.
#[must_use]
pub fn mcp_tool_naming_preamble(patterns: &BTreeMap<String, String>) -> String {
    if patterns.is_empty() {
        return String::new();
    }
    let mut preamble = String::from("<mcp-tools>");
    for (name, pattern) in patterns {
        preamble.push_str("\n<server name=\"");
        preamble.push_str(name);
        preamble.push_str("\" tool-name=\"");
        preamble.push_str(pattern);
        preamble.push_str("\"/>");
    }
    preamble.push_str("\n</mcp-tools>");
    preamble
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::BTreeMap;

    use super::{mcp_tool_naming_preamble, TOOL_PLACEHOLDER};

    fn patterns(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(name, pattern)| ((*name).to_owned(), (*pattern).to_owned()))
            .collect()
    }

    #[test]
    fn placeholder_is_python_wire_value() {
        assert_eq!(TOOL_PLACEHOLDER, "{tool}");
    }

    #[test]
    fn empty_mapping_renders_nothing() {
        assert_eq!(mcp_tool_naming_preamble(&BTreeMap::new()), "");
    }

    #[test]
    fn one_entry_renders_the_block() {
        let preamble = mcp_tool_naming_preamble(&patterns(&[("fs", "mcp__fs__{tool}")]));
        assert_eq!(
            preamble,
            "<mcp-tools>\n<server name=\"fs\" tool-name=\"mcp__fs__{tool}\"/>\n</mcp-tools>"
        );
    }

    #[test]
    fn entries_are_sorted_by_server_name_without_trailing_newline() {
        let preamble = mcp_tool_naming_preamble(&patterns(&[
            ("zeta", "mcp__zeta__{tool}"),
            ("alpha", "mcp__alpha__{tool}"),
        ]));
        assert_eq!(
            preamble,
            "<mcp-tools>\n<server name=\"alpha\" tool-name=\"mcp__alpha__{tool}\"/>\n\
             <server name=\"zeta\" tool-name=\"mcp__zeta__{tool}\"/>\n</mcp-tools>"
        );
        assert!(!preamble.ends_with('\n'));
    }
}
