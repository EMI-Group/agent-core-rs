//! Rendering of the `MEMORY.md`-shaped index and of fact citations.
//!
//! The functions here are pure string builders: the lite module never touches the
//! filesystem, so the harness decides where a rendered index is materialized.
//! [`render_index`] groups facts by [`CodingFactType`] in declaration order and sorts
//! them by name inside each group, which makes the output independent of storage
//! order. [`citation`] renders the `path#L1-L<n>` form when a fact has a source path
//! and falls back to the stable `coding:<name>` form otherwise.
use super::fact::{CodingFact, CodingFactType, METADATA_NAME, METADATA_PATH};
use crate::record::MemoryRecord;

/// Title line of the rendered index.
const TITLE: &str = "# Memory";

/// Renders the facts as a Markdown index capped at `max_lines` lines.
///
/// The first line is always [`TITLE`]; facts follow as
/// `- [<Type>] <name> — <description>` (with `` (`<path>`) `` appended when the fact
/// has a path), grouped in the fixed order user, feedback, project, reference and
/// name-ascending inside each group. There is no blank line between groups, and the
/// result always ends with a trailing newline. `max_lines` counts the title, so
/// `max_lines <= 1` yields the title alone.
pub(crate) fn render_index(facts: &[CodingFact], max_lines: usize) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(facts.len() + 1);
    lines.push(TITLE.to_owned());
    let mut sorted: Vec<&CodingFact> = facts.iter().collect();
    sorted.sort_by(|left, right| {
        type_rank(left.fact_type)
            .cmp(&type_rank(right.fact_type))
            .then_with(|| left.name.cmp(&right.name))
    });
    for fact in sorted {
        if lines.len() >= max_lines {
            break;
        }
        lines.push(bullet(fact));
    }
    let mut rendered = lines.join("\n");
    rendered.push('\n');
    rendered
}

/// Renders one fact as an index bullet.
fn bullet(fact: &CodingFact) -> String {
    let path = fact
        .path
        .as_deref()
        .map_or_else(String::new, |path| format!(" (`{path}`)"));
    format!(
        "- [{}] {} — {}{path}",
        type_label(fact.fact_type),
        fact.name,
        fact.description
    )
}

/// The index sort rank of a fact type (declaration order).
const fn type_rank(fact_type: CodingFactType) -> u8 {
    match fact_type {
        CodingFactType::User => 0,
        CodingFactType::Feedback => 1,
        CodingFactType::Project => 2,
        CodingFactType::Reference => 3,
    }
}

/// The capitalized label the index shows for a fact type.
pub(crate) fn type_label(fact_type: CodingFactType) -> &'static str {
    match fact_type.as_str() {
        "user" => "User",
        "feedback" => "Feedback",
        "reference" => "Reference",
        _ => "Project",
    }
}

/// Renders the citation of a recorded fact.
///
/// A fact with a `path` metadata entry cites `<path>#L1-L<n>`, where `n` is the
/// content's newline count plus one. Without a path the citation is
/// `coding:<name>`, falling back to `coding:<id>` when the record has no usable name.
pub(crate) fn citation(record: &MemoryRecord) -> String {
    if let Some(path) = record.metadata_str(METADATA_PATH) {
        let lines = record.content.matches('\n').count() + 1;
        return format!("{path}#L1-L{lines}");
    }
    let name = record
        .metadata_str(METADATA_NAME)
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(&record.id);
    format!("coding:{name}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, MemoryType};
    use serde_json::json;

    fn fact(name: &str, fact_type: CodingFactType, description: &str) -> CodingFact {
        CodingFact::new(&MemoryScope::new("alice", "ws"), name, "body")
            .with_type(fact_type)
            .with_description(description)
    }

    fn stored_record(content: &str) -> MemoryRecord {
        MemoryRecord::new(MemoryScope::new("alice", "ws"), content, MemoryType::Coding)
            .with_id("00000000000000000000000a")
    }

    #[test]
    fn an_empty_index_is_just_the_title() {
        assert_eq!(render_index(&[], 200), "# Memory\n");
        assert_eq!(render_index(&[], 1), "# Memory\n");
        assert_eq!(render_index(&[], 0), "# Memory\n");
    }

    #[test]
    fn facts_are_grouped_by_type_then_name_without_blank_lines() {
        let facts = vec![
            fact("zeta", CodingFactType::Project, "last project"),
            fact("alpha", CodingFactType::Reference, "a reference"),
            fact("beta", CodingFactType::Project, "a project"),
            fact("gamma", CodingFactType::User, "the user"),
            fact("delta", CodingFactType::Feedback, "some feedback"),
        ];
        let expected = concat!(
            "# Memory\n",
            "- [User] gamma — the user\n",
            "- [Feedback] delta — some feedback\n",
            "- [Project] beta — a project\n",
            "- [Project] zeta — last project\n",
            "- [Reference] alpha — a reference\n",
        );
        assert_eq!(render_index(&facts, 200), expected);
    }

    #[test]
    fn ordering_is_deterministic_regardless_of_input_order() {
        let facts = vec![
            fact("zeta", CodingFactType::Project, "last project"),
            fact("alpha", CodingFactType::User, "the user"),
            fact("beta", CodingFactType::Project, "a project"),
        ];
        let expected = concat!(
            "# Memory\n",
            "- [User] alpha — the user\n",
            "- [Project] beta — a project\n",
            "- [Project] zeta — last project\n",
        );
        assert_eq!(render_index(&facts, 200), expected);
        let mut reversed = facts.clone();
        reversed.reverse();
        assert_eq!(render_index(&reversed, 200), expected);
    }

    #[test]
    fn a_fact_with_a_path_appends_it() {
        let mut fact = fact("build", CodingFactType::Project, "how to build");
        fact.path = Some("docs/BUILD.md".to_owned());
        assert_eq!(
            render_index(&[fact], 200),
            concat!(
                "# Memory\n",
                "- [Project] build — how to build (`docs/BUILD.md`)\n",
            )
        );
    }

    #[test]
    fn truncation_counts_the_title_line() {
        let facts = vec![
            fact("a", CodingFactType::Project, "first"),
            fact("b", CodingFactType::Project, "second"),
            fact("c", CodingFactType::Project, "third"),
        ];
        assert_eq!(
            render_index(&facts, 3),
            concat!(
                "# Memory\n",
                "- [Project] a — first\n",
                "- [Project] b — second\n",
            )
        );
        assert_eq!(
            render_index(&facts, 2),
            concat!("# Memory\n", "- [Project] a — first\n")
        );
        assert_eq!(render_index(&facts, 1), "# Memory\n");
        assert_eq!(render_index(&facts, 0), "# Memory\n");
    }

    #[test]
    fn citation_uses_the_path_and_counts_the_content_lines() {
        let record = stored_record("line one\nline two\nline three")
            .with_metadata(METADATA_PATH, json!("docs/BUILD.md"))
            .with_metadata(METADATA_NAME, json!("build"));
        assert_eq!(citation(&record), "docs/BUILD.md#L1-L3");
    }

    #[test]
    fn citation_counts_a_single_line_as_one() {
        let record = stored_record("body").with_metadata(METADATA_PATH, json!("src/lib.rs"));
        assert_eq!(citation(&record), "src/lib.rs#L1-L1");
    }

    #[test]
    fn citation_without_a_path_uses_the_name() {
        let record = stored_record("body").with_metadata(METADATA_NAME, json!("build-command"));
        assert_eq!(citation(&record), "coding:build-command");
    }

    #[test]
    fn citation_falls_back_to_the_id_when_the_name_is_missing_or_empty() {
        let record = stored_record("body").with_metadata(METADATA_NAME, json!(""));
        assert_eq!(citation(&record), "coding:00000000000000000000000a");
        let record = stored_record("body").with_metadata(METADATA_NAME, json!("   "));
        assert_eq!(citation(&record), "coding:00000000000000000000000a");
        let record = stored_record("body");
        assert_eq!(citation(&record), "coding:00000000000000000000000a");
    }
}
