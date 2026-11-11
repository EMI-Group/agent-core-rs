//! Markdown rendering of records and recall hits with a token budget.
//!
//! Both renderers emit the same document shape: a `## Long-term memory` header
//! followed by one bullet per entry, in input order. Entries are added while the
//! running estimate fits the budget; the first entry that does not fit stops the
//! list, so a line is never split and the header only appears when at least one
//! bullet fits.
use crate::record::{MemoryRecord, RecallHit};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, TokenCounter};
use std::fmt::Write as _;

use super::MEMORY_SECTION_TITLE;

/// Collapses every whitespace run in `content` to a single space.
///
/// Splitting on whitespace also trims the edges, so multi-line memory text stays a
/// single Markdown list item.
fn collapse_whitespace(content: &str) -> String {
    content.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Builds one bullet line: `- [<kind>] <content>`.
fn bullet_line(kind: &str, content: &str) -> String {
    format!("- [{kind}] {content}")
}

/// Renders `lines` under `max_tokens` with the shared header and budget rules.
///
/// Returns an empty string when the budget is zero or the first line does not fit.
/// Newline separators are not charged against the budget.
fn render_lines(lines: impl Iterator<Item = String>, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let header = format!("## {MEMORY_SECTION_TITLE}");
    let counter = ApproxTokenCounter::new();
    let mut used = counter.count_text(&header);
    let mut rendered: Vec<String> = Vec::new();
    for line in lines {
        let cost = counter.count_text(&line);
        if used.saturating_add(cost) > max_tokens {
            break;
        }
        used = used.saturating_add(cost);
        rendered.push(line);
    }
    if rendered.is_empty() {
        return String::new();
    }
    let mut out = header;
    for line in rendered {
        // Writing into a `String` cannot fail; the result is ignored deliberately.
        let _ = write!(out, "\n{line}");
    }
    out
}

/// Renders records as a Markdown bullet list without scores.
///
/// Returns `""` for empty input or when no record fits `max_tokens`.
#[must_use]
pub fn render_records(records: &[MemoryRecord], max_tokens: usize) -> String {
    render_lines(
        records
            .iter()
            .map(|record| bullet_line(record.kind.as_str(), &collapse_whitespace(&record.content))),
        max_tokens,
    )
}

/// Renders recall hits as a Markdown bullet list including the score.
///
/// Each bullet is `- [<kind>] <content> (score: <0.000>)`; the input order is
/// preserved. Returns `""` for empty input or when no hit fits `max_tokens`.
#[must_use]
pub fn render_snippets(hits: &[RecallHit], max_tokens: usize) -> String {
    render_lines(
        hits.iter().map(|hit| {
            format!(
                "{} (score: {:.3})",
                bullet_line(
                    hit.record.kind.as_str(),
                    &collapse_whitespace(&hit.record.content)
                ),
                hit.score
            )
        }),
        max_tokens,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, MemoryType};
    use openjiuwen_foundation::prompt::ApproxTokenCounter;

    /// A very large budget, so the budget never trims anything.
    const UNLIMITED: usize = usize::MAX;

    fn record(kind: MemoryType, content: &str, id: &str) -> MemoryRecord {
        MemoryRecord::new(MemoryScope::new("alice", "ws"), content, kind).with_id(id)
    }

    fn hit(record: MemoryRecord, score: f32) -> RecallHit {
        RecallHit {
            record,
            score,
            similarity: score,
            keyword: 0.0,
            importance: 0.5,
            recency: 0.5,
        }
    }

    fn counter() -> ApproxTokenCounter {
        ApproxTokenCounter::new()
    }

    #[test]
    fn empty_input_renders_nothing() {
        assert_eq!(render_records(&[], UNLIMITED), "");
        assert_eq!(render_snippets(&[], UNLIMITED), "");
    }

    #[test]
    fn records_render_header_and_bullets_without_scores() {
        let records = [
            record(MemoryType::SemanticMemory, "Alice prefers dark mode", "a"),
            record(MemoryType::UserProfile, "Alice lives in Berlin", "b"),
        ];
        let rendered = render_records(&records, UNLIMITED);
        assert_eq!(
            rendered,
            "## Long-term memory\n\
             - [semantic_memory] Alice prefers dark mode\n\
             - [user_profile] Alice lives in Berlin"
        );
        assert!(!rendered.contains("score"));
    }

    #[test]
    fn snippets_preserve_hit_order_and_format_scores() {
        let hits = [
            hit(record(MemoryType::SemanticMemory, "first", "a"), 0.87),
            hit(record(MemoryType::EpisodicMemory, "second", "b"), 0.5),
            hit(record(MemoryType::Coding, "third", "c"), 1.0),
        ];
        let rendered = render_snippets(&hits, UNLIMITED);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[0], "## Long-term memory");
        assert_eq!(lines[1], "- [semantic_memory] first (score: 0.870)");
        assert_eq!(lines[2], "- [episodic_memory] second (score: 0.500)");
        assert_eq!(lines[3], "- [coding] third (score: 1.000)");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn score_is_rounded_to_three_decimals() {
        let hits = [
            hit(record(MemoryType::Summary, "short", "a"), 0.123_456),
            hit(record(MemoryType::Summary, "long", "b"), 0.9),
        ];
        let rendered = render_snippets(&hits, UNLIMITED);
        assert!(rendered.contains("(score: 0.123)"), "{rendered}");
        assert!(rendered.contains("(score: 0.900)"), "{rendered}");
    }

    #[test]
    fn multi_line_and_whitespace_runs_collapse_to_one_line() {
        let records = [record(
            MemoryType::SemanticMemory,
            "  Alice\tprefers\n\ndark   mode  ",
            "a",
        )];
        assert_eq!(
            render_records(&records, UNLIMITED),
            "## Long-term memory\n- [semantic_memory] Alice prefers dark mode"
        );
        let hits = [hit(records[0].clone(), 0.5)];
        assert_eq!(
            render_snippets(&hits, UNLIMITED),
            "## Long-term memory\n- [semantic_memory] Alice prefers dark mode (score: 0.500)"
        );
    }

    #[test]
    fn empty_content_still_renders_a_bullet() {
        let records = [record(MemoryType::Unknown, "   ", "a")];
        assert_eq!(
            render_records(&records, UNLIMITED),
            "## Long-term memory\n- [unknown] "
        );
    }

    #[test]
    fn zero_budget_renders_nothing() {
        let records = [record(MemoryType::SemanticMemory, "content", "a")];
        let hits = [hit(records[0].clone(), 0.5)];
        assert_eq!(render_records(&records, 0), "");
        assert_eq!(render_snippets(&hits, 0), "");
    }

    #[test]
    fn header_only_budget_renders_nothing() {
        let header_cost = counter().count_text(&format!("## {MEMORY_SECTION_TITLE}"));
        let records = [record(MemoryType::SemanticMemory, "content", "a")];
        assert_eq!(render_records(&records, header_cost), "");
        assert!(header_cost > 1);
    }

    #[test]
    fn restrictive_budget_keeps_whole_lines_only() {
        let records = [
            record(MemoryType::SemanticMemory, "aaa bbb", "a"),
            record(MemoryType::SemanticMemory, "ccc ddd", "b"),
            record(MemoryType::SemanticMemory, "eee fff", "c"),
        ];
        let full = render_records(&records, UNLIMITED);
        let lines: Vec<&str> = full.lines().collect();
        assert_eq!(lines.len(), 4);

        let header = lines[0];
        let first = lines[1];
        let second = lines[2];
        // Exactly enough for the header plus the first two bullets.
        let budget = counter().count_text(header)
            + counter().count_text(first)
            + counter().count_text(second);
        let trimmed = render_records(&records, budget);
        assert_eq!(trimmed, format!("{header}\n{first}\n{second}"));

        // One token less drops the second bullet entirely; no partial line appears.
        let trimmed = render_records(&records, budget - 1);
        assert_eq!(trimmed, format!("{header}\n{first}"));
        assert!(!trimmed.contains("ccc"));
    }

    #[test]
    fn snippets_budget_trims_the_tail_without_splitting() {
        let hits = [
            hit(record(MemoryType::SemanticMemory, "alpha beta", "a"), 0.6),
            hit(record(MemoryType::SemanticMemory, "gamma delta", "b"), 0.4),
        ];
        let full = render_snippets(&hits, UNLIMITED);
        let lines: Vec<&str> = full.lines().collect();
        assert_eq!(lines.len(), 3);
        let budget = counter().count_text(lines[0])
            + counter().count_text(lines[1])
            + counter().count_text(lines[2]);
        assert_eq!(
            render_snippets(&hits, budget),
            format!("{}\n{}\n{}", lines[0], lines[1], lines[2])
        );
        // One token less drops the second bullet entirely; no partial line appears.
        assert_eq!(
            render_snippets(&hits, budget - 1),
            format!("{}\n{}", lines[0], lines[1])
        );
        assert!(!render_snippets(&hits, budget - 1).contains("gamma"));
    }

    #[test]
    fn output_is_deterministic() {
        let records = [record(MemoryType::SemanticMemory, "content", "a")];
        assert_eq!(
            render_records(&records, UNLIMITED),
            render_records(&records, UNLIMITED)
        );
    }
}
