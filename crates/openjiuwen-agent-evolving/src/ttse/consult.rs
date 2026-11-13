//! Consultation: selecting and rendering tips for prompt injection.
//!
//! Ports the offline intent of Python `openjiuwen.agent_evolving.ttse.consult`
//! and `ttse/render.py`: the bank is ranked against the current query and the
//! winners are rendered as a bounded, category-grouped bullet list.
//!
//! Rendering is pure: the category order (`TipCategory::order`) is the outer
//! grouping key and the effective score is the inner one, so identical inputs
//! always produce identical text. Selection does not record hits; observation
//! bookkeeping belongs to the engine.

use crate::ttse::bank::TipBank;
use crate::ttse::config::TtseConfig;
use crate::ttse::models::{truncate_chars, ConsultResult, TipRecord};

/// Renders tips grouped by category, bounded to `max_chars` characters.
///
/// Tips are grouped in the stable `TipCategory::order` sequence (tool,
/// workflow, pitfall, environment, communication, other) and ordered inside a
/// group by descending [`TipRecord::score`], ties going to the smallest
/// `tip_id`. Each tip becomes one `- [category] rule` line; lines are joined by
/// `\n` and the whole block is cut to `max_chars` characters, so a cut block
/// ends with `…`. An empty slice or a zero budget renders nothing.
#[must_use]
pub fn render_tips(tips: &[TipRecord], max_chars: usize) -> String {
    if tips.is_empty() || max_chars == 0 {
        return String::new();
    }
    let mut ordered: Vec<&TipRecord> = tips.iter().collect();
    ordered.sort_by(|left, right| {
        left.category
            .order()
            .cmp(&right.category.order())
            .then_with(|| right.score().total_cmp(&left.score()))
            .then_with(|| left.tip_id.cmp(&right.tip_id))
    });
    let lines: Vec<String> = ordered
        .into_iter()
        .map(|tip| format!("- [{}] {}", tip.category.as_str(), tip.rule))
        .collect();
    truncate_chars(&lines.join("\n"), max_chars)
}

/// Selects the tips relevant to `query` and renders them for injection.
///
/// The bank is asked for the top [`TtseConfig::consult_limit`] tips for `query`,
/// so the tighter of `consult_top_k` and `consult_max_rules` wins. The selection
/// is rendered once at unbounded width to decide
/// [`ConsultResult::truncated`], then cut to
/// [`TtseConfig::consult_max_chars`] characters. The bank is only read: hit
/// recording happens where a consultation is actually used.
#[must_use]
pub fn consult(bank: &TipBank, query: &str, config: &TtseConfig) -> ConsultResult {
    let tips = bank.top(config.consult_limit(), Some(query));
    let full = render_tips(&tips, usize::MAX);
    let truncated = full.chars().count() > config.consult_max_chars;
    let rendered = if truncated {
        truncate_chars(&full, config.consult_max_chars)
    } else {
        full
    };
    ConsultResult {
        tips,
        rendered,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use crate::ttse::models::TipCategory;

    use super::*;

    fn tip(rule: &str, category: TipCategory, tip_id: &str) -> TipRecord {
        TipRecord::new(rule, category)
            .expect("the test rules are not blank")
            .with_id(tip_id)
    }

    fn bank_of(rules: &[&str]) -> TipBank {
        let mut bank = TipBank::with_capacity(16);
        for rule in rules {
            bank.add(TipRecord::new(*rule, TipCategory::Tool).expect("valid rule"));
        }
        bank
    }

    #[test]
    fn rendering_groups_by_category_order() {
        let tips = vec![
            tip("ask before deleting", TipCategory::Communication, "comm"),
            tip("use rg instead of grep", TipCategory::Tool, "tool"),
            tip("avoid rewriting history", TipCategory::Pitfall, "pitfall"),
            tip(
                "run fmt before the tests",
                TipCategory::Workflow,
                "workflow",
            ),
            tip("note anything unusual", TipCategory::Other, "other"),
            tip("warm the cache first", TipCategory::Environment, "env"),
        ];

        let rendered = render_tips(&tips, usize::MAX);

        assert_eq!(
            rendered,
            [
                "- [tool] use rg instead of grep",
                "- [workflow] run fmt before the tests",
                "- [pitfall] avoid rewriting history",
                "- [environment] warm the cache first",
                "- [communication] ask before deleting",
                "- [other] note anything unusual",
            ]
            .join("\n")
        );
    }

    #[test]
    fn rendering_orders_by_score_inside_a_group() {
        let weak = tip("use rg instead of grep", TipCategory::Tool, "b-weak");
        let strong =
            tip("use fd instead of find", TipCategory::Tool, "a-strong").with_observation(4, 4, 0);
        let tied = tip("quote every path", TipCategory::Tool, "a-tied");

        let rendered = render_tips(&[weak, strong, tied], usize::MAX);

        assert_eq!(
            rendered,
            [
                "- [tool] use fd instead of find",
                "- [tool] quote every path",
                "- [tool] use rg instead of grep",
            ]
            .join("\n")
        );
    }

    #[test]
    fn rendering_handles_empty_input_and_zero_budget() {
        let tips = vec![tip("use rg instead of grep", TipCategory::Tool, "tool")];
        assert_eq!(render_tips(&[], usize::MAX), "");
        assert_eq!(render_tips(&tips, 0), "");
        assert_eq!(render_tips(&tips, 1_000), "- [tool] use rg instead of grep");
    }

    #[test]
    fn consult_respects_the_configured_limit() {
        let bank = bank_of(&[
            "always read the issue first",
            "prefer small commits",
            "check the exit status",
            "quote the path",
            "explain the failure",
        ]);
        let before = bank.clone();

        let tight = TtseConfig {
            consult_top_k: 2,
            ..TtseConfig::default()
        };
        let result = consult(&bank, "check the path", &tight);
        assert_eq!(result.tips.len(), 2);
        assert!(!result.truncated);
        assert!(result.rendered.contains("- [tool]"));
        assert!(result.tips.iter().all(|tip| tip.hits == 0));

        let capped = TtseConfig {
            consult_top_k: 5,
            consult_max_rules: 3,
            ..TtseConfig::default()
        };
        assert_eq!(consult(&bank, "check the path", &capped).tips.len(), 3);
        assert_eq!(bank, before, "consultation never mutates the bank");
    }

    #[test]
    fn consult_truncates_to_the_character_budget() {
        let bank = bank_of(&["always check the workspace path before running the tests"]);
        let config = TtseConfig {
            consult_max_chars: 20,
            ..TtseConfig::default()
        };

        let result = consult(&bank, "workspace path", &config);

        assert_eq!(result.tips.len(), 1);
        assert!(result.truncated);
        assert_eq!(result.rendered.chars().count(), 20);
        assert!(result.rendered.ends_with('…'), "cut text is marked");
    }

    #[test]
    fn consult_on_an_empty_bank_or_zero_budget_is_empty() {
        let empty = TipBank::with_capacity(8);
        let result = consult(&empty, "anything", &TtseConfig::default());
        assert_eq!(result.tips.len(), 0);
        assert_eq!(result.rendered, "");
        assert!(!result.truncated);

        let bank = bank_of(&["always read the issue first"]);
        let zero = TtseConfig {
            consult_max_chars: 0,
            ..TtseConfig::default()
        };
        let result = consult(&bank, "issue", &zero);
        assert_eq!(result.tips.len(), 1);
        assert_eq!(result.rendered, "");
        assert!(result.truncated);
    }
}
