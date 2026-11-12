//! Size and token accounting for a rendered system prompt.
//!
//! Ports `openjiuwen/harness/prompts/report.py`: the report groups the sections that
//! actually reach the prompt by [`PromptCategory`] and estimates their size so callers
//! can log or assert on prompt growth. The token estimate is a heuristic — Chinese
//! text costs roughly one token per 2.5 characters, other languages one token per four.
//!
//! ```
//! use openjiuwen_harness::prompts::{HarnessPromptBuilder, PromptMode, PromptSection};
//!
//! let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
//! builder
//!     .add_section(PromptSection::bilingual("identity", "身份", "hello", 10))
//!     .expect("first registration");
//! let report = builder.build_report();
//! assert_eq!(report.total_chars, 5);
//! assert!(report.summary().contains("est_tokens"));
//! ```

use super::builder::{is_chinese, PromptCategory, PromptMode, PromptSection};

/// One reported section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionInfo {
    /// Section name.
    pub name: String,
    /// Render priority of the section.
    pub priority: i32,
    /// Rendered length in characters.
    pub char_count: usize,
    /// Category the section was reported under (never `None`; defaults to
    /// [`PromptCategory::SystemPrompt`]).
    pub category: PromptCategory,
}

/// The sections of one category, with their aggregate size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoryInfo {
    /// Category of the group.
    pub category: PromptCategory,
    /// Human-readable category name in the report language.
    pub display_name: String,
    /// Sections of this category, in ascending priority order.
    pub sections: Vec<SectionInfo>,
    /// Summed character count of [`Self::sections`].
    pub char_count: usize,
    /// Estimated tokens of [`Self::char_count`].
    pub estimated_tokens: usize,
}

/// Size and token accounting of a generated prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptReport {
    /// Summed character count over every reported section.
    pub total_chars: usize,
    /// Estimated tokens of [`Self::total_chars`].
    pub estimated_tokens: usize,
    /// Number of reported sections.
    pub section_count: usize,
    /// Every reported section, grouped by category in [`PromptCategory::ORDER`].
    pub sections: Vec<SectionInfo>,
    /// Prompt mode the report was built for.
    pub mode: PromptMode,
    /// Language the report measured.
    pub language: String,
    /// Non-empty categories, in [`PromptCategory::ORDER`].
    pub categories: Vec<CategoryInfo>,
}

impl PromptReport {
    /// Builds a report over `sections` for `mode` and `language`.
    ///
    /// Sections without a category count as [`PromptCategory::SystemPrompt`]; empty
    /// categories are omitted, and sections are sorted by priority inside their group.
    #[must_use]
    pub fn from_sections(mode: PromptMode, language: &str, sections: &[&PromptSection]) -> Self {
        let mut reported: Vec<SectionInfo> = Vec::with_capacity(sections.len());
        let mut categories: Vec<CategoryInfo> = Vec::new();

        for category in PromptCategory::ORDER {
            let mut group: Vec<&PromptSection> = sections
                .iter()
                .filter(|section| {
                    section.category.unwrap_or(PromptCategory::SystemPrompt) == category
                })
                .copied()
                .collect();
            group.sort_by_key(|section| section.priority);
            if group.is_empty() {
                continue;
            }
            let infos: Vec<SectionInfo> = group
                .into_iter()
                .map(|section| SectionInfo {
                    name: section.name.clone(),
                    priority: section.priority,
                    char_count: section.char_count(language),
                    category,
                })
                .collect();
            let char_count: usize = infos.iter().map(|info| info.char_count).sum();
            reported.extend(infos.iter().cloned());
            categories.push(CategoryInfo {
                category,
                display_name: category.display_name(language).to_string(),
                sections: infos,
                char_count,
                estimated_tokens: estimate_tokens(char_count, language),
            });
        }

        let total_chars: usize = reported.iter().map(|info| info.char_count).sum();
        Self {
            total_chars,
            estimated_tokens: estimate_tokens(total_chars, language),
            section_count: reported.len(),
            sections: reported,
            mode,
            language: language.to_string(),
            categories,
        }
    }

    /// Returns the characters-per-token ratio assumed for `language`.
    ///
    /// `2.5` for Chinese, `4.0` otherwise. [`Self::estimated_tokens`] uses the exact
    /// integer equivalents (`chars * 2 / 5` and `chars / 4`) to stay free of casts.
    #[must_use]
    pub fn chars_per_token(language: &str) -> f64 {
        if is_chinese(language) {
            2.5
        } else {
            4.0
        }
    }

    /// Returns a one-line human-readable summary of the report.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "[PromptReport] mode={} lang={} sections={} chars={} est_tokens≈{}",
            self.mode.as_str(),
            self.language,
            self.section_count,
            self.total_chars,
            self.estimated_tokens
        )
    }
}

/// Estimates the token count of `chars` characters of `language`.
///
/// Integer form of `chars / chars_per_token(language)`; truncating instead of rounding.
fn estimate_tokens(chars: usize, language: &str) -> usize {
    if is_chinese(language) {
        chars.saturating_mul(2) / 5
    } else {
        chars / 4
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use super::{PromptCategory, PromptMode, PromptReport, PromptSection};

    fn section(name: &str, cn: &str, en: &str, priority: i32) -> PromptSection {
        PromptSection::bilingual(name, cn, en, priority)
    }

    #[test]
    fn categories_follow_the_fixed_order() {
        let system = section("identity", "身份", "hi", 10);
        let tools = section("tools", "工具", "tools", 30).with_category(PromptCategory::Tools);
        let memory = section("memory", "记忆", "mem", 5).with_category(PromptCategory::Memory);
        let own = section("custom", "自定义", "custom", 1);

        let sections: Vec<&PromptSection> = vec![&memory, &tools, &own, &system];
        let report = PromptReport::from_sections(PromptMode::Full, "en", &sections);

        let names: Vec<&str> = report
            .categories
            .iter()
            .map(|group| group.category.as_str())
            .collect();
        assert_eq!(names, ["system_prompt", "tools", "memory"]);
        assert_eq!(report.categories[0].display_name, "System Prompt");
        assert_eq!(
            report.categories[0]
                .sections
                .iter()
                .map(|inner| inner.name.as_str())
                .collect::<Vec<_>>(),
            ["custom", "identity"]
        );
        assert_eq!(
            report
                .sections
                .iter()
                .map(|inner| inner.name.as_str())
                .collect::<Vec<_>>(),
            ["custom", "identity", "tools", "memory"]
        );
        assert_eq!(report.section_count, 4);
    }

    #[test]
    fn empty_categories_are_omitted() {
        let only_tools = section("tools", "工具", "tools", 30).with_category(PromptCategory::Tools);
        let report = PromptReport::from_sections(PromptMode::Full, "en", &[&only_tools]);
        assert_eq!(report.categories.len(), 1);
        assert_eq!(report.categories[0].category, PromptCategory::Tools);
        assert!(report.categories[0]
            .sections
            .iter()
            .all(|s| s.category == PromptCategory::Tools));
    }

    #[test]
    fn uncategorised_sections_count_as_system_prompt() {
        let plain = section("plain", "普通", "plain", 10);
        let report = PromptReport::from_sections(PromptMode::Minimal, "en", &[&plain]);
        assert_eq!(report.categories.len(), 1);
        assert_eq!(report.categories[0].category, PromptCategory::SystemPrompt);
        assert_eq!(
            report.categories[0].sections[0].category,
            PromptCategory::SystemPrompt
        );
        assert_eq!(report.mode, PromptMode::Minimal);
    }

    #[test]
    fn token_estimate_uses_language_divisor() {
        // 10 Chinese characters -> 4 tokens (10 * 2 / 5); 10 Latin characters -> 2.
        let cn = section("cn", &"字".repeat(10), "x", 1);
        let report = PromptReport::from_sections(PromptMode::Full, "cn", &[&cn]);
        assert_eq!(report.total_chars, 10);
        assert_eq!(report.estimated_tokens, 4);
        assert!((PromptReport::chars_per_token("cn") - 2.5).abs() < f64::EPSILON);

        let en = section("en", "x", "abcdefghij", 1);
        let report = PromptReport::from_sections(PromptMode::Full, "en", &[&en]);
        assert_eq!(report.total_chars, 10);
        assert_eq!(report.estimated_tokens, 2);
        assert!((PromptReport::chars_per_token("en") - 4.0).abs() < f64::EPSILON);
        assert!((PromptReport::chars_per_token("fr") - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn group_tokens_and_total_tokens_are_consistent() {
        let a = section("a", "a", "abcd", 1);
        let b = section("b", "b", "efgh", 2).with_category(PromptCategory::Tools);
        let report = PromptReport::from_sections(PromptMode::Full, "en", &[&a, &b]);
        assert_eq!(report.total_chars, 8);
        assert_eq!(report.estimated_tokens, 2);
        let summed: usize = report.categories.iter().map(|c| c.estimated_tokens).sum();
        assert_eq!(summed, 2);
        for group in &report.categories {
            assert_eq!(
                group.char_count,
                group.sections.iter().map(|s| s.char_count).sum::<usize>()
            );
        }
    }

    #[test]
    fn empty_input_yields_an_empty_report() {
        let report = PromptReport::from_sections(PromptMode::None, "en", &[]);
        assert_eq!(report.section_count, 0);
        assert_eq!(report.total_chars, 0);
        assert_eq!(report.estimated_tokens, 0);
        assert_eq!(report.sections.len(), 0);
        assert_eq!(report.categories.len(), 0);
        assert_eq!(
            report.summary(),
            "[PromptReport] mode=none lang=en sections=0 chars=0 est_tokens≈0"
        );
    }

    #[test]
    fn summary_format_is_stable() {
        let s = section("a", "a", "abcd", 1);
        let report = PromptReport::from_sections(PromptMode::Full, "en", &[&s]);
        assert_eq!(
            report.summary(),
            "[PromptReport] mode=full lang=en sections=1 chars=4 est_tokens≈1"
        );
    }

    #[test]
    fn section_without_content_is_reported_as_zero_chars() {
        let empty = PromptSection::new("empty", BTreeMap::new(), 1);
        let report = PromptReport::from_sections(PromptMode::Full, "en", &[&empty]);
        assert_eq!(report.section_count, 1);
        assert_eq!(report.total_chars, 0);
        assert_eq!(report.estimated_tokens, 0);
    }
}
