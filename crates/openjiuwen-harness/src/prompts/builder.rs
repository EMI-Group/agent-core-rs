//! Ordered, bilingual system-prompt sections and the builder that renders them.
//!
//! Ports the builder half of `openjiuwen/core/single_agent/prompts/builder.py` and
//! `openjiuwen/harness/prompts/` (section registry, mode filtering, language
//! resolution). A [`PromptSection`] carries its text in a language-keyed map, so a
//! single section serves both the Chinese and the English prompt; the builder only
//! decides ordering, filtering and joining.
//!
//! Everything here is pure data assembly: no IO, no async, no filesystem access.
//! [`HarnessPromptBuilder::build`] is idempotent, and ties between sections with the
//! same priority are always broken by insertion order, so the same builder contents
//! always render identically.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};

pub use crate::schema::PromptMode;

use super::report::PromptReport;

/// Language used when a caller supplies none or an unsupported one.
pub const DEFAULT_LANGUAGE: &str = "cn";

/// Languages a prompt can be rendered in.
pub const SUPPORTED_LANGUAGES: [&str; 2] = ["cn", "en"];

/// Sections kept by [`PromptMode::Minimal`], in no particular order.
pub const MINIMAL_SECTIONS: &[&str] = &[
    SectionName::IDENTITY,
    SectionName::SAFETY,
    SectionName::SKILLS,
    SectionName::TOOLS,
    SectionName::MEMORY,
];

/// Names of the standard prompt sections.
///
/// A namespace rather than an enum: section names are open-ended strings (a caller may
/// register its own), these constants only name the sections the harness itself adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectionName;

impl SectionName {
    /// Agent identity and general working rules.
    pub const IDENTITY: &'static str = "identity";
    /// Safety and authorisation rules.
    pub const SAFETY: &'static str = "safety";
    /// Tool usage rules plus the model-facing tool list.
    pub const TOOLS: &'static str = "tools";
    /// Available skills and how to use them.
    pub const SKILLS: &'static str = "skills";
    /// Long-term memory guidance.
    pub const MEMORY: &'static str = "memory";
    /// Working directory and the well-known files inside it.
    pub const WORKSPACE: &'static str = "workspace";
    /// Contents of project files preloaded into the prompt.
    pub const CONTEXT: &'static str = "context";
    /// Rules for delegating work to ephemeral subagents.
    pub const TASK_LOOP: &'static str = "task_loop";
    /// Rules for managing persistent subagents.
    pub const SUBAGENT_TOOLS: &'static str = "subagent_tools";
    /// The completion-promise marker instructions.
    pub const COMPLETION_SIGNAL: &'static str = "completion_signal";
    /// Mode-specific instructions (plan mode).
    pub const MODE_INSTRUCTIONS: &'static str = "mode_instructions";
    /// The current todo plan.
    pub const TODO: &'static str = "todo";
    /// Working directory, model, date and git branch.
    pub const ENVIRONMENT: &'static str = "environment";
}

/// Grouping of sections used by the prompt report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PromptCategory {
    /// The system prompt proper (identity, safety, rules, environment).
    SystemPrompt,
    /// Tool usage rules and the tool list.
    Tools,
    /// Skill list.
    Skills,
    /// Memory and project-context material.
    Memory,
}

impl PromptCategory {
    /// Categories in the fixed order used by [`PromptReport`].
    pub const ORDER: [Self; 4] = [Self::SystemPrompt, Self::Tools, Self::Skills, Self::Memory];

    /// Returns the snake-case name of the category (`"system_prompt"`, `"tools"`, …).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SystemPrompt => "system_prompt",
            Self::Tools => "tools",
            Self::Skills => "skills",
            Self::Memory => "memory",
        }
    }

    /// Parses an exact snake-case name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "system_prompt" => Some(Self::SystemPrompt),
            "tools" => Some(Self::Tools),
            "skills" => Some(Self::Skills),
            "memory" => Some(Self::Memory),
            _ => None,
        }
    }

    /// Returns the human-readable name of the category in `language`.
    #[must_use]
    pub fn display_name(&self, language: &str) -> &'static str {
        if language_key(language) == "en" {
            match self {
                Self::SystemPrompt => "System Prompt",
                Self::Tools => "Tools",
                Self::Skills => "Skills",
                Self::Memory => "Memory",
            }
        } else {
            match self {
                Self::SystemPrompt => "系统提示",
                Self::Tools => "工具",
                Self::Skills => "技能",
                Self::Memory => "记忆",
            }
        }
    }
}

/// One named, bilingual block of the system prompt.
///
/// `content` maps a language key (`"cn"`, `"en"`) to the rendered text; a section may
/// carry either or both. `priority` orders sections in the built prompt (ascending),
/// `category` only drives report grouping, and `carrier` names the message channel the
/// section travels on (`"system"` unless a caller overrides it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSection {
    /// Section name; unique within a builder and used for upserts.
    pub name: String,
    /// Rendered text per language key.
    pub content: BTreeMap<String, String>,
    /// Render order; lower comes first.
    pub priority: i32,
    /// Report grouping; `None` counts as [`PromptCategory::SystemPrompt`].
    pub category: Option<PromptCategory>,
    /// Message channel carrying the section.
    pub carrier: String,
}

impl PromptSection {
    /// Builds a section from explicit per-language content.
    #[must_use]
    pub fn new(name: impl Into<String>, content: BTreeMap<String, String>, priority: i32) -> Self {
        Self {
            name: name.into(),
            content,
            priority,
            category: None,
            carrier: "system".to_string(),
        }
    }

    /// Builds a section carrying exactly the Chinese and English texts.
    #[must_use]
    pub fn bilingual(
        name: impl Into<String>,
        cn: impl Into<String>,
        en: impl Into<String>,
        priority: i32,
    ) -> Self {
        Self::new(name, bilingual_content(cn, en), priority)
    }

    /// Sets the report category.
    #[must_use]
    pub fn with_category(mut self, category: PromptCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Sets the message channel carrying this section.
    #[must_use]
    pub fn with_carrier(mut self, carrier: impl Into<String>) -> Self {
        self.carrier = carrier.into();
        self
    }

    /// Renders the section for `language`.
    ///
    /// Falls back from the requested language to [`DEFAULT_LANGUAGE`], then to the
    /// first value in key order, then to the empty string — never panics, never fails.
    #[must_use]
    pub fn render(&self, language: &str) -> String {
        self.content
            .get(language)
            .or_else(|| self.content.get(DEFAULT_LANGUAGE))
            .or_else(|| self.content.values().next())
            .cloned()
            .unwrap_or_default()
    }

    /// Returns the number of characters [`Self::render`] produces for `language`.
    #[must_use]
    pub fn char_count(&self, language: &str) -> usize {
        self.render(language).chars().count()
    }

    /// Returns whether the section renders to nothing but whitespace in `language`.
    #[must_use]
    pub fn is_empty(&self, language: &str) -> bool {
        self.render(language).trim().is_empty()
    }
}

/// Accumulates [`PromptSection`]s and renders the system prompt.
///
/// Sections are stored in insertion order; [`Self::build`] sorts by priority with ties
/// broken by that order, so rendering is deterministic. Adding a section with a name
/// that already exists replaces its text in place but never its priority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessPromptBuilder {
    language: String,
    mode: PromptMode,
    sections: Vec<PromptSection>,
}

impl HarnessPromptBuilder {
    /// Builds an empty builder rendering `language` in `mode`.
    ///
    /// The language is normalised with [`resolve_language`], so `"en-US"` becomes
    /// `"en"` and anything unrecognised becomes [`DEFAULT_LANGUAGE`].
    #[must_use]
    pub fn new(language: impl AsRef<str>, mode: PromptMode) -> Self {
        Self {
            language: resolve_language(Some(language.as_ref())),
            mode,
            sections: Vec::new(),
        }
    }

    /// Returns the language the builder renders in.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the prompt mode the builder renders with.
    #[must_use]
    pub fn mode(&self) -> PromptMode {
        self.mode
    }

    /// Returns this builder rendering in `mode`.
    #[must_use]
    pub fn with_mode(mut self, mode: PromptMode) -> Self {
        self.mode = mode;
        self
    }

    /// Adds a section, or replaces the text of the section with the same name.
    ///
    /// A replacement keeps the original position in insertion order and must keep the
    /// original priority.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a section with the same name is already
    /// registered with a different priority.
    pub fn add_section(&mut self, section: PromptSection) -> Result<()> {
        let name = section.name.clone();
        if let Some(existing) = self.sections.iter_mut().find(|s| s.name == name) {
            if existing.priority != section.priority {
                return Err(Error::validation(format!(
                    "prompt section {name:?} is already registered with priority {}; cannot replace it with priority {}",
                    existing.priority, section.priority
                )));
            }
            *existing = section;
            tracing::debug!(section = %name, "replaced prompt section");
            return Ok(());
        }
        self.sections.push(section);
        tracing::debug!(section = %name, "added prompt section");
        Ok(())
    }

    /// Removes and returns the section called `name`, if any.
    pub fn remove_section(&mut self, name: &str) -> Option<PromptSection> {
        let index = self.sections.iter().position(|s| s.name == name)?;
        let removed = self.sections.remove(index);
        tracing::debug!(section = %name, "removed prompt section");
        Some(removed)
    }

    /// Returns the section called `name`, if any.
    #[must_use]
    pub fn get_section(&self, name: &str) -> Option<&PromptSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Returns whether a section called `name` is registered.
    #[must_use]
    pub fn has_section(&self, name: &str) -> bool {
        self.get_section(name).is_some()
    }

    /// Returns every registered section in insertion order.
    #[must_use]
    pub fn sections(&self) -> Vec<&PromptSection> {
        self.sections.iter().collect()
    }

    /// Renders the system prompt.
    ///
    /// [`PromptMode::None`] returns the identity section's text unchanged (empty when no
    /// identity section is registered). Otherwise sections are filtered by mode and by
    /// emptiness, sorted by priority (insertion order breaking ties) and joined with a
    /// blank line.
    #[must_use]
    pub fn build(&self) -> String {
        if self.mode == PromptMode::None {
            return self
                .get_section(SectionName::IDENTITY)
                .map_or_else(String::new, |section| section.render(&self.language));
        }
        self.visible_sections()
            .into_iter()
            .map(|section| section.render(&self.language))
            .filter(|part| !part.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Builds the size/token report of the prompt [`Self::build`] would produce.
    #[must_use]
    pub fn build_report(&self) -> PromptReport {
        PromptReport::from_sections(self.mode, &self.language, &self.visible_sections())
    }

    /// Returns the sections actually rendered, in render order.
    ///
    /// [`PromptMode::None`] yields nothing (nothing is rendered); [`PromptMode::Minimal`]
    /// keeps only [`MINIMAL_SECTIONS`]; empty sections are dropped in every mode.
    fn visible_sections(&self) -> Vec<&PromptSection> {
        if self.mode == PromptMode::None {
            return Vec::new();
        }
        let mut sections: Vec<&PromptSection> = self
            .sections
            .iter()
            .filter(|section| {
                self.mode == PromptMode::Full || MINIMAL_SECTIONS.contains(&section.name.as_str())
            })
            .filter(|section| !section.is_empty(&self.language))
            .collect();
        sections.sort_by_key(|section| section.priority);
        sections
    }
}

impl Default for HarnessPromptBuilder {
    /// Chinese language, [`PromptMode::Full`], no sections.
    fn default() -> Self {
        Self::new(DEFAULT_LANGUAGE, PromptMode::Full)
    }
}

/// Resolves a configured language to a supported one.
///
/// An absent, blank or unsupported value falls back to [`DEFAULT_LANGUAGE`]; `en` and
/// `en-*` resolve to `en`; `cn`, `zh` and `zh-*` resolve to `cn`.
#[must_use]
pub fn resolve_language(input: Option<&str>) -> String {
    input.map_or_else(
        || DEFAULT_LANGUAGE.to_string(),
        |raw| language_key(raw).to_string(),
    )
}

/// Normalises a language tag to `"cn"` or `"en"`.
pub(crate) fn language_key(raw: &str) -> &'static str {
    let normalised = raw.trim().to_ascii_lowercase();
    if normalised.starts_with("en") {
        SUPPORTED_LANGUAGES[1]
    } else {
        DEFAULT_LANGUAGE
    }
}

/// Returns whether `language` names Chinese (`cn`, `zh`, `zh-*`).
///
/// Unlike [`language_key`], an unknown language is *not* folded into Chinese: the token
/// heuristic treats an unrecognised language like any other non-Chinese script.
pub(crate) fn is_chinese(language: &str) -> bool {
    let normalised = language.trim().to_ascii_lowercase();
    normalised == "cn" || normalised.starts_with("zh")
}

/// Builds the per-language content map shared by every section.
pub(crate) fn bilingual_content(
    cn: impl Into<String>,
    en: impl Into<String>,
) -> BTreeMap<String, String> {
    let mut content = BTreeMap::new();
    content.insert("cn".to_string(), cn.into());
    content.insert("en".to_string(), en.into());
    content
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        bilingual_content, resolve_language, HarnessPromptBuilder, PromptCategory, PromptMode,
        PromptSection, SectionName, DEFAULT_LANGUAGE, MINIMAL_SECTIONS, SUPPORTED_LANGUAGES,
    };

    fn section(name: &str, cn: &str, en: &str, priority: i32) -> PromptSection {
        PromptSection::bilingual(name, cn, en, priority)
    }

    #[test]
    fn language_resolution_falls_back_to_chinese() {
        assert_eq!(resolve_language(None), "cn");
        assert_eq!(resolve_language(Some("")), "cn");
        assert_eq!(resolve_language(Some("  ")), "cn");
        assert_eq!(resolve_language(Some("cn")), "cn");
        assert_eq!(resolve_language(Some("zh")), "cn");
        assert_eq!(resolve_language(Some("zh-CN")), "cn");
        assert_eq!(resolve_language(Some("en")), "en");
        assert_eq!(resolve_language(Some("EN")), "en");
        assert_eq!(resolve_language(Some("en-US")), "en");
        assert_eq!(resolve_language(Some("klingon")), DEFAULT_LANGUAGE);
        assert_eq!(DEFAULT_LANGUAGE, "cn");
        assert_eq!(SUPPORTED_LANGUAGES, ["cn", "en"]);
    }

    #[test]
    fn builder_default_is_chinese_full() {
        let builder = HarnessPromptBuilder::default();
        assert_eq!(builder.language(), "cn");
        assert_eq!(builder.mode(), PromptMode::Full);
        assert_eq!(builder.sections().len(), 0);
        assert_eq!(builder.build(), "");
    }

    #[test]
    fn builder_normalises_language() {
        let builder = HarnessPromptBuilder::new("en-GB", PromptMode::Full);
        assert_eq!(builder.language(), "en");
        let builder = HarnessPromptBuilder::new("bogus", PromptMode::Full);
        assert_eq!(builder.language(), DEFAULT_LANGUAGE);
    }

    #[test]
    fn section_render_falls_back_through_cn_then_first_value() {
        let both = section("both", "你好", "hello", 1);
        assert_eq!(both.render("en"), "hello");
        assert_eq!(both.render("cn"), "你好");
        assert_eq!(both.render("fr"), "你好");

        let mut only_en = PromptSection::new("only_en", bilingual_content("", "hi"), 2);
        only_en.content.remove("cn");
        assert_eq!(only_en.render("fr"), "hi");

        let empty = PromptSection::new("empty", std::collections::BTreeMap::new(), 3);
        assert_eq!(empty.render("cn"), "");
        assert_eq!(empty.char_count("cn"), 0);
        assert!(empty.is_empty("cn"));
    }

    #[test]
    fn char_count_counts_characters_not_bytes() {
        let s = section("identity", "你好世界", "hello", 1);
        assert_eq!(s.char_count("cn"), 4);
        assert_eq!(s.char_count("en"), 5);
        assert!(!s.is_empty("en"));
        assert!(section("blank", "  \n", " ", 1).is_empty("cn"));
        assert!(section("blank", "  \n", " ", 1).is_empty("en"));
    }

    #[test]
    fn build_sorts_by_priority_and_joins_with_blank_line() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder
            .add_section(section("second", "二", "second", 20))
            .unwrap();
        builder
            .add_section(section("first", "一", "first", 10))
            .unwrap();
        assert_eq!(builder.build(), "first\n\nsecond");

        let mut builder = HarnessPromptBuilder::new("cn", PromptMode::Full);
        builder
            .add_section(section("second", "二", "second", 20))
            .unwrap();
        builder
            .add_section(section("first", "一", "first", 10))
            .unwrap();
        assert_eq!(builder.build(), "一\n\n二");
    }

    #[test]
    fn build_breaks_priority_ties_by_insertion_order() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        for name in ["alpha", "beta", "gamma"] {
            builder.add_section(section(name, name, name, 10)).unwrap();
        }
        assert_eq!(builder.build(), "alpha\n\nbeta\n\ngamma");
    }

    #[test]
    fn build_drops_blank_sections() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder.add_section(section("blank", " ", "\n", 1)).unwrap();
        builder
            .add_section(section("real", "实", "real", 2))
            .unwrap();
        assert_eq!(builder.build(), "real");
        assert_eq!(builder.sections().len(), 2);
    }

    #[test]
    fn add_section_upserts_in_place() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder.add_section(section("a", "a", "a", 1)).unwrap();
        builder.add_section(section("b", "b", "b", 2)).unwrap();
        builder.add_section(section("a", "a2", "a2", 1)).unwrap();

        let names: Vec<&str> = builder.sections().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(
            builder.get_section("a").unwrap().render("cn"),
            "a2".to_string()
        );
        assert!(builder.has_section("a"));
        assert!(!builder.has_section("missing"));
        assert_eq!(builder.build(), "a2\n\nb");
    }

    #[test]
    fn add_section_rejects_priority_change() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder.add_section(section("a", "a", "a", 1)).unwrap();
        let error = builder
            .add_section(section("a", "a", "a", 5))
            .expect_err("priority change must be rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        let text = error.to_string();
        assert!(text.contains("\"a\""), "message names the section: {text}");
        assert!(
            text.contains('1') && text.contains('5'),
            "message names both priorities: {text}"
        );
        assert_eq!(builder.get_section("a").unwrap().priority, 1);
    }

    #[test]
    fn remove_section_returns_and_forgets_it() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder.add_section(section("a", "a", "a", 1)).unwrap();
        let removed = builder.remove_section("a").expect("section present");
        assert_eq!(removed.name, "a");
        assert!(builder.remove_section("a").is_none());
        assert_eq!(builder.sections().len(), 0);
    }

    #[test]
    fn minimal_mode_keeps_only_the_minimal_sections() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Minimal);
        for (index, name) in MINIMAL_SECTIONS.iter().enumerate() {
            let priority = i32::try_from(index).unwrap() * 10;
            builder
                .add_section(section(name, name, name, priority))
                .unwrap();
        }
        builder
            .add_section(section(SectionName::TODO, "t", "todo", 100))
            .unwrap();
        builder
            .add_section(section(SectionName::ENVIRONMENT, "e", "env", 110))
            .unwrap();

        let built = builder.build();
        assert_eq!(built, MINIMAL_SECTIONS.join("\n\n"));
        assert!(!built.contains("todo"));
        assert!(!built.contains("env"));
        // Switching the mode back to Full exposes the extra sections.
        assert!(builder.with_mode(PromptMode::Full).build().contains("todo"));
    }

    #[test]
    fn none_mode_renders_the_identity_section_raw() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::None);
        builder
            .add_section(section(SectionName::SAFETY, "安全", "safety", 20))
            .unwrap();
        assert_eq!(builder.build(), "");
        builder
            .add_section(section(
                SectionName::IDENTITY,
                "身份",
                "  identity text  ",
                10,
            ))
            .unwrap();
        assert_eq!(builder.build(), "  identity text  ");
    }

    #[test]
    fn report_tracks_the_rendered_prompt() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder
            .add_section(
                section(SectionName::IDENTITY, "身份", "hello", 10)
                    .with_category(PromptCategory::SystemPrompt),
            )
            .unwrap();
        builder
            .add_section(
                section(SectionName::TOOLS, "工具", "abcdefgh", 30)
                    .with_category(PromptCategory::Tools),
            )
            .unwrap();
        builder.add_section(section("blank", " ", "  ", 1)).unwrap();

        let report = builder.build_report();
        assert_eq!(report.mode, PromptMode::Full);
        assert_eq!(report.language, "en");
        assert_eq!(report.section_count, 2);
        assert_eq!(report.total_chars, 5 + 8);
        assert_eq!(report.estimated_tokens, 13 / 4);
        assert_eq!(report.categories.len(), 2);
        assert_eq!(report.categories[0].category, PromptCategory::SystemPrompt);
        assert_eq!(report.categories[1].category, PromptCategory::Tools);
        assert!(report
            .summary()
            .starts_with("[PromptReport] mode=full lang=en"));
    }

    #[test]
    fn report_of_none_mode_is_empty() {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::None);
        builder
            .add_section(section(SectionName::IDENTITY, "身份", "hello", 10))
            .unwrap();
        let report = builder.build_report();
        assert_eq!(report.section_count, 0);
        assert_eq!(report.total_chars, 0);
        assert_eq!(report.categories.len(), 0);
    }

    #[test]
    fn category_names_round_trip() {
        for category in PromptCategory::ORDER {
            assert_eq!(PromptCategory::parse(category.as_str()), Some(category));
        }
        assert_eq!(PromptCategory::parse("System_Prompt"), None);
        assert_eq!(PromptCategory::parse(""), None);
        assert_eq!(PromptCategory::SystemPrompt.display_name("cn"), "系统提示");
        assert_eq!(
            PromptCategory::SystemPrompt.display_name("en"),
            "System Prompt"
        );
        assert_eq!(PromptCategory::Tools.display_name("cn"), "工具");
        assert_eq!(PromptCategory::Skills.display_name("en"), "Skills");
        assert_eq!(PromptCategory::Memory.display_name("zh-CN"), "记忆");
        assert_eq!(
            PromptCategory::ORDER.map(|category| category.as_str()),
            ["system_prompt", "tools", "skills", "memory"]
        );
    }

    #[test]
    fn section_builders_set_carrier_and_category() {
        let s = section("x", "x", "x", 1)
            .with_category(PromptCategory::Memory)
            .with_carrier("user");
        assert_eq!(s.category, Some(PromptCategory::Memory));
        assert_eq!(s.carrier, "user");
        assert_eq!(section("y", "y", "y", 1).carrier, "system");
        assert_eq!(section("y", "y", "y", 1).category, None);
    }
}
