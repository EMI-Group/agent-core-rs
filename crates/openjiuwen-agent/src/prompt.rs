//! System prompt sections and their ordered assembly.
//!
//! [`SystemPromptBuilder`] collects named [`PromptSection`]s and renders them in
//! ascending priority; the identity section uses [`IDENTITY_PRIORITY`], matching
//! Python's prompt builder ordering.
use serde::{Deserialize, Serialize};
/// Priority of the agent identity section, rendered before ordinary sections.
pub const IDENTITY_PRIORITY: i32 = 10;
/// One named block of the system prompt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptSection {
    /// Unique section name; re-adding a name replaces the section in place.
    pub name: String,
    /// Rendered content; empty content is skipped by
    /// [`SystemPromptBuilder::build`].
    pub content: String,
    /// Ordering key; lower priorities render first.
    pub priority: i32,
    /// Optional grouping label for hosts that render sections by category.
    pub category: Option<String>,
}
impl PromptSection {
    /// Builds a section.
    #[must_use]
    pub fn new(name: impl Into<String>, content: impl Into<String>, priority: i32) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            priority,
            category: None,
        }
    }
    /// Sets the category.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}
/// An ordered collection of [`PromptSection`]s rendered into a system prompt.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemPromptBuilder {
    sections: Vec<PromptSection>,
}
impl SystemPromptBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Adds `section`, replacing an existing section with the same name.
    pub fn add_section(&mut self, section: PromptSection) {
        match self
            .sections
            .iter()
            .position(|existing| existing.name == section.name)
        {
            Some(index) => self.sections[index] = section,
            None => self.sections.push(section),
        }
    }
    /// Adds or replaces a section by name.
    pub fn set_section(
        &mut self,
        name: impl Into<String>,
        content: impl Into<String>,
        priority: i32,
    ) {
        self.add_section(PromptSection::new(name, content, priority));
    }
    /// Removes and returns the section named `name`.
    pub fn remove_section(&mut self, name: &str) -> Option<PromptSection> {
        self.sections
            .iter()
            .position(|section| section.name == name)
            .map(|index| self.sections.remove(index))
    }
    /// Returns the section named `name`.
    #[must_use]
    pub fn section(&self, name: &str) -> Option<&PromptSection> {
        self.sections.iter().find(|section| section.name == name)
    }
    /// Returns the sections in insertion order.
    #[must_use]
    pub fn sections(&self) -> &[PromptSection] {
        &self.sections
    }
    /// Returns `true` when no section has been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }
    /// Renders the non-empty sections in ascending priority, joined by two
    /// newlines; ties keep insertion order.
    #[must_use]
    pub fn build(&self) -> String {
        let mut ordered: Vec<&PromptSection> = self
            .sections
            .iter()
            .filter(|section| !section.content.is_empty())
            .collect();
        ordered.sort_by_key(|section| section.priority);
        ordered
            .iter()
            .map(|section| section.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn builder() -> SystemPromptBuilder {
        let mut builder = SystemPromptBuilder::new();
        builder.set_section("identity", "you are an agent", IDENTITY_PRIORITY);
        builder.set_section("tools", "tools go here", 20);
        builder.set_section("rules", "be nice", 5);
        builder
    }
    #[test]
    fn renders_ascending_priority() {
        assert_eq!(
            builder().build(),
            "be nice\n\nyou are an agent\n\ntools go here"
        );
    }
    #[test]
    fn ties_keep_insertion_order() {
        let mut builder = SystemPromptBuilder::new();
        builder.set_section("first", "a", 1);
        builder.set_section("second", "b", 1);
        builder.set_section("third", "c", 1);
        assert_eq!(builder.build(), "a\n\nb\n\nc");
    }
    #[test]
    fn add_section_replaces_by_name() {
        let mut builder = builder();
        builder.set_section("tools", "new tools", 20);
        assert_eq!(builder.sections().len(), 3);
        assert_eq!(builder.section("tools").unwrap().content, "new tools");
        assert_eq!(builder.build(), "be nice\n\nyou are an agent\n\nnew tools");
    }
    #[test]
    fn remove_section_returns_the_removed_section() {
        let mut builder = builder();
        let removed = builder.remove_section("rules").unwrap();
        assert_eq!(removed.content, "be nice");
        assert!(builder.section("rules").is_none());
        assert!(builder.remove_section("rules").is_none());
        assert_eq!(builder.sections().len(), 2);
    }
    #[test]
    fn empty_content_is_skipped() {
        let mut builder = SystemPromptBuilder::new();
        builder.set_section("a", "alpha", 1);
        builder.set_section("empty", "", 2);
        builder.set_section("z", "omega", 3);
        assert_eq!(builder.build(), "alpha\n\nomega");
    }
    #[test]
    fn empty_builder_renders_empty_string() {
        let builder = SystemPromptBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.build(), "");
    }
    #[test]
    fn category_builder_sets_category() {
        let section = PromptSection::new("a", "b", 1).with_category("identity");
        assert_eq!(section.category.as_deref(), Some("identity"));
    }
    #[test]
    fn section_round_trips_through_json() {
        let section = PromptSection::new("a", "b", 3).with_category("cat");
        let value = serde_json::to_value(&section).unwrap();
        let restored: PromptSection = serde_json::from_value(value).unwrap();
        assert_eq!(restored, section);
    }
}
