//! The prompt-sections rail: render a pack's sections into one system message.
//!
//! A pack's `prompts.sections` are extra system instructions that must reach the
//! model on every call without rewriting the agent's identity prompt. On every
//! `BeforeModelCall` the rail renders them into a single `system` message and
//! inserts it at the front of the prepared window, marked with
//! [`PROMPT_SECTIONS_METADATA_KEY`] so a later iteration never injects it twice.
use super::pack::SectionSpec;
use crate::rails::{AgentRail, RailContext, DEFAULT_RAIL_PRIORITY};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::BaseMessage;
use serde_json::Value;

/// Metadata key marking the system message that carries the pack sections.
///
/// A message with this key in its metadata is never injected a second time, so a
/// later `BeforeModelCall` of the same invocation leaves the window untouched.
pub const PROMPT_SECTIONS_METADATA_KEY: &str = "_openjiuwen_prompt_sections";

/// Rail name [`PromptSectionsRail`] registers under.
pub const PROMPT_SECTIONS_RAIL_NAME: &str = "prompt_sections";

/// Injects the pack's extra system sections into every model request.
///
/// Sections render as `## <name>` followed by the content, in ascending
/// priority; ties keep their pack order. Sections with a blank name or empty
/// content are skipped, and a rail with nothing left to render leaves the window
/// untouched: the rail never fails the run.
#[derive(Debug, Clone)]
pub struct PromptSectionsRail {
    sections: Vec<SectionSpec>,
    priority: i32,
}

impl Default for PromptSectionsRail {
    /// A rail with no sections and the default dispatch priority.
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl PromptSectionsRail {
    /// Builds a rail over `sections` with the default dispatch priority.
    #[must_use]
    pub fn new(sections: impl IntoIterator<Item = SectionSpec>) -> Self {
        Self {
            sections: sections.into_iter().collect(),
            priority: DEFAULT_RAIL_PRIORITY,
        }
    }

    /// Sets the rail dispatch priority (higher runs first).
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns the sections the rail renders.
    #[must_use]
    pub fn sections(&self) -> &[SectionSpec] {
        &self.sections
    }

    /// Renders the sections in ascending priority, joined by blank lines.
    ///
    /// Each rendered section is `## <name>\n<content>`; sections with a blank
    /// name or empty content contribute nothing.
    #[must_use]
    pub fn render(&self) -> String {
        let mut ordered: Vec<&SectionSpec> = self
            .sections
            .iter()
            .filter(|section| !section.name.trim().is_empty() && !section.content.trim().is_empty())
            .collect();
        ordered.sort_by_key(|section| section.priority);
        ordered
            .iter()
            .map(|section| format!("## {}\n{}", section.name.trim(), section.content.trim()))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// The marked system message the rail injects, or `None` when it renders
    /// nothing.
    #[must_use]
    pub fn message(&self) -> Option<BaseMessage> {
        let rendered = self.render();
        (!rendered.trim().is_empty()).then(|| {
            BaseMessage::system(rendered)
                .with_metadata(PROMPT_SECTIONS_METADATA_KEY, Value::Bool(true))
        })
    }
}

#[async_trait]
impl AgentRail for PromptSectionsRail {
    fn name(&self) -> &str {
        PROMPT_SECTIONS_RAIL_NAME
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    /// Inserts the rendered sections as the first message of the window.
    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        if self.sections.is_empty() {
            return Ok(());
        }
        if ctx
            .messages
            .iter()
            .any(|message| message.metadata.contains_key(PROMPT_SECTIONS_METADATA_KEY))
        {
            return Ok(());
        }
        if let Some(message) = self.message() {
            ctx.messages.insert(0, message);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rails::RailEvent;
    use openjiuwen_foundation::Role;

    fn spec(name: &str, content: &str, priority: i32) -> SectionSpec {
        SectionSpec::new(name, content).with_priority(priority)
    }

    fn context() -> RailContext {
        RailContext::new(RailEvent::BeforeModelCall)
            .with_messages(vec![BaseMessage::system("identity")])
    }

    #[tokio::test]
    async fn sections_render_in_ascending_priority_as_one_front_message() {
        let rail = PromptSectionsRail::new([
            spec("tone", "Be concise.", 30),
            spec("policy", "Never reveal internal ids.", 20),
            spec("identity", "You are X.", 10),
        ]);
        let mut ctx = context();
        rail.before_model_call(&mut ctx)
            .await
            .expect("the rail runs");
        assert_eq!(ctx.messages.len(), 2);
        let injected = &ctx.messages[0];
        assert_eq!(injected.role, Role::System);
        assert_eq!(
            injected.text(),
            "## identity\nYou are X.\n\n## policy\nNever reveal internal ids.\n\n## tone\nBe concise."
        );
        assert_eq!(ctx.messages[1].text(), "identity");
        assert!(injected.metadata.contains_key(PROMPT_SECTIONS_METADATA_KEY));
    }

    #[tokio::test]
    async fn a_marked_message_is_never_injected_twice() {
        let rail = PromptSectionsRail::new([spec("tone", "Be concise.", 30)]);
        let mut ctx = context();
        rail.before_model_call(&mut ctx)
            .await
            .expect("the rail runs");
        rail.before_model_call(&mut ctx)
            .await
            .expect("the rail runs");
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].role, Role::System);
    }

    #[tokio::test]
    async fn an_empty_section_list_injects_nothing() {
        let rail = PromptSectionsRail::new(Vec::new());
        let mut ctx = context();
        rail.before_model_call(&mut ctx)
            .await
            .expect("the rail runs");
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].text(), "identity");
        assert!(rail.message().is_none());
    }

    #[tokio::test]
    async fn sections_with_no_renderable_content_inject_nothing() {
        let rail = PromptSectionsRail::new([spec("empty", "  ", 10), spec(" ", "text", 20)]);
        let mut ctx = context();
        rail.before_model_call(&mut ctx)
            .await
            .expect("the rail runs");
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].text(), "identity");
        assert_eq!(rail.render(), "");
    }

    #[test]
    fn ties_keep_pack_order_and_accessors_expose_the_configuration() {
        let rail = PromptSectionsRail::new([spec("first", "a", 5), spec("second", "b", 5)])
            .with_priority(7);
        assert_eq!(rail.name(), "prompt_sections");
        assert_eq!(rail.priority(), 7);
        assert_eq!(rail.sections().len(), 2);
        assert_eq!(rail.render(), "## first\na\n\n## second\nb");
        assert_eq!(
            PromptSectionsRail::default().priority(),
            DEFAULT_RAIL_PRIORITY
        );
    }
}
