//! System-prompt assembly and steering injection for the `DeepAgent` loop.
//!
//! The static half of the system prompt lives in a [`HarnessPromptBuilder`]
//! owned by [`PromptState`]; rails publish per-invocation sections through
//! [`crate::rails::prompt_sections`]. [`PromptRail`] fires last (priority `5`)
//! and, before every model call, replaces the leading system message with
//! `builder.build()` plus the published sections that have no static
//! counterpart, then drains queued steering inputs into one user message each.
//!
//! Prompt text is never logged: only section names and the built size, through
//! the builder's own `tracing` calls.

use std::collections::BTreeSet;
use std::sync::Arc;

use openjiuwen_foundation::{BaseMessage, Role};
use parking_lot::RwLock;

use openjiuwen_common::error::Result;

use crate::prompts::{HarnessPromptBuilder, PromptReport, PromptSection};
use crate::rails::{HarnessRail, HarnessRailContext};
use crate::task_loop::LoopQueues;

/// Priority of [`PromptRail`]: the last rail to run, so it observes every
/// section published by earlier rails.
const PROMPT_RAIL_PRIORITY: i32 = 5;

/// Stable name of [`PromptRail`].
const PROMPT_RAIL_NAME: &str = "harness_prompt";

/// Prefix prepended to every drained steering input.
const STEERING_PREFIX: &str = "User steering: ";

/// The mutable system-prompt state of one `DeepAgent`.
///
/// Shared as `Arc<PromptState>` between the agent (which owns the static
/// sections) and [`PromptRail`] (which renders them). The inner builder is
/// behind an `RwLock`; the guard is never held across an `.await` because every
/// accessor here is synchronous.
pub struct PromptState {
    builder: RwLock<HarnessPromptBuilder>,
}

impl PromptState {
    /// Wraps `builder` into a shared prompt state.
    #[must_use]
    pub fn new(builder: HarnessPromptBuilder) -> Arc<Self> {
        Arc::new(Self {
            builder: RwLock::new(builder),
        })
    }

    /// Renders the static prompt only, without published sections.
    #[must_use]
    pub fn system_text(&self) -> String {
        self.builder.read().build()
    }

    /// Renders the static prompt plus every published section.
    ///
    /// A published section is appended in `published` order (the channel is
    /// already sorted by priority/name) when no static section has the same
    /// name and it renders to non-empty text for the builder language.
    /// Duplicates are removed by section name, static sections always win.
    #[must_use]
    pub fn render(&self, published: &[PromptSection]) -> String {
        let builder = self.builder.read();
        let mut text = builder.build();
        let language = builder.language().to_string();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for section in published {
            if builder.has_section(&section.name) || !seen.insert(section.name.clone()) {
                continue;
            }
            let rendered = section.render(&language);
            if rendered.trim().is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&rendered);
        }
        text
    }

    /// Replaces every static section with `sections`.
    ///
    /// The builder keeps its language and [`PromptMode`], so a mode switch only
    /// swaps the section set. Used by `switch_mode`/todo refreshes.
    ///
    /// [`PromptMode`]: crate::prompts::PromptMode
    ///
    /// # Errors
    ///
    /// Propagates [`HarnessPromptBuilder::add_section`] errors, e.g. two sections
    /// with one name and different priorities.
    pub fn replace_sections(&self, sections: Vec<PromptSection>) -> Result<()> {
        let mut guard = self.builder.write();
        let language = guard.language().to_string();
        let mode = guard.mode();
        let mut builder = HarnessPromptBuilder::new(language, mode);
        for section in sections {
            builder.add_section(section)?;
        }
        *guard = builder;
        Ok(())
    }

    /// Mutates the builder under its write lock.
    ///
    /// The closure is synchronous, so the guard is released before the caller
    /// can await again.
    pub fn update(&self, f: impl FnOnce(&mut HarnessPromptBuilder)) {
        f(&mut self.builder.write());
    }

    /// Builds the size/token report of the static prompt.
    #[must_use]
    pub fn report(&self) -> PromptReport {
        self.builder.read().build_report()
    }
}

/// Renders the system prompt and injects steering inputs before each model call.
pub struct PromptRail {
    prompt: Arc<PromptState>,
    queues: LoopQueues,
}

impl PromptRail {
    /// Creates a rail rendering `prompt` and draining `queues`.
    #[must_use]
    pub fn new(prompt: Arc<PromptState>, queues: LoopQueues) -> Self {
        Self { prompt, queues }
    }
}

#[async_trait::async_trait]
impl HarnessRail for PromptRail {
    fn name(&self) -> &str {
        PROMPT_RAIL_NAME
    }

    fn priority(&self) -> i32 {
        PROMPT_RAIL_PRIORITY
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let published = crate::rails::prompt_sections(ctx);
        let text = self.prompt.render(&published);

        let has_leading_system = ctx
            .base
            .messages
            .first()
            .is_some_and(|message| message.role == Role::System);
        if has_leading_system {
            if let Some(message) = ctx.base.messages.first_mut() {
                message.content = text.into();
            }
        } else {
            ctx.base.messages.insert(0, BaseMessage::system(text));
        }

        for input in self.queues.drain_steering() {
            ctx.base.messages.push(BaseMessage::user(format!(
                "{STEERING_PREFIX}{}",
                input.content()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_agent::RailEvent;
    use openjiuwen_foundation::BaseMessage;

    use super::{PromptRail, PromptState, PROMPT_RAIL_NAME, PROMPT_RAIL_PRIORITY};
    use crate::prompts::{identity, HarnessPromptBuilder, PromptMode, PromptSection};
    use crate::rails::{publish_prompt_section, HarnessRail, HarnessRailContext};
    use crate::task_loop::LoopQueues;

    fn prompt_with_identity() -> std::sync::Arc<PromptState> {
        let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
        builder.add_section(identity()).unwrap();
        PromptState::new(builder)
    }

    #[test]
    fn render_appends_published_sections_absent_from_the_builder() {
        let prompt = prompt_with_identity();
        let published = vec![PromptSection::bilingual(
            "skills",
            "技能内容",
            "Skill content",
            40,
        )];
        let rendered = prompt.render(&published);
        assert!(rendered.contains("general-purpose AI assistant"));
        assert!(rendered.contains("Skill content"));
        assert!(!rendered.contains("技能内容"));
        assert!(rendered.ends_with("Skill content"));
        assert!(rendered.contains("\n\n"));
    }

    #[test]
    fn render_does_not_duplicate_a_static_section() {
        let prompt = prompt_with_identity();
        let published = vec![PromptSection::bilingual(
            "identity",
            "静态中文",
            "Published identity text",
            10,
        )];
        let rendered = prompt.render(&published);
        assert!(!rendered.contains("Published identity text"));
        assert_eq!(
            rendered
                .matches("You are a general-purpose AI assistant.")
                .count(),
            1
        );
    }

    #[test]
    fn render_is_empty_for_an_empty_builder_and_no_sections() {
        let prompt = PromptState::new(HarnessPromptBuilder::new("en", PromptMode::Full));
        assert_eq!(prompt.render(&[]), String::new());
        assert_eq!(prompt.system_text(), String::new());
        assert_eq!(prompt.report().section_count, 0);
    }

    #[test]
    fn replace_sections_keeps_language_and_mode() {
        let prompt = PromptState::new(HarnessPromptBuilder::new("en", PromptMode::Full));
        prompt
            .replace_sections(vec![PromptSection::bilingual(
                "only",
                "只有中文",
                "Only English",
                12,
            )])
            .unwrap();
        assert_eq!(prompt.system_text(), "Only English");
    }

    #[tokio::test]
    async fn before_model_call_replaces_the_leading_system_message() {
        let prompt = prompt_with_identity();
        let queues = LoopQueues::default();
        let rail = PromptRail::new(prompt, queues);

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages.push(BaseMessage::system("old system"));
        ctx.base.messages.push(BaseMessage::user("hello"));
        ctx.base.messages.push(BaseMessage::assistant("hi"));
        let section = PromptSection::bilingual("memory", "记忆", "Memory text", 30);
        publish_prompt_section(&mut ctx, &section);

        rail.before_model_call(&mut ctx).await.unwrap();

        assert_eq!(ctx.base.messages.len(), 3);
        assert_eq!(
            ctx.base.messages[0].role,
            openjiuwen_foundation::Role::System
        );
        assert!(ctx.base.messages[0].text().contains("Memory text"));
        assert_eq!(ctx.base.messages[1].text().as_ref(), "hello");
        assert_eq!(ctx.base.messages[2].text().as_ref(), "hi");
    }

    #[tokio::test]
    async fn before_model_call_inserts_a_system_message_when_absent() {
        let prompt = prompt_with_identity();
        let rail = PromptRail::new(prompt, LoopQueues::default());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages.push(BaseMessage::user("hello"));

        rail.before_model_call(&mut ctx).await.unwrap();

        assert_eq!(ctx.base.messages.len(), 2);
        assert_eq!(
            ctx.base.messages[0].role,
            openjiuwen_foundation::Role::System
        );
        assert!(ctx.base.messages[0]
            .text()
            .contains("general-purpose AI assistant"));
        assert_eq!(ctx.base.messages[1].text().as_ref(), "hello");
    }

    #[tokio::test]
    async fn before_model_call_drains_steering_into_user_messages() {
        let prompt = prompt_with_identity();
        let queues = LoopQueues::default();
        queues.push_steer("focus on tests");
        queues.push_steer("keep it short");
        let rail = PromptRail::new(prompt, queues.clone());

        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.messages.push(BaseMessage::system("old"));
        rail.before_model_call(&mut ctx).await.unwrap();

        assert_eq!(ctx.base.messages.len(), 3);
        assert_eq!(
            ctx.base.messages[1].text().as_ref(),
            "User steering: focus on tests"
        );
        assert_eq!(
            ctx.base.messages[2].text().as_ref(),
            "User steering: keep it short"
        );
        assert_eq!(queues.drain_steering().len(), 0);
    }

    #[test]
    fn rail_identity_and_priority() {
        let rail = PromptRail::new(
            PromptState::new(HarnessPromptBuilder::new("en", PromptMode::Full)),
            LoopQueues::default(),
        );
        assert_eq!(rail.name(), PROMPT_RAIL_NAME);
        assert_eq!(rail.name(), "harness_prompt");
        assert_eq!(rail.priority(), PROMPT_RAIL_PRIORITY);
        assert_eq!(rail.priority(), 5);
    }
}
