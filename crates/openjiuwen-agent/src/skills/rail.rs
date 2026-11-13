//! The skill rail: advertise the discovered skills in the model prompt.
//!
//! Python's `ReActAgent` injects `SkillUtil.get_skill_prompt()` as a
//! `legacy_skills` prompt section; this crate keeps the same idea behind the rail
//! seam so the skill library can change between invocations. On every
//! `BeforeModelCall` the rail inserts one system message at the front of the
//! prepared window, marked with [`SKILL_PROMPT_METADATA_KEY`] so a later
//! iteration never injects it twice.
use super::skill::Skill;
use super::util::SkillUtil;
use super::SkillManager;
use crate::input::AgentInput;
use crate::rails::{AgentRail, RailContext, DEFAULT_RAIL_PRIORITY};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::BaseMessage;
use serde_json::Value;

/// Metadata key marking the system message that carries the skill prompt.
///
/// A message with this key in its metadata is never injected a second time, so a
/// later `BeforeModelCall` of the same invocation leaves the window untouched.
pub const SKILL_PROMPT_METADATA_KEY: &str = "_openjiuwen_skill_prompt";

/// Character budget a [`SkillRail`] gets when it does not override it.
pub const DEFAULT_SKILL_PROMPT_BUDGET: usize = 4_000;

/// Injects the skill catalog into every model request of an invocation.
///
/// The rail holds its own [`SkillManager`]; the skills it advertises are the
/// ones selected for the invocation's query
/// ([`SkillUtil::select`], which falls back to the whole catalog when nothing
/// matches) and rendered by [`SkillUtil::render_prompt`] under
/// [`SkillRail::budget`]. An empty library, or an empty rendering, leaves the
/// window untouched: the rail never fails the run.
#[derive(Debug, Clone)]
pub struct SkillRail {
    manager: SkillManager,
    budget: usize,
    priority: i32,
}

impl SkillRail {
    /// Builds a rail over `manager` with the default budget and priority.
    #[must_use]
    pub fn new(manager: SkillManager) -> Self {
        Self {
            manager,
            budget: DEFAULT_SKILL_PROMPT_BUDGET,
            priority: DEFAULT_RAIL_PRIORITY,
        }
    }

    /// Sets the character budget of the rendered prompt.
    #[must_use]
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// Sets the rail dispatch priority (higher runs first).
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns the skill library the rail advertises.
    #[must_use]
    pub fn manager(&self) -> &SkillManager {
        &self.manager
    }

    /// Returns the character budget of the rendered prompt.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Returns the skills to advertise for `input`: the query's matches when the
    /// invocation carries text, the whole catalog otherwise.
    fn skills_for(&self, input: Option<&AgentInput>) -> Vec<Skill> {
        let skills = self.manager.list();
        match input.and_then(AgentInput::query_text) {
            Some(query) if !query.trim().is_empty() => SkillUtil::select(&skills, query),
            _ => skills,
        }
    }
}

#[async_trait]
impl AgentRail for SkillRail {
    fn name(&self) -> &'static str {
        "skill"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    /// Inserts the rendered skill prompt as the first message of the window.
    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        if self.manager.is_empty() {
            return Ok(());
        }
        if ctx
            .messages
            .iter()
            .any(|message| message.metadata.contains_key(SKILL_PROMPT_METADATA_KEY))
        {
            return Ok(());
        }
        let prompt = SkillUtil::render_prompt(&self.skills_for(ctx.input.as_ref()), self.budget);
        if prompt.trim().is_empty() {
            return Ok(());
        }
        let message =
            BaseMessage::system(prompt).with_metadata(SKILL_PROMPT_METADATA_KEY, Value::Bool(true));
        ctx.messages.insert(0, message);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::rails::RailEvent;
    use openjiuwen_foundation::Role;

    fn manager() -> SkillManager {
        SkillManager::with_skills([
            Skill::new("deploy", "Deploy the service"),
            Skill::new("rollback", "Roll back the service"),
        ])
    }

    fn context(input: Option<AgentInput>) -> RailContext {
        let mut ctx = RailContext::new(RailEvent::BeforeModelCall)
            .with_messages(vec![BaseMessage::system("identity")]);
        ctx.input = input;
        ctx
    }

    #[tokio::test]
    async fn inserts_a_marked_system_message_at_the_front() {
        let rail = SkillRail::new(manager());
        let mut ctx = context(None);
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 2);
        let injected = &ctx.messages[0];
        assert_eq!(injected.role, Role::System);
        assert!(injected.metadata.contains_key(SKILL_PROMPT_METADATA_KEY));
        assert_eq!(ctx.messages[1].text(), "identity");
        assert!(injected.text().starts_with("## Skills"));
        assert!(injected.text().contains("deploy"));
        assert!(injected.text().contains("rollback"));
    }

    #[tokio::test]
    async fn a_marked_message_is_never_injected_twice() {
        let rail = SkillRail::new(manager());
        let mut ctx = context(None);
        rail.before_model_call(&mut ctx).await.unwrap();
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].role, Role::System);
    }

    #[tokio::test]
    async fn an_empty_manager_injects_nothing() {
        let rail = SkillRail::new(SkillManager::new());
        let mut ctx = context(None);
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].text(), "identity");
    }

    #[tokio::test]
    async fn the_query_selects_the_advertised_skills() {
        let rail = SkillRail::new(manager());
        let mut ctx = context(Some(AgentInput::from_text("please rollback")));
        rail.before_model_call(&mut ctx).await.unwrap();
        let injected = ctx.messages[0].text().into_owned();
        assert!(injected.contains("rollback"));
        assert!(!injected.contains("deploy"));

        let mut unmatched = context(Some(AgentInput::from_text("unrelated question")));
        rail.before_model_call(&mut unmatched).await.unwrap();
        let injected = unmatched.messages[0].text().into_owned();
        assert!(injected.contains("rollback"));
        assert!(injected.contains("deploy"));
    }

    #[tokio::test]
    async fn the_budget_bounds_the_rendered_prompt() {
        let rail = SkillRail::new(manager()).with_budget(20);
        let mut ctx = context(None);
        rail.before_model_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages[0].text(), "## Skills\n\n…");
    }

    #[test]
    fn accessors_expose_the_configuration() {
        let rail = SkillRail::new(manager()).with_budget(123).with_priority(7);
        assert_eq!(rail.name(), "skill");
        assert_eq!(rail.priority(), 7);
        assert_eq!(rail.budget(), 123);
        assert_eq!(rail.manager().len(), 2);
        assert_eq!(
            SkillRail::new(SkillManager::new()).priority(),
            DEFAULT_RAIL_PRIORITY
        );
    }
}
