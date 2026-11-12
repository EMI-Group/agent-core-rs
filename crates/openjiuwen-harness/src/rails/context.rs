//! The per-firing-point context handed to a [`crate::rails::HarnessRail`].
//!
//! [`HarnessRailContext`] wraps the kernel [`RailContext`] (messages, tools,
//! tool call/result, output, control flags, `extra`) and adds the outer
//! task-loop information that only the harness has. It also carries the
//! prompt-section channel: rails publish [`PromptSection`]s into
//! `base.extra` and the agent renders them when it assembles the system prompt,
//! so rails stay pure context mutators.
use std::collections::BTreeMap;
use std::time::Duration;

use openjiuwen_agent::{AgentOutput, InterruptRequest, RailContext, RailEvent};
use openjiuwen_foundation::{BaseMessage, Role};
use serde_json::{json, Value};

use crate::prompts::{PromptCategory, PromptSection};
use crate::schema::DeepLoopEventType;

/// Key in [`RailContext::extra`] holding the published prompt sections.
pub const PROMPT_SECTIONS_EXTRA_KEY: &str = "harness.prompt_sections";

/// Key in [`RailContext::extra`] holding the injected memory context text.
pub const MEMORY_CONTEXT_EXTRA_KEY: &str = "harness.memory_context";

/// Key in [`RailContext::extra`] holding the recall query of the current turn.
pub const MEMORY_QUERY_EXTRA_KEY: &str = "harness.memory_query";

/// The data and control flags one harness rail hook observes.
#[derive(Debug)]
pub struct HarnessRailContext {
    /// The kernel context: event, messages, tools, control flags and `extra`.
    pub base: RailContext,
    /// Outer task-loop information; set only while a task-iteration hook fires.
    pub task: Option<TaskIterationInfo>,
}

impl HarnessRailContext {
    /// Wraps a kernel context without task-iteration information.
    #[must_use]
    pub fn new(base: RailContext) -> Self {
        Self { base, task: None }
    }

    /// Builds an empty context for `event`.
    #[must_use]
    pub fn for_event(event: RailEvent) -> Self {
        Self::new(RailContext::new(event))
    }

    /// Attaches the outer task-loop information of the firing point.
    #[must_use]
    pub fn with_task(mut self, info: TaskIterationInfo) -> Self {
        self.task = Some(info);
        self
    }

    /// The event being fired.
    #[must_use]
    pub fn event(&self) -> RailEvent {
        self.base.event
    }

    /// The wrapped kernel context.
    #[must_use]
    pub fn base(&self) -> &RailContext {
        &self.base
    }

    /// The wrapped kernel context, mutably.
    pub fn base_mut(&mut self) -> &mut RailContext {
        &mut self.base
    }

    /// The outer task-loop information, when a task-iteration hook fires.
    #[must_use]
    pub fn task_info(&self) -> Option<&TaskIterationInfo> {
        self.task.as_ref()
    }

    /// The per-invocation scratch map shared by all rails.
    #[must_use]
    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.base.extra
    }

    /// The per-invocation scratch map, mutably.
    pub fn extra_mut(&mut self) -> &mut BTreeMap<String, Value> {
        &mut self.base.extra
    }

    /// Replaces the tool result with `message` instead of executing the tool.
    pub fn request_skip_tool(&mut self, message: impl Into<String>) {
        self.base.skip_tool_with(message);
    }

    /// Records an interrupt request raised during this firing point.
    pub fn request_interrupt(&mut self, request: InterruptRequest) {
        self.base.request_interrupt(request);
    }

    /// Asks the loop to finish now with `output`.
    pub fn request_force_finish(&mut self, output: AgentOutput) {
        self.base.request_force_finish(output);
    }

    /// Asks the loop to retry the model call, optionally after `delay`.
    pub fn request_retry(&mut self, delay: Option<Duration>) {
        self.base.request_retry(delay);
    }

    /// Asks the loop to continue instead of finishing.
    pub fn request_model_continue(&mut self) {
        self.base.request_model_continue();
    }

    /// Whether any interrupt has been raised for this firing point.
    #[must_use]
    pub fn is_interrupted(&self) -> bool {
        self.base.is_interrupted()
    }
}

/// Outer task-loop information of one firing point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskIterationInfo {
    /// Id of the outer round the iteration belongs to.
    pub round_id: u64,
    /// Iteration index inside the round, `0`-based.
    pub iteration: usize,
    /// Whether the iteration runs a follow-up message rather than the original task.
    pub is_follow_up: bool,
    /// Kind of loop event that triggered the iteration, when one did.
    pub loop_event: Option<DeepLoopEventType>,
}

impl TaskIterationInfo {
    /// Builds the information of a first task iteration.
    #[must_use]
    pub fn new(round_id: u64, iteration: usize) -> Self {
        Self {
            round_id,
            iteration,
            is_follow_up: false,
            loop_event: None,
        }
    }

    /// Marks the iteration as running a follow-up message.
    #[must_use]
    pub fn with_follow_up(mut self) -> Self {
        self.is_follow_up = true;
        self
    }

    /// Records the loop event that triggered the iteration.
    #[must_use]
    pub fn with_loop_event(mut self, event: DeepLoopEventType) -> Self {
        self.loop_event = Some(event);
        self
    }
}

/// Which task-iteration hook a registry call fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskIterationPhase {
    /// Fire [`crate::rails::HarnessRail::before_task_iteration`].
    Before,
    /// Fire [`crate::rails::HarnessRail::after_task_iteration`].
    After,
}

/// Merges `section` into the prompt-section channel of `ctx`.
///
/// The channel is a JSON object under [`PROMPT_SECTIONS_EXTRA_KEY`] keyed by
/// section name, so publishing a section twice keeps the last content
/// (Python's upsert-by-name behaviour). Each entry is
/// `{"content": {"cn": …, "en": …}, "priority": 40, "carrier": "system_message", "category": "skills"}`;
/// `category` is `null` when the section declares none.
pub fn publish_prompt_section(ctx: &mut HarnessRailContext, section: &PromptSection) {
    let value = ctx
        .extra_mut()
        .entry(PROMPT_SECTIONS_EXTRA_KEY.to_string())
        .or_insert_with(|| json!({}));
    if !value.is_object() {
        *value = json!({});
    }
    let Some(entries) = value.as_object_mut() else {
        return;
    };
    let content: serde_json::Map<String, Value> = section
        .content
        .iter()
        .map(|(language, text)| (language.clone(), Value::String(text.clone())))
        .collect();
    let category = section.category.map_or(Value::Null, |category| {
        Value::String(category.as_str().to_string())
    });
    entries.insert(
        section.name.clone(),
        json!({
            "content": content,
            "priority": section.priority,
            "carrier": section.carrier,
            "category": category,
        }),
    );
}

/// Collects the prompt sections published into `ctx`.
///
/// Unknown fields are ignored, a missing `content` becomes an empty map, a
/// missing `priority` becomes `0`, a missing `carrier` becomes
/// `"system_message"`, and an unknown `category` becomes `None`. Non-object
/// entries are skipped. The result is sorted by `(priority, name)`.
#[must_use]
pub fn prompt_sections(ctx: &HarnessRailContext) -> Vec<PromptSection> {
    let Some(entries) = ctx
        .extra()
        .get(PROMPT_SECTIONS_EXTRA_KEY)
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    let mut sections = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let content: BTreeMap<String, String> = entry
            .get("content")
            .and_then(Value::as_object)
            .map(|content| {
                content
                    .iter()
                    .filter_map(|(language, text)| {
                        text.as_str()
                            .map(|text| (language.clone(), text.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let priority = entry
            .get("priority")
            .and_then(Value::as_i64)
            .and_then(|priority| i32::try_from(priority).ok())
            .unwrap_or_default();
        let carrier = entry
            .get("carrier")
            .and_then(Value::as_str)
            .unwrap_or("system_message")
            .to_string();
        let category = entry
            .get("category")
            .and_then(Value::as_str)
            .and_then(PromptCategory::parse);
        sections.push(PromptSection {
            name: name.clone(),
            content,
            priority,
            carrier,
            category,
        });
    }
    sections.sort_by(|left, right| {
        left.priority
            .cmp(&right.priority)
            .then_with(|| left.name.cmp(&right.name))
    });
    sections
}

/// Returns the current-turn text: the flattened content of the last
/// [`Role::User`] message, or `None` when there is none or it is blank.
#[must_use]
pub fn last_user_text(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.content.to_plain_text())
        .filter(|text| !text.trim().is_empty())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::prompts::PromptCategory;
    use openjiuwen_agent::{AgentOutput, InterruptKind, InterruptRequest};
    use openjiuwen_foundation::BaseMessage;
    use std::time::Duration;

    fn user(text: &str) -> BaseMessage {
        BaseMessage::user(text)
    }

    #[test]
    fn for_event_sets_the_event_and_no_task() {
        let ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        assert_eq!(ctx.event(), RailEvent::BeforeModelCall);
        assert!(ctx.task_info().is_none());
    }

    #[test]
    fn with_task_attaches_iteration_info() {
        let info = TaskIterationInfo::new(3, 2)
            .with_follow_up()
            .with_loop_event(DeepLoopEventType::Steer);
        let ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke).with_task(info);
        assert_eq!(ctx.task_info(), Some(&info));
        assert_eq!(info.round_id, 3);
        assert_eq!(info.iteration, 2);
        assert!(info.is_follow_up);
        assert_eq!(info.loop_event, Some(DeepLoopEventType::Steer));
    }

    #[test]
    fn task_iteration_info_defaults() {
        let info = TaskIterationInfo::new(1, 0);
        assert!(!info.is_follow_up);
        assert!(info.loop_event.is_none());
    }

    #[test]
    fn extra_mut_round_trips_and_is_shared_with_base() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.extra_mut().insert("k".to_string(), json!(7));
        assert_eq!(ctx.extra().get("k"), Some(&json!(7)));
        assert_eq!(ctx.base().extra.get("k"), Some(&json!(7)));
    }

    #[test]
    fn control_signals_delegate_to_the_kernel_context() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeToolCall);
        ctx.request_skip_tool("blocked");
        assert_eq!(ctx.base().skip_tool.as_deref(), Some("blocked"));

        ctx.request_force_finish(AgentOutput::answer("done"));
        assert_eq!(
            ctx.base()
                .force_finish
                .as_ref()
                .map(|out| out.output.as_str()),
            Some("done")
        );

        ctx.request_model_continue();
        assert!(ctx.base().model_continue);

        ctx.request_retry(Some(Duration::from_millis(25)));
        assert_eq!(
            ctx.base().retry.as_ref().and_then(|retry| retry.delay),
            Some(Duration::from_millis(25))
        );

        assert!(!ctx.is_interrupted());
        ctx.request_interrupt(InterruptRequest::new(
            InterruptKind::Permission,
            "rail",
            "allow?",
        ));
        assert!(ctx.is_interrupted());
        assert_eq!(ctx.base().interrupts.len(), 1);
    }

    #[test]
    fn prompt_section_round_trips_content_priority_carrier_and_category() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        let section = PromptSection::bilingual("skills", "技能", "Skills", 40)
            .with_category(PromptCategory::Skills);
        publish_prompt_section(&mut ctx, &section);

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        let decoded = &sections[0];
        assert_eq!(decoded.name, "skills");
        assert_eq!(decoded.content.get("cn").map(String::as_str), Some("技能"));
        assert_eq!(
            decoded.content.get("en").map(String::as_str),
            Some("Skills")
        );
        assert_eq!(decoded.priority, 40);
        assert_eq!(decoded.carrier, "system_message");
        assert_eq!(decoded.category, Some(PromptCategory::Skills));
    }

    #[test]
    fn prompt_section_without_category_decodes_as_none() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        publish_prompt_section(&mut ctx, &PromptSection::bilingual("a", "甲", "A", 1));

        let channel = ctx.extra().get(PROMPT_SECTIONS_EXTRA_KEY).unwrap();
        assert_eq!(channel["a"]["category"], Value::Null);
        assert_eq!(prompt_sections(&ctx)[0].category, None);
    }

    #[test]
    fn prompt_section_overwrites_by_name() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        publish_prompt_section(
            &mut ctx,
            &PromptSection::bilingual("skills", "旧", "old", 40),
        );
        publish_prompt_section(
            &mut ctx,
            &PromptSection::bilingual("skills", "新", "new", 41),
        );

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].priority, 41);
        assert_eq!(
            sections[0].content.get("en").map(String::as_str),
            Some("new")
        );
    }

    #[test]
    fn prompt_sections_are_sorted_by_priority_then_name() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        publish_prompt_section(&mut ctx, &PromptSection::bilingual("b", "乙", "B", 20));
        publish_prompt_section(&mut ctx, &PromptSection::bilingual("a", "甲", "A", 20));
        publish_prompt_section(&mut ctx, &PromptSection::bilingual("z", "祖", "Z", 5));

        let sections = prompt_sections(&ctx);
        let names: Vec<&str> = sections
            .iter()
            .map(|section| section.name.as_str())
            .collect();
        assert_eq!(names, ["z", "a", "b"]);
    }

    #[test]
    fn prompt_sections_tolerate_missing_fields_and_unknown_entries() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.extra_mut().insert(
            PROMPT_SECTIONS_EXTRA_KEY.to_string(),
            json!({
                "minimal": {},
                "broken": 3,
                "unknown_category": {"content": {"en": "text"}, "category": "nope", "extra": true},
            }),
        );

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].name, "minimal");
        assert!(sections[0].content.is_empty());
        assert_eq!(sections[0].priority, 0);
        assert_eq!(sections[0].carrier, "system_message");
        assert_eq!(sections[0].category, None);
        assert_eq!(sections[1].name, "unknown_category");
        assert_eq!(
            sections[1].content.get("en").map(String::as_str),
            Some("text")
        );
        assert_eq!(sections[1].category, None);
    }

    #[test]
    fn prompt_sections_absent_or_non_object_yield_nothing() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        assert_eq!(prompt_sections(&ctx).len(), 0);

        ctx.extra_mut()
            .insert(PROMPT_SECTIONS_EXTRA_KEY.to_string(), json!("nope"));
        assert_eq!(prompt_sections(&ctx).len(), 0);
    }

    #[test]
    fn publishing_into_a_non_object_channel_resets_it() {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        ctx.extra_mut()
            .insert(PROMPT_SECTIONS_EXTRA_KEY.to_string(), json!([]));
        publish_prompt_section(&mut ctx, &PromptSection::bilingual("a", "甲", "A", 1));
        assert_eq!(prompt_sections(&ctx).len(), 1);
    }

    #[test]
    fn last_user_text_picks_the_last_user_message() {
        let messages = vec![
            user("first"),
            BaseMessage::system("system text"),
            user("second"),
            BaseMessage::assistant("reply"),
            BaseMessage::tool("call-1", "tool output"),
        ];
        assert_eq!(last_user_text(&messages).as_deref(), Some("second"));
    }

    #[test]
    fn last_user_text_is_none_without_a_user_message() {
        let messages = vec![
            BaseMessage::system("system text"),
            BaseMessage::assistant("reply"),
        ];
        assert!(last_user_text(&messages).is_none());
        assert!(last_user_text(&[]).is_none());
    }

    #[test]
    fn last_user_text_ignores_blank_text() {
        assert!(last_user_text(&[user("   ")]).is_none());
        assert!(last_user_text(&[user("real"), user("\n\t")]).is_none());
    }

    #[test]
    fn task_iteration_phase_variants_are_distinct() {
        assert_ne!(TaskIterationPhase::Before, TaskIterationPhase::After);
    }
}
