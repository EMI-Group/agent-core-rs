//! The persistent-goal rail: re-orients the model on its goal and emits `goal.updated`.
//!
//! This is the loop-side counterpart of [`super::tools`]: before every outer task iteration it
//! reads the persisted [`GoalRecord`] and, while the goal is active, publishes a prompt section
//! carrying the objective, the lifecycle status, the attempt count and — when configured — the
//! last assessment's evidence, remaining work and next instruction. Once the goal settles, the
//! rail emits one `goal.updated` interaction event so the host can refresh its goal UI.
//!
//! Ports the reminder/event half of the Python goal capability (`GoalManager` driving
//! `openjiuwen/harness/tools/goal.py`); attempt scheduling and pause/resume policy stay with the
//! caller, and every store mutation still goes through [`GoalStore`].
//!
//! # Prompt section
//!
//! The rail publishes the section named `goal` (title `Current Goal`, Chinese `当前目标`) with
//! render priority 55 and category [`PromptCategory::SystemPrompt`], the same way
//! [`crate::rails::MemoryRail`] publishes the memory section. The text is rendered once in
//! [`GoalRailConfig::prompt_language`] and stored under both language keys, so the assembled
//! prompt shows the reminder identically whichever language it resolves.
//!
//! # Events
//!
//! Rails have no direct event sink, so `goal.updated` events are appended to a JSON array under
//! [`GOAL_EVENT_EXTRA_KEY`] in [`HarnessRailContext::extra_mut`]. The agent or task loop drains
//! that key after firing the hooks and forwards every entry to the interaction output stream.
//! Entries use [`InteractionEvent::to_output_value`]: `{"type": "goal.updated", "index": 0,
//! "payload": {"goal": { … }}}`.
//!
//! # Registration
//!
//! Hosts register the rail with `HarnessRailRegistry::register`. This module is the canonical
//! path today (`crate::tools::goal::GoalRail`); a `crate::rails::goal` re-export is pending, at
//! which point this file moves next to the other built-in rails.
//!
//! # Failure policy
//!
//! A missing, unreadable or corrupt goal file is logged and the hook returns `Ok(())`: goal
//! tracking is an enhancement and must never abort the agent loop.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;

use crate::prompts::{PromptCategory, PromptSection};
use crate::rails::{publish_prompt_section, HarnessRail, HarnessRailContext};
use crate::schema::InteractionEvent;
use crate::tools::{is_english, render_fields, truncate_chars};

use super::record::GoalRecord;
use super::store::GoalStore;

/// Dispatch priority of the goal rail.
///
/// 90 is the Python planning/interrupt tier (`rail_priority::PLANNING` and
/// `rail_priority::INTERRUPT`), where the goal reminder is registered. The constant is local
/// because `src/rails/mod.rs` owns that ladder and this rail is scheduled to move there; the
/// value must stay equal to `crate::rails::rail_priority::PLANNING`.
const GOAL_RAIL_PRIORITY: i32 = 90;

/// Name of the prompt section the rail publishes.
const GOAL_SECTION_NAME: &str = "goal";

/// Render priority of the goal section inside the assembled system prompt.
const GOAL_SECTION_PRIORITY: i32 = 55;

/// Marker opening the heartbeat nudge appended to the goal section.
const GOAL_HEARTBEAT_MARKER: &str = "[goal-heartbeat]";

/// Key in [`HarnessRailContext::extra_mut`] holding the queued `goal.updated` events.
///
/// The value is a JSON array; the rail appends to it and never overwrites it, so several
/// iterations of one invocation accumulate their events. The agent or task loop drains the key
/// after firing the hooks and forwards each entry to the interaction stream.
pub const GOAL_EVENT_EXTRA_KEY: &str = "harness.goal_events";

/// Tunables of [`GoalRail`].
///
/// Serializes with `snake_case` keys, tolerates unknown fields and falls back to [`Default`] for
/// every missing field, so a partial host configuration is valid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GoalRailConfig {
    /// Iterations between "continue/verify" nudges; `None` disables the heartbeat.
    ///
    /// `Some(0)` is also treated as disabled, and iteration `0` counts as due (`0 % n == 0`).
    pub heartbeat_interval: Option<usize>,
    /// Language of the generated reminder (`"en"` or `"cn"`; anything else selects Chinese).
    pub prompt_language: String,
    /// Whether the reminder is published before every iteration.
    ///
    /// When `false`, only the first iteration of a round (`iteration == 0`) publishes it.
    pub remind_on_every_iteration: bool,
    /// Whether the last assessment's evidence, remaining work and next instruction are included.
    pub include_evidence: bool,
    /// Hard cap in characters on the rendered reminder; `0` means unlimited.
    pub max_section_chars: usize,
}

impl Default for GoalRailConfig {
    /// English reminders, no heartbeat, evidence included, bounded to 2000 characters.
    fn default() -> Self {
        Self {
            heartbeat_interval: None,
            prompt_language: "en".to_string(),
            remind_on_every_iteration: true,
            include_evidence: true,
            max_section_chars: 2000,
        }
    }
}

impl GoalRailConfig {
    /// Returns this config with `interval` as the heartbeat interval.
    #[must_use]
    pub const fn with_heartbeat_interval(mut self, interval: Option<usize>) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    /// Returns this config rendering the reminder in `language`.
    #[must_use]
    pub fn with_prompt_language(mut self, language: impl Into<String>) -> Self {
        self.prompt_language = language.into();
        self
    }

    /// Returns this config with per-iteration reminders toggled.
    #[must_use]
    pub const fn with_remind_on_every_iteration(mut self, enabled: bool) -> Self {
        self.remind_on_every_iteration = enabled;
        self
    }

    /// Returns this config with assessment evidence inclusion toggled.
    #[must_use]
    pub const fn with_include_evidence(mut self, enabled: bool) -> Self {
        self.include_evidence = enabled;
        self
    }

    /// Returns this config with `max_chars` as the reminder cap (`0` means unlimited).
    #[must_use]
    pub const fn with_max_section_chars(mut self, max_chars: usize) -> Self {
        self.max_section_chars = max_chars;
        self
    }
}

/// Keeps the model oriented on the persistent goal and emits `goal.updated` events.
///
/// Dispatch priority is `GOAL_RAIL_PRIORITY` (90), the Python planning/interrupt slot.
pub struct GoalRail {
    store: Arc<GoalStore>,
    config: GoalRailConfig,
}

impl GoalRail {
    /// Builds a rail over `store` with `config` as its tunables.
    #[must_use]
    pub fn new(store: Arc<GoalStore>, config: GoalRailConfig) -> Self {
        Self { store, config }
    }

    /// The tunables of this rail.
    #[must_use]
    pub fn config(&self) -> &GoalRailConfig {
        &self.config
    }

    /// The store this rail reads.
    #[must_use]
    pub fn store(&self) -> &Arc<GoalStore> {
        &self.store
    }

    /// Returns this rail with `config` as its tunables.
    #[must_use]
    pub fn with_config(mut self, config: GoalRailConfig) -> Self {
        self.config = config;
        self
    }

    /// The language the reminder is rendered in.
    fn prompt_language(&self) -> &str {
        &self.config.prompt_language
    }

    /// The iteration index of the firing point; iteration `0` when no task info is attached.
    fn iteration(ctx: &HarnessRailContext) -> usize {
        ctx.task_info().map_or(0, |info| info.iteration)
    }

    /// Whether a heartbeat nudge is due at `iteration`.
    fn heartbeat_due(&self, iteration: usize) -> bool {
        self.config
            .heartbeat_interval
            .is_some_and(|interval| interval > 0 && iteration % interval == 0)
    }

    /// Renders the goal body: heading plus the non-empty `key: value` lines.
    fn render_body(&self, record: &GoalRecord, language: &str) -> String {
        let assessment = record.last_assessment.clone().unwrap_or_default();
        let mut fields: Vec<(&str, Value)> = vec![
            ("objective", json!(record.objective)),
            ("status", json!(record.status.as_str())),
            ("attempt_count", json!(record.attempt_count)),
        ];
        if self.config.include_evidence {
            fields.push(("evidence", json!(assessment.evidence)));
            fields.push(("remaining_work", json!(assessment.remaining_work)));
            fields.push(("next_instruction", json!(assessment.next_instruction)));
        }
        let heading = if is_english(language) {
            "# Current Goal"
        } else {
            "# 当前目标"
        };
        format!("{heading}\n\n{}", render_fields(&fields))
    }

    /// Renders the reminder body plus the optional heartbeat nudge, capped at
    /// [`GoalRailConfig::max_section_chars`].
    fn render_text(&self, record: &GoalRecord, language: &str, iteration: usize) -> String {
        let mut text = self.render_body(record, language);
        if self.heartbeat_due(iteration) {
            text.push_str("\n\n");
            text.push_str(&heartbeat_nudge(language));
        }
        truncate_chars(&text, self.config.max_section_chars).text
    }

    /// Builds the prompt section published before an iteration.
    fn goal_section(&self, record: &GoalRecord, iteration: usize) -> PromptSection {
        let language = self.prompt_language();
        let text = self.render_text(record, language, iteration);
        // One rendering serves both language keys; see the module docs.
        PromptSection::bilingual(GOAL_SECTION_NAME, text.clone(), text, GOAL_SECTION_PRIORITY)
            .with_category(PromptCategory::SystemPrompt)
    }
}

impl fmt::Debug for GoalRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.store.path().display().to_string();
        f.debug_struct("GoalRail")
            .field("store", &path)
            .field("config", &self.config)
            .finish()
    }
}

/// The heartbeat "continue / verify" nudge in `language`.
fn heartbeat_nudge(language: &str) -> String {
    if is_english(language) {
        format!(
            "{GOAL_HEARTBEAT_MARKER} Verify progress before continuing: re-check that the \
             expected changes are actually in place, then call `submit_goal_report` with \
             `continue`, `complete` or `blocked`."
        )
    } else {
        format!(
            "{GOAL_HEARTBEAT_MARKER} 继续之前先验证进展：重新确认预期的改动确实已经生效，然后调用 \
             `submit_goal_report`，把本次尝试报告为 `continue`、`complete` 或 `blocked`。"
        )
    }
}

/// Appends `event` to the queued-events array of `ctx`, never overwriting earlier entries.
fn push_event(ctx: &mut HarnessRailContext, event: &InteractionEvent) {
    let entry = event.to_output_value();
    let channel = ctx
        .extra_mut()
        .entry(GOAL_EVENT_EXTRA_KEY.to_string())
        .or_insert_with(|| json!([]));
    if !channel.is_array() {
        *channel = json!([]);
    }
    if let Some(events) = channel.as_array_mut() {
        events.push(entry);
    }
}

#[async_trait::async_trait]
impl HarnessRail for GoalRail {
    fn name(&self) -> &'static str {
        "goal"
    }

    fn priority(&self) -> i32 {
        GOAL_RAIL_PRIORITY
    }

    async fn before_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let record = match self.store.load().await {
            Ok(Some(record)) => record,
            Ok(None) => return Ok(()),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    path = %self.store.path().display(),
                    "goal record could not be read; skipping the goal reminder"
                );
                return Ok(());
            }
        };
        if !record.is_active() {
            return Ok(());
        }
        let iteration = Self::iteration(ctx);
        if !self.config.remind_on_every_iteration && iteration != 0 {
            return Ok(());
        }
        publish_prompt_section(ctx, &self.goal_section(&record, iteration));
        Ok(())
    }

    async fn after_task_iteration(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let record = match self.store.load().await {
            Ok(Some(record)) => record,
            Ok(None) => return Ok(()),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    path = %self.store.path().display(),
                    "goal record could not be read; skipping the goal event"
                );
                return Ok(());
            }
        };
        if !record.status.is_terminal() {
            return Ok(());
        }
        push_event(ctx, &InteractionEvent::goal_updated(Some(record.to_dict())));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use chrono::{DateTime, TimeZone, Utc};
    use openjiuwen_agent::RailEvent;

    use crate::rails::{prompt_sections, TaskIterationInfo};
    use crate::tools::goal::{GoalAssessment, GoalAssessmentStatus, GoalStore};

    use super::*;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap()
    }

    fn build_rail(dir: &tempfile::TempDir, config: GoalRailConfig) -> (GoalRail, Arc<GoalStore>) {
        let store = GoalStore::new(dir.path().join("goal.json"));
        (
            GoalRail::new(Arc::clone(&store), config),
            Arc::clone(&store),
        )
    }

    fn context(iteration: Option<usize>) -> HarnessRailContext {
        let ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
        match iteration {
            Some(iteration) => ctx.with_task(TaskIterationInfo::new(0, iteration)),
            None => ctx,
        }
    }

    fn only_section(ctx: &HarnessRailContext) -> PromptSection {
        let mut sections = prompt_sections(ctx);
        assert_eq!(sections.len(), 1, "exactly one prompt section");
        sections.remove(0)
    }

    fn queued_events(ctx: &HarnessRailContext) -> Vec<Value> {
        ctx.extra()
            .get(GOAL_EVENT_EXTRA_KEY)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    }

    async fn complete(store: &GoalStore) {
        let assessment =
            GoalAssessment::new(GoalAssessmentStatus::Complete, "tests pass", None, None);
        store
            .update(|record| record.record_progress(&assessment, now()), now())
            .await
            .unwrap()
            .expect("a persisted goal");
    }

    #[test]
    fn name_and_priority_match_the_python_slot() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, _) = build_rail(&dir, GoalRailConfig::default());
        assert_eq!(rail.name(), "goal");
        assert_eq!(rail.priority(), GOAL_RAIL_PRIORITY);
        assert_eq!(GOAL_RAIL_PRIORITY, 90);
    }

    #[test]
    fn event_key_is_stable() {
        assert_eq!(GOAL_EVENT_EXTRA_KEY, "harness.goal_events");
        assert_eq!(GOAL_SECTION_NAME, "goal");
        assert_eq!(GOAL_SECTION_PRIORITY, 55);
    }

    #[test]
    fn config_defaults_and_builders() {
        let default = GoalRailConfig::default();
        assert_eq!(default.heartbeat_interval, None);
        assert_eq!(default.prompt_language, "en");
        assert!(default.remind_on_every_iteration);
        assert!(default.include_evidence);
        assert_eq!(default.max_section_chars, 2000);

        let configured = GoalRailConfig::default()
            .with_heartbeat_interval(Some(3))
            .with_prompt_language("cn")
            .with_remind_on_every_iteration(false)
            .with_include_evidence(false)
            .with_max_section_chars(128);
        assert_eq!(configured.heartbeat_interval, Some(3));
        assert_eq!(configured.prompt_language, "cn");
        assert!(!configured.remind_on_every_iteration);
        assert!(!configured.include_evidence);
        assert_eq!(configured.max_section_chars, 128);
    }

    #[test]
    fn config_round_trips_and_tolerates_partial_input() {
        let config = GoalRailConfig::default().with_heartbeat_interval(Some(3));
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["heartbeat_interval"], json!(3));
        assert_eq!(
            serde_json::from_value::<GoalRailConfig>(encoded).unwrap(),
            config
        );

        let partial: GoalRailConfig =
            serde_json::from_value(json!({"prompt_language": "cn", "unknown": true})).unwrap();
        assert_eq!(partial.prompt_language, "cn");
        assert_eq!(
            partial,
            GoalRailConfig::default().with_prompt_language("cn")
        );
    }

    #[test]
    fn accessors_expose_the_store_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default().with_max_section_chars(10);
        let (rail, store) = build_rail(&dir, config.clone());
        assert!(Arc::ptr_eq(rail.store(), &store));
        assert_eq!(rail.config(), &config);

        let replaced = GoalRailConfig::default().with_prompt_language("cn");
        let rail = rail.with_config(replaced.clone());
        assert_eq!(rail.config(), &replaced);
        let debug = format!("{rail:?}");
        assert!(debug.contains("GoalRail"));
        assert!(debug.contains("cn"));
    }

    #[tokio::test]
    async fn active_goal_publishes_a_section_with_the_objective_and_status() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        store.start("ship the goal rail", now()).await.unwrap();

        let mut ctx = context(Some(0));
        rail.before_task_iteration(&mut ctx).await.unwrap();

        let section = only_section(&ctx);
        assert_eq!(section.name, GOAL_SECTION_NAME);
        assert_eq!(section.priority, GOAL_SECTION_PRIORITY);
        assert_eq!(section.category, Some(PromptCategory::SystemPrompt));
        let text = section.render("en");
        assert!(text.contains("# Current Goal"));
        assert!(text.contains("objective: ship the goal rail"));
        assert!(text.contains("status: active"));
        assert!(text.contains("attempt_count: 0"));
        // One rendering serves both language keys.
        assert_eq!(section.render("cn"), text);
        assert_eq!(queued_events(&ctx).len(), 0);
    }

    #[tokio::test]
    async fn a_chinese_configuration_renders_the_chinese_heading() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default().with_prompt_language("cn");
        let (rail, store) = build_rail(&dir, config);
        store.start("发布目标轨", now()).await.unwrap();

        let mut ctx = context(None);
        rail.before_task_iteration(&mut ctx).await.unwrap();
        let text = only_section(&ctx).render("cn");
        assert!(text.contains("# 当前目标"));
        assert!(text.contains("发布目标轨"));
        assert!(text.contains("status: active"));
    }

    #[tokio::test]
    async fn missing_goal_publishes_nothing_and_emits_no_event() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, _) = build_rail(&dir, GoalRailConfig::default());

        let mut before = context(Some(0));
        rail.before_task_iteration(&mut before).await.unwrap();
        assert_eq!(prompt_sections(&before).len(), 0);
        assert_eq!(queued_events(&before).len(), 0);

        let mut after = context(Some(0));
        rail.after_task_iteration(&mut after).await.unwrap();
        assert_eq!(prompt_sections(&after).len(), 0);
        assert_eq!(queued_events(&after).len(), 0);
    }

    #[tokio::test]
    async fn paused_and_terminal_goals_publish_no_section() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        store.start("ship it", now()).await.unwrap();
        store
            .update(
                |record| record.set_status(crate::tools::goal::GoalStatus::Paused, now()),
                now(),
            )
            .await
            .unwrap();

        let mut paused = context(Some(0));
        rail.before_task_iteration(&mut paused).await.unwrap();
        assert_eq!(prompt_sections(&paused).len(), 0);
        // A paused goal is not terminal, so the after hook stays silent too.
        let mut after_paused = context(Some(0));
        rail.after_task_iteration(&mut after_paused).await.unwrap();
        assert_eq!(queued_events(&after_paused).len(), 0);

        complete(&store).await;
        let mut completed = context(Some(1));
        rail.before_task_iteration(&mut completed).await.unwrap();
        assert_eq!(prompt_sections(&completed).len(), 0);
        assert_eq!(queued_events(&completed).len(), 0);
    }

    #[tokio::test]
    async fn terminal_goal_emits_one_goal_updated_event() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        store.start("ship it", now()).await.unwrap();
        complete(&store).await;

        let mut ctx = context(Some(1));
        rail.after_task_iteration(&mut ctx).await.unwrap();

        let events = queued_events(&ctx);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], json!("goal.updated"));
        assert_eq!(events[0]["index"], json!(0));
        assert_eq!(events[0]["payload"]["goal"]["status"], json!("completed"));
        assert_eq!(events[0]["payload"]["goal"]["objective"], json!("ship it"));
        assert_eq!(
            events[0]["payload"]["goal"]["last_stop_reason"],
            json!("completed")
        );
        assert_eq!(prompt_sections(&ctx).len(), 0);
    }

    #[tokio::test]
    async fn successive_iterations_append_events_instead_of_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        store.start("ship it", now()).await.unwrap();
        complete(&store).await;

        let mut ctx = context(Some(1));
        rail.after_task_iteration(&mut ctx).await.unwrap();
        rail.after_task_iteration(&mut ctx).await.unwrap();

        let events = queued_events(&ctx);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], events[1]);
        assert_eq!(events[1]["payload"]["goal"]["status"], json!("completed"));
    }

    #[tokio::test]
    async fn heartbeat_nudges_every_n_iterations() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default().with_heartbeat_interval(Some(2));
        let (rail, store) = build_rail(&dir, config);
        store.start("ship it", now()).await.unwrap();

        for (iteration, expected) in [(0, true), (1, false), (2, true), (3, false), (4, true)] {
            let mut ctx = context(Some(iteration));
            rail.before_task_iteration(&mut ctx).await.unwrap();
            let text = only_section(&ctx).render("en");
            assert_eq!(
                text.contains(GOAL_HEARTBEAT_MARKER),
                expected,
                "iteration {iteration}"
            );
            if expected {
                assert!(text.contains("Verify progress before continuing"));
                assert!(text.contains("`submit_goal_report`"));
                assert!(text.contains("`continue`, `complete` or `blocked`"));
            }
        }
    }

    #[tokio::test]
    async fn heartbeat_none_or_zero_never_nudges() {
        for interval in [None, Some(0)] {
            let dir = tempfile::tempdir().unwrap();
            let config = GoalRailConfig::default().with_heartbeat_interval(interval);
            let (rail, store) = build_rail(&dir, config);
            store.start("ship it", now()).await.unwrap();

            for iteration in 0..=4 {
                let mut ctx = context(Some(iteration));
                rail.before_task_iteration(&mut ctx).await.unwrap();
                let text = only_section(&ctx).render("en");
                assert!(
                    !text.contains(GOAL_HEARTBEAT_MARKER),
                    "interval {interval:?}, iteration {iteration}"
                );
            }
        }
    }

    #[tokio::test]
    async fn heartbeat_nudge_uses_the_configured_language() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default()
            .with_heartbeat_interval(Some(1))
            .with_prompt_language("cn");
        let (rail, store) = build_rail(&dir, config);
        store.start("发布目标轨", now()).await.unwrap();

        let mut ctx = context(Some(1));
        rail.before_task_iteration(&mut ctx).await.unwrap();
        let text = only_section(&ctx).render("cn");
        assert!(text.contains(GOAL_HEARTBEAT_MARKER));
        assert!(text.contains("继续之前先验证进展"));
        assert!(text.contains("`submit_goal_report`"));
    }

    #[tokio::test]
    async fn corrupt_goal_file_never_aborts_the_loop() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        std::fs::write(store.path(), b"{ not a goal").unwrap();

        let mut before = context(Some(0));
        rail.before_task_iteration(&mut before).await.unwrap();
        assert_eq!(prompt_sections(&before).len(), 0);

        let mut after = context(Some(0));
        rail.after_task_iteration(&mut after).await.unwrap();
        assert_eq!(queued_events(&after).len(), 0);
    }

    #[tokio::test]
    async fn max_section_chars_bounds_the_reminder() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default().with_max_section_chars(60);
        let (rail, store) = build_rail(&dir, config);
        store.start("x".repeat(400), now()).await.unwrap();

        let mut ctx = context(Some(0));
        rail.before_task_iteration(&mut ctx).await.unwrap();
        let text = only_section(&ctx).render("en");
        assert!(text.contains("output truncated"));
        assert!(text.chars().count() < 400);
        assert!(text.chars().count() <= 60 + 80, "{text}");
    }

    #[tokio::test]
    async fn zero_max_section_chars_means_unlimited() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default().with_max_section_chars(0);
        let (rail, store) = build_rail(&dir, config);
        store.start("y".repeat(400), now()).await.unwrap();

        let mut ctx = context(Some(0));
        rail.before_task_iteration(&mut ctx).await.unwrap();
        let text = only_section(&ctx).render("en");
        assert!(!text.contains("output truncated"));
        assert!(text.contains(&"y".repeat(400)));
    }

    #[tokio::test]
    async fn first_iteration_only_publishes_when_per_iteration_reminders_are_off() {
        let dir = tempfile::tempdir().unwrap();
        let config = GoalRailConfig::default().with_remind_on_every_iteration(false);
        let (rail, store) = build_rail(&dir, config);
        store.start("ship it", now()).await.unwrap();

        let mut first = context(Some(0));
        rail.before_task_iteration(&mut first).await.unwrap();
        assert_eq!(prompt_sections(&first).len(), 1);

        let mut second = context(Some(1));
        rail.before_task_iteration(&mut second).await.unwrap();
        assert_eq!(prompt_sections(&second).len(), 0);
    }

    #[tokio::test]
    async fn evidence_can_be_omitted_from_the_reminder() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        store.start("ship it", now()).await.unwrap();
        let assessment = GoalAssessment::new(
            GoalAssessmentStatus::Continue,
            "half way",
            Some("wire the rail".to_string()),
            Some("run the suite".to_string()),
        );
        store
            .update(|record| record.record_progress(&assessment, now()), now())
            .await
            .unwrap();

        let mut with = context(Some(0));
        rail.before_task_iteration(&mut with).await.unwrap();
        let text = only_section(&with).render("en");
        assert!(text.contains("evidence: half way"));
        assert!(text.contains("remaining_work: wire the rail"));
        assert!(text.contains("next_instruction: run the suite"));

        let (without_evidence, _) =
            build_rail(&dir, GoalRailConfig::default().with_include_evidence(false));
        let mut without = context(Some(0));
        without_evidence
            .before_task_iteration(&mut without)
            .await
            .unwrap();
        let text = only_section(&without).render("en");
        assert!(!text.contains("evidence:"));
        assert!(!text.contains("remaining_work:"));
        assert!(!text.contains("next_instruction:"));
        assert!(text.contains("objective: ship it"));
    }

    #[tokio::test]
    async fn publishing_into_a_non_array_event_channel_resets_it() {
        let dir = tempfile::tempdir().unwrap();
        let (rail, store) = build_rail(&dir, GoalRailConfig::default());
        store.start("ship it", now()).await.unwrap();
        complete(&store).await;

        let mut ctx = context(Some(0));
        ctx.extra_mut()
            .insert(GOAL_EVENT_EXTRA_KEY.to_string(), json!({"stale": true}));
        rail.after_task_iteration(&mut ctx).await.unwrap();

        let events = queued_events(&ctx);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], json!("goal.updated"));
    }
}
