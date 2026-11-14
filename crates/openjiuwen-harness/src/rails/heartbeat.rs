//! The heartbeat rail: injects the heartbeat instructions for heartbeat runs.
//!
//! Ports `openjiuwen/harness/rails/heartbeat_rail.py` and
//! `openjiuwen/harness/prompts/sections/heartbeat.py`. The Python rail watches
//! `run_kind == "heartbeat"` on the invocation context and injects one system
//! prompt section explaining how a heartbeat turn must answer
//! (`HEARTBEAT_OK` when the `<heartbeat_user_task>` tags are empty, the task
//! body's own result otherwise).
//!
//! The Rust rail keeps that firing condition ([`HEARTBEAT_RUN_KIND`]) and adds a
//! periodic cadence on top of it, because a host that drives the agent itself has
//! no scheduler to mark a run as a heartbeat run:
//!
//! * an explicitly marked heartbeat run always injects
//!   (`force_on_heartbeat_run`), exactly like Python;
//! * otherwise a `(agent_id, session_id)` pair becomes due once
//!   `interval_ms` have elapsed since its baseline.
//!
//! The first observation of a pair only records the baseline and never fires:
//! a fresh run must not receive heartbeat text on its first model call. Use
//! [`HeartbeatRail::reset_session`] to restart the cadence of one pair.
//!
//! Every firing increments that pair's sequence and queues a
//! [`HeartbeatEvent`] in a bounded, oldest-evicted-first queue, so a host can
//! observe the cadence without installing a callback. The queue and the
//! per-pair bookkeeping live in this rail instance (never in process-global
//! state) and are only read through short `parking_lot` critical sections.
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::prompts::{PromptCategory, PromptSection};

use super::{publish_prompt_section, rail_priority, HarnessRail, HarnessRailContext};

/// Name of the heartbeat prompt section.
pub const HEARTBEAT_SECTION: &str = "heartbeat";

/// Key in [`openjiuwen_agent::RailContext::extra`] carrying the run kind.
pub const RUN_KIND_EXTRA_KEY: &str = "run_kind";

/// Run-kind value marking a run as a scheduled heartbeat run.
pub const HEARTBEAT_RUN_KIND: &str = "heartbeat";

/// Render priority of the heartbeat section (Python `priority=78`).
pub const HEARTBEAT_SECTION_PRIORITY: i32 = 78;

/// Default period of the periodic heartbeat path, in milliseconds.
pub const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 60_000;

/// Default capacity of the queued-event buffer.
pub const DEFAULT_HEARTBEAT_EVENT_CAPACITY: usize = 64;

/// The Chinese heartbeat system prompt, verbatim from the Python SDK.
const HEARTBEAT_PROMPT_CN: &str = r"
## 心跳检测

判定规则：
1. 若 `<heartbeat_user_task>` 与 `</heartbeat_user_task>` 之间仅有空白（含空行）或完全为空：视为**无心跳用户任务**。你必须且仅能输出一行，内容**精确**为 `HEARTBEAT_OK`（不要解释、不要前后缀、不要 Markdown、不要工具调用说明）。
2. 若 `<heartbeat_user_task>` 与 `</heartbeat_user_task>` 之间存在**任意非空白字符**：该段即用户下发的心跳任务正文。你必须**完整阅读并执行**其中的指令并给出**直接回答**；**禁止**在回复中出现 `HEARTBEAT_OK` 四字（含单独一行、前缀、后缀、用标点或破折号拼接，例如 `HEARTBEAT_OK — …` 一律视为违规）。

系统**仅在**满足上一条规则 1（标签内无任务）时，才把单独一行的 `HEARTBEAT_OK` 视为心跳确认；有任务时**不得**为了「确认心跳」而输出或附带 `HEARTBEAT_OK`。
**禁止**用「心跳任务已完成」「任务已完成 ✓」「已处理」「无新内容」「安静待着」等**状态话术**代替任务正文所要求的**具体可核验输出**（若正文要求输出某文本，回复中必须出现该文本本身，而不是完成声明）。

重要约束：
- 每一轮心跳调用都是独立调度，只要 `<heartbeat_user_task>` 标签内有非空白任务正文，你就必须**当场**按正文完成指令所要求的动作或输出。**禁止**以「上一轮刚执行过」等理由省略执行或把「记录」当成完成——**记录不等于执行**。
- 若需修改 HEARTBEAT.md 文件，禁止给原本没有 <!-- --> 注释的内容添加注释标记
- 非注释文本仅可在用户明确要求时修改或删除，否则必须保持原样
- 心跳执行结果必须直接返回；除非心跳内容明确要求更新 HEARTBEAT.md，不要写入 daily memory 或其他记忆文件
";

/// The English heartbeat system prompt, verbatim from the Python SDK.
const HEARTBEAT_PROMPT_EN: &str = r"
## Heartbeat

Decision rules:
1. If between `<heartbeat_user_task>` and `</heartbeat_user_task>` there is only whitespace (including blank lines) or nothing: treat as **no heartbeat user task**. You MUST output exactly one line whose content is **precisely** `HEARTBEAT_OK` (no explanation, no prefix/suffix, no Markdown, no tool narration).
2. If there is **any** non-whitespace character between `<heartbeat_user_task>` and `</heartbeat_user_task>`: that span is the **user-issued heartbeat task body**. You MUST read and carry out the instructions in full and reply **directly**; the substring `HEARTBEAT_OK` MUST NOT appear anywhere in your reply (not alone, not as a prefix/suffix, not glued with punctuation or em dashes—e.g. `HEARTBEAT_OK — …` is forbidden).

The system treats a single line of exactly `HEARTBEAT_OK` **only** under rule 1 (empty task) as the heartbeat acknowledgment; when there is a task body, you MUST NOT emit or append `HEARTBEAT_OK` “to confirm the heartbeat”.
You MUST NOT replace substantive output with status-only phrases such as “heartbeat task completed”, “task done ✓”, “nothing new”, “stay quiet”, etc. If the body asks for specific text, that text MUST appear in the reply itself, not a declaration that you completed it.

Important Constraints:
- Each heartbeat invocation is scheduled independently; whenever the `<heartbeat_user_task>` tags contain a non-empty task body, you MUST **on the spot** complete whatever actions or outputs the body requires. You MUST NOT skip execution or treat “logging/recording” as completion with excuses such as “already executed last round”—**recording is not execution**.
- When modifying HEARTBEAT.md, DO NOT add <!-- --> comment markers to content that originally had no such markers
- Non-commented text may only be modified or deleted when explicitly requested by the user; otherwise preserve it as-is
- Return heartbeat execution results directly; unless heartbeat content explicitly asks you to update HEARTBEAT.md, do not write them to daily memory or other memory files
";

/// The heartbeat system prompt for `language`.
///
/// `"en"` yields the English text and any other value the Chinese one — the
/// fallback of Python's `HEARTBEAT_SYSTEM_PROMPT.get(language, …)`. The lookup
/// is exact, so `"EN"` falls back too.
#[must_use]
pub fn heartbeat_prompt(language: &str) -> &'static str {
    if language == "en" {
        HEARTBEAT_PROMPT_EN
    } else {
        HEARTBEAT_PROMPT_CN
    }
}

/// Builds the heartbeat prompt section for `language`.
///
/// The section is named [`HEARTBEAT_SECTION`], has priority
/// [`HEARTBEAT_SECTION_PRIORITY`] and a [`PromptCategory::SystemPrompt`] group.
/// Its content holds exactly one entry whose key is the requested `language`
/// (Python keys the content the same way, so an unrecognised language carries
/// the Chinese text under its own key).
#[must_use]
pub fn build_heartbeat_section(language: &str) -> PromptSection {
    let mut content = BTreeMap::new();
    content.insert(language.to_string(), heartbeat_prompt(language).to_string());
    PromptSection::new(HEARTBEAT_SECTION, content, HEARTBEAT_SECTION_PRIORITY)
        .with_category(PromptCategory::SystemPrompt)
}

/// Source of the millisecond timestamps recorded by [`HeartbeatRail`].
///
/// The default ([`default_heartbeat_clock`]) is monotonic; a test injects a
/// counter through [`HeartbeatRail::with_clock`] instead.
pub type HeartbeatClock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Anchor of the default monotonic clock: the first call is the zero point.
static ANCHOR: OnceLock<Instant> = OnceLock::new();

/// The default [`HeartbeatClock`]: monotonic milliseconds since the process
/// started reading heartbeats.
#[must_use]
pub fn default_heartbeat_clock() -> HeartbeatClock {
    Arc::new(|| {
        let anchor = *ANCHOR.get_or_init(Instant::now);
        u64::try_from(anchor.elapsed().as_millis()).unwrap_or(u64::MAX)
    })
}

/// Tunables of [`HeartbeatRail`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HeartbeatRailConfig {
    /// Period of the periodic path in milliseconds; `0` disables it.
    pub interval_ms: u64,
    /// Prompt language; `"en"` selects the English text, anything else Chinese.
    pub language: String,
    /// Capacity of the queued-event buffer; `0` keeps every event.
    pub event_capacity: usize,
    /// Whether a run marked [`HEARTBEAT_RUN_KIND`] always fires.
    pub force_on_heartbeat_run: bool,
    /// Whether a firing publishes the heartbeat prompt section.
    pub publish_prompt_section: bool,
}

impl Default for HeartbeatRailConfig {
    /// A one-minute cadence, Chinese prompts, 64 kept events, both switches on.
    fn default() -> Self {
        Self {
            interval_ms: DEFAULT_HEARTBEAT_INTERVAL_MS,
            language: "cn".to_string(),
            event_capacity: DEFAULT_HEARTBEAT_EVENT_CAPACITY,
            force_on_heartbeat_run: true,
            publish_prompt_section: true,
        }
    }
}

impl HeartbeatRailConfig {
    /// Returns this config with `interval_ms` as its period (`0` disables it).
    #[must_use]
    pub const fn with_interval_ms(mut self, interval_ms: u64) -> Self {
        self.interval_ms = interval_ms;
        self
    }

    /// Returns this config rendering `language` prompts.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Returns this config keeping at most `capacity` events (`0` unbounded).
    #[must_use]
    pub const fn with_event_capacity(mut self, capacity: usize) -> Self {
        self.event_capacity = capacity;
        self
    }

    /// Returns this config with the marked-run path toggled.
    #[must_use]
    pub const fn with_force_on_heartbeat_run(mut self, enabled: bool) -> Self {
        self.force_on_heartbeat_run = enabled;
        self
    }

    /// Returns this config with prompt-section publishing toggled.
    #[must_use]
    pub const fn with_publish_prompt_section(mut self, enabled: bool) -> Self {
        self.publish_prompt_section = enabled;
        self
    }
}

/// Why a [`HeartbeatEvent`] fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeartbeatReason {
    /// The periodic cadence became due.
    Scheduled,
    /// The run was explicitly marked [`HEARTBEAT_RUN_KIND`].
    RunKind,
}

/// One heartbeat firing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatEvent {
    /// Id of the agent the firing belongs to.
    pub agent_id: String,
    /// Id of the session the firing belongs to (empty without a session).
    pub session_id: String,
    /// Per-`(agent, session)` counter, `1`-based.
    pub sequence: u64,
    /// Clock reading when the firing happened, in milliseconds.
    pub fired_at_ms: u64,
    /// Whether the cadence or the run kind triggered the firing.
    pub reason: HeartbeatReason,
}

/// Bookkeeping of one `(agent_id, session_id)` pair.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HeartbeatState {
    /// Timestamp the current cadence is measured from.
    baseline_ms: u64,
    /// Timestamp of the last firing, `None` before the first one.
    last_fired_ms: Option<u64>,
    /// Number of firings of this pair, `0` before the first one.
    sequence: u64,
}

/// Injects the heartbeat instructions when a run is a heartbeat run or due.
///
/// Dispatch priority is [`rail_priority::HEARTBEAT`] (80), the slot of the
/// Python heartbeat rail. The rail publishes
/// [`build_heartbeat_section`] on the firing model call and records the firing
/// in a bounded event queue; it never fails and never aborts a run.
pub struct HeartbeatRail {
    /// Tunables, immutable after construction.
    config: HeartbeatRailConfig,
    /// Source of every timestamp this rail records.
    clock: HeartbeatClock,
    /// Cadence and sequence per `(agent_id, session_id)` pair.
    state: Mutex<BTreeMap<(String, String), HeartbeatState>>,
    /// Queued events, oldest first.
    events: Mutex<VecDeque<HeartbeatEvent>>,
}

impl HeartbeatRail {
    /// Builds a rail with `config` and the default monotonic clock.
    #[must_use]
    pub fn new(config: HeartbeatRailConfig) -> Self {
        Self {
            config,
            clock: default_heartbeat_clock(),
            state: Mutex::new(BTreeMap::new()),
            events: Mutex::new(VecDeque::new()),
        }
    }

    /// Returns this rail reading time from `clock`.
    #[must_use]
    pub fn with_clock(mut self, clock: HeartbeatClock) -> Self {
        self.clock = clock;
        self
    }

    /// The tunables of this rail.
    #[must_use]
    pub fn config(&self) -> &HeartbeatRailConfig {
        &self.config
    }

    /// The current clock reading, in milliseconds.
    fn now_ms(&self) -> u64 {
        (self.clock)()
    }

    /// Whether the periodic path is due for `(agent_id, session_id)` right now,
    /// without recording anything.
    ///
    /// A pair with no baseline yet is never due; neither is any pair when
    /// `interval_ms` is `0`.
    #[must_use]
    pub fn is_due(&self, agent_id: &str, session_id: &str) -> bool {
        let now = self.now_ms();
        let key = (agent_id.to_string(), session_id.to_string());
        let baseline = self.state.lock().get(&key).map(|state| state.baseline_ms);
        baseline.is_some_and(|baseline| self.periodic_due(now, baseline))
    }

    /// Whether `now` is at least `interval_ms` past `baseline`.
    fn periodic_due(&self, now: u64, baseline: u64) -> bool {
        self.config.interval_ms > 0 && now.saturating_sub(baseline) >= self.config.interval_ms
    }

    /// Decides whether a heartbeat fires for `(agent_id, session_id)` and
    /// records it.
    ///
    /// A run whose `run_kind` is [`HEARTBEAT_RUN_KIND`] fires when
    /// `force_on_heartbeat_run` is set; otherwise the pair fires once its
    /// cadence is due. The first observation of a pair only records the
    /// baseline (except for a marked heartbeat run, which fires immediately) and
    /// returns `None`.
    ///
    /// Returns the queued event, or `None` when nothing fired.
    #[must_use]
    pub fn observe(
        &self,
        agent_id: &str,
        session_id: &str,
        run_kind: Option<&str>,
    ) -> Option<HeartbeatEvent> {
        let now = self.now_ms();
        let forced = self.config.force_on_heartbeat_run
            && run_kind.is_some_and(|kind| kind == HEARTBEAT_RUN_KIND);
        let key = (agent_id.to_string(), session_id.to_string());

        let (sequence, reason) = {
            let mut state = self.state.lock();
            let due = state
                .get(&key)
                .is_some_and(|current| self.periodic_due(now, current.baseline_ms));
            if !forced && !due {
                state.entry(key).or_insert(HeartbeatState {
                    baseline_ms: now,
                    ..HeartbeatState::default()
                });
                return None;
            }
            let entry = state.entry(key.clone()).or_insert(HeartbeatState {
                baseline_ms: now,
                ..HeartbeatState::default()
            });
            entry.baseline_ms = now;
            entry.last_fired_ms = Some(now);
            entry.sequence = entry.sequence.saturating_add(1);
            let reason = if forced {
                HeartbeatReason::RunKind
            } else {
                HeartbeatReason::Scheduled
            };
            (entry.sequence, reason)
        };

        let event = HeartbeatEvent {
            agent_id: key.0,
            session_id: key.1,
            sequence,
            fired_at_ms: now,
            reason,
        };
        tracing::debug!(
            agent_id = %event.agent_id,
            session_id = %event.session_id,
            sequence = event.sequence,
            reason = ?event.reason,
            "heartbeat fired"
        );
        self.push_event(event.clone());
        Some(event)
    }

    /// Appends `event`, evicting the oldest queued events past the capacity.
    fn push_event(&self, event: HeartbeatEvent) {
        let mut events = self.events.lock();
        if self.config.event_capacity > 0 && events.len() >= self.config.event_capacity {
            let excess = events
                .len()
                .saturating_add(1)
                .saturating_sub(self.config.event_capacity)
                .min(events.len());
            events.drain(..excess);
        }
        events.push_back(event);
    }

    /// Snapshot of the queued events, oldest first.
    #[must_use]
    pub fn events(&self) -> Vec<HeartbeatEvent> {
        self.events.lock().iter().cloned().collect()
    }

    /// Returns the queued events, oldest first, and clears the queue.
    #[must_use]
    pub fn drain_events(&self) -> Vec<HeartbeatEvent> {
        let mut events = self.events.lock();
        events.drain(..).collect()
    }

    /// Timestamp of the last firing of `(agent_id, session_id)`, when it fired.
    #[must_use]
    pub fn last_fired_ms(&self, agent_id: &str, session_id: &str) -> Option<u64> {
        let key = (agent_id.to_string(), session_id.to_string());
        self.state
            .lock()
            .get(&key)
            .and_then(|state| state.last_fired_ms)
    }

    /// Forgets the cadence and sequence of `(agent_id, session_id)`.
    ///
    /// The next observation of that pair records a fresh baseline, so the
    /// cadence restarts without firing. Queued events are kept.
    pub fn reset_session(&self, agent_id: &str, session_id: &str) {
        let key = (agent_id.to_string(), session_id.to_string());
        self.state.lock().remove(&key);
    }

    /// Forgets the cadence and sequence of every pair.
    ///
    /// Queued events are kept; use [`HeartbeatRail::drain_events`] to clear them.
    pub fn reset_all(&self) {
        self.state.lock().clear();
    }
}

impl Default for HeartbeatRail {
    /// A rail with the default config and clock.
    fn default() -> Self {
        Self::new(HeartbeatRailConfig::default())
    }
}

impl fmt::Debug for HeartbeatRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeartbeatRail")
            .field("config", &self.config)
            .field("clock", &"<heartbeat clock>")
            .field("state", &self.state.lock().len())
            .field("events", &self.events.lock().len())
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for HeartbeatRail {
    fn name(&self) -> &'static str {
        HEARTBEAT_SECTION
    }

    fn priority(&self) -> i32 {
        rail_priority::HEARTBEAT
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let agent_id = ctx.base.agent_id.clone();
        let session_id = ctx
            .base
            .session
            .as_ref()
            .map(|session| session.id().to_string())
            .unwrap_or_default();
        let run_kind = ctx
            .extra()
            .get(RUN_KIND_EXTRA_KEY)
            .and_then(Value::as_str)
            .map(str::to_string);

        if self
            .observe(&agent_id, &session_id, run_kind.as_deref())
            .is_none()
        {
            return Ok(());
        }
        if self.config.publish_prompt_section {
            publish_prompt_section(ctx, &build_heartbeat_section(&self.config.language));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::sync::atomic::{AtomicU64, Ordering};

    use openjiuwen_agent::RailEvent;
    use openjiuwen_session::AgentSession;
    use serde_json::json;

    use super::*;
    use crate::rails::prompt_sections;

    /// A clock a test advances by hand.
    fn manual_clock() -> (HeartbeatClock, Arc<AtomicU64>) {
        let now = Arc::new(AtomicU64::new(0));
        let handle = Arc::clone(&now);
        (Arc::new(move || handle.load(Ordering::SeqCst)), now)
    }

    /// A `BeforeModelCall` context for `agent_id`/`session_id` with `run_kind`.
    fn ctx_for(event: RailEvent, agent_id: &str, session_id: &str) -> HarnessRailContext {
        let mut ctx = HarnessRailContext::for_event(event);
        ctx.base.agent_id = agent_id.to_string();
        ctx.base.session = Some(Arc::new(AgentSession::new(session_id)));
        ctx
    }

    #[test]
    fn section_is_named_heartbeat_with_the_python_priority_and_texts() {
        let en = build_heartbeat_section("en");
        assert_eq!(en.name, HEARTBEAT_SECTION);
        assert_eq!(en.name, "heartbeat");
        assert_eq!(en.priority, HEARTBEAT_SECTION_PRIORITY);
        assert_eq!(en.priority, 78);
        assert_eq!(en.category, Some(PromptCategory::SystemPrompt));
        assert_eq!(en.carrier, "system_message");
        assert_eq!(en.content.len(), 1);
        assert_eq!(
            en.content.get("en").map(String::as_str),
            Some(HEARTBEAT_PROMPT_EN)
        );
        assert!(HEARTBEAT_PROMPT_EN.contains("## Heartbeat"));
        assert!(HEARTBEAT_PROMPT_EN.contains("HEARTBEAT_OK"));
        assert!(HEARTBEAT_PROMPT_EN.contains("<heartbeat_user_task>"));
        assert!(HEARTBEAT_PROMPT_EN.contains("</heartbeat_user_task>"));

        let cn = build_heartbeat_section("cn");
        assert_eq!(
            cn.content.get("cn").map(String::as_str),
            Some(HEARTBEAT_PROMPT_CN)
        );
        assert_eq!(cn.priority, HEARTBEAT_SECTION_PRIORITY);
        assert!(HEARTBEAT_PROMPT_CN.contains("## 心跳检测"));
        assert!(HEARTBEAT_PROMPT_CN.contains("HEARTBEAT_OK"));
        assert!(HEARTBEAT_PROMPT_CN.contains("<heartbeat_user_task>"));
        assert!(HEARTBEAT_PROMPT_CN.contains("</heartbeat_user_task>"));
        assert_ne!(HEARTBEAT_PROMPT_CN, HEARTBEAT_PROMPT_EN);
    }

    #[test]
    fn unknown_languages_fall_back_to_chinese() {
        assert_eq!(heartbeat_prompt("en"), HEARTBEAT_PROMPT_EN);
        assert_eq!(heartbeat_prompt("cn"), HEARTBEAT_PROMPT_CN);
        assert_eq!(heartbeat_prompt("EN"), HEARTBEAT_PROMPT_CN);
        assert_eq!(heartbeat_prompt("klingon"), HEARTBEAT_PROMPT_CN);
        assert_eq!(heartbeat_prompt(""), HEARTBEAT_PROMPT_CN);

        // Python keys the content by the requested language, text or fallback.
        let other = build_heartbeat_section("klingon");
        assert_eq!(
            other.content.get("klingon").map(String::as_str),
            Some(HEARTBEAT_PROMPT_CN)
        );
        assert_eq!(other.priority, HEARTBEAT_SECTION_PRIORITY);
    }

    #[test]
    fn the_default_clock_is_monotonic() {
        let clock = default_heartbeat_clock();
        let first = clock();
        let second = clock();
        assert!(second >= first);
    }

    #[tokio::test]
    async fn a_heartbeat_run_publishes_one_section_and_one_event() {
        let (clock, now) = manual_clock();
        now.store(5_000, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default()).with_clock(clock);
        let mut ctx = ctx_for(RailEvent::BeforeModelCall, "agent-a", "session-a");
        ctx.extra_mut()
            .insert(RUN_KIND_EXTRA_KEY.to_string(), json!(HEARTBEAT_RUN_KIND));

        rail.before_model_call(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, HEARTBEAT_SECTION);
        assert_eq!(sections[0].priority, HEARTBEAT_SECTION_PRIORITY);
        assert_eq!(sections[0].category, Some(PromptCategory::SystemPrompt));
        assert_eq!(
            sections[0].content.get("cn").map(String::as_str),
            Some(HEARTBEAT_PROMPT_CN)
        );

        let events = rail.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].agent_id, "agent-a");
        assert_eq!(events[0].session_id, "session-a");
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[0].fired_at_ms, 5_000);
        assert_eq!(events[0].reason, HeartbeatReason::RunKind);
        assert_eq!(rail.last_fired_ms("agent-a", "session-a"), Some(5_000));
    }

    #[tokio::test]
    async fn a_context_without_a_session_uses_an_empty_session_id() {
        let (clock, now) = manual_clock();
        now.store(7, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default()).with_clock(clock);
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.agent_id = "agent-a".to_string();
        ctx.extra_mut()
            .insert(RUN_KIND_EXTRA_KEY.to_string(), json!(HEARTBEAT_RUN_KIND));

        rail.before_model_call(&mut ctx).await.unwrap();

        let events = rail.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].session_id, "");
    }

    #[tokio::test]
    async fn the_marked_run_path_can_be_disabled() {
        let (clock, now) = manual_clock();
        now.store(1_000, Ordering::SeqCst);
        let config = HeartbeatRailConfig::default().with_force_on_heartbeat_run(false);
        let rail = HeartbeatRail::new(config).with_clock(clock);
        let mut ctx = ctx_for(RailEvent::BeforeModelCall, "agent-a", "session-a");
        ctx.extra_mut()
            .insert(RUN_KIND_EXTRA_KEY.to_string(), json!(HEARTBEAT_RUN_KIND));

        rail.before_model_call(&mut ctx).await.unwrap();

        assert!(prompt_sections(&ctx).is_empty());
        assert!(rail.events().is_empty());
        assert!(rail.last_fired_ms("agent-a", "session-a").is_none());
    }

    #[test]
    fn a_non_heartbeat_run_kind_never_forces_a_firing() {
        let (clock, now) = manual_clock();
        now.store(1_000, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default()).with_clock(clock);

        assert!(rail
            .observe("agent-a", "session-a", Some("interactive"))
            .is_none());
        assert!(rail.observe("agent-a", "session-a", None).is_none());
        assert!(rail.events().is_empty());
    }

    #[test]
    fn the_periodic_cadence_uses_the_injected_clock() {
        let (clock, now) = manual_clock();
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_interval_ms(1_000))
            .with_clock(clock);

        // First observation only records the baseline.
        now.store(1_000, Ordering::SeqCst);
        assert!(rail.observe("agent-a", "session-a", None).is_none());
        assert!(rail.last_fired_ms("agent-a", "session-a").is_none());
        assert!(!rail.is_due("agent-a", "session-a"));
        assert!(rail.events().is_empty());

        // One millisecond short of the period: still nothing.
        now.store(1_999, Ordering::SeqCst);
        assert!(rail.observe("agent-a", "session-a", None).is_none());
        assert!(!rail.is_due("agent-a", "session-a"));

        // Exactly one period after the baseline: fires, sequence 1.
        now.store(2_000, Ordering::SeqCst);
        assert!(rail.is_due("agent-a", "session-a"));
        let first = rail.observe("agent-a", "session-a", None).unwrap();
        assert_eq!(first.reason, HeartbeatReason::Scheduled);
        assert_eq!(first.sequence, 1);
        assert_eq!(first.fired_at_ms, 2_000);
        assert_eq!(rail.last_fired_ms("agent-a", "session-a"), Some(2_000));
        assert!(!rail.is_due("agent-a", "session-a"));

        // Below the next period: nothing.
        now.store(2_999, Ordering::SeqCst);
        assert!(rail.observe("agent-a", "session-a", None).is_none());
        assert!(!rail.is_due("agent-a", "session-a"));

        // One more period: sequence 2.
        now.store(3_000, Ordering::SeqCst);
        let second = rail.observe("agent-a", "session-a", None).unwrap();
        assert_eq!(second.sequence, 2);
        assert_eq!(second.reason, HeartbeatReason::Scheduled);
        assert_eq!(second.fired_at_ms, 3_000);
        assert_eq!(rail.events().len(), 2);
    }

    #[test]
    fn a_zero_interval_disables_the_periodic_path() {
        let (clock, now) = manual_clock();
        now.store(1_000, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_interval_ms(0))
            .with_clock(clock);

        assert!(rail.observe("agent-a", "session-a", None).is_none());
        now.store(10_000_000, Ordering::SeqCst);
        assert!(!rail.is_due("agent-a", "session-a"));
        assert!(rail.observe("agent-a", "session-a", None).is_none());
        assert!(rail.events().is_empty());

        // The marked-run path is independent of the interval.
        assert!(rail
            .observe("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND))
            .is_some());
        assert_eq!(rail.events().len(), 1);
    }

    #[test]
    fn the_marked_run_path_takes_precedence_over_the_cadence() {
        let (clock, now) = manual_clock();
        now.store(1_000, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_interval_ms(1_000))
            .with_clock(clock);

        // Baseline plus a marked run at the same timestamp: the run kind wins.
        let event = rail
            .observe("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND))
            .unwrap();
        assert_eq!(event.reason, HeartbeatReason::RunKind);
        assert_eq!(event.sequence, 1);
        assert_eq!(event.fired_at_ms, 1_000);
        assert!(!rail.is_due("agent-a", "session-a"));
    }

    #[test]
    fn session_pairs_are_isolated() {
        let (clock, now) = manual_clock();
        now.store(1_000, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_interval_ms(1_000))
            .with_clock(clock);

        assert!(rail.observe("agent-a", "session-1", None).is_none());
        now.store(1_500, Ordering::SeqCst);
        assert!(rail.observe("agent-a", "session-2", None).is_none());
        assert!(rail.observe("agent-b", "session-1", None).is_none());

        // Only the first pair is due.
        now.store(2_000, Ordering::SeqCst);
        assert!(rail.is_due("agent-a", "session-1"));
        assert!(!rail.is_due("agent-a", "session-2"));
        assert!(!rail.is_due("agent-b", "session-1"));
        let first = rail.observe("agent-a", "session-1", None).unwrap();
        assert_eq!(first.sequence, 1);
        assert!(rail.observe("agent-a", "session-2", None).is_none());

        // Each pair counts its own firings.
        now.store(2_500, Ordering::SeqCst);
        let second = rail.observe("agent-a", "session-2", None).unwrap();
        assert_eq!(second.sequence, 1);
        assert_eq!(second.session_id, "session-2");
        now.store(3_000, Ordering::SeqCst);
        assert_eq!(
            rail.observe("agent-a", "session-1", None).unwrap().sequence,
            2
        );
        assert_eq!(
            rail.observe("agent-b", "session-1", None).unwrap().sequence,
            1
        );
        assert_eq!(rail.events().len(), 4);
    }

    #[test]
    fn a_bounded_queue_keeps_the_newest_events() {
        let (clock, now) = manual_clock();
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_event_capacity(1))
            .with_clock(clock);

        for tick in [1_u64, 2, 3] {
            now.store(tick, Ordering::SeqCst);
            assert!(rail
                .observe("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND))
                .is_some());
        }
        let events = rail.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 3);
        assert_eq!(events[0].fired_at_ms, 3);
    }

    #[test]
    fn a_zero_capacity_queue_keeps_every_event() {
        let (clock, now) = manual_clock();
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_event_capacity(0))
            .with_clock(clock);

        for tick in 1..=5_u64 {
            now.store(tick, Ordering::SeqCst);
            assert!(rail
                .observe("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND))
                .is_some());
        }
        let events = rail.events();
        let sequences: Vec<u64> = events.iter().map(|event| event.sequence).collect();
        assert_eq!(sequences, vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn disabling_publication_still_queues_events() {
        let (clock, now) = manual_clock();
        now.store(900, Ordering::SeqCst);
        let config = HeartbeatRailConfig::default().with_publish_prompt_section(false);
        let rail = HeartbeatRail::new(config).with_clock(clock);
        let mut ctx = ctx_for(RailEvent::BeforeModelCall, "agent-a", "session-a");
        ctx.extra_mut()
            .insert(RUN_KIND_EXTRA_KEY.to_string(), json!(HEARTBEAT_RUN_KIND));

        rail.before_model_call(&mut ctx).await.unwrap();

        assert!(prompt_sections(&ctx).is_empty());
        assert_eq!(rail.events().len(), 1);
        assert_eq!(rail.events()[0].sequence, 1);
    }

    #[tokio::test]
    async fn a_non_firing_model_call_publishes_nothing() {
        let (clock, now) = manual_clock();
        now.store(100, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default()).with_clock(clock);
        let mut ctx = ctx_for(RailEvent::BeforeModelCall, "agent-a", "session-a");

        rail.before_model_call(&mut ctx).await.unwrap();

        assert!(prompt_sections(&ctx).is_empty());
        assert!(rail.events().is_empty());
    }

    #[test]
    fn events_is_a_snapshot_and_drain_clears() {
        let (clock, now) = manual_clock();
        now.store(1, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default()).with_clock(clock);
        assert!(rail.drain_events().is_empty());

        assert!(rail
            .observe("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND))
            .is_some());
        let snapshot = rail.events();
        assert_eq!(snapshot.len(), 1);
        // Repeated reads see the same queue; neither drains it.
        assert_eq!(rail.events(), snapshot);

        let drained = rail.drain_events();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].sequence, 1);
        assert!(rail.events().is_empty());
        assert!(rail.drain_events().is_empty());
    }

    #[test]
    fn reset_session_restarts_only_that_pair() {
        let (clock, now) = manual_clock();
        now.store(1_000, Ordering::SeqCst);
        let rail = HeartbeatRail::new(HeartbeatRailConfig::default().with_interval_ms(1_000))
            .with_clock(clock);

        assert!(rail.observe("agent-a", "session-1", None).is_none());
        assert!(rail.observe("agent-a", "session-2", None).is_none());
        now.store(2_000, Ordering::SeqCst);
        assert!(rail.observe("agent-a", "session-1", None).is_some());
        assert!(rail.observe("agent-a", "session-2", None).is_some());
        assert_eq!(rail.last_fired_ms("agent-a", "session-1"), Some(2_000));
        assert_eq!(rail.last_fired_ms("agent-a", "session-2"), Some(2_000));

        rail.reset_session("agent-a", "session-1");
        assert!(rail.last_fired_ms("agent-a", "session-1").is_none());
        assert!(!rail.is_due("agent-a", "session-1"));
        assert_eq!(rail.last_fired_ms("agent-a", "session-2"), Some(2_000));
        // The reset pair re-baselines instead of firing.
        now.store(3_000, Ordering::SeqCst);
        assert!(rail.observe("agent-a", "session-1", None).is_none());
        // The untouched pair keeps its cadence and sequence.
        let kept = rail.observe("agent-a", "session-2", None).unwrap();
        assert_eq!(kept.sequence, 2);

        rail.reset_all();
        assert!(rail.last_fired_ms("agent-a", "session-2").is_none());
        assert!(!rail.is_due("agent-a", "session-2"));
        // Events survive a reset; only the cadence is forgotten.
        assert_eq!(rail.events().len(), 3);
    }

    #[test]
    fn config_round_trips_and_tolerates_unknown_fields() {
        let config = HeartbeatRailConfig::default();
        assert_eq!(config.interval_ms, DEFAULT_HEARTBEAT_INTERVAL_MS);
        assert_eq!(config.interval_ms, 60_000);
        assert_eq!(config.language, "cn");
        assert_eq!(config.event_capacity, DEFAULT_HEARTBEAT_EVENT_CAPACITY);
        assert_eq!(config.event_capacity, 64);
        assert!(config.force_on_heartbeat_run);
        assert!(config.publish_prompt_section);

        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["interval_ms"], json!(60_000));
        assert_eq!(json["language"], json!("cn"));
        assert_eq!(json["event_capacity"], json!(64));
        assert_eq!(json["force_on_heartbeat_run"], json!(true));
        assert_eq!(json["publish_prompt_section"], json!(true));

        let back: HeartbeatRailConfig = serde_json::from_value(json).unwrap();
        assert_eq!(back, config);

        // Missing fields fall back to the defaults, unknown fields are ignored.
        let partial: HeartbeatRailConfig =
            serde_json::from_value(json!({"language": "en", "unknown": true})).unwrap();
        assert_eq!(partial.language, "en");
        assert_eq!(partial.interval_ms, DEFAULT_HEARTBEAT_INTERVAL_MS);
        assert_eq!(partial.event_capacity, DEFAULT_HEARTBEAT_EVENT_CAPACITY);
        assert!(partial.force_on_heartbeat_run);
        assert!(partial.publish_prompt_section);
    }

    #[test]
    fn config_builders_replace_every_field() {
        let config = HeartbeatRailConfig::default()
            .with_interval_ms(250)
            .with_language("en")
            .with_event_capacity(0)
            .with_force_on_heartbeat_run(false)
            .with_publish_prompt_section(false);
        assert_eq!(
            config,
            HeartbeatRailConfig {
                interval_ms: 250,
                language: "en".to_string(),
                event_capacity: 0,
                force_on_heartbeat_run: false,
                publish_prompt_section: false,
            }
        );
    }

    #[test]
    fn events_serialize_with_a_snake_case_reason() {
        let event = HeartbeatEvent {
            agent_id: "agent-a".to_string(),
            session_id: "session-a".to_string(),
            sequence: 3,
            fired_at_ms: 42,
            reason: HeartbeatReason::Scheduled,
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["agent_id"], json!("agent-a"));
        assert_eq!(json["session_id"], json!("session-a"));
        assert_eq!(json["sequence"], json!(3));
        assert_eq!(json["fired_at_ms"], json!(42));
        assert_eq!(json["reason"], json!("scheduled"));
        assert_eq!(
            serde_json::to_value(HeartbeatReason::RunKind).unwrap(),
            json!("run_kind")
        );
        let back: HeartbeatEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn names_priorities_and_defaults() {
        let rail = HeartbeatRail::default();
        assert_eq!(rail.name(), "heartbeat");
        assert_eq!(rail.name(), HEARTBEAT_SECTION);
        assert_eq!(rail.priority(), rail_priority::HEARTBEAT);
        assert_eq!(rail.priority(), 80);
        assert_eq!(rail.config(), &HeartbeatRailConfig::default());
        assert!(rail.events().is_empty());
        assert!(rail.last_fired_ms("agent-a", "session-a").is_none());
        assert!(!rail.is_due("agent-a", "session-a"));

        let debug = format!("{rail:?}");
        assert!(debug.starts_with("HeartbeatRail { config: HeartbeatRailConfig {"));
        assert!(debug.contains("clock: \"<heartbeat clock>\""));
        assert!(debug.contains("state: 0, events: 0"));
    }
}
