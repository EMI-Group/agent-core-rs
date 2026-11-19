//! Runtime limits, activity phases, and deterministic LRU candidate selection for subagents.
//!
//! Ports the limits of `openjiuwen/harness/subagent_runtime/config.py`
//! (`SubagentRuntimeConfig`) and the quota ledger of `subagent_runtime/registry.py`.
//! [`SubAgentQuota`] is the immutable policy the runtime consults before reserving a slot;
//! [`SpawnBudget`] is the demand a queued spawn would add, counted by the runtime;
//! [`SubagentActivity`] is the derived lifecycle phase that drives eviction; and
//! [`select_lru_candidate`] is the pure, deterministic least-recently-used pick.
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::activity::{
    DEFAULT_ACTIVITY_CAPACITY, DEFAULT_ACTIVITY_TEXT_MAX_LEN, DEFAULT_ACTIVITY_THROTTLE_MS,
};
use super::transcript::{DEFAULT_TRANSCRIPT_CAPACITY, DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY};

/// Default live-instance cap (Python `max_subagents`).
const DEFAULT_MAX_SUBAGENTS: usize = 10;
/// Default concurrent-turn cap (Python `max_concurrent_running`).
const DEFAULT_MAX_CONCURRENT_RUNNING: usize = 5;
/// Default cumulative spawn budget of one runtime.
///
/// Python has no cumulative cap: its registry only bounds live instances, so a long session can
/// spawn without limit. Three waves of the ten-instance live cap is a bounded, documented
/// budget for the Rust runtime while still leaving ample room to reuse slots after they close.
const DEFAULT_MAX_TOTAL_SPAWNED: usize = 32;
/// Default live children of one spawning parent.
///
/// Four is below the ten-instance live cap and the five concurrent-turn cap, so one parent can
/// fan out widely without monopolising the pool other parents (or the root session) draw from.
const DEFAULT_MAX_CHILDREN_PER_PARENT: usize = 4;
/// Default nesting depth of spawned instances (top-level = 1).
///
/// Three admits a parent, its children, and their children, while stopping unbounded recursive
/// delegation before it can consume the spawn budget.
const DEFAULT_MAX_DEPTH: usize = 3;

/// Tunable limits of one subagent runtime (Python `SubagentRuntimeConfig`).
///
/// Python only bounds the live instance count and the concurrent turns; the Rust runtime adds
/// three additive caps ([`Self::max_total_spawned`], [`Self::max_children_per_parent`] and
/// [`Self::max_depth`]) so a model-driven spawn loop cannot fan out forever. Their defaults
/// are 32, 4 and 3; a caller can raise any of them with the matching `with_*` setter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubAgentQuota {
    /// Maximum live (non-closed) instances.
    pub max_subagents: usize,
    /// Maximum concurrently running turns.
    pub max_concurrent_running: usize,
    /// Maximum cumulative spawns of the session; `None` = unlimited.
    pub max_total_spawned: Option<usize>,
    /// Maximum live instances per spawning parent; `None` = unlimited.
    pub max_children_per_parent: Option<usize>,
    /// Maximum nesting depth of spawned instances (top-level = 1); `None` = unlimited.
    pub max_depth: Option<usize>,
    /// Evict the least-recently-used idle instance when the live cap is hit.
    ///
    /// Python parity: enabled by default (`SubagentRuntimeConfig.enable_lru_eviction`). When
    /// enabled, a spawn that finds the live cap reached closes the LRU evictable instance
    /// first (see [`crate::subagents::EVICTION_REASON`]); when disabled, the spawn is refused
    /// with the legacy `subagent capacity exceeded` error.
    pub enable_lru_eviction: bool,
    /// Ring-buffer cap of one subagent transcript.
    pub transcript_capacity: usize,
    /// Broadcast capacity of one subagent transcript feed.
    pub transcript_channel_capacity: usize,
    /// Record and publish activity frames for every instance (Python `enable_activity_stream`).
    ///
    /// When disabled, each instance still owns an [`ActivityStream`](super::activity::ActivityStream)
    /// but it stores and publishes nothing, so
    /// [`ActivityStream::is_enabled`](super::activity::ActivityStream::is_enabled) is `false` and
    /// the activity accessors report an empty history.
    pub enable_activity_stream: bool,
    /// Ring-buffer cap and per-turn frame cap of one activity stream.
    pub activity_capacity: usize,
    /// Longest activity summary kept, in characters.
    pub activity_text_max_len: usize,
    /// Window a coalesced thinking frame waits for, in milliseconds.
    pub activity_throttle_ms: i64,
}

impl Default for SubAgentQuota {
    fn default() -> Self {
        Self {
            max_subagents: DEFAULT_MAX_SUBAGENTS,
            max_concurrent_running: DEFAULT_MAX_CONCURRENT_RUNNING,
            max_total_spawned: Some(DEFAULT_MAX_TOTAL_SPAWNED),
            max_children_per_parent: Some(DEFAULT_MAX_CHILDREN_PER_PARENT),
            max_depth: Some(DEFAULT_MAX_DEPTH),
            enable_lru_eviction: true,
            transcript_capacity: DEFAULT_TRANSCRIPT_CAPACITY,
            transcript_channel_capacity: DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY,
            enable_activity_stream: true,
            activity_capacity: DEFAULT_ACTIVITY_CAPACITY,
            activity_text_max_len: DEFAULT_ACTIVITY_TEXT_MAX_LEN,
            activity_throttle_ms: DEFAULT_ACTIVITY_THROTTLE_MS,
        }
    }
}

impl SubAgentQuota {
    /// Replaces the live-instance cap.
    #[must_use]
    pub fn with_max_subagents(mut self, max: usize) -> Self {
        self.max_subagents = max;
        self
    }

    /// Replaces the concurrent-turn cap.
    #[must_use]
    pub fn with_concurrent_running(mut self, max: usize) -> Self {
        self.max_concurrent_running = max;
        self
    }

    /// Bounds the cumulative spawns of the session.
    #[must_use]
    pub fn with_total_spawned(mut self, max: usize) -> Self {
        self.max_total_spawned = Some(max);
        self
    }

    /// Bounds the live instances of one spawning parent.
    #[must_use]
    pub fn with_children_per_parent(mut self, max: usize) -> Self {
        self.max_children_per_parent = Some(max);
        self
    }

    /// Bounds the nesting depth of spawned instances (top-level = 1).
    #[must_use]
    pub fn with_max_depth(mut self, max: usize) -> Self {
        self.max_depth = Some(max);
        self
    }

    /// Enables or disables LRU eviction of idle instances.
    #[must_use]
    pub fn with_lru_eviction(mut self, enable: bool) -> Self {
        self.enable_lru_eviction = enable;
        self
    }

    /// Replaces the transcript ring-buffer cap.
    #[must_use]
    pub fn with_transcript_capacity(mut self, capacity: usize) -> Self {
        self.transcript_capacity = capacity;
        self
    }

    /// Replaces the transcript broadcast capacity.
    #[must_use]
    pub fn with_transcript_channel_capacity(mut self, capacity: usize) -> Self {
        self.transcript_channel_capacity = capacity;
        self
    }

    /// Enables or disables activity recording and publishing.
    #[must_use]
    pub fn with_activity_stream(mut self, enable: bool) -> Self {
        self.enable_activity_stream = enable;
        self
    }

    /// Replaces the activity ring capacity and per-turn frame cap.
    #[must_use]
    pub fn with_activity_capacity(mut self, capacity: usize) -> Self {
        self.activity_capacity = capacity;
        self
    }

    /// Replaces the longest activity summary kept, in characters.
    #[must_use]
    pub fn with_activity_text_max_len(mut self, max_len: usize) -> Self {
        self.activity_text_max_len = max_len;
        self
    }

    /// Replaces the thinking-coalescing window, in milliseconds.
    #[must_use]
    pub fn with_activity_throttle_ms(mut self, throttle_ms: i64) -> Self {
        self.activity_throttle_ms = throttle_ms;
        self
    }

    /// Checks one spawn demand against every limit.
    ///
    /// Checks run in a fixed order so a caller sees the most fundamental refusal first:
    /// nesting depth, per-parent fan-out, cumulative spawns, then the live-instance cap.
    /// `parent` labels the requesting parent in the per-parent message (the parent instance id,
    /// or the parent session id for a top-level spawn).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the limit that would be exceeded.
    pub fn check_spawn(&self, budget: &SpawnBudget, parent: &str) -> Result<()> {
        if let Some(limit) = self.max_depth {
            if budget.depth > limit {
                return Err(Error::config(format!(
                    "subagent depth limit exceeded: depth={}, limit={limit}",
                    budget.depth
                )));
            }
        }
        if let Some(limit) = self.max_children_per_parent {
            if budget.parent_live >= limit {
                return Err(Error::config(format!(
                    "subagent child limit exceeded: parent={parent}, used={}, limit={limit}",
                    budget.parent_live
                )));
            }
        }
        if let Some(limit) = self.max_total_spawned {
            if budget.spawned >= limit {
                return Err(Error::config(format!(
                    "subagent spawn budget exhausted: spawned={}, limit={limit}",
                    budget.spawned
                )));
            }
        }
        if budget.live >= self.max_subagents {
            return Err(Error::config(format!(
                "subagent capacity exceeded: used={}, limit={}",
                budget.live, self.max_subagents
            )));
        }
        Ok(())
    }
}

/// The spawn demand a queued spawn would add, as counted by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnBudget {
    /// Live instances right now.
    pub live: usize,
    /// Cumulative spawns so far.
    pub spawned: usize,
    /// Live instances whose parent is the requesting parent.
    pub parent_live: usize,
    /// Depth the new instance would have (top-level = 1).
    pub depth: usize,
}

/// Derived lifecycle phase of one instance; drives LRU eviction.
///
/// The phase advances `Queued -> Running -> Idle`, and `Closed` is terminal: only `Idle` and
/// `Queued` instances may ever be evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentActivity {
    /// Reserved, no turn has started yet.
    #[default]
    Queued,
    /// A turn is executing right now.
    Running,
    /// Started at least one turn and is not running one.
    Idle,
    /// Closed; only an in-memory tombstone remains.
    Closed,
}

impl SubagentActivity {
    /// Returns the `snake_case` wire string of this phase.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Idle => "idle",
            Self::Closed => "closed",
        }
    }

    /// Whether an instance in this phase may be evicted to free a slot.
    #[must_use]
    pub fn is_evictable(&self) -> bool {
        matches!(self, Self::Idle | Self::Queued)
    }
}

/// One eviction candidate seen by the LRU policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LruCandidate {
    /// Instance id.
    pub subagent_id: String,
    /// Derived lifecycle phase of the instance.
    pub activity: SubagentActivity,
    /// Milliseconds since the Unix epoch of the instance's last state transition.
    pub last_activity_ms: i64,
    /// Spawn sequence number of the instance (stable ordering tie-break).
    pub spawn_seq: u64,
}

/// Deterministic LRU pick: never a `Running`/`Closed` instance, prefer `Idle` over `Queued`,
/// then the oldest `last_activity_ms`, then the smallest `subagent_id`.
#[must_use]
pub fn select_lru_candidate(candidates: &[LruCandidate]) -> Option<&LruCandidate> {
    candidates
        .iter()
        .filter(|candidate| candidate.activity.is_evictable())
        .min_by(|left, right| compare_lru(left, right))
}

/// Orders two candidates by LRU preference: `Idle` before `Queued`, then the oldest
/// `last_activity_ms`, then the smallest `subagent_id`.
///
/// [`select_lru_candidate`] takes the first candidate under this order and the runtime sorts
/// its candidate list with it, so the picked instance and the reported order always agree.
pub(super) fn compare_lru(left: &LruCandidate, right: &LruCandidate) -> std::cmp::Ordering {
    activity_rank(left.activity)
        .cmp(&activity_rank(right.activity))
        .then_with(|| left.last_activity_ms.cmp(&right.last_activity_ms))
        .then_with(|| left.subagent_id.cmp(&right.subagent_id))
}

/// Eviction preference: `Idle` first, then `Queued`; non-evictable phases sort last.
fn activity_rank(activity: SubagentActivity) -> u8 {
    match activity {
        SubagentActivity::Idle => 0,
        SubagentActivity::Queued => 1,
        SubagentActivity::Running | SubagentActivity::Closed => u8::MAX,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{select_lru_candidate, LruCandidate, SpawnBudget, SubAgentQuota, SubagentActivity};

    fn budget(live: usize, spawned: usize, parent_live: usize, depth: usize) -> SpawnBudget {
        SpawnBudget {
            live,
            spawned,
            parent_live,
            depth,
        }
    }

    fn candidate(
        id: &str,
        activity: SubagentActivity,
        last_activity_ms: i64,
        spawn_seq: u64,
    ) -> LruCandidate {
        LruCandidate {
            subagent_id: id.to_owned(),
            activity,
            last_activity_ms,
            spawn_seq,
        }
    }

    #[test]
    fn defaults_match_python_and_bound_fan_out() {
        let quota = SubAgentQuota::default();
        assert_eq!(quota.max_subagents, 10);
        assert_eq!(quota.max_concurrent_running, 5);
        assert_eq!(quota.max_total_spawned, Some(32));
        assert_eq!(quota.max_children_per_parent, Some(4));
        assert_eq!(quota.max_depth, Some(3));
        assert!(quota.enable_lru_eviction);
        assert_eq!(quota.transcript_capacity, 256);
        assert_eq!(quota.transcript_channel_capacity, 256);
        assert!(quota.enable_activity_stream);
        assert_eq!(quota.activity_capacity, 256);
        assert_eq!(quota.activity_text_max_len, 2_000);
        assert_eq!(quota.activity_throttle_ms, 500);
    }

    #[test]
    fn fluent_setters_replace_every_field() {
        let quota = SubAgentQuota::default()
            .with_max_subagents(2)
            .with_concurrent_running(3)
            .with_total_spawned(4)
            .with_children_per_parent(5)
            .with_max_depth(6)
            .with_lru_eviction(false)
            .with_transcript_capacity(7)
            .with_transcript_channel_capacity(8)
            .with_activity_stream(false)
            .with_activity_capacity(9)
            .with_activity_text_max_len(10)
            .with_activity_throttle_ms(11);
        assert_eq!(
            quota,
            SubAgentQuota {
                max_subagents: 2,
                max_concurrent_running: 3,
                max_total_spawned: Some(4),
                max_children_per_parent: Some(5),
                max_depth: Some(6),
                enable_lru_eviction: false,
                transcript_capacity: 7,
                transcript_channel_capacity: 8,
                enable_activity_stream: false,
                activity_capacity: 9,
                activity_text_max_len: 10,
                activity_throttle_ms: 11,
            }
        );
    }

    #[test]
    fn live_cap_reports_the_legacy_message() {
        let quota = SubAgentQuota::default().with_max_subagents(1);
        let error = quota
            .check_spawn(&budget(1, 0, 0, 1), "parent")
            .unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message == "subagent capacity exceeded: used=1, limit=1"
        ));
        assert!(quota.check_spawn(&budget(0, 0, 0, 1), "parent").is_ok());
    }

    #[test]
    fn cumulative_spawn_budget_reports_its_message() {
        let quota = SubAgentQuota::default().with_total_spawned(2);
        assert!(quota.check_spawn(&budget(0, 1, 0, 1), "parent").is_ok());
        let error = quota
            .check_spawn(&budget(0, 2, 0, 1), "parent")
            .unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message == "subagent spawn budget exhausted: spawned=2, limit=2"
        ));
    }

    #[test]
    fn per_parent_limit_reports_its_message() {
        let quota = SubAgentQuota::default().with_children_per_parent(1);
        assert!(quota
            .check_spawn(&budget(0, 0, 0, 1), "parent-session")
            .is_ok());
        let error = quota
            .check_spawn(&budget(0, 0, 1, 2), "parent-session")
            .unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message == "subagent child limit exceeded: parent=parent-session, used=1, limit=1"
        ));
    }

    #[test]
    fn depth_limit_reports_its_message() {
        let quota = SubAgentQuota::default().with_max_depth(2);
        assert!(quota.check_spawn(&budget(0, 0, 0, 2), "parent").is_ok());
        let error = quota
            .check_spawn(&budget(0, 0, 0, 3), "parent")
            .unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message == "subagent depth limit exceeded: depth=3, limit=2"
        ));
    }

    #[test]
    fn absent_caps_never_fire() {
        let quota = SubAgentQuota {
            max_total_spawned: None,
            max_children_per_parent: None,
            max_depth: None,
            ..SubAgentQuota::default()
        };
        assert!(quota
            .check_spawn(&budget(0, 10_000, 10_000, 1_000), "p")
            .is_ok());
    }

    #[test]
    fn checks_run_depth_then_parent_then_spawns_then_live() {
        let quota = SubAgentQuota::default()
            .with_max_subagents(0)
            .with_total_spawned(1)
            .with_children_per_parent(1)
            .with_max_depth(1);
        let error = quota.check_spawn(&budget(0, 1, 1, 2), "p").unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message.starts_with("subagent depth limit exceeded:")
        ));
        let error = quota.check_spawn(&budget(0, 1, 1, 1), "p").unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message.starts_with("subagent child limit exceeded:")
        ));
        let error = quota.check_spawn(&budget(0, 1, 0, 1), "p").unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message.starts_with("subagent spawn budget exhausted:")
        ));
        let error = quota.check_spawn(&budget(0, 0, 0, 1), "p").unwrap_err();
        assert!(matches!(
            &error,
            openjiuwen_common::error::Error::Config(message)
                if message.starts_with("subagent capacity exceeded:")
        ));
    }

    #[test]
    fn activity_wire_strings_and_evictability() {
        let cases = [
            (SubagentActivity::Queued, "queued", true),
            (SubagentActivity::Running, "running", false),
            (SubagentActivity::Idle, "idle", true),
            (SubagentActivity::Closed, "closed", false),
        ];
        for (activity, wire, evictable) in cases {
            assert_eq!(activity.as_str(), wire);
            assert_eq!(activity.is_evictable(), evictable);
        }
        assert_eq!(SubagentActivity::default(), SubagentActivity::Queued);
    }

    #[test]
    fn quota_and_activity_round_trip_through_json() {
        let quota = SubAgentQuota::default()
            .with_max_subagents(3)
            .with_total_spawned(9)
            .with_lru_eviction(false);
        let json = serde_json::to_string(&quota).unwrap();
        assert_eq!(serde_json::from_str::<SubAgentQuota>(&json).unwrap(), quota);
        let lenient: SubAgentQuota =
            serde_json::from_str(r#"{"max_subagents": 2, "unknown": true}"#).unwrap();
        assert_eq!(lenient.max_subagents, 2);
        assert_eq!(lenient.max_concurrent_running, 5);
        let activity = serde_json::to_string(&SubagentActivity::Idle).unwrap();
        assert_eq!(activity, "\"idle\"");
        assert_eq!(
            serde_json::from_str::<SubagentActivity>("\"running\"").unwrap(),
            SubagentActivity::Running
        );
    }

    #[test]
    fn no_candidates_select_nothing() {
        assert!(select_lru_candidate(&[]).is_none());
        assert!(select_lru_candidate(&[
            candidate("running", SubagentActivity::Running, 0, 0),
            candidate("closed", SubagentActivity::Closed, 0, 1),
        ])
        .is_none());
    }

    #[test]
    fn idle_is_preferred_over_an_older_queued_instance() {
        let candidates = [
            candidate("queued", SubagentActivity::Queued, 10, 1),
            candidate("idle", SubagentActivity::Idle, 20, 2),
            candidate("running", SubagentActivity::Running, 0, 3),
        ];
        let picked = select_lru_candidate(&candidates).unwrap();
        assert_eq!(picked.subagent_id, "idle");
    }

    #[test]
    fn oldest_wins_within_one_activity_class() {
        let candidates = [
            candidate("newer", SubagentActivity::Idle, 50, 1),
            candidate("oldest", SubagentActivity::Idle, 10, 2),
            candidate("middle", SubagentActivity::Idle, 30, 3),
        ];
        assert_eq!(
            select_lru_candidate(&candidates).unwrap().subagent_id,
            "oldest"
        );
        let queued = [
            candidate("b", SubagentActivity::Queued, 10, 4),
            candidate("a", SubagentActivity::Queued, 10, 5),
        ];
        assert_eq!(select_lru_candidate(&queued).unwrap().subagent_id, "a");
    }

    #[test]
    fn spawn_seq_does_not_override_the_documented_preference() {
        let candidates = [
            candidate("first", SubagentActivity::Queued, 10, 0),
            candidate("second", SubagentActivity::Idle, 99, 9),
        ];
        assert_eq!(
            select_lru_candidate(&candidates).unwrap().subagent_id,
            "second"
        );
    }
}
