//! Per-tool resilience: timeouts, attempt budgets, and backoff.
//!
//! Ports the timeout half of `_resolve_call_timeout` / `_resolve_invocation_timeout`
//! in `openjiuwen/core/single_agent/ability_manager.py` and the retry budget of
//! the harness `ToolCallResilienceRail` as plain data. The policy only
//! describes budgets; executing a retry is the caller's job.
//!
//! [`Duration`] fields serialize as whole milliseconds
//! (`Option<u64>`), and deserialization accepts a JSON number or `null`; this
//! keeps the policy JSON-friendly without another dependency.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;
/// Budget for one tool call: its timeout, attempt count, and retry backoff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolResilience {
    /// Per-call timeout; `None` leaves the budget to the invocation default.
    #[serde(
        default,
        with = "optional_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub timeout: Option<Duration>,
    /// Attempt budget including the first try; `1` means no retry.
    pub max_attempts: u32,
    /// Delay before the first retry, doubled for every further attempt.
    #[serde(
        default,
        with = "optional_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub initial_backoff: Option<Duration>,
}
impl Default for ToolResilience {
    /// No timeout and no retry (`max_attempts = 1`).
    fn default() -> Self {
        Self {
            timeout: None,
            max_attempts: 1,
            initial_backoff: None,
        }
    }
}
impl ToolResilience {
    /// The default budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets [`ToolResilience::timeout`]; a consuming builder.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    /// Sets [`ToolResilience::max_attempts`]; a consuming builder.
    #[must_use]
    pub fn with_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }
    /// Sets [`ToolResilience::initial_backoff`]; a consuming builder.
    #[must_use]
    pub fn with_backoff(mut self, initial_backoff: Duration) -> Self {
        self.initial_backoff = Some(initial_backoff);
        self
    }
    /// The effective attempt count, clamped to at least one.
    #[must_use]
    pub fn attempts(&self) -> u32 {
        self.max_attempts.max(1)
    }
    /// The backoff before `attempt` (1-based): `initial_backoff * 2^(attempt-1)`,
    /// saturating on overflow. `None` when no backoff is configured.
    ///
    /// Attempt `0` behaves like the first attempt.
    #[must_use]
    pub fn backoff_for(&self, attempt: u32) -> Option<Duration> {
        let base = self.initial_backoff?;
        let exponent = attempt.saturating_sub(1).min(31);
        Some(base.saturating_mul(1_u32 << exponent))
    }
}
/// A default resilience budget plus per-tool overrides.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolResiliencePolicy {
    /// Budget for tools without an override.
    pub default: ToolResilience,
    /// Budgets keyed by tool name; these win over [`Self::default`].
    pub per_tool: BTreeMap<String, ToolResilience>,
}
impl ToolResiliencePolicy {
    /// Creates an empty policy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Replaces the default budget.
    pub fn set_default(&mut self, resilience: ToolResilience) {
        self.default = resilience;
    }
    /// Sets the budget for `tool_name`.
    pub fn set(&mut self, tool_name: impl Into<String>, resilience: ToolResilience) {
        self.per_tool.insert(tool_name.into(), resilience);
    }
    /// The budget for `tool_name`: its override when present, else the default.
    #[must_use]
    pub fn for_tool(&self, tool_name: &str) -> ToolResilience {
        self.per_tool
            .get(tool_name)
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }
    /// Returns `true` when the default budget is [`ToolResilience::default`]
    /// and no per-tool override exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.default == ToolResilience::default() && self.per_tool.is_empty()
    }
}
/// Serde support for `Option<Duration>` as whole milliseconds.
mod optional_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    // serde's `with` contract hands the field over as `&Option<Duration>`.
    #[allow(clippy::ref_option)]
    pub(super) fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(duration) => serializer.serialize_u64(millis(duration)),
            None => serializer.serialize_none(),
        }
    }
    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<u64>::deserialize(deserializer)?;
        Ok(value.map(Duration::from_millis))
    }
    fn millis(duration: &Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn defaults_have_no_timeout_and_no_retry() {
        let resilience = ToolResilience::new();
        assert_eq!(resilience.timeout, None);
        assert_eq!(resilience.initial_backoff, None);
        assert_eq!(resilience.attempts(), 1);
        assert_eq!(resilience, ToolResilience::default());
    }
    #[test]
    fn attempts_are_clamped_to_one() {
        let resilience = ToolResilience::new().with_attempts(0);
        assert_eq!(resilience.attempts(), 1);
        assert_eq!(resilience.with_attempts(4).attempts(), 4);
    }
    #[test]
    fn backoff_grows_exponentially() {
        let resilience = ToolResilience::new().with_backoff(Duration::from_millis(100));
        assert_eq!(resilience.backoff_for(0), Some(Duration::from_millis(100)));
        assert_eq!(resilience.backoff_for(1), Some(Duration::from_millis(100)));
        assert_eq!(resilience.backoff_for(2), Some(Duration::from_millis(200)));
        assert_eq!(resilience.backoff_for(3), Some(Duration::from_millis(400)));
        assert_eq!(ToolResilience::new().backoff_for(1), None);
    }
    #[test]
    fn backoff_saturates_instead_of_overflowing() {
        let resilience = ToolResilience::new().with_backoff(Duration::from_secs(u64::MAX) / 2);
        assert_eq!(resilience.backoff_for(u32::MAX), Some(Duration::MAX));
    }
    #[test]
    fn per_tool_override_wins_over_the_default() {
        let mut policy = ToolResiliencePolicy::new();
        policy.set_default(ToolResilience::new().with_timeout(Duration::from_secs(5)));
        policy.set(
            "shell",
            ToolResilience::new()
                .with_timeout(Duration::from_secs(60))
                .with_attempts(3),
        );
        assert_eq!(
            policy.for_tool("shell").timeout,
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            policy.for_tool("read_file").timeout,
            Some(Duration::from_secs(5))
        );
        assert!(!policy.is_empty());
        assert!(ToolResiliencePolicy::new().is_empty());
    }
    #[test]
    fn durations_serialize_as_milliseconds() {
        let resilience = ToolResilience::new()
            .with_timeout(Duration::from_millis(1500))
            .with_backoff(Duration::from_millis(250))
            .with_attempts(2);
        let value = serde_json::to_value(&resilience).expect("serialize");
        assert_eq!(
            value,
            json!({"timeout": 1500, "max_attempts": 2, "initial_backoff": 250})
        );
        let back: ToolResilience = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back, resilience);
    }
    #[test]
    fn durations_deserialize_from_plain_integers_and_null() {
        let resilience: ToolResilience =
            serde_json::from_value(json!({"timeout": 250})).expect("integer millis");
        assert_eq!(resilience.timeout, Some(Duration::from_millis(250)));
        assert_eq!(resilience.max_attempts, 1);
        let resilience: ToolResilience =
            serde_json::from_value(json!({"timeout": null})).expect("null");
        assert_eq!(resilience.timeout, None);
        let value = serde_json::to_value(ToolResilience::new()).expect("serialize");
        assert_eq!(value, json!({"max_attempts": 1}));
    }
    #[test]
    fn policy_round_trips_through_json() {
        let mut policy = ToolResiliencePolicy::new();
        policy.set(
            "shell",
            ToolResilience::new().with_timeout(Duration::from_secs(1)),
        );
        let value = serde_json::to_value(&policy).expect("serialize");
        let back: ToolResiliencePolicy = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back, policy);
    }
}
