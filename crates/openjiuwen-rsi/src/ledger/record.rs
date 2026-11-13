//! Model-call usage records persisted by the run ledger.
//!
//! Ported from Python `openjiuwen.rsi.events` (`EventUsage`, `NodeStageEvent`)
//! and `openjiuwen.rsi.schema` (`RsiUsageTokens`, `RsiUsage`, `RsiModelCall`).
//!
//! A [`UsageRecord`] is one completed model request without prompts,
//! credentials, or response content: the model name, a stable caller-supplied
//! `call_id`, provider token counters, a status, and optional duration, node,
//! and stage references. A [`StageRecord`] is a pass-through subprocess-stage
//! transition attached to an existing tree node; the engine owns the stage
//! taxonomy and the record carries it as uninterpreted JSON.
//!
//! Token normalisation follows Python `usage.usage_tokens` exactly: the first
//! present, non-negative integer alias wins, cache hits are a subset of the
//! input tokens rather than extra tokens, and a counter the provider did not
//! report stays [`None`] instead of being estimated as zero.
//!
//! [`RunEvent`] serialises each record flat, exactly like the Python dataclasses
//! (a `progress.usage` object and a `node.stage` object share one JSONL file and
//! are distinguished by their `event_type` field), so the persisted format stays
//! compatible with Python's `model_calls.jsonl`.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Family marker carried by every usage event (Python `EventUsage.family`).
pub const USAGE_FAMILY: &str = "progress";
/// Kind marker carried by every usage event (Python `EventUsage.kind`).
pub const USAGE_KIND: &str = "usage";
/// The `event_type` of a [`UsageRecord`] (Python `EventUsage.event_type`).
pub const USAGE_EVENT_TYPE: &str = "progress.usage";
/// The `event_type` of a [`StageRecord`] (Python `NodeStageEvent.event_type`).
pub const STAGE_EVENT_TYPE: &str = "node.stage";

/// Outcome of one model request (Python's `succeeded` / `failed` / `incomplete`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCallStatus {
    /// The call completed and reported usage.
    #[default]
    Succeeded,
    /// The call failed.
    Failed,
    /// The call was interrupted or never reported a result.
    Incomplete,
}

impl ModelCallStatus {
    /// Returns the Python wire name of the status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Incomplete => "incomplete",
        }
    }
}

impl fmt::Display for ModelCallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provider token counters; [`None`] means the counter was not reported.
///
/// Cache hits are included in the input tokens, so they are a subset of
/// `input`, never additional tokens. Reasoning tokens are included in `output`
/// when the provider reports them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UsageTokens {
    /// Input (prompt) tokens, including cache hits.
    pub input: Option<u64>,
    /// Output (completion) tokens.
    pub output: Option<u64>,
    /// Input tokens served from the provider cache.
    pub cache_hit: Option<u64>,
}

impl UsageTokens {
    /// Creates a token triple from the three counters.
    #[must_use]
    pub const fn new(input: Option<u64>, output: Option<u64>, cache_hit: Option<u64>) -> Self {
        Self {
            input,
            output,
            cache_hit,
        }
    }

    /// Reports whether no counter was reported.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.input.is_none() && self.output.is_none() && self.cache_hit.is_none()
    }
}

/// A cumulative usage projection loaded from persisted state.
///
/// Python's `RsiUsage`: token counters, an optional cost estimate, and the
/// number of calls the projection covers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UsageSnapshot {
    /// Cumulative token counters.
    pub tokens: UsageTokens,
    /// Cumulative cost estimate in US dollars, when the caller reported one.
    pub cost_estimate: Option<f64>,
    /// Number of calls covered by the projection.
    pub call_count: Option<u64>,
}

/// One completed model request (Python `RsiModelCall`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelCall {
    /// Model identifier as reported by the client.
    pub model: String,
    /// Number of provider calls this record represents (Python always writes `1`).
    pub call_count: u64,
    /// Provider token counters.
    pub tokens: UsageTokens,
    /// Call outcome.
    pub status: ModelCallStatus,
    /// Wall-clock duration in milliseconds, when the observer measured one.
    pub duration_ms: Option<u64>,
}

impl ModelCall {
    /// Creates a successful single-call record for `model`.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            call_count: 1,
            ..Self::default()
        }
    }

    /// Replaces the token counters.
    #[must_use]
    pub fn with_tokens(mut self, tokens: UsageTokens) -> Self {
        self.tokens = tokens;
        self
    }

    /// Replaces the call status.
    #[must_use]
    pub fn with_status(mut self, status: ModelCallStatus) -> Self {
        self.status = status;
        self
    }

    /// Records the call duration in milliseconds.
    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// One persisted model-call usage event (Python `EventUsage`).
///
/// The record is content-free by construction: there is no field that could
/// carry a prompt, a response, or a credential.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UsageRecord {
    /// Unique id of this ledger event (a UUID v4).
    pub event_id: String,
    /// Owning run's task id; a resumed ledger verifies it before replaying.
    pub task_id: String,
    /// UTC timestamp of the record, RFC 3339 with millisecond precision.
    pub ts: String,
    /// Stable caller-supplied call id used for deduplication.
    pub call_id: String,
    /// The model call this event describes.
    pub model_call: ModelCall,
    /// Iteration node the call belonged to, when known.
    pub node_ref: Option<String>,
    /// Engine stage the call belonged to, when known.
    pub stage_ref: Option<String>,
    /// Event family; always [`USAGE_FAMILY`].
    pub family: String,
    /// Event kind; always [`USAGE_KIND`].
    pub kind: String,
    /// Event discriminator; always [`USAGE_EVENT_TYPE`].
    pub event_type: String,
    /// Caller-reported cost of this call in US dollars (never derived here).
    pub cost_usd: f64,
}

impl Default for UsageRecord {
    fn default() -> Self {
        Self {
            event_id: String::new(),
            task_id: String::new(),
            ts: String::new(),
            call_id: String::new(),
            model_call: ModelCall::default(),
            node_ref: None,
            stage_ref: None,
            family: USAGE_FAMILY.to_owned(),
            kind: USAGE_KIND.to_owned(),
            event_type: USAGE_EVENT_TYPE.to_owned(),
            cost_usd: 0.0,
        }
    }
}

impl UsageRecord {
    /// Creates a usage record with a fresh event id and current UTC timestamp.
    #[must_use]
    pub fn new(
        task_id: impl Into<String>,
        call_id: impl Into<String>,
        model_call: ModelCall,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            task_id: task_id.into(),
            ts: now_rfc3339(),
            call_id: call_id.into(),
            model_call,
            ..Self::default()
        }
    }

    /// Attributes the record to an iteration node.
    #[must_use]
    pub fn with_node_ref(mut self, node_ref: impl Into<String>) -> Self {
        self.node_ref = Some(node_ref.into());
        self
    }

    /// Attributes the record to an engine stage.
    #[must_use]
    pub fn with_stage_ref(mut self, stage_ref: impl Into<String>) -> Self {
        self.stage_ref = Some(stage_ref.into());
        self
    }

    /// Sets the caller-reported cost of this call in US dollars.
    #[must_use]
    pub fn with_cost_usd(mut self, cost_usd: f64) -> Self {
        self.cost_usd = cost_usd;
        self
    }
}

/// A subprocess-stage transition for an existing tree node (Python `NodeStageEvent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StageRecord {
    /// Tree node the stage belongs to.
    pub node_ref: String,
    /// Opaque stage description owned by the engine.
    pub stage: Value,
    /// Human-readable note; empty when absent.
    pub note: String,
    /// Event discriminator; always [`STAGE_EVENT_TYPE`].
    pub event_type: String,
}

impl Default for StageRecord {
    fn default() -> Self {
        Self {
            node_ref: String::new(),
            stage: Value::Null,
            note: String::new(),
            event_type: STAGE_EVENT_TYPE.to_owned(),
        }
    }
}

impl StageRecord {
    /// Creates a stage transition for `node_ref` carrying `stage`.
    #[must_use]
    pub fn new(node_ref: impl Into<String>, stage: Value) -> Self {
        Self {
            node_ref: node_ref.into(),
            stage,
            ..Self::default()
        }
    }

    /// Attaches a human-readable note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = note.into();
        self
    }
}

/// One line of the run ledger.
///
/// Both variants serialise flat (the Python dataclass shape) and are
/// distinguished on deserialisation by their `event_type` field, so a
/// `progress.usage` line never has to be wrapped in an extra JSON object.
#[derive(Debug, Clone, PartialEq)]
pub enum RunEvent {
    /// A model-call usage record.
    Usage(UsageRecord),
    /// A node stage transition.
    Stage(StageRecord),
}

impl RunEvent {
    /// Returns the event's Python `event_type` discriminator.
    #[must_use]
    pub fn event_type(&self) -> &str {
        match self {
            Self::Usage(record) => &record.event_type,
            Self::Stage(record) => &record.event_type,
        }
    }

    /// Returns the usage record when this is a [`RunEvent::Usage`].
    #[must_use]
    pub fn as_usage(&self) -> Option<&UsageRecord> {
        match self {
            Self::Usage(record) => Some(record),
            Self::Stage(_) => None,
        }
    }

    /// Returns the stage record when this is a [`RunEvent::Stage`].
    #[must_use]
    pub fn as_stage(&self) -> Option<&StageRecord> {
        match self {
            Self::Usage(_) => None,
            Self::Stage(record) => Some(record),
        }
    }
}

impl From<UsageRecord> for RunEvent {
    fn from(record: UsageRecord) -> Self {
        Self::Usage(record)
    }
}

impl From<StageRecord> for RunEvent {
    fn from(record: StageRecord) -> Self {
        Self::Stage(record)
    }
}

impl Serialize for RunEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Usage(record) => record.serialize(serializer),
            Self::Stage(record) => record.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for RunEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let event_type = value.get("event_type").and_then(Value::as_str);
        if event_type == Some(STAGE_EVENT_TYPE) {
            StageRecord::deserialize(value)
                .map(Self::Stage)
                .map_err(<D::Error as serde::de::Error>::custom)
        } else {
            UsageRecord::deserialize(value)
                .map(Self::Usage)
                .map_err(<D::Error as serde::de::Error>::custom)
        }
    }
}

/// Normalises provider usage counters exactly like Python `usage.usage_tokens`.
///
/// Input aliases are `input_tokens`, `prompt_tokens`, `input`; output aliases
/// are `output_tokens`, `completion_tokens`, `output`; the cache-hit aliases
/// are `cache_read_tokens`, `prompt_cache_hit_tokens`, `cache_hit`, and the
/// nested `prompt_tokens_details.cached_tokens`. The first alias whose value is
/// a non-negative integer wins; an alias that is present but invalid is skipped
/// in favour of the next one, and a counter with no valid alias stays [`None`].
///
/// Numbers may arrive as unsigned, signed, or whole floating-point JSON numbers;
/// strings, booleans, fractions, and negative numbers are ignored.
#[must_use]
pub fn tokens_from_usage(raw: &Value) -> UsageTokens {
    let cache = counter(
        raw,
        &["cache_read_tokens", "prompt_cache_hit_tokens", "cache_hit"],
    )
    .or_else(|| cache_details_counter(raw));
    UsageTokens {
        input: counter(raw, &["input_tokens", "prompt_tokens", "input"]),
        output: counter(raw, &["output_tokens", "completion_tokens", "output"]),
        cache_hit: cache,
    }
}

/// Loads a cumulative usage projection from persisted state (Python `usage_snapshot`).
///
/// Token counters are read from the `tokens` object. The cost comes from
/// `cost_estimate`, `cost_usd`, or `cost`, and the call count from `call_count`
/// or `calls`. Returns [`None`] when the value carries neither token nor cost
/// information, which is Python's "nothing persisted yet" signal.
#[must_use]
pub fn usage_snapshot(raw: &Value) -> Option<UsageSnapshot> {
    if !raw.is_object() {
        return None;
    }
    let tokens = tokens_from_usage(raw.get("tokens").unwrap_or(&Value::Null));
    let cost_estimate = number(raw, &["cost_estimate", "cost_usd", "cost"]);
    if tokens.is_empty() && cost_estimate.is_none() {
        return None;
    }
    Some(UsageSnapshot {
        tokens,
        cost_estimate,
        call_count: counter(raw, &["call_count", "calls"]),
    })
}

/// Returns the first valid counter among `keys` on the object `value`.
fn counter(value: &Value, keys: &[&str]) -> Option<u64> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(non_negative_integer)
}

/// Returns `prompt_tokens_details.cached_tokens` when it is a valid counter.
fn cache_details_counter(value: &Value) -> Option<u64> {
    let details = value.get("prompt_tokens_details")?;
    counter(details, &["cached_tokens"])
}

/// Returns the first valid floating-point number among `keys`.
fn number(value: &Value, keys: &[&str]) -> Option<f64> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_f64)
}

/// Converts a JSON number to a non-negative integer counter.
///
/// Unsigned and signed integers are accepted when non-negative; floating-point
/// numbers are accepted only when finite, non-negative, and whole. Everything
/// else (including booleans and strings) yields [`None`].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn non_negative_integer(value: &Value) -> Option<u64> {
    let number = value.as_number()?;
    if let Some(unsigned) = number.as_u64() {
        return Some(unsigned);
    }
    if let Some(signed) = number.as_i64() {
        return (signed >= 0).then_some(signed as u64);
    }
    let float = number.as_f64()?;
    if float.is_finite() && float >= 0.0 && float.fract() == 0.0 && float <= 9_007_199_254_740_992.0
    {
        return Some(float as u64);
    }
    None
}

/// Highest `Unix` second that still formats inside four-digit years (`9999-12-31`).
const MAX_RFC3339_SECONDS: u64 = 253_402_300_799;

/// Formats a system time as RFC 3339 in UTC with millisecond precision.
///
/// The formatter is self-contained (no `chrono` dependency in this crate) and
/// uses the civil-from-days algorithm. Times before the Unix epoch clamp to
/// `1970-01-01T00:00:00.000Z`, and times past year 9999 clamp to the last
/// representable second; `SystemTime::now` always falls inside the range.
///
/// ```
/// use std::time::{Duration, UNIX_EPOCH};
/// use openjiuwen_rsi::ledger::format_rfc3339;
///
/// let at = UNIX_EPOCH + Duration::from_millis(1_700_000_000_123);
/// assert_eq!(format_rfc3339(at), "2023-11-14T22:13:20.123Z");
/// ```
#[must_use]
pub fn format_rfc3339(time: SystemTime) -> String {
    let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = duration.as_secs().min(MAX_RFC3339_SECONDS);
    let millis = duration.subsec_millis();
    let days = i64::try_from(seconds / 86_400).unwrap_or(i64::MAX);
    let second_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Returns the current UTC time as an RFC 3339 string with millisecond precision.
#[must_use]
pub fn now_rfc3339() -> String {
    format_rfc3339(SystemTime::now())
}

/// Converts days since the Unix epoch into a `(year, month, day)` civil date.
///
/// Howard Hinnant's `civil_from_days` algorithm, valid for the whole `i64`
/// range of day counts.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097; // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153; // [0, 11]
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1; // [1, 31]
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (
        year,
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn format_rfc3339_renders_utc_millis() {
        assert_eq!(format_rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        let at = UNIX_EPOCH + Duration::from_millis(1_700_000_000_123);
        assert_eq!(format_rfc3339(at), "2023-11-14T22:13:20.123Z");
        let leap = UNIX_EPOCH + Duration::from_secs(1_709_164_800);
        assert_eq!(format_rfc3339(leap), "2024-02-29T00:00:00.000Z");
    }

    #[test]
    fn format_rfc3339_clamps_times_outside_the_range() {
        let before = UNIX_EPOCH - Duration::from_secs(60);
        assert_eq!(format_rfc3339(before), "1970-01-01T00:00:00.000Z");
        let after =
            UNIX_EPOCH + Duration::from_secs(MAX_RFC3339_SECONDS + 1) + Duration::from_millis(999);
        assert_eq!(format_rfc3339(after), "9999-12-31T23:59:59.999Z");
    }

    #[test]
    fn usage_events_serialize_flat_with_a_type_discriminator() {
        let record = UsageRecord::new("task", "call", ModelCall::new("model"));
        let value = serde_json::to_value(RunEvent::Usage(record.clone())).unwrap();
        assert_eq!(value["event_type"], USAGE_EVENT_TYPE);
        assert_eq!(value["family"], USAGE_FAMILY);
        assert_eq!(value["call_id"], "call");
        let round_trip: RunEvent = serde_json::from_value(value).unwrap();
        assert_eq!(round_trip, RunEvent::Usage(record));

        let stage = StageRecord::new("h0", json!({"stage": "build"}));
        let value = serde_json::to_value(RunEvent::Stage(stage.clone())).unwrap();
        assert_eq!(value["event_type"], STAGE_EVENT_TYPE);
        assert_eq!(value["node_ref"], "h0");
        let round_trip: RunEvent = serde_json::from_value(value).unwrap();
        assert_eq!(round_trip, RunEvent::Stage(stage));
    }

    #[test]
    fn recording_defaults_and_builders_are_stable() {
        let record = UsageRecord::new(
            "task",
            "call",
            ModelCall::new("m").with_tokens(UsageTokens::new(Some(1), Some(2), None)),
        );
        assert_ne!(record.event_id, "");
        assert!(record.ts.ends_with('Z'));
        assert_eq!(record.family, USAGE_FAMILY);
        assert_eq!(record.kind, USAGE_KIND);
        assert_eq!(record.cost_usd, 0.0);
        let record = record
            .with_node_ref("h1")
            .with_stage_ref("plan")
            .with_cost_usd(0.5);
        assert_eq!(record.node_ref.as_deref(), Some("h1"));
        assert_eq!(record.stage_ref.as_deref(), Some("plan"));
        assert!((record.cost_usd - 0.5).abs() < f64::EPSILON);
        assert_eq!(ModelCall::new("m").call_count, 1);
    }

    #[test]
    fn tokens_accept_every_python_alias() {
        for (key, expected) in [("input_tokens", 1_u64), ("prompt_tokens", 2), ("input", 3)] {
            let raw = json!({ key: expected });
            assert_eq!(tokens_from_usage(&raw).input, Some(expected), "{key}");
        }
        for (key, expected) in [
            ("output_tokens", 1_u64),
            ("completion_tokens", 2),
            ("output", 3),
        ] {
            let raw = json!({ key: expected });
            assert_eq!(tokens_from_usage(&raw).output, Some(expected), "{key}");
        }
        for (key, expected) in [
            ("cache_read_tokens", 1_u64),
            ("prompt_cache_hit_tokens", 2),
            ("cache_hit", 3),
        ] {
            let raw = json!({ key: expected });
            assert_eq!(tokens_from_usage(&raw).cache_hit, Some(expected), "{key}");
        }
        let nested = json!({"prompt_tokens_details": {"cached_tokens": 9}});
        assert_eq!(tokens_from_usage(&nested).cache_hit, Some(9));
    }

    #[test]
    fn tokens_skip_invalid_aliases_and_accept_whole_floats() {
        let negative = json!({"input_tokens": -5, "prompt_tokens": 7});
        assert_eq!(tokens_from_usage(&negative).input, Some(7));

        let invalid = json!({"input_tokens": "12", "prompt_tokens": true});
        assert_eq!(tokens_from_usage(&invalid).input, None);

        let fraction = json!({"input_tokens": 1.5});
        assert_eq!(tokens_from_usage(&fraction).input, None);

        let whole = json!({"input_tokens": 12.0, "output": 3.0});
        let tokens = tokens_from_usage(&whole);
        assert_eq!(tokens.input, Some(12));
        assert_eq!(tokens.output, Some(3));

        assert!(tokens_from_usage(&json!({})).is_empty());
        assert!(tokens_from_usage(&json!(null)).is_empty());
    }

    #[test]
    fn usage_snapshot_returns_none_without_usage() {
        assert!(usage_snapshot(&json!({})).is_none());
        assert!(usage_snapshot(&json!({"call_count": 4})).is_none());
        assert!(usage_snapshot(&json!(null)).is_none());
        assert!(usage_snapshot(&json!("tokens")).is_none());
    }

    #[test]
    fn usage_snapshot_reads_cost_and_call_count_aliases() {
        let snapshot = usage_snapshot(&json!({
            "tokens": {"input_tokens": 5, "output_tokens": 2},
            "cost_estimate": 0.25,
            "call_count": 2,
        }))
        .unwrap();
        assert_eq!(snapshot.tokens.input, Some(5));
        assert_eq!(snapshot.tokens.output, Some(2));
        assert_eq!(snapshot.call_count, Some(2));
        let cost = snapshot.cost_estimate.unwrap();
        assert!((cost - 0.25).abs() < f64::EPSILON);

        let alias = usage_snapshot(&json!({"cost_usd": 1, "calls": 3})).unwrap();
        assert_eq!(alias.call_count, Some(3));
        let cost = alias.cost_estimate.unwrap();
        assert!((cost - 1.0).abs() < f64::EPSILON);
    }
}
