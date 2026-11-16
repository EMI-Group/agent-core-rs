//! Serialized `context.usage` value objects.
//!
//! Ports `usage/models.py`: the four-category part model, the window usage
//! block, the per-request and per-session KV-cache contracts, the measurement
//! block, the emitted [`ContextUsageSnapshot`], and the immutable-by-convention
//! [`ContextWindowTokenReport`].
//!
//! Every type is `serde`-serializable with `snake_case` field names, defaults
//! for every field, and tolerance for unknown fields, so a snapshot written by
//! another producer still deserializes.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::stats::ContextStats;
use crate::tokenizers::{ModelTokenCounter, SelectedTokenizer, TokenizerSource};

use super::session_aggregator::CacheAggregationKey;

/// The `type` value carried by every emitted snapshot.
pub const CONTEXT_USAGE_EVENT_TYPE: &str = "context.usage";

/// The schema version carried by every emitted snapshot.
pub const CONTEXT_USAGE_SCHEMA_VERSION: &str = "context-usage.v1";

/// The pre-call phase, the default for [`ContextUsageSnapshot::phase`].
pub const PHASE_PRE_CALL: &str = "pre_call";

/// The post-call phase.
pub const PHASE_POST_CALL: &str = "post_call";

/// The post-compaction phase.
pub const PHASE_POST_COMPACT: &str = "post_compact";

/// The source reported when a value was not measured or not reported.
pub const NOT_REPORTED: &str = "not_reported";

/// A measurement source that mixes several tokenizers or sources.
pub const MIXED: &str = "mixed";

/// The `tokens_source` of a window whose total came from the provider.
pub const SOURCE_PROVIDER_USAGE: &str = "provider_usage";

/// The `source` of a messages part holding the provider residual.
pub const SOURCE_PROVIDER_USAGE_RESIDUAL: &str = "provider_usage_residual";

/// The `source` of a part scaled proportionally to the provider total.
pub const SOURCE_PROVIDER_USAGE_PROPORTIONAL: &str = "provider_usage_proportional";

/// The `source` of a messages part holding an unattributable provider total.
pub const SOURCE_PROVIDER_USAGE_UNATTRIBUTED: &str = "provider_usage_unattributed";

/// The `status` of a request whose cache usage was not reported.
pub const STATUS_NOT_REPORTED: &str = "not_reported";

/// The `status` of a request whose reported cache usage is internally
/// inconsistent.
pub const STATUS_INVALID: &str = "invalid";

/// The `status` of a call that read nothing from the provider cache.
pub const STATUS_MISS: &str = "miss";

/// The `status` of a call whose prompt was only partly served from cache.
pub const STATUS_PARTIAL_HIT: &str = "partial_hit";

/// The `status` of a call whose prompt was fully served from cache.
pub const STATUS_FULL_HIT: &str = "full_hit";

/// The measurement-version tag mixed into every measurement-cache key.
pub const MEASUREMENT_VERSION: &str = "1";

/// The measurement-cache key salt, so a policy change invalidates old entries.
pub(crate) const MEASUREMENT_KEY_VERSION: &str = "measurement-v1";

/// One logical category of a prompt.
///
/// Serializes as `system_prompt`, `tools`, `skills` or `messages`. The legacy
/// `memory` category has no variant: [`ContextCategory::from_name`] (and thus
/// deserialization) normalizes it — and any unknown name — to
/// [`ContextCategory::SystemPrompt`], matching Python's `_normalize_category`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextCategory {
    /// The system prompt: persona, instructions and attached context blocks.
    SystemPrompt,
    /// The serialized tool definitions sent with the request.
    Tools,
    /// Skill documentation injected into the prompt.
    Skills,
    /// The conversation history itself.
    Messages,
}

impl ContextCategory {
    /// Every category, in Python's protocol order.
    pub const ALL: [Self; 4] = [
        Self::SystemPrompt,
        Self::Tools,
        Self::Skills,
        Self::Messages,
    ];

    /// The wire name of the category.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SystemPrompt => "system_prompt",
            Self::Tools => "tools",
            Self::Skills => "skills",
            Self::Messages => "messages",
        }
    }

    /// Parses a category name, normalizing `memory` (and anything unknown) to
    /// [`ContextCategory::SystemPrompt`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim() {
            "system_prompt" | "memory" => Some(Self::SystemPrompt),
            "tools" => Some(Self::Tools),
            "skills" => Some(Self::Skills),
            "messages" => Some(Self::Messages),
            _ => None,
        }
    }

    /// Whether an attachment message may be routed into this category.
    ///
    /// Only `system_prompt` and `skills` accept attachments; a `tools` or
    /// `messages` attachment would double-count a category the window measures
    /// on its own, so the analyzer keeps such messages in `messages`.
    #[must_use]
    pub fn accepts_attachment(self) -> bool {
        matches!(self, Self::SystemPrompt | Self::Skills)
    }
}

impl<'de> Deserialize<'de> for ContextCategory {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::from_name(&raw).unwrap_or(Self::SystemPrompt))
    }
}

/// The usage of one logical prompt category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextPartUsage {
    /// Tokens attributed to the category.
    pub tokens: u64,
    /// `tokens` as a fraction of the context-window limit, when one is known.
    pub percentage_of_window: Option<f64>,
    /// `tokens` as a fraction of the request input, when that is non-zero.
    pub percentage_of_input: Option<f64>,
    /// Whether the count is an estimate rather than a native tokenizer result.
    pub estimated: bool,
    /// The tokenizer that produced the count, when one can be named.
    pub tokenizer: Option<String>,
    /// Where the counted payload came from (`system_message`, `tools`, ...).
    pub carrier: Option<String>,
    /// How the token count was produced, for example `tiktoken`.
    pub source: String,
    /// The post-reconciliation source, for example `provider_usage_residual`.
    pub measurement_source: String,
    /// Why a fallback strategy was used, when one was.
    pub fallback_reason: Option<String>,
    /// The model whose vocabulary had to stand in for the exact tokenizer.
    pub fallback_tokenizer_model: Option<String>,
    /// Reserved: the final-window position of the logical fragment.
    pub order: Option<i64>,
    /// Reserved: a stable id for the logical fragment.
    pub stable_id: Option<String>,
}

impl Default for ContextPartUsage {
    fn default() -> Self {
        Self {
            tokens: 0,
            percentage_of_window: None,
            percentage_of_input: None,
            estimated: true,
            tokenizer: None,
            carrier: None,
            source: NOT_REPORTED.to_owned(),
            measurement_source: NOT_REPORTED.to_owned(),
            fallback_reason: None,
            fallback_tokenizer_model: None,
            order: None,
            stable_id: None,
        }
    }
}

impl ContextPartUsage {
    /// Whether the part carries a locally measured or provider-derived count.
    #[must_use]
    pub fn is_reported(&self) -> bool {
        self.source != NOT_REPORTED
    }
}

/// The window-level usage of one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextWindowUsage {
    /// The authoritative input tokens: the provider total when one is known.
    pub tokens: u64,
    /// How `tokens` was produced (`provider_usage`, a tokenizer source, ...).
    pub tokens_source: String,
    /// The context-window limit, when one is configured.
    pub limit_tokens: Option<u64>,
    /// `tokens` as a fraction of the limit.
    pub occupancy_rate: Option<f64>,
    /// The local measurement total, before provider reconciliation.
    pub local_estimated_input_tokens: u64,
    /// `provider_total - local_total`, a diagnostic and never a correction.
    pub reconciliation_delta_tokens: Option<i64>,
    /// The id of the context snapshot this window was measured from.
    pub context_snapshot_id: Option<String>,
    /// The four category parts, keyed by [`ContextCategory::as_str`].
    pub parts: BTreeMap<String, ContextPartUsage>,
}

impl Default for ContextWindowUsage {
    fn default() -> Self {
        Self {
            tokens: 0,
            tokens_source: NOT_REPORTED.to_owned(),
            limit_tokens: None,
            occupancy_rate: None,
            local_estimated_input_tokens: 0,
            reconciliation_delta_tokens: None,
            context_snapshot_id: None,
            parts: BTreeMap::new(),
        }
    }
}

/// The normalized KV-cache usage of one request.
///
/// `None` means "not reported"; an inconsistent provider report keeps its raw
/// values and sets [`RequestKVCacheUsage::invalid_reason`] instead of being
/// clamped into a plausible miss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RequestKVCacheUsage {
    /// The prompt tokens of the request, as reported by the provider.
    pub input_tokens: Option<u64>,
    /// Prompt tokens served from the provider's prefix cache.
    pub cache_read_tokens: Option<u64>,
    /// Prompt tokens that had to be recomputed.
    pub cache_miss_tokens: Option<u64>,
    /// Prompt tokens written into the provider's prefix cache.
    pub cache_write_tokens: Option<u64>,
    /// `cache_read_tokens / input_tokens`, absent when the report is invalid.
    pub hit_rate: Option<f64>,
    /// `invalid`, `not_reported`, or the trusted/recomputed `miss`,
    /// `partial_hit` / `full_hit` verdict.
    pub status: String,
    /// Where the numbers came from (`provider_usage`, a provider cache source).
    pub source: String,
    /// Whether the provider itself flagged the cache numbers as authoritative.
    pub authoritative: bool,
    /// Whether `cache_miss_tokens` was derived as `input - read`.
    pub miss_tokens_derived: bool,
    /// The cache mode; left empty by the normalizer and carried by the
    /// snapshot/aggregation key instead.
    pub cache_mode: Option<String>,
    /// The cache scope; left empty by the normalizer and carried by the
    /// snapshot/aggregation key instead.
    pub cache_scope: Option<String>,
    /// Why the provider report is inconsistent, when it is.
    pub invalid_reason: Option<String>,
}

impl Default for RequestKVCacheUsage {
    fn default() -> Self {
        Self {
            input_tokens: None,
            cache_read_tokens: None,
            cache_miss_tokens: None,
            cache_write_tokens: None,
            hit_rate: None,
            status: STATUS_NOT_REPORTED.to_owned(),
            source: NOT_REPORTED.to_owned(),
            authoritative: false,
            miss_tokens_derived: false,
            cache_mode: None,
            cache_scope: None,
            invalid_reason: None,
        }
    }
}

impl RequestKVCacheUsage {
    /// Whether the request reported any cache counter at all.
    #[must_use]
    pub fn reports_cache_fields(&self) -> bool {
        self.cache_read_tokens.is_some()
            || self.cache_miss_tokens.is_some()
            || self.cache_write_tokens.is_some()
    }
}

/// The token-weighted KV-cache rollup of one aggregation scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionKVCacheUsage {
    /// The aggregation scope name; always `session`.
    pub scope: String,
    /// How the rollup was computed; always `token_weighted`.
    pub aggregation_method: String,
    /// The identity all recorded calls shared.
    pub aggregation_key: CacheAggregationKey,
    /// Calls recorded for the scope, including repeated request ids once.
    pub calls_total: u64,
    /// Calls whose numbers entered the totals.
    pub calls_observed: u64,
    /// Calls excluded because their numbers were incomplete or inconsistent.
    pub calls_partial: u64,
    /// Calls that reported no usage at all.
    pub calls_not_reported: u64,
    /// Observed input tokens.
    pub input_tokens_total: u64,
    /// Observed cache-read tokens.
    pub cache_read_tokens_total: u64,
    /// Observed cache-miss tokens.
    pub cache_miss_tokens_total: u64,
    /// Observed cache-write tokens, only when every observed call reported one.
    pub cache_write_tokens_total: Option<u64>,
    /// `none`, `partial` or `complete` coverage of the write counter.
    pub cache_write_coverage: String,
    /// `input_tokens_total / calls_observed`.
    pub average_input_per_call: Option<f64>,
    /// `cache_read_tokens_total / input_tokens_total`.
    pub weighted_hit_rate: Option<f64>,
    /// `cache_read_tokens_total / calls_observed`.
    pub average_cache_read_per_call: Option<f64>,
    /// `cache_miss_tokens_total / calls_observed`.
    pub average_cache_miss_per_call: Option<f64>,
    /// `not_reported`, `invalid`, `miss`, `full_hit` or `partial_hit`.
    pub status: String,
    /// `not_reported`, `authoritative`, `derived`, `invalid`, `partial` or
    /// `mixed`.
    pub data_quality: String,
}

impl Default for SessionKVCacheUsage {
    fn default() -> Self {
        Self {
            scope: "session".to_owned(),
            aggregation_method: "token_weighted".to_owned(),
            aggregation_key: CacheAggregationKey::default(),
            calls_total: 0,
            calls_observed: 0,
            calls_partial: 0,
            calls_not_reported: 0,
            input_tokens_total: 0,
            cache_read_tokens_total: 0,
            cache_miss_tokens_total: 0,
            cache_write_tokens_total: None,
            cache_write_coverage: "none".to_owned(),
            average_input_per_call: None,
            weighted_hit_rate: None,
            average_cache_read_per_call: None,
            average_cache_miss_per_call: None,
            status: STATUS_NOT_REPORTED.to_owned(),
            data_quality: STATUS_NOT_REPORTED.to_owned(),
        }
    }
}

/// How the window total and its category sources were produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextUsageMeasurement {
    /// The merged `source` of the four parts.
    pub category_source: String,
    /// How the window total was produced (`provider_usage` or a merge of part
    /// sources).
    pub total_source: String,
    /// The merged tokenizer of the four parts.
    pub tokenizer: Option<String>,
    /// Whether any reported part is an estimate.
    pub estimated: bool,
    /// Whether the total came from the provider.
    pub authoritative_total: bool,
    /// The measurement contract version.
    pub measurement_version: String,
    /// The id of the context snapshot the measurement describes.
    pub context_snapshot_id: Option<String>,
}

impl Default for ContextUsageMeasurement {
    fn default() -> Self {
        Self {
            category_source: NOT_REPORTED.to_owned(),
            total_source: NOT_REPORTED.to_owned(),
            tokenizer: None,
            estimated: true,
            authoritative_total: false,
            measurement_version: MEASUREMENT_VERSION.to_owned(),
            context_snapshot_id: None,
        }
    }
}

/// The outcome of the measurement cache for one analysis call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementCacheHit {
    /// Nothing was reused; every part had to be measured.
    #[default]
    Miss,
    /// Some parts were reused, others had to be measured.
    Partial,
    /// Every part was reused from the cache.
    Full,
}

impl MeasurementCacheHit {
    /// The wire name of the hit kind.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
            Self::Miss => "miss",
        }
    }
}

/// Tokenizer-cache observability for one analysis call.
///
/// This describes tokenizer work only; it says nothing about the provider's
/// prompt cache.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeasurementCacheInfo {
    /// Whether every, some, or no part was reused from the measurement cache.
    pub hit: MeasurementCacheHit,
    /// The parts that were reused, sorted.
    pub reused_parts: Vec<String>,
    /// The parts that had to be measured, sorted.
    pub recomputed_parts: Vec<String>,
}

impl MeasurementCacheInfo {
    /// Derives the hit kind from the two part lists, matching Python's
    /// `_public_cache_info`.
    #[must_use]
    pub fn from_parts(mut reused_parts: Vec<String>, mut recomputed_parts: Vec<String>) -> Self {
        reused_parts.sort();
        reused_parts.dedup();
        recomputed_parts.sort();
        recomputed_parts.dedup();
        let hit = if !reused_parts.is_empty() && recomputed_parts.is_empty() {
            MeasurementCacheHit::Full
        } else if reused_parts.is_empty() {
            MeasurementCacheHit::Miss
        } else {
            MeasurementCacheHit::Partial
        };
        Self {
            hit,
            reused_parts,
            recomputed_parts,
        }
    }
}

/// The per-request and per-session KV-cache blocks of a snapshot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KvCacheUsage {
    /// The normalized usage of the emitted request, when one is known.
    pub request: Option<RequestKVCacheUsage>,
    /// The session-to-date rollup of the aggregation scope, when one exists.
    pub session: Option<SessionKVCacheUsage>,
}

impl KvCacheUsage {
    /// Builds the block from an optional request usage and an optional session
    /// rollup.
    #[must_use]
    pub fn new(request: Option<RequestKVCacheUsage>, session: Option<SessionKVCacheUsage>) -> Self {
        Self { request, session }
    }

    /// The token-weighted session hit rate, when the rollup carries one.
    #[must_use]
    pub fn session_hit_rate(&self) -> Option<f64> {
        self.session
            .as_ref()
            .and_then(|session| session.weighted_hit_rate)
    }
}

/// The owner/lineage attribution carried by a report and its snapshots.
///
/// These fields keep a teammate or child agent that shares a product session
/// from being treated as the same context or cache owner.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Attribution {
    /// The product-level session id.
    pub product_session_id: Option<String>,
    /// The execution-level session id.
    pub execution_session_id: Option<String>,
    /// The id of the context owner.
    pub context_owner_id: Option<String>,
    /// The team id, for agents running inside a team.
    pub team_id: Option<String>,
    /// The team member name.
    pub member_name: Option<String>,
    /// The invocation id of the current call.
    pub invocation_id: Option<String>,
    /// The invocation id of the parent call.
    pub parent_invocation_id: Option<String>,
    /// The delegation id linking a child agent to its delegating call.
    pub delegation_id: Option<String>,
    /// The agent path from the root agent down to this one.
    pub agent_path: Vec<String>,
    /// The nesting depth of this agent.
    pub depth: u32,
    /// The provider cache identity of this owner.
    pub cache_identity: Option<String>,
    /// The cache identity of the parent owner.
    pub parent_cache_identity: Option<String>,
    /// The provider cache mode (`provider`, `manual`, ...).
    pub cache_mode: Option<String>,
    /// The provider cache scope.
    pub cache_scope: Option<String>,
}

impl Attribution {
    /// Copies the attribution carried by `report`.
    #[must_use]
    pub fn from_report(report: &ContextWindowTokenReport) -> Self {
        Self {
            product_session_id: report.product_session_id.clone(),
            execution_session_id: report.execution_session_id.clone(),
            context_owner_id: report.context_owner_id.clone(),
            team_id: report.team_id.clone(),
            member_name: report.member_name.clone(),
            invocation_id: report.invocation_id.clone(),
            parent_invocation_id: report.parent_invocation_id.clone(),
            delegation_id: report.delegation_id.clone(),
            agent_path: report.agent_path.clone(),
            depth: report.depth,
            cache_identity: report.cache_identity.clone(),
            parent_cache_identity: report.parent_cache_identity.clone(),
            cache_mode: report.cache_mode.clone(),
            cache_scope: report.cache_scope.clone(),
        }
    }
}

/// The emitted `context.usage` payload of one request.
///
/// Serializes with `"type": "context.usage"` (see
/// [`CONTEXT_USAGE_EVENT_TYPE`]); the `type` key is exposed as
/// [`ContextUsageSnapshot::event_type`] because `type` is a Rust keyword.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextUsageSnapshot {
    /// The event type; serialized as the `type` key.
    #[serde(rename = "type", alias = "event_type")]
    pub event_type: String,
    /// The payload schema version.
    pub schema_version: String,
    /// `pre_call`, `post_call` or `post_compact`.
    pub phase: String,
    /// The id of the request the snapshot describes.
    pub request_id: String,
    /// The session id, kept for compatibility.
    pub session_id: String,
    /// The product-level session id.
    pub product_session_id: Option<String>,
    /// The execution-level session id.
    pub execution_session_id: Option<String>,
    /// The id of the context owner.
    pub context_owner_id: Option<String>,
    /// The id of the agent that emitted the snapshot.
    pub agent_id: Option<String>,
    /// The team id, for agents running inside a team.
    pub team_id: Option<String>,
    /// The team member name.
    pub member_name: Option<String>,
    /// The invocation id of the current call.
    pub invocation_id: Option<String>,
    /// The invocation id of the parent call.
    pub parent_invocation_id: Option<String>,
    /// The delegation id linking a child agent to its delegating call.
    pub delegation_id: Option<String>,
    /// The agent path from the root agent down to this one.
    pub agent_path: Vec<String>,
    /// The nesting depth of this agent.
    pub depth: u32,
    /// The provider cache identity of this owner.
    pub cache_identity: Option<String>,
    /// The cache identity of the parent owner.
    pub parent_cache_identity: Option<String>,
    /// The provider cache mode.
    pub cache_mode: Option<String>,
    /// The provider cache scope.
    pub cache_scope: Option<String>,
    /// The emission sequence number of the request.
    pub sequence: u64,
    /// The provider id, when known.
    pub provider: Option<String>,
    /// The model id, when known.
    pub model: Option<String>,
    /// The deployment (`api_base`-level) id, when known.
    pub deployment: Option<String>,
    /// The UTC timestamp of the emission (RFC 3339).
    pub timestamp: String,
    /// The id of the context snapshot this payload describes.
    pub context_snapshot_id: Option<String>,
    /// Free-form description of what changed in the window.
    pub context_changes: BTreeMap<String, Value>,
    /// The measurement-cache outcome of the analysis call.
    pub measurement_cache: MeasurementCacheInfo,
    /// The window usage and its four parts.
    pub context_window: ContextWindowUsage,
    /// The four category parts, keyed by [`ContextCategory::as_str`].
    pub parts: BTreeMap<String, ContextPartUsage>,
    /// The per-request and per-session KV-cache blocks.
    pub kv_cache: KvCacheUsage,
    /// The session token-weighted hit rate, duplicated for single-number
    /// consumers.
    pub session_kv_cache_hit_rate: Option<f64>,
    /// How the measurement was produced.
    pub measurement: ContextUsageMeasurement,
    /// The conversation statistics of the measured window, when supplied.
    pub context_stats: Option<ContextStats>,
}

impl Default for ContextUsageSnapshot {
    fn default() -> Self {
        Self {
            event_type: CONTEXT_USAGE_EVENT_TYPE.to_owned(),
            schema_version: CONTEXT_USAGE_SCHEMA_VERSION.to_owned(),
            phase: PHASE_PRE_CALL.to_owned(),
            request_id: String::new(),
            session_id: String::new(),
            product_session_id: None,
            execution_session_id: None,
            context_owner_id: None,
            agent_id: None,
            team_id: None,
            member_name: None,
            invocation_id: None,
            parent_invocation_id: None,
            delegation_id: None,
            agent_path: Vec::new(),
            depth: 0,
            cache_identity: None,
            parent_cache_identity: None,
            cache_mode: None,
            cache_scope: None,
            sequence: 0,
            provider: None,
            model: None,
            deployment: None,
            timestamp: String::new(),
            context_snapshot_id: None,
            context_changes: BTreeMap::new(),
            measurement_cache: MeasurementCacheInfo::default(),
            context_window: ContextWindowUsage::default(),
            parts: BTreeMap::new(),
            kv_cache: KvCacheUsage::default(),
            session_kv_cache_hit_rate: None,
            measurement: ContextUsageMeasurement::default(),
            context_stats: None,
        }
    }
}

impl ContextUsageSnapshot {
    /// Copies the owner/lineage attribution of `attribution` into the payload.
    pub fn set_attribution(&mut self, attribution: &Attribution) {
        self.product_session_id
            .clone_from(&attribution.product_session_id);
        self.execution_session_id
            .clone_from(&attribution.execution_session_id);
        self.context_owner_id
            .clone_from(&attribution.context_owner_id);
        self.team_id.clone_from(&attribution.team_id);
        self.member_name.clone_from(&attribution.member_name);
        self.invocation_id.clone_from(&attribution.invocation_id);
        self.parent_invocation_id
            .clone_from(&attribution.parent_invocation_id);
        self.delegation_id.clone_from(&attribution.delegation_id);
        self.agent_path.clone_from(&attribution.agent_path);
        self.depth = attribution.depth;
        self.cache_identity.clone_from(&attribution.cache_identity);
        self.parent_cache_identity
            .clone_from(&attribution.parent_cache_identity);
        self.cache_mode.clone_from(&attribution.cache_mode);
        self.cache_scope.clone_from(&attribution.cache_scope);
    }
}

/// The local token report of one final context window.
///
/// A report is immutable by convention: it carries the revisions, signatures
/// and measurement cache outcome of the window it measured, so a later
/// concurrent window cannot rewrite a post-call event retroactively. It is the
/// input of [`crate::usage::ContextUsageAnalyzer::analyze_from_report`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextWindowTokenReport {
    /// The id of the context snapshot the report was measured from.
    pub context_snapshot_id: String,
    /// The id of the context that produced the window.
    pub context_id: String,
    /// The session id, kept for compatibility.
    pub session_id: String,
    /// The product-level session id.
    pub product_session_id: Option<String>,
    /// The execution-level session id.
    pub execution_session_id: Option<String>,
    /// The id of the context owner.
    pub context_owner_id: Option<String>,
    /// The id of the agent that owns the context.
    pub agent_id: Option<String>,
    /// The team id, for agents running inside a team.
    pub team_id: Option<String>,
    /// The team member name.
    pub member_name: Option<String>,
    /// The invocation id of the current call.
    pub invocation_id: Option<String>,
    /// The invocation id of the parent call.
    pub parent_invocation_id: Option<String>,
    /// The delegation id linking a child agent to its delegating call.
    pub delegation_id: Option<String>,
    /// The agent path from the root agent down to this one.
    pub agent_path: Vec<String>,
    /// The nesting depth of this agent.
    pub depth: u32,
    /// The provider cache identity of this owner.
    pub cache_identity: Option<String>,
    /// The cache identity of the parent owner.
    pub parent_cache_identity: Option<String>,
    /// The provider cache mode.
    pub cache_mode: Option<String>,
    /// The provider cache scope.
    pub cache_scope: Option<String>,
    /// The message-list revision the report measured.
    pub message_revision: u64,
    /// The prompt revision the report measured.
    pub prompt_revision: u64,
    /// The skills revision the report measured.
    pub skills_revision: u64,
    /// The attachment revision the report measured.
    pub attachment_revision: u64,
    /// A stable signature of the tool definitions that were sent.
    pub tools_signature: Option<String>,
    /// A stable identity of the tokenizer that produced the counts.
    pub tokenizer_identity: Option<String>,
    /// The measurement contract version.
    pub measurement_version: String,
    /// Additional per-component signatures the caller wants recorded.
    pub component_signatures: BTreeMap<String, String>,
    /// The window usage and its four parts.
    pub context_window: ContextWindowUsage,
    /// The four category parts, keyed by [`ContextCategory::as_str`].
    pub parts: BTreeMap<String, ContextPartUsage>,
    /// The conversation statistics of the measured window.
    pub context_stats: ContextStats,
    /// How the measurement was produced.
    pub measurement: ContextUsageMeasurement,
    /// Free-form description of what changed in the window.
    pub context_changes: BTreeMap<String, Value>,
    /// The measurement-cache outcome of the measurement call.
    pub measurement_cache: MeasurementCacheInfo,
}

impl Default for ContextWindowTokenReport {
    fn default() -> Self {
        Self {
            context_snapshot_id: String::new(),
            context_id: String::new(),
            session_id: String::new(),
            product_session_id: None,
            execution_session_id: None,
            context_owner_id: None,
            agent_id: None,
            team_id: None,
            member_name: None,
            invocation_id: None,
            parent_invocation_id: None,
            delegation_id: None,
            agent_path: Vec::new(),
            depth: 0,
            cache_identity: None,
            parent_cache_identity: None,
            cache_mode: None,
            cache_scope: None,
            message_revision: 0,
            prompt_revision: 0,
            skills_revision: 0,
            attachment_revision: 0,
            tools_signature: None,
            tokenizer_identity: None,
            measurement_version: MEASUREMENT_VERSION.to_owned(),
            component_signatures: BTreeMap::new(),
            context_window: ContextWindowUsage::default(),
            parts: BTreeMap::new(),
            context_stats: ContextStats::default(),
            measurement: ContextUsageMeasurement::default(),
            context_changes: BTreeMap::new(),
            measurement_cache: MeasurementCacheInfo::default(),
        }
    }
}

impl ContextWindowTokenReport {
    /// Copies the owner/lineage attribution of the report.
    #[must_use]
    pub fn attribution(&self) -> Attribution {
        Attribution::from_report(self)
    }
}

/// The measurement configuration of one analyzer: the counting strategy and
/// the metadata that identifies it.
///
/// The class and tokenizer are part of every measurement-cache key, so two
/// analyzers with different identities never share a cached result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterIdentity {
    /// The counter's Rust type name, as used in the cache key.
    pub class: String,
    /// How the counter produces counts.
    pub source: TokenizerSource,
    /// The tokenizer name, when one can be named.
    pub tokenizer: Option<String>,
    /// Why a fallback strategy was chosen, when one was.
    pub fallback_reason: Option<String>,
    /// The model whose vocabulary had to stand in, when one did.
    pub fallback_tokenizer_model: Option<String>,
}

impl CounterIdentity {
    /// Builds an identity from a class name and a counting source.
    #[must_use]
    pub fn new(class: impl Into<String>, source: TokenizerSource) -> Self {
        Self {
            class: class.into(),
            source,
            tokenizer: None,
            fallback_reason: None,
            fallback_tokenizer_model: None,
        }
    }

    /// Adds a tokenizer name.
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: impl Into<String>) -> Self {
        self.tokenizer = Some(tokenizer.into());
        self
    }

    /// Adds a fallback reason.
    #[must_use]
    pub fn with_fallback_reason(mut self, reason: impl Into<String>) -> Self {
        self.fallback_reason = Some(reason.into());
        self
    }

    /// Adds the model whose vocabulary stood in.
    #[must_use]
    pub fn with_fallback_tokenizer_model(mut self, model: impl Into<String>) -> Self {
        self.fallback_tokenizer_model = Some(model.into());
        self
    }

    /// The identity of an [`openjiuwen_foundation::prompt::ApproxTokenCounter`].
    #[must_use]
    pub fn approx() -> Self {
        Self::new("ApproxTokenCounter", TokenizerSource::StringLengthFallback)
    }

    /// The identity of a [`ModelTokenCounter`].
    #[must_use]
    pub fn model(counter: &ModelTokenCounter) -> Self {
        let source = counter.source();
        let mut identity = Self::new("ModelTokenCounter", source);
        identity.tokenizer = Some(counter.tokenizer().to_owned());
        identity.fallback_reason = counter.fallback_reason().map(str::to_owned);
        if source != TokenizerSource::NativeTokenizer {
            identity.fallback_tokenizer_model = counter.model().map(str::to_owned);
        }
        identity
    }

    /// The identity of a [`SelectedTokenizer`].
    #[must_use]
    pub fn selected(counter: &SelectedTokenizer) -> Self {
        let source = counter.source();
        let mut identity = Self::new("SelectedTokenizer", source);
        identity.tokenizer = counter.tokenizer().map(str::to_owned);
        identity.fallback_reason = counter.fallback_reason().map(str::to_owned);
        if source != TokenizerSource::NativeTokenizer {
            identity.fallback_tokenizer_model = counter.model().map(str::to_owned);
        }
        identity
    }

    /// The wire name of the counting source.
    #[must_use]
    pub fn source_name(&self) -> &'static str {
        source_name(self.source)
    }

    /// Whether the count is a fallback estimate rather than a model tokenizer.
    ///
    /// This is Python's narrow `_fallback` suffix rule, which selects the
    /// reconciliation method: a fallback counter carries the same measurement
    /// error in every category, so the analyzer scales all categories
    /// proportionally instead of keeping the non-message categories fixed.
    ///
    /// [`CounterIdentity::estimated`] is the broader honesty flag: it is `true`
    /// for everything but a native tokenizer, so a resolved `tiktoken` identity
    /// is estimated without being a fallback.
    #[must_use]
    pub fn is_fallback(&self) -> bool {
        is_fallback_source(self.source_name())
    }

    /// Whether a count produced under this identity is an estimate.
    ///
    /// `true` for every source but [`TokenizerSource::NativeTokenizer`], so an
    /// exact `tiktoken` vocabulary and a string-length estimate are both
    /// estimates while only a loaded native artifact counts exactly.
    #[must_use]
    pub fn estimated(&self) -> bool {
        !self.source.is_native()
    }

    /// A stable `class:tokenizer` identity string for reports.
    #[must_use]
    pub fn identity_string(&self) -> String {
        format!(
            "{}:{}",
            self.class,
            self.tokenizer.as_deref().unwrap_or_default()
        )
    }
}

/// A counter that can name its own counting strategy.
///
/// [`CounterIdentity`] is normally supplied by the caller, which means the
/// analyzer cannot tell whether it really describes the counter that runs, and a
/// native identity over an approximate counter would publish an approximate
/// count as exact.
/// Implementing this trait lets [`super::analyzer::ContextUsageAnalyzer::from_counter`]
/// derive the identity from the counter itself, so the estimate flag can never
/// be wrong.
pub trait CounterProvenance {
    /// The identity of the counts this counter produces.
    fn counter_identity(&self) -> CounterIdentity;
}

impl CounterProvenance for openjiuwen_foundation::prompt::ApproxTokenCounter {
    /// An approximate counter is always a string-length estimate.
    ///
    /// A deliberately configured characters-per-token counter
    /// (`TokenCounterKind::Characters`) reports the same identity: the count is
    /// exact by configuration but still an estimate of the model's tokens, so it
    /// keeps `estimated = true` rather than claiming a native vocabulary.
    fn counter_identity(&self) -> CounterIdentity {
        CounterIdentity::approx()
    }
}

impl CounterProvenance for ModelTokenCounter {
    /// A model counter names its resolved vocabulary and degradation reason.
    fn counter_identity(&self) -> CounterIdentity {
        CounterIdentity::model(self)
    }
}

impl CounterProvenance for SelectedTokenizer {
    /// A selection names its resolved strategy and degradation reason.
    fn counter_identity(&self) -> CounterIdentity {
        CounterIdentity::selected(self)
    }
}

/// The wire name of a [`TokenizerSource`].
#[must_use]
pub fn source_name(source: TokenizerSource) -> &'static str {
    match source {
        TokenizerSource::NativeTokenizer => "native_tokenizer",
        TokenizerSource::FamilyTokenizerFallback => "family_tokenizer_fallback",
        TokenizerSource::Tiktoken => "tiktoken",
        TokenizerSource::TiktokenFallback => "tiktoken_fallback",
        TokenizerSource::StringLengthFallback => "string_length_fallback",
    }
}

/// Whether a measurement source is a fallback rather than a model tokenizer.
///
/// Python's rule is a `_fallback` suffix, so `tiktoken_fallback`,
/// `family_tokenizer_fallback` and `string_length_fallback` all count while
/// `native_tokenizer` and `tiktoken` do not.
#[must_use]
pub fn is_fallback_source(source: &str) -> bool {
    source.trim().to_lowercase().ends_with("_fallback")
}
