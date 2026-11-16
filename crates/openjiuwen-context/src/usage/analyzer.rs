//! Four-category measurement with provider reconciliation.
//!
//! Ports `usage/analyzer.py`: [`ContextUsageAnalyzer::analyze`] measures the
//! final window, reconciles it with the provider total when one is known, and
//! returns the emittable [`ContextUsageSnapshot`];
//! [`ContextUsageAnalyzer::build_report`] freezes the same measurement into a
//! [`ContextWindowTokenReport`], and
//! [`ContextUsageAnalyzer::analyze_from_report`] rebuilds a post-call or
//! post-compaction snapshot from that report without calling the tokenizer.
//!
//! The analyzer never decides compression, truncation or routing: it explains
//! the window that processors already produced.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};

use openjiuwen_common::utils::id::now_rfc3339;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, TokenCounter};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::tokenizers::TokenMeasurement;

pub use super::analyzer_options::{
    AnalysisInput, PromptSection, SnapshotOptions, WindowSignatures,
};
use super::context_tokens::{
    canonical_json, count_tools_tokens_with_provenance, ContextTokenCount,
};
use super::models::{
    is_fallback_source, ContextCategory, ContextPartUsage, ContextUsageMeasurement,
    ContextUsageSnapshot, ContextWindowTokenReport, ContextWindowUsage, CounterIdentity,
    CounterProvenance, MeasurementCacheInfo, CONTEXT_USAGE_EVENT_TYPE,
    CONTEXT_USAGE_SCHEMA_VERSION, MEASUREMENT_KEY_VERSION, MEASUREMENT_VERSION, NOT_REPORTED,
    SOURCE_PROVIDER_USAGE, SOURCE_PROVIDER_USAGE_PROPORTIONAL, SOURCE_PROVIDER_USAGE_RESIDUAL,
    SOURCE_PROVIDER_USAGE_UNATTRIBUTED,
};
use super::parts::{
    attachment_category, merge_carriers, merge_measurements, merge_names, message_value,
    messages_value, part_from_measurement, part_from_merge, split_fragment_measurement,
    PartMeasurement,
};

/// The metadata key routing a whole message into one category.
pub const ATTACHMENT_CATEGORY_METADATA_KEY: &str = "_context_usage_category";

/// The metadata key naming the carrier of a whole-message attachment.
pub const ATTACHMENT_CARRIER_METADATA_KEY: &str = "_context_usage_carrier";

/// The metadata key splitting one message's tokens across logical fragments.
pub const ATTACHMENT_FRAGMENTS_METADATA_KEY: &str = "_context_usage_fragments";

/// The default capacity of an analyzer's measurement cache.
pub const DEFAULT_MEASUREMENT_CACHE_CAPACITY: usize = 512;

/// The fallback reason of a messages part holding the provider residual.
const RESIDUAL_REASON: &str = "provider_total_residual";

/// The fallback reason of a provider total that is below the fixed categories.
const RESIDUAL_NEGATIVE_REASON: &str = "provider_total_below_fixed_categories";

/// The fallback reason of a proportional reconciliation.
const PROPORTIONAL_REASON: &str = "provider_total_proportional_reconciliation";

/// The fallback reason of an unattributable provider total.
const UNATTRIBUTED_REASON: &str = "provider_total_without_local_measurement";

/// The measured window and its cache outcome, before reconciliation.
struct MeasuredWindow {
    /// The four category parts.
    parts: BTreeMap<String, ContextPartUsage>,
    /// The local measurement total.
    local_total: u64,
    /// The measurement-cache outcome of the call.
    cache_info: MeasurementCacheInfo,
}

/// The measured parts of one window, grouped by category.
#[derive(Debug, Default)]
struct CategoryGroups {
    /// Measurements routed to `system_prompt`.
    system_prompt: Vec<PartMeasurement>,
    /// Measurements routed to `skills`.
    skills: Vec<PartMeasurement>,
    /// Carriers routed to `system_prompt`.
    system_carriers: Vec<String>,
    /// Carriers routed to `skills`.
    skills_carriers: Vec<String>,
    /// Messages that stay in the `messages` category.
    ordinary: Vec<BaseMessage>,
}

impl CategoryGroups {
    /// Routes one measurement and its carrier into its category.
    fn push(&mut self, category: ContextCategory, part: PartMeasurement, carrier: String) {
        if category == ContextCategory::Skills {
            self.skills.push(part);
            self.skills_carriers.push(carrier);
        } else {
            self.system_prompt.push(part);
            self.system_carriers.push(carrier);
        }
    }
}

/// Builds the four parts out of the measured groups.
fn assemble_parts(
    groups: &CategoryGroups,
    tools: &TokenMeasurement,
    messages: &TokenMeasurement,
    identity: &CounterIdentity,
) -> BTreeMap<String, ContextPartUsage> {
    let mut parts = BTreeMap::new();
    parts.insert(
        ContextCategory::SystemPrompt.as_str().to_owned(),
        part_from_merge(
            &merge_measurements(&groups.system_prompt),
            &merge_carriers(&groups.system_carriers),
        ),
    );
    parts.insert(
        ContextCategory::Skills.as_str().to_owned(),
        part_from_merge(
            &merge_measurements(&groups.skills),
            &merge_carriers(&groups.skills_carriers),
        ),
    );
    parts.insert(
        ContextCategory::Tools.as_str().to_owned(),
        part_from_measurement(tools, identity, "tools"),
    );
    parts.insert(
        ContextCategory::Messages.as_str().to_owned(),
        part_from_measurement(messages, identity, "context_message"),
    );
    parts
}

/// The parts reused and recomputed during one analysis call.
#[derive(Debug, Default)]
struct CacheTracker {
    /// Parts served from the cache.
    reused: BTreeSet<String>,
    /// Parts that had to be measured.
    recomputed: BTreeSet<String>,
}

impl CacheTracker {
    /// Records one part outcome.
    fn record(&mut self, part_name: &str, reused: bool) {
        let bucket = if reused {
            &mut self.reused
        } else {
            &mut self.recomputed
        };
        bucket.insert(part_name.to_owned());
    }

    /// The public cache info of the call.
    fn info(&self) -> MeasurementCacheInfo {
        MeasurementCacheInfo::from_parts(
            self.reused.iter().cloned().collect(),
            self.recomputed.iter().cloned().collect(),
        )
    }
}

/// A bounded memo of token measurements, owned by one analyzer.
///
/// Only [`TokenMeasurement`] values are memoized; window assembly and the
/// processors run on every request. Entries are evicted first-in first-out once
/// [`MeasurementCache::capacity`] is reached, and a capacity of `0` disables
/// the cache.
#[derive(Debug, Clone)]
pub struct MeasurementCache {
    /// The maximum number of entries.
    capacity: usize,
    /// The memoized measurements.
    entries: HashMap<String, TokenMeasurement>,
    /// Insertion order, used for FIFO eviction.
    order: VecDeque<String>,
}

impl MeasurementCache {
    /// Creates a cache holding at most `capacity` measurements.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// The maximum number of entries.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The current number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up one measurement.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&TokenMeasurement> {
        self.entries.get(key)
    }

    /// Stores one measurement, evicting the oldest entries when full.
    pub fn insert(&mut self, key: String, measurement: TokenMeasurement) {
        if self.capacity == 0 {
            return;
        }
        if let Some(slot) = self.entries.get_mut(&key) {
            *slot = measurement;
            return;
        }
        while self.entries.len() >= self.capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, measurement);
    }

    /// Drops every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}

impl Default for MeasurementCache {
    fn default() -> Self {
        Self::new(DEFAULT_MEASUREMENT_CACHE_CAPACITY)
    }
}

/// The fallback reason published for a native claim the analyzer cannot verify.
const UNVERIFIED_NATIVE_IDENTITY_REASON: &str = "unverified_native_tokenizer_identity";

/// The fallback reason published when a declared identity disagrees with the
/// counter that runs.
const DISAGREEING_IDENTITY_REASON: &str = "declared_identity_disagrees_with_counter";

/// Computes local category estimates and reconciles them with provider totals.
pub struct ContextUsageAnalyzer {
    /// The counter every category count comes from.
    counter: Arc<dyn TokenCounter + Send + Sync>,
    /// The counting strategy and its cache-key identity.
    identity: CounterIdentity,
    /// Whether `identity` was derived from `counter` rather than supplied.
    identity_verified: bool,
    /// The identity a caller declared when it disagreed with the counter.
    declared_identity: Option<CounterIdentity>,
    /// The model the request is sent to, when known.
    model: String,
    /// The context-window limit, when one is configured.
    context_window_limit: Option<u64>,
    /// The per-analyzer measurement cache.
    measurement_cache: Mutex<MeasurementCache>,
}

impl std::fmt::Debug for ContextUsageAnalyzer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContextUsageAnalyzer")
            .field("identity", &self.identity)
            .field("model", &self.model)
            .field("context_window_limit", &self.context_window_limit)
            .field("measurement_cache", &self.measurement_cache)
            .finish_non_exhaustive()
    }
}

/// The analyzer a caller gets without configuring anything: foundation's
/// approximate counter with its own identity, no model and no window limit.
///
/// The window limit stays unset so the engine substitutes the window it
/// resolves for the context; a caller that wants a specific limit (or a
/// model-aware counter) builds the analyzer with [`ContextUsageAnalyzer::new`].
impl Default for ContextUsageAnalyzer {
    fn default() -> Self {
        Self::new(
            Arc::new(ApproxTokenCounter::new()),
            CounterIdentity::approx(),
        )
    }
}

impl ContextUsageAnalyzer {
    /// Creates an analyzer over `counter`.
    ///
    /// The identity names the counting strategy; it is part of every
    /// measurement-cache key and selects the reconciliation method.
    ///
    /// The pairing is *declared*, not verified: this constructor cannot see what
    /// `counter` really is, so a native identity is published as an estimate
    /// (with the `unverified_native_tokenizer_identity` reason) until the caller either
    /// builds the analyzer with [`Self::from_counter`] or vouches for the
    /// pairing with [`Self::with_verified_identity`].
    #[must_use]
    pub fn new(counter: Arc<dyn TokenCounter + Send + Sync>, identity: CounterIdentity) -> Self {
        Self::assemble(counter, identity, false, None)
    }

    /// Creates an analyzer whose published identity comes from `counter`.
    ///
    /// The identity is derived from the counter through
    /// [`CounterProvenance`], so the estimate flag and the published source
    /// always describe the counter that actually runs: an approximate counter
    /// publishes an estimate and a native artifact publishes an exact count.
    #[must_use]
    pub fn from_counter<C>(counter: Arc<C>) -> Self
    where
        C: CounterProvenance + TokenCounter + Send + Sync + 'static,
    {
        let identity = counter.counter_identity();
        Self::assemble(counter, identity, true, None)
    }

    /// Creates an analyzer for `counter` that records a disagreeing `declared` identity.
    ///
    /// The counter's own provenance wins, because it is what actually runs; the
    /// disagreement is logged at `debug` and republished as
    /// `declared_identity_disagrees_with_counter` in every measurement's
    /// `fallback_reason`.
    #[must_use]
    pub fn from_counter_with<C>(counter: Arc<C>, declared: CounterIdentity) -> Self
    where
        C: CounterProvenance + TokenCounter + Send + Sync + 'static,
    {
        let executed = counter.counter_identity();
        let disagreement = (executed != declared).then(|| declared.clone());
        if disagreement.is_some() {
            tracing::debug!(
                target: "openjiuwen_context::usage",
                executed = %executed.identity_string(),
                executed_source = executed.source_name(),
                declared = %declared.identity_string(),
                declared_source = declared.source_name(),
                "declared counter identity disagrees with the counter that runs; the counter wins"
            );
        }
        Self::assemble(counter, executed, true, disagreement)
    }

    /// Assembles an analyzer from its counter, identity and verification state.
    fn assemble(
        counter: Arc<dyn TokenCounter + Send + Sync>,
        identity: CounterIdentity,
        identity_verified: bool,
        declared_identity: Option<CounterIdentity>,
    ) -> Self {
        Self {
            counter,
            identity,
            identity_verified,
            declared_identity,
            model: String::new(),
            context_window_limit: None,
            measurement_cache: Mutex::new(MeasurementCache::default()),
        }
    }

    /// Declares that the supplied identity really describes the counter.
    ///
    /// [`Self::new`] cannot check the pairing, so a native claim is published as
    /// an estimate. Call this when the counter is a custom adapter over a native
    /// tokenizer and the identity is known to be right.
    #[must_use]
    pub fn with_verified_identity(mut self) -> Self {
        self.identity_verified = true;
        self
    }

    /// Sets the model name reported in the snapshot and used in cache keys.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the context-window limit used for occupancy and percentages.
    #[must_use]
    pub fn with_context_window_limit(mut self, limit: Option<u64>) -> Self {
        self.context_window_limit = limit;
        self
    }

    /// Sets the capacity of the measurement cache; `0` disables it.
    #[must_use]
    pub fn with_measurement_cache_capacity(mut self, capacity: usize) -> Self {
        self.measurement_cache = Mutex::new(MeasurementCache::new(capacity));
        self
    }

    /// The model name reported in snapshots.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The context-window limit, when one is configured.
    #[must_use]
    pub fn context_window_limit(&self) -> Option<u64> {
        self.context_window_limit
    }

    /// The counting strategy identity.
    ///
    /// This is the declared or derived identity itself; use
    /// [`Self::published_identity`] for the identity a measurement really
    /// carries, which additionally names an unverified native claim.
    #[must_use]
    pub fn counter_identity(&self) -> &CounterIdentity {
        &self.identity
    }

    /// The counter the analyzer measures with.
    #[must_use]
    pub fn counter(&self) -> &Arc<dyn TokenCounter + Send + Sync> {
        &self.counter
    }

    /// Whether the identity was derived from the counter instead of supplied.
    #[must_use]
    pub fn identity_verified(&self) -> bool {
        self.identity_verified
    }

    /// The identity published with every measurement.
    ///
    /// The source and tokenizer are the identity's; the fallback reason names an
    /// unverified native claim or a disagreement with the counter, so a reader
    /// can never mistake either for an exact count.
    #[must_use]
    pub fn published_identity(&self) -> CounterIdentity {
        let mut identity = self.identity.clone();
        identity.fallback_reason = self.published_reason();
        identity
    }

    /// Whether the count this analyzer produces is an estimate.
    ///
    /// Derived from the counter that actually runs: the declared source decides
    /// for a verified pairing, and an unverified native claim is an estimate
    /// because nothing can prove it.
    fn counted_estimated(&self) -> bool {
        self.identity.estimated() || !self.identity_verified
    }

    /// The fallback reason published with every measurement.
    fn published_reason(&self) -> Option<String> {
        if self.declared_identity.is_some() {
            return Some(DISAGREEING_IDENTITY_REASON.to_owned());
        }
        if !self.identity_verified && self.identity.source.is_native() {
            return Some(UNVERIFIED_NATIVE_IDENTITY_REASON.to_owned());
        }
        self.identity.fallback_reason.clone()
    }

    /// A copy of this analyzer that reports `limit` as its context-window limit.
    ///
    /// The counter, its identity, its verification state, the model and the
    /// measurement-cache capacity are kept, so only the window changes.
    #[must_use]
    pub(crate) fn with_derived_limit(&self, limit: u64) -> Self {
        let mut derived = Self::assemble(
            Arc::clone(&self.counter),
            self.identity.clone(),
            self.identity_verified,
            self.declared_identity.clone(),
        );
        derived.model.clone_from(&self.model);
        derived.context_window_limit = Some(limit);
        derived.measurement_cache = Mutex::new(MeasurementCache::new(self.lock_cache().capacity()));
        derived
    }

    /// The current number of memoized measurements.
    #[must_use]
    pub fn measurement_cache_len(&self) -> usize {
        self.lock_cache().len()
    }

    /// Drops every memoized measurement.
    pub fn clear_measurement_cache(&self) {
        self.lock_cache().clear();
    }

    /// Measures the window, reconciles it with `provider_input_tokens`, and
    /// builds the emittable snapshot.
    ///
    /// `provider_input_tokens` is authoritative for the total when present; the
    /// local measurement then either keeps the non-message categories fixed
    /// (native tokenizer) or scales every category proportionally (fallback
    /// counter), so the parts always sum to the provider total.
    #[must_use]
    pub fn analyze(&self, input: &AnalysisInput<'_>) -> ContextUsageSnapshot {
        let measured = self.measure_window(input);
        let MeasuredWindow {
            mut parts,
            local_total,
            cache_info,
        } = measured;
        let reconciliation_delta = reconcile(
            &mut parts,
            input.provider_input_tokens,
            self.identity.is_fallback(),
        );
        let total_tokens = input.provider_input_tokens.unwrap_or(local_total);
        let total_source = if input.provider_input_tokens.is_some() {
            SOURCE_PROVIDER_USAGE.to_owned()
        } else {
            merge_source(&parts)
        };
        let limit = self.context_window_limit;
        apply_percentages(&mut parts, limit, total_tokens);
        let category_source = merge_source(&parts);
        let measurement = ContextUsageMeasurement {
            category_source: category_source.clone(),
            total_source: total_source.clone(),
            tokenizer: merge_tokenizer(&parts),
            estimated: any_reported_estimated(&parts),
            authoritative_total: input.provider_input_tokens.is_some(),
            measurement_version: MEASUREMENT_VERSION.to_owned(),
            context_snapshot_id: input
                .context_snapshot_id
                .and_then(non_empty)
                .map(str::to_owned),
        };
        let context_window = ContextWindowUsage {
            tokens: total_tokens,
            tokens_source: total_source,
            limit_tokens: limit,
            occupancy_rate: ratio(total_tokens, limit),
            local_estimated_input_tokens: local_total,
            reconciliation_delta_tokens: reconciliation_delta,
            context_snapshot_id: input
                .context_snapshot_id
                .and_then(non_empty)
                .map(str::to_owned),
            parts: parts.clone(),
        };
        let kv_cache = input.kv_cache.cloned().unwrap_or_default();
        let attribution = input.attribution.cloned().unwrap_or_default();
        let mut snapshot = ContextUsageSnapshot {
            event_type: CONTEXT_USAGE_EVENT_TYPE.to_owned(),
            schema_version: CONTEXT_USAGE_SCHEMA_VERSION.to_owned(),
            phase: input.phase.to_owned(),
            request_id: input.request_id.to_owned(),
            session_id: input.session_id.to_owned(),
            agent_id: input.agent_id.map(str::to_owned),
            sequence: input.sequence,
            provider: input.provider.map(str::to_owned),
            model: non_empty(&self.model).map(str::to_owned),
            deployment: input.deployment.map(str::to_owned),
            timestamp: now_rfc3339(),
            context_snapshot_id: context_window.context_snapshot_id.clone(),
            context_changes: input.context_changes.cloned().unwrap_or_default(),
            measurement_cache: cache_info,
            context_window,
            parts,
            session_kv_cache_hit_rate: kv_cache.session_hit_rate(),
            kv_cache,
            measurement,
            context_stats: input.context_stats.cloned(),
            ..ContextUsageSnapshot::default()
        };
        snapshot.set_attribution(&attribution);
        snapshot
    }

    /// Freezes the current measurement of `input` into a report.
    ///
    /// No provider reconciliation happens here: the report keeps the local
    /// measurement plus the revisions and signatures the caller supplies, so
    /// [`Self::analyze_from_report`] can later rebuild a snapshot without
    /// touching the tokenizer or a concurrently mutated window.
    #[must_use]
    pub fn build_report(&self, input: &AnalysisInput<'_>) -> ContextWindowTokenReport {
        let measured = self.measure_window(input);
        let MeasuredWindow {
            mut parts,
            local_total,
            cache_info,
        } = measured;
        let limit = self.context_window_limit;
        apply_percentages(&mut parts, limit, local_total);
        let category_source = merge_source(&parts);
        let signatures = input.signatures.copied().unwrap_or_default();
        let measurement_version = signatures
            .measurement_version
            .unwrap_or(MEASUREMENT_VERSION)
            .to_owned();
        let context_snapshot_id = input.context_snapshot_id.unwrap_or_default().to_owned();
        let measurement = ContextUsageMeasurement {
            category_source: category_source.clone(),
            total_source: category_source,
            tokenizer: merge_tokenizer(&parts),
            estimated: any_reported_estimated(&parts),
            authoritative_total: false,
            measurement_version: measurement_version.clone(),
            context_snapshot_id: non_empty(&context_snapshot_id).map(str::to_owned),
        };
        let context_window = ContextWindowUsage {
            tokens: local_total,
            tokens_source: merge_source(&parts),
            limit_tokens: limit,
            occupancy_rate: ratio(local_total, limit),
            local_estimated_input_tokens: local_total,
            reconciliation_delta_tokens: None,
            context_snapshot_id: non_empty(&context_snapshot_id).map(str::to_owned),
            parts: parts.clone(),
        };
        let attribution = input.attribution.cloned().unwrap_or_default();
        ContextWindowTokenReport {
            context_snapshot_id,
            context_id: input.context_id.unwrap_or(input.session_id).to_owned(),
            session_id: input.session_id.to_owned(),
            product_session_id: attribution.product_session_id,
            execution_session_id: attribution.execution_session_id,
            context_owner_id: attribution.context_owner_id,
            agent_id: input.agent_id.map(str::to_owned),
            team_id: attribution.team_id,
            member_name: attribution.member_name,
            invocation_id: attribution.invocation_id,
            parent_invocation_id: attribution.parent_invocation_id,
            delegation_id: attribution.delegation_id,
            agent_path: attribution.agent_path,
            depth: attribution.depth,
            cache_identity: attribution.cache_identity,
            parent_cache_identity: attribution.parent_cache_identity,
            cache_mode: attribution.cache_mode,
            cache_scope: attribution.cache_scope,
            message_revision: signatures.message_revision,
            prompt_revision: signatures.prompt_revision,
            skills_revision: signatures.skills_revision,
            attachment_revision: signatures.attachment_revision,
            tools_signature: signatures.tools_signature.map(str::to_owned),
            tokenizer_identity: signatures
                .tokenizer_identity
                .map(str::to_owned)
                .or_else(|| Some(self.identity.identity_string())),
            measurement_version,
            component_signatures: signatures.component_signatures.cloned().unwrap_or_default(),
            context_window,
            parts,
            context_stats: input.context_stats.cloned().unwrap_or_default(),
            measurement,
            context_changes: input.context_changes.cloned().unwrap_or_default(),
            measurement_cache: cache_info,
        }
    }

    /// Builds a snapshot from an already measured window report.
    ///
    /// No tokenizer method is called: `post_call` and `post_compact` use this
    /// path so a concurrent context mutation cannot replace the report captured
    /// for the corresponding operation.
    #[must_use]
    pub fn analyze_from_report(
        &self,
        report: &ContextWindowTokenReport,
        options: &SnapshotOptions<'_>,
    ) -> ContextUsageSnapshot {
        let mut parts = report.parts.clone();
        let local_total = report.context_window.local_estimated_input_tokens;
        let proportional =
            self.identity.is_fallback() || is_fallback_source(&report.measurement.category_source);
        let reconciliation_delta =
            reconcile(&mut parts, options.provider_input_tokens, proportional);
        let total_tokens = options.provider_input_tokens.unwrap_or(local_total);
        let total_source = if options.provider_input_tokens.is_some() {
            SOURCE_PROVIDER_USAGE.to_owned()
        } else {
            report.context_window.tokens_source.clone()
        };
        apply_percentages(&mut parts, report.context_window.limit_tokens, total_tokens);
        let snapshot_id = non_empty(&report.context_snapshot_id).map(str::to_owned);
        let context_window = ContextWindowUsage {
            tokens: total_tokens,
            tokens_source: total_source.clone(),
            limit_tokens: report.context_window.limit_tokens,
            occupancy_rate: ratio(total_tokens, report.context_window.limit_tokens),
            local_estimated_input_tokens: local_total,
            reconciliation_delta_tokens: reconciliation_delta,
            context_snapshot_id: snapshot_id.clone(),
            parts: parts.clone(),
        };
        let measurement = ContextUsageMeasurement {
            category_source: merge_source(&parts),
            total_source,
            tokenizer: report.measurement.tokenizer.clone(),
            estimated: any_reported_estimated(&parts),
            authoritative_total: options.provider_input_tokens.is_some(),
            measurement_version: report.measurement.measurement_version.clone(),
            context_snapshot_id: snapshot_id.clone(),
        };
        let kv_cache = options.kv_cache.cloned().unwrap_or_default();
        let attribution = options
            .attribution
            .cloned()
            .unwrap_or_else(|| report.attribution());
        let mut snapshot = ContextUsageSnapshot {
            event_type: CONTEXT_USAGE_EVENT_TYPE.to_owned(),
            schema_version: CONTEXT_USAGE_SCHEMA_VERSION.to_owned(),
            phase: options.phase.to_owned(),
            request_id: options.request_id.to_owned(),
            session_id: options.session_id.to_owned(),
            agent_id: options.agent_id.map(str::to_owned),
            sequence: options.sequence,
            provider: options.provider.map(str::to_owned),
            model: non_empty(&self.model).map(str::to_owned),
            deployment: options.deployment.map(str::to_owned),
            timestamp: now_rfc3339(),
            context_snapshot_id: snapshot_id,
            context_changes: report.context_changes.clone(),
            measurement_cache: report.measurement_cache.clone(),
            context_window,
            parts,
            session_kv_cache_hit_rate: kv_cache.session_hit_rate(),
            kv_cache,
            measurement,
            context_stats: None,
            ..ContextUsageSnapshot::default()
        };
        snapshot.set_attribution(&attribution);
        if snapshot.agent_id.is_none() {
            snapshot.agent_id.clone_from(&report.agent_id);
        }
        snapshot
    }

    /// Locks the measurement cache, recovering a poisoned mutex.
    fn lock_cache(&self) -> MutexGuard<'_, MeasurementCache> {
        match self.measurement_cache.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!(
                    target: "openjiuwen_context::usage",
                    "measurement cache mutex was poisoned; recovering the inner cache"
                );
                poisoned.into_inner()
            }
        }
    }

    /// Measures the four categories of `input`, using the measurement cache.
    fn measure_window(&self, input: &AnalysisInput<'_>) -> MeasuredWindow {
        let mut tracker = CacheTracker::default();
        let mut groups = CategoryGroups::default();
        self.measure_system_prompt(input, &mut tracker, &mut groups);
        self.measure_context_messages(input.context_messages, &mut tracker, &mut groups);
        let tools = self.measure_tools(input.tools, &mut tracker);
        let messages = self.measure_messages(&groups.ordinary, &mut tracker);
        let parts = assemble_parts(&groups, &tools, &messages, &self.identity);
        let local_total = parts
            .values()
            .map(|part| part.tokens)
            .fold(0u64, u64::saturating_add);
        MeasuredWindow {
            parts,
            local_total,
            cache_info: tracker.info(),
        }
    }

    /// Measures the system prompt and the skills categories.
    ///
    /// A section list replaces `system_messages` entirely; without sections the
    /// system messages are measured as one carrier and skills report nothing.
    fn measure_system_prompt(
        &self,
        input: &AnalysisInput<'_>,
        tracker: &mut CacheTracker,
        groups: &mut CategoryGroups,
    ) {
        let Some(sections) = input.system_prompt_sections else {
            let messages = input.system_messages.unwrap_or(&[]);
            let measurement = self.measure_cached(
                ContextCategory::SystemPrompt.as_str(),
                &messages_value(messages),
                tracker,
                || self.measurement(self.counter.count_messages(messages) as u64),
            );
            groups.push(
                ContextCategory::SystemPrompt,
                PartMeasurement::from_measurement(&measurement, &self.identity),
                "system_message".to_owned(),
            );
            groups.skills.push(PartMeasurement::not_reported());
            return;
        };
        for section in sections {
            let category = if section.category.accepts_attachment() {
                section.category
            } else {
                ContextCategory::SystemPrompt
            };
            let carrier = section.carrier.unwrap_or("system_message");
            let measurement = self.measure_cached(
                category.as_str(),
                &json!({"text": section.text, "carrier": carrier}),
                tracker,
                || self.measurement(self.counter.count_text(section.text) as u64),
            );
            groups.push(
                category,
                PartMeasurement::from_measurement(&measurement, &self.identity),
                carrier.to_owned(),
            );
        }
    }

    /// Routes the context messages into their categories.
    ///
    /// A [`ATTACHMENT_FRAGMENTS_METADATA_KEY`] message keeps its rendered text
    /// whole and splits its measured tokens across fragments; a
    /// [`ATTACHMENT_CATEGORY_METADATA_KEY`] message moves whole into an
    /// accepting category; everything else stays in `messages`.
    fn measure_context_messages(
        &self,
        messages: &[BaseMessage],
        tracker: &mut CacheTracker,
        groups: &mut CategoryGroups,
    ) {
        for message in messages {
            if let Some(fragments) = message
                .metadata
                .get(ATTACHMENT_FRAGMENTS_METADATA_KEY)
                .and_then(Value::as_array)
                .filter(|fragments| !fragments.is_empty())
            {
                let measurement = self.measure_attachment(message, "attachments", tracker);
                for (category, part, carrier) in
                    split_fragment_measurement(&measurement, fragments, &self.identity)
                {
                    groups.push(category, part, carrier);
                }
                continue;
            }
            let Some(value) = message.metadata.get(ATTACHMENT_CATEGORY_METADATA_KEY) else {
                groups.ordinary.push(message.clone());
                continue;
            };
            let category = attachment_category(value);
            if !category.accepts_attachment() {
                groups.ordinary.push(message.clone());
                continue;
            }
            let carrier = message
                .metadata
                .get(ATTACHMENT_CARRIER_METADATA_KEY)
                .and_then(Value::as_str)
                .filter(|carrier| !carrier.is_empty())
                .unwrap_or("context_message")
                .to_owned();
            let measurement = self.measure_attachment(message, category.as_str(), tracker);
            groups.push(
                category,
                PartMeasurement::from_measurement(&measurement, &self.identity),
                carrier,
            );
        }
    }

    /// Measures one attachment message as a whole carrier.
    fn measure_attachment(
        &self,
        message: &BaseMessage,
        part_name: &str,
        tracker: &mut CacheTracker,
    ) -> TokenMeasurement {
        let value = message_value(message);
        self.measure_cached(part_name, &value, tracker, || {
            self.measurement(self.counter.count_messages(std::slice::from_ref(message)) as u64)
        })
    }

    /// Measures the serialized tool definitions.
    ///
    /// The sizing helper reports its own provenance, so a count that had to be
    /// estimated stays an estimate here too; with the analyzer's counter present
    /// the counter's identity decides.
    fn measure_tools(&self, tools: &[Value], tracker: &mut CacheTracker) -> TokenMeasurement {
        self.measure_cached(
            ContextCategory::Tools.as_str(),
            &Value::Array(tools.to_vec()),
            tracker,
            || {
                let counted =
                    count_tools_tokens_with_provenance(tools, Some(self.counter.as_ref()));
                self.measurement_with(counted)
            },
        )
    }

    /// Measures the messages that stay in the `messages` category.
    fn measure_messages(
        &self,
        messages: &[BaseMessage],
        tracker: &mut CacheTracker,
    ) -> TokenMeasurement {
        self.measure_cached(
            ContextCategory::Messages.as_str(),
            &messages_value(messages),
            tracker,
            || self.measurement(self.counter.count_messages(messages) as u64),
        )
    }

    /// Measures one logical input, reusing only token results.
    fn measure_cached<F>(
        &self,
        part_name: &str,
        value: &Value,
        tracker: &mut CacheTracker,
        measure: F,
    ) -> TokenMeasurement
    where
        F: FnOnce() -> TokenMeasurement,
    {
        let key = self.measurement_key(part_name, value);
        if let Some(cached) = self.lock_cache().get(&key) {
            let cached = cached.clone();
            tracker.record(part_name, true);
            return cached;
        }
        let measurement = measure();
        self.lock_cache().insert(key, measurement.clone());
        tracker.record(part_name, false);
        measurement
    }

    /// Builds the measurement-cache key of one part measurement.
    fn measurement_key(&self, part_name: &str, value: &Value) -> String {
        let identity = [
            self.identity.class.as_str(),
            self.identity.tokenizer.as_deref().unwrap_or_default(),
            self.model.as_str(),
            MEASUREMENT_KEY_VERSION,
            part_name,
            &canonical_json(value),
        ]
        .join(":");
        let digest = Sha256::digest(identity.as_bytes());
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            let _ = write!(hex, "{byte:02x}");
        }
        hex
    }

    /// Builds a measurement from a count helper's provenance.
    ///
    /// The analyzer's own counter is the executed counter, so its identity wins;
    /// a helper that had to estimate without a counter still forces `estimated`
    /// and carries its own note.
    fn measurement_with(&self, counted: ContextTokenCount) -> TokenMeasurement {
        let mut measurement = self.measurement(counted.tokens);
        if counted.estimated {
            measurement.estimated = true;
            measurement.fallback_reason = counted.note;
        }
        measurement
    }

    /// Builds a measurement from a token count and the counter that ran.
    ///
    /// Both provenance facts come from what the analyzer can vouch for, never
    /// from an unverified claim: an approximate counter behind a native identity
    /// is published as an estimate naming the reason.
    fn measurement(&self, tokens: u64) -> TokenMeasurement {
        TokenMeasurement {
            tokens: usize::try_from(tokens).unwrap_or(usize::MAX),
            source: self.identity.source,
            estimated: self.counted_estimated(),
            tokenizer: self.identity.tokenizer.clone(),
            model: self.identity.fallback_tokenizer_model.clone(),
            fallback_reason: self.published_reason(),
        }
    }
}

/// Reconciles a provider total with the locally measured categories.
///
/// Returns `provider_total - local_total` as a diagnostic; the total itself is
/// never corrected by that delta.
fn reconcile(
    parts: &mut BTreeMap<String, ContextPartUsage>,
    provider_input_tokens: Option<u64>,
    proportional: bool,
) -> Option<i64> {
    let provider_total = provider_input_tokens?;
    let local_total: u64 = parts
        .values()
        .map(|part| part.tokens)
        .fold(0u64, u64::saturating_add);
    if proportional {
        if local_total > 0 {
            let allocations = proportional_allocations(parts, local_total, provider_total);
            for (name, part) in parts.iter_mut() {
                let allocation = allocations.get(name.as_str()).copied().unwrap_or(0);
                if part.tokens == 0 && allocation == 0 {
                    continue;
                }
                part.tokens = allocation;
                SOURCE_PROVIDER_USAGE_PROPORTIONAL.clone_into(&mut part.source);
                SOURCE_PROVIDER_USAGE_PROPORTIONAL.clone_into(&mut part.measurement_source);
                part.estimated = true;
                part.fallback_reason = Some(PROPORTIONAL_REASON.to_owned());
            }
        } else if let Some(messages) = parts.get_mut(ContextCategory::Messages.as_str()) {
            messages.tokens = provider_total;
            SOURCE_PROVIDER_USAGE_UNATTRIBUTED.clone_into(&mut messages.source);
            SOURCE_PROVIDER_USAGE_UNATTRIBUTED.clone_into(&mut messages.measurement_source);
            messages.estimated = true;
            messages.fallback_reason = Some(UNATTRIBUTED_REASON.to_owned());
        }
    } else {
        let fixed_tokens: u64 = parts
            .iter()
            .filter(|(name, _)| name.as_str() != ContextCategory::Messages.as_str())
            .map(|(_, part)| part.tokens)
            .fold(0u64, u64::saturating_add);
        let Some(messages) = parts.get_mut(ContextCategory::Messages.as_str()) else {
            return Some(
                i64::try_from(i128::from(provider_total) - i128::from(local_total))
                    .unwrap_or(i64::MAX),
            );
        };
        let residual = i128::from(provider_total) - i128::from(fixed_tokens);
        messages.tokens = u64::try_from(residual).unwrap_or(0);
        SOURCE_PROVIDER_USAGE_RESIDUAL.clone_into(&mut messages.source);
        SOURCE_PROVIDER_USAGE_RESIDUAL.clone_into(&mut messages.measurement_source);
        messages.estimated = true;
        messages.fallback_reason = Some(
            if residual >= 0 {
                RESIDUAL_REASON
            } else {
                RESIDUAL_NEGATIVE_REASON
            }
            .to_owned(),
        );
    }
    Some(i64::try_from(i128::from(provider_total) - i128::from(local_total)).unwrap_or(i64::MAX))
}

/// Scales every category by `provider_total / local_total` with
/// largest-remainder rounding, so the allocations sum exactly to the provider
/// total.
fn proportional_allocations(
    parts: &BTreeMap<String, ContextPartUsage>,
    local_total: u64,
    provider_total: u64,
) -> HashMap<String, u64> {
    let names = ordered_part_names(parts);
    let mut allocations: HashMap<String, u64> = HashMap::new();
    let mut remainders: Vec<(String, u128, usize)> = Vec::with_capacity(names.len());
    let mut allocated = 0u64;
    for (position, name) in names.iter().enumerate() {
        let tokens = parts.get(name.as_str()).map_or(0, |part| part.tokens);
        let numerator = u128::from(provider_total) * u128::from(tokens);
        let allocation = numerator / u128::from(local_total);
        let allocation = u64::try_from(allocation).unwrap_or(u64::MAX);
        allocated = allocated.saturating_add(allocation);
        remainders.push((name.clone(), numerator % u128::from(local_total), position));
        allocations.insert(name.clone(), allocation);
    }
    let remaining = provider_total.saturating_sub(allocated);
    remainders.sort_by(|left, right| right.1.cmp(&left.1).then(left.2.cmp(&right.2)));
    for (name, _, _) in remainders
        .into_iter()
        .take(usize::try_from(remaining).unwrap_or(usize::MAX))
    {
        if let Some(allocation) = allocations.get_mut(&name) {
            *allocation = allocation.saturating_add(1);
        }
    }
    allocations
}

/// The part names in Python's protocol order, with any unknown trailing.
fn ordered_part_names(parts: &BTreeMap<String, ContextPartUsage>) -> Vec<String> {
    let mut names: Vec<String> = ContextCategory::ALL
        .iter()
        .map(|category| category.as_str().to_owned())
        .filter(|name| parts.contains_key(name))
        .collect();
    for name in parts.keys() {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    names
}

/// Fills the percentage fields of every part.
fn apply_percentages(
    parts: &mut BTreeMap<String, ContextPartUsage>,
    limit: Option<u64>,
    input_tokens: u64,
) {
    for part in parts.values_mut() {
        part.percentage_of_window = ratio(part.tokens, limit);
        part.percentage_of_input = ratio(part.tokens, Some(input_tokens));
    }
}

/// `numerator / denominator`, or `None` when the denominator is absent or zero.
fn ratio(numerator: u64, denominator: Option<u64>) -> Option<f64> {
    match denominator {
        Some(denominator) if denominator > 0 => Some(numerator as f64 / denominator as f64),
        _ => None,
    }
}

/// Merges the sources of the parts, ignoring unreported ones.
fn merge_source(parts: &BTreeMap<String, ContextPartUsage>) -> String {
    let sources: BTreeSet<&str> = parts
        .values()
        .map(|part| part.source.as_str())
        .filter(|source| *source != NOT_REPORTED)
        .collect();
    merge_names(sources).unwrap_or_else(|| NOT_REPORTED.to_owned())
}

/// Merges the tokenizers of the parts.
fn merge_tokenizer(parts: &BTreeMap<String, ContextPartUsage>) -> Option<String> {
    let tokenizers: BTreeSet<&str> = parts
        .values()
        .filter_map(|part| part.tokenizer.as_deref())
        .collect();
    merge_names(tokenizers)
}

/// Whether any reported part is an estimate.
fn any_reported_estimated(parts: &BTreeMap<String, ContextPartUsage>) -> bool {
    parts
        .values()
        .any(|part| part.estimated && part.source != NOT_REPORTED)
}

/// Whether a string is non-empty after trimming.
fn non_empty(text: &str) -> Option<&str> {
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}
