//! LLM-validated `can_feed` ontology matching.
//!
//! Python's `orchestration/graph/matcher/` asks a model whether each candidate
//! capability's outputs can feed another capability's inputs, and validates the answer
//! before it becomes a graph edge. This module ports that gate onto the crate's
//! [`CompletionModel`] seam while keeping the crate's deterministic candidate rules as
//! the fallback.
//!
//! # The LLM path
//!
//! Candidates are numbered `c1..cN` and sent as one compact JSON batch:
//!
//! ```json
//! {"candidates":[{"id":"c1","source":{"name":"…","description":"…"},
//!  "target":{"name":"…","description":"…"},
//!  "directions":{"forward":{"outputs":[…],"inputs":[…],"ports":[…]},
//!                "reverse":{"outputs":[…],"inputs":[…],"ports":[…]}}}]}
//! ```
//!
//! Blank fields and empty arrays are pruned so the prompt stays short, and descriptions
//! are truncated to 240 characters. The reply must be
//! `{"matches":[{"id","direction","confidence","reason"}]}`; everything after that is
//! validated here, never trusted:
//!
//! - an unknown id, a duplicate id, a direction other than `forward`/`reverse`, or a
//!   non-numeric/non-finite confidence is rejected (dropped, with a `tracing` record);
//! - a confidence outside `[0, 1]` is clamped (and therefore typically rejected by the
//!   threshold);
//! - `"accepted": false` forces the confidence to `0.0`;
//! - `direction: "reverse"` swaps the match orientation, so the reported
//!   [`OntologyMatch`] always reads "`source_id` can feed `target_id`";
//! - `accepted` is `confidence >= threshold` ([`OntologyConfig::threshold`], `0.7` by
//!   default, Python's `can_feed` gate).
//!
//! The relation vocabulary is always [`CAN_FEED`](crate::graph::CAN_FEED) — it is never read from the model.
//!
//! # Consensus
//!
//! With [`OntologyConfig::require_consensus`] (the default) the batch is asked twice,
//! second time in reversed order with re-numbered ids. Only candidates accepted in both
//! passes survive, with the lower of the two confidences and method
//! [`METHOD_CONSENSUS`](crate::llm::ontology::METHOD_CONSENSUS); a pass that disagrees about the *direction* is ambiguous and
//! dropped. Without consensus, one pass answers with method [`METHOD_LLM`](crate::llm::ontology::METHOD_LLM).
//!
//! # The deterministic fallback
//!
//! With no model configured, or when the model fails and
//! [`FallbackMode::Deterministic`](crate::llm::FallbackMode) is set, the confidence comes from the candidate's own
//! rule: [`MatchStrategy::ExactIo`](crate::orchestration::MatchStrategy) `1.0`, [`MatchStrategy::SemanticOverlap`](crate::orchestration::MatchStrategy) the
//! candidate's confidence, and [`MatchStrategy::TextualCoercion`](crate::orchestration::MatchStrategy) `0.75`. Every match
//! then reports method [`METHOD_DETERMINISTIC`](crate::llm::ontology::METHOD_DETERMINISTIC) and `accepted = confidence >= threshold`.
//! [`FallbackMode::Error`](crate::llm::FallbackMode) surfaces the model failure instead.
//!
//! # Bounds
//!
//! At most [`OntologyConfig::max_candidates`] candidates are matched per call (`0` means
//! unlimited); the batch is sorted by `(source, target)` before truncation, so which
//! candidates survive is deterministic. Candidates naming a capability that is not in
//! the matcher's catalog carry no name or IO description to show the model and are
//! skipped.
//!
//! # The persistent cache
//!
//! [`OntologyMatcher::with_cache`] installs a [`MatchCache`] (shared through a
//! [`tokio::sync::Mutex`]), and [`OntologyMatcher::match_candidates_with`] then serves a
//! repeated answer from disk instead of asking the model again:
//!
//! - every candidate is keyed by [`cache_key`] over the candidate's own JSON, the
//!   [graph identity digests](graph_identity_sha256) of its source and target, and
//!   [`OntologyMatcher::matcher_signature`] — a canonical document covering the matcher,
//!   prompt, and reply-schema versions, the [`ModelIdentity`], the threshold, the
//!   consensus switch, and the relation/direction/method strategy constants;
//! - cached candidates are replayed and only the rest are sent to the model (in one
//!   batch, exactly as before), so a fully cached call performs **no** model calls;
//! - a fresh model answer is written back per candidate immediately; a deterministic
//!   fallback answer is never cached, so a later run with a working model still asks;
//! - [`MatchOptions::force`] ignores every hit and refreshes the entries;
//! - a cache read, decode, or write problem is logged with `tracing::warn!` and the call
//!   degrades to a normal model call — the model result is always returned, and a
//!   changed signature or model identity simply means a miss, never a stale hit.
//!
//! Without a cache the matcher behaves exactly as it did before: one batch per call (two
//! with consensus), same numbering, same validation, same fallbacks.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::Mutex;

use crate::error::{Error, Result};
use crate::graph::CAN_FEED;
use crate::llm::cache::{
    cache_key, candidate_identity, covers, decode_cached_matches, CachedMatch, MatchCache,
    CACHE_SCHEMA_VERSION,
};
use crate::llm::identity::{
    canonical_json, graph_identity_sha256, stable_metadata_sha256, ModelIdentity, IDENTITY_VERSION,
};
use crate::model::{CapabilityDescriptor, CapabilityIo};
use crate::orchestration::{MatchStrategy, RelationCandidate};

use super::{
    clamp_unit, complete_json, dedupe_strings, truncate_text, CompletionModel, FallbackMode,
    ModelConfig,
};

/// The `method` recorded for a single LLM pass.
pub const METHOD_LLM: &str = "llm_ontology_match";
/// The `method` recorded for a consensus of two passes (forward and reversed).
pub const METHOD_CONSENSUS: &str = "llm_consensus_match";
/// The `method` recorded when the deterministic candidate confidence was used.
pub const METHOD_DETERMINISTIC: &str = "deterministic_fallback";
/// The matcher version folded into [`OntologyMatcher::matcher_signature`].
pub const MATCHER_VERSION: &str = "Symphony-ontology-matcher-v2";
/// The prompt version folded into [`OntologyMatcher::matcher_signature`].
pub const PROMPT_VERSION: &str = "Orchestration-graph-match-v2";
/// The reply-schema version folded into [`OntologyMatcher::matcher_signature`].
pub const MATCH_SCHEMA_VERSION: &str = "Symphony-ontology-match-schema-v1";
/// The backend recorded when no model is configured.
const NO_MODEL_BACKEND: &str = "none";
/// The default acceptance threshold (Python's `can_feed` gate).
const DEFAULT_THRESHOLD: f64 = 0.7;
/// The default cap on candidates matched per call.
const DEFAULT_MAX_CANDIDATES: usize = 32;
/// How many characters of a description are sent to the model.
const DESCRIPTION_LIMIT: usize = 240;
/// The system instruction sent with every match request.
const SYSTEM_PROMPT: &str = "Validate whether each candidate Skill output can feed the target input. \
Return JSON only: {\"matches\":[{\"id\":\"c1\",\"direction\":\"forward|reverse\",\"confidence\":0.0,\"reason\":\"optional short reason\"}]}. \
Use \"forward\" when the source output feeds the target input and \"reverse\" when the target output feeds the source. \
Confidence must be between 0 and 1. Omit candidates that cannot feed each other.";
/// The forward direction token.
const DIRECTION_FORWARD: &str = "forward";
/// The reverse direction token.
const DIRECTION_REVERSE: &str = "reverse";

/// One validated `can_feed` verdict between two capabilities.
///
/// The orientation is always resolved: `source_id` can feed `target_id`, whatever
/// `direction` the model reported. `reasons` holds the model's short reasons (never
/// more than one per pass), and `accepted` is the threshold decision a graph builder
/// uses to decide whether to publish the edge.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct OntologyMatch {
    /// The qualified id of the producing capability.
    pub source_id: String,
    /// The qualified id of the consuming capability.
    pub target_id: String,
    /// The relation kind; always [`CAN_FEED`].
    pub relation: String,
    /// The validated confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// How the verdict was produced ([`METHOD_LLM`], [`METHOD_CONSENSUS`], or
    /// [`METHOD_DETERMINISTIC`]).
    pub method: String,
    /// Short reasons reported by the model, deduplicated.
    pub reasons: Vec<String>,
    /// Whether the verdict passes the configured threshold.
    pub accepted: bool,
}

impl OntologyMatch {
    /// Builds a rejected `can_feed` verdict with `source_id` feeding `target_id`.
    #[must_use]
    pub fn new(source_id: impl Into<String>, target_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            relation: CAN_FEED.to_string(),
            confidence: 0.0,
            method: METHOD_DETERMINISTIC.to_string(),
            reasons: Vec::new(),
            accepted: false,
        }
    }
}

/// Tuning knobs for an [`OntologyMatcher`].
#[derive(Clone)]
pub struct OntologyConfig {
    /// The inclusive confidence a verdict must reach to be accepted.
    pub threshold: f64,
    /// Whether the batch is asked twice (forward and reversed) and intersected.
    pub require_consensus: bool,
    /// The maximum number of candidates matched per call; `0` means unlimited.
    pub max_candidates: usize,
    /// The completion model; `None` keeps the deterministic candidate confidence.
    pub model: Option<Arc<dyn CompletionModel>>,
    /// What to do when the model fails or answers unusably.
    pub fallback: FallbackMode,
}

impl Default for OntologyConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            require_consensus: true,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            model: None,
            fallback: FallbackMode::Deterministic,
        }
    }
}

impl OntologyConfig {
    /// Builds the default configuration: threshold `0.7`, consensus on, no model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the acceptance threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets whether two passes must agree.
    #[must_use]
    pub fn with_consensus(mut self, require_consensus: bool) -> Self {
        self.require_consensus = require_consensus;
        self
    }

    /// Sets the per-call candidate cap (`0` = unlimited).
    #[must_use]
    pub fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.max_candidates = max_candidates;
        self
    }

    /// Sets the completion model.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the fallback mode.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }
}

impl fmt::Debug for OntologyConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OntologyConfig")
            .field("threshold", &self.threshold)
            .field("require_consensus", &self.require_consensus)
            .field("max_candidates", &self.max_candidates)
            .field("model", &self.model.as_deref().map(CompletionModel::name))
            .field("fallback", &self.fallback)
            .finish()
    }
}

/// Per-call behaviour for [`OntologyMatcher::match_candidates_with`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MatchOptions {
    /// Ignore cached answers and refresh them from the model.
    ///
    /// `false` (the default) serves a cache hit without calling the model; `true` makes
    /// every candidate a miss while still writing the fresh answers back.
    pub force: bool,
}

impl MatchOptions {
    /// Builds the default options (the cache is honoured).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether the cache is bypassed.
    #[must_use]
    pub fn with_force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }
}

/// Validates `can_feed` relation candidates with a model, or with their own rule.
#[derive(Debug, Clone)]
pub struct OntologyMatcher {
    /// The catalog candidates are described from.
    capabilities: Vec<CapabilityDescriptor>,
    /// The thresholds, model, and fallback in use.
    config: OntologyConfig,
    /// The optional persistent answer cache shared between matchers.
    cache: Option<Arc<Mutex<MatchCache>>>,
}

impl OntologyMatcher {
    /// Builds a matcher over `capabilities` with the default configuration.
    #[must_use]
    pub fn new(capabilities: impl IntoIterator<Item = CapabilityDescriptor>) -> Self {
        Self {
            capabilities: capabilities.into_iter().collect(),
            config: OntologyConfig::default(),
            cache: None,
        }
    }

    /// Replaces the configuration.
    #[must_use]
    pub fn with_config(mut self, config: OntologyConfig) -> Self {
        self.config = config;
        self
    }

    /// Configures the matcher to use `model`, keeping the other settings.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.config.model = Some(model);
        self
    }

    /// Installs the persistent match cache.
    ///
    /// The cache is only consulted when a model is configured (without one the
    /// deterministic path is already free), and it is shared through the supplied mutex
    /// so several matchers can memoize into one file.
    #[must_use]
    pub fn with_cache(mut self, cache: Arc<Mutex<MatchCache>>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// The installed match cache, if any.
    #[must_use]
    pub fn cache(&self) -> Option<&Arc<Mutex<MatchCache>>> {
        self.cache.as_ref()
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &OntologyConfig {
        &self.config
    }

    /// The catalog the matcher describes candidates from.
    #[must_use]
    pub fn capabilities(&self) -> &[CapabilityDescriptor] {
        &self.capabilities
    }

    /// The matcher identity folded into every cache key.
    ///
    /// The signature is canonical JSON (sorted keys, no whitespace) over:
    /// [`MATCHER_VERSION`], [`PROMPT_VERSION`], [`MATCH_SCHEMA_VERSION`],
    /// [`CACHE_SCHEMA_VERSION`], [`IDENTITY_VERSION`], a digest of the system prompt, the
    /// [`ModelIdentity`], the acceptance threshold, the consensus switch, the candidate
    /// cap, and the relation/direction/method/description-limit strategy constants.
    ///
    /// Any change to those inputs produces a different signature, which invalidates
    /// every entry recorded under the previous one.
    #[must_use]
    pub fn matcher_signature(&self) -> String {
        canonical_json(&self.signature_document())
    }

    /// The [`ModelConfig`] view of this matcher, used for its model identity.
    fn model_config(&self) -> ModelConfig {
        ModelConfig {
            model: self.config.model.clone(),
            temperature: None,
            max_tokens: None,
            fallback: self.config.fallback,
        }
    }

    /// The backend name folded into the model identity.
    ///
    /// The crate has no separate provider field on [`OntologyConfig`], so the completion
    /// model's own name identifies the backend.
    fn backend(&self) -> &str {
        self.config
            .model
            .as_deref()
            .map_or(NO_MODEL_BACKEND, CompletionModel::name)
    }

    /// Builds the canonical signature document.
    fn signature_document(&self) -> Value {
        let identity = ModelIdentity::of(&self.model_config(), self.backend());
        let mut document = Map::new();
        document.insert("matcher_version".to_string(), Value::from(MATCHER_VERSION));
        document.insert("prompt_version".to_string(), Value::from(PROMPT_VERSION));
        document.insert(
            "match_schema_version".to_string(),
            Value::from(MATCH_SCHEMA_VERSION),
        );
        document.insert(
            "cache_schema_version".to_string(),
            Value::from(CACHE_SCHEMA_VERSION),
        );
        document.insert(
            "identity_version".to_string(),
            Value::from(IDENTITY_VERSION),
        );
        document.insert(
            "system_prompt_sha256".to_string(),
            Value::from(stable_metadata_sha256(&Value::from(SYSTEM_PROMPT))),
        );
        document.insert(
            "model".to_string(),
            serde_json::to_value(&identity).unwrap_or(Value::Null),
        );
        document.insert("threshold".to_string(), Value::from(self.config.threshold));
        document.insert(
            "require_consensus".to_string(),
            Value::Bool(self.config.require_consensus),
        );
        document.insert(
            "max_candidates".to_string(),
            Value::from(u64::try_from(self.config.max_candidates).unwrap_or(u64::MAX)),
        );
        document.insert("relation".to_string(), Value::from(CAN_FEED));
        document.insert(
            "directions".to_string(),
            Value::from(vec![DIRECTION_FORWARD, DIRECTION_REVERSE]),
        );
        document.insert(
            "methods".to_string(),
            Value::from(vec![METHOD_LLM, METHOD_CONSENSUS, METHOD_DETERMINISTIC]),
        );
        document.insert(
            "description_limit".to_string(),
            Value::from(u64::try_from(DESCRIPTION_LIMIT).unwrap_or(u64::MAX)),
        );
        Value::Object(document)
    }

    /// Validates `candidates`, answering one [`OntologyMatch`] per usable candidate.
    ///
    /// Matches are sorted by `(source_id, target_id)`; low-confidence verdicts are
    /// returned with `accepted: false` so a caller can log them. Cache hits are honoured;
    /// use [`OntologyMatcher::match_candidates_with`] with
    /// [`MatchOptions::with_force`] to bypass them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when [`FallbackMode::Error`] is configured and the model
    /// call fails or its reply is unusable.
    pub async fn match_candidates(
        &self,
        candidates: &[RelationCandidate],
    ) -> Result<Vec<OntologyMatch>> {
        self.match_candidates_with(candidates, &MatchOptions::default())
            .await
    }

    /// Validates `candidates` with explicit [`MatchOptions`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when [`FallbackMode::Error`] is configured and the model
    /// call fails or its reply is unusable.
    pub async fn match_candidates_with(
        &self,
        candidates: &[RelationCandidate],
        options: &MatchOptions,
    ) -> Result<Vec<OntologyMatch>> {
        let batch = self.select_batch(candidates);
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let Some(model) = self.config.model.as_deref() else {
            return Ok(self.deterministic_matches(&batch));
        };
        let Some(cache) = self.cache.as_ref() else {
            return Ok(self.resolve(model, &batch).await?.matches);
        };

        let signature = self.matcher_signature();
        let keys: Vec<String> = batch
            .iter()
            .map(|entry| cache_key_for(entry, &signature))
            .collect();
        let mut resolved: Vec<OntologyMatch> = Vec::new();
        let mut misses: Vec<(usize, BatchEntry<'_>)> = Vec::new();
        if options.force {
            misses.extend(batch.iter().cloned().enumerate());
        } else {
            // The async mutex is held only for the in-memory lookups, never across a
            // model call.
            let guard = cache.lock().await;
            for (index, entry) in batch.iter().enumerate() {
                if let Some(record) = guard.lookup(&keys[index]).await {
                    if let Some(matches) = decode_cached_matches(&record) {
                        resolved.extend(matches);
                        continue;
                    }
                    tracing::warn!(
                        candidate = %candidate_identity(entry.candidate),
                        "ontology matcher: cached answer is unusable; asking the model"
                    );
                }
                misses.push((index, entry.clone()));
            }
        }
        if misses.is_empty() {
            sort_matches(&mut resolved);
            return Ok(resolved);
        }

        let miss_batch: Vec<BatchEntry<'_>> =
            misses.iter().map(|(_, entry)| entry.clone()).collect();
        let fresh = self.resolve(model, &miss_batch).await?;
        if fresh.from_model {
            let mut guard = cache.lock().await;
            for (index, entry) in &misses {
                let selected: Vec<OntologyMatch> = fresh
                    .matches
                    .iter()
                    .filter(|item| covers(item, entry.candidate))
                    .cloned()
                    .collect();
                let recorded = match serde_json::to_value(&selected) {
                    Ok(value) => value,
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "ontology matcher: a model answer could not be serialized; not cached"
                        );
                        continue;
                    }
                };
                let record = CachedMatch::new(candidate_identity(entry.candidate), recorded);
                if let Err(error) = guard.record(keys[*index].clone(), record).await {
                    tracing::warn!(
                        %error,
                        "ontology matcher: match-cache write failed; the model answer is still used"
                    );
                }
            }
        }
        resolved.extend(fresh.matches);
        sort_matches(&mut resolved);
        Ok(resolved)
    }

    /// Runs the model passes for `batch`, applying the configured fallback.
    async fn resolve(
        &self,
        model: &dyn CompletionModel,
        batch: &[BatchEntry<'_>],
    ) -> Result<Resolved> {
        let forward = match self.ask(model, batch, false).await {
            Ok(matches) => matches,
            Err(error) => {
                return Ok(Resolved {
                    matches: self.on_failure(batch, &error)?,
                    from_model: false,
                });
            }
        };
        if !self.config.require_consensus {
            return Ok(Resolved {
                matches: forward,
                from_model: true,
            });
        }
        let reverse = match self.ask(model, batch, true).await {
            Ok(matches) => matches,
            Err(error) => {
                return Ok(Resolved {
                    matches: self.on_failure(batch, &error)?,
                    from_model: false,
                });
            }
        };
        Ok(Resolved {
            matches: self.consensus(&forward, &reverse),
            from_model: true,
        })
    }

    /// Filters `candidates` to the describable ones, sorts them, and numbers them.
    fn select_batch<'a>(&'a self, candidates: &'a [RelationCandidate]) -> Vec<BatchEntry<'a>> {
        let mut entries: Vec<BatchEntry<'a>> = candidates
            .iter()
            .filter_map(|candidate| {
                let source = self
                    .capabilities
                    .iter()
                    .find(|capability| capability.qualified_id() == candidate.source)?;
                let target = self
                    .capabilities
                    .iter()
                    .find(|capability| capability.qualified_id() == candidate.target)?;
                Some(BatchEntry {
                    id: String::new(),
                    source,
                    target,
                    candidate,
                })
            })
            .collect();
        let describable = entries.len();
        entries.sort_by(|left, right| {
            left.candidate
                .source
                .cmp(&right.candidate.source)
                .then(left.candidate.target.cmp(&right.candidate.target))
        });
        if self.config.max_candidates > 0 {
            entries.truncate(self.config.max_candidates);
        }
        if entries.len() < candidates.len() {
            tracing::warn!(
                candidates = candidates.len(),
                matched = entries.len(),
                describable,
                "ontology matcher: some candidates were skipped (unknown capability or candidate cap)"
            );
        }
        for (index, entry) in entries.iter_mut().enumerate() {
            entry.id = format!("c{}", index + 1);
        }
        entries
    }

    /// Asks the model once, optionally in reversed candidate order.
    async fn ask(
        &self,
        model: &dyn CompletionModel,
        batch: &[BatchEntry<'_>],
        reversed: bool,
    ) -> Result<Vec<OntologyMatch>> {
        let ordered: Vec<&BatchEntry<'_>> = if reversed {
            batch.iter().rev().collect()
        } else {
            batch.iter().collect()
        };
        let labelled: Vec<(String, &BatchEntry<'_>)> = ordered
            .into_iter()
            .enumerate()
            .map(|(index, entry)| (format!("c{}", index + 1), entry))
            .collect();
        let request = request_json(&labelled);
        let user = serde_json::to_string(&request).map_err(|error| {
            Error::llm(format!(
                "ontology matcher: request could not be serialized: {error}"
            ))
        })?;
        let reply = complete_json(Some(model), SYSTEM_PROMPT, &user).await?;
        let Some(value) = reply else {
            return Err(Error::llm(
                "ontology matcher: the model reply contained no JSON object",
            ));
        };
        self.parse_matches(&value, &labelled)
    }

    /// Validates one reply, dropping every malformed item.
    fn parse_matches(
        &self,
        value: &Value,
        labelled: &[(String, &BatchEntry<'_>)],
    ) -> Result<Vec<OntologyMatch>> {
        let Some(items) = value.get("matches").and_then(Value::as_array) else {
            return Err(Error::llm(
                "ontology matcher: the reply JSON has no `matches` array",
            ));
        };
        let mut seen: Vec<String> = Vec::new();
        let mut matches = Vec::new();
        for item in items {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                tracing::warn!("ontology matcher: dropping a match without a string `id`");
                continue;
            };
            let Some((candidate_id, entry)) =
                labelled.iter().find(|(candidate_id, _)| candidate_id == id)
            else {
                tracing::warn!(id, "ontology matcher: dropping a match with an unknown id");
                continue;
            };
            let Some(direction) = item.get("direction").and_then(Value::as_str) else {
                tracing::warn!(id, "ontology matcher: dropping a match without a direction");
                continue;
            };
            let reverse = match direction {
                DIRECTION_FORWARD => false,
                DIRECTION_REVERSE => true,
                other => {
                    tracing::warn!(
                        id,
                        direction = other,
                        "ontology matcher: dropping an invalid direction"
                    );
                    continue;
                }
            };
            let Some(confidence) = item.get("confidence").and_then(Value::as_f64) else {
                tracing::warn!(id, "ontology matcher: dropping a non-numeric confidence");
                continue;
            };
            if !confidence.is_finite() {
                tracing::warn!(id, "ontology matcher: dropping a non-finite confidence");
                continue;
            }
            if seen.iter().any(|existing| existing == candidate_id) {
                tracing::warn!(id, "ontology matcher: dropping a duplicate match id");
                continue;
            }
            seen.push(candidate_id.clone());
            let rejected = item.get("accepted").and_then(Value::as_bool) == Some(false);
            let confidence = if rejected {
                0.0
            } else {
                clamp_unit(confidence)
            };
            let (source_id, target_id) = if reverse {
                (
                    entry.candidate.target.clone(),
                    entry.candidate.source.clone(),
                )
            } else {
                (
                    entry.candidate.source.clone(),
                    entry.candidate.target.clone(),
                )
            };
            let reasons = item
                .get("reason")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .map(|reason| vec![reason.to_string()])
                .unwrap_or_default();
            matches.push(OntologyMatch {
                source_id,
                target_id,
                relation: CAN_FEED.to_string(),
                confidence,
                method: METHOD_LLM.to_string(),
                reasons,
                accepted: confidence >= self.config.threshold,
            });
        }
        sort_matches(&mut matches);
        Ok(matches)
    }

    /// Intersects two accepted passes; a disagreement about direction drops the pair.
    fn consensus(
        &self,
        forward: &[OntologyMatch],
        reverse: &[OntologyMatch],
    ) -> Vec<OntologyMatch> {
        let mut index: BTreeMap<(String, String), &OntologyMatch> = BTreeMap::new();
        for item in reverse {
            index.insert(unordered_pair(item), item);
        }
        let mut matches = Vec::new();
        for item in forward {
            let Some(partner) = index.get(&unordered_pair(item)) else {
                continue;
            };
            if !item.accepted || !partner.accepted {
                continue;
            }
            if item.source_id != partner.source_id {
                tracing::warn!(
                    source = item.source_id.as_str(),
                    target = item.target_id.as_str(),
                    "ontology matcher: dropping a pair the two passes oriented differently"
                );
                continue;
            }
            let mut reasons = item.reasons.clone();
            for reason in &partner.reasons {
                if !reasons.contains(reason) {
                    reasons.push(reason.clone());
                }
            }
            let confidence = item.confidence.min(partner.confidence);
            matches.push(OntologyMatch {
                source_id: item.source_id.clone(),
                target_id: item.target_id.clone(),
                relation: CAN_FEED.to_string(),
                confidence,
                method: METHOD_CONSENSUS.to_string(),
                reasons,
                accepted: confidence >= self.config.threshold,
            });
        }
        sort_matches(&mut matches);
        matches
    }

    /// Derives the deterministic verdicts from the candidate rules.
    fn deterministic_matches(&self, batch: &[BatchEntry<'_>]) -> Vec<OntologyMatch> {
        let mut matches: Vec<OntologyMatch> = batch
            .iter()
            .map(|entry| {
                let confidence = match entry.candidate.strategy {
                    MatchStrategy::ExactIo => 1.0,
                    MatchStrategy::SemanticOverlap => clamp_unit(entry.candidate.confidence),
                    MatchStrategy::TextualCoercion => 0.75,
                };
                OntologyMatch {
                    source_id: entry.candidate.source.clone(),
                    target_id: entry.candidate.target.clone(),
                    relation: CAN_FEED.to_string(),
                    confidence,
                    method: METHOD_DETERMINISTIC.to_string(),
                    reasons: Vec::new(),
                    accepted: confidence >= self.config.threshold,
                }
            })
            .collect();
        sort_matches(&mut matches);
        matches
    }

    /// Applies the configured fallback to a failed model pass.
    fn on_failure(&self, batch: &[BatchEntry<'_>], error: &Error) -> Result<Vec<OntologyMatch>> {
        match self.config.fallback {
            FallbackMode::Deterministic => {
                tracing::warn!(
                    %error,
                    "ontology matcher: model pass failed; using the deterministic candidate confidence"
                );
                Ok(self.deterministic_matches(batch))
            }
            FallbackMode::Error => Err(Error::llm(format!(
                "ontology matcher: model pass failed: {}",
                error.message()
            ))),
        }
    }
}

/// One numbered candidate, resolved against the catalog.
#[derive(Clone)]
struct BatchEntry<'a> {
    /// The batch id (`c1`, `c2`, …).
    id: String,
    /// The producing capability.
    source: &'a CapabilityDescriptor,
    /// The consuming capability.
    target: &'a CapabilityDescriptor,
    /// The candidate being validated.
    candidate: &'a RelationCandidate,
}

/// The outcome of one model resolution (with or without consensus).
struct Resolved {
    /// The matches the passes produced.
    matches: Vec<OntologyMatch>,
    /// Whether they came from the model (`false` means the deterministic fallback ran).
    from_model: bool,
}

/// Returns the cache key for one batch entry and the matcher `signature`.
fn cache_key_for(entry: &BatchEntry<'_>, signature: &str) -> String {
    let candidate = serde_json::to_value(entry.candidate).unwrap_or_else(|error| {
        tracing::warn!(
            %error,
            candidate = %candidate_identity(entry.candidate),
            "ontology matcher: candidate could not be serialized for its cache key"
        );
        Value::Null
    });
    cache_key(
        &candidate,
        &graph_identity_sha256(entry.source),
        &graph_identity_sha256(entry.target),
        signature,
    )
}

/// Builds the compact request batch, pruning blank fields and empty arrays.
fn request_json(labelled: &[(String, &BatchEntry<'_>)]) -> Value {
    let candidates: Vec<Value> = labelled
        .iter()
        .map(|(id, entry)| {
            let mut object = Map::new();
            object.insert("id".to_string(), Value::String(id.clone()));
            object.insert("source".to_string(), brief(entry.source));
            object.insert("target".to_string(), brief(entry.target));
            let directions = directions(entry.source, entry.target);
            if directions.as_object().is_some_and(|map| !map.is_empty()) {
                object.insert("directions".to_string(), directions);
            }
            Value::Object(object)
        })
        .collect();
    let mut root = Map::new();
    root.insert("candidates".to_string(), Value::Array(candidates));
    Value::Object(root)
}

/// The short source/target description sent to the model.
fn brief(capability: &CapabilityDescriptor) -> Value {
    let mut object = Map::new();
    insert_text(&mut object, "name", &capability.name);
    let description = truncate_text(&capability.description, DESCRIPTION_LIMIT);
    insert_text(&mut object, "description", description);
    Value::Object(object)
}

/// Both directions of one candidate, each pruned to its non-empty parts.
fn directions(source: &CapabilityDescriptor, target: &CapabilityDescriptor) -> Value {
    let mut object = Map::new();
    let forward = direction(&source.outputs, &target.inputs);
    if forward.as_object().is_some_and(|map| !map.is_empty()) {
        object.insert(DIRECTION_FORWARD.to_string(), forward);
    }
    let reverse = direction(&target.outputs, &source.inputs);
    if reverse.as_object().is_some_and(|map| !map.is_empty()) {
        object.insert(DIRECTION_REVERSE.to_string(), reverse);
    }
    Value::Object(object)
}

/// One direction: the produced outputs, the consumed inputs, and their port names.
fn direction(outputs: &[CapabilityIo], inputs: &[CapabilityIo]) -> Value {
    let mut object = Map::new();
    let produced = io_values(outputs);
    if !produced.is_empty() {
        object.insert("outputs".to_string(), Value::Array(produced));
    }
    let consumed = io_values(inputs);
    if !consumed.is_empty() {
        object.insert("inputs".to_string(), Value::Array(consumed));
    }
    let ports = dedupe_strings(outputs.iter().chain(inputs).map(|io| io.name.clone()));
    if !ports.is_empty() {
        object.insert(
            "ports".to_string(),
            Value::Array(ports.into_iter().map(Value::String).collect()),
        );
    }
    Value::Object(object)
}

/// The pruned `{name, type}` list of one IO declaration list.
fn io_values(ios: &[CapabilityIo]) -> Vec<Value> {
    ios.iter()
        .filter_map(|io| {
            let mut object = Map::new();
            insert_text(&mut object, "name", &io.name);
            insert_text(&mut object, "type", &io.io_type);
            if object.is_empty() {
                None
            } else {
                Some(Value::Object(object))
            }
        })
        .collect()
}

/// Inserts `text` under `key` unless it is blank.
fn insert_text(object: &mut Map<String, Value>, key: &str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    object.insert(key.to_string(), Value::String(text.to_string()));
}

/// The unordered capability pair of a match, used to compare the two passes.
fn unordered_pair(item: &OntologyMatch) -> (String, String) {
    if item.source_id <= item.target_id {
        (item.source_id.clone(), item.target_id.clone())
    } else {
        (item.target_id.clone(), item.source_id.clone())
    }
}

/// Sorts matches by `(source_id, target_id)`.
fn sort_matches(matches: &mut [OntologyMatch]) {
    matches.sort_by(|left, right| {
        left.source_id
            .cmp(&right.source_id)
            .then(left.target_id.cmp(&right.target_id))
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn entry() -> Value {
        let source = CapabilityDescriptor::new("tool", "a", "Extract")
            .with_description("Extracts text from a document")
            .with_outputs(vec![CapabilityIo::new("text", "string")]);
        let target = CapabilityDescriptor::new("tool", "b", "Summarise")
            .with_inputs(vec![CapabilityIo::new("text", "string")]);
        directions(&source, &target)
    }

    #[test]
    fn brief_prunes_blank_fields_and_truncates_descriptions() {
        let capability = CapabilityDescriptor::new("tool", "a", "Name")
            .with_description("x".repeat(DESCRIPTION_LIMIT + 40));
        let value = brief(&capability);
        let object = value.as_object().unwrap();
        assert_eq!(object.len(), 2);
        assert_eq!(
            object["description"].as_str().unwrap().chars().count(),
            DESCRIPTION_LIMIT
        );
    }

    #[test]
    fn directions_carry_outputs_inputs_and_ports() {
        let value = entry();
        let object = value.as_object().unwrap();
        let forward = object[DIRECTION_FORWARD].as_object().unwrap();
        assert_eq!(forward["outputs"][0]["name"], "text");
        assert_eq!(forward["ports"][0], "text");
        // The reverse direction has no outputs and no inputs, so it is pruned.
        assert!(!object.contains_key(DIRECTION_REVERSE));

        let source = CapabilityDescriptor::new("tool", "a", "A")
            .with_inputs(vec![CapabilityIo::new("in", "string")]);
        let target = CapabilityDescriptor::new("tool", "b", "B")
            .with_outputs(vec![CapabilityIo::new("out", "string")]);
        let reverse_only = directions(&source, &target);
        let object = reverse_only.as_object().unwrap();
        assert!(!object.contains_key(DIRECTION_FORWARD));
        let reverse = object[DIRECTION_REVERSE].as_object().unwrap();
        assert_eq!(reverse["outputs"][0]["name"], "out");
        assert_eq!(reverse["ports"][0], "out");
    }

    #[test]
    fn insert_text_skips_blank_values() {
        let mut object = Map::new();
        insert_text(&mut object, "name", "  ");
        assert!(object.is_empty());
        insert_text(&mut object, "name", " value ");
        assert_eq!(object["name"], "value");
    }

    #[test]
    fn unordered_pair_is_orientation_agnostic() {
        let mut item = OntologyMatch::new("tool:b", "tool:a");
        assert_eq!(
            unordered_pair(&item),
            ("tool:a".to_string(), "tool:b".to_string())
        );
        item.source_id = "tool:a".to_string();
        item.target_id = "tool:b".to_string();
        assert_eq!(
            unordered_pair(&item),
            ("tool:a".to_string(), "tool:b".to_string())
        );
    }
}
