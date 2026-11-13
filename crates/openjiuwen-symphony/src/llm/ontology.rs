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
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::graph::CAN_FEED;
use crate::model::{CapabilityDescriptor, CapabilityIo};
use crate::orchestration::{MatchStrategy, RelationCandidate};

use super::{
    clamp_unit, complete_json, dedupe_strings, truncate_text, CompletionModel, FallbackMode,
};

/// The `method` recorded for a single LLM pass.
pub const METHOD_LLM: &str = "llm_ontology_match";
/// The `method` recorded for a consensus of two passes (forward and reversed).
pub const METHOD_CONSENSUS: &str = "llm_consensus_match";
/// The `method` recorded when the deterministic candidate confidence was used.
pub const METHOD_DETERMINISTIC: &str = "deterministic_fallback";
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

/// Validates `can_feed` relation candidates with a model, or with their own rule.
#[derive(Debug, Clone)]
pub struct OntologyMatcher {
    /// The catalog candidates are described from.
    capabilities: Vec<CapabilityDescriptor>,
    /// The thresholds, model, and fallback in use.
    config: OntologyConfig,
}

impl OntologyMatcher {
    /// Builds a matcher over `capabilities` with the default configuration.
    #[must_use]
    pub fn new(capabilities: impl IntoIterator<Item = CapabilityDescriptor>) -> Self {
        Self {
            capabilities: capabilities.into_iter().collect(),
            config: OntologyConfig::default(),
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

    /// Validates `candidates`, answering one [`OntologyMatch`] per usable candidate.
    ///
    /// Matches are sorted by `(source_id, target_id)`; low-confidence verdicts are
    /// returned with `accepted: false` so a caller can log them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when [`FallbackMode::Error`] is configured and the model
    /// call fails or its reply is unusable.
    pub async fn match_candidates(
        &self,
        candidates: &[RelationCandidate],
    ) -> Result<Vec<OntologyMatch>> {
        let batch = self.select_batch(candidates);
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let Some(model) = self.config.model.as_deref() else {
            return Ok(self.deterministic_matches(&batch));
        };
        let forward = match self.ask(model, &batch, false).await {
            Ok(matches) => matches,
            Err(error) => return self.on_failure(&batch, &error),
        };
        if !self.config.require_consensus {
            return Ok(forward);
        }
        let reverse = match self.ask(model, &batch, true).await {
            Ok(matches) => matches,
            Err(error) => return self.on_failure(&batch, &error),
        };
        Ok(self.consensus(&forward, &reverse))
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
