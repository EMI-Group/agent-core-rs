//! Deterministic `can_feed` relation candidates between capabilities.
//!
//! Python's `orchestration.graph.matcher` discovers which capabilities can feed which
//! with an LLM ontology matcher. The Rust port keeps the same graph-building job with
//! three deterministic rules (see [`CandidateGenerator`]):
//!
//! - [`MatchStrategy::ExactIo`] — an output and an input share the normalized name
//!   *and* the normalized data type; confidence `1.0`.
//! - [`MatchStrategy::SemanticOverlap`] — the token Jaccard similarity of the IO texts
//!   (`"{name} {description}"`) reaches [`CandidateConfig::min_token_overlap`];
//!   confidence `0.70 + 0.25 * overlap`.
//! - [`MatchStrategy::TextualCoercion`] — the data-type vocabulary says the output type
//!   can feed the input type *and* the names share a token; confidence `0.75`.
//!
//! Rules are evaluated pairwise over every ordered capability pair
//! (`source != target`), the best rule wins per `(source, target)` pair, candidates
//! below [`CandidateConfig::min_confidence`] are dropped, and each source keeps at most
//! [`CandidateConfig::max_candidates_per_source`] candidates (highest confidence
//! first). The final ordering is fully deterministic: strategy rank, then confidence
//! descending, then source, then target — no hash-map iteration order leaks out.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::{CapabilityEdge, CAN_FEED};
use crate::model::{normalize_name_key, CapabilityDescriptor, CapabilityIo, DataTypeVocabulary};
use crate::search::tokenize;

/// The default inclusive minimum confidence a candidate must reach.
const DEFAULT_MIN_CONFIDENCE: f64 = 0.7;
/// The default minimum token Jaccard overlap for [`MatchStrategy::SemanticOverlap`].
const DEFAULT_MIN_TOKEN_OVERLAP: f64 = 0.6;
/// The default cap on candidates kept per source capability.
const DEFAULT_MAX_CANDIDATES_PER_SOURCE: usize = 32;

/// The deterministic rule that produced a [`RelationCandidate`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MatchStrategy {
    /// Normalized IO names and types are equal; the strongest signal.
    ExactIo,
    /// The IO texts share enough tokens (token Jaccard).
    SemanticOverlap,
    /// The output type can feed the input type and the names share a token.
    #[default]
    TextualCoercion,
}

impl MatchStrategy {
    /// The ordering rank: exact first, then semantic, then coercion.
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            Self::ExactIo => 0,
            Self::SemanticOverlap => 1,
            Self::TextualCoercion => 2,
        }
    }

    /// The `snake_case` name of the strategy, used in edge metadata.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactIo => "exact_io",
            Self::SemanticOverlap => "semantic_overlap",
            Self::TextualCoercion => "textual_coercion",
        }
    }
}

/// One deterministic `can_feed` proposal between two capabilities.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RelationCandidate {
    /// The qualified id of the producing capability.
    pub source: String,
    /// The qualified id of the consuming capability.
    pub target: String,
    /// The relation kind; always `can_feed`.
    pub relation: String,
    /// The rule confidence in `[0.0, 1.0]`; higher is stronger.
    pub confidence: f64,
    /// The rule that matched.
    pub strategy: MatchStrategy,
    /// The name of the matched input IO, when a rule matched one.
    pub matched_input: Option<String>,
    /// The name of the matched output IO, when a rule matched one.
    pub matched_output: Option<String>,
}

/// Tuning knobs for [`CandidateGenerator`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CandidateConfig {
    /// The inclusive minimum confidence a candidate must reach.
    pub min_confidence: f64,
    /// The inclusive minimum token Jaccard overlap for `SemanticOverlap`.
    pub min_token_overlap: f64,
    /// The maximum number of candidates kept per source capability.
    pub max_candidates_per_source: usize,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            min_token_overlap: DEFAULT_MIN_TOKEN_OVERLAP,
            max_candidates_per_source: DEFAULT_MAX_CANDIDATES_PER_SOURCE,
        }
    }
}

/// Derives [`RelationCandidate`]s from capability cards with deterministic rules.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CandidateGenerator {
    /// The thresholds and per-source cap applied to every run.
    config: CandidateConfig,
}

impl CandidateGenerator {
    /// Builds a generator with the given configuration.
    #[must_use]
    pub fn new(config: CandidateConfig) -> Self {
        Self { config }
    }

    /// The configuration this generator applies.
    #[must_use]
    pub fn config(&self) -> &CandidateConfig {
        &self.config
    }

    /// Generates the ordered relation candidates for a capability catalog.
    ///
    /// Every ordered pair of distinct capabilities is examined; for each pair the
    /// best-ranked rule match across all output/input combinations wins. Candidates
    /// below [`CandidateConfig::min_confidence`] are dropped and each source keeps at
    /// most [`CandidateConfig::max_candidates_per_source`] candidates, highest
    /// confidence first. The result is sorted by strategy rank, confidence descending,
    /// source, then target.
    #[must_use]
    pub fn generate(&self, capabilities: &[CapabilityDescriptor]) -> Vec<RelationCandidate> {
        let vocabulary = DataTypeVocabulary::standard();
        let mut unique: BTreeMap<(String, String), RelationCandidate> = BTreeMap::new();
        for source in capabilities {
            for target in capabilities {
                if source.qualified_id() == target.qualified_id() {
                    continue;
                }
                let Some(candidate) = self.best_between(source, target, &vocabulary) else {
                    continue;
                };
                if candidate.confidence < self.config.min_confidence {
                    continue;
                }
                let key = (candidate.source.clone(), candidate.target.clone());
                match unique.get(&key) {
                    Some(existing) if candidate_order(&candidate, existing) != Ordering::Less => {}
                    _ => {
                        unique.insert(key, candidate);
                    }
                }
            }
        }

        let mut per_source: BTreeMap<String, Vec<RelationCandidate>> = BTreeMap::new();
        for candidate in unique.into_values() {
            per_source
                .entry(candidate.source.clone())
                .or_default()
                .push(candidate);
        }
        let mut selected: Vec<RelationCandidate> = Vec::new();
        for mut candidates in per_source.into_values() {
            candidates.sort_by(|left, right| {
                right
                    .confidence
                    .total_cmp(&left.confidence)
                    .then_with(|| left.strategy.rank().cmp(&right.strategy.rank()))
                    .then_with(|| left.target.cmp(&right.target))
            });
            candidates.truncate(self.config.max_candidates_per_source);
            selected.append(&mut candidates);
        }
        selected.sort_by(candidate_order);
        selected
    }

    /// Maps candidates onto graph edges (`relation = can_feed`, `weight = confidence`).
    ///
    /// The edge metadata records the strategy name and, when a rule matched a specific
    /// IO, the matched output and input names. Input order is preserved; sort the
    /// candidates first for a canonical edge order.
    #[must_use]
    pub fn edges(&self, candidates: &[RelationCandidate]) -> Vec<CapabilityEdge> {
        candidates
            .iter()
            .map(|candidate| {
                let mut metadata = BTreeMap::new();
                metadata.insert(
                    "strategy".to_string(),
                    Value::String(candidate.strategy.as_str().to_string()),
                );
                if let Some(output) = &candidate.matched_output {
                    metadata.insert("matched_output".to_string(), Value::String(output.clone()));
                }
                if let Some(input) = &candidate.matched_input {
                    metadata.insert("matched_input".to_string(), Value::String(input.clone()));
                }
                CapabilityEdge {
                    source: candidate.source.clone(),
                    target: candidate.target.clone(),
                    relation: CAN_FEED.to_string(),
                    weight: candidate.confidence,
                    metadata,
                }
            })
            .collect()
    }

    /// Finds the best rule match between one ordered capability pair.
    fn best_between(
        &self,
        source: &CapabilityDescriptor,
        target: &CapabilityDescriptor,
        vocabulary: &DataTypeVocabulary,
    ) -> Option<RelationCandidate> {
        let mut best: Option<RelationCandidate> = None;
        for output in &source.outputs {
            for input in &target.inputs {
                let Some(candidate) =
                    self.pair_candidate(source, target, output, input, vocabulary)
                else {
                    continue;
                };
                let better = match &best {
                    None => true,
                    Some(current) => pair_order(&candidate, current) == Ordering::Less,
                };
                if better {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Applies the three rules, in priority order, to one output/input combination.
    fn pair_candidate(
        &self,
        source: &CapabilityDescriptor,
        target: &CapabilityDescriptor,
        output: &CapabilityIo,
        input: &CapabilityIo,
        vocabulary: &DataTypeVocabulary,
    ) -> Option<RelationCandidate> {
        let output_key = normalize_name_key(&output.name);
        let input_key = normalize_name_key(&input.name);
        let (strategy, confidence) = if !output_key.is_empty()
            && output_key == input_key
            && vocabulary.normalize(output.io_type()) == vocabulary.normalize(input.io_type())
        {
            (MatchStrategy::ExactIo, 1.0)
        } else {
            let overlap = token_jaccard(&io_tokens(output), &io_tokens(input));
            if overlap >= self.config.min_token_overlap {
                (MatchStrategy::SemanticOverlap, 0.70 + 0.25 * overlap)
            } else if vocabulary.can_feed(output.io_type(), input.io_type())
                && shares_name_token(output, input)
            {
                (MatchStrategy::TextualCoercion, 0.75)
            } else {
                return None;
            }
        };
        Some(RelationCandidate {
            source: source.qualified_id(),
            target: target.qualified_id(),
            relation: CAN_FEED.to_string(),
            confidence,
            strategy,
            matched_input: Some(input.name.clone()),
            matched_output: Some(output.name.clone()),
        })
    }
}

/// Orders candidates by the documented final ordering key.
fn candidate_order(left: &RelationCandidate, right: &RelationCandidate) -> Ordering {
    left.strategy
        .rank()
        .cmp(&right.strategy.rank())
        .then_with(|| right.confidence.total_cmp(&left.confidence))
        .then_with(|| left.source.cmp(&right.source))
        .then_with(|| left.target.cmp(&right.target))
}

/// Orders candidates for picking the best match between one pair: the final order plus
/// the matched IO names, so the choice never depends on iteration order.
fn pair_order(left: &RelationCandidate, right: &RelationCandidate) -> Ordering {
    candidate_order(left, right)
        .then_with(|| left.matched_output.cmp(&right.matched_output))
        .then_with(|| left.matched_input.cmp(&right.matched_input))
}

/// The token set of `"{name} {description}"`.
fn io_tokens(io: &CapabilityIo) -> BTreeSet<String> {
    tokenize(&format!("{} {}", io.name, io.description))
        .into_iter()
        .collect()
}

/// The token Jaccard similarity of two token sets, `0.0` when both are empty.
#[allow(clippy::cast_precision_loss)] // token counts are far below 2^53
fn token_jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    let intersection = left.intersection(right).count();
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Reports whether the two IO names share at least one token.
fn shares_name_token(output: &CapabilityIo, input: &CapabilityIo) -> bool {
    let output_tokens: BTreeSet<String> = tokenize(&output.name).into_iter().collect();
    let input_tokens: BTreeSet<String> = tokenize(&input.name).into_iter().collect();
    output_tokens.intersection(&input_tokens).next().is_some()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn capability(
        id: &str,
        inputs: Vec<CapabilityIo>,
        outputs: Vec<CapabilityIo>,
    ) -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", id, format!("cap {id}"))
            .with_inputs(inputs)
            .with_outputs(outputs)
    }

    fn io(name: &str, io_type: &str) -> CapabilityIo {
        CapabilityIo::new(name, io_type)
    }

    #[test]
    fn default_config_matches_the_contract() {
        let config = CandidateConfig::default();
        assert!((config.min_confidence - 0.7).abs() < f64::EPSILON);
        assert!((config.min_token_overlap - 0.6).abs() < f64::EPSILON);
        assert_eq!(config.max_candidates_per_source, 32);
        assert_eq!(
            CandidateGenerator::default().config(),
            &CandidateConfig::default()
        );
    }

    #[test]
    fn exact_io_rule_scores_one() {
        let source = capability("src", Vec::new(), vec![io("Value", "int")]);
        let target = capability("dst", vec![io("value", "integer")], Vec::new());
        let candidates = CandidateGenerator::default().generate(&[source, target]);
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.source, "tool:src");
        assert_eq!(candidate.target, "tool:dst");
        assert_eq!(candidate.relation, CAN_FEED);
        assert_eq!(candidate.strategy, MatchStrategy::ExactIo);
        assert!((candidate.confidence - 1.0).abs() < f64::EPSILON);
        assert_eq!(candidate.matched_output.as_deref(), Some("Value"));
        assert_eq!(candidate.matched_input.as_deref(), Some("value"));
    }

    #[test]
    fn exact_io_requires_equal_normalized_types() {
        // `int` normalizes to `integer`, `number` stays `number`: not exact, but the
        // type vocabulary can still coerce and the names share `value`. The extra
        // output token keeps the token Jaccard below the semantic threshold.
        let source = capability("src", Vec::new(), vec![io("value data", "int")]);
        let target = capability("dst", vec![io("value", "number")], Vec::new());
        let candidates = CandidateGenerator::default().generate(&[source, target]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].strategy, MatchStrategy::TextualCoercion);
        assert!((candidates[0].confidence - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn semantic_overlap_uses_token_jaccard() {
        let mut output = io("alpha", "string");
        output.description = "beta gamma delta".to_string();
        let mut input = io("alpha", "object");
        input.description = "beta gamma".to_string();
        // Tokens: {alpha,beta,gamma,delta} vs {alpha,beta,gamma} -> 3/4 = 0.75.
        let source = capability("src", Vec::new(), vec![output]);
        let target = capability("dst", vec![input], Vec::new());
        let candidates = CandidateGenerator::default().generate(&[source, target]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].strategy, MatchStrategy::SemanticOverlap);
        assert!((candidates[0].confidence - (0.70 + 0.25 * 0.75)).abs() < 1e-9);
    }

    #[test]
    fn textual_coercion_needs_a_compatible_type_and_a_shared_name_token() {
        let source = capability("src", Vec::new(), vec![io("payload data", "integer")]);
        let coercible = capability("ok", vec![io("payload value", "number")], Vec::new());
        let incompatible = capability("no", vec![io("payload value", "boolean")], Vec::new());
        let candidates = CandidateGenerator::default().generate(&[source, coercible, incompatible]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].target, "tool:ok");
        assert_eq!(candidates[0].strategy, MatchStrategy::TextualCoercion);
    }

    #[test]
    fn low_confidence_candidates_are_dropped() {
        let source = capability("src", Vec::new(), vec![io("value", "string")]);
        let target = capability("dst", vec![io("value extra", "string")], Vec::new());
        let generator = CandidateGenerator::new(CandidateConfig {
            min_confidence: 0.8,
            ..CandidateConfig::default()
        });
        assert!(
            generator.generate(&[source, target]).is_empty(),
            "0.75 must not reach the 0.8 confidence threshold"
        );
    }

    #[test]
    fn candidates_are_ordered_by_strategy_then_confidence() {
        let source = capability(
            "src",
            Vec::new(),
            vec![io("value", "string"), io("count", "integer")],
        );
        let exact = capability("exact", vec![io("value", "string")], Vec::new());
        let semantic = capability("semantic", vec![io("value", "object")], Vec::new());
        let coercion = capability("coerce", vec![io("count total", "number")], Vec::new());
        let candidates =
            CandidateGenerator::default().generate(&[source, exact, semantic, coercion]);
        let targets: Vec<&str> = candidates
            .iter()
            .map(|candidate| candidate.target.as_str())
            .collect();
        assert_eq!(targets, ["tool:exact", "tool:semantic", "tool:coerce"]);
    }

    #[test]
    fn each_source_keeps_the_best_candidates_up_to_the_cap() {
        let source = capability("src", Vec::new(), vec![io("value", "string")]);
        let exact = capability("exact", vec![io("value", "string")], Vec::new());
        let semantic = capability("semantic", vec![io("value", "object")], Vec::new());
        let coercion = capability("coerce", vec![io("value extra", "string")], Vec::new());
        let generator = CandidateGenerator::new(CandidateConfig {
            max_candidates_per_source: 2,
            ..CandidateConfig::default()
        });
        let candidates = generator.generate(&[source, exact, semantic, coercion]);
        let targets: Vec<&str> = candidates
            .iter()
            .map(|candidate| candidate.target.as_str())
            .collect();
        assert_eq!(targets, ["tool:exact", "tool:semantic"]);
    }

    #[test]
    fn identical_pairs_deduplicate_and_self_pairs_are_skipped() {
        let source = capability("src", Vec::new(), vec![io("value", "string")]);
        let target = capability("dst", vec![io("value", "string")], Vec::new());
        let candidates = CandidateGenerator::default().generate(&[
            source.clone(),
            source,
            target.clone(),
            target,
        ]);
        assert_eq!(candidates.len(), 1);
        let single = capability("only", vec![io("value", "string")], Vec::new());
        assert!(
            CandidateGenerator::default().generate(&[single]).is_empty(),
            "a capability cannot feed itself"
        );
    }

    #[test]
    fn edges_carry_the_strategy_and_matched_ios() {
        let source = capability("src", Vec::new(), vec![io("value", "string")]);
        let target = capability("dst", vec![io("value", "string")], Vec::new());
        let generator = CandidateGenerator::default();
        let candidates = generator.generate(&[source, target]);
        let edges = generator.edges(&candidates);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, "tool:src");
        assert_eq!(edges[0].target, "tool:dst");
        assert_eq!(edges[0].relation, CAN_FEED);
        assert!((edges[0].weight - 1.0).abs() < f64::EPSILON);
        assert_eq!(edges[0].metadata["strategy"], "exact_io");
        assert_eq!(edges[0].metadata["matched_output"], "value");
        assert_eq!(edges[0].metadata["matched_input"], "value");
    }

    #[test]
    fn zero_candidate_cap_keeps_nothing() {
        let source = capability("src", Vec::new(), vec![io("value", "string")]);
        let target = capability("dst", vec![io("value", "string")], Vec::new());
        let generator = CandidateGenerator::new(CandidateConfig {
            max_candidates_per_source: 0,
            ..CandidateConfig::default()
        });
        assert!(
            generator.generate(&[source, target]).is_empty(),
            "a zero per-source cap keeps nothing"
        );
    }
}
