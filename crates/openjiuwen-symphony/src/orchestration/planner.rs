//! Bounded, deterministic plan search over the capability relationship graph.
//!
//! [`Planner`] answers the engine's central planning question — given a request, which
//! capabilities must run, and in what order — by walking the `can_feed` edges of a
//! [`CapabilityGraph`]. The search is a bounded graph walk, not an LLM beam search:
//! every candidate plan is scored with the Python orchestration formula
//!
//! ```text
//! score = 0.45*mean(semantic) + 0.25*mean(edge_confidence)
//!       + 0.20*input_coverage - 0.03*(steps - 1)
//! ```
//!
//! and candidates rank on `(-score, missing_inputs, step_count, step_ids,
//! edge_weights)`, so ties break on identifiers rather than on hash order.
//!
//! # Search semantics
//!
//! - The **start set** is every capability with a positive [`SemanticScorer`] value
//!   (all `available` capabilities when the query has no tokens), minus
//!   [`PlanRequest::disabled_ids`], intersected with [`PlanRequest::candidate_ids`]
//!   when given, and restricted to `available` capabilities. `disabled_ids` and
//!   `candidate_ids` additionally prune expansion targets, matching Python's
//!   `RuntimeEdgeResolver`, which only resolves edges inside the candidate set.
//! - **Expansion** follows `can_feed` graph edges with a weight of at least
//!   [`PlanConfig::min_edge_confidence`], never revisits a capability already in the
//!   path, and stops at [`PlanConfig::max_depth`] steps, [`PlanConfig::max_expansions`]
//!   edge evaluations, or [`PlanConfig::max_plans`] retained candidate plans.
//! - Availability gates the start set only: a withdrawn capability is never a seed,
//!   but it may still be planned downstream of a seed. This is the documented
//!   start-set rule; the runtime normally builds the graph from available assets only.
//! - [`PlanConfig::dynamic_graph_enabled`] has no separate behavior here: the planner
//!   always reads the static graph it is handed.
//! - [`PlanMode::Fast`] keeps one greedily extended path per seed plus that seed's
//!   single-node plan. [`PlanMode::Beam`] keeps the best [`PlanConfig::top_k`] partial
//!   paths per depth (and every seed's single-node plan) before final ranking.
//!
//! # Determinism
//!
//! Scores are sanitized to `0.0` when non-finite, all sets and result vectors are
//! ordered, and no wall clock or hash-map iteration order leaks into the plan. The one
//! intentional exception is [`super::plan::OrchestrationPlan::plan_id`], which is a
//! fresh [`uuid_hex`] identifier per call (Python parity for plan identity); every
//! other plan field is reproducible from the inputs.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::utils::id::uuid_hex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Result};
use crate::graph::{CapabilityGraph, CAN_FEED};
use crate::model::{normalize_name_key, CapabilityDescriptor, CapabilityIo, DataTypeVocabulary};
use crate::search::tokenize;

use super::plan::{OrchestrationPlan, PlanEdge, PlanStatus, PlanStep};

/// Weight of the mean semantic score in the plan-score formula.
const SEMANTIC_WEIGHT: f64 = 0.45;
/// Weight of the mean edge confidence in the plan-score formula.
const EDGE_WEIGHT: f64 = 0.25;
/// Weight of the input coverage in the plan-score formula.
const COVERAGE_WEIGHT: f64 = 0.20;
/// Per-extra-step penalty in the plan-score formula.
const DEPTH_PENALTY: f64 = 0.03;
/// The semantic baseline Python uses when a state carries no semantic scores.
const SEED_BASELINE_SCORE: f64 = 0.5;
/// The default beam width.
const DEFAULT_TOP_K: usize = 3;
/// The default maximum number of steps in a plan.
const DEFAULT_MAX_DEPTH: usize = 4;
/// The default minimum edge weight the planner follows.
const DEFAULT_MIN_EDGE_CONFIDENCE: f64 = 0.7;
/// The default cap on edge evaluations.
const DEFAULT_MAX_EXPANSIONS: usize = 128;
/// The default cap on retained candidate plans.
const DEFAULT_MAX_PLANS: usize = 256;
/// The canonical type names treated as "unknown" during input matching.
const UNKNOWN_TYPE: &str = "unknown";
/// The canonical untyped name.
const ANY_TYPE: &str = "any";

/// How the planner explores the capability graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanMode {
    /// One greedy best path per start capability plus each start's single-node plan.
    #[default]
    Fast,
    /// The best [`PlanConfig::top_k`] partial paths per depth before final ranking.
    Beam,
}

/// Tuning knobs for the bounded plan search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PlanConfig {
    /// The search mode; [`PlanMode::Fast`] by default.
    pub mode: PlanMode,
    /// The beam width for [`PlanMode::Beam`], and a sensible limit for path pools.
    pub top_k: usize,
    /// The maximum number of steps in a plan (clamped to at least `1`).
    pub max_depth: usize,
    /// The inclusive minimum `can_feed` edge weight the planner follows.
    pub min_edge_confidence: f64,
    /// The maximum number of graph edges evaluated during the search.
    pub max_expansions: usize,
    /// The maximum number of candidate plans retained for final ranking.
    pub max_plans: usize,
    /// Reserved for Python's dynamic graph overlay; the static graph is always used.
    pub dynamic_graph_enabled: bool,
}

impl Default for PlanConfig {
    /// Returns the documented defaults: `Fast`, beam width `3`, depth `4`, minimum edge
    /// confidence `0.7`, `128` expansions, `256` candidate plans, static graph.
    fn default() -> Self {
        Self {
            mode: PlanMode::Fast,
            top_k: DEFAULT_TOP_K,
            max_depth: DEFAULT_MAX_DEPTH,
            min_edge_confidence: DEFAULT_MIN_EDGE_CONFIDENCE,
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            max_plans: DEFAULT_MAX_PLANS,
            dynamic_graph_enabled: false,
        }
    }
}

/// One planning request.
///
/// `query`, `candidate_ids`, and `provided_inputs` drive the search; `top_k` and
/// `max_depth` override [`PlanConfig`] for this call only. `language` is normalized by
/// [`normalize_language`], and `provided_inputs` are input names the caller promises to
/// supply (they are matched after [`normalize_name_key`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct PlanRequest {
    /// The natural-language request the plan must answer.
    pub query: String,
    /// When set, only these qualified ids may appear in the plan.
    pub candidate_ids: Option<Vec<String>>,
    /// Qualified ids excluded from the plan entirely.
    pub disabled_ids: Vec<String>,
    /// Input names the caller supplies, so they are never reported missing.
    pub provided_inputs: Vec<String>,
    /// The response language: `"cn"` (default) or `"en"`.
    pub language: String,
    /// An optional beam-width override for this request.
    pub top_k: Option<usize>,
    /// An optional maximum-depth override for this request.
    pub max_depth: Option<usize>,
}

impl PlanRequest {
    /// Builds a Chinese, default-config request for `query`.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Self::default()
        }
    }
}

impl Default for PlanRequest {
    /// Returns an empty `"cn"` request with no filters and no overrides.
    fn default() -> Self {
        Self {
            query: String::new(),
            candidate_ids: None,
            disabled_ids: Vec::new(),
            provided_inputs: Vec::new(),
            language: "cn".to_string(),
            top_k: None,
            max_depth: None,
        }
    }
}

/// Assigns a semantic relevance score to each capability for a query.
///
/// Keys are qualified capability ids. The planner treats a missing key as `0.0`, so an
/// implementation may score only the capabilities it recognizes. This is the seam where
/// an LLM scorer can replace the deterministic [`LexicalSemanticScorer`].
#[async_trait]
pub trait SemanticScorer: Send + Sync {
    /// Scores each capability in `capabilities` against `query`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error`] when the scorer cannot produce scores; the
    /// planner propagates the error unchanged.
    async fn score(
        &self,
        query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<BTreeMap<String, f64>>;
}

/// The deterministic default scorer: query-term coverage of the retrieval text.
///
/// Each score is the fraction of distinct query tokens present in the capability's
/// [`CapabilityDescriptor::retrieval_text`], in `[0, 1]`. A blank or tokenless query
/// scores every capability `1.0`, because such a request does not discriminate between
/// capabilities (the planner then starts from every available capability).
#[derive(Debug, Clone, Copy, Default)]
pub struct LexicalSemanticScorer;

impl LexicalSemanticScorer {
    /// Builds the lexical scorer (it is stateless).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl SemanticScorer for LexicalSemanticScorer {
    async fn score(
        &self,
        query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<BTreeMap<String, f64>> {
        let query_terms: BTreeSet<String> = tokenize(query).into_iter().collect();
        let mut scores = BTreeMap::new();
        for capability in capabilities {
            let value = if query_terms.is_empty() {
                1.0
            } else {
                let document: BTreeSet<String> =
                    tokenize(&capability.retrieval_text()).into_iter().collect();
                let matched = query_terms
                    .iter()
                    .filter(|term| document.contains(*term))
                    .count();
                ratio(matched, query_terms.len())
            };
            scores.insert(capability.qualified_id(), value);
        }
        Ok(scores)
    }
}

/// The bounded, deterministic planner.
///
/// A planner is cheap to clone: it shares the [`SemanticScorer`] behind an [`Arc`].
/// [`Planner::plan`] takes `&self`, so one planner can serve concurrent requests.
#[derive(Clone)]
pub struct Planner {
    /// The search configuration.
    config: PlanConfig,
    /// The semantic scorer used to seed and score plans.
    scorer: Arc<dyn SemanticScorer>,
}

impl Planner {
    /// Builds a planner with the lexical [`LexicalSemanticScorer`].
    #[must_use]
    pub fn new(config: PlanConfig) -> Self {
        Self {
            config,
            scorer: Arc::new(LexicalSemanticScorer::new()),
        }
    }

    /// Builds a planner with a custom [`SemanticScorer`].
    #[must_use]
    pub fn with_scorer(config: PlanConfig, scorer: Arc<dyn SemanticScorer>) -> Self {
        Self { config, scorer }
    }

    /// Returns the planner's configuration.
    #[must_use]
    pub fn config(&self) -> &PlanConfig {
        &self.config
    }

    /// Plans the request over `graph` and the capability catalog.
    ///
    /// `request.top_k` and `request.max_depth` override the configuration for this call.
    /// The result always has a normalized [`OrchestrationPlan::language`] and a
    /// [`plan_title`]; when no start capability qualifies, it is an empty
    /// [`PlanStatus::NoPlan`] plan with score `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] naming `query` when the request query
    /// is blank, or any error raised by the configured [`SemanticScorer`].
    pub async fn plan(
        &self,
        request: &PlanRequest,
        graph: &CapabilityGraph,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<OrchestrationPlan> {
        if request.query.trim().is_empty() {
            return Err(validation("query", "must not be blank"));
        }

        let language = normalize_language(&request.language);
        let max_depth = request.max_depth.unwrap_or(self.config.max_depth).max(1);
        let top_k = request.top_k.unwrap_or(self.config.top_k);

        let scores = self.scorer.score(&request.query, capabilities).await?;

        // A BTreeMap keeps catalog iteration sorted by qualified id, so the start set
        // and every downstream ordering are deterministic.
        let catalog: BTreeMap<String, &CapabilityDescriptor> = capabilities
            .iter()
            .map(|capability| (capability.qualified_id(), capability))
            .collect();

        let provided: BTreeSet<String> = request
            .provided_inputs
            .iter()
            .map(|name| normalize_name_key(name))
            .filter(|key| !key.is_empty())
            .collect();
        let disabled: BTreeSet<&str> = request.disabled_ids.iter().map(String::as_str).collect();
        let candidate_filter: Option<BTreeSet<&str>> = request
            .candidate_ids
            .as_ref()
            .map(|ids| ids.iter().map(String::as_str).collect());

        let tokenless = tokenize(&request.query).is_empty();
        let mut starts = Vec::new();
        for (id, capability) in &catalog {
            if !capability.available || disabled.contains(id.as_str()) {
                continue;
            }
            if let Some(filter) = &candidate_filter {
                if !filter.contains(id.as_str()) {
                    continue;
                }
            }
            if tokenless || finite(scores.get(id).copied().unwrap_or(0.0)) > 0.0 {
                starts.push(id.clone());
            }
        }

        if starts.is_empty() {
            return Ok(no_plan(request, &language));
        }

        let search = Search {
            graph,
            catalog: &catalog,
            scores: &scores,
            provided: &provided,
            disabled: &disabled,
            candidate_filter: candidate_filter.as_ref(),
            vocabulary: DataTypeVocabulary::standard(),
            min_edge_confidence: self.config.min_edge_confidence,
            max_expansions: self.config.max_expansions,
            max_plans: self.config.max_plans,
            mode: self.config.mode,
            top_k,
            max_depth,
            expansions: 0,
            candidates: Vec::new(),
        };
        let candidates = search.run(&starts);

        let Some(best) = candidates.into_iter().next() else {
            return Ok(no_plan(request, &language));
        };

        let root_missing = best
            .steps
            .first()
            .map(|step| step.missing_inputs.clone())
            .unwrap_or_default();
        let status = if root_missing.is_empty() {
            PlanStatus::Ready
        } else {
            PlanStatus::NeedsInput
        };

        let mut plan = OrchestrationPlan::new(uuid_hex(), request.query.clone(), language.clone())
            .with_title(plan_title(&request.query, &language))
            .with_status(status)
            .with_steps(best.steps)
            .with_edges(best.edges)
            .with_score(best.score);
        if !root_missing.is_empty() {
            plan = plan.with_message(missing_message(&root_missing, &language));
        }
        Ok(plan)
    }
}

/// Returns the empty [`PlanStatus::NoPlan`] answer for a request.
fn no_plan(request: &PlanRequest, language: &str) -> OrchestrationPlan {
    OrchestrationPlan::new(uuid_hex(), request.query.clone(), language.to_string())
        .with_title(plan_title(&request.query, language))
        .with_status(PlanStatus::NoPlan)
}

/// Builds the caller-facing explanation for unfilled root inputs.
fn missing_message(names: &[String], language: &str) -> String {
    let joined = names.join(", ");
    if normalize_language(language) == "en" {
        format!("missing inputs: {joined}")
    } else {
        format!("缺少输入：{joined}")
    }
}

/// Normalizes a language tag to the plan vocabulary: `"cn"` or `"en"`.
///
/// `"zh"`, `"cn"`, `"zh_cn"`, `"zh-cn"`, and `"zh-Hans"`-style tags normalize to
/// `"cn"`; `"en"`, `"en_us"`, and `"en-US"` normalize to `"en"`. The comparison is
/// case-insensitive and trimmed, and anything unrecognized falls back to `"cn"`.
#[must_use]
pub fn normalize_language(raw: &str) -> String {
    let normalized = raw.trim().to_lowercase().replace('-', "_");
    match normalized.as_str() {
        "en" | "en_us" => "en".to_string(),
        _ => "cn".to_string(),
    }
}

/// Builds the display title for a plan: `"Plan: {query}"` or `"计划：{query}"`.
#[must_use]
pub fn plan_title(query: &str, language: &str) -> String {
    if normalize_language(language) == "en" {
        format!("Plan: {query}")
    } else {
        format!("计划：{query}")
    }
}

/// Converts a pair of counts into the ratio `numerator / denominator`.
#[allow(clippy::cast_precision_loss)] // counts are far below 2^53
fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

/// Returns `value` when it is finite, and `0.0` otherwise.
fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Compares two weight sequences deterministically.
fn cmp_weights(left: &[f64], right: &[f64]) -> Ordering {
    for (left_weight, right_weight) in left.iter().zip(right) {
        let ordering = left_weight.total_cmp(right_weight);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

/// A plan under construction: the full steps, the edges, and the ranking metadata.
#[derive(Debug, Clone, PartialEq)]
struct BuiltPlan {
    /// The plan steps in execution order.
    steps: Vec<PlanStep>,
    /// The consecutive data-flow edges between steps.
    edges: Vec<PlanEdge>,
    /// The weight of each edge, used as the final ranking tie-break.
    edge_weights: Vec<f64>,
    /// The composite score from the ranking formula.
    score: f64,
}

impl BuiltPlan {
    /// The total number of unfilled required inputs across all steps.
    fn missing_count(&self) -> usize {
        self.steps
            .iter()
            .map(|step| step.missing_inputs.len())
            .sum()
    }

    /// The step ids in execution order, the deterministic identity of a linear path.
    fn step_ids(&self) -> Vec<&String> {
        self.steps.iter().map(|step| &step.step_id).collect()
    }
}

/// Ranks two candidate plans: higher score first, then fewer missing inputs, then fewer
/// steps, then step ids, then edge weights.
fn rank_cmp(left: &BuiltPlan, right: &BuiltPlan) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.missing_count().cmp(&right.missing_count()))
        .then_with(|| left.steps.len().cmp(&right.steps.len()))
        .then_with(|| left.step_ids().cmp(&right.step_ids()))
        .then_with(|| cmp_weights(&left.edge_weights, &right.edge_weights))
}

/// A partial path during the search: ordered step ids plus the traversed edge weights.
#[derive(Debug, Clone)]
struct PartialPath {
    /// The step ids visited so far.
    ids: Vec<String>,
    /// The edge weights between consecutive steps.
    weights: Vec<f64>,
}

/// One bounded search over the capability graph.
struct Search<'a> {
    /// The capability relationship graph.
    graph: &'a CapabilityGraph,
    /// The capability catalog, keyed by qualified id.
    catalog: &'a BTreeMap<String, &'a CapabilityDescriptor>,
    /// The semantic scorer output.
    scores: &'a BTreeMap<String, f64>,
    /// The caller-supplied input names, normalized.
    provided: &'a BTreeSet<String>,
    /// Qualified ids excluded from the plan.
    disabled: &'a BTreeSet<&'a str>,
    /// The optional candidate-id allowlist.
    candidate_filter: Option<&'a BTreeSet<&'a str>>,
    /// The data-type compatibility vocabulary.
    vocabulary: DataTypeVocabulary,
    /// The minimum edge weight followed during expansion.
    min_edge_confidence: f64,
    /// The cap on edge evaluations.
    max_expansions: usize,
    /// The cap on retained candidate plans.
    max_plans: usize,
    /// The search mode.
    mode: PlanMode,
    /// The beam width.
    top_k: usize,
    /// The maximum number of steps per plan.
    max_depth: usize,
    /// The number of edge evaluations performed so far.
    expansions: usize,
    /// The retained candidate plans, ranked when the search finishes.
    candidates: Vec<BuiltPlan>,
}

impl Search<'_> {
    /// Runs the configured search and returns the ranked candidate plans.
    fn run(mut self, starts: &[String]) -> Vec<BuiltPlan> {
        let seeds: Vec<PartialPath> = starts
            .iter()
            .map(|id| PartialPath {
                ids: vec![id.clone()],
                weights: Vec::new(),
            })
            .collect();
        match self.mode {
            PlanMode::Fast => self.run_fast(seeds),
            PlanMode::Beam => self.run_beam(seeds),
        }
        self.candidates.sort_by(rank_cmp);
        self.candidates
    }

    /// Greedy search: one best extension per seed, plus the seed's single-node plan.
    fn run_fast(&mut self, seeds: Vec<PartialPath>) {
        for mut path in seeds {
            let Some(singleton) = self.build(&path.ids, &path.weights) else {
                continue;
            };
            if self.candidates.len() < self.max_plans {
                self.candidates.push(singleton);
            }
            while path.ids.len() < self.max_depth && self.candidates.len() < self.max_plans {
                let extensions = self.eligible(&path);
                if extensions.is_empty() {
                    break;
                }
                let mut best: Option<(PartialPath, BuiltPlan)> = None;
                for (target, weight) in extensions {
                    let mut next = path.clone();
                    next.ids.push(target);
                    next.weights.push(weight);
                    let Some(candidate) = self.build(&next.ids, &next.weights) else {
                        continue;
                    };
                    let better = match &best {
                        None => true,
                        Some((_, current)) => rank_cmp(&candidate, current) == Ordering::Less,
                    };
                    if better {
                        best = Some((next, candidate));
                    }
                }
                let Some((next, candidate)) = best else {
                    break;
                };
                path = next;
                self.candidates.push(candidate);
            }
        }
    }

    /// Beam search: the best `top_k` partial paths per depth, plus every seed's plan.
    fn run_beam(&mut self, mut frontier: Vec<PartialPath>) {
        for path in &frontier {
            let Some(plan) = self.build(&path.ids, &path.weights) else {
                continue;
            };
            if self.candidates.len() < self.max_plans {
                self.candidates.push(plan);
            }
        }

        while !frontier.is_empty() {
            let mut expanded: Vec<(PartialPath, BuiltPlan)> = Vec::new();
            for path in &frontier {
                if path.ids.len() >= self.max_depth {
                    continue;
                }
                for (target, weight) in self.eligible(path) {
                    let mut next = path.clone();
                    next.ids.push(target);
                    next.weights.push(weight);
                    if let Some(plan) = self.build(&next.ids, &next.weights) {
                        expanded.push((next, plan));
                    }
                }
            }
            if expanded.is_empty() {
                break;
            }
            expanded.sort_by(|left, right| rank_cmp(&left.1, &right.1));
            expanded.truncate(self.top_k);
            let mut next_frontier = Vec::with_capacity(expanded.len());
            for (path, plan) in expanded {
                if self.candidates.len() < self.max_plans {
                    self.candidates.push(plan);
                }
                next_frontier.push(path);
            }
            frontier = next_frontier;
        }
    }

    /// The eligible extensions of `path`, sorted by target id and weight, truncated to
    /// the remaining expansion budget.
    fn eligible(&mut self, path: &PartialPath) -> Vec<(String, f64)> {
        let Some(tail) = path.ids.last() else {
            return Vec::new();
        };
        let remaining = self.max_expansions.saturating_sub(self.expansions);
        if remaining == 0 {
            return Vec::new();
        }
        let mut extensions: Vec<(String, f64)> = self
            .graph
            .outgoing(tail)
            .into_iter()
            .filter(|edge| edge.relation == CAN_FEED)
            .filter(|edge| edge.weight.is_finite() && edge.weight >= self.min_edge_confidence)
            .filter(|edge| !path.ids.iter().any(|id| id == &edge.target))
            .filter(|edge| {
                self.catalog.contains_key(&edge.target) && !self.is_forbidden(&edge.target)
            })
            .map(|edge| (edge.target.clone(), edge.weight))
            .collect();
        extensions.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| right.1.total_cmp(&left.1))
        });
        extensions.truncate(remaining);
        self.expansions += extensions.len();
        extensions
    }

    /// Reports whether `id` is excluded by `disabled_ids` or outside `candidate_ids`.
    fn is_forbidden(&self, id: &str) -> bool {
        if self.disabled.contains(id) {
            return true;
        }
        match self.candidate_filter {
            Some(filter) => !filter.contains(id),
            None => false,
        }
    }

    /// Builds a scored plan from a path of step ids and edge weights.
    fn build(&self, ids: &[String], weights: &[f64]) -> Option<BuiltPlan> {
        let mut steps = Vec::with_capacity(ids.len());
        let mut all_inputs: BTreeSet<String> = BTreeSet::new();
        let mut covered_inputs: BTreeSet<String> = BTreeSet::new();
        for (index, id) in ids.iter().enumerate() {
            let capability = self.catalog.get(id)?;
            let predecessors: Vec<&CapabilityDescriptor> = ids[..index]
                .iter()
                .filter_map(|predecessor| self.catalog.get(predecessor).copied())
                .collect();
            let mut missing = BTreeSet::new();
            let mut filled = BTreeSet::new();
            for input in &capability.inputs {
                let available = self.input_available(input, &predecessors);
                let name = display_name(input);
                let key = normalize_name_key(&input.name);
                if !key.is_empty() {
                    all_inputs.insert(key.clone());
                    if available {
                        covered_inputs.insert(key);
                    }
                }
                if input.required != Some(false) {
                    if available {
                        filled.insert(name);
                    } else {
                        missing.insert(name);
                    }
                }
            }
            steps.push(PlanStep {
                step_id: id.clone(),
                capability_type: capability.capability_type.clone(),
                name: capability.name.clone(),
                inputs: capability.inputs.clone(),
                outputs: capability.outputs.clone(),
                missing_inputs: missing.into_iter().collect(),
                filled_inputs: filled.into_iter().collect(),
                score: self.semantic_score(id),
            });
        }

        let edges: Vec<PlanEdge> = ids
            .windows(2)
            .map(|pair| PlanEdge {
                source: pair[0].clone(),
                target: pair[1].clone(),
            })
            .collect();
        let expected_weights = ids.len().saturating_sub(1);
        let mut edge_weights: Vec<f64> = weights
            .iter()
            .take(expected_weights)
            .map(|weight| finite(*weight))
            .collect();
        while edge_weights.len() < expected_weights {
            edge_weights.push(1.0);
        }
        let score = composite_score(
            &steps,
            &edge_weights,
            covered_inputs.len(),
            all_inputs.len(),
        );
        Some(BuiltPlan {
            steps,
            edges,
            edge_weights,
            score,
        })
    }

    /// The semantic score of one capability (`0.0` when the scorer omitted it).
    fn semantic_score(&self, id: &str) -> f64 {
        finite(self.scores.get(id).copied().unwrap_or(0.0))
    }

    /// Reports whether an earlier step or the caller supplies `input`.
    fn input_available(
        &self,
        input: &CapabilityIo,
        predecessors: &[&CapabilityDescriptor],
    ) -> bool {
        let key = normalize_name_key(&input.name);
        if key.is_empty() {
            return false;
        }
        if self.provided.contains(&key) {
            return true;
        }
        predecessors.iter().any(|predecessor| {
            predecessor.outputs.iter().any(|output| {
                normalize_name_key(&output.name) == key
                    && self.types_compatible(output.io_type(), input.io_type())
            })
        })
    }

    /// Reports whether `output_type` can feed `input_type`.
    ///
    /// Types are equal, either side is `any`/`unknown`, or the vocabulary's subtype
    /// relation holds (`integer` → `number`, `text` → `string`).
    fn types_compatible(&self, output_type: &str, input_type: &str) -> bool {
        let output = self.vocabulary.normalize(output_type);
        let input = self.vocabulary.normalize(input_type);
        output == input
            || output == ANY_TYPE
            || input == ANY_TYPE
            || output == UNKNOWN_TYPE
            || input == UNKNOWN_TYPE
            || self.vocabulary.is_subtype(&output, &input)
    }
}

/// The display form of an input name: its normalized key, or the trimmed raw name when
/// the name normalizes to nothing.
fn display_name(input: &CapabilityIo) -> String {
    let key = normalize_name_key(&input.name);
    if key.is_empty() {
        input.name.trim().to_string()
    } else {
        key
    }
}

/// Computes the composite plan score from the built steps and coverage counts.
fn composite_score(
    steps: &[PlanStep],
    edge_weights: &[f64],
    covered_inputs: usize,
    total_inputs: usize,
) -> f64 {
    let semantic = if steps.is_empty() {
        SEED_BASELINE_SCORE
    } else {
        steps.iter().map(|step| finite(step.score)).sum::<f64>() / len_f64(steps.len())
    };
    let edge_confidence = if edge_weights.is_empty() {
        1.0
    } else {
        edge_weights.iter().copied().map(finite).sum::<f64>() / len_f64(edge_weights.len())
    };
    let coverage = if total_inputs == 0 {
        1.0
    } else {
        ratio(covered_inputs, total_inputs)
    };
    let depth_penalty = len_f64(steps.len().saturating_sub(1));
    finite(
        SEMANTIC_WEIGHT * semantic + EDGE_WEIGHT * edge_confidence + COVERAGE_WEIGHT * coverage
            - DEPTH_PENALTY * depth_penalty,
    )
}

/// The `usize` length as an `f64` (`0.0` and `1.0` are exact; plan lengths are small).
#[allow(clippy::cast_precision_loss)] // plan lengths are far below 2^53
fn len_f64(length: usize) -> f64 {
    length as f64
}
