//! The flow engine: evidence ingestion, distillation, and recipe access.
//!
//! One engine owns a flow root (or an empty temporary root) and the in-memory
//! catalog hydrated from it. Writes are eager and atomic, so `close` has nothing
//! left to flush. Distillation is serialized by an async mutex, while ordinary
//! reads take a short `RwLock` snapshot.
//!
//! # Distillation
//!
//! Every successful trace is grouped by its sorted successful-edge signature
//! (`source > target : relation`, cycles broken deterministically). The recipe id
//! is `recipe_<sha256(signature)[:12]>`; a group's first distillation produces
//! version 1 and a changed content identity (name, applicability, structure, or
//! narrative) produces the next version. Replaying unchanged evidence reports the
//! recipe as unchanged and writes nothing, so published versions stay immutable.
//!
//! Quality counts are captured when a version is written. A group whose structure
//! and narrative did not change keeps its existing version and its existing
//! quality snapshot.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::utils::id::now_rfc3339;
use openjiuwen_retrieval::{extract_json_object, CompletionModel};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};
use tracing::warn;

use crate::error::{validation, Result};
use crate::flow::model::{
    normalize_meta_name, sha256_short, topological_order, CombinationCandidate,
    CombinationStructure, ExperienceRecipe, FlowTrace, RecipeEdge, RecipeQuality, TraceNode,
    MIN_PACK_SUCCESS_RATE, MIN_SUCCESSES, RECIPE_GRADE_CANDIDATE, RECIPE_GRADE_VERIFIED,
    RECIPE_SCHEMA_VERSION, RECIPE_STATUS_ACTIVE, RECIPE_STATUS_DRAFT, SKILL_PACK_TYPE,
};
use crate::flow::pack::SkillPack;
use crate::flow::store::FlowStore;
use crate::redaction::{
    redact_sensitive_json, redact_sensitive_text, sanitize_distilled_text, sanitize_with_queries,
};

/// The summary of one batch distillation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DistillReport {
    /// Number of successful evidence traces considered.
    pub evidence_total: usize,
    /// Number of distinct edge-structure groups.
    pub groups_total: usize,
    /// Trace ids that produced no structure and were skipped.
    pub groups_without_structure: Vec<String>,
    /// Recipe ids that produced a new version.
    pub recipes_saved: Vec<String>,
    /// Recipe ids whose content was unchanged.
    pub recipes_unchanged: Vec<String>,
    /// Recipe ids whose model narrative failed and fell back to the template.
    pub narrative_fallbacks: Vec<String>,
}

/// The engine's mutable in-memory catalog.
#[derive(Default)]
struct FlowState {
    /// Evidence keyed by trace id.
    evidence: BTreeMap<String, FlowTrace>,
    /// Current recipe versions keyed by recipe id.
    recipes: BTreeMap<String, ExperienceRecipe>,
    /// Frozen packages keyed by package id.
    packages: BTreeMap<String, SkillPack>,
}

/// One edge-structure group of successful traces.
struct StructureGroup {
    /// The sha256 prefix that identifies the structure.
    signature: String,
    /// The cycle-free edge triples.
    edges: Vec<(String, String, String)>,
    /// Member traces, sorted by trace id.
    records: Vec<FlowTrace>,
}

/// The texts distilled for one recipe version.
#[derive(Debug, Clone)]
struct RecipeTexts {
    /// The installable display name.
    name: String,
    /// The applicability metadata.
    applicability: BTreeMap<String, Value>,
    /// The execution narrative.
    narrative: String,
    /// `template` or `llm`.
    source: &'static str,
}

/// Distills execution evidence into versioned recipes and prepares packs.
pub struct FlowEngine {
    /// Filesystem persistence.
    store: FlowStore,
    /// Optional narrative model; structure never depends on it.
    model: Option<Arc<dyn CompletionModel>>,
    /// The in-memory catalog.
    state: RwLock<FlowState>,
    /// Serializes distillation runs.
    distill_guard: Mutex<()>,
}

impl FlowEngine {
    /// Creates an engine rooted at `root` without reading or writing the disk.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            store: FlowStore::new(root),
            model: None,
            state: RwLock::new(FlowState::default()),
            distill_guard: Mutex::new(()),
        }
    }

    /// Opens (or creates) a flow root and hydrates its evidence, recipes, and
    /// packages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`](openjiuwen_common::error::Error::external)
    /// when the root or one of its files cannot be read; malformed individual
    /// records are logged and skipped.
    pub async fn open(root: impl AsRef<Path>) -> Result<Self> {
        let store = FlowStore::new(root.as_ref().to_path_buf());
        store.ensure_layout().await?;
        let mut evidence = BTreeMap::new();
        for trace in store.load_evidence().await? {
            evidence.entry(trace.trace_id.clone()).or_insert(trace);
        }
        let recipes = store
            .load_recipes()
            .await?
            .into_iter()
            .map(|recipe| (recipe.recipe_id.clone(), recipe))
            .collect();
        let packages = store
            .load_packages()
            .await?
            .into_iter()
            .map(|pack| (pack.package_id.clone(), pack))
            .collect();
        Ok(Self {
            store,
            model: None,
            state: RwLock::new(FlowState {
                evidence,
                recipes,
                packages,
            }),
            distill_guard: Mutex::new(()),
        })
    }

    /// Attaches a narrative model; structure stays deterministic either way.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Returns the flow root.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.store.root()
    }

    /// Ingests one normalized trace.
    ///
    /// Returns `true` when the trace was stored and `false` when its `trace_id`
    /// was already present. Traces must be successful and carry at least one
    /// successful edge.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) for a blank trace id, a non-successful
    /// outcome, or a graph without successful edges, and
    /// [`Error::External`](openjiuwen_common::error::Error::external) when the
    /// evidence file cannot be written.
    pub async fn ingest(&self, trace: FlowTrace) -> Result<bool> {
        if trace.trace_id.trim().is_empty() {
            return Err(validation("trace_id", "must not be blank"));
        }
        if !trace.outcome.is_success() {
            return Err(validation(
                "outcome",
                format!("trace {} is not a successful execution", trace.trace_id),
            ));
        }
        if trace.graph.edges.is_empty() {
            return Err(validation(
                "graph",
                format!("trace {} has no successful edges", trace.trace_id),
            ));
        }
        let mut state = self.state.write().await;
        if state.evidence.contains_key(&trace.trace_id) {
            return Ok(false);
        }
        let mut next = state.evidence.clone();
        next.insert(trace.trace_id.clone(), trace);
        let records: Vec<FlowTrace> = next.values().cloned().collect();
        self.store.write_evidence(&records).await?;
        state.evidence = next;
        Ok(true)
    }

    /// Normalizes and ingests one raw trace payload.
    ///
    /// # Errors
    ///
    /// Same as [`FlowEngine::ingest`], plus the normalization errors from
    /// [`FlowTrace::from_value`].
    pub async fn ingest_value(&self, value: Value) -> Result<bool> {
        self.ingest(FlowTrace::from_value(value)?).await
    }

    /// Returns the stored evidence, sorted by trace id.
    pub async fn evidence(&self) -> Vec<FlowTrace> {
        self.state.read().await.evidence.values().cloned().collect()
    }

    /// Distills every successful trace into versioned recipes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`](openjiuwen_common::error::Error::external)
    /// when persisting a new recipe version fails. A model failure never fails
    /// distillation; it is reported in
    /// [`DistillReport::narrative_fallbacks`].
    pub async fn distill(&self) -> Result<DistillReport> {
        let _guard = self.distill_guard.lock().await;
        let snapshot: Vec<FlowTrace> = {
            let state = self.state.read().await;
            state.evidence.values().cloned().collect()
        };
        let records: Vec<FlowTrace> = snapshot
            .into_iter()
            .filter(|trace| trace.outcome.is_success())
            .collect();
        let groups = group_by_structure(&records);
        let grouped: BTreeSet<&str> = groups
            .values()
            .flat_map(|group| group.records.iter().map(|trace| trace.trace_id.as_str()))
            .collect();
        let mut report = DistillReport {
            evidence_total: records.len(),
            groups_total: groups.len(),
            groups_without_structure: records
                .iter()
                .filter(|trace| !grouped.contains(trace.trace_id.as_str()))
                .map(|trace| trace.trace_id.clone())
                .collect(),
            ..DistillReport::default()
        };
        report.groups_without_structure.sort();
        for (recipe_id, group) in &groups {
            let existing = {
                let state = self.state.read().await;
                state.recipes.get(recipe_id).cloned()
            };
            let (built, fallback) = self.build_recipe(group, existing.as_ref()).await;
            if fallback {
                report.narrative_fallbacks.push(recipe_id.clone());
            }
            match built {
                Some(recipe) => {
                    self.store.save_recipe(&recipe).await?;
                    self.state
                        .write()
                        .await
                        .recipes
                        .insert(recipe_id.clone(), recipe);
                    report.recipes_saved.push(recipe_id.clone());
                }
                None => report.recipes_unchanged.push(recipe_id.clone()),
            }
        }
        report.recipes_saved.sort();
        report.recipes_unchanged.sort();
        report.narrative_fallbacks.sort();
        Ok(report)
    }

    /// Builds one recipe version from a structure group.
    ///
    /// Returns the recipe only when its content identity changed, plus whether a
    /// configured model fell back to the deterministic template.
    async fn build_recipe(
        &self,
        group: &StructureGroup,
        existing: Option<&ExperienceRecipe>,
    ) -> (Option<ExperienceRecipe>, bool) {
        let nodes = merge_nodes(&group.records);
        let structure = build_structure(&group.edges, &nodes);
        let order = topological_order(&structure);
        let source_queries = source_queries(&group.records);
        let template = template_texts(&group.records, &order, &nodes);
        let (texts, fallback) = self
            .distill_texts(&group.records, &order, &nodes, template, &source_queries)
            .await;
        let quality = compute_quality(&group.records, group.edges.len());
        let (status, grade) = resolve_status_grade(&quality);
        let version = existing.map_or(1, |recipe| recipe.version + 1);
        let mut provenance = provenance(group, texts.source);
        sanitize_provenance(&mut provenance);
        let recipe = ExperienceRecipe {
            schema_version: RECIPE_SCHEMA_VERSION.to_string(),
            recipe_id: format!("recipe_{}", group.signature),
            name: texts.name,
            version,
            status,
            grade,
            applicability: texts.applicability,
            combination_structure: structure,
            execution_narrative: texts.narrative,
            quality,
            provenance,
        };
        if existing.is_some_and(|previous| previous.content_identity() == recipe.content_identity())
        {
            return (None, fallback);
        }
        (Some(recipe), fallback)
    }

    /// Returns the current recipe versions, sorted by recipe id.
    pub async fn recipes(&self) -> Vec<ExperienceRecipe> {
        self.state.read().await.recipes.values().cloned().collect()
    }

    /// Returns the current version of one recipe.
    pub async fn recipe(&self, recipe_id: &str) -> Option<ExperienceRecipe> {
        self.state.read().await.recipes.get(recipe_id).cloned()
    }

    /// Returns verified capability-combination display candidates.
    pub async fn candidates(&self) -> Vec<CombinationCandidate> {
        self.state
            .read()
            .await
            .recipes
            .values()
            .filter(|recipe| recipe.is_active_verified())
            .map(ExperienceRecipe::candidate)
            .collect()
    }

    /// Returns a frozen package by id.
    pub async fn package(&self, package_id: &str) -> Option<SkillPack> {
        self.state.read().await.packages.get(package_id).cloned()
    }

    /// Closes the engine. Every write is already durable, so this is a no-op.
    ///
    /// # Errors
    ///
    /// This implementation never fails; the `Result` keeps room for a future
    /// flush.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn close(&self) -> Result<()> {
        Ok(())
    }

    /// Persists a package and records it in the in-memory catalog.
    pub(crate) async fn persist_package(&self, pack: &SkillPack) -> Result<()> {
        self.store.save_package(pack).await?;
        self.state
            .write()
            .await
            .packages
            .insert(pack.package_id.clone(), pack.clone());
        Ok(())
    }

    /// Distils narrative texts, falling back to `template` on any model failure.
    async fn distill_texts(
        &self,
        records: &[FlowTrace],
        order: &[String],
        nodes: &BTreeMap<String, TraceNode>,
        template: RecipeTexts,
        source_queries: &[String],
    ) -> (RecipeTexts, bool) {
        let Some(model) = self.model.as_ref() else {
            return (template, false);
        };
        let prompt = narrative_prompt(records, order, nodes);
        match model.complete(&prompt).await {
            Ok(reply) => {
                if let Some(texts) = parse_model_texts(&reply, &template, source_queries) {
                    (texts, false)
                } else {
                    warn!(
                        model = model.name(),
                        "flow narrative fallback: reply has no usable narrative"
                    );
                    (template, true)
                }
            }
            Err(error) => {
                warn!(
                    model = model.name(),
                    %error,
                    "flow narrative fallback: model call failed"
                );
                (template, true)
            }
        }
    }
}

/// Groups successful traces by their cycle-free successful-edge signature.
fn group_by_structure(records: &[FlowTrace]) -> BTreeMap<String, StructureGroup> {
    let mut groups: BTreeMap<String, StructureGroup> = BTreeMap::new();
    for record in records {
        let successful = record.successful_edges();
        if successful.is_empty() {
            continue;
        }
        let edges = break_cycles(successful);
        if edges.is_empty() {
            continue;
        }
        let serialized = edges
            .iter()
            .map(|(source, target, relation)| format!("{source}>{target}:{relation}"))
            .collect::<Vec<_>>()
            .join("|");
        let signature = sha256_short(&serialized, 12);
        let recipe_id = format!("recipe_{signature}");
        groups
            .entry(recipe_id)
            .or_insert_with(|| StructureGroup {
                signature,
                edges,
                records: Vec::new(),
            })
            .records
            .push(record.clone());
    }
    for group in groups.values_mut() {
        group
            .records
            .sort_by(|left, right| left.trace_id.cmp(&right.trace_id));
    }
    groups
}

/// Removes one deterministically chosen edge per directed cycle.
///
/// Every edge in a single trace has support 1, so Python's
/// `(support, edge)` ordering reduces to the lexicographic edge order: the
/// smallest edge that still lies on a cycle is dropped until the graph is
/// acyclic.
fn break_cycles(edges: Vec<(String, String, String)>) -> Vec<(String, String, String)> {
    let mut kept = edges;
    while has_cycle(&kept) {
        let candidate = kept
            .iter()
            .filter(|edge| edge_on_cycle(edge, &kept))
            .min()
            .cloned();
        let Some(candidate) = candidate else {
            break;
        };
        kept.retain(|edge| edge != &candidate);
    }
    kept
}

/// Reports whether the directed edge list contains a cycle.
fn has_cycle(edges: &[(String, String, String)]) -> bool {
    let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (source, target, _) in edges {
        adjacency
            .entry(source.clone())
            .or_default()
            .push(target.clone());
    }
    let mut visiting: BTreeSet<String> = BTreeSet::new();
    let mut done: BTreeSet<String> = BTreeSet::new();
    for node in adjacency.keys() {
        if visit_cycle(node, &adjacency, &mut visiting, &mut done) {
            return true;
        }
    }
    false
}

/// Depth-first cycle visit used by [`has_cycle`].
fn visit_cycle(
    node: &str,
    adjacency: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    done: &mut BTreeSet<String>,
) -> bool {
    if visiting.contains(node) {
        return true;
    }
    if done.contains(node) {
        return false;
    }
    visiting.insert(node.to_string());
    if let Some(targets) = adjacency.get(node) {
        for target in targets {
            if visit_cycle(target, adjacency, visiting, done) {
                return true;
            }
        }
    }
    visiting.remove(node);
    done.insert(node.to_string());
    false
}

/// Reports whether `edge` lies on a directed cycle in `edges`.
fn edge_on_cycle(edge: &(String, String, String), edges: &[(String, String, String)]) -> bool {
    reachable(&edge.1, &edge.0, edges)
}

/// Reports whether `target` is reachable from `start`.
fn reachable(start: &str, target: &str, edges: &[(String, String, String)]) -> bool {
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (source, destination, _) in edges {
        adjacency
            .entry(source.as_str())
            .or_default()
            .push(destination.as_str());
    }
    let mut pending = vec![start];
    let mut visited: BTreeSet<&str> = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if node == target {
            return true;
        }
        if !visited.insert(node) {
            continue;
        }
        if let Some(next) = adjacency.get(node) {
            pending.extend(next.iter().copied());
        }
    }
    false
}

/// Merges nodes from the group's traces, first non-empty value winning.
///
/// Every trace-derived free-text field is redacted here, once, at the boundary
/// where it enters the recipe (and later the `SkillPack`): node descriptions go
/// through [`sanitize_distilled_text`] and IO port name/type/description through
/// [`redact_sensitive_text`].
fn merge_nodes(records: &[FlowTrace]) -> BTreeMap<String, TraceNode> {
    let mut merged: BTreeMap<String, TraceNode> = BTreeMap::new();
    for record in records {
        for (id, node) in &record.graph.nodes {
            let entry = merged.entry(id.clone()).or_insert_with(|| node.clone());
            if entry.skill_type.is_empty() {
                entry.skill_type.clone_from(&node.skill_type);
            }
            if entry.version.is_empty() {
                entry.version.clone_from(&node.version);
            }
            if entry.description.is_empty() {
                entry.description.clone_from(&node.description);
            }
            if entry.inputs.is_empty() {
                entry.inputs.clone_from(&node.inputs);
            }
            if entry.outputs.is_empty() {
                entry.outputs.clone_from(&node.outputs);
            }
        }
    }
    for node in merged.values_mut() {
        sanitize_node(node);
    }
    merged
}

/// Redacts one merged trace node's model-facing text in place.
fn sanitize_node(node: &mut TraceNode) {
    node.description = sanitize_distilled_text(&node.description, None);
    for port in node.inputs.iter_mut().chain(node.outputs.iter_mut()) {
        port.name = redact_sensitive_text(&port.name);
        port.io_type = redact_sensitive_text(&port.io_type);
        port.description = redact_sensitive_text(&port.description);
    }
}

/// Returns the group's non-empty source queries, in record order.
fn source_queries(records: &[FlowTrace]) -> Vec<String> {
    records
        .iter()
        .map(|record| record.query.trim().to_string())
        .filter(|query| !query.is_empty())
        .collect()
}

/// Builds the recipe combination structure from the group's edges.
fn build_structure(
    edges: &[(String, String, String)],
    nodes: &BTreeMap<String, TraceNode>,
) -> CombinationStructure {
    let mut members: BTreeSet<String> = BTreeSet::new();
    for (source, target, _) in edges {
        members.insert(source.clone());
        members.insert(target.clone());
    }
    let mut values = BTreeMap::new();
    for id in members {
        let node = nodes.get(&id).cloned().unwrap_or_else(|| TraceNode {
            capability_id: id.clone(),
            ..TraceNode::default()
        });
        values.insert(id, node.to_combination_value());
    }
    CombinationStructure {
        structure_type: SKILL_PACK_TYPE.to_string(),
        direction: "directed".to_string(),
        nodes: values,
        edges: edges
            .iter()
            .map(|(source, target, relation)| RecipeEdge {
                source: source.clone(),
                target: target.clone(),
                relation: relation.clone(),
            })
            .collect(),
        loop_guards: Vec::new(),
    }
}

/// Computes the quality statistics of one group.
#[allow(clippy::cast_precision_loss)]
fn compute_quality(records: &[FlowTrace], qualified_edge_count: usize) -> RecipeQuality {
    let execution_count = records.len() as u64;
    let success_count = records
        .iter()
        .filter(|record| record.outcome.is_success())
        .count() as u64;
    let pack_success_rate = if execution_count == 0 {
        0.0
    } else {
        let rate = success_count as f64 / execution_count as f64;
        (rate * 10_000.0).round() / 10_000.0
    };
    RecipeQuality {
        execution_count,
        success_count,
        failure_count: execution_count.saturating_sub(success_count),
        pack_success_rate,
        qualified_edge_count,
    }
}

/// Resolves the recipe status and grade from its quality.
fn resolve_status_grade(quality: &RecipeQuality) -> (String, String) {
    if quality.success_count >= MIN_SUCCESSES && quality.pack_success_rate >= MIN_PACK_SUCCESS_RATE
    {
        (
            RECIPE_STATUS_ACTIVE.to_string(),
            RECIPE_GRADE_VERIFIED.to_string(),
        )
    } else {
        (
            RECIPE_STATUS_DRAFT.to_string(),
            RECIPE_GRADE_CANDIDATE.to_string(),
        )
    }
}

/// Builds the deterministic provenance metadata.
///
/// `sample_queries` are trace-derived free text, so they carry the sanitized
/// queries rather than the raw ones; [`sanitize_provenance`] then runs the JSON
/// redactor over the whole blob, which is the one raw metadata structure a
/// recipe embeds.
fn provenance(group: &StructureGroup, narrative_source: &str) -> BTreeMap<String, Value> {
    let trace_ids: Vec<Value> = group
        .records
        .iter()
        .map(|record| Value::String(record.trace_id.clone()))
        .collect();
    let sample_queries: Vec<Value> = distinct_queries(&group.records, 5)
        .into_iter()
        .map(Value::String)
        .collect();
    let mut provenance = BTreeMap::new();
    provenance.insert(
        "source".to_string(),
        Value::String("flow_distillation".to_string()),
    );
    provenance.insert(
        "structure_signature".to_string(),
        Value::String(group.signature.clone()),
    );
    provenance.insert("evidence_trace_ids".to_string(), Value::Array(trace_ids));
    provenance.insert(
        "evidence_count".to_string(),
        serde_json::json!(group.records.len()),
    );
    provenance.insert("sample_queries".to_string(), Value::Array(sample_queries));
    provenance.insert("distilled_at".to_string(), Value::String(now_rfc3339()));
    provenance.insert(
        "narrative_source".to_string(),
        Value::String(narrative_source.to_string()),
    );
    provenance
}

/// Redacts credential material from the recipe's provenance metadata blob.
fn sanitize_provenance(provenance: &mut BTreeMap<String, Value>) {
    let blob: serde_json::Map<String, Value> = provenance.clone().into_iter().collect();
    if let Value::Object(redacted) = redact_sensitive_json(&Value::Object(blob)) {
        provenance.clear();
        provenance.extend(redacted);
    }
}

/// Builds the deterministic template texts for one group.
fn template_texts(
    records: &[FlowTrace],
    order: &[String],
    nodes: &BTreeMap<String, TraceNode>,
) -> RecipeTexts {
    let task_description = format!(
        "基于 {} 个能力协作完成的任务（{}）",
        order.len(),
        order.join(" → ")
    );
    let trigger_conditions =
        "用户请求需要多个检索、整理或生成能力接力完成，且任务形态与历史成功执行相似。";
    let examples: Vec<Value> = distinct_queries(records, 3)
        .into_iter()
        .map(|query| Value::String(format!("请帮我完成类似任务：{query}")))
        .collect();
    let steps = order
        .iter()
        .map(|id| {
            let description = nodes
                .get(id)
                .map(|node| node.description.trim())
                .filter(|description| !description.is_empty())
                .unwrap_or("协作成员能力");
            format!("由能力 {id}（{description}）接力执行")
        })
        .collect::<Vec<_>>()
        .join("；");
    let mut applicability = BTreeMap::new();
    applicability.insert(
        "task_description".to_string(),
        Value::String(task_description),
    );
    applicability.insert(
        "trigger_conditions".to_string(),
        Value::String(trigger_conditions.to_string()),
    );
    applicability.insert("example_requests".to_string(), Value::Array(examples));
    RecipeTexts {
        name: normalize_meta_name(
            order.last().map_or("skill-pack", String::as_str),
            "skill-pack",
        ),
        applicability,
        narrative: format!("执行过程：{steps}。"),
        source: "template",
    }
}

/// Returns up to `limit` distinct non-empty queries in record order.
///
/// Queries are trace-derived free text, so each one is sanitized before it is
/// used: this is the single boundary for the template example requests, the
/// narrative prompt material, and the recipe provenance.
fn distinct_queries(records: &[FlowTrace], limit: usize) -> Vec<String> {
    let mut queries = Vec::new();
    for record in records {
        let query = sanitize_distilled_text(&record.query, None);
        if query.is_empty() || queries.iter().any(|existing| existing == &query) {
            continue;
        }
        queries.push(query);
        if queries.len() == limit {
            break;
        }
    }
    queries
}

/// Builds the narrative prompt sent to an optional completion model.
fn narrative_prompt(
    records: &[FlowTrace],
    order: &[String],
    nodes: &BTreeMap<String, TraceNode>,
) -> String {
    let members: Vec<Value> = order
        .iter()
        .map(|id| {
            let description = nodes
                .get(id)
                .map(|node| node.description.trim())
                .filter(|description| !description.is_empty())
                .unwrap_or("协作成员能力");
            serde_json::json!({"id": id, "description": description})
        })
        .collect();
    let material = serde_json::json!({
        "queries": distinct_queries(records, 6),
        "steps": order,
        "members": members,
    });
    let mut prompt = String::from(
        "你是编排经验蒸馏助手。根据历史任务请求和已验证的能力组合结构，归纳任务类型描述、\
         触发条件、示例请求与执行过程说明。只依据给定材料，不得虚构能力或步骤。只输出一个 \
         JSON 对象，字段为 {\"name\": str, \"task_description\": str, \
         \"trigger_conditions\": str, \"example_requests\": [str], \
         \"execution_narrative\": str}。材料：",
    );
    prompt.push_str(&material.to_string());
    prompt
}

/// Parses a model reply into texts, returning `None` when it is unusable.
///
/// This is the one boundary where model-written prose enters a recipe: every
/// accepted field is sanitized against the group's source queries, so a model
/// that echoes a credential or a verbatim request cannot leak it into the
/// distilled artifact.
fn parse_model_texts(
    reply: &str,
    template: &RecipeTexts,
    source_queries: &[String],
) -> Option<RecipeTexts> {
    let object_text = extract_json_object(reply)?;
    let value: Value = serde_json::from_str(object_text).ok()?;
    let object = value.as_object()?;
    let narrative = object
        .get("execution_narrative")
        .and_then(Value::as_str)?
        .trim();
    if narrative.is_empty() {
        return None;
    }
    let mut texts = template.clone();
    texts.narrative = sanitize_with_queries(narrative, source_queries);
    texts.source = "llm";
    if let Some(description) = non_empty_string(object.get("task_description")) {
        texts.applicability.insert(
            "task_description".to_string(),
            Value::String(sanitize_with_queries(description, source_queries)),
        );
    }
    if let Some(trigger) = non_empty_string(object.get("trigger_conditions")) {
        texts.applicability.insert(
            "trigger_conditions".to_string(),
            Value::String(sanitize_with_queries(trigger, source_queries)),
        );
    }
    if let Some(examples) = object.get("example_requests").and_then(Value::as_array) {
        let examples: Vec<Value> = examples
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|example| !example.is_empty())
            .take(5)
            .filter_map(|example| {
                let sanitized = sanitize_with_queries(example, source_queries);
                (!sanitized.is_empty()).then_some(Value::String(sanitized))
            })
            .collect();
        if !examples.is_empty() {
            texts
                .applicability
                .insert("example_requests".to_string(), Value::Array(examples));
        }
    }
    if let Some(name) = non_empty_string(object.get("name")) {
        let name = sanitize_with_queries(name, source_queries);
        if !name.contains('→') && !name.contains("->") {
            texts.name = normalize_meta_name(&name, &template.name);
        }
    }
    Some(texts)
}

/// Returns a trimmed non-empty string from an optional JSON value.
fn non_empty_string(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::flow::model::CAN_FEED;

    fn trace(trace_id: &str, edges: &[(&str, &str)], query: &str) -> FlowTrace {
        let mut nodes = BTreeMap::new();
        for id in edges.iter().flat_map(|(source, target)| [*source, *target]) {
            nodes.entry(id.to_string()).or_insert_with(|| TraceNode {
                capability_id: id.to_string(),
                ..TraceNode::default()
            });
        }
        FlowTrace {
            trace_id: trace_id.to_string(),
            query: query.to_string(),
            outcome: crate::flow::model::TraceOutcome::Success,
            graph: crate::flow::model::ExecutionGraph {
                nodes,
                edges: edges
                    .iter()
                    .map(|(source, target)| crate::flow::model::TraceEdge {
                        source: (*source).to_string(),
                        target: (*target).to_string(),
                        relation: CAN_FEED.to_string(),
                        success: true,
                        reason: String::new(),
                    })
                    .collect(),
                ..crate::flow::model::ExecutionGraph::default()
            },
        }
    }

    #[test]
    fn cycles_are_broken_deterministically() {
        let edges = vec![
            ("a".to_string(), "b".to_string(), CAN_FEED.to_string()),
            ("b".to_string(), "a".to_string(), CAN_FEED.to_string()),
            ("b".to_string(), "c".to_string(), CAN_FEED.to_string()),
        ];
        let kept = break_cycles(edges);
        assert!(!has_cycle(&kept));
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&("b".to_string(), "c".to_string(), CAN_FEED.to_string())));
    }

    #[test]
    fn quality_rounds_and_resolves_status() {
        let records = vec![
            trace("t1", &[("a", "b")], "q1"),
            trace("t2", &[("a", "b")], "q2"),
        ];
        let quality = compute_quality(&records, 1);
        assert_eq!(quality.execution_count, 2);
        assert_eq!(quality.success_count, 2);
        assert_eq!(quality.failure_count, 0);
        assert!((quality.pack_success_rate - 1.0).abs() < f64::EPSILON);
        let (status, grade) = resolve_status_grade(&quality);
        assert_eq!(status, RECIPE_STATUS_ACTIVE);
        assert_eq!(grade, RECIPE_GRADE_VERIFIED);
    }

    #[test]
    fn model_texts_fall_back_when_unusable() {
        let template = template_texts(
            &[trace("t1", &[("a", "b")], "q1")],
            &["a".to_string(), "b".to_string()],
            &BTreeMap::new(),
        );
        assert!(parse_model_texts("no json", &template, &[]).is_none());
        assert!(parse_model_texts("{\"name\": \"x\"}", &template, &[]).is_none());
        let parsed = parse_model_texts(
            "```json\n{\"execution_narrative\": \"模型叙述\"}\n```",
            &template,
            &[],
        )
        .expect("parses");
        assert_eq!(parsed.narrative, "模型叙述");
        assert_eq!(parsed.source, "llm");
    }
}
