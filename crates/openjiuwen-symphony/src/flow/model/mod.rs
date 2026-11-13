//! Data models for flow evidence, distilled recipes, and display candidates.
//!
//! Ports the dataclasses in Python `openjiuwen.symphony.flow.models` and the
//! execution-graph normalization in `flow.distill`. The wire shape follows the
//! Python contract: graph objects use the JSON key `"type"`, capability ports
//! reuse [`CapabilityIo`](crate::model::CapabilityIo) (whose type key is also
//! `"type"`), and all collections default to empty.
//!
//! Normalization (`FlowTrace::from_value`) is intentionally forgiving but
//! deterministic: unknown fields are ignored, leading `skill:`/`capability:`
//! prefixes are stripped, nodes without an id or object body are dropped, and an
//! edge survives only when its `metadata.success` is truthy and both endpoints
//! exist after normalization. The surviving edges are sorted by
//! `(source, target, relation)`, so a stored trace is byte-stable.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

mod trace;
pub use trace::*;

/// Schema version written into every distilled [`ExperienceRecipe`].
pub const RECIPE_SCHEMA_VERSION: &str = "symphony.experience_recipe.v1";
/// Schema version written into every [`SkillPack`](crate::flow::pack::SkillPack).
pub const PACKAGE_SCHEMA_VERSION: &str = "symphony.capability_package.v1";
/// The structure type of a distilled recipe's capability combination.
pub const SKILL_PACK_TYPE: &str = "skill_pack";
/// The default edge relation between two collaborating capabilities.
pub const CAN_FEED: &str = "can_feed";
/// A recipe that has not yet met the verification thresholds.
pub const RECIPE_STATUS_DRAFT: &str = "draft";
/// A recipe that met the verification thresholds and may be packaged.
pub const RECIPE_STATUS_ACTIVE: &str = "active";
/// A recipe whose quality is still below the verification thresholds.
pub const RECIPE_GRADE_CANDIDATE: &str = "candidate";
/// A recipe whose quality met every verification threshold.
pub const RECIPE_GRADE_VERIFIED: &str = "verified";
/// Minimum successful executions before a recipe is verified.
pub const MIN_SUCCESSES: u64 = 1;
/// Minimum pack success rate before a recipe is verified.
pub const MIN_PACK_SUCCESS_RATE: f64 = 0.8;

/// Returns the lowercase hex sha256 digest of `text`.
#[must_use]
pub fn sha256_hex(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let mut rendered = String::with_capacity(64);
    for byte in digest {
        rendered.push(HEX[usize::from(byte >> 4)] as char);
        rendered.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    rendered
}

/// Returns the first `length` hex characters of the sha256 digest of `text`.
///
/// A `length` of 64 or more returns the full digest; an ASCII hex slice never
/// splits a character.
#[must_use]
pub fn sha256_short(text: &str, length: usize) -> String {
    let digest = sha256_hex(text);
    digest.chars().take(length).collect()
}

/// Renders a JSON value as canonical JSON: object keys sorted, no whitespace.
///
/// The crate's `serde_json` build preserves insertion order, so this helper is
/// the only place where a stable content hash can be derived from a value.
#[must_use]
pub fn canonical_json(value: &Value) -> String {
    let mut rendered = String::new();
    write_canonical(value, &mut rendered);
    rendered
}

/// Appends the canonical rendering of `value` to `output`.
fn write_canonical(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(flag) => output.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::String(text) => {
            let encoded = serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string());
            output.push_str(&encoded);
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical(item, output);
            }
            output.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            output.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                let encoded = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
                output.push_str(&encoded);
                output.push(':');
                if let Some(item) = map.get(key) {
                    write_canonical(item, output);
                }
            }
            output.push('}');
        }
    }
}

/// Returns the full sha256 hex digest of the canonical JSON of `value`.
#[must_use]
pub fn content_hash(value: &Value) -> String {
    sha256_hex(&canonical_json(value))
}

/// Strips a leading `skill:`/`capability:` namespace from a graph node id.
///
/// The prefixes are removed repeatedly (Python uses two sequential
/// `removeprefix` calls) and the result is trimmed.
#[must_use]
pub fn normalize_capability_id(raw: &str) -> String {
    let mut id = raw.trim();
    loop {
        if let Some(rest) = id.strip_prefix("skill:") {
            id = rest;
        } else if let Some(rest) = id.strip_prefix("capability:") {
            id = rest;
        } else {
            break;
        }
    }
    id.trim().to_string()
}

/// Normalizes an arbitrary label into a valid `SkillPack` `meta.name`.
///
/// The result matches `^[a-z0-9][a-z0-9-]{0,63}$`: lowercase ASCII
/// alphanumerics separated by single hyphens, at most 64 characters. An empty
/// slug falls back to `fallback` (or `combo` when that is blank too).
#[must_use]
pub fn normalize_meta_name(raw: &str, fallback: &str) -> String {
    let lowered = raw.trim().to_lowercase();
    let mut slug = String::with_capacity(lowered.len());
    for character in lowered.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character);
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let trimmed = slug.trim_matches('-').to_string();
    let fallback = {
        let candidate = fallback.trim();
        if candidate.is_empty() {
            "combo".to_string()
        } else {
            candidate.to_string()
        }
    };
    let mut name = if trimmed.is_empty() {
        fallback.clone()
    } else {
        trimmed
    };
    if !name
        .chars()
        .next()
        .is_some_and(|first| first.is_ascii_lowercase() || first.is_ascii_digit())
    {
        name = format!("{fallback}-{name}");
    }
    name.truncate(64);
    while name.ends_with('-') {
        name.pop();
    }
    if name.is_empty() {
        "combo".to_string()
    } else {
        name
    }
}

/// Reports whether `name` is a valid `SkillPack` `meta.name`.
#[must_use]
pub fn is_valid_meta_name(name: &str) -> bool {
    let mut characters = name.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    name.len() <= 64
        && characters.all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
        })
}

/// Quality statistics for one distilled recipe version.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecipeQuality {
    /// Number of traces in the recipe's structure group.
    pub execution_count: u64,
    /// Number of fully successful traces in the group.
    pub success_count: u64,
    /// Number of non-successful traces in the group.
    pub failure_count: u64,
    /// `success_count / execution_count`, rounded to four decimals.
    pub pack_success_rate: f64,
    /// Number of qualified collaboration edges in the structure.
    pub qualified_edge_count: usize,
}

/// One directed dependency edge of a distilled recipe.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecipeEdge {
    /// The upstream capability id.
    pub source: String,
    /// The downstream capability id.
    pub target: String,
    /// The dependency relation; defaults to `can_feed`.
    pub relation: String,
}

/// The capability combination frozen into a recipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CombinationStructure {
    /// The structure type (`skill_pack`); the JSON key is `"type"`.
    #[serde(rename = "type", alias = "structure_type")]
    pub structure_type: String,
    /// The edge direction (`directed`).
    pub direction: String,
    /// Recipe nodes keyed by capability id.
    pub nodes: BTreeMap<String, Value>,
    /// The sorted dependency edges.
    pub edges: Vec<RecipeEdge>,
    /// Loop guards; always empty in the deterministic v1 flow.
    pub loop_guards: Vec<Value>,
}

impl Default for CombinationStructure {
    fn default() -> Self {
        Self {
            structure_type: SKILL_PACK_TYPE.to_string(),
            direction: "directed".to_string(),
            nodes: BTreeMap::new(),
            edges: Vec::new(),
            loop_guards: Vec::new(),
        }
    }
}

/// One immutable, versioned distilled recipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceRecipe {
    /// The recipe schema version.
    pub schema_version: String,
    /// The stable recipe id derived from the edge signature.
    pub recipe_id: String,
    /// The installable display name.
    pub name: String,
    /// The immutable version, starting at 1.
    pub version: u64,
    /// `draft` or `active`.
    pub status: String,
    /// `candidate` or `verified`.
    pub grade: String,
    /// Applicability metadata (`task_description`, `trigger_conditions`,
    /// `example_requests`).
    pub applicability: BTreeMap<String, Value>,
    /// The capability combination.
    pub combination_structure: CombinationStructure,
    /// The deterministic or model-written execution narrative.
    pub execution_narrative: String,
    /// Quality statistics at version-creation time.
    pub quality: RecipeQuality,
    /// Provenance metadata (source, signature, evidence ids, timestamps).
    pub provenance: BTreeMap<String, Value>,
}

impl Default for ExperienceRecipe {
    fn default() -> Self {
        Self {
            schema_version: RECIPE_SCHEMA_VERSION.to_string(),
            recipe_id: String::new(),
            name: String::new(),
            version: 1,
            status: RECIPE_STATUS_DRAFT.to_string(),
            grade: RECIPE_GRADE_CANDIDATE.to_string(),
            applicability: BTreeMap::new(),
            combination_structure: CombinationStructure::default(),
            execution_narrative: String::new(),
            quality: RecipeQuality::default(),
            provenance: BTreeMap::new(),
        }
    }
}

impl ExperienceRecipe {
    /// Returns the sorted capability ids of the combination.
    #[must_use]
    pub fn capability_ids(&self) -> Vec<String> {
        self.combination_structure.nodes.keys().cloned().collect()
    }

    /// Returns a string value from the applicability map.
    #[must_use]
    pub fn applicability_text(&self, key: &str) -> &str {
        self.applicability
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// Returns the content fingerprint used for change detection.
    ///
    /// Only content participates: `name`, `applicability`, the structure
    /// (`type`, `direction`, `nodes`, `edges`, `loop_guards`), and the
    /// execution narrative. Quality counts, versions, and provenance are
    /// excluded, so replaying the same evidence keeps the same version.
    #[must_use]
    pub fn content_identity(&self) -> String {
        let identity = serde_json::json!({
            "name": self.name,
            "applicability": self.applicability,
            "combination_structure": {
                "type": self.combination_structure.structure_type,
                "direction": self.combination_structure.direction,
                "nodes": self.combination_structure.nodes,
                "edges": self.combination_structure.edges,
                "loop_guards": self.combination_structure.loop_guards,
            },
            "execution_narrative": self.execution_narrative,
        });
        content_hash(&identity)
    }

    /// Reports whether the recipe is packable (`active` + `verified`).
    #[must_use]
    pub fn is_active_verified(&self) -> bool {
        self.status == RECIPE_STATUS_ACTIVE && self.grade == RECIPE_GRADE_VERIFIED
    }

    /// Builds the privacy-safe display snapshot for this recipe.
    #[must_use]
    pub fn candidate(&self) -> CombinationCandidate {
        CombinationCandidate {
            recipe_id: self.recipe_id.clone(),
            version: self.version,
            name: self.name.clone(),
            applicability: self.applicability_text("task_description").to_string(),
            structure: self.combination_structure.edges.clone(),
            execution_count: self.quality.execution_count,
            success_count: self.quality.success_count,
            success_rate: self.quality.pack_success_rate,
        }
    }
}

/// A stable display snapshot of one verified capability combination.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CombinationCandidate {
    /// The recipe id.
    pub recipe_id: String,
    /// The recipe version.
    pub version: u64,
    /// The installable display name.
    pub name: String,
    /// The task description.
    pub applicability: String,
    /// The combination's dependency edges.
    pub structure: Vec<RecipeEdge>,
    /// Number of traced executions.
    pub execution_count: u64,
    /// Number of successful executions.
    pub success_count: u64,
    /// The pack success rate.
    pub success_rate: f64,
}

/// Returns the deterministic Kahn topological order of a recipe structure.
///
/// Ready nodes are emitted in sorted id order, so an acyclic structure always
/// yields the same order. A cycle leaves the remaining nodes out of the result;
/// callers that require a complete order validate the chain first.
#[must_use]
pub fn topological_order(structure: &CombinationStructure) -> Vec<String> {
    let mut indegree: BTreeMap<String, usize> =
        structure.nodes.keys().map(|id| (id.clone(), 0)).collect();
    let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for edge in &structure.edges {
        if indegree.contains_key(&edge.source) && indegree.contains_key(&edge.target) {
            adjacency
                .entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
            if let Some(degree) = indegree.get_mut(&edge.target) {
                *degree += 1;
            }
        }
    }
    let mut ready: Vec<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    ready.sort();
    let mut order = Vec::with_capacity(indegree.len());
    while !ready.is_empty() {
        let id = ready.remove(0);
        order.push(id.clone());
        if let Some(targets) = adjacency.get(&id) {
            let mut targets = targets.clone();
            targets.sort();
            for target in targets {
                if let Some(degree) = indegree.get_mut(&target) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.push(target);
                    }
                }
            }
        }
        ready.sort();
    }
    order
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_keys_and_hashes_stably() {
        let first = serde_json::json!({"b": 1, "a": [2, {"d": true, "c": null}]});
        let second = serde_json::json!({"a": [2, {"c": null, "d": true}], "b": 1});
        assert_eq!(canonical_json(&first), canonical_json(&second));
        assert_eq!(content_hash(&first), content_hash(&second));
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(sha256_short("abc", 4).len(), 4);
    }

    #[test]
    fn capability_ids_strip_repeated_prefixes() {
        assert_eq!(normalize_capability_id(" skill:search "), "search");
        assert_eq!(normalize_capability_id("skill:capability:sum"), "sum");
        assert_eq!(normalize_capability_id("plain"), "plain");
        assert_eq!(normalize_capability_id("skill:"), "");
    }

    #[test]
    fn meta_names_are_normalized_and_validated() {
        assert_eq!(
            normalize_meta_name("My  Skill_Pack!", "fallback"),
            "my-skill-pack"
        );
        assert_eq!(normalize_meta_name("", "fallback"), "fallback");
        assert!(is_valid_meta_name("my-skill-pack"));
        assert!(!is_valid_meta_name("My-Skill"));
        assert!(!is_valid_meta_name(""));
        assert!(!is_valid_meta_name(&"a".repeat(65)));
    }

    #[test]
    fn outcomes_accept_python_aliases() {
        assert!(TraceOutcome::from_raw("SUCCESS").is_success());
        assert!(TraceOutcome::from_raw("done").is_success());
        assert_eq!(TraceOutcome::from_raw("failure"), TraceOutcome::Failed);
        assert_eq!(TraceOutcome::from_raw("partially"), TraceOutcome::Partial);
        assert_eq!(TraceOutcome::from_raw("weird"), TraceOutcome::Unknown);
    }

    #[test]
    fn graph_normalization_strips_prefixes_and_filters_edges() {
        let payload = serde_json::json!({
            "trace_id": " t1 ",
            "query": " do it ",
            "outcome": "success",
            "graph": {
                "id": "g",
                "type": "execution_graph",
                "label": "run",
                "nodes": {
                    "skill:a": {"label": "capability", "metadata": {"capability_type": "skill"}},
                    "capability:b": {"label": "capability", "metadata": {"skill_type": "Skill"}},
                    "": {"label": "capability"}
                },
                "edges": [
                    {"source": "skill:a", "target": "capability:b", "metadata": {"success": true}},
                    {"source": "a", "target": "ghost", "metadata": {"success": true}},
                    {"source": "a", "target": "b", "metadata": {"success": false}}
                ]
            }
        });
        let trace = FlowTrace::from_value(payload).expect("normalizes");
        assert_eq!(trace.trace_id, "t1");
        assert_eq!(trace.query, "do it");
        assert_eq!(trace.graph.nodes.len(), 2);
        assert_eq!(trace.graph.edges.len(), 1);
        assert_eq!(trace.graph.edges[0].source, "a");
        assert_eq!(trace.graph.edges[0].target, "b");
        assert_eq!(trace.graph.edges[0].relation, CAN_FEED);
        assert_eq!(trace.graph.nodes["b"].skill_type, "skill");
        assert_eq!(trace.graph.nodes["b"].capability_type, "capability");
    }

    #[test]
    fn invalid_traces_are_rejected() {
        assert!(FlowTrace::from_value(serde_json::json!([])).is_err());
        assert!(FlowTrace::from_value(serde_json::json!({"graph": {}})).is_err());
        assert!(FlowTrace::from_value(serde_json::json!({"trace_id": "t"})).is_err());
        assert!(FlowTrace::from_value(serde_json::json!({"trace_id": "t", "graph": "x"})).is_err());
    }
}
