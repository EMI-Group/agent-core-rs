//! `SkillPack` artifacts: content-addressed, renderable, and loadable.
//!
//! Ports the Python `flow.packager` + `flow.render.SkillPackAdapter` subset: an
//! active, verified recipe whose combination is a simple linear chain is frozen
//! into a [`SkillPack`] identified by the sha256 of its canonical materials. The
//! pack renders to one `SKILL.md` (YAML front matter plus the documented body
//! sections) and one `dependencies.yaml` (the ordered capability dependencies and
//! the package header needed to reconstruct the pack).
//!
//! Everything is deterministic: the package id is content-addressed, the steps
//! follow the chain order, and the rendered artifacts do not include timestamps.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{validation, Error, Result};
use crate::flow::codegen::{
    extend_materials, generate_swarmflow_script, insert_script_artifact, read_script_artifact,
};
use crate::flow::engine::FlowEngine;
use crate::flow::model::{
    content_hash, is_valid_meta_name, normalize_meta_name, ExperienceRecipe, TraceNode,
    PACKAGE_SCHEMA_VERSION,
};
use crate::flow::store::atomic_write;
use crate::model::CapabilityIo;

/// The only supported artifact target kind.
pub const TARGET_KIND_SKILL: &str = "skill";
/// The default license recorded by a generated pack.
pub const DEFAULT_LICENSE: &str = "Proprietary";
/// The front-matter `kind` value of a `SkillPack` root document.
pub const SKILL_PACK_KIND: &str = "skillpack";
/// The file name of the rendered skill document.
pub const SKILL_MD_FILENAME: &str = "SKILL.md";
/// The file name of the rendered dependency manifest.
pub const DEPENDENCIES_FILENAME: &str = "dependencies.yaml";

/// One step of a rendered `SkillPack`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillPackStep {
    /// The capability id.
    pub id: String,
    /// The display name (the capability id in the deterministic v1 flow).
    pub name: String,
    /// The capability description.
    pub description: String,
    /// The capability kind (`skill` for ordinary skills).
    pub capability_type: String,
    /// The declared inputs.
    pub inputs: Vec<CapabilityIo>,
    /// The declared outputs.
    pub outputs: Vec<CapabilityIo>,
    /// The ids this step consumes; exactly one predecessor in a linear chain.
    pub depends_on: Vec<String>,
}

/// A frozen, content-addressed capability package.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillPack {
    /// The package schema version.
    pub schema_version: String,
    /// `cap-<sha256(materials)[:12]>`.
    pub package_id: String,
    /// The recipe this pack freezes.
    pub recipe_id: String,
    /// The recipe version this pack freezes.
    pub recipe_version: u64,
    /// The artifact target kind (`skill`).
    pub target_kind: String,
    /// The installable `meta.name` slug.
    pub name: String,
    /// The package description (`[技能包] ...`).
    pub description: String,
    /// The ordered chain steps.
    pub steps: Vec<SkillPackStep>,
    /// The rendered artifacts, keyed by file name.
    pub artifacts: BTreeMap<String, String>,
    /// The generated `SwarmFlow` script, when the recipe yielded one.
    ///
    /// The script is also rendered as the `swarmflow/run.py` artifact and enters
    /// the canonical materials under the `swarmflow_script` key, so it
    /// participates in the content-addressed package id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swarmflow_script: Option<String>,
    /// The recipe's applicability trigger conditions (empty when undeclared).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub trigger_conditions: String,
    /// Declared permissions; a generated pack declares none.
    pub permissions: Vec<String>,
    /// The declared license; a generated pack is `Proprietary`.
    pub license: String,
}

impl Default for SkillPack {
    fn default() -> Self {
        Self {
            schema_version: PACKAGE_SCHEMA_VERSION.to_string(),
            package_id: String::new(),
            recipe_id: String::new(),
            recipe_version: 1,
            target_kind: TARGET_KIND_SKILL.to_string(),
            name: String::new(),
            description: String::new(),
            steps: Vec::new(),
            artifacts: BTreeMap::new(),
            swarmflow_script: None,
            trigger_conditions: String::new(),
            permissions: Vec::new(),
            license: String::new(),
        }
    }
}

impl SkillPack {
    /// Returns the canonical materials object used for identity and integrity.
    ///
    /// The materials cover the package name, description, target kind, recipe
    /// id, and every ordered step; the package id, artifacts, permissions, and
    /// license are deliberately excluded. The generated `SwarmFlow` script and the
    /// recipe's trigger conditions join the materials only when present, so a
    /// pack without them keeps its previous content hash.
    #[must_use]
    pub fn materials(&self) -> Value {
        let steps = serde_json::to_value(&self.steps).unwrap_or_default();
        let mut materials = serde_json::json!({
            "recipe_id": self.recipe_id,
            "target_kind": self.target_kind,
            "name": self.name,
            "description": self.description,
            "steps": steps,
        });
        extend_materials(
            &mut materials,
            self.swarmflow_script.as_deref(),
            &self.trigger_conditions,
        );
        materials
    }

    /// Returns the sha256 hex digest of the canonical materials.
    #[must_use]
    pub fn materials_hash(&self) -> String {
        content_hash(&self.materials())
    }

    /// Returns the content-addressed package id for the current materials.
    #[must_use]
    pub fn compute_package_id(&self) -> String {
        format!("cap-{}", &self.materials_hash()[..12])
    }

    /// Recomputes the package id and rendered artifacts after a content change.
    ///
    /// Callers that intentionally edit a pack (for example a review test) use
    /// this to keep identity and artifacts consistent before reviewing.
    pub fn recompute_identity(&mut self) {
        self.package_id = self.compute_package_id();
        self.artifacts = self.render();
    }

    /// Returns the ordered step ids.
    #[must_use]
    pub fn step_ids(&self) -> Vec<&str> {
        self.steps.iter().map(|step| step.id.as_str()).collect()
    }

    /// Returns the task description without the `[技能包]` marker.
    #[must_use]
    pub fn task_description(&self) -> &str {
        self.description
            .strip_prefix("[技能包] ")
            .unwrap_or(&self.description)
    }

    /// Renders the `SKILL.md`, `dependencies.yaml`, and optional
    /// `swarmflow/run.py` artifacts.
    #[must_use]
    pub fn render(&self) -> BTreeMap<String, String> {
        let mut artifacts = BTreeMap::new();
        artifacts.insert(SKILL_MD_FILENAME.to_string(), self.render_skill_markdown());
        artifacts.insert(
            DEPENDENCIES_FILENAME.to_string(),
            self.render_dependencies_yaml(),
        );
        insert_script_artifact(&mut artifacts, self.swarmflow_script.as_deref());
        artifacts
    }

    /// Renders the `SKILL.md` root document.
    fn render_skill_markdown(&self) -> String {
        let front_matter = SkillFrontMatter {
            name: self.name.clone(),
            kind: SKILL_PACK_KIND.to_string(),
            description: self.description.clone(),
            skills: self.step_ids().into_iter().map(str::to_string).collect(),
        };
        let yaml = serde_yaml_ng::to_string(&front_matter).unwrap_or_default();
        let body = self.render_body();
        format!("---\n{}\n---\n\n{}", yaml.trim_end(), body)
    }

    /// Renders the documented body sections of `SKILL.md`.
    fn render_body(&self) -> String {
        let mut body = String::new();
        let _ = writeln!(body, "# {}\n", self.name);
        body.push_str("## When to use\n\n");
        body.push_str(self.task_description().trim());
        body.push_str("\n\n## Required inputs\n\n");
        for line in self.required_input_lines() {
            body.push_str(&line);
            body.push('\n');
        }
        body.push_str("\n## Included Skills\n\n");
        for step in &self.steps {
            let description = if step.description.trim().is_empty() {
                "具体用途见该成员的 SKILL.md。"
            } else {
                step.description.trim()
            };
            let _ = writeln!(body, "- `{}`：{}", step.id, description);
        }
        body.push_str("\n## Execution Process\n\n");
        for line in self.execution_process_lines() {
            body.push_str(&line);
            body.push('\n');
        }
        body.push_str("\n## Failure handling\n\n");
        body.push_str(
            "成员执行失败时停止依赖该结果的后续步骤，并返回已有结果、失败原因和未完成步骤。\n",
        );
        body.push_str("\n## Final output\n\n");
        for line in self.final_output_lines() {
            body.push_str(&line);
            body.push('\n');
        }
        body.push_str("\n## Workflow Graph\n\n```json\n");
        body.push_str(&self.render_workflow_graph());
        body.push_str("\n```\n");
        body
    }

    /// Renders the required-input list (entry capabilities only).
    fn required_input_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for step in &self.steps {
            for input in &step.inputs {
                if input.required != Some(true) {
                    continue;
                }
                let detail = if input.io_type.is_empty() {
                    String::new()
                } else {
                    format!("（{}）", input.io_type)
                };
                let suffix = if input.description.trim().is_empty() {
                    String::new()
                } else {
                    format!("：{}", input.description.trim())
                };
                lines.push(format!(
                    "- `{}.{}`{}{}",
                    step.id, input.name, detail, suffix
                ));
            }
        }
        if lines.is_empty() {
            lines.push("提供完整的用户目标；各成员未声明额外的必填输入。".to_string());
        }
        lines
    }

    /// Renders the numbered execution-process description.
    fn execution_process_lines(&self) -> Vec<String> {
        let mut lines = vec!["当前为线性流程，无可并行步骤。".to_string(), String::new()];
        for (index, step) in self.steps.iter().enumerate() {
            let input_text = if index == 0 {
                "接收用户原始目标和该成员声明的必要输入".to_string()
            } else {
                format!(
                    "接收用户原始目标，并将 `{}` 的完整输出作为上下文输入",
                    self.steps[index - 1].id
                )
            };
            let outputs: Vec<String> = step
                .outputs
                .iter()
                .filter(|output| !output.name.trim().is_empty())
                .map(|output| format!("`{}`", output.name.trim()))
                .collect();
            let output_text = if outputs.is_empty() {
                "本步骤结果".to_string()
            } else {
                outputs.join("、")
            };
            lines.push(format!(
                "{}. 调用 `{}`：{}；产出 {}。",
                index + 1,
                step.id,
                input_text,
                output_text
            ));
        }
        if let Some(last) = self.steps.last() {
            lines.push(String::new());
            lines.push(format!(
                "结果汇合：以 `{}` 的输出作为最终结果，前序结果作为上下文和依据。",
                last.id
            ));
        }
        lines
    }

    /// Renders the final-output description from the last step's outputs.
    fn final_output_lines(&self) -> Vec<String> {
        let Some(last) = self.steps.last() else {
            return vec!["返回最终结果，并说明任何失败、跳过或不完整部分。".to_string()];
        };
        if last.outputs.is_empty() {
            return vec![format!(
                "返回 `{}` 生成的最终结果，并说明任何失败、跳过或不完整部分。",
                last.id
            )];
        }
        let mut lines = vec![format!("返回 `{}` 生成的以下最终产物：", last.id)];
        for output in &last.outputs {
            let suffix = if output.description.trim().is_empty() {
                String::new()
            } else {
                format!("：{}", output.description.trim())
            };
            lines.push(format!("- `{}`{}", output.name, suffix));
        }
        lines.push("并说明任何失败、跳过或不完整部分。".to_string());
        lines
    }

    /// Renders the workflow-graph JSON document embedded in `SKILL.md`.
    fn render_workflow_graph(&self) -> String {
        let mut nodes = serde_json::Map::new();
        for step in &self.steps {
            nodes.insert(
                step.id.clone(),
                serde_json::json!({"label": step.id, "metadata": {"skill": step.id}}),
            );
        }
        let mut edges = Vec::new();
        for step in &self.steps {
            for dependency in &step.depends_on {
                edges.push(serde_json::json!({
                    "source": dependency,
                    "target": step.id,
                }));
            }
        }
        let graph = serde_json::json!({
            "graph": {
                "id": format!("{}-workflow", self.name),
                "type": "skillpack_workflow",
                "label": self.description,
                "directed": true,
                "nodes": nodes,
                "edges": edges,
            }
        });
        serde_json::to_string_pretty(&graph).unwrap_or_default()
    }

    /// Renders the `dependencies.yaml` manifest.
    fn render_dependencies_yaml(&self) -> String {
        let capabilities = self
            .steps
            .iter()
            .map(|step| DependencyEntry {
                id: step.id.clone(),
                version: self.recipe_version.to_string(),
                capability_type: step.capability_type.clone(),
                name: step.name.clone(),
                description: step.description.clone(),
                depends_on: step.depends_on.clone(),
                inputs: step.inputs.clone(),
                outputs: step.outputs.clone(),
            })
            .collect();
        let document = DependenciesDoc {
            capabilities,
            package: Some(PackageHeader {
                schema_version: self.schema_version.clone(),
                package_id: self.package_id.clone(),
                recipe_id: self.recipe_id.clone(),
                recipe_version: self.recipe_version,
                target_kind: self.target_kind.clone(),
                name: self.name.clone(),
                description: self.description.clone(),
                trigger_conditions: self.trigger_conditions.clone(),
                permissions: self.permissions.clone(),
                license: self.license.clone(),
            }),
        };
        serde_yaml_ng::to_string(&document).unwrap_or_default()
    }

    /// Atomically writes every rendered artifact under `dir`.
    ///
    /// Nested artifact names (such as `swarmflow/run.py`) create their parent
    /// directories; a name that escapes `dir` is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unsafe artifact name and
    /// [`Error::External`](openjiuwen_common::error::Error::external) for I/O
    /// failures.
    pub async fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        let directory = dir.as_ref();
        let artifacts = self.render();
        for (name, content) in &artifacts {
            if let Some(reason) = crate::flow::review::escaping_material_path(name) {
                return Err(validation("artifact", reason));
            }
            atomic_write(&directory.join(name), content.as_bytes()).await?;
        }
        Ok(())
    }

    /// Loads a pack previously written by [`SkillPack::save`].
    ///
    /// The name, description, and ordered skills come from the `SKILL.md` front
    /// matter; the workflow-graph block is validated against that order; steps
    /// and package metadata come from `dependencies.yaml`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for missing files, malformed front matter,
    /// an inconsistent skill list, or an unparsable manifest, and
    /// [`Error::External`](openjiuwen_common::error::Error::external) for I/O
    /// failures.
    pub async fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let directory = dir.as_ref();
        let skill_text = read_artifact(directory, SKILL_MD_FILENAME).await?;
        let dependencies_text = read_artifact(directory, DEPENDENCIES_FILENAME).await?;

        let (front_yaml, body) = split_front_matter(&skill_text)?;
        let front: SkillFrontMatter = serde_yaml_ng::from_str(front_yaml)
            .map_err(|error| validation("SKILL.md", format!("invalid front matter: {error}")))?;
        if front.kind != SKILL_PACK_KIND {
            return Err(validation(
                "kind",
                format!("expected `{SKILL_PACK_KIND}`, got `{}`", front.kind),
            ));
        }
        let manifest: DependenciesDoc =
            serde_yaml_ng::from_str(&dependencies_text).map_err(|error| {
                validation("dependencies.yaml", format!("invalid manifest: {error}"))
            })?;
        let graph = workflow_graph(body)?;
        validate_graph(graph.as_ref(), &front.skills)?;
        let steps = build_steps(manifest.capabilities, &front.skills, graph.as_ref())?;
        let header = manifest.package;
        let swarmflow_script = read_script_artifact(directory).await?;

        let mut pack = SkillPack {
            schema_version: header.as_ref().map_or_else(
                || PACKAGE_SCHEMA_VERSION.to_string(),
                |value| value.schema_version.clone(),
            ),
            package_id: header
                .as_ref()
                .map_or_else(String::new, |value| value.package_id.clone()),
            recipe_id: header
                .as_ref()
                .map_or_else(String::new, |value| value.recipe_id.clone()),
            recipe_version: header.as_ref().map_or(1, |value| value.recipe_version),
            target_kind: header.as_ref().map_or_else(
                || TARGET_KIND_SKILL.to_string(),
                |value| value.target_kind.clone(),
            ),
            name: front.name,
            description: front.description,
            steps,
            artifacts: BTreeMap::new(),
            swarmflow_script,
            trigger_conditions: header
                .as_ref()
                .map_or_else(String::new, |value| value.trigger_conditions.clone()),
            permissions: header
                .as_ref()
                .map_or_else(Vec::new, |value| value.permissions.clone()),
            license: header
                .as_ref()
                .map_or_else(String::new, |value| value.license.clone()),
        };
        if pack.schema_version.trim().is_empty() {
            pack.schema_version = PACKAGE_SCHEMA_VERSION.to_string();
        }
        if pack.target_kind.trim().is_empty() {
            pack.target_kind = TARGET_KIND_SKILL.to_string();
        }
        if pack.package_id.trim().is_empty() {
            pack.package_id = pack.compute_package_id();
        }
        pack.artifacts
            .insert(SKILL_MD_FILENAME.to_string(), skill_text);
        pack.artifacts
            .insert(DEPENDENCIES_FILENAME.to_string(), dependencies_text);
        insert_script_artifact(&mut pack.artifacts, pack.swarmflow_script.as_deref());
        Ok(pack)
    }

    /// Serializes the pack as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`](openjiuwen_common::error::Error::external) when
    /// serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::external)
    }

    /// Deserializes a pack from JSON, defaulting every missing field.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `value` is not a valid pack object.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| validation("skill_pack", format!("invalid package: {error}")))
    }
}

impl FlowEngine {
    /// Freezes the current version of `recipe_id` into a `SkillPack` and persists it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`](openjiuwen_common::error::Error::not_found)
    /// when the recipe is unknown, [`Error::Validation`] when the recipe is not
    /// active+verified or its structure is not a simple chain, and
    /// [`Error::External`](openjiuwen_common::error::Error::external) for
    /// persistence failures.
    pub async fn pack(&self, recipe_id: &str) -> Result<SkillPack> {
        let recipe = self
            .recipe(recipe_id)
            .await
            .ok_or_else(|| Error::not_found(format!("recipe not found: {recipe_id}")))?;
        let pack = build_skill_pack(&recipe)?;
        self.persist_package(&pack).await?;
        Ok(pack)
    }
}

/// Builds a pack from an active, verified, simple-chain recipe.
fn build_skill_pack(recipe: &ExperienceRecipe) -> Result<SkillPack> {
    if recipe.status != crate::flow::model::RECIPE_STATUS_ACTIVE {
        return Err(validation(
            "recipe.status",
            format!(
                "only active recipes can be packaged, got `{}`",
                recipe.status
            ),
        ));
    }
    if recipe.grade != crate::flow::model::RECIPE_GRADE_VERIFIED {
        return Err(validation(
            "recipe.grade",
            format!(
                "only verified recipes can be packaged, got `{}`",
                recipe.grade
            ),
        ));
    }
    let order = chain_order_from_structure(&recipe.combination_structure)
        .map_err(|reason| validation("recipe.combination_structure", reason))?;
    let task_description = recipe.applicability_text("task_description").trim();
    let base_description = if task_description.is_empty() {
        recipe.name.clone()
    } else {
        task_description.to_string()
    };
    let description = if base_description.starts_with("[技能包]") {
        base_description
    } else {
        format!("[技能包] {base_description}")
    };
    let mut steps = Vec::with_capacity(order.len());
    for (index, id) in order.iter().enumerate() {
        let node = recipe.combination_structure.nodes.get(id).map_or_else(
            || TraceNode {
                capability_id: id.clone(),
                ..TraceNode::default()
            },
            |value| TraceNode::from_combination_value(id, value),
        );
        steps.push(SkillPackStep {
            id: id.clone(),
            name: node.capability_id.clone(),
            description: node.description,
            capability_type: node.capability_type,
            inputs: node.inputs,
            outputs: node.outputs,
            depends_on: if index == 0 {
                Vec::new()
            } else {
                vec![order[index - 1].clone()]
            },
        });
    }
    let name = if is_valid_meta_name(&recipe.name) {
        recipe.name.clone()
    } else {
        normalize_meta_name(&recipe.name, "skill-pack")
    };
    let swarmflow_script = generate_swarmflow_script(recipe)?.source;
    let mut pack = SkillPack {
        schema_version: PACKAGE_SCHEMA_VERSION.to_string(),
        package_id: String::new(),
        recipe_id: recipe.recipe_id.clone(),
        recipe_version: recipe.version,
        target_kind: TARGET_KIND_SKILL.to_string(),
        name,
        description,
        steps,
        artifacts: BTreeMap::new(),
        swarmflow_script: Some(swarmflow_script),
        trigger_conditions: recipe.applicability_text("trigger_conditions").to_string(),
        permissions: Vec::new(),
        license: DEFAULT_LICENSE.to_string(),
    };
    pack.recompute_identity();
    Ok(pack)
}

/// Validates a recipe structure as a simple linear chain and returns the order.
pub(crate) fn chain_order_from_structure(
    structure: &crate::flow::model::CombinationStructure,
) -> std::result::Result<Vec<String>, String> {
    let ids: BTreeSet<String> = structure.nodes.keys().cloned().collect();
    let edges: Vec<(String, String)> = structure
        .edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    chain_order(&ids, &edges)
}

/// Validates the steps of a pack as a simple linear chain and returns the order.
pub(crate) fn chain_order_from_steps(
    steps: &[SkillPackStep],
) -> std::result::Result<Vec<String>, String> {
    let mut ids = BTreeSet::new();
    let mut edges = Vec::new();
    for step in steps {
        if step.id.trim().is_empty() {
            return Err("every step needs a non-blank id".to_string());
        }
        if !ids.insert(step.id.clone()) {
            return Err(format!("duplicate step id `{}`", step.id));
        }
        for dependency in &step.depends_on {
            edges.push((dependency.clone(), step.id.clone()));
        }
    }
    chain_order(&ids, &edges)
}

/// Validates a node set plus directed edges as a simple linear chain.
fn chain_order(
    ids: &BTreeSet<String>,
    edges: &[(String, String)],
) -> std::result::Result<Vec<String>, String> {
    if ids.len() < 2 {
        return Err("a SkillPack needs at least two capabilities".to_string());
    }
    if edges.len() != ids.len().saturating_sub(1) {
        return Err("a SkillPack needs exactly one edge per non-entry capability".to_string());
    }
    let mut indegree: BTreeMap<String, usize> = ids.iter().map(|id| (id.clone(), 0)).collect();
    let mut outdegree: BTreeMap<String, usize> = ids.iter().map(|id| (id.clone(), 0)).collect();
    let mut adjacency: BTreeMap<String, String> = BTreeMap::new();
    for (source, target) in edges {
        if !ids.contains(source) || !ids.contains(target) {
            return Err(format!(
                "edge `{source}` → `{target}` references an unknown capability"
            ));
        }
        if source == target {
            return Err("a SkillPack chain cannot contain a self edge".to_string());
        }
        if adjacency.insert(source.clone(), target.clone()).is_some() {
            return Err("a SkillPack chain cannot branch".to_string());
        }
        if let Some(degree) = indegree.get_mut(target) {
            *degree += 1;
        }
        if let Some(degree) = outdegree.get_mut(source) {
            *degree += 1;
        }
    }
    if indegree.values().any(|degree| *degree > 1) || outdegree.values().any(|degree| *degree > 1) {
        return Err("a SkillPack chain cannot branch or merge".to_string());
    }
    let starts: Vec<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    if starts.len() != 1 {
        return Err("a SkillPack chain needs exactly one start".to_string());
    }
    let mut order = Vec::with_capacity(ids.len());
    let mut current = starts[0].clone();
    while !current.is_empty() {
        if order.contains(&current) {
            return Err("a SkillPack chain cannot contain a cycle".to_string());
        }
        order.push(current.clone());
        current = adjacency.get(&current).cloned().unwrap_or_default();
    }
    if order.len() != ids.len() {
        return Err("a SkillPack chain must visit every capability".to_string());
    }
    Ok(order)
}

/// Reads one rendered pack artifact, mapping I/O failures to `Error::External`.
async fn read_artifact(directory: &Path, filename: &str) -> Result<String> {
    tokio::fs::read_to_string(directory.join(filename))
        .await
        .map_err(|error| {
            Error::external(std::io::Error::new(
                error.kind(),
                format!("read {filename}: {error}"),
            ))
        })
}
/// Rebuilds the ordered steps from the dependency entries and skill order.
fn build_steps(
    capabilities: Vec<DependencyEntry>,
    skills: &[String],
    graph: Option<&Value>,
) -> Result<Vec<SkillPackStep>> {
    let mut entries: BTreeMap<String, DependencyEntry> = capabilities
        .into_iter()
        .map(|entry| (entry.id.clone(), entry))
        .collect();
    let mut steps = Vec::with_capacity(skills.len());
    for skill in skills {
        let Some(entry) = entries.remove(skill) else {
            return Err(validation(
                "skills",
                format!("`{skill}` is missing from dependencies.yaml"),
            ));
        };
        let mut depends_on = entry.depends_on;
        if depends_on.is_empty() {
            depends_on = graph_dependencies(graph, skill);
        }
        steps.push(SkillPackStep {
            id: entry.id,
            name: entry.name,
            description: entry.description,
            capability_type: entry.capability_type,
            inputs: entry.inputs,
            outputs: entry.outputs,
            depends_on,
        });
    }
    if let Some((extra, _)) = entries.into_iter().next() {
        return Err(validation(
            "capabilities",
            format!("`{extra}` is not listed in the SKILL.md skills list"),
        ));
    }
    Ok(steps)
}
/// Splits a `SKILL.md` document into its YAML front matter and body.
fn split_front_matter(text: &str) -> Result<(&str, &str)> {
    let after_open = text
        .strip_prefix("---")
        .ok_or_else(|| validation(SKILL_MD_FILENAME, "must start with YAML front matter"))?;
    let after_open = after_open
        .strip_prefix('\n')
        .or_else(|| after_open.strip_prefix("\r\n"))
        .ok_or_else(|| {
            validation(
                SKILL_MD_FILENAME,
                "front matter delimiter must end its line",
            )
        })?;
    let (yaml, rest) = after_open.split_once("\n---").ok_or_else(|| {
        validation(
            SKILL_MD_FILENAME,
            "front matter closing delimiter is missing",
        )
    })?;
    let body = rest.trim_start_matches('\r').trim_start_matches('\n');
    Ok((yaml, body))
}

/// Extracts the workflow-graph JSON object from a `SKILL.md` body.
fn workflow_graph(body: &str) -> Result<Option<Value>> {
    let Some(start) = body.find("```json") else {
        return Ok(None);
    };
    let after_fence = body[start + "```json".len()..].trim_start();
    let Some(end) = after_fence.find("```") else {
        return Err(validation(
            SKILL_MD_FILENAME,
            "workflow graph fence is not terminated",
        ));
    };
    let text = after_fence[..end].trim();
    serde_json::from_str(text).map(Some).map_err(|error| {
        validation(
            SKILL_MD_FILENAME,
            format!("workflow graph is not JSON: {error}"),
        )
    })
}

/// Validates the workflow graph against the front-matter skill order.
fn validate_graph(graph: Option<&Value>, skills: &[String]) -> Result<()> {
    let Some(graph) = graph else {
        return Ok(());
    };
    let nodes = graph
        .get("graph")
        .and_then(|value| value.get("nodes"))
        .and_then(Value::as_object);
    let Some(nodes) = nodes else {
        return Err(validation(
            SKILL_MD_FILENAME,
            "workflow graph has no nodes object",
        ));
    };
    let mut node_ids: Vec<&str> = nodes.keys().map(String::as_str).collect();
    node_ids.sort_unstable();
    let mut expected: Vec<&str> = skills.iter().map(String::as_str).collect();
    expected.sort_unstable();
    if node_ids != expected {
        return Err(validation(
            "skills",
            "the workflow graph nodes do not match the front-matter skills",
        ));
    }
    Ok(())
}

/// Reads the incoming edges of `skill` from the workflow graph.
fn graph_dependencies(graph: Option<&Value>, skill: &str) -> Vec<String> {
    let Some(edges) = graph
        .and_then(|value| value.get("graph"))
        .and_then(|value| value.get("edges"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let mut dependencies: Vec<String> = edges
        .iter()
        .filter(|edge| edge.get("target").and_then(Value::as_str) == Some(skill))
        .filter_map(|edge| edge.get("source").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    dependencies.sort();
    dependencies.dedup();
    dependencies
}

/// The `SKILL.md` YAML front matter.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct SkillFrontMatter {
    name: String,
    kind: String,
    description: String,
    skills: Vec<String>,
}

impl Default for SkillFrontMatter {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: SKILL_PACK_KIND.to_string(),
            description: String::new(),
            skills: Vec::new(),
        }
    }
}

/// The `dependencies.yaml` manifest.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct DependenciesDoc {
    /// The ordered capability dependencies.
    #[serde(default)]
    pub capabilities: Vec<DependencyEntry>,
    /// The package header needed to reconstruct the pack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<PackageHeader>,
}

/// One capability dependency entry.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct DependencyEntry {
    /// The capability id.
    pub id: String,
    /// The recipe version frozen by the pack.
    pub version: String,
    /// The capability kind.
    #[serde(default)]
    pub capability_type: String,
    /// The display name.
    #[serde(default)]
    pub name: String,
    /// The capability description.
    #[serde(default)]
    pub description: String,
    /// The predecessors consumed by this capability.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// The declared inputs.
    #[serde(default)]
    pub inputs: Vec<CapabilityIo>,
    /// The declared outputs.
    #[serde(default)]
    pub outputs: Vec<CapabilityIo>,
}

/// The package header embedded in `dependencies.yaml`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct PackageHeader {
    /// The package schema version.
    #[serde(default)]
    pub schema_version: String,
    /// The content-addressed package id.
    #[serde(default)]
    pub package_id: String,
    /// The frozen recipe id.
    #[serde(default)]
    pub recipe_id: String,
    /// The frozen recipe version.
    #[serde(default)]
    pub recipe_version: u64,
    /// The artifact target kind.
    #[serde(default)]
    pub target_kind: String,
    /// The installable `meta.name`.
    #[serde(default)]
    pub name: String,
    /// The package description.
    #[serde(default)]
    pub description: String,
    /// The recipe's applicability trigger conditions, when declared.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub trigger_conditions: String,
    /// The declared permissions.
    #[serde(default)]
    pub permissions: Vec<String>,
    /// The declared license.
    #[serde(default)]
    pub license: String,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn chain_order_accepts_a_linear_chain() {
        let ids: BTreeSet<String> = ["a", "b", "c"].iter().map(|id| (*id).to_string()).collect();
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        assert_eq!(chain_order(&ids, &edges).unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn chain_order_rejects_branches_and_cycles() {
        let ids: BTreeSet<String> = ["a", "b", "c"].iter().map(|id| (*id).to_string()).collect();
        let branch = vec![
            ("a".to_string(), "b".to_string()),
            ("a".to_string(), "c".to_string()),
        ];
        assert!(chain_order(&ids, &branch).is_err());
        let cyclic = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "a".to_string()),
        ];
        assert!(chain_order(&ids, &cyclic).is_err());
        let single: BTreeSet<String> = ["a"].iter().map(|id| (*id).to_string()).collect();
        assert!(chain_order(&single, &[]).is_err());
    }

    #[test]
    fn front_matter_split_round_trips() {
        let text = "---\nname: demo\n---\n\n# demo\n\nbody\n";
        let (yaml, body) = split_front_matter(text).unwrap();
        assert_eq!(yaml, "name: demo");
        assert!(body.starts_with("# demo"));
    }
}
