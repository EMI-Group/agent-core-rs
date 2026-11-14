//! Artifact rendering adapters for reviewed packages.
//!
//! Ports the deterministic half of Python `openjiuwen.symphony.flow.render`: a
//! reviewed [`SkillPack`] is rendered as one SDD-0010 `SkillPack` root document
//! (`SKILL.md`) whose front matter lists the chain members and whose body carries
//! the task description, required inputs, execution process, and the embedded
//! workflow graph.
//!
//! Python's adapter receives a `package` dict; the Rust model is a [`SkillPack`],
//! so the adapter renders from the pack's ordered [`SkillPackStep`]s rather than a
//! parallel materials model. The recipe-level `trigger_conditions` that Python
//! reads from `materials.recipe.applicability` travel on
//! [`SkillPack::trigger_conditions`](crate::flow::pack::SkillPack::trigger_conditions).
//!
//! Only the `skill` target kind is supported; any other target is a typed
//! [`Error::Unsupported`](crate::error::Error), mirroring Python's
//! `UnsupportedTargetException`.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{unsupported, validation, Error, Result};
use crate::flow::model::CAN_FEED;
use crate::flow::pack::{
    chain_order_from_steps, SkillPack, SkillPackStep, SKILL_MD_FILENAME, TARGET_KIND_SKILL,
};

/// The name used when a pack declares no installable name.
pub const SKILL_PACK_NAME_FALLBACK: &str = "symphony-combination";
/// The task description used when a pack declares none.
pub const SKILL_PACK_TASK_FALLBACK: &str = "由 Symphony 成功执行生成的组合 Skill";
/// The marker a `SkillPack` description carries.
pub const SKILL_PACK_DESCRIPTION_PREFIX: &str = "[技能包]";
/// The fallback description of one included member.
pub const INCLUDED_SKILL_FALLBACK: &str = "具体用途见该成员的 SKILL.md。";
/// The character limit of one included-member description.
pub const INCLUDED_SKILL_LIMIT: usize = 240;
/// The front-matter `kind` value of a `SkillPack` document.
pub const SKILL_PACK_DOCUMENT_KIND: &str = "skillpack";
/// The workflow-graph `type` value of a `SkillPack` document.
pub const SKILL_PACK_GRAPH_TYPE: &str = "skillpack_workflow";
/// The maximum length of the pack name before `-pack` is appended.
pub const SKILL_PACK_NAME_LIMIT: usize = 123;

/// One rendered artifact, ready to be written under its relative path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RenderedArtifact {
    /// The artifact target kind (`skill`).
    pub target: String,
    /// The path of the artifact relative to the package root.
    pub path: String,
    /// The file contents.
    pub content: String,
}

/// A rendering boundary for one artifact target kind.
pub trait SkillAdapter: Send + Sync {
    /// Returns the target kind this adapter renders (`skill`).
    fn target(&self) -> &str;

    /// Renders `pack` into one artifact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pack is not an installable linear
    /// Skill chain, and [`Error::External`] when rendering fails.
    fn render(&self, pack: &SkillPack) -> Result<RenderedArtifact>;
}

/// Renders a reviewed pack as one SDD-0010 `SkillPack` root document.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkillPackAdapter;

impl SkillPackAdapter {
    /// Creates the adapter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Requires an installable linear Skill chain.
    ///
    /// The pack must contain at least two members, every member must be an
    /// ordinary `skill` capability with a unique non-blank id, and the members
    /// plus their `can_feed` edges must form a simple linear chain.
    ///
    /// Packaging already enforces the recipe's `active` + `verified` status (see
    /// [`FlowEngine::pack`](crate::flow::engine::FlowEngine::pack)); a pack does
    /// not retain the recipe status or grade, so this check covers the structure
    /// contract the renderer depends on.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the first contract violation.
    pub fn validate_members(pack: &SkillPack) -> Result<()> {
        if pack.steps.len() < 2 {
            return Err(validation("skillpack", "recipe is not a Skill chain"));
        }
        let mut seen = BTreeSet::new();
        for step in &pack.steps {
            if step.id.trim().is_empty() {
                return Err(validation("skillpack", "capability id is blank"));
            }
            if !seen.insert(step.id.as_str()) {
                return Err(validation(
                    "skillpack",
                    format!("capability {} is duplicated", step.id),
                ));
            }
            if !step.capability_type.trim().eq_ignore_ascii_case("skill") {
                return Err(validation(
                    "skillpack",
                    format!("capability {} is not a Skill", step.id),
                ));
            }
        }
        chain_order_from_steps(&pack.steps).map_err(|reason| validation("skillpack", reason))?;
        Ok(())
    }

    /// Renders the pack's `SKILL.md` document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pack is not an installable linear
    /// chain, and [`Error::External`] when YAML or JSON rendering fails.
    pub fn render(&self, pack: &SkillPack) -> Result<RenderedArtifact> {
        let (members, edges) = skillpack_chain(pack)?;
        Ok(RenderedArtifact {
            target: TARGET_KIND_SKILL.to_string(),
            path: SKILL_MD_FILENAME.to_string(),
            content: render_document(pack, &members, &edges)?,
        })
    }
}

impl SkillAdapter for SkillPackAdapter {
    fn target(&self) -> &str {
        TARGET_KIND_SKILL
    }

    fn render(&self, pack: &SkillPack) -> Result<RenderedArtifact> {
        SkillPackAdapter::render(self, pack)
    }
}

/// Renders a pack for `target_kind`, rejecting unsupported targets.
///
/// An empty target kind defaults to `skill`, matching Python's
/// `package.get("target_kind") or TARGET_KIND_SKILL`.
///
/// # Errors
///
/// Returns [`Error::Unsupported`](crate::error::Error) for a target other than
/// `skill` and propagates the adapter's validation and rendering failures.
pub fn render_package(pack: &SkillPack, target_kind: &str) -> Result<RenderedArtifact> {
    let target = if target_kind.trim().is_empty() {
        TARGET_KIND_SKILL
    } else {
        target_kind
    };
    if target == TARGET_KIND_SKILL {
        return SkillPackAdapter.render(pack);
    }
    Err(unsupported(&format!("flow render target `{target}`")))
}

/// Validates the pack and returns its ordered members plus chain edges.
fn skillpack_chain(pack: &SkillPack) -> Result<(Vec<String>, Vec<Value>)> {
    SkillPackAdapter::validate_members(pack)?;
    let order =
        chain_order_from_steps(&pack.steps).map_err(|reason| validation("skillpack", reason))?;
    let edges = order
        .windows(2)
        .map(|pair| {
            serde_json::json!({
                "source": pair[0].clone(),
                "target": pair[1].clone(),
                "relation": CAN_FEED,
            })
        })
        .collect();
    Ok((order, edges))
}

/// Renders the complete `SKILL.md` content for a validated chain.
fn render_document(pack: &SkillPack, members: &[String], edges: &[Value]) -> Result<String> {
    let name = document_name(pack, members);
    let task_description = compact_text(pack.task_description(), SKILL_PACK_TASK_FALLBACK, None);
    let description = if task_description.starts_with(SKILL_PACK_DESCRIPTION_PREFIX) {
        task_description.clone()
    } else {
        format!("{SKILL_PACK_DESCRIPTION_PREFIX} {task_description}")
    };
    let trigger = compact_text(&pack.trigger_conditions, "", None);
    let front_matter = FrontMatter {
        name: &name,
        kind: SKILL_PACK_DOCUMENT_KIND,
        description: &description,
        skills: members,
    };
    let yaml = serde_yaml_ng::to_string(&front_matter).map_err(Error::external)?;
    let mut body = String::new();
    let _ = writeln!(body, "# {name}\n");
    body.push_str("## When to use\n\n");
    body.push_str(&task_description);
    body.push('\n');
    if !trigger.is_empty() && trigger != task_description {
        let _ = write!(body, "\n触发条件：{trigger}\n");
    }
    body.push_str("\n## Do not use\n\n");
    body.push_str("仅需要其中一个成员 Skill 的单项能力时，不使用本技能包。\n");
    body.push_str("\n## Required inputs\n\n");
    for line in required_input_lines(pack, members) {
        let _ = writeln!(body, "{line}");
    }
    body.push_str("\n## Side effects and confirmation\n\n");
    body.push_str("遵循各成员 Skill 的权限与确认要求，不扩大其工具权限或副作用范围。\n");
    body.push_str("\n## Included Skills\n\n");
    for line in included_skill_lines(pack, members) {
        let _ = writeln!(body, "{line}");
    }
    body.push_str("\n## Execution Process\n\n");
    for line in execution_process_lines(pack, members) {
        let _ = writeln!(body, "{line}");
    }
    body.push_str("\n## Failure handling\n\n");
    body.push_str(
        "成员执行失败时停止依赖该结果的后续步骤，并返回已有结果、失败原因和未完成步骤。\n",
    );
    body.push_str("\n## Final output\n\n");
    for line in final_output_lines(pack, members) {
        let _ = writeln!(body, "{line}");
    }
    body.push_str("\n## Workflow Graph\n\n```json\n");
    body.push_str(&render_workflow_graph(
        &name,
        &task_description,
        members,
        edges,
    )?);
    body.push_str("\n```\n");
    Ok(format!("---\n{}\n---\n\n{body}", yaml.trim()))
}

/// Derives the document name, avoiding a collision with a member id.
fn document_name(pack: &SkillPack, members: &[String]) -> String {
    let trimmed = pack.name.trim();
    let mut name = if trimmed.is_empty() {
        SKILL_PACK_NAME_FALLBACK.to_string()
    } else {
        trimmed.to_string()
    };
    if members.contains(&name) {
        let prefix: String = name.chars().take(SKILL_PACK_NAME_LIMIT).collect();
        name = format!("{prefix}-pack");
    }
    name
}

/// Renders the embedded workflow-graph object.
fn render_workflow_graph(
    name: &str,
    task_description: &str,
    members: &[String],
    edges: &[Value],
) -> Result<String> {
    let mut nodes = Map::new();
    for member in members {
        nodes.insert(
            member.clone(),
            serde_json::json!({"label": member, "metadata": {"skill": member}}),
        );
    }
    let graph = serde_json::json!({
        "graph": {
            "id": format!("{name}-workflow"),
            "type": SKILL_PACK_GRAPH_TYPE,
            "label": task_description,
            "directed": true,
            "nodes": Value::Object(nodes),
            "edges": edges,
        }
    });
    let rendered = serde_json::to_string_pretty(&graph).map_err(Error::external)?;
    Ok(rendered)
}

/// Finds one member step by id.
fn member_step<'a>(pack: &'a SkillPack, member: &str) -> Option<&'a SkillPackStep> {
    pack.steps.iter().find(|step| step.id == member)
}

/// Renders the required-input list, or the documented no-extra-inputs line.
fn required_input_lines(pack: &SkillPack, members: &[String]) -> Vec<String> {
    let mut lines = Vec::new();
    for member in members {
        let Some(step) = member_step(pack, member) else {
            continue;
        };
        for input in &step.inputs {
            if input.required != Some(true) {
                continue;
            }
            let name = compact_text(&input.name, "", None);
            let port_type = compact_text(&input.io_type, "", None);
            let description = compact_text(&input.description, "", None);
            let detail = if port_type.is_empty() {
                String::new()
            } else {
                format!("（{port_type}）")
            };
            let suffix = if description.is_empty() {
                String::new()
            } else {
                format!("：{description}")
            };
            lines.push(format!("- `{member}.{name}`{detail}{suffix}"));
        }
    }
    if lines.is_empty() {
        lines.push("提供完整的用户目标；各成员未声明额外的必填输入。".to_string());
    }
    lines
}

/// Renders one line per included member.
fn included_skill_lines(pack: &SkillPack, members: &[String]) -> Vec<String> {
    members
        .iter()
        .map(|member| {
            let description =
                member_step(pack, member).map_or("", |step| step.description.as_str());
            format!(
                "- `{member}`：{}",
                compact_text(
                    description,
                    INCLUDED_SKILL_FALLBACK,
                    Some(INCLUDED_SKILL_LIMIT)
                )
            )
        })
        .collect()
}

/// Renders the numbered execution-process description.
fn execution_process_lines(pack: &SkillPack, members: &[String]) -> Vec<String> {
    let mut lines = vec!["当前为线性流程，无可并行步骤。".to_string(), String::new()];
    for (index, member) in members.iter().enumerate() {
        let input_text = if index == 0 {
            "接收用户原始目标和该成员声明的必要输入".to_string()
        } else {
            format!(
                "接收用户原始目标，并将 `{}` 的完整输出作为上下文输入",
                members[index - 1]
            )
        };
        let outputs: Vec<String> = member_step(pack, member)
            .map(|step| {
                step.outputs
                    .iter()
                    .filter_map(|output| {
                        let name = compact_text(&output.name, "", None);
                        (!name.is_empty()).then(|| format!("`{name}`"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let output_text = if outputs.is_empty() {
            "本步骤结果".to_string()
        } else {
            outputs.join("、")
        };
        lines.push(format!(
            "{}. 调用 `{member}`：{input_text}；产出 {output_text}。",
            index + 1
        ));
    }
    if let Some(last) = members.last() {
        lines.push(String::new());
        lines.push(format!(
            "结果汇合：以 `{last}` 的输出作为最终结果，前序结果作为上下文和依据。"
        ));
    }
    lines
}

/// Renders the final-output description from the last member's outputs.
fn final_output_lines(pack: &SkillPack, members: &[String]) -> Vec<String> {
    let Some(last) = members.last() else {
        return vec!["返回本技能包生成的最终结果，并说明任何失败、跳过或不完整部分。".to_string()];
    };
    let outputs = member_step(pack, last).map_or(&[][..], |step| step.outputs.as_slice());
    if outputs.is_empty() {
        return vec![format!(
            "返回 `{last}` 生成的最终结果，并说明任何失败、跳过或不完整部分。"
        )];
    }
    let mut lines = vec![format!("返回 `{last}` 生成的以下最终产物：")];
    for output in outputs {
        let name = compact_text(&output.name, "", None);
        let description = compact_text(&output.description, "", None);
        let suffix = if description.is_empty() {
            String::new()
        } else {
            format!("：{description}")
        };
        lines.push(format!("- `{name}`{suffix}"));
    }
    lines.push("并说明任何失败、跳过或不完整部分。".to_string());
    lines
}

/// Collapses whitespace and optionally truncates to `limit` characters.
///
/// Mirrors Python's `_compact_text`: whitespace runs become one space, an empty
/// result falls back, and an over-long result keeps `limit - 1` characters plus
/// an ellipsis (counted in Unicode scalar values, like Python's `len`).
fn compact_text(value: &str, fallback: &str, limit: Option<usize>) -> String {
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = if collapsed.is_empty() {
        fallback.to_string()
    } else {
        collapsed
    };
    match limit {
        Some(limit) if text.chars().count() > limit => {
            let kept: String = text.chars().take(limit.saturating_sub(1)).collect();
            format!("{}…", kept.trim_end())
        }
        _ => text,
    }
}

/// The `SKILL.md` YAML front matter.
#[derive(Debug, Serialize)]
struct FrontMatter<'a> {
    /// The document name.
    name: &'a str,
    /// The document kind (`skillpack`).
    kind: &'a str,
    /// The one-line description.
    description: &'a str,
    /// The ordered chain members.
    skills: &'a [String],
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::flow::pack::SkillPackStep;

    fn demo_pack() -> SkillPack {
        let mut pack = SkillPack {
            recipe_id: "recipe_0123456789ab".to_string(),
            name: "demo-pack".to_string(),
            description: "[技能包] summarize the corpus".to_string(),
            trigger_conditions: "when a request needs a summary".to_string(),
            steps: vec![
                SkillPackStep {
                    id: "search".to_string(),
                    name: "search".to_string(),
                    description: "search the corpus".to_string(),
                    capability_type: "skill".to_string(),
                    ..SkillPackStep::default()
                },
                SkillPackStep {
                    id: "summarize".to_string(),
                    name: "summarize".to_string(),
                    description: "summarize the hits".to_string(),
                    capability_type: "skill".to_string(),
                    depends_on: vec!["search".to_string()],
                    ..SkillPackStep::default()
                },
            ],
            license: "Proprietary".to_string(),
            ..SkillPack::default()
        };
        pack.recompute_identity();
        pack
    }

    #[test]
    fn compact_text_collapses_and_truncates() {
        assert_eq!(compact_text("  a \n b ", "", None), "a b");
        assert_eq!(compact_text("", "fallback", None), "fallback");
        assert_eq!(compact_text("abcdef", "", Some(4)), "abc…");
    }

    #[test]
    fn a_linear_chain_renders_a_skillpack_document() {
        let pack = demo_pack();
        let artifact = SkillPackAdapter.render(&pack).unwrap();
        assert_eq!(artifact.target, TARGET_KIND_SKILL);
        assert_eq!(artifact.path, SKILL_MD_FILENAME);
        assert!(artifact
            .content
            .starts_with("---\nname: demo-pack\nkind: skillpack"));
        assert!(artifact.content.contains("## Included Skills"));
        assert!(artifact.content.contains("- `search`：search the corpus"));
        assert!(artifact
            .content
            .contains("触发条件：when a request needs a summary"));
        assert!(artifact
            .content
            .contains("\"type\": \"skillpack_workflow\""));
        assert!(artifact.content.ends_with("```\n"));
        assert_eq!(SkillPackAdapter.target(), TARGET_KIND_SKILL);
    }

    #[test]
    fn invalid_chains_are_rejected() {
        let mut pack = demo_pack();
        pack.steps.pop();
        assert!(SkillPackAdapter::validate_members(&pack).is_err());

        let mut pack = demo_pack();
        pack.steps[1].id = "search".to_string();
        assert!(SkillPackAdapter::validate_members(&pack).is_err());

        let mut pack = demo_pack();
        pack.steps[0].capability_type = "workflow".to_string();
        assert!(SkillPackAdapter::validate_members(&pack).is_err());
    }

    #[test]
    fn an_unsupported_target_is_reported() {
        let pack = demo_pack();
        let error = render_package(&pack, "plugin").unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Unsupported);
        assert!(error.message().contains("flow render target `plugin`"));
        assert!(render_package(&pack, "").is_ok());
    }
}
