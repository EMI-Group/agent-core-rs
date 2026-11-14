//! Behaviour tests for `SwarmFlow` script generation and `SkillPack` rendering.
//!
//! Everything is offline and deterministic: recipes are constructed in memory,
//! the generated script is compared byte for byte, and the package path uses a
//! temporary flow root. Timestamps are never compared.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp, clippy::assert_is_empty)]

use std::collections::BTreeMap;
use std::fs;

use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::flow::{
    generate_swarmflow_script, render_package, swarmflow_identifier, validate_generated_script,
    CombinationStructure, ExperienceRecipe, FlowEngine, RecipeEdge, SkillPack, SkillPackAdapter,
    SkillPackStep, SwarmflowScript, CAN_FEED, RECIPE_GRADE_VERIFIED, RECIPE_STATUS_ACTIVE,
    VERDICT_APPROVED,
};
use openjiuwen_symphony::model::CapabilityIo;
use serde_json::{json, Value};
use tempfile::TempDir;

/// Builds one recipe node payload.
fn recipe_node(description: &str) -> Value {
    json!({
        "label": "capability",
        "metadata": {
            "capability_type": "skill",
            "skill_type": "skill",
            "version": "1.0.0",
            "description": description,
            "inputs": [],
            "outputs": [],
        },
    })
}

/// Builds the canonical `search → summarize` two-step recipe.
fn demo_recipe() -> ExperienceRecipe {
    let mut structure = CombinationStructure::default();
    structure
        .nodes
        .insert("search".to_string(), recipe_node("search the corpus"));
    structure
        .nodes
        .insert("summarize".to_string(), recipe_node("summarize the hits"));
    structure.edges = vec![RecipeEdge {
        source: "search".to_string(),
        target: "summarize".to_string(),
        relation: CAN_FEED.to_string(),
    }];
    let mut applicability = BTreeMap::new();
    applicability.insert("task_description".to_string(), json!("把查询整理成摘要"));
    applicability.insert(
        "trigger_conditions".to_string(),
        json!("多个检索能力接力时使用"),
    );
    ExperienceRecipe {
        recipe_id: "recipe_0123456789ab".to_string(),
        name: "demo-pack".to_string(),
        version: 1,
        status: RECIPE_STATUS_ACTIVE.to_string(),
        grade: RECIPE_GRADE_VERIFIED.to_string(),
        applicability,
        combination_structure: structure,
        execution_narrative: "执行过程：先检索，再总结。".to_string(),
        ..ExperienceRecipe::default()
    }
}

/// Builds one declared capability port.
fn port(name: &str, io_type: &str, required: bool, description: &str) -> Value {
    json!({
        "name": name,
        "type": io_type,
        "required": required,
        "description": description,
    })
}

/// Builds one capability node payload for a trace.
fn node(description: &str, inputs: &Value, outputs: &Value) -> Value {
    json!({
        "label": "capability",
        "metadata": {
            "capability_type": "skill",
            "skill_type": "skill",
            "version": "1.0.0",
            "description": description,
            "inputs": inputs,
            "outputs": outputs,
        },
    })
}

/// Builds one successful trace edge.
fn edge(source: &str, target: &str) -> Value {
    json!({
        "source": source,
        "target": target,
        "relation": "can_feed",
        "metadata": {"success": true, "reason": "observed"},
    })
}

/// The canonical `search → summarize` trace payload.
fn chain_payload(trace_id: &str, query: &str) -> Value {
    json!({
        "trace_id": trace_id,
        "query": query,
        "outcome": "success",
        "graph": {
            "id": format!("graph-{trace_id}"),
            "type": "execution_graph",
            "label": "run",
            "directed": true,
            "nodes": {
                "skill:search": node(
                    "search the corpus",
                    &json!([port("query", "string", true, "user query")]),
                    &json!([port("results", "array", true, "ranked hits")]),
                ),
                "skill:summarize": node(
                    "summarize the hits",
                    &json!([port("results", "array", true, "ranked hits")]),
                    &json!([port("summary", "text", false, "final answer")]),
                ),
            },
            "edges": [edge("skill:search", "skill:summarize")],
        },
    })
}

/// Builds a hand-made installable pack with a linear `a → b` chain.
fn demo_pack() -> SkillPack {
    let mut pack = SkillPack {
        recipe_id: "recipe_0123456789ab".to_string(),
        name: "demo-pack".to_string(),
        description: "[技能包] 把查询整理成摘要".to_string(),
        trigger_conditions: "多个检索能力接力时使用".to_string(),
        steps: vec![
            SkillPackStep {
                id: "search".to_string(),
                name: "search".to_string(),
                description: "search the corpus".to_string(),
                capability_type: "skill".to_string(),
                inputs: vec![CapabilityIo::new("query", "string")
                    .with_description("user query")
                    .with_required(true)],
                outputs: Vec::new(),
                depends_on: Vec::new(),
            },
            SkillPackStep {
                id: "summarize".to_string(),
                name: "summarize".to_string(),
                description: "summarize the hits".to_string(),
                capability_type: "skill".to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                depends_on: vec!["search".to_string()],
            },
        ],
        license: "Proprietary".to_string(),
        ..SkillPack::default()
    };
    pack.recompute_identity();
    pack
}

#[test]
fn a_two_step_recipe_renders_the_pinned_script() {
    let script = generate_swarmflow_script(&demo_recipe()).expect("generate");
    let expected = concat!(
        "\"\"\"SwarmFlow script generated from recipe recipe_0123456789ab.\"\"\"\n",
        "\n",
        "from swarmflow import agent\n",
        "\n",
        "META = {'name': 'demo-pack', 'phases': ['step_1', 'step_2']}\n",
        "\n",
        "TASK = '把查询整理成摘要'\n",
        "\n",
        "async def run(args):\n",
        "    \"\"\"按 skill pack 拓扑序执行能力组合。\"\"\"\n",
        "    step_1 = await agent('使用能力 search（search the corpus）完成本阶段任务。输入来自任务参数。', label='search', phase='step_1')\n",
        "    step_2 = await agent('使用能力 summarize（summarize the hits）完成本阶段任务。上一步产出：search。', label='summarize', phase='step_2')\n",
        "    return step_2\n",
    );
    assert_eq!(script.source, expected);
    assert_eq!(script.identifier, "demo-pack");
    assert_eq!(script.task, "把查询整理成摘要");
    assert_eq!(script.steps, ["search", "summarize"]);
    assert_eq!(
        script.meta,
        json!({"name": "demo-pack", "phases": ["step_1", "step_2"]})
    );
    validate_generated_script(&script.source).expect("generated script validates");

    // The serialized form round-trips.
    let value = serde_json::to_value(&script).expect("serialize");
    let restored: SwarmflowScript = serde_json::from_value(value).expect("deserialize");
    assert_eq!(restored, script);
}

#[test]
fn an_empty_combination_still_renders_a_valid_script() {
    let recipe = ExperienceRecipe {
        recipe_id: "recipe_000000000000".to_string(),
        name: "empty-pack".to_string(),
        ..ExperienceRecipe::default()
    };
    let script = generate_swarmflow_script(&recipe).expect("generate");
    assert!(script.steps.is_empty());
    assert!(script.source.contains("    return None\n"));
    assert!(script
        .source
        .contains("META = {'name': 'empty-pack', 'phases': []}"));
    validate_generated_script(&script.source).expect("validates");
}

#[test]
fn identifiers_are_stable_and_content_addressed() {
    let first = serde_json::json!({"nodes": {"a": 1}, "edges": []});
    let same = serde_json::json!({"edges": [], "nodes": {"a": 1}});
    let changed = serde_json::json!({"nodes": {"a": 2}, "edges": []});

    let identifier = swarmflow_identifier(&first, "recipe_abc");
    assert!(identifier.starts_with("combo-"));
    assert_eq!(identifier.len(), "combo-".len() + 12);
    assert_eq!(identifier, swarmflow_identifier(&same, "recipe_abc"));
    assert_ne!(identifier, swarmflow_identifier(&changed, "recipe_abc"));

    // A named recipe overrides the derived identifier.
    let named = generate_swarmflow_script(&demo_recipe()).expect("generate");
    assert_eq!(named.identifier, "demo-pack");

    // A nameless recipe falls back to the content-addressed identifier.
    let mut recipe = demo_recipe();
    recipe.name.clear();
    let derived = generate_swarmflow_script(&recipe).expect("generate");
    assert!(derived.identifier.starts_with("combo-"));
    let materials =
        serde_json::to_value(&recipe.combination_structure).expect("structure serializes");
    assert_eq!(
        derived.identifier,
        swarmflow_identifier(&materials, "recipe_0123456789ab")
    );
}

#[test]
fn quotes_newlines_and_cjk_are_escaped() {
    let mut recipe = demo_recipe();
    recipe.name.clear();
    recipe.applicability.insert(
        "task_description".to_string(),
        json!("first line\nsecond 'quoted' 摘要"),
    );
    recipe
        .combination_structure
        .nodes
        .insert("search".to_string(), recipe_node("it's\nmulti-line"));
    let script = generate_swarmflow_script(&recipe).expect("generate");
    assert!(script
        .source
        .contains("TASK = 'first line\\nsecond \\'quoted\\' 摘要'"));
    assert!(script
        .source
        .contains("使用能力 search（it\\'s\\nmulti-line）"));
    assert!(script.source.contains("摘要"));
    validate_generated_script(&script.source).expect("escaped script validates");
    assert_eq!(script.task, "first line\nsecond 'quoted' 摘要");
}

#[test]
fn the_validator_accepts_generated_output_and_names_the_first_bad_line() {
    let generated = generate_swarmflow_script(&demo_recipe()).expect("generate");
    validate_generated_script(&generated.source).expect("generated script validates");

    let cases: [(&str, &str); 7] = [
        (
            "META = {}\nTASK = ''\nasync def run(args):\n    import json\n",
            "plain import",
        ),
        (
            "META = {}\nTASK = ''\nasync def run(args):\n    from sys import exit\n",
            "disallowed import from `sys`",
        ),
        (
            "META = {}\nTASK = ''\nasync def run(args):\n    from swarmflow import tool\n",
            "disallowed operator import `tool`",
        ),
        (
            "META = {}\nTASK = ''\nasync def run(args):\n    x = open('file')\n",
            "disallowed call `open`",
        ),
        (
            "META = {}\nTASK = ''\nasync def run(args):\n    x = subprocess.run([])\n",
            "forbidden token `subprocess`",
        ),
        ("META = {'name': 'x'\n", "unclosed `{`"),
        (
            "META = {'name': 'x'}\nasync def run(args):\n    return None\n",
            "missing `TASK`",
        ),
    ];
    for (source, needle) in cases {
        let error = validate_generated_script(source).expect_err(source);
        assert_eq!(error.kind(), ErrorKind::Validation, "{source}");
        assert!(
            error.message().contains(needle),
            "{source}: {} should mention {needle}",
            error.message()
        );
    }

    let line_error = validate_generated_script(
        "META = {}\nTASK = ''\nasync def run(args):\n    x = eval('2')\n",
    )
    .expect_err("forbidden eval call");
    assert!(
        line_error.message().contains("line 4"),
        "{}",
        line_error.message()
    );
}

#[test]
fn member_validation_rejects_incomplete_duplicate_or_branched_chains() {
    let mut short = demo_pack();
    short.steps.pop();
    let error = SkillPackAdapter::validate_members(&short).expect_err("one member");
    assert!(error.message().contains("not a Skill chain"));

    let mut duplicate = demo_pack();
    duplicate.steps[1].id = "search".to_string();
    let error = SkillPackAdapter::validate_members(&duplicate).expect_err("duplicate");
    assert!(error.message().contains("duplicated"));

    let mut not_skill = demo_pack();
    not_skill.steps[0].capability_type = "workflow".to_string();
    let error = SkillPackAdapter::validate_members(&not_skill).expect_err("not a skill");
    assert!(error.message().contains("not a Skill"));

    let mut branched = demo_pack();
    branched.steps.push(SkillPackStep {
        id: "third".to_string(),
        name: "third".to_string(),
        capability_type: "skill".to_string(),
        depends_on: vec!["search".to_string()],
        ..SkillPackStep::default()
    });
    let error = SkillPackAdapter::validate_members(&branched).expect_err("branched");
    assert!(error.message().contains("branch"));

    let mut blank = demo_pack();
    blank.steps[0].id = "  ".to_string();
    let error = SkillPackAdapter::validate_members(&blank).expect_err("blank id");
    assert!(error.message().contains("blank"));

    SkillPackAdapter::validate_members(&demo_pack()).expect("the demo pack is installable");
}

#[test]
fn an_unsupported_target_reports_the_render_boundary() {
    let pack = demo_pack();
    let error = render_package(&pack, "plugin").expect_err("unsupported");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("flow render target `plugin`"));

    let artifact = render_package(&pack, "").expect("empty target defaults to skill");
    assert_eq!(artifact.target, "skill");
    assert_eq!(artifact.path, "SKILL.md");
    assert!(artifact
        .content
        .starts_with("---\nname: demo-pack\nkind: skillpack"));
    assert!(artifact
        .content
        .contains("## Side effects and confirmation"));
    assert!(artifact.content.contains("- `search`：search the corpus"));
    assert!(artifact
        .content
        .contains("触发条件：多个检索能力接力时使用"));
    assert!(artifact
        .content
        .contains("\"type\": \"skillpack_workflow\""));
    assert!(artifact
        .content
        .contains("- `search.query`（string）：user query"));
}

#[test]
fn the_script_participates_in_the_package_identity() {
    let plain = demo_pack();
    assert!(plain
        .materials()
        .as_object()
        .unwrap()
        .contains_key("trigger_conditions"));

    // A pack without a script or trigger conditions keeps the original materials.
    let mut minimal = plain.clone();
    minimal.trigger_conditions.clear();
    let materials = minimal.materials();
    let object = materials.as_object().unwrap();
    assert!(!object.contains_key("swarmflow_script"));
    assert!(!object.contains_key("trigger_conditions"));

    let mut with_script = demo_pack();
    with_script.swarmflow_script = Some(
        generate_swarmflow_script(&demo_recipe())
            .expect("generate")
            .source,
    );
    with_script.recompute_identity();
    assert_ne!(plain.package_id, with_script.package_id);
    assert!(with_script
        .materials()
        .as_object()
        .unwrap()
        .contains_key("swarmflow_script"));
    assert!(with_script.render().contains_key("swarmflow/run.py"));
    assert!(with_script.render()["swarmflow/run.py"].contains("from swarmflow import agent"));
}

#[tokio::test]
async fn a_packed_recipe_ships_run_py_and_passes_review_and_install() {
    let dir = TempDir::new().unwrap();
    let engine = FlowEngine::open(dir.path()).await.unwrap();
    for (trace_id, query) in [
        ("t1", "find and summarize the docs"),
        ("t2", "summarize the release notes"),
    ] {
        assert!(engine
            .ingest_value(chain_payload(trace_id, query))
            .await
            .unwrap());
    }
    let report = engine.distill().await.unwrap();
    assert_eq!(report.recipes_saved.len(), 1);
    let recipe = engine.recipes().await.first().cloned().expect("recipe");

    let pack = engine.pack(&recipe.recipe_id).await.unwrap();
    let script = pack.swarmflow_script.as_deref().expect("generated script");
    validate_generated_script(script).expect("packed script validates");
    assert!(script.contains("label='search'"));
    assert!(script.contains("label='summarize'"));
    assert!(script.contains("上一步产出：search"));

    let review = openjiuwen_symphony::flow::Reviewer::new().review(&pack);
    assert_eq!(
        review.verdict,
        VERDICT_APPROVED,
        "failed checks: {:?}",
        review.failure_reasons()
    );
    assert!(review
        .checks
        .iter()
        .find(|check| check.check == "code_safety")
        .is_some_and(|check| check.result == "pass"));

    let destination = TempDir::new().unwrap();
    let preparation = engine
        .review_and_prepare_install(&recipe.recipe_id, "skill", destination.path())
        .await
        .unwrap();
    assert_eq!(
        preparation.verdict, VERDICT_APPROVED,
        "reasons: {:?}",
        preparation.reasons
    );
    let run_py = destination.path().join("swarmflow").join("run.py");
    let written = fs::read_to_string(&run_py).expect("run.py is written");
    assert_eq!(Some(written.as_str()), pack.swarmflow_script.as_deref());
    assert!(written.contains("async def run(args):"));

    // The artifact directory reloads into an equal pack, script included.
    let loaded = SkillPack::load(destination.path()).await.unwrap();
    assert_eq!(loaded, preparation.package.expect("package"));
    assert_eq!(loaded.artifacts, pack.render());
}

#[tokio::test]
async fn a_pack_with_a_broken_script_fails_code_safety() {
    let mut pack = demo_pack();
    pack.swarmflow_script =
        Some("META = {}\nTASK = ''\nasync def run(args):\n    x = open('file')\n".to_string());
    pack.recompute_identity();
    let review = openjiuwen_symphony::flow::Reviewer::new().review(&pack);
    let safety = review
        .checks
        .iter()
        .find(|check| check.check == "code_safety")
        .expect("code_safety present");
    assert_eq!(safety.result, "fail");
    assert!(safety
        .reasons
        .iter()
        .any(|reason| reason.contains("disallowed call `open`")));
}
