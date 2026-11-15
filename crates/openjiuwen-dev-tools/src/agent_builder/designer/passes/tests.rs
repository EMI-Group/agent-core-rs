//! Tests for the three designer passes: the basic structure, the branch
//! insertion, the reflection repairs, the whole pipeline, and its vocabulary.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock or generated-id dependence.

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

use super::super::MAX_STEPS;
use super::*;
use crate::agent_builder::cycle_checker::has_cycle;
use crate::agent_builder::progress::ProgressStage;

/// A valid basic-pass design answer.
const BASIC_FENCE: &str = r#"```json
{"name": "model-basic", "description": "the basic design",
 "nodes": [
   {"id": "start", "type": "start", "title": "Start", "description": "entry"},
   {"id": "work", "type": "llm", "title": "Do the work",
    "config": {"instruction": "do the work"}},
   {"id": "end", "type": "end", "title": "End", "description": "exit"}],
 "edges": [{"source": "start", "target": "work"}, {"source": "work", "target": "end"}]}
```"#;

/// A valid branch-pass design answer holding one branch node.
const BRANCH_FENCE: &str = r#"```json
{"name": "model-branch", "description": "the branch design",
 "nodes": [
   {"id": "start", "type": "start", "title": "Start", "description": "entry"},
   {"id": "work", "type": "llm", "title": "Do the work",
    "config": {"instruction": "do the work"}},
   {"id": "branch", "type": "branch", "title": "Decide", "description": "choose",
    "config": {"conditions": ["condition", "default"]}},
   {"id": "end", "type": "end", "title": "End", "description": "exit"}],
 "edges": [{"source": "start", "target": "work"},
           {"source": "work", "target": "branch"},
           {"source": "branch", "target": "end"}]}
```"#;

/// A valid reflection-pass design answer.
const REFLECTION_FENCE: &str = r#"```json
{"name": "model-reflection", "description": "the final design",
 "nodes": [
   {"id": "start", "type": "start", "title": "Start", "description": "entry"},
   {"id": "work", "type": "llm", "title": "Do the work",
    "config": {"instruction": "do the work"}},
   {"id": "end", "type": "end", "title": "End", "description": "exit"}],
 "edges": [{"source": "start", "target": "work"}, {"source": "work", "target": "end"}]}
```"#;

/// Coerces a scripted model into the trait object the designer takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

fn node(id: &str, kind: DesignNodeKind) -> DesignNode {
    DesignNode {
        id: id.to_string(),
        kind,
        ..DesignNode::default()
    }
}

fn edge(source: &str, target: &str) -> DesignEdge {
    DesignEdge {
        source: source.to_string(),
        target: target.to_string(),
        condition: None,
    }
}

fn design(nodes: Vec<DesignNode>, edges: Vec<DesignEdge>) -> WorkflowDesign {
    WorkflowDesign {
        name: "demo".to_string(),
        description: "a demo design".to_string(),
        nodes,
        edges,
    }
}

fn llm_nodes(design: &WorkflowDesign) -> Vec<&DesignNode> {
    design
        .nodes
        .iter()
        .filter(|node| node.kind == DesignNodeKind::Llm)
        .collect()
}

fn pairs(design: &WorkflowDesign) -> Vec<(String, String)> {
    design
        .edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect()
}

fn has_pair(design: &WorkflowDesign, source: &str, target: &str) -> bool {
    pairs(design)
        .iter()
        .any(|pair| pair.0 == source && pair.1 == target)
}

/// The id of the last LLM node, which the branch pass rewires.
fn last_llm(design: &WorkflowDesign) -> String {
    llm_nodes(design)
        .last()
        .map_or_else(|| "start".to_string(), |node| node.id.clone())
}

fn reflect(design: &WorkflowDesign) -> (WorkflowDesign, Vec<String>) {
    reflection_design(design).expect("reflection")
}

#[test]
fn basic_pass_synthesizes_a_single_step() {
    let designer = WorkflowDesigner::new();
    let design = designer
        .design_basic_offline("Summarize the document.")
        .expect("basic design");

    let ids: Vec<&str> = design.nodes.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(ids, ["start", "llm_1", "end"]);
    assert_eq!(
        pairs(&design),
        [
            ("start".to_string(), "llm_1".to_string()),
            ("llm_1".to_string(), "end".to_string())
        ]
    );
    assert_eq!(llm_nodes(&design)[0].title, "Summarize the document");
    assert_eq!(
        llm_nodes(&design)[0].config["instruction"],
        json!("Summarize the document")
    );
    assert_eq!(design.name, "summarize-the-document");
    assert_eq!(design.description, "Summarize the document.");
    design.validate().expect("design is valid");
}

#[test]
fn basic_pass_splits_sequenced_steps() {
    let design = WorkflowDesigner::new()
        .design_basic_offline(
            "Collect the order. Then validate the address. Then generate the label.",
        )
        .expect("basic design");

    let steps = llm_nodes(&design);
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[2].description, "generate the label");
    assert_eq!(
        pairs(&design),
        [
            ("start".to_string(), "llm_1".to_string()),
            ("llm_1".to_string(), "llm_2".to_string()),
            ("llm_2".to_string(), "llm_3".to_string()),
            ("llm_3".to_string(), "end".to_string())
        ]
    );
    design.validate().expect("design is valid");
}

#[test]
fn basic_pass_caps_the_step_count() {
    let design = WorkflowDesigner::new()
        .design_basic_offline("One. Two. Three. Four. Five. Six.")
        .expect("basic design");
    assert_eq!(llm_nodes(&design).len(), MAX_STEPS);
    assert_eq!(design.edges.len(), MAX_STEPS + 1);
    design.validate().expect("design is valid");
}

#[test]
fn basic_pass_never_inserts_a_branch() {
    let queries = [
        "If the order is valid, confirm it; otherwise reject it.",
        "当订单有效时确认，否则拒绝。",
        "Check whether the address is complete.",
    ];
    for query in queries {
        let design = WorkflowDesigner::new()
            .design_basic_offline(query)
            .expect("basic design");
        assert!(
            design
                .nodes
                .iter()
                .all(|node| node.kind != DesignNodeKind::Branch),
            "query {query} has no branch node"
        );
        assert!(design.nodes.iter().all(|node| node.id != "branch"));
        design.validate().expect("design is valid");
    }
}

#[test]
fn basic_pass_handles_cjk_requests() {
    let design = WorkflowDesigner::new()
        .design_basic_offline("创建客服助手。然后回答问题。")
        .expect("basic design");

    let steps = llm_nodes(&design);
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].description, "创建客服助手");
    assert_eq!(steps[1].description, "回答问题");
    assert_ne!(steps[0].title, "");
    assert_eq!(design.name, "创建客服助手");
    design.validate().expect("design is valid");
}

#[test]
fn basic_pass_rejects_a_blank_query() {
    let error = WorkflowDesigner::new()
        .design_basic_offline("   \n\t ")
        .expect_err("blank query");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
}

#[tokio::test]
async fn offline_passes_never_call_the_model() {
    let model = Arc::new(ScriptedModel::text(BASIC_FENCE));
    let designer = WorkflowDesigner::with_model(dynamic(&model));
    let query = "Check the order. If it is valid, confirm it.";

    let basic = designer.design_basic_offline(query).expect("basic design");
    let branched = designer
        .design_branch_offline(query, &basic)
        .expect("branch design");
    let (repaired, notes) = designer
        .design_reflection_offline(query, &branched)
        .expect("reflection");
    assert_eq!(notes.len(), 0);
    assert_eq!(repaired, branched);
    assert_eq!(model.call_count(), 0);
}

#[test]
fn branch_pass_inserts_a_branch_for_every_marker_class() {
    let designer = WorkflowDesigner::new();
    let queries = [
        "Collect the order. If the order is valid, confirm it.",
        "检查订单。当订单有效时确认。",
        "Validate the address. Whether the address is complete decides.",
        "Check the stock. Otherwise reject the order.",
        "Check the order. Else refund it.",
    ];

    for query in queries {
        let basic = designer.design_basic_offline(query).expect("basic design");
        let design = designer
            .design_branch_offline(query, &basic)
            .expect("branch design");

        let branches: Vec<&DesignNode> = design
            .nodes
            .iter()
            .filter(|node| node.kind == DesignNodeKind::Branch)
            .collect();
        assert_eq!(branches.len(), 1, "query {query}");
        assert_eq!(branches[0].id, "branch");
        assert_ne!(branches[0].title, "");
        assert_eq!(
            branches[0].config["conditions"],
            json!(["condition", "default"])
        );

        let previous = last_llm(&basic);
        assert!(
            has_pair(&design, &previous, "branch"),
            "query {query} rewires {previous} through the branch"
        );
        assert!(has_pair(&design, "branch", "end"));
        assert!(!has_pair(&design, &previous, "end"));
        design.validate().expect("design is valid");
    }
}

#[test]
fn branch_pass_keeps_the_basic_design_without_markers() {
    let designer = WorkflowDesigner::new();
    let query = "Collect the order. Then ship it.";
    let basic = designer.design_basic_offline(query).expect("basic design");
    let design = designer
        .design_branch_offline(query, &basic)
        .expect("branch design");
    assert_eq!(design, basic);
}

#[test]
fn branch_pass_is_idempotent() {
    let designer = WorkflowDesigner::new();
    let query = "Check the order. If it is valid, confirm it.";
    let basic = designer.design_basic_offline(query).expect("basic design");
    let once = designer
        .design_branch_offline(query, &basic)
        .expect("branch design");
    let twice = designer
        .design_branch_offline(query, &once)
        .expect("branch design");

    assert_eq!(once, twice);
    assert_eq!(
        once.nodes
            .iter()
            .filter(|node| node.kind == DesignNodeKind::Branch)
            .count(),
        1
    );
    assert_eq!(
        once.nodes.iter().filter(|node| node.id == "branch").count(),
        1
    );
}

#[test]
fn branch_pass_reports_its_detail() {
    let query = "Check the order. If it is valid, confirm it.";
    let basic = basic_design(query).expect("basic design");

    let (design, detail) = branch_design(query, &basic);
    assert!(detail.contains("decision markers found"));
    assert!(design.nodes.iter().any(|node| node.id == "branch"));

    let plain = basic_design("Collect the order.").expect("basic design");
    let (unchanged, detail) = branch_design("Collect the order.", &plain);
    assert_eq!(unchanged, plain);
    assert!(detail.contains("no decision markers"));

    let (again, detail) = branch_design(query, &design);
    assert_eq!(again, design);
    assert!(detail.contains("already has a branch node"));
}

#[test]
fn reflection_pass_leaves_a_valid_design_unchanged() {
    let design = basic_design("Collect the order. Then ship it.").expect("basic design");
    let (repaired, notes) = reflect(&design);
    assert_eq!(notes.len(), 0);
    assert_eq!(repaired, design);
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_repairs_an_empty_design() {
    let empty = WorkflowDesign::default();
    let (repaired, notes) = reflect(&empty);

    assert_eq!(notes.len(), 2, "one note per inserted node: {notes:?}");
    assert!(notes[0].contains("start node"));
    assert!(notes[1].contains("end node"));
    assert_eq!(repaired.nodes.len(), 2);
    assert!(has_pair(&repaired, "start", "end"));
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_renames_duplicate_ids() {
    let mut design = basic_design("Collect the order.").expect("basic design");
    design.nodes.push(node("llm_1", DesignNodeKind::Llm));

    let (repaired, notes) = reflect(&design);
    assert!(
        notes
            .iter()
            .any(|note| note.contains("renamed the duplicate node id `llm_1` to `llm_1_2`")),
        "notes: {notes:?}"
    );
    assert_eq!(
        repaired
            .nodes
            .iter()
            .filter(|node| node.id == "llm_1_2")
            .count(),
        1
    );
    let mut ids: Vec<&str> = repaired.nodes.iter().map(|node| node.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), repaired.nodes.len(), "every id is unique");
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_drops_dangling_edges() {
    let mut design = basic_design("Collect the order.").expect("basic design");
    design.edges.push(edge("ghost", "end"));

    let (repaired, notes) = reflect(&design);
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("dropped the edge `ghost` -> `end`"));
    assert!(repaired.edges.iter().all(|edge| edge.source != "ghost"));
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_inserts_a_missing_start() {
    let design = design(
        vec![
            node("work", DesignNodeKind::Llm),
            node("end", DesignNodeKind::End),
        ],
        vec![edge("work", "end")],
    );

    let (repaired, notes) = reflect(&design);
    assert_eq!(notes, ["inserted the missing start node `start`"]);
    assert_eq!(repaired.nodes[0].kind, DesignNodeKind::Start);
    assert_eq!(repaired.nodes[0].id, "start");
    assert!(has_pair(&repaired, "start", "work"));
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_appends_a_missing_end() {
    let design = design(
        vec![
            node("start", DesignNodeKind::Start),
            node("work", DesignNodeKind::Llm),
        ],
        vec![edge("start", "work")],
    );

    let (repaired, notes) = reflect(&design);
    assert_eq!(notes, ["appended the missing end node `end`"]);
    assert_eq!(
        repaired.nodes.last().expect("an end node").kind,
        DesignNodeKind::End
    );
    assert!(has_pair(&repaired, "work", "end"));
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_connects_unreachable_nodes() {
    let design = design(
        vec![
            node("start", DesignNodeKind::Start),
            node("work", DesignNodeKind::Llm),
            node("end", DesignNodeKind::End),
            node("ghost", DesignNodeKind::Llm),
        ],
        vec![edge("start", "work"), edge("work", "end")],
    );

    let (repaired, notes) = reflect(&design);
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("connected the unreachable node `ghost` from `end`"));
    assert!(has_pair(&repaired, "end", "ghost"));
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_breaks_a_cycle() {
    let design = design(
        vec![
            node("start", DesignNodeKind::Start),
            node("a", DesignNodeKind::Llm),
            node("b", DesignNodeKind::Llm),
            node("end", DesignNodeKind::End),
        ],
        vec![
            edge("start", "a"),
            edge("a", "b"),
            edge("b", "a"),
            edge("b", "end"),
        ],
    );

    let (repaired, notes) = reflect(&design);
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("removed the back edge `b` -> `a`"));
    assert!(!has_pair(&repaired, "b", "a"));
    repaired.validate().expect("design is valid");
    assert!(!has_cycle(&repaired.to_workflow_dsl().expect("dsl")));
}

#[test]
fn reflection_pass_notes_one_per_repair() {
    let design = design(
        vec![
            node("start", DesignNodeKind::Start),
            node("work", DesignNodeKind::Llm),
        ],
        vec![edge("start", "work"), edge("work", "ghost")],
    );

    let (repaired, notes) = reflect(&design);
    assert_eq!(
        notes.len(),
        2,
        "one dangling edge and one end node: {notes:?}"
    );
    assert!(notes[0].contains("dropped the edge `work` -> `ghost`"));
    assert!(notes[1].contains("appended the missing end node"));
    repaired.validate().expect("design is valid");
}

#[test]
fn reflection_pass_rejects_an_unrepairable_design() {
    let mut design = basic_design("Collect the order.").expect("basic design");
    design.nodes.push(node("start_2", DesignNodeKind::Start));

    let error = WorkflowDesigner::new()
        .design_reflection_offline("Collect the order.", &design)
        .expect_err("two start nodes cannot be repaired");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("exactly one start node"));
}

#[tokio::test]
async fn three_pass_returns_its_records_in_order() {
    let designer = WorkflowDesigner::new();
    let result = designer
        .design_three_pass("Collect the order. Then ship it.")
        .await
        .expect("three pass");

    let kinds: Vec<DesignPassKind> = result.passes.iter().map(|pass| pass.kind).collect();
    assert_eq!(
        kinds,
        [
            DesignPassKind::Basic,
            DesignPassKind::Branch,
            DesignPassKind::Reflection
        ]
    );
    assert!(result
        .passes
        .iter()
        .all(|pass| pass.source == GenerationSource::Deterministic));
    assert_eq!(result.provenance.len(), 3);
    assert_eq!(
        result.provenance.stages[0].stage,
        ProgressStage::GeneratingWorkflowDesign
    );
    assert_eq!(
        result.provenance.stages[1].stage,
        ProgressStage::GeneratingWorkflowDesign
    );
    assert_eq!(
        result.provenance.stages[2].stage,
        ProgressStage::ValidatingDl
    );
    assert_eq!(result.passes[2].node_count, result.design.nodes.len());
    assert_eq!(result.passes[2].edge_count, result.design.edges.len());
    assert_eq!(result.notes.len(), 0);
    result.design.validate().expect("design is valid");
}

#[tokio::test]
async fn three_pass_adds_a_branch_for_a_decision_request() {
    let result = WorkflowDesigner::new()
        .design_three_pass("Check the order. If it is valid, confirm it.")
        .await
        .expect("three pass");

    assert!(result.design.nodes.iter().any(|node| node.id == "branch"));
    assert_eq!(result.passes[1].node_count, result.passes[0].node_count + 1);
    assert_eq!(
        result.passes[1].detail,
        "decision markers found; inserted the `branch` node before the end node"
    );
    assert!(result.passes[0].detail.ends_with("steps synthesised"));
    assert_eq!(
        result.passes[2].detail,
        "the design validated without repair"
    );
    result.design.validate().expect("design is valid");
}

#[tokio::test]
async fn three_pass_is_deterministic() {
    let designer = WorkflowDesigner::new();
    let query = "Collect the order. Then ship it.";
    let first = designer.design_three_pass(query).await.expect("three pass");
    let second = designer.design_three_pass(query).await.expect("three pass");
    assert_eq!(first, second);
}

#[tokio::test]
async fn three_pass_uses_the_model_for_every_pass() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(BASIC_FENCE),
        ScriptedTurn::text(BRANCH_FENCE),
        ScriptedTurn::text(REFLECTION_FENCE),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));
    let query = "Check the order. If it is valid, confirm it.";

    let result = designer.design_three_pass(query).await.expect("three pass");
    assert!(result
        .passes
        .iter()
        .all(|pass| pass.source == GenerationSource::Model));
    assert_eq!(
        result
            .provenance
            .stages_matching(GenerationSource::Fallback)
            .len(),
        0
    );
    assert_eq!(result.design.name, "model-reflection");
    result.design.validate().expect("design is valid");

    assert_eq!(model.call_count(), 3);
    let calls = model.calls();
    assert!(calls[0].messages[0].text().contains("basic design pass"));
    assert!(calls[0].messages[1].text().contains(query));
    assert!(calls[1].messages[0].text().contains("branch design pass"));
    assert!(calls[1].messages[1].text().contains("## Basic design"));
    assert!(calls[2].messages[0].text().contains("reflection pass"));
    assert!(calls[2].messages[1].text().contains("## Workflow design"));
}

#[tokio::test]
async fn three_pass_falls_back_per_pass() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text("which workflow do you want?"),
        ScriptedTurn::text("```json\n{not: valid}\n```"),
        ScriptedTurn::fail("boom"),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));
    let query = "Check the order. If it is valid, confirm it.";

    let result = designer.design_three_pass(query).await.expect("three pass");
    assert!(result
        .passes
        .iter()
        .all(|pass| pass.source == GenerationSource::Fallback));
    assert!(result
        .passes
        .iter()
        .all(|pass| pass.detail.starts_with("model answer rejected: ")));
    assert_eq!(result.provenance.fallback_count(), 3);
    assert_eq!(model.call_count(), 3);

    let offline = WorkflowDesigner::new()
        .design_three_pass(query)
        .await
        .expect("offline three pass");
    assert_eq!(
        result.design, offline.design,
        "the deterministic design survives"
    );
    assert!(result.design.nodes.iter().any(|node| node.id == "branch"));
}

#[tokio::test]
async fn three_pass_rejects_a_branch_answer_without_a_branch() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(BASIC_FENCE),
        ScriptedTurn::text(BASIC_FENCE),
        ScriptedTurn::text(REFLECTION_FENCE),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));
    let query = "Check the order. If it is valid, confirm it.";

    let result = designer.design_three_pass(query).await.expect("three pass");
    let sources: Vec<GenerationSource> = result.passes.iter().map(|pass| pass.source).collect();
    assert_eq!(
        sources,
        [
            GenerationSource::Model,
            GenerationSource::Fallback,
            GenerationSource::Model
        ]
    );
    assert!(result.passes[1].detail.contains("no branch node"));
    assert_eq!(
        result.passes[1].node_count,
        result.passes[0].node_count + 1,
        "the deterministic branch design replaces the rejected answer"
    );
    assert_eq!(
        result.design.name, "model-reflection",
        "the reflection answer is accepted"
    );
    assert_eq!(model.call_count(), 3);
}

#[tokio::test]
async fn three_pass_rejects_an_invalid_reflection_answer() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(BASIC_FENCE),
        ScriptedTurn::text(BRANCH_FENCE),
        ScriptedTurn::text("no fence here"),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));

    let result = designer
        .design_three_pass("Check the order. If it is valid, confirm it.")
        .await
        .expect("three pass");
    assert_eq!(result.passes[2].source, GenerationSource::Fallback);
    assert!(result.passes[2].detail.contains("no JSON code fence"));
    assert_eq!(
        result.design.name, "model-branch",
        "the merged design survives"
    );
    result.design.validate().expect("design is valid");
    assert_eq!(model.call_count(), 3);
}

#[tokio::test]
async fn three_pass_rejects_a_blank_query_without_a_model_call() {
    let model = Arc::new(ScriptedModel::text(BASIC_FENCE));
    let designer = WorkflowDesigner::with_model(dynamic(&model));

    let error = designer
        .design_three_pass("   ")
        .await
        .expect_err("blank query");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
    assert_eq!(model.call_count(), 0);
}

#[test]
fn pass_kind_serializes_its_wire_name() {
    for (kind, wire) in [
        (DesignPassKind::Basic, "basic"),
        (DesignPassKind::Branch, "branch"),
        (DesignPassKind::Reflection, "reflection"),
    ] {
        assert_eq!(kind.as_str(), wire);
        assert_eq!(serde_json::to_value(kind).expect("serialize"), json!(wire));
        assert_eq!(
            serde_json::from_value::<DesignPassKind>(json!(wire)).expect("parse"),
            kind
        );
    }
    assert_eq!(DesignPassKind::default(), DesignPassKind::Basic);
}

#[test]
fn design_pass_round_trips_and_tolerates_missing_fields() {
    let pass = DesignPass {
        kind: DesignPassKind::Branch,
        source: GenerationSource::Fallback,
        detail: "model answer rejected: boom".to_string(),
        node_count: 4,
        edge_count: 5,
    };
    let value = serde_json::to_value(&pass).expect("serialize");
    assert_eq!(value["kind"], json!("branch"));
    assert_eq!(value["source"], json!("fallback"));
    assert_eq!(
        serde_json::from_value::<DesignPass>(value).expect("parse"),
        pass
    );

    let defaulted: DesignPass = serde_json::from_str(r#"{"kind": "reflection"}"#).expect("parse");
    assert_eq!(defaulted.kind, DesignPassKind::Reflection);
    assert_eq!(defaulted.source, GenerationSource::Model);
    assert_eq!(defaulted.node_count, 0);
    assert_eq!(defaulted.detail, "");
}

#[test]
fn three_pass_design_round_trips_through_json() {
    let design = basic_design("Collect the order.").expect("basic design");
    let result = ThreePassDesign {
        design: design.clone(),
        passes: vec![DesignPass {
            kind: DesignPassKind::Basic,
            source: GenerationSource::Deterministic,
            detail: "1 step synthesised".to_string(),
            node_count: design.nodes.len(),
            edge_count: design.edges.len(),
        }],
        provenance: GenerationProvenance::default(),
        notes: Vec::new(),
    };

    let text = serde_json::to_string(&result).expect("serialize");
    let parsed: ThreePassDesign = serde_json::from_str(&text).expect("parse");
    assert_eq!(parsed, result);

    let partial: ThreePassDesign = serde_json::from_str(r#"{"notes": ["kept"]}"#).expect("parse");
    assert_eq!(partial.notes, ["kept"]);
    assert_eq!(partial.passes.len(), 0);
    assert_eq!(partial.design.nodes.len(), 0);
}
