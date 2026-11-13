//! Tests for the workflow designer: offline synthesis, the model-assisted path,
//! design validation, and the design-to-DSL conversion.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence.
use super::*;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

/// A model answer holding a valid design that uses Python node-type aliases.
const MODEL_FENCE: &str = r#"Here is the design you asked for.
```json
{"name": "order-flow", "description": "handle an order",
 "nodes": [
   {"id": "start", "type": "Start", "title": "Start", "description": "entry"},
   {"id": "work", "type": "LLM", "title": "Do the work",
    "config": {"instruction": "handle the order"}},
   {"id": "done", "type": "Output", "title": "Done", "description": "exit"}],
 "edges": [{"source": "start", "target": "work"}, {"source": "work", "target": "done"}]}
```"#;

/// A model answer whose design has no start node, so it fails validation.
const INVALID_FENCE: &str = r#"```json
{"name": "broken", "description": "no start node",
 "nodes": [{"id": "work", "type": "llm"}, {"id": "done", "type": "end"}],
 "edges": [{"source": "work", "target": "done"}]}
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

fn sample_design() -> WorkflowDesign {
    WorkflowDesign {
        name: "demo".to_string(),
        description: "a demo design".to_string(),
        nodes: vec![
            node("start", DesignNodeKind::Start),
            node("work", DesignNodeKind::Llm),
            node("end", DesignNodeKind::End),
        ],
        edges: vec![edge("start", "work"), edge("work", "end")],
    }
}

fn llm_nodes(design: &WorkflowDesign) -> Vec<&DesignNode> {
    design
        .nodes
        .iter()
        .filter(|node| node.kind == DesignNodeKind::Llm)
        .collect()
}

fn descriptions(design: &WorkflowDesign) -> Vec<&str> {
    llm_nodes(design)
        .iter()
        .map(|node| node.description.as_str())
        .collect()
}

fn pairs(design: &WorkflowDesign) -> Vec<(&str, &str)> {
    design
        .edges
        .iter()
        .map(|edge| (edge.source.as_str(), edge.target.as_str()))
        .collect()
}

#[test]
fn offline_design_synthesizes_a_single_step() {
    let design = WorkflowDesigner::new()
        .design_offline("Summarize the document.")
        .expect("offline design");

    let ids: Vec<&str> = design.nodes.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(ids, ["start", "llm_1", "end"]);
    assert_eq!(design.nodes[0].kind, DesignNodeKind::Start);
    assert_eq!(design.nodes[1].kind, DesignNodeKind::Llm);
    assert_eq!(design.nodes[2].kind, DesignNodeKind::End);
    assert_eq!(design.nodes[1].title, "Summarize the document");
    assert_eq!(descriptions(&design), ["Summarize the document"]);
    assert_eq!(
        pairs(&design),
        [("start", "llm_1"), ("llm_1", "end")],
        "two sequential edges"
    );
    assert_eq!(design.name, "summarize-the-document");
    assert_eq!(design.description, "Summarize the document.");

    design.validate().expect("design is valid");
    let dsl = design.to_workflow_dsl().expect("design converts");
    dsl.validate().expect("dsl is valid");
    assert_eq!(dsl.nodes[0].node_type, "start");
    assert_eq!(
        dsl.nodes[0].inputs.get("query"),
        Some(&json!({"type": "string", "description": "user input"}))
    );
    assert_eq!(
        dsl.nodes[1].data["instruction"],
        json!("Summarize the document")
    );
}

#[test]
fn offline_design_splits_sequenced_steps() {
    let design = WorkflowDesigner::new()
        .design_offline("Collect the order. Then validate the address. Then generate the label.")
        .expect("offline design");

    assert_eq!(
        descriptions(&design),
        [
            "Collect the order",
            "validate the address",
            "generate the label"
        ]
    );
    assert_eq!(
        pairs(&design),
        [
            ("start", "llm_1"),
            ("llm_1", "llm_2"),
            ("llm_2", "llm_3"),
            ("llm_3", "end")
        ]
    );
    design.validate().expect("design is valid");
}

#[test]
fn offline_design_caps_the_number_of_steps() {
    let design = WorkflowDesigner::new()
        .design_offline("One. Two. Three. Four. Five. Six.")
        .expect("offline design");
    assert_eq!(llm_nodes(&design).len(), MAX_STEPS);
    assert_eq!(design.edges.len(), MAX_STEPS + 1);
    design.validate().expect("design is valid");
}

#[test]
fn offline_design_splits_numbered_steps() {
    let design = WorkflowDesigner::new()
        .design_offline("1. Collect the order. 2) Validate the address. 3、Generate the label.")
        .expect("offline design");
    assert_eq!(
        descriptions(&design),
        [
            "Collect the order",
            "Validate the address",
            "Generate the label"
        ]
    );
    design.validate().expect("design is valid");
}

#[test]
fn offline_design_dedupes_repeated_steps() {
    let design = WorkflowDesigner::new()
        .design_offline("Check the order. Check the order. Send it.")
        .expect("offline design");
    assert_eq!(descriptions(&design), ["Check the order", "Send it"]);
}

#[test]
fn offline_design_adds_a_branch_node() {
    let design = WorkflowDesigner::new()
        .design_offline(
            "Check the request. If the order is valid, confirm it; otherwise reject it.",
        )
        .expect("offline design");

    let branches: Vec<&DesignNode> = design
        .nodes
        .iter()
        .filter(|node| node.kind == DesignNodeKind::Branch)
        .collect();
    assert_eq!(branches.len(), 1, "exactly one branch node");
    assert_eq!(branches[0].id, "branch");
    assert_eq!(branches[0].title, "Decide");
    assert!(
        design
            .edges
            .iter()
            .any(|edge| edge.source == "branch" && edge.target == "end"),
        "the branch leads to the end node"
    );
    assert_eq!(llm_nodes(&design).len(), 3);
    design.validate().expect("design is valid");
}

#[test]
fn offline_design_does_not_branch_on_unrelated_words() {
    let design = WorkflowDesigner::new()
        .design_offline("Classify the request.")
        .expect("offline design");
    assert!(design
        .nodes
        .iter()
        .all(|node| node.kind != DesignNodeKind::Branch));
    assert_eq!(design.nodes.len(), 3);
}

#[test]
fn offline_design_handles_cjk_requests() {
    let design = WorkflowDesigner::new()
        .design_offline("创建客服助手。然后回答问题。")
        .expect("offline design");

    let steps = llm_nodes(&design);
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].description, "创建客服助手");
    assert_eq!(steps[1].description, "回答问题");
    assert_ne!(steps[0].title, "");
    assert_ne!(steps[1].title, "");
    assert_eq!(design.name, "创建客服助手");
    design.validate().expect("design is valid");
}

#[test]
fn offline_design_falls_back_to_one_step_and_a_generic_name() {
    let design = WorkflowDesigner::new()
        .design_offline("...")
        .expect("offline design");
    assert_eq!(descriptions(&design), ["..."]);
    assert_eq!(llm_nodes(&design)[0].title, "Step 1");
    assert_eq!(design.name, "workflow");
    design.validate().expect("design is valid");
}

#[test]
fn offline_design_is_deterministic() {
    let designer = WorkflowDesigner::new();
    let first = designer
        .design_offline("Collect the order. Then send it.")
        .expect("offline design");
    let second = designer
        .design_offline("Collect the order. Then send it.")
        .expect("offline design");
    assert_eq!(first, second);
}

#[test]
fn offline_design_rejects_a_blank_query() {
    let error = WorkflowDesigner::new()
        .design_offline("   \n\t ")
        .expect_err("blank query");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
}

#[tokio::test]
async fn blank_query_never_reaches_the_model() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let designer = WorkflowDesigner::with_model(dynamic(&model));

    let error = designer.design("  ").await.expect_err("blank query");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn design_returns_the_model_document() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let designer = WorkflowDesigner::with_model(dynamic(&model));

    let design = designer.design("handle an order").await.expect("design");
    assert_eq!(design.name, "order-flow");
    assert_eq!(design.description, "handle an order");
    assert_eq!(design.nodes.len(), 3);
    assert_eq!(design.nodes[1].kind, DesignNodeKind::Llm);
    assert_eq!(design.nodes[2].kind, DesignNodeKind::End);
    assert_eq!(
        design.nodes[1].config["instruction"],
        json!("handle the order")
    );
    design.validate().expect("design is valid");

    assert_eq!(model.call_count(), 1);
    let call = model.last_call().expect("one call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), DESIGN_SYSTEM_PROMPT);
    assert_eq!(call.messages[1].text(), "handle an order");
}

#[tokio::test]
async fn model_problems_fall_back_to_the_offline_designer() {
    let query = "Collect the order.";
    let offline = WorkflowDesigner::new()
        .design_offline(query)
        .expect("offline design");
    let scenarios = [
        vec![ScriptedTurn::text(INVALID_FENCE)],
        vec![ScriptedTurn::text("which workflow do you want?")],
        vec![ScriptedTurn::text("```json\n{not: valid}\n```")],
        vec![ScriptedTurn::fail("boom")],
    ];

    for turns in scenarios {
        let model = Arc::new(ScriptedModel::new(turns));
        let designer = WorkflowDesigner::with_model(dynamic(&model));
        let design = designer.design(query).await.expect("fallback design");
        assert_eq!(design, offline, "the offline design replaces the answer");
        assert_eq!(model.call_count(), 1);
    }
}

#[test]
fn node_kind_parses_wire_names_and_aliases() {
    let cases = [
        ("start", DesignNodeKind::Start),
        ("BEGIN", DesignNodeKind::Start),
        ("Entry", DesignNodeKind::Start),
        ("llm", DesignNodeKind::Llm),
        ("LLM", DesignNodeKind::Llm),
        ("llm_node", DesignNodeKind::Llm),
        ("llm-nodes", DesignNodeKind::Llm),
        ("LLMNode", DesignNodeKind::Llm),
        ("Questioner", DesignNodeKind::Llm),
        ("generate", DesignNodeKind::Llm),
        ("branch", DesignNodeKind::Branch),
        ("Condition", DesignNodeKind::Branch),
        ("conditions", DesignNodeKind::Branch),
        ("router", DesignNodeKind::Branch),
        ("switch", DesignNodeKind::Branch),
        ("decision", DesignNodeKind::Branch),
        ("IntentDetection", DesignNodeKind::Branch),
        ("tool", DesignNodeKind::Tool),
        ("Plugin", DesignNodeKind::Tool),
        ("Code", DesignNodeKind::Tool),
        ("api", DesignNodeKind::Tool),
        ("function", DesignNodeKind::Tool),
        ("end", DesignNodeKind::End),
        ("Output", DesignNodeKind::End),
        ("Finish", DesignNodeKind::End),
        ("Exit", DesignNodeKind::End),
        ("end_node", DesignNodeKind::End),
    ];
    for (value, expected) in cases {
        assert_eq!(
            DesignNodeKind::from_str(value).expect("known kind"),
            expected,
            "alias {value}"
        );
        assert_eq!(
            value.parse::<DesignNodeKind>().expect("known kind"),
            expected,
            "alias {value} through FromStr"
        );
    }
}

#[test]
fn node_kind_rejects_unknown_names() {
    for value in ["", "widget", "start-node"] {
        let error = DesignNodeKind::from_str(value).expect_err("unknown kind");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("unknown design node type"));
        assert!(error.to_string().contains(value));
        assert!(value.parse::<DesignNodeKind>().is_err());
        assert!(DesignNodeKind::try_from(value.to_string()).is_err());
    }
}

#[test]
fn node_kind_round_trips_through_display() {
    for kind in [
        DesignNodeKind::Start,
        DesignNodeKind::Llm,
        DesignNodeKind::Branch,
        DesignNodeKind::Tool,
        DesignNodeKind::End,
    ] {
        assert_eq!(kind.as_str(), kind.to_string());
        assert_eq!(
            kind.to_string()
                .parse::<DesignNodeKind>()
                .expect("round trip"),
            kind
        );
        assert_eq!(String::from(kind), kind.as_str());
    }
    assert_eq!(DesignNodeKind::default(), DesignNodeKind::Start);
}

#[test]
fn node_serializes_the_lowercase_kind() {
    let node: DesignNode =
        serde_json::from_str(r#"{"id": "n1", "type": "Output", "title": "Out"}"#).expect("parse");
    assert_eq!(node.kind, DesignNodeKind::End);

    let value = serde_json::to_value(&node).expect("serialize");
    assert_eq!(value["type"], json!("end"));
    assert_eq!(value["id"], json!("n1"));
    assert_eq!(value["config"], Value::Null);
    assert_eq!(
        serde_json::from_value::<DesignNode>(value).expect("reparse"),
        node
    );

    let defaulted: DesignNode = serde_json::from_str(r#"{"id": "n2"}"#).expect("parse");
    assert_eq!(defaulted.kind, DesignNodeKind::Start);
}

#[test]
fn design_round_trips_through_json() {
    let design = sample_design();
    let text = serde_json::to_string(&design).expect("serialize");
    let value: Value = serde_json::from_str(&text).expect("reparse");
    assert_eq!(value["nodes"][1]["type"], json!("llm"));
    assert!(value["edges"][0]["condition"].is_null());
    let parsed: WorkflowDesign = serde_json::from_str(&text).expect("parse");
    assert_eq!(parsed, design);
}

#[test]
fn to_workflow_dsl_maps_nodes_and_config() {
    let mut design = sample_design();
    design.nodes[0].config = json!({"inputs": {"query": {"type": "string"}}});
    design.nodes[1].description = "do the work".to_string();
    design.nodes[1].config = json!({
        "instruction": "do the work",
        "inputs": {"question": {"type": "string"}},
        "outputs": {"result": {"type": "string"}},
    });
    design.edges[1].condition = Some("done".to_string());

    let dsl = design.to_workflow_dsl().expect("converts");
    assert_eq!(dsl.name, "demo");
    assert_eq!(dsl.description, "a demo design");
    let types: Vec<&str> = dsl
        .nodes
        .iter()
        .map(|node| node.node_type.as_str())
        .collect();
    assert_eq!(types, ["start", "llm", "end"]);
    assert_eq!(dsl.nodes[1].description, "do the work");
    assert_eq!(
        dsl.nodes[1].inputs,
        BTreeMap::from([("question".to_string(), json!({"type": "string"}))])
    );
    assert_eq!(
        dsl.nodes[1].outputs,
        BTreeMap::from([("result".to_string(), json!({"type": "string"}))])
    );
    assert_eq!(dsl.nodes[1].data, design.nodes[1].config);
    assert!(dsl.nodes[2].inputs.is_empty());
    assert_eq!(dsl.edges[1].condition.as_deref(), Some("done"));
    dsl.validate().expect("dsl is valid");
}

#[test]
fn validate_rejects_structural_errors() {
    let error = WorkflowDesign::default().validate().expect_err("no nodes");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("workflow design has no nodes"));

    let mut duplicate = sample_design();
    duplicate.nodes[1].id = "end".to_string();
    duplicate.edges.clear();
    let error = duplicate.validate().expect_err("duplicate ids");
    assert!(error.to_string().contains("duplicate node id: end"));

    let mut dangling = sample_design();
    dangling.edges[1].target = "ghost".to_string();
    let error = dangling.validate().expect_err("dangling edge");
    assert!(error
        .to_string()
        .contains("edge references unknown node: ghost"));

    let mut no_start = sample_design();
    no_start.nodes[0].kind = DesignNodeKind::Tool;
    let error = no_start.validate().expect_err("no start node");
    assert!(error
        .to_string()
        .contains("workflow design needs exactly one start node"));

    let mut two_starts = sample_design();
    two_starts.nodes[1].kind = DesignNodeKind::Start;
    let error = two_starts.validate().expect_err("two start nodes");
    assert!(error
        .to_string()
        .contains("workflow design needs exactly one start node"));

    let mut no_end = sample_design();
    no_end.nodes[2].kind = DesignNodeKind::Tool;
    let error = no_end.validate().expect_err("no end node");
    assert!(error
        .to_string()
        .contains("workflow design needs an end node"));

    sample_design().validate().expect("the sample is valid");
}

#[test]
fn to_workflow_dsl_rejects_an_invalid_design() {
    let mut design = sample_design();
    design.nodes.pop();
    let error = design.to_workflow_dsl().expect_err("invalid design");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn debug_reports_whether_a_model_is_configured() {
    assert_eq!(
        format!("{:?}", WorkflowDesigner::new()),
        "WorkflowDesigner { has_model: false }"
    );
    assert_eq!(
        format!("{:?}", WorkflowDesigner::default()),
        "WorkflowDesigner { has_model: false }"
    );

    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let wired = WorkflowDesigner::with_model(dynamic(&model));
    assert_eq!(format!("{wired:?}"), "WorkflowDesigner { has_model: true }");
}
