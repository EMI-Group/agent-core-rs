//! Tests for the DL generation stage: the deterministic conversion, the model
//! path with its fallbacks, the candidate form the repair loop consumes, and
//! the refine entry point.
//!
//! Every case is offline and deterministic: a `ScriptedModel` answers the model
//! calls, ids and timestamps are fixed, and no test reads the environment or a
//! clock.

use std::collections::BTreeMap;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::{json, Value};

use super::*;
use crate::agent_builder::designer::WorkflowDesigner;
use crate::agent_builder::outcome::{GenerationSource, StageProvenance};
use crate::agent_builder::workflow_dsl::{DslEdge, DslNode};

/// The request the offline designer is fed in the deterministic tests.
const QUERY: &str = "Collect the order. Then confirm it.";

/// A model answer holding a valid two-node workflow document.
const MODEL_FENCE: &str = r#"```json
{"name": "model-name", "description": "a model workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"}, {"id": "end", "type": "end"}],
 "edges": [{"source": "start", "target": "end"}]}
```"#;

/// Coerces a scripted model into the trait object the generator takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// A design produced by the offline designer.
fn design() -> WorkflowDesign {
    WorkflowDesigner::new()
        .design_offline(QUERY)
        .expect("the offline designer accepts the query")
}

/// A two-node, one-edge node list: `start` -> `end`.
fn demo_nodes() -> Vec<DslNode> {
    vec![
        DslNode {
            id: "start".to_string(),
            node_type: "start".to_string(),
            title: "Start".to_string(),
            description: "entry".to_string(),
            inputs: BTreeMap::from([("query".to_string(), json!({"type": "string"}))]),
            outputs: BTreeMap::new(),
            data: Value::Null,
        },
        DslNode {
            id: "end".to_string(),
            node_type: "end".to_string(),
            title: "End".to_string(),
            description: "exit".to_string(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            data: Value::Null,
        },
    ]
}

/// A valid two-node workflow specification.
fn demo_spec() -> WorkflowSpec {
    WorkflowSpec {
        name: "demo".to_string(),
        description: "a demo workflow".to_string(),
        nodes: demo_nodes(),
        edges: vec![DslEdge {
            source: "start".to_string(),
            target: "end".to_string(),
            condition: None,
        }],
    }
}

#[test]
fn deterministic_generation_matches_the_design_conversion() {
    let design = design();
    let generator = DlGenerator::new();

    let document = generator.generate(&design).expect("generate");
    assert_eq!(document, design.to_workflow_dsl().expect("convert"));
    document.validate().expect("the document is valid");
    assert!(document.node("start").is_some(), "the start node survives");
    assert_eq!(document.name, design.name);

    let spec = WorkflowSpec {
        name: document.name.clone(),
        description: document.description.clone(),
        nodes: document.nodes.clone(),
        edges: document.edges.clone(),
    };
    assert_eq!(
        generator.from_spec(&spec).expect("the spec is valid"),
        document
    );
}

#[test]
fn generate_rejects_an_invalid_design() {
    let generator = DlGenerator::new();
    let error = generator
        .generate(&WorkflowDesign::default())
        .expect_err("an empty design is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn from_spec_keeps_the_document_and_validates_it() {
    let generator = DlGenerator::new();
    let spec = demo_spec();

    let document = generator.from_spec(&spec).expect("the spec is valid");
    assert_eq!(document.name, "demo");
    assert_eq!(document.description, "a demo workflow");
    assert_eq!(document.nodes, spec.nodes);
    assert_eq!(document.edges, spec.edges);
    assert_eq!(document, spec.to_dsl());

    let mut dangling = demo_spec();
    dangling.edges[0].target = "ghost".to_string();
    let error = generator
        .from_spec(&dangling)
        .expect_err("a dangling edge is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("ghost"));
}

#[tokio::test]
async fn model_document_replaces_the_baseline_when_it_validates() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let generator = DlGenerator::with_model(dynamic(&model));
    let design = design();

    let document = generator
        .generate_with_model(&design)
        .await
        .expect("the model path succeeds");
    assert_eq!(document.name, "model-name");
    document.validate().expect("the model document is valid");

    assert_eq!(model.call_count(), 1);
    let call = model.last_call().expect("one call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), DL_GENERATOR_SYSTEM_PROMPT);
    assert!(
        call.messages[1].text().contains(&design.name),
        "the user message carries the design"
    );
    assert_eq!(call.request.model_name, "scripted-model");
}

#[tokio::test]
async fn invalid_model_documents_fall_back_to_the_baseline() {
    let fences = [
        r#"```json
{"name": "dangling", "description": "broken",
 "nodes": [{"id": "start", "type": "start"}],
 "edges": [{"source": "start", "target": "ghost"}]}
```"#,
        r#"```json
{"name": "empty", "description": "no nodes", "nodes": [], "edges": []}
```"#,
    ];

    for fence in fences {
        let model = Arc::new(ScriptedModel::text(fence));
        let generator = DlGenerator::with_model(dynamic(&model));
        let design = design();

        let document = generator
            .generate_with_model(&design)
            .await
            .expect("model problems never fail generation");
        assert_eq!(
            document,
            generator.generate(&design).expect("baseline"),
            "the baseline replaces an invalid model document"
        );
        assert_eq!(model.call_count(), 1);
    }
}

#[tokio::test]
async fn model_errors_fall_back_to_the_baseline() {
    let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")]));
    let generator = DlGenerator::with_model(dynamic(&model));
    let design = design();

    let document = generator
        .generate_with_model(&design)
        .await
        .expect("model errors never fail generation");
    assert_eq!(document, generator.generate(&design).expect("baseline"));
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn an_invalid_design_fails_before_the_model_is_called() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let generator = DlGenerator::with_model(dynamic(&model));

    let error = generator
        .generate_with_model(&WorkflowDesign::default())
        .await
        .expect_err("an empty design is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn a_model_less_generator_returns_the_baseline() {
    let design = design();
    let generator = DlGenerator::default();

    let document = generator
        .generate_with_model(&design)
        .await
        .expect("generate without a model");
    assert_eq!(document, generator.generate(&design).expect("baseline"));
}

#[test]
fn new_and_debug_describe_the_configuration() {
    assert_eq!(
        format!("{:?}", DlGenerator::new()),
        "DlGenerator { has_model: false }"
    );
    assert_eq!(
        format!("{:?}", DlGenerator::default()),
        "DlGenerator { has_model: false }"
    );

    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    assert_eq!(
        format!("{:?}", DlGenerator::with_model(dynamic(&model))),
        "DlGenerator { has_model: true }"
    );
}

// -----------------------------------------------------------------------
// Candidate mode
// -----------------------------------------------------------------------

/// A model answer whose edge names a node that does not exist, so the document
/// parses but fails [`WorkflowDsl::validate`].
const INVALID_FENCE: &str = r#"```json
{"name": "dangling", "description": "an edge to nowhere",
 "nodes": [{"id": "start", "type": "start", "title": "Start"}],
 "edges": [{"source": "start", "target": "ghost"}]}
```"#;

#[tokio::test]
async fn generate_candidate_returns_an_invalid_model_answer_with_its_issues() {
    let model = Arc::new(ScriptedModel::text(INVALID_FENCE));
    let generator = DlGenerator::with_model(dynamic(&model));
    let design = design();

    let generated = generator
        .generate_candidate(&design)
        .await
        .expect("model problems never fail generation");

    let candidate = generated
        .candidate
        .as_ref()
        .expect("the invalid model document is kept");
    assert_eq!(candidate.name, "dangling");
    assert!(
        candidate.validate().is_err(),
        "the candidate is handed over unvalidated"
    );
    assert_eq!(generated.issues.len(), 1);
    assert!(
        generated.issues[0].contains("edge references unknown node"),
        "{:?}",
        generated.issues[0]
    );
    assert!(generated.issues[0].contains("ghost"));

    assert!(generated.is_model_answer());
    assert!(!generated.is_accepted());
    assert_eq!(
        generated.start(),
        candidate,
        "the candidate is the start point"
    );
    assert_eq!(
        generated.baseline,
        generator.generate(&design).expect("baseline")
    );

    assert_eq!(generated.provenance.len(), 1);
    let record = &generated.provenance.stages[0];
    assert_eq!(record.stage, ProgressStage::GeneratingDl);
    assert_eq!(record.source, GenerationSource::Model);
    assert!(
        record.detail.contains("needs repair"),
        "detail {:?} names the repair",
        record.detail
    );
    assert!(record.detail.contains("ghost"));

    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn generate_candidate_reports_why_no_candidate_exists() {
    let design = design();
    let cases: [(Arc<dyn Model>, bool); 2] = [
        (
            dynamic(&Arc::new(ScriptedModel::text("there is no document here"))),
            false,
        ),
        (
            dynamic(&Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")]))),
            true,
        ),
    ];

    for (model, call_fails) in cases {
        let generator = DlGenerator::with_model(Arc::clone(&model));

        let generated = generator
            .generate_candidate(&design)
            .await
            .expect("model problems never fail generation");

        assert!(generated.candidate.is_none());
        assert_eq!(generated.issues.len(), 0, "{:?}", generated.issues);
        assert!(!generated.is_model_answer());
        assert!(!generated.is_accepted());
        assert_eq!(generated.start(), &generated.baseline);
        assert_eq!(
            generated.baseline,
            generator.generate(&design).expect("baseline")
        );

        assert_eq!(generated.provenance.len(), 1);
        let record = &generated.provenance.stages[0];
        assert_eq!(record.stage, ProgressStage::GeneratingDl);
        assert_eq!(record.source, GenerationSource::Fallback);
        assert!(
            record.detail.contains(if call_fails {
                "boom"
            } else {
                "no JSON code fence"
            }),
            "detail {:?} names the reason",
            record.detail
        );
    }

    let baseline_only = DlGenerator::new()
        .generate_candidate(&design)
        .await
        .expect("a model-less generator still produces a candidate");
    assert!(baseline_only.candidate.is_none());
    assert_eq!(baseline_only.issues.len(), 0, "{:?}", baseline_only.issues);
    assert_eq!(
        baseline_only.provenance.stages[0].source,
        GenerationSource::Deterministic
    );
    assert!(baseline_only.provenance.stages[0]
        .detail
        .contains("no model configured"));
}

#[tokio::test]
async fn generate_candidate_accepts_a_valid_model_answer() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let generator = DlGenerator::with_model(dynamic(&model));
    let design = design();

    let generated = generator
        .generate_candidate(&design)
        .await
        .expect("the model path succeeds");

    assert!(generated.is_model_answer());
    assert!(generated.is_accepted());
    assert_eq!(generated.issues.len(), 0, "{:?}", generated.issues);
    let candidate = generated.candidate.as_ref().expect("a candidate");
    assert_eq!(candidate.name, "model-name");
    candidate.validate().expect("the candidate is valid");
    assert_eq!(generated.provenance.len(), 1);
    assert_eq!(
        generated.provenance.stages[0].source,
        GenerationSource::Model
    );
    assert_eq!(generated.provenance.stages[0].detail, "");
    assert!(!generated.provenance.has_fallback());
    assert_eq!(model.call_count(), 1);

    // The tolerant form keeps the document the candidate path reports.
    let tolerant_model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let tolerant = DlGenerator::with_model(dynamic(&tolerant_model));
    assert_eq!(
        tolerant
            .generate_with_model(&design)
            .await
            .expect("the tolerant form succeeds"),
        *candidate,
        "a valid candidate is what generate_with_model returns"
    );
    assert_eq!(tolerant_model.call_count(), 1);
}

#[test]
fn generated_dl_round_trips_as_json() {
    let mut provenance = GenerationProvenance::new();
    provenance.record(
        ProgressStage::GeneratingDl,
        GenerationSource::Model,
        "model document needs repair: edge references unknown node: ghost",
    );
    let generated = GeneratedDl {
        candidate: Some(WorkflowDsl {
            name: "dangling".to_string(),
            ..WorkflowDsl::default()
        }),
        baseline: demo_spec().to_dsl(),
        issues: vec!["edge references unknown node: ghost".to_string()],
        provenance,
    };

    let value = serde_json::to_value(&generated).expect("serialize");
    assert_eq!(value["candidate"]["name"], "dangling");
    assert_eq!(value["baseline"]["name"], "demo");
    assert_eq!(value["issues"][0], "edge references unknown node: ghost");
    assert_eq!(value["provenance"]["stages"][0]["stage"], "generating_dl");
    let restored: GeneratedDl = serde_json::from_value(value).expect("parse");
    assert_eq!(restored, generated);

    let empty: GeneratedDl = serde_json::from_value(json!({})).expect("parse an empty object");
    assert_eq!(empty, GeneratedDl::default());
    assert!(empty.candidate.is_none());
    assert_eq!(empty.start(), &WorkflowDsl::default());
}

// -----------------------------------------------------------------------
// Refine mode
// -----------------------------------------------------------------------

/// The requested change the refine tests send.
const REFINE_FEEDBACK: &str = "Add a confirmation step after the order is collected.";

/// The Mermaid diagram of the document refine starts from.
const REFINE_MERMAID: &str =
    "graph TD\n  start[Node start: Start]\n  end[Node end: End]\n  start --> end";

/// The document refine starts from.
fn existing() -> WorkflowDsl {
    demo_spec().to_dsl()
}

/// The one provenance record of a refinement.
fn only_record(refinement: &Refinement) -> &StageProvenance {
    assert_eq!(refinement.provenance.len(), 1);
    &refinement.provenance.stages[0]
}

#[tokio::test]
async fn refine_rejects_blank_feedback_before_the_model() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let generator = DlGenerator::with_model(dynamic(&model));
    let current = existing();

    for feedback in ["", "   ", "\n\t "] {
        let error = generator
            .refine(feedback, &current, REFINE_MERMAID)
            .await
            .expect_err("blank feedback is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    assert_eq!(model.call_count(), 0, "no model call for blank feedback");
}

#[tokio::test]
async fn refine_accepts_a_valid_model_document() {
    let model = Arc::new(ScriptedModel::text(MODEL_FENCE));
    let generator = DlGenerator::with_model(dynamic(&model));
    let current = existing();

    let refinement = generator
        .refine(REFINE_FEEDBACK, &current, REFINE_MERMAID)
        .await
        .expect("the model path succeeds");

    assert_eq!(refinement.workflow.name, "model-name");
    refinement
        .workflow
        .validate()
        .expect("the revision is valid");
    let record = only_record(&refinement);
    assert_eq!(record.stage, ProgressStage::RefiningDl);
    assert_eq!(record.source, GenerationSource::Model);
    assert_eq!(record.detail, "");
    assert!(!refinement.provenance.has_fallback());

    assert_eq!(model.call_count(), 1);
    let call = model.last_call().expect("one call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), DL_REFINE_SYSTEM_PROMPT);
    let user = call.messages[1].text();
    assert!(
        user.contains(REFINE_FEEDBACK),
        "the user message carries the feedback"
    );
    assert!(
        user.contains(REFINE_MERMAID),
        "the user message carries the Mermaid diagram"
    );
    let document_json = serde_json::to_string_pretty(&current).expect("serialize");
    assert!(
        user.contains(&document_json),
        "the user message carries the existing document"
    );
    assert!(user.contains("## Existing workflow document"));
    assert_eq!(call.request.model_name, "scripted-model");
}

#[tokio::test]
async fn refine_keeps_the_existing_document_on_unusable_answers() {
    let cases = [
        (
            Arc::new(ScriptedModel::text("there is no document here")),
            "answer is not a workflow document",
        ),
        (
            Arc::new(ScriptedModel::text(
                r#"```json
{"name": "dangling", "description": "broken",
 "nodes": [{"id": "start", "type": "start"}],
 "edges": [{"source": "start", "target": "ghost"}]}
```"#,
            )),
            "revised document is invalid",
        ),
        (
            Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")])),
            "model call failed",
        ),
    ];

    let current = existing();
    for (model, reason) in cases {
        let generator = DlGenerator::with_model(dynamic(&model));

        let refinement = generator
            .refine(REFINE_FEEDBACK, &current, REFINE_MERMAID)
            .await
            .expect("refine problems never fail");

        assert_eq!(
            refinement.workflow, current,
            "the existing document is kept"
        );
        let record = only_record(&refinement);
        assert_eq!(record.stage, ProgressStage::RefiningDl);
        assert_eq!(record.source, GenerationSource::Fallback);
        assert!(
            record.detail.contains(reason),
            "detail {:?} lacks {reason:?}",
            record.detail
        );
        assert_eq!(refinement.provenance.fallback_count(), 1);
        assert_eq!(model.call_count(), 1);
    }
}

#[tokio::test]
async fn refine_without_a_model_keeps_the_existing_document() {
    let current = existing();
    let refinement = DlGenerator::new()
        .refine(REFINE_FEEDBACK, &current, REFINE_MERMAID)
        .await
        .expect("refine without a model succeeds");

    assert_eq!(refinement.workflow, current);
    let record = only_record(&refinement);
    assert_eq!(record.stage, ProgressStage::RefiningDl);
    assert_eq!(record.source, GenerationSource::Deterministic);
    assert!(record.detail.contains("no model configured"));
    assert!(!refinement.provenance.has_fallback());
}

#[test]
fn refinement_round_trips_as_json() {
    let mut provenance = GenerationProvenance::new();
    provenance.record_model(ProgressStage::RefiningDl);
    let refinement = Refinement {
        workflow: existing(),
        provenance,
    };

    let value = serde_json::to_value(&refinement).expect("serialize");
    assert_eq!(value["provenance"]["stages"][0]["stage"], "refining_dl");
    let restored: Refinement = serde_json::from_value(value).expect("parse");
    assert_eq!(restored, refinement);

    let empty: Refinement = serde_json::from_value(json!({})).expect("parse an empty object");
    assert_eq!(empty, Refinement::default());
}
