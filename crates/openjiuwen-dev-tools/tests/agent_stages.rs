//! Integration suite for the staged agent-design seam, in pipeline order: the
//! deterministic and model-backed [`IntentionDetector`] verdicts, the
//! multi-round [`Clarifier`] dialogue and its stream form, [`WorkflowDesigner`]'s
//! provenance plus its three-pass and streaming views, [`DlGenerator::refine`],
//! the Mermaid `When condition met[...]` labels rendered through
//! [`when_condition_label`]/[`render_mermaid`], [`DlReflector`]'s variable rules
//! and repairs, and the plugin identity a [`DlTransformer`] resolves.
//!
//! Everything runs offline and deterministically on
//! [`openjiuwen_test_support::ScriptedModel`]; ids and timestamps are explicit.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_dev_tools::agent_builder::{
    render_design, render_mermaid, when_condition_label, BuildIntent, ClarificationTermination,
    Clarifier, ClarifyConfig, ClarifyEvent, DesignEdge, DesignEvent, DesignNode, DesignNodeKind,
    DesignPassKind, DlDocument, DlGenerator, DlNode, DlReflector, DlTransformer, DlVariable,
    DslEdge, DslNode, GenerationSource, HistoryManager, IntentionDetector, PluginResource,
    ProgressStage, ReflectionIssueKind, WorkflowDesign, WorkflowDesigner, WorkflowDsl,
    DEFAULT_MAX_CLARIFY_ROUNDS,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

/// Coerces a scripted model into the `Arc<dyn Model>` the stages take.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// The request every designer and generator assertion runs against; it names no
/// decision marker, so no designer pass inserts a branch node.
const FLOW_QUERY: &str = "Collect the order. Then confirm it.";

/// A model answer declaring that the request describes a process.
const INTENT_PROCESS: &str = r#"```json
{"provide_process": true}
```"#;

/// A model answer declaring that the request asks for a change.
const INTENT_REFINE: &str = r#"```json
{"need_refined": true}
```"#;

/// A clarification answer naming the one missing detail.
const CLARIFY_NEEDS_MODEL: &str = r#"```json
{"ready": false, "question": "Which model should the agent use?", "summary": "goal, input"}
```"#;

/// A clarification answer declaring the request complete.
const CLARIFY_READY: &str = r#"```json
{"ready": true, "question": "", "summary": "goal, input, output"}
```"#;

/// A designer answer holding a complete, valid `start -> llm -> end` design.
const DESIGN_FENCE: &str = r#"```json
{
  "name": "order-flow",
  "description": "handle an order",
  "nodes": [
    {"id": "start", "type": "start", "title": "Start", "description": "entry"},
    {"id": "work", "type": "llm", "title": "Work", "description": "handle the order"},
    {"id": "end", "type": "end", "title": "End", "description": "exit"}
  ],
  "edges": [
    {"source": "start", "target": "work"},
    {"source": "work", "target": "end"}
  ]
}
```"#;

/// A refinement answer inserting one `code` node between `start` and `end`.
const REFINED_FENCE: &str = r#"```json
{
  "name": "order-flow",
  "description": "handle an order",
  "nodes": [
    {"id": "start", "type": "start", "title": "Start", "description": "entry"},
    {"id": "validate", "type": "code", "title": "Validate", "description": "validate the order"},
    {"id": "end", "type": "end", "title": "End", "description": "exit"}
  ],
  "edges": [
    {"source": "start", "target": "validate"},
    {"source": "validate", "target": "end"}
  ]
}
```"#;

/// Drains an event stream, asserting that no item is an error.
async fn drain<T>(mut stream: BoxStream<'static, openjiuwen_common::error::Result<T>>) -> Vec<T> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("every streamed event must be Ok"));
    }
    items
}

/// A workflow DSL node carrying only the fields the stages read.
fn dsl_node(id: &str, node_type: &str, description: &str) -> DslNode {
    DslNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        title: id.to_string(),
        description: description.to_string(),
        ..DslNode::default()
    }
}

/// An unconditional workflow DSL edge.
fn dsl_edge(source: &str, target: &str) -> DslEdge {
    DslEdge {
        source: source.to_string(),
        target: target.to_string(),
        condition: None,
    }
}

/// A design node carrying only the fields the renderer and connectors read.
fn design_node(id: &str, kind: DesignNodeKind, description: &str) -> DesignNode {
    DesignNode {
        id: id.to_string(),
        kind,
        title: id.to_string(),
        description: description.to_string(),
        config: json!({}),
    }
}

/// A design edge with an optional branch condition.
fn design_edge(source: &str, target: &str, condition: Option<&str>) -> DesignEdge {
    DesignEdge {
        source: source.to_string(),
        target: target.to_string(),
        condition: condition.map(str::to_string),
    }
}

/// A two-node `start -> end` document, the shape every refine case starts from.
fn two_node_document() -> WorkflowDsl {
    WorkflowDsl {
        name: "order-flow".to_string(),
        description: "handle an order".to_string(),
        nodes: vec![
            dsl_node("start", "start", "begin the order"),
            dsl_node("end", "end", "finish the order"),
        ],
        edges: vec![dsl_edge("start", "end")],
    }
}

/// A DL node with a description and an optional `next` link.
fn dl_node(id: &str, node_type: &str, description: &str, next: Option<&str>) -> DlNode {
    DlNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        description: description.to_string(),
        next: next.map(str::to_string),
        ..DlNode::default()
    }
}

/// A DL input variable bound to the `${node.field}` reference `value`.
fn dl_reference(name: &str, value: &str) -> DlVariable {
    DlVariable {
        name: name.to_string(),
        value: Some(json!(value)),
        ..DlVariable::default()
    }
}

/// A described DL output variable.
fn dl_output(name: &str, description: &str) -> DlVariable {
    DlVariable {
        name: name.to_string(),
        description: Some(description.to_string()),
        ..DlVariable::default()
    }
}

#[tokio::test]
async fn intention_detector_applies_the_deterministic_rules() {
    let history = HistoryManager::new(20);
    let detector = IntentionDetector::new();

    let described = detector
        .detect_initial("Create a workflow with two steps.", &history)
        .await
        .expect("a non-blank query is never an error");
    assert_eq!(
        described.intent,
        BuildIntent::Build,
        "naming a workflow must be enough to build from"
    );
    assert_eq!(described.intent.as_str(), "build");
    assert_eq!(
        described.source,
        GenerationSource::Deterministic,
        "a detector without a model must record the deterministic source"
    );
    assert_eq!(
        described.detail, "the request names a workflow (workflow, steps)",
        "the detail must list the workflow keywords in the documented order"
    );

    let vague = detector
        .detect_initial("Build an agent.", &history)
        .await
        .expect("a non-blank query is never an error");
    assert_eq!(
        vague.intent,
        BuildIntent::Clarify,
        "a request without steps or a workflow keyword must ask for a process"
    );
    assert_eq!(vague.detail, "the request states no process steps yet");

    let refine = detector
        .detect_refine(
            "please modify the workflow",
            &history,
            "graph TD\n  start --> end",
        )
        .await
        .expect("a non-blank query is never an error");
    assert_eq!(refine.intent, BuildIntent::Refine);
    assert_eq!(
        refine.detail,
        "the request asks to modify the workflow (`modify`)"
    );

    let fresh = detector
        .detect_refine("please modify the workflow", &history, "   ")
        .await
        .expect("a non-blank query is never an error");
    assert_eq!(
        fresh.intent,
        BuildIntent::Build,
        "a blank diagram leaves nothing to refine"
    );
    assert_eq!(fresh.detail, "there is no existing diagram to refine");

    let error = detector
        .detect_initial("   ", &history)
        .await
        .expect_err("a blank query must be rejected before any detection");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("must not be blank"), "{error}");
}

#[tokio::test]
async fn intention_detector_records_a_model_verdict() {
    let history = HistoryManager::new(20);
    let model = Arc::new(
        ScriptedModel::builder()
            .respond_text(INTENT_PROCESS)
            .respond_text(INTENT_REFINE)
            .respond_text("I am not sure.")
            .fallback(ScriptedTurn::text(INTENT_REFINE))
            .build(),
    );
    let guided = IntentionDetector::with_model(dynamic(&model));
    assert_eq!(
        format!("{guided:?}"),
        "IntentionDetector { has_model: true }",
        "the debug form must report that a model is wired"
    );

    let initial = guided
        .detect_initial("Summarize the document.", &history)
        .await
        .expect("a model verdict is not an error");
    assert_eq!(initial.intent, BuildIntent::Build);
    assert_eq!(initial.source, GenerationSource::Model);
    assert_eq!(initial.detail, "model answered provide_process=true");

    let refined = guided
        .detect_refine("please modify it", &history, "graph TD")
        .await
        .expect("a model verdict is not an error");
    assert_eq!(refined.intent, BuildIntent::Refine);
    assert_eq!(refined.source, GenerationSource::Model);
    assert_eq!(refined.detail, "model answered need_refined=true");

    let unusable = guided
        .detect_initial("Summarize the document.", &history)
        .await
        .expect("an unusable answer falls back instead of failing");
    assert_eq!(
        unusable.intent,
        BuildIntent::Clarify,
        "an answer without a JSON fence must keep the deterministic verdict"
    );
    assert_eq!(unusable.source, GenerationSource::Fallback);
    assert!(
        unusable
            .detail
            .starts_with("answer had no JSON fence; deterministic rule: "),
        "the fallback detail must name the reason: {}",
        unusable.detail
    );

    assert_eq!(
        model.call_count(),
        3,
        "each detection asks the model exactly once"
    );
    let refine_call = model.calls()[1]
        .last_user_text()
        .expect("the refine call carries a user message");
    assert!(
        refine_call.contains("## Current Mermaid diagram\ngraph TD"),
        "the refine prompt must carry the current diagram: {refine_call}"
    );
}

#[tokio::test]
async fn clarification_dialogue_asks_then_reports_ready() {
    assert_eq!(
        ClarifyConfig::new(0).resolved_max_rounds(),
        DEFAULT_MAX_CLARIFY_ROUNDS,
        "a zero round cap must resolve to the documented default"
    );

    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(CLARIFY_NEEDS_MODEL),
        ScriptedTurn::text(CLARIFY_READY),
    ]));
    let clarifier = Clarifier::with_model(dynamic(&model));
    let history = HistoryManager::new(20);
    let query = "Build an agent that reads the input.";

    let mut asked = 0;
    let dialogue = clarifier
        .clarify_rounds_with(query, &history, &ClarifyConfig::new(3), |round| {
            asked += 1;
            assert_eq!(round.question, "Which model should the agent use?");
            assert!(
                !round.ready,
                "the responder is only asked when a round is not ready"
            );
            Some("Use the scripted model.".to_string())
        })
        .await
        .expect("the dialogue succeeds");

    assert_eq!(asked, 1, "only the not-ready round asks the responder");
    assert_eq!(
        dialogue.termination,
        ClarificationTermination::Ready,
        "a ready round must terminate the dialogue"
    );
    assert_eq!(dialogue.termination.as_str(), "ready");
    assert_eq!(dialogue.rounds.len(), 2);

    let first = &dialogue.rounds[0];
    assert_eq!(first.index, 0);
    assert!(!first.ready);
    assert_eq!(first.question, "Which model should the agent use?");
    assert_eq!(first.answer.as_deref(), Some("Use the scripted model."));
    assert_eq!(first.source, GenerationSource::Model);

    let second = &dialogue.rounds[1];
    assert_eq!(second.index, 1);
    assert!(second.ready);
    assert_eq!(second.question, "");
    assert_eq!(second.answer, None);
    assert_eq!(second.summary, "goal, input, output");

    assert!(dialogue.clarification.ready);
    assert_eq!(
        dialogue.clarification.provenance, dialogue.provenance,
        "the final verdict must carry the dialogue's aggregate provenance"
    );
    assert_eq!(dialogue.provenance.stages.len(), 2);
    assert!(dialogue
        .provenance
        .stages
        .iter()
        .all(|stage| stage.stage == ProgressStage::Clarifying));
    assert_eq!(model.call_count(), 2);
    assert!(
        history.is_empty(),
        "a dialogue must judge a working copy and leave the caller's history alone"
    );

    let offline = Clarifier::new()
        .clarify_rounds(query, &history, &ClarifyConfig::default())
        .await
        .expect("the offline dialogue succeeds");
    assert_eq!(
        offline.termination,
        ClarificationTermination::NoAnswer,
        "without a responder a not-ready round stops the dialogue"
    );
    assert_eq!(offline.rounds.len(), 1);
    assert_eq!(
        offline.rounds[0].question, "What should the agent produce as its output?",
        "the heuristic asks for the first missing requirement, here the output"
    );
    assert_eq!(offline.rounds[0].summary, "goal, input");
    assert!(!offline.clarification.ready);
}

#[tokio::test]
async fn streamed_clarification_matches_the_buffered_verdict() {
    let model = Arc::new(
        ScriptedModel::builder()
            .respond_text(CLARIFY_READY)
            .fallback(ScriptedTurn::text(CLARIFY_READY))
            .build(),
    );
    let clarifier = Clarifier::with_model(dynamic(&model));
    let history = HistoryManager::new(20);
    let query = "Build an agent that summarizes the document and returns a summary.";

    let buffered = clarifier
        .clarify(query, &history)
        .await
        .expect("the single-pass verdict succeeds");
    assert!(buffered.ready);
    assert_eq!(buffered.question, "");
    assert_eq!(buffered.summary, "goal, input, output");

    let events = drain(clarifier.stream_clarify(query, &history)).await;
    assert_eq!(events.len(), 2, "one round event then one completion event");
    let completed = match events.last() {
        Some(ClarifyEvent::Completed(dialogue)) => dialogue,
        other => panic!("the stream must end with the dialogue, got {other:?}"),
    };
    assert_eq!(
        completed.clarification, buffered,
        "the streamed dialogue must reproduce the buffered verdict exactly"
    );
    assert_eq!(completed.termination, ClarificationTermination::Ready);
    assert_eq!(completed.rounds.len(), 1);
    assert_eq!(
        events[0],
        ClarifyEvent::Round(completed.rounds[0].clone()),
        "the round event must carry the round the dialogue recorded"
    );
    assert_eq!(model.call_count(), 2, "one model call per entry point");

    let mut stream = clarifier.stream_clarify("   ", &history);
    let error = stream
        .next()
        .await
        .expect("a blank query still yields one item")
        .expect_err("a blank query is a validation error");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        stream.next().await.is_none(),
        "a blank query must end the stream after the error"
    );
}

#[tokio::test]
async fn designer_records_its_source_and_falls_back() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(DESIGN_FENCE),
        ScriptedTurn::text("I cannot design that."),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));

    let accepted = designer
        .design_with_provenance(FLOW_QUERY)
        .await
        .expect("a valid model answer is not an error");
    accepted
        .design
        .validate()
        .expect("an accepted model design must validate");
    assert_eq!(accepted.design.name, "order-flow");
    assert_eq!(accepted.provenance.stages.len(), 1);
    assert_eq!(
        accepted.provenance.stages[0].stage,
        ProgressStage::GeneratingWorkflowDesign
    );
    assert_eq!(
        accepted.provenance.stages[0].source,
        GenerationSource::Model
    );
    assert!(
        accepted.provenance.stages[0].detail.is_empty(),
        "an accepted answer carries no rejection detail"
    );

    let rejected = designer
        .design_with_provenance(FLOW_QUERY)
        .await
        .expect("an unusable answer falls back instead of failing");
    assert_eq!(
        rejected.provenance.stages[0].source,
        GenerationSource::Fallback
    );
    assert!(
        !rejected.provenance.stages[0].detail.is_empty(),
        "a fallback must record why the answer was rejected"
    );
    rejected
        .design
        .validate()
        .expect("the fallback design must validate");

    let offline = WorkflowDesigner::new()
        .design_with_provenance(FLOW_QUERY)
        .await
        .expect("the offline designer succeeds");
    assert_eq!(
        offline.provenance.stages[0].source,
        GenerationSource::Deterministic
    );
    assert_eq!(
        offline.provenance.stages[0].detail,
        "no model configured; deterministic design synthesis"
    );
    assert_eq!(
        rejected.design, offline.design,
        "the fallback must be the documented offline design"
    );
    let descriptions: Vec<&str> = offline
        .design
        .nodes
        .iter()
        .filter(|node| node.kind == DesignNodeKind::Llm)
        .map(|node| node.description.as_str())
        .collect();
    assert_eq!(
        descriptions,
        ["Collect the order", "confirm it"],
        "the offline designer must split the request into its two clauses"
    );

    let error = designer
        .design_with_provenance("  ")
        .await
        .expect_err("a blank query is a validation error");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        model.call_count(),
        2,
        "the blank query must be rejected before any model call"
    );
}

#[tokio::test]
async fn three_pass_design_and_stream_agree() {
    let model = Arc::new(
        ScriptedModel::builder()
            .respond_text(DESIGN_FENCE)
            .respond_text(DESIGN_FENCE)
            .respond_text(DESIGN_FENCE)
            .fallback(ScriptedTurn::text(DESIGN_FENCE))
            .build(),
    );
    let designer = WorkflowDesigner::with_model(dynamic(&model));

    let buffered = designer
        .design_three_pass(FLOW_QUERY)
        .await
        .expect("the three passes succeed");
    let kinds: Vec<DesignPassKind> = buffered.passes.iter().map(|pass| pass.kind).collect();
    assert_eq!(
        kinds,
        vec![
            DesignPassKind::Basic,
            DesignPassKind::Branch,
            DesignPassKind::Reflection
        ],
        "the passes must run in Python's basic, branch, reflection order"
    );
    assert_eq!(DesignPassKind::Reflection.as_str(), "reflection");
    assert!(
        buffered
            .passes
            .iter()
            .all(|pass| pass.source == GenerationSource::Model
                && pass.node_count == 3
                && pass.edge_count == 2),
        "every accepted pass must report the model source over the three-node design: {:?}",
        buffered.passes
    );
    assert!(
        buffered.notes.is_empty(),
        "a design that needs no repair must record no repair note"
    );
    buffered
        .design
        .validate()
        .expect("the final three-pass design must validate");
    assert_eq!(buffered.provenance.stages.len(), 3);
    assert_eq!(
        buffered.provenance.stages[0].stage,
        ProgressStage::GeneratingWorkflowDesign
    );
    assert_eq!(
        buffered.provenance.stages[2].stage,
        ProgressStage::ValidatingDl,
        "the reflection pass is recorded as the validation stage"
    );
    assert_eq!(model.call_count(), 3);

    let events = drain(designer.stream_design(FLOW_QUERY)).await;
    assert_eq!(
        events.len(),
        7,
        "two events per pass plus one completion event"
    );
    assert_eq!(events[0], DesignEvent::PassStarted(DesignPassKind::Basic));
    assert_eq!(
        events[1],
        DesignEvent::PassCompleted(buffered.passes[0].clone()),
        "the streamed record must match the buffered pass exactly"
    );
    assert_eq!(events[2], DesignEvent::PassStarted(DesignPassKind::Branch));
    assert_eq!(
        events[4],
        DesignEvent::PassStarted(DesignPassKind::Reflection)
    );
    let completed = match events.last() {
        Some(DesignEvent::Completed(three_pass)) => three_pass,
        other => panic!("the stream must end with the finished design, got {other:?}"),
    };
    assert_eq!(
        completed, &buffered,
        "streaming the pipeline must reproduce the buffered three-pass design"
    );
    assert_eq!(model.call_count(), 6, "each entry point runs three passes");
}

#[tokio::test]
async fn refine_rewrites_the_document_and_records_provenance() {
    let existing = two_node_document();
    existing
        .validate()
        .expect("the document under refinement must be valid");
    let mermaid = render_mermaid(&existing);

    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(REFINED_FENCE),
        ScriptedTurn::text("I could not revise that document."),
    ]));
    let generator = DlGenerator::with_model(dynamic(&model));

    let refinement = generator
        .refine("validate the order before finishing", &existing, &mermaid)
        .await
        .expect("a usable refinement answer is not an error");
    refinement
        .workflow
        .validate()
        .expect("a refined document must validate");
    assert_ne!(
        refinement.workflow, existing,
        "an accepted refinement must change the document"
    );
    assert!(
        refinement.workflow.node("validate").is_some(),
        "the refinement must apply the requested change"
    );
    assert_eq!(refinement.workflow.edges.len(), 2);
    assert_eq!(refinement.provenance.stages.len(), 1);
    assert_eq!(
        refinement.provenance.stages[0].stage,
        ProgressStage::RefiningDl
    );
    assert_eq!(
        refinement.provenance.stages[0].source,
        GenerationSource::Model
    );

    let kept = generator
        .refine("validate the order before finishing", &existing, &mermaid)
        .await
        .expect("an unusable refinement answer falls back");
    assert_eq!(
        kept.workflow, existing,
        "a rejected refinement must keep the caller's document"
    );
    assert_eq!(kept.provenance.stages[0].source, GenerationSource::Fallback);
    assert!(
        !kept.provenance.stages[0].detail.is_empty(),
        "a rejected refinement must record why"
    );

    let offline = DlGenerator::new()
        .refine("validate the order before finishing", &existing, &mermaid)
        .await
        .expect("a generator without a model keeps the document");
    assert_eq!(offline.workflow, existing);
    assert_eq!(
        offline.provenance.stages[0].source,
        GenerationSource::Deterministic
    );
    assert_eq!(
        offline.provenance.stages[0].detail,
        "no model configured; keeping the existing workflow"
    );

    let error = generator
        .refine("   ", &existing, &mermaid)
        .await
        .expect_err("blank feedback is a validation error");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        model.call_count(),
        2,
        "blank feedback must be rejected before any model call"
    );
    let prompt = model.calls()[0]
        .last_user_text()
        .expect("the refine call carries a user message");
    assert!(
        prompt.contains("## Existing Mermaid diagram\ngraph TD")
            && prompt.contains("## Existing workflow document"),
        "the refine prompt must carry the diagram and the document: {prompt}"
    );
}

#[test]
fn mermaid_extracts_when_condition_labels() {
    assert_eq!(
        when_condition_label("When condition met[order valid]").as_deref(),
        Some("order valid")
    );
    assert_eq!(
        when_condition_label("When condition met: order valid").as_deref(),
        Some("order valid")
    );
    assert_eq!(
        when_condition_label("When condition met[a[b]c]").as_deref(),
        Some("a[b]c"),
        "nested brackets belong to the label"
    );
    assert_eq!(
        when_condition_label("when condition met[order valid]"),
        None,
        "the prefix match is case-sensitive like Python's regex"
    );
    assert_eq!(when_condition_label("score > 0.5"), None);
}

#[test]
fn mermaid_renders_labelled_branches() {
    let document = WorkflowDsl {
        name: "route".to_string(),
        description: "route an order".to_string(),
        nodes: vec![
            dsl_node("router", "branch", "Route the order"),
            dsl_node("ship", "end", "Ship the order"),
            dsl_node("hold", "end", "Hold the order"),
        ],
        edges: vec![
            DslEdge {
                condition: Some("When condition met[order valid]".to_string()),
                ..dsl_edge("router", "ship")
            },
            DslEdge {
                condition: Some("When condition met: order invalid".to_string()),
                ..dsl_edge("router", "hold")
            },
        ],
    };
    assert_eq!(
        render_mermaid(&document),
        "\
graph TD
  router[Noderouter: Route the order]
  ship[Nodeship: Ship the order]
  hold[Nodehold: Hold the order]
  router -- order valid --> ship
  router -- order invalid --> hold",
        "a multi-successor source must render the extracted label, never the wrapper"
    );

    let design = WorkflowDesign {
        name: "route".to_string(),
        description: "route an order".to_string(),
        nodes: vec![
            design_node("start", DesignNodeKind::Start, "Begin"),
            DesignNode {
                config: json!({"conditions": ["order valid", "otherwise"]}),
                ..design_node("decide", DesignNodeKind::Branch, "Decide")
            },
            design_node("end", DesignNodeKind::End, "Finish"),
        ],
        edges: vec![
            design_edge("start", "decide", None),
            design_edge("decide", "end", Some("When condition met[order valid]")),
            design_edge("decide", "end", Some("otherwise")),
        ],
    };
    design
        .validate()
        .expect("the branch design must be structurally valid");
    let converted = design
        .to_workflow_dsl()
        .expect("the design must convert into a workflow document");
    assert_eq!(
        render_design(&design),
        render_mermaid(&converted),
        "a design and its converted document must render byte-identically"
    );
    assert!(
        render_design(&design).contains("  decide -- order valid --> end"),
        "the design's branch edges must keep the extracted labels: {}",
        render_design(&design)
    );
}

#[test]
fn reflector_reports_variable_rule_issues() {
    let mut start = dl_node("start", "Start", "entry", Some("llm"));
    start.parameters.outputs = vec![dl_output("query", "用户输入")];

    let mut llm = dl_node("llm", "LLM", "generate", Some("end"));
    llm.parameters.inputs = vec![
        dl_reference("query", "${start.query}"),
        dl_reference("query", "${start.query} and ${start.query}"),
        dl_reference("ghost", "${ghost.result}"),
    ];
    llm.parameters.outputs = vec![dl_output("result", "model output")];
    llm.parameters
        .configs
        .insert("system_prompt".to_string(), json!("You are helpful."));
    llm.parameters
        .configs
        .insert("user_prompt".to_string(), json!("${start.query}"));

    let mut end = dl_node("end", "End", "exit", None);
    end.parameters.inputs = vec![dl_reference("result", "${llm.result}")];
    end.parameters
        .configs
        .insert("template".to_string(), json!("{{result}}"));

    let document = DlDocument {
        nodes: vec![start, llm, end],
    };
    document
        .validate()
        .expect("the document is structurally valid even with variable defects");

    let validation = DlReflector::new().validate_dl(&document);
    let kinds: Vec<ReflectionIssueKind> =
        validation.issues.iter().map(|issue| issue.kind).collect();
    assert_eq!(
        kinds,
        vec![
            ReflectionIssueKind::DuplicateVariable,
            ReflectionIssueKind::MultiPlaceholderValue,
            ReflectionIssueKind::UndeclaredReference
        ],
        "one issue per violated variable rule, in node order then check order"
    );
    assert!(!validation.passed);
    assert!(
        validation
            .issues
            .iter()
            .all(|issue| issue.node_id.as_deref() == Some("llm")),
        "every variable issue must name the node that declares the defect"
    );
    assert!(
        validation.issues[0]
            .message
            .contains("`parameters.inputs` declares the name `query` more than once"),
        "{}",
        validation.issues[0].message
    );
    assert!(
        validation.issues[1]
            .message
            .contains("2 reference variables"),
        "{}",
        validation.issues[1].message
    );
    assert!(
        validation.issues[2].message.contains("${ghost.result}"),
        "{}",
        validation.issues[2].message
    );
}

#[test]
fn reflector_repairs_a_damaged_document() {
    let damaged = WorkflowDsl {
        name: "damaged".to_string(),
        description: "a document the deterministic rules must repair".to_string(),
        nodes: vec![
            dsl_node("start", "start", "Begin"),
            dsl_node("work", "llm", "Work"),
            dsl_node("work", "llm", "Work again"),
            dsl_node("end", "end", "Finish"),
        ],
        edges: vec![
            dsl_edge("start", "work"),
            dsl_edge("work", "ghost"),
            dsl_edge("work", "end"),
        ],
    };
    let reflection = DlReflector::new()
        .reflect(&damaged)
        .expect("every repairable defect is repaired deterministically");
    let repaired_kinds: Vec<ReflectionIssueKind> =
        reflection.issues.iter().map(|issue| issue.kind).collect();
    assert!(
        repaired_kinds.contains(&ReflectionIssueKind::DuplicateNodeId),
        "the repeated id must be repaired: {repaired_kinds:?}"
    );
    assert!(
        repaired_kinds.contains(&ReflectionIssueKind::DanglingEdge),
        "the edge naming a missing node must be dropped: {repaired_kinds:?}"
    );
    assert!(!reflection.passed);
    reflection
        .repaired
        .validate()
        .expect("a repaired document must always validate");
    assert_eq!(
        reflection.repaired.nodes.len(),
        4,
        "renaming a duplicate id must keep every node"
    );
    assert!(
        reflection.repaired.node("work_2").is_some(),
        "the second occurrence must be renamed `work_2`"
    );
}

#[test]
fn transformer_resolves_plugin_identity() {
    let mut plugin = dl_node("plugin", "Plugin", "call the search tool", Some("end"));
    plugin.parameters.inputs = vec![dl_reference("query", "${start.query}")];
    plugin.parameters.outputs = vec![dl_output("result", "tool result")];
    for (key, value) in [
        ("tool_id", json!("tool-1")),
        ("tool_name", json!("search")),
        ("plugin_id", json!("plugin-9")),
        ("plugin_name", json!("Search")),
        ("plugin_version", json!("2.0.0")),
    ] {
        plugin.parameters.configs.insert(key.to_string(), value);
    }
    let document = DlDocument {
        nodes: vec![plugin, dl_node("end", "End", "exit", None)],
    };

    let workflow = DlTransformer::new()
        .transform(&document)
        .expect("the plugin document transforms");
    let value = workflow
        .to_value()
        .expect("the platform document serializes");
    assert_eq!(value["nodes"][0]["id"], json!("plugin"));
    assert_eq!(
        value["nodes"][0]["type"],
        json!("19"),
        "a plugin node must take the platform plugin type id"
    );
    assert_eq!(
        value["nodes"][0]["meta"]["position"],
        json!({"x": 0, "y": 0}),
        "the first node sits at the origin"
    );
    assert_eq!(
        value["nodes"][1]["meta"]["position"],
        json!({"x": 20, "y": 20}),
        "each following node steps by twenty on both axes"
    );
    assert_eq!(
        value["nodes"][0]["data"]["title"],
        json!("call the search tool")
    );
    assert_eq!(
        value["nodes"][0]["data"]["inputs"]["plugin_param"],
        json!({
            "toolID": "tool-1",
            "toolName": "search",
            "pluginID": "plugin-9",
            "pluginName": "Search",
            "pluginVersion": "2.0.0",
        }),
        "the node's own configs must supply the plugin identity"
    );
    assert_eq!(
        value["nodes"][0]["data"]["inputs"]["input_parameters"]["query"],
        json!({"type": "ref", "content": ["start", "query"], "extra": {"index": 0}}),
        "a `${{node.field}}` input must become a reference parameter"
    );
    assert_eq!(
        value["edges"],
        json!([{"source_node_id": "plugin", "target_node_id": "end"}]),
        "an unconditional plugin edge carries no source port id"
    );

    let resources = [PluginResource {
        tool_id: "tool-1".to_string(),
        tool_name: "search".to_string(),
        plugin_id: "plugin-catalog".to_string(),
        plugin_name: "Search catalog".to_string(),
        plugin_version: "3.1.0".to_string(),
    }];
    let resolved = DlTransformer::new()
        .transform_with_resources(&document, &resources)
        .expect("a matched resource still transforms");
    let resolved = resolved
        .to_value()
        .expect("the resolved platform document serializes");
    assert_eq!(
        resolved["nodes"][0]["data"]["inputs"]["plugin_param"],
        json!({
            "toolID": "tool-1",
            "toolName": "search",
            "pluginID": "plugin-catalog",
            "pluginName": "Search catalog",
            "pluginVersion": "3.1.0",
        }),
        "a matched plugin resource must win over the node's own configs"
    );
}
