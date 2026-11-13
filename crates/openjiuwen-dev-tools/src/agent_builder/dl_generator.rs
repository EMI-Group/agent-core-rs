//! The DL generator stage: a workflow design becomes a workflow DSL document.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_generator`.
//! Python fills a large prompt template with component lists, node schemas, and
//! few-shot examples, then asks a model for a JSON workflow document. This port
//! keeps the observable artifact — a valid [`WorkflowDsl`] — and produces it
//! deterministically through [`WorkflowDesign::to_workflow_dsl`], with one
//! optional model round trip that may replace the deterministic document when
//! the answer parses and validates.
//!
//! Model problems are best-effort: a failed call, a choice-less answer, a
//! missing JSON fence, an unparseable document, or a document failing
//! [`WorkflowDsl::validate`] is logged and replaced by the deterministic
//! baseline, so [`DlGenerator::generate_with_model`] never fails because of the
//! model. Only an invalid *design* is a [`Error::Validation`].

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;

use super::designer::WorkflowDesign;
use super::spec::WorkflowSpec;
use super::workflow_dsl::WorkflowDsl;
use crate::util::extract_json_fence;

/// The instruction sent to the model by [`DlGenerator::generate_with_model`].
///
/// It names every key the parser reads, so a compliant answer round-trips into
/// [`WorkflowDsl`] without repair.
const DL_GENERATOR_SYSTEM_PROMPT: &str = r#"You are the openJiuwen workflow DL generator.
Turn the given workflow design into one complete workflow DSL document.

Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `name`: the workflow name
- `description`: one sentence describing the workflow
- `nodes`: a list of {"id", "type", "title", "description", "inputs",
  "outputs", "data"} objects, keeping the design's node ids
- `edges`: a list of {"source", "target", "condition"} objects naming node ids

The document needs at least one node, unique node ids, and every edge endpoint
must resolve to a node id.
Do not add other keys and do not wrap the document in prose."#;

/// Generates a workflow DSL document from a workflow design.
///
/// [`Self::generate`] is deterministic and needs no model; [`Self::with_model`]
/// adds one best-effort model round trip behind
/// [`Self::generate_with_model`], and [`Self::from_spec`] converts a
/// programmatic specification without ever calling a model.
pub struct DlGenerator {
    /// The optional model the model path talks to.
    model: Option<Arc<dyn Model>>,
}

impl DlGenerator {
    /// Creates a deterministic generator; the model path never calls a model.
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Creates a generator whose [`Self::generate_with_model`] asks `model`
    /// first and falls back to the deterministic document.
    #[must_use]
    pub fn with_model(model: Arc<dyn Model>) -> Self {
        Self { model: Some(model) }
    }

    /// Converts a design into a validated workflow DSL document.
    ///
    /// The conversion is the one design-to-DSL mapping of the crate
    /// ([`WorkflowDesign::to_workflow_dsl`]), which validates both the design
    /// and the produced document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the design or the produced document
    /// is structurally invalid.
    pub fn generate(&self, design: &WorkflowDesign) -> Result<WorkflowDsl> {
        design.to_workflow_dsl()
    }

    /// Converts a specification into a validated workflow DSL document.
    ///
    /// The name, description, nodes, and edges are kept unchanged; the document
    /// is validated before it is returned. No model is ever called.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document has no nodes, repeated
    /// node ids, or an edge endpoint that does not name a node.
    pub fn from_spec(&self, spec: &WorkflowSpec) -> Result<WorkflowDsl> {
        let document = spec.to_dsl();
        document.validate()?;
        Ok(document)
    }

    /// Converts a design into a workflow DSL document, using the model when one
    /// is configured.
    ///
    /// The deterministic baseline is computed first, so an invalid design is a
    /// validation error regardless of the model. Without a model the baseline
    /// is returned. Otherwise one system message describing the JSON contract is
    /// followed by one user message carrying the design as JSON. A model error,
    /// a choice-less answer, a missing JSON fence, an unparseable document, or a
    /// document failing [`WorkflowDsl::validate`] is logged and replaced by the
    /// baseline.
    ///
    /// # Errors
    ///
    /// Returns the [`Error::Validation`] of [`Self::generate`] for an invalid
    /// design — the only error this entry point produces.
    pub async fn generate_with_model(&self, design: &WorkflowDesign) -> Result<WorkflowDsl> {
        let baseline = self.generate(design)?;
        let Some(model) = self.model.as_deref() else {
            return Ok(baseline);
        };

        match generate_with_model(model, design).await {
            Ok(document) => Ok(document),
            Err(error) => {
                tracing::debug!(
                    %error,
                    "workflow DL generation failed; falling back to the deterministic document"
                );
                Ok(baseline)
            }
        }
    }
}

impl Default for DlGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DlGenerator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DlGenerator")
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

/// Asks the model for one fenced workflow document and validates it.
async fn generate_with_model(model: &dyn Model, design: &WorkflowDesign) -> Result<WorkflowDsl> {
    let design_json =
        serde_json::to_string_pretty(design).unwrap_or_else(|_| format!("{design:?}"));
    let messages = vec![
        BaseMessage::system(DL_GENERATOR_SYSTEM_PROMPT),
        BaseMessage::user(design_json),
    ];
    let request = ModelRequestConfig::new(model.model_name());
    let response = model.invoke(&messages, &request).await?;
    let Some(choice) = response.choices.into_iter().next() else {
        return Err(Error::llm("model returned no choices"));
    };

    let text = choice.message.text().into_owned();
    let value = extract_json_fence(&text)
        .ok_or_else(|| Error::validation("model answer contained no JSON code fence"))?;
    let document: WorkflowDsl = serde_json::from_value(value).map_err(|error| {
        Error::validation(format!("failed to parse workflow DSL JSON: {error}"))
    })?;
    document.validate()?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::{json, Value};

    use super::*;
    use crate::agent_builder::designer::WorkflowDesigner;
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
}
