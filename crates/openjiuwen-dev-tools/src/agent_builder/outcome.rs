//! The builder's session state and result vocabulary.
//!
//! These types are the observable response shape of
//! [`crate::agent_builder::builder::AgentBuilder`]: [`BuildState`] tracks where
//! a session is, [`map_state_to_status`] renders the Python status string,
//! [`BuildOutput`] carries one produced artifact, and [`BuildResult`] flattens
//! the two into the `status`/`session_id`/`agent_type`/output response object.
//!
//! The module also owns the shared generation-provenance vocabulary every stage
//! records into: [`GenerationSource`] says whether an output came from a model
//! answer, the documented deterministic path, or a fallback after a failed model
//! call; [`StageProvenance`] records that for one [`ProgressStage`];
//! [`GenerationProvenance`] is the ordered list a build hands back; and
//! [`RepairReport`] carries the bounded repair/retry episode that produced a
//! workflow artifact.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::builder::AgentType;
use super::dsl::LlmAgentDsl;
use super::progress::{progress_for, ProgressStage};
use super::workflow_dsl::WorkflowDsl;

/// Where a session is in Python's build-state machine.
///
/// The port adds [`BuildState::Clarifying`] and [`BuildState::Requesting`],
/// which Python only models as status strings next to its state enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildState {
    /// No request has been processed yet.
    #[default]
    Initial,
    /// The builder is waiting for the user to answer a clarification question.
    Clarifying,
    /// The builder is waiting for the user to describe the workflow steps.
    Requesting,
    /// A request is being processed.
    Processing,
    /// The requested artifact was produced.
    Completed,
}

/// The status string reported for `state` and `agent_type`.
///
/// The six reachable strings are `"clarifying"`, `"requesting"`,
/// `"processing"`, and `"completed"`; [`BuildState::Initial`] maps to
/// `"clarifying"` for [`AgentType::LlmAgent`] and to `"requesting"` for
/// [`AgentType::Workflow`], because a workflow session starts by asking for the
/// processing steps.
#[must_use]
pub fn map_state_to_status(state: BuildState, agent_type: AgentType) -> String {
    let status = match state {
        BuildState::Initial => match agent_type {
            AgentType::LlmAgent => "clarifying",
            AgentType::Workflow => "requesting",
        },
        BuildState::Clarifying => "clarifying",
        BuildState::Requesting => "requesting",
        BuildState::Processing => "processing",
        BuildState::Completed => "completed",
    };
    status.to_string()
}

/// A snapshot of one builder session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildStatus {
    /// The session the snapshot belongs to.
    pub session_id: String,
    /// The artifact the session is building.
    pub agent_type: AgentType,
    /// The state the session is in.
    pub state: BuildState,
    /// How many dialogue turns the session retains.
    pub resource_count: usize,
}

impl Default for BuildStatus {
    /// An empty snapshot: no session id, an LLM agent, [`BuildState::Initial`].
    fn default() -> Self {
        Self {
            session_id: String::new(),
            agent_type: AgentType::LlmAgent,
            state: BuildState::Initial,
            resource_count: 0,
        }
    }
}

impl BuildStatus {
    /// The overall build progress (0-100) reported for `stage`.
    ///
    /// Delegates to [`progress_for`] with this snapshot's agent type, so a stage
    /// the artifact never reaches reports `0`. The serialized snapshot is
    /// unchanged: progress is derived, never stored.
    #[must_use]
    pub fn progress(&self, stage: ProgressStage) -> u8 {
        progress_for(self.agent_type, stage)
    }
}

/// What one build produced.
///
/// The variants serialize externally tagged, matching the Python response
/// shape: `{"dsl": {...}}`, `{"workflow": {...}}`, or `{"response": "..."}`.
// The DSL document is carried inline because it is the payload callers hand to
// the platform; the size difference between the variants is deliberate.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildOutput {
    /// A parsed LLM-agent document.
    Dsl(LlmAgentDsl),
    /// A parsed and structurally valid workflow document.
    Workflow(WorkflowDsl),
    /// A plain-text model answer, typically a clarifying question.
    Response(String),
}

impl Default for BuildOutput {
    /// An empty plain-text answer, so [`BuildResult`] can default.
    fn default() -> Self {
        Self::Response(String::new())
    }
}

impl BuildOutput {
    /// Converts the output into a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the output cannot be serialized, with
    /// the serde error in the message.
    pub fn as_json(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(|error| {
            Error::validation(format!("failed to serialize build output: {error}"))
        })
    }
}

/// Where one stage's output came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationSource {
    /// A model answer was requested and accepted.
    #[default]
    Model,
    /// No model is wired for this stage, so the documented offline path ran.
    Deterministic,
    /// A model was wired but its answer failed (an error, a parse failure, or a
    /// validation failure), so the deterministic path was substituted.
    Fallback,
}

impl GenerationSource {
    /// The source's wire name: `"model"`, `"deterministic"`, or `"fallback"`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Deterministic => "deterministic",
            Self::Fallback => "fallback",
        }
    }
}

/// One stage's recorded provenance.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StageProvenance {
    /// The stage that produced an output, from [`ProgressStage`].
    pub stage: ProgressStage,
    /// Where that output came from.
    pub source: GenerationSource,
    /// A short human-readable note, e.g. `"model answer was not a valid workflow
    /// design"`; empty for an accepted model answer.
    pub detail: String,
}

/// The stages that produced one artifact, oldest first.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GenerationProvenance {
    /// One record per stage that produced part of the artifact, oldest first.
    pub stages: Vec<StageProvenance>,
}

impl GenerationProvenance {
    /// An artifact with no recorded stage yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no stage has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// How many stages have been recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Appends one stage record with an explicit source and detail.
    pub fn record(
        &mut self,
        stage: ProgressStage,
        source: GenerationSource,
        detail: impl Into<String>,
    ) {
        self.stages.push(StageProvenance {
            stage,
            source,
            detail: detail.into(),
        });
    }

    /// Records a stage whose output came from an accepted model answer.
    pub fn record_model(&mut self, stage: ProgressStage) {
        self.record(stage, GenerationSource::Model, String::new());
    }

    /// Records a stage that ran its documented offline path without a model.
    pub fn record_deterministic(&mut self, stage: ProgressStage, detail: impl Into<String>) {
        self.record(stage, GenerationSource::Deterministic, detail);
    }

    /// Records a stage whose model answer was rejected, so the deterministic
    /// path was substituted.
    pub fn record_fallback(&mut self, stage: ProgressStage, detail: impl Into<String>) {
        self.record(stage, GenerationSource::Fallback, detail);
    }

    /// Appends every record of `other`, keeping this artifact's records first.
    pub fn extend_from(&mut self, other: GenerationProvenance) {
        self.stages.extend(other.stages);
    }

    /// Whether any stage fell back from a failed model answer.
    #[must_use]
    pub fn has_fallback(&self) -> bool {
        self.stages
            .iter()
            .any(|record| record.source == GenerationSource::Fallback)
    }

    /// Whether any stage deviated from an accepted model answer.
    #[must_use]
    pub fn has_deviation(&self) -> bool {
        self.stages
            .iter()
            .any(|record| record.source != GenerationSource::Model)
    }

    /// How many stages fell back from a failed model answer.
    #[must_use]
    pub fn fallback_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|record| record.source == GenerationSource::Fallback)
            .count()
    }

    /// Every record whose source is `source`, oldest first.
    #[must_use]
    pub fn stages_matching(&self, source: GenerationSource) -> Vec<&StageProvenance> {
        self.stages
            .iter()
            .filter(|record| record.source == source)
            .collect()
    }

    /// Logs one `warn!` per stage whose output did not come from a model.
    ///
    /// The warning names the stage and its recorded detail, so a degraded build
    /// is visible without changing any result type.
    /// Logging goes through [`tracing::warn!`], which is the `tracing` stack
    /// [`openjiuwen_common::logging`] configures (that module re-exports the
    /// macros and owns the subscriber, and [`openjiuwen_common::logging`] offers
    /// no warning helper beyond them).
    pub fn warn_if_degraded(&self) {
        for record in &self.stages {
            if record.source == GenerationSource::Model {
                continue;
            }
            tracing::warn!(
                stage = record.stage.as_str(),
                source = record.source.as_str(),
                detail = %record.detail,
                "build stage output was not model-generated"
            );
        }
    }
}

/// One bounded repair/retry episode of a build pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RepairReport {
    /// How many generate/reflect attempts ran; `0` when no pipeline ran.
    pub attempts: usize,
    /// One note per re-attempt or rejected model revision, oldest first.
    pub notes: Vec<String>,
}

impl RepairReport {
    /// Whether no attempt ran and no note was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attempts == 0 && self.notes.is_empty()
    }

    /// Counts one more generate/reflect attempt.
    pub fn record_attempt(&mut self) {
        self.attempts += 1;
    }

    /// Appends one note, e.g. why a model revision was rejected.
    pub fn record_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

/// The result of one [`AgentBuilder`](crate::agent_builder::builder::AgentBuilder)
/// call.
///
/// [`Self::to_json`] flattens the output into the Python response shape, so the
/// JSON has `status`, `session_id`, `agent_type`, and exactly one of `dsl`,
/// `workflow`, or `response`.
/// The container-level `default` attribute makes both provenance fields optional
/// on the wire, so a result serialized before they existed still deserializes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildResult {
    /// The status string produced by [`map_state_to_status`].
    pub status: String,
    /// The session the call belongs to.
    pub session_id: String,
    /// The artifact the session is building.
    pub agent_type: AgentType,
    /// What the call produced.
    pub output: BuildOutput,
    /// How the produced artifact was generated.
    pub provenance: GenerationProvenance,
    /// The bounded repair/retry episode that produced a workflow artifact.
    pub repair: RepairReport,
}

impl BuildResult {
    /// Flattens the result into the Python response object.
    ///
    /// The four Python keys (`status`, `session_id`, `agent_type`, and the one
    /// output key) are always present, so a purely model-driven first-attempt
    /// result serializes exactly as before.
    /// `provenance` is appended only when [`GenerationProvenance::has_deviation`]
    /// reports a non-model stage, and `repair` only when [`RepairReport::is_empty`]
    /// is false; both values are the serialized record objects.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the output cannot be serialized or is
    /// not a JSON object, or when the provenance or repair report cannot be
    /// serialized.
    pub fn to_json(&self) -> Result<Value> {
        let mut map = Map::new();
        map.insert("status".to_string(), Value::String(self.status.clone()));
        map.insert(
            "session_id".to_string(),
            Value::String(self.session_id.clone()),
        );
        map.insert(
            "agent_type".to_string(),
            Value::String(self.agent_type.as_str().to_string()),
        );
        match self.output.as_json()? {
            Value::Object(entries) => {
                for (key, value) in entries {
                    map.insert(key, value);
                }
            }
            _ => {
                return Err(Error::validation(
                    "build output did not serialize to a JSON object",
                ));
            }
        }
        if self.provenance.has_deviation() {
            let provenance = serde_json::to_value(&self.provenance).map_err(|error| {
                Error::validation(format!("failed to serialize build provenance: {error}"))
            })?;
            map.insert("provenance".to_string(), provenance);
        }
        if !self.repair.is_empty() {
            let repair = serde_json::to_value(&self.repair).map_err(|error| {
                Error::validation(format!("failed to serialize repair report: {error}"))
            })?;
            map.insert("repair".to_string(), repair);
        }
        Ok(Value::Object(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The workflow document used by the output and result round-trip tests.
    fn parsed_workflow() -> WorkflowDsl {
        WorkflowDsl::from_json_str(
            r#"{"name": "demo", "description": "a demo workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"}, {"id": "end", "type": "end"}],
 "edges": [{"source": "start", "target": "end"}]}"#,
        )
        .expect("valid workflow document")
    }

    #[test]
    fn status_mapping_covers_every_state_and_agent_type() {
        let cases = [
            (BuildState::Initial, AgentType::LlmAgent, "clarifying"),
            (BuildState::Initial, AgentType::Workflow, "requesting"),
            (BuildState::Clarifying, AgentType::LlmAgent, "clarifying"),
            (BuildState::Clarifying, AgentType::Workflow, "clarifying"),
            (BuildState::Requesting, AgentType::LlmAgent, "requesting"),
            (BuildState::Requesting, AgentType::Workflow, "requesting"),
            (BuildState::Processing, AgentType::LlmAgent, "processing"),
            (BuildState::Processing, AgentType::Workflow, "processing"),
            (BuildState::Completed, AgentType::LlmAgent, "completed"),
            (BuildState::Completed, AgentType::Workflow, "completed"),
        ];
        for (state, agent_type, expected) in cases {
            assert_eq!(map_state_to_status(state, agent_type), expected);
        }
        let distinct: std::collections::BTreeSet<&str> = cases.iter().map(|case| case.2).collect();
        assert_eq!(
            distinct.into_iter().collect::<Vec<_>>(),
            vec!["clarifying", "completed", "processing", "requesting"]
        );
    }

    #[test]
    fn enums_serialize_to_their_wire_names() {
        assert_eq!(
            serde_json::to_value(AgentType::LlmAgent).expect("serialize"),
            json!("llm_agent")
        );
        assert_eq!(
            serde_json::to_value(AgentType::Workflow).expect("serialize"),
            json!("workflow")
        );
        assert!(serde_json::from_value::<AgentType>(json!("other")).is_err());
        assert_eq!(AgentType::default(), AgentType::LlmAgent);
        assert_eq!(BuildState::default(), BuildState::Initial);
        for (state, expected) in [
            (BuildState::Initial, "initial"),
            (BuildState::Clarifying, "clarifying"),
            (BuildState::Requesting, "requesting"),
            (BuildState::Processing, "processing"),
            (BuildState::Completed, "completed"),
        ] {
            assert_eq!(
                serde_json::to_value(state).expect("serialize"),
                json!(expected)
            );
            assert_eq!(
                serde_json::from_value::<BuildState>(json!(expected)).expect("read"),
                state
            );
        }

        let status = BuildStatus::default();
        assert_eq!(status.session_id, "");
        assert_eq!(status.agent_type, AgentType::LlmAgent);
        assert_eq!(status.state, BuildState::Initial);
        assert_eq!(status.resource_count, 0);
        assert_eq!(
            serde_json::from_value::<BuildStatus>(json!({"session_id": "s1"})).expect("read"),
            BuildStatus {
                session_id: "s1".to_string(),
                ..BuildStatus::default()
            }
        );
    }

    #[test]
    fn build_output_serializes_each_variant_and_round_trips() {
        let dsl = LlmAgentDsl::from_json_str(r#"{"agent_id": "agent-1"}"#).expect("parse");
        let dsl_json = BuildOutput::Dsl(dsl.clone()).as_json().expect("serialize");
        assert_eq!(dsl_json, json!({"dsl": dsl.to_value().expect("value")}));

        let workflow_json = BuildOutput::Workflow(parsed_workflow())
            .as_json()
            .expect("serialize");
        assert_eq!(workflow_json["workflow"]["name"], json!("demo"));
        assert_eq!(workflow_json.as_object().expect("object").len(), 1);

        let response = BuildOutput::Response("which agent?".to_string());
        assert_eq!(
            response.as_json().expect("serialize"),
            json!({"response": "which agent?"})
        );
        assert_eq!(BuildOutput::default(), BuildOutput::Response(String::new()));

        let outputs = [
            BuildOutput::Dsl(dsl),
            BuildOutput::Workflow(parsed_workflow()),
            response,
        ];
        for output in outputs {
            let value = output.as_json().expect("serialize");
            let restored: BuildOutput = serde_json::from_value(value.clone()).expect("read");
            assert_eq!(restored, output);
        }
    }

    #[test]
    fn build_result_flattens_the_output() {
        let result = BuildResult {
            status: "completed".to_string(),
            session_id: "s1".to_string(),
            agent_type: AgentType::LlmAgent,
            output: BuildOutput::Response("hi".to_string()),
            ..BuildResult::default()
        };
        let value = result.to_json().expect("serialize");
        let keys: Vec<&String> = value.as_object().expect("object").keys().collect();
        assert_eq!(keys, vec!["status", "session_id", "agent_type", "response"]);
        assert_eq!(value["agent_type"], json!("llm_agent"));
        assert_eq!(value["response"], json!("hi"));
        assert!(value.get("output").is_none());
        assert!(value.get("provenance").is_none());
        assert!(value.get("repair").is_none());

        let workflow = BuildResult {
            output: BuildOutput::Workflow(parsed_workflow()),
            agent_type: AgentType::Workflow,
            ..BuildResult::default()
        };
        let value = workflow.to_json().expect("serialize");
        assert_eq!(value["workflow"]["name"], json!("demo"));
        assert!(value.get("response").is_none());
    }

    #[test]
    fn generation_source_round_trips_through_its_wire_names() {
        assert_eq!(GenerationSource::default(), GenerationSource::Model);
        for (source, wire_name) in [
            (GenerationSource::Model, "model"),
            (GenerationSource::Deterministic, "deterministic"),
            (GenerationSource::Fallback, "fallback"),
        ] {
            assert_eq!(source.as_str(), wire_name);
            assert_eq!(
                serde_json::to_value(source).expect("serialize"),
                json!(wire_name)
            );
            assert_eq!(
                serde_json::from_value::<GenerationSource>(json!(wire_name)).expect("read"),
                source
            );
        }
        assert!(serde_json::from_value::<GenerationSource>(json!("guessed")).is_err());
    }

    #[test]
    fn generation_provenance_records_and_queries_stages() {
        let empty = GenerationProvenance::default();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(!empty.has_fallback());
        assert!(!empty.has_deviation());
        assert_eq!(empty.fallback_count(), 0);
        let no_model_stages: Vec<&StageProvenance> = Vec::new();
        assert_eq!(
            empty.stages_matching(GenerationSource::Model),
            no_model_stages
        );
        assert_eq!(GenerationProvenance::new(), empty);

        let mut provenance = GenerationProvenance::new();
        provenance.record_model(ProgressStage::Clarifying);
        provenance.record_deterministic(
            ProgressStage::GeneratingWorkflowDesign,
            "no model wired for the designer",
        );
        provenance.record_fallback(
            ProgressStage::GeneratingDl,
            "model answer was not a valid workflow document",
        );
        provenance.record(ProgressStage::ValidatingDl, GenerationSource::Model, "");

        assert_eq!(provenance.len(), 4);
        assert!(!provenance.is_empty());
        assert!(provenance.has_fallback());
        assert!(provenance.has_deviation());
        assert_eq!(provenance.fallback_count(), 1);
        assert_eq!(
            provenance.stages,
            vec![
                StageProvenance {
                    stage: ProgressStage::Clarifying,
                    source: GenerationSource::Model,
                    detail: String::new(),
                },
                StageProvenance {
                    stage: ProgressStage::GeneratingWorkflowDesign,
                    source: GenerationSource::Deterministic,
                    detail: "no model wired for the designer".to_string(),
                },
                StageProvenance {
                    stage: ProgressStage::GeneratingDl,
                    source: GenerationSource::Fallback,
                    detail: "model answer was not a valid workflow document".to_string(),
                },
                StageProvenance {
                    stage: ProgressStage::ValidatingDl,
                    source: GenerationSource::Model,
                    detail: String::new(),
                },
            ]
        );
        assert_eq!(
            provenance
                .stages_matching(GenerationSource::Model)
                .iter()
                .map(|record| record.stage)
                .collect::<Vec<_>>(),
            vec![ProgressStage::Clarifying, ProgressStage::ValidatingDl]
        );
        assert_eq!(
            provenance.stages_matching(GenerationSource::Fallback),
            vec![&provenance.stages[2]]
        );

        let mut joined = GenerationProvenance::new();
        joined.record_model(ProgressStage::Initializing);
        joined.extend_from(provenance.clone());
        assert_eq!(joined.len(), 5);
        assert_eq!(joined.stages[0].stage, ProgressStage::Initializing);
        assert_eq!(joined.stages[1..], provenance.stages[..]);

        // The degraded log is a pure observer: it never mutates the record.
        joined.warn_if_degraded();
        GenerationProvenance::new().warn_if_degraded();
        assert_eq!(joined.fallback_count(), 1);

        let value = serde_json::to_value(&provenance).expect("serialize");
        assert_eq!(
            value,
            json!({"stages": [
                {"stage": "clarifying", "source": "model", "detail": ""},
                {"stage": "generating_workflow_design", "source": "deterministic",
                 "detail": "no model wired for the designer"},
                {"stage": "generating_dl", "source": "fallback",
                 "detail": "model answer was not a valid workflow document"},
                {"stage": "validating_dl", "source": "model", "detail": ""},
            ]})
        );
        let restored: GenerationProvenance = serde_json::from_value(value).expect("deserialize");
        assert_eq!(restored, provenance);
        assert_eq!(
            serde_json::from_value::<GenerationProvenance>(json!({})).expect("read"),
            GenerationProvenance::default()
        );
    }

    #[test]
    fn repair_report_counts_attempts_and_notes() {
        let mut repair = RepairReport::default();
        assert!(repair.is_empty());
        assert_eq!(repair.attempts, 0);
        assert_eq!(repair.notes, Vec::<String>::new());

        repair.record_attempt();
        assert!(!repair.is_empty());
        assert_eq!(repair.attempts, 1);
        repair.record_note("reflection repaired a dangling edge");
        repair.record_attempt();
        repair.record_note("model revision rejected: missing end node");

        assert_eq!(repair.attempts, 2);
        assert_eq!(
            repair.notes,
            vec![
                "reflection repaired a dangling edge".to_string(),
                "model revision rejected: missing end node".to_string(),
            ]
        );
        assert!(!repair.is_empty());

        let value = serde_json::to_value(&repair).expect("serialize");
        assert_eq!(
            value,
            json!({"attempts": 2, "notes": [
                "reflection repaired a dangling edge",
                "model revision rejected: missing end node",
            ]})
        );
        let restored: RepairReport = serde_json::from_value(value).expect("deserialize");
        assert_eq!(restored, repair);
        assert_eq!(
            serde_json::from_value::<RepairReport>(json!({})).expect("read"),
            RepairReport::default()
        );
        assert_eq!(
            serde_json::to_value(RepairReport::default()).expect("serialize"),
            json!({"attempts": 0, "notes": []})
        );
    }

    #[test]
    fn build_result_reads_legacy_json_and_round_trips_provenance() {
        let legacy: BuildResult = serde_json::from_value(json!({
            "status": "completed",
            "session_id": "s1",
            "agent_type": "llm_agent",
            "response": "hi",
        }))
        .expect("legacy result");
        assert_eq!(legacy.status, "completed");
        assert_eq!(legacy.session_id, "s1");
        assert_eq!(legacy.agent_type, AgentType::LlmAgent);
        assert_eq!(legacy.output, BuildOutput::default());
        assert_eq!(legacy.provenance, GenerationProvenance::default());
        assert_eq!(legacy.repair, RepairReport::default());

        let mut provenance = GenerationProvenance::new();
        provenance.record_model(ProgressStage::Clarifying);
        provenance.record_fallback(ProgressStage::GeneratingDl, "unparseable answer");
        let mut repair = RepairReport::default();
        repair.record_attempt();
        repair.record_note("reflection repaired a cycle");
        let full = BuildResult {
            status: "completed".to_string(),
            session_id: "s2".to_string(),
            agent_type: AgentType::Workflow,
            output: BuildOutput::Workflow(parsed_workflow()),
            provenance,
            repair,
        };
        let value = serde_json::to_value(&full).expect("serialize");
        let restored: BuildResult = serde_json::from_value(value).expect("deserialize");
        assert_eq!(restored, full);
        assert_eq!(restored.provenance.len(), 2);
        assert_eq!(restored.repair.attempts, 1);
    }

    #[test]
    fn build_result_json_gains_provenance_only_when_degraded() {
        // A model-driven, first-attempt success keeps the four Python keys.
        let mut clean_provenance = GenerationProvenance::new();
        clean_provenance.record_model(ProgressStage::Clarifying);
        let clean = BuildResult {
            status: "completed".to_string(),
            session_id: "s1".to_string(),
            agent_type: AgentType::LlmAgent,
            output: BuildOutput::Response("hi".to_string()),
            provenance: clean_provenance,
            repair: RepairReport::default(),
        };
        let value = clean.to_json().expect("serialize");
        let keys: Vec<&String> = value.as_object().expect("object").keys().collect();
        assert_eq!(keys, vec!["status", "session_id", "agent_type", "response"]);

        // A degraded stage appends `provenance` and leaves the base keys alone.
        let mut provenance = GenerationProvenance::new();
        provenance.record_model(ProgressStage::Clarifying);
        provenance.record_fallback(ProgressStage::GeneratingConfig, "model answer was garbage");
        let degraded = BuildResult {
            provenance,
            ..clean.clone()
        };
        let value = degraded.to_json().expect("serialize");
        let keys: Vec<&String> = value.as_object().expect("object").keys().collect();
        assert_eq!(
            keys,
            vec![
                "status",
                "session_id",
                "agent_type",
                "response",
                "provenance"
            ]
        );
        assert_eq!(
            value["provenance"],
            json!({"stages": [
                {"stage": "clarifying", "source": "model", "detail": ""},
                {"stage": "generating_config", "source": "fallback",
                 "detail": "model answer was garbage"},
            ]})
        );
        assert!(value.get("repair").is_none());

        // A repair episode appends `repair` even when nothing degraded.
        let mut repair = RepairReport::default();
        repair.record_attempt();
        repair.record_note("model revision rejected: cycle");
        let repaired = BuildResult { repair, ..clean };
        let value = repaired.to_json().expect("serialize");
        let keys: Vec<&String> = value.as_object().expect("object").keys().collect();
        assert_eq!(
            keys,
            vec!["status", "session_id", "agent_type", "response", "repair"]
        );
        assert_eq!(
            value["repair"],
            json!({"attempts": 1, "notes": ["model revision rejected: cycle"]})
        );
        assert!(value.get("provenance").is_none());
    }
}
