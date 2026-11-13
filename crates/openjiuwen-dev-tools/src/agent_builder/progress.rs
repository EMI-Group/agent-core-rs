//! Stage weights ported from `agent_builder/utils/progress.py`.
//!
//! Python's `ProgressReporter` carries mutable per-session state, wall-clock
//! timestamps, observer callbacks, and a process-global `ProgressManager`
//! singleton. This port keeps only the deterministic core: the [`ProgressStage`]
//! vocabulary from `agent_builder/utils/enums.py`, the two
//! `_calculate_progress` weight tables ([`LLM_AGENT_STAGE_WEIGHTS`] and
//! [`WORKFLOW_STAGE_WEIGHTS`]), and [`progress_for`], which looks a stage up in
//! the table of one agent type and reports `0` for a stage that agent type never
//! reaches — exactly Python's `.get(stage, 0.0)` fallback.
//!
//! Progress is therefore a pure function of `(agent_type, stage)`: no clock, no
//! callbacks, no global state. [`crate::agent_builder::outcome::BuildStatus`]
//! exposes it through `BuildStatus::progress` without changing the serialized
//! snapshot.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::builder::AgentType;

/// The build stages Python reports progress for.
///
/// Mirrors `ProgressStage` in `agent_builder/utils/enums.py`; the wire names are
/// the Python values.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ProgressStage {
    /// The build started (0% for both agent types).
    #[default]
    Initializing,
    /// The builder is clarifying the request.
    Clarifying,
    /// Platform resources are being retrieved.
    ResourceRetrieving,
    /// The artifact was produced (100% for both agent types).
    Completed,
    /// The build failed.
    Error,
    /// The artifact is being optimised.
    Optimizing,
    /// The LLM agent configuration is being generated.
    GeneratingConfig,
    /// The LLM agent configuration is being transformed into the platform DSL.
    TransformingDsl,
    /// The workflow agent's intention is being detected.
    DetectingIntention,
    /// The workflow design is being generated.
    GeneratingWorkflowDesign,
    /// The design is being transformed into DL nodes.
    GeneratingDl,
    /// The DL nodes are being validated.
    ValidatingDl,
    /// The DL nodes are being refined.
    RefiningDl,
    /// The design is being rendered as Mermaid.
    TransformingMermaid,
    /// The workflow document is being transformed into the platform DSL.
    TransformingWorkflowDsl,
}

impl ProgressStage {
    /// The stage's wire name, e.g. `"transforming_workflow_dsl"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initializing => "initializing",
            Self::Clarifying => "clarifying",
            Self::ResourceRetrieving => "resource_retrieving",
            Self::Completed => "completed",
            Self::Error => "error",
            Self::Optimizing => "optimizing",
            Self::GeneratingConfig => "generating_config",
            Self::TransformingDsl => "transforming_dsl",
            Self::DetectingIntention => "detecting_intention",
            Self::GeneratingWorkflowDesign => "generating_workflow_design",
            Self::GeneratingDl => "generating_dl",
            Self::ValidatingDl => "validating_dl",
            Self::RefiningDl => "refining_dl",
            Self::TransformingMermaid => "transforming_mermaid",
            Self::TransformingWorkflowDsl => "transforming_workflow_dsl",
        }
    }
}

impl fmt::Display for ProgressStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The `llm_agent` progress table from `ProgressReporter._calculate_progress`.
pub const LLM_AGENT_STAGE_WEIGHTS: [(ProgressStage, u8); 6] = [
    (ProgressStage::Initializing, 0),
    (ProgressStage::Clarifying, 20),
    (ProgressStage::ResourceRetrieving, 40),
    (ProgressStage::GeneratingConfig, 60),
    (ProgressStage::TransformingDsl, 80),
    (ProgressStage::Completed, 100),
];

/// The `workflow` progress table from `ProgressReporter._calculate_progress`.
pub const WORKFLOW_STAGE_WEIGHTS: [(ProgressStage, u8); 9] = [
    (ProgressStage::Initializing, 0),
    (ProgressStage::DetectingIntention, 10),
    (ProgressStage::GeneratingWorkflowDesign, 25),
    (ProgressStage::GeneratingDl, 45),
    (ProgressStage::ValidatingDl, 60),
    (ProgressStage::RefiningDl, 70),
    (ProgressStage::TransformingMermaid, 85),
    (ProgressStage::TransformingWorkflowDsl, 95),
    (ProgressStage::Completed, 100),
];

/// The overall progress percentage (0-100) of `stage` for `agent_type`.
///
/// A stage the agent type never reaches reports `0`, matching Python's
/// `stage_weights.get(stage, 0.0)`.
///
/// # Example
///
/// ```
/// use openjiuwen_dev_tools::agent_builder::progress::{progress_for, ProgressStage};
/// use openjiuwen_dev_tools::agent_builder::AgentType;
///
/// assert_eq!(
///     progress_for(AgentType::LlmAgent, ProgressStage::TransformingDsl),
///     80
/// );
/// assert_eq!(
///     progress_for(AgentType::Workflow, ProgressStage::TransformingDsl),
///     0
/// );
/// ```
#[must_use]
pub fn progress_for(agent_type: AgentType, stage: ProgressStage) -> u8 {
    let table: &[(ProgressStage, u8)] = match agent_type {
        AgentType::LlmAgent => &LLM_AGENT_STAGE_WEIGHTS,
        AgentType::Workflow => &WORKFLOW_STAGE_WEIGHTS,
    };
    table
        .iter()
        .find(|(listed, _)| *listed == stage)
        .map_or(0, |(_, weight)| *weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_builder::outcome::{BuildState, BuildStatus};
    use serde_json::{json, Value};

    /// Every stage with its wire name, in Python's declaration order.
    const ALL_STAGES: [(ProgressStage, &str); 15] = [
        (ProgressStage::Initializing, "initializing"),
        (ProgressStage::Clarifying, "clarifying"),
        (ProgressStage::ResourceRetrieving, "resource_retrieving"),
        (ProgressStage::Completed, "completed"),
        (ProgressStage::Error, "error"),
        (ProgressStage::Optimizing, "optimizing"),
        (ProgressStage::GeneratingConfig, "generating_config"),
        (ProgressStage::TransformingDsl, "transforming_dsl"),
        (ProgressStage::DetectingIntention, "detecting_intention"),
        (
            ProgressStage::GeneratingWorkflowDesign,
            "generating_workflow_design",
        ),
        (ProgressStage::GeneratingDl, "generating_dl"),
        (ProgressStage::ValidatingDl, "validating_dl"),
        (ProgressStage::RefiningDl, "refining_dl"),
        (ProgressStage::TransformingMermaid, "transforming_mermaid"),
        (
            ProgressStage::TransformingWorkflowDsl,
            "transforming_workflow_dsl",
        ),
    ];

    #[test]
    fn llm_agent_table_matches_python() {
        assert_eq!(LLM_AGENT_STAGE_WEIGHTS.len(), 6);
        assert_eq!(
            LLM_AGENT_STAGE_WEIGHTS,
            [
                (ProgressStage::Initializing, 0),
                (ProgressStage::Clarifying, 20),
                (ProgressStage::ResourceRetrieving, 40),
                (ProgressStage::GeneratingConfig, 60),
                (ProgressStage::TransformingDsl, 80),
                (ProgressStage::Completed, 100),
            ]
        );
        for (stage, weight) in LLM_AGENT_STAGE_WEIGHTS {
            assert_eq!(progress_for(AgentType::LlmAgent, stage), weight);
        }
    }

    #[test]
    fn workflow_table_matches_python() {
        assert_eq!(WORKFLOW_STAGE_WEIGHTS.len(), 9);
        assert_eq!(
            WORKFLOW_STAGE_WEIGHTS,
            [
                (ProgressStage::Initializing, 0),
                (ProgressStage::DetectingIntention, 10),
                (ProgressStage::GeneratingWorkflowDesign, 25),
                (ProgressStage::GeneratingDl, 45),
                (ProgressStage::ValidatingDl, 60),
                (ProgressStage::RefiningDl, 70),
                (ProgressStage::TransformingMermaid, 85),
                (ProgressStage::TransformingWorkflowDsl, 95),
                (ProgressStage::Completed, 100),
            ]
        );
        for (stage, weight) in WORKFLOW_STAGE_WEIGHTS {
            assert_eq!(progress_for(AgentType::Workflow, stage), weight);
        }
    }

    #[test]
    fn unlisted_stages_report_zero() {
        // LLM-agent-only stages on a workflow, and vice versa.
        for stage in [
            ProgressStage::Clarifying,
            ProgressStage::ResourceRetrieving,
            ProgressStage::GeneratingConfig,
            ProgressStage::TransformingDsl,
        ] {
            assert_eq!(progress_for(AgentType::Workflow, stage), 0, "{stage}");
        }
        for stage in [
            ProgressStage::DetectingIntention,
            ProgressStage::GeneratingWorkflowDesign,
            ProgressStage::GeneratingDl,
            ProgressStage::ValidatingDl,
            ProgressStage::RefiningDl,
            ProgressStage::TransformingMermaid,
            ProgressStage::TransformingWorkflowDsl,
        ] {
            assert_eq!(progress_for(AgentType::LlmAgent, stage), 0, "{stage}");
        }
        // Stages no table lists.
        for stage in [ProgressStage::Error, ProgressStage::Optimizing] {
            assert_eq!(progress_for(AgentType::LlmAgent, stage), 0, "{stage}");
            assert_eq!(progress_for(AgentType::Workflow, stage), 0, "{stage}");
        }
    }

    #[test]
    fn every_stage_round_trips_and_renders_its_wire_name() {
        for (stage, wire_name) in ALL_STAGES {
            assert_eq!(stage.as_str(), wire_name);
            assert_eq!(stage.to_string(), wire_name);
            let value: Value = serde_json::to_value(stage).expect("serialize");
            assert_eq!(value, json!(wire_name));
            let restored: ProgressStage = serde_json::from_value(value).expect("deserialize");
            assert_eq!(restored, stage);
        }
        let distinct: std::collections::BTreeSet<&str> =
            ALL_STAGES.iter().map(|(_, name)| *name).collect();
        assert_eq!(distinct.len(), ALL_STAGES.len());
        assert_eq!(ProgressStage::default(), ProgressStage::Initializing);
        assert!(serde_json::from_value::<ProgressStage>(json!("unknown_stage")).is_err());
    }

    #[test]
    fn tables_cover_every_table_stage_exactly_once() {
        let listed: std::collections::BTreeSet<ProgressStage> = LLM_AGENT_STAGE_WEIGHTS
            .iter()
            .chain(WORKFLOW_STAGE_WEIGHTS.iter())
            .map(|(stage, _)| *stage)
            .collect();
        let all: std::collections::BTreeSet<ProgressStage> =
            ALL_STAGES.iter().map(|(stage, _)| *stage).collect();
        let unlisted: Vec<&ProgressStage> = all.difference(&listed).collect();
        assert_eq!(
            unlisted,
            vec![&ProgressStage::Error, &ProgressStage::Optimizing]
        );
        for table in [&LLM_AGENT_STAGE_WEIGHTS[..], &WORKFLOW_STAGE_WEIGHTS[..]] {
            let stages: std::collections::BTreeSet<ProgressStage> =
                table.iter().map(|(stage, _)| *stage).collect();
            assert_eq!(stages.len(), table.len());
        }
    }

    #[test]
    fn build_status_progress_delegates_to_progress_for() {
        let llm_status = BuildStatus {
            session_id: "s1".to_string(),
            agent_type: AgentType::LlmAgent,
            state: BuildState::Processing,
            resource_count: 2,
        };
        assert_eq!(
            llm_status.progress(ProgressStage::Clarifying),
            progress_for(AgentType::LlmAgent, ProgressStage::Clarifying)
        );
        assert_eq!(llm_status.progress(ProgressStage::Clarifying), 20);
        assert_eq!(llm_status.progress(ProgressStage::GeneratingDl), 0);

        let workflow_status = BuildStatus {
            agent_type: AgentType::Workflow,
            ..BuildStatus::default()
        };
        assert_eq!(workflow_status.progress(ProgressStage::GeneratingDl), 45);
        assert_eq!(workflow_status.progress(ProgressStage::Completed), 100);
        assert_eq!(workflow_status.progress(ProgressStage::Clarifying), 0);

        // The serialized snapshot is unchanged by the progress view.
        let value = serde_json::to_value(&llm_status).expect("serialize");
        assert_eq!(
            value,
            json!({
                "session_id": "s1",
                "agent_type": "llm_agent",
                "state": "processing",
                "resource_count": 2,
            })
        );
    }
}
