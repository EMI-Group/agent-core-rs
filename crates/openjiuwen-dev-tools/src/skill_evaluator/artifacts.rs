//! Per-file artifact writers for an [`EvaluationReport`].
//!
//! The Python pipeline writes four machine-readable files into
//! `<output_dir>/evals/skill-tests/<skill-name>/`; this module provides the
//! Rust equivalent so a caller can persist one `EvaluationReport` in exactly the
//! layout the Python skills expect:
//!
//! | File | Report member | Writer |
//! |---|---|---|
//! | `llm_judge_score.json` | [`design`](EvaluationReport::design) | [`EvaluationReport::write_design_report`] |
//! | `skill_tester_report.json` | [`functional`](EvaluationReport::functional) | [`EvaluationReport::write_functional_report`] |
//! | `skill_safety_score.json` | [`safety`](EvaluationReport::safety) | [`EvaluationReport::write_safety_report`] |
//! | `final_score.json` | [`final_score`](EvaluationReport::final_score) | [`EvaluationReport::write_final_score`] |
//!
//! Every writer serializes the report member itself (no hand-assembled JSON, no
//! duplicated key lists), so the files always agree with
//! [`EvaluationReport::to_json`]. Each writer ensures its directory exists,
//! rejects a path that exists but is not a directory, and reports I/O failures
//! with the offending path.
//!
//! `final_score.json` is only written for a completed run
//! ([`FinalScore::is_completed`](super::report::FinalScore::is_completed)): a
//! blocked pipeline has no valid final score, which is exactly what the Python
//! skill enforces.
//!
//! Nothing here reads the clock or the environment; every path is caller-supplied
//! and handed to the OS as given.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::Serialize;

use super::report::EvaluationReport;

/// The aggregated pipeline score (`final_score.json`).
pub const FINAL_SCORE_ARTIFACT: &str = "final_score.json";
/// The design review score (`llm_judge_score.json`).
pub const DESIGN_ARTIFACT: &str = "llm_judge_score.json";
/// The safety review score (`skill_safety_score.json`).
pub const SAFETY_ARTIFACT: &str = "skill_safety_score.json";
/// The functional test report (`skill_tester_report.json`).
pub const FUNCTIONAL_ARTIFACT: &str = "skill_tester_report.json";

/// The files one [`EvaluationReport::write_artifacts`] call produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArtifactPaths {
    /// The directory the artifacts were written into.
    pub dir: PathBuf,
    /// The design score file.
    pub design: PathBuf,
    /// The safety score file.
    pub safety: PathBuf,
    /// The functional test report file.
    pub functional: PathBuf,
    /// The final score file; `None` when the pipeline did not complete.
    pub final_score: Option<PathBuf>,
}

impl EvaluationReport {
    /// Writes [`design`](Self::design) as `llm_judge_score.json`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when
    /// `dir` exists and is not a directory, and
    /// [`Error::external`](openjiuwen_common::error::Error::external) naming the
    /// path for a serialization or I/O failure.
    pub async fn write_design_report(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let path = artifact_path(dir.as_ref(), DESIGN_ARTIFACT).await?;
        write_json(&path, &self.design).await?;
        Ok(path)
    }

    /// Writes [`safety`](Self::safety) as `skill_safety_score.json`.
    ///
    /// # Errors
    ///
    /// As [`write_design_report`](Self::write_design_report).
    pub async fn write_safety_report(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let path = artifact_path(dir.as_ref(), SAFETY_ARTIFACT).await?;
        write_json(&path, &self.safety).await?;
        Ok(path)
    }

    /// Writes [`functional`](Self::functional) as `skill_tester_report.json`.
    ///
    /// # Errors
    ///
    /// As [`write_design_report`](Self::write_design_report).
    pub async fn write_functional_report(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let path = artifact_path(dir.as_ref(), FUNCTIONAL_ARTIFACT).await?;
        write_json(&path, &self.functional).await?;
        Ok(path)
    }

    /// Writes [`final_score`](Self::final_score) as `final_score.json`.
    ///
    /// Returns `None` — and writes nothing — unless the pipeline completed:
    /// a blocked run has no valid final score.
    ///
    /// # Errors
    ///
    /// As [`write_design_report`](Self::write_design_report).
    pub async fn write_final_score(&self, dir: impl AsRef<Path>) -> Result<Option<PathBuf>> {
        if !self.final_score.is_completed() {
            return Ok(None);
        }
        let path = artifact_path(dir.as_ref(), FINAL_SCORE_ARTIFACT).await?;
        write_json(&path, &self.final_score).await?;
        Ok(Some(path))
    }

    /// Writes all four artifacts into `dir`.
    ///
    /// # Errors
    ///
    /// As [`write_design_report`](Self::write_design_report). The writers run in
    /// the order design, safety, functional, final score, so a failure names the
    /// first artifact that could not be written.
    pub async fn write_artifacts(&self, dir: impl AsRef<Path>) -> Result<ArtifactPaths> {
        let dir = dir.as_ref();
        let design = self.write_design_report(dir).await?;
        let safety = self.write_safety_report(dir).await?;
        let functional = self.write_functional_report(dir).await?;
        let final_score = self.write_final_score(dir).await?;
        Ok(ArtifactPaths {
            dir: dir.to_path_buf(),
            design,
            safety,
            functional,
            final_score,
        })
    }
}

/// Ensures `dir` is a directory and returns the path of `file_name` inside it.
async fn artifact_path(dir: &Path, file_name: &str) -> Result<PathBuf> {
    ensure_directory(dir).await?;
    Ok(dir.join(file_name))
}

/// Creates `dir` (with parents) or confirms it already is a directory.
async fn ensure_directory(dir: &Path) -> Result<()> {
    match tokio::fs::metadata(dir).await {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(Error::validation(format!(
            "artifact directory `{}` exists and is not a directory",
            dir.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tokio::fs::create_dir_all(dir).await.map_err(|error| {
                Error::external(error)
                    .with_context(format!("creating artifact directory `{}`", dir.display()))
            })
        }
        Err(error) => Err(Error::external(error)
            .with_context(format!("inspecting artifact directory `{}`", dir.display()))),
    }
}

/// Serializes `value` as pretty JSON and writes it to `path`.
async fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(|error| {
        Error::external(error).with_context(format!("serializing artifact `{}`", path.display()))
    })?;
    tokio::fs::write(path, text.as_bytes())
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!("writing artifact `{}`", path.display()))
        })
}

#[cfg(test)]
mod tests {
    use openjiuwen_test_support::TempWorkspace;

    use super::*;
    use crate::skill_evaluator::report::{
        CaseSummary, DesignReport, EvaluatorScore, EvaluatorScores, FinalScore, FunctionalReport,
        SafetyReport, ScoreDimension, PIPELINE_COMPLETED,
    };
    use crate::skill_evaluator::{grade_for, AutoFailCode};

    use std::collections::BTreeMap;

    /// A completed report with one value in every member.
    fn completed_report() -> EvaluationReport {
        let mut dimensions = BTreeMap::new();
        dimensions.insert(
            "d1_knowledge_ratio".to_string(),
            ScoreDimension {
                score: 7.0,
                max: 10.0,
            },
        );
        let mut safety_dimensions = BTreeMap::new();
        safety_dimensions.insert(
            "d1_harmful_instruction".to_string(),
            ScoreDimension {
                score: 26.0,
                max: 30.0,
            },
        );
        EvaluationReport {
            design: DesignReport {
                skill_name: "demo".to_string(),
                design_score: 74.0,
                design_score_max: 100.0,
                design_score_pct: 0.74,
                grade: grade_for(0.74),
                knowledge_delta: 0.74,
                ready_for_functional_testing: true,
                dimensions,
            },
            functional: FunctionalReport {
                skill_name: "demo".to_string(),
                skill_path: "skills/demo".to_string(),
                timestamp: 1_700_000_000_000,
                summary: CaseSummary {
                    total: 4,
                    passed: 3,
                    partial: 1,
                    failed: 0,
                    pass_rate: 0.75,
                },
                results: Vec::new(),
                recommendations: vec!["Edge cases".to_string()],
            },
            safety: SafetyReport {
                skill_name: "demo".to_string(),
                score_pct: 0.83,
                gate_threshold: 0.8,
                blocked: false,
                auto_fail: None,
                dimensions: Some(safety_dimensions),
                findings: Vec::new(),
            },
            final_score: FinalScore {
                skill_name: "demo".to_string(),
                pipeline_status: PIPELINE_COMPLETED.to_string(),
                final_score_pct: 0.818,
                grade: grade_for(0.818),
                evaluators: EvaluatorScores {
                    e1_llm_judge: Some(EvaluatorScore::weighted(0.74, 0.4)),
                    e2_functional: EvaluatorScore::weighted(0.87, 0.6),
                },
                gates_passed: vec!["e3_safety".to_string()],
                recommendation: "ready for release".to_string(),
            },
            judge: None,
        }
    }

    fn keys(value: &serde_json::Value) -> Vec<String> {
        let mut keys: Vec<String> = value.as_object().expect("object").keys().cloned().collect();
        keys.sort();
        keys
    }

    fn sorted(keys: &[&str]) -> Vec<String> {
        let mut keys: Vec<String> = keys.iter().map(|key| (*key).to_string()).collect();
        keys.sort();
        keys
    }

    #[tokio::test]
    async fn each_writer_emits_the_python_artifact_shape() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let dir = workspace.path("evals/skill-tests/demo");
        let report = completed_report();

        let paths = report.write_artifacts(&dir).await.expect("artifacts");

        assert_eq!(paths.dir, dir);
        assert_eq!(paths.design, dir.join(DESIGN_ARTIFACT));
        assert_eq!(paths.safety, dir.join(SAFETY_ARTIFACT));
        assert_eq!(paths.functional, dir.join(FUNCTIONAL_ARTIFACT));
        assert_eq!(
            paths.final_score.as_ref(),
            Some(&dir.join(FINAL_SCORE_ARTIFACT))
        );

        let design: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(&paths.design)
                .await
                .expect("design"),
        )
        .expect("design json");
        assert_eq!(
            keys(&design),
            sorted(&[
                "skill_name",
                "design_score",
                "design_score_max",
                "design_score_pct",
                "grade",
                "knowledge_delta",
                "dimensions",
                "ready_for_functional_testing",
            ])
        );
        assert_eq!(design, serde_json::to_value(&report.design).expect("value"));
        assert!((design["design_score_pct"].as_f64().expect("number") - 0.74).abs() < 1e-9);

        let safety: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(&paths.safety)
                .await
                .expect("safety"),
        )
        .expect("safety json");
        assert_eq!(
            keys(&safety),
            sorted(&[
                "skill_name",
                "score_pct",
                "gate_threshold",
                "blocked",
                "auto_fail",
                "dimensions",
                "findings",
            ])
        );
        assert_eq!(safety, serde_json::to_value(&report.safety).expect("value"));

        let functional: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(&paths.functional)
                .await
                .expect("functional"),
        )
        .expect("functional json");
        assert_eq!(
            keys(&functional),
            sorted(&[
                "skill_name",
                "skill_path",
                "timestamp",
                "summary",
                "results",
                "recommendations",
            ])
        );

        let final_score: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(paths.final_score.as_ref().expect("final score path"))
                .await
                .expect("final score"),
        )
        .expect("final score json");
        assert_eq!(
            keys(&final_score),
            sorted(&[
                "skill_name",
                "pipeline_status",
                "final_score_pct",
                "grade",
                "evaluators",
                "gates_passed",
                "recommendation",
            ])
        );
        assert_eq!(
            final_score["evaluators"]["e1_llm_judge"]["contribution"],
            serde_json::json!(0.296)
        );
    }

    #[tokio::test]
    async fn a_blocked_run_skips_the_final_score_file() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let dir = workspace.path("blocked");
        let mut report = completed_report();
        report.safety.blocked = true;
        report.safety.auto_fail = Some(AutoFailCode::HardcodedCredentials);
        report.safety.score_pct = 0.0;
        report.safety.dimensions = None;
        report.final_score.pipeline_status = "BLOCKED_AT_E3".to_string();
        report.final_score.final_score_pct = 0.0;

        let paths = report.write_artifacts(&dir).await.expect("artifacts");

        assert!(paths.final_score.is_none());
        assert!(!dir.join(FINAL_SCORE_ARTIFACT).exists());
        assert!(dir.join(DESIGN_ARTIFACT).exists());
        let safety: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(&paths.safety)
                .await
                .expect("safety"),
        )
        .expect("safety json");
        assert_eq!(
            safety["auto_fail"],
            serde_json::json!("AF2: hardcoded credentials, API keys, or secrets")
        );
        assert_eq!(safety["dimensions"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn writers_create_nested_directories() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let report = completed_report();
        let nested = workspace.path("a/b/c");

        let path = report.write_design_report(&nested).await.expect("write");

        assert_eq!(path, nested.join(DESIGN_ARTIFACT));
        assert!(path.is_file());
    }

    #[tokio::test]
    async fn a_non_directory_target_is_a_validation_error() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let file = workspace.path("artifact.json");
        tokio::fs::write(&file, b"{}").await.expect("seed file");
        let report = completed_report();

        for result in [
            report.write_design_report(&file).await.map(|_| ()),
            report.write_safety_report(&file).await.map(|_| ()),
            report.write_functional_report(&file).await.map(|_| ()),
            report.write_final_score(&file).await.map(|_| ()),
            report.write_artifacts(&file).await.map(|_| ()),
        ] {
            let error = result.expect_err("not a directory");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
            assert!(error.to_string().contains("artifact.json"), "{error}");
            assert!(error.to_string().contains("not a directory"), "{error}");
        }
    }

    #[tokio::test]
    async fn artifact_names_match_the_python_pipeline() {
        assert_eq!(FINAL_SCORE_ARTIFACT, "final_score.json");
        assert_eq!(DESIGN_ARTIFACT, "llm_judge_score.json");
        assert_eq!(SAFETY_ARTIFACT, "skill_safety_score.json");
        assert_eq!(FUNCTIONAL_ARTIFACT, "skill_tester_report.json");
    }
}
