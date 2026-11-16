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
//! # Human-readable reports
//!
//! The Python pipeline also writes four prose reports beside the JSON files.
//! [`markdown`] renders them as deterministic strings and the writers here
//! persist them into the same directory:
//!
//! | File | Renderer | Writer |
//! |---|---|---|
//! | `llm_judge_report.md` | [`markdown::llm_judge_report`] | [`EvaluationReport::write_llm_judge_report`] |
//! | `skill_safety_report.md` | [`markdown::skill_safety_report`] | [`EvaluationReport::write_safety_report_md`] |
//! | `final_report.md` | [`markdown::final_report`] | [`EvaluationReport::write_final_report`] |
//! | `pipeline_report.md` | [`markdown::pipeline_report`] | [`EvaluationReport::write_pipeline_report`] |
//!
//! [`EvaluationReport::write_markdown_artifacts`] writes that set in one call and
//! returns the resulting [`MarkdownArtifactPaths`]. The skip rule mirrors
//! `final_score.json`: a completed pipeline writes `final_report.md` and not
//! `pipeline_report.md`, while a blocked pipeline writes `pipeline_report.md` and
//! not `final_report.md`; both writers return `None` when they skip.
//!
//! Nothing here reads the clock or the environment; every path is caller-supplied
//! and handed to the OS as given, and the report timestamp comes from
//! [`FunctionalReport::timestamp_iso`](super::report::FunctionalReport::timestamp_iso).

pub mod markdown;

#[cfg(test)]
mod tests;

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
/// The human-readable design review (`llm_judge_report.md`).
pub const LLM_JUDGE_REPORT_ARTIFACT: &str = "llm_judge_report.md";
/// The human-readable safety review (`skill_safety_report.md`).
pub const SAFETY_REPORT_ARTIFACT: &str = "skill_safety_report.md";
/// The completed-run pipeline report (`final_report.md`).
pub const FINAL_REPORT_ARTIFACT: &str = "final_report.md";
/// The blocked-run pipeline report (`pipeline_report.md`).
pub const PIPELINE_REPORT_ARTIFACT: &str = "pipeline_report.md";

/// The Markdown files one [`EvaluationReport::write_markdown_artifacts`] call
/// produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MarkdownArtifactPaths {
    /// The directory the reports were written into.
    pub dir: PathBuf,
    /// The design review report.
    pub llm_judge: PathBuf,
    /// The safety review report.
    pub safety: PathBuf,
    /// The completed-run pipeline report; `None` when the pipeline was blocked.
    pub final_report: Option<PathBuf>,
    /// The blocked-run pipeline report; `None` when the pipeline completed.
    pub pipeline_report: Option<PathBuf>,
}

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

    /// Writes [`design`](Self::design) as the Markdown `llm_judge_report.md`.
    ///
    /// The body is [`markdown::llm_judge_report`], so the file and the renderer
    /// cannot drift apart.
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when
    /// `dir` exists and is not a directory, and
    /// [`Error::external`](openjiuwen_common::error::Error::external) naming the
    /// path for an I/O failure.
    pub async fn write_llm_judge_report(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let path = artifact_path(dir.as_ref(), LLM_JUDGE_REPORT_ARTIFACT).await?;
        write_text(&path, &markdown::llm_judge_report(&self.design)).await?;
        Ok(path)
    }

    /// Writes [`safety`](Self::safety) as the Markdown `skill_safety_report.md`.
    ///
    /// The body is [`markdown::skill_safety_report`].
    ///
    /// # Errors
    ///
    /// As [`write_llm_judge_report`](Self::write_llm_judge_report).
    pub async fn write_safety_report_md(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let path = artifact_path(dir.as_ref(), SAFETY_REPORT_ARTIFACT).await?;
        write_text(&path, &markdown::skill_safety_report(&self.safety)).await?;
        Ok(path)
    }

    /// Writes the Markdown `final_report.md` for a completed pipeline.
    ///
    /// Returns `None` — and writes nothing — unless the pipeline completed,
    /// mirroring [`write_final_score`](Self::write_final_score): a blocked run
    /// has no final verdict, so it gets
    /// [`write_pipeline_report`](Self::write_pipeline_report) instead.
    ///
    /// # Errors
    ///
    /// As [`write_llm_judge_report`](Self::write_llm_judge_report).
    pub async fn write_final_report(&self, dir: impl AsRef<Path>) -> Result<Option<PathBuf>> {
        if !self.final_score.is_completed() {
            return Ok(None);
        }
        let path = artifact_path(dir.as_ref(), FINAL_REPORT_ARTIFACT).await?;
        write_text(&path, &markdown::final_report(self)).await?;
        Ok(Some(path))
    }

    /// Writes the Markdown `pipeline_report.md` for a blocked pipeline.
    ///
    /// Returns `None` — and writes nothing — when the pipeline completed, in
    /// which case [`write_final_report`](Self::write_final_report) renders the
    /// verdict instead.
    ///
    /// # Errors
    ///
    /// As [`write_llm_judge_report`](Self::write_llm_judge_report).
    pub async fn write_pipeline_report(&self, dir: impl AsRef<Path>) -> Result<Option<PathBuf>> {
        if self.final_score.is_completed() {
            return Ok(None);
        }
        let path = artifact_path(dir.as_ref(), PIPELINE_REPORT_ARTIFACT).await?;
        write_text(&path, &markdown::pipeline_report(self)).await?;
        Ok(Some(path))
    }

    /// Writes all four Markdown reports into `dir`.
    ///
    /// `final_report.md` and `pipeline_report.md` are mutually exclusive, so one
    /// of [`MarkdownArtifactPaths::final_report`] and
    /// [`MarkdownArtifactPaths::pipeline_report`] is always `None`.
    ///
    /// # Errors
    ///
    /// As [`write_llm_judge_report`](Self::write_llm_judge_report). The writers
    /// run in the order design, safety, final report, pipeline report, so a
    /// failure names the first report that could not be written.
    pub async fn write_markdown_artifacts(
        &self,
        dir: impl AsRef<Path>,
    ) -> Result<MarkdownArtifactPaths> {
        let dir = dir.as_ref();
        let llm_judge = self.write_llm_judge_report(dir).await?;
        let safety = self.write_safety_report_md(dir).await?;
        let final_report = self.write_final_report(dir).await?;
        let pipeline_report = self.write_pipeline_report(dir).await?;
        Ok(MarkdownArtifactPaths {
            dir: dir.to_path_buf(),
            llm_judge,
            safety,
            final_report,
            pipeline_report,
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

/// Writes `text` to `path`, naming the file on any I/O failure.
async fn write_text(path: &Path, text: &str) -> Result<()> {
    tokio::fs::write(path, text.as_bytes())
        .await
        .map_err(|error| {
            Error::external(error)
                .with_context(format!("writing markdown artifact `{}`", path.display()))
        })
}

pub use markdown::{final_report, llm_judge_report, pipeline_report, skill_safety_report};
