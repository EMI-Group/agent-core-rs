//! Signal records and the closed vocabularies they route through.
//!
//! Ports the signal data model from Python `openjiuwen.agent_evolving.signal.base`
//! and the signal records in `optimizer/base.py`, extended with the category and
//! target vocabularies the Rust evolution subsystems share:
//!
//! - [`EvolutionCategory`] — what kind of artefact a signal is about.
//! - [`EvolutionTarget`] — which layer of an operator/skill the signal should
//!   update; [`EvolutionTarget::category`] projects it onto a category.
//! - [`EvolutionSignal`] — one piece of feedback with a bounded excerpt, a
//!   category, a target and provenance metadata.
//!
//! Excerpt handling is deliberately textual and normalisation-free at
//! construction time: trimming is the only rewrite [`EvolutionSignal::new`]
//! performs, so the stored evidence stays recognisable. Dedupe and display use
//! [`normalize_excerpt`] and [`truncate_excerpt`], both deterministic.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The longest excerpt kept by the extractors, in characters.
///
/// Excerpts are evidence shown to a human and fed back into a prompt, so they
/// are bounded rather than unbounded slices of a conversation.
pub const MAX_EXCERPT_CHARS: usize = 240;

/// The kind of artefact an evolution signal is about.
///
/// The vocabulary is closed so optimizers can route on it exhaustively;
/// [`EvolutionCategory::Other`] carries everything the crate does not classify.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionCategory {
    /// Feedback about the operator description.
    Description,
    /// Feedback about the operator body (its main instruction block).
    Body,
    /// Feedback about a script or code asset.
    Script,
    /// Feedback about a tool implementation or its schema.
    Tool,
    /// Feedback about stored long-term memory.
    Memory,
    /// Feedback about a skill document.
    Skill,
    /// Feedback about a workflow definition.
    Workflow,
    /// Feedback about the context window, for example what was dropped.
    Context,
    /// Anything the crate does not classify.
    #[default]
    Other,
}

impl EvolutionCategory {
    /// The category as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Description => "description",
            Self::Body => "body",
            Self::Script => "script",
            Self::Tool => "tool",
            Self::Memory => "memory",
            Self::Skill => "skill",
            Self::Workflow => "workflow",
            Self::Context => "context",
            Self::Other => "other",
        }
    }
}

/// Which layer of an operator or skill a signal should update.
///
/// The target is the routing key of the evolution pipeline;
/// [`EvolutionTarget::category`] maps it onto the broader
/// [`EvolutionCategory`] vocabulary recorded on a signal.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionTarget {
    /// Feedback about instruction text; it lives in the description layer.
    Instruction,
    /// Feedback about the operator description.
    Description,
    /// Feedback about the operator body.
    Body,
    /// Feedback about a script or code asset.
    Script,
    /// Feedback about a tool implementation or its schema.
    Tool,
    /// Feedback about stored long-term memory.
    Memory,
    /// Feedback about a skill document.
    Skill,
    /// Feedback about the context window.
    Context,
    /// Anything the crate does not classify.
    #[default]
    Other,
}

impl EvolutionTarget {
    /// The target as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Instruction => "instruction",
            Self::Description => "description",
            Self::Body => "body",
            Self::Script => "script",
            Self::Tool => "tool",
            Self::Memory => "memory",
            Self::Skill => "skill",
            Self::Context => "context",
            Self::Other => "other",
        }
    }

    /// The category this target belongs to.
    ///
    /// [`EvolutionTarget::Instruction`] maps to
    /// [`EvolutionCategory::Description`] because instruction text lives in the
    /// description layer; there is no `Workflow` target, so a
    /// [`EvolutionCategory::Workflow`] signal is always recorded on a different
    /// target.
    #[must_use]
    pub const fn category(self) -> EvolutionCategory {
        match self {
            Self::Instruction | Self::Description => EvolutionCategory::Description,
            Self::Body => EvolutionCategory::Body,
            Self::Script => EvolutionCategory::Script,
            Self::Tool => EvolutionCategory::Tool,
            Self::Memory => EvolutionCategory::Memory,
            Self::Skill => EvolutionCategory::Skill,
            Self::Context => EvolutionCategory::Context,
            Self::Other => EvolutionCategory::Other,
        }
    }
}

/// Normalises an excerpt for dedupe: trim, collapse whitespace, lowercase.
///
/// This is the stable comparison form used by
/// [`EvolutionSignal::dedupe_key`]; it is intentionally lossy, so never show the
/// normalised text to a user.
#[must_use]
pub fn normalize_excerpt(excerpt: &str) -> String {
    excerpt
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Hard-cuts `text` to at most `max_chars` characters without an ellipsis.
///
/// The cut respects UTF-8 boundaries, so multi-byte characters are never split;
/// text shorter than `max_chars` is returned unchanged.
#[must_use]
pub fn truncate_excerpt(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((end, _)) => text[..end].to_owned(),
        None => text.to_owned(),
    }
}

/// One piece of feedback about one target of one operator, skill or tool.
///
/// The record is pure serde data: no model calls, no clocks and no environment
/// reads. Unknown JSON fields are ignored so newer producers stay readable, and
/// [`EvolutionSignal::validate`] is the explicit entry point for the bounds the
/// constructor enforces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EvolutionSignal {
    /// The signal kind, for example `execution_failure` or `low_score`.
    pub signal_type: String,
    /// The category the signal is about; defaults to
    /// [`EvolutionCategory::Other`] when absent.
    #[serde(default)]
    pub category: EvolutionCategory,
    /// The layer the signal should update; defaults to
    /// [`EvolutionTarget::Other`] when absent.
    #[serde(default)]
    pub target: EvolutionTarget,
    /// The target section of the document, for example `Troubleshooting`.
    #[serde(default)]
    pub section: String,
    /// The bounded evidence excerpt the signal was derived from.
    pub excerpt: String,
    /// The skill the signal applies to, when known.
    #[serde(default)]
    pub skill_name: Option<String>,
    /// The tool the signal applies to, when known.
    #[serde(default)]
    pub tool_name: Option<String>,
    /// Who produced the signal, for example `evaluator` or `review`.
    #[serde(default)]
    pub source: Option<String>,
    /// Confidence in `0.0..=1.0`; defaults to `1.0`.
    #[serde(default = "default_unit")]
    pub confidence: f64,
    /// Relative weight in `0.0..=1.0`; defaults to `1.0`.
    #[serde(default = "default_unit")]
    pub weight: f64,
    /// Free-form provenance, for example the case id or the failing score.
    #[serde(default)]
    pub context: BTreeMap<String, Value>,
}

impl EvolutionSignal {
    /// Creates a validated signal for `target`, deriving its category.
    ///
    /// Both `signal_type` and `excerpt` are trimmed; optional fields stay
    /// `None`, `section` is empty, and confidence/weight start at `1.0`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `signal_type` or `excerpt` is empty after
    /// trimming.
    pub fn new(
        signal_type: impl Into<String>,
        target: EvolutionTarget,
        excerpt: impl Into<String>,
    ) -> Result<Self> {
        let signal_type = signal_type.into();
        let signal_type = signal_type.trim();
        if signal_type.is_empty() {
            return Err(Error::validation(
                "EvolutionSignal.signal_type must not be empty",
            ));
        }
        let excerpt = excerpt.into();
        let excerpt = excerpt.trim();
        if excerpt.is_empty() {
            return Err(Error::validation(
                "EvolutionSignal.excerpt must not be empty",
            ));
        }
        Ok(Self {
            signal_type: signal_type.to_owned(),
            category: target.category(),
            target,
            section: String::new(),
            excerpt: excerpt.to_owned(),
            skill_name: None,
            tool_name: None,
            source: None,
            confidence: 1.0,
            weight: 1.0,
            context: BTreeMap::new(),
        })
    }

    /// Checks the invariants the constructor enforces on a deserialized value.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the first empty string or out-of-range
    /// numeric field.
    pub fn validate(&self) -> Result<()> {
        if self.signal_type.trim().is_empty() {
            return Err(Error::validation(
                "EvolutionSignal.signal_type must not be empty",
            ));
        }
        if self.excerpt.trim().is_empty() {
            return Err(Error::validation(
                "EvolutionSignal.excerpt must not be empty",
            ));
        }
        validate_unit("confidence", self.confidence)?;
        validate_unit("weight", self.weight)
    }

    /// Returns the signal with `category` replaced.
    #[must_use]
    pub fn with_category(mut self, category: EvolutionCategory) -> Self {
        self.category = category;
        self
    }

    /// Returns the signal with `section` replaced.
    #[must_use]
    pub fn with_section(mut self, section: impl Into<String>) -> Self {
        self.section = section.into();
        self
    }

    /// Returns the signal with its skill name set.
    #[must_use]
    pub fn with_skill(mut self, skill: impl Into<String>) -> Self {
        self.skill_name = Some(skill.into());
        self
    }

    /// Returns the signal with its tool name set.
    #[must_use]
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool_name = Some(tool.into());
        self
    }

    /// Returns the signal with its source set.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Returns the signal with `confidence` replaced.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `confidence` is not finite or outside
    /// `0.0..=1.0`.
    pub fn with_confidence(mut self, confidence: f64) -> Result<Self> {
        validate_unit("confidence", confidence)?;
        self.confidence = confidence;
        Ok(self)
    }

    /// Returns the signal with `weight` replaced.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `weight` is not finite or outside `0.0..=1.0`.
    pub fn with_weight(mut self, weight: f64) -> Result<Self> {
        validate_unit("weight", weight)?;
        self.weight = weight;
        Ok(self)
    }

    /// Returns the signal with its context map replaced.
    #[must_use]
    pub fn with_context(mut self, context: BTreeMap<String, Value>) -> Self {
        self.context = context;
        self
    }

    /// The signal kind.
    #[must_use]
    pub fn signal_type(&self) -> &str {
        &self.signal_type
    }

    /// The evidence excerpt.
    #[must_use]
    pub fn excerpt(&self) -> &str {
        &self.excerpt
    }

    /// The category the signal is about.
    #[must_use]
    pub const fn category(&self) -> EvolutionCategory {
        self.category
    }

    /// The layer the signal should update.
    #[must_use]
    pub const fn target(&self) -> EvolutionTarget {
        self.target
    }

    /// The confidence the producer assigned to the signal.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    /// The relative weight the producer assigned to the signal.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// The stable dedupe key: category, target and normalised excerpt.
    ///
    /// The key is deliberately readable (`"tool:tool:schema missing x"`) instead
    /// of hashed so collisions are debuggable, and it excludes `signal_type` so
    /// the same evidence reported under a different kind still dedupes.
    #[must_use]
    pub fn dedupe_key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.category.as_str(),
            self.target.as_str(),
            normalize_excerpt(&self.excerpt)
        )
    }
}

/// Default for confidence and weight: fully trusted.
fn default_unit() -> f64 {
    1.0
}

/// Rejects non-finite and out-of-range unit-interval values.
fn validate_unit(field: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::validation(format!(
            "EvolutionSignal.{field} must be finite and between 0.0 and 1.0, got {value}"
        )));
    }
    Ok(())
}
