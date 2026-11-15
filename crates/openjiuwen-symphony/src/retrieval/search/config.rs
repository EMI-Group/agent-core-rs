//! The serializable configuration of the query-time search runtime.
//!
//! Ports the config-shaped part of Python
//! `openjiuwen.symphony.retrieval.search.runtime` (v0.1.20):
//! `runtime/types.py::ProgressiveRetrieverConfig` (the traversal bounds and the selection
//! mode) and `render/disclosure.py::DisclosureConfig` (the exposure knobs), flattened into
//! one [`ProgressiveSearchConfig`] the way Python's facade flattens them for the engine.
//!
//! # Two layers, one default
//!
//! Python has two layers of defaults: `RetrieverConfig` (the public facade, production
//! shape: full-tree flattening, compact codes, effectively unbounded exposure) and
//! `ProgressiveRetrieverConfig` (bare runtime defaults: 2 exposure levels, threshold 12,
//! nested tree). This crate keeps exactly one default —
//! [`ProgressiveSearchConfig::default`] is the **production** one, because that is what a
//! caller of the umbrella API gets — while
//! [`DisclosureConfig::default`] keeps the conservative Python values, so a caller can
//! start from production bounds and dial the disclosure back explicitly.
//!
//! # Rust-only knobs
//!
//! `max_chars`, `max_items_per_branch`, `max_description_chars` and `code_width` inside
//! [`DisclosureConfig`] are Rust-only bounds (Python's renderer takes the first three as
//! call arguments and derives the width from the codes), and
//! [`SelectionMode::Lexical`] is Rust-only: it selects without a model. A `0` value means
//! "unlimited" (or "derive it") everywhere.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{validation, Result};
use crate::llm::FallbackMode;
use crate::retrieval::search::codebooks::CompactCodebook;
use crate::retrieval::search::truncation::ScoreTruncationConfig;

/// The `top_k` of the public retriever facade (Python `RetrieverConfig.top_k`).
///
/// Python's facade defaults to 10 while the bare runtime defaults to 5, and a later wave's
/// facade keeps that split; the runtime uses
/// [`RUNTIME_TOP_K`](crate::retrieval::search::RUNTIME_TOP_K).
pub const DEFAULT_TOP_K: usize = 10;

/// How a subtree's candidates are selected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    /// Ask a model to generate the selected codes (Python's `generate` mode).
    #[default]
    Generate,
    /// Rank the candidates locally and select without a model (Rust-only).
    ///
    /// The lexical scorer in
    /// [`lexical`](crate::retrieval::search::lexical) supplies the ranking, so an offline
    /// deployment can walk the tree with no model at all.
    Lexical,
}

/// The exposure knobs of one disclosure rendering.
///
/// Mirrors Python `DisclosureConfig`'s five fields and their conservative defaults, plus
/// the four Rust-only bounds documented at the module level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DisclosureConfig {
    /// How many levels below the current node one call may expose (`0` exposes only the
    /// immediate children).
    pub max_exposure_depth_per_call: usize,
    /// Expand a child when its subtree holds at most this many items; a larger subtree
    /// collapses into one selectable branch.
    pub exposure_threshold: usize,
    /// Address candidates by compact code-book codes instead of numeric codes.
    pub compact_codes_enabled: bool,
    /// The code book to draw compact codes from; empty uses numeric codes `1`, `2`, ….
    pub compact_codebook: Vec<String>,
    /// Render the whole flattened tree in one prompt instead of a nested fragment.
    pub flatten_full_tree_in_prompt: bool,
    /// Rust-only: the rendered fragment's char budget; `0` is unlimited.
    pub max_chars: usize,
    /// Rust-only: how many items one branch may disclose; `0` is unlimited.
    pub max_items_per_branch: usize,
    /// Rust-only: the description clamp; `0` is unlimited.
    pub max_description_chars: usize,
    /// Rust-only: a code-width override; `0` derives it from the assigned codes.
    pub code_width: usize,
}

impl Default for DisclosureConfig {
    /// The conservative Python `DisclosureConfig` defaults.
    fn default() -> Self {
        Self {
            max_exposure_depth_per_call: 2,
            exposure_threshold: 12,
            compact_codes_enabled: false,
            compact_codebook: Vec::new(),
            flatten_full_tree_in_prompt: false,
            max_chars: 0,
            max_items_per_branch: 0,
            max_description_chars: 150,
            code_width: 0,
        }
    }
}

impl DisclosureConfig {
    /// Sets [`DisclosureConfig::max_exposure_depth_per_call`].
    #[must_use]
    pub fn with_max_exposure_depth_per_call(mut self, depth: usize) -> Self {
        self.max_exposure_depth_per_call = depth;
        self
    }

    /// Sets [`DisclosureConfig::exposure_threshold`].
    #[must_use]
    pub fn with_exposure_threshold(mut self, threshold: usize) -> Self {
        self.exposure_threshold = threshold;
        self
    }

    /// Enables or disables compact codes.
    #[must_use]
    pub fn with_compact_codes_enabled(mut self, enabled: bool) -> Self {
        self.compact_codes_enabled = enabled;
        self
    }

    /// Replaces the compact code book.
    #[must_use]
    pub fn with_compact_codebook(mut self, codebook: Vec<String>) -> Self {
        self.compact_codebook = codebook;
        self
    }

    /// Enables or disables full-tree flattening.
    #[must_use]
    pub fn with_flatten_full_tree_in_prompt(mut self, flatten: bool) -> Self {
        self.flatten_full_tree_in_prompt = flatten;
        self
    }

    /// Sets the Rust-only rendered-fragment char budget (`0` = unlimited).
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }

    /// Sets the Rust-only per-branch item cap (`0` = unlimited).
    #[must_use]
    pub fn with_max_items_per_branch(mut self, max_items: usize) -> Self {
        self.max_items_per_branch = max_items;
        self
    }

    /// Sets the Rust-only description clamp (`0` = unlimited).
    #[must_use]
    pub fn with_max_description_chars(mut self, max_chars: usize) -> Self {
        self.max_description_chars = max_chars;
        self
    }

    /// Sets the Rust-only code-width override (`0` = derive it).
    #[must_use]
    pub fn with_code_width(mut self, code_width: usize) -> Self {
        self.code_width = code_width;
        self
    }

    /// Validates the code book, mirroring Python's `_normalize_codebook` rejections.
    ///
    /// An empty code book is valid (numeric codes are used instead). A non-empty one must
    /// hold non-blank, whitespace-free, unique entries that are never the reserved abstain
    /// token `"0"`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `compact_codebook`.
    pub fn validate(&self) -> Result<()> {
        if self.compact_codebook.is_empty() {
            return Ok(());
        }
        CompactCodebook::new(self.compact_codebook.iter().cloned()).map(|_| ())
    }
}

/// How embedding scores are blended into the lexical ranking.
///
/// Rust-only: Python's runtime has no vector path. `weight` is the blend weight and
/// `max_documents` bounds how much of an index a build embeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EmbeddingScoringConfig {
    /// The blend weight: `0.0` is pure lexical (the default), `1.0` is pure embedding.
    pub weight: f64,
    /// How many documents one index build may embed; `0` is unlimited.
    pub max_documents: usize,
}

impl Default for EmbeddingScoringConfig {
    /// Pure lexical scoring, unlimited documents.
    fn default() -> Self {
        Self {
            weight: 0.0,
            max_documents: 0,
        }
    }
}

impl EmbeddingScoringConfig {
    /// Sets the blend weight, clamped into `[0, 1]` (a non-finite weight becomes `0.0`).
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = if weight.is_finite() {
            weight.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    /// Sets the embed cap (`0` = unlimited).
    #[must_use]
    pub fn with_max_documents(mut self, max_documents: usize) -> Self {
        self.max_documents = max_documents;
        self
    }

    /// Validates the blend weight.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming `embedding.weight` when it is not
    /// finite or outside `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if !self.weight.is_finite() || !(0.0..=1.0).contains(&self.weight) {
            return Err(validation(
                "embedding.weight",
                "must be a finite number in [0, 1]",
            ));
        }
        Ok(())
    }
}

/// The traversal configuration of a progressive search.
///
/// Mirrors Python `ProgressiveRetrieverConfig`, keeping one default: the production shape
/// the Python facade composes ([`ProgressiveSearchConfig::default`]). See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ProgressiveSearchConfig {
    /// How many results one search returns (Python `ProgressiveRetrieverConfig.top_k`).
    pub top_k: usize,
    /// The advisory generation cap handed to a model.
    pub max_tokens: usize,
    /// Rust-only: a hard bound on how deep the engine may descend; `0` is unlimited.
    pub max_depth: usize,
    /// Rust-only: a hard bound on how many nodes the engine may visit; `0` is unlimited.
    pub max_steps: usize,
    /// How many branches one selection may expand.
    pub max_branch_choices: usize,
    /// How many extra branches a selection may pick beyond `top_k`.
    pub branch_choice_slack: usize,
    /// How many extra candidates one branch may contribute beyond its share.
    pub branch_candidate_slack: usize,
    /// Reduce branch results round-robin so no branch monopolizes the list.
    pub round_robin_branch_reduce: bool,
    /// How a subtree's candidates are selected.
    pub selection_mode: SelectionMode,
    /// What a stage does when the model fails.
    pub fallback_mode: FallbackMode,
    /// The exposure knobs; the default carries Python's production overrides.
    pub disclosure: DisclosureConfig,
    /// The Rust-only embedding blend.
    pub embedding: EmbeddingScoringConfig,
    /// The score-truncation decision applied to a result list.
    pub truncation: ScoreTruncationConfig,
}

impl Default for ProgressiveSearchConfig {
    /// The runtime defaults with the production disclosure overrides Python's facade applies:
    /// `max_exposure_depth_per_call = 99`, `exposure_threshold = 1_000_000_000`,
    /// `compact_codes_enabled = true`, `flatten_full_tree_in_prompt = true`.
    ///
    /// The compact code book stays empty, so codes are numeric `1`, `2`, …; supply
    /// [`default_compact_boundary_codebook`](crate::retrieval::search::codebooks::default_compact_boundary_codebook)
    /// to address candidates by letters instead (see that function's docs for the deviation).
    fn default() -> Self {
        Self {
            top_k: crate::retrieval::search::RUNTIME_TOP_K,
            max_tokens: 96,
            max_depth: 0,
            max_steps: 0,
            max_branch_choices: 6,
            branch_choice_slack: 2,
            branch_candidate_slack: 1,
            round_robin_branch_reduce: true,
            selection_mode: SelectionMode::Generate,
            fallback_mode: FallbackMode::Deterministic,
            disclosure: DisclosureConfig {
                max_exposure_depth_per_call: 99,
                exposure_threshold: 1_000_000_000,
                compact_codes_enabled: true,
                flatten_full_tree_in_prompt: true,
                ..DisclosureConfig::default()
            },
            embedding: EmbeddingScoringConfig::default(),
            truncation: ScoreTruncationConfig::default(),
        }
    }
}

impl ProgressiveSearchConfig {
    /// Validates the config before a run uses it.
    ///
    /// The rules:
    ///
    /// - `top_k` must be at least `1` — a search that returns nothing is a caller mistake;
    /// - the disclosure code book obeys
    ///   [`DisclosureConfig::validate`];
    /// - `embedding.weight` is a finite number in `[0, 1]`;
    /// - the truncation thresholds are finite, and `autocut_min_relative_drop` lies in
    ///   `[0, 1]`: a relative drop above `1.0` asks for a drop larger than the previous
    ///   score, which no score list can satisfy, so it contradicts the autocut bound.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`](crate::error::Error) naming the offending field.
    pub fn validate(&self) -> Result<()> {
        if self.top_k == 0 {
            return Err(validation("top_k", "must be at least 1"));
        }
        self.disclosure.validate()?;
        self.embedding.validate()?;
        if let Some(min_score) = self.truncation.min_score {
            if !min_score.is_finite() {
                return Err(validation("truncation.min_score", "must be finite"));
            }
        }
        if let Some(relative) = self.truncation.relative_min_score {
            if !relative.is_finite() {
                return Err(validation(
                    "truncation.relative_min_score",
                    "must be finite",
                ));
            }
        }
        let drop = self.truncation.autocut_min_relative_drop;
        if !drop.is_finite() || !(0.0..=1.0).contains(&drop) {
            return Err(validation(
                "truncation.autocut_min_relative_drop",
                "must be a finite number in [0, 1]",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn disclosure_defaults_are_the_conservative_python_values() {
        let disclosure = DisclosureConfig::default();
        assert_eq!(disclosure.max_exposure_depth_per_call, 2);
        assert_eq!(disclosure.exposure_threshold, 12);
        assert!(!disclosure.compact_codes_enabled);
        assert_eq!(disclosure.compact_codebook, Vec::<String>::new());
        assert!(!disclosure.flatten_full_tree_in_prompt);
        assert_eq!(disclosure.max_chars, 0);
        assert_eq!(disclosure.max_items_per_branch, 0);
        assert_eq!(disclosure.max_description_chars, 150);
        assert_eq!(disclosure.code_width, 0);
    }

    #[test]
    fn runtime_defaults_are_the_python_runtime_values() {
        let config = ProgressiveSearchConfig::default();
        assert_eq!(config.top_k, 5);
        assert_eq!(config.max_tokens, 96);
        assert_eq!(config.max_depth, 0);
        assert_eq!(config.max_steps, 0);
        assert_eq!(config.max_branch_choices, 6);
        assert_eq!(config.branch_choice_slack, 2);
        assert_eq!(config.branch_candidate_slack, 1);
        assert!(config.round_robin_branch_reduce);
        assert_eq!(config.selection_mode, SelectionMode::Generate);
        assert_eq!(config.fallback_mode, FallbackMode::Deterministic);
        assert_eq!(config.embedding, EmbeddingScoringConfig::default());
        assert_eq!(config.truncation, ScoreTruncationConfig::default());
    }

    #[test]
    fn runtime_defaults_carry_the_production_disclosure_overrides() {
        let disclosure = ProgressiveSearchConfig::default().disclosure;
        assert_eq!(disclosure.max_exposure_depth_per_call, 99);
        assert_eq!(disclosure.exposure_threshold, 1_000_000_000);
        assert!(disclosure.compact_codes_enabled);
        assert!(disclosure.flatten_full_tree_in_prompt);
        // The Rust-only bounds are not part of the override and keep the conservative values.
        assert_eq!(disclosure.max_chars, 0);
        assert_eq!(disclosure.max_description_chars, 150);
        assert_eq!(disclosure.code_width, 0);
    }

    #[test]
    fn defaults_validate() {
        assert!(ProgressiveSearchConfig::default().validate().is_ok());
        assert_eq!(DEFAULT_TOP_K, 10);
    }

    #[test]
    fn validation_rejects_a_zero_top_k_and_a_blank_codebook_entry() {
        let mut config = ProgressiveSearchConfig {
            top_k: 0,
            ..ProgressiveSearchConfig::default()
        };
        let error = config.validate().unwrap_err();
        assert!(error.message().contains("top_k"), "{}", error.message());

        config.top_k = 5;
        config.disclosure.compact_codebook = vec!["AB".into(), "  ".into()];
        let error = config.validate().unwrap_err();
        assert!(error.message().contains("compact_codebook"));
    }

    #[test]
    fn validation_rejects_an_out_of_range_embedding_weight() {
        let mut config = ProgressiveSearchConfig {
            embedding: EmbeddingScoringConfig {
                weight: 1.5,
                max_documents: 0,
            },
            ..ProgressiveSearchConfig::default()
        };
        let error = config.validate().unwrap_err();
        assert!(error.message().contains("embedding.weight"));

        config.embedding.weight = f64::NAN;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_rejects_contradicting_truncation_bounds() {
        let mut config = ProgressiveSearchConfig::default();
        config.truncation.autocut_min_relative_drop = 1.5;
        assert!(config.validate().is_err());
        config.truncation.autocut_min_relative_drop = 0.25;
        assert!(config.validate().is_ok());
        config.truncation.min_score = Some(f64::INFINITY);
        assert!(config.validate().is_err());
        config.truncation.min_score = None;
        config.truncation.relative_min_score = Some(f64::NEG_INFINITY);
        assert!(config.validate().is_err());
    }

    #[test]
    fn with_weight_clamps_and_documents_the_clamp() {
        assert_eq!(
            EmbeddingScoringConfig::default().with_weight(2.0).weight,
            1.0
        );
        assert_eq!(
            EmbeddingScoringConfig::default().with_weight(-1.0).weight,
            0.0
        );
        assert_eq!(
            EmbeddingScoringConfig::default()
                .with_weight(f64::NAN)
                .weight,
            0.0
        );
        assert_eq!(
            EmbeddingScoringConfig::default().with_weight(0.25).weight,
            0.25
        );
        assert_eq!(
            EmbeddingScoringConfig::default()
                .with_max_documents(3)
                .max_documents,
            3
        );
    }

    #[test]
    fn disclosure_builders_replace_one_field_at_a_time() {
        let disclosure = DisclosureConfig::default()
            .with_max_exposure_depth_per_call(4)
            .with_exposure_threshold(20)
            .with_compact_codes_enabled(true)
            .with_compact_codebook(vec!["AA".into()])
            .with_flatten_full_tree_in_prompt(true)
            .with_max_chars(120)
            .with_max_items_per_branch(3)
            .with_max_description_chars(40)
            .with_code_width(2);
        assert_eq!(disclosure.max_exposure_depth_per_call, 4);
        assert_eq!(disclosure.exposure_threshold, 20);
        assert!(disclosure.compact_codes_enabled);
        assert_eq!(disclosure.compact_codebook, vec!["AA".to_owned()]);
        assert!(disclosure.flatten_full_tree_in_prompt);
        assert_eq!(disclosure.max_chars, 120);
        assert_eq!(disclosure.max_items_per_branch, 3);
        assert_eq!(disclosure.max_description_chars, 40);
        assert_eq!(disclosure.code_width, 2);
        assert!(disclosure.validate().is_ok());
    }

    #[test]
    fn configs_round_trip_through_json_and_tolerate_unknown_fields() {
        let config = ProgressiveSearchConfig::default();
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["selection_mode"], "generate");
        assert_eq!(json["fallback_mode"], "deterministic");
        assert_eq!(json["disclosure"]["max_exposure_depth_per_call"], 99);
        let parsed: ProgressiveSearchConfig = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, config);

        let parsed: DisclosureConfig =
            serde_json::from_str(r#"{"exposure_threshold": 7, "unknown_knob": true}"#).unwrap();
        assert_eq!(parsed.exposure_threshold, 7);
        assert_eq!(
            parsed.max_exposure_depth_per_call,
            DisclosureConfig::default().max_exposure_depth_per_call
        );
    }
}
