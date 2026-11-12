//! Configuration for the `PUCT` program search.
//!
//! Python parity: the fields mirror the scorecard / card values that
//! `openjiuwen/rsi/artifact_rsi/program_opt/puct_provider.py` (v0.1.20) and
//! `engine.py` read when they build a run — `iterations`, `workers`, the
//! `options.c_puct` / `options.prior_exponent` knobs
//! (`puct_engine.py::_prior_exponent`), `candidate_timeout_seconds`,
//! `entrypoint` and `reply_format` (`reply_format.py::format_for`).
//!
//! Intentional deviations from Python, all in the direction of a typed record:
//!
//! - [`MAX_WORKERS`] and [`DEFAULT_WORKERS`] are named here as constants,
//!   matching the module-level names in `puct_provider.py` (lines 88 and 92)
//!   instead of being inlined into the worker clamp.
//! - [`ReplyFormat`] is the *name* of an output protocol, not the Python
//!   `ReplyFormat` dataclass. That dataclass carries prompt text, a renderer and
//!   a parser (functions), which is exactly the deferred model/prompt plumbing;
//!   the portable half is "which of `files` / `tagged` was chosen".
//! - Values are validated by [`ProgramOptConfig::validate`] rather than at the
//!   point of use, and none of the fields carry process-global defaults read
//!   from the environment.
//!
//! ```
//! use openjiuwen_rsi::artifact_rsi::program_opt::{ProgramOptConfig, ReplyFormat};
//!
//! let config = ProgramOptConfig::default();
//! assert_eq!(config.workers, 1);
//! assert_eq!(config.reply_format, ReplyFormat::Files);
//! assert!(config.validate().is_ok());
//! ```
use crate::artifact_rsi::program_opt::tree::PuctTree;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
/// Upper bound on concurrent candidate evaluations.
///
/// Python parity: `MAX_WORKERS` in `puct_provider.py`, used to clamp the card's
/// `workers` value (`_workers_from`, `max(1, min(workers, MAX_WORKERS))`).
pub const MAX_WORKERS: usize = 8;
/// Default number of concurrent candidate evaluations.
///
/// Python parity: `DEFAULT_WORKERS` in `puct_provider.py`. Serial is the default
/// because concurrent mutation calls measured slower than sequential ones on a
/// real provider.
pub const DEFAULT_WORKERS: usize = 1;
/// The output protocol a candidate is asked to reply in.
///
/// Python parity: the names of the two protocols shipped by
/// `reply_format.py::_FORMATS`. `Files` is the default there too
/// (`DEFAULT_FORMAT = "files"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplyFormat {
    /// Whole files, one fenced block each, labelled with their path.
    #[default]
    Files,
    /// One program between `<PROGRAM>` tags, for single-file genomes.
    Tagged,
}
/// Everything the program search needs to know before it touches a program.
///
/// Serialisable and tolerant of unknown fields, so a scorecard can grow without
/// breaking older readers. [`ProgramOptConfig::default`] carries the Python
/// defaults; [`ProgramOptConfig::validate`] is the single place that rejects a
/// nonsensical combination, and [`ProgramOptConfig::to_tree`] turns the search
/// knobs into an empty [`PuctTree`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProgramOptConfig {
    /// Exploration constant of the `PUCT` formula (`options.c_puct`).
    pub c_puct: f64,
    /// Exponent applied to a node's model-given promise before it is normalised
    /// into the prior (`options.prior_exponent`).
    ///
    /// `0.0` — the upstream value — ignores promises entirely and gives every
    /// node the uniform prior `1/N`.
    pub prior_exponent: f64,
    /// Maximum number of nodes the tree may hold; `None` means unbounded.
    pub candidate_limit: Option<usize>,
    /// Maximum number of expansions one run may propose.
    pub max_expansions: usize,
    /// Maximum depth of the candidate tree.
    pub max_depth: usize,
    /// Concurrent candidate evaluations, clamped by callers to
    /// `1..=MAX_WORKERS`.
    pub workers: usize,
    /// File in the seed program that the evaluator imports.
    pub entrypoint: String,
    /// Wall-clock ceiling for one candidate execution, in seconds.
    pub candidate_timeout_secs: f64,
    /// Reply protocol the mutation prompt asks for and the parser reads.
    pub reply_format: ReplyFormat,
}
impl Default for ProgramOptConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.0,
            prior_exponent: 0.0,
            candidate_limit: None,
            max_expansions: 16,
            max_depth: 8,
            workers: DEFAULT_WORKERS,
            entrypoint: "candidate.py".to_owned(),
            candidate_timeout_secs: 60.0,
            reply_format: ReplyFormat::Files,
        }
    }
}
impl ProgramOptConfig {
    /// Rejects values the search cannot run with.
    ///
    /// The accepted ranges are the ones the Python engine silently assumes:
    /// a positive exploration constant, a non-negative prior exponent, at least
    /// one expansion, a non-empty tree depth, a positive per-candidate timeout,
    /// a named entrypoint, a worker count inside `1..=MAX_WORKERS`, and a
    /// `candidate_limit` that still allows at least the root node.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for the first violated constraint, with a
    /// message naming the field and the rejected value. Checks run in field
    /// order, so a document with several faults reports the first one.
    pub fn validate(&self) -> Result<()> {
        if self.workers == 0 || self.workers > MAX_WORKERS {
            return Err(Error::validation(format!(
                "program_opt: workers must be between 1 and {MAX_WORKERS}, got {}",
                self.workers
            )));
        }
        if !self.c_puct.is_finite() || self.c_puct < 0.0 {
            return Err(Error::validation(format!(
                "program_opt: c_puct must be a finite non-negative number, got {}",
                self.c_puct
            )));
        }
        if !self.prior_exponent.is_finite() || self.prior_exponent < 0.0 {
            return Err(Error::validation(format!(
                "program_opt: prior_exponent must be a finite non-negative number, got {}",
                self.prior_exponent
            )));
        }
        if self.max_expansions == 0 {
            return Err(Error::validation(
                "program_opt: max_expansions must be at least 1",
            ));
        }
        if self.max_depth == 0 {
            return Err(Error::validation(
                "program_opt: max_depth must be at least 1",
            ));
        }
        if !self.candidate_timeout_secs.is_finite() || self.candidate_timeout_secs <= 0.0 {
            return Err(Error::validation(format!(
                "program_opt: candidate_timeout_secs must be a finite positive number, got {}",
                self.candidate_timeout_secs
            )));
        }
        if self.entrypoint.trim().is_empty() {
            return Err(Error::validation(
                "program_opt: entrypoint must not be empty",
            ));
        }
        if self.candidate_limit == Some(0) {
            return Err(Error::validation(
                "program_opt: candidate_limit must be at least 1 when set",
            ));
        }
        Ok(())
    }
    /// Builds an empty [`PuctTree`] from the search knobs of this configuration.
    ///
    /// Validation runs first, so a tree is never built from a value
    /// [`ProgramOptConfig::validate`] would reject. Only `c_puct`,
    /// `prior_exponent` and `candidate_limit` reach the tree; budgets, workers,
    /// the entrypoint and the reply format stay with the caller that owns the
    /// run.
    ///
    /// # Errors
    ///
    /// Propagates the [`Error::Validation`] of
    /// [`ProgramOptConfig::validate`].
    pub fn to_tree(&self) -> Result<PuctTree> {
        self.validate()?;
        Ok(PuctTree::new(self))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    #[test]
    fn defaults_match_python() {
        let config = ProgramOptConfig::default();
        assert!((config.c_puct - 1.0).abs() < f64::EPSILON);
        assert!(config.prior_exponent.abs() < f64::EPSILON);
        assert_eq!(config.candidate_limit, None);
        assert_eq!(config.max_expansions, 16);
        assert_eq!(config.max_depth, 8);
        assert_eq!(config.workers, DEFAULT_WORKERS);
        assert_eq!(config.entrypoint, "candidate.py");
        assert!((config.candidate_timeout_secs - 60.0).abs() < f64::EPSILON);
        assert_eq!(config.reply_format, ReplyFormat::Files);
        assert!(config.validate().is_ok());
    }
    #[test]
    fn reply_format_serde_names() {
        assert_eq!(
            serde_json::to_value(ReplyFormat::Files).expect("serialize"),
            json!("files")
        );
        assert_eq!(
            serde_json::to_value(ReplyFormat::Tagged).expect("serialize"),
            json!("tagged")
        );
        let parsed: ReplyFormat = serde_json::from_value(json!("tagged")).expect("deserialize");
        assert_eq!(parsed, ReplyFormat::Tagged);
        assert_eq!(ReplyFormat::default(), ReplyFormat::Files);
    }
    #[test]
    fn json_round_trip_ignores_unknown_fields() {
        let document = json!({
            "c_puct": 2.5,
            "prior_exponent": 2.0,
            "max_expansions": 4,
            "workers": 3,
            "reply_format": "tagged",
            "unknown_field": {"nested": true},
        });
        let config: ProgramOptConfig =
            serde_json::from_value(document).expect("unknown fields are tolerated");
        assert!((config.c_puct - 2.5).abs() < f64::EPSILON);
        assert!((config.prior_exponent - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.max_expansions, 4);
        assert_eq!(config.workers, 3);
        assert_eq!(config.reply_format, ReplyFormat::Tagged);
        // Omitted keys keep their Python defaults.
        assert_eq!(config.max_depth, 8);
        assert_eq!(config.entrypoint, "candidate.py");
        assert_eq!(config.candidate_limit, None);
        let round_tripped: ProgramOptConfig =
            serde_json::from_value(serde_json::to_value(&config).expect("serialize"))
                .expect("deserialize");
        assert_eq!(round_tripped, config);
    }
    #[test]
    fn empty_document_is_the_default() {
        let config: ProgramOptConfig = serde_json::from_value(json!({})).expect("deserialize");
        assert_eq!(config, ProgramOptConfig::default());
    }
    #[test]
    fn validate_rejects_bad_numeric_fields() {
        let cases: Vec<(ProgramOptConfig, &str)> = vec![
            (
                ProgramOptConfig {
                    workers: 0,
                    ..ProgramOptConfig::default()
                },
                "workers",
            ),
            (
                ProgramOptConfig {
                    workers: MAX_WORKERS + 1,
                    ..ProgramOptConfig::default()
                },
                "workers",
            ),
            (
                ProgramOptConfig {
                    c_puct: -0.1,
                    ..ProgramOptConfig::default()
                },
                "c_puct",
            ),
            (
                ProgramOptConfig {
                    c_puct: f64::NAN,
                    ..ProgramOptConfig::default()
                },
                "c_puct",
            ),
            (
                ProgramOptConfig {
                    c_puct: f64::INFINITY,
                    ..ProgramOptConfig::default()
                },
                "c_puct",
            ),
            (
                ProgramOptConfig {
                    prior_exponent: -1.0,
                    ..ProgramOptConfig::default()
                },
                "prior_exponent",
            ),
            (
                ProgramOptConfig {
                    prior_exponent: f64::NAN,
                    ..ProgramOptConfig::default()
                },
                "prior_exponent",
            ),
            (
                ProgramOptConfig {
                    max_expansions: 0,
                    ..ProgramOptConfig::default()
                },
                "max_expansions",
            ),
            (
                ProgramOptConfig {
                    max_depth: 0,
                    ..ProgramOptConfig::default()
                },
                "max_depth",
            ),
            (
                ProgramOptConfig {
                    candidate_timeout_secs: 0.0,
                    ..ProgramOptConfig::default()
                },
                "candidate_timeout_secs",
            ),
            (
                ProgramOptConfig {
                    candidate_timeout_secs: f64::INFINITY,
                    ..ProgramOptConfig::default()
                },
                "candidate_timeout_secs",
            ),
        ];
        assert_rejected(cases);
    }

    #[test]
    fn validate_rejects_bad_shape_fields() {
        let cases: Vec<(ProgramOptConfig, &str)> = vec![
            (
                ProgramOptConfig {
                    entrypoint: "   ".to_owned(),
                    ..ProgramOptConfig::default()
                },
                "entrypoint",
            ),
            (
                ProgramOptConfig {
                    candidate_limit: Some(0),
                    ..ProgramOptConfig::default()
                },
                "candidate_limit",
            ),
        ];
        assert_rejected(cases);
    }

    fn assert_rejected(cases: Vec<(ProgramOptConfig, &str)>) {
        for (config, field) in cases {
            let error = config.validate().expect_err("should be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation, "{field}");
            assert!(
                error.to_string().contains(field),
                "message {error:?} should name {field}"
            );
        }
    }
    #[test]
    fn boundary_values_validate() {
        let config = ProgramOptConfig {
            workers: MAX_WORKERS,
            c_puct: 0.0,
            prior_exponent: 0.0,
            candidate_limit: Some(1),
            candidate_timeout_secs: f64::MIN_POSITIVE,
            ..ProgramOptConfig::default()
        };
        assert!(config.validate().is_ok());
    }
    #[test]
    fn to_tree_copies_search_knobs_and_validates() {
        let config = ProgramOptConfig {
            c_puct: 1.5,
            prior_exponent: 2.0,
            candidate_limit: Some(5),
            ..ProgramOptConfig::default()
        };
        let tree = config.to_tree().expect("valid config");
        assert!((tree.c_puct - 1.5).abs() < f64::EPSILON);
        assert!((tree.prior_exponent - 2.0).abs() < f64::EPSILON);
        assert_eq!(tree.candidate_limit, Some(5));
        assert!(tree.is_empty());
        assert_eq!(tree.next_iteration(), 1);
        let invalid = ProgramOptConfig {
            max_depth: 0,
            ..ProgramOptConfig::default()
        };
        let error = invalid.to_tree().expect_err("invalid config");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
