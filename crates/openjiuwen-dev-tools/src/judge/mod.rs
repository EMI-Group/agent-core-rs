//! Rubric-driven LLM judging with tolerant structured-verdict parsing.
//!
//! This module is the shared scoring foundation for the crate's
//! [`skill_evaluator`](crate::skill_evaluator) and [`tune`](crate::tune)
//! pipelines. It turns "ask a model how good this is" into a typed, repeatable
//! operation:
//!
//! 1. A [`Rubric`] declares the scoring dimensions, their weights, and the raw
//!    scale; [`Rubric::validate`] rejects malformed rubrics before any model
//!    call.
//! 2. [`prompt::rubric_messages`] renders the rubric, the subject, the
//!    instructions, and an optional reference into exactly one system and one
//!    user message, stating the required answer format.
//! 3. [`ModelJudge`] (a [`Judge`]) sends that conversation through
//!    [`openjiuwen_foundation::llm::Model`] and reads the answer with
//!    [`Verdict::parse`], which accepts fenced JSON, tag sections, or a bare JSON
//!    object.
//! 4. [`Verdict`] carries the normalized `0.0..=1.0` score, the pass flag, the
//!    reasoning, the raw dimension scores, and the [`VerdictSource`] that
//!    produced it.
//!
//! # Degrading instead of failing
//!
//! [`ModelJudge::judge_with_fallback`] returns a caller-supplied deterministic
//! verdict — marked [`VerdictSource::Deterministic`] — whenever the rubric is
//! invalid, the provider fails, the answer is unparsable, or the verdict fails
//! validation, logging the reason with [`tracing::warn!`]. Pipelines can thus keep
//! a stable score while still recording that the model path was skipped.
//!
//! # Example
//!
//! ```
//! use openjiuwen_dev_tools::judge::{Rubric, Verdict};
//!
//! let rubric = Rubric::uniform("skill quality", "Is the skill usable?", ["clarity", "depth"]);
//! let verdict = Verdict::parse(
//!     "<score>0.9</score><dimension name=\"clarity\">0.9</dimension>\
//!      <dimension name=\"depth\">0.8</dimension>",
//!     &rubric,
//! )
//! .expect("the tag form parses");
//! assert!(verdict.passed);
//! assert_eq!(verdict.source, openjiuwen_dev_tools::judge::VerdictSource::Model);
//! ```

pub mod model_judge;
pub mod prompt;
pub mod rubric;
pub mod verdict;

pub use model_judge::{Judge, JudgeRequest, ModelJudge};
pub use rubric::{Rubric, RubricDimension, DEFAULT_PASS_THRESHOLD};
pub use verdict::{Verdict, VerdictSource};
