//! Agent builder, prompt builder, skill creator/evaluator, and tuning helpers.
//!
//! Maps the Python package `openjiuwen.dev_tools` onto idiomatic Rust. The crate
//! is a layer-5 consumer of the kernel: prompts and models come from
//! [`openjiuwen_foundation`], agent cards and configurations from
//! [`openjiuwen_agent`], and every fallible operation returns
//! [`openjiuwen_common::error::Result`].
//!
//! # Modules
//!
//! - [`prompt_builder`] — standardize, rewrite, and repair prompts with bundled
//!   `zh-CN`/`en-US` template banks, optionally driven by a [`Model`].
//! - [`skill_creator`] — build, validate, template, and write harness-compatible
//!   skill bundles (`SKILL.md` plus resources), including the `omni` pipeline
//!   that turns a fetched page or tar archive into a skill via `create_from_url`.
//! - [`agent_builder`] — programmatic and serialized agent specifications, the
//!   platform DSL, the bridge into [`openjiuwen_agent`], and the staged
//!   clarify → design → generate → reflect pipeline with Mermaid rendering.
//! - [`skill_evaluator`] — deterministic and judge-driven evaluation of skills
//!   against test cases, producing design/functional/safety reports and a final
//!   score, plus the `EvalHarness` with JSON/JSONL run artifacts.
//! - [`tune`] — the `Case`/`EvaluatedCase`/`CaseLoader` models, the evaluator
//!   seam, four deterministic prompt searches, the `Trainer`, and
//!   `optimize_prompt`, each with an optional [`Model`] rewrite path.
//! - [`judge`] — rubric-driven LLM judging over the [`Model`] seam with tolerant
//!   structured-verdict parsing, shared by `skill_evaluator` and `tune`.
//!
//! [`Model`]: openjiuwen_foundation::llm::Model
//!
//! # Design notes
//!
//! Deterministic transforms (template rendering, frontmatter parsing, DSL
//! conversion, scoring) are ordinary public functions; a [`Model`] only supplies
//! text. Nothing in this crate reads the environment or blocks on a runtime, so
//! every entry point composes inside an existing tokio runtime.
//!
//! [`Model`]: openjiuwen_foundation::llm::Model
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::unused_async,
    clippy::significant_drop_tightening,
    clippy::manual_let_else,
    clippy::needless_pass_by_value,
    clippy::missing_const_for_fn
)]
pub mod agent_builder;
pub mod judge;
pub mod prompt_builder;
pub mod skill_creator;
pub mod skill_evaluator;
pub mod tune;
pub mod util;
/// The crate name, useful in diagnostics and documentation.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// The crate version.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
