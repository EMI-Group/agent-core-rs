//! Prompt standardization, feedback rewriting, and bad-case repair.
//!
//! Ports `openjiuwen/dev_tools/prompt_builder/`: the deterministic template
//! banks and message assembly plus the three optional model-driven builders.
//! Every builder keeps the Python call sequence (one model call for
//! standardization, two for feedback and bad-case repair) and the bundled
//! `zh-CN`/`en-US` template texts verbatim.
//!
//! # Modules
//!
//! - [`language`] — the [`Language`] selector (`zh-CN` by default).
//! - [`templates`] — the two verbatim template banks and message constructors.
//! - [`utils`] — deterministic helpers: prompt flattening, char-indexed
//!   splicing, bad-case rendering, and model-output parsing.
//! - [`meta`] — [`MetaTemplateBuilder`] and its custom-template registry.
//! - [`feedback`] — [`FeedbackPromptBuilder`] for general/insert/select
//!   feedback modes.
//! - [`bad_case`] — [`BadCasePromptBuilder`] for evaluated-case repair.
//!
//! # Deviations from Python
//!
//! Invalid input is a validation error instead of a logged warning plus a
//! silent fallback; positions are character indices, never byte offsets; and a
//! model answer with no usable text is an [`openjiuwen_common::error::Error::Llm`]
//! instead of `None`.

pub mod bad_case;
pub mod feedback;
pub mod language;
pub mod meta;
pub mod templates;
pub mod utils;

pub use bad_case::{BadCasePromptBuilder, MAX_CASES_LIMIT};
pub use feedback::{FeedbackMode, FeedbackOptions, FeedbackPromptBuilder};
pub use language::Language;
pub use meta::{IntoMetaTemplate, MetaBuildOptions, MetaTemplateBuilder, TemplateType};
pub use templates::{bank, PromptBank};

/// The marker spliced into the prompt by insert-mode feedback.
///
/// Ports Python's `INSERT_STR`; the marker is fixed Chinese text because the
/// optimize template instructs the model to produce content for that exact
/// placeholder, in every language.
pub const INSERT_STR: &str = "[用户要插入的位置]";

/// Maximum number of characters a serialized JSON string may occupy where the
/// Python implementation truncates with a regex (feedback intent parsing, JSON
/// values of bad cases).
pub const JSON_STRING_MAX_LENGTH: usize = 10000;
