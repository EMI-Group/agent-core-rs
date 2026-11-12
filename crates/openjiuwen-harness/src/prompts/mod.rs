//! System prompt builder, prompt sections, tool descriptions, prompt report, and sanitization.
//!
//! Ports `openjiuwen/harness/prompts/` and the section half of
//! `openjiuwen/core/single_agent/prompts/builder.py`. The module assembles the
//! `DeepAgent` system prompt from ordered bilingual [`PromptSection`]s, owns the
//! model-facing tool description table, reports the size/token cost of the result, and
//! sanitizes values interpolated into prompts.
//!
//! Everything is pure data assembly: no IO, no async, no environment or filesystem
//! access, and no dependency on [`crate::tools`] — tool cards and todo items arrive as
//! values. A typical caller adds one section per harness capability and renders once:
//!
//! ```
//! use openjiuwen_harness::prompts::{identity, safety, HarnessPromptBuilder, PromptMode};
//!
//! let mut builder = HarnessPromptBuilder::new("en", PromptMode::Full);
//! builder.add_section(identity()).expect("first identity section");
//! builder.add_section(safety()).expect("first safety section");
//! let prompt = builder.build();
//! assert!(prompt.starts_with("You are a general-purpose AI assistant."));
//! assert!(prompt.contains("# Safety"));
//! ```
//!
//! Module layout:
//!
//! - `builder` — [`PromptSection`], [`PromptCategory`], [`SectionName`], [`HarnessPromptBuilder`]
//! - `sections` — one factory per standard section, all bilingual
//! - `tool_descriptions` — the bilingual tool description table and [`build_tool_card`]
//! - `report` — [`PromptReport`] size/token accounting
//! - `sanitize` — [`sanitize_path`] and [`sanitize_user_content`]

mod builder;
mod report;
mod sanitize;
mod sections;
mod tool_descriptions;

pub use builder::{
    resolve_language, HarnessPromptBuilder, PromptCategory, PromptMode, PromptSection, SectionName,
    DEFAULT_LANGUAGE, MINIMAL_SECTIONS, SUPPORTED_LANGUAGES,
};
pub use report::{CategoryInfo, PromptReport, SectionInfo};
pub use sanitize::{
    sanitize_path, sanitize_user_content, sanitize_user_content_with_limit,
    DEFAULT_MAX_USER_CONTENT_LEN,
};
pub use sections::{
    completion_signal, completion_signal_with, context_section, environment_section, identity,
    memory_section, mode_instructions, promise_tag, safety, skills_section, subagent_section,
    task_loop_section, todo_section, tools_section, workspace_section, DEFAULT_COMPLETION_PROMISE,
};
pub use tool_descriptions::{
    build_tool_card, description, tool_names, validate_all, ToolDescription, TOOL_DESCRIPTIONS,
};
