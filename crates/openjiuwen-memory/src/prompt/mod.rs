//! Recall results rendered as prompt text or `BaseMessage`s.
//!
//! The module is deliberately free of `openjiuwen-context` (and session) imports:
//! prompt injection is expressed as a string for a template placeholder and as
//! [`BaseMessage`](openjiuwen_foundation::prompt::BaseMessage) values the caller
//! appends itself, so memory never manipulates a context window.
//!
//! ```
//! use openjiuwen_memory::{render_snippets, hits_to_system_message};
//! use openjiuwen_memory::{MemoryRecord, MemoryScope, MemoryType, RecallHit};
//!
//! let record = MemoryRecord::new(
//!     MemoryScope::new("alice", "ws"),
//!     "Alice prefers dark mode",
//!     MemoryType::SemanticMemory,
//! );
//! let hits = [RecallHit {
//!     record,
//!     score: 0.9,
//!     similarity: 0.9,
//!     keyword: 0.0,
//!     importance: 0.5,
//!     recency: 0.5,
//! }];
//! let body = render_snippets(&hits, 512);
//! assert!(body.starts_with("## Long-term memory"));
//! let message = hits_to_system_message(&hits, 512);
//! assert!(message.is_some());
//! ```
pub mod messages;
pub mod render;

pub use messages::*;
pub use render::*;

/// Heading of the rendered memory section (also the Markdown header text).
pub const MEMORY_SECTION_TITLE: &str = "Long-term memory";
/// Tag some Python prompts place in front of the memory section.
pub const MEMORY_SECTION_TAG: &str = "[MEMORY]";
/// Prompt-template placeholder that receives [`render_snippets`] output.
pub const MEMORY_PLACEHOLDER: &str = "sys_long_term_memory";
/// Prompt-template placeholder that receives rendered memory variables.
pub const VARIABLES_PLACEHOLDER: &str = "sys_memory_variables";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_the_python_prompt_contract() {
        assert_eq!(MEMORY_SECTION_TITLE, "Long-term memory");
        assert_eq!(MEMORY_SECTION_TAG, "[MEMORY]");
        assert_eq!(MEMORY_PLACEHOLDER, "sys_long_term_memory");
        assert_eq!(VARIABLES_PLACEHOLDER, "sys_memory_variables");
    }

    #[test]
    fn re_exports_expose_both_renderers_and_message_helpers() {
        use crate::record::{MemoryRecord, MemoryScope, MemoryType};
        let record = MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "Alice prefers dark mode",
            MemoryType::SemanticMemory,
        )
        .with_id("id-1");
        let rendered = render_records(std::slice::from_ref(&record), 100);
        assert_eq!(
            rendered,
            "## Long-term memory\n- [semantic_memory] Alice prefers dark mode"
        );
        assert_eq!(hits_to_messages(&[], 100).len(), 0);
        let variables = std::collections::BTreeMap::new();
        assert!(variables_to_system_message(&variables).is_none());
        assert_eq!(to_memory_json(&[]).unwrap(), "[]");
    }
}
