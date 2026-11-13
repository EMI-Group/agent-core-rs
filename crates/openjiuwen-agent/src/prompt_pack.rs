//! YAML prompt packs: load an agent's prompts, tools policy, and rails from one
//! YAML document.
//!
//! The pack is an additive configuration format over [`crate::ReActAgent`]:
//! prompts become `prompt_template` messages, sections become system-prompt
//! sections, and rails are constructed by name.
