//! Skill definitions, validation, templating, and harness-compatible bundles.
//!
//! Ports the artifact-producing part of `openjiuwen/dev_tools/skill_creator`
//! (`skill_omni_creation`): a typed skill definition, `SKILL.md` frontmatter
//! rendering/parsing, text resources, and optional model-assisted drafting.
//!
//! A skill is one directory:
//!
//! ```text
//! <root>/<name>/
//!   SKILL.md              # --- YAML frontmatter --- + markdown body
//!   references/guide.md   # any number of text resources at relative paths
//! ```
//!
//! Three usage styles compose:
//!
//! ```
//! # use openjiuwen_dev_tools::skill_creator::{SkillCreator, SkillSpec};
//! let definition = SkillCreator::new()
//!     .create(&SkillSpec::new("demo", "A demo skill", "Explain {{topic}}.")
//!         .with_variable("topic", "skills"))
//!     .unwrap();
//! assert_eq!(definition.instructions, "Explain skills.");
//! assert!(definition.to_markdown().unwrap().starts_with("---\nname: demo\n"));
//! ```
//!
//! - [`definition`] — [`SkillDefinition`], [`SkillResource`], [`SkillFrontmatter`],
//!   [`parse_markdown`], [`render_markdown`].
//! - [`template`] — strict `{{name}}` rendering.
//! - [`bundle`] — [`SkillBundle`] disk read/write.
//! - [`creator`] — [`SkillSpec`], [`SkillRequest`], [`SkillCreator`].
//! - [`omni`] — the `skill_omni_creation` pipeline: URL → scraped page → assets
//!   → skill bundle ([`OmniCreation`], also reached through
//!   [`SkillCreator::create_from_url`]). The page may be HTML, an unpacked
//!   `tar`/`zip`/`gzip` archive, an office/`PDF` document, or plain text, and
//!   [`omni::GenerationProvenance`] reports whether the drafting model or the
//!   deterministic template renderer produced the document.

pub mod bundle;
pub mod creator;
pub mod definition;
pub mod omni;
pub mod template;

pub use bundle::SkillBundle;
pub use creator::{SkillCreator, SkillRequest, SkillSpec};
pub use definition::{
    parse_markdown, render_markdown, SkillDefinition, SkillFrontmatter, SkillResource,
};
pub use omni::{OmniConfig, OmniCreation, OmniOutcome};
