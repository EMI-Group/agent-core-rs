//! Skill utilities: skill definitions, parsing, discovery, and prompt injection.
//!
//! Ports `openjiuwen/core/single_agent/skills/` (`Skill`, `SkillManager`,
//! `SkillUtil`) as a Markdown-front-matter skill library the agent can discover
//! and expose to the model through a prompt section.
//!
//! - [`Skill`] is one skill: front-matter metadata plus the Markdown body the
//!   model reads through `read_file`.
//! - [`SkillManager`] discovers skills from a directory or a single document and
//!   keeps them in a name-sorted registry.
//! - [`SkillUtil`] renders the catalog and the budgeted `## Skills` prompt block
//!   and selects the skills relevant to a query.
//! - [`SkillRail`] inserts that block at the front of the model window once per
//!   invocation, marked with [`SKILL_PROMPT_METADATA_KEY`].
//!
//! Python's remote skill download (`RemoteSkillUtil`, GitHub) is not ported.
pub mod manager;
pub mod rail;
pub mod skill;
pub mod util;

pub use manager::SkillManager;
pub use rail::{SkillRail, DEFAULT_SKILL_PROMPT_BUDGET, SKILL_PROMPT_METADATA_KEY};
pub use skill::{Skill, SkillFrontMatter};
pub use util::{SkillUtil, SKILLS_HEADING, TRUNCATION_MARKER};
