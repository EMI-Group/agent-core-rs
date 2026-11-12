//! Skill discovery, library state, mounting, and `SKILL.md` validation.

mod mount;

pub use mount::{
    parse_skill_front_matter, split_skill_front_matter, SkillMetadata, SkillResourceValidator,
    SkillRootMount, FRONT_MATTER_PROBE_LINES, SKILL_FILENAME,
};
