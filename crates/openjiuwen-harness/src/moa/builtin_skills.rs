//! Embedded `consult-reference-models` skill descriptor and its materialization helper.
//!
//! Ports `openjiuwen/harness/moa/builtin_skills.py`. Python ships the consultation skill as a
//! packaged directory next to the module and resolves it with `get_builtin_skill_dir`; the
//! Rust port embeds the `SKILL.md` body at compile time with [`include_str!`] and materializes
//! it on request, so the crate keeps working from a single binary with no resource directory
//! to install. The name guard and its error message mirror Python exactly.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use crate::skills::Skill;

/// Name of the builtin consultation skill, which is also its leaf directory name.
pub const CONSULT_SKILL_NAME: &str = "consult-reference-models";

/// Description published in the embedded `SKILL.md` front matter.
///
/// The value is exactly the `description:` field of [`CONSULT_SKILL_BODY`], so a materialized
/// skill and a discovered one report the same description.
pub const CONSULT_SKILL_DESCRIPTION: &str = "Seek independent candidate proposals for early approach selection, costly commitments, conflicting evidence, or stalled reasoning. Use when consult_reference_models is available and another perspective could materially change the next decision.";

/// File name of the skill descriptor inside a builtin skill directory.
pub const CONSULT_SKILL_FILE: &str = "SKILL.md";

/// Embedded `SKILL.md` of the consultation skill, front matter included.
pub const CONSULT_SKILL_BODY: &str = include_str!("skills/consult-reference-models/SKILL.md");

/// Checks that `name` can be used as a single builtin skill directory name.
///
/// The rule is Python's `get_builtin_skill_dir` guard: the name must be non-empty and must
/// not be `.`, `..`, or contain a path separator, so a lookup can never escape the builtin
/// skill library.
///
/// # Errors
///
/// Returns [`Error::validation`] with the Python message
/// `A builtin skill name must be a single directory name` for an empty name, `.`, `..`, or a
/// name containing `/` or `\`.
pub fn validate_builtin_skill_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(Error::validation(
            "A builtin skill name must be a single directory name",
        ));
    }
    Ok(())
}

/// Returns the embedded `SKILL.md` body of the builtin skill named `name`.
///
/// # Errors
///
/// Returns [`Error::validation`] when [`validate_builtin_skill_name`] rejects the name, and
/// [`Error::not_found`] with `unknown builtin skill: {name}` when `name` passes the guard but
/// describes no embedded skill.
pub fn builtin_skill_content(name: &str) -> Result<&'static str> {
    validate_builtin_skill_name(name)?;
    if name == CONSULT_SKILL_NAME {
        Ok(CONSULT_SKILL_BODY)
    } else {
        Err(Error::not_found(format!("unknown builtin skill: {name}")))
    }
}

/// Builds the consultation [`Skill`] descriptor for `directory`.
///
/// The directory is used verbatim; callers decide whether it is the packaged location, a
/// materialized copy, or a host-provided skill root.
#[must_use]
pub fn consult_skill(directory: impl Into<PathBuf>) -> Skill {
    Skill::new(CONSULT_SKILL_NAME, CONSULT_SKILL_DESCRIPTION, directory)
}

/// Materializes the embedded consultation skill under `parent` and returns its descriptor.
///
/// Creates `<parent>/consult-reference-models/SKILL.md` and writes [`CONSULT_SKILL_BODY`] to
/// it, so the skill becomes discoverable by a [`crate::skills::SkillLibrary`] root pointing at
/// `parent`. Python ships a packaged directory; this port embeds the body and writes it on
/// request, which is why the function is asynchronous and idempotent: an existing directory
/// and file are reused and rewritten with identical content.
///
/// # Errors
///
/// Returns [`Error::sys_op`] when the skill directory or its `SKILL.md` cannot be created or
/// written.
pub async fn materialize_builtin_skill(parent: &Path) -> Result<Skill> {
    let directory = parent.join(CONSULT_SKILL_NAME);
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|error| {
            Error::sys_op(format!(
                "failed to create builtin skill directory {}: {error}",
                directory.display()
            ))
        })?;
    let path = directory.join(CONSULT_SKILL_FILE);
    tokio::fs::write(&path, CONSULT_SKILL_BODY)
        .await
        .map_err(|error| {
            Error::sys_op(format!(
                "failed to write builtin skill {}: {error}",
                path.display()
            ))
        })?;
    Ok(consult_skill(directory))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        builtin_skill_content, consult_skill, materialize_builtin_skill,
        validate_builtin_skill_name, CONSULT_SKILL_BODY, CONSULT_SKILL_DESCRIPTION,
        CONSULT_SKILL_FILE, CONSULT_SKILL_NAME,
    };
    use crate::skills::parse_skill_front_matter;
    use openjiuwen_common::error::{Error, ErrorKind};

    const NAME_ERROR: &str = "validation: A builtin skill name must be a single directory name";

    #[test]
    fn name_validation_rejects_empty_dots_and_separators() {
        for name in ["", ".", "..", "a/b", "../escape", "a\\b", "/absolute"] {
            let error = validate_builtin_skill_name(name).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "name: {name:?}");
            assert_eq!(error.to_string(), NAME_ERROR, "name: {name:?}");
        }
        assert!(validate_builtin_skill_name(CONSULT_SKILL_NAME).is_ok());
        assert!(validate_builtin_skill_name("other-skill").is_ok());
    }

    #[test]
    fn builtin_content_exposes_the_embedded_skill() {
        let body = builtin_skill_content(CONSULT_SKILL_NAME).unwrap();
        assert_eq!(body, CONSULT_SKILL_BODY);
        let (metadata, markdown) = parse_skill_front_matter(body).unwrap().unwrap();
        assert_eq!(metadata["name"], CONSULT_SKILL_NAME);
        assert_eq!(metadata["description"], CONSULT_SKILL_DESCRIPTION);
        assert_ne!(CONSULT_SKILL_DESCRIPTION, "");
        assert_ne!(markdown, "");
    }

    #[test]
    fn builtin_content_rejects_unknown_and_malformed_names() {
        let error = builtin_skill_content("missing-skill").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: unknown builtin skill: missing-skill"
        );

        let error = builtin_skill_content("../escape").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), NAME_ERROR);
    }

    #[test]
    fn consult_skill_uses_the_embedded_descriptor() {
        let skill = consult_skill("/tmp/skills/consult-reference-models");
        assert_eq!(skill.name, CONSULT_SKILL_NAME);
        assert_eq!(skill.description, CONSULT_SKILL_DESCRIPTION);
        assert_eq!(
            skill.path(),
            std::path::Path::new("/tmp/skills/consult-reference-models").join(CONSULT_SKILL_FILE)
        );
    }

    #[tokio::test]
    async fn materialize_writes_the_skill_and_is_repeatable() {
        let temp = tempfile::tempdir().unwrap();
        let skill = materialize_builtin_skill(temp.path()).await.unwrap();
        let directory = temp.path().join(CONSULT_SKILL_NAME);
        assert_eq!(skill.name, CONSULT_SKILL_NAME);
        assert_eq!(skill.description, CONSULT_SKILL_DESCRIPTION);
        assert_eq!(skill.directory, directory);
        assert_eq!(skill.path(), directory.join(CONSULT_SKILL_FILE));
        assert_eq!(
            std::fs::read_to_string(skill.path()).unwrap(),
            CONSULT_SKILL_BODY
        );

        let again = materialize_builtin_skill(temp.path()).await.unwrap();
        assert_eq!(again, skill);
    }

    #[tokio::test]
    async fn materialize_reports_a_non_directory_parent() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("not-a-directory");
        std::fs::write(&file, b"x").unwrap();

        let error = materialize_builtin_skill(&file).await.unwrap_err();
        assert!(
            matches!(error, Error::SysOp(_)),
            "unexpected error: {error}"
        );
    }
}
