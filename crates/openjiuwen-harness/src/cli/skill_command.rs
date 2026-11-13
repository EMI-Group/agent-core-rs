//! `/<skill>` slash-command dispatch for skills exposed to the REPL.
//!
//! Python registers every discovered skill as `/<name>`, and a typed line becomes a
//! prompt that wraps the whole `SKILL.md` so the model reads the instructions
//! before the user's request. This module owns that mapping in three pure parts —
//! [`parse_skill_command`], [`build_skill_query`], [`help_rows`] — plus the two
//! async steps that touch the filesystem ([`load_skills`] and
//! [`render_skill_prompt`]).
//!
//! The wrapper is byte-exact with Python's `_build_skill_query`, because the
//! wrapped text is what gets stored in the session transcript and sent to the
//! model:
//!
//! ```text
//! <skill-instructions>
//! <SKILL.md content>
//! </skill-instructions>
//!
//! User request: <args>
//! ```
//!
//! With no arguments the last line is `Please follow the skill instructions above.`
//! instead (the blank line stays). A missing `SKILL.md` is a configuration error
//! naming the path, where Python instead returns an `Error reading skill file: …`
//! text as the prompt.
//!
//! Built-in commands are resolved by the REPL before this module is consulted, so
//! a skill sharing a built-in name never reaches [`parse_skill_command`].

use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};

use crate::skills::{Skill, SkillLibrary};

/// Longest description shown in a `/help` row before it is elided.
const HELP_DESCRIPTION_LIMIT: usize = 60;

/// Returns the default skill discovery roots: `<workspace>/skills`, then `<home>/skills`.
///
/// The workspace root comes first so project-local skills win a name clash; the
/// home root is where user-installed skills live. Neither directory has to exist:
/// discovery skips missing roots.
#[must_use]
pub fn default_skill_roots(workspace: &Path, home: &OpenJiuwenHome) -> Vec<PathBuf> {
    vec![workspace.join("skills"), home.join("skills")]
}

/// Splits a typed `/<name> <args>` line when `skills` contains that exact name.
///
/// The first whitespace-separated token after `/` is matched case-sensitively
/// against [`Skill::name`]; the rest of the line, trimmed, is the argument string
/// and is empty when the line has no arguments. Returns `None` for lines that do
/// not start with `/` and for names that are not in `skills`. Internal whitespace
/// of the arguments is preserved so the model sees the request as typed.
#[must_use]
pub fn parse_skill_command(line: &str, skills: &[Skill]) -> Option<(String, String)> {
    let rest = line.trim().strip_prefix('/')?;
    let (name, arguments) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    if name.is_empty() || !skills.iter().any(|skill| skill.name == name) {
        return None;
    }
    Some((name.to_string(), arguments.trim().to_string()))
}

/// Wraps a skill body and the typed arguments into the exact Python prompt template.
///
/// The result is `<skill-instructions>`, the skill content, `</skill-instructions>`,
/// a blank line and then either `User request: <arguments>` or, for empty
/// arguments, `Please follow the skill instructions above.`.
#[must_use]
pub fn build_skill_query(skill_content: &str, arguments: &str) -> String {
    let request = if arguments.is_empty() {
        "Please follow the skill instructions above.".to_string()
    } else {
        format!("User request: {arguments}")
    };
    format!("<skill-instructions>\n{skill_content}\n</skill-instructions>\n\n{request}")
}

/// Discovers skills from `roots`.
///
/// An empty root list yields an empty vector, and a missing or unreadable root is
/// skipped by discovery rather than reported as an error.
///
/// # Errors
///
/// Returns whatever [`SkillLibrary::discover`] reports; discovery currently logs
/// and skips IO problems, so this does not fail in practice.
pub async fn load_skills(roots: &[PathBuf]) -> Result<Vec<Skill>> {
    if roots.is_empty() {
        return Ok(Vec::new());
    }
    SkillLibrary::with_roots(roots.iter().cloned())
        .discover()
        .await
}

/// Reads `<skill>/SKILL.md` and wraps it with `arguments`.
///
/// Returns `Ok(None)` when no skill in `skills` is named `name`. Arguments are
/// used as given, so the caller passes the value returned by
/// [`parse_skill_command`].
///
/// # Errors
///
/// Returns [`Error::Config`] naming the `SKILL.md` path when the file cannot be
/// read.
pub async fn render_skill_prompt(
    skills: &[Skill],
    name: &str,
    arguments: &str,
) -> Result<Option<String>> {
    let Some(skill) = skills.iter().find(|skill| skill.name == name) else {
        return Ok(None);
    };
    let path = skill.path();
    let content = tokio::fs::read_to_string(&path).await.map_err(|error| {
        Error::config(format!(
            "cannot read skill file {}: {error}",
            path.display()
        ))
    })?;
    Ok(Some(build_skill_query(&content, arguments)))
}

/// Returns the `/help` rows for the discovered skills, in input order.
///
/// Each row is `  /{name}  {description}`; a description longer than 60 characters
/// is cut to 57 characters and suffixed with `...`, matching Python's skill table.
#[must_use]
pub fn help_rows(skills: &[Skill]) -> Vec<String> {
    skills
        .iter()
        .map(|skill| {
            let mut description = skill.description.clone();
            if description.chars().count() > HELP_DESCRIPTION_LIMIT {
                description = description
                    .chars()
                    .take(HELP_DESCRIPTION_LIMIT - 3)
                    .collect::<String>();
                description.push_str("...");
            }
            format!("  /{}  {}", skill.name, description)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::skills::SKILL_FILENAME;

    fn skill(name: &str, description: &str) -> Skill {
        Skill::new(name, description, format!("/skills/{name}"))
    }

    fn write_skill(root: &Path, name: &str, description: &str, body: &str) {
        let directory = root.join(name);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join(SKILL_FILENAME),
            format!("---\nname: {name}\ndescription: {description}\n---\n{body}"),
        )
        .unwrap();
    }

    #[test]
    fn build_skill_query_is_byte_exact_with_arguments() {
        let query = build_skill_query("# Body\nDo the thing.", "run it");
        assert_eq!(
            query,
            "<skill-instructions>\n# Body\nDo the thing.\n</skill-instructions>\n\nUser request: run it"
        );
    }

    #[test]
    fn build_skill_query_uses_the_followup_line_without_arguments() {
        let query = build_skill_query("# Body\nDo the thing.", "");
        assert_eq!(
            query,
            "<skill-instructions>\n# Body\nDo the thing.\n</skill-instructions>\n\nPlease follow the skill instructions above."
        );
    }

    #[test]
    fn parse_skill_command_accepts_a_known_skill_with_and_without_arguments() {
        let skills = [skill("alpha", "first"), skill("beta", "second")];
        assert_eq!(
            parse_skill_command("/alpha", &skills),
            Some(("alpha".to_string(), String::new()))
        );
        assert_eq!(
            parse_skill_command("/beta do it", &skills),
            Some(("beta".to_string(), "do it".to_string()))
        );
    }

    #[test]
    fn parse_skill_command_normalises_the_whitespace_around_the_name() {
        let skills = [skill("alpha", "first")];
        assert_eq!(
            parse_skill_command("  /alpha \t  do it  ", &skills),
            Some(("alpha".to_string(), "do it".to_string()))
        );
    }

    #[test]
    fn parse_skill_command_rejects_unknown_names_and_non_slash_lines() {
        let skills = [skill("alpha", "first")];
        assert_eq!(parse_skill_command("/gamma do it", &skills), None);
        assert_eq!(parse_skill_command("alpha do it", &skills), None);
        assert_eq!(parse_skill_command("/", &skills), None);
        assert_eq!(parse_skill_command("", &skills), None);
    }

    #[test]
    fn help_rows_format_and_elide_long_descriptions() {
        let short = "a".repeat(HELP_DESCRIPTION_LIMIT);
        let long = "b".repeat(HELP_DESCRIPTION_LIMIT + 1);
        let skills = [
            skill("alpha", "Short description"),
            skill("beta", &short),
            skill("gamma", &long),
        ];

        let rows = help_rows(&skills);
        assert_eq!(rows[0], "  /alpha  Short description");
        assert_eq!(rows[1], format!("  /beta  {short}"));
        let expected = format!("  /gamma  {}...", "b".repeat(HELP_DESCRIPTION_LIMIT - 3));
        assert_eq!(rows[2], expected);
        assert_eq!(rows[2].chars().count(), "  /gamma  ".chars().count() + 60);
    }

    #[tokio::test]
    async fn load_skills_discovers_two_skills_and_tolerates_a_missing_root() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).unwrap();
        write_skill(&root, "alpha", "First skill", "# alpha\n");
        write_skill(&root, "beta", "Second skill", "# beta\n");

        let skills = load_skills(std::slice::from_ref(&root)).await.unwrap();
        let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta"]);
        assert_eq!(skills[0].description, "First skill");

        let empty = load_skills(&[temp.path().join("missing")]).await.unwrap();
        assert_eq!(empty, Vec::<Skill>::new());

        let no_roots = load_skills(&[]).await.unwrap();
        assert_eq!(no_roots, Vec::<Skill>::new());
    }

    #[tokio::test]
    async fn render_skill_prompt_wraps_the_body_and_skips_unknown_skills() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).unwrap();
        write_skill(&root, "alpha", "First skill", "# alpha body\n");
        let skills = load_skills(&[root]).await.unwrap();

        let prompt = render_skill_prompt(&skills, "alpha", "go")
            .await
            .unwrap()
            .unwrap();
        assert!(prompt.contains("# alpha body\n"));
        assert!(prompt.starts_with("<skill-instructions>\n---\n"));
        assert!(prompt.ends_with("</skill-instructions>\n\nUser request: go"));

        assert_eq!(
            render_skill_prompt(&skills, "missing", "go").await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn render_skill_prompt_reports_an_unreadable_file_with_its_path() {
        let temp = tempfile::tempdir().unwrap();
        let skills = [Skill::new(
            "alpha",
            "First skill",
            temp.path().join("alpha"),
        )];
        let error = render_skill_prompt(&skills, "alpha", "go")
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&skills[0].path().display().to_string()));
        assert!(message.contains(SKILL_FILENAME));
    }

    #[test]
    fn default_skill_roots_puts_the_workspace_before_the_home_directory() {
        let home = OpenJiuwenHome::from_root("/home/user/.openjiuwen");
        let roots = default_skill_roots(Path::new("/work/project"), &home);
        assert_eq!(
            roots,
            vec![
                PathBuf::from("/work/project/skills"),
                PathBuf::from("/home/user/.openjiuwen/skills")
            ]
        );
    }
}
