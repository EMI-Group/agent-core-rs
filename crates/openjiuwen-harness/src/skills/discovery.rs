//! Recursive `SKILL.md` discovery under skill library roots.
//!
//! Ports the discovery half of `openjiuwen/harness/rails/skills/skill_use_rail.py`: a
//! library is not necessarily flat, so the walk descends through grouping directories but
//! stops at the first `SKILL.md` it finds on a branch. A directory holding a `SKILL.md`
//! is a skill and what it keeps inside is its own business, not a top-level entry.
//!
//! Root order is preserved, traversal within a root is depth-first pre-order with
//! siblings sorted by file name, and visited directories are deduplicated by canonical
//! path so symlinked libraries (a normal way to share a skill across teams) cannot make
//! the walk loop forever.
//!
//! Discovery never fails: unreadable roots and directories are logged and skipped, so a
//! broken branch cannot hide the rest of a library. The [`Result`] in the signature is
//! kept for API consistency with the other skill operations.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::Result;
use serde_json::Value;

use super::mount::{parse_skill_front_matter, read_front_matter_text, SKILL_FILENAME};
use super::Skill;

/// Directory names the skill scan never descends into, at every depth.
const SKILL_SCAN_SKIP_DIRS: [&str; 4] = ["output", "temp", "assets", "node_modules"];

/// Finds every skill directory under `roots`, ordered by root and then by name.
///
/// `language` selects a localized `description_<language>` front-matter key before the
/// plain `description`; the key is trimmed and lowercased. Skills are deduplicated by
/// bare directory name, first-wins across the whole walk.
pub(crate) async fn discover_skills(
    roots: &[PathBuf],
    language: Option<&str>,
) -> Result<Vec<Skill>> {
    let mut skills: Vec<Skill> = Vec::new();
    let mut seen_names: HashMap<String, PathBuf> = HashMap::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();

    for root in roots {
        match tokio::fs::metadata(root).await {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => {
                tracing::debug!(
                    root = %root.display(),
                    "skill library root is not a directory, skipping"
                );
                continue;
            }
            Err(error) => {
                tracing::debug!(
                    root = %root.display(),
                    error = %error,
                    "skill library root is not usable, skipping"
                );
                continue;
            }
        }

        let mut stack = vec![root.clone()];
        while let Some(directory) = stack.pop() {
            let key = tokio::fs::canonicalize(&directory)
                .await
                .unwrap_or_else(|_| directory.clone());
            if !visited.insert(key) {
                continue;
            }

            let skill_path = directory.join(SKILL_FILENAME);
            if tokio::fs::metadata(&skill_path)
                .await
                .is_ok_and(|metadata| metadata.is_file())
            {
                record_skill(&directory, language, &mut skills, &mut seen_names).await;
                continue;
            }

            let mut children = Vec::new();
            let mut entries = match tokio::fs::read_dir(&directory).await {
                Ok(entries) => entries,
                Err(error) => {
                    tracing::debug!(
                        directory = %directory.display(),
                        error = %error,
                        "cannot list skill directory, skipping"
                    );
                    continue;
                }
            };
            loop {
                let entry = match entries.next_entry().await {
                    Ok(Some(entry)) => entry,
                    Ok(None) => break,
                    Err(error) => {
                        tracing::debug!(
                            directory = %directory.display(),
                            error = %error,
                            "cannot read a skill directory entry, skipping the rest of it"
                        );
                        break;
                    }
                };
                let file_name = entry.file_name();
                let Some(name) = file_name.to_str() else {
                    continue;
                };
                if name.starts_with('.') || SKILL_SCAN_SKIP_DIRS.contains(&name) {
                    continue;
                }
                let path = entry.path();
                if tokio::fs::metadata(&path)
                    .await
                    .is_ok_and(|metadata| metadata.is_dir())
                {
                    children.push((name.to_string(), path));
                }
            }
            children.sort_by(|left, right| left.0.cmp(&right.0));
            for (_, path) in children.into_iter().rev() {
                stack.push(path);
            }
        }
    }

    Ok(skills)
}

/// Records one skill directory, skipping and warning about a duplicate name.
async fn record_skill(
    directory: &Path,
    language: Option<&str>,
    skills: &mut Vec<Skill>,
    seen_names: &mut HashMap<String, PathBuf>,
) {
    let Some(name) = directory
        .file_name()
        .and_then(OsStr::to_str)
        .map(str::to_string)
    else {
        tracing::warn!(
            directory = %directory.display(),
            "skill directory name is not valid UTF-8, skipping"
        );
        return;
    };
    if let Some(previous) = seen_names.get(&name) {
        tracing::warn!(
            skill = %name,
            keep = %previous.display(),
            skip = %directory.display(),
            "duplicate skill name detected, skipping"
        );
        return;
    }
    let description = load_description(directory, language)
        .await
        .unwrap_or_else(|| format!("Skill located in {}", directory.display()));
    seen_names.insert(name.clone(), directory.to_path_buf());
    skills.push(Skill {
        name,
        description,
        directory: directory.to_path_buf(),
    });
}

/// Loads a skill's description from its `SKILL.md` front matter.
///
/// When `language` is given, `description_<language>` (with the language trimmed and
/// lowercased) wins over `description`. Only string values count. Missing front matter,
/// a missing key, a whitespace-only value, and IO or parse failures all yield `None` so
/// the caller can fall back to a generated description.
pub(crate) async fn load_description(directory: &Path, language: Option<&str>) -> Option<String> {
    let path = directory.join(SKILL_FILENAME);
    let text = read_front_matter_text(&path).await.ok()?;
    let (metadata, _) = parse_skill_front_matter(&text).ok()??;
    let description = language
        .map(|language| format!("description_{}", language.trim().to_lowercase()))
        .and_then(|key| metadata.get(&key).and_then(Value::as_str))
        .or_else(|| metadata.get("description").and_then(Value::as_str))?
        .trim();
    if description.is_empty() {
        return None;
    }
    Some(description.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn write_skill(root: &Path, relative: &str, description: Option<&str>) {
        let directory = root.join(relative);
        std::fs::create_dir_all(&directory).unwrap();
        let content = match description {
            Some(description) => {
                format!("---\nname: ignored\ndescription: {description}\n---\n# skill\n")
            }
            None => "# no front matter\n".to_string(),
        };
        std::fs::write(directory.join(SKILL_FILENAME), content).unwrap();
    }

    fn names(skills: &[Skill]) -> Vec<&str> {
        skills.iter().map(|skill| skill.name.as_str()).collect()
    }

    #[tokio::test]
    async fn groups_are_walked_and_skip_or_prune_entries_are_dropped() {
        let temp = temp_dir();
        let root = temp.path();
        write_skill(root, "a-group/first", None);
        write_skill(root, "z-last", None);
        write_skill(root, "z-last/private-nested", None);
        write_skill(root, ".hidden/hidden", None);
        write_skill(root, "assets/asset", None);
        write_skill(root, "output/out", None);
        write_skill(root, "temp/t", None);
        write_skill(root, "node_modules/n", None);

        let skills = discover_skills(&[root.to_path_buf()], None).await.unwrap();
        assert_eq!(names(&skills), ["first", "z-last"]);
    }

    #[tokio::test]
    async fn sibling_directories_are_visited_in_name_order() {
        let temp = temp_dir();
        let root = temp.path();
        write_skill(root, "b", None);
        write_skill(root, "a", None);
        write_skill(root, "c", None);

        let skills = discover_skills(&[root.to_path_buf()], None).await.unwrap();
        assert_eq!(names(&skills), ["a", "b", "c"]);
    }

    #[tokio::test]
    async fn duplicate_names_keep_the_first_root_and_later_unique_skills() {
        let first = temp_dir();
        let second = temp_dir();
        write_skill(first.path(), "alpha", None);
        write_skill(second.path(), "alpha", None);
        write_skill(second.path(), "beta", None);

        let roots = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        let skills = discover_skills(&roots, None).await.unwrap();
        assert_eq!(names(&skills), ["alpha", "beta"]);
        assert_eq!(skills[0].directory, first.path().join("alpha"));
        assert_eq!(skills[1].directory, second.path().join("beta"));
    }

    #[tokio::test]
    async fn a_root_that_is_itself_a_skill_yields_only_itself() {
        let temp = temp_dir();
        let root = temp.path();
        write_skill(root, "", None);
        write_skill(root, "nested", None);

        let skills = discover_skills(&[root.to_path_buf()], None).await.unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].directory, root);
        assert_eq!(
            skills[0].name,
            temp.path().file_name().unwrap().to_str().unwrap()
        );
    }

    #[tokio::test]
    async fn missing_roots_and_file_roots_are_skipped() {
        let temp = temp_dir();
        let missing = temp.path().join("missing");
        let file = temp.path().join("library.txt");
        std::fs::write(&file, "not a directory").unwrap();

        let skills = discover_skills(&[missing, file], None).await.unwrap();
        assert_eq!(skills, Vec::<Skill>::new());
    }

    #[tokio::test]
    async fn descriptions_prefer_the_language_key_and_fall_back() {
        let temp = temp_dir();
        let root = temp.path();
        write_skill(root, "plain", Some("Plain description"));
        let localized = root.join("localized");
        std::fs::create_dir_all(&localized).unwrap();
        std::fs::write(
            localized.join(SKILL_FILENAME),
            "---\nname: localized\ndescription: Plain description\ndescription_zh: Chinese description\n---\n# skill\n",
        )
        .unwrap();
        write_skill(root, "fallback", None);

        let skills = discover_skills(&[root.to_path_buf()], Some(" ZH "))
            .await
            .unwrap();
        assert_eq!(skills[0].name, "fallback");
        assert_eq!(skills[1].name, "localized");
        assert_eq!(skills[1].description, "Chinese description");
        assert_eq!(skills[2].name, "plain");
        assert_eq!(skills[2].description, "Plain description");
        assert_eq!(
            skills[0].description,
            format!("Skill located in {}", root.join("fallback").display())
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_cycle_terminates_and_does_not_duplicate() {
        let temp = temp_dir();
        let root = temp.path();
        write_skill(root, "group/alpha", None);
        std::os::unix::fs::symlink(root, root.join("group").join("loop")).unwrap();

        let skills = discover_skills(&[root.to_path_buf()], None).await.unwrap();
        assert_eq!(names(&skills), ["alpha"]);
    }

    #[tokio::test]
    async fn load_description_returns_none_without_front_matter() {
        let temp = temp_dir();
        let root = temp.path();
        write_skill(root, "plain", None);
        assert_eq!(load_description(&root.join("plain"), None).await, None);
    }
}
