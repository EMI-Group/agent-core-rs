//! Discovery and lookup of skills over an in-memory registry.
//!
//! Ports the registry of `openjiuwen/core/single_agent/skills/skill_manager.py`
//! onto `tokio::fs`: [`SkillManager::load_dir`] scans a directory for `*.md`
//! documents (plus one level of `<skill>/SKILL.md` subdirectories) and
//! [`SkillManager::load_file`] parses a single document.
use super::skill::Skill;
use openjiuwen_common::error::{Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// The in-memory skill registry: names mapped to skills, sorted by name.
///
/// A skill name is unique; adding a second skill with an existing name replaces
/// the earlier one and logs a warning with `tracing::warn!`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkillManager {
    skills: BTreeMap<String, Skill>,
}

impl SkillManager {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry holding `skills`, in iteration order.
    #[must_use]
    pub fn with_skills(skills: impl IntoIterator<Item = Skill>) -> Self {
        let mut manager = Self::new();
        for skill in skills {
            manager.add(skill);
        }
        manager
    }

    /// Adds `skill`, replacing — and warning about — a skill of the same name.
    pub fn add(&mut self, skill: Skill) {
        if self.skills.contains_key(&skill.name) {
            tracing::warn!(
                target: "openjiuwen_agent::skills",
                name = %skill.name,
                path = skill.path.as_deref().unwrap_or("<memory>"),
                "duplicate skill name replaces the earlier skill"
            );
        }
        self.skills.insert(skill.name.clone(), skill);
    }

    /// Removes and returns the skill named `name`.
    pub fn remove(&mut self, name: &str) -> Option<Skill> {
        self.skills.remove(name)
    }

    /// Returns the skill named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// Returns the number of registered skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether no skill is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Returns every skill, sorted by name.
    #[must_use]
    pub fn list(&self) -> Vec<Skill> {
        self.skills.values().cloned().collect()
    }

    /// Returns the skills matching `query` (see [`Skill::matches`]), sorted by
    /// name.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<Skill> {
        self.skills
            .values()
            .filter(|skill| skill.matches(query))
            .cloned()
            .collect()
    }

    /// Loads every skill under `path`.
    ///
    /// `path` may be a single skill document or a directory. A directory is
    /// scanned non-recursively: every `*.md` file directly inside it is a skill,
    /// and every immediate subdirectory is a skill bundle when it holds a
    /// `SKILL.md` (matched case-insensitively); other entries are skipped.
    /// Carriage-return line endings and front matter are handled by
    /// [`Skill::from_markdown`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when `path` cannot be inspected or read, and
    /// [`Error::Validation`] when a loaded document is not a valid skill.
    pub async fn load_dir(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)
            .await
            .map_err(|error| scan_error(path, &error))?;
        let mut manager = Self::new();
        if metadata.is_file() {
            manager.add(Self::load_file(path).await?);
            return Ok(manager);
        }
        if !metadata.is_dir() {
            return Err(Error::validation(format!(
                "skill path {} is neither a file nor a directory",
                path.display()
            )));
        }
        let mut files = Vec::new();
        let mut directories = Vec::new();
        let mut entries = fs::read_dir(path)
            .await
            .map_err(|error| scan_error(path, &error))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| scan_error(path, &error))?
        {
            let entry_path = entry.path();
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| scan_error(&entry_path, &error))?;
            if file_type.is_dir() {
                directories.push(entry_path);
            } else if is_markdown(&entry_path) {
                files.push(entry_path);
            }
        }
        files.sort();
        directories.sort();
        for file in files {
            manager.add(Self::load_file(&file).await?);
        }
        for directory in directories {
            if let Some(document) = find_skill_document(&directory).await? {
                manager.add(Self::load_file(&document).await?);
            }
        }
        Ok(manager)
    }

    /// Loads the skill document at `path` and records that path on the skill.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when the file cannot be read, and
    /// [`Error::Validation`] when it is not a valid skill (the message names the
    /// file).
    pub async fn load_file(path: impl AsRef<Path>) -> Result<Skill> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .await
            .map_err(|error| scan_error(path, &error))?;
        let mut skill = Skill::parse_markdown(&text, &path.display().to_string())?;
        skill.path = Some(path.display().to_string());
        Ok(skill)
    }

    /// Loads `path` and returns its skills sorted by name.
    ///
    /// A convenience wrapper over [`SkillManager::load_dir`] for callers that
    /// only want the skills.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`SkillManager::load_dir`].
    pub async fn discover(path: impl AsRef<Path>) -> Result<Vec<Skill>> {
        Ok(Self::load_dir(path).await?.list())
    }
}

/// Whether `path` has a `.md` extension (case-insensitive).
fn is_markdown(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
}

/// Returns the `SKILL.md` document directly inside `directory`, if any.
async fn find_skill_document(directory: &Path) -> Result<Option<PathBuf>> {
    let mut entries = fs::read_dir(directory)
        .await
        .map_err(|error| scan_error(directory, &error))?;
    let mut candidates = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| scan_error(directory, &error))?
    {
        let entry_path = entry.path();
        let is_skill_document = entry_path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("skill.md"));
        if !is_skill_document {
            continue;
        }
        let file_type = entry
            .file_type()
            .await
            .map_err(|error| scan_error(&entry_path, &error))?;
        if file_type.is_file() {
            candidates.push(entry_path);
        }
    }
    candidates.sort();
    Ok(candidates.into_iter().next())
}

/// Builds a [`Error::SysOp`] naming the path a filesystem call failed on.
fn scan_error(path: &Path, error: &std::io::Error) -> Error {
    Error::SysOp(format!("skill path {}: {error}", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn skill(name: &str, description: &str) -> Skill {
        Skill::new(name, description)
    }

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }

    #[test]
    fn add_get_remove_and_len() {
        let mut manager = SkillManager::new();
        assert!(manager.is_empty());
        manager.add(skill("beta", "second"));
        manager.add(skill("alpha", "first"));
        assert_eq!(manager.len(), 2);
        assert!(!manager.is_empty());
        assert_eq!(
            manager.get("alpha").map(|skill| skill.description.as_str()),
            Some("first")
        );
        assert!(manager.get("missing").is_none());
        assert_eq!(manager.remove("beta").unwrap().name, "beta");
        assert!(manager.remove("beta").is_none());
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn list_and_search_are_sorted_by_name() {
        let manager = SkillManager::with_skills([
            skill("gamma", "Deploy the service"),
            skill("alpha", "Build the service"),
            skill("beta", "Rollback the service"),
        ]);
        let names: Vec<String> = manager.list().into_iter().map(|skill| skill.name).collect();
        assert_eq!(names, ["alpha", "beta", "gamma"]);
        let matched: Vec<String> = manager
            .search("deploy")
            .into_iter()
            .map(|skill| skill.name)
            .collect();
        assert_eq!(matched, ["gamma"]);
        assert!(manager.search("nothing here").is_empty());
    }

    #[test]
    fn duplicate_names_keep_the_last_skill() {
        let mut manager = SkillManager::with_skills([skill("alpha", "first")]);
        manager.add(skill("alpha", "second").with_metadata("owner", json!("platform")));
        assert_eq!(manager.len(), 1);
        let stored = manager.get("alpha").unwrap();
        assert_eq!(stored.description, "second");
        assert_eq!(stored.metadata["owner"], json!("platform"));
    }

    #[tokio::test]
    async fn load_dir_reads_markdown_files_with_and_without_front_matter() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir.path().join("deploy.md"),
            "---\r\nname: deploy\r\ndescription: Deploy the service\r\nkeywords:\r\n  - release\r\n---\r\n\r\nRun it.\r\n",
        );
        write(&dir.path().join("notes.md"), "# Notes\n\nTake notes.\n");
        write(&dir.path().join("ignored.txt"), "not a skill");
        let manager = SkillManager::load_dir(dir.path()).await.unwrap();
        assert_eq!(manager.len(), 2);
        let deploy = manager.get("deploy").unwrap();
        assert_eq!(deploy.description, "Deploy the service");
        assert_eq!(deploy.keywords, ["release"]);
        assert_eq!(deploy.body, "Run it.");
        assert!(deploy.path.as_deref().unwrap().ends_with("deploy.md"));
        let notes = manager.get("Notes").unwrap();
        assert_eq!(notes.body, "Take notes.");
        assert!(notes.path.is_some());
    }

    #[tokio::test]
    async fn load_dir_reads_one_level_of_skill_bundles() {
        let dir = tempfile::tempdir().unwrap();
        write(
            &dir.path().join("alpha/SKILL.md"),
            "---\nname: alpha\ndescription: first\n---\nalpha body",
        );
        write(
            &dir.path().join("beta/skill.md"),
            "---\nname: beta\ndescription: second\n---\nbeta body",
        );
        write(&dir.path().join("gamma/README.md"), "# not a bundle\n");
        let manager = SkillManager::load_dir(dir.path()).await.unwrap();
        let names: Vec<String> = manager.list().into_iter().map(|skill| skill.name).collect();
        assert_eq!(names, ["alpha", "beta"]);
    }

    #[tokio::test]
    async fn load_dir_accepts_a_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single.md");
        write(&path, "# Single\n\nOnly one.\n");
        let manager = SkillManager::load_dir(&path).await.unwrap();
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.get("Single").unwrap().body, "Only one.");
    }

    #[tokio::test]
    async fn load_file_sets_the_path_and_reports_invalid_documents() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.md");
        write(&good, "# Good\n\nBody.\n");
        let skill = SkillManager::load_file(&good).await.unwrap();
        assert_eq!(skill.name, "Good");
        assert_eq!(skill.path.as_deref(), Some(good.to_string_lossy().as_ref()));

        let bad = dir.path().join("bad.md");
        write(&bad, "---\nname: only-name\n---\nbody");
        let error = SkillManager::load_file(&bad).await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("bad.md"), "{error}");

        let missing = dir.path().join("missing.md");
        let error = SkillManager::load_file(&missing).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::SysOp);
        assert!(error.to_string().contains("missing.md"), "{error}");
    }

    #[tokio::test]
    async fn load_dir_reports_a_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let error = SkillManager::load_dir(dir.path().join("nope"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::SysOp);
    }

    #[tokio::test]
    async fn discover_returns_the_skills_sorted_by_name() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("a.md"), "# Zebra\n\nz\n");
        write(&dir.path().join("b.md"), "# Ant\n\na\n");
        let skills = SkillManager::discover(dir.path()).await.unwrap();
        let names: Vec<String> = skills.into_iter().map(|skill| skill.name).collect();
        assert_eq!(names, ["Ant", "Zebra"]);
    }
}
