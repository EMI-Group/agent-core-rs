//! On-disk skill bundles: `<root>/<name>/SKILL.md` plus its text resources.
//!
//! The layout is the one the harness discovers (`_discover`/`_name` in the
//! Python `harness_providers.skills`): one directory per skill, a `SKILL.md`
//! document at its root, and any number of files beneath it at relative paths.
//! [`SkillBundle::write`] creates that tree, [`SkillBundle::read`] reconstructs a
//! [`SkillDefinition`] from it.
//!
//! Reading is conservative: symlinks are skipped (never followed), files that are
//! not valid UTF-8 are skipped with a [`tracing::debug`] event, and the remaining
//! resources come back sorted by path so the result is deterministic.

use std::path::{Component, Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use crate::skill_creator::definition::{parse_markdown, SkillDefinition, SkillResource};

/// The bundle document name, at the root of every skill directory.
pub const SKILL_DOCUMENT: &str = "SKILL.md";

/// Associated functions for writing and reading skill bundles.
///
/// This type is never instantiated: it only groups the two filesystem entry
/// points so call sites read as [`SkillBundle::write`] / [`SkillBundle::read`],
/// matching the Python module-level helpers.
#[derive(Debug, Clone, Copy)]
pub struct SkillBundle;

impl SkillBundle {
    /// Writes `definition` under `root/<name>/`.
    ///
    /// `SKILL.md` is created first, then every resource at its relative path;
    /// missing parent directories are created. Existing files at the same paths
    /// are overwritten. Returns the skill directory.
    ///
    /// # Errors
    ///
    /// Returns whatever [`SkillDefinition::validate`] rejects, plus
    /// [`Error::external`](openjiuwen_common::error::Error::external) when a
    /// filesystem operation fails.
    pub async fn write(definition: &SkillDefinition, root: &Path) -> Result<PathBuf> {
        definition.validate()?;
        let directory = root.join(definition.skill_dir_name());
        tokio::fs::create_dir_all(&directory).await?;

        let markdown = definition.to_markdown()?;
        tokio::fs::write(directory.join(SKILL_DOCUMENT), markdown.as_bytes()).await?;

        for resource in &definition.resources {
            let target = directory.join(relative_path(&resource.path));
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&target, resource.content.as_bytes()).await?;
        }
        Ok(directory)
    }

    /// Reads the skill bundle stored in `directory`.
    ///
    /// The document supplies name, description, instructions, version, and
    /// metadata; every other regular UTF-8 file beneath `directory` becomes a
    /// [`SkillResource`] whose path is `/`-joined relative to the bundle root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// when `directory` holds no `SKILL.md`, the document is malformed, or the
    /// recovered definition does not validate;
    /// [`Error::external`](openjiuwen_common::error::Error::external) when
    /// reading the tree fails.
    pub async fn read(directory: &Path) -> Result<SkillDefinition> {
        let document = directory.join(SKILL_DOCUMENT);
        let text = match tokio::fs::read_to_string(&document).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::validation(format!(
                    "no {SKILL_DOCUMENT} in skill directory {}",
                    directory.display()
                )));
            }
            Err(error) => return Err(Error::from(error)),
        };

        let (frontmatter, body) = parse_markdown(&text)?;
        let mut definition = SkillDefinition::new(frontmatter.name, frontmatter.description, body);
        definition.version = frontmatter.version;
        definition.metadata = frontmatter.extra;

        let mut resources = Vec::new();
        collect_resources(directory, directory, &mut resources).await?;
        resources.sort_by(|left, right| left.path.cmp(&right.path));
        definition.resources = resources;
        definition.validate()?;
        Ok(definition)
    }
}

/// Recursively collects the text files beneath `directory`.
///
/// Paths are made relative to `root`; the root `SKILL.md` itself, symlinks, and
/// non-UTF-8 files are skipped.
async fn collect_resources(
    root: &Path,
    directory: &Path,
    resources: &mut Vec<SkillResource>,
) -> Result<()> {
    let mut entries = tokio::fs::read_dir(directory).await?;
    let mut paths = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        paths.push(entry.path());
    }
    paths.sort();

    for path in paths {
        let file_type = tokio::fs::symlink_metadata(&path).await?.file_type();
        if file_type.is_symlink() {
            tracing::debug!(path = %path.display(), "skipping symlink in skill bundle");
            continue;
        }
        if file_type.is_dir() {
            Box::pin(collect_resources(root, &path, resources)).await?;
            continue;
        }
        if !file_type.is_file() {
            tracing::debug!(path = %path.display(), "skipping non-regular file in skill bundle");
            continue;
        }

        let relative = posix_relative(root, &path)?;
        if relative.eq_ignore_ascii_case(SKILL_DOCUMENT) {
            continue;
        }
        match tokio::fs::read(&path).await {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(content) => resources.push(SkillResource::new(relative, content)),
                Err(error) => {
                    tracing::debug!(path = %path.display(), %error, "skipping non-UTF-8 skill resource");
                }
            },
            Err(error) => return Err(Error::from(error)),
        }
    }
    Ok(())
}

/// Builds the on-disk path for a validated relative POSIX resource path.
fn relative_path(path: &str) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.split('/') {
        result.push(component);
    }
    result
}

/// Formats `path` relative to `root` with `/` separators.
fn posix_relative(root: &Path, path: &Path) -> Result<String> {
    let relative = path.strip_prefix(root).map_err(|_| {
        Error::validation(format!(
            "skill resource {} escapes its bundle",
            path.display()
        ))
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(name) => parts.push(name.to_string_lossy().into_owned()),
            other => {
                return Err(Error::validation(format!(
                    "unsupported skill resource path component: {other:?}"
                )));
            }
        }
    }
    if parts.is_empty() {
        return Err(Error::validation(format!(
            "skill resource {} has an empty relative path",
            path.display()
        )));
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_test_support::TempWorkspace;
    use serde_json::json;

    fn skill() -> SkillDefinition {
        SkillDefinition::new(
            "demo-skill",
            "A demo skill",
            "# Demo\n\nUse `references/guide.md`.",
        )
        .with_version("0.1.0")
        .with_metadata("author", json!("alice"))
        .with_resource(SkillResource::new(
            "references/guide.md",
            "# Guide\n\nStep by step.\n",
        ))
        .with_resource(SkillResource::new("notes.txt", "plain notes"))
    }

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let root = workspace.path("skills");

        let directory = SkillBundle::write(&skill(), &root).await.expect("writes");
        assert_eq!(directory, root.join("demo-skill"));

        let document = tokio::fs::read_to_string(directory.join(SKILL_DOCUMENT))
            .await
            .expect("document");
        assert!(document.starts_with("---\nname: demo-skill\ndescription: A demo skill\n"));
        assert!(document.contains("version: 0.1.0"));
        assert!(document.contains("author: alice"));
        assert!(document.ends_with("---\n\n# Demo\n\nUse `references/guide.md`."));

        let guide = tokio::fs::read_to_string(directory.join("references/guide.md"))
            .await
            .expect("nested resource");
        assert_eq!(guide, "# Guide\n\nStep by step.\n");

        let mut expected = skill();
        expected
            .resources
            .sort_by(|left, right| left.path.cmp(&right.path));
        assert_eq!(
            SkillBundle::read(&directory).await.expect("reads"),
            expected
        );
    }

    #[tokio::test]
    async fn write_creates_missing_root() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let root = workspace.path("deeply/nested/skills");
        let directory = SkillBundle::write(&skill(), &root).await.expect("writes");
        assert!(tokio::fs::try_exists(directory.join(SKILL_DOCUMENT))
            .await
            .expect("probe"));
    }

    #[tokio::test]
    async fn write_rejects_an_invalid_definition() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let mut invalid = skill();
        invalid.name = "../escape".to_owned();
        let error = SkillBundle::write(&invalid, workspace.root())
            .await
            .expect_err("invalid name");
        assert!(error.to_string().starts_with("validation:"), "{error}");
        assert!(!workspace.path("../escape").exists());
    }

    #[tokio::test]
    async fn read_requires_a_document() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let directory = workspace.create_dir("empty-skill").expect("directory");
        let error = SkillBundle::read(&directory)
            .await
            .expect_err("no SKILL.md");
        assert!(error.to_string().contains("SKILL.md"), "{error}");
    }

    #[tokio::test]
    async fn read_skips_non_utf8_and_symlinks() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        let directory = SkillBundle::write(&skill(), workspace.root())
            .await
            .expect("writes");

        tokio::fs::write(directory.join("binary.bin"), [0xff, 0xfe, 0x00])
            .await
            .expect("binary file");

        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc/hostname", directory.join("link.txt")).expect("symlink");

        let parsed = SkillBundle::read(&directory).await.expect("reads");
        let paths: Vec<&str> = parsed
            .resources
            .iter()
            .map(|resource| resource.path.as_str())
            .collect();
        assert_eq!(paths, vec!["notes.txt", "references/guide.md"]);
    }

    #[tokio::test]
    async fn read_keeps_the_body_and_ignores_extra_document_fields() {
        let workspace = TempWorkspace::new().expect("temp workspace");
        workspace
            .write(
                "skills/manual/SKILL.md",
                "---\nname: manual\ndescription: Hand written\ncustom: 1\n---\n\nDo it.\n",
            )
            .expect("document");
        let parsed = SkillBundle::read(&workspace.path("skills/manual"))
            .await
            .expect("reads");
        assert_eq!(parsed.name, "manual");
        assert_eq!(parsed.instructions, "Do it.");
        assert_eq!(parsed.metadata.get("custom"), Some(&json!(1)));
        assert_eq!(parsed.resources.len(), 0);
    }

    #[test]
    fn relative_path_splits_on_slashes() {
        assert_eq!(
            relative_path("a/b/c.md"),
            PathBuf::from("a").join("b").join("c.md")
        );
    }
}
