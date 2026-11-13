//! Atomic filesystem persistence for flow evidence, recipes, and packages.
//!
//! Layout (all relative to the flow root):
//!
//! - `evidence.jsonl` — one normalized [`FlowTrace`] per line, ordered by
//!   `trace_id`; the whole file is rewritten atomically so a reader never sees a
//!   partial record.
//! - `recipes/<recipe_id>/v<N>.json` — immutable recipe versions.
//! - `recipes/<recipe_id>/current.json` — `{"recipe_id", "version"}` pointer.
//! - `packages/<package_id>.json` — the frozen, content-addressed
//!   [`SkillPack`].
//!
//! Every write goes to a uniquely named `.<name>.<uuid>.tmp` sibling and is then
//! renamed over the target. Malformed files found while loading are logged and
//! skipped instead of failing the whole scan, matching the crate's registry
//! behaviour.

use std::path::{Path, PathBuf};

use openjiuwen_common::utils::id::uuid_hex;
use tokio::fs;
use tracing::warn;

use crate::error::{validation, Error, Result};
use crate::flow::model::ExperienceRecipe;
use crate::flow::model::FlowTrace;
use crate::flow::pack::SkillPack;

/// The evidence file name.
pub(crate) const EVIDENCE_FILENAME: &str = "evidence.jsonl";
/// The recipe directory name.
pub(crate) const RECIPES_DIRNAME: &str = "recipes";
/// The package directory name.
pub(crate) const PACKAGES_DIRNAME: &str = "packages";
/// The current-version pointer file name.
pub(crate) const CURRENT_FILENAME: &str = "current.json";

/// A filesystem-backed flow store rooted at one directory.
pub(crate) struct FlowStore {
    root: PathBuf,
}

impl FlowStore {
    /// Creates a store rooted at `root` without touching the filesystem.
    pub(crate) fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the configured root directory.
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// Creates the root directory (and the optional subdirectories lazily).
    pub(crate) async fn ensure_layout(&self) -> Result<()> {
        fs::create_dir_all(&self.root)
            .await
            .map_err(|error| io_error("create flow root", error))
    }

    /// Atomically rewrites `evidence.jsonl` from `traces`.
    pub(crate) async fn write_evidence(&self, traces: &[FlowTrace]) -> Result<()> {
        let mut body = String::new();
        for trace in traces {
            let line = serde_json::to_string(trace)
                .map_err(|error| json_error("serialize evidence", &error))?;
            body.push_str(&line);
            body.push('\n');
        }
        atomic_write(&self.root.join(EVIDENCE_FILENAME), body.as_bytes()).await
    }

    /// Loads every valid evidence record, in file order.
    pub(crate) async fn load_evidence(&self) -> Result<Vec<FlowTrace>> {
        let path = self.root.join(EVIDENCE_FILENAME);
        let text = match fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_error("read evidence", error)),
        };
        let mut traces = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<FlowTrace>(line) {
                Ok(trace) => traces.push(trace),
                Err(error) => warn!(%error, "skipping malformed evidence record"),
            }
        }
        Ok(traces)
    }

    /// Persists one immutable recipe version and refreshes `current.json`.
    pub(crate) async fn save_recipe(&self, recipe: &ExperienceRecipe) -> Result<()> {
        if !is_safe_segment(&recipe.recipe_id) {
            return Err(validation("recipe_id", "must be a safe path segment"));
        }
        if recipe.version == 0 {
            return Err(validation("version", "must be a positive integer"));
        }
        let directory = self.root.join(RECIPES_DIRNAME).join(&recipe.recipe_id);
        let version_path = directory.join(format!("v{}.json", recipe.version));
        let body = serde_json::to_vec_pretty(recipe)
            .map_err(|error| json_error("serialize recipe", &error))?;
        atomic_write(&version_path, &body).await?;
        let pointer = serde_json::json!({
            "recipe_id": recipe.recipe_id,
            "version": recipe.version,
        });
        let pointer_body = serde_json::to_vec_pretty(&pointer)
            .map_err(|error| json_error("serialize recipe pointer", &error))?;
        atomic_write(&directory.join(CURRENT_FILENAME), &pointer_body).await
    }

    /// Loads the current version of every recipe, sorted by recipe id.
    pub(crate) async fn load_recipes(&self) -> Result<Vec<ExperienceRecipe>> {
        let root = self.root.join(RECIPES_DIRNAME);
        let mut directories = match fs::read_dir(&root).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_error("list recipes", error)),
        };
        let mut recipe_ids = Vec::new();
        while let Some(entry) = directories
            .next_entry()
            .await
            .map_err(|error| io_error("read recipe entry", error))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| io_error("read recipe entry type", error))?;
            if !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if is_safe_segment(&name) {
                recipe_ids.push(name);
            } else {
                warn!(name = %name, "skipping unsafe recipe directory");
            }
        }
        recipe_ids.sort();
        let mut recipes = Vec::with_capacity(recipe_ids.len());
        for recipe_id in recipe_ids {
            if let Some(recipe) = self.load_recipe(&recipe_id).await? {
                recipes.push(recipe);
            }
        }
        Ok(recipes)
    }

    /// Loads the current version of one recipe directory.
    async fn load_recipe(&self, recipe_id: &str) -> Result<Option<ExperienceRecipe>> {
        let directory = self.root.join(RECIPES_DIRNAME).join(recipe_id);
        let version = match self.read_current_version(&directory).await? {
            Some(version) => Some(version),
            None => highest_version(&directory)
                .await
                .map_err(|error| io_error("scan recipe versions", error))?,
        };
        let Some(version) = version else {
            return Ok(None);
        };
        let path = directory.join(format!("v{version}.json"));
        match fs::read_to_string(&path).await {
            Ok(text) => match serde_json::from_str::<ExperienceRecipe>(&text) {
                Ok(recipe) => Ok(Some(recipe)),
                Err(error) => {
                    warn!(%error, path = %path.display(), "skipping malformed recipe");
                    Ok(None)
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(io_error("read recipe", error)),
        }
    }

    /// Reads the `current.json` version pointer of one recipe directory.
    async fn read_current_version(&self, directory: &Path) -> Result<Option<u64>> {
        let path = directory.join(CURRENT_FILENAME);
        let text = match fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error("read recipe pointer", error)),
        };
        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(error) => {
                warn!(%error, path = %path.display(), "ignoring malformed recipe pointer");
                return Ok(None);
            }
        };
        Ok(value.get("version").and_then(serde_json::Value::as_u64))
    }

    /// Persists one `SkillPack` JSON artifact.
    pub(crate) async fn save_package(&self, pack: &SkillPack) -> Result<()> {
        if !is_safe_segment(&pack.package_id) {
            return Err(validation("package_id", "must be a safe path segment"));
        }
        let path = self
            .root
            .join(PACKAGES_DIRNAME)
            .join(format!("{}.json", pack.package_id));
        let body = serde_json::to_vec_pretty(pack)
            .map_err(|error| json_error("serialize package", &error))?;
        atomic_write(&path, &body).await
    }

    /// Loads every valid persisted `SkillPack`, sorted by package id.
    pub(crate) async fn load_packages(&self) -> Result<Vec<SkillPack>> {
        let root = self.root.join(PACKAGES_DIRNAME);
        let mut entries = match fs::read_dir(&root).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_error("list packages", error)),
        };
        let mut paths = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| io_error("read package entry", error))?
        {
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                paths.push(path);
            }
        }
        paths.sort();
        let mut packs = Vec::with_capacity(paths.len());
        for path in paths {
            match fs::read_to_string(&path).await {
                Ok(text) => match serde_json::from_str::<SkillPack>(&text) {
                    Ok(pack) => packs.push(pack),
                    Err(error) => {
                        warn!(%error, path = %path.display(), "skipping malformed package");
                    }
                },
                Err(error) => warn!(%error, path = %path.display(), "skipping unreadable package"),
            }
        }
        Ok(packs)
    }
}

/// Finds the highest `vN.json` version in a recipe directory.
async fn highest_version(directory: &Path) -> std::io::Result<Option<u64>> {
    let mut entries = match fs::read_dir(directory).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut highest = None;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(number) = name
            .strip_prefix('v')
            .and_then(|rest| rest.strip_suffix(".json"))
        else {
            continue;
        };
        if let Ok(version) = number.parse::<u64>() {
            if version > 0 {
                highest = Some(highest.map_or(version, |current: u64| current.max(version)));
            }
        }
    }
    Ok(highest)
}

/// Atomically writes `contents` to `path` via a unique temporary sibling.
pub(crate) async fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| validation("path", "has no parent directory"))?;
    fs::create_dir_all(parent)
        .await
        .map_err(|error| io_error("create directory", error))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| validation("path", "has no usable file name"))?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", uuid_hex()));
    fs::write(&temporary, contents)
        .await
        .map_err(|error| io_error("write temporary file", error))?;
    fs::rename(&temporary, path)
        .await
        .map_err(|error| io_error("publish atomic file", error))
}

/// Reports whether `segment` is safe to use as a single path component.
fn is_safe_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment != "."
        && segment != ".."
        && !segment.contains('/')
        && !segment.contains('\\')
}

/// Maps an I/O failure to [`Error::External`] with context.
#[allow(clippy::needless_pass_by_value)]
fn io_error(context: &str, error: std::io::Error) -> Error {
    Error::external(std::io::Error::new(
        error.kind(),
        format!("{context}: {error}"),
    ))
}

/// Maps a JSON failure to [`Error::External`] with context.
fn json_error(context: &str, error: &serde_json::Error) -> Error {
    Error::external(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("{context}: {error}"),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn safe_segments_reject_traversal() {
        assert!(is_safe_segment("recipe_0123456789ab"));
        assert!(!is_safe_segment(""));
        assert!(!is_safe_segment(".."));
        assert!(!is_safe_segment("a/b"));
        assert!(!is_safe_segment("a\\b"));
    }

    #[test]
    fn json_and_io_errors_keep_their_kind() {
        let error = io_error(
            "ctx",
            std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
        );
        assert_eq!(error.kind(), crate::error::ErrorKind::External);
        let parse = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        assert_eq!(
            json_error("ctx", &parse).kind(),
            crate::error::ErrorKind::External
        );
    }
}
