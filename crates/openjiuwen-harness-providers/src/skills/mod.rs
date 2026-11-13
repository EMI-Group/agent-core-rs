//! Portable skill-bundle normalization and installation.
//!
//! Ports Python `openjiuwen.harness_providers.skills` (v0.1.20): a provider configuration names
//! one or more skill sources — a single bundle, or a library directory containing bundles — and
//! the installer copies the complete bundles into the CLI's project-local discovery directory
//! (`claudecode` → `.claude/skills`, `codex` → `.agents/skills`, `dsh` → `.dsh/skills`).
//!
//! A bundle is a directory that contains `SKILL.md`; libraries are searched recursively, sorted by
//! path, and a directory that contains `SKILL.md` is not descended further. The skill name comes
//! from the `name:` entry of the document's front matter, falling back to the bundle directory
//! name. Internal symlinks are materialized (their contents are copied), while escaping links,
//! directory cycles and non-regular files are rejected before anything is copied.
//!
//! Installation is staged inside the project so the final swap stays on one filesystem: the copy
//! is prepared under `.openjiuwen-skill-<uuid>/new`, an existing bundle is moved to
//! `.openjiuwen-skill-<uuid>/previous`, and a failed swap restores it. The stage directory is
//! always removed, so no partial destination is left behind.
//!
//! [`SkillInstaller`] adds the per-vendor policy on top of [`install_skills`]: the vendor's
//! discovery layout ([`SkillLayout`]) and its skill-name rule. DSH additionally
//! discovers a flat `<scan>/<name>.md` file, so a `*.md` source root is accepted there and
//! normalized into the portable `<scan>/<name>/SKILL.md` directory form that every vendor
//! discovers; the directory layouts reject such a source instead of guessing.
//!
//! # Python parity
//!
//! Conflict policies (`skip` / `replace` / `append`), error wording and the front-matter rewrite
//! of an appended bundle follow Python. Two deviations are deliberate and documented at the
//! affected items:
//!
//! - [`SkillConflict`] is a typed enum (Python passes one of three strings), so the conflict is
//!   validated at parse time and [`normalize_skills`] keeps the parameter only for signature
//!   parity;
//! - the portable-name rule accepts any non-empty name that is not `..` and uses only
//!   `[A-Za-z0-9._-]` (Python additionally requires an alphanumeric first character and at most
//!   128 characters).

use std::path::{Path, PathBuf};
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{JsonObject, JsonValue};
use serde::{Deserialize, Serialize};

use crate::decode::reject_unknown_keys;

mod installer;

pub use installer::{install_for_provider, SkillInstaller, SkillLayout};

/// Which skills a source exposes to the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillMode {
    /// Every bundle found in the source is installed.
    All,
    /// Only the bundles listed in [`SkillSource::enabled_skills`] are installed.
    AutoList,
}

/// What to do when the destination already declares the same skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillConflict {
    /// Keep the existing bundle and install nothing.
    Skip,
    /// Replace the existing bundle with the source copy.
    Replace,
    /// Install the source copy under the next free `name-N` directory.
    Append,
}

impl FromStr for SkillConflict {
    type Err = Error;

    /// Parses the Python wire values `"skip"`, `"replace"` and `"append"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `"skill_conflict must be skip, replace or append"` for
    /// any other value (Python's message for an invalid conflict string).
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "skip" => Ok(Self::Skip),
            "replace" => Ok(Self::Replace),
            "append" => Ok(Self::Append),
            _ => Err(Error::validation(
                "skill_conflict must be skip, replace or append",
            )),
        }
    }
}

/// A manifest skill directory or a library containing skill directories.
///
/// Construction never touches the filesystem; directories are resolved when
/// [`install_skills`] runs. `enabled_skills` of `None` or an empty list means "all bundles".
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SkillSource {
    /// Directory of a single bundle, or a library of bundles.
    pub dir: PathBuf,
    /// Which bundles of the source are exposed.
    pub mode: SkillMode,
    /// Names to install when the source is a library; `None` installs every bundle.
    pub enabled_skills: Option<Vec<String>>,
}

impl SkillSource {
    /// Creates a source that installs every bundle found under `dir` ([`SkillMode::All`]).
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            mode: SkillMode::All,
            enabled_skills: None,
        }
    }

    /// Replaces the skill selection mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SkillMode) -> Self {
        self.mode = mode;
        self
    }

    /// Restricts a library source to the given skill names.
    #[must_use]
    pub fn with_enabled_skills(
        mut self,
        enabled: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.enabled_skills = Some(enabled.into_iter().map(Into::into).collect());
        self
    }

    /// Decodes a skill declaration: a directory string or a manifest object.
    ///
    /// A string is a directory path with [`SkillMode::All`] and no name filter. An object accepts
    /// `dir` (or its alias `source`) plus the optional `mode` and `enabled_skills` keys; any other
    /// key is rejected. `null` for `mode` and `enabled_skills` means "not configured".
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the declaration has another type, the directory is
    /// missing or empty (`"skill dir must be a non-empty path"`), the mode is not `all` or
    /// `auto_list` (`"skill mode must be all or auto_list"`), `enabled_skills` is not an array of
    /// strings (`"enabled_skills must be an array of names"`), or the object has unknown keys
    /// (`"unknown skill source fields: …"`).
    pub fn from_json(value: &JsonValue) -> Result<Self> {
        match value {
            JsonValue::String(dir) => {
                if dir.is_empty() {
                    return Err(Error::validation("skill dir must be a non-empty path"));
                }
                Ok(Self::new(dir))
            }
            JsonValue::Object(object) => {
                let object = to_json_object(object);
                reject_unknown_keys(
                    &object,
                    &["dir", "source", "mode", "enabled_skills"],
                    "unknown skill source fields",
                )?;
                let dir = match object.get("dir").or_else(|| object.get("source")) {
                    Some(JsonValue::String(dir)) if !dir.is_empty() => dir.clone(),
                    _ => return Err(Error::validation("skill dir must be a non-empty path")),
                };
                Ok(Self {
                    dir: PathBuf::from(dir),
                    mode: parse_mode(object.get("mode"))?,
                    enabled_skills: parse_enabled_skills(object.get("enabled_skills"))?,
                })
            }
            _ => Err(Error::validation(
                "skill source must be a directory or manifest skill declaration",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for SkillSource {
    /// Accepts either a directory string or a manifest object, delegating to
    /// [`SkillSource::from_json`].
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = JsonValue::deserialize(deserializer)?;
        Self::from_json(&value).map_err(serde::de::Error::custom)
    }
}

/// Decodes every value of a skills array into a [`SkillSource`].
///
/// The `conflict` argument is kept for Python signature parity: [`SkillConflict`] is already a
/// validated enum, so the value is only recorded by the provider configuration and does not
/// affect normalization here.
///
/// # Errors
///
/// Returns the first [`SkillSource::from_json`] error encountered.
pub fn normalize_skills(
    values: &[JsonValue],
    _conflict: SkillConflict,
) -> Result<Vec<SkillSource>> {
    values.iter().map(SkillSource::from_json).collect()
}

/// Decodes the optional `skills` value of a provider configuration.
///
/// Absent and JSON `null` are treated as "no skills"; any other non-array value is rejected.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"skills must be an array"` for a non-null, non-array
/// value, and propagates the element errors of [`normalize_skills`].
// The vendor configuration decoders added in a later phase call this helper.
#[allow(dead_code)]
pub(crate) fn parse_skills_value(value: Option<&JsonValue>) -> Result<Vec<SkillSource>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(values) = value.as_array() else {
        return Err(Error::validation("skills must be an array"));
    };
    values.iter().map(SkillSource::from_json).collect()
}

/// Returns the project-relative skill discovery directory of a built-in provider.
///
/// `claudecode` maps to `.claude/skills`, `codex` to `.agents/skills` and `dsh` to `.dsh/skills`;
/// every other provider has no built-in discovery directory and returns [`None`].
#[must_use]
pub fn skill_scan_dir(provider: &str) -> Option<&'static str> {
    match provider {
        "claudecode" => Some(".claude/skills"),
        "codex" => Some(".agents/skills"),
        "dsh" => Some(".dsh/skills"),
        _ => None,
    }
}

/// Copies portable skill bundles into the provider's project discovery directory.
///
/// The bundle or library is discovered before anything is written, so an invalid source fails
/// without side effects. Existing skills are handled according to `conflict`: `skip` keeps them,
/// `replace` swaps the bundle, and `append` installs under the next free `name-N` and rewrites the
/// copied front-matter `name:` entry. Bundles are staged under `.openjiuwen-skill-<uuid>` inside
/// `project` and moved into place atomically, restoring the previous bundle if the final rename
/// fails.
///
/// The discovery directory is created before it is canonicalized (Python resolves the not yet
/// existing path non-strictly); the canonical path must stay inside `cwd`, so a symlinked
/// discovery directory that escapes the project is rejected.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the provider has no discovery directory, the discovery
/// directory escapes the project, a source is not a directory, a source contains no bundle, a
/// skill name is not portable, a bundle contains an escaping symlink or a directory cycle, a
/// bundle contains a non-regular file, the source and destination contain each other, or multiple
/// existing directories declare the skill being replaced.
///
/// Returns [`Error::External`] for filesystem failures (including dangling symlinks in a bundle,
/// a missing source or project directory, and a destination that appears while an append copy is
/// staged).
pub fn install_skills(
    sources: &[SkillSource],
    provider: &str,
    cwd: &Path,
    conflict: SkillConflict,
) -> Result<Vec<PathBuf>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let (project, scan) = resolve_scan_dir(provider, cwd)?;
    let planned = plan_install(sources)?;
    let mut installed = Vec::new();
    for (name, source) in &planned {
        if let Some(destination) = install_one(&project, &scan, name, source, conflict)? {
            installed.push(destination);
        }
    }
    installed.sort();
    Ok(installed)
}

/// Converts the borrowed JSON object into the sorted [`JsonObject`] used by the decode helpers.
fn to_json_object(object: &serde_json::Map<String, JsonValue>) -> JsonObject {
    object
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Decodes the `mode` value, defaulting to [`SkillMode::All`].
fn parse_mode(value: Option<&JsonValue>) -> Result<SkillMode> {
    match value {
        None | Some(JsonValue::Null) => Ok(SkillMode::All),
        Some(JsonValue::String(mode)) if mode == "all" => Ok(SkillMode::All),
        Some(JsonValue::String(mode)) if mode == "auto_list" => Ok(SkillMode::AutoList),
        Some(_) => Err(Error::validation("skill mode must be all or auto_list")),
    }
}

/// Decodes the `enabled_skills` value, defaulting to "all bundles".
fn parse_enabled_skills(value: Option<&JsonValue>) -> Result<Option<Vec<String>>> {
    match value {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Array(items)) => {
            let mut names = Vec::with_capacity(items.len());
            for item in items {
                let Some(name) = item.as_str() else {
                    return Err(Error::validation(
                        "enabled_skills must be an array of names",
                    ));
                };
                names.push(name.to_owned());
            }
            Ok(Some(names))
        }
        Some(_) => Err(Error::validation(
            "enabled_skills must be an array of names",
        )),
    }
}

/// Resolves the project root and the canonical discovery directory inside it.
fn resolve_scan_dir(provider: &str, cwd: &Path) -> Result<(PathBuf, PathBuf)> {
    let relative = skill_scan_dir(provider).ok_or_else(|| {
        Error::validation(format!(
            "provider '{provider}' declares no skill discovery directory"
        ))
    })?;
    let project = cwd.canonicalize()?;
    let scan = project.join(relative);
    std::fs::create_dir_all(&scan)?;
    let scan = scan.canonicalize()?;
    if !scan.starts_with(&project) {
        return Err(Error::validation(
            "skill discovery directory escapes the project",
        ));
    }
    Ok((project, scan))
}

/// Discovers the bundles of every source and applies the name filter.
fn plan_install(sources: &[SkillSource]) -> Result<Vec<(String, PathBuf)>> {
    let mut planned = Vec::new();
    for source in sources {
        let root = source.dir.canonicalize()?;
        if !root.is_dir() {
            return Err(Error::validation(format!(
                "skill source is not a directory: {}",
                root.display()
            )));
        }
        let bundles = discover(&root)?;
        if bundles.is_empty() {
            return Err(Error::validation(format!(
                "no SKILL.md bundles found in {}",
                root.display()
            )));
        }
        for bundle in bundles {
            let name = skill_name(&bundle)?;
            if is_enabled(source.enabled_skills.as_deref(), &name) {
                planned.push((name, bundle));
            }
        }
    }
    Ok(planned)
}

/// Returns `true` when a bundle with `name` is selected by the (optional) name filter.
fn is_enabled(enabled: Option<&[String]>, name: &str) -> bool {
    enabled.is_none_or(|names| names.is_empty() || names.iter().any(|entry| entry == name))
}

/// Returns the bundles of `root` recursively, mirroring Python `_discover`.
fn discover(root: &Path) -> Result<Vec<PathBuf>> {
    if root.join("SKILL.md").is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut bundles = Vec::new();
    for entry in read_dir_sorted(root)? {
        // `file_type` does not follow symlinks, matching Python's `is_dir() and not is_symlink()`.
        if entry.file_type()?.is_dir() {
            bundles.extend(discover(&entry.path())?);
        }
    }
    Ok(bundles)
}

/// Reads the skill name of a bundle from its front matter, falling back to the directory name.
fn skill_name(bundle: &Path) -> Result<String> {
    let name = match read_front_matter_name(bundle)? {
        Some(name) => name,
        None => bundle
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
    };
    validate_portable_name(&name)?;
    Ok(name)
}

/// Checks `name` against the shared portable-name rule.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"invalid portable skill name: '<name>'"` when the name is
/// empty, is `..`, or uses a character outside `[A-Za-z0-9._-]`.
fn validate_portable_name(name: &str) -> Result<()> {
    if is_portable_name(name) {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "invalid portable skill name: '{name}'"
        )))
    }
}

/// Reads the top-level `name:` value of a bundle's front matter.
fn read_front_matter_name(bundle: &Path) -> Result<Option<String>> {
    let document = bundle.join("SKILL.md");
    let resolved = document.canonicalize()?;
    if !resolved.starts_with(bundle.canonicalize()?) {
        return Err(Error::validation("SKILL.md must stay inside its bundle"));
    }
    let text = std::fs::read_to_string(&resolved)?;
    document_front_matter_name(&text, bundle)
}

/// Returns the declared skill name of a skill document's text, if its front matter has one.
///
/// A byte-order mark is ignored, and a document without front matter declares no name.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `"unterminated skill front matter: <label>"` when the
/// document opens a front-matter block that it never closes.
fn document_front_matter_name(text: &str, label: &Path) -> Result<Option<String>> {
    let text = text
        .strip_prefix('\u{feff}')
        .map_or(text, |stripped| stripped);
    let lines: Vec<&str> = text.lines().collect();
    if lines.first().is_none_or(|line| line.trim() != "---") {
        return Ok(None);
    }
    let Some(end) = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, line)| line.trim() == "---")
        .map(|(index, _)| index)
    else {
        return Err(Error::validation(format!(
            "unterminated skill front matter: {}",
            label.display()
        )));
    };
    Ok(front_matter_name(&lines[1..end]))
}

/// Returns the last top-level `name:` value of the given front-matter lines.
fn front_matter_name(lines: &[&str]) -> Option<String> {
    let mut name = None;
    for line in lines {
        let Some(value) = top_level_value(line, "name") else {
            continue;
        };
        if value.is_empty() {
            // YAML `name:` without a value is null, so Python falls back to the directory name.
            continue;
        }
        name = Some(strip_quotes(value).to_owned());
    }
    name
}

/// Returns the trimmed value of a top-level `key:` line (the key may not be indented).
fn top_level_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let (name, value) = line.split_once(':')?;
    (name.trim() == key).then(|| value.trim())
}

/// Strips one pair of surrounding single or double quotes.
fn strip_quotes(value: &str) -> &str {
    for quote in ['\'', '"'] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}

/// Returns `true` when `name` is a portable skill name.
///
/// Documented deviation from Python's stricter regex: any non-empty name other than `..` that
/// uses only `[A-Za-z0-9._-]` is accepted.
fn is_portable_name(name: &str) -> bool {
    !name.is_empty()
        && name != ".."
        && name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
}

/// Installs one planned bundle, returning its destination unless it was skipped.
fn install_one(
    project: &Path,
    scan: &Path,
    name: &str,
    source: &Path,
    conflict: SkillConflict,
) -> Result<Option<PathBuf>> {
    let matches = existing_matches(scan, name)?;
    if !matches.is_empty() && conflict == SkillConflict::Skip {
        return Ok(None);
    }
    if matches.len() > 1 && conflict == SkillConflict::Replace {
        return Err(Error::validation(format!(
            "multiple existing directories declare skill '{name}'"
        )));
    }
    let copy_name = if !matches.is_empty() && conflict == SkillConflict::Append {
        next_copy_name(scan, name)?
    } else {
        name.to_owned()
    };
    let destination = if !matches.is_empty() && conflict == SkillConflict::Replace {
        matches[0].clone()
    } else {
        scan.join(&copy_name)
    };
    let Some(destination_name) = destination.file_name().map(PathBuf::from) else {
        return Err(Error::validation("skill destination has no file name"));
    };
    let target_location = scan.join(&destination_name);
    let source_root = source.canonicalize()?;
    if source_root == target_location {
        return Ok(None);
    }
    if source_root.starts_with(&target_location) || target_location.starts_with(&source_root) {
        return Err(Error::validation(
            "skill source and destination must not contain each other",
        ));
    }
    validate_tree(source, &source_root, &mut Vec::new())?;

    let stage = StageDir::create(project)?;
    let copied = stage.copied();
    copy_tree(source, &copied, &mut Vec::new())?;
    if copy_name != name {
        rewrite_skill_name(&copied, &copy_name)?;
    }
    let existed = std::fs::symlink_metadata(&destination).is_ok();
    if existed && conflict == SkillConflict::Append {
        return Err(Error::external(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "skill destination appeared during copy: {}",
                destination.display()
            ),
        )));
    }
    let backup = stage.previous().join(&destination_name);
    if existed {
        std::fs::rename(&destination, &backup)?;
    }
    if let Err(error) = std::fs::rename(&copied, &destination) {
        if existed {
            // Restoring the previous bundle is mandatory; if that also fails, report it.
            std::fs::rename(&backup, &destination)?;
        }
        return Err(Error::from(error));
    }
    Ok(Some(destination))
}

/// Finds the existing entries of `scan` that declare `name`.
///
/// Every immediate child is compared by directory name; a real directory containing `SKILL.md`
/// is additionally compared by its declared name. An unparsable declaration falls back to the
/// directory name instead of failing, mirroring Python's `_matches`.
fn existing_matches(scan: &Path, name: &str) -> Result<Vec<PathBuf>> {
    let needle = name.to_lowercase();
    let mut matches = Vec::new();
    for entry in read_dir_sorted(scan)? {
        let path = entry.path();
        let directory_name = entry.file_name().to_string_lossy().into_owned();
        let mut declared = directory_name.clone();
        if entry.file_type()?.is_dir() && path.join("SKILL.md").is_file() {
            if let Ok(parsed) = skill_name(&path) {
                declared = parsed;
            }
        }
        if directory_name.to_lowercase() == needle || declared.to_lowercase() == needle {
            matches.push(path);
        }
    }
    matches.sort();
    Ok(matches)
}

/// Returns the first free `name-N` copy name, truncating the base so the name stays ≤128 bytes.
fn next_copy_name(scan: &Path, name: &str) -> Result<String> {
    let mut index: u64 = 2;
    loop {
        let suffix = format!("-{index}");
        let base = truncate_bytes(name, 128 - suffix.len());
        let candidate = format!("{base}{suffix}");
        if existing_matches(scan, &candidate)?.is_empty() {
            return Ok(candidate);
        }
        index += 1;
    }
}

/// Truncates `name` to at most `limit` bytes without splitting a character.
fn truncate_bytes(name: &str, limit: usize) -> &str {
    if name.len() <= limit {
        return name;
    }
    let mut end = limit;
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name[..end]
}

/// Validates that `path` resolves inside the bundle root and contains only regular files.
///
/// Symlinks are resolved before the checks, so an escaping link is reported with its original
/// path; a dangling symlink propagates the underlying [`std::io::Error`].
fn validate_tree(path: &Path, root: &Path, ancestors: &mut Vec<PathBuf>) -> Result<()> {
    let resolved = path.canonicalize()?;
    if !resolved.starts_with(root) {
        return Err(Error::validation(format!(
            "skill symlink escapes its bundle: {}",
            path.display()
        )));
    }
    if resolved.is_dir() {
        if ancestors.contains(&resolved) {
            return Err(Error::validation(format!(
                "cyclic skill directory: {}",
                path.display()
            )));
        }
        ancestors.push(resolved.clone());
        for entry in read_dir_sorted(&resolved)? {
            validate_tree(&entry.path(), root, ancestors)?;
        }
        ancestors.pop();
    } else if !resolved.is_file() {
        return Err(Error::validation(format!(
            "skill contains a non-regular file: {}",
            path.display()
        )));
    }
    Ok(())
}

/// Copies `source` to `destination`, materializing symlinks and guarding against cycles.
fn copy_tree(source: &Path, destination: &Path, ancestors: &mut Vec<PathBuf>) -> Result<()> {
    let resolved = source.canonicalize()?;
    if resolved.is_dir() {
        if ancestors.contains(&resolved) {
            return Err(Error::validation(format!(
                "cyclic skill directory: {}",
                source.display()
            )));
        }
        std::fs::create_dir_all(destination)?;
        ancestors.push(resolved.clone());
        for entry in read_dir_sorted(&resolved)? {
            copy_tree(
                &entry.path(),
                &destination.join(entry.file_name()),
                ancestors,
            )?;
        }
        ancestors.pop();
    } else {
        // `fs::copy` copies the permission bits, so executable resources stay executable.
        std::fs::copy(&resolved, destination)?;
    }
    Ok(())
}

/// Rewrites the top-level `name:` entry of a copied bundle's front matter.
///
/// Every other front-matter line and the whole body are preserved byte for byte. When the front
/// matter has no `name:` line, one is inserted as its first line; a document without front matter
/// is left untouched (the directory name is the skill name).
fn rewrite_skill_name(bundle: &Path, name: &str) -> Result<()> {
    let document = bundle.join("SKILL.md");
    let text = std::fs::read_to_string(&document)?;
    let text = text
        .strip_prefix('\u{feff}')
        .map_or(text.as_str(), |stripped| stripped);
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if lines.first().is_none_or(|line| line.trim() != "---") {
        return Ok(());
    }
    let Some(end) = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, line)| line.trim() == "---")
        .map(|(index, _)| index)
    else {
        return Ok(());
    };
    let mut target = None;
    for (index, line) in lines.iter().enumerate().take(end).skip(1) {
        if top_level_value(line, "name").is_some() {
            target = Some(index);
        }
    }
    let mut rewritten = String::with_capacity(text.len() + name.len() + 8);
    for (index, line) in lines.iter().enumerate() {
        if target == Some(index) {
            rewritten.push_str("name: ");
            rewritten.push_str(name);
            rewritten.push_str(line_terminator(line));
        } else {
            if index == 1 && target.is_none() {
                rewritten.push_str("name: ");
                rewritten.push_str(name);
                rewritten.push('\n');
            }
            rewritten.push_str(line);
        }
    }
    std::fs::write(&document, rewritten)?;
    Ok(())
}

/// Returns the line terminator of `line` (`"\r\n"`, `"\n"`, or the empty string).
fn line_terminator(line: &str) -> &'static str {
    if line.ends_with("\r\n") {
        "\r\n"
    } else if line.ends_with('\n') {
        "\n"
    } else {
        ""
    }
}

/// Reads a directory and sorts its entries by path for deterministic traversal.
fn read_dir_sorted(path: &Path) -> Result<Vec<std::fs::DirEntry>> {
    let mut entries = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(std::fs::DirEntry::path);
    Ok(entries)
}

/// A staging directory inside the project, removed on every exit path.
struct StageDir {
    path: PathBuf,
}

impl StageDir {
    /// Creates the stage with its `new/` and `previous/` subdirectories.
    fn create(project: &Path) -> Result<Self> {
        let path = project.join(format!(".openjiuwen-skill-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path)?;
        let stage = Self { path };
        std::fs::create_dir(stage.copied())?;
        std::fs::create_dir(stage.previous())?;
        Ok(stage)
    }

    /// Directory receiving the copied bundle.
    fn copied(&self) -> PathBuf {
        self.path.join("new")
    }

    /// Directory receiving the replaced bundle.
    fn previous(&self) -> PathBuf {
        self.path.join("previous")
    }
}

impl Drop for StageDir {
    fn drop(&mut self) {
        // Best-effort cleanup: the stage never holds the only copy of user data after a
        // successful swap, and a failure here must not mask the install result.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{parse_skills_value, SkillMode, SkillSource};

    #[test]
    fn parse_skills_value_treats_absent_and_null_as_empty() {
        assert_eq!(
            parse_skills_value(None).expect("absent"),
            Vec::<SkillSource>::new()
        );
        assert_eq!(
            parse_skills_value(Some(&serde_json::Value::Null)).expect("null"),
            Vec::<SkillSource>::new()
        );
    }

    #[test]
    fn parse_skills_value_normalizes_arrays() {
        let value = json!(["/tmp/one", {"dir": "/tmp/two", "mode": "auto_list"}]);
        let sources = parse_skills_value(Some(&value)).expect("array");
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0], SkillSource::new("/tmp/one"));
        assert_eq!(sources[1].mode, SkillMode::AutoList);
    }

    #[test]
    fn parse_skills_value_rejects_non_arrays() {
        for value in [json!("dir"), json!(1), json!({"dir": "/x"})] {
            let error = parse_skills_value(Some(&value)).expect_err("non-array");
            assert_eq!(error.message().as_ref(), "skills must be an array");
        }
    }
}
