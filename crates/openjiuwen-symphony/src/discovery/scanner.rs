//! `SKILL.md` folder scanning into capability descriptors.
//!
//! Ports two Python halves of skill discovery:
//!
//! - `openjiuwen.symphony.discovery.models.scan_skill_directories` — the default,
//!   non-recursive mode: every immediate child directory of the root is a skill folder,
//!   probed for `SKILL.md`/`skill.md`/`Skill.md`, with no recursion below a skill
//!   folder. Names starting with `.` or `_` are skipped.
//! - `openjiuwen.symphony.shared.fingerprint.scanner` — the recursive mode: walk to a
//!   bounded depth looking for an exact `SKILL.md` match, parse its front matter, and
//!   derive one [`CapabilityDescriptor`] per skill folder.
//!
//! The scan never fails for a single bad skill: unreadable files, non-UTF-8 skill
//! files, missing front matter, malformed YAML, and unterminated front matter become
//! [`SkillDiagnostic`]s in the [`SkillScanResult`], and the scan continues. An
//! unterminated front matter is fatal for *that* skill (no descriptor). Only an
//! unusable root is an error, and it is
//! [`Error::Validation`](crate::error::Error::Validation) naming `root`.
//!
//! Everything is deterministic: directory entries are sorted case-insensitively (then
//! by exact name), skills are ordered by capability id, and duplicate ids keep the
//! first skill found.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::discovery::frontmatter::{
    io_values, normalize_capability_type, parse_front_matter, tag_values, FrontMatterOutcome,
    FrontMatterStatus,
};
use crate::error::{validation, Result};
use crate::model::{normalize_name_key, CapabilityDescriptor};

/// The capability type assigned to a discovered skill folder.
pub const SKILL_CAPABILITY_TYPE: &str = "skill";
/// The skill file names probed inside an immediate child folder, in priority order.
pub const SKILL_FILE_NAMES: [&str; 3] = ["SKILL.md", "skill.md", "Skill.md"];
/// The skill file name the recursive scanner matches exactly.
pub const RECURSIVE_SKILL_FILE: &str = "SKILL.md";
/// The recursion depth used when `recursive` is set without a `max_depth`.
pub const DEFAULT_MAX_DEPTH: usize = 4;
/// The longest description taken from a skill body when the front matter omits one.
pub const MAX_DESCRIPTION_CHARS: usize = 500;

/// Diagnostic code: the skill file has no front matter.
pub const DIAGNOSTIC_MISSING_FRONT_MATTER: &str = "missing_front_matter";
/// Diagnostic code: the front matter opens but never closes.
pub const DIAGNOSTIC_UNTERMINATED_FRONT_MATTER: &str = "unterminated_front_matter";
/// Diagnostic code: the front matter is not a usable YAML mapping.
pub const DIAGNOSTIC_MALFORMED_FRONT_MATTER: &str = "malformed_front_matter";
/// Diagnostic code: a skill file could not be read (I/O or encoding).
pub const DIAGNOSTIC_UNREADABLE_FILE: &str = "unreadable_file";
/// Diagnostic code: a directory could not be listed.
pub const DIAGNOSTIC_UNREADABLE_DIRECTORY: &str = "unreadable_directory";
/// Diagnostic code: a symlink was skipped instead of followed.
pub const DIAGNOSTIC_SYMLINK_SKIPPED: &str = "symlink_skipped";
/// Diagnostic code: two skill folders produced the same capability id.
pub const DIAGNOSTIC_DUPLICATE_SKILL: &str = "duplicate_skill";
/// Diagnostic code: the parsed skill has no usable identity.
pub const DIAGNOSTIC_INVALID_SKILL: &str = "invalid_skill";

/// How severe one [`SkillDiagnostic`] is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// The skill was kept; the problem is advisory.
    #[default]
    Warning,
    /// The skill was skipped.
    Error,
}

/// One problem found while scanning a skill root.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillDiagnostic {
    /// A stable machine-readable code (one of the `DIAGNOSTIC_*` constants).
    pub code: String,
    /// Whether the skill was kept ([`DiagnosticSeverity::Warning`]) or skipped.
    pub severity: DiagnosticSeverity,
    /// The file or directory the problem was found in.
    pub path: String,
    /// A human-readable explanation.
    pub message: String,
}

impl SkillDiagnostic {
    /// Builds a warning diagnostic.
    #[must_use]
    pub fn warning(
        code: impl Into<String>,
        path: impl AsRef<Path>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity: DiagnosticSeverity::Warning,
            path: path.as_ref().to_string_lossy().to_string(),
            message: message.into(),
        }
    }

    /// Builds an error diagnostic.
    #[must_use]
    pub fn error(
        code: impl Into<String>,
        path: impl AsRef<Path>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity: DiagnosticSeverity::Error,
            path: path.as_ref().to_string_lossy().to_string(),
            message: message.into(),
        }
    }

    /// Whether the diagnostic skipped the skill.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.severity == DiagnosticSeverity::Error
    }
}

/// How a [`SkillFolderScanner`] walks a root.
///
/// The default scans immediate child directories (Python's `scan_skill_directories`)
/// with source `"local"`. Set `recursive` (or use [`SkillScanConfig::recursive`]) to
/// walk nested folders down to `max_depth`, where a folder holding `SKILL.md` is a
/// skill and is not descended into.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillScanConfig {
    /// The deepest directory level to inspect in recursive mode; `None` uses
    /// [`DEFAULT_MAX_DEPTH`]. Ignored by the default immediate-child scan.
    pub max_depth: Option<usize>,
    /// The `source` stamped on every discovered descriptor and record.
    pub source: String,
    /// Skill folder names, capability ids, or display names to drop after scanning.
    /// Matching is case-insensitive.
    pub disabled_skills: BTreeSet<String>,
    /// Whether to walk nested folders instead of only immediate children.
    pub recursive: bool,
}

impl Default for SkillScanConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            source: "local".to_string(),
            disabled_skills: BTreeSet::new(),
            recursive: false,
        }
    }
}

impl SkillScanConfig {
    /// Builds a recursive config with an optional depth bound.
    #[must_use]
    pub fn recursive(max_depth: Option<usize>) -> Self {
        Self {
            max_depth,
            recursive: true,
            ..Self::default()
        }
    }

    /// Sets the descriptor source.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Sets the recursion depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Replaces the disabled skill names.
    #[must_use]
    pub fn with_disabled_skills(
        mut self,
        skills: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.disabled_skills = skills.into_iter().map(Into::into).collect();
        self
    }

    /// The depth actually used by a recursive scan.
    #[must_use]
    pub fn effective_max_depth(&self) -> usize {
        self.max_depth.unwrap_or(DEFAULT_MAX_DEPTH)
    }

    /// Whether `folder_name` or `descriptor` matches a disabled skill name.
    #[must_use]
    pub fn is_disabled(&self, folder_name: &str, descriptor: &CapabilityDescriptor) -> bool {
        if self.disabled_skills.is_empty() {
            return false;
        }
        let candidates = [
            folder_name,
            descriptor.capability_id.as_str(),
            descriptor.name.as_str(),
        ];
        self.disabled_skills.iter().any(|disabled| {
            let disabled = disabled.trim();
            !disabled.is_empty()
                && candidates
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(disabled))
        })
    }
}

/// One discovered skill: its descriptor and its markdown body.
#[derive(Debug, Clone, PartialEq)]
pub struct ScannedSkill {
    /// The capability descriptor derived from the skill folder.
    pub descriptor: CapabilityDescriptor,
    /// The markdown body after the front matter (used as retrieval content).
    pub body: String,
}

/// Everything one scan produced.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillDiscovery {
    /// The discovered skills, sorted by capability id.
    pub skills: Vec<ScannedSkill>,
    /// The problems found, in traversal order.
    pub diagnostics: Vec<SkillDiagnostic>,
    /// The configured source stamped on the descriptors.
    pub source: String,
}

/// The serializable outcome of a scan: descriptors plus diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillScanResult {
    /// The discovered capability descriptors, sorted by capability id.
    pub capabilities: Vec<CapabilityDescriptor>,
    /// The problems found while scanning.
    pub diagnostics: Vec<SkillDiagnostic>,
    /// The configured source.
    pub source: String,
}

impl SkillScanResult {
    /// The diagnostic codes that skipped a skill.
    #[must_use]
    pub fn error_codes(&self) -> Vec<&str> {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_error())
            .map(|diagnostic| diagnostic.code.as_str())
            .collect()
    }

    /// The capability ids, in sorted order.
    #[must_use]
    pub fn capability_ids(&self) -> Vec<&str> {
        self.capabilities
            .iter()
            .map(|descriptor| descriptor.capability_id.as_str())
            .collect()
    }
}

/// Scans a skill root for `SKILL.md` folders.
///
/// ```
/// # use openjiuwen_symphony::discovery::SkillFolderScanner;
/// # async fn example() -> openjiuwen_symphony::Result<()> {
/// let scanner = SkillFolderScanner::new("skills");
/// let capabilities = scanner.scan().await?;
/// # let _ = capabilities;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SkillFolderScanner {
    root: PathBuf,
    config: SkillScanConfig,
}

impl SkillFolderScanner {
    /// Builds a scanner over `root` with the default (immediate-child) config.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            config: SkillScanConfig::default(),
        }
    }

    /// Builds a scanner over `root` with an explicit config.
    #[must_use]
    pub fn with_config(root: impl Into<PathBuf>, config: SkillScanConfig) -> Self {
        Self {
            root: root.into(),
            config,
        }
    }

    /// The scan root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The scan config.
    #[must_use]
    pub fn config(&self) -> &SkillScanConfig {
        &self.config
    }

    /// Scans the root and returns the discovered descriptors, sorted by capability id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `root` when the root does not exist or is
    /// not a directory.
    pub async fn scan(&self) -> Result<Vec<CapabilityDescriptor>> {
        Ok(self
            .discover()
            .await?
            .skills
            .into_iter()
            .map(|skill| skill.descriptor)
            .collect())
    }

    /// Scans the root and returns the descriptors together with their markdown bodies.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `root` when the root does not exist or is
    /// not a directory.
    pub async fn scan_skills(&self) -> Result<Vec<ScannedSkill>> {
        Ok(self.discover().await?.skills)
    }

    /// Scans the root and returns the serializable [`SkillScanResult`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `root` when the root does not exist or is
    /// not a directory.
    pub async fn scan_result(&self) -> Result<SkillScanResult> {
        let discovery = self.discover().await?;
        Ok(SkillScanResult {
            capabilities: discovery
                .skills
                .into_iter()
                .map(|skill| skill.descriptor)
                .collect(),
            diagnostics: discovery.diagnostics,
            source: discovery.source,
        })
    }

    /// Runs the scan once and returns skills, diagnostics, and the configured source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `root` when the root does not exist or is
    /// not a directory.
    pub async fn discover(&self) -> Result<SkillDiscovery> {
        let metadata = tokio::fs::metadata(&self.root).await.map_err(|error| {
            validation(
                "root",
                format!(
                    "`{}` is not a readable skill root: {error}",
                    self.root.display()
                ),
            )
        })?;
        if !metadata.is_dir() {
            return Err(validation(
                "root",
                format!("`{}` is not a directory", self.root.display()),
            ));
        }

        let mut context = ScanContext::new(&self.root, &self.config);
        if self.config.recursive {
            context.walk().await;
        } else {
            context.scan_children().await;
        }
        context.skills.sort_by(|left, right| {
            left.descriptor
                .capability_id
                .cmp(&right.descriptor.capability_id)
        });

        Ok(SkillDiscovery {
            skills: context.skills,
            diagnostics: context.diagnostics,
            source: self.config.source.clone(),
        })
    }
}

/// The mutable state of one scan.
struct ScanContext<'a> {
    /// The scan root, used to derive relative entry points.
    root: &'a Path,
    /// The scan config.
    config: &'a SkillScanConfig,
    /// The skills discovered so far.
    skills: Vec<ScannedSkill>,
    /// The diagnostics accumulated so far.
    diagnostics: Vec<SkillDiagnostic>,
    /// The capability ids already claimed, so a duplicate keeps the first skill.
    seen_ids: BTreeSet<String>,
}

impl<'a> ScanContext<'a> {
    /// Builds the empty scan state.
    fn new(root: &'a Path, config: &'a SkillScanConfig) -> Self {
        Self {
            root,
            config,
            skills: Vec::new(),
            diagnostics: Vec::new(),
            seen_ids: BTreeSet::new(),
        }
    }

    /// Scans the immediate child directories of the root.
    async fn scan_children(&mut self) {
        let root = self.root.to_path_buf();
        for (directory, name) in self.subdirectories(&root).await {
            if is_skipped_name(&name) {
                continue;
            }
            if let Some(skill_file) = probe_skill_file(&directory).await {
                self.load(&directory, &skill_file, &name).await;
            }
        }
    }

    /// Walks the root breadth-first, bounded by `max_depth`.
    async fn walk(&mut self) {
        let max_depth = self.config.effective_max_depth();
        let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::new();
        queue.push_back((self.root.to_path_buf(), 0));
        while let Some((directory, depth)) = queue.pop_front() {
            let skill_file = directory.join(RECURSIVE_SKILL_FILE);
            if is_file(&skill_file).await {
                let name = directory
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_default();
                self.load(&directory, &skill_file, &name).await;
                continue;
            }
            if depth >= max_depth {
                continue;
            }
            for (child, name) in self.subdirectories(&directory).await {
                if is_skipped_name(&name) {
                    continue;
                }
                queue.push_back((child, depth + 1));
            }
        }
    }

    /// Lists the sorted directory children of `directory`, recording read failures.
    async fn subdirectories(&mut self, directory: &Path) -> Vec<(PathBuf, String)> {
        let mut entries = match tokio::fs::read_dir(directory).await {
            Ok(entries) => entries,
            Err(error) => {
                self.diagnostics.push(SkillDiagnostic::error(
                    DIAGNOSTIC_UNREADABLE_DIRECTORY,
                    directory,
                    format!("cannot list directory: {error}"),
                ));
                return Vec::new();
            }
        };

        let mut children: Vec<(PathBuf, String)> = Vec::new();
        loop {
            let entry = match entries.next_entry().await {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(error) => {
                    self.diagnostics.push(SkillDiagnostic::error(
                        DIAGNOSTIC_UNREADABLE_DIRECTORY,
                        directory,
                        format!("cannot list directory: {error}"),
                    ));
                    break;
                }
            };
            let name = entry.file_name().to_string_lossy().to_string();
            let file_type = match entry.file_type().await {
                Ok(file_type) => file_type,
                Err(error) => {
                    self.diagnostics.push(SkillDiagnostic::warning(
                        DIAGNOSTIC_UNREADABLE_FILE,
                        entry.path(),
                        format!("cannot inspect `{name}`: {error}"),
                    ));
                    continue;
                }
            };
            if file_type.is_symlink() {
                self.diagnostics.push(SkillDiagnostic::warning(
                    DIAGNOSTIC_SYMLINK_SKIPPED,
                    entry.path(),
                    format!("skipping symlink `{name}`"),
                ));
                continue;
            }
            if !file_type.is_dir() {
                continue;
            }
            children.push((entry.path(), name));
        }

        children.sort_by(|left, right| {
            left.1
                .to_lowercase()
                .cmp(&right.1.to_lowercase())
                .then_with(|| left.1.cmp(&right.1))
        });
        children
    }

    /// Reads and converts one skill file, recording every problem it finds.
    async fn load(&mut self, directory: &Path, skill_file: &Path, folder_name: &str) {
        let bytes = match tokio::fs::read(skill_file).await {
            Ok(bytes) => bytes,
            Err(error) => {
                self.diagnostics.push(SkillDiagnostic::error(
                    DIAGNOSTIC_UNREADABLE_FILE,
                    skill_file,
                    format!("cannot read skill file: {error}"),
                ));
                return;
            }
        };
        let content_hash = sha256_hex(&bytes);
        let Ok(text) = String::from_utf8(bytes) else {
            self.diagnostics.push(SkillDiagnostic::error(
                DIAGNOSTIC_UNREADABLE_FILE,
                skill_file,
                "skill file is not valid UTF-8",
            ));
            return;
        };

        let outcome = parse_front_matter(&text);
        match outcome.status {
            FrontMatterStatus::Unterminated => {
                self.diagnostics.push(SkillDiagnostic::error(
                    DIAGNOSTIC_UNTERMINATED_FRONT_MATTER,
                    skill_file,
                    outcome
                        .error
                        .clone()
                        .unwrap_or_else(|| "unterminated front matter".to_string()),
                ));
                return;
            }
            FrontMatterStatus::Missing => {
                self.diagnostics.push(SkillDiagnostic::warning(
                    DIAGNOSTIC_MISSING_FRONT_MATTER,
                    skill_file,
                    format!(
                        "`{folder_name}` has no front matter; deriving its descriptor from the body"
                    ),
                ));
            }
            FrontMatterStatus::Ok => {
                if let Some(error) = &outcome.error {
                    self.diagnostics.push(SkillDiagnostic::warning(
                        DIAGNOSTIC_MALFORMED_FRONT_MATTER,
                        skill_file,
                        error.clone(),
                    ));
                }
            }
        }

        let entrypoint = relative_entrypoint(self.root, skill_file);
        let source_root = self.root.to_string_lossy().to_string();
        let descriptor = build_descriptor(
            self.config,
            &outcome,
            folder_name,
            &content_hash,
            &entrypoint,
            &source_root,
        );

        if let Err(error) = descriptor.validate() {
            self.diagnostics.push(SkillDiagnostic::error(
                DIAGNOSTIC_INVALID_SKILL,
                directory,
                error.to_string(),
            ));
            return;
        }
        if self.config.is_disabled(folder_name, &descriptor) {
            return;
        }
        if !self.seen_ids.insert(descriptor.capability_id.clone()) {
            self.diagnostics.push(SkillDiagnostic::warning(
                DIAGNOSTIC_DUPLICATE_SKILL,
                skill_file,
                format!(
                    "duplicate skill id `{}`; keeping the first one found",
                    descriptor.capability_id
                ),
            ));
            return;
        }

        self.skills.push(ScannedSkill {
            descriptor,
            body: outcome.body.clone(),
        });
    }
}

/// Builds one descriptor from a parsed skill file.
fn build_descriptor(
    config: &SkillScanConfig,
    outcome: &FrontMatterOutcome,
    folder_name: &str,
    content_hash: &str,
    entrypoint: &str,
    source_root: &str,
) -> CapabilityDescriptor {
    let capability_id = outcome
        .first_str(&["capability_id", "id"])
        .map_or_else(|| folder_slug(folder_name), ToString::to_string);
    let name = outcome
        .get_str("name")
        .map_or_else(|| folder_name.trim(), |name| name)
        .to_string();
    let name = if name.is_empty() {
        capability_id.clone()
    } else {
        name
    };
    let description = outcome.get_str("description").map_or_else(
        || first_paragraph(&outcome.body).unwrap_or_else(|| name.clone()),
        ToString::to_string,
    );
    let capability_type = normalize_capability_type(
        outcome
            .get_str("capability_type")
            .unwrap_or(SKILL_CAPABILITY_TYPE),
    );
    let inputs = outcome.get("inputs").map(io_values).unwrap_or_default();
    let outputs = outcome.get("outputs").map(io_values).unwrap_or_default();
    let tags = outcome.get("tags").map(tag_values).unwrap_or_default();
    let aliases = outcome.get("aliases").map(tag_values).unwrap_or_default();

    let mut descriptor = CapabilityDescriptor::new(capability_type, capability_id, name)
        .with_description(description)
        .with_source(config.source.clone())
        .with_available(outcome.get_bool("available").unwrap_or(true))
        .with_tags(tags)
        .with_inputs(inputs)
        .with_outputs(outputs)
        .with_classification(outcome.get_str("category").unwrap_or_default())
        .with_metadata("entrypoint", Value::String(entrypoint.to_string()))
        .with_metadata(
            "version",
            Value::String(outcome.get_str("version").unwrap_or_default().to_string()),
        )
        .with_metadata(
            "author",
            Value::String(outcome.get_str("author").unwrap_or_default().to_string()),
        )
        .with_metadata("source_root", Value::String(source_root.to_string()));
    descriptor.content_hash = content_hash.to_string();
    if !aliases.is_empty() {
        descriptor.metadata.insert(
            "aliases".to_string(),
            Value::Array(aliases.into_iter().map(Value::String).collect()),
        );
    }
    descriptor
}

/// Derives the fallback capability id from a folder name: a hyphenated slug.
fn folder_slug(folder_name: &str) -> String {
    let slug = normalize_name_key(folder_name).replace('_', "-");
    if slug.is_empty() {
        SKILL_CAPABILITY_TYPE.to_string()
    } else {
        slug
    }
}

/// Extracts the first paragraph of a markdown body, collapsed and length-capped.
fn first_paragraph(body: &str) -> Option<String> {
    let mut collected: Vec<&str> = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if collected.is_empty() {
                continue;
            }
            break;
        }
        collected.push(trimmed);
    }
    if collected.is_empty() {
        return None;
    }
    let paragraph = collected.join(" ");
    let paragraph = paragraph.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(truncate_chars(&paragraph, MAX_DESCRIPTION_CHARS))
}

/// Collapses consecutive whitespace and caps the text at `max` characters with `…`.
fn truncate_chars(text: &str, max: usize) -> String {
    let characters: Vec<char> = text.chars().collect();
    if characters.len() <= max {
        return text.to_string();
    }
    let kept: String = characters.iter().take(max.saturating_sub(1)).collect();
    format!("{}…", kept.trim_end())
}

/// Returns the path of `skill_file` relative to `root`, with `/` separators.
fn relative_entrypoint(root: &Path, skill_file: &Path) -> String {
    skill_file
        .strip_prefix(root)
        .unwrap_or(skill_file)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Whether a directory name is skipped by both scan modes.
fn is_skipped_name(name: &str) -> bool {
    name.starts_with('.') || name.starts_with('_')
}

/// Whether `path` exists and is a regular file.
async fn is_file(path: &Path) -> bool {
    tokio::fs::metadata(path)
        .await
        .is_ok_and(|metadata| metadata.is_file())
}

/// Probes the [`SKILL_FILE_NAMES`] of an immediate child folder in priority order.
async fn probe_skill_file(directory: &Path) -> Option<PathBuf> {
    for name in SKILL_FILE_NAMES {
        let candidate = directory.join(name);
        if is_file(&candidate).await {
            return Some(candidate);
        }
    }
    None
}

/// The SHA-256 hex digest of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn folder_slugs_are_hyphenated_and_never_blank() {
        assert_eq!(folder_slug("Pdf Export"), "pdf-export");
        assert_eq!(folder_slug("pdf_export"), "pdf-export");
        assert_eq!(folder_slug("!!!"), SKILL_CAPABILITY_TYPE);
    }

    #[test]
    fn first_paragraph_skips_leading_blanks_and_stops_at_a_blank_line() {
        assert_eq!(
            first_paragraph("\n\n First line\nsecond   line\n\nSecond paragraph"),
            Some("First line second line".to_string())
        );
        assert_eq!(first_paragraph("\n \n"), None);
    }

    #[test]
    fn long_descriptions_are_capped_with_an_ellipsis() {
        let text = "a".repeat(MAX_DESCRIPTION_CHARS + 20);
        let capped = truncate_chars(&text, MAX_DESCRIPTION_CHARS);
        assert_eq!(capped.chars().count(), MAX_DESCRIPTION_CHARS);
        assert!(capped.ends_with('…'));
        assert_eq!(truncate_chars("short", 10), "short");
    }

    #[test]
    fn relative_entrypoints_use_forward_slashes() {
        let root = Path::new("/tmp/skills");
        assert_eq!(
            relative_entrypoint(root, Path::new("/tmp/skills/alpha/SKILL.md")),
            "alpha/SKILL.md"
        );
    }

    #[test]
    fn disabled_matching_is_case_insensitive() {
        let config = SkillScanConfig::default().with_disabled_skills(["Alpha"]);
        let descriptor = CapabilityDescriptor::new("skill", "alpha", "Alpha");
        assert!(config.is_disabled("alpha", &descriptor));
        let other = CapabilityDescriptor::new("skill", "beta", "Beta");
        assert!(!config.is_disabled("beta", &other));
        assert!(!config.is_disabled("unrelated", &other));
        assert!(!SkillScanConfig::default().is_disabled("alpha", &descriptor));
    }
}
