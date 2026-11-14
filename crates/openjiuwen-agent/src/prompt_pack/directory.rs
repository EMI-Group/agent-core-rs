//! Python-style prompt directories: load `prompts/` trees into agent sections.
//!
//! The Python SDK ships prompts as files on disk in two shapes:
//!
//! - **language directories** — `prompts/{cn,en}/<name>.md`, where the file
//!   stem is the template key and `zh` aliases `cn`;
//! - **suffixed files** — `<name>.<lang>.md` anywhere below the root, including
//!   the evolved `prompts/system/`, `prompts/identity/`, and
//!   `prompts/tool/<domain>/` trees.
//!
//! [`PromptDirectory::load`] reads either shape (and both at once) into ordered
//! sections, each carrying a language map. The runtime unit is one section: its
//! name is the file stem minus the `.<lang>` suffix, its priority comes from the
//! front matter (default [`DEFAULT_PROMPT_PRIORITY`]), and its text is selected
//! with the fallback chain *requested → `en` → `cn` → first*.
//!
//! # Front matter
//!
//! A file may start with a YAML front matter block; see
//! [`read_frontmatter`] for the fields and the
//! `baseline_sha256` evolution rule. Bodies are kept byte-faithfully, so
//! `{{var}}` placeholders stay untouched until [`PromptFile::render`] is called.
//!
//! # Ordering and duplicates
//!
//! Sections are ordered by priority ascending, then name (byte order). Files
//! are visited in relative-path order. Two files that resolve to the same
//! section name *and* the same language are duplicates: the higher priority
//! wins, and on a tie the lexicographically later relative path wins. Files with
//! the same name but different languages are variants of one section, not
//! duplicates.
//!
//! # Applying a directory
//!
//! [`PromptDirectory::section_specs`] and [`PromptDirectory::prompt_sections`]
//! expose the ordered sections, and [`PromptDirectory::pack`] (or
//! [`PromptPack::from_prompt_dir`]) builds the [`PromptPack`] value that
//! [`PromptPack::apply_to`] installs on a [`crate::ReActAgent`] through the
//! existing [`PromptSectionsRail`](super::PromptSectionsRail).
use super::pack::{PromptPack, PromptSpec, SectionSpec};
use crate::prompt::PromptSection;
use front_matter::read_frontmatter_from;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{PromptTemplate, TemplateContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::fs;

mod front_matter;
mod sha256;
#[cfg(test)]
mod tests;

pub use front_matter::{
    body_sha256, is_evolved, read_frontmatter, PromptFileMeta, DEFAULT_PROMPT_PRIORITY,
};

/// Languages with a documented prompt directory; `zh` is an alias of `cn`.
pub const SUPPORTED_LANGUAGES: [&str; 2] = ["cn", "en"];

/// Fallback order used when the requested language has no file.
pub const FALLBACK_LANGUAGES: [&str; 2] = ["en", "cn"];

/// Normalizes a language token: trims, lowercases, and maps `zh` to `cn`.
#[must_use]
pub fn normalize_language(language: &str) -> String {
    let raw = language.trim().to_ascii_lowercase();
    if raw == "zh" {
        return "cn".to_owned();
    }
    raw
}

/// One prompt file: a section variant in one language.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptFile {
    /// Absolute (root-joined) path the file was read from.
    pub path: PathBuf,
    /// Section name: the file stem minus a `.<lang>` suffix.
    pub name: String,
    /// Normalized language token (`cn`, `en`, ...), or `None` when unqualified.
    pub language: Option<String>,
    /// Priority from the front matter, or [`DEFAULT_PROMPT_PRIORITY`].
    pub priority: i32,
    /// The body as read (byte-faithful after newline normalization).
    pub body: String,
    /// The parsed front matter.
    pub meta: PromptFileMeta,
}

impl Default for PromptFile {
    /// An unnamed, empty, unqualified file at [`DEFAULT_PROMPT_PRIORITY`].
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            name: String::new(),
            language: None,
            priority: DEFAULT_PROMPT_PRIORITY,
            body: String::new(),
            meta: PromptFileMeta::default(),
        }
    }
}

impl PromptFile {
    /// The SHA-256 of the file's body, as recorded by `baseline_sha256`.
    #[must_use]
    pub fn body_sha256(&self) -> String {
        body_sha256(&self.body)
    }

    /// Whether the body diverged from the front matter baseline.
    #[must_use]
    pub fn is_evolved(&self) -> bool {
        is_evolved(&self.meta, &self.body)
    }

    /// Renders the body's `{{var}}` placeholders with `values`.
    ///
    /// Rendering is single-pass: unknown placeholders stay verbatim and a
    /// substituted value is never re-scanned, matching
    /// [`PromptTemplate::format`]. The stored body is not changed.
    #[must_use]
    pub fn render(&self, values: &BTreeMap<String, Value>) -> String {
        let template = PromptTemplate::new(self.body.as_str());
        match template.format(values).content {
            TemplateContent::Text(text) => text,
            // A text template never renders to a message list.
            TemplateContent::Messages(_) => self.body.clone(),
        }
    }
}

/// One ordered section and its language variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DirectorySection {
    /// Section name (the file stem minus a `.<lang>` suffix).
    pub name: String,
    /// Lowest priority among the variants; [`DirectorySection::spec`] uses the
    /// selected variant's own priority.
    pub priority: i32,
    /// Variants sorted by relative path.
    pub variants: Vec<PromptFile>,
}

impl Default for DirectorySection {
    /// An unnamed, empty section at [`DEFAULT_PROMPT_PRIORITY`].
    fn default() -> Self {
        Self {
            name: String::new(),
            priority: DEFAULT_PROMPT_PRIORITY,
            variants: Vec::new(),
        }
    }
}

impl DirectorySection {
    /// The variant serving `language`, following *requested → en → cn → first*.
    #[must_use]
    pub fn variant(&self, language: &str) -> Option<&PromptFile> {
        let requested = normalize_language(language);
        if !requested.is_empty() {
            if let Some(variant) = self.find(&requested) {
                return Some(variant);
            }
        }
        for fallback in FALLBACK_LANGUAGES {
            if let Some(variant) = self.find(fallback) {
                return Some(variant);
            }
        }
        self.variants.first()
    }

    /// The selected variant's raw body, or `None` for a section with no variant.
    #[must_use]
    pub fn text(&self, language: &str) -> Option<&str> {
        self.variant(language).map(|file| file.body.as_str())
    }

    /// The section as a pack [`SectionSpec`], with the selected body trimmed.
    #[must_use]
    pub fn spec(&self, language: &str) -> Option<SectionSpec> {
        self.variant(language).map(|file| {
            SectionSpec::new(self.name.clone(), file.body.trim()).with_priority(file.priority)
        })
    }

    /// The section as a kernel [`PromptSection`], categorised by its `kind`.
    #[must_use]
    pub fn prompt_section(&self, language: &str) -> Option<PromptSection> {
        self.variant(language).map(|file| {
            let section = PromptSection::new(self.name.clone(), file.body.trim(), file.priority);
            match &file.meta.kind {
                Some(kind) => section.with_category(kind.clone()),
                None => section,
            }
        })
    }

    /// The language tokens this section has a variant for, in path order.
    #[must_use]
    pub fn languages(&self) -> Vec<&str> {
        let mut languages: Vec<&str> = Vec::new();
        for file in &self.variants {
            if let Some(language) = file.language.as_deref() {
                if !languages.contains(&language) {
                    languages.push(language);
                }
            }
        }
        languages
    }

    /// The first `kind` any variant declares.
    #[must_use]
    pub fn kind(&self) -> Option<&str> {
        self.variants
            .iter()
            .find_map(|file| file.meta.kind.as_deref())
    }

    /// Whether any variant's body has evolved past its baseline.
    #[must_use]
    pub fn is_evolved(&self) -> bool {
        self.variants.iter().any(PromptFile::is_evolved)
    }

    /// The variant whose language is exactly `language`.
    fn find(&self, language: &str) -> Option<&PromptFile> {
        self.variants
            .iter()
            .find(|file| file.language.as_deref() == Some(language))
    }
}

/// A loaded prompt directory: ordered sections with their language variants.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PromptDirectory {
    /// The directory that was scanned.
    pub root: PathBuf,
    /// Sections ordered by priority ascending, then name.
    pub sections: Vec<DirectorySection>,
}

impl PromptDirectory {
    /// Scans `path` recursively and parses every Markdown prompt file.
    ///
    /// Hidden entries (names starting with `.`), symbolic links, and files whose
    /// names are not valid UTF-8 are skipped. A language is taken from the front
    /// matter when present, else from a `.<lang>` suffix, else from the first
    /// path component naming a language (`cn`, `en`, or its `zh` alias).
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when `path` is missing or a file/directory cannot
    /// be read, [`Error::Validation`] when `path` is not a directory, and the
    /// [`read_frontmatter`] errors (malformed YAML, non-mapping root, bad
    /// `priority`) for an invalid file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).await.map_err(|error| {
            Error::sys_op(format!(
                "cannot read prompt directory {}: {error}",
                path.display()
            ))
        })?;
        if !metadata.is_dir() {
            return Err(Error::validation(format!(
                "prompt directory {} is not a directory",
                path.display()
            )));
        }

        let root = path.to_path_buf();
        let mut files = Vec::new();
        for (file_path, dirs, name) in collect_files(&root).await? {
            files.push(load_file(&file_path, &dirs, &name).await?);
        }

        Ok(Self {
            root,
            sections: group_sections(files),
        })
    }

    /// The section named `name`.
    #[must_use]
    pub fn section(&self, name: &str) -> Option<&DirectorySection> {
        self.sections.iter().find(|section| section.name == name)
    }

    /// The number of sections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Whether the directory declared no sections.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// The ordered sections as pack [`SectionSpec`]s, resolved for `language`.
    #[must_use]
    pub fn section_specs(&self, language: &str) -> Vec<SectionSpec> {
        self.sections
            .iter()
            .filter_map(|section| section.spec(language))
            .collect()
    }

    /// The ordered sections as kernel [`PromptSection`]s, resolved for
    /// `language`.
    #[must_use]
    pub fn prompt_sections(&self, language: &str) -> Vec<PromptSection> {
        self.sections
            .iter()
            .filter_map(|section| section.prompt_section(language))
            .collect()
    }

    /// Renders the non-empty sections for `language`, joined by blank lines.
    #[must_use]
    pub fn render(&self, language: &str) -> String {
        self.section_specs(language)
            .iter()
            .filter(|section| !section.content.is_empty())
            .map(|section| format!("## {}\n{}", section.name, section.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Builds the [`PromptPack`] that installs this directory's sections.
    ///
    /// The pack carries no template, name, description, tools, or rails, so
    /// applying it only registers a [`PromptSectionsRail`](super::PromptSectionsRail) over the sections.
    #[must_use]
    pub fn pack(&self, id: impl Into<String>, language: &str) -> PromptPack {
        PromptPack {
            id: id.into(),
            prompts: PromptSpec {
                template: Vec::new(),
                sections: self.section_specs(language),
            },
            ..PromptPack::default()
        }
    }
}

impl PromptPack {
    /// Loads a Python-style prompt directory and returns the pack that installs
    /// its sections.
    ///
    /// The pack id is the directory's file name (or `"prompt_pack"` when the
    /// path has none). The pack declares no identity name or description, so
    /// applying it leaves the agent's prompt template alone.
    ///
    /// # Errors
    ///
    /// Returns the [`PromptDirectory::load`] errors.
    pub async fn from_prompt_dir(path: impl AsRef<Path>, language: &str) -> Result<Self> {
        let path = path.as_ref();
        let directory = PromptDirectory::load(path).await?;
        Ok(Self::from_prompt_directory(
            &directory,
            directory_id(path),
            language,
        ))
    }

    /// Builds a pack over an already-loaded directory.
    #[must_use]
    pub fn from_prompt_directory(
        directory: &PromptDirectory,
        id: impl Into<String>,
        language: &str,
    ) -> Self {
        directory.pack(id, language)
    }
}

/// The pack id derived from a directory path.
fn directory_id(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("prompt_pack")
        .to_owned()
}

/// Collects every Markdown file below `root`, sorted by relative path.
///
/// Each entry is `(absolute path, relative directory components, file name)`.
async fn collect_files(root: &Path) -> Result<Vec<(PathBuf, Vec<String>, String)>> {
    let mut pending = vec![(root.to_path_buf(), Vec::<String>::new())];
    let mut found = Vec::new();
    while let Some((dir, dirs)) = pending.pop() {
        let mut entries = fs::read_dir(&dir).await.map_err(|error| {
            Error::sys_op(format!(
                "cannot read prompt directory {}: {error}",
                dir.display()
            ))
        })?;
        let mut children: Vec<(String, bool)> = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(|error| {
            Error::sys_op(format!(
                "cannot read prompt directory {}: {error}",
                dir.display()
            ))
        })? {
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                tracing::warn!(
                    target: "openjiuwen_agent::prompt_pack",
                    directory = %dir.display(),
                    "skipping a prompt entry whose name is not valid UTF-8"
                );
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let file_type = entry.file_type().await.map_err(|error| {
                Error::sys_op(format!("cannot inspect prompt entry {name}: {error}"))
            })?;
            if file_type.is_symlink() {
                tracing::warn!(
                    target: "openjiuwen_agent::prompt_pack",
                    path = %entry.path().display(),
                    "skipping a symbolic link in a prompt directory"
                );
                continue;
            }
            if file_type.is_dir() {
                children.push((name, true));
            } else if file_type.is_file() && is_markdown(&name) {
                children.push((name, false));
            }
        }
        children.sort_by(|left, right| left.0.cmp(&right.0));
        for (name, is_dir) in children {
            let path = dir.join(&name);
            if is_dir {
                let mut child_dirs = dirs.clone();
                child_dirs.push(name);
                pending.push((path, child_dirs));
            } else {
                found.push((path, dirs.clone(), name));
            }
        }
    }
    found.sort_by_key(relative_key);
    Ok(found)
}

/// The sort key of a collected file: its relative directory path plus name.
fn relative_key(entry: &(PathBuf, Vec<String>, String)) -> String {
    let (_, dirs, name) = entry;
    if dirs.is_empty() {
        name.clone()
    } else {
        format!("{}/{}", dirs.join("/"), name)
    }
}

/// Whether `name` ends with a Markdown extension (case-insensitive).
fn is_markdown(name: &str) -> bool {
    Path::new(name)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
}

/// Reads and parses one prompt file.
async fn load_file(path: &Path, dirs: &[String], file_name: &str) -> Result<PromptFile> {
    let text = fs::read_to_string(path).await.map_err(|error| {
        Error::sys_op(format!(
            "cannot read prompt file {}: {error}",
            path.display()
        ))
    })?;
    let (meta, body) = read_frontmatter_from(&text, &path.display().to_string())
        .map_err(|error| Error::validation(format!("{}: {error}", path.display())))?;

    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(file_name);
    let (name, suffix_language) = split_language_suffix(stem);
    let language = declared_language(&meta)
        .or(suffix_language)
        .or_else(|| directory_language(dirs));
    let priority = meta.priority.unwrap_or(DEFAULT_PROMPT_PRIORITY);

    Ok(PromptFile {
        path: path.to_path_buf(),
        name,
        language,
        priority,
        body,
        meta,
    })
}

/// Splits a file stem into its section name and an optional language suffix.
///
/// A suffix counts as a language when the final dot segment is exactly two ASCII
/// letters (`leader_policy.en`), so version-like suffixes (`notes.v2`) stay part
/// of the name.
fn split_language_suffix(stem: &str) -> (String, Option<String>) {
    let Some((base, suffix)) = stem.rsplit_once('.') else {
        return (stem.to_owned(), None);
    };
    let is_language =
        !base.is_empty() && suffix.len() == 2 && suffix.chars().all(|c| c.is_ascii_alphabetic());
    if is_language {
        (base.to_owned(), Some(normalize_language(suffix)))
    } else {
        (stem.to_owned(), None)
    }
}

/// The front-matter language, normalized, ignoring a blank declaration.
fn declared_language(meta: &PromptFileMeta) -> Option<String> {
    meta.language
        .as_deref()
        .map(normalize_language)
        .filter(|language| !language.is_empty())
}

/// The language named by the first relative path component, when there is one.
fn directory_language(dirs: &[String]) -> Option<String> {
    dirs.iter().find_map(|component| {
        let normalized = normalize_language(component);
        SUPPORTED_LANGUAGES
            .contains(&normalized.as_str())
            .then_some(normalized)
    })
}

/// Groups files into ordered sections, resolving duplicate name/language pairs.
fn group_sections(files: Vec<PromptFile>) -> Vec<DirectorySection> {
    let mut sections: Vec<DirectorySection> = Vec::new();
    for file in files {
        match sections
            .iter_mut()
            .find(|section| section.name == file.name)
        {
            Some(section) => insert_variant(section, file),
            None => sections.push(DirectorySection {
                name: file.name.clone(),
                priority: file.priority,
                variants: vec![file],
            }),
        }
    }
    for section in &mut sections {
        section
            .variants
            .sort_by(|left, right| left.path.cmp(&right.path));
        if let Some(priority) = section.variants.iter().map(|file| file.priority).min() {
            section.priority = priority;
        }
    }
    sections.sort_by(|left, right| {
        left.priority
            .cmp(&right.priority)
            .then_with(|| left.name.cmp(&right.name))
    });
    sections
}

/// Adds `file` to `section`, applying the duplicate-resolution rule.
fn insert_variant(section: &mut DirectorySection, file: PromptFile) {
    let Some(index) = section
        .variants
        .iter()
        .position(|existing| existing.language == file.language)
    else {
        section.variants.push(file);
        return;
    };
    let existing = &section.variants[index];
    let new_wins = file.priority > existing.priority
        || (file.priority == existing.priority && file.path > existing.path);
    tracing::warn!(
        target: "openjiuwen_agent::prompt_pack",
        name = %section.name,
        language = ?file.language,
        kept = %if new_wins { file.path.display().to_string() } else { existing.path.display().to_string() },
        "duplicate prompt section variant; keeping the higher priority then the later path"
    );
    if new_wins {
        section.variants[index] = file;
    }
}
