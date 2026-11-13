//! Project instruction files loaded into the system prompt.
//!
//! Python parity: `~/.openjiuwen/OPENJIUWEN.md` is user-level memory and
//! `<project>/OPENJIUWEN.md` is project-level memory; both are concatenated with
//! the user layer first, a 40 000-character cap and a `[...truncated]` marker.
//!
//! # Documented additions
//!
//! The Rust port also reads `IDENTITY.md` as an identity preamble that is
//! composed before the memory layers. The workspace copy wins over the home
//! copy and at most one identity document is kept. This file is a Rust
//! addition, not a Python behaviour.
//!
//! # Discovery and precedence
//!
//! - Identity: `<workspace>/IDENTITY.md` wins over `<home>/IDENTITY.md`; when
//!   both exist only the workspace document is kept.
//! - Memory: `<home>/OPENJIUWEN.md` first, then `<workspace>/OPENJIUWEN.md`.
//! - Missing files are skipped silently; an empty or whitespace-only file is
//!   treated as absent.
//! - Any other read failure — including a directory where a file is expected —
//!   is logged with `tracing::warn!` and the file is skipped, so a broken
//!   instruction file never blocks a run.
//!
//! If an individual document exceeds [`MAX_INSTRUCTION_CHARS`] it is cut to the
//! cap and marked [`InstructionDoc::truncated`]; the composed prompt is capped
//! the same way. Cutting is done by `chars().count()`, never by bytes, so
//! multi-byte content cannot panic.
#![allow(clippy::module_name_repetitions)] // names mirror the phase plan's public API

use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::Result;

/// Name of the identity preamble file looked up in the workspace and home.
pub const IDENTITY_FILENAME: &str = "IDENTITY.md";
/// Name of the memory document looked up in the workspace and home.
pub const MEMORY_FILENAME: &str = "OPENJIUWEN.md";
/// Maximum number of characters kept from the composed instruction text.
pub const MAX_INSTRUCTION_CHARS: usize = 40_000;
/// Marker appended when instruction text had to be cut.
pub const TRUNCATION_MARKER: &str = "\n[...truncated]";
/// Header placed before the home-level memory document.
pub const USER_MEMORY_HEADER: &str = "### User-level memory";
/// Header placed before the project-level memory document.
pub const PROJECT_MEMORY_HEADER: &str = "### Project-level memory";

/// Where one instruction document was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionSource {
    /// `<home>/IDENTITY.md`, used only when the workspace has no identity.
    HomeIdentity,
    /// `<workspace>/IDENTITY.md`, the preferred identity document.
    WorkspaceIdentity,
    /// `<home>/OPENJIUWEN.md`, the user-level memory layer.
    HomeMemory,
    /// `<workspace>/OPENJIUWEN.md`, the project-level memory layer.
    WorkspaceMemory,
}

impl InstructionSource {
    /// Returns the stable `location:filename` label of this source.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HomeIdentity => "home:IDENTITY.md",
            Self::WorkspaceIdentity => "workspace:IDENTITY.md",
            Self::HomeMemory => "home:OPENJIUWEN.md",
            Self::WorkspaceMemory => "workspace:OPENJIUWEN.md",
        }
    }
}

/// One discovered instruction document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionDoc {
    /// Where the document was found.
    pub source: InstructionSource,
    /// Absolute path the document was read from.
    pub path: PathBuf,
    /// Document text, already capped at [`MAX_INSTRUCTION_CHARS`] when needed.
    pub content: String,
    /// Whether the document was cut and the truncation marker appended.
    pub truncated: bool,
}

/// The identity preamble and memory layers discovered for one run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstructionFiles {
    /// The single identity document, workspace-first.
    pub identity: Option<InstructionDoc>,
    /// Memory documents ordered home-level then project-level.
    pub memory: Vec<InstructionDoc>,
}

impl InstructionFiles {
    /// Loads the identity and memory documents for `workspace` and `home`.
    ///
    /// Missing files, empty files and unreadable files are skipped; see the
    /// module documentation for precedence.
    ///
    /// # Errors
    ///
    /// Reserved for a later wiring phase; discovery itself never fails, so this
    /// currently always returns `Ok`.
    #[allow(clippy::unnecessary_wraps)] // kept fallible for the documented public API
    pub fn load(workspace: &Path, home: &OpenJiuwenHome) -> Result<Self> {
        let identity = read_document(
            &workspace.join(IDENTITY_FILENAME),
            InstructionSource::WorkspaceIdentity,
        )
        .or_else(|| {
            read_document(
                &home.join(IDENTITY_FILENAME),
                InstructionSource::HomeIdentity,
            )
        });

        let mut memory = Vec::new();
        for (path, source) in [
            (home.join(MEMORY_FILENAME), InstructionSource::HomeMemory),
            (
                workspace.join(MEMORY_FILENAME),
                InstructionSource::WorkspaceMemory,
            ),
        ] {
            if let Some(doc) = read_document(&path, source) {
                memory.push(doc);
            }
        }

        Ok(Self { identity, memory })
    }

    /// Returns whether no instruction document was found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.identity.is_none() && self.memory.is_empty()
    }

    /// Returns the number of documents found.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::from(self.identity.is_some()) + self.memory.len()
    }

    /// Iterates over the identity document first, then the memory layers.
    pub fn documents(&self) -> impl Iterator<Item = &InstructionDoc> {
        self.identity.iter().chain(self.memory.iter())
    }

    /// Composes the identity preamble plus the memory layers.
    ///
    /// The identity text comes first, then [`USER_MEMORY_HEADER`] with the home
    /// memory and [`PROJECT_MEMORY_HEADER`] with the workspace memory; parts are
    /// separated by a blank line. Returns `None` when nothing was found and caps
    /// the result at [`MAX_INSTRUCTION_CHARS`], appending
    /// [`TRUNCATION_MARKER`] when it had to cut.
    #[must_use]
    pub fn system_prompt(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(doc) = &self.identity {
            parts.push(trim_trailing_newlines(&doc.content).to_owned());
        }
        for doc in &self.memory {
            let header = match doc.source {
                InstructionSource::HomeMemory => USER_MEMORY_HEADER,
                InstructionSource::HomeIdentity | InstructionSource::WorkspaceIdentity => {
                    continue;
                }
                InstructionSource::WorkspaceMemory => PROJECT_MEMORY_HEADER,
            };
            parts.push(format!(
                "{header}\n{}",
                trim_trailing_newlines(&doc.content)
            ));
        }
        if parts.is_empty() {
            return None;
        }
        Some(cut_to_cap(&parts.join("\n\n")).0)
    }
}

/// Reads one instruction document, skipping absent, empty and unreadable files.
fn read_document(path: &Path, source: InstructionSource) -> Option<InstructionDoc> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "cannot read instruction file; skipping"
            );
            return None;
        }
    };
    if raw.trim().is_empty() {
        return None;
    }
    let (content, truncated) = cut_to_cap(&raw);
    Some(InstructionDoc {
        source,
        path: path.to_path_buf(),
        content,
        truncated,
    })
}

/// Cuts `text` to [`MAX_INSTRUCTION_CHARS`] characters, appending the marker.
///
/// The returned flag reports whether a cut happened. The marker is counted
/// inside the cap so the result never exceeds it by `chars().count()`.
fn cut_to_cap(text: &str) -> (String, bool) {
    if text.chars().count() <= MAX_INSTRUCTION_CHARS {
        return (text.to_owned(), false);
    }
    let keep = MAX_INSTRUCTION_CHARS.saturating_sub(TRUNCATION_MARKER.chars().count());
    let mut cut: String = text.chars().take(keep).collect();
    cut.push_str(TRUNCATION_MARKER);
    (cut, true)
}

/// Trims trailing `\n` and `\r` characters from a document body.
fn trim_trailing_newlines(text: &str) -> &str {
    text.trim_end_matches(['\n', '\r'])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use tempfile::TempDir;

    fn fixture() -> (TempDir, PathBuf, OpenJiuwenHome) {
        let temp = TempDir::new().unwrap();
        let workspace = temp.path().join("workspace");
        let home_root = temp.path().join("home");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&home_root).unwrap();
        (temp, workspace, OpenJiuwenHome::from_root(home_root))
    }

    fn write(path: &Path, text: &str) {
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn workspace_identity_and_memory_are_loaded() {
        let (_temp, workspace, home) = fixture();
        write(&workspace.join(IDENTITY_FILENAME), "identity\n");
        write(&workspace.join(MEMORY_FILENAME), "project memory\n");

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(
            files.identity.as_ref().map(|doc| doc.source),
            Some(InstructionSource::WorkspaceIdentity)
        );
        assert_eq!(files.memory.len(), 1);
        assert_eq!(files.memory[0].source, InstructionSource::WorkspaceMemory);
        assert_eq!(files.documents().count(), 2);
    }

    #[test]
    fn home_identity_and_memory_are_loaded() {
        let (_temp, workspace, home) = fixture();
        write(&home.join(IDENTITY_FILENAME), "home identity");
        write(&home.join(MEMORY_FILENAME), "home memory");

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(
            files.identity.as_ref().map(|doc| doc.source),
            Some(InstructionSource::HomeIdentity)
        );
        assert_eq!(files.memory[0].source, InstructionSource::HomeMemory);
    }

    #[test]
    fn workspace_identity_wins_over_home() {
        let (_temp, workspace, home) = fixture();
        write(&workspace.join(IDENTITY_FILENAME), "workspace identity");
        write(&home.join(IDENTITY_FILENAME), "home identity");

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        let identity: Vec<_> = files
            .documents()
            .filter(|doc| {
                doc.source == InstructionSource::WorkspaceIdentity
                    || doc.source == InstructionSource::HomeIdentity
            })
            .collect();
        assert_eq!(identity.len(), 1);
        assert_eq!(identity[0].content, "workspace identity");
        assert_eq!(identity[0].source, InstructionSource::WorkspaceIdentity);
    }

    #[test]
    fn memory_layers_are_home_then_workspace() {
        let (_temp, workspace, home) = fixture();
        write(&home.join(MEMORY_FILENAME), "user layer");
        write(&workspace.join(MEMORY_FILENAME), "project layer");

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        assert_eq!(files.memory.len(), 2);
        assert_eq!(files.memory[0].source, InstructionSource::HomeMemory);
        assert_eq!(files.memory[1].source, InstructionSource::WorkspaceMemory);

        let prompt = files.system_prompt().unwrap();
        assert_eq!(
            prompt,
            "### User-level memory\nuser layer\n\n### Project-level memory\nproject layer"
        );
        assert!(prompt.contains(USER_MEMORY_HEADER));
        assert!(prompt.contains(PROJECT_MEMORY_HEADER));
    }

    #[test]
    fn nothing_present_is_empty() {
        let (_temp, workspace, home) = fixture();
        let files = InstructionFiles::load(&workspace, &home).unwrap();
        assert!(files.is_empty());
        assert_eq!(files.len(), 0);
        assert_eq!(files.documents().count(), 0);
        assert_eq!(files.system_prompt(), None);
    }

    #[test]
    fn whitespace_only_file_is_absent() {
        let (_temp, workspace, home) = fixture();
        write(&workspace.join(IDENTITY_FILENAME), "  \n\t\n");
        write(&workspace.join(MEMORY_FILENAME), "\n \r\n");

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        assert!(files.is_empty());
        assert_eq!(files.system_prompt(), None);
    }

    #[test]
    fn oversized_memory_is_truncated() {
        let (_temp, workspace, home) = fixture();
        write(&workspace.join(MEMORY_FILENAME), &"a".repeat(50_000));

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        let doc = &files.memory[0];
        assert!(doc.truncated);
        assert!(doc.content.ends_with(TRUNCATION_MARKER));
        assert!(doc.content.chars().count() <= MAX_INSTRUCTION_CHARS);

        let prompt = files.system_prompt().unwrap();
        assert!(prompt.ends_with(TRUNCATION_MARKER));
        assert!(prompt.chars().count() <= MAX_INSTRUCTION_CHARS);
    }

    #[test]
    fn multibyte_content_at_cut_point_does_not_panic() {
        let (_temp, workspace, home) = fixture();
        write(&workspace.join(MEMORY_FILENAME), &"é".repeat(50_000));

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        let doc = &files.memory[0];
        assert!(doc.truncated);
        assert!(doc.content.ends_with(TRUNCATION_MARKER));
        assert!(doc.content.chars().count() <= MAX_INSTRUCTION_CHARS);
        assert!(files.system_prompt().is_some());
    }

    #[test]
    fn directory_instead_of_memory_file_is_skipped() {
        let (_temp, workspace, home) = fixture();
        std::fs::create_dir_all(home.join(MEMORY_FILENAME)).unwrap();
        write(&workspace.join(MEMORY_FILENAME), "project memory");

        let files = InstructionFiles::load(&workspace, &home).unwrap();
        assert_eq!(files.memory.len(), 1);
        assert_eq!(files.memory[0].source, InstructionSource::WorkspaceMemory);
        assert_eq!(files.memory[0].content, "project memory");
    }

    #[test]
    fn source_labels_are_stable() {
        assert_eq!(InstructionSource::HomeIdentity.as_str(), "home:IDENTITY.md");
        assert_eq!(
            InstructionSource::WorkspaceIdentity.as_str(),
            "workspace:IDENTITY.md"
        );
        assert_eq!(InstructionSource::HomeMemory.as_str(), "home:OPENJIUWEN.md");
        assert_eq!(
            InstructionSource::WorkspaceMemory.as_str(),
            "workspace:OPENJIUWEN.md"
        );
    }
}
