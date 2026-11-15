//! The file-backed [`MemoryToolBackend`] over a harness [`Workspace`].
//!
//! Python serves the eight memory tools through `MemoryToolContext`/`CodingMemoryToolContext`,
//! whose `memory_tool_ops`/`coding_memory_tool_ops` read and write the workspace's memory markdown
//! files and search a SQLite index. This backend ports the storage half faithfully and replaces
//! the index with a deterministic, offline keyword scan:
//!
//! | model path | resolved workspace path |
//! | --- | --- |
//! | `USER.md` | `<workspace>/USER.md` |
//! | `MEMORY.md` | `<workspace>/<memory>/MEMORY.md` |
//! | `YYYY-MM-DD.md` | `<workspace>/<memory>/<daily_memory>/YYYY-MM-DD.md` |
//! | anything else | `<workspace>/<memory>/<basename>` |
//! | coding `*.md` | `<workspace>/<coding_memory>/<basename>` (must end in `.md`) |
//!
//! The `<memory>`, `<daily_memory>`, `<coding_memory>` and `USER.md` nodes come from the
//! workspace schema ([`Workspace::get_directory`]), so a host that renamed a node keeps its
//! mapping; only the basename of a model-supplied path is used, exactly like Python's
//! `os.path.basename` (an absolute path or any `..` segment is rejected before that).
//! Every resolved path goes through [`Workspace::resolve`], so a symlink escaping the workspace is
//! rejected with [`Error::security`](openjiuwen_common::error::Error::Security).
//!
//! # Deviations from Python
//!
//! - **Search is keyword-only.** No embedding model and no SQLite index exist offline, so the
//!   backend scans the memory tree line by line and reports `provider: "local"`,
//!   `search_mode: "keyword_only"` and `embedding_available: false` in the Python result object.
//!   The scan walks `<memory>/**/*.md`, `<memory>/sessions/**/*.jsonl`, `USER.md` and (for the
//!   coding scope) `<coding_memory>/**/*.md`, skipping files larger than
//!   [`MAX_SEARCH_FILE_BYTES`] and searching at most [`MAX_SEARCH_FILES`] of them.
//! - **`session_key` filters session logs.** Python's lite manager ignores the option; here a hit
//!   in `<memory>/sessions/**` is kept only when its path is under `sessions/<session_key>/`.
//! - **Coding-memory append** validates the new chunk's front matter, then appends only its body
//!   and refreshes `updated_at`, like Python's `_append_to_existing_file`; it does not run
//!   Python's LLM conflict/redundancy checker nor update the `MEMORY.md` index.
//! - `read` returns the real `total_lines` of the file even when `limit` truncates the window
//!   (Python derives it from the already-truncated read).
//!
//! `edit` follows Python: exactly one replacement, an error when `old_text` is absent and an
//! error when it is ambiguous. No file lock is taken (single-threaded tool calls in one session),
//! and the write is a plain [`tokio::fs::write`].

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::lite::markdown::FrontMatter;

use crate::workspace::Workspace;

use super::super::support::truncate_chars;
use super::backend::{
    MemoryEditData, MemoryReadData, MemoryScope, MemorySearchRequest, MemoryToolBackend,
    MemoryWriteData,
};

/// Maximum number of characters a search snippet carries (`SNIPPET_MAX_CHARS`).
const SNIPPET_MAX_CHARS: usize = 700;

/// The shortest query token that survives stop-word filtering.
const MIN_TOKEN_CHARS: usize = 3;

/// The `provider` reported in the Python search result object.
const SEARCH_PROVIDER: &str = "local";

/// The `search_mode` reported in the Python search result object.
const SEARCH_MODE: &str = "keyword_only";

/// The `source` of memory-tree and coding-memory hits.
const SOURCE_MEMORY: &str = "memory";

/// The `source` of session-transcript hits.
const SOURCE_SESSIONS: &str = "sessions";

/// Largest file the keyword scan reads.
const MAX_SEARCH_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Largest number of files the keyword scan searches (after sorting by relative path).
const MAX_SEARCH_FILES: usize = 512;

/// The workspace node holding the long-term memory tree.
const MEMORY_NODE: &str = "memory";

/// The workspace node holding the daily memory files.
const DAILY_MEMORY_NODE: &str = "daily_memory";

/// The workspace node holding the coding-memory tree.
const CODING_MEMORY_NODE: &str = "coding_memory";

/// The workspace node of the user profile file.
const USER_NODE: &str = "USER.md";

/// The file name of the long-term memory index.
const MEMORY_INDEX: &str = "MEMORY.md";

/// The directory (below `<memory>`) holding session transcripts.
const SESSIONS_DIR: &str = "sessions";

/// The `type` values a coding-memory front matter accepts (`VALID_TYPES`).
const CODING_TYPES: [&str; 4] = ["user", "feedback", "project", "reference"];

/// The Python `type must be one of: {VALID_TYPES}` message.
const CODING_TYPES_MESSAGE: &str =
    "type must be one of: ('user', 'feedback', 'project', 'reference')";

/// The Python "no front matter" message of `coding_memory_write`.
const FRONT_MATTER_REQUIRED: &str = "must contain frontmatter(name/description/type)";

/// The Python "empty body" message of `coding_memory_write`.
const EMPTY_BODY: &str = "no content body";

/// The Python traversal rejection message of `validate_memory_path`.
const TRAVERSAL_MESSAGE: &str = "Invalid path: directory traversal not allowed";

/// A memory file resolved to a workspace-relative and an absolute path.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedTarget {
    /// The path relative to the workspace root, as reported to the model.
    rel: String,
    /// The absolute path the filesystem sees.
    full: PathBuf,
}

/// One file the keyword scan may search.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    /// The path relative to the workspace root.
    rel: String,
    /// The absolute path.
    full: PathBuf,
    /// The Python `source` of the file (`memory` or `sessions`).
    source: &'static str,
}

/// One grouped run of contiguous matching lines.
#[derive(Debug, Clone, PartialEq)]
struct SearchHit {
    /// The path relative to the workspace root.
    rel: String,
    /// The Python `source` of the file.
    source: &'static str,
    /// 1-based first line of the hit.
    start_line: usize,
    /// 1-based last line of the hit (inclusive).
    end_line: usize,
    /// Normalized relevance score in `0..=1`.
    score: f64,
    /// The (truncated) text of the hit.
    snippet: String,
}

impl SearchHit {
    /// Renders the hit in the Python `memory_search` result shape.
    fn to_json(&self) -> Value {
        json!({
            "id": format!("{}:{}-{}", self.rel, self.start_line, self.end_line),
            "path": self.rel,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": self.score,
            "snippet": self.snippet,
            "citation": self.citation(),
        })
    }

    /// The Python citation form: `path#L<n>` or `path#L<a>-L<b>`.
    fn citation(&self) -> String {
        if self.start_line == self.end_line {
            format!("{}#L{}", self.rel, self.start_line)
        } else {
            format!("{}#L{}-L{}", self.rel, self.start_line, self.end_line)
        }
    }
}

/// A validated coding-memory chunk: canonical front matter plus the free-form body.
#[derive(Debug, Clone, PartialEq)]
struct CodingChunk {
    /// The front matter, already enriched with `created_at`/`updated_at`.
    front_matter: FrontMatter,
    /// The body after the front matter block.
    body: String,
}

impl CodingChunk {
    /// Parses and validates `content`, returning the chunk to render.
    fn parse(content: &str) -> Result<Self> {
        let Some(mut front_matter) = parse_front_matter(content) else {
            return Err(Error::validation(FRONT_MATTER_REQUIRED));
        };
        validate_front_matter(&front_matter)?;
        enrich_front_matter(&mut front_matter, false);
        let body = extract_body(content);
        if body.is_empty() {
            return Err(Error::validation(EMPTY_BODY));
        }
        Ok(Self { front_matter, body })
    }
}

/// A file-backed [`MemoryToolBackend`] over a workspace.
///
/// The constructor performs no I/O: every root is derived from the workspace schema when a call
/// needs it, so a backend can be built before the workspace directories exist.
pub struct WorkspaceMemoryBackend {
    /// The workspace the memory trees live in.
    workspace: Workspace,
}

impl WorkspaceMemoryBackend {
    /// Builds a backend over `workspace`.
    #[must_use]
    pub fn new(workspace: Workspace) -> Self {
        Self { workspace }
    }

    /// The workspace this backend reads and writes.
    #[must_use]
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The workspace-relative `memory` node (`MEMORY_NODE`, default `memory`).
    fn memory_rel(&self) -> String {
        self.workspace
            .get_directory(MEMORY_NODE)
            .unwrap_or_else(|| MEMORY_NODE.to_owned())
    }

    /// The workspace-relative `daily_memory` node (default `daily_memory`).
    fn daily_rel(&self) -> String {
        join_rel(&self.memory_rel(), &self.daily_directory_node())
    }

    /// The `daily_memory` node path, without the `memory` prefix.
    fn daily_directory_node(&self) -> String {
        self.workspace
            .get_directory(DAILY_MEMORY_NODE)
            .unwrap_or_else(|| DAILY_MEMORY_NODE.to_owned())
    }

    /// The workspace-relative `coding_memory` node (default `coding_memory`).
    fn coding_rel(&self) -> String {
        self.workspace
            .get_directory(CODING_MEMORY_NODE)
            .unwrap_or_else(|| CODING_MEMORY_NODE.to_owned())
    }

    /// The workspace-relative `USER.md` node (default `USER.md`).
    fn user_rel(&self) -> String {
        self.workspace
            .get_directory(USER_NODE)
            .unwrap_or_else(|| USER_NODE.to_owned())
    }

    /// The workspace-relative session-transcript root (`<memory>/sessions`).
    fn sessions_rel(&self) -> String {
        join_rel(&self.memory_rel(), SESSIONS_DIR)
    }

    /// Maps a model-supplied `path` onto the workspace-relative path of `scope`.
    ///
    /// Python parity: `validate_memory_path`/`validate_coding_memory_path`. Only the basename of
    /// `path` is used; absolute paths and any `..` segment are rejected before that so a caller
    /// cannot aim at a different tree.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an empty path, a traversal attempt, or (coding memory) a
    /// name that does not end in `.md`.
    fn relative_target(&self, scope: MemoryScope, path: &str) -> Result<String> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return Err(Error::validation("path is required"));
        }
        if trimmed.starts_with('/') || trimmed.starts_with('\\') || trimmed.contains("..") {
            return Err(Error::validation(TRAVERSAL_MESSAGE));
        }
        let basename = basename_of(trimmed);
        if basename.is_empty() || basename == "." {
            return Err(Error::validation(format!("Invalid path: {path}")));
        }
        match scope {
            MemoryScope::Memory => {
                if basename == USER_NODE {
                    Ok(self.user_rel())
                } else if basename == MEMORY_INDEX {
                    Ok(join_rel(&self.memory_rel(), basename))
                } else if is_daily_name(basename) {
                    Ok(join_rel(&self.daily_rel(), basename))
                } else {
                    Ok(join_rel(&self.memory_rel(), basename))
                }
            }
            MemoryScope::CodingMemory => {
                if !basename.to_ascii_lowercase().ends_with(".md") {
                    return Err(Error::validation("Path must end with .md"));
                }
                Ok(join_rel(&self.coding_rel(), basename))
            }
        }
    }

    /// Resolves `path` of `scope` through the workspace security gate.
    async fn resolve_target(&self, scope: MemoryScope, path: &str) -> Result<ResolvedTarget> {
        let rel = self.relative_target(scope, path)?;
        let full = self.workspace.resolve(&rel).await?;
        Ok(ResolvedTarget { rel, full })
    }

    /// Reads the whole file at `target`, returning `None` when it does not exist.
    async fn read_optional(&self, target: &ResolvedTarget) -> Result<Option<String>> {
        match tokio::fs::read(&target.full).await {
            Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(read_error(&target.rel, &error)),
        }
    }

    /// Writes `text` to `target`, creating the parent directories first.
    async fn write_text(&self, target: &ResolvedTarget, text: &str) -> Result<()> {
        if let Some(parent) = target.full.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| write_error(&target.rel, &error))?;
        }
        tokio::fs::write(&target.full, text)
            .await
            .map_err(|error| write_error(&target.rel, &error))
    }

    /// The [`MemoryToolBackend::read`] body.
    async fn read_impl(
        &self,
        scope: MemoryScope,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<MemoryReadData> {
        let target = self.resolve_target(scope, path).await?;
        let bytes = tokio::fs::read(&target.full)
            .await
            .map_err(|error| read_error(&target.rel, &error))?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let lines: Vec<&str> = text.split('\n').collect();
        let window = LineWindow::of(&lines, offset, limit);
        Ok(MemoryReadData {
            path: target.rel,
            text: window.text,
            total_lines: lines.len(),
            start_line: window.start_line,
            end_line: window.end_line,
        })
    }

    /// The [`MemoryToolBackend::write`] body.
    async fn write_impl(
        &self,
        scope: MemoryScope,
        path: &str,
        content: &str,
        append: bool,
    ) -> Result<MemoryWriteData> {
        let target = self.resolve_target(scope, path).await?;
        let existing = self.read_optional(&target).await?;
        let file_existed = existing.as_deref().is_some_and(|text| !text.is_empty());
        let text = match scope {
            MemoryScope::Memory => {
                if append {
                    append_memory_text(existing.as_deref().unwrap_or_default(), content)
                } else {
                    content.to_owned()
                }
            }
            MemoryScope::CodingMemory => coding_document(append, existing.as_deref(), content)?,
        };
        self.write_text(&target, &text).await?;
        Ok(MemoryWriteData {
            path: target.rel,
            full_path: target.full.to_string_lossy().into_owned(),
            appended: append,
            file_existed,
        })
    }

    /// The [`MemoryToolBackend::edit`] body.
    async fn edit_impl(
        &self,
        scope: MemoryScope,
        path: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<MemoryEditData> {
        if scope == MemoryScope::CodingMemory && old_text.is_empty() {
            return Err(Error::validation("old_text cannot be empty"));
        }
        let target = self.resolve_target(scope, path).await?;
        let content = self.read_optional(&target).await?.ok_or_else(|| {
            if scope == MemoryScope::CodingMemory {
                Error::not_found(format!("failed to read file: {path}"))
            } else {
                read_error(
                    &target.rel,
                    &std::io::Error::from(std::io::ErrorKind::NotFound),
                )
            }
        })?;
        let occurrences = content.matches(old_text).count();
        if occurrences == 0 {
            return Err(Error::tool(match scope {
                MemoryScope::Memory => {
                    "old_text not found in file. Use read_memory tool to check exact content."
                }
                MemoryScope::CodingMemory => "old_text not found in file",
            }));
        }
        if occurrences > 1 {
            return Err(Error::tool(match scope {
                MemoryScope::Memory => {
                    format!("old_text appears {occurrences} times in file. Be more specific.")
                }
                MemoryScope::CodingMemory => {
                    format!("old_text appears {occurrences} times, please be more specific")
                }
            }));
        }
        let updated = content.replacen(old_text, new_text, 1);
        self.write_text(&target, &updated).await?;
        Ok(MemoryEditData {
            path: target.rel,
            replacements: 1,
        })
    }

    /// The [`MemoryToolBackend::search`] body: a deterministic keyword scan.
    async fn search_impl(&self, request: &MemorySearchRequest) -> Result<Value> {
        let tokens = query_tokens(&request.query);
        let mut hits = Vec::new();
        if !tokens.is_empty() {
            for candidate in self.candidates(request).await {
                collect_hits(&candidate, &tokens, &mut hits).await;
            }
        }
        hits.retain(|hit| hit.score >= request.min_score);
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.rel.cmp(&right.rel))
                .then_with(|| left.start_line.cmp(&right.start_line))
        });
        hits.truncate(request.max_results);
        Ok(json!({
            "results": hits.iter().map(SearchHit::to_json).collect::<Vec<_>>(),
            "provider": SEARCH_PROVIDER,
            "model": Value::Null,
            "search_mode": SEARCH_MODE,
            "embedding_available": false,
            "embedding_error": Value::Null,
            "disabled": false,
        }))
    }

    /// Collects the files the keyword scan of `request` searches.
    async fn candidates(&self, request: &MemorySearchRequest) -> Vec<Candidate> {
        let root = &self.workspace.root_path;
        let mut candidates: Vec<Candidate> = Vec::new();
        match request.scope {
            MemoryScope::Memory => {
                self.push_markdown_candidates(&mut candidates).await;
                let sessions_rel = self.sessions_rel();
                let session_key = request.session_key.as_deref();
                for path in collect_files(&root.join(&sessions_rel), "jsonl").await {
                    let rel = self.relative(&path);
                    if let Some(key) = session_key {
                        let prefix = join_rel(&sessions_rel, key);
                        if rel != prefix && !rel.starts_with(&format!("{prefix}/")) {
                            continue;
                        }
                    }
                    candidates.push(Candidate {
                        rel,
                        full: path,
                        source: SOURCE_SESSIONS,
                    });
                }
                let user_rel = self.user_rel();
                let user_full = root.join(&user_rel);
                if tokio::fs::metadata(&user_full)
                    .await
                    .is_ok_and(|metadata| metadata.is_file())
                {
                    candidates.push(Candidate {
                        rel: user_rel,
                        full: user_full,
                        source: SOURCE_MEMORY,
                    });
                }
            }
            MemoryScope::CodingMemory => {
                for path in collect_files(&root.join(self.coding_rel()), "md").await {
                    candidates.push(Candidate {
                        rel: self.relative(&path),
                        full: path,
                        source: SOURCE_MEMORY,
                    });
                }
            }
        }
        candidates.sort_by(|left, right| left.rel.cmp(&right.rel));
        candidates.truncate(MAX_SEARCH_FILES);
        candidates
    }

    /// Collects every markdown file of the long-term memory tree.
    async fn push_markdown_candidates(&self, candidates: &mut Vec<Candidate>) {
        let root = &self.workspace.root_path;
        for path in collect_files(&root.join(self.memory_rel()), "md").await {
            let rel = self.relative(&path);
            candidates.push(Candidate {
                rel,
                full: path,
                source: SOURCE_MEMORY,
            });
        }
    }

    /// Renders `path` relative to the workspace root with `/` separators.
    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.workspace.root_path)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }
}

#[async_trait]
impl MemoryToolBackend for WorkspaceMemoryBackend {
    async fn search(&self, request: MemorySearchRequest) -> Result<Value> {
        self.search_impl(&request).await
    }

    async fn read(
        &self,
        scope: MemoryScope,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<MemoryReadData> {
        self.read_impl(scope, path, offset, limit).await
    }

    async fn write(
        &self,
        scope: MemoryScope,
        path: &str,
        content: &str,
        append: bool,
    ) -> Result<MemoryWriteData> {
        self.write_impl(scope, path, content, append).await
    }

    async fn edit(
        &self,
        scope: MemoryScope,
        path: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<MemoryEditData> {
        self.edit_impl(scope, path, old_text, new_text).await
    }
}

/// A line window of a memory file, in the shape the tools render.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LineWindow {
    /// The text of the window, lines joined by `\n`.
    text: String,
    /// 1-based line number of the first returned line.
    start_line: usize,
    /// 1-based line number of the last returned line (inclusive).
    end_line: usize,
}

impl LineWindow {
    /// Windows `lines` by the 1-based `offset` (`0` means the first line) and `limit`.
    fn of(lines: &[&str], offset: usize, limit: Option<usize>) -> Self {
        let total = lines.len();
        let start = if offset == 0 {
            0
        } else {
            (offset - 1).min(total)
        };
        let end = limit
            .map_or(total, |cap| start.saturating_add(cap).min(total))
            .max(start);
        Self {
            text: lines[start..end].join("\n"),
            start_line: start + 1,
            end_line: end,
        }
    }
}

/// Builds the Python `read_memory` failure for an I/O error.
fn read_error(rel: &str, error: &std::io::Error) -> Error {
    if error.kind() == std::io::ErrorKind::NotFound {
        Error::not_found(format!("memory file not found: {rel}"))
    } else {
        Error::tool(format!("failed to read memory file {rel}: {error}"))
    }
}

/// Builds the Python `write_memory` failure for an I/O error.
fn write_error(rel: &str, error: &std::io::Error) -> Error {
    Error::tool(format!("failed to write memory file {rel}: {error}"))
}

/// Returns the last path segment of `path` (`os.path.basename` over `/` and `\`).
fn basename_of(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Joins a workspace-relative base with `name`, never duplicating the prefix.
fn join_rel(base: &str, name: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.is_empty() {
        return name.to_owned();
    }
    if name == base || name.starts_with(&format!("{base}/")) {
        return name.to_owned();
    }
    format!("{base}/{name}")
}

/// Whether `name` is a Python daily memory file (`^\d{4}-\d{2}-\d{2}\.md$`).
fn is_daily_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() == 13
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[8..10].iter().all(u8::is_ascii_digit)
        && &bytes[10..] == b".md"
}

/// Appends `content` to `existing`, inserting a newline when the file needs one.
fn append_memory_text(existing: &str, content: &str) -> String {
    if existing.is_empty() {
        return content.to_owned();
    }
    if existing.ends_with('\n') {
        format!("{existing}{content}")
    } else {
        format!("{existing}\n{content}")
    }
}

/// Renders the canonical content of a coding-memory write.
///
/// Python parity: `coding_memory_write_with_context`. A fresh write renders the validated front
/// matter plus the body; an append validates the new chunk, appends only its body to the existing
/// one and refreshes `updated_at` (Python's `_append_to_existing_file`).
fn coding_document(append: bool, existing: Option<&str>, content: &str) -> Result<String> {
    let chunk = CodingChunk::parse(content)?;
    let Some(existing) = existing.filter(|text| !text.is_empty()).filter(|_| append) else {
        return Ok(render_coding_document(&chunk.front_matter, &chunk.body));
    };
    let mut front_matter =
        parse_front_matter(existing).unwrap_or_else(|| chunk.front_matter.clone());
    enrich_front_matter(&mut front_matter, true);
    let body = match extract_body(existing) {
        existing_body if existing_body.is_empty() => chunk.body,
        existing_body => format!("{existing_body}\n\n{}", chunk.body),
    };
    Ok(render_coding_document(&front_matter, &body))
}

/// Parses the `---` front matter block of `content` (`parse_frontmatter`).
///
/// Returns `None` when the content does not open with a fence, has no closing fence, or carries no
/// `key: value` line.
fn parse_front_matter(content: &str) -> Option<FrontMatter> {
    let content = content.trim();
    let rest = content.strip_prefix("---")?;
    let end = rest.find("---")?;
    let mut front_matter = FrontMatter::new();
    for line in rest[..end].trim().split('\n') {
        if let Some((key, value)) = line.split_once(':') {
            front_matter.insert(key.trim(), value.trim());
        }
    }
    (!front_matter.is_empty()).then_some(front_matter)
}

/// Validates the required `name`/`description`/`type` fields (`validate_frontmatter`).
fn validate_front_matter(front_matter: &FrontMatter) -> Result<()> {
    for field in ["name", "description", "type"] {
        if front_matter.get(field).is_none_or(str::is_empty) {
            return Err(Error::validation(format!(
                "Missing required field: {field}"
            )));
        }
    }
    let kind = front_matter.get("type").unwrap_or_default();
    if !CODING_TYPES.contains(&kind) {
        return Err(Error::validation(CODING_TYPES_MESSAGE));
    }
    Ok(())
}

/// Fills `created_at` on creation and refreshes `updated_at` (`enrich_frontmatter`).
fn enrich_front_matter(front_matter: &mut FrontMatter, is_edit: bool) {
    let today = chrono::Utc::now()
        .date_naive()
        .format("%Y-%m-%d")
        .to_string();
    if !is_edit && front_matter.get("created_at").is_none() {
        front_matter.insert("created_at", &today);
    }
    front_matter.insert("updated_at", today);
}

/// Extracts the body after the front matter block (`_extract_body`).
fn extract_body(content: &str) -> String {
    let content = content.trim();
    let Some(rest) = content.strip_prefix("---") else {
        return content.to_owned();
    };
    let Some(end) = rest.find("---") else {
        return String::new();
    };
    rest[end + 3..].trim().to_owned()
}

/// Renders front matter plus body in the canonical coding-memory shape
/// (`rebuild_content_with_frontmatter`).
fn render_coding_document(front_matter: &FrontMatter, body: &str) -> String {
    let mut rendered = String::from("---\n");
    for (key, value) in front_matter.iter() {
        rendered.push_str(key);
        rendered.push_str(": ");
        rendered.push_str(value);
        rendered.push('\n');
    }
    rendered.push_str("---");
    if body.is_empty() {
        rendered
    } else {
        format!("{rendered}\n\n{body}")
    }
}

/// Splits `query` into the keyword tokens the scan scores.
///
/// Tokens are lowercased alphanumeric runs with duplicates removed. Runs shorter than
/// [`MIN_TOKEN_CHARS`] are dropped unless every run is that short, so a two-letter query still
/// searches instead of silently returning nothing.
fn query_tokens(query: &str) -> Vec<String> {
    let mut all: Vec<String> = Vec::new();
    for run in query.split(|character: char| !character.is_alphanumeric()) {
        if run.is_empty() {
            continue;
        }
        let token = run.to_lowercase();
        if !all.contains(&token) {
            all.push(token);
        }
    }
    let meaningful: Vec<String> = all
        .iter()
        .filter(|token| token.chars().count() >= MIN_TOKEN_CHARS)
        .cloned()
        .collect();
    if meaningful.is_empty() {
        all
    } else {
        meaningful
    }
}

/// Scores one line: the share of query tokens it contains, normalized to `0..=1`.
#[allow(clippy::cast_precision_loss)] // token counts are far below 2^53
fn line_score(line_lower: &str, tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let matched = tokens
        .iter()
        .filter(|token| line_lower.contains(token.as_str()))
        .count();
    matched as f64 / tokens.len() as f64
}

/// Groups the matching lines of `candidate` into hits and appends them to `out`.
async fn collect_hits(candidate: &Candidate, tokens: &[String], out: &mut Vec<SearchHit>) {
    let Ok(metadata) = tokio::fs::metadata(&candidate.full).await else {
        return;
    };
    if !metadata.is_file() || metadata.len() > MAX_SEARCH_FILE_BYTES {
        return;
    }
    let Ok(bytes) = tokio::fs::read(&candidate.full).await else {
        return;
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.split('\n').collect();
    let mut window: Option<(usize, usize, f64)> = None;
    for (index, line) in lines.iter().enumerate() {
        let number = index + 1;
        let score = line_score(&line.to_lowercase(), tokens);
        if score <= 0.0 {
            flush_window(window.take(), candidate, &lines, out);
            continue;
        }
        match window {
            Some((start, end, best)) if end + 1 == number => {
                window = Some((start, number, best.max(score)));
            }
            _ => {
                flush_window(window.take(), candidate, &lines, out);
                window = Some((number, number, score));
            }
        }
    }
    flush_window(window.take(), candidate, &lines, out);
}

/// Emits the grouped window as a [`SearchHit`].
fn flush_window(
    window: Option<(usize, usize, f64)>,
    candidate: &Candidate,
    lines: &[&str],
    out: &mut Vec<SearchHit>,
) {
    let Some((start, end, score)) = window else {
        return;
    };
    let text = lines[start - 1..end].join("\n");
    out.push(SearchHit {
        rel: candidate.rel.clone(),
        source: candidate.source,
        start_line: start,
        end_line: end,
        score,
        snippet: truncate_chars(&text, SNIPPET_MAX_CHARS).text,
    });
}

/// Lists every file with `extension` below `root`, recursively and without following dir symlinks.
///
/// A missing or unreadable root yields an empty list; the caller sorts the result, so the
/// directory-entry order never reaches a search result.
async fn collect_files(root: &Path, extension: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(mut entries) = tokio::fs::read_dir(&directory).await else {
            continue;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            let Ok(file_type) = entry.file_type().await else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if !has_extension(&path, extension) {
                continue;
            }
            if tokio::fs::metadata(&path)
                .await
                .is_ok_and(|metadata| metadata.is_file())
            {
                found.push(path);
            }
        }
    }
    found
}

/// Whether the file name of `path` ends with `.extension`, case-insensitively.
fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .is_some_and(|candidate| candidate.to_string_lossy().eq_ignore_ascii_case(extension))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds a backend over a throwaway English workspace.
    fn backend() -> (tempfile::TempDir, WorkspaceMemoryBackend) {
        let dir = tempfile::tempdir().expect("temp workspace");
        let workspace = Workspace::for_language(dir.path(), "en");
        (dir, WorkspaceMemoryBackend::new(workspace))
    }

    #[test]
    fn memory_paths_follow_the_python_mapping() {
        let (_dir, backend) = backend();

        let cases = [
            ("USER.md", "USER.md"),
            ("MEMORY.md", "memory/MEMORY.md"),
            ("memory/MEMORY.md", "memory/MEMORY.md"),
            ("2026-11-15.md", "memory/daily_memory/2026-11-15.md"),
            ("notes.md", "memory/notes.md"),
            ("nested/notes.md", "memory/notes.md"),
        ];
        for (path, expected) in cases {
            assert_eq!(
                backend
                    .relative_target(MemoryScope::Memory, path)
                    .expect("valid memory path"),
                expected,
                "{path}"
            );
        }
    }

    #[test]
    fn coding_paths_require_a_markdown_name() {
        let (_dir, backend) = backend();

        assert_eq!(
            backend
                .relative_target(MemoryScope::CodingMemory, "projects/rust.md")
                .expect("valid coding path"),
            "coding_memory/rust.md"
        );
        let error = backend
            .relative_target(MemoryScope::CodingMemory, "notes.txt")
            .expect_err("txt is rejected");
        assert_eq!(error.message(), "Path must end with .md");
    }

    #[test]
    fn traversal_and_empty_paths_are_validation_errors() {
        let (_dir, backend) = backend();

        for path in ["../x.md", "/etc/passwd", "\\evil.md", "a/../../b.md"] {
            let error = backend
                .relative_target(MemoryScope::Memory, path)
                .expect_err("unsafe path");
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation,
                "{path}"
            );
        }
        assert_eq!(
            backend
                .relative_target(MemoryScope::Memory, "  ")
                .expect_err("blank path")
                .message(),
            "path is required"
        );
    }

    #[test]
    fn line_windows_follow_python_offsets() {
        let lines = ["one", "two", "three", "four"];

        let window = LineWindow::of(&lines, 0, None);
        assert_eq!(window.text, "one\ntwo\nthree\nfour");
        assert_eq!((window.start_line, window.end_line), (1, 4));

        let window = LineWindow::of(&lines, 1, Some(2));
        assert_eq!(window.text, "one\ntwo");
        assert_eq!((window.start_line, window.end_line), (1, 2));

        let window = LineWindow::of(&lines, 3, Some(2));
        assert_eq!(window.text, "three\nfour");
        assert_eq!((window.start_line, window.end_line), (3, 4));

        let window = LineWindow::of(&lines, 9, Some(2));
        assert_eq!(window.text, "");
        assert_eq!((window.start_line, window.end_line), (5, 4));
    }

    #[test]
    fn query_tokens_drop_short_words_and_duplicates() {
        assert_eq!(query_tokens("go to the SHOP"), ["the", "shop"]);
        assert_eq!(query_tokens("build BUILD steps"), ["build", "steps"]);
        assert_eq!(query_tokens("go"), ["go"]);
        assert!(query_tokens(" ... ").is_empty());
    }

    #[test]
    fn coding_front_matter_round_trips() {
        let content =
            "---\nname: build\ndescription: how to build\ntype: project\n---\n\ncargo build";
        let chunk = CodingChunk::parse(content).expect("valid chunk");

        assert_eq!(chunk.front_matter.get("name"), Some("build"));
        assert_eq!(chunk.body, "cargo build");
        assert!(chunk.front_matter.get("created_at").is_some());
        let rendered = render_coding_document(&chunk.front_matter, &chunk.body);
        assert!(
            rendered.starts_with("---\nname: build\ndescription: how to build\ntype: project\n")
        );
        assert!(rendered.ends_with("---\n\ncargo build"));
    }

    #[test]
    fn coding_front_matter_reports_the_python_messages() {
        let error = CodingChunk::parse("no front matter").expect_err("missing front matter");
        assert_eq!(error.message(), FRONT_MATTER_REQUIRED);

        let error = CodingChunk::parse("---\nname: x\ntype: project\n---\n\nbody")
            .expect_err("missing description");
        assert_eq!(error.message(), "Missing required field: description");

        let error = CodingChunk::parse("---\nname: x\ndescription: y\ntype: other\n---\n\nbody")
            .expect_err("bad type");
        assert_eq!(error.message(), CODING_TYPES_MESSAGE);

        let error = CodingChunk::parse("---\nname: x\ndescription: y\ntype: user\n---\n\n")
            .expect_err("empty body");
        assert_eq!(error.message(), EMPTY_BODY);
    }

    #[test]
    fn daily_names_match_only_iso_dates() {
        assert!(is_daily_name("2026-11-15.md"));
        assert!(!is_daily_name("2026-11-15.MD"));
        assert!(!is_daily_name("26-11-15.md"));
        assert!(!is_daily_name("2026-11-15.md.bak"));
    }
}
