//! The memory backend seam: the request/result data types and the [`MemoryToolBackend`] trait.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

/// The memory tree a tool addresses.
///
/// Python parity: `memory` and `coding_memory` are served by separate contexts
/// (`MemoryToolContext`, `CodingMemoryToolContext`); the scope lets one backend serve both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// The long-term `memory` tree.
    Memory,
    /// The `coding_memory` tree with its front-matter rules.
    CodingMemory,
}

/// One `memory_search` call, as the backend sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorySearchRequest {
    /// The keywords or question to look up.
    pub query: String,
    /// Maximum number of hits to return.
    pub max_results: usize,
    /// Minimum relevance score a hit must reach.
    pub min_score: f64,
    /// Optional session key used to scope or filter the hits.
    pub session_key: Option<String>,
    /// The memory tree to search.
    pub scope: MemoryScope,
}

/// The window of a memory file a read returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryReadData {
    /// The resolved path relative to the memory root.
    pub path: String,
    /// The text of the returned window.
    pub text: String,
    /// Total number of lines in the file.
    pub total_lines: usize,
    /// 1-based line number the window starts at.
    pub start_line: usize,
    /// 1-based line number the window ends at.
    pub end_line: usize,
}

/// The outcome of a memory write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryWriteData {
    /// The resolved path relative to the memory root.
    pub path: String,
    /// The absolute path of the written file.
    pub full_path: String,
    /// Whether the content was appended instead of replacing the file.
    pub appended: bool,
    /// Whether the file already existed before the write.
    pub file_existed: bool,
}

/// The outcome of a memory edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEditData {
    /// The resolved path relative to the memory root.
    pub path: String,
    /// Number of `old_text` occurrences that were replaced.
    pub replacements: usize,
}

/// The seam between the memory tools and the memory implementation.
///
/// Python parity: the `memory_tool_ops`/`coding_memory_tool_ops` functions, which own the
/// settings, the SQLite index and the embedding search. The harness only speaks this trait, so
/// `openjiuwen_memory` (or a test double) can provide the storage.
#[async_trait]
pub trait MemoryToolBackend: Send + Sync {
    /// Searches long-term memory and returns the Python `memory_search` result object, which
    /// the tool forwards to the model verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](openjiuwen_common::error::Error) when the search cannot run; the tool forwards the message.
    async fn search(&self, request: MemorySearchRequest) -> Result<Value>;

    /// Reads the window of `path` starting at `offset` (1-based; `0` means the first line).
    ///
    /// # Errors
    ///
    /// Returns [`Error`](openjiuwen_common::error::Error) when the file does not exist or cannot be read.
    async fn read(
        &self,
        scope: MemoryScope,
        path: &str,
        offset: usize,
        limit: Option<usize>,
    ) -> Result<MemoryReadData>;

    /// Writes `content` to `path`, appending when `append` is set.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](openjiuwen_common::error::Error) when the file cannot be written or violates the tree's rules.
    async fn write(
        &self,
        scope: MemoryScope,
        path: &str,
        content: &str,
        append: bool,
    ) -> Result<MemoryWriteData>;

    /// Replaces every `old_text` occurrence in `path` with `new_text`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](openjiuwen_common::error::Error) when the file cannot be read or written.
    async fn edit(
        &self,
        scope: MemoryScope,
        path: &str,
        old_text: &str,
        new_text: &str,
    ) -> Result<MemoryEditData>;
}
