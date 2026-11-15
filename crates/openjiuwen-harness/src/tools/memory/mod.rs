//! Port of `openjiuwen/harness/tools/memory.py` and `coding_memory.py`: the eight memory tools.
//!
//! Python wires each tool to a `MemoryToolContext`/`CodingMemoryToolContext` and calls the
//! `memory_tool_ops`/`coding_memory_tool_ops` functions. Here the storage and retrieval
//! implementation sits behind [`MemoryToolBackend`], so this module owns only the model-facing
//! contract: the cards, the Python validation messages, the payload shapes and the per-kind
//! [`Tool::render_for_llm`](openjiuwen_foundation::tool::Tool::render_for_llm) strings.
//!
//! [`WorkspaceMemoryBackend`] is the production backend: it reads and writes the workspace
//! markdown files with the Python path mapping, enforces the coding-memory front-matter rules and
//! answers `memory_search` with a deterministic offline keyword scan.
//!
//! The module is split into the backend seam (`backend`), the file-backed implementation
//! (`workspace_backend`), the kind enum (`kinds`) and the tool implementation (`tool`).

mod backend;
mod kinds;
mod tool;
mod workspace_backend;

#[cfg(test)]
mod tests;

pub use self::backend::{
    MemoryEditData, MemoryReadData, MemoryScope, MemorySearchRequest, MemoryToolBackend,
    MemoryWriteData,
};
pub use self::kinds::MemoryToolKind;
pub use self::tool::MemoryTool;
pub use self::workspace_backend::WorkspaceMemoryBackend;
