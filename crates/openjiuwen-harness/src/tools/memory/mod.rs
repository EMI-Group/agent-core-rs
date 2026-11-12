//! Port of `openjiuwen/harness/tools/memory.py` and `coding_memory.py`: the eight memory tools.
//!
//! Python wires each tool to a `MemoryToolContext`/`CodingMemoryToolContext` and calls the
//! `memory_tool_ops`/`coding_memory_tool_ops` functions. Here the storage and retrieval
//! implementation sits behind [`MemoryToolBackend`] (the SQLite/embedding hybrid lives in
//! `openjiuwen_memory`), so this module owns only the model-facing contract: the cards, the
//! Python validation messages, the payload shapes and the per-kind [`Tool::render_for_llm`]
//! strings.
//!
//! The module is split into the backend seam (`backend`), the kind enum (`kinds`) and the
//! tool implementation (`tool`).

mod backend;
mod kinds;
mod tool;

#[cfg(test)]
mod tests;

pub use self::backend::{
    MemoryEditData, MemoryReadData, MemoryScope, MemorySearchRequest, MemoryToolBackend,
    MemoryWriteData,
};
pub use self::kinds::MemoryToolKind;
pub use self::tool::MemoryTool;
