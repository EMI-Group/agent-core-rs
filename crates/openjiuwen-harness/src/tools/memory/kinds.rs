//! The eight model-facing memory tool kinds and their names, scopes and lookup.

use super::backend::MemoryScope;

/// One of the eight model-facing memory tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryToolKind {
    /// `memory_search`: search long-term memory.
    Search,
    /// `memory_get`: read a line slice of a memory file.
    Get,
    /// `write_memory`: overwrite or append a memory file.
    Write,
    /// `read_memory`: page through a memory file by offset/limit.
    Read,
    /// `edit_memory`: exact string replacement in a memory file.
    Edit,
    /// `coding_memory_read`: page through a coding-memory file.
    CodingRead,
    /// `coding_memory_write`: write a coding-memory file.
    CodingWrite,
    /// `coding_memory_edit`: exact string replacement in a coding-memory file.
    CodingEdit,
}

impl MemoryToolKind {
    /// Every kind, in Python registry order (`create_memory_tools`, then the coding tools).
    pub const ALL: [Self; 8] = [
        Self::Search,
        Self::Get,
        Self::Write,
        Self::Read,
        Self::Edit,
        Self::CodingRead,
        Self::CodingWrite,
        Self::CodingEdit,
    ];

    /// The model-facing tool name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Search => "memory_search",
            Self::Get => "memory_get",
            Self::Write => "write_memory",
            Self::Read => "read_memory",
            Self::Edit => "edit_memory",
            Self::CodingRead => "coding_memory_read",
            Self::CodingWrite => "coding_memory_write",
            Self::CodingEdit => "coding_memory_edit",
        }
    }

    /// The memory tree the kind addresses.
    ///
    /// Every kind addresses exactly one tree today; the `Option` mirrors the Python split
    /// where a future kind (for example a cross-tree search) could address none, so the
    /// `Some`-only body is intentional and not narrowed.
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn scope(self) -> Option<MemoryScope> {
        Some(match self {
            Self::Search | Self::Get | Self::Write | Self::Read | Self::Edit => MemoryScope::Memory,
            Self::CodingRead | Self::CodingWrite | Self::CodingEdit => MemoryScope::CodingMemory,
        })
    }

    /// Looks a kind up by model-facing name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}
