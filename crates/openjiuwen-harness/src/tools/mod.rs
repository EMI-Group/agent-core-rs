//! Model-facing coding tools, the tool registry, and shared tool helpers.
//!
//! Ports `openjiuwen/harness/tools/`: the shell, filesystem, todo, ask-user, memory, web,
//! media-metadata, cron, MCP-resource, tool-discovery and goal tools, plus their shared helpers.
//! The language-server navigation (`lsp`) and git-worktree (`worktree`) tools come from the
//! sibling [`crate::lsp`] and [`crate::worktree`] modules and are mounted here as well.
//! Upper layers only see [`ToolRegistry`], [`BuiltinTool`] and the tool seam traits.
//!
//! # Tool sets
//!
//! [`BuiltinTool::DEFAULT`] is the twenty tools [`ToolRegistry::new`] mounts; [`BuiltinTool::EXTENDED`]
//! appends the twelve extended tools (web, media metadata, cron, MCP resources, tool discovery,
//! goals, LSP navigation and git worktrees) and [`BuiltinTool::ALL`] is an alias of it.
//! [`tool_names`] names the default twenty, [`extended_tool_names`] the thirty-two.
//!
//! # Deferred tools
//!
//! Tools passed to [`ToolRegistry::with_deferred_tools`] are **not** mounted by
//! [`ToolRegistry::build`] and **not** listed by [`ToolRegistry::cards`]: the model finds them
//! with `tool_search` (or a host calls [`ToolRegistry::discover`]) and the host unions the cards
//! of the picked names ([`ToolRegistry::reveal`]) into the next model call. A deferred tool never
//! shadows a mounted one; with nothing deferred, `discover`/`reveal` cover every extended tool.
mod ask_user;
pub mod cron;
pub mod descriptions;
pub mod discovery;
pub mod goal;
pub mod mcp_resources;
pub mod media;
mod memory;
mod read;
mod read_state;
mod registry;
mod search;
mod shell;
mod support;
#[cfg(test)]
mod tests;
mod todo;
pub mod web;
mod write;

pub use registry::ToolRegistry;

pub use read_state::{
    file_fingerprint, first_unread_offset, merge_range, ranges_cover, EditGate, FileFingerprint,
    FileReadState, ReadCoverage, ReadStateRegistry, WriteGate,
};
pub use support::{
    compact_json, is_english, parse_args, render_fields, schema_for, tool_card, truncate_chars,
    truncation_marker, ToolEnv, ToolPayload, Truncated,
};

pub use ask_user::{AskOption, AskQuestion, AskUserArgs, AskUserTool, UserAsker};
pub use cron::{CronAction, CronArgs, CronJob, CronSchedule, CronScheduler, CronTool};
pub use discovery::{ToolIndex, ToolSearchArgs, ToolSearchHit, ToolSearchTool};
pub use goal::{
    GetCurrentGoalArgs, GetCurrentGoalTool, GoalAssessment, GoalAssessmentStatus, GoalRecord,
    GoalStatus, GoalStore, SubmitGoalReportArgs, SubmitGoalReportTool, TokenUsage,
};
pub use mcp_resources::{
    FakeMcpResourceProvider, ListMcpResourcesTool, McpResource, McpResourceContent,
    McpResourceProvider, ReadMcpResourceArgs, ReadMcpResourceTool,
};
pub use media::{
    AudioMetadata, AudioMetadataArgs, AudioMetadataTool, Cursor, Failure, ImageMetadata,
    ImageMetadataArgs, ImageMetadataTool, MediaKind, ParseResult,
};
pub use memory::{
    MemoryEditData, MemoryReadData, MemoryScope, MemorySearchRequest, MemoryTool,
    MemoryToolBackend, MemoryToolKind, MemoryWriteData,
};
pub use read::{ReadFileArgs, ReadFileTool};
pub use search::{
    GlobArgs, GlobTool, GrepArgs, GrepOutputMode, GrepTool, ListDirArgs, ListDirTool,
};
pub use shell::{BashArgs, BashTool};
pub use todo::{
    session_key, TodoAction, TodoCreateArgs, TodoCreateTool, TodoData, TodoGetArgs, TodoGetTool,
    TodoListTool, TodoModifyArgs, TodoModifyTool, TodoStore,
};
pub use web::{
    FakeSearchBackend, FetchRequest, FetchResponse, FetchWebpageArgs, FetchWebpageTool,
    HttpFetcher, ReqwestFetcher, SearchBackend, SearchResult, WebSearchArgs, WebSearchTool,
};
pub use write::{EditFileArgs, EditFileTool, WriteFileArgs, WriteFileTool};

/// A built-in model-facing tool of the `DeepAgent`.
///
/// The variant order is the registry order and matches the order of [`tool_names`]; the
/// model-facing names are the Python tool names verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinTool {
    /// Execute a shell command (`bash`).
    Bash,
    /// Read a text file with line numbers (`read_file`).
    ReadFile,
    /// Create or overwrite a file (`write_file`).
    WriteFile,
    /// Replace a string inside an existing file (`edit_file`).
    EditFile,
    /// Find files by glob pattern (`glob`).
    Glob,
    /// List the entries of one directory (`list_files`).
    ListFiles,
    /// Search file contents with a regular expression (`grep`).
    Grep,
    /// Create one or more todo items (`todo_create`).
    TodoCreate,
    /// List the active todo items (`todo_list`).
    TodoList,
    /// Fetch one todo item by id (`todo_get`).
    TodoGet,
    /// Update, delete or reorder todo items (`todo_modify`).
    TodoModify,
    /// Ask the human user one or more questions (`ask_user`).
    AskUser,
    /// Search long-term memory (`memory_search`).
    MemorySearch,
    /// Read long-term memory by path (`memory_get`).
    MemoryGet,
    /// Write a long-term memory file (`write_memory`).
    WriteMemory,
    /// Read a long-term memory file (`read_memory`).
    ReadMemory,
    /// Edit a long-term memory file (`edit_memory`).
    EditMemory,
    /// Read a coding-memory file (`coding_memory_read`).
    CodingMemoryRead,
    /// Write a coding-memory file (`coding_memory_write`).
    CodingMemoryWrite,
    /// Edit a coding-memory file (`coding_memory_edit`).
    CodingMemoryEdit,
    /// Fetch a web page and return its title and extracted text (`fetch_webpage`).
    FetchWebpage,
    /// Search the web through the injected provider (`web_search`).
    WebSearch,
    /// Report the dimensions and container fields of an image file (`image_metadata`).
    ImageMetadata,
    /// Report the duration and stream parameters of an audio file (`audio_metadata`).
    AudioMetadata,
    /// Create, list and cancel persisted cron jobs (`cron`).
    Cron,
    /// List the resources of the bound MCP server (`list_mcp_resources`).
    ListMcpResources,
    /// Read one resource of the bound MCP server (`read_mcp_resource`).
    ReadMcpResource,
    /// Search the discoverable tool set by intent (`tool_search`).
    ToolSearch,
    /// Read the current session goal (`get_current_goal`).
    GetCurrentGoal,
    /// Submit the structured result of the current goal attempt (`submit_goal_report`).
    SubmitGoalReport,
    /// Navigate code through a language server (`lsp`).
    Lsp,
    /// Create and manage git worktrees (`worktree`).
    Worktree,
}

impl BuiltinTool {
    /// The default built-in tool set, in registry order: the twenty tools [`ToolRegistry::new`]
    /// mounts, which [`Self::EXTENDED`] extends.
    pub const DEFAULT: [Self; 20] = [
        Self::Bash,
        Self::ReadFile,
        Self::WriteFile,
        Self::EditFile,
        Self::Glob,
        Self::ListFiles,
        Self::Grep,
        Self::TodoCreate,
        Self::TodoList,
        Self::TodoGet,
        Self::TodoModify,
        Self::AskUser,
        Self::MemorySearch,
        Self::MemoryGet,
        Self::WriteMemory,
        Self::ReadMemory,
        Self::EditMemory,
        Self::CodingMemoryRead,
        Self::CodingMemoryWrite,
        Self::CodingMemoryEdit,
    ];

    /// Every built-in tool, in registry order: [`Self::DEFAULT`] then the twelve extended tools.
    pub const EXTENDED: [Self; 32] = [
        Self::Bash,
        Self::ReadFile,
        Self::WriteFile,
        Self::EditFile,
        Self::Glob,
        Self::ListFiles,
        Self::Grep,
        Self::TodoCreate,
        Self::TodoList,
        Self::TodoGet,
        Self::TodoModify,
        Self::AskUser,
        Self::MemorySearch,
        Self::MemoryGet,
        Self::WriteMemory,
        Self::ReadMemory,
        Self::EditMemory,
        Self::CodingMemoryRead,
        Self::CodingMemoryWrite,
        Self::CodingMemoryEdit,
        Self::FetchWebpage,
        Self::WebSearch,
        Self::ImageMetadata,
        Self::AudioMetadata,
        Self::Cron,
        Self::ListMcpResources,
        Self::ReadMcpResource,
        Self::ToolSearch,
        Self::GetCurrentGoal,
        Self::SubmitGoalReport,
        Self::Lsp,
        Self::Worktree,
    ];

    /// Every built-in tool, in registry order; an alias of [`Self::EXTENDED`].
    pub const ALL: [Self; 32] = Self::EXTENDED;

    /// The model-facing name of this tool.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::ReadFile => "read_file",
            Self::WriteFile => "write_file",
            Self::EditFile => "edit_file",
            Self::Glob => "glob",
            Self::ListFiles => "list_files",
            Self::Grep => "grep",
            Self::TodoCreate => "todo_create",
            Self::TodoList => "todo_list",
            Self::TodoGet => "todo_get",
            Self::TodoModify => "todo_modify",
            Self::AskUser => "ask_user",
            Self::MemorySearch => "memory_search",
            Self::MemoryGet => "memory_get",
            Self::WriteMemory => "write_memory",
            Self::ReadMemory => "read_memory",
            Self::EditMemory => "edit_memory",
            Self::CodingMemoryRead => "coding_memory_read",
            Self::CodingMemoryWrite => "coding_memory_write",
            Self::CodingMemoryEdit => "coding_memory_edit",
            Self::FetchWebpage => "fetch_webpage",
            Self::WebSearch => "web_search",
            Self::ImageMetadata => "image_metadata",
            Self::AudioMetadata => "audio_metadata",
            Self::Cron => "cron",
            Self::ListMcpResources => "list_mcp_resources",
            Self::ReadMcpResource => "read_mcp_resource",
            Self::ToolSearch => "tool_search",
            Self::GetCurrentGoal => "get_current_goal",
            Self::SubmitGoalReport => "submit_goal_report",
            Self::Lsp => "lsp",
            Self::Worktree => "worktree",
        }
    }

    /// The tool whose model-facing name is exactly `name`, if any; every one of the thirty-two
    /// names in [`Self::ALL`] resolves.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tool| tool.name() == name)
    }

    /// Whether the tool may run concurrently with sibling tool calls.
    ///
    /// The file-write tools, `cron` (job-list mutations), `submit_goal_report` (goal-record
    /// mutations) and `worktree` (repository/registry mutations) are serialised; every other
    /// built-in tool is parallel-safe.
    #[must_use]
    pub const fn parallel_safe(self) -> bool {
        !matches!(
            self,
            Self::WriteFile | Self::EditFile | Self::Cron | Self::SubmitGoalReport | Self::Worktree
        )
    }
}

/// The model-facing names of the default tool set, in registry order.
#[must_use]
pub fn tool_names() -> &'static [&'static str] {
    // The default set is the prefix of the extended set, so the two cannot drift apart.
    &extended_tool_names()[..BuiltinTool::DEFAULT.len()]
}

/// The model-facing names of every built-in tool, in [`BuiltinTool::EXTENDED`] order; the first
/// twenty entries are exactly [`tool_names`].
#[must_use]
pub fn extended_tool_names() -> &'static [&'static str] {
    &[
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "list_files",
        "grep",
        "todo_create",
        "todo_list",
        "todo_get",
        "todo_modify",
        "ask_user",
        "memory_search",
        "memory_get",
        "write_memory",
        "read_memory",
        "edit_memory",
        "coding_memory_read",
        "coding_memory_write",
        "coding_memory_edit",
        "fetch_webpage",
        "web_search",
        "image_metadata",
        "audio_metadata",
        "cron",
        "list_mcp_resources",
        "read_mcp_resource",
        "tool_search",
        "get_current_goal",
        "submit_goal_report",
        "lsp",
        "worktree",
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod registry_tests {
    use super::*;
    use crate::workspace::Workspace;

    const EXPECTED_NAMES: [&str; 20] = [
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "list_files",
        "grep",
        "todo_create",
        "todo_list",
        "todo_get",
        "todo_modify",
        "ask_user",
        "memory_search",
        "memory_get",
        "write_memory",
        "read_memory",
        "edit_memory",
        "coding_memory_read",
        "coding_memory_write",
        "coding_memory_edit",
    ];

    #[test]
    fn builtin_tool_sets_have_the_documented_shapes() {
        assert_eq!(BuiltinTool::DEFAULT.len(), 20);
        assert_eq!(BuiltinTool::EXTENDED.len(), 32);
        assert_eq!(BuiltinTool::ALL, BuiltinTool::EXTENDED);
        assert_eq!(&BuiltinTool::EXTENDED[..20], &BuiltinTool::DEFAULT[..]);

        let names: Vec<&str> = BuiltinTool::DEFAULT
            .iter()
            .map(|tool| tool.name())
            .collect();
        assert_eq!(names, EXPECTED_NAMES.to_vec());
        let extended: Vec<&str> = BuiltinTool::EXTENDED
            .iter()
            .map(|tool| tool.name())
            .collect();
        assert_eq!(extended[..20], EXPECTED_NAMES);
        assert_eq!(extended[20], "fetch_webpage");
        assert_eq!(extended[29], "submit_goal_report");
        assert_eq!(extended[30], "lsp");
        assert_eq!(extended[31], "worktree");
    }

    #[test]
    fn tool_names_match_registry_order() {
        assert_eq!(tool_names(), &EXPECTED_NAMES);
        assert_eq!(&extended_tool_names()[..20], tool_names());
        assert_eq!(extended_tool_names().len(), BuiltinTool::EXTENDED.len());
        let names: Vec<&str> = BuiltinTool::EXTENDED
            .iter()
            .map(|tool| tool.name())
            .collect();
        assert_eq!(names, extended_tool_names().to_vec());
    }

    #[test]
    fn from_name_round_trips_every_name() {
        for name in extended_tool_names() {
            let tool = BuiltinTool::from_name(name).expect("known tool name");
            assert_eq!(tool.name(), *name, "round-trip mismatch for {name}");
        }
        assert_eq!(BuiltinTool::from_name("nope"), None);
    }

    #[test]
    fn only_the_mutating_tools_are_not_parallel_safe() {
        for tool in BuiltinTool::ALL {
            let expected = !matches!(
                tool,
                BuiltinTool::WriteFile
                    | BuiltinTool::EditFile
                    | BuiltinTool::Cron
                    | BuiltinTool::SubmitGoalReport
                    | BuiltinTool::Worktree
            );
            assert_eq!(tool.parallel_safe(), expected, "for {}", tool.name());
        }
        for name in [
            "write_file",
            "edit_file",
            "cron",
            "submit_goal_report",
            "worktree",
        ] {
            assert!(
                !BuiltinTool::from_name(name).unwrap().parallel_safe(),
                "{name}"
            );
        }
        assert!(BuiltinTool::from_name("fetch_webpage")
            .unwrap()
            .parallel_safe());
        assert!(BuiltinTool::from_name("tool_search")
            .unwrap()
            .parallel_safe());
        assert!(BuiltinTool::from_name("lsp").unwrap().parallel_safe());
    }

    #[test]
    fn registry_new_builds_a_bash_card() {
        let dir = tempfile::tempdir().expect("temp workspace");
        let registry = ToolRegistry::new(Workspace::new(dir.path()));
        let card = registry.card(BuiltinTool::Bash, "en");
        assert_eq!(card.name, "bash");
        assert_eq!(card.id.len(), 32, "card id is a uuid hex: {}", card.id);
        assert!(card.id.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(card.parallel_safe);
        assert_eq!(card.description, descriptions::description("bash", "en"));
    }
}
