//! Model-facing coding tools, the tool registry, and shared tool helpers.
//!
//! Ports `openjiuwen/harness/tools/`: the shell, filesystem, todo, ask-user, memory, web,
//! media-metadata, cron, MCP-resource, tool-discovery and goal tools, plus their shared helpers.
//! Upper layers only see [`ToolRegistry`], [`BuiltinTool`] and the tool seam traits.
//!
//! # Tool sets
//!
//! [`BuiltinTool::DEFAULT`] is the twenty tools [`ToolRegistry::new`] mounts; [`BuiltinTool::EXTENDED`]
//! appends the ten extended tools (web, media metadata, cron, MCP resources, tool discovery,
//! goals) and [`BuiltinTool::ALL`] is an alias of it. [`tool_names`] names the default twenty,
//! [`extended_tool_names`] the thirty.
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
mod search;
mod shell;
mod support;
#[cfg(test)]
mod tests;
mod todo;
pub mod web;
mod write;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_sysop::operation::{FsOperation, ShellOperation};
use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

use crate::permission::PermissionEngine;
use crate::workspace::Workspace;

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

    /// Every built-in tool, in registry order: [`Self::DEFAULT`] then the ten extended tools.
    pub const EXTENDED: [Self; 30] = [
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
    ];

    /// Every built-in tool, in registry order; an alias of [`Self::EXTENDED`].
    pub const ALL: [Self; 30] = Self::EXTENDED;

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
        }
    }

    /// The tool whose model-facing name is exactly `name`, if any; every one of the thirty names
    /// in [`Self::ALL`] resolves.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tool| tool.name() == name)
    }

    /// Whether the tool may run concurrently with sibling tool calls.
    ///
    /// The file-write tools, `cron` (job-list mutations) and `submit_goal_report` (goal-record
    /// mutations) are serialised; every other built-in tool is parallel-safe.
    #[must_use]
    pub const fn parallel_safe(self) -> bool {
        !matches!(
            self,
            Self::WriteFile | Self::EditFile | Self::Cron | Self::SubmitGoalReport
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
    ]
}

/// Maps a memory [`BuiltinTool`] variant to its [`MemoryToolKind`].
///
/// # Panics
///
/// Panics only when called with a non-memory variant, which the callers never do; the helper
/// exists to keep the eight `build`/`card` arms explicit without duplicating the mapping.
fn memory_kind(tool: BuiltinTool) -> MemoryToolKind {
    match tool {
        BuiltinTool::MemorySearch => MemoryToolKind::Search,
        BuiltinTool::MemoryGet => MemoryToolKind::Get,
        BuiltinTool::WriteMemory => MemoryToolKind::Write,
        BuiltinTool::ReadMemory => MemoryToolKind::Read,
        BuiltinTool::EditMemory => MemoryToolKind::Edit,
        BuiltinTool::CodingMemoryRead => MemoryToolKind::CodingRead,
        BuiltinTool::CodingMemoryWrite => MemoryToolKind::CodingWrite,
        BuiltinTool::CodingMemoryEdit => MemoryToolKind::CodingEdit,
        other => unreachable!("{} is not a memory tool", other.name()),
    }
}

/// The sibling-shared caches of one [`ToolRegistry::build`] pass, each populated on first use.
#[derive(Default)]
struct BuildCaches {
    /// The todo store shared by the four todo tools.
    todo_store: Option<Arc<TodoStore>>,
    /// The cron scheduler behind the `cron` tool.
    cron_scheduler: Option<Arc<cron::CronScheduler>>,
    /// The goal store shared by the two goal tools.
    goal_store: Option<Arc<goal::GoalStore>>,
    /// The discovery index behind `tool_search`.
    tool_index: Option<Arc<discovery::ToolIndex>>,
}

/// The built-in tool set of one agent, with its workspace, backends and shared read state.
///
/// The registry owns the [`ToolEnv`] path/permission gate, the sys-operation backends, the
/// read-before-write registry, the todo directory and the extended tools' seams;
/// [`Self::build`] materialises the tool implementations and [`Self::cards`] their cards.
/// [`Self::new`] mounts [`BuiltinTool::DEFAULT`]; the extended tools are selected through
/// [`Self::with_tools`], and tools passed to [`Self::with_deferred_tools`] stay out of both
/// [`Self::build`] and [`Self::cards`] until the host reveals them with [`Self::reveal`].
pub struct ToolRegistry {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The sys-operation facade the local backends came from, when one was built.
    sys_operation: Option<Arc<SysOperation>>,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// The shell backend, when one is configured.
    shell: Option<Arc<dyn ShellOperation>>,
    /// Read-before-write bookkeeping shared by the filesystem tools.
    read_state: Arc<ReadStateRegistry>,
    /// Override for the todo file directory; `None` uses the workspace `todo` node.
    todo_dir: Option<PathBuf>,
    /// The user-interaction handler the `ask_user` tool delegates to, when one is wired.
    user_asker: Option<Arc<dyn UserAsker>>,
    /// The memory backend the eight memory tools delegate to, when one is wired.
    memory_backend: Option<Arc<dyn MemoryToolBackend>>,
    /// The HTTP seam behind `fetch_webpage`, when one is wired.
    http_fetcher: Option<Arc<dyn web::HttpFetcher>>,
    /// The search provider behind `web_search`, when one is wired.
    search: Option<Arc<dyn web::SearchBackend>>,
    /// The MCP resource provider behind the two resource tools, when one is wired.
    mcp_provider: Option<Arc<dyn mcp_resources::McpResourceProvider>>,
    /// The cron scheduler behind the `cron` tool, when one is wired.
    cron_scheduler: Option<Arc<cron::CronScheduler>>,
    /// The goal store behind the two goal tools, when one is wired.
    goal_store: Option<Arc<goal::GoalStore>>,
    /// Tools held back from [`ToolRegistry::build`]/[`ToolRegistry::cards`] until discovered.
    deferred: Vec<BuiltinTool>,
    /// The tools this registry builds, in registry order.
    tools: Vec<BuiltinTool>,
}

impl ToolRegistry {
    /// Builds a registry over `workspace`, deriving a local sys-operation backend from it.
    ///
    /// When the local backend cannot be initialised the registry still exists with no
    /// filesystem or shell backend; the failure is logged and surfaced by the tools.
    #[must_use]
    pub fn new(workspace: Workspace) -> Self {
        let root = workspace.root_path.to_string_lossy().to_string();
        let (sys_operation, fs, shell) =
            match SysOperation::local(LocalWorkConfig::unrestricted(root)) {
                Ok(operation) => {
                    let operation = Arc::new(operation);
                    let fs = operation.fs();
                    let shell = operation.shell();
                    (Some(operation), fs, shell)
                }
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "failed to initialise the local sys-operation backend"
                    );
                    (None, None, None)
                }
            };
        Self {
            env: ToolEnv::new(workspace),
            sys_operation,
            fs,
            shell,
            read_state: ReadStateRegistry::new(),
            todo_dir: None,
            user_asker: None,
            memory_backend: None,
            http_fetcher: None,
            search: None,
            mcp_provider: None,
            cron_scheduler: None,
            goal_store: None,
            deferred: Vec::new(),
            tools: BuiltinTool::DEFAULT.to_vec(),
        }
    }

    /// Uses `operation` as the backend source, refreshing the filesystem and shell handles.
    #[must_use]
    pub fn with_sys_operation(mut self, operation: Arc<SysOperation>) -> Self {
        self.fs = operation.fs();
        self.shell = operation.shell();
        self.sys_operation = Some(operation);
        self
    }

    /// Overrides the filesystem backend.
    #[must_use]
    pub fn with_fs_operation(mut self, operation: Arc<dyn FsOperation>) -> Self {
        self.fs = Some(operation);
        self
    }

    /// Overrides the shell backend.
    #[must_use]
    pub fn with_shell_operation(mut self, operation: Arc<dyn ShellOperation>) -> Self {
        self.shell = Some(operation);
        self
    }

    /// Wires the permission engine the tools consult before touching a path.
    #[must_use]
    pub fn with_permissions(mut self, engine: Arc<PermissionEngine>) -> Self {
        self.env = self.env.clone().with_permission(engine);
        self
    }

    /// Wires the user-interaction handler behind the `ask_user` tool, replacing any previous
    /// handler.
    #[must_use]
    pub fn with_user_asker(mut self, asker: Arc<dyn UserAsker>) -> Self {
        self.user_asker = Some(asker);
        self
    }

    /// Wires the memory backend behind the eight memory tools, replacing any previous backend.
    #[must_use]
    pub fn with_memory(mut self, backend: Arc<dyn MemoryToolBackend>) -> Self {
        self.memory_backend = Some(backend);
        self
    }

    /// Wires the HTTP seam behind `fetch_webpage`; without one it uses [`web::ReqwestFetcher`].
    #[must_use]
    pub fn with_http_fetcher(mut self, fetcher: Arc<dyn web::HttpFetcher>) -> Self {
        self.http_fetcher = Some(fetcher);
        self
    }

    /// Wires the search provider behind `web_search`; without one it reports a typed failure.
    #[must_use]
    pub fn with_search_backend(mut self, backend: Arc<dyn web::SearchBackend>) -> Self {
        self.search = Some(backend);
        self
    }

    /// Wires the MCP resource provider behind the two resource tools; without one they fail.
    #[must_use]
    pub fn with_mcp_resources(
        mut self,
        provider: Arc<dyn mcp_resources::McpResourceProvider>,
    ) -> Self {
        self.mcp_provider = Some(provider);
        self
    }

    /// Wires the `cron` scheduler; without one it defaults to [`cron::CronScheduler::for_workspace`].
    #[must_use]
    pub fn with_cron_scheduler(mut self, scheduler: Arc<cron::CronScheduler>) -> Self {
        self.cron_scheduler = Some(scheduler);
        self
    }

    /// Wires the goal store; without one the goal tools default to [`goal::GoalStore::for_workspace`].
    #[must_use]
    pub fn with_goal_store(mut self, store: Arc<goal::GoalStore>) -> Self {
        self.goal_store = Some(store);
        self
    }

    /// Holds `tools` back as deferred tools, replacing any previous deferred list.
    ///
    /// They are not mounted by [`Self::build`] and not listed by [`Self::cards`]; the model finds
    /// them with `tool_search` (or [`Self::discover`]) and the host reveals them with
    /// [`Self::reveal`]. Deferring a mounted tool keeps it mounted.
    #[must_use]
    pub fn with_deferred_tools(mut self, tools: impl IntoIterator<Item = BuiltinTool>) -> Self {
        self.deferred = tools.into_iter().collect();
        self
    }

    /// Sets the description language used for the rendered tool cards.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.env = self.env.clone().with_language(language);
        self
    }

    /// Overrides the directory the todo store persists into.
    #[must_use]
    pub fn with_todo_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.todo_dir = Some(dir.into());
        self
    }

    /// Restricts the registry to `tools`, keeping their registry order.
    #[must_use]
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = BuiltinTool>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    /// Builds the tool implementations of this registry.
    ///
    /// Every selected [`BuiltinTool`] yields exactly one [`Tool`], in registry order; sibling
    /// tools share one todo store, read-state registry, cron scheduler, goal store and discovery
    /// index. Deferred tools are never mounted here.
    #[must_use]
    pub fn build(&self) -> Vec<Arc<dyn Tool>> {
        let language = self.env.language();
        let mut caches = BuildCaches::default();
        self.tools
            .iter()
            .map(|tool| match *tool {
                BuiltinTool::Bash => Arc::new(BashTool::new(self.env.clone(), self.shell.clone())),
                BuiltinTool::ReadFile | BuiltinTool::WriteFile | BuiltinTool::EditFile => {
                    self.build_filesystem_tool(*tool)
                }
                BuiltinTool::Glob | BuiltinTool::ListFiles | BuiltinTool::Grep => {
                    self.build_search_tool(*tool)
                }
                BuiltinTool::TodoCreate
                | BuiltinTool::TodoList
                | BuiltinTool::TodoGet
                | BuiltinTool::TodoModify => self.build_todo_tool(*tool, language, &mut caches),
                BuiltinTool::AskUser => {
                    Arc::new(AskUserTool::new(self.user_asker.clone(), language))
                }
                BuiltinTool::MemorySearch
                | BuiltinTool::MemoryGet
                | BuiltinTool::WriteMemory
                | BuiltinTool::ReadMemory
                | BuiltinTool::EditMemory
                | BuiltinTool::CodingMemoryRead
                | BuiltinTool::CodingMemoryWrite
                | BuiltinTool::CodingMemoryEdit => self.build_memory_tool(*tool, language),
                BuiltinTool::FetchWebpage => Arc::new(web::FetchWebpageTool::new(
                    self.env.clone(),
                    Some(self.fetcher()),
                    language,
                )),
                BuiltinTool::WebSearch => {
                    Arc::new(web::WebSearchTool::new(self.search.clone(), language))
                }
                BuiltinTool::ImageMetadata => {
                    Arc::new(media::ImageMetadataTool::new(self.env.clone(), language))
                }
                BuiltinTool::AudioMetadata => {
                    Arc::new(media::AudioMetadataTool::new(self.env.clone(), language))
                }
                BuiltinTool::Cron => Arc::new(cron::CronTool::new(
                    self.cron_scheduler(&mut caches.cron_scheduler),
                    language,
                )),
                BuiltinTool::ListMcpResources => Arc::new(
                    mcp_resources::ListMcpResourcesTool::new(self.mcp_provider.clone(), language),
                ),
                BuiltinTool::ReadMcpResource => Arc::new(mcp_resources::ReadMcpResourceTool::new(
                    self.mcp_provider.clone(),
                    language,
                )),
                BuiltinTool::ToolSearch => Arc::new(discovery::ToolSearchTool::new(
                    self.tool_index(&mut caches.tool_index, language),
                    language,
                )),
                BuiltinTool::GetCurrentGoal => Arc::new(goal::GetCurrentGoalTool::new(
                    self.goal_store(&mut caches.goal_store),
                    language,
                )),
                BuiltinTool::SubmitGoalReport => Arc::new(goal::SubmitGoalReportTool::new(
                    self.goal_store(&mut caches.goal_store),
                    language,
                )),
            })
            .collect()
    }

    /// Builds one of the three filesystem tools over the registry's shared read-before-write state.
    ///
    /// Any variant other than `write_file`/`edit_file` falls back to `read_file`.
    fn build_filesystem_tool(&self, tool: BuiltinTool) -> Arc<dyn Tool> {
        let env = self.env.clone();
        let fs = self.fs.clone();
        let read_state = self.read_state.clone();
        match tool {
            BuiltinTool::WriteFile => Arc::new(WriteFileTool::new(env, fs, read_state)),
            BuiltinTool::EditFile => Arc::new(EditFileTool::new(env, fs, read_state)),
            _ => Arc::new(ReadFileTool::new(env, fs, read_state)),
        }
    }

    /// Builds one of the three discovery tools (`list_files`, `grep`, else `glob`).
    fn build_search_tool(&self, tool: BuiltinTool) -> Arc<dyn Tool> {
        match tool {
            BuiltinTool::ListFiles => Arc::new(ListDirTool::new(self.env.clone(), self.fs.clone())),
            BuiltinTool::Grep => Arc::new(GrepTool::new(self.env.clone())),
            _ => Arc::new(GlobTool::new(self.env.clone(), self.fs.clone())),
        }
    }

    /// Builds one of the four todo tools over the store shared by this build pass.
    ///
    /// Any variant other than `todo_list`/`todo_get`/`todo_modify` falls back to `todo_create`.
    fn build_todo_tool(
        &self,
        tool: BuiltinTool,
        language: &str,
        caches: &mut BuildCaches,
    ) -> Arc<dyn Tool> {
        let store = self.todo_store(&mut caches.todo_store);
        match tool {
            BuiltinTool::TodoList => Arc::new(TodoListTool::new(store, language)),
            BuiltinTool::TodoGet => Arc::new(TodoGetTool::new(store, language)),
            BuiltinTool::TodoModify => Arc::new(TodoModifyTool::new(store, language)),
            _ => Arc::new(TodoCreateTool::new(store, language)),
        }
    }

    /// Builds one of the eight memory tools, whose kind [`memory_kind`] maps from the variant.
    fn build_memory_tool(&self, tool: BuiltinTool, language: &str) -> Arc<dyn Tool> {
        Arc::new(MemoryTool::new(
            memory_kind(tool),
            self.memory_backend.clone(),
            language,
        ))
    }

    /// The HTTP seam of `fetch_webpage`: the wired fetcher, else a [`web::ReqwestFetcher`].
    fn fetcher(&self) -> Arc<dyn web::HttpFetcher> {
        self.http_fetcher
            .clone()
            .unwrap_or_else(|| Arc::new(web::ReqwestFetcher::new()))
    }

    /// Returns the cron scheduler of this registry, creating and caching it on first use.
    fn cron_scheduler(
        &self,
        cache: &mut Option<Arc<cron::CronScheduler>>,
    ) -> Arc<cron::CronScheduler> {
        cache
            .get_or_insert_with(|| match &self.cron_scheduler {
                Some(scheduler) => Arc::clone(scheduler),
                None => cron::CronScheduler::for_workspace(self.env.workspace()),
            })
            .clone()
    }

    /// Returns the goal store of this registry, creating and caching it on first use.
    fn goal_store(&self, cache: &mut Option<Arc<goal::GoalStore>>) -> Arc<goal::GoalStore> {
        cache
            .get_or_insert_with(|| match &self.goal_store {
                Some(store) => Arc::clone(store),
                None => goal::GoalStore::for_workspace(self.env.workspace()),
            })
            .clone()
    }

    /// Returns the discovery index of this registry, building and caching it on first use.
    fn tool_index(
        &self,
        cache: &mut Option<Arc<discovery::ToolIndex>>,
        language: &str,
    ) -> Arc<discovery::ToolIndex> {
        cache
            .get_or_insert_with(|| {
                Arc::new(discovery::ToolIndex::from_cards(
                    &self.discovery_cards(language),
                ))
            })
            .clone()
    }

    /// Returns the todo store of this registry, creating and caching it on first use.
    fn todo_store(&self, cache: &mut Option<Arc<TodoStore>>) -> Arc<TodoStore> {
        cache
            .get_or_insert_with(|| match &self.todo_dir {
                Some(dir) => TodoStore::new(dir.clone()),
                None => TodoStore::for_workspace(self.env.workspace()),
            })
            .clone()
    }

    /// Builds the model-facing cards of every selected tool, in registry order.
    #[must_use]
    pub fn cards(&self, language: &str) -> Vec<ToolCard> {
        self.tools
            .iter()
            .map(|tool| self.card(*tool, language))
            .collect()
    }

    /// Builds the model-facing card of one tool.
    #[must_use]
    pub fn card(&self, tool: BuiltinTool, language: &str) -> ToolCard {
        match tool {
            BuiltinTool::Bash => BashTool::card_for(language),
            BuiltinTool::ReadFile => ReadFileTool::card_for(language),
            BuiltinTool::WriteFile => WriteFileTool::card_for(language),
            BuiltinTool::EditFile => EditFileTool::card_for(language),
            BuiltinTool::Glob => GlobTool::card_for(language),
            BuiltinTool::ListFiles => ListDirTool::card_for(language),
            BuiltinTool::Grep => GrepTool::card_for(language),
            BuiltinTool::TodoCreate => TodoCreateTool::card_for(language),
            BuiltinTool::TodoList => TodoListTool::card_for(language),
            BuiltinTool::TodoGet => TodoGetTool::card_for(language),
            BuiltinTool::TodoModify => TodoModifyTool::card_for(language),
            BuiltinTool::AskUser => AskUserTool::card_for(language),
            BuiltinTool::MemorySearch
            | BuiltinTool::MemoryGet
            | BuiltinTool::WriteMemory
            | BuiltinTool::ReadMemory
            | BuiltinTool::EditMemory
            | BuiltinTool::CodingMemoryRead
            | BuiltinTool::CodingMemoryWrite
            | BuiltinTool::CodingMemoryEdit => MemoryTool::card_for(memory_kind(tool), language),
            BuiltinTool::FetchWebpage => web::FetchWebpageTool::card_for(language),
            BuiltinTool::WebSearch => web::WebSearchTool::card_for(language),
            BuiltinTool::ImageMetadata => media::ImageMetadataTool::card_for(language),
            BuiltinTool::AudioMetadata => media::AudioMetadataTool::card_for(language),
            BuiltinTool::Cron => cron::CronTool::card_for(language),
            BuiltinTool::ListMcpResources => {
                mcp_resources::ListMcpResourcesTool::card_for(language)
            }
            BuiltinTool::ReadMcpResource => mcp_resources::ReadMcpResourceTool::card_for(language),
            BuiltinTool::ToolSearch => discovery::ToolSearchTool::card_for(language),
            BuiltinTool::GetCurrentGoal => goal::GetCurrentGoalTool::card_for(language),
            BuiltinTool::SubmitGoalReport => goal::SubmitGoalReportTool::card_for(language),
        }
    }

    /// The tools the discovery API searches: the deferred set when one is configured, otherwise
    /// every extended tool.
    fn discovery_set(&self) -> &[BuiltinTool] {
        if self.deferred.is_empty() {
            &BuiltinTool::EXTENDED
        } else {
            &self.deferred
        }
    }

    /// The cards the discovery index is built from, in registry order.
    fn discovery_cards(&self, language: &str) -> Vec<ToolCard> {
        let candidates = self.discovery_set();
        BuiltinTool::EXTENDED
            .iter()
            .filter(|tool| candidates.contains(tool))
            .map(|tool| self.card(*tool, language))
            .collect()
    }

    /// Searches the discoverable tools with `query` and returns at most `limit` hits.
    ///
    /// Ranking is BM25 over `"{name} {description}"` ([`discovery::ToolIndex`]), exact-name hits
    /// first; the index covers the deferred set when one is configured and every extended tool
    /// otherwise. The `tool_search` tool exposes the same search to the model.
    #[must_use]
    pub fn discover(&self, query: &str, limit: usize) -> Vec<discovery::ToolSearchHit> {
        discovery::ToolIndex::from_cards(&self.discovery_cards(self.env.language()))
            .search(query, limit)
    }

    /// Returns the cards of the named discoverable tools, in registry order.
    ///
    /// Names outside the discoverable set are skipped, so a model cannot conjure a tool the host
    /// did not offer. A host unions the returned cards into the next model call's tool list.
    #[must_use]
    pub fn reveal(&self, names: &[&str]) -> Vec<ToolCard> {
        let candidates = self.discovery_set();
        BuiltinTool::EXTENDED
            .iter()
            .filter(|tool| candidates.contains(tool) && names.contains(&tool.name()))
            .map(|tool| self.card(*tool, self.env.language()))
            .collect()
    }

    /// The cards of the deferred tools, in registry order; empty when nothing is deferred.
    #[must_use]
    pub fn deferred_cards(&self, language: &str) -> Vec<ToolCard> {
        BuiltinTool::EXTENDED
            .iter()
            .filter(|tool| self.deferred.contains(tool))
            .map(|tool| self.card(*tool, language))
            .collect()
    }

    /// The deferred tools, in the order they were configured.
    #[must_use]
    pub fn deferred_tools(&self) -> &[BuiltinTool] {
        &self.deferred
    }

    /// The path resolution and permission environment shared by the tools.
    #[must_use]
    pub fn env(&self) -> &ToolEnv {
        &self.env
    }

    /// The read-before-write registry shared by the filesystem tools.
    #[must_use]
    pub fn read_state(&self) -> &Arc<ReadStateRegistry> {
        &self.read_state
    }

    /// The sys-operation facade the backends came from, when one was configured.
    #[must_use]
    pub fn sys_operation(&self) -> Option<&Arc<SysOperation>> {
        self.sys_operation.as_ref()
    }

    /// The filesystem backend, when one is configured.
    #[must_use]
    pub fn fs_operation(&self) -> Option<&Arc<dyn FsOperation>> {
        self.fs.as_ref()
    }

    /// The shell backend, when one is configured.
    #[must_use]
    pub fn shell_operation(&self) -> Option<&Arc<dyn ShellOperation>> {
        self.shell.as_ref()
    }

    /// The user-interaction handler wired into the `ask_user` tool, when one is configured.
    #[must_use]
    pub fn user_asker(&self) -> Option<&Arc<dyn UserAsker>> {
        self.user_asker.as_ref()
    }

    /// The memory backend wired into the memory tools, when one is configured.
    #[must_use]
    pub fn memory_backend(&self) -> Option<&Arc<dyn MemoryToolBackend>> {
        self.memory_backend.as_ref()
    }

    /// The HTTP seam wired into `fetch_webpage`, when one is configured.
    ///
    /// `None` means the tool is built over a [`web::ReqwestFetcher`].
    #[must_use]
    pub fn http_fetcher(&self) -> Option<&Arc<dyn web::HttpFetcher>> {
        self.http_fetcher.as_ref()
    }

    /// The search provider wired into `web_search`, when one is configured.
    #[must_use]
    pub fn search_backend(&self) -> Option<&Arc<dyn web::SearchBackend>> {
        self.search.as_ref()
    }

    /// The MCP resource provider wired into the two resource tools, when one is configured.
    #[must_use]
    pub fn mcp_provider(&self) -> Option<&Arc<dyn mcp_resources::McpResourceProvider>> {
        self.mcp_provider.as_ref()
    }

    /// The todo directory override, when one was set.
    #[must_use]
    pub fn todo_dir(&self) -> Option<&Path> {
        self.todo_dir.as_deref()
    }

    /// The tools this registry builds, in registry order.
    #[must_use]
    pub fn tools(&self) -> &[BuiltinTool] {
        &self.tools
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod registry_tests {
    use super::*;

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
        assert_eq!(BuiltinTool::EXTENDED.len(), 30);
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
            );
            assert_eq!(tool.parallel_safe(), expected, "for {}", tool.name());
        }
        for name in ["write_file", "edit_file", "cron", "submit_goal_report"] {
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
