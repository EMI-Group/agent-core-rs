//! Model-facing coding tools, the tool registry, and shared tool helpers.
//!
//! Ports `openjiuwen/harness/tools/`: the shell, filesystem, todo, ask-user and memory tools
//! exposed to the model, plus their shared payload/path helpers and the registry.
//! Upper layers only see [`ToolRegistry`], [`BuiltinTool`] and the tool seam traits.
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
}

impl BuiltinTool {
    /// Every built-in tool, in registry order.
    pub const ALL: [Self; 20] = [
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
        }
    }

    /// The tool whose model-facing name is exactly `name`, if any.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tool| tool.name() == name)
    }

    /// Whether the tool may run concurrently with sibling tool calls.
    ///
    /// Only the write tools are serialised; every other built-in tool is parallel-safe.
    #[must_use]
    pub const fn parallel_safe(self) -> bool {
        !matches!(self, Self::WriteFile | Self::EditFile)
    }
}

/// The model-facing tool names, in registry order.
#[must_use]
pub fn tool_names() -> &'static [&'static str] {
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

/// The built-in tool set of one agent, with its workspace, backends and shared read state.
///
/// The registry owns the [`ToolEnv`] path/permission gate, the sys-operation backends, the
/// read-before-write registry and the todo directory; [`Self::build`] materialises the tool
/// implementations and [`Self::cards`] their model-facing cards.
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
            tools: BuiltinTool::ALL.to_vec(),
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
    /// Every selected [`BuiltinTool`] yields exactly one [`Tool`], in registry order. The todo
    /// tools share one [`TodoStore`] and the filesystem tools share one [`ReadStateRegistry`],
    /// so a registry behaves like a single agent's tool set.
    #[must_use]
    pub fn build(&self) -> Vec<Arc<dyn Tool>> {
        let language = self.env.language();
        let mut todo_store: Option<Arc<TodoStore>> = None;
        self.tools
            .iter()
            .map(|tool| {
                let tool: Arc<dyn Tool> = match tool {
                    BuiltinTool::Bash => {
                        Arc::new(BashTool::new(self.env.clone(), self.shell.clone()))
                    }
                    BuiltinTool::ReadFile => Arc::new(ReadFileTool::new(
                        self.env.clone(),
                        self.fs.clone(),
                        self.read_state.clone(),
                    )),
                    BuiltinTool::WriteFile => Arc::new(WriteFileTool::new(
                        self.env.clone(),
                        self.fs.clone(),
                        self.read_state.clone(),
                    )),
                    BuiltinTool::EditFile => Arc::new(EditFileTool::new(
                        self.env.clone(),
                        self.fs.clone(),
                        self.read_state.clone(),
                    )),
                    BuiltinTool::Glob => Arc::new(GlobTool::new(self.env.clone(), self.fs.clone())),
                    BuiltinTool::ListFiles => {
                        Arc::new(ListDirTool::new(self.env.clone(), self.fs.clone()))
                    }
                    BuiltinTool::Grep => Arc::new(GrepTool::new(self.env.clone())),
                    BuiltinTool::TodoCreate => Arc::new(TodoCreateTool::new(
                        self.todo_store(&mut todo_store),
                        language,
                    )),
                    BuiltinTool::TodoList => Arc::new(TodoListTool::new(
                        self.todo_store(&mut todo_store),
                        language,
                    )),
                    BuiltinTool::TodoGet => {
                        Arc::new(TodoGetTool::new(self.todo_store(&mut todo_store), language))
                    }
                    BuiltinTool::TodoModify => Arc::new(TodoModifyTool::new(
                        self.todo_store(&mut todo_store),
                        language,
                    )),
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
                    | BuiltinTool::CodingMemoryEdit => Arc::new(MemoryTool::new(
                        memory_kind(*tool),
                        self.memory_backend.clone(),
                        language,
                    )),
                };
                tool
            })
            .collect()
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
        }
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
    fn builtin_tool_all_has_twenty_variants_in_registry_order() {
        assert_eq!(BuiltinTool::ALL.len(), 20);
        let names: Vec<&str> = BuiltinTool::ALL.iter().map(|tool| tool.name()).collect();
        assert_eq!(names, EXPECTED_NAMES.to_vec());
    }

    #[test]
    fn tool_names_match_registry_order() {
        assert_eq!(tool_names(), &EXPECTED_NAMES);
    }

    #[test]
    fn from_name_round_trips_every_name() {
        for name in EXPECTED_NAMES {
            let tool = BuiltinTool::from_name(name).expect("known tool name");
            assert_eq!(tool.name(), name, "round-trip mismatch for {name}");
        }
        assert_eq!(BuiltinTool::from_name("nope"), None);
    }

    #[test]
    fn only_write_tools_are_not_parallel_safe() {
        for tool in BuiltinTool::ALL {
            let expected = !matches!(tool, BuiltinTool::WriteFile | BuiltinTool::EditFile);
            assert_eq!(tool.parallel_safe(), expected, "for {}", tool.name());
        }
        assert!(!BuiltinTool::from_name("write_file")
            .unwrap()
            .parallel_safe());
        assert!(!BuiltinTool::from_name("edit_file").unwrap().parallel_safe());
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
