//! The [`ToolRegistry`]: the built-in tool set of one agent, with its seams and arm tables.
//!
//! Split out of `tools/mod.rs` so the variant tables, the shared helpers and the registry stay
//! within the crate's per-file size budget; the public path is still
//! [`crate::tools::ToolRegistry`] through the flat re-export in the parent module.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_sysop::operation::{FsOperation, ShellOperation};
use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

use crate::lsp::{LspTool, LspToolConfig};
use crate::permission::PermissionEngine;
use crate::workspace::Workspace;
use crate::worktree::{
    EnterWorktreeTool, ExitWorktreeTool, MemoryWorktreeSessionStore, WorktreeManager,
    WorktreeSessionStore, WorktreeTool,
};

use super::{
    cron, discovery, goal, mcp_resources, mcp_subscription_tool, mcp_subscriptions, media, web,
    AskUserTool, BashTool, BuiltinTool, EditFileTool, GlobTool, GrepTool, ListDirTool, MemoryTool,
    MemoryToolBackend, MemoryToolKind, ReadFileTool, ReadStateRegistry, TodoCreateTool,
    TodoGetTool, TodoListTool, TodoModifyTool, TodoStore, ToolEnv, UserAsker,
    WorkspaceMemoryBackend, WriteFileTool,
};

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
    /// The MCP resource subscription registry behind `mcp_resource_subscription`, when wired.
    mcp_subscriptions: Option<Arc<mcp_subscriptions::McpResourceSubscriptions>>,
    /// The cron scheduler behind the `cron` tool, when one is wired.
    cron_scheduler: Option<Arc<cron::CronScheduler>>,
    /// The goal store behind the two goal tools, when one is wired.
    goal_store: Option<Arc<goal::GoalStore>>,
    /// The language-server configuration behind the `lsp` tool, when one is wired.
    lsp_config: Option<LspToolConfig>,
    /// The worktree manager behind the `worktree` tool, when one is wired.
    worktree_manager: Option<Arc<WorktreeManager>>,
    /// Repository-root override of the default worktree manager; `None` uses the workspace root.
    worktree_root: Option<PathBuf>,
    /// The worktree session store shared by `enter_worktree` and `exit_worktree`.
    worktree_sessions: Arc<dyn WorktreeSessionStore>,
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
            mcp_subscriptions: None,
            cron_scheduler: None,
            goal_store: None,
            lsp_config: None,
            worktree_manager: None,
            worktree_root: None,
            worktree_sessions: Arc::new(MemoryWorktreeSessionStore::new()),
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

    /// Narrows every tool's path resolution to the project root, when configured.
    ///
    /// Python parity: `project_root` (`init_cwd`'s project identity anchor) is one of the
    /// allowed roots the filesystem guards accept. `None` keeps the workspace root as the
    /// only boundary; a configured root also rejects a path outside it that the workspace
    /// itself would allow (a sibling directory of the project root, for instance).
    #[must_use]
    pub fn with_project_root(mut self, project_root: Option<PathBuf>) -> Self {
        self.env = self.env.clone().with_access_root(project_root);
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
    /// Wires the file-backed [`WorkspaceMemoryBackend`] over this registry's workspace.
    ///
    /// [`DeepAgentBuilder`](crate::agent::DeepAgentBuilder) calls this by default, so the eight
    /// memory tools work out of the box; [`Self::with_memory`] still replaces the backend when a
    /// host has a richer one (for example an embedding-backed store).
    #[must_use]
    pub fn with_workspace_memory(mut self) -> Self {
        let mut backend = WorkspaceMemoryBackend::new(self.env.workspace().clone());
        if let Some(fs) = self.fs.clone() {
            backend = backend.with_fs_operation(fs);
        }
        self.memory_backend = Some(Arc::new(backend));
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

    /// Wires the MCP resource subscription registry behind `mcp_resource_subscription`.
    ///
    /// Without one every action of that tool reports a typed "not available" payload. The
    /// provider's `subscribe`/`unsubscribe` live on the registry
    /// ([`mcp_subscriptions::McpResourceSubscriptions::with_provider`]), so a host that already
    /// implements the subscription trait shares one instance between the registry and the tool.
    #[must_use]
    pub fn with_mcp_subscriptions(
        mut self,
        subscriptions: Arc<mcp_subscriptions::McpResourceSubscriptions>,
    ) -> Self {
        self.mcp_subscriptions = Some(subscriptions);
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

    /// Wires the configuration of the `lsp` tool, replacing any previous one.
    ///
    /// Accepts a bare [`LspToolConfig`] or an `Option`. The registry fills the configuration's
    /// workspace root from the registry workspace when it is unset and always renders the built
    /// tool in the registry language. Without a configuration the tool is still mounted and
    /// every call reports the typed "no LSP server configured" failure, so the model is told
    /// what to configure instead of the tool silently disappearing.
    #[must_use]
    pub fn with_lsp_config(mut self, config: impl Into<Option<LspToolConfig>>) -> Self {
        self.lsp_config = config.into();
        self
    }

    /// Wires the worktree manager behind the `worktree` tool, replacing any previous manager.
    #[must_use]
    pub fn with_worktree_manager(mut self, manager: Arc<WorktreeManager>) -> Self {
        self.worktree_manager = Some(manager);
        self
    }

    /// Sets the repository root of the default worktree manager; `None` uses the workspace root.
    #[must_use]
    pub fn with_worktree_root(mut self, root: impl Into<Option<PathBuf>>) -> Self {
        self.worktree_root = root.into();
        self
    }

    /// Wires the session store shared by `enter_worktree` and `exit_worktree`.
    ///
    /// Without one both tools share an in-process [`MemoryWorktreeSessionStore`], which is what a
    /// single agent needs; a host that keeps the active worktree per session, per team member or
    /// on disk implements [`WorktreeSessionStore`] and wires it here.
    #[must_use]
    pub fn with_worktree_session_store(mut self, store: Arc<dyn WorktreeSessionStore>) -> Self {
        self.worktree_sessions = store;
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
                BuiltinTool::McpResourceSubscription => {
                    Arc::new(mcp_subscription_tool::McpResourceSubscriptionTool::new(
                        self.mcp_subscriptions.clone(),
                        language,
                    ))
                }
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
                BuiltinTool::Lsp => Arc::new(self.lsp_tool()),
                BuiltinTool::Worktree => Arc::new(self.worktree_tool(language)),
                BuiltinTool::EnterWorktree => Arc::new(self.enter_worktree_tool(language)),
                BuiltinTool::ExitWorktree => Arc::new(self.exit_worktree_tool(language)),
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

    /// Builds the `lsp` tool from the wired [`LspToolConfig`], else from its defaults.
    ///
    /// The workspace root falls back to the registry workspace and the description language
    /// always follows the registry. With no configuration (or no server in it) the tool is still
    /// mounted: every call answers with the LSP module's typed "no server configured" payload
    /// instead of a construction-time error, so the model is told what the host must set.
    fn lsp_tool(&self) -> LspTool {
        let mut config = self.lsp_config.clone().unwrap_or_default();
        if config.workspace_root.is_none() {
            config.workspace_root = Some(self.env.workspace().root_path.clone());
        }
        self.env.language().clone_into(&mut config.language);
        LspTool::new(config)
    }

    /// Builds the `worktree` tool over the wired manager, else one at the configured repository
    /// root (or the workspace root when none is set), forwarding the permission engine.
    ///
    /// Nothing is checked at construction time: a workspace outside a git repository surfaces the
    /// manager's typed git error as a `success=false` payload when the tool runs.
    fn worktree_tool(&self, language: &str) -> WorktreeTool {
        let tool = WorktreeTool::new(self.shared_worktree_manager(), language);
        match self.env.permission() {
            Some(engine) => tool.with_permission(Arc::clone(engine)),
            None => tool,
        }
    }

    /// Builds `enter_worktree` over the shared manager and worktree session store.
    ///
    /// Like `worktree`, the default manager is rooted at [`Self::worktree_repo_root`] and nothing
    /// is checked at construction time: outside a git repository the manager's typed git failure
    /// becomes a `success=false` payload naming the problem.
    fn enter_worktree_tool(&self, language: &str) -> EnterWorktreeTool {
        let tool = EnterWorktreeTool::new(self.shared_worktree_manager(), language)
            .with_sessions(Arc::clone(&self.worktree_sessions));
        match self.env.permission() {
            Some(engine) => tool.with_permission(Arc::clone(engine)),
            None => tool,
        }
    }

    /// Builds `exit_worktree` over the shared manager and worktree session store.
    fn exit_worktree_tool(&self, language: &str) -> ExitWorktreeTool {
        let tool = ExitWorktreeTool::new(self.shared_worktree_manager(), language)
            .with_sessions(Arc::clone(&self.worktree_sessions));
        match self.env.permission() {
            Some(engine) => tool.with_permission(Arc::clone(engine)),
            None => tool,
        }
    }

    /// The worktree manager shared by the three worktree tools: the wired manager, else one
    /// rooted at [`Self::worktree_repo_root`].
    fn shared_worktree_manager(&self) -> Arc<WorktreeManager> {
        self.worktree_manager
            .clone()
            .unwrap_or_else(|| Arc::new(WorktreeManager::new(self.worktree_repo_root())))
    }

    /// The repository root of the default worktree manager.
    fn worktree_repo_root(&self) -> PathBuf {
        self.worktree_root
            .clone()
            .unwrap_or_else(|| self.env.workspace().root_path.clone())
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
            BuiltinTool::McpResourceSubscription => {
                mcp_subscription_tool::McpResourceSubscriptionTool::card_for(language)
            }
            BuiltinTool::ToolSearch => discovery::ToolSearchTool::card_for(language),
            BuiltinTool::GetCurrentGoal => goal::GetCurrentGoalTool::card_for(language),
            BuiltinTool::SubmitGoalReport => goal::SubmitGoalReportTool::card_for(language),
            BuiltinTool::Lsp => LspTool::card_for(language),
            BuiltinTool::Worktree => WorktreeTool::card_for(language),
            BuiltinTool::EnterWorktree => EnterWorktreeTool::card_for(language),
            BuiltinTool::ExitWorktree => ExitWorktreeTool::card_for(language),
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

    /// The MCP resource subscription registry wired into `mcp_resource_subscription`, when one
    /// is configured.
    #[must_use]
    pub fn mcp_subscriptions(&self) -> Option<&Arc<mcp_subscriptions::McpResourceSubscriptions>> {
        self.mcp_subscriptions.as_ref()
    }

    /// The language-server configuration wired into the `lsp` tool, when one is configured.
    ///
    /// `None` means the tool is built from [`LspToolConfig::default`] and reports the typed
    /// "no LSP server configured" failure on every call.
    #[must_use]
    pub fn lsp_config(&self) -> Option<&LspToolConfig> {
        self.lsp_config.as_ref()
    }

    /// The worktree manager wired into the `worktree` tool, when one is configured.
    ///
    /// `None` means the three worktree tools are built over a manager rooted at
    /// [`Self::worktree_root`] or the workspace root.
    #[must_use]
    pub fn worktree_manager(&self) -> Option<&Arc<WorktreeManager>> {
        self.worktree_manager.as_ref()
    }

    /// The repository-root override of the default worktree manager, when one was set.
    #[must_use]
    pub fn worktree_root(&self) -> Option<&Path> {
        self.worktree_root.as_deref()
    }

    /// The worktree session store shared by `enter_worktree` and `exit_worktree`.
    #[must_use]
    pub fn worktree_session_store(&self) -> &Arc<dyn WorktreeSessionStore> {
        &self.worktree_sessions
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
