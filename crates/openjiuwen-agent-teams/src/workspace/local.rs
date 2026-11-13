//! The local-directory [`WorkspaceProvider`].
//!
//! Python parity source: `openjiuwen/agent_teams/workspace_layout.py` plus the workspace setup in
//! `team_agent.py` / `blueprint.py`. A member either works in its own member workspace directory
//! (Python `enabled=False`, the default) or in a session-scoped git worktree
//! (Python `isolation="worktree"`).
//!
//! Worktree requests go to the injected [`WorktreeBackend`] when one serves
//! [`IsolationMode::Worktree`], else to a [`GitWorktreeBackend`] that discovers the repository from
//! the session tree. Allocation therefore succeeds for a team whose session tree lives inside a git
//! repository and fails with [`Error::Config`] naming the directory when it does not.
//!
//! Allocation is fail-closed and release never deletes anything: Python retains member
//! workspaces until the team is deleted or the session is cleaned, so a released workspace stays on
//! disk for inspection and reuse.

use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};

use crate::schema::WorkspaceSpec;

use super::layout::{sanitize_segment, MemberWorkspaceLayout};
use super::worktree::{
    build_teammate_worktree_name, worktree_digest, WorktreeBackend, WorktreeCreateRequest,
};
use super::worktree_git::GitWorktreeBackend;
use super::{IsolationMode, MemberWorkspace, WorkspaceProvider, WorkspaceRequest};

/// The injected worktree backend and the isolation mode it is able to serve.
struct WorktreeSupport {
    /// Backend that creates and removes worktrees.
    backend: Arc<dyn WorktreeBackend>,
    /// Isolation mode the backend serves; allocation only delegates when it is [`IsolationMode::Worktree`].
    mode: IsolationMode,
}

/// Allocates member workspaces below one [`MemberWorkspaceLayout`].
///
/// [`IsolationMode::None`] allocates the plain session-scoped directory.
/// [`IsolationMode::Worktree`] allocates a git worktree through
/// [`Self::worktree_backend`]: the injected backend when one serves worktrees, otherwise
/// [`GitWorktreeBackend`] against the repository the session tree lives in.
pub struct LocalWorkspaceProvider {
    /// Path layout the provider allocates into.
    layout: MemberWorkspaceLayout,
    /// Optional worktree backend and the mode it serves.
    worktree: Option<WorktreeSupport>,
}

impl fmt::Debug for LocalWorkspaceProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalWorkspaceProvider")
            .field("layout", &self.layout)
            .field(
                "worktree",
                &self
                    .worktree
                    .as_ref()
                    .map(|support| (support.backend.name(), support.mode)),
            )
            .finish()
    }
}

impl LocalWorkspaceProvider {
    /// Creates a provider that allocates into `layout` and serves no worktrees.
    #[must_use]
    pub fn new(layout: MemberWorkspaceLayout) -> Self {
        Self {
            layout,
            worktree: None,
        }
    }

    /// Injects the worktree `backend`, which then serves `mode`.
    ///
    /// A backend is only used when `mode` is [`IsolationMode::Worktree`]; an other-mode backend is
    /// recorded (for diagnostics) but worktree requests keep going to
    /// [`Self::worktree_backend`]'s default (a [`GitWorktreeBackend`]). Injecting
    /// [`UnsupportedWorktreeBackend`](super::worktree::UnsupportedWorktreeBackend) with
    /// [`IsolationMode::Worktree`] therefore turns worktree isolation off even inside a repository.
    #[must_use]
    pub fn with_worktree_backend(
        mut self,
        backend: Arc<dyn WorktreeBackend>,
        mode: IsolationMode,
    ) -> Self {
        self.worktree = Some(WorktreeSupport { backend, mode });
        self
    }

    /// Returns the path layout of the provider.
    #[must_use]
    pub fn layout(&self) -> &MemberWorkspaceLayout {
        &self.layout
    }

    /// Returns the backend that serves [`IsolationMode::Worktree`].
    ///
    /// The injected backend wins when it was registered for [`IsolationMode::Worktree`]. Otherwise a
    /// fresh [`GitWorktreeBackend`] is returned; it holds no state beyond its (default) discovery
    /// and lock configuration, so building one per allocation costs nothing.
    #[must_use]
    pub fn worktree_backend(&self) -> Arc<dyn WorktreeBackend> {
        match self
            .worktree
            .as_ref()
            .filter(|support| support.mode == IsolationMode::Worktree)
        {
            Some(support) => Arc::clone(&support.backend),
            None => Arc::new(GitWorktreeBackend::new()),
        }
    }

    /// Allocates a session-scoped worktree for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unsafe team/session/member name, [`Error::Config`] when
    /// the session tree is not inside a git repository (or `git` is missing) and no backend was
    /// injected, and whatever the backend reports when the worktree cannot be created.
    async fn allocate_worktree(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
        let backend = self.worktree_backend();
        let seed = worktree_digest(&format!(
            "{}:{}:{}",
            request.team_name, request.member_name, request.session_id
        ));
        let name = build_teammate_worktree_name(&request.team_name, &request.member_name, &seed);
        let path = self
            .layout
            .worktrees_dir(&request.team_name, &request.session_id)?
            .join(&name);
        let create = WorktreeCreateRequest {
            team_name: request.team_name.clone(),
            session_id: request.session_id.clone(),
            member_name: request.member_name.clone(),
            name,
            path,
            base_ref: None,
        };
        let info = backend.create(&create).await?;
        debug!(
            backend = backend.name(),
            worktree = %info.name,
            path = %info.path.display(),
            "allocated teammate worktree"
        );
        Ok(MemberWorkspace {
            team_name: request.team_name.clone(),
            session_id: request.session_id.clone(),
            member_name: request.member_name.clone(),
            root: info.path.clone(),
            isolation: IsolationMode::Worktree,
            worktree: Some(info),
        })
    }

    /// Allocates (or merely addresses) the member workspace directory of `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unsafe team/session/member name or a project
    /// `root_path` outside the workspace, and [`Error::SysOp`] when a directory cannot be created.
    async fn allocate_shared(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
        let root = self.layout.member_workspace(
            &request.team_name,
            &request.session_id,
            &request.member_name,
        )?;
        let workspace = MemberWorkspace::new(
            request.team_name.clone(),
            request.session_id.clone(),
            request.member_name.clone(),
            root.clone(),
        );
        if !request.spec.enabled {
            warn!(
                team = %request.team_name,
                member = %request.member_name,
                root = %root.display(),
                "shared team workspace is disabled; no directory is created"
            );
            return Ok(workspace);
        }
        ensure_dir(&root).await?;
        let member_dir = self.layout.member_dir(
            &request.team_name,
            &request.session_id,
            &request.member_name,
        )?;
        ensure_dir(&member_dir).await?;
        if let Some(project) = project_dir(&root, request.member_spec.as_ref())? {
            ensure_dir(&project).await?;
        }
        debug!(
            team = %workspace.team_name,
            member = %workspace.member_name,
            root = %workspace.root_str(),
            "allocated member workspace"
        );
        Ok(workspace)
    }
}

/// Creates `path` and its parents, mapping an I/O failure to [`Error::SysOp`] with the path.
async fn ensure_dir(path: &Path) -> Result<()> {
    tokio::fs::create_dir_all(path)
        .await
        .map_err(|error| Error::sys_op(format!("creating directory {}: {error}", path.display())))
}

/// Resolves the project sub-directory a member spec asks for inside `root`.
///
/// A `root_path` of `""`, `"."` or `"./"` means "the workspace root itself" and yields `None`.
/// Every other value is relative to `root`; each component is sanitized, so `..`, absolute paths
/// and root/prefix components are rejected by [`Error::Validation`].
fn project_dir(root: &Path, member_spec: Option<&WorkspaceSpec>) -> Result<Option<PathBuf>> {
    let Some(spec) = member_spec else {
        return Ok(None);
    };
    let root_path = spec.root_path.trim();
    if root_path.is_empty() || root_path == "." || root_path == "./" {
        return Ok(None);
    }
    let mut relative = PathBuf::new();
    for component in Path::new(root_path).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(segment) => {
                let Some(segment) = segment.to_str() else {
                    return Err(Error::validation(format!(
                        "invalid member workspace root_path {:?}: not valid UTF-8",
                        spec.root_path
                    )));
                };
                relative.push(sanitize_segment(segment)?);
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Error::validation(format!(
                    "invalid member workspace root_path {:?}: must stay inside the workspace",
                    spec.root_path
                )));
            }
        }
    }
    if relative.as_os_str().is_empty() {
        return Ok(None);
    }
    Ok(Some(root.join(relative)))
}

#[async_trait]
impl WorkspaceProvider for LocalWorkspaceProvider {
    async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
        match request.isolation {
            IsolationMode::Worktree => self.allocate_worktree(request).await,
            IsolationMode::None => self.allocate_shared(request).await,
        }
    }

    async fn release(&self, workspace: &MemberWorkspace) -> Result<()> {
        if let Some(info) = workspace.worktree.as_ref() {
            return self.worktree_backend().remove(info).await;
        }
        debug!(
            team = %workspace.team_name,
            member = %workspace.member_name,
            root = %workspace.root.display(),
            "retaining member workspace; cleanup happens on team delete or session clean"
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use tempfile::TempDir;

    use super::*;
    use crate::schema::TeamWorkspaceSpec;
    use crate::workspace::worktree::{worktree_branch, WorktreeInfo};

    struct FakeBackend {
        created: Mutex<Vec<WorktreeCreateRequest>>,
        removed: Mutex<Vec<WorktreeInfo>>,
        fail: bool,
    }

    impl FakeBackend {
        fn new(fail: bool) -> Arc<Self> {
            Arc::new(Self {
                created: Mutex::new(Vec::new()),
                removed: Mutex::new(Vec::new()),
                fail,
            })
        }

        fn created(&self) -> Vec<WorktreeCreateRequest> {
            self.created.lock().clone()
        }

        fn removed(&self) -> Vec<WorktreeInfo> {
            self.removed.lock().clone()
        }
    }

    #[async_trait]
    impl WorktreeBackend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }

        async fn create(&self, request: &WorktreeCreateRequest) -> Result<WorktreeInfo> {
            self.created.lock().push(request.clone());
            if self.fail {
                return Err(Error::unsupported("fake backend cannot create worktrees"));
            }
            Ok(WorktreeInfo {
                name: request.name.clone(),
                path: request.path.clone(),
                branch: worktree_branch(&request.name),
            })
        }

        async fn remove(&self, worktree: &WorktreeInfo) -> Result<()> {
            self.removed.lock().push(worktree.clone());
            Ok(())
        }
    }

    fn layout(temp: &TempDir) -> MemberWorkspaceLayout {
        MemberWorkspaceLayout::new(temp.path())
    }

    fn enabled_request(team: &str, session: &str, member: &str) -> WorkspaceRequest {
        WorkspaceRequest {
            spec: TeamWorkspaceSpec {
                enabled: true,
                ..TeamWorkspaceSpec::default()
            },
            ..WorkspaceRequest::new(team, session, member)
        }
    }

    #[tokio::test]
    async fn enabled_spec_creates_the_workspace_and_member_dir() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let request = enabled_request("my-team", "session-1", "dev-1");
        let workspace = provider.allocate(&request).await.unwrap();
        let expected_root = temp
            .path()
            .join("my-team")
            .join("sessions")
            .join("session-1")
            .join("workspaces")
            .join("dev-1_workspace");
        assert_eq!(workspace.root, expected_root);
        assert_eq!(workspace.root_str(), expected_root.to_string_lossy());
        assert!(workspace.is_shared());
        assert_eq!(workspace.isolation, IsolationMode::None);
        assert!(workspace.worktree.is_none());
        assert!(workspace.root.is_dir());
        assert!(temp
            .path()
            .join("my-team")
            .join("sessions")
            .join("session-1")
            .join("members")
            .join("dev-1")
            .is_dir());
        assert!(provider.layout().base().starts_with(temp.path()));
    }

    #[tokio::test]
    async fn disabled_spec_creates_nothing() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let request = WorkspaceRequest::new("my-team", "session-1", "dev-1");
        let workspace = provider.allocate(&request).await.unwrap();
        assert!(!workspace.root.exists());
        assert!(workspace.is_shared());
    }

    #[tokio::test]
    async fn member_spec_root_path_creates_the_project_directory() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let request = WorkspaceRequest {
            member_spec: Some(WorkspaceSpec {
                root_path: "src/x".to_owned(),
                ..WorkspaceSpec::default()
            }),
            ..enabled_request("my-team", "session-1", "dev-1")
        };
        let workspace = provider.allocate(&request).await.unwrap();
        assert!(workspace.root.join("src").join("x").is_dir());
    }

    #[tokio::test]
    async fn member_spec_root_path_stays_inside_the_workspace() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let base = temp.path().to_path_buf();
        for root_path in ["/abs", "../escape", "src/../../escape"] {
            let request = WorkspaceRequest {
                member_spec: Some(WorkspaceSpec {
                    root_path: root_path.to_owned(),
                    ..WorkspaceSpec::default()
                }),
                ..enabled_request("my-team", "session-1", "dev-1")
            };
            let err = provider.allocate(&request).await.expect_err(root_path);
            assert!(
                matches!(err, Error::Validation(_)),
                "{root_path:?} produced {err:?}"
            );
            assert!(!base.join("escape").exists());
        }
        // The benign spellings of "the root itself" allocate the workspace only.
        for root_path in ["", ".", "./"] {
            let request = WorkspaceRequest {
                member_spec: Some(WorkspaceSpec {
                    root_path: root_path.to_owned(),
                    ..WorkspaceSpec::default()
                }),
                ..enabled_request("my-team", "session-1", "dev-1")
            };
            let workspace = provider.allocate(&request).await.unwrap();
            assert!(workspace.root.is_dir());
        }
    }

    #[tokio::test]
    async fn hostile_names_are_rejected_before_touching_the_disk() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let request = enabled_request("../../etc", "session-1", "dev-1");
        let err = provider.allocate(&request).await.expect_err("rejected");
        assert!(matches!(err, Error::Validation(_)), "{err:?}");
        assert!(!temp.path().join("etc").exists());
    }

    #[tokio::test]
    async fn release_retains_the_shared_workspace() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let workspace = provider
            .allocate(&enabled_request("my-team", "session-1", "dev-1"))
            .await
            .unwrap();
        provider.release(&workspace).await.unwrap();
        assert!(workspace.root.is_dir());
    }

    #[tokio::test]
    async fn worktree_outside_a_repository_fails_with_config() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let request = WorkspaceRequest {
            isolation: IsolationMode::Worktree,
            ..enabled_request("my-team", "session-1", "dev-1")
        };
        let err = provider
            .allocate(&request)
            .await
            .expect_err("no repository");
        assert!(matches!(err, Error::Config(_)), "{err:?}");
        assert!(
            err.to_string().contains("requires a git repository"),
            "{err}"
        );

        // A backend recorded under another mode does not serve worktree requests either, so the
        // default (git) backend keeps reporting the missing repository.
        let provider = provider.with_worktree_backend(FakeBackend::new(false), IsolationMode::None);
        let err = provider
            .allocate(&request)
            .await
            .expect_err("no repository");
        assert!(matches!(err, Error::Config(_)), "{err:?}");
    }

    #[tokio::test]
    async fn injected_unsupported_backend_disables_worktrees() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp)).with_worktree_backend(
            Arc::new(super::super::worktree::UnsupportedWorktreeBackend),
            IsolationMode::Worktree,
        );
        let request = WorkspaceRequest {
            isolation: IsolationMode::Worktree,
            ..enabled_request("my-team", "session-1", "dev-1")
        };
        let err = provider.allocate(&request).await.expect_err("unsupported");
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("agent_teams.worktree.lifecycle"));
    }

    #[tokio::test]
    async fn worktree_allocation_delegates_to_the_backend() {
        let temp = TempDir::new().unwrap();
        let backend = FakeBackend::new(false);
        let provider = LocalWorkspaceProvider::new(layout(&temp))
            .with_worktree_backend(backend.clone(), IsolationMode::Worktree);
        let request = WorkspaceRequest {
            isolation: IsolationMode::Worktree,
            ..enabled_request("my-team", "session-1", "dev-1")
        };
        let workspace = provider.allocate(&request).await.unwrap();

        let created = backend.created();
        assert_eq!(created.len(), 1);
        let create = &created[0];
        assert!(create.name.starts_with("agent-"), "{}", create.name);
        assert_eq!(create.team_name, "my-team");
        assert_eq!(create.session_id, "session-1");
        assert_eq!(create.member_name, "dev-1");
        assert!(create.base_ref.is_none());
        let worktrees = temp
            .path()
            .join("my-team")
            .join("sessions")
            .join("session-1")
            .join("worktrees");
        assert!(create.path.starts_with(&worktrees));
        assert_eq!(
            create.path.file_name().unwrap().to_str().unwrap(),
            create.name
        );

        assert_eq!(workspace.isolation, IsolationMode::Worktree);
        assert!(!workspace.is_shared());
        let info = workspace.worktree.as_ref().expect("worktree info");
        assert_eq!(workspace.root, info.path);
        assert_eq!(info.path, create.path);
        assert_eq!(info.branch, worktree_branch(&info.name));
        assert_eq!(info.name, create.name);
    }

    #[tokio::test]
    async fn worktree_backend_failure_propagates() {
        let temp = TempDir::new().unwrap();
        let backend = FakeBackend::new(true);
        let provider = LocalWorkspaceProvider::new(layout(&temp))
            .with_worktree_backend(backend.clone(), IsolationMode::Worktree);
        let request = WorkspaceRequest {
            isolation: IsolationMode::Worktree,
            ..enabled_request("my-team", "session-1", "dev-1")
        };
        let err = provider
            .allocate(&request)
            .await
            .expect_err("backend fails");
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert_eq!(backend.created().len(), 1);
        assert_eq!(backend.removed(), vec![]);
    }

    #[tokio::test]
    async fn release_of_a_worktree_calls_the_backend() {
        let temp = TempDir::new().unwrap();
        let backend = FakeBackend::new(false);
        let provider = LocalWorkspaceProvider::new(layout(&temp))
            .with_worktree_backend(backend.clone(), IsolationMode::Worktree);
        let request = WorkspaceRequest {
            isolation: IsolationMode::Worktree,
            ..enabled_request("my-team", "session-1", "dev-1")
        };
        let workspace = provider.allocate(&request).await.unwrap();
        provider.release(&workspace).await.unwrap();
        let removed = backend.removed();
        assert_eq!(removed.len(), 1);
        assert_eq!(Some(&removed[0]), workspace.worktree.as_ref());
    }

    #[test]
    fn debug_reports_the_backend_name() {
        let temp = TempDir::new().unwrap();
        let provider = LocalWorkspaceProvider::new(layout(&temp));
        let debug = format!("{provider:?}");
        assert!(debug.contains("LocalWorkspaceProvider"));
        assert!(debug.contains("worktree: None"));
        let provider =
            provider.with_worktree_backend(FakeBackend::new(false), IsolationMode::Worktree);
        assert!(format!("{provider:?}").contains("fake"));
    }
}
