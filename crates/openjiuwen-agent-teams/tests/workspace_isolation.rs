//! Real git-worktree isolation through the crate's public API.
//!
//! `metadata["workspace"] = {"isolation": "worktree"}` makes the runtime hand every
//! `WorkspaceRequest` the `IsolationMode::Worktree` mode; this suite proves the whole chain
//! against a **real** temporary git repository and a real `LocalWorkspaceProvider`:
//!
//! 1. every member (leader and each spawned teammate) is allocated a git worktree, reported as
//!    `IsolationMode::Worktree` with `worktree` metadata, inside the provider's layout, carrying
//!    the worktree `.git` *file* marker, and distinct from the repository's own working tree;
//! 2. `git worktree list` in the repository registers one entry per member, and the harness of
//!    each member is created with exactly that worktree as its workspace root;
//! 3. two members built from the same teammate blueprint still get distinct worktrees;
//! 4. teardown releases the worktrees through the product path: `stop_team` ends in
//!    `TeamAgent::stop`, which hands every retained workspace back to the provider it allocated
//!    from, so the member directories are gone, `git worktree list` no longer registers them and
//!    their `worktree-*` branches are deleted without any manual release;
//! 5. a team without the metadata key stays on plain local directories — no `.git` marker, no
//!    repository entry, and a release that retains the directory (Python keeps shared
//!    workspaces until the team is deleted).
//!
//! The repository is created by the test itself (`git init` plus one commit, with the identity
//! passed as `-c user.email=… -c user.name=…` flags so no global git configuration is read and
//! nothing can prompt). Every wait is bounded by [`support::BOUND`]; nothing touches the network.
//!
//! The provider is pinned to the repository with `GitWorktreeBackend::with_repo_root`: the
//! session tree lives in its own temporary layout directory (outside the repository), so
//! discovery from the worktree target would not resolve the repository this suite created.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod support;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent_teams::harness::MemberHarnessFactory;
use openjiuwen_agent_teams::runtime::{TeamAgent, TeamBuildDeps, TeamRuntimeManager};
use openjiuwen_agent_teams::schema::{PredefinedMemberSpec, TeamAgentSpec};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen_agent_teams::workspace::{
    GitWorktreeBackend, IsolationMode, LocalWorkspaceProvider, MemberWorkspace,
    MemberWorkspaceLayout, WorkspaceProvider, WorkspaceRequest,
};
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::time::timeout;

use support::{sample_spec, wait_until, FakeHarnessFactory, BOUND};

/// Session every team in this suite is activated under.
const SESSION: &str = "session-worktree";

/// Team name [`sample_spec`] uses.
const TEAM: &str = "alpha";

/// Runs one `git` command in `cwd` and returns its stdout.
///
/// The identity is passed per invocation (`-c user.email=…`), so the suite works on a host with no
/// git configuration at all, and a missing `git` binary is a clear panic rather than a silent
/// skip: the whole point of this suite is the real backend.
fn git(cwd: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(cwd)
        .args([
            "-c",
            "user.email=teams@example.invalid",
            "-c",
            "user.name=openJiuwen Teams Tests",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "`git {}` could not be started; this suite needs a git binary: {error}",
                args.join(" ")
            )
        });
    assert!(
        output.status.success(),
        "`git {}` failed with {}: {}",
        args.join(" "),
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// A throwaway git repository with one commit, owned for the lifetime of a test.
struct TempRepo {
    /// Keeps the repository on disk for the test.
    dir: TempDir,
}

impl TempRepo {
    /// Initializes the repository and commits one seed file.
    fn init() -> Self {
        let dir = TempDir::new().expect("a temporary repository directory");
        git(dir.path(), &["init", "--quiet"]);
        std::fs::write(dir.path().join("README.md"), "# workspace isolation\n")
            .expect("the seed file");
        git(dir.path(), &["add", "README.md"]);
        git(dir.path(), &["commit", "--quiet", "-m", "seed"]);
        Self { dir }
    }

    /// Returns the repository's working tree (its main worktree).
    fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Returns the worktree paths `git worktree list` reports, main worktree first.
    fn worktree_paths(&self) -> Vec<PathBuf> {
        git(self.path(), &["worktree", "list", "--porcelain"])
            .lines()
            .filter_map(|line| line.strip_prefix("worktree "))
            .map(PathBuf::from)
            .collect()
    }

    /// Returns the branches the worktree backend still has registered.
    fn worktree_branches(&self) -> Vec<String> {
        git(self.path(), &["branch", "--list", "worktree-*"])
            .lines()
            .map(|line| line.trim().trim_start_matches("* ").to_owned())
            .filter(|line| !line.is_empty())
            .collect()
    }
}

/// Canonicalizes `path` when it exists, so comparisons survive a symlinked temp directory.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// [`LocalWorkspaceProvider`] wrapper that records every allocation and release.
///
/// The wrapper delegates to the real provider (and therefore to the real
/// [`GitWorktreeBackend`]), so what it records is exactly what the runtime asked for and what the
/// git backend did.
struct RecordingProvider {
    /// The real provider under test.
    inner: LocalWorkspaceProvider,
    /// Workspaces handed out, in allocation order.
    allocations: Mutex<Vec<MemberWorkspace>>,
    /// Workspaces released through [`WorkspaceProvider::release`], in call order.
    releases: Mutex<Vec<MemberWorkspace>>,
}

impl RecordingProvider {
    /// Wraps `inner` in a recording provider.
    fn new(inner: LocalWorkspaceProvider) -> Arc<Self> {
        Arc::new(Self {
            inner,
            allocations: Mutex::new(Vec::new()),
            releases: Mutex::new(Vec::new()),
        })
    }

    /// Returns the allocated workspaces, in allocation order.
    fn allocations(&self) -> Vec<MemberWorkspace> {
        self.allocations.lock().clone()
    }

    /// Returns the released workspaces, in release order.
    fn releases(&self) -> Vec<MemberWorkspace> {
        self.releases.lock().clone()
    }

    /// Returns the workspace allocated to `member`, when it was allocated.
    fn workspace(&self, member: &str) -> Option<MemberWorkspace> {
        self.allocations
            .lock()
            .iter()
            .find(|workspace| workspace.member_name == member)
            .cloned()
    }
}

#[async_trait]
impl WorkspaceProvider for RecordingProvider {
    async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
        let workspace = self.inner.allocate(request).await?;
        self.allocations.lock().push(workspace.clone());
        Ok(workspace)
    }

    async fn release(&self, workspace: &MemberWorkspace) -> Result<()> {
        self.inner.release(workspace).await?;
        self.releases.lock().push(workspace.clone());
        Ok(())
    }
}

/// Builds a provider that allocates into `layout` and serves worktrees from `repo`.
fn worktree_provider(layout: &Path, repo: &Path) -> Arc<RecordingProvider> {
    let backend = GitWorktreeBackend::new().with_repo_root(repo);
    let provider = LocalWorkspaceProvider::new(MemberWorkspaceLayout::new(layout))
        .with_worktree_backend(Arc::new(backend), IsolationMode::Worktree);
    RecordingProvider::new(provider)
}

/// Activates `spec` for [`SESSION`] over `provider`, with scripted harnesses and a fresh store.
async fn activate(
    spec: &TeamAgentSpec,
    provider: &Arc<RecordingProvider>,
) -> (
    Arc<TeamRuntimeManager>,
    Arc<TeamAgent>,
    Arc<FakeHarnessFactory>,
) {
    let factory = FakeHarnessFactory::new();
    let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
    let deps = TeamBuildDeps::default()
        .with_harness_factory(Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>)
        .with_workspace(Arc::clone(provider) as Arc<dyn WorkspaceProvider>)
        .with_store(store)
        .with_language("en");
    let manager = Arc::new(TeamRuntimeManager::new(deps));
    let agent = timeout(BOUND, manager.activate(spec, SESSION, &json!(null)))
        .await
        .expect("activating the team did not finish within the bound")
        .expect("the team did not activate");
    (manager, agent, factory)
}

/// Stops `agent`'s teammate roster through `auto_start_all`'s sibling `spawn_member`, bounded.
async fn spawn_teammates(agent: &Arc<TeamAgent>) {
    timeout(BOUND, agent.auto_start_all())
        .await
        .expect("spawning the teammates did not finish within the bound")
        .expect("a teammate did not spawn");
}

/// Builds the sample blueprint with `metadata["workspace"] = value`.
fn spec_with_workspace(value: Value) -> TeamAgentSpec {
    let mut spec = sample_spec();
    spec.metadata.insert("workspace".to_owned(), value);
    spec
}

/// Adds a second teammate that is a byte-for-byte copy of `dev-1` except for its name.
fn with_second_teammate(mut spec: TeamAgentSpec) -> TeamAgentSpec {
    let Some(PredefinedMemberSpec::Teammate(first)) = spec.predefined_members.first() else {
        panic!("the sample blueprint starts with a predefined teammate");
    };
    let mut second = first.clone();
    "dev-2".clone_into(&mut second.member_name);
    "Dev Two".clone_into(&mut second.display_name);
    spec.predefined_members
        .push(PredefinedMemberSpec::Teammate(second));
    spec
}

/// Asserts that `workspace` is a real git worktree of the repository rooted at `repo_root`.
///
/// `worktrees_dir` is the provider layout directory the worktree must live under, `session` the
/// hosting session the workspace must record, `listed` the canonical worktree paths
/// `git worktree list` reports and `factory` the harness factory that must have received the very
/// same root for that member.
fn assert_registered_worktree(
    workspace: &MemberWorkspace,
    repo_root: &Path,
    worktrees_dir: &Path,
    session: &str,
    listed: &BTreeSet<PathBuf>,
    factory: &FakeHarnessFactory,
) {
    assert_eq!(
        workspace.isolation,
        IsolationMode::Worktree,
        "{} was not isolated: {workspace:?}",
        workspace.member_name
    );
    assert!(!workspace.is_shared());
    let info = workspace
        .worktree
        .as_ref()
        .unwrap_or_else(|| panic!("{} has no worktree info", workspace.member_name));
    assert_eq!(workspace.root, info.path, "the root is the worktree path");
    assert!(
        workspace.root.is_dir(),
        "{}'s worktree is not a directory: {workspace:?}",
        workspace.member_name
    );
    assert!(
        workspace.root.join(".git").is_file(),
        "{}'s root is not a worktree checkout (no .git file): {workspace:?}",
        workspace.member_name
    );
    assert!(
        workspace.root.starts_with(worktrees_dir),
        "{} lives outside the provider layout: {workspace:?}",
        workspace.member_name
    );
    assert_eq!(
        workspace.team_name, TEAM,
        "the workspace records its team: {workspace:?}"
    );
    assert_eq!(
        workspace.session_id, session,
        "the workspace records the hosting session: {workspace:?}"
    );
    assert_eq!(
        canonical(&workspace.root),
        canonical(&info.path),
        "the recorded worktree path must be stable"
    );
    assert_ne!(
        canonical(&workspace.root),
        repo_root,
        "{} was handed the repository's own working tree",
        workspace.member_name
    );
    assert!(
        listed.contains(&canonical(&workspace.root)),
        "{}'s worktree is not registered in `git worktree list` ({}): {listed:?}",
        workspace.member_name,
        workspace.root.display()
    );
    assert_eq!(
        factory.workspace_root(&workspace.member_name).as_deref(),
        Some(workspace.root.as_path()),
        "the harness of {} was not built with its worktree",
        workspace.member_name
    );
}

#[tokio::test]
async fn every_member_of_a_worktree_team_gets_its_own_registered_worktree() {
    let repo = TempRepo::init();
    let layout = TempDir::new().expect("a temporary layout directory");
    let provider = worktree_provider(layout.path(), repo.path());
    let spec = with_second_teammate(spec_with_workspace(json!({"isolation": "worktree"})));

    let (manager, agent, factory) = activate(&spec, &provider).await;
    spawn_teammates(&agent).await;

    // The built team generates its own hosting session; every member workspace lives under it.
    let session = agent.session_id().to_owned();
    let allocations = provider.allocations();
    let mut members: Vec<String> = allocations
        .iter()
        .map(|workspace| workspace.member_name.clone())
        .collect();
    members.sort();
    assert_eq!(
        members,
        agent.roster(),
        "one worktree per roster member: {allocations:?}"
    );
    for name in ["team_leader", "dev-1", "dev-2"] {
        assert!(
            members.iter().any(|member| member == name),
            "'{name}' was not allocated a worktree: {allocations:?}"
        );
    }

    let repo_root = canonical(repo.path());
    let worktrees_dir = layout
        .path()
        .join(TEAM)
        .join("sessions")
        .join(&session)
        .join("worktrees");
    let listed: BTreeSet<PathBuf> = repo
        .worktree_paths()
        .iter()
        .map(|path| canonical(path))
        .collect();
    let mut roots = BTreeSet::new();

    for workspace in &allocations {
        assert_registered_worktree(
            workspace,
            &repo_root,
            &worktrees_dir,
            &session,
            &listed,
            &factory,
        );
        roots.insert(canonical(&workspace.root));
    }

    assert_eq!(
        roots.len(),
        allocations.len(),
        "every member needs a distinct worktree: {allocations:?}"
    );
    assert_eq!(
        listed.len(),
        allocations.len() + 1,
        "one git worktree per member plus the main working tree: {listed:?}"
    );

    // Two members built from the same teammate blueprint still get their own checkout.
    let first = spec.member_spec("dev-1").expect("dev-1 is on the roster");
    let second = spec.member_spec("dev-2").expect("dev-2 is on the roster");
    assert_eq!(
        first.base.prompt, second.base.prompt,
        "the two teammates share one blueprint"
    );
    let dev_one = provider.workspace("dev-1").expect("dev-1 allocation");
    let dev_two = provider.workspace("dev-2").expect("dev-2 allocation");
    assert_ne!(dev_one.root, dev_two.root);
    assert_ne!(
        dev_one.worktree.as_ref().map(|info| info.name.clone()),
        dev_two.worktree.as_ref().map(|info| info.name.clone()),
        "the worktree names are per member"
    );

    timeout(BOUND, manager.stop_team(TEAM, SESSION))
        .await
        .expect("stopping the team did not finish within the bound")
        .expect("stopping the team failed");
}

#[tokio::test]
async fn a_bare_worktree_string_isolates_the_leader_too() {
    let repo = TempRepo::init();
    let layout = TempDir::new().expect("a temporary layout directory");
    let provider = worktree_provider(layout.path(), repo.path());
    let spec = spec_with_workspace(json!("worktree"));

    let (manager, _agent, _factory) = activate(&spec, &provider).await;

    let leader = provider
        .workspace("team_leader")
        .expect("the leader is allocated at build");
    assert_eq!(
        leader.isolation,
        IsolationMode::Worktree,
        "Python accepts the bare \"worktree\" string as well: {leader:?}"
    );
    assert!(
        leader.root.join(".git").is_file(),
        "the leader did not get a real worktree: {leader:?}"
    );

    timeout(BOUND, manager.stop_team(TEAM, SESSION))
        .await
        .expect("stopping the team did not finish within the bound")
        .expect("stopping the team failed");
}

#[tokio::test]
async fn stopping_the_team_releases_the_member_worktrees_and_unregisters_them() {
    let repo = TempRepo::init();
    let layout = TempDir::new().expect("a temporary layout directory");
    let provider = worktree_provider(layout.path(), repo.path());
    let spec = with_second_teammate(spec_with_workspace(json!({"isolation": "worktree"})));

    let (manager, agent, _factory) = activate(&spec, &provider).await;
    spawn_teammates(&agent).await;

    let allocations = provider.allocations();
    assert_eq!(
        allocations.len(),
        agent.roster().len(),
        "one worktree per roster member: {}",
        agent.roster().len()
    );
    assert!(allocations.len() >= 3, "leader plus two teammates");
    let roots: Vec<PathBuf> = allocations.iter().map(|w| w.root.clone()).collect();
    assert!(
        provider.releases().is_empty(),
        "the team holds its worktrees while it is live"
    );

    timeout(BOUND, manager.stop_team(TEAM, SESSION))
        .await
        .expect("stopping the team did not finish within the bound")
        .expect("stopping the team failed");

    // Teardown is the product path: `TeamAgent::stop` hands every retained workspace back to the
    // provider it allocated from, which removes the worktree and deletes its branch. Nothing here
    // releases anything itself.
    let mut released: Vec<String> = provider
        .releases()
        .iter()
        .map(|workspace| workspace.member_name.clone())
        .collect();
    released.sort();
    let mut allocated: Vec<String> = allocations
        .iter()
        .map(|workspace| workspace.member_name.clone())
        .collect();
    allocated.sort();
    assert_eq!(
        released, allocated,
        "the runtime releases every member worktree exactly once"
    );
    wait_until("every member worktree to be removed from disk", || {
        roots.iter().all(|root| !root.exists())
    })
    .await;

    let listed: BTreeSet<PathBuf> = repo
        .worktree_paths()
        .iter()
        .map(|path| canonical(path))
        .collect();
    assert_eq!(
        listed,
        BTreeSet::from([canonical(repo.path())]),
        "the repository keeps only its own working tree: {listed:?}"
    );
    for root in &roots {
        assert!(
            !listed.contains(&canonical(root)),
            "{} is still registered after its release",
            root.display()
        );
    }
    assert_eq!(
        repo.worktree_branches(),
        Vec::<String>::new(),
        "the worktree branches are deleted with their worktrees"
    );
}

#[tokio::test]
async fn a_team_without_the_metadata_key_stays_on_plain_local_directories() {
    let repo = TempRepo::init();
    let layout = TempDir::new().expect("a temporary layout directory");
    let provider = worktree_provider(layout.path(), repo.path());
    let mut spec = sample_spec();
    spec.workspace.enabled = true;

    let (manager, agent, _factory) = activate(&spec, &provider).await;
    spawn_teammates(&agent).await;

    let session = agent.session_id().to_owned();
    let allocations = provider.allocations();
    assert_eq!(
        allocations.len(),
        agent.roster().len(),
        "every roster member is allocated a local directory"
    );
    for workspace in &allocations {
        assert_eq!(
            workspace.isolation,
            IsolationMode::None,
            "no metadata key means no isolation: {workspace:?}"
        );
        assert!(workspace.is_shared());
        assert!(workspace.worktree.is_none());
        assert!(
            workspace
                .root
                .starts_with(layout.path().join(TEAM).join("sessions").join(&session)),
            "{} lives outside the local layout: {workspace:?}",
            workspace.member_name
        );
        assert!(
            workspace.root.is_dir(),
            "the shared workspace directory exists: {workspace:?}"
        );
        assert!(
            !workspace.root.join(".git").exists(),
            "a plain local directory carries no worktree marker: {workspace:?}"
        );
    }
    assert_eq!(
        repo.worktree_paths().len(),
        1,
        "no worktree was registered in the repository"
    );

    // Releasing a shared workspace retains it, per Python's cleanup-on-team-delete rule.
    let leader = provider
        .workspace("team_leader")
        .expect("the leader allocation");
    timeout(BOUND, provider.release(&leader))
        .await
        .expect("releasing the shared workspace did not finish within the bound")
        .expect("releasing the shared workspace failed");
    assert!(
        leader.root.is_dir(),
        "a shared workspace is retained after release: {leader:?}"
    );

    timeout(BOUND, manager.stop_team(TEAM, SESSION))
        .await
        .expect("stopping the team did not finish within the bound")
        .expect("stopping the team failed");
}
