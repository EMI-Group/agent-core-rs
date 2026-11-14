//! Watch targets, the workspace plan and classification of raw changes.
//!
//! A [`WatchTarget`] is one path the watcher cares about, together with the
//! [`WatchTargetKind`] it belongs to and a stable label a reload handler can
//! switch on. A [`WatchPlan`] collects targets and can discover every harness
//! root of a workspace plus its [`OpenJiuwenHome`].
//!
//! [`classify`] turns a raw backend change into the [`DebounceInput`]s that
//! enter the debouncer: it drops paths no target covers, resolves a path
//! covered by several targets (file target first, then the deepest directory
//! target), and maps renames across the watched tree.
//!
//! Nothing here touches a filesystem handle; `validate` and `existing_only`
//! are the only functions that stat paths, and both report what they skipped or
//! rejected instead of silently ignoring it.
use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};

use crate::cli::instruction::{IDENTITY_FILENAME, MEMORY_FILENAME};
use crate::cli::paths::{mcp_config_paths, OPENJIUWEN_MCP_CONFIG};
use crate::cli::EnvSource;
use crate::spec::AGENTS_DIR;
use crate::watcher::debounce::DebounceInput;
use crate::watcher::event::{ChangeKind, WatchTargetKind};

/// Directory of agent templates that a workspace, the home directory and the
/// plugin root share.
///
/// Mirrors the private `spec::plugin::PLUGINS_DIR`; the plugin constant is not
/// public, so the spelling is repeated here and kept in sync by the tests.
const PLUGINS_DIR: &str = ".openjiuwen/plugins";

/// One path the watcher observes.
///
/// A `directory` target recursively covers the directory and everything below
/// it. A `file` target observes exactly one path: it watches the file's parent
/// directory non-recursively and filters to that path, so a delete→recreate
/// cycle — or a file that does not exist yet, such as `OPENJIUWEN.md` — is still
/// seen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchTarget {
    kind: WatchTargetKind,
    path: PathBuf,
    recursive: bool,
    label: Option<String>,
}

impl WatchTarget {
    /// Creates a recursive directory target.
    #[must_use]
    pub fn directory(kind: WatchTargetKind, path: impl Into<PathBuf>) -> Self {
        Self {
            kind,
            path: path.into(),
            recursive: true,
            label: None,
        }
    }

    /// Creates a single-file target that watches the file's parent directory.
    #[must_use]
    pub fn file(kind: WatchTargetKind, path: impl Into<PathBuf>) -> Self {
        Self {
            kind,
            path: path.into(),
            recursive: false,
            label: None,
        }
    }

    /// Sets the stable label reported with every event of this target.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns the kind of harness root this target belongs to.
    #[must_use]
    pub fn kind(&self) -> WatchTargetKind {
        self.kind
    }

    /// Returns the path this target represents.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the target's label, when one was set.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns whether the target covers a whole directory tree.
    ///
    /// `false` for a single-file target.
    #[must_use]
    pub fn is_recursive(&self) -> bool {
        self.recursive
    }

    /// Returns the path the backend has to watch.
    ///
    /// A directory target watches its own path; a file target watches the
    /// parent directory (`.` when the path has no usable parent).
    #[must_use]
    pub fn watch_path(&self) -> &Path {
        if self.recursive {
            return &self.path;
        }
        match self.path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        }
    }

    /// Returns whether `path` belongs to this target.
    ///
    /// A directory target matches itself and every descendant; a file target
    /// matches only the exact path.
    #[must_use]
    pub fn matches(&self, path: &Path) -> bool {
        if self.recursive {
            path.starts_with(&self.path)
        } else {
            path == self.path
        }
    }
}

/// The set of targets a watcher observes, plus the paths it had to skip.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WatchPlan {
    targets: Vec<WatchTarget>,
    skipped: Vec<PathBuf>,
}

impl WatchPlan {
    /// Creates an empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether the plan holds no target.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Returns how many targets the plan holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Returns the targets in insertion order.
    #[must_use]
    pub fn targets(&self) -> &[WatchTarget] {
        &self.targets
    }

    /// Returns the watch paths dropped by [`Self::existing_only`], in discovery
    /// order.
    #[must_use]
    pub fn skipped(&self) -> &[PathBuf] {
        &self.skipped
    }

    /// Appends one target.
    pub fn push(&mut self, target: WatchTarget) {
        self.targets.push(target);
    }

    /// Appends one target and returns the plan, for chained construction.
    #[must_use]
    pub fn with_target(mut self, target: WatchTarget) -> Self {
        self.push(target);
        self
    }

    /// Discovers every harness root of `workspace` and `home`.
    ///
    /// Skill libraries, agent templates (workspace, home and plugins),
    /// instruction documents and MCP configurations are covered, each with the
    /// stable label its reload handler keys on. Nothing has to exist yet.
    #[must_use]
    pub fn workspace(workspace: &Path, home: &OpenJiuwenHome, env: &dyn EnvSource) -> Self {
        let mut plan = Self::new();
        plan.push(
            WatchTarget::directory(WatchTargetKind::Skills, workspace.join("skills"))
                .with_label("skills:workspace"),
        );
        plan.push(
            WatchTarget::directory(WatchTargetKind::Spec, workspace.join(AGENTS_DIR))
                .with_label("spec:workspace"),
        );
        plan.push(
            WatchTarget::directory(WatchTargetKind::Spec, home.join(AGENTS_DIR))
                .with_label("spec:home"),
        );
        plan.push(
            WatchTarget::directory(WatchTargetKind::Spec, workspace.join(PLUGINS_DIR))
                .with_label("spec:plugins"),
        );
        for (path, label) in [
            (
                workspace.join(IDENTITY_FILENAME),
                "instruction:workspace:IDENTITY.md",
            ),
            (home.join(IDENTITY_FILENAME), "instruction:home:IDENTITY.md"),
            (home.join(MEMORY_FILENAME), "instruction:home:OPENJIUWEN.md"),
            (
                workspace.join(MEMORY_FILENAME),
                "instruction:workspace:OPENJIUWEN.md",
            ),
        ] {
            plan.push(WatchTarget::file(WatchTargetKind::InstructionFile, path).with_label(label));
        }

        let overridden = env.get(OPENJIUWEN_MCP_CONFIG).is_some();
        for (index, path) in mcp_config_paths(home, workspace, env)
            .into_iter()
            .enumerate()
        {
            let label = if overridden {
                "mcp:override"
            } else if index == 0 {
                "mcp:home"
            } else {
                "mcp:workspace"
            };
            plan.push(WatchTarget::file(WatchTargetKind::McpConfig, path).with_label(label));
        }
        plan
    }

    /// Drops every target whose watch path does not exist.
    ///
    /// Dropped watch paths are recorded in [`Self::skipped`] rather than
    /// silently ignored; a target whose file does not exist yet still survives
    /// when its parent directory does, because that is exactly what makes a
    /// later create observable.
    #[must_use]
    pub fn existing_only(mut self) -> Self {
        let targets = std::mem::take(&mut self.targets);
        let mut kept = Vec::with_capacity(targets.len());
        for target in targets {
            let watch_path = target.watch_path().to_path_buf();
            if watch_path.exists() {
                kept.push(target);
            } else {
                self.skipped.push(watch_path);
            }
        }
        self.targets = kept;
        self
    }

    /// Checks that every watch path exists and has the expected file type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::not_found`] when a watch path is missing,
    /// [`Error::sys_op`] when it cannot be inspected (for example a permission
    /// failure), and [`Error::config`] when a directory target points at a file
    /// or a file target points at a directory. Every message names both the
    /// offending path and the target label.
    pub fn validate(&self) -> Result<()> {
        for target in &self.targets {
            let watch_path = target.watch_path();
            let metadata = std::fs::metadata(watch_path)
                .map_err(|error| map_watch_error(target, watch_path, &error))?;
            if !metadata.is_dir() {
                return Err(Error::config(format!(
                    "watch target `{}` expects a directory at {} but found a file",
                    target_name(target),
                    watch_path.display()
                )));
            }
            if target.is_recursive() {
                continue;
            }
            // A file target tolerates a missing file (it may be created later)
            // but rejects a directory where the file should be.
            match std::fs::metadata(target.path()) {
                Ok(metadata) if metadata.is_dir() => {
                    return Err(Error::config(format!(
                        "watch target `{}` expects a file at {} but found a directory",
                        target_name(target),
                        target.path().display()
                    )));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(map_watch_error(target, target.path(), &error)),
            }
        }
        Ok(())
    }
}

/// Raw change kinds reported by the OS backend, before target resolution.
///
/// Kept crate-private: it is the vocabulary of the backend adapter added with
/// the `notify` feature, not part of the public watcher API.
#[allow(dead_code)] // consumed by the notify backend added in the next round
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawChangeKind {
    /// The path appeared.
    Created,
    /// The path content or metadata changed.
    Modified,
    /// The path disappeared.
    Removed,
    /// The path moved; the first two paths are `(from, to)`.
    Renamed,
    /// Any other backend event.
    Other,
}

/// One raw backend change, as reported before debouncing.
#[allow(dead_code)] // consumed by the notify backend added in the next round
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawChange {
    /// How the watched paths changed.
    pub kind: RawChangeKind,
    /// The paths the change applies to; `(from, to)` for a rename.
    pub paths: Vec<PathBuf>,
}

/// Resolves a raw backend change against the watched targets.
///
/// Only paths covered by a target survive. A path covered by several targets is
/// attributed to the file target first and otherwise to the deepest matching
/// directory target. A rename inside the tree becomes one `Renamed` event at the
/// destination carrying the source path; a move out of the tree becomes
/// `Removed` at the source and a move into the tree `Created` at the
/// destination. Any non-created/removed/renamed backend kind reads as
/// `Modified`.
#[allow(dead_code)] // consumed by the notify backend added in the next round
pub(crate) fn classify(targets: &[WatchTarget], raw: &RawChange) -> Vec<DebounceInput> {
    if raw.kind == RawChangeKind::Renamed {
        return classify_rename(targets, &raw.paths);
    }
    let Some(change) = change_for(raw.kind) else {
        return Vec::new();
    };
    raw.paths
        .iter()
        .filter_map(|path| {
            let target = best_target(targets, path)?;
            Some(input_for(target, path, change, None))
        })
        .collect()
}

/// Maps a raw kind onto the public change vocabulary.
///
/// `Renamed` is handled by [`classify_rename`] because it needs both paths;
/// `Other` reads as `Modified` so an unrecognised backend event still reloads.
fn change_for(kind: RawChangeKind) -> Option<ChangeKind> {
    match kind {
        RawChangeKind::Created => Some(ChangeKind::Created),
        RawChangeKind::Modified | RawChangeKind::Other => Some(ChangeKind::Modified),
        RawChangeKind::Removed => Some(ChangeKind::Removed),
        RawChangeKind::Renamed => None,
    }
}

/// Resolves a rename reported as `(from, to)`.
fn classify_rename(targets: &[WatchTarget], paths: &[PathBuf]) -> Vec<DebounceInput> {
    let endpoints = paths
        .split_first()
        .and_then(|(from, rest)| rest.first().map(|to| (from, to)));
    let Some((from, to)) = endpoints else {
        // A rename without both endpoints cannot be resolved as a move; report
        // it at whichever path a target covers.
        return paths
            .iter()
            .filter_map(|path| {
                let target = best_target(targets, path)?;
                Some(input_for(target, path, ChangeKind::Renamed, None))
            })
            .collect();
    };
    match (best_target(targets, from), best_target(targets, to)) {
        (Some(_), Some(to_target)) => vec![input_for(
            to_target,
            to,
            ChangeKind::Renamed,
            Some(from.clone()),
        )],
        (Some(from_target), None) => vec![input_for(from_target, from, ChangeKind::Removed, None)],
        (None, Some(to_target)) => vec![input_for(to_target, to, ChangeKind::Created, None)],
        (None, None) => Vec::new(),
    }
}

/// Returns the target a path is attributed to.
///
/// A file target outranks a directory target; among directory targets the
/// deepest (longest) matching path wins, so a nested root overrides the tree
/// that contains it.
fn best_target<'a>(targets: &'a [WatchTarget], path: &Path) -> Option<&'a WatchTarget> {
    let mut file_match: Option<&WatchTarget> = None;
    let mut deepest_dir: Option<&WatchTarget> = None;
    for target in targets {
        if !target.matches(path) {
            continue;
        }
        if target.is_recursive() {
            let deeper = deepest_dir.is_none_or(|current: &WatchTarget| {
                target.path().as_os_str().len() > current.path().as_os_str().len()
            });
            if deeper {
                deepest_dir = Some(target);
            }
        } else {
            file_match = Some(target);
        }
    }
    file_match.or(deepest_dir)
}

/// Builds the debouncer input for one resolved change.
fn input_for(
    target: &WatchTarget,
    path: &Path,
    change: ChangeKind,
    previous_path: Option<PathBuf>,
) -> DebounceInput {
    DebounceInput {
        kind: target.kind(),
        path: path.to_path_buf(),
        change,
        previous_path,
        label: target.label().map(str::to_owned),
    }
}

/// Returns the label used in error messages, `"<unlabelled>"` when unset.
fn target_name(target: &WatchTarget) -> &str {
    target.label().unwrap_or("<unlabelled>")
}

/// Maps an `io::Error` from a watch-path probe onto a harness error.
///
/// A missing path is [`Error::not_found`]; a permission failure and every other
/// IO error are [`Error::sys_op`]. The message always names the path and the
/// target label.
fn map_watch_error(target: &WatchTarget, path: &Path, error: &std::io::Error) -> Error {
    let message = format!(
        "watch target `{}` cannot watch {}: {error}",
        target_name(target),
        path.display()
    );
    if error.kind() == std::io::ErrorKind::NotFound {
        return Error::not_found(message);
    }
    Error::sys_op(message)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use openjiuwen_common::error::ErrorKind;

    fn temp_workspace() -> (tempfile::TempDir, PathBuf, OpenJiuwenHome) {
        let root = tempfile::tempdir().unwrap();
        let workspace = root.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let home = OpenJiuwenHome::from_root(root.path().join("home"));
        (root, workspace, home)
    }

    fn labels(plan: &WatchPlan) -> Vec<&str> {
        plan.targets()
            .iter()
            .filter_map(|target| target.label())
            .collect()
    }

    #[test]
    fn directory_target_matches_itself_and_descendants() {
        let target = WatchTarget::directory(WatchTargetKind::Skills, "/w/skills");
        assert!(target.is_recursive());
        assert_eq!(target.watch_path(), Path::new("/w/skills"));
        assert!(target.matches(Path::new("/w/skills")));
        assert!(target.matches(Path::new("/w/skills/a/b.md")));
        assert!(!target.matches(Path::new("/w/skills-other/a.md")));
    }

    #[test]
    fn file_target_watches_its_parent_and_matches_exactly() {
        let target = WatchTarget::file(WatchTargetKind::InstructionFile, "/w/OPENJIUWEN.md");
        assert!(!target.is_recursive());
        assert_eq!(target.watch_path(), Path::new("/w"));
        assert!(target.matches(Path::new("/w/OPENJIUWEN.md")));
        assert!(!target.matches(Path::new("/w/IDENTITY.md")));
        assert!(!target.matches(Path::new("/w/sub/OPENJIUWEN.md")));
    }

    #[test]
    fn file_target_without_parent_falls_back_to_the_current_directory() {
        let target = WatchTarget::file(WatchTargetKind::McpConfig, "mcp.json");
        assert_eq!(target.watch_path(), Path::new("."));
        let labelled = target.with_label("mcp:override");
        assert_eq!(labelled.label(), Some("mcp:override"));
        assert_eq!(labelled.path(), Path::new("mcp.json"));
    }

    #[test]
    fn workspace_plan_discovers_every_labelled_target() {
        let (_root, workspace, home) = temp_workspace();
        let env = crate::cli::MapEnv::new();
        let plan = WatchPlan::workspace(&workspace, &home, &env);
        assert!(!plan.is_empty());
        assert_eq!(plan.len(), 10);
        assert_eq!(
            labels(&plan),
            vec![
                "skills:workspace",
                "spec:workspace",
                "spec:home",
                "spec:plugins",
                "instruction:workspace:IDENTITY.md",
                "instruction:home:IDENTITY.md",
                "instruction:home:OPENJIUWEN.md",
                "instruction:workspace:OPENJIUWEN.md",
                "mcp:home",
                "mcp:workspace",
            ]
        );

        let kinds: Vec<WatchTargetKind> = plan.targets().iter().map(WatchTarget::kind).collect();
        assert_eq!(kinds[0], WatchTargetKind::Skills);
        assert_eq!(kinds[1], WatchTargetKind::Spec);
        assert_eq!(kinds[4], WatchTargetKind::InstructionFile);
        assert_eq!(kinds[8], WatchTargetKind::McpConfig);

        assert_eq!(
            plan.targets()[1].path(),
            workspace.join(AGENTS_DIR).as_path()
        );
        assert_eq!(plan.targets()[2].path(), home.join(AGENTS_DIR).as_path());
        assert_eq!(
            plan.targets()[3].path(),
            workspace.join(PLUGINS_DIR).as_path()
        );
        assert_eq!(plan.skipped().len(), 0);
    }

    #[test]
    fn workspace_plan_honours_the_mcp_config_override() {
        let (_root, workspace, home) = temp_workspace();
        let env = crate::cli::MapEnv::from_pairs([(OPENJIUWEN_MCP_CONFIG, "custom/mcp.json")]);
        let plan = WatchPlan::workspace(&workspace, &home, &env);
        assert_eq!(plan.len(), 9);
        let labels = labels(&plan);
        assert_eq!(labels.last(), Some(&"mcp:override"));
        assert!(!labels.contains(&"mcp:home"));
        assert_eq!(
            plan.targets().last().unwrap().path(),
            &home.join("custom/mcp.json")
        );
    }

    #[test]
    fn existing_only_keeps_reachable_targets_and_reports_skipped() {
        let (_root, workspace, home) = temp_workspace();
        std::fs::create_dir_all(home.root()).unwrap();
        std::fs::create_dir_all(workspace.join("skills")).unwrap();
        std::fs::write(workspace.join(MEMORY_FILENAME), "memory").unwrap();
        let env = crate::cli::MapEnv::new();

        let plan = WatchPlan::workspace(&workspace, &home, &env).existing_only();
        assert!(plan
            .targets()
            .iter()
            .all(|target| target.watch_path().exists()));

        let kept = labels(&plan);
        assert!(kept.contains(&"skills:workspace"));
        assert!(kept.contains(&"instruction:workspace:OPENJIUWEN.md"));
        assert!(kept.contains(&"mcp:home"));
        assert!(!kept.contains(&"spec:workspace"));
        assert!(!kept.contains(&"mcp:workspace"));

        let skipped = plan.skipped().to_vec();
        assert!(skipped.contains(&workspace.join(AGENTS_DIR)));
        assert!(skipped.contains(&home.join(AGENTS_DIR)));
        assert!(skipped.contains(&workspace.join(PLUGINS_DIR)));
        assert!(skipped.contains(&workspace.join(".openjiuwen")));
        assert_eq!(plan.len() + skipped.len(), 10);
    }

    #[test]
    fn validate_accepts_an_existing_tree() {
        let (_root, workspace, home) = temp_workspace();
        std::fs::create_dir_all(home.root()).unwrap();
        std::fs::create_dir_all(workspace.join("skills")).unwrap();
        std::fs::create_dir_all(workspace.join(AGENTS_DIR)).unwrap();
        std::fs::create_dir_all(home.join(AGENTS_DIR)).unwrap();
        std::fs::create_dir_all(workspace.join(PLUGINS_DIR)).unwrap();
        std::fs::create_dir_all(workspace.join(".openjiuwen")).unwrap();
        let env = crate::cli::MapEnv::new();
        let plan = WatchPlan::workspace(&workspace, &home, &env).existing_only();
        plan.validate().unwrap();
    }

    #[test]
    fn validate_reports_a_missing_watch_path() {
        let (_root, workspace, home) = temp_workspace();
        let env = crate::cli::MapEnv::new();
        let plan = WatchPlan::workspace(&workspace, &home, &env);
        let error = plan.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        let message = error.message().into_owned();
        assert!(message.contains("skills:workspace"), "{message}");
        assert!(message.contains("skills"), "{message}");
    }

    #[test]
    fn validate_rejects_a_directory_target_pointing_at_a_file() {
        let (_root, workspace, _home) = temp_workspace();
        std::fs::write(workspace.join("skills"), "not a directory").unwrap();
        let plan = WatchPlan::new().with_target(
            WatchTarget::directory(WatchTargetKind::Skills, workspace.join("skills"))
                .with_label("skills:workspace"),
        );
        let error = plan.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        let message = error.message().into_owned();
        assert!(message.contains("skills:workspace"), "{message}");
    }

    #[test]
    fn validate_rejects_a_file_target_pointing_at_a_directory() {
        let (_root, workspace, _home) = temp_workspace();
        std::fs::create_dir_all(workspace.join(MEMORY_FILENAME)).unwrap();
        let plan = WatchPlan::new().with_target(
            WatchTarget::file(
                WatchTargetKind::InstructionFile,
                workspace.join(MEMORY_FILENAME),
            )
            .with_label("instruction:workspace:OPENJIUWEN.md"),
        );
        let error = plan.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        let message = error.message().into_owned();
        assert!(
            message.contains("instruction:workspace:OPENJIUWEN.md"),
            "{message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn validate_maps_a_permission_failure_to_sys_op() {
        use std::os::unix::fs::PermissionsExt;

        let (_root, workspace, _home) = temp_workspace();
        let locked = workspace.join(".openjiuwen");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();

        // Probe the exact path `validate` reads while the lock is in place, so
        // the test states what the process is actually allowed to do.
        let probe = std::fs::metadata(locked.join("mcp.json")).unwrap_err();
        let plan = WatchPlan::new().with_target(
            WatchTarget::file(WatchTargetKind::McpConfig, locked.join("mcp.json"))
                .with_label("mcp:workspace"),
        );
        let validation = plan.validate();
        // Restore before asserting so a failure still leaves a removable tree.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

        let error = validation.unwrap_err();
        if probe.kind() == std::io::ErrorKind::PermissionDenied {
            assert_eq!(error.kind(), ErrorKind::SysOp);
            let message = error.message().into_owned();
            assert!(message.contains("mcp:workspace"), "{message}");
            assert!(message.contains(".openjiuwen"), "{message}");
        } else {
            // The process bypasses the permission bits (typically root), so the
            // locked path simply does not exist.
            assert_eq!(error.kind(), ErrorKind::NotFound);
        }
    }

    #[test]
    fn classify_drops_paths_no_target_covers() {
        let targets = vec![WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")
            .with_label("skills:workspace")];
        let raw = RawChange {
            kind: RawChangeKind::Created,
            paths: vec![
                PathBuf::from("/w/skills/a.md"),
                PathBuf::from("/w/other/b.md"),
            ],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].path, PathBuf::from("/w/skills/a.md"));
        assert_eq!(inputs[0].kind, WatchTargetKind::Skills);
        assert_eq!(inputs[0].label.as_deref(), Some("skills:workspace"));
    }

    #[test]
    fn classify_prefers_a_file_target_over_a_directory_target() {
        let file = "/w/skills/OPENJIUWEN.md";
        let targets = vec![
            WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")
                .with_label("skills:workspace"),
            WatchTarget::file(WatchTargetKind::InstructionFile, file)
                .with_label("instruction:workspace:OPENJIUWEN.md"),
        ];
        let raw = RawChange {
            kind: RawChangeKind::Modified,
            paths: vec![PathBuf::from(file)],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].kind, WatchTargetKind::InstructionFile);
        assert_eq!(
            inputs[0].label.as_deref(),
            Some("instruction:workspace:OPENJIUWEN.md")
        );
    }

    #[test]
    fn classify_prefers_the_deepest_matching_directory() {
        let targets = vec![
            WatchTarget::directory(WatchTargetKind::Spec, "/w/.openjiuwen")
                .with_label("spec:workspace"),
            WatchTarget::directory(WatchTargetKind::Spec, "/w/.openjiuwen/plugins")
                .with_label("spec:plugins"),
        ];
        let raw = RawChange {
            kind: RawChangeKind::Created,
            paths: vec![PathBuf::from("/w/.openjiuwen/plugins/a/plugin.yaml")],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].label.as_deref(), Some("spec:plugins"));

        // The order of the targets must not matter.
        let mut reversed = targets;
        reversed.reverse();
        let inputs = classify(&reversed, &raw);
        assert_eq!(inputs[0].label.as_deref(), Some("spec:plugins"));
    }

    #[test]
    fn classify_maps_other_to_modified_and_keeps_created_and_removed() {
        let targets = vec![
            WatchTarget::directory(WatchTargetKind::Spec, "/w/.openjiuwen")
                .with_label("spec:workspace"),
        ];
        for (raw_kind, expected) in [
            (RawChangeKind::Created, ChangeKind::Created),
            (RawChangeKind::Modified, ChangeKind::Modified),
            (RawChangeKind::Other, ChangeKind::Modified),
            (RawChangeKind::Removed, ChangeKind::Removed),
        ] {
            let raw = RawChange {
                kind: raw_kind,
                paths: vec![PathBuf::from("/w/.openjiuwen/a.yaml")],
            };
            let inputs = classify(&targets, &raw);
            assert_eq!(inputs.len(), 1, "{raw_kind:?}");
            assert_eq!(inputs[0].change, expected, "{raw_kind:?}");
            assert!(inputs[0].previous_path.is_none(), "{raw_kind:?}");
        }
    }

    #[test]
    fn classify_rename_inside_the_tree_reports_the_destination() {
        let targets = vec![WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")
            .with_label("skills:workspace")];
        let raw = RawChange {
            kind: RawChangeKind::Renamed,
            paths: vec![
                PathBuf::from("/w/skills/old.md"),
                PathBuf::from("/w/skills/new.md"),
            ],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].change, ChangeKind::Renamed);
        assert_eq!(inputs[0].path, PathBuf::from("/w/skills/new.md"));
        assert_eq!(
            inputs[0].previous_path.as_deref(),
            Some(Path::new("/w/skills/old.md"))
        );
    }

    #[test]
    fn classify_rename_out_of_the_tree_reports_removal_at_the_source() {
        let targets = vec![WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")
            .with_label("skills:workspace")];
        let raw = RawChange {
            kind: RawChangeKind::Renamed,
            paths: vec![
                PathBuf::from("/w/skills/old.md"),
                PathBuf::from("/elsewhere/new.md"),
            ],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].change, ChangeKind::Removed);
        assert_eq!(inputs[0].path, PathBuf::from("/w/skills/old.md"));
        assert!(inputs[0].previous_path.is_none());
    }

    #[test]
    fn classify_rename_into_the_tree_reports_creation_at_the_destination() {
        let targets = vec![WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")
            .with_label("skills:workspace")];
        let raw = RawChange {
            kind: RawChangeKind::Renamed,
            paths: vec![
                PathBuf::from("/elsewhere/old.md"),
                PathBuf::from("/w/skills/new.md"),
            ],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].change, ChangeKind::Created);
        assert_eq!(inputs[0].path, PathBuf::from("/w/skills/new.md"));
        assert!(inputs[0].previous_path.is_none());
    }

    #[test]
    fn classify_rename_ignores_paths_outside_every_target() {
        let targets = vec![WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")];
        let raw = RawChange {
            kind: RawChangeKind::Renamed,
            paths: vec![PathBuf::from("/a/old.md"), PathBuf::from("/b/new.md")],
        };
        assert_eq!(classify(&targets, &raw).len(), 0);
    }

    #[test]
    fn classify_rename_with_one_endpoint_reports_that_path() {
        let targets = vec![WatchTarget::directory(WatchTargetKind::Skills, "/w/skills")
            .with_label("skills:workspace")];
        let raw = RawChange {
            kind: RawChangeKind::Renamed,
            paths: vec![PathBuf::from("/w/skills/old.md")],
        };
        let inputs = classify(&targets, &raw);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].change, ChangeKind::Renamed);
        assert!(inputs[0].previous_path.is_none());
    }

    #[test]
    fn plan_push_and_with_target_accumulate_in_order() {
        let mut plan = WatchPlan::new();
        assert!(plan.is_empty());
        plan.push(WatchTarget::directory(WatchTargetKind::Other, "/a"));
        let plan = plan.with_target(WatchTarget::directory(WatchTargetKind::Other, "/b"));
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.targets()[0].path(), Path::new("/a"));
        assert_eq!(plan.targets()[1].path(), Path::new("/b"));
        assert!(plan.targets()[0].label().is_none());
    }
}
