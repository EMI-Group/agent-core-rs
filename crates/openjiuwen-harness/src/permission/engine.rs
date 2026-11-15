//! Facade of the permission pipeline: folds pipelines A/B/C and inlines the
//! packaged builtin floors.
//!
//! Ports `openjiuwen/harness/security/permission_engine/core.py`
//! (`PermissionEngine`, `prepare_permissions_for_engine`, `strictest`, v0.1.20).
//!
//! The decision path is three layers deep and strictest-wins
//! ([`strictest`], `deny < ask < allow`):
//!
//! 1. [`tiered::evaluate_tiered_policy`] — the tool/parameter ladder, with no
//!    opinion only when [`tiered::FALLBACK_NO_CONFIG`] is returned;
//! 2. [`FileGuardChecker::evaluate`] — path read/write/exec policy;
//! 3. [`NetGuardChecker::evaluate`] — URL allow/deny policy for fetch tools.
//!
//! The engine performs no IO: it owns the compiled policy behind a
//! [`parking_lot::RwLock`] and is shared as an `Arc<PermissionEngine>`. The HITL
//! confirmation flow lives in [`crate::permission::interrupt`].
//!
//! Legacy host YAML is upgraded once, at construction time, by
//! [`prepare_permissions_for_engine`] (idempotent via the `layer: "builtin"`
//! marker), which also fills missing rule `action`s from `severity`.
//!
//! Size note: ~464 lines of implementation plus an inline test module; the decision
//! path (engine, precedence folding, project-root boundary, session-overlay
//! hot-update) is one unit, and the 1000-line overrun is the test module alone.

use std::path::PathBuf;

use parking_lot::RwLock;
use serde_json::Value;

use crate::permission::builtin;
use crate::permission::file_guard::FileGuardChecker;
use crate::permission::model::{PermissionLevel, PermissionResult, PermissionsSection};
use crate::permission::net_guard::NetGuardChecker;
use crate::permission::paths;
use crate::permission::rules;
use crate::permission::tiered;

/// Marker of a packaged rule/path entry (Python `layer == "builtin"`).
const BUILTIN_LAYER: &str = "builtin";

/// Label used when pipeline A had no opinion (Python `matched_rule = "default"`).
const DEFAULT_LABEL: &str = "default";

/// Construction data for a [`PermissionEngine`].
///
/// Python parity: the `PermissionEngine.__init__` arguments `config`,
/// `workspace_root` and `trusted_dirs`; the `llm`/`model_name` handles of the
/// Python class are not used by the decision path and are not ported.
#[derive(Debug, Clone, PartialEq)]
pub struct PermissionEngineConfig {
    /// The `permissions:` section of the agent configuration.
    pub permissions: PermissionsSection,
    /// Workspace root relative file paths are resolved against.
    pub workspace_root: Option<PathBuf>,
    /// Project identity anchor (repository root) every path must stay inside.
    ///
    /// Python parity: the `project_root` layer of the CWD state, which the filesystem
    /// guards treat as an allowed root. `None` leaves the workspace root as the only
    /// boundary.
    pub project_root: Option<PathBuf>,
    /// Runtime trusted directories; a non-empty list enables the file guard.
    pub trusted_dirs: Vec<PathBuf>,
    /// Whether the pipeline runs at all; a disabled engine allows everything.
    pub enabled: bool,
}

impl Default for PermissionEngineConfig {
    fn default() -> Self {
        Self {
            permissions: PermissionsSection::default(),
            workspace_root: None,
            project_root: None,
            trusted_dirs: Vec::new(),
            enabled: true,
        }
    }
}

impl PermissionEngineConfig {
    /// Builds a config from a `permissions:` section; the engine is enabled.
    #[must_use]
    pub fn new(permissions: PermissionsSection) -> Self {
        Self {
            permissions,
            enabled: true,
            ..Self::default()
        }
    }

    /// Sets the workspace root relative paths are resolved against.
    #[must_use]
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// Sets the project root every path must stay inside.
    #[must_use]
    pub fn with_project_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.project_root = Some(root.into());
        self
    }
    /// Sets the runtime trusted directories.
    #[must_use]
    pub fn with_trusted_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.trusted_dirs = dirs;
        self
    }

    /// Enables or disables the whole pipeline.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Strictest-wins fold over permission levels: `deny < ask < allow`.
///
/// Ports `tool_policy.strictest`: a `deny` always wins, an `ask` wins over an
/// `allow`, and the fold is commutative and idempotent.
#[must_use]
pub fn strictest(a: PermissionLevel, b: PermissionLevel) -> PermissionLevel {
    match (a, b) {
        (PermissionLevel::Deny, _) | (_, PermissionLevel::Deny) => PermissionLevel::Deny,
        (PermissionLevel::Ask, _) | (_, PermissionLevel::Ask) => PermissionLevel::Ask,
        _ => PermissionLevel::Allow,
    }
}

/// [`strictest`] over any number of levels; an empty iterator yields
/// [`PermissionLevel::Ask`] (fail closed).
#[must_use]
pub fn strictest_all(levels: impl IntoIterator<Item = PermissionLevel>) -> PermissionLevel {
    levels
        .into_iter()
        .fold(None, |acc: Option<PermissionLevel>, level| {
            Some(acc.map_or(level, |current| strictest(current, level)))
        })
        .unwrap_or(PermissionLevel::Ask)
}

/// Inlines the packaged builtin floors and fills legacy `severity` actions.
///
/// Ports `prepare_permissions_for_engine`. The returned section is what the
/// tiered policy and both guards evaluate:
///
/// - when [`PermissionsSection::package_builtin_rules_enabled`], the packaged
///   command rules are prepended (unless the section already carries a
///   `layer: "builtin"` rule), the packaged sensitive paths are appended to an
///   enabled `file_guard` (same marker rule) and the packaged metadata URLs are
///   folded into an enabled `net_guard.urls` with [`strictest`], so a host
///   `allow` can never widen a packaged `deny`;
/// - every rule without an `action` gets
///   [`rules::severity_to_decision`] applied under the normalized
///   `permission_mode`.
///
/// The step is idempotent: `prepare(prepare(x)) == prepare(x)`.
#[must_use]
pub fn prepare_permissions_for_engine(section: &PermissionsSection) -> PermissionsSection {
    prepare_owned(section.clone())
}

/// [`prepare_permissions_for_engine`] consuming its input, to avoid a clone on
/// the engine's own ingest path.
fn prepare_owned(mut prepared: PermissionsSection) -> PermissionsSection {
    let mode = rules::normalize_permission_mode(prepared.permission_mode.as_deref());

    if prepared.package_builtin_rules_enabled() {
        if !has_builtin_layer(&prepared.rules, |rule| rule.layer.as_deref()) {
            let mut rules = builtin::package_command_rules();
            rules.append(&mut prepared.rules);
            prepared.rules = rules;
        }

        if prepared.file_guard.enabled
            && !has_builtin_layer(&prepared.file_guard.paths, |entry| entry.layer.as_deref())
        {
            prepared
                .file_guard
                .paths
                .extend(builtin::package_sensitive_paths());
        }

        if prepared.net_guard.enabled {
            for (pattern, level) in builtin::package_net_urls() {
                prepared
                    .net_guard
                    .urls
                    .entry(pattern)
                    .and_modify(|current| *current = strictest(*current, level))
                    .or_insert(level);
            }
        }
    }

    // Legacy host rules carry only a `severity`; the packaged ones already have an
    // `action`, so this is a no-op for them.
    for rule in &mut prepared.rules {
        if rule.action.is_none() {
            rule.action = rules::severity_to_decision(rule.severity.as_deref(), mode);
        }
    }

    prepared
}

/// Whether any entry of `entries` is marked `layer: "builtin"`.
fn has_builtin_layer<T, F>(entries: &[T], layer: F) -> bool
where
    F: Fn(&T) -> Option<&str>,
{
    entries
        .iter()
        .any(|entry| layer(entry) == Some(BUILTIN_LAYER))
}

/// The compiled policy behind the engine.
#[derive(Debug)]
struct CompiledPolicy {
    /// The prepared `permissions:` section pipelines A consumes.
    permissions: PermissionsSection,
    /// Pipeline B, absent when the path guard is disabled.
    file_guard: Option<FileGuardChecker>,
    /// Pipeline C, absent when the network guard is disabled.
    net_guard: Option<NetGuardChecker>,
}

/// Permission façade: owns the compiled policy and folds pipelines A/B/C.
///
/// Ports `PermissionEngine`. Construct once and share as
/// `Arc<PermissionEngine>`; [`PermissionEngine::update_permissions`] is the
/// hot-update seam used by the session permission overlay.
///
/// ```
/// use openjiuwen_harness::permission::engine::{PermissionEngine, PermissionEngineConfig};
/// use openjiuwen_harness::permission::model::{PermissionLevel, PermissionsSection};
/// use serde_json::json;
///
/// let engine = PermissionEngine::new(PermissionEngineConfig::new(PermissionsSection::default()));
/// let result = engine.check_permission("unknown_tool", &json!({}));
/// assert_eq!(result.permission, PermissionLevel::Ask);
/// ```
#[derive(Debug)]
pub struct PermissionEngine {
    /// Prepared section plus both compiled guards.
    policy: RwLock<CompiledPolicy>,
    /// Workspace root both the engine and the file guard resolve against.
    workspace_root: Option<PathBuf>,
    /// Optional project root every touched path must stay inside.
    project_root: Option<PathBuf>,
    /// Runtime trusted directories compiled into the file guard.
    trusted_dirs: Vec<PathBuf>,
    /// Whether the whole pipeline is enabled (the `enabled` config switch).
    enabled: bool,
}

impl PermissionEngine {
    /// Builds an engine from `config`: prepares the section and compiles both
    /// guards.
    #[must_use]
    pub fn new(config: PermissionEngineConfig) -> Self {
        let PermissionEngineConfig {
            permissions,
            workspace_root,
            project_root,
            trusted_dirs,
            enabled,
        } = config;
        let prepared = prepare_owned(permissions);
        let file_guard =
            FileGuardChecker::build(&prepared, workspace_root.as_deref(), &trusted_dirs);
        let net_guard = NetGuardChecker::build(&prepared);
        Self {
            policy: RwLock::new(CompiledPolicy {
                permissions: prepared,
                file_guard,
                net_guard,
            }),
            workspace_root,
            project_root,
            trusted_dirs,
            enabled,
        }
    }

    /// A snapshot clone of the prepared `permissions:` section.
    #[must_use]
    pub fn config(&self) -> PermissionsSection {
        self.policy.read().permissions.clone()
    }

    /// Whether the pipeline runs; a disabled engine allows every tool call.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The workspace root relative paths are resolved against.
    #[must_use]
    pub fn workspace_root(&self) -> Option<PathBuf> {
        self.workspace_root.clone()
    }

    /// The project root every touched path must stay inside, when configured.
    #[must_use]
    pub fn project_root(&self) -> Option<PathBuf> {
        self.project_root.clone()
    }
    /// The runtime trusted directories compiled into the file guard.
    #[must_use]
    pub fn trusted_dirs(&self) -> Vec<PathBuf> {
        self.trusted_dirs.clone()
    }

    /// Replaces the policy: re-prepares the section and recompiles both guards.
    ///
    /// The workspace root, the trusted directories and the enabled switch are
    /// kept. Ports `update_config`; the session overlay calls this after merging
    /// an "always allow" decision into the section.
    pub fn update_permissions(&self, permissions: PermissionsSection) {
        let prepared = prepare_owned(permissions);
        let file_guard = FileGuardChecker::build(
            &prepared,
            self.workspace_root.as_deref(),
            &self.trusted_dirs,
        );
        let net_guard = NetGuardChecker::build(&prepared);
        *self.policy.write() = CompiledPolicy {
            permissions: prepared,
            file_guard,
            net_guard,
        };
    }

    /// Evaluates one tool call through pipelines A, B and C.
    ///
    /// Synchronous and non-blocking: no IO, no awaits, and the compiled policy
    /// lock is released before returning. A disabled engine allows the call with
    /// the Python reason `"Permission system is disabled"`.
    ///
    /// The `matched_rule` label joins the contributing rules with `|` in
    /// pipeline order (`tiered_policy:…`, `file_guard…`, `net_guard…`); the
    /// reason follows the Python templates and the ask template names the
    /// normalized tool.
    #[must_use]
    pub fn check_permission(&self, tool_name: &str, args: &Value) -> PermissionResult {
        if !self.enabled {
            return PermissionResult::new(PermissionLevel::Allow)
                .with_reason("Permission system is disabled");
        }

        let tool = rules::normalize_tool_name(tool_name);
        let empty = Value::Object(serde_json::Map::new());
        let args = if args.is_object() { args } else { &empty };

        let policy = self.policy.read();

        // 1. Pipeline A: tiered tool/parameter policy; "no opinion" starts at ask.
        let (mut permission, mut matched_rule) =
            tiered::evaluate_tiered_policy(&policy.permissions, &tool, args);
        if matched_rule == tiered::FALLBACK_NO_CONFIG {
            permission = PermissionLevel::Ask;
            DEFAULT_LABEL.clone_into(&mut matched_rule);
        }

        // 2. Pipeline B: path guard.
        let mut external_paths = None;
        if let Some(result) = policy
            .file_guard
            .as_ref()
            .and_then(|guard| guard.evaluate(&tool, args))
        {
            let path_rule = result
                .matched_rule
                .unwrap_or_else(|| "file_guard".to_owned());
            permission = strictest(permission, result.permission);
            matched_rule = format!("{matched_rule}|{path_rule}");
            external_paths = result.external_paths;
        }

        // 2b. Project-root boundary: a path outside the configured root is denied.
        if let Some(paths) = self.project_root_violations(&tool, args) {
            permission = PermissionLevel::Deny;
            matched_rule = format!("{matched_rule}|project_root");
            external_paths = Some(paths);
        }
        // 3. Pipeline C: network guard (deny only).
        if let Some(result) = policy
            .net_guard
            .as_ref()
            .and_then(|guard| guard.evaluate(&tool, args))
        {
            let net_rule = result
                .matched_rule
                .unwrap_or_else(|| "net_guard".to_owned());
            permission = strictest(permission, result.permission);
            matched_rule = format!("{matched_rule}|{net_rule}");
        }

        drop(policy);

        let reason = match permission {
            PermissionLevel::Allow => format!("Allowed by rule: {matched_rule}"),
            PermissionLevel::Deny => format!("Denied by rule: {matched_rule}"),
            PermissionLevel::Ask => {
                format!("Approval required for {tool} (rule: {matched_rule})")
            }
        };

        let result = PermissionResult::new(permission)
            .with_matched_rule(matched_rule)
            .with_reason(reason);
        match external_paths {
            Some(paths) => result.with_external_paths(paths),
            None => result,
        }
    }

    /// Paths of this call that fall outside the configured project root.
    ///
    /// Python parity: `project_root` is an access boundary for the filesystem guards, so
    /// a `file_path` pointing outside it is denied even when the workspace itself would
    /// allow it. The comparison is lexical (the engine performs no IO); the tools resolve
    /// and canonicalize the same path before touching it. `None` without a project root or
    /// while every touched path stays inside it.
    fn project_root_violations(&self, tool: &str, args: &Value) -> Option<Vec<String>> {
        let root = paths::normalize_lexically(self.project_root.as_deref()?);
        let violations: Vec<String> = paths::extract_file_accesses(tool, args, &root)
            .into_iter()
            .filter(|access| access.path.is_absolute())
            .filter(|access| {
                let path = paths::normalize_lexically(&access.path);
                !(path == root || path.starts_with(&root))
            })
            .map(|access| access.path.to_string_lossy().into_owned())
            .collect();
        (!violations.is_empty()).then_some(violations)
    }

    /// Evaluates pipeline A only, ignoring the enabled switch and both guards.
    ///
    /// Ports `evaluate_global_policy_directly` (as used for owner-scope lookups):
    /// `(None, None)` means the section had no opinion at all
    /// ([`tiered::FALLBACK_NO_CONFIG`]). Non-object `args` are treated as `{}`.
    #[must_use]
    pub fn evaluate_global_policy_directly(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> (Option<PermissionLevel>, Option<String>) {
        let tool = rules::normalize_tool_name(tool_name);
        let empty = Value::Object(serde_json::Map::new());
        let args = if args.is_object() { args } else { &empty };

        let policy = self.policy.read();
        let (level, label) = tiered::evaluate_tiered_policy(&policy.permissions, &tool, args);
        if label == tiered::FALLBACK_NO_CONFIG {
            (None, None)
        } else {
            (Some(level), Some(label))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::PathBuf;

    use serde_json::json;

    use super::*;
    use crate::permission::model::{
        FileGuardDefaults, FileGuardSection, NetGuardSection, PermissionRule, RuleMatchType,
    };

    /// A section with no `defaults`, disabling the packaged floors: pipeline A has
    /// no opinion for any tool.
    fn no_config_section() -> PermissionsSection {
        PermissionsSection {
            package_builtin_rules: false,
            defaults: std::collections::BTreeMap::new(),
            ..PermissionsSection::default()
        }
    }

    /// A section carrying the given tool baselines and `defaults["*"]`.
    fn tool_section(
        tools: &[(&str, PermissionLevel)],
        default: PermissionLevel,
    ) -> PermissionsSection {
        PermissionsSection {
            package_builtin_rules: false,
            defaults: std::collections::BTreeMap::from([("*".to_owned(), default)]),
            tools: tools
                .iter()
                .map(|(tool, level)| ((*tool).to_owned(), *level))
                .collect(),
            ..PermissionsSection::default()
        }
    }

    /// An engine over `section` with no workspace root.
    fn engine(section: PermissionsSection) -> PermissionEngine {
        PermissionEngine::new(PermissionEngineConfig::new(section))
    }

    /// An enabled `file_guard` with these default axes and no path entries.
    fn file_guard(defaults: FileGuardDefaults) -> FileGuardSection {
        FileGuardSection {
            enabled: true,
            defaults,
            workspace: None,
            paths: Vec::new(),
        }
    }

    /// A file guard that denies every axis.
    fn deny_file_guard() -> FileGuardSection {
        file_guard(FileGuardDefaults {
            read: PermissionLevel::Deny,
            write: PermissionLevel::Deny,
            exec: PermissionLevel::Deny,
        })
    }

    /// A host (non-packaged) command rule.
    fn host_rule(
        id: &str,
        pattern: &str,
        action: Option<PermissionLevel>,
        severity: Option<&str>,
    ) -> PermissionRule {
        PermissionRule {
            id: id.to_owned(),
            tools: vec!["shell".to_owned()],
            pattern: pattern.to_owned(),
            match_type: RuleMatchType::Command,
            action,
            severity: severity.map(str::to_owned),
            description: None,
            layer: None,
        }
    }

    // ----- strictest ------------------------------------------------------

    #[test]
    fn strictest_orders_deny_below_ask_below_allow() {
        use PermissionLevel::{Allow, Ask, Deny};

        assert_eq!(strictest(Allow, Allow), Allow);
        assert_eq!(strictest(Allow, Ask), Ask);
        assert_eq!(strictest(Ask, Allow), Ask);
        assert_eq!(strictest(Ask, Ask), Ask);
        assert_eq!(strictest(Allow, Deny), Deny);
        assert_eq!(strictest(Deny, Allow), Deny);
        assert_eq!(strictest(Ask, Deny), Deny);
        assert_eq!(strictest(Deny, Deny), Deny);
    }

    #[test]
    fn strictest_all_folds_lists_and_fails_closed_when_empty() {
        use PermissionLevel::{Allow, Ask, Deny};

        assert_eq!(strictest_all([]), Ask, "an empty fold is ask");
        assert_eq!(strictest_all([Allow]), Allow);
        assert_eq!(strictest_all([Allow, Ask]), Ask);
        assert_eq!(strictest_all([Allow, Ask, Deny]), Deny);
        assert_eq!(strictest_all([Deny, Allow, Ask]), Deny);
        assert_eq!(strictest_all([Allow, Allow, Allow]), Allow);
    }

    // ----- enabled / defaults ---------------------------------------------

    #[test]
    fn disabled_engine_allows_everything() {
        let engine = PermissionEngine::new(
            PermissionEngineConfig::new(tool_section(
                &[("bash", PermissionLevel::Deny)],
                PermissionLevel::Deny,
            ))
            .with_enabled(false),
        );
        assert!(!engine.enabled());
        let result = engine.check_permission("bash", &json!({"command": "rm -rf /"}));
        assert_eq!(result.permission, PermissionLevel::Allow);
        assert_eq!(
            result.reason.as_deref(),
            Some("Permission system is disabled")
        );
        assert_eq!(result.matched_rule, None);
    }

    #[test]
    fn empty_section_yields_the_default_ask() {
        let engine = engine(no_config_section());
        let result = engine.check_permission("foo", &json!({}));
        assert_eq!(result.permission, PermissionLevel::Ask);
        assert_eq!(result.matched_rule.as_deref(), Some("default"));
        assert_eq!(
            result.reason.as_deref(),
            Some("Approval required for foo (rule: default)")
        );
        assert_eq!(result.external_paths, None);
    }

    #[test]
    fn enabled_defaults_to_true_and_builders_compose() {
        let config = PermissionEngineConfig::default();
        assert!(config.enabled);
        assert_eq!(config.workspace_root, None);
        assert_eq!(config.trusted_dirs, Vec::<PathBuf>::new());
        assert!(config.permissions.enabled);

        let config = PermissionEngineConfig::new(no_config_section())
            .with_workspace_root("/ws")
            .with_trusted_dirs(vec![PathBuf::from("/data")])
            .with_enabled(false);
        assert_eq!(config.workspace_root, Some(PathBuf::from("/ws")));
        assert_eq!(config.trusted_dirs, vec![PathBuf::from("/data")]);
        assert!(!config.enabled);

        let engine = PermissionEngine::new(config);
        assert_eq!(engine.workspace_root(), Some(PathBuf::from("/ws")));
        assert_eq!(engine.trusted_dirs(), vec![PathBuf::from("/data")]);
        assert!(!engine.enabled());
    }

    // ----- reason templates ----------------------------------------------

    #[test]
    fn tool_allow_baseline_allows_with_the_python_reason() {
        let engine = engine(tool_section(
            &[("bash", PermissionLevel::Allow)],
            PermissionLevel::Allow,
        ));
        let result = engine.check_permission("bash", &json!({}));
        assert_eq!(result.permission, PermissionLevel::Allow);
        assert_eq!(result.matched_rule.as_deref(), Some("tools.bash"));
        assert_eq!(
            result.reason.as_deref(),
            Some("Allowed by rule: tools.bash")
        );
    }

    #[test]
    fn tool_deny_baseline_denies_with_the_python_reason() {
        let engine = engine(tool_section(
            &[("bash", PermissionLevel::Deny)],
            PermissionLevel::Allow,
        ));
        let result = engine.check_permission("bash", &json!({"command": "ls"}));
        assert_eq!(result.permission, PermissionLevel::Deny);
        assert_eq!(result.matched_rule.as_deref(), Some("tools.bash"));
        assert_eq!(result.reason.as_deref(), Some("Denied by rule: tools.bash"));
    }

    #[test]
    fn non_object_args_are_treated_as_empty() {
        let engine = engine(tool_section(
            &[("bash", PermissionLevel::Allow)],
            PermissionLevel::Ask,
        ));
        let result = engine.check_permission("bash", &json!("ls"));
        assert_eq!(result.permission, PermissionLevel::Allow);
        assert_eq!(result.matched_rule.as_deref(), Some("tools.bash"));
    }

    // ----- pipeline folds -------------------------------------------------

    #[test]
    fn file_guard_fold_denies_and_joins_the_label() {
        let section = PermissionsSection {
            package_builtin_rules: false,
            tools: std::collections::BTreeMap::from([(
                "read_file".to_owned(),
                PermissionLevel::Allow,
            )]),
            file_guard: deny_file_guard(),
            ..PermissionsSection::default()
        };
        let engine =
            PermissionEngine::new(PermissionEngineConfig::new(section).with_workspace_root("/ws"));

        let result = engine.check_permission("read_file", &json!({"file_path": "/etc/passwd"}));
        assert_eq!(result.permission, PermissionLevel::Deny);
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("tools.read_file|file_guard:defaults")
        );
        assert_eq!(
            result.reason.as_deref(),
            Some("Denied by rule: tools.read_file|file_guard:defaults")
        );
        assert_eq!(result.external_paths, Some(vec!["/etc/passwd".to_owned()]));
    }

    #[test]
    fn net_guard_fold_denies_packaged_metadata_urls() {
        let section = PermissionsSection {
            package_builtin_rules: true,
            tools: std::collections::BTreeMap::from([(
                "mcp_fetch_webpage".to_owned(),
                PermissionLevel::Allow,
            )]),
            net_guard: NetGuardSection {
                enabled: true,
                defaults: PermissionLevel::Allow,
                urls: std::collections::BTreeMap::new(),
            },
            ..PermissionsSection::default()
        };
        let engine = engine(section);
        let result = engine.check_permission(
            "mcp_fetch_webpage",
            &json!({"url": "http://169.254.169.254/latest/meta-data/"}),
        );
        assert_eq!(result.permission, PermissionLevel::Deny);
        let matched = result.matched_rule.as_deref().unwrap();
        assert!(matched.contains("net_guard"), "{matched}");
        assert!(matched.starts_with("tools.mcp_fetch_webpage|"), "{matched}");
        assert!(result
            .reason
            .as_deref()
            .unwrap()
            .starts_with("Denied by rule: "));
    }

    #[test]
    fn ask_from_pipeline_a_survives_a_net_guard_allow() {
        let section = PermissionsSection {
            package_builtin_rules: false,
            tools: std::collections::BTreeMap::from([(
                "mcp_fetch_webpage".to_owned(),
                PermissionLevel::Ask,
            )]),
            net_guard: NetGuardSection {
                enabled: true,
                defaults: PermissionLevel::Allow,
                urls: std::collections::BTreeMap::from([(
                    "https://example.com/**".to_owned(),
                    PermissionLevel::Allow,
                )]),
            },
            ..PermissionsSection::default()
        };
        let engine = engine(section);
        let result = engine.check_permission(
            "mcp_fetch_webpage",
            &json!({"url": "https://example.com/page"}),
        );
        assert_eq!(result.permission, PermissionLevel::Ask);
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("tools.mcp_fetch_webpage")
        );
    }

    // ----- evaluate_global_policy_directly --------------------------------

    #[test]
    fn direct_policy_returns_the_tiered_decision() {
        let engine = engine(tool_section(
            &[("bash", PermissionLevel::Deny)],
            PermissionLevel::Allow,
        ));
        assert_eq!(
            engine.evaluate_global_policy_directly("bash", &json!({})),
            (Some(PermissionLevel::Deny), Some("tools.bash".to_owned()))
        );
    }

    #[test]
    fn direct_policy_reports_no_opinion() {
        let engine = engine(no_config_section());
        assert_eq!(
            engine.evaluate_global_policy_directly("foo", &json!({})),
            (None, None)
        );
    }

    #[test]
    fn direct_policy_ignores_the_enabled_switch_and_the_guards() {
        let section = PermissionsSection {
            package_builtin_rules: false,
            tools: std::collections::BTreeMap::from([(
                "read_file".to_owned(),
                PermissionLevel::Allow,
            )]),
            file_guard: deny_file_guard(),
            ..PermissionsSection::default()
        };
        let engine = PermissionEngine::new(
            PermissionEngineConfig::new(section)
                .with_workspace_root("/ws")
                .with_enabled(false),
        );
        assert_eq!(
            engine.evaluate_global_policy_directly("read_file", &json!({"file_path": "/x"})),
            (
                Some(PermissionLevel::Allow),
                Some("tools.read_file".to_owned())
            )
        );
    }

    // ----- prepare_permissions_for_engine ---------------------------------

    /// A section with an enabled file and net guard and no packaged entries yet.
    fn packaging_section() -> PermissionsSection {
        PermissionsSection {
            package_builtin_rules: true,
            file_guard: FileGuardSection {
                enabled: true,
                ..FileGuardSection::default()
            },
            net_guard: NetGuardSection {
                enabled: true,
                ..NetGuardSection::default()
            },
            ..PermissionsSection::default()
        }
    }

    #[test]
    fn prepare_injects_the_packaged_floors() {
        let prepared = prepare_permissions_for_engine(&packaging_section());
        assert_eq!(prepared.rules.len(), 17);
        assert!(prepared
            .rules
            .iter()
            .all(|rule| rule.layer.as_deref() == Some("builtin")));
        assert_eq!(prepared.file_guard.paths.len(), 18);
        assert!(prepared
            .file_guard
            .paths
            .iter()
            .all(|entry| entry.layer.as_deref() == Some("builtin")));
        assert_eq!(prepared.net_guard.urls.len(), 4);
        assert!(prepared
            .net_guard
            .urls
            .values()
            .all(|level| *level == PermissionLevel::Deny));
    }

    #[test]
    fn prepare_keeps_host_rules_after_the_package_rules() {
        let mut section = packaging_section();
        section.rules.push(host_rule(
            "host-rule",
            "hostcmd*",
            Some(PermissionLevel::Ask),
            None,
        ));
        let prepared = prepare_permissions_for_engine(&section);
        assert_eq!(prepared.rules.len(), 18);
        assert_eq!(prepared.rules[17].id, "host-rule");
        assert!(prepared
            .rules
            .iter()
            .take(17)
            .all(|rule| rule.layer.as_deref() == Some("builtin")));
    }

    #[test]
    fn prepare_is_idempotent() {
        let mut section = packaging_section();
        section
            .rules
            .push(host_rule("host-rule", "hostcmd*", None, Some("HIGH")));
        let once = prepare_permissions_for_engine(&section);
        let twice = prepare_permissions_for_engine(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn prepare_maps_severity_under_the_permission_mode() {
        let mut section = PermissionsSection {
            package_builtin_rules: false,
            permission_mode: Some("strict".to_owned()),
            ..PermissionsSection::default()
        };
        section
            .rules
            .push(host_rule("medium", "medium*", None, Some("MEDIUM")));

        let strict = prepare_permissions_for_engine(&section);
        assert_eq!(strict.rules[0].action, Some(PermissionLevel::Ask));

        section.permission_mode = Some("normal".to_owned());
        let normal = prepare_permissions_for_engine(&section);
        assert_eq!(normal.rules[0].action, Some(PermissionLevel::Allow));

        section.permission_mode = None;
        let default_mode = prepare_permissions_for_engine(&section);
        assert_eq!(default_mode.rules[0].action, Some(PermissionLevel::Allow));
    }

    #[test]
    fn prepare_honours_an_explicit_action_over_severity() {
        let mut section = PermissionsSection {
            package_builtin_rules: false,
            permission_mode: Some("strict".to_owned()),
            ..PermissionsSection::default()
        };
        section.rules.push(host_rule(
            "explicit",
            "explicit*",
            Some(PermissionLevel::Allow),
            Some("CRITICAL"),
        ));
        let prepared = prepare_permissions_for_engine(&section);
        assert_eq!(prepared.rules[0].action, Some(PermissionLevel::Allow));
    }

    #[test]
    fn prepare_does_nothing_extra_when_packaging_is_off() {
        let mut section = packaging_section();
        section.package_builtin_rules = false;
        section
            .rules
            .push(host_rule("host-rule", "hostcmd*", None, Some("LOW")));
        let prepared = prepare_permissions_for_engine(&section);
        assert_eq!(prepared.rules.len(), 1);
        assert_eq!(prepared.rules[0].action, Some(PermissionLevel::Allow));
        assert_eq!(prepared.file_guard.paths, Vec::new());
        assert!(prepared.net_guard.urls.is_empty());
    }

    #[test]
    fn prepare_never_widens_a_packaged_net_deny() {
        let mut section = packaging_section();
        section
            .net_guard
            .urls
            .insert("169.254.169.254".to_owned(), PermissionLevel::Allow);
        section
            .net_guard
            .urls
            .insert("blocked.example".to_owned(), PermissionLevel::Deny);
        let prepared = prepare_permissions_for_engine(&section);
        assert_eq!(
            prepared.net_guard.urls.get("169.254.169.254"),
            Some(&PermissionLevel::Deny)
        );
        assert_eq!(
            prepared.net_guard.urls.get("blocked.example"),
            Some(&PermissionLevel::Deny)
        );
        assert_eq!(prepared.net_guard.urls.len(), 5);
    }

    // ----- update_permissions ---------------------------------------------

    #[test]
    fn update_permissions_swaps_the_policy_and_recompiles_the_guards() {
        let engine = engine(tool_section(
            &[("bash", PermissionLevel::Deny)],
            PermissionLevel::Ask,
        ));
        assert_eq!(
            engine.check_permission("bash", &json!({})).permission,
            PermissionLevel::Deny
        );

        engine.update_permissions(tool_section(
            &[("bash", PermissionLevel::Allow)],
            PermissionLevel::Ask,
        ));
        let result = engine.check_permission("bash", &json!({}));
        assert_eq!(result.permission, PermissionLevel::Allow);
        assert_eq!(result.matched_rule.as_deref(), Some("tools.bash"));
        assert_eq!(
            engine.config().tools.get("bash"),
            Some(&PermissionLevel::Allow)
        );

        // The file guard is rebuilt from the new section: its deny defaults now fold in.
        let engine = PermissionEngine::new(
            PermissionEngineConfig::new(tool_section(
                &[("read_file", PermissionLevel::Allow)],
                PermissionLevel::Ask,
            ))
            .with_workspace_root("/ws"),
        );
        assert_eq!(
            engine
                .check_permission("read_file", &json!({"file_path": "/etc/passwd"}))
                .permission,
            PermissionLevel::Allow,
            "the file guard starts disabled"
        );
        let mut section = tool_section(
            &[("read_file", PermissionLevel::Allow)],
            PermissionLevel::Ask,
        );
        section.file_guard = deny_file_guard();
        engine.update_permissions(section);
        let result = engine.check_permission("read_file", &json!({"file_path": "/etc/passwd"}));
        assert_eq!(result.permission, PermissionLevel::Deny);
        assert!(result
            .matched_rule
            .as_deref()
            .unwrap()
            .contains("file_guard"));

        // The net guard is rebuilt as well: the packaged metadata floors are inlined.
        let engine = PermissionEngine::new(
            PermissionEngineConfig::new(tool_section(
                &[("mcp_fetch_webpage", PermissionLevel::Allow)],
                PermissionLevel::Allow,
            ))
            .with_workspace_root("/ws"),
        );
        assert_eq!(
            engine
                .check_permission(
                    "mcp_fetch_webpage",
                    &json!({"url": "http://169.254.169.254/"})
                )
                .permission,
            PermissionLevel::Allow
        );
        let mut section = tool_section(
            &[("mcp_fetch_webpage", PermissionLevel::Allow)],
            PermissionLevel::Allow,
        );
        section.net_guard = NetGuardSection {
            enabled: true,
            defaults: PermissionLevel::Allow,
            urls: std::collections::BTreeMap::new(),
        };
        section.package_builtin_rules = true;
        engine.update_permissions(section);
        let result = engine.check_permission(
            "mcp_fetch_webpage",
            &json!({"url": "http://169.254.169.254/"}),
        );
        assert_eq!(result.permission, PermissionLevel::Deny);
        assert!(result
            .matched_rule
            .as_deref()
            .unwrap()
            .contains("net_guard"));
    }
}
