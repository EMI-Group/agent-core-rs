//! File-path guard (pipeline B): per-path read/write/exec policy.
//!
//! Ports `openjiuwen/harness/security/permission_engine/fileguard/file_guard.py`
//! (Python v0.1.20) — the `FileGuardChecker` and the native compilation of
//! `permissions.file_guard`. The guard sits beside the tool policy (pipeline A):
//! it never lifts an upstream decision, it only contributes an `ask`/`deny`
//! opinion of its own.
//!
//! The checker compiles its rules once at construction:
//!
//! - every `file_guard.paths` entry, as a `prefix` or `**`-aware `glob` rule;
//! - one `allow/allow/ask` prefix rule per runtime trusted directory;
//! - one prefix rule for the workspace root carrying the `file_guard.workspace`
//!   axes, when that section is present.
//!
//! Prefix paths are expanded (`~`, env-less), joined onto the workspace root when
//! relative, folded lexically with [`paths::normalize_lexically`] and rendered as
//! POSIX strings; entries that reduce to a whole drive (`"C:"`), a blank string or
//! `"*"` are dropped, mirroring Python's unsafe-rule filter. Glob patterns keep
//! their text (only a leading `~` is expanded), because the matcher owns the
//! `**` semantics. Write/exec `allow` implies read `allow`, except that an explicit
//! read `deny` wins (Python `_apply_implications`).
//!
//! No filesystem IO, no locking and no panics: decisions depend only on the
//! compiled rules and on the text of the tool call.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::model::{
    FileAction, FileGuardDefaults, FileGuardPathEntry, PathAccess, PathMatch, PermissionLevel,
    PermissionResult, PermissionsSection,
};
use super::{paths, patterns, rules};

/// How a compiled [`CompiledRule`] compares against an access path.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RuleMatcher {
    /// Everything at, or below, a normalized POSIX prefix.
    Prefix(String),
    /// A `**`-aware POSIX glob.
    Glob(String),
}

impl RuleMatcher {
    /// The path text of the rule; used in `matched_rule` labels.
    fn pattern(&self) -> &str {
        match self {
            Self::Prefix(pattern) | Self::Glob(pattern) => pattern,
        }
    }
}

/// One compiled path rule with its read/write/exec axes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledRule {
    /// Prefix or glob the rule covers.
    matcher: RuleMatcher,
    /// Decision for read accesses below the pattern.
    read: PermissionLevel,
    /// Decision for write accesses below the pattern.
    write: PermissionLevel,
    /// Decision for execute accesses below the pattern.
    exec: PermissionLevel,
}

impl CompiledRule {
    /// The level this rule assigns to `action`.
    fn level(&self, action: FileAction) -> PermissionLevel {
        match action {
            FileAction::Read => self.read,
            FileAction::Write => self.write,
            FileAction::Exec => self.exec,
        }
    }
}

/// Path guard (pipeline B) of the permission engine.
///
/// Construct with [`FileGuardChecker::build`]; the layer is disabled by returning
/// `None` from the constructor, so a live checker always evaluates.
///
/// Python parity: `file_guard.FileGuardChecker` (native mode only — the legacy
/// `external_directory` projection is not ported).
#[derive(Debug, Clone)]
pub struct FileGuardChecker {
    /// Compiled rules in configuration order.
    rules: Vec<CompiledRule>,
    /// Decisions for paths no rule covers.
    defaults: FileGuardDefaults,
    /// Workspace root every relative path is resolved against.
    workspace_root: Option<PathBuf>,
    /// Resolved shell-tool set of the permissions section.
    shell_tools: BTreeSet<String>,
}

impl FileGuardChecker {
    /// Compiles a checker, or returns `None` when the path layer is disabled.
    ///
    /// The layer is enabled when `section.file_guard.enabled` is set or when
    /// runtime `trusted_dirs` are supplied — a trusted directory alone turns the
    /// guard on, exactly like Python's `normalize_path_guard_config`.
    #[must_use]
    pub fn build(
        section: &PermissionsSection,
        workspace_root: Option<&Path>,
        trusted_dirs: &[PathBuf],
    ) -> Option<Self> {
        let file_guard = &section.file_guard;
        if !file_guard.enabled && trusted_dirs.is_empty() {
            return None;
        }

        let root = workspace_root.map(paths::normalize_lexically);
        let mut compiled = Vec::new();

        for entry in &file_guard.paths {
            if let Some(rule) = compile_path_entry(entry, root.as_deref()) {
                compiled.push(rule);
            }
        }

        for dir in trusted_dirs {
            let (read, write, exec) = apply_implications(
                PermissionLevel::Allow,
                PermissionLevel::Allow,
                PermissionLevel::Ask,
            );
            if let Some(rule) =
                compile_prefix(&dir.to_string_lossy(), read, write, exec, root.as_deref())
            {
                compiled.push(rule);
            }
        }

        if let (Some(axes), Some(root)) = (&file_guard.workspace, root.as_deref()) {
            let (read, write, exec) = apply_implications(axes.read, axes.write, axes.exec);
            if let Some(rule) =
                compile_prefix(&root.to_string_lossy(), read, write, exec, Some(root))
            {
                compiled.push(rule);
            }
        }

        let defaults = file_guard.defaults.clone();
        let (read, write, exec) = apply_implications(defaults.read, defaults.write, defaults.exec);

        Some(Self {
            rules: compiled,
            defaults: FileGuardDefaults { read, write, exec },
            workspace_root: root,
            shell_tools: rules::shell_tools_from_config(section),
        })
    }

    /// The workspace root the checker resolves relative paths against.
    #[must_use]
    pub fn workspace_root(&self) -> Option<&Path> {
        self.workspace_root.as_deref()
    }

    /// Evaluates the path layer for one tool call.
    ///
    /// Returns `None` when the layer has no opinion: no workspace root, no path
    /// access at all, or an overall [`PermissionLevel::Allow`]. A `None` result
    /// never lifts an upstream decision; the engine only folds an `ask`/`deny`.
    #[must_use]
    pub fn evaluate(&self, tool_name: &str, args: &Value) -> Option<PermissionResult> {
        let accesses = self.accesses(tool_name, args);
        if accesses.is_empty() {
            return None;
        }

        let mut overall = PermissionLevel::Allow;
        let mut external_paths: Vec<String> = Vec::new();
        let mut matched: Vec<String> = Vec::new();

        for access in &accesses {
            let path_posix = render_posix(&access.path);
            let (level, contributors) = self.resolve_one(&path_posix, access.action);
            overall = strictest(overall, level);
            if level == PermissionLevel::Allow {
                continue;
            }
            if !external_paths.contains(&path_posix) {
                external_paths.push(path_posix);
            }
            for contributor in contributors {
                if !matched.contains(&contributor) {
                    matched.push(contributor);
                }
            }
        }

        if overall == PermissionLevel::Allow {
            return None;
        }

        let matched_rule = if matched.is_empty() {
            "file_guard".to_owned()
        } else {
            matched.join("|")
        };
        let reason = if overall == PermissionLevel::Deny {
            format!("file_guard denied: {matched_rule}")
        } else {
            format!("file_guard requires approval: {matched_rule}")
        };

        Some(PermissionResult {
            permission: overall,
            matched_rule: Some(matched_rule),
            reason: Some(reason),
            external_paths: if external_paths.is_empty() {
                None
            } else {
                Some(external_paths)
            },
        })
    }

    /// Accesses of this call whose computed level is [`PermissionLevel::Ask`].
    ///
    /// Used by the "always allow" merge, which pins the actually touched paths per
    /// axis. Empty when the layer is disabled or no workspace root is known.
    #[must_use]
    pub fn collect_ask_accesses(&self, tool_name: &str, args: &Value) -> Vec<PathAccess> {
        if self.workspace_root.is_none() {
            return Vec::new();
        }

        let mut seen: HashSet<(String, FileAction)> = HashSet::new();
        let mut asks = Vec::new();
        for access in self.accesses(tool_name, args) {
            let path_posix = render_posix(&access.path);
            let (level, _) = self.resolve_one(&path_posix, access.action);
            if level != PermissionLevel::Ask {
                continue;
            }
            if seen.insert((path_posix, access.action)) {
                asks.push(access);
            }
        }
        asks
    }

    /// Path accesses of a tool call, deduplicated in first-seen order.
    fn accesses(&self, tool_name: &str, args: &Value) -> Vec<PathAccess> {
        let Some(root) = self.workspace_root.as_deref() else {
            return Vec::new();
        };
        let normalized = rules::normalize_tool_name(tool_name);
        let raw = if rules::is_shell_tool(&normalized, &self.shell_tools) {
            // The raw command text (never canonicalized) keeps every path of a
            // compound command line visible to the extractor.
            let command = rules::command_text(args);
            let workdir = args
                .get("workdir")
                .or_else(|| args.get("cwd"))
                .and_then(Value::as_str);
            paths::extract_shell_accesses(&command, workdir, root)
        } else {
            paths::extract_file_accesses(&normalized, args, root)
        };
        dedupe_accesses(raw)
    }

    /// Level and contributing rule labels of one access.
    fn resolve_one(&self, path_posix: &str, action: FileAction) -> (PermissionLevel, Vec<String>) {
        let mut best_prefix: Option<(usize, &CompiledRule)> = None;
        let mut glob_levels: Vec<PermissionLevel> = Vec::new();

        for rule in &self.rules {
            match &rule.matcher {
                RuleMatcher::Glob(pattern) => {
                    if patterns::match_glob(pattern, path_posix) {
                        glob_levels.push(rule.level(action));
                    }
                }
                RuleMatcher::Prefix(prefix) => {
                    if path_posix == prefix || path_posix.starts_with(&format!("{prefix}/")) {
                        let length = prefix.len();
                        if best_prefix.is_none_or(|(best, _)| length > best) {
                            best_prefix = Some((length, rule));
                        }
                    }
                }
            }
        }

        let mut candidates: Vec<PermissionLevel> = Vec::new();
        if let Some((_, rule)) = best_prefix {
            candidates.push(rule.level(action));
        }
        candidates.extend(glob_levels.iter().copied());
        if candidates.is_empty() {
            return (
                defaults_axis(&self.defaults, action),
                vec!["file_guard:defaults".to_owned()],
            );
        }

        let level = candidates
            .iter()
            .copied()
            .fold(PermissionLevel::Allow, strictest);

        let mut contributors = Vec::new();
        if let Some((_, rule)) = best_prefix {
            if rule.level(action) == level {
                contributors.push(format!("file_guard:prefix:{}", rule.matcher.pattern()));
            }
        }
        if glob_levels.contains(&level) {
            contributors.push("file_guard:glob".to_owned());
        }
        (level, contributors)
    }
}

/// Resolves the default axis decision of an access action.
fn defaults_axis(defaults: &FileGuardDefaults, action: FileAction) -> PermissionLevel {
    match action {
        FileAction::Read => defaults.read,
        FileAction::Write => defaults.write,
        FileAction::Exec => defaults.exec,
    }
}

/// `deny < ask < allow` fold of two levels.
fn strictest(current: PermissionLevel, candidate: PermissionLevel) -> PermissionLevel {
    match (current, candidate) {
        (PermissionLevel::Deny, _) | (_, PermissionLevel::Deny) => PermissionLevel::Deny,
        (PermissionLevel::Ask, _) | (_, PermissionLevel::Ask) => PermissionLevel::Ask,
        _ => PermissionLevel::Allow,
    }
}

/// Compiles one configured path entry, applying the read implication first.
fn compile_path_entry(entry: &FileGuardPathEntry, root: Option<&Path>) -> Option<CompiledRule> {
    let (read, write, exec) = apply_implications(entry.read, entry.write, entry.exec);
    match entry.r#match {
        PathMatch::Glob => {
            let pattern = expand_tilde(&entry.path);
            if pattern.is_empty() {
                return None;
            }
            Some(CompiledRule {
                matcher: RuleMatcher::Glob(pattern),
                read,
                write,
                exec,
            })
        }
        PathMatch::Prefix => compile_prefix(&entry.path, read, write, exec, root),
    }
}

/// Compiles a prefix rule, dropping unsafe whole-drive/relative patterns.
fn compile_prefix(
    raw: &str,
    read: PermissionLevel,
    write: PermissionLevel,
    exec: PermissionLevel,
    root: Option<&Path>,
) -> Option<CompiledRule> {
    let expanded = expand_tilde(raw);
    if expanded.is_empty() {
        return None;
    }
    let path = PathBuf::from(&expanded);
    let joined = if path.is_absolute() || has_drive_prefix(&expanded) {
        path
    } else {
        match root {
            Some(root) => root.join(path),
            None => path,
        }
    };
    let pattern = render_posix(&paths::normalize_lexically(&joined));
    if pattern.is_empty() || !pattern.contains('/') || pattern == "*" {
        return None;
    }
    Some(CompiledRule {
        matcher: RuleMatcher::Prefix(pattern),
        read,
        write,
        exec,
    })
}

/// Write/exec `allow` implies read `allow`, unless read is an explicit `deny`.
fn apply_implications(
    read: PermissionLevel,
    write: PermissionLevel,
    exec: PermissionLevel,
) -> (PermissionLevel, PermissionLevel, PermissionLevel) {
    let opens = write == PermissionLevel::Allow || exec == PermissionLevel::Allow;
    if opens && read != PermissionLevel::Deny {
        (PermissionLevel::Allow, write, exec)
    } else {
        (read, write, exec)
    }
}

/// Deduplicates accesses by `(path, action)`, preserving first-seen order.
fn dedupe_accesses(accesses: Vec<PathAccess>) -> Vec<PathAccess> {
    let mut seen: HashSet<(PathBuf, FileAction)> = HashSet::new();
    let mut unique = Vec::with_capacity(accesses.len());
    for access in accesses {
        if seen.insert((access.path.clone(), access.action)) {
            unique.push(access);
        }
    }
    unique
}

/// Renders a path as a POSIX string (`/`-joined).
fn render_posix(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Expands a leading `~` to the home directory (`HOME`, then `USERPROFILE`).
///
/// The token is left literal when neither variable is set, matching
/// `os.path.expanduser` on a machine without a home directory.
fn expand_tilde(raw: &str) -> String {
    let trimmed = raw.trim();
    let rest = trimmed
        .strip_prefix('~')
        .filter(|rest| rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\'));
    let Some(rest) = rest else {
        return trimmed.to_owned();
    };
    let Some(home) = home_dir() else {
        return trimmed.to_owned();
    };
    format!("{}{rest}", home.trim_end_matches(['/', '\\']))
}

/// The user's home directory, if the environment names one.
fn home_dir() -> Option<String> {
    for key in ["HOME", "USERPROFILE"] {
        if let Ok(value) = std::env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

/// Whether `text` starts with a `Windows` drive prefix such as `C:\` or `C:/`.
fn has_drive_prefix(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(
        (chars.next(), chars.next(), chars.next()),
        (Some(letter), Some(':'), Some('/' | '\\')) if letter.is_ascii_alphabetic()
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn enabled_section() -> PermissionsSection {
        let mut section = PermissionsSection::default();
        section.file_guard.enabled = true;
        section
    }

    fn ask_defaults() -> FileGuardDefaults {
        FileGuardDefaults {
            read: PermissionLevel::Ask,
            write: PermissionLevel::Ask,
            exec: PermissionLevel::Ask,
        }
    }

    fn deny_defaults() -> FileGuardDefaults {
        FileGuardDefaults {
            read: PermissionLevel::Deny,
            write: PermissionLevel::Deny,
            exec: PermissionLevel::Deny,
        }
    }

    fn entry(
        path: &str,
        read: PermissionLevel,
        write: PermissionLevel,
        exec: PermissionLevel,
        matcher: PathMatch,
    ) -> FileGuardPathEntry {
        FileGuardPathEntry {
            path: path.to_owned(),
            read,
            write,
            exec,
            r#match: matcher,
            ..FileGuardPathEntry::default()
        }
    }

    fn path_arg(dir: &Path, name: &str) -> String {
        dir.join(name).to_string_lossy().into_owned()
    }

    // ----- construction ---------------------------------------------------

    #[test]
    fn disabled_section_yields_no_checker() {
        let section = PermissionsSection::default();
        assert!(FileGuardChecker::build(&section, Some(Path::new("/ws")), &[]).is_none());
        // A runtime trusted directory alone enables the layer.
        assert!(FileGuardChecker::build(
            &section,
            Some(Path::new("/ws")),
            &[PathBuf::from("/trusted")]
        )
        .is_some());
    }

    #[test]
    fn enabled_without_workspace_root_has_no_opinion() {
        let checker = FileGuardChecker::build(&enabled_section(), None, &[]).unwrap();
        assert_eq!(checker.workspace_root(), None);
        assert!(checker
            .evaluate("read_file", &json!({"path": "/tmp/a.py"}))
            .is_none());
        assert_eq!(
            checker.collect_ask_accesses("read_file", &json!({"path": "/tmp/a.py"})),
            Vec::new()
        );
    }

    // ----- defaults -------------------------------------------------------

    #[test]
    fn unmatched_path_falls_back_to_defaults() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let result = checker
            .evaluate("read_file", &json!({"path": "a.py"}))
            .unwrap();
        assert_eq!(result.permission, PermissionLevel::Ask);
        assert_eq!(result.matched_rule.as_deref(), Some("file_guard:defaults"));
        assert_eq!(
            result.reason.as_deref(),
            Some("file_guard requires approval: file_guard:defaults")
        );
        assert_eq!(
            result.external_paths,
            Some(vec![render_posix(&ws.path().join("a.py"))])
        );
    }

    #[test]
    fn default_allow_produces_no_opinion() {
        let ws = TempDir::new().unwrap();
        let section = enabled_section();
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();
        assert!(checker
            .evaluate("read_file", &json!({"path": "a.py"}))
            .is_none());
    }

    // ----- prefix rules ---------------------------------------------------

    #[test]
    fn prefix_rule_allows_reads_and_denies_writes() {
        let ws = TempDir::new().unwrap();
        let data = ws.path().join("data");
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        section.file_guard.paths = vec![entry(
            &data.to_string_lossy(),
            PermissionLevel::Allow,
            PermissionLevel::Deny,
            PermissionLevel::Deny,
            PathMatch::Prefix,
        )];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        assert!(checker
            .evaluate("read_file", &json!({"path": path_arg(&data, "a.py")}))
            .is_none());

        let denied = checker
            .evaluate("write_file", &json!({"path": path_arg(&data, "a.py")}))
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert_eq!(
            denied.matched_rule.as_deref(),
            Some(format!("file_guard:prefix:{}", render_posix(&data)).as_str())
        );
        assert_eq!(
            denied.reason.as_deref(),
            Some(
                format!(
                    "file_guard denied: file_guard:prefix:{}",
                    render_posix(&data)
                )
                .as_str()
            )
        );
    }

    #[test]
    fn longest_prefix_wins() {
        let ws = TempDir::new().unwrap();
        let root = ws.path().join("root");
        let nested = root.join("nested");
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        section.file_guard.paths = vec![
            entry(
                &root.to_string_lossy(),
                PermissionLevel::Allow,
                PermissionLevel::Allow,
                PermissionLevel::Ask,
                PathMatch::Prefix,
            ),
            entry(
                &nested.to_string_lossy(),
                PermissionLevel::Deny,
                PermissionLevel::Deny,
                PermissionLevel::Ask,
                PathMatch::Prefix,
            ),
        ];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        assert!(checker
            .evaluate("read_file", &json!({"path": path_arg(&root, "a.py")}))
            .is_none());

        let denied = checker
            .evaluate("read_file", &json!({"path": path_arg(&nested, "a.py")}))
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert_eq!(
            denied.matched_rule.as_deref(),
            Some(format!("file_guard:prefix:{}", render_posix(&nested)).as_str())
        );
    }

    #[test]
    fn unsafe_prefixes_are_dropped() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        section.file_guard.paths = vec![
            entry(
                "",
                PermissionLevel::Allow,
                PermissionLevel::Allow,
                PermissionLevel::Ask,
                PathMatch::Prefix,
            ),
            entry(
                "*",
                PermissionLevel::Allow,
                PermissionLevel::Allow,
                PermissionLevel::Ask,
                PathMatch::Prefix,
            ),
        ];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();
        let result = checker
            .evaluate("read_file", &json!({"path": "a.py"}))
            .unwrap();
        // The unsafe entries never match; the fallback is the defaults.
        assert_eq!(result.matched_rule.as_deref(), Some("file_guard:defaults"));
    }

    // ----- read implication ----------------------------------------------

    #[test]
    fn write_allow_implies_read_allow() {
        let ws = TempDir::new().unwrap();
        let data = ws.path().join("data");
        let mut section = enabled_section();
        section.file_guard.defaults = deny_defaults();
        section.file_guard.paths = vec![entry(
            &data.to_string_lossy(),
            PermissionLevel::Ask,
            PermissionLevel::Allow,
            PermissionLevel::Ask,
            PathMatch::Prefix,
        )];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        assert!(checker
            .evaluate("read_file", &json!({"path": path_arg(&data, "a.py")}))
            .is_none());
        assert!(checker
            .evaluate("write_file", &json!({"path": path_arg(&data, "a.py")}))
            .is_none());
    }

    #[test]
    fn explicit_read_deny_survives_write_allow() {
        let ws = TempDir::new().unwrap();
        let data = ws.path().join("data");
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        section.file_guard.paths = vec![entry(
            &data.to_string_lossy(),
            PermissionLevel::Deny,
            PermissionLevel::Allow,
            PermissionLevel::Ask,
            PathMatch::Prefix,
        )];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let denied = checker
            .evaluate("read_file", &json!({"path": path_arg(&data, "a.py")}))
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert!(checker
            .evaluate("write_file", &json!({"path": path_arg(&data, "a.py")}))
            .is_none());
    }

    // ----- glob rules -----------------------------------------------------

    #[test]
    fn glob_rule_denies_dot_env_files() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.paths = vec![entry(
            "**/.env",
            PermissionLevel::Deny,
            PermissionLevel::Deny,
            PermissionLevel::Ask,
            PathMatch::Glob,
        )];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let denied = checker
            .evaluate("read_file", &json!({"path": ".env"}))
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert_eq!(denied.matched_rule.as_deref(), Some("file_guard:glob"));

        assert!(checker
            .evaluate("read_file", &json!({"path": "a.py"}))
            .is_none());
    }

    #[test]
    fn prefix_and_glob_contributors_are_joined() {
        let ws = TempDir::new().unwrap();
        let src = ws.path().join("src");
        let mut section = enabled_section();
        section.file_guard.paths = vec![
            entry(
                &src.to_string_lossy(),
                PermissionLevel::Deny,
                PermissionLevel::Deny,
                PermissionLevel::Ask,
                PathMatch::Prefix,
            ),
            entry(
                "**/.env",
                PermissionLevel::Deny,
                PermissionLevel::Deny,
                PermissionLevel::Ask,
                PathMatch::Glob,
            ),
        ];
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let denied = checker
            .evaluate("read_file", &json!({"path": path_arg(&src, ".env")}))
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert_eq!(
            denied.matched_rule.as_deref(),
            Some(format!("file_guard:prefix:{}|file_guard:glob", render_posix(&src)).as_str())
        );
    }

    // ----- shell tools ----------------------------------------------------

    #[test]
    fn shell_commands_use_the_raw_command_text() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = FileGuardDefaults {
            read: PermissionLevel::Allow,
            write: PermissionLevel::Deny,
            exec: PermissionLevel::Ask,
        };
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let denied = checker
            .evaluate(
                "bash",
                &json!({"command": format!("rm -rf {}", path_arg(ws.path(), "x"))}),
            )
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert_eq!(denied.matched_rule.as_deref(), Some("file_guard:defaults"));

        assert!(checker
            .evaluate(
                "bash",
                &json!({"command": format!("cat {}", path_arg(ws.path(), "a.txt"))}),
            )
            .is_none());
    }

    #[test]
    fn compound_shell_commands_keep_every_path() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = FileGuardDefaults {
            read: PermissionLevel::Allow,
            write: PermissionLevel::Deny,
            exec: PermissionLevel::Ask,
        };
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let denied = checker
            .evaluate(
                "bash",
                &json!({
                    "command": format!(
                        "cat {}; rm {}",
                        path_arg(ws.path(), "a.txt"),
                        path_arg(ws.path(), "b.txt")
                    )
                }),
            )
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
        assert_eq!(
            denied.external_paths,
            Some(vec![render_posix(&ws.path().join("b.txt"))])
        );
    }

    #[test]
    fn shell_workdir_is_the_base_for_relative_paths() {
        let ws = TempDir::new().unwrap();
        let sub = ws.path().join("sub");
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let result = checker
            .evaluate("bash", &json!({"command": "cat x.txt", "workdir": sub}))
            .unwrap();
        assert_eq!(
            result.external_paths,
            Some(vec![render_posix(&sub.join("x.txt"))])
        );
    }

    #[test]
    fn shell_tool_aliases_are_normalized() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        section
            .categories
            .insert("shell".to_owned(), vec!["mcp_exec_command".to_owned()]);
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        // `exec_command` aliases to `mcp_exec_command`, so it is a shell tool.
        let result = checker
            .evaluate("exec_command", &json!({"command": "cat a.txt"}))
            .unwrap();
        assert_eq!(
            result.external_paths,
            Some(vec![render_posix(&ws.path().join("a.txt"))])
        );
    }

    // ----- trusted dirs and workspace axes --------------------------------

    #[test]
    fn trusted_dirs_allow_external_paths() {
        let ws = TempDir::new().unwrap();
        let trusted = TempDir::new().unwrap();
        let mut section = PermissionsSection::default();
        // `file_guard.enabled` stays false: the trusted directory enables the layer.
        section.file_guard.defaults = deny_defaults();
        let checker =
            FileGuardChecker::build(&section, Some(ws.path()), &[trusted.path().to_path_buf()])
                .unwrap();

        assert!(checker
            .evaluate(
                "read_file",
                &json!({"path": path_arg(trusted.path(), "x.py")})
            )
            .is_none());

        let denied = checker
            .evaluate("read_file", &json!({"path": path_arg(ws.path(), "a.py")}))
            .unwrap();
        assert_eq!(denied.permission, PermissionLevel::Deny);
    }

    #[test]
    fn workspace_axes_bind_to_the_workspace_root() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = deny_defaults();
        section.file_guard.workspace = Some(FileGuardDefaults {
            read: PermissionLevel::Allow,
            write: PermissionLevel::Ask,
            exec: PermissionLevel::Deny,
        });
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        assert!(checker
            .evaluate("read_file", &json!({"path": "a.py"}))
            .is_none());

        let ask = checker
            .evaluate("write_file", &json!({"path": "a.py"}))
            .unwrap();
        assert_eq!(ask.permission, PermissionLevel::Ask);
        assert_eq!(
            ask.matched_rule.as_deref(),
            Some(format!("file_guard:prefix:{}", render_posix(ws.path())).as_str())
        );
    }

    // ----- collect_ask_accesses ------------------------------------------

    #[test]
    fn collect_ask_accesses_returns_only_ask_paths() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        let asks = checker.collect_ask_accesses("read_file", &json!({"path": "a.py"}));
        assert_eq!(asks.len(), 1);
        assert_eq!(asks[0].action, FileAction::Read);
        assert_eq!(asks[0].path, ws.path().join("a.py"));

        let mut allow_section = enabled_section();
        allow_section.file_guard.defaults = FileGuardDefaults {
            read: PermissionLevel::Allow,
            write: PermissionLevel::Allow,
            exec: PermissionLevel::Allow,
        };
        let allow_checker = FileGuardChecker::build(&allow_section, Some(ws.path()), &[]).unwrap();
        assert_eq!(
            allow_checker.collect_ask_accesses("read_file", &json!({"path": "a.py"})),
            Vec::new()
        );
    }

    #[test]
    fn tools_without_path_accesses_have_no_opinion() {
        let ws = TempDir::new().unwrap();
        let mut section = enabled_section();
        section.file_guard.defaults = ask_defaults();
        let checker = FileGuardChecker::build(&section, Some(ws.path()), &[]).unwrap();

        assert!(checker.evaluate("todo_list", &json!({})).is_none());
        assert!(checker.evaluate("read_file", &json!({})).is_none());
        assert_eq!(
            checker.collect_ask_accesses("read_file", &json!({})),
            Vec::new()
        );
    }

    // ----- path rendering helpers ----------------------------------------

    #[test]
    fn render_posix_and_tilde_expansion() {
        assert_eq!(render_posix(Path::new("/ws/a")), "/ws/a");
        assert_eq!(expand_tilde("  /ws/a "), "/ws/a");
        assert_eq!(expand_tilde("~someone/x"), "~someone/x");
    }
}
