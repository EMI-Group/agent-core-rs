//! Host hooks, session-scoped decisions and allow-rule persistence merges.
//!
//! Ports three Python pieces of the permission engine (v0.1.20):
//!
//! - `permission_engine/host.py` (`ToolPermissionHost`): the callbacks a product
//!   layer injects — permission snapshot, confirmation, persistence, workspace
//!   resolution and the pre-emption [scene hook](SceneDecision);
//! - the session-scoped state of `rails/security/tool_security_rail.py`:
//!   [`SessionPermissionStore`] holds the auto-confirmed tool calls of the running
//!   session (Python `__interrupt_auto_confirm__`) and
//!   [`should_store_auto_confirm`] is the "remember this answer" truth table;
//! - `permission_engine/approve/persist_rule_merge.py`: the copy-on-write merges
//!   that turn an "always allow" answer into configuration —
//!   [`merge_allow_rule_into_permissions`] for the tiered policy and
//!   [`merge_file_guard_access_allows`] for the path guard.
//!
//! This module performs no IO: persisting the merged section is the host's job
//! ([`PermissionHost::persist_allow_rule`] /
//! [`PermissionHost::persist_session_allow_rule`]), and the YAML write-back belongs
//! to the CLI/config layer. Nothing here can widen a decision on its own: both
//! merges first re-evaluate the current decision and only remember an **ask**.

use std::collections::BTreeMap;
use std::path::PathBuf;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value;

use crate::permission::model::{
    ApprovalOverrideEntry, FileAction, FileGuardPathEntry, PathAccess, PathMatch,
    PermissionConfirmResponse, PermissionLevel, PermissionResult, PermissionsSection,
    RuleMatchType,
};
use crate::permission::shell::{self, ShellKind};
use crate::permission::{rules, tiered};

/// Longest approval-override id kept when remembering a command.
///
/// Python `_build_approval_override_id` truncates the sanitized id to 120 characters.
const MAX_APPROVAL_OVERRIDE_ID_LEN: usize = 120;

/// Fallback approval-override id when sanitizing leaves nothing behind.
///
/// Python `_build_approval_override_id` returns the same literal.
const FALLBACK_APPROVAL_OVERRIDE_ID: &str = "user_allow_override";

/// Input of the optional scene hook a host may install to pre-empt the engine.
///
/// Python parity: `PermissionSceneHookInput`. The hook runs before the tiered
/// policy, so a product layer (a digital twin, an owner-scope check, …) can decide
/// the call without consulting the rules. The Python context/tool-call objects are
/// flattened here into the data the hook actually needs.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneHookInput {
    /// Model-facing tool name, not yet normalized through the tool aliases.
    pub tool_name: String,
    /// Identifier of the tool call within the round.
    pub tool_call_id: String,
    /// JSON arguments of the tool call.
    pub arguments: Value,
    /// `true` when the answer was delivered in an earlier resume round of this
    /// still-paused call and was already acted on, so it must not be persisted
    /// again.
    pub answer_carried: bool,
}

/// What a scene hook wants done instead of running the tiered policy.
///
/// Python parity: the `("approve",)` / `("reject", msg)` tuples returned by
/// `PermissionSceneHook`. `None` from the hook means "no opinion, continue".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneDecision {
    /// Let the call run without asking.
    Approve,
    /// Reject the call, optionally with feedback for the model.
    Reject(Option<String>),
}

/// What a host wants the rail to do instead of asking the user.
///
/// Python parity: `PermissionConfirmationRequest`. The rail hands one of these to
/// [`PermissionHost::request_confirmation`] when the pipeline yields an ask.
#[derive(Debug, Clone, PartialEq)]
pub struct PermissionConfirmationRequest {
    /// Model-facing tool name.
    pub tool_name: String,
    /// Identifier of the tool call within the round.
    pub tool_call_id: String,
    /// JSON arguments of the tool call.
    pub arguments: Value,
    /// The decision record that produced the ask.
    pub result: PermissionResult,
    /// Session key under which an approval may be remembered; empty when the call
    /// has no safe session key (risky or compound shell commands).
    pub auto_confirm_key: String,
}

/// Answer of [`PermissionHost::request_confirmation`].
///
/// Python parity: `PermissionConfirmResponse | Literal["interrupt"] | None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfirmationOutcome {
    /// No host answer: fall back to the built-in confirm interrupt.
    Interrupt,
    /// The host answered on the user's behalf.
    Response(PermissionConfirmResponse),
    /// The host failed to obtain an answer; the call is rejected.
    Failed,
    /// The host answered with a payload that could not be interpreted; the call is
    /// rejected.
    Invalid,
}

/// Callbacks a product layer injects into the permission rail.
///
/// Python parity: the `ToolPermissionHost` dataclass. Python stores optional
/// callables; the Rust host is a trait whose default methods reproduce the "hook
/// not installed" behaviour, so [`NoopPermissionHost`] is an empty `impl`.
///
/// The sync hooks are pure reads and never block the async runtime; only
/// confirmation, the scene hook and persistence are `async`, mirroring the
/// `await` points of the Python rail.
#[async_trait]
pub trait PermissionHost: Send + Sync {
    /// The current `permissions:` section, used to re-sync disk configuration.
    fn permissions_snapshot(&self) -> Option<PermissionsSection> {
        None
    }

    /// Workspace root used to validate paths outside the workspace.
    fn resolve_workspace_dir(&self) -> Option<PathBuf> {
        None
    }

    /// Identifier of the running session, if any.
    fn session_id(&self) -> Option<String> {
        None
    }

    /// Agent configuration file that a permanent allow rule is written to.
    fn permission_yaml_path(&self) -> Option<PathBuf> {
        None
    }

    /// Host scene hook: `Some` pre-empts the tiered policy, `None` continues.
    async fn scene_hook(&self, _input: &SceneHookInput) -> Option<SceneDecision> {
        None
    }

    /// Asks the host for a user answer; the default falls back to the built-in
    /// interrupt.
    async fn request_confirmation(
        &self,
        _request: &PermissionConfirmationRequest,
    ) -> ConfirmationOutcome {
        ConfirmationOutcome::Interrupt
    }

    /// Persists a permanent allow rule (the whole merged section); `false` when the
    /// host could not write it.
    async fn persist_allow_rule(&self, _permissions: &PermissionsSection) -> bool {
        false
    }

    /// Persists a session-scoped allow rule; `false` when the host could not write
    /// it.
    async fn persist_session_allow_rule(&self, _permissions: &PermissionsSection) -> bool {
        false
    }
}

/// Host with every hook disabled.
///
/// Python parity: a `ToolPermissionHost()` with all callables `None`: asks go to
/// the built-in interrupt, nothing is persisted, and no configuration is reloaded.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopPermissionHost;

impl PermissionHost for NoopPermissionHost {}

/// In-memory session decisions of the running agent session.
///
/// Python parity: the `__interrupt_auto_confirm__` bucket of the session state. A
/// stored key means "this tool call was already approved for the session", so the
/// next identical call is allowed without asking.
///
/// Keys are opaque strings — [`crate::permission::shell::shell_auto_confirm_key`]
/// builds the conservative shell form (`"bash:ls"`), an empty key is never stored.
#[derive(Debug, Default)]
pub struct SessionPermissionStore {
    /// Key → confirmed flag; Python stores `True` for every remembered call.
    auto_confirm: RwLock<BTreeMap<String, bool>>,
}

impl Clone for SessionPermissionStore {
    /// Clones the current contents; the copy is independent of the original.
    fn clone(&self) -> Self {
        Self {
            auto_confirm: RwLock::new(self.auto_confirm.read().clone()),
        }
    }
}

impl SessionPermissionStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `key` was already auto-confirmed; `false` for an empty key.
    #[must_use]
    pub fn is_auto_confirmed(&self, key: &str) -> bool {
        if key.is_empty() {
            return false;
        }
        self.auto_confirm.read().get(key).copied().unwrap_or(false)
    }

    /// Remembers `key`; an empty key is ignored (Python: no session key, no
    /// memory).
    pub fn store_auto_confirm(&self, key: &str) {
        if key.is_empty() {
            return;
        }
        self.auto_confirm.write().insert(key.to_owned(), true);
    }

    /// Drops every remembered decision.
    pub fn clear(&self) {
        self.auto_confirm.write().clear();
    }

    /// Number of remembered decisions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.auto_confirm.read().len()
    }

    /// Whether no decision is remembered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.auto_confirm.read().is_empty()
    }
}

/// Result of an allow-rule merge.
///
/// The input section is never mutated: the merged copy lives in `permissions`, and
/// the two flags say which of the two targets changed.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeOutcome {
    /// Merged copy of the input section.
    pub permissions: PermissionsSection,
    /// Whether a tiered-policy allow was added (`tools.<tool>` or an
    /// `approval_overrides` entry).
    pub rule_merged: bool,
    /// Whether a `file_guard.paths` entry was added or widened.
    pub path_merged: bool,
}

impl MergeOutcome {
    /// Whether anything was merged at all.
    #[must_use]
    pub fn applied(&self) -> bool {
        self.rule_merged || self.path_merged
    }
}

/// Copy of `section` with nothing merged.
fn unchanged(section: &PermissionsSection) -> MergeOutcome {
    MergeOutcome {
        permissions: section.clone(),
        rule_merged: false,
        path_merged: false,
    }
}

/// Remembers an "always allow" answer in the tiered policy.
///
/// Python parity: `merge_permission_allow_rule_into_permissions`, restricted to
/// the two safe targets of `persist_rule_suggestions`:
///
/// - a shell tool with one simple, non-risky command remembers that exact command
///   as an `approval_overrides` entry — never the whole tool, so `bash` stays
///   gated for everything else;
/// - any other tool is widened to `tools.<tool> = allow`.
///
/// The merge only happens when the current decision is
/// [`PermissionLevel::Ask`]: an allow needs no rule, and a
/// deny must never be remembered away. The result is therefore idempotent — a
/// second merge on the merged section reports `rule_merged == false`.
#[must_use]
pub fn merge_allow_rule_into_permissions(
    section: &PermissionsSection,
    tool_name: &str,
    args: &Value,
) -> MergeOutcome {
    let tool = rules::normalize_tool_name(tool_name);
    if tool.is_empty() {
        return unchanged(section);
    }
    let shell_tools = rules::shell_tools_from_config(section);
    if tiered::evaluate_tiered_policy(section, &tool, args).0 != PermissionLevel::Ask {
        return unchanged(section);
    }

    let mut permissions = section.clone();
    if rules::is_shell_tool(&tool, &shell_tools) {
        let pattern = shell::canonicalize_shell_command(&rules::command_text(args));
        let ast = shell::analyze_shell(&pattern);
        let safe_command = !pattern.is_empty()
            && ast.kind == ShellKind::Simple
            && ast.subcommands.len() == 1
            && !ast.flags.has_risky_structure();
        if !safe_command {
            // Risky, compound or unparseable commands are never remembered, and a
            // shell tool is never widened to a whole-tool allow.
            return unchanged(section);
        }
        permissions.approval_overrides.push(ApprovalOverrideEntry {
            id: approval_override_id(&tool, RuleMatchType::Command, &pattern),
            tools: vec![tool],
            match_type: RuleMatchType::Command,
            pattern,
            action: PermissionLevel::Allow,
        });
        return MergeOutcome {
            permissions,
            rule_merged: true,
            path_merged: false,
        };
    }

    permissions.tools.insert(tool, PermissionLevel::Allow);
    MergeOutcome {
        permissions,
        rule_merged: true,
        path_merged: false,
    }
}

/// Remembers "always allow" answers for paths in the file guard.
///
/// Python parity: `merge_file_guard_access_allows` + `merge_file_guard_path_rule`.
/// Every access is turned into per-axis decisions by its action, so remembering a
/// read never opens a write:
///
/// - [`FileAction::Read`] → `read: allow`, `write: ask`, `exec: ask`;
/// - [`FileAction::Write`] → `read: allow`, `write: allow`, `exec: ask`;
/// - [`FileAction::Exec`] → `read: allow`, `write: ask`, `exec: allow`.
///
/// Accesses are folded in first-seen order: repeated paths merge into a single
/// [`PathMatch::Prefix`] entry (Python merges the axes the same way), and a `glob`
/// entry for the same path is left untouched. `allow` always wins over a stricter
/// axis, while an incoming `ask` never downgrades an existing decision.
///
/// The file guard is enabled by the merge — a remembered path rule is useless while
/// the guard is off.
#[must_use]
pub fn merge_file_guard_access_allows(
    section: &PermissionsSection,
    accesses: &[PathAccess],
) -> MergeOutcome {
    let mut permissions = section.clone();
    let mut path_merged = false;
    let mut seen_any = false;

    for access in accesses {
        let path = normalize_path_key(&access.path.to_string_lossy());
        if path.is_empty() {
            continue;
        }
        seen_any = true;
        let (read, write, exec) = axes_for_file_guard_action(access.action);
        let existing = permissions.file_guard.paths.iter_mut().find(|entry| {
            entry.r#match == PathMatch::Prefix && normalize_path_key(&entry.path) == path
        });
        if let Some(entry) = existing {
            let merged_read = escalate_axis_toward_allow(entry.read, read);
            let merged_write = escalate_axis_toward_allow(entry.write, write);
            let merged_exec = escalate_axis_toward_allow(entry.exec, exec);
            if (merged_read, merged_write, merged_exec) != (entry.read, entry.write, entry.exec) {
                entry.read = merged_read;
                entry.write = merged_write;
                entry.exec = merged_exec;
                path_merged = true;
            }
        } else {
            permissions.file_guard.paths.push(FileGuardPathEntry {
                id: None,
                path,
                read,
                write,
                exec,
                r#match: PathMatch::Prefix,
                layer: None,
            });
            path_merged = true;
        }
    }

    if seen_any {
        permissions.file_guard.enabled = true;
    }
    MergeOutcome {
        permissions,
        rule_merged: false,
        path_merged,
    }
}

/// Whether an answer should be remembered in the session store.
///
/// Python parity: `PermissionInterruptRail._should_store_auto_confirm`. The answer
/// must be an approval that asked to be remembered, the session must exist and the
/// call must carry a session key.
///
/// A **permanent** rule that was actually persisted is skipped: it is reloaded from
/// the host snapshot on the next check. A session-scoped persist still stores the
/// key, because the next check reloads the disk snapshot and would otherwise drop
/// the in-memory session rule.
#[must_use]
// The six Python keyword arguments are part of the ported contract; grouping them
// into a struct would change the call shape the rail is written against.
#[allow(clippy::fn_params_excessive_bools)]
pub fn should_store_auto_confirm(
    approved: bool,
    auto_confirm: bool,
    has_session: bool,
    auto_confirm_key: &str,
    persisted: bool,
    permanent: bool,
) -> bool {
    if !(approved && auto_confirm && has_session && !auto_confirm_key.is_empty()) {
        return false;
    }
    !(permanent && persisted)
}

/// Builds the approval-override id of a remembered command.
///
/// Python parity: `_build_approval_override_id` — `user_allow_<tool>_<match>_<pattern>`
/// lowercased, every non-`[a-z0-9_]` run replaced by a single `_`, leading and
/// trailing separators trimmed, and the result cut to
/// [`MAX_APPROVAL_OVERRIDE_ID_LEN`] characters.
fn approval_override_id(tool: &str, match_type: RuleMatchType, pattern: &str) -> String {
    let raw = format!("user_allow_{tool}_{}_{pattern}", match_type.as_str());
    let mut sanitized = String::with_capacity(raw.len().min(MAX_APPROVAL_OVERRIDE_ID_LEN));
    let mut last_was_separator = false;
    for character in raw.chars().flat_map(char::to_lowercase) {
        if character.is_ascii_alphanumeric() {
            sanitized.push(character);
            last_was_separator = false;
        } else if !last_was_separator {
            sanitized.push('_');
            last_was_separator = true;
        }
    }
    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        return FALLBACK_APPROVAL_OVERRIDE_ID.to_owned();
    }
    trimmed.chars().take(MAX_APPROVAL_OVERRIDE_ID_LEN).collect()
}

/// Per-axis decisions a remembered access writes.
///
/// Python parity: `_axes_for_file_guard_action`; the fallback keeps an unknown
/// action at the read-only shape.
fn axes_for_file_guard_action(
    action: FileAction,
) -> (PermissionLevel, PermissionLevel, PermissionLevel) {
    match action {
        FileAction::Read => (
            PermissionLevel::Allow,
            PermissionLevel::Ask,
            PermissionLevel::Ask,
        ),
        FileAction::Write => (
            PermissionLevel::Allow,
            PermissionLevel::Allow,
            PermissionLevel::Ask,
        ),
        FileAction::Exec => (
            PermissionLevel::Allow,
            PermissionLevel::Ask,
            PermissionLevel::Allow,
        ),
    }
}

/// Merges one axis of a remembered path rule.
///
/// Python parity: `_escalate_axis_toward_allow` — an incoming `allow` wins, and
/// anything else keeps the existing decision so a remembered read cannot narrow an
/// explicit deny either.
fn escalate_axis_toward_allow(
    existing: PermissionLevel,
    incoming: PermissionLevel,
) -> PermissionLevel {
    if incoming == PermissionLevel::Allow {
        PermissionLevel::Allow
    } else {
        existing
    }
}

/// Renders a path the way the file guard compares it: forward slashes, no trailing
/// separator.
fn normalize_path_key(path: &str) -> String {
    path.replace('\\', "/").trim_end_matches('/').to_owned()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::Path;

    use serde_json::json;

    use super::*;
    use crate::permission::builtin;

    // ----- session store ---------------------------------------------------

    #[test]
    fn session_store_remembers_and_forgets_keys() {
        let store = SessionPermissionStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.store_auto_confirm("bash:ls");
        store.store_auto_confirm("read_file:/ws/a.txt");
        assert!(store.is_auto_confirmed("bash:ls"));
        assert!(store.is_auto_confirmed("read_file:/ws/a.txt"));
        assert!(!store.is_auto_confirmed("bash:rm -rf /"));
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());

        // Storing the same key twice keeps a single entry.
        store.store_auto_confirm("bash:ls");
        assert_eq!(store.len(), 2);

        let copy = store.clone();
        store.clear();
        assert!(store.is_empty());
        assert!(!store.is_auto_confirmed("bash:ls"));
        assert!(copy.is_auto_confirmed("bash:ls"));
    }

    #[test]
    fn session_store_never_stores_an_empty_key() {
        let store = SessionPermissionStore::new();
        store.store_auto_confirm("");
        assert!(store.is_empty());
        assert!(!store.is_auto_confirmed(""));
        assert_eq!(store.len(), 0);
    }

    // ----- host defaults ---------------------------------------------------

    #[tokio::test]
    async fn noop_host_hooks_are_inert() {
        let host = NoopPermissionHost;
        assert!(host.permissions_snapshot().is_none());
        assert!(host.resolve_workspace_dir().is_none());
        assert!(host.session_id().is_none());
        assert!(host.permission_yaml_path().is_none());

        let input = SceneHookInput {
            tool_name: "bash".to_owned(),
            tool_call_id: "call-1".to_owned(),
            arguments: json!({"command": "ls"}),
            answer_carried: false,
        };
        assert_eq!(host.scene_hook(&input).await, None);

        let request = PermissionConfirmationRequest {
            tool_name: "bash".to_owned(),
            tool_call_id: "call-1".to_owned(),
            arguments: json!({"command": "ls"}),
            result: PermissionResult::new(PermissionLevel::Ask),
            auto_confirm_key: "bash:ls".to_owned(),
        };
        assert_eq!(
            host.request_confirmation(&request).await,
            ConfirmationOutcome::Interrupt
        );

        let section = PermissionsSection::default();
        assert!(!host.persist_allow_rule(&section).await);
        assert!(!host.persist_session_allow_rule(&section).await);
    }

    // ----- auto-confirm truth table ---------------------------------------

    #[test]
    fn should_store_auto_confirm_matches_the_python_truth_table() {
        // Every combination that misses one precondition must not store.
        for approved in [false, true] {
            for auto_confirm in [false, true] {
                for has_session in [false, true] {
                    for key in ["", "bash:ls"] {
                        for persisted in [false, true] {
                            let expected =
                                approved && auto_confirm && has_session && !key.is_empty();
                            assert_eq!(
                                should_store_auto_confirm(
                                    approved,
                                    auto_confirm,
                                    has_session,
                                    key,
                                    persisted,
                                    false
                                ),
                                expected,
                                "approved={approved} auto_confirm={auto_confirm} \
                                 has_session={has_session} key={key:?} persisted={persisted}"
                            );
                            // A permanent rule that was persisted is reloaded from
                            // the host snapshot, so it is not stored in the session.
                            assert_eq!(
                                should_store_auto_confirm(
                                    approved,
                                    auto_confirm,
                                    has_session,
                                    key,
                                    persisted,
                                    true
                                ),
                                expected && !persisted,
                                "permanent approved={approved} persisted={persisted}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn should_store_auto_confirm_keeps_session_scope_rules() {
        // Session-scoped persist (permanent == false) must store the key even when
        // the host reported a successful write.
        assert!(should_store_auto_confirm(
            true, true, true, "bash:ls", true, false
        ));
        assert!(!should_store_auto_confirm(
            true, true, true, "bash:ls", true, true
        ));
        // Permanence that failed to persist still falls back to the session store.
        assert!(should_store_auto_confirm(
            true, true, true, "bash:ls", false, true
        ));
    }

    // ----- tiered allow-rule merge ----------------------------------------

    /// Section whose `bash` baseline asks.
    fn ask_bash_section() -> PermissionsSection {
        let mut section = PermissionsSection::default();
        section
            .tools
            .insert("bash".to_owned(), PermissionLevel::Ask);
        section
    }

    #[test]
    fn non_shell_tool_allow_merge_is_idempotent() {
        let section = ask_bash_section();
        let args = json!({"path": "/ws/a.txt"});

        let outcome = merge_allow_rule_into_permissions(&section, "read_file", &args);
        assert!(outcome.rule_merged);
        assert!(!outcome.path_merged);
        assert!(outcome.applied());
        assert_eq!(
            outcome.permissions.tools.get("read_file"),
            Some(&PermissionLevel::Allow)
        );
        assert_eq!(outcome.permissions.approval_overrides.len(), 0);
        // The input section is untouched.
        assert_eq!(section.tools.get("read_file"), None);

        let again = merge_allow_rule_into_permissions(&outcome.permissions, "read_file", &args);
        assert!(!again.rule_merged);
        assert!(!again.applied());
        assert_eq!(again.permissions, outcome.permissions);
    }

    #[test]
    fn tool_aliases_are_normalized_before_the_merge() {
        let section = PermissionsSection::default();
        let outcome = merge_allow_rule_into_permissions(
            &section,
            "fetch_webpage",
            &json!({"url": "https://example.com"}),
        );
        // `mcp_fetch_webpage` is a network tool, not a shell tool, so the merge
        // records a whole-tool allow under its canonical name.
        assert!(outcome.rule_merged);
        assert_eq!(
            outcome.permissions.tools.get("mcp_fetch_webpage"),
            Some(&PermissionLevel::Allow)
        );
        assert_eq!(outcome.permissions.tools.get("fetch_webpage"), None);
    }

    #[test]
    fn simple_shell_command_merges_one_command_override() {
        let section = ask_bash_section();
        let args = json!({"command": "git status"});
        assert_eq!(
            tiered::evaluate_tiered_policy(&section, "bash", &args).0,
            PermissionLevel::Ask
        );

        let outcome = merge_allow_rule_into_permissions(&section, "bash", &args);
        assert!(outcome.rule_merged);
        assert!(outcome.applied());
        assert_eq!(outcome.permissions.approval_overrides.len(), 1);
        let entry = outcome.permissions.approval_overrides.first().unwrap();
        assert_eq!(entry.action, PermissionLevel::Allow);
        assert_eq!(entry.match_type, RuleMatchType::Command);
        assert!(entry.pattern.contains("git status"), "{}", entry.pattern);
        assert_eq!(entry.tools, vec!["bash".to_owned()]);
        assert_ne!(entry.id, "");
        assert!(
            entry
                .id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{}",
            entry.id
        );
        // A shell tool is never widened to a whole-tool allow.
        assert_eq!(
            outcome.permissions.tools.get("bash"),
            Some(&PermissionLevel::Ask)
        );

        let (level, label) = tiered::evaluate_tiered_policy(&outcome.permissions, "bash", &args);
        assert_eq!(level, PermissionLevel::Allow);
        assert!(label.contains("approval_overrides"), "{label}");
        assert!(!label.contains("tools.bash"), "{label}");
    }

    #[test]
    fn risky_shell_commands_are_never_remembered() {
        for command in ["rm a.txt && echo done", "cat f | sh", "echo $(id)", ""] {
            let section = ask_bash_section();
            let args = json!({"command": command});
            let outcome = merge_allow_rule_into_permissions(&section, "bash", &args);
            assert!(!outcome.applied(), "{command}");
            assert!(
                outcome.permissions.approval_overrides.is_empty(),
                "{command}"
            );
            assert_eq!(
                outcome.permissions.tools.get("bash"),
                Some(&PermissionLevel::Ask)
            );
            assert_eq!(outcome.permissions, section, "{command}");
        }
    }

    #[test]
    fn packaged_builtin_deny_is_never_remembered() {
        let mut section = ask_bash_section();
        section.rules = builtin::package_command_rules();
        let args = json!({"command": "shutdown -h now"});
        assert_eq!(
            tiered::evaluate_tiered_policy(&section, "bash", &args).0,
            PermissionLevel::Deny
        );

        let outcome = merge_allow_rule_into_permissions(&section, "bash", &args);
        assert!(!outcome.applied());
        assert!(!outcome.rule_merged);
        assert_eq!(outcome.permissions, section);
        assert_eq!(outcome.permissions.approval_overrides.len(), 0);
    }

    #[test]
    fn approval_override_id_is_sanitized_and_bounded() {
        let id = approval_override_id("bash", RuleMatchType::Command, "git status");
        assert_eq!(id, "user_allow_bash_command_git_status");

        let id = approval_override_id("bash", RuleMatchType::Command, "/tmp/Weird-Path:$(x)");
        assert!(
            !id.contains('/') && !id.contains(':') && !id.contains('('),
            "{id}"
        );
        assert!(id.starts_with("user_allow_bash_command_"), "{id}");

        let long = approval_override_id("bash", RuleMatchType::Command, &"a".repeat(400));
        assert_eq!(long.len(), MAX_APPROVAL_OVERRIDE_ID_LEN);

        // The fixed prefix keeps the id non-empty, exactly like Python; the
        // `user_allow_override` fallback is only a safety net.
        let empty = approval_override_id("", RuleMatchType::Command, "   ");
        assert_eq!(empty, "user_allow_command");
    }

    // ----- file-guard allow merge -----------------------------------------

    fn entry_for<'a>(outcome: &'a MergeOutcome, path: &str) -> &'a FileGuardPathEntry {
        outcome
            .permissions
            .file_guard
            .paths
            .iter()
            .find(|entry| entry.path == path)
            .unwrap()
    }

    #[test]
    fn file_guard_read_merge_never_opens_a_write() {
        let section = PermissionsSection::default();
        assert!(!section.file_guard.enabled);

        let outcome = merge_file_guard_access_allows(
            &section,
            &[PathAccess::new(Path::new("/data/a.txt"), FileAction::Read)],
        );
        assert!(outcome.path_merged);
        assert!(!outcome.rule_merged);
        assert!(outcome.applied());
        assert!(outcome.permissions.file_guard.enabled);
        assert_eq!(outcome.permissions.file_guard.paths.len(), 1);
        let entry = entry_for(&outcome, "/data/a.txt");
        assert_eq!(entry.read, PermissionLevel::Allow);
        assert_eq!(entry.write, PermissionLevel::Ask);
        assert_eq!(entry.exec, PermissionLevel::Ask);
        assert_eq!(entry.r#match, PathMatch::Prefix);
        assert_eq!(entry.id, None);
        // The input section is untouched.
        assert!(!section.file_guard.enabled);
        assert_eq!(section.file_guard.paths.len(), 0);
    }

    #[test]
    fn file_guard_write_merge_widens_the_same_entry() {
        let section = PermissionsSection::default();
        let first = merge_file_guard_access_allows(
            &section,
            &[PathAccess::new(Path::new("/data/a.txt"), FileAction::Read)],
        );
        assert!(first.path_merged);

        let second = merge_file_guard_access_allows(
            &first.permissions,
            &[PathAccess::new(Path::new("/data/a.txt"), FileAction::Write)],
        );
        assert!(second.path_merged);
        assert!(second.permissions.file_guard.enabled);
        assert_eq!(second.permissions.file_guard.paths.len(), 1);
        let entry = entry_for(&second, "/data/a.txt");
        assert_eq!(entry.read, PermissionLevel::Allow);
        assert_eq!(entry.write, PermissionLevel::Allow);
        assert_eq!(entry.exec, PermissionLevel::Ask);

        // Merging the same write again changes nothing.
        let third = merge_file_guard_access_allows(
            &second.permissions,
            &[PathAccess::new(Path::new("/data/a.txt"), FileAction::Write)],
        );
        assert!(!third.path_merged);
        assert_eq!(third.permissions, second.permissions);
    }

    #[test]
    fn file_guard_exec_axes_and_duplicates() {
        let section = PermissionsSection::default();
        let outcome = merge_file_guard_access_allows(
            &section,
            &[
                PathAccess::new(Path::new("/bin/tool"), FileAction::Exec),
                PathAccess::new(Path::new("/bin/tool/"), FileAction::Exec),
                PathAccess::new(Path::new("/bin/tool"), FileAction::Read),
                PathAccess::new(Path::new(""), FileAction::Read),
            ],
        );
        assert!(outcome.path_merged);
        assert_eq!(outcome.permissions.file_guard.paths.len(), 1);
        let entry = entry_for(&outcome, "/bin/tool");
        assert_eq!(entry.read, PermissionLevel::Allow);
        assert_eq!(entry.write, PermissionLevel::Ask);
        assert_eq!(entry.exec, PermissionLevel::Allow);
    }

    #[test]
    fn file_guard_merge_cannot_narrow_an_existing_deny() {
        let mut section = PermissionsSection::default();
        section.file_guard.enabled = true;
        section.file_guard.paths.push(FileGuardPathEntry {
            path: "/secret".to_owned(),
            read: PermissionLevel::Allow,
            write: PermissionLevel::Deny,
            exec: PermissionLevel::Deny,
            ..FileGuardPathEntry::default()
        });

        let outcome = merge_file_guard_access_allows(
            &section,
            &[PathAccess::new(Path::new("/secret"), FileAction::Read)],
        );
        assert!(!outcome.path_merged);
        let entry = entry_for(&outcome, "/secret");
        assert_eq!(entry.write, PermissionLevel::Deny);
        assert_eq!(entry.exec, PermissionLevel::Deny);
    }

    #[test]
    fn empty_file_guard_accesses_leave_the_section_alone() {
        let section = PermissionsSection::default();
        let outcome = merge_file_guard_access_allows(&section, &[]);
        assert!(!outcome.applied());
        assert!(!outcome.path_merged);
        assert_eq!(outcome.permissions, section);
        assert!(!outcome.permissions.file_guard.enabled);
    }
}
