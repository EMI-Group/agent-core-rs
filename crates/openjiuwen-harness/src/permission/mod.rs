//! Permission model, engine, file/shell guards, session decisions, and HITL interrupt rail.
//!
//! A `DeepAgent` tool call passes one tiered policy pipeline plus two targeted guards
//! before it runs:
//!
//! 1. [`tiered`] — builtin command rules, user rules, approval overrides, tool
//!    baselines and `defaults["*"]`, always yielding a decision.
//! 2. [`file_guard`] — path read/write/exec policy for file and shell tools.
//! 3. [`net_guard`] — URL allow/deny policy for fetch tools.
//!
//! [`engine::PermissionEngine`] folds those three with [`engine::strictest`]
//! (`deny < ask < allow`) and renders the Python-compatible `matched_rule` and
//! `reason` strings. [`interrupt::PermissionInterruptRail`] turns an `ask` into a
//! human-in-the-loop confirmation on the kernel's [`openjiuwen_agent::AgentRail`]
//! seam, with session-scoped and permanent "remember" persistence routed through
//! [`session::PermissionHost`].
//!
//! # Layers
//!
//! The module is layered from pure data upwards; each layer only uses the ones
//! before it:
//!
//! - **Data and matchers** — [`model`], [`patterns`], [`builtin`], [`rules`],
//!   [`shell`], [`paths`]: config/decision types, glob/regex/URL matching, the
//!   packaged rule table, tool categorization, shell analysis and lexical path
//!   extraction. No IO, no decisions.
//! - **Decision pipelines** — [`tiered`], [`file_guard`], [`net_guard`]: each
//!   answers allow/ask/deny for one axis, or "no opinion" where the pipeline has
//!   nothing to say.
//! - **Fold and façade** — [`engine`]: combines the pipelines with `strictest`
//!   and exposes the single synchronous `check_permission` entry point.
//! - **Session decisions and presentation** — [`session`], [`presentation`]: host
//!   hooks, session-scoped approvals and the human-facing ask copy.
//! - **Human-in-the-loop** — [`interrupt`]: the agent rail that raises an `ask`
//!   into a confirmation prompt and persists remembered answers.
//!
//! The module contract, deviations from the Python reference and the test strategy
//! are documented in `CONTEXT.md`.

pub mod builtin;
pub mod engine;
pub mod file_guard;
pub mod interrupt;
pub mod model;
pub mod net_guard;
pub mod paths;
pub mod patterns;
pub mod presentation;
pub mod rules;
pub mod session;
pub mod shell;
pub mod tiered;

#[cfg(test)]
mod tests;

// --- model ---
pub use model::{
    ApprovalOverrideEntry, FileAction, FileGuardDefaults, FileGuardPathEntry, FileGuardSection,
    NetGuardSection, PathAccess, PathMatch, PermissionConfirmResponse, PermissionLevel,
    PermissionResult, PermissionRule, PermissionsSection, RuleMatchType, ShellGuardSection,
};

// --- patterns ---
pub use patterns::{match_glob, match_net_pattern, match_wildcard, PolicyMatcher};

// --- builtin ---
pub use builtin::{
    package_command_rule_count, package_command_rules, package_net_urls, package_sensitive_paths,
};

// --- rules ---
pub use rules::{
    command_text, is_shell_tool, normalize_permission_mode, normalize_tool_name, rule_ref,
    severity_to_action, severity_to_decision, shell_tools_from_config, tool_category, ToolCategory,
    DEFAULT_SHELL_TOOLS, NETWORK_TOOLS, OS_CONTROL_TOOLS, PATH_TOOLS, TOOL_NAME_ALIASES,
};

// --- shell ---
pub use shell::{
    analyze_shell, canonicalize_shell_command, has_interpreter_sink, shell_auto_confirm_key,
    ShellCommand, ShellFlags, ShellKind, INTERPRETER_SINKS,
};

// --- paths ---
pub use paths::{extract_file_accesses, extract_shell_accesses, resolve_path, tool_action};

// --- tiered ---
pub use tiered::{evaluate_tiered_policy, FALLBACK_NO_CONFIG};

// --- file_guard / net_guard ---
pub use file_guard::FileGuardChecker;
pub use net_guard::{extract_fetch_url, NetGuardChecker, FETCH_TOOLS};

// --- engine ---
pub use engine::{
    prepare_permissions_for_engine, strictest, strictest_all, PermissionEngine,
    PermissionEngineConfig,
};

// --- session ---
pub use session::{
    merge_allow_rule_into_permissions, merge_file_guard_access_allows, should_store_auto_confirm,
    ConfirmationOutcome, MergeOutcome, NoopPermissionHost, PermissionConfirmationRequest,
    PermissionHost, SceneDecision, SceneHookInput, SessionPermissionStore,
};

// --- presentation ---
pub use presentation::{
    always_allow_hint, build_ask_presentation, render_ask_message, AskCategory, AskPresentation,
};

// --- interrupt ---
pub use interrupt::{
    PermissionInterruptRail, DENIED_PREFIX, PERSISTED_APPROVALS_KEY, RAIL_OWNER, RAIL_PRIORITY,
    REJECTED_MESSAGE, RESUME_ANSWERS_KEY,
};
