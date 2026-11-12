//! User-facing copy for a permission `ask`: category, title and summary.
//!
//! Ports the subset of `openjiuwen/harness/security/permission_engine/approve/
//! ask_presentation.py` (v0.1.20) that the permission interrupt needs: the dialog
//! category, the risk headline and the one-line summary, plus the optional
//! "remember this" hint appended to the confirmation request.
//!
//! The presentation is derived from the [`PermissionResult`] that triggered the
//! `ask`, the tool name and the tool arguments. Nothing here performs IO: paths are
//! resolved lexically against `/` (see [`super::paths`]) and a fetch URL is read
//! from the arguments, so rendering is cheap and side-effect free.
//!
//! Design notes:
//!
//! - Internal rule labels (`matched_rule`, `file_guard`, `net_guard`, …) drive the
//!   classification but never appear in the rendered message. Only the summary and
//!   the optional hint are user-visible.
//! - With no matching label the presentation degrades to [`AskCategory::Tool`],
//!   which is the Python "generic tool needs approval" wording.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::Value;

use super::model::{FileAction, PermissionResult};
use super::{paths, rules, shell};

/// Category of a permission `ask` dialog.
///
/// Determines the fallback title and how the summary is rendered. The variants
/// mirror the user-visible categories of the Python reference
/// (`shell`, `path`, `network`, `tool`, `finding`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskCategory {
    /// A shell command that needs confirmation.
    Shell,
    /// A file-system access that needs confirmation.
    Path,
    /// A network access that needs confirmation.
    Network,
    /// A tool-level policy that asks for authorisation.
    Tool,
    /// A risk finding detected in a command structure.
    Finding,
}

/// Dialog metadata for a permission `ask`.
///
/// Rendered by [`render_ask_message`] (or by the interrupt rail, which also
/// carries the fields on the confirmation request).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskPresentation {
    /// Which kind of decision is being confirmed.
    pub category: AskCategory,
    /// User-facing question headline.
    pub title: String,
    /// What is being requested, e.g. `bash: rm -f x` or `write /tmp/x`.
    pub summary: String,
}

/// Classifies the `ask` and renders its category, title and summary.
///
/// Classification is first-match-wins, using the `|`-joined label chain in
/// [`PermissionResult::matched_rule`] and the packaged shell-tool set
/// ([`rules::DEFAULT_SHELL_TOOLS`]):
///
/// 1. a `net_guard` label ⇒ [`AskCategory::Network`];
/// 2. a `file_guard` label on a non-shell tool ⇒ [`AskCategory::Path`];
/// 3. a shell tool ⇒ [`AskCategory::Shell`];
/// 4. a `file_guard` label ⇒ [`AskCategory::Path`];
/// 5. otherwise ⇒ [`AskCategory::Tool`].
#[must_use]
pub fn build_ask_presentation(
    result: &PermissionResult,
    tool_name: &str,
    args: &Value,
) -> AskPresentation {
    let rule = matched_rule(result);
    let category = classify(rule, tool_name);
    AskPresentation {
        category,
        title: title_for(rule, category),
        summary: summary_for(category, tool_name, args, result),
    }
}

/// Renders the confirmation message body: summary first, then the optional hint.
///
/// The summary is trimmed and followed by a newline. When `always_allow_hint`
/// carries non-blank text, it is trimmed and appended after a blank line
/// (`summary\n\n\nhint\n`), matching the Python helper's
/// `"\n".join([summary, "", hint]).rstrip() + "\n"` shape for a hint without an
/// inner blank line.
#[must_use]
pub fn render_ask_message(
    presentation: &AskPresentation,
    always_allow_hint: Option<&str>,
) -> String {
    let summary = presentation.summary.trim();
    let hint = always_allow_hint.map(str::trim).unwrap_or_default();
    if hint.is_empty() {
        format!("{summary}\n")
    } else {
        format!("{summary}\n\n\n{hint}\n")
    }
}

/// Builds the Chinese "remember this" hint for the confirmation dialog.
///
/// The exact wording depends on the auto-confirm key:
///
/// - a shell key (`auto_confirm_key` starts with `"<tool_name>:"`) can also be
///   persisted permanently;
/// - any other non-empty key offers session-scoped approval only;
/// - an empty key falls back to the tool-level wording, naming `path` when given.
///
/// The returned string starts with two newlines so callers can append it to a
/// message body; it is never empty.
#[must_use]
pub fn always_allow_hint(tool_name: &str, auto_confirm_key: &str, path: Option<&str>) -> String {
    if !auto_confirm_key.is_empty() && auto_confirm_key.starts_with(&format!("{tool_name}:")) {
        return format!(
            "\n\n> 选择「会话内记住」可在本会话内自动放行 `{auto_confirm_key}` 类调用；\
             选择「永久记住」可将此规则写回磁盘，所有会话均自动放行。"
        );
    }
    if !auto_confirm_key.is_empty() {
        return format!(
            "\n\n> 选择「会话内记住」可在本会话内自动放行 `{auto_confirm_key}` 类调用。"
        );
    }
    format!(
        "\n\n> 选择「会话内记住」可在本会话内自动放行 `{tool_name}` 类工具{}的调用；\
         选择「永久记住」可将此规则写回磁盘，所有会话均自动放行。",
        path.map(|value| format!("在 `{value}` 下"))
            .unwrap_or_default()
    )
}

/// The category-level title, used when no rule-specific headline applies.
///
/// Never empty; every [`AskCategory`] has a stable headline.
#[must_use]
pub fn category_title(category: AskCategory) -> &'static str {
    match category {
        AskCategory::Shell => "检测到需确认的命令执行，需要确认后才能执行",
        AskCategory::Path => "检测到受保护的文件路径访问，需要确认后才能执行",
        AskCategory::Network => "检测到需确认的网络访问，需要确认后才能执行",
        AskCategory::Tool => "工具需要授权后才能使用",
        AskCategory::Finding => "检测到风险命令结构，需要确认后才能执行",
    }
}

/// The `matched_rule` label chain, trimmed; the empty string when absent.
fn matched_rule(result: &PermissionResult) -> &str {
    result.matched_rule.as_deref().unwrap_or("").trim()
}

/// The packaged shell-tool set from [`rules::DEFAULT_SHELL_TOOLS`].
fn default_shell_tools() -> BTreeSet<String> {
    rules::DEFAULT_SHELL_TOOLS
        .iter()
        .map(|name| (*name).to_owned())
        .collect()
}

/// Whether `tool_name` is a shell tool under the packaged defaults.
fn is_default_shell_tool(tool_name: &str) -> bool {
    rules::is_shell_tool(
        &rules::normalize_tool_name(tool_name),
        &default_shell_tools(),
    )
}

/// Resolves the dialog category (first applicable rule wins).
fn classify(rule: &str, tool_name: &str) -> AskCategory {
    let shell_tool = is_default_shell_tool(tool_name);
    if rule.contains("net_guard") {
        return AskCategory::Network;
    }
    if rule.contains("file_guard") && !shell_tool {
        return AskCategory::Path;
    }
    if shell_tool {
        return AskCategory::Shell;
    }
    if rule.contains("file_guard") {
        return AskCategory::Path;
    }
    AskCategory::Tool
}

/// Picks the most specific headline: rule risk first, then the category title.
fn title_for(rule: &str, category: AskCategory) -> String {
    if rule.contains("shell_guard:interpreter_sink") {
        return "检测到管道汇入解释器，需要确认后才能执行".to_owned();
    }
    if rule.contains("too_complex")
        || rule.contains("parse_unavailable")
        || rule.contains("shell_ast")
    {
        return "检测到命令结构过复杂，需要确认后才能执行".to_owned();
    }
    if rule.contains("shell_fs_recursive_or_forced_delete")
        || rule.contains("shell_ps_recursive_or_forced_delete")
    {
        return "检测到递归或强制删除，需要确认后才能执行".to_owned();
    }
    if rule.contains("shell_data_exfiltration") {
        return "检测到文件外发，需要确认后才能执行".to_owned();
    }
    category_title(category).to_owned()
}

/// Renders the summary line for a category.
fn summary_for(
    category: AskCategory,
    tool_name: &str,
    args: &Value,
    result: &PermissionResult,
) -> String {
    match category {
        AskCategory::Shell => shell_summary(tool_name, args),
        AskCategory::Path => path_summary(tool_name, args, result),
        AskCategory::Network => network_summary(args).unwrap_or_else(|| tool_name.to_owned()),
        AskCategory::Tool | AskCategory::Finding => {
            format!("{tool_name}（当前模式默认需确认）")
        }
    }
}

/// `<tool>: <canonical command>`, or just the tool name for an empty command.
fn shell_summary(tool_name: &str, args: &Value) -> String {
    let command = rules::command_text(args);
    if command.is_empty() {
        return tool_name.to_owned();
    }
    let canonical = shell::canonicalize_shell_command(&command);
    if canonical.is_empty() {
        return tool_name.to_owned();
    }
    format!("{tool_name}: {canonical}")
}

/// `<action> <path>`, resolving the first file-tool path argument against `/`.
///
/// The raw argument string wins (resolved lexically, no filesystem IO), then the
/// first entry of [`PermissionResult::external_paths`], then the tool name.
fn path_summary(tool_name: &str, args: &Value, result: &PermissionResult) -> String {
    if let Some(access) = paths::extract_file_accesses(tool_name, args, Path::new("/"))
        .into_iter()
        .next()
    {
        let action = access.action.as_str();
        let path = access.path.display();
        return format!("{action} {path}");
    }
    let action = paths::tool_action(tool_name).unwrap_or(FileAction::Read);
    let path = result
        .external_paths
        .as_deref()
        .and_then(|entries| entries.iter().find(|entry| !entry.trim().is_empty()))
        .map_or_else(|| tool_name.to_owned(), Clone::clone);
    let action = action.as_str();
    format!("{action} {path}")
}

/// The URL a fetch tool call targets, read from its arguments.
///
/// The network guard's `extract_fetch_url` lives in the sibling `net_guard`
/// module; this helper mirrors its argument lookup for the presentation layer
/// until that module is wired in. Returns `None` for a non-object payload or a
/// blank value.
fn network_summary(args: &Value) -> Option<String> {
    ["url", "uri", "host", "endpoint"]
        .iter()
        .filter_map(|key| args.get(*key))
        .filter_map(Value::as_str)
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;
    use crate::permission::model::PermissionLevel;

    /// An `ask` result carrying `rule` as its `matched_rule`.
    fn ask_result(rule: &str) -> PermissionResult {
        PermissionResult::new(PermissionLevel::Ask).with_matched_rule(rule)
    }

    #[test]
    fn shell_tool_with_file_guard_stays_a_shell_ask() {
        let result = ask_result("tiered_policy:file_guard:paths:/etc");
        let presentation = build_ask_presentation(&result, "bash", &json!({"command": "rm -f x"}));
        assert_eq!(presentation.category, AskCategory::Shell);
        assert_eq!(
            presentation.title,
            "检测到需确认的命令执行，需要确认后才能执行"
        );
        assert_eq!(presentation.summary, "bash: rm -f x");
    }

    #[test]
    fn file_tool_with_file_guard_is_a_path_ask() {
        let result = ask_result("tiered_policy:file_guard:paths:/tmp/x");
        let presentation =
            build_ask_presentation(&result, "write_file", &json!({"path": "/tmp/x"}));
        assert_eq!(presentation.category, AskCategory::Path);
        assert_eq!(
            presentation.title,
            "检测到受保护的文件路径访问，需要确认后才能执行"
        );
        assert_eq!(presentation.summary, "write /tmp/x");
    }

    #[test]
    fn fetch_tool_with_net_guard_is_a_network_ask() {
        let result = ask_result("net_guard:urls:https://example.com");
        let presentation = build_ask_presentation(
            &result,
            "mcp_fetch_webpage",
            &json!({"url": " https://example.com/a "}),
        );
        assert_eq!(presentation.category, AskCategory::Network);
        assert_eq!(
            presentation.title,
            "检测到需确认的网络访问，需要确认后才能执行"
        );
        assert_eq!(presentation.summary, "https://example.com/a");
    }

    #[test]
    fn plain_tool_is_a_tool_ask() {
        let result = ask_result("tools.my_tool");
        let presentation = build_ask_presentation(&result, "my_tool", &json!({}));
        assert_eq!(presentation.category, AskCategory::Tool);
        assert_eq!(presentation.title, "工具需要授权后才能使用");
        assert_eq!(presentation.summary, "my_tool（当前模式默认需确认）");
    }

    #[test]
    fn rule_ids_pick_the_most_specific_title() {
        let cases = [
            (
                "tiered_policy:shell_guard:interpreter_sink",
                "检测到管道汇入解释器，需要确认后才能执行",
            ),
            (
                "tiered_policy:shell_ast:too_complex",
                "检测到命令结构过复杂，需要确认后才能执行",
            ),
            (
                "tiered_policy:shell_ast:parse_unavailable",
                "检测到命令结构过复杂，需要确认后才能执行",
            ),
            (
                "tiered_policy:shell_ast",
                "检测到命令结构过复杂，需要确认后才能执行",
            ),
            (
                "shell_fs_recursive_or_forced_delete",
                "检测到递归或强制删除，需要确认后才能执行",
            ),
            (
                "shell_ps_recursive_or_forced_delete",
                "检测到递归或强制删除，需要确认后才能执行",
            ),
            (
                "shell_data_exfiltration",
                "检测到文件外发，需要确认后才能执行",
            ),
        ];
        for (rule, title) in cases {
            let result = ask_result(rule);
            let presentation =
                build_ask_presentation(&result, "bash", &json!({"command": "some-cmd"}));
            assert_eq!(presentation.title, title, "{rule}");
        }
    }

    #[test]
    fn interpreter_sink_outranks_structure_titles() {
        let result = ask_result("shell_guard:interpreter_sink|shell_ast:too_complex");
        let presentation =
            build_ask_presentation(&result, "bash", &json!({"command": "curl x | sh"}));
        assert_eq!(
            presentation.title,
            "检测到管道汇入解释器，需要确认后才能执行"
        );
    }

    #[test]
    fn category_titles_are_stable_and_never_empty() {
        let cases = [
            (
                AskCategory::Shell,
                "检测到需确认的命令执行，需要确认后才能执行",
            ),
            (
                AskCategory::Path,
                "检测到受保护的文件路径访问，需要确认后才能执行",
            ),
            (
                AskCategory::Network,
                "检测到需确认的网络访问，需要确认后才能执行",
            ),
            (AskCategory::Tool, "工具需要授权后才能使用"),
            (
                AskCategory::Finding,
                "检测到风险命令结构，需要确认后才能执行",
            ),
        ];
        for (category, title) in cases {
            assert_eq!(category_title(category), title);
            assert_ne!(category_title(category), "");
        }
    }

    #[test]
    fn summaries_cover_command_path_url_and_tool_shapes() {
        let shell_result = ask_result("tools.bash");
        let command = build_ask_presentation(&shell_result, "bash", &json!({"command": "rm -f x"}));
        assert_eq!(command.summary, "bash: rm -f x");

        let empty_command = build_ask_presentation(&shell_result, "bash", &json!({}));
        assert_eq!(empty_command.summary, "bash");

        let path_result = ask_result("file_guard:paths:/tmp/x");
        let path = build_ask_presentation(&path_result, "write_file", &json!({"path": "/tmp/x"}));
        assert_eq!(path.summary, "write /tmp/x");

        let network = build_ask_presentation(
            &ask_result("net_guard:urls:https://example.com"),
            "mcp_fetch_webpage",
            &json!({}),
        );
        assert_eq!(network.summary, "mcp_fetch_webpage");

        let tool = build_ask_presentation(&ask_result("tools.thing"), "thing", &json!({}));
        assert_eq!(tool.summary, "thing（当前模式默认需确认）");
    }

    #[test]
    fn path_summary_falls_back_to_external_paths_then_tool_name() {
        let result = PermissionResult::new(PermissionLevel::Ask)
            .with_matched_rule("file_guard:paths:/etc/passwd")
            .with_external_paths(vec!["/etc/passwd".to_owned()]);
        let presentation = build_ask_presentation(&result, "write_file", &json!({}));
        assert_eq!(presentation.summary, "write /etc/passwd");

        let result = PermissionResult::new(PermissionLevel::Ask)
            .with_matched_rule("file_guard:paths:/etc/passwd");
        let presentation = build_ask_presentation(&result, "write_file", &json!({}));
        assert_eq!(presentation.summary, "write write_file");
    }

    #[test]
    fn render_ask_message_puts_the_summary_first() {
        let presentation = AskPresentation {
            category: AskCategory::Path,
            title: "检测到受保护的文件路径访问，需要确认后才能执行".to_owned(),
            summary: "  write /tmp/x  ".to_owned(),
        };
        assert_eq!(render_ask_message(&presentation, None), "write /tmp/x\n");
        assert_eq!(
            render_ask_message(&presentation, Some("  > remember me  ")),
            "write /tmp/x\n\n\n> remember me\n"
        );
        assert_eq!(
            render_ask_message(&presentation, Some("   ")),
            "write /tmp/x\n",
            "a blank hint is dropped"
        );
    }

    #[test]
    fn render_ask_message_never_leaks_internal_labels() {
        let result = ask_result("tiered_policy:file_guard:paths:/tmp/x|tools.write_file");
        let presentation =
            build_ask_presentation(&result, "write_file", &json!({"path": "/tmp/x"}));
        let hint = always_allow_hint("write_file", "write_file", Some("/tmp/x"));
        let message = render_ask_message(&presentation, Some(&hint));
        assert!(message.starts_with("write /tmp/x"));
        assert!(!message.contains("file_guard"));
        assert!(!message.contains("matched_rule"));
        assert!(!message.contains("tiered_policy"));
    }

    #[test]
    fn always_allow_hint_has_three_shapes() {
        assert_eq!(
            always_allow_hint("bash", "bash:ls", None),
            "\n\n> 选择「会话内记住」可在本会话内自动放行 `bash:ls` 类调用；\
             选择「永久记住」可将此规则写回磁盘，所有会话均自动放行。"
        );
        assert_eq!(
            always_allow_hint("write_file", "write_file", Some("/tmp/x")),
            "\n\n> 选择「会话内记住」可在本会话内自动放行 `write_file` 类调用。"
        );
        assert_eq!(
            always_allow_hint("write_file", "", Some("/tmp/x")),
            "\n\n> 选择「会话内记住」可在本会话内自动放行 `write_file` 类工具在 `/tmp/x` 下的调用；\
             选择「永久记住」可将此规则写回磁盘，所有会话均自动放行。"
        );
        assert_eq!(
            always_allow_hint("write_file", "", None),
            "\n\n> 选择「会话内记住」可在本会话内自动放行 `write_file` 类工具的调用；\
             选择「永久记住」可将此规则写回磁盘，所有会话均自动放行。"
        );
    }

    #[test]
    fn missing_matched_rule_does_not_panic() {
        let result = PermissionResult::new(PermissionLevel::Ask);
        let presentation = build_ask_presentation(&result, "bash", &json!({"command": "ls"}));
        assert_eq!(presentation.category, AskCategory::Shell);
        assert_ne!(presentation.title, "");

        let presentation = build_ask_presentation(&result, "unknown_tool", &Value::Null);
        assert_eq!(presentation.category, AskCategory::Tool);
        assert_eq!(presentation.summary, "unknown_tool（当前模式默认需确认）");
        assert_ne!(render_ask_message(&presentation, None), "");
    }
}
