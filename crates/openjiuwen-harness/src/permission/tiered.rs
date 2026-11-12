//! Pipeline A of the permission engine: the tiered tool/command policy.
//!
//! Ports `openjiuwen/harness/security/permission_engine/toolguard/tool_policy.py`
//! (`evaluate_tiered_policy` and `_evaluate_single_invocation`, v0.1.20): the
//! parameter-rule ladder that turns one tool call into a
//! [`PermissionLevel`] plus the behavioural `matched_rule` label.
//!
//! The ladder is ordered strictest-first and yields exactly one decision — it
//! never reports "no opinion" and never widens to an allow on its own:
//!
//! 1. a whole-tool `deny` baseline short-circuits every parameter rule;
//! 2. packaged (`layer: "builtin"`) `deny` hits;
//! 3. user `deny` hits;
//! 4. a remembered `approval_overrides` allow;
//! 5. packaged hits of any level, then user hits of any level;
//! 6. the `tools.<tool>` baseline;
//! 7. `defaults["*"]`;
//! 8. [`FALLBACK_NO_CONFIG`] with [`PermissionLevel::Ask`].
//!
//! Shell tools additionally get three fail-closed floors — an unparsable or
//! unsupported command structure and a pipeline into an interpreter — and a
//! per-subcommand evaluation, so a remembered compound command can never relax a
//! subcommand denial.
//!
//! Once the ladder, the subcommand aggregate and the floors have produced a
//! decision, a remembered full-command `approval_overrides` allow relaxes an ASK
//! to an ALLOW (Python `_allow_overrides_unless_denied`); a DENY is returned
//! unchanged, so a remembered allow never lifts a denial.
//!
//! One deviation from the Python reference is deliberate: the legacy literal-shell
//! fast path (`_literal_policy_result`) is not ported.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::permission::model::{ApprovalOverrideEntry, PermissionLevel, PermissionsSection};
use crate::permission::rules;
use crate::permission::shell::{self, ShellCommand, ShellKind};

/// Byte-exact Python sentinel: when returned, the caller treats it as "no opinion".
///
/// Python raises the same label from `_evaluate_single_invocation` when the section
/// carries neither rules, an override, a tool baseline nor `defaults["*"]`; the
/// decision that travels with it is always [`PermissionLevel::Ask`].
pub const FALLBACK_NO_CONFIG: &str = "tiered_policy:fallback(no_config)";

/// Label namespace of every decision produced here (Python `_MR`).
const NAMESPACE: &str = "tiered_policy";

/// Label of the `defaults["*"]` rung.
const DEFAULTS_LABEL: &str = "tiered_policy:defaults.*";

/// Label prefix of the remembered-approval rung.
const APPROVAL_OVERRIDES_PREFIX: &str = "tiered_policy:approval_overrides";

/// Label prefix of the subcommand-aggregate rung.
const SUBCOMMANDS_PREFIX: &str = "tiered_policy:shell_subcommands";

/// One parameter rule that matched the invocation.
#[derive(Debug, Clone)]
struct RuleHit {
    /// Decision the rule resolves to (`action`, else the `severity` table).
    level: PermissionLevel,
    /// Rule reference used in labels, e.g. `builtin[shell_ps_..]` or `rules[deny_rm]`.
    reference: String,
}

/// Evaluates the tiered rule pipeline for one tool call. Always yields a decision.
///
/// `section` is the compiled `permissions:` configuration (builtin rules are
/// expected to be inlined with `layer: "builtin"` by
/// `engine::prepare_permissions_for_engine`), `tool_name` is the model-facing tool
/// name and `args` its JSON arguments.
///
/// The returned label is the Python `matched_rule` summary: the rungs are joined
/// with `:` (`tiered_policy:builtin:deny:<refs>`), competing references are joined
/// with `+` in rule order, and a shell floor that tightened the decision is
/// prefixed with `|`.
///
/// ```
/// use openjiuwen_harness::permission::model::PermissionsSection;
/// use openjiuwen_harness::permission::tiered::evaluate_tiered_policy;
/// use serde_json::json;
///
/// let section = PermissionsSection::default();
/// let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": "ls"}));
/// assert_eq!(level.as_str(), "ask");
/// assert_eq!(label, "tiered_policy:defaults.*");
/// ```
#[must_use]
pub fn evaluate_tiered_policy(
    section: &PermissionsSection,
    tool_name: &str,
    args: &Value,
) -> (PermissionLevel, String) {
    let tool = rules::normalize_tool_name(tool_name);
    let shell_tools = rules::shell_tools_from_config(section);
    let is_shell = rules::is_shell_tool(&tool, &shell_tools);
    let mode = rules::normalize_permission_mode(section.permission_mode.as_deref());

    // A whole-tool deny is final: no parameter rule and no remembered approval can
    // lift it.
    if section.level_for_tool(&tool) == Some(PermissionLevel::Deny) {
        return (PermissionLevel::Deny, format!("tools.{tool}"));
    }

    let (mut level, mut label) = evaluate_ladder(section, &tool, &shell_tools, mode, args);

    let raw_command = rules::command_text(args);
    let command = if is_shell {
        shell::canonicalize_shell_command(&raw_command)
    } else {
        String::new()
    };

    if !command.is_empty() {
        // Python parity: shell tools are evaluated on the raw arguments *and* on the
        // canonicalized command (`canon_args`), so a launcher wrapper such as
        // `bash -c "rm -v x"` cannot hide the command being run. The stricter result wins.
        if command != raw_command {
            let canonical_args = command_args(&command);
            let (canonical_level, canonical_label) =
                evaluate_ladder(section, &tool, &shell_tools, mode, &canonical_args);
            if rank(canonical_level) < rank(level) {
                level = canonical_level;
                label = canonical_label;
            }
        }

        let ast = shell::analyze_shell(&command);
        if ast.kind == ShellKind::Simple && ast.subcommands.len() > 1 {
            let (subcommand_level, subcommand_label) =
                aggregate_subcommands(section, &tool, &shell_tools, mode, &ast.subcommands);
            if rank(subcommand_level) < rank(level) {
                level = subcommand_level;
                label = subcommand_label;
            }
        }

        for (floor_level, floor_label) in shell_floors(section, &ast, &command) {
            // Floors never relax: only a stricter floor replaces the current decision.
            if rank(floor_level) < rank(level) {
                level = floor_level;
                label = format!("{floor_label}|{label}");
            }
        }
    }

    relax_with_approval_overrides(section, &tool, &shell_tools, args, &command, level, label)
}

/// Python `_allow_overrides_unless_denied`: a remembered full-command allow relaxes
/// an ASK produced by the ladder, the subcommand aggregate or a shell floor.
///
/// A DENY (including a subcommand denial) is returned unchanged, so a remembered
/// allow never lifts a denial; an ALLOW is already final and keeps its label.
/// `canonical_command` is retried as `{"command": …}` so a remembered quoted or
/// canonicalized command still matches, mirroring the override rung.
fn relax_with_approval_overrides(
    section: &PermissionsSection,
    tool: &str,
    shell_tools: &BTreeSet<String>,
    args: &Value,
    canonical_command: &str,
    level: PermissionLevel,
    label: String,
) -> (PermissionLevel, String) {
    if level != PermissionLevel::Ask {
        return (level, label);
    }
    let mut ids = matching_override_ids(section, tool, shell_tools, args);
    if ids.is_empty() && !canonical_command.is_empty() {
        ids = matching_override_ids(section, tool, shell_tools, &command_args(canonical_command));
    }
    if ids.is_empty() {
        return (level, label);
    }
    (PermissionLevel::Allow, override_label(&ids))
}

/// Ids of the `allow` overrides matching `args`.
fn matching_override_ids(
    section: &PermissionsSection,
    tool: &str,
    shell_tools: &BTreeSet<String>,
    args: &Value,
) -> Vec<String> {
    section
        .approval_overrides
        .iter()
        .filter(|entry| entry.action == PermissionLevel::Allow)
        .filter(|entry| rules::approval_override_applies(entry, tool, shell_tools, args))
        .map(override_id)
        .collect()
}

/// Label id of one override entry: its trimmed `id`, or `?` when blank.
fn override_id(entry: &ApprovalOverrideEntry) -> String {
    let id = entry.id.trim();
    if id.is_empty() {
        "?".to_owned()
    } else {
        id.to_owned()
    }
}

/// Renders the remembered-approval label of `ids` (sorted, deduped, `+`-joined).
fn override_label(ids: &[String]) -> String {
    let mut ids = ids.to_vec();
    ids.sort();
    ids.dedup();
    format!(
        "{APPROVAL_OVERRIDES_PREFIX}:approval_overrides[{}]",
        ids.join("+")
    )
}

/// The per-invocation ladder (steps 1–2 of the module docs) for one command text.
///
/// Always returns a decision; ties inside a tier resolve through [`rank`], which
/// keeps the first-listed (strictest) level.
fn evaluate_ladder(
    section: &PermissionsSection,
    tool: &str,
    shell_tools: &BTreeSet<String>,
    mode: &str,
    args: &Value,
) -> (PermissionLevel, String) {
    let (builtin_hits, user_hits) = collect_hits(section, tool, shell_tools, mode, args);

    if let Some(decision) = finalize_deny(&builtin_hits, "builtin") {
        return decision;
    }
    if let Some(decision) = finalize_deny(&user_hits, "rules") {
        return decision;
    }
    if let Some(decision) = approval_override_allow(section, tool, shell_tools, args) {
        return decision;
    }
    if !builtin_hits.is_empty() {
        return finalize_hits(&builtin_hits, "builtin");
    }
    if !user_hits.is_empty() {
        return finalize_hits(&user_hits, "rules");
    }
    if let Some(level) = section.level_for_tool(tool) {
        return (level, format!("tools.{tool}"));
    }
    if section.defaults.contains_key("*") {
        return (section.default_level(), DEFAULTS_LABEL.to_owned());
    }
    (PermissionLevel::Ask, FALLBACK_NO_CONFIG.to_owned())
}

/// Collects the parameter-rule hits of `args`, split into packaged and user rules.
fn collect_hits(
    section: &PermissionsSection,
    tool: &str,
    shell_tools: &BTreeSet<String>,
    mode: &str,
    args: &Value,
) -> (Vec<RuleHit>, Vec<RuleHit>) {
    let mut builtin_hits = Vec::new();
    let mut user_hits = Vec::new();
    for rule in &section.rules {
        if !rules::rule_tools_consistent(rule, shell_tools)
            || !rules::rule_applies_to(rule, tool, shell_tools)
            || !rules::rule_matches(section, rule, tool, args)
        {
            continue;
        }
        let level = rule
            .action
            .or_else(|| rules::severity_to_decision(rule.severity.as_deref(), mode));
        let Some(level) = level else {
            continue;
        };
        let hit = RuleHit {
            level,
            reference: rules::rule_ref(rule),
        };
        if rule.layer.as_deref() == Some("builtin") {
            builtin_hits.push(hit);
        } else {
            user_hits.push(hit);
        }
    }
    (builtin_hits, user_hits)
}

/// Rung `deny` of one hit tier; `None` when no hit of `hits` denies.
fn finalize_deny(hits: &[RuleHit], namespace: &str) -> Option<(PermissionLevel, String)> {
    if !hits.iter().any(|hit| hit.level == PermissionLevel::Deny) {
        return None;
    }
    let references = join_references(hits, Some(PermissionLevel::Deny));
    Some((
        PermissionLevel::Deny,
        format!("{NAMESPACE}:{namespace}:deny:{references}"),
    ))
}

/// Rung "hits of any level" of one hit tier: strictest level plus its references.
fn finalize_hits(hits: &[RuleHit], namespace: &str) -> (PermissionLevel, String) {
    let level = strictest_all(hits.iter().map(|hit| hit.level));
    let references = join_references(hits, Some(level));
    if references.is_empty() {
        return (level, format!("{NAMESPACE}:{namespace}"));
    }
    (level, format!("{NAMESPACE}:{namespace}:{references}"))
}

/// Rung "remembered approval": an accepted `allow` override wins over baselines.
fn approval_override_allow(
    section: &PermissionsSection,
    tool: &str,
    shell_tools: &BTreeSet<String>,
    args: &Value,
) -> Option<(PermissionLevel, String)> {
    let accepted: Vec<&ApprovalOverrideEntry> = section
        .approval_overrides
        .iter()
        .filter(|entry| rules::approval_override_applies(entry, tool, shell_tools, args))
        .collect();
    if accepted.is_empty()
        || !accepted
            .iter()
            .any(|entry| entry.action == PermissionLevel::Allow)
    {
        return None;
    }
    let ids: Vec<String> = accepted.iter().map(|entry| override_id(entry)).collect();
    Some((PermissionLevel::Allow, override_label(&ids)))
}

/// Evaluates every shell subcommand of a simple command and aggregates the results.
fn aggregate_subcommands(
    section: &PermissionsSection,
    tool: &str,
    shell_tools: &BTreeSet<String>,
    mode: &str,
    subcommands: &[String],
) -> (PermissionLevel, String) {
    let mut results: Vec<(String, PermissionLevel, String)> = Vec::new();
    for subcommand in subcommands {
        let text = subcommand.trim();
        if text.is_empty() {
            continue;
        }
        let args = command_args(text);
        let (level, label) = evaluate_ladder(section, tool, shell_tools, mode, &args);
        results.push((text.to_owned(), level, label));
    }
    aggregate_results(&results)
}

/// Folds per-subcommand decisions into one decision.
///
/// The strictest level wins and keeps the labels of the decisions that produced
/// it; a single result is passed through unchanged.
fn aggregate_results(results: &[(String, PermissionLevel, String)]) -> (PermissionLevel, String) {
    match results {
        [] => (
            PermissionLevel::Ask,
            format!("{SUBCOMMANDS_PREFIX}:fallback"),
        ),
        [(_, level, label)] => (*level, label.clone()),
        _ => {
            let level = strictest_all(results.iter().map(|(_, level, _)| *level));
            let mut contributing: Vec<String> = Vec::new();
            for (command, result_level, label) in results {
                if *result_level != level {
                    continue;
                }
                let entry = format!("{command}=>{label}");
                if !contributing.contains(&entry) {
                    contributing.push(entry);
                }
            }
            if contributing.is_empty() {
                return (level, SUBCOMMANDS_PREFIX.to_owned());
            }
            (
                level,
                format!("{SUBCOMMANDS_PREFIX}:{}", contributing.join("+")),
            )
        }
    }
}

/// The fail-closed shell floors that apply to `ast`, in Python order.
///
/// Every floor asks: an unsupported structure, an unparsable structure that still
/// carries risky syntax, and a pipeline into an interpreter.
fn shell_floors(
    section: &PermissionsSection,
    ast: &ShellCommand,
    command: &str,
) -> Vec<(PermissionLevel, String)> {
    let guard = &section.shell_guard;
    let mut floors = Vec::new();
    if guard.unknown_structure {
        match ast.kind {
            ShellKind::TooComplex => floors.push((
                PermissionLevel::Ask,
                format!("{NAMESPACE}:shell_ast:too_complex:unsupported_structure"),
            )),
            ShellKind::ParseUnavailable if ast.flags.has_risky_structure() => floors.push((
                PermissionLevel::Ask,
                format!("{NAMESPACE}:shell_ast:parse_unavailable:unparsable_structure"),
            )),
            _ => {}
        }
    }
    if guard.interpreter_sink && shell::has_interpreter_sink(command) {
        floors.push((
            PermissionLevel::Ask,
            format!("{NAMESPACE}:shell_guard:interpreter_sink"),
        ));
    }
    floors
}

/// Synthesizes the arguments a subcommand is evaluated with (`{"command": text}`).
fn command_args(command: &str) -> Value {
    let mut map = Map::new();
    map.insert("command".to_owned(), Value::String(command.to_owned()));
    Value::Object(map)
}

/// Strictness rank: `deny` < `ask` < `allow`, smaller is stricter.
fn rank(level: PermissionLevel) -> u8 {
    match level {
        PermissionLevel::Deny => 0,
        PermissionLevel::Ask => 1,
        PermissionLevel::Allow => 2,
    }
}

/// Folds levels with [`rank`], keeping the strictest; an empty fold is ask.
fn strictest_all(levels: impl IntoIterator<Item = PermissionLevel>) -> PermissionLevel {
    let mut strictest: Option<PermissionLevel> = None;
    for level in levels {
        strictest = Some(match strictest {
            Some(current) if rank(current) <= rank(level) => current,
            _ => level,
        });
    }
    strictest.unwrap_or(PermissionLevel::Ask)
}

/// Joins hit references with `+` in rule order, skipping duplicates.
///
/// `level` restricts the references to hits resolving to that level.
fn join_references(hits: &[RuleHit], level: Option<PermissionLevel>) -> String {
    let mut references: Vec<&str> = Vec::new();
    for hit in hits {
        if let Some(wanted) = level {
            if hit.level != wanted {
                continue;
            }
        }
        if !references.contains(&hit.reference.as_str()) {
            references.push(hit.reference.as_str());
        }
    }
    references.join("+")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{evaluate_tiered_policy, FALLBACK_NO_CONFIG};
    use crate::permission::builtin;
    use crate::permission::model::{
        ApprovalOverrideEntry, PermissionLevel, PermissionRule, PermissionsSection, RuleMatchType,
    };

    fn level_map(pairs: &[(&str, PermissionLevel)]) -> BTreeMap<String, PermissionLevel> {
        pairs
            .iter()
            .map(|(key, level)| ((*key).to_owned(), *level))
            .collect()
    }

    /// `tools.bash: allow` plus `defaults: {"*": allow}` — the permissive baseline.
    fn allow_section() -> PermissionsSection {
        PermissionsSection {
            defaults: level_map(&[("*", PermissionLevel::Allow)]),
            tools: level_map(&[("bash", PermissionLevel::Allow)]),
            ..PermissionsSection::default()
        }
    }

    fn rule(id: &str, tools: &[&str], pattern: &str, action: PermissionLevel) -> PermissionRule {
        PermissionRule {
            id: id.to_owned(),
            tools: tools.iter().map(|tool| (*tool).to_owned()).collect(),
            pattern: pattern.to_owned(),
            match_type: RuleMatchType::Command,
            action: Some(action),
            ..PermissionRule::default()
        }
    }

    fn override_entry(id: &str, tools: &[&str], pattern: &str) -> ApprovalOverrideEntry {
        ApprovalOverrideEntry {
            id: id.to_owned(),
            tools: tools.iter().map(|tool| (*tool).to_owned()).collect(),
            match_type: RuleMatchType::Command,
            pattern: pattern.to_owned(),
            action: PermissionLevel::Allow,
        }
    }

    // ----- fallback / baselines -------------------------------------------

    #[test]
    fn empty_section_asks_with_the_python_fallback_label() {
        let section = PermissionsSection {
            defaults: BTreeMap::new(),
            ..PermissionsSection::default()
        };
        let (level, label) = evaluate_tiered_policy(&section, "unknown_tool", &json!({}));
        assert_eq!(level, PermissionLevel::Ask);
        assert_eq!(label, FALLBACK_NO_CONFIG);
        assert_eq!(label, "tiered_policy:fallback(no_config)");
    }

    #[test]
    fn tool_baseline_beats_defaults() {
        let section = PermissionsSection {
            defaults: level_map(&[("*", PermissionLevel::Allow)]),
            tools: level_map(&[("bash", PermissionLevel::Allow)]),
            ..PermissionsSection::default()
        };
        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": "ls"}));
        assert_eq!(level, PermissionLevel::Allow);
        assert_eq!(label, "tools.bash");
    }

    #[test]
    fn defaults_star_is_the_last_resort_rung() {
        let section = PermissionsSection {
            defaults: level_map(&[("*", PermissionLevel::Allow)]),
            ..PermissionsSection::default()
        };
        let (level, label) =
            evaluate_tiered_policy(&section, "write_file", &json!({"path": "a.txt"}));
        assert_eq!(level, PermissionLevel::Allow);
        assert_eq!(label, "tiered_policy:defaults.*");
    }

    #[test]
    fn whole_tool_deny_short_circuits_every_rule() {
        let mut section = allow_section();
        section
            .tools
            .insert("bash".to_owned(), PermissionLevel::Deny);
        section.rules = vec![rule(
            "allow_all",
            &["bash"],
            "re:.*",
            PermissionLevel::Allow,
        )];
        section.approval_overrides = vec![override_entry("remember_all", &["bash"], "re:.*")];

        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": "ls"}));
        assert_eq!(level, PermissionLevel::Deny);
        assert_eq!(label, "tools.bash");
    }

    // ----- deny beats the remembered approval -----------------------------

    fn deny_rule_section(with_deny: bool, override_pattern: &str) -> PermissionsSection {
        let mut section = allow_section();
        if with_deny {
            section.rules = vec![rule("deny_rm", &["bash"], "rm *", PermissionLevel::Deny)];
        }
        section.approval_overrides = vec![override_entry(
            "user_allow_bash_command_rm_perm_a_txt",
            &["bash"],
            override_pattern,
        )];
        section
    }

    #[test]
    fn deny_rule_beats_an_approval_override() {
        let section = deny_rule_section(true, "rm perm_a.txt");
        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "rm perm_a.txt"}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(label.contains("deny_rm"), "{label}");
        assert!(!label.contains("approval_overrides"), "{label}");
    }

    #[test]
    fn approval_override_allows_when_no_deny_remains() {
        let section = deny_rule_section(false, "rm perm_a.txt");
        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "rm perm_a.txt"}));
        assert_eq!(level, PermissionLevel::Allow);
        assert!(label.contains("approval_overrides"), "{label}");
        assert!(
            label.contains("user_allow_bash_command_rm_perm_a_txt"),
            "{label}"
        );
    }

    #[test]
    fn deny_rule_still_denies_another_target() {
        let section = deny_rule_section(true, "rm perm_a.txt");
        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "rm other.txt"}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(label.contains("deny_rm"), "{label}");
    }

    #[test]
    fn compound_remembrance_cannot_relax_a_subcommand_deny() {
        let command = "rm perm_a.txt && echo done";
        let section = deny_rule_section(true, command);
        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(label.contains("deny_rm"), "{label}");
        assert!(!label.contains("approval_overrides"), "{label}");
    }

    // ----- full-command remembrance relaxes an ASK -------------------------

    fn override_allow_section(patterns: &[&str]) -> PermissionsSection {
        let mut section = allow_section();
        section.approval_overrides = patterns
            .iter()
            .map(|pattern| override_entry("remembered", &["bash"], pattern))
            .collect();
        section
    }

    #[test]
    fn remembered_command_relaxes_the_interpreter_sink_floor() {
        let command = "cat f | sh";
        let (level, label) =
            evaluate_tiered_policy(&allow_section(), "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Ask);
        assert!(label.contains("interpreter_sink"), "{label}");

        let section = override_allow_section(&[command]);
        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Allow);
        assert!(label.contains("approval_overrides"), "{label}");

        let mut off = allow_section();
        off.shell_guard.interpreter_sink = false;
        let (level, _) = evaluate_tiered_policy(&off, "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Allow);
    }

    #[test]
    fn remembered_command_relaxes_the_structure_floor() {
        let command = "echo $(id)";
        let (level, _) =
            evaluate_tiered_policy(&allow_section(), "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Ask);

        let section = override_allow_section(&[command]);
        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Allow);
        assert!(label.contains("approval_overrides"), "{label}");
    }

    #[test]
    fn remembered_command_never_lifts_a_deny_rule() {
        let command = "cat f | sh";
        let mut section = override_allow_section(&[command]);
        section.rules = vec![rule("deny_cat", &["bash"], "cat *", PermissionLevel::Deny)];

        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(label.contains("deny_cat"), "{label}");
        assert!(!label.contains("approval_overrides"), "{label}");
    }

    #[test]
    fn whole_command_remembrance_never_lifts_a_subcommand_deny() {
        let command = "rm perm_a.txt && echo done";
        let mut section = override_allow_section(&[command]);
        section.rules = vec![rule("deny_rm", &["bash"], "rm *", PermissionLevel::Deny)];

        let (level, label) = evaluate_tiered_policy(&section, "bash", &json!({"command": command}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(label.contains("deny_rm"), "{label}");
        assert!(!label.contains("approval_overrides"), "{label}");
    }

    // ----- packaged builtin rules -----------------------------------------

    #[test]
    fn packaged_builtin_deny_wins_over_defaults_allow() {
        let mut section = allow_section();
        section.rules = builtin::package_command_rules();

        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "shutdown -h now"}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(
            label.starts_with("tiered_policy:builtin:deny:"),
            "the namespace appears twice, as in Python: {label}"
        );
        assert!(label.contains("shell_system_shutdown_or_reboot"), "{label}");

        let (level, _) = evaluate_tiered_policy(&section, "bash", &json!({"command": "echo hi"}));
        assert_eq!(level, PermissionLevel::Allow);
    }

    #[test]
    fn packaged_builtin_ask_is_reported_with_the_doubled_namespace() {
        let mut section = allow_section();
        section.rules = builtin::package_command_rules();

        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "rm -rf /tmp/x"}));
        assert_eq!(level, PermissionLevel::Ask);
        assert!(
            label.starts_with("tiered_policy:builtin:builtin["),
            "{label}"
        );
    }

    // ----- shell floors ----------------------------------------------------

    #[test]
    fn interpreter_sink_asks() {
        let section = allow_section();
        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "cat f | sh"}));
        assert_eq!(level, PermissionLevel::Ask);
        assert!(label.contains("interpreter_sink"), "{label}");

        let mut off = allow_section();
        off.shell_guard.interpreter_sink = false;
        let (level, _) = evaluate_tiered_policy(&off, "bash", &json!({"command": "cat f | sh"}));
        assert_eq!(level, PermissionLevel::Allow);
    }

    #[test]
    fn unknown_structure_asks() {
        let section = allow_section();
        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "echo $(id)"}));
        assert_eq!(level, PermissionLevel::Ask);
        assert!(label.contains("shell_ast"), "{label}");
        assert!(label.contains("too_complex"), "{label}");

        let mut off = allow_section();
        off.shell_guard.unknown_structure = false;
        let (level, _) = evaluate_tiered_policy(&off, "bash", &json!({"command": "echo $(id)"}));
        assert_eq!(level, PermissionLevel::Allow);
    }

    #[test]
    fn floors_never_relax_a_stricter_decision() {
        let mut section = allow_section();
        section.rules = builtin::package_command_rules();
        let (level, label) = evaluate_tiered_policy(
            &section,
            "bash",
            &json!({"command": "shutdown -h now | sh"}),
        );
        assert_eq!(level, PermissionLevel::Deny);
        assert!(!label.contains("interpreter_sink"), "{label}");
    }

    #[test]
    fn launcher_wrapper_does_not_hide_the_command() {
        let mut section = allow_section();
        section.rules = vec![rule(
            "shell_ask_rm",
            &["bash"],
            "rm *",
            PermissionLevel::Ask,
        )];

        let (level, label) = evaluate_tiered_policy(
            &section,
            "bash",
            &json!({"command": "bash -c \"rm -v /tmp/target\""}),
        );
        assert_eq!(level, PermissionLevel::Ask);
        assert!(label.contains("shell_ask_rm"), "{label}");
    }

    // ----- shell tool categories ------------------------------------------

    #[test]
    fn configured_shell_category_drives_rule_matching() {
        let mut section = PermissionsSection {
            defaults: level_map(&[("*", PermissionLevel::Allow)]),
            rules: builtin::package_command_rules(),
            ..PermissionsSection::default()
        };
        section.categories.insert(
            "shell".to_owned(),
            vec!["run_cmd".to_owned(), String::new()],
        );

        let (level, label) =
            evaluate_tiered_policy(&section, "run_cmd", &json!({"command": "rm -rf /tmp/x"}));
        assert_eq!(level, PermissionLevel::Ask);
        assert!(label.contains("builtin["), "{label}");
        assert!(!label.contains("defaults"), "{label}");

        let (level, label) =
            evaluate_tiered_policy(&section, "bash", &json!({"command": "rm -rf /tmp/x"}));
        assert_eq!(level, PermissionLevel::Allow);
        assert_eq!(label, "tiered_policy:defaults.*");
    }

    // ----- severity fallback ----------------------------------------------

    fn severity_section(mode: Option<&str>) -> PermissionsSection {
        let mut section = allow_section();
        section.permission_mode = mode.map(str::to_owned);
        section.rules = vec![PermissionRule {
            id: "critical_rule".to_owned(),
            tools: vec!["bash".to_owned()],
            pattern: "re:.*".to_owned(),
            action: None,
            severity: Some("CRITICAL".to_owned()),
            ..PermissionRule::default()
        }];
        section
    }

    #[test]
    fn severity_action_depends_on_the_permission_mode() {
        let strict = severity_section(Some("strict"));
        let (level, label) = evaluate_tiered_policy(&strict, "bash", &json!({"command": "ls"}));
        assert_eq!(level, PermissionLevel::Deny);
        assert!(label.contains("rules[critical_rule]"), "{label}");

        let normal = severity_section(None);
        let (level, label) = evaluate_tiered_policy(&normal, "bash", &json!({"command": "ls"}));
        assert_eq!(level, PermissionLevel::Ask);
        assert!(label.contains("rules[critical_rule]"), "{label}");
    }
}
