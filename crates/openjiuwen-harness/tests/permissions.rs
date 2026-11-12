//! Permission integration suite: the strictest-wins precedence fold, session
//! scoped approvals, the human-in-the-loop interrupt rail, and confirmation
//! payload parsing.
//!
//! Everything here is offline and deterministic: the engine is pure data, the
//! rail is driven directly through its kernel hook, and the only filesystem
//! handle is a `tempfile` workspace used as a path-guard root.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent::interrupt::InterruptKind;
use openjiuwen_agent::rails::{AgentRail, RailContext, RailEvent};
use openjiuwen_foundation::ToolCall;
use openjiuwen_harness::permission::engine::{PermissionEngine, PermissionEngineConfig};
use openjiuwen_harness::permission::interrupt::{PermissionInterruptRail, DENIED_PREFIX};
use openjiuwen_harness::permission::model::{
    FileGuardDefaults, FileGuardSection, PermissionConfirmResponse, PermissionLevel,
    PermissionRule, PermissionsSection, RuleMatchType,
};
use openjiuwen_harness::permission::session::SessionPermissionStore;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Section with the packaged rules off and `defaults["*"] = default`.
///
/// `package_builtin_rules: false` keeps the packaged command table (which denies
/// `shutdown`, asks on `rm -rf`, …) out of these assertions.
fn base_section(default: PermissionLevel) -> PermissionsSection {
    PermissionsSection {
        package_builtin_rules: false,
        defaults: BTreeMap::from([("*".to_owned(), default)]),
        ..PermissionsSection::default()
    }
}

/// Section with no `defaults` at all, so the ladder reaches its last rung.
fn section_without_defaults() -> PermissionsSection {
    PermissionsSection {
        package_builtin_rules: false,
        defaults: BTreeMap::new(),
        ..PermissionsSection::default()
    }
}

/// Replaces the per-tool baselines of `section`.
fn with_tools(
    mut section: PermissionsSection,
    tools: &[(&str, PermissionLevel)],
) -> PermissionsSection {
    section.tools = tools
        .iter()
        .map(|(name, level)| ((*name).to_owned(), *level))
        .collect();
    section
}

/// Replaces the parameter rules of `section`.
fn with_rules(mut section: PermissionsSection, rules: Vec<PermissionRule>) -> PermissionsSection {
    section.rules = rules;
    section
}

/// One command rule (`match_type: command`) for `section`'s `rules` list.
fn command_rule(id: &str, tool: &str, pattern: &str, action: PermissionLevel) -> PermissionRule {
    PermissionRule {
        id: id.to_owned(),
        tools: vec![tool.to_owned()],
        pattern: pattern.to_owned(),
        match_type: RuleMatchType::Command,
        action: Some(action),
        ..PermissionRule::default()
    }
}

/// Builds the engine façade over `section` (no workspace root, no guards).
fn engine(section: PermissionsSection) -> Arc<PermissionEngine> {
    Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)))
}

/// A `bash` tool call carrying `command` as its JSON arguments.
fn bash(command: &str) -> ToolCall {
    ToolCall {
        id: Some("call_1".to_owned()),
        name: "bash".to_owned(),
        arguments: json!({ "command": command }).to_string(),
        ..ToolCall::default()
    }
}

/// Minimal `BeforeToolCall` context carrying one tool call.
fn context_for(call: ToolCall) -> RailContext {
    let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
    ctx.tool_call = Some(call);
    ctx
}

/// Fires the rail's kernel `before_tool_call` hook, asserting it never errors.
async fn fire(rail: &PermissionInterruptRail, ctx: &mut RailContext) {
    rail.before_tool_call(ctx).await.expect("rail hook");
}

// ---------------------------------------------------------------------------
// 1. engine precedence and the strictest fold
// ---------------------------------------------------------------------------

#[test]
fn tool_baseline_allow_beats_the_ask_default() {
    // Rung 6 (`tools.<tool>`) decides before rung 7 (`defaults["*"]`).
    let engine = engine(with_tools(
        base_section(PermissionLevel::Ask),
        &[("read_file", PermissionLevel::Allow)],
    ));

    let result = engine.check_permission("read_file", &json!({ "file_path": "a.txt" }));
    assert_eq!(result.permission, PermissionLevel::Allow);
    assert!(result.is_allowed());
    assert_eq!(result.matched_rule.as_deref(), Some("tools.read_file"));
    assert_eq!(
        result.reason.as_deref(),
        Some("Allowed by rule: tools.read_file")
    );
}

#[test]
fn explicit_deny_rule_beats_default_and_tool_allow() {
    // The ladder is ordered strictest-first: a user `deny` hit (rung 3) is
    // returned before the `tools.<tool>` baseline (rung 6) and `defaults["*"]`
    // (rung 7), so widening the weaker layers cannot lift the denial.
    let section = with_rules(
        with_tools(
            base_section(PermissionLevel::Allow),
            &[("bash", PermissionLevel::Allow)],
        ),
        vec![command_rule(
            "deny_rm",
            "bash",
            "rm -rf *",
            PermissionLevel::Deny,
        )],
    );
    let engine = engine(section);

    let result = engine.check_permission("bash", &json!({ "command": "rm -rf /tmp/x" }));
    assert_eq!(result.permission, PermissionLevel::Deny);
    assert!(result.is_denied());
    let rule = result
        .matched_rule
        .as_deref()
        .expect("a deny carries a matched rule");
    assert_eq!(rule, "tiered_policy:rules:deny:rules[deny_rm]");
    assert_eq!(
        result.reason.as_deref(),
        Some("Denied by rule: tiered_policy:rules:deny:rules[deny_rm]")
    );
}

#[test]
fn explicit_allow_rule_matches_a_command() {
    // Rung 5: a user allow hit beats the `defaults["*"]` rung below it.
    let section = with_rules(
        base_section(PermissionLevel::Ask),
        vec![command_rule(
            "allow_echo",
            "bash",
            "echo *",
            PermissionLevel::Allow,
        )],
    );
    let engine = engine(section);

    let result = engine.check_permission("bash", &json!({ "command": "echo hi" }));
    assert_eq!(result.permission, PermissionLevel::Allow);
    assert_eq!(
        result.matched_rule.as_deref(),
        Some("tiered_policy:rules:rules[allow_echo]")
    );

    // The rule is anchored to its pattern: another command falls back to the ask
    // default instead of inheriting the allow.
    let other = engine.check_permission("bash", &json!({ "command": "ls" }));
    assert_eq!(other.permission, PermissionLevel::Ask);
    assert_eq!(
        other.matched_rule.as_deref(),
        Some("tiered_policy:defaults.*")
    );
}

#[test]
fn deny_rule_outranks_an_allow_rule_for_the_same_command() {
    // Both rules hit the same command; `finalize_deny` runs before the
    // any-level rung, so deny wins and only the deny rule is labelled.
    let section = with_rules(
        base_section(PermissionLevel::Ask),
        vec![
            command_rule("allow_rm", "bash", "rm -rf *", PermissionLevel::Allow),
            command_rule("deny_rm", "bash", "rm -rf *", PermissionLevel::Deny),
        ],
    );
    let engine = engine(section);

    let result = engine.check_permission("bash", &json!({ "command": "rm -rf /tmp/x" }));
    assert_eq!(result.permission, PermissionLevel::Deny);
    let rule = result
        .matched_rule
        .as_deref()
        .expect("a deny carries a matched rule");
    assert!(rule.contains("rules[deny_rm]"), "{rule}");
    assert!(
        !rule.contains("rules[allow_rm]"),
        "an allow must not be credited for a denied call: {rule}"
    );
}

#[test]
fn unknown_tool_asks_via_the_star_default() {
    let engine = engine(base_section(PermissionLevel::Ask));

    let result = engine.check_permission("mystery_tool", &json!({}));
    assert_eq!(result.permission, PermissionLevel::Ask);
    assert!(result.needs_approval());
    assert!(!result.is_allowed());
    assert_eq!(
        result.matched_rule.as_deref(),
        Some("tiered_policy:defaults.*")
    );
}

#[test]
fn unknown_tool_without_defaults_asks_never_allows() {
    // No rule, no baseline, no `defaults["*"]`: the ladder reports "no opinion"
    // and the engine folds that to ask with the Python `default` label.
    let engine = engine(section_without_defaults());

    let result = engine.check_permission("mystery_tool", &json!({}));
    assert_eq!(result.permission, PermissionLevel::Ask);
    assert!(result.needs_approval());
    assert_eq!(result.matched_rule.as_deref(), Some("default"));
    assert_eq!(
        result.reason.as_deref(),
        Some("Approval required for mystery_tool (rule: default)")
    );
}

#[test]
fn file_guard_deny_folds_over_a_tiered_allow() {
    // Cross-pipeline precedence: pipeline A allows the call, pipeline B denies
    // the write, and `strictest` keeps the denial.
    let temp = tempfile::tempdir().expect("temp workspace");
    let mut section = base_section(PermissionLevel::Allow);
    section.file_guard = FileGuardSection {
        enabled: true,
        defaults: FileGuardDefaults {
            read: PermissionLevel::Allow,
            write: PermissionLevel::Deny,
            exec: PermissionLevel::Ask,
        },
        ..FileGuardSection::default()
    };
    let engine = Arc::new(PermissionEngine::new(
        PermissionEngineConfig::new(section).with_workspace_root(temp.path()),
    ));
    let path = temp.path().join("a.txt").to_string_lossy().into_owned();

    let write = engine.check_permission("write_file", &json!({ "file_path": path }));
    assert_eq!(write.permission, PermissionLevel::Deny);
    let rule = write.matched_rule.as_deref().expect("deny label");
    assert!(rule.contains("file_guard"), "{rule}");

    let read = engine.check_permission("read_file", &json!({ "file_path": path }));
    assert_eq!(read.permission, PermissionLevel::Allow);
    assert!(
        !read
            .matched_rule
            .as_deref()
            .unwrap_or_default()
            .contains("file_guard"),
        "the guard had no opinion on a read"
    );
}

// ---------------------------------------------------------------------------
// 2. session-scoped approvals
// ---------------------------------------------------------------------------

#[test]
fn session_store_memoizes_and_clears() {
    let store = SessionPermissionStore::new();
    assert!(!store.is_auto_confirmed("bash"));
    assert!(store.is_empty());

    store.store_auto_confirm("bash");
    assert!(store.is_auto_confirmed("bash"));
    assert_eq!(store.len(), 1);

    // An empty key is never remembered.
    store.store_auto_confirm("");
    assert_eq!(store.len(), 1);

    store.clear();
    assert!(!store.is_auto_confirmed("bash"));
    assert!(store.is_empty());
}

#[tokio::test]
async fn auto_confirmed_session_key_skips_the_rail_prompt() {
    let engine = engine(base_section(PermissionLevel::Ask));
    let store = Arc::new(SessionPermissionStore::new());
    let rail = PermissionInterruptRail::new(Arc::clone(&engine)).with_store(Arc::clone(&store));
    let call = bash("git status");
    let key = rail.auto_confirm_key(&call);
    assert_eq!(key, "bash:git status");

    // First round: the ask becomes one permission interrupt.
    let mut first = context_for(call.clone());
    fire(&rail, &mut first).await;
    assert!(first.take_skip_tool().is_none());
    assert_eq!(first.interrupts.len(), 1);
    assert_eq!(first.interrupts[0].kind, InterruptKind::Permission);
    assert_eq!(first.interrupts[0].tool_call_id, "call_1");

    // A remembered session decision short-circuits the next identical call.
    store.store_auto_confirm(&key);
    let mut second = context_for(call);
    fire(&rail, &mut second).await;
    assert!(second.take_skip_tool().is_none());
    assert!(
        second.interrupts.is_empty(),
        "an auto-confirmed call must not ask again"
    );
}

// ---------------------------------------------------------------------------
// 3. the interrupt rail's deny path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn denied_call_is_skipped_by_the_rail() {
    let section = with_rules(
        base_section(PermissionLevel::Allow),
        vec![command_rule(
            "deny_rm",
            "bash",
            "rm -rf *",
            PermissionLevel::Deny,
        )],
    );
    let engine = engine(section);
    let rail = PermissionInterruptRail::new(Arc::clone(&engine));

    // The engine's own verdict is the rail's skip text.
    let verdict = engine.check_permission("bash", &json!({ "command": "rm -rf /tmp/x" }));
    assert!(verdict.is_denied());

    let mut ctx = context_for(bash("rm -rf /tmp/x"));
    fire(&rail, &mut ctx).await;
    assert!(
        ctx.interrupts.is_empty(),
        "a denied call is never surfaced as a question"
    );
    let skip = ctx
        .take_skip_tool()
        .expect("deny substitutes a tool result");
    assert_eq!(
        skip,
        format!(
            "{DENIED_PREFIX} {}",
            verdict.reason.as_deref().expect("deny reason")
        )
    );
    assert!(skip.starts_with(DENIED_PREFIX), "{skip}");
    assert!(skip.contains("rules[deny_rm]"), "{skip}");
}

// ---------------------------------------------------------------------------
// 4. confirmation payload parsing
// ---------------------------------------------------------------------------

#[test]
fn confirm_response_parses_object_shapes() {
    let approved = PermissionConfirmResponse::parse(&json!({ "approved": true }))
        .expect("an approved object parses");
    assert!(approved.approved);
    assert_eq!(approved.feedback, "");
    assert!(!approved.auto_confirm);
    assert_eq!(approved.persist_allow, None);
    assert!(
        !approved.wants_permanent_persist(),
        "allow-once must not persist"
    );
    assert!(!approved.wants_session_persist());

    let rejected =
        PermissionConfirmResponse::parse(&json!({ "approved": false, "feedback": "too risky" }))
            .expect("a rejected object parses");
    assert!(!rejected.approved);
    assert_eq!(rejected.feedback, "too risky");
    assert!(!rejected.wants_permanent_persist());
    assert!(!rejected.wants_session_persist());

    let session = PermissionConfirmResponse::parse(
        &json!({ "approved": true, "auto_confirm": true, "persist_allow": false }),
    )
    .expect("a session-remember answer parses");
    assert!(session.wants_session_persist());
    assert!(!session.wants_permanent_persist());

    let permanent = PermissionConfirmResponse::parse(
        &json!({ "approved": true, "auto_confirm": true, "persist_allow": true }),
    )
    .expect("a permanent-remember answer parses");
    assert!(permanent.wants_permanent_persist());
    assert!(!permanent.wants_session_persist());

    // A legacy host omits `persist_allow`, which means "permanent".
    let legacy =
        PermissionConfirmResponse::parse(&json!({ "approved": true, "auto_confirm": true }))
            .expect("a legacy answer parses");
    assert_eq!(legacy.persist_allow, None);
    assert!(legacy.wants_permanent_persist());

    // Unknown keys are ignored.
    let extra = PermissionConfirmResponse::parse(
        &json!({ "approved": true, "unknown": [1, 2], "note": "hi" }),
    )
    .expect("unknown keys are tolerated");
    assert!(extra.approved);
    assert!(!extra.auto_confirm);
}

#[test]
fn confirm_response_parses_bool_and_string_forms() {
    let bare = PermissionConfirmResponse::parse(&json!(true)).expect("a bool parses");
    assert!(bare.approved);
    assert!(!bare.auto_confirm);

    let bare_no = PermissionConfirmResponse::parse(&json!(false)).expect("a bool parses");
    assert!(!bare_no.approved);

    // A JSON string holding an object is re-entered as that object.
    let nested = PermissionConfirmResponse::parse(&json!(
        "{\"approved\": true, \"auto_confirm\": true, \"persist_allow\": false}"
    ))
    .expect("a nested JSON object parses");
    assert!(nested.approved);
    assert!(nested.auto_confirm);
    assert!(nested.wants_session_persist());

    // Answer words fall back to the kernel's `parse_approval`, which only maps
    // the approved flag; the "always" semantics live in `is_always_approval`.
    let word_yes = PermissionConfirmResponse::parse(&json!("yes")).expect("a word parses");
    assert!(word_yes.approved);
    assert!(!word_yes.auto_confirm);

    let word_no = PermissionConfirmResponse::parse(&json!("deny")).expect("a word parses");
    assert!(!word_no.approved);

    let word_always =
        PermissionConfirmResponse::parse(&json!("allow_always")).expect("a word parses");
    assert!(word_always.approved);
    assert!(!word_always.auto_confirm);
}

#[test]
fn confirm_response_rejects_unusable_payloads() {
    for payload in [
        json!(null),
        json!(7),
        json!([true]),
        json!({}),
        json!({ "feedback": "approved is missing" }),
        json!({ "approved": "yes" }),
        json!("maybe"),
    ] {
        assert!(
            PermissionConfirmResponse::parse(&payload).is_none(),
            "payload must not parse: {payload}"
        );
    }
}

#[test]
fn confirm_response_payload_schema_names_the_flags() {
    let schema: Value = PermissionConfirmResponse::payload_schema();
    assert_eq!(schema["required"], json!(["approved"]));
    assert_eq!(schema["properties"]["approved"]["type"], json!("boolean"));
    assert_eq!(schema["properties"]["feedback"]["type"], json!("string"));
    assert_eq!(
        schema["properties"]["auto_confirm"]["type"],
        json!("boolean")
    );
    assert_eq!(
        schema["properties"]["persist_allow"]["type"],
        json!(["boolean", "null"])
    );
}
