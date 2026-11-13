//! Unit tests for `fork_compact`: fork blueprint derivation and transcript compaction.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

use super::*;
use crate::schema::{MemberSpecBase, TeamRole};

/// Builds a tool spec with a recognizable description and an object schema.
fn tool(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.to_owned(),
        description: format!("{name} tool"),
        input_schema: json!({"type": "object", "properties": {}}),
    }
}

/// Builds a roster entry with a prompt derived from its name.
fn member(name: &str) -> TeamMemberSpec {
    TeamMemberSpec {
        base: MemberSpecBase {
            member_name: name.to_owned(),
            display_name: format!("{name} display"),
            desc: format!("{name} description"),
            prompt: format!("You are {name}."),
            model_name: None,
        },
        role_type: TeamRole::Teammate,
        goal: None,
    }
}

/// The parent everything forks from: a model, a card, a system prompt, three tools and a
/// resolved workspace layout.
fn parent_source() -> ForkSource {
    let spec = DeepAgentSpec {
        model: Some(ModelRef {
            name: None,
            model_name: Some("gpt-4o".to_owned()),
            provider: Some("openai".to_owned()),
            params: BTreeMap::new(),
        }),
        card: Some(AgentCard::new("parent")),
        system_prompt: Some("You are the parent member.".to_owned()),
        tools: vec![tool("bash"), tool("grep"), tool("read_file")],
        workspace: Some(WorkspaceSpec {
            root_path: "/work".to_owned(),
            language: "en".to_owned(),
            stable_base: true,
        }),
        cwd: Some("/work/project".to_owned()),
        project_root: Some("/work/project".to_owned()),
        language: Some("en".to_owned()),
        ..DeepAgentSpec::default()
    };
    ForkSource::new("alpha", "parent", "session-1", member("parent"), spec)
}

fn tool_names(tools: &[ToolSpec]) -> Vec<&str> {
    tools.iter().map(|tool| tool.name.as_str()).collect()
}

// ---------------------------------------------------------------------------------------------
// Fork
// ---------------------------------------------------------------------------------------------

#[test]
fn fork_inherits_context_and_clears_parent_identity() {
    let source = parent_source();
    let child = member("dev-1");
    let forked = fork_member_context(&source, child.clone(), None, &ForkToolPolicy::new()).unwrap();

    assert_eq!(forked.member_name(), "dev-1");
    assert_eq!(forked.session_id, "session-1");
    assert_eq!(forked.parent_team, "alpha");
    assert_eq!(forked.parent_member, "parent");

    // Workspace layout is shared with the parent.
    assert_eq!(
        forked.workspace.as_ref().map(|ws| ws.root_path.as_str()),
        Some("/work")
    );
    assert_eq!(forked.cwd.as_deref(), Some("/work/project"));
    assert_eq!(forked.project_root.as_deref(), Some("/work/project"));
    assert_eq!(forked.spec.workspace, source.spec.workspace);

    // The model and every non-identity field survive; identity does not.
    assert_eq!(forked.spec.model, source.spec.model);
    assert_eq!(forked.spec.model_name(), Some("gpt-4o"));
    assert_eq!(forked.spec.system_prompt, None);
    assert_eq!(forked.spec.card, None);
    assert_eq!(forked.spec.language.as_deref(), Some("en"));
    assert!(forked.spec.enable_task_loop);

    // The child's role comes from its own roster entry, not from the fork source.
    assert_eq!(forked.member, child);
    assert_eq!(forked.member.base.prompt, "You are dev-1.");

    assert_eq!(forked.lineage, vec!["parent".to_owned()]);
    assert_eq!(forked.fork_depth, 1);
    assert!(forked.is_root());
    assert!(forked.task.is_none());
    assert_eq!(
        tool_names(&forked.spec.tools),
        ["bash", "grep", "read_file"]
    );
}

#[test]
fn fork_uses_the_pinned_child_model_and_keeps_the_parent_model_otherwise() {
    let source = parent_source();

    let mut pinned = member("dev-2");
    pinned.base.model_name = Some("gpt-4o-mini".to_owned());
    let forked = fork_member_context(&source, pinned, None, &ForkToolPolicy::new()).unwrap();
    let model = forked.spec.model.expect("a model is pinned");
    assert_eq!(model.model_name.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(model.name, None);
    assert_eq!(model.provider.as_deref(), Some("openai"));
    assert_eq!(model.params, BTreeMap::new());

    // A blank model name counts as absent, so the parent's model is kept.
    let mut blank = member("dev-3");
    blank.base.model_name = Some(String::new());
    let forked = fork_member_context(&source, blank, None, &ForkToolPolicy::new()).unwrap();
    assert_eq!(forked.spec.model, source.spec.model);
}

#[test]
fn fork_tool_policy_denies_restricts_and_appends() {
    let source = parent_source();

    let denied = fork_member_context(
        &source,
        member("dev-1"),
        None,
        &ForkToolPolicy::new().denying("bash"),
    )
    .unwrap();
    assert_eq!(tool_names(&denied.spec.tools), ["grep", "read_file"]);

    // The allow-list keeps the parent's order, not the order of the set.
    let restricted = fork_member_context(
        &source,
        member("dev-1"),
        None,
        &ForkToolPolicy::new().allowing_only(vec!["read_file".to_owned(), "grep".to_owned()]),
    )
    .unwrap();
    assert_eq!(tool_names(&restricted.spec.tools), ["grep", "read_file"]);

    // Denial wins over the allow-list.
    let both = fork_member_context(
        &source,
        member("dev-1"),
        None,
        &ForkToolPolicy::new()
            .allowing_only(vec!["grep".to_owned(), "read_file".to_owned()])
            .denying("grep"),
    )
    .unwrap();
    assert_eq!(tool_names(&both.spec.tools), ["read_file"]);

    // Additive tools append after the inherited ones, and a duplicate name appears once.
    let additive = fork_member_context(
        &source,
        member("dev-1"),
        None,
        &ForkToolPolicy::new()
            .with_tool(tool("web_search"))
            .with_tool(tool("grep"))
            .with_tool(tool("web_search")),
    )
    .unwrap();
    assert_eq!(
        tool_names(&additive.spec.tools),
        ["bash", "grep", "read_file", "web_search"]
    );
    assert_eq!(
        additive.spec.tools[3],
        tool("web_search"),
        "the additive spec is kept verbatim"
    );

    // An explicit denial also applies to the additive list.
    let denied_additive = fork_member_context(
        &source,
        member("dev-1"),
        None,
        &ForkToolPolicy::new()
            .denying("web_search")
            .with_tool(tool("web_search")),
    )
    .unwrap();
    assert_eq!(
        tool_names(&denied_additive.spec.tools),
        ["bash", "grep", "read_file"]
    );
}

#[test]
fn empty_tool_policy_passes_the_parent_tools_through() {
    let source = parent_source();
    let policy = ForkToolPolicy::new();
    assert_eq!(policy, ForkToolPolicy::default());
    assert_eq!(policy.denied_tools, BTreeSet::new());
    assert_eq!(policy.allowed_tools, None);
    assert_eq!(policy.additive_tools, Vec::new());
    assert_eq!(policy.apply(&source.spec.tools), source.spec.tools);
    // An already-empty parent list stays empty.
    assert_eq!(policy.apply(&[]), Vec::new());
}

#[test]
fn fork_rejects_malformed_and_reserved_child_names() {
    let source = parent_source();
    for bad in ["Dev-1", "", "dev 1", "1dev", "dev.1", "-dev", "dev/1"] {
        let error =
            fork_member_context(&source, member(bad), None, &ForkToolPolicy::new()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{bad}");
        assert!(
            error.to_string().contains("invalid member name"),
            "{bad}: {error}"
        );
    }

    for reserved in ["user", "team_leader", "human_agent"] {
        let error = fork_member_context(&source, member(reserved), None, &ForkToolPolicy::new())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .ends_with(&format!("member name '{reserved}' is reserved")),
            "{error}"
        );
    }
}

#[test]
fn fork_rejects_forking_a_member_into_itself() {
    let source = parent_source();
    let error =
        fork_member_context(&source, member("parent"), None, &ForkToolPolicy::new()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("a fork must name a new member"),
        "{error}"
    );
}

#[test]
fn fork_lineage_and_depth_span_a_two_level_chain() {
    let source = parent_source();
    let first =
        fork_member_context(&source, member("child"), None, &ForkToolPolicy::new()).unwrap();
    assert_eq!(first.fork_depth, 1);
    assert_eq!(first.lineage, vec!["parent".to_owned()]);
    assert!(first.is_root());

    let second_source = ForkSource {
        team_name: first.parent_team.clone(),
        member_name: first.member_name().to_owned(),
        session_id: first.session_id.clone(),
        member: first.member.clone(),
        spec: first.spec.clone(),
        lineage: first.lineage.clone(),
    };
    let second = fork_member_context(
        &second_source,
        member("grandchild"),
        Some(ForkTaskOverride::new("audit", "review the parser")),
        &ForkToolPolicy::new(),
    )
    .unwrap();

    assert_eq!(second.fork_depth, 2);
    assert_eq!(
        second.lineage,
        vec!["parent".to_owned(), "child".to_owned()]
    );
    assert!(!second.is_root());
    assert_eq!(second.parent_member, "child");
    assert_eq!(
        second.session_id, "session-1",
        "the session is still the root's"
    );
    assert_eq!(second.spec.system_prompt, None, "identity stays cleared");
    assert_eq!(
        second.task.as_ref().map(|task| task.title.as_str()),
        Some("audit")
    );
}

#[test]
fn forked_member_spec_round_trips_and_exposes_its_json_shape() {
    let source = parent_source();
    let task =
        ForkTaskOverride::new("write docs", "document the fork path").with_parent_task("task-7");
    let forked = fork_member_context(
        &source,
        member("dev-1"),
        Some(task),
        &ForkToolPolicy::new().denying("bash"),
    )
    .unwrap();

    let value = serde_json::to_value(&forked).unwrap();
    assert_eq!(value["parent_team"], json!("alpha"));
    assert_eq!(value["parent_member"], json!("parent"));
    assert_eq!(value["session_id"], json!("session-1"));
    assert_eq!(value["fork_depth"], json!(1));
    assert_eq!(value["lineage"], json!(["parent"]));
    assert_eq!(value["member"]["member_name"], json!("dev-1"));
    assert_eq!(value["member"]["role_type"], json!("teammate"));
    assert_eq!(value["member"]["prompt"], json!("You are dev-1."));
    assert_eq!(value["task"]["title"], json!("write docs"));
    assert_eq!(value["task"]["parent_task_id"], json!("task-7"));
    assert_eq!(value["spec"]["system_prompt"], json!(null));
    assert_eq!(value["spec"]["card"], json!(null));
    assert_eq!(value["spec"]["model"]["model_name"], json!("gpt-4o"));
    assert_eq!(value["workspace"]["root_path"], json!("/work"));

    // Unknown fields are tolerated and do not disturb the parsed value.
    let mut wire = value.clone();
    wire["unknown_python_field"] = json!({"nested": true});
    let reparsed: ForkedMemberSpec = serde_json::from_value(wire).unwrap();
    assert_eq!(reparsed, forked);
    assert_eq!(serde_json::to_value(&reparsed).unwrap(), value);
    assert_eq!(ForkedMemberSpec::default().member_name(), "");
}

#[test]
fn fork_tool_policy_round_trips_through_json() {
    let policy = ForkToolPolicy::new()
        .denying("bash")
        .allowing_only(vec!["grep".to_owned()])
        .with_tool(tool("web_search"));
    let value = serde_json::to_value(&policy).unwrap();
    assert_eq!(value["denied_tools"], json!(["bash"]));
    assert_eq!(value["allowed_tools"], json!(["grep"]));
    assert_eq!(value["additive_tools"][0]["name"], json!("web_search"));

    let reparsed: ForkToolPolicy = serde_json::from_value(value).unwrap();
    assert_eq!(reparsed, policy);
    assert_eq!(
        reparsed.apply(&[tool("bash"), tool("grep"), tool("read_file")]),
        vec![tool("grep"), tool("web_search")]
    );
    assert_eq!(
        serde_json::from_value::<ForkToolPolicy>(json!({"unknown": 1})).unwrap(),
        ForkToolPolicy::new()
    );
    assert!(ForkToolPolicy::new().allowed_tools.is_none());
    assert_eq!(
        ForkToolPolicy::new().denied_tools,
        BTreeSet::<String>::new()
    );
}

#[test]
fn fork_task_override_defaults_and_builders() {
    let task = ForkTaskOverride::new("title", "body");
    assert_eq!(task.title, "title");
    assert_eq!(task.content, "body");
    assert_eq!(task.parent_task_id, None);
    assert_eq!(task.metadata, BTreeMap::new());
    assert_eq!(
        task.clone()
            .with_parent_task("task-1")
            .parent_task_id
            .as_deref(),
        Some("task-1")
    );
    assert_eq!(ForkTaskOverride::default().title, "");
    let wire = serde_json::to_value(&task).unwrap();
    assert_eq!(wire["metadata"], json!({}));
    assert_eq!(
        serde_json::from_value::<ForkTaskOverride>(wire).unwrap(),
        task
    );
}

// ---------------------------------------------------------------------------------------------
// Compact
// ---------------------------------------------------------------------------------------------

/// Twelve non-system messages: 3 user, 5 assistant, 4 tool, in that order.
fn twelve_messages() -> Vec<TranscriptMessage> {
    let mut messages = Vec::new();
    for index in 0..3 {
        messages.push(TranscriptMessage::user(format!("user message {index}")));
    }
    for index in 0..5 {
        messages.push(TranscriptMessage::assistant(format!(
            "assistant message {index}"
        )));
    }
    for index in 0..4 {
        messages.push(TranscriptMessage::tool(
            "bash",
            format!("tool result {index}"),
        ));
    }
    messages
}

#[test]
fn compaction_keeps_system_messages_at_the_front() {
    let mut messages = vec![
        TranscriptMessage::system("You are the parent."),
        TranscriptMessage::system("Tools run in /work."),
    ];
    messages.extend(twelve_messages());

    let result = compact_transcript(&messages, &CompactPolicy::new().keeping_last(2));
    assert_eq!(result.messages[0].role, CompactRole::System);
    assert_eq!(result.messages[0].content, "You are the parent.");
    assert_eq!(result.messages[1].content, "Tools run in /work.");
    assert_eq!(result.messages[2].role, CompactRole::System);
    assert_eq!(
        result.summary.as_deref(),
        Some("[compacted] 10 earlier messages dropped (3 user, 5 assistant, 2 tool)")
    );
    assert_eq!(
        Some(result.messages[2].content.as_str()),
        result.summary.as_deref()
    );
    assert_eq!(result.messages.len(), 5);
    assert_eq!(result.messages[3].content, "tool result 2");
    assert_eq!(result.messages[4].content, "tool result 3");
    assert_eq!(result.dropped_messages, 10);
    assert_eq!(result.truncated_tool_results, 0);
    assert!(!result.is_lossless());
}

#[test]
fn compaction_reports_the_exact_summary_line() {
    let messages = twelve_messages();
    let result = compact_transcript(&messages, &CompactPolicy::new().keeping_last(0));

    assert_eq!(
        result.summary.as_deref(),
        Some("[compacted] 12 earlier messages dropped (3 user, 5 assistant, 4 tool)")
    );
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].role, CompactRole::System);
    assert_eq!(
        Some(result.messages[0].content.as_str()),
        result.summary.as_deref()
    );
    assert_eq!(result.dropped_messages, 12);
    assert!(result.compacted_bytes <= result.original_bytes);

    // One dropped message renders the singular form, with zero categories omitted.
    let single = compact_transcript(
        &[TranscriptMessage::user(
            "please read the whole parser module and report every deviation you find in it",
        )],
        &CompactPolicy::new().keeping_last(0),
    );
    assert_eq!(
        single.summary.as_deref(),
        Some("[compacted] 1 earlier message dropped (1 user)")
    );
}

#[test]
fn compaction_without_system_messages_drops_and_counts_them() {
    let mut messages = vec![TranscriptMessage::system("You are the parent.")];
    messages.extend(twelve_messages());

    let policy = CompactPolicy::new()
        .keeping_last(0)
        .without_system_messages();
    let result = compact_transcript(&messages, &policy);
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].role, CompactRole::System);
    assert_eq!(
        result.summary.as_deref(),
        Some("[compacted] 12 earlier messages dropped (3 user, 5 assistant, 4 tool)")
    );
    assert_eq!(result.dropped_messages, 13, "the system message counts too");
    assert!(!result.is_lossless());
}

#[test]
fn compaction_keeps_only_the_summary_when_the_tail_budget_is_zero() {
    let mut messages = vec![TranscriptMessage::system("You are the parent.")];
    messages.extend(twelve_messages());

    let result = compact_transcript(&messages, &CompactPolicy::new().keeping_last(0));
    assert_eq!(result.messages.len(), 2);
    assert_eq!(result.messages[0].content, "You are the parent.");
    assert_eq!(
        Some(result.messages[1].content.as_str()),
        result.summary.as_deref()
    );
    assert_eq!(result.dropped_messages, 12);
}

#[test]
fn default_policy_keeps_the_last_six_non_system_messages() {
    let mut messages = vec![TranscriptMessage::system("system prompt")];
    for index in 0..8 {
        messages.push(TranscriptMessage::user(format!(
            "message {index}: please inspect the parser module and report its behaviour in detail"
        )));
    }

    let result = compact_transcript(&messages, &CompactPolicy::new());
    assert_eq!(result.dropped_messages, 2);
    assert_eq!(result.messages.len(), 8, "system + summary + six kept");
    assert_eq!(result.messages[0].content, "system prompt");
    assert_eq!(
        result.summary.as_deref(),
        Some("[compacted] 2 earlier messages dropped (2 user)")
    );
    let kept: Vec<&str> = result.messages[2..]
        .iter()
        .map(|message| message.content.split(':').next().unwrap_or_default())
        .collect();
    assert_eq!(
        kept,
        [
            "message 2",
            "message 3",
            "message 4",
            "message 5",
            "message 6",
            "message 7"
        ]
    );
}

#[test]
fn long_tool_results_are_truncated_on_a_char_boundary() {
    // `a` shifts the two-byte `é` run so byte 2048 falls inside a character.
    let content = format!("a{}", "é".repeat(3_000));
    assert_eq!(content.len(), 6_001);

    let messages = vec![
        TranscriptMessage::user("run the build"),
        TranscriptMessage::assistant("running it"),
        TranscriptMessage::tool("bash", content.clone()),
    ];
    let policy = CompactPolicy::new();
    let result = compact_transcript(&messages, &policy);

    assert_eq!(result.truncated_tool_results, 1);
    let truncated = result
        .messages
        .iter()
        .find(|message| message.role == CompactRole::Tool)
        .unwrap();
    assert!(truncated.content.ends_with(&policy.truncated_suffix));
    assert!(
        truncated.content.len() <= policy.max_tool_result_bytes + policy.truncated_suffix.len()
    );
    assert!(truncated.content.len() < content.len());
    assert!(std::str::from_utf8(truncated.content.as_bytes()).is_ok());
    assert!(!truncated.content.contains('\u{fffd}'));

    let prefix_len = truncated.content.len() - policy.truncated_suffix.len();
    assert_eq!(prefix_len, 2_047, "the cut backs off to a char boundary");
    assert!(content.starts_with(&truncated.content[..prefix_len]));
    assert_eq!(truncated.content.chars().count(), 1 + 1_023 + 13);
    assert!(!result.is_lossless());
}

#[test]
fn only_tool_results_are_truncated() {
    let long_user = "x".repeat(5_000);
    let long_system = "y".repeat(5_000);
    let messages = vec![
        TranscriptMessage::system(long_system.clone()),
        TranscriptMessage::user(long_user.clone()),
        TranscriptMessage::tool("bash", "short result"),
    ];
    let result = compact_transcript(&messages, &CompactPolicy::new().with_tool_budget(16));

    assert_eq!(result.truncated_tool_results, 0, "the budget is tool-only");
    assert_eq!(result.messages[0].content, long_system);
    assert_eq!(result.messages[1].content, long_user);
    // A short tool result is untouched, and nothing was dropped.
    assert_eq!(result.messages[2].content, "short result");
    assert!(result.is_lossless());
    assert_eq!(result.compacted_bytes, result.original_bytes);
}

#[test]
fn short_transcripts_are_lossless_and_a_dropped_one_is_not() {
    let short = vec![
        TranscriptMessage::system("You are the parent."),
        TranscriptMessage::user("read the file"),
        TranscriptMessage::assistant("read it"),
        TranscriptMessage::tool("bash", "output"),
    ];
    let result = compact_transcript(&short, &CompactPolicy::new());
    assert_eq!(result.summary, None);
    assert_eq!(result.dropped_messages, 0);
    assert_eq!(result.truncated_tool_results, 0);
    assert!(result.is_lossless());
    assert_eq!(result.messages, short);
    assert_eq!(result.compacted_bytes, result.original_bytes);

    let dropped = compact_transcript(&twelve_messages(), &CompactPolicy::new().keeping_last(6));
    assert_eq!(dropped.dropped_messages, 6);
    assert!(!dropped.is_lossless());
}

#[test]
fn compaction_never_grows_a_transcript() {
    let long_tool = TranscriptMessage::tool("bash", "é".repeat(4_000));
    let cases: Vec<(Vec<TranscriptMessage>, CompactPolicy)> = vec![
        (Vec::new(), CompactPolicy::new()),
        (
            vec![TranscriptMessage::user("hi")],
            CompactPolicy::new().keeping_last(0),
        ),
        (twelve_messages(), CompactPolicy::new().keeping_last(0)),
        (twelve_messages(), CompactPolicy::new()),
        (vec![long_tool], CompactPolicy::new().with_tool_budget(0)),
        (
            twelve_messages(),
            CompactPolicy::new()
                .keeping_last(1)
                .without_system_messages(),
        ),
    ];
    for (messages, policy) in cases {
        let result = compact_transcript(&messages, &policy);
        assert!(
            result.compacted_bytes <= result.original_bytes,
            "{policy:?} produced {} bytes from {}",
            result.compacted_bytes,
            result.original_bytes
        );
    }

    // A summary that would cost more than the single message it replaces is omitted, but the
    // message is still dropped and counted.
    let tiny = compact_transcript(
        &[TranscriptMessage::user("hi")],
        &CompactPolicy::new().keeping_last(0),
    );
    assert_eq!(tiny.summary, None);
    assert_eq!(tiny.messages, Vec::new());
    assert_eq!(tiny.dropped_messages, 1);
    assert_eq!(tiny.compacted_bytes, 0);
    assert!(!tiny.is_lossless());
}

#[test]
fn compaction_is_idempotent() {
    let policy = CompactPolicy::new().keeping_last(3);
    let first = compact_transcript(&twelve_messages(), &policy);
    assert_eq!(first.dropped_messages, 9);
    assert!(first.summary.is_some());

    let second = compact_transcript(&first.messages, &policy);
    assert_eq!(second.messages, first.messages);
    assert_eq!(second.compacted_bytes, first.compacted_bytes);
    assert_eq!(second.summary, None, "no second summary");
    assert_eq!(second.dropped_messages, 0);
    assert_eq!(second.truncated_tool_results, 0);
    assert_eq!(
        first
            .messages
            .iter()
            .filter(|message| message.content.starts_with(SUMMARY_PREFIX))
            .count(),
        1
    );
}

#[test]
fn empty_transcripts_stay_empty() {
    let result = compact_transcript(&[], &CompactPolicy::new());
    assert_eq!(result.messages, Vec::new());
    assert_eq!(result.summary, None);
    assert_eq!(result.dropped_messages, 0);
    assert_eq!(result.truncated_tool_results, 0);
    assert_eq!(result.original_bytes, 0);
    assert_eq!(result.compacted_bytes, 0);
    assert!(result.is_lossless());
    assert_eq!(result.text(), "");
    assert_eq!(CompactedTranscript::default(), result);
}

#[test]
fn compacted_transcript_and_roles_round_trip_through_json() {
    assert_eq!(
        serde_json::to_value(CompactRole::System).unwrap(),
        json!("system")
    );
    assert_eq!(
        serde_json::to_value(CompactRole::User).unwrap(),
        json!("user")
    );
    assert_eq!(
        serde_json::to_value(CompactRole::Assistant).unwrap(),
        json!("assistant")
    );
    assert_eq!(
        serde_json::to_value(CompactRole::Tool).unwrap(),
        json!("tool")
    );
    assert_eq!(CompactRole::default(), CompactRole::System);

    let message = TranscriptMessage::tool("bash", "out");
    let wire = serde_json::to_value(&message).unwrap();
    assert_eq!(
        wire,
        json!({"role": "tool", "content": "out", "tool_name": "bash", "tool_call_id": null})
    );
    assert_eq!(
        serde_json::from_value::<TranscriptMessage>(wire).unwrap(),
        message
    );

    let mut noisy = serde_json::to_value(&twelve_messages()[0]).unwrap();
    noisy["unknown_python_field"] = json!(7);
    assert_eq!(
        serde_json::from_value::<TranscriptMessage>(noisy).unwrap(),
        twelve_messages()[0]
    );

    let result = compact_transcript(&twelve_messages(), &CompactPolicy::new().keeping_last(2));
    let value = serde_json::to_value(&result).unwrap();
    assert_eq!(value["dropped_messages"], json!(10));
    assert_eq!(value["messages"][0]["role"], json!("system"));
    let reparsed: CompactedTranscript = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(reparsed, result);
    assert_eq!(serde_json::to_value(&reparsed).unwrap(), value);
}

#[test]
fn compact_policy_builders_and_json_shape() {
    let policy = CompactPolicy::new()
        .keeping_last(2)
        .with_tool_budget(16)
        .without_system_messages();
    assert!(!policy.keep_system);
    assert_eq!(policy.keep_last_messages, 2);
    assert_eq!(policy.max_tool_result_bytes, 16);
    assert_eq!(policy.truncated_suffix, DEFAULT_TRUNCATED_SUFFIX);
    assert_eq!(CompactPolicy::default(), CompactPolicy::new());
    assert_eq!(
        CompactPolicy::default().keep_last_messages,
        DEFAULT_KEEP_LAST_MESSAGES
    );
    assert_eq!(
        CompactPolicy::default().max_tool_result_bytes,
        DEFAULT_MAX_TOOL_RESULT_BYTES
    );

    let value = serde_json::to_value(&policy).unwrap();
    assert_eq!(
        value,
        json!({
            "keep_system": false,
            "keep_last_messages": 2,
            "max_tool_result_bytes": 16,
            "truncated_suffix": "… [truncated]",
        })
    );
    assert_eq!(
        serde_json::from_value::<CompactPolicy>(value).unwrap(),
        policy
    );
    assert_eq!(
        serde_json::from_value::<CompactPolicy>(json!({"unknown": 1})).unwrap(),
        CompactPolicy::new()
    );
}

#[test]
fn transcript_byte_len_weighs_content_and_labels() {
    assert_eq!(TranscriptMessage::user("abcd").byte_len(), 4 + 4);
    assert_eq!(TranscriptMessage::assistant("abcd").byte_len(), 4 + 9);
    assert_eq!(TranscriptMessage::system("abcd").byte_len(), 4 + 6);
    assert_eq!(TranscriptMessage::tool("bash", "out").byte_len(), 3 + 4 + 4);
    let mut message = TranscriptMessage::tool("bash", "out");
    message.tool_call_id = Some("call-1".to_owned());
    assert_eq!(message.byte_len(), 3 + 4 + 4 + 6);
}

#[test]
fn compacted_text_renders_one_line_per_message() {
    let messages = vec![
        TranscriptMessage::system("rules"),
        TranscriptMessage::user("question"),
        TranscriptMessage::tool("bash", "output"),
    ];
    let result = compact_transcript(&messages, &CompactPolicy::new());
    assert_eq!(
        result.text(),
        "[system] rules\n[user] question\n[tool:bash] output"
    );

    let mut unnamed = TranscriptMessage::tool("", "output");
    unnamed.tool_name = None;
    assert_eq!(unnamed.render(), "[tool] output");
    assert_eq!(TranscriptMessage::system("s").render(), "[system] s");
}
