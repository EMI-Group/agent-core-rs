//! Behavior tests for the incremental skill-notice session layer.
//!
//! Covers the port of Python `openjiuwen.symphony.discovery.toolkit`: the session tool
//! id, card normalization, the shared reminder rendering, the bounded notice registry
//! with least-recently-used eviction, and the output-budget rollback of
//! [`IncrementalNoticeSession`].
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;

use openjiuwen_symphony::discovery::toolkit::{
    cards_from_inventory, normalize_cards, render_reminder, tool_id, IncrementalNoticeRegistry,
    IncrementalNoticeSession, SkillCard, SkillCardSnapshot, SkillCards, MAX_SESSION_STATES,
    SKILL_INDEX_TOOL_NAME, TOOL_ID_DOMAIN,
};
use openjiuwen_symphony::discovery::{SkillInventory, SkillRecord};
use openjiuwen_symphony::error::ErrorKind;

/// One raw card with a name and a description.
fn card(name: &str, description: &str) -> SkillCard {
    SkillCard {
        name: Some(name.to_string()),
        description: Some(description.to_string()),
    }
}

/// Builds the raw card map from `(worker_id, name, description)` triples.
fn cards(entries: &[(&str, &str, &str)]) -> SkillCards {
    entries
        .iter()
        .map(|(worker_id, name, description)| ((*worker_id).to_string(), card(name, description)))
        .collect()
}

/// Builds one normalized snapshot.
fn snapshot(worker_id: &str, name: &str, description: &str) -> SkillCardSnapshot {
    SkillCardSnapshot {
        worker_id: worker_id.to_string(),
        name: name.to_string(),
        description: description.to_string(),
    }
}

/// Python's full `<system-reminder>` block around `rows`.
fn reminder_block(rows: &str) -> String {
    format!(
        "<system-reminder>\n\
         The following skills are newly available for use with the Skill tool:\n\n\
         {rows}\n\
         </system-reminder>"
    )
}

/// Python's compact fallback reminder.
fn compact_reminder(count: usize, tool_name: &str) -> String {
    format!(
        "<system-reminder>\n\
         {count} new skills are available. Use `{tool_name}` to search the current catalog.\n\
         </system-reminder>"
    )
}

#[test]
fn tool_id_matches_pythons_digest_and_is_deterministic() {
    let id = tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap();
    assert_eq!(
        id,
        "skill_index__39609f8e36bef90bd9f0443e3d527c4e0e27e3a68b29d085b4ff979407a3a447"
    );
    assert_eq!(id, tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap());
    // Python strips both arguments before hashing.
    assert_eq!(id, tool_id("  session-a  ", " skill_index ").unwrap());
    assert_eq!(TOOL_ID_DOMAIN, "openjiuwen.skill-index.v1");
    assert_eq!(id.len(), "skill_index__".len() + 64);
}

#[test]
fn tool_id_varies_with_scope_and_tool_name() {
    let base = tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap();
    let other_scope = tool_id("session-b", SKILL_INDEX_TOOL_NAME).unwrap();
    let other_tool = tool_id("session-a", "other_tool").unwrap();
    assert_ne!(base, other_scope);
    assert_ne!(base, other_tool);
    assert_ne!(other_scope, other_tool);
    assert!(other_tool.starts_with("other_tool__"));
    // A scope/tool pair is never ambiguous even when the parts contain underscores.
    assert_ne!(tool_id("a_b", "c").unwrap(), tool_id("a", "b_c").unwrap());
}

#[test]
fn tool_id_rejects_blank_scope_and_tool_name() {
    for (scope, name, field) in [
        ("", SKILL_INDEX_TOOL_NAME, "session_scope"),
        ("   ", SKILL_INDEX_TOOL_NAME, "session_scope"),
        ("session-a", "", "discovery_tool_name"),
        ("session-a", "  ", "discovery_tool_name"),
    ] {
        let error = tool_id(scope, name).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains(field),
            "unexpected message: {}",
            error.message()
        );
    }
    // The scope is validated first, exactly like Python's single guard.
    let error = tool_id("", "").unwrap_err();
    assert!(error.message().contains("session_scope"));
}

#[test]
fn cards_without_a_worker_id_are_skipped_and_ordered_by_id() {
    let normalized = normalize_cards(&cards(&[
        ("  ", "Ghost", "Ignored"),
        (" beta ", "Beta", "Second"),
        ("alpha", "Alpha", "First"),
    ]));
    assert_eq!(normalized.len(), 2);
    let ids: Vec<&str> = normalized.keys().map(String::as_str).collect();
    assert_eq!(ids, ["alpha", "beta"]);
    assert_eq!(normalized["beta"].worker_id, "beta");
    assert_eq!(normalized["beta"].name, "Beta");
    assert!(!normalized
        .keys()
        .any(|worker_id| worker_id.trim().is_empty()));
}

#[test]
fn names_and_descriptions_fall_back_to_the_worker_id() {
    let mut raw = cards(&[("pdf", "PDF", "Adds numbers")]);
    raw.insert(
        "bare".to_string(),
        SkillCard {
            name: None,
            description: None,
        },
    );
    raw.insert(
        "named".to_string(),
        SkillCard {
            name: Some("  Named   Skill ".to_string()),
            description: None,
        },
    );
    raw.insert("blank-description".to_string(), card("Blank", "   "));
    let normalized = normalize_cards(&raw);
    assert_eq!(normalized["bare"].name, "bare");
    assert_eq!(normalized["bare"].description, "bare");
    assert_eq!(normalized["named"].name, "Named Skill");
    assert_eq!(normalized["named"].description, "Named Skill");
    assert_eq!(normalized["blank-description"].description, "Blank");
    assert_eq!(normalized["pdf"].name, "PDF");
    assert_eq!(normalized["pdf"].description, "Adds numbers");
}

#[test]
fn descriptions_shorten_on_a_word_boundary_at_three_hundred_chars() {
    let long_description = "alpha ".repeat(100);
    let normalized = normalize_cards(&cards(&[("pdf", "PDF", &long_description)]));
    let description = &normalized["pdf"].description;
    assert_eq!(description, &format!("{}...", vec!["alpha"; 49].join(" ")));
    assert_eq!(description.chars().count(), 296);
    assert!(!description.ends_with(' '));

    let single_word = "x".repeat(400);
    let normalized = normalize_cards(&cards(&[("pdf", "PDF", &single_word)]));
    assert_eq!(normalized["pdf"].description, "...");

    let normalized = normalize_cards(&cards(&[("pdf", "PDF", "Adds numbers")]));
    assert_eq!(normalized["pdf"].description, "Adds numbers");
}

#[test]
fn names_and_descriptions_escape_angle_brackets_and_collapse_whitespace() {
    let normalized = normalize_cards(&cards(&[("  pdf\t", "  A   <b>\n c ", "Use <x> > y")]));
    let card = &normalized["pdf"];
    assert_eq!(card.worker_id, "pdf");
    assert_eq!(card.name, "A &lt;b&gt; c");
    assert_eq!(card.description, "Use &lt;x&gt; &gt; y");
}

#[test]
fn reminder_pins_pythons_exact_block_for_one_and_two_rows() {
    let one = vec![snapshot("alpha", "Alpha", "Adds numbers")];
    assert_eq!(
        render_reminder(&one, 4_000, SKILL_INDEX_TOOL_NAME).unwrap(),
        reminder_block("- Alpha: Adds numbers")
    );

    let two = vec![
        snapshot("beta", "beta", "Second"),
        snapshot("Alpha", "Alpha", "First"),
    ];
    assert_eq!(
        render_reminder(&two, 4_000, SKILL_INDEX_TOOL_NAME).unwrap(),
        reminder_block("- Alpha: First\n- beta: Second")
    );
}

#[test]
fn reminder_orders_rows_by_case_insensitive_name() {
    let added = vec![
        snapshot("zeta", "zeta", "Last"),
        snapshot("mixed", "alpha", "Tie"),
        snapshot("upper", "Alpha", "First"),
    ];
    assert_eq!(
        render_reminder(&added, 4_000, SKILL_INDEX_TOOL_NAME).unwrap(),
        reminder_block("- Alpha: First\n- alpha: Tie\n- zeta: Last")
    );
}

#[test]
fn reminder_is_none_without_additions_and_falls_back_when_the_budget_is_tight() {
    assert!(render_reminder(&[], 4_000, SKILL_INDEX_TOOL_NAME).is_none());
    assert!(render_reminder(&[], 0, SKILL_INDEX_TOOL_NAME).is_none());

    let one = vec![snapshot("alpha", "Alpha", "Adds numbers")];
    let full = render_reminder(&one, 4_000, SKILL_INDEX_TOOL_NAME).unwrap();
    let exact = full.chars().count();
    assert_eq!(
        render_reminder(&one, exact, SKILL_INDEX_TOOL_NAME).unwrap(),
        full
    );
    assert_eq!(
        render_reminder(&one, exact - 1, SKILL_INDEX_TOOL_NAME).unwrap(),
        compact_reminder(1, SKILL_INDEX_TOOL_NAME)
    );
    assert_eq!(
        render_reminder(&one, 0, SKILL_INDEX_TOOL_NAME).unwrap(),
        compact_reminder(1, SKILL_INDEX_TOOL_NAME)
    );

    let two = vec![
        snapshot("alpha", "Alpha", "First"),
        snapshot("beta", "beta", "Second"),
    ];
    assert_eq!(
        render_reminder(&two, 40, "other_tool").unwrap(),
        compact_reminder(2, "other_tool")
    );
}

#[test]
fn registry_initialize_baselines_without_consuming_a_delta() {
    let registry = IncrementalNoticeRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.capacity(), MAX_SESSION_STATES);
    assert_eq!(MAX_SESSION_STATES, 1_024);

    let alpha = cards(&[("alpha", "Alpha", "First")]);
    registry
        .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    assert_eq!(registry.len(), 1);
    assert_eq!(
        registry.tool_ids(),
        vec![tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap()]
    );
    assert!(registry
        .consume("session-a", &alpha, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_none());
}

#[test]
fn registry_consume_reports_additions_once() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let alpha_beta = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    let grown = cards(&[
        ("alpha", "Alpha", "First"),
        ("beta", "Beta", "Second"),
        ("gamma", "Gamma", "Third"),
    ]);

    let registry = IncrementalNoticeRegistry::new();
    registry
        .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    assert_eq!(
        registry
            .consume("session-a", &alpha_beta, 4_000, SKILL_INDEX_TOOL_NAME)
            .unwrap()
            .unwrap(),
        reminder_block("- Beta: Second")
    );
    assert!(registry
        .consume("session-a", &alpha_beta, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_none());
    assert_eq!(
        registry
            .consume("session-a", &grown, 4_000, SKILL_INDEX_TOOL_NAME)
            .unwrap()
            .unwrap(),
        reminder_block("- Gamma: Third")
    );

    // An unknown session is baselined by its first consume, so that call is silent.
    let fresh = IncrementalNoticeRegistry::new();
    assert!(fresh
        .consume("session-b", &alpha_beta, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_none());
    assert_eq!(
        fresh
            .consume("session-b", &grown, 4_000, SKILL_INDEX_TOOL_NAME)
            .unwrap()
            .unwrap(),
        reminder_block("- Gamma: Third")
    );
    // Two sessions keep independent cursors.
    assert!(registry
        .consume("session-c", &grown, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_none());
}

#[test]
fn registry_initialize_does_not_reset_an_existing_cursor() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let alpha_beta = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    let grown = cards(&[
        ("alpha", "Alpha", "First"),
        ("beta", "Beta", "Second"),
        ("gamma", "Gamma", "Third"),
    ]);

    let registry = IncrementalNoticeRegistry::new();
    registry
        .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    assert!(registry
        .consume("session-a", &alpha_beta, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_some());
    // Re-initializing with the grown catalog must not swallow the pending skill.
    registry
        .initialize("session-a", &grown, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    assert_eq!(
        registry
            .consume("session-a", &grown, 4_000, SKILL_INDEX_TOOL_NAME)
            .unwrap()
            .unwrap(),
        reminder_block("- Gamma: Third")
    );
}

#[test]
fn registry_clear_removes_every_cursor() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let registry = IncrementalNoticeRegistry::new();
    registry
        .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    registry
        .initialize("session-b", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    assert_eq!(registry.len(), 2);
    registry.clear();
    assert!(registry.is_empty());
    assert_eq!(registry.tool_ids(), Vec::<String>::new());
    // A cleared registry re-baselines the next session instead of announcing everything.
    assert!(registry
        .consume("session-a", &alpha, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_none());
    assert_eq!(registry.len(), 1);
}

#[test]
fn registry_evicts_the_least_recently_used_cursor() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let registry = IncrementalNoticeRegistry::with_capacity(2);
    assert_eq!(registry.capacity(), 2);
    for scope in ["session-a", "session-b", "session-c"] {
        registry
            .initialize(scope, &alpha, SKILL_INDEX_TOOL_NAME)
            .unwrap();
    }
    assert_eq!(registry.len(), 2);
    let tracked = registry.tool_ids();
    assert!(!tracked.contains(&tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap()));
    assert!(tracked.contains(&tool_id("session-b", SKILL_INDEX_TOOL_NAME).unwrap()));
    assert!(tracked.contains(&tool_id("session-c", SKILL_INDEX_TOOL_NAME).unwrap()));
    // The newest cursor keeps working and still reports its delta.
    let grown = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    assert_eq!(
        registry
            .consume("session-c", &grown, 4_000, SKILL_INDEX_TOOL_NAME)
            .unwrap()
            .unwrap(),
        reminder_block("- Beta: Second")
    );
}

#[test]
fn registry_lookup_refreshes_recency() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let registry = IncrementalNoticeRegistry::with_capacity(2);
    registry
        .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    registry
        .initialize("session-b", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    // Touching session-a makes session-b the least recently used cursor.
    assert!(registry
        .consume("session-a", &alpha, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap()
        .is_none());
    registry
        .initialize("session-c", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    let tracked = registry.tool_ids();
    assert!(tracked.contains(&tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap()));
    assert!(tracked.contains(&tool_id("session-c", SKILL_INDEX_TOOL_NAME).unwrap()));
    assert!(!tracked.contains(&tool_id("session-b", SKILL_INDEX_TOOL_NAME).unwrap()));
}

#[test]
fn registry_capacity_is_clamped_and_tool_ids_are_sorted() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let single = IncrementalNoticeRegistry::with_capacity(0);
    assert_eq!(single.capacity(), 1);
    single
        .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    single
        .initialize("session-b", &alpha, SKILL_INDEX_TOOL_NAME)
        .unwrap();
    assert_eq!(
        single.tool_ids(),
        vec![tool_id("session-b", SKILL_INDEX_TOOL_NAME).unwrap()]
    );

    let wide = IncrementalNoticeRegistry::with_capacity(8);
    for scope in ["session-c", "session-a", "session-b"] {
        wide.initialize(scope, &alpha, SKILL_INDEX_TOOL_NAME)
            .unwrap();
    }
    let tracked = wide.tool_ids();
    assert_eq!(tracked.len(), 3);
    assert!(tracked.windows(2).all(|pair| pair[0] < pair[1]));
    // Default and `new` share the Python capacity.
    assert_eq!(
        IncrementalNoticeRegistry::default().capacity(),
        IncrementalNoticeRegistry::new().capacity()
    );
}

#[test]
fn registry_rejects_a_zero_budget_before_touching_a_cursor() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let registry = IncrementalNoticeRegistry::new();
    let error = registry
        .consume("session-a", &alpha, 0, SKILL_INDEX_TOOL_NAME)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "max_chars: must be positive");
    assert!(registry.is_empty());

    let error = registry
        .consume("", &alpha, 4_000, SKILL_INDEX_TOOL_NAME)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("session_scope"));
    let error = registry.initialize("session-a", &alpha, "  ").unwrap_err();
    assert!(error.message().contains("discovery_tool_name"));
}

#[test]
fn session_append_combines_with_a_blank_line() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let alpha_beta = cards(&[("alpha", "Alpha", "First"), ("beta", "beta", "Second")]);
    let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    assert_eq!(
        session.tool_id(),
        tool_id("session-a", SKILL_INDEX_TOOL_NAME).unwrap()
    );
    assert_eq!(session.max_chars(), 4_000);
    assert_eq!(session.seen().len(), 1);
    assert!(session.seen().contains_key("alpha"));

    let (output, reminder) = session.append("result", &alpha_beta, None);
    let reminder = reminder.unwrap();
    assert_eq!(reminder, reminder_block("- beta: Second"));
    assert_eq!(output, format!("result\n\n{reminder}"));
    assert!(session.seen().contains_key("beta"));

    // Repeating the same cards is a no-op.
    let (output, reminder) = session.append("result", &alpha_beta, None);
    assert_eq!(output, "result");
    assert!(reminder.is_none());
}

#[test]
fn session_append_returns_only_the_reminder_for_empty_output() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    let (output, reminder) = session.append("", &alpha, None);
    assert!(reminder.is_none());
    assert_eq!(output, "");

    let grown = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    let (output, reminder) = session.append("", &grown, None);
    let reminder = reminder.unwrap();
    assert_eq!(output, reminder);
    assert_eq!(output, reminder_block("- Beta: Second"));
}

#[test]
fn session_append_rolls_back_an_oversized_reminder_and_delivers_it_later() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let grown = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();

    let (output, reminder) = session.append("body", &grown, Some(5));
    assert_eq!(output, "body");
    assert!(reminder.is_none());
    // Not acknowledged: the same skill is still pending.
    assert!(!session.seen().contains_key("beta"));
    assert_eq!(session.seen().len(), 1);

    let (output, reminder) = session.append("body", &grown, None);
    assert!(reminder.is_some());
    assert!(output.starts_with("body\n\n<system-reminder>"));
    assert!(session.seen().contains_key("beta"));
    let (output, reminder) = session.append("body", &grown, None);
    assert_eq!(output, "body");
    assert!(reminder.is_none());
}

#[test]
fn session_append_delivers_at_exactly_the_budget() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let grown = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    let (combined, reminder) = session.append("body", &grown, None);
    assert!(reminder.is_some());
    let budget = combined.chars().count();

    let mut exact = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    let (delivered, reminder) = exact.append("body", &grown, Some(budget));
    assert_eq!(delivered, combined);
    assert!(reminder.is_some());

    let mut tight = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    let (delivered, reminder) = tight.append("body", &grown, Some(budget - 1));
    assert_eq!(delivered, "body");
    assert!(reminder.is_none());
}

#[test]
fn session_rollback_drops_cards_that_disappeared() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let beta = cards(&[("beta", "Beta", "Second")]);
    let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    let (output, reminder) = session.append("body", &beta, Some(0));
    assert_eq!(output, "body");
    assert!(reminder.is_none());
    // Python keeps only the acknowledged subset of the current cards after a rollback.
    assert!(session.seen().is_empty());
    let (output, reminder) = session.append("body", &beta, None);
    let reminder = reminder.unwrap();
    assert_eq!(output, format!("body\n\n{reminder}"));
    assert_eq!(reminder, reminder_block("- Beta: Second"));
}

#[test]
fn session_append_acknowledges_when_nothing_was_added() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let edited = cards(&[("alpha", "Alpha", "First, now longer")]);
    let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
    // No skill was added, so the reminder is empty and the snapshot follows the cards,
    // exactly like Python's `prepare` finally block.
    let (output, reminder) = session.append("body", &edited, Some(0));
    assert_eq!(output, "body");
    assert!(reminder.is_none());
    assert_eq!(session.seen().len(), 1);
    assert_eq!(session.seen()["alpha"].description, "First, now longer");
}

#[test]
fn session_rebuilt_from_the_same_cards_reports_nothing_new() {
    let grown = cards(&[("alpha", "Alpha", "First"), ("beta", "Beta", "Second")]);
    let mut session = IncrementalNoticeSession::new("session-a", &grown, 4_000).unwrap();
    let (output, reminder) = session.append("body", &grown, None);
    assert_eq!(output, "body");
    assert!(reminder.is_none());

    let mut rebuilt = IncrementalNoticeSession::new("session-a", &grown, 4_000).unwrap();
    let (output, reminder) = rebuilt.append("body", &grown, None);
    assert_eq!(output, "body");
    assert!(reminder.is_none());
}

#[test]
fn session_rejects_a_zero_budget_and_a_blank_scope() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let error = IncrementalNoticeSession::new("session-a", &alpha, 0).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("max_chars"));
    let error = IncrementalNoticeSession::new("  ", &alpha, 4_000).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("session_scope"));
}

#[test]
fn cards_round_trip_through_json() {
    let normalized = normalize_cards(&cards(&[("pdf", "PDF <docs>", &"x ".repeat(2))]));
    let encoded = serde_json::to_string(&normalized).unwrap();
    let decoded: BTreeMap<String, SkillCardSnapshot> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, normalized);
    assert!(encoded.contains("\"worker_id\":\"pdf\""));

    let decoded_snapshot: SkillCardSnapshot =
        serde_json::from_str(r#"{"worker_id":"a","name":"A","future":1}"#).unwrap();
    assert_eq!(decoded_snapshot, snapshot("a", "A", ""));
    assert_eq!(
        serde_json::to_value(&decoded_snapshot).unwrap(),
        serde_json::json!({"worker_id": "a", "name": "A", "description": ""})
    );

    assert_eq!(
        serde_json::from_str::<SkillCard>("{}").unwrap(),
        SkillCard::default()
    );
    let card: SkillCard = serde_json::from_str(r#"{"name":"A","future_field":true}"#).unwrap();
    assert_eq!(card.name.as_deref(), Some("A"));
    assert!(card.description.is_none());
}

#[test]
fn cards_from_inventory_projects_records_onto_cards() {
    let inventory = SkillInventory::new(vec![
        SkillRecord::new("alpha", "Alpha").with_description("First <skill>"),
        SkillRecord::new("beta", "Beta"),
    ]);
    let raw = cards_from_inventory(&inventory);
    assert_eq!(raw.len(), 2);
    assert_eq!(raw["alpha"].name.as_deref(), Some("Alpha"));
    assert_eq!(raw["alpha"].description.as_deref(), Some("First <skill>"));
    assert_eq!(raw["beta"].description.as_deref(), Some(""));

    let normalized = normalize_cards(&raw);
    assert_eq!(normalized["alpha"].description, "First &lt;skill&gt;");
    // An empty record description falls back to the display name.
    assert_eq!(normalized["beta"].description, "Beta");
}

#[test]
fn reminders_are_byte_identical_across_runs() {
    let alpha = cards(&[("alpha", "Alpha", "First")]);
    let grown = cards(&[
        ("alpha", "Alpha", "First"),
        ("beta", "beta", "Second"),
        ("Gamma", "gamma", "Third"),
    ]);

    let run = || {
        let registry = IncrementalNoticeRegistry::new();
        registry
            .initialize("session-a", &alpha, SKILL_INDEX_TOOL_NAME)
            .unwrap();
        let registry_reminder = registry
            .consume("session-a", &grown, 4_000, SKILL_INDEX_TOOL_NAME)
            .unwrap();
        let mut session = IncrementalNoticeSession::new("session-a", &alpha, 4_000).unwrap();
        let (output, session_reminder) = session.append("body", &grown, None);
        (registry_reminder, output, session_reminder)
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
    assert_eq!(first.0, first.2);
    assert_eq!(
        first.0.unwrap(),
        reminder_block("- beta: Second\n- gamma: Third")
    );

    // Insertion order into the card map never reaches the reminder.
    let mut reversed: SkillCards = BTreeMap::new();
    for (worker_id, description) in [("Gamma", "Third"), ("beta", "Second"), ("alpha", "First")] {
        reversed.insert(worker_id.to_string(), card(worker_id, description));
    }
    assert_eq!(
        render_reminder(
            &normalize_cards(&reversed).into_values().collect::<Vec<_>>(),
            4_000,
            SKILL_INDEX_TOOL_NAME
        )
        .unwrap(),
        reminder_block("- alpha: First\n- beta: Second\n- Gamma: Third")
    );
}
