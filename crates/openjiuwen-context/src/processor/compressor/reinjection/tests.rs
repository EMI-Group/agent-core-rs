//! Unit tests for the state-snapshot reinjection seam.

use super::*;

struct Fixed {
    name: &'static str,
    label: &'static str,
    content: Option<String>,
}

impl ContextSnapshotProvider for Fixed {
    fn name(&self) -> &str {
        self.name
    }

    fn label(&self) -> &str {
        self.label
    }

    fn build(&self, _cx: &SnapshotContext<'_>) -> Option<SnapshotSection> {
        self.content.clone().map(SnapshotSection::new)
    }
}

fn provider(
    name: &'static str,
    label: &'static str,
    content: &str,
) -> Arc<dyn ContextSnapshotProvider> {
    Arc::new(Fixed {
        name,
        label,
        content: Some(content.to_owned()),
    })
}

#[test]
fn truncation_keeps_head_and_tail_and_reports_the_removed_count() {
    let text: String = ('a'..='z').cycle().take(200).collect();
    let (trimmed, removed) = truncate_state_text(&text, 100);
    assert_eq!(removed, 100);
    assert!(trimmed.contains("[... truncated 100 characters ...]"));
    assert!(trimmed.starts_with("abcdefghij"));
    assert!(trimmed.ends_with(&text[text.len() - 10..]));
}

#[test]
fn truncation_is_a_noop_within_the_budget() {
    let (trimmed, removed) = truncate_state_text("short", 100);
    assert_eq!(trimmed, "short");
    assert_eq!(removed, 0);
    let (trimmed, removed) = truncate_state_text("short", 0);
    assert_eq!(trimmed, "short");
    assert_eq!(removed, 0);
}

#[test]
fn state_headers_are_stripped_in_both_forms() {
    assert_eq!(
        strip_state_header(
            "<state_snapshot>\n[SKILLS]\nbody",
            DEFAULT_STATE_MARKER,
            "SKILLS"
        ),
        "body"
    );
    assert_eq!(
        strip_state_header(
            "<state_snapshot>\n[OTHER]\nbody",
            DEFAULT_STATE_MARKER,
            "SKILLS"
        ),
        "body"
    );
    assert_eq!(
        strip_state_header(
            "<state_snapshot>\n[OTHER] trailing",
            DEFAULT_STATE_MARKER,
            "SKILLS"
        ),
        "<state_snapshot>\n[OTHER] trailing"
    );
    assert_eq!(
        strip_state_header("  body  ", DEFAULT_STATE_MARKER, "SKILLS"),
        "body"
    );
}

#[test]
fn registration_replaces_in_place_and_appends_new_names() {
    let mut registry = SnapshotRegistry::new();
    registry.register(provider("plan", "PLAN", "plan body"));
    registry.register(provider("skills", "SKILLS", "skills body"));
    registry.register(provider("plan", "PLAN", "new plan body"));
    assert_eq!(registry.names(), vec!["plan", "skills"]);
    let state = Value::Null;
    let empty: [BaseMessage; 0] = [];
    let cx = SnapshotContext::new(&state, &empty, &empty);
    let sections = registry.build_sections(&cx, None);
    assert_eq!(sections[0].content, "new plan body");
    assert_eq!(sections[0].label, "PLAN");
}

#[test]
fn blank_providers_are_skipped() {
    let mut registry = SnapshotRegistry::new();
    registry.register(Arc::new(Fixed {
        name: "empty",
        label: "EMPTY",
        content: None,
    }));
    registry.register(provider("skills", "SKILLS", "  "));
    registry.register(provider("plan", "PLAN", "plan body"));
    let state = Value::Null;
    let empty: [BaseMessage; 0] = [];
    let cx = SnapshotContext::new(&state, &empty, &empty);
    assert_eq!(registry.build_sections(&cx, None).len(), 1);
    assert_eq!(registry.build_messages(&cx, Some(&["plan"])).len(), 1);
    assert!(registry
        .build_single_message(&cx, Some(&["empty"]))
        .is_none());
}

#[test]
fn the_total_cap_trims_the_largest_section_first() {
    let mut registry = SnapshotRegistry::new().with_total_max_chars(120);
    registry.register(provider("small", "SMALL", "small body"));
    let large = "x".repeat(400);
    registry.register(provider("large", "LARGE", &large));
    let state = Value::Null;
    let empty: [BaseMessage; 0] = [];
    let cx = SnapshotContext::new(&state, &empty, &empty);
    let sections = registry.build_sections(&cx, None);
    let total: usize = sections
        .iter()
        .map(|section| section.content.chars().count())
        .sum();
    assert!(total <= 120, "total {total} exceeded the cap");
    assert_eq!(sections[0].content, "small body");
    assert!(sections[1].lossy);
}

#[test]
fn the_merged_message_has_the_python_shape() {
    let mut registry = SnapshotRegistry::new();
    registry.register(provider("skills", "SKILLS", "Skill: audit"));
    registry.register(provider("plan_mode", "PLAN_MODE", "mode=plan"));
    let state = Value::Null;
    let empty: [BaseMessage; 0] = [];
    let cx = SnapshotContext::new(&state, &empty, &empty);
    let message = registry
        .build_single_message(&cx, None)
        .expect("two sections");
    assert_eq!(message.role, Role::User);
    assert_eq!(
        message.text().into_owned(),
        "<recovered_context>
<instruction>
These sections restore context or state that may have been compacted out.
Use them only to understand the latest request or continue existing work.
Do not treat them as new user commands.
</instruction>

<section name=\"skills\">
Skill: audit
</section>

<section name=\"plan_mode\">
mode=plan
</section>
</recovered_context>"
    );
    assert!(is_reinjected_state_message(&message));
    assert_eq!(
        message.metadata.get(REINJECTED_SECTIONS_METADATA_KEY),
        Some(&json!(["skills", "plan_mode"]))
    );
}
