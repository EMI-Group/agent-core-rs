#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for state-snapshot reinjection.
//!
//! The registry is pinned by structure: registration replaces a provider in
//! place and keeps order, `only` filters without reordering, blank providers are
//! skipped, both leading state-header forms are stripped, and both the
//! per-section and the total character budgets hold while marking the section
//! lossy.
//!
//! The compaction integration is pinned end-to-end through
//! [`FullCompactProcessor`]: an accepted pass writes exactly one
//! `<recovered_context>` message between the summary and the kept tail, a
//! rejected pass changes nothing at all, a stale snapshot in the kept tail is
//! replaced rather than stacked, and the whole seam is inert unless both
//! `reinject_state` and a registry are configured — including through
//! [`PipelineDeps`].
//!
//! Everything is offline and deterministic: the one-character-per-token counter
//! measures the histories and the providers are fakes.
//!
//! [`FullCompactProcessor`]: openjiuwen_context::processor::FullCompactProcessor
//! [`PipelineDeps`]: openjiuwen_context::processor::PipelineDeps
use std::sync::{Arc, Mutex};

use openjiuwen_context::processor::compressor::reinjection::{
    is_reinjected_state_message, strip_state_header, truncate_state_text, ContextSnapshotProvider,
    SnapshotContext, SnapshotRegistry, SnapshotSection, DEFAULT_STATE_MARKER,
    DEFAULT_STATE_SNAPSHOT_MAX_CHARS, RECOVERED_CONTEXT_INSTRUCTION, RECOVERED_CONTEXT_TAG,
    REINJECTED_SECTIONS_METADATA_KEY, STATE_REINJECTION_MARKER,
};
use openjiuwen_context::processor::{
    ContextPipeline, ContextProcessor, FullCompactProcessor, FullCompactProcessorConfig,
    PipelineDeps, ProcessorConfig, ProcessorContext, FULL_COMPACT_BOUNDARY_TEXT,
    FULL_COMPACT_SUMMARY_INTRO,
};
use openjiuwen_context::{
    count_messages, ContextConfig, ContextEventKind, InMemoryOffloadStore, OffloadScope,
    OffloadStore,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// What a capturing provider saw when the registry called it.
#[derive(Default)]
struct SeenContext {
    session_state: Value,
    source_messages: usize,
    kept_messages: usize,
    has_workspace_root: bool,
    state_marker: String,
    max_chars: usize,
}

/// A fake provider with a fixed body, optionally capturing its context.
struct FakeProvider {
    name: &'static str,
    label: &'static str,
    content: Option<String>,
    lossy: bool,
    seen: Option<Arc<Mutex<SeenContext>>>,
}

impl FakeProvider {
    fn new(name: &'static str, label: &'static str, content: Option<&str>) -> Self {
        Self {
            name,
            label,
            content: content.map(str::to_owned),
            lossy: false,
            seen: None,
        }
    }

    fn lossy(name: &'static str, label: &'static str, content: &str) -> Self {
        Self {
            lossy: true,
            ..Self::new(name, label, Some(content))
        }
    }

    fn capturing(self, seen: Arc<Mutex<SeenContext>>) -> Self {
        Self {
            seen: Some(seen),
            ..self
        }
    }
}

impl ContextSnapshotProvider for FakeProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn label(&self) -> &str {
        self.label
    }

    fn build(&self, cx: &SnapshotContext<'_>) -> Option<SnapshotSection> {
        if let Some(seen) = &self.seen {
            *seen.lock().unwrap() = SeenContext {
                session_state: cx.session_state.clone(),
                source_messages: cx.source_messages.len(),
                kept_messages: cx.kept_messages.len(),
                has_workspace_root: cx.workspace_root.is_some(),
                state_marker: cx.state_marker.to_owned(),
                max_chars: cx.max_chars,
            };
        }
        self.content.clone().map(|content| {
            if self.lossy {
                SnapshotSection::lossy(content)
            } else {
                SnapshotSection::new(content)
            }
        })
    }
}

/// Boxes a fake provider as the registry's trait object.
fn provider(
    name: &'static str,
    label: &'static str,
    content: Option<&str>,
) -> Arc<dyn ContextSnapshotProvider> {
    Arc::new(FakeProvider::new(name, label, content))
}

/// A registry holding `providers`, in order.
fn registry(providers: Vec<Arc<dyn ContextSnapshotProvider>>) -> SnapshotRegistry {
    let mut registry = SnapshotRegistry::new();
    registry.register_all(providers);
    registry
}

/// A snapshot context over the given state and histories.
fn snapshot<'a>(
    state: &'a Value,
    source: &'a [BaseMessage],
    kept: &'a [BaseMessage],
) -> SnapshotContext<'a> {
    SnapshotContext::new(state, source, kept)
}

/// An empty history for the rendering tests.
const NO_MESSAGES: [BaseMessage; 0] = [];

#[test]
fn registration_replaces_in_place_and_preserves_order() {
    let mut registry = SnapshotRegistry::new();
    assert!(registry.is_empty());
    registry.register(provider("plan", "PLAN", Some("plan body")));
    registry.register(provider("skills", "SKILLS", Some("skills body")));
    registry.register(provider("plan", "PLAN", Some("new plan body")));
    assert_eq!(registry.names(), vec!["plan", "skills"]);
    assert_eq!(registry.len(), 2);
    assert!(!registry.is_empty());
    assert_eq!(registry.providers()[0].label(), "PLAN");
    assert_eq!(registry.iter().count(), 2);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES);
    let sections = registry.build_sections(&cx, None);
    assert_eq!(sections[0].name, "plan");
    assert_eq!(sections[0].content, "new plan body");
    assert_eq!(sections[1].name, "skills");
    assert!(format!("{registry:?}").contains("plan"));
}

#[test]
fn only_filters_without_reordering_and_blank_providers_are_skipped() {
    let registry = registry(vec![
        provider("plan", "PLAN", Some("plan body")),
        provider("skills", "SKILLS", Some("skills body")),
        provider("empty", "EMPTY", None),
        provider("blank", "BLANK", Some("   ")),
    ]);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES);
    assert_eq!(
        registry
            .build_sections(&cx, Some(&["skills", "plan"]))
            .iter()
            .map(|section| section.name.as_str())
            .collect::<Vec<_>>(),
        vec!["plan", "skills"]
    );
    assert_eq!(registry.build_sections(&cx, Some(&["empty"])).len(), 0);
    assert_eq!(registry.build_sections(&cx, None).len(), 2);
}

#[test]
fn a_provider_sees_the_snapshot_context_it_was_given() {
    let seen = Arc::new(Mutex::new(SeenContext::default()));
    let registry = registry(vec![Arc::new(
        FakeProvider::new("skills", "SKILLS", Some("Skill: audit")).capturing(Arc::clone(&seen)),
    )]);
    let state = json!({"team_task_status": {"name": "alpha"}});
    let source = vec![BaseMessage::user("u1"), BaseMessage::assistant("a1")];
    let kept = vec![BaseMessage::assistant("a1")];
    let cx = snapshot(&state, &source, &kept)
        .with_max_chars(321)
        .with_state_marker(STATE_REINJECTION_MARKER);
    let _ = registry.build_sections(&cx, None);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.session_state, state);
    assert_eq!(seen.source_messages, 2);
    assert_eq!(seen.kept_messages, 1);
    assert!(!seen.has_workspace_root);
    assert_eq!(seen.state_marker, STATE_REINJECTION_MARKER);
    assert_eq!(seen.max_chars, 321);
}

#[test]
fn both_state_header_forms_are_stripped() {
    assert_eq!(
        strip_state_header(
            &format!("{DEFAULT_STATE_MARKER}\n[SKILLS]\nbody"),
            DEFAULT_STATE_MARKER,
            "SKILLS"
        ),
        "body"
    );
    assert_eq!(
        strip_state_header(
            &format!("{DEFAULT_STATE_MARKER}\n[OTHER]\nbody"),
            DEFAULT_STATE_MARKER,
            "SKILLS"
        ),
        "body"
    );
    assert_eq!(
        strip_state_header(
            &format!("{DEFAULT_STATE_MARKER}\n[OTHER] trailing"),
            DEFAULT_STATE_MARKER,
            "SKILLS"
        ),
        format!("{DEFAULT_STATE_MARKER}\n[OTHER] trailing")
    );
    assert_eq!(
        strip_state_header("  body  ", DEFAULT_STATE_MARKER, "SKILLS"),
        "body"
    );
    let registry = registry(vec![provider(
        "skills",
        "SKILLS",
        Some(&format!("{DEFAULT_STATE_MARKER}\n[SKILLS]\nSkill: audit")),
    )]);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES);
    assert_eq!(
        registry.build_sections(&cx, None)[0].content,
        "Skill: audit"
    );
}

#[test]
fn per_section_trimming_keeps_head_and_tail_with_a_marker() {
    let body: String = ('a'..='z').cycle().take(1_000).collect();
    let registry = registry(vec![provider("skills", "SKILLS", Some(&body))]);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES).with_max_chars(100);
    let section = &registry.build_sections(&cx, None)[0];
    assert!(section.lossy);
    assert!(section.truncated_chars > 0);
    assert!(section.content.contains("[... truncated "));
    assert!(section.content.starts_with("abcdefghij"));
    assert!(section.content.chars().count() <= 140);
    let (trimmed, removed) = truncate_state_text(&body, 100);
    assert_eq!(trimmed, section.content);
    assert_eq!(section.truncated_chars, removed);
    assert_eq!(removed, 1_000 - 100);
}

#[test]
fn lossy_provider_flags_reach_the_rendered_section() {
    let registry = registry(vec![
        Arc::new(FakeProvider::new("complete", "COMPLETE", Some("body"))),
        Arc::new(FakeProvider::lossy("partial", "PARTIAL", "body")),
    ]);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES);
    let sections = registry.build_sections(&cx, None);
    assert!(!sections[0].lossy);
    assert_eq!(sections[0].truncated_chars, 0);
    assert!(sections[1].lossy);
    assert_eq!(sections[1].truncated_chars, 0);
}

#[test]
fn the_total_cap_keeps_every_section_within_budget() {
    let large = "x".repeat(400);
    let providers = vec![
        provider("small", "SMALL", Some("small body")),
        provider("large", "LARGE", Some(&large)),
    ];
    let capped = registry(providers.clone()).with_total_max_chars(120);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES);
    let sections = capped.build_sections(&cx, None);
    let total: usize = sections
        .iter()
        .map(|section| section.content.chars().count())
        .sum();
    assert!(total <= 120, "total {total} exceeded the total cap");
    assert_eq!(sections[0].content, "small body");
    assert!(sections[1].lossy);
    // A zero cap disables the total budget.
    let uncapped = registry(providers).with_total_max_chars(0);
    assert_eq!(uncapped.total_max_chars(), None);
    let sections = uncapped.build_sections(&cx, None);
    assert_eq!(sections[1].content.chars().count(), 400);
}

#[test]
fn the_per_section_messages_have_the_python_shape() {
    let registry = registry(vec![
        provider("skills", "SKILLS", Some("Skill: audit")),
        provider("plan", "PLAN", Some("plan body")),
    ]);
    let state = Value::Null;
    let cx =
        snapshot(&state, &NO_MESSAGES, &NO_MESSAGES).with_state_marker(STATE_REINJECTION_MARKER);
    let messages = registry.build_messages(&cx, None);
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0].text().into_owned(),
        format!("{STATE_REINJECTION_MARKER}\n[SKILLS]\nSkill: audit")
    );
    assert_eq!(
        messages[1].text().into_owned(),
        format!("{STATE_REINJECTION_MARKER}\n[PLAN]\nplan body")
    );
    assert_eq!(
        messages[0].metadata.get(REINJECTED_SECTIONS_METADATA_KEY),
        Some(&json!(["skills"]))
    );
}

#[test]
fn the_merged_message_has_the_exact_python_shape() {
    let registry = registry(vec![
        provider("skills", "SKILLS", Some("Skill: audit")),
        provider("plan_mode", "PLAN_MODE", Some("mode=plan")),
    ]);
    let state = Value::Null;
    let cx = snapshot(&state, &NO_MESSAGES, &NO_MESSAGES);
    let message = registry
        .build_single_message(&cx, None)
        .expect("two sections");
    assert_eq!(message.role, Role::User);
    let expected = "<recovered_context>
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
</recovered_context>";
    assert_eq!(message.text().into_owned(), expected);
    assert!(expected.contains(RECOVERED_CONTEXT_INSTRUCTION));
    assert_eq!(
        message.metadata.get(REINJECTED_SECTIONS_METADATA_KEY),
        Some(&json!(["skills", "plan_mode"]))
    );
    assert!(is_reinjected_state_message(&message));
    assert!(registry
        .build_single_message(&cx, Some(&["missing"]))
        .is_none());
}

#[test]
fn a_python_written_snapshot_is_also_recognised() {
    let python = BaseMessage::user(format!("{RECOVERED_CONTEXT_TAG}\n<section name=\"plan\">…"));
    assert!(is_reinjected_state_message(&python));
    let plain = BaseMessage::user("hello");
    assert!(!is_reinjected_state_message(&plain));
}

#[test]
fn config_defaults_and_old_configs_are_tolerated() {
    let config = FullCompactProcessorConfig::default();
    assert!(!config.reinject_state);
    assert_eq!(
        config.state_snapshot_max_chars,
        DEFAULT_STATE_SNAPSHOT_MAX_CHARS
    );
    assert_eq!(config.reinject_sections.len(), 0);
    assert_eq!(config.state_marker.as_deref(), Some(DEFAULT_STATE_MARKER));

    let old: FullCompactProcessorConfig =
        serde_json::from_value(json!({"trigger_total_tokens": 1_000})).unwrap();
    assert_eq!(old.trigger_total_tokens, 1_000);
    assert!(!old.reinject_state);
    assert_eq!(old.state_snapshot_max_chars, 8_000);
    assert_eq!(old.state_marker.as_deref(), Some(DEFAULT_STATE_MARKER));

    let round: FullCompactProcessorConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(round, config);

    let configured: FullCompactProcessorConfig = serde_json::from_value(json!({
        "reinject_state": true,
        "state_snapshot_max_chars": 123,
        "reinject_sections": ["plan", "skills"],
        "state_marker": "[STATE_REINJECTION]",
        "future_field": {"ignored": true},
    }))
    .unwrap();
    assert!(configured.reinject_state);
    assert_eq!(configured.state_snapshot_max_chars, 123);
    assert_eq!(configured.reinject_sections, vec!["plan", "skills"]);
    assert_eq!(
        configured.state_marker.as_deref(),
        Some(STATE_REINJECTION_MARKER)
    );
}

// ---------------------------------------------------------------------------
// Compaction integration
// ---------------------------------------------------------------------------

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a summary is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(1_200))
}

/// A four-message conversation with long bodies ending on a completed round.
fn conversation() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user(long("u1")),
        BaseMessage::assistant(long("a1")),
        BaseMessage::user(long("u2")),
        BaseMessage::assistant(long("a2")),
    ]
}

/// Shared processor inputs.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    store: Arc<dyn OffloadStore>,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            counter: counter(),
            store: Arc::new(InMemoryOffloadStore::new()),
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    fn context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.store,
            scope: &self.scope,
            budget: 1_000,
            trigger_tokens: 1_000,
            manual: false,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
        }
    }

    fn manual_context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            manual: true,
            ..self.context()
        }
    }
}

/// A compaction configuration that triggers on every long history here.
fn compact_config() -> FullCompactProcessorConfig {
    FullCompactProcessorConfig {
        trigger_total_tokens: 2_000,
        compression_call_max_tokens: 100_000,
        // Keep no tail unless a test asks for one.
        messages_to_keep: 0,
        reinject_state: true,
        ..FullCompactProcessorConfig::default()
    }
}

/// The number of recovered-context messages in `messages`.
fn snapshot_count(messages: &[BaseMessage]) -> usize {
    messages
        .iter()
        .filter(|message| is_reinjected_state_message(message))
        .count()
}

#[tokio::test]
async fn an_accepted_compaction_reinjects_one_snapshot_after_the_summary() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![
        provider("skills", "SKILLS", Some("Skill: audit")),
        provider("plan_mode", "PLAN_MODE", Some("mode=plan")),
    ]));
    let compressor =
        FullCompactProcessor::new(compact_config(), None, None).with_reinjection(registry);
    let mut messages = conversation();
    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].text(), FULL_COMPACT_BOUNDARY_TEXT);
    assert!(messages[1].text().starts_with(FULL_COMPACT_SUMMARY_INTRO));
    assert_eq!(messages[2].role, Role::User);
    let text = messages[2].text().into_owned();
    assert!(text.starts_with(RECOVERED_CONTEXT_TAG));
    assert!(text.contains(RECOVERED_CONTEXT_INSTRUCTION));
    assert!(text.contains("<section name=\"skills\">\nSkill: audit\n</section>"));
    assert!(text.contains("<section name=\"plan_mode\">\nmode=plan\n</section>"));
    assert_eq!(
        messages[2].metadata.get(REINJECTED_SECTIONS_METADATA_KEY),
        Some(&json!(["skills", "plan_mode"]))
    );
    assert_eq!(event.kind, ContextEventKind::Summarized);
    assert_eq!(
        event.tokens_after,
        count_messages(&fixture.counter, &messages)
    );
}

#[tokio::test]
async fn the_kept_tail_follows_the_reinjected_snapshot() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![provider(
        "skills",
        "SKILLS",
        Some("Skill: audit"),
    )]));
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            messages_to_keep: 2,
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(registry);
    let mut messages = conversation();
    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    assert_eq!(messages.len(), 5);
    assert!(is_reinjected_state_message(&messages[2]));
    assert_eq!(messages[3].text().into_owned(), long("u2"));
    assert_eq!(messages[4].text().into_owned(), long("a2"));
}

#[tokio::test]
async fn reinjection_is_off_by_default() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![provider(
        "skills",
        "SKILLS",
        Some("Skill: audit"),
    )]));
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            reinject_state: false,
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(registry);
    let mut messages = conversation();
    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    assert_eq!(snapshot_count(&messages), 0);
    assert!(!messages
        .iter()
        .any(|message| message.text().contains(RECOVERED_CONTEXT_TAG)));
}

#[tokio::test]
async fn a_missing_registry_leaves_an_enabled_reinjection_inert() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = conversation();
    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    assert_eq!(messages.len(), 2);
    assert_eq!(snapshot_count(&messages), 0);
}

#[tokio::test]
async fn a_rejected_compaction_changes_nothing() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![provider(
        "skills",
        "SKILLS",
        Some("Skill: audit"),
    )]));
    let compressor =
        FullCompactProcessor::new(compact_config(), None, None).with_reinjection(registry);
    // A two-message history that the summary would only make longer.
    let mut messages = vec![BaseMessage::user("hi"), BaseMessage::assistant("yo")];
    let untouched = messages.clone();
    assert!(compressor
        .process(&mut messages, &fixture.manual_context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);
    assert_eq!(snapshot_count(&messages), 0);
}

#[tokio::test]
async fn repeated_compaction_does_not_stack_snapshots() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![provider(
        "skills",
        "SKILLS",
        Some("Skill: audit"),
    )]));
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            messages_to_keep: 2,
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(Arc::clone(&registry));
    let mut messages = conversation();
    compressor
        .process(&mut messages, &fixture.manual_context())
        .await
        .unwrap()
        .expect("the first pass compacts");
    assert_eq!(snapshot_count(&messages), 1);
    // More conversation arrives, then a second compaction replaces everything
    // after the boundary — including the first snapshot.
    messages.push(BaseMessage::user(long("u3")));
    messages.push(BaseMessage::assistant(long("a3")));
    compressor
        .process(&mut messages, &fixture.manual_context())
        .await
        .unwrap()
        .expect("the second pass compacts again");
    assert_eq!(snapshot_count(&messages), 1);
}

#[tokio::test]
async fn a_stale_snapshot_in_the_kept_tail_is_replaced() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![provider(
        "skills",
        "SKILLS",
        Some("Skill: audit"),
    )]));
    let restart = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            messages_to_keep: 3,
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(Arc::clone(&registry));
    // The newest three messages carry a snapshot from an earlier session, so
    // the kept tail would otherwise duplicate it.
    let stale = {
        let mut message = BaseMessage::user(format!("{RECOVERED_CONTEXT_TAG}\nstale"));
        message.metadata.insert(
            REINJECTED_SECTIONS_METADATA_KEY.to_owned(),
            json!(["skills"]),
        );
        message
    };
    let mut messages = vec![
        BaseMessage::user(long("u0")),
        BaseMessage::assistant(long("a0")),
        stale,
        BaseMessage::user(long("u1")),
        BaseMessage::assistant(long("a1")),
    ];
    restart
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    assert_eq!(snapshot_count(&messages), 1);
    assert!(!messages
        .iter()
        .any(|message| message.text().contains("stale")));
    assert!(messages[2]
        .text()
        .contains("<section name=\"skills\">\nSkill: audit"));
}

#[tokio::test]
async fn reinject_sections_filters_the_registry_by_name() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![
        provider("skills", "SKILLS", Some("Skill: audit")),
        provider("plan", "PLAN", Some("plan body")),
    ]));
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            reinject_sections: vec!["plan".to_owned()],
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(registry);
    let mut messages = conversation();
    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    let text = messages[2].text().into_owned();
    assert!(text.contains("<section name=\"plan\">\nplan body"));
    assert!(!text.contains("skills"));
    assert_eq!(
        messages[2].metadata.get(REINJECTED_SECTIONS_METADATA_KEY),
        Some(&json!(["plan"]))
    );
}

#[tokio::test]
async fn the_snapshot_respects_the_configured_section_budget() {
    let fixture = Fixture::new();
    let body = "y".repeat(2_000);
    let registry = Arc::new(registry(vec![provider("notes", "NOTES", Some(&body))]));
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            state_snapshot_max_chars: 120,
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(registry);
    let mut messages = conversation();
    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    let text = messages[2].text().into_owned();
    assert!(text.contains("[... truncated "));
    assert!(!text.contains(&body));
}

#[tokio::test]
async fn a_provider_sees_the_compaction_context() {
    let fixture = Fixture::new();
    let seen = Arc::new(Mutex::new(SeenContext::default()));
    let registry = Arc::new(registry(vec![Arc::new(
        FakeProvider::new("skills", "SKILLS", Some("Skill: audit")).capturing(Arc::clone(&seen)),
    )]));
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            messages_to_keep: 2,
            state_snapshot_max_chars: 500,
            ..compact_config()
        },
        None,
        None,
    )
    .with_reinjection(registry);
    let mut messages = conversation();
    let history_len = messages.len();
    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");
    let seen = seen.lock().unwrap();
    // The provider sees the pre-compaction history, the surviving tail and the
    // configured budget; the crate holds no session state or workspace here.
    assert_eq!(seen.source_messages, history_len);
    assert_eq!(seen.kept_messages, 2);
    assert_eq!(seen.session_state, Value::Null);
    assert!(!seen.has_workspace_root);
    assert_eq!(seen.state_marker, DEFAULT_STATE_MARKER);
    assert_eq!(seen.max_chars, 500);
}

#[tokio::test]
async fn pipeline_deps_wire_the_registry_into_full_compaction() {
    let fixture = Fixture::new();
    let registry = Arc::new(registry(vec![provider(
        "skills",
        "SKILLS",
        Some("Skill: audit"),
    )]));
    let deps =
        PipelineDeps::new(Arc::clone(&fixture.store)).with_reinjection(Some(Arc::clone(&registry)));
    let pipeline = ContextPipeline::from_configs_with(
        &[ProcessorConfig::FullCompact(compact_config())],
        &deps,
    )
    .unwrap();
    let mut messages = conversation();
    pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap();
    assert_eq!(snapshot_count(&messages), 1);
    assert_eq!(messages[2].role, Role::User);
    assert!(messages[2]
        .text()
        .contains("<section name=\"skills\">\nSkill: audit"));
}

#[tokio::test]
async fn a_pipeline_without_the_dependency_never_reinjects() {
    let fixture = Fixture::new();
    let deps = PipelineDeps::new(Arc::clone(&fixture.store));
    assert!(deps.reinjection.is_none());
    let pipeline = ContextPipeline::from_configs_with(
        &[ProcessorConfig::FullCompact(compact_config())],
        &deps,
    )
    .unwrap();
    let mut messages = conversation();
    pipeline
        .run(&mut messages, &fixture.context())
        .await
        .unwrap();
    assert_eq!(snapshot_count(&messages), 0);
}
