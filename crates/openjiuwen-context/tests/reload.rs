#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for the reload hint section.
//!
//! These tests pin the exact shipped hint texts (English and Chinese), the
//! marker-detection predicate, `reload_system_message`, and the idempotent
//! `inject_reload_hint` placement rule (after leading system messages, before
//! context messages). Everything is a pure function: no I/O, no model.

use openjiuwen_context::{
    has_offload_markers, inject_reload_hint, reload_hint, reload_system_message, ReloadLanguage,
    RELOAD_HINT_CN, RELOAD_HINT_EN,
};
use openjiuwen_foundation::prompt::{BaseMessage, Role};

/// A tool message carrying an in-memory offload marker.
fn marked_tool() -> BaseMessage {
    BaseMessage::tool("call_1", "[[OFFLOAD: handle=x, type=in_memory]]")
}

// ---------------------------------------------------------------------------
// Hint texts
// ---------------------------------------------------------------------------

#[test]
fn english_hint_documents_the_marker_and_path_guidance() {
    assert!(RELOAD_HINT_EN.starts_with("# Context Compression and Recovery"));
    assert!(RELOAD_HINT_EN.contains("```text"));
    assert!(RELOAD_HINT_EN.contains("[[OFFLOAD: handle=<id>, type=<type>, path=<path>]]"));
    assert!(RELOAD_HINT_EN.contains("use its exact value as `read_file.file_path`"));
    assert!(RELOAD_HINT_EN.contains("do not use it as `file_path`"));
    assert!(RELOAD_HINT_EN.contains("do not guess or construct one"));
    assert!(RELOAD_HINT_EN.contains("`filesystem` means the content is persisted at `path`"));
    assert!(RELOAD_HINT_EN.contains(
        "`in_memory` means it exists only in the session cache and cannot be recovered with `read_file`"
    ));
}

#[test]
fn chinese_hint_documents_the_marker_and_path_guidance() {
    assert!(RELOAD_HINT_CN.starts_with("# 上下文压缩与恢复"));
    assert!(RELOAD_HINT_CN.contains("```text"));
    assert!(RELOAD_HINT_CN.contains("[[OFFLOAD: handle=<id>, type=<type>, path=<path>]]"));
    assert!(RELOAD_HINT_CN.contains("使用其精确值作为 `read_file.file_path`"));
    assert!(RELOAD_HINT_CN.contains("不要把它当成 `file_path`"));
    assert!(RELOAD_HINT_CN.contains("不要猜测或拼接路径"));
    assert!(RELOAD_HINT_CN.contains("`filesystem` 表示内容已经持久化到 `path`"));
    assert!(RELOAD_HINT_CN.contains("`in_memory` 表示内容只在会话缓存中"));
}

#[test]
fn reload_hint_selects_the_requested_language() {
    assert_eq!(reload_hint(ReloadLanguage::En), RELOAD_HINT_EN);
    assert_eq!(reload_hint(ReloadLanguage::Cn), RELOAD_HINT_CN);
    assert_eq!(ReloadLanguage::default(), ReloadLanguage::En);
}

#[test]
fn reload_language_round_trips_through_serde() {
    assert_eq!(
        serde_json::to_string(&ReloadLanguage::En).unwrap(),
        "\"en\""
    );
    assert_eq!(
        serde_json::to_string(&ReloadLanguage::Cn).unwrap(),
        "\"cn\""
    );

    for language in [ReloadLanguage::En, ReloadLanguage::Cn] {
        let encoded = serde_json::to_string(&language).unwrap();
        let decoded: ReloadLanguage = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, language);
    }
    assert_eq!(
        serde_json::from_str::<ReloadLanguage>("\"cn\"").unwrap(),
        ReloadLanguage::Cn
    );
    assert_eq!(
        serde_json::from_str::<ReloadLanguage>("\"en\"").unwrap(),
        ReloadLanguage::En
    );
    assert!(serde_json::from_str::<ReloadLanguage>("\"fr\"").is_err());
}

// ---------------------------------------------------------------------------
// Detection and system message
// ---------------------------------------------------------------------------

#[test]
fn has_offload_markers_detects_marker_text_only() {
    assert!(has_offload_markers(&[marked_tool()]));
    assert!(has_offload_markers(&[
        BaseMessage::system("persona"),
        BaseMessage::user("hi"),
        marked_tool(),
    ]));
    assert!(has_offload_markers(&[BaseMessage::assistant(
        "output was [[OFFLOAD: handle=y, type=filesystem, path=/tmp/y.txt]]"
    )]));

    assert!(!has_offload_markers(&[]));
    assert!(!has_offload_markers(&[
        BaseMessage::system("persona"),
        BaseMessage::user("no markers here"),
        BaseMessage::assistant("plain answer"),
    ]));
    assert!(!has_offload_markers(&[BaseMessage::user(
        "[OFFLOAD: not a marker]"
    )]));
}

#[test]
fn reload_system_message_exists_only_with_markers() {
    let plain = vec![BaseMessage::system("persona"), BaseMessage::user("hi")];
    assert_eq!(reload_system_message(&plain, ReloadLanguage::En), None);
    assert_eq!(reload_system_message(&plain, ReloadLanguage::Cn), None);
    assert_eq!(reload_system_message(&[], ReloadLanguage::En), None);

    let marked = vec![marked_tool()];
    let english = reload_system_message(&marked, ReloadLanguage::En).unwrap();
    assert_eq!(english.role, Role::System);
    assert_eq!(english.text(), RELOAD_HINT_EN);

    let chinese = reload_system_message(&marked, ReloadLanguage::Cn).unwrap();
    assert_eq!(chinese.role, Role::System);
    assert_eq!(chinese.text(), RELOAD_HINT_CN);
}

// ---------------------------------------------------------------------------
// Injection
// ---------------------------------------------------------------------------

#[test]
fn inject_inserts_after_leading_system_messages_and_before_context() {
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::system("rules"),
        BaseMessage::user("hi"),
        marked_tool(),
    ];

    assert!(inject_reload_hint(&mut messages, ReloadLanguage::En));
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0].text(), "persona");
    assert_eq!(messages[1].text(), "rules");
    assert_eq!(messages[2].role, Role::System);
    assert_eq!(messages[2].text(), RELOAD_HINT_EN);
    assert_eq!(messages[3].text(), "hi");
    assert_eq!(messages[4].text(), marked_tool().text());
}

#[test]
fn inject_is_idempotent_for_the_same_and_the_other_language() {
    let original = vec![BaseMessage::system("persona"), marked_tool()];
    let mut messages = original.clone();

    assert!(inject_reload_hint(&mut messages, ReloadLanguage::En));
    let after_first = messages.clone();
    assert!(!inject_reload_hint(&mut messages, ReloadLanguage::En));
    assert_eq!(messages, after_first, "the second call changes nothing");

    // Switching language does not add a second reload section.
    assert!(!inject_reload_hint(&mut messages, ReloadLanguage::Cn));
    assert_eq!(messages, after_first);
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.text().starts_with("# Context Compression"))
            .count(),
        1
    );
}

#[test]
fn inject_does_nothing_without_markers() {
    let plain = vec![BaseMessage::system("persona"), BaseMessage::user("hi")];
    let mut messages = plain.clone();
    assert!(!inject_reload_hint(&mut messages, ReloadLanguage::En));
    assert_eq!(messages, plain);
}

#[test]
fn inject_places_the_hint_first_when_there_is_no_system_message() {
    let mut messages = vec![BaseMessage::user("hi"), marked_tool()];
    assert!(inject_reload_hint(&mut messages, ReloadLanguage::Cn));
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), RELOAD_HINT_CN);
    assert_eq!(messages[1].text(), "hi");
}

#[test]
fn inject_appends_after_a_history_that_is_all_system_messages() {
    let mut messages = vec![BaseMessage::system(
        "marker: [[OFFLOAD: handle=x, type=in_memory]]",
    )];
    assert!(inject_reload_hint(&mut messages, ReloadLanguage::En));
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(
        messages[0].text(),
        "marker: [[OFFLOAD: handle=x, type=in_memory]]"
    );
    assert_eq!(messages[1].text(), RELOAD_HINT_EN);
    assert!(!inject_reload_hint(&mut messages, ReloadLanguage::En));
}
