//! External-consumer integration tests for logging scopes, structured events,
//! redaction, and subscriber initialization.
//!
//! Everything here goes through the crate's public surface only
//! (`openjiuwen_common::logging::*` and `openjiuwen_common::security::*`), the way
//! downstream workspace crates consume it, and pins the user-visible behavior
//! ported from the Python SDK's `tests/unit_tests/core/common/log/test_logger.py`
//! (trace/member id defaults) and `test_structured_log.py` (`BaseLogEvent` shape).
//!
//! No environment variable is mutated in this file: the one test that inspects the
//! environment guards itself instead, so the suite stays deterministic on hosts
//! that export `OPENJIUWEN_LOG_*`. Exactly one test installs the process-global
//! subscriber (`init_with`), and its log records may appear on stdout.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use chrono::DateTime;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::logging::{
    current, current_member_id, current_trace_id, emit, event_types, info, init_with,
    is_initialized, targets, with_scope, EventStatus, LevelFilter, LogEvent, LogFormat, LogScope,
    LogWriter, LoggingConfig, ModuleType, DEFAULT_CONFIG_LEVEL, DEFAULT_TRACE_ID,
};
use openjiuwen_common::security::REDACTED;
use serde_json::json;

// port of tests/unit_tests/core/common/log/test_logger.py (default trace/member id)
#[tokio::test]
async fn defaults_apply_outside_any_scope() {
    assert!(current().is_none(), "no scope exists outside one");
    assert_eq!(DEFAULT_TRACE_ID, "default_trace_id");
    assert_eq!(current_trace_id(), DEFAULT_TRACE_ID);
    assert_eq!(current_member_id(), "");
}

#[test]
fn new_scopes_are_fresh_and_empty() {
    let first = LogScope::new();
    let second = LogScope::new();

    assert!(!first.trace_id.is_empty(), "a fresh scope gets a trace id");
    assert_ne!(first.trace_id, second.trace_id, "trace ids are unique");
    assert_eq!(first.trace_id.len(), 36, "trace ids are hyphenated uuids");
    assert_eq!(first.member_id, "");
    assert!(first.session_id.is_none());
    assert!(first.invocation_id.is_none());
    assert!(first.agent_id.is_none());
    assert!(first.conversation_id.is_none());
    assert_eq!(LogScope::default().member_id, "");
}

#[tokio::test]
async fn with_scope_is_visible_inside_and_does_not_leak() {
    let scope = Arc::new(
        LogScope::new()
            .with_member_id("teammate-1")
            .with_session_id("session-1"),
    );
    let expected = scope.trace_id.clone();

    let seen = with_scope(Arc::clone(&scope), async {
        let active = current().expect("the scope must be active inside the future");
        assert!(Arc::ptr_eq(&active, &scope), "the exact scope is installed");
        assert_eq!(active.session_id.as_deref(), Some("session-1"));
        assert_eq!(current_member_id(), "teammate-1");
        current_trace_id()
    })
    .await;

    assert_eq!(seen, expected);
    assert!(
        current().is_none(),
        "the scope must not leak past the future"
    );
    assert_eq!(current_trace_id(), DEFAULT_TRACE_ID);
    assert_eq!(current_member_id(), "");
}

/// A `tokio` task-local is **not** inherited by `tokio::spawn`; carrying a scope
/// into a spawned task is what `LogScope::instrument` is for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn instrument_installs_the_scope_in_a_spawned_task() {
    let scope = Arc::new(LogScope::new().with_trace_id("spawned-trace"));

    let not_inherited = with_scope(Arc::clone(&scope), async {
        tokio::spawn(async { current().is_none() }).await.unwrap()
    })
    .await;
    assert!(
        not_inherited,
        "tokio task-locals are not inherited by spawned tasks"
    );

    let visible = {
        let scope = Arc::clone(&scope);
        tokio::spawn(async move { scope.instrument(async { current_trace_id() }).await })
            .await
            .unwrap()
    };
    assert_eq!(visible, "spawned-trace");

    assert!(
        current().is_none(),
        "spawning must not leak into the parent"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_tasks_keep_separate_scopes() {
    let first = Arc::new(LogScope::new().with_trace_id("first"));
    let second = Arc::new(LogScope::new().with_trace_id("second"));

    let first_task = tokio::spawn({
        let scope = Arc::clone(&first);
        async move {
            scope
                .instrument(async {
                    tokio::task::yield_now().await;
                    (current_trace_id(), current_member_id())
                })
                .await
        }
    });
    let second_task = tokio::spawn({
        let scope = Arc::clone(&second);
        async move {
            scope
                .instrument(async {
                    tokio::task::yield_now().await;
                    current_trace_id()
                })
                .await
        }
    });

    let (first_seen, second_seen) = tokio::join!(first_task, second_task);
    assert_eq!(first_seen.unwrap(), ("first".to_owned(), String::new()));
    assert_eq!(second_seen.unwrap(), "second");
    assert!(current().is_none());
}

#[tokio::test]
async fn nested_scopes_restore_the_outer_scope() {
    let outer = Arc::new(LogScope::new().with_trace_id("outer"));
    let inner = Arc::new(LogScope::new().with_trace_id("inner"));

    with_scope(Arc::clone(&outer), async {
        assert_eq!(current_trace_id(), "outer");
        let inner_trace = with_scope(Arc::clone(&inner), async {
            assert_eq!(current_trace_id(), "inner");
            current_trace_id()
        })
        .await;
        assert_eq!(inner_trace, "inner");
        assert_eq!(current_trace_id(), "outer", "the outer scope is restored");
    })
    .await;

    assert!(current().is_none());
}

#[test]
fn span_carries_the_scope_fields() {
    let scope = LogScope::new()
        .with_trace_id("trace")
        .with_member_id("member")
        .with_session_id("session")
        .with_invocation_id("invocation")
        .with_agent_id("agent")
        .with_conversation_id("conversation");

    // A thread-local subscriber makes the callsite live without touching the
    // process-global subscriber installed by the `init_with` test.
    tracing::subscriber::with_default(tracing_subscriber::Registry::default(), || {
        let span = scope.span();
        let metadata = span.metadata().expect("a real subscriber yields metadata");
        assert_eq!(metadata.name(), "openjiuwen_scope");
        assert_eq!(metadata.target(), targets::COMMON);
        assert!(metadata.is_span());
        assert!(!metadata.is_event());
        let field_names: Vec<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        for expected in [
            "trace_id",
            "member_id",
            "session_id",
            "invocation_id",
            "agent_id",
            "conversation_id",
        ] {
            assert!(
                field_names.contains(&expected),
                "span field {expected} is missing from {field_names:?}"
            );
        }
    });
}

// port of tests/unit_tests/core/common/log/test_structured_log.py (event defaults)
#[test]
fn log_event_new_has_python_defaults() {
    let event = LogEvent::new(event_types::LLM_REQUEST);
    let event_id = event.event_id.clone();

    assert_eq!(event.event_type.as_str(), "llm_request");
    assert_eq!(event.log_level.as_str(), "info");
    assert_eq!(event.module_type, ModuleType::System);
    assert_eq!(event.status, EventStatus::Success);

    let groups: Vec<&str> = event_id.split('-').collect();
    assert_eq!(groups.len(), 5, "hyphenated uuid: {event_id}");
    assert!(
        groups.iter().map(|group| group.len()).eq([8, 4, 4, 4, 12]),
        "uuid v4 group lengths: {event_id}"
    );
    assert!(
        groups[2].starts_with('4'),
        "uuid v4 version nibble: {event_id}"
    );
    assert!(event.session_id.is_none());
}

#[test]
fn to_dict_drops_unset_optional_fields() {
    let event = LogEvent::new(event_types::LLM_REQUEST).with_message("hello");
    let dict = event.to_dict();

    for required in [
        "event_id",
        "event_type",
        "log_level",
        "timestamp",
        "module_type",
        "status",
        "metadata",
    ] {
        assert!(
            dict.contains_key(required),
            "missing required key {required}"
        );
    }
    for absent in [
        "conversation_id",
        "trace_id",
        "correlation_id",
        "parent_event_id",
        "error_code",
        "error_message",
        "stacktrace",
        "exception",
    ] {
        assert!(
            !dict.contains_key(absent),
            "unset key {absent} must be dropped"
        );
    }

    assert_eq!(dict.get("event_type"), Some(&json!("llm_request")));
    assert_eq!(dict.get("log_level"), Some(&json!("info")));
    assert_eq!(dict.get("module_type"), Some(&json!("system")));
    assert_eq!(dict.get("status"), Some(&json!("success")));
    assert_eq!(dict.get("message"), Some(&json!("hello")));
    assert_eq!(dict.get("metadata"), Some(&json!({})));
    assert_eq!(
        event.to_value(),
        serde_json::Value::Object(dict),
        "to_value wraps to_dict"
    );
}

#[test]
fn timestamp_serializes_as_iso8601() {
    let event = LogEvent::new(event_types::SYSTEM_START);
    let dict = event.to_dict();
    let timestamp = dict
        .get("timestamp")
        .and_then(serde_json::Value::as_str)
        .expect("the timestamp is a string");
    let parsed = DateTime::parse_from_rfc3339(timestamp).expect("RFC 3339 timestamp");
    assert_eq!(parsed.timestamp(), event.timestamp.timestamp());
}

// port of tests/unit_tests/core/common/log/test_structured_log.py (exception parity)
#[test]
fn with_exception_sets_python_error_fields() {
    let event = LogEvent::new(event_types::TOOL_CALL_END).with_exception("boom");
    assert_eq!(event.error_code.as_deref(), Some("-1"));
    assert_eq!(event.error_message.as_deref(), Some("boom"));
    assert_eq!(event.exception.as_deref(), Some("boom"));

    let dict = event.to_dict();
    assert_eq!(dict.get("error_code"), Some(&json!("-1")));
    assert_eq!(dict.get("error_message"), Some(&json!("boom")));
    assert_eq!(dict.get("exception"), Some(&json!("boom")));

    let explicit = LogEvent::new(event_types::TOOL_CALL_END)
        .with_error_code("E42")
        .with_exception("boom")
        .to_dict();
    assert_eq!(
        explicit.get("error_code"),
        Some(&json!("E42")),
        "an explicit error code wins over the exception default"
    );
}

#[test]
fn redacted_masks_sensitive_metadata() {
    let event = LogEvent::new(event_types::LLM_REQUEST)
        .with_metadata("query", json!("what is the secret?"))
        .with_metadata("messages", json!([{"role": "user", "content": "hi"}]))
        .with_metadata("nested", json!({"result": "top-secret", "count": 3}))
        .with_metadata("model", json!("gpt-4o"));

    let redacted = event.redacted(None);
    let metadata = &redacted["metadata"];
    assert_eq!(REDACTED, "<REDACTED>");
    assert_eq!(metadata["query"], json!(REDACTED));
    assert_eq!(metadata["messages"], json!(REDACTED));
    assert_eq!(metadata["nested"]["result"], json!(REDACTED));
    assert_eq!(metadata["nested"]["count"], json!(3));
    assert_eq!(metadata["model"], json!("gpt-4o"), "a safe key survives");
    assert_eq!(redacted["event_type"], json!("llm_request"));

    let custom = event
        .with_metadata("custom", json!("private"))
        .redacted(Some(&["custom"]));
    let custom_metadata = &custom["metadata"];
    assert_eq!(custom_metadata["custom"], json!(REDACTED));
    assert_eq!(
        custom_metadata["query"],
        json!("what is the secret?"),
        "an explicit list replaces the defaults"
    );
}

#[test]
fn logging_config_defaults_and_format_parsing() {
    let config = LoggingConfig::default();
    assert_eq!(config.level, DEFAULT_CONFIG_LEVEL);
    assert_eq!(DEFAULT_CONFIG_LEVEL, LevelFilter::INFO);
    assert_eq!(config.format, LogFormat::Text);
    assert_eq!(config.writer, LogWriter::Stdout);
    assert!(config.ansi);
    assert!(config.filter.is_none());

    assert_eq!("json".parse::<LogFormat>().unwrap(), LogFormat::Json);
    assert_eq!(" TEXT ".parse::<LogFormat>().unwrap(), LogFormat::Text);
    assert_eq!(LogFormat::Json.as_str(), "json");
    assert_eq!(LogFormat::Text.to_string(), "text");
    assert_eq!(
        "yaml".parse::<LogFormat>().unwrap_err().kind(),
        ErrorKind::Validation
    );
}

#[test]
fn from_env_starts_from_defaults_when_nothing_is_set() {
    let level = openjiuwen_common::config::env::log_level();
    let format = openjiuwen_common::config::env::log_format();
    if level.is_none() && format.is_none() {
        let config = LoggingConfig::from_env();
        assert_eq!(config.level, DEFAULT_CONFIG_LEVEL);
        assert_eq!(config.format, LogFormat::Text);
        assert_eq!(config.writer, LogWriter::Stdout);
        assert!(config.ansi);
        assert!(config.filter.is_none());
    } else {
        // A host exporting `OPENJIUWEN_LOG_*` is honored; only the invariant that
        // JSON output never carries ANSI styling can be asserted unconditionally.
        let config = LoggingConfig::from_env();
        if config.format == LogFormat::Json {
            assert!(!config.ansi, "JSON output must not contain ANSI escapes");
        }
    }
}

/// The only test that installs the process-global subscriber.
#[test]
fn subscriber_initialization_is_idempotent() {
    let config = LoggingConfig::default();
    assert!(
        init_with(&config).is_ok(),
        "first init installs or observes"
    );
    assert!(init_with(&config).is_ok(), "a second init is a no-op");
    assert!(is_initialized());
}

#[test]
fn structured_logging_helpers_do_not_panic() {
    let event = LogEvent::new(event_types::SESSION_CREATE)
        .with_module("session-1", "SessionManager")
        .with_message("session created")
        .with_metadata("query", json!("secret"));

    // Without any subscriber both paths degrade to no-ops and must never panic.
    info!(target: targets::LLM, "no subscriber installed");
    emit(&event);

    // A thread-local sink exercises the real record path (field capture, JSON
    // serialization, redaction) without touching the global subscriber.
    tracing::subscriber::with_default(tracing_subscriber::Registry::default(), || {
        info!(target: targets::LLM, model = "gpt-4o", "sink check");
        emit(&event);
        let span = LogScope::new().with_trace_id("sink").span();
        let _entered = span.enter();
        emit(&event);
    });
}
