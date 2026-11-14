#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for the provider policy: context-window resolution, KV-cache
//! diffing, and the `OpenRouter` cache.
//!
//! Everything is offline and deterministic: resolution tests use injected caches
//! (never the process-wide one, except the one test that owns it), and the two
//! network paths use a loopback `wiremock` server. No test depends on the wall
//! clock — cache expiry is exercised with a zero TTL, not a sleep.
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::http::{HttpClient, HttpClientConfig};
use openjiuwen_context::policy::{
    detect_context_window_change, diff_positions, global_openrouter_window_cache,
    lookup_model_default_window, openrouter_disabled_error, parse_openrouter_models,
    tools_signature, ContextWindowChange, ContextWindowPolicy, ContextWindowSnapshot,
    ContextWindowSource, MessageFingerprint, OpenRouterModelTable, OpenRouterWindowCache,
    PositionChange, PromptCacheImpact, ResolvedContextWindow, DEFAULT_OPENROUTER_CACHE_TTL,
    DEFAULT_OPENROUTER_REQUEST_TIMEOUT, OPENROUTER_DISABLED_CODE, OPENROUTER_INVALID_RESPONSE_CODE,
    OPENROUTER_MODELS_URL, OPENROUTER_UNAVAILABLE_CODE,
};
use openjiuwen_foundation::prompt::{BaseMessage, ToolCall};
use serde_json::{json, Map, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A realistic slice of the `OpenRouter` catalogue body.
const OPENROUTER_PAYLOAD: &str = r#"{
  "data": [
    {"id": "openai/gpt-4o", "name": "GPT-4o", "context_length": 128000, "pricing": {"prompt": "0"}},
    {"id": "openai/gpt-4o-mini", "context_length": 128000},
    {"id": "anthropic/claude-sonnet-4.5", "context_length": 1000000},
    {"id": "broken/payload", "context_length": "not-a-number"},
    {"context_length": 4096},
    {"id": "zero/window", "context_length": 0},
    {"id": "negative/window", "context_length": -3}
  ]
}"#;

/// A policy pointing at `server`, with the fetch flag on and a private cache.
fn enabled_policy(server: &MockServer, cache: &Arc<OpenRouterWindowCache>) -> ContextWindowPolicy {
    ContextWindowPolicy::new()
        .with_openrouter_enabled(true)
        .with_openrouter_endpoint(format!("{}/api/v1/models", server.uri()))
        .with_openrouter_cache(Arc::clone(cache))
}

/// A snapshot from string messages; system messages come first.
fn window(system: &[&str], context: &[&str], tools: Vec<Value>) -> ContextWindowSnapshot {
    let system: Vec<BaseMessage> = system
        .iter()
        .map(|text| BaseMessage::system(*text))
        .collect();
    let context: Vec<BaseMessage> = context
        .iter()
        .map(|text| BaseMessage::user(*text))
        .collect();
    ContextWindowSnapshot::new(system, context, tools)
}

/// One tool definition.
fn tool(name: &str) -> Value {
    json!({"type": "function", "function": {"name": name, "description": "d"}})
}

/// An assistant message carrying one tool call.
fn call_message(id: &str, arguments: Value) -> BaseMessage {
    BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("search")
        .with_id(id)
        .with_arguments(arguments)])
}

// ---------------------------------------------------------------------------
// Window resolution
// ---------------------------------------------------------------------------

#[test]
fn global_cap_wins_over_every_other_rung() {
    let cache = Arc::new(OpenRouterWindowCache::new());
    cache.insert("gpt-4o", 999_999);
    let policy = ContextWindowPolicy::new()
        .with_global_tokens(42_000)
        .with_model_override(43_000)
        .with_model_window("gpt-4o", 44_000)
        .with_openrouter_cache(Arc::clone(&cache));

    let resolved = policy.resolve(Some("gpt-4o"), None);
    assert_eq!(resolved.tokens, 42_000);
    assert_eq!(resolved.source, ContextWindowSource::Global);
    assert_eq!(resolved.source.as_str(), "global");
    assert!(!resolved.is_default());
}

#[test]
fn model_override_beats_the_configured_map() {
    let policy = ContextWindowPolicy::new()
        .with_model_override(64_000)
        .with_model_window("gpt-4o", 44_000);

    let resolved = policy.resolve(Some("gpt-4o"), None);
    assert_eq!(resolved.tokens, 64_000);
    assert_eq!(resolved.source, ContextWindowSource::Override);
    assert_eq!(resolved.source.as_str(), "override");
}

#[test]
fn configured_map_is_case_insensitive() {
    let policy = ContextWindowPolicy::new().with_model_window("GPT-4o", 77_000);

    let lower = policy.resolve(Some("gpt-4o"), None);
    assert_eq!(lower.tokens, 77_000);
    assert_eq!(lower.source, ContextWindowSource::ConfiguredModel);
    assert_eq!(lower.source.as_str(), "configured_model");

    let upper = policy.resolve(Some("GPT-4O"), None);
    assert_eq!(upper.tokens, 77_000);
    assert_eq!(upper.source, ContextWindowSource::ConfiguredModel);
}

#[test]
fn builtin_table_beats_the_openrouter_cache() {
    let cache = Arc::new(OpenRouterWindowCache::new());
    cache.insert("gpt-4o", 1_000);
    let policy = ContextWindowPolicy::new().with_openrouter_cache(Arc::clone(&cache));

    let resolved = policy.resolve(Some("gpt-4o"), None);
    assert_eq!(resolved.tokens, 128_000);
    assert_eq!(resolved.source, ContextWindowSource::BuiltIn);
    assert_eq!(resolved.source.as_str(), "builtin");
}

#[test]
fn cache_answers_when_the_table_has_no_entry() {
    let cache = Arc::new(OpenRouterWindowCache::new());
    cache.insert("openai/acme-9", 4_096);
    let policy = ContextWindowPolicy::new()
        .with_openrouter_cache(Arc::clone(&cache))
        .with_provider("openai");

    let from_provider_field = policy.resolve(Some("acme-9"), None);
    assert_eq!(from_provider_field.tokens, 4_096);
    assert_eq!(
        from_provider_field.source,
        ContextWindowSource::OpenRouterCache
    );
    assert_eq!(from_provider_field.source.as_str(), "openrouter_cache");

    let from_provider_arg = policy.resolve(Some("acme-9"), Some("openai"));
    assert_eq!(from_provider_arg.tokens, 4_096);
    assert_eq!(
        from_provider_arg.source,
        ContextWindowSource::OpenRouterCache
    );

    // A bare model id resolves through its own cache key when no
    // `provider/model` entry exists.
    cache.insert("acme-7", 2_048);
    let from_bare_key = policy.resolve(Some("acme-7"), None);
    assert_eq!(from_bare_key.tokens, 2_048);
    assert_eq!(from_bare_key.source, ContextWindowSource::OpenRouterCache);
}

#[test]
fn default_is_the_last_rung_and_uses_the_configured_model_name() {
    let policy = ContextWindowPolicy::new().with_model("gpt-4o");
    let resolved = policy.resolve(None, None);
    assert_eq!(resolved.tokens, 128_000);
    assert_eq!(resolved.source, ContextWindowSource::BuiltIn);

    let unknown = policy.resolve(Some("acme/unknown-model"), None);
    assert_eq!(unknown.tokens, 200_000);
    assert_eq!(unknown.source, ContextWindowSource::Default);
    assert_eq!(unknown.source.as_str(), "default");
    assert!(unknown.is_default());
}

#[test]
fn zero_overrides_are_ignored_rather_than_winning() {
    let mut policy = ContextWindowPolicy::new()
        .with_global_tokens(0)
        .with_model_override(0)
        .with_model_window("gpt-4o", 0);
    policy.enable_openrouter_model_context_window_tokens = false;

    let resolved = policy.resolve(Some("gpt-4o"), None);
    assert_eq!(resolved.tokens, 128_000);
    assert_eq!(resolved.source, ContextWindowSource::BuiltIn);

    // A zero override still lets a positive override through.
    let with_override = ContextWindowPolicy::new()
        .with_global_tokens(0)
        .with_model_override(123_000);
    let resolved = with_override.resolve(Some("gpt-4o"), None);
    assert_eq!(resolved.tokens, 123_000);
    assert_eq!(resolved.source, ContextWindowSource::Override);
}

#[test]
fn builtin_table_covers_representative_families() {
    let cases = [
        ("glm-4", 128_000_usize),
        ("glm-5-turbo", 202_752),
        ("gpt-4o", 128_000),
        ("gpt-4.1", 1_047_576),
        ("gpt-3.5-turbo", 16_385),
        ("o3-mini", 200_000),
        ("deepseek-chat", 163_840),
        ("deepseek-v3", 128_000),
        ("claude-opus-4-5", 200_000),
        ("claude-sonnet-4.5", 1_000_000),
        ("gemini-2.5-pro", 1_048_576),
        ("llama-4-scout", 10_000_000),
        ("qwen-turbo", 8_192),
        ("qwen-max", 262_144),
        ("kimi-k2", 131_072),
        ("MiniMax-M2.1", 204_800),
        ("grok-4.20", 2_000_000),
    ];
    for (model, expected) in cases {
        assert_eq!(
            lookup_model_default_window(model),
            Some(expected),
            "built-in window mismatch for {model}"
        );
    }
    assert_eq!(lookup_model_default_window("minimax-m2.1"), Some(204_800));
    assert_eq!(lookup_model_default_window("  GPT-4O  "), Some(128_000));
    assert_eq!(lookup_model_default_window("acme/unknown-model"), None);
    assert_eq!(lookup_model_default_window(""), None);
    assert_eq!(lookup_model_default_window("   "), None);
}

#[test]
fn zero_request_timeout_is_a_config_error() {
    let mut policy = ContextWindowPolicy::new();
    assert_eq!(
        policy.openrouter_request_timeout,
        DEFAULT_OPENROUTER_REQUEST_TIMEOUT
    );
    assert!(policy.validate().is_ok());

    policy.openrouter_request_timeout = Duration::ZERO;
    let error = policy.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("openrouter_request_timeout"));
}

#[test]
fn endpoint_falls_back_to_the_public_catalogue() {
    assert_eq!(ContextWindowPolicy::new().endpoint(), OPENROUTER_MODELS_URL);
    assert_eq!(
        ContextWindowPolicy::new()
            .with_openrouter_endpoint("  ")
            .endpoint(),
        OPENROUTER_MODELS_URL
    );
    assert_eq!(
        ContextWindowPolicy::new()
            .with_openrouter_endpoint("http://localhost:1/models")
            .endpoint(),
        "http://localhost:1/models"
    );
}

#[test]
fn policy_and_resolution_round_trip_through_json() {
    let policy = ContextWindowPolicy::new()
        .with_global_tokens(11)
        .with_model_override(22)
        .with_model_window("gpt-4o", 33)
        .with_openrouter_enabled(true)
        .with_openrouter_timeout(Duration::from_millis(2_500))
        .with_openrouter_endpoint("http://localhost:1/models")
        .with_model("gpt-4o")
        .with_provider("openai");

    let value = serde_json::to_value(&policy).unwrap();
    assert_eq!(value["openrouter_request_timeout"], json!(2.5));
    assert_eq!(value["context_window_tokens"], json!(11));
    let restored: ContextWindowPolicy = serde_json::from_value(value).unwrap();
    assert_eq!(restored.context_window_tokens, Some(11));
    assert_eq!(restored.model_context_window_tokens_override, Some(22));
    assert_eq!(
        restored.model_context_window_tokens.get("gpt-4o"),
        Some(&33)
    );
    assert!(restored.enable_openrouter_model_context_window_tokens);
    assert_eq!(
        restored.openrouter_request_timeout,
        Duration::from_millis(2_500)
    );
    assert_eq!(restored.model_provider.as_deref(), Some("openai"));
    // The cache handle is not serialized, so a restored policy uses the global.
    assert!(restored.openrouter_cache.is_none());

    let partial: ContextWindowPolicy = serde_json::from_value(json!({
        "model_name": "claude-sonnet-4.5",
        "unknown_future_field": {"ignored": true}
    }))
    .unwrap();
    assert_eq!(partial.model_name.as_deref(), Some("claude-sonnet-4.5"));
    assert_eq!(
        partial.openrouter_request_timeout,
        DEFAULT_OPENROUTER_REQUEST_TIMEOUT
    );
    assert_eq!(partial.resolve(None, None).tokens, 1_000_000);

    // The explicit `{secs, nanos}` duration shape also loads.
    let explicit: ContextWindowPolicy =
        serde_json::from_value(json!({"openrouter_request_timeout": {"secs": 4, "nanos": 0}}))
            .unwrap();
    assert_eq!(explicit.openrouter_request_timeout, Duration::from_secs(4));

    let resolution = ContextWindowPolicy::new().resolve(Some("gpt-4o"), None);
    let restored: ResolvedContextWindow =
        serde_json::from_value(serde_json::to_value(resolution).unwrap()).unwrap();
    assert_eq!(restored, resolution);
}

// ---------------------------------------------------------------------------
// Fingerprints and snapshots
// ---------------------------------------------------------------------------

#[test]
fn message_fingerprints_are_stable_and_content_sensitive() {
    let base = BaseMessage::user("hello");
    assert_eq!(MessageFingerprint::of(&base), MessageFingerprint::of(&base));
    assert_eq!(MessageFingerprint::of(&base).as_str().len(), 64);

    let different_content = MessageFingerprint::of(&BaseMessage::user("hello!"));
    let different_role = MessageFingerprint::of(&BaseMessage::assistant("hello"));
    let name_changed = {
        let mut message = base.clone();
        message.name = Some("alice".to_string());
        MessageFingerprint::of(&message)
    };
    assert_ne!(MessageFingerprint::of(&base), different_content);
    assert_ne!(MessageFingerprint::of(&base), different_role);
    assert_ne!(MessageFingerprint::of(&base), name_changed);

    // Tool-call ids and arguments are part of the wire payload.
    let call_a = call_message("call-1", json!({"query": "x"}));
    let call_b = call_message("call-2", json!({"query": "x"}));
    let call_c = call_message("call-1", json!({"query": "y"}));
    assert_eq!(
        MessageFingerprint::of(&call_a),
        MessageFingerprint::of(&call_a)
    );
    assert_ne!(
        MessageFingerprint::of(&call_a),
        MessageFingerprint::of(&call_b)
    );
    assert_ne!(
        MessageFingerprint::of(&call_a),
        MessageFingerprint::of(&call_c)
    );

    // A tool result is fingerprinted by the call it answers.
    let result_one = BaseMessage::tool("call-1", "42");
    let result_two = BaseMessage::tool("call-2", "42");
    assert_ne!(
        MessageFingerprint::of(&result_one),
        MessageFingerprint::of(&result_two)
    );

    // Local bookkeeping metadata is not on the wire, so it never changes a
    // fingerprint.
    let mut annotated = base.clone();
    annotated
        .metadata
        .insert("context_message_id".to_string(), json!("abc"));
    assert_eq!(
        MessageFingerprint::of(&base),
        MessageFingerprint::of(&annotated)
    );

    // Fingerprints round-trip through JSON.
    let fingerprint = MessageFingerprint::of(&call_a);
    let restored: MessageFingerprint =
        serde_json::from_value(serde_json::to_value(&fingerprint).unwrap()).unwrap();
    assert_eq!(restored, fingerprint);
}

#[test]
fn tools_signature_ignores_key_order_but_not_content_or_order() {
    let mut first = Map::new();
    first.insert("name".to_string(), json!("search"));
    first.insert("description".to_string(), json!("find things"));
    let mut second = Map::new();
    second.insert("description".to_string(), json!("find things"));
    second.insert("name".to_string(), json!("search"));

    let left = vec![Value::Object(first.clone())];
    let right = vec![Value::Object(second)];
    assert_eq!(tools_signature(&left), tools_signature(&right));
    assert_eq!(tools_signature(&[]), tools_signature(&[]));
    assert_ne!(tools_signature(&[]), tools_signature(&left));

    let mut extra = left.clone();
    extra.push(tool("read_file"));
    assert_ne!(tools_signature(&left), tools_signature(&extra));

    let reordered = vec![tool("read_file"), tool("search")];
    let original = vec![tool("search"), tool("read_file")];
    assert_ne!(tools_signature(&reordered), tools_signature(&original));
}

#[test]
fn snapshot_derives_fingerprints_and_round_trips() {
    let system = vec![BaseMessage::system("persona")];
    let context = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
    let tools = vec![tool("search")];
    let snapshot = ContextWindowSnapshot::from_messages(&system, &context, &tools);

    assert_eq!(snapshot.message_count(), 3);
    assert_eq!(snapshot.system_message_count(), 1);
    assert_eq!(snapshot.message_fingerprints.len(), 3);
    assert_eq!(snapshot.messages().count(), 3);
    assert_eq!(snapshot.tools_signature, tools_signature(&tools));
    // System messages come first in the fingerprint order.
    assert_eq!(
        snapshot.message_fingerprints[0],
        MessageFingerprint::of(&BaseMessage::system("persona"))
    );

    let mut mutated = snapshot.clone();
    mutated.set_context_messages(vec![BaseMessage::user("hi")]);
    assert_eq!(mutated.message_count(), 2);
    assert_eq!(mutated.message_fingerprints.len(), 2);

    let mut derived = snapshot.clone();
    derived.message_fingerprints.clear();
    derived.tools_signature.clear();
    derived.recompute_derived();
    assert_eq!(derived, snapshot);

    let restored: ContextWindowSnapshot =
        serde_json::from_value(serde_json::to_value(&snapshot).unwrap()).unwrap();
    assert_eq!(restored, snapshot);
    assert_eq!(restored.message_fingerprints, snapshot.message_fingerprints);
}

// ---------------------------------------------------------------------------
// KV-cache diffing
// ---------------------------------------------------------------------------

#[test]
fn identical_windows_report_unchanged() {
    let previous = window(&["sys"], &["u1", "a1"], vec![tool("search")]);
    let current = window(&["sys"], &["u1", "a1"], vec![tool("search")]);

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(change, ContextWindowChange::Unchanged { messages: 3 });
    assert_eq!(change.kind(), "unchanged");
    assert_eq!(change.reusable_prefix_len(), 3);
    assert!(!change.invalidates_cache());

    let impact = change.prompt_cache_impact();
    assert_eq!(impact.reused_messages, 3);
    assert_eq!(impact.invalidated_messages, Some(0));
    assert_eq!(impact.reason, "unchanged");
}

#[test]
fn appended_windows_keep_the_whole_prefix() {
    let previous = window(&["sys"], &["u1", "a1"], vec![tool("search")]);
    let current = window(&["sys"], &["u1", "a1", "u2"], vec![tool("search")]);

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(change, ContextWindowChange::Appended { prefix_messages: 3 });
    assert_eq!(change.kind(), "appended");
    assert_eq!(change.reusable_prefix_len(), 3);
    assert!(!change.invalidates_cache());

    let impact = change.prompt_cache_impact();
    assert_eq!(impact.reused_messages, 3);
    assert_eq!(impact.invalidated_messages, Some(0));
    assert_eq!(impact.reason, "appended");
}

#[test]
fn system_prompt_changes_name_the_divergence_index() {
    let previous = window(&["sys-a", "sys-b"], &["u1"], vec![tool("search")]);
    let current = window(&["sys-a", "sys-b-changed"], &["u1"], vec![tool("search")]);

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(change, ContextWindowChange::SystemChanged { index: 1 });
    assert_eq!(change.kind(), "system_changed");
    assert_eq!(change.reusable_prefix_len(), 1);
    assert!(change.invalidates_cache());

    let impact = change.prompt_cache_impact();
    assert_eq!(impact.reused_messages, 1);
    assert_eq!(impact.invalidated_messages, None);
    assert!(impact.reason.starts_with("system_changed"));
}

#[test]
fn appending_a_tool_invalidates_the_message_suffix() {
    let previous = window(&["sys"], &["u1", "a1"], vec![tool("search")]);
    let current = window(
        &["sys"],
        &["u1", "a1"],
        vec![tool("search"), tool("read_file")],
    );

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(
        change,
        ContextWindowChange::ToolsChanged {
            requires_suffix_invalidation: true
        }
    );
    assert_eq!(change.kind(), "tools_changed");
    assert_eq!(change.reusable_prefix_len(), 0);
    assert!(change.invalidates_cache());

    let impact = change.prompt_cache_impact();
    assert_eq!(impact.reused_messages, 0);
    assert_eq!(impact.invalidated_messages, None);
    assert!(impact.reason.contains("tools_changed"));
}

#[test]
fn tool_changes_win_over_an_append() {
    // A new turn that also adds a tool: the tool shift moves every token after
    // the system prompt, so it is the change worth reporting.
    let previous = window(&["sys"], &["u1"], vec![tool("search")]);
    let current = window(
        &["sys"],
        &["u1", "a1"],
        vec![tool("search"), tool("read_file")],
    );

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(change.kind(), "tools_changed");
    assert!(change.invalidates_cache());
}

#[test]
fn rewritten_windows_report_the_first_divergence() {
    let previous_context = vec![
        BaseMessage::user("u1"),
        BaseMessage::assistant("a1"),
        BaseMessage::user("u2"),
    ];
    let mut current_context = previous_context.clone();
    current_context[1] = BaseMessage::assistant("a1 rewritten");
    let previous =
        ContextWindowSnapshot::from_messages(&[BaseMessage::system("sys")], &previous_context, &[]);
    let current =
        ContextWindowSnapshot::from_messages(&[BaseMessage::system("sys")], &current_context, &[]);

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(
        change,
        ContextWindowChange::Rewritten {
            first_divergence: 2
        }
    );
    assert_eq!(change.kind(), "rewritten");
    assert_eq!(change.reusable_prefix_len(), 2);
    assert!(change.invalidates_cache());

    // Truncation (a shorter window with an intact shared prefix) diverges at the
    // first missing message.
    let full = window(&["sys"], &["u1", "a1"], vec![]);
    let truncated = window(&["sys"], &["u1"], vec![]);
    let change = detect_context_window_change(&full, &truncated);
    assert_eq!(
        change,
        ContextWindowChange::Rewritten {
            first_divergence: 2
        }
    );
    assert!(change.invalidates_cache());

    let impact = change.prompt_cache_impact();
    assert_eq!(impact.reused_messages, 2);
    assert_eq!(impact.invalidated_messages, None);
    assert!(impact.reason.starts_with("rewritten"));
}

#[test]
fn model_revision_changes_invalidate_everything() {
    let previous = window(&["sys"], &["u1"], vec![]).with_model_revision("rev-a");
    let current = window(&["sys"], &["u1"], vec![]).with_model_revision("rev-b");

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(
        change,
        ContextWindowChange::ModelChanged {
            from_revision: Some("rev-a".to_string()),
            to_revision: Some("rev-b".to_string()),
        }
    );
    assert_eq!(change.kind(), "model_changed");
    assert_eq!(change.reusable_prefix_len(), 0);
    assert!(change.invalidates_cache());
    assert!(change
        .prompt_cache_impact()
        .reason
        .contains("model_changed from rev-a to rev-b"));
}

#[test]
fn position_only_changes_keep_the_prompt_identical() {
    let previous = window(&["sys"], &["u1"], vec![]).with_position(1);
    let current = window(&["sys"], &["u1"], vec![]).with_position(2);

    let change = detect_context_window_change(&previous, &current);
    assert_eq!(
        change,
        ContextWindowChange::PositionChanged {
            from: Some(1),
            to: Some(2),
        }
    );
    assert_eq!(change.kind(), "position_changed");
    assert_eq!(change.reusable_prefix_len(), 0);
    assert!(!change.invalidates_cache());

    let impact = change.prompt_cache_impact();
    assert_eq!(impact.invalidated_messages, Some(0));
    assert!(impact.reason.contains("position_changed"));

    // A position move next to an append is not what matters: the append is.
    let appended = window(&["sys"], &["u1", "a1"], vec![]).with_position(2);
    assert_eq!(
        detect_context_window_change(&previous, &appended).kind(),
        "appended"
    );
}

#[test]
fn change_variants_and_impacts_round_trip_through_json() {
    let samples = [
        ContextWindowChange::Unchanged { messages: 2 },
        ContextWindowChange::Appended { prefix_messages: 2 },
        ContextWindowChange::SystemChanged { index: 0 },
        ContextWindowChange::ToolsChanged {
            requires_suffix_invalidation: true,
        },
        ContextWindowChange::Rewritten {
            first_divergence: 1,
        },
        ContextWindowChange::ModelChanged {
            from_revision: Some("a".to_string()),
            to_revision: None,
        },
        ContextWindowChange::PositionChanged {
            from: None,
            to: Some(3),
        },
    ];
    for change in samples {
        let value = serde_json::to_value(&change).unwrap();
        assert_eq!(value["kind"], json!(change.kind()));
        let restored: ContextWindowChange = serde_json::from_value(value).unwrap();
        assert_eq!(restored, change);

        let impact = change.prompt_cache_impact();
        let restored: PromptCacheImpact =
            serde_json::from_value(serde_json::to_value(&impact).unwrap()).unwrap();
        assert_eq!(restored, impact);
    }

    // The tag alone is enough for `unchanged` (its size is optional).
    let restored: ContextWindowChange =
        serde_json::from_value(json!({"kind": "unchanged"})).unwrap();
    assert_eq!(restored, ContextWindowChange::Unchanged { messages: 0 });
}

#[test]
fn invalidates_cache_matches_the_prefix_rules() {
    assert!(!ContextWindowChange::Unchanged { messages: 5 }.invalidates_cache());
    assert!(!ContextWindowChange::Appended { prefix_messages: 5 }.invalidates_cache());
    assert!(ContextWindowChange::SystemChanged { index: 0 }.invalidates_cache());
    assert!(ContextWindowChange::ToolsChanged {
        requires_suffix_invalidation: false
    }
    .invalidates_cache());
    assert!(ContextWindowChange::Rewritten {
        first_divergence: 0
    }
    .invalidates_cache());
    assert!(!ContextWindowChange::PositionChanged {
        from: Some(1),
        to: Some(2)
    }
    .invalidates_cache());
}

#[test]
fn diff_positions_reports_revisions_and_deltas() {
    let base = window(&["sys"], &["u1"], vec![]);
    assert_eq!(diff_positions(&base, &base.clone()), Vec::new());

    let revised = base.clone().with_model_revision("m2");
    let original = base.clone().with_model_revision("m1");
    let changes = diff_positions(&original, &revised);
    assert_eq!(
        changes,
        vec![PositionChange::ModelRevision {
            from: Some("m1".to_string()),
            to: Some("m2".to_string()),
        }]
    );
    assert!(changes[0].invalidates_everything());

    // A revision label known on one side only is not a change.
    assert_eq!(diff_positions(&original, &base), Vec::new());

    let forward = diff_positions(
        &base.clone().with_position(2),
        &base.clone().with_position(5),
    );
    assert_eq!(
        forward,
        vec![PositionChange::Position {
            from: Some(2),
            to: Some(5),
            delta: Some(3),
        }]
    );
    assert!(!forward[0].invalidates_everything());

    let backward = diff_positions(
        &base.clone().with_position(5),
        &base.clone().with_position(2),
    );
    assert_eq!(
        backward,
        vec![PositionChange::Position {
            from: Some(5),
            to: Some(2),
            delta: Some(-3),
        }]
    );

    let unknown_origin = diff_positions(&base, &base.clone().with_position(7));
    assert_eq!(
        unknown_origin,
        vec![PositionChange::Position {
            from: None,
            to: Some(7),
            delta: None,
        }]
    );

    // Revision first, then position.
    let both = diff_positions(&original.with_position(1), &revised.with_position(2));
    assert_eq!(both.len(), 2);
    assert!(both[0].invalidates_everything());

    let restored: PositionChange =
        serde_json::from_value(serde_json::to_value(&both[1]).unwrap()).unwrap();
    assert_eq!(restored, both[1]);
}

// ---------------------------------------------------------------------------
// OpenRouter cache
// ---------------------------------------------------------------------------

#[test]
fn cache_is_case_insensitive_and_ttl_bounded() {
    let cache = OpenRouterWindowCache::new();
    assert_eq!(cache.ttl(), DEFAULT_OPENROUTER_CACHE_TTL);
    assert!(cache.is_empty());
    assert_eq!(cache.get("gpt-4o"), None);

    cache.insert("OpenAI/GPT-4o", 128_000);
    assert_eq!(cache.get("openai/gpt-4o"), Some(128_000));
    assert_eq!(cache.get("OPENAI/GPT-4O"), Some(128_000));
    assert!(cache.contains(" OpenAI/gpt-4o "));
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.snapshot().get("openai/gpt-4o").copied(),
        Some(128_000)
    );

    // Unusable entries never shadow a good one.
    cache.insert("  ", 1);
    cache.insert("zero", 0);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get("zero"), None);

    let mut entries = cache.snapshot();
    entries.insert("other".to_string(), 1);
    assert_eq!(cache.len(), 1, "snapshot is a copy");

    // A zero TTL expires everything without waiting.
    cache.set_ttl(Duration::ZERO);
    assert_eq!(cache.get("openai/gpt-4o"), None);
    assert_eq!(cache.len(), 0);
    assert!(cache.snapshot().is_empty());

    // Restoring the TTL makes new inserts fresh again.
    cache.set_ttl(DEFAULT_OPENROUTER_CACHE_TTL);
    cache.insert("openai/gpt-4o", 128_000);
    assert_eq!(cache.get("openai/gpt-4o"), Some(128_000));

    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn global_cache_is_one_shared_instance() {
    let first = global_openrouter_window_cache();
    let second = global_openrouter_window_cache();
    assert!(std::ptr::eq(first, second));

    first.insert("policy-global-test-model", 4_242);
    assert_eq!(second.get("POLICY-GLOBAL-TEST-MODEL"), Some(4_242));
    first.clear();
    assert_eq!(second.get("policy-global-test-model"), None);
}

#[test]
fn catalogue_parser_skips_unusable_records_and_adds_aliases() {
    let table = parse_openrouter_models(&json!({
        "data": [
            {"id": "openai/gpt-4o", "context_length": 128_000},
            {"id": "openai/gpt-4o-mini", "context_length": "128000"},
            {"id": "", "context_length": 1},
            {"id": "no-window"},
            {"id": "zero/window", "context_length": 0},
            {"context_length": 4096}
        ]
    }))
    .unwrap();
    assert_eq!(table.models, 2);
    assert_eq!(table.windows.get("openai/gpt-4o"), Some(&128_000));
    assert_eq!(table.windows.get("gpt-4o"), Some(&128_000));
    assert_eq!(table.windows.get("openai/gpt-4o-mini"), Some(&128_000));
    assert_eq!(table.windows.get("gpt-4o-mini"), Some(&128_000));
    assert!(!table.windows.contains_key("no-window"));

    // Ambiguous aliases are dropped rather than guessed.
    let ambiguous = parse_openrouter_models(&json!({
        "data": [
            {"id": "alpha/model-x", "context_length": 1000},
            {"id": "beta/model-x", "context_length": 2000},
            {"id": "gamma/solo", "context_length": 3000}
        ]
    }))
    .unwrap();
    assert_eq!(ambiguous.models, 3);
    assert!(!ambiguous.windows.contains_key("model-x"));
    assert_eq!(ambiguous.windows.get("alpha/model-x"), Some(&1000));
    assert_eq!(ambiguous.windows.get("beta/model-x"), Some(&2000));
    assert_eq!(ambiguous.windows.get("solo"), Some(&3000));

    let empty = parse_openrouter_models(&json!({"data": []})).unwrap();
    assert_eq!(empty, OpenRouterModelTable::default());
}

#[test]
fn catalogue_parser_rejects_shapes_that_are_not_a_catalogue() {
    for payload in [
        json!({"models": []}),
        json!({"data": "nope"}),
        json!([1, 2, 3]),
    ] {
        let error = parse_openrouter_models(&payload).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains(OPENROUTER_INVALID_RESPONSE_CODE),
            "unexpected message: {error}"
        );
    }
}

#[test]
fn disabled_error_is_a_typed_config_error() {
    let error = openrouter_disabled_error();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains(OPENROUTER_DISABLED_CODE));
}

// ---------------------------------------------------------------------------
// Fetch / prefetch against a loopback server
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fetch_warms_the_cache_and_prefetch_then_skips_the_request() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(OPENROUTER_PAYLOAD, "application/json"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);

    let resolved = policy
        .fetch(&[
            "openai/gpt-4o",
            "gpt-4o",
            "claude-sonnet-4.5",
            "zero/window",
        ])
        .await
        .unwrap();
    assert_eq!(resolved, 3, "three requested ids are covered");
    assert_eq!(cache.get("openai/gpt-4o"), Some(128_000));
    assert_eq!(cache.get("claude-sonnet-4.5"), Some(1_000_000));
    assert_eq!(cache.get("broken/payload"), None);
    assert_eq!(cache.get("zero/window"), None);

    let client = HttpClient::new(HttpClientConfig::default()).unwrap();
    assert_eq!(policy.prefetch(&["gpt-4o"], &client).await.unwrap(), 0);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn prefetch_fetches_when_the_cache_is_cold() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(OPENROUTER_PAYLOAD, "application/json"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);
    let client = HttpClient::new(HttpClientConfig::default()).unwrap();

    assert_eq!(
        policy
            .prefetch(&["anthropic/claude-sonnet-4.5"], &client)
            .await
            .unwrap(),
        1
    );
    assert_eq!(cache.get("anthropic/claude-sonnet-4.5"), Some(1_000_000));
}

#[tokio::test]
async fn fetch_is_offline_and_free_when_disabled() {
    let server = MockServer::start().await;
    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = ContextWindowPolicy::new()
        .with_openrouter_endpoint(format!("{}/api/v1/models", server.uri()))
        .with_openrouter_cache(Arc::clone(&cache));
    let client = HttpClient::new(HttpClientConfig::default()).unwrap();

    assert_eq!(policy.fetch(&["openai/gpt-4o"]).await.unwrap(), 0);
    assert_eq!(
        policy.prefetch(&["openai/gpt-4o"], &client).await.unwrap(),
        0
    );
    let no_models: [&str; 0] = [];
    assert_eq!(policy.fetch(&no_models).await.unwrap(), 0);
    assert!(cache.is_empty());
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "a disabled policy must never touch the network"
    );
}

#[tokio::test]
async fn resolve_never_fetches_even_when_the_cache_is_cold() {
    let server = MockServer::start().await;
    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);

    let resolved = policy.resolve(Some("acme/cold-model"), None);
    assert_eq!(resolved.tokens, 200_000);
    assert_eq!(resolved.source, ContextWindowSource::Default);
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "resolve is offline-only"
    );
}

#[tokio::test]
async fn a_malformed_payload_is_a_typed_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(r#"{"data": "nope"}"#, "application/json"),
        )
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);
    let error = policy.fetch(&["openai/gpt-4o"]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains(OPENROUTER_INVALID_RESPONSE_CODE));
    assert!(cache.is_empty());
}

#[tokio::test]
async fn a_payload_with_no_usable_windows_is_a_typed_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(r#"{"data": [{"id": "x"}]}"#, "application/json"),
        )
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);
    let error = policy.fetch(&["openai/gpt-4o"]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains(OPENROUTER_INVALID_RESPONSE_CODE));
}

#[tokio::test]
async fn an_http_failure_is_a_typed_transport_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);
    let error = policy.fetch(&["openai/gpt-4o"]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains(OPENROUTER_UNAVAILABLE_CODE));
    assert!(error.to_string().contains("500"));
}

#[tokio::test]
async fn the_request_is_bounded_by_the_policy_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_raw(OPENROUTER_PAYLOAD, "application/json"),
        )
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache).with_openrouter_timeout(Duration::from_millis(20));

    let error = policy.fetch(&["openai/gpt-4o"]).await.unwrap_err();
    assert!(
        matches!(error.kind(), ErrorKind::Timeout | ErrorKind::Transport),
        "unexpected error kind: {:?}",
        error.kind()
    );
    assert!(error.to_string().contains(OPENROUTER_UNAVAILABLE_CODE));
    assert!(cache.is_empty());
}

#[tokio::test]
async fn an_empty_request_list_never_hits_the_network() {
    let server = MockServer::start().await;
    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);

    let empty: [&str; 0] = [];
    assert_eq!(policy.fetch(&empty).await.unwrap(), 0);
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "no model ids means no request"
    );
}

#[tokio::test]
async fn a_fetched_model_resolves_through_the_cache() {
    // A model the built-in table does not know resolves from the fetched cache.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            r#"{"data": [{"id": "acme/fresh-model", "context_length": 65536}]}"#,
            "application/json",
        ))
        .mount(&server)
        .await;

    let cache = Arc::new(OpenRouterWindowCache::new());
    let policy = enabled_policy(&server, &cache);
    assert_eq!(policy.fetch(&["acme/fresh-model"]).await.unwrap(), 1);

    let resolved = policy.resolve(Some("fresh-model"), Some("acme"));
    assert_eq!(resolved.tokens, 65_536);
    assert_eq!(resolved.source, ContextWindowSource::OpenRouterCache);
}
