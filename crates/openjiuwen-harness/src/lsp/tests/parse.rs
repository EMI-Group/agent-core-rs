//! Tests for the text forms of [`LspToolConfig`], [`LspAction`] and [`LspArgs`].
//!
//! Every constructor, every grammar form and every error path is covered; the serde and JSON
//! schema shapes asserted here are the pre-existing ones, so the parsers cannot drift the wire
//! contract.
use std::str::FromStr;
use std::time::Duration;

use openjiuwen_common::error::Error;
use serde_json::json;

use crate::lsp::tool::{LspAction, LspArgs, LspTool, LspToolConfig};

/// Asserts two configurations are equal, field by field (`LspToolConfig` has no `PartialEq`).
fn assert_config_eq(actual: &LspToolConfig, expected: &LspToolConfig) {
    assert_eq!(actual.workspace_root, expected.workspace_root);
    assert_eq!(actual.language, expected.language);
    assert_eq!(actual.server, expected.server);
    assert_eq!(actual.servers, expected.servers);
    assert_eq!(actual.restart_policy, expected.restart_policy);
    assert_eq!(actual.language_id, expected.language_id);
    assert_eq!(actual.request_timeout, expected.request_timeout);
    assert_eq!(actual.max_file_size, expected.max_file_size);
    assert_eq!(actual.context_lines, expected.context_lines);
}

/// Asserts `error` is a validation error whose message contains every `needle`.
fn assert_validation(error: &Error, needles: &[&str]) {
    assert!(
        matches!(error, Error::Validation(_)),
        "expected a validation error, got {error:?}"
    );
    let message = error.message();
    for needle in needles {
        assert!(
            message.contains(needle),
            "message {message:?} does not contain {needle:?}"
        );
    }
}

#[test]
fn config_new_is_the_default_configuration() {
    assert_config_eq(&LspToolConfig::new(), &LspToolConfig::default());
    let config = LspToolConfig::new();
    assert!(config.workspace_root.is_none());
    assert_eq!(config.language, "cn");
    assert_eq!(config.language_id, "plaintext");
    assert_eq!(config.request_timeout, Duration::from_secs(15));
    assert_eq!(config.max_file_size, 10 * 1024 * 1024);
    assert_eq!(config.context_lines, 0);
    assert!(config.server.is_none());
    assert_eq!(config.servers.len(), 0);
}

#[test]
fn config_spec_is_empty_or_whitespace_for_the_defaults() {
    assert_config_eq(
        &LspToolConfig::from_str("").unwrap(),
        &LspToolConfig::default(),
    );
    assert_config_eq(
        &LspToolConfig::from_str("   ").unwrap(),
        &LspToolConfig::default(),
    );
    assert_eq!(LspToolConfig::default().to_string(), "");
}

#[test]
fn config_spec_parses_every_scalar_key() {
    let config = LspToolConfig::from_str(
        "workspace=/tmp/ws,language=en,language_id=rust,request_timeout=1500ms,max_file_size=2048,context_lines=3",
    )
    .unwrap();
    assert_eq!(
        config.workspace_root.as_deref(),
        Some(std::path::Path::new("/tmp/ws"))
    );
    assert_eq!(config.language, "en");
    assert_eq!(config.language_id, "rust");
    assert_eq!(config.request_timeout, Duration::from_millis(1500));
    assert_eq!(config.max_file_size, 2048);
    assert_eq!(config.context_lines, 3);
}

#[test]
fn config_spec_accepts_any_key_order_and_padding() {
    let config = LspToolConfig::from_str(" context_lines = 2 , language_id = python ").unwrap();
    assert_eq!(config.context_lines, 2);
    assert_eq!(config.language_id, "python");
    assert_eq!(config.request_timeout, Duration::from_secs(15));
}

#[test]
fn config_spec_parses_duration_suffixes() {
    let millis = LspToolConfig::from_str("request_timeout=250ms").unwrap();
    assert_eq!(millis.request_timeout, Duration::from_millis(250));
    let seconds = LspToolConfig::from_str("request_timeout=30s").unwrap();
    assert_eq!(seconds.request_timeout, Duration::from_secs(30));
    let nanos = LspToolConfig::from_str("request_timeout=1000000ns").unwrap();
    assert_eq!(nanos.request_timeout, Duration::from_millis(1));
}

#[test]
fn config_spec_rejects_unknown_keys() {
    let error = LspToolConfig::from_str("colour=blue").unwrap_err();
    assert_validation(&error, &["colour", "colour=blue", "workspace"]);
    // Server definitions are builder-only, never part of the spec.
    let error = LspToolConfig::from_str("server=rust").unwrap_err();
    assert_validation(&error, &["unknown key 'server'"]);
    let error = LspToolConfig::from_str("servers=rust").unwrap_err();
    assert_validation(&error, &["unknown key 'servers'"]);
}

#[test]
fn config_spec_rejects_duplicate_keys() {
    let error = LspToolConfig::from_str("language=en,language=cn").unwrap_err();
    assert_validation(&error, &["duplicate key 'language'"]);
}

#[test]
fn config_spec_rejects_empty_and_malformed_items() {
    let error = LspToolConfig::from_str("language=en,,context_lines=1").unwrap_err();
    assert_validation(&error, &["empty item"]);
    let error = LspToolConfig::from_str("language").unwrap_err();
    assert_validation(
        &error,
        &["invalid lsp config item 'language'", "<key>=<value>"],
    );
}

#[test]
fn config_spec_rejects_empty_values() {
    for (spec, key) in [
        ("workspace=", "workspace"),
        ("language=", "language"),
        ("language_id=", "language_id"),
    ] {
        let error = LspToolConfig::from_str(spec).unwrap_err();
        assert_validation(&error, &[key, spec]);
    }
}

#[test]
fn config_spec_rejects_bad_values() {
    let error = LspToolConfig::from_str("request_timeout=15").unwrap_err();
    assert_validation(&error, &["request_timeout=15", "'ms', 's' or 'ns'"]);
    let error = LspToolConfig::from_str("request_timeout=fast").unwrap_err();
    assert_validation(&error, &["request_timeout=fast"]);
    let error = LspToolConfig::from_str("request_timeout=1.5s").unwrap_err();
    assert_validation(&error, &["request_timeout=1.5s", "'1.5'"]);
    let error = LspToolConfig::from_str("request_timeout=0s").unwrap_err();
    assert_validation(&error, &["request_timeout=0s", "greater than zero"]);
    let error = LspToolConfig::from_str("max_file_size=lots").unwrap_err();
    assert_validation(&error, &["max_file_size=lots", "whole number"]);
    let error = LspToolConfig::from_str("context_lines=-1").unwrap_err();
    assert_validation(&error, &["context_lines=-1"]);
    let error = LspToolConfig::from_str("max_file_size=4294967296x").unwrap_err();
    assert_validation(&error, &["max_file_size=4294967296x"]);
}

#[test]
fn config_display_renders_only_non_default_knobs() {
    assert_eq!(
        LspToolConfig::new().with_language("en").to_string(),
        "language=en"
    );
    assert_eq!(
        LspToolConfig::new()
            .with_workspace_root("/tmp/ws")
            .to_string(),
        "workspace=/tmp/ws"
    );
    assert_eq!(
        LspToolConfig::new()
            .with_context_lines(4)
            .with_max_file_size(64)
            .with_request_timeout(Duration::from_millis(750))
            .with_language_id("rust")
            .to_string(),
        "language_id=rust,request_timeout=750ms,max_file_size=64,context_lines=4"
    );
}

#[test]
fn config_display_round_trips_through_from_str() {
    let cases = [
        LspToolConfig::new(),
        LspToolConfig::new().with_language("en"),
        LspToolConfig::new().with_workspace_root("/tmp/ws"),
        LspToolConfig::new()
            .with_workspace_root("/tmp/a b/ws")
            .with_language("en")
            .with_language_id("rust")
            .with_request_timeout(Duration::from_secs(45))
            .with_max_file_size(1024)
            .with_context_lines(20),
        LspToolConfig::new().with_request_timeout(Duration::from_millis(1)),
        LspToolConfig::new().with_request_timeout(Duration::from_nanos(1_500_000_000_001)),
    ];
    for config in cases {
        let rendered = config.to_string();
        let parsed = LspToolConfig::from_str(&rendered)
            .unwrap_or_else(|error| panic!("{rendered:?} did not parse: {error}"));
        assert_config_eq(&parsed, &config);
        assert_eq!(parsed.to_string(), rendered);
    }
}

#[test]
fn action_parses_every_wire_name() {
    for action in LspAction::ALL {
        assert_eq!(LspAction::from_str(action.as_str()).unwrap(), action);
        let upper = action.as_str().to_ascii_uppercase();
        assert_eq!(LspAction::from_str(&upper).unwrap(), action);
        let kebab = action.as_str().replace('_', "-");
        assert_eq!(LspAction::from_str(&kebab).unwrap(), action);
        assert_eq!(LspAction::from_str(&format!(" {kebab} ")).unwrap(), action);
    }
}

#[test]
fn action_errors_name_every_accepted_value() {
    let error = LspAction::from_str("rename").unwrap_err();
    assert_validation(&error, &["rename"]);
    let message = error.message();
    for action in LspAction::ALL {
        assert!(
            message.contains(action.as_str()),
            "{message:?} does not list {}",
            action.as_str()
        );
    }
    let error = LspAction::from_str("").unwrap_err();
    assert_validation(&error, &["''", "hover"]);
}

#[test]
fn action_display_round_trips() {
    for action in LspAction::ALL {
        assert_eq!(action.to_string(), action.as_str());
        assert_eq!(LspAction::from_str(&action.to_string()).unwrap(), action);
    }
}

#[test]
fn args_builders_set_only_the_requested_fields() {
    let args = LspArgs::new(LspAction::Hover);
    assert_eq!(args.action, LspAction::Hover);
    assert!(args.path.is_none());
    assert!(args.line.is_none());
    assert!(args.column.is_none());
    assert!(args.query.is_none());
    assert!(args.context_lines.is_none());

    let args = LspArgs::new(LspAction::Definition)
        .with_path("src/lib.rs")
        .with_position(10, 5)
        .with_query("Thing")
        .with_context_lines(3);
    assert_eq!(args.path.as_deref(), Some("src/lib.rs"));
    assert_eq!(args.line, Some(10));
    assert_eq!(args.column, Some(5));
    assert_eq!(args.query.as_deref(), Some("Thing"));
    assert_eq!(args.context_lines, Some(3));
}

#[test]
fn args_parses_positional_forms() {
    let args = LspArgs::from_str("hover src/lib.rs:10:5").unwrap();
    assert_eq!(args.action, LspAction::Hover);
    assert_eq!(args.path.as_deref(), Some("src/lib.rs"));
    assert_eq!(args.line, Some(10));
    assert_eq!(args.column, Some(5));
    assert!(args.query.is_none());
    assert!(args.context_lines.is_none());

    let args = LspArgs::from_str("definition src/lib.rs:10").unwrap();
    assert_eq!(args.line, Some(10));
    assert!(args.column.is_none());

    let args = LspArgs::from_str("symbols src/lib.rs").unwrap();
    assert_eq!(args.action, LspAction::Symbols);
    assert!(args.line.is_none());
    assert!(args.column.is_none());

    let args = LspArgs::from_str("incoming-calls src/lib.rs:2:3").unwrap();
    assert_eq!(args.action, LspAction::IncomingCalls);
}

#[test]
fn args_parses_windows_and_absolute_paths() {
    let args = LspArgs::from_str(r"hover C:\dir\main.rs:10:5").unwrap();
    assert_eq!(args.path.as_deref(), Some(r"C:\dir\main.rs"));
    assert_eq!(args.line, Some(10));
    assert_eq!(args.column, Some(5));

    let args = LspArgs::from_str("hover /home/me/src/lib.rs:1").unwrap();
    assert_eq!(args.path.as_deref(), Some("/home/me/src/lib.rs"));
    assert_eq!(args.line, Some(1));

    // No position suffix: a colon inside the path stays part of the path.
    let args = LspArgs::from_str(r"hover C:\dir\main.rs").unwrap();
    assert_eq!(args.path.as_deref(), Some(r"C:\dir\main.rs"));
    assert!(args.line.is_none());
}

#[test]
fn args_parses_the_query_and_diagnostics_forms() {
    let args = LspArgs::from_str("workspace_symbols Foo Bar").unwrap();
    assert_eq!(args.action, LspAction::WorkspaceSymbols);
    assert_eq!(args.query.as_deref(), Some("Foo Bar"));
    assert!(args.path.is_none());

    let args = LspArgs::from_str("WORKSPACE-SYMBOLS").unwrap();
    assert_eq!(args.action, LspAction::WorkspaceSymbols);
    assert!(args.query.is_none());

    let args = LspArgs::from_str("diagnostics src/main.rs").unwrap();
    assert_eq!(args.action, LspAction::Diagnostics);
    assert_eq!(args.path.as_deref(), Some("src/main.rs"));

    let args = LspArgs::from_str("diagnostics").unwrap();
    assert_eq!(args.action, LspAction::Diagnostics);
    assert!(args.path.is_none());
}

#[test]
fn args_rejects_bad_actions_and_empty_specs() {
    assert_validation(
        &LspArgs::from_str("").unwrap_err(),
        &["empty lsp tool spec"],
    );
    assert_validation(
        &LspArgs::from_str("   ").unwrap_err(),
        &["empty lsp tool spec"],
    );
    assert_validation(&LspArgs::from_str("rename a.rs").unwrap_err(), &["rename"]);
}

#[test]
fn args_rejects_a_missing_or_malformed_path() {
    let error = LspArgs::from_str("hover").unwrap_err();
    assert_validation(&error, &["hover", "requires a path"]);
    let error = LspArgs::from_str("symbols").unwrap_err();
    assert_validation(&error, &["symbols", "requires a path"]);
    let error = LspArgs::from_str("hover :10").unwrap_err();
    assert_validation(&error, &["missing path", ":10"]);
}

#[test]
fn args_rejects_bad_positions() {
    let error = LspArgs::from_str("hover src/lib.rs:10:x").unwrap_err();
    assert_validation(&error, &["invalid column 'x'", "src/lib.rs:10:x"]);
    let error = LspArgs::from_str("hover src/lib.rs:0").unwrap_err();
    assert_validation(&error, &["invalid line '0'", "1-based"]);
    let error = LspArgs::from_str("hover src/lib.rs:10:0").unwrap_err();
    assert_validation(&error, &["invalid column '0'", "1-based"]);
    let error = LspArgs::from_str("hover src/lib.rs:4294967296").unwrap_err();
    assert_validation(&error, &["4294967296", "between 1 and 4294967295"]);
    let error = LspArgs::from_str("hover src/lib.rs:1:2:3").unwrap_err();
    assert_validation(&error, &["too many numeric position components"]);
}

#[test]
fn args_rejects_positions_on_diagnostics_and_trailing_tokens() {
    let error = LspArgs::from_str("diagnostics src/main.rs:10:5").unwrap_err();
    assert_validation(&error, &["diagnostics", "does not take a line/column"]);
    let error = LspArgs::from_str("hover src/lib.rs extra").unwrap_err();
    assert_validation(&error, &["unexpected trailing token 'extra'", "hover"]);
    let error = LspArgs::from_str("diagnostics src/main.rs extra").unwrap_err();
    assert_validation(&error, &["unexpected trailing token 'extra'"]);
}

#[test]
fn args_display_round_trips_the_canonical_forms() {
    let cases = [
        "hover src/lib.rs:10:5",
        "definition src/lib.rs:10",
        "symbols src/lib.rs",
        "workspace_symbols Foo",
        "workspace_symbols Foo Bar",
        "workspace_symbols",
        "diagnostics src/main.rs",
        "diagnostics",
        "incoming_calls /home/me/src/lib.rs:1:1",
        "prepare_call_hierarchy Cargo.toml:3",
    ];
    for case in cases {
        let args = LspArgs::from_str(case)
            .unwrap_or_else(|error| panic!("{case:?} did not parse: {error}"));
        assert_eq!(args.to_string(), case);
        let reparsed = LspArgs::from_str(&args.to_string()).unwrap();
        assert_eq!(reparsed.to_string(), args.to_string());
        assert_eq!(reparsed.action, args.action);
        assert_eq!(reparsed.path, args.path);
        assert_eq!(reparsed.line, args.line);
        assert_eq!(reparsed.column, args.column);
        assert_eq!(reparsed.query, args.query);
    }
}

#[test]
fn args_display_renders_only_the_fields_the_grammar_carries() {
    // A `workspace_symbols` value carries its query, not a path.
    let args = LspArgs::new(LspAction::WorkspaceSymbols)
        .with_path("src/lib.rs")
        .with_query("Thing");
    assert_eq!(args.to_string(), "workspace_symbols Thing");
    // A position action carries no query.
    let args = LspArgs::new(LspAction::Hover)
        .with_path("src/lib.rs")
        .with_query("Thing");
    assert_eq!(args.to_string(), "hover src/lib.rs");
    // A column without a line is not representable.
    let mut args = LspArgs::new(LspAction::Hover).with_path("src/lib.rs");
    args.column = Some(4);
    assert_eq!(args.to_string(), "hover src/lib.rs");
}

#[test]
fn serde_and_schema_shapes_are_unchanged() {
    for action in LspAction::ALL {
        assert_eq!(
            serde_json::to_value(action).unwrap(),
            json!(action.as_str())
        );
        assert_eq!(
            serde_json::from_value::<LspAction>(json!(action.as_str())).unwrap(),
            action
        );
    }

    // Unset fields stay out of the wire payload.
    assert_eq!(
        serde_json::to_value(LspArgs::new(LspAction::Hover)).unwrap(),
        json!({"action": "hover"})
    );
    let args = LspArgs::from_str("hover src/lib.rs:10:5").unwrap();
    assert_eq!(
        serde_json::to_value(&args).unwrap(),
        json!({"action": "hover", "path": "src/lib.rs", "line": 10, "column": 5})
    );

    // Unknown fields stay tolerated, and the documented defaults still apply.
    let parsed: LspArgs = serde_json::from_value(json!({
        "action": "workspace_symbols",
        "query": "Foo",
        "path": "ignored.rs",
        "unexpected": true,
    }))
    .unwrap();
    assert_eq!(parsed.action, LspAction::WorkspaceSymbols);
    assert_eq!(parsed.query.as_deref(), Some("Foo"));
    assert_eq!(parsed.path.as_deref(), Some("ignored.rs"));

    // The card keeps its six input properties and the action enum.
    let card = LspTool::card_for("en");
    let properties = card.input_params["properties"].as_object().unwrap();
    let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["action", "column", "context_lines", "line", "path", "query"]
    );
    let names: Vec<&str> = LspAction::ALL
        .iter()
        .map(|action| action.as_str())
        .collect();
    assert_eq!(
        properties["action"]["enum"],
        serde_json::to_value(names).unwrap()
    );
}
