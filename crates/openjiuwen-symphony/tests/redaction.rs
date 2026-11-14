//! End-to-end tests for credential/PII redaction.
//!
//! Everything is offline and deterministic: no model, no network, and no
//! clock-dependent assertions. The suite covers every public entry point of
//! [`openjiuwen_symphony::redaction`] plus the flow pipeline boundary, where a
//! scripted model reply and a trace payload carrying fake credentials must not
//! reach a distilled recipe or a `SkillPack` material.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp, clippy::assert_is_empty)]

use std::sync::Arc;

use openjiuwen_retrieval::StaticCompletionModel;
use openjiuwen_symphony::flow::FlowEngine;
use openjiuwen_symphony::redaction::{
    is_sensitive_name, normalize_sensitive_name, redact_common_credentials, redact_private_json,
    redact_sensitive_json, redact_sensitive_text, sanitize_distilled_text, DISTILLED_TEXT_LIMIT,
    REDACTED, REDACTED_CREDENTIAL, REDACTED_REQUEST,
};
use serde_json::{json, Value};
use tempfile::TempDir;

/// A realistic-looking, entirely fake PEM block.
const FAKE_PEM: &str = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ\n-----END OPENSSH PRIVATE KEY-----";
/// A realistic-looking, entirely fake API key.
const FAKE_API_KEY: &str = "sk-live-0123456789abcdef";
/// A realistic-looking, entirely fake bearer token.
const FAKE_BEARER: &str = "abcdefgh12345678";
/// A realistic-looking, entirely fake AWS access key id.
const FAKE_AWS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";

#[test]
fn constants_are_stable() {
    assert_eq!(REDACTED, "<redacted>");
    assert_eq!(REDACTED_CREDENTIAL, "<redacted-credential>");
    assert_eq!(REDACTED_REQUEST, "<redacted-request>");
    assert_eq!(DISTILLED_TEXT_LIMIT, 2_000);
}

#[test]
fn every_documented_pattern_is_redacted() {
    let pem = redact_sensitive_text(&format!("before\n{FAKE_PEM}\nafter"));
    assert_eq!(pem, format!("before\n{REDACTED_CREDENTIAL}\nafter"));

    let bearer = redact_sensitive_text(&format!("GET /docs\nBearer {FAKE_BEARER}"));
    assert_eq!(bearer, format!("GET /docs\n{REDACTED_CREDENTIAL}"));

    let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl";
    assert_eq!(
        redact_sensitive_text(&format!("jwt {jwt} end")),
        format!("jwt {REDACTED_CREDENTIAL} end")
    );

    assert_eq!(
        redact_sensitive_text(&format!("key {FAKE_API_KEY} end")),
        format!("key {REDACTED_CREDENTIAL} end")
    );

    assert_eq!(
        redact_sensitive_text(&format!("aws {FAKE_AWS_KEY} end")),
        format!("aws {REDACTED_CREDENTIAL} end")
    );

    for assignment in [
        format!("api_key: {FAKE_API_KEY}"),
        format!("api_key={FAKE_API_KEY}"),
        format!("\"api_key\": \"{FAKE_API_KEY}\""),
        format!("'api_key': '{FAKE_API_KEY}'"),
        format!("API-Key := {FAKE_API_KEY}"),
    ] {
        let redacted = redact_sensitive_text(&assignment);
        assert!(
            redacted.contains(REDACTED),
            "{assignment} -> {redacted} should carry the marker"
        );
        assert!(
            !redacted.contains(FAKE_API_KEY),
            "{assignment} leaked the key"
        );
    }

    let header = redact_sensitive_text(&format!("Authorization: Bearer {FAKE_BEARER}"));
    assert_eq!(header, "Authorization=<redacted>");
    assert_eq!(
        redact_sensitive_text("password: hunter2\nhost: db.local"),
        "password=<redacted>\nhost: db.local"
    );
}

#[test]
fn surrounding_text_and_line_structure_survive() {
    // An unquoted assignment value runs to `;`, `,`, a bracket, or the end of
    // the line (Python's `_ASSIGNMENT` alternation), so the line break and the
    // following line survive untouched.
    let text = "step 1: read the docs\nstep 2: api_key=abc123def456; still here\nstep 3: done";
    let redacted = redact_sensitive_text(text);
    assert_eq!(
        redacted,
        "step 1: read the docs\nstep 2: api_key=<redacted>; still here\nstep 3: done"
    );
    assert_eq!(redacted.lines().count(), 3);
    assert_eq!(redact_sensitive_text("token: abc ok"), "token=<redacted>");
}

#[test]
fn recognition_is_conservative() {
    for text in [
        "query: user input",
        "timeout=30",
        "token_count: 4",
        "secret_name: demo",
        "note: nothing to hide",
        "用法：先检索，再总结。",
    ] {
        assert_eq!(redact_sensitive_text(text), text);
    }
    assert!(!is_sensitive_name("token_count"));
    assert!(is_sensitive_name("client_secret"));
    assert_eq!(
        normalize_sensitive_name("AWSSecretAccessKey"),
        "aws_secret_access_key"
    );
}

#[test]
fn redaction_is_idempotent() {
    let samples = [
        format!("api_key: {FAKE_API_KEY}, password = hunter2"),
        format!("Authorization: Bearer {FAKE_BEARER}"),
        format!("pem: {FAKE_PEM}"),
        format!("aws {FAKE_AWS_KEY} and sk-proj-0123456789abcdef"),
        "secret = \"quoted value\" ; trailing".to_string(),
        "密码=秘密 token: abc".to_string(),
    ];
    for sample in samples {
        let once = redact_sensitive_text(&sample);
        assert_eq!(
            once,
            redact_sensitive_text(&once),
            "not idempotent: {sample}"
        );
        assert_eq!(
            once,
            redact_common_credentials(&redact_common_credentials(&once))
        );
    }
}

#[test]
fn non_ascii_input_never_panics_and_round_trips() {
    let long_accented = "é".repeat(500);
    for text in [
        "🔐 password=🔑",
        "密码 api_key=秘密",
        long_accented.as_str(),
        "-----BEGIN 秘密 PRIVATE KEY-----",
    ] {
        let redacted = redact_sensitive_text(text);
        assert_eq!(redacted, redact_sensitive_text(&redacted));
    }
}

#[test]
fn common_credentials_cover_bare_secrets_only() {
    assert_eq!(
        redact_common_credentials(&format!("aws {FAKE_AWS_KEY}")),
        format!("aws {REDACTED_CREDENTIAL}")
    );
    // Assignment-shaped secrets are the text scanner's job, not this pass.
    assert_eq!(
        redact_common_credentials("password: hunter2"),
        "password: hunter2"
    );
    // Too short to be a credential.
    assert_eq!(redact_common_credentials("sk-abc"), "sk-abc");
}

#[test]
fn json_redaction_preserves_shape_and_order() {
    let value = json!({
        "package_id": "cap-0123456789ab",
        "description": "run with api_key: sk-live-0123456789abcdef",
        "steps": [
            {"id": "search", "inputs": [{"name": "query", "type": "string"}]},
            {"id": "summarize", "outputs": [{"name": "summary", "type": "text"}]},
        ],
        "count": 2,
        "flag": false,
        "nothing": null,
    });
    let redacted = redact_sensitive_json(&value);
    assert_eq!(redacted["description"], "run with api_key=<redacted>");
    assert_eq!(redacted["steps"][0]["id"], "search");
    assert_eq!(redacted["count"], 2);
    assert_eq!(redacted["flag"], false);
    assert_eq!(redacted["nothing"], Value::Null);
    let keys: Vec<&str> = redacted
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        [
            "package_id",
            "description",
            "steps",
            "count",
            "flag",
            "nothing"
        ]
    );

    let secrets = json!({
        "api_key": {"nested": "whole value"},
        "client_secret": ["a", "b"],
        "list": [{"password": "x"}, "token: y", 5],
    });
    let redacted = redact_sensitive_json(&secrets);
    assert_eq!(redacted["api_key"], REDACTED);
    assert_eq!(redacted["client_secret"], REDACTED);
    assert_eq!(redacted["list"][0]["password"], REDACTED);
    assert_eq!(redacted["list"][1], "token=<redacted>");
    assert_eq!(redacted["list"][2], 5);

    // Repeating the pass changes nothing.
    assert_eq!(redact_sensitive_json(&redacted), redacted);
}

#[test]
fn json_depth_is_capped() {
    let mut value = json!("leaf");
    for _ in 0..80 {
        value = Value::Array(vec![value]);
    }
    let redacted = redact_sensitive_json(&value);
    assert!(redacted.to_string().contains(REDACTED));
    assert_eq!(redacted, redact_sensitive_json(&redacted));
}

#[test]
fn private_json_redacts_caller_named_keys() {
    let value = json!({
        "userEmail": "a@b.c",
        "phone_number": "+10000000000",
        "note": "api_key: sk-live-0123456789abcdef",
        "kept": 1,
    });
    let redacted = redact_private_json(&value, &["user_email", "phone-number"]);
    assert_eq!(redacted["userEmail"], REDACTED);
    assert_eq!(redacted["phone_number"], REDACTED);
    assert_eq!(redacted["note"], "api_key=<redacted>");
    assert_eq!(redacted["kept"], 1);
    // Sensitive keys are redacted even when the caller names nothing.
    let redacted = redact_private_json(&json!({"token": "abc"}), &[]);
    assert_eq!(redacted["token"], REDACTED);
}

#[test]
fn sanitize_orders_passes_and_truncates() {
    // Credentials first, then the verbatim request, then the length cut.
    let text = "  ask about api_key: sk-live-0123456789abcdef  ";
    assert_eq!(
        sanitize_distilled_text(text, Some("ask about api_key: sk-live-0123456789abcdef")),
        REDACTED_REQUEST
    );
    assert_eq!(
        sanitize_distilled_text("api_key: sk-live-0123456789abcdef", None),
        "api_key=<redacted>"
    );
    assert_eq!(
        sanitize_distilled_text("   summarize it   ", None),
        "summarize it"
    );

    let long = "密".repeat(DISTILLED_TEXT_LIMIT + 25);
    let truncated = sanitize_distilled_text(&long, None);
    assert_eq!(truncated.chars().count(), DISTILLED_TEXT_LIMIT);
    assert!(truncated.chars().all(|character| character == '密'));

    assert_eq!(sanitize_distilled_text("", Some("query")), "");
    assert_eq!(sanitize_distilled_text("text", Some("   ")), "text");
}

/// Builds one capability node payload.
fn node(description: &str, inputs: Value, outputs: Value) -> Value {
    json!({
        "label": "capability",
        "metadata": {
            "capability_type": "skill",
            "skill_type": "skill",
            "version": "1.0.0",
            "description": description,
            "inputs": inputs,
            "outputs": outputs,
        },
    })
}

#[tokio::test]
async fn a_flow_run_keeps_credentials_out_of_recipes_and_packs() {
    let dir = TempDir::new().unwrap();

    // The model reply echoes a fake API key and a fake PEM block.
    let reply = json!({
        "name": "pem-and-key-flow",
        "task_description": format!("rotate the api_key: {FAKE_API_KEY} for search"),
        "trigger_conditions": format!("the user pastes a key:\n{FAKE_PEM}"),
        "example_requests": ["use password: hunter2 to login".to_string()],
        "execution_narrative": format!("step 1 passes Bearer {FAKE_BEARER} and api_key: {FAKE_API_KEY}"),
    })
    .to_string();
    let engine = FlowEngine::open(dir.path())
        .await
        .unwrap()
        .with_model(Arc::new(StaticCompletionModel::single(reply)));

    let trace = json!({
        "trace_id": "t1",
        "query": format!("rotate the api_key: {FAKE_API_KEY}"),
        "outcome": "success",
        "graph": {
            "id": "g1",
            "type": "execution_graph",
            "label": "run",
            "directed": true,
            "nodes": {
                "search": node(
                    &format!("search with password: hunter2 and {FAKE_PEM}"),
                    json!([{"name": "query", "type": "string", "required": true, "description": format!("input token: {FAKE_API_KEY}")}]),
                    json!([{"name": "results", "type": "array", "required": true, "description": "ranked hits"}]),
                ),
                "summarize": node(
                    &format!("summarize with Bearer {FAKE_BEARER}"),
                    json!([{"name": "results", "type": "array", "required": true, "description": "ranked hits"}]),
                    json!([{"name": "summary", "type": "text", "required": false, "description": "final answer"}]),
                ),
            },
            "edges": [{"source": "search", "target": "summarize", "relation": "can_feed", "metadata": {"success": true}}],
        },
    });
    assert!(engine.ingest_value(trace).await.unwrap());

    let report = engine.distill().await.unwrap();
    assert_eq!(report.recipes_saved.len(), 1);
    assert!(report.narrative_fallbacks.is_empty());
    let recipe = engine.recipes().await.into_iter().next().unwrap();

    let narrative = recipe.execution_narrative.clone();
    assert!(narrative.contains(REDACTED), "{narrative}");
    assert!(!narrative.contains(FAKE_API_KEY));
    assert!(!narrative.contains(FAKE_BEARER));
    // The model's own task description repeats the request, so the verbatim
    // request replacement claims it before the length cut.
    let task_description = recipe.applicability_text("task_description");
    assert!(
        task_description.contains(REDACTED) || task_description.contains(REDACTED_REQUEST),
        "{task_description}"
    );
    assert!(recipe
        .applicability_text("trigger_conditions")
        .contains(REDACTED_CREDENTIAL));
    let examples = serde_json::to_string(&recipe.applicability["example_requests"]).unwrap();
    assert!(!examples.contains("hunter2"));

    let pack = engine.pack(&recipe.recipe_id).await.unwrap();
    let materials = serde_json::to_string(&pack.materials()).unwrap();
    assert!(materials.contains(REDACTED));
    assert!(materials.contains(REDACTED_CREDENTIAL));
    for secret in [
        FAKE_API_KEY,
        FAKE_BEARER,
        FAKE_AWS_KEY,
        "hunter2",
        "b3BlbnNzaC1rZXk",
    ] {
        assert!(!materials.contains(secret), "{secret} leaked into the pack");
    }
    for (name, text) in pack.render() {
        for secret in [FAKE_API_KEY, FAKE_BEARER, "hunter2", "b3BlbnNzaC1rZXk"] {
            assert!(!text.contains(secret), "{name} leaked {secret}");
        }
        let redacted = redact_sensitive_text(&text);
        assert!(
            text == redacted || text.contains(REDACTED) || text.contains(REDACTED_CREDENTIAL),
            "{name} still carries credential material"
        );
    }

    // The pack review agrees: nothing left to redact.
    assert!(redact_private_json(&pack.materials(), &[]) == pack.materials());
}
