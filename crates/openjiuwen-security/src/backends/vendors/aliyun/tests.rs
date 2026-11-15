//! Wire-contract, configuration, and response-parsing tests for the Aliyun
//! Green adapter.
//!
//! Everything here runs offline against the loopback `wiremock` server, the
//! deterministic signing helpers, and hand-built `TextModeration` replies.

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;

use super::*;
use crate::backends::vendors::aliyun::signing::{
    sha256_hex, sign_authorization, SIGNED_HEADER_NAMES,
};
use crate::ScanStage;

/// Distinctive text the tests assert never leaks into details or errors.
const SCANNED_TEXT: &str = "sensitive sample text 42";
const BEARER_TOKEN: &str = "green-bearer-token";
const ACCESS_KEY_ID: &str = "test-access-key-id";
const ACCESS_KEY_SECRET: &str = "test-access-key-secret";

fn offline_backend() -> AliyunGreenBackend {
    AliyunGreenBackend::new(AliyunGreenConfig::default()).unwrap()
}

fn backend_for(server: &MockServer, auth: AliyunGreenAuth) -> AliyunGreenBackend {
    AliyunGreenBackend::new(
        AliyunGreenConfig::default()
            .with_endpoint(server.uri())
            .with_auth(auth)
            .with_max_retries(0),
    )
    .unwrap()
}

async fn server_with_reply(reply: &Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply))
        .expect(1)
        .mount(&server)
        .await;
    server
}

fn header<'a>(request: &'a wiremock::Request, name: &str) -> Option<&'a str> {
    request.headers.get(name)?.to_str().ok()
}

fn recorded_reply(confidence: f64) -> Value {
    json!({
        "Code": 200, "Message": "OK", "RequestId": "req-aliyun-1",
        "Data": {
            "Labels": "political_content,profanity",
            "Reason": "vendor reason that must stay private",
            "Result": [{
                "Label": "political_content",
                "Confidence": confidence,
                "RiskWords": "secret-risk-word"
            }]
        }
    })
}

fn ctx(text: &str) -> GuardrailContext {
    GuardrailContext::text(text, ScanStage::UserInput)
}

/// Asserts the headers and body hash every signed request carries.
fn assert_signed_request(sent: &wiremock::Request, host: &str) {
    assert_eq!(sent.method.as_str(), "POST");
    assert_eq!(sent.url.path(), "/");
    assert_eq!(header(sent, "x-acs-action"), Some(ACTION));
    assert_eq!(header(sent, "x-acs-version"), Some(DEFAULT_API_VERSION));
    assert_eq!(header(sent, "host"), Some(host));
    let date = header(sent, "x-acs-date").expect("date");
    assert!(
        chrono::DateTime::parse_from_rfc3339(date).is_ok(),
        "date {date}"
    );
    let nonce = header(sent, "x-acs-signature-nonce").expect("nonce");
    assert!(uuid::Uuid::parse_str(nonce).is_ok(), "nonce {nonce}");
    let expected_sha = sha256_hex(&sent.body);
    assert_eq!(
        header(sent, "x-acs-content-sha256"),
        Some(expected_sha.as_str())
    );
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["Service"], json!(DEFAULT_SERVICE));
    let parameters = body["ServiceParameters"].as_str().expect("encoded");
    assert_eq!(parameters, json!({"content": SCANNED_TEXT}).to_string());
    assert_eq!(
        serde_json::from_str::<Value>(parameters).unwrap()["content"],
        json!(SCANNED_TEXT)
    );
}

#[test]
fn config_defaults_serde_auth_and_debug() {
    let defaults = AliyunGreenConfig::default();
    assert_eq!(defaults.endpoint, DEFAULT_ENDPOINT);
    assert_eq!(defaults.auth, AliyunGreenAuth::None);
    assert_eq!(defaults.auth.kind(), "none");
    assert_eq!(defaults.service, DEFAULT_SERVICE);
    assert_eq!(defaults.api_version, DEFAULT_API_VERSION);
    assert_eq!(defaults.region, "cn-shanghai");
    assert!((defaults.timeout - 30.0).abs() < f64::EPSILON);
    assert_eq!(defaults.max_retries, 2);
    assert_eq!(defaults.risk_type, DEFAULT_RISK_TYPE);

    let config = defaults
        .with_endpoint("https://green.example/")
        .with_auth(AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)))
        .with_service("text_moderation")
        .with_api_version("2023-01-01")
        .with_region("cn-hangzhou")
        .with_timeout(12.5)
        .with_max_retries(3);
    let serialized = serde_json::to_string(&config).unwrap();
    assert!(serialized.contains("\"timeout\":12.5"), "{serialized}");
    assert!(serialized.contains("\"service\":\"text_moderation\""));
    let restored: AliyunGreenConfig = serde_json::from_str(&serialized).unwrap();
    assert_eq!(restored, config);

    let partial: AliyunGreenConfig = serde_json::from_str(
        r#"{"endpoint": "https://green.example/", "vendor": {"name": "aliyun"}}"#,
    )
    .unwrap();
    assert_eq!(partial.endpoint, "https://green.example/");
    assert_eq!(partial.auth, AliyunGreenAuth::None);
    assert_eq!(partial.service, DEFAULT_SERVICE);
    assert_eq!(partial.max_retries, 2);

    // Blank service/version/region fall back to their defaults.
    let backend = AliyunGreenBackend::new(
        AliyunGreenConfig::default()
            .with_service("  ")
            .with_api_version("")
            .with_region(" "),
    )
    .unwrap();
    assert_eq!(backend.config().service, DEFAULT_SERVICE);
    assert_eq!(backend.config().api_version, DEFAULT_API_VERSION);
    assert_eq!(backend.config().region, DEFAULT_REGION);
    assert_eq!(backend.name(), NAME);

    // Every auth variant round-trips and never prints its secrets.
    for auth in [
        AliyunGreenAuth::None,
        AliyunGreenAuth::Bearer(SecretString::from("bearer-secret-value")),
        AliyunGreenAuth::Aksk {
            access_key_id: SecretString::from("ak-id-secret"),
            access_key_secret: SecretString::from("ak-secret-value"),
        },
    ] {
        let serialized = serde_json::to_string(&auth).unwrap();
        let restored: AliyunGreenAuth = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored, auth);
        let debug = format!("{auth:?}");
        assert!(!debug.contains("bearer-secret-value"), "{debug}");
        assert!(!debug.contains("ak-id-secret"), "{debug}");
        assert!(!debug.contains("ak-secret-value"), "{debug}");
    }
    assert_eq!(
        AliyunGreenAuth::Bearer(SecretString::from("x")).kind(),
        "bearer"
    );
    assert_eq!(
        AliyunGreenAuth::Aksk {
            access_key_id: SecretString::from("x"),
            access_key_secret: SecretString::from("y"),
        }
        .kind(),
        "aksk"
    );
}

#[test]
fn debug_never_prints_credentials() {
    let config = AliyunGreenConfig::default()
        .with_auth(AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)));
    let debug = format!("{config:?}");
    assert!(
        debug.contains("green-cip.cn-shanghai.aliyuncs.com"),
        "{debug}"
    );
    assert!(!debug.contains(BEARER_TOKEN), "{debug}");
    let debug = format!("{:?}", AliyunGreenBackend::new(config).unwrap());
    assert!(debug.contains("AliyunGreenBackend"), "{debug}");
    assert!(!debug.contains(BEARER_TOKEN), "{debug}");

    let config = AliyunGreenConfig::default().with_auth(AliyunGreenAuth::Aksk {
        access_key_id: SecretString::from(ACCESS_KEY_ID),
        access_key_secret: SecretString::from(ACCESS_KEY_SECRET),
    });
    let debug = format!("{:?}", AliyunGreenBackend::new(config).unwrap());
    assert!(debug.contains("aksk"), "{debug}");
    assert!(!debug.contains(ACCESS_KEY_ID), "{debug}");
    assert!(!debug.contains(ACCESS_KEY_SECRET), "{debug}");
}

#[test]
fn new_rejects_blank_endpoint_and_blank_aksk() {
    let error =
        AliyunGreenBackend::new(AliyunGreenConfig::default().with_endpoint("   ")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    for (access_key_id, access_key_secret) in [
        ("", ACCESS_KEY_SECRET),
        (ACCESS_KEY_ID, ""),
        ("  ", ACCESS_KEY_SECRET),
    ] {
        let error = AliyunGreenBackend::new(AliyunGreenConfig::default().with_auth(
            AliyunGreenAuth::Aksk {
                access_key_id: SecretString::from(access_key_id),
                access_key_secret: SecretString::from(access_key_secret),
            },
        ))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    // A blank bearer token is allowed and omitted from the request.
    let backend = AliyunGreenBackend::new(
        AliyunGreenConfig::default().with_auth(AliyunGreenAuth::Bearer(SecretString::from(""))),
    )
    .unwrap();
    assert_eq!(backend.config().auth.kind(), "bearer");
}

#[tokio::test]
async fn bearer_request_matches_the_rpc_contract() {
    let server = server_with_reply(&recorded_reply(95.0)).await;
    let backend = backend_for(
        &server,
        AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)),
    );
    let assessment = backend.analyze(&ctx(SCANNED_TEXT)).await.unwrap();
    assert!(assessment.has_risk);
    let details = assessment.details.expect("details");
    assert!(!details.to_string().contains(SCANNED_TEXT));
    assert!(!details.to_string().contains("secret-risk-word"));

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    assert_signed_request(&received[0], &server.address().to_string());
    assert_eq!(
        header(&received[0], "authorization"),
        Some(format!("Bearer {BEARER_TOKEN}").as_str())
    );
}

#[tokio::test]
async fn aksk_request_carries_the_acs3_signature() {
    let server = server_with_reply(&recorded_reply(80.0)).await;
    let backend = backend_for(
        &server,
        AliyunGreenAuth::Aksk {
            access_key_id: SecretString::from(ACCESS_KEY_ID),
            access_key_secret: SecretString::from(ACCESS_KEY_SECRET),
        },
    );
    backend.analyze(&ctx(SCANNED_TEXT)).await.unwrap();

    let received = server.received_requests().await.unwrap();
    let sent = &received[0];
    let host = server.address().to_string();
    assert_signed_request(sent, &host);
    let authorization = header(sent, "authorization").expect("authorization");
    assert!(
        authorization.starts_with("ACS3-HMAC-SHA256 Credential="),
        "{authorization}"
    );
    assert!(
        authorization.contains(&format!(
            "Credential={ACCESS_KEY_ID},SignedHeaders={SIGNED_HEADER_NAMES},Signature="
        )),
        "{authorization}"
    );
    assert!(!authorization.contains(ACCESS_KEY_SECRET));
    let signature = authorization.split("Signature=").nth(1).expect("signature");
    assert_eq!(signature.len(), 64, "{authorization}");
    assert!(
        signature
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "{authorization}"
    );

    // The received header matches the deterministic helper for the same inputs.
    assert_eq!(
        authorization,
        sign_authorization(
            ACCESS_KEY_ID,
            ACCESS_KEY_SECRET,
            &host,
            DEFAULT_API_VERSION,
            header(sent, "x-acs-date").expect("date"),
            header(sent, "x-acs-signature-nonce").expect("nonce"),
            header(sent, "x-acs-content-sha256").expect("sha"),
        )
    );
}

#[tokio::test]
async fn empty_text_skips_requests_and_status_errors_are_transport() {
    let server = MockServer::start().await;
    let backend = backend_for(
        &server,
        AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)),
    );
    for text in ["", "   ", "\n\t "] {
        let assessment = backend.analyze(&ctx(text)).await.unwrap();
        assert!(!assessment.has_risk, "text {text:?}");
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert!(assessment.risk_type.is_none());
        assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
    }
    assert!(server.received_requests().await.unwrap().is_empty());

    // A blank bearer token is omitted rather than sent empty.
    let server = server_with_reply(&recorded_reply(95.0)).await;
    let backend = backend_for(&server, AliyunGreenAuth::Bearer(SecretString::from("")));
    backend.analyze(&ctx(SCANNED_TEXT)).await.unwrap();
    let received = server.received_requests().await.unwrap();
    assert_eq!(header(&received[0], "authorization"), None);

    // A non-2xx status is a transport error that never quotes the text.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(403).set_body_json(json!({"Code": 403, "Message": "forbidden"})),
        )
        .mount(&server)
        .await;
    let error = backend_for(
        &server,
        AliyunGreenAuth::Bearer(SecretString::from(BEARER_TOKEN)),
    )
    .analyze(&ctx(SCANNED_TEXT))
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(!error.to_string().contains(SCANNED_TEXT), "{error}");
}

#[test]
fn parse_response_levels_details_and_vendor_content() {
    for (confidence, expected) in [
        (95.0, RiskLevel::Critical),
        (90.0, RiskLevel::Critical),
        (89.9, RiskLevel::High),
        (80.0, RiskLevel::High),
        (75.0, RiskLevel::High),
        (60.0, RiskLevel::Medium),
        (50.0, RiskLevel::Medium),
        (49.9, RiskLevel::Low),
        (20.0, RiskLevel::Low),
    ] {
        let assessment = offline_backend()
            .parse_response(&recorded_reply(confidence))
            .unwrap();
        assert!(assessment.has_risk);
        assert_eq!(assessment.risk_level, expected, "confidence {confidence}");
        assert_eq!(assessment.risk_type.as_deref(), Some("political_content"));
        let expected_reported = round_confidence(confidence / 100.0);
        assert!((assessment.confidence - expected_reported).abs() < f64::EPSILON);
    }

    let assessment = offline_backend()
        .parse_response(&recorded_reply(95.555_5))
        .unwrap();
    let details = assessment.details.expect("details");
    assert_eq!(details["provider"], json!("aliyun"));
    assert_eq!(details["service"], json!(DEFAULT_SERVICE));
    assert_eq!(details["region"], json!(DEFAULT_REGION));
    assert_eq!(details["labels"], json!(["political_content", "profanity"]));
    assert_eq!(details["request_id"], json!("req-aliyun-1"));
    let confidence = details["confidence"].as_f64().expect("confidence");
    assert!((confidence - 0.9556).abs() < 1e-9, "{confidence}");
    let serialized = details.to_string();
    assert!(!serialized.contains("secret-risk-word"), "{serialized}");
    assert!(!serialized.contains("vendor reason"), "{serialized}");
    assert!(!serialized.contains("RiskWords"), "{serialized}");
    assert!(!serialized.contains("Reason"), "{serialized}");

    // The highest confidence wins across results; labels without scores are high.
    let reply = json!({"Code": 200, "Data": {"Labels": "profanity", "Result": [
        {"Confidence": 20.0}, {"Confidence": 88.5}, {"Confidence": 60.0}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert_eq!(assessment.risk_level, RiskLevel::High);
    assert_eq!(assessment.risk_type.as_deref(), Some("profanity"));
    assert!((assessment.confidence - 0.885).abs() < f64::EPSILON);

    let assessment = offline_backend()
        .parse_response(&json!({"Code": 200, "Data": {"Labels": "political_content"}}))
        .unwrap();
    assert_eq!(assessment.risk_level, RiskLevel::High);
    assert!((assessment.confidence - 1.0).abs() < f64::EPSILON);
    let details = assessment.details.expect("details");
    assert_eq!(details["confidence"].as_f64(), Some(1.0));
    assert!(details.get("request_id").is_none());
}

#[test]
fn parse_response_result_labels_are_authoritative() {
    // The RPC may omit `Labels` and report only the per-label detail list.
    let reply = json!({"Code": 200, "RequestId": "req-result-only", "Data": {"Result": [
        {"Label": "political_content", "Confidence": 99, "RiskWords": "secret-risk-word",
         "Reason": "vendor reason that must stay private"}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert!(assessment.has_risk);
    assert_eq!(assessment.risk_level, RiskLevel::Critical);
    assert_eq!(assessment.risk_type.as_deref(), Some("political_content"));
    assert!((assessment.confidence - 0.99).abs() < f64::EPSILON);
    let details = assessment.details.expect("details");
    assert_eq!(details["labels"], json!(["political_content"]));
    assert_eq!(details["request_id"], json!("req-result-only"));
    let serialized = details.to_string();
    assert!(!serialized.contains("secret-risk-word"), "{serialized}");
    assert!(!serialized.contains("vendor reason"), "{serialized}");
    assert!(!serialized.contains("RiskWords"), "{serialized}");
    assert!(!serialized.contains("Reason"), "{serialized}");

    // Regression: an empty `Labels` plus a scored `Result` entry is risky.
    let reply = json!({"Code": 200, "Data": {"Labels": "", "Result": [
        {"Label": "profanity", "Confidence": 60.0}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert!(assessment.has_risk);
    assert_eq!(assessment.risk_level, RiskLevel::Medium);
    assert_eq!(assessment.risk_type.as_deref(), Some("profanity"));
    assert!((assessment.confidence - 0.6).abs() < f64::EPSILON);
    assert_eq!(
        assessment.details.expect("details")["labels"],
        json!(["profanity"])
    );

    // The union keeps `Labels` first and drops duplicates.
    let reply = json!({"Code": 200, "Data": {"Labels": "abuse", "Result": [
        {"Label": "abuse", "Confidence": 80.0}, {"Label": "profanity", "Confidence": 80.0}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert!(assessment.has_risk);
    assert_eq!(assessment.risk_type.as_deref(), Some("abuse"));
    let details = assessment.details.expect("details");
    assert_eq!(details["labels"], json!(["abuse", "profanity"]));
    assert_eq!(details["labels"].as_array().map(Vec::len), Some(2));

    // `nonLabel` never masks a real `Result` label.
    let reply = json!({"Code": 200, "Data": {"Labels": "nonLabel", "Result": [
        {"Label": " abuse ", "Confidence": 88.0}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert!(assessment.has_risk);
    assert_eq!(assessment.risk_type.as_deref(), Some("abuse"));
    assert_eq!(
        assessment.details.expect("details")["labels"],
        json!(["abuse"])
    );

    // Blank and `nonLabel` `Result` labels are dropped; an unscored or
    // zero-scored `Result` entry is not risk on its own.
    let reply = json!({"Code": 200, "Data": {"Result": [
        {"Label": " ", "Confidence": 0.0}, {"Label": "nonLabel"}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert!(!assessment.has_risk);
    assert_eq!(assessment.risk_level, RiskLevel::Safe);
    assert!(assessment.details.is_none());
}

#[test]
fn parse_response_scored_result_without_labels_uses_config_risk_type() {
    let reply = json!({"Code": 200, "Data": {"Labels": "nonLabel", "Result": [
        {"Label": "", "Confidence": 72.5}
    ]}});
    let assessment = offline_backend().parse_response(&reply).unwrap();
    assert!(assessment.has_risk);
    assert_eq!(assessment.risk_level, RiskLevel::Medium);
    assert_eq!(assessment.risk_type.as_deref(), Some(DEFAULT_RISK_TYPE));
    assert!((assessment.confidence - 0.725).abs() < f64::EPSILON);
    let details = assessment.details.expect("details");
    assert_eq!(details["labels"], json!([]));

    let backend = AliyunGreenBackend::new(AliyunGreenConfig {
        risk_type: "aliyun_green_fallback".to_owned(),
        ..AliyunGreenConfig::default()
    })
    .unwrap();
    let assessment = backend.parse_response(&reply).unwrap();
    assert_eq!(
        assessment.risk_type.as_deref(),
        Some("aliyun_green_fallback")
    );
}

#[test]
fn parse_response_safe_and_code_errors() {
    for labels in [
        json!(""),
        json!("   "),
        json!("nonLabel"),
        json!(" nonLabel ,  "),
        json!(",,"),
        json!("NONLABEL"),
    ] {
        let reply = json!({"Code": 200, "Data": {"Labels": labels, "Result": []}});
        let assessment = offline_backend().parse_response(&reply).unwrap();
        assert!(!assessment.has_risk, "labels {labels:?}");
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert!(assessment.risk_type.is_none());
        assert!(assessment.details.is_none());
    }
    for reply in [json!({"Code": 200, "Data": {"Result": []}})] {
        let assessment = offline_backend().parse_response(&reply).unwrap();
        assert!(!assessment.has_risk, "reply {reply}");
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
    }

    for reply in [json!({"Code": 500, "Message": "internal error"})] {
        let error = offline_backend().parse_response(&reply).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport, "reply {reply}");
    }
    let error = offline_backend()
        .parse_response(&json!({"Code": 500, "Message": "bad\nrequest"}))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("aliyun green returned code 500: bad request"),
        "{error}"
    );
}

#[test]
fn unrecognized_reply_shapes_are_validation_errors() {
    for reply in [
        json!({}),
        json!({"Message": "missing code"}),
        json!({"Code": "200"}),
        json!({"Code": 500.5}),
        json!({"Code": null}),
        json!({"Code": 200}),
        json!({"Code": 200, "Data": {}}),
        json!({"Code": 200, "Data": null}),
        json!({"Code": 200, "Data": "nonLabel"}),
        json!({"Code": 200, "Data": {"Reason": "vendor reason must stay private"}}),
        json!({"Data": {"Labels": "nonLabel", "Result": []}}),
    ] {
        let error = offline_backend().parse_response(&reply).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::Validation,
            "reply {reply}: {error}"
        );
        let message = error.to_string();
        assert!(message.contains("aliyun green reply"), "{message}");
        assert!(message.contains("`Code`"), "{message}");
        assert!(message.contains("`Data`"), "{message}");
        assert!(message.contains("`Labels`"), "{message}");
        assert!(!message.contains("vendor reason"), "{message}");
        assert!(!message.contains(&reply.to_string()), "{message}");
    }
}

#[test]
fn a_recognized_benign_reply_is_a_safe_assessment() {
    for reply in [
        json!({"Code": 200, "Data": {"Labels": "nonLabel", "Result": []}}),
        json!({"Code": 200, "Data": {"Labels": ""}}),
        json!({"Code": 200, "Data": {"Result": []}}),
    ] {
        let assessment = offline_backend().parse_response(&reply).unwrap();
        assert!(!assessment.has_risk, "reply {reply}");
        assert_eq!(assessment.risk_level, RiskLevel::Safe, "reply {reply}");
        assert_eq!(assessment.risk_type, None, "reply {reply}");
        assert_eq!(assessment.confidence, 1.0, "reply {reply}");
        assert_eq!(assessment.details, None, "reply {reply}");
    }
}
