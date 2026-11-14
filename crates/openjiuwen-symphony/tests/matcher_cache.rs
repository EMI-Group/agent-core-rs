//! Behaviour tests for the LLM matcher cache and the identity fingerprints.
//!
//! Every test is offline and deterministic: the model is a [`StaticCompletionModel`]
//! whose call counter proves when the cache spared a call, the cache lives in a
//! `tempfile` directory, and no assertion depends on the wall clock (`updated_at` is
//! metadata only). The suite covers the four layers the feature adds:
//!
//! - the identity helpers ([`endpoint_sha256`], [`stable_metadata_sha256`],
//!   [`sanitize_metadata`]) and their determinism;
//! - the [`MatchCache`] file format: round-trip, byte stability, malformed files, clear;
//! - [`cache_key`] sensitivity to each of its four components;
//! - the matcher wiring: replay without a model call, partial hits, `force` bypass,
//!   reopen, signature/model invalidation, and degradation on cache failures.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::similar_names)] // `matcher`/`matches` read better than invented names

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::graph::CAN_FEED;
use openjiuwen_symphony::llm::cache::{
    cache_key, cache_key_document, CachedMatch, MatchCache, CACHE_SCHEMA_VERSION,
};
use openjiuwen_symphony::llm::identity::{
    endpoint_sha256, graph_identity_sha256, sanitize_metadata, stable_metadata_sha256,
};
use openjiuwen_symphony::llm::{
    CompletionModel, FallbackMode, MatchOptions, OntologyConfig, OntologyMatcher,
    StaticCompletionModel, MATCHER_VERSION, MATCH_SCHEMA_VERSION, PROMPT_VERSION,
};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::orchestration::{MatchStrategy, RelationCandidate};

/// The forward consensus pass: `c1` is the first candidate.
const FORWARD: &str =
    r#"{"matches":[{"id":"c1","direction":"forward","confidence":0.9,"reason":"text flows"}]}"#;
/// The reversed consensus pass: the same candidate, re-numbered `c1`.
const REVERSE: &str =
    r#"{"matches":[{"id":"c1","direction":"forward","confidence":0.85,"reason":"text flows"}]}"#;

/// Builds a `tool` capability with the given IO declarations.
fn capability(
    id: &str,
    name: &str,
    inputs: Vec<CapabilityIo>,
    outputs: Vec<CapabilityIo>,
) -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", id, name)
        .with_inputs(inputs)
        .with_outputs(outputs)
}

/// The three-capability chain used by the matcher tests (`a` → `b` → `c`).
fn catalog() -> Vec<CapabilityDescriptor> {
    vec![
        capability(
            "a",
            "Extract",
            vec![],
            vec![CapabilityIo::new("text", "string")],
        ),
        capability(
            "b",
            "Summarise",
            vec![CapabilityIo::new("text", "string")],
            vec![CapabilityIo::new("summary", "string")],
        ),
        capability(
            "c",
            "Publish",
            vec![CapabilityIo::new("summary", "string")],
            vec![],
        ),
    ]
}

/// Builds a `can_feed` relation candidate.
fn candidate(source: &str, target: &str) -> RelationCandidate {
    RelationCandidate {
        source: source.to_string(),
        target: target.to_string(),
        relation: CAN_FEED.to_string(),
        confidence: 1.0,
        strategy: MatchStrategy::ExactIo,
        matched_input: None,
        matched_output: None,
    }
}

/// A scripted model replaying `replies`, with a public call counter.
fn scripted(replies: &[&str]) -> Arc<StaticCompletionModel> {
    Arc::new(StaticCompletionModel::new(
        replies.iter().map(|reply| (*reply).to_string()).collect(),
    ))
}

/// Wraps a scripted model behind a different reported name, so the model identity can
/// change without changing the script.
#[derive(Debug)]
struct RenamedModel {
    name: &'static str,
    inner: Arc<StaticCompletionModel>,
}

#[async_trait]
impl CompletionModel for RenamedModel {
    fn name(&self) -> &str {
        self.name
    }

    async fn complete(&self, prompt: &str) -> openjiuwen_symphony::Result<String> {
        self.inner.complete(prompt).await
    }
}

/// Builds an entry with a pinned timestamp, so files can be compared byte for byte.
fn stable_entry(candidate_id: &str) -> CachedMatch {
    CachedMatch::new(candidate_id, json!([])).with_updated_at("2024-01-02T03:04:05.000Z")
}

/// Opens a cache at `path` behind the mutex the matcher expects.
async fn open_cache(path: &std::path::Path) -> Arc<Mutex<MatchCache>> {
    Arc::new(Mutex::new(
        MatchCache::open(path).await.expect("cache opens"),
    ))
}

// --- identity fingerprints -------------------------------------------------
#[test]
fn canonical_digests_ignore_key_order_and_detect_changes() {
    let first: Value =
        serde_json::from_str(r#"{"model": "m", "config": {"top_p": 0.9, "stop": ["x"]}}"#)
            .expect("JSON");
    let reordered: Value =
        serde_json::from_str(r#"{"config": {"stop": ["x"], "top_p": 0.9}, "model": "m"}"#)
            .expect("JSON");
    assert_eq!(
        stable_metadata_sha256(&first),
        stable_metadata_sha256(&reordered)
    );
    assert_eq!(stable_metadata_sha256(&first).len(), 64);
    // Array order is content, so it must move the digest.
    let reordered_stop: Value =
        serde_json::from_str(r#"{"config": {"stop": ["y"], "top_p": 0.9}, "model": "m"}"#)
            .expect("JSON");
    assert_ne!(
        stable_metadata_sha256(&first),
        stable_metadata_sha256(&reordered_stop)
    );
}

#[test]
fn endpoint_digests_normalize_but_never_expose_the_query() {
    let base = endpoint_sha256("https://api.example.com/v1");
    assert_eq!(endpoint_sha256("  HTTPS://API.Example.COM:443/v1/  "), base);
    assert_eq!(
        endpoint_sha256("https://api.example.com/v1?api_key=sk-live&signed=1"),
        base
    );
    assert_eq!(endpoint_sha256("https://api.example.com/v1#fragment"), base);
    assert_ne!(endpoint_sha256("https://api.example.com:8443/v1"), base);
    assert_ne!(endpoint_sha256("http://api.example.com/v1"), base);
}

#[test]
fn sanitize_metadata_drops_and_hashes_the_right_keys_at_every_depth() {
    let value: Value = serde_json::from_str(
        r#"{
            "api_key": "sk-live-0123456789abcdef",
            "authorization": "Bearer abc",
            "private_key": "-----BEGIN",
            "client_secret": {"nested": "x"},
            "api_base": "https://API.Example.com/v1/?signature=abc",
            "base_url": null,
            "cache_key": "abc",
            "max_tokens": 512,
            "layers": [
                {"token": "t", "model": "m1"},
                {"password": "p", "model": "m2"},
                "plain"
            ],
            "timeout": 30
        }"#,
    )
    .expect("JSON");
    let sanitized = sanitize_metadata(&value);
    let object = sanitized.as_object().expect("object");
    assert!(!object.contains_key("api_key"));
    assert!(!object.contains_key("authorization"));
    assert!(!object.contains_key("private_key"));
    assert!(!object.contains_key("client_secret"));
    assert!(!object.contains_key("base_url"));
    assert_eq!(
        object.get("api_base_sha256"),
        Some(&Value::String(endpoint_sha256(
            "https://api.example.com/v1"
        )))
    );
    assert_eq!(object.get("cache_key"), Some(&json!("abc")));
    assert_eq!(object.get("max_tokens"), Some(&json!(512)));
    assert_eq!(object.get("timeout"), Some(&json!(30)));
    assert_eq!(
        object.get("layers"),
        Some(&json!([{"model": "m1"}, {"model": "m2"}, "plain"]))
    );
    // Key order is preserved and no credential value survives anywhere.
    // The hashed endpoint keeps the position of the key it replaced.
    let keys: Vec<&str> = object.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "api_base_sha256",
            "cache_key",
            "max_tokens",
            "layers",
            "timeout"
        ]
    );
    assert!(!sanitize_metadata(&value).to_string().contains("sk-live"));
}

// --- cache file ------------------------------------------------------------
#[tokio::test]
async fn the_cache_round_trips_through_a_temp_directory() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nested/match-cache.json");
    let cache = open_cache(&path).await;
    {
        let mut guard = cache.lock().await;
        guard
            .record(
                "key-1".to_string(),
                CachedMatch::new("tool:a->tool:b:can_feed", json!([{"confidence": 0.9}]))
                    .with_diagnostics(json!([{"code": "ok"}]))
                    .with_updated_at("2024-01-02T03:04:05.000Z"),
            )
            .await
            .unwrap();
    }
    let reopened = MatchCache::open(&path).await.unwrap();
    assert_eq!(reopened.len(), 1);
    let entry = reopened.lookup("key-1").await.expect("recorded");
    assert_eq!(entry.candidate_id, "tool:a->tool:b:can_feed");
    assert_eq!(entry.matches, json!([{"confidence": 0.9}]));
    assert_eq!(entry.diagnostics, json!([{"code": "ok"}]));
    assert_eq!(entry.updated_at, "2024-01-02T03:04:05.000Z");
    // The envelope names its schema.
    let text = tokio::fs::read_to_string(&path).await.unwrap();
    assert!(text.contains(CACHE_SCHEMA_VERSION));
}

#[tokio::test]
async fn the_cache_file_layout_is_byte_stable() {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("first.json");
    let second_path = directory.path().join("second.json");
    let mut first = MatchCache::open(&first_path).await.unwrap();
    let mut second = MatchCache::open(&second_path).await.unwrap();
    let entries = [
        ("zeta", "tool:c->tool:a:can_feed"),
        ("alpha", "tool:a->tool:b:can_feed"),
    ];
    // A pinned timestamp keeps the two files comparable (`updated_at` is metadata).
    for (key, id) in entries {
        first
            .record(key.to_string(), stable_entry(id))
            .await
            .unwrap();
    }
    for (key, id) in entries.iter().rev() {
        second
            .record((*key).to_string(), stable_entry(*id))
            .await
            .unwrap();
    }
    let first_bytes = tokio::fs::read(&first_path).await.unwrap();
    let second_bytes = tokio::fs::read(&second_path).await.unwrap();
    assert_eq!(first_bytes, second_bytes);
    let text = String::from_utf8(first_bytes).unwrap();
    assert!(text.starts_with(
        "{\n  \"schema_version\": \"Symphony-relation-match-cache-v1\",\n  \"records\": {"
    ));
    assert!(text.find("\"alpha\": {").unwrap() < text.find("\"zeta\": {").unwrap());
}

#[tokio::test]
async fn a_malformed_cache_file_is_an_error_that_names_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("broken-cache.json");
    tokio::fs::write(&path, b"{\"schema_version\": ")
        .await
        .unwrap();
    let error = MatchCache::open(&path).await.expect_err("malformed JSON");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("broken-cache.json"));

    tokio::fs::write(
        &path,
        br#"{"schema_version": "something-else", "records": {}}"#,
    )
    .await
    .unwrap();
    let error = MatchCache::open(&path).await.expect_err("wrong schema");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("broken-cache.json"));
}

// --- cache keys ------------------------------------------------------------
#[test]
fn cache_keys_are_order_independent_and_component_sensitive() {
    let candidate = json!({"source": "tool:a", "target": "tool:b", "confidence": 1.0});
    let reordered = json!({"confidence": 1.0, "target": "tool:b", "source": "tool:a"});
    let base = cache_key(&candidate, "source-hash", "target-hash", "signature");
    assert_eq!(
        cache_key(&reordered, "source-hash", "target-hash", "signature"),
        base
    );
    assert_ne!(
        cache_key(&candidate, "other", "target-hash", "signature"),
        base
    );
    assert_ne!(
        cache_key(&candidate, "source-hash", "other", "signature"),
        base
    );
    assert_ne!(
        cache_key(&candidate, "source-hash", "target-hash", "other"),
        base
    );
    assert!(cache_key_document(&candidate, "s", "t", "g").contains("\"matcher_signature\":\"g\""));
}

// --- matcher wiring --------------------------------------------------------
#[tokio::test]
async fn a_repeated_match_replays_the_cache_without_calling_the_model() {
    let directory = tempfile::tempdir().unwrap();
    let cache = open_cache(&directory.path().join("cache.json")).await;
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(cache.clone());
    let candidates = [candidate("tool:a", "tool:b")];

    let first = matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(model.call_count(), 2, "consensus asks the model twice");
    assert_eq!(first.len(), 1);
    assert_eq!(
        first[0].confidence, 0.85,
        "the lower consensus confidence wins"
    );
    assert_eq!(cache.lock().await.len(), 1);

    let replayed = matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(model.call_count(), 2, "the replay must not call the model");
    assert_eq!(replayed, first);

    // The recorded entry carries the candidate identity the cache documents.
    let entry = {
        let guard = cache.lock().await;
        let key = guard.keys().next().expect("one record").to_string();
        guard.lookup(&key).await.expect("recorded")
    };
    assert_eq!(entry.candidate_id, "tool:a->tool:b:can_feed");
    assert!(entry.matches.is_array());
}

#[tokio::test]
async fn only_the_unseen_candidates_cost_a_model_call() {
    let directory = tempfile::tempdir().unwrap();
    let cache = open_cache(&directory.path().join("cache.json")).await;
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(cache.clone());

    let first = matcher
        .match_candidates(&[candidate("tool:a", "tool:b")])
        .await
        .unwrap();
    assert_eq!(model.call_count(), 2);
    assert_eq!(cache.lock().await.len(), 1);

    let mixed = matcher
        .match_candidates(&[candidate("tool:a", "tool:b"), candidate("tool:b", "tool:c")])
        .await
        .unwrap();
    assert_eq!(
        model.call_count(),
        4,
        "only the new candidate is sent to the model (two consensus passes)"
    );
    assert_eq!(cache.lock().await.len(), 2);
    assert_eq!(mixed.len(), 2);
    assert!(mixed.contains(&first[0]));

    // Everything is cached now: a third call is entirely free.
    let third = matcher
        .match_candidates(&[candidate("tool:a", "tool:b"), candidate("tool:b", "tool:c")])
        .await
        .unwrap();
    assert_eq!(model.call_count(), 4);
    assert_eq!(third, mixed);
}

#[tokio::test]
async fn force_bypasses_the_cache_and_refreshes_the_entry() {
    let directory = tempfile::tempdir().unwrap();
    let cache = open_cache(&directory.path().join("cache.json")).await;
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(cache.clone());
    let candidates = [candidate("tool:a", "tool:b")];
    matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(model.call_count(), 2);

    let forced = matcher
        .match_candidates_with(&candidates, &MatchOptions::new().with_force(true))
        .await
        .unwrap();
    assert_eq!(model.call_count(), 4, "force asks the model again");
    assert_eq!(forced.len(), 1);
    assert_eq!(
        cache.lock().await.len(),
        1,
        "the entry was refreshed in place"
    );

    matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(model.call_count(), 4, "the refreshed entry is honoured");
}

#[tokio::test]
async fn a_reopened_cache_spares_the_model() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.json");
    let candidates = [candidate("tool:a", "tool:b")];
    {
        let model = scripted(&[FORWARD, REVERSE]);
        let matcher = OntologyMatcher::new(catalog())
            .with_model(model.clone())
            .with_cache(open_cache(&path).await);
        matcher.match_candidates(&candidates).await.unwrap();
        assert_eq!(model.call_count(), 2);
    }

    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(open_cache(&path).await);
    let replayed = matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(model.call_count(), 0, "the persisted answer is reused");
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].method, "llm_consensus_match");
}

#[tokio::test]
async fn a_changed_matcher_signature_or_model_identity_invalidates_the_key() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.json");
    let candidates = [candidate("tool:a", "tool:b")];
    let cache = open_cache(&path).await;
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(cache.clone());
    matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(model.call_count(), 2);

    // A different acceptance threshold changes the matcher signature.
    let stricter_model = scripted(&[FORWARD, REVERSE]);
    let stricter = OntologyMatcher::new(catalog())
        .with_config(
            OntologyConfig::new()
                .with_threshold(0.8)
                .with_model(stricter_model.clone()),
        )
        .with_cache(cache.clone());
    assert_ne!(
        stricter.matcher_signature(),
        matcher.matcher_signature(),
        "the signature covers the threshold"
    );
    let strict_matches = stricter.match_candidates(&candidates).await.unwrap();
    assert_eq!(
        stricter_model.call_count(),
        2,
        "a signature change is a miss"
    );
    assert_eq!(
        cache.lock().await.len(),
        2,
        "the record is keyed by signature"
    );
    assert_eq!(strict_matches[0].confidence, 0.85);
    assert!(strict_matches[0].accepted, "0.85 clears the 0.8 threshold");

    // A different model name changes the model identity (and so the signature).
    let renamed_inner = scripted(&[FORWARD, REVERSE]);
    let renamed: Arc<dyn CompletionModel> = Arc::new(RenamedModel {
        name: "another-model",
        inner: renamed_inner.clone(),
    });
    let renamed_matcher = OntologyMatcher::new(catalog())
        .with_model(renamed)
        .with_cache(cache.clone());
    assert_ne!(
        renamed_matcher.matcher_signature(),
        matcher.matcher_signature()
    );
    renamed_matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(renamed_inner.call_count(), 2, "a model change is a miss");
    assert_eq!(cache.lock().await.len(), 3);
}

#[tokio::test]
async fn the_signature_names_its_inputs() {
    let matcher = OntologyMatcher::new(catalog()).with_model(scripted(&[FORWARD]));
    let signature = matcher.matcher_signature();
    assert!(signature.contains(MATCHER_VERSION));
    assert!(signature.contains(PROMPT_VERSION));
    assert!(signature.contains(MATCH_SCHEMA_VERSION));
    assert!(signature.contains(CACHE_SCHEMA_VERSION));
    assert!(signature.contains(&format!("\"relation\":\"{CAN_FEED}\"")));
    assert!(signature.contains("\"require_consensus\":true"));
    // Canonical rendering: no whitespace between tokens and sorted keys.
    assert!(signature.starts_with("{\"cache_schema_version\""));
    assert!(!signature.contains(": "));
}

#[tokio::test]
async fn a_deterministic_fallback_answer_never_enters_the_cache() {
    let directory = tempfile::tempdir().unwrap();
    let cache = open_cache(&directory.path().join("cache.json")).await;
    let model = scripted(&["no json here"]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model)
        .with_cache(cache.clone());
    let matches = matcher
        .match_candidates(&[candidate("tool:a", "tool:b")])
        .await
        .unwrap();
    assert_eq!(matches[0].method, "deterministic_fallback");
    assert!(
        cache.lock().await.is_empty(),
        "a fallback answer must not shadow a future model answer"
    );
}

#[tokio::test]
async fn an_unusable_cached_answer_degrades_to_a_model_call() {
    let directory = tempfile::tempdir().unwrap();
    let cache = open_cache(&directory.path().join("cache.json")).await;
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(cache.clone());
    let candidates = [candidate("tool:a", "tool:b")];

    // Pre-seed the exact key the matcher will compute with a payload it cannot decode.
    let catalog = catalog();
    let (source, target) = (&catalog[0], &catalog[1]);
    let key = cache_key(
        &serde_json::to_value(&candidates[0]).unwrap(),
        &graph_identity_sha256(source),
        &graph_identity_sha256(target),
        &matcher.matcher_signature(),
    );
    cache
        .lock()
        .await
        .record(
            key.clone(),
            CachedMatch::new("tool:a->tool:b:can_feed", json!("not a list")),
        )
        .await
        .unwrap();

    let matches = matcher.match_candidates(&candidates).await.unwrap();
    assert_eq!(
        model.call_count(),
        2,
        "the unusable entry forced a model call"
    );
    assert_eq!(matches.len(), 1);
    // The entry was refreshed with a decodable answer.
    let refreshed = cache.lock().await.lookup(&key).await.expect("refreshed");
    assert!(refreshed.matches.is_array());
}

#[tokio::test]
async fn a_cache_write_failure_still_returns_the_model_answer() {
    let directory = tempfile::tempdir().unwrap();
    // A regular file where the cache's parent directory should be: every write fails.
    let blocker = directory.path().join("blocked");
    tokio::fs::write(&blocker, b"not a directory")
        .await
        .unwrap();
    let cache = Arc::new(Mutex::new(MatchCache::new(blocker.join("cache.json"))));
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog())
        .with_model(model.clone())
        .with_cache(cache.clone());
    let matches = matcher
        .match_candidates(&[candidate("tool:a", "tool:b")])
        .await
        .unwrap();
    assert_eq!(model.call_count(), 2);
    assert_eq!(
        matches.len(),
        1,
        "the model answer survives a cache failure"
    );
    // The entry is kept in memory for a later successful write, but nothing reached disk.
    assert_eq!(cache.lock().await.len(), 1);
    assert!(!directory.path().join("blocked/cache.json").exists());

    // The same call, un-cached, is byte-identical in outcome.
    let uncached_model = scripted(&[FORWARD, REVERSE]);
    let uncached = OntologyMatcher::new(catalog()).with_model(uncached_model.clone());
    let plain = uncached
        .match_candidates(&[candidate("tool:a", "tool:b")])
        .await
        .unwrap();
    assert_eq!(plain, matches);
    assert_eq!(uncached_model.call_count(), model.call_count());
}

#[tokio::test]
async fn error_fallback_mode_surfaces_before_the_cache_is_touched() {
    let directory = tempfile::tempdir().unwrap();
    let cache = open_cache(&directory.path().join("cache.json")).await;
    let matcher = OntologyMatcher::new(catalog())
        .with_config(
            OntologyConfig::new()
                .with_fallback(FallbackMode::Error)
                .with_model(scripted(&["no json here"])),
        )
        .with_cache(cache.clone());
    let error = matcher
        .match_candidates(&[candidate("tool:a", "tool:b")])
        .await
        .expect_err("the failure must surface");
    assert!(error.message().contains("ontology matcher"));
    assert!(cache.lock().await.is_empty());
}

#[tokio::test]
async fn matching_without_a_cache_is_unchanged() {
    let model = scripted(&[FORWARD, REVERSE]);
    let matcher = OntologyMatcher::new(catalog()).with_model(model.clone());
    let matches = matcher
        .match_candidates(&[candidate("tool:a", "tool:b")])
        .await
        .unwrap();
    assert_eq!(model.call_count(), 2);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].method, "llm_consensus_match");
    assert!(matcher.cache().is_none());
}

#[test]
fn match_options_default_to_honouring_the_cache() {
    assert!(!MatchOptions::new().force);
    assert!(MatchOptions::new().with_force(true).force);
    assert_eq!(MatchOptions::default(), MatchOptions::new());
}
