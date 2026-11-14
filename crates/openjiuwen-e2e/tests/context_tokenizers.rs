#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Tokenizer scenario: native token counting through the context engine and the
//! tokenizer artifact manager's cache/validation contract.
//!
//! This binary is the cross-crate seam between `openjiuwen-context`'s tokenizer
//! seam and the context engine that consumes it. Each crate's own tests pin its
//! module; here the public engine is driven the way a deployment drives it — a
//! `tokenizer.json` artifact is declared in a `ContextConfig`, the engine sizes
//! its window and compresses with *native* counts, and the artifact manager
//! downloads and validates that artifact against a loopback `wiremock` hub.
//!
//! Everything is offline and deterministic: artifacts are written into a
//! `tempfile::TempDir`, the hub is `wiremock` on `127.0.0.1:0`, no credentials
//! are read, and assertions read back recorded evidence (token counts, message
//! counts, request counts, file digests) instead of timestamps or durations.
//!
//! Two documented facts shape the assertions below: the workspace chat framing
//! adds four tokens per message, and the fixture's byte-level vocabulary has one
//! token per UTF-8 byte, so `"abc"` is three tokens plus framing.
mod support;

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use openjiuwen::core::common::error::ErrorKind;
use openjiuwen::core::context_engine::tokenizers::artifacts::{
    sha256_file, sha256_hex, ArtifactLock, TokenizerArtifactManager, HUGGINGFACE_SOURCE,
    REMOTE_DOWNLOAD_DISABLED_CODE, STALE_LOCK_AFTER, TOKENIZER_NOT_IN_LOCAL_CACHE_CODE,
    TOKENIZER_REVISION_UNAVAILABLE_CODE, TOKENIZER_SHA256_MISMATCH_CODE,
};
use openjiuwen::core::context_engine::tokenizers::{
    byte_level_alphabet, load_native_tokenizer, SelectedTokenizer, TokenizerEngine,
    TokenizerRegistry, TokenizerSelector, TokenizerSource, TokenizerSpec,
};
use openjiuwen::core::context_engine::{
    CompressionStatus, ContextConfig, ContextEngine, ContextEventKind, ModelContext,
    PrepareOptions, ProcessorConfig, TokenCounterConfig, TokenCounterKind, TruncationConfig,
};
use openjiuwen::core::foundation::prompt::{BaseMessage, TokenCounter};
use openjiuwen_test_support::wiremock::matchers::{method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use support::within;
use tempfile::TempDir;

/// A byte-level `tokenizer.json` with no merges: every UTF-8 byte is one token.
///
/// The vocabulary is the 256-entry GPT-2 alphabet, so the document is a valid
/// artifact the native decoder fully supports and the expected counts are exact
/// (`"abc"` is three tokens, not an estimate).
fn byte_level_fixture() -> String {
    let mut vocab = serde_json::Map::new();
    for (byte, ch) in byte_level_alphabet().iter().enumerate() {
        vocab.insert(ch.to_string(), serde_json::Value::from(byte as u64));
    }
    serde_json::json!({
        "name": "byte-level-artifact",
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "decoder": {"type": "ByteLevel"}
    })
    .to_string()
}

/// Writes `content` to `dir/name`, creating the directory, and returns the path.
fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(&path, content).expect("write fixture");
    path
}

/// Three complete user/assistant rounds, every message the text `abc`.
fn three_rounds() -> Vec<BaseMessage> {
    let mut messages = Vec::new();
    for _ in 0..3 {
        messages.push(BaseMessage::user("abc"));
        messages.push(BaseMessage::assistant("abc"));
    }
    messages
}

/// A configuration whose model-aware counter resolves through `artifact`.
///
/// The window is 40 tokens with a compression trigger at half of it, and the
/// only processor is truncation with a 20-token budget that protects the newest
/// two messages, so both `prepare` and `compress` have work to do.
fn native_config(cache: &Path, artifact: &Path) -> ContextConfig {
    ContextConfig {
        max_context_tokens: Some(40),
        compression_threshold: 0.5,
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Model,
            ..TokenCounterConfig::default()
        },
        tokenizer_provider: Some("acme".to_owned()),
        model_name: Some("acme-mini".to_owned()),
        tokenizer_registry: vec![TokenizerSpec {
            provider: "acme".to_owned(),
            model: "acme-mini".to_owned(),
            artifact_path: Some(artifact.to_string_lossy().into_owned()),
            engine: TokenizerEngine::Native,
            ..TokenizerSpec::default()
        }],
        tokenizer_cache_dir: Some(cache.to_path_buf()),
        processors: vec![ProcessorConfig::Truncation(TruncationConfig {
            max_tokens: Some(20),
            keep_recent_messages: 2,
            keep_system: true,
        })],
        ..ContextConfig::default()
    }
}

/// The selection `ContextConfig::token_counter` builds: the registry entry
/// through the configured artifact manager.
fn configured_selection(config: &ContextConfig) -> SelectedTokenizer {
    TokenizerSelector::new(
        "acme",
        "acme-mini",
        None,
        Some(TokenizerRegistry::new(config.tokenizer_registry.clone())),
    )
    .with_artifact_manager(config.tokenizer_artifact_manager())
    .select()
}

/// A hub spec for one `huggingface` artifact revision.
fn hub_spec(id: &str, revision: &str, sha256: Option<String>) -> TokenizerSpec {
    TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-mini".to_owned(),
        tokenizer_id: Some(id.to_owned()),
        source: Some(HUGGINGFACE_SOURCE.to_owned()),
        engine: TokenizerEngine::Native,
        revision: Some(revision.to_owned()),
        sha256,
        ..TokenizerSpec::default()
    }
}

/// A manager over `cache` that downloads from `server`.
fn downloading_manager(cache: &Path, server: &MockServer) -> TokenizerArtifactManager {
    TokenizerArtifactManager::new()
        .with_cache_dir(cache)
        .with_endpoint(server.uri())
        .with_enable_download(true)
        .with_request_timeout(Duration::from_secs(5))
}

#[tokio::test]
async fn native_token_counts_drive_the_context_engine() {
    let dir = TempDir::new().expect("temp dir");
    let artifact = write_file(dir.path(), "tokenizer.json", &byte_level_fixture());
    let config = native_config(dir.path(), &artifact);
    config.validate().expect("valid config");

    let selected = configured_selection(&config);
    assert_eq!(
        selected.source(),
        TokenizerSource::NativeTokenizer,
        "a declared artifact must win over the estimate"
    );
    assert!(!selected.estimated());
    assert_eq!(selected.fallback_reason(), None);
    assert_eq!(selected.tokenizer(), Some("byte-level-artifact"));

    // Three bytes plus four framing tokens per message.
    let counter = config.token_counter();
    assert_eq!(counter.count_text("abc"), 3);
    assert_eq!(counter.count_messages(&three_rounds()), 42);
    assert_eq!(selected.count_messages(&three_rounds()), 42);
    // The falling-back estimate would have said 30: the engine cannot be using it.
    let estimate = TokenizerSelector::new("acme", "acme-mini", None, None).select();
    assert_eq!(estimate.source(), TokenizerSource::StringLengthFallback);
    assert_eq!(estimate.count_messages(&three_rounds()), 30);

    let engine = ContextEngine::new(config);
    let context = engine.create_context("main", None).await.expect("context");
    context
        .set_messages(three_rounds())
        .await
        .expect("store history");

    // `prepare` truncates on a copy: the window shrinks, the history does not.
    let prepared = within(
        "the prepared window",
        engine.prepare(&context, &PrepareOptions::default()),
    )
    .await
    .expect("prepare");
    assert_eq!(prepared.messages.len(), 2);
    assert_eq!(prepared.token_count, 14);
    assert_eq!(
        prepared.token_count,
        selected.count_messages(&prepared.messages)
    );
    assert_eq!(prepared.stats.total_tokens, prepared.token_count);
    assert!(!prepared.overflow);
    assert_eq!(context.messages().await.len(), 6, "prepare never persists");

    // `compress` is the pass that persists the same native accounting.
    let report = within("the compression pass", engine.compress(&context))
        .await
        .expect("compress");
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert_eq!(report.events.len(), 1);
    assert_eq!(report.events[0].processor, "truncation");
    assert_eq!(report.events[0].kind, ContextEventKind::Truncated);
    assert_eq!(report.tokens_before, 42);
    assert_eq!(report.tokens_after, 14);
    assert_eq!(report.messages, 2);
    let kept = context.messages().await;
    assert_eq!(report.tokens_after, selected.count_messages(&kept));
    assert_eq!(
        report.tokens_after,
        context.token_counter().count_messages(&kept)
    );
}

#[tokio::test]
async fn a_downloaded_tokenizer_is_cached_and_reused() {
    let fixture = byte_level_fixture();
    let digest = sha256_hex(fixture.as_bytes());
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/acme/tok/resolve/r1/tokenizer.json"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(fixture.clone(), "application/json"))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/acme/tok/resolve/missing/tokenizer.json"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("acme/tok", "r1", Some(digest.clone()));

    let cached = within("the tokenizer download", manager.resolve(&spec))
        .await
        .expect("the download resolves");
    let expected = cache.path().join("huggingface/acme/tok/r1/tokenizer.json");
    assert_eq!(cached, expected);
    assert_eq!(
        std::fs::read_to_string(&expected).expect("read artifact"),
        fixture
    );
    assert_eq!(sha256_file(&expected).expect("digest"), digest);
    assert_eq!(manager.last_error(), None);

    // A second resolution is a cache hit: the mock still expects one request.
    assert_eq!(
        within("the cached resolution", manager.resolve(&spec)).await,
        Some(expected.clone())
    );
    assert_eq!(manager.resolve_cached(&spec), Some(expected.clone()));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);

    // The cached artifact is an exact native tokenizer, not an estimate.
    let selected = load_native_tokenizer(&expected).expect("the cached artifact loads");
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert!(!selected.estimated());
    assert_eq!(selected.tokenizer(), Some("byte-level-artifact"));
    assert_eq!(selected.count_text("abc"), 3);

    // A missing revision is a typed error, and nothing is cached for it.
    let missing = hub_spec("acme/tok", "missing", Some(digest));
    let error = within("the missing-revision download", manager.download(&missing))
        .await
        .expect_err("a 404 is an error");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error
        .to_string()
        .contains(TOKENIZER_REVISION_UNAVAILABLE_CODE));
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_REVISION_UNAVAILABLE_CODE)
    );
    assert!(!cache
        .path()
        .join("huggingface/acme/tok/missing/tokenizer.json")
        .exists());
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn a_tampered_artifact_is_rejected_and_refetched() {
    let fixture = byte_level_fixture();
    let digest = sha256_hex(fixture.as_bytes());
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/acme/tok/resolve/main/tokenizer.json"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(fixture.clone(), "application/json"))
        .expect(2)
        .mount(&server)
        .await;

    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("acme/tok", "main", Some(digest.clone()));
    let target = manager.cache_target(&spec).expect("a cache target");
    let parent = target.parent().expect("a cache parent");
    std::fs::create_dir_all(parent).expect("create cache dir");

    // A leftover temporary file from an interrupted write is not an artifact.
    std::fs::write(parent.join(".openjiuwen-partial.tmp"), &fixture).expect("partial file");
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_NOT_IN_LOCAL_CACHE_CODE)
    );

    let cached = within("the first download", manager.download(&spec))
        .await
        .expect("the download succeeds");
    assert_eq!(cached, target);

    // A body that does not match the declared digest is not a valid cache entry.
    std::fs::write(&target, r#"{"model": {"type": "BPE", "vocab": {"x": 0}}}"#)
        .expect("tamper with the cache");
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_SHA256_MISMATCH_CODE)
    );
    assert_eq!(
        within("the tampered resolution", manager.resolve(&spec)).await,
        None
    );
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_SHA256_MISMATCH_CODE)
    );

    // A download re-fetches instead of trusting the tampered entry.
    let restored = within("the refetch", manager.download(&spec))
        .await
        .expect("the refetch succeeds");
    assert_eq!(restored, target);
    assert_eq!(sha256_file(&target).expect("digest"), digest);
    assert_eq!(
        std::fs::read_to_string(&target).expect("read artifact"),
        fixture
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    assert_eq!(manager.resolve_cached(&spec), Some(target));
}

#[tokio::test]
async fn a_stale_lock_is_replaced_not_honoured() {
    let dir = TempDir::new().expect("temp dir");
    let lock_path = dir.path().join("tokenizer.lock");

    // A marker another downloader still holds blocks a bounded wait.
    let held = ArtifactLock::acquire(lock_path.clone())
        .await
        .expect("the first lock");
    assert_eq!(held.path(), lock_path);
    assert!(lock_path.is_file());
    let error = within(
        "the blocked lock wait",
        ArtifactLock::acquire_within(lock_path.clone(), Duration::from_millis(100)),
    )
    .await
    .expect_err("a held marker blocks the waiter");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("was not released"));
    drop(held);
    assert!(!lock_path.exists(), "the guard removes its marker");

    // An abandoned marker (a crashed downloader) is expired, never honoured.
    let stale = std::fs::File::create(&lock_path).expect("stale marker");
    stale
        .set_modified(
            SystemTime::now()
                .checked_sub(STALE_LOCK_AFTER * 5)
                .expect("a past instant"),
        )
        .expect("age the marker");
    drop(stale);
    assert!(lock_path.is_file());
    let recovered = within(
        "the stale lock acquisition",
        ArtifactLock::acquire(lock_path.clone()),
    )
    .await
    .expect("a stale marker is replaced");
    assert_eq!(recovered.path(), lock_path);
    assert!(lock_path.is_file());
    drop(recovered);
    assert!(!lock_path.exists());

    // The manager itself stays network-free unless a download is asked for.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let manager = TokenizerArtifactManager::new()
        .with_cache_dir(dir.path())
        .with_endpoint(server.uri());
    let spec = hub_spec("acme/tok", "main", None);
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(REMOTE_DOWNLOAD_DISABLED_CODE)
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}
