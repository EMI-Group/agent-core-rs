#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Artefact-manager integration tests: cache precedence, naming, validation,
//! offline `modelscope` layouts, `huggingface` downloads against a loopback
//! server, locking, and the selector wiring.
//!
//! Everything is offline and deterministic: local files come from `tempfile`,
//! downloads hit a `wiremock` server, and no test reads the host's tokenizer
//! cache.
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_context::tokenizers::artifacts::{
    artifact_name, default_cache_dir, sha256_hex, ArtifactLock, TokenizerArtifactManager,
    ARTIFACT_ERROR_CODES, CACHE_DIR_ENV, DEFAULT_HUGGINGFACE_ENDPOINT, DEFAULT_MODELSCOPE_ENDPOINT,
    LOCAL_ARTIFACT_MISSING_CODE, PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE,
    REMOTE_DOWNLOAD_DISABLED_CODE, REMOTE_URL_NOT_ALLOWED_CODE, TIKTOKEN_MODEL_MISSING_CODE,
    TOKENIZER_DOWNLOAD_FAILED_CODE, TOKENIZER_JSON_MISSING_CODE,
    TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE, TOKENIZER_NETWORK_UNAVAILABLE_CODE,
    TOKENIZER_NOT_IN_LOCAL_CACHE_CODE, TOKENIZER_REVISION_UNAVAILABLE_CODE,
    TOKENIZER_SHA256_MISMATCH_CODE,
};
use openjiuwen_context::tokenizers::selector::NATIVE_ARTIFACT_UNDECODABLE_REASON;
use openjiuwen_context::tokenizers::CompatibleTokenizerSpec;
use openjiuwen_context::{
    ContextConfig, SelectedTokenizer, TokenCounterConfig, TokenCounterKind, TokenizerEngine,
    TokenizerRegistry, TokenizerSelector, TokenizerSource, TokenizerSpec,
};
use openjiuwen_foundation::prompt::TokenCounter;
use openjiuwen_test_support::fixtures::EnvGuard;
use tempfile::TempDir;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Tracing capture
// ---------------------------------------------------------------------------

/// A [`Subscriber`] keeping the level and rendered text of every event.
///
/// The selection ladder is synchronous, so a local subscriber captures exactly
/// the warnings one `select()` emits.
#[derive(Clone, Default)]
struct EventCapture {
    /// One `(level, rendered event)` pair per event.
    events: Arc<Mutex<Vec<(Level, String)>>>,
}

impl EventCapture {
    /// An empty capture.
    fn new() -> Self {
        Self::default()
    }

    /// The rendered text of every `warn` event, in emission order.
    fn warnings(&self) -> Vec<String> {
        self.events
            .lock()
            .expect("capture lock")
            .iter()
            .filter(|(level, _)| *level == Level::WARN)
            .map(|(_, message)| message.clone())
            .collect()
    }

    /// The warnings whose rendered text contains `needle`.
    fn warnings_containing(&self, needle: &str) -> Vec<String> {
        self.warnings()
            .into_iter()
            .filter(|warning| warning.contains(needle))
            .collect()
    }
}

impl Subscriber for EventCapture {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        self.events
            .lock()
            .expect("capture lock")
            .push((*event.metadata().level(), visitor.0));
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

/// Renders every field of an event, so a warning's structured fields are
/// asserted alongside its message.
struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, " {}={value:?}", field.name());
    }
}

/// A tiny BPE document the native decoder understands: `abc` is one token.
const TOKENIZER_JSON: &str = r#"{
    "name": "tiny-artifact",
    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": ["a b", "ab c"]},
    "pre_tokenizer": {"type": "Whitespace"}
}"#;

/// A tiny `tiktoken.model` vocabulary: the tokens `a`, `b` and `ab`.
const TIKTOKEN_MODEL: &str = "YQ== 0\nYg== 1\nYWI= 2\n";

/// The SHA-256 of [`TOKENIZER_JSON`], so a test does not re-derive it.
fn tokenizer_json_sha256() -> String {
    sha256_hex(TOKENIZER_JSON.as_bytes())
}

/// Writes `content` to `dir/name`, creating the directory.
fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(&path, content).expect("write fixture");
    path
}

/// A spec that names one hub source, id and engine.
fn hub_spec(source: &str, id: &str, engine: TokenizerEngine) -> TokenizerSpec {
    TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-mini".to_owned(),
        tokenizer_id: Some(id.to_owned()),
        source: Some(source.to_owned()),
        engine,
        ..TokenizerSpec::default()
    }
}

/// A spec that points at a local artefact file or directory.
fn local_spec(path: &Path, engine: TokenizerEngine) -> TokenizerSpec {
    TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-mini".to_owned(),
        artifact_path: Some(path.to_string_lossy().into_owned()),
        engine,
        ..TokenizerSpec::default()
    }
}

/// Every file named `name` below `root`, sorted.
fn find_files(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(std::result::Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|file| file == name) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

// ---------------------------------------------------------------------------
// Cache-dir precedence
// ---------------------------------------------------------------------------

#[test]
fn the_explicit_cache_dir_wins_over_the_environment_and_home() {
    let guard = EnvGuard::new(&[CACHE_DIR_ENV, "HOME"]);
    guard.set(CACHE_DIR_ENV, "/from/env");
    guard.set("HOME", "/from/home");
    let explicit = TempDir::new().expect("temp dir");
    let manager = TokenizerArtifactManager::new().with_cache_dir(explicit.path());
    assert_eq!(manager.cache_dir(), explicit.path());
    assert_eq!(
        default_cache_dir(),
        PathBuf::from("/from/env"),
        "the environment is the next rung"
    );
}

#[test]
fn the_environment_overrides_the_home_default() {
    let guard = EnvGuard::new(&[CACHE_DIR_ENV, "HOME"]);
    let home = TempDir::new().expect("temp dir");
    guard.set("HOME", home.path());
    assert_eq!(
        default_cache_dir(),
        home.path().join(".cache/openjiuwen/tokenizers")
    );
    guard.set(CACHE_DIR_ENV, "~/tokenizers");
    assert_eq!(default_cache_dir(), home.path().join("tokenizers"));
}

#[test]
fn without_home_the_documented_relative_default_applies() {
    let _guard = EnvGuard::new(&[CACHE_DIR_ENV, "HOME"]);
    assert_eq!(
        default_cache_dir(),
        PathBuf::from(".cache/openjiuwen/tokenizers")
    );
}

// ---------------------------------------------------------------------------
// Artefact naming and local resolution
// ---------------------------------------------------------------------------

#[test]
fn artefact_names_follow_the_engine() {
    assert_eq!(artifact_name(TokenizerEngine::Tiktoken), "tiktoken.model");
    assert_eq!(artifact_name(TokenizerEngine::Native), "tokenizer.json");
    assert_eq!(artifact_name(TokenizerEngine::Auto), "tokenizer.json");
}

#[test]
fn a_file_artifact_path_is_used_directly() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "custom-name.json", TOKENIZER_JSON);
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    let spec = local_spec(&file, TokenizerEngine::Native);
    assert_eq!(manager.resolve_cached(&spec), Some(file.clone()));
    assert_eq!(manager.last_error(), None);
}

#[test]
fn a_directory_artifact_path_picks_deterministically() {
    let dir = TempDir::new().expect("temp dir");
    let nested_b = write_file(dir.path(), "b/tokenizer.json", TOKENIZER_JSON);
    let nested_a = write_file(dir.path(), "a/tokenizer.json", TOKENIZER_JSON);
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    let spec = local_spec(dir.path(), TokenizerEngine::Native);
    assert_eq!(
        manager.resolve_cached(&spec),
        Some(nested_a),
        "the lexicographically first match wins, never an rglob race"
    );
    drop(nested_b);

    // A direct `tokenizer.json` beats the nested matches.
    let direct = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    assert_eq!(manager.resolve_cached(&spec), Some(direct));
}

#[test]
fn a_tiktoken_engine_looks_for_the_model_file() {
    let dir = TempDir::new().expect("temp dir");
    let model = write_file(dir.path(), "tiktoken.model", TIKTOKEN_MODEL);
    write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    assert_eq!(
        manager.resolve_cached(&local_spec(dir.path(), TokenizerEngine::Tiktoken)),
        Some(model)
    );
}

#[test]
fn a_declared_artifact_path_without_a_file_records_the_engine_code() {
    let dir = TempDir::new().expect("temp dir");
    let empty = dir.path().join("empty");
    std::fs::create_dir_all(&empty).expect("create dir");
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());

    assert_eq!(
        manager.resolve_cached(&local_spec(&empty, TokenizerEngine::Native)),
        None
    );
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_JSON_MISSING_CODE)
    );

    assert_eq!(
        manager.resolve_cached(&local_spec(&empty, TokenizerEngine::Tiktoken)),
        None
    );
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TIKTOKEN_MODEL_MISSING_CODE)
    );
}

#[test]
fn a_spec_without_any_artifact_records_the_local_code() {
    let dir = TempDir::new().expect("temp dir");
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    let spec = TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-mini".to_owned(),
        tokenizer_id: Some("acme/mini".to_owned()),
        ..TokenizerSpec::default()
    };
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(LOCAL_ARTIFACT_MISSING_CODE)
    );
}

#[test]
fn sha256_must_match_the_local_artifact() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());

    let mut good = local_spec(&file, TokenizerEngine::Native);
    good.sha256 = Some(tokenizer_json_sha256().to_uppercase());
    assert_eq!(manager.resolve_cached(&good), Some(file.clone()));

    let mut bad = local_spec(&file, TokenizerEngine::Native);
    bad.sha256 = Some("0".repeat(64));
    assert_eq!(manager.resolve_cached(&bad), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_SHA256_MISMATCH_CODE)
    );
}

#[tokio::test]
async fn a_url_tokenizer_id_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    let spec = hub_spec(
        "huggingface",
        "https://evil.example/tokenizer.json",
        TokenizerEngine::Native,
    );
    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(REMOTE_URL_NOT_ALLOWED_CODE)
    );
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(REMOTE_URL_NOT_ALLOWED_CODE)
    );
}

// ---------------------------------------------------------------------------
// ModelScope offline layouts
// ---------------------------------------------------------------------------

#[test]
fn all_three_modelscope_layouts_are_probed_offline() {
    for layout in ["", "models", "hub/models"] {
        let dir = TempDir::new().expect("temp dir");
        let root = if layout.is_empty() {
            dir.path().join("acme/mini")
        } else {
            dir.path().join(layout).join("acme/mini")
        };
        let file = write_file(&root, "tokenizer.json", TOKENIZER_JSON);
        let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
        let spec = hub_spec("modelscope", "acme/mini", TokenizerEngine::Native);
        assert_eq!(
            manager.resolve_cached(&spec),
            Some(file),
            "layout {layout:?} must be found"
        );
        assert_eq!(manager.last_error(), None);
    }
}

#[test]
fn a_modelscope_cache_miss_with_downloads_off_records_the_gate() {
    let dir = TempDir::new().expect("temp dir");
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    let spec = hub_spec("modelscope", "acme/mini", TokenizerEngine::Native);
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(REMOTE_DOWNLOAD_DISABLED_CODE)
    );
}

#[test]
fn a_modelscope_id_that_escapes_the_cache_is_never_searched() {
    let dir = TempDir::new().expect("temp dir");
    let outside = TempDir::new().expect("temp dir");
    write_file(outside.path(), "tokenizer.json", TOKENIZER_JSON);
    let manager = TokenizerArtifactManager::new().with_cache_dir(dir.path());
    let mut spec = hub_spec("modelscope", "acme/mini", TokenizerEngine::Native);
    spec.tokenizer_id = Some("../escape".to_owned());
    assert_eq!(manager.resolve_cached(&spec), None);
}

// ---------------------------------------------------------------------------
// HuggingFace downloads against a loopback server
// ---------------------------------------------------------------------------

/// Mounts one artefact response on `path` and returns the server.
async fn mount_artifact(server_path: &str, body: &str) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(server_path.to_owned()))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json"))
        .expect(1)
        .mount(&server)
        .await;
    server
}

/// A manager pointed at `server` with downloads enabled.
fn downloading_manager(cache: &Path, server: &MockServer) -> TokenizerArtifactManager {
    TokenizerArtifactManager::new()
        .with_cache_dir(cache)
        .with_endpoint(server.uri())
        .with_enable_download(true)
        .with_request_timeout(Duration::from_secs(5))
}

#[tokio::test]
async fn a_huggingface_download_lands_in_the_cache_and_is_reused() {
    let server = mount_artifact("/acme/tok/resolve/r1/tokenizer.json", TOKENIZER_JSON).await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let mut spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);
    spec.revision = Some("r1".to_owned());

    let resolved = manager.resolve(&spec).await.expect("artifact resolves");
    let expected = cache.path().join("huggingface/acme/tok/r1/tokenizer.json");
    assert_eq!(resolved, expected);
    assert_eq!(
        std::fs::read_to_string(&expected).expect("read artifact"),
        TOKENIZER_JSON
    );
    assert_eq!(manager.last_error(), None);

    // A second resolution is a cache hit: the mock still expects one request.
    assert_eq!(manager.resolve(&spec).await, Some(expected));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_download_is_skipped_when_the_cache_already_holds_the_artifact() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(TOKENIZER_JSON, "application/json"))
        .expect(0)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let cached = cache
        .path()
        .join("huggingface/acme/tok/main/tokenizer.json");
    std::fs::create_dir_all(cached.parent().unwrap()).unwrap();
    std::fs::write(&cached, TOKENIZER_JSON).unwrap();

    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);
    assert_eq!(manager.resolve(&spec).await, Some(cached));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn downloads_disabled_by_default_never_touch_the_network() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(TOKENIZER_JSON, "application/json"))
        .expect(0)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let manager = TokenizerArtifactManager::new()
        .with_cache_dir(cache.path())
        .with_endpoint(server.uri());
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(REMOTE_DOWNLOAD_DISABLED_CODE)
    );
    let error = manager
        .download(&spec)
        .await
        .expect_err("download is gated");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn offline_mode_never_downloads_even_when_enabled() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(TOKENIZER_JSON, "application/json"))
        .expect(0)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server).with_offline(true);
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_NOT_IN_LOCAL_CACHE_CODE)
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_404_records_the_revision_code() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/acme/tok/resolve/missing/tokenizer.json"))
        .respond_with(ResponseTemplate::new(404))
        .expect(2)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let mut spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);
    spec.revision = Some("missing".to_owned());

    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_REVISION_UNAVAILABLE_CODE),
        "the code is the compact one a caller matches on"
    );
    let recorded = manager
        .last_error_detail()
        .expect("the typed 404 message is recorded");
    assert!(
        recorded.starts_with(TOKENIZER_REVISION_UNAVAILABLE_CODE),
        "the detail keeps its compact code and the status: {recorded}"
    );
    assert!(recorded.contains("404"), "{recorded}");
    let error = manager.download(&spec).await.expect_err("404 is an error");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error
        .to_string()
        .contains(TOKENIZER_REVISION_UNAVAILABLE_CODE));
}

#[tokio::test]
async fn a_server_error_records_the_download_code() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(500))
        .expect(1)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    let error = manager.download(&spec).await.expect_err("500 is an error");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains(TOKENIZER_DOWNLOAD_FAILED_CODE));
}

#[tokio::test]
async fn an_unreachable_endpoint_records_the_network_code() {
    let cache = TempDir::new().expect("temp dir");
    // Port 1 on loopback refuses connections.
    let manager = TokenizerArtifactManager::new()
        .with_cache_dir(cache.path())
        .with_endpoint("http://127.0.0.1:1")
        .with_enable_download(true)
        .with_request_timeout(Duration::from_millis(500));
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);
    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_NETWORK_UNAVAILABLE_CODE)
    );
    let recorded = manager
        .last_error_detail()
        .expect("an unreachable hub is recorded");
    assert!(
        recorded.starts_with(TOKENIZER_NETWORK_UNAVAILABLE_CODE),
        "{recorded}"
    );
    assert!(recorded.contains("127.0.0.1:1"), "{recorded}");
}

#[tokio::test]
async fn a_malformed_body_is_a_typed_error_and_leaves_no_file() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("not a tokenizer", "text/plain"))
        .expect(2)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    let error = manager
        .download(&spec)
        .await
        .expect_err("a malformed body is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains(TOKENIZER_DOWNLOAD_FAILED_CODE));
    assert!(
        find_files(cache.path(), "tokenizer.json").is_empty(),
        "no partial artefact may be left behind"
    );
    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_DOWNLOAD_FAILED_CODE)
    );
    let recorded = manager
        .last_error_detail()
        .expect("a malformed body is recorded");
    assert!(
        recorded.starts_with(TOKENIZER_DOWNLOAD_FAILED_CODE),
        "{recorded}"
    );
    assert!(recorded.contains("not a usable tokenizer"), "{recorded}");
}

/// A well-formed document whose model type this crate does not decode.
const UNDECODED_TOKENIZER_JSON: &str = r#"{"model": {"type": "UnigramV2", "vocab": [["a", 0.0]]}}"#;

/// The compact code such a body reports, published through [`ARTIFACT_ERROR_CODES`].
const MODEL_TYPE_UNSUPPORTED_CODE: &str = "tokenizer_model_type_unsupported";

#[tokio::test]
async fn a_downloaded_body_of_an_undecoded_model_type_is_a_typed_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(UNDECODED_TOKENIZER_JSON, "application/json"),
        )
        .expect(2)
        .mount(&server)
        .await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    let error = manager
        .download(&spec)
        .await
        .expect_err("an undecoded model type is refused");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(ARTIFACT_ERROR_CODES.contains(&MODEL_TYPE_UNSUPPORTED_CODE));
    assert!(
        error.message().starts_with(MODEL_TYPE_UNSUPPORTED_CODE),
        "{error}"
    );
    assert!(error.message().contains("UnigramV2"), "{error}");
    assert!(error.message().contains("acme/tok"), "{error}");
    let message = error.message().into_owned();
    assert_eq!(
        manager.last_error().as_deref(),
        Some(MODEL_TYPE_UNSUPPORTED_CODE),
        "the code is the compact one a caller matches on"
    );
    assert_eq!(
        manager.last_error_detail().as_deref(),
        Some(message.as_str())
    );
    assert!(
        find_files(cache.path(), "tokenizer.json").is_empty(),
        "an artefact this loader cannot decode is not cached"
    );

    // The resolving call reports the same typed reason instead of a silent miss.
    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(MODEL_TYPE_UNSUPPORTED_CODE)
    );
    let recorded = manager
        .last_error_detail()
        .expect("the model type is recorded");
    assert!(
        recorded.starts_with(MODEL_TYPE_UNSUPPORTED_CODE),
        "{recorded}"
    );
    assert!(recorded.contains("UnigramV2"), "{recorded}");
}

#[tokio::test]
async fn a_download_must_match_the_declared_sha256() {
    let server = mount_artifact("/acme/tok/resolve/main/tokenizer.json", TOKENIZER_JSON).await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let mut spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);
    spec.sha256 = Some("0".repeat(64));

    let error = manager.download(&spec).await.expect_err("digest mismatch");
    assert!(error.to_string().contains(TOKENIZER_SHA256_MISMATCH_CODE));
    assert_eq!(
        find_files(cache.path(), "tokenizer.json"),
        Vec::<PathBuf>::new()
    );
}

#[tokio::test]
async fn concurrent_downloads_share_one_request_and_one_artefact() {
    let server = mount_artifact("/acme/tok/resolve/main/tokenizer.json", TOKENIZER_JSON).await;
    let cache = TempDir::new().expect("temp dir");
    let manager = downloading_manager(cache.path(), &server);
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    let (first, second) = tokio::join!(manager.download(&spec), manager.download(&spec));
    let first = first.expect("first download");
    let second = second.expect("second download");
    assert_eq!(first, second);
    assert_eq!(
        std::fs::read_to_string(&first).expect("read artifact"),
        TOKENIZER_JSON
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(find_files(cache.path(), "tokenizer.json").len(), 1);
    assert!(find_files(cache.path(), "tokenizer.json")
        .iter()
        .all(|path| !path.to_string_lossy().ends_with(".tmp")));
}

// ---------------------------------------------------------------------------
// Selector wiring
// ---------------------------------------------------------------------------

/// A manager over `dir` with downloads off.
fn offline_manager(dir: &Path) -> TokenizerArtifactManager {
    TokenizerArtifactManager::new().with_cache_dir(dir)
}

#[test]
fn a_native_engine_with_an_artefact_selects_the_native_tokenizer() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let spec = local_spec(&file, TokenizerEngine::Native);
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert!(!selected.estimated());
    assert_eq!(selected.fallback_reason(), None);
    assert_eq!(selected.model(), Some("acme-mini"));
    assert_eq!(selected.count_text("abc"), 1);
}

#[test]
fn an_auto_engine_prefers_the_native_artefact() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let spec = local_spec(&file, TokenizerEngine::Auto);
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert_eq!(selected.count_text("abc"), 1);
}

#[test]
fn a_tiktoken_engine_uses_the_model_file_when_it_resolves() {
    let dir = TempDir::new().expect("temp dir");
    let model = write_file(dir.path(), "tiktoken.model", TIKTOKEN_MODEL);
    let spec = local_spec(&model, TokenizerEngine::Tiktoken);
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert!(!selected.estimated());
    assert_eq!(selected.count_text("ab"), 1);
}

#[test]
fn a_missing_artefact_keeps_the_native_fallback_reason() {
    let dir = TempDir::new().expect("temp dir");
    let missing = dir.path().join("absent");
    let spec = local_spec(&missing, TokenizerEngine::Native);
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(
        selected.fallback_reason(),
        Some("native_tokenizer_unavailable")
    );
    assert!(selected.estimated());
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
}

#[test]
fn a_selection_without_a_manager_keeps_the_previous_ladder() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let spec = local_spec(&file, TokenizerEngine::Native);
    let selected = TokenizerSelector::new("acme", "acme-mini", Some(spec), None).select();
    assert_eq!(
        selected.fallback_reason(),
        Some("native_tokenizer_unavailable")
    );
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);
}

#[test]
fn a_one_hop_compatible_fallback_is_used() {
    let dir = TempDir::new().expect("temp dir");
    let fallback_file = write_file(dir.path(), "sibling/tokenizer.json", TOKENIZER_JSON);
    let mut spec = local_spec(&dir.path().join("absent"), TokenizerEngine::Native);
    spec.compatible_fallbacks = vec![CompatibleTokenizerSpec {
        model: "acme-sibling".to_owned(),
        tokenizer_id: Some("acme-sibling".to_owned()),
        artifact_path: Some(fallback_file.to_string_lossy().into_owned()),
        engine: TokenizerEngine::Native,
        ..CompatibleTokenizerSpec::default()
    }];
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(
        selected.source(),
        TokenizerSource::FamilyTokenizerFallback,
        "a sibling vocabulary is not the model's own"
    );
    assert_eq!(
        selected.fallback_reason(),
        Some("target_tokenizer_unavailable")
    );
    assert_eq!(selected.tokenizer(), Some("acme-sibling"));
    assert!(selected.estimated());
    assert_eq!(selected.count_text("abc"), 1);
}

#[test]
fn only_the_first_compatible_fallback_is_considered() {
    let dir = TempDir::new().expect("temp dir");
    let second_file = write_file(dir.path(), "second/tokenizer.json", TOKENIZER_JSON);
    let mut spec = local_spec(&dir.path().join("absent"), TokenizerEngine::Native);
    spec.compatible_fallbacks = vec![
        CompatibleTokenizerSpec {
            model: "absent".to_owned(),
            artifact_path: Some(
                dir.path()
                    .join("also-absent")
                    .to_string_lossy()
                    .into_owned(),
            ),
            engine: TokenizerEngine::Native,
            ..CompatibleTokenizerSpec::default()
        },
        CompatibleTokenizerSpec {
            model: "second".to_owned(),
            artifact_path: Some(second_file.to_string_lossy().into_owned()),
            engine: TokenizerEngine::Native,
            ..CompatibleTokenizerSpec::default()
        },
    ];
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(
        selected.fallback_reason(),
        Some("native_tokenizer_unavailable")
    );
}

#[test]
fn a_family_registry_match_uses_the_family_artefact() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let registry = TokenizerRegistry::new(vec![TokenizerSpec {
        provider: "acme".to_owned(),
        model: "acme-base".to_owned(),
        artifact_path: Some(file.to_string_lossy().into_owned()),
        engine: TokenizerEngine::Auto,
        ..TokenizerSpec::default()
    }]);
    let selector = TokenizerSelector::new("acme", "acme-base-mini", None, Some(registry))
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();
    assert_eq!(selected.source(), TokenizerSource::FamilyTokenizerFallback);
    assert_eq!(
        selected.fallback_reason(),
        Some("model_variant_tokenizer_family")
    );
    assert!(selected.estimated());
    assert_eq!(selected.count_text("abc"), 1);
}

#[tokio::test]
async fn a_provider_official_spec_names_the_vendor_and_the_tokenizer() {
    let dir = TempDir::new().expect("temp dir");
    let manager = offline_manager(dir.path());
    let spec = hub_spec("provider_official", "acme/tok", TokenizerEngine::Native);
    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE),
        "the code is the compact one a caller matches on"
    );
    let recorded = manager
        .last_error_detail()
        .expect("the adapter message is recorded");
    assert!(
        recorded.starts_with(PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE),
        "the detail keeps its compact code: {recorded}"
    );
    assert!(recorded.contains("acme"), "the vendor is named: {recorded}");
    assert!(
        recorded.contains("acme/tok"),
        "the tokenizer id is named: {recorded}"
    );
    assert!(
        recorded.contains("tokenizer adapter"),
        "the missing adapter is named: {recorded}"
    );

    // The synchronous rung reports the same reason.
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(PROVIDER_OFFICIAL_ADAPTER_NOT_CONFIGURED_CODE)
    );
    assert_eq!(
        manager.last_error_detail().as_deref(),
        Some(recorded.as_str()),
        "the synchronous rung records the same vendor message"
    );

    let enabled = manager.with_enable_download(true);
    let error = enabled.download(&spec).await.expect_err("no adapter");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("acme:acme/tok"), "{error}");
}

#[test]
fn the_manager_is_observable_from_the_selector() {
    let dir = TempDir::new().expect("temp dir");
    let selector = TokenizerSelector::new("acme", "acme-mini", None, None)
        .with_artifact_manager(offline_manager(dir.path()));
    let manager = selector.artifact_manager().expect("manager");
    assert_eq!(manager.cache_dir(), dir.path());
    assert_eq!(manager.endpoint(), DEFAULT_HUGGINGFACE_ENDPOINT);
    assert_eq!(manager.modelscope_endpoint(), DEFAULT_MODELSCOPE_ENDPOINT);
}

// ---------------------------------------------------------------------------
// ContextConfig plumbing
// ---------------------------------------------------------------------------

#[test]
fn config_round_trips_the_tokenizer_artifact_fields() {
    let config = ContextConfig {
        tokenizer_cache_dir: Some(PathBuf::from("/tmp/tokenizers")),
        enable_tokenizer_download: true,
        tokenizer_offline: false,
        ..ContextConfig::default()
    };
    let json = serde_json::to_string(&config).expect("serialize");
    let restored: ContextConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, config);
    assert_eq!(
        restored.tokenizer_cache_dir,
        Some(PathBuf::from("/tmp/tokenizers"))
    );

    let defaults = ContextConfig::default();
    assert_eq!(defaults.tokenizer_cache_dir, None);
    assert!(!defaults.enable_tokenizer_download);
    assert!(!defaults.tokenizer_offline);

    let partial: ContextConfig =
        serde_json::from_str(r#"{"enable_tokenizer_download": true, "unknown": 1}"#)
            .expect("partial");
    assert!(partial.enable_tokenizer_download);
    assert!(!partial.tokenizer_offline);

    let contradictory = ContextConfig {
        enable_tokenizer_download: true,
        tokenizer_offline: true,
        ..ContextConfig::default()
    };
    assert!(contradictory.validate().is_err());
}

#[test]
fn the_model_counter_uses_the_configured_artefact() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Model,
            chars_per_token: 4,
        },
        tokenizer_provider: Some("acme".to_owned()),
        model_name: Some("acme-mini".to_owned()),
        tokenizer_registry: vec![TokenizerSpec {
            provider: "acme".to_owned(),
            model: "acme-mini".to_owned(),
            artifact_path: Some(file.to_string_lossy().into_owned()),
            engine: TokenizerEngine::Auto,
            ..TokenizerSpec::default()
        }],
        tokenizer_cache_dir: Some(dir.path().to_path_buf()),
        ..ContextConfig::default()
    };
    config.validate().expect("valid config");
    let counter = config.token_counter();
    assert_eq!(counter.count_text("abc"), 1);
    assert_eq!(counter.count_text(""), 0);
}

#[test]
fn the_configured_manager_is_network_free_by_default() {
    let config = ContextConfig::default();
    let manager = config.tokenizer_artifact_manager();
    assert!(!manager.enable_download());
    assert!(!manager.offline());
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(REMOTE_DOWNLOAD_DISABLED_CODE)
    );
}

/// The helper types keep their documented shapes.
#[test]
fn selected_tokenizer_exposes_native_provenance() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", TOKENIZER_JSON);
    let selected: SelectedTokenizer =
        openjiuwen_context::load_native_tokenizer(&file).expect("load");
    assert_eq!(selected.source(), TokenizerSource::NativeTokenizer);
    assert!(!selected.estimated());
}

// ---------------------------------------------------------------------------
// A hand-placed artifact this loader cannot decode
// ---------------------------------------------------------------------------

#[test]
fn the_model_type_code_is_exported_and_listed() {
    assert_eq!(
        TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE, MODEL_TYPE_UNSUPPORTED_CODE,
        "the re-exported constant is the code the manager records"
    );
    assert!(ARTIFACT_ERROR_CODES.contains(&TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE));
    assert_eq!(
        ARTIFACT_ERROR_CODES.len(),
        13,
        "every compact code the manager can report is published once: {ARTIFACT_ERROR_CODES:?}"
    );
    let unique: std::collections::BTreeSet<&str> = ARTIFACT_ERROR_CODES.iter().copied().collect();
    assert_eq!(
        unique.len(),
        ARTIFACT_ERROR_CODES.len(),
        "no code is listed twice"
    );
}

#[test]
fn a_hand_placed_artifact_of_an_undecoded_model_type_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", UNDECODED_TOKENIZER_JSON);
    let manager = offline_manager(dir.path());
    let spec = local_spec(&file, TokenizerEngine::Native);

    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE)
    );
    let recorded = manager
        .last_error_detail()
        .expect("the model type is recorded");
    assert!(
        recorded.starts_with(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
        "{recorded}"
    );
    assert!(recorded.contains("UnigramV2"), "{recorded}");
    assert!(
        recorded.contains(&file.display().to_string()),
        "the refusal names the artifact: {recorded}"
    );
}

#[tokio::test]
async fn a_hand_placed_hub_artifact_of_an_undecoded_model_type_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let cached = write_file(
        &dir.path().join("huggingface/acme/tok/main"),
        "tokenizer.json",
        UNDECODED_TOKENIZER_JSON,
    );
    let manager = offline_manager(dir.path());
    let spec = hub_spec("huggingface", "acme/tok", TokenizerEngine::Native);

    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE)
    );
    let recorded = manager
        .last_error_detail()
        .expect("the model type is recorded");
    assert!(
        recorded.starts_with(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
        "{recorded}"
    );
    assert!(
        recorded.contains(&cached.display().to_string()),
        "the refusal names the cached artifact: {recorded}"
    );

    // The asynchronous rung refuses it for the same typed reason.
    assert_eq!(manager.resolve(&spec).await, None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE)
    );
    let recorded = manager
        .last_error_detail()
        .expect("the model type is recorded");
    assert!(
        recorded.starts_with(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
        "{recorded}"
    );
}

#[test]
fn an_undecoded_artifact_degrades_with_the_named_reason() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", UNDECODED_TOKENIZER_JSON);
    let spec = local_spec(&file, TokenizerEngine::Native);
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));

    let capture = EventCapture::new();
    let selected = tracing::subscriber::with_default(capture.clone(), || selector.select());

    assert_eq!(
        selected.fallback_reason(),
        Some(NATIVE_ARTIFACT_UNDECODABLE_REASON),
        "the estimate names the decode failure, not a cache miss"
    );
    assert_eq!(
        selected.fallback_reason(),
        Some("native_artifact_undecodable")
    );
    assert!(selected.estimated());
    assert_eq!(selected.source(), TokenizerSource::StringLengthFallback);

    // Exactly one warning names the refused artifact and the typed error; the
    // degradation warning of the estimate is a second, separate event.
    let refused = capture.warnings_containing("could not be decoded");
    assert_eq!(refused.len(), 1, "one warning: {:?}", capture.warnings());
    let warning = &refused[0];
    assert!(
        warning.contains(&file.display().to_string()),
        "the warning names the artifact: {warning}"
    );
    assert!(
        warning.contains(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE),
        "the warning names the typed error: {warning}"
    );
    // The warning carries the compact code the manager reports; the declared
    // model type travels with the full message.
    let manager = selector.artifact_manager().expect("manager");
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_MODEL_TYPE_UNSUPPORTED_CODE)
    );
    let detail = manager
        .last_error_detail()
        .expect("the refused artifact keeps its typed detail");
    assert!(
        detail.contains("UnigramV2"),
        "the detail carries the declared model type: {detail}"
    );
}

#[test]
fn an_undecoded_artifact_that_falls_to_tiktoken_keeps_the_reason() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", UNDECODED_TOKENIZER_JSON);
    let mut spec = local_spec(&file, TokenizerEngine::Native);
    // The spec's own model wins for encoding resolution, so name the model here.
    spec.model = "gpt-4o".to_owned();
    let selector = TokenizerSelector::new("acme", "gpt-4o", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();

    assert_eq!(selected.source(), TokenizerSource::TiktokenFallback);
    assert_eq!(
        selected.fallback_reason(),
        Some(NATIVE_ARTIFACT_UNDECODABLE_REASON)
    );
    assert!(selected.estimated());
}

#[test]
fn an_undecoded_primary_artifact_names_its_cause_on_the_fallback_rung() {
    let dir = TempDir::new().expect("temp dir");
    let primary = write_file(
        dir.path(),
        "primary/tokenizer.json",
        UNDECODED_TOKENIZER_JSON,
    );
    let sibling = write_file(dir.path(), "sibling/tokenizer.json", TOKENIZER_JSON);
    let mut spec = local_spec(&primary, TokenizerEngine::Native);
    spec.compatible_fallbacks = vec![CompatibleTokenizerSpec {
        model: "acme-sibling".to_owned(),
        tokenizer_id: Some("acme-sibling".to_owned()),
        artifact_path: Some(sibling.to_string_lossy().into_owned()),
        engine: TokenizerEngine::Native,
        ..CompatibleTokenizerSpec::default()
    }];
    let selector = TokenizerSelector::new("acme", "acme-mini", Some(spec), None)
        .with_artifact_manager(offline_manager(dir.path()));
    let selected = selector.select();

    assert_eq!(selected.source(), TokenizerSource::FamilyTokenizerFallback);
    assert_eq!(
        selected.fallback_reason(),
        Some(NATIVE_ARTIFACT_UNDECODABLE_REASON),
        "the sibling rung reports why the primary was not used"
    );
    assert_eq!(selected.tokenizer(), Some("acme-sibling"));
    assert!(selected.estimated());
    assert_eq!(selected.count_text("abc"), 1);
}

#[test]
fn a_rejected_digest_still_names_the_digest() {
    let dir = TempDir::new().expect("temp dir");
    let file = write_file(dir.path(), "tokenizer.json", UNDECODED_TOKENIZER_JSON);
    let manager = offline_manager(dir.path());
    let mut spec = local_spec(&file, TokenizerEngine::Native);
    spec.sha256 = Some("0".repeat(64));
    assert_eq!(manager.resolve_cached(&spec), None);
    assert_eq!(
        manager.last_error().as_deref(),
        Some(TOKENIZER_SHA256_MISMATCH_CODE),
        "the digest is checked before the body"
    );
}

// ---------------------------------------------------------------------------
// Lock markers with a clock-skewed mtime
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_lock_marker_from_the_future_is_live_and_named() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("skewed.lock");
    let marker = std::fs::File::create(&path).expect("marker");
    marker
        .set_modified(SystemTime::now() + Duration::from_secs(600))
        .expect("a future mtime");
    drop(marker);
    assert!(path.is_file());

    let capture = EventCapture::new();
    let _guard = tracing::subscriber::set_default(capture.clone());
    let error = ArtifactLock::acquire_within(path.clone(), Duration::from_millis(50))
        .await
        .expect_err("a future-dated marker is not stale");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        path.is_file(),
        "a live marker is never removed by the waiter"
    );

    // The waiter re-inspects the marker on every attempt, so the skew is logged
    // once per poll; each warning names the marker and both mtimes.
    let warned = capture.warnings_containing("clock skew");
    assert!(
        !warned.is_empty(),
        "the skew is logged: {:?}",
        capture.warnings()
    );
    for warning in &warned {
        assert!(
            warning.contains(&path.display().to_string()),
            "the warning names the marker: {warning}"
        );
        assert!(
            warning.contains("mtime_unix_ms") && warning.contains("now_unix_ms"),
            "the warning carries both mtimes: {warning}"
        );
    }
}
