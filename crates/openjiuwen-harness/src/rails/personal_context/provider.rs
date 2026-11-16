//! Source-provider seam for the offline `PersonalContext` subset.
//!
//! Port of the `Provider` literal in
//! `openjiuwen/harness/personal_context/config.py` and the provider surface of
//! `personal_context/fetch/` (`base.py` and `local_files.py`). The vendor
//! fetchers (`github.py`, `gitcode.py`, `feishu.py`, `browser_bookmarks.py`,
//! `zhihu_reader.py`, `toutiao_reader.py`, `rss_feed.py`) are deliberately not
//! ported: each one needs a vendor SDK or a remote transport, so this module
//! keeps only their kind and a typed deferred path.
//!
//! The seam exists so vendor transports can be added without touching the
//! store or the rail: a [`SourceProvider`] turns a [`FetchRequest`] into
//! [`FetchedSource`] values and the [`SourceProviderRegistry`] resolves one by
//! [`SourceKind`]. Every vendor path returns a typed
//! [`Error::unsupported`] naming the vendor and the reason — there are no
//! silent no-ops — and local directory ingestion is the only real provider.
//!
//! [`FetchRequest::source`] is provider-specific free text: for local files it
//! is a directory, while a deferred vendor fetcher would read its own locator
//! grammar (a repository, a document URL, a feed URL, ...). Nothing here opens
//! a network connection and nothing panics on user input.
// Size note: ~430 lines of implementation plus an inline test module; the
// literal, message and registry parity assertions stay beside the code they
// verify.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use super::ingest::{ingest_local_directory, IngestedDocument, LocalDirectoryConfig};
use super::source::{normalize_source_locator, source_id_for_locator};

/// Service id stamped on a request-driven local scan whose service is blank.
const DEFAULT_LOCAL_SERVICE: &str = "local_files";

/// The source kinds of Python's `Provider` literal, in declaration order.
///
/// The serde representation is the `snake_case` name, so a persisted source
/// round-trips as `"local_files"`, `"browser_bookmarks"`, and so on.
/// [`SourceKind::LocalFiles`] is the default because it is the only kind this
/// subtree can serve today.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Local directory ingestion; the only implemented provider.
    #[default]
    LocalFiles,
    /// `GitHub` repositories (`github.py`): deferred vendor transport.
    Github,
    /// `GitCode` repositories (`gitcode.py`): deferred vendor transport.
    Gitcode,
    /// `Feishu` documents, tasks and calendars (`feishu.py`): deferred vendor transport.
    Feishu,
    /// Browser bookmark exports (`browser_bookmarks.py`): deferred vendor transport.
    BrowserBookmarks,
    /// Zhihu reader collections (`zhihu_reader.py`): deferred vendor transport.
    ZhihuReader,
    /// Toutiao reader collections (`toutiao_reader.py`): deferred vendor transport.
    ToutiaoReader,
    /// RSS and Atom feeds (`rss_feed.py`): deferred vendor transport.
    RssFeed,
}

impl SourceKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 8] = [
        Self::LocalFiles,
        Self::Github,
        Self::Gitcode,
        Self::Feishu,
        Self::BrowserBookmarks,
        Self::ZhihuReader,
        Self::ToutiaoReader,
        Self::RssFeed,
    ];

    /// Returns the `snake_case` literal used by Python's `Provider` literal.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalFiles => "local_files",
            Self::Github => "github",
            Self::Gitcode => "gitcode",
            Self::Feishu => "feishu",
            Self::BrowserBookmarks => "browser_bookmarks",
            Self::ZhihuReader => "zhihu_reader",
            Self::ToutiaoReader => "toutiao_reader",
            Self::RssFeed => "rss_feed",
        }
    }

    /// Parses an exact `snake_case` literal; no trimming and no case folding.
    ///
    /// Returns `None` for an unknown or differently written value.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == raw)
    }

    /// Returns whether this kind is a deferred vendor transport.
    ///
    /// [`SourceKind::LocalFiles`] is the only implemented kind; every other
    /// kind is registered as an [`UnsupportedSourceProvider`].
    #[must_use]
    pub fn is_deferred(self) -> bool {
        !matches!(self, Self::LocalFiles)
    }
}

impl fmt::Display for SourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One provider-neutral fetch request.
///
/// Every field defaults, so a stored request tolerates missing keys and
/// unknown keys are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FetchRequest {
    /// Service id used to namespace the fetched sources.
    pub service: String,
    /// Provider-specific source expression, e.g. a directory for local files.
    pub source: String,
    /// Maximum number of sources wanted; `0` means the provider's own default.
    pub limit: usize,
}

impl FetchRequest {
    /// Builds a request with the provider's default limit.
    #[must_use]
    pub fn new(service: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            service: service.into(),
            source: source.into(),
            limit: 0,
        }
    }
}

/// One source materialised by a provider.
///
/// The shape mirrors the fields the local scan produces, so the store can
/// dedupe by `source_id` or `content_hash` without knowing the provider.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FetchedSource {
    /// Stable `src_<32 hex>` identifier derived from the normalised locator.
    pub source_id: String,
    /// Kind of provider that produced the source.
    pub kind: SourceKind,
    /// Service id the source is namespaced under.
    pub service: String,
    /// Display title of the source.
    pub title: String,
    /// Provider-specific locator of the source (an absolute path for local files).
    pub locator: String,
    /// Location shown for the source, with `/` separators; empty falls back to `source_id`.
    pub relative_path: String,
    /// Provider revision marker; the content digest for local files.
    pub revision: String,
    /// SHA-256 hex digest (64 lowercase characters) of the indexed text.
    pub content_hash: String,
    /// Size of the indexed content in bytes.
    pub size_bytes: u64,
    /// Indexed text content of the source.
    pub content: String,
    /// Provider-specific extra facts, always rendered as strings.
    pub metadata: BTreeMap<String, String>,
}

/// A provider that materialises [`FetchedSource`] values for one [`SourceKind`].
///
/// Implementations must be `Send + Sync` because the registry shares them
/// across tasks. Deferred kinds are served by [`UnsupportedSourceProvider`],
/// which reports [`SourceProvider::is_available`] as `false` instead of
/// pretending to work.
#[async_trait::async_trait]
pub trait SourceProvider: Send + Sync {
    /// Returns the source kind this provider serves.
    #[must_use]
    fn kind(&self) -> SourceKind;

    /// Fetches the sources described by `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::unsupported`] when the provider's transport is
    /// deferred, [`Error::validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when the request or its source expression is unusable, and any
    /// transport error the provider's own IO reports.
    async fn fetch(&self, request: &FetchRequest) -> Result<Vec<FetchedSource>>;

    /// Returns whether the provider can attempt a fetch.
    ///
    /// Deferred providers report `false` without attempting a call, so a
    /// caller can skip them without provoking an error.
    #[must_use]
    fn is_available(&self) -> bool {
        true
    }
}

/// The real provider: a local-directory scan over plain-text files.
///
/// The configured [`LocalDirectoryConfig`] is used as-is when it names a
/// directory; otherwise the directory and service come from the
/// [`FetchRequest`], which lets one provider serve many roots. The scan's
/// `max_files` bounds the result, so [`FetchRequest::limit`] is advisory here
/// and is left to paged vendor transports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalFilesProvider {
    config: LocalDirectoryConfig,
}

impl LocalFilesProvider {
    /// Builds a provider over `config`.
    #[must_use]
    pub fn new(config: LocalDirectoryConfig) -> Self {
        Self { config }
    }

    /// Returns the configured local-directory scan.
    #[must_use]
    pub fn config(&self) -> &LocalDirectoryConfig {
        &self.config
    }

    /// Resolves the scan configuration for one request.
    ///
    /// A configured directory wins over the request; a blank configured
    /// directory is filled from `request.source` and `request.service`, with a
    /// blank service becoming [`DEFAULT_LOCAL_SERVICE`].
    fn resolve_config(&self, request: &FetchRequest) -> LocalDirectoryConfig {
        if !self.config.directory.as_os_str().is_empty() {
            return self.config.clone();
        }
        let service = request.service.trim();
        let service = if service.is_empty() {
            DEFAULT_LOCAL_SERVICE
        } else {
            service
        };
        LocalDirectoryConfig {
            service: service.to_string(),
            directory: PathBuf::from(&request.source),
            ..self.config.clone()
        }
    }
}

#[async_trait::async_trait]
impl SourceProvider for LocalFilesProvider {
    fn kind(&self) -> SourceKind {
        SourceKind::LocalFiles
    }

    async fn fetch(&self, request: &FetchRequest) -> Result<Vec<FetchedSource>> {
        let config = self.resolve_config(request);
        let report = ingest_local_directory(&config)?;
        Ok(report.documents.into_iter().map(fetched_source).collect())
    }
}

/// Maps one ingested document onto the provider-neutral shape.
fn fetched_source(document: IngestedDocument) -> FetchedSource {
    let locator =
        normalize_source_locator(&document.locator).unwrap_or_else(|_| document.locator.clone());
    let source_id = source_id_for_locator(&locator).unwrap_or_else(|_| document.source_id.clone());
    let mut metadata = BTreeMap::new();
    metadata.insert("relative_path".to_string(), document.relative_path.clone());
    metadata.insert("service".to_string(), document.service.clone());
    metadata.insert("size_bytes".to_string(), document.size_bytes.to_string());
    FetchedSource {
        source_id,
        kind: SourceKind::LocalFiles,
        service: document.service,
        title: document.title,
        locator,
        relative_path: document.relative_path,
        revision: document.content_hash.clone(),
        content_hash: document.content_hash,
        size_bytes: document.size_bytes,
        content: document.text,
        metadata,
    }
}

/// The deferred provider registered for a vendor-transport kind.
///
/// It exists so [`SourceProviderRegistry::with_unsupported_vendors`] can
/// complete the kind table without pretending a fetch could work. Constructing
/// it for [`SourceKind::LocalFiles`] is a programming error: the registry never
/// does so, and the error surfaces as [`Error::unsupported`] instead of a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedSourceProvider {
    kind: SourceKind,
}

impl UnsupportedSourceProvider {
    /// Builds a deferred provider for `kind`.
    #[must_use]
    pub fn new(kind: SourceKind) -> Self {
        Self { kind }
    }

    /// Returns the kind this provider rejects; inherent so callers need no trait import.
    ///
    /// The `&self` receiver mirrors [`SourceProvider::kind`], so callers can
    /// use either form; borrowing a one-byte value is deliberate here.
    #[must_use]
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn kind(&self) -> SourceKind {
        self.kind
    }
}

#[async_trait::async_trait]
impl SourceProvider for UnsupportedSourceProvider {
    fn kind(&self) -> SourceKind {
        self.kind
    }

    fn is_available(&self) -> bool {
        false
    }

    async fn fetch(&self, _request: &FetchRequest) -> Result<Vec<FetchedSource>> {
        Err(Error::unsupported(unsupported_message(self.kind)))
    }
}

/// Renders the deferred-fetch message for `kind`.
///
/// The wording is part of the contract: every deferred path — a provider's
/// fetch and the registry's unregistered-kind path — returns exactly this
/// message as its [`Error::unsupported`] payload.
#[must_use]
pub fn unsupported_message(kind: SourceKind) -> String {
    format!(
        "`personal_context`: `{}` source fetcher is not implemented; vendor SDK/remote transport is deferred",
        kind.as_str()
    )
}

/// Kind-indexed set of [`SourceProvider`] handles.
///
/// Registration replaces a provider of the same kind, and lookups clone an
/// `Arc`, so a registry can be shared while a fetch is in flight.
#[derive(Clone, Default)]
pub struct SourceProviderRegistry {
    providers: BTreeMap<SourceKind, Arc<dyn SourceProvider>>,
}

impl SourceProviderRegistry {
    /// Builds an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `provider`, replacing any provider already serving its kind.
    pub fn register(&mut self, provider: Arc<dyn SourceProvider>) {
        self.providers.insert(provider.kind(), provider);
    }

    /// Returns the provider registered for `kind`, if any.
    #[must_use]
    pub fn get(&self, kind: SourceKind) -> Option<Arc<dyn SourceProvider>> {
        self.providers.get(&kind).map(Arc::clone)
    }

    /// Returns every registered kind, sorted in declaration order.
    #[must_use]
    pub fn kinds(&self) -> Vec<SourceKind> {
        self.providers.keys().copied().collect()
    }

    /// Returns the number of registered providers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Returns whether no provider is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Registers an [`UnsupportedSourceProvider`] for every deferred kind.
    ///
    /// The result resolves all eight [`SourceKind`] values while keeping the
    /// vendor paths typed: fetching one returns [`unsupported_message`].
    #[must_use]
    pub fn with_unsupported_vendors(mut self) -> Self {
        for kind in SourceKind::ALL
            .into_iter()
            .filter(|kind| kind.is_deferred())
        {
            self.register(Arc::new(UnsupportedSourceProvider::new(kind)));
        }
        self
    }

    /// Fetches `kind` through its registered provider.
    ///
    /// # Errors
    ///
    /// Returns [`Error::unsupported`] carrying [`unsupported_message`] when no
    /// provider is registered for `kind`, including every deferred vendor kind
    /// of an empty registry; otherwise the registered provider's own error.
    pub async fn fetch(
        &self,
        kind: SourceKind,
        request: &FetchRequest,
    ) -> Result<Vec<FetchedSource>> {
        match self.providers.get(&kind) {
            Some(provider) => provider.fetch(request).await,
            None => Err(Error::unsupported(unsupported_message(kind))),
        }
    }
}

impl fmt::Debug for SourceProviderRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceProviderRegistry")
            .field("kinds", &self.kinds())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::Path;

    use openjiuwen_common::error::ErrorKind;
    use tempfile::tempdir;

    use super::*;
    use crate::rails::personal_context::source::source_id_for_locator;
    use openjiuwen_sysop::util::sha256_hex;

    /// The seven vendor kinds, in declaration order.
    fn vendor_kinds() -> [SourceKind; 7] {
        [
            SourceKind::Github,
            SourceKind::Gitcode,
            SourceKind::Feishu,
            SourceKind::BrowserBookmarks,
            SourceKind::ZhihuReader,
            SourceKind::ToutiaoReader,
            SourceKind::RssFeed,
        ]
    }

    /// Writes `body` into `dir/name` and returns the path.
    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    /// A provider that returns canned sources, for registry delegation tests.
    struct FakeProvider {
        kind: SourceKind,
        sources: Vec<FetchedSource>,
    }

    impl FakeProvider {
        fn new(kind: SourceKind, sources: Vec<FetchedSource>) -> Self {
            Self { kind, sources }
        }
    }

    #[async_trait::async_trait]
    impl SourceProvider for FakeProvider {
        fn kind(&self) -> SourceKind {
            self.kind
        }

        async fn fetch(&self, _request: &FetchRequest) -> Result<Vec<FetchedSource>> {
            Ok(self.sources.clone())
        }
    }

    fn sample_source(kind: SourceKind) -> FetchedSource {
        FetchedSource {
            source_id: "src_00000000000000000000000000000000".to_string(),
            kind,
            title: "Sample".to_string(),
            ..FetchedSource::default()
        }
    }

    #[test]
    fn every_literal_round_trips() {
        assert_eq!(SourceKind::ALL.len(), 8);
        let expected = [
            "local_files",
            "github",
            "gitcode",
            "feishu",
            "browser_bookmarks",
            "zhihu_reader",
            "toutiao_reader",
            "rss_feed",
        ];
        for (kind, literal) in SourceKind::ALL.into_iter().zip(expected) {
            assert_eq!(kind.as_str(), literal);
            assert_eq!(SourceKind::parse(literal), Some(kind));
            assert_eq!(kind.to_string(), literal);
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{literal}\""));
            let decoded: SourceKind = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, kind);
        }
    }

    #[test]
    fn unknown_and_miscased_literals_do_not_parse() {
        assert_eq!(SourceKind::parse(""), None);
        assert_eq!(SourceKind::parse("unknown"), None);
        assert_eq!(SourceKind::parse("local-files"), None);
        assert_eq!(SourceKind::parse("LOCAL_FILES"), None);
        assert_eq!(SourceKind::parse(" local_files"), None);
        assert_eq!(SourceKind::parse("local_files "), None);
        assert!(serde_json::from_str::<SourceKind>("\"unknown\"").is_err());
    }

    #[test]
    fn only_local_files_is_implemented() {
        for kind in SourceKind::ALL {
            assert_eq!(kind.is_deferred(), kind != SourceKind::LocalFiles);
        }
    }

    #[test]
    fn unsupported_messages_name_each_vendor_verbatim() {
        let expected = [
            (
                SourceKind::Github,
                "`personal_context`: `github` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
            (
                SourceKind::Gitcode,
                "`personal_context`: `gitcode` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
            (
                SourceKind::Feishu,
                "`personal_context`: `feishu` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
            (
                SourceKind::BrowserBookmarks,
                "`personal_context`: `browser_bookmarks` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
            (
                SourceKind::ZhihuReader,
                "`personal_context`: `zhihu_reader` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
            (
                SourceKind::ToutiaoReader,
                "`personal_context`: `toutiao_reader` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
            (
                SourceKind::RssFeed,
                "`personal_context`: `rss_feed` source fetcher is not implemented; vendor SDK/remote transport is deferred",
            ),
        ];
        assert_eq!(expected.len(), vendor_kinds().len());
        for (kind, message) in expected {
            assert_eq!(unsupported_message(kind), message);
        }
    }

    #[test]
    fn fetch_request_round_trips_with_defaults() {
        let request = FetchRequest::new("notes", "/tmp/notes");
        assert_eq!(request.limit, 0);
        assert_eq!(request.service, "notes");
        assert_eq!(request.source, "/tmp/notes");

        let json = serde_json::to_string(&request).unwrap();
        let decoded: FetchRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);

        let partial: FetchRequest = serde_json::from_str("{\"source\":\"/tmp/x\"}").unwrap();
        assert_eq!(partial.service, "");
        assert_eq!(partial.limit, 0);
        assert_eq!(partial.source, "/tmp/x");
    }

    #[test]
    fn fetched_source_round_trips_and_defaults_its_kind() {
        let mut source = sample_source(SourceKind::Feishu);
        source
            .metadata
            .insert("service".to_string(), "vendor".to_string());
        let json = serde_json::to_string(&source).unwrap();
        let decoded: FetchedSource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, source);

        let empty: FetchedSource = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, FetchedSource::default());
        assert_eq!(empty.kind, SourceKind::LocalFiles);
    }

    #[tokio::test]
    async fn unsupported_provider_rejects_every_vendor() {
        for kind in vendor_kinds() {
            let provider = UnsupportedSourceProvider::new(kind);
            assert_eq!(provider.kind(), kind);
            assert_eq!(SourceProvider::kind(&provider), kind);
            assert!(!provider.is_available());

            let error = provider.fetch(&FetchRequest::default()).await.unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(matches!(
                &error,
                Error::Unsupported(message) if *message == unsupported_message(kind)
            ));
            assert_eq!(
                error.to_string(),
                format!("unsupported: {}", unsupported_message(kind))
            );
        }
    }

    #[tokio::test]
    async fn unsupported_provider_still_rejects_local_files() {
        let provider = UnsupportedSourceProvider::new(SourceKind::LocalFiles);

        let error = provider.fetch(&FetchRequest::default()).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(matches!(
            &error,
            Error::Unsupported(message) if *message == unsupported_message(SourceKind::LocalFiles)
        ));
        assert!(!provider.is_available());
    }

    #[tokio::test]
    async fn local_provider_maps_ingested_documents() {
        let temp = tempdir().unwrap();
        let body = "# Daily Note\n\nremember the milk\n";
        let path = write(temp.path(), "note.md", body);
        let provider = LocalFilesProvider::new(LocalDirectoryConfig::new("notes", temp.path()));

        assert_eq!(provider.kind(), SourceKind::LocalFiles);
        assert_eq!(provider.config().service, "notes");
        assert!(provider.is_available());

        let fetched = provider.fetch(&FetchRequest::default()).await.unwrap();

        assert_eq!(fetched.len(), 1);
        let source = &fetched[0];
        assert_eq!(source.kind, SourceKind::LocalFiles);
        assert_eq!(source.title, "Daily Note");
        assert!(source.source_id.starts_with("src_"));
        assert_eq!(source.source_id.len(), 4 + 32);
        assert_eq!(
            source.source_id,
            source_id_for_locator(&source.locator).unwrap()
        );
        let canonical = std::fs::canonicalize(&path).unwrap();
        assert_eq!(PathBuf::from(&source.locator), canonical);
        assert_eq!(source.locator, canonical.to_string_lossy());
        assert_eq!(source.content, "# Daily Note\n\nremember the milk");
        assert_eq!(source.revision, source.content_hash);
        assert_eq!(source.content_hash, sha256_hex(body.as_bytes()));
        assert_eq!(source.content_hash.len(), 64);
        assert_eq!(source.metadata["relative_path"], "note.md");
        assert_eq!(source.metadata["service"], "notes");
        assert_eq!(source.metadata["size_bytes"], body.len().to_string());
    }

    #[tokio::test]
    async fn local_provider_falls_back_to_the_request_source() {
        let temp = tempdir().unwrap();
        write(temp.path(), "alpha.txt", "alpha");

        let provider = LocalFilesProvider::new(LocalDirectoryConfig::default());
        let request = FetchRequest::new("   ", temp.path().to_string_lossy());

        let fetched = provider.fetch(&request).await.unwrap();

        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].metadata["service"], DEFAULT_LOCAL_SERVICE);
        assert_eq!(fetched[0].content, "alpha");
    }

    #[tokio::test]
    async fn configured_directory_wins_over_the_request_source() {
        let temp = tempdir().unwrap();
        write(temp.path(), "alpha.txt", "alpha");
        let provider = LocalFilesProvider::new(LocalDirectoryConfig::new("notes", temp.path()));
        let missing = temp.path().join("does-not-exist");
        let request = FetchRequest::new("other", missing.to_string_lossy());

        let fetched = provider.fetch(&request).await.unwrap();

        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].metadata["service"], "notes");
    }

    #[tokio::test]
    async fn a_missing_directory_is_a_validation_error() {
        let temp = tempdir().unwrap();
        let provider = LocalFilesProvider::new(LocalDirectoryConfig::new(
            "notes",
            temp.path().join("missing"),
        ));

        let error = provider.fetch(&FetchRequest::default()).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn registry_registers_and_replaces_by_kind() {
        let mut registry = SourceProviderRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.kinds(), Vec::<SourceKind>::new());

        registry.register(Arc::new(FakeProvider::new(
            SourceKind::Github,
            vec![sample_source(SourceKind::Github)],
        )));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert_eq!(registry.kinds(), vec![SourceKind::Github]);
        assert_eq!(
            registry.get(SourceKind::Github).unwrap().kind(),
            SourceKind::Github
        );
        assert!(registry.get(SourceKind::Feishu).is_none());

        registry.register(Arc::new(FakeProvider::new(SourceKind::Github, Vec::new())));
        assert_eq!(registry.len(), 1);

        let fetched = registry
            .fetch(SourceKind::Github, &FetchRequest::default())
            .await
            .unwrap();
        assert_eq!(fetched, Vec::<FetchedSource>::new());
    }

    #[tokio::test]
    async fn registry_reports_unregistered_kinds_as_unsupported() {
        let registry = SourceProviderRegistry::new();

        for kind in SourceKind::ALL {
            let error = registry
                .fetch(kind, &FetchRequest::default())
                .await
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(matches!(
                &error,
                Error::Unsupported(message) if *message == unsupported_message(kind)
            ));
        }
    }

    #[tokio::test]
    async fn registry_with_unsupported_vendors_covers_every_vendor() {
        let registry = SourceProviderRegistry::new().with_unsupported_vendors();

        assert_eq!(registry.len(), vendor_kinds().len());
        assert_eq!(registry.kinds(), vendor_kinds().to_vec());
        assert!(registry.get(SourceKind::LocalFiles).is_none());

        for kind in vendor_kinds() {
            let provider = registry.get(kind).expect("vendor kind is registered");
            assert_eq!(provider.kind(), kind);
            assert!(!provider.is_available());

            let error = registry
                .fetch(kind, &FetchRequest::default())
                .await
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(matches!(
                &error,
                Error::Unsupported(message) if *message == unsupported_message(kind)
            ));
        }
    }

    #[tokio::test]
    async fn registry_delegates_to_a_registered_provider() {
        let mut registry = SourceProviderRegistry::new();
        registry.register(Arc::new(FakeProvider::new(
            SourceKind::RssFeed,
            vec![sample_source(SourceKind::RssFeed)],
        )));

        let fetched = registry
            .fetch(
                SourceKind::RssFeed,
                &FetchRequest::new("feeds", "https://example.test/rss"),
            )
            .await
            .unwrap();

        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].kind, SourceKind::RssFeed);
        assert_eq!(fetched[0].title, "Sample");
    }

    #[test]
    fn registry_debug_lists_kinds() {
        let registry = SourceProviderRegistry::new().with_unsupported_vendors();

        let rendered = format!("{registry:?}");

        assert!(rendered.contains("SourceProviderRegistry"));
        assert!(rendered.contains("Feishu"));
    }
}
