//! Automatic dispatch to the right loader by URL or file extension.
//!
//! Port of the Python `AutoLinkParser` / `AutoFileParser` / `AutoParser` trio: a
//! URL goes to the [`UrlLoader`], a local path goes to the loader registered for
//! its extension, and the resulting documents carry the provenance metadata
//! Python stamps in `AutoFileParser.parse` (`doc_id`, `title`, `file_path`,
//! `file_ext`). Unsupported formats are rejected instead of returning an empty
//! list.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::{debug, warn};

use crate::common::Document;

use super::{
    file_extension, file_stem, generate_document_id, is_url, CsvLoader, DocumentLoader, HtmlLoader,
    JsonLoader, MarkdownLoader, TextLoader, UrlLoader,
};

/// A lowercase extension map (`".txt"`) to loaders.
///
/// Mirrors Python's `_PARSER_REGISTRY`: the first registration for an extension
/// wins, so a later [`LoaderRegistry::register`] cannot replace a default loader.
#[derive(Default)]
pub struct LoaderRegistry {
    loaders: HashMap<String, Arc<dyn DocumentLoader>>,
}

impl LoaderRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loaders: HashMap::new(),
        }
    }

    /// Creates a registry with the built-in text/Markdown/JSON/CSV/HTML loaders.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(&[".txt"], Arc::new(TextLoader::new()));
        registry.register(&[".md", ".markdown"], Arc::new(MarkdownLoader::new()));
        registry.register(&[".json"], Arc::new(JsonLoader::new()));
        registry.register(&[".csv", ".tsv"], Arc::new(CsvLoader::new()));
        registry.register(&[".html", ".htm"], Arc::new(HtmlLoader::new()));
        registry
    }

    /// Registers `loader` for every extension in `extensions`.
    ///
    /// Extensions are lowercased and given a leading dot (`"TXT"` and `".txt"`
    /// are the same key). Extensions that are already registered are skipped, so
    /// the first registration wins (Python parity).
    ///
    /// The loader arrives by value because one loader usually serves several
    /// extensions and the registry keeps a shared reference per extension.
    #[allow(clippy::needless_pass_by_value)]
    pub fn register(&mut self, extensions: &[&str], loader: Arc<dyn DocumentLoader>) {
        for extension in extensions {
            let normalized = normalize_extension(extension);
            if normalized.is_empty() {
                warn!(extension, "ignoring an empty loader extension");
                continue;
            }
            if self.loaders.contains_key(&normalized) {
                debug!(
                    extension = %normalized,
                    "keeping the loader registered first for this extension"
                );
                continue;
            }
            debug!(extension = %normalized, "registered a document loader");
            self.loaders.insert(normalized, Arc::clone(&loader));
        }
    }

    /// Returns the loader registered for the extension of `path`.
    #[must_use]
    pub fn get_for_path(&self, path: &str) -> Option<Arc<dyn DocumentLoader>> {
        self.loaders.get(&file_extension(path)).cloned()
    }

    /// Returns every registered extension, sorted for deterministic messages.
    #[must_use]
    pub fn extensions(&self) -> Vec<String> {
        let mut extensions: Vec<String> = self.loaders.keys().cloned().collect();
        extensions.sort();
        extensions
    }
}

/// Normalizes one registration extension to lowercase with a leading dot.
fn normalize_extension(extension: &str) -> String {
    let bare = extension.trim().trim_start_matches('.').to_lowercase();
    if bare.is_empty() {
        String::new()
    } else {
        format!(".{bare}")
    }
}

/// Dispatches a source to a [`UrlLoader`] or an extension-specific loader.
///
/// ```no_run
/// # use openjiuwen_retrieval::loaders::AutoLoader;
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let documents = AutoLoader::new().load("notes.md").await?;
/// assert_eq!(documents.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct AutoLoader {
    /// Extension-based loaders.
    registry: LoaderRegistry,
    /// Loader used for `HTTP(S)` sources.
    url_loader: UrlLoader,
}

impl AutoLoader {
    /// Creates an auto loader with the default registry and URL loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: LoaderRegistry::with_defaults(),
            url_loader: UrlLoader::new(),
        }
    }

    /// Replaces the URL loader.
    #[must_use]
    pub fn with_url_loader(mut self, url_loader: UrlLoader) -> Self {
        self.url_loader = url_loader;
        self
    }

    /// Registers an additional loader; existing extensions are not replaced.
    pub fn register(&mut self, extensions: &[&str], loader: Arc<dyn DocumentLoader>) {
        self.registry.register(extensions, loader);
    }

    /// Returns the extension registry.
    #[must_use]
    pub fn registry(&self) -> &LoaderRegistry {
        &self.registry
    }

    /// Loads `source` (a path or an `HTTP(S)` URL).
    ///
    /// # Errors
    ///
    /// See [`AutoLoader::load_as`].
    pub async fn load(&self, source: &str) -> Result<Vec<Document>> {
        self.load_as(source, None, None).await
    }

    /// Loads `source`, optionally overriding the document id and title.
    ///
    /// Mirrors Python's `parser.parse(path, doc_id, file_name=...)`: `doc_id`
    /// overrides the id the child loader generated, and `title` overrides the
    /// file-stem title stamped on documents loaded from a file. URLs are returned
    /// untouched.
    ///
    /// Single-document sources (text, Markdown, JSON, HTML) take `doc_id`
    /// verbatim, while multi-document file loaders (CSV/TSV) use it as the id
    /// base prefix — the generated ids are `"<source>_s0_r2"`/`"<source>_s0_c0"`,
    /// and the leading `source` is replaced by `doc_id`, mirroring Python's
    /// `base_id = doc_id or doc`. Every document therefore keeps a distinct id,
    /// and `metadata["doc_id"]` always equals the final id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for a missing path, [`Error::Unsupported`] for
    /// an unregistered extension, and the child loader's errors otherwise.
    pub async fn load_as(
        &self,
        source: &str,
        doc_id: Option<&str>,
        title: Option<&str>,
    ) -> Result<Vec<Document>> {
        if is_url(source) {
            return self.url_loader.load(source).await;
        }
        if tokio::fs::metadata(Path::new(source)).await.is_err() {
            return Err(Error::NotFound(format!("File {source} does not exist")));
        }
        let extension = file_extension(source);
        let Some(loader) = self.registry.get_for_path(source) else {
            let supported = self.registry.extensions();
            return Err(Error::Unsupported(format!(
                "Unsupported format: {extension}, only {supported:?} are supported"
            )));
        };

        debug!(
            file = source,
            "delegating to the loader registered for the extension"
        );
        let mut documents = loader.load(source).await?;
        let doc_id = doc_id.map(str::trim).filter(|value| !value.is_empty());
        let title = title.map_or_else(|| file_stem(source), str::to_owned);
        for document in &mut documents {
            if let Some(doc_id) = doc_id {
                apply_doc_id(source, doc_id, &mut document.id);
            }
            if document.id.trim().is_empty() {
                document.id = generate_document_id();
            }
            let id = document.id.clone();
            document
                .metadata
                .insert("doc_id".to_owned(), Value::String(id));
            document
                .metadata
                .insert("title".to_owned(), Value::String(title.clone()));
            document
                .metadata
                .insert("file_path".to_owned(), Value::String(source.to_owned()));
            document
                .metadata
                .insert("file_ext".to_owned(), Value::String(extension.clone()));
        }
        Ok(documents)
    }
}

/// Applies the caller-supplied `doc_id` to one document id.
///
/// Multi-document file loaders (CSV/TSV) generate ids prefixed with the `source`
/// string, so `doc_id` becomes the base prefix (`format!("{doc_id}{rest}")`).
/// Every other loader's id is replaced verbatim.
fn apply_doc_id(source: &str, doc_id: &str, id: &mut String) {
    match id.strip_prefix(source) {
        Some(rest) => *id = format!("{doc_id}{rest}"),
        None => doc_id.clone_into(id),
    }
}

impl Default for AutoLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentLoader for AutoLoader {
    fn supports(&self, source: &str) -> bool {
        if is_url(source) {
            return self.url_loader.supports(source);
        }
        self.registry.get_for_path(source).is_some()
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        // The inherent method (not this trait method) does the dispatch.
        AutoLoader::load(self, source).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// A loader that returns one fixed document for `.stub` files.
    struct StubLoader {
        text: &'static str,
    }

    #[async_trait]
    impl DocumentLoader for StubLoader {
        fn supports(&self, source: &str) -> bool {
            std::path::Path::new(source)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("stub"))
        }

        async fn load(&self, _source: &str) -> Result<Vec<Document>> {
            Ok(vec![Document::new(self.text)])
        }
    }

    /// A loader used to prove that the first registration wins.
    struct ShadowLoader;

    #[async_trait]
    impl DocumentLoader for ShadowLoader {
        fn supports(&self, _source: &str) -> bool {
            true
        }

        async fn load(&self, _source: &str) -> Result<Vec<Document>> {
            Ok(vec![Document::new("shadow")])
        }
    }

    #[test]
    fn normalize_extension_adds_one_dot_and_lowercases() {
        assert_eq!(normalize_extension(".TXT"), ".txt");
        assert_eq!(normalize_extension("txt"), ".txt");
        assert_eq!(normalize_extension("  .Md "), ".md");
        assert_eq!(normalize_extension(""), "");
        assert_eq!(normalize_extension("."), "");
    }

    #[test]
    fn defaults_cover_the_documented_extensions() {
        let registry = LoaderRegistry::with_defaults();
        assert_eq!(
            registry.extensions(),
            vec![
                ".csv",
                ".htm",
                ".html",
                ".json",
                ".markdown",
                ".md",
                ".tsv",
                ".txt",
            ]
        );
        assert_eq!(LoaderRegistry::new().extensions(), Vec::<String>::new());
        assert!(registry.get_for_path("data.TXT").is_some());
        assert!(registry.get_for_path("data.pdf").is_none());
    }

    #[test]
    fn first_registration_wins_and_case_is_ignored() {
        let mut registry = LoaderRegistry::new();
        registry.register(&[".stub"], Arc::new(StubLoader { text: "first" }));
        registry.register(&[".STUB"], Arc::new(ShadowLoader));
        let loader = registry.get_for_path("a.STUB").unwrap();
        assert!(loader.supports("a.stub"));
    }

    #[test]
    fn auto_loader_supports_urls_and_registered_extensions() {
        let loader = AutoLoader::new();
        assert!(DocumentLoader::supports(
            &loader,
            "https://example.com/page"
        ));
        assert!(DocumentLoader::supports(&loader, "notes.md"));
        assert!(!DocumentLoader::supports(&loader, "notes.pdf"));
        // `supports` is purely extension-based (it cannot await a file check).
        assert!(DocumentLoader::supports(&loader, "missing-file.txt"));
    }

    #[tokio::test]
    async fn load_stamps_provenance_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("readme.md");
        std::fs::write(&path, "# Title\n\nbody").unwrap();

        let documents = AutoLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(
            document.metadata.get("doc_id"),
            Some(&Value::String(document.id.clone()))
        );
        assert_eq!(
            document.metadata.get("title"),
            Some(&Value::String("readme".to_owned()))
        );
        assert_eq!(
            document.metadata.get("file_path"),
            Some(&Value::String(path.to_str().unwrap().to_owned()))
        );
        assert_eq!(
            document.metadata.get("file_ext"),
            Some(&Value::String(".md".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_as_overrides_the_title_and_doc_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("readme.txt");
        std::fs::write(&path, "body").unwrap();

        let documents = AutoLoader::new()
            .load_as(path.to_str().unwrap(), Some("doc-7"), Some("Custom"))
            .await
            .unwrap();
        assert_eq!(documents[0].id, "doc-7");
        assert_eq!(
            documents[0].metadata.get("doc_id"),
            Some(&Value::String("doc-7".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("title"),
            Some(&Value::String("Custom".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_as_uses_the_doc_id_as_the_base_prefix_for_multi_document_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.csv");
        std::fs::write(&path, "Name,Age\nAlice,30\n").unwrap();

        let documents = AutoLoader::new()
            .load_as(path.to_str().unwrap(), Some("doc-base"), None)
            .await
            .unwrap();
        let ids: Vec<&str> = documents
            .iter()
            .map(|document| document.id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["doc-base_s0_r2", "doc-base_s0_c0", "doc-base_s0_c1"]
        );
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "ids must be unique: {ids:?}");
        for document in &documents {
            assert_eq!(
                document.metadata.get("doc_id"),
                Some(&Value::String(document.id.clone()))
            );
        }
    }

    #[tokio::test]
    async fn missing_paths_and_unsupported_formats_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.txt");
        let error = AutoLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);

        let pdf = dir.path().join("paper.pdf");
        std::fs::write(&pdf, "not really a pdf").unwrap();
        let error = AutoLoader::new()
            .load(pdf.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("Unsupported format: .pdf"));
    }

    #[tokio::test]
    async fn custom_loaders_extend_the_registry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom.stub");
        std::fs::write(&path, "ignored").unwrap();

        let mut loader = AutoLoader::new();
        loader.register(&["STUB"], Arc::new(StubLoader { text: "stub text" }));
        let documents = loader.load(path.to_str().unwrap()).await.unwrap();
        assert_eq!(documents[0].text, "stub text");
        assert_eq!(
            documents[0].metadata.get("file_ext"),
            Some(&Value::String(".stub".to_owned()))
        );
    }
}
