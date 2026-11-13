//! Offline image loader producing one captioned [`Document`] per image file.
//!
//! Port of the Python `ImageParser`
//! (`indexing/processor/parser/image_parser.py`): the parser runs an
//! `ImageCaptioner` over the image and returns a single `Document` whose `text`
//! is the captions joined by newlines and whose metadata exposes `image_path` so
//! a multimodal embedder can pick the file up later.
//!
//! Differences from Python, all documented in the module [`super`] docs:
//! Python silently returns `[]` on any failure and copies the image into a
//! shared `images/` directory; the Rust loader reports `NotFound`/`Unsupported`
//! instead of swallowing them, never copies files, and defaults to the offline
//! captioner (its metadata marks the caption as `caption_status: "offline"`).
//!
//! ```
//! # use std::sync::Arc;
//! # use openjiuwen_retrieval::loaders::{DocumentBytes, DocumentLoader, LoadOptions};
//! # use openjiuwen_retrieval::multimodal::ImageLoader;
//! # async fn run() -> openjiuwen_common::error::Result<()> {
//! let source = DocumentBytes::new("cat.png", b"\x89PNG\r\n\x1a\n".to_vec());
//! let documents = ImageLoader::new().load_bytes(&source, &LoadOptions::new()).await?;
//! assert_eq!(documents[0].metadata["image_path"], "cat.png");
//! # Ok(())
//! # }
//! ```
//!
//! [`DocumentLoader::load_bytes`] reads the same metadata as the path loader
//! (with `image_path` set to [`DocumentBytes::name`], since bytes have no path),
//! so in-memory uploads never need a temporary file.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::common::{Document, Metadata};
use crate::loaders::{
    apply_load_options, file_extension, DocumentBytes, DocumentLoader, LoadOptions,
};

use super::caption::{ImageCaptioner, OfflineCaptioner};
use super::document::to_data_uri;
use super::modality::{guess_mime_type, Modality};

/// The image extensions the Python parser registers.
pub const IMAGE_EXTENSIONS: [&str; 6] = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".jfif"];

/// Loads `.png .jpg .jpeg .webp .gif .jfif` files as captioned documents.
///
/// The default captioner is [`OfflineCaptioner`], so loading never needs a model
/// or the network. Inject a real captioner with [`ImageLoader::with_captioner`].
pub struct ImageLoader {
    captioner: Arc<dyn ImageCaptioner>,
    include_data_uri: bool,
}

impl ImageLoader {
    /// Creates a loader with the offline captioner and no `data_uri` metadata.
    #[must_use]
    pub fn new() -> Self {
        Self {
            captioner: Arc::new(OfflineCaptioner::new()),
            include_data_uri: false,
        }
    }

    /// Replaces the captioner used for every image.
    #[must_use]
    pub fn with_captioner(mut self, captioner: Arc<dyn ImageCaptioner>) -> Self {
        self.captioner = captioner;
        self
    }

    /// Also stamps the image bytes as a base64 `data_uri` metadata entry.
    #[must_use]
    pub fn with_data_uri(mut self, include: bool) -> Self {
        self.include_data_uri = include;
        self
    }

    /// Returns `true` for the extensions the loader accepts.
    #[must_use]
    pub fn is_supported(source: &str) -> bool {
        IMAGE_EXTENSIONS.contains(&file_extension(source).as_str())
    }

    /// Rejects extensions the image parser does not register.
    fn check_extension(extension: &str) -> Result<()> {
        if IMAGE_EXTENSIONS.contains(&extension) {
            return Ok(());
        }
        Err(Error::Unsupported(format!(
            "Unsupported image format: {extension}, only [{}] are supported",
            IMAGE_EXTENSIONS.join(", ")
        )))
    }

    /// Builds the one captioned document shared by the path and byte loaders.
    ///
    /// `name` is the path or in-memory source name: it seeds the caption, the
    /// `image_path` metadata, and the MIME guess. Captioning is skipped for a
    /// zero-byte payload, which keeps the text empty instead of describing a
    /// file name that holds nothing (Python returns `""` when no model is set).
    async fn captioned_document(&self, name: &str, bytes: &[u8]) -> Result<Document> {
        let text = if bytes.is_empty() {
            String::new()
        } else {
            self.captioner
                .caption(std::slice::from_ref(&name.to_owned()))
                .await?
                .into_iter()
                .filter(|caption| !caption.is_empty())
                .collect::<Vec<String>>()
                .join("\n")
        };

        let mime = guess_mime_type(name).unwrap_or("application/octet-stream");
        let mut metadata = Metadata::new();
        metadata.insert("image_path".to_owned(), Value::String(name.to_owned()));
        metadata.insert("mime_type".to_owned(), Value::String(mime.to_owned()));
        metadata.insert("byte_size".to_owned(), Value::from(bytes.len() as u64));
        metadata.insert("modality".to_owned(), Value::String("image".to_owned()));
        metadata.insert(
            "caption_status".to_owned(),
            Value::String(self.captioner.caption_status().to_owned()),
        );
        if self.include_data_uri {
            metadata.insert(
                "data_uri".to_owned(),
                Value::String(to_data_uri(Modality::Image, mime, bytes)),
            );
        }
        Ok(Document::new(text).with_metadata(metadata))
    }
}

impl Default for ImageLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentLoader for ImageLoader {
    fn supports(&self, source: &str) -> bool {
        Self::is_supported(source)
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        Self::check_extension(&file_extension(source))?;
        let bytes = match tokio::fs::read(source).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotFound(format!("File {source} does not exist")));
            }
            Err(error) => {
                return Err(Error::Transport(format!("reading {source}: {error}")));
            }
        };
        Ok(vec![self.captioned_document(source, &bytes).await?])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        Self::check_extension(&source.extension())?;
        if source.is_empty() {
            return Err(Error::Validation(format!("File {} is empty", source.name)));
        }
        let mut documents = vec![self.captioned_document(&source.name, &source.bytes).await?];
        apply_load_options(&source.name, &mut documents, options);
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// A captioner that returns a fixed caption, proving injection works.
    struct FixedCaptioner;

    #[async_trait]
    impl ImageCaptioner for FixedCaptioner {
        async fn caption(&self, image_paths: &[String]) -> Result<Vec<String>> {
            Ok(image_paths
                .iter()
                .map(|_| "a red square".to_owned())
                .collect())
        }
    }

    fn write_image(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path.to_str().unwrap().to_owned()
    }

    #[tokio::test]
    async fn load_returns_one_captioned_document_with_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let source = write_image(&dir, "cat.png", b"\x89PNG\r\n");

        let documents = ImageLoader::new().load(&source).await.unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(document.text, "[image] cat.png");
        assert_eq!(
            document.metadata.get("image_path"),
            Some(&Value::String(source.clone()))
        );
        assert_eq!(
            document.metadata.get("mime_type"),
            Some(&Value::String("image/png".to_owned()))
        );
        assert_eq!(
            document.metadata.get("byte_size"),
            Some(&Value::from(6_u64))
        );
        assert_eq!(
            document.metadata.get("modality"),
            Some(&Value::String("image".to_owned()))
        );
        assert_eq!(
            document.metadata.get("caption_status"),
            Some(&Value::String("offline".to_owned()))
        );
        assert!(document.metadata.get("data_uri").is_none());
        assert_eq!(document.id.len(), 36);
    }

    #[tokio::test]
    async fn data_uri_metadata_is_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        let source = write_image(&dir, "dot.jpeg", b"x");

        let documents = ImageLoader::new()
            .with_data_uri(true)
            .load(&source)
            .await
            .unwrap();
        assert_eq!(
            documents[0].metadata.get("data_uri"),
            Some(&Value::String("data:image/jpeg;base64,eA==".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("mime_type"),
            Some(&Value::String("image/jpeg".to_owned()))
        );
    }

    #[tokio::test]
    async fn zero_byte_image_keeps_empty_text() {
        let dir = tempfile::tempdir().unwrap();
        let source = write_image(&dir, "empty.gif", b"");

        let documents = ImageLoader::new().load(&source).await.unwrap();
        assert_eq!(documents[0].text, "");
        assert_eq!(
            documents[0].metadata.get("byte_size"),
            Some(&Value::from(0_u64))
        );
    }

    #[tokio::test]
    async fn injected_captioner_replaces_the_offline_one() {
        let dir = tempfile::tempdir().unwrap();
        let source = write_image(&dir, "shape.webp", b"x");

        let documents = ImageLoader::new()
            .with_captioner(Arc::new(FixedCaptioner))
            .load(&source)
            .await
            .unwrap();
        assert_eq!(documents[0].text, "a red square");
        assert_eq!(
            documents[0].metadata.get("caption_status"),
            Some(&Value::String("custom".to_owned()))
        );
    }

    #[tokio::test]
    async fn missing_and_unsupported_files_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.png");
        let error = ImageLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("does not exist"));

        let error = ImageLoader::new().load("notes.pdf").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("Unsupported image format: .pdf"));
    }

    #[test]
    fn supports_the_registered_extensions_only() {
        let loader = ImageLoader::new();
        for source in ["a.png", "a.JPG", "a.jpeg", "a.webp", "a.gif", "a.jfif"] {
            assert!(loader.supports(source), "{source}");
        }
        assert!(!loader.supports("a.bmp"));
        assert!(!loader.supports("a.pdf"));
        assert!(!loader.supports("a"));
    }

    /// A real 1×1 PNG: signature plus `IHDR`, `IDAT`, and `IEND` with valid CRCs.
    const ONE_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";

    fn one_pixel_png() -> Vec<u8> {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine as _;
        STANDARD.decode(ONE_PIXEL_PNG_BASE64).unwrap()
    }

    #[tokio::test]
    async fn load_bytes_matches_the_path_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_image(&dir, "pixel.png", &one_pixel_png());

        let from_path = ImageLoader::new().load(&path).await.unwrap();
        let from_bytes = ImageLoader::new()
            .load_bytes(
                &DocumentBytes::new("pixel.png", one_pixel_png()),
                &LoadOptions::new(),
            )
            .await
            .unwrap();

        assert_eq!(from_bytes.len(), 1);
        let expected = &from_path[0];
        let actual = &from_bytes[0];
        assert_eq!(actual.text, expected.text);
        assert_eq!(actual.text, "[image] pixel.png");
        for key in [
            "mime_type",
            "byte_size",
            "modality",
            "caption_status",
            "data_uri",
        ] {
            assert_eq!(
                actual.metadata.get(key),
                expected.metadata.get(key),
                "{key}"
            );
        }
        assert_eq!(
            actual.metadata.get("mime_type"),
            Some(&Value::String("image/png".to_owned()))
        );
        assert_eq!(actual.metadata.get("byte_size"), Some(&Value::from(69_u64)));
        // The byte source has no filesystem path, so its name stands in for it.
        assert_eq!(
            actual.metadata.get("image_path"),
            Some(&Value::String("pixel.png".to_owned()))
        );
        assert_eq!(actual.id.len(), 36);
        assert_eq!(
            actual.metadata.get("doc_id").and_then(Value::as_str),
            Some(actual.id.as_str())
        );
    }

    #[tokio::test]
    async fn load_bytes_data_uri_matches_the_path_loader() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_image(&dir, "pixel.png", &one_pixel_png());

        let from_path = ImageLoader::new()
            .with_data_uri(true)
            .load(&path)
            .await
            .unwrap();
        let from_bytes = ImageLoader::new()
            .with_data_uri(true)
            .load_bytes(
                &DocumentBytes::new("pixel.png", one_pixel_png()),
                &LoadOptions::new(),
            )
            .await
            .unwrap();

        let expected = from_path[0].metadata.get("data_uri").unwrap();
        assert_eq!(from_bytes[0].metadata.get("data_uri"), Some(expected));
        let uri = expected.as_str().unwrap();
        assert!(uri.starts_with("data:image/png;base64,"), "{uri}");
        assert_eq!(
            from_bytes[0].metadata.get("mime_type"),
            Some(&Value::String("image/png".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_bytes_applies_doc_id_and_title() {
        let options = LoadOptions::new()
            .with_doc_id("img-1")
            .with_title("Figure 1");
        let documents = ImageLoader::new()
            .load_bytes(&DocumentBytes::new("pixel.png", one_pixel_png()), &options)
            .await
            .unwrap();

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].id, "img-1");
        assert_eq!(
            documents[0].metadata.get("doc_id"),
            Some(&Value::String("img-1".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("title"),
            Some(&Value::String("Figure 1".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("image_path"),
            Some(&Value::String("pixel.png".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_bytes_never_touches_the_filesystem() {
        let source = DocumentBytes::new("upload/absent/cat.png", one_pixel_png());
        let documents = ImageLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "[image] cat.png");
        assert_eq!(
            documents[0].metadata.get("image_path"),
            Some(&Value::String("upload/absent/cat.png".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_bytes_rejects_unsupported_extensions() {
        let source = DocumentBytes::new("notes.pdf", b"bytes".to_vec());
        let error = ImageLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("Unsupported image format: .pdf"),
            "unexpected message: {error}"
        );

        let source = DocumentBytes::new("no-extension", b"bytes".to_vec());
        let error = ImageLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("Unsupported image format: ,"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn load_bytes_rejects_empty_bytes() {
        let source = DocumentBytes::new("empty.png", Vec::new());
        let error = ImageLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("File empty.png is empty"),
            "unexpected message: {error}"
        );
    }
}
