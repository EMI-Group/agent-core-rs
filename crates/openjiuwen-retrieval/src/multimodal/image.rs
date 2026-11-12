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

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::common::{Document, Metadata};
use crate::loaders::{file_extension, DocumentLoader};

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
        let extension = file_extension(source);
        if !IMAGE_EXTENSIONS.contains(&extension.as_str()) {
            return Err(Error::Unsupported(format!(
                "Unsupported image format: {extension}, only [{}] are supported",
                IMAGE_EXTENSIONS.join(", ")
            )));
        }
        let bytes = match tokio::fs::read(source).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotFound(format!("File {source} does not exist")));
            }
            Err(error) => {
                return Err(Error::Transport(format!("reading {source}: {error}")));
            }
        };

        // Python returns `""` for a missing model; a zero-byte file has nothing to
        // caption, so it keeps an empty text instead of a file-name placeholder.
        let text = if bytes.is_empty() {
            String::new()
        } else {
            self.captioner
                .caption(std::slice::from_ref(&source.to_owned()))
                .await?
                .into_iter()
                .filter(|caption| !caption.is_empty())
                .collect::<Vec<String>>()
                .join("\n")
        };

        let mime = guess_mime_type(source).unwrap_or("application/octet-stream");
        let mut metadata = Metadata::new();
        metadata.insert("image_path".to_owned(), Value::String(source.to_owned()));
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
                Value::String(to_data_uri(Modality::Image, mime, &bytes)),
            );
        }
        Ok(vec![Document::new(text).with_metadata(metadata)])
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
}
