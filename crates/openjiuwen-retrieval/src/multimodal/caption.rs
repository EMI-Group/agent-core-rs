//! Image captioning seam with an offline, deterministic fallback.
//!
//! Port of the interface of the Python `ImageCaptioner`
//! (`indexing/processor/parser/captioner.py`), which calls a vision model with
//! [`IMAGE_CAPTION_PROMPT`]-style instructions and returns `""` whenever the call
//! fails or no model is configured. The Rust port keeps the failure semantics as
//! a trait seam and ships only the offline degradation path:
//! [`OfflineCaptioner`] never performs I/O and never fails, so image ingestion
//! works without a VLM. A model-backed captioner belongs in a crate that depends
//! on `openjiuwen-foundation` (the LLM layer), not here.

use async_trait::async_trait;

use openjiuwen_common::error::Result;

/// The prompt the reference implementation sends to its vision model.
///
/// Exposed so an upper-crate captioner reuses the same wording; the offline
/// fallback in this crate does not call a model.
pub const IMAGE_CAPTION_PROMPT: &str =
    "You are an assistant specialized in document and image analysis. \
Your task is to provide a detailed, qualitative description of the provided image \
so that it can be embedded and used for semantic retrieval. \
Describe all visible content including text, figures, charts, tables, diagrams, and layout. \
Focus on what the image conveys and means, not just what it literally depicts. \
Do not include any preamble — output only the description.";

/// Generates retrieval text for image files.
///
/// Mirrors Python's `ImageCaptioner.caption_images`: one caption per input path,
/// in input order.
#[async_trait]
pub trait ImageCaptioner: Send + Sync {
    /// Captions every path in `image_paths`, returning one entry per path.
    ///
    /// # Errors
    ///
    /// Implementations that call a model or read the filesystem may fail; the
    /// offline fallback never does.
    async fn caption(&self, image_paths: &[String]) -> Result<Vec<String>>;

    /// Short label describing the caption source, stamped into image document
    /// metadata as `caption_status` (`"offline"` for the fallback).
    fn caption_status(&self) -> &'static str {
        "custom"
    }
}

/// Deterministic offline captioner used when no vision model is available.
///
/// Returns `format!("{prefix}{file_name}")` per path — enough text for the chunk
/// to be embedded and found by file name, and never dependent on the file
/// existing or on a network call. This is the documented degradation path: the
/// caption is *not* a description of the image content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineCaptioner {
    prefix: String,
}

/// The caption prefix used by [`OfflineCaptioner::new`].
pub const DEFAULT_CAPTION_PREFIX: &str = "[image] ";

impl OfflineCaptioner {
    /// Creates the default captioner (`"[image] "` prefix).
    #[must_use]
    pub fn new() -> Self {
        Self::with_prefix(DEFAULT_CAPTION_PREFIX)
    }

    /// Creates a captioner with a custom prefix.
    #[must_use]
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    /// The prefix prepended to every caption.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

impl Default for OfflineCaptioner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageCaptioner for OfflineCaptioner {
    async fn caption(&self, image_paths: &[String]) -> Result<Vec<String>> {
        Ok(image_paths
            .iter()
            .map(|path| format!("{}{}", self.prefix, file_name(path)))
            .collect())
    }

    fn caption_status(&self) -> &'static str {
        "offline"
    }
}

/// Returns the last path component of `path` (both separators are recognized).
fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn offline_captions_are_deterministic_and_ordered() {
        let captioner = OfflineCaptioner::new();
        let paths = vec![
            "/tmp/photos/cat.png".to_owned(),
            "C:\\photos\\dog.JPG".to_owned(),
            "scan.jpeg".to_owned(),
        ];
        let captions = captioner.caption(&paths).await.unwrap();
        assert_eq!(
            captions,
            vec!["[image] cat.png", "[image] dog.JPG", "[image] scan.jpeg"]
        );
        assert_eq!(
            captioner.caption(&paths).await.unwrap(),
            captioner.caption(&paths).await.unwrap()
        );
        assert_eq!(captioner.caption(&[]).await.unwrap(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn custom_prefix_is_applied() {
        let captioner = OfflineCaptioner::with_prefix("fig: ");
        assert_eq!(captioner.prefix(), "fig: ");
        assert_eq!(
            captioner.caption(&["a/b.png".to_owned()]).await.unwrap(),
            vec!["fig: b.png"]
        );
        assert_eq!(OfflineCaptioner::default().caption_status(), "offline");
    }

    #[test]
    fn prompt_matches_the_reference_wording() {
        assert!(IMAGE_CAPTION_PROMPT.starts_with("You are an assistant specialized"));
        assert!(IMAGE_CAPTION_PROMPT.ends_with("output only the description."));
        assert_eq!(DEFAULT_CAPTION_PREFIX, "[image] ");
    }
}
