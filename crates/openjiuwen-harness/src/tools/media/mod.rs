//! Media metadata tools: `image_metadata` and `audio_metadata`.
//!
//! Pure header parsing (no decoding crates) for the formats the model can be pointed at:
//! PNG, JPEG, GIF, WebP and BMP (images), plus WAV, FLAC, MP3 and OGG (audio). Every field is
//! read from the container header with bounds-checked byte access, so a truncated or malformed
//! file becomes a `success: false` payload naming the problem rather than a panic.
//!
//! Ports the metadata half of `openjiuwen/harness/tools/multimodal/` (`audio.py`
//! `AudioMetadataTool`, `vision.py`) as a model-free, offline subset.
//!
//! # Deviations from Python
//!
//! - No OCR, transcription, visual Q&A or ACR track identification: Python delegates those to a
//!   vision/audio model or an external service, and this crate stays offline and model-free.
//! - No URL inputs: `image_path`/`audio_path` are local paths, resolved inside the workspace.
//!   The Python parameter names (`image_path_or_url`, `audio_path_or_url`) are accepted as
//!   aliases, but an `http(s)` value fails the workspace path check.
//! - Metadata only: nothing is decoded, no pixel or sample is read, and the reported duration is
//!   derived from declared sizes, frame counts and granule positions.
//! - Detection is by magic bytes, never by file extension, and an `ID3v2` tag in front of an MP3 or
//!   FLAC stream is skipped rather than interpreted.
//!
//! # Public API
//!
//! - [`ImageMetadataTool`] — model-facing `image_metadata`; `new(env, language)`, `card_for`.
//! - [`AudioMetadataTool`] — model-facing `audio_metadata`; `new(env, language)`, `card_for`.
//! - [`parse_image_header`] / [`parse_audio_header`] — the format dispatch behind the tools.
//! - [`Cursor`], [`Failure`], [`MediaKind`], [`ParseResult`] — the shared header-parsing helpers.
//!
//! Both tools resolve model-supplied paths through [`ToolEnv::resolve_path`](super::support::ToolEnv::resolve_path)
//! and read the bytes from the resolved path, never from a raw `Path::new(model_input)`.

// The format modules repeat their format name (`png::PngInfo`, `wav::WavInfo`), which is the
// clearest spelling for both the module path and the flat re-export.
#![allow(clippy::module_name_repetitions)]
// Tool payloads report expected failures, so the `ToolPayload` envelope is a legitimate `Err`
// type here; boxing it would change every caller of the tool modules.
#![allow(clippy::result_large_err)]

pub mod audio;
pub mod image;
pub mod parse;

#[cfg(test)]
mod fixtures;

use std::path::Path;

use serde_json::{json, Value};

use super::support::ToolPayload;

pub use audio::{flac::FlacInfo, mp3::Mp3Info, ogg::OggInfo, wav::WavInfo};
pub use audio::{parse_audio_header, AudioMetadata, AudioMetadataArgs, AudioMetadataTool};
pub use image::{bmp::BmpInfo, gif::GifInfo, jpeg::JpegInfo, png::PngInfo, webp::WebpInfo};
pub use image::{parse_image_header, ImageMetadata, ImageMetadataArgs, ImageMetadataTool};
pub use parse::{Cursor, Failure, MediaKind, ParseResult};

/// Largest media file the metadata tools read into memory, in bytes (256 MiB).
const MAX_MEDIA_BYTES: u64 = 256 * 1024 * 1024;

/// Reads the bytes of an already-resolved media file.
///
/// Both tools need the whole file for the header walk (OGG looks for the granule position of the
/// last page, and the MP3 estimate uses the stream size), so the file is read in one piece after
/// a size check.
///
/// # Errors
///
/// Returns a failure payload when the file cannot be inspected or read, or when it exceeds
/// [`MAX_MEDIA_BYTES`].
async fn read_media_bytes(path: &Path) -> Result<Vec<u8>, ToolPayload> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|error| read_failure(path, &error))?;
    if metadata.len() > MAX_MEDIA_BYTES {
        return Err(ToolPayload::fail(format!(
            "failed to read {}: file is {} bytes, larger than the {MAX_MEDIA_BYTES} byte media limit",
            path.display(),
            metadata.len()
        )));
    }
    tokio::fs::read(path)
        .await
        .map_err(|error| read_failure(path, &error))
}

/// Forces `field` into a generated schema's `required` list.
///
/// The argument structs default their path field so that a missing key produces the Python
/// `image_path is required` / `audio_path is required` message instead of a serde error; schemars
/// then treats the field as optional, and each card restores it here.
fn require_field(schema: &mut Value, field: &str) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    match object.get_mut("required") {
        Some(Value::Array(required)) => {
            if !required.iter().any(|value| value.as_str() == Some(field)) {
                required.push(Value::String(field.to_owned()));
            }
        }
        Some(_) | None => {
            object.insert("required".to_owned(), json!([field]));
        }
    }
}

/// Builds the payload message for a failed media read.
fn read_failure(path: &Path, error: &std::io::Error) -> ToolPayload {
    ToolPayload::fail(format!("failed to read {}: {error}", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reading_a_directory_reports_a_typed_failure() {
        let dir = tempfile::tempdir().unwrap();
        let payload = read_media_bytes(dir.path()).await.unwrap_err();
        assert!(!payload.success);
        assert!(
            payload
                .error
                .as_deref()
                .is_some_and(|message| message.starts_with("failed to read ")),
            "unexpected message: {:?}",
            payload.error
        );
    }

    #[tokio::test]
    async fn reading_a_missing_file_reports_a_typed_failure() {
        let dir = tempfile::tempdir().unwrap();
        let payload = read_media_bytes(&dir.path().join("missing.png"))
            .await
            .unwrap_err();
        assert!(!payload.success);
        assert!(
            payload
                .error
                .as_deref()
                .is_some_and(|message| message.starts_with("failed to read ")),
            "unexpected message: {:?}",
            payload.error
        );
    }

    #[tokio::test]
    async fn reads_a_media_file_into_memory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("media.bin");
        std::fs::write(&path, fixtures::png()).unwrap();
        assert_eq!(read_media_bytes(&path).await.unwrap(), fixtures::png());
    }
}
