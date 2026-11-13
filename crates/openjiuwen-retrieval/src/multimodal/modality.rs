//! The [`Modality`] enum and the MIME/extension helpers shared by the module.
//!
//! Port of the Python `Literal["text", "image", "audio", "video"]` used by
//! `MultimodalDocument.add_field` in
//! `openjiuwen/core/retrieval/common/document.py`, plus the extension-based MIME
//! lookup Python performs with the stdlib `mimetypes` module.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

/// The content types a [`super::MultimodalDocument`] can carry.
///
/// Serializes as the Python literal values `"text"`, `"image"`, `"audio"`, and
/// `"video"`; the default is [`Modality::Text`] so a bare content part is a text
/// part.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Plain text content.
    #[default]
    Text,
    /// Image content, addressed by a `data:image/...;base64,` URI or a URL.
    Image,
    /// Audio content, addressed by a `data:audio/...;base64,` URI.
    Audio,
    /// Video content, addressed by a `data:video/...;base64,` URI or a URL.
    Video,
}

impl Modality {
    /// Every modality, in the order Python lists them.
    pub const ALL: [Self; 4] = [Self::Text, Self::Image, Self::Audio, Self::Video];

    /// The lowercase wire name (`"text"`, `"image"`, `"audio"`, `"video"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }

    /// The MIME type prefix of the modality, including the slash (`"image/"`).
    ///
    /// Used to validate and repair data-URI media types, matching Python's
    /// `"/".join([kind] + mime_type.split("/")[1:])` rewrite.
    #[must_use]
    pub const fn mime_prefix(self) -> &'static str {
        match self {
            Self::Text => "text/",
            Self::Image => "image/",
            Self::Audio => "audio/",
            Self::Video => "video/",
        }
    }

    /// Maps a file extension to its modality, ignoring a leading dot and case.
    ///
    /// Unknown extensions return `None`; Python instead fails with a pydantic
    /// validation error when it cannot determine the MIME type.
    #[must_use]
    pub fn from_extension(extension: &str) -> Option<Self> {
        let extension = normalize_extension(extension);
        let mime = mime_for_extension(&extension)?;
        if mime.starts_with(Self::Image.mime_prefix()) {
            Some(Self::Image)
        } else if mime.starts_with(Self::Audio.mime_prefix()) {
            Some(Self::Audio)
        } else if mime.starts_with(Self::Video.mime_prefix()) {
            Some(Self::Video)
        } else {
            Some(Self::Text)
        }
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Modality {
    type Err = Error;

    /// Parses the lowercase wire name; anything else is a validation error.
    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "video" => Ok(Self::Video),
            _ => Err(Error::Validation(format!(
                "Unknown kind of multimodal file: {value}, supported option: [\"text\", \"image\", \"audio\", \"video\"]"
            ))),
        }
    }
}

/// Guesses the MIME type of `path` from its extension.
///
/// Replaces Python's `mimetypes.guess_type`; `.jfif` maps to `image/jpeg` like
/// the reference implementation registers it explicitly. Unknown extensions
/// return `None`.
#[must_use]
pub fn guess_mime_type(path: &str) -> Option<&'static str> {
    mime_for_extension(&normalize_extension(path))
}

/// Returns the lowercase extension of `path` without the leading dot.
fn normalize_extension(extension: &str) -> String {
    let name = extension.rsplit(['/', '\\']).next().unwrap_or(extension);
    let suffix = match name.rfind('.') {
        Some(index) if index > 0 => &name[index + 1..],
        _ => name,
    };
    suffix.trim_start_matches('.').to_ascii_lowercase()
}

/// The extension-to-MIME table backing [`guess_mime_type`].
fn mime_for_extension(extension: &str) -> Option<&'static str> {
    Some(match extension {
        "png" => "image/png",
        "jpg" | "jpeg" | "jfif" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "svg" => "image/svg+xml",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",
        "aac" => "audio/aac",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "avi" => "video/x-msvideo",
        "mkv" => "video/x-matroska",
        "webm" => "video/webm",
        "txt" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "csv" => "text/csv",
        "html" | "htm" => "text/html",
        _ => return None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_match_the_python_literals() {
        assert_eq!(Modality::Text.to_string(), "text");
        assert_eq!(Modality::Image.to_string(), "image");
        assert_eq!(Modality::Audio.to_string(), "audio");
        assert_eq!(Modality::Video.to_string(), "video");
        assert_eq!(
            serde_json::to_string(&Modality::Image).unwrap(),
            "\"image\""
        );
        let restored: Modality = serde_json::from_str("\"video\"").unwrap();
        assert_eq!(restored, Modality::Video);
        assert_eq!(Modality::default(), Modality::Text);
    }

    #[test]
    fn from_str_accepts_names_case_insensitively() {
        assert_eq!("image".parse::<Modality>().unwrap(), Modality::Image);
        assert_eq!(" AUDIO ".parse::<Modality>().unwrap(), Modality::Audio);
        let error = "picture".parse::<Modality>().unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error
            .to_string()
            .contains("Unknown kind of multimodal file"));
    }

    #[test]
    fn mime_prefix_matches_the_modality() {
        assert_eq!(Modality::Text.mime_prefix(), "text/");
        assert_eq!(Modality::Image.mime_prefix(), "image/");
        assert_eq!(Modality::Audio.mime_prefix(), "audio/");
        assert_eq!(Modality::Video.mime_prefix(), "video/");
        for modality in Modality::ALL {
            assert!(guess_mime_type(&format!("a.{}", extension_for(modality)))
                .unwrap()
                .starts_with(modality.mime_prefix()));
        }
    }

    #[test]
    fn extension_lookup_is_case_insensitive_and_dot_agnostic() {
        assert_eq!(Modality::from_extension(".PNG"), Some(Modality::Image));
        assert_eq!(Modality::from_extension("jfif"), Some(Modality::Image));
        assert_eq!(
            Modality::from_extension("/tmp/talk.MP3"),
            Some(Modality::Audio)
        );
        assert_eq!(Modality::from_extension("clip.webm"), Some(Modality::Video));
        assert_eq!(Modality::from_extension("notes.md"), Some(Modality::Text));
        assert_eq!(Modality::from_extension("sheet.xlsx"), None);
        assert_eq!(Modality::from_extension(""), None);
    }

    #[test]
    fn mime_guessing_covers_the_reference_extension_set() {
        assert_eq!(guess_mime_type("a.jfif"), Some("image/jpeg"));
        assert_eq!(guess_mime_type("/tmp/a.PNG"), Some("image/png"));
        assert_eq!(guess_mime_type("a.wav"), Some("audio/wav"));
        assert_eq!(guess_mime_type("a.mp4"), Some("video/mp4"));
        assert_eq!(guess_mime_type("a.pdf"), None);
        assert_eq!(guess_mime_type(".env"), None);
    }

    /// Returns a representative extension for a modality.
    fn extension_for(modality: Modality) -> &'static str {
        match modality {
            Modality::Text => "txt",
            Modality::Image => "png",
            Modality::Audio => "wav",
            Modality::Video => "mp4",
        }
    }
}
