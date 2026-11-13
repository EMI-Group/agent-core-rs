//! `image_metadata`: header-only image inspection for the model.
//!
//! Ports the metadata half of `openjiuwen/harness/tools/multimodal/vision.py` as a model-free,
//! offline subset: Python delegates OCR and visual Q&A to a vision model, while this tool only
//! reads the container header and reports what it declares. No decoding crate is involved and
//! nothing leaves the process.
//!
//! Formats are detected by magic bytes, never by file extension: PNG, JPEG, GIF, WebP and BMP.
//! Anything else, a truncated header or an empty file yields a `success: false` payload whose
//! message names the problem.

// The format modules repeat their format name (`png::PngInfo`), which is the clearest spelling
// for both the module path and the flat re-export, so the repetition lint does not apply here.
#![allow(clippy::module_name_repetitions)]

pub mod bmp;
pub mod gif;
pub mod jpeg;
pub mod png;
pub mod webp;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::super::descriptions::description;
use super::super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};
use super::parse::{fourcc, Failure, MediaKind, ParseResult};
use super::read_media_bytes;

/// The model-facing name of the tool.
pub const TOOL_NAME: &str = "image_metadata";

/// Parsed image header, one variant per supported container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageMetadata {
    /// PNG: `IHDR`, plus the `tRNS`/`acTL` hints.
    Png(png::PngInfo),
    /// JPEG: the `SOF` marker, plus the EXIF orientation.
    Jpeg(jpeg::JpegInfo),
    /// GIF: the logical screen descriptor, plus the frame walk.
    Gif(gif::GifInfo),
    /// WebP: the RIFF `VP8X`/`VP8 `/`VP8L` headers.
    Webp(webp::WebpInfo),
    /// BMP: the file header and the DIB header.
    Bmp(bmp::BmpInfo),
}

impl ImageMetadata {
    /// The container name as reported in the payload: `png`, `jpeg`, `gif`, `webp` or `bmp`.
    #[must_use]
    pub fn format(&self) -> &'static str {
        match self {
            Self::Png(_) => "png",
            Self::Jpeg(_) => "jpeg",
            Self::Gif(_) => "gif",
            Self::Webp(_) => "webp",
            Self::Bmp(_) => "bmp",
        }
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        match self {
            Self::Png(info) => info.width,
            Self::Jpeg(info) => info.width,
            Self::Gif(info) => u32::from(info.width),
            Self::Webp(info) => info.width,
            Self::Bmp(info) => info.width,
        }
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        match self {
            Self::Png(info) => info.height,
            Self::Jpeg(info) => info.height,
            Self::Gif(info) => u32::from(info.height),
            Self::Webp(info) => info.height,
            Self::Bmp(info) => info.height,
        }
    }

    /// Renders the payload `data` object for a file of `byte_size` bytes.
    ///
    /// Every format reports `kind`, `format`, `byte_size`, `width` and `height`; the remaining
    /// keys are the container's own fields.
    #[must_use]
    pub fn to_data(&self, byte_size: u64) -> Value {
        match self {
            Self::Png(info) => json!({
                "kind": MediaKind::Image.as_str(),
                "format": "png",
                "byte_size": byte_size,
                "width": info.width,
                "height": info.height,
                "bit_depth": info.bit_depth,
                "color_type": info.color_type,
                "color_type_name": info.color_type_name,
                "has_alpha": info.has_alpha,
                "interlaced": info.interlaced,
                "frames": info.frames,
            }),
            Self::Jpeg(info) => json!({
                "kind": MediaKind::Image.as_str(),
                "format": "jpeg",
                "byte_size": byte_size,
                "width": info.width,
                "height": info.height,
                "components": info.components,
                "precision": info.precision,
                "progressive": info.progressive,
                "orientation": info.orientation,
            }),
            Self::Gif(info) => json!({
                "kind": MediaKind::Image.as_str(),
                "format": "gif",
                "byte_size": byte_size,
                "width": info.width,
                "height": info.height,
                "version": info.version,
                "frames": info.frames,
                "animated": info.animated,
                "has_alpha": info.has_alpha,
                "global_color_table": info.global_color_table,
                "color_resolution": info.color_resolution,
            }),
            Self::Webp(info) => json!({
                "kind": MediaKind::Image.as_str(),
                "format": "webp",
                "byte_size": byte_size,
                "width": info.width,
                "height": info.height,
                "variant": info.variant,
                "has_alpha": info.has_alpha,
                "animated": info.animated,
                "frames": info.frames,
            }),
            Self::Bmp(info) => json!({
                "kind": MediaKind::Image.as_str(),
                "format": "bmp",
                "byte_size": byte_size,
                "width": info.width,
                "height": info.height,
                "top_down": info.top_down,
                "bit_count": info.bit_count,
                "compression": info.compression,
                "compression_name": info.compression_name,
                "dib_header_size": info.dib_header_size,
                "data_offset": info.data_offset,
            }),
        }
    }
}

/// Parses an image header, detecting the format from its magic bytes.
///
/// # Errors
///
/// Returns a [`Failure`] when the bytes are empty, when the magic bytes match no supported
/// format (`unsupported image format: <4cc>`), or when the matched format's header is truncated
/// or malformed.
pub fn parse_image_header(bytes: &[u8]) -> ParseResult<ImageMetadata> {
    if bytes.is_empty() {
        return Err(Failure::new("empty file: no image header bytes"));
    }
    if bytes.starts_with(png::SIGNATURE.as_slice()) {
        png::parse(bytes).map(ImageMetadata::Png)
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        jpeg::parse(bytes).map(ImageMetadata::Jpeg)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        gif::parse(bytes).map(ImageMetadata::Gif)
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
        webp::parse(bytes).map(ImageMetadata::Webp)
    } else if bytes.starts_with(b"BM") {
        bmp::parse(bytes).map(ImageMetadata::Bmp)
    } else {
        Err(Failure::new(format!(
            "unsupported image format: {}",
            fourcc(bytes)
        )))
    }
}

/// Arguments of the `image_metadata` tool.
///
/// Python parity: `ImageOCRTool` takes `image_path_or_url`. Only local paths are supported here,
/// and the Python spellings are accepted as aliases; the card advertises `image_path` only.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImageMetadataArgs {
    /// Local path of the image file; relative paths resolve inside the workspace.
    #[serde(
        default,
        alias = "path",
        alias = "file_path",
        alias = "image_path_or_url"
    )]
    #[schemars(required)]
    pub image_path: String,
}

/// `image_metadata`: reports the dimensions and container fields of an image file.
pub struct ImageMetadataTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The model-facing card, built from [`ToolEnv::language`].
    card: ToolCard,
}

impl ImageMetadataTool {
    /// Builds the tool over `env`, using `language` for the card description.
    #[must_use]
    pub fn new(env: ToolEnv, language: impl Into<String>) -> Self {
        let env = env.with_language(language);
        let card = Self::card_for(env.language());
        Self { env, card }
    }

    /// Builds the model-facing card for the description language `language`.
    ///
    /// The description text is owned by the shared [`description`] table, exactly like the other
    /// built-in tools, so this module never inlines prompt text.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<ImageMetadataArgs>();
        super::require_field(&mut input_params, "image_path");
        tool_card(
            TOOL_NAME,
            description(TOOL_NAME, language),
            input_params,
            true,
        )
    }

    /// Resolves the path, reads the file and renders the payload.
    async fn inspect(&self, inputs: &Value) -> ToolPayload {
        let args: ImageMetadataArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.image_path.is_empty() {
            return ToolPayload::fail("image_path is required");
        }
        let resolved = match self
            .env
            .resolve_path(TOOL_NAME, "image_path", &args.image_path, inputs)
            .await
        {
            Ok(path) => path,
            Err(payload) => return payload,
        };
        let bytes = match read_media_bytes(&resolved).await {
            Ok(bytes) => bytes,
            Err(payload) => return payload,
        };
        let byte_size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        match parse_image_header(&bytes) {
            Ok(metadata) => ToolPayload::ok(metadata.to_data(byte_size)),
            Err(failure) => ToolPayload::fail(failure.message()),
        }
    }
}

#[async_trait]
impl Tool for ImageMetadataTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.inspect(&inputs).await.to_json())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::tools::media::fixtures;
    use crate::workspace::Workspace;

    /// The tool over a fresh temporary workspace.
    fn setup() -> (ImageMetadataTool, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let tool = ImageMetadataTool::new(ToolEnv::new(Workspace::new(dir.path())), "en");
        (tool, dir)
    }

    /// Invokes the tool with a fresh context and unwraps the payload envelope.
    async fn invoke(tool: &ImageMetadataTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    /// Writes `bytes` to `name` inside `dir`.
    fn write(dir: &TempDir, name: &str, bytes: &[u8]) -> String {
        std::fs::write(dir.path().join(name), bytes).unwrap();
        name.to_owned()
    }

    #[test]
    fn card_exposes_only_the_canonical_path_argument() {
        let card = ImageMetadataTool::card_for("en");
        assert_eq!(card.name, "image_metadata");
        assert!(card.parallel_safe);
        assert_eq!(card.description, description("image_metadata", "en"));
        assert_eq!(
            card.input_params["properties"]["image_path"]["type"],
            "string"
        );
        assert_eq!(card.input_params["required"], json!(["image_path"]));
        assert_eq!(
            card.input_params["properties"].as_object().unwrap().len(),
            1
        );
    }

    #[test]
    fn dispatch_detects_every_format_by_magic_bytes() {
        let png = parse_image_header(&fixtures::png()).unwrap();
        assert_eq!(png.format(), "png");
        assert_eq!((png.width(), png.height()), (2, 2));

        let jpeg = parse_image_header(&fixtures::jpeg()).unwrap();
        assert_eq!(jpeg.format(), "jpeg");
        assert_eq!((jpeg.width(), jpeg.height()), (2, 2));

        let gif = parse_image_header(&fixtures::gif()).unwrap();
        assert_eq!(gif.format(), "gif");
        assert_eq!((gif.width(), gif.height()), (1, 1));

        let webp = parse_image_header(&fixtures::webp()).unwrap();
        assert_eq!(webp.format(), "webp");
        assert_eq!((webp.width(), webp.height()), (100, 50));

        let bmp = parse_image_header(&fixtures::bmp()).unwrap();
        assert_eq!(bmp.format(), "bmp");
        assert_eq!((bmp.width(), bmp.height()), (4, 3));
    }

    #[test]
    fn payload_reports_kind_format_and_byte_size() {
        let metadata = parse_image_header(&fixtures::png()).unwrap();
        let data = metadata.to_data(42);
        assert_eq!(data["kind"], json!("image"));
        assert_eq!(data["format"], json!("png"));
        assert_eq!(data["byte_size"], json!(42));
        assert_eq!(data["width"], json!(2));
        assert_eq!(data["height"], json!(2));
        assert_eq!(data["color_type_name"], json!("truecolor_alpha"));
        assert_eq!(data["has_alpha"], json!(true));
    }

    #[test]
    fn unknown_magic_and_empty_input_are_reported() {
        assert_eq!(
            parse_image_header(&[]).unwrap_err().message(),
            "empty file: no image header bytes"
        );
        assert_eq!(
            parse_image_header(b"<!DOCTYPE html>")
                .unwrap_err()
                .message(),
            "unsupported image format: <!DO"
        );
        assert_eq!(
            parse_image_header(b"RIFF\0\0\0\0WAVEfmt ")
                .unwrap_err()
                .message(),
            "unsupported image format: RIFF"
        );
    }

    #[tokio::test]
    async fn reads_an_image_end_to_end_ignoring_the_extension() {
        let (tool, dir) = setup();
        let path = write(&dir, "photo.txt", &fixtures::png());
        let out = invoke(&tool, json!({"image_path": path})).await;
        assert_eq!(out["success"], json!(true));
        assert_eq!(out["data"]["format"], json!("png"));
        assert_eq!(out["data"]["width"], json!(2));
        assert_eq!(out["data"]["byte_size"], json!(fixtures::png().len()));
        assert_eq!(out["error"], Value::Null);
    }

    #[tokio::test]
    async fn accepts_python_style_path_aliases() {
        let (tool, dir) = setup();
        let path = write(&dir, "photo.png", &fixtures::gif());
        for key in ["image_path", "path", "file_path", "image_path_or_url"] {
            let out = invoke(&tool, json!({ key: path })).await;
            assert_eq!(out["success"], json!(true), "alias {key} rejected");
            assert_eq!(out["data"]["format"], json!("gif"));
        }
    }

    #[tokio::test]
    async fn missing_path_is_reported() {
        let (tool, dir) = setup();
        let _ = &dir;
        let out = invoke(&tool, json!({})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("image_path is required"));

        let blank = invoke(&tool, json!({"image_path": ""})).await;
        assert_eq!(blank["error"], json!("image_path is required"));

        let not_an_object = invoke(&tool, json!(["image_path"])).await;
        assert_eq!(
            not_an_object["error"],
            json!("tool arguments must be an object")
        );
    }

    #[tokio::test]
    async fn a_path_outside_the_workspace_is_refused() {
        let (tool, dir) = setup();
        let outside = dir.path().parent().unwrap().join("escape.png");
        let parent_relative = invoke(&tool, json!({"image_path": "../escape.png"})).await;
        assert_eq!(parent_relative["success"], json!(false));
        assert!(
            parent_relative["error"]
                .as_str()
                .unwrap()
                .starts_with("Invalid path '../escape.png':"),
            "unexpected message: {}",
            parent_relative["error"]
        );

        let absolute = invoke(&tool, json!({"image_path": outside.to_string_lossy()})).await;
        assert_eq!(absolute["success"], json!(false));
        assert!(
            absolute["error"]
                .as_str()
                .unwrap()
                .starts_with("Access denied:"),
            "unexpected message: {}",
            absolute["error"]
        );
    }

    #[tokio::test]
    async fn malformed_and_empty_files_are_reported() {
        let (tool, dir) = setup();
        let empty = write(&dir, "empty.png", &[]);
        let out = invoke(&tool, json!({"image_path": empty})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("empty file: no image header bytes"));

        let truncated = write(&dir, "truncated.png", &fixtures::png()[..20]);
        let out = invoke(&tool, json!({"image_path": truncated})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("truncated PNG header: IHDR is incomplete")
        );

        let missing = invoke(&tool, json!({"image_path": "nope.png"})).await;
        assert_eq!(missing["success"], json!(false));
        assert!(
            missing["error"].as_str().unwrap().starts_with("failed to "),
            "unexpected message: {}",
            missing["error"]
        );
    }
}
