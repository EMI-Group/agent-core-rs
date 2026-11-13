//! `audio_metadata`: header-only audio inspection for the model.
//!
//! Ports the metadata half of `openjiuwen/harness/tools/multimodal/audio.py`
//! (`AudioMetadataTool`) as a model-free, offline subset: Python asks an audio model to
//! transcribe, answer questions, or identify a track through an ACR service, while this tool
//! reports duration and stream parameters from the container header. Nothing is transcribed,
//! no network call is made, and no URL input is accepted.
//!
//! Formats are detected by magic bytes (an `ID3v2` tag in front of the stream is skipped, never
//! interpreted), never by file extension: WAV, FLAC, MP3 and OGG.

// The format modules repeat their format name (`wav::WavInfo`), which is the clearest spelling
// for both the module path and the flat re-export, so the repetition lint does not apply here.
#![allow(clippy::module_name_repetitions)]

pub mod flac;
pub mod mp3;
pub mod ogg;
pub mod wav;

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
pub const TOOL_NAME: &str = "audio_metadata";

/// Parsed audio header, one variant per supported container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioMetadata {
    /// WAV: the RIFF `fmt ` and `data` chunks.
    Wav(wav::WavInfo),
    /// FLAC: the `STREAMINFO` metadata block.
    Flac(flac::FlacInfo),
    /// MP3: the first MPEG frame header, plus the `Xing`/`Info` frame count.
    Mp3(mp3::Mp3Info),
    /// OGG: the Vorbis identification header or the Opus head.
    Ogg(ogg::OggInfo),
}

impl AudioMetadata {
    /// The container name as reported in the payload: `wav`, `flac`, `mp3` or `ogg`.
    #[must_use]
    pub fn format(&self) -> &'static str {
        match self {
            Self::Wav(_) => "wav",
            Self::Flac(_) => "flac",
            Self::Mp3(_) => "mp3",
            Self::Ogg(_) => "ogg",
        }
    }

    /// Sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Wav(info) => info.sample_rate,
            Self::Flac(info) => info.sample_rate,
            Self::Mp3(info) => info.sample_rate,
            Self::Ogg(info) => info.sample_rate,
        }
    }

    /// Channel count.
    #[must_use]
    pub fn channels(&self) -> u16 {
        match self {
            Self::Wav(info) => info.channels,
            Self::Flac(info) => u16::from(info.channels),
            Self::Mp3(info) => u16::from(info.channels),
            Self::Ogg(info) => u16::from(info.channels),
        }
    }

    /// Duration in milliseconds, when the header declares enough information to compute it.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        match self {
            Self::Wav(info) => info.duration_ms,
            Self::Flac(info) => info.duration_ms,
            Self::Mp3(info) => info.duration_ms,
            Self::Ogg(info) => info.duration_ms,
        }
    }

    /// Renders the payload `data` object for a file of `byte_size` bytes.
    ///
    /// Every format reports `kind`, `format`, `byte_size`, `sample_rate`, `channels` and
    /// `duration_ms` (plus `duration_seconds`, the Python `AudioMetadataTool` spelling); the
    /// remaining keys are the container's own fields.
    #[must_use]
    pub fn to_data(&self, byte_size: u64) -> Value {
        match self {
            Self::Wav(info) => json!({
                "kind": MediaKind::Audio.as_str(),
                "format": "wav",
                "byte_size": byte_size,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "duration_ms": info.duration_ms,
                "duration_seconds": info.duration_ms.map(duration_seconds),
                "bits_per_sample": info.bits_per_sample,
                "format_tag": info.format_tag,
                "format_name": info.format_name,
                "byte_rate": info.byte_rate,
                "block_align": info.block_align,
                "data_bytes": info.data_bytes,
            }),
            Self::Flac(info) => json!({
                "kind": MediaKind::Audio.as_str(),
                "format": "flac",
                "byte_size": byte_size,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "duration_ms": info.duration_ms,
                "duration_seconds": info.duration_ms.map(duration_seconds),
                "bits_per_sample": info.bits_per_sample,
                "total_samples": info.total_samples,
                "min_block_size": info.min_block_size,
                "max_block_size": info.max_block_size,
                "id3v2_bytes": info.id3v2_bytes,
            }),
            Self::Mp3(info) => json!({
                "kind": MediaKind::Audio.as_str(),
                "format": "mp3",
                "byte_size": byte_size,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "duration_ms": info.duration_ms,
                "duration_seconds": info.duration_ms.map(duration_seconds),
                "mpeg_version": info.version,
                "layer": info.layer,
                "bitrate_kbps": info.bitrate_kbps,
                "xing_frames": info.xing_frames,
                "id3v2_bytes": info.id3v2_bytes,
            }),
            Self::Ogg(info) => json!({
                "kind": MediaKind::Audio.as_str(),
                "format": "ogg",
                "byte_size": byte_size,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "duration_ms": info.duration_ms,
                "duration_seconds": info.duration_ms.map(duration_seconds),
                "codec": info.codec,
                "granule": info.granule,
                "nominal_bitrate": info.nominal_bitrate,
                "pre_skip": info.pre_skip,
                "input_sample_rate": info.input_sample_rate,
            }),
        }
    }
}

/// Converts milliseconds to the Python `duration_seconds` spelling, rounded to two decimals.
#[must_use]
pub fn duration_seconds(duration_ms: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)] // durations are far below 2^53 milliseconds
    let rounded = (duration_ms as f64 / 10.0).round();
    rounded / 100.0
}

/// Parses an audio header, detecting the format from its magic bytes.
///
/// # Errors
///
/// Returns a [`Failure`] when the bytes are empty, when the magic bytes match no supported
/// format (`unsupported audio format: <4cc>`), or when the matched format's header is truncated
/// or malformed.
pub fn parse_audio_header(bytes: &[u8]) -> ParseResult<AudioMetadata> {
    if bytes.is_empty() {
        return Err(Failure::new("empty file: no audio header bytes"));
    }
    if bytes.starts_with(b"RIFF") {
        // WAV is the only RIFF container this tool understands, so a short prefix is reported as a
        // truncated WAV container rather than as an unknown format.
        if bytes.len() < 12 || bytes.get(8..12) == Some(b"WAVE".as_slice()) {
            return wav::parse(bytes).map(AudioMetadata::Wav);
        }
        return Err(Failure::new(format!(
            "unsupported audio format: {}",
            fourcc(bytes)
        )));
    }
    if bytes.starts_with(b"OggS") {
        return ogg::parse(bytes).map(AudioMetadata::Ogg);
    }
    // An ID3v2 tag may precede a FLAC stream as well as an MPEG stream.
    let body = strip_id3v2(bytes);
    if body.starts_with(b"fLaC") {
        return flac::parse(bytes).map(AudioMetadata::Flac);
    }
    if mp3::looks_like_frame(body) {
        return mp3::parse(bytes).map(AudioMetadata::Mp3);
    }
    Err(Failure::new(format!(
        "unsupported audio format: {}",
        fourcc(body)
    )))
}

/// Skips a leading `ID3v2` tag, returning the stream bytes and the tag length in bytes.
///
/// The tag header is ten bytes (`ID3`, version, flags, synchsafe size); a footer adds ten more
/// bytes when the footer flag is set. A malformed or truncated tag is left in place, so the
/// caller's magic-byte check reports the real problem.
pub(crate) fn strip_id3v2(bytes: &[u8]) -> &[u8] {
    let Some(header) = bytes.get(..10) else {
        return bytes;
    };
    if !header.starts_with(b"ID3") {
        return bytes;
    }
    let size = match header.get(6..10) {
        Some(raw) => synchsafe_size(raw),
        None => return bytes,
    };
    let has_footer = header.get(5).is_some_and(|flags| flags & 0x10 != 0);
    let total = 10usize
        .saturating_add(size)
        .saturating_add(if has_footer { 10 } else { 0 });
    bytes.get(total..).unwrap_or(&[])
}

/// Decodes the four-byte synchsafe integer used by `ID3v2` tag sizes.
fn synchsafe_size(raw: &[u8]) -> usize {
    raw.iter()
        .fold(0usize, |size, byte| (size << 7) | usize::from(*byte & 0x7F))
}

/// Arguments of the `audio_metadata` tool.
///
/// Python parity: `AudioMetadataTool` takes `audio_path_or_url`. Only local paths are supported
/// here, and the Python spelling is accepted as an alias; the card advertises `audio_path` only.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct AudioMetadataArgs {
    /// Local path of the audio file; relative paths resolve inside the workspace.
    #[serde(
        default,
        alias = "path",
        alias = "file_path",
        alias = "audio_path_or_url"
    )]
    #[schemars(required)]
    pub audio_path: String,
}

/// `audio_metadata`: reports the duration and stream parameters of an audio file.
pub struct AudioMetadataTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The model-facing card, built from [`ToolEnv::language`].
    card: ToolCard,
}

impl AudioMetadataTool {
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
        let mut input_params = schema_for::<AudioMetadataArgs>();
        super::require_field(&mut input_params, "audio_path");
        tool_card(
            TOOL_NAME,
            description(TOOL_NAME, language),
            input_params,
            true,
        )
    }

    /// Resolves the path, reads the file and renders the payload.
    async fn inspect(&self, inputs: &Value) -> ToolPayload {
        let args: AudioMetadataArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.audio_path.is_empty() {
            return ToolPayload::fail("audio_path is required");
        }
        let resolved = match self
            .env
            .resolve_path(TOOL_NAME, "audio_path", &args.audio_path, inputs)
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
        match parse_audio_header(&bytes) {
            Ok(metadata) => ToolPayload::ok(metadata.to_data(byte_size)),
            Err(failure) => ToolPayload::fail(failure.message()),
        }
    }
}

#[async_trait]
impl Tool for AudioMetadataTool {
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
    fn setup() -> (AudioMetadataTool, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let tool = AudioMetadataTool::new(ToolEnv::new(Workspace::new(dir.path())), "en");
        (tool, dir)
    }

    /// Invokes the tool with a fresh context and unwraps the payload envelope.
    async fn invoke(tool: &AudioMetadataTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    /// Writes `bytes` to `name` inside `dir` and returns the relative path.
    fn write(dir: &TempDir, name: &str, bytes: &[u8]) -> String {
        std::fs::write(dir.path().join(name), bytes).unwrap();
        name.to_owned()
    }

    #[test]
    fn card_exposes_only_the_canonical_path_argument() {
        let card = AudioMetadataTool::card_for("en");
        assert_eq!(card.name, "audio_metadata");
        assert!(card.parallel_safe);
        assert_eq!(card.description, description("audio_metadata", "en"));
        assert_eq!(
            card.input_params["properties"]["audio_path"]["type"],
            "string"
        );
        assert_eq!(card.input_params["required"], json!(["audio_path"]));
        assert_eq!(
            card.input_params["properties"].as_object().unwrap().len(),
            1
        );
    }

    #[test]
    fn dispatch_detects_every_format_by_magic_bytes() {
        let wav = parse_audio_header(&fixtures::wav()).unwrap();
        assert_eq!(wav.format(), "wav");
        assert_eq!(wav.sample_rate(), 44_100);
        assert_eq!(wav.channels(), 2);
        assert_eq!(wav.duration_ms(), Some(1000));

        let flac = parse_audio_header(&fixtures::flac()).unwrap();
        assert_eq!(flac.format(), "flac");
        assert_eq!(flac.duration_ms(), Some(1000));

        let mp3 = parse_audio_header(&fixtures::mp3()).unwrap();
        assert_eq!(mp3.format(), "mp3");
        assert_eq!(mp3.sample_rate(), 44_100);
        assert_eq!(mp3.duration_ms(), Some(2612));

        let ogg = parse_audio_header(&fixtures::ogg()).unwrap();
        assert_eq!(ogg.format(), "ogg");
        assert_eq!(ogg.sample_rate(), 48_000);
        assert_eq!(ogg.duration_ms(), Some(1000));
    }

    #[test]
    fn duration_seconds_matches_the_python_rounding() {
        assert!((duration_seconds(1000) - 1.0).abs() < f64::EPSILON);
        assert!((duration_seconds(2612) - 2.61).abs() < f64::EPSILON);
        assert!((duration_seconds(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn payload_reports_kind_format_and_byte_size() {
        let metadata = parse_audio_header(&fixtures::wav()).unwrap();
        let data = metadata.to_data(9);
        assert_eq!(data["kind"], json!("audio"));
        assert_eq!(data["format"], json!("wav"));
        assert_eq!(data["byte_size"], json!(9));
        assert_eq!(data["duration_ms"], json!(1000));
        assert_eq!(data["duration_seconds"], json!(1.0));
        assert_eq!(data["format_name"], json!("pcm"));
    }

    #[test]
    fn unknown_magic_and_empty_input_are_reported() {
        assert_eq!(
            parse_audio_header(&[]).unwrap_err().message(),
            "empty file: no audio header bytes"
        );
        assert_eq!(
            parse_audio_header(b"XXXXnot audio").unwrap_err().message(),
            "unsupported audio format: XXXX"
        );
        assert_eq!(
            parse_audio_header(b"RIFF\0\0\0\0AVI LIST")
                .unwrap_err()
                .message(),
            "unsupported audio format: RIFF"
        );
    }

    #[tokio::test]
    async fn reads_audio_end_to_end_ignoring_the_extension() {
        let (tool, dir) = setup();
        let path = write(&dir, "recording.bin", &fixtures::flac());
        let out = invoke(&tool, json!({"audio_path": path})).await;
        assert_eq!(out["success"], json!(true));
        assert_eq!(out["data"]["format"], json!("flac"));
        assert_eq!(out["data"]["duration_ms"], json!(1000));
        assert_eq!(out["data"]["byte_size"], json!(fixtures::flac().len()));
    }

    #[tokio::test]
    async fn accepts_python_style_path_aliases() {
        let (tool, dir) = setup();
        let path = write(&dir, "clip.wav", &fixtures::wav());
        for key in ["audio_path", "path", "file_path", "audio_path_or_url"] {
            let out = invoke(&tool, json!({ key: path })).await;
            assert_eq!(out["success"], json!(true), "alias {key} rejected");
            assert_eq!(out["data"]["format"], json!("wav"));
        }
    }

    #[tokio::test]
    async fn missing_path_is_reported() {
        let (tool, dir) = setup();
        let _ = &dir;
        let out = invoke(&tool, json!({})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("audio_path is required"));

        let blank = invoke(&tool, json!({"audio_path": ""})).await;
        assert_eq!(blank["error"], json!("audio_path is required"));
    }

    #[tokio::test]
    async fn a_path_outside_the_workspace_is_refused() {
        let (tool, dir) = setup();
        let outside = dir.path().parent().unwrap().join("escape.mp3");
        let relative = invoke(&tool, json!({"audio_path": "../escape.mp3"})).await;
        assert_eq!(relative["success"], json!(false));
        assert!(
            relative["error"]
                .as_str()
                .unwrap()
                .starts_with("Invalid path '../escape.mp3':"),
            "unexpected message: {}",
            relative["error"]
        );

        let absolute = invoke(&tool, json!({"audio_path": outside.to_string_lossy()})).await;
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
        let empty = write(&dir, "empty.wav", &[]);
        let out = invoke(&tool, json!({"audio_path": empty})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(out["error"], json!("empty file: no audio header bytes"));

        let truncated = write(&dir, "truncated.wav", &fixtures::wav()[..10]);
        let out = invoke(&tool, json!({"audio_path": truncated})).await;
        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("truncated WAV header: RIFF container is incomplete")
        );

        let missing = invoke(&tool, json!({"audio_path": "nope.wav"})).await;
        assert_eq!(missing["success"], json!(false));
        assert!(
            missing["error"].as_str().unwrap().starts_with("failed to "),
            "unexpected message: {}",
            missing["error"]
        );
    }

    #[test]
    fn id3v2_tags_are_skipped_before_detection() {
        let mut bytes = b"ID3\x04\x00\x00\x00\x00\x00\x0A".to_vec();
        bytes.extend_from_slice(&[0; 10]);
        bytes.extend_from_slice(&fixtures::mp3());
        let metadata = parse_audio_header(&bytes).unwrap();
        assert_eq!(metadata.format(), "mp3");
        let data = metadata.to_data(u64::try_from(bytes.len()).unwrap());
        assert_eq!(data["id3v2_bytes"], json!(20));
    }
}
