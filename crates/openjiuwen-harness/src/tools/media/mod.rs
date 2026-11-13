//! Media metadata tools: `image_metadata` and `audio_metadata`.
//!
//! Pure header parsing (no decoding crates) for the formats the model can be pointed at:
//! PNG, JPEG, GIF, WebP, BMP (images) and WAV, FLAC, MP3, OGG (audio).
//! Ports the metadata half of `openjiuwen/harness/tools/multimodal/` (`audio.py`
//! `audio_metadata`, `vision.py`) as a model-free, offline subset.
//!
//! # Required public API (registry contract)
//!
//! - `ImageMetadataTool::new(env: ToolEnv, language: impl Into<String>) -> Self` and
//!   `ImageMetadataTool::card_for(language: &str) -> ToolCard`; model-facing name
//!   `image_metadata`.
//! - `AudioMetadataTool::new(env: ToolEnv, language: impl Into<String>) -> Self` and
//!   `AudioMetadataTool::card_for(language: &str) -> ToolCard`; model-facing name
//!   `audio_metadata`.
//! - Both resolve model-supplied paths through `ToolEnv::resolve_path` and read the bytes
//!   from the resolved path (never a raw `Path::new(model_input)`).
