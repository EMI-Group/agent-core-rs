//! `DashScope` media generation: image synthesis, speech synthesis, and video
//! synthesis over the vendor's plain HTTP API.
//!
//! The Python SDK reaches these models through the vendor `dashscope` SDK
//! (`MultiModalConversation`, `VideoSynthesis`) from the OpenAI-compatible client,
//! gated on `endpoint_profile == "dashscope"`. There is no vendor SDK in this
//! workspace, so the same HTTP surface is implemented directly on
//! [`HttpClient`](openjiuwen_common::http::HttpClient):
//!
//! - [`DashScopeMediaClient::generate_image`] — `POST
//!   {base}/api/v1/services/aigc/multimodal-generation/generation`.
//! - [`DashScopeMediaClient::generate_speech`] — the same endpoint, text-to-speech
//!   body.
//! - [`DashScopeMediaClient::generate_video`] — `POST
//!   {base}/api/v1/services/aigc/video-generation/video-synthesis` with
//!   `X-DashScope-Async: enable`, then `GET {base}/api/v1/tasks/{task_id}` until
//!   the task is `SUCCEEDED` or `FAILED`.
//!
//! The client is **`DashScope`-specific**. The Python SDK refuses these calls for
//! any other endpoint profile; here the gate belongs at the call site — construct
//! a [`DashScopeMediaClient`] only for a `ModelClientConfig` whose provider is
//! `qwen`/`dashscope` ([`DashScopeConfig::from_client_config`] adapts one).
//!
//! # Error contract
//!
//! | Situation | Kind |
//! | --- | --- |
//! | blank key / blank `api_base` | [`Error::Config`](openjiuwen_common::error::Error::config) |
//! | blank model/prompt, conflicting video sizing | [`Error::Validation`](openjiuwen_common::error::Error::validation) |
//! | HTTP 401/403 | [`Error::Config`](openjiuwen_common::error::Error::config) |
//! | HTTP 408/504, task budget elapsed | [`Error::Timeout`](openjiuwen_common::error::Error::timeout) |
//! | any other non-2xx, vendor error in a `200`, `FAILED` task, empty result | [`Error::Llm`](openjiuwen_common::error::Error::llm) |
//!
//! Credentials are held in [`SecretString`](openjiuwen_common::security::SecretString)
//! and never logged. A vendor `code`/`message` pair is surfaced (truncated to 500
//! characters) because it is the only actionable part of a `DashScope` failure; any
//! unrecognized response body is rendered through
//! [`redact_str`](openjiuwen_common::security::redact_str) instead of being echoed.
//!
//! # Deferrals
//!
//! - No streaming media: every call returns the finished URL/data set, matching the
//!   Python `stream=False` multimodal call.
//! - No byte download: `audio_data`/`video_data` are only filled when the vendor
//!   returns the payload inline; the client never fetches the result URL.
//! - No `X-DashScope-WorkSpace` (workspace scoping), no voice/language allow-list
//!   validation (Python checks `DASHSCOPE_VOICE`/`DASHSCOPE_LANGUAGE_TYPE`), no
//!   local-file or `oss://` inputs, and no per-call `**kwargs` escape hatch — extra
//!   vendor parameters are not forwarded.
//!
//! ```no_run
//! use openjiuwen_foundation::llm::dashscope::{
//!     DashScopeConfig, DashScopeMediaClient, ImageRequest, VideoRequest,
//! };
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let config = DashScopeConfig::new("sk-dashscope").with_api_base("https://dashscope.aliyuncs.com");
//! let client = DashScopeMediaClient::new(config)?;
//!
//! let images = client
//!     .generate_image(
//!         &[BaseMessage::user("a cat asleep in the sun")],
//!         ImageRequest::default(),
//!     )
//!     .await?;
//! assert!(!images.images.is_empty() || !images.images_base64.is_empty());
//!
//! let video = client
//!     .generate_video(
//!         &[BaseMessage::user("a paper boat drifting down a stream")],
//!         VideoRequest::default(),
//!     )
//!     .await?;
//! assert!(!video.video_url.is_empty());
//! # Ok(())
//! # }
//! ```

mod client;
mod config;
#[cfg(test)]
mod tests;
mod types;
mod wire;

pub use client::DashScopeMediaClient;
pub use config::{
    DashScopeConfig, DASHSCOPE_API_BASE_ENV, DASHSCOPE_API_KEY_ENV, DEFAULT_DASHSCOPE_API_BASE,
    DEFAULT_MODEL_TIMEOUT, DEFAULT_POLL_INTERVAL, DEFAULT_TASK_TIMEOUT,
    OPENJIUWEN_DASHSCOPE_API_BASE_ENV, OPENJIUWEN_DASHSCOPE_API_KEY_ENV,
};
pub use types::{
    AudioGenerationResponse, ImageGenerationResponse, ImageRequest, SpeechRequest,
    VideoGenerationResponse, VideoRequest, VideoTask, VideoTaskStatus, DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_SIZE, DEFAULT_SPEECH_LANGUAGE_TYPE, DEFAULT_SPEECH_MODEL, DEFAULT_SPEECH_VOICE,
    DEFAULT_VIDEO_DURATION, DEFAULT_VIDEO_MODEL,
};
pub use wire::{
    build_image_body, build_speech_body, build_video_body, last_user_text, parse_image_response,
    parse_speech_response, parse_video_response, parse_video_task, task_path,
    MULTIMODAL_GENERATION_PATH, TASK_PATH_PREFIX, VIDEO_SYNTHESIS_PATH,
};
