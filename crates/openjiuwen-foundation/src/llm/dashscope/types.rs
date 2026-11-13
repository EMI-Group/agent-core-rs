//! `DashScope` media request and response types.
//!
//! These are the caller-facing structures: they are `serde`-serializable so a
//! request can be persisted or logged, but they are **not** the vendor wire shape
//! — [`super::wire`] projects them onto the `DashScope` bodies. Every optional
//! parameter is omitted from the wire body when it is `None`, so a caller can
//! either accept the [`Default`] values (which mirror the Python SDK's keyword
//! defaults) or state every knob explicitly.
//!
//! ```no_run
//! use openjiuwen_foundation::llm::dashscope::{ImageRequest, VideoRequest};
//!
//! let image = ImageRequest::new("qwen-image-max").with_prompt("a cat in the sun");
//! assert_eq!(image.n, Some(1));
//!
//! let video = VideoRequest::new("wan2.6-t2v").with_duration(10);
//! assert_eq!(video.duration, Some(10));
//! ```

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Default image model (`qwen-image-max`, Python parity).
pub const DEFAULT_IMAGE_MODEL: &str = "qwen-image-max";

/// Default speech model (`cosyvoice-v1`, Python parity).
pub const DEFAULT_SPEECH_MODEL: &str = "cosyvoice-v1";

/// Default video model (`wan2.6-t2v`, Python parity).
pub const DEFAULT_VIDEO_MODEL: &str = "wan2.6-t2v";

/// Default image resolution (`1664*928`, Python parity).
pub const DEFAULT_IMAGE_SIZE: &str = "1664*928";

/// Default speech voice (`Cherry`, Python parity).
pub const DEFAULT_SPEECH_VOICE: &str = "Cherry";

/// Default speech language type (`Auto`, Python parity).
pub const DEFAULT_SPEECH_LANGUAGE_TYPE: &str = "Auto";

/// Default video duration in seconds (`5`, Python parity).
pub const DEFAULT_VIDEO_DURATION: u32 = 5;

/// A text-to-image (or image-edit) generation request.
///
/// The prompt is [`ImageRequest::prompt`] when set, otherwise the text of the
/// last user message of the [`BaseMessage`](crate::prompt::BaseMessage) slice
/// passed to the client. [`ImageRequest::images`] are the reference images sent
/// after the text entry; each becomes a `{"image": ...}` content entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ImageRequest {
    /// Model name, for example `qwen-image-max` or `wanx-v1`.
    pub model: String,
    /// Prompt text; `None` falls back to the last user message.
    pub prompt: Option<String>,
    /// Output size as `"<width>*<height>"`, for example `1664*928`.
    pub size: Option<String>,
    /// Number of images to generate.
    pub n: Option<u32>,
    /// Whether the vendor rewrites the prompt before generating.
    pub prompt_extend: Option<bool>,
    /// Whether a watermark is added to the output.
    pub watermark: Option<bool>,
    /// Sampling seed; `None` lets the vendor pick one.
    pub seed: Option<u64>,
    /// Things the image should not contain.
    pub negative_prompt: Option<String>,
    /// Reference images as `http(s)` URLs or `data:` URIs.
    pub images: Vec<String>,
}

impl Default for ImageRequest {
    /// `qwen-image-max`, `1664*928`, one image, prompt extension on, watermark off.
    fn default() -> Self {
        Self {
            model: DEFAULT_IMAGE_MODEL.to_owned(),
            prompt: None,
            size: Some(DEFAULT_IMAGE_SIZE.to_owned()),
            n: Some(1),
            prompt_extend: Some(true),
            watermark: Some(false),
            seed: None,
            negative_prompt: None,
            images: Vec::new(),
        }
    }
}

impl ImageRequest {
    /// Builds a request for `model` with the default output parameters.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }

    /// Sets the prompt text.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the output size.
    #[must_use]
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the number of images to generate.
    #[must_use]
    pub fn with_n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    /// Enables or disables vendor prompt rewriting.
    #[must_use]
    pub fn with_prompt_extend(mut self, prompt_extend: bool) -> Self {
        self.prompt_extend = Some(prompt_extend);
        self
    }

    /// Enables or disables the watermark.
    #[must_use]
    pub fn with_watermark(mut self, watermark: bool) -> Self {
        self.watermark = Some(watermark);
        self
    }

    /// Sets the sampling seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the negative prompt.
    #[must_use]
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Replaces the reference images.
    #[must_use]
    pub fn with_images<I, S>(mut self, images: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.images = images.into_iter().map(Into::into).collect();
        self
    }

    /// Appends one reference image.
    #[must_use]
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.images.push(image.into());
        self
    }
}

/// The result of one image generation call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ImageGenerationResponse {
    /// Generated images as `http(s)` URLs, in vendor order.
    pub images: Vec<String>,
    /// Generated images that came back inline; the `data:<mime>;base64,` prefix
    /// is stripped, so each entry is the base64 payload itself.
    pub images_base64: Vec<String>,
    /// Vendor creation timestamp, when reported.
    pub created: Option<i64>,
    /// Model that produced the images (echoed from the request when the vendor
    /// does not report it).
    pub model: String,
}

/// A text-to-speech request.
///
/// The text is [`SpeechRequest::text`] when set, otherwise the text of the last
/// user message of the [`BaseMessage`](crate::prompt::BaseMessage) slice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SpeechRequest {
    /// Model name, for example `cosyvoice-v1`.
    pub model: String,
    /// Text to speak; `None` falls back to the last user message.
    pub text: Option<String>,
    /// Voice name, for example `Cherry`.
    pub voice: Option<String>,
    /// Language hint, for example `Auto`, `Chinese`, or `English`.
    pub language_type: Option<String>,
}

impl Default for SpeechRequest {
    /// `cosyvoice-v1`, voice `Cherry`, language type `Auto` (Python parity).
    fn default() -> Self {
        Self {
            model: DEFAULT_SPEECH_MODEL.to_owned(),
            text: None,
            voice: Some(DEFAULT_SPEECH_VOICE.to_owned()),
            language_type: Some(DEFAULT_SPEECH_LANGUAGE_TYPE.to_owned()),
        }
    }
}

impl SpeechRequest {
    /// Builds a request for `model` with the default voice and language type.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }

    /// Sets the text to speak.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Sets the voice; `None` leaves the field off the wire body.
    #[must_use]
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Sets the language hint; `None` leaves the field off the wire body.
    #[must_use]
    pub fn with_language_type(mut self, language_type: impl Into<String>) -> Self {
        self.language_type = Some(language_type.into());
        self
    }
}

/// The result of one speech generation call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AudioGenerationResponse {
    /// Download URL of the generated audio.
    pub audio_url: Option<String>,
    /// Inline audio payload (base64), when the vendor returns the bytes directly.
    pub audio_data: Option<String>,
    /// Duration in seconds, when reported.
    pub duration: Option<f64>,
    /// Container format inferred from the URL or reported by the vendor
    /// (`mp3`, `wav`, `pcm`).
    pub format: Option<String>,
}

/// A text-to-video or image-to-video request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VideoRequest {
    /// Model name, for example `wan2.6-t2v`.
    pub model: String,
    /// Prompt text; `None` falls back to the last user message.
    pub prompt: Option<String>,
    /// First-frame image URL; setting it turns the call into image-to-video.
    pub img_url: Option<String>,
    /// Optional audio track URL for a lip-synced video.
    pub audio_url: Option<String>,
    /// Clip length in seconds.
    pub duration: Option<u32>,
    /// Output size in text-to-video mode; mutually exclusive with
    /// [`VideoRequest::resolution`].
    pub size: Option<String>,
    /// Output resolution in image-to-video mode; mutually exclusive with
    /// [`VideoRequest::size`].
    pub resolution: Option<String>,
    /// Whether the vendor rewrites the prompt before generating.
    pub prompt_extend: Option<bool>,
    /// Whether a watermark is added to the output.
    pub watermark: Option<bool>,
    /// Things the video should not contain.
    pub negative_prompt: Option<String>,
    /// Sampling seed; `None` lets the vendor pick one.
    pub seed: Option<u64>,
}

impl Default for VideoRequest {
    /// `wan2.6-t2v`, five seconds, prompt extension on, watermark off.
    fn default() -> Self {
        Self {
            model: DEFAULT_VIDEO_MODEL.to_owned(),
            prompt: None,
            img_url: None,
            audio_url: None,
            duration: Some(DEFAULT_VIDEO_DURATION),
            size: None,
            resolution: None,
            prompt_extend: Some(true),
            watermark: Some(false),
            negative_prompt: None,
            seed: None,
        }
    }
}

impl VideoRequest {
    /// Builds a request for `model` with the default clip parameters.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }

    /// Sets the prompt text.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the first-frame image, turning the call into image-to-video.
    #[must_use]
    pub fn with_img_url(mut self, img_url: impl Into<String>) -> Self {
        self.img_url = Some(img_url.into());
        self
    }

    /// Sets the audio track URL.
    #[must_use]
    pub fn with_audio_url(mut self, audio_url: impl Into<String>) -> Self {
        self.audio_url = Some(audio_url.into());
        self
    }

    /// Sets the clip length.
    #[must_use]
    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Sets the text-to-video output size.
    #[must_use]
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the image-to-video output resolution.
    #[must_use]
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Enables or disables vendor prompt rewriting.
    #[must_use]
    pub fn with_prompt_extend(mut self, prompt_extend: bool) -> Self {
        self.prompt_extend = Some(prompt_extend);
        self
    }

    /// Enables or disables the watermark.
    #[must_use]
    pub fn with_watermark(mut self, watermark: bool) -> Self {
        self.watermark = Some(watermark);
        self
    }

    /// Sets the negative prompt.
    #[must_use]
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Sets the sampling seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// The result of one video generation call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VideoGenerationResponse {
    /// Download URL of the generated video.
    pub video_url: String,
    /// Inline video payload, when the vendor returns the bytes directly.
    pub video_data: Option<String>,
    /// Clip length in seconds: the vendor's usage report, else the request's.
    pub duration: Option<f64>,
    /// Resolution reported by the vendor, else the request's `resolution`/`size`.
    pub resolution: Option<String>,
    /// Container format (`mp4` unless the vendor says otherwise).
    pub format: Option<String>,
}

impl Default for VideoGenerationResponse {
    /// An empty URL and no optional metadata; only used by `serde`.
    fn default() -> Self {
        Self {
            video_url: String::new(),
            video_data: None,
            duration: None,
            resolution: None,
            format: None,
        }
    }
}

/// Lifecycle state of an asynchronous `DashScope` task.
///
/// `DashScope` reports `PENDING`/`RUNNING` while working and `SUCCEEDED`/`FAILED`
/// once finished; any other spelling is kept in [`VideoTaskStatus::Unknown`] and
/// treated like a pending state by the polling loop, so an unrecognized status
/// never turns into a spurious result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoTaskStatus {
    /// The task is queued.
    Pending,
    /// The task is being generated.
    Running,
    /// The task finished and the result is available.
    Succeeded,
    /// The task finished with an error.
    Failed,
    /// A status this crate does not model, preserved verbatim.
    Unknown(String),
}

impl VideoTaskStatus {
    /// Parses a vendor status, case-insensitively.
    ///
    /// `"SUCCEEDED"`/`"SUCCESS"` map to [`VideoTaskStatus::Succeeded`],
    /// `"FAILED"` to [`VideoTaskStatus::Failed`], `"PENDING"` and `"RUNNING"` to
    /// their variants, and everything else (including an empty string) to
    /// [`VideoTaskStatus::Unknown`].
    #[must_use]
    pub fn parse(status: &str) -> Self {
        match status.trim().to_ascii_uppercase().as_str() {
            "PENDING" => Self::Pending,
            "RUNNING" => Self::Running,
            "SUCCEEDED" | "SUCCESS" => Self::Succeeded,
            "FAILED" => Self::Failed,
            other => Self::Unknown(other.to_owned()),
        }
    }

    /// The vendor spelling of the status.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pending => "PENDING",
            Self::Running => "RUNNING",
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
            Self::Unknown(status) => status,
        }
    }

    /// Whether the task finished successfully.
    #[must_use]
    pub fn is_succeeded(&self) -> bool {
        matches!(self, Self::Succeeded)
    }

    /// Whether the task finished with an error.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed)
    }

    /// Whether the polling loop may stop: the task succeeded or failed.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.is_succeeded() || self.is_failed()
    }
}

impl fmt::Display for VideoTaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for VideoTaskStatus {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for VideoTaskStatus {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let status = String::deserialize(deserializer)?;
        Ok(Self::parse(&status))
    }
}

/// A parsed asynchronous video task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct VideoTask {
    /// Vendor task id used to poll [`DashScope task`](super::DashScopeMediaClient).
    pub task_id: String,
    /// Current lifecycle state.
    pub task_status: VideoTaskStatus,
    /// Vendor error code, present once a task failed.
    pub code: Option<String>,
    /// Vendor error message, present once a task failed.
    pub message: Option<String>,
}

impl Default for VideoTask {
    /// No task id, [`VideoTaskStatus::Pending`], no error detail.
    fn default() -> Self {
        Self {
            task_id: String::new(),
            task_status: VideoTaskStatus::Pending,
            code: None,
            message: None,
        }
    }
}
