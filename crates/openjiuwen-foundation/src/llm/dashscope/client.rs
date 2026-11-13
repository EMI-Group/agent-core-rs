//! The `DashScope` media generation client.
//!
//! [`DashScopeMediaClient`] turns the caller-facing request types into `DashScope`
//! HTTP calls: image and speech go to the multimodal generation endpoint in one
//! round trip, video is submitted with `X-DashScope-Async: enable` and then polled
//! until the vendor reports `SUCCEEDED` or `FAILED`.

use std::time::{Duration, Instant};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use serde_json::Value;

use crate::llm::transport::join_url;
use crate::prompt::BaseMessage;

use super::config::DashScopeConfig;
use super::types::{
    AudioGenerationResponse, ImageGenerationResponse, ImageRequest, SpeechRequest,
    VideoGenerationResponse, VideoRequest, VideoTaskStatus,
};
use super::wire;

/// The `tracing` target every log line of this module uses.
const LOG_TARGET: &str = "openjiuwen_foundation::llm::dashscope";

/// Header that turns a `DashScope` generation call into an asynchronous task.
const ASYNC_HEADER: (&str, &str) = ("X-DashScope-Async", "enable");

/// Client for the `DashScope` image, speech, and video generation APIs.
///
/// The client is `DashScope`-specific: the Python SDK raises unless the configured
/// endpoint profile is `dashscope`, and the Rust port leaves that gate at the call
/// site — construct this client only for a `ModelClientConfig` whose provider is
/// `qwen`/`dashscope` (see [`DashScopeConfig::from_client_config`]).
///
/// The client is cheap to clone (the [`HttpClient`] handle is shared) and keeps
/// only the configuration, so several calls can run concurrently from clones.
#[derive(Debug, Clone)]
pub struct DashScopeMediaClient {
    http: HttpClient,
    config: DashScopeConfig,
}

impl DashScopeMediaClient {
    /// Builds a client from `config` with a default HTTP stack.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when the key or `api_base` is blank, [`Error::Validation`]
    /// when `poll_interval` is zero, and the errors of [`HttpClient::new`] when the
    /// HTTP stack cannot be built.
    pub fn new(config: DashScopeConfig) -> Result<Self> {
        validate_config(&config)?;
        let http = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;
        Ok(Self { http, config })
    }

    /// Builds a client whose HTTP transport is supplied by the caller.
    ///
    /// Use this to share one [`HttpClient`] (and its retry policy, proxy, and TLS
    /// settings) between several clients, or to inject a test transport.
    ///
    /// # Errors
    ///
    /// The same validation errors as [`DashScopeMediaClient::new`].
    pub fn with_http_client(config: DashScopeConfig, http: HttpClient) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self { http, config })
    }

    /// Returns the configuration this client was built with.
    #[must_use]
    pub fn config(&self) -> &DashScopeConfig {
        &self.config
    }

    /// Generates images from a prompt.
    ///
    /// The prompt is `request.prompt`, else the text of the last user message of
    /// `messages`; `request.images` are sent as reference images after the text.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for a blank model/prompt/image, [`Error::Config`] for
    /// rejected credentials, [`Error::Timeout`] for a timed-out request, and
    /// [`Error::Llm`] for any other vendor failure, a vendor error reported inside
    /// a `200` body, or a response that carries no image.
    pub async fn generate_image(
        &self,
        messages: &[BaseMessage],
        request: ImageRequest,
    ) -> Result<ImageGenerationResponse> {
        let model = request.model.trim().to_owned();
        let body = wire::build_image_body(&request, messages)?;
        let url = join_url(&self.config.api_base, wire::MULTIMODAL_GENERATION_PATH);
        tracing::debug!(
            target: LOG_TARGET,
            model = %model,
            images = request.images.len(),
            "submitting DashScope image generation"
        );
        let payload = self
            .send(HttpRequest::post(url).json(&body)?, "image generation")
            .await?;
        wire::parse_image_response(&payload, &model)
    }

    /// Generates speech (text-to-speech) from a prompt.
    ///
    /// The text is `request.text`, else the text of the last user message of
    /// `messages`.
    ///
    /// # Errors
    ///
    /// The same error surface as [`DashScopeMediaClient::generate_image`], plus a
    /// response that carries neither an audio URL nor inline audio data.
    pub async fn generate_speech(
        &self,
        messages: &[BaseMessage],
        request: SpeechRequest,
    ) -> Result<AudioGenerationResponse> {
        let body = wire::build_speech_body(&request, messages)?;
        let url = join_url(&self.config.api_base, wire::MULTIMODAL_GENERATION_PATH);
        tracing::debug!(
            target: LOG_TARGET,
            model = %request.model.trim(),
            "submitting DashScope speech generation"
        );
        let payload = self
            .send(HttpRequest::post(url).json(&body)?, "speech generation")
            .await?;
        wire::parse_speech_response(&payload)
    }

    /// Generates a video, polling the asynchronous task until it finishes.
    ///
    /// The submission carries `X-DashScope-Async: enable`; the returned task is
    /// polled every [`poll_interval`](DashScopeConfig::poll_interval) for at most
    /// [`task_timeout`](DashScopeConfig::task_timeout). A vendor that answers the
    /// submission synchronously with `SUCCEEDED` is read directly, without polling.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for a blank model/prompt or a `size`/`resolution`
    /// combination the vendor rejects, [`Error::Config`] for rejected credentials,
    /// [`Error::Timeout`] when the task does not finish within
    /// [`task_timeout`](DashScopeConfig::task_timeout), and [`Error::Llm`] for a
    /// rejected submission, a `FAILED` task (carrying the vendor `code`/`message`),
    /// or a task result without a video URL.
    pub async fn generate_video(
        &self,
        messages: &[BaseMessage],
        request: VideoRequest,
    ) -> Result<VideoGenerationResponse> {
        let body = wire::build_video_body(&request, messages)?;
        let url = join_url(&self.config.api_base, wire::VIDEO_SYNTHESIS_PATH);
        let submit = HttpRequest::post(url)
            .header(ASYNC_HEADER.0, ASYNC_HEADER.1)
            .json(&body)?;
        let payload = self.send(submit, "video generation").await?;
        let task = wire::parse_video_task(&payload)?;
        tracing::debug!(
            target: LOG_TARGET,
            task_id = %task.task_id,
            status = %task.task_status,
            "submitted DashScope video task"
        );
        if task.task_status.is_failed() {
            return Err(wire::task_failure("video generation", &task));
        }
        if task.task_status.is_succeeded() {
            return wire::parse_video_response(&payload, &request);
        }
        self.wait_for_video_task(&task.task_id, &request).await
    }

    /// Polls `task_id` until it succeeds, fails, or the task budget elapses.
    async fn wait_for_video_task(
        &self,
        task_id: &str,
        request: &VideoRequest,
    ) -> Result<VideoGenerationResponse> {
        let deadline = Instant::now() + self.config.task_timeout;
        let url = join_url(&self.config.api_base, &wire::task_path(task_id));
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(task_timed_out(task_id, self.config.task_timeout));
            }
            tokio::time::sleep(self.config.poll_interval.min(remaining)).await;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(task_timed_out(task_id, self.config.task_timeout));
            }
            let poll = HttpRequest::get(url.clone())
                .bearer(self.config.api_key.clone())
                .timeout(remaining);
            let response = match tokio::time::timeout(remaining, self.http.execute(poll)).await {
                Ok(response) => response?,
                Err(_) => return Err(task_timed_out(task_id, self.config.task_timeout)),
            };
            let payload = decode("video generation", &response)?;
            let task = wire::parse_video_task(&payload)?;
            tracing::debug!(
                target: LOG_TARGET,
                task_id = %task.task_id,
                status = %task.task_status,
                "polled DashScope video task"
            );
            match task.task_status {
                VideoTaskStatus::Succeeded => return wire::parse_video_response(&payload, request),
                VideoTaskStatus::Failed => {
                    return Err(wire::task_failure("video generation", &task));
                }
                VideoTaskStatus::Pending
                | VideoTaskStatus::Running
                | VideoTaskStatus::Unknown(_) => {}
            }
        }
    }

    /// Executes one authorized request and decodes its payload.
    async fn send(&self, request: HttpRequest, operation: &str) -> Result<Value> {
        let response = self
            .http
            .execute(request.bearer(self.config.api_key.clone()))
            .await?;
        decode(operation, &response)
    }
}

/// Rejects a configuration that cannot produce a request.
fn validate_config(config: &DashScopeConfig) -> Result<()> {
    if config.api_key.is_empty() || config.api_key.expose_secret().trim().is_empty() {
        return Err(Error::config(
            "DashScope media client requires a non-empty API key",
        ));
    }
    if config.api_base.trim().is_empty() {
        return Err(Error::config(
            "DashScope media client requires a non-empty api_base",
        ));
    }
    if config.poll_interval.is_zero() {
        return Err(Error::validation(
            "DashScope poll_interval must be non-zero to avoid a busy polling loop",
        ));
    }
    Ok(())
}

/// The error returned when the task budget elapses.
fn task_timed_out(task_id: &str, task_timeout: Duration) -> Error {
    Error::timeout(format!(
        "DashScope video generation task {task_id} did not finish within {task_timeout:?}"
    ))
}

/// Turns a buffered response into its JSON payload, or into the matching error.
///
/// A non-2xx status becomes `wire::http_failure`; a `200` body without parseable
/// JSON becomes [`Error::Llm`]; a vendor error reported inside a `200` body becomes
/// `wire::vendor_failure`.
fn decode(operation: &str, response: &HttpResponse) -> Result<Value> {
    let text = response.text();
    if !response.is_success() {
        return Err(wire::http_failure(
            operation,
            response.status,
            &response.url,
            &text,
        ));
    }
    let payload: Value = serde_json::from_str(&text).map_err(|err| {
        Error::llm(format!(
            "DashScope {operation} returned an unparsable response from {}: {err}",
            response.url
        ))
    })?;
    if wire::has_vendor_error(&payload) {
        return Err(wire::vendor_failure(operation, &payload));
    }
    Ok(payload)
}
