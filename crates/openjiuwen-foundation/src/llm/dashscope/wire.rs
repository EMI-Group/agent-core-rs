//! `DashScope` wire projection: request bodies, response parsers, and failures.
//!
//! Everything here is a pure function of its arguments, so the exact JSON the
//! client sends and the way responses are read can be asserted without a server.
//! [`build_image_body`], [`build_speech_body`], and [`build_video_body`] also own
//! the caller-input validation (empty model, empty prompt, conflicting video
//! sizing), which keeps the client a thin transport wrapper.
//!
//! Failure text never carries an unrecognized response body verbatim: a body that
//! is not a structured vendor error is rendered through
//! [`redact_str`](openjiuwen_common::security::redact_str) — a non-empty body
//! becomes the shared `<REDACTED>` placeholder — and every failure detail is
//! truncated to [`MAX_DETAIL_CHARS`] characters.

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::redact_str;
use serde_json::{json, Map, Value};

use crate::prompt::{BaseMessage, Role};

use super::types::{
    AudioGenerationResponse, ImageGenerationResponse, ImageRequest, SpeechRequest,
    VideoGenerationResponse, VideoRequest, VideoTask, VideoTaskStatus,
};

/// Path of the multimodal (image and speech) generation endpoint.
pub const MULTIMODAL_GENERATION_PATH: &str =
    "/api/v1/services/aigc/multimodal-generation/generation";

/// Path of the asynchronous video synthesis endpoint.
pub const VIDEO_SYNTHESIS_PATH: &str = "/api/v1/services/aigc/video-generation/video-synthesis";

/// Prefix of the asynchronous task status endpoint.
pub const TASK_PATH_PREFIX: &str = "/api/v1/tasks/";

/// Maximum number of characters of a failure detail kept in an error message.
const MAX_DETAIL_CHARS: usize = 500;

/// Marker appended to a detail that had to be truncated.
const TRUNCATION_MARKER: char = '…';

/// Vendor status value that means "no error".
const SUCCESS_CODE: &str = "Success";

/// Returns the polling path of `task_id`.
#[must_use]
pub fn task_path(task_id: &str) -> String {
    format!("{TASK_PATH_PREFIX}{}", task_id.trim())
}

/// Returns the trimmed model name, rejecting a blank one.
fn required_model(model: &str) -> Result<String> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err(Error::validation(
            "DashScope media requests require a non-empty model name",
        ));
    }
    Ok(trimmed.to_owned())
}

/// Returns the text of the last [`Role::User`] message, trimmed, when non-blank.
#[must_use]
pub fn last_user_text(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.text().into_owned())
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

/// Resolves a prompt: the explicit one when non-blank, else the last user text.
fn resolve_prompt(explicit: Option<&str>, messages: &[BaseMessage]) -> Option<String> {
    explicit
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(str::to_owned)
        .or_else(|| last_user_text(messages))
}

/// Inserts `key` when `value` is present.
fn insert_optional<T: Into<Value>>(map: &mut Map<String, Value>, key: &str, value: Option<T>) {
    if let Some(value) = value {
        map.insert(key.to_owned(), value.into());
    }
}

/// Inserts `key` when `value` is a non-blank string.
fn insert_text(map: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        map.insert(key.to_owned(), Value::String(value.to_owned()));
    }
}

/// Builds the body of an image generation call.
///
/// ```text
/// {"model": <model>,
///  "input": {"messages": [{"role": "user", "content": [{"text": <prompt>}, {"image": <url>}...]}]},
///  "parameters": {"size": ..., "n": ..., "prompt_extend": ..., "watermark": ...,
///                 "seed": ..., "negative_prompt": ...}}
/// ```
///
/// The prompt is `request.prompt`, else the text of the last user message; the
/// text entry always comes first and every entry of `request.images` follows as
/// `{"image": <url>}`.
///
/// # Errors
///
/// [`Error::validation`] for a blank model, a blank prompt, or a blank image
/// entry.
pub fn build_image_body(request: &ImageRequest, messages: &[BaseMessage]) -> Result<Value> {
    let model = required_model(&request.model)?;
    let prompt = resolve_prompt(request.prompt.as_deref(), messages).ok_or_else(|| {
        Error::validation(
            "DashScope image generation requires a non-empty prompt or user message text",
        )
    })?;

    let mut content = vec![json!({ "text": prompt })];
    for image in &request.images {
        let image = image.trim();
        if image.is_empty() {
            return Err(Error::validation(
                "DashScope image generation received a blank reference image",
            ));
        }
        content.push(json!({ "image": image }));
    }

    let mut parameters = Map::new();
    insert_text(&mut parameters, "size", request.size.as_deref());
    insert_optional(&mut parameters, "n", request.n);
    insert_optional(&mut parameters, "prompt_extend", request.prompt_extend);
    insert_optional(&mut parameters, "watermark", request.watermark);
    insert_optional(&mut parameters, "seed", request.seed);
    insert_text(
        &mut parameters,
        "negative_prompt",
        request.negative_prompt.as_deref(),
    );

    Ok(json!({
        "model": model,
        "input": { "messages": [{ "role": "user", "content": content }] },
        "parameters": Value::Object(parameters),
    }))
}

/// Builds the body of a speech generation call.
///
/// ```text
/// {"model": <model>, "input": {"text": <text>, "voice": ..., "language_type": ...}}
/// ```
///
/// The text is `request.text`, else the text of the last user message.
///
/// # Errors
///
/// [`Error::validation`] for a blank model or a blank text.
pub fn build_speech_body(request: &SpeechRequest, messages: &[BaseMessage]) -> Result<Value> {
    let model = required_model(&request.model)?;
    let text = resolve_prompt(request.text.as_deref(), messages).ok_or_else(|| {
        Error::validation(
            "DashScope speech generation requires a non-empty text or user message text",
        )
    })?;

    let mut input = Map::new();
    input.insert("text".to_owned(), Value::String(text));
    insert_text(&mut input, "voice", request.voice.as_deref());
    insert_text(
        &mut input,
        "language_type",
        request.language_type.as_deref(),
    );

    Ok(json!({ "model": model, "input": Value::Object(input) }))
}

/// Builds the body of an asynchronous video synthesis call.
///
/// ```text
/// {"model": <model>,
///  "input": {"prompt": <prompt>, "img_url": ..., "audio_url": ...},
///  "parameters": {"duration": ..., "prompt_extend": ..., "watermark": ...,
///                 "negative_prompt": ..., "seed": ..., "size" | "resolution"}}
/// ```
///
/// Sizing follows the vendor split between the two modes: image-to-video
/// (`img_url` set) accepts `resolution` only, and text-to-video accepts `size`
/// only. [`build_video_body`] rejects the other spelling instead of letting the
/// vendor drop it silently.
///
/// # Errors
///
/// [`Error::validation`] for a blank model, a blank prompt, a blank `img_url`, or
/// a `size`/`resolution` combination the vendor rejects.
pub fn build_video_body(request: &VideoRequest, messages: &[BaseMessage]) -> Result<Value> {
    let model = required_model(&request.model)?;
    let img_url = request
        .img_url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty());
    let size = request
        .size
        .as_deref()
        .map(str::trim)
        .filter(|size| !size.is_empty());
    let resolution = request
        .resolution
        .as_deref()
        .map(str::trim)
        .filter(|resolution| !resolution.is_empty());

    if img_url.is_some() {
        if size.is_some() {
            return Err(Error::validation(
                "DashScope image-to-video generation uses resolution; remove size",
            ));
        }
    } else if resolution.is_some() {
        return Err(Error::validation(
            "DashScope text-to-video generation uses size; remove resolution",
        ));
    }
    if request.img_url.is_some() && img_url.is_none() {
        return Err(Error::validation(
            "DashScope image-to-video generation received a blank img_url",
        ));
    }

    let prompt = resolve_prompt(request.prompt.as_deref(), messages).ok_or_else(|| {
        Error::validation(
            "DashScope video generation requires a non-empty prompt or user message text",
        )
    })?;

    let mut input = Map::new();
    input.insert("prompt".to_owned(), Value::String(prompt));
    if let Some(img_url) = img_url {
        input.insert("img_url".to_owned(), Value::String(img_url.to_owned()));
    }
    insert_text(&mut input, "audio_url", request.audio_url.as_deref());

    let mut parameters = Map::new();
    insert_optional(&mut parameters, "duration", request.duration);
    insert_optional(&mut parameters, "prompt_extend", request.prompt_extend);
    insert_optional(&mut parameters, "watermark", request.watermark);
    insert_text(
        &mut parameters,
        "negative_prompt",
        request.negative_prompt.as_deref(),
    );
    insert_optional(&mut parameters, "seed", request.seed);
    if let Some(size) = size {
        parameters.insert("size".to_owned(), Value::String(size.to_owned()));
    }
    if let Some(resolution) = resolution {
        parameters.insert(
            "resolution".to_owned(),
            Value::String(resolution.to_owned()),
        );
    }

    Ok(json!({
        "model": model,
        "input": Value::Object(input),
        "parameters": Value::Object(parameters),
    }))
}

/// Reads a non-blank string field of `value`.
fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .map(str::to_owned)
}

/// Returns `payload["output"]`, or an empty object when it is absent.
fn output_of(payload: &Value) -> &Value {
    payload.get("output").unwrap_or(&Value::Null)
}

/// Returns `value[key]` as an array, or an empty slice when it is absent.
fn array_field<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Splits a `data:<mime>;base64,<payload>` URL into its base64 payload.
fn base64_payload(url: &str) -> Option<String> {
    let lower = url.get(..5).map(str::to_ascii_lowercase);
    if lower.as_deref() != Some("data:") {
        return None;
    }
    let (_, payload) = url.split_once(',')?;
    let payload = payload.trim();
    if payload.is_empty() {
        None
    } else {
        Some(payload.to_owned())
    }
}

/// Parses an image generation response.
///
/// Every `output.choices[*].message.content[*].image` entry is collected: a
/// `data:<mime>;base64,…` entry lands in
/// [`images_base64`](ImageGenerationResponse::images_base64) without its prefix,
/// every other entry in [`images`](ImageGenerationResponse::images). `model` is the
/// response's own model when it reports one, `fallback_model` otherwise.
///
/// # Errors
///
/// [`Error::llm`] when the response carries no image at all.
pub fn parse_image_response(
    payload: &Value,
    fallback_model: &str,
) -> Result<ImageGenerationResponse> {
    let mut images = Vec::new();
    let mut images_base64 = Vec::new();
    for choice in array_field(output_of(payload), "choices") {
        let message = choice.get("message").unwrap_or(&Value::Null);
        let content = array_field(message, "content");
        for item in content {
            let Some(image) = item
                .get("image")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|image| !image.is_empty())
            else {
                continue;
            };
            match base64_payload(image) {
                Some(data) => images_base64.push(data),
                None => images.push(image.to_owned()),
            }
        }
    }
    if images.is_empty() && images_base64.is_empty() {
        return Err(Error::llm("DashScope image generation returned no images"));
    }
    let model = string_field(payload, "model")
        .or_else(|| string_field(output_of(payload), "model"))
        .unwrap_or_else(|| fallback_model.to_owned());
    let created = payload
        .get("created")
        .and_then(Value::as_i64)
        .or_else(|| output_of(payload).get("created").and_then(Value::as_i64));
    Ok(ImageGenerationResponse {
        images,
        images_base64,
        created,
        model,
    })
}

/// Infers an audio container format from a URL or file name.
fn audio_format_from_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    ["wav", "mp3", "pcm"]
        .into_iter()
        .find(|extension| lower.ends_with(&format!(".{extension}")))
        .map(str::to_owned)
}

/// Parses a speech generation response.
///
/// The audio is read from `output.audio.{url,data}` first, then from
/// `output.choices[*].message.content[*]` (`audio`, `audio_url`, `audio_data`,
/// `format`), matching the two shapes the vendor returns. The format is the
/// reported one, else inferred from the URL suffix; the duration comes from
/// `usage.duration`.
///
/// # Errors
///
/// [`Error::llm`] when neither a URL nor inline data is returned.
pub fn parse_speech_response(payload: &Value) -> Result<AudioGenerationResponse> {
    let output = output_of(payload);
    let mut audio_url = None;
    let mut audio_data = None;
    let mut format = None;

    if let Some(audio) = output.get("audio") {
        audio_url = string_field(audio, "url");
        audio_data = string_field(audio, "data");
    }
    for choice in array_field(output, "choices") {
        let message = choice.get("message").unwrap_or(&Value::Null);
        let content = array_field(message, "content");
        for item in content {
            audio_url = string_field(item, "audio")
                .or_else(|| string_field(item, "audio_url"))
                .or(audio_url);
            audio_data = string_field(item, "audio_data").or(audio_data);
            format = string_field(item, "format").or(format);
        }
    }
    if let Some(url) = &audio_url {
        format = format.or_else(|| audio_format_from_url(url));
    }
    if audio_url.is_none() && audio_data.is_none() {
        return Err(Error::llm(
            "DashScope speech generation returned no audio url or data",
        ));
    }
    let duration = payload
        .get("usage")
        .and_then(|usage| usage.get("duration"))
        .and_then(Value::as_f64);
    Ok(AudioGenerationResponse {
        audio_url,
        audio_data,
        duration,
        format,
    })
}

/// Parses an asynchronous task report (`output.task_id`/`output.task_status`).
///
/// # Errors
///
/// [`Error::llm`] when the payload carries no `output.task_id`.
pub fn parse_video_task(payload: &Value) -> Result<VideoTask> {
    let output = output_of(payload);
    let task_id = string_field(output, "task_id")
        .ok_or_else(|| Error::llm("DashScope video submission returned no task id"))?;
    let task_status = output
        .get("task_status")
        .and_then(Value::as_str)
        .map_or_else(
            || VideoTaskStatus::Unknown(String::new()),
            VideoTaskStatus::parse,
        );
    Ok(VideoTask {
        task_id,
        task_status,
        code: string_field(output, "code"),
        message: string_field(output, "message"),
    })
}

/// Parses the result of a finished video task.
///
/// The URL is `output.video_url`, else `output.url`. The duration is
/// `usage.duration`, then `usage.output_video_duration`, then the request's
/// `duration`; the resolution is `usage.size`, then the request's `resolution`
/// or `size`; the format defaults to `mp4`.
///
/// # Errors
///
/// [`Error::llm`] when the task reports no video URL.
pub fn parse_video_response(
    payload: &Value,
    request: &VideoRequest,
) -> Result<VideoGenerationResponse> {
    let output = output_of(payload);
    let video_url = string_field(output, "video_url")
        .or_else(|| string_field(output, "url"))
        .ok_or_else(|| Error::llm("DashScope video generation returned no video url"))?;
    let usage = payload.get("usage").unwrap_or(&Value::Null);
    let duration = usage
        .get("duration")
        .and_then(Value::as_f64)
        .or_else(|| usage.get("output_video_duration").and_then(Value::as_f64))
        .or_else(|| request.duration.map(f64::from));
    let resolution = string_field(usage, "size")
        .or_else(|| request.resolution.clone())
        .or_else(|| request.size.clone());
    let format = string_field(usage, "format")
        .or_else(|| string_field(output, "format"))
        .or_else(|| Some("mp4".to_owned()));
    Ok(VideoGenerationResponse {
        video_url,
        video_data: string_field(output, "video_data"),
        duration,
        resolution,
        format,
    })
}

/// Returns the vendor error `code`/`message` of a payload, when it reports one.
///
/// `DashScope` reports failures both as a non-2xx status and as an HTTP 200 body
/// whose `code`/`output.code` is not `Success`; both are treated the same way.
fn vendor_error_fields(payload: &Value) -> Option<(Option<String>, Option<String>)> {
    let code = string_field(payload, "code").or_else(|| string_field(output_of(payload), "code"));
    let message =
        string_field(payload, "message").or_else(|| string_field(output_of(payload), "message"));
    match &code {
        Some(code) if !code.eq_ignore_ascii_case(SUCCESS_CODE) => {
            Some((Some(code.clone()), message))
        }
        Some(_) => None,
        None => message.map(|message| (None, Some(message))),
    }
}

/// Whether a payload reports a vendor error (`code` other than `Success`).
///
/// A payload that carries `output.task_status` is *not* treated as a vendor error,
/// even when it also carries `output.code`: a `FAILED` task is reported by the
/// task path ([`task_failure`]), which names the task id.
#[must_use]
pub(crate) fn has_vendor_error(payload: &Value) -> bool {
    if output_of(payload).get("task_status").is_some() {
        return false;
    }
    vendor_error_fields(payload).is_some()
}

/// Truncates `text` to at most [`MAX_DETAIL_CHARS`] characters on a char boundary.
fn truncate_detail(text: &str) -> String {
    let mut truncated: String = text.chars().take(MAX_DETAIL_CHARS).collect();
    if truncated.chars().count() < text.chars().count() {
        truncated.push(TRUNCATION_MARKER);
    }
    truncated
}

/// Renders the human-readable detail of a failure.
///
/// A structured vendor error contributes its `code`/`message`; otherwise the raw
/// body is rendered through [`redact_str`] so no unrecognized body — which could
/// echo the request — reaches an error message. The result is truncated to
/// [`MAX_DETAIL_CHARS`] characters.
fn failure_detail(code: Option<&str>, message: Option<&str>, body: &str) -> String {
    let detail = match (code, message) {
        (Some(code), Some(message)) => format!("code={code}, message={message}"),
        (Some(code), None) => format!("code={code}"),
        (None, Some(message)) => format!("message={message}"),
        (None, None) => redact_str(body),
    };
    truncate_detail(detail.trim())
}

/// Maps a rejected HTTP response to an error that names the vendor reason.
///
/// Kinds follow the rest of the `llm` module: `401`/`403` are
/// [`Error::Config`] (an authentication problem no retry can fix), `408`/`504`
/// are [`Error::Timeout`], and every other status is [`Error::Llm`].
///
/// # Errors
///
/// Always returns an error.
pub(crate) fn http_failure(operation: &str, status: u16, url: &str, body: &str) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let (code, message) = parsed
        .as_ref()
        .and_then(vendor_error_fields)
        .unwrap_or((None, None));
    let detail = failure_detail(code.as_deref(), message.as_deref(), body);
    let suffix = if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    };
    match status {
        401 | 403 => Error::config(format!(
            "DashScope {operation} authentication failed with HTTP {status} for {url}{suffix}"
        )),
        408 | 504 => Error::timeout(format!(
            "DashScope {operation} timed out with HTTP {status} for {url}{suffix}"
        )),
        _ => Error::llm(format!(
            "DashScope {operation} failed with HTTP {status} for {url}{suffix}"
        )),
    }
}

/// Maps a vendor error reported inside a successful response to an error.
///
/// # Errors
///
/// Always returns an error.
pub(crate) fn vendor_failure(operation: &str, payload: &Value) -> Error {
    let (code, message) = vendor_error_fields(payload).unwrap_or((None, None));
    let detail = failure_detail(code.as_deref(), message.as_deref(), &payload.to_string());
    Error::llm(format!("DashScope {operation} failed: {detail}"))
}

/// Maps a failed asynchronous task to an error carrying the vendor reason.
///
/// # Errors
///
/// Always returns an error.
pub(crate) fn task_failure(operation: &str, task: &VideoTask) -> Error {
    let detail = failure_detail(
        task.code.as_deref(),
        task.message.as_deref(),
        &format!("task {} reported FAILED", task.task_id),
    );
    Error::llm(format!(
        "DashScope {operation} task {} failed: {detail}",
        task.task_id
    ))
}
