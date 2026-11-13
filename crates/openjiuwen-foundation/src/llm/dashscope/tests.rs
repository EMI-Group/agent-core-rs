//! Tests of the `DashScope` media client, its wire projection, and its failures.
//!
//! Every HTTP test talks to a [`MockServer`] on loopback, so the suite needs no
//! network and no credential. Request bodies are asserted exactly through
//! [`build_*_body`], and the transport tests use `wiremock` matchers for the path,
//! the bearer token, the async header, and the JSON body.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::http::HttpClient;
use openjiuwen_common::security::REDACTED;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use super::*;
use crate::prompt::BaseMessage;

/// A configuration pointing at `server` with test-sized polling budgets.
fn test_config(server: &MockServer) -> DashScopeConfig {
    DashScopeConfig::new("test-key")
        .with_api_base(server.uri())
        .with_timeout(Duration::from_secs(5))
        .with_poll_interval(Duration::from_millis(10))
        .with_task_timeout(Duration::from_secs(2))
}

/// Builds a client for `server`.
fn test_client(server: &MockServer) -> DashScopeMediaClient {
    DashScopeMediaClient::new(test_config(server)).expect("the test client builds")
}

// ---------------------------------------------------------------------------
// Request bodies
// ---------------------------------------------------------------------------

#[test]
fn image_body_carries_the_prompt_content_and_parameters() {
    let request = ImageRequest::new("qwen-image-max")
        .with_prompt("a cat asleep in the sun")
        .with_seed(7)
        .with_negative_prompt("blurry");

    let body = build_image_body(&request, &[]).expect("the request is valid");

    assert_eq!(
        body,
        json!({
            "model": "qwen-image-max",
            "input": {"messages": [{"role": "user", "content": [{"text": "a cat asleep in the sun"}]}]},
            "parameters": {
                "size": "1664*928",
                "n": 1,
                "prompt_extend": true,
                "watermark": false,
                "seed": 7,
                "negative_prompt": "blurry"
            }
        })
    );
}

#[test]
fn image_body_falls_back_to_the_last_user_message_and_appends_references() {
    let request = ImageRequest::new("wanx-v1")
        .with_size("1024*1024")
        .with_n(2)
        .with_prompt_extend(false)
        .with_watermark(true)
        .with_image("https://example.test/reference.png")
        .with_image("data:image/png;base64,QUJD");

    let messages = [
        BaseMessage::system("ignored"),
        BaseMessage::user("first prompt"),
        BaseMessage::assistant("ignored too"),
        BaseMessage::user("  a red bicycle  "),
    ];
    let body = build_image_body(&request, &messages).expect("the request is valid");

    assert_eq!(
        body,
        json!({
            "model": "wanx-v1",
            "input": {"messages": [{"role": "user", "content": [
                {"text": "a red bicycle"},
                {"image": "https://example.test/reference.png"},
                {"image": "data:image/png;base64,QUJD"}
            ]}]},
            "parameters": {"size": "1024*1024", "n": 2, "prompt_extend": false, "watermark": true}
        })
    );
}

#[test]
fn the_explicit_prompt_wins_over_the_user_message() {
    let request = ImageRequest::new("qwen-image-max").with_prompt("explicit");
    let body = build_image_body(&request, &[BaseMessage::user("from the message")])
        .expect("the request is valid");
    assert_eq!(
        body["input"]["messages"][0]["content"][0]["text"],
        "explicit"
    );
}

#[test]
fn an_image_request_without_a_prompt_is_a_validation_error() {
    let error = build_image_body(&ImageRequest::default(), &[BaseMessage::user("   ")])
        .expect_err("there is no prompt");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("non-empty prompt"));
}

#[test]
fn a_blank_reference_image_is_a_validation_error() {
    let request = ImageRequest::new("qwen-image-max")
        .with_prompt("a cat")
        .with_image("   ");
    let error = build_image_body(&request, &[]).expect_err("the image is blank");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn a_blank_model_is_a_validation_error_for_every_media_kind() {
    for error in [
        build_image_body(&ImageRequest::new("  ").with_prompt("x"), &[]).expect_err("blank model"),
        build_speech_body(&SpeechRequest::new("").with_text("x"), &[]).expect_err("blank model"),
        build_video_body(&VideoRequest::new(" ").with_prompt("x"), &[]).expect_err("blank model"),
    ] {
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("model"));
    }
}

#[test]
fn speech_body_uses_the_explicit_text_or_the_user_message() {
    let explicit = build_speech_body(
        &SpeechRequest::new("cosyvoice-v1").with_text("hello there"),
        &[BaseMessage::user("ignored")],
    )
    .expect("the request is valid");
    assert_eq!(
        explicit,
        json!({
            "model": "cosyvoice-v1",
            "input": {"text": "hello there", "voice": "Cherry", "language_type": "Auto"}
        })
    );

    let from_message = build_speech_body(
        &SpeechRequest::new("cosyvoice-v2"),
        &[BaseMessage::user("read this out")],
    )
    .expect("the request is valid");
    assert_eq!(from_message["input"]["text"], "read this out");
}

#[test]
fn speech_omits_a_voice_and_language_type_set_to_none() {
    let mut request = SpeechRequest::new("cosyvoice-v1").with_text("hello");
    request.voice = None;
    request.language_type = None;
    let body = build_speech_body(&request, &[]).expect("the request is valid");
    assert_eq!(
        body,
        json!({"model": "cosyvoice-v1", "input": {"text": "hello"}})
    );
}

#[test]
fn a_speech_request_without_text_is_a_validation_error() {
    let error = build_speech_body(&SpeechRequest::default(), &[]).expect_err("there is no text");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("non-empty text"));
}

#[test]
fn video_body_text_to_video_uses_size() {
    let request = VideoRequest::new("wan2.6-t2v")
        .with_prompt("a paper boat drifting down a stream")
        .with_size("1280*720")
        .with_seed(11)
        .with_negative_prompt("rain");

    let body = build_video_body(&request, &[]).expect("the request is valid");

    assert_eq!(
        body,
        json!({
            "model": "wan2.6-t2v",
            "input": {"prompt": "a paper boat drifting down a stream"},
            "parameters": {
                "duration": 5,
                "prompt_extend": true,
                "watermark": false,
                "negative_prompt": "rain",
                "seed": 11,
                "size": "1280*720"
            }
        })
    );
}

#[test]
fn video_body_image_to_video_uses_resolution_and_the_image_inputs() {
    let request = VideoRequest::new("wan2.6-i2v")
        .with_prompt("animate this")
        .with_img_url("https://example.test/frame.png")
        .with_audio_url("https://example.test/track.mp3")
        .with_resolution("1920*1080")
        .with_duration(10);

    let body = build_video_body(&request, &[]).expect("the request is valid");

    assert_eq!(
        body,
        json!({
            "model": "wan2.6-i2v",
            "input": {
                "prompt": "animate this",
                "img_url": "https://example.test/frame.png",
                "audio_url": "https://example.test/track.mp3"
            },
            "parameters": {
                "duration": 10,
                "prompt_extend": true,
                "watermark": false,
                "resolution": "1920*1080"
            }
        })
    );
}

#[test]
fn video_body_rejects_size_together_with_an_image() {
    let request = VideoRequest::new("wan2.6-i2v")
        .with_prompt("animate this")
        .with_img_url("https://example.test/frame.png")
        .with_size("1280*720");
    let error = build_video_body(&request, &[]).expect_err("the vendor rejects size here");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("resolution"));
}

#[test]
fn video_body_rejects_resolution_together_with_size_and_without_an_image() {
    let both = VideoRequest::new("wan2.6-t2v")
        .with_prompt("x")
        .with_size("1280*720")
        .with_resolution("1920*1080");
    let error = build_video_body(&both, &[]).expect_err("the vendor rejects resolution here");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("uses size"));
}

#[test]
fn video_body_rejects_an_empty_prompt() {
    let error = build_video_body(&VideoRequest::default(), &[]).expect_err("there is no prompt");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("non-empty prompt"));
}

#[test]
fn last_user_text_ignores_other_roles_and_blank_text() {
    let messages = [
        BaseMessage::system("system"),
        BaseMessage::user("  keep me  "),
        BaseMessage::assistant("no"),
    ];
    assert_eq!(last_user_text(&messages).as_deref(), Some("keep me"));
    assert_eq!(last_user_text(&[BaseMessage::user("   ")]), None);
    assert_eq!(last_user_text(&[]), None);
}

#[test]
fn task_path_prefixes_the_id() {
    assert_eq!(task_path("task-1"), "/api/v1/tasks/task-1");
    assert_eq!(task_path("  task-2  "), "/api/v1/tasks/task-2");
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

#[test]
fn image_response_splits_urls_and_inline_base64() {
    let payload = json!({
        "output": {"choices": [{"message": {"content": [
            {"image": "https://cdn.example/a.png"},
            {"image": "data:image/png;base64,QUJD"},
            {"text": "ignored"}
        ]}}]},
        "created": 1_700_000_000_i64,
        "model": "qwen-image-max"
    });

    let parsed = parse_image_response(&payload, "fallback").expect("there is an image");
    assert_eq!(parsed.images, vec!["https://cdn.example/a.png"]);
    assert_eq!(parsed.images_base64, vec!["QUJD"]);
    assert_eq!(parsed.created, Some(1_700_000_000));
    assert_eq!(parsed.model, "qwen-image-max");
}

#[test]
fn image_response_falls_back_to_the_request_model() {
    let payload = json!({"output": {"choices": [{"message": {"content": [{"image": "https://cdn.example/a.png"}]}}]}});
    let parsed = parse_image_response(&payload, "wanx-v1").expect("there is an image");
    assert_eq!(parsed.model, "wanx-v1");
    assert_eq!(parsed.created, None);
}

#[test]
fn image_response_without_any_image_is_an_llm_error() {
    let payload = json!({"output": {"choices": [{"message": {"content": [{"text": "sorry"}]}}]}});
    let error = parse_image_response(&payload, "qwen-image-max").expect_err("there is no image");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("no images"));
}

#[test]
fn speech_response_reads_output_audio_and_infers_the_format() {
    let payload = json!({
        "output": {"audio": {"url": "https://cdn.example/speech.wav", "data": "QUJD"}},
        "usage": {"duration": 1.5}
    });

    let parsed = parse_speech_response(&payload).expect("there is audio");
    assert_eq!(
        parsed.audio_url.as_deref(),
        Some("https://cdn.example/speech.wav")
    );
    assert_eq!(parsed.audio_data.as_deref(), Some("QUJD"));
    assert_eq!(parsed.format.as_deref(), Some("wav"));
    assert_eq!(parsed.duration, Some(1.5));
}

#[test]
fn speech_response_reads_the_choice_content_shape() {
    let payload = json!({
        "output": {"choices": [{"message": {"content": [
            {"audio": "https://cdn.example/speech.mp3", "format": "mp3"}
        ]}}]}
    });

    let parsed = parse_speech_response(&payload).expect("there is audio");
    assert_eq!(
        parsed.audio_url.as_deref(),
        Some("https://cdn.example/speech.mp3")
    );
    assert_eq!(parsed.format.as_deref(), Some("mp3"));
    assert_eq!(parsed.audio_data, None);
    assert_eq!(parsed.duration, None);
}

#[test]
fn speech_response_without_audio_is_an_llm_error() {
    let error = parse_speech_response(&json!({"output": {}})).expect_err("there is no audio");
    assert_eq!(error.kind(), ErrorKind::Llm);
}

#[test]
fn video_task_parses_every_status() {
    let pending = parse_video_task(&json!({"output": {"task_id": "t1", "task_status": "PENDING"}}))
        .expect("the task parses");
    assert_eq!(pending.task_status, VideoTaskStatus::Pending);
    assert!(!pending.task_status.is_terminal());

    let running = parse_video_task(&json!({"output": {"task_id": "t1", "task_status": "running"}}))
        .expect("the task parses");
    assert_eq!(running.task_status, VideoTaskStatus::Running);

    let succeeded =
        parse_video_task(&json!({"output": {"task_id": "t1", "task_status": "SUCCEEDED"}}))
            .expect("the task parses");
    assert!(succeeded.task_status.is_succeeded());
    assert!(succeeded.task_status.is_terminal());

    let failed = parse_video_task(&json!({"output": {
        "task_id": "t1", "task_status": "FAILED", "code": "InvalidParameter", "message": "bad"
    }}))
    .expect("the task parses");
    assert!(failed.task_status.is_failed());
    assert_eq!(failed.code.as_deref(), Some("InvalidParameter"));
    assert_eq!(failed.message.as_deref(), Some("bad"));

    let unknown =
        parse_video_task(&json!({"output": {"task_id": "t1", "task_status": "CANCELED"}}))
            .expect("the task parses");
    assert_eq!(
        unknown.task_status,
        VideoTaskStatus::Unknown("CANCELED".to_owned())
    );
    assert!(!unknown.task_status.is_terminal());
}

#[test]
fn video_task_without_an_id_is_an_llm_error() {
    let error = parse_video_task(&json!({"output": {"task_status": "PENDING"}}))
        .expect_err("there is no task id");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("no task id"));
}

#[test]
fn video_task_status_round_trips_through_serde() {
    let statuses = [
        VideoTaskStatus::Pending,
        VideoTaskStatus::Running,
        VideoTaskStatus::Succeeded,
        VideoTaskStatus::Failed,
        VideoTaskStatus::Unknown("CANCELED".to_owned()),
    ];
    for status in statuses {
        let json = serde_json::to_value(&status).expect("the status serializes");
        let restored: VideoTaskStatus =
            serde_json::from_value(json.clone()).expect("the status deserializes");
        assert_eq!(restored, status, "json: {json}");
    }
    let task = VideoTask {
        task_id: "t1".to_owned(),
        task_status: VideoTaskStatus::Failed,
        code: Some("InvalidParameter".to_owned()),
        message: None,
    };
    let restored: VideoTask =
        serde_json::from_value(serde_json::to_value(&task).expect("serializes"))
            .expect("deserializes");
    assert_eq!(restored, task);
}

#[test]
fn video_response_reads_the_url_and_the_usage_report() {
    let request = VideoRequest::new("wan2.6-t2v")
        .with_duration(5)
        .with_size("1280*720");
    let payload = json!({
        "output": {"task_id": "t1", "task_status": "SUCCEEDED", "video_url": "https://cdn.example/v.mp4"},
        "usage": {"duration": 6.5, "size": "1920*1080"}
    });

    let parsed = parse_video_response(&payload, &request).expect("there is a video");
    assert_eq!(parsed.video_url, "https://cdn.example/v.mp4");
    assert_eq!(parsed.duration, Some(6.5));
    assert_eq!(parsed.resolution.as_deref(), Some("1920*1080"));
    assert_eq!(parsed.format.as_deref(), Some("mp4"));
    assert_eq!(parsed.video_data, None);
}

#[test]
fn video_response_accepts_the_short_url_field_and_falls_back_to_the_request() {
    let request = VideoRequest::new("wan2.6-i2v")
        .with_duration(10)
        .with_resolution("1280*720");
    let payload = json!({"output": {"task_id": "t1", "task_status": "SUCCEEDED", "url": "https://cdn.example/short.mp4"}});

    let parsed = parse_video_response(&payload, &request).expect("there is a video");
    assert_eq!(parsed.video_url, "https://cdn.example/short.mp4");
    assert_eq!(parsed.duration, Some(10.0));
    assert_eq!(parsed.resolution.as_deref(), Some("1280*720"));
}

#[test]
fn video_response_without_a_url_is_an_llm_error() {
    let error = parse_video_response(
        &json!({"output": {"task_status": "SUCCEEDED"}}),
        &VideoRequest::default(),
    )
    .expect_err("there is no url");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("no video url"));
}

// ---------------------------------------------------------------------------
// Transport: image and speech
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_image_posts_the_documented_body_and_parses_the_result() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "qwen-image-max",
            "input": {"messages": [{"role": "user", "content": [{"text": "a cat"}]}]},
            "parameters": {"size": "1664*928", "n": 1, "prompt_extend": true, "watermark": false}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"choices": [{"message": {"content": [{"image": "https://cdn.example/a.png"}]}}]}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let response = test_client(&server)
        .generate_image(&[BaseMessage::user("a cat")], ImageRequest::default())
        .await
        .expect("the call succeeds");

    assert_eq!(response.images, vec!["https://cdn.example/a.png"]);
    assert_eq!(response.images_base64, Vec::<String>::new());
    assert_eq!(response.model, "qwen-image-max");
}

#[tokio::test]
async fn generate_image_joins_a_base_with_a_trailing_slash() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"choices": [{"message": {"content": [{"image": "https://cdn.example/a.png"}]}}]}
        })))
        .mount(&server)
        .await;

    let config = DashScopeConfig::new("test-key")
        .with_api_base(format!("{}/", server.uri()))
        .with_timeout(Duration::from_secs(5));
    let client = DashScopeMediaClient::new(config).expect("the client builds");

    let response = client
        .generate_image(
            &[BaseMessage::user("a cat")],
            ImageRequest::new("qwen-image-max").with_prompt("a cat"),
        )
        .await
        .expect("the call succeeds");
    assert_eq!(response.images.len(), 1);
}

#[tokio::test]
async fn generate_speech_posts_the_documented_body_and_parses_the_audio() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "cosyvoice-v1",
            "input": {"text": "hello there", "voice": "Cherry", "language_type": "Auto"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"audio": {"url": "https://cdn.example/speech.mp3"}},
            "usage": {"duration": 2.0}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let response = test_client(&server)
        .generate_speech(
            &[BaseMessage::user("hello there")],
            SpeechRequest::default(),
        )
        .await
        .expect("the call succeeds");

    assert_eq!(
        response.audio_url.as_deref(),
        Some("https://cdn.example/speech.mp3")
    );
    assert_eq!(response.format.as_deref(), Some("mp3"));
    assert_eq!(response.duration, Some(2.0));
}

#[tokio::test]
async fn an_image_request_without_a_prompt_never_reaches_the_network() {
    // The base points at the public cloud; a request would need the network, so a
    // validation error before any I/O is what proves the call never left.
    let client =
        DashScopeMediaClient::new(DashScopeConfig::new("test-key")).expect("the client builds");
    let error = client
        .generate_image(&[], ImageRequest::default())
        .await
        .expect_err("there is no prompt");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

// ---------------------------------------------------------------------------
// Transport: video
// ---------------------------------------------------------------------------

/// Mounts the video submission mock and returns the server.
async fn mount_video_submission(server: &MockServer, task_status: &str) {
    Mock::given(method("POST"))
        .and(path(VIDEO_SYNTHESIS_PATH))
        .and(header("authorization", "Bearer test-key"))
        .and(header("x-dashscope-async", "enable"))
        .and(body_json(json!({
            "model": "wan2.6-t2v",
            "input": {"prompt": "a paper boat"},
            "parameters": {"duration": 5, "prompt_extend": true, "watermark": false}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"task_id": "task-1", "task_status": task_status}
        })))
        .expect(1)
        .mount(server)
        .await;
}

#[tokio::test]
async fn generate_video_submits_async_and_polls_until_succeeded() {
    let server = MockServer::start().await;
    mount_video_submission(&server, "PENDING").await;

    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    Mock::given(method("GET"))
        .and(path("/api/v1/tasks/task-1"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(move |_: &Request| {
            let step = counter.fetch_add(1, Ordering::SeqCst);
            match step {
                0 => ResponseTemplate::new(200).set_body_json(
                    json!({"output": {"task_id": "task-1", "task_status": "PENDING"}}),
                ),
                1 => ResponseTemplate::new(200).set_body_json(
                    json!({"output": {"task_id": "task-1", "task_status": "RUNNING"}}),
                ),
                _ => ResponseTemplate::new(200).set_body_json(json!({
                    "output": {
                        "task_id": "task-1",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example/v.mp4"
                    },
                    "usage": {"duration": 5.0, "size": "1280*720"}
                })),
            }
        })
        .mount(&server)
        .await;

    let response = test_client(&server)
        .generate_video(
            &[BaseMessage::user("a paper boat")],
            VideoRequest::new("wan2.6-t2v"),
        )
        .await
        .expect("the task succeeds");

    assert_eq!(response.video_url, "https://cdn.example/v.mp4");
    assert_eq!(response.duration, Some(5.0));
    assert_eq!(response.resolution.as_deref(), Some("1280*720"));
    assert_eq!(response.format.as_deref(), Some("mp4"));
    assert_eq!(
        polls.load(Ordering::SeqCst),
        3,
        "PENDING, RUNNING, SUCCEEDED"
    );
}

#[tokio::test]
async fn a_failed_video_task_surfaces_the_vendor_code_and_message() {
    let server = MockServer::start().await;
    mount_video_submission(&server, "PENDING").await;
    Mock::given(method("GET"))
        .and(path("/api/v1/tasks/task-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "task_id": "task-1",
                "task_status": "FAILED",
                "code": "InvalidParameter",
                "message": "audio_url is not supported for this model"
            }
        })))
        .mount(&server)
        .await;

    let error = test_client(&server)
        .generate_video(
            &[BaseMessage::user("a paper boat")],
            VideoRequest::new("wan2.6-t2v"),
        )
        .await
        .expect_err("the task failed");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("task-1"), "message: {message}");
    assert!(message.contains("InvalidParameter"), "message: {message}");
    assert!(
        message.contains("audio_url is not supported for this model"),
        "message: {message}"
    );
}

#[tokio::test]
async fn a_video_task_that_never_finishes_times_out_without_hanging() {
    let server = MockServer::start().await;
    mount_video_submission(&server, "PENDING").await;
    Mock::given(method("GET"))
        .and(path("/api/v1/tasks/task-1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"output": {"task_id": "task-1", "task_status": "PENDING"}})),
        )
        .mount(&server)
        .await;

    let config = DashScopeConfig::new("test-key")
        .with_api_base(server.uri())
        .with_poll_interval(Duration::from_millis(10))
        .with_task_timeout(Duration::from_millis(150));
    let client = DashScopeMediaClient::new(config).expect("the client builds");

    let started = Instant::now();
    let error = client
        .generate_video(
            &[BaseMessage::user("a paper boat")],
            VideoRequest::new("wan2.6-t2v"),
        )
        .await
        .expect_err("the budget elapses");
    let elapsed = started.elapsed();

    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        error.to_string().contains("did not finish"),
        "error: {error}"
    );
    assert!(elapsed < Duration::from_secs(2), "elapsed: {elapsed:?}");
}

#[tokio::test]
async fn a_poll_that_outlasts_the_budget_reports_the_task_timeout() {
    // The poll request carries the remaining budget as its own timeout, so a slow
    // vendor response can expire client-side first; that is still "the task did
    // not finish within the budget", never a bare transport timeout.
    let server = MockServer::start().await;
    mount_video_submission(&server, "PENDING").await;
    Mock::given(method("GET"))
        .and(path("/api/v1/tasks/task-1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_json(json!({"output": {"task_id": "task-1", "task_status": "PENDING"}})),
        )
        .mount(&server)
        .await;
    let config = DashScopeConfig::new("test-key")
        .with_api_base(server.uri())
        .with_poll_interval(Duration::from_millis(5))
        .with_task_timeout(Duration::from_millis(100));
    let client = DashScopeMediaClient::new(config).expect("the client builds");
    let error = client
        .generate_video(
            &[BaseMessage::user("a paper boat")],
            VideoRequest::new("wan2.6-t2v"),
        )
        .await
        .expect_err("the budget elapses");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        error.to_string().contains("did not finish"),
        "error: {error}"
    );
}
#[tokio::test]
async fn a_synchronously_succeeded_video_submission_skips_polling() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(VIDEO_SYNTHESIS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "task_id": "task-1",
                "task_status": "SUCCEEDED",
                "video_url": "https://cdn.example/v.mp4"
            }
        })))
        .mount(&server)
        .await;

    let response = test_client(&server)
        .generate_video(
            &[BaseMessage::user("a paper boat")],
            VideoRequest::new("wan2.6-t2v"),
        )
        .await
        .expect("the submission already carries the video");
    assert_eq!(response.video_url, "https://cdn.example/v.mp4");

    let requests = server
        .received_requests()
        .await
        .expect("the server recorded requests");
    assert_eq!(requests.len(), 1, "no task poll should happen");
}

// ---------------------------------------------------------------------------
// Transport: failures
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_non_2xx_response_maps_to_an_llm_error_with_the_vendor_reason() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": "InvalidParameter",
            "message": "size is invalid",
            "request_id": "req-1"
        })))
        .mount(&server)
        .await;

    let error = test_client(&server)
        .generate_image(&[BaseMessage::user("a cat")], ImageRequest::default())
        .await
        .expect_err("the request is rejected");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("HTTP 400"), "message: {message}");
    assert!(message.contains("InvalidParameter"), "message: {message}");
    assert!(message.contains("size is invalid"), "message: {message}");
}

#[tokio::test]
async fn an_unstructured_error_body_is_redacted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .respond_with(
            ResponseTemplate::new(500)
                .set_body_string("upstream exploded while handling the request of sk-test"),
        )
        .mount(&server)
        .await;

    let error = test_client(&server)
        .generate_image(&[BaseMessage::user("a cat")], ImageRequest::default())
        .await
        .expect_err("the request is rejected");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains(REDACTED), "message: {message}");
    assert!(!message.contains("upstream exploded"), "message: {message}");
    assert!(!message.contains("sk-test"), "message: {message}");
}

#[tokio::test]
async fn rejected_credentials_map_to_a_config_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "code": "InvalidApiKey",
            "message": "Invalid API-key provided."
        })))
        .mount(&server)
        .await;

    let error = test_client(&server)
        .generate_image(&[BaseMessage::user("a cat")], ImageRequest::default())
        .await
        .expect_err("the credential is rejected");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("InvalidApiKey"),
        "error: {error}"
    );
}

#[tokio::test]
async fn a_vendor_error_inside_a_200_body_is_an_llm_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "code": "InvalidParameter",
            "message": "prompt must not be empty",
            "request_id": "req-2"
        })))
        .mount(&server)
        .await;

    let error = test_client(&server)
        .generate_speech(&[BaseMessage::user("hello")], SpeechRequest::default())
        .await
        .expect_err("the vendor reports an error");

    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(
        error.to_string().contains("prompt must not be empty"),
        "error: {error}"
    );
}

#[tokio::test]
async fn an_unparsable_200_body_is_an_llm_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(MULTIMODAL_GENERATION_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let error = test_client(&server)
        .generate_image(&[BaseMessage::user("a cat")], ImageRequest::default())
        .await
        .expect_err("the body is not JSON");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("unparsable"), "error: {error}");
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn an_empty_api_key_is_rejected_at_construction() {
    let error =
        DashScopeMediaClient::new(DashScopeConfig::default()).expect_err("the key is empty");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("API key"));
}

#[test]
fn a_blank_api_base_is_rejected_at_construction() {
    let error = DashScopeMediaClient::new(DashScopeConfig::new("test-key").with_api_base("   "))
        .expect_err("the base is blank");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("api_base"));
}

#[test]
fn a_zero_poll_interval_is_rejected_at_construction() {
    let error = DashScopeMediaClient::new(
        DashScopeConfig::new("test-key").with_poll_interval(Duration::ZERO),
    )
    .expect_err("a busy loop is not acceptable");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn with_http_client_keeps_the_configuration() {
    let http = HttpClient::with_defaults().expect("the HTTP stack builds");
    let config = DashScopeConfig::new("test-key").with_api_base("https://example.test");
    let client =
        DashScopeMediaClient::with_http_client(config.clone(), http).expect("the client builds");
    assert_eq!(client.config(), &config);
    assert_eq!(client.config().api_key.expose_secret(), "test-key");
}
