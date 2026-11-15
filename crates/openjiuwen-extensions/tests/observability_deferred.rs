//! Crate-boundary suite for the observability deferral seam.
//!
//! Python's `openjiuwen/extensions/observability/` also ships a process-global SDK runtime,
//! process-wide HTTP instrumentation and a trajectory data plane.
//! This crate does not carry them, so `openjiuwen_extensions::observability::deferred` keeps their
//! Python-shaped entry points and answers `ErrorKind::Unsupported` with a reason that names both
//! the missing capability and the supported alternative.
//! Every assertion goes through the re-exported public API, so a dropped re-export or a vague
//! reason fails here first.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::config::env::OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_extensions::observability::{
    acquire_observability_demand, emit_compaction_window_commit, emit_context_window_commit,
    ensure_global_http_instrumentation, force_flush_provider, init_observability,
    instrumentation_env_flag, is_initialized, publish_span_snapshot, publish_stream_frame,
    register_span_consumer, release_observability_demand,
    reset_global_http_instrumentation_for_tests, shutdown_observability, tracer,
    ObservabilityDemand, INSTRUMENTATION_DEFERRED_REASON, PROVIDER_LIFECYCLE_DEFERRED_REASON,
    TRAJECTORY_DATA_PLANE_DEFERRED_REASON,
};
use openjiuwen_extensions::otel::OtelTracerConfig;
use serde_json::json;

/// Asserts `error` is a typed deferral whose text names the capability and the alternative.
fn assert_deferred(error: &Error, capability: &str, alternative: &[&str]) -> String {
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(
        message.starts_with("unsupported: "),
        "message was {message}"
    );
    assert!(
        message.contains(capability),
        "message must name `{capability}`: {message}"
    );
    for needle in alternative {
        assert!(
            message.contains(needle),
            "message must name `{needle}`: {message}"
        );
    }
    // Shared tail: no deferral may read as a silent no-op.
    assert!(
        message.contains("nothing was initialized, exported, patched or replaced"),
        "message was {message}"
    );
    message
}

// Port of openjiuwen/extensions/observability/setup.py (`init_observability`) and runtime.py.
#[test]
fn init_observability_is_deferred_with_the_tracer_handle_alternative() {
    let config = OtelTracerConfig::default();
    let error = init_observability(&config).expect_err("the global runtime is deferred");
    assert_deferred(
        &error,
        "process-global observability runtime",
        &[
            "setup.py",
            "TracerProvider",
            "TracerHandle",
            "init_otel_tracer",
        ],
    );
}

// Port of openjiuwen/extensions/observability/setup.py (`shutdown_observability`).
#[test]
fn shutdown_observability_is_deferred() {
    let error = shutdown_observability().expect_err("the global runtime is deferred");
    assert_deferred(
        &error,
        "process-global observability runtime",
        &["runtime.py", "shutdown_with_timeout"],
    );
}

// Port of openjiuwen/extensions/observability/setup.py (`force_flush_provider`).
#[test]
fn force_flush_provider_is_deferred() {
    let error = force_flush_provider().expect_err("the global provider is deferred");
    assert_deferred(
        &error,
        "process-global observability runtime",
        &["force_flush", "init_otel_tracer"],
    );
}

// Port of openjiuwen/extensions/observability/setup.py (`is_initialized`).
#[test]
fn is_initialized_is_honestly_false_and_not_an_error() {
    // The accessor is infallible: no `Result` to unwrap, no error to hide the invariant.
    let initialized: bool = is_initialized();
    assert!(!initialized);
    assert!(!is_initialized(), "the value is invariant, not latched");
}

// Port of openjiuwen/extensions/observability/setup.py (`get_tracer`).
#[test]
fn tracer_lookup_is_deferred() {
    let error = tracer("openjiuwen.agent").expect_err("no global provider exists");
    assert_deferred(
        &error,
        "process-global observability runtime",
        &["setup.py", "TracerHandle", "init_otel_tracer"],
    );
}

// Port of openjiuwen/extensions/observability/demand.py (`acquire_observability_demand`).
#[test]
fn acquire_observability_demand_is_deferred_for_both_runtimes() {
    for runtime in ObservabilityDemand::ALL {
        let error =
            acquire_observability_demand(runtime).expect_err("demand bookkeeping is deferred");
        let message = assert_deferred(
            &error,
            "process-global observability runtime",
            &["demand.py", "reference-counted", "TracerHandle"],
        );
        assert!(message.contains("acquire/release"), "message was {message}");
    }
}

// Port of openjiuwen/extensions/observability/demand.py (`release_observability_demand`).
#[test]
fn release_observability_demand_is_deferred_for_both_runtimes() {
    for runtime in ObservabilityDemand::ALL {
        let error =
            release_observability_demand(runtime).expect_err("demand bookkeeping is deferred");
        assert_deferred(
            &error,
            "process-global observability runtime",
            &["demand.py", "TracerHandle"],
        );
    }
}

// Port of openjiuwen/extensions/observability/instrumentation.py (`ENV_FLAG`).
#[test]
fn instrumentation_env_flag_is_the_exact_python_literal() {
    assert_eq!(
        instrumentation_env_flag(),
        "OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE"
    );
    // The same literal the live resolver reads, so the seam cannot drift from the switch.
    assert_eq!(
        instrumentation_env_flag(),
        OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE
    );
}

// Port of openjiuwen/extensions/observability/instrumentation.py
// (`ensure_global_http_instrumentation`).
#[test]
fn ensure_global_http_instrumentation_is_deferred() {
    let error = ensure_global_http_instrumentation().expect_err("HTTP instrumentation is deferred");
    assert_deferred(
        &error,
        "process-wide HTTP instrumentation",
        &[
            "instrumentation.py",
            "httpx",
            "fastapi",
            "hook surface",
            "global_instrument_enabled",
        ],
    );
}

// Port of openjiuwen/extensions/observability/instrumentation.py
// (`reset_global_http_instrumentation_for_tests`).
#[test]
fn reset_global_http_instrumentation_for_tests_is_deferred() {
    let error = reset_global_http_instrumentation_for_tests()
        .expect_err("there is no instrumentation latch to reset");
    assert_deferred(
        &error,
        "process-wide HTTP instrumentation",
        &["hook surface", "traceparent"],
    );
}

// Port of openjiuwen/extensions/observability/span_record_processor.py
// (`SpanRecordProcessor.register_consumer`).
#[test]
fn register_span_consumer_is_deferred() {
    let error = register_span_consumer("trajectory-writer").expect_err("no consumer registry");
    assert_deferred(
        &error,
        "trajectory data plane",
        &[
            "span_record_processor.py",
            "ReadableSpan",
            "project_semconv",
        ],
    );
}

// Port of openjiuwen/extensions/observability/demand.py (`publish_span_snapshot`).
#[test]
fn publish_span_snapshot_is_deferred() {
    let error = publish_span_snapshot("status_update", &json!({"span": "root"}))
        .expect_err("no span-record fan-out");
    assert_deferred(
        &error,
        "trajectory data plane",
        &["content_addressing", "OtelTraceSink"],
    );
}

// Port of openjiuwen/extensions/observability/demand.py (`publish_stream_frame`).
#[test]
fn publish_stream_frame_is_deferred() {
    let error = publish_stream_frame(&json!({"kind": "delta", "text": "hi"}))
        .expect_err("no frame fan-out");
    assert_deferred(
        &error,
        "trajectory data plane",
        &["model-stream frames", "content_addressing"],
    );
}

// Port of openjiuwen/extensions/observability/trajectory_events.py
// (`emit_context_window_commit`).
#[test]
fn emit_context_window_commit_is_deferred() {
    let error = emit_context_window_commit("conversation", &json!([{"role": "user"}]))
        .expect_err("no trajectory data plane");
    assert_deferred(
        &error,
        "trajectory data plane",
        &["trajectory_events.py", "canonical context window"],
    );
}

// Port of openjiuwen/extensions/observability/trajectory_events.py
// (`emit_compaction_window_commit`).
#[test]
fn emit_compaction_window_commit_is_deferred() {
    let error = emit_compaction_window_commit("op-1", &json!([{"role": "system"}]))
        .expect_err("no trajectory data plane");
    assert_deferred(
        &error,
        "trajectory data plane",
        &["context_compression_handler.py", "ReadableSpan"],
    );
}

// Port of openjiuwen/extensions/observability/demand.py (`_RUNTIMES`).
#[test]
fn observability_demand_round_trips_its_wire_spellings() {
    assert_eq!(ObservabilityDemand::Agent.as_str(), "agent");
    assert_eq!(ObservabilityDemand::Team.as_str(), "team");
    assert_eq!(ObservabilityDemand::ALL.len(), 2);

    for (demand, wire) in [
        (ObservabilityDemand::Agent, json!("agent")),
        (ObservabilityDemand::Team, json!("team")),
    ] {
        // serde: the closed set is the lowercase Python key.
        assert_eq!(serde_json::to_value(demand).expect("serialize"), wire);
        let decoded: ObservabilityDemand =
            serde_json::from_value(wire.clone()).expect("deserialize");
        assert_eq!(decoded, demand);

        // Display / FromStr agree with the wire spelling.
        assert_eq!(demand.to_string(), wire.as_str().expect("string"));
        assert_eq!(
            wire.as_str()
                .expect("string")
                .parse::<ObservabilityDemand>()
                .expect("known key"),
            demand
        );
    }

    // An unknown key is a validation error, like Python's `ValueError`.
    let error = "swarm"
        .parse::<ObservabilityDemand>()
        .expect_err("closed set");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("swarm"), "{error}");
}

// Port of openjiuwen/extensions/observability/demand.py (`_RUNTIMES` closed-set comment).
#[test]
fn observability_demand_rejects_a_misspelled_runtime() {
    // A typo must not deserialize into a demand nobody would ever release.
    assert!(serde_json::from_value::<ObservabilityDemand>(json!("Agent")).is_err());
    assert!(serde_json::from_value::<ObservabilityDemand>(json!(null)).is_err());
}

// The three reason constants stay distinct, so a caller can tell the areas apart.
#[test]
fn deferral_reasons_are_distinct_and_named() {
    let reasons = [
        PROVIDER_LIFECYCLE_DEFERRED_REASON,
        INSTRUMENTATION_DEFERRED_REASON,
        TRAJECTORY_DATA_PLANE_DEFERRED_REASON,
    ];
    for (index, reason) in reasons.iter().enumerate() {
        assert_ne!(*reason, "");
        for (other_index, other) in reasons.iter().enumerate() {
            if index != other_index {
                assert_ne!(reason, other);
            }
        }
    }
    assert!(PROVIDER_LIFECYCLE_DEFERRED_REASON.contains("runtime.py"));
    assert!(INSTRUMENTATION_DEFERRED_REASON.contains("instrumentation.py"));
    assert!(TRAJECTORY_DATA_PLANE_DEFERRED_REASON.contains("span_context.py"));
}
