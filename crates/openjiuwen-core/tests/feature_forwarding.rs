//! Proves the umbrella's feature forwarders reach the concrete crates' gated surfaces.
//!
//! `openjiuwen-core` declares four forwarding features (`otel`, `sqlite`, `sandbox`,
//! `live-tests`). Each one is only meaningful if the feature-gated items behind it
//! really become reachable through the Python-shaped path, so this suite checks both
//! the *identity* of the forwarded items (`TypeId` against the concrete crate) and,
//! where a configuration change alters behaviour, the behaviour itself.
//!
//! Gate matrix:
//!
//! - `cargo test -p openjiuwen-core` — no forwarded feature: the baseline aliases and
//!   the always-compiled card surfaces resolve, and the `otel` helper is the
//!   documented no-op.
//! - `cargo test -p openjiuwen-core --all-features` — every forwarded feature on:
//!   `sqlite`, `sandbox`, and `otel` surfaces are reachable and `otel` really
//!   installs the sink. (`live-tests` is compile-checked here; it adds only the
//!   concrete crate's opt-in network tests, so this file adds no network test.)
//! - `cargo test -p openjiuwen-core --no-default-features` — same as the first gate,
//!   because the default feature set is empty.
//!
//! Everything here is offline: the sandbox gateway's remote call performs no I/O by
//! contract, the trace sink is in-process, and no SQLite file is opened.

use std::any::TypeId;

/// Asserts at compile time that the umbrella path and the concrete-crate item are
/// the same type.
fn assert_same_type<T: ?Sized + 'static, U: ?Sized + 'static>() {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
}

/// Drives an immediately-ready future to completion without a runtime.
///
/// The only async call used here (`HttpSandboxGateway::invoke`) has no `.await`
/// points — it returns its `Unsupported` error at once — so a single poll with a
/// no-op waker is enough. Polling is bounded to one turn: a future that yields panics
/// instead of silently returning nothing.
#[cfg(feature = "sandbox")]
fn poll_once<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Waker};

    let mut context = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the future yielded; this helper drives one poll only"),
    }
}

#[test]
fn baseline_aliases_resolve_in_every_configuration() {
    assert_eq!(
        openjiuwen_core::common::CRATE_NAME,
        "openjiuwen-common",
        "the common alias forwards the concrete crate with no feature enabled"
    );
    assert_same_type::<
        openjiuwen_core::foundation::prompt::BaseMessage,
        openjiuwen_foundation::prompt::BaseMessage,
    >();
    assert_same_type::<openjiuwen_core::session::SessionConfig, openjiuwen_session::SessionConfig>(
    );
}

/// `StoreRegistry` is part of the card shape, not of the `sqlite` capability, so it
/// must still resolve when the sqlite forwarder is off.
#[cfg(not(feature = "sqlite"))]
#[test]
fn reduced_configuration_keeps_the_always_compiled_store_surface() {
    assert_same_type::<
        openjiuwen_core::foundation::store::StoreRegistry,
        openjiuwen_foundation::store::StoreRegistry,
    >();
}

/// The sandbox gateway *configuration* always compiles (it is serialized on the
/// `SysOperationCard`); only the HTTP client behind `sandbox` is gated.
#[cfg(not(feature = "sandbox"))]
#[test]
fn reduced_configuration_keeps_the_always_compiled_sandbox_config_surface() {
    assert_same_type::<
        openjiuwen_core::sys_operation::sandbox::config::SandboxGatewayConfig,
        openjiuwen_sysop::sandbox::config::SandboxGatewayConfig,
    >();
}

/// With `sqlite` forwarded, foundation's SQLite stores, session's SQLite
/// checkpointer, and memory's FTS5 coding index all resolve through the umbrella.
#[cfg(feature = "sqlite")]
#[test]
fn sqlite_feature_forwarding_installs_sqlite_surfaces() {
    use openjiuwen_core::foundation::store::{SqliteDocumentStore, SqliteKvStore};
    use openjiuwen_core::memory::lite::fts::{FtsHit, FtsIndex};
    use openjiuwen_core::session::SqliteCheckpointer;

    assert_same_type::<SqliteKvStore, openjiuwen_foundation::store::SqliteKvStore>();
    assert_same_type::<SqliteDocumentStore, openjiuwen_foundation::store::SqliteDocumentStore>();

    // Root re-export and the nested module path reach the same item.
    assert_same_type::<SqliteCheckpointer, openjiuwen_session::SqliteCheckpointer>();
    assert_same_type::<SqliteCheckpointer, openjiuwen_session::checkpoint::SqliteCheckpointer>();

    assert_same_type::<FtsIndex, openjiuwen_memory::lite::fts::FtsIndex>();
    assert_same_type::<FtsHit, openjiuwen_memory::lite::fts::FtsHit>();
}

/// With `sandbox` forwarded the HTTP gateway compiles, and its remote call keeps the
/// documented `Unsupported` contract (no network I/O is performed).
#[cfg(feature = "sandbox")]
#[test]
fn sandbox_feature_forwarding_exposes_gateway() {
    use openjiuwen_core::common::error::Error;
    use openjiuwen_core::sys_operation::sandbox::config::{
        GatewayInvokeRequest, SandboxEndpoint, SandboxGatewayConfig,
    };
    use openjiuwen_core::sys_operation::sandbox::http::HttpSandboxGateway;

    assert_same_type::<HttpSandboxGateway, openjiuwen_sysop::sandbox::http::HttpSandboxGateway>();

    // `new` only builds an HTTP client; it sends nothing.
    let Ok(gateway) =
        HttpSandboxGateway::new(SandboxEndpoint::default(), SandboxGatewayConfig::default())
    else {
        panic!("building a gateway client is offline and must not fail");
    };
    assert_eq!(gateway.endpoint().base_url, "");

    let request = GatewayInvokeRequest {
        op_type: "fs".to_string(),
        method: "read_file".to_string(),
        params: serde_json::Value::Null,
        isolation_key: None,
    };
    let Err(error) = poll_once(gateway.invoke(request)) else {
        panic!("the AIO gateway protocol is deferred; invoke must fail");
    };
    assert!(
        matches!(error, Error::Unsupported(_)),
        "expected Unsupported, got {error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("sandbox gateway protocol is not implemented"),
        "unexpected message: {error}"
    );
}

/// With `otel` forwarded, `session::install_trace_sink` really installs the sink: a
/// span started afterwards is recorded by it.
#[cfg(feature = "otel")]
#[test]
fn otel_feature_forwarding_installs_trace_sink() {
    use std::sync::Arc;

    use openjiuwen_core::session::{
        install_trace_sink, InvokeType, MemoryTraceSink, Session, WorkflowSession,
    };

    let session = WorkflowSession::new("feature-forwarding-otel");
    let sink = Arc::new(MemoryTraceSink::new());
    if let Err(error) = install_trace_sink(&session, sink.clone()) {
        panic!("a workflow session accepts a trace sink: {error}");
    }
    assert!(
        sink.is_empty(),
        "installing a sink records nothing by itself"
    );

    let Some(tracer) = session.tracer() else {
        panic!("a workflow session owns a tracer");
    };
    let span = tracer.start_span("forwarded", InvokeType::Prompt);
    span.finish(serde_json::Value::Null);

    assert_eq!(
        sink.len(),
        1,
        "the forwarded feature makes the sink observe the span"
    );
}

/// Without `otel`, the same helper is the documented no-op: it still succeeds, but the
/// session keeps its own sink, so nothing is recorded.
#[cfg(not(feature = "otel"))]
#[test]
fn otel_feature_forwarding_is_a_documented_noop_without_the_feature() {
    use std::sync::Arc;

    use openjiuwen_core::session::{
        install_trace_sink, InvokeType, MemoryTraceSink, Session, WorkflowSession,
    };

    let session = WorkflowSession::new("feature-forwarding-no-otel");
    let sink = Arc::new(MemoryTraceSink::new());
    if let Err(error) = install_trace_sink(&session, sink.clone()) {
        panic!("the feature-disabled helper never fails: {error}");
    }

    let Some(tracer) = session.tracer() else {
        panic!("a workflow session owns a tracer");
    };
    let span = tracer.start_span("not-forwarded", InvokeType::Prompt);
    span.finish(serde_json::Value::Null);

    assert!(
        sink.is_empty(),
        "without the forwarded `otel` feature the helper must not touch the session"
    );
}
