//! Integration tests for the public error contract seen by an *external* consumer.
//!
//! Only `openjiuwen_common::error::...` paths are used, exactly as downstream
//! workspace crates see them; port of `tests/unit_tests/core/common/test_errors.py`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::fmt;

use openjiuwen_common::error::{Error, ErrorKind, OptionExt, Result, ResultExt};

/// A foreign error used to exercise the `External` source chain.
#[derive(Debug)]
struct Cause;

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("boom")
    }
}

impl std::error::Error for Cause {}

fn assert_send_sync_static<T: Send + Sync + 'static>() {}

fn assert_copy_eq_hash<T: Copy + Eq + std::hash::Hash>() {}

// ---------------------------------------------------------------------------
// 1. Constructors -> ErrorKind
// ---------------------------------------------------------------------------

#[test]
fn every_constructor_maps_to_expected_kind_name() {
    let cases: [(Error, &str); 17] = [
        (Error::config("m"), "config"),
        (Error::validation("m"), "validation"),
        (Error::llm("m"), "llm"),
        (Error::tool("m"), "tool"),
        (Error::workflow("m"), "workflow"),
        (Error::graph("m"), "graph"),
        (Error::session("m"), "session"),
        (Error::memory("m"), "memory"),
        (Error::retrieval("m"), "retrieval"),
        (Error::sys_op("m"), "sys_op"),
        (Error::security("m"), "security"),
        (Error::mcp("m"), "mcp"),
        (Error::transport("m"), "transport"),
        (Error::timeout("m"), "timeout"),
        (Error::not_found("m"), "not_found"),
        (Error::unsupported("m"), "unsupported"),
        (Error::cancelled(), "cancelled"),
    ];
    for (error, expected) in cases {
        assert_eq!(error.kind().as_str(), expected, "kind name for {error}");
        assert_eq!(error.kind().to_string(), expected, "Display for {error:?}");
        assert_eq!(error.to_string().split(':').next().unwrap(), expected);
    }

    // `external` is the one non-`String` variant but names the same way.
    let external = Error::external(Cause);
    assert_eq!(external.kind(), ErrorKind::External);
    assert_eq!(external.kind().as_str(), "external");
    assert_eq!(external.kind().to_string(), "external");

    // Distinct constructors never collapse onto one kind.
    assert_ne!(Error::sys_op("m").kind(), ErrorKind::Tool);
    assert_ne!(Error::mcp("m").kind(), ErrorKind::Tool);
    assert_ne!(Error::not_found("m").kind(), ErrorKind::Unsupported);
    assert_ne!(Error::cancelled().kind(), ErrorKind::Timeout);
}

#[test]
fn error_kind_is_serde_and_hashable() {
    assert_eq!(
        serde_json::to_value(ErrorKind::SysOp).unwrap(),
        serde_json::json!("sys_op")
    );
    assert_eq!(
        serde_json::to_value(ErrorKind::NotFound).unwrap(),
        serde_json::json!("not_found")
    );
    assert_eq!(
        serde_json::to_value(ErrorKind::Unsupported).unwrap(),
        serde_json::json!("unsupported")
    );
    assert_eq!(
        serde_json::to_value(ErrorKind::Cancelled).unwrap(),
        serde_json::json!("cancelled")
    );
    assert_eq!(
        serde_json::to_value(ErrorKind::External).unwrap(),
        serde_json::json!("external")
    );

    // Round-trips through serde exactly.
    let kind: ErrorKind = serde_json::from_value(serde_json::json!("sys_op")).unwrap();
    assert_eq!(kind, ErrorKind::SysOp);

    // Copy + Eq + Hash: usable as a set key without cloning.
    let mut kinds = HashSet::new();
    kinds.insert(Error::config("m").kind());
    kinds.insert(Error::config("other").kind());
    kinds.insert(Error::cancelled().kind());
    assert_eq!(kinds.len(), 2);
    assert!(kinds.contains(&ErrorKind::Config));
    assert!(kinds.contains(&ErrorKind::Cancelled));

    // Copy semantics: the same value is still usable after storage.
    let copied = ErrorKind::Timeout;
    let mut set = HashSet::new();
    set.insert(copied);
    assert!(set.contains(&ErrorKind::Timeout));
    assert_eq!(copied, ErrorKind::Timeout);
}

// ---------------------------------------------------------------------------
// 2. Static contract
// ---------------------------------------------------------------------------

#[test]
fn error_is_send_sync_static() {
    assert_send_sync_static::<Error>();
    assert_send_sync_static::<ErrorKind>();
    assert_copy_eq_hash::<ErrorKind>();

    // `Result<T>` really defaults `E = Error`.
    let defaulted: Result<()> = Ok(());
    let explicit: std::result::Result<(), Error> = defaulted;
    assert!(explicit.is_ok());
}

/// The kind name is reachable through `std::error::Error` object-safe surfaces.
fn typed_error() -> Result<()> {
    Err(Error::not_found("session"))
}

#[test]
fn error_converts_into_boxed_std_error_via_question_mark() {
    fn boxed() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        typed_error()?;
        Ok(())
    }

    let boxed = boxed().unwrap_err();
    let as_std: &(dyn std::error::Error + Send + Sync) = boxed.as_ref();
    assert_eq!(as_std.to_string(), "not_found: session");

    // Downstream code can still recover the typed error and its kind.
    let recovered = boxed.downcast_ref::<Error>().expect("Error is preserved");
    assert_eq!(recovered.kind(), ErrorKind::NotFound);
    assert_eq!(recovered.message(), "session");
}

#[test]
fn external_source_is_reachable_at_one_and_two_levels() {
    let error = Error::external(Cause);
    let source = <Error as std::error::Error>::source(&error).expect("external keeps its source");
    assert_eq!(source.to_string(), "boom");
    assert!(std::error::Error::source(source).is_none(), "no deep chain");

    let contextual = Error::external(Cause).with_context("opening file");
    assert_eq!(contextual.kind(), ErrorKind::External);
    assert_eq!(contextual.to_string(), "external: opening file: boom");
    assert_eq!(contextual.message(), "opening file: boom");

    let level_one =
        <Error as std::error::Error>::source(&contextual).expect("context wrapper is the source");
    assert_eq!(level_one.to_string(), "opening file: boom");
    let level_two =
        std::error::Error::source(level_one).expect("the foreign cause stays reachable");
    assert_eq!(level_two.to_string(), "boom");
    assert!(std::error::Error::source(level_two).is_none());

    // Context stacks on top of an already-contextualized external error.
    let twice = contextual.with_context("loading config");
    assert_eq!(
        twice.to_string(),
        "external: loading config: opening file: boom"
    );
}

// ---------------------------------------------------------------------------
// 3. Display and message
// ---------------------------------------------------------------------------

#[test]
fn display_is_kind_colon_message() {
    assert_eq!(
        Error::validation("bad input").to_string(),
        "validation: bad input"
    );
    assert_eq!(Error::config("no key").to_string(), "config: no key");
    assert_eq!(Error::sys_op("rm failed").to_string(), "sys_op: rm failed");
    assert_eq!(
        Error::not_found("skill").to_string(),
        "not_found: skill".to_owned()
    );
    assert_eq!(Error::cancelled().to_string(), "cancelled");
    assert_eq!(Error::external(Cause).to_string(), "external: boom");
}

#[test]
fn message_returns_the_bare_payload() {
    assert_eq!(Error::validation("bad input").message(), "bad input");
    assert_eq!(Error::mcp("handshake failed").message(), "handshake failed");
    assert_eq!(Error::cancelled().message(), "cancelled");
    assert_eq!(Error::external(Cause).message(), "boom");

    // The payload is not prefixed with the kind.
    assert!(!Error::tool("open").message().contains("tool:"));
}

// ---------------------------------------------------------------------------
// 4. ResultExt / OptionExt / with_context
// ---------------------------------------------------------------------------

fn parse_u8(text: &str) -> std::result::Result<u8, Error> {
    text.parse::<u8>()
        .map_err(|error| Error::validation(error.to_string()))
}

#[test]
fn result_ext_context_prefixes_and_preserves_kind() {
    let ok: std::result::Result<u8, Error> = Ok(7);
    assert_eq!(ok.context("parsing").unwrap(), 7);

    let error = parse_u8("nope").context("parsing retry limit").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error
        .to_string()
        .starts_with("validation: parsing retry limit: "));

    // Kind is preserved, not reset to a generic error.
    let not_found = Err::<u8, _>(Error::not_found("session"))
        .context("loading agent")
        .unwrap_err();
    assert_eq!(not_found.kind(), ErrorKind::NotFound);
    assert_eq!(not_found.to_string(), "not_found: loading agent: session");
}

#[test]
fn result_ext_context_converts_foreign_errors() {
    let io_result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        "denied",
    ));
    let error = io_result.context("reading config").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
    assert_eq!(error.message(), "reading config: denied");
    assert!(<Error as std::error::Error>::source(&error).is_some());
}

#[test]
fn with_context_leaves_cancelled_untouched() {
    let error = Error::cancelled().with_context("running agent");
    assert!(error.is_cancelled());
    assert_eq!(error.to_string(), "cancelled");
    assert_eq!(error.message(), "cancelled");
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    // String variants do gain the prefix.
    let annotated = Error::timeout("5s").with_context("calling model");
    assert_eq!(annotated.to_string(), "timeout: calling model: 5s");
    assert_eq!(annotated.kind(), ErrorKind::Timeout);
}

/// A `?`-based consumer of [`OptionExt::ok_or_validation`].
fn require_model_name(config: Option<String>) -> Result<usize> {
    let value = config.ok_or_validation("model name")?;
    Ok(value.len())
}

#[test]
fn option_ext_helpers_name_the_missing_thing() {
    let present: Option<u8> = Some(3);
    assert_eq!(present.ok_or_not_found("session").unwrap(), 3);
    assert_eq!(present.ok_or_validation("api key").unwrap(), 3);

    let missing: Option<u8> = None;
    let error = missing.ok_or_not_found("session").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(error.to_string(), "not_found: session not found");

    let error = missing.ok_or_validation("api key").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: api key is required");

    // Both are usable directly in a `?`-based consumer function.
    assert_eq!(require_model_name(Some("gpt-4o".to_owned())).unwrap(), 6);
    assert_eq!(
        require_model_name(None).unwrap_err().to_string(),
        "validation: model name is required"
    );
}

// ---------------------------------------------------------------------------
// 5. Conversions and classification
// ---------------------------------------------------------------------------

#[test]
fn foreign_errors_convert_with_documented_kinds() {
    let serde_error: Error = serde_json::from_str::<serde_json::Value>("{")
        .unwrap_err()
        .into();
    assert_eq!(serde_error.kind(), ErrorKind::Validation);
    assert!(!serde_error.message().is_empty());

    let io_error: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "missing").into();
    assert_eq!(io_error.kind(), ErrorKind::External);
    assert_eq!(io_error.message(), "missing");
    assert!(<Error as std::error::Error>::source(&io_error).is_some());

    let url_error: Error = url::Url::parse("not a url").unwrap_err().into();
    assert_eq!(url_error.kind(), ErrorKind::Validation);
    assert!(!url_error.message().is_empty());

    let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(Cause);
    let boxed_error: Error = boxed.into();
    assert_eq!(boxed_error.kind(), ErrorKind::External);
    assert_eq!(boxed_error.message(), "boom");
    assert_eq!(boxed_error.to_string(), "external: boom");
}

#[test]
fn retryable_is_transport_or_timeout_only() {
    assert!(Error::transport("connection reset").is_retryable());
    assert!(Error::timeout("deadline exceeded").is_retryable());

    let non_retryable = [
        Error::config("m"),
        Error::validation("m"),
        Error::llm("m"),
        Error::tool("m"),
        Error::workflow("m"),
        Error::graph("m"),
        Error::session("m"),
        Error::memory("m"),
        Error::retrieval("m"),
        Error::sys_op("m"),
        Error::security("m"),
        Error::mcp("m"),
        Error::not_found("m"),
        Error::unsupported("m"),
        Error::cancelled(),
        Error::external(Cause),
    ];
    for error in non_retryable {
        assert!(!error.is_retryable(), "unexpectedly retryable: {error:?}");
    }
}

#[test]
fn is_cancelled_is_only_the_cancelled_variant() {
    assert!(Error::cancelled().is_cancelled());
    assert!(Error::cancelled().with_context("ignored").is_cancelled());

    let others = [
        Error::timeout("deadline"),
        Error::transport("reset"),
        Error::config("m"),
        Error::not_found("m"),
        Error::unsupported("m"),
        Error::external(Cause),
    ];
    for error in others {
        assert!(!error.is_cancelled(), "unexpectedly cancelled: {error:?}");
    }
}

// ---------------------------------------------------------------------------
// 6. Consumer-style function
// ---------------------------------------------------------------------------

/// Contract check for `docs/ARCHITECTURE.md` §2.1: a downstream crate returns
/// `openjiuwen_common::error::Result<T>` and relies on `?` for foreign errors.
fn summarize_report(
    document: &str,
    sink: std::result::Result<(), std::io::Error>,
) -> Result<usize> {
    let value: serde_json::Value = serde_json::from_str(document)?;
    sink?;
    value
        .as_object()
        .map(serde_json::Map::len)
        .ok_or_validation("report object")
}

#[test]
fn consumer_style_function_maps_foreign_errors_to_kinds() {
    assert_eq!(summarize_report(r#"{"a": 1, "b": 2}"#, Ok(())).unwrap(), 2);

    let malformed = summarize_report("{", Ok(())).unwrap_err();
    assert_eq!(malformed.kind(), ErrorKind::Validation);

    let io_failure = summarize_report(
        "{}",
        Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "pipe closed",
        )),
    )
    .unwrap_err();
    assert_eq!(io_failure.kind(), ErrorKind::External);
    assert_eq!(io_failure.message(), "pipe closed");
    assert!(<Error as std::error::Error>::source(&io_failure).is_some());

    let not_an_object = summarize_report("[1, 2]", Ok(())).unwrap_err();
    assert_eq!(not_an_object.kind(), ErrorKind::Validation);
    assert_eq!(
        not_an_object.to_string(),
        "validation: report object is required"
    );

    // The whole path is also usable through `Box<dyn Error + Send + Sync>`.
    let boxed =
        |document: &str| -> std::result::Result<usize, Box<dyn std::error::Error + Send + Sync>> {
            Ok(summarize_report(document, Ok(()))?)
        };
    assert_eq!(boxed(r#"{"a": 1}"#).unwrap(), 1);
    assert_eq!(
        boxed("]").unwrap_err().to_string(),
        summarize_report("]", Ok(())).unwrap_err().to_string()
    );
}
