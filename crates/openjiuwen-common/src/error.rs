//! The shared [`Error`] type, [`ErrorKind`] discriminant, [`Result`] alias, and
//! error-context helpers used across every openJiuwen crate.
//!
//! Python's exception class hierarchy is replaced by one enum plus [`ErrorKind`]:
//! the enum carries the payload, the kind identifies the category for programmatic
//! handling (matching on the kind is the analogue of Python's `except SomeError`).
//!
//! ```
//! use openjiuwen_common::error::{Error, ErrorKind, ResultExt};
//!
//! let err = "not a number"
//!     .parse::<u32>()
//!     .map_err(|e| Error::validation(e.to_string()))
//!     .context("reading retry limit")
//!     .unwrap_err();
//! assert_eq!(err.kind(), ErrorKind::Validation);
//! assert!(err.to_string().starts_with("validation: reading retry limit: "));
//! ```

use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

/// The crate-wide result alias; `E` defaults to [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Payload-free category of an [`Error`].
///
/// This is the discriminant of [`Error`] and is `Copy + Eq + Hash`, so it can be
/// matched, stored, and used as a map key without cloning the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[must_use]
pub enum ErrorKind {
    /// Invalid configuration or configuration-derived value.
    Config,
    /// A value failed validation (bad input, malformed document).
    Validation,
    /// LLM provider or model failure.
    Llm,
    /// Tool invocation failure.
    Tool,
    /// Workflow definition or execution failure.
    Workflow,
    /// Graph execution failure.
    Graph,
    /// Session or checkpoint failure.
    Session,
    /// Memory subsystem failure.
    Memory,
    /// Retrieval, embedding, or vector-store failure.
    Retrieval,
    /// System operation (filesystem, shell, code execution) failure.
    SysOp,
    /// Security, guardrail, or permission failure.
    Security,
    /// MCP client/server failure.
    Mcp,
    /// Transport-level failure (connection, request encoding).
    Transport,
    /// An operation exceeded its deadline.
    Timeout,
    /// A requested entity does not exist.
    NotFound,
    /// The operation is not supported by this implementation.
    Unsupported,
    /// The operation was cancelled.
    Cancelled,
    /// Any other error, usually wrapped from a foreign crate.
    External,
}

impl ErrorKind {
    /// The `snake_case` name used in [`Display`] and serde output, Python-style.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Validation => "validation",
            Self::Llm => "llm",
            Self::Tool => "tool",
            Self::Workflow => "workflow",
            Self::Graph => "graph",
            Self::Session => "session",
            Self::Memory => "memory",
            Self::Retrieval => "retrieval",
            Self::SysOp => "sys_op",
            Self::Security => "security",
            Self::Mcp => "mcp",
            Self::Transport => "transport",
            Self::Timeout => "timeout",
            Self::NotFound => "not_found",
            Self::Unsupported => "unsupported",
            Self::Cancelled => "cancelled",
            Self::External => "external",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The shared error type of every openJiuwen library crate.
///
/// String variants render as `"<kind>: <message>"`; [`Error::Cancelled`] renders as
/// `"cancelled"`; [`Error::External`] renders as `"external: <source>"` and keeps the
/// cause reachable through [`std::error::Error::source`].
#[derive(Debug, thiserror::Error)]
#[must_use]
pub enum Error {
    /// Invalid configuration or a configuration-derived value.
    #[error("config: {0}")]
    Config(String),
    /// A value failed validation (bad user input, malformed document).
    #[error("validation: {0}")]
    Validation(String),
    /// LLM provider or model failure.
    #[error("llm: {0}")]
    Llm(String),
    /// Tool invocation failure.
    #[error("tool: {0}")]
    Tool(String),
    /// Workflow definition or execution failure.
    #[error("workflow: {0}")]
    Workflow(String),
    /// Graph execution failure.
    #[error("graph: {0}")]
    Graph(String),
    /// Session or checkpoint failure.
    #[error("session: {0}")]
    Session(String),
    /// Memory subsystem failure.
    #[error("memory: {0}")]
    Memory(String),
    /// Retrieval, embedding, or vector-store failure.
    #[error("retrieval: {0}")]
    Retrieval(String),
    /// System operation (filesystem, shell, code execution) failure.
    #[error("sys_op: {0}")]
    SysOp(String),
    /// Security, guardrail, or permission failure.
    #[error("security: {0}")]
    Security(String),
    /// MCP client/server failure.
    #[error("mcp: {0}")]
    Mcp(String),
    /// Transport-level failure (connection, request encoding).
    #[error("transport: {0}")]
    Transport(String),
    /// An operation exceeded its deadline.
    #[error("timeout: {0}")]
    Timeout(String),
    /// A requested entity does not exist.
    #[error("not_found: {0}")]
    NotFound(String),
    /// The operation is not supported by this implementation.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The operation was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// A foreign error, kept whole so [`std::error::Error::source`] keeps working.
    #[error("external: {0}")]
    External(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Invalid configuration message.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    /// Validation failure message.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }

    /// LLM failure message.
    pub fn llm(message: impl Into<String>) -> Self {
        Self::Llm(message.into())
    }

    /// Tool failure message.
    pub fn tool(message: impl Into<String>) -> Self {
        Self::Tool(message.into())
    }

    /// Workflow failure message.
    pub fn workflow(message: impl Into<String>) -> Self {
        Self::Workflow(message.into())
    }

    /// Graph failure message.
    pub fn graph(message: impl Into<String>) -> Self {
        Self::Graph(message.into())
    }

    /// Session failure message.
    pub fn session(message: impl Into<String>) -> Self {
        Self::Session(message.into())
    }

    /// Memory failure message.
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory(message.into())
    }

    /// Retrieval failure message.
    pub fn retrieval(message: impl Into<String>) -> Self {
        Self::Retrieval(message.into())
    }

    /// System operation failure message.
    pub fn sys_op(message: impl Into<String>) -> Self {
        Self::SysOp(message.into())
    }

    /// Security failure message.
    pub fn security(message: impl Into<String>) -> Self {
        Self::Security(message.into())
    }

    /// MCP failure message.
    pub fn mcp(message: impl Into<String>) -> Self {
        Self::Mcp(message.into())
    }

    /// Transport failure message.
    pub fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }

    /// Timeout failure message.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::Timeout(message.into())
    }

    /// Missing-entity message.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound(message.into())
    }

    /// Unsupported-operation message.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    /// A cancellation (no payload).
    pub const fn cancelled() -> Self {
        Self::Cancelled
    }

    /// Wraps a foreign error, preserving it as the [`std::error::Error::source`].
    pub fn external(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::External(Box::new(error))
    }

    /// The payload of this error: the message of [`Error::Cancelled`] is `"cancelled"`,
    /// and [`Error::External`] exposes its source's `Display`.
    #[must_use]
    pub fn message(&self) -> Cow<'_, str> {
        match self {
            Self::Config(message)
            | Self::Validation(message)
            | Self::Llm(message)
            | Self::Tool(message)
            | Self::Workflow(message)
            | Self::Graph(message)
            | Self::Session(message)
            | Self::Memory(message)
            | Self::Retrieval(message)
            | Self::SysOp(message)
            | Self::Security(message)
            | Self::Mcp(message)
            | Self::Transport(message)
            | Self::Timeout(message)
            | Self::NotFound(message)
            | Self::Unsupported(message) => Cow::Borrowed(message.as_str()),
            Self::Cancelled => Cow::Borrowed("cancelled"),
            Self::External(source) => Cow::Owned(source.to_string()),
        }
    }

    /// The category of this error.
    pub const fn kind(&self) -> ErrorKind {
        match self {
            Self::Config(_) => ErrorKind::Config,
            Self::Validation(_) => ErrorKind::Validation,
            Self::Llm(_) => ErrorKind::Llm,
            Self::Tool(_) => ErrorKind::Tool,
            Self::Workflow(_) => ErrorKind::Workflow,
            Self::Graph(_) => ErrorKind::Graph,
            Self::Session(_) => ErrorKind::Session,
            Self::Memory(_) => ErrorKind::Memory,
            Self::Retrieval(_) => ErrorKind::Retrieval,
            Self::SysOp(_) => ErrorKind::SysOp,
            Self::Security(_) => ErrorKind::Security,
            Self::Mcp(_) => ErrorKind::Mcp,
            Self::Transport(_) => ErrorKind::Transport,
            Self::Timeout(_) => ErrorKind::Timeout,
            Self::NotFound(_) => ErrorKind::NotFound,
            Self::Unsupported(_) => ErrorKind::Unsupported,
            Self::Cancelled => ErrorKind::Cancelled,
            Self::External(_) => ErrorKind::External,
        }
    }

    /// `true` only for [`Error::Cancelled`].
    ///
    /// Matches Python's cooperative cancellation checks and never treats a
    /// timeout or transport failure as cancellation.
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// `true` when retrying the operation may succeed ([`Error::Transport`],
    /// [`Error::Timeout`]).
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Transport(_) | Self::Timeout(_))
    }

    /// Prefixes the error with `ctx`, preserving its kind.
    ///
    /// String variants become `"<ctx>: <message>"`; [`Error::External`] wraps its
    /// source so that `source()` keeps chaining ([`std::error::Error::source`]);
    /// [`Error::Cancelled`] is returned unchanged.
    pub fn with_context(self, ctx: impl fmt::Display) -> Self {
        let ctx = ctx.to_string();
        match self {
            Self::Cancelled => Self::Cancelled,
            Self::External(source) => Self::External(Box::new(ContextError {
                context: ctx,
                source,
            })),
            Self::Config(message) => Self::Config(prefix(&ctx, &message)),
            Self::Validation(message) => Self::Validation(prefix(&ctx, &message)),
            Self::Llm(message) => Self::Llm(prefix(&ctx, &message)),
            Self::Tool(message) => Self::Tool(prefix(&ctx, &message)),
            Self::Workflow(message) => Self::Workflow(prefix(&ctx, &message)),
            Self::Graph(message) => Self::Graph(prefix(&ctx, &message)),
            Self::Session(message) => Self::Session(prefix(&ctx, &message)),
            Self::Memory(message) => Self::Memory(prefix(&ctx, &message)),
            Self::Retrieval(message) => Self::Retrieval(prefix(&ctx, &message)),
            Self::SysOp(message) => Self::SysOp(prefix(&ctx, &message)),
            Self::Security(message) => Self::Security(prefix(&ctx, &message)),
            Self::Mcp(message) => Self::Mcp(prefix(&ctx, &message)),
            Self::Transport(message) => Self::Transport(prefix(&ctx, &message)),
            Self::Timeout(message) => Self::Timeout(prefix(&ctx, &message)),
            Self::NotFound(message) => Self::NotFound(prefix(&ctx, &message)),
            Self::Unsupported(message) => Self::Unsupported(prefix(&ctx, &message)),
        }
    }
}

/// Prefixes `message` with `ctx`, inserting the `": "` separator.
fn prefix(ctx: &str, message: &str) -> String {
    format!("{ctx}: {message}")
}

/// A foreign error that was annotated with context.
///
/// Kept crate-private: callers only ever see it as an [`Error::External`] source.
#[derive(Debug)]
struct ContextError {
    context: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.context, self.source)
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Adds [`ResultExt::context`] to any `Result` whose error converts into [`Error`].
///
/// Port of the Python habit of re-raising with a message prefix (`raise ... from`).
pub trait ResultExt<T> {
    /// Converts the error into an [`Error`] and prefixes it with `ctx`.
    ///
    /// # Errors
    ///
    /// Returns the converted and annotated error when this result is `Err`.
    fn context<C: fmt::Display>(self, ctx: C) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: Into<Error>,
{
    fn context<C: fmt::Display>(self, ctx: C) -> Result<T> {
        self.map_err(|error| error.into().with_context(ctx))
    }
}

/// Converts an `Option` into a [`Result`] with a [`Error::NotFound`] or
/// [`Error::Validation`] error.
pub trait OptionExt<T> {
    /// Turns `None` into [`Error::NotFound`] (`"<what> not found"`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when this option is `None`.
    fn ok_or_not_found(self, what: impl fmt::Display) -> Result<T>;

    /// Turns `None` into [`Error::Validation`] (`"<what> is required"`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when this option is `None`.
    fn ok_or_validation(self, what: impl fmt::Display) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_not_found(self, what: impl fmt::Display) -> Result<T> {
        self.ok_or_else(|| Error::not_found(format!("{what} not found")))
    }

    fn ok_or_validation(self, what: impl fmt::Display) -> Result<T> {
        self.ok_or_else(|| Error::validation(format!("{what} is required")))
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::external(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::validation(error.to_string())
    }
}

impl From<url::ParseError> for Error {
    fn from(error: url::ParseError) -> Self {
        Self::validation(error.to_string())
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::timeout(error.to_string())
        } else {
            // Connection, request-encoding, decode, and status errors are all
            // transport failures (`is_connect()` / `is_request()` included).
            Self::transport(error.to_string())
        }
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(error: tokio::task::JoinError) -> Self {
        if error.is_cancelled() {
            Self::Cancelled
        } else {
            Self::external(error)
        }
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(error: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::External(error)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
    fn assert_copy_eq_hash<T: Copy + Eq + Hash>() {}

    #[test]
    fn static_assertions() {
        assert_send_sync_static::<Error>();
        assert_copy_eq_hash::<ErrorKind>();

        // `Result<()>` really defaults `E = Error`.
        let defaulted: Result<()> = Ok(());
        let explicit: std::result::Result<(), Error> = defaulted;
        assert!(explicit.is_ok(), "default error type is Error");
    }

    #[test]
    fn kind_maps_every_variant() {
        let cases = [
            (Error::config("m"), ErrorKind::Config),
            (Error::validation("m"), ErrorKind::Validation),
            (Error::llm("m"), ErrorKind::Llm),
            (Error::tool("m"), ErrorKind::Tool),
            (Error::workflow("m"), ErrorKind::Workflow),
            (Error::graph("m"), ErrorKind::Graph),
            (Error::session("m"), ErrorKind::Session),
            (Error::memory("m"), ErrorKind::Memory),
            (Error::retrieval("m"), ErrorKind::Retrieval),
            (Error::sys_op("m"), ErrorKind::SysOp),
            (Error::security("m"), ErrorKind::Security),
            (Error::mcp("m"), ErrorKind::Mcp),
            (Error::transport("m"), ErrorKind::Transport),
            (Error::timeout("m"), ErrorKind::Timeout),
            (Error::not_found("m"), ErrorKind::NotFound),
            (Error::unsupported("m"), ErrorKind::Unsupported),
            (Error::cancelled(), ErrorKind::Cancelled),
            (Error::external(Cause), ErrorKind::External),
        ];
        let kinds: HashSet<ErrorKind> = cases.iter().map(|(error, _)| error.kind()).collect();
        assert_eq!(
            kinds.len(),
            cases.len(),
            "every variant has a distinct kind"
        );
        for (error, expected) in cases {
            assert_eq!(error.kind(), expected, "kind of {error:?}");
        }
    }

    #[test]
    fn kind_serializes_as_python_style_snake_case() {
        for (kind, expected) in [
            (ErrorKind::Config, "config"),
            (ErrorKind::SysOp, "sys_op"),
            (ErrorKind::NotFound, "not_found"),
            (ErrorKind::Unsupported, "unsupported"),
            (ErrorKind::Cancelled, "cancelled"),
            (ErrorKind::External, "external"),
        ] {
            assert_eq!(kind.as_str(), expected);
            assert_eq!(kind.to_string(), expected);
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{expected}\"")
            );
        }
    }

    #[test]
    fn display_renders_kind_and_message() {
        assert_eq!(
            Error::config("missing API_KEY").to_string(),
            "config: missing API_KEY"
        );
        assert_eq!(Error::sys_op("rm failed").to_string(), "sys_op: rm failed");
        assert_eq!(Error::cancelled().to_string(), "cancelled");
        assert_eq!(Error::external(Cause).to_string(), "external: boom");
        assert_eq!(Error::not_found("session").message(), "session");
        assert_eq!(Error::cancelled().message(), "cancelled");
        assert_eq!(Error::external(Cause).message(), "boom");
    }

    #[test]
    fn external_exposes_source() {
        let error = Error::external(Cause);
        let source = std::error::Error::source(&error).expect("external keeps its source");
        assert_eq!(source.to_string(), "boom");
        assert!(std::error::Error::source(source).is_none());

        // String variants have no source.
        assert!(std::error::Error::source(&Error::config("m")).is_none());
    }

    #[test]
    fn with_context_prefixes_string_variants_and_keeps_kind() {
        let error = Error::transport("connection reset").with_context("fetching model list");
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert_eq!(
            error.to_string(),
            "transport: fetching model list: connection reset"
        );
        assert_eq!(error.message(), "fetching model list: connection reset");

        let error = Error::not_found("session").with_context("loading");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(error.to_string(), "not_found: loading: session");
    }

    #[test]
    fn with_context_wraps_external_and_chains_sources() {
        let error = Error::external(Cause).with_context("opening file");
        assert_eq!(error.kind(), ErrorKind::External);
        assert_eq!(error.to_string(), "external: opening file: boom");
        assert_eq!(error.message(), "opening file: boom");

        let source = std::error::Error::source(&error).expect("context error is the source");
        assert_eq!(source.to_string(), "opening file: boom");
        let root = std::error::Error::source(source).expect("cause is reachable");
        assert_eq!(root.to_string(), "boom");

        // Context stacks.
        let twice = error.with_context("loading config");
        assert_eq!(
            twice.to_string(),
            "external: loading config: opening file: boom"
        );
    }

    #[test]
    fn with_context_leaves_cancelled_untouched() {
        let error = Error::cancelled().with_context("running agent");
        assert!(error.is_cancelled());
        assert_eq!(error.to_string(), "cancelled");
    }

    #[test]
    fn retryable_and_cancelled_classification() {
        assert!(Error::transport("reset").is_retryable());
        assert!(Error::timeout("deadline").is_retryable());
        assert!(!Error::cancelled().is_retryable());
        assert!(!Error::config("bad").is_retryable());
        assert!(!Error::not_found("x").is_retryable());

        assert!(Error::cancelled().is_cancelled());
        assert!(!Error::timeout("deadline").is_cancelled());
        assert!(!Error::external(Cause).is_cancelled());
    }

    #[test]
    fn from_serde_json_error_is_validation() {
        let error: Error = serde_json::from_str::<serde_json::Value>("{")
            .unwrap_err()
            .into();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(!error.message().is_empty());
    }

    #[test]
    fn from_io_error_preserves_source() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let error: Error = io_error.into();
        assert_eq!(error.kind(), ErrorKind::External);
        assert_eq!(error.message(), "missing");
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn from_boxed_error_is_external() {
        let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(Cause);
        let error: Error = boxed.into();
        assert_eq!(error.kind(), ErrorKind::External);
        assert_eq!(error.message(), "boom");
    }

    #[test]
    fn result_ext_context() {
        let ok: std::result::Result<u8, Error> = Ok(7);
        assert_eq!(ok.context("parsing").unwrap(), 7);

        let error = "nope"
            .parse::<u8>()
            .map_err(|e| Error::validation(e.to_string()))
            .context("parsing retry limit")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .starts_with("validation: parsing retry limit: "));
    }

    #[test]
    fn option_ext_helpers() {
        let some: Option<u8> = Some(3);
        assert_eq!(some.ok_or_not_found("session").unwrap(), 3);

        let missing: Option<u8> = None;
        let error = missing.ok_or_not_found("session").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(error.to_string(), "not_found: session not found");

        let error = missing.ok_or_validation("api key").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), "validation: api key is required");
    }
}
