//! Request-purpose correlation for compaction model calls.
//!
//! A compaction pass that calls a model must be attributable in callbacks and
//! logs: Python binds the operation with `context_compression_operation(...)`
//! (a `ContextVar`) and stamps `request_purpose="compaction"` plus the operation
//! id onto the model call. The Rust port uses a `tokio` task-local instead of a
//! process-global, so concurrent tasks (and therefore concurrent sessions) each
//! see their own operation.
//!
//! ```
//! use openjiuwen_context::telemetry::{
//!     current_compression_operation_id, with_operation,
//! };
//!
//! # async fn example() {
//! assert!(current_compression_operation_id().is_none());
//! with_operation("op-1", async {
//!     assert_eq!(current_compression_operation_id().as_deref(), Some("op-1"));
//! })
//! .await;
//! assert!(current_compression_operation_id().is_none());
//! # }
//! ```
use std::future::Future;

use openjiuwen_foundation::llm::ModelRequestConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The `request_purpose` a compaction pass stamps onto its model calls.
pub const COMPACTION_PURPOSE: &str = "compaction";

/// Key of the request purpose inside [`ModelRequestConfig::extra`].
pub const REQUEST_PURPOSE_KEY: &str = "request_purpose";

/// Key of the compression operation id inside [`ModelRequestConfig::extra`].
pub const CONTEXT_OPERATION_ID_KEY: &str = "context_operation_id";

/// One context-processor operation, correlated with the model calls it makes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompressionOperation {
    /// Id of the processor pass; empty means "not in an operation".
    pub id: String,
    /// Request purpose reported alongside the id, [`COMPACTION_PURPOSE`] by default.
    pub purpose: String,
}

impl CompressionOperation {
    /// Creates a [`COMPACTION_PURPOSE`] operation for `id`.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            purpose: COMPACTION_PURPOSE.to_owned(),
        }
    }

    /// Creates an operation with an explicit request purpose.
    #[must_use]
    pub fn with_purpose(id: impl Into<String>, purpose: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            purpose: purpose.into(),
        }
    }

    /// Whether the operation carries no id, and therefore correlates nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// The purpose to report; [`COMPACTION_PURPOSE`] when none was configured.
    #[must_use]
    pub fn request_purpose(&self) -> &str {
        if self.purpose.is_empty() {
            COMPACTION_PURPOSE
        } else {
            &self.purpose
        }
    }

    /// The operation id to report, or `None` outside an operation.
    #[must_use]
    pub fn context_operation_id(&self) -> Option<&str> {
        (!self.id.is_empty()).then_some(self.id.as_str())
    }

    /// The correlation keys as a JSON object, in Python's kwarg spelling.
    #[must_use]
    pub fn model_kwargs(&self) -> Map<String, Value> {
        let mut kwargs = Map::new();
        if let Some(operation_id) = self.context_operation_id() {
            kwargs.insert(
                REQUEST_PURPOSE_KEY.to_owned(),
                Value::String(self.request_purpose().to_owned()),
            );
            kwargs.insert(
                CONTEXT_OPERATION_ID_KEY.to_owned(),
                Value::String(operation_id.to_owned()),
            );
        }
        kwargs
    }

    /// Stamps this operation onto an outbound model request.
    ///
    /// The correlation keys are merged into [`ModelRequestConfig::extra`], which
    /// is the request's only free-form field; existing keys are preserved and an
    /// empty operation changes nothing. Python's keys are callback-only kwargs,
    /// while a Rust provider merges `extra` into the request body last, so this
    /// is opt-in: a caller that must not widen the wire body should read
    /// [`CompressionOperation::request_purpose`] and
    /// [`CompressionOperation::context_operation_id`] instead and report them
    /// through its own callback.
    ///
    /// An `extra` that is neither an object nor `null` cannot hold the keys, so
    /// the request keeps its value and the pass is logged as uncorrelated at
    /// `warn`; the model call itself still runs.
    pub fn apply_to(&self, request: &mut ModelRequestConfig) {
        let kwargs = self.model_kwargs();
        if kwargs.is_empty() {
            return;
        }
        match &mut request.extra {
            Value::Object(extra) => {
                for (key, value) in kwargs {
                    extra.insert(key, value);
                }
            }
            Value::Null => request.extra = Value::Object(kwargs),
            other => tracing::warn!(
                extra_shape = json_shape(other),
                operation_id = self.context_operation_id().unwrap_or_default(),
                request_purpose = self.request_purpose(),
                "cannot stamp the compaction correlation: the request's `extra` is neither a JSON \
                 object nor null, so `request_purpose`/`context_operation_id` are dropped and this \
                 model call stays uncorrelated"
            ),
        }
    }

    /// Removes the correlation keys from an outbound model request.
    pub fn clear_from(request: &mut ModelRequestConfig) {
        let Value::Object(extra) = &mut request.extra else {
            return;
        };
        extra.remove(REQUEST_PURPOSE_KEY);
        extra.remove(CONTEXT_OPERATION_ID_KEY);
    }
}

/// The JSON shape of `value`, for a warning that names what it received.
fn json_shape(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

tokio::task_local! {
    static CURRENT_OPERATION: CompressionOperation;
}

/// Runs `future` with `operation_id` bound as the current operation.
///
/// The binding is a `tokio` task-local, so nested scopes restore the outer id on
/// exit, concurrent tasks never observe each other's id, and nothing is visible
/// outside a scope. Tasks spawned inside the scope inherit the binding.
pub async fn with_operation<F>(operation_id: impl Into<String>, future: F) -> F::Output
where
    F: Future,
{
    with_operation_scope(CompressionOperation::new(operation_id), future).await
}

/// Runs `future` with an explicit [`CompressionOperation`] bound.
pub async fn with_operation_scope<F>(operation: CompressionOperation, future: F) -> F::Output
where
    F: Future,
{
    CURRENT_OPERATION.scope(operation, future).await
}

/// The operation bound to the current task, if any.
#[must_use]
pub fn current_compression_operation() -> Option<CompressionOperation> {
    CURRENT_OPERATION.try_with(Clone::clone).ok()
}

/// The operation id bound to the current task, if any.
#[must_use]
pub fn current_compression_operation_id() -> Option<String> {
    current_compression_operation()
        .map(|operation| operation.id)
        .filter(|id| !id.is_empty())
}

/// The request purpose bound to the current task, if any.
#[must_use]
pub fn current_request_purpose() -> Option<String> {
    current_compression_operation()
        .filter(|operation| !operation.is_empty())
        .map(|operation| operation.request_purpose().to_owned())
}

/// Stamps the current operation onto `request`, when one is bound.
pub fn stamp_current_operation(request: &mut ModelRequestConfig) {
    if let Some(operation) = current_compression_operation() {
        operation.apply_to(request);
    }
}

/// Stamps `operation` onto `request`; `None` clears a previous stamp.
pub fn apply_operation(request: &mut ModelRequestConfig, operation: Option<&CompressionOperation>) {
    match operation {
        Some(operation) => operation.apply_to(request),
        None => CompressionOperation::clear_from(request),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_operation_is_visible_inside_and_absent_outside_its_scope() {
        assert!(current_compression_operation_id().is_none());
        let seen = with_operation("op-1", async {
            let operation = current_compression_operation().expect("operation is bound");
            assert_eq!(operation.id, "op-1");
            assert_eq!(
                current_request_purpose().as_deref(),
                Some(COMPACTION_PURPOSE)
            );
            current_compression_operation_id()
        })
        .await;
        assert_eq!(seen.as_deref(), Some("op-1"));
        assert!(current_compression_operation_id().is_none());
        assert!(current_request_purpose().is_none());
    }

    #[test]
    fn stamping_merges_into_extra_without_losing_keys() {
        let mut request = ModelRequestConfig::new("m").with_extra(serde_json::json!({"seed": 1}));
        CompressionOperation::new("op-7").apply_to(&mut request);
        assert_eq!(request.extra[REQUEST_PURPOSE_KEY], "compaction");
        assert_eq!(request.extra[CONTEXT_OPERATION_ID_KEY], "op-7");
        assert_eq!(request.extra["seed"], 1);

        CompressionOperation::clear_from(&mut request);
        assert_eq!(request.extra, serde_json::json!({"seed": 1}));

        let mut empty = ModelRequestConfig::new("m");
        CompressionOperation::new("op-8").apply_to(&mut empty);
        assert_eq!(empty.extra[REQUEST_PURPOSE_KEY], "compaction");
    }
}
