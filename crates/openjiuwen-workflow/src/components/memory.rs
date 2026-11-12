//! Built-in memory components and the seams they run over.
//!
//! Ports the two resource components
//! `openjiuwen/core/workflow/components/resource/memory_retrieval_comp.py` and
//! `memory_write_comp.py`. Python calls `LongTermMemory` directly; `openjiuwen-memory`
//! is still a stub in this workspace, so the components depend on the narrow traits
//! [`MemoryRecall`] and [`MemoryRecord`] instead. Adapters over the real long-term
//! memory land in `openjiuwen-memory`.
//!
//! Returned shapes:
//!
//! - [`MemoryRetrievalComponent`] → `{"memories": [ … ]}`;
//! - [`MemoryWriteComponent`] → whatever [`MemoryRecord::record`] returned.
//!
//! Inputs are resolved like the knowledge-retrieval component: the query comes from
//! the component's `query_key` entry, or from the whole input when it is a bare
//! string.

use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::BaseMessage;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::components::retrieval::resolve_query;
use crate::context::ComponentContext;
use crate::error::{
    memory_retrieval_input_param_error, memory_retrieval_invoke_call_failed,
    memory_write_input_param_error, memory_write_invoke_call_failed,
};

/// Reads relevant memories for a query.
///
/// Adapters over `openjiuwen-memory` implement this; the component only needs the
/// three-argument lookup.
#[async_trait::async_trait]
pub trait MemoryRecall: Send + Sync {
    /// Returns the memories matching `query` in `scope_id` for `user_id`.
    ///
    /// # Errors
    ///
    /// Implementations report memory-subsystem failures; the component wraps them
    /// in [`memory_retrieval_invoke_call_failed`].
    async fn recall(&self, query: &str, scope_id: &str, user_id: &str) -> Result<Vec<Value>>;
}

/// Writes a conversation to long-term memory.
///
/// Adapters over `openjiuwen-memory` implement this; the returned value is the
/// component's output verbatim.
#[async_trait::async_trait]
pub trait MemoryRecord: Send + Sync {
    /// Stores `messages` in `scope_id` for `user_id` and returns the record summary.
    ///
    /// # Errors
    ///
    /// Implementations report memory-subsystem failures; the component wraps them
    /// in [`memory_write_invoke_call_failed`].
    async fn record(
        &self,
        messages: Vec<BaseMessage>,
        scope_id: &str,
        user_id: &str,
    ) -> Result<Value>;
}

/// The built-in `memory_retrieval` component.
pub struct MemoryRetrievalComponent {
    recall: Arc<dyn MemoryRecall>,
    scope_id: String,
    user_id: String,
    query_key: String,
}

impl MemoryRetrievalComponent {
    /// Builds a component reading from `recall` in `scope_id` for `user_id`.
    ///
    /// The query key defaults to `"query"`.
    #[must_use]
    pub fn new(
        recall: Arc<dyn MemoryRecall>,
        scope_id: impl Into<String>,
        user_id: impl Into<String>,
    ) -> Self {
        Self {
            recall,
            scope_id: scope_id.into(),
            user_id: user_id.into(),
            query_key: "query".to_owned(),
        }
    }

    /// Sets the input key the query is read from (default `"query"`).
    #[must_use]
    pub fn with_query_key(mut self, query_key: impl Into<String>) -> Self {
        self.query_key = query_key.into();
        self
    }

    /// The input key the query is read from.
    #[must_use]
    pub fn query_key(&self) -> &str {
        &self.query_key
    }
}

#[async_trait::async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for MemoryRetrievalComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("memory_retrieval", "memory_retrieval")
            .with_description("retrieves long-term memories for a query")
            .with_abilities(vec![ComponentAbility::Invoke])
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "memory_retrieval"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let query = resolve_query(&inputs, &self.query_key)
            .filter(|query| !query.trim().is_empty())
            .ok_or_else(|| {
                memory_retrieval_input_param_error("Query must be a non-empty string")
            })?;
        let memories = self
            .recall
            .recall(&query, &self.scope_id, &self.user_id)
            .await
            .map_err(|error| memory_retrieval_invoke_call_failed(&format!("{error}")))?;
        Ok(json!({"memories": memories}))
    }
}

/// The built-in `memory_write` component.
pub struct MemoryWriteComponent {
    record: Arc<dyn MemoryRecord>,
    scope_id: String,
    user_id: String,
}

impl MemoryWriteComponent {
    /// Builds a component writing through `record` in `scope_id` for `user_id`.
    #[must_use]
    pub fn new(
        record: Arc<dyn MemoryRecord>,
        scope_id: impl Into<String>,
        user_id: impl Into<String>,
    ) -> Self {
        Self {
            record,
            scope_id: scope_id.into(),
            user_id: user_id.into(),
        }
    }
}

/// Deserializes the `"messages"` input entry into a non-empty conversation.
///
/// A missing entry, a non-array entry, an empty array, and a malformed message all
/// become [`memory_write_input_param_error`] (`101150`).
fn messages_from_input(inputs: &Value) -> Result<Vec<BaseMessage>> {
    let items = match inputs.get("messages") {
        Some(Value::Array(items)) => items,
        Some(other) => {
            return Err(memory_write_input_param_error(&format!(
                "Messages must be an array, got {}",
                type_name(other)
            )));
        }
        None => {
            return Err(memory_write_input_param_error(
                "Messages list cannot be empty",
            ));
        }
    };
    if items.is_empty() {
        return Err(memory_write_input_param_error(
            "Messages list cannot be empty",
        ));
    }
    items
        .iter()
        .map(|item| {
            serde_json::from_value::<BaseMessage>(item.clone()).map_err(|error| {
                memory_write_input_param_error(&format!("Invalid message: {error}"))
            })
        })
        .collect()
}

/// The JSON type name of `value`, for a readable input error.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[async_trait::async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for MemoryWriteComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("memory_write", "memory_write")
            .with_description("writes a conversation to long-term memory")
            .with_abilities(vec![ComponentAbility::Invoke])
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "memory_write"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let messages = messages_from_input(&inputs)?;
        self.record
            .record(messages, &self.scope_id, &self.user_id)
            .await
            .map_err(|error| memory_write_invoke_call_failed(&format!("{error}")))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::type_complexity,
    clippy::assert_is_empty
)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, ErrorKind, Result};
    use openjiuwen_foundation::{BaseMessage, Role};
    use openjiuwen_session::{Session, WorkflowSession};
    use parking_lot::Mutex;
    use serde_json::{json, Value};

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("memory-tests"));
        ComponentContext::new(session, "memory", "wf")
    }

    /// A recall seam that returns fixed memories or fails.
    struct FakeRecall {
        memories: Vec<Value>,
        fail: bool,
        calls: Arc<Mutex<Vec<(String, String, String)>>>,
    }

    impl FakeRecall {
        fn returning(memories: Vec<Value>) -> Self {
            Self {
                memories,
                fail: false,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn failing() -> Self {
            Self {
                memories: Vec::new(),
                fail: true,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl MemoryRecall for FakeRecall {
        async fn recall(&self, query: &str, scope_id: &str, user_id: &str) -> Result<Vec<Value>> {
            self.calls
                .lock()
                .push((query.to_owned(), scope_id.to_owned(), user_id.to_owned()));
            if self.fail {
                Err(Error::memory("recall boom"))
            } else {
                Ok(self.memories.clone())
            }
        }
    }

    /// A record seam that echoes `{"success": true}` or fails.
    struct FakeRecord {
        result: Value,
        fail: bool,
        calls: Arc<Mutex<Vec<(Vec<BaseMessage>, String, String)>>>,
    }

    impl FakeRecord {
        fn succeeding(result: Value) -> Self {
            Self {
                result,
                fail: false,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn failing() -> Self {
            Self {
                result: Value::Null,
                fail: true,
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl MemoryRecord for FakeRecord {
        async fn record(
            &self,
            messages: Vec<BaseMessage>,
            scope_id: &str,
            user_id: &str,
        ) -> Result<Value> {
            self.calls
                .lock()
                .push((messages, scope_id.to_owned(), user_id.to_owned()));
            if self.fail {
                Err(Error::memory("record boom"))
            } else {
                Ok(self.result.clone())
            }
        }
    }

    #[tokio::test]
    async fn memories_are_returned_with_scope_and_user() {
        let recall = Arc::new(FakeRecall::returning(vec![json!({"text": "likes rust"})]));
        let component = MemoryRetrievalComponent::new(
            Arc::clone(&recall) as Arc<dyn MemoryRecall>,
            "scope",
            "user",
        );

        let output = component
            .invoke(json!({"query": "rust"}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"memories": [{"text": "likes rust"}]}));
        assert_eq!(
            *recall.calls.lock(),
            vec![("rust".to_owned(), "scope".to_owned(), "user".to_owned())]
        );
    }

    #[tokio::test]
    async fn blank_or_missing_query_is_a_101200_error_without_a_call() {
        let recall = Arc::new(FakeRecall::returning(vec![]));
        let component =
            MemoryRetrievalComponent::new(Arc::clone(&recall) as Arc<dyn MemoryRecall>, "s", "u");

        for inputs in [json!({}), json!({"query": "  "})] {
            let error = component.invoke(inputs, &context()).await.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Workflow);
            assert!(error.message().contains("[101200]"));
            assert!(error.message().contains("Query must be a non-empty string"));
        }
        assert!(recall.calls.lock().is_empty());
    }

    #[tokio::test]
    async fn a_bare_string_input_is_the_query() {
        let recall = Arc::new(FakeRecall::returning(vec![]));
        let component =
            MemoryRetrievalComponent::new(Arc::clone(&recall) as Arc<dyn MemoryRecall>, "s", "u");

        let output = component.invoke(json!("bare"), &context()).await.unwrap();
        assert_eq!(output, json!({"memories": []}));
        assert_eq!(recall.calls.lock()[0].0, "bare");
    }

    #[tokio::test]
    async fn a_custom_query_key_is_honored() {
        let recall = Arc::new(FakeRecall::returning(vec![]));
        let component =
            MemoryRetrievalComponent::new(Arc::clone(&recall) as Arc<dyn MemoryRecall>, "s", "u")
                .with_query_key("question");
        assert_eq!(component.query_key(), "question");

        component
            .invoke(json!({"question": "why"}), &context())
            .await
            .unwrap();
        assert_eq!(recall.calls.lock()[0].0, "why");

        let error = component
            .invoke(json!({"query": "why"}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101200]"));
    }

    #[tokio::test]
    async fn recall_failure_is_wrapped_with_code_101201() {
        let recall = Arc::new(FakeRecall::failing());
        let component = MemoryRetrievalComponent::new(recall as Arc<dyn MemoryRecall>, "s", "u");

        let error = component
            .invoke(json!({"query": "rust"}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101201]"));
        assert!(error.message().contains("recall boom"));
    }

    #[tokio::test]
    async fn record_output_is_returned_verbatim() {
        let record = Arc::new(FakeRecord::succeeding(json!({"success": true, "count": 1})));
        let component =
            MemoryWriteComponent::new(Arc::clone(&record) as Arc<dyn MemoryRecord>, "s", "u");

        let output = component
            .invoke(
                json!({"messages": [{"role": "user", "content": "hello"}]}),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output, json!({"success": true, "count": 1}));
    }

    #[tokio::test]
    async fn messages_are_deserialized_and_scope_and_user_forwarded() {
        let record = Arc::new(FakeRecord::succeeding(json!(null)));
        let component = MemoryWriteComponent::new(
            Arc::clone(&record) as Arc<dyn MemoryRecord>,
            "scope",
            "user",
        );

        component
            .invoke(
                json!({"messages": [
                    {"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hello"}
                ]}),
                &context(),
            )
            .await
            .unwrap();

        let calls = record.calls.lock();
        let (messages, scope_id, user_id) = &calls[0];
        assert_eq!(*scope_id, "scope");
        assert_eq!(*user_id, "user");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].text(), "be nice");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].text(), "hello");
    }

    #[tokio::test]
    async fn empty_or_missing_messages_are_101150_errors() {
        let record = Arc::new(FakeRecord::succeeding(json!(null)));
        let component =
            MemoryWriteComponent::new(Arc::clone(&record) as Arc<dyn MemoryRecord>, "s", "u");

        for inputs in [json!({}), json!({"messages": []})] {
            let error = component.invoke(inputs, &context()).await.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Workflow);
            assert!(error.message().contains("[101150]"));
            assert!(error.message().contains("Messages list cannot be empty"));
        }
        assert!(record.calls.lock().is_empty());
    }

    #[tokio::test]
    async fn non_array_messages_are_a_101150_error() {
        let record = Arc::new(FakeRecord::succeeding(json!(null)));
        let component =
            MemoryWriteComponent::new(Arc::clone(&record) as Arc<dyn MemoryRecord>, "s", "u");

        let error = component
            .invoke(json!({"messages": "hello"}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101150]"));
        assert!(error.message().contains("Messages must be an array"));
    }

    #[tokio::test]
    async fn a_malformed_message_is_a_101150_error() {
        let record = Arc::new(FakeRecord::succeeding(json!(null)));
        let component =
            MemoryWriteComponent::new(Arc::clone(&record) as Arc<dyn MemoryRecord>, "s", "u");

        let error = component
            .invoke(json!({"messages": [{"role": 7}]}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101150]"));
        assert!(error.message().contains("Invalid message"));
        assert!(record.calls.lock().is_empty());
    }

    #[tokio::test]
    async fn record_failure_is_wrapped_with_code_101151() {
        let record = Arc::new(FakeRecord::failing());
        let component = MemoryWriteComponent::new(record as Arc<dyn MemoryRecord>, "s", "u");

        let error = component
            .invoke(
                json!({"messages": [{"role": "user", "content": "hi"}]}),
                &context(),
            )
            .await
            .unwrap_err();
        assert!(error.message().contains("[101151]"));
        assert!(error.message().contains("record boom"));
    }

    #[test]
    fn component_identity() {
        let recall = Arc::new(FakeRecall::returning(vec![]));
        let retrieval = MemoryRetrievalComponent::new(recall as Arc<dyn MemoryRecall>, "s", "u");
        assert_eq!(retrieval.component_type(), "memory_retrieval");
        assert_eq!(retrieval.abilities(), vec![ComponentAbility::Invoke]);
        assert_eq!(retrieval.card().component_type, "memory_retrieval");

        let record = Arc::new(FakeRecord::succeeding(json!(null)));
        let write = MemoryWriteComponent::new(record as Arc<dyn MemoryRecord>, "s", "u");
        assert_eq!(write.component_type(), "memory_write");
        assert_eq!(write.abilities(), vec![ComponentAbility::Invoke]);
        assert_eq!(write.card().component_type, "memory_write");
    }
}
