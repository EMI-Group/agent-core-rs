//! Built-in `knowledge_retrieval` workflow component.
//!
//! Ports `openjiuwen/core/workflow/components/resource/knowledge_retrieval_comp.py`
//! onto [`SimpleKnowledgeBase`]. The component validates a query, runs the
//! knowledge base with its [`RetrievalConfig`], and returns a document array.
//!
//! The Python component accepts several knowledge bases and merges their results;
//! the Rust port takes one [`SimpleKnowledgeBase`] because the merged
//! multi-source payload (`MultiKBRetrievalResult`) has no place in this crate's
//! stable output shape yet.
//!
//! Returned shape:
//!
//! ```json
//! {"query": "…", "documents": [{"text": "…", "score": 0.9, "metadata": {}}]}
//! ```

use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_retrieval::{RetrievalConfig, RetrievalResult, SimpleKnowledgeBase};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::error::{knowledge_retrieval_input_param_error, knowledge_retrieval_invoke_call_failed};

/// Resolves a query string from `inputs`.
///
/// An object contributes its `query_key` entry, and a bare string input is used
/// as the query itself — the same fallback the memory-retrieval component needs.
/// A missing entry, a non-string entry, or a non-object/non-string input yields
/// `None`.
pub(crate) fn resolve_query(inputs: &Value, query_key: &str) -> Option<String> {
    match inputs {
        Value::Object(object) => object
            .get(query_key)
            .and_then(Value::as_str)
            .map(str::to_owned),
        Value::String(text) => Some(text.clone()),
        _ => None,
    }
}

/// Projects one retrieval hit onto the component's wire shape.
fn document(result: RetrievalResult) -> Value {
    let mut document = serde_json::Map::new();
    document.insert("text".to_owned(), Value::String(result.text));
    // A non-finite score cannot be a JSON number; report it as `null` instead of
    // failing (or panicking) on an otherwise usable hit.
    let score =
        serde_json::Number::from_f64(f64::from(result.score)).map_or(Value::Null, Value::Number);
    document.insert("score".to_owned(), score);
    document.insert("metadata".to_owned(), Value::Object(result.metadata));
    Value::Object(document)
}

/// The built-in `knowledge_retrieval` component.
pub struct KnowledgeRetrievalComponent {
    kb: Arc<SimpleKnowledgeBase>,
    retrieval: RetrievalConfig,
    query_key: String,
}

impl KnowledgeRetrievalComponent {
    /// Builds a component over `kb` with the default retrieval config and the
    /// query key `"query"`.
    #[must_use]
    pub fn new(kb: Arc<SimpleKnowledgeBase>) -> Self {
        Self {
            kb,
            retrieval: RetrievalConfig::default(),
            query_key: "query".to_owned(),
        }
    }

    /// Replaces the retrieval configuration.
    #[must_use]
    pub fn with_retrieval_config(mut self, retrieval: RetrievalConfig) -> Self {
        self.retrieval = retrieval;
        self
    }

    /// Sets the input key the query is read from (default `"query"`).
    #[must_use]
    pub fn with_query_key(mut self, query_key: impl Into<String>) -> Self {
        self.query_key = query_key.into();
        self
    }

    /// The knowledge base queried by this component.
    #[must_use]
    pub fn kb(&self) -> &Arc<SimpleKnowledgeBase> {
        &self.kb
    }

    /// The retrieval configuration.
    #[must_use]
    pub fn retrieval(&self) -> &RetrievalConfig {
        &self.retrieval
    }

    /// The input key the query is read from.
    #[must_use]
    pub fn query_key(&self) -> &str {
        &self.query_key
    }
}

#[async_trait::async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for KnowledgeRetrievalComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("knowledge_retrieval", "knowledge_retrieval")
            .with_description("retrieves documents from a knowledge base")
            .with_abilities(vec![ComponentAbility::Invoke])
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "knowledge_retrieval"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let query = resolve_query(&inputs, &self.query_key)
            .filter(|query| !query.trim().is_empty())
            .ok_or_else(|| {
                knowledge_retrieval_input_param_error("Query must be a non-empty string")
            })?;
        let results = self
            .kb
            .retrieve_with_config(&query, &self.retrieval)
            .await
            .map_err(|error| knowledge_retrieval_invoke_call_failed(&format!("{error}")))?;
        let documents = results.into_iter().map(document).collect::<Vec<_>>();
        Ok(json!({"query": query, "documents": documents}))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_retrieval::{HashEmbedding, InMemoryVectorStore, KnowledgeBaseConfig};
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("retrieval-tests"));
        ComponentContext::new(session, "knowledge_retrieval", "wf")
    }

    async fn knowledge_base() -> Arc<SimpleKnowledgeBase> {
        let kb = SimpleKnowledgeBase::from_config(
            KnowledgeBaseConfig::new("kb"),
            Arc::new(InMemoryVectorStore::new()),
            Arc::new(HashEmbedding::new(64)),
        )
        .expect("knowledge base builds");
        kb.add_texts(vec![
            "rust ownership".to_owned(),
            "tokio runtime".to_owned(),
        ])
        .await
        .expect("documents are indexed");
        Arc::new(kb)
    }

    #[tokio::test]
    async fn query_returns_documents_with_the_wire_shape() {
        let component = KnowledgeRetrievalComponent::new(knowledge_base().await);
        let output = component
            .invoke(json!({"query": "rust"}), &context())
            .await
            .unwrap();

        assert_eq!(output["query"], json!("rust"));
        let documents = output["documents"].as_array().expect("documents array");
        assert!(!documents.is_empty(), "at least one hit");
        let first = documents[0].as_object().expect("document object");
        assert!(first.contains_key("text"));
        assert!(first.contains_key("score"));
        assert!(first.contains_key("metadata"));
        assert!(first["text"].is_string());
        assert!(first["score"].is_number());
        assert!(first["metadata"].is_object());
    }

    #[tokio::test]
    async fn missing_query_is_a_101102_error() {
        let component = KnowledgeRetrievalComponent::new(knowledge_base().await);
        let error = component.invoke(json!({}), &context()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101102]"));
        assert!(error.message().contains("Query must be a non-empty string"));
    }

    #[tokio::test]
    async fn blank_query_is_a_101102_error() {
        let component = KnowledgeRetrievalComponent::new(knowledge_base().await);
        let error = component
            .invoke(json!({"query": "   "}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101102]"));
    }

    #[tokio::test]
    async fn non_string_query_is_a_101102_error() {
        let component = KnowledgeRetrievalComponent::new(knowledge_base().await);
        let error = component
            .invoke(json!({"query": 42}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101102]"));
    }

    #[tokio::test]
    async fn a_bare_string_input_is_the_query() {
        let component = KnowledgeRetrievalComponent::new(knowledge_base().await);
        let output = component.invoke(json!("rust"), &context()).await.unwrap();
        assert_eq!(output["query"], json!("rust"));
        assert!(!output["documents"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_custom_query_key_is_honored() {
        let component =
            KnowledgeRetrievalComponent::new(knowledge_base().await).with_query_key("question");
        assert_eq!(component.query_key(), "question");

        let output = component
            .invoke(json!({"question": "tokio"}), &context())
            .await
            .unwrap();
        assert_eq!(output["query"], json!("tokio"));

        let error = component
            .invoke(json!({"query": "tokio"}), &context())
            .await
            .unwrap_err();
        assert!(error.message().contains("[101102]"));
    }

    #[tokio::test]
    async fn retrieval_failure_is_wrapped_with_code_101100() {
        let component = KnowledgeRetrievalComponent::new(knowledge_base().await)
            .with_retrieval_config(RetrievalConfig {
                top_k: 0,
                ..RetrievalConfig::default()
            });
        let error = component
            .invoke(json!({"query": "rust"}), &context())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101100]"));
        assert!(error.message().contains("top_k"));
    }

    #[test]
    fn resolve_query_rules() {
        assert_eq!(
            resolve_query(&json!({"query": "q"}), "query"),
            Some("q".to_owned())
        );
        assert_eq!(resolve_query(&json!({"query": 1}), "query"), None);
        assert_eq!(resolve_query(&json!({}), "query"), None);
        assert_eq!(
            resolve_query(&json!("bare"), "query"),
            Some("bare".to_owned())
        );
        assert_eq!(resolve_query(&Value::Null, "query"), None);
    }
}
