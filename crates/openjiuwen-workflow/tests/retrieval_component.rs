#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    dead_code,
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound,
    clippy::assert_is_empty
)]
//! Knowledge-retrieval component behavior through the public workflow API.
//!
//! The component validates its query, delegates to `SimpleKnowledgeBase`, and reports
//! `{"query", "documents": [{"text", "score", "metadata"}]}`. A blank or missing query is
//! the coded `101102` component error, which fails the workflow run.
//!
//! Ports of `tests/unit_tests/core/component/test_knowledge_retrieval_comp.py`, rewritten
//! against `Workflow`/`ComponentOptions`. The knowledge base uses `HashEmbedding` and
//! `InMemoryVectorStore`, so the suite is offline and deterministic.
mod common;

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::{
    Document, HashEmbedding, InMemoryVectorStore, KnowledgeBaseConfig, SimpleKnowledgeBase,
};
use serde_json::json;

use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, EndComponent, KnowledgeRetrievalComponent,
    StartComponent, Workflow, WorkflowExecutionState,
};

use common::{card, session};

/// Two documents, one of which carries metadata, indexed with the offline embedder.
async fn knowledge_base() -> Arc<SimpleKnowledgeBase> {
    let kb = SimpleKnowledgeBase::from_config(
        KnowledgeBaseConfig::new("kb"),
        Arc::new(InMemoryVectorStore::new()),
        Arc::new(HashEmbedding::new(64)),
    )
    .expect("the knowledge base builds");
    kb.add_documents(vec![
        Document::new("rust ownership"),
        Document::new("tokio runtime").metadata_entry("source", "docs"),
    ])
    .await
    .expect("the documents are indexed");
    Arc::new(kb)
}

/// `start → retrieval → end`, where the retrieval node reads its `question` key and the
/// end node reports the component's value flat.
fn retrieval_workflow(id: &str, kb: Arc<SimpleKnowledgeBase>) -> Workflow {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"question": "${inputs.question}"})),
        )
        .add_workflow_comp(
            "retrieval",
            KnowledgeRetrievalComponent::new(kb).with_query_key("question"),
            ComponentOptions::new().with_inputs_schema(json!({"question": "${start.question}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!("${retrieval}")),
        )
        .add_connection("start", "retrieval")
        .add_connection("retrieval", "end");
    workflow
}

// port of tests/unit_tests/core/component/test_knowledge_retrieval_comp.py::test_start_kr_end_workflow
#[tokio::test]
async fn a_retrieval_workflow_returns_documents_and_echoes_the_query() {
    let output = retrieval_workflow("kr-workflow", knowledge_base().await)
        .invoke(json!({"question": "tokio"}), session("kr-workflow"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let result = &output.result["output"];
    assert_eq!(result["query"], json!("tokio"));

    let documents = result["documents"].as_array().expect("a documents array");
    assert!(!documents.is_empty(), "at least one hit");
    for document in documents {
        assert!(document["text"].is_string(), "{document}");
        assert!(document["score"].is_number(), "{document}");
        assert!(document["metadata"].is_object(), "{document}");
    }
    assert!(
        documents
            .iter()
            .any(|document| document["metadata"]["source"] == json!("docs")),
        "the document metadata survives retrieval: {documents:?}"
    );
}

// port of tests/unit_tests/core/component/test_knowledge_retrieval_comp.py::test_invoke_empty_query_raises
#[tokio::test]
async fn a_blank_or_missing_query_fails_the_run_with_101102() {
    let workflow = retrieval_workflow("kr-invalid", knowledge_base().await);

    let missing = workflow
        .invoke(json!({"other": 1}), session("kr-missing"))
        .await
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::Workflow);
    assert!(missing.message().contains("[101102]"), "{missing}");

    let blank = workflow
        .invoke(json!({"question": "   "}), session("kr-blank"))
        .await
        .unwrap_err();
    assert_eq!(blank.kind(), ErrorKind::Workflow);
    assert!(blank.message().contains("[101102]"), "{blank}");
}

// port of tests/unit_tests/core/component/test_knowledge_retrieval_comp.py::test_invoke_returns_results
#[tokio::test]
async fn a_bare_string_input_is_used_as_the_query() {
    let component = KnowledgeRetrievalComponent::new(knowledge_base().await);
    let context = ComponentContext::new(session("kr-direct"), "retrieval", "wf");

    let output = component.invoke(json!("rust"), &context).await.unwrap();

    assert_eq!(output["query"], json!("rust"));
    assert!(!output["documents"]
        .as_array()
        .expect("a documents array")
        .is_empty());
}
