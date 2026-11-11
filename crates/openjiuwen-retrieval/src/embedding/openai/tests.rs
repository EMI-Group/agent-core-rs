//! Wiremock-backed tests for the OpenAI-compatible embedding client.

use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;

use super::*;

fn embedder(server: &MockServer) -> OpenAiEmbedding {
    OpenAiEmbedding::new(
        EmbeddingConfig::new("text-embedding-3-small", format!("{}/v1", server.uri()))
            .with_api_key("sk-test"),
    )
    .unwrap()
}

#[test]
fn new_rejects_blank_configuration() {
    let error =
        OpenAiEmbedding::new(EmbeddingConfig::new("  ", "https://api.example/v1")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error =
        OpenAiEmbedding::new(EmbeddingConfig::new("text-embedding-3-small", " ")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn endpoint_is_normalised() {
    let trailing = EmbeddingConfig::new("m", "https://api.example/v1/");
    let embedder = OpenAiEmbedding::new(trailing).unwrap();
    assert_eq!(embedder.endpoint(), "https://api.example/v1/embeddings");

    let explicit = EmbeddingConfig::new("m", "https://api.example/v1/embeddings/");
    let embedder = OpenAiEmbedding::new(explicit).unwrap();
    assert_eq!(embedder.endpoint(), "https://api.example/v1/embeddings");
}

#[tokio::test]
async fn posts_model_input_and_bearer_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer sk-test"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": "hello"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
        })))
        .mount(&server)
        .await;

    let embedder = embedder(&server);
    assert_eq!(embedder.dimension_hint(), None);
    assert_eq!(
        embedder.embed_query("hello").await.unwrap(),
        vec![0.1, 0.2, 0.3]
    );
    // The vector length is cached as the dimension hint.
    assert_eq!(embedder.dimension_hint(), Some(3));
}

#[tokio::test]
async fn configured_dimension_is_sent_and_wins_the_hint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": "hello",
            "dimensions": 2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"embedding": [1.0, 0.0]}]
        })))
        .mount(&server)
        .await;

    let config = EmbeddingConfig::new("text-embedding-3-small", format!("{}/v1", server.uri()))
        .with_dimension(2);
    let embedder = OpenAiEmbedding::new(config).unwrap();
    assert_eq!(embedder.dimension_hint(), Some(2));
    assert_eq!(embedder.embed_query("hello").await.unwrap(), vec![1.0, 0.0]);
    assert_eq!(embedder.dimension_hint(), Some(2));
}

#[tokio::test]
async fn documents_are_sent_in_sequential_batches() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0]}
            ]
        })))
        .mount(&server)
        .await;

    let config = EmbeddingConfig::new("m", server.uri()).with_max_batch_size(2);
    let embedder = OpenAiEmbedding::new(config).unwrap();
    let texts = ["a", "b", "c", "d"].map(str::to_owned).to_vec();
    let vectors = embedder.embed_documents(&texts, None).await.unwrap();

    assert_eq!(
        vectors,
        vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0]
        ]
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2, "one request per batch");
    let first: Value = requests[0].body_json().unwrap();
    assert_eq!(first["input"], json!(["a", "b"]));
    let second: Value = requests[1].body_json().unwrap();
    assert_eq!(second["input"], json!(["c", "d"]));
}

#[tokio::test]
async fn data_entries_are_ordered_by_index() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]}
            ]
        })))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    let texts = ["first".to_owned(), "second".to_owned()];
    assert_eq!(
        embedder.embed_documents(&texts, None).await.unwrap(),
        vec![vec![1.0, 0.0], vec![0.0, 1.0]]
    );
}

#[tokio::test]
async fn top_level_embedding_and_embeddings_shapes_are_accepted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": [0.5, 0.5]
        })))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    assert_eq!(embedder.embed_query("hello").await.unwrap(), vec![0.5, 0.5]);
    drop(server);

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [[0.1], [0.2]]
        })))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    let texts = ["one".to_owned(), "two".to_owned()];
    assert_eq!(
        embedder.embed_documents(&texts, None).await.unwrap(),
        vec![vec![0.1], vec![0.2]]
    );
}

#[tokio::test]
async fn non_success_statuses_become_retrieval_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(
        EmbeddingConfig::new("m", server.uri())
            .with_api_key("sk-secret")
            .with_max_retries(0),
    )
    .unwrap();
    let error = embedder.embed_query("hello").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("401"), "{message}");
    assert!(message.contains("invalid api key"), "{message}");
    assert!(!message.contains("sk-secret"), "{message}");
}

#[tokio::test]
async fn vector_count_mismatch_is_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.1]}]
        })))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    let texts = ["one".to_owned(), "two".to_owned()];
    let error = embedder.embed_documents(&texts, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains('1') && message.contains('2'), "{message}");
}

#[tokio::test]
async fn malformed_payloads_become_retrieval_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"unexpected": true})))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    let error = embedder.embed_query("hello").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.to_string().contains("embedding response"), "{error}");
}
