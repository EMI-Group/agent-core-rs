#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end tests for the public embedding surface.
//!
//! Exercises [`Embedding`], [`HashEmbedding`], and [`OpenAiEmbedding`] through the
//! crate root against a `wiremock` server: request shape (model/input/dimensions),
//! bearer authentication, endpoint normalisation, batching, response-shape
//! tolerance, and error mapping. The unit tests inside `src/embedding/openai/`
//! cover the same client at module scope; these tests pin the behavior callers see.
//!
//! Every test is offline and deterministic: no wall clock, no random data, and the
//! wiremock server is loopback-only.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;
use openjiuwen_retrieval::{Embedding, EmbeddingConfig, HashEmbedding, OpenAiEmbedding};
use serde_json::{json, Value};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Asserts two vectors match within `f32` rounding tolerance.
fn assert_vector_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "actual {actual:?} vs expected {expected:?}"
    );
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "actual {actual:?} vs expected {expected:?}"
        );
    }
}

/// Responds with one vector per input text, tagged by the text's numeric suffix.
///
/// Entries deliberately omit the `index` field so the output order can only come
/// from the client's batch concatenation, not from response-internal sorting.
fn tagged_batch_response(request: &Request) -> ResponseTemplate {
    let body: Value = request.body_json().unwrap();
    let inputs = body["input"].as_array().unwrap();
    let data: Vec<Value> = inputs
        .iter()
        .map(|input| {
            let text = input.as_str().unwrap();
            let index: i32 = text.trim_start_matches("text-").parse().unwrap();
            json!({"embedding": [f64::from(index), 1.0]})
        })
        .collect();
    ResponseTemplate::new(200).set_body_json(json!({"data": data, "model": "m"}))
}

#[tokio::test]
async fn hash_embedding_is_deterministic_and_reports_its_dimension() {
    let embedder = HashEmbedding::default();
    assert_eq!(embedder.model_name(), "hash-embedding");
    assert_eq!(embedder.dimension_hint(), Some(64));
    assert_eq!(embedder.dimension().await.unwrap(), 64);

    let query = embedder.embed_query("the quick brown fox").await.unwrap();
    assert_eq!(query.len(), 64);
    // Same text, same vectors: across calls and across independently built instances.
    assert_eq!(
        embedder.embed_query("the quick brown fox").await.unwrap(),
        query
    );
    assert_eq!(
        HashEmbedding::new(64)
            .embed_query("the quick brown fox")
            .await
            .unwrap(),
        query
    );

    let other = embedder.embed_query("a different sentence").await.unwrap();
    assert_ne!(other, query);

    // Zero falls back to the documented default dimension.
    assert_eq!(HashEmbedding::new(0).dimension_hint(), Some(64));
    assert_eq!(HashEmbedding::new(8).dimension().await.unwrap(), 8);
}

#[tokio::test]
async fn hash_embedding_embed_documents_preserves_order() {
    let embedder = HashEmbedding::new(16);
    let texts = vec![
        "alpha".to_owned(),
        "beta gamma".to_owned(),
        "delta".to_owned(),
        "alpha".to_owned(),
    ];
    // A batch-size override that the deterministic fake ignores must not reorder.
    let vectors = embedder.embed_documents(&texts, Some(1)).await.unwrap();

    assert_eq!(vectors.len(), texts.len());
    for (vector, text) in vectors.iter().zip(&texts) {
        assert_eq!(vector.len(), 16);
        assert_eq!(vector, &embedder.embed_query(text).await.unwrap());
    }
    // Repeated input text yields the identical vector in position.
    assert_eq!(vectors[0], vectors[3]);
}

#[tokio::test]
async fn openai_embedding_posts_model_and_plain_string_input() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "m"
        })))
        .mount(&server)
        .await;

    let embedder =
        OpenAiEmbedding::new(EmbeddingConfig::new("text-embedding-3-small", server.uri())).unwrap();
    assert_eq!(embedder.endpoint(), format!("{}/embeddings", server.uri()));
    assert_eq!(embedder.dimension_hint(), None);

    let vector = embedder.embed_query("hello").await.unwrap();
    assert_vector_close(&vector, &[0.1, 0.2, 0.3]);
    assert_eq!(embedder.dimension_hint(), Some(3));

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = requests[0].body_json().unwrap();
    assert_eq!(body["model"], json!("text-embedding-3-small"));
    assert_eq!(body["input"], json!("hello"));
    assert!(
        body.get("dimensions").is_none(),
        "dimensions must be omitted when unset: {body}"
    );
}

#[tokio::test]
async fn openai_embedding_sends_bearer_auth_to_the_v1_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.5, 0.25]}]
        })))
        .mount(&server)
        .await;

    let config = EmbeddingConfig::new("m", format!("{}/v1", server.uri()))
        .with_api_key(SecretString::new("sk-test"));
    let embedder = OpenAiEmbedding::new(config).unwrap();
    assert_eq!(
        embedder.endpoint(),
        format!("{}/v1/embeddings", server.uri())
    );

    assert_eq!(
        embedder.embed_query("hello").await.unwrap(),
        vec![0.5, 0.25]
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1, "the bearer matcher must have matched");
    assert_eq!(requests[0].url.path(), "/v1/embeddings");
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer sk-test"
    );
}

#[tokio::test]
async fn openai_embedding_uses_an_explicit_embeddings_endpoint_unchanged() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [3.0]}]
        })))
        .mount(&server)
        .await;

    let base = format!("{}/v1/embeddings", server.uri());
    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", base.clone())).unwrap();
    assert_eq!(embedder.endpoint(), base);
    assert_eq!(embedder.embed_query("hello").await.unwrap(), vec![3.0]);

    // A trailing slash on an explicit endpoint normalises to the same URL.
    let trailing = OpenAiEmbedding::new(EmbeddingConfig::new(
        "m",
        format!("{}/v1/embeddings/", server.uri()),
    ))
    .unwrap();
    assert_eq!(trailing.endpoint(), base);
    assert_eq!(trailing.embed_query("hello").await.unwrap(), vec![3.0]);

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(request.url.path(), "/v1/embeddings");
    }
}

#[tokio::test]
async fn openai_embedding_sends_configured_dimensions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_json(json!({
            "model": "m",
            "input": "hello",
            "dimensions": 8
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}]
        })))
        .mount(&server)
        .await;

    let embedder =
        OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri()).with_dimension(8)).unwrap();
    assert_eq!(embedder.dimension_hint(), Some(8));
    assert_eq!(embedder.dimension().await.unwrap(), 8);

    let vector = embedder.embed_query("hello").await.unwrap();
    assert_eq!(vector.len(), 8);
    assert_eq!(vector, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let requests = server.received_requests().await.unwrap();
    let body: Value = requests[0].body_json().unwrap();
    assert_eq!(body["dimensions"], json!(8));
}

#[tokio::test]
async fn openai_embedding_batches_documents_and_preserves_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(tagged_batch_response)
        .expect(3)
        .named("batched embeddings")
        .mount(&server)
        .await;

    let embedder =
        OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri()).with_max_batch_size(2))
            .unwrap();
    let texts: Vec<String> = (0..5).map(|index| format!("text-{index}")).collect();

    let vectors = embedder.embed_documents(&texts, None).await.unwrap();
    assert_eq!(
        vectors,
        vec![
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![3.0, 1.0],
            vec![4.0, 1.0],
        ]
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3, "five inputs at batch size two");
    let inputs: Vec<Value> = requests
        .iter()
        .map(|request| request.body_json::<Value>().unwrap()["input"].clone())
        .collect();
    assert_eq!(
        inputs,
        vec![
            json!(["text-0", "text-1"]),
            json!(["text-2", "text-3"]),
            json!(["text-4"]),
        ]
    );
}

#[tokio::test]
async fn openai_embedding_accepts_top_level_embedding_shapes() {
    let single = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": [0.5, 0.25]
        })))
        .mount(&single)
        .await;
    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", single.uri())).unwrap();
    assert_eq!(
        embedder.embed_query("hello").await.unwrap(),
        vec![0.5, 0.25]
    );

    let multiple = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [[1.0], [2.0]]
        })))
        .mount(&multiple)
        .await;
    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", multiple.uri())).unwrap();
    let texts = ["one".to_owned(), "two".to_owned()];
    assert_eq!(
        embedder.embed_documents(&texts, None).await.unwrap(),
        vec![vec![1.0], vec![2.0]]
    );
}

#[tokio::test]
async fn openai_embedding_orders_data_entries_by_index() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"index": 2, "embedding": [2.0]},
                {"index": 0, "embedding": [0.0]},
                {"index": 1, "embedding": [1.0]}
            ]
        })))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    let texts = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
    assert_eq!(
        embedder.embed_documents(&texts, None).await.unwrap(),
        vec![vec![0.0], vec![1.0], vec![2.0]]
    );
}

#[tokio::test]
async fn openai_embedding_maps_server_errors_to_retrieval_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
        .mount(&server)
        .await;

    // Retries are disabled so the test observes the terminal 500 directly.
    let embedder =
        OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri()).with_max_retries(0)).unwrap();
    let error = embedder.embed_query("hello").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("upstream exploded"), "{message}");
}

#[tokio::test]
async fn openai_embedding_rejects_a_vector_count_mismatch() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.5]}]
        })))
        .mount(&server)
        .await;

    let embedder =
        OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri()).with_max_retries(0)).unwrap();
    let texts = ["one".to_owned(), "two".to_owned()];
    let error = embedder.embed_documents(&texts, None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("1 vectors"), "{message}");
    assert!(message.contains("2 inputs"), "{message}");
}

#[tokio::test]
async fn openai_embedding_maps_malformed_json_to_retrieval_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json at all"))
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();
    let error = embedder.embed_query("hello").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("decoding embedding response"), "{message}");

    // A well-formed payload carrying none of the known fields is also rejected.
    let wrong_shape = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"unexpected": true})))
        .mount(&wrong_shape)
        .await;
    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", wrong_shape.uri())).unwrap();
    let error = embedder.embed_query("hello").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.to_string().contains("embedding response"), "{error}");
}

#[tokio::test]
async fn blank_inputs_are_rejected_before_any_request() {
    let server = MockServer::start().await;
    // The expectation fails the test if validation ever reaches the transport.
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [1.0]}]
        })))
        .expect(0)
        .named("no embedding requests")
        .mount(&server)
        .await;

    let embedder = OpenAiEmbedding::new(EmbeddingConfig::new("m", server.uri())).unwrap();

    let error = embedder.embed_query("  ").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = embedder.embed_query("").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = embedder.embed_documents(&[], None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("must not be empty"), "{error}");

    let error = embedder
        .embed_documents(&["ok".to_owned(), "   ".to_owned()], None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("index 1"), "{error}");
}
