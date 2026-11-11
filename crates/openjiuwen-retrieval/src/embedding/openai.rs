//! OpenAI-compatible embedding client.
//!
//! Port of the Python `retrieval/embedding/openai_embedding.py` client (also used
//! for every OpenAI-compatible gateway). Requests go through the shared
//! [`openjiuwen_common::http`] transport, so retries, timeouts, proxies, and TLS
//! verification are configured with [`openjiuwen_common::http::HttpClientConfig`]
//! instead of a provider SDK.

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};

use crate::common::EmbeddingConfig;

use super::{validate_query, validate_texts, Embedding};

/// Number of response-body characters kept in a non-2xx error message.
const MAX_ERROR_BODY_CHARS: usize = 512;

/// An embedding client for OpenAI-compatible `/embeddings` endpoints.
///
/// The client keeps one [`HttpClient`] for connection reuse and a cached vector
/// length so [`Embedding::dimension_hint`] can answer after the first successful
/// call even when the configuration does not fix a dimension.
#[derive(Debug)]
pub struct OpenAiEmbedding {
    config: EmbeddingConfig,
    client: HttpClient,
    cached_dimension: Mutex<Option<usize>>,
}

impl OpenAiEmbedding {
    /// Builds a client from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `model_name` or `base_url` is blank, and
    /// the errors of [`HttpClient::new`] when the transport cannot be constructed.
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        if config.model_name.trim().is_empty() {
            return Err(Error::Validation(
                "embedding model_name must not be empty".to_owned(),
            ));
        }
        if config.base_url.trim().is_empty() {
            return Err(Error::Validation(
                "embedding base_url must not be empty".to_owned(),
            ));
        }
        let client_config = HttpClientConfig::default()
            .with_timeout(config.timeout)
            .with_max_retries(config.max_retries)
            .with_verify_ssl(config.verify_ssl)
            .with_headers(config.headers.clone());
        let client = HttpClient::new(client_config)?;
        Ok(Self {
            config,
            client,
            cached_dimension: Mutex::new(None),
        })
    }

    /// Returns the configuration this client was built from.
    #[must_use]
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Returns the resolved embeddings endpoint.
    ///
    /// A `base_url` already ending in `/embeddings` is used unchanged; otherwise
    /// trailing slashes are trimmed and `/embeddings` is appended.
    #[must_use]
    pub fn endpoint(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/embeddings") {
            base.to_owned()
        } else {
            format!("{base}/embeddings")
        }
    }

    /// Builds one embeddings request for `input`.
    fn build_request(&self, input: Value) -> Result<HttpRequest> {
        let mut body = serde_json::Map::new();
        body.insert(
            "model".to_owned(),
            Value::String(self.config.model_name.clone()),
        );
        body.insert("input".to_owned(), input);
        if let Some(dimension) = self.config.dimension {
            body.insert("dimensions".to_owned(), json!(dimension));
        }
        let mut request = HttpRequest::post(self.endpoint()).json(&Value::Object(body))?;
        if let Some(api_key) = self.config.api_key.as_ref().filter(|key| !key.is_empty()) {
            // The secret crosses into the header builder here and nowhere else.
            request = request.bearer(api_key.clone());
        }
        Ok(request)
    }

    /// Sends one request and parses exactly `expected` vectors out of the response.
    fn parse_response(&self, response: &HttpResponse, expected: usize) -> Result<Vec<Vec<f32>>> {
        if !response.is_success() {
            let body = response.text();
            let snippet: String = body.chars().take(MAX_ERROR_BODY_CHARS).collect();
            return Err(Error::Retrieval(format!(
                "embedding request to {} failed with HTTP {}: {}",
                self.endpoint(),
                response.status,
                snippet.trim()
            )));
        }
        let payload: Value = response.json().map_err(|error| {
            Error::Retrieval(format!(
                "decoding embedding response from {}: {error}",
                self.endpoint()
            ))
        })?;
        let vectors = extract_vectors(&payload)?;
        if vectors.len() != expected {
            return Err(Error::Retrieval(format!(
                "embedding response from {} contained {} vectors for {} inputs",
                self.endpoint(),
                vectors.len(),
                expected
            )));
        }
        if let Some(first) = vectors.first() {
            *self.cached_dimension.lock() = Some(first.len());
        }
        Ok(vectors)
    }

    /// Resolves the batch size for a document request.
    fn resolve_batch_size(&self, batch_size: Option<usize>) -> usize {
        let configured = self.config.max_batch_size.max(1);
        batch_size.unwrap_or(configured).max(1).min(configured)
    }
}

#[async_trait]
impl Embedding for OpenAiEmbedding {
    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    fn dimension_hint(&self) -> Option<usize> {
        self.config
            .dimension
            .or_else(|| *self.cached_dimension.lock())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        validate_query(text)?;
        let request = self.build_request(Value::String(text.to_owned()))?;
        let response = self.client.execute(request).await?;
        let mut vectors = self.parse_response(&response, 1)?;
        vectors
            .pop()
            .ok_or_else(|| Error::Retrieval("embedding response contained no vector".to_owned()))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        validate_texts(texts)?;
        let batch_size = self.resolve_batch_size(batch_size);
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(batch_size) {
            let input = Value::Array(batch.iter().cloned().map(Value::String).collect());
            let request = self.build_request(input)?;
            let response = self.client.execute(request).await?;
            vectors.extend(self.parse_response(&response, batch.len())?);
        }
        Ok(vectors)
    }
}

/// Extracts every vector from an embeddings response payload.
///
/// Shape precedence mirrors the Python client: `data[].embedding` first (ordered
/// by `index` when every entry carries one), then a top-level `embedding`, then a
/// top-level `embeddings`.
fn extract_vectors(payload: &Value) -> Result<Vec<Vec<f32>>> {
    if let Some(data) = payload.get("data") {
        let entries = data.as_array().ok_or_else(|| {
            Error::Retrieval("embedding response field `data` must be an array".to_owned())
        })?;
        return vectors_from_entries(entries);
    }
    if let Some(embedding) = payload.get("embedding") {
        return flatten_vectors(embedding, "embedding");
    }
    if let Some(embeddings) = payload.get("embeddings") {
        return flatten_vectors(embeddings, "embeddings");
    }
    Err(Error::Retrieval(
        "embedding response contained none of `data`, `embedding`, or `embeddings`".to_owned(),
    ))
}

/// Parses the `OpenAI` `data` array, honouring `index` when it is present everywhere.
fn vectors_from_entries(entries: &[Value]) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(entries.len());
    let mut indices = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let vector = entry
            .get("embedding")
            .and_then(numbers_to_vector)
            .ok_or_else(|| {
                Error::Retrieval(format!(
                    "embedding response entry {position} has no numeric `embedding` array"
                ))
            })?;
        vectors.push(vector);
        indices.push(entry.get("index").and_then(Value::as_i64));
    }
    if !vectors.is_empty() && indices.iter().all(Option::is_some) {
        let mut ordered: Vec<(i64, Vec<f32>)> =
            indices.into_iter().flatten().zip(vectors).collect();
        ordered.sort_by_key(|(index, _)| *index);
        return Ok(ordered.into_iter().map(|(_, vector)| vector).collect());
    }
    Ok(vectors)
}

/// Parses a single vector (`[0.1, 0.2]`) or a list of vectors (`[[0.1], [0.2]]`).
fn flatten_vectors(value: &Value, field: &str) -> Result<Vec<Vec<f32>>> {
    let array = value.as_array().ok_or_else(|| {
        Error::Retrieval(format!(
            "embedding response field `{field}` must be an array"
        ))
    })?;
    if array.is_empty() {
        return Ok(Vec::new());
    }
    if array.iter().all(Value::is_number) {
        return numbers_to_vector(value)
            .map(|vector| vec![vector])
            .ok_or_else(|| {
                Error::Retrieval(format!(
                    "embedding response field `{field}` contains a non-numeric component"
                ))
            });
    }
    array
        .iter()
        .map(|item| {
            numbers_to_vector(item).ok_or_else(|| {
                Error::Retrieval(format!(
                    "embedding response field `{field}` contains a non-numeric entry"
                ))
            })
        })
        .collect()
}

/// Reads a JSON array of numbers into an `f32` vector.
fn numbers_to_vector(value: &Value) -> Option<Vec<f32>> {
    let array = value.as_array()?;
    let mut vector = Vec::with_capacity(array.len());
    for component in array {
        vector.push(component.as_f64()? as f32);
    }
    Some(vector)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
