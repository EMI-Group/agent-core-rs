//! The REST tool itself: request building, invocation, and response envelopes.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse, Method};
use reqwest::header::HeaderMap;
use serde_json::{json, Value};
use url::Url;

use super::config::RestfulApiConfig;
use super::mapper::{
    quoted_set, value_to_text, ApiParamMapper, ParamLocation, ParamMap, PLACEHOLDER,
};
use crate::tool::card::ToolCard;
use crate::tool::context::ToolContext;
use crate::tool::schema::{apply_defaults, ensure_object_schema, validate_inputs};
use crate::tool::traits::Tool;

/// A tool that calls an HTTP endpoint and returns Python's response envelope.
///
/// Python parity: `RestfulApi` in
/// `openjiuwen/core/foundation/tool/service_api/restful_api.py`. Invocations
/// return the envelope object `{"code", "message", "data", "url", "headers"}`
/// and never stream: [`Tool::stream`] reports [`Error::Unsupported`], matching
/// Python's `TOOL_STREAM_NOT_SUPPORTED`.
#[derive(Debug, Clone)]
pub struct RestfulApiTool {
    card: ToolCard,
    config: RestfulApiConfig,
    mapper: ApiParamMapper,
    http: HttpClient,
}

impl RestfulApiTool {
    /// Builds a REST tool from a card and a transport configuration.
    ///
    /// The card's `input_params` schema declares the accepted arguments and their
    /// locations; the config carries the url, method, defaults, timeout, and
    /// response cap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the url is not an absolute `http`/
    /// `https` url, or when a `{name}` placeholder in the url is not declared as
    /// `"location": "path"` in the card's schema. Path properties that the url
    /// does not use are only logged at [`tracing::warn`] level, as in Python.
    pub fn new(card: ToolCard, config: RestfulApiConfig) -> Result<Self> {
        validate_absolute_url(&config.url)?;

        let placeholders = url_placeholders(&config.url);
        let path_properties = path_properties(&card.input_params);
        let missing: Vec<&str> = placeholders
            .iter()
            .filter(|name| !path_properties.contains(*name))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(Error::validation(placeholder_message(
                &card.input_params,
                &missing,
            )));
        }

        let unused: Vec<&str> = path_properties
            .iter()
            .filter(|name| !placeholders.contains(*name))
            .map(String::as_str)
            .collect();
        if !unused.is_empty() {
            tracing::warn!(
                target: "openjiuwen_foundation::tool::restful",
                url = %config.url,
                parameters = ?unused,
                "schema declares path parameters that the url does not use"
            );
        }

        let mapper = ApiParamMapper::with_defaults(
            card.input_params.clone(),
            config.queries.clone(),
            config.headers.clone(),
            config.paths.clone(),
        );
        // The client-level `max_response_bytes` is deliberately left unset: the
        // cap is enforced per call so an oversized body surfaces as
        // `Error::Tool` rather than as a transport error.
        let http = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;

        Ok(Self {
            card,
            config,
            mapper,
            http,
        })
    }

    /// Returns the transport configuration of this tool.
    #[must_use]
    pub fn config(&self) -> &RestfulApiConfig {
        &self.config
    }

    /// Builds the HTTP request for already-mapped `params`.
    fn build_request(
        &self,
        url: &str,
        params: &ParamMap,
        ctx: &ToolContext,
    ) -> Result<HttpRequest> {
        let method = self.config.method;
        let mut request = HttpRequest::new(Method::from(method), url)
            .with_headers(header_pairs(params))
            .queries(query_pairs(params));

        if method.is_bodyless() {
            request = request.queries(
                params
                    .get(&ParamLocation::Body)
                    .map_or_else(Vec::new, query_pairs_for_body),
            );
        } else {
            let body = params.get(&ParamLocation::Body).map_or_else(
                || json!({}),
                |body| Value::Object(body.clone().into_iter().collect()),
            );
            request = request.json(&body)?;
        }

        Ok(request
            .timeout(self.config.timeout)
            .cancel(ctx.cancellation.clone()))
    }

    /// Turns a successful response into the Python envelope.
    fn envelope(response: HttpResponse) -> Result<Value> {
        let message = if response.is_success() {
            "success".to_owned()
        } else {
            canonical_reason(response.status).unwrap_or_default()
        };
        let content_type = response
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let data = if is_json_content_type(content_type) {
            if response.body.is_empty() {
                json!({})
            } else {
                serde_json::from_slice::<Value>(&response.body).map_err(|error| {
                    Error::tool(format!(
                        "restful api response from {} is not valid json: {error}",
                        response.url
                    ))
                })?
            }
        } else {
            Value::String(response.text())
        };
        let response_headers = header_object(&response.headers);

        Ok(json!({
            "code": response.status,
            "message": message,
            "data": data,
            "url": response.url,
            "headers": response_headers,
        }))
    }
}

#[async_trait]
impl Tool for RestfulApiTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;

        // Python `SchemaUtils.format_with_schema`: fill schema defaults, then
        // validate the resulting object against the same schema.
        let schema = ensure_object_schema(self.card.input_params.clone());
        let mut inputs = inputs;
        apply_defaults(&schema, &mut inputs);
        validate_inputs(&schema, &inputs)?;

        let params = self.mapper.map(&inputs)?;
        let url = self.mapper.render_url(&self.config.url, &params)?;
        let request = self.build_request(&url, &params, ctx)?;

        let response = match self.http.execute(request).await {
            Ok(response) => response,
            Err(error) if matches!(error, Error::Timeout(_) | Error::Cancelled) => {
                return Err(error)
            }
            Err(error) => {
                return Err(Error::tool(format!(
                    "restful api request to {} failed: {error}",
                    self.config.url
                )))
            }
        };

        if !response.is_success() {
            return Err(Error::tool(format!(
                "restful api {} {} returned HTTP {}{}",
                self.config.method.as_str(),
                response.url,
                response.status,
                body_excerpt(&response.text())
            )));
        }

        let length = response.body.len() as u64;
        if length > self.config.max_response_bytes {
            return Err(Error::tool(format!(
                "restful api response from {} is {length} bytes, exceeding the {} byte limit",
                response.url, self.config.max_response_bytes
            )));
        }

        Self::envelope(response)
    }

    fn stream(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: ToolContext,
    ) -> BoxStream<'static, Result<Value>> {
        Box::pin(futures::stream::once(async {
            Err(Error::unsupported(
                "restful api tools do not stream".to_owned(),
            ))
        }))
    }
}

/// Collects the `{name}` placeholders of `url`.
fn url_placeholders(url: &str) -> BTreeSet<String> {
    PLACEHOLDER
        .captures_iter(url)
        .filter_map(|captures| captures.get(1))
        .map(|matched| matched.as_str().to_owned())
        .collect()
}

/// Collects the schema properties declared as `"location": "path"`.
fn path_properties(schema: &Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return names;
    };
    for (name, property_schema) in properties {
        if property_schema
            .get("location")
            .and_then(Value::as_str)
            .is_some_and(|location| location.eq_ignore_ascii_case("path"))
        {
            names.insert(name.clone());
        }
    }
    names
}

/// Builds the Python-shaped message for undeclared url placeholders.
fn placeholder_message(schema: &Value, missing: &[&str]) -> String {
    let first = missing.first().copied().unwrap_or_default();
    let declared = schema
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|properties| !properties.is_empty());

    if declared {
        format!(
            "URL contains path parameters {} that are not defined in input_params schema with 'location': 'path'. \
             Please add them to your schema. Example: '{first}': {{'type': 'string', 'description': 'Parameter description', 'location': 'path'}}",
            quoted_set(missing)
        )
    } else {
        format!(
            "URL contains path parameters {} but input_params schema is not defined. \
             You must define input_params with 'location': 'path' for each path parameter. \
             Example: {{'type': 'object', 'properties': {{'{first}': {{'type': 'string', 'location': 'path'}}}}}}",
            quoted_set(missing)
        )
    }
}

/// Checks that `url` is an absolute `http`/`https` url.
fn validate_absolute_url(url: &str) -> Result<()> {
    let parsed = Url::parse(url)
        .map_err(|error| Error::validation(format!("invalid restful api url '{url}': {error}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(Error::validation(format!(
            "invalid restful api url '{url}': an absolute http or https url is required"
        )));
    }
    Ok(())
}

/// Converts the header bucket into request header pairs.
fn header_pairs(params: &ParamMap) -> Vec<(String, String)> {
    params
        .get(&ParamLocation::Header)
        .map_or_else(Vec::new, |headers| {
            headers
                .iter()
                .map(|(name, value)| (name.clone(), value_to_text(value)))
                .collect()
        })
}

/// Converts the query bucket into request query pairs.
///
/// Arrays expand to one parameter per item (Python `urlencode` behavior).
fn query_pairs(params: &ParamMap) -> Vec<(String, String)> {
    params
        .get(&ParamLocation::Query)
        .map_or_else(Vec::new, query_pairs_for_body)
}

/// Converts one bucket into request query pairs, expanding arrays.
fn query_pairs_for_body(bucket: &BTreeMap<String, Value>) -> Vec<(String, String)> {
    let mut pairs = Vec::with_capacity(bucket.len());
    for (name, value) in bucket {
        match value {
            Value::Array(items) => {
                pairs.extend(items.iter().map(|item| (name.clone(), value_to_text(item))));
            }
            other => pairs.push((name.clone(), value_to_text(other))),
        }
    }
    pairs
}

/// Returns whether `content_type` denotes a JSON body.
///
/// Python parity: `ParserRegistry.parse` consults the lower-cased
/// `content-type`; `JsonResponseParser.can_parse` accepts `application/json`,
/// `text/json`, `text/x-json`, `application/javascript`, and any RFC 6839
/// `+json` suffix. A `charset` parameter is ignored.
fn is_json_content_type(content_type: &str) -> bool {
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    matches!(
        essence.as_str(),
        "application/json" | "text/json" | "text/x-json" | "application/javascript"
    ) || essence.ends_with("+json")
}

/// Returns the canonical reason phrase of `status`, when it is a known status.
fn canonical_reason(status: u16) -> Option<String> {
    reqwest::StatusCode::from_u16(status)
        .ok()
        .and_then(|status| status.canonical_reason())
        .map(str::to_owned)
}

/// Renders the response headers as a JSON object, joining duplicates with `", "`.
fn header_object(headers: &HeaderMap) -> Value {
    let mut joined: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in headers {
        let rendered = String::from_utf8_lossy(value.as_bytes()).into_owned();
        joined
            .entry(name.as_str().to_ascii_lowercase())
            .and_modify(|existing| {
                existing.push_str(", ");
                existing.push_str(&rendered);
            })
            .or_insert(rendered);
    }
    Value::Object(
        joined
            .into_iter()
            .map(|(name, value)| (name, Value::String(value)))
            .collect(),
    )
}

/// Renders `": <excerpt>"` for an error message, or nothing for an empty body.
fn body_excerpt(body: &str) -> String {
    const MAX_CHARS: usize = 200;

    let trimmed = body.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut excerpt: String = trimmed.chars().take(MAX_CHARS).collect();
    if trimmed.chars().count() > MAX_CHARS {
        excerpt.push('…');
    }
    format!(": {excerpt}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::config::HttpMethod;
    use super::*;
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use wiremock::matchers::{body_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn bucket(params: &ParamMap, location: ParamLocation) -> Value {
        serde_json::to_value(params.get(&location).cloned().unwrap_or_default()).unwrap()
    }

    fn card(schema: Value) -> ToolCard {
        ToolCard::new("call_api", "calls an api", schema)
    }

    fn tool(url: String, method: HttpMethod, schema: Value) -> RestfulApiTool {
        RestfulApiTool::new(card(schema), RestfulApiConfig::new(url).with_method(method)).unwrap()
    }

    // port of tests/unit_tests/core/foundation/tool/test_restfulapi.py

    #[tokio::test]
    async fn post_routes_parameters_and_returns_the_json_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/users/42"))
            .and(query_param("q", "hello"))
            .and(header("token", "secret"))
            .and(body_json(json!({"name": "bob"})))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(r#"{"ok": true}"#, "application/json"),
            )
            .mount(&server)
            .await;

        let schema = json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "location": "path"},
                "q": {"type": "string", "location": "query"},
                "token": {"type": "string", "location": "header"},
                "name": {"type": "string", "location": "body"},
            },
        });
        let tool = tool(
            format!("{}/users/{{id}}", server.uri()),
            HttpMethod::Post,
            schema,
        );

        let envelope = tool
            .invoke(
                json!({"id": 42, "q": "hello", "token": "secret", "name": "bob"}),
                &ToolContext::new(),
            )
            .await
            .unwrap();

        assert_eq!(envelope["code"], json!(200));
        assert_eq!(envelope["message"], json!("success"));
        assert_eq!(envelope["data"], json!({"ok": true}));
        assert_eq!(
            envelope["url"],
            json!(format!("{}/users/42?q=hello", server.uri()))
        );
        assert_eq!(
            envelope["headers"]["content-type"],
            json!("application/json")
        );

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.method.as_str(), "POST");
        assert_eq!(request.url.path(), "/users/42");
        assert_eq!(
            request
                .headers
                .get("token")
                .map(|value| value.to_str().unwrap()),
            Some("secret")
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({"name": "bob"})
        );
    }

    #[tokio::test]
    async fn get_sends_body_parameters_as_query_pairs_and_returns_text() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "rust"))
            .and(query_param("page", "2"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/plain")
                    .set_body_string("plain text"),
            )
            .mount(&server)
            .await;

        let schema = json!({
            "type": "object",
            "properties": {
                "q": {"type": "string", "location": "query"},
                "page": {"type": "integer", "location": "body"},
            },
        });
        let tool = tool(format!("{}/search", server.uri()), HttpMethod::Get, schema);

        let envelope = tool
            .invoke(json!({"q": "rust", "page": 2}), &ToolContext::new())
            .await
            .unwrap();

        assert_eq!(envelope["code"], json!(200));
        assert_eq!(envelope["message"], json!("success"));
        assert_eq!(envelope["data"], json!("plain text"));

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].body.is_empty(),
            "a GET must not carry a request body"
        );
        assert!(requests[0].url.query().unwrap().contains("page=2"));
    }

    #[tokio::test]
    async fn json_suffix_content_types_are_parsed_as_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/items"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"{"items": [1, 2]}"#,
                "application/hal+json; charset=utf-8",
            ))
            .mount(&server)
            .await;

        let tool = tool(
            format!("{}/items", server.uri()),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        );
        let envelope = tool.invoke(json!({}), &ToolContext::new()).await.unwrap();

        assert_eq!(envelope["data"], json!({"items": [1, 2]}));
    }

    #[tokio::test]
    async fn non_success_statuses_are_tool_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/missing"))
            .respond_with(ResponseTemplate::new(404).set_body_string("no such endpoint"))
            .mount(&server)
            .await;

        let tool = tool(
            format!("{}/missing", server.uri()),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        );
        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Tool);
        assert!(error.to_string().contains("404"), "{error}");
        assert!(error.to_string().contains("no such endpoint"), "{error}");
    }

    #[tokio::test]
    async fn oversized_responses_are_tool_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/large"))
            .respond_with(ResponseTemplate::new(200).set_body_string("0123456789abcdef"))
            .mount(&server)
            .await;

        let config =
            RestfulApiConfig::new(format!("{}/large", server.uri())).with_max_response_bytes(8);
        let tool =
            RestfulApiTool::new(card(json!({"type": "object", "properties": {}})), config).unwrap();

        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Tool);
        assert!(error.to_string().contains("8 byte limit"), "{error}");
    }

    #[test]
    fn new_rejects_url_placeholders_without_path_properties() {
        let error = RestfulApiTool::new(
            card(json!({"type": "object", "properties": {"id": {"type": "string"}}})),
            RestfulApiConfig::new("http://localhost:9/users/{id}"),
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("'id'"), "{error}");
        assert!(error.to_string().contains("location"), "{error}");

        let no_schema = RestfulApiTool::new(
            card(json!({})),
            RestfulApiConfig::new("http://localhost:9/users/{id}"),
        )
        .unwrap_err();
        assert_eq!(no_schema.kind(), ErrorKind::Validation);
        assert!(
            no_schema.to_string().contains("is not defined"),
            "{no_schema}"
        );
    }

    #[tokio::test]
    async fn missing_path_values_are_validation_errors_at_invoke_time() {
        let tool = tool(
            "http://localhost:9/users/{id}".to_owned(),
            HttpMethod::Get,
            json!({"type": "object", "properties": {"id": {"type": "string", "location": "path"}}}),
        );

        let error = tool
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("'id'"), "{error}");
    }

    #[tokio::test]
    async fn new_rejects_relative_or_non_http_urls() {
        let relative = RestfulApiTool::new(
            card(json!({"type": "object", "properties": {}})),
            RestfulApiConfig::new("/api/v1/users"),
        )
        .unwrap_err();
        assert_eq!(relative.kind(), ErrorKind::Validation);

        let file = RestfulApiTool::new(
            card(json!({"type": "object", "properties": {}})),
            RestfulApiConfig::new("file:///tmp/api"),
        )
        .unwrap_err();
        assert_eq!(file.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn stream_is_unsupported() {
        let tool: Arc<dyn Tool> = Arc::new(tool(
            "http://localhost:9/users".to_owned(),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        ));

        let items: Vec<Result<Value>> = tool
            .clone()
            .stream(json!({"id": 1}), ToolContext::new())
            .collect()
            .await;

        assert_eq!(items.len(), 1);
        let error = items[0].as_ref().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[tokio::test]
    async fn cancelled_contexts_short_circuit_the_call() {
        let tool = tool(
            "http://localhost:9/users".to_owned(),
            HttpMethod::Post,
            json!({"type": "object", "properties": {}}),
        );
        let ctx = ToolContext::new();
        ctx.cancellation.cancel();

        let error = tool.invoke(json!({}), &ctx).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
    }

    #[test]
    fn config_defaults_are_merged_and_blank_inputs_keep_them() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "location": "query"},
                "lang": {"type": "string", "location": "query"},
            },
        });
        let mut queries = BTreeMap::new();
        queries.insert("lang".to_owned(), json!("en"));
        queries.insert("name".to_owned(), json!("default name"));
        let mut headers = BTreeMap::new();
        headers.insert("X-API-Key".to_owned(), json!("key"));
        let mut paths = BTreeMap::new();
        paths.insert("version".to_owned(), json!("v1"));

        let mapper = ApiParamMapper::with_defaults(schema, queries, headers, paths);

        let overridden = mapper.map(&json!({"name": "given", "lang": "zh"})).unwrap();
        assert_eq!(bucket(&overridden, ParamLocation::Query)["name"], "given");
        assert_eq!(bucket(&overridden, ParamLocation::Query)["lang"], "zh");

        let blank = mapper.map(&json!({"name": "", "lang": null})).unwrap();
        assert_eq!(
            bucket(&blank, ParamLocation::Query)["name"],
            json!("default name")
        );
        assert_eq!(bucket(&blank, ParamLocation::Query)["lang"], json!("en"));

        assert_eq!(bucket(&blank, ParamLocation::Header)["X-API-Key"], "key");
        assert_eq!(bucket(&blank, ParamLocation::Path)["version"], "v1");
        assert_eq!(bucket(&blank, ParamLocation::Body), json!({}));
        assert_eq!(
            mapper.defaults_for(ParamLocation::Query).map(BTreeMap::len),
            Some(2)
        );
        assert_eq!(
            mapper.schema(),
            &json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "location": "query"},
                    "lang": {"type": "string", "location": "query"},
                },
            })
        );
    }

    #[tokio::test]
    async fn query_arrays_expand_to_one_parameter_per_item() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/batch"))
            .and(query_param("ids", "1"))
            .and(query_param("ids", "2"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .expect(1)
            .mount(&server)
            .await;

        let schema = json!({
            "type": "object",
            "properties": {"ids": {"type": "array", "location": "query"}},
        });
        let tool = tool(format!("{}/batch", server.uri()), HttpMethod::Get, schema);

        let envelope = tool
            .invoke(json!({"ids": [1, 2]}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(envelope["data"], json!("ok"));

        let requests = server.received_requests().await.unwrap();
        let query = requests[0].url.query().unwrap();
        assert!(
            query.contains("ids=1") && query.contains("ids=2"),
            "{query}"
        );
    }
}
