//! `HTTP(S)` loader for web pages and plain-text resources.
//!
//! Port of the Python `WebPageParser` (`web_page_parser.py`): the URL is fetched
//! with a browser-like `User-Agent`, non-2xx responses fail, and `HTML` bodies are
//! reduced to text with [`html_to_text`]. Python's readability-style main-content
//! extraction is not ported; the whole page becomes one document.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};

use crate::common::Document;

use super::{extract_title, html_to_text, is_url, DocumentLoader};

/// Default request timeout, matching Python's `DEFAULT_TIMEOUT = 30.0`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Browser-like `User-Agent`, matching Python's `DEFAULT_USER_AGENT`.
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                          (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/// Loads web pages and plain-text resources over `HTTP(S)`.
///
/// A caller-supplied [`HttpClient`] (proxy, retries, TLS policy) is used when
/// present; otherwise a default client is built lazily on each load, because
/// [`UrlLoader::new`] cannot fail.
#[derive(Debug, Clone)]
pub struct UrlLoader {
    /// Optional caller-supplied transport; `None` means "build the default one".
    client: Option<HttpClient>,
    /// Per-request timeout, also applied to the default client.
    timeout: Duration,
}

impl UrlLoader {
    /// Creates a loader with the default 30-second timeout and Chrome-120 agent.
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Reuses `client` for every request instead of building a default one.
    #[must_use]
    pub fn with_client(mut self, client: HttpClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Overrides the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Returns the caller-supplied client or builds the default one.
    fn http_client(&self) -> Result<HttpClient> {
        if let Some(client) = &self.client {
            return Ok(client.clone());
        }
        HttpClient::new(
            HttpClientConfig::default()
                .with_timeout(self.timeout)
                .with_verify_ssl(true)
                .with_user_agent(USER_AGENT),
        )
    }

    /// Builds the `GET` request for `url`.
    fn build_request(&self, url: &str) -> HttpRequest {
        HttpRequest::get(url)
            .header("User-Agent", USER_AGENT)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            )
            .timeout(self.timeout)
    }
}

impl Default for UrlLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentLoader for UrlLoader {
    fn supports(&self, source: &str) -> bool {
        is_url(source)
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        if !is_url(source) {
            return Err(Error::Unsupported(format!(
                "Not a valid HTTP URL: {source}"
            )));
        }
        let url = source.trim();
        let client = self.http_client()?;
        let response = client.execute(self.build_request(url)).await?;
        if !response.is_success() {
            return Err(Error::Retrieval(format!(
                "Web page request failed: {} for {url}",
                response.status
            )));
        }

        let body = response.text();
        let is_html = content_type(&response)
            .is_some_and(|value| value.to_ascii_lowercase().contains("html"))
            || body.trim_start().starts_with('<');
        let text = if is_html {
            html_to_text(&body)
        } else {
            body.trim().to_owned()
        };
        if text.is_empty() {
            return Err(Error::Validation(format!(
                "No text content found in web page: {url}"
            )));
        }

        // A plain-text body has no title element, so this is empty for it.
        let title = extract_title(&body).unwrap_or_default();
        let mut metadata = Map::new();
        metadata.insert("source_url".to_owned(), Value::String(url.to_owned()));
        metadata.insert("title".to_owned(), Value::String(title));
        metadata.insert(
            "source_type".to_owned(),
            Value::String("web_page".to_owned()),
        );
        Ok(vec![Document {
            id: url.to_owned(),
            text,
            metadata,
        }])
    }
}

/// Returns the response `Content-Type` header value when it is valid `ASCII`.
fn content_type(response: &HttpResponse) -> Option<&str> {
    response
        .headers
        .iter()
        .find(|(name, _)| name.as_str().eq_ignore_ascii_case("content-type"))
        .and_then(|(_, value)| value.to_str().ok())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn supports_only_http_schemes() {
        let loader = UrlLoader::new();
        assert!(loader.supports("https://example.com/page"));
        assert!(loader.supports("HTTP://EXAMPLE.COM"));
        assert!(!loader.supports("ftp://example.com"));
        assert!(!loader.supports("/tmp/page.html"));
    }

    #[tokio::test]
    async fn non_http_sources_are_rejected_before_any_request() {
        let error = UrlLoader::new()
            .load("file:///tmp/page.html")
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("Not a valid HTTP URL"));
    }

    #[test]
    fn default_client_and_timeout_are_configured() {
        let loader = UrlLoader::new();
        assert_eq!(loader.timeout, DEFAULT_TIMEOUT);
        assert!(loader.http_client().is_ok());
        assert!(loader.supports("https://example.com"));
        assert!(USER_AGENT.contains("Chrome/120.0.0.0"));
        let custom = UrlLoader::new().with_timeout(Duration::from_secs(2));
        assert_eq!(custom.timeout, Duration::from_secs(2));
    }
}
