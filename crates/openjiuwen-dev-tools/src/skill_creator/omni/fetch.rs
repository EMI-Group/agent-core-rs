//! The fetching seam of the omni pipeline: one URL to one bounded resource.
//!
//! Ports the download half of the Python
//! `skill_omni_creation/scripts/stage_02_download.py` (`fetch_one`): a `GET`
//! with a timeout, a response-size cap, and the `content-type` preserved so
//! later stages can classify the payload. The Python module silently returns
//! `None` on every failure; here each failure is a typed
//! [`Error`] and the caller decides whether to
//! skip the URL.
//!
//! - [`FetchedResource`] — a buffered response: requested URL, final URL after
//!   redirects, status, `content-type`, and the body bytes.
//! - [`Fetcher`] — the object-safe seam the pipeline depends on.
//! - [`HttpFetcher`] — the real implementation over
//!   [`openjiuwen_common::http`], with a 8 MiB body cap and no retries beyond the
//!   client defaults.
//! - [`ScriptedFetcher`] — a programmable implementation for offline tests.
//!
//! Ordering invariants: an [`HttpFetcher`] builds its [`HttpClient`] once;
//! [`HttpFetcher::with_timeout`] rebuilds it so the timeout applies whatever the
//! builder order, while [`HttpFetcher::with_max_bytes`] only changes the cap
//! enforced in [`Fetcher::fetch`].

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest};
use serde::{Deserialize, Serialize};

/// The default response-body cap: 8 MiB, as in the Python download stage.
const DEFAULT_MAX_BYTES: u64 = 8 * 1024 * 1024;

/// How many leading body bytes are inspected when sniffing for HTML.
const SNIFF_BYTES: usize = 1024;

/// A buffered fetch result.
///
/// `url` is the URL that was requested, `final_url` the one that answered after
/// redirects; both are kept so a pipeline can resolve relative links against the
/// document it actually received.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FetchedResource {
    /// The URL that was requested.
    pub url: String,
    /// The final URL after redirects.
    pub final_url: String,
    /// The numeric HTTP status code.
    pub status: u16,
    /// The `content-type` header, verbatim (including any `charset` parameter).
    pub content_type: Option<String>,
    /// The raw response body.
    pub body: Vec<u8>,
}

impl FetchedResource {
    /// Decodes the body as UTF-8, replacing invalid sequences.
    #[must_use]
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Reports whether the payload is HTML.
    ///
    /// The `content-type` header wins when it names `text/html` or
    /// `application/xhtml…`, ignoring case and any `; charset=…` parameters.
    /// Otherwise the first `SNIFF_BYTES` bytes of the body are inspected for a
    /// `<!doctype html` prefix or an `<html` tag, again ignoring case.
    #[must_use]
    pub fn is_html(&self) -> bool {
        if let Some(content_type) = self.content_type.as_deref() {
            let mime = content_type
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            if mime.starts_with("text/html") || mime.starts_with("application/xhtml") {
                return true;
            }
        }
        sniff_html(&self.body)
    }

    /// Reports whether the status is 2xx.
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Reports whether the body head looks like an HTML document.
fn sniff_html(body: &[u8]) -> bool {
    let head = &body[..body.len().min(SNIFF_BYTES)];
    let lowered = String::from_utf8_lossy(head)
        .trim_start()
        .to_ascii_lowercase();
    lowered.starts_with("<!doctype html") || lowered.contains("<html")
}

/// The seam the omni pipeline fetches through.
///
/// Object-safe by construction (`dyn Fetcher` / `Arc<dyn Fetcher>`), so a
/// pipeline can be driven by [`HttpFetcher`] in production and by
/// [`ScriptedFetcher`] in tests. Implementations are also `Send + Sync` so a
/// fetch future can be moved across tasks.
#[async_trait::async_trait]
pub trait Fetcher: Send + Sync {
    /// Fetches `url` and buffers its body.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for a URL that is not `http`/`https`, and
    /// [`Error::transport`](openjiuwen_common::error::Error::transport) for a
    /// non-2xx response, a body over the configured cap, a connection failure, or
    /// a missing scripted response.
    async fn fetch(&self, url: &str) -> Result<FetchedResource>;
}

/// The production [`Fetcher`], backed by [`openjiuwen_common::http`].
///
/// The [`HttpClient`] is built once and shared across requests; redirects,
/// proxies, and retries follow [`HttpClientConfig`]'s defaults. The body cap is
/// enforced after buffering, so an oversized page fails with a message naming
/// both the cap and the URL instead of being silently truncated.
#[derive(Debug)]
pub struct HttpFetcher {
    /// The configuration the cached client was (or will be) built from.
    config: HttpClientConfig,
    /// The shared client, or the build failure surfaced by the next fetch.
    client: Result<Arc<HttpClient>>,
    /// The largest body accepted, in bytes.
    max_bytes: u64,
}

impl HttpFetcher {
    /// Builds a fetcher with [`HttpClientConfig::default`] and an 8 MiB cap.
    ///
    /// This is [`HttpClient::with_defaults`] plus a stored configuration, so a
    /// later [`with_timeout`](Self::with_timeout) can rebuild the client. A build
    /// failure never panics here: it is kept and returned by the first
    /// [`Fetcher::fetch`].
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(HttpClientConfig::default())
    }

    /// Builds a fetcher from an explicit configuration.
    fn from_config(config: HttpClientConfig) -> Self {
        let client = HttpClient::new(config.clone()).map(Arc::new);
        Self {
            config,
            client,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Sets the largest accepted response body, in bytes (default 8 MiB).
    ///
    /// The cap is checked after the body is buffered, which is what lets the
    /// error name both the cap and the URL.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Sets the total request timeout and rebuilds the cached client.
    ///
    /// The client is rebuilt eagerly, so the timeout applies regardless of the
    /// builder order; a rebuild failure is surfaced by the next
    /// [`Fetcher::fetch`] exactly like a failure in [`new`](Self::new).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self.client = HttpClient::new(self.config.clone()).map(Arc::new);
        self
    }
}

impl Default for HttpFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Fetcher for HttpFetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedResource> {
        validate_scheme(url)?;
        let client = self.client.as_ref().map_err(|error| {
            Error::transport(format!(
                "building the HTTP client failed for {url}: {error}"
            ))
        })?;

        let response = client.execute(HttpRequest::get(url)).await?;
        let status = response.status;
        if !response.is_success() {
            return Err(Error::transport(format!("HTTP {status} for {url}")));
        }

        let content_type = response
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = response.body.to_vec();
        let length = body.len() as u64;
        if length > self.max_bytes {
            return Err(Error::transport(format!(
                "response body for {url} is {length} bytes, over the {}-byte cap",
                self.max_bytes
            )));
        }

        Ok(FetchedResource {
            url: url.to_owned(),
            final_url: response.url,
            status,
            content_type,
            body,
        })
    }
}

/// Rejects any URL whose scheme is not `http` or `https`.
fn validate_scheme(url: &str) -> Result<()> {
    let scheme = url.split_once("://").map_or("", |(scheme, _)| scheme);
    if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "fetch URL must use the http or https scheme: {url}"
        )))
    }
}

/// A programmable [`Fetcher`] for offline tests.
///
/// Responses are keyed by URL; every [`Fetcher::fetch`] call records the
/// requested URL so a test can assert the pipeline's request order. Cloning is
/// cheap (the state is behind [`Arc`]), and a poisoned lock is recovered from
/// rather than propagated, so a failing assertion in one test thread cannot
/// wedge the fetcher.
#[derive(Debug, Clone, Default)]
pub struct ScriptedFetcher {
    /// The responses registered by [`insert`](Self::insert).
    responses: Arc<Mutex<BTreeMap<String, FetchedResource>>>,
    /// The requested URLs, in call order.
    calls: Arc<Mutex<Vec<String>>>,
}

impl ScriptedFetcher {
    /// Creates an empty fetcher.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `resource` for `url`, replacing any earlier entry.
    ///
    /// Empty [`FetchedResource::url`]/[`FetchedResource::final_url`] fields are
    /// filled from the requested URL when the resource is served.
    pub fn insert(&self, url: impl Into<String>, resource: FetchedResource) {
        lock(&self.responses).insert(url.into(), resource);
    }

    /// Registers a 200 response with the supplied `content-type` and body.
    pub fn with_page(&self, url: impl Into<String>, content_type: &str, body: impl Into<Vec<u8>>) {
        self.insert(
            url,
            FetchedResource {
                status: 200,
                content_type: Some(content_type.to_owned()),
                body: body.into(),
                ..FetchedResource::default()
            },
        );
    }

    /// Registers a 200 `text/html; charset=utf-8` response for `url`.
    pub fn with_html(&self, url: impl Into<String>, body: impl Into<Vec<u8>>) {
        self.with_page(url, "text/html; charset=utf-8", body);
    }

    /// Returns the requested URLs in call order.
    #[must_use]
    pub fn calls(&self) -> Vec<String> {
        lock(&self.calls).clone()
    }
}

#[async_trait::async_trait]
impl Fetcher for ScriptedFetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedResource> {
        lock(&self.calls).push(url.to_owned());
        let mut resource = lock(&self.responses).get(url).cloned().ok_or_else(|| {
            Error::transport(format!("no scripted response registered for {url}"))
        })?;
        if resource.url.is_empty() {
            url.clone_into(&mut resource.url);
        }
        if resource.final_url.is_empty() {
            url.clone_into(&mut resource.final_url);
        }
        Ok(resource)
    }
}

/// Locks `mutex`, recovering the guard when a previous holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

    use super::*;

    /// Serves exactly one HTTP response on a loopback listener.
    ///
    /// Returns the base URL and the server thread. The accept loop is bounded by
    /// a deadline so a fetch that fails before connecting cannot hang the test.
    fn spawn_server(
        status: &str,
        headers: &[(&str, &str)],
        body: Vec<u8>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let address = listener.local_addr().expect("local address");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let status = status.to_owned();
        let headers: Vec<(String, String)> = headers
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();

        let handle = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => return,
                }
            };
            // Accepted sockets may inherit non-blocking mode on some platforms.
            stream.set_nonblocking(false).expect("blocking stream");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("read timeout");

            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&buffer[..read]),
                }
            }

            let mut response = format!("HTTP/1.1 {status}\r\n");
            for (name, value) in headers {
                write!(response, "{name}: {value}\r\n").expect("header line");
            }
            write!(
                response,
                "Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .expect("length header");
            stream.write_all(response.as_bytes()).expect("headers");
            stream.write_all(&body).expect("body");
            stream.flush().expect("flush");
        });
        (format!("http://{address}/"), handle)
    }

    #[tokio::test]
    async fn http_fetcher_returns_a_page() {
        let body = b"<!doctype html><html><body>hello</body></html>".to_vec();
        let (url, server) = spawn_server(
            "200 OK",
            &[("Content-Type", "text/html; charset=utf-8")],
            body.clone(),
        );

        let resource =
            tokio::time::timeout(Duration::from_secs(10), HttpFetcher::new().fetch(&url))
                .await
                .expect("fetch timed out")
                .expect("fetch succeeds");

        assert_eq!(resource.status, 200);
        assert_eq!(
            resource.content_type.as_deref(),
            Some("text/html; charset=utf-8")
        );
        assert!(resource.is_html());
        assert_eq!(resource.text(), String::from_utf8(body).expect("utf-8"));
        assert!(resource.is_success());
        assert_eq!(resource.url, url);
        assert_eq!(resource.final_url, url);

        server.join().expect("server thread");
    }

    #[tokio::test]
    async fn http_fetcher_reports_non_success_status() {
        let (url, server) = spawn_server(
            "404 Not Found",
            &[("Content-Type", "text/plain")],
            b"missing".to_vec(),
        );

        let error = tokio::time::timeout(Duration::from_secs(10), HttpFetcher::new().fetch(&url))
            .await
            .expect("fetch timed out")
            .expect_err("404 is an error");

        let message = error.to_string();
        assert!(message.contains("404"), "{message}");
        assert!(message.contains(&url), "{message}");

        server.join().expect("server thread");
    }

    #[tokio::test]
    async fn http_fetcher_enforces_the_size_cap() {
        let (url, server) = spawn_server(
            "200 OK",
            &[("Content-Type", "application/octet-stream")],
            vec![b'x'; 64],
        );

        let error = tokio::time::timeout(
            Duration::from_secs(10),
            HttpFetcher::new().with_max_bytes(16).fetch(&url),
        )
        .await
        .expect("fetch timed out")
        .expect_err("body exceeds the cap");

        let message = error.to_string();
        assert!(message.contains("16-byte cap"), "{message}");
        assert!(message.contains(&url), "{message}");

        server.join().expect("server thread");
    }

    #[tokio::test]
    async fn http_fetcher_accepts_a_body_at_the_cap() {
        let (url, server) = spawn_server(
            "200 OK",
            &[("Content-Type", "application/octet-stream")],
            vec![b'y'; 16],
        );

        let resource = tokio::time::timeout(
            Duration::from_secs(10),
            HttpFetcher::new()
                .with_max_bytes(16)
                .with_timeout(Duration::from_secs(5))
                .fetch(&url),
        )
        .await
        .expect("fetch timed out")
        .expect("body is at the cap");

        assert_eq!(resource.body.len(), 16);
        assert!(resource.is_success());
        assert!(!resource.is_html());

        server.join().expect("server thread");
    }

    #[tokio::test]
    async fn http_fetcher_rejects_non_http_schemes() {
        let fetcher = HttpFetcher::new();
        for url in ["ftp://example.com/file", "file:///tmp/x"] {
            let error = fetcher.fetch(url).await.expect_err("scheme is rejected");
            let message = error.to_string();
            assert!(message.starts_with("validation:"), "{message}");
            assert!(message.contains(url), "{message}");
        }
    }

    #[tokio::test]
    async fn scripted_fetcher_serves_and_records_requests() {
        let fetcher = ScriptedFetcher::new();
        fetcher.with_html("https://example.com/a", b"<html>a</html>".to_vec());
        fetcher.insert(
            "https://example.com/b",
            FetchedResource {
                url: "https://example.com/b-final".to_owned(),
                status: 204,
                ..FetchedResource::default()
            },
        );

        let first = fetcher.fetch("https://example.com/a").await.expect("a");
        assert_eq!(first.url, "https://example.com/a");
        assert_eq!(first.final_url, "https://example.com/a");
        assert_eq!(first.status, 200);
        assert!(first.is_html());

        let second = fetcher.fetch("https://example.com/b").await.expect("b");
        assert_eq!(second.url, "https://example.com/b-final");
        assert_eq!(second.final_url, "https://example.com/b");
        assert_eq!(second.status, 204);

        assert_eq!(
            fetcher.calls(),
            vec![
                "https://example.com/a".to_owned(),
                "https://example.com/b".to_owned()
            ]
        );

        let error = fetcher
            .fetch("https://example.com/missing")
            .await
            .expect_err("unknown URL");
        assert!(error.to_string().contains("missing"), "{error}");
        assert_eq!(fetcher.calls().len(), 3);
    }

    #[tokio::test]
    async fn fetcher_is_object_safe_and_cheap_to_clone() {
        let scripted = ScriptedFetcher::new();
        scripted.with_html("https://example.com", b"ok".to_vec());
        let cloned = scripted.clone();
        let fetcher: Arc<dyn Fetcher> = Arc::new(cloned);

        let resource = fetcher.fetch("https://example.com").await.expect("fetch");
        assert_eq!(resource.text(), "ok");
        assert_eq!(scripted.calls(), vec!["https://example.com".to_owned()]);
    }

    #[test]
    fn fetched_resource_defaults_and_success_range() {
        let mut resource = FetchedResource::default();
        assert!(!resource.is_success());
        assert!(!resource.is_html());
        assert_eq!(resource.text(), "");

        resource.status = 199;
        assert!(!resource.is_success());
        resource.status = 200;
        assert!(resource.is_success());
        resource.status = 299;
        assert!(resource.is_success());
        resource.status = 300;
        assert!(!resource.is_success());
    }

    #[test]
    fn is_html_matrix() {
        let mut resource = FetchedResource::default();
        assert!(!resource.is_html());

        resource.content_type = Some("TEXT/HTML; charset=ISO-8859-1".to_owned());
        assert!(resource.is_html());

        resource.content_type = Some("application/xhtml+xml".to_owned());
        assert!(resource.is_html());

        resource.content_type = Some("application/pdf".to_owned());
        resource.body = b"%PDF-1.7".to_vec();
        assert!(!resource.is_html());

        resource.content_type = None;
        resource.body = b"  <!DOCTYPE html><html></html>".to_vec();
        assert!(resource.is_html());

        resource.body = b"<HTML><body>sniffed</body></HTML>".to_vec();
        assert!(resource.is_html());

        resource.body = b"plain text".to_vec();
        assert!(!resource.is_html());
    }
}
