//! Test-only helpers shared by the web tool suites.
//!
// Test-only helpers: unwrap/expect are the assertion vocabulary of this module.
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Two seams make the web tools testable offline: [`StubFetcher`] injects canned responses and
//! failures through [`HttpFetcher`], and [`TestServer`] is a loopback HTTP server that exercises
//! the real `reqwest` path without leaving the machine.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinHandle;

use openjiuwen_common::error::{Error, Result};

use super::fetch::{FetchRequest, FetchResponse, HttpFetcher};

/// An [`HttpFetcher`] returning canned responses and recording the requests it received.
pub(crate) struct StubFetcher {
    /// The responses handed out, in call order.
    responses: Mutex<std::collections::VecDeque<Result<FetchResponse>>>,
    /// Number of calls made.
    calls: AtomicUsize,
    /// The request of the last call.
    last_request: Mutex<Option<FetchRequest>>,
}

impl StubFetcher {
    /// Builds a stub answer with `response` once.
    pub(crate) fn new(response: Result<FetchResponse>) -> Arc<Self> {
        Self::sequence(vec![response])
    }

    /// Builds a stub answering its calls with `responses` in order.
    pub(crate) fn sequence(responses: Vec<Result<FetchResponse>>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into_iter().collect()),
            calls: AtomicUsize::new(0),
            last_request: Mutex::new(None),
        })
    }

    /// Builds a stub returning a `200` response with `body`.
    pub(crate) fn ok(content_type: &str, body: impl Into<Vec<u8>>) -> Arc<Self> {
        Self::new(Ok(fetch_response(200, content_type, body)))
    }

    /// Builds a stub failing with `error`.
    pub(crate) fn failing(error: Error) -> Arc<Self> {
        Self::new(Err(error))
    }

    /// The number of calls made so far.
    pub(crate) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// The request of the last call.
    pub(crate) fn last_request(&self) -> Option<FetchRequest> {
        self.last_request.lock().clone()
    }
}

#[async_trait]
impl HttpFetcher for StubFetcher {
    async fn fetch(&self, request: FetchRequest) -> Result<FetchResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_request.lock() = Some(request);
        self.responses.lock().pop_front().unwrap_or_else(|| {
            Err(Error::tool(
                "stub fetcher was called more times than it has responses",
            ))
        })
    }
}

/// Builds a [`FetchResponse`] with one `Content-Type` header and the given body.
pub(crate) fn fetch_response(
    status: u16,
    content_type: &str,
    body: impl Into<Vec<u8>>,
) -> FetchResponse {
    let headers = if content_type.is_empty() {
        Vec::new()
    } else {
        vec![("Content-Type".to_owned(), content_type.to_owned())]
    };
    FetchResponse {
        status,
        final_url: "https://example.com/final".to_owned(),
        headers,
        body: body.into(),
    }
}

/// A loopback HTTP server answering every connection with `handler(request_head)`.
pub(crate) struct TestServer {
    /// The `http://127.0.0.1:PORT` base URL.
    base_url: String,
    /// The accept loop; aborted on drop.
    task: JoinHandle<()>,
}

impl TestServer {
    /// Binds an ephemeral loopback port and starts serving.
    pub(crate) async fn start<F>(handler: F) -> Self
    where
        F: Fn(&str) -> Vec<u8> + Send + Sync + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback listener");
        let address = listener.local_addr().expect("read the local address");
        let handler = Arc::new(handler);
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    let mut buffer = Vec::new();
                    let mut chunk = [0u8; 2048];
                    loop {
                        match socket.read(&mut chunk).await {
                            Ok(0) => break,
                            Ok(read) => {
                                buffer.extend_from_slice(&chunk[..read]);
                                if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                                    break;
                                }
                            }
                            Err(_) => return,
                        }
                    }
                    let head = String::from_utf8_lossy(&buffer).into_owned();
                    let response = handler(&head);
                    let _ = socket.write_all(&response).await;
                    let _ = socket.flush().await;
                });
            }
        });
        Self {
            base_url: format!("http://{address}"),
            task,
        }
    }

    /// The absolute URL of `path` on this server.
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Builds a complete HTTP/1.1 response with `status`, one `Content-Type` header and `body`.
pub(crate) fn http_response(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut head = format!("HTTP/1.1 {status} {}\r\n", reason(status));
    if !content_type.is_empty() {
        head.push_str("Content-Type: ");
        head.push_str(content_type);
        head.push_str("\r\n");
    }
    head.push_str("Content-Length: ");
    head.push_str(&body.len().to_string());
    head.push_str("\r\nConnection: close\r\n\r\n");
    let mut response = head.into_bytes();
    response.extend_from_slice(body);
    response
}

/// The reason phrase of the statuses the suites use.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        302 => "Found",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Status",
    }
}
