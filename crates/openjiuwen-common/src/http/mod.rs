//! Provider-agnostic asynchronous HTTP transport.
//!
//! This module is the shared transport for every openJiuwen crate; the Python
//! SDK's connector pools, session managers, and `ClientRegistry` are deliberately
//! not ported. [`HttpClient`] adds retry/backoff on top of [`reqwest`] because the
//! LLM layers require it, and it never raises on non-2xx responses — call
//! [`HttpResponse::error_for_status`] when a strict check is wanted.
//!
//! - [`HttpClientConfig`] — timeouts, retry policy, default headers, proxy, TLS.
//! - [`RetryPolicy`] — retry classification and exponential backoff.
//! - [`HttpRequest`] — builder for method, URL, query, headers, body, bearer token,
//!   timeout, and cancellation.
//! - [`HttpResponse`] / [`HttpStreamResponse`] — buffered and streaming responses.
//!
//! ```no_run
//! use openjiuwen_common::error::Result;
//! use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest};
//!
//! # async fn example() -> Result<()> {
//! let client = HttpClient::new(HttpClientConfig::default().with_max_retries(2))?;
//! let response = client
//!     .execute(HttpRequest::get("https://api.example.com/v1/models"))
//!     .await?;
//! println!("{}", response.status);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod config;
pub mod retry;

pub use client::{HttpBody, HttpClient, HttpRequest, HttpResponse, HttpStream, HttpStreamResponse};
pub use config::HttpClientConfig;
pub use reqwest::Method;
pub use retry::RetryPolicy;
