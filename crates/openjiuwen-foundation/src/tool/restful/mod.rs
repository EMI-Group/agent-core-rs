//! REST tools: call an HTTP endpoint described by [`RestfulApiConfig`].
//!
//! Ports `openjiuwen/core/foundation/tool/service_api/restful_api.py`
//! (`RestfulApiCard`, `RestfulApi`), `api_param_mapper.py` (`APIParamMapper`,
//! `APIParamLocation`) and the JSON/text half of `response_parser.py`.
//!
//! The card keeps the model-facing shape, so there is no `RestfulApiCard` type
//! here: a [`ToolCard`] plus a [`RestfulApiConfig`]
//! carries the same data, and [`RestfulApiTool`] joins them with a parameter
//! mapper and the shared
//! [`HttpClient`](openjiuwen_common::http::HttpClient).
//!
//! Deviations from Python, all documented on the affected item:
//! - multipart `form` parameters (and their `form_handler_type` handlers) are out
//!   of scope; a `"location": "form"` property is a validation error;
//! - gzip/deflate decompression is not reimplemented — the transport negotiates
//!   and decodes it, so the parser only splits JSON from text;
//! - the SSL/auth callback framework is not ported: authentication material
//!   travels in the configured headers, query, or path parameters.
//!
//! ```
//! use openjiuwen_foundation::tool::card::ToolCard;
//! use openjiuwen_foundation::tool::restful::{RestfulApiConfig, RestfulApiTool, HttpMethod};
//! use serde_json::json;
//!
//! let card = ToolCard::new(
//!     "get_user",
//!     "fetches a user",
//!     json!({"type": "object", "properties": {
//!         "id": {"type": "integer", "location": "path"},
//!     }}),
//! );
//! let config = RestfulApiConfig::new("https://api.example.com/users/{id}")
//!     .with_method(HttpMethod::Get);
//! let tool = RestfulApiTool::new(card, config).unwrap();
//! assert_eq!(tool.config().url, "https://api.example.com/users/{id}");
//! ```

mod config;
mod mapper;
mod tool;

pub use config::{HttpMethod, RestfulApiConfig, DEFAULT_MAX_RESPONSE_BYTES, DEFAULT_TIMEOUT};
pub use mapper::{ApiParamMapper, ParamLocation, ParamMap};
pub use tool::RestfulApiTool;
