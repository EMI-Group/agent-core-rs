//! Anthropic Messages transport.
//!
//! Handles the Anthropic-specific message projection (system hoisting,
//! `tool_use`/`tool_result` blocks, base64/URL images, content-block cache
//! breakpoints) and the event-oriented SSE stream. Request, response, and stream
//! conversion live in the private submodules and are re-exported from here.

mod request;
mod response;
mod stream;
