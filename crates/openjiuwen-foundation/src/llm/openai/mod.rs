//! OpenAI-compatible chat-completions transport.
//!
//! All OpenAI-compatible vendors share one client; [`crate::llm::provider`] only
//! supplies the base URL, auth header shape, and profile flags. Request,
//! response, and stream conversion live in the private submodules and are
//! re-exported from here so unit tests and extension crates can exercise the
//! pure conversion helpers without a network.

mod request;
mod response;
mod stream;
