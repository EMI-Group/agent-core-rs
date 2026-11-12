//! `HTTP` moderation transport over the shared async client.
//!
//! Owns the network vocabulary the Python SDK inlined with `httpx`:
//! endpoint, headers, credential, timeout, retries, and error mapping.
//! The wire payload is `{"<text_field>": ...}` and the JSON reply is handed
//! back to a [`crate::parsers::ModelOutputParser`].
