//! A minimal completion-model seam shared by retrieval components that reason
//! with an LLM.
//!
//! Python's query rewriter calls `openjiuwen.core.foundation.llm.Model`
//! directly. This crate never depends on `openjiuwen-foundation`, so the model
//! is abstracted behind [`CompletionModel`]: one prompt in, one text reply out —
//! all the retrieval layer needs. [`StaticCompletionModel`] is the deterministic
//! offline implementation used by tests and examples, and
//! [`extract_json_object`] is the shared helper that pulls a JSON object out of
//! a free-form model reply.
//!
//! An adapter over `openjiuwen_foundation::Model` belongs in an upper crate
//! (for example `openjiuwen-extensions`), mirroring the [`crate::reranker`]
//! seam.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::{Error, Result};

/// Model name reported by [`CompletionModel`] when an implementation does not
/// override it.
const DEFAULT_MODEL_NAME: &str = "model";

/// A single-turn text completion source.
///
/// Implementations are called with a fully rendered prompt and return the
/// model's reply verbatim; prompt construction and reply parsing are the
/// caller's responsibility.
#[async_trait]
pub trait CompletionModel: Send + Sync {
    /// Human-readable model name, used in logs and error messages.
    fn name(&self) -> &str {
        DEFAULT_MODEL_NAME
    }

    /// Completes `prompt` and returns the model's reply.
    async fn complete(&self, prompt: &str) -> Result<String>;
}

/// Model name reported by [`StaticCompletionModel`].
const STATIC_MODEL_NAME: &str = "static-completion-model";

/// A deterministic [`CompletionModel`] that replays scripted replies.
///
/// The model returns `responses[0]`, then `responses[1]`, and so on; once the
/// script is exhausted the last entry is returned repeatedly, so a multi-call
/// workflow (rewrite, then keywords) can be scripted without counting calls
/// exactly. Every prompt is recorded for assertions.
///
/// This type exists for tests and offline examples: it performs no I/O and is
/// fully reproducible.
#[derive(Debug)]
pub struct StaticCompletionModel {
    /// Scripted replies, returned in order.
    responses: Vec<String>,
    /// Number of completed calls; also the index into `responses`.
    calls: AtomicUsize,
    /// Prompts received so far, in call order.
    prompts: Mutex<Vec<String>>,
}

impl StaticCompletionModel {
    /// Creates a model replaying `responses`.
    ///
    /// An empty script is allowed at construction time; calling
    /// [`CompletionModel::complete`] then fails with [`Error::Llm`] instead of
    /// silently returning an empty reply.
    #[must_use]
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            calls: AtomicUsize::new(0),
            prompts: Mutex::new(Vec::new()),
        }
    }

    /// Creates a model replaying a single reply, however often it is called.
    #[must_use]
    pub fn single(response: impl Into<String>) -> Self {
        Self::new(vec![response.into()])
    }

    /// Number of calls made so far.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Prompts received so far, in call order.
    #[must_use]
    pub fn prompts(&self) -> Vec<String> {
        self.prompts.lock().clone()
    }

    /// The most recent prompt, if the model has been called.
    #[must_use]
    pub fn last_prompt(&self) -> Option<String> {
        self.prompts.lock().last().cloned()
    }
}

#[async_trait]
impl CompletionModel for StaticCompletionModel {
    fn name(&self) -> &str {
        STATIC_MODEL_NAME
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        self.prompts.lock().push(prompt.to_owned());
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        let Some(last) = self.responses.len().checked_sub(1) else {
            return Err(Error::llm(
                "StaticCompletionModel has no scripted responses to return",
            ));
        };
        Ok(self.responses[index.min(last)].clone())
    }
}

/// Extracts the JSON object embedded in a model reply.
///
/// The reply may be surrounded by prose: the returned slice runs from the first
/// `{` to the last `}`. When the reply contains a triple-backtick fenced block
/// (optionally tagged `json`), only the fence's content is searched, so a brace
/// in the surrounding prose cannot widen the slice. Returns `None` when no
/// non-empty object-looking span exists.
pub fn extract_json_object(text: &str) -> Option<&str> {
    if let Some(block) = fenced_block(text) {
        return slice_object(block);
    }
    slice_object(text)
}

/// Returns the content of the first triple-backtick fenced block, if that block
/// is a JSON (or untagged) fence.
fn fenced_block(text: &str) -> Option<&str> {
    let open = text.find("```")?;
    let rest = &text[open + 3..];
    let newline = rest.find('\n')?;
    let tag = rest[..newline].trim();
    if !tag.is_empty() && !tag.eq_ignore_ascii_case("json") {
        return None;
    }
    let content = &rest[newline + 1..];
    let close = content.find("```")?;
    Some(&content[..close])
}

/// Slices from the first `{` to the last `}`, requiring a non-empty span.
fn slice_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&text[start..=end])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[tokio::test]
    async fn static_model_replays_replies_in_order_then_repeats_the_last() {
        let model = StaticCompletionModel::new(vec!["first".to_owned(), "second".to_owned()]);
        assert_eq!(model.name(), "static-completion-model");
        assert_eq!(model.complete("a").await.unwrap(), "first");
        assert_eq!(model.complete("b").await.unwrap(), "second");
        assert_eq!(model.complete("c").await.unwrap(), "second");
        assert_eq!(model.call_count(), 3);
        assert_eq!(model.prompts(), vec!["a", "b", "c"]);
        assert_eq!(model.last_prompt().unwrap(), "c");
    }

    #[tokio::test]
    async fn static_model_without_a_script_fails_instead_of_returning_empty() {
        let model = StaticCompletionModel::new(Vec::new());
        let error = model.complete("prompt").await.expect_err("must fail");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert_eq!(model.call_count(), 1);
    }

    #[test]
    fn extract_json_object_takes_the_first_open_and_last_close() {
        let text = "sure! {\"a\": 1} and then {\"b\": 2} done";
        assert_eq!(
            extract_json_object(text).unwrap(),
            "{\"a\": 1} and then {\"b\": 2}"
        );
    }

    #[test]
    fn extract_json_object_handles_json_fences_and_surrounding_prose() {
        let text = "Here you go:\n```json\n{\n  \"before\": \"x\"\n}\n```\nhope that helps";
        assert_eq!(
            extract_json_object(text).unwrap(),
            "{\n  \"before\": \"x\"\n}"
        );

        let untagged = "```\n{\"a\": 1}\n```";
        assert_eq!(extract_json_object(untagged).unwrap(), "{\"a\": 1}");
    }

    #[test]
    fn extract_json_object_ignores_non_json_fences() {
        let text = "```python\nprint('}')\n```\n{\"a\": 1}";
        assert_eq!(extract_json_object(text).unwrap(), "{\"a\": 1}");
    }

    #[test]
    fn extract_json_object_returns_none_without_a_complete_object() {
        assert!(extract_json_object("no json here").is_none());
        assert!(extract_json_object("just {").is_none());
        assert!(extract_json_object("just }").is_none());
        assert!(extract_json_object("").is_none());
        // An empty object is a complete span; the caller repairs its fields.
        assert_eq!(extract_json_object("{}").unwrap(), "{}");
    }
}
