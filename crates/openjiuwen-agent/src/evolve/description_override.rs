//! [`DescriptionOverrideTool`]: a tool view whose card description was tuned.
//!
//! `ReActAgent` exposes no way to mutate a mounted tool's [`ToolCard`], so a
//! description update replaces the tool with this wrapper: the card is a copy
//! of the wrapped tool's card with only `description` replaced, while
//! execution, streaming, rendering, and the parallel-safety flag all delegate
//! to the wrapped tool.
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{Tool, ToolCard, ToolContext};
use serde_json::Value;

/// A [`Tool`] wrapper that presents an updated [`ToolCard`] description.
///
/// Every other card field (id, name, input schema, exposure, execution,
/// parallel safety, metadata) is preserved from the wrapped tool, and every
/// call is forwarded unchanged.
pub struct DescriptionOverrideTool {
    /// The tool that actually executes the call.
    inner: Arc<dyn Tool>,
    /// The model-facing card, copied from `inner` with the new description.
    card: ToolCard,
}

impl DescriptionOverrideTool {
    /// Wraps `inner`, replacing the description of its card.
    #[must_use]
    pub fn new(inner: Arc<dyn Tool>, description: impl Into<String>) -> Self {
        let mut card = inner.card().clone();
        card.description = description.into();
        Self { inner, card }
    }

    /// Wraps `inner` with an already-patched card.
    ///
    /// The caller is responsible for the card being a copy of
    /// `inner.card()`; only the description is expected to differ.
    #[must_use]
    pub fn from_card(inner: Arc<dyn Tool>, card: ToolCard) -> Self {
        Self { inner, card }
    }

    /// The wrapped tool.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn Tool> {
        &self.inner
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for DescriptionOverrideTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DescriptionOverrideTool")
            .field("card", &self.card)
            .finish()
    }
}

#[async_trait]
impl Tool for DescriptionOverrideTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        self.inner.invoke(inputs, ctx).await
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        ctx: ToolContext,
    ) -> BoxStream<'static, Result<Value>> {
        Arc::clone(&self.inner).stream(inputs, ctx)
    }

    fn render_for_llm(&self, output: &Value) -> String {
        self.inner.render_for_llm(output)
    }

    fn is_parallel_safe(&self) -> bool {
        self.inner.is_parallel_safe()
    }
}
