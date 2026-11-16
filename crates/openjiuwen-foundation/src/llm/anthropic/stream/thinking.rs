//! Extended-thinking block accumulation for the Anthropic event stream.
//!
//! The Messages API splits one thinking block across events: `content_block_start`
//! carries the block type (and, for a `redacted_thinking` block, the opaque
//! `data`), `thinking_delta` carries the text, `signature_delta` carries the
//! provider's proof, and `content_block_stop` ends it. [`ThinkingBlocks`] buffers
//! those fragments per content-block index and seals each block once the server
//! stopped it (or the stream ended), keeping only blocks the provider vouched for.
//!
//! Sealing goes through the same [`replayable_block`] rule the non-streaming
//! parser uses, so a streamed turn and an invoked turn are replayable under one
//! contract, and an unfinished block is dropped with the same warning instead of
//! reaching a request body the API would reject.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::prompt::BaseMessage;

use super::super::response::{attach_thinking_blocks, replayable_block};

/// One `thinking`/`redacted_thinking` content block being assembled.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
struct ThinkingBlock {
    /// Provider block type: `thinking` or `redacted_thinking`.
    kind: String,
    /// Concatenated `thinking_delta` text.
    text: String,
    /// Concatenated `signature_delta` proof.
    signature: String,
    /// Concatenated redacted `data`.
    data: String,
}

impl ThinkingBlock {
    /// A `thinking` block, the assumed kind for a text fragment.
    fn thinking() -> Self {
        Self {
            kind: "thinking".to_owned(),
            ..Self::default()
        }
    }

    /// A `redacted_thinking` block, the assumed kind for a data fragment.
    fn redacted() -> Self {
        Self {
            kind: "redacted_thinking".to_owned(),
            ..Self::default()
        }
    }

    /// The provider-shaped content block this fragment set describes.
    ///
    /// The shape matches the non-streaming parser's blocks, so both paths deal in
    /// one representation: `{type, thinking, signature}` or `{type, data}`.
    fn to_block(&self) -> Map<String, Value> {
        let mut block = Map::new();
        let kind = if self.kind.is_empty() {
            "thinking"
        } else {
            self.kind.as_str()
        };
        block.insert("type".to_owned(), Value::String(kind.to_owned()));
        if kind == "redacted_thinking" {
            block.insert("data".to_owned(), Value::String(self.data.clone()));
        } else {
            block.insert("thinking".to_owned(), Value::String(self.text.clone()));
            block.insert(
                "signature".to_owned(),
                Value::String(self.signature.clone()),
            );
        }
        block
    }
}

/// The thinking blocks of one streamed turn.
///
/// Blocks are buffered per content-block index while the stream runs and sealed
/// by `content_block_stop` (or by the end of the stream), so
/// [`preserved`](Self::preserved) only ever yields blocks the provider vouched
/// for, in response order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct ThinkingBlocks {
    /// Blocks still being assembled, keyed by content-block index.
    in_progress: BTreeMap<u32, ThinkingBlock>,
    /// Sealed, replayable blocks, keyed by content-block index so they come back
    /// in response order rather than in the order the server stopped them.
    preserved: BTreeMap<u32, Value>,
}

impl ThinkingBlocks {
    /// Opens a `thinking` block with what its `content_block_start` carries.
    pub(super) fn open_thinking(
        &mut self,
        index: u32,
        text: Option<&str>,
        signature: Option<&str>,
    ) {
        let block = self
            .in_progress
            .entry(index)
            .or_insert_with(ThinkingBlock::thinking);
        "thinking".clone_into(&mut block.kind);
        if let Some(text) = text {
            block.text.push_str(text);
        }
        if let Some(signature) = signature {
            block.signature.push_str(signature);
        }
    }

    /// Opens a `redacted_thinking` block with the data its start event carries.
    pub(super) fn open_redacted(&mut self, index: u32, data: Option<&str>) {
        let block = self
            .in_progress
            .entry(index)
            .or_insert_with(ThinkingBlock::redacted);
        "redacted_thinking".clone_into(&mut block.kind);
        if let Some(data) = data {
            block.data.push_str(data);
        }
    }

    /// Appends one `thinking_delta` fragment to a thinking block.
    pub(super) fn append_text(&mut self, index: u32, text: &str) {
        self.in_progress
            .entry(index)
            .or_insert_with(ThinkingBlock::thinking)
            .text
            .push_str(text);
    }

    /// Appends one `signature_delta` fragment to a thinking block.
    pub(super) fn append_signature(&mut self, index: u32, signature: &str) {
        self.in_progress
            .entry(index)
            .or_insert_with(ThinkingBlock::thinking)
            .signature
            .push_str(signature);
    }

    /// Appends one redacted `data` fragment to a redacted block.
    pub(super) fn append_data(&mut self, index: u32, data: &str) {
        self.in_progress
            .entry(index)
            .or_insert_with(ThinkingBlock::redacted)
            .data
            .push_str(data);
    }

    /// Seals one block the stream stopped, keeping it when it is replayable.
    ///
    /// The block is judged by [`replayable_block`]: without the provider's
    /// `signature` (or a redacted block's `data`) it is dropped with a warning,
    /// never sent back, and this client never invents one.
    pub(super) fn seal(&mut self, index: u32) {
        let Some(block) = self.in_progress.remove(&index) else {
            return;
        };
        if let Some(value) = replayable_block(&block.to_block()) {
            self.preserved.insert(index, value);
        }
    }

    /// Seals every block the stream never stopped, in block order.
    pub(super) fn seal_pending(&mut self) {
        let pending = std::mem::take(&mut self.in_progress);
        for (index, block) in pending {
            if let Some(value) = replayable_block(&block.to_block()) {
                self.preserved.insert(index, value);
            }
        }
    }

    /// The replayable blocks preserved so far, in block order.
    pub(super) fn preserved(&self) -> Vec<Value> {
        self.preserved.values().cloned().collect()
    }

    /// Attaches the preserved blocks to an accumulated assistant message.
    pub(super) fn apply(&self, message: &mut BaseMessage) {
        attach_thinking_blocks(message, self.preserved());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::anthropic::AnthropicStreamState;
    use crate::llm::anthropic::THINKING_BLOCKS_METADATA_KEY;
    use crate::llm::model::StreamChunk;
    use crate::llm::sse::SseEvent;
    use serde_json::json;

    fn event(name: &str, data: Value) -> SseEvent {
        SseEvent {
            event: Some(name.to_owned()),
            data: data.to_string(),
        }
    }

    fn thinking_start(index: u32) -> SseEvent {
        event(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            }),
        )
    }

    fn thinking_delta(index: u32, text: &str) -> SseEvent {
        event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": text},
            }),
        )
    }

    fn signature_delta(index: u32, signature: &str) -> SseEvent {
        event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": signature},
            }),
        )
    }

    fn redacted_start(index: u32, data: Option<&str>) -> SseEvent {
        let block = match data {
            Some(data) => json!({"type": "redacted_thinking", "data": data}),
            None => json!({"type": "redacted_thinking"}),
        };
        event(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            }),
        )
    }

    fn data_delta(index: u32, data: &str) -> SseEvent {
        event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "data", "data": data},
            }),
        )
    }

    fn block_stop(index: u32) -> SseEvent {
        event(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        )
    }

    /// The message a [`StreamAccumulator`](crate::llm::model::StreamAccumulator)
    /// would build from a thinking turn, with the blocks attached.
    fn accumulated(state: &AnthropicStreamState) -> BaseMessage {
        let mut message = BaseMessage::assistant("checking");
        message.reasoning_content = Some("Paris is in France.".to_owned());
        state.apply_thinking_blocks(&mut message);
        message
    }

    #[test]
    fn a_signed_thinking_block_is_preserved_and_replayable() {
        let mut state = AnthropicStreamState::new();
        state.handle_event(&thinking_start(0)).expect("event");
        let chunks = state
            .handle_event(&thinking_delta(0, "Paris is in France."))
            .expect("event");
        assert_eq!(
            chunks.len(),
            1,
            "the thinking text still arrives as a reasoning fragment"
        );
        match &chunks[0] {
            StreamChunk::Delta { reasoning, .. } => {
                assert_eq!(reasoning.as_deref(), Some("Paris is in France."));
            }
            other => panic!("expected a reasoning delta, got {other:?}"),
        }
        assert!(
            state
                .handle_event(&signature_delta(0, "sig-thinking"))
                .expect("event")
                .is_empty(),
            "a signature is not content"
        );
        assert!(
            state.thinking_blocks().is_empty(),
            "a block is only sealed when the stream stops it"
        );

        state.handle_event(&block_stop(0)).expect("event");
        assert_eq!(
            state.thinking_blocks(),
            vec![json!({
                "type": "thinking",
                "thinking": "Paris is in France.",
                "signature": "sig-thinking",
            })]
        );

        let message = accumulated(&state);
        assert_eq!(
            message.reasoning_content.as_deref(),
            Some("Paris is in France.")
        );
        assert_eq!(
            message.metadata.get(THINKING_BLOCKS_METADATA_KEY).cloned(),
            Some(json!([{
                "type": "thinking",
                "thinking": "Paris is in France.",
                "signature": "sig-thinking",
            }])),
            "the accumulated message carries the non-streaming metadata key"
        );
    }

    #[test]
    fn a_thinking_block_without_a_signature_is_never_preserved() {
        let mut state = AnthropicStreamState::new();
        state.handle_event(&thinking_start(0)).expect("event");
        state
            .handle_event(&thinking_delta(0, "unsigned"))
            .expect("event");
        state.handle_event(&block_stop(0)).expect("event");

        assert!(state.thinking_blocks().is_empty(), "no signature, no block");
        let message = accumulated(&state);
        assert!(
            !message.metadata.contains_key(THINKING_BLOCKS_METADATA_KEY),
            "an unreplayable block never reaches the message"
        );
    }

    #[test]
    fn redacted_thinking_blocks_keep_their_data() {
        let mut state = AnthropicStreamState::new();
        state
            .handle_event(&redacted_start(1, Some("opaque-blob")))
            .expect("event");
        state.handle_event(&block_stop(1)).expect("event");
        assert_eq!(
            state.thinking_blocks(),
            vec![json!({"type": "redacted_thinking", "data": "opaque-blob"})]
        );

        let mut fragmented = AnthropicStreamState::new();
        fragmented
            .handle_event(&redacted_start(2, None))
            .expect("event");
        fragmented
            .handle_event(&data_delta(2, "opaque-"))
            .expect("event");
        fragmented
            .handle_event(&data_delta(2, "blob"))
            .expect("event");
        fragmented.handle_event(&block_stop(2)).expect("event");
        assert_eq!(
            fragmented.thinking_blocks(),
            vec![json!({"type": "redacted_thinking", "data": "opaque-blob"})],
            "data fragments concatenate"
        );

        let mut empty = AnthropicStreamState::new();
        empty.handle_event(&redacted_start(3, None)).expect("event");
        empty.handle_event(&block_stop(3)).expect("event");
        assert!(
            empty.thinking_blocks().is_empty(),
            "a redacted block without data is not replayable"
        );
    }

    #[test]
    fn preserved_blocks_keep_the_stream_order() {
        let mut state = AnthropicStreamState::new();
        state.handle_event(&thinking_start(0)).expect("event");
        state
            .handle_event(&thinking_delta(0, "first"))
            .expect("event");
        state
            .handle_event(&signature_delta(0, "sig-0"))
            .expect("event");
        state
            .handle_event(&redacted_start(1, Some("blob")))
            .expect("event");
        state.handle_event(&block_stop(1)).expect("event");
        state.handle_event(&block_stop(0)).expect("event");

        assert_eq!(
            state.thinking_blocks(),
            vec![
                json!({"type": "thinking", "thinking": "first", "signature": "sig-0"}),
                json!({"type": "redacted_thinking", "data": "blob"}),
            ],
            "blocks come back by block index, not by the order they stopped"
        );
    }

    #[test]
    fn finishing_seals_a_thinking_block_the_server_never_stopped() {
        let mut state = AnthropicStreamState::new();
        state.handle_event(&thinking_start(0)).expect("event");
        state
            .handle_event(&thinking_delta(0, "cut off"))
            .expect("event");
        state
            .handle_event(&signature_delta(0, "sig-0"))
            .expect("event");

        assert_eq!(state.finish(), vec![StreamChunk::Done]);
        assert_eq!(
            state.thinking_blocks(),
            vec![json!({"type": "thinking", "thinking": "cut off", "signature": "sig-0"})]
        );

        let mut unsigned = AnthropicStreamState::new();
        unsigned.handle_event(&thinking_start(0)).expect("event");
        unsigned
            .handle_event(&thinking_delta(0, "cut off"))
            .expect("event");
        unsigned.finish();
        assert!(
            unsigned.thinking_blocks().is_empty(),
            "the end of the stream does not invent a signature"
        );
    }
}
