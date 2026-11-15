//! Disclosure rendering: subtree exposure, code assignment, prompts and reply parsing.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.render` (v0.1.20) — the
//! `disclosure` module, its `default` renderer and the Chinese prompt bank of
//! `common/prompts/retrieval_disclosure.yaml` — as one synchronous, model-free layer.
//!
//! # Layers
//!
//! - [`expose`] — [`build_exposed_fragment`] walks a
//!   [`RetrieverNode`](crate::retrieval::types::RetrieverNode) depth-first, decides which
//!   nodes are selectable, assigns one code per selectable candidate, renders the
//!   prompt-facing candidate block and hashes the exposure
//!   ([`fragmentation_fingerprint`]).
//! - [`prompt`] — the prompt bank plus the two message builders
//!   ([`build_disclosure_messages`], [`build_disclosure_prompt_parts`]) that embed the
//!   candidate block in a `system` message and the query in a `user` message.
//! - [`parse`] — [`parse_selected_codes`], the tolerant cascade that turns a model reply
//!   into [`SelectableResolution`](crate::retrieval::search::flow::SelectableResolution)s.
//!
//! # The renderer seam
//!
//! [`SubtreeRenderer`] is the hook the traversal layer calls:
//! [`DefaultSubtreeRenderer`] passes the subtree's fragment and the caller's
//! [`SelectionProtocol`] through unchanged and, when it is built with messages, renders the
//! disclosure prompt for the cursor's `top_k`. It never parses a reply — that is the
//! caller's job, through [`parse_selected_codes`], so a renderer can be reused for warmup
//! (fragment and protocol only) without touching the parser.
//!
//! # Determinism
//!
//! Nothing here reads the clock or an RNG, no output depends on hash-map iteration order
//! (codes, resolutions and flat candidates are `BTreeMap`/sorted-vector ordered), and the
//! request marker is a digest instead of Python's `uuid4` — see [`prompt`] for that
//! documented deviation. Identical inputs therefore produce byte-identical fragments and
//! prompts.

pub mod expose;
pub mod parse;
pub mod prompt;

pub use expose::{build_exposed_fragment, exposure_item_count, fragmentation_fingerprint};
pub use parse::parse_selected_codes;
pub use prompt::{
    build_disclosure_messages, build_disclosure_prompt_parts, disclosure_request_marker,
    disclosure_system_prompt, disclosure_user_prefix, DisclosurePromptParts,
    DISCLOSURE_PROMPT_VERSION,
};

use crate::error::Result;
use crate::retrieval::search::flow::{
    ChatMessage, CurrentSubtree, PromptBundle, SelectionProtocol,
};

/// Renders one exposed subtree into the prompt a selection call sends.
///
/// Implementations are synchronous and pure: they never call a model and never parse the
/// reply. `query_messages` is the caller's conversation; a renderer built without messages
/// returns an empty [`PromptBundle::messages`].
pub trait SubtreeRenderer: Send + Sync {
    /// Renders `subtree` into a prompt bundle; `messages` is empty when the renderer was
    /// built without messages.
    ///
    /// # Errors
    ///
    /// Propagates the renderer's own failures; [`DefaultSubtreeRenderer`] never fails.
    fn render_subtree(
        &self,
        subtree: &CurrentSubtree,
        query_messages: &[ChatMessage],
        protocol: &SelectionProtocol,
    ) -> Result<PromptBundle>;
}

/// The default [`SubtreeRenderer`]: the crate's disclosure prompt, or protocol only.
///
/// Mirrors Python's `DefaultSubtreeRenderer`: the fragment and the caller's protocol are
/// passed through unchanged, and the messages come from
/// [`build_disclosure_messages`] for the cursor's
/// [`top_k`](crate::retrieval::search::flow::SearchCursor::top_k) when messages are
/// enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefaultSubtreeRenderer {
    /// Whether [`SubtreeRenderer::render_subtree`] fills in the messages.
    render_messages: bool,
}

impl DefaultSubtreeRenderer {
    /// Builds a renderer that fills in the disclosure messages.
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_messages: true,
        }
    }

    /// Builds a renderer that returns the fragment and the protocol only.
    ///
    /// This is the warmup path: the caller wants the exposure to build its own prompt.
    #[must_use]
    pub fn without_messages() -> Self {
        Self {
            render_messages: false,
        }
    }

    /// Turns message rendering on or off.
    #[must_use]
    pub fn with_messages(mut self, enabled: bool) -> Self {
        self.render_messages = enabled;
        self
    }

    /// Whether this renderer fills in the messages.
    #[must_use]
    pub fn renders_messages(&self) -> bool {
        self.render_messages
    }
}

impl Default for DefaultSubtreeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl SubtreeRenderer for DefaultSubtreeRenderer {
    fn render_subtree(
        &self,
        subtree: &CurrentSubtree,
        query_messages: &[ChatMessage],
        protocol: &SelectionProtocol,
    ) -> Result<PromptBundle> {
        let messages = if self.render_messages {
            build_disclosure_messages(
                &subtree.fragment,
                query_messages,
                Some(subtree.cursor.top_k),
            )
        } else {
            Vec::new()
        };
        Ok(PromptBundle {
            fragment: subtree.fragment.clone(),
            protocol: protocol.clone(),
            messages,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::manual_assert_eq
)]
mod tests {
    use super::*;
    use crate::retrieval::search::config::DisclosureConfig;
    use crate::retrieval::search::flow::{SearchCursor, SelectableTarget};
    use crate::retrieval::types::{RetrieverItem, RetrieverNode, ROOT_NODE_ID};

    fn subtree() -> CurrentSubtree {
        let tree = RetrieverNode::new(ROOT_NODE_ID, "Root")
            .with_children(vec![RetrieverNode::new("Skills", "Skills").with_items(
                vec![RetrieverItem::new("a", "custom.a").with_label("Alpha")],
            )]);
        let fragment = build_exposed_fragment(
            &tree,
            &[ROOT_NODE_ID.to_owned()],
            &DisclosureConfig::default(),
            &exposure_item_count,
        )
        .unwrap();
        let targets = fragment
            .resolutions()
            .into_iter()
            .cloned()
            .map(SelectableTarget::new)
            .collect();
        CurrentSubtree {
            cursor: SearchCursor::new(&tree, 3),
            fragment,
            selectable_targets: targets,
        }
    }

    #[test]
    fn the_default_renderer_fills_in_the_messages() {
        let subtree = subtree();
        let protocol = SelectionProtocol {
            compact_codes_enabled: false,
            candidate_codes: subtree.fragment.candidate_codes.clone(),
            code_width: 1,
            abstain_token: crate::retrieval::search::flow::ABSTAIN_TOKEN.to_owned(),
        };
        let bundle = DefaultSubtreeRenderer::new()
            .render_subtree(&subtree, &[ChatMessage::user("摘要")], &protocol)
            .unwrap();
        assert_eq!(bundle.fragment, subtree.fragment);
        assert_eq!(bundle.protocol, protocol);
        assert_eq!(bundle.messages.len(), 2);
        assert_eq!(bundle.messages[0].role, "system");
        assert!(bundle.messages[0].content.contains("# 候选列表"));
        assert!(bundle.messages[1].content.contains("摘要"));
        assert!(DefaultSubtreeRenderer::default().renders_messages());
    }

    #[test]
    fn the_warmup_renderer_leaves_the_messages_empty() {
        let subtree = subtree();
        let renderer = DefaultSubtreeRenderer::without_messages();
        assert!(!renderer.renders_messages());
        let bundle = renderer
            .render_subtree(
                &subtree,
                &[ChatMessage::user("摘要")],
                &SelectionProtocol::default(),
            )
            .unwrap();
        assert!(bundle.messages.is_empty());
        assert!(!bundle.fragment.candidate_codes.is_empty());
        assert_eq!(bundle.fragment, subtree.fragment);
        assert!(renderer.with_messages(true).renders_messages());
        assert!(!DefaultSubtreeRenderer::new()
            .with_messages(false)
            .renders_messages());
    }

    #[test]
    fn the_cursored_top_k_reaches_the_prompt() {
        let mut subtree = subtree();
        subtree.cursor.top_k = 2;
        let bundle = DefaultSubtreeRenderer::new()
            .render_subtree(&subtree, &[], &SelectionProtocol::default())
            .unwrap();
        assert!(bundle.messages[0]
            .content
            .contains("默认输出 2 个 Candidate skill 节点 id"));
    }

    #[test]
    fn the_renderer_never_parses() {
        let subtree = subtree();
        let bundle = DefaultSubtreeRenderer::new()
            .render_subtree(
                &subtree,
                &[ChatMessage::user("摘要")],
                &SelectionProtocol::default(),
            )
            .unwrap();
        assert_eq!(bundle.messages.len(), 2);
        assert!(parse_selected_codes(&subtree.fragment, "Alpha").len() == 1);
        assert!(bundle.fragment.truncated == subtree.fragment.truncated);
    }
}
