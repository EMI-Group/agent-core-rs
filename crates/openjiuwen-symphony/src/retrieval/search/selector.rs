//! The default top-K selectors: model delegation and a lexical fallback.
//!
//! Ports Python `openjiuwen.symphony.retrieval.search.runtime.selector` (v0.1.20) —
//! `DefaultTopKSelector`, the `GenerateFragmentSelector` callback wrapper and
//! `_is_abstain_output` — plus two Rust-only additions:
//!
//! - [`LexicalTopKSelector`] ranks the exposed candidates with the crate's
//!   [`LexicalIndex`] instead of asking a model, so a progressive walk works with no model
//!   configured at all ([`SelectionMode::Lexical`], or the offline fallback of a failed
//!   model call);
//! - [`DefaultTopKSelector`] holds an **optional** generate selector: with none wired,
//!   [`SelectionMode::Generate`] plus
//!   [`FallbackMode::Error`](crate::llm::FallbackMode) answers
//!   [`Error::Unsupported`](crate::error::Error) instead of panicking, and
//!   [`FallbackMode::Deterministic`](crate::llm::FallbackMode) (the default) uses the
//!   lexical selector.
//!
//! # Selection order
//!
//! 1. [`SelectionMode::Lexical`] always uses [`LexicalTopKSelector`] — the mode wins over a
//!    configured generate selector.
//! 2. [`SelectionMode::Generate`] with a generate selector calls it; its
//!    [`GenerateFn`](crate::retrieval::search::contracts::GenerateFn) owns rendering, the
//!    model call and the reply parsing. An error is surfaced as-is under
//!    [`FallbackMode::Error`](crate::llm::FallbackMode) and logged-then-degraded to the
//!    lexical selector under [`FallbackMode::Deterministic`](crate::llm::FallbackMode).
//! 3. [`SelectionMode::Generate`] without a generate selector: the deterministic mode uses
//!    [`LexicalTopKSelector`], the error mode is [`Error::Unsupported`](crate::error::Error).
//!
//! Every selection records exactly one `fragment_selected` trace event carrying the mode,
//! the selected codes, the selected canonical ids and the raw model output.
//!
//! # The lexical selector
//!
//! One [`LexicalDocument`] is built per selectable target — key = the canonical id, name =
//! the display name, description = the description, category = the branch path joined with
//! `" > "` — and the concatenated query (every message's content, joined with `\n`) ranks
//! it. That text is compiled into `|`-separated term alternatives before it reaches the
//! index (see `ranking_expression`), because a whole conversation compared as one literal
//! phrase would gate every candidate out. A blank query selects nothing (`is_abstain`
//! stays `false`: nothing was asked), and so does an unusable query expression (logged,
//! not fatal). The score-truncation config of
//! [`ProgressiveSearchConfig::truncation`] then applies, and `top_k == 0` means unlimited.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{unsupported, Result};
use crate::llm::FallbackMode;
use crate::retrieval::search::config::{ProgressiveSearchConfig, SelectionMode};
use crate::retrieval::search::contracts::{
    GenerateFragmentSelector, GenerateRequest, SelectRequest, TopKSelector,
};
use crate::retrieval::search::flow::{
    ChatMessage, CurrentSubtree, SearchCursor, SelectableTarget, SelectionProtocol,
    SelectionResult, ABSTAIN_TOKEN,
};
use crate::retrieval::search::lexical::{LexicalDocument, LexicalIndex, MAX_QUERY_CHARS};
use crate::retrieval::search::truncation::{truncate_chars, truncate_sorted_by_score};
use crate::retrieval::types::RetrieverTrace;

/// The trace event every selection records.
pub const FRAGMENT_SELECTED_EVENT: &str = "fragment_selected";

/// The `mode` a lexical selection reports in its trace event.
pub const MODE_LEXICAL: &str = "lexical_fallback";

/// The `mode` a model-backed selection reports in its trace event.
pub const MODE_GENERATE: &str = "generate";

/// The "nothing here is relevant" phrasings, all lowercase.
///
/// Python matches them with one case-insensitive regex; the port does hand-rolled
/// `contains` checks (no regex engine is available offline). The list is Python's, plus
/// the typographic-apostrophe spelling of `can't determine`, which real replies use.
pub const ABSTAIN_HINTS: &[&str] = &[
    "none",
    "no suitable",
    "no relevant",
    "not relevant",
    "unrelated",
    "cannot determine",
    "can't determine",
    "can’t determine",
    "无合适",
    "没有合适",
    "无相关",
    "没有相关",
    "不相关",
    "无法判断",
    "无法确定",
];

/// Whether a model answer says "nothing here is relevant".
///
/// Mirrors Python `DefaultTopKSelector._is_abstain_output`: a blank answer is **not** an
/// abstention (nothing was answered), the reserved [`ABSTAIN_TOKEN`] is, and so is any
/// answer carrying one of [`ABSTAIN_HINTS`].
#[must_use]
pub fn is_abstain_output(output: &str) -> bool {
    let text = output.trim();
    if text.is_empty() {
        return false;
    }
    if text == ABSTAIN_TOKEN {
        return true;
    }
    let lowered = text.to_lowercase();
    ABSTAIN_HINTS.iter().any(|hint| lowered.contains(hint))
}

/// The query text of `query_messages`: every content, joined with `\n`.
///
/// This is what the lexical selector ranks against — the same text a model would read,
/// minus the exposure.
#[must_use]
pub fn query_text(query_messages: &[ChatMessage]) -> String {
    query_messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Ranks the exposed candidates lexically and selects the best ones.
///
/// Rust-only: Python has no model-free selector. The ranking is the crate's deterministic
/// field-weighted scorer over the subsystem's own documents, so the result is offline,
/// reproducible and tie-broken by canonical id (the index emits that order).
#[derive(Debug, Clone, Default)]
pub struct LexicalTopKSelector {
    /// The traversal config; only its truncation rules and `top_k` semantics are read.
    config: ProgressiveSearchConfig,
}

impl LexicalTopKSelector {
    /// Builds the selector over `config`.
    #[must_use]
    pub fn new(config: ProgressiveSearchConfig) -> Self {
        Self { config }
    }

    /// Selects at most `top_k` of `subtree`'s targets for `query_messages`.
    ///
    /// `top_k == 0` means unlimited. The returned [`SelectionResult`] never marks itself
    /// as an abstention: a lexical selector that finds nothing has no opinion, whereas an
    /// abstention is a model's explicit "nothing relevant here".
    #[must_use]
    pub fn select(
        &self,
        subtree: &CurrentSubtree,
        query_messages: &[ChatMessage],
        top_k: usize,
    ) -> SelectionResult {
        if subtree.selectable_targets.is_empty() {
            return SelectionResult::default();
        }
        let query = query_text(query_messages);
        if query.trim().is_empty() {
            return SelectionResult::default();
        }
        let by_canonical_id: BTreeMap<&str, &SelectableTarget> = subtree
            .selectable_targets
            .iter()
            .map(|target| (target.canonical_id(), target))
            .collect();
        let index = LexicalIndex::build(subtree.selectable_targets.iter().map(document));
        let hits = match index.search(&ranking_expression(&query)) {
            Ok(hits) => hits,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    query_chars = query.chars().count(),
                    "lexical selection skipped an unusable query"
                );
                return SelectionResult::default();
            }
        };
        let (kept, _decision) =
            truncate_sorted_by_score(&hits, |hit| hit.score, &self.config.truncation);
        let limit = if top_k == 0 { usize::MAX } else { top_k };
        let selected_targets = kept
            .iter()
            .filter_map(|hit| by_canonical_id.get(hit.key.as_str()).copied().cloned())
            .take(limit)
            .collect();
        SelectionResult {
            raw_output: String::new(),
            selected_targets,
            is_abstain: false,
        }
    }
}

/// Turns a conversational query into the index's alternative expression.
///
/// [`LexicalIndex`] takes one short query and gates every document on it, so a whole chat
/// history would be compared as a single literal phrase and match nothing. The selector
/// therefore compiles the text into `|`-separated term alternatives — the index's own
/// syntax — which is what makes "rank the exposed candidates against the conversation"
/// mean something:
///
/// - whitespace is collapsed and the tokens are deduplicated case-insensitively in
///   first-appearance order (so the expression is deterministic);
/// - a single-token query is passed through unchanged (a phrase of one term is the common
///   case, and it keeps identity matching exact);
/// - the expression is bounded by [`MAX_QUERY_CHARS`] by dropping trailing alternatives,
///   never by cutting inside one — except for a single token longer than the bound, which
///   is cut on a character boundary.
fn ranking_expression(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<&str>>().join(" ");
    if collapsed.is_empty() {
        return String::new();
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let alternatives: Vec<&str> = collapsed
        .split(' ')
        .filter(|token| seen.insert(token.to_lowercase()))
        .collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut length = 0_usize;
    for alternative in &alternatives {
        let extra = alternative.chars().count() + usize::from(!kept.is_empty());
        if length + extra > MAX_QUERY_CHARS {
            break;
        }
        length += extra;
        kept.push(alternative);
    }
    if kept.is_empty() {
        let (first, _truncated) = truncate_chars(alternatives[0], MAX_QUERY_CHARS);
        return first;
    }
    kept.join("|")
}

/// Renders one selectable target as a lexical document.
///
/// The body and the aliases stay empty: the exposure carries no long text for a candidate,
/// which is exactly what Python's tree walk sees too.
fn document(target: &SelectableTarget) -> LexicalDocument {
    LexicalDocument {
        key: target.resolution.canonical_id.clone(),
        name: target.resolution.display_name.clone(),
        description: target.resolution.description.clone(),
        body: String::new(),
        aliases: Vec::new(),
        category: target.resolution.branch_path.join(" > "),
    }
}

/// The default [`TopKSelector`].
///
/// Mirrors Python `DefaultTopKSelector`, except that the generate selector is optional
/// (`with_generate_selector`) and a lexical selector fills in when it is absent or fails.
#[derive(Debug, Clone, Default)]
pub struct DefaultTopKSelector {
    /// The traversal config (the selection mode, the fallback mode and the truncation).
    config: ProgressiveSearchConfig,
    /// The injected model-facing selector; `None` selects lexically.
    generate: Option<GenerateFragmentSelector>,
}

impl DefaultTopKSelector {
    /// Builds the selector over `config`, with no generate selector (lexical only).
    #[must_use]
    pub fn new(config: ProgressiveSearchConfig) -> Self {
        Self {
            config,
            generate: None,
        }
    }

    /// Enables model delegation through `selector`.
    #[must_use]
    pub fn with_generate_selector(mut self, selector: GenerateFragmentSelector) -> Self {
        self.generate = Some(selector);
        self
    }

    /// Whether a generate selector is wired (and would be used in
    /// [`SelectionMode::Generate`]).
    #[must_use]
    pub fn is_model_backed(&self) -> bool {
        self.generate.is_some()
    }

    /// The lexical selector this one falls back to.
    fn lexical(&self) -> LexicalTopKSelector {
        LexicalTopKSelector::new(self.config.clone())
    }

    /// Selects through the generate selector, or reports why it cannot.
    fn select_generated(
        selector: &GenerateFragmentSelector,
        request: &SelectRequest<'_>,
        trace: &mut RetrieverTrace,
    ) -> Result<SelectionResult> {
        let cursor = request.cursor;
        let (raw_output, resolutions) = selector.generate(GenerateRequest {
            model: request.model,
            node_id: &cursor.node_id,
            depth: cursor.depth,
            top_k: cursor.top_k,
            query_messages: request.query_messages,
            fragment: &request.subtree.fragment,
            messages: &request.prompt.messages,
            trace,
        })?;
        let limit = if cursor.top_k == 0 {
            usize::MAX
        } else {
            cursor.top_k
        };
        let selected_targets: Vec<SelectableTarget> = resolutions
            .into_iter()
            .map(SelectableTarget::new)
            .take(limit)
            .collect();
        let is_abstain = selected_targets.is_empty() && is_abstain_output(&raw_output);
        Ok(SelectionResult {
            raw_output,
            selected_targets,
            is_abstain,
        })
    }
}

#[async_trait]
impl TopKSelector for DefaultTopKSelector {
    /// Describes the answer format of `subtree` from its exposure.
    fn build_protocol(&self, subtree: &CurrentSubtree) -> SelectionProtocol {
        SelectionProtocol {
            compact_codes_enabled: subtree.fragment.compact_codes_enabled,
            candidate_codes: subtree.fragment.candidate_codes.clone(),
            code_width: subtree.fragment.code_width,
            abstain_token: ABSTAIN_TOKEN.to_owned(),
        }
    }

    /// Selects the targets of `request.subtree`, degrading per the config.
    ///
    /// # Errors
    ///
    /// - the generate selector's own error, under
    ///   [`FallbackMode::Error`](crate::llm::FallbackMode);
    /// - [`Error::Unsupported`](crate::error::Error) when
    ///   [`SelectionMode::Generate`] has no generate selector and the fallback mode is
    ///   [`FallbackMode::Error`](crate::llm::FallbackMode).
    async fn select_topk(
        &self,
        request: SelectRequest<'_>,
        trace: &mut RetrieverTrace,
    ) -> Result<SelectionResult> {
        let cursor = request.cursor;
        if self.config.selection_mode == SelectionMode::Generate {
            match self.generate.as_ref() {
                Some(selector) => match Self::select_generated(selector, &request, trace) {
                    Ok(result) => {
                        record_selection(trace, cursor, MODE_GENERATE, &result);
                        return Ok(result);
                    }
                    Err(error) => match self.config.fallback_mode {
                        FallbackMode::Error => return Err(error),
                        FallbackMode::Deterministic => tracing::warn!(
                            node_id = %cursor.node_id,
                            depth = cursor.depth,
                            error = %error,
                            "progressive selection failed; falling back to the lexical selector"
                        ),
                    },
                },
                None => {
                    if self.config.fallback_mode == FallbackMode::Error {
                        return Err(unsupported(
                            "progressive selection needs a model: configure a generate selector \
                             with DefaultTopKSelector::with_generate_selector, use \
                             SelectionMode::Lexical, or FallbackMode::Deterministic",
                        ));
                    }
                }
            }
        }
        let result = self
            .lexical()
            .select(request.subtree, request.query_messages, cursor.top_k);
        record_selection(trace, cursor, MODE_LEXICAL, &result);
        Ok(result)
    }
}

/// Records the one `fragment_selected` event of a selection.
fn record_selection(
    trace: &mut RetrieverTrace,
    cursor: &SearchCursor,
    mode: &str,
    result: &SelectionResult,
) {
    let codes: Vec<&str> = result
        .selected_targets
        .iter()
        .map(SelectableTarget::code)
        .collect();
    let canonical_ids: Vec<&str> = result
        .selected_targets
        .iter()
        .map(SelectableTarget::canonical_id)
        .collect();
    let detail: BTreeMap<String, Value> = BTreeMap::from([
        ("mode".to_owned(), json!(mode)),
        ("raw_output".to_owned(), json!(result.raw_output)),
        ("selected_canonical_ids".to_owned(), json!(canonical_ids)),
        ("selected_codes".to_owned(), json!(codes)),
    ]);
    trace.record(
        FRAGMENT_SELECTED_EVENT,
        &cursor.node_id,
        cursor.depth,
        detail,
    );
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::type_complexity
)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::retrieval::search::flow::{PromptBundle, SelectableResolution};
    use crate::retrieval::search::truncation::ScoreTruncationConfig;
    use crate::retrieval::types::ROOT_NODE_ID;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// A resolution for `code`, terminal at `payload`.
    fn resolution(
        code: &str,
        canonical: &str,
        name: &str,
        description: &str,
    ) -> SelectableResolution {
        SelectableResolution {
            code: code.to_owned(),
            canonical_id: canonical.to_owned(),
            display_name: name.to_owned(),
            label: name.to_owned(),
            description: description.to_owned(),
            is_terminal: true,
            branch_path: vec![ROOT_NODE_ID.to_owned(), "Skills".to_owned()],
            node_id: None,
            item: None,
        }
    }

    /// A subtree exposing `resolutions` with the given codes.
    fn subtree(resolutions: Vec<SelectableResolution>) -> CurrentSubtree {
        let mut subtree = CurrentSubtree::default();
        for resolution in resolutions {
            subtree
                .fragment
                .candidate_codes
                .push(resolution.code.clone());
            subtree
                .fragment
                .code_to_resolution
                .insert(resolution.code.clone(), resolution.clone());
            subtree
                .selectable_targets
                .push(SelectableTarget::new(resolution));
        }
        subtree.fragment.fragment_fingerprint = "fingerprint".to_owned();
        subtree
    }

    /// A subtree whose three candidates match different terms.
    ///
    /// `item::alpha` carries "markdown" in its display name and `item::gamma` only in its
    /// description, so the field weighting ranks alpha first; `item::beta` matches neither
    /// and is dropped.
    fn ranked_subtree() -> CurrentSubtree {
        subtree(vec![
            resolution(
                "1",
                "item::alpha",
                "Markdown",
                "writes markdown documentation",
            ),
            resolution("2", "item::beta", "Shell", "runs shell commands"),
            resolution(
                "3",
                "item::gamma",
                "Gamma",
                "writes documentation and markdown files",
            ),
        ])
    }

    /// The cursor of a default subtree.
    fn cursor(top_k: usize) -> SearchCursor {
        SearchCursor::at(ROOT_NODE_ID, 0, vec![ROOT_NODE_ID.to_owned()], top_k)
    }

    /// A request borrowing the given pieces.
    fn request<'a>(
        cursor: &'a SearchCursor,
        messages: &'a [ChatMessage],
        subtree: &'a CurrentSubtree,
        prompt: &'a PromptBundle,
    ) -> SelectRequest<'a> {
        SelectRequest {
            model: Some("scripted"),
            cursor,
            query_messages: messages,
            subtree,
            prompt,
        }
    }

    #[test]
    fn abstain_output_recognizes_the_token_and_the_hints() {
        assert!(is_abstain_output("0"));
        assert!(is_abstain_output(" 0 \n"));
        assert!(is_abstain_output("No suitable candidate"));
        assert!(is_abstain_output("NO RELEVANT SKILL"));
        assert!(is_abstain_output("unrelated"));
        assert!(is_abstain_output("I cannot determine"));
        assert!(is_abstain_output("can't determine"));
        assert!(is_abstain_output("can’t determine"));
        assert!(is_abstain_output("无合适候选"));
        assert!(is_abstain_output("无法判断"));
        assert!(!is_abstain_output(""));
        assert!(!is_abstain_output("   "));
        assert!(!is_abstain_output("1,2"));
        assert!(!is_abstain_output("Alpha and Beta"));
    }

    #[test]
    fn query_text_joins_every_message() {
        assert_eq!(
            query_text(&[ChatMessage::system("sys"), ChatMessage::user("ask")]),
            "sys\nask"
        );
        assert_eq!(query_text(&[]), "");
    }

    #[test]
    fn lexical_selection_ranks_by_the_query() {
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig::default());
        let subtree = ranked_subtree();
        let messages = [ChatMessage::user("markdown")];
        let result = selector.select(&subtree, &messages, 5);
        assert!(!result.is_abstain);
        assert_eq!(result.raw_output, "");
        let ids: Vec<&str> = result
            .selected_targets
            .iter()
            .map(SelectableTarget::canonical_id)
            .collect();
        assert_eq!(
            ids,
            vec!["item::alpha", "item::gamma"],
            "the name match outranks the description-only match, and the non-match is dropped"
        );
        let codes: Vec<&str> = result
            .selected_targets
            .iter()
            .map(SelectableTarget::code)
            .collect();
        assert_eq!(codes, vec!["1", "3"]);
    }

    #[test]
    fn lexical_selection_clamps_and_treats_zero_as_unlimited() {
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig::default());
        let subtree = ranked_subtree();
        let messages = [ChatMessage::user("markdown documentation shell")];
        assert_eq!(
            selector
                .select(&subtree, &messages, 1)
                .selected_targets
                .len(),
            1
        );
        assert_eq!(
            selector
                .select(&subtree, &messages, 2)
                .selected_targets
                .len(),
            2
        );
        assert_eq!(
            selector
                .select(&subtree, &messages, 0)
                .selected_targets
                .len(),
            3,
            "top_k 0 is unlimited"
        );
        assert_eq!(
            selector
                .select(&subtree, &messages, 9)
                .selected_targets
                .len(),
            3
        );
    }

    #[test]
    fn a_blank_query_selects_nothing_and_is_not_an_abstention() {
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig::default());
        let blank = selector.select(&ranked_subtree(), &[], 5);
        assert!(blank.selected_targets.is_empty());
        assert!(!blank.is_abstain);
        let whitespace = selector.select(&ranked_subtree(), &[ChatMessage::user("   ")], 5);
        assert!(whitespace.selected_targets.is_empty());
        assert!(!whitespace.is_abstain);
    }

    #[test]
    fn a_conversational_query_still_ranks() {
        // The whole conversation is one matching expression; without the alternative
        // compilation no document would contain it as a phrase and nothing would rank.
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig::default());
        let messages = [
            ChatMessage::system("You are a careful routing assistant."),
            ChatMessage::user("Which skill runs shell commands for me?"),
        ];
        let result = selector.select(&ranked_subtree(), &messages, 5);
        let ids: Vec<&str> = result
            .selected_targets
            .iter()
            .map(SelectableTarget::canonical_id)
            .collect();
        assert_eq!(ids.first().copied(), Some("item::beta"));
        assert!(!result.selected_targets.is_empty());
    }

    #[test]
    fn an_oversized_query_is_bounded_and_repeatable() {
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig::default());
        let long = format!("markdown {}", "filler ".repeat(2_000));
        let first = selector.select(&ranked_subtree(), &[ChatMessage::user(long.clone())], 5);
        let second = selector.select(&ranked_subtree(), &[ChatMessage::user(long)], 5);
        assert_eq!(first, second, "the bounded expression is deterministic");
        let ids: Vec<&str> = first
            .selected_targets
            .iter()
            .map(SelectableTarget::canonical_id)
            .collect();
        assert_eq!(ids, vec!["item::alpha", "item::gamma"]);
    }

    #[test]
    fn ranking_alternatives_dedupe_case_insensitively() {
        assert_eq!(ranking_expression("  Alpha   alpha ALPHA "), "Alpha");
        assert_eq!(ranking_expression("alpha beta"), "alpha|beta");
        assert_eq!(ranking_expression(""), "");
        assert_eq!(ranking_expression("   \n\t "), "");
        assert_eq!(
            ranking_expression(&"x".repeat(MAX_QUERY_CHARS + 10)).len(),
            MAX_QUERY_CHARS
        );
    }

    #[test]
    fn an_empty_subtree_selects_nothing() {
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig::default());
        let result = selector.select(&CurrentSubtree::default(), &[ChatMessage::user("x")], 5);
        assert!(result.selected_targets.is_empty());
        assert!(!result.is_abstain);
    }

    #[test]
    fn the_truncation_config_is_honoured() {
        let selector = LexicalTopKSelector::new(ProgressiveSearchConfig {
            truncation: ScoreTruncationConfig {
                min_score: Some(1_000_000.0),
                ..ScoreTruncationConfig::default()
            },
            ..ProgressiveSearchConfig::default()
        });
        let result = selector.select(
            &ranked_subtree(),
            &[ChatMessage::user("markdown documentation")],
            5,
        );
        assert!(result.selected_targets.is_empty());
    }

    #[test]
    fn the_default_selector_builds_the_protocol_from_the_fragment() {
        let mut subtree = ranked_subtree();
        subtree.fragment.compact_codes_enabled = true;
        subtree.fragment.code_width = 2;
        let protocol =
            DefaultTopKSelector::new(ProgressiveSearchConfig::default()).build_protocol(&subtree);
        assert!(protocol.compact_codes_enabled);
        assert_eq!(protocol.code_width, 2);
        assert_eq!(protocol.candidate_codes, subtree.fragment.candidate_codes);
        assert_eq!(protocol.abstain_token, ABSTAIN_TOKEN);
    }

    #[tokio::test]
    async fn generate_mode_delegates_and_truncates() {
        let seen: Arc<Mutex<Vec<(Option<String>, String, usize, usize)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig::default())
            .with_generate_selector(GenerateFragmentSelector::new(Arc::new(
                move |request: GenerateRequest<'_>| {
                    recorder.lock().unwrap().push((
                        request.model.map(str::to_owned),
                        request.node_id.to_owned(),
                        request.depth,
                        request.top_k,
                    ));
                    assert_eq!(request.messages.len(), 2);
                    assert_eq!(request.fragment.fragment_fingerprint, "fingerprint");
                    assert_eq!(request.query_messages.len(), 1);
                    request.trace.record_simple("generate_called", "n", 0);
                    Ok((
                        "1".to_owned(),
                        vec![
                            resolution("1", "item::alpha", "Alpha", ""),
                            resolution("2", "item::beta", "Beta", ""),
                        ],
                    ))
                },
            )));
        assert!(selector.is_model_backed());

        let subtree = ranked_subtree();
        let cursor = cursor(1);
        let messages = [ChatMessage::user("ask")];
        let prompt = PromptBundle {
            fragment: subtree.fragment.clone(),
            protocol: SelectionProtocol::default(),
            messages: vec![ChatMessage::system("sys"), ChatMessage::user("ask")],
        };
        let mut trace = RetrieverTrace::default();
        let result = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap();
        assert_eq!(result.raw_output, "1");
        assert_eq!(result.selected_targets.len(), 1, "cursor.top_k truncates");
        assert!(!result.is_abstain);
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[(Some("scripted".to_owned()), ROOT_NODE_ID.to_owned(), 0, 1)]
        );
        let events = trace.event_types();
        assert!(
            events.contains(&"generate_called"),
            "the callback saw the trace"
        );
        assert_eq!(
            events.last().copied(),
            Some(FRAGMENT_SELECTED_EVENT),
            "one selection event, recorded last"
        );
        let event = trace.last().unwrap();
        assert_eq!(event.detail["mode"], json!(MODE_GENERATE));
        assert_eq!(event.detail["selected_codes"], json!(["1"]));
        assert_eq!(
            event.detail["selected_canonical_ids"],
            json!(["item::alpha"])
        );
        assert_eq!(event.detail["raw_output"], json!("1"));
    }

    #[tokio::test]
    async fn generate_mode_recognizes_an_abstention() {
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig::default())
            .with_generate_selector(GenerateFragmentSelector::new(Arc::new(|_| {
                Ok(("0".to_owned(), Vec::new()))
            })));
        let subtree = ranked_subtree();
        let cursor = cursor(5);
        let messages = [ChatMessage::user("ask")];
        let prompt = PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let result = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap();
        assert!(result.is_abstain);
        assert!(result.selected_targets.is_empty());
    }

    #[tokio::test]
    async fn a_failing_generate_fn_falls_back_to_lexical_deterministically() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig::default())
            .with_generate_selector(GenerateFragmentSelector::new(Arc::new(
                move |_request: GenerateRequest<'_>| {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Err(Error::llm("model unavailable"))
                },
            )));
        let subtree = ranked_subtree();
        let cursor = cursor(5);
        let messages = [ChatMessage::user("markdown documentation")];
        let prompt = PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let result = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!result.selected_targets.is_empty());
        let event = trace.last().unwrap();
        assert_eq!(event.detail["mode"], json!(MODE_LEXICAL));
        assert!(event.detail["raw_output"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_failing_generate_fn_surfaces_under_the_error_mode() {
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig {
            fallback_mode: FallbackMode::Error,
            ..ProgressiveSearchConfig::default()
        })
        .with_generate_selector(GenerateFragmentSelector::new(Arc::new(|_| {
            Err(Error::llm("model unavailable"))
        })));
        let subtree = ranked_subtree();
        let cursor = cursor(5);
        let messages = [ChatMessage::user("ask")];
        let prompt = PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let error = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Llm);
        assert!(error.message().contains("model unavailable"));
        assert!(trace.is_empty(), "a failed selection records nothing");
    }

    #[tokio::test]
    async fn generate_mode_without_a_selector_is_unsupported_under_the_error_mode() {
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig {
            fallback_mode: FallbackMode::Error,
            ..ProgressiveSearchConfig::default()
        });
        let subtree = ranked_subtree();
        let cursor = cursor(5);
        let messages = [ChatMessage::user("ask")];
        let prompt = PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let error = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap_err();
        assert!(crate::error::is_unsupported(&error));
        assert!(error.message().contains("needs a model"));
    }

    #[tokio::test]
    async fn generate_mode_without_a_selector_is_lexical_by_default() {
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig::default());
        let subtree = ranked_subtree();
        let cursor = cursor(5);
        let messages = [ChatMessage::user("shell commands")];
        let prompt = PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let result = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap();
        assert_eq!(
            result.selected_targets[0].canonical_id(),
            "item::beta",
            "the shell candidate ranks first"
        );
        assert_eq!(trace.event_types(), vec![FRAGMENT_SELECTED_EVENT]);
    }

    #[tokio::test]
    async fn the_lexical_mode_ignores_a_configured_generate_selector() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let selector = DefaultTopKSelector::new(ProgressiveSearchConfig {
            selection_mode: SelectionMode::Lexical,
            ..ProgressiveSearchConfig::default()
        })
        .with_generate_selector(GenerateFragmentSelector::new(Arc::new(
            move |_request: GenerateRequest<'_>| {
                counted.fetch_add(1, Ordering::SeqCst);
                Ok(("1".to_owned(), Vec::new()))
            },
        )));
        let subtree = ranked_subtree();
        let cursor = cursor(5);
        let messages = [ChatMessage::user("markdown")];
        let prompt = PromptBundle::default();
        let mut trace = RetrieverTrace::default();
        let result = selector
            .select_topk(request(&cursor, &messages, &subtree, &prompt), &mut trace)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.selected_targets.len(), 2);
        assert_eq!(trace.last().unwrap().detail["mode"], json!(MODE_LEXICAL));
    }
}
