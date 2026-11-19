//! Token counting and message truncation.
//!
//! Every provider counts tokens differently, so this module defines the small
//! [`TokenCounter`] seam the rest of the SDK uses instead of hard-coding one
//! tokenizer: [`ApproxTokenCounter`] is the dependency-free estimate
//! (≈[`DEFAULT_CHARS_PER_TOKEN`] characters per token), and
//! `TiktokenCounter` (feature `tiktoken`) uses the `OpenAI` `cl100k_base` BPE
//! with the approximation as a fallback when the BPE cannot be built.
//!
//! **The two counters are not interchangeable.** [`ApproxTokenCounter`] is a
//! heuristic whose error margin is discussed on the type; it is the default
//! because it needs no vocabulary and never fails. Every counter reports which
//! kind it is through [`TokenCounter::is_exact`], so a caller that makes a
//! budget decision (like [`truncate_messages`]) can tell whether the number it
//! is comparing against a limit is the provider's own count or an estimate.
//!
//! Python keeps its real counters in `context_engine`; this module only offers
//! what prompt construction and the provider clients need to stay under a
//! context budget, plus [`truncate_messages`] to evict old turns while keeping
//! the system prompt intact.
//!
//! ```
//! use openjiuwen_foundation::prompt::message::BaseMessage;
//! use openjiuwen_foundation::prompt::tokens::{ApproxTokenCounter, TokenCounter};
//!
//! let counter = ApproxTokenCounter::new();
//! assert_eq!(counter.count_text("hello world"), 3);
//! assert!(!counter.is_exact(), "the character heuristic is an estimate");
//!
//! let messages = vec![BaseMessage::system("be brief"), BaseMessage::user("hi")];
//! // 4 framing tokens + text cost, per message: (4 + 2) + (4 + 1).
//! assert_eq!(counter.count_messages(&messages), 11);
//! ```

use std::sync::Once;

use serde::{Deserialize, Serialize};

use super::message::BaseMessage;
use super::role::Role;

/// The `tracing` target of every event emitted by this module.
const LOG_TARGET: &str = "openjiuwen_foundation::prompt::tokens";

/// Approximate characters per token used by [`ApproxTokenCounter::default`].
///
/// `OpenAI`'s rule of thumb for English prose is four characters per token.
pub const DEFAULT_CHARS_PER_TOKEN: usize = 4;

/// Per-message overhead of the chat wire format: role markers, separators, and
/// the reply priming the provider adds around the content.
const CHAT_FRAMING_TOKENS: usize = 4;

/// Guards the once-per-process warning for an inexact budget decision.
static INEXACT_BUDGET_WARNING: Once = Once::new();

/// Counts tokens for text fragments and for whole conversations.
///
/// The only required method is [`TokenCounter::count_text`]; the default
/// [`TokenCounter::count_messages`] applies one chat framing formula on top of
/// it, so every implementation counts conversations identically.
pub trait TokenCounter {
    /// Counts the tokens of one text fragment.
    ///
    /// The empty string always counts as `0`; leading, trailing, and inner
    /// whitespace is part of the payload and is counted like any other
    /// character.
    fn count_text(&self, text: &str) -> usize;

    /// Whether the counts are the provider's exact token counts.
    ///
    /// The default is `false`: a counter that only estimates must say so, so a
    /// budget decision can be weighed against its error margin.
    /// [`ApproxTokenCounter`] keeps the estimate and answers `false`;
    /// `TiktokenCounter` answers `true` only while its BPE vocabulary is really
    /// loaded, and `false` while it is falling back to the estimate.
    fn is_exact(&self) -> bool {
        false
    }

    /// Counts the tokens of a conversation.
    ///
    /// The formula for every message is
    /// `4 + count_text(message.text())` — the chat framing overhead plus the
    /// message's plain-text view ([`BaseMessage::text`], so multimodal
    /// non-text parts are not counted) — plus
    /// `count_text(call.name) + count_text(call.arguments)` for each entry in
    /// [`BaseMessage::tool_calls`]: a tool call costs what its name and its raw
    /// JSON arguments cost.
    ///
    /// The per-message costs are summed with saturating arithmetic, so a huge
    /// conversation cannot overflow into a smaller count.
    fn count_messages(&self, messages: &[BaseMessage]) -> usize {
        let mut total = 0usize;
        for message in messages {
            let mut cost = CHAT_FRAMING_TOKENS.saturating_add(self.count_text(&message.text()));
            for call in &message.tool_calls {
                cost = cost
                    .saturating_add(self.count_text(&call.name))
                    .saturating_add(self.count_text(&call.arguments));
            }
            total = total.saturating_add(cost);
        }
        total
    }
}

/// A dependency-free token estimate of roughly N characters per token.
///
/// Counting is [`ApproxTokenCounter::count_text`] =
/// `ceil(chars / chars_per_token)` by Unicode scalar values ([`str::chars`]),
/// not bytes, so multi-byte text is not over-counted. The counter is
/// `serde`-serializable so it can be embedded in a context-engine or model
/// configuration.
///
/// # It is an estimate, not a tokenizer
///
/// No vocabulary is consulted, so the result is only as good as the
/// characters-per-token ratio:
///
/// - English prose is usually within roughly ±20–30 % of the provider's BPE
///   count, because words, punctuation, and numbers tokenize unevenly;
/// - code, JSON, log output, and other symbol-dense text often costs 1.5–2×
///   more tokens than the character ratio suggests, so the estimate
///   *under*-counts;
/// - CJK text and emoji cost one or more tokens per character — far more than
///   the quarter the default ratio implies — so those are under-counted as
///   well.
///
/// [`TokenCounter::is_exact`] therefore answers `false` for this counter. Use it
/// for ordering, for a deliberately generous headroom, or where no vocabulary is
/// available (offline, unknown model); use `TiktokenCounter` when a limit must
/// be respected for real.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApproxTokenCounter {
    /// Characters per token; clamped to at least `1` (also when deserialized).
    #[serde(deserialize_with = "clamp_chars_per_token")]
    chars_per_token: usize,
}

impl ApproxTokenCounter {
    /// Creates a counter using [`DEFAULT_CHARS_PER_TOKEN`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_chars_per_token(DEFAULT_CHARS_PER_TOKEN)
    }

    /// Creates a counter with an explicit characters-per-token ratio.
    ///
    /// A ratio of `0` would divide by zero, so it is clamped to `1`: with that
    /// ratio one character counts as one token.
    #[must_use]
    pub fn with_chars_per_token(chars_per_token: usize) -> Self {
        Self {
            chars_per_token: chars_per_token.max(1),
        }
    }

    /// The effective characters-per-token ratio; never `0`.
    #[must_use]
    pub fn chars_per_token(&self) -> usize {
        self.chars_per_token.max(1)
    }
}

impl Default for ApproxTokenCounter {
    /// A counter using [`DEFAULT_CHARS_PER_TOKEN`].
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCounter for ApproxTokenCounter {
    /// `count_text(text) = ceil(chars / chars_per_token)`, and `0` for the
    /// empty string.
    fn count_text(&self, text: &str) -> usize {
        let chars = text.chars().count();
        if chars == 0 {
            return 0;
        }
        chars.div_ceil(self.chars_per_token())
    }
}

/// Deserializes a characters-per-token ratio, clamping `0` to `1`.
fn clamp_chars_per_token<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let chars_per_token = usize::deserialize(deserializer)?;
    Ok(chars_per_token.max(1))
}

#[cfg(feature = "tiktoken")]
mod tiktoken {
    use once_cell::sync::Lazy;
    use tiktoken_rs::CoreBPE;

    use super::{ApproxTokenCounter, TokenCounter};

    /// The shared `cl100k_base` BPE, built on first use.
    ///
    /// `CoreBPE` is `Send + Sync` (it keeps one regex per worker thread in an
    /// immutable vector), so it can live in a process-wide [`Lazy`]. Building
    /// the BPE reads an embedded vocabulary and does not touch the network; on
    /// failure the error is logged exactly once and the approximation is used
    /// instead.
    // `once_cell` is already a workspace dependency and the module contract
    // names it; `std::sync::LazyLock` would be an equivalent but unasked-for
    // substitution, so the lint is silenced here rather than at crate level.
    #[allow(clippy::non_std_lazy_statics)]
    static CL100K_BASE: Lazy<Option<CoreBPE>> = Lazy::new(|| match tiktoken_rs::cl100k_base() {
        Ok(bpe) => Some(bpe),
        Err(error) => {
            tracing::warn!(
                target: "openjiuwen_foundation::prompt::tokens",
                %error,
                "cl100k_base BPE unavailable; falling back to approximate token counts"
            );
            None
        }
    });

    /// A token counter backed by the `OpenAI` `cl100k_base` BPE.
    ///
    /// When the BPE cannot be initialized, [`TiktokenCounter::count_text`] falls
    /// back to [`ApproxTokenCounter`] instead of failing, so context accounting
    /// keeps working in every environment.
    #[derive(Debug, Clone, Default)]
    pub struct TiktokenCounter {
        /// Approximation used when the BPE is unavailable.
        fallback: ApproxTokenCounter,
    }

    impl TiktokenCounter {
        /// Creates a counter that uses the shared `cl100k_base` BPE.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl TokenCounter for TiktokenCounter {
        /// The number of `cl100k_base` tokens, or the approximate count when the
        /// BPE is unavailable. The empty string counts as `0`.
        fn count_text(&self, text: &str) -> usize {
            if text.is_empty() {
                return 0;
            }
            match CL100K_BASE.as_ref() {
                Some(bpe) => bpe.encode_ordinary(text).len(),
                None => self.fallback.count_text(text),
            }
        }

        /// `true` only while the `cl100k_base` vocabulary is loaded: after a
        /// failed initialization the counts come from the approximation, so
        /// reporting exactness would be a lie.
        fn is_exact(&self) -> bool {
            CL100K_BASE.is_some()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn exactness_tracks_whether_the_bpe_is_really_loaded() {
            let counter = TiktokenCounter::new();
            assert_eq!(counter.is_exact(), CL100K_BASE.is_some());
        }
    }
}

#[cfg(feature = "tiktoken")]
pub use tiktoken::TiktokenCounter;

/// Counts the tokens of a conversation with `counter`.
///
/// A convenience wrapper over [`TokenCounter::count_messages`] for call sites
/// that do not want the trait in scope.
#[must_use]
pub fn count_message_tokens(messages: &[BaseMessage], counter: &dyn TokenCounter) -> usize {
    counter.count_messages(messages)
}

/// Drops the oldest turns until the conversation fits `max_tokens`.
///
/// System messages are never removed and keep their order: they are the
/// instructions that frame every turn, so a truncated conversation still has
/// them. The remaining messages are grouped into *turns* — a `user` message plus
/// the `assistant`/`tool` messages that follow it — and a turn is evicted as a
/// whole, oldest first. Evicting whole turns keeps the sequence a provider
/// accepts: a `tool` result never survives without the `assistant` message whose
/// `tool_calls` produced it, and no `assistant`/`tool` turn is left at the front.
/// A conversation that begins with an `assistant`/`tool` run therefore loses that
/// run even when it already fits the budget (the return value reports it).
///
/// Truncation stops when the budget is met and the retained sequence starts with
/// a `user` (or only system messages remain), when no evictable turn is left, or
/// when the list is empty — `max_tokens = 0` therefore always terminates,
/// removing every non-system message.
///
/// The token total is computed once and each evicted turn's own cost is
/// subtracted from it, so truncating a long conversation costs one count per
/// turn instead of one count per removal.
///
/// Returns the number of messages removed; `0` means nothing was dropped (the
/// conversation already fit and starts with a `user` message). System messages
/// alone may exceed the budget.
///
/// This is the one token-budget decision in this crate, so a counter that only
/// estimates is called out: the first inexact truncation in the process logs a
/// `tracing::warn!` explaining that the eviction point carries the counter's
/// error margin. The warning is emitted once because truncation runs on every
/// turn.
pub fn truncate_messages(
    messages: &mut Vec<BaseMessage>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> usize {
    warn_once_for_inexact_budget(counter);
    let turns = turn_spans(messages);
    if turns.is_empty() {
        return 0;
    }
    let costs: Vec<usize> = turns
        .iter()
        .map(|turn| counter.count_messages(&messages[turn.start..turn.end]))
        .collect();
    let mut total = counter.count_messages(messages);
    let mut removed = 0usize;
    let mut first_kept = 0usize;
    while let Some(turn) = turns.get(first_kept) {
        let over_budget = total > max_tokens;
        let leading_non_user = messages[turn.start].role != Role::User;
        if !over_budget && !leading_non_user {
            break;
        }
        total = total.saturating_sub(costs[first_kept]);
        removed += turn.end - turn.start;
        first_kept += 1;
    }
    // Drain from the back: every earlier span keeps its original indices.
    for turn in turns[..first_kept].iter().rev() {
        messages.drain(turn.start..turn.end);
    }
    removed
}

/// One evictable run of non-system messages, as a half-open index range.
struct Turn {
    start: usize,
    end: usize,
}

/// Groups `messages` into turns: a `user` message plus the `assistant`/`tool`
/// messages after it.
///
/// A `system` message closes the current turn and is never part of one; a run of
/// `assistant`/`tool` messages with no preceding `user` forms its own turn (and
/// is evicted first, keeping the retained sequence provider-valid).
fn turn_spans(messages: &[BaseMessage]) -> Vec<Turn> {
    let mut turns = Vec::new();
    let mut current: Option<Turn> = None;
    for (index, message) in messages.iter().enumerate() {
        match message.role {
            Role::System => {
                if let Some(turn) = current.take() {
                    turns.push(turn);
                }
            }
            Role::User => {
                if let Some(turn) = current.take() {
                    turns.push(turn);
                }
                current = Some(Turn {
                    start: index,
                    end: index + 1,
                });
            }
            _ => match &mut current {
                Some(turn) => turn.end = index + 1,
                None => {
                    current = Some(Turn {
                        start: index,
                        end: index + 1,
                    });
                }
            },
        }
    }
    if let Some(turn) = current {
        turns.push(turn);
    }
    turns
}

/// Warns once per process when a budget decision uses an inexact counter.
fn warn_once_for_inexact_budget(counter: &dyn TokenCounter) {
    if counter.is_exact() {
        return;
    }
    INEXACT_BUDGET_WARNING.call_once(|| {
        tracing::warn!(
            target: LOG_TARGET,
            chars_per_token = DEFAULT_CHARS_PER_TOKEN,
            "truncating a conversation against a token budget with an approximate counter; \
             the eviction point can be off by the counter's error margin, so leave headroom \
             or use an exact counter"
        );
    });
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::prompt::tool_call::ToolCall;

    fn approx() -> ApproxTokenCounter {
        ApproxTokenCounter::new()
    }

    #[test]
    fn default_ratio_is_documented_constant() {
        assert_eq!(DEFAULT_CHARS_PER_TOKEN, 4);
        assert_eq!(approx().chars_per_token(), DEFAULT_CHARS_PER_TOKEN);
        assert_eq!(approx(), ApproxTokenCounter::default());
    }

    #[test]
    fn empty_text_costs_nothing() {
        assert_eq!(approx().count_text(""), 0);
        assert_eq!(
            ApproxTokenCounter::with_chars_per_token(1).count_text(""),
            0
        );
    }

    #[test]
    fn whitespace_counts_as_payload() {
        // Only the empty string is free; spaces have width like any character.
        assert_eq!(approx().count_text(" "), 1);
        assert_eq!(approx().count_text("    "), 1);
        assert_eq!(approx().count_text("     "), 2);
        assert_eq!(approx().count_text("\n\t"), 1);
    }

    #[test]
    fn ascii_counts_by_character_not_bytes() {
        assert_eq!(approx().count_text("hi"), 1);
        assert_eq!(approx().count_text("hello"), 2);
        assert_eq!(approx().count_text("hello world"), 3);
        assert_eq!(approx().count_text("abcdefgh"), 2);
        assert_eq!(approx().count_text("abcdefghi"), 3);
    }

    #[test]
    fn multi_byte_text_is_not_over_counted() {
        // Four CJK characters: one token at four chars per token, where a
        // byte-based count (12 bytes) would report three.
        assert_eq!(approx().count_text("你好世界"), 1);
        assert_eq!(approx().count_text("héllo"), 2);
        assert_eq!(approx().count_text("😀😀😀😀"), 1);
    }

    #[test]
    fn ratio_variants_change_the_count() {
        let per_char = ApproxTokenCounter::with_chars_per_token(1);
        assert_eq!(per_char.chars_per_token(), 1);
        assert_eq!(per_char.count_text("hello"), 5);

        let two = ApproxTokenCounter::with_chars_per_token(2);
        assert_eq!(two.count_text("hello"), 3);

        let wide = ApproxTokenCounter::with_chars_per_token(16);
        assert_eq!(wide.count_text("hello"), 1);
    }

    #[test]
    fn zero_ratio_is_clamped_to_one() {
        let clamped = ApproxTokenCounter::with_chars_per_token(0);
        assert_eq!(clamped.chars_per_token(), 1);
        assert_eq!(clamped.count_text("hello"), 5);
    }

    #[test]
    fn counter_round_trips_through_json() {
        let counter = ApproxTokenCounter::with_chars_per_token(3);
        let value = serde_json::to_value(counter).unwrap();
        assert_eq!(value, json!({"chars_per_token": 3}));
        assert_eq!(
            serde_json::from_value::<ApproxTokenCounter>(value).unwrap(),
            counter
        );
    }

    #[test]
    fn json_missing_ratio_defaults_and_zero_ratio_clamps() {
        let defaulted: ApproxTokenCounter = serde_json::from_value(json!({})).unwrap();
        assert_eq!(defaulted, approx());

        let clamped: ApproxTokenCounter =
            serde_json::from_value(json!({"chars_per_token": 0})).unwrap();
        assert_eq!(clamped.chars_per_token(), 1);
        assert_eq!(clamped.count_text("hi"), 2);
    }

    #[test]
    fn count_messages_of_nothing_is_zero() {
        assert_eq!(approx().count_messages(&[]), 0);
        assert_eq!(count_message_tokens(&[], &approx()), 0);
    }

    #[test]
    fn the_approximation_reports_itself_as_inexact() {
        assert!(!approx().is_exact());
        assert!(!ApproxTokenCounter::with_chars_per_token(1).is_exact());
        assert_eq!(
            count_message_tokens(&[BaseMessage::user("hi")], &approx()),
            5
        );
    }

    #[test]
    fn a_custom_counter_inherits_the_inexact_default() {
        struct FixedOne;

        impl TokenCounter for FixedOne {
            fn count_text(&self, text: &str) -> usize {
                usize::from(!text.is_empty())
            }
        }

        assert!(!FixedOne.is_exact());
        assert_eq!(
            count_message_tokens(&[BaseMessage::user("hi")], &FixedOne),
            5
        );
    }

    #[test]
    fn the_exactness_flag_does_not_change_the_counted_numbers() {
        let messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::user("hello world"),
            BaseMessage::assistant("hi").with_tool_calls(vec![ToolCall::new("get_weather")
                .with_id("call_1")
                .with_arguments(json!({"city": "Paris"}))]),
        ];
        assert_eq!(count_message_tokens(&messages, &approx()), 25);
        assert_eq!(approx().count_messages(&messages), 25);
        assert!(!approx().is_exact());
    }

    #[test]
    fn count_messages_adds_framing_and_text() {
        // "hello world" = 11 chars -> 3 tokens, plus 4 framing tokens.
        let messages = vec![BaseMessage::user("hello world")];
        assert_eq!(approx().count_messages(&messages), 7);

        // Two messages: 2 * 4 + 1 ("hi") + 2 ("hello").
        let messages = vec![BaseMessage::user("hello"), BaseMessage::assistant("hi")];
        assert_eq!(approx().count_messages(&messages), 11);
    }

    #[test]
    fn count_messages_counts_tool_calls_by_name_and_arguments() {
        // 4 framing + 0 text + 3 ("get_weather") + 4 ({"city":"Paris"}, 16 chars).
        let call = ToolCall::new("get_weather").with_arguments(json!({"city": "Paris"}));
        let messages = vec![BaseMessage::assistant("").with_tool_calls(vec![call])];
        assert_eq!(approx().count_messages(&messages), 11);
    }

    #[test]
    fn count_messages_uses_the_plain_text_view() {
        let messages = vec![BaseMessage::user(vec![
            crate::prompt::content::ContentPart::Text {
                text: "abcd".to_owned(),
            },
            crate::prompt::content::ContentPart::ImageUrl {
                image_url: crate::prompt::content::ImageUrl {
                    url: "https://x/y.png".to_owned(),
                    detail: None,
                },
            },
        ])];
        // Only the text part is counted: 4 framing + 1 token.
        assert_eq!(approx().count_messages(&messages), 5);
    }

    #[test]
    fn free_function_matches_the_trait_method() {
        let messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::user("hello world"),
            BaseMessage::assistant("hi").with_tool_calls(vec![ToolCall::new("noop")]),
        ];
        assert_eq!(
            count_message_tokens(&messages, &approx()),
            approx().count_messages(&messages)
        );
    }

    #[test]
    fn truncate_does_nothing_when_within_budget() {
        let mut messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::user("hello"),
            BaseMessage::assistant("hi"),
        ];
        let before = messages.clone();
        let budget = approx().count_messages(&messages);
        assert_eq!(truncate_messages(&mut messages, budget, &approx()), 0);
        assert_eq!(messages, before);
    }

    #[test]
    fn truncate_drops_the_oldest_non_system_message_first() {
        let mut messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::user("first question"),
            BaseMessage::assistant("first answer"),
            BaseMessage::user("second question"),
        ];
        // Budget only fits the system message and the last user message.
        let kept = vec![messages[0].clone(), messages[3].clone()];
        let budget = approx().count_messages(&kept);

        assert_eq!(truncate_messages(&mut messages, budget, &approx()), 2);
        assert_eq!(messages, kept);
        assert!(approx().count_messages(&messages) <= budget);
    }

    #[test]
    fn truncate_preserves_every_system_message_in_order() {
        let mut messages = vec![
            BaseMessage::system("system one"),
            BaseMessage::user("old turn"),
            BaseMessage::system("system two"),
            BaseMessage::assistant("newest turn"),
        ];
        let removed = truncate_messages(&mut messages, 0, &approx());
        assert_eq!(removed, 2);
        assert_eq!(
            messages,
            vec![
                BaseMessage::system("system one"),
                BaseMessage::system("system two"),
            ]
        );
    }

    #[test]
    fn truncate_keeps_tool_results_with_their_assistant_tool_calls() {
        let first_call = ToolCall::new("lookup")
            .with_id("call_1")
            .with_arguments(json!({"q": "first"}));
        let second_call = ToolCall::new("lookup")
            .with_id("call_2")
            .with_arguments(json!({"q": "second"}));
        let mut messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::user("first question"),
            BaseMessage::assistant("").with_tool_calls(vec![first_call]),
            BaseMessage::tool("call_1", "first result"),
            BaseMessage::assistant("first answer"),
            BaseMessage::user("second question"),
            BaseMessage::assistant("").with_tool_calls(vec![second_call]),
            BaseMessage::tool("call_2", "second result"),
            BaseMessage::assistant("second answer"),
        ];
        let last_turn = &messages[5..];
        let budget =
            approx().count_messages(&[messages[0].clone()]) + approx().count_messages(last_turn);

        let removed = truncate_messages(&mut messages, budget, &approx());

        assert_eq!(removed, 4, "the whole first turn is evicted");
        assert_eq!(
            messages,
            vec![
                BaseMessage::system("be brief"),
                BaseMessage::user("second question"),
                BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("lookup")
                    .with_id("call_2")
                    .with_arguments(json!({"q": "second"}))]),
                BaseMessage::tool("call_2", "second result"),
                BaseMessage::assistant("second answer"),
            ]
        );
        assert!(
            approx().count_messages(&messages) <= budget,
            "the retained conversation exceeds the budget"
        );
        assert_eq!(
            messages[1].role,
            Role::User,
            "a truncated history must start with a user message"
        );
        // Every retained `tool` result has its `tool_calls` message in front.
        for (index, message) in messages.iter().enumerate() {
            if message.role != Role::Tool {
                continue;
            }
            let call_id = message.tool_call_id.as_deref();
            assert!(
                messages[..index].iter().any(|earlier| earlier
                    .tool_calls
                    .iter()
                    .any(|call| call.id.as_deref() == call_id)),
                "orphaned tool result at index {index}"
            );
        }
    }

    #[test]
    fn truncate_drops_a_leading_assistant_tool_run() {
        let call = ToolCall::new("lookup")
            .with_id("call_1")
            .with_arguments(json!({"q": "first"}));
        let mut messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::assistant("").with_tool_calls(vec![call]),
            BaseMessage::tool("call_1", "first result"),
            BaseMessage::user("only question"),
        ];
        // The history fits, but it starts with an assistant/tool run a provider
        // rejects, so that run is evicted anyway.
        let budget = approx().count_messages(&messages);
        assert_eq!(truncate_messages(&mut messages, budget, &approx()), 2);
        assert_eq!(
            messages,
            vec![
                BaseMessage::system("be brief"),
                BaseMessage::user("only question"),
            ]
        );
    }

    #[test]
    fn truncate_terminates_for_zero_budget() {
        let mut messages = vec![
            BaseMessage::system("s"),
            BaseMessage::user("a"),
            BaseMessage::tool("call_1", "b"),
            BaseMessage::assistant("c"),
        ];
        assert_eq!(truncate_messages(&mut messages, 0, &approx()), 3);
        assert_eq!(messages, vec![BaseMessage::system("s")]);
    }

    #[test]
    fn truncate_keeps_system_messages_that_exceed_the_budget() {
        let mut messages = vec![
            BaseMessage::system("a very long system prompt that costs several tokens"),
            BaseMessage::user("hi"),
        ];
        let removed = truncate_messages(&mut messages, 1, &approx());
        assert_eq!(removed, 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::System);
        // The system prompt alone is over budget, which is expected and finite.
        assert!(approx().count_messages(&messages) > 1);
    }

    #[test]
    fn truncate_empty_list_is_a_no_op() {
        let mut messages: Vec<BaseMessage> = Vec::new();
        assert_eq!(truncate_messages(&mut messages, 0, &approx()), 0);
        assert_eq!(messages, Vec::<BaseMessage>::new());
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_counts_a_sentence_and_agrees_with_the_estimate() {
        let counter = TiktokenCounter::new();
        let sentence = "The quick brown fox jumps over the lazy dog.";

        let bpe_tokens = counter.count_text(sentence);
        assert!(
            bpe_tokens > 0,
            "a real sentence must cost at least one token"
        );
        assert_eq!(counter.count_text(""), 0);

        // Loose cross-check so the test is not tied to one vocabulary revision:
        // the BPE count and the ~4 chars/token estimate are the same order.
        let estimate = approx().count_text(sentence);
        assert!(
            bpe_tokens <= estimate * 2 && bpe_tokens * 2 >= estimate,
            "BPE count {bpe_tokens} should be within 2x of the estimate {estimate}"
        );

        // Whatever path ran (BPE or fallback), messages are counted positively.
        assert!(counter.count_messages(&[BaseMessage::user(sentence)]) > 0);
    }
}
