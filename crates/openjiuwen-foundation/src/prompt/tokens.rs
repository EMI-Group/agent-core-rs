//! Token counting and message truncation.
//!
//! Every provider counts tokens differently, so this module defines the small
//! [`TokenCounter`] seam the rest of the SDK uses instead of hard-coding one
//! tokenizer: [`ApproxTokenCounter`] is the dependency-free estimate
//! (≈[`DEFAULT_CHARS_PER_TOKEN`] characters per token), and
//! [`TiktokenCounter`] (feature `tiktoken`) uses the `OpenAI` `cl100k_base` BPE
//! with the approximation as a fallback when the BPE cannot be built.
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
//!
//! let messages = vec![BaseMessage::system("be brief"), BaseMessage::user("hi")];
//! // 4 framing tokens + text cost, per message: (4 + 2) + (4 + 1).
//! assert_eq!(counter.count_messages(&messages), 11);
//! ```

use serde::{Deserialize, Serialize};

use super::message::BaseMessage;
use super::role::Role;

/// Approximate characters per token used by [`ApproxTokenCounter::default`].
///
/// `OpenAI`'s rule of thumb for English prose is four characters per token.
pub const DEFAULT_CHARS_PER_TOKEN: usize = 4;

/// Per-message overhead of the chat wire format: role markers, separators, and
/// the reply priming the provider adds around the content.
const CHAT_FRAMING_TOKENS: usize = 4;

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
/// Counting is by Unicode scalar values ([`str::chars`]), not bytes, so
/// multi-byte text is not over-counted. The counter is `serde`-serializable so
/// it can be embedded in a context-engine or model configuration.
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

/// Drops the oldest non-system messages until the conversation fits `max_tokens`.
///
/// System messages are never removed and keep their order: they are the
/// instructions that frame every turn, so a truncated conversation still has
/// them. Messages are evicted oldest-first among the non-system entries, and the
/// total is recomputed after every removal with the same
/// [`TokenCounter::count_messages`] the caller would use, so the returned count
/// can be trusted. Truncation stops when the budget is met, when only system
/// messages remain, or when the list is empty — `max_tokens = 0` therefore
/// always terminates, removing every non-system message.
///
/// Returns the number of messages removed; `0` means the conversation already
/// fit (system messages alone may exceed the budget).
pub fn truncate_messages(
    messages: &mut Vec<BaseMessage>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> usize {
    let mut removed = 0usize;
    loop {
        if counter.count_messages(messages) <= max_tokens {
            break;
        }
        let Some(index) = messages.iter().position(|m| m.role != Role::System) else {
            break;
        };
        messages.remove(index);
        removed += 1;
    }
    removed
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
