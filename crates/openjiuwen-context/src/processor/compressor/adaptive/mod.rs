//! Adaptive compression strategy selection.
//!
//! Ports the model-chosen branch of Python's compressor family. A compressor
//! that enables the flag asks one bounded, non-streaming model call which pass
//! the history actually needs, and receives the durable task state of the span
//! as a [`MemoryBlockPlan`] in the same reply.
//!
//! Two paths always exist:
//!
//! - **Model** ([`StrategySource::Model`]) — the configured [`AdaptiveStrategy`]
//!   has an enabled flag and a [`Model`]; its reply names a
//!   [`CompressionStrategy`] plus the plan. The call is bounded by
//!   [`AdaptiveStrategyConfig::max_decision_tokens`], never streams, and is
//!   refused outside the allowlist.
//! - **Deterministic fallback** ([`StrategySource::DeterministicFallback`]) —
//!   no model, a disabled flag, a history too small to be worth a call, a
//!   provider error, an empty or unparsable reply, or an unknown strategy name.
//!   [`AdaptiveStrategy::deterministic`] measures the history and walks a fixed
//!   ladder, so selection is total and works offline.
//!
//! The deterministic ladder, first match wins:
//!
//! 1. no measurable history → [`CompressionStrategy::Noop`];
//! 2. stale tool results hold at least `min_savings_ratio` of the tokens →
//!    [`CompressionStrategy::MicroCompact`] (a tool's older results can be
//!    cleared in place);
//! 3. an assistant turn is in flight and that round holds at least the ratio →
//!    [`CompressionStrategy::CurrentRound`];
//! 4. at least three completed rounds hold at least the ratio →
//!    [`CompressionStrategy::RoundLevel`];
//! 5. at least one completed round holds at least the ratio →
//!    [`CompressionStrategy::DialogueSummary`];
//! 6. the history is over its budget → [`CompressionStrategy::FullCompact`];
//! 7. otherwise [`AdaptiveStrategyConfig::fallback_strategy`] (default
//!    [`CompressionStrategy::Noop`]) with a measured reason.
//!
//! `min_savings_ratio` is the fraction of the history's tokens a candidate span
//! must hold before rewriting it is considered worth a pass.
//!
//! # Compressor contract
//!
//! A compressor that enables [`AdaptiveConfig`] runs the selection after its own
//! trigger gate, and then:
//!
//! - [`CompressionStrategy::Noop`] — leave the history and the counters alone and
//!   report nothing;
//! - the compressor's own strategy — proceed with the normal span/summary logic,
//!   but build the block body from the [`MemoryBlockPlan`] (model-backed when the
//!   reply carried one, deterministic otherwise) instead of the plain summary;
//! - another strategy — return a [`ContextEvent`] whose `summary` records the
//!   recommendation and whose `messages_changed` is empty, and leave the history
//!   unchanged, so the processor the recommendation names can run instead.
//!
//! With [`AdaptiveConfig::enabled`] off, no selection happens at all: a
//! compressor behaves exactly as it did before this module existed.
//!
//! [`Model`]: openjiuwen_foundation::llm::Model
//! [`ContextEvent`]: crate::events::ContextEvent
pub mod plan;
pub mod prompt;

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::{BaseMessage, Role, TokenCounter};
use serde::{Deserialize, Serialize};

use crate::events::{ContextEvent, ContextEventKind};
use crate::processor::compressor::{
    group_completed_rounds, is_memory_block, MemoryBlockKind, RecallArchiveMarker,
    CURRENT_ROUND_MEMORY_BLOCK, DIALOGUE_MEMORY_BLOCK, ROUND_LEVEL_MEMORY_BLOCK,
};
use crate::processor::offloader::resolve_tool_name;
use crate::processor::ProcessorContext;
use crate::tokens::{count_message, count_messages};

pub use self::plan::MemoryBlockPlan;

/// Completed rounds at which the ladder prefers round-level compaction.
const ROUND_LEVEL_MIN_ROUNDS: usize = 3;

/// The passes a model (or the deterministic ladder) may choose.
///
/// The wire names are the compressor family's vocabulary:
/// `dialogue_summary`, `current_round`, `round_level`, `full_compact`,
/// `micro_compact` and `noop`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionStrategy {
    /// Summarise older completed dialogue rounds into one memory block.
    DialogueSummary,
    /// Compress the earlier part of the in-flight round.
    CurrentRound,
    /// Build per-round memory blocks and merge them.
    RoundLevel,
    /// Compact the whole history down to its boundary.
    FullCompact,
    /// Clear the bodies of stale tool results in place.
    MicroCompact,
    /// Leave the history alone: nothing is worth compressing yet.
    #[default]
    Noop,
}

impl CompressionStrategy {
    /// The snake-case name used on the wire and in model prompts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DialogueSummary => "dialogue_summary",
            Self::CurrentRound => "current_round",
            Self::RoundLevel => "round_level",
            Self::FullCompact => "full_compact",
            Self::MicroCompact => "micro_compact",
            Self::Noop => "noop",
        }
    }

    /// Whether this strategy leaves the history alone.
    #[must_use]
    pub fn is_noop(self) -> bool {
        self == Self::Noop
    }

    /// Parses a strategy name out of a model reply.
    ///
    /// Parsing is tolerant: surrounding whitespace, quotes and backticks are
    /// stripped, case is ignored, and `-`/spaces normalise to `_`. Documented
    /// aliases are accepted (`dialogue`, `current`, `round`, `compact`, `micro`,
    /// `none`, `skip`). Nothing here fails: an empty or unknown name yields
    /// [`CompressionStrategy::Noop`] with a reason the caller can record.
    #[must_use]
    pub fn parse(text: &str) -> StrategyParse {
        let normalised = normalise_name(text);
        let strategy = match normalised.as_str() {
            "dialogue_summary" | "dialogue" | "dialogue_compressor" | "dialogue_compression" => {
                Self::DialogueSummary
            }
            "current_round" | "current" | "current_round_compressor" => Self::CurrentRound,
            "round_level" | "round" | "round_level_compressor" => Self::RoundLevel,
            "full_compact" | "compact" | "full" => Self::FullCompact,
            "micro_compact" | "micro" => Self::MicroCompact,
            "noop" | "no_op" | "none" | "skip" | "nothing" => Self::Noop,
            _ => {
                return StrategyParse {
                    strategy: Self::Noop,
                    reason: Some(if normalised.is_empty() {
                        "the compression strategy name is empty; using noop".to_owned()
                    } else {
                        format!("unknown compression strategy `{normalised}`; using noop")
                    }),
                }
            }
        };
        StrategyParse {
            strategy,
            reason: None,
        }
    }
}

impl std::fmt::Display for CompressionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome of [`CompressionStrategy::parse`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StrategyParse {
    /// The parsed strategy, [`CompressionStrategy::Noop`] when unknown.
    pub strategy: CompressionStrategy,
    /// Why the name could not be parsed, `None` on a clean parse.
    pub reason: Option<String>,
}

impl Default for StrategyParse {
    fn default() -> Self {
        Self {
            strategy: CompressionStrategy::Noop,
            reason: None,
        }
    }
}

/// Where a [`StrategyDecision`] came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategySource {
    /// A model named the strategy.
    Model,
    /// The deterministic ladder chose it.
    #[default]
    DeterministicFallback,
}

/// One strategy choice plus the state the chosen pass should carry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StrategyDecision {
    /// The pass to run next.
    pub strategy: CompressionStrategy,
    /// Why it was chosen; never empty.
    pub reason: String,
    /// Whether a model or the deterministic ladder decided.
    pub source: StrategySource,
    /// The durable state of the span, when the model supplied one.
    pub plan: Option<MemoryBlockPlan>,
}

impl StrategyDecision {
    /// Whether the decision asks for no change at all.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.strategy.is_noop()
    }
}

/// The measured view the selection reads.
///
/// Build it with [`StrategyInput::measure`] from a history and a token counter,
/// or construct it directly in tests. Every field is a plain measurement, so the
/// view is deterministic and serializes for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdaptiveContext {
    /// Tokens the whole history holds.
    pub total_tokens: usize,
    /// Hard budget handed to the compressor, `0` when unknown.
    pub budget_tokens: usize,
    /// Trigger threshold handed to the compressor, `0` when unknown.
    pub trigger_tokens: usize,
    /// Number of messages in the history.
    pub messages: usize,
    /// Number of system messages.
    pub system_messages: usize,
    /// Number of user messages.
    pub user_messages: usize,
    /// Number of assistant messages.
    pub assistant_messages: usize,
    /// Number of tool-result messages.
    pub tool_messages: usize,
    /// Number of completed dialogue rounds.
    pub completed_rounds: usize,
    /// Tokens held by the completed rounds.
    pub completed_round_tokens: usize,
    /// Whether the newest message is not a closing assistant turn.
    pub assistant_in_flight: bool,
    /// Tokens after the last user message, the in-flight round's bulk.
    pub in_flight_round_tokens: usize,
    /// Number of tool-result messages.
    pub tool_result_count: usize,
    /// Tokens held by tool-result messages.
    pub tool_result_tokens: usize,
    /// Number of tool results a newer result of the same tool supersedes.
    pub stale_tool_result_count: usize,
    /// Tokens held by those stale tool results.
    pub stale_tool_result_tokens: usize,
    /// Number of memory blocks already in the history.
    pub memory_blocks: usize,
    /// Compressions the caller already performed on this context.
    pub recent_compressions: usize,
    /// Strategy of the most recent adaptive decision, when known.
    pub last_strategy: Option<CompressionStrategy>,
}

impl AdaptiveContext {
    /// Measures `messages` and leaves the caller-supplied fields at their
    /// defaults.
    #[must_use]
    pub fn from_messages(
        messages: &[BaseMessage],
        counter: &(dyn TokenCounter + Send + Sync),
    ) -> Self {
        let mut context = Self {
            total_tokens: count_messages(counter, messages),
            messages: messages.len(),
            ..Self::default()
        };
        context.fill_roles(messages);
        context.fill_rounds(messages, counter);
        context.fill_in_flight(messages, counter);
        context.fill_tool_results(messages, counter);
        context.memory_blocks = messages
            .iter()
            .filter(|message| {
                [
                    DIALOGUE_MEMORY_BLOCK,
                    CURRENT_ROUND_MEMORY_BLOCK,
                    ROUND_LEVEL_MEMORY_BLOCK,
                ]
                .iter()
                .any(|marker| is_memory_block(message, marker))
            })
            .count();
        context
    }

    /// The budget the ladder compares against: the hard budget when known, the
    /// trigger threshold otherwise, `0` when neither is set.
    #[must_use]
    pub fn effective_budget(&self) -> usize {
        if self.budget_tokens > 0 {
            self.budget_tokens
        } else {
            self.trigger_tokens
        }
    }

    /// Whether there is nothing measurable to compress.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages == 0 || self.total_tokens == 0
    }

    /// Per-role message counts.
    fn fill_roles(&mut self, messages: &[BaseMessage]) {
        for message in messages {
            match message.role {
                Role::System => self.system_messages += 1,
                Role::User => self.user_messages += 1,
                Role::Assistant => self.assistant_messages += 1,
                Role::Tool => self.tool_messages += 1,
                Role::Other(_) => {}
            }
        }
    }

    /// Completed-round count and token volume.
    fn fill_rounds(
        &mut self,
        messages: &[BaseMessage],
        counter: &(dyn TokenCounter + Send + Sync),
    ) {
        let rounds = group_completed_rounds(messages);
        self.completed_rounds = rounds.len();
        self.completed_round_tokens = rounds
            .into_iter()
            .map(|round| count_messages(counter, &messages[round]))
            .sum();
    }

    /// Whether an assistant turn is in flight and how large that round is.
    fn fill_in_flight(
        &mut self,
        messages: &[BaseMessage],
        counter: &(dyn TokenCounter + Send + Sync),
    ) {
        let Some(last) = messages.last() else {
            return;
        };
        self.assistant_in_flight = !(last.role == Role::Assistant && last.tool_calls.is_empty());
        if !self.assistant_in_flight {
            return;
        }
        if let Some(last_user) = messages
            .iter()
            .rposition(|message| message.role == Role::User)
        {
            self.in_flight_round_tokens = count_messages(counter, &messages[last_user + 1..]);
        }
    }

    /// Tool-result volume and the stale subset a newer result supersedes.
    fn fill_tool_results(
        &mut self,
        messages: &[BaseMessage],
        counter: &(dyn TokenCounter + Send + Sync),
    ) {
        let mut per_tool: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (index, message) in messages.iter().enumerate() {
            if message.role != Role::Tool {
                continue;
            }
            self.tool_result_count += 1;
            self.tool_result_tokens += count_message(counter, message);
            if let Some(name) = resolve_tool_name(messages, index) {
                per_tool.entry(name).or_default().push(index);
            }
        }
        for indices in per_tool.values() {
            for index in indices.iter().take(indices.len().saturating_sub(1)) {
                self.stale_tool_result_count += 1;
                self.stale_tool_result_tokens += count_message(counter, &messages[*index]);
            }
        }
    }
}

/// The inputs [`AdaptiveContext`] is measured from.
///
/// The builder keeps the measurement call sites short: a compressor supplies the
/// history, its counter, and the budget/trigger it was handed, then calls
/// [`StrategyInput::measure`].
pub struct StrategyInput<'a> {
    /// History to measure.
    pub messages: &'a [BaseMessage],
    /// Counter every measurement uses.
    pub counter: &'a (dyn TokenCounter + Send + Sync),
    /// Hard budget, `0` when unknown.
    pub budget_tokens: usize,
    /// Trigger threshold, `0` when unknown.
    pub trigger_tokens: usize,
    /// Compressions already performed on this context.
    pub recent_compressions: usize,
    /// Strategy of the most recent adaptive decision.
    pub last_strategy: Option<CompressionStrategy>,
}

impl<'a> StrategyInput<'a> {
    /// Creates inputs over `messages` with no budget or history.
    #[must_use]
    pub fn new(messages: &'a [BaseMessage], counter: &'a (dyn TokenCounter + Send + Sync)) -> Self {
        Self {
            messages,
            counter,
            budget_tokens: 0,
            trigger_tokens: 0,
            recent_compressions: 0,
            last_strategy: None,
        }
    }

    /// Sets the hard budget.
    #[must_use]
    pub fn with_budget(mut self, budget_tokens: usize) -> Self {
        self.budget_tokens = budget_tokens;
        self
    }

    /// Sets the trigger threshold.
    #[must_use]
    pub fn with_trigger(mut self, trigger_tokens: usize) -> Self {
        self.trigger_tokens = trigger_tokens;
        self
    }

    /// Sets the recent-compression history.
    #[must_use]
    pub fn with_history(
        mut self,
        recent_compressions: usize,
        last_strategy: Option<CompressionStrategy>,
    ) -> Self {
        self.recent_compressions = recent_compressions;
        self.last_strategy = last_strategy;
        self
    }

    /// Measures the history into an [`AdaptiveContext`].
    #[must_use]
    pub fn measure(&self) -> AdaptiveContext {
        let mut context = AdaptiveContext::from_messages(self.messages, self.counter);
        context.budget_tokens = self.budget_tokens;
        context.trigger_tokens = self.trigger_tokens;
        context.recent_compressions = self.recent_compressions;
        context.last_strategy = self.last_strategy;
        context
    }
}

/// Configuration of the model-chosen strategy path.
///
/// The model is not serializable, so it is supplied through
/// [`AdaptiveStrategy::with_model`] (or
/// [`AdaptiveStrategy::new`], which adopts
/// [`AdaptiveStrategyConfig::model`]) and skipped by serde. Every other field is
/// `serde`-defaulted.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdaptiveStrategyConfig {
    /// Whether a model may choose the strategy; `false` keeps the ladder.
    pub enabled: bool,
    /// Upper bound on the decision reply's tokens.
    pub max_decision_tokens: usize,
    /// Fraction of the history a span must hold to be worth compressing.
    pub min_savings_ratio: f64,
    /// Strategies a model may choose; empty allows every strategy.
    pub strategy_allowlist: Vec<CompressionStrategy>,
    /// Whether the deterministic extractor fills plan sections gaps.
    pub task_extraction: bool,
    /// Whether rendered blocks carry the structured plan sections.
    pub json_schema_blocks: bool,
    /// Strategy the ladder ends on when no measured span dominates.
    pub fallback_strategy: CompressionStrategy,
    /// Model that may choose the strategy.
    #[serde(skip)]
    pub model: Option<Arc<dyn Model>>,
}

impl Default for AdaptiveStrategyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_decision_tokens: 1_024,
            min_savings_ratio: 0.15,
            strategy_allowlist: Vec::new(),
            task_extraction: true,
            json_schema_blocks: true,
            fallback_strategy: CompressionStrategy::Noop,
            model: None,
        }
    }
}

impl std::fmt::Debug for AdaptiveStrategyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveStrategyConfig")
            .field("enabled", &self.enabled)
            .field("max_decision_tokens", &self.max_decision_tokens)
            .field("min_savings_ratio", &self.min_savings_ratio)
            .field("strategy_allowlist", &self.strategy_allowlist)
            .field("task_extraction", &self.task_extraction)
            .field("json_schema_blocks", &self.json_schema_blocks)
            .field("fallback_strategy", &self.fallback_strategy)
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

impl AdaptiveStrategyConfig {
    /// Checks the decision budget and the savings ratio.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for a zero decision budget and for a savings
    /// ratio outside `(0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.max_decision_tokens == 0 {
            return Err(Error::Config(
                "adaptive strategy: max_decision_tokens must be greater than zero".to_owned(),
            ));
        }
        if self.min_savings_ratio <= 0.0 || self.min_savings_ratio > 1.0 {
            return Err(Error::Config(format!(
                "adaptive strategy: min_savings_ratio ({}) must be in (0, 1]",
                self.min_savings_ratio
            )));
        }
        Ok(())
    }

    /// Whether a model may choose `strategy`.
    ///
    /// [`CompressionStrategy::Noop`] is always allowed — a model must be able to
    /// say "leave it alone" — and an empty allowlist allows everything.
    #[must_use]
    pub fn allows(&self, strategy: CompressionStrategy) -> bool {
        strategy.is_noop()
            || self.strategy_allowlist.is_empty()
            || self.strategy_allowlist.contains(&strategy)
    }
}

/// Picks the compression pass, with a model when one is configured.
#[derive(Clone)]
pub struct AdaptiveStrategy {
    config: AdaptiveStrategyConfig,
    model: Option<Arc<dyn Model>>,
}

impl std::fmt::Debug for AdaptiveStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveStrategy")
            .field("config", &self.config)
            .field("has_model", &self.model.is_some())
            .finish()
    }
}

impl AdaptiveStrategy {
    /// Creates the strategy described by `config`, adopting its model.
    #[must_use]
    pub fn new(config: AdaptiveStrategyConfig) -> Self {
        let model = config.model.clone();
        Self { config, model }
    }

    /// Sets the model used for the decision call.
    #[must_use]
    pub fn with_model(mut self, model: Option<Arc<dyn Model>>) -> Self {
        self.model = model;
        self
    }

    /// The configuration this strategy runs with.
    #[must_use]
    pub fn config(&self) -> &AdaptiveStrategyConfig {
        &self.config
    }

    /// Whether a model is wired, so the model path is available.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Chooses the pass for `context`.
    ///
    /// The deterministic ladder is always computed first; it is also the whole
    /// answer when the flag is off, no model is wired, or the ladder already
    /// finds nothing worth compressing (a payload too small for a decision
    /// call). Otherwise one bounded `Model::invoke` asks for the JSON decision;
    /// every failure — transport, empty reply, unparsable JSON, unknown strategy,
    /// allowlist rejection — returns the ladder's decision with the failure in
    /// [`StrategyDecision::reason`] and
    /// [`StrategySource::DeterministicFallback`]. This function never panics and
    /// never propagates a model error.
    pub async fn select(&self, context: &AdaptiveContext) -> StrategyDecision {
        let ladder = self.deterministic(context);
        if !self.config.enabled || self.config.validate().is_err() {
            return ladder;
        }
        let Some(model) = self.model.as_ref() else {
            return ladder;
        };
        if ladder.strategy.is_noop() {
            return ladder;
        }

        let user_prompt = prompt::strategy_request(context, &self.config.strategy_allowlist);
        let mut request = ModelRequestConfig::default()
            .with_model_name(model.model_name())
            .with_max_tokens(u32::try_from(self.config.max_decision_tokens).unwrap_or(u32::MAX));
        // No-op outside a `with_operation` scope; inside one it tags the request
        // with the purpose and operation id that correlate the provider call.
        crate::telemetry::stamp_current_operation(&mut request);
        let messages = [
            BaseMessage::system(prompt::STRATEGY_SYSTEM_PROMPT),
            BaseMessage::user(user_prompt),
        ];
        let reply = match model.invoke(&messages, &request).await {
            Ok(response) => {
                // Usage is observability only: a decision that answered nothing
                // still cost tokens, so report it before parsing the reply.
                crate::processor::record_model_usage(&response.usage, model.model_name());
                response
                    .message()
                    .map(|message| message.text().into_owned())
                    .unwrap_or_default()
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "adaptive strategy model call failed; using the deterministic ladder"
                );
                return fallback_with(&ladder, format!("the strategy model call failed: {error}"));
            }
        };
        let parsed = match parse_reply(&reply) {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "cannot parse the adaptive strategy reply; using the deterministic ladder"
                );
                return fallback_with(&ladder, error.to_string());
            }
        };
        if !self.config.allows(parsed.strategy) {
            return fallback_with(
                &ladder,
                format!(
                    "the model chose `{}`, which is outside the configured allowlist",
                    parsed.strategy
                ),
            );
        }
        let reason = parsed
            .reason
            .unwrap_or_else(|| format!("the model chose `{}` for this history", parsed.strategy));
        StrategyDecision {
            strategy: parsed.strategy,
            reason,
            source: StrategySource::Model,
            plan: (!parsed.plan.is_empty()).then_some(parsed.plan),
        }
    }

    /// The deterministic ladder: a total function of the measurements.
    ///
    /// See the module docs for the rule order. The result is cached nowhere and
    /// calls no model, so it is safe to run in a hot counting path.
    #[must_use]
    pub fn deterministic(&self, context: &AdaptiveContext) -> StrategyDecision {
        let ratio = self.config.min_savings_ratio;
        let percent = percent_of(ratio);
        let total = context.total_tokens;
        let fallback = |strategy: CompressionStrategy, reason: String| StrategyDecision {
            strategy,
            reason,
            source: StrategySource::DeterministicFallback,
            plan: None,
        };

        if context.is_empty() {
            return fallback(
                CompressionStrategy::Noop,
                "the history is empty: nothing to compress".to_owned(),
            );
        }
        if context.stale_tool_result_count > 0
            && dominates(context.stale_tool_result_tokens, total, ratio)
        {
            return fallback(
                CompressionStrategy::MicroCompact,
                format!(
                    "{} stale tool results hold {} of {total} tokens (at least {percent}%): \
                     micro-compact clears their bodies in place",
                    context.stale_tool_result_count, context.stale_tool_result_tokens
                ),
            );
        }
        if context.assistant_in_flight && dominates(context.in_flight_round_tokens, total, ratio) {
            return fallback(
                CompressionStrategy::CurrentRound,
                format!(
                    "the in-flight round holds {} of {total} tokens (at least {percent}%): \
                     compress the current round",
                    context.in_flight_round_tokens
                ),
            );
        }
        if context.completed_rounds >= ROUND_LEVEL_MIN_ROUNDS
            && dominates(context.completed_round_tokens, total, ratio)
        {
            return fallback(
                CompressionStrategy::RoundLevel,
                format!(
                    "{} completed rounds hold {} of {total} tokens (at least {percent}%): \
                     build round-level blocks",
                    context.completed_rounds, context.completed_round_tokens
                ),
            );
        }
        if context.completed_rounds > 0 && dominates(context.completed_round_tokens, total, ratio) {
            return fallback(
                CompressionStrategy::DialogueSummary,
                format!(
                    "{} completed rounds hold {} of {total} tokens (at least {percent}%): \
                     summarise the older dialogue",
                    context.completed_rounds, context.completed_round_tokens
                ),
            );
        }
        let budget = context.effective_budget();
        if budget > 0 && total > budget {
            return fallback(
                CompressionStrategy::FullCompact,
                format!(
                    "the history holds {total} tokens over its {budget}-token budget and no \
                     smaller span dominates: compact the whole history"
                ),
            );
        }
        if self.config.fallback_strategy.is_noop() {
            fallback(
                CompressionStrategy::Noop,
                format!(
                    "no measured span reaches {percent}% of {total} tokens: leaving the history \
                     alone"
                ),
            )
        } else {
            fallback(
                self.config.fallback_strategy,
                format!(
                    "no measured span reaches {percent}% of {total} tokens: using the configured \
                     fallback strategy `{}`",
                    self.config.fallback_strategy
                ),
            )
        }
    }
}

/// The compressing configs' serde-facing adaptive sub-config.
///
/// It carries only the serializable knobs — the model comes from the
/// compressor's own `Option<Arc<dyn Model>>` — so a compressor config round-trips
/// through JSON unchanged. [`AdaptiveConfig::enabled`] defaults to `false`, which
/// keeps every existing compressor byte-identical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdaptiveConfig {
    /// Whether this compressor runs adaptive selection before compressing.
    pub enabled: bool,
    /// Whether the deterministic extractor fills gaps in the plan.
    pub task_extraction: bool,
    /// Whether rendered blocks carry the structured plan sections.
    pub json_schema_blocks: bool,
    /// Upper bound on the decision reply's tokens.
    pub max_decision_tokens: usize,
    /// Strategies the model may choose; empty allows every strategy.
    pub strategy_allowlist: Vec<CompressionStrategy>,
    /// Strategy the ladder ends on when no measured span dominates.
    pub fallback_strategy: CompressionStrategy,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            task_extraction: true,
            json_schema_blocks: true,
            max_decision_tokens: 1_024,
            strategy_allowlist: Vec::new(),
            fallback_strategy: CompressionStrategy::Noop,
        }
    }
}

impl AdaptiveConfig {
    /// Checks the adaptive configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the flag is on with a zero decision budget
    /// or a savings ratio outside `(0, 1]`. A disabled config never fails, so
    /// existing configurations keep validating.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.strategy_config().validate()
    }

    /// The full strategy configuration this sub-config describes.
    #[must_use]
    pub fn strategy_config(&self) -> AdaptiveStrategyConfig {
        AdaptiveStrategyConfig {
            enabled: self.enabled,
            max_decision_tokens: self.max_decision_tokens,
            strategy_allowlist: self.strategy_allowlist.clone(),
            task_extraction: self.task_extraction,
            json_schema_blocks: self.json_schema_blocks,
            fallback_strategy: self.fallback_strategy,
            ..AdaptiveStrategyConfig::default()
        }
    }

    /// Builds the selection seam, wiring `model` as the decision model.
    #[must_use]
    pub fn strategy(&self, model: Option<Arc<dyn Model>>) -> AdaptiveStrategy {
        AdaptiveStrategy::new(self.strategy_config()).with_model(model)
    }
}

/// Renders `plan` inside `kind`'s memory-block envelope.
///
/// The envelope is unchanged (marker line, authority/instruction-status/
/// conflict-priority lines, `Summary:` label); the plan supplies the body: the
/// summary first, then the plan's structured sections. An empty plan renders the
/// same envelope a plain summary does.
#[must_use]
pub fn render_memory_block_from_plan(kind: MemoryBlockKind, plan: &MemoryBlockPlan) -> String {
    render_memory_block_from_plan_with_marker(kind, plan, None)
}

/// Renders `plan` inside `kind`'s envelope and appends a recall marker.
///
/// The marker is appended through [`RecallArchiveMarker::append`], the same
/// helper the non-adaptive path uses, so a block produced here points at its
/// recall archive exactly like a plain block.
#[must_use]
pub fn render_memory_block_from_plan_with_marker(
    kind: MemoryBlockKind,
    plan: &MemoryBlockPlan,
    marker: Option<&RecallArchiveMarker>,
) -> String {
    let block = kind.render(&plan.render_body());
    match marker {
        Some(marker) => marker.append(block),
        None => block,
    }
}

/// Renders the memory block a compressor writes, adaptive or not.
///
/// This is the one place that decides what a compressed block says:
///
/// - with [`AdaptiveConfig::enabled`] off the body is exactly
///   [`MemoryBlockKind::render`]'s plain `Summary:` text, so the non-adaptive
///   path stays byte-identical;
/// - otherwise the plan is the decision's model plan, gaps are filled from the
///   span when [`AdaptiveConfig::task_extraction`] is set, the summary is the
///   span summary the compressor computed, and the structured sections are
///   dropped when [`AdaptiveConfig::json_schema_blocks`] is off.
///
/// The recall marker, when the archive produced one, is appended through the
/// same [`RecallArchiveMarker::append`] helper the plain path uses.
pub(crate) fn render_block(
    kind: MemoryBlockKind,
    config: &AdaptiveConfig,
    decision: Option<&StrategyDecision>,
    span: &[BaseMessage],
    summary: &str,
    marker: Option<&RecallArchiveMarker>,
) -> String {
    let mut plan = if config.enabled {
        decision
            .and_then(|decision| decision.plan.clone())
            .unwrap_or_default()
    } else {
        MemoryBlockPlan::default()
    };
    if config.enabled && config.task_extraction {
        plan.fill_missing(&MemoryBlockPlan::from_messages(span, true));
    }
    summary.trim().clone_into(&mut plan.summary);
    if !config.enabled || !config.json_schema_blocks {
        plan = MemoryBlockPlan {
            summary: plan.summary,
            ..MemoryBlockPlan::default()
        };
    }
    render_memory_block_from_plan_with_marker(kind, &plan, marker)
}

/// The event a compressor reports when the model recommends another pass.
///
/// The event carries the recommendation in `summary` and changes no message, so
/// the history stays untouched. `ContextEngine::compress` treats a non-empty
/// event list as a report and may call the pass `compressed`; its token counts
/// are equal, which is how a caller recognises a recommendation.
pub(crate) fn recommendation_event(
    processor: &str,
    decision: &StrategyDecision,
    tokens: usize,
) -> ContextEvent {
    ContextEvent {
        processor: processor.to_owned(),
        kind: ContextEventKind::Summarized,
        messages_changed: Vec::new(),
        summary: Some(format!(
            "adaptive strategy recommendation: `{}` ({})",
            decision.strategy, decision.reason
        )),
        offload_handle: None,
        tokens_before: tokens,
        tokens_after: tokens,
    }
}

/// What a compressor does after the adaptive selection ran.
pub(crate) enum AdaptiveOutcome {
    /// Run the compressor's own pass; the decision carries the plan when a model
    /// chose the same strategy.
    Proceed(Option<StrategyDecision>),
    /// Leave the history and the counters alone.
    Skip,
    /// The model recommends another pass: report it and change nothing.
    Recommend(ContextEvent),
}

/// Runs the adaptive selection for `processor` and maps it to an outcome.
///
/// A compressor that did not enable the seam (`strategy` is `None`) proceeds
/// unchanged. Otherwise the selection runs after the compressor's own trigger
/// gate, and the cancellation check happens after the model call so a cancelled
/// context does not apply the decision.
pub(crate) async fn select_for_processor(
    strategy: Option<AdaptiveStrategy>,
    processor: &'static str,
    own: CompressionStrategy,
    messages: &[BaseMessage],
    cx: &ProcessorContext<'_>,
    recent_compressions: u64,
) -> Result<AdaptiveOutcome> {
    let Some(strategy) = strategy else {
        return Ok(AdaptiveOutcome::Proceed(None));
    };
    let input = StrategyInput::new(messages, cx.counter)
        .with_budget(cx.budget)
        .with_trigger(cx.trigger_tokens)
        .with_history(
            usize::try_from(recent_compressions).unwrap_or(usize::MAX),
            None,
        );
    let decision = strategy.select(&input.measure()).await;
    if cx.cancellation.is_cancelled() {
        return Err(Error::Cancelled);
    }
    if decision.strategy.is_noop() {
        tracing::debug!(
            reason = decision.reason.as_str(),
            "adaptive selection chose noop; the pass is skipped"
        );
        return Ok(AdaptiveOutcome::Skip);
    }
    if decision.strategy == own {
        return Ok(AdaptiveOutcome::Proceed(Some(decision)));
    }
    let tokens = count_messages(cx.counter, messages);
    tracing::debug!(
        recommended = decision.strategy.as_str(),
        reason = decision.reason.as_str(),
        "adaptive selection deferred to the recommended pass"
    );
    Ok(AdaptiveOutcome::Recommend(recommendation_event(
        processor, &decision, tokens,
    )))
}

/// The parsed form of one strategy reply.
struct StrategyReply {
    /// The strategy the model named.
    strategy: CompressionStrategy,
    /// The model's justification, when it gave one.
    reason: Option<String>,
    /// The plan the reply carried; empty when it carried none.
    plan: MemoryBlockPlan,
}

/// Parses a strategy reply leniently.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the reply carries no JSON object, when the
/// JSON is invalid, when it names no string `strategy`, or when the strategy
/// name is unknown. The caller turns every error into a ladder decision.
fn parse_reply(text: &str) -> Result<StrategyReply> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::Validation("the strategy reply is empty".to_owned()));
    }
    let object = prompt::extract_json_object(text)
        .ok_or_else(|| Error::Validation("the strategy reply carries no JSON object".to_owned()))?;
    let value: serde_json::Value = serde_json::from_str(object).map_err(|error| {
        Error::Validation(format!("the strategy reply is not valid JSON: {error}"))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::Validation("the strategy reply is not a JSON object".to_owned()))?;
    let name = object
        .get("strategy")
        .ok_or_else(|| Error::Validation("the strategy reply names no strategy".to_owned()))?
        .as_str()
        .ok_or_else(|| {
            Error::Validation("the strategy reply's strategy is not a string".to_owned())
        })?;
    let parsed = CompressionStrategy::parse(name);
    if let Some(reason) = parsed.reason {
        return Err(Error::Validation(reason));
    }
    let reason = object
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
        .map(str::to_owned);
    Ok(StrategyReply {
        strategy: parsed.strategy,
        reason,
        plan: MemoryBlockPlan::from_object(object),
    })
}

/// The ladder's decision with `failure` prepended to its reason.
fn fallback_with(ladder: &StrategyDecision, failure: String) -> StrategyDecision {
    StrategyDecision {
        strategy: ladder.strategy,
        reason: format!(
            "{failure}; the deterministic ladder chose `{}` ({})",
            ladder.strategy, ladder.reason
        ),
        source: StrategySource::DeterministicFallback,
        plan: None,
    }
}

/// Whether `tokens` reaches `ratio` of `total`.
fn dominates(tokens: usize, total: usize, ratio: f64) -> bool {
    tokens > 0 && total > 0 && tokens as f64 >= ratio * total as f64
}

/// `ratio` as a whole percentage, for reason strings.
fn percent_of(ratio: f64) -> u32 {
    (ratio * 100.0).round() as u32
}

/// Normalises a strategy name: lowercase, separators collapsed, quotes removed.
fn normalise_name(text: &str) -> String {
    let trimmed = text.trim().trim_matches(|ch: char| {
        ch == '`' || ch == '"' || ch == '\'' || ch == '\u{201c}' || ch == '\u{201d}'
    });
    let mut out = String::with_capacity(trimmed.len());
    let mut last_separator = false;
    for ch in trimmed.chars() {
        let ch = ch.to_ascii_lowercase();
        if ch == '_' || ch == '-' || ch.is_whitespace() {
            if !out.is_empty() && !last_separator {
                out.push('_');
                last_separator = true;
            }
        } else {
            out.push(ch);
            last_separator = false;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_names_parse_tolerantly() {
        assert_eq!(
            CompressionStrategy::parse(" Dialogue-Summary ").strategy,
            CompressionStrategy::DialogueSummary
        );
        assert_eq!(
            CompressionStrategy::parse("`current_round`").strategy,
            CompressionStrategy::CurrentRound
        );
        assert_eq!(
            CompressionStrategy::parse("NOOP").strategy,
            CompressionStrategy::Noop
        );
        let unknown = CompressionStrategy::parse("summarise everything");
        assert_eq!(unknown.strategy, CompressionStrategy::Noop);
        assert!(unknown
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("unknown compression strategy")));
        assert!(CompressionStrategy::parse("").reason.is_some());
        assert_eq!(
            serde_json::to_string(&CompressionStrategy::RoundLevel).unwrap(),
            "\"round_level\""
        );
    }

    #[test]
    fn the_ladder_is_total_and_measured() {
        let strategy = AdaptiveStrategy::new(AdaptiveStrategyConfig::default());
        let empty = strategy.deterministic(&AdaptiveContext::default());
        assert_eq!(empty.strategy, CompressionStrategy::Noop);
        assert_ne!(empty.reason, "");

        let dominant = AdaptiveContext {
            total_tokens: 1_000,
            messages: 4,
            completed_rounds: 2,
            completed_round_tokens: 900,
            ..AdaptiveContext::default()
        };
        assert_eq!(
            strategy.deterministic(&dominant).strategy,
            CompressionStrategy::DialogueSummary
        );

        let over_budget = AdaptiveContext {
            total_tokens: 1_000,
            messages: 2,
            budget_tokens: 900,
            ..AdaptiveContext::default()
        };
        assert_eq!(
            strategy.deterministic(&over_budget).strategy,
            CompressionStrategy::FullCompact
        );

        let nothing = AdaptiveContext {
            total_tokens: 1_000,
            messages: 3,
            completed_rounds: 1,
            completed_round_tokens: 10,
            budget_tokens: 5_000,
            ..AdaptiveContext::default()
        };
        assert_eq!(
            strategy.deterministic(&nothing).strategy,
            CompressionStrategy::Noop
        );
    }
}
