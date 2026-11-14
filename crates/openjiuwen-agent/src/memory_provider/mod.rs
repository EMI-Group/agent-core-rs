//! The external long-term-memory provider bridge: a rail plus the provider's
//! model-facing tools.
//!
//! Ports the tool-rail half of the memory SPI (`openjiuwen.core.memory`'s
//! provider callback) onto [`openjiuwen_memory::MemoryProvider`]. The SPI is
//! intentionally small: every rail entry point is a *provided* trait method with
//! a working default, so a backend only overrides what it wants to change. This
//! module supplies the two things a provider cannot supply for itself — a place
//! in the [`ReActAgent`](crate::ReActAgent) loop and an [`Tool`] wrapper for its
//! tool schemas.
//!
//! # [`ProviderMemoryRail`]
//!
//! Registered by [`ReActAgent::set_memory_provider`](crate::ReActAgent::set_memory_provider)
//! under the fixed rail name `"memory_provider"`, it is
//! the provider counterpart of [`MemoryRail`](crate::rails::MemoryRail):
//!
//! - `BeforeInvoke` publishes [`MemoryProvider::system_prompt_block`] — when it
//!   is not blank — under the `ctx.extra` key `memory_provider_block`
//!   (`"memory_provider_block"`). The loop appends that block to the identity
//!   section of the system prompt, so provider guidance reaches every model call
//!   of the invocation, not only the first one.
//! - `BeforeModelCall` calls [`MemoryProvider::prefetch`] once per invocation
//!   (guarded by the `openjiuwen.rails.memory_provider.prefetched` flag in
//!   `ctx.extra`), using the text of
//!   the last user message as the query, and inserts the returned text as a
//!   system message **directly after the leading system messages** — the same
//!   placement [`MemoryRail`](crate::rails::MemoryRail) uses, so the agent's own
//!   identity section stays first.
//! - `AfterInvoke` records the turn with [`MemoryProvider::sync_turn`] when the
//!   output is an answer and the invocation had a text query.
//!
//! Both directions are best-effort: a provider failure is logged with
//! `tracing::warn!` and never aborts the run. `prefetch`/`handle_tool_call` are
//! the SPI's own fail-soft entry points (an empty string and a `{"error": …}`
//! object respectively), so the rail only has to log the [`sync_turn`] error.
//!
//! # [`ProviderTool`] and [`provider_tools`]
//!
//! One [`ProviderTool`] wraps one OpenAI-style function spec from
//! [`MemoryProvider::get_tool_schemas`], deriving its [`ToolCard`] from
//! `function.name`/`function.description`/`function.parameters` exactly like the
//! dynamic MCP abilities do. `invoke` dispatches to
//! [`MemoryProvider::handle_tool_call`] and returns its JSON payload unchanged: a
//! `{"error": …}` object is a *successful* tool result whose text the model
//! reads, never an `Err` — the SPI's rail contract, which keeps a degraded memory
//! backend from aborting the conversation. A schema without a usable name is
//! skipped and logged; nothing here panics.
//!
//! # Owned-session finalization
//!
//! The loop calls [`MemoryProvider::on_session_end`] once per run whose session
//! the run created itself ([`AgentRun::is_owned`](crate::AgentRun::is_owned)),
//! with the turns it gathered from the run's message history (the user query and
//! every assistant reply, in order). A caller-supplied session is the caller's to
//! finalize through
//! [`ReActAgent::end_memory_session`](crate::ReActAgent::end_memory_session).
//! Interrupt pauses are excluded because the conversation is not over.
//!
//! [`sync_turn`]: MemoryProvider::sync_turn
use crate::input::AgentInput;
use crate::output::{AgentOutput, ResultType};
use crate::rails::{AgentRail, RailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, Role, Tool, ToolCard, ToolContext};
use openjiuwen_memory::{ConversationTurn, MemoryProvider};
use serde_json::{Map, Value};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
mod tests;

/// Tracing target of the provider bridge.
const TARGET: &str = "openjiuwen_agent::memory_provider";

/// Rail name [`ReActAgent::set_memory_provider`] registers the rail under.
///
/// [`ReActAgent::set_memory_provider`]: crate::ReActAgent::set_memory_provider
pub(crate) const PROVIDER_RAIL_NAME: &str = "memory_provider";

/// `ctx.extra` key carrying the provider's system-prompt block.
pub(crate) const MEMORY_PROVIDER_BLOCK_KEY: &str = "memory_provider_block";

/// `ctx.extra` key marking that this invocation already prefetched.
const PREFETCHED_KEY: &str = "openjiuwen.rails.memory_provider.prefetched";

/// The provider's rail: prompt block, prefetch, and turn write-back.
///
/// See the [module docs](self) for the firing points and the placement rules.
pub struct ProviderMemoryRail {
    provider: Arc<dyn MemoryProvider>,
    name: String,
}

impl ProviderMemoryRail {
    /// Builds the rail over `provider` under the default name
    /// `"memory_provider"`.
    #[must_use]
    pub fn new(provider: Arc<dyn MemoryProvider>) -> Self {
        Self {
            provider,
            name: PROVIDER_RAIL_NAME.to_owned(),
        }
    }

    /// Returns the rail with its registry name replaced by `name`.
    ///
    /// [`ReActAgent::set_memory_provider`](crate::ReActAgent::set_memory_provider)
    /// always registers the default name; a hand-built rail may use another one.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns the provider the rail reads and writes.
    #[must_use]
    pub fn provider(&self) -> &Arc<dyn MemoryProvider> {
        &self.provider
    }
}

impl fmt::Debug for ProviderMemoryRail {
    /// Prints the rail name and the provider id; the provider itself is opaque.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderMemoryRail")
            .field("name", &self.name)
            .field("provider", &self.provider.name())
            .finish()
    }
}

/// Returns the text of the last user message, if any.
fn last_user_text(messages: &[BaseMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.text().into_owned())
}

/// The provider block a `BeforeInvoke` firing published, if any.
///
/// The loop hands it to the prompt renderer, which appends it to the identity
/// section after the template's own text.
pub(crate) fn injected_prompt_block(ctx: &RailContext) -> Option<&str> {
    ctx.extra
        .get(MEMORY_PROVIDER_BLOCK_KEY)
        .and_then(Value::as_str)
        .filter(|block| !block.trim().is_empty())
}

#[async_trait]
impl AgentRail for ProviderMemoryRail {
    fn name(&self) -> &str {
        &self.name
    }

    /// Publishes the provider's system-prompt block for the prompt renderer;
    /// an empty block publishes nothing.
    async fn before_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        let block = self.provider.system_prompt_block();
        if block.trim().is_empty() {
            return Ok(());
        }
        ctx.extra
            .insert(MEMORY_PROVIDER_BLOCK_KEY.to_owned(), Value::String(block));
        Ok(())
    }

    /// Prefetches once per invocation and inserts the result after the leading
    /// system messages; an empty prefetch or a missing user message changes
    /// nothing.
    async fn before_model_call(&self, ctx: &mut RailContext) -> Result<()> {
        if ctx.extra.contains_key(PREFETCHED_KEY) {
            return Ok(());
        }
        let Some(query) = last_user_text(&ctx.messages) else {
            return Ok(());
        };
        if query.trim().is_empty() {
            return Ok(());
        }
        // Mark before the call so a failing prefetch is not retried on every
        // iteration, and an injected block can never be prepended twice.
        ctx.extra
            .insert(PREFETCHED_KEY.to_owned(), Value::Bool(true));
        let recalled = self.provider.prefetch(&query).await;
        if recalled.trim().is_empty() {
            return Ok(());
        }
        let index = ctx
            .messages
            .iter()
            .take_while(|message| message.role == Role::System)
            .count();
        ctx.messages.insert(index, BaseMessage::system(recalled));
        Ok(())
    }

    /// Records the turn with [`MemoryProvider::sync_turn`] when the output is an
    /// answer and the invocation carried a text query; failures are logged and
    /// the run's output is left untouched.
    async fn after_invoke(&self, ctx: &mut RailContext) -> Result<()> {
        let Some(output) = ctx.output.as_ref() else {
            return Ok(());
        };
        if !matches!(output.result_type, ResultType::Answer) {
            return Ok(());
        }
        let Some(query) = query_text(ctx.input.as_ref()) else {
            return Ok(());
        };
        if query.trim().is_empty() || output.output.trim().is_empty() {
            return Ok(());
        }
        if let Err(error) = self.provider.sync_turn(&query, &output.output).await {
            tracing::warn!(
                target: TARGET,
                provider = %self.provider.name(),
                %error,
                "memory provider turn write-back failed"
            );
        }
        Ok(())
    }
}

/// The invocation's text query, or `None` for an interactive resume.
fn query_text(input: Option<&AgentInput>) -> Option<String> {
    input.and_then(AgentInput::query_text).map(str::to_owned)
}

/// One provider tool: a [`MemoryProvider::handle_tool_call`] dispatcher bound to
/// one tool schema.
pub struct ProviderTool {
    provider: Arc<dyn MemoryProvider>,
    card: ToolCard,
}

impl ProviderTool {
    /// Wraps the already-derived `card` for `provider`.
    ///
    /// Use [`ProviderTool::from_schema`] to derive the card from a tool spec.
    #[must_use]
    pub fn new(provider: Arc<dyn MemoryProvider>, card: ToolCard) -> Self {
        Self { provider, card }
    }

    /// Derives a tool from one OpenAI-style function spec, or `None` when the
    /// spec carries no usable `function.name`.
    #[must_use]
    pub fn from_schema(provider: Arc<dyn MemoryProvider>, schema: &Value) -> Option<Self> {
        let card = card_from_schema(schema, provider.name())?;
        Some(Self { provider, card })
    }

    /// Returns the provider this tool dispatches to.
    #[must_use]
    pub fn provider(&self) -> &Arc<dyn MemoryProvider> {
        &self.provider
    }
}

impl fmt::Debug for ProviderTool {
    /// Prints the card; the provider itself is opaque.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderTool")
            .field("card", &self.card)
            .field("provider", &self.provider.name())
            .finish()
    }
}

#[async_trait]
impl Tool for ProviderTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    /// Dispatches the call to [`MemoryProvider::handle_tool_call`].
    ///
    /// The result is the provider's JSON payload verbatim: an `{"error": …}`
    /// object is a normal tool result, not a failure.
    ///
    /// # Errors
    ///
    /// Only returns [`openjiuwen_common::error::Error::Cancelled`], and only when
    /// the run was cancelled before the provider was called — the SPI's tool
    /// entry point reports its own failures in the payload.
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self
            .provider
            .handle_tool_call(&self.card.name, &inputs)
            .await)
    }
}

/// Derives a [`ToolCard`] from one OpenAI-style function spec.
///
/// Returns `None` when the spec has no object `function` with a non-blank
/// `name`; a missing `description` becomes an empty string and a missing
/// `parameters` an empty object schema.
fn card_from_schema(schema: &Value, provider: &str) -> Option<ToolCard> {
    let function = schema.get("function")?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())?;
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    // Provider tools touch one remote backend each; running them concurrently
    // with sibling calls would not make the backend's writes deterministic.
    Some(
        ToolCard::new(name, description, parameters)
            .with_metadata_entry("provider", Value::String(provider.to_owned())),
    )
}

/// Builds one [`ProviderTool`] per tool schema the provider advertises.
///
/// Schemas that cannot be turned into a card are skipped with a `tracing::warn!`
/// — never a panic and never an empty placeholder card. Mounting does not consult
/// [`MemoryProvider::is_available`], so an unavailable provider still exposes its
/// tools and reports its own typed behaviour through them.
#[must_use]
#[allow(clippy::needless_pass_by_value)] // the caller hands over its owned handle
pub fn provider_tools(provider: Arc<dyn MemoryProvider>) -> Vec<Arc<dyn Tool>> {
    provider
        .get_tool_schemas()
        .iter()
        .filter_map(|schema| {
            let tool = ProviderTool::from_schema(Arc::clone(&provider), schema);
            if tool.is_none() {
                tracing::warn!(
                    target: TARGET,
                    provider = %provider.name(),
                    schema = %schema,
                    "skipping a malformed memory provider tool schema"
                );
            }
            tool
        })
        .map(|tool| Arc::new(tool) as Arc<dyn Tool>)
        .collect()
}

/// The cards [`provider_tools`] derives, in the provider's schema order.
#[must_use]
pub fn provider_tool_cards(provider: &Arc<dyn MemoryProvider>) -> Vec<ToolCard> {
    provider
        .get_tool_schemas()
        .iter()
        .filter_map(|schema| card_from_schema(schema, provider.name()))
        .collect()
}

/// The turns a finished run contributes to [`MemoryProvider::on_session_end`].
///
/// Only text turns travel: the user query becomes one `user` turn and every
/// assistant reply one `assistant` turn, in the order they happened. Tool and
/// system messages are provider-internal (the rail has already written what it
/// keeps), so they are not forwarded.
pub(crate) fn finalize_turns(turns: &[ConversationTurn]) -> Vec<ConversationTurn> {
    turns
        .iter()
        .filter(|turn| !turn.is_blank())
        .cloned()
        .collect()
}

/// Calls [`MemoryProvider::on_session_end`] once for a finished owned-session run.
///
/// `output` is the run's final output: an interrupt pause is not the end of the
/// conversation, so it is skipped. Every failure is logged and never changes the
/// run's result.
pub(crate) async fn finalize_session(
    provider: Option<&Arc<dyn MemoryProvider>>,
    turns: &[ConversationTurn],
    owned: bool,
    output: &AgentOutput,
) {
    if !owned || output.is_interrupt() {
        return;
    }
    let Some(provider) = provider else {
        return;
    };
    let turns = finalize_turns(turns);
    if let Err(error) = provider.on_session_end(&turns).await {
        tracing::warn!(
            target: TARGET,
            provider = %provider.name(),
            %error,
            "memory provider session finalization failed"
        );
    }
}
