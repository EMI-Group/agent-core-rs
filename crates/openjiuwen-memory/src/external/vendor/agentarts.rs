//! The `AgentArts` memory-service adapter (`agentarts`) — SDK-only.
//!
//! Unlike the other four vendors this adapter never speaks raw HTTP. The Python adapter
//! imports `agentarts.sdk.memory.MemoryClient` and drives the vendor SDK, which owns its
//! own connection handling; no documented REST contract exists to fall back on, so there
//! is no [`WireSpec`](super::super::rest::WireSpec) for this vendor and an `endpoint` in
//! the configuration cannot make the handle operational.
//!
//! Every operation therefore returns the same typed
//! [`openjiuwen_common::error::Error::Unsupported`] naming the missing SDK, whether or not
//! an `endpoint` is configured, mirroring the Python package's lazy client acquisition (a
//! missing extra surfaces at first use, not at import).
//!
//! # Rail
//!
//! The rail surface is complete even though no data operation is: the handle advertises
//! Python's single `external_memory_search` tool, its prompt block, and its prefetch, and
//! every one of them ends in the same typed `Unsupported` once it reaches the backend. A
//! failed prefetch renders `""` and a failed or malformed tool call answers
//! `{"error": …}`, so a missing SDK degrades the model's answer instead of failing the
//! turn — exactly the Python adapter's "degrade, never raise" behaviour.
use std::fmt;

use async_trait::async_trait;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};

use super::super::provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
use super::super::rail;
use super::super::registry::{ProviderConfig, ProviderKind};
use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

/// Registry id of the adapter (the Python provider id).
const ID: &str = "agentarts";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "AgentArts";
/// The dependency every operation names.
const MISSING: &str = "the `agentarts-sdk` package (`MemoryClient`) and its HTTP transport";
/// `tracing` target of this module's rail events.
const LOG_TARGET: &str = "openjiuwen_memory::external::vendor::agentarts";
/// The single rail tool the adapter advertises.
const SEARCH_TOOL: &str = "external_memory_search";
/// The recall size of a tool call that names none (Python's `_DEFAULT_TOP_K`).
const DEFAULT_TOP_K: i64 = 10;
/// The largest recall size a tool call may request (Python's `_MAX_TOP_K`).
const MAX_TOP_K: i64 = 100;
/// The recall size of a prefetch, which carries no arguments (Python's `_DEFAULT_TOP_K`).
const PREFETCH_TOP_K: usize = 10;
/// The minimum score a tool call or a prefetch applies when it names none
/// (Python's `_DEFAULT_MIN_SCORE`).
const DEFAULT_MIN_SCORE: f64 = 0.5;
/// The strategy filters the vendor documents (Python's `_STRATEGY_TYPES`).
const STRATEGY_TYPES: [&str; 6] = [
    "semantic",
    "summary",
    "user_preference",
    "episodic",
    "event",
    "custom",
];

/// The SDK-only backend of the `AgentArts` adapter.
///
/// It keeps the configuration (so `config()` and the redacting `Debug` can report it) and
/// nothing else: there is no transport to construct and no wire format to describe.
struct SdkOnlyProvider {
    /// The configuration the handle was built from.
    config: ProviderConfig,
}

impl SdkOnlyProvider {
    /// Builds the helper for `config`; never fails.
    fn new(config: &ProviderConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// The kind this backend implements.
    fn kind() -> ProviderKind {
        ProviderKind::AgentArts
    }

    /// The single typed error every operation returns.
    fn error() -> Error {
        Error::unsupported(format!(
            "the {DISPLAY} memory provider requires {MISSING}, which this build does not provide"
        ))
    }
}

#[async_trait]
impl MemoryProvider for SdkOnlyProvider {
    fn name(&self) -> &str {
        ID
    }

    /// A handle with no transport declares no capability at all.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    async fn initialize(&self) -> Result<()> {
        Err(Self::error())
    }

    async fn add(
        &self,
        _scope: &MemoryScope,
        _memories: Vec<NewMemory>,
        _options: &AddOptions,
    ) -> Result<AddOutcome> {
        Err(Self::error())
    }

    async fn search(
        &self,
        _scope: &MemoryScope,
        _query: &str,
        _options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        Err(Self::error())
    }

    async fn get(&self, _scope: &MemoryScope, _id: &str) -> Result<Option<MemoryRecord>> {
        Err(Self::error())
    }

    async fn update(
        &self,
        _scope: &MemoryScope,
        _id: &str,
        _update: MemoryUpdate,
        _options: &CallOptions,
    ) -> Result<MemoryRecord> {
        Err(Self::error())
    }

    async fn delete(&self, _scope: &MemoryScope, _id: &str) -> Result<bool> {
        Err(Self::error())
    }

    async fn list(&self, _filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        Err(Self::error())
    }

    async fn clear(&self, _filter: &MemoryFilter) -> Result<usize> {
        Err(Self::error())
    }

    async fn health(&self) -> Result<ProviderHealth> {
        Err(Self::error())
    }
}

/// Handle for the `AgentArts` memory service (`agentarts`).
///
/// Construction never fails: the vendor SDK is required at
/// [`MemoryProvider::initialize`], and configuring an `endpoint` changes nothing because
/// this adapter has no HTTP path.
pub struct AgentArtsMemoryProvider {
    /// The SDK-only backend.
    inner: SdkOnlyProvider,
}

impl AgentArtsMemoryProvider {
    /// Builds the adapter handle for `config`.
    #[must_use]
    pub fn new(config: &ProviderConfig) -> Self {
        Self {
            inner: SdkOnlyProvider::new(config),
        }
    }

    /// The configuration this handle was built from.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        &self.inner.config
    }

    /// The kind this adapter implements.
    #[must_use]
    pub fn kind(&self) -> ProviderKind {
        SdkOnlyProvider::kind()
    }
}

impl fmt::Debug for AgentArtsMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentArtsMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config.endpoint)
            .finish_non_exhaustive()
    }
}

/// Logs a failed rail call; the model sees the payload, the cause stays on `debug`.
fn log_rail_failure(tool: &str, error: &Error) {
    tracing::debug!(
        target: LOG_TARGET,
        provider = ID,
        tool,
        %error,
        "the AgentArts memory rail call failed; the model sees the degraded payload"
    );
}

/// The single `external_memory_search` spec the rail advertises, mirroring the Python
/// adapter's `EXTERNAL_MEMORY_SEARCH_SCHEMA`.
///
/// `strategy_type` is advertised for parity with the vendor schema; the record SPI has no
/// strategy filter, so a call's value is accepted but not forwarded.
fn tool_schemas() -> Vec<Value> {
    vec![rail::tool_schema(
        SEARCH_TOOL,
        "Retrieve durable facts, user preferences, and prior conversation context from long-term external memory.",
        json!({
            "query": rail::string_property("Memory search query."),
            "top_k": rail::integer_property("Max results, default 10, max 100."),
            "strategy_type": rail::enum_property(
                "Optional memory strategy type filter.",
                &STRATEGY_TYPES,
            ),
            "min_score": rail::number_property("Optional minimum similarity score (default 0.5)."),
        }),
        &["query"],
    )]
}

/// The recall options of one `external_memory_search` call.
///
/// `top_k` defaults to [`DEFAULT_TOP_K`] and is clamped to `1..=MAX_TOP_K`; `min_score`
/// defaults to [`DEFAULT_MIN_SCORE`] and is clamped to `0.0..=1.0` (a non-finite value
/// falls back to the default rather than poisoning the options).
fn search_options(map: &serde_json::Map<String, Value>) -> SearchOptions {
    let top_k = rail::integer_arg(map, "top_k")
        .unwrap_or(DEFAULT_TOP_K)
        .clamp(1, MAX_TOP_K);
    let top_k = usize::try_from(top_k).unwrap_or(PREFETCH_TOP_K);
    let min_score = rail::number_arg(map, "min_score")
        .filter(|score| score.is_finite())
        .map_or(DEFAULT_MIN_SCORE, |score| score.clamp(0.0, 1.0));
    SearchOptions::default()
        .with_top_k(top_k)
        .with_threshold(min_score as f32)
}

implement_vendor_provider!(AgentArtsMemoryProvider, inner;

    /// The single `external_memory_search` spec, replacing the vendor-neutral four tools.
    fn get_tool_schemas(&self) -> Vec<Value> {
        tool_schemas()
    }

    /// The scope the rail calls use: the configured `user_id`/`scope_id` fallback.
    fn default_scope(&self) -> MemoryScope {
        super::scope_from_options(self.config())
    }

    /// Python's `AgentArts` prompt block, naming the rail's single tool.
    fn system_prompt_block(&self) -> String {
        "# External Memory\nUse `external_memory_search` to retrieve durable facts, user \
         preferences, and prior conversation context from long-term external memory."
            .to_owned()
    }

    /// Renders one recall as a `## External Memory` bullet list.
    ///
    /// A blank query renders nothing without touching the backend. This override
    /// deliberately degrades to an empty string: this backend is SDK-only and answers
    /// every recall with the typed `Unsupported`, and Python's adapter likewise returns
    /// `""` when its search fails, so a missing SDK never breaks the turn (the cause is
    /// logged at `debug`).
    async fn prefetch(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }
        let options = SearchOptions::default()
            .with_top_k(PREFETCH_TOP_K)
            .with_threshold(DEFAULT_MIN_SCORE as f32);
        match self.search(&self.default_scope(), query, &options).await {
            Ok(hits) => rail::render_bullets(
                "External Memory",
                hits.iter().map(|hit| hit.record.content.clone()),
            ),
            Err(error) => {
                log_rail_failure("prefetch", &error);
                String::new()
            }
        }
    }

    /// Dispatches the single `external_memory_search` tool over the record SPI.
    ///
    /// The payloads are Python's: `{"results": [{"memory", "score"}], "count"}` on a hit
    /// list, `{"result": "No relevant memories found."}` when nothing matched, and
    /// `{"error": …}` for a missing parameter, an unknown tool name, or a backend failure.
    /// A failure is never propagated — the caller is a model, and Python's rail answers
    /// with an error object for the same reason.
    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        if tool != SEARCH_TOOL {
            return rail::tool_error(format!("unknown tool: {tool}"));
        }
        let map = rail::args_object(args);
        let Some(query) = rail::string_arg(map, "query")
            .map(str::trim)
            .filter(|query| !query.is_empty())
        else {
            return rail::tool_error("Missing required parameter: query");
        };
        let options = search_options(map);
        match self.search(&self.default_scope(), query, &options).await {
            Ok(hits) if hits.is_empty() => json!({ "result": "No relevant memories found." }),
            Ok(hits) => {
                let results: Vec<Value> = hits
                    .iter()
                    .map(|hit| json!({ "memory": hit.record.content, "score": hit.score }))
                    .collect();
                let count = results.len();
                json!({ "results": results, "count": count })
            }
            Err(error) => {
                log_rail_failure(SEARCH_TOOL, &error);
                rail::tool_error(error.to_string())
            }
        }
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    /// The credential every redaction assertion looks for.
    const SECRET: &str = "sk-agentarts-secret";

    /// A configured handle; nothing ever connects anywhere.
    fn handle() -> AgentArtsMemoryProvider {
        AgentArtsMemoryProvider::new(
            &ProviderConfig::new(ProviderKind::AgentArts).with_api_key(SECRET),
        )
    }

    #[test]
    fn the_rail_advertises_exactly_the_external_memory_search_tool() {
        let schemas = handle().get_tool_schemas();
        assert_eq!(schemas.len(), 1, "the generic four tools must be replaced");
        let schema = &schemas[0];
        assert_eq!(schema["name"], json!(SEARCH_TOOL));
        assert_eq!(schema["parameters"]["type"], json!("object"));
        assert_eq!(schema["parameters"]["required"], json!(["query"]));

        let properties = schema["parameters"]["properties"]
            .as_object()
            .expect("properties is an object");
        for key in ["query", "top_k", "strategy_type", "min_score"] {
            assert!(properties.contains_key(key), "missing parameter `{key}`");
        }
        assert_eq!(properties["query"]["type"], json!("string"));
        assert_eq!(properties["top_k"]["type"], json!("integer"));
        assert_eq!(properties["min_score"]["type"], json!("number"));
        assert_eq!(
            properties["strategy_type"]["enum"],
            json!([
                "semantic",
                "summary",
                "user_preference",
                "episodic",
                "event",
                "custom"
            ])
        );
    }

    #[test]
    fn the_prompt_block_names_the_external_memory_tool() {
        let block = handle().system_prompt_block();
        assert!(block.contains("# External Memory"), "{block}");
        assert!(block.contains(SEARCH_TOOL), "{block}");
    }

    #[tokio::test]
    async fn prefetch_degrades_to_an_empty_rendering() {
        let adapter = handle();
        assert_eq!(adapter.prefetch("").await, "");
        assert_eq!(adapter.prefetch("   ").await, "");
        assert_eq!(adapter.prefetch("dark mode").await, "");
    }

    #[tokio::test]
    async fn the_search_tool_answers_an_error_instead_of_panicking() {
        let payload = handle()
            .handle_tool_call(SEARCH_TOOL, &json!({ "query": "q" }))
            .await;
        let object = payload.as_object().expect("an object payload");
        assert!(object.contains_key("error"), "{payload}");
        assert!(
            object["error"]
                .as_str()
                .expect("an error message")
                .contains("agentarts-sdk"),
            "{payload}"
        );
    }

    #[tokio::test]
    async fn the_search_tool_requires_a_query() {
        let adapter = handle();
        assert_eq!(
            adapter.handle_tool_call(SEARCH_TOOL, &json!({})).await,
            json!({ "error": "Missing required parameter: query" })
        );
        assert_eq!(
            adapter
                .handle_tool_call(SEARCH_TOOL, &json!({ "query": "   " }))
                .await,
            json!({ "error": "Missing required parameter: query" })
        );
        assert_eq!(
            adapter
                .handle_tool_call(SEARCH_TOOL, &json!({ "query": 7 }))
                .await,
            json!({ "error": "Missing required parameter: query" })
        );
    }

    #[tokio::test]
    async fn unknown_tools_are_rejected_by_name() {
        assert_eq!(
            handle().handle_tool_call("nope", &json!({})).await,
            json!({ "error": "unknown tool: nope" })
        );
        // The generic default tools are gone, so their names are unknown here too.
        assert_eq!(
            handle().handle_tool_call("search_memory", &json!({})).await,
            json!({ "error": "unknown tool: search_memory" })
        );
    }

    #[test]
    fn the_handle_is_sdk_only_and_redacts_its_credential() {
        let adapter = handle();
        assert_eq!(adapter.name(), "agentarts");
        assert_eq!(adapter.kind(), ProviderKind::AgentArts);
        assert_eq!(adapter.capabilities(), ProviderCapabilities::none());

        let printed = format!("{adapter:?}");
        assert!(printed.contains("AgentArtsMemoryProvider"), "{printed}");
        assert!(printed.contains("agentarts-sdk"), "{printed}");
        assert!(!printed.contains(SECRET), "{printed}");
    }

    #[test]
    fn search_options_apply_pythons_defaults_and_clamps() {
        let default = search_options(rail::args_object(&json!({})));
        assert_eq!(default.top_k, 10);
        assert!((default.threshold - 0.5).abs() < f32::EPSILON);

        let bounded = search_options(rail::args_object(&json!({
            "top_k": 5_000,
            "min_score": 9.0,
        })));
        assert_eq!(bounded.top_k, 100);
        assert!((bounded.threshold - 1.0).abs() < f32::EPSILON);

        let floored = search_options(rail::args_object(&json!({
            "top_k": -3,
            "min_score": -1.0,
        })));
        assert_eq!(floored.top_k, 1);
        assert!(floored.threshold.abs() < f32::EPSILON);
    }
}
