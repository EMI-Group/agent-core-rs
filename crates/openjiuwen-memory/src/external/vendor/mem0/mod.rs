//! The `mem0` hosted memory-service adapter (`mem0`).
//!
//! The Python adapter drives `mem0ai`'s `AsyncMemoryClient`; this Rust adapter speaks the
//! same record surface over the mem0 platform's HTTP API through the shared
//! [`RestProvider`](super::super::rest::RestProvider). The wire shape lives in
//! [`spec`], so the transport's validation, authentication, retries, failover, status
//! mapping, and cancellation apply unchanged.
//!
//! # Wire contract
//!
//! | Operation | Request |
//! | --- | --- |
//! | add | `POST /v1/memories/` with `messages`, `user_id`, optional `agent_id`, `infer: false` |
//! | search | `POST /v1/memories/search/` with `query`, `user_id`, optional `agent_id`, `limit`, optional `threshold` |
//! | list | `GET /v1/memories/` with `user_id`, optional `agent_id`, `page`, `page_size` |
//! | get | `GET /v1/memories/{id}/` |
//! | update | `PUT /v1/memories/{id}/` with `text` and/or `metadata` |
//! | delete | `DELETE /v1/memories/{id}/` |
//! | health | `GET /v1/ping/` |
//!
//! `infer: false` is deliberate: the record SPI writes what the caller handed over and
//! never asks the vendor to run its own LLM extraction, so what was stored is what is
//! recalled. mem0 has no memory kinds, so every record decoded from a vendor body is
//! [`MemoryType::SemanticMemory`](crate::record::MemoryType::SemanticMemory).
//!
//! # Rail
//!
//! The adapter keeps Python's three mem0 tools — `mem0_profile`, `mem0_search`, and
//! `mem0_conclude` — their payload shapes, and its two-line prompt block, dispatching them
//! over the same record SPI so the wire paths are exercised by the rail too.
use std::fmt;

use serde_json::{json, Value};

use super::super::provider::MemoryProvider;
use super::super::rail;
use super::super::registry::{ProviderConfig, ProviderKind};
use super::super::rest::{RestProvider, WireSpec};
use crate::record::{AddOptions, MemoryFilter, MemoryScope, NewMemory, SearchOptions};

mod spec;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

use spec::Mem0Spec;

/// Registry id of the adapter (the Python provider id).
const ID: &str = "mem0";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "mem0";
/// The dependency an unconfigured handle names.
const MISSING: &str = "the `mem0ai` SDK (`AsyncMemoryClient`)";
/// `tracing` target of this module's events.
const LOG_TARGET: &str = "openjiuwen_memory::external::vendor::mem0";
/// Hits a rail prefetch renders.
const PREFETCH_TOP_K: usize = 5;
/// `mem0_search` result ceiling.
const MAX_TOOL_TOP_K: i64 = 50;
/// `mem0_search` result count when the tool call names none.
const DEFAULT_TOOL_TOP_K: i64 = 10;
/// Rail tool that lists every stored memory.
const PROFILE_TOOL: &str = "mem0_profile";
/// Rail tool that searches memories by meaning.
const SEARCH_TOOL: &str = "mem0_search";
/// Rail tool that stores one durable fact.
const CONCLUDE_TOOL: &str = "mem0_conclude";

/// The one [`WireSpec`] instance every mem0 adapter shares.
pub(super) static SPEC: &dyn WireSpec = &Mem0Spec;

/// Handle for the `mem0` hosted memory service (`mem0`).
///
/// Construction never fails: without an `endpoint` the handle is inert and every
/// operation names `MISSING` verbatim.
pub struct Mem0MemoryProvider {
    /// The REST backend driven by [`SPEC`].
    pub(super) inner: RestProvider,
}

impl Mem0MemoryProvider {
    /// Builds the adapter handle for `config`.
    #[must_use]
    pub fn new(config: &ProviderConfig) -> Self {
        Self {
            inner: RestProvider::from_config(SPEC, config),
        }
    }

    /// The configuration this handle was built from.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        self.inner.config()
    }

    /// The kind this adapter implements.
    #[must_use]
    pub fn kind(&self) -> ProviderKind {
        ProviderKind::Mem0
    }

    /// The wire shape this adapter speaks.
    #[must_use]
    pub fn spec(&self) -> &'static dyn WireSpec {
        self.inner.spec()
    }
}

impl fmt::Debug for Mem0MemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mem0MemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config().endpoint)
            .finish_non_exhaustive()
    }
}

/// The `{"error": …}` payload of a failed rail tool call, logged at `debug`.
fn tool_failure(tool: &str, error: &openjiuwen_common::error::Error) -> Value {
    tracing::debug!(
        target: LOG_TARGET,
        tool,
        %error,
        "the mem0 rail tool call failed; the model sees the error payload"
    );
    json!({ "error": error.to_string() })
}

/// The three mem0 rail tool specs, mirroring the Python adapter.
fn tool_schemas() -> Vec<Value> {
    vec![
        rail::tool_schema(
            PROFILE_TOOL,
            "Retrieve all stored memories about the user — preferences, facts, project context. Fast, no reranking. Use at conversation start.",
            json!({}),
            &[],
        ),
        rail::tool_schema(
            SEARCH_TOOL,
            "Search memories by meaning. Returns relevant facts ranked by similarity. Set rerank=true for higher accuracy on important queries.",
            json!({
                "query": rail::string_property("What to search for."),
                "rerank": rail::boolean_property("Enable reranking for precision (default: false)."),
                "top_k": rail::integer_property("Max results (default: 10, max: 50)."),
            }),
            &["query"],
        ),
        rail::tool_schema(
            CONCLUDE_TOOL,
            "Store a durable fact about the user. Stored verbatim (no LLM extraction). Use for explicit preferences, corrections, or decisions.",
            json!({
                "conclusion": rail::string_property("The fact to store."),
            }),
            &["conclusion"],
        ),
    ]
}

implement_vendor_provider!(
    Mem0MemoryProvider,
    inner;

    /// The three mem0 tool specs (profile, search, conclude).
    fn get_tool_schemas(&self) -> Vec<Value> {
        tool_schemas()
    }

    /// The scope mem0 rail calls use: the configured `user_id`/`agent_id` fallback.
    fn default_scope(&self) -> MemoryScope {
        super::scope_from_options(self.config())
    }

    /// Python's mem0 prompt block, naming the configured user.
    fn system_prompt_block(&self) -> String {
        let user = self
            .config()
            .option("user_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|user| !user.is_empty())
            .unwrap_or("?");
        format!(
            "# Mem0 Memory\nActive. User: {user}.\nUse mem0_search to find memories, mem0_conclude to store facts, mem0_profile for a full overview."
        )
    }

    /// Renders one recall of at most five hits; blank queries and failures render nothing.
    async fn prefetch(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }
        let scope = self.default_scope();
        let options = SearchOptions::default().with_top_k(PREFETCH_TOP_K);
        match MemoryProvider::search(&self.inner, &scope, query, &options).await {
            Ok(hits) => {
                let lines: Vec<String> = hits
                    .iter()
                    .filter(|hit| !hit.record.content.trim().is_empty())
                    .map(|hit| format!("- {}", hit.record.content))
                    .collect();
                if lines.is_empty() {
                    String::new()
                } else {
                    format!("## Mem0 Memory\n{}", lines.join("\n"))
                }
            }
            Err(error) => {
                tracing::debug!(
                    target: LOG_TARGET,
                    tool = "prefetch",
                    %error,
                    "the mem0 prefetch failed; the turn continues without recalled memories"
                );
                String::new()
            }
        }
    }

    /// Dispatches the three mem0 rail tools over the record SPI.
    ///
    /// Python's payloads are kept verbatim; a backend error is returned as
    /// `{"error": <message>}` instead of propagating, because the caller is a model.
    async fn handle_tool_call(&self, tool: &str, args: &Value) -> Value {
        let scope = self.default_scope();
        let arguments = rail::args_object(args);
        match tool {
            PROFILE_TOOL => match MemoryProvider::list(
                &self.inner,
                &MemoryFilter::for_scope(&scope),
            )
            .await
            {
                Ok(records) => {
                    let contents: Vec<&str> = records
                        .iter()
                        .map(|record| record.content.trim())
                        .filter(|content| !content.is_empty())
                        .collect();
                    if contents.is_empty() {
                        json!({ "result": "No memories stored yet." })
                    } else {
                        let count = contents.len();
                        json!({ "result": contents.join("\n"), "count": count })
                    }
                }
                Err(error) => tool_failure(tool, &error),
            },
            SEARCH_TOOL => {
                let Some(query) = rail::string_arg(arguments, "query")
                    .map(str::trim)
                    .filter(|query| !query.is_empty())
                else {
                    return json!({ "error": "Missing required parameter: query" });
                };
                let top_k = rail::integer_arg(arguments, "top_k")
                    .unwrap_or(DEFAULT_TOOL_TOP_K)
                    .clamp(1, MAX_TOOL_TOP_K);
                let top_k = usize::try_from(top_k).unwrap_or(PREFETCH_TOP_K);
                let options = SearchOptions::default()
                    .with_top_k(top_k)
                    .with_threshold(0.0);
                match MemoryProvider::search(&self.inner, &scope, query, &options).await {
                    Ok(hits) if hits.is_empty() => {
                        json!({ "result": "No relevant memories found." })
                    }
                    Ok(hits) => {
                        let results: Vec<Value> = hits
                            .iter()
                            .map(|hit| {
                                json!({ "memory": hit.record.content, "score": hit.score })
                            })
                            .collect();
                        let count = results.len();
                        json!({ "results": results, "count": count })
                    }
                    Err(error) => tool_failure(tool, &error),
                }
            }
            CONCLUDE_TOOL => {
                let Some(conclusion) = rail::string_arg(arguments, "conclusion")
                    .map(str::trim)
                    .filter(|conclusion| !conclusion.is_empty())
                else {
                    return json!({ "error": "Missing required parameter: conclusion" });
                };
                let memory = NewMemory::semantic(conclusion);
                match MemoryProvider::add(
                    &self.inner,
                    &scope,
                    vec![memory],
                    &AddOptions::default(),
                )
                .await
                {
                    Ok(_outcome) => json!({ "result": "Fact stored." }),
                    Err(error) => tool_failure(tool, &error),
                }
            }
            other => json!({ "error": format!("unknown tool: {other}") }),
        }
    }
);
