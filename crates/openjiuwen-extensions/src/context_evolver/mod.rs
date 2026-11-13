//! Memory-evolution schemas, algorithm names, and the service seam.
//!
//! Ports the dependency-free core of the Python
//! `openjiuwen/extensions/context_evolver/` package (v0.1.20). The Python tree
//! mixes small, dependency-free pieces — memory schemas, algorithm names, JSON
//! persistence shapes, a prompt table, and an in-memory cosine vector store —
//! with pipelines driven by an LLM, an embedding model, and Milvus. Rust ports
//! the former and models the latter behind [`TaskMemoryService`], so a host can
//! inject any model or persistence backend; the bundled
//! [`UnsupportedTaskMemoryService`] answers [`Error::Unsupported`] for every
//! LLM-driven flow instead of pretending to run it.
//!
//! Python counterparts for the ported behavior:
//! `tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py`
//! (`TestAlgorithmNormalization`) and `test_summary_flow.py`
//! (`test_all_algorithm_normalizations`). The live retrieve/summarize flows are
//! skipped in Python without an API key and are deferred here.
//!
//! # Example
//!
//! ```
//! use openjiuwen_extensions::context_evolver::{normalize_algo_name, Algo};
//!
//! assert_eq!(normalize_algo_name("Reasoning-Bank").unwrap(), Algo::ReasoningBank);
//! assert_eq!(normalize_algo_name("rb").unwrap(), Algo::ReasoningBank);
//! assert_eq!(Algo::Refcon.family(), openjiuwen_extensions::context_evolver::AlgoFamily::Reme);
//! ```
use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod memory_store;
pub mod prompt;
pub mod record;
pub mod service;
pub mod store;

pub use memory_store::{MemoryVectorStore, ScoredNode, VectorNode};
pub use prompt::{
    parse_retrieve, parse_summarize, parse_trajectories, render_memory_string, retrieve_prompt,
    summarize_prompt, PromptPair, RetrievedHit, SummaryDraft, SummaryOperation, Trajectory,
    TrajectoryFeedback,
};
pub use record::{generate_memory_id, MemoryDraft, MemoryRecord};
pub use service::{
    create_task_memory_service, DeterministicTaskMemoryService, HashingEmbedding, MemoryBackend,
    MemoryEmbedding, TaskMemoryConfig,
};
pub use store::{InMemoryMemoryStore, JsonFileMemoryStore, MemoryStore};

/// The memory algorithm selected for retrieval and summarization.
///
/// The variants mirror Python's accepted names (`ACE`, `ReasoningBank`/`RB`,
/// `ReMe`, `RefCon`, `DivCon`, `Cognition`); serde represents them in
/// `snake_case`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Algo {
    /// ACE playbook memory.
    Ace,
    /// `ReasoningBank` memory (Python alias `RB`).
    ReasoningBank,
    /// `ReMe` memory.
    Reme,
    /// `RefCon` memory (a `ReMe`-family pipeline).
    Refcon,
    /// `DivCon` memory (a `ReMe`-family pipeline).
    Divcon,
    /// Cognition memory.
    Cognition,
}

impl Algo {
    /// The canonical lower-case name used in configuration and error messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ace => "ace",
            Self::ReasoningBank => "reasoningbank",
            Self::Reme => "reme",
            Self::Refcon => "refcon",
            Self::Divcon => "divcon",
            Self::Cognition => "cognition",
        }
    }

    /// The implementation family that provides the algorithm.
    ///
    /// `Refcon` and `Divcon` are variants of the `ReMe` family; every other
    /// algorithm maps to its own family.
    #[must_use]
    pub const fn family(self) -> AlgoFamily {
        match self {
            Self::Ace => AlgoFamily::Ace,
            Self::ReasoningBank => AlgoFamily::ReasoningBank,
            Self::Reme | Self::Refcon | Self::Divcon => AlgoFamily::Reme,
            Self::Cognition => AlgoFamily::Cognition,
        }
    }
}

/// The pipeline family an [`Algo`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgoFamily {
    /// The ACE playbook pipeline.
    Ace,
    /// The `ReasoningBank` pipeline.
    ReasoningBank,
    /// The `ReMe` pipeline, shared by `ReMe`, `RefCon`, and `DivCon`.
    Reme,
    /// The Cognition pipeline.
    Cognition,
}

/// Normalizes an algorithm name to an [`Algo`].
///
/// Ports Python's `TaskMemoryService.normalize_algo_name`: matching is
/// case-insensitive and separators (`-`, `_`, and whitespace) are ignored, so
/// `"Reasoning-Bank"`, `"reasoning_bank"`, and `"reason ing_bank"` all select
/// [`Algo::ReasoningBank`]. `"rb"` is the Python alias for `ReasoningBank`,
/// while `"reme"`, `"refcon"`/`"ref_con"`, and `"divcon"`/`"div_con"` select
/// their explicit variants.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an unknown name, preserving the caller's
/// original spelling in the message.
pub fn normalize_algo_name(name: &str) -> Result<Algo> {
    let key = normalize_key(name);
    match key.as_str() {
        "ace" => Ok(Algo::Ace),
        "rb" | "reasoningbank" => Ok(Algo::ReasoningBank),
        "reme" => Ok(Algo::Reme),
        "refcon" => Ok(Algo::Refcon),
        "divcon" => Ok(Algo::Divcon),
        "cognition" => Ok(Algo::Cognition),
        _ => Err(Error::Validation(format!(
            "unknown memory algorithm '{name}'"
        ))),
    }
}

/// Collapses case and separators so the accepted spellings share one key.
fn normalize_key(name: &str) -> String {
    name.chars()
        .filter(|character| !character.is_whitespace() && *character != '_' && *character != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// A request to persist one memory entry.
///
/// Mirrors Python's `AddMemoryRequest` dataclass defaults: [`section`]
/// defaults to `"general"` and every other optional field is empty.
///
/// [`section`]: AddMemoryRequest::section
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct AddMemoryRequest {
    /// The memory text to store.
    pub content: String,
    /// Optional retrieval query associated with the memory.
    pub query: Option<String>,
    /// Optional description of when the memory is useful.
    pub when_to_use: Option<String>,
    /// Optional human-readable title.
    pub title: Option<String>,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Memory section; Python's default is `"general"`.
    #[serde(default = "default_section")]
    pub section: String,
    /// Optional label used for grouping.
    pub label: Option<String>,
    /// Arbitrary extra attributes; JSON `null` and a missing field both mean an
    /// empty map.
    #[serde(default, deserialize_with = "deserialize_attributes")]
    pub attributes: BTreeMap<String, Value>,
}

impl Default for AddMemoryRequest {
    fn default() -> Self {
        Self {
            content: String::new(),
            query: None,
            when_to_use: None,
            title: None,
            description: None,
            section: default_section(),
            label: None,
            attributes: BTreeMap::new(),
        }
    }
}

/// The Python `AddMemoryRequest.section` default.
fn default_section() -> String {
    "general".to_string()
}

/// Accepts a missing map, an explicit `null`, or an object.
fn deserialize_attributes<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let attributes = Option::<BTreeMap<String, Value>>::deserialize(deserializer)?;
    Ok(attributes.unwrap_or_default())
}

/// Input card for [`TaskMemoryService::retrieve`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RetrieveRequest {
    /// Identifier of the memory owner.
    pub user_id: String,
    /// Query used to rank stored memories.
    pub query: String,
}

/// Output card for [`TaskMemoryService::retrieve`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RetrieveResponse {
    /// Pipeline status, mirroring Python's response card.
    pub status: String,
    /// The rendered memory string injected into the prompt.
    pub memory_string: String,
    /// Structured retrieval payload, mirroring Python's response card.
    pub retrieved_memory: Value,
}

/// Input card for [`TaskMemoryService::summarize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SummarizeRequest {
    /// Identifier of the memory owner.
    pub user_id: String,
    /// Trajectories to summarize into new memories.
    pub trajectories: Value,
}

/// Output card for [`TaskMemoryService::summarize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SummarizeResponse {
    /// Pipeline status, mirroring Python's response card.
    pub status: String,
    /// Structured memories produced by the summary pipeline.
    pub memories: Value,
}

/// The pluggable seam for the LLM-, embedding-, and Milvus-driven flows.
///
/// A host supplies a model-backed implementation; the bundled
/// [`UnsupportedTaskMemoryService`] is the documented deferred default.
#[async_trait]
pub trait TaskMemoryService: Send + Sync {
    /// Retains the memories relevant to [`RetrieveRequest::query`].
    ///
    /// # Errors
    ///
    /// Implementations report backend, model, or persistence failures.
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse>;

    /// Turns trajectories into new executable memories.
    ///
    /// # Errors
    ///
    /// Implementations report backend, model, or persistence failures.
    async fn summarize(&self, request: SummarizeRequest) -> Result<SummarizeResponse>;

    /// Persists one memory entry for `user_id`.
    ///
    /// # Errors
    ///
    /// Implementations report validation or persistence failures.
    async fn add_memory(&self, user_id: &str, request: AddMemoryRequest) -> Result<()>;

    /// Loads every persisted memory for `user_id`.
    ///
    /// # Errors
    ///
    /// Implementations report persistence or deserialization failures.
    fn load_memories(&self, user_id: &str) -> Result<Vec<AddMemoryRequest>>;
}

/// The documented deferred [`TaskMemoryService`].
///
/// Every flow answers [`Error::Unsupported`], naming the selected algorithm, so
/// a configuration that selects an LLM-driven pipeline fails loudly instead of
/// returning fabricated memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedTaskMemoryService {
    algo: Algo,
}

impl UnsupportedTaskMemoryService {
    /// Wraps an algorithm whose pipeline is deferred.
    #[must_use]
    pub const fn new(algo: Algo) -> Self {
        Self { algo }
    }

    /// The algorithm whose pipeline is deferred.
    #[must_use]
    pub const fn algo(&self) -> Algo {
        self.algo
    }

    /// The documented deferral error for every trait method.
    fn unsupported(self) -> Error {
        Error::Unsupported(format!(
            "context_evolver '{}' pipeline is deferred; inject a TaskMemoryService",
            self.algo.as_str()
        ))
    }
}

#[async_trait]
impl TaskMemoryService for UnsupportedTaskMemoryService {
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    async fn retrieve(&self, _request: RetrieveRequest) -> Result<RetrieveResponse> {
        Err(self.unsupported())
    }

    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    async fn summarize(&self, _request: SummarizeRequest) -> Result<SummarizeResponse> {
        Err(self.unsupported())
    }

    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    async fn add_memory(&self, _user_id: &str, _request: AddMemoryRequest) -> Result<()> {
        Err(self.unsupported())
    }

    fn load_memories(&self, _user_id: &str) -> Result<Vec<AddMemoryRequest>> {
        Err(self.unsupported())
    }
}

#[cfg(test)]
mod tests {
    // Python counterparts: `tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py`
    // (`TestAlgorithmNormalization`) and `test_summary_flow.py`
    // (`test_all_algorithm_normalizations`). Live flows are skipped in Python.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    #[test]
    fn normalize_accepts_every_python_spelling() {
        let cases = [
            ("ace", Algo::Ace),
            ("ACE", Algo::Ace),
            ("Ace", Algo::Ace),
            ("rb", Algo::ReasoningBank),
            ("RB", Algo::ReasoningBank),
            ("reasoningbank", Algo::ReasoningBank),
            ("REASONINGBANK", Algo::ReasoningBank),
            ("reasoning_bank", Algo::ReasoningBank),
            ("Reasoning-Bank", Algo::ReasoningBank),
            ("reason ing_bank", Algo::ReasoningBank),
            ("reme", Algo::Reme),
            ("ReMe", Algo::Reme),
            ("refcon", Algo::Refcon),
            ("REFCON", Algo::Refcon),
            ("ref_con", Algo::Refcon),
            ("Ref-Con", Algo::Refcon),
            ("divcon", Algo::Divcon),
            ("DIVCON", Algo::Divcon),
            ("div_con", Algo::Divcon),
            ("Div-Con", Algo::Divcon),
            ("cognition", Algo::Cognition),
            ("COGNITION", Algo::Cognition),
        ];
        for (name, expected) in cases {
            assert_eq!(normalize_algo_name(name).unwrap(), expected, "name: {name}");
        }
    }

    #[test]
    fn normalize_trims_surrounding_whitespace() {
        assert_eq!(normalize_algo_name("  reme  ").unwrap(), Algo::Reme);
        assert_eq!(normalize_algo_name("\tREF_CON\n").unwrap(), Algo::Refcon);
    }

    #[test]
    fn normalize_rejects_unknown_names_with_original_spelling() {
        for name in ["bogus", "INVALID_ALGO", "", "   "] {
            let error = normalize_algo_name(name).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "name: {name:?}");
            assert!(
                error.message().contains(name),
                "message must keep the original spelling: {}",
                error.message()
            );
            assert!(error.message().contains("unknown memory algorithm"));
        }
    }

    #[test]
    fn as_str_round_trips_through_normalize() {
        let algos = [
            Algo::Ace,
            Algo::ReasoningBank,
            Algo::Reme,
            Algo::Refcon,
            Algo::Divcon,
            Algo::Cognition,
        ];
        for algo in algos {
            assert_eq!(normalize_algo_name(algo.as_str()).unwrap(), algo);
        }
        assert_eq!(Algo::Ace.as_str(), "ace");
        assert_eq!(Algo::ReasoningBank.as_str(), "reasoningbank");
        assert_eq!(Algo::Reme.as_str(), "reme");
        assert_eq!(Algo::Refcon.as_str(), "refcon");
        assert_eq!(Algo::Divcon.as_str(), "divcon");
        assert_eq!(Algo::Cognition.as_str(), "cognition");
    }

    #[test]
    fn family_maps_refcon_and_divcon_to_reme() {
        assert_eq!(Algo::Ace.family(), AlgoFamily::Ace);
        assert_eq!(Algo::ReasoningBank.family(), AlgoFamily::ReasoningBank);
        assert_eq!(Algo::Reme.family(), AlgoFamily::Reme);
        assert_eq!(Algo::Refcon.family(), AlgoFamily::Reme);
        assert_eq!(Algo::Divcon.family(), AlgoFamily::Reme);
        assert_eq!(Algo::Cognition.family(), AlgoFamily::Cognition);
    }

    #[test]
    fn algo_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(Algo::ReasoningBank).unwrap(),
            json!("reasoning_bank")
        );
        assert_eq!(
            serde_json::from_value::<Algo>(json!("refcon")).unwrap(),
            Algo::Refcon
        );
        assert_eq!(
            serde_json::from_value::<AlgoFamily>(json!("reasoning_bank")).unwrap(),
            AlgoFamily::ReasoningBank
        );
    }

    #[test]
    fn add_memory_request_defaults_section_to_general() {
        assert_eq!(AddMemoryRequest::default().section, "general");
        assert!(AddMemoryRequest::default().attributes.is_empty());
        assert_eq!(AddMemoryRequest::default().content, "");
    }

    #[test]
    fn add_memory_request_round_trips_through_json() {
        let mut request = AddMemoryRequest {
            content: "remember this".into(),
            query: Some("what?".into()),
            when_to_use: Some("when testing".into()),
            title: Some("title".into()),
            description: Some("description".into()),
            section: "general".into(),
            label: Some("label".into()),
            attributes: BTreeMap::new(),
        };
        request
            .attributes
            .insert("target".into(), json!("workspace"));
        let encoded = serde_json::to_string(&request).unwrap();
        let decoded: AddMemoryRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn add_memory_request_tolerates_missing_null_and_unknown_fields() {
        let minimal: AddMemoryRequest =
            serde_json::from_value(json!({"content": "hello"})).unwrap();
        assert_eq!(minimal.section, "general");
        assert!(minimal.attributes.is_empty());
        assert_eq!(minimal.content, "hello");

        let nulls: AddMemoryRequest = serde_json::from_value(json!({
            "content": "hello",
            "query": null,
            "attributes": null
        }))
        .unwrap();
        assert_eq!(nulls.query, None);
        assert!(nulls.attributes.is_empty());

        let extras: AddMemoryRequest = serde_json::from_value(json!({
            "content": "hello",
            "unknown_field": 42,
            "attributes": {"target": "x"}
        }))
        .unwrap();
        assert_eq!(extras.attributes.get("target"), Some(&json!("x")));
    }

    #[test]
    fn cards_round_trip_through_json() {
        let retrieve_request = RetrieveRequest {
            user_id: "user-1".into(),
            query: "where?".into(),
        };
        let encoded = serde_json::to_string(&retrieve_request).unwrap();
        let decoded: RetrieveRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, retrieve_request);

        let retrieve_response = RetrieveResponse {
            status: "success".into(),
            memory_string: "memory".into(),
            retrieved_memory: json!({"memories": []}),
        };
        let encoded = serde_json::to_string(&retrieve_response).unwrap();
        let decoded: RetrieveResponse = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, retrieve_response);

        let summarize_request = SummarizeRequest {
            user_id: "user-1".into(),
            trajectories: json!([{"steps": []}]),
        };
        let encoded = serde_json::to_string(&summarize_request).unwrap();
        let decoded: SummarizeRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, summarize_request);

        let summarize_response = SummarizeResponse {
            status: "success".into(),
            memories: json!({"memories": [{"content": "x"}]}),
        };
        let encoded = serde_json::to_string(&summarize_response).unwrap();
        let decoded: SummarizeResponse = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, summarize_response);
    }

    #[tokio::test]
    async fn unsupported_service_reports_every_flow() {
        let service = UnsupportedTaskMemoryService::new(Algo::Reme);
        assert_eq!(service.algo(), Algo::Reme);
        let expected = "context_evolver 'reme' pipeline is deferred; inject a TaskMemoryService";

        let error = service
            .retrieve(RetrieveRequest {
                user_id: "user".into(),
                query: "query".into(),
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.message().as_ref(), expected);

        let error = service
            .summarize(SummarizeRequest {
                user_id: "user".into(),
                trajectories: json!([]),
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.message().as_ref(), expected);

        let error = service
            .add_memory("user", AddMemoryRequest::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.message().as_ref(), expected);

        let error = service.load_memories("user").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(error.message().as_ref(), expected);
    }

    #[tokio::test]
    async fn unsupported_service_names_each_algorithm() {
        for (algo, name) in [
            (Algo::Ace, "ace"),
            (Algo::ReasoningBank, "reasoningbank"),
            (Algo::Refcon, "refcon"),
            (Algo::Divcon, "divcon"),
            (Algo::Cognition, "cognition"),
        ] {
            let error = UnsupportedTaskMemoryService::new(algo)
                .load_memories("user")
                .unwrap_err();
            assert!(
                error
                    .message()
                    .contains(&format!("context_evolver '{name}' pipeline is deferred")),
                "{}",
                error.message()
            );
        }
    }

    #[tokio::test]
    async fn exposed_vector_store_types_are_reachable() {
        let node = VectorNode::new("id", vec![1.0]);
        let scored = ScoredNode { node, score: 1.0 };
        let store = MemoryVectorStore::new();
        assert_eq!(scored.node.id, "id");
        assert_eq!(store.len().await, 0);
    }
}

pub mod completion;
pub use completion::{
    ChatCompletionConfig, MemoryCompletion, OpenAiChatCompletion, ScriptedCompletion,
};
