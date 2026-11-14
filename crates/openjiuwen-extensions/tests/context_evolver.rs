//! Crate-boundary integration suite for the context-evolver feature.
//!
//! Every assertion goes through the public
//! `openjiuwen_extensions::context_evolver` surface — no `crate::` internals — so
//! a regression in a re-export, a factory signature, a response card, or the
//! deferral contract fails here first. It proves the offline deterministic path
//! (lexical retrieve, extractive summarize, `{"memories": [...]}` persistence
//! over a `tempfile` directory), the model-driven path over a loopback
//! `wiremock` peer speaking `OpenAI`'s `/chat/completions` format — including
//! fenced, prose, and hallucinated replies — and the unchanged deferral.
//!
//! Python counterparts are named per test under
//! `tests/unit_tests/extensions/context_evolver/`; Python skips the live model
//! flows without an API key, which is why the offline service exists.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::{InMemoryVectorStore, VectorStore};
use openjiuwen_extensions::context_evolver::{
    create_task_memory_service, generate_memory_id, normalize_algo_name, AddMemoryRequest, Algo,
    AlgoFamily, ChatCompletionConfig, DeterministicTaskMemoryService, HashingEmbedding,
    InMemoryMemoryStore, JsonFileMemoryStore, KernelVectorIndex, MemoryBackend, MemoryDraft,
    MemoryRecord, MemoryStore, OpenAiChatCompletion, RetrieveRequest, ScriptedCompletion,
    SummarizeRequest, TaskMemoryConfig, TaskMemoryService, UnsupportedTaskMemoryService,
    VectorBackend, VectorIndex, VectorNode, DEFAULT_VECTOR_COLLECTION,
};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The user every offline scenario seeds memories under.
const USER: &str = "context-evolver-user";
/// A second user, used to prove that memory files are per-user.
const OTHER_USER: &str = "context-evolver-other";
/// A user whose batches are always empty.
const EMPTY_USER: &str = "context-evolver-empty";
/// The `OpenAI`-compatible chat-completions path the mock serves.
const CHAT_PATH: &str = "/chat/completions";

/// Every algorithm the Python table accepts, for exhaustive loops.
const ALL_ALGOS: [Algo; 6] = [
    Algo::Ace,
    Algo::ReasoningBank,
    Algo::Reme,
    Algo::Refcon,
    Algo::Divcon,
    Algo::Cognition,
];

/// The strongly matching offline memory the deterministic retrieve ranks first.
const RIPGREP_MEMORY: &str =
    "Prefer ripgrep with a glob filter when searching large repositories, because it skips \
     vendored trees.";

/// The pagination memory the model-driven retrieve pre-ranks first.
const PAGINATION_CURSOR: &str =
    "Use cursor pagination with a stable sort key for large result sets.";
/// The pagination memory the model is asked to rank first instead.
const PAGINATION_TOKEN: &str = "Return the next page token in the response envelope.";
/// The query the offline memories are ranked against.
const RANK_QUERY: &str = "ripgrep glob filter searching repositories";
/// The query the model-driven retrieve ranks candidates against.
const PAGINATION_QUERY: &str = "stable cursor pagination for large result sets";

/// Builds a memory entry holding only `content`.
fn memory(content: &str) -> AddMemoryRequest {
    AddMemoryRequest {
        content: content.to_string(),
        ..AddMemoryRequest::default()
    }
}

/// Builds a memory entry with the applicability condition the `ReMe` family uses.
fn conditional_memory(content: &str, when_to_use: &str) -> AddMemoryRequest {
    AddMemoryRequest {
        content: content.to_string(),
        when_to_use: Some(when_to_use.to_string()),
        ..AddMemoryRequest::default()
    }
}

/// A retrieve input card for `user_id`.
fn retrieve_request(user_id: &str, query: &str) -> RetrieveRequest {
    RetrieveRequest {
        user_id: user_id.to_string(),
        query: query.to_string(),
    }
}

/// A summarize input card for `user_id`.
fn summarize_request(user_id: &str, trajectories: Value) -> SummarizeRequest {
    SummarizeRequest {
        user_id: user_id.to_string(),
        trajectories,
    }
}

/// One successful trajectory, as the summarize scenarios submit it.
fn helpful_batch() -> Value {
    json!([{
        "query": "count every playlist",
        "response": "The agent looped with a stable cursor and counted 12 playlists.",
        "feedback": "helpful",
    }])
}

/// The `{"memories": [...]}` entries of a response card.
fn entries(card: &Value) -> &[Value] {
    card["memories"]
        .as_array()
        .expect("every response card carries a memories array")
}

/// The `id` of every entry of a response card, in card order.
fn entry_ids(card: &Value) -> Vec<String> {
    entries(card)
        .iter()
        .map(|entry| {
            entry["id"]
                .as_str()
                .expect("every entry carries an id")
                .to_string()
        })
        .collect()
}

/// The id the store holds for the record whose content is `content`.
async fn stored_id(store: &InMemoryMemoryStore, user_id: &str, content: &str) -> String {
    store
        .load(user_id)
        .await
        .expect("the in-memory store loads")
        .into_iter()
        .find(|record| record.content == content)
        .unwrap_or_else(|| panic!("no stored memory with content '{content}'"))
        .id
}

/// Persists the two pagination memories and returns their `(cursor, token)` ids.
async fn seed_pagination(
    service: &dyn TaskMemoryService,
    store: &InMemoryMemoryStore,
) -> (String, String) {
    let cursor = conditional_memory(PAGINATION_CURSOR, "listing rows");
    let token = conditional_memory(PAGINATION_TOKEN, "answering a page request");
    service
        .add_memory(USER, cursor)
        .await
        .expect("the cursor memory is persisted");
    service
        .add_memory(USER, token)
        .await
        .expect("the token memory is persisted");
    let cursor_id = stored_id(store, USER, PAGINATION_CURSOR).await;
    let token_id = stored_id(store, USER, PAGINATION_TOKEN).await;
    assert_ne!(cursor_id, token_id, "the two memories have distinct ids");
    (cursor_id, token_id)
}

/// A chat-completions mock answering `reply` as `choices[0].message.content`.
async fn mount_reply(server: &MockServer, reply: String) {
    Mock::given(method("POST"))
        .and(path(CHAT_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}],
        })))
        .mount(server)
        .await;
}

/// Persists the four distinct-topic memories the offline ranking scenarios use.
async fn seed_offline_memories(service: &dyn TaskMemoryService) {
    for request in [
        memory(
            "Wrap every database transaction in a retry so a transient deadlock is replayed \
             instead of surfaced.",
        ),
        memory(RIPGREP_MEMORY),
        memory(
            "Pin the toolchain with a rustup override so a clean checkout builds with the \
             tested compiler.",
        ),
        memory(
            "Return the next page token in the response envelope instead of exposing raw \
             offsets.",
        ),
    ] {
        service
            .add_memory(USER, request)
            .await
            .expect("the memory is persisted");
    }
}

/// The two memories the file-shape scenarios persist, in seeding order.
fn file_scenario_memories() -> [AddMemoryRequest; 2] {
    [
        memory("Page with a cursor so a long listing keeps its order."),
        conditional_memory(
            "Summarise a failed build by quoting the first compiler error.",
            "debugging a broken build",
        ),
    ]
}

/// Asserts every trait method of `service` answers the documented deferral.
async fn assert_deferred(service: &dyn TaskMemoryService, expected: &str) {
    let error = service
        .retrieve(retrieve_request(USER, "query"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message().as_ref(), expected);

    let error = service
        .summarize(summarize_request(USER, helpful_batch()))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message().as_ref(), expected);

    let error = service
        .add_memory(USER, memory("content"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message().as_ref(), expected);

    let error = service.load_memories(USER).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message().as_ref(), expected);
}

// Port of `tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py`:
// the offline lexical ranking, which Python skips without an API key.
#[tokio::test]
async fn deterministic_retrieve_ranks_the_file_store_lexically() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Deterministic,
        Arc::new(JsonFileMemoryStore::new(dir.path())),
    );
    seed_offline_memories(service.as_ref()).await;

    let response = service
        .retrieve(retrieve_request(USER, RANK_QUERY))
        .await
        .expect("the offline retrieve succeeds");
    assert_eq!(response.status, "success");
    assert_ne!(response.memory_string, "");
    assert!(
        response
            .memory_string
            .contains("Prefer ripgrep with a glob filter"),
        "{}",
        response.memory_string
    );

    let card = &response.retrieved_memory;
    let found = entries(card);
    assert_eq!(found.len(), 4, "every seeded memory is a candidate: {card}");
    let expected_id = generate_memory_id(
        Algo::Reme,
        USER,
        &MemoryDraft::from_request(&memory(RIPGREP_MEMORY)),
    );
    assert_eq!(found[0]["id"].as_str(), Some(expected_id.as_str()));
    assert_eq!(found[0]["content"].as_str(), Some(RIPGREP_MEMORY));
    assert_eq!(found[0]["section"].as_str(), Some("general"));
    let scores: Vec<f64> = found
        .iter()
        .map(|entry| {
            entry["score"]
                .as_f64()
                .expect("every entry carries a score")
        })
        .collect();
    assert!(
        scores.windows(2).all(|pair| pair[0] >= pair[1]),
        "hits descend by score: {scores:?}"
    );
    assert!(
        scores[0] > scores[1],
        "the strongest match outranks the rest: {scores:?}"
    );
}

// Port of `tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py`
// (the `top_k` and score-floor knobs Python's service context carries).
#[tokio::test]
async fn deterministic_retrieve_honours_top_k_and_min_score() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme).with_top_k(1),
        MemoryBackend::Deterministic,
        Arc::new(JsonFileMemoryStore::new(dir.path())),
    );
    seed_offline_memories(service.as_ref()).await;
    let expected_id = generate_memory_id(
        Algo::Reme,
        USER,
        &MemoryDraft::from_request(&memory(RIPGREP_MEMORY)),
    );

    let response = service
        .retrieve(retrieve_request(USER, RANK_QUERY))
        .await
        .expect("the narrowed retrieve succeeds");
    assert_eq!(response.status, "success");
    assert_eq!(entry_ids(&response.retrieved_memory), [expected_id]);
    assert!(
        response.memory_string.starts_with("Memory 1:"),
        "{}",
        response.memory_string
    );
    assert!(!response.memory_string.contains("Memory 2:"));

    // A score floor above zero drops every hit of an unmatched query.
    let unrelated = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme).with_min_score(0.5),
        MemoryBackend::Deterministic,
        Arc::new(JsonFileMemoryStore::new(dir.path())),
    );
    let response = unrelated
        .retrieve(retrieve_request(
            USER,
            "quantum chromodynamics lattice gauge",
        ))
        .await
        .expect("an unmatched query is not an error");
    assert_eq!(response.status, "success");
    assert_eq!(entries(&response.retrieved_memory), [] as [Value; 0]);
    assert_eq!(response.memory_string, "");
}

// Port of `tests/unit_tests/extensions/context_evolver/test_file_connector.py`
// (the on-disk `{"memories": [...]}` shape and the `load_memories` projection).
#[tokio::test]
async fn json_file_store_round_trips_the_python_file_shape() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let store = JsonFileMemoryStore::new(dir.path());
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::ReasoningBank),
        MemoryBackend::Deterministic,
        Arc::new(store.clone()),
    );
    let [paged, build] = file_scenario_memories();
    service
        .add_memory(USER, paged.clone())
        .await
        .expect("the first memory is persisted");
    service
        .add_memory(USER, build.clone())
        .await
        .expect("the second memory is persisted");

    let path = store.path_for(USER);
    assert!(path.starts_with(dir.path()), "{}", path.display());
    let raw = std::fs::read_to_string(&path).expect("the store wrote a file");
    let file: Value = serde_json::from_str(&raw).expect("the file is JSON");
    let object = file.as_object().expect("the file is a JSON object");
    assert_eq!(object.len(), 1, "the file holds exactly one key: {file}");
    let persisted = entries(&file);
    assert_eq!(persisted.len(), 2);
    let ids = entry_ids(&file);

    // The order follows the generated ids, which the public id builder predicts.
    let expected: BTreeMap<String, String> = [
        (
            generate_memory_id(
                Algo::ReasoningBank,
                USER,
                &MemoryDraft::from_request(&paged),
            ),
            paged.content.clone(),
        ),
        (
            generate_memory_id(
                Algo::ReasoningBank,
                USER,
                &MemoryDraft::from_request(&build),
            ),
            build.content.clone(),
        ),
    ]
    .into_iter()
    .collect();
    let expected_ids: Vec<String> = expected.keys().cloned().collect();
    let expected_contents: Vec<String> = expected.values().cloned().collect();
    assert_eq!(
        ids, expected_ids,
        "the file is ordered by the generated ids"
    );
    let persisted_contents: Vec<String> = persisted
        .iter()
        .map(|entry| entry["content"].as_str().expect("content").to_string())
        .collect();
    assert_eq!(persisted_contents, expected_contents);

    let condition = persisted
        .iter()
        .find(|entry| entry["content"].as_str() == Some(build.content.as_str()))
        .expect("the conditional memory is in the file");
    assert_eq!(condition["section"].as_str(), Some("general"));
    assert_eq!(
        condition["when_to_use"].as_str(),
        Some("debugging a broken build")
    );

    // `load_memories` projects the same records back onto the request card.
    let loaded = service.load_memories(USER).expect("the records load");
    let loaded_contents: Vec<String> = loaded.iter().map(|item| item.content.clone()).collect();
    assert_eq!(loaded_contents, expected_contents);
    let conditional = loaded
        .iter()
        .find(|item| item.content == build.content)
        .expect("the conditional memory is loaded");
    assert_eq!(
        conditional.when_to_use.as_deref(),
        Some("debugging a broken build")
    );
}

// Port of `tests/unit_tests/extensions/context_evolver/test_file_connector.py`:
// one file per user, visible to a fresh service and isolated on clear.
#[tokio::test]
async fn json_file_store_persists_across_services_and_isolates_users() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let store = JsonFileMemoryStore::new(dir.path());
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::ReasoningBank),
        MemoryBackend::Deterministic,
        Arc::new(store.clone()),
    );
    for request in file_scenario_memories() {
        service
            .add_memory(USER, request)
            .await
            .expect("the memory is persisted");
    }

    // A fresh service over the same directory sees the persisted records.
    let reopened = create_task_memory_service(
        TaskMemoryConfig::new(Algo::ReasoningBank),
        MemoryBackend::Deterministic,
        Arc::new(JsonFileMemoryStore::new(dir.path())),
    );
    assert_eq!(
        reopened
            .load_memories(USER)
            .expect("the records load")
            .len(),
        2
    );

    reopened
        .add_memory(
            OTHER_USER,
            memory("Keep the other user's memories separate."),
        )
        .await
        .expect("the second user's memory is persisted");
    assert_ne!(store.path_for(USER), store.path_for(OTHER_USER));
    assert_eq!(
        reopened
            .load_memories(USER)
            .expect("the records load")
            .len(),
        2,
        "a second user does not touch the first"
    );

    store
        .clear(OTHER_USER)
        .await
        .expect("the per-user clear succeeds");
    assert_eq!(
        reopened
            .load_memories(OTHER_USER)
            .expect("the cleared user loads"),
        Vec::<AddMemoryRequest>::new()
    );
    assert_eq!(
        reopened
            .load_memories(USER)
            .expect("the records load")
            .len(),
        2,
        "clearing one user leaves the others"
    );
}

// Port of `tests/unit_tests/extensions/context_evolver/test_summary_flow.py`
// (the extractive offline summarize; Python distills with an LLM instead).
#[tokio::test]
async fn deterministic_summarize_extracts_and_dedupes_trajectories() {
    let store = Arc::new(InMemoryMemoryStore::new());
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Deterministic,
        store.clone(),
    );
    let helpful = "Loop over the playlists with a stable cursor until the next token is empty.";
    let harmful = "Read the password from the environment instead of the source tree.";
    let trajectories = json!([
        {"query": "count every playlist", "response": helpful, "feedback": "helpful"},
        {"query": "log in with a stored password", "response": harmful, "feedback": "harmful"},
    ]);

    let response = service
        .summarize(summarize_request(USER, trajectories.clone()))
        .await
        .expect("the offline summarize succeeds");
    assert_eq!(response.status, "success");
    let drafts = entries(&response.memories);
    assert_eq!(drafts.len(), 2, "one extractive draft per trajectory");
    assert_eq!(drafts[0]["content"].as_str(), Some(helpful));
    assert_eq!(
        drafts[0]["when_to_use"].as_str(),
        Some("count every playlist")
    );
    assert_eq!(drafts[0]["section"].as_str(), Some("general"));
    assert_eq!(drafts[1]["content"].as_str(), Some(harmful));

    let loaded = service.load_memories(USER).expect("the drafts persisted");
    let contents: Vec<&str> = loaded.iter().map(|item| item.content.as_str()).collect();
    assert_eq!(contents, vec![helpful, harmful]);

    // The ids are deterministic, so a replayed batch is an upsert, not a copy.
    service
        .summarize(summarize_request(USER, trajectories))
        .await
        .expect("the replayed summarize succeeds");
    assert_eq!(
        service
            .load_memories(USER)
            .expect("the drafts persisted")
            .len(),
        2,
        "a replayed batch does not duplicate the records"
    );

    // An empty batch is a success that writes nothing at all.
    let empty = service
        .summarize(summarize_request(EMPTY_USER, json!([])))
        .await
        .expect("an empty batch is a success");
    assert_eq!(empty.status, "success");
    assert_eq!(entries(&empty.memories), [] as [Value; 0]);
    assert_eq!(
        service
            .load_memories(EMPTY_USER)
            .expect("the empty user loads"),
        Vec::<AddMemoryRequest>::new()
    );
    assert_eq!(
        store.load(EMPTY_USER).await.expect("the empty user loads"),
        Vec::<MemoryRecord>::new()
    );
}

// Port of `tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py`
// (the live retrieve flow) driven against a loopback `wiremock` peer.
#[tokio::test]
async fn llm_retrieve_ranks_candidates_over_the_http_adapter() {
    let server = MockServer::start().await;
    let store = Arc::new(InMemoryMemoryStore::new());
    let completion = OpenAiChatCompletion::new(
        ChatCompletionConfig::new(server.uri(), "test-model").with_api_key("test-key"),
    )
    .expect("the HTTP adapter builds");
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Llm(Arc::new(completion)),
        store.clone(),
    );
    let (cursor_id, token_id) = seed_pagination(service.as_ref(), &store).await;

    // The reply ranks the weaker lexical candidate first, so an assertion on the
    // order cannot pass on the pre-ranked fallback.
    let reply = json!({
        "ranked_ids": [token_id.clone(), cursor_id.clone()],
        "reasoning": "the token directly answers the request",
    })
    .to_string();
    mount_reply(&server, reply).await;

    let response = service
        .retrieve(retrieve_request(USER, PAGINATION_QUERY))
        .await
        .expect("the model-driven retrieve succeeds");
    assert_eq!(response.status, "success");
    assert_eq!(
        entry_ids(&response.retrieved_memory),
        [token_id, cursor_id],
        "the model's ranking order wins over the lexical pre-rank"
    );
    let expected = format!(
        "Memory 1:\n  When to use: answering a page request\n  Content: {PAGINATION_TOKEN}\n\n\
         Memory 2:\n  When to use: listing rows\n  Content: {PAGINATION_CURSOR}"
    );
    assert_eq!(
        response.memory_string, expected,
        "the `ReMe` family rendering follows the ranked hits"
    );

    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one completion call per retrieve");
    let request = &requests[0];
    assert_eq!(
        request
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-key")
    );
    let body: Value = serde_json::from_slice(&request.body).expect("the request body is JSON");
    assert_eq!(body["model"].as_str(), Some("test-model"));
    let messages = body["messages"]
        .as_array()
        .expect("the request carries a messages array");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"].as_str(), Some("system"));
    assert_ne!(
        messages[0]["content"]
            .as_str()
            .expect("the system prompt is text"),
        ""
    );
    let prompt = messages[1]["content"]
        .as_str()
        .expect("the user prompt is text");
    for needle in [PAGINATION_QUERY, PAGINATION_CURSOR, PAGINATION_TOKEN] {
        assert!(
            prompt.contains(needle),
            "the prompt carries '{needle}': {prompt}"
        );
    }
}

// Port of `tests/unit_tests/extensions/context_evolver/test_summary_flow.py`
// (the live summarize flow over the `OpenAI`-compatible wire format).
#[tokio::test]
async fn llm_summarize_persists_the_extracted_memory() {
    let server = MockServer::start().await;
    let title = "Cursor paging";
    let description = "Cursor pagination keeps the ordering stable across pages";
    let content = "Page a large listing with a stable cursor instead of offsets.";
    let reply = json!([{
        "title": title,
        "description": description,
        "content": content,
        "source_type": "success",
    }])
    .to_string();
    mount_reply(&server, reply).await;

    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::ReasoningBank),
        MemoryBackend::Llm(Arc::new(
            OpenAiChatCompletion::new(ChatCompletionConfig::new(server.uri(), "test-model"))
                .expect("the HTTP adapter builds"),
        )),
        Arc::new(InMemoryMemoryStore::new()),
    );
    let trajectory = "The agent counted 12 playlists with cursor pagination.";
    let response = service
        .summarize(summarize_request(
            USER,
            json!([{"query": "list every playlist", "response": trajectory, "feedback": "helpful"}]),
        ))
        .await
        .expect("the model-driven summarize succeeds");
    assert_eq!(response.status, "success");
    let written = entries(&response.memories);
    assert_eq!(written.len(), 1);
    assert_eq!(written[0]["content"].as_str(), Some(content));
    assert_eq!(written[0]["title"].as_str(), Some(title));

    let loaded = service
        .load_memories(USER)
        .expect("the extraction persisted");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].content, content);
    assert_eq!(loaded[0].title.as_deref(), Some(title));
    assert_eq!(loaded[0].description.as_deref(), Some(description));

    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one completion call per summarize");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("the request body is JSON");
    let prompt = body["messages"][1]["content"]
        .as_str()
        .expect("the user prompt is text");
    assert!(
        prompt.contains(trajectory),
        "the prompt carries the trajectory: {prompt}"
    );
    assert!(
        prompt.contains("Query: list every playlist"),
        "the prompt carries the trajectory query: {prompt}"
    );
}

// Port of `tests/unit_tests/extensions/context_evolver/test_summary_flow.py`
// (fenced replies, which is the shape chat models actually emit).
#[tokio::test]
async fn fenced_summarize_reply_is_parsed_and_persisted() {
    let content = "Anchor a long listing on a cursor instead of an offset.";
    let payload = json!([{
        "title": "Cursor anchors",
        "description": "Stable ordering across pages",
        "content": content,
    }])
    .to_string();
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::ReasoningBank),
        MemoryBackend::Llm(Arc::new(ScriptedCompletion::new([format!(
            "```json\n{payload}\n```"
        )]))),
        Arc::new(InMemoryMemoryStore::new()),
    );
    let response = service
        .summarize(summarize_request(USER, helpful_batch()))
        .await
        .expect("a fenced reply is tolerated");
    assert_eq!(response.status, "success");
    assert_eq!(
        vec![entries(&response.memories)[0]["content"]
            .as_str()
            .expect("content")],
        [content.to_string()]
    );
    let loaded = service
        .load_memories(USER)
        .expect("the extraction persisted");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].content, content);
}

// Port of `tests/unit_tests/extensions/context_evolver/test_summary_flow.py`:
// `TOOLCHAIN_EVOLVING_MEMORY_SUMMARIZE_EXECUTION_ERROR`, a typed failure.
#[tokio::test]
async fn prose_summarize_reply_fails_naming_the_summarize_phase() {
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::ReasoningBank),
        MemoryBackend::Llm(Arc::new(ScriptedCompletion::new([
            "I could not find anything worth remembering.",
        ]))),
        Arc::new(InMemoryMemoryStore::new()),
    );
    let error = service
        .summarize(summarize_request(USER, helpful_batch()))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("summarize"),
        "the message names the phase: {}",
        error.message()
    );
    assert!(
        error.message().contains("reasoningbank"),
        "the message names the algorithm: {}",
        error.message()
    );
}

// Port of `retrieve/task/*/run.py`: the rerank step logs "Failed to parse
// rerank response, using original order" and keeps the pre-ranked candidates.
#[tokio::test]
async fn prose_retrieve_reply_degrades_to_the_pre_ranked_order() {
    let store = Arc::new(InMemoryMemoryStore::new());
    let completion: Arc<ScriptedCompletion> =
        Arc::new(ScriptedCompletion::new(["I cannot rank these candidates."]));
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Llm(completion.clone()),
        store.clone(),
    );
    let (cursor_id, _) = seed_pagination(service.as_ref(), &store).await;

    let response = service
        .retrieve(retrieve_request(USER, PAGINATION_QUERY))
        .await
        .expect("a broken ranking degrades instead of failing");
    assert_eq!(response.status, "success");
    assert_eq!(
        entry_ids(&response.retrieved_memory)
            .first()
            .map(String::as_str),
        Some(cursor_id.as_str()),
        "the pre-ranked lexical order survives"
    );
    assert!(
        response.memory_string.contains(PAGINATION_CURSOR),
        "{}",
        response.memory_string
    );
    assert_eq!(
        completion.call_count(),
        1,
        "the model was called before the fallback"
    );
}

// Port of `retrieve/task/*/run.py` (`RerankMemoryOp`: unknown ids in the reply
// are dropped, the rest of the ranking still applies).
#[tokio::test]
async fn hallucinated_ranked_id_is_ignored() {
    let store = Arc::new(InMemoryMemoryStore::new());
    let seeded = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Deterministic,
        store.clone(),
    );
    let (_, token_id) = seed_pagination(seeded.as_ref(), &store).await;

    let reply = json!({"ranked_ids": ["ghost-memory-id", token_id.clone()]}).to_string();
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Llm(Arc::new(ScriptedCompletion::new([reply]))),
        store,
    );
    let response = service
        .retrieve(retrieve_request(USER, PAGINATION_QUERY))
        .await
        .expect("a hallucinated id does not fail the retrieve");
    assert_eq!(response.status, "success");
    assert_eq!(
        entry_ids(&response.retrieved_memory),
        [token_id],
        "only the resolvable entry is kept"
    );
    assert!(response.memory_string.contains(PAGINATION_TOKEN));
    assert!(!response.memory_string.contains(PAGINATION_CURSOR));
}

// Port of `tests/unit_tests/extensions/context_evolver/test_retrieve_flow.py`
// (`TestAlgorithmNormalization`) and `test_summary_flow.py`: every Python
// spelling drives a working factory service across all four families.
#[tokio::test]
async fn factory_builds_a_working_service_for_every_python_algo_name() {
    let cases = [
        ("ACE", Algo::Ace, AlgoFamily::Ace),
        ("rb", Algo::ReasoningBank, AlgoFamily::ReasoningBank),
        ("ReMe", Algo::Reme, AlgoFamily::Reme),
        ("ref_con", Algo::Refcon, AlgoFamily::Reme),
        ("Div-Con", Algo::Divcon, AlgoFamily::Reme),
        ("cognition", Algo::Cognition, AlgoFamily::Cognition),
    ];
    for (name, expected, family) in cases {
        assert_eq!(
            normalize_algo_name(name).expect("the Python spelling normalizes"),
            expected,
            "name: {name}"
        );
        assert_eq!(expected.family(), family, "name: {name}");
        let service = create_task_memory_service(
            TaskMemoryConfig::new(expected),
            MemoryBackend::Deterministic,
            Arc::new(InMemoryMemoryStore::new()),
        );
        let user = format!("user-{name}");
        service
            .add_memory(
                &user,
                conditional_memory(
                    "Search a repository with ripgrep and a glob filter.",
                    "searching a repository",
                ),
            )
            .await
            .expect("the memory is persisted");
        let response = service
            .retrieve(retrieve_request(
                &user,
                "ripgrep glob filter repository search",
            ))
            .await
            .unwrap_or_else(|error| panic!("{name} retrieves: {error}"));
        assert_eq!(response.status, "success", "name: {name}");
        assert_ne!(response.memory_string, "", "name: {name}");
        let found = entries(&response.retrieved_memory);
        assert_eq!(found.len(), 1, "name: {name}");
        assert_eq!(
            found[0]["section"].as_str(),
            Some("general"),
            "name: {name}"
        );
    }

    let error = normalize_algo_name("remebank").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("remebank"),
        "the message keeps the caller's spelling: {}",
        error.message()
    );
}

// Port of `tests/unit_tests/extensions/context_evolver/test_summary_flow.py`
// (the deferred seam; a configuration that cannot run a pipeline fails loudly).
#[tokio::test]
async fn unsupported_service_keeps_the_deferred_contract() {
    for algo in ALL_ALGOS {
        let expected = format!(
            "context_evolver '{}' pipeline is deferred; inject a TaskMemoryService",
            algo.as_str()
        );
        let service = UnsupportedTaskMemoryService::new(algo);
        assert_eq!(service.algo(), algo);
        assert_deferred(&service, &expected).await;
    }

    let from_factory = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Unsupported,
        Arc::new(InMemoryMemoryStore::new()),
    );
    assert_deferred(
        from_factory.as_ref(),
        "context_evolver 'reme' pipeline is deferred; inject a TaskMemoryService",
    )
    .await;
}

// Port of `tests/unit_tests/extensions/context_evolver/test_summary_flow.py`:
// prompt → completion → parse → persist over the in-process double.
#[tokio::test]
async fn scripted_completion_records_the_prompts_and_persists_the_reply() {
    let store = Arc::new(InMemoryMemoryStore::new());
    let experience = "Loop over the playlists with a stable cursor until the next token is empty.";
    let completion: Arc<ScriptedCompletion> = Arc::new(ScriptedCompletion::new([
        json!([{"when_to_use": "when listing every playlist", "experience": experience}])
            .to_string(),
    ]));
    let service = create_task_memory_service(
        TaskMemoryConfig::new(Algo::Reme),
        MemoryBackend::Llm(completion.clone()),
        store,
    );
    let response = service
        .summarize(summarize_request(USER, helpful_batch()))
        .await
        .expect("the scripted summarize succeeds");
    assert_eq!(response.status, "success");
    assert_eq!(
        vec![entries(&response.memories)[0]["content"]
            .as_str()
            .expect("content")],
        [experience.to_string()]
    );
    assert_eq!(
        entries(&response.memories)[0]["when_to_use"].as_str(),
        Some("when listing every playlist")
    );

    let calls = completion.calls();
    assert_eq!(calls.len(), 1, "one prompt pair per summarize");
    assert!(
        calls[0].system.contains("expert AI analyst"),
        "the `ReMe` system prompt is sent: {}",
        calls[0].system
    );
    assert!(
        calls[0]
            .user
            .contains("# Original Query\ncount every playlist"),
        "{}",
        calls[0].user
    );
    assert!(
        calls[0]
            .user
            .contains("The agent looped with a stable cursor and counted 12 playlists."),
        "{}",
        calls[0].user
    );
    assert!(
        calls[0].user.contains("feedback: helpful"),
        "{}",
        calls[0].user
    );

    let loaded = service
        .load_memories(USER)
        .expect("the extraction persisted");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].content, experience);
    assert_eq!(completion.call_count(), 1);
}

// --- store-backed scoring ---------------------------------------------------
//
// Python counterpart: `extensions/context_evolver/retrieve/task/*/run.py` scores
// memories through `Milvus`, a path its own suite only runs with credentials;
// these tests pin the same seam offline over a kernel `VectorStore`.
/// The `{"memories": [...]}` `(id, score)` pairs of a retrieve response.
fn scored_entries(response: &Value) -> Vec<(String, f64)> {
    response["memories"]
        .as_array()
        .expect("memories")
        .iter()
        .map(|memory| {
            (
                memory["id"].as_str().expect("id").to_string(),
                memory["score"].as_f64().expect("score"),
            )
        })
        .collect()
}
/// Seeds the offline memories and returns an in-process deterministic service.
async fn seeded_service(store: Arc<dyn MemoryStore>) -> DeterministicTaskMemoryService {
    let service =
        DeterministicTaskMemoryService::new(Algo::Reme, TaskMemoryConfig::new(Algo::Reme), store)
            .with_embedding(Arc::new(HashingEmbedding::default()));
    for content in [
        RIPGREP_MEMORY,
        "Always page through results with a stable cursor.",
        "Cache the embedding model, not the vectors.",
    ] {
        service.add_memory(USER, memory(content)).await.unwrap();
    }
    service
}
/// Retrieves the offline memories for `RANK_QUERY` under `user`.
async fn rank(service: &dyn TaskMemoryService, user: &str) -> Value {
    service
        .retrieve(RetrieveRequest {
            user_id: user.into(),
            query: RANK_QUERY.into(),
        })
        .await
        .unwrap()
        .retrieved_memory
}
#[tokio::test]
async fn vector_store_backend_scores_like_the_in_process_index_and_survives_a_rebuild() {
    let memories: Arc<dyn MemoryStore> = Arc::new(InMemoryMemoryStore::new());
    let in_process = seeded_service(Arc::clone(&memories)).await;
    let expected = scored_entries(&rank(&in_process, USER).await);
    assert_eq!(expected.len(), 3);

    let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    let config =
        TaskMemoryConfig::new(Algo::Reme).with_vector_backend(VectorBackend::VectorStore {
            collection: "memories".into(),
        });
    let store_backed =
        DeterministicTaskMemoryService::new(Algo::Reme, config.clone(), Arc::clone(&memories))
            .with_embedding(Arc::new(HashingEmbedding::default()))
            .with_vector_store(vector_store.clone())
            .unwrap();
    let actual = scored_entries(&rank(&store_backed, USER).await);
    assert_eq!(
        actual.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        expected.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    for ((_, stored), (_, in_process)) in actual.iter().zip(&expected) {
        assert!(
            (stored - in_process).abs() < 1e-5,
            "store-backed {stored} vs in-process {in_process}"
        );
    }

    // The service was rebuilt over the same store, which now holds the nodes the
    // first one upserted; it ranks identically without recreating the collection.
    let rebuilt = DeterministicTaskMemoryService::new(Algo::Reme, config, memories)
        .with_embedding(Arc::new(HashingEmbedding::default()))
        .with_vector_store(vector_store.clone())
        .unwrap();
    assert_eq!(scored_entries(&rank(&rebuilt, USER).await), actual);
}
#[tokio::test]
async fn store_backed_index_reports_unknown_collections_and_dimension_mismatches() {
    let index = KernelVectorIndex::new(
        Arc::new(InMemoryVectorStore::new()) as Arc<dyn VectorStore>,
        "memories",
    )
    .unwrap();
    let error = index.search(&[1.0, 0.0], 1).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("memories"), "{}", error.message());

    index
        .upsert(vec![VectorNode::new("a", vec![1.0, 0.0])])
        .await
        .unwrap();
    let error = index
        .upsert(vec![VectorNode::new("b", vec![1.0, 0.0, 0.0])])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("3-dimension") && error.message().contains("2-dimension"),
        "{}",
        error.message()
    );
    assert_eq!(
        index.search(&[1.0], 1).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
}
#[tokio::test]
async fn default_configuration_keeps_the_in_process_index() {
    assert_eq!(
        TaskMemoryConfig::default().vector_backend,
        VectorBackend::InMemory
    );
    assert_eq!(
        VectorBackend::InMemory.collection(),
        DEFAULT_VECTOR_COLLECTION
    );
    let configured =
        TaskMemoryConfig::new(Algo::Reme).with_vector_backend(VectorBackend::VectorStore {
            collection: "memories".into(),
        });
    assert_eq!(configured.vector_backend.collection(), "memories");
    let decoded: TaskMemoryConfig =
        serde_json::from_str(&serde_json::to_string(&configured).unwrap()).unwrap();
    assert_eq!(decoded, configured);

    // The default factory still scores in process memory, with no store in
    // sight, exactly as before the seam existed.
    let service = seeded_service(Arc::new(InMemoryMemoryStore::new())).await;
    assert_eq!(scored_entries(&rank(&service, USER).await).len(), 3);
}
