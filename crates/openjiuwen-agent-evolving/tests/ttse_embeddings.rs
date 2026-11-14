//! Public-API scenarios for TTSE embedding retrieval and model-driven dreaming.
//!
//! The vectors come from the kernel's deterministic `HashEmbedding` and the one
//! model path is a `ScriptedModel`, so everything here runs offline. The index
//! files live in `tempfile` directories.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use openjiuwen_agent_evolving::{
    consult_async, Dreamer, EmbeddingIndex, RetrieveMode, TipCategory, TipRecord, TtseConfig,
    TtseEngine,
};
use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::retrieval::embedding::{Embedding, HashEmbedding};
use openjiuwen_test_support::ScriptedModel;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;

const DIM: usize = 64;
const RG: &str = "use rg instead of grep";
const OFFLINE: &str = "run the test suite offline";
const WORKSPACE: &str = "check the workspace path first";
const CANONICAL: &str = "When running tests: use cargo to run them offline";
const D1: &str = "always check the workspace path before running tests";
const D2: &str = "always check the workspace path before running the tests";
const D3: &str = "always check the workspace path before running all tests";
const UNRELATED: &str = "prefer rg over grep for speed";

fn embedder() -> Arc<HashEmbedding> {
    Arc::new(HashEmbedding::new(DIM))
}

fn config() -> TtseConfig {
    TtseConfig {
        consult_top_k: 3,
        consult_max_rules: 3,
        ..TtseConfig::default()
    }
}

fn dream_config() -> TtseConfig {
    TtseConfig {
        dream_prune_enabled: false,
        dream_merge_threshold: 0.72,
        dream_cluster_min_size: 2,
        dream_ttl_ticks: 90,
        consult_top_k: 3,
        consult_max_rules: 3,
        ..TtseConfig::default()
    }
}

fn engine() -> TtseEngine {
    TtseEngine::new(config()).expect("valid config")
}

/// Pushes tips straight into the bank, bypassing the induction dedupe.
fn seed(engine: &mut TtseEngine, rules: &[&str]) -> Vec<String> {
    let mut ids = Vec::with_capacity(rules.len());
    for rule in rules {
        let tip = TipRecord::new(*rule, TipCategory::Tool).expect("valid rule");
        ids.push(tip.tip_id.clone());
        engine.bank_mut().tips.push(tip);
    }
    ids
}

fn scripted(answer: &str) -> Arc<dyn Model> {
    Arc::new(ScriptedModel::text(answer))
}

fn merge_answer() -> String {
    [
        "THINKING: they all say the same thing",
        "REASON: paraphrases",
        "VERDICT: MERGE",
        &format!("CANONICAL: {CANONICAL}"),
        "MERGE_INDICES:",
        "KEEP_INDICES:",
    ]
    .join("\n")
}

fn keep_distinct_answer() -> String {
    [
        "THINKING: different conditions",
        "REASON: the cases differ",
        "VERDICT: KEEP_DISTINCT",
        "CANONICAL:",
    ]
    .join("\n")
}

fn sibling_index(bank: &Path) -> PathBuf {
    let mut sibling = bank.as_os_str().to_owned();
    sibling.push(".embedding.json");
    PathBuf::from(sibling)
}

#[tokio::test]
async fn cold_fill_embeds_every_missing_tip_and_is_idempotent() {
    let mut engine = engine().with_embedding(embedder());
    let ids = seed(&mut engine, &[RG, OFFLINE, WORKSPACE]);
    assert_eq!(engine.embedding_index().expect("index").len(), 0);

    assert_eq!(engine.sync_index().await.expect("cold fill"), 3);
    let index = engine.embedding_index().expect("index");
    assert_eq!(index.len(), 3);
    assert_eq!(index.dimension(), DIM);
    assert!(ids.iter().all(|id| index.contains(id)));
    assert_eq!(engine.sync_index().await.expect("second fill"), 0);
}

#[tokio::test]
async fn sync_reports_zero_without_an_embedder() {
    let mut engine = engine();
    seed(&mut engine, &[RG]);
    assert!(engine.embedding_index().is_none());
    assert_eq!(engine.sync_index().await.expect("no-op sync"), 0);
}

#[tokio::test]
async fn index_query_ranks_by_cosine_and_is_deterministic() {
    let mut engine = engine().with_embedding(embedder());
    seed(&mut engine, &[RG, OFFLINE, WORKSPACE]);
    engine.sync_index().await.expect("sync");
    let index = engine.embedding_index().expect("index");

    let first = index.query(RG, 3).await.expect("query");
    let second = index.query(RG, 3).await.expect("query");
    assert_eq!(first, second, "the ranking is deterministic");
    assert!(
        first.windows(2).all(|pair| pair[0].score >= pair[1].score),
        "descending by cosine: {first:?}"
    );
    let top = engine.bank().get(&first[0].tip_id).expect("known tip");
    assert_eq!(top.rule, RG);
}

#[tokio::test]
async fn consult_async_fuses_both_branches_for_an_exact_question() {
    let mut engine = engine().with_embedding(embedder());
    seed(&mut engine, &[RG, OFFLINE, WORKSPACE, UNRELATED]);
    engine.sync_index().await.expect("sync");

    let result = engine.consult_async(RG).await.expect("hybrid consult");
    assert_eq!(result.tips[0].rule, RG);
    assert_ne!(result.rendered, "");
    assert!(engine
        .embedding_index()
        .expect("index")
        .contains(&result.tips[0].tip_id));
}

#[tokio::test]
async fn retrieve_modes_still_answer_without_an_index() {
    let mut engine = engine();
    seed(&mut engine, &[RG, OFFLINE, WORKSPACE, UNRELATED]);
    let modes = [
        RetrieveMode::Hybrid,
        RetrieveMode::Embed,
        RetrieveMode::Lexical,
        RetrieveMode::Dump,
    ];
    for mode in modes {
        let result = engine
            .consult_with(RG, mode)
            .await
            .unwrap_or_else(|error| panic!("{mode:?} consult failed: {error}"));
        assert!(!result.tips.is_empty(), "{mode:?} returned nothing");
        assert!(result.tips.len() <= 3, "{mode:?} honours the limit");
        if mode == RetrieveMode::Dump {
            assert_eq!(result.tips.len(), 3, "the dump keeps the bank order");
        }
    }
    assert_eq!(engine.consult_count(), 4);
    let embed = engine
        .consult_with(RG, RetrieveMode::Embed)
        .await
        .expect("embed consult");
    assert_eq!(embed.tips[0].rule, RG, "the lexical fallback ranks it");
}

#[tokio::test]
async fn free_consult_async_is_the_same_seam_as_the_engine() {
    let mut engine = engine();
    seed(&mut engine, &[RG, OFFLINE]);
    let result = consult_async(
        engine.bank(),
        None,
        RG,
        engine.config(),
        RetrieveMode::Lexical,
    )
    .await
    .expect("free consult");
    assert_eq!(result.tips[0].rule, RG);
    assert_eq!(
        engine.consult_count(),
        0,
        "the free function has no counter"
    );
}

#[tokio::test]
async fn consult_async_counts_one_hit_per_served_tip() {
    let mut engine = engine().with_embedding(embedder());
    let ids = seed(
        &mut engine,
        &[RG, OFFLINE, WORKSPACE, UNRELATED, "keep the build green"],
    );
    engine.sync_index().await.expect("sync");
    let result = engine.consult_async(RG).await.expect("consult");

    assert_eq!(engine.consult_count(), 1);
    for tip in &result.tips {
        let stored = engine.bank().get(&tip.tip_id).expect("known tip");
        assert_eq!(stored.hits, 1);
        assert_eq!(stored.successes, 1);
    }
    let unused = ids
        .iter()
        .find(|id| !result.tips.iter().any(|tip| &tip.tip_id == *id))
        .expect("the bank is larger than the served limit");
    assert_eq!(engine.bank().get(unused).expect("known tip").hits, 0);
}

#[tokio::test]
async fn consult_async_is_a_no_op_when_injection_is_disabled() {
    let disabled = TtseConfig {
        inject_enabled: false,
        ..config()
    };
    disabled.validate().expect("valid config");
    let mut engine = TtseEngine::new(disabled)
        .expect("valid config")
        .with_embedding(embedder());
    seed(&mut engine, &[RG]);
    engine.sync_index().await.expect("sync");
    let result = engine.consult_async(RG).await.expect("disabled consult");
    assert_eq!(result.tips, Vec::new());
    assert_eq!(result.rendered, "");
    assert_eq!(engine.consult_count(), 0);
}

#[tokio::test]
async fn save_with_index_writes_the_sibling_index_and_round_trips() {
    let dir = TempDir::new().expect("temp dir");
    let bank_path = dir.path().join("state").join("tips.json");
    let mut engine = engine().with_embedding(embedder());
    let ids = seed(&mut engine, &[RG, OFFLINE, WORKSPACE]);
    engine.sync_index().await.expect("sync");
    engine.save_with_index(&bank_path).await.expect("save");

    let index_path = sibling_index(&bank_path);
    assert!(index_path.exists(), "the sibling index was written");
    let loaded = TtseEngine::load(&bank_path).await.expect("load");
    assert_eq!(loaded.bank(), engine.bank());
    let reopened = TtseEngine::load(&bank_path)
        .await
        .expect("load")
        .with_embedding_index(&index_path, embedder())
        .expect("reopen index");
    let index = reopened.embedding_index().expect("index");
    assert_eq!(index.len(), 3);
    assert!(ids.iter().all(|id| index.contains(id)));
    assert_eq!(index.dimension(), DIM);
}

#[tokio::test]
async fn a_bound_index_path_is_reused_when_saving() {
    let dir = TempDir::new().expect("temp dir");
    let index_path = dir.path().join("index.json");
    let bank_path = dir.path().join("tips.json");
    let mut engine = engine()
        .with_embedding_index(&index_path, embedder())
        .expect("open index");
    seed(&mut engine, &[RG]);
    engine.sync_index().await.expect("sync");
    engine.save_with_index(&bank_path).await.expect("save");

    assert!(index_path.exists(), "the bound path was written");
    assert!(
        !sibling_index(&bank_path).exists(),
        "no second index beside the bank"
    );
}

#[tokio::test]
async fn load_rejects_a_dimension_mismatch() {
    let dir = TempDir::new().expect("temp dir");
    let bank_path = dir.path().join("tips.json");
    let mut engine = engine().with_embedding(embedder());
    seed(&mut engine, &[RG]);
    engine.sync_index().await.expect("sync");
    engine.save_with_index(&bank_path).await.expect("save");

    let index_path = sibling_index(&bank_path);
    let small: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(32));
    let error = TtseEngine::load(&bank_path)
        .await
        .expect("load")
        .with_embedding_index(&index_path, small)
        .expect_err("a different dimension is rejected");
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    let message = error.to_string();
    assert!(message.contains("64-dimensional"), "{message}");
    assert!(message.contains("produces 32"), "{message}");
}

#[tokio::test]
async fn load_rejects_a_model_mismatch() {
    let dir = TempDir::new().expect("temp dir");
    let bank_path = dir.path().join("tips.json");
    let mut engine = engine().with_embedding(embedder());
    seed(&mut engine, &[RG]);
    engine.sync_index().await.expect("sync");
    engine.save_with_index(&bank_path).await.expect("save");

    let index_path = sibling_index(&bank_path);
    let other: Arc<dyn Embedding> =
        Arc::new(HashEmbedding::new(DIM).with_model_name("other-embedder"));
    let error = TtseEngine::load(&bank_path)
        .await
        .expect("load")
        .with_embedding_index(&index_path, other)
        .expect_err("a different model is rejected");
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    assert!(error.to_string().contains("other-embedder"), "{error}");
}

#[tokio::test]
async fn a_missing_index_file_starts_empty() {
    let dir = TempDir::new().expect("temp dir");
    let index_path = dir.path().join("absent.json");
    let engine = engine()
        .with_embedding_index(&index_path, embedder())
        .expect("a missing file opens an empty index");
    let index = engine.embedding_index().expect("index");
    assert!(index.is_empty());
    assert_eq!(index.dimension(), DIM, "the hint fixes the dimension");
    assert_eq!(index.path(), Some(index_path.as_path()));
}

#[tokio::test]
async fn remove_and_resync_drop_vectors_for_deleted_tips() {
    let mut engine = engine().with_embedding(embedder());
    let ids = seed(&mut engine, &[RG, OFFLINE]);
    engine.sync_index().await.expect("sync");
    assert_eq!(engine.embedding_index().expect("index").len(), 2);

    engine.bank_mut().remove(&ids[0]).expect("known tip");
    assert_eq!(engine.sync_index().await.expect("resync"), 0);
    let index = engine.embedding_index().expect("index");
    assert_eq!(index.len(), 1);
    assert!(!index.contains(&ids[0]));
    assert!(index.contains(&ids[1]));
}

#[tokio::test]
async fn dream_async_merges_a_cluster_into_the_canonical_rule() {
    let model = Arc::new(ScriptedModel::text(merge_answer()));
    let mut engine = TtseEngine::new(dream_config())
        .expect("valid config")
        .with_model(model.clone(), ModelRequestConfig::default())
        .with_embedding(embedder());
    seed(&mut engine, &[D1, D2, D3]);
    engine.sync_index().await.expect("sync");

    let report = engine.dream_async().await.expect("dream");
    assert_eq!((report.merged, report.promoted), (2, 1));
    assert_eq!(engine.bank().len(), 1, "bank={:?}", engine.bank().tips());
    let surviving = engine.bank().tips()[0].clone();
    assert_eq!(surviving.rule, CANONICAL);
    assert_eq!(surviving.category, TipCategory::Tool);
    assert_eq!(model.call_count(), 2, "one merge call, one purge call");
    let index = engine.embedding_index().expect("index");
    assert_eq!(index.len(), 1, "the merged-away vectors are gone");
    assert!(index.contains(&surviving.tip_id));
}

#[tokio::test]
async fn dream_async_keep_distinct_leaves_the_bank_untouched() {
    let mut engine = TtseEngine::new(dream_config())
        .expect("valid config")
        .with_model(
            scripted(&keep_distinct_answer()),
            ModelRequestConfig::default(),
        )
        .with_embedding(embedder());
    seed(&mut engine, &[D1, D2]);
    let before = engine.bank().clone();

    let report = engine.dream_async().await.expect("dream");
    assert_eq!((report.merged, report.promoted, report.pruned), (0, 0, 0));
    assert_eq!(engine.bank(), &before);
}

#[tokio::test]
async fn malformed_dream_output_keeps_every_tip() {
    let mut engine = TtseEngine::new(dream_config())
        .expect("valid config")
        .with_model(
            scripted("I would merge those two"),
            ModelRequestConfig::default(),
        )
        .with_embedding(embedder());
    seed(&mut engine, &[D1, D2, UNRELATED]);
    let before = engine.bank().clone();

    let report = engine.dream_async().await.expect("dream");
    assert_eq!((report.merged, report.promoted), (0, 0));
    assert_eq!(engine.bank(), &before);
    assert_eq!(engine.embedding_index().expect("index").len(), 3);
}

#[tokio::test]
async fn a_without_model_dream_uses_the_deterministic_pass() {
    let mut engine = TtseEngine::new(dream_config())
        .expect("valid config")
        .with_embedding(embedder());
    seed(&mut engine, &[D1, D2]);
    let before = engine.bank().clone();
    let report = engine.dream_async().await.expect("dream");
    assert_eq!(
        (report.merged, report.promoted),
        (1, 1),
        "the deterministic pass still clusters"
    );
    assert_eq!(engine.bank().len(), before.len() - 1);
}

#[test]
fn dreamer_is_reachable_from_the_crate_root() {
    let dreamer = Dreamer::new(dream_config());
    assert!(!dreamer.has_model());
    assert!(dreamer.max_llm_merges() > 0);
    let index = EmbeddingIndex::new(embedder());
    assert!(index.is_empty());
    assert_eq!(RetrieveMode::default(), RetrieveMode::Hybrid);
    assert_eq!(RetrieveMode::Lexical.as_str(), "lexical");
}
