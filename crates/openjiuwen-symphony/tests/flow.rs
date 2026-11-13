//! Behaviour tests for flow distillation and reusable `SkillPack` artifacts.
//!
//! Everything is offline and deterministic: temporary roots, scripted completion
//! models, and no clock-dependent assertions (timestamps are never compared).
//! The tests cover trace normalization, evidence ingestion, grouping, versioning,
//! persistence, packing/rendering/loading, the seven review checks, installation
//! preparation, and the model-narrative fallback.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    clippy::float_cmp,
    clippy::assert_is_empty,
    clippy::needless_pass_by_value
)]

use std::fs;
use std::sync::Arc;

use openjiuwen_retrieval::StaticCompletionModel;
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::flow::{
    FlowEngine, FlowTrace, ReviewResult, Reviewer, SkillPack, TraceOutcome, CHECK_FAIL,
    VERDICT_APPROVED, VERDICT_NEEDS_HUMAN_REVIEW, VERDICT_REJECTED,
};
use serde_json::{json, Value};
use tempfile::TempDir;

/// Builds one declared capability port.
fn port(name: &str, io_type: &str, required: bool, description: &str) -> Value {
    json!({
        "name": name,
        "type": io_type,
        "required": required,
        "description": description,
    })
}

/// Builds one capability node payload.
fn node(description: &str, inputs: Value, outputs: Value) -> Value {
    json!({
        "label": "capability",
        "metadata": {
            "capability_type": "skill",
            "skill_type": "skill",
            "version": "1.0.0",
            "description": description,
            "inputs": inputs,
            "outputs": outputs,
        },
    })
}

/// Builds one edge payload.
fn edge(source: &str, target: &str, success: bool) -> Value {
    json!({
        "source": source,
        "target": target,
        "relation": "can_feed",
        "metadata": {"success": success, "reason": "observed"},
    })
}

/// Builds a complete trace payload.
fn trace_payload(trace_id: &str, query: &str, outcome: &str, nodes: Value, edges: Value) -> Value {
    json!({
        "trace_id": trace_id,
        "query": query,
        "outcome": outcome,
        "graph": {
            "id": format!("graph-{trace_id}"),
            "type": "execution_graph",
            "label": "run",
            "directed": true,
            "nodes": nodes,
            "edges": edges,
        },
    })
}

/// The canonical `search → summarize` chain, with `skill:` prefixes on purpose.
fn chain_payload(trace_id: &str, query: &str) -> Value {
    trace_payload(
        trace_id,
        query,
        "success",
        json!({
            "skill:search": node(
                "search the corpus",
                json!([port("query", "string", true, "user query")]),
                json!([port("results", "array", true, "ranked hits")]),
            ),
            "skill:summarize": node(
                "summarize the hits",
                json!([port("results", "array", true, "ranked hits")]),
                json!([port("summary", "text", false, "final answer")]),
            ),
        }),
        json!([edge("skill:search", "skill:summarize", true)]),
    )
}

/// A second, independent `alpha → beta` chain.
fn alternate_payload(trace_id: &str, query: &str) -> Value {
    trace_payload(
        trace_id,
        query,
        "success",
        json!({
            "skill:alpha": node("gather facts", json!([]), json!([port("facts", "array", false, "")])),
            "skill:beta": node("compose answer", json!([port("facts", "array", false, "")]), json!([port("answer", "text", false, "")])),
        }),
        json!([edge("skill:alpha", "skill:beta", true)]),
    )
}

/// Ingests one payload and asserts it was stored.
async fn ingest(engine: &FlowEngine, payload: Value) {
    assert!(engine.ingest_value(payload).await.expect("ingest"));
}

/// Builds an engine with two ingested chain traces and one distillation run.
async fn distilled_engine(root: &std::path::Path) -> (FlowEngine, String) {
    let engine = FlowEngine::open(root).await.expect("open");
    ingest(&engine, chain_payload("t1", "find and summarize the docs")).await;
    ingest(&engine, chain_payload("t2", "summarize the release notes")).await;
    let report = engine.distill().await.expect("distill");
    assert_eq!(report.recipes_saved.len(), 1);
    let recipe = engine.recipes().await.first().cloned().expect("recipe");
    (engine, recipe.recipe_id)
}

#[test]
fn trace_normalization_strips_prefixes_and_drops_bad_edges() {
    let mut payload = chain_payload(" t1 ", "  find docs  ");
    payload["graph"]["edges"] = json!([
        edge("skill:search", "skill:summarize", true),
        edge("search", "summarize", false),
        edge("search", "ghost", true),
    ]);
    payload["graph"]["nodes"]["skill:unused"] = node("unused", json!([]), json!([]));
    payload["graph"]["nodes"][""] = node("blank id", json!([]), json!([]));

    let trace = FlowTrace::from_value(payload).expect("normalizes");
    assert_eq!(trace.trace_id, "t1");
    assert_eq!(trace.query, "find docs");
    assert_eq!(trace.outcome, TraceOutcome::Success);
    assert!(trace.graph.nodes.contains_key("search"));
    assert!(trace.graph.nodes.contains_key("summarize"));
    assert!(!trace.graph.nodes.contains_key("skill:search"));
    // Nodes survive normalization even when no successful edge touches them;
    // only blank ids are dropped.
    assert!(trace.graph.nodes.contains_key("unused"));
    assert_eq!(trace.graph.edges.len(), 1);
    assert_eq!(
        (
            trace.graph.edges[0].source.as_str(),
            trace.graph.edges[0].target.as_str()
        ),
        ("search", "summarize")
    );
    assert_eq!(trace.graph.edges[0].relation, "can_feed");
    assert!(trace.graph.edges[0].success);
}

#[test]
fn invalid_trace_payloads_are_rejected() {
    let cases = [
        json!(null),
        json!([]),
        json!({"graph": {}}),
        json!({"trace_id": "   ", "graph": {}}),
        json!({"trace_id": "t1"}),
        json!({"trace_id": "t1", "graph": "not-an-object"}),
    ];
    for case in cases {
        assert!(
            FlowTrace::from_value(case.clone()).is_err(),
            "expected rejection for {case}"
        );
    }
}

#[tokio::test]
async fn ingest_rejects_non_success_and_is_idempotent() {
    let dir = TempDir::new().unwrap();
    let engine = FlowEngine::open(dir.path()).await.unwrap();

    let failed = trace_payload(
        "failed",
        "q",
        "failed",
        json!({
            "a": node("a", json!([]), json!([port("x", "string", false, "")])),
            "b": node("b", json!([port("x", "string", false, "")]), json!([])),
        }),
        json!([edge("a", "b", true)]),
    );
    let trace = FlowTrace::from_value(failed).unwrap();
    assert_eq!(trace.outcome, TraceOutcome::Failed);
    let error = engine.ingest(trace).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let edgeless = trace_payload(
        "edgeless",
        "q",
        "success",
        json!({"a": node("a", json!([]), json!([]))}),
        json!([]),
    );
    let error = engine
        .ingest_value(edgeless)
        .await
        .expect_err("no successful edges");
    assert_eq!(error.kind(), ErrorKind::Validation);

    assert!(engine
        .ingest_value(chain_payload("t1", "q1"))
        .await
        .unwrap());
    assert!(!engine
        .ingest_value(chain_payload("t1", "q1"))
        .await
        .unwrap());
    assert_eq!(engine.evidence().await.len(), 1);
}

#[tokio::test]
async fn distill_groups_by_edge_signature_and_counts_quality() {
    let dir = TempDir::new().unwrap();
    let engine = FlowEngine::open(dir.path()).await.unwrap();
    ingest(&engine, chain_payload("t1", "q1")).await;
    ingest(&engine, chain_payload("t2", "q2")).await;
    ingest(&engine, alternate_payload("t3", "q3")).await;

    let report = engine.distill().await.unwrap();
    assert_eq!(report.evidence_total, 3);
    assert_eq!(report.groups_total, 2);
    assert!(report.groups_without_structure.is_empty());
    assert_eq!(report.recipes_saved.len(), 2);

    let recipes = engine.recipes().await;
    assert_eq!(recipes.len(), 2);
    let chain = recipes
        .iter()
        .find(|recipe| recipe.capability_ids() == vec!["search", "summarize"])
        .expect("chain recipe");
    assert_eq!(chain.quality.execution_count, 2);
    assert_eq!(chain.quality.success_count, 2);
    assert_eq!(chain.quality.failure_count, 0);
    assert_eq!(chain.quality.pack_success_rate, 1.0);
    assert_eq!(chain.quality.qualified_edge_count, 1);
    assert_eq!(chain.status, "active");
    assert_eq!(chain.grade, "verified");
    assert!(chain.execution_narrative.starts_with("执行过程："));
    assert_eq!(chain.combination_structure.edges.len(), 1);

    let alternate = recipes
        .iter()
        .find(|recipe| recipe.capability_ids() == vec!["alpha", "beta"])
        .expect("alternate recipe");
    assert_eq!(alternate.quality.execution_count, 1);

    let candidates = engine.candidates().await;
    assert_eq!(candidates.len(), 2);
    assert!(candidates
        .iter()
        .all(|candidate| candidate.success_rate == 1.0));
}

#[tokio::test]
async fn distill_reports_evidence_without_structure() {
    let dir = TempDir::new().unwrap();
    let payload = json!({
        "trace_id": "t-empty",
        "query": "nothing traced",
        "outcome": "success",
        "graph": {
            "id": "g",
            "type": "execution_graph",
            "label": "",
            "directed": true,
            "nodes": {"a": {"capability_id": "a", "capability_type": "skill"}},
            "edges": [],
        },
    });
    fs::write(
        dir.path().join("evidence.jsonl"),
        format!("{}\n", serde_json::to_string(&payload).unwrap()),
    )
    .unwrap();

    let engine = FlowEngine::open(dir.path()).await.unwrap();
    let report = engine.distill().await.unwrap();
    assert_eq!(report.evidence_total, 1);
    assert_eq!(report.groups_total, 0);
    assert_eq!(report.groups_without_structure, vec!["t-empty"]);
    assert!(engine.recipes().await.is_empty());
}

#[tokio::test]
async fn distillation_versions_on_content_change_and_replays_unchanged() {
    let dir = TempDir::new().unwrap();
    let (engine, recipe_id) = distilled_engine(dir.path()).await;

    let replay = engine.distill().await.unwrap();
    assert_eq!(replay.recipes_unchanged, vec![recipe_id.clone()]);
    assert!(replay.recipes_saved.is_empty());
    assert_eq!(engine.recipe(&recipe_id).await.unwrap().version, 1);

    ingest(&engine, chain_payload("t3", "a third distinct request")).await;
    let changed = engine.distill().await.unwrap();
    assert_eq!(changed.recipes_saved, vec![recipe_id.clone()]);
    assert_eq!(engine.recipe(&recipe_id).await.unwrap().version, 2);

    let stable = engine.distill().await.unwrap();
    assert_eq!(stable.recipes_unchanged, vec![recipe_id.clone()]);
    assert_eq!(engine.recipe(&recipe_id).await.unwrap().version, 2);

    let recipe_dir = dir.path().join("recipes").join(&recipe_id);
    assert!(recipe_dir.join("v1.json").is_file());
    assert!(recipe_dir.join("v2.json").is_file());
    let pointer: Value =
        serde_json::from_str(&fs::read_to_string(recipe_dir.join("current.json")).unwrap())
            .unwrap();
    assert_eq!(pointer["version"], 2);
    assert_eq!(pointer["recipe_id"], recipe_id);
}

#[tokio::test]
async fn persistence_rehydrates_evidence_and_current_recipe() {
    let dir = TempDir::new().unwrap();
    let (engine, recipe_id) = distilled_engine(dir.path()).await;
    let original = engine.recipe(&recipe_id).await.unwrap();
    drop(engine);

    let evidence_lines = fs::read_to_string(dir.path().join("evidence.jsonl")).unwrap();
    assert_eq!(evidence_lines.lines().count(), 2);

    let reopened = FlowEngine::open(dir.path()).await.unwrap();
    assert_eq!(reopened.evidence().await.len(), 2);
    assert_eq!(reopened.recipes().await.len(), 1);
    let restored = reopened.recipe(&recipe_id).await.expect("recipe survives");
    assert_eq!(restored.version, 1);
    assert_eq!(restored.content_identity(), original.content_identity());

    let replay = reopened.distill().await.unwrap();
    assert_eq!(replay.recipes_unchanged, vec![recipe_id]);
}

#[tokio::test]
async fn pack_renders_front_matter_and_workflow_graph() {
    let dir = TempDir::new().unwrap();
    let (engine, recipe_id) = distilled_engine(dir.path()).await;
    let pack = engine.pack(&recipe_id).await.unwrap();

    assert!(pack.package_id.starts_with("cap-"));
    assert_eq!(pack.package_id.len(), "cap-".len() + 12);
    assert_eq!(pack.target_kind, "skill");
    assert_eq!(pack.recipe_version, 1);
    assert_eq!(pack.step_ids(), vec!["search", "summarize"]);
    assert_eq!(pack.steps[0].depends_on, Vec::<String>::new());
    assert_eq!(pack.steps[1].depends_on, vec!["search"]);
    assert!(dir
        .path()
        .join("packages")
        .join(format!("{}.json", pack.package_id))
        .is_file());

    let rendered = pack.render();
    let skill_md = rendered.get("SKILL.md").expect("SKILL.md");
    let (front, body) = skill_md.split_once("\n---\n").expect("front matter");
    let yaml = front.strip_prefix("---\n").expect("leading delimiter");
    let front: serde_yaml_ng::Value = serde_yaml_ng::from_str(yaml).expect("front matter yaml");
    assert_eq!(front["name"].as_str(), Some(pack.name.as_str()));
    assert_eq!(front["kind"].as_str(), Some("skillpack"));
    assert_eq!(
        front["description"].as_str(),
        Some(pack.description.as_str())
    );
    let skills: Vec<&str> = front["skills"]
        .as_sequence()
        .expect("skills list")
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(skills, ["search", "summarize"]);

    for heading in [
        "## When to use",
        "## Required inputs",
        "## Included Skills",
        "## Execution Process",
        "## Failure handling",
        "## Final output",
        "## Workflow Graph",
    ] {
        assert!(body.contains(heading), "missing heading {heading}");
    }
    let graph_text = body
        .split_once("```json")
        .and_then(|(_, rest)| rest.split_once("```"))
        .map(|(graph, _)| graph.trim())
        .expect("workflow graph block");
    let graph: Value = serde_json::from_str(graph_text).expect("workflow graph json");
    assert_eq!(graph["graph"]["id"], format!("{}-workflow", pack.name));
    assert_eq!(graph["graph"]["type"], "skillpack_workflow");
    assert_eq!(graph["graph"]["directed"], true);
    assert_eq!(
        graph["graph"]["nodes"]["search"]["metadata"]["skill"],
        "search"
    );
    assert_eq!(graph["graph"]["edges"][0]["source"], "search");
    assert_eq!(graph["graph"]["edges"][0]["target"], "summarize");

    let dependencies = rendered
        .get("dependencies.yaml")
        .expect("dependencies.yaml");
    let manifest: serde_yaml_ng::Value = serde_yaml_ng::from_str(dependencies).expect("yaml");
    let capabilities = manifest["capabilities"]
        .as_sequence()
        .expect("capabilities");
    assert_eq!(capabilities.len(), 2);
    assert_eq!(capabilities[0]["id"].as_str(), Some("search"));
    assert_eq!(capabilities[1]["id"].as_str(), Some("summarize"));
}

#[tokio::test]
async fn pack_save_load_and_json_round_trip() {
    let dir = TempDir::new().unwrap();
    let (engine, recipe_id) = distilled_engine(dir.path()).await;
    let pack = engine.pack(&recipe_id).await.unwrap();

    let artifacts = TempDir::new().unwrap();
    pack.save(artifacts.path()).await.unwrap();
    assert!(artifacts.path().join("SKILL.md").is_file());
    assert!(artifacts.path().join("dependencies.yaml").is_file());
    let loaded = SkillPack::load(artifacts.path()).await.unwrap();
    assert_eq!(loaded, pack);
    assert_eq!(loaded.artifacts, pack.render());

    let json = pack.to_json().unwrap();
    let from_value: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(SkillPack::from_value(from_value).unwrap(), pack);

    assert_eq!(engine.package(&pack.package_id).await, Some(pack.clone()));
}

#[tokio::test]
async fn a_non_chain_recipe_is_not_packable() {
    let dir = TempDir::new().unwrap();
    let engine = FlowEngine::open(dir.path()).await.unwrap();
    let diamond = trace_payload(
        "t-diamond",
        "q",
        "success",
        json!({
            "a": node("a", json!([]), json!([port("x", "string", false, "")])),
            "b": node("b", json!([port("x", "string", false, "")]), json!([])),
            "c": node("c", json!([port("x", "string", false, "")]), json!([])),
            "d": node("d", json!([port("x", "string", false, "")]), json!([])),
        }),
        json!([
            edge("a", "b", true),
            edge("a", "c", true),
            edge("b", "d", true),
            edge("c", "d", true),
        ]),
    );
    ingest(&engine, diamond).await;
    let report = engine.distill().await.unwrap();
    assert_eq!(report.recipes_saved.len(), 1);
    let recipe_id = engine.recipes().await[0].recipe_id.clone();

    let error = engine.pack(&recipe_id).await.expect_err("not a chain");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("combination_structure"));
}

#[tokio::test]
async fn review_approves_a_valid_pack_and_flags_name_license_and_permissions() {
    let dir = TempDir::new().unwrap();
    let (engine, recipe_id) = distilled_engine(dir.path()).await;
    let pack = engine.pack(&recipe_id).await.unwrap();

    let valid = Reviewer::new().review(&pack);
    assert_eq!(valid.verdict, VERDICT_APPROVED);
    assert!(valid.failed_checks().is_empty());
    let names: Vec<&str> = valid
        .checks
        .iter()
        .map(|check| check.check.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "structure",
            "dependencies",
            "integrity",
            "meta_naming",
            "materials",
            "permissions",
            "license"
        ]
    );
    assert_eq!(valid.materials_hash, pack.materials_hash());
    assert!(valid.review_id.starts_with("review_"));
    assert_eq!(valid.review_id.len(), "review_".len() + 12);

    let mut invalid_name = pack.clone();
    invalid_name.name = "Not A Slug".to_string();
    invalid_name.recompute_identity();
    let flagged = Reviewer::new().review(&invalid_name);
    assert_eq!(flagged.verdict, VERDICT_NEEDS_HUMAN_REVIEW);
    assert_eq!(flagged.failed_checks().len(), 1);
    assert_eq!(flagged.failed_checks()[0].check, "meta_naming");

    let mut missing_license = pack.clone();
    missing_license.license.clear();
    missing_license.recompute_identity();
    let rejected_license = Reviewer::new().review(&missing_license);
    assert_eq!(rejected_license.verdict, VERDICT_REJECTED);
    assert_eq!(failed_names(&rejected_license), vec!["license"]);

    let mut forbidden = pack.clone();
    forbidden.permissions = vec!["shell".to_string(), "filesystem_write".to_string()];
    forbidden.recompute_identity();
    let rejected_permissions = Reviewer::new().review(&forbidden);
    assert_eq!(rejected_permissions.verdict, VERDICT_REJECTED);
    assert_eq!(failed_names(&rejected_permissions), vec!["permissions"]);
    assert!(rejected_permissions.failure_reasons()[0].contains("shell"));

    let mut tampered = pack.clone();
    tampered
        .artifacts
        .insert("SKILL.md".to_string(), "tampered".to_string());
    let rejected_integrity = Reviewer::new().review(&tampered);
    assert_eq!(rejected_integrity.verdict, VERDICT_REJECTED);
    assert_eq!(failed_names(&rejected_integrity), vec!["integrity"]);
}

/// Returns the names of failing checks.
fn failed_names(result: &ReviewResult) -> Vec<&str> {
    result
        .failed_checks()
        .iter()
        .map(|check| check.check.as_str())
        .collect()
}

#[tokio::test]
async fn install_writes_artifacts_and_rejects_stale_versions() {
    let dir = TempDir::new().unwrap();
    let (engine, recipe_id) = distilled_engine(dir.path()).await;
    let destination = TempDir::new().unwrap();

    let preparation = engine
        .review_and_prepare_install(&recipe_id, "skill", destination.path())
        .await
        .unwrap();
    assert_eq!(preparation.verdict, VERDICT_APPROVED);
    assert!(preparation.reasons.is_empty());
    assert_eq!(
        preparation.artifact_dir.as_deref(),
        Some(destination.path())
    );
    assert!(destination.path().join("SKILL.md").is_file());
    assert!(destination.path().join("dependencies.yaml").is_file());
    let package = preparation.package.expect("approved package");
    assert_eq!(SkillPack::load(destination.path()).await.unwrap(), package);
    assert_eq!(
        preparation.review.expect("review").verdict,
        VERDICT_APPROVED
    );

    let unsupported = engine
        .review_and_prepare_install(&recipe_id, "plugin", destination.path())
        .await
        .unwrap();
    assert_eq!(unsupported.verdict, VERDICT_REJECTED);
    assert!(unsupported.reasons[0].contains("unsupported target_kind"));

    let unknown = engine
        .review_and_prepare_install("recipe_000000000000", "skill", destination.path())
        .await
        .unwrap();
    assert_eq!(unknown.verdict, VERDICT_REJECTED);
    assert!(unknown.reasons[0].contains("recipe not found"));

    ingest(&engine, chain_payload("t3", "another distinct request")).await;
    let report = engine.distill().await.unwrap();
    assert_eq!(report.recipes_saved, vec![recipe_id.clone()]);
    assert_eq!(engine.recipe(&recipe_id).await.unwrap().version, 2);

    let stale = engine
        .prepare_install(&recipe_id, 1, "skill", destination.path())
        .await
        .unwrap();
    assert_eq!(stale.verdict, VERDICT_REJECTED);
    assert!(stale.reasons[0].starts_with("stale recipe version"));

    let current = engine
        .prepare_install(&recipe_id, 2, "skill", destination.path())
        .await
        .unwrap();
    assert_eq!(current.verdict, VERDICT_APPROVED);
    assert_eq!(current.package.expect("package").recipe_version, 2);
}

#[tokio::test]
async fn llm_narrative_is_used_and_malformed_replies_fall_back() {
    let dir = TempDir::new().unwrap();
    let model = StaticCompletionModel::single(
        r#"{"name": "custom-flow", "execution_narrative": "LLM narrative"}"#,
    );
    let engine = FlowEngine::open(dir.path())
        .await
        .unwrap()
        .with_model(Arc::new(model));
    ingest(&engine, chain_payload("t1", "q1")).await;
    ingest(&engine, chain_payload("t2", "q2")).await;
    let report = engine.distill().await.unwrap();
    assert!(report.narrative_fallbacks.is_empty());
    let recipe = engine.recipes().await.into_iter().next().unwrap();
    assert_eq!(recipe.name, "custom-flow");
    assert_eq!(recipe.execution_narrative, "LLM narrative");
    assert_eq!(
        recipe.provenance.get("narrative_source").unwrap(),
        &Value::String("llm".to_string())
    );

    let fallback_dir = TempDir::new().unwrap();
    let malformed = Arc::new(StaticCompletionModel::single("not json at all"));
    let fallback_engine = FlowEngine::open(fallback_dir.path())
        .await
        .unwrap()
        .with_model(malformed);
    ingest(&fallback_engine, chain_payload("t1", "q1")).await;
    let fallback_report = fallback_engine.distill().await.unwrap();
    let fallback_recipe = fallback_engine.recipes().await.into_iter().next().unwrap();
    assert_eq!(
        fallback_report.narrative_fallbacks,
        vec![fallback_recipe.recipe_id.clone()]
    );
    assert!(fallback_recipe
        .execution_narrative
        .starts_with("执行过程："));
    assert_eq!(
        fallback_recipe.provenance.get("narrative_source").unwrap(),
        &Value::String("template".to_string())
    );

    let error_dir = TempDir::new().unwrap();
    let failing = Arc::new(StaticCompletionModel::new(Vec::new()));
    let error_engine = FlowEngine::open(error_dir.path())
        .await
        .unwrap()
        .with_model(failing);
    ingest(&error_engine, chain_payload("t1", "q1")).await;
    let error_report = error_engine.distill().await.unwrap();
    assert_eq!(error_report.recipes_saved.len(), 1);
    assert_eq!(error_report.narrative_fallbacks.len(), 1);
    assert_eq!(
        error_engine.recipes().await[0]
            .provenance
            .get("narrative_source")
            .unwrap(),
        &Value::String("template".to_string())
    );
}

#[test]
fn review_constants_are_stable() {
    assert_eq!(CHECK_FAIL, "fail");
    assert_eq!(VERDICT_APPROVED, "approved");
    assert_eq!(VERDICT_REJECTED, "rejected");
    assert_eq!(VERDICT_NEEDS_HUMAN_REVIEW, "needs_human_review");
}
