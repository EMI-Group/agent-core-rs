#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration coverage for the [`CodingMemory`](openjiuwen_memory::lite::CodingMemory)
//! facade.
//!
//! The `lite` submodules have their own unit tests for fact/record encoding, name
//! validation, and pure index rendering; this suite goes through the engine:
//! write/read/edit/delete/list/search, name-keyed upserts, index grouping and
//! truncation, citations, and scope isolation.
//!
//! No embedding model is attached anywhere, so recall uses the engine's keyword
//! (BM25) fallback and every test stays offline and deterministic.
use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_memory::config::{CodingMemoryConfig, MemoryEngineConfig};
use openjiuwen_memory::engine::MemoryEngine;
use openjiuwen_memory::lite::{CodingFact, CodingFactType, CodingMemory};
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryScope, MemoryType, NewMemory, SearchOptions,
};
use openjiuwen_memory::store::MemoryStores;

/// A fixed instant `offset` seconds after the shared test epoch.
fn ts(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("a valid timestamp")
}

/// The scope shared by the single-scope tests; each test builds its own engine, so
/// reusing the scope never couples tests.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// A fresh in-memory engine with the default configuration.
fn engine() -> Arc<MemoryEngine> {
    MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory()).expect("engine")
}

/// The coding-memory facade over `engine`.
fn coding(engine: &Arc<MemoryEngine>) -> CodingMemory {
    CodingMemory::new(Arc::clone(engine))
}

/// Search options that keep every candidate, so partially matching keyword-only text
/// is still recalled (the default `threshold = 0.3` favors semantic hits).
fn search_options() -> SearchOptions {
    SearchOptions {
        threshold: 0.0,
        ..SearchOptions::default()
    }
}

#[tokio::test]
async fn write_and_read_round_trip_every_fact_field() {
    let coding = coding(&engine());
    let scope = scope();
    let mut fact = CodingFact::new(&scope, "verify-build", "Run `cargo test` before pushing.")
        .with_description("Run the test suite")
        .with_type(CodingFactType::Feedback)
        .with_tags(vec!["build".to_owned(), "ci".to_owned()])
        .with_path("docs/BUILD.md");
    fact.updated_at = ts(10);

    let record = coding
        .write(fact.clone(), &AddOptions::default())
        .await
        .expect("write");
    assert_eq!(record.kind, MemoryType::Coding);
    assert_eq!(record.content, fact.content);
    assert_eq!(record.scope, scope);

    let stored = coding
        .read(&scope, "verify-build")
        .await
        .expect("read")
        .expect("the written fact");
    assert_eq!(stored, fact, "every fact field must survive the round-trip");

    assert!(
        coding
            .read(&scope, "missing")
            .await
            .expect("read missing")
            .is_none(),
        "an unknown name must read as None"
    );
}

#[tokio::test]
async fn write_upserts_by_name_and_preserves_the_record_id() {
    let engine = engine();
    let coding = coding(&engine);
    let scope = scope();

    let first = coding
        .write(
            CodingFact::new(&scope, "build-command", "cargo build").with_description("debug build"),
            &AddOptions::default(),
        )
        .await
        .expect("first write");
    let second = coding
        .write(
            CodingFact::new(&scope, "build-command", "cargo build --release")
                .with_description("release build"),
            &AddOptions::default(),
        )
        .await
        .expect("second write");
    assert_eq!(
        second.id, first.id,
        "an upsert must update the stored record, not create a new one"
    );

    let facts = coding.list(&scope).await.expect("list");
    assert_eq!(facts.len(), 1, "the same name must never create two facts");
    assert_eq!(facts[0].content, "cargo build --release");
    assert_eq!(facts[0].description, "release build");
    assert_eq!(facts[0].fact_type, CodingFactType::Project);

    let records = engine
        .list(&MemoryFilter::for_scope(&scope).kind(MemoryType::Coding))
        .await
        .expect("engine list");
    assert_eq!(
        records.len(),
        1,
        "the engine must still hold exactly one coding record for the scope"
    );
    assert_eq!(records[0].id, first.id);
}

#[tokio::test]
async fn edit_replaces_the_body_and_reports_missing_facts() {
    let coding = coding(&engine());
    let scope = scope();
    coding
        .write(
            CodingFact::new(&scope, "runbook", "original body").with_description("Ops runbook"),
            &AddOptions::default(),
        )
        .await
        .expect("write");

    let edited = coding
        .edit(&scope, "runbook", "updated body", &CallOptions::new())
        .await
        .expect("edit");
    assert_eq!(edited.content, "updated body");
    assert_eq!(
        edited.description, "Ops runbook",
        "an edit must preserve the descriptive fields"
    );

    let stored = coding
        .read(&scope, "runbook")
        .await
        .expect("read")
        .expect("the edited fact");
    assert_eq!(stored, edited);

    let error = coding
        .edit(&scope, "no-such-fact", "body", &CallOptions::new())
        .await
        .expect_err("editing a missing fact must fail");
    assert!(
        matches!(error, Error::NotFound(_)),
        "expected NotFound, got {error:?}"
    );
}

#[tokio::test]
async fn delete_removes_a_fact_once() {
    let coding = coding(&engine());
    let scope = scope();
    coding
        .write(
            CodingFact::new(&scope, "stale-note", "This guidance is obsolete"),
            &AddOptions::default(),
        )
        .await
        .expect("write");

    assert!(
        coding
            .delete(&scope, "stale-note")
            .await
            .expect("first delete"),
        "the first delete removes the fact"
    );
    assert!(
        !coding
            .delete(&scope, "stale-note")
            .await
            .expect("second delete"),
        "a repeated delete reports that nothing was removed"
    );
    assert!(coding
        .read(&scope, "stale-note")
        .await
        .expect("read")
        .is_none());
    assert_eq!(
        coding.list(&scope).await.expect("list"),
        Vec::<CodingFact>::new(),
        "the deleted fact must not remain listed"
    );
}

#[tokio::test]
async fn list_orders_by_updated_at_descending_then_name_ascending() {
    let coding = coding(&engine());
    let scope = scope();
    for (name, offset, body) in [
        ("mid", 5, "middle fact body"),
        ("zeta", 20, "zulu fact body"),
        ("alpha", 20, "alpha fact body"),
    ] {
        let mut fact = CodingFact::new(&scope, name, body);
        fact.updated_at = ts(offset);
        coding
            .write(fact, &AddOptions::default())
            .await
            .expect("write");
    }

    let facts = coding.list(&scope).await.expect("list");
    let names: Vec<&str> = facts.iter().map(|fact| fact.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["alpha", "zeta", "mid"],
        "newest first, ties broken by name ascending"
    );
}

#[tokio::test]
async fn search_returns_only_coding_records() {
    let engine = engine();
    let coding = coding(&engine);
    let scope = scope();
    engine
        .add(
            &scope,
            NewMemory::semantic("Deployment notes describe the release branch handoff"),
            &AddOptions::default(),
        )
        .await
        .expect("semantic add");
    coding
        .write(
            CodingFact::new(
                &scope,
                "deploy-pipeline",
                "The deployment pipeline runs cargo and the nightly toolchain",
            ),
            &AddOptions::default(),
        )
        .await
        .expect("first fact");
    coding
        .write(
            CodingFact::new(
                &scope,
                "deploy-rollback",
                "Deployment rollback steps live in the runbook",
            ),
            &AddOptions::default(),
        )
        .await
        .expect("second fact");

    let hits = coding
        .search(&scope, "deployment pipeline", &search_options())
        .await
        .expect("search");
    assert!(
        !hits.is_empty(),
        "the coding facts must be recalled by the keyword path"
    );
    assert_eq!(
        hits.len(),
        2,
        "the overlapping semantic record must be filtered out"
    );
    for hit in &hits {
        assert_eq!(
            hit.record.kind,
            MemoryType::Coding,
            "coding search must never return a non-coding record"
        );
    }
}

#[tokio::test]
async fn facts_are_isolated_between_scopes_and_users() {
    let coding = coding(&engine());
    let own_scope = MemoryScope::new("alice", "ws1");
    let other_workspace = MemoryScope::new("alice", "ws2");
    let other_user = MemoryScope::new("bob", "ws1");
    coding
        .write(
            CodingFact::new(&own_scope, "private", "Only the ws1 scope may see this"),
            &AddOptions::default(),
        )
        .await
        .expect("write");

    assert!(coding
        .read(&own_scope, "private")
        .await
        .expect("read own")
        .is_some());
    for foreign in [&other_workspace, &other_user] {
        assert!(
            coding
                .read(foreign, "private")
                .await
                .expect("read foreign")
                .is_none(),
            "a fact must not leak into another scope or user ({foreign:?})"
        );
        assert_eq!(
            coding.list(foreign).await.expect("list foreign"),
            Vec::<CodingFact>::new(),
            "the listing must be empty for another scope or user ({foreign:?})"
        );
        assert_eq!(
            coding.render_index(foreign).await.expect("foreign index"),
            "# Memory\n",
            "an empty scope renders the title alone"
        );
    }
}

#[tokio::test]
async fn render_index_is_capped_including_the_title() {
    let config = MemoryEngineConfig {
        coding: CodingMemoryConfig {
            max_index_lines: 4,
            ..CodingMemoryConfig::default()
        },
        ..MemoryEngineConfig::default()
    };
    let engine = MemoryEngine::new(config, MemoryStores::in_memory()).expect("engine");
    assert_eq!(coding(&engine).config().max_index_lines, 4);
    let coding = coding(&engine);
    let scope = scope();
    for fact in [
        CodingFact::new(
            &scope,
            "user-style",
            "Prefer concise commit messages and small diffs",
        )
        .with_type(CodingFactType::User)
        .with_description("Style preferences"),
        CodingFact::new(
            &scope,
            "feedback-tests",
            "Always run the full test suite before committing",
        )
        .with_type(CodingFactType::Feedback)
        .with_description("Testing guidance"),
        CodingFact::new(
            &scope,
            "project-layout",
            "Workspace crates live under the crates directory",
        )
        .with_type(CodingFactType::Project)
        .with_description("Repo layout"),
        CodingFact::new(
            &scope,
            "project-release",
            "Releases are tagged from the release branch",
        )
        .with_type(CodingFactType::Project)
        .with_description("Release process"),
        CodingFact::new(
            &scope,
            "reference-pointer",
            "Upstream protocol reference lives in the docs directory",
        )
        .with_type(CodingFactType::Reference)
        .with_description("External pointer"),
    ] {
        coding
            .write(fact, &AddOptions::default())
            .await
            .expect("write");
    }

    let index = coding.render_index(&scope).await.expect("index");
    let lines: Vec<&str> = index.lines().collect();
    assert_eq!(
        lines.len(),
        4,
        "the cap counts the title line, got {index:?}"
    );
    assert!(index.starts_with("# Memory\n"), "got {index:?}");
    assert!(
        !index.contains("reference-pointer"),
        "the truncated fact must not appear, got {index:?}"
    );
    assert!(
        index.contains("project-layout"),
        "the first project fact must still be rendered, got {index:?}"
    );
}

#[tokio::test]
async fn render_index_groups_facts_by_type_and_shows_paths() {
    let coding = coding(&engine());
    let scope = scope();
    for fact in [
        CodingFact::new(&scope, "beta", "Beta project body")
            .with_type(CodingFactType::Project)
            .with_description("second project"),
        CodingFact::new(&scope, "gamma", "Gamma user body")
            .with_type(CodingFactType::User)
            .with_description("the user"),
        CodingFact::new(&scope, "alpha", "Alpha project body")
            .with_type(CodingFactType::Project)
            .with_description("first project")
            .with_path("docs/ALPHA.md"),
    ] {
        coding
            .write(fact, &AddOptions::default())
            .await
            .expect("write");
    }

    assert_eq!(
        coding.render_index(&scope).await.expect("index"),
        concat!(
            "# Memory\n",
            "- [User] gamma — the user\n",
            "- [Project] alpha — first project (`docs/ALPHA.md`)\n",
            "- [Project] beta — second project\n",
        )
    );
}

#[tokio::test]
async fn citation_uses_the_path_and_line_count_or_the_fact_name() {
    let coding = coding(&engine());

    let sourced_scope = scope();
    let mut sourced = CodingFact::new(
        &sourced_scope,
        "build-script",
        "step one\nstep two\nstep three\n",
    )
    .with_description("How to build")
    .with_path("scripts/build.sh");
    sourced.updated_at = ts(0);
    coding
        .write(sourced, &AddOptions::default())
        .await
        .expect("write sourced");
    let hits = coding
        .search(&sourced_scope, "step one build script", &search_options())
        .await
        .expect("search sourced");
    let hit = hits.first().expect("the sourced fact must be recalled");
    assert_eq!(
        coding.citation(hit),
        "scripts/build.sh#L1-L4",
        "four content lines including the trailing newline"
    );

    let plain_scope = MemoryScope::new("alice", "plain");
    coding
        .write(
            CodingFact::new(
                &plain_scope,
                "k8s-manifests",
                "Kubernetes manifests live under the k8s directory",
            ),
            &AddOptions::default(),
        )
        .await
        .expect("write plain");
    let hits = coding
        .search(&plain_scope, "Kubernetes manifests", &search_options())
        .await
        .expect("search plain");
    let hit = hits.first().expect("the plain fact must be recalled");
    assert_eq!(coding.citation(hit), "coding:k8s-manifests");
}
