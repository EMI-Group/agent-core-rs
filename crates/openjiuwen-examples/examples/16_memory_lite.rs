//! `16_memory_lite` — lite coding memory, engine-backed and local (`MEMORY.md` + FTS5).
//!
//! Teaches both paths of `openjiuwen::core::memory::lite`, the Rust port of Python
//! `openjiuwen/memory/lite/` and its `MemoryIndexManager`: `CodingMemory` over an
//! in-memory `MemoryEngine` (typed facts, keyword recall, name-keyed dedupe/merge,
//! `render_index`, `citation`) and `LocalCodingMemory`, where a human-editable
//! `MEMORY.md` is the source of truth and a derived SQLite FTS5 index under `.memory/`
//! backs full-text search. Scope isolation closes the engine half.
//! Everything is offline and deterministic: in-memory stores, a `tempfile` workspace,
//! no embedding model and no model call at all.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use anyhow::{ensure, Context, Result};
use openjiuwen_memory::config::{CodingMemoryConfig, MemoryEngineConfig};
use openjiuwen_memory::engine::MemoryEngine;
use openjiuwen_memory::lite::fts::FtsIndex;
use openjiuwen_memory::lite::local::{LocalCodingMemory, INDEX_FILE, STATE_DIR};
use openjiuwen_memory::lite::{CodingFact, CodingFactType, CodingMemory};
use openjiuwen_memory::record::{AddOptions, MemoryScope, RecallHit, SearchMode, SearchOptions};
use openjiuwen_memory::store::MemoryStores;
use tempfile::TempDir;

use common::banner;

/// The user owning every fact of the tour.
const USER: &str = "alice";
/// The workspace owning every fact of the tour.
const WORKSPACE: &str = "openjiuwen-rs";
/// The project body seeded first; the merge appends the fourth step, which is also why
/// the citation below spans four lines.
const BUILD_STEPS: &str = "Step 1: cargo build --workspace --offline\n\
                           Step 2: cargo test --workspace --offline\n\
                           Step 3: cargo clippy --workspace --offline";
/// [`BUILD_STEPS`] plus the line the re-write adds.
const BUILD_STEPS_WITH_ROLLBACK: &str = "Step 1: cargo build --workspace --offline\n\
                                        Step 2: cargo test --workspace --offline\n\
                                        Step 3: cargo clippy --workspace --offline\n\
                                        Step 4: cargo test --workspace --offline --no-fail-fast";
/// The engine query; it matches the project and feedback facts and nothing else.
const QUERY: &str = "offline test";
/// The local query, shared with the FTS5 `MATCH`, so the two rankings line up.
const LOCAL_QUERY: &str = "offline";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("16 lite memory");
    banner::note(&format!(
        "run mode: {}; every mode prints the same story",
        mode.describe()
    ));

    let scope = MemoryScope::new(USER, WORKSPACE);
    engine_path(&scope).await?;
    let workspace = common::tempdir()?;
    local_path(&workspace, &scope).await?;

    banner::title("done");
    banner::answer(
        "engine facts were scoped, merged and rendered; the local store answered from MEMORY.md",
    );
    Ok(())
}

// --- engine-backed path ---------------------------------------------------

/// Runs the `CodingMemory` half: seed, merge, recall, cite, render, isolate.
async fn engine_path(scope: &MemoryScope) -> Result<()> {
    banner::title("engine-backed CodingMemory");
    let engine = MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory())
        .context("failed to build the in-memory memory engine")?;
    let coding = CodingMemory::new(engine);
    banner::step("built CodingMemory over an in-memory MemoryEngine (keyword recall)");

    // The record id is captured so the merge below can prove the upsert kept it.
    let mut build_id = None;
    for fact in facts(scope) {
        let name = fact.name.clone();
        let label = describe(&fact);
        let record = coding
            .write(fact, &AddOptions::default())
            .await
            .with_context(|| format!("failed to write the `{name}` fact"))?;
        if name == "project-build" {
            build_id = Some(record.id);
        }
        banner::step(&format!("wrote {label}"));
    }
    let first_id = build_id.context("the seeded `project-build` fact was never written")?;

    // A repeated name updates the stored record instead of appending a duplicate.
    let merged = coding
        .write(
            project_fact(scope, BUILD_STEPS_WITH_ROLLBACK),
            &AddOptions::default(),
        )
        .await
        .context("failed to re-write the `project-build` fact")?;
    ensure!(
        merged.id == first_id,
        "a repeated name must update the existing record"
    );
    let listed = coding
        .list(scope)
        .await
        .context("failed to list the facts")?;
    ensure!(listed.len() == 4, "the scope must still hold four facts");
    banner::step("re-wrote `project-build`: same record id, still 4 facts (name-keyed merge)");

    let matched = recall(&coding, scope).await?;
    let build = matched.first().context("the recall returned no hit")?;
    let feedback = matched.get(1).context("the recall returned a single hit")?;
    let build_fact =
        CodingFact::from_record(&build.record).context("a coding hit without a fact")?;
    let feedback_fact =
        CodingFact::from_record(&feedback.record).context("a coding hit without a fact")?;
    ensure!(build_fact.name == "project-build", "unexpected top hit");
    ensure!(
        feedback_fact.name == "feedback-tests",
        "unexpected second hit"
    );
    banner::step(&format!(
        "citation with a `path`: {}",
        coding.citation(build)
    ));
    banner::step(&format!(
        "citation without one: {}",
        coding.citation(feedback)
    ));

    let index = coding
        .render_index(scope)
        .await
        .context("failed to render the index")?;
    banner::step(&format!(
        "render_index ({} line(s), cap {} including the title):",
        index.lines().count(),
        coding.config().max_index_lines
    ));
    for line in index.lines() {
        banner::step(&format!("  {line}"));
    }
    scope_isolation(&coding, scope).await
}

/// Recalls the facts matching [`QUERY`] and prints their scores in rank order.
async fn recall(coding: &CodingMemory, scope: &MemoryScope) -> Result<Vec<RecallHit>> {
    let hits = coding
        .search(scope, QUERY, &loose_options())
        .await
        .context("failed to recall coding facts")?;
    // A zero keyword component means the fact shares no term with the query; the
    // threshold is 0.0, so the engine still returns those facts and the tour drops them.
    let matched: Vec<RecallHit> = hits.into_iter().filter(|hit| hit.keyword > 0.0).collect();
    ensure!(
        matched.len() == 2,
        "`{QUERY}` must match the project and feedback facts"
    );
    banner::step(&format!(
        "search `{QUERY}` -> {} of 4 fact(s) matched, best first",
        matched.len()
    ));
    for hit in &matched {
        let fact = CodingFact::from_record(&hit.record).context("a coding hit without a fact")?;
        banner::step(&format!(
            "  score {:.3} (keyword {:.3}) {}",
            hit.score,
            hit.keyword,
            describe(&fact)
        ));
    }
    Ok(matched)
}

/// Writes one fact under a second scope and proves neither scope sees the other.
async fn scope_isolation(coding: &CodingMemory, scope: &MemoryScope) -> Result<()> {
    banner::title("scope isolation");
    let other = MemoryScope::new("bob", WORKSPACE);
    let note = CodingFact::new(&other, "bob-notes", "Bob's own scratch notes.")
        .with_description("Bob's notes");
    coding
        .write(note, &AddOptions::default())
        .await
        .context("failed to write the second scope's fact")?;

    let mine = coding
        .list(scope)
        .await
        .context("failed to list alice's facts")?;
    let theirs = coding
        .list(&other)
        .await
        .context("failed to list bob's facts")?;
    ensure!(
        mine.len() == 4 && theirs.len() == 1,
        "each scope keeps its own facts"
    );
    let unrelated = coding.search(&other, QUERY, &loose_options()).await?;
    ensure!(
        unrelated.iter().all(|hit| hit.keyword == 0.0),
        "the query must not leak across scopes"
    );
    banner::step(&format!(
        "scope '{}': {} fact(s); scope '{}': {} fact(s), no hit for `{QUERY}`",
        scope.user_id,
        mine.len(),
        other.user_id,
        theirs.len()
    ));
    Ok(())
}

/// The four seeded facts, one of each [`CodingFactType`]; the local half reuses them.
fn facts(scope: &MemoryScope) -> Vec<CodingFact> {
    vec![
        CodingFact::new(
            scope,
            "user-style",
            "Prefers two-space indents and one statement per line.",
        )
        .with_type(CodingFactType::User)
        .with_description("Editor preferences")
        .with_tags(vec!["style".to_owned()]),
        CodingFact::new(
            scope,
            "feedback-tests",
            "Always run the test suite offline before pushing.",
        )
        .with_type(CodingFactType::Feedback)
        .with_description("Testing guidance")
        .with_tags(vec!["ci".to_owned(), "tests".to_owned()]),
        project_fact(scope, BUILD_STEPS),
        CodingFact::new(scope, "reference-protocol", "Protocol notes live upstream.")
            .with_type(CodingFactType::Reference)
            .with_description("Upstream protocol docs")
            .with_tags(vec!["docs".to_owned()])
            .with_path("docs/PROTOCOL.md"),
    ]
}

/// The sourced project fact; `content` differs between the seed and the merge.
fn project_fact(scope: &MemoryScope, content: &str) -> CodingFact {
    CodingFact::new(scope, "project-build", content)
        .with_type(CodingFactType::Project)
        .with_description("How to build and test offline")
        .with_tags(vec!["build".to_owned()])
        .with_path("docs/BUILD.md")
}

// --- local path -----------------------------------------------------------

/// Runs the `LocalCodingMemory` half against a throwaway workspace directory.
async fn local_path(workspace: &TempDir, scope: &MemoryScope) -> Result<()> {
    banner::title("local coding memory: MEMORY.md + derived FTS5 index");
    let store = open(workspace, scope).await?;
    banner::step(&format!(
        "opened a store in a temp workspace; MEMORY.md exists: {}, state dir `{STATE_DIR}/` exists: {}",
        store.memory_path().exists(),
        store.root().join(STATE_DIR).exists()
    ));
    for fact in facts(scope) {
        let label = describe(&fact);
        store
            .write(fact)
            .await
            .context("failed to write a local fact")?;
        banner::step(&format!("wrote {label}"));
    }

    // MEMORY.md is the source of truth: read it back, reopen the store, query again.
    let document =
        std::fs::read_to_string(store.memory_path()).context("failed to read MEMORY.md")?;
    ensure!(
        document.starts_with("---\nschema: openjiuwen-memory-lite"),
        "missing front matter"
    );
    for fact_type in CodingFactType::ALL {
        ensure!(
            document.contains(&format!("## {fact_type:?}")),
            "missing a fact section"
        );
    }
    banner::step(&format!(
        "read MEMORY.md back: {} line(s), schema front matter, one `## <Type>` section per type",
        document.lines().count()
    ));

    let reopened = open(workspace, scope).await?;
    let listed = reopened
        .list()
        .await
        .context("failed to list the reopened store")?;
    ensure!(listed.len() == 4, "the reopened store must hold four facts");
    banner::step("re-opened the store from the same directory: 4 facts came back from MEMORY.md");

    let hits = reopened
        .search(LOCAL_QUERY, &loose_options())
        .await
        .context("failed to search the reopened store")?;
    let matched: Vec<&RecallHit> = hits.iter().filter(|hit| hit.score > 0.0).collect();
    ensure!(
        matched.len() == 2,
        "`{LOCAL_QUERY}` must match the project and feedback facts"
    );
    banner::step(&format!(
        "search `{LOCAL_QUERY}` on the reopened store -> {} fact(s), best first",
        matched.len()
    ));
    for hit in &matched {
        let fact = CodingFact::from_record(&hit.record).context("a coding hit without a fact")?;
        banner::step(&format!("  score {:.3} {}", hit.score, describe(&fact)));
    }
    full_text_search(&store, LOCAL_QUERY).await
}

/// Opens (or reopens) the local store on `workspace` in `scope`.
async fn open(workspace: &TempDir, scope: &MemoryScope) -> Result<LocalCodingMemory> {
    LocalCodingMemory::open_in_scope(
        workspace.path(),
        CodingMemoryConfig::default(),
        scope.clone(),
    )
    .await
    .context("failed to open the local coding-memory store")
}

/// Reads the derived FTS5 database directly, which is what makes the index visible.
async fn full_text_search(store: &LocalCodingMemory, query: &str) -> Result<()> {
    let index = FtsIndex::open_path(&store.root().join(STATE_DIR).join(INDEX_FILE))
        .await
        .context("failed to open the derived FTS5 index")?;
    let rows = index
        .count()
        .await
        .context("failed to count the indexed rows")?;
    ensure!(rows == 4, "the derived index must hold four rows");
    banner::step(&format!(
        "derived FTS5 index at `{STATE_DIR}/{INDEX_FILE}`: {rows} row(s); MEMORY.md stays authoritative"
    ));

    let hits = index
        .search(query, 5)
        .await
        .context("failed to query the FTS5 index")?;
    let names: Vec<&str> = hits.iter().map(|hit| hit.name.as_str()).collect();
    ensure!(
        names == ["project-build", "feedback-tests"],
        "the FTS5 ranking must stay best-first"
    );
    banner::step(&format!(
        "FTS5 MATCH `{query}` -> {} hit(s), bm25 ascending (more negative is better):",
        hits.len()
    ));
    for (rank, hit) in hits.iter().enumerate() {
        let snippet = hit.snippet.split_whitespace().collect::<Vec<_>>().join(" ");
        banner::step(&format!(
            "  {}. {} (bm25 {:.3e}): {snippet}",
            rank + 1,
            hit.name,
            hit.bm25
        ));
    }
    Ok(())
}

// --- shared helpers -------------------------------------------------------

/// `SearchOptions::default` with the threshold removed, so every candidate is visible,
/// and keyword scoring requested explicitly: no embedding model is attached anywhere.
fn loose_options() -> SearchOptions {
    SearchOptions::default()
        .with_mode(SearchMode::Keyword)
        .with_threshold(0.0)
        .with_top_k(10)
}

/// One-line rendering of a fact: `[Project] name — description (`path`)`.
fn describe(fact: &CodingFact) -> String {
    let path = fact
        .path
        .as_deref()
        .map(|path| format!(" (`{path}`)"))
        .unwrap_or_default();
    format!(
        "[{:?}] {} — {}{path}",
        fact.fact_type, fact.name, fact.description
    )
}
