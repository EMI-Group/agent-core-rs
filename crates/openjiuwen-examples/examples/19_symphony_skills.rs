//! `19_symphony_skills` — `SKILL.md` discovery, the fingerprinted inventory, and lexical
//! skill search.
//!
//! Ports the offline half of Python `openjiuwen/symphony/discovery/`: a directory of skill
//! folders becomes ordinary `CapabilityDescriptor`s, so a skill tree needs no plugin
//! protocol to reach the registry, the capability graph, or the planner. The example
//! walks that chain on files it writes into a `tempfile` directory: `parse_front_matter`
//! (fences, YAML, body), `SkillFolderScanner` (immediate children, then bounded
//! recursion, with the malformed skill reported as a diagnostic instead of failing the
//! scan), `SkillFs` (the fingerprinted `SkillInventory` plus a BM25 catalog, searched for
//! a phrase and for something absent), and finally `CapabilityRegistry` and
//! `CapabilityGraph` over the same descriptors.
//!
//! Offline and deterministic: temp files, no model, no network, and only names, tags,
//! scores, and digests reach stdout — never the temporary path.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements,
    // The narrative is one linear tour from the directory to the graph; splitting
    // `main` around its sections would scatter the story the reader follows.
    clippy::too_many_lines
)]
mod common;

use std::path::Path;

use anyhow::{ensure, Context, Result};
use common::banner;
use openjiuwen::symphony::discovery::{
    parse_front_matter, DiagnosticSeverity, FrontMatterStatus, SkillDiagnostic, SkillFolderScanner,
    SkillFs, SkillInventory, SkillScanConfig,
};
use openjiuwen::symphony::graph::{CapabilityEdge, CapabilityGraph};
use openjiuwen::symphony::model::CapabilityDescriptor;
use openjiuwen::symphony::registry::CapabilityRegistry;
use openjiuwen::symphony::search::SearchRequest;

/// The immediate-child skills, as `(folder, SKILL.md contents)`.
const SKILLS: [(&str, &str); 3] = [
    ("pdf-export", PDF_EXPORT),
    ("table-extract", TABLE_EXTRACT),
    ("web-search", WEB_SEARCH),
];
/// A skill one level below the root, so the immediate-child scan cannot see it.
const NESTED: (&str, &str) = ("toolkits/text-summarize", TEXT_SUMMARIZE);
/// A skill whose front matter never closes: an error diagnostic and no descriptor.
const BROKEN: (&str, &str) = (
    "broken-skill",
    "---\nname: Broken Skill\ndescription: this front matter never closes\n\
     tags: [documents]\n# the closing `---` fence is missing on purpose\n",
);
/// A document skill with typed I/O; the query below ranks it first.
const PDF_EXPORT: &str = "\
---
capability_id: pdf-export
name: PDF Export
description: Exports documents as PDF files with a stable page layout
capability_type: skills
tags: [documents, pdf, export]
version: 1.2.0
inputs: [{name: source, type: string, description: the document to render, required: true}]
outputs: [{name: artifact, type: file, description: the rendered PDF file}]
---

Renders a document into a PDF file with a stable page layout, so a result can leave the
workspace as a printable artifact.
";
/// The `can_feed` target: it consumes the PDF artifact the previous skill produces.
const TABLE_EXTRACT: &str = "\
---
name: Table Extract
description: Extracts tables from a PDF or a spreadsheet into rows
tags: [documents, tables]
inputs: [{name: artifact, type: file, description: a PDF or spreadsheet file, required: true}]
outputs: [{name: rows, type: array, description: one row per table line}]
---

Extracts tables from a PDF or a spreadsheet into rows, from the artifact a previous skill
produced.
";
/// A research skill: no document vocabulary, so the tagged query below leaves it out.
const WEB_SEARCH: &str = "\
---
name: Web Search
description: Searches the public web and returns ranked result snippets
capability_type: skill
tags: [research, search, web]
inputs: [{name: query, type: string, description: words to search for, required: true}]
outputs: [{name: results, type: array, description: ranked snippets with their urls}]
---

Searches the public web and returns ranked result snippets, each keeping its title and
its url.
";
/// The nested skill, reachable only through bounded recursion.
const TEXT_SUMMARIZE: &str = "\
---
name: Text Summarize
description: Condenses a long document into a short summary
tags: [documents, summary]
inputs: [{name: text, type: string, description: the text to condense, required: true}]
outputs: [{name: summary, type: string, description: three bullet points}]
---

Condenses a long document into a short summary that keeps the sentences answering who did
what. It sits in `toolkits/`, so an immediate-child scan never sees it.
";
/// The phrase that should rank `pdf-export` first.
const QUERY: &str = "export documents as pdf";
/// A phrase no skill mentions: the documented empty answer.
const MISSING_QUERY: &str = "kubernetes helm chart";
/// How many hits each query asks for.
const TOP_K: usize = 3;
/// Characters of an excerpt printed per hit, so stdout stays readable.
const EXCERPT_CHARS: usize = 72;
/// Characters of a digest printed; the full SHA-256 hex is 64.
const DIGEST_CHARS: usize = 12;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("19 symphony skills — discover, fingerprint, search");
    banner::note(&format!(
        "run mode {}: filesystem only, no model and no network; every mode prints the same story",
        mode.describe()
    ));

    // ------------------------------------------------------- write the tree --
    banner::step("write a skill tree: four skill folders plus one deliberately broken skill");
    let workspace = common::tempdir()?;
    let root = workspace.path().join("skills");
    for (folder, contents) in SKILLS {
        write_skill(&root, folder, contents).await?;
    }
    write_skill(&root, NESTED.0, NESTED.1).await?;
    write_skill(&root, BROKEN.0, BROKEN.1).await?;
    let folders: Vec<&str> = SKILLS
        .iter()
        .map(|(folder, _)| *folder)
        .chain([NESTED.0])
        .collect();
    banner::note(&format!(
        "skills/{{{}}}/SKILL.md, plus skills/{}/SKILL.md (front matter never closes)",
        folders.join(","),
        BROKEN.0
    ));

    // --------------------------------------------------------- front matter --
    banner::step("front matter: one document parses, one reports an unterminated fence");
    let parsed = parse_front_matter(&read_skill(&root, "pdf-export").await?);
    ensure!(
        parsed.fences_ok(),
        "the valid skill file should have closed front matter"
    );
    let name = parsed
        .get_str("name")
        .context("the valid skill's front matter should carry a name")?;
    banner::note(&format!(
        "status={} name={name:?} body starts {:?}",
        parsed.status.as_str(),
        shorten(&first_line(&parsed.body), EXCERPT_CHARS)
    ));
    let broken = parse_front_matter(&read_skill(&root, BROKEN.0).await?);
    ensure!(
        broken.status == FrontMatterStatus::Unterminated,
        "the broken skill file should report an unterminated fence"
    );
    banner::note(&format!(
        "status={} error={:?} — parsed, not rejected; the scan decides what to do",
        broken.status.as_str(),
        broken.error.as_deref().unwrap_or("none")
    ));

    // ---------------------------------------------------- immediate children --
    banner::step("scan immediate children: SkillFolderScanner's default config");
    let immediate = SkillFolderScanner::new(&root)
        .scan_result()
        .await
        .context("scanning the immediate children of the skill root")?;
    for descriptor in &immediate.capabilities {
        banner::step(&describe(descriptor));
    }
    banner::note(&format!(
        "{} capabilities discovered, source {:?}; error codes {:?}",
        immediate.capabilities.len(),
        immediate.source,
        immediate.error_codes()
    ));
    for diagnostic in &immediate.diagnostics {
        banner::step(&format!(
            "diagnostic {}",
            render_diagnostic(&root, diagnostic)
        ));
    }

    // ----------------------------------------------------- bounded recursion --
    banner::step(
        "bounded recursion: SkillScanConfig::recursive(max_depth) reaches the nested skill",
    );
    let shallow = SkillFolderScanner::with_config(&root, SkillScanConfig::recursive(Some(1)))
        .scan()
        .await
        .context("scanning the skill root one level deep")?;
    let config = SkillScanConfig::recursive(Some(2)).with_source("examples");
    let discovered = SkillFolderScanner::with_config(&root, config.clone())
        .scan_skills()
        .await
        .context("scanning the skill root two levels deep")?;
    banner::note(&format!(
        "max_depth=1 finds {} skills; max_depth={} finds {} — `{}` sits one level too deep for the first",
        shallow.len(),
        config.effective_max_depth(),
        discovered.len(),
        NESTED.0
    ));
    for skill in &discovered {
        banner::step(&describe(&skill.descriptor));
    }

    // ------------------------------------------------------------ inventory --
    banner::step(
        "inventory: SkillFs fingerprints the catalog; the digest survives a rescan and a reshuffle",
    );
    let catalog = SkillFs::open_with(config.clone(), &root)
        .await
        .context("opening the skill catalog")?;
    let inventory = catalog.inventory();
    ensure!(
        inventory.count == discovered.len(),
        "the catalog should hold every discovered skill, got {} of {}",
        inventory.count,
        discovered.len()
    );
    banner::note(&format!(
        "count={} worker ids {:?}",
        inventory.count,
        inventory.worker_ids()
    ));
    for record in catalog.records() {
        banner::step(&format!(
            "record {} name={:?} file={} src={} version={:?} hash={}",
            record.worker_id,
            record.name,
            record.skill_file,
            record.source,
            record.version,
            shorten(&record.content_hash, DIGEST_CHARS)
        ));
    }
    banner::note(&format!("fingerprint {}", inventory.fingerprint));
    let rescanned = SkillFs::open_with(config.clone(), &root)
        .await
        .context("reopening the skill catalog")?;
    ensure!(
        rescanned.inventory().fingerprint == inventory.fingerprint,
        "a second scan of the same tree must keep the inventory fingerprint"
    );
    banner::note(&format!(
        "rescan fingerprint {} (equal)",
        rescanned.inventory().fingerprint
    ));
    let mut shuffled = catalog.records().to_vec();
    shuffled.reverse();
    let rebuilt = SkillInventory::new(shuffled);
    ensure!(
        rebuilt.fingerprint == inventory.fingerprint,
        "record order must not change the inventory fingerprint"
    );
    banner::note(&format!(
        "reverse-order fingerprint {} (equal) — the digest sorts by worker id before hashing; it \
         covers `source_root` too, so it identifies this tree and not a copy elsewhere",
        rebuilt.fingerprint
    ));

    // ------------------------------------------------------------- search --
    banner::step("search: the installed catalog, ranked lexically with BM25");
    let hits = catalog
        .search(QUERY, TOP_K)
        .context("searching the skill catalog")?;
    ensure!(
        !hits.is_empty(),
        "the query `{QUERY}` should match at least one skill"
    );
    for (rank, hit) in hits.iter().enumerate() {
        banner::step(&format!(
            "{}. {} name={:?} score={:.3} terms={:?} excerpt={:?}",
            rank + 1,
            hit.worker_id,
            hit.name,
            hit.score,
            hit.matched_terms,
            shorten(&hit.excerpt, EXCERPT_CHARS)
        ));
    }
    let top = hits.first().context("the query should rank a first hit")?;
    ensure!(
        top.worker_id == "pdf-export",
        "`{QUERY}` should rank `pdf-export` first, got `{}`",
        top.worker_id
    );
    banner::note(
        "hits rank by BM25 over name, description, tags, and body; an empty excerpt means the term \
         matched a tag or the name only, since the excerpt is cut from the description or the body",
    );
    let nothing = catalog
        .search(MISSING_QUERY, TOP_K)
        .context("searching the skill catalog for an unrelated phrase")?;
    ensure!(
        nothing.is_empty(),
        "`{MISSING_QUERY}` should match no skill, got {} hits",
        nothing.len()
    );
    banner::note(&format!(
        "{MISSING_QUERY:?} matched {} skills — an empty result is the documented answer, not an error",
        nothing.len()
    ));

    // ------------------------------------------------ registry and graph --
    banner::step("registry: the same descriptors register without a plugin protocol, then search");
    // The catalog ranks a skill's markdown body; a scan descriptor carries it as plain
    // `semantic_content`, so registering the body keeps both rankings on the same text.
    let descriptors: Vec<CapabilityDescriptor> = discovered
        .iter()
        .map(|skill| {
            skill
                .descriptor
                .clone()
                .with_semantic_content(skill.body.clone())
        })
        .collect();
    let registry = CapabilityRegistry::new();
    let registered = registry
        .register_all(descriptors.clone())
        .await
        .context("registering the discovered skills")?;
    banner::note(&format!(
        "{} descriptors registered in one batch; revision {}, catalog holds {}",
        registered,
        registry.revision(),
        registry.len()
    ));
    let registry_hits = registry
        .search(&SearchRequest::for_text(QUERY))
        .await
        .context("searching the registry")?;
    for (rank, hit) in registry_hits.iter().take(TOP_K).enumerate() {
        banner::step(&format!(
            "{}. {} name={:?} score={:.3} terms={:?}",
            rank + 1,
            hit.descriptor.qualified_id(),
            hit.descriptor.name,
            hit.score,
            hit.matched_terms
        ));
    }
    banner::step("graph: the same descriptors, plus one can_feed edge between two of them");
    let mut graph = CapabilityGraph::from_descriptors("19-symphony-skills", &descriptors);
    graph
        .add_edge(CapabilityEdge::can_feed(
            "skill:pdf-export",
            "skill:table-extract",
        ))
        .context("adding the can_feed edge")?;
    banner::step(
        "skill:pdf-export --can_feed--> skill:table-extract  (a PDF artifact is a table-extract input)",
    );
    banner::note(&format!(
        "graph {:?}: nodes={} edges={}",
        graph.graph_id,
        graph.node_count(),
        graph.edge_count()
    ));

    // ------------------------------------------------------------- answer --
    banner::answer(&format!(
        "{} skills discovered from a plain directory tree, fingerprinted as {}, ranked by BM25, and registered as {} capabilities with one can_feed edge",
        inventory.count,
        shorten(&inventory.fingerprint, DIGEST_CHARS),
        registry.len()
    ));
    Ok(())
}

/// Writes one `SKILL.md` under `root`, creating the skill folder.
async fn write_skill(root: &Path, folder: &str, contents: &str) -> Result<()> {
    let directory = root.join(folder);
    tokio::fs::create_dir_all(&directory)
        .await
        .with_context(|| format!("creating the skill folder `{folder}`"))?;
    tokio::fs::write(directory.join("SKILL.md"), contents)
        .await
        .with_context(|| format!("writing the SKILL.md of `{folder}`"))?;
    Ok(())
}

/// Reads the `SKILL.md` of `folder` back out of the tree.
async fn read_skill(root: &Path, folder: &str) -> Result<String> {
    tokio::fs::read_to_string(root.join(folder).join("SKILL.md"))
        .await
        .with_context(|| format!("reading the SKILL.md of `{folder}`"))
}

/// One stable line per discovered descriptor: id, name, type, source, and sorted tags.
fn describe(descriptor: &CapabilityDescriptor) -> String {
    let mut tags = descriptor.tags.clone();
    tags.sort();
    format!(
        "{} name={:?} type={} src={} tags={tags:?}",
        descriptor.qualified_id(),
        descriptor.name,
        descriptor.capability_type,
        descriptor.source
    )
}

/// One stable line per diagnostic, with the path made relative to the scan root.
fn render_diagnostic(root: &Path, diagnostic: &SkillDiagnostic) -> String {
    let severity = match diagnostic.severity {
        DiagnosticSeverity::Warning => "warning",
        DiagnosticSeverity::Error => "error",
    };
    format!(
        "{severity} {} path={} message={:?}",
        diagnostic.code,
        relative_path(root, &diagnostic.path),
        diagnostic.message
    )
}

/// Strips the temporary scan root from a diagnostic path.
fn relative_path(root: &Path, path: &str) -> String {
    Path::new(path).strip_prefix(root).map_or_else(
        |_| "<outside the scan root>".to_string(),
        |relative| relative.to_string_lossy().replace('\\', "/"),
    )
}

/// The first non-blank line of `text`, so a body can be previewed in one line.
fn first_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_string()
}

/// Truncates `text` to `width` characters, marking a cut with `…`.
fn shorten(text: &str, width: usize) -> String {
    let mut characters = text.chars();
    let head: String = characters.by_ref().take(width).collect();
    if characters.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}
