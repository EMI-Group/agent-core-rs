//! `06_retrieval` — chunk, embed, store, and search, entirely offline.
//!
//! Teaches the retrieval layer as three separable pieces: a [`CharChunker`]
//! turning a document into overlapping text windows, a [`HashEmbedding`] turning
//! text into deterministic vectors, and an [`InMemoryVectorStore`] ranking those
//! vectors with cosine similarity. [`SimpleKnowledgeBase`] then wires the three
//! together into the load → chunk → embed → store → retrieve pipeline, and the
//! example finishes by using the store directly, one layer down.
//!
//! Python counterparts: `examples/retrieval/showcase_text_embedding.py` (embed
//! and retrieve) and `examples/store/es_vector_store_example.py` (the raw vector
//! store surface). Both need a provider or an Elasticsearch cluster; this port
//! uses the deterministic hash embedder and the in-memory store, so it runs with
//! no network and no credential.

#![allow(
    // The narrative is one linear tour: splitting `main` around its sections
    // would scatter the story the reader is meant to follow top to bottom.
    clippy::too_many_lines
)]

mod common;

use std::sync::Arc;

use anyhow::{bail, Context};
use openjiuwen::core::retrieval::{
    CharChunker, Chunker, Document, Embedding, HashEmbedding, InMemoryVectorStore,
    KnowledgeBaseConfig, RetrievalConfig, RetrievalResult, SimpleKnowledgeBase, VectorQuery,
    VectorRecord, VectorStore,
};

use common::banner;

/// Characters per chunk: small enough that the overview spans several windows.
const CHUNK_SIZE: usize = 200;
/// Characters shared by consecutive chunks, so no sentence is cut in half.
const CHUNK_OVERLAP: usize = 40;
/// Components per hash vector. The fake needs no model, only a width.
const EMBEDDING_DIMENSION: usize = 64;
/// How many hits each query asks for.
const TOP_K: usize = 3;
/// Character budget for a one-line preview, so chunks never flood stdout.
const PREVIEW_CHARS: usize = 64;

/// The document the chunking step walks through, paragraph by paragraph.
const OVERVIEW: &str = "Retrieval turns a corpus into answers. A loader reads a document, a chunker cuts it \
into windows, an embedding model maps each window to a vector, and a vector store ranks those vectors \
against the vector of a query.\n\nThe same pipeline is three separable layers. The embedder only knows \
text and vectors. The store only knows vectors, payloads, and cosine similarity. The knowledge base \
wires the stages together and applies the per-query configuration.\n\nEverything here runs in process: \
the hash embedder is deterministic and the store keeps its records in a map, so no server and no \
credential are involved.";

/// A document about the store layer, phrased to win its own query.
const STORE_DOC: &str = "A vector store keeps every record in memory here: an id, a text payload, and a \
vector. Search scores each record with cosine similarity, returns the best matches first, and needs no \
server.";

/// A document about the chunking layer, phrased to win its own query.
const CHUNK_DOC: &str = "Character chunking cuts a long document into overlapping windows. The overlap \
repeats the boundary characters, so a sentence stays whole across two chunks instead of being cut in \
half.";

/// A document about the embedding layer, phrased to win its own query.
const EMBED_DOC: &str =
    "Deterministic hash embeddings turn each whitespace token into signed components. \
The same text always yields the same vector, so offline search is reproducible.";

/// The text of a record added to the store directly, bypassing the knowledge base.
const RAW_RECORD: &str =
    "A record upserted straight into the store, with no document and no chunk \
behind it.";

/// The two queries, chosen so each ranks a different document first.
const QUERIES: [&str; 2] = [
    "overlapping windows keep a sentence whole across chunks",
    "which store scores cosine similarity in memory?",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;

    banner::title("06 retrieval — chunk, embed, store, search");
    banner::note(&format!(
        "retrieval mode {}: deterministic hash embeddings and an in-memory store, no network",
        mode.describe()
    ));

    // ---------------------------------------------------------------- chunk --
    banner::step("chunk: a char chunker cuts one document into overlapping windows");

    let overview = Document::with_id("kb-overview", OVERVIEW)
        .metadata_entry("source", "06_retrieval")
        .metadata_entry("topic", "pipeline");
    let chunker =
        CharChunker::new(CHUNK_SIZE, CHUNK_OVERLAP).context("building a character chunker")?;
    let chunks = chunker
        .chunk_documents(std::slice::from_ref(&overview))
        .context("chunking the overview document")?;
    if chunks.is_empty() {
        bail!(
            "the chunker produced no chunks for a {} character document",
            OVERVIEW.len()
        );
    }
    tracing::info!(
        documents = 1,
        chunks = chunks.len(),
        chunk_size = chunker.chunk_size(),
        chunk_overlap = chunker.chunk_overlap(),
        "chunked the overview document"
    );
    banner::note(&format!(
        "{} chunks of at most {} characters with {} shared (chunk ids are fresh uuid4s per run)",
        chunks.len(),
        chunker.chunk_size(),
        chunker.chunk_overlap()
    ));
    for (index, chunk) in chunks.iter().enumerate() {
        banner::step(&format!(
            "chunk[{index}] doc={} id={} {:?}",
            chunk.doc_id,
            short_id(&chunk.id),
            snippet(&chunk.text, PREVIEW_CHARS)
        ));
    }

    // ------------------------------------------------------- embed + store --
    banner::step("embed + store: hash vectors into an in-memory store, wired by a knowledge base");

    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION));
    let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::with_collection("06-chunks"));
    let config = KnowledgeBaseConfig {
        chunk_size: CHUNK_SIZE,
        chunk_overlap: CHUNK_OVERLAP,
        ..KnowledgeBaseConfig::new("retrieval-06")
    };
    let kb = SimpleKnowledgeBase::from_config(config, Arc::clone(&store), Arc::clone(&embedder))
        .context("building the knowledge base")?;

    let documents = vec![
        overview,
        Document::with_id("kb-store", STORE_DOC).metadata_entry("source", "06_retrieval"),
        Document::with_id("kb-chunking", CHUNK_DOC).metadata_entry("source", "06_retrieval"),
        Document::with_id("kb-embedding", EMBED_DOC).metadata_entry("source", "06_retrieval"),
    ];
    let document_ids = kb
        .add_documents(documents)
        .await
        .context("chunking, embedding, and storing the corpus")?;
    let kb_chunks = kb.count().await?;
    tracing::info!(
        documents = document_ids.len(),
        chunks = kb_chunks,
        "ingested the corpus"
    );
    banner::note(&format!(
        "knowledge base `{}` holds {} documents in {} chunks",
        kb.id(),
        document_ids.len(),
        kb_chunks
    ));

    // ------------------------------------------------------------- retrieve --
    banner::step("retrieve: the same corpus, ranked differently by each query");
    for query in QUERIES {
        let retrieval = RetrievalConfig::default().with_top_k(TOP_K);
        let results = kb
            .retrieve_with_config(query, &retrieval)
            .await
            .with_context(|| format!("retrieving for `{query}`"))?;
        if results.is_empty() {
            bail!("nothing was retrieved for `{query}`, so the knowledge base is empty");
        }
        banner::step(&format!("query {query:?} (top {TOP_K}):"));
        for (rank, result) in results.iter().enumerate() {
            print_result(rank, result);
        }
    }

    // ----------------------------------------------------------- raw store --
    banner::step("raw store: the layer below the knowledge base, used directly");

    let vector = embedder
        .embed_query(RAW_RECORD)
        .await
        .context("embedding the raw record")?;
    let record = VectorRecord {
        id: "raw-note".to_owned(),
        text: RAW_RECORD.to_owned(),
        vector,
        ..VectorRecord::default()
    };
    let written = store
        .upsert(vec![record])
        .await
        .context("upserting the raw record")?;
    tracing::info!(records = written, store = store.name(), "upserted a record");

    let query_vector = embedder
        .embed_query("record upserted straight into the store")
        .await
        .context("embedding the raw query")?;
    let hits = store
        .search(VectorQuery::new(query_vector).with_top_k(2))
        .await
        .context("searching the store directly")?;
    if hits.is_empty() {
        bail!("the store returned no hits for the raw record");
    }
    banner::note(&format!(
        "`{}` now holds {} records; `{}` counts its store, so the raw record is visible there too",
        store.name(),
        store.len().await?,
        kb.id()
    ));
    for (rank, hit) in hits.iter().enumerate() {
        tracing::info!(rank, id = %hit.id, score = hit.score, "store hit");
        banner::step(&format!(
            "store hit[{rank}] {} score={:.4} {:?}",
            hit.id,
            hit.score,
            snippet(&hit.text, PREVIEW_CHARS)
        ));
    }

    banner::answer(&format!(
        "{kb_chunks} knowledge-base chunks from {} documents, two query-dependent rankings, and one raw store query — all offline",
        document_ids.len()
    ));
    Ok(())
}

/// Prints one knowledge-base hit as `result[rank] doc=… score=… "snippet"`.
fn print_result(rank: usize, result: &RetrievalResult) {
    tracing::info!(
        rank,
        doc_id = result.doc_id.as_deref().unwrap_or("<none>"),
        chunk_id = result.chunk_id.as_deref().unwrap_or("<none>"),
        score = result.score,
        "retrieval hit"
    );
    banner::step(&format!(
        "result[{rank}] doc={} score={:.4} {:?}",
        result.doc_id.as_deref().unwrap_or("<none>"),
        result.score,
        snippet(&result.text, PREVIEW_CHARS)
    ));
}

/// Shortens an identifier to its first eight characters.
///
/// Chunk ids are generated `uuid4`s, which are long enough to wrap the terminal;
/// the prefix is enough to tell two chunks apart in the narrative.
fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Collapses whitespace and truncates `text` to `limit` characters.
///
/// Chunk text carries the source document's newlines, which would break the
/// one-line narrative, so the preview is normalized first. Truncation counts
/// characters rather than bytes so a multi-byte document cannot panic.
fn snippet(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let head: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}
