//! `15_retrieval_quality` — sparse retrieval, query rewriting, and multi-format parsing.
//!
//! Three seams decide whether retrieval returns the right thing, and this example
//! takes them one at a time: the **lexical** channel ([`Bm25Index`], whose ranking
//! is compared position by position with the dense channel of example 06), the
//! **query** that reaches the retriever ([`HeuristicQueryRewriter`] resolving an
//! elliptical follow-up, then [`ModelQueryRewriter`] over the completion-model
//! seam), and the **documents** themselves (`AutoLoader` dispatching DOCX, XLSX,
//! and PDF fixtures this example writes into a temp directory).
//!
//! Python counterparts: `openjiuwen/retrieval/` — `query_rewriter/` (LLM-only
//! there, complemented here by the offline heuristic), the sparse channel Python
//! delegates to its vector databases, and
//! `retrieval/indexing/processor/parser/` (Word/Excel/PDF parsing).
//! Everything is offline: BM25 and `HashEmbedding` are in process, the fixtures are
//! generated from string constants, and only `OPENJIUWEN_LIVE=1` reaches a real
//! provider for the model rewrite.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements,
    // The narrative is one linear tour of three seams; splitting `main` around
    // its sections would scatter the story the reader is meant to follow.
    clippy::too_many_lines
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use openjiuwen::core::common::error::{Error, Result as SdkResult};
use openjiuwen::core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen::core::foundation::prompt::BaseMessage;
use openjiuwen::core::retrieval::{
    crc32, tokenize, AutoLoader, Bm25Config, Bm25Index, Bm25Query, CharChunker, Chunker,
    CompletionModel, Document, DocxParser, Embedding, HashEmbedding, HeuristicQueryRewriter,
    HeuristicRewriteConfig, InMemoryVectorStore, Metadata, ModelQueryRewriter, QueryRewriter,
    RewriteOutput, RewriteRequest, SearchResult, StaticCompletionModel, VectorQuery, VectorRecord,
    VectorStore,
};

use common::banner;

/// The six documents both rankers see, keyed by a stable id.
const CORPUS: [(&str, &str); 6] = [
    (
        "bm25-lexical",
        "BM25 scores a document by term frequency and inverse document frequency, so a rare exact \
         word dominates the ranking while a common one barely moves it.",
    ),
    (
        "dense-vectors",
        "Dense embeddings map text to vectors, and cosine similarity ranks those vectors by \
         direction, which is what lets a paraphrase match a document it shares no word with.",
    ),
    (
        "hybrid-fusion",
        "Hybrid retrieval fuses a lexical ranking and a dense ranking with reciprocal rank fusion, \
         so exact matches and paraphrases both survive into the final list.",
    ),
    (
        "chunking",
        "Chunking cuts a document into overlapping character windows before embedding and \
         indexing, because a whole document is too coarse a unit to rank.",
    ),
    (
        "rewriting",
        "Query rewriting turns an elliptical follow-up into a standalone query by resolving its \
         pronouns against the conversation history.",
    ),
    (
        "parsing",
        "Office formats are ZIP containers of XML parts, so DOCX holds WordprocessingML and XLSX \
         holds worksheets, while PDF keeps its text in content streams.",
    ),
];

/// The query both rankers answer, chosen to reward one exact rare word.
const QUERY: &str = "term frequency of a rare word in a document ranking";
/// How many hits each ranker returns.
const TOP_K: usize = 4;
/// Components per hash vector; the fake needs a width, not a model.
const EMBEDDING_DIMENSION: usize = 64;
/// Characters per chunk when the parsed document is indexed.
const CHUNK_SIZE: usize = 120;
/// Characters shared by consecutive chunks.
const CHUNK_OVERLAP: usize = 20;
/// Character budget for a one-line preview.
const PREVIEW_CHARS: usize = 56;
/// Conversation history the follow-up depends on, oldest first.
const HISTORY: [&str; 2] = [
    "how do the lexical and dense channels combine",
    "hybrid retrieval fuses bm25 and dense rankings with reciprocal rank fusion",
];
/// An elliptical follow-up: "it"/"what about" refer to the history above.
const FOLLOW_UP: &str = "what about duplicate scores?";
/// The query the chunk index is probed with, taken from the DOCX text.
const CHUNK_QUERY: &str = "bm25 dense channel container";
/// The rewrite the offline scripted model replies with.
///
/// A model rewrite is one JSON object; the rewriter repairs missing keys, so only
/// the fields this example prints are spelled out.
const SCRIPTED_REWRITE: &str = r#"{
  "before": "what about duplicate scores?",
  "intention": "learn how ties between equal scores are broken",
  "standalone_query": "how does reciprocal rank fusion break ties between equal bm25 and dense scores",
  "references": {"ties": ["tie-break"]},
  "missing": ["tie-break policy"],
  "typo": [],
  "gibberish": [],
  "from_history": "hybrid retrieval fuses bm25 and dense rankings with reciprocal rank fusion"
}"#;
/// `word/document.xml`: a Title paragraph and two body paragraphs.
const DOCX_DOCUMENT: &str = r#"<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Retrieval notes</w:t></w:r></w:p><w:p><w:r><w:t>Hybrid retrieval fuses a BM25 channel and a dense channel with reciprocal rank fusion.</w:t></w:r></w:p><w:p><w:r><w:t>A DOCX file is a ZIP container of WordprocessingML parts.</w:t></w:r></w:p></w:body></w:document>"#;
/// `docProps/core.xml`: supplies `dc:title`.
const DOCX_CORE: &str = r#"<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties xmlns:cp="urn:cp" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Retrieval notes</dc:title></cp:coreProperties>"#;
/// `xl/workbook.xml`: one sheet, reached through relationship `rId1`.
const XLSX_WORKBOOK: &str = r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Scores" sheetId="1" r:id="rId1"/></sheets></workbook>"#;
/// `xl/_rels/workbook.xml.rels`: maps `rId1` to the worksheet part.
const XLSX_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#;
/// `xl/sharedStrings.xml`: the header cells and the two names.
const XLSX_SHARED_STRINGS: &str = r#"<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4"><si><t>Name</t></si><si><t>Score</t></si><si><t>alice</t></si><si><t>bob</t></si></sst>"#;
/// `xl/worksheets/sheet1.xml`: a header row and two data rows.
const XLSX_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>12.0</v></c></row><row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>7.5</v></c></row></sheetData></worksheet>"#;
/// Metadata keys printed for a loaded document, in a fixed order.
const METADATA_KEYS: [&str; 10] = [
    "title",
    "file_ext",
    "source_type",
    "paragraph_count",
    "table_count",
    "page_count",
    "has_text",
    "sheet_name",
    "row_index",
    "column_name",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;

    banner::title("15 retrieval quality — sparse search, query rewriting, multi-format parsing");
    banner::note(&format!(
        "run mode {}: BM25, hash embeddings, and generated fixtures stay in process; only the \
live rewrite needs a provider",
        mode.describe()
    ));

    let documents: Vec<Document> = CORPUS
        .iter()
        .map(|(id, text)| Document::with_id(*id, *text))
        .collect();

    // -------------------------------------------------------------- bm25 --
    banner::step("bm25: a lexical index over six short documents");

    let mut index = Bm25Index::new(Bm25Config::default()).context("building a BM25 index")?;
    let indexed = index.add_documents(&documents);
    ensure!(
        indexed == documents.len(),
        "indexed {indexed} of {} documents",
        documents.len()
    );
    let tokens = tokenize(QUERY);
    let stats = index.stats();
    banner::note(&format!(
        "tokenize({QUERY:?}) -> {} terms: {}",
        tokens.len(),
        tokens.join(" ")
    ));
    banner::note(&format!(
        "index: {} documents, {} distinct terms, average length {:.1} tokens, k1={} b={}",
        stats.document_count,
        stats.term_count,
        stats.average_document_length,
        index.config().k1,
        index.config().b
    ));

    let bm25_hits = index.search(&Bm25Query::new(QUERY).with_top_k(TOP_K));
    ensure!(!bm25_hits.is_empty(), "BM25 matched nothing for `{QUERY}`");
    banner::step(&format!("bm25 ranking for {QUERY:?}:"));
    print_hits("bm25", &bm25_hits);

    // ------------------------------------------------------------- dense --
    banner::step("dense: the same query through hash vectors and cosine similarity");

    let embedder = HashEmbedding::new(EMBEDDING_DIMENSION);
    let texts: Vec<String> = documents.iter().map(|doc| doc.text.clone()).collect();
    let vectors = embedder
        .embed_documents(&texts, None)
        .await
        .context("embedding the corpus")?;
    ensure!(
        vectors.len() == documents.len(),
        "embedded {} of {} documents",
        vectors.len(),
        documents.len()
    );
    let records: Vec<VectorRecord> = documents
        .iter()
        .zip(vectors)
        .map(|(document, vector)| VectorRecord {
            id: document.id.clone(),
            text: document.text.clone(),
            vector,
            ..VectorRecord::default()
        })
        .collect();
    let store = InMemoryVectorStore::with_collection("15-dense");
    let written = store
        .upsert(records)
        .await
        .context("storing the corpus vectors")?;
    ensure!(
        written == documents.len(),
        "stored {written} of {} records",
        documents.len()
    );
    let query_vector = embedder
        .embed_query(QUERY)
        .await
        .context("embedding the query")?;
    let dense_hits = store
        .search(VectorQuery::new(query_vector).with_top_k(TOP_K))
        .await
        .context("searching the vector store")?;
    ensure!(
        !dense_hits.is_empty(),
        "the store returned no hit for `{QUERY}`"
    );
    banner::step(&format!("dense ranking for {QUERY:?}:"));
    print_hits("dense", &dense_hits);

    // ------------------------------------------------------ hybrid view --
    banner::step("hybrid view: the same query, both rankers, position by position");

    for (dense_rank, hit) in dense_hits.iter().enumerate() {
        match bm25_hits
            .iter()
            .position(|candidate| candidate.id == hit.id)
        {
            Some(bm25_rank) if bm25_rank == dense_rank => {}
            Some(bm25_rank) => banner::step(&format!(
                "{:<14} bm25 #{} / dense #{}",
                hit.id,
                bm25_rank + 1,
                dense_rank + 1
            )),
            None => banner::step(&format!(
                "{:<14} bm25 outside top {TOP_K} / dense #{}",
                hit.id,
                dense_rank + 1
            )),
        }
    }
    for (bm25_rank, hit) in bm25_hits.iter().enumerate() {
        if !dense_hits.iter().any(|candidate| candidate.id == hit.id) {
            banner::step(&format!(
                "{:<14} bm25 #{} / dense outside top {TOP_K}",
                hit.id,
                bm25_rank + 1
            ));
        }
    }
    let shared = dense_hits
        .iter()
        .filter(|hit| bm25_hits.iter().any(|candidate| candidate.id == hit.id))
        .count();
    banner::note(&format!(
        "both rankings share {shared} of their {TOP_K} positions. BM25 pays for exact scarce terms and \
ignores the rest of the document; the hash embedder pays for whole-token overlap only, so a paraphrase \
scores zero with either. That is the gap a fusion step such as reciprocal rank fusion is for."
    ));

    // ----------------------------------------------------------- rewrite --
    banner::step("rewrite: an elliptical follow-up resolved against the conversation");

    let mut synonyms = BTreeMap::new();
    synonyms.insert(
        "scores".to_owned(),
        vec!["rankings".to_owned(), "similarity".to_owned()],
    );
    let heuristic =
        HeuristicQueryRewriter::new(HeuristicRewriteConfig::default().with_synonyms(synonyms));
    let request = RewriteRequest::new(FOLLOW_UP).with_history(HISTORY);
    let heuristic_output = heuristic
        .rewrite(&request)
        .await
        .context("rewriting the follow-up heuristically")?;
    print_rewrite("heuristic", &heuristic_output);

    let variants = heuristic
        .expand(FOLLOW_UP)
        .await
        .context("expanding the follow-up into retrieval variants")?;
    banner::note(&format!(
        "expansion variants ({}): {}",
        variants.len(),
        variants.join(" | ")
    ));

    banner::step("model rewrite: the same follow-up through the completion-model seam");

    let rewriter: Arc<dyn QueryRewriter> = if mode.is_live() {
        let model = common::live_model().await?;
        banner::note(&format!(
            "live rewrite through {}/{}",
            mode.provider().unwrap_or("<provider>"),
            mode.model().unwrap_or("<model>")
        ));
        Arc::new(ModelQueryRewriter::with_default_config(Arc::new(
            LiveCompletion::new(model),
        )))
    } else {
        banner::note(
            "using the scripted completion model; set OPENJIUWEN_LIVE=1 plus a model name and \
credential to rewrite with a real provider",
        );
        Arc::new(ModelQueryRewriter::with_default_config(Arc::new(
            StaticCompletionModel::single(SCRIPTED_REWRITE),
        )))
    };
    let model_output = rewriter
        .rewrite(&request)
        .await
        .context("rewriting the follow-up with the completion model")?;
    print_rewrite("model", &model_output);

    let rewritten_query = model_output.standalone_query.clone();
    let resolved_hits = index.search(&Bm25Query::new(rewritten_query.clone()).with_top_k(2));
    ensure!(
        !resolved_hits.is_empty(),
        "the rewritten query `{rewritten_query}` matched no document"
    );
    banner::note(&format!(
        "the query that actually reaches the retriever: {rewritten_query:?}"
    ));
    print_hits("rewritten", &resolved_hits);

    // ----------------------------------------------------------- parsing --
    banner::step("parse: DOCX, XLSX and PDF fixtures written into a temp dir");

    let workspace = common::tempdir()?;
    let loader = AutoLoader::new();

    let docx_bytes = zip_archive(&[
        ("word/document.xml", DOCX_DOCUMENT),
        ("docProps/core.xml", DOCX_CORE),
    ]);
    let parsed = DocxParser::new()
        .parse_bytes(&docx_bytes)
        .context("parsing the DOCX fixture")?;
    banner::step(&format!(
        "docx parser: dc:title={:?} blocks={} paragraphs={} tables={}",
        parsed.title.as_deref().unwrap_or("<none>"),
        parsed.blocks.len(),
        parsed.paragraph_count,
        parsed.table_count
    ));
    banner::step(&format!(
        "docx text: {:?}",
        snippet(&parsed.text, PREVIEW_CHARS)
    ));
    let docx = write_fixture(&workspace, "notes.docx", &docx_bytes).await?;
    let docx_documents = loader
        .load_as(&docx, Some("notes"), None)
        .await
        .context("loading notes.docx through the auto loader")?;
    ensure!(
        docx_documents.len() == 1,
        "the DOCX loader returned {} documents",
        docx_documents.len()
    );
    print_documents(&docx_documents);
    banner::note(
        "the auto loader stamps the file stem as `title`, over the dc:title the parser read — \
Python's AutoFileParser does the same; `load_as` is how a caller overrides it",
    );

    let xlsx_bytes = zip_archive(&[
        ("xl/workbook.xml", XLSX_WORKBOOK),
        ("xl/_rels/workbook.xml.rels", XLSX_RELS),
        ("xl/sharedStrings.xml", XLSX_SHARED_STRINGS),
        ("xl/worksheets/sheet1.xml", XLSX_SHEET),
    ]);
    let xlsx = write_fixture(&workspace, "scores.xlsx", &xlsx_bytes).await?;
    let xlsx_documents = loader
        .load_as(&xlsx, Some("scores"), None)
        .await
        .context("loading scores.xlsx through the auto loader")?;
    ensure!(
        !xlsx_documents.is_empty(),
        "the XLSX loader returned no document"
    );
    banner::step("xlsx: one row document and one column document per sheet");
    print_documents(&xlsx_documents);

    let pdf_bytes = pdf_document(&[
        "Retrieval quality report",
        "A PDF keeps its text in uncompressed content streams.",
    ]);
    let pdf = write_fixture(&workspace, "report.pdf", &pdf_bytes).await?;
    let pdf_documents = loader
        .load_as(&pdf, Some("report"), None)
        .await
        .context("loading report.pdf through the auto loader")?;
    ensure!(
        pdf_documents.len() == 1,
        "the PDF loader returned {} documents",
        pdf_documents.len()
    );
    banner::step("pdf: one document holding the decoded content-stream text");
    print_documents(&pdf_documents);

    // ------------------------------------------------- chunk + index --
    banner::step("chunk + index: the parsed DOCX text back through a chunker and BM25");

    let notes = docx_documents
        .first()
        .context("the DOCX loader returned no document")?;
    let chunker =
        CharChunker::new(CHUNK_SIZE, CHUNK_OVERLAP).context("building a character chunker")?;
    let chunks = chunker
        .chunk_documents(std::slice::from_ref(notes))
        .context("chunking the DOCX text")?;
    ensure!(
        !chunks.is_empty(),
        "the chunker produced no chunks for the DOCX text"
    );
    let mut chunk_index =
        Bm25Index::new(Bm25Config::default()).context("building the chunk index")?;
    let indexed_chunks = chunk_index.add_chunks(&chunks);
    ensure!(
        indexed_chunks == chunks.len(),
        "indexed {indexed_chunks} of {} chunks",
        chunks.len()
    );
    let chunk_hits = chunk_index.search(&Bm25Query::new(CHUNK_QUERY).with_top_k(2));
    ensure!(!chunk_hits.is_empty(), "no chunk matched `{CHUNK_QUERY}`");
    banner::note(&format!(
        "{} chunks of at most {CHUNK_SIZE} characters with {CHUNK_OVERLAP} shared; chunk ids are \
generated uuids, so the hits show their index instead",
        chunks.len()
    ));
    for (rank, hit) in chunk_hits.iter().enumerate() {
        let position = hit
            .metadata
            .get("chunk_index")
            .and_then(serde_json::Value::as_u64)
            .map_or_else(|| "-".to_owned(), |index| index.to_string());
        let total = hit
            .metadata
            .get("total_chunks")
            .and_then(serde_json::Value::as_u64)
            .map_or_else(|| "-".to_owned(), |count| count.to_string());
        tracing::info!(rank, score = hit.score, "chunk hit");
        banner::step(&format!(
            "chunk[{rank}] chunk={position}/{total} score={:.4} {:?}",
            hit.score,
            snippet(&hit.text, PREVIEW_CHARS)
        ));
    }

    banner::answer(&format!(
        "{} documents ranked by BM25 and by hash vectors, one follow-up rewritten two ways, and \
{} documents recovered from DOCX, XLSX and PDF fixtures — all offline",
        documents.len(),
        docx_documents.len() + xlsx_documents.len() + pdf_documents.len()
    ));
    Ok(())
}

/// Adapts the foundation [`Model`] to the retrieval layer's one-prompt seam.
///
/// `openjiuwen-retrieval` deliberately does not depend on `openjiuwen-foundation`,
/// so a model-driven component there talks to [`CompletionModel`]; the live branch
/// bridges the two with the same small impl a caller would write.
struct LiveCompletion {
    /// The provider client built from the environment.
    model: Arc<dyn Model>,
    /// Per-call settings (model name, sampling) read from the environment.
    request: ModelRequestConfig,
}

impl LiveCompletion {
    /// Wraps `model` with a request built from the environment.
    fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            request: ModelRequestConfig::from_env(),
        }
    }
}

#[async_trait]
impl CompletionModel for LiveCompletion {
    fn name(&self) -> &str {
        self.model.model_name()
    }

    async fn complete(&self, prompt: &str) -> SdkResult<String> {
        let response = self
            .model
            .invoke(&[BaseMessage::user(prompt)], &self.request)
            .await?;
        response
            .message()
            .map(|message| message.text().into_owned())
            .ok_or_else(|| Error::llm("the model returned no choice, so the rewrite has no reply"))
    }
}

/// Prints one ranking as `label[rank] id score "snippet"`.
fn print_hits(label: &str, hits: &[SearchResult]) {
    for (rank, hit) in hits.iter().enumerate() {
        tracing::info!(label, rank, id = %hit.id, score = hit.score, "ranking hit");
        banner::step(&format!(
            "{label}[{rank}] {:<14} score={:.4} {:?}",
            hit.id,
            hit.score,
            snippet(&hit.text, PREVIEW_CHARS)
        ));
    }
}

/// Prints the fields of a [`RewriteOutput`] a retriever consumes.
fn print_rewrite(label: &str, output: &RewriteOutput) {
    tracing::info!(
        label,
        before = %output.before,
        standalone = %output.standalone_query,
        "rewrite output"
    );
    banner::step(&format!("{label}: before {:?}", output.before));
    banner::step(&format!(
        "{label}: intention {:?}",
        snippet(&output.intention, PREVIEW_CHARS)
    ));
    banner::step(&format!(
        "{label}: standalone_query {:?}",
        output.standalone_query
    ));
    if !output.from_history.is_empty() {
        banner::step(&format!(
            "{label}: from_history {:?}",
            snippet(&output.from_history, PREVIEW_CHARS)
        ));
    }
    if !output.missing.is_empty() {
        banner::step(&format!("{label}: missing {}", output.missing.join(", ")));
    }
    if !output.references.is_empty() {
        let rendered = output
            .references
            .iter()
            .map(|(term, expansions)| format!("{term} -> {expansions}"))
            .collect::<Vec<_>>()
            .join(", ");
        banner::step(&format!("{label}: references {rendered}"));
    }
}

/// Prints one `id metadata "snippet"` line per loaded document.
fn print_documents(documents: &[Document]) {
    for document in documents {
        banner::step(&format!(
            "{} {} {:?}",
            document.id,
            metadata_summary(&document.metadata),
            snippet(&document.text, PREVIEW_CHARS)
        ));
    }
}

/// Renders the loader metadata keys present on a document, in a fixed order.
fn metadata_summary(metadata: &Metadata) -> String {
    METADATA_KEYS
        .iter()
        .filter_map(|key| metadata.get(*key).map(|value| format!("{key}={value}")))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Writes `bytes` as `name` inside `workspace` and returns the path as a string.
async fn write_fixture(workspace: &tempfile::TempDir, name: &str, bytes: &[u8]) -> Result<String> {
    let path = workspace.path().join(name);
    tokio::fs::write(&path, bytes)
        .await
        .with_context(|| format!("writing the {name} fixture"))?;
    Ok(path
        .to_str()
        .context("the fixture path is not valid UTF-8")?
        .to_owned())
}

/// Builds a minimal single-page PDF with an uncompressed text content stream.
///
/// PDF text lives in content streams, and the parser reads exactly those: the
/// objects around them only have to be well formed enough for the stream keyword
/// to be found. The text is a plain `Tj` string, so it needs no escaping here.
fn pdf_document(lines: &[&str]) -> Vec<u8> {
    let mut content = String::from("BT /F1 11 Tf 72 720 Td");
    for (index, line) in lines.iter().enumerate() {
        if index > 0 {
            content.push_str(" 0 -16 Td");
        }
        content.push_str(" (");
        content.push_str(line);
        content.push_str(") Tj");
    }
    content.push_str(" ET");
    format!(
        "%PDF-1.4\n\
         1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
         2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
         3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n\
         4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n\
         trailer\n<< /Root 1 0 R >>\n%%EOF\n",
        content.len(),
        content
    )
    .into_bytes()
}

/// Builds a minimal ZIP archive whose entries are **stored** (uncompressed).
///
/// `openjiuwen-retrieval` ships a ZIP reader but no writer, and its own fixture
/// builder is `#[cfg(test)]`-only, so the example writes the container itself:
/// one local file header per entry, then the central directory, then the
/// end-of-central-directory record. Only the stored method is used — that is what
/// keeps this to a page instead of a `DEFLATE` encoder — and the checksum comes
/// from the parser's public `crc32` helper rather than a second implementation,
/// since the reader verifies CRC-32 and both declared sizes on extraction.
#[allow(
    // A fixture is a few hundred bytes, so no offset or length can exceed 16 or 32
    // bits; checked conversions would only add error plumbing to a test-shaped
    // writer whose inputs this file controls.
    clippy::cast_possible_truncation
)]
fn zip_archive(entries: &[(&str, &str)]) -> Vec<u8> {
    /// Version 2.0, the version that introduced the stored/deflate methods.
    const VERSION: u16 = 20;
    /// `APPNOTE` method 0: the payload is stored verbatim.
    const STORED: u16 = 0;

    let mut archive: Vec<u8> = Vec::new();
    let mut directory: Vec<u8> = Vec::new();
    for (name, text) in entries {
        let data = text.as_bytes();
        let crc = crc32(data);
        let size = data.len() as u32;
        let name_len = name.len() as u16;
        let offset = archive.len() as u32;

        // Local file header: signature, version, flags, method, DOS time/date,
        // CRC-32, compressed and uncompressed size, name and extra lengths.
        archive.extend_from_slice(&0x0403_4B50_u32.to_le_bytes());
        archive.extend_from_slice(&VERSION.to_le_bytes());
        archive.extend_from_slice(&0_u16.to_le_bytes());
        archive.extend_from_slice(&STORED.to_le_bytes());
        archive.extend_from_slice(&0_u16.to_le_bytes());
        archive.extend_from_slice(&0_u16.to_le_bytes());
        archive.extend_from_slice(&crc.to_le_bytes());
        archive.extend_from_slice(&size.to_le_bytes());
        archive.extend_from_slice(&size.to_le_bytes());
        archive.extend_from_slice(&name_len.to_le_bytes());
        archive.extend_from_slice(&0_u16.to_le_bytes());
        archive.extend_from_slice(name.as_bytes());
        archive.extend_from_slice(data);

        // Central directory header: the same facts plus the local header offset,
        // which is how the reader finds the payload it just verified.
        directory.extend_from_slice(&0x0201_4B50_u32.to_le_bytes());
        directory.extend_from_slice(&VERSION.to_le_bytes());
        directory.extend_from_slice(&VERSION.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&STORED.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&crc.to_le_bytes());
        directory.extend_from_slice(&size.to_le_bytes());
        directory.extend_from_slice(&size.to_le_bytes());
        directory.extend_from_slice(&name_len.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&0_u16.to_le_bytes());
        directory.extend_from_slice(&0_u32.to_le_bytes());
        directory.extend_from_slice(&offset.to_le_bytes());
        directory.extend_from_slice(name.as_bytes());
    }
    let directory_offset = archive.len() as u32;
    let directory_size = directory.len() as u32;
    let count = entries.len() as u16;
    archive.extend_from_slice(&directory);

    // End of central directory: this disk's and the total entry count, the
    // directory's size and offset, and an empty archive comment.
    archive.extend_from_slice(&0x0605_4B50_u32.to_le_bytes());
    archive.extend_from_slice(&0_u16.to_le_bytes());
    archive.extend_from_slice(&0_u16.to_le_bytes());
    archive.extend_from_slice(&count.to_le_bytes());
    archive.extend_from_slice(&count.to_le_bytes());
    archive.extend_from_slice(&directory_size.to_le_bytes());
    archive.extend_from_slice(&directory_offset.to_le_bytes());
    archive.extend_from_slice(&0_u16.to_le_bytes());
    archive
}

/// Collapses whitespace and truncates `text` to `limit` characters.
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
