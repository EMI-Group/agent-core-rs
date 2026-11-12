# graph — triples, rule extraction, graph store

## Intent

Triple extraction and graph storage over retrieval chunks.
Ports the Python `openjiuwen/core/retrieval/common/triple*.py` models and the `indexing/processor/extractor/` interface; Python's only extractor is LLM-backed, so extraction is a deterministic pure-Rust rule engine here.
Python has no graph store (it embeds triple text into a vector index), so `GraphStore` is a documented Rust addition with an in-memory implementation.

## API Surface

- `triple` — `Triple { subject, predicate, object, metadata }` with `new`, `with_metadata`, `metadata_entry`, `text` (`"s p o"`), `normalized_key` (trimmed, lowercased `s|p|o`), `entities`, `has_blank_component`; serde snake_case with Python field names, defaults, unknown-field tolerant.
- `triple::Entity { name, kind, metadata }` with `new`, `with_kind`, `with_metadata`, `normalized_name`; kinds are `"capitalised"`, `"cjk"`, `"quoted"`.
- `triple::TripleMemory` — insertion-ordered dedupe by `normalized_key`: `extend` (owned or borrowed), `batch_extend`, `triples`, `triples_str` (newline-joined lowercase `"(s p o)"`), `contains`, `len`, `is_empty`, `clear` (Python `TripleMemory`).
- `triple::TripleBeam { nodes: Vec<SearchResult>, score }` with `new`, `triples`, `texts`, `contains_text`, `len`, `is_empty`, `From<(Vec<SearchResult>, f32)>`, `From<SearchResult>`, `Default` (Python `TripleBeam` over `SearchResult`s).
- `extract::TripleExtractor` — `async fn extract(&self, chunks: &[TextChunk]) -> Result<Vec<Triple>>`; an empty chunk list yields `Ok(vec![])`.
- `extract::RuleTripleExtractor` — `new`, `with_config` (validated), `config`, `extract_chunks` (sync core), `entities_in_text`.
- `extract::RuleExtractionConfig` — `max_triples_per_chunk = 16`, `min_entity_chars = 2`, `max_entity_chars = 80`, `predicates` (the `BUILT_IN_PREDICATES` set), `include_single_entity = false`, plus builder methods and `allows(predicate)`.
- `store::GraphStore` — `add_triples`, `triples_for_entity`, `neighbors(entity, hops)`, `search(&TripleQuery)`, `all_triples`, `len`, `is_empty`, `clear`, `statistics`.
- `store::TripleQuery { subject, predicate, object, entity, metadata_filter, limit }` — exact case-insensitive component matches, subject-or-object `entity`, metadata filter, post-filter `limit`; blank strings count as absent.
- `store::GraphStatistics { triple_count, entity_count, predicate_count }`.
- `store::InMemoryGraphStore` — `new` (sync), `with_triples` (validated), adjacency indexes for entities and predicates, insertion-ordered reads.

## Extraction Rules

- Sentences split on `.`, `!`, `?`, `;`, and newlines; decimals (`3.5`), dotted chains (`e.g.`), and word abbreviations before a lowercase word (`Inc. was`) do not split.
- The earliest usable pattern occurrence in a sentence wins; ties go to table order, and the glossary form competes on its colon offset (patterns win ties).
- Pattern families: organisation (`founded_by`, `owned_by`, `acquired`), creation (`created`), exemplar lists (`includes`, split on `,`/`and`/`or`/`&`), possession (`contains`, `has`), copula (`is_a`), usage (`uses`, `with`), and the `X: Y` glossary form (`describes`).
- Objects are the sentence remainder after the keyword, cut at relative clauses and finite verbs, then trimmed of articles, connectors, and punctuation; subjects drop a sentence label, a prepositional prefix, and comma-separated lead-ins.
- Entity candidates are quoted terms, CJK runs of two or more characters, and capitalised phrases (joined by connectors such as `of`/`and`/`van`).
- Each triple carries `doc_id`, `chunk_id`, and a numeric `confidence` in `[0, 1]` derived from pattern strength plus entity evidence, rounded to two decimals.
- Output is deterministic: chunk-major, then sentence order, deduplicated per chunk by `normalized_key`, capped per chunk.
- `include_single_entity` (off by default) turns a sentence with exactly one entity into `(entity, "appears_in", <parent doc id>)`.

## Store Rules

- `add_triples` trims components, rejects blank subject/predicate/object with `Error::Validation` before mutating, and returns how many triples were new.
- Duplicates are stored once; each triple is indexed under both its subject and its object (unless they are equal).
- `triples_for_entity` and `neighbors` match entities case-insensitively and trimmed.
- `neighbors(entity, 0)` equals `triples_for_entity(entity)`; each extra hop adds the triples of newly discovered entities in breadth-first, insertion order, deduplicated and bounded by the graph size.
- Every read returns insertion order; `clear` resets the indexes.

## Constraints

- Only `openjiuwen-common` and in-workspace dependencies already declared by the crate; no new dependencies and no model calls.
- Pure, deterministic processing: no randomness, no wall clock, no I/O, no global mutable state.
- Errors use `openjiuwen_common::error::{Error, Result}`; invalid configuration and blank triples are `Error::Validation`.
- Files stay under 900 lines; text helpers live in the private `text` module.
- Everything is documented for `missing_docs`; no panics on user input.

## Routing Table

- `./triple.rs` → `Triple`, `Entity`, `TripleMemory`, `TripleBeam`
- `./extract.rs` → `TripleExtractor` trait, `RuleTripleExtractor`, pattern table, configuration
- `./text.rs` → private sentence/keyword/entity text helpers used by extraction
- `./store.rs` → `GraphStore` trait, `TripleQuery`, `GraphStatistics`, `InMemoryGraphStore`
- `../tests/graph.rs` → extraction families, dedupe/cap, CJK, metadata, memory/beam, store traversal and queries
