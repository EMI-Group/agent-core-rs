# extract

## Intent

Entity/relation extraction for graph memory: the `EntityExtractor` seam plus its two implementations.
`HeuristicExtractor` is the deterministic, dependency-free default; `LlmExtractor` asks a `Model` for JSON and falls back to the heuristic extractor when the model output stays unusable, recording the fallback and its attempt count on `Extraction::degraded`.
The module owns the extraction data types and never touches graph persistence.

## API Surface

- `EntityExtractor` — `name()` and `async extract(text, scope, options)`; `Send + Sync`.
- `ExtractedEntity { name, entity_type, importance: Option<f32> }`, `ExtractedRelation { from, to, relation_type }`, `Extraction { entities, relations, degraded: Option<String> }`; `serde`, snake_case fields, `degraded` omitted from JSON when unset.
- `Extraction::is_degraded()` and `Extraction::degraded_reason()` report the fallback; a plain heuristic pass and a successful model pass leave both empty.
- `HeuristicExtractor::new(&GraphMemoryConfig)` — candidate scans (quotes, backticks, `@mentions`, path/filename tokens, capitalized runs), stopword/dedupe filtering, entity typing, verb-pattern relations; no regex crate, no I/O.
- `LlmExtractor::new(Arc<dyn Model>, &GraphMemoryConfig)` — one `Model::invoke` per attempt with `ExtractionConfig::model_retries` retries, then the heuristic fallback with `degraded = Some("LLM entity extraction failed after N attempt(s); fell back to the heuristic extractor")`.
- `EXTRACTION_SYSTEM_PROMPT` — the exact JSON shape the parser accepts.

## Constraints

- `mod.rs` owns the data types and the trait and re-exports both implementations, so `graph::extract::{...}` paths stay unchanged.
- The two implementations are independent except that `llm` uses `heuristic` for its fallback; shared items come from `super`.
- Extraction never panics on arbitrary model output: unusable answers are rejected by the parser and retried.
- A degraded pass is never silent: the heuristic fallback still succeeds but the `EntityExtractor` contract requires reporting it on `Extraction::degraded`, so a caller can tell a full-quality pass from a fallback.
- Cancellation is checked before any model call and surfaces as `Error::Cancelled`.

## Routing Table

- `./mod.rs` → extraction data types, `EntityExtractor` trait, re-exports
- `./heuristic.rs` → `HeuristicExtractor`, candidate scanners, stopword/verb/relation tables, entity type inference
- `./llm.rs` → `LlmExtractor`, `EXTRACTION_SYSTEM_PROMPT`, tolerant JSON parsing helpers
