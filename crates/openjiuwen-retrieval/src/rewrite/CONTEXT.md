# rewrite — query rewriting

## Intent

Turns a user's latest turn into a standalone retrieval query, with the history resolved and keywords extracted.
Port of Python's `openjiuwen/core/retrieval/query_rewriter/` package, which is LLM-only; the Rust crate adds a deterministic heuristic rewriter so retrieval works fully offline.
Both implementations return the same Python JSON shape (`before`, `intention`, `standalone_query`, `references`, `missing`, `typo`, `gibberish`, `from_history`), so callers can swap them freely.

## API Surface

- `RewriteOutput` — the rewrite result; serde snake_case with Python field names, `#[serde(default)]` on `intention`/`references`/`missing`/`typo`/`gibberish`/`from_history`, unknown fields tolerated.
- `RewriteOutput::from_query(query)` — an unchanged rewrite: `before`, `intention`, and `standalone_query` mirror the input; `is_usable()` reports a non-blank standalone query.
- `TypoFix { original, corrected, reason }` — one spelling correction, `TypoFix::new(original, corrected, reason)`.
- `RewriteRequest { query, history }` — `new(query)`, `with_history(iter)`; turns are oldest-first and carry no role.
- `QueryRewriter` — `rewrite(&RewriteRequest)`, `keywords(&str)`, `expand(&str)`, and a sequential `rewrite_batch(&[RewriteRequest])` default.
- `HeuristicRewriteConfig` — `max_keywords=8`, `max_queries=3`, `use_stopwords=true`, `stopwords`, `synonyms`, `corpus_vocabulary` (term → document frequency), `history_window=4`; builder setters for keywords/queries/synonyms/vocabulary.
- `HeuristicQueryRewriter` — `new(config)`, `Default`, `config()`, `build_output(&RewriteRequest)`, `extract_keywords(&str)`, `expand_query(&str)`.
- `normalize_query(text)` / `tokenize(text)` — public helpers: whitespace plus filler normalisation and lowercase Unicode tokenisation.
- `ModelRewriteConfig` — `prompt_template` (default `DEFAULT_REWRITE_PROMPT`), `history_window=4`, `fallback: Option<Arc<dyn QueryRewriter>>`; builder setters.
- `ModelQueryRewriter` — `new(Arc<dyn CompletionModel>, ModelRewriteConfig)`, `with_default_config(model)`, `model()`, `config()`, `render_prompt(&RewriteRequest)`.
- `DEFAULT_MAX_KEYWORDS`, `DEFAULT_MAX_QUERIES`, `DEFAULT_HISTORY_WINDOW`, `DEFAULT_REWRITE_PROMPT` — documented defaults.

## Behaviour

- `normalize_query` collapses whitespace, strips filler words (`please`, `um`, `so`, …) from both ends, keeps at least one token, and collapses adjacent duplicate tokens.
- `extract_keywords` lowercases, splits on non-alphanumerics, drops stopwords, keeps the first occurrence of each term, and ranks by descending IDF when `corpus_vocabulary` is set (unknown terms weigh as document frequency 1, corpus size is the vocabulary's largest frequency) or otherwise by descending length, ties in query order, truncated to `max_keywords`.
- `build_output` keeps the user's wording in `before`, normalises it into `standalone_query`, and lowercases the first sentence into `intention` (bounded to 120 characters).
- A follow-up query (at most three tokens, a referential first token, or a referential phrase such as `what about`) with non-blank history becomes `"<last history topic> <query>"`, and that turn is echoed in `from_history`.
- `missing` lists query keywords absent from the windowed history (empty without history); `references` maps each keyword with configured synonyms to a JSON array of expansions; `typo` and `gibberish` stay empty.
- `expand_query` returns the original query first, then the keyword-only query, then a synonym-expanded query; variants are deduplicated and bounded by `max_queries` (at least one).
- `ModelQueryRewriter` renders the template by explicit `{query}`/`{history}` replacement (query first), sends one completion, and repairs the reply: missing fields default, scalars become strings or single-element lists, JSON-in-a-string is decoded, and a blank `standalone_query` falls back to the request query.
- A reply with no parsable JSON object uses the configured fallback rewriter when present and otherwise fails with `Error::Retrieval` quoting a bounded preview; model errors propagate unchanged.
- Model-backed `keywords`/`expand` rewrite first and then apply the heuristic extractor to `standalone_query`.

## Constraints

- Depends on `crate::model` (`CompletionModel`, `extract_json_object`) and `openjiuwen_common`; no `openjiuwen-foundation` dependency, so this crate never depends upward.
- Blank queries are `Error::Validation` before any model call; unparsable model output is `Error::Retrieval`; never panics on user input or model output.
- Deterministic: no randomness, no wall-clock reads, no I/O; identical inputs produce identical outputs.
- `#![forbid(unsafe_code)]` applies crate-wide; libraries never use `println!`/`eprintln!`.

## Design Decisions

- Python's `_extract_json`/`_schema_repair` are ported as `extract_json_object` plus explicit coercion helpers; Python's optional `json_repair` pass is not reimplemented, so a reply needs to be valid JSON once the object span is cut out.
- Python's LLM-only rewriter is complemented by the heuristic rewriter because the crate's tests and examples must run offline; the heuristic also supplies the keyword/expansion views Python does not expose.
- Follow-up detection is a documented heuristic (short query, referential token, or referential phrase) rather than an LLM judgement, and the history topic is the first 16 tokens of the last windowed turn.
- Synonyms live in configuration (`references`), so no thesaurus is embedded and expansion stays reproducible.
- `force_string` renders numbers/booleans as JSON text (`true`, `42`) where Python's `str()` would print `True`/`42`; documented deviation.

## Test Strategy

- Unit tests beside the modules cover normalisation, tokenisation, keyword ranking with and without IDF, stopwords, follow-ups, history windows, references, expansion bounds, serde round-trip, prompt rendering, reply repair, fallback, and error kinds.
- `tests/rewrite.rs` exercises the public API end to end with `StaticCompletionModel`: keyword extraction, follow-up resolution, plain-query rewrite, expansion determinism, `RewriteOutput` JSON field names, and model replies with prose, fenced JSON, missing fields, and malformed content.
- Everything is offline and deterministic: scripted replies only, no network, no clock.

## Routing Table

- `./heuristic.rs` → model-free rewriter, keyword extraction, synonyms, expansion, normalisation helpers (`heuristic.rs` docs)
- `./model.rs` → LLM-backed rewriter, prompt rendering, reply parsing/repair, fallback policy
- `../model.rs` → `CompletionModel` seam, `StaticCompletionModel`, `extract_json_object`
