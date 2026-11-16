# search — query-time (online) capability retrieval

## Intent

The query-time half of Python `openjiuwen.symphony.retrieval` (v0.1.20): `retrieval/search/runtime/**` plus `service/models.py` and `service/retriever.py`.
It loads a prebuilt index (nothing here builds, mutates or persists one) and answers a query by walking the taxonomy or by ranking a flattened catalog in one pass.
`Retriever` is the public shape (validate query → resolve `top_k` → walk or flat rank → publicize into `RetrievalResult` rows); everything below it is engine detail.
The model is optional (`SelectionMode::Lexical`, `FallbackMode::Deterministic`), and no fallback is silent: `SelectionProvenance` rides on every `SelectionResult` and is summarized on all three result types, so a configured-but-unused model is visible in the returned value, not only in the log.

### Pipeline

1. **Load** — `artifacts::load_index` returns a `LoadedIndex`; `Retriever::load` wraps it, `Retriever::from_loaded_index` accepts one built in memory.
2. **Translate** — `runtime_config_from_config` flattens `RetrieverConfig` into the engine's `ProgressiveSearchConfig`, clamping every bound.
3. **Optional embedding** — `Retriever::with_embedding` installs an `EmbeddingScorer` inside `EmbeddingBlendedTopKSelector` (the walk's fallback selector) and the flat retriever; `weight == 0.0` (default) reproduces the lexical order exactly.
4. **Retrieve** — `ProgressiveRetriever` (tree walk over `RecursiveSearchEngine`), `EmbeddingBlendedTopKSelector`, or `FlatRetriever` (one-shot ranking).
5. **Expose, render, select** — `DefaultCurrentSubtreeProvider` → `CurrentSubtree` (through an injected `FragmentBuildFn`), `DefaultSubtreeRenderer` → `PromptBundle`, a `TopKSelector` → `SelectionResult`.
6. **Expand and reduce** — `DefaultTargetExpander` → `ExpansionPlan`; `DefaultBranchReducer` merges leaves with children into one `NodeSearchResult`.
7. **Publicize and trace** — `search_details_with` resolves each candidate `payload` (a leaf `cid`) through the catalog and exposes the public `worker_id`, name, description, branch path and category (internal cids never reach a caller), while every stage records `RetrieverTraceEvent`s into one `RetrieverTrace` that the facade copies into `RetrievalResult::trace_events`.

### Degradation observability

Every answer that differs from what was configured is visible in the returned value, the trace, or a `tracing` record.

| Path | How it is observable |
| --- | --- |
| `Generate`, model wired, model answered | `MODE_GENERATE` + `Model`, `degraded = false` |
| `Generate`, no generate selector (no model; the default) | `MODE_LEXICAL_NO_MODEL` + `LexicalNoModel`, `degraded = true`, one `warn!` per instance, `degraded_selections ≥ 1` |
| same, under `FallbackMode::Error` | typed `Error::Unsupported` naming the fix, before any selection |
| model call (or prompt build) failed under `Deterministic` | `MODE_LEXICAL` + `LexicalAfterModelError`, `degraded = true`, a warn per call naming node, depth, model and error |
| reply with no known code (non-blank, not abstaining) | `MODE_GENERATE` + `UnusableReply`, `degraded = true`, a warn naming the reply length; nothing is selected |
| model abstained or replied blankly | `Model` with `is_abstain` — a legitimate empty answer, not a degradation |
| `SelectionMode::Lexical` | `MODE_LEXICAL` + `Lexical`, `degraded = false` |
| embedder wired with a positive weight | `MODE_EMBEDDING_BLEND` + `EmbeddingBlend`; flat reports `METHOD_FLAT` + `EmbeddingBlend` |
| embedder with weight `0.0`, or none | inactive by configuration: `Lexical` / `METHOD_FLAT_LEXICAL`, never degraded |
| exactly one selectable target | `MODE_SINGLE_SELECTABLE_SHORTCUT` + `SingleSelectable` — a documented shortcut, not a degradation |
| no selectable target / empty subtree | the node ends with no `fragment_selected` event; nothing was configured away, so nothing is marked |
| over-large `SearchRequest::top_k` | a `warn!` naming both bounds, `top_k_clamped = true` (`Some(0)` is the sentinel, not a clamp) |
| `max_steps` reached | one `search_truncated` event carrying `max_steps`, state reported exhausted (never an error) |
| disclosure budget exceeded | `ExposedFragment::{rendered_chars, truncated}`, plus a `tracing` record naming the withheld count for `max_items_per_branch` |
| subtree provider built without a `FragmentBuildFn` | typed `Error::Unsupported` (never an empty subtree) |
| unusable lexical query expression | a `warn!`; the selection still runs |
| flat retriever without a usable embedder | `method`/`selection_provenance` state the lexical pass; never a silent blend |

## API Surface

Crate-relative paths; every public item is documented (`missing_docs` under `clippy -D warnings`). Field lists live in the code's docs — below are the inventory, the constants and the behaviour a caller cannot infer.

### `flow` (`src/retrieval/search/flow.rs`)

`ABSTAIN_TOKEN = "0"` (reserved, never a candidate code), `ChatMessage`, `SearchCursor` (`new`/`at`/`resolve`), `ExposedNode`, `SelectableResolution`, `ExposedFragment` (`resolution`/`resolutions`/`selectable_count`/`is_empty`), `SelectableTarget`, `CurrentSubtree`, `SelectionProtocol` (custom `Default`, `abstain_token` = `ABSTAIN_TOKEN`), `PromptBundle`, `ChildSearchCursor`, `ExpansionPlan`, `NodeSearchResult`.
`SelectionProvenance` { `Model`, `Lexical` (serde default), `LexicalNoModel`, `LexicalAfterModelError`, `EmbeddingBlend`, `SingleSelectable`, `UnusableReply` } with `as_str`/`parse`/`is_degraded`; `SelectionResult` { raw_output, selected_targets, is_abstain, provenance, degraded } with `with_provenance`, where `degraded == provenance.is_degraded()`.

### `config` (`src/retrieval/search/config.rs`)

`DEFAULT_TOP_K = 10` (facade, Python `RetrieverConfig.top_k`), `RUNTIME_TOP_K = 5` (engine, declared in `mod.rs`), `SelectionMode` { `Generate` (default), `Lexical` (Rust-only: rank without a model) }.
`DisclosureConfig` defaults: `max_exposure_depth_per_call 2`, `exposure_threshold 12`, `compact_codes_enabled false`, `compact_codebook []` (empty ⇒ numeric `"1"`, `"2"`, …), `flatten_full_tree_in_prompt false`, plus Rust-only `max_chars 0`, `max_items_per_branch 0`, `max_description_chars 150`, `code_width 0`; `EmbeddingScoringConfig` { weight 0.0, max_documents 0 }, whose `with_weight` clamps into `[0, 1]` and turns a non-finite weight into `0.0`.
`ProgressiveSearchConfig` { top_k 5, max_tokens 96, max_depth, max_steps, max_branch_choices 6, branch_choice_slack 2, branch_candidate_slack 1, round_robin_branch_reduce true, selection_mode `Generate`, fallback_mode `Deterministic`, disclosure, embedding, truncation }; both are `with_*`-builder + `validate()`, which rejects `top_k == 0`, a bad codebook and contradicting truncation bounds, and `0` means "unlimited"/"derive it" in every Rust-only bound.
`ProgressiveSearchConfig::default()` is the production shape with Python's facade overrides (`99`, `1_000_000_000`, compact codes on, flatten on).

### `lexical` (`src/retrieval/search/lexical.rs`)

`LexicalIndex` (`build`/`from_loaded_index`/`search`/`search_in`/`missing_terms`/`match_snippet`/`document`), `LexicalDocument`, `LexicalHit`, `tokenize`, `normalize_identity`, `without_front_matter`, `QueryMatcher`, `compile_matcher`.
Constants: `BM25_K1 1.5`, `BM25_B 0.72`, `FIELD_WEIGHTS [5, 5, 5, 3.4, 0.35, 0.1]` (key, name, aliases, description, body, category), `IDENTITY_BONUS 16`, `PHRASE_FIELD_WEIGHTS [12, 5, 0.5, 0.1]`, `FREQUENCY_FIELD_WEIGHTS [5, 5, 5, 4, 1]`, `BRANCH_DOCUMENT_PREFIX "node::"`, `MAX_QUERY_CHARS 512`.
`score = bm25 + field_score + phrase_score`: `bm25 = Σ idf² · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))` over the distinct query terms present (`idf = ln(1 + (N - df + 0.5)/(df + 0.5))`), plus the strongest single `idf²` once.
`field_score = Σ_f w_f · |terms ∩ field_f| / |terms|` over `FIELD_WEIGHTS` plus `IDENTITY_BONUS · |terms ∩ (key ∪ name ∪ aliases)| / |key ∪ name ∪ aliases|`; `phrase_score` adds `PHRASE_FIELD_WEIGHTS[f]` once per `|`-separated alternative on a substring hit.
Ranking tuple: identity tier (`0` key, `1` name, `2` alias, `3` prefix of one of those, `4` otherwise), descending score (`f64::total_cmp`), case-folded key, key.
`from_loaded_index` indexes one document per catalog record (`key = cid`, body = `retrieval_text`) plus one per branch node; `search_in` treats an **empty** `keys` slice as the whole corpus and ignores unknown or duplicated keys.

### `embedding` (`src/retrieval/search/embedding.rs`)

`EmbeddingScorer` (`new(Arc<dyn Embedding>)`/`index`/`index_capped`/`upsert`/`remove`/`clear`/`scores`/`search`), `cosine_similarity` (0.0 for degenerate inputs), `blend_scores` (Rust-only, additive).
`blended = (1 - w) · lex/max_lex + w · (cos + 1)/2` over the union of both key sets (a key missing on one side scores `0.0`), `w` clamped into `[0, 1]`, so `0.0` reproduces the lexical order exactly and `1.0` the embedding order.
`index_capped` honours `max_documents` (`0` = unlimited) in key order; `search` sorts descending, ties on key.

### `codebooks` (`src/retrieval/search/codebooks.rs`)

`CompactCodebook` (`new`/`codes`/`code`/`prefix`/`trie`/`from_value`), `CodeTrie` (`is_valid_prefix`/`is_code`/`completions`), `default_compact_boundary_codebook`, `DEFAULT_COMPACT_CODEBOOK_SIZE = 512`.
`new` trims each entry and rejects a blank, whitespace-containing, duplicated or `"0"` code with `Error::Validation` naming it; `prefix(count)` fails when the book is shorter.
The generated book is the first 512 of `AA, AB, … ZZ` positionally — a documented deviation from Python's tokenizer-tuned permutation (single-token-ness cannot be verified).

### `truncation` (`src/retrieval/search/truncation.rs`)

`ScoreTruncationConfig`, `ScoreTruncationDecision`, `truncate_sorted_by_score`, `truncate_chars`, `truncate_chars_with_ellipsis`, `take_budget`; reasons `REASON_MIN_SCORE`/`REASON_RELATIVE_MIN_SCORE`/`REASON_AUTOCUT`.
Decision order is the reference's — absolute `min_score`, `relative_min_score`, then the `autocut` gap — and each rule may only *lower* the cutoff, so the strictest wins and `reason` names the rule that moved it.
`relative_min_score` applies only when the best score is positive; `autocut` ignores non-positive or sub-threshold adjacent drops and cuts after the `autocut_jumps`-th qualifying one; the input is never reordered, and empty input gives `kept_count = 0`, `cutoff_index = None`.
The budget helpers treat `0` as unlimited, are character-boundary safe, and count the ellipsis inside `max_chars`.

### `contracts` (`src/retrieval/search/contracts.rs`)

The five seams: `CurrentSubtreeProvider::get_current_subtree`, `TopKSelector::{build_protocol, select_topk}` (async), `TargetExpander::expand_selected_targets`, `BranchReducer::reduce_branch_results`, plus `SubtreeRenderer` in `render/`.
Callbacks/requests: `FragmentBuildFn`, `GenerateFn`, `GenerateFragmentSelector`, `SelectRequest`, `GenerateRequest`.
The engine decides only *when* to call a seam; rendering is the structural deviation — Python's provider calls `render.disclosure` directly, while here it takes a `FragmentBuildFn` and answers `Error::Unsupported` until one is supplied.
The request structs borrow their inputs plus `&mut RetrieverTrace`, so they are not `serde` types.

### `subtree` (`src/retrieval/search/subtree.rs`)

`DefaultCurrentSubtreeProvider` (`new`/`with_builder`/`with_item_counter`/`rebind`/`clear_cache`/`generation`/`evictions`), `ItemCounterFn`, and the `choices_cache_key`/`build_progressive_*` catalog helpers.
The exposure cache is keyed by `(generation, node_id, branch_path, top_k)`; `rebind` swaps the tree and bumps the generation under one write lock before clearing the cache, so a stale fragment can never be served, and a hit pairs the cached fragment with the caller's own cursor.


### `selector` (`src/retrieval/search/selector.rs`)

`LexicalTopKSelector` (`select`, sync), `DefaultTopKSelector` (`with_generate_selector`/`is_model_backed`/`has_warned_no_model`), `selection_provenance_for_mode`, `trace_selection_provenance`, `trace_degraded_selections`, `is_abstain_output` (blank is NOT abstention; `"0"` and the hints are), `query_text`; `FRAGMENT_SELECTED_EVENT "fragment_selected"`, `MODE_LEXICAL "lexical_fallback"`, `MODE_GENERATE "generate"`, `MODE_LEXICAL_NO_MODEL "lexical_no_model"`.
`SelectionMode::Lexical` always ranks lexically; `Generate` with a generate selector calls it and honours `FallbackMode` (`Error` surfaces the failure, `Deterministic` warns and degrades); `Generate` without one ranks lexically under `Deterministic` and answers `Error::Unsupported` under `Error`.

| path | `mode` | `provenance` | `degraded` |
| --- | --- | --- | --- |
| model answered | `MODE_GENERATE` | `Model` | false |
| unparseable reply (non-blank, not abstaining, no known code) | `MODE_GENERATE` | `UnusableReply` | true |
| model failed, lexical fell in | `MODE_LEXICAL` | `LexicalAfterModelError` | true |
| `SelectionMode::Lexical` | `MODE_LEXICAL` | `Lexical` | false |
| `Generate` with no generate selector | `MODE_LEXICAL_NO_MODEL` | `LexicalNoModel` | true |

Selection records exactly one `fragment_selected` with `mode`, `raw_output`, `selected_canonical_ids`, `selected_codes`, `provenance`, `degraded`; the single-selectable shortcut records the same keys except `raw_output` (see `engine`).
`warn_no_model_once` warns **once per selector instance** (clones share the `Arc<AtomicBool>` behind `has_warned_no_model`), because a walk would otherwise repeat one configuration problem at every node; a model failure and an unusable reply each warn on the failing call.
`trace_selection_provenance` reads the last `fragment_selected` (`provenance` detail, else `mode`) and `trace_degraded_selections` counts the degraded ones.
The lexical selector builds one `LexicalDocument` per selectable target (key = canonical id, category = the branch path joined with `" > "`), ranks the query through `ranking_expression` (a `|`-separated alternative list), applies the truncation rules, and never abstains.

### `expand` (`src/retrieval/search/expand.rs`) and `reduce` (`src/retrieval/search/reduce.rs`)

`DefaultTargetExpander` (`branch_top_k(top_k, branch_count)`), `DefaultBranchReducer` (`merge_branch_candidates`).
The expander emits terminal resolutions as leaves in declared selection order (rank `1`; the reducer re-ranks nothing) and branches as `ChildSearchCursor`s one level deeper, at `min(top_k, ceil(top_k / max(1, branch_count)) + branch_candidate_slack)` floored at `1` (`top_k == 0` ⇒ unlimited); `max_branch_choices > 0` keeps the first N child cursors and `max_depth > 0` stops descending at `cursor.depth >= max_depth`.
The reducer merges `[local_leaves, child_results…]` round-robin (one candidate per branch, then the next index across every branch), deduping by `payload` (falling back to `item_id` when blank), then appends across the concatenated branches when the round-robin stalls; `round_robin_branch_reduce = false` switches to concatenate-then-dedupe, and `top_k == 0` consumes every branch.

### `engine` (`src/retrieval/search/engine.rs`)

`RecursiveSearchEngine` (`new(five seams + config)`/`search`/`step`), `EngineRequest`, `TraversalState` (serializable, resumable), `TraversalStep`; `FRAGMENT_BUILT_EVENT`, `FRAGMENT_CONTINUE_EVENT`, `REDUCE_COMPLETE_EVENT`, `SEARCH_COMPLETE_EVENT`, `SEARCH_TRUNCATED_EVENT`, `MODE_SINGLE_SELECTABLE_SHORTCUT`.
Per-node order: `get_current_subtree` (then `fragment_built` with `selectable_count`, `rendered_tree`) → no target ends the node → exactly one target takes the shortcut (no render, no model; `fragment_selected` with `provenance = "single_selectable"`, `degraded = false`) → otherwise `build_protocol` → `render_subtree` → `select_topk` (an abstention or empty selection ends the node) → `expand_selected_targets` (then `fragment_continue` with `leaf_count`, `branch_count`, `selected_terminal_ids`, `selected_branch_ids`) → children pushed in reverse so declared order is visit order → `reduce_branch_results` (then `reduce_complete` with `input_candidates`, `output_candidates`).
No target at a node is not a degradation (nothing was selectable, so no provenance is recorded), and the walk is an explicit frame stack: `max_depth` is a real bound, children are visited sequentially (hence a deterministic trace), and a saved `TraversalState` resumes with the same candidates and trace.
`search` closes with one `search_complete` (`steps`, `candidates`) and installs the root's reduction as `TraversalState::candidates`, bounded by the root cursor's `top_k` (`0` = unlimited); `max_steps > 0` caps node visits, keeps every leaf found so far (deduped, bounded by that `top_k`), records one `search_truncated` with `max_steps`, and reports the state exhausted instead of erroring.

### `progressive` (`src/retrieval/search/progressive.rs`)

`ProgressiveRetriever` (`new(root, config)`/`with_model`/`with_embedding`/`rebind`/`prepare_root`/`search`/`search_messages`/`begin`/`step`/`finish`), `ProgressiveSearchResult` (adds `method`, `selection_provenance`, `degraded_selections`), `ModelTopKSelector`, `validate_query` (`Error::Validation` naming `query`); `METHOD_PROGRESSIVE "progressive"`, `METHOD_LEXICAL_FALLBACK "lexical_fallback"`, `LLM_REQUEST_EVENT "llm_request"`, `LLM_RESPONSE_EVENT "llm_response"`.
`new` binds the five seams (`build_exposed_fragment`, `DefaultSubtreeRenderer`, `DefaultTopKSelector` or `EmbeddingBlendedTopKSelector`, `DefaultTargetExpander`, `DefaultBranchReducer`) and keeps two selectors: the primary (model-backed when a model is wired) and `fallback_selector`, a clone forced to `SelectionMode::Lexical`, so a model failure degrades to lexical selection and never re-enters the model.
`ModelTopKSelector` marks `LexicalAfterModelError` when the completion fails (or the reply cannot be split into a prompt) and `UnusableReply` when a non-blank, non-abstaining reply yields no known code; per call it records `llm_request`, calls `CompletionModel::complete` with ONE prompt (Python's system and user contents joined with `"\n\n"`), records `llm_response`, and parses with `parse_selected_codes`.
`prepare_root` exposes every reachable node once and renders each prompt (a later search costs no rendering); `warmed_fragments` is the cache size.
`method` is `METHOD_PROGRESSIVE` when the walk recorded a `fragment_selected` with `mode = "generate"`, else `METHOD_LEXICAL_FALLBACK`; `finish` fills `selection_provenance` from `trace_selection_provenance` (`Lexical` when nothing was selected) and `degraded_selections` from `trace_degraded_selections`.

### `blend_selector` (`src/retrieval/search/blend_selector.rs`)

`EmbeddingBlendedTopKSelector` (`new(config, Arc<dyn Embedding>)`/`select`, async), `MODE_EMBEDDING_BLEND = "embedding_blend"`.
The Rust-only third path: the lexical ranking as backbone nudged by `blend_scores`, then the config's truncation rules and the cursor's `top_k` (`0` = unlimited); it never calls a model, never abstains, records one `fragment_selected` with the same six detail keys and `EmbeddingBlend`, and skips the embedder when the weight is `0.0`.

### `flat` (`src/retrieval/search/flat.rs`)

`FlatRetriever` (`new`/`with_embedding`/`config`/`method`/`selection_provenance`/`retrieve`/`retrieve_choices`), `FlatSearchResult` { candidates, trace, method, ranked_keys, selection_provenance }; `METHOD_FLAT "flat"`, `METHOD_FLAT_LEXICAL "flat_lexical"`, `FLAT_ROOT_NODE_ID "flat_root"`, plus `fragment_built`/`reduce_complete`/`search_complete` under the engine's names.
One call, one bounded ranked list: a single deterministic pass over the flattened catalog (`RetrieverNode::flatten_choices`, or the caller's choices) instead of Python's recursion; documents are keyed by payload (a repeat collapses onto its first occurrence) and candidates are ranked `1..n` with `item_id`/`label` = the choice id.
Branch paths come from the tree for `retrieve` and from the payload's dotted prefixes for `retrieve_choices`; a blank query is `Error::Validation` and an empty catalog an empty result.
`method` is `METHOD_FLAT` only when an embedder is wired AND its weight is positive, else `METHOD_FLAT_LEXICAL`; `selection_provenance` mirrors it and is never degraded.

### `service` (`src/retrieval/search/service.rs`)

`Retriever` (`load(dir, config)`/`from_loaded_index(index, config)`/`with_model`/`with_embedding`/`prepare`/`search`/`search_with`/`search_details`/`search_details_with` plus the accessors `index`/`config`/`runtime_config`/`flat`/`method`/`selection_provenance`/`warmed_fragments`); `RetrieverConfig` { top_k 10, traversal, render, generation, embedding_weight 0.0, truncation } plus `TraversalConfig`/`RenderConfig`/`GenerationConfig`, `SearchRequest` { top_k } (the only per-query override), `runtime_config_from_config`, `CandidateRecord`, `RetrievalResult` (carries `selection_provenance`, `degraded_selections` and `top_k_clamped` beside the public rows).
The translator floors `top_k`, `max_tokens` and `max_branch_choices` at `1`, takes the disclosure knobs from `RenderConfig`, keeps the Rust-only disclosure bounds at their conservative defaults (the public config does not expose them), clamps `embedding_weight` into `[0, 1]` and validates.
`search_details_with` validates the query, resolves `top_k` (`None` = configured, `Some(0)` = the unlimited sentinel, else clamped), runs the walk and publicizes: resolve payloads through the catalog (no record → the candidate's own ids), dedupe by payload then worker id, re-rank from `1`, mark the first row `selected`, trim to `top_k`, and derive `item_ids`/`summary_lines`/the selected row from the trimmed rows; an over-large request is reported twice (a `warn!` naming both bounds, and `top_k_clamped = true`).
`item_ids` is the public `worker_id` per row, so `search` and `search_with` cannot disagree; `summary_lines` render `"{rank}. {id} (source={method})"` (no score, because candidates carry none).
`method`/`selection_provenance` predict the next search (`METHOD_PROGRESSIVE` + `Model` with a model in generate mode, else `METHOD_LEXICAL_FALLBACK` + `LexicalNoModel`/`EmbeddingBlend`/`Lexical`); each result carries the observed provenance (the walk's last selection, else the prediction) and `degraded_selections`.
The progressive runtime binds lazily on the first `prepare` or search behind an `RwLock` never held across an `.await`.

### `render` (`src/retrieval/search/render/`)

`SubtreeRenderer::render_subtree`, `DefaultSubtreeRenderer` (`new`/`without_messages`/`with_messages`/`renders_messages`), `DISCLOSURE_PROMPT_VERSION "symphony-disclosure-v1"`.
`expose.rs` — `build_exposed_fragment`, `exposure_item_count`, `fragmentation_fingerprint`, `FLAT_COMPACT_FIELD_ORDER ["category", "name", "raw_name", "description", "id"]`; `prompt.rs` — the Chinese prompt bank and its variants, `build_disclosure_messages`, `build_disclosure_prompt_parts`, `disclosure_system_prompt`, `disclosure_request_marker`; `parse.rs` — `parse_selected_codes`.
`DefaultSubtreeRenderer` passes the fragment and protocol through and fills in `build_disclosure_messages` for the cursor's `top_k` when messages are enabled (`without_messages` is the warmup path); it never parses a reply.
`build_exposed_fragment` walks depth-first — a node's own items first, then its children in sorted order — and that is the code-assignment order; a non-root node with no items and one child is spliced away, and a child becomes one selectable branch when the exposure depth is exhausted or its subtree exceeds `exposure_threshold` items, else it expands.
Codes are names mode (`PascalCase` boundary name from the terminal segment of the selectable canonical id, disambiguated by the branch path and finally `name__{index}`), compact-codebook mode (`CompactCodebook::prefix` positionally; short book ⇒ `Error::Validation`), or numeric `"1"`, `"2"`, …; `ABSTAIN_TOKEN` is never assigned, and `code_width` is the common length (`0` when they differ, `1` with no codes) unless `DisclosureConfig::code_width` overrides it.
Flat mode (`flatten_full_tree_in_prompt`) sorts selectable entries alphabetically by identity (ties by label), assigns codes in that order, and renders a pretty JSON array with `FLAT_COMPACT_FIELD_ORDER`, else an `- <name>: <description>` list; descriptions are sanitized (`REPRESENTATIVE_DESCENDANTS` stripped) and clamped by `max_description_chars` at a `。 . ； ;` in the budget's second half, else cut with `...`.
`max_items_per_branch` withholds items from the prompt line only (a `tracing` record names the count) and `max_chars` truncates into `truncated`/`rendered_chars`; every slice is character-boundary safe.
The bank is a shared preamble (`SHARED`, or `SHARED_TOP2` when `top_k == 2`) plus a structure and an output block selected by `(compact, flat)` — codes/tree, codes/flat, names/flat, names/tree — with the four `{…}` placeholders substituted in one pass; names-plus-flat elsewhere uses `LEGACY_FLAT_NON_COMPACT_SYSTEM_PROMPT` (fixes `top_k = 5`).
`build_disclosure_messages` returns exactly two messages: a `system` one with the bank plus the candidate block (`# 候选列表` and `<CANDIDATE_TREE>…</CANDIDATE_TREE>`, or `Available options:` in flat mode) and a `user` one with the request marker and the query inside `<USER_REQUEST>…`; the marker is `[req:<16 hex>]` over the SHA-256 of the query, fingerprint and `top_k` instead of Python's `uuid4`, so identical inputs give byte-identical prompts.
`parse_selected_codes` is a tolerant cascade with no error path: normalize each line (trim, backticks, one list marker), then match the code-handle shapes (JSON `id`/`c`/`code`, the line as a code, a `[XX]` bracket prefix, `[id: XX]`, an all-known-code token list, a `Candidate`/`id:`/`c:`/`code:` prefix, a longest-first prefix match requiring a delimiter); names mode also accepts a case-insensitive code, a unique label, a unique `camelCase`-token overlap on `display_name`/`label`, a numeric code, a 1-based position and a code list (compact mode skips the last two), plus a joined-code fallback for one run of fixed-width codes.
Line order is preserved, each `canonical_id` is emitted once, and garbage output yields an empty vector, never an error.

## Constraints

The crate document owns the shared rules (determinism, no `unsafe`, serde shape, errors, doc comments); module-specific:

- Only std logic plus the crate's own dependencies (no regex engine, clock or RNG); errors are `crate::error::{validation, unsupported}` and name the offending field; `select_topk` is `async` and no `std::sync` guard is held across an `.await`.
- Documented deviations, not silent ones: the matcher is a bounded subset, the code book is generated, `search_terms` is not ported, `casefold()` is approximated by `to_lowercase()`, and the model call takes one flattened prompt.

## Deferred Scope

- `logit`/vLLM branch selection (`selection_mode="logit_selection"`) and the `transformers`/vLLM scoring backends: `CompactCodebook`/`CodeTrie` IS ported; the backends are `Error::Unsupported` in `../../deferred.rs`.
- Python's `LexicalIndex.search_terms` term families: every family here is the token plus its own derivations (`missing_terms`).
- The legacy `progressive.py` direct recursion (`_search_node`, `_continue_from_resolutions`, `_select_items*`): `FlatRetriever` covers that shape in one pass.
- Branch parallelism: children are visited sequentially; the two config fields are carried for parity only.
- The fixed-prefix-cache request rejection: no such backend exists, so a larger request is clamped (`top_k_clamped` plus a warn).
- S3/HTTP index materialization, `service/serving.py`, Python's per-query LLM client cache and `Retriever.close()`: the index is a local directory and the model/embedder are `Arc`s

## Known Issues

- `selector.rs` (~1290), `lexical.rs` (~1584) and `render/expose.rs` (~1566) exceed the 1000-line soft cap only because their in-file `#[cfg(test)]` suites stay beside the logic; split by responsibility, never for size.
- Two `SearchCursor` types exist: `retrieval::types::SearchCursor` (the build side's pre-order plan) and `search::flow::SearchCursor` (the runtime's one-node cursor). Do not confuse them.
- `ProgressiveSearchConfig::default()` is Python's production disclosure shape, so depth/threshold only matter when a caller overrides them.
- Trace detail maps are not byte-stable across a key addition: `fragment_selected` carries `provenance`/`degraded` on top of `mode`/`raw_output`/`selected_canonical_ids`/`selected_codes`.

## Error Strategy

- `Error::Validation` — blank query, illegal compact code book (blank, whitespace, duplicate, `"0"`), short book, unusable `ProgressiveSearchConfig`, missing root node id, corrupt index artifact.
- `Error::NotFound` — a missing index artifact, and a cursor whose `node_id` is not in the bound tree.
- `Error::Unsupported` — a selection mode needing a model with none wired under `FallbackMode::Error`, and a subtree provider without a `FragmentBuildFn`.
- `Error::Llm` — the model path's own failures, surfaced only under `FallbackMode::Error`.
- `Error::Retrieval` — the embedder's backend failures (including "no vector returned"), propagated and never swallowed.
- A `FallbackMode::Deterministic` degradation is never an error: it is a `SelectionProvenance` on the result, a `mode`/`provenance`/`degraded` triple on the trace event, and one `warn!`.
- Parse and truncation have no error channel; `Error::Graph` is not produced here.

## Test Strategy

- Unit tests beside modules: tokenizer/matcher rules, BM25/field/phrase arithmetic, truncation bounds, code assignment and flat rendering, prompt variants, every parser shape, the exposure cache and generation invariant, expansion budgets, reduction order, engine step order and resumability, selector fallback order and provenance, config translation.
- `selector` — the no-model degradation is marked and warns exactly once per instance (shared across clones), a failing generate function is `LexicalAfterModelError`, an unusable reply is marked and selects nothing, `SelectionMode::Lexical` stays undegraded, and the trace helpers classify events with or without the provenance keys.
- `../../../tests/retrieval_search.rs` — the facade and everything under it: loading/publicization, corrupt/missing artifacts, the scripted-model walk, the lexical fallback, `FallbackMode::Error` without a model, the pinned trace sequence and detail keys, `prepare` warming, a resumed walk versus an uninterrupted search, the three walk bounds, the flat retriever and its blend, the per-request `top_k`, identical reruns, public identity (worker ids, never cids).
  `case_18`–`case_22` pin provenance: generate-without-a-model degrades to the identical ranking but records `MODE_LEXICAL_NO_MODEL`/`LexicalNoModel` with `degraded_selections ≥ 1`; `FallbackMode::Error` still answers the typed `Unsupported`; `Lexical` is never degraded; the fields round-trip and an old payload parses as `Lexical`; the degraded walk renders a byte-identical disclosure; the `top_k` clamp is reported.
- `../../../tests/retrieval_search_lexical.rs` (foundation layer on a hand-made `LoadedIndex`) and `../../../tests/retrieval_search_disclosure.rs` (render layer): documents and field ranking, phrase/alternative rules, `search_in` scoping, blank queries, the empty index, hand-computed truncation; then exposure order with pinned codes and text, positional compact codes and the short-book error, five reply shapes recovering one selection, the two-message prompt, multibyte truncation, byte-identical reruns, flat mode's order and fields, fingerprint stability.
- Fixtures are offline (`StaticCompletionModel`, `HashEmbedding`, `tempfile`); gates are `cargo fmt`, `cargo test` and `cargo clippy --all-targets --all-features -- -D warnings`.

## Routing Table

- `./mod.rs` → module docs, declarations, re-exports, `RUNTIME_TOP_K`
- `./flow.rs` → flow value objects, `SelectionProvenance`, `SelectionResult`, `ABSTAIN_TOKEN`
- `./config.rs` → `ProgressiveSearchConfig`, `DisclosureConfig`, `EmbeddingScoringConfig`, `SelectionMode`, `DEFAULT_TOP_K`
- `./contracts.rs` → the five seams, `FragmentBuildFn`, `GenerateFn`, `SelectRequest`, `GenerateRequest`
- `./lexical.rs` → `LexicalIndex`, `LexicalDocument`, `tokenize`, `QueryMatcher`, the weights, snippet helpers
- `./embedding.rs` → `EmbeddingScorer`, `cosine_similarity`, `blend_scores`
- `./codebooks.rs` → `CompactCodebook`, `CodeTrie`, `default_compact_boundary_codebook`
- `./truncation.rs` → `truncate_sorted_by_score`, the truncation configs, the char/item budgets
- `./subtree.rs` → `DefaultCurrentSubtreeProvider`, the exposure cache, the progressive-root catalog helpers
- `./selector.rs` → the two selectors, provenance helpers, `is_abstain_output`, the mode constants
- `./expand.rs` / `./reduce.rs` → `DefaultTargetExpander` (branch budget, caps) and `DefaultBranchReducer` (round-robin merge, dedupe key)
- `./engine.rs` → `RecursiveSearchEngine`, `TraversalState`, `EngineRequest`, the event constants
- `./progressive.rs` → `ProgressiveRetriever`, `ProgressiveSearchResult`, `ModelTopKSelector`, the method/event constants
- `./blend_selector.rs` → `EmbeddingBlendedTopKSelector`, `MODE_EMBEDDING_BLEND`
- `./flat.rs` → `FlatRetriever`, `FlatSearchResult`, `FLAT_ROOT_NODE_ID`, the flat method constants
- `./service.rs` → `Retriever`, `RetrieverConfig` + sub-configs, `SearchRequest`, `CandidateRecord`, `RetrievalResult`
- `./render/` → `mod.rs` (`SubtreeRenderer`, `DefaultSubtreeRenderer`), `expose.rs` (exposure walking, code assignment, flat rendering, budgets), `prompt.rs` (the bank, `build_disclosure_messages`, the request marker), `parse.rs` (`parse_selected_codes`)
- `../../../tests/retrieval_search.rs`, `../../../tests/retrieval_search_lexical.rs`, `../../../tests/retrieval_search_disclosure.rs` → the three crate-level suites

## Notes for Agents

- A new fallback path must state itself: set a `SelectionProvenance` via `with_provenance`, let `record_selection` copy it into the trace event, and add a row to the degradation table.
- Keep `SelectionProvenance`'s serde default `Lexical` and keep new fields `#[serde(default)]`, so older `SelectionResult`/`RetrievalResult` payloads keep parsing.
- Publicization is the only place that maps internal ids to public ones; a new `RetrievalResult` field must come from the trimmed `CandidateRecord` rows so `search` and `search_details` cannot diverge.
- `top_k == 0` is the unlimited sentinel everywhere here; a new stage that slices by `top_k` must handle it.
- Prompt text belongs in `render/prompt.rs`, is intentionally Chinese (the reference's text), and keeps its `{…}` placeholders substituted in one pass.

