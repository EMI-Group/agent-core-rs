# search — query-time (online) capability retrieval

## Intent

The query-time half of Python `openjiuwen.symphony.retrieval` (v0.1.20): `retrieval/search/runtime/**` (value objects, config, lexical scoring, codebooks, truncation, disclosure rendering, the traversal protocols, the engine, the progressive and flat retrievers) plus `service/models.py` and `service/retriever.py` (the public configs, the config translator and the `Retriever` facade).
It loads a prebuilt index — nothing here builds, mutates or persists one — and answers a query with public skill names by walking the taxonomy or by ranking a flattened catalog in one pass.
Everything is offline and deterministic: no network, no clock, no RNG, no hash-map iteration order in any output, and every model-facing stage has a documented model-free path.
The public shape is [`Retriever`](crate::retrieval::search::service::Retriever), which validates a query, resolves `top_k`, runs the walk and publicizes the raw candidates into [`RetrievalResult`](crate::retrieval::search::service::RetrievalResult) rows; everything below it is engine detail.
The model is optional, unlike Python: [`SelectionMode::Lexical`](crate::retrieval::search::config::SelectionMode) and [`FallbackMode::Deterministic`](crate::llm::FallbackMode) make a model-free deployment a supported configuration.

### Pipeline

1. **Load** — `retrieval::artifacts::load_index(dir)` returns the [`LoadedIndex`](crate::retrieval::types::LoadedIndex) (tree, choices, catalog records, manifest, item paths); [`Retriever::load`](crate::retrieval::search::service::Retriever::load) wraps it and [`Retriever::from_loaded_index`](crate::retrieval::search::service::Retriever::from_loaded_index) accepts one built in memory.
2. **Translate the config** — [`runtime_config_from_config`](crate::retrieval::search::service::runtime_config_from_config) flattens [`RetrieverConfig`](crate::retrieval::search::service::RetrieverConfig) into the engine's [`ProgressiveSearchConfig`](crate::retrieval::search::config::ProgressiveSearchConfig), clamping every bound.
3. **Optional embedding path** — [`Retriever::with_embedding`](crate::retrieval::search::service::Retriever::with_embedding) installs an [`EmbeddingScorer`](crate::retrieval::search::embedding::EmbeddingScorer) inside [`EmbeddingBlendedTopKSelector`](crate::retrieval::search::blend_selector::EmbeddingBlendedTopKSelector) as the progressive fallback selector and inside the flat retriever; `EmbeddingScoringConfig::weight == 0.0` (the default) reproduces the lexical order exactly.
4. **Retrieve** — one of three runtime shapes: [`ProgressiveRetriever`](crate::retrieval::search::progressive::ProgressiveRetriever) (the tree walk over [`RecursiveSearchEngine`](crate::retrieval::search::engine::RecursiveSearchEngine)), [`EmbeddingBlendedTopKSelector`](crate::retrieval::search::blend_selector::EmbeddingBlendedTopKSelector) (the blend the walk's fallback slot holds), or [`FlatRetriever`](crate::retrieval::search::flat::FlatRetriever) (the one-shot whole-catalog ranking).
5. **Expose, render, select** — [`DefaultCurrentSubtreeProvider`](crate::retrieval::search::subtree::DefaultCurrentSubtreeProvider) turns a cursor into a [`CurrentSubtree`](crate::retrieval::search::flow::CurrentSubtree) through the injected [`FragmentBuildFn`](crate::retrieval::search::contracts::FragmentBuildFn) (built on `render::build_exposed_fragment`), [`DefaultSubtreeRenderer`](crate::retrieval::search::render::DefaultSubtreeRenderer) produces the [`PromptBundle`](crate::retrieval::search::flow::PromptBundle), and a [`TopKSelector`](crate::retrieval::search::contracts::TopKSelector) produces the [`SelectionResult`](crate::retrieval::search::flow::SelectionResult) (lexical, embedding-blended, or model-generated).
6. **Expand and reduce** — [`DefaultTargetExpander`](crate::retrieval::search::expand::DefaultTargetExpander) turns selected targets into an [`ExpansionPlan`](crate::retrieval::search::flow::ExpansionPlan) of leaves and child cursors, and [`DefaultBranchReducer`](crate::retrieval::search::reduce::DefaultBranchReducer) merges a node's own leaves with its children's results into one [`NodeSearchResult`](crate::retrieval::search::flow::NodeSearchResult).
7. **Publicize** — [`Retriever::search_details_with`](crate::retrieval::search::service::Retriever::search_details_with) resolves each candidate's `payload` (a leaf `cid`) through the catalog to a [`CatalogRecord`](crate::retrieval::types::CatalogRecord) and exposes the public `worker_id`, name, description, branch path and category.
8. **Trace** — every stage records [`RetrieverTraceEvent`](crate::retrieval::types::RetrieverTraceEvent)s into one [`RetrieverTrace`](crate::retrieval::types::RetrieverTrace), which the facade copies into `RetrievalResult::trace_events`.

## API Surface

All paths crate-relative. Every public item is documented; `cargo clippy -D warnings` includes `missing_docs`.

### `flow` (`src/retrieval/search/flow.rs`)

The Python flow value objects, all `Debug + Clone + Default + PartialEq + Serialize + Deserialize + JsonSchema` with `#[serde(rename_all = "snake_case", default)]` (the `RetrieverItem`-carrying types are `PartialEq` only, because `RetrieverItem` is not `Eq`).

```rust
pub const ABSTAIN_TOKEN: &str = "0";                 // reserved, never a candidate code

pub struct ChatMessage { pub role: String, pub content: String }   // new / system / user
pub struct SearchCursor { pub node_id: String, pub depth: usize, pub branch_path: Vec<String>, pub top_k: usize }
// new(&RetrieverNode, top_k) seeds node_id/branch_path from the root; at(..); resolve(&root) -> Option<&RetrieverNode>
pub struct ExposedNode { pub canonical_id, label, description, is_selectable,
                         selectable_canonical_id: Option<String>, children: Vec<ExposedNode> }
pub struct SelectableResolution { pub code, canonical_id, display_name, label, description, is_terminal,
                                  branch_path: Vec<String>, node_id: Option<String>, item: Option<RetrieverItem> }
pub struct ExposedFragment { pub root: ExposedNode, rendered_tree: String,
                             code_to_resolution: BTreeMap<String, SelectableResolution>,
                             candidate_codes: Vec<String>, fragment_fingerprint: String, code_width: usize,
                             compact_codes_enabled: bool, flat_list_mode: bool, rendered_chars: usize, truncated: bool }
// resolution(code) / resolutions() (candidate-code order) / selectable_count() / is_empty()
pub struct SelectableTarget { pub resolution: SelectableResolution }   // new / is_terminal / branch_path / code / canonical_id
pub struct CurrentSubtree { pub cursor: SearchCursor, pub fragment: ExposedFragment, pub selectable_targets: Vec<SelectableTarget> }
pub struct SelectionProtocol { pub compact_codes_enabled: bool, pub candidate_codes: Vec<String>, pub code_width: usize, pub abstain_token: String }
pub struct PromptBundle { pub fragment: ExposedFragment, pub protocol: SelectionProtocol, pub messages: Vec<ChatMessage> }
pub struct SelectionResult { pub raw_output: String, pub selected_targets: Vec<SelectableTarget>, pub is_abstain: bool }
pub struct ChildSearchCursor { pub cursor: SearchCursor, pub target: SelectableTarget }
pub struct ExpansionPlan { pub leaf_results: Vec<RetrieverCandidate>, pub child_cursors: Vec<ChildSearchCursor> }
pub struct NodeSearchResult { pub candidates: Vec<RetrieverCandidate> }
```

`SelectionProtocol::default().abstain_token == ABSTAIN_TOKEN` (a custom `Default`).
`SearchCursor::node_id`/`SelectableResolution::node_id` replace Python's embedded node objects, so both are serializable and cheap to copy.
`ExposedFragment` adds the Rust-only `rendered_chars`/`truncated` bookkeeping and drops Python's `selectable_nodes`/`score_key`/`token_id`, which no ported stage reads.
All collections are `Vec`/`BTreeMap`, so serialization order is deterministic.

### `config` (`src/retrieval/search/config.rs`)

```rust
pub const DEFAULT_TOP_K: usize = 10;                 // Python RetrieverConfig.top_k (the facade's default)
pub enum SelectionMode { Generate, Lexical }         // Lexical is Rust-only: rank without a model

pub struct DisclosureConfig {
    pub max_exposure_depth_per_call: usize,  // 2
    pub exposure_threshold: usize,           // 12
    pub compact_codes_enabled: bool,         // false
    pub compact_codebook: Vec<String>,       // empty => numeric codes "1", "2", ...
    pub flatten_full_tree_in_prompt: bool,   // false
    pub max_chars: usize,                    // Rust-only rendered-fragment budget; 0 = unlimited
    pub max_items_per_branch: usize,         // Rust-only per-node item cap; 0 = unlimited
    pub max_description_chars: usize,        // Rust-only description clamp; 150; 0 = unlimited
    pub code_width: usize,                   // Rust-only width override; 0 = derive from the assigned codes
}                                            // with_* builders + validate()

pub struct EmbeddingScoringConfig { pub weight: f64, pub max_documents: usize }   // 0.0 / 0
// with_weight clamps into [0, 1] (a non-finite weight becomes 0.0) and documents the clamp; with_max_documents; validate()

pub struct ProgressiveSearchConfig {
    pub top_k: usize,               // 5
    pub max_tokens: usize,          // 96
    pub max_depth: usize,           // Rust-only hard depth bound; 0 = unlimited
    pub max_steps: usize,           // Rust-only engine-visit bound; 0 = unlimited
    pub max_branch_choices: usize,  // 6
    pub branch_choice_slack: usize, // 2
    pub branch_candidate_slack: usize, // 1
    pub round_robin_branch_reduce: bool, // true
    pub selection_mode: SelectionMode,   // Generate
    pub fallback_mode: crate::llm::FallbackMode, // Deterministic
    pub disclosure: DisclosureConfig,
    pub embedding: EmbeddingScoringConfig,
    pub truncation: ScoreTruncationConfig,
}                                    // validate() rejects top_k == 0, a bad codebook, contradicting bounds
pub const RUNTIME_TOP_K: usize = 5;  // in `mod.rs`
```

`ProgressiveSearchConfig::default()` keeps the runtime defaults (`top_k = 5`, `max_tokens = 96`, the two slacks, `round_robin_branch_reduce`) but applies the disclosure overrides Python's facade applies: `max_exposure_depth_per_call = 99`, `exposure_threshold = 1_000_000_000`, `compact_codes_enabled = true`, `flatten_full_tree_in_prompt = true`.
`DisclosureConfig::default()` keeps the conservative Python `ProgressiveRetrieverConfig` values (2 / 12 / false / empty / false) plus the Rust-only budgets `max_description_chars = 150` and `0` for the two "unlimited" caps.
The crate therefore has ONE engine default — the production shape — and a caller wanting a conservative engine starts from `DisclosureConfig::default()`.

### `lexical` (`src/retrieval/search/lexical.rs`)

Ports Python `runtime/lexical.py`; the module docs carry the full formula and the deviation list.

```rust
pub const BM25_K1: f64 = 1.5;
pub const BM25_B: f64 = 0.72;
pub const FIELD_WEIGHTS: [f64; 6] = [5.0, 5.0, 5.0, 3.4, 0.35, 0.1];  // key, name, aliases, description, body, category
pub const IDENTITY_BONUS: f64 = 16.0;
pub const PHRASE_FIELD_WEIGHTS: [f64; 4] = [12.0, 5.0, 0.5, 0.1];     // identity, description, body, category
pub const FREQUENCY_FIELD_WEIGHTS: [usize; 5] = [5, 5, 5, 4, 1];      // corpus frequency map
pub const BRANCH_DOCUMENT_PREFIX: &str = "node::";
pub const MAX_QUERY_CHARS: usize = 512;

pub struct LexicalDocument { pub key, name, description, body, aliases: Vec<String>, category }
pub struct LexicalHit { pub key: String, pub score: f64 }
pub struct LexicalIndex;   // build / from_loaded_index / search / search_in / missing_terms /
                           // match_snippet / document / len / is_empty
pub fn tokenize(text: &str) -> Vec<String>;
pub fn normalize_identity(value: &str) -> String;
pub fn without_front_matter(body: &str) -> String;
pub struct QueryMatcher;   // matches / is_literal / alternatives
pub fn compile_matcher(query: &str, case_insensitive: bool, fixed_strings: bool) -> Result<QueryMatcher>;
```

`score = bm25(terms) + field_score(terms) + phrase_score(raw_query)`.
`bm25` sums `idf² · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))` over the distinct query terms present (`idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, `k1 = 1.5`, `b = 0.72`) and then adds the strongest single `idf²` once.
`field_score` adds `Σ_f w_f · |terms ∩ field_f| / |terms|` over `[5, 5, 5, 3.4, 0.35, 0.1]` plus `IDENTITY_BONUS · |terms ∩ (key ∪ name ∪ aliases)| / |key ∪ name ∪ aliases|` with `IDENTITY_BONUS = 16`.
`phrase_score` adds `12 / 5 / 0.5 / 0.1` once per `|`-separated alternative for a substring hit in the identity fields, description, body and category (first hit wins).
Ranking tuple: identity tier (`0` key, `1` name, `2` alias, `3` prefix of one of those, `4` otherwise), then descending score (`f64::total_cmp`), then the case-folded key, then the key.
`from_loaded_index` indexes one document per catalog record (`key = cid`, body = `retrieval_text`, category = `branch_path.join(" > ")`) plus one per tree branch node (`key = node::<node_id>`).
`search_in` scopes the ranking and treats an **empty** `keys` slice as "the whole corpus"; unknown and duplicated keys are ignored.

### `embedding` (`src/retrieval/search/embedding.rs`)

Rust-only, additive over the lexical scorer.

```rust
pub struct EmbeddingScorer;   // new(Arc<dyn Embedding>) / index / index_capped / upsert / remove / clear /
                              // scores / search / len / is_empty
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64;                          // 0.0 for degenerate inputs
pub fn blend_scores(lexical: &[LexicalHit], embedding: &BTreeMap<String, f64>, weight: f64) -> Vec<LexicalHit>;
```

`blend_scores` clamps `weight` into `[0, 1]` (a non-finite weight becomes `0.0`) and computes `blended = (1 - w) · lex/max_lex + w · (cos + 1)/2`, unioning both key sets (a key missing on one side scores `0.0` there).
`weight == 0.0` therefore reproduces the lexical ordering exactly and `weight == 1.0` the embedding ordering; the default `EmbeddingScoringConfig::weight` is `0.0`, i.e. pure lexical.
`index_capped` honours `EmbeddingScoringConfig::max_documents` (`0` = unlimited) in key order; `search` returns the descending order with ties on the key and `top_k == 0` meaning unlimited.
No `std::sync` guard is held across an `.await`: the scorer snapshots its vector map into a `Vec`, drops the guard, then embeds.

### `codebooks` (`src/retrieval/search/codebooks.rs`)

```rust
pub const DEFAULT_COMPACT_CODEBOOK_SIZE: usize = 512;
pub struct CompactCodebook;   // new / codes / len / is_empty / code / prefix / trie / from_value
pub struct CodeTrie;          // new / is_valid_prefix / is_code / completions / len / is_empty
pub fn default_compact_boundary_codebook() -> CompactCodebook;
```

`CompactCodebook::new` trims each entry and rejects a blank, whitespace-containing, duplicated or `"0"` code with `Error::Validation` naming the code; `prefix(count)` is `Error::Validation` when the book is shorter than `count`; `from_value` accepts a JSON list or an object holding one.
`CodeTrie` is the prefix-validation core a constrained decoder needs (is this still a prefix of a code? is this a whole code?); `completions` is sorted.
`default_compact_boundary_codebook` generates the first 512 of `AA, AB, … ZZ` positionally — a documented deviation from Python's qwen-tokenizer-tuned permutation, because no tokenizer exists in this workspace and single-token-ness cannot be verified.

### `truncation` (`src/retrieval/search/truncation.rs`)

```rust
pub struct ScoreTruncationConfig { pub min_score: Option<f64>, pub relative_min_score: Option<f64>,
                                   pub autocut_jumps: usize, pub autocut_min_relative_drop: f64 }
pub struct ScoreTruncationDecision { pub initial_count, kept_count: usize, pub cutoff_index: Option<usize>,
                                     pub reason: String, pub best_score: Option<f64>,
                                     pub min_score: Option<f64>, pub relative_min_score: Option<f64>,
                                     pub autocut_jumps: usize, pub autocut_min_relative_drop: f64 }
pub const REASON_MIN_SCORE: &str = "min_score";
pub const REASON_RELATIVE_MIN_SCORE: &str = "relative_min_score";
pub const REASON_AUTOCUT: &str = "autocut";
pub fn truncate_sorted_by_score<T: Clone>(items: &[T], score: impl Fn(&T) -> f64,
                                          config: &ScoreTruncationConfig) -> (Vec<T>, ScoreTruncationDecision);
pub fn truncate_chars(text: &str, max_chars: usize) -> (String, bool);
pub fn truncate_chars_with_ellipsis(text: &str, max_chars: usize) -> (String, bool);
pub fn take_budget<T>(items: Vec<T>, max_items: usize) -> (Vec<T>, bool);
```

The decision order is the reference's — absolute `min_score`, then `relative_min_score`, then the `autocut` gap — and each rule may only *lower* the cutoff, so the strictest rule wins and `reason` names the one that moved it.
`relative_min_score` applies only when the best score is positive; `autocut` compares each adjacent pair, ignores non-positive drops and drops below `autocut_min_relative_drop`, and cuts after the `autocut_jumps`-th qualifying drop.
Empty input is `kept_count = 0`, `cutoff_index = None`, `reason = ""`; the input is never reordered.
The budget helpers treat `0` as "unlimited", are character-boundary safe, and the ellipsis counts inside the `max_chars` budget.

### `contracts` (`src/retrieval/search/contracts.rs`)

```rust
pub type FragmentBuildFn = Arc<dyn Fn(&RetrieverNode, &[String], &DisclosureConfig, usize) -> Result<ExposedFragment> + Send + Sync>;
pub trait CurrentSubtreeProvider { fn get_current_subtree(&self, cursor: &SearchCursor) -> Result<CurrentSubtree>; }
pub trait TopKSelector {
    fn build_protocol(&self, subtree: &CurrentSubtree) -> SelectionProtocol;
    async fn select_topk(&self, request: SelectRequest<'_>, trace: &mut RetrieverTrace) -> Result<SelectionResult>;
}
pub trait TargetExpander { fn expand_selected_targets(&self, cursor: &SearchCursor, selected: &[SelectableTarget]) -> ExpansionPlan; }
pub trait BranchReducer { fn reduce_branch_results(&self, cursor: &SearchCursor, local_leaves: &[RetrieverCandidate],
                                                   child_results: &[NodeSearchResult]) -> NodeSearchResult; }
pub struct SelectRequest<'a> { pub model: Option<&'a str>, pub cursor, pub query_messages, pub subtree, pub prompt }
pub type GenerateFn = Arc<dyn Fn(GenerateRequest<'_>) -> Result<(String, Vec<SelectableResolution>)> + Send + Sync>;
pub struct GenerateRequest<'a> { pub model, node_id, depth, top_k, query_messages, fragment, messages, trace }
pub struct GenerateFragmentSelector;   // new(GenerateFn) / generate(GenerateRequest) -> (raw output, resolutions)
```

The five seams are [`CurrentSubtreeProvider`], [`SubtreeRenderer`](crate::retrieval::search::render::SubtreeRenderer), [`TopKSelector`], [`TargetExpander`] and [`BranchReducer`]; the engine decides only *when* to call them.
Rendering is the crate's documented structural deviation: Python's provider calls `render.disclosure.build_exposed_fragment` directly, while here the provider takes a [`FragmentBuildFn`] and answers `Error::Unsupported` until one is supplied, so the traversal layer compiles and runs with no renderer at all.
[`SubtreeRenderer`](crate::retrieval::search::render::SubtreeRenderer) therefore lives with its default implementation in `render/` rather than here, and `TopKSelector::select_topk` is `async` because the model path awaits a completion.
[`SelectRequest`]/[`GenerateRequest`] borrow their inputs and carry `&mut RetrieverTrace`, so they are deliberately not `serde` types; the values they point at are.

### `subtree` (`src/retrieval/search/subtree.rs`)

```rust
pub type ItemCounterFn = Arc<dyn Fn(&RetrieverNode) -> usize + Send + Sync>;
pub struct DefaultCurrentSubtreeProvider;   // new(root, config) / with_builder(root, config, FragmentBuildFn) /
                                            // with_item_counter / rebind(root) / clear_cache / cached_len /
                                            // generation / evictions
pub fn choices_cache_key(choices: &[RetrieverChoice]) -> String;
pub fn build_progressive_root(choices: &[RetrieverChoice]) -> RetrieverNode;
pub fn build_progressive_item_label(choice_id: &str, payload: &str) -> String;
pub fn build_progressive_branch_description(branch_path: &[String]) -> String;
```

The exposure cache is keyed by `(generation, node_id, branch_path, top_k)`; `generation` belongs to the binding (one immutable tree plus a counter), and `rebind` swaps the tree and bumps the generation under one write lock before clearing the cache, so a stale fragment can never be served even if a reader inserts an old-generation entry afterwards (`clear_cache` frees it, `evictions` counts it).
A cache hit pairs the cached fragment and targets with the caller's own cursor, so the emitted `CurrentSubtree` always describes the request that asked for it.
`new` without `with_builder` is usable for planning and catalog helpers but `get_current_subtree` answers `Error::Unsupported`; the provider is synchronous and its two `std::sync::RwLock`s are never held across an `.await`.

### `selector` (`src/retrieval/search/selector.rs`)

```rust
pub const FRAGMENT_SELECTED_EVENT: &str = "fragment_selected";
pub const MODE_LEXICAL: &str = "lexical_fallback";
pub const MODE_GENERATE: &str = "generate";
pub const ABSTAIN_HINTS: &[&str];                    // English + Chinese "nothing relevant" phrasings
pub fn is_abstain_output(output: &str) -> bool;      // blank is NOT an abstention; "0" and hints are
pub fn query_text(query_messages: &[ChatMessage]) -> String;   // every content joined with "\n"
pub struct LexicalTopKSelector;   // new(config) / select(&subtree, &messages, top_k) -> SelectionResult (sync)
pub struct DefaultTopKSelector;   // new(config) / with_generate_selector(GenerateFragmentSelector) / is_model_backed
```

Selection order: `SelectionMode::Lexical` always uses [`LexicalTopKSelector`]; `SelectionMode::Generate` with a generate selector calls it and honours [`FallbackMode`](crate::llm::FallbackMode) (`Error` surfaces the failure, `Deterministic` warns and degrades to the lexical selector); `Generate` without a generate selector selects lexically under `Deterministic` and answers `Error::Unsupported` under `Error`.
Every selection records exactly one `fragment_selected` event with the detail keys `mode`, `raw_output`, `selected_canonical_ids` and `selected_codes`.
The lexical selector builds one `LexicalDocument` per selectable target (key = canonical id, name = display name, category = the branch path joined with `" > "`), ranks the concatenated query through `ranking_expression` (a `|`-separated alternative list, so a whole conversation does not gate every candidate out), applies the config's truncation rules, and never marks itself as an abstention.
A blank query selects nothing and an unusable query expression is warned, not fatal; `top_k == 0` means unlimited.

### `expand` (`src/retrieval/search/expand.rs`) and `reduce` (`src/retrieval/search/reduce.rs`)

```rust
pub struct DefaultTargetExpander;   // new(config) / branch_top_k(top_k, branch_count)
pub struct DefaultBranchReducer;    // new(config) / merge_branch_candidates(&[Vec<RetrieverCandidate>], top_k)
```

The expander emits terminal resolutions as leaf candidates in declared selection order (rank `1`, the reduce stage re-ranks nothing) and branch resolutions as [`ChildSearchCursor`]s one level deeper.
The per-branch budget is `min(top_k, ceil(top_k / max(1, branch_count)) + branch_candidate_slack)` floored at `1`, except that `top_k == 0` (unlimited) yields an unlimited budget (`0`) instead of Python's `1`.
`max_branch_choices > 0` keeps only the first N child cursors in declared order, and `max_depth > 0` stops descending at `cursor.depth >= max_depth` (leaves selected at the deepest allowed node are still emitted); both bounds are configuration, so nothing is logged per call.
The reducer merges `[local_leaves, child_results…]` round-robin (one candidate per branch, then the next index across every branch), deduping by `payload` and falling back to `item_id` when the payload is blank, then runs a sequential append pass over the concatenated branches when the round-robin stalls.
`round_robin_branch_reduce = false` switches to concatenate-then-dedupe (Python's legacy order), and `top_k == 0` consumes every branch where Python's `[:top_k]` slicing would keep nothing.

### `engine` (`src/retrieval/search/engine.rs`)

```rust
pub const FRAGMENT_BUILT_EVENT: &str = "fragment_built";
pub const FRAGMENT_CONTINUE_EVENT: &str = "fragment_continue";
pub const REDUCE_COMPLETE_EVENT: &str = "reduce_complete";
pub const SEARCH_COMPLETE_EVENT: &str = "search_complete";
pub const SEARCH_TRUNCATED_EVENT: &str = "search_truncated";
pub const MODE_SINGLE_SELECTABLE_SHORTCUT: &str = "single_selectable_shortcut";

pub struct EngineRequest<'a> { pub model: Option<&'a str>, pub query_messages: &'a [ChatMessage] }
pub struct TraversalState;   // new(cursor) / with_query_messages / query_messages / root_cursor / is_exhausted /
                             // steps / pending / candidates / is_truncated / trace / trace_mut / take_candidates
pub struct TraversalStep { pub node_id, depth, selectable_count, selected_codes: Vec<String>,
                           leaf_count, branch_count, candidates_after }
pub struct RecursiveSearchEngine;   // new(provider, renderer, selector, expander, reducer, config) / config /
                                    // search(&EngineRequest, &mut TraversalState) / step(...) -> Option<TraversalStep>
```

Per-node order: `get_current_subtree` (then a `fragment_built` event with `selectable_count` and `rendered_tree`) → no target ends the node → exactly one target takes the single-selectable shortcut (no render, no model, one `fragment_selected` event with `mode = "single_selectable_shortcut"`) → otherwise `build_protocol` → `render_subtree` → `select_topk` (an abstention or an empty selection ends the node) → `expand_selected_targets` (then `fragment_continue` with `leaf_count`, `branch_count`, `selected_terminal_ids` and `selected_branch_ids`) → child cursors pushed in reverse so the declared order is the visit order → `reduce_branch_results` (then `reduce_complete` with `input_candidates` and `output_candidates`).
The walk is an explicit frame stack, so the depth bound is a real bound with no recursion; `TraversalState` is `serde`-serializable and round-trips through `serde_json`, so a walk can be begun, stepped, saved and resumed with the same candidates and trace.
Child cursors are visited sequentially in declared order (no `ThreadPoolExecutor`), which is what makes the trace order deterministic.
`search` closes the walk with one `search_complete` event (`steps`, `candidates`), and the root's reduction becomes `TraversalState::candidates`, bounded by the root cursor's `top_k` (`0` = unlimited).
`max_steps > 0` caps node visits: the walk stops, keeps every leaf discovered so far (deduped and bounded by the root's `top_k`), records one `search_truncated` event carrying `max_steps`, and reports the state as exhausted instead of erroring.

### `progressive` (`src/retrieval/search/progressive.rs`)

```rust
pub const METHOD_PROGRESSIVE: &str = "progressive";
pub const METHOD_LEXICAL_FALLBACK: &str = "lexical_fallback";
pub const LLM_REQUEST_EVENT: &str = "llm_request";
pub const LLM_RESPONSE_EVENT: &str = "llm_response";
pub struct ProgressiveSearchResult { pub candidates: Vec<RetrieverCandidate>, pub trace: RetrieverTrace,
                                     pub method: String, pub steps: usize, pub state: TraversalState }
pub struct ProgressiveRetriever;   // new(root, config) / with_model(model, name) / with_embedding(embedding) /
                                   // rebind(root) / config / root / model_name / warmed_fragments /
                                   // prepare_root(top_k) / search(query, top_k) / search_messages(messages, top_k) /
                                   // begin(messages, top_k) -> TraversalState / step(&mut state) / finish(state)
pub struct ModelTopKSelector;      // new(model, model_name, config, fallback: Arc<dyn TopKSelector>) / model_name
pub fn validate_query(query: &str) -> Result<()>;   // Error::Validation naming `query` when blank
```

`new` binds the five seams: `DefaultCurrentSubtreeProvider` with a `FragmentBuildFn` built on `build_exposed_fragment`, `DefaultSubtreeRenderer`, `DefaultTopKSelector` (or `EmbeddingBlendedTopKSelector` after `with_embedding`), `DefaultTargetExpander` and `DefaultBranchReducer`.
With a model configured the engine's selector is `ModelTopKSelector`, which per call builds the prompt with `build_disclosure_messages` (preferring the messages the engine already rendered), records `llm_request` (`model`, `node_id`, `depth`, `prompt_chars`, `messages`), calls [`CompletionModel::complete`](crate::llm::CompletionModel) with ONE prompt — Python's rendered system and user contents joined with `"\n\n"`, the crate's documented flattening — records `llm_response` (`model`, `node_id`, `reply_chars`), and parses the reply with `parse_selected_codes`.
`prepare_root` exposes every reachable node once and renders the prompt of each selectable exposure, so a later search costs no rendering; `warmed_fragments` reports the cache size.
`method` is [`METHOD_PROGRESSIVE`] when the walk recorded at least one `fragment_selected` event with `mode = "generate"`, and [`METHOD_LEXICAL_FALLBACK`] otherwise (no model configured, or `SelectionMode::Lexical`).
`rebind` replaces the root and clears the generation-keyed exposure cache; the retriever itself is `&self`-only apart from that binding and the provider's cache.

### `blend_selector` (`src/retrieval/search/blend_selector.rs`)

```rust
pub const MODE_EMBEDDING_BLEND: &str = "embedding_blend";
pub struct EmbeddingBlendedTopKSelector;   // new(config, Arc<dyn Embedding>) / config / weight /
                                           // indexed_documents / select(&subtree, &query_messages, top_k).await
```

This is the Rust-only third selection path: it keeps the lexical ranking as its backbone and nudges it with embedding similarity through `blend_scores`, then applies the config's truncation rules and the cursor's `top_k` (`0` = unlimited).
It never calls a model and never reports an abstention, records one `fragment_selected` event with `mode = "embedding_blend"` and the same four detail keys as the other selectors, and skips the embedder entirely when the weight is `0.0`.
Documents are embedded in key order and the blend sorts on `(score, key)`, so the same catalog and query always select the same targets.

### `flat` (`src/retrieval/search/flat.rs`)

```rust
pub const METHOD_FLAT: &str = "flat";
pub const METHOD_FLAT_LEXICAL: &str = "flat_lexical";
pub const FLAT_ROOT_NODE_ID: &str = "flat_root";
pub const FRAGMENT_BUILT_EVENT / REDUCE_COMPLETE_EVENT / SEARCH_COMPLETE_EVENT: &str;   // same names as the engine
pub struct FlatSearchResult { pub candidates: Vec<RetrieverCandidate>, pub trace: RetrieverTrace,
                              pub method: String, pub ranked_keys: Vec<String> }
pub struct FlatRetriever;   // new(config) / with_embedding(embedding) / config / method /
                            // retrieve(&root, query, top_k).await / retrieve_choices(&choices, query, top_k).await
```

The one-shot retriever keeps Python's class contract (one call, one bounded ranked list, `search_complete` in the trace) but implements it as a single deterministic pass over the flattened catalog instead of the legacy `_select_items` recursion.
It flattens via `RetrieverNode::flatten_choices` (or the caller's choices), keys documents by payload so a repeated payload collapses onto its first occurrence, ranks them with `LexicalIndex`, optionally blends embeddings with the same formula and weight as the blend selector, applies the truncation rules, and emits candidates ranked `1..n` with `item_id`/`label` = the choice id.
Branch paths come from the tree for `retrieve` (the node id chain below the root) and from the payload's dotted proper prefixes for `retrieve_choices`; a choice-list trace reports `FLAT_ROOT_NODE_ID`.
`method` is `METHOD_FLAT` only when an embedder is configured AND its weight is positive, else `METHOD_FLAT_LEXICAL`; a blank query is `Error::Validation` and an empty catalog is an empty result.

### `service` (`src/retrieval/search/service.rs`)

```rust
pub use crate::retrieval::search::config::DEFAULT_TOP_K;   // 10
pub struct TraversalConfig { pub max_branch_choices, max_parallel_branches, enable_parallel_branches: bool,
                             pub branch_choice_slack, pub branch_candidate_slack, pub round_robin_branch_reduce: bool }
pub struct RenderConfig { pub compact_codes_enabled: bool, pub compact_codebook: Vec<String>, pub flatten_tree: bool,
                          pub max_exposure_depth, pub exposure_threshold }
pub struct GenerationConfig { pub mode: SelectionMode, pub max_tokens, pub trie_constrained_decoding_enabled: bool,
                              pub fallback_mode: FallbackMode }
pub struct RetrieverConfig { pub top_k, pub traversal, pub render, pub generation, pub embedding_weight: f64,
                             pub truncation: ScoreTruncationConfig }   // Default: top_k 10, weight 0.0
pub struct SearchRequest { pub top_k: Option<usize> }                  // the only per-query override
pub fn runtime_config_from_config(config: Option<&RetrieverConfig>) -> Result<ProgressiveSearchConfig>;
pub struct CandidateRecord { pub rank, item_id, worker_id, cid, name, description,
                             pub branch_path: Vec<String>, pub category: String, pub selected: bool }
pub struct RetrievalResult { pub method: String, pub item_ids: Vec<String>, pub candidates: Vec<CandidateRecord>,
                             pub summary_lines: Vec<String>, pub selected_item_id: Option<String>,
                             pub selected_rank: Option<usize>, pub trace_events: Vec<RetrieverTraceEvent>,
                             pub steps: usize }
pub struct Retriever;   // load(dir, config).await / from_loaded_index(index, config).await / with_model / with_embedding /
                        // prepare().await / search(query).await / search_with(query, &SearchRequest).await /
                        // search_details(query).await / search_details_with(query, &SearchRequest).await /
                        // index / config / runtime_config / flat / method / warmed_fragments
```

The translator floors `top_k`, `max_tokens` and `max_branch_choices` at `1`, takes the disclosure knobs from `RenderConfig`, keeps the Rust-only disclosure bounds at their conservative defaults (the public config does not expose them), clamps `embedding_weight` into `[0, 1]` and validates the result; `max_parallel_branches`, `enable_parallel_branches` and `trie_constrained_decoding_enabled` are carried for configuration parity and ignored.
`search_details_with` validates the query, resolves `top_k` (`None` = the configured value, `Some(0)` = the unlimited sentinel, anything else clamped to the configured value), runs the walk and publicizes: resolve the payload through the catalog (a payload with no record falls back to the candidate's own ids), dedupe by payload then by worker id, re-rank from `1`, mark the first row `selected`, trim to `top_k`, and derive `item_ids`, `summary_lines` and the selected row from the trimmed rows.
`item_ids` is what `search`/`search_with` return — the public `worker_id` per row, falling back to the row's own item id, so the two entry points cannot disagree; `summary_lines` render one `"{rank}. {id} (source={method})"` line per row (no `score=` component, because candidates carry no score).
`method` reports the backend the next search will use: `METHOD_PROGRESSIVE` with a model wired in generate mode, `METHOD_LEXICAL_FALLBACK` otherwise.
The progressive runtime is bound lazily on the first `prepare` or search and cached behind an `RwLock` that is never held across an `.await`; the config is translated eagerly in the constructor, so an unusable embedder is reported by the call that needs it.

### `render` (`src/retrieval/search/render/`)

```rust
pub trait SubtreeRenderer { fn render_subtree(&self, subtree: &CurrentSubtree, query_messages: &[ChatMessage],
                                              protocol: &SelectionProtocol) -> Result<PromptBundle>; }
pub struct DefaultSubtreeRenderer;   // new() / without_messages() / with_messages(bool) / renders_messages()
pub const DISCLOSURE_PROMPT_VERSION: &str = "symphony-disclosure-v1";

// expose.rs
pub const BRANCH_CANONICAL_PREFIX: &str = "node::";
pub const ITEM_CANONICAL_PREFIX: &str = "item::";
pub const TREE_INDENT: &str = "  ";
pub const FLAT_COMPACT_FIELD_ORDER: [&str; 5] = ["category", "name", "raw_name", "description", "id"];
pub const REPRESENTATIVE_DESCENDANTS: &str = "Representative descendants:";
pub fn exposure_item_count(node: &RetrieverNode) -> usize;
pub fn fragmentation_fingerprint(root: &ExposedNode, candidate_codes: &[String]) -> String;
pub fn build_exposed_fragment(root: &RetrieverNode, branch_path: &[String], config: &DisclosureConfig,
                              subtree_item_count: &dyn Fn(&RetrieverNode) -> usize) -> Result<ExposedFragment>;

// prompt.rs
pub struct DisclosurePromptParts { pub full_messages, pub prefix_messages: Vec<ChatMessage>,
                                   pub suffix_text, pub cache_id, pub prefix_token_hash: String }
pub fn build_disclosure_messages(fragment: &ExposedFragment, query_messages: &[ChatMessage],
                                 top_k: Option<usize>) -> Vec<ChatMessage>;
pub fn build_disclosure_prompt_parts(fragment: &ExposedFragment, query_messages: &[ChatMessage],
                                     top_k: Option<usize>) -> DisclosurePromptParts;
pub fn disclosure_system_prompt(fragment: &ExposedFragment, top_k: Option<usize>) -> String;
pub fn disclosure_user_prefix() -> String;
pub fn disclosure_request_marker(query_text: &str, fingerprint: &str) -> String;

// parse.rs
pub fn parse_selected_codes(fragment: &ExposedFragment, output: &str) -> Vec<SelectableResolution>;
```

`DefaultSubtreeRenderer` passes the subtree's fragment and the caller's protocol through unchanged and fills in `build_disclosure_messages` for the cursor's `top_k` when messages are enabled; the `without_messages` form is the warmup path. It never parses a reply.
`build_exposed_fragment` walks depth-first — a node's own items first, then its children in sorted order — and that is the code-assignment order.
A non-root node with no items of its own and exactly one child is spliced away, so a unary chain costs no prompt line; a child becomes one selectable branch when the remaining exposure depth is exhausted or its subtree holds more than `exposure_threshold` items, and is expanded otherwise.
Codes are names mode (a `PascalCase` boundary name from the terminal segment of the selectable canonical id, disambiguated by the branch path and finally `name__{index}`), compact-codebook mode (`CompactCodebook::prefix` positionally, so a short book is `Error::Validation`), or compact numeric codes `"1"`, `"2"`, `…`; `ABSTAIN_TOKEN` (`"0"`) is never assigned.
`code_width` is the common code length, `0` when the lengths differ, `1` when there are no codes, and an explicit `DisclosureConfig::code_width` overrides it.
`fragmentation_fingerprint` is the first 16 hex characters of the SHA-256 over the serialized exposed tree plus the joined codes, so a changed label or code changes it.
Flat mode (`flatten_full_tree_in_prompt`) sorts the selectable entries alphabetically by selectable identity (ties by label), assigns codes in that order, renders compact flat mode as a pretty JSON array with the fixed field order `category,name,raw_name,description,id`, and names flat mode as an `- <name>: <description>` list.
Descriptions are sanitized (the `Representative descendants:` remark is stripped) and clamped by `max_description_chars`, preferring a `。 . ； ;` boundary in the second half of the budget (and above a 40-character floor) and otherwise hard-cutting with `...`.
`max_items_per_branch` withholds items from the prompt line only — they stay in `code_to_resolution` and `candidate_codes`, and a `tracing` record names the count; `max_chars` truncates the rendered text and sets `truncated`/`rendered_chars`. Every slice is character-boundary safe.
The prompt bank is the reference's Chinese text: a shared preamble (`SHARED`, or the tuned `SHARED_TOP2` preamble when `top_k == 2`) plus a structure block and an output block selected by `(compact, flat)` — `(true,false)` codes/tree, `(true,true)` codes/flat, `(false,true)` names/flat, `(false,false)` names/tree — with `{top_k}`, `{structure_block}`, `{candidate_block}` and `{output_block}` substituted in one pass.
Names mode plus flat mode at any other `top_k` uses the legacy flat non-compact prompt plus the candidate block (its text fixes `top_k = 5`), and `OUTPUT_CODES_FLAT` is ported for parity but selected by no variant.
`build_disclosure_messages` returns exactly two messages: a `system` message carrying the prompt bank plus the candidate block (`# 候选列表` and `<CANDIDATE_TREE>…</CANDIDATE_TREE>`, or the `Available options:` block in flat mode) and a `user` message carrying the request marker and the query inside `<USER_REQUEST>…`.
The request marker is `[req:<16 hex>]` over the SHA-256 of the query, the fragment fingerprint and `top_k` instead of Python's fresh `uuid4`, so identical inputs produce byte-identical prompts.
`parse_selected_codes` is the tolerant cascade, with no error path: a line is normalized (trim, backticks, one list marker), then matched against the code-handle shapes (JSON `id`/`c`/`code`, the whole line as an exact code, a `[XX]` bracket prefix, `[id: XX]` handles, an all-known-code token list, a `Candidate`/`id:`/`c:`/`code:` prefix, and a longest-first prefix match requiring a delimiter).
Names mode additionally accepts a case-insensitive code match, a unique label match, a unique `camelCase`-token overlap against `display_name`/`label`, a matching numeric code, a 1-based position and a code list; compact mode deliberately skips the label and position fallbacks.
A joined-code fallback reads one run of fixed-width codes (e.g. `ABAC` for `AB`, `AC`) only when nothing parsed, compact codes are on, and the output has no newline or space.
The model's line order is preserved and each `canonical_id` is emitted once (first occurrence wins); empty, unknown or garbage output yields an empty vector and never an error.

## Constraints

- Only standard-library logic plus the crate's own dependencies; no regex engine, no clock, no RNG, no new dependency (`regex`, `chrono`, `futures` are unavailable to this crate).
- Errors are `crate::error::{validation, unsupported}` over `openjiuwen_common::error::{Error, Result}`; a validation error names the offending field (`query`, the code, the config field, …). No `unwrap`/`expect` outside tests, no panics on a malformed query, reply or document.
- `tracing` only (no `println!`/`eprintln!`), no `unsafe`, no `std::sync` guard held across an `.await`.
- Every public item carries a doc comment and a `//!` block links same-file or sibling items with an absolute `crate::retrieval::search::…` path; `cargo doc -p openjiuwen-symphony --no-deps` must stay warning-free.
- Public data types are `serde`-serializable with `snake_case` field names, tolerant of unknown fields, and `JsonSchema` where the field types allow it (the borrowed request structs and the callback aliases are the documented exceptions).
- Determinism: `BTreeMap`/`BTreeSet` everywhere in public and serialized shapes, sorted outputs, ties on the key, no clock, no RNG; the request marker is a digest for that reason.
- Files stay ≲1000 lines; `lexical.rs` (~1584 lines) and `render/expose.rs` (~1566 lines) sit over the soft cap only because their in-file `#[cfg(test)]` suites stay beside the logic — see Notes for Agents.
- Deviations from the reference are documented, not silent: the matcher is a bounded subset, the code book is generated, `LexicalIndex.search_terms` is not ported, `casefold()` is approximated by `to_lowercase()`, and the model call takes one flattened prompt.

## Design Decisions

- **The engine is composed from five seams, not one recursion.** `RecursiveSearchEngine` only decides *when* to expose, render, select, expand and reduce, so each stage is replaceable and each is unit-testable alone.
- **Rendering reaches the traversal through a callback.** `FragmentBuildFn` keeps `subtree` model-free and renderer-free, and `SubtreeRenderer` keeps prompt construction in `render/`; Python's direct `render.disclosure` call in the provider is the one structural difference.
- **Selection is async and has a model-free default.** `TopKSelector::select_topk` awaits a completion, `DefaultTopKSelector` holds an optional generate selector, and `LexicalTopKSelector` is the documented offline answer — so a walk degrades instead of failing.
- **`top_k == 0` means unlimited** in the engine, the expander, both reducers, both selectors, the flat retriever, the facade and truncation helpers; Python clamps or slices instead. This is the crate-wide sentinel.
- **The walk is an explicit stack, not recursion.** That makes `max_depth` a real bound, `TraversalState` serializable/resumable, and the trace order deterministic (children visited sequentially).
- **The exposure cache is generation-keyed.** Rebinding a tree bumps a counter inside the lock that swaps the tree, so a stale fragment is unreachable rather than merely unlikely.
- **The single-selectable shortcut skips render and model.** One candidate at a node is selected deterministically with its own `fragment_selected` mode, which is what keeps deep unique chains cheap.
- **Two search cursors, deliberately.** `flow::SearchCursor` is the runtime traversal cursor (one node plus the path that reached it); `types::SearchCursor` is the build half's resumable pre-order plan. They are unrelated types.
- **Lexical scoring is field-weighted BM25 plus an identity bonus and a phrase rule.** One rare identity term must outrank a body word repeated often, which is why the strongest `idf²` is added once and `IDENTITY_BONUS` is 16.
- **The embedding blend is optional and degenerate-safe.** `weight == 0.0` reproduces the lexical order bit for bit, so a deployment without an embedder loses nothing.
- **Compact codes are positional and validated.** `CompactCodebook` rejects blank, whitespace-bearing, duplicated and reserved-`"0"` entries, and the generated 512-entry book keeps positional uniqueness and determinism without a tokenizer.
- **Rendering is deterministic by construction.** Codes, resolutions and flat entries live in `BTreeMap`/sorted vectors, and the prompt marker is a content digest, so identical inputs give byte-identical prompts.
- **Parsing is a tolerant cascade.** Recovery of five reply shapes for one selection and an empty vector for garbage is the contract; the parser never errors.
- **The facade has two config layers and one translator.** `RetrieverConfig` is the caller's nested shape, `ProgressiveSearchConfig` is the engine's flat one, and `runtime_config_from_config` clamps between them so the engine never sees the public schema.
- **Publicization happens once, in the facade.** Internal cids never reach a caller, identities fall back rather than fail on a stale catalog, and `search` is derived from `search_details`' with the same rows.
- **Trace events are the observability contract.** `fragment_built`, `fragment_selected`, `fragment_continue`, `reduce_complete`, `search_complete`, `search_truncated`, `llm_request` and `llm_response` are pinned names with pinned detail keys.

## Deferred Scope

- `logit`/vLLM **branch selection** (`selection_mode="logit_selection"`), the `transformers`/vLLM scoring backends and their constrained tokenizer: the `CompactCodebook`/`CodeTrie` prefix-validation core IS ported, but no logit or tokenizer backend exists in this crate — that stays with [`crate::deferred`](../deferred.rs), whose entry points return `Error::Unsupported`.
- Python's `LexicalIndex.search_terms` plural/verb term-family coverage ranking: every family here is the token plus its own derivations, which `missing_terms` documents.
- The legacy direct-recursion path in `progressive.py` (`_search_node`, `_continue_from_resolutions`, `_select_items*`): `FlatRetriever` covers that shape with a one-pass ranking, and the engine path is the only traversal.
- Branch parallelism (`ThreadPoolExecutor`, `max_parallel_branches`, `enable_parallel_branches`): the walk visits children sequentially in declared order; the config fields are carried for parity and ignored.
- The fixed-prefix-cache request rejection (Python raises for a per-request `top_k` differing from the initialized one on `transformers_prefix_cached*`/`vllm`/`local_vllm` backends): there is no prefix-cache backend here, so a larger request is clamped instead.
- S3/HTTP index materialization (`artifacts.loading`'s download-then-parse path): the index is a local directory read by `artifacts::load_index`.
- `service/serving.py` (the vLLM demo script with hardcoded engine args): a dev script, not an entry point.
- `agent/retrieval_toolkit.py`, the agent-facing browsing tools over this runtime (tree browsing, subtree disclosure rendering to an agent), which are not ported.
- Python's per-query LLM client cache and `Retriever.close()`: the model and embedder are injected as `Arc`s, and dropping the facade releases everything.

## Known Issues

- `lexical.rs` (~1584 lines) and `render/expose.rs` (~1566 lines) exceed the 1000-line soft cap only because their in-file `#[cfg(test)]` suites stay beside the logic; split them for responsibility, never for size alone.
- Two `SearchCursor` types exist and must not be confused: `retrieval::types::SearchCursor` (the build side's resumable pre-order plan) and `search::flow::SearchCursor` (the runtime's one-node traversal cursor).
- `artifacts::write_index` cannot publish a zero-node index: it renders `nodes:` with no entries, which its own `parse_tree_yaml` rejects as "`nodes` is not a sequence". That is a limitation of the build half, not of this module, so a caller must publish at least one node.
- Python's production disclosure defaults (`flatten_full_tree_in_prompt = true`, `max_exposure_depth_per_call = 99`, `exposure_threshold = 1e9`) are this crate's `ProgressiveSearchConfig::default()` too, so the depth/threshold machinery is only exercised when a caller overrides them.
- `TraversalConfig::max_parallel_branches`, `TraversalConfig::enable_parallel_branches` and `GenerationConfig::trie_constrained_decoding_enabled` change nothing; they exist so a Python config round-trips.

## Error Strategy

- `Error::Validation` — malformed input: a blank query, an illegal compact code book (blank, whitespace, duplicate, `"0"`), a book shorter than the exposure, an unusable `ProgressiveSearchConfig` (`top_k == 0`, out-of-range embedding weight, contradicting truncation bounds), a missing root node id, a corrupt index artifact.
- `Error::NotFound` — an explicitly requested index artifact that is missing, and a cursor whose `node_id` is not in the bound tree.
- `Error::Unsupported` — a selection mode that needs a model with none wired under `FallbackMode::Error`, and a subtree provider constructed without a `FragmentBuildFn`.
- `Error::Llm` — the model path's own failures, surfaced only under `FallbackMode::Error`; the deterministic mode warns and degrades to the fallback selector instead.
- `Error::Retrieval` — the embedder's backend failures (including "no vector returned") propagated from `openjiuwen-retrieval`; never swallowed.
- `Error::Graph` — not produced here; `retrieval/build` owns that path.
- Parse and truncation paths have no error channel by design: a garbage reply yields no selection, and a bad record is warned and skipped.

## Test Strategy

- Unit tests live beside modules (`#[cfg(test)] mod tests`) and cover the helpers: tokenizer and matcher rules, BM25/field/phrase arithmetic, truncation bounds, code assignment and flat rendering, prompt variant selection, every parser shape, the exposure cache and generation invariant, expansion budgets, reduction order, engine step order and resumability, selector fallback order, and the facade's config translation.
- `tests/retrieval_search.rs` — the end-to-end service facade and the layers under it: index loading and publicization, corrupt/missing artifacts, the scripted-model progressive path, the lexical fallback, `FallbackMode::Error` without a model, an unusable reply, the pinned trace sequence and detail keys, `prepare` warming, a resumed walk versus an uninterrupted search, the `max_steps`/`max_depth`/`max_branch_choices` bounds, the flat retriever and its blend, field-weighted ranking with a bounded CJK snippet, the per-request `top_k`, identical reruns, public identity (worker ids, never cids), the config translation, and the accessors.
- `tests/retrieval_search_lexical.rs` — the foundation layer built on a hand-made `LoadedIndex`: item and branch documents, field-weighted ranking, phrase and alternative rules, the CJK snippet, `search_in` scoping, blank queries and the empty index, order-independent determinism, hand-computed truncation decisions and budgets, and composition of the foundation types.
- `tests/retrieval_search_disclosure.rs` — the render layer: nested exposure order with pinned codes and text, positional compact codes and the short-book validation error, five reply shapes recovering one selection, the two-message prompt naming every candidate once, character-boundary truncation of a multibyte budget, byte-identical reruns, the warmup renderer, flat mode's alphabetical order and fixed JSON fields, and fingerprint stability/sensitivity.
- Everything is offline and deterministic: `StaticCompletionModel` for scripted replies, `HashEmbedding` for the blend, `tempfile` for index directories, no network, no clock, no hash-map-order assertions.
- `cargo fmt -p openjiuwen-symphony`, `cargo test -p openjiuwen-symphony`, and `cargo clippy -p openjiuwen-symphony --all-targets --all-features -- -D warnings` must pass.

## Routing Table

- `./mod.rs` → module docs, module declarations, the re-exports, `RUNTIME_TOP_K`
- `./flow.rs` → the flow value objects (`ChatMessage`, `SearchCursor`, `ExposedNode`, `ExposedFragment`, `SelectableResolution`, `SelectableTarget`, `CurrentSubtree`, `SelectionProtocol`, `PromptBundle`, `SelectionResult`, `ChildSearchCursor`, `ExpansionPlan`, `NodeSearchResult`, `ABSTAIN_TOKEN`)
- `./config.rs` → `ProgressiveSearchConfig`, `DisclosureConfig`, `EmbeddingScoringConfig`, `SelectionMode`, `DEFAULT_TOP_K`
- `./contracts.rs` → the five seams, `FragmentBuildFn`, `GenerateFn`, `GenerateFragmentSelector`, `SelectRequest`, `GenerateRequest`
- `./lexical.rs` → `LexicalIndex`, `LexicalDocument`, `LexicalHit`, `tokenize`, `QueryMatcher`, `compile_matcher`, the BM25/field/phrase weights, snippet and front-matter helpers
- `./embedding.rs` → `EmbeddingScorer`, `cosine_similarity`, `blend_scores`
- `./codebooks.rs` → `CompactCodebook`, `CodeTrie`, `default_compact_boundary_codebook`
- `./truncation.rs` → `truncate_sorted_by_score`, `ScoreTruncationConfig`, `ScoreTruncationDecision`, the bounded char/item helpers
- `./subtree.rs` → `DefaultCurrentSubtreeProvider`, `ItemCounterFn`, the progressive-root catalog helpers
- `./selector.rs` → `LexicalTopKSelector`, `DefaultTopKSelector`, `is_abstain_output`, `query_text`, `ABSTAIN_HINTS`, the `fragment_selected`/mode constants
- `./expand.rs` → `DefaultTargetExpander` and the per-branch budget
- `./reduce.rs` → `DefaultBranchReducer`, the round-robin merge and dedupe key
- `./engine.rs` → `RecursiveSearchEngine`, `EngineRequest`, `TraversalState`, `TraversalStep`, the engine event constants
- `./progressive.rs` → `ProgressiveRetriever`, `ProgressiveSearchResult`, `ModelTopKSelector`, `validate_query`, the method/LLM event constants
- `./blend_selector.rs` → `EmbeddingBlendedTopKSelector`, `MODE_EMBEDDING_BLEND`
- `./flat.rs` → `FlatRetriever`, `FlatSearchResult`, `FLAT_ROOT_NODE_ID`, the flat method constants
- `./service.rs` → `Retriever`, `RetrieverConfig` with `TraversalConfig`/`RenderConfig`/`GenerationConfig`, `SearchRequest`, `runtime_config_from_config`, `CandidateRecord`, `RetrievalResult`
- `./render/mod.rs` → `SubtreeRenderer`, `DefaultSubtreeRenderer`, `DISCLOSURE_PROMPT_VERSION`
- `./render/expose.rs` → `build_exposed_fragment`, `exposure_item_count`, `fragmentation_fingerprint`, code assignment, flat rendering, the Rust-only budgets
- `./render/prompt.rs` → the Chinese prompt bank, `build_disclosure_messages`, `build_disclosure_prompt_parts`, the request marker
- `./render/parse.rs` → `parse_selected_codes` and the cascade helpers
- `./CONTEXT.md` → this file
- `../../tests/retrieval_search.rs`, `../../tests/retrieval_search_lexical.rs`, `../../tests/retrieval_search_disclosure.rs` → the three crate-level suites

## Notes for Agents

- `lexical.rs` and `render/expose.rs` are over the 1000-line soft cap only because their `#[cfg(test)]` suites stay in-file; their non-test halves are under it, so do not split them on size alone.
- `flow::SearchCursor` and `types::SearchCursor` are unrelated: the first names one runtime node plus the path that reached it, the second is the build half's pre-order plan. Never convert one into the other.
- Publicization is the only place that maps internal ids to public ones; adding a public field to `RetrievalResult` means deriving it from the trimmed `CandidateRecord` rows so `search` and `search_details` cannot diverge.
- `top_k == 0` is the crate's unlimited sentinel everywhere in this module; a new stage that slices by `top_k` must handle it.
- Prompt text belongs in `render/prompt.rs` and is intentionally Chinese (the reference's text); do not translate or reflow it, and keep the `{…}` placeholders substituted in one pass.
- `artifacts::write_index` cannot round-trip a zero-node index (see Known Issues); a fixture that needs an empty catalog uses a hand-built `LoadedIndex` instead.
