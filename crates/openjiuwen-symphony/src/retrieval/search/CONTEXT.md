# src/retrieval/search

## Intent

The query-time search runtime of the capability-tree retrieval module, ported from Python `openjiuwen.symphony.retrieval.search.runtime` (v0.1.20).
This directory currently holds the runtime's **foundation layer**: the flow value objects, the serializable engine configuration, the field-weighted lexical scorer, the optional embedding scorer, the compact boundary code book, and score/budget truncation.
Three later waves add the layers above it — progressive disclosure rendering, the traversal protocols (subtree exposure/selection, expansion, reduction), and the engine/retrievers/facade — strictly on top of this surface.
Everything here is offline and deterministic: no model, no network, no clock, and no hash-map iteration order in any output.

## API Surface

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
    pub max_description_chars: usize,        // Rust-only description clamp; 150 (Python flat mode); 0 = unlimited
    pub code_width: usize,                   // Rust-only width override; 0 = derive from the assigned codes
}                                            // with_* builders + validate()

pub struct EmbeddingScoringConfig { pub weight: f64, pub max_documents: usize }   // 0.0 / 0
// with_weight clamps into [0, 1] (NaN -> 0.0) and documents the clamp; with_max_documents; validate()

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
    pub disclosure: DisclosureConfig,    // the PRODUCTION overrides (see below)
    pub embedding: EmbeddingScoringConfig,
    pub truncation: crate::retrieval::search::truncation::ScoreTruncationConfig,
}                                    // validate() rejects top_k == 0, a bad codebook, contradicting bounds
pub const RUNTIME_TOP_K: usize = 5;  // in `mod.rs`
```

`ProgressiveSearchConfig::default()` keeps the runtime defaults (`top_k = 5`, `max_tokens = 96`, the two slacks and `round_robin_branch_reduce`) but applies the disclosure overrides Python's facade applies: `max_exposure_depth_per_call = 99`, `exposure_threshold = 1_000_000_000`, `compact_codes_enabled = true`, `flatten_full_tree_in_prompt = true`.
`DisclosureConfig::default()` keeps the conservative Python `ProgressiveRetrieverConfig` values (2 / 12 / false / empty / false) plus the Rust-only budgets `max_description_chars = 150` and `0` for the two "unlimited" caps.
The crate therefore has ONE engine default — the production shape — and a caller building a conservative engine starts from `DisclosureConfig::default()`.

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
`bm25` sums `idf² · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))` over the distinct query terms present (`idf = ln(1 + (N - df + 0.5)/(df + 0.5))`) and then adds the strongest single `idf²` once.
`field_score` adds `Σ_f w_f · |terms ∩ field_f| / |terms|` plus `IDENTITY_BONUS · |terms ∩ (key ∪ name ∪ aliases)| / |key ∪ name ∪ aliases|`.
`phrase_score` adds `12 / 5 / 0.5 / 0.1` once per `|`-separated alternative for a substring hit in the identity fields, description, body and category (first hit wins).
Ranking order: identity tier (`0` key, `1` name, `2` alias, `3` prefix, `4` otherwise), then descending score (`f64::total_cmp`), then key.
`from_loaded_index` indexes one document per catalog record (`key = cid`, body = `retrieval_text`, category = `branch_path.join(" > ")`) plus one per tree branch node (`key = node::<node_id>`, category = the ancestor node ids joined with `" > "`, the synthetic root excluded).
`search_in` scopes the ranking and treats an **empty** `keys` slice as "the whole corpus"; unknown and duplicated keys are ignored.
`match_snippet` labels its evidence `name: `, `alias: `, `description: ` or `body: `, is bounded by `max_chars`, and returns `""` when nothing hidden matched.

### `embedding` (`src/retrieval/search/embedding.rs`)

Rust-only, additive over the lexical scorer.

```rust
pub struct EmbeddingScorer;   // new(Arc<dyn Embedding>) / index / index_capped / upsert / remove / clear /
                              // scores / search / len / is_empty
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64;                          // 0.0 for degenerate inputs
pub fn blend_scores(lexical: &[LexicalHit], embedding: &BTreeMap<String, f64>, weight: f64) -> Vec<LexicalHit>;
```

`index` embeds each document's retrieval text (name + description + body + category, falling back to the key) in key order; `index_capped` honours `EmbeddingScoringConfig::max_documents` (`0` = unlimited).
`scores` returns one cosine per indexed key; `search` returns the descending order with ties on the key and `top_k == 0` meaning unlimited.
`blend_scores` clamps `weight` into `[0, 1]`, normalizes the lexical score by the largest one, maps the cosine with `(cos + 1) / 2` and unions both key sets (a key missing on one side scores `0.0` there); `weight == 0.0` therefore reproduces the lexical ranking and `weight == 1.0` the embedding ranking.
No `std::sync` guard is ever held across an `.await`: the scorer snapshots its vector map into a `Vec`, drops the guard, then embeds.

### `codebooks` (`src/retrieval/search/codebooks.rs`)

```rust
pub const DEFAULT_COMPACT_CODEBOOK_SIZE: usize = 512;
pub struct CompactCodebook;   // new / codes / len / is_empty / code / prefix / trie / from_value
pub struct CodeTrie;          // new / is_valid_prefix / is_code / completions / len / is_empty
pub fn default_compact_boundary_codebook() -> CompactCodebook;
```

`CompactCodebook::new` trims each entry and rejects a blank, whitespace-containing, duplicated or `"0"` code with `Error::Validation` naming the code; `prefix(count)` is `Error::Validation` when the book is shorter than `count`; `from_value` accepts a JSON list or an object holding one.
`CodeTrie` is the prefix-validation structure a constrained decoder needs; `completions` is sorted.
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

The three rules run in the reference order (min_score → relative_min_score → autocut), each can only lower the cutoff, and `reason` names the last rule that actually moved it — so a later rule reaching the same cutoff leaves the earlier name in place.
`relative_min_score` is skipped when the best score is not positive; `autocut` compares each adjacent pair, ignores non-positive drops and drops below `autocut_min_relative_drop`, and cuts after the `autocut_jumps`-th qualifying drop.
Empty input is `kept_count = 0`, `cutoff_index = None`, `reason = ""`.
The budget helpers treat `0` as "unlimited", are character-boundary safe, and the ellipsis counts inside the `max_chars` budget.

## Constraints

- Only standard-library logic plus the crate's own dependencies; no regex engine, no clock, no RNG, no new dependency (`regex`, `chrono`, `futures` are unavailable to this crate).
- Errors are `crate::error::{validation, unsupported}` over `openjiuwen_common::error::{Error, Result}`; a validation error names the offending field (`query`, the code, …). No `unwrap`/`expect` outside tests, no panics on a malformed query or document.
- `tracing` only (no `println!`/`eprintln!`), no `unsafe`, no `std::sync` guard held across an `.await`.
- Every public item carries a doc comment and a `//!` block links same-file or sibling items with an absolute `crate::retrieval::search::…` path; `cargo doc -p openjiuwen-symphony --no-deps` must stay warning-free.
- Public data types are `serde`-serializable with `snake_case` field names, tolerant of unknown fields, and `JsonSchema` where the field types allow it.
- Determinism: `BTreeMap`/`BTreeSet` everywhere in public and serialized shapes, sorted outputs, ties on the key, no clock.
- Files stay ≲1000 lines; `lexical.rs` sits just over the soft cap with its in-file unit tests, which is the crate's documented pattern for test-heavy modules.
- Deviations from the reference are documented, not silent: the matcher is a bounded subset, the code book is generated, `LexicalIndex.search_terms` is not ported, and `casefold()` is approximated by `to_lowercase()`.

## Routing Table

### Present

- `./mod.rs` → module docs, module declarations, the re-exports, `RUNTIME_TOP_K`
- `./flow.rs` → the flow value objects (`ChatMessage`, `SearchCursor`, `ExposedNode`, `ExposedFragment`, `SelectableResolution`, `SelectableTarget`, `CurrentSubtree`, `SelectionProtocol`, `PromptBundle`, `SelectionResult`, `ChildSearchCursor`, `ExpansionPlan`, `NodeSearchResult`, `ABSTAIN_TOKEN`)
- `./config.rs` → `ProgressiveSearchConfig`, `DisclosureConfig`, `EmbeddingScoringConfig`, `SelectionMode`, `DEFAULT_TOP_K`
- `./lexical.rs` → `LexicalIndex`, `LexicalDocument`, `LexicalHit`, `tokenize`, `QueryMatcher`, `compile_matcher`, the BM25/field/phrase weights, snippet and front-matter helpers
- `./embedding.rs` → `EmbeddingScorer`, `cosine_similarity`, `blend_scores`
- `./codebooks.rs` → `CompactCodebook`, `CodeTrie`, `default_compact_boundary_codebook`
- `./truncation.rs` → `truncate_sorted_by_score`, `ScoreTruncationConfig`, `ScoreTruncationDecision`, the bounded char/item helpers
- `./CONTEXT.md` → this file
- `../../tests/retrieval_search_lexical.rs` → the crate-level suite over the loaded-index, ranking, phrase, snippet, scoping, determinism, embedding and truncation behaviour

### Not yet present (the next waves)

- `./render/` → progressive disclosure rendering: the exposed tree, code assignment, the prompt snapshot and the bounded fragments — to be added by the rendering wave.
- `./subtree.rs` → subtree exposure and selection-protocol construction — to be added by the traversal wave.
- `./selector.rs` → the model reply parser that turns raw output into `SelectionResult`s — to be added by the traversal wave.
- `./expand.rs` → child-cursor expansion from the selected targets (`ExpansionPlan`) — to be added by the traversal wave.
- `./reduce.rs` → candidate aggregation across branches, including the round-robin reduce — to be added by the traversal wave.
- `./engine.rs` → the bounded traversal state machine over `ProgressiveSearchConfig` — to be added by the engine wave.
- `./progressive.rs` → the progressive retriever facade (`ProgressiveRetriever`) — to be added by the engine wave.
- `./flat.rs` → the flat retriever that ranks the whole catalog in one pass — to be added by the engine wave.
- `./service.rs` → the public search service facade (`DEFAULT_TOP_K`, mode selection, checkpointing) — to be added by the facade wave.
