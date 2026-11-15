# retrieval — capability-tree index and query-time search runtime

## Intent

Ports Python `openjiuwen.symphony.retrieval` (v0.1.20) in two halves: the index-BUILD half, where a set of capability assets becomes a persisted, browsable taxonomy index, and the query-time runtime in `search`.
One index is a directory of exactly three artifacts — `tree_index.yaml` (flat node list keyed by dot-joined cid), `catalog.jsonl` (one record per leaf item) and `manifest.json` (mode/count/membership) — consumed by the query-time retriever in [`search`](src/retrieval/search/CONTEXT.md).
Construction is fully offline and deterministic: the item tree, the cids, the catalog text and the byte layout of every artifact are reproducible bit for bit.
Rule grouping, deterministic node assembly and cid-derived branch descriptions are the default, so a build with no model configured is fully offline and byte-reproducible.
Embedding-backed k-means grouping, model-guided grouping and the four tree-shaping stages are opt-in and each carries a documented deterministic fallback; the `tree_index.html` visualizer is an opt-in extra outside the artifact contract.
The query-time half lives in [`./search/`](src/retrieval/search/CONTEXT.md): index loading, the progressive tree walk, lexical/embedding/model selection, disclosure rendering, expansion and reduction, the flat one-shot retriever and the public `Retriever` facade.

## API Surface

All paths crate-relative. Every public item is documented; `cargo clippy -D warnings` includes `missing_docs`.

### `types` (`src/retrieval/types.rs`)

```rust
pub const TREE_INDEX_FILENAME: &str = "tree_index.yaml";
pub const CATALOG_FILENAME: &str = "catalog.jsonl";
pub const MANIFEST_FILENAME: &str = "manifest.json";
pub const TREE_HTML_FILENAME: &str = "tree_index.html";   // opt-in extra, outside the three-artifact contract
pub const MODE_FULL: &str = "full";      pub const MODE_INCREMENTAL: &str = "incremental";
pub const NODE_TYPE_BRANCH: &str = "branch";
pub const NODE_TYPE_LEAF: &str = "leaf";
pub const NODE_TYPE_ROOT: &str = "root";  pub const ROOT_NODE_ID: &str = "ROOT";
pub const DEFAULT_ITEM_TYPE: &str = "skill";
pub const META_CONTENT / META_SOURCE_DESCRIPTION / META_SELECT_WHEN / META_DONT_SELECT_WHEN: &str;
pub const RETRIEVAL_TEXT_LIMITS: [usize; 5] = [200, 400, 1200, 120, 200];

pub struct RetrieverItem { pub item_id, pub payload, pub label, pub description: String }
impl RetrieverItem { pub fn new(item_id, payload) -> Self; pub fn with_label(self, …) -> Self; pub fn with_description(self, …) -> Self; }

pub struct RetrieverNode { pub node_id, pub label, pub description: String, pub children: Vec<RetrieverNode>, pub items: Vec<RetrieverItem> }
impl RetrieverNode {
    pub fn new(node_id, label) -> Self;  pub fn with_description / with_children / with_items (builder style);
    pub fn push_child(&mut self, Self);  pub fn push_item(&mut self, RetrieverItem);
    pub fn children(&self) -> Vec<&Self>;      // ALWAYS sorted by node_id
    pub fn items(&self) -> Vec<&RetrieverItem>; // ALWAYS sorted by item_id
    pub fn find(&self, node_id: &str) -> Option<&Self>;
    pub fn walk(&self) -> Vec<&Self>;           // pre-order, children by node_id
    pub fn branches(&self) -> Vec<&Self>;  pub fn leaves(&self) -> Vec<&Self>;
    pub fn depth(&self) -> usize;  pub fn item_count(&self) -> usize;  pub fn item_ids(&self) -> Vec<String>;
    pub fn flatten_choices(&self) -> Vec<RetrieverChoice>;   // walk order; choice_id = label else item_id, payload = item payload
}

pub struct RetrieverChoice { pub choice_id, pub payload, pub description: String }   // + new
pub struct RetrieverCandidate { pub rank: usize, pub item_id, pub payload: String, pub branch_path: Vec<String>, pub label, pub description: String }   // + new(rank, item_id, payload)
pub struct RetrieverTraceEvent { pub event_type, pub node_id: String, pub depth: usize, pub detail: BTreeMap<String, Value> }
pub struct RetrieverTrace { pub events: Vec<RetrieverTraceEvent> }
impl RetrieverTrace {
    pub fn record(&mut self, event_type, node_id, depth: usize, detail: BTreeMap<String, Value>);
    pub fn record_simple(&mut self, event_type, node_id, depth: usize);
    pub fn events(&self) -> &[RetrieverTraceEvent];  pub fn len / is_empty;
    pub fn event_types(&self) -> Vec<&str>;  pub fn last(&self) -> Option<&RetrieverTraceEvent>;
}

pub struct CursorStep { pub node_id: String, pub depth: usize, pub path: Vec<String> }   // path = ancestors, root first, self excluded
pub struct SearchCursor { pub pending: Vec<CursorStep>, pub children: BTreeMap<String, Vec<String>>, pub visited: usize, pub exhausted: bool }
impl SearchCursor {
    pub fn new(root: &RetrieverNode) -> Self;        // one pre-order pass records the child plan
    pub fn reset(&mut self, root: &RetrieverNode);   // re-seed, keeping the same instance
    pub fn advance(&mut self) -> Option<CursorStep>; // O(log n) per step, no tree argument
    pub fn visited(&self) -> usize;  pub fn pending_len(&self) -> usize;  pub fn is_exhausted(&self) -> bool;
}

pub struct IndexNode { pub cid: String, #[serde(rename = "type")] pub node_type: String, pub description, pub select_when, pub dont_select_when, pub source_description, pub worker_id: String }
impl IndexNode {
    pub fn new(cid, node_type) -> Self;  pub fn branch(cid) -> Self;  pub fn leaf(cid, worker_id) -> Self;
    pub fn with_description / with_select_when / with_dont_select_when / with_source_description;
    pub fn is_branch(&self) -> bool;  pub fn is_leaf(&self) -> bool;
    pub fn segments(&self) -> Vec<&str>;  pub fn depth(&self) -> usize;  pub fn parent_cid(&self) -> Option<&str>;
}

pub struct IndexManifest { pub mode: String, pub count: usize, pub item_paths: Vec<String>, pub worker_ids: Vec<String>, pub item_type: String, pub content_hash: String }
pub struct CatalogRecord { pub worker_id, pub cid, pub name, pub description, pub skill_path, pub branch_path: Vec<String>, pub category, pub retrieval_text: String, pub metadata: BTreeMap<String, String> }
impl CatalogRecord { pub fn new(worker_id, cid) -> Self; pub fn with_name / with_description / with_skill_path / with_retrieval_text / with_metadata; pub fn metadata_of(&self, key) -> &str; }
pub struct LoadedIndex { pub root: RetrieverNode, pub choices: Vec<RetrieverChoice>, pub records: Vec<CatalogRecord>, pub manifest: IndexManifest, pub item_paths: BTreeMap<String, Vec<String>> }

pub fn compact_text(text: &str, limit: usize) -> String;   // whitespace-collapsed, char-truncated, "..." suffix
pub fn build_retrieval_text(name, description, content, worker_id, cid) -> String;   // 5 parts, newline-joined
```

`IndexManifest::content_hash`, `SearchCursor` and `LoadedIndex::item_paths` are Rust-only additions; the Python loader has no equivalent.
`RetrieverTrace::record` appends in call order (there is no timestamp and no sequence number, matching Python).

### `artifacts` (`src/retrieval/artifacts.rs`)

```rust
pub async fn write_index(dir: impl AsRef<Path>, nodes: &[IndexNode], records: &[CatalogRecord], manifest: &IndexManifest) -> Result<()>;
pub async fn load_index(dir: impl AsRef<Path>) -> Result<LoadedIndex>;

pub fn render_tree_yaml(nodes: &[IndexNode]) -> String;
pub fn parse_tree_yaml(text: &str) -> Result<Vec<IndexNode>>;
pub fn render_catalog(records: &[CatalogRecord]) -> String;
pub fn parse_catalog(text: &str) -> Result<Vec<CatalogRecord>>;
pub fn render_manifest(manifest: &IndexManifest) -> String;
pub fn parse_manifest(text: &str) -> Result<IndexManifest>;
pub fn catalog_content_hash(records: &[CatalogRecord]) -> String;
pub fn build_tree_root(nodes: &[IndexNode], records: &[CatalogRecord]) -> RetrieverNode;
pub fn compose_routing_description(description, select_when, dont_select_when) -> String;
```

`write_index` validates the tree, the records and the manifest BEFORE the first byte is written, then writes each file through a same-directory `<name>.tmp` + `rename` (tree, catalog, manifest last).
`load_index` requires all three files (a missing one is `Error::NotFound` naming it) and rejects: a `count` differing from the record count, a non-empty `content_hash` differing from the recomputed one, a blank `mode`/`item_type`, and any record `cid` with no leaf node in the tree — all `Error::Validation` naming the file.
It does not check the tree's shape (that is `build::schema`), worker-id agreement between the tree and the catalog, or the derived catalog text.

### `build` (`src/retrieval/build/`, see [`build`](src/retrieval/build/mod.rs))

- `scanner`: `ItemSources` (descriptors / `SkillInventory` / `SKILL.md` directory / JSONL rows) → `SourceItem`s plus `SourceDiagnostic`s; `normalize_item_type`, `lenient_item_type`.
- `grouping`: `GroupingStrategy` (six spellings; `is_supported` is `true` for all and `requires_runtime` marks `Kmeans`/`FaissKmeans`/`LlmGrouping`), `GroupingConfig` (`strategy`, `max_children`, `max_depth`, `root_segment`, `taxonomy`, `clusters`), `GroupAssignment`, `category_segments`, `group_items`, plus the runtime layer: `GroupingRuntime` (`new`, `with_embedding`, `with_model`, `with_fallback`, accessors, `is_empty`), `GroupingOutcome { assignments, strategy, provenance }`, `GroupingProvenance { Rule, Embedding, Model, DeterministicFallback }` and `group_items_with`.
- `grouping::kmeans`: `embedding_text`, `resolve_clusters`, `cluster_paths` and the `MAX_ITERATIONS`/`CONVERGENCE_EPSILON`/`NAME_LIMIT`/`DESCRIPTION_LIMIT`/`CONTENT_LIMIT`/`TAG_LIMIT` bounds; deterministic k-means over L2-normalised embeddings with a SHA-256-seeded k-means++-style initialisation, Lloyd iteration, empty-cluster reseeding, ties by `worker_id` and labels from the dominant leading name token (fallback `cluster-{n}`).
- `grouping::llm`: `STAGE`, `SYSTEM_PROMPT`, `NAME_LIMIT`/`DESCRIPTION_LIMIT`/`MAX_REQUEST_ITEMS`, `render_request`, `parse_reply` and `group_paths` — Python's two-phase grouping as one strict, byte-stable round trip over `CompletionModel` with `Error::Llm` on an unusable reply and no lexical fallback.
- `tree`: `build_tree`, `build_nodes`, `branch_chain`, `insert_item`, `leaf_node`, `branch_description`, `prune_workers`, `subtree_cids`, `remove_subtree`, `node_index`, `worker_node`, plus the `shaping` and `visualizer` submodules.
- `tree::shaping`: `ShapingMode { None, OneShot, Expansion, Repair, Equivalence }`, `ShapingConfig` (`mode`, `fallback`, `max_calls`, `max_items_per_prompt`, `description_limit`), `ShapingProvenance { Model, DeterministicFallback, Skipped }`, `ShapingOutcome { assignments, mode, provenance, model_calls, dropped }`/`is_model`, `apply`, and the four stage functions `one_shot`, `expansion`, `repair`, `equivalence`.
- `tree::visualizer`: `render_tree_html(nodes, records, root_label)` and `write_tree_html(dir, nodes, records, root_label)`.
- `tree_text`: `slug_term`, `cid_term`, `join_cid`, `parent_cid`, `cid_depth`, `unique_child_cid`, `text_tokens`, `sort_nodes`.
- `schema`: `TreeSchema`, `TreeDiagnostic`, the ten diagnostic-code constants, `RootCategory`, `FIXED_ROOT_CATEGORIES`, `root_category`.
- `presets`: `TreePreset`, `PresetNode` (nested-or-flat `tree_index.yaml` reader, cid derivation, `paths`/`leaf_paths`).
- `catalog`: `catalog_record(node, item) -> CatalogRecord`.
- `workflows`: `BuildConfig` (`grouping`, `schema`, `item_type`, `escalation_ratio`, `preset`, `options`; `options`/`with_options`/`with_html`), `BuildOptions { write_html, shaping }`, `BuildReport` (`output_dir`, `item_type`, `node_count`, `record_count`, `manifest`, `diagnostics`, `tree_diagnostics`, `grouping_provenance`, `shaping_provenance`, `html_written`), `IncrementalReport`, `IndexBuilder::{new, build, build_with_runtime, open, add, add_with_runtime, delete, rebuild_subtree, manifest, write}`.

### `search` (`src/retrieval/search/`)

The query-time runtime over a loaded index, documented in full in [`search`](src/retrieval/search/CONTEXT.md).

```rust
pub struct Retriever;   // load(dir, config) / from_loaded_index(index, config) / with_model / with_embedding /
                        // prepare / search / search_with / search_details / search_details_with / method
pub struct RetrieverConfig { pub top_k, pub traversal, pub render, pub generation, pub embedding_weight, pub truncation }
pub struct SearchRequest { pub top_k: Option<usize> }
pub fn runtime_config_from_config(config: Option<&RetrieverConfig>) -> Result<ProgressiveSearchConfig>;
pub struct RetrievalResult { pub method, item_ids, candidates: Vec<CandidateRecord>, summary_lines,
                             selected_item_id, selected_rank, trace_events, steps }

pub struct ProgressiveRetriever;   // new(root, config) / with_model / with_embedding / rebind / prepare_root /
                                   // search / search_messages / begin / step / finish
pub struct FlatRetriever;          // new(config) / with_embedding / retrieve / retrieve_choices
pub struct RecursiveSearchEngine;  // new(five seams + config) / search / step over a resumable TraversalState
pub struct ProgressiveSearchConfig;   // the engine's flat config; DisclosureConfig, EmbeddingScoringConfig, SelectionMode
pub struct LexicalIndex;              // field-weighted BM25 over LexicalDocument
pub fn build_exposed_fragment(..) -> Result<ExposedFragment>;   // disclosure exposure + code assignment
pub fn parse_selected_codes(fragment: &ExposedFragment, output: &str) -> Vec<SelectableResolution>;
```

`search` owns index loading through `artifacts::load_index`, the five traversal seams (`CurrentSubtreeProvider`, `SubtreeRenderer`, `TopKSelector`, `TargetExpander`, `BranchReducer`), disclosure rendering and prompt text, the tolerant reply parser, the code book and truncation helpers, the lexical and embedding scorers, the flat retriever, and the publicization that turns internal leaf cids into public `worker_id`s.
`types` is the hand-off surface it consumes; nothing in `search` writes an index.

## Artifact layout

```text
<index>/
  tree_index.yaml   # nodes: [ {cid, type, description?, select_when?, dont_select_when?, source_description?, worker_id?} ]
  catalog.jsonl     # one JSON object per leaf, sorted by cid
  manifest.json     # {mode, count, item_paths, worker_ids, item_type, content_hash}
```

`tree_index.yaml` is hand-rendered exactly like Python `build/io/tree.py`: one `nodes:` list sorted by `(cid segment count, cid)`, `cid` and `type` always emitted, the five optional keys only when non-blank, every scalar a JSON-quoted string (JSON string literals are valid YAML double-quoted scalars), 2/4-space indentation, trailing newline.
Hierarchy is implicit in the cid: `Skills.foo.bar`'s parent is `Skills.foo`; there is no `children:` key on disk.
`catalog.jsonl` keys are `worker_id, cid, name, description, skill_path, branch_path, category, retrieval_text, metadata{content, source_description, select_when, dont_select_when}`; an empty catalog renders as an empty file.
`manifest.json` is pretty JSON with `count` = catalog-record count (not the item-path count), `worker_ids` in catalog order and `content_hash` = SHA-256 of the rendered catalog.
`tree_index.html` is an opt-in extra: a build renders it only when `BuildOptions::write_html` is set, writes it after the three artifacts through a same-directory temp file plus rename, and never reads it back, so the three-artifact bytes are unchanged by the flag.
`TREE_HTML_FILENAME` stays declared so a caller can locate or delete a document produced by another run.

## Constraints

- `serde_yaml_ng` is the only YAML parser; rendering is hand-rolled because Python's emitter shape is part of the on-disk contract.
- Every public data type is `serde`-serializable (`snake_case`, tolerant of unknown fields, `BTreeMap`/`BTreeSet` only — never `HashMap`); `RootCategory` is the documented exception (it is a `const` table of `&'static str`, not a data record).
- Determinism: no clock, no RNG, no hash-map iteration order; nodes sort by `(depth, cid)`, records by `cid`, items by `worker_id`, choices by walk order, ties by id.
- Async: `tokio::fs` only; no `std::sync` guard is held across an `.await` (the build half holds no locks at all; `search` snapshots its cache and runtime under a lock before awaiting).
- Errors: `crate::error::{validation, unsupported}` over `openjiuwen_common::error::Error`; no `unwrap`/`expect` outside tests; no panic on user input.
- Dependencies: `openjiuwen-common` plus crate-internal `error`, `model`, `discovery`, `llm`, `search::tokenize`; both halves use `openjiuwen-retrieval`'s `Embedding` trait, the runtime-backed stages use `crate::llm::CompletionModel`, and k-means seeding uses the crate's existing `sha2` dependency, so no new crate dependency was added for either half.

## Design Decisions

- **Grouping has three interchangeable families.** The rule strategies (`Deterministic`, `CategoryPath`, `NameTokens`) derive a path per item — `classification` segments, else the first tag, else the leading name token — and are the offline default.
  `Kmeans`/`FaissKmeans` cluster the items with a supplied `Embedding` and `LlmGrouping` asks a supplied `CompletionModel`; both run only through `group_items_with`, `build_with_runtime` and `add_with_runtime`, because the synchronous [`group_items`](src/retrieval/build/grouping.rs) path cannot await and answers `Error::Unsupported` naming the runtime entry point.
- **Every runtime strategy has a documented fallback, and the caller chooses it.** Under `FallbackMode::Deterministic` (the default) a missing or failing embedder/model degrades to the deterministic rule and the outcome reports `GroupingProvenance::DeterministicFallback`; under `FallbackMode::Error` it surfaces as `Error::Config` (missing runtime) or the model/embedder's own error.
  A partial model reply leaves the omitted items unplaced — each omitted or unknown id is warned and dropped — and node assembly then fails loudly with `Error::Validation` rather than silently publishing a tree without them.
- **k-means is deterministic by construction.** Vectors are L2-normalised once and compared by squared euclidean distance (the same ordering as cosine for unit vectors), the first centroid is picked from a SHA-256 digest of the sorted item ids, Lloyd's algorithm runs at most `MAX_ITERATIONS` rounds to `CONVERGENCE_EPSILON`, and empty clusters are reseeded from the item farthest from its own centroid; `k` is `clusters` or `ceil(items / max_children)`, clamped to `[1, min(items, max_children)]`.
- **Model-guided grouping is one strict round trip, not two passes.** `render_request` sends a byte-stable inventory payload (sorted by `worker_id`, truncated fields), `parse_reply` slugs and truncates every path and drops unknown or repeated ids with a warning, and `group_paths` turns an unusable reply into `Error::Llm`; whether that failure degrades is the caller's fallback mode.
  The whole inventory must fit `MAX_REQUEST_ITEMS` (1000) because batching is not implemented, and an empty inventory is answered without a model call.
- **The shared placement pipeline bounds every strategy.** All six candidate paths pass through the taxonomy override, the `max_depth - 1` cut and the `max_children` rebalancing, so a model or embedding can never bypass the preset or the tree bounds.
  Fan-out is hard above `max_depth` and soft exactly at it, where Python's own builder also stops splitting.
- **Tree shaping is opt-in and post-grouping.** `ShapingMode::None` is the default and calls no model; `OneShot`, `Expansion` and `Equivalence` refine the placements before assembly, and `Repair` runs after a fatal schema finding and re-validates its answer.
  Every stage keeps one placement per item (omitted items are backfilled with the deterministic answer), normalises segments and applies the shared bounds, and degrades under `FallbackMode::Deterministic`; an answer that would lose, duplicate or invent an item is rejected rather than accepted.
- **Branch descriptions come from the cid path, never from a model.** Python's `branch_descriptions.py` asks a model; the port always derives the sentence from `cid`, and a shaping reply's branch description is not persisted, so a tree never embeds prose that could drift between runs.
- **The visualizer sits outside the three-artifact contract.** `tree_index.html` is derived data written after the three artifacts through temp+rename, nothing reads it back, and with `write_html` off the published bytes are exactly the three artifacts.
- **A curated preset is the tree, not a hint.** `BuildConfig::with_preset` installs the preset's paths as the grouping taxonomy; a leading segment that slug-matches `root_segment` is stripped so a preset rooted at `Skills` does not double the root, and only the leaf cids are derived.
- **`TreeSchema` is a Rust-only guard.** Python has no tree validator (`schema.py` validates nothing and `io/tree.py` accepts any node list); here `blank_cid`, `duplicate_cid`, `missing_parent`, `cycle`, `excessive_depth` and `too_many_nodes` abort a build, while `excessive_fan_out`, `blank_label` and `missing_worker_id` are reported as advisory diagnostics.
  A cycle is only expressible in a nested preset (a child whose cid equals an ancestor's), because a flat cid list cannot loop.
- **cids are dot-joined slugs**, `tree_text::slug_term` for category and leaf segments and `unique_child_cid` for the `-2`/`-3` disambiguation; every writer and reader derives parents by string prefix, so a cid must never contain a dot.
- **Publication is atomic per file and validated up front**, so a rejected build leaves the previous index untouched; a successful write leaves no `*.tmp` sibling behind.
- **An incremental patch is atomic in memory too**: `add` computes every placement before the first mutation, so a strategy failure leaves the builder unchanged.
- **Escalation flags, it does not rebuild.** Python re-runs a full build above `incremental_max_change_ratio` because its orchestrator still holds the sources; here `add`/`delete` apply the patch and set `IncrementalReport::escalated` (strict `>`, denominator = record count), leaving the full rebuild to the caller.
- **Incremental placement runs the configured strategy per item** — the rule path in `add`, the runtime path in `add_with_runtime` — instead of Python's lexical token-overlap placement with confidence/margin gates, and a patch is placement-only (the shaping stages run on a full build, never on a patch).
  `rebuild_subtree` re-derives a branch deterministically from its catalog records; a tree a schema finding rejected can instead be repaired by the opt-in `ShapingMode::Repair` stage on a full build.
- **The synthetic `ROOT` exists in memory only.** `artifacts::build_tree_root` adds it above the persisted top-level nodes (Python's loader does the same) and prunes branches that end up childless; the on-disk tree has no root node.
- **`SearchCursor` carries its own child plan**, so a walk is O(n) rather than O(n²) and a serialized cursor resumes exactly where it stopped.

## Deferred Scope

The build half is implemented end to end: item scanning, the three rule strategies, the embedding-backed k-means strategy, the model-guided round trip, the four tree-shaping stages with their deterministic fallbacks, node assembly, preset override, the schema guard, catalog derivation, the full/incremental/subtree workflows, the three-artifact IO and the opt-in visualizer.
The query-time runtime is likewise implemented (index loading, the progressive walk, disclosure rendering, lexical/embedding/model selection, the flat retriever and the publicizing facade).
The genuinely remaining gaps, absent or typed `Error::Unsupported` and listed here rather than stubbed:

| Area | Status |
|---|---|
| `search/` — `transformers`/vLLM logit-based branch selection | `Error::Unsupported` in [`deferred`](src/deferred.rs); the code-book/prefix-trie core, the LLM tree walk, the lexical and embedding selectors and the flat retriever are implemented (see [`search`](src/retrieval/search/CONTEXT.md)) |
| `build/tree/presets.py` `SKILLSMP_ROOT_CATEGORIES` | unported example taxonomy (a different taxonomy from the ported `root_categories.py` one in `build::schema`) |
| `build/io/config_loader.py`, `load_tree_root_categories` file loading, S3/OBS/HTTP ZIP item sources, `tree_sketch` rendering | absent; item sources are local directories, inventories, descriptors and JSONL text |
| Python's lexical incremental placement (`tree_ops.py` confidence/margin gates) | replaced by running the configured grouping strategy per added item |
| `search`'s remaining pure-logic gaps | see [`search`](src/retrieval/search/CONTEXT.md): term-family ranking, `casefold`, the legacy direct recursion, branch parallelism, the fixed-prefix-cache rejection, S3/HTTP index materialization and `service/serving.py` |

## Known Issues
- `GroupingStrategy::parse` maps an unknown strategy name to `Deterministic`, so a config naming a strategy this build does not implement groups deterministically with no warning instead of failing the build.
- Model budgets are hard limits, not batching hints: `llm_grouping` refuses an inventory larger than `MAX_REQUEST_ITEMS` (1000) with `Error::Llm`, and tree shaping bounds one run at `ShapingConfig::max_calls` (default 8) model calls and one prompt at `max_items_per_prompt` (default 24, `0` = unlimited) entries, leaving the unreached items on their deterministic answer.

## Notes for Agents

- A `//!` doc block cannot link items defined in the same file module; write `[`X`](crate::retrieval::types::X)`.
- `types.rs`, `artifacts.rs`, `build/scanner.rs`, `build/grouping.rs`, `build/tree/shaping/mod.rs` and `build/workflows.rs` sit over the 1000-line soft cap because their `#[cfg(test)]` suites must stay in-file; their non-test halves are under it, so do not split them further on size alone.
- The same holds for `search/lexical.rs` and `search/render/expose.rs`; do not split them on size alone.
- The artifact bytes are a contract: changing `render_tree_yaml`/`render_catalog`/`render_manifest` breaks index compatibility with Python's loader, and `catalog_content_hash` covers the catalog only.
- `tree_index.html` is not part of that contract: it is derived, written after the three artifacts and never read back, so its markup may change without an index migration.
- The rule path (`group_items`, `IndexBuilder::build`/`add`) places every item without awaiting; anything that reaches an embedder or model goes through `group_items_with`, `build_with_runtime` or `add_with_runtime`, and the shaping stage reads its model from that same `GroupingRuntime`.
- Never write a cid containing a dot, a blank segment, or a `.`/`..` segment; `parse_tree_yaml` and `build_tree_root` trust the prefix rule.
- `tests/retrieval_build.rs` and `tests/retrieval_artifacts.rs` share a 1–10 coverage numbering; keep it ascending when adding cases.

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`): naming/cid helpers, tree-text truncation, JSON/YAML render+parse round-trips, schema diagnostics, preset derivation, grouping determinism and bounds, tree assembly and pruning, catalog derivation, builder workflows.
- `grouping` (`src/retrieval/build/grouping.rs`, `grouping/kmeans.rs`, `grouping/llm.rs`) — strategy parsing/`as_str`/`requires_runtime`, taxonomy override, depth/fan-out rebalancing, runtime dispatch and every provenance/fallback/error-mode branch; embedding-text parts and truncation, derived/explicit cluster counts, k-means determinism across runs, disjoint-vocabulary separation, dominant-token labels and root placement at `max_depth == 1`; byte-stable and bounded grouping requests, reply slugging/dedup, unknown/unassigned id handling, and the unusable-reply/no-known-item/failed-call errors.
- `tree::shaping` (`src/retrieval/build/tree/shaping/{mod,one_shot,expansion,repair,equivalence}.rs`) — config defaults and JSON round-trips, shared slug/bound/fold/parse/finalize helpers, `apply`'s `None` and `Repair` answers, and each stage's model path, deterministic fallback, `FallbackMode::Error` surfacing, call budget, item preservation/backfill, prompt byte stability, path bounds and repair re-validation.
- `tree::visualizer` (`src/retrieval/build/tree/visualizer.rs`) — byte-identical renders, cid nesting and ordering, escaping of every HTML metacharacter and a script-injection attempt, per-branch counts, empty tree, leaf-without-record and record-without-leaf, and the atomic write.
- `workflows` (`src/retrieval/build/workflows.rs`) — the three-artifact full build, byte-identical rebuilds, open/add/replace/delete/escalation/subtree rebuild, preset override, fatal versus advisory diagnostics, config validation, `build_with_runtime` k-means and model grouping, the synchronous deferral, `add_with_runtime`, the one-shot/repair shaping paths, and `write_html` on and off.
- `tests/retrieval_build.rs` — item sources (descriptor/inventory/JSONL/skill directory incl. a broken folder), grouping determinism and category derivation, preset override, schema validation failures, incremental `add`/`delete`/`rebuild_subtree`, a failed `add` leaving the builder unchanged, escalation, and the synchronous grouping deferral; its runtime-strategy coverage is being extended separately.
- `tests/retrieval_artifacts.rs` — the three-artifact set and their exact shapes, byte-identical rebuilds, no `*.tmp` leftovers, rejected publications leaving nothing behind, `load_index` round-trip, and the corrupt/tampered/missing-file errors (`NotFound`, `count`, `content_hash`, foreign catalog cid).
- `tests/retrieval_search.rs`, `tests/retrieval_search_lexical.rs` and `tests/retrieval_search_disclosure.rs` — the query-time runtime's service facade, foundation layer and render layer; see [`search`](src/retrieval/search/CONTEXT.md) for their coverage.
- Everything is offline and deterministic: `tempfile::tempdir()`, `HashEmbedding` and `StaticCompletionModel` for the embedding/model stages and the search suites, no network, no clock, no map-order assertions.

## Routing Table

- `./mod.rs` → module docs, `pub mod artifacts/build/search/types`
- `./types.rs` → value objects: items/nodes/choices/candidates/trace, `SearchCursor`, `IndexNode`, `IndexManifest`, `CatalogRecord`, `LoadedIndex`, retrieval-text helpers
- `./artifacts.rs` → the three-artifact render/parse/write/load set, `catalog_content_hash`, `build_tree_root`, `compose_routing_description`
- `./build/mod.rs` → the build pipeline overview and module list
- `./build/scanner.rs` → `ItemSources`/`SourceItem`/`SourceDiagnostic`, item-type normalization
- `./build/grouping.rs` → the strategy table, `GroupingConfig`/`GroupAssignment`, the shared deterministic placement pipeline, `GroupingRuntime`/`group_items_with`/`GroupingOutcome`/`GroupingProvenance`, and the synchronous `group_items` deferral
  - `./build/grouping/kmeans.rs` → `embedding_text`, `resolve_clusters`, `cluster_paths` (deterministic embedding-backed k-means)
  - `./build/grouping/llm.rs` → `render_request`, `parse_reply`, `group_paths` (the strict model round trip)
- `./build/tree.rs` → node assembly, cid derivation, insertion/pruning/subtree helpers
  - `./build/tree/shaping/` → the opt-in model stages `one_shot`, `expansion`, `repair`, `equivalence`, plus `ShapingMode`/`ShapingConfig`/`ShapingProvenance`/`ShapingOutcome`/`apply`
  - `./build/tree/visualizer.rs` → `render_tree_html`, `write_tree_html` (the opt-in `tree_index.html` document)
- `./build/tree_text.rs` → slug/cid/join/parent/token helpers, `sort_nodes`
- `./build/schema.rs` → `TreeSchema` validation and the fixed root taxonomy
- `./build/presets.rs` → `TreePreset`/`PresetNode`, nested-or-flat preset parsing and cid derivation
- `./build/catalog.rs` → `catalog_record` derivation
- `./build/workflows.rs` → `BuildConfig`/`BuildOptions`, `BuildReport`/`IncrementalReport`, `IndexBuilder` (`build`, `build_with_runtime`, `add`, `add_with_runtime`, `open`, `delete`, `rebuild_subtree`, `write`), the repair path and the opt-in visualizer
- `./search/` → the query-time runtime: facade and config, engine and retrievers, the five seams and their defaults, lexical/embedding/model selectors, disclosure render and parse; see [`search/CONTEXT.md`](src/retrieval/search/CONTEXT.md)
