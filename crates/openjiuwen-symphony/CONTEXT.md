# openjiuwen-symphony

## Intent

The capability-asset engine of openJiuwen, mapping to the Python package `openjiuwen.symphony` (v0.1.20, ~52k lines, 201 files).
Python's engine is host-driven: it fingerprints capability assets, indexes them, builds a capability relationship graph, plans asset compositions with an LLM, and distils observed flows into skills.
This crate ports that engine end to end: the asset cards and their I/O normalization layer, a registry with CRUD plus keyword and embedding search, SKILL.md discovery with a lexical skill catalog and the model-facing `skill_index` tool, a versioned `can_feed` capability graph with atomic artifact publication in memory, on disk and in three object stores, a bounded deterministic planner and plan executor over `openjiuwen-graph`, flow distillation into versioned recipes and reviewed `SkillPack` artifacts, a deduplicating experience layer with trace ingestion and clustering, a capability-tree index builder plus the query-time search runtime over it, fail-soft evaluation, LLM-guided stages with deterministic fallbacks, and the graph-observation service that turns caller-supplied evidence into immutable merged revisions.
Every LLM-guided stage (progressive tree retrieval, ontology matching, semantic plan scoring, fingerprint extraction, LLM judging, flow narratives) sits behind the `CompletionModel` seam and carries a documented deterministic fallback, so the crate is fully usable offline with no model configured.
Everything here is offline, deterministic, and free of external services by default; credentials are redacted before text reaches an artifact.
The typed `Error::Unsupported` gaps that remain are the GPU/ML ones (faiss IVF index training in `ExperienceIndex`, faiss/k-means and LLM grouping in `retrieval::build`, `transformers`/`vLLM` logit selection in [`deferred`](src/deferred.rs)) and two vendor credential flows that need a signing/issuer stack (`GcsObjectStore::from_service_account_file`, `AzureBlobStore::from_client_credentials`).
A few pure-logic Python areas are still unported; they are listed under `## Deferred Scope` and are absent rather than stubbed.

## API Surface

All paths below are crate-relative. Every public item is documented; `cargo clippy -D warnings` includes `missing_docs`.
Areas with their own `CONTEXT.md` are summarized here and specified there.

### `error` (`src/error.rs`)

```rust
pub use openjiuwen_common::error::{Error, ErrorKind, Result};

pub fn unsupported(feature: &str) -> Error;                        // Error::Unsupported + Deferred Scope pointer
pub fn validation(field: &str, reason: impl fmt::Display) -> Error; // Error::Validation "<field>: <reason>"
pub fn is_unsupported(err: &Error) -> bool;
```

### `model` (`src/model/`)

Capability cards and their deterministic derived views: `CapabilityKey` (+ `Display`/`FromStr` on `"{type}:{id}"`), `CapabilityIo`, `CapabilityDescriptor` (`key`/`qualified_id`/`validate`/`retrieval_text`, `input_schema`/`output_schema`, `compute_content_hash`, `fingerprint`, `Card` impl), `CapabilityFingerprint`/`SemanticProfile`/`FingerprintArtifact`, `DataTypeVocabulary` + `normalize_type`, the naming helpers, and the ingest-side layer (`normalize_io_name`, `IONameVocabulary`, `normalize_io_specs{,_with_issues,_with_audit}`, `sanitize_io_specs`).
`semantic_content` is excluded from serialization, as in Python; `available` defaults to `true`.
See [`src/model/CONTEXT.md`](src/model/CONTEXT.md) for the data-type aliases and `can_feed` rules, the name-fold order, the issue/decision codes and the bounds.

### `store` (`src/store.rs`)

`AssetStore` (async `get`/`put`/`remove`/`list`/`clear`, with `list` sorted by qualified id) plus `KvAssetStore` (`new`, `with_prefix`, `in_memory`, `json_file`, `sqlite`).
KV keys are `<prefix><qualified_id>` (default prefix `capability/`) and values are the descriptor's JSON; `get`/`list` skip (and warn) malformed entries instead of failing the scan.

### `registry` (`src/registry.rs`)

`CapabilityRegistry` (`new`, `with_store`, `load`, `reload`, `store`, `revision`, `len`/`is_empty`, `register`, `register_all`, `get`, `contains`, `remove`, `clear`, `list`, `search`) plus `Default`.
The registry is authoritative in memory and write-through to the optional store; its `std::sync::RwLock` is held only for map mutations and snapshot clones, never across an `.await`.
`revision()` is an `AtomicU64` (`SeqCst`) that advances on a real mutation and never on a no-op or failure, so the semantic index re-embeds only when the catalog moved.

### `search` (`src/search/`)

`SearchRequest` (`text`, `capability_types`, `tags`, `exclude_tags`, `available`, `metadata_filter`, `top_k` default 10 with 0 = unlimited, `min_score`, `for_text`), `SearchHit` (`descriptor`, `score`, `matched_terms`) and `search_descriptors`, which filters → ranks → applies `min_score` → truncates; `CapabilityRegistry::search` delegates to it after dropping its lock.
`LexicalIndex` (`build`, `search`, `len`, `is_empty`) with `LexicalHit` and `tokenize` is rebuilt from a snapshot per search (BM25 `k1 = 1.2`, `b = 0.75`, ties by qualified id); filter-only requests score `0.0` sorted by qualified id, and `metadata_filter` is a Mongo-style predicate evaluated by `openjiuwen_foundation::store::filter::matches`.
`SemanticIndex` (`new`, `index`, `upsert`, `remove`, `clear`, `search`) keeps one record per descriptor (`id = qualified id`, `text = retrieval_text()`, metadata plus the descriptor JSON) and reconstructs the descriptor on a hit; `DEFAULT_TOP_K` is 10 and the metadata keys are `CAPABILITY_TYPE_FIELD`, `CAPABILITY_ID_FIELD` and `DESCRIPTOR_FIELD`.

### `graph` (`src/graph.rs`)

`CapabilityGraph` (`graph_id`, `graph_type` = `"capability_graph"`, `directed`, `nodes: BTreeMap<String, CapabilityNode>`, `edges: Vec<CapabilityEdge>`, `metadata`) with `CAN_FEED`: `new`/`from_descriptors`, `node`/`has_node`, `upsert_node`, `remove_node` (dropping incident edges), `add_edge`/`remove_edge`/`remove_incident_edges`, `outgoing`/`incoming`/`roots`/`leaves`/counts, and JGF v2 conversion (`to_value` emits `{"graph": {...}}`, `from_value` parses it).
`add_edge` rejects a missing endpoint, a self-loop, a blank relation, a non-finite weight and a weight outside `[0, 1]` with `Error::Validation`; edge reads sort by `(source, target, relation)` and insertion order is kept in `edges` with upsert-in-place.

### `artifact` (`src/artifact.rs`)

`GraphArtifactStore` (`publish(graph, version)`, `read(version)` where `None` is current, `status`, `versions`) over the immutable `GraphRevision` and `GraphArtifactStatus` (`STATUS_MISSING`/`STATUS_READY`, `missing`/`ready`/`is_ready`, snapshot `{"graph_id","current_version"}`), implemented by `MemoryGraphArtifactStore` and `FileGraphArtifactStore` (`new(root)`, `root`).
Republishing a version is `Error::Validation`; a `None` version publishes `v{n+1}`, and explicit versions must match `[A-Za-z0-9._-]+` and must not be `.`/`..`.
The file backend writes `root/versions/<version>/graph.json` plus `root/current.json` through `<name>.tmp` + `rename`, so no partial file is ever visible.

### `orchestration` (`src/orchestration/`)

Deterministic relation candidates, bounded planning, plan execution over `openjiuwen-graph` and the graph-mutation request contract: `CandidateGenerator`/`CandidateConfig` (`ExactIo`, `SemanticOverlap`, `TextualCoercion`), `Planner`/`PlanConfig`/`PlanRequest`/`OrchestrationPlan`/`PlanStatus`/`PlanMode`, `SemanticScorer`/`LexicalSemanticScorer`, `PlanExecutor`/`CapabilityRunner`/`UnsupportedRunner`/`PlanExecution`, and `MutationRequest`/`request_digest`/`MutationJournal`/`validate_mutation_against_inventory`.
Candidates rank by strategy, confidence, source then target and cap per source; plans rank by `0.45*mean(semantic) + 0.25*mean(edge_confidence) + 0.20*input_coverage - 0.03*(steps - 1)` under `max_depth`/`max_expansions`/`max_plans`; the matcher adapter in `orchestration::matcher` renders the compact request and validates the reply without calling a model.
See [`src/orchestration/CONTEXT.md`](src/orchestration/CONTEXT.md) for the formulas, bounds and journal semantics, and [`src/orchestration/matcher/CONTEXT.md`](src/orchestration/matcher/CONTEXT.md) for the protocol pipeline and diagnostic codes.

### `evaluation` (`src/evaluation/`)

Fail-soft evaluation over cards and recorded traces: `Metric`/`MetricScore`/`MetricStatus`/`EvaluationInput`/`EvaluationReport`, six static and five trace metrics, `EvaluationSuite` (`default_suite`/`with_llm_judge`), `select_cases`, `percentile`, `latency_statistics`, `aggregate_metric` and `SYMPHONY_EVALUATION_PROTOCOL`.
A metric either scores or reports a status without one (latency is always `Observed`, a missing trace or case is `NotApplicable`, a failure is `Error` and is also recorded in `EvaluationReport::errors`).
See [`src/evaluation/CONTEXT.md`](src/evaluation/CONTEXT.md) for the per-metric rubrics, check lists, thresholds and confidence tiers.

### `flow` (`src/flow/`)

Trace ingestion and normalization into `FlowTrace`/`ExecutionGraph`, structural distillation into versioned `ExperienceRecipe`s, content-addressed `SkillPack` packaging with a ten-check review (`structure`, `dependencies`, `code_safety`, `secrets`, `artifact_paths`, `integrity`, `meta_naming`, `materials`, `permissions`, `license`), `SwarmFlow` script generation (`generate_swarmflow_script`, `swarmflow_identifier`, `validate_generated_script`), package rendering (`SkillAdapter`/`SkillPackAdapter`/`render_package`) and an eager atomic flow store.
Every model-written narrative and trace-derived free-text field is redacted before it enters a recipe or package.
See [`src/flow/CONTEXT.md`](src/flow/CONTEXT.md) for the script shape, rendering rules and review checks.

### `redaction` (`src/redaction/`)

`redact_sensitive_text`, `redact_common_credentials`, `sanitize_distilled_text`, `normalize_sensitive_name`, `is_sensitive_name`, `redact_sensitive_json` and `redact_private_json`, plus the `REDACTED`/`REDACTED_CREDENTIAL`/`REDACTED_REQUEST` markers and `DISTILLED_TEXT_LIMIT`.
The scanner is hand rolled (Python's patterns use lookbehind) and covers PEM private-key blocks, `Bearer` tokens, JWTs, `sk-…` keys, AWS access key ids and `key=value`/`"key": "value"` assignments whose key `is_sensitive_name` recognizes; every pass is idempotent, works on char boundaries and replaces only the secret.
The JSON helpers recurse with a depth cap of 64 and preserve key order, and `llm::identity::sanitize_metadata` reuses `is_sensitive_name`.

### `llm` (`src/llm/`)

The single `CompletionModel` seam and its five stages — `TreeRetriever`, `OntologyMatcher`, `LlmSemanticScorer`, `FingerprintExtractor` and `LlmJudgeMetric` — each with a documented deterministic fallback under `FallbackMode::Deterministic` (`FallbackMode::Error` surfaces model failures instead).
`identity` derives model/metadata digests (`ModelIdentity`, `stable_metadata_sha256`, `endpoint_sha256`, `graph_identity_sha256`, `sanitize_metadata`); `cache` is the opt-in persistent `MatchCache` used via `with_cache` + `MatchOptions { force }`, keyed by candidate, both fingerprint digests and `matcher_signature()`.
See [`src/llm/CONTEXT.md`](src/llm/CONTEXT.md) for the fallback table and the stage rules.

### `experience` (`src/experience/`)

The experience layer: `ExperienceBank` (query patterns bound to skill sets, deterministic dedupe/merge, embedding-optional ranking), the persisted `ExperienceIndex` (one vector per item, Python's `index_dir` layout with a JSON vector matrix instead of faiss/numpy files, digest-verified loads, exact flat cosine search, `ivf*` names deferred), and the input half — `TraceStore` (idempotent `trace_store/{records.jsonl,processed_index.json}` cache), `cluster_traces` (skill set, then token Jaccard) and `ExperienceCollector` (5-example cap, 0.9 similarity merge, 20-item auto-flush).
See [`src/experience/CONTEXT.md`](src/experience/CONTEXT.md) for the full surface and rules.

### `discovery` (`src/discovery/`)

SKILL.md discovery plus the lexical skill catalog: both Python scan modes (immediate-child and bounded recursive), per-skill diagnostics instead of scan failure, descriptor derivation with SHA-256 content hashes, an order-independent inventory fingerprint, BM25 search with excerpts (`SkillFs`), the model-facing prompt surface (`visible_skill_names`, `prompt_snapshot`, `notice_state`) and the read-only `skill_index` tool (`SkillIndexToolkit`: list/search/read over a virtual category tree, `SkillNoticeState` for incremental reminders).
See [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md) for the full surface, rendering rules, notice semantics and the toolkit's request contract.

### `object_artifact` (`src/object_artifact.rs`)

`ObjectGraphArtifactStore` implements `GraphArtifactStore` over an `ObjectStore`, mirroring the filesystem key layout (`<prefix>/versions/<version>/graph.json`, `<prefix>/current.json`) and the `{"revision", "graph"}` envelope; it writes the revision before the `current` pointer, serializes publications through an async mutex and skips pinned ids.
`ArtifactTransport` exposes `ObjectStore`, `S3`, `Gcs` and `AzureBlob`, resolved by `ObjectGraphArtifactStore::{s3, gcs, azure_blob, open}`; a blank bucket is `Error::Validation`.
`GcsObjectStore` speaks the GCS JSON API with an OAuth bearer token or anonymously, and `AzureBlobStore` speaks Blob REST with SharedKey HMAC-SHA256, a SAS token or anonymously; `from_service_account_file` and `from_client_credentials` are typed `Error::Unsupported`.

### `retrieval` (`src/retrieval/`)

The capability-tree retrieval module, both halves: the index BUILD half in `types`, `artifacts` and `build`, and the query-time runtime in `search`.
The build half resolves item sources (descriptors, `SkillInventory`, `SKILL.md` directories, JSONL rows) with per-item diagnostics, groups deterministically (classification → tag → name token, bounded fan-out), assembles the cid tree under `TreeSchema` and the fixed root taxonomy, honours a `TreePreset`, derives one `CatalogRecord` per leaf, publishes exactly three artifacts (`tree_index.yaml`, `catalog.jsonl`, `manifest.json`) atomically and exposes `IndexBuilder::{build, open, add, delete, rebuild_subtree, write}`.
The query half's public shape is `Retriever::{load, from_loaded_index, search, search_details}`, which publicizes leaf cids into `worker_id`s; below it sit `ProgressiveRetriever` + `RecursiveSearchEngine` over the five seams, the lexical/embedding-blended/model `TopKSelector`s, `FlatRetriever` and disclosure render/parse.
See [`src/retrieval/CONTEXT.md`](src/retrieval/CONTEXT.md) for the artifact shapes and the grouping substitute, and [`src/retrieval/search/CONTEXT.md`](src/retrieval/search/CONTEXT.md) for the runtime's pipeline, scoring formulas, disclosure rules and deferrals.

### `runtime` (`src/runtime/`)

The composition root: `RuntimeState`, `SymphonyConfig` (graph root, scope id, journal path, plan/executor/candidate configs), `SymphonyRuntimeBuilder` and `SymphonyRuntime` (`open`/`state`/`registry`, `start`/`stop`/`close`, `build`/`graph`/`graph_status`/`read_graph`, `plan`/`invoke`/`search`/`semantic_search`/`reindex_semantics`, `mutate` and the capability helpers, `submit_evolution`, the observation methods and `capture_graph_snapshot`), plus `InvokeRequest`/`InvokeResponse`, `MutationOperation`/`MutationStatus` (`Published`/`Noop`/`Replayed`), `GraphMutationResult` (identity, counts, revision, and serde-defaulted `status`/`digest`/`source_revision`/`diagnostics`) and `EvolutionResult`.
`start`/`stop` are idempotent, `close` is `stop` plus artifact flush, and `start` restarts a stopped runtime and publishes a graph when none exists; the mutation order is replay → digest conflict → `Running` gate → change-set/CAS → apply → publish → record, where a replay with a different digest or a stale `source_revision` is `Error::Validation`.
`submit_evolution` applies successful transitions as an EMA on edge weights (`w' = (w + 1.0) / 2`); `semantic_search` re-embeds only when `CapabilityRegistry::revision()` moved, and the observation methods are `Error::Unsupported` unless `graph_artifact_root` is configured (its log lives at `<root>/evolution/`).
See [`src/runtime/CONTEXT.md`](src/runtime/CONTEXT.md) for the full surface and rules.

### `observation` (`src/observation/`)

Caller-supplied execution evidence distilled into derived, immutable observation and merged graph revisions; the static `can_feed` graph is never rewritten.
`GraphEvolutionInput`/`TraceEvidence`/`TaskOutcome` (contract v2, `validate`, `normalized`) feed `GraphObservationService::{new, submit, get_snapshot, flush, reproject, reproject_all, publish_safe_fallback, close}`, which returns `ObservationReceipt`s and `GraphSnapshot`s built from the JSONL `EvidenceStore`, the per-scope `RevisionStore`, `EdgeIdentity`/`StaticGraphIndex` and `stable_hash`.
Evidence is qualified first (a truncated trace, quality flags, weak evidence, unknown capability ids, a stale pinned snapshot or an edge without a countable outcome is `audit_only` with its reason and never re-qualifies); only `verified_success` plus orchestration-attributed `verified_failure` edges count, so `runtime_weight = 1.0 + clamp(0.05 * (success - failure), -1.0, 1.0)`.
See [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md) for the full contract, reason codes, rebinding states and the on-disk layout.

### `deferred` (`src/deferred.rs`)

One API-shaped stub remains, and every entry point returns `Error::Unsupported` via [`error::unsupported`]: the `transformers`/vLLM logit-selection backend, whose constructor and `select` both report `vLLM logit selection` and whose data types stay ordinary `serde` types.
The other typed deferrals live with their implementations: `ExperienceIndex`'s `ivf*` algorithms and `retrieval::build`'s faiss/k-means and LLM grouping strategies.

## Notes for Agents

- Intra-doc links in a module-level (`//!`) doc block do not resolve names defined or re-exported in that same file module; write them as labeled links to absolute paths, e.g. `[`AssetStore`](crate::store::AssetStore)`.
- Items in sibling modules and dependency types are linkable the same way (`crate::model::CapabilityDescriptor`, `openjiuwen_foundation::store::KvStore`); std generics used in prose and private items stay plain code spans.
- `cargo doc -p openjiuwen-symphony --no-deps` is the definition of done for doc links: it must print no warnings.

## Constraints

- Cargo: `edition 2021`, `rust-version 1.85`, `publish = false`, `#![forbid(unsafe_code)]`; `default = []` and the sole feature `sqlite` only enables the SQLite backends (`store::KvAssetStore::sqlite` and `experience::SqliteExperienceStore`) via `openjiuwen-foundation/sqlite`.
- Cargo dependencies additionally include `serde_yaml_ng` (front-matter parsing and rendered YAML manifests), `bytes` (object-store artifact payloads) and `base64` (Azure SharedKey signatures); all three are workspace dependencies, and `wiremock` is the dev-dependency behind the GCS/Azure emulator suites.
- Cloud transports use `openjiuwen_common::http` (never raw `reqwest`); no `hmac` and no XML crate exists offline, so SharedKey HMAC-SHA256 and the Azure list XML scanner are hand rolled and unit-tested (RFC 4231 vectors).
- The umbrella crate `openjiuwen` re-exports this crate as `openjiuwen::symphony`; `crate_name()` returns `"openjiuwen-symphony"`.
- Only files under `crates/openjiuwen-symphony/` may change; the root manifest, sibling crates, and `docs/` are read-only from this node.
- Depend downward only: `openjiuwen-common`, `openjiuwen-foundation`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-session`, and the `openjiuwen-core` umbrella.
- No `unsafe`; no `println!`/`eprintln!` (use `tracing`); no panics on user input (no `unwrap`/`expect` outside tests).
- Every public item has a doc comment; files stay ≲1000 lines and split by responsibility (`src/runtime/mod.rs` is the documented composition-root exception; `src/flow/pack.rs`, `src/retrieval/search/lexical.rs` and `src/retrieval/search/render/expose.rs` sit just over the soft cap because their in-file `#[cfg(test)]` suites stay beside the logic).
- All public data types are `serde`-serializable with `snake_case` field names, tolerant unknown fields, and empty collections as defaults.
- Async-first: tokio; `std::sync::RwLock` is never held across `.await` (snapshot, drop, then await); blocking filesystem work uses tokio's `fs`.
- Errors use `openjiuwen_common::error::Error` only; validation errors name the offending field.
- Determinism: no wall-clock ordering, no hash-map iteration order in outputs; ties break on qualified id (or `worker_id`/record id in the catalog and experience layers); timestamps are metadata only.

## Design Decisions

- **A real registry exists in Rust.** Python inverts discovery to the host (`CapabilityProvider` protocol); `CapabilityRegistry` is the host-side provider plus a write-through `AssetStore`.
- **`CapabilityDescriptor::available` defaults to `true`**, and the planner's start set filters on it, so a capability is plannable unless explicitly withdrawn.
- **`top_k == 0` means unlimited** in search requests and `SemanticIndex::search`, a Rust-only sentinel; Python clamps to `max(1, ...)`.
- **The LLM stages share one seam and one fallback switch**: `openjiuwen_retrieval::CompletionModel` keeps them model-agnostic, `FallbackMode::Deterministic` guarantees a documented offline answer and `FallbackMode::Error` surfaces model failures.
- **SKILL.md discovery is a scan, not a plugin protocol**: per-skill problems become diagnostics and discovered skills become ordinary descriptors for the registry, graph and planner.
- **The skill catalog reuses the crate's own retrieval**: `SkillFs` builds a `LexicalIndex` over discovered descriptors with the markdown body as `semantic_content`, and the inventory fingerprint hashes canonical JSON sorted by `(worker_id, skill_file)`.
- **The I/O-name normalization layer is auditable**: every rename, coercion, default, redaction or drop emits a `NormalizationDecision` or `NormalizationIssue` naming its field, and the vocabulary carries a version plus an order-independent content hash.
- **Two retrieval paths over one catalog**: BM25 keyword search is the deterministic default, and `llm::TreeRetriever` adds Python's LLM-guided tree walk with the same lexical ranking as its fallback.
- **Semantic search is additive**: `SearchHit` carries the embedding-backed score, `CapabilityRegistry::revision()` drives re-embedding, and `HashEmbedding` keeps it deterministic in tests.
- **Experience ranking is embedding-optional**, persists items rather than vectors, and merges on the normalized pattern plus sorted skill set so repeated observations reinforce one record.
- **Flow distillation is structural, not narrative**: evidence groups on its cycle-broken successful-edge signature, versions are content-addressed and immutable, and a model can only write prose.
- **A SkillPack is a frozen, reviewable artifact**: only an `active` + `verified` linear chain can be packed, the package id hashes canonical materials, and a static ten-check review decides the verdict without a model or network.
- **Redaction happens where text enters an artifact**, not at render time, and the review's `secrets` check re-runs the redactor as the last line of defence.
- **Evaluation separates scored from scoreless verdicts**: `MetricScore::score` is optional and `status` is explicit, so latency stays `Observed` and a missing case is `NotApplicable` rather than a zero.
- **One judge implementation**: `evaluation::LlmJudgedMetric` delegates to `llm::LlmJudgeMetric`, so the prompt, the single repair retry and the deterministic rubric live in one place and static evaluation stays the offline default.
- **Planning is a bounded graph search** over Python's score formula and rank tuple; `PlanExecutor` compiles a plan into an `openjiuwen_graph::CompiledGraph` while `CapabilityGraph`/`GraphArtifactStore` keep the JGF document semantics.
- **`SemanticScorer` has two implementations**: `LexicalSemanticScorer` is the deterministic default and `llm::LlmSemanticScorer` adds model scores with it as fallback.
- **Object storage is the third artifact backend, not a new abstraction**: `ObjectGraphArtifactStore` reuses the revision envelope and key layout of `FileGraphArtifactStore`, and the GCS JSON API and Blob REST transports are Rust-only additions over `openjiuwen_common::http` with hand-rolled HMAC-SHA256 and XML scanning.
- **Revisions are immutable and atomically published**: every write (flow stores, experience indexes, graph artifacts, evidence lines, observation revisions) uses a temp sibling plus rename, an object store writes the revision before the pointer, publications serialize through an async mutex, and a generated version bumps past pinned ids so an explicit `v2` never breaks a later publish.
- **Timestamps never decide behavior**: `created_at`, `last_hit_at`, `observed_at` and `generated_at` are metadata, and ordering comes from sorted ids or content hashes.
- **Evolution feedback is an EMA on observed edge weight**, reducing Python's append-only evidence to one observable weight update.
- **The mutation contract is id-based and journaled**: the digest covers the declared batch rather than the payloads, so an id batch replays even if descriptor content changed, a conflicting digest for a used id is rejected rather than answered, and the journal is opt-in and separate from the artifact store (in-memory by default, surviving rebuilds when configured).
- **SwarmFlow generation is content-addressed and validator-gated**: the identifier is `combo-<sha256(materials)[:12]>`, the script enters the pack's materials, and `code_safety` re-validates it; `FlowEngine::close` is a no-op because every write is eager.
- **Trace ingestion is deterministic, not statistical**: grouping is skill set plus token Jaccard with a deterministic centroid, no RNG and no clock.
- **Capability-tree grouping is a deterministic substitute**: classification → first tag → leading name token under `max_children`, and every LLM/k-means `GroupingStrategy` is a typed `Error::Unsupported` rather than a silent fallback.
- **The capability-tree index publishes a fixed three-file contract** in a fixed order through per-file temp+rename after up-front validation (`TreeSchema` makes `blank_cid`/`duplicate_cid`/`missing_parent`/`cycle`/`excessive_depth`/`too_many_nodes` fatal), so a rejected build leaves the previous index intact.
- **The query-time runtime ranks field-weighted BM25 and blends embeddings optionally**: the blend weight defaults to `0.0`, which reproduces the lexical order exactly.
- **The matcher adapter is a model-free protocol layer**: `orchestration::matcher` renders the compact request and expands/validates the reply, while `llm::OntologyMatcher` owns the model call.
- **The matcher cache is opt-in and signature-keyed** over the candidate, both fingerprint digests and `matcher_signature()`, and a cache failure degrades to a model call rather than stale matches.
- **Observation state is a JSONL append log plus immutable content-hashed revisions**: `evolution/evidence.jsonl` and per-scope `observations/<rev>/overlay.json` / `merged/<rev>/manifest.json` replace Python's SQLite table and file locks, with `tokio::fs` renames and an async mutex serialising writers; the crate declares no `sqlx`.
- **Observation revision ids ignore `generated_at`** and aggregation runs inside `submit` and `flush`: identical state reproduces the same ids, republishing an id with different content is a validation error, a failed projection is warned and surfaced by the next `flush()`, and `close()` is a no-op.
- **Static edge identity is a point edge whose content hashes come from node metadata**, because `CapabilityGraph` carries no `capability_hashes` map; a changed body invalidates prior evidence and a deleted endpoint orphans it.
- **Observation timestamps are hand rolled** in `observation::timestamp` (Hinnant civil-date arithmetic) because `chrono` is not a dependency.

## Deferred Scope

Two kinds of gaps remain: typed `Error::Unsupported` entry points and pure-logic Python areas that are not ported yet (absent, not stubbed).

| Typed deferral | Why deferred |
|---|---|
| `retrieval/` — `transformers`/`vLLM` logit-based branch selection | Needs a GPU inference stack; the code-book/prefix-trie core, the LLM tree walk over `CompletionModel`, the lexical/embedding selectors and the flat retriever are implemented, so only `crate::deferred` answers |
| `retrieval/build/` — faiss/k-means and LLM grouping strategies | `GroupingStrategy::{FaissKmeans, Kmeans, LlmGrouping}` return `Error::Unsupported` (native ML stack / model prompt); deterministic classification/tag/name-token grouping is the default |
| `experience/` — faiss IVF index training | Needs faiss' k-means/inverted lists; `ExperienceIndex` implements the exact flat cosine index instead, so `ivf`/`ivfpq` names return `Error::Unsupported` while every item stays searchable |
| GCS service-account keys (`GcsObjectStore::from_service_account_file`) | Signing an RS256 JWT needs an RSA signer the offline workspace does not ship; OAuth bearer tokens, anonymous access and emulators work |
| Azure AD client credentials (`AzureBlobStore::from_client_credentials`) | An identity-endpoint token exchange plus a token cache is out of scope; SharedKey, SAS and anonymous cover the artifact path |

Pure-logic Python areas not ported:

- `orchestration/graph/artifacts.py`'s artifact-embedded journal and any diffing beyond the request contract — the request digest, durable journal, source-revision CAS and change-set validation are implemented (see [`orchestration::mutation`](src/orchestration/mutation.rs) and `runtime::mutations`); the journal is a separate JSON document, its digests are not chained over the file, and `source_revision` is optional.
- `orchestration/graph/matcher/cache.py`'s file lock and pending-merge batching — the persistent cache, identity digests, endpoint hashing and the request/response protocol are implemented (see [`orchestration::matcher`](src/orchestration/matcher/mod.rs)).
- `retrieval/search`'s remaining pure-logic gaps — Python's `search_terms` plural/verb term-family ranking, `casefold()`, the legacy `progressive.py` direct recursion, branch parallelism, the fixed-prefix-cache rejection, S3/HTTP index materialization and `service/serving.py`; the runtime reads a local index directory, walks children sequentially and clamps an over-large `top_k` (see [`retrieval::search`](src/retrieval/search/CONTEXT.md)).
- `agent/retrieval_toolkit.py` — the agent-facing browsing tools over the query-time runtime (tree browsing, subtree disclosure rendering to an agent); the runtime itself is implemented and the toolkit is not ported.
- `retrieval/build/tree/{grouping,expansion,repair,equivalence,one_shot}.py` LLM paths, model-written branch descriptions, `tree_index.html`, Python's zip/S3/HTTP item sources and its `SKILLSMP_ROOT_CATEGORIES` taxonomy — the build half substitutes deterministic grouping, a schema guard, cid-derived branch descriptions, and local-directory/inventory/descriptor/JSONL item sources.
- `experience/cluster.py`'s faiss k-means and LLM pattern distillation — replaced by deterministic token-Jaccard clustering and the collector's similarity merge.
- `discovery`'s Python filesystem caching (per-inode parse cache, single-flight scans, `id()`-keyed inventory cache) and its `tree`/`fresh`/`stale` index states — `SkillFs` is an immutable snapshot and `SkillIndexResult::skillfs_layout` is always `flat`.

Every other area is implemented with a deterministic offline substitute rather than stubbed: the artifact backends, discovery and its toolkit, flow distillation and packaging, the mutation contract and journal, the experience layer, evaluation, LLM matching with the matcher protocol, I/O-name normalization, observation and merged revisions, and the capability-tree build and query runtime.
Typed deferrals are proven, not just documented: `tests/deferred.rs` asserts the stub shape, and `tests/{artifact_gcs,artifact_azure,experience_index,retrieval_build}.rs` assert the remaining GPU/vendor boundaries.

## Error Strategy

- `Error::Validation` — malformed user input: blank ids/names/patterns/trace ids, illegal graph edge, bad version or meta name, draft/unverified/non-chain recipe, blank bucket, unknown artifact transport detail, runtime in the wrong state, a corrupted experience-index file, a malformed match cache, an illegal observation input.
- `Error::NotFound` — explicit lookups that must exist (packing an unknown recipe, a pinned index artifact or merged revision); registry, store, and catalog reads return `Option`.
- `Error::Graph` — plan execution failures, wrapping the failing step id and the runner error message.
- `Error::Unsupported` — the remaining GPU/ML and vendor-credential boundaries: `vLLM` logit selection, faiss IVF/faiss-k-means/LLM grouping, GCS service-account signing, Azure AD client credentials, and execution without a configured runner or semantic index.
- `Error::Llm` — model call or unusable-reply failures surfaced when `FallbackMode::Error` is configured; the deterministic mode converts them into the stage's documented fallback instead.
- `Error::Validation`/`Error::Retrieval` — argument and backend failures propagated from `openjiuwen-retrieval`; they are never swallowed.
- Artifact/store/flow/observation I/O failures map to `Error::Config` (invalid root or endpoint) or `Error::External` (I/O); malformed individual records are warned and skipped rather than aborting a scan.
- A match-cache read/write failure is warned and degrades to a model call; it never yields stale matches and never hides the model result.

## Test Strategy

Unit tests live beside modules (`#[cfg(test)] mod tests`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.

| Suite | Pins |
|---|---|
| `models` | serde round-trips, `Card` impl, schemas, content hash, datatype `can_feed`, naming helpers. |
| `normalization` | I/O-name folding, vocabulary build/hash/serde round-trip with version and size rejection, spec normalization from list and mapping inputs, issue/decision coverage, redaction, depth cap. |
| `registry` | CRUD, validation, ordering, persistence, `load`/`reload`, revision counter. |
| `search` | BM25, filters, `top_k`, filter-only ordering, `HashEmbedding` semantic index. |
| `graph` | graph from descriptors, edge validation/upsert, incident-edge removal, JGF round-trip. |
| `artifact` | memory/file stores, immutability, `current` pointer, atomicity, illegal versions, reopen hydration. |
| `object_artifact` | numbered revisions, `current` pointer/status, reopen persistence, object keys, pinned-version skipping, blank bucket, malformed `current.json`. |
| `artifact_gcs`, `artifact_azure` | publish → read → status → versions against a wiremock emulator, auth headers (bearer / `SharedKey` / SAS / anonymous), pagination, 404 and endpoint failures. |
| `planner` | candidate strategies/ordering, plan selection, filters, depth, scoring/ties, statuses, serialization, `stages`. |
| `executor` | compile + execute with a fake runner, per-step state keys, input overlay, `max_steps`, `fail_fast=false`, `UnsupportedRunner`. |
| `runtime` | lifecycle idempotency, invoke, build revisions, mutation idempotency and digest/CAS shape, journal persistence across a rebuild, evolution EMA, lazy re-embedding, wrong-state errors, snapshot. |
| `mutation_journal` | request preparation and rejections, pinned digests, change-set validation, round-trip/reopen, atomic layout, malformed documents. |
| `discovery` | scans, front-matter diagnostics, content-hash stability, inventory fingerprint, catalog search/excerpts, disabled skills, symlinks, settings. |
| `discovery_toolkit` | the `skill_index` tool: category tree, paths, pinned rows/summaries, the validation matrix, list/search/read, pagination, cursors, pipelines, notices, card, byte-identical repeats. |
| `discovery_toolkit_notice` | tool-id digest and validation, card normalization, pinned reminder text and budget fallback, registry baseline/delta/LRU, session rollback, serde round-trip. |
| `skill_prompt` | visible-name ordering/filtering, pinned snapshot text, multi-byte truncation bookkeeping, added/removed/changed notices and exact reminder wording. |
| `experience` | generated/explicit ids, dedupe merging, lexical and cosine ranking, skill filters, JSONL/SQLite persistence, id continuity. |
| `experience_index` | add/search/dedupe, ordering, skill-id flattening, digest/count/dimension corruption, persist round-trip, reopen id continuity, IVF/unknown-algorithm errors. |
| `experience_trace` | fixtures → exact `TraceRecord`s, idempotent re-parse, malformed-session tolerance, clustering/centroid determinism, collector behaviour, end-to-end bank search. |
| `retrieval_build` | item sources, grouping determinism, presets, schema failures, incremental `add`/`delete`/`rebuild_subtree`, a failed `add` leaving the builder unchanged, typed grouping deferrals. |
| `retrieval_artifacts` | the three-artifact set and shapes, byte-identical rebuilds, no `*.tmp` leftovers, rejected publications leaving nothing behind, `load_index` round-trip, corrupt/tampered/missing files. |
| `retrieval_search` | the facade and the layers under it: loading/publicization, scripted-model walk and lexical fallback, `FallbackMode::Error`, trace events, resumed walk, traversal bounds, flat retriever, ranking, `top_k`, rerun identity. |
| `retrieval_search_lexical` | the foundation layer on a hand-made `LoadedIndex`: item/branch documents, field/phrase/alternative rules, CJK snippets, scoping, blank queries, truncation, composition. |
| `retrieval_search_disclosure` | the render layer: exposure order and pinned codes, compact codes and validation, five reply shapes, the two-message prompt, character-boundary truncation, byte-identical reruns, flat mode, fingerprint stability. |
| `flow` | trace normalization, ingest rejection/idempotency, grouping/quality, versioning, rehydration, `SkillPack` render/save/load, review verdicts, install preparation, the rendered `run.py`, narrative fallback. |
| `flow_codegen` | the pinned `SwarmFlow` script, identifier stability and content sensitivity, escaping, validator rejects, member validation, the unsupported render target, script participation in the package identity. |
| `redaction` | every entry point, idempotence, non-ASCII input, the JSON depth cap and key order, and a flow run whose model emits a key and a PEM block. |
| `evaluation` | static metrics, structure conformance, mixed success rates, `Observed` latency, percentile interpolation, window boundaries, aggregation precedence, `cache_signature` stability, the judge, serde round-trips. |
| `llm` | every stage with a scripted model and with no model: tree walk/fallback, ontology consensus, scoring, fingerprint, judge repair, error mode. |
| `matcher_protocol` | compact request shape and positional ids, character-boundary truncation, reply expansion, every diagnostic, thresholds, the `OntologyMatch` bridge, byte-identical re-parse. |
| `matcher_cache` | identity digests, endpoint normalization, `sanitize_metadata`, byte-stable layout, replay with zero model calls, `force`, signature invalidation, degradation. |
| `observation` | contract validation, `observed_at` UTC normalisation, trace bridges, stable hashing and static-graph indexes, the evidence/revision layout with dedupe, cursor, content-derived ids and atomic writes. |
| `observation_service` | one scenario per qualification reason, duplicate idempotency, the runtime-only threshold, bounded session-deduped weights, high-water cursor, `safe_fallback`, quarantine/rebind. |
| `observation_runtime` | end-to-end submission through `SymphonyRuntime`, the `Unsupported` answer without a filesystem root, and rehydration after `close`. |
| `deferred` | the `vLLM` stub and the `ivf*` deferral return `Error::Unsupported`, and `error::is_unsupported` recognizes them. |

Tests are offline and deterministic: `tempfile` for filesystem stores and index directories, `HashEmbedding` for semantic search and the retrieval blend, `StaticCompletionModel` for scripted replies, `MemoryGraphArtifactStore` for observation roots, `wiremock` for cloud transports, no live network, no clock-dependent assertions.
Gates: `cargo fmt -p openjiuwen-symphony`, `cargo test -p openjiuwen-symphony`, and `cargo clippy -p openjiuwen-symphony --all-targets --all-features -- -D warnings` plus `--features sqlite` compile checks must pass.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, `crate_name`
- `./src/error.rs` → error helpers (`unsupported`, `validation`, `is_unsupported`)
- `./src/model/` → capability cards, fingerprints, data-type vocabulary, naming, I/O normalization; see [`src/model/CONTEXT.md`](src/model/CONTEXT.md)
  - `./src/model/descriptor.rs` → `CapabilityIo`, `CapabilityDescriptor`, schema derivation, content hash
  - `./src/model/fingerprint.rs` → `CapabilityFingerprint`, `SemanticProfile`, `FingerprintArtifact`
  - `./src/model/datatype.rs` → `DataTypeVocabulary`, `can_feed`, `is_subtype`
  - `./src/model/naming.rs` → `normalize_name_key`, fuzzy name distance/similarity
  - `./src/model/normalization/` → `normalize_io_name`, `IONameVocabulary`, `normalize_io_specs`, `sanitize_io_specs`, issues/decisions audit; see [`src/model/CONTEXT.md`](src/model/CONTEXT.md)
- `./src/store.rs` → `AssetStore`, `KvAssetStore` (memory/JSON/SQLite)
- `./src/registry.rs` → `CapabilityRegistry` CRUD + search
- `./src/search/`
  - `./src/search/mod.rs` → `SearchRequest`, `SearchHit`, filtering, BM25 orchestration
  - `./src/search/lexical.rs` → `tokenize`, `LexicalIndex`, BM25 scoring
  - `./src/search/semantic.rs` → `SemanticIndex` over `openjiuwen-retrieval`
- `./src/graph.rs` → `CapabilityGraph`, nodes/edges, JGF v2 conversion
- `./src/artifact.rs` → `GraphArtifactStore`, immutable revisions, `current` pointer
- `./src/object_artifact.rs` → `ObjectGraphArtifactStore` over `ObjectStore` (memory/local/S3/GCS/Azure), `ArtifactTransport`, `open`
  - `./src/object_artifact/gcs.rs` → `GcsConfig`, `GcsObjectStore` (GCS JSON API)
  - `./src/object_artifact/azure/` → `AzureBlobConfig`, `AzureBlobAuth`, `AzureBlobStore` (Blob REST, SharedKey, SAS, XML)
  - `./src/object_artifact/http.rs` → shared non-2xx error mapping and the hand-rolled HMAC-SHA256
- `./src/discovery/` → SKILL.md discovery, the lexical skill catalog, prompt snapshot, notice cursor; see [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md)
  - `./src/discovery/directory_toolkit/` → the `skill_index` tool: `SkillDirectoryView`, validation, ops, rendering, cursors; see [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md)
- `./src/retrieval/` → capability-tree build half: value objects + `SearchCursor`, the three-artifact IO, item scanning, deterministic grouping, presets, schema guard, node assembly, full/incremental builder; see [`src/retrieval/CONTEXT.md`](src/retrieval/CONTEXT.md)
- `./src/retrieval/search/` → query-time runtime: facade/config, `ProgressiveRetriever` + `RecursiveSearchEngine`, the five seams, lexical/embedding/model selectors, `FlatRetriever`, disclosure render/parse; see [`src/retrieval/search/CONTEXT.md`](src/retrieval/search/CONTEXT.md)
- `./src/experience/` → the experience bank, the persisted `ExperienceIndex`, trace ingestion, clustering and collector; see [`src/experience/CONTEXT.md`](src/experience/CONTEXT.md)
- `./src/flow/` → trace distillation, recipes, SkillPacks, review, `SwarmFlow` codegen, package rendering; see [`src/flow/CONTEXT.md`](src/flow/CONTEXT.md)
  - `./src/flow/codegen.rs` → `generate_swarmflow_script`, `swarmflow_identifier`, `python_string_literal`, `validate_generated_script`
  - `./src/flow/render.rs` → `SkillPackAdapter`, `SkillAdapter`, `RenderedArtifact`, `render_package`
- `./src/redaction/` → credential/PII redaction: scanner, sensitive-name matching, JSON redaction
- `./src/llm/` → LLM-guided stages with deterministic fallbacks, model/metadata identity, match cache; see [`src/llm/CONTEXT.md`](src/llm/CONTEXT.md)
- `./src/orchestration/` → planning, execution, the matcher protocol and the mutation contract; see [`src/orchestration/CONTEXT.md`](src/orchestration/CONTEXT.md)
  - `./src/orchestration/plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanStatus`, JGF projection, stages
  - `./src/orchestration/candidates.rs` → `CandidateGenerator`, relation candidates
  - `./src/orchestration/planner.rs` → `Planner`, `PlanConfig`, `SemanticScorer`, language helpers
  - `./src/orchestration/executor.rs` → `PlanExecutor`, `CapabilityRunner`, plan → `openjiuwen-graph`
  - `./src/orchestration/mutation.rs` → `MutationRequest`, `request_digest`, `MutationJournal`, `validate_mutation_against_inventory`
  - `./src/orchestration/matcher/` → the matcher request/response protocol adapter: `build_match_request`, `parse_match_response`, `ValidatedMatch`, `GraphDiagnostic`; see [`src/orchestration/matcher/CONTEXT.md`](src/orchestration/matcher/CONTEXT.md)
- `./src/evaluation/` → fail-soft evaluation: shared contract, static and trace metrics, suite aggregation, windows; see [`src/evaluation/CONTEXT.md`](src/evaluation/CONTEXT.md)
- `./src/runtime/` → `SymphonyRuntime`, config, lifecycle, evolution, mutations; see [`src/runtime/CONTEXT.md`](src/runtime/CONTEXT.md)
  - `./src/runtime/mod.rs` → lifecycle, planning/invocation, lazy semantic sync, evolution
  - `./src/runtime/mutations.rs` → the mutation path, `GraphMutationResult`, `MutationStatus`, journal-backed replay and source-revision CAS
- `./src/observation/` → the graph-evolution evidence contract, identity, append-only evidence log, immutable revisions and the `GraphObservationService`; see [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md)
  - `./src/observation/contracts/` → `GraphEvolutionInput`, evidence/enums, `validate`, `normalized`, trace bridges; see [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md)
  - `./src/observation/identity.rs` → `stable_hash`, `EdgeIdentity`, `StaticGraphIndex`
  - `./src/observation/store/` → the JSONL `EvidenceStore` and the per-scope `RevisionStore`; see [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md)
  - `./src/observation/service/` → qualification, aggregation, rebinding, snapshot/fallback; see [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md)
  - `./src/observation/payload.rs`, `timestamp.rs` → sanitised payload builder and the hand-rolled RFC 3339 layer
- `./src/deferred.rs` → the remaining GPU-only `Error::Unsupported` stub (`vLLM` logit selection)
- `./tests/` → behavior suites: `models`, `normalization`, `registry`, `search`, `graph`, `artifact`, `object_artifact`, `artifact_gcs`, `artifact_azure`, `planner`, `executor`, `runtime`, `mutation_journal`, `discovery`, `discovery_toolkit`, `discovery_toolkit_notice`, `skill_prompt`, `experience`, `experience_index`, `experience_trace`, `retrieval_build`, `retrieval_artifacts`, `retrieval_search`, `retrieval_search_lexical`, `retrieval_search_disclosure`, `flow`, `flow_codegen`, `redaction`, `evaluation`, `llm`, `matcher_protocol`, `matcher_cache`, `observation`, `observation_service`, `observation_runtime`, `deferred`
