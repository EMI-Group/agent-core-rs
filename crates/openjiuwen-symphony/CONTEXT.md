# openjiuwen-symphony

## Intent

The capability-asset engine of openJiuwen, mapping to the Python package `openjiuwen.symphony` (v0.1.20, ~52k lines, 201 files): Python's host-driven engine — fingerprint assets, index them, build a capability graph, plan compositions with an LLM, distil observed flows into skills — ported end to end, with every area routed in `## API Surface`.
Every LLM-guided stage (tree grouping, shaping, progressive retrieval, ontology matching, semantic scoring, fingerprint extraction, judging, the package reviewer, flow narratives) sits behind the `CompletionModel` seam, `FallbackMode::Deterministic` keeps the crate fully usable offline with no model configured, and every stage reports its provenance (`StageProvenance` or the stage's equivalent) so a degraded answer is data, not only a log line.
Everything is offline, deterministic and free of external services by default; credentials are redacted before text reaches an artifact.
Four typed `Error::Unsupported` gaps remain: `transformers`/`vLLM` logit selection in [`deferred`](src/deferred.rs), the sync single-shot entry points for the three runtime grouping strategies (`group_items`/`IndexBuilder::build`, which name `group_items_with`), Azure AD client credentials (`AzureBlobStore::from_client_credentials`), and execution without a configured capability runner or semantic index.
Unported pure-logic Python areas are listed under `## Deferred Scope`.

## API Surface

All paths are crate-relative; every public item is documented (`missing_docs` is on with `cargo clippy -D warnings`).
Each block is a routing entry — role plus crate-level entry points; the linked `CONTEXT.md` carries the full surface, formulas and rules.

### `error` (`src/error.rs`)

```rust
pub use openjiuwen_common::error::{Error, ErrorKind, Result};
pub fn unsupported(feature: &str) -> Error;                         // Unsupported + Deferred Scope pointer
pub fn validation(field: &str, reason: impl fmt::Display) -> Error; // Validation "<field>: <reason>"
pub fn is_unsupported(err: &Error) -> bool;
```

### `model` (`src/model/`)

Capability cards and their deterministic derived views: `CapabilityDescriptor`/`CapabilityKey`/`CapabilityIo`, the fingerprint types, `DataTypeVocabulary`/`normalize_type`, the naming helpers, and the ingest layer (`normalize_io_name`, `IONameVocabulary`, `normalize_io_specs*`, `sanitize_io_specs`); `semantic_content` is not serialized and `available` defaults to `true`.
See [`src/model/CONTEXT.md`](src/model/CONTEXT.md).

### `store` (`src/store.rs`)

`AssetStore` (`get`/`put`/`remove`/`list`/`clear`, sorted by qualified id) plus `KvAssetStore` (`in_memory`, `json_file`, `sqlite`, `with_prefix`); keys are `<prefix><qualified_id>` with descriptor JSON as value, and `get`/`list` warn and skip malformed entries instead of failing the scan.

### `registry` (`src/registry.rs`)

`CapabilityRegistry` (`with_store`, `load`/`reload`, `register`, `get`, `remove`, `list`, `search`, `revision`) is authoritative in memory, write-through to its optional store, and never holds its `RwLock` across an `.await`; `revision()` advances only on a real mutation, so the semantic index re-embeds only when the catalog moved.

### `search` (`src/search/`)

`SearchRequest`/`SearchHit`/`search_descriptors` filter → rank → apply `min_score` → truncate over a per-search `LexicalIndex` (BM25) or the `SemanticIndex`; `top_k` defaults to 10 with `0` = unlimited, filter-only requests score `0.0`, `metadata_filter` uses `openjiuwen_foundation::store::filter::matches`, and ties break on qualified id.

### `graph` (`src/graph.rs`)

`CapabilityGraph` with `CAN_FEED`: `from_descriptors`, node/edge mutation (`upsert_node`, `remove_node` dropping incident edges, `add_edge`/`remove_edge`, `outgoing`/`incoming`/`roots`/`leaves`) and JGF v2 conversion; `add_edge` rejects a missing endpoint, a self-loop, a blank relation and a non-finite or out-of-range weight with `Error::Validation`.

### `artifact` (`src/artifact.rs`)

`GraphArtifactStore` (`publish`, `read` with `None` = current, `status`, `versions`) over immutable `GraphRevision`s, implemented by `MemoryGraphArtifactStore` and `FileGraphArtifactStore`; republishing a version is `Error::Validation`, a `None` version publishes `v{n+1}`, and the file backend writes the revision then `current.json` through temp+rename so no partial file is visible.

### `orchestration` (`src/orchestration/`)

Deterministic candidates, bounded planning, plan execution over `openjiuwen-graph` and the mutation contract: `CandidateGenerator`, `Planner`, `SemanticScorer`, `PlanExecutor`/`CapabilityRunner`/`UnsupportedRunner`, `MutationRequest`/`MutationJournal`, and the model-free matcher adapter.
`StepOutcome` records the failure class (`error_kind`, `deferred`), `PlanExecution::{deferred_steps, has_deferrals}` expose that subset, and `ExecutorConfig::fail_on_deferral` (default `false`) promotes the first `Unsupported` step into a typed run error naming it, while every other degradation is recorded with its reason and a `warn!`.
See [`src/orchestration/CONTEXT.md`](src/orchestration/CONTEXT.md) and [`src/orchestration/matcher/CONTEXT.md`](src/orchestration/matcher/CONTEXT.md).

### `evaluation` (`src/evaluation/`)

Fail-soft evaluation over cards and traces: `Metric`/`MetricScore`/`MetricStatus`/`EvaluationReport`, six static and five trace metrics, `EvaluationSuite` (`default_suite`/`with_llm_judge`) and `SYMPHONY_EVALUATION_PROTOCOL`; a metric either scores or reports a status, so latency stays `Observed` and a missing case is `NotApplicable` rather than a zero.
See [`src/evaluation/CONTEXT.md`](src/evaluation/CONTEXT.md).

### `flow` (`src/flow/`)

Trace ingestion into `FlowTrace`/`ExecutionGraph`, structural distillation into versioned `ExperienceRecipe`s, content-addressed `SkillPack` packaging with a ten-check `Reviewer`, the opt-in `LlmPackageReviewer`, `SwarmFlow` codegen (`generate_swarmflow_script`, `validate_generated_script`) and `render_package`.
The review stages compose fail-closed: the deterministic `Reviewer` decides, `LlmPackageReviewer` may only add findings (a model `pass` can never clear a `fail`, and `secrets`/`code_safety` stay authoritative), an unusable reply is `Error::Llm` under `FallbackMode::Error` and a marked deterministic verdict under `Deterministic`, the stage is skipped when the deterministic verdict is not approved, and narratives are redacted before they enter an artifact.
See [`src/flow/CONTEXT.md`](src/flow/CONTEXT.md).

### `redaction` (`src/redaction/`)

`redact_sensitive_text`, `redact_common_credentials`, `sanitize_distilled_text`, `is_sensitive_name`/`normalize_sensitive_name`, `redact_sensitive_json` and `redact_private_json`: a hand-rolled, idempotent, char-boundary-safe scanner over PEM blocks, `Bearer` tokens, JWTs, `sk-…` keys, AWS access key ids and sensitive `key=value` assignments, with JSON redaction capped at depth 64 and key order preserved.

### `llm` (`src/llm/`)

The single `CompletionModel` seam and its five stages — `TreeRetriever`, `OntologyMatcher`, `LlmSemanticScorer`, `FingerprintExtractor`, `LlmJudgeMetric` — each with a documented deterministic fallback under `FallbackMode::Deterministic` (`FallbackMode::Error` surfaces failures as `Error::Llm`); every stage marks its answer with `StageProvenance` (`Model`/`DeterministicFallback`/`NoModel`) on a serde-defaulted wrapper, and legacy payloads read as `NoModel`.
`identity` derives model/metadata digests and `cache` is the opt-in persistent `MatchCache` (`with_cache`, `MatchOptions { force }`) keyed by candidate, both fingerprint digests and `matcher_signature()`.
See [`src/llm/CONTEXT.md`](src/llm/CONTEXT.md).

### `experience` (`src/experience/`)

The experience layer: `ExperienceBank`, the persisted `ExperienceIndex`, the outbound `ExperienceRetriever` (`RankingProvenance`), `TraceEvaluator`, `EmbeddingClient`, and the input half `TraceStore`, `cluster_traces`/`cluster_traces_with_embedding` and `ExperienceCollector`.
`ExperienceIndex` accepts `flat` and the IVF spellings (`ivf`, `ivfflat`, `ivfpq`, `ivf<n>`, `ivf<n>,pq<m>`) and runs deterministic coarse quantisation with `nprobe`-list probing and exact rerank inside the probed lists, so `nprobe >= nlist` reproduces the flat neighbours; centroids and assignments persist in `vector/ivf.json` and PQ is parsed-and-recorded rather than applied.
See [`src/experience/CONTEXT.md`](src/experience/CONTEXT.md).

### `discovery` (`src/discovery/`)

SKILL.md discovery (both Python scan modes, per-skill diagnostics instead of scan failure, SHA-256 content hashes, an order-independent inventory fingerprint), the lexical catalog `SkillFs`, the prompt surface (`visible_skill_names`, `prompt_snapshot`, `notice_state`) and the read-only `skill_index` tool (`SkillIndexToolkit`, `SkillNoticeState`).
`SkillIndexResult` derives its observability: `skillfs_layout` (`flat`/`tree`) from the skill-file depth and `index_state` (`missing` → `fresh` → `stale`) from an `IndexBaseline` compared with the current inventory fingerprint.
See [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md).

### `object_artifact` (`src/object_artifact.rs`)

`ObjectGraphArtifactStore` implements `GraphArtifactStore` over an `ObjectStore`, mirroring the filesystem key layout, writing the revision before the `current` pointer, serialising publications through an async mutex and skipping pinned ids; `ArtifactTransport` (`ObjectStore`, `S3`, `Gcs`, `AzureBlob`) is resolved by `ObjectGraphArtifactStore::{s3, gcs, azure_blob, open}` and a blank bucket is `Error::Validation`.
`GcsObjectStore` authenticates in a fixed precedence — an explicit OAuth bearer token, else an RS256 JWT-bearer minted from a service-account key file, else anonymous — minting lazily and caching until 60 s before expiry; `AzureBlobStore` speaks Blob REST with SharedKey HMAC-SHA256, a SAS token or anonymously, and key material and tokens never reach `Debug` or error text.

### `retrieval` (`src/retrieval/`)

Both halves of the capability-tree module: the index BUILD half (`types`, `artifacts`, `build`) and the query-time runtime (`search`).
The build half resolves item sources, groups by rule (classification → tag → name token), deterministic k-means or `LlmGrouping`, optionally runs the `one_shot`/`expansion`/`repair`/`equivalence` shaping stages, publishes exactly three artifacts (`tree_index.yaml`, `catalog.jsonl`, `manifest.json`) atomically, and exposes `IndexBuilder::{build, build_with_runtime, open, add, add_with_runtime, delete, rebuild_subtree, write}`.
`group_items_with` reports `GroupingProvenance` (`Rule`/`Embedding`/`Model`/`DeterministicFallback`), the sync `group_items`/`IndexBuilder::build` answer a typed `Unsupported` naming it, no model failure is ever reported as a rule success, and `tree::visualizer` renders an opt-in `tree_index.html` (`BuildOptions::write_html`) that leaves the default publish byte-identical.
The query half's shape is `Retriever::{load, from_loaded_index, search, search_details}`, publicizing leaf cids into `worker_id`s over `ProgressiveRetriever`/`RecursiveSearchEngine`, the `TopKSelector`s, `FlatRetriever` and disclosure render/parse; every selection reports `SelectionProvenance` and the no-model degradation has its own `MODE_LEXICAL_NO_MODEL`.
See [`src/retrieval/CONTEXT.md`](src/retrieval/CONTEXT.md) and [`src/retrieval/search/CONTEXT.md`](src/retrieval/search/CONTEXT.md).

### `agent` (`src/agent/`)

The agent-facing half of the retrieval runtime: `RetrievalToolkit` over `crate::retrieval::search::Retriever` with `list`/`browse`/`search`/`read`, hash page cursors pinned to the index fingerprint, clamping, abstention (`blank_query`/`no_results`) and a typed validation matrix, plus the model-facing surface (`RetrievalToolCard`, `tool_input_schema()`, `SKILL_RETRIEVAL_TOOL_NAME`).
See [`src/agent/CONTEXT.md`](src/agent/CONTEXT.md).

### `runtime` (`src/runtime/`)

The composition root: `RuntimeState`, `SymphonyConfig`, `SymphonyRuntimeBuilder` and `SymphonyRuntime` (`open`/`start`/`stop`/`close`, `build`/`graph`/`read_graph`, `plan`/`invoke`/`search`/`semantic_search`/`reindex_semantics`, `mutate`, `submit_evolution`, observation and snapshot methods), plus `InvokeRequest`/`InvokeResponse`, `MutationOperation`/`MutationStatus` and `GraphMutationResult`.
`start`/`stop` are idempotent, `close` flushes artifacts, and the mutation order is replay → digest conflict → `Running` gate → change-set/CAS → apply → publish → record, where a conflicting digest or a stale `source_revision` is `Error::Validation`; `submit_evolution` applies an EMA, `semantic_search` re-embeds only when `revision()` moved, and the observation methods are `Unsupported` without a `graph_artifact_root`.
See [`src/runtime/CONTEXT.md`](src/runtime/CONTEXT.md).

### `observation` (`src/observation/`)

Caller-supplied execution evidence distilled into derived, immutable observation and merged graph revisions, never rewriting the static `can_feed` graph: `GraphEvolutionInput`/`TraceEvidence`/`TaskOutcome` (contract v2) feed `GraphObservationService::{new, submit, get_snapshot, flush, reproject, reproject_all, publish_safe_fallback, close}` over the JSONL `EvidenceStore`, the per-scope `RevisionStore` and `stable_hash`.
Evidence is qualified first (an `audit_only` verdict carries its reason and never re-qualifies) and only `verified_success` plus orchestration-attributed `verified_failure` edges count: `runtime_weight = 1.0 + clamp(0.05 * (success - failure), -1.0, 1.0)`.
See [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md).

### `deferred` (`src/deferred.rs`)

One API-shaped stub remains and every entry point returns `Error::Unsupported` via [`error::unsupported`]: the `transformers`/vLLM logit-selection backend, whose data types stay ordinary serde types.

## Notes for Agents

- Intra-doc links in a module-level (`//!`) doc block do not resolve names defined or re-exported in that same file module; write them as labeled links to absolute paths, e.g. `[`AssetStore`](crate::store::AssetStore)`.
- `cargo doc -p openjiuwen-symphony --no-deps` is the definition of done for doc links: it must print no warnings.

## Constraints

- Cargo: `edition 2021`, `rust-version 1.85`, `publish = false`, `#![forbid(unsafe_code)]`; `default = []` and the sole feature `sqlite` only enables `store::KvAssetStore::sqlite` and `experience::SqliteExperienceStore` via `openjiuwen-foundation/sqlite`.
- Dependencies additionally include `serde_yaml_ng` (front matter, rendered manifests), `bytes` (object payloads), `base64` (SharedKey, PEM/JWT) and `rsa` 0.9 (`default-features = false, features = ["std", "pem"]`, pinned inline because the workspace manifest lacks it) for the GCS RS256 assertion; `wiremock` backs the GCS/Azure/token suites.
- Cloud transports use `openjiuwen_common::http` (never raw `reqwest`); no `hmac` and no XML crate exists offline, so SharedKey HMAC-SHA256 and the Azure list XML scanner are hand rolled and unit-tested (RFC 4231 vectors).
- GCS signing uses `rsa::Pkcs1v15Sign` with an explicit SHA-256 `DigestInfo` prefix because the workspace `sha2` has no `oid` feature, so `SigningKey::<Sha256>::new` does not compile; the result is signature-identical to the typed helper and verified against a real key in the tests.
- The umbrella crate `openjiuwen` re-exports this crate as `openjiuwen::symphony`; `crate_name()` returns `"openjiuwen-symphony"`.
- Only files under `crates/openjiuwen-symphony/` may change; the root manifest, sibling crates and `docs/` are read-only from this node.
- Depend downward only: `openjiuwen-common`, `openjiuwen-foundation`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-session`, and the `openjiuwen-core` umbrella.
- No `unsafe`; no `println!`/`eprintln!` (use `tracing`); no panics on user input (no `unwrap`/`expect` outside tests); every public item has a doc comment.
- All public data types are `serde`-serializable with `snake_case` field names, tolerant unknown fields and empty collections as defaults.
- Async-first: tokio; `std::sync::RwLock` is never held across `.await` (snapshot, drop, then await); blocking filesystem work uses tokio's `fs`.
- Errors use `openjiuwen_common::error::Error` only, and validation errors name the offending field.
- Determinism: no wall-clock ordering, no hash-map iteration order in outputs; ties break on qualified id (or `worker_id`/record id in the catalog and experience layers); timestamps are metadata only.
- Files stay ≲1000 lines; the documented over-cap exceptions keep their in-file tests beside the logic: `src/runtime/mod.rs` (composition root), `src/flow/pack.rs`, `src/retrieval/search/{lexical,selector}.rs`, `src/retrieval/search/render/expose.rs`, `src/llm/ontology.rs`, `src/agent/retrieval_toolkit/{validate,tests}.rs`.

## Design Decisions

- **A real registry exists in Rust** — Python inverts discovery to the host, so `CapabilityRegistry` is the host-side provider plus a write-through `AssetStore`.
- **`available` defaults to `true`** — the planner's start set filters on it, so a capability is plannable unless explicitly withdrawn.
- **`top_k == 0` means unlimited** — a Rust-only sentinel in search requests and `SemanticIndex::search`; Python clamps to `max(1, ...)`.
- **The LLM stages share one seam and one fallback switch** — `Deterministic` guarantees a documented offline answer and `Error` surfaces model failures.
- **Provenance is data, not log-only** — the `*Provenance` markers, `review_mode` and `StepOutcome::deferred` let a caller see a degraded answer without parsing logs, with serde defaults keeping old payloads readable.
- **A model reply is never trusted** — every grouping/shaping stage validates ids, slug and bounds item by item, and `Deterministic` marks the answer `DeterministicFallback` rather than presenting it as model output.
- **Capability-tree grouping has four real strategies** — the classification → tag → name-token rule (default), `LlmGrouping`, and embedding k-means for `Kmeans`/`FaissKmeans`; the runtime ones go through `group_items_with` with `GroupingProvenance`, and the sync entry points answer a typed `Unsupported`.
- **The capability-tree index publishes a fixed three-file contract** — validated up front (`TreeSchema` makes bad cids, cycles, depth and node-count violations fatal) and written per file through temp+rename, so a rejected build leaves the previous index intact; `tree_index.html` is opt-in, keeping the default publish byte-identical.
- **The experience index has two algorithms over one matrix** — flat cosine is exact, IVF adds deterministic coarse quantisation persisted in `vector/ivf.json`, `nprobe >= nlist` reproduces the flat neighbours, and PQ is recorded rather than applied.
- **A publish encodes only what changed** — a dirty-set cache keyed by id (seeded on `load`) plus `spawn_blocking` makes `n` appends cost `O(n)` encodings instead of `O(n²)`, and the cached quantiser is reused with each new id assigned to its nearest centroid; it is rebuilt only when absent, when `nlist` changed, on a removal, or when the catalog grew by `max(trained / 4, 64)` items.
- **A multi-file snapshot commits at `meta.json`** — unique fsynced temps, all temps written before the first rename, renames in dependency order with the manifest last, directory fsync and one async mutex per canonical directory, so an interrupted publish is a loadable old snapshot rather than a rejected mixture.
- **Trace ingestion is deterministic, not statistical** — k-means seeded from a content hash, RNG-free, centroid = the member closest to the mean, and no clock.
- **The package review's deterministic stage always wins** — a model can only add findings, never clear `secrets`/`code_safety`, and the stage is skipped when the deterministic verdict is not approved.
- **Redaction happens where text enters an artifact**, not at render time, and the review's `secrets` check re-runs the redactor as the last line of defence.
- **A SkillPack is frozen and reviewable** — only an `active` + `verified` linear chain packs, the id hashes canonical materials, and a static ten-check review decides without a model or network.
- **SKILL.md discovery is a scan, not a plugin protocol** — problems become diagnostics, skills become ordinary descriptors, and `SkillFs` reuses the crate's `LexicalIndex` with the markdown body as `semantic_content`.
- **The I/O-name layer is auditable** — every rename, coercion, default, redaction or drop emits a decision or issue naming its field.
- **Revisions are immutable and atomically published** — temp sibling plus rename, revision before pointer, one async mutex per publisher, and generated versions bump past pinned ids.
- **Object storage is the third artifact backend, not a new abstraction** — it reuses the revision envelope and key layout, with GCS and Blob transports over `openjiuwen_common::http` plus hand-rolled HMAC-SHA256 and XML scanning.
- **The mutation contract is id-based and journaled** — the digest covers the declared batch, a conflicting digest for a used id is rejected rather than answered, and the journal is opt-in and separate from the artifact store.
- **The matcher adapter is model-free protocol code**, and its cache is opt-in and signature-keyed; a cache failure degrades to a model call rather than stale matches.
- **Observation state is a JSONL append log plus immutable content-hashed revisions** — no SQLite table, no file locks; ids ignore `generated_at`, republishing an id with different content is a validation error, and `close()` is a no-op.
- **Static edge identity hashes node metadata** (the graph carries no `capability_hashes` map), so a changed body invalidates prior evidence and a deleted endpoint orphans it.
- **Timestamps never decide behavior** — ordering comes from sorted ids or content hashes, and RFC 3339 stamps are hand rolled because `chrono` is not a dependency.

## Deferred Scope

Two kinds of gaps remain: typed `Error::Unsupported` entry points and pure-logic Python areas that are not ported yet (absent, not stubbed).

| Typed deferral | Why deferred |
|---|---|
| `crate::deferred` — `transformers`/`vLLM` logit-based branch selection | Needs a GPU inference stack; the code-book/prefix-trie core, the LLM tree walk, the selectors and the flat retriever exist |
| Sync grouping entry points (`group_items`, `IndexBuilder::build`) for `LlmGrouping`/`Kmeans`/`FaissKmeans` | They need an embedder or a model the sync entry point lacks; they run through `group_items_with`/`build_with_runtime`, and the sync call names that entry point instead of falling back by rule |
| `retrieval/build/tree/shaping` under `FallbackMode::Error`, and `retrieval/search` without a model in `SelectionMode::Generate` | The runtime strategies have a documented deterministic fallback, so asking for strict mode is a typed error by design |
| Azure AD client credentials (`AzureBlobStore::from_client_credentials`) | Needs a live OIDC client-credentials exchange plus a token cache; SharedKey, SAS and anonymous cover the artifact path |
| `SymphonyRuntime`'s observation methods without a `graph_artifact_root`, and execution without a runner / semantic index | Those surfaces need a filesystem root or a wired runner, and every entry point says so instead of degrading |

Pure-logic Python areas not ported:

- `orchestration/graph/artifacts.py`'s artifact-embedded journal and diffing beyond the request contract — the digest, durable journal, CAS and change-set validation exist, but the journal is a separate document with unchained digests.
- `orchestration/graph/matcher/cache.py`'s file lock and pending-merge batching — the persistent cache, digests, endpoint hashing and the request/response protocol are implemented.
- `retrieval/search`'s remaining gaps — Python's `search_terms` term-family ranking, `casefold()`, the legacy `progressive.py` recursion, branch parallelism, the fixed-prefix-cache rejection, S3/HTTP index materialization and `service/serving.py`.
- `retrieval/build/tree`'s item sources and taxonomy — Python's zip/S3/HTTP sources and `SKILLSMP_ROOT_CATEGORIES` are unported, and model-written branch descriptions stay out (they are cid-derived unless a shaping stage supplies them).
- `experience/cluster.py`'s LLM pattern distillation and Python's extra `evaluator.py`/`retriever.py`/`embed.py` behaviour — the deterministic clustering, similarity merge, `TraceEvaluator`, `ExperienceRetriever` and `EmbeddingClient` cover the offline contract.
- `discovery`'s filesystem caching (per-inode parse cache, single-flight scans, `id()`-keyed inventory cache) — `SkillFs` is an immutable snapshot, so nothing on disk needs invalidation.
- `agent/retrieval_toolkit`'s live index operations — async build/status/cancel, publish locks, `update_skills` refresh and S3/HTTP materialization are unported; the toolkit reads a built index and delegates construction to `crate::retrieval::build`.
- `object_artifact`'s GCS credential discovery — `external_account`, `authorized_user`, the metadata server and `GOOGLE_APPLICATION_CREDENTIALS` are not read.

Every other area is implemented with a deterministic offline substitute rather than stubbed, and tests prove the boundaries: `tests/deferred.rs` pins the stub and sync-grouping shapes, `tests/{artifact_gcs,artifact_azure,experience_index,retrieval_build}.rs` the vendor boundaries and their successors, and each strict-mode path (`FallbackMode::Error`, `fail_on_deferral`, `SelectionMode::Generate` without a selector) its typed error.

## Error Strategy

- `Error::Validation` — malformed input: blank ids/names/patterns/trace ids, an illegal graph edge, a bad version or meta name, a draft/unverified/non-chain recipe, a blank bucket, an unknown transport detail, the runtime in the wrong state, a corrupted index file, a malformed match cache, an illegal observation input.
- `Error::NotFound` — explicit lookups that must exist (an unknown recipe, a pinned index artifact or merged revision); registry, store and catalog reads return `Option`.
- `Error::Graph` — plan execution failures, wrapping the failing step id and the runner error message.
- `Error::Unsupported` — the four remaining boundaries listed under `## Deferred Scope`.
- `Error::Llm` — model calls and unusable replies under `FallbackMode::Error` and in the strict stages; the deterministic mode converts them into the stage's documented, marked fallback instead.
- `Error::Config` — an invalid credential or key file (a GCS key missing `client_email`/`private_key`/`token_uri`, a malformed PEM) names the offending field, and the assertion and key material never appear in the message.
- `Error::Validation`/`Error::Retrieval` — argument and backend failures propagated from `openjiuwen-retrieval`; they are never swallowed.
- Artifact/store/flow/observation I/O failures map to `Error::Config` (invalid root or endpoint) or `Error::External`; malformed records are warned and skipped rather than aborting a scan.
- A match-cache read/write failure is warned and degrades to a model call; it never yields stale matches and never hides the model result.

## Test Strategy

Unit tests live beside modules (`#[cfg(test)] mod tests`); cross-module behavior lives in `crates/openjiuwen-symphony/tests/`.

| Suite | Pins |
|---|---|
| `models` | serde, `Card`, schemas, content hash, `can_feed`, naming. |
| `normalization` | folding, vocabulary bounds, spec normalization, issues/decisions, redaction, depth cap. |
| `registry` | CRUD, validation, ordering, persistence, `load`/`reload`, revision. |
| `search` | BM25, filters, `top_k`, filter-only ordering, semantic index. |
| `graph` | descriptors build, edge validation, incident-edge removal, JGF round-trip. |
| `artifact` | memory/file stores, immutability, `current`, atomicity, bad versions, reopen. |
| `object_artifact` | revisions, pointer/status, reopen, pinned skipping, blank bucket, bad `current.json`. |
| `artifact_gcs`, `artifact_azure` | wiremock emulator, auth headers, pagination, 404s, key parsing, JWT, one minted token. |
| `planner` | candidates, selection, filters, depth, scoring/ties, statuses, `stages`. |
| `executor` | fake runner, state keys, input overlay, `max_steps`, `fail_fast`, `UnsupportedRunner`, deferrals. |
| `runtime` | lifecycle idempotency, invoke, revisions, mutation idempotency/CAS, journal, EMA, lazy re-embedding. |
| `mutation_journal` | preparation/rejections, pinned digests, change-set validation, reopen, atomic layout. |
| `discovery` | scans, diagnostics, content hashes, fingerprint, catalog search, symlinks, settings. |
| `discovery_toolkit` | tree, paths, pinned rows, validation matrix, ops, pagination, cursors, notices, layout/index states. |
| `discovery_toolkit_notice` | tool-id digest, card normalization, reminder text/budget, LRU, session rollback. |
| `skill_prompt` | ordering, pinned snapshot, truncation, added/removed/changed notices. |
| `experience` | ids, dedupe, ranking, filters, persistence, retriever provenance, evaluator, embedding cache. |
| `experience_index` | add/search/dedupe, corruption, round-trip, IVF names, `nprobe >= nlist` = flat, probes, incremental encode counters and bounded publishes, cached-encoding reuse, injected mid-publish failure leaving a loadable snapshot, concurrent adds surviving a reopen. |
| `experience_trace` | exact `TraceRecord`s, re-parse, malformed sessions, k-means determinism, collector, bank search. |
| `retrieval_build` | sources, grouping, presets, schema failures, incremental ops, provenance, shaping, deferral, HTML. |
| `retrieval_artifacts` | three-artifact set, byte-identical rebuilds, no `*.tmp`, rejected publishes, corruption, HTML, the `MAX_CID_DEPTH` rejection of a 10 000-segment cid and an over-deep branch chain, and a normal five-level index still loading. |
| `retrieval_search` | loading, scripted walk and fallback, strict mode, traces, bounds, flat retriever, provenance. |
| `retrieval_search_lexical` | item/branch docs, field rules, CJK snippets, scoping, blank queries, truncation. |
| `retrieval_search_disclosure` | exposure order, codes, reply shapes, two-message prompt, truncation, fingerprint stability. |
| `flow` | normalization, ingest idempotency, grouping, versioning, rehydration, pack save/load, fail-closed LLM review. |
| `flow_codegen` | pinned script, identifier stability, escaping, validator rejects, unsupported render target. |
| `retrieval_toolkit` | four ops, pagination, cursor expiry, validation matrix, abstention, card/prompt snapshot. |
| `redaction` | entry points, idempotence, non-ASCII, JSON depth/order, model-emitted key and PEM. |
| `evaluation` | static metrics, conformance, `Observed` latency, percentiles, aggregation, `cache_signature`, judge. |
| `llm` | each stage with a scripted, failing and no model, `StageProvenance`, legacy defaults. |
| `matcher_protocol` | request shape and ids, truncation, reply expansion, diagnostics, `OntologyMatch`. |
| `matcher_cache` | digests, endpoint normalization, byte-stable layout, replay without model calls, `force`. |
| `observation` | contract validation, normalisation, bridges, hashing, evidence/revision layout, atomic writes. |
| `observation_service` | a scenario per qualification reason, duplicate idempotency, thresholds, weights, rebind. |
| `observation_runtime` | submission through `SymphonyRuntime`, `Unsupported` without a root, rehydration. |
| `deferred` | the `vLLM` stub, the sync grouping deferral, IVF open/search, the Azure deferral. |

Tests are offline and deterministic: `tempfile` for stores and index directories, `HashEmbedding` for semantic search and the retrieval blend, `StaticCompletionModel` for scripted replies, `MemoryGraphArtifactStore` for observation roots, `wiremock` for cloud transports, no live network and no clock-dependent assertions.
Gates: `cargo fmt -p openjiuwen-symphony`, `cargo test -p openjiuwen-symphony`, and `cargo clippy -p openjiuwen-symphony --all-targets --all-features -- -D warnings` plus `--features sqlite` compile checks.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, `crate_name`
- `./src/error.rs` → error helpers (`unsupported`, `validation`, `is_unsupported`)
- `./src/model/` → capability cards, fingerprints, data types, naming, I/O normalization; see [`src/model/CONTEXT.md`](src/model/CONTEXT.md)
  - `./src/model/descriptor.rs` → `CapabilityIo`, `CapabilityDescriptor`, schema derivation, content hash
  - `./src/model/fingerprint.rs` → `CapabilityFingerprint`, `SemanticProfile`, `FingerprintArtifact`
  - `./src/model/datatype.rs` → `DataTypeVocabulary`, `can_feed`, `is_subtype`
  - `./src/model/naming.rs` → `normalize_name_key`, fuzzy name distance/similarity
  - `./src/model/normalization/` → `normalize_io_name`, `IONameVocabulary`, `normalize_io_specs`, issues/decisions audit
- `./src/store.rs` → `AssetStore`, `KvAssetStore` (memory/JSON/SQLite)
- `./src/registry.rs` → `CapabilityRegistry` CRUD + search
- `./src/search/`
  - `./src/search/mod.rs` → `SearchRequest`, `SearchHit`, filtering, BM25 orchestration
  - `./src/search/lexical.rs` → `tokenize`, `LexicalIndex`, BM25 scoring
  - `./src/search/semantic.rs` → `SemanticIndex` over `openjiuwen-retrieval`
- `./src/graph.rs` → `CapabilityGraph`, nodes/edges, JGF v2 conversion
- `./src/artifact.rs` → `GraphArtifactStore`, immutable revisions, `current` pointer
- `./src/object_artifact.rs` → `ObjectGraphArtifactStore` over `ObjectStore`, `ArtifactTransport`, `open`
  - `./src/object_artifact/gcs.rs` → `GcsConfig`, `GcsObjectStore`, auth precedence
  - `./src/object_artifact/gcs/` → `key` (service-account JSON, PKCS#8), `signing` (RS256 assertion), `token` (lazy mint, cache)
  - `./src/object_artifact/azure/` → `AzureBlobConfig`, `AzureBlobAuth`, `AzureBlobStore` (Blob REST, SharedKey, SAS, XML)
  - `./src/object_artifact/http.rs` → non-2xx error mapping, hand-rolled HMAC-SHA256
- `./src/discovery/` → SKILL.md discovery, the lexical catalog, prompt snapshot, notice cursor; see [`src/discovery/CONTEXT.md`](src/discovery/CONTEXT.md)
  - `./src/discovery/directory_toolkit/` → the `skill_index` tool: validation, ops, rendering, cursors, derived `SkillFsLayout`/`IndexState`
- `./src/agent/` → the agent-facing retrieval toolkit: ops, cursors, validation matrix, card and prompt snapshot; see [`src/agent/CONTEXT.md`](src/agent/CONTEXT.md)
  - `./src/agent/retrieval_toolkit/` → `toolkit` (entry points), `ops` (list/browse/search/read), `render` (card/prompt), `validate` (request contract)
- `./src/retrieval/` → capability-tree build half: value objects + `SearchCursor`, three-artifact IO, item scanning, grouping, shaping, opt-in visualizer, presets, schema guard, node assembly, builder; see [`src/retrieval/CONTEXT.md`](src/retrieval/CONTEXT.md)
  - `./src/retrieval/build/grouping/` → `kmeans` (embedding clusterer) and `llm` (`LlmGrouping` over the model seam)
  - `./src/retrieval/build/tree/shaping/` → the `one_shot`/`expansion`/`repair`/`equivalence` stages
  - `./src/retrieval/build/tree/visualizer.rs` → the opt-in `tree_index.html` renderer and atomic writer
- `./src/retrieval/search/` → query-time runtime: facade/config, `ProgressiveRetriever`/`RecursiveSearchEngine`, the seams, selectors, `FlatRetriever`, disclosure; see [`src/retrieval/search/CONTEXT.md`](src/retrieval/search/CONTEXT.md)
- `./src/experience/` → the bank, `ExperienceIndex` (flat + IVF), trace ingestion, k-means clustering, the collector, `evaluator`/`retriever`/`embed`; see [`src/experience/CONTEXT.md`](src/experience/CONTEXT.md)
- `./src/flow/` → trace distillation, recipes, SkillPacks, review (deterministic + `review_llm`), `SwarmFlow` codegen, rendering; see [`src/flow/CONTEXT.md`](src/flow/CONTEXT.md)
  - `./src/flow/codegen.rs` → `generate_swarmflow_script`, `swarmflow_identifier`, `python_string_literal`, `validate_generated_script`
  - `./src/flow/render.rs` → `SkillPackAdapter`, `SkillAdapter`, `RenderedArtifact`, `render_package`
- `./src/redaction/` → credential/PII redaction: scanner, sensitive-name matching, JSON redaction
- `./src/llm/` → LLM-guided stages with deterministic fallbacks, identity digests, match cache; see [`src/llm/CONTEXT.md`](src/llm/CONTEXT.md)
- `./src/orchestration/` → planning, execution, the matcher protocol and the mutation contract; see [`src/orchestration/CONTEXT.md`](src/orchestration/CONTEXT.md)
  - `./src/orchestration/plan.rs` → `OrchestrationPlan`, `PlanStep`, `PlanStatus`, JGF projection, stages
  - `./src/orchestration/candidates.rs` → `CandidateGenerator`, relation candidates
  - `./src/orchestration/planner.rs` → `Planner`, `PlanConfig`, `SemanticScorer`, language helpers
  - `./src/orchestration/executor.rs` → `PlanExecutor`, `CapabilityRunner`, plan → `openjiuwen-graph`, deferral accounting
  - `./src/orchestration/mutation.rs` → `MutationRequest`, `request_digest`, `MutationJournal`, `validate_mutation_against_inventory`
  - `./src/orchestration/matcher/` → the request/response protocol adapter (`build_match_request`, `parse_match_response`, `GraphDiagnostic`); see [`src/orchestration/matcher/CONTEXT.md`](src/orchestration/matcher/CONTEXT.md)
- `./src/evaluation/` → fail-soft evaluation: shared contract, static and trace metrics, suite aggregation, windows; see [`src/evaluation/CONTEXT.md`](src/evaluation/CONTEXT.md)
- `./src/runtime/` → `SymphonyRuntime`, config, lifecycle, evolution, mutations; see [`src/runtime/CONTEXT.md`](src/runtime/CONTEXT.md)
  - `./src/runtime/mod.rs` → lifecycle, planning/invocation, lazy semantic sync, evolution
  - `./src/runtime/mutations.rs` → the mutation path, `GraphMutationResult`, `MutationStatus`, replay and source-revision CAS
- `./src/observation/` → the evidence contract, identity, append-only log, immutable revisions and `GraphObservationService`; see [`src/observation/CONTEXT.md`](src/observation/CONTEXT.md)
  - `./src/observation/contracts/` → `GraphEvolutionInput`, evidence/enums, `validate`, `normalized`, trace bridges
  - `./src/observation/identity.rs` → `stable_hash`, `EdgeIdentity`, `StaticGraphIndex`
  - `./src/observation/store/` → the JSONL `EvidenceStore` and the per-scope `RevisionStore`
  - `./src/observation/service/` → qualification, aggregation, rebinding, snapshot/fallback
  - `./src/observation/payload.rs`, `timestamp.rs` → sanitised payload builder and the RFC 3339 layer
- `./src/deferred.rs` → the remaining GPU-only `Error::Unsupported` stub (`vLLM` logit selection)
- `./tests/` → behavior suites: `models`, `normalization`, `registry`, `search`, `graph`, `artifact`, `object_artifact`, `artifact_gcs`, `artifact_azure`, `planner`, `executor`, `runtime`, `mutation_journal`, `discovery`, `discovery_toolkit`, `discovery_toolkit_notice`, `skill_prompt`, `retrieval_toolkit`, `experience`, `experience_index`, `experience_trace`, `retrieval_build`, `retrieval_artifacts`, `retrieval_search`, `retrieval_search_lexical`, `retrieval_search_disclosure`, `flow`, `flow_codegen`, `redaction`, `evaluation`, `llm`, `matcher_protocol`, `matcher_cache`, `observation`, `observation_service`, `observation_runtime`, `deferred`
