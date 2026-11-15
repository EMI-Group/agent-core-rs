# observation — Symphony graph observation & merged revisions

## Intent

Turns caller-supplied execution evidence into derived, immutable observation and merged graph revisions that planners can consume without ever rewriting the static `graph.json`.
Ports Python `openjiuwen.symphony.observation` (v0.1.20): the canonical `GraphEvolutionInput` contract, an append-only evidence log, content-hashed immutable revision manifests, and the `runtime_weight` / `binding_status` overlay layered on top at planning time.
The static capability graph stays authoritative; nothing here mutates it, and GraphEngine never parses Session JSON or guesses a failed edge.
Evidence is caller-supplied: this package never derives task outcomes, failure domains or edge success from traces — a rail or evaluator resolves them and passes a `GraphEvolutionInput`.
Everything is offline, deterministic and dependency-free: no database, no file lock, no chrono, no background thread.

## API Surface

All paths below are crate-relative.

### `contracts` (`src/observation/contracts/mod.rs`)

```rust
pub const GRAPH_EVOLUTION_INPUT_SCHEMA: &str = "symphony.graph_evolution_input.v2";
pub const TRUNCATED_TRACE, TRACE_QUALITY_FLAGS, OUTCOME_NOT_STRONG,
             CAPABILITY_IDENTITY_MISMATCH, UNKNOWN_STATIC_REVISION, UNKNOWN_MERGED_REVISION,
             MERGED_STATIC_REVISION_MISMATCH, MERGED_OBSERVATION_REVISION_MISMATCH,
             NO_QUALIFIED_EDGE_OUTCOME, SAFE_FALLBACK: &str;   // one constant each, snake_case text

pub enum TaskOutcomeLabel { VerifiedSuccess, VerifiedFailure, Partial, Cancelled, Unknown }
pub enum EvidenceStrength { Strong, Weak, None }
pub enum FailureDomain { Orchestration, SkillInput, SkillLogic, Permission, Network,
                         ExternalService, Timeout, Cancelled, Unknown }
pub enum ObservationStatus { Accepted, AuditOnly, Duplicate }
// every enum: Copy, Eq, Default, serde snake_case, #[must_use] const fn as_str(self) -> &'static str

pub struct GraphSnapshotRef { pub static_revision, pub observation_revision, pub merged_revision: Option<String> }
pub struct EvolutionGraphNode { pub label: String /* "skill" */, pub metadata: BTreeMap<String, Value> }
pub struct EvolutionEdgeMetadata { pub success: Option<bool>, pub failure_domain: Option<FailureDomain> }
pub struct EvolutionGraphEdge {
    #[serde(alias = "source")] pub source_id: String,
    #[serde(alias = "target")] pub target_id: String,
    #[serde(alias = "relation")] pub relation_type: String,   // "can_feed"
    pub metadata: EvolutionEdgeMetadata,
}
pub struct EvolutionGraph {
    pub id: Option<String>, #[serde(rename = "type")] pub graph_type: String,
    pub directed: bool, pub nodes: BTreeMap<String, EvolutionGraphNode>,
    pub edges: Vec<EvolutionGraphEdge>,
}
pub struct TraceEvidence { trajectory_id, session_id, capture_mode, request_id: Option<String>,
                           truncated, quality_flags, span_refs, member_trajectory_refs }
pub struct TaskOutcome { pub label, pub evidence_strength, pub failure_domain: Option<FailureDomain> }
pub struct TaskEvidence { pub query: Option<String>, pub task_cluster_id: Option<String>, pub outcome: TaskOutcome }
pub struct GraphEvolutionInput {
    pub schema_version, pub evidence_id, pub graph_scope_id, pub observed_at: String,
    pub graph_snapshot: GraphSnapshotRef, pub trace: TraceEvidence, pub task: TaskEvidence,
    pub planned_graph: Option<EvolutionGraph>, pub execution_graph: EvolutionGraph,
}
impl GraphEvolutionInput {
    pub fn validate(&self) -> Result<()>;    // Error::Validation naming the offending field
    pub fn normalized(self) -> Result<Self>; // validate + UTC `observed_at` + canonical 4-key ports
}
impl TraceEvidence { pub fn from_trace_record(&TraceRecord) -> Self;
                     pub fn from_trace_record_with_mode(&TraceRecord, impl Into<String>) -> Self; }
impl TaskOutcome { pub fn from_trace_record(&TraceRecord) -> Self; }
impl TaskEvidence { pub fn from_trace_record(&TraceRecord) -> Self; }

pub struct ObservationReceipt { evidence_id, graph_scope_id, sequence: u64, status: ObservationStatus, reason: String }
pub struct GraphSnapshot { graph_scope_id, static_revision, observation_revision, merged_revision,
                           high_water_sequence: u64, overlay: BTreeMap<String, Value> }
```

Structs derive `Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema` (plus `Default` where every field defaults) with `#[serde(rename_all = "snake_case", default)]`; unknown fields are tolerated.
`validate` enforces: schema version, non-blank contract text, timezone-carrying `observed_at`, `execution_graph.type == "execution_graph"`, `planned_graph.type == "planned_graph"`, every execution node label `skill`, node metadata allow-listed to `{capability_type, version, description, inputs, outputs}` with `capability_type == label` and non-blank `version`, ports shaped `{name, type, required: bool, description}`, every edge endpoint present, and `verified_failure` carrying a `failure_domain`.
Contract text is non-empty, already trimmed, and free of `Cc`/`Cf` characters (the `Cf` check is a documented hand-rolled range table approximating `unicodedata.category == "Cf"`).
`TraceEvidence::from_trace_record` uses capture mode `agent`; a failed record becomes `verified_failure`/`strong`/`FailureDomain::Unknown` because only a caller-supplied verdict can attribute a domain.

### `identity` (`src/observation/identity.rs`)

```rust
pub const POINT_EDGE_IDENTITY_SCHEMA: &str = "symphony.point-edge.v1";
pub fn stable_hash(&Value) -> String;          // SHA-256 hex over canonical JSON (sorted keys, no whitespace)
pub fn private_hash(&str) -> String;           // SHA-256 hex over UTF-8 bytes
pub fn normalize_capability_id(&str) -> String; // trim, strip one `skill:`, then one `capability:`
pub fn capability_key(capability_type: &str, capability_id: &str) -> String; // "{type.lower()}:{normalized id}"

pub struct EdgeIdentity { source_id, target_id, relation_type, source_content_hash, target_content_hash }
impl EdgeIdentity { new(..), identity_hash() -> String, to_value() -> Value, from_value(&Value) -> Option<Self> }

pub struct StaticGraphIndex {
    pub revision: String, pub capability_ids: BTreeSet<String>,
    pub graph_hash_by_id, pub content_hash_by_id: BTreeMap<String, String>,
    pub edges_by_identity: BTreeMap<String, Value>,
    pub edge_identities: BTreeMap<String, EdgeIdentity>,
}
impl StaticGraphIndex {
    pub fn from_graph(revision, &CapabilityGraph) -> Self;
    pub fn from_graph_with_descriptors(revision, &CapabilityGraph, &[CapabilityDescriptor]) -> Self;
    pub fn validates_nodes<'a>(&self, impl IntoIterator<Item = &'a str>) -> bool;
}
pub fn edge_identity_from_observation(&EvolutionGraphEdge, &StaticGraphIndex) -> Option<EdgeIdentity>;
```

`from_value` requires `identity_schema == POINT_EDGE_IDENTITY_SCHEMA`, so a record from another schema rebinds as `legacy_ignored`.
`from_graph` derives node ids from the graph, the capability type from `metadata["capability_type"]` → `node_type` → `"skill"`, the graph hash from the node identity JSON, and the content hash from `metadata["content_hash"]` when present, else that identity hash.
`from_graph_with_descriptors` lets `CapabilityDescriptor::compute_content_hash()` win, matched by normalized qualified id and by bare capability id.

### `store` (`src/observation/store/`)

```rust
pub const EVIDENCE_SCHEMA: &str = "symphony.graph-evidence.v2";
pub const OBSERVATION_SCHEMA: &str = "symphony.graph-observation.v2";
pub const MERGED_SCHEMA: &str = "symphony.graph-merged.v2";

pub struct EvidenceRow { sequence: u64, graph_scope_id, evidence_id, observed_at,
                         eligible: bool, reason, payload_digest, payload_json }
impl EvidenceRow { pub fn payload(&self) -> Result<Value>; }

pub struct EvidenceAppendInput<'a> { graph_scope_id, evidence_id, observed_at, eligible, reason, payload: &'a Value }
pub struct EvidenceAppendResult { pub sequence: u64, pub duplicate: bool, pub eligible: bool, pub reason: String }

pub struct EvidenceStore;   // new(root) / path() / append() / iter_after(scope, after) / scope_ids()
                            // rows() / len() / is_empty() / lookup(scope, id) / reload()

pub struct OverlayInput<'a> { graph_scope_id, static_revision, high_water_sequence: u64,
                              edges: &'a BTreeMap<String, Value>, generated_at,
                              projection_status, diagnostics: &'a [String] }
pub struct MergedInput<'a> { graph_scope_id, static_revision, observation_revision,
                             high_water_sequence: u64, generated_at }
pub struct CurrentPointer { schema_version, graph_scope_id, static_revision, observation_revision,
                            merged_revision, high_water_sequence: u64, generated_at }
impl CurrentPointer { pub fn from_snapshot(&GraphSnapshot, impl Into<String>) -> Self; }

pub struct RevisionStore;   // new(root) / scope_root(scope) / publish(&OverlayInput) -> GraphSnapshot
                            // publish_overlay(&OverlayInput) -> String / publish_merged(&MergedInput) -> String
                            // write_current(&CurrentPointer) / current_pointer(scope) / current(scope)
                            // read_overlay(scope, observation_revision) / current_overlay(scope)
                            // read_snapshot(scope, Option<&str>) / scopes()
```

The evidence log is append-only, 1-based and monotonic, deduplicated on `(graph_scope_id, evidence_id)` plus payload digest; a reused id with a different digest is `Error::Validation` naming `evidence_id`.
A blank log line is skipped, a malformed one is `tracing::warn!`-ed and skipped, and a missing root reads as an empty log without creating anything.
`read_snapshot(scope, Some(unknown))` is `Ok(None)`; turning that into `NotFound` is the service's decision.

### `service` (`src/observation/service/mod.rs`)

```rust
pub const MIN_RUNTIME_ONLY_SUCCESS_SESSIONS: usize = 2;
pub const RUNTIME_ADJUSTMENT_STEP: f64 = 0.05;
pub const MIN_RUNTIME_ADJUSTMENT: f64 = -1.0;
pub const MAX_RUNTIME_ADJUSTMENT: f64 = 1.0;

pub struct GraphObservationService;
impl GraphObservationService {
    pub fn new(artifacts: Arc<dyn GraphArtifactStore>, root: impl AsRef<Path>) -> Self;
    pub async fn submit(&self, &GraphEvolutionInput) -> Result<ObservationReceipt>;
    pub async fn get_snapshot(&self, graph_scope_id: &str, merged_revision: Option<&str>) -> Result<GraphSnapshot>;
    pub async fn flush(&self) -> Result<()>;
    pub async fn reproject(&self, graph_scope_id: &str) -> Result<GraphSnapshot>;
    pub async fn reproject_all(&self) -> Result<Vec<GraphSnapshot>>;
    pub async fn reproject_all_safely(&self) -> Result<Vec<GraphSnapshot>>;
    pub async fn publish_safe_fallback(&self, graph_scope_id: &str, reason: &str) -> Result<GraphSnapshot>;
    pub async fn close(&self) -> Result<()>;   // documented no-op: nothing is buffered
}
```

`submit` normalises and validates the input, qualifies it, appends one row, and — unless the append was a duplicate — reprojects the scope before returning the receipt.
Qualification produces `audit_only` with these reasons, in this order: `truncated_trace`, `trace_quality_flags`, `outcome_not_strong`, `capability_identity_mismatch`, the pinned-snapshot reason, `no_qualified_edge_outcome`; an unknown static revision is prefixed with `unknown_static_revision` and is qualified against the current index.
A failed reprojection does not lose the row: it is `tracing::warn!`-ed, stored, returned by the next `flush()`, and the receipt is still `accepted`.
`get_snapshot` reprojects when no snapshot exists, when its static revision moved, or when its overlay is a `safe_fallback`; a failed reprojection publishes a neutral fallback instead of failing the read, while an explicitly pinned unknown `merged_revision` is `Error::NotFound` naming `merged_revision`.
A missing static revision is `Error::NotFound` naming `static_revision` for `submit`, `get_snapshot`, `reproject` and `publish_safe_fallback`.

### `payload` / `timestamp` (crate-internal)

`payload::evidence_payload(&GraphEvolutionInput, &[QualifiedEdge]) -> Value` renders the persisted payload; `EdgeOutcome` is `success`/`failure`, `QualifiedEdge` carries the identity, outcome and `observed_as_static`.
`timestamp::{parse_epoch, format_epoch, normalize_observed_at, utc_now}` is the hand-rolled RFC 3339 layer (Hinnant civil-date arithmetic, integer only, no chrono).

## Overlay edge record (cross-module contract)

Each overlay entry is keyed by `edge_identity` and carries: `identity_schema`, `edge_identity`, `source_id`, `target_id`, `relation_type`, `source_content_hash`, `target_content_hash`, `success_count`, `failure_count`, `attempt_count`, `success_session_hashes`, `success_sessions_by_revision`, `failure_session_hashes`, `task_cluster_ids`, `ever_static`, `first_observed_at`, `last_observed_at`, `last_evidence_static_revision`, `runtime_adjustment`, `runtime_weight`, `runtime_only`, `binding_status`.
`binding_status` precedence: `legacy_ignored` (foreign identity schema) → `orphaned` (endpoint gone) → `invalidated` (endpoint content hash moved) → `active_static` (the edge is in the static index) → `quarantined` (`ever_static` but no evidence on this static revision) → `active_runtime_only` (≥ 2 distinct success sessions on this static revision, zero failures) → `insufficient_evidence`.
`runtime_adjustment = clamp(0.05 * (success_count - failure_count), -1.0, 1.0)` and `runtime_weight = 1.0 + runtime_adjustment`, so weights span `0.0..=2.0`.
`success_count`/`failure_count` are session-deduped; `attempt_count` is the size of the union of the two session-hash lists; `success_sessions_by_revision[revision]` is what the runtime-only threshold counts.
`binding_status`, `runtime_weight`, `edge_identity`, `attempt_count`, `success_count` and `runtime_only` are read by planners and must never be renamed.
Only `verified_success` on an edge with `metadata.success == true`, and `verified_failure` on an edge with `success == false` where both the edge and the task name `FailureDomain::Orchestration`, count; `partial`, `cancelled` and `unknown` count nothing.

## On-disk layout

```text
<root>/evolution/evidence.jsonl                        # one JSON row per line, append-only
<root>/evolution/scopes/<first 20 hex of sha256({"graph_scope_id": scope})>/
  current.json                                         # CurrentPointer to the active revisions
  observations/observation-<sha256(canonical overlay)[:20]>/overlay.json
  merged/merged-<sha256(schema, scope, static, observation)[:20]>/manifest.json
```

An evidence row holds `sequence`, `graph_scope_id`, `evidence_id`, `observed_at`, `eligible`, `reason`, `payload_digest` and `payload_json` (the canonical JSON *text* of the payload, mirroring Python's column).
The overlay document holds `schema_version`, `graph_scope_id`, `static_revision`, `high_water_sequence`, `edges`, `projection_status`, `diagnostics`, plus its `observation_revision` and `generated_at`.
The merged manifest holds `schema_version`, `graph_scope_id`, `static_revision`, `observation_revision`, plus `merged_revision`, `high_water_sequence` and `generated_at`.
`generated_at` never enters a hash (the merged hash also excludes `high_water_sequence`), so re-aggregating identical state reproduces the same ids and a retried publish is an idempotent no-op.
A revision document whose stored content (ignoring `generated_at`) differs from the requested content is `Error::Validation` naming the path; every write is a `<name>.tmp` sibling plus `tokio::fs::rename`.

## Constraints

- Only files under `crates/openjiuwen-symphony/src/observation/` belong to this node; `src/lib.rs` already declares `pub mod observation;`, and the crate-level `CONTEXT.md` / routing table is maintained one level up.
- Depend downward only: `crate::artifact`, `crate::graph`, `crate::model`, `crate::experience::trace`, `crate::redaction`, `crate::llm::canonical_json`, `crate::error`, plus `openjiuwen_common`. No new dependencies: no chrono, no sqlx, no regex, no file-lock crate.
- `#![forbid(unsafe_code)]`; no `unsafe`, no `println!`/`eprintln!` (use `tracing`), no `unwrap`/`expect` outside `#[cfg(test)]`.
- Every public item and field carries a doc comment (`missing_docs` under `-D warnings`); module-level `//!` links to same-file items use absolute `crate::observation::…` paths.
- `std::sync` guards are never held across an `.await`; all mutual exclusion is `tokio::sync` (`Mutex` for append/aggregation, `RwLock` for the static-index cache).
- Files stay under ~1000 lines: `contracts/mod.rs` 892, `service/mod.rs` 951, `store/revision.rs` 853, `identity.rs` 867, with the test bodies in sibling `tests.rs` files and the shared store helpers in `store/files.rs`.
- All public data types are `serde`-serializable, `snake_case`, tolerate unknown fields, and default empty collections.
- Determinism: no hash-map iteration order in outputs (every map is a `BTreeMap`), no wall-clock ordering, and timestamps are metadata only.
- Errors use `crate::error::{Error, Result}` only; validation messages name the offending field, and `Error::NotFound` is never raised where an `Option` is the honest answer (registry-style reads return `Option`, `read_snapshot(Some(unknown))` returns `Ok(None)`).

## Design Decisions

- **The evidence log is append-only JSONL, not SQLite.** The crate ships no database; sequences are derived from the last row on load, the parsed rows live in memory, and appends rewrite nothing, which preserves Python's append/dedupe/cursor semantics without `sqlx`.
- **Appends and aggregation are serialized by one `tokio::sync::Mutex` each**, replacing Python's `.evidence.lock`, `.revision.lock` and `.projection.lock`; one store instance is the writer of one artifact root, and cross-process sharing is deliberately out of scope.
- **Aggregation runs inside the caller's task**, replacing the single-worker `ThreadPoolExecutor`: `submit` appends and reprojects before returning, a failed reprojection is remembered for `flush()` instead of being raised from the submit, and `close()` has nothing to drain.
- **RFC 3339 is hand rolled** (`timestamp.rs`): a tolerant parser plus Hinnant civil-date arithmetic, keeping the fractional digits exactly as supplied and returning `...Z` UTC text — no chrono dependency.
- **Static content hashes come from the Rust artifact, not Python's hash maps.** Python's artifact carries `capability_hashes` / `graph_identity_hashes`; here the index reads `metadata["content_hash"]` when a publisher supplied it and otherwise hashes the node identity JSON, with `from_graph_with_descriptors` letting a descriptor's `compute_content_hash()` win.
- **Point-edge identity** (endpoints + relation + endpoint content hashes) replaces a port mapping, so a changed capability body `invalidated`s prior evidence and a deleted endpoint `orphaned`s it.
- **Persisted payloads carry only hashes and redacted text**: trajectory/session/request ids become SHA-256 hex, and the task query is kept only in redacted form (`redact_sensitive_text`) — a Rust-only addition, since Python drops the query and persists no free text at all.
- **Content-derived revision ids, immutable documents**: `observation-<hash>` and `merged-<hash>` are stable across retries and `generated_at`, so a planner can pin the exact revision it planned against and a retried publish is a no-op.
- **A `safe_fallback` overlay is a retryable neutral snapshot**: empty edges, `projection_status = "safe_fallback"`, a diagnostic reason, and `high_water_sequence = 0` so the next successful reprojection replays the whole scope log instead of silently skipping rows.
- **`high_water_sequence` advances past ineligible rows**, matching Python: an `audit_only` row is never re-qualified later, which keeps the cursor monotonic and the replay bounded.
- **Runtime-only edges need two failure-free success sessions in the current static revision** before they become plannable, so a single lucky observation cannot mint a route.
- **`GraphEvolutionInput` tolerates unknown fields** where pydantic forbids them; the known fields are validated strictly by `validate()`, and `submit` always calls `normalized()` first so nothing invalid reaches the log.
- **`NotFound` is a service-level decision.** The store answers `Ok(None)` for a missing or unknown revision; the service turns a missing static graph or an explicitly pinned unknown `merged_revision` into `Error::NotFound` naming the field, and never falls back silently for an explicit pin.
- **The store is split by responsibility** (`evidence`, `revision`, shared atomic-write `files`) with the public surface re-exported from `store/mod.rs`, so the module paths never moved.

## Known Issues

- `high_water_sequence` advancing past ineligible rows means a rejected row is never retried, even if the static graph or the qualification rules would later accept it (Python parity).
- `MIN_RUNTIME_ONLY_SUCCESS_SESSIONS` counts successes *per static revision*, so publishing a new static revision resets a runtime-only edge to `insufficient_evidence` until evidence accrues again.
- Every new observation revision mints a new merged revision (the merged hash includes the observation revision), so `merged/` accumulates directories even when the planner-visible content is equivalent for a caller that pins only the static revision.
- `submit` can race a concurrent static publication: the evidence is qualified against the index it read, and only the next reprojection corrects the binding (`unknown_static_revision`, or a rebind).
- There is no cross-process locking: two processes writing the same artifact root can interleave; the crate documents one writer per root instead of a `FileLock`.

## Notes for Agents

- The overlay keys are an untyped cross-module contract; changing a key in `service/mod.rs` silently breaks planners. Add or rename keys only with a crate-level review.
- `POINT_EDGE_IDENTITY_SCHEMA` is read by the overlay rebinding and embedded in every persisted edge; it is not free to rename either.
- `payload.rs` and `timestamp.rs` are `pub(crate)`: they are implementation detail, and `crate::observation::payload::*` wire-key constants are the single source for payload/overlay field names inside this package.
- `contracts/tests.rs` and `service/tests.rs` are `#[cfg(test)] mod tests;` children of their `mod.rs`, so they reach private items through `use super::*;` and keep the implementation file under the soft cap.
- Integration coverage for this package lives in the parent-owned `crates/openjiuwen-symphony/tests/observation*.rs` suites; do not add cross-module scenarios here, and do not edit those tests from this node.

## Test Strategy

- 80 unit tests live beside the code: 29 in `contracts/` (18 contract + 11 RFC 3339), 16 in `identity.rs`, 19 in `store/` (9 evidence + 10 revision), 16 in `service/`+`payload.rs` (13 service + 3 payload).
- `contracts/tests.rs` covers serde aliases/defaults and unknown-field tolerance, every validation rule, `normalized()` UTC conversion and port canonicalisation, and the trace bridges for success and failure.
- `timestamp.rs` covers leap day and `2023-02-29` rejection, an offset crossing midnight backwards, `+0830`/`-0830`, lowercase `z`, a six-digit fraction preserved verbatim, a missing timezone, trailing garbage, and a pre-epoch instant.
- `identity.rs` covers key-order-insensitive hashing, id normalisation and prefix stripping, identity round-trips and schema rejection, index construction from a `CapabilityGraph`, descriptor-override content hashes, and `validates_nodes`.
- `store/` covers sequence numbering, reopen continuity, duplicate replay, id reuse with different content, scope isolation and cursor boundaries, malformed-line tolerance, content-derived ids that ignore `generated_at`, idempotent republish, immutable-overwrite rejection, pinned-manifest reads and traversal-safe revision names.
- `service/` covers acceptance and `active_static` weights, duplicate receipts, id reuse rejection, audit-only gating and cursor advancement, unknown static revisions, orchestration vs network failure attribution, the runtime-only threshold, quarantine after a static republication, pinned merged revisions vs `NotFound`, the `safe_fallback` replay, and multi-scope reprojection.
- Tests are offline and deterministic: `tempfile::TempDir` roots, `MemoryGraphArtifactStore`, no clock-dependent assertions (`generated_at` is never compared), no live network.
- Gates: `cargo fmt -p openjiuwen-symphony`, `cargo test -p openjiuwen-symphony --offline --all-features`, `cargo clippy -p openjiuwen-symphony --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-symphony --no-deps --offline`.

## Routing Table

- `./mod.rs` → module declarations for the package
- `./contracts/` → the public evidence contract; see `./contracts/mod.rs`
  - `./contracts/mod.rs` → constants, enums, graph/evidence models, `validate`, `normalized`, trace bridges
  - `./contracts/tests.rs` → contract validation and serialization unit tests
- `./identity.rs` → `stable_hash`, `private_hash`, capability-id normalisation, `EdgeIdentity`, `StaticGraphIndex`, `edge_identity_from_observation`
- `./store/` → `EvidenceStore` and `RevisionStore`; see `./store/mod.rs`
  - `./store/evidence.rs` → append-only JSONL log, sequences, dedupe, in-memory index
  - `./store/revision.rs` → overlay/manifest/pointer documents, content-derived ids, snapshot reads
  - `./store/files.rs` → shared atomic write, immutability and revision-name helpers
- `./service/` → `GraphObservationService`; see `./service/mod.rs`
  - `./service/mod.rs` → qualification, aggregation, rebinding, snapshot/fallback orchestration
  - `./service/tests.rs` → service behaviour unit tests
- `./payload.rs` → the sanitised persisted payload builder and its wire-key constants (crate-internal)
- `./timestamp.rs` → hand-rolled RFC 3339 parsing, UTC formatting and `utc_now` (crate-internal)
