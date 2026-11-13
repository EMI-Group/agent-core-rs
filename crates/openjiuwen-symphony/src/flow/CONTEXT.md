# openjiuwen-symphony / flow

## Intent

Flow distillation and package production, mapping to Python `openjiuwen.symphony.flow`.
Observed execution traces are ingested, grouped by their successful-edge structure, and distilled into versioned `ExperienceRecipe`s; a frozen recipe can then be packed into a reviewed `SkillPack` artifact.
Distillation is structural, not narrative: evidence groups on its cycle-broken successful-edge signature, versions are content-addressed and immutable, and a model can only write prose.
The flow store writes eagerly and atomically, so there is no flush on close and a reopened engine reproduces its evidence, recipe versions, and packages.
Everything here is offline and deterministic; an optional `CompletionModel` supplies only names, applicability, and narrative text, with a template fallback.

## API Surface

### Trace normalization

`FlowTrace::from_value` is intentionally forgiving but deterministic: unknown fields are ignored, leading `skill:`/`capability:` prefixes are stripped, nodes without an id or object body are dropped, and an edge survives only when its `metadata.success` is truthy and both endpoints exist after normalization; surviving edges are sorted by `(source, target, relation)`, so a stored trace is byte-stable.

```rust
pub enum TraceOutcome { Success, Failed, Partial, Unknown }   // aliases: succeeded/ok/done, failure/error, partially
pub struct TraceNode { pub capability_id: String, pub capability_type: String, pub skill_type: String, pub version: String, pub description: String, pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo> }
pub struct TraceEdge { pub source: String, pub target: String, pub relation: String /*can_feed*/, pub success: bool, pub reason: String }
pub struct ExecutionGraph { pub id: String, #[serde(rename = "type")] pub graph_type: String, pub label: String, pub directed: bool, pub nodes: BTreeMap<String, TraceNode>, pub edges: Vec<TraceEdge> }
pub struct FlowTrace { pub trace_id: String, pub query: String, pub outcome: TraceOutcome, pub graph: ExecutionGraph }
impl FlowTrace {
    pub fn from_value(value: Value) -> Result<Self>;
    pub fn successful_edges(&self) -> Vec<(String, String, String)>;
}
```

### Recipes

```rust
pub const RECIPE_SCHEMA_VERSION: &str = "symphony.experience_recipe.v1";
pub const RECIPE_STATUS_DRAFT: &str = "draft";
pub const RECIPE_STATUS_ACTIVE: &str = "active";
pub const RECIPE_GRADE_CANDIDATE: &str = "candidate";
pub const RECIPE_GRADE_VERIFIED: &str = "verified";
pub const MIN_SUCCESSES: u64 = 1;
pub const MIN_PACK_SUCCESS_RATE: f64 = 0.8;

pub struct RecipeQuality { pub execution_count: u64, pub success_count: u64, pub failure_count: u64, pub pack_success_rate: f64, pub qualified_edge_count: usize }
pub struct RecipeEdge { pub source: String, pub target: String, pub relation: String }
pub struct CombinationStructure { #[serde(rename = "type")] pub structure_type: String /*skill_pack*/, pub direction: String /*directed*/, pub nodes: BTreeMap<String, Value>, pub edges: Vec<RecipeEdge>, pub loop_guards: Vec<Value> }
pub struct ExperienceRecipe {
    pub schema_version: String, pub recipe_id: String, pub name: String,
    pub version: u64 /*starts at 1*/, pub status: String /*draft|active*/, pub grade: String /*candidate|verified*/,
    pub applicability: BTreeMap<String, Value>, pub combination_structure: CombinationStructure,
    pub execution_narrative: String, pub quality: RecipeQuality, pub provenance: BTreeMap<String, Value>,
}
impl ExperienceRecipe { pub fn capability_ids(&self) -> Vec<String>; pub fn applicability_text(&self, key: &str) -> &str; pub fn content_identity(&self) -> String; pub fn is_active_verified(&self) -> bool; pub fn candidate(&self) -> CombinationCandidate; }
pub struct CombinationCandidate;    // display projection of an active, verified recipe
pub fn topological_order(structure: &CombinationStructure) -> Vec<String>;
pub fn sha256_hex(text: &str) -> String; pub fn sha256_short(text: &str, length: usize) -> String;
pub fn canonical_json(value: &Value) -> String; pub fn content_hash(value: &Value) -> String;
pub fn normalize_capability_id(raw: &str) -> String; pub fn normalize_meta_name(raw: &str, fallback: &str) -> String; pub fn is_valid_meta_name(name: &str) -> bool;
```

Distillation groups successful traces by the cycle-broken, sorted successful-edge signature `source>target:relation` joined with `|`; the recipe id is `recipe_<sha256(signature)[:12]>`.

The first distillation of a group writes version `1`, and a changed content identity (name, applicability, structure, or narrative) writes `version + 1`; replaying unchanged evidence writes nothing and appears in `recipes_unchanged`.

Quality counts are captured when a version is written: `pack_success_rate` is `success_count / execution_count` rounded to four decimals.

A recipe is `active` + `verified` when `success_count >= MIN_SUCCESSES` (1) and `pack_success_rate >= MIN_PACK_SUCCESS_RATE` (0.8); otherwise it stays `draft` + `candidate`.

### Engine

```rust
pub struct FlowEngine;
impl FlowEngine {
    pub fn new(root: impl Into<PathBuf>) -> Self;            // no disk reads or writes
    pub async fn open(root: impl AsRef<Path>) -> Result<Self>;
    pub fn with_model(self, model: Arc<dyn CompletionModel>) -> Self;
    pub fn root(&self) -> &Path;
    pub async fn ingest(&self, trace: FlowTrace) -> Result<bool>;
    pub async fn ingest_value(&self, value: Value) -> Result<bool>;
    pub async fn evidence(&self) -> Vec<FlowTrace>;
    pub async fn distill(&self) -> Result<DistillReport>;
    pub async fn recipes(&self) -> Vec<ExperienceRecipe>;
    pub async fn recipe(&self, recipe_id: &str) -> Option<ExperienceRecipe>;
    pub async fn candidates(&self) -> Vec<CombinationCandidate>;
    pub async fn pack(&self, recipe_id: &str) -> Result<SkillPack>;
    pub async fn package(&self, package_id: &str) -> Option<SkillPack>;
    pub async fn review_and_prepare_install(&self, recipe_id: &str, target_kind: &str, dest_dir: impl AsRef<Path>) -> Result<InstallPreparation>;
    pub async fn prepare_install(&self, recipe_id: &str, recipe_version: u64, target_kind: &str, dest_dir: impl AsRef<Path>) -> Result<InstallPreparation>;
    pub async fn close(&self) -> Result<()>;                 // every write is already durable
}
pub struct DistillReport { pub evidence_total: usize, pub groups_total: usize, pub groups_without_structure: Vec<String>, pub recipes_saved: Vec<String>, pub recipes_unchanged: Vec<String>, pub narrative_fallbacks: Vec<String> }
```

Lifecycle: `new` is in-memory only, `open` rehydrates from disk, and `close` is a no-op because every ingestion, recipe version, and package is written eagerly and atomically.

`ingest` accepts only successful traces with at least one successful edge and is idempotent per `trace_id` (a replay answers `false`); a blank id, a non-success outcome, or an edgeless graph is `Error::Validation`.

A configured `CompletionModel` only supplies name, applicability, and narrative prose; every structural decision is deterministic, and a failed or unusable reply falls back to the template and is reported in `narrative_fallbacks`.

### Flow store

The flow store writes eagerly and atomically: `evidence.jsonl` (one trace per line, ordered by `trace_id`), `recipes/<recipe_id>/v<N>.json`, `recipes/<recipe_id>/current.json`, and `packages/<package_id>.json`, each through a unique `.<name>.<uuid>.tmp` sibling plus a rename; malformed files are warned and skipped on load.

### SkillPack

```rust
pub const TARGET_KIND_SKILL: &str = "skill";
pub const DEFAULT_LICENSE: &str = "Proprietary";
pub const SKILL_PACK_KIND: &str = "skillpack";
pub const SKILL_MD_FILENAME: &str = "SKILL.md";
pub const DEPENDENCIES_FILENAME: &str = "dependencies.yaml";
pub const PACKAGE_SCHEMA_VERSION: &str = "symphony.capability_package.v1";

pub struct SkillPackStep { pub id: String, pub name: String, pub description: String, pub capability_type: String, pub inputs: Vec<CapabilityIo>, pub outputs: Vec<CapabilityIo>, pub depends_on: Vec<String> }
pub struct SkillPack {
    pub schema_version: String, pub package_id: String /*cap-<sha256(materials)[:12]*/,
    pub recipe_id: String, pub recipe_version: u64, pub target_kind: String /*skill*/, pub name: String,
    pub description: String /*[技能包] ...*/, pub steps: Vec<SkillPackStep>,
    pub artifacts: BTreeMap<String, String>, pub permissions: Vec<String>, pub license: String,
}
impl SkillPack {
    pub fn materials(&self) -> Value;  pub fn materials_hash(&self) -> String;  pub fn compute_package_id(&self) -> String;
    pub fn recompute_identity(&mut self); pub fn step_ids(&self) -> Vec<&str>; pub fn task_description(&self) -> &str;
    pub fn render(&self) -> BTreeMap<String, String>;
    pub async fn save(&self, dir: impl AsRef<Path>) -> Result<()>;
    pub async fn load(dir: impl AsRef<Path>) -> Result<Self>;
    pub fn to_json(&self) -> Result<String>; pub fn from_value(value: Value) -> Result<Self>;
}
```

`pack` freezes only an `active`, `verified` recipe whose structure is a simple linear chain (otherwise `Error::Validation`), and an unknown recipe is `Error::NotFound`.

`SkillPack` materials are `recipe_id`, `target_kind`, `name`, `description`, and the ordered steps, and the content-addressed id is `cap-<sha256(canonical materials)[:12]>`; artifacts, permissions, and license are excluded.

The rendered `SKILL.md` has YAML front matter (`name`, `kind: skillpack`, `description`, `skills` in step order) and the body sections `# name`, `When to use`, `Required inputs`, `Included Skills`, `Execution Process`, `Failure handling`, `Final output`, and a fenced `Workflow Graph` block.

The workflow-graph block is `{"graph":{"id":"<name>-workflow","type":"skillpack_workflow","label":<description>,"directed":true,"nodes":...,"edges":...}}`; `dependencies.yaml` carries the ordered capabilities plus a package header, and `load` validates the front-matter skill order and the graph block against it.

### Review and install

```rust
pub const REVIEW_SCHEMA_VERSION: &str = "symphony.package_review.v1";
pub const VERDICT_APPROVED: &str = "approved";
pub const VERDICT_REJECTED: &str = "rejected";
pub const VERDICT_NEEDS_HUMAN_REVIEW: &str = "needs_human_review";
pub const CHECK_PASS: &str = "pass";
pub const CHECK_FAIL: &str = "fail";

pub struct Reviewer;
impl Reviewer { pub fn new() -> Self; pub fn review(&self, pack: &SkillPack) -> ReviewResult; }
pub struct ReviewCheck { pub check: String, pub result: String /*pass|fail*/, pub reasons: Vec<String> }
pub struct ReviewResult { pub review_id: String, pub package_id: String, pub target_kind: String, pub verdict: String, pub checks: Vec<ReviewCheck>, pub materials_hash: String, pub reviewed_at: String, pub schema_version: String }
impl ReviewResult { pub fn failed_checks(&self) -> Vec<&ReviewCheck>; pub fn failure_reasons(&self) -> Vec<String>; }
pub struct InstallPreparation { pub recipe_id: String, pub target_kind: String, pub verdict: String, pub package: Option<SkillPack>, pub artifact_dir: Option<PathBuf>, pub review: Option<ReviewResult>, pub reasons: Vec<String> }
```

The `Reviewer` runs seven checks in order — `structure`, `dependencies`, `integrity`, `meta_naming`, `materials`, `permissions`, `license` — and returns `approved` when all pass, `needs_human_review` when `meta_naming` is the only failure, and `rejected` otherwise.

A pack may not request `credential_access`, `filesystem_write`, `process`, or `shell`, and may only declare `Apache-2.0`, `BSD-3-Clause`, `MIT`, or `Proprietary`; `review_id` is `review_<12 hex>` over the identity fields, and `reviewed_at` is metadata only.

`review_and_prepare_install`/`prepare_install` return structural rejections (unsupported target kind, unknown recipe, stale requested version, failed review) as `InstallPreparation::verdict` with reasons, not errors, and only I/O failures are `Err`.

## Constraints

- Offline and deterministic: no network; `reviewed_at` and `distilled_at` are metadata and never decide ordering or equality.
- Every write is atomic (`.<name>.<uuid>.tmp` plus rename) and eager; `close` has nothing to flush and never fails.
- Recipe versions are immutable; a content change appends `version + 1`, an unchanged replay writes nothing.
- Only an `active`, `verified` linear chain can be packed; every other shape is a typed validation error.
- Every public item is documented; no `unsafe`, no `println!`, no panics on user input.
- Reuse `CapabilityIo` and `openjiuwen-retrieval`'s `CompletionModel`; do not duplicate shared types.
- Files stay around or under 1000 lines: `pack.rs` and `engine.rs` are the largest and are split by responsibility.

## Routing Table

- `./mod.rs` → module docs and re-exports of the flow surface
- `./model/` → data models and normalization
  - `./model/mod.rs` → `ExperienceRecipe`, `CombinationStructure`, `RecipeQuality`, canonical JSON/labels, schema constants, thresholds
  - `./model/trace.rs` → `FlowTrace`, `ExecutionGraph`, `TraceNode`/`TraceEdge`, `TraceOutcome`, `from_value` normalization
- `./store.rs` → atomic flow-root persistence (`evidence.jsonl`, recipe versions, packages)
- `./engine.rs` → `FlowEngine`, `DistillReport`, grouping, versioning, quality/status, optional narrative
- `./pack.rs` → `SkillPack`, content-addressed identity, `SKILL.md`/`dependencies.yaml` render, save/load
- `./review.rs` → `Reviewer`, the seven checks, verdicts, `InstallPreparation`
