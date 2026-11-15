# openjiuwen-symphony / flow

## Intent

Flow distillation and package production, mapping to Python `openjiuwen.symphony.flow`.
Observed execution traces are ingested, grouped by their successful-edge structure, and distilled into versioned `ExperienceRecipe`s; a frozen recipe can then be packed into a reviewed `SkillPack` artifact.
A packed recipe also carries the generated SwarmFlow orchestrator script (`swarmflow/run.py`), and a reviewed pack renders as one SDD-0010 SkillPack `SKILL.md` document.
Distillation is structural, not narrative: evidence groups on its cycle-broken successful-edge signature, versions are content-addressed and immutable, and a model can only write prose.
The flow store writes eagerly and atomically, so there is no flush on close and a reopened engine reproduces its evidence, recipe versions, and packages.
Everything here is offline and deterministic; an optional `CompletionModel` supplies names, applicability, and narrative text (with a template fallback) and, in the opt-in review stage, additional package findings (with a fail-closed merge).
The deterministic ten-check gate is always model-free; the model stage runs after it, only over packages it approved.

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
    pub artifacts: BTreeMap<String, String>,
    pub swarmflow_script: Option<String> /*rendered as swarmflow/run.py*/,
    pub trigger_conditions: String /*recipe applicability, empty when undeclared*/,
    pub permissions: Vec<String>, pub license: String,
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

`SkillPack` materials are `recipe_id`, `target_kind`, `name`, `description`, and the ordered steps, plus `swarmflow_script` and `trigger_conditions` when they are present; the content-addressed id is `cap-<sha256(canonical materials)[:12]>`; artifacts, permissions, and license are excluded.
A pack built from a recipe therefore carries a different id than an identical pack without a generated script; hand-built packs without a script keep the previous hash because the optional keys are omitted entirely.

`render` returns `SKILL.md`, `dependencies.yaml`, and (when the recipe yielded one) `swarmflow/run.py`; `save` writes nested artifact names by creating their parent directories and rejects anything that escapes the destination, and `load` restores the script from `swarmflow/run.py` and the trigger conditions from the `dependencies.yaml` package header.

The rendered `SKILL.md` has YAML front matter (`name`, `kind: skillpack`, `description`, `skills` in step order) and the body sections `# name`, `When to use`, `Required inputs`, `Included Skills`, `Execution Process`, `Failure handling`, `Final output`, and a fenced `Workflow Graph` block.

The workflow-graph block is `{"graph":{"id":"<name>-workflow","type":"skillpack_workflow","label":<description>,"directed":true,"nodes":...,"edges":...}}`; `dependencies.yaml` carries the ordered capabilities plus a package header, and `load` validates the front-matter skill order and the graph block against it.

### SwarmFlow script

```rust
pub const SWARMFLOW_MODULE: &str = "swarmflow";
pub const ALLOWED_OPERATOR: &str = "agent";
pub const SCRIPT_FILENAME: &str = "swarmflow/run.py";
pub const SCRIPT_MATERIAL_KEY: &str = "swarmflow_script";
pub const TRIGGER_CONDITIONS_KEY: &str = "trigger_conditions";
pub const DEFAULT_STEP_DESCRIPTION: &str = "协作成员能力";
pub const META_NAME_FALLBACK: &str = "skill-pack";
pub const IDENTIFIER_PREFIX: &str = "combo-";
pub const IDENTIFIER_HASH_LENGTH: usize = 12;
pub const RECIPE_ID_PREFIX: &str = "recipe_";

pub struct SwarmflowScript { pub identifier: String, pub source: String, pub meta: Value, pub task: String, pub steps: Vec<String> }
impl SwarmflowScript { /* Debug, Clone, PartialEq, Eq, Serialize, Deserialize */ }

pub fn swarmflow_identifier(materials: &Value, recipe_id: &str) -> String;   // combo-<12 hex over canonical materials>
pub fn generate_swarmflow_script(recipe: &ExperienceRecipe) -> Result<SwarmflowScript>;
pub fn validate_generated_script(source: &str) -> Result<()>;
pub fn python_string_literal(value: &str) -> String;                          // repr-like single quotes, UTF-8 preserved
pub use crate::flow::model::normalize_meta_name;
// crate-internal packaging plumbing (keeps `pack.rs` to its call sites)
pub(crate) fn extend_materials(materials: &mut Value, script: Option<&str>, trigger_conditions: &str);
pub(crate) fn insert_script_artifact(artifacts: &mut BTreeMap<String, String>, script: Option<&str>);
pub(crate) async fn read_script_artifact(directory: &Path) -> Result<Option<String>>;
```

The generated script is exactly Python's shape: a module docstring `"""SwarmFlow script generated from recipe <id>."""`, `from swarmflow import agent`, `META = {'name': <identifier>, 'phases': ['step_1', …]}`, `TASK = <task description>`, `async def run(args):`, one `step_N = await agent(<prompt>, label=<id>, phase='step_N')` per capability in topological order, and `return step_<last>`.
Steps follow `topological_order` (sorted Kahn), predecessors are sorted and joined with ` + `, and an empty combination emits `return None`; `TASK` is always emitted (Python omits it for an empty description) so the validator can require the marker.
Each prompt is `使用能力 <id>（<description>）完成本阶段任务` plus `。输入来自任务参数` or `。上一步产出：<ids>。`, and a node without a description uses `协作成员能力`.
Interpolation goes through `python_string_literal`: `\\`, `'`, `\n`, `\r`, `\t`, and other control characters are escaped, multi-byte CJK is emitted verbatim, and the module docstring escapes `\` and `"`.
`validate_generated_script` is a line-oriented approximation of Python's AST whitelist: it masks strings and comments (so their contents cannot spoof a token), requires balanced brackets and quotes, requires the `META`, `TASK`, and `async def run(args):` markers, allows only `from swarmflow import agent`, allows only `agent(...)` calls (excluding `def`/keyword heads and attribute calls), and rejects the review gate's forbidden tokens; every failure names the first offending line.
A syntax error inside an expression (for example a missing colon) is not detected — the approximation is documented on the function.

### Rendering

```rust
pub const SKILL_PACK_NAME_FALLBACK: &str = "symphony-combination";
pub const SKILL_PACK_TASK_FALLBACK: &str = "由 Symphony 成功执行生成的组合 Skill";
pub const SKILL_PACK_DESCRIPTION_PREFIX: &str = "[技能包]";
pub const INCLUDED_SKILL_FALLBACK: &str = "具体用途见该成员的 SKILL.md。";
pub const INCLUDED_SKILL_LIMIT: usize = 240;
pub const SKILL_PACK_DOCUMENT_KIND: &str = "skillpack";
pub const SKILL_PACK_GRAPH_TYPE: &str = "skillpack_workflow";
pub const SKILL_PACK_NAME_LIMIT: usize = 123;

pub struct RenderedArtifact { pub target: String, pub path: String, pub content: String }
pub trait SkillAdapter: Send + Sync {
    fn target(&self) -> &str;
    fn render(&self, pack: &SkillPack) -> Result<RenderedArtifact>;
}
pub struct SkillPackAdapter;
impl SkillPackAdapter {
    pub fn new() -> Self;
    pub fn validate_members(pack: &SkillPack) -> Result<()>;
    pub fn render(&self, pack: &SkillPack) -> Result<RenderedArtifact>;
}
pub fn render_package(pack: &SkillPack, target_kind: &str) -> Result<RenderedArtifact>;
```

`SkillPackAdapter` renders one SDD-0010 `SKILL.md` root document: YAML front matter (`name`, `kind: skillpack`, `description`, `skills` in chain order) and the sections `When to use`, `Do not use`, `Required inputs`, `Side effects and confirmation`, `Included Skills`, `Execution Process`, `Failure handling`, `Final output`, and `Workflow Graph`.
`validate_members` requires at least two unique, non-blank members whose `capability_type` is `skill` (case-insensitive) and a simple linear `can_feed` chain; packaging already enforces the recipe's `active` + `verified` status, and a pack does not retain it, so that part of Python's contract is checked at pack time.
`render_package` dispatches `skill` (and the empty target kind) to `SkillPackAdapter`; any other target is `Error::Unsupported` naming `flow render target \`<name>\``.

### Review and install

```rust
pub const REVIEW_SCHEMA_VERSION: &str = "symphony.package_review.v1";
pub const VERDICT_APPROVED: &str = "approved";
pub const VERDICT_REJECTED: &str = "rejected";
pub const VERDICT_NEEDS_HUMAN_REVIEW: &str = "needs_human_review";
pub const CHECK_PASS: &str = "pass";
pub const CHECK_FAIL: &str = "fail";
pub const REVIEW_MODE_DETERMINISTIC: &str = "deterministic";
pub const REVIEW_MODE_LLM: &str = "llm";

pub struct Reviewer;
impl Reviewer { pub fn new() -> Self; pub fn review(&self, pack: &SkillPack) -> ReviewResult; }
pub struct ReviewCheck { pub check: String, pub result: String /*pass|fail*/, pub reasons: Vec<String> }
pub struct ReviewResult { pub review_id: String, pub package_id: String, pub target_kind: String, pub verdict: String, pub checks: Vec<ReviewCheck>, pub materials_hash: String, pub reviewed_at: String, pub schema_version: String, pub review_mode: String /*deterministic|llm*/, pub notes: Vec<String> }
impl ReviewResult { pub fn failed_checks(&self) -> Vec<&ReviewCheck>; pub fn failure_reasons(&self) -> Vec<String>; }
pub struct InstallPreparation { pub recipe_id: String, pub target_kind: String, pub verdict: String, pub package: Option<SkillPack>, pub artifact_dir: Option<PathBuf>, pub review: Option<ReviewResult>, pub reasons: Vec<String> }
```

The `Reviewer` runs ten checks in order — `structure`, `dependencies`, `code_safety`, `secrets`, `artifact_paths`, `integrity`, `meta_naming`, `materials`, `permissions`, `license` — and returns `approved` when all pass, `needs_human_review` when `meta_naming` is the only failure, and `rejected` otherwise.

`code_safety` scans every rendered artifact (including `swarmflow/run.py`) for the forbidden tokens, and runs `validate_generated_script` over a pack's generated script.

A pack may not request `credential_access`, `filesystem_write`, `process`, or `shell`, and may only declare `Apache-2.0`, `BSD-3-Clause`, `MIT`, or `Proprietary`; `review_id` is `review_<12 hex>` over `package_id`, `target_kind`, `verdict`, `checks` and `materials_hash`, and `reviewed_at`, `review_mode` and `notes` are metadata only.

`review_and_prepare_install`/`prepare_install` return structural rejections (unsupported target kind, unknown recipe, stale requested version, failed review) as `InstallPreparation::verdict` with reasons, not errors, and only I/O failures are `Err`.

### Model-driven review

```rust
pub const REVIEW_CHECK_NAMES: [&str; 10];                     // the ten check names, in review order
pub const MIN_FINDING_CONFIDENCE: f64 = 0.5;
pub const MAX_REASON_CHARS: usize = 500;
pub const MAX_NOTE_CHARS: usize = 500;
pub const MAX_NOTES: usize = 8;
pub const MAX_REASONS_PER_CHECK: usize = 3;
pub const MODEL_REASON_PREFIX: &str = "model review: ";
pub const FALLBACK_NOTE_PREFIX: &str = "deterministic_fallback: ";
pub const SKIPPED_NOTE_PREFIX: &str = "model review skipped: ";
pub const MAX_ARTIFACT_CHARS: usize = 2_000;
pub const MAX_ARTIFACTS: usize = 8;
pub const MAX_STEP_DESCRIPTION_CHARS: usize = 400;
pub const MAX_PROMPT_STEPS: usize = 32;
pub const MAX_FIELD_CHARS: usize = 1_000;
pub const REVIEW_SYSTEM_PROMPT: &str;                          // the pinned system instruction

pub struct LlmPackageReviewer;
impl LlmPackageReviewer {
    pub fn new(model: Arc<dyn CompletionModel>) -> Self;
    pub fn without_model() -> Self;
    pub fn with_fallback(self, fallback: FallbackMode) -> Self;
    pub fn fallback(&self) -> FallbackMode;
    pub fn has_model(&self) -> bool;
    pub async fn review(&self, pack: &SkillPack, deterministic: ReviewResult) -> Result<ReviewResult>;
    pub async fn review_pack(&self, pack: &SkillPack) -> Result<ReviewResult>;
    pub fn render_prompt(pack: &SkillPack, deterministic: &ReviewResult) -> String;
}
pub struct LlmReviewFinding { pub check: String, pub result: String /*pass|fail*/, pub reason: String, pub confidence: f64 /*defaults to 1.0*/ }
```

Python's `flow.review.LLMPackageReviewAgent` returns one verdict; this port returns findings bound to the ten deterministic check names, so the model can add detail but never rewrite machine-checked ground truth.

The merge is fail-closed: a model `fail` at or above `MIN_FINDING_CONFIDENCE` appends `model review: <reason>` to that check (capped at `MAX_REASONS_PER_CHECK`) and marks it failed; a model `pass` is ignored entirely and can never clear a deterministic failure, so the deterministic `secrets` and `code_safety` verdicts stay authoritative.
The verdict is then re-derived with the three-state rule and `review_id` is refreshed over the merged checks.

When the deterministic verdict is not `approved` the stage is not run at all (Python's gate does the same), so a rejected or human-review package is never sent to a model and the result gains a `model review skipped:` note; `LlmPackageReviewer::without_model` answers the deterministic result unchanged.

The reply must be an object with a `checks` array; unknown check names, non-object items, unknown `result` values and non-string notes are dropped with a `tracing::warn!` naming the reply item, an absent `confidence` means `1.0` and a non-numeric one warns, and only a missing or non-array `checks` (or no JSON object) is an unusable reply.
Reasons are redacted with `redact_sensitive_text` and clamped on character boundaries before they enter a `ReviewResult`.

Provenance is explicit: a merged review carries `review_mode = "llm"`, a review the stage did not touch carries `"deterministic"`, an unusable reply is `Error::Llm` naming `llm package review` under `FallbackMode::Error`, and under `FallbackMode::Deterministic` it keeps the deterministic verdict with a `deterministic_fallback: <reason>` note plus a `tracing::warn!`.
A `deterministic` review handed to `review` for another package id, or one that is already an `llm` review, is `Error::Validation`.

The prompt is one byte-stable rendering: the pinned `REVIEW_SYSTEM_PROMPT` plus a canonical JSON payload of the package (redacted with `redact_private_json`, descriptions and trigger conditions capped at `MAX_FIELD_CHARS`, artifact text at `MAX_ARTIFACT_CHARS` for at most `MAX_ARTIFACTS`, steps at `MAX_PROMPT_STEPS` with `steps_total`/`artifacts_truncated` flags) and the deterministic checks.
`FlowEngine::review_and_prepare_install_with_reviewer` runs the deterministic gate, then the model stage, then the same approve-or-report path; `review_and_prepare_install` and `prepare_install` stay deterministic-only.

## Constraints

- Offline and deterministic: no network; `reviewed_at` and `distilled_at` are metadata and never decide ordering or equality.
- Every write is atomic (`.<name>.<uuid>.tmp` plus rename) and eager; `close` has nothing to flush and never fails.
- Recipe versions are immutable; a content change appends `version + 1`, an unchanged replay writes nothing.
- Only an `active`, `verified` linear chain can be packed; every other shape is a typed validation error.
- Every public item is documented; no `unsafe`, no `println!`, no panics on user input.
- Reuse `CapabilityIo` and `openjiuwen-retrieval`'s `CompletionModel`; do not duplicate shared types.
- The ten deterministic checks never call a model; only `review_llm.rs` does, and it can only add failures — never remove one and never upgrade a rejected package.
- Files stay around or under 1000 lines: `pack.rs`, `review.rs` and `review_llm.rs` are the largest and are split by responsibility.

## Routing Table

- `./mod.rs` → module docs and re-exports of the flow surface
- `./model/` → data models and normalization
  - `./model/mod.rs` → `ExperienceRecipe`, `CombinationStructure`, `RecipeQuality`, canonical JSON/labels, schema constants, thresholds
  - `./model/trace.rs` → `FlowTrace`, `ExecutionGraph`, `TraceNode`/`TraceEdge`, `TraceOutcome`, `from_value` normalization
- `./store.rs` → atomic flow-root persistence (`evidence.jsonl`, recipe versions, packages)
- `./engine.rs` → `FlowEngine`, `DistillReport`, grouping, versioning, quality/status, optional narrative
- `./codegen.rs` → `generate_swarmflow_script`, `swarmflow_identifier`, `validate_generated_script`, `python_string_literal`, `SwarmflowScript`
- `./pack.rs` → `SkillPack`, content-addressed identity, `SKILL.md`/`dependencies.yaml`/`swarmflow/run.py` render, save/load
- `./render.rs` → `SkillPackAdapter`, `SkillAdapter`, `RenderedArtifact`, `render_package`
- `./review.rs` → `Reviewer`, the ten checks, verdicts, `ReviewResult`, `InstallPreparation`, the install preparation path
- `./review_llm.rs` → `LlmPackageReviewer`, the pinned system prompt, reply parsing, the fail-closed merge, prompt rendering

## Tests

- `tests/flow.rs` — normalization, ingestion, grouping, versioning, persistence, packing/rendering/loading (including `swarmflow/run.py`), the ten review checks, installation preparation, and the model-narrative fallback.
- `tests/flow.rs` (model review) — findings added to an approved pack, a deterministic `secrets` failure the model cannot overturn (and never sees), malformed replies under both fallback modes, dropped unknown checks and low-confidence findings, redacted/clamped reasons, byte-identical reruns of one scripted reply, the model-free reviewer, and installation preparation both blocked and approved by the stage.
- `tests/flow_codegen.rs` — pinned two-step script, empty-combination script, identifier stability/content sensitivity, quote/newline/CJK escaping, the validator's accept/reject cases and first-bad-line reporting, member validation failures, the unsupported render target, script participation in the package id, and a full pack → review → install path containing `run.py`.
