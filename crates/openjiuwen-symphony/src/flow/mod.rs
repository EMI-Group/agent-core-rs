//! Flow distillation and reusable SkillPack artifacts.
//!
//! Ports the offline, deterministic core of Python `openjiuwen.symphony.flow`:
//! execution-trace evidence is normalized, success traces are grouped by their
//! capability-edge structure, each group is distilled into a versioned
//! [`ExperienceRecipe`](crate::flow::model::ExperienceRecipe), and an active,
//! verified recipe can be frozen into a content-addressed
//! [`SkillPack`](crate::flow::pack::SkillPack) with a static review gate.
//!
//! The subsystem is host-driven: it never executes a user task, never installs
//! anything outside the caller's destination directory, and never depends on the
//! network. An optional [`CompletionModel`](openjiuwen_retrieval::CompletionModel)
//! only supplies narrative prose; every structural decision is deterministic and
//! a model failure falls back to the documented template.
//!
//! # Module map
//!
//! - [`model`] — execution graphs, traces, recipe quality, recipes, recipes and
//!   display candidates, plus the shared canonical-JSON and naming helpers.
//! - [`store`] — atomic filesystem persistence (`evidence.jsonl`,
//!   `recipes/<id>/vN.json`, `recipes/<id>/current.json`, `packages/<id>.json`).
//! - [`engine`](crate::flow::engine) — `FlowEngine`: open/ingest/distill/list/candidates/close.
//! - [`pack`](crate::flow::pack) — `SkillPack`, its `SKILL.md`/`dependencies.yaml` rendering,
//!   content-addressed identity, and save/load round-tripping.
//! - [`review`](crate::flow::review) — `Reviewer`, the seven deterministic checks, `ReviewResult`,
//!   and `FlowEngine::review_and_prepare_install`.
//!
//! # Determinism
//!
//! Every iteration boundary is a sorted collection, timestamps are metadata-only
//! (`reviewed_at`, `distilled_at`) and never participate in ids or equality, and
//! content-addressed ids are sha256 over canonical JSON (object keys sorted).

pub mod engine;
pub mod model;
pub mod pack;
pub mod review;
pub mod store;

pub use engine::{DistillReport, FlowEngine};
pub use model::{
    canonical_json, content_hash, is_valid_meta_name, normalize_capability_id, normalize_meta_name,
    sha256_hex, sha256_short, CombinationCandidate, CombinationStructure, ExecutionGraph,
    ExperienceRecipe, FlowTrace, RecipeEdge, RecipeQuality, TraceEdge, TraceNode, TraceOutcome,
    CAN_FEED, MIN_PACK_SUCCESS_RATE, MIN_SUCCESSES, PACKAGE_SCHEMA_VERSION, RECIPE_GRADE_CANDIDATE,
    RECIPE_GRADE_VERIFIED, RECIPE_SCHEMA_VERSION, RECIPE_STATUS_ACTIVE, RECIPE_STATUS_DRAFT,
    SKILL_PACK_TYPE,
};
pub use pack::{SkillPack, SkillPackStep};
pub use review::{
    InstallPreparation, ReviewCheck, ReviewResult, Reviewer, CHECK_FAIL, CHECK_PASS,
    REVIEW_SCHEMA_VERSION, VERDICT_APPROVED, VERDICT_NEEDS_HUMAN_REVIEW, VERDICT_REJECTED,
};
