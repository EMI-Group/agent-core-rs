//! The deterministic package review gate and installation preparation.
//!
//! Ports Python `flow.review.PackageReviewGate.review_static` plus
//! `flow.engine.SymphonyFlowEngine.review_and_prepare_install`, reduced to the
//! ten checks the Rust objective keeps, in Python's order:
//! `structure`, `dependencies`, `code_safety`, `secrets`, `artifact_paths`,
//! `integrity`, `meta_naming`, `materials`, `permissions`, and `license`.
//!
//! The verdict is `approved` when every check passes, `needs_human_review` when
//! the only failure is `meta_naming` (the name is cosmetic and fixable), and
//! `rejected` otherwise. No check invokes a model or touches the network, so this
//! gate is always available offline.
//!
//! # The optional model stage
//!
//! [`LlmPackageReviewer`] is a second,
//! opt-in stage that asks a [`CompletionModel`](openjiuwen_retrieval::CompletionModel)
//! for extra findings over an otherwise approved package. Its merge is fail-closed:
//! it can only add failures, never remove one. A review produced by the model stage
//! carries [`REVIEW_MODE_LLM`] in [`ReviewResult::review_mode`], and a review whose
//! model stage was skipped or degraded carries [`REVIEW_MODE_DETERMINISTIC`] plus a
//! note explaining why — neither path is silent.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use openjiuwen_common::utils::id::now_rfc3339;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::flow::engine::FlowEngine;
use crate::flow::model::{canonical_json, is_valid_meta_name, sha256_short};
use crate::flow::pack::{
    chain_order_from_steps, DependenciesDoc, SkillPack, SKILL_MD_FILENAME, TARGET_KIND_SKILL,
};
use crate::flow::review_llm::LlmPackageReviewer;
use crate::redaction::redact_private_json;

/// The review schema version.
pub const REVIEW_SCHEMA_VERSION: &str = "symphony.package_review.v1";
/// The verdict for a package that passed every check.
pub const VERDICT_APPROVED: &str = "approved";
/// The verdict for a package with at least one non-cosmetic failure.
pub const VERDICT_REJECTED: &str = "rejected";
/// The verdict for a package whose only failure is the installable name.
pub const VERDICT_NEEDS_HUMAN_REVIEW: &str = "needs_human_review";
/// A passing check result.
pub const CHECK_PASS: &str = "pass";
/// A failing check result.
pub const CHECK_FAIL: &str = "fail";
/// [`ReviewResult::review_mode`] for a review produced by the deterministic gate.
pub const REVIEW_MODE_DETERMINISTIC: &str = "deterministic";
/// [`ReviewResult::review_mode`] for a review whose model stage answered usablely.
pub const REVIEW_MODE_LLM: &str = "llm";

/// Permissions a generated package may never request.
const FORBIDDEN_PERMISSIONS: [&str; 4] =
    ["credential_access", "filesystem_write", "process", "shell"];
/// Licenses a generated package may declare.
const ALLOWED_LICENSES: [&str; 4] = ["Apache-2.0", "BSD-3-Clause", "MIT", "Proprietary"];
/// Tokens embedded script text may never contain (Python's `_FORBIDDEN_CODE_TOKENS`).
pub(crate) const FORBIDDEN_CODE_TOKENS: [&str; 5] =
    ["import os", "import sys", "subprocess", "eval(", "exec("];
/// The review check that only warrants human review when it is the sole failure.
const META_NAMING_CHECK: &str = "meta_naming";

/// One static check result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReviewCheck {
    /// The check name, in review order.
    pub check: String,
    /// `pass` or `fail`.
    pub result: String,
    /// Human-readable failure reasons; empty on success.
    pub reasons: Vec<String>,
}

/// The static review verdict for one package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReviewResult {
    /// The deterministic review id (`review_<12 hex>`).
    pub review_id: String,
    /// The reviewed package id.
    pub package_id: String,
    /// The artifact target kind.
    pub target_kind: String,
    /// `approved`, `rejected`, or `needs_human_review`.
    pub verdict: String,
    /// The deterministic checks, in review order.
    pub checks: Vec<ReviewCheck>,
    /// The sha256 hex digest of the package's canonical materials.
    pub materials_hash: String,
    /// When the review ran (metadata only).
    pub reviewed_at: String,
    /// The review schema version.
    pub schema_version: String,
    /// How the verdict was produced: [`REVIEW_MODE_DETERMINISTIC`] or
    /// [`REVIEW_MODE_LLM`].
    ///
    /// The deterministic gate always sets `deterministic`; only
    /// [`LlmPackageReviewer`] sets
    /// `llm`, and only after it accepted a usable model reply.
    pub review_mode: String,
    /// Model-authored notes, plus the `deterministic_fallback:` /
    /// `model review skipped:` markers that explain a stage which did not run.
    ///
    /// Empty for a review the deterministic gate produced on its own.
    pub notes: Vec<String>,
}

impl Default for ReviewResult {
    fn default() -> Self {
        Self {
            review_id: String::new(),
            package_id: String::new(),
            target_kind: TARGET_KIND_SKILL.to_string(),
            verdict: VERDICT_REJECTED.to_string(),
            checks: Vec::new(),
            materials_hash: String::new(),
            reviewed_at: String::new(),
            schema_version: REVIEW_SCHEMA_VERSION.to_string(),
            review_mode: REVIEW_MODE_DETERMINISTIC.to_string(),
            notes: Vec::new(),
        }
    }
}

impl ReviewResult {
    /// Returns the failing checks.
    #[must_use]
    pub fn failed_checks(&self) -> Vec<&ReviewCheck> {
        self.checks
            .iter()
            .filter(|check| check.result == CHECK_FAIL)
            .collect()
    }

    /// Renders `check: reason` lines for every failing check.
    #[must_use]
    pub fn failure_reasons(&self) -> Vec<String> {
        self.failed_checks()
            .into_iter()
            .map(|check| {
                let detail = if check.reasons.is_empty() {
                    "no detail".to_string()
                } else {
                    check.reasons.join("; ")
                };
                format!("{}: {detail}", check.check)
            })
            .collect()
    }
}

/// The deterministic, offline package reviewer.
#[derive(Debug, Clone, Copy, Default)]
pub struct Reviewer;

impl Reviewer {
    /// Creates a reviewer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Runs all ten static checks against `pack`.
    ///
    /// The result is fully deterministic apart from `reviewed_at`: `review_id`
    /// hashes the package id, target kind, verdict, checks, and materials hash.
    #[must_use]
    pub fn review(&self, pack: &SkillPack) -> ReviewResult {
        let materials_hash = pack.materials_hash();
        let checks = vec![
            check_structure(pack),
            check_dependencies(pack),
            check_code_safety(pack),
            check_secrets(pack),
            check_artifact_paths(pack),
            check_integrity(pack, &materials_hash),
            check_meta_naming(pack),
            check_materials(pack),
            check_permissions(pack),
            check_license(pack),
        ];
        let verdict = resolve_verdict(&checks);
        let review_id = compute_review_id(
            &pack.package_id,
            &pack.target_kind,
            &verdict,
            &checks,
            &materials_hash,
        );
        ReviewResult {
            review_id,
            package_id: pack.package_id.clone(),
            target_kind: pack.target_kind.clone(),
            verdict,
            checks,
            materials_hash,
            reviewed_at: now_rfc3339(),
            schema_version: REVIEW_SCHEMA_VERSION.to_string(),
            review_mode: REVIEW_MODE_DETERMINISTIC.to_string(),
            notes: Vec::new(),
        }
    }
}

/// Hashes the verdict identity into a `review_<12 hex>` id.
///
/// `reviewed_at`, `review_mode` and `notes` are metadata and deliberately stay out
/// of the digest, so the same verdict over the same materials keeps the same id
/// whichever stage produced it.
fn compute_review_id(
    package_id: &str,
    target_kind: &str,
    verdict: &str,
    checks: &[ReviewCheck],
    materials_hash: &str,
) -> String {
    let identity = serde_json::json!({
        "package_id": package_id,
        "target_kind": target_kind,
        "verdict": verdict,
        "checks": checks,
        "materials_hash": materials_hash,
    });
    format!("review_{}", sha256_short(&canonical_json(&identity), 12))
}

/// Re-derives [`ReviewResult::review_id`] and [`ReviewResult::verdict`] after a
/// stage merged extra findings.
pub(crate) fn refresh_review_id(result: &mut ReviewResult) {
    result.review_id = compute_review_id(
        &result.package_id,
        &result.target_kind,
        &result.verdict,
        &result.checks,
        &result.materials_hash,
    );
}

/// Resolves the three-state verdict from the checks.
pub(crate) fn resolve_verdict(checks: &[ReviewCheck]) -> String {
    let failures: Vec<&ReviewCheck> = checks
        .iter()
        .filter(|check| check.result == CHECK_FAIL)
        .collect();
    if failures.is_empty() {
        VERDICT_APPROVED.to_string()
    } else if failures.len() == 1 && failures[0].check == META_NAMING_CHECK {
        VERDICT_NEEDS_HUMAN_REVIEW.to_string()
    } else {
        VERDICT_REJECTED.to_string()
    }
}

/// Builds one check from its reasons.
fn check(name: &str, reasons: Vec<String>) -> ReviewCheck {
    ReviewCheck {
        check: name.to_string(),
        result: if reasons.is_empty() {
            CHECK_PASS.to_string()
        } else {
            CHECK_FAIL.to_string()
        },
        reasons,
    }
}

/// Checks that the package has the required fields and a linear chain.
fn check_structure(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    if pack.package_id.trim().is_empty() {
        reasons.push("package_id is required".to_string());
    }
    if pack.recipe_id.trim().is_empty() {
        reasons.push("recipe_id is required".to_string());
    }
    if pack.schema_version.trim().is_empty() {
        reasons.push("schema_version is required".to_string());
    }
    if pack.target_kind != TARGET_KIND_SKILL {
        reasons.push(format!("unsupported target_kind: {}", pack.target_kind));
    }
    if pack.name.trim().is_empty() {
        reasons.push("name is required".to_string());
    }
    if let Err(reason) = chain_order_from_steps(&pack.steps) {
        reasons.push(reason);
    }
    check("structure", reasons)
}

/// Checks that the dependency manifest agrees with the steps and the chain.
fn check_dependencies(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    let step_ids: Vec<&str> = pack.step_ids();
    let known: BTreeSet<&str> = step_ids.iter().copied().collect();
    for step in &pack.steps {
        if step.capability_type.trim().is_empty() {
            reasons.push(format!("capability {} has no capability_type", step.id));
        }
        if step.name.trim().is_empty() {
            reasons.push(format!("capability {} has no name", step.id));
        }
        for dependency in &step.depends_on {
            if dependency == &step.id {
                reasons.push(format!("capability {} depends on itself", step.id));
            } else if !known.contains(dependency.as_str()) {
                reasons.push(format!(
                    "capability {} depends on unknown capability {}",
                    step.id, dependency
                ));
            }
        }
    }
    match pack.artifacts.get(crate::flow::pack::DEPENDENCIES_FILENAME) {
        None => reasons.push("dependencies.yaml artifact is missing".to_string()),
        Some(text) => match serde_yaml_ng::from_str::<DependenciesDoc>(text) {
            Err(error) => reasons.push(format!("dependencies.yaml is invalid: {error}")),
            Ok(document) => {
                let manifest_ids: Vec<&str> = document
                    .capabilities
                    .iter()
                    .map(|entry| entry.id.as_str())
                    .collect();
                if manifest_ids != step_ids {
                    reasons.push(
                        "dependencies.yaml capabilities do not match the step order".to_string(),
                    );
                }
                for entry in &document.capabilities {
                    if entry.version.trim().is_empty() {
                        reasons.push(format!("capability {} has no version", entry.id));
                    }
                }
            }
        },
    }
    check("dependencies", reasons)
}

/// Checks the content-addressed identity and rendered artifact consistency.
fn check_integrity(pack: &SkillPack, materials_hash: &str) -> ReviewCheck {
    let mut reasons = Vec::new();
    let expected = format!("cap-{}", &materials_hash[..12]);
    if pack.package_id != expected {
        reasons.push(format!(
            "integrity hash mismatch: package_id is `{}`, materials give `{expected}`",
            pack.package_id
        ));
    }
    if !pack.artifacts.is_empty() {
        let rendered = pack.render();
        if pack.artifacts != rendered {
            reasons.push("rendered artifacts do not match the package materials".to_string());
        }
    }
    check("integrity", reasons)
}

/// Checks the installable `meta.name` pattern.
fn check_meta_naming(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    if !is_valid_meta_name(&pack.name) {
        reasons.push(format!(
            "meta_name invalid: {:?} (expected ^[a-z0-9][a-z0-9-]{{0,63}}$)",
            pack.name
        ));
    }
    check(META_NAMING_CHECK, reasons)
}

/// Checks that the human-facing materials are present.
fn check_materials(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    if pack.description.trim().is_empty() {
        reasons.push("description is required".to_string());
    }
    match pack.artifacts.get(SKILL_MD_FILENAME) {
        None => reasons.push("SKILL.md artifact is missing".to_string()),
        Some(text) if text.trim().is_empty() => {
            reasons.push("SKILL.md artifact is empty".to_string());
        }
        Some(_) => {}
    }
    if pack.steps.is_empty() {
        reasons.push("package has no steps".to_string());
    }
    check("materials", reasons)
}

/// Checks that the package embeds no forbidden script text.
///
/// Mirrors Python's `_check_code_safety`: generated or embedded script text may
/// not contain `import os`, `import sys`, `subprocess`, `eval(`, or `exec(`, and
/// a generated `SwarmFlow` script must also pass
/// [`validate_generated_script`](crate::flow::codegen::validate_generated_script).
/// The token scan is case-insensitive and covers every rendered artifact, in
/// artifact name order.
fn check_code_safety(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    for (name, text) in &pack.artifacts {
        let lowered = text.to_lowercase();
        for token in FORBIDDEN_CODE_TOKENS {
            if lowered.contains(token) {
                reasons.push(format!("forbidden token in {name}: {token}"));
            }
        }
    }
    if let Some(script) = &pack.swarmflow_script {
        if let Err(error) = crate::flow::codegen::validate_generated_script(script) {
            reasons.push(format!("swarmflow_script: {}", error.message()));
        }
    }
    check("code_safety", reasons)
}

/// Checks that no material still carries credential material.
///
/// Mirrors Python's `_check_secrets`: the package materials are passed through
/// [`redact_private_json`] and the check fails when redaction changes them.
fn check_secrets(pack: &SkillPack) -> ReviewCheck {
    let materials = pack.materials();
    let mut reasons = Vec::new();
    if redact_private_json(&materials, &[]) != materials {
        reasons.push("package contains unredacted credential material".to_string());
    }
    check("secrets", reasons)
}

/// Checks that every material path stays inside the package root.
///
/// Mirrors and extends Python's `_check_artifact_paths`: absolute paths, Windows
/// drive or UNC prefixes, and `..` escapes are all rejected after path
/// normalization, so an artifact can never be written outside the directory the
/// pack claims.
fn check_artifact_paths(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    for name in pack.artifacts.keys() {
        if let Some(reason) = escaping_material_path(name) {
            reasons.push(reason);
        }
    }
    check("artifact_paths", reasons)
}

/// Returns why `name` escapes the package root, or `None` when it is contained.
///
/// Shared with [`SkillPack::save`](crate::flow::pack::SkillPack::save), so the
/// writer accepts exactly the relative paths the review gate allows.
pub(crate) fn escaping_material_path(name: &str) -> Option<String> {
    if name.trim().is_empty() {
        return Some("material path is blank".to_string());
    }
    if name.starts_with('/') || name.starts_with('\\') {
        return Some(format!("material path `{name}` is absolute"));
    }
    let bytes = name.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Some(format!("material path `{name}` has a drive prefix"));
    }
    let parts: Vec<&str> = name
        .split(['/', '\\'])
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let mut depth = 0usize;
    for part in parts {
        if part == ".." {
            if depth == 0 {
                return Some(format!("material path `{name}` escapes the package root"));
            }
            depth -= 1;
        } else {
            depth += 1;
        }
    }
    if depth == 0 {
        return Some(format!("material path `{name}` does not name a file"));
    }
    None
}

/// Checks that the declared permissions exclude the forbidden capabilities.
fn check_permissions(pack: &SkillPack) -> ReviewCheck {
    let forbidden: Vec<&str> = FORBIDDEN_PERMISSIONS
        .iter()
        .copied()
        .filter(|permission| pack.permissions.iter().any(|item| item == permission))
        .collect();
    let mut reasons = Vec::new();
    if !forbidden.is_empty() {
        reasons.push(format!("forbidden permissions: {}", forbidden.join(", ")));
    }
    check("permissions", reasons)
}

/// Checks that the declared license is present and supported.
fn check_license(pack: &SkillPack) -> ReviewCheck {
    let mut reasons = Vec::new();
    let license = pack.license.trim();
    if license.is_empty() {
        reasons.push("license is required".to_string());
    } else if !ALLOWED_LICENSES.contains(&license) {
        reasons.push(format!("unsupported license: {license}"));
    }
    check("license", reasons)
}

/// The result of preparing one package for installation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InstallPreparation {
    /// The recipe that was prepared.
    pub recipe_id: String,
    /// The artifact target kind.
    pub target_kind: String,
    /// The final verdict.
    pub verdict: String,
    /// The frozen package, when packing succeeded.
    pub package: Option<SkillPack>,
    /// The directory the artifacts were rendered into, when approved.
    pub artifact_dir: Option<PathBuf>,
    /// The static review, when packing succeeded.
    pub review: Option<ReviewResult>,
    /// Rejection reasons; empty on approval.
    pub reasons: Vec<String>,
}

impl FlowEngine {
    /// Reviews the current version of `recipe_id` and renders it into `dest_dir`
    /// when the review approves it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`](openjiuwen_common::error::Error::external)
    /// when writing the approved artifacts fails. Structural rejections (unknown
    /// recipe, unsupported target kind, draft recipe, failed review) are returned
    /// as [`InstallPreparation::verdict`] values with reasons, not errors.
    pub async fn review_and_prepare_install(
        &self,
        recipe_id: &str,
        target_kind: &str,
        dest_dir: impl AsRef<Path>,
    ) -> Result<InstallPreparation> {
        self.prepare_target(recipe_id, None, target_kind, dest_dir.as_ref(), None)
            .await
    }

    /// Reviews the current version of `recipe_id` with an optional model stage and
    /// renders it into `dest_dir` when the merged review still approves it.
    ///
    /// The model stage can only add failures, so this is never more permissive
    /// than [`FlowEngine::review_and_prepare_install`]; a pack the deterministic
    /// gate rejects is returned unchanged without a model call.
    ///
    /// # Errors
    ///
    /// Same as [`FlowEngine::review_and_prepare_install`], plus the model-stage
    /// error a [`FallbackMode::Error`](crate::llm::FallbackMode::Error) reviewer
    /// returns for an unusable reply.
    pub async fn review_and_prepare_install_with_reviewer(
        &self,
        recipe_id: &str,
        target_kind: &str,
        dest_dir: impl AsRef<Path>,
        reviewer: &LlmPackageReviewer,
    ) -> Result<InstallPreparation> {
        self.prepare_target(
            recipe_id,
            None,
            target_kind,
            dest_dir.as_ref(),
            Some(reviewer),
        )
        .await
    }

    /// Reviews one explicitly requested recipe version and renders it when
    /// approved; a version that is not current is rejected as stale.
    ///
    /// # Errors
    ///
    /// Same as [`FlowEngine::review_and_prepare_install`].
    pub async fn prepare_install(
        &self,
        recipe_id: &str,
        recipe_version: u64,
        target_kind: &str,
        dest_dir: impl AsRef<Path>,
    ) -> Result<InstallPreparation> {
        self.prepare_target(
            recipe_id,
            Some(recipe_version),
            target_kind,
            dest_dir.as_ref(),
            None,
        )
        .await
    }

    /// Shared implementation of the installation-preparation entry points.
    async fn prepare_target(
        &self,
        recipe_id: &str,
        requested_version: Option<u64>,
        target_kind: &str,
        dest_dir: &Path,
        reviewer: Option<&LlmPackageReviewer>,
    ) -> Result<InstallPreparation> {
        if target_kind != TARGET_KIND_SKILL {
            return Ok(rejected(
                recipe_id,
                target_kind,
                vec![format!("unsupported target_kind: {target_kind}")],
            ));
        }
        let Some(recipe) = self.recipe(recipe_id).await else {
            return Ok(rejected(
                recipe_id,
                target_kind,
                vec![format!("recipe not found: {recipe_id}")],
            ));
        };
        if let Some(requested) = requested_version {
            if recipe.version != requested {
                return Ok(rejected(
                    recipe_id,
                    target_kind,
                    vec![format!(
                        "stale recipe version: current=v{}, requested=v{requested}",
                        recipe.version
                    )],
                ));
            }
        }
        let pack = match self.pack(recipe_id).await {
            Ok(pack) => pack,
            Err(error) => {
                return Ok(rejected(
                    recipe_id,
                    target_kind,
                    vec![error.message().to_string()],
                ));
            }
        };
        let review = Reviewer::new().review(&pack);
        let review = match reviewer {
            Some(reviewer) => reviewer.review(&pack, review).await?,
            None => review,
        };
        if review.verdict != VERDICT_APPROVED {
            return Ok(InstallPreparation {
                recipe_id: recipe_id.to_string(),
                target_kind: target_kind.to_string(),
                verdict: review.verdict.clone(),
                package: Some(pack),
                artifact_dir: None,
                reasons: review.failure_reasons(),
                review: Some(review),
            });
        }
        pack.save(dest_dir).await?;
        Ok(InstallPreparation {
            recipe_id: recipe_id.to_string(),
            target_kind: target_kind.to_string(),
            verdict: VERDICT_APPROVED.to_string(),
            package: Some(pack),
            artifact_dir: Some(dest_dir.to_path_buf()),
            review: Some(review),
            reasons: Vec::new(),
        })
    }
}

/// Builds a rejection preparation.
fn rejected(recipe_id: &str, target_kind: &str, reasons: Vec<String>) -> InstallPreparation {
    InstallPreparation {
        recipe_id: recipe_id.to_string(),
        target_kind: target_kind.to_string(),
        verdict: VERDICT_REJECTED.to_string(),
        package: None,
        artifact_dir: None,
        review: None,
        reasons,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn demo_pack(name: &str) -> SkillPack {
        let mut pack = SkillPack {
            recipe_id: "recipe_0123456789ab".to_string(),
            name: name.to_string(),
            description: "[技能包] demo".to_string(),
            steps: vec![
                crate::flow::pack::SkillPackStep {
                    id: "a".to_string(),
                    name: "a".to_string(),
                    description: "first".to_string(),
                    capability_type: "skill".to_string(),
                    ..crate::flow::pack::SkillPackStep::default()
                },
                crate::flow::pack::SkillPackStep {
                    id: "b".to_string(),
                    name: "b".to_string(),
                    description: "second".to_string(),
                    capability_type: "skill".to_string(),
                    depends_on: vec!["a".to_string()],
                    ..crate::flow::pack::SkillPackStep::default()
                },
            ],
            license: "Proprietary".to_string(),
            ..SkillPack::default()
        };
        pack.recompute_identity();
        pack
    }

    #[test]
    fn a_valid_pack_is_approved() {
        let result = Reviewer::new().review(&demo_pack("demo-pack"));
        assert_eq!(result.verdict, VERDICT_APPROVED);
        assert_eq!(result.failed_checks().len(), 0);
        let names: Vec<&str> = result
            .checks
            .iter()
            .map(|check| check.check.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "structure",
                "dependencies",
                "code_safety",
                "secrets",
                "artifact_paths",
                "integrity",
                "meta_naming",
                "materials",
                "permissions",
                "license"
            ]
        );
        assert!(result.review_id.starts_with("review_"));
        assert_eq!(result.review_id.len(), "review_".len() + 12);
    }

    #[test]
    fn an_invalid_meta_name_needs_human_review() {
        let mut pack = demo_pack("Bad Name");
        pack.recompute_identity();
        let result = Reviewer::new().review(&pack);
        assert_eq!(result.verdict, VERDICT_NEEDS_HUMAN_REVIEW);
        assert_eq!(result.failed_checks().len(), 1);
        assert_eq!(result.failed_checks()[0].check, META_NAMING_CHECK);
    }

    #[test]
    fn a_missing_license_or_forbidden_permission_is_rejected() {
        let mut pack = demo_pack("demo-pack");
        pack.license.clear();
        let result = Reviewer::new().review(&pack);
        assert_eq!(result.verdict, VERDICT_REJECTED);
        assert!(result
            .failed_checks()
            .iter()
            .any(|check| check.check == "license"));

        let mut pack = demo_pack("demo-pack");
        pack.permissions = vec!["shell".to_string()];
        let result = Reviewer::new().review(&pack);
        assert_eq!(result.verdict, VERDICT_REJECTED);
        assert!(result
            .failed_checks()
            .iter()
            .any(|check| check.check == "permissions"));
    }

    #[test]
    fn tampered_artifacts_fail_integrity() {
        let mut pack = demo_pack("demo-pack");
        pack.artifacts
            .insert(SKILL_MD_FILENAME.to_string(), "tampered".to_string());
        let result = Reviewer::new().review(&pack);
        assert_eq!(result.verdict, VERDICT_REJECTED);
        assert!(result
            .failed_checks()
            .iter()
            .any(|check| check.check == "integrity"));
    }

    #[test]
    fn embedded_script_tokens_fail_code_safety() {
        for token in FORBIDDEN_CODE_TOKENS {
            let mut pack = demo_pack("demo-pack");
            pack.artifacts.insert(
                crate::flow::pack::DEPENDENCIES_FILENAME.to_string(),
                format!("#!/usr/bin/env python\n{token} payload\n"),
            );
            let result = Reviewer::new().review(&pack);
            let safety = result
                .checks
                .iter()
                .find(|check| check.check == "code_safety")
                .expect("code_safety present");
            assert_eq!(safety.result, CHECK_FAIL, "{token} should fail");
            assert!(safety
                .reasons
                .iter()
                .any(|reason| reason.contains(token) && reason.contains("dependencies.yaml")));
        }

        let mut clean = demo_pack("demo-pack");
        clean.artifacts.insert(
            SKILL_MD_FILENAME.to_string(),
            "import os is fine".to_string(),
        );
        let result = Reviewer::new().review(&clean);
        let safety = result
            .checks
            .iter()
            .find(|check| check.check == "code_safety")
            .expect("code_safety present");
        assert_eq!(safety.result, CHECK_FAIL);
    }

    #[test]
    fn unredacted_secrets_fail_the_secrets_check() {
        let mut pack = demo_pack("demo-pack");
        pack.steps[0].description = "call with api_key: sk-live-0123456789abcdef".to_string();
        pack.recompute_identity();
        let result = Reviewer::new().review(&pack);
        let secrets = result
            .checks
            .iter()
            .find(|check| check.check == "secrets")
            .expect("secrets present");
        assert_eq!(secrets.result, CHECK_FAIL);
        assert!(secrets.reasons[0].contains("unredacted credential material"));

        let clean = demo_pack("demo-pack");
        let result = Reviewer::new().review(&clean);
        let secrets = result
            .checks
            .iter()
            .find(|check| check.check == "secrets")
            .expect("secrets present");
        assert_eq!(secrets.result, CHECK_PASS);
    }

    #[test]
    fn material_paths_must_stay_inside_the_root() {
        for name in [
            "SKILL.md",
            "nested/dir/file.md",
            "./SKILL.md",
            "a/../SKILL.md",
            "nested\\file.md",
        ] {
            assert_eq!(escaping_material_path(name), None, "{name} is contained");
        }
        for name in [
            "",
            "   ",
            "/etc/passwd",
            "\\server\\share\\file",
            "//server/share/file",
            "C:/Windows/system32",
            "c:file.md",
            "../escape.md",
            "nested/../../escape.md",
            ".",
            "./",
        ] {
            assert!(
                escaping_material_path(name).is_some(),
                "{name} should be rejected"
            );
        }
    }

    #[test]
    fn escaping_material_paths_fail_the_check() {
        let mut pack = demo_pack("demo-pack");
        pack.artifacts
            .insert("../escape.md".to_string(), "payload".to_string());
        let result = Reviewer::new().review(&pack);
        let paths = result
            .checks
            .iter()
            .find(|check| check.check == "artifact_paths")
            .expect("artifact_paths present");
        assert_eq!(paths.result, CHECK_FAIL);
        assert!(paths.reasons[0].contains("escapes the package root"));
    }
}
