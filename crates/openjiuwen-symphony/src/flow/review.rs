//! The deterministic package review gate and installation preparation.
//!
//! Ports Python `flow.review.PackageReviewGate.review_static` plus
//! `flow.engine.SymphonyFlowEngine.review_and_prepare_install`, reduced to the
//! seven checks the Rust objective keeps:
//! `structure`, `dependencies`, `integrity`, `meta_naming`, `materials`,
//! `permissions`, and `license`.
//!
//! The verdict is `approved` when every check passes, `needs_human_review` when
//! the only failure is `meta_naming` (the name is cosmetic and fixable), and
//! `rejected` otherwise. No check invokes a model or touches the network.

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

/// Permissions a generated package may never request.
const FORBIDDEN_PERMISSIONS: [&str; 4] =
    ["credential_access", "filesystem_write", "process", "shell"];
/// Licenses a generated package may declare.
const ALLOWED_LICENSES: [&str; 4] = ["Apache-2.0", "BSD-3-Clause", "MIT", "Proprietary"];
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
    /// The seven checks, in order.
    pub checks: Vec<ReviewCheck>,
    /// The sha256 hex digest of the package's canonical materials.
    pub materials_hash: String,
    /// When the review ran (metadata only).
    pub reviewed_at: String,
    /// The review schema version.
    pub schema_version: String,
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

    /// Runs all seven static checks against `pack`.
    ///
    /// The result is fully deterministic apart from `reviewed_at`: `review_id`
    /// hashes the package id, target kind, verdict, checks, and materials hash.
    #[must_use]
    pub fn review(&self, pack: &SkillPack) -> ReviewResult {
        let materials_hash = pack.materials_hash();
        let checks = vec![
            check_structure(pack),
            check_dependencies(pack),
            check_integrity(pack, &materials_hash),
            check_meta_naming(pack),
            check_materials(pack),
            check_permissions(pack),
            check_license(pack),
        ];
        let verdict = resolve_verdict(&checks);
        let identity = serde_json::json!({
            "package_id": pack.package_id,
            "target_kind": pack.target_kind,
            "verdict": verdict,
            "checks": &checks,
            "materials_hash": materials_hash,
        });
        let review_id = format!("review_{}", sha256_short(&canonical_json(&identity), 12));
        ReviewResult {
            review_id,
            package_id: pack.package_id.clone(),
            target_kind: pack.target_kind.clone(),
            verdict,
            checks,
            materials_hash,
            reviewed_at: now_rfc3339(),
            schema_version: REVIEW_SCHEMA_VERSION.to_string(),
        }
    }
}

/// Resolves the three-state verdict from the checks.
fn resolve_verdict(checks: &[ReviewCheck]) -> String {
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
        self.prepare_target(recipe_id, None, target_kind, dest_dir.as_ref())
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
        )
        .await
    }

    /// Shared implementation of the two installation-preparation entry points.
    async fn prepare_target(
        &self,
        recipe_id: &str,
        requested_version: Option<u64>,
        target_kind: &str,
        dest_dir: &Path,
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
        assert_eq!(result.checks.len(), 7);
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
}
