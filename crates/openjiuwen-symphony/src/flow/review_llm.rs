//! The optional model-driven semantic review stage.
//!
//! Ports Python `flow.review.LLMPackageReviewAgent` onto the crate's one-prompt-in /
//! one-reply-out [`CompletionModel`] seam. Python's agent returns a single verdict;
//! [`LlmPackageReviewer`] instead returns *findings* bound to the ten deterministic
//! check names, so a model can never silently rewrite the machine-checked ground
//! truth. It runs **after** the deterministic gate and merges into its result.
//!
//! # Fail-closed merge
//!
//! The merge rule is one-directional, and it is the whole safety argument of this
//! module:
//!
//! - a model `fail` finding **adds** a failure reason to that check and marks it
//!   failed — it can never turn a `pass` into a `fail` anywhere else, and it can
//!   never shorten or remove a deterministic reason;
//! - a model `pass` finding is ignored entirely, so it can never clear a
//!   deterministic failure. The deterministic `secrets` and `code_safety`
//!   verdicts therefore stay authoritative in both directions;
//! - a failure finding below [`MIN_FINDING_CONFIDENCE`] is dropped with a warning;
//! - when the deterministic verdict is not `approved` the stage is not run at all
//!   (Python's gate does the same), so a rejected or human-review package never
//!   reaches a model and can never be upgraded.
//!
//! After the merge the verdict is re-derived with the deterministic three-state
//! rule and [`ReviewResult::review_id`](crate::flow::review::ReviewResult::review_id)
//! is refreshed, so the id always names the merged identity.
//!
//! # Provenance and failure semantics
//!
//! A merged review carries [`REVIEW_MODE_LLM`]; a review the stage did not touch
//! carries [`REVIEW_MODE_DETERMINISTIC`]. An unusable reply (no JSON object, no
//! `checks` array, a non-array `checks`) or a model invocation failure becomes
//! [`Error::Llm`](openjiuwen_common::error::Error::llm) naming this stage under
//! [`FallbackMode::Error`], and under [`FallbackMode::Deterministic`] keeps the
//! deterministic verdict with a `deterministic_fallback:` note — never a silent
//! downgrade.
//!
//! # Request
//!
//! The prompt is one byte-stable rendering of the redacted, truncated package plus
//! the deterministic checks. Artifact text is capped at [`MAX_ARTIFACT_CHARS`] and
//! every string passes through [`redact_private_json`] before it reaches the model,
//! so a recognisable credential never leaves the process.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{validation, Error, Result};
use crate::flow::model::canonical_json;
use crate::flow::pack::SkillPack;
use crate::flow::review::{
    refresh_review_id, resolve_verdict, ReviewResult, CHECK_FAIL, CHECK_PASS,
    REVIEW_MODE_DETERMINISTIC, REVIEW_MODE_LLM, VERDICT_APPROVED,
};
use crate::llm::{
    clamp_unit, complete_json, dedupe_strings, render_prompt as join_prompt, truncate_text,
    CompletionModel, FallbackMode,
};
use crate::redaction::{redact_private_json, redact_sensitive_text};

/// The check names the stage recognizes, in deterministic review order.
pub const REVIEW_CHECK_NAMES: [&str; 10] = [
    "structure",
    "dependencies",
    "code_safety",
    "secrets",
    "artifact_paths",
    "integrity",
    "meta_naming",
    "materials",
    "permissions",
    "license",
];

/// The confidence a model failure finding needs to enter the verdict.
///
/// A reply without a `confidence` field means full confidence (`1.0`), so a model
/// that does not emit confidences is judged on its findings alone.
pub const MIN_FINDING_CONFIDENCE: f64 = 0.5;
/// The longest model reason kept inside a check.
pub const MAX_REASON_CHARS: usize = 500;
/// The longest model note kept.
pub const MAX_NOTE_CHARS: usize = 500;
/// The maximum number of model notes kept.
pub const MAX_NOTES: usize = 8;
/// The maximum number of model reasons appended to one check.
pub const MAX_REASONS_PER_CHECK: usize = 3;
/// The prefix that marks a reason as model-authored.
pub const MODEL_REASON_PREFIX: &str = "model review: ";
/// The longest artifact excerpt sent to the model.
pub const MAX_ARTIFACT_CHARS: usize = 2_000;
/// The maximum number of artifacts sent.
pub const MAX_ARTIFACTS: usize = 8;
/// The longest step description sent.
pub const MAX_STEP_DESCRIPTION_CHARS: usize = 400;
/// The maximum number of steps sent.
pub const MAX_PROMPT_STEPS: usize = 32;
/// The longest free-text package field (`description`, `trigger_conditions`) sent.
pub const MAX_FIELD_CHARS: usize = 1_000;
/// The prefix that marks a note explaining a stage that did not run.
pub const FALLBACK_NOTE_PREFIX: &str = "deterministic_fallback: ";
/// The prefix that marks a note explaining a skipped stage.
pub const SKIPPED_NOTE_PREFIX: &str = "model review skipped: ";
/// The system instruction sent with every review request.
pub const REVIEW_SYSTEM_PROMPT: &str = "\
You are the semantic review stage for an immutable capability package.
A deterministic gate has already validated its structure, dependencies, code safety, \
secrets, artifact paths, integrity, naming, materials, permissions, and license.
Report only concrete contradictions or unsafe instructions that are visible in the package; \
never infer missing facts.
The Proprietary license is explicitly allowed and must not block approval.
Referenced member capabilities intentionally keep credentials, permissions, and runtime \
configuration outside this package, so a service reference alone is not evidence of a \
missing dependency or of unsafe behaviour.
Package timestamps are metadata only and no trusted current time is supplied, so never treat \
content as suspicious for being dated in the past or the future.
You cannot approve a package the deterministic gate failed, and you cannot clear a \
deterministic failure; you can only report additional problems.
Return JSON only: {\"checks\":[{\"check\":\"<name>\",\"result\":\"pass|fail\",\"reason\":\"<one sentence>\",\"confidence\":0.0}],\"notes\":[\"<observation>\"]}.
`check` must be one of: structure, dependencies, code_safety, secrets, artifact_paths, \
integrity, meta_naming, materials, permissions, license.
Report only checks where you have a finding; omit the rest. `confidence` is optional and \
defaults to 1.0.";

/// One finding the model reported for one deterministic check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LlmReviewFinding {
    /// The check name; always one of [`REVIEW_CHECK_NAMES`].
    pub check: String,
    /// [`CHECK_PASS`] or [`CHECK_FAIL`].
    pub result: String,
    /// The model's redacted, length-clamped reason.
    pub reason: String,
    /// The model's confidence, clamped to `[0, 1]`; absent in a reply means `1.0`.
    pub confidence: f64,
}

impl Default for LlmReviewFinding {
    fn default() -> Self {
        Self {
            check: String::new(),
            result: CHECK_PASS.to_string(),
            reason: String::new(),
            confidence: 1.0,
        }
    }
}

/// The model-driven semantic reviewer over an approved package.
///
/// `model: None` ([`LlmPackageReviewer::without_model`]) keeps the stage off and
/// answers the deterministic result unchanged, so a caller can wire the reviewer
/// unconditionally.
#[derive(Clone, Default)]
pub struct LlmPackageReviewer {
    /// The completion model; `None` never calls anything.
    model: Option<Arc<dyn CompletionModel>>,
    /// What to do when the model fails or answers unusably.
    fallback: FallbackMode,
}

impl LlmPackageReviewer {
    /// Builds a reviewer over `model` with [`FallbackMode::Deterministic`].
    #[must_use]
    pub fn new(model: Arc<dyn CompletionModel>) -> Self {
        Self {
            model: Some(model),
            fallback: FallbackMode::Deterministic,
        }
    }

    /// Builds a reviewer with no model: the stage is off and never called.
    #[must_use]
    pub fn without_model() -> Self {
        Self::default()
    }

    /// Sets the fallback mode.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }

    /// The configured fallback mode.
    #[must_use]
    pub fn fallback(&self) -> FallbackMode {
        self.fallback
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Reviews `pack`, merging the model's findings into `deterministic`.
    ///
    /// Returns `deterministic` unchanged (with [`REVIEW_MODE_DETERMINISTIC`]) when no
    /// model is configured or when the deterministic verdict is not
    /// [`VERDICT_APPROVED`]. A merged result carries [`REVIEW_MODE_LLM`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::validation) when
    /// `deterministic` belongs to another package or is already an LLM review, and
    /// the model-stage [`Error::Llm`](openjiuwen_common::error::Error::llm) under
    /// [`FallbackMode::Error`].
    pub async fn review(
        &self,
        pack: &SkillPack,
        deterministic: ReviewResult,
    ) -> Result<ReviewResult> {
        if deterministic.package_id != pack.package_id {
            return Err(validation(
                "review_result",
                format!(
                    "reviewed package `{}` does not match package `{}`",
                    deterministic.package_id, pack.package_id
                ),
            ));
        }
        if deterministic.review_mode != REVIEW_MODE_DETERMINISTIC {
            return Err(validation(
                "review_result",
                format!(
                    "expected a deterministic review, got a `{}` review",
                    deterministic.review_mode
                ),
            ));
        }
        let Some(model) = self.model.as_deref() else {
            return Ok(deterministic);
        };
        if deterministic.verdict != VERDICT_APPROVED {
            let mut result = deterministic;
            result.notes.push(format!(
                "{SKIPPED_NOTE_PREFIX}the deterministic verdict is `{}`",
                result.verdict
            ));
            return Ok(result);
        }
        let system = REVIEW_SYSTEM_PROMPT;
        let payload = render_user_payload(pack, &deterministic);
        match self
            .attempt(model, system, &payload, deterministic.clone())
            .await
        {
            Ok(result) => Ok(result),
            Err(error) => match self.fallback {
                FallbackMode::Error => Err(error),
                FallbackMode::Deterministic => {
                    tracing::warn!(
                        stage = "llm package review",
                        model = model.name(),
                        %error,
                        "model review failed; keeping the deterministic verdict"
                    );
                    let mut result = deterministic;
                    result.notes.push(format!(
                        "{FALLBACK_NOTE_PREFIX}{}",
                        truncate_text(&error.message(), MAX_NOTE_CHARS)
                    ));
                    Ok(result)
                }
            },
        }
    }

    /// Runs the deterministic gate and then [`LlmPackageReviewer::review`].
    ///
    /// # Errors
    ///
    /// Same as [`LlmPackageReviewer::review`].
    pub async fn review_pack(&self, pack: &SkillPack) -> Result<ReviewResult> {
        let deterministic = crate::flow::review::Reviewer::new().review(pack);
        self.review(pack, deterministic).await
    }

    /// Renders the exact prompt the model receives for `pack` and `deterministic`.
    #[must_use]
    pub fn render_prompt(pack: &SkillPack, deterministic: &ReviewResult) -> String {
        join_prompt(
            REVIEW_SYSTEM_PROMPT,
            &render_user_payload(pack, deterministic),
        )
    }

    /// Calls the model once and merges a usable reply; every failure is `Err`.
    async fn attempt(
        &self,
        model: &dyn CompletionModel,
        system: &str,
        payload: &str,
        deterministic: ReviewResult,
    ) -> Result<ReviewResult> {
        let reply = complete_json(Some(model), system, payload).await?;
        let parsed = parse_reply(reply.as_ref()).map_err(|problem| {
            Error::llm(format!(
                "llm package review: unusable model reply: {problem}"
            ))
        })?;
        Ok(merge_findings(deterministic, &parsed))
    }
}

/// Builds the canonical, redacted, truncated user payload.
fn render_user_payload(pack: &SkillPack, deterministic: &ReviewResult) -> String {
    let artifacts: BTreeMap<&str, &str> = pack
        .artifacts
        .iter()
        .take(MAX_ARTIFACTS)
        .map(|(name, text)| (name.as_str(), truncate_text(text, MAX_ARTIFACT_CHARS)))
        .collect();
    let steps: Vec<Value> = pack
        .steps
        .iter()
        .take(MAX_PROMPT_STEPS)
        .map(|step| {
            json!({
                "id": step.id,
                "name": step.name,
                "description": truncate_text(&step.description, MAX_STEP_DESCRIPTION_CHARS),
                "capability_type": step.capability_type,
                "depends_on": step.depends_on,
                "inputs": step.inputs,
                "outputs": step.outputs,
            })
        })
        .collect();
    let checks: Vec<Value> = deterministic
        .checks
        .iter()
        .map(|check| {
            json!({
                "check": check.check,
                "result": check.result,
                "reasons": check.reasons,
            })
        })
        .collect();
    let payload = json!({
        "package": {
            "package_id": pack.package_id,
            "recipe_id": pack.recipe_id,
            "recipe_version": pack.recipe_version,
            "target_kind": pack.target_kind,
            "name": pack.name,
            "description": truncate_text(&pack.description, MAX_FIELD_CHARS),
            "trigger_conditions": truncate_text(&pack.trigger_conditions, MAX_FIELD_CHARS),
            "permissions": pack.permissions,
            "license": pack.license,
            "materials_hash": deterministic.materials_hash,
            "steps_total": pack.steps.len(),
            "steps_truncated": pack.steps.len() > steps.len(),
            "steps": steps,
            "artifacts_total": pack.artifacts.len(),
            "artifacts_truncated": pack.artifacts.len() > artifacts.len(),
            "artifacts": artifacts,
        },
        "deterministic_checks": checks,
    });
    canonical_json(&redact_private_json(&payload, &[]))
}

impl fmt::Debug for LlmPackageReviewer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmPackageReviewer")
            .field("model", &self.model.as_deref().map(CompletionModel::name))
            .field("fallback", &self.fallback)
            .finish()
    }
}

/// One parsed model reply: the findings that survived validation plus its notes.
#[derive(Debug, Clone, Default, PartialEq)]
struct ParsedReply {
    /// The validated findings, in reply order.
    findings: Vec<LlmReviewFinding>,
    /// The redacted, clamped notes.
    notes: Vec<String>,
}

/// Parses one model reply, or explains why it is unusable.
///
/// Unknown check names, non-object items and non-`pass`/`fail` results are dropped
/// with a warning naming the reply item; only a reply that carries no `checks` array
/// at all is unusable.
fn parse_reply(reply: Option<&Value>) -> std::result::Result<ParsedReply, String> {
    let Some(value) = reply else {
        return Err("the reply contained no JSON object".to_string());
    };
    let Some(items) = value.get("checks") else {
        return Err("the reply has no `checks` array".to_string());
    };
    let Some(items) = items.as_array() else {
        return Err("`checks` must be an array".to_string());
    };
    let mut findings = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let Some(object) = item.as_object() else {
            tracing::warn!(
                index,
                "llm package review: dropping a reply item that is not an object"
            );
            continue;
        };
        let Some(name) = object.get("check").and_then(Value::as_str).map(str::trim) else {
            tracing::warn!(
                index,
                "llm package review: dropping a reply item with no `check`"
            );
            continue;
        };
        if !REVIEW_CHECK_NAMES.contains(&name) {
            tracing::warn!(
                index,
                check = name,
                "llm package review: dropping an unknown check name"
            );
            continue;
        }
        let Some(result) = object.get("result").and_then(Value::as_str) else {
            tracing::warn!(
                index,
                check = name,
                "llm package review: dropping a reply item with no `result`"
            );
            continue;
        };
        let result = result.trim().to_ascii_lowercase();
        if result != CHECK_PASS && result != CHECK_FAIL {
            tracing::warn!(
                index,
                check = name,
                result = result.as_str(),
                "llm package review: dropping a reply item with an unknown `result`"
            );
            continue;
        }
        let confidence = match object.get("confidence") {
            None | Some(Value::Null) => 1.0,
            Some(value) => {
                let parsed = value.as_f64();
                if parsed.is_none() {
                    tracing::warn!(
                        index,
                        check = name,
                        "llm package review: `confidence` is not a number; assuming full confidence"
                    );
                }
                parsed.map_or(1.0, clamp_unit)
            }
        };
        findings.push(LlmReviewFinding {
            check: name.to_string(),
            result,
            reason: clamp_reason(
                object.get("reason").and_then(Value::as_str).map(str::trim),
                name,
            ),
            confidence,
        });
    }
    Ok(ParsedReply {
        findings,
        notes: parse_notes(value.get("notes")),
    })
}

/// Redacts and length-clamps one model reason, defaulting a blank one.
fn clamp_reason(raw: Option<&str>, check: &str) -> String {
    let raw = raw.unwrap_or_default();
    if raw.trim().is_empty() {
        return format!("the model reported an issue for `{check}` without a reason");
    }
    let redacted = redact_sensitive_text(raw.trim());
    truncate_text(&redacted, MAX_REASON_CHARS).to_string()
}

/// Parses the reply's `notes` array: redacted, clamped, deduplicated, capped.
fn parse_notes(value: Option<&Value>) -> Vec<String> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut notes: Vec<String> = Vec::new();
    for item in items {
        let Some(text) = item.as_str().map(str::trim).filter(|text| !text.is_empty()) else {
            continue;
        };
        notes.push(truncate_text(&redact_sensitive_text(text), MAX_NOTE_CHARS).to_string());
    }
    let mut notes = dedupe_strings(notes);
    notes.truncate(MAX_NOTES);
    notes
}

/// Merges validated findings into a deterministic result, fail-closed.
///
/// Only [`CHECK_FAIL`] findings at or above [`MIN_FINDING_CONFIDENCE`] are merged;
/// every other finding (including a `pass`) leaves the deterministic check exactly
/// as it was.
fn merge_findings(mut deterministic: ReviewResult, parsed: &ParsedReply) -> ReviewResult {
    for finding in &parsed.findings {
        if finding.result != CHECK_FAIL {
            continue;
        }
        if finding.confidence < MIN_FINDING_CONFIDENCE {
            tracing::warn!(
                check = finding.check.as_str(),
                confidence = finding.confidence,
                "llm package review: dropping a low-confidence failure finding"
            );
            continue;
        }
        let Some(target) = deterministic
            .checks
            .iter_mut()
            .find(|check| check.check == finding.check)
        else {
            continue;
        };
        if target
            .reasons
            .iter()
            .filter(|reason| reason.starts_with(MODEL_REASON_PREFIX))
            .count()
            >= MAX_REASONS_PER_CHECK
        {
            tracing::warn!(
                check = finding.check.as_str(),
                "llm package review: dropping a finding beyond the per-check cap"
            );
            continue;
        }
        let reason = format!("{MODEL_REASON_PREFIX}{}", finding.reason);
        if target.reasons.contains(&reason) {
            continue;
        }
        target.reasons.push(reason);
        target.result = CHECK_FAIL.to_string();
    }
    deterministic.verdict = resolve_verdict(&deterministic.checks);
    deterministic.review_mode = REVIEW_MODE_LLM.to_string();
    deterministic.notes.clone_from(&parsed.notes);
    refresh_review_id(&mut deterministic);
    deterministic
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::flow::pack::{SkillPack, SkillPackStep};
    use crate::flow::review::Reviewer;

    fn demo_pack(name: &str) -> SkillPack {
        let mut pack = SkillPack {
            recipe_id: "recipe_0123456789ab".to_string(),
            name: name.to_string(),
            description: "[技能包] demo".to_string(),
            steps: vec![
                SkillPackStep {
                    id: "a".to_string(),
                    name: "a".to_string(),
                    description: "first".to_string(),
                    capability_type: "skill".to_string(),
                    ..SkillPackStep::default()
                },
                SkillPackStep {
                    id: "b".to_string(),
                    name: "b".to_string(),
                    description: "second".to_string(),
                    capability_type: "skill".to_string(),
                    depends_on: vec!["a".to_string()],
                    ..SkillPackStep::default()
                },
            ],
            license: "Proprietary".to_string(),
            ..SkillPack::default()
        };
        pack.recompute_identity();
        pack
    }

    #[test]
    fn parse_reply_accepts_findings_and_notes() {
        let reply = json!({
            "checks": [
                {"check": "materials", "result": "FAIL", "reason": "the description is vague"},
                {"check": "permissions", "result": "pass", "confidence": 2.0},
            ],
            "notes": ["looks fine", "looks fine", "  "],
        });
        let parsed = parse_reply(Some(&reply)).expect("usable");
        assert_eq!(parsed.findings.len(), 2);
        assert_eq!(parsed.findings[0].check, "materials");
        assert_eq!(parsed.findings[0].result, CHECK_FAIL);
        assert_eq!(parsed.findings[0].reason, "the description is vague");
        assert_eq!(parsed.findings[0].confidence, 1.0);
        assert_eq!(parsed.findings[1].confidence, 1.0);
        assert_eq!(parsed.notes, vec!["looks fine".to_string()]);
    }

    #[test]
    fn parse_reply_drops_items_that_are_not_findings() {
        let reply = json!({
            "checks": [
                "nonsense",
                {"result": "fail", "reason": "no name"},
                {"check": "quality", "result": "fail", "reason": "unknown check"},
                {"check": "materials", "result": "maybe", "reason": "bad result"},
                {"check": "materials", "result": "fail", "confidence": "high"},
                {"check": "license", "result": "fail"},
            ],
        });
        let parsed = parse_reply(Some(&reply)).expect("usable");
        assert_eq!(parsed.findings.len(), 2);
        assert_eq!(parsed.findings[0].check, "materials");
        assert_eq!(parsed.findings[0].confidence, 1.0);
        assert!(parsed.findings[0].reason.contains("without a reason"));
        assert_eq!(parsed.findings[1].check, "license");
        assert!(parsed.findings[1].reason.contains("without a reason"));
    }

    #[test]
    fn parse_reply_rejects_only_a_missing_checks_array() {
        assert!(parse_reply(None).is_err());
        assert!(parse_reply(Some(&json!({"notes": []}))).is_err());
        assert!(parse_reply(Some(&json!({"checks": {}}))).is_err());
        assert!(parse_reply(Some(&json!({"checks": []}))).is_ok());
    }

    #[test]
    fn reasons_are_redacted_and_clamped() {
        let reason = format!("Bearer abcdefgh12345678 {}", "x".repeat(2_000));
        let clamped = clamp_reason(Some(&reason), "secrets");
        assert!(!clamped.contains("abcdefgh12345678"));
        assert!(clamped.contains("<redacted"));
        assert_eq!(clamped.chars().count(), MAX_REASON_CHARS);
        assert_eq!(
            clamp_reason(Some(" héllo "), "materials"),
            "héllo".to_string()
        );
    }

    #[test]
    fn merge_is_fail_closed() {
        let pack = demo_pack("demo-pack");
        let deterministic = Reviewer::new().review(&pack);
        assert_eq!(deterministic.verdict, VERDICT_APPROVED);

        // A model `pass` never clears a deterministic failure.
        let mut failing = deterministic.clone();
        for check in &mut failing.checks {
            if check.check == "license" {
                check.result = CHECK_FAIL.to_string();
                check.reasons.push("license is required".to_string());
            }
        }
        let passing_reply = ParsedReply {
            findings: vec![LlmReviewFinding {
                check: "license".to_string(),
                result: CHECK_PASS.to_string(),
                reason: "looks fine".to_string(),
                confidence: 1.0,
            }],
            notes: Vec::new(),
        };
        let merged = merge_findings(failing, &passing_reply);
        assert_eq!(merged.verdict, crate::flow::review::VERDICT_REJECTED);
        let license = merged
            .checks
            .iter()
            .find(|check| check.check == "license")
            .expect("license check");
        assert_eq!(license.result, CHECK_FAIL);
        assert_eq!(license.reasons, vec!["license is required".to_string()]);

        // A model `fail` adds a finding and re-derives the verdict.
        let fail_reply = ParsedReply {
            findings: vec![failure_finding(
                "materials",
                "the steps contradict each other",
            )],
            notes: vec!["checked by hand".to_string()],
        };
        let merged = merge_findings(deterministic.clone(), &fail_reply);
        assert_eq!(merged.review_mode, REVIEW_MODE_LLM);
        assert_eq!(merged.verdict, crate::flow::review::VERDICT_REJECTED);
        let materials = merged
            .checks
            .iter()
            .find(|check| check.check == "materials")
            .expect("materials check");
        assert_eq!(materials.result, CHECK_FAIL);
        assert_eq!(
            materials.reasons,
            vec![format!(
                "{MODEL_REASON_PREFIX}the steps contradict each other"
            )]
        );
        assert_eq!(merged.notes, vec!["checked by hand".to_string()]);
        assert_ne!(merged.review_id, deterministic.review_id);
    }

    #[test]
    fn low_confidence_and_excess_findings_are_dropped() {
        let pack = demo_pack("demo-pack");
        let deterministic = Reviewer::new().review(&pack);
        let reply = ParsedReply {
            findings: vec![
                LlmReviewFinding {
                    check: "materials".to_string(),
                    result: CHECK_FAIL.to_string(),
                    reason: "unsure".to_string(),
                    confidence: 0.4,
                },
                LlmReviewFinding {
                    check: "permissions".to_string(),
                    result: CHECK_FAIL.to_string(),
                    reason: "first".to_string(),
                    confidence: 1.0,
                },
                LlmReviewFinding {
                    check: "permissions".to_string(),
                    result: CHECK_FAIL.to_string(),
                    reason: "second".to_string(),
                    confidence: 1.0,
                },
            ],
            notes: Vec::new(),
        };
        let merged = merge_findings(deterministic, &reply);
        let materials = merged
            .checks
            .iter()
            .find(|check| check.check == "materials")
            .expect("materials check");
        assert_eq!(materials.result, CHECK_PASS);
        let permissions = merged
            .checks
            .iter()
            .find(|check| check.check == "permissions")
            .expect("permissions check");
        assert_eq!(permissions.reasons.len(), 2);
    }

    #[test]
    fn the_rendered_prompt_is_byte_stable_and_redacted() {
        let mut pack = demo_pack("demo-pack");
        pack.steps[0].description = "search with api_key: sk-live-0123456789abcdef".to_string();
        pack.recompute_identity();
        let reviewer = LlmPackageReviewer::without_model();
        let deterministic = Reviewer::new().review(&pack);
        let rendered = LlmPackageReviewer::render_prompt(&pack, &deterministic);
        assert_eq!(
            rendered,
            LlmPackageReviewer::render_prompt(&pack, &deterministic)
        );
        assert!(!rendered.contains("sk-live-0123456789abcdef"));
        assert!(rendered.contains("<redacted>"));
        assert!(rendered.starts_with(REVIEW_SYSTEM_PROMPT));
        assert!(rendered.contains("\"deterministic_checks\""));
        assert!(rendered.contains("\"secrets\""));
        assert!(!reviewer.has_model());
        assert_eq!(reviewer.fallback(), FallbackMode::Deterministic);
    }

    /// Builds a failure finding for `check` with a full-confidence reason.
    fn failure_finding(check: &str, reason: &str) -> LlmReviewFinding {
        LlmReviewFinding {
            check: check.to_string(),
            result: CHECK_FAIL.to_string(),
            reason: reason.to_string(),
            confidence: 1.0,
        }
    }
}
