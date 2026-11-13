//! Deterministic judge scoring.
//!
//! Ports the scoring half of Python's
//! `harness_rsi/evaluator/judger/scoring.py` — the part that does not need a
//! model: verdict parsing, the weighted-behaviour mean, forbidden-behaviour
//! penalties, the pass threshold, and the failure classification used for
//! retries. Judge *models* stay out of scope (a guarded judge `DeepAgent`
//! belongs to the harness crate); this module only owns what happens once a
//! verdict document exists.
//!
//! Scoring contract ([`score_verdict`]):
//!
//! ```text
//! base    = Σ(behaviour.score · weight) / Σ(weight)   over the contract's required behaviours
//! penalty = Σ(spec.penalty)                           over the triggered forbidden behaviours
//! ceiling → score = min(base, 1.0 - penalty)
//! subtract → score = max(0.0, base - penalty)
//! passed  = score >= success_threshold
//! ```
//!
//! A required behaviour with no verdict contributes `0.0`, so a judge that
//! skips a criterion can never score higher than one that grades it. Python's
//! legacy `ceiling` mode caps with the *largest* single penalty; the port sums
//! the triggered penalties instead (documented deviation), which keeps the
//! ceiling and subtract modes on one penalty rule.
//!
//! Verdict parsing rejects what Python rejects: non-finite numbers, duplicate
//! behaviour ids, a non-object root, and a status other than `completed`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::case::{BehaviorSpec, CaseReference, ForbiddenBehaviorSpec, PenaltyMode};

/// Python's default pass threshold for a judge score.
pub const DEFAULT_SUCCESS_THRESHOLD: f64 = 0.8;

/// Python's default forbidden-behaviour penalty.
pub const DEFAULT_FORBIDDEN_PENALTY: f64 = 0.3;

/// The `status` value a usable verdict must carry.
pub const VERDICT_COMPLETED: &str = "completed";

/// The `method` recorded on a verdict-graded result.
pub const METHOD_JUDGE: &str = "llm_as_judge";

/// The `method` recorded on a single-score response.
pub const METHOD_RESPONSE_SCORE: &str = "response_score";

/// Matches a fenced JSON block, optionally tagged `json`.
///
/// Kept as an `Option` so a regex-engine failure degrades to "no fence found"
/// instead of panicking; the pattern is a constant, so in practice it is `Some`.
static FENCE: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?ims)^```(?:json)?[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|$)").ok());

/// One required behaviour as judged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BehaviorVerdict {
    /// The behaviour id; matched against [`BehaviorSpec::id`].
    pub id: String,
    /// The judged score, `0.0..=1.0`.
    pub score: f64,
    /// The judge's rationale.
    pub reason: String,
    /// The concrete evidence the judge relied on.
    pub evidence: String,
}

/// One forbidden behaviour as judged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ForbiddenHit {
    /// The forbidden behaviour id.
    pub id: String,
    /// Whether the behaviour was observed.
    pub triggered: bool,
    /// The judge's rationale.
    pub reason: String,
    /// The concrete evidence the judge relied on.
    pub evidence: String,
}

/// A parsed judge verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JudgeVerdict {
    /// The verdict status; only `completed` is usable.
    pub status: String,
    /// The judge's overall rationale.
    pub overall_reason: String,
    /// One entry per required behaviour.
    pub behaviors: Vec<BehaviorVerdict>,
    /// One entry per forbidden behaviour.
    pub forbidden_hits: Vec<ForbiddenHit>,
}

impl Default for JudgeVerdict {
    /// Returns a completed, empty verdict (no behaviours, no hits).
    fn default() -> Self {
        Self {
            status: VERDICT_COMPLETED.to_owned(),
            overall_reason: String::new(),
            behaviors: Vec::new(),
            forbidden_hits: Vec::new(),
        }
    }
}

/// The scored outcome of a judge verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JudgeResult {
    /// The grading method (`llm_as_judge`, `response_score`, …).
    pub method: String,
    /// The score handed back to the caller (`0.0..=1.0`).
    pub score: f64,
    /// Whether the score cleared the contract threshold.
    pub passed: bool,
    /// The human-readable rationale.
    pub reason: String,
    /// Diagnostics: raw score, base score, penalty, triggered behaviours, …
    pub metadata: BTreeMap<String, Value>,
}

/// The trusted, dataset-derived scoring contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScoringContract {
    /// Positively weighted behaviours.
    pub required: Vec<BehaviorSpec>,
    /// Behaviours that deduct when triggered.
    pub forbidden: Vec<ForbiddenBehaviorSpec>,
    /// How the deduction applies.
    pub penalty_mode: PenaltyMode,
    /// The score at or above which the case passes.
    pub success_threshold: f64,
    /// Whether the returned score collapses to `1.0`/`0.0`.
    pub binary: bool,
}

impl Default for ScoringContract {
    /// Returns an empty ceiling-mode contract with Python's `0.8` threshold.
    fn default() -> Self {
        Self {
            required: Vec::new(),
            forbidden: Vec::new(),
            penalty_mode: PenaltyMode::Ceiling,
            success_threshold: DEFAULT_SUCCESS_THRESHOLD,
            binary: false,
        }
    }
}

/// One per-requirement reporting row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RequirementResult {
    /// The behaviour id.
    pub id: String,
    /// The weight from the contract (`0.0` for forbidden behaviours, whose
    /// contract field is a penalty, not a weight).
    pub weight: f64,
    /// `0.0..=1.0`: the judged score, or `1.0`/`0.0` for an untriggered/triggered
    /// forbidden behaviour.
    pub score: f64,
    /// Whether the requirement was fully met.
    pub met: bool,
}

/// Why raw judge text could not be turned into a usable verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgeFailure {
    /// The payload was cut off (an unterminated fence or unbalanced brackets).
    Truncated,
    /// A repair attempt returned a schema-invalid payload.
    RepairInvalid,
    /// A repair attempt was itself unavailable.
    RepairUnavailable,
    /// The judge declared itself unavailable.
    Unavailable,
    /// The payload is complete but does not match the verdict schema.
    InvalidSchema,
}

impl JudgeFailure {
    /// Returns the stable lowercase tag recorded in diagnostics.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Truncated => "truncated",
            Self::RepairInvalid => "repair_invalid",
            Self::RepairUnavailable => "repair_unavailable",
            Self::Unavailable => "unavailable",
            Self::InvalidSchema => "invalid_schema",
        }
    }
}

/// Parses raw judge text into a verdict.
///
/// Accepts a bare verdict document, a verdict wrapped in prose, or a ```` ```json ````
/// fenced block. When several structured payloads are present the parse fails
/// instead of guessing which one the judge meant.
///
/// # Errors
///
/// Returns [`Error::Validation`] (Python's `ValueError` kinds preserved in the
/// message) when the text holds no verdict, more than one payload, a non-object
/// root, non-finite numbers, duplicate behaviour ids, or a status other than
/// `completed`.
pub fn parse_judge_verdict(text: &str) -> Result<JudgeVerdict> {
    let value = parse_judge_json(text)?;
    judge_verdict_from_value(&value)
}

/// Parses a verdict from an already-decoded JSON value.
///
/// `status` defaults to `completed`; `behaviors` is required; `forbidden_hits`
/// defaults to empty.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a non-object root, an unknown status, a
/// malformed behaviour/hit entry, a score outside `0.0..=1.0`, or duplicate ids.
pub(crate) fn judge_verdict_from_value(value: &Value) -> Result<JudgeVerdict> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::validation("judge output must be an object"))?;
    let status = match object.get("status").filter(|value| !value.is_null()) {
        None => VERDICT_COMPLETED.to_owned(),
        Some(value) => value
            .as_str()
            .ok_or_else(|| Error::validation("judge status must be a string"))?
            .to_owned(),
    };
    if status != VERDICT_COMPLETED {
        return Err(Error::validation(format!(
            "judge status must be `{VERDICT_COMPLETED}`, got `{status}`"
        )));
    }
    let overall_reason = match object
        .get("overall_reason")
        .filter(|value| !value.is_null())
    {
        None => String::new(),
        Some(value) => value
            .as_str()
            .ok_or_else(|| Error::validation("judge overall_reason must be a string"))?
            .to_owned(),
    };
    let behaviors = parse_behaviors(object.get("behaviors"))?;
    let forbidden_hits = parse_forbidden_hits(object.get("forbidden_hits"))?;
    let behavior_ids = behaviors.iter().map(|item| item.id.as_str());
    reject_duplicate_ids(behavior_ids)?;
    let hit_ids = forbidden_hits.iter().map(|item| item.id.as_str());
    reject_duplicate_ids(hit_ids)?;
    Ok(JudgeVerdict {
        status,
        overall_reason,
        behaviors,
        forbidden_hits,
    })
}

/// Parses the required `behaviors` list.
fn parse_behaviors(value: Option<&Value>) -> Result<Vec<BehaviorVerdict>> {
    let value = value
        .filter(|value| !value.is_null())
        .ok_or_else(|| Error::validation("judge output must contain `behaviors`"))?;
    let items = value
        .as_array()
        .ok_or_else(|| Error::validation("judge behaviors must be a list of objects"))?;
    let mut behaviors = Vec::with_capacity(items.len());
    for item in items {
        let object = item
            .as_object()
            .ok_or_else(|| Error::validation("judge behaviors must be a list of objects"))?;
        behaviors.push(BehaviorVerdict {
            id: required_id(object, "judge behavior")?,
            score: bounded_number(object.get("score"), "score")?,
            reason: optional_text(object.get("reason")),
            evidence: optional_text(object.get("evidence")),
        });
    }
    Ok(behaviors)
}

/// Parses the optional `forbidden_hits` list.
fn parse_forbidden_hits(value: Option<&Value>) -> Result<Vec<ForbiddenHit>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| Error::validation("judge forbidden_hits must be a list of objects"))?;
    let mut hits = Vec::with_capacity(items.len());
    for item in items {
        let object = item
            .as_object()
            .ok_or_else(|| Error::validation("judge forbidden_hits must be a list of objects"))?;
        hits.push(ForbiddenHit {
            id: required_id(object, "forbidden hit")?,
            triggered: object
                .get("triggered")
                .and_then(Value::as_bool)
                .ok_or_else(|| Error::validation("forbidden triggered must be a boolean"))?,
            reason: optional_text(object.get("reason")),
            evidence: optional_text(object.get("evidence")),
        });
    }
    Ok(hits)
}

/// Reads a required non-empty `id`.
fn required_id(object: &serde_json::Map<String, Value>, label: &str) -> Result<String> {
    object
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| Error::validation(format!("{label} id must be a non-empty string")))
}

/// Reads an optional text field, defaulting to empty.
fn optional_text(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// Reads a required finite `0.0..=1.0` number.
fn bounded_number(value: Option<&Value>, name: &str) -> Result<f64> {
    let number = value
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::validation(format!("{name} must be a number")))?;
    if !number.is_finite() || !(0.0..=1.0).contains(&number) {
        return Err(Error::validation(format!(
            "{name} must be finite and between 0 and 1"
        )));
    }
    Ok(number)
}

/// Rejects repeated ids inside one verdict list.
fn reject_duplicate_ids<'a>(ids: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(Error::validation("behavior IDs must be unique"));
        }
    }
    Ok(())
}

/// Decodes the JSON verdict document from raw text.
fn parse_judge_json(raw: &str) -> Result<Value> {
    let text = raw.trim();
    if text.is_empty() {
        return Err(Error::validation("judge output is empty"));
    }
    reject_non_finite(text)?;
    let text = extract_fenced_payload(text)?;
    if text.starts_with('{') || text.starts_with('[') {
        return serde_json::from_str(&text)
            .map_err(|error| Error::validation(format!("invalid judge JSON: {error}")));
    }
    let candidates = payload_objects(&text);
    match candidates.len() {
        1 => Ok(candidates.into_iter().next().unwrap_or(Value::Null)),
        0 => Err(Error::validation("judge output contains no JSON verdict")),
        _ => Err(Error::validation(
            "ambiguous judge output: more than one structured payload",
        )),
    }
}

/// Unwraps one fenced JSON block, matching Python's fence rule.
fn extract_fenced_payload(text: &str) -> Result<String> {
    if text.starts_with('{') || text.starts_with('[') || !text.contains("```") {
        return Ok(text.to_owned());
    }
    let Some(fence) = FENCE.as_ref() else {
        return Err(Error::validation(
            "judge output must contain one complete JSON fence",
        ));
    };
    let captures = fence
        .captures(text)
        .ok_or_else(|| Error::validation("judge output must contain one complete JSON fence"))?;
    let whole = captures.get(0).map_or(0..0, |value| value.range());
    let body = captures
        .get(1)
        .map_or(String::new(), |value| value.as_str().trim().to_owned());
    let mut outside = String::with_capacity(text.len());
    outside.push_str(&text[..whole.start]);
    outside.push_str(&text[whole.end..]);
    if outside.contains("```") || !payload_objects(&outside).is_empty() {
        return Err(Error::validation(
            "ambiguous judge output: more than one structured payload",
        ));
    }
    Ok(body)
}

/// Collects every decoded JSON object that looks like a judge payload.
fn payload_objects(text: &str) -> Vec<Value> {
    let mut found = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = text[cursor..].find('{') {
        let start = cursor + relative;
        let mut stream = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>();
        if let Some(Ok(value)) = stream.next() {
            let end = start + stream.byte_offset();
            if contains_judge_payload(&value) {
                found.push(value);
            }
            cursor = end;
        } else {
            cursor = start + 1;
        }
    }
    found
}

/// Whether a decoded value carries the keys a verdict payload uses.
fn contains_judge_payload(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let verdict_key = ["overall_reason", "behaviors", "forbidden_hits", "score"]
        .iter()
        .any(|key| object.contains_key(*key));
    let unavailable = object.get("status").and_then(Value::as_str) == Some("unavailable");
    verdict_key || unavailable
}

/// Rejects bare `NaN` / `Infinity` JSON constants before decoding.
fn reject_non_finite(text: &str) -> Result<()> {
    let stripped = strip_json_strings(text);
    if let Some(found) = non_finite_token(&stripped) {
        return Err(Error::validation(format!(
            "non-finite judge JSON value: {found}"
        )));
    }
    Ok(())
}

/// Finds a bare `nan`/`infinity` token on word boundaries, case-insensitively.
fn non_finite_token(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    for token in ["nan", "infinity"] {
        let mut from = 0;
        while let Some(found) = lower[from..].find(token) {
            let start = from + found;
            let end = start + token.len();
            let before = lower[..start].chars().next_back();
            let after = lower[end..].chars().next();
            let boundary = |neighbour: Option<char>| {
                neighbour.is_none_or(|ch| !ch.is_alphanumeric() && ch != '_')
            };
            if boundary(before) && boundary(after) {
                return Some(text[start..end].to_owned());
            }
            from = end;
        }
    }
    None
}

/// Removes JSON string contents, keeping the quotes, so token scans ignore text.
fn strip_json_strings(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
                stripped.push(ch);
            }
        } else if ch == '"' {
            in_string = true;
            stripped.push(ch);
        } else {
            stripped.push(ch);
        }
    }
    stripped
}

/// Applies the contract to a parsed verdict.
///
/// Required behaviours the verdict omits contribute `0.0`; forbidden behaviours
/// the verdict omits count as untriggered. Only behaviours declared in the
/// contract can score or penalise, so a judge cannot invent its own weights.
/// An empty required list scores `0.0` because nothing was demonstrated.
#[must_use]
pub fn score_verdict(contract: &ScoringContract, verdict: &JudgeVerdict) -> JudgeResult {
    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for spec in &contract.required {
        let score = verdict
            .behaviors
            .iter()
            .find(|behavior| behavior.id == spec.id)
            .map_or(0.0, |behavior| clamp_unit(behavior.score));
        weighted_sum += score * spec.weight;
        weight_total += spec.weight;
    }
    let base = if weight_total > 0.0 {
        clamp_unit(weighted_sum / weight_total)
    } else {
        0.0
    };

    let mut triggered = Vec::new();
    let mut penalty = 0.0;
    for spec in &contract.forbidden {
        let hit = verdict
            .forbidden_hits
            .iter()
            .any(|hit| hit.id == spec.id && hit.triggered);
        if hit {
            penalty += spec.penalty;
            triggered.push(Value::String(spec.id.clone()));
        }
    }
    penalty = penalty.clamp(0.0, 1.0);

    let raw = match contract.penalty_mode {
        PenaltyMode::Ceiling => clamp_unit(base.min(1.0 - penalty)),
        PenaltyMode::Subtract => clamp_unit(base - penalty),
    };
    let passed = raw >= contract.success_threshold;
    let score = if contract.binary {
        if passed {
            1.0
        } else {
            0.0
        }
    } else {
        raw
    };

    let mut metadata = BTreeMap::new();
    metadata.insert("raw_score".to_owned(), Value::from(raw));
    metadata.insert("base_score".to_owned(), Value::from(base));
    metadata.insert("penalty".to_owned(), Value::from(penalty));
    metadata.insert(
        "penalty_mode".to_owned(),
        Value::String(contract.penalty_mode.as_str().to_owned()),
    );
    metadata.insert(
        "success_threshold".to_owned(),
        Value::from(contract.success_threshold),
    );
    metadata.insert("binary".to_owned(), Value::from(contract.binary));
    metadata.insert(
        "graded_behaviors".to_owned(),
        Value::from(verdict.behaviors.len()),
    );
    metadata.insert(
        "triggered_forbidden_behaviors".to_owned(),
        Value::Array(triggered),
    );
    JudgeResult {
        method: METHOD_JUDGE.to_owned(),
        score,
        passed,
        reason: verdict.overall_reason.clone(),
        metadata,
    }
}

/// Clamps a score into `0.0..=1.0`, mapping a non-finite value to `0.0`.
fn clamp_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Builds the scoring contract a case's reference implies.
///
/// The explicit required behaviours come first; a string [`CaseReference::rubric`]
/// adds one unit-weight `rubric_{index:03d}` behaviour per entry; a present
/// [`CaseReference::answer`] with [`super::case::AnswerRole::Criterion`] inserts
/// a `reference_answer` behaviour at the front (Python's
/// `scoring_contract`). Ids already present are not duplicated, and a
/// non-finite threshold falls back to [`DEFAULT_SUCCESS_THRESHOLD`].
#[must_use]
pub fn verdict_from_case(
    reference: &CaseReference,
    threshold: f64,
    binary: bool,
) -> ScoringContract {
    let mut required = reference.required_behaviors.clone();
    if let Some(items) = reference.rubric.as_array() {
        let mut seen: BTreeSet<String> = required.iter().map(|spec| spec.id.clone()).collect();
        for (index, item) in items.iter().enumerate() {
            let Some(text) = item.as_str().map(str::trim).filter(|text| !text.is_empty()) else {
                continue;
            };
            let id = format!("rubric_{:03}", index + 1);
            if text.is_empty() || !seen.insert(id.clone()) {
                continue;
            }
            required.push(BehaviorSpec::new(id));
        }
    }
    if reference.answer_role == super::case::AnswerRole::Criterion
        && !reference.answer.is_null()
        && !required.iter().any(|spec| spec.id == "reference_answer")
    {
        required.insert(0, BehaviorSpec::new("reference_answer"));
    }
    ScoringContract {
        required,
        forbidden: reference.forbidden_behaviors.clone(),
        penalty_mode: reference.penalty_mode,
        success_threshold: if threshold.is_finite() {
            threshold
        } else {
            DEFAULT_SUCCESS_THRESHOLD
        },
        binary,
    }
}

/// Reports each contract requirement against the verdict, sorted by id.
///
/// Required behaviours carry the contract weight; forbidden behaviours carry
/// `weight = 0.0` (their contract field is a penalty) and score `1.0` when
/// untriggered, `0.0` when triggered.
#[must_use]
pub fn requirement_results(
    contract: &ScoringContract,
    verdict: &JudgeVerdict,
) -> Vec<RequirementResult> {
    let mut results = Vec::with_capacity(contract.required.len() + contract.forbidden.len());
    for spec in &contract.required {
        let score = verdict
            .behaviors
            .iter()
            .find(|behavior| behavior.id == spec.id)
            .map_or(0.0, |behavior| clamp_unit(behavior.score));
        results.push(RequirementResult {
            id: spec.id.clone(),
            weight: spec.weight,
            score,
            met: score >= 1.0,
        });
    }
    for spec in &contract.forbidden {
        let triggered = verdict
            .forbidden_hits
            .iter()
            .any(|hit| hit.id == spec.id && hit.triggered);
        results.push(RequirementResult {
            id: spec.id.clone(),
            weight: 0.0,
            score: if triggered { 0.0 } else { 1.0 },
            met: !triggered,
        });
    }
    results.sort_by(|left, right| left.id.cmp(&right.id));
    results
}

/// Classifies raw judge text that failed to produce a verdict.
///
/// Deterministic heuristics, applied in order:
///
/// 1. text mentioning both `repair` and `unavailable` → [`JudgeFailure::RepairUnavailable`];
/// 2. text mentioning both `repair` and `invalid` → [`JudgeFailure::RepairInvalid`];
/// 3. text declaring itself unavailable (`"status": "unavailable"`, or the word
///    `unavailable`) → [`JudgeFailure::Unavailable`];
/// 4. an unterminated ```` ``` ```` fence or unbalanced brackets outside string
///    literals → [`JudgeFailure::Truncated`];
/// 5. anything else → [`JudgeFailure::InvalidSchema`].
#[must_use]
pub fn classify_judge_failure(text: &str) -> JudgeFailure {
    let lower = text.to_lowercase();
    let mentions_repair = lower.contains("repair");
    if mentions_repair && lower.contains("unavailable") {
        return JudgeFailure::RepairUnavailable;
    }
    if mentions_repair && lower.contains("invalid") {
        return JudgeFailure::RepairInvalid;
    }
    if lower.contains("unavailable") {
        return JudgeFailure::Unavailable;
    }
    if looks_truncated(text) {
        return JudgeFailure::Truncated;
    }
    JudgeFailure::InvalidSchema
}

/// Whether the text looks like a cut-off payload.
fn looks_truncated(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.matches("```").count() % 2 == 1 {
        return true;
    }
    let stripped = strip_json_strings(trimmed);
    if !stripped.starts_with('{') && !stripped.starts_with('[') {
        return false;
    }
    bracket_depth(&stripped) > 0
}

/// Counts unclosed `{`/`[` brackets (negative means more closers).
fn bracket_depth(text: &str) -> i64 {
    let mut depth: i64 = 0;
    for ch in text.chars() {
        match ch {
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            _ => {}
        }
    }
    depth
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_a_complete_verdict() {
        let verdict = parse_judge_verdict(
            r#"{"status":"completed","overall_reason":"ok","behaviors":[{"id":"a","score":1.0,"reason":"r","evidence":"e"}],"forbidden_hits":[{"id":"x","triggered":true,"reason":"r","evidence":"e"}]}"#,
        )
        .expect("verdict parses");
        assert_eq!(verdict.status, VERDICT_COMPLETED);
        assert_eq!(verdict.behaviors.len(), 1);
        assert!(verdict.forbidden_hits[0].triggered);
    }

    #[test]
    fn fenced_and_prose_wrapped_verdicts_parse() {
        let fenced = parse_judge_verdict(
            "Here is the verdict:\n```json\n{\"behaviors\":[{\"id\":\"a\",\"score\":0.5}]}\n```\n",
        )
        .expect("fenced verdict parses");
        assert_eq!(fenced.behaviors[0].score, 0.5);
    }

    #[test]
    fn non_finite_and_duplicate_payloads_are_rejected() {
        assert!(parse_judge_verdict(r#"{"behaviors":[],"score":NaN}"#).is_err());
        assert!(parse_judge_verdict(
            r#"{"behaviors":[{"id":"a","score":1.0},{"id":"a","score":0.0}]}"#
        )
        .is_err());
        assert!(parse_judge_verdict("[1, 2, 3]").is_err());
        assert!(parse_judge_verdict(r#"{"status":"unavailable","behaviors":[]}"#).is_err());
        assert!(parse_judge_verdict("").is_err());
    }

    #[test]
    fn ceiling_and_subtract_penalties_differ() {
        let verdict = parse_judge_verdict(
            r#"{"overall_reason":"r","behaviors":[{"id":"a","score":1.0}],"forbidden_hits":[{"id":"x","triggered":true}]}"#,
        )
        .expect("verdict parses");
        let mut contract = ScoringContract {
            required: vec![BehaviorSpec::new("a")],
            forbidden: vec![ForbiddenBehaviorSpec::new("x")],
            ..ScoringContract::default()
        };
        let ceiling = score_verdict(&contract, &verdict);
        assert!((ceiling.score - 0.7).abs() < 1e-9);
        contract.penalty_mode = PenaltyMode::Subtract;
        let subtract = score_verdict(&contract, &verdict);
        assert!((subtract.score - 0.7).abs() < 1e-9);
        contract.required = vec![BehaviorSpec {
            id: "a".to_owned(),
            weight: 1.0,
        }];
        contract.required.push(BehaviorSpec::new("missing"));
        let missing = score_verdict(&contract, &verdict);
        assert!((missing.score - 0.2).abs() < 1e-9);
        assert!(!missing.passed);
    }

    #[test]
    fn failure_classification_is_deterministic() {
        assert_eq!(
            classify_judge_failure("{\"behaviors\": [{\"id\": \"a\""),
            JudgeFailure::Truncated
        );
        assert_eq!(
            classify_judge_failure("{\"status\":\"unavailable\"}"),
            JudgeFailure::Unavailable
        );
        assert_eq!(
            classify_judge_failure("repair invalid schema"),
            JudgeFailure::RepairInvalid
        );
        assert_eq!(
            classify_judge_failure("repair model unavailable"),
            JudgeFailure::RepairUnavailable
        );
        assert_eq!(
            classify_judge_failure("{\"foo\":1}"),
            JudgeFailure::InvalidSchema
        );
        assert_eq!(JudgeFailure::Truncated.as_str(), "truncated");
    }

    #[test]
    fn requirement_results_are_sorted_and_cover_forbidden() {
        let verdict = parse_judge_verdict(
            r#"{"behaviors":[{"id":"b","score":1.0},{"id":"a","score":0.5}],"forbidden_hits":[{"id":"z","triggered":true}]}"#,
        )
        .expect("verdict parses");
        let contract = ScoringContract {
            required: vec![BehaviorSpec::new("a"), BehaviorSpec::new("b")],
            forbidden: vec![ForbiddenBehaviorSpec::new("z")],
            ..ScoringContract::default()
        };
        let results = requirement_results(&contract, &verdict);
        assert_eq!(
            results
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "z"]
        );
        assert!(!results[0].met);
        assert!(results[1].met);
        assert!(!results[2].met);
        assert!((results[0].weight - 1.0).abs() < f64::EPSILON);
        assert!(results[2].weight.abs() < f64::EPSILON);
    }

    #[test]
    fn contract_expands_rubric_and_reference_answer() {
        let reference = CaseReference {
            answer: json!("42"),
            rubric: json!(["be correct"]),
            ..CaseReference::default()
        };
        let contract = verdict_from_case(&reference, f64::NAN, true);
        assert_eq!(contract.required[0].id, "reference_answer");
        assert_eq!(contract.required[1].id, "rubric_001");
        assert!(contract.binary);
        assert!((contract.success_threshold - DEFAULT_SUCCESS_THRESHOLD).abs() < f64::EPSILON);
    }

    #[test]
    fn binary_mode_keeps_the_raw_score_in_metadata() {
        let verdict = JudgeVerdict {
            behaviors: vec![BehaviorVerdict {
                id: "a".to_owned(),
                score: 0.9,
                ..BehaviorVerdict::default()
            }],
            ..JudgeVerdict::default()
        };
        let contract = ScoringContract {
            required: vec![BehaviorSpec::new("a")],
            binary: true,
            ..ScoringContract::default()
        };
        let result = score_verdict(&contract, &verdict);
        assert!(result.passed);
        assert!((result.score - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.metadata.get("raw_score"), Some(&json!(0.9)));
        assert_eq!(result.method, METHOD_JUDGE);
    }
}
