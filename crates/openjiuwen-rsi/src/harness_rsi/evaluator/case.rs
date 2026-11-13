//! Evaluation case and dataset records.
//!
//! Ports the case side of Python's `harness_rsi/evaluator` (`case_backend.py`,
//! `team_evaluator.py`, and the `data_loader.grading_contract` reference
//! contract) as plain serde data:
//!
//! - [`CaseReference`] is the dataset-neutral grading contract. It carries the
//!   reference answer, the rubric, the explicit required/forbidden behaviours,
//!   and the answer role/penalty mode that decide how a judge verdict scores.
//! - [`SweBenchCase`] is the benchmark half of a case (instance id, patch,
//!   `FAIL_TO_PASS`/`PASS_TO_PASS` lists).
//! - [`EvaluationCase`] is one case: id, task input, public assets, the
//!   reference contract, an optional workspace template, and an optional
//!   SWE-bench binding.
//! - [`Dataset`] is a list of cases loaded from JSON or JSONL.
//!
//! Malformed input is a [`openjiuwen_common::error::Error::Validation`] naming
//! the offending case, never a silent default: a case that cannot be parsed
//! must not quietly grade as a pass.
//!
//! Python's `reference_solution` / `judge_rubrics` top-level aliases (applied
//! by `data_loader.grading_contract.normalize_grading_case`) are **not**
//! expanded here; a caller that needs the weighted-rubric parsing normalises
//! the document first and hands the port a `reference` object.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

/// The dataset id used when a document does not name one.
pub const DEFAULT_DATASET_ID: &str = "dataset";

/// Input-field aliases in precedence order (first present wins).
const INPUT_ALIASES: [&str; 6] = [
    "input",
    "inputs",
    "task_input",
    "query",
    "prompt",
    "question",
];

/// Top-level case keys that are consumed by [`EvaluationCase`] itself.
///
/// Any other key is preserved in [`EvaluationCase::metadata`].
const CASE_KEYS: [&str; 14] = [
    "case_id",
    "id",
    "name",
    "input",
    "inputs",
    "task_input",
    "query",
    "prompt",
    "question",
    "assets",
    "reference",
    "workspace_source_dir",
    "swebench",
    "metadata",
];

/// How the reference *answer* participates in grading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerRole {
    /// The reference answer is one grading criterion among the behaviours.
    #[default]
    Criterion,
    /// The reference answer is context for the judge, not a criterion.
    Reference,
}

impl AnswerRole {
    /// Returns Python's lowercase literal (`"criterion"` / `"reference"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Criterion => "criterion",
            Self::Reference => "reference",
        }
    }

    /// Parses Python's lowercase literal.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for any value other than `criterion` or
    /// `reference`.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "criterion" => Ok(Self::Criterion),
            "reference" => Ok(Self::Reference),
            other => Err(Error::validation(format!(
                "reference.answer_role must be criterion or reference, got `{other}`"
            ))),
        }
    }
}

/// How forbidden behaviours turn into a deduction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PenaltyMode {
    /// `score = min(base, 1.0 - penalty)`: the penalty caps the score.
    #[default]
    Ceiling,
    /// `score = max(0.0, base - penalty)`: the penalty is subtracted.
    Subtract,
}

impl PenaltyMode {
    /// Returns Python's lowercase literal (`"ceiling"` / `"subtract"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ceiling => "ceiling",
            Self::Subtract => "subtract",
        }
    }

    /// Parses Python's lowercase literal.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for any value other than `ceiling` or
    /// `subtract`.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "ceiling" => Ok(Self::Ceiling),
            "subtract" => Ok(Self::Subtract),
            other => Err(Error::validation(format!(
                "reference.penalty_mode must be ceiling or subtract, got `{other}`"
            ))),
        }
    }
}

/// One positively weighted behaviour the judge must score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BehaviorSpec {
    /// Stable behaviour id, matched against [`super::scoring::BehaviorVerdict::id`].
    pub id: String,
    /// Weight in the weighted mean; `0.0` excludes the behaviour.
    pub weight: f64,
}

impl Default for BehaviorSpec {
    /// Returns a behaviour with Python's unit weight.
    fn default() -> Self {
        Self {
            id: String::new(),
            weight: 1.0,
        }
    }
}

impl BehaviorSpec {
    /// Creates a behaviour with Python's unit weight.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            weight: 1.0,
        }
    }
}

/// One forbidden behaviour; triggering it deducts [`ForbiddenBehaviorSpec::penalty`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ForbiddenBehaviorSpec {
    /// Stable behaviour id, matched against [`super::scoring::ForbiddenHit::id`].
    pub id: String,
    /// Deduction applied when the behaviour is triggered (Python default `0.3`).
    pub penalty: f64,
}

impl Default for ForbiddenBehaviorSpec {
    /// Returns a forbidden behaviour with Python's `0.3` penalty.
    fn default() -> Self {
        Self {
            id: String::new(),
            penalty: 0.3,
        }
    }
}

impl ForbiddenBehaviorSpec {
    /// Creates a forbidden behaviour with Python's `0.3` penalty.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            penalty: 0.3,
        }
    }
}

/// The dataset-neutral grading contract of one case.
///
/// Every field has a Python default, so an empty reference grades nothing but
/// still deserialises.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseReference {
    /// The reference answer (`reference.answer`).
    pub answer: Value,
    /// The expected output document (`reference.expected_output`).
    pub expected_output: Value,
    /// The rubric; a string list becomes one behaviour per entry.
    pub rubric: Value,
    /// Input asset files copied into the case workspace.
    pub files: Vec<String>,
    /// Positively weighted behaviours.
    pub required_behaviors: Vec<BehaviorSpec>,
    /// Behaviours that deduct when present.
    pub forbidden_behaviors: Vec<ForbiddenBehaviorSpec>,
    /// Whether [`CaseReference::answer`] is a criterion or context.
    pub answer_role: AnswerRole,
    /// How forbidden behaviours deduct.
    pub penalty_mode: PenaltyMode,
    /// The reference solution document (`reference.solution`).
    pub solution: Value,
    /// Artifact paths the run is expected to produce.
    pub expected_artifacts: Vec<String>,
    /// Deliverables the run is expected to hand back.
    pub deliverables: Vec<String>,
    /// Machine-readable artifact contract.
    pub artifact_contract: Value,
}

impl CaseReference {
    /// Parses a `reference` object.
    ///
    /// A `null` value yields the default (empty) reference. Behaviour entries
    /// may be strings (`id` with unit weight) or objects
    /// (`{"id" | "description", "weight" | "penalty"}`), matching Python's
    /// `_normalize_items`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the reference is not an object, when a
    /// behaviour entry is malformed, when behaviour ids repeat, or when
    /// `answer_role`/`penalty_mode` carry an unknown literal.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        let object = value
            .as_object()
            .ok_or_else(|| Error::validation("reference must be an object"))?;
        let answer_role = match object.get("answer_role").filter(|value| !value.is_null()) {
            None => AnswerRole::default(),
            Some(value) => AnswerRole::parse(
                value
                    .as_str()
                    .ok_or_else(|| Error::validation("reference.answer_role must be a string"))?,
            )?,
        };
        let penalty_mode =
            match object.get("penalty_mode").filter(|value| !value.is_null()) {
                None => PenaltyMode::default(),
                Some(value) => PenaltyMode::parse(value.as_str().ok_or_else(|| {
                    Error::validation("reference.penalty_mode must be a string")
                })?)?,
            };
        Ok(Self {
            answer: object.get("answer").cloned().unwrap_or(Value::Null),
            expected_output: object
                .get("expected_output")
                .cloned()
                .unwrap_or(Value::Null),
            rubric: object.get("rubric").cloned().unwrap_or(Value::Null),
            files: string_list(object.get("files"), "reference.files")?,
            required_behaviors: required_behaviors(object.get("required_behaviors"))?,
            forbidden_behaviors: forbidden_behaviors(object.get("forbidden_behaviors"))?,
            answer_role,
            penalty_mode,
            solution: object.get("solution").cloned().unwrap_or(Value::Null),
            expected_artifacts: string_list(
                object.get("expected_artifacts"),
                "reference.expected_artifacts",
            )?,
            deliverables: string_list(object.get("deliverables"), "reference.deliverables")?,
            artifact_contract: object
                .get("artifact_contract")
                .cloned()
                .unwrap_or(Value::Null),
        })
    }
}

/// The SWE-bench binding of a case.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SweBenchCase {
    /// The benchmark instance id (`repo__owner-123`); required by the evaluator.
    pub instance_id: String,
    /// The upstream repository slug.
    pub repo: Option<String>,
    /// The commit the patch applies to.
    pub base_commit: Option<String>,
    /// The model patch to evaluate.
    pub patch: Option<String>,
    /// The benchmark's test patch; applied by the official harness.
    pub test_patch: Option<String>,
    /// Tests that must go from failing to passing.
    pub fail_to_pass: Vec<String>,
    /// Tests that must stay passing.
    pub pass_to_pass: Vec<String>,
    /// The benchmark split the instance belongs to.
    pub split: Option<String>,
    /// Any other benchmark fields, preserved verbatim.
    pub extra: Value,
}

impl SweBenchCase {
    /// Parses a nested `swebench` object.
    ///
    /// `instance_id` accepts the `instance` / `id` aliases; unknown keys are
    /// preserved in [`SweBenchCase::extra`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not an object, when no
    /// non-empty instance id is present, or when a test list is not a string
    /// list.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| Error::validation("case.swebench must be an object"))?;
        let instance_id = ["instance_id", "instance", "id"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_str))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                Error::validation("case.swebench.instance_id must be a non-empty string")
            })?;
        let known = [
            "instance_id",
            "instance",
            "id",
            "repo",
            "base_commit",
            "patch",
            "test_patch",
            "fail_to_pass",
            "pass_to_pass",
            "split",
        ];
        let mut extra = Map::new();
        for (key, value) in object {
            if !known.contains(&key.as_str()) {
                extra.insert(key.clone(), value.clone());
            }
        }
        Ok(Self {
            instance_id: instance_id.to_owned(),
            repo: optional_string(object.get("repo")),
            base_commit: optional_string(object.get("base_commit")),
            patch: optional_string(object.get("patch")),
            test_patch: optional_string(object.get("test_patch")),
            fail_to_pass: string_list(object.get("fail_to_pass"), "swebench.fail_to_pass")?,
            pass_to_pass: string_list(object.get("pass_to_pass"), "swebench.pass_to_pass")?,
            split: optional_string(object.get("split")),
            extra: Value::Object(extra),
        })
    }
}

/// One evaluation case.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationCase {
    /// Stable case id; generated as `case_{index:03}` when the document omits it.
    pub case_id: String,
    /// Optional display name.
    pub name: Option<String>,
    /// The task input handed to the solver.
    pub input: Value,
    /// Public asset paths staged into the workspace.
    pub assets: Vec<String>,
    /// The grading contract.
    pub reference: CaseReference,
    /// Optional directory copied into the workspace before the run.
    pub workspace_source_dir: Option<PathBuf>,
    /// The SWE-bench binding, when the case is a benchmark instance.
    pub swebench: Option<SweBenchCase>,
    /// Unknown top-level keys plus an explicit `metadata` object.
    pub metadata: BTreeMap<String, Value>,
}

impl EvaluationCase {
    /// Parses one case document.
    ///
    /// `case_id` falls back to `id` and then to `case_{index:03d}`; `input`
    /// takes the first present alias of `input`, `inputs`, `task_input`,
    /// `query`, `prompt`, `question`; unknown top-level keys are preserved in
    /// [`EvaluationCase::metadata`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the case index when the document is
    /// not an object or a nested record is malformed.
    pub fn from_json_value(index: usize, value: &Value) -> Result<Self> {
        parse_case(index, value)
            .map_err(|error| Error::validation(format!("case {index}: {error}")))
    }
}

/// Parses one case document without the index prefix.
fn parse_case(index: usize, value: &Value) -> Result<EvaluationCase> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::validation("case must be a JSON object"))?;
    let case_id = match ["case_id", "id"]
        .iter()
        .find_map(|key| object.get(*key).filter(|value| !value.is_null()))
    {
        None => format!("case_{index:03}"),
        Some(value) => {
            let text = value
                .as_str()
                .ok_or_else(|| Error::validation("case_id must be a string"))?
                .trim();
            if text.is_empty() {
                return Err(Error::validation("case_id must be a non-empty string"));
            }
            text.to_owned()
        }
    };
    let input = INPUT_ALIASES
        .iter()
        .find_map(|key| object.get(*key).filter(|value| !value.is_null()))
        .cloned()
        .unwrap_or(Value::Null);
    let reference = match object.get("reference") {
        None => CaseReference::default(),
        Some(value) => CaseReference::from_json_value(value)?,
    };
    let swebench = match object.get("swebench").filter(|value| !value.is_null()) {
        None => None,
        Some(value) => Some(SweBenchCase::from_json_value(value)?),
    };
    let mut metadata = match object.get("metadata").filter(|value| !value.is_null()) {
        None => BTreeMap::new(),
        Some(value) => value
            .as_object()
            .ok_or_else(|| Error::validation("case.metadata must be an object"))?
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    };
    for (key, value) in object {
        if !CASE_KEYS.contains(&key.as_str()) {
            metadata.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    Ok(EvaluationCase {
        case_id,
        name: optional_string(object.get("name")),
        input,
        assets: string_list(object.get("assets"), "case.assets")?,
        reference,
        workspace_source_dir: optional_string(object.get("workspace_source_dir"))
            .map(PathBuf::from),
        swebench,
        metadata,
    })
}

/// A dataset: an id plus an ordered case list.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Dataset {
    /// The dataset id, used in reports and artifact paths.
    pub dataset_id: String,
    /// The cases, in document order.
    pub cases: Vec<EvaluationCase>,
}

impl Dataset {
    /// Parses a dataset document: either an array of cases or an object
    /// `{dataset_id, cases}`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document has an unsupported shape,
    /// when a case is malformed, or when two cases share an id.
    pub fn from_json_value(value: &Value) -> Result<Self> {
        match value {
            Value::Array(items) => Self::from_cases(DEFAULT_DATASET_ID, items),
            Value::Object(object) => {
                let dataset_id = ["dataset_id", "name", "id"]
                    .iter()
                    .find_map(|key| object.get(*key).and_then(Value::as_str))
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(DEFAULT_DATASET_ID)
                    .to_owned();
                let cases = object
                    .get("cases")
                    .ok_or_else(|| Error::validation("dataset object must contain a `cases` list"))?
                    .as_array()
                    .ok_or_else(|| Error::validation("dataset `cases` must be a list"))?;
                Self::from_cases(&dataset_id, cases)
            }
            _ => Err(Error::validation(
                "dataset must be a list of cases or an object with a `cases` list",
            )),
        }
    }

    /// Parses one JSON case per non-blank line.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the case index when a line is not
    /// valid JSON, when a case is malformed, or when two cases share an id.
    pub fn from_jsonl_str(text: &str, dataset_id: &str) -> Result<Self> {
        let mut cases = Vec::new();
        for (line_number, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let index = cases.len();
            let value: Value = serde_json::from_str(trimmed).map_err(|error| {
                Error::validation(format!(
                    "case {index} (line {}): not valid JSON: {error}",
                    line_number + 1
                ))
            })?;
            cases.push(EvaluationCase::from_json_value(index, &value)?);
        }
        Self::from_parsed_cases(dataset_id, cases)
    }

    /// Loads a dataset from disk: `.jsonl`/`.ndjson` by extension, JSON
    /// otherwise.
    ///
    /// A JSON array document takes the file stem as its dataset id when it does
    /// not name one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the file cannot be read and
    /// [`Error::Validation`] when its contents are malformed.
    pub async fn load(path: &Path) -> Result<Self> {
        let text = tokio::fs::read_to_string(path).await.map_err(|error| {
            Error::session(format!(
                "failed to read evaluation dataset `{}`: {error}",
                path.display()
            ))
        })?;
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_DATASET_ID)
            .to_owned();
        if extension == "jsonl" || extension == "ndjson" {
            return Self::from_jsonl_str(&text, &stem);
        }
        let value: Value = serde_json::from_str(&text).map_err(|error| {
            Error::validation(format!(
                "evaluation dataset `{}` is not valid JSON: {error}",
                path.display()
            ))
        })?;
        let mut dataset = Self::from_json_value(&value)?;
        if value.is_array() && dataset.dataset_id == DEFAULT_DATASET_ID {
            dataset.dataset_id = stem;
        }
        Ok(dataset)
    }

    /// Builds a dataset from already-parsed cases.
    fn from_parsed_cases(dataset_id: &str, cases: Vec<EvaluationCase>) -> Result<Self> {
        for (index, case) in cases.iter().enumerate() {
            if cases[..index]
                .iter()
                .any(|earlier| earlier.case_id == case.case_id)
            {
                return Err(Error::validation(format!(
                    "duplicate case id `{}`",
                    case.case_id
                )));
            }
        }
        Ok(Self {
            dataset_id: dataset_id.to_owned(),
            cases,
        })
    }

    /// Parses a case array under `dataset_id`.
    fn from_cases(dataset_id: &str, items: &[Value]) -> Result<Self> {
        let mut cases = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            cases.push(EvaluationCase::from_json_value(index, item)?);
        }
        Self::from_parsed_cases(dataset_id, cases)
    }

    /// Returns the case ids in document order.
    #[must_use]
    pub fn case_ids(&self) -> Vec<&str> {
        self.cases
            .iter()
            .map(|case| case.case_id.as_str())
            .collect()
    }

    /// Returns how many cases the dataset holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Returns whether the dataset holds no cases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

/// Reads an optional non-empty string field, ignoring non-strings.
fn optional_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

/// Reads a string list, rejecting non-lists and non-string entries.
fn string_list(value: Option<&Value>, field: &str) -> Result<Vec<String>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| Error::validation(format!("{field} must be a list of strings")))?;
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::validation(format!("{field}[{index}] must be a string")))
        })
        .collect()
}

/// Parses a non-negative finite weight, defaulting to `1.0`.
fn weight_field(object: &Map<String, Value>, fallback: f64) -> Result<f64> {
    let Some(value) = object.get("weight").filter(|value| !value.is_null()) else {
        return Ok(fallback);
    };
    let weight = value
        .as_f64()
        .ok_or_else(|| Error::validation("behavior weight must be a number"))?;
    if !weight.is_finite() || weight < 0.0 {
        return Err(Error::validation(
            "behavior weight must be finite and non-negative",
        ));
    }
    Ok(weight)
}

/// Parses a `0.0..=1.0` penalty, defaulting to `0.3`.
fn penalty_field(object: &Map<String, Value>) -> Result<f64> {
    let Some(value) = object.get("penalty").filter(|value| !value.is_null()) else {
        return Ok(ForbiddenBehaviorSpec::default().penalty);
    };
    let penalty = value
        .as_f64()
        .ok_or_else(|| Error::validation("behavior penalty must be a number"))?;
    if !penalty.is_finite() || !(0.0..=1.0).contains(&penalty) {
        return Err(Error::validation(
            "behavior penalty must be finite and between 0 and 1",
        ));
    }
    Ok(penalty)
}

/// Parses `reference.required_behaviors`.
fn required_behaviors(value: Option<&Value>) -> Result<Vec<BehaviorSpec>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| Error::validation("reference.required_behaviors must be a list"))?;
    let mut behaviors = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            Value::String(id) => {
                let id = id.trim();
                if id.is_empty() {
                    return Err(Error::validation(format!(
                        "reference.required_behaviors[{index}] must be a non-empty string"
                    )));
                }
                behaviors.push(BehaviorSpec::new(id));
            }
            Value::Object(object) => {
                let id = object
                    .get("id")
                    .or_else(|| object.get("description"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        Error::validation(format!(
                            "reference.required_behaviors[{index}].id must be a non-empty string"
                        ))
                    })?;
                behaviors.push(BehaviorSpec {
                    id: id.to_owned(),
                    weight: weight_field(object, 1.0)?,
                });
            }
            _ => {
                return Err(Error::validation(format!(
                    "reference.required_behaviors[{index}] must be a string or object"
                )))
            }
        }
    }
    reject_duplicate_ids(
        behaviors.iter().map(|behavior| behavior.id.as_str()),
        "reference.required_behaviors",
    )?;
    Ok(behaviors)
}

/// Parses `reference.forbidden_behaviors`.
fn forbidden_behaviors(value: Option<&Value>) -> Result<Vec<ForbiddenBehaviorSpec>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| Error::validation("reference.forbidden_behaviors must be a list"))?;
    let mut behaviors = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            Value::String(id) => {
                let id = id.trim();
                if id.is_empty() {
                    return Err(Error::validation(format!(
                        "reference.forbidden_behaviors[{index}] must be a non-empty string"
                    )));
                }
                behaviors.push(ForbiddenBehaviorSpec::new(id));
            }
            Value::Object(object) => {
                let id = object
                    .get("id")
                    .or_else(|| object.get("description"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        Error::validation(format!(
                            "reference.forbidden_behaviors[{index}].id must be a non-empty string"
                        ))
                    })?;
                behaviors.push(ForbiddenBehaviorSpec {
                    id: id.to_owned(),
                    penalty: penalty_field(object)?,
                });
            }
            _ => {
                return Err(Error::validation(format!(
                    "reference.forbidden_behaviors[{index}] must be a string or object"
                )))
            }
        }
    }
    reject_duplicate_ids(
        behaviors.iter().map(|behavior| behavior.id.as_str()),
        "reference.forbidden_behaviors",
    )?;
    Ok(behaviors)
}

/// Rejects repeated ids inside one behaviour list.
fn reject_duplicate_ids<'a>(ids: impl Iterator<Item = &'a str>, field: &str) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(Error::validation(format!("{field} ids must be unique")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn role_and_penalty_literals_round_trip() {
        assert_eq!(AnswerRole::Criterion.as_str(), "criterion");
        assert_eq!(AnswerRole::Reference.as_str(), "reference");
        assert!(matches!(
            AnswerRole::parse("reference"),
            Ok(AnswerRole::Reference)
        ));
        assert_eq!(PenaltyMode::Ceiling.as_str(), "ceiling");
        assert_eq!(PenaltyMode::Subtract.as_str(), "subtract");
        assert!(PenaltyMode::parse("voodoo").is_err());
        assert_eq!(AnswerRole::default(), AnswerRole::Criterion);
        assert_eq!(PenaltyMode::default(), PenaltyMode::Ceiling);
    }

    #[test]
    fn case_parses_aliases_and_keeps_unknown_keys() {
        let case = EvaluationCase::from_json_value(
            4,
            &json!({"id": "c-1", "task_input": {"q": 1}, "extra": [1], "swebench": {"instance": "o__r-1"}}),
        )
        .expect("case parses");
        assert_eq!(case.case_id, "c-1");
        assert_eq!(case.input, json!({"q": 1}));
        assert_eq!(case.metadata.get("extra"), Some(&json!([1])));
        assert_eq!(
            case.swebench.map(|swebench| swebench.instance_id),
            Some("o__r-1".to_owned())
        );
    }

    #[test]
    fn generated_ids_follow_the_case_index() {
        let case =
            EvaluationCase::from_json_value(7, &json!({"name": "nameless"})).expect("case parses");
        assert_eq!(case.case_id, "case_007");
    }

    #[test]
    fn reference_defaults_and_lists_parse() {
        let reference = CaseReference::from_json_value(&json!({
            "required_behaviors": ["a", {"description": "b", "weight": 2.0}],
            "forbidden_behaviors": [{"id": "c", "penalty": 0.5}],
            "penalty_mode": "subtract",
            "answer_role": "reference",
            "files": ["x.py"],
            "rubric": ["be correct"],
            "answer": "42"
        }))
        .expect("reference parses");
        assert_eq!(reference.required_behaviors[0], BehaviorSpec::new("a"));
        assert!((reference.required_behaviors[1].weight - 2.0).abs() < f64::EPSILON);
        assert!((reference.forbidden_behaviors[0].penalty - 0.5).abs() < f64::EPSILON);
        assert_eq!(reference.penalty_mode, PenaltyMode::Subtract);
        assert_eq!(reference.answer_role, AnswerRole::Reference);
        assert_eq!(reference.files, vec!["x.py".to_owned()]);
        assert_eq!(
            CaseReference::from_json_value(&Value::Null)
                .expect("null reference")
                .required_behaviors,
            Vec::new()
        );
    }

    #[test]
    fn duplicate_and_malformed_records_are_rejected() {
        let duplicate = Dataset::from_json_value(&json!([{"case_id": "a"}, {"case_id": "a"}]));
        assert!(duplicate.is_err());
        let bad_line = Dataset::from_jsonl_str("{\"case_id\":\"a\"}\nnot json\n", "ds");
        assert!(bad_line.is_err());
        let bad_index = EvaluationCase::from_json_value(3, &json!({"reference": []}));
        let message = bad_index
            .expect_err("reference must be an object")
            .to_string();
        assert!(message.contains("case 3"), "{message}");
    }

    #[test]
    fn jsonl_skips_blank_lines_and_numbers_cases() {
        let dataset = Dataset::from_jsonl_str("\n{\"input\": 1}\n\n{\"input\": 2}\n", "jsonl_ds")
            .expect("jsonl parses");
        assert_eq!(dataset.dataset_id, "jsonl_ds");
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.case_ids(), vec!["case_000", "case_001"]);
    }
}
