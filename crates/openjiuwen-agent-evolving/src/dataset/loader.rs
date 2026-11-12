//! [`CaseLoader`]: an in-memory collection of [`Case`]s with deterministic
//! shuffling and splitting, plus JSON/JSONL dataset loading.
//!
//! Ports `CaseLoader` from the Python `openjiuwen.agent_evolving.dataset`. The
//! Python loader uses `random.shuffle`; here the shuffle is driven by a small,
//! self-contained `xorshift64*` generator so the same seed always yields the
//! same order and the crate needs no random-number dependency. The Python
//! package leaves dataset file formats to the host, so JSON/JSONL parsing and
//! [`load_cases_from_path`] are additions here.

use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::dataset::Case;

/// A deterministic `xorshift64*` generator used by [`CaseLoader::shuffle`].
///
/// The algorithm is Marsaglia's xorshift64 shifted through a 64-bit multiply,
/// which is plenty for a reproducible permutation and keeps the crate free of
/// external random-number dependencies.
struct Rng64 {
    /// The generator state; never zero after construction.
    state: u64,
}

impl Rng64 {
    /// Seed the generator; seed `0` is remapped to the golden-ratio constant
    /// because xorshift requires a non-zero state.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// The next pseudo-random 64-bit word.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A pseudo-random index in `0..bound`.
    ///
    /// The modulo bias is irrelevant for shuffling a short case list.
    fn below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

/// A collection of evaluation [`Case`]s.
///
/// Python: `openjiuwen.agent_evolving.dataset.CaseLoader`. Construction is
/// infallible; ratio arguments are validated by [`CaseLoader::split`].
///
/// Serializes to a single-key object holding the cases in their current order,
/// for example `{"cases": [{...}, {...}]}`. Deserialization is permissive: the
/// `cases` key may be omitted (yielding an empty loader) and unknown keys are
/// ignored.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CaseLoader {
    /// The contained cases, in their current order.
    #[serde(default)]
    cases: Vec<Case>,
}

impl CaseLoader {
    /// Collects cases into a loader, preserving their order.
    pub fn new(cases: impl IntoIterator<Item = Case>) -> Self {
        Self {
            cases: cases.into_iter().collect(),
        }
    }

    /// The number of cases.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// The number of cases; Python's `size()` spelling of [`Self::len`].
    #[must_use]
    pub fn size(&self) -> usize {
        self.len()
    }

    /// Whether the loader holds no cases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// The cases in their current order.
    #[must_use]
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// Consumes the loader, returning the contained cases.
    #[must_use]
    pub fn into_cases(self) -> Vec<Case> {
        self.cases
    }

    /// The case at `index`, or `None` when the index is out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Case> {
        self.cases.get(index)
    }

    /// Iterates the cases by reference, in their current order.
    pub fn iter(&self) -> std::slice::Iter<'_, Case> {
        self.cases.iter()
    }

    /// Returns a copy of the loader with the cases in a shuffled order.
    ///
    /// A deterministic Fisher-Yates shuffle: the same `random_seed` always
    /// produces the same permutation of the same cases.
    #[must_use]
    pub fn shuffle(&self, random_seed: u64) -> Self {
        Self {
            cases: shuffle_cases(self.cases.clone(), random_seed),
        }
    }

    /// Splits the loader into a left and a right part at `floor(ratio * size)`.
    ///
    /// Unlike Python's `CaseLoader.split`, no shuffle is applied first: the
    /// current order is cut in place. `ratio` `0.0` puts every case on the
    /// right, `1.0` puts every case on the left. The left part is
    /// `[0, split)`, the right part `[split, size)`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `ratio` is not finite or outside `0.0..=1.0`.
    pub fn split(&self, ratio: f64) -> Result<(Self, Self)> {
        let (left, right) = split_cases(self.cases.clone(), ratio)?;
        Ok((Self::new(left), Self::new(right)))
    }
}

impl IntoIterator for CaseLoader {
    type Item = Case;
    type IntoIter = std::vec::IntoIter<Case>;

    /// Iterates the cases by value, in their current order.
    fn into_iter(self) -> Self::IntoIter {
        self.cases.into_iter()
    }
}

impl<'a> IntoIterator for &'a CaseLoader {
    type Item = &'a Case;
    type IntoIter = std::slice::Iter<'a, Case>;

    /// Iterates the cases by reference, in their current order.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Returns the cases in a deterministic shuffled order.
///
/// The shuffle is a Fisher-Yates pass driven by `Rng64`; the same seed always
/// yields the same permutation.
#[must_use]
pub fn shuffle_cases(mut cases: Vec<Case>, seed: u64) -> Vec<Case> {
    let mut rng = Rng64::new(seed);
    for i in (1..cases.len()).rev() {
        let j = rng.below(i + 1);
        cases.swap(i, j);
    }
    cases
}

/// Splits cases positionally at `floor(ratio * len)` without shuffling.
///
/// This is the Python module helper: unlike [`CaseLoader::split`], it does not
/// shuffle first, so the front of the input stays in the left part.
///
/// # Errors
///
/// [`Error::validation`] when `ratio` is not finite or outside `0.0..=1.0`.
pub fn split_cases(cases: Vec<Case>, ratio: f64) -> Result<(Vec<Case>, Vec<Case>)> {
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return Err(Error::validation(format!(
            "split ratio must be finite and between 0.0 and 1.0, got {ratio}"
        )));
    }
    let split_at = (ratio * cases.len() as f64).floor() as usize;
    let mut cases = cases;
    let right = cases.split_off(split_at);
    Ok((cases, right))
}

/// The dataset encoding accepted by [`parse_cases_str`] and
/// [`load_cases_from_path`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CaseFormat {
    /// One JSON document holding a case array, a `{"cases": [...]}` object, or
    /// a single case object.
    #[default]
    Json,
    /// One JSON case object per line, with blank lines and `#` comments
    /// skipped.
    JsonLines,
}

/// Parses a JSON dataset into validated cases.
///
/// The document may be a bare array of case objects, an object
/// `{"cases": [...]}`, or a single case object.
///
/// # Errors
///
/// [`Error::validation`] for malformed JSON or a case violating the Python
/// field constraints.
pub fn parse_cases_json(content: &str) -> Result<Vec<Case>> {
    let value: Value = serde_json::from_str(content)
        .map_err(|error| Error::validation(format!("invalid JSON dataset: {error}")))?;

    let cases = match value {
        Value::Array(items) => items
            .into_iter()
            .enumerate()
            .map(|(index, item)| parse_case_value(item, &index.to_string()))
            .collect::<Result<Vec<_>>>()?,
        Value::Object(mut object) => match object.remove("cases") {
            Some(Value::Array(items)) => items
                .into_iter()
                .enumerate()
                .map(|(index, item)| parse_case_value(item, &index.to_string()))
                .collect::<Result<Vec<_>>>()?,
            Some(_) => {
                return Err(Error::validation(
                    "dataset `cases` must be a JSON array of case objects",
                ));
            }
            None => vec![parse_case_value(Value::Object(object), "object")?],
        },
        _ => {
            return Err(Error::validation(
                "dataset must be a JSON array, a `{\"cases\": [...]}` object, or a case object",
            ));
        }
    };

    Ok(cases)
}

/// Converts one JSON value into a validated [`Case`], naming its position on
/// failure.
fn parse_case_value(value: Value, position: &str) -> Result<Case> {
    let case: Case = serde_json::from_value(value)
        .map_err(|error| Error::validation(format!("invalid case at {position}: {error}")))?;
    case.validate()?;
    Ok(case)
}

/// Parses a JSONL dataset: one JSON case object per line.
///
/// Blank lines and lines whose trimmed form starts with `#` are skipped.
///
/// # Errors
///
/// [`Error::validation`] naming the offending line number, or a case violating
/// the Python field constraints.
pub fn parse_cases_jsonl(content: &str) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|error| {
            Error::validation(format!("invalid JSON on line {}: {error}", index + 1))
        })?;
        cases.push(parse_case_value(value, &format!("line {}", index + 1))?);
    }
    Ok(cases)
}

/// Parses dataset content in the given [`CaseFormat`].
///
/// # Errors
///
/// The errors of [`parse_cases_json`] or [`parse_cases_jsonl`].
pub fn parse_cases_str(content: &str, format: CaseFormat) -> Result<Vec<Case>> {
    match format {
        CaseFormat::Json => parse_cases_json(content),
        CaseFormat::JsonLines => parse_cases_jsonl(content),
    }
}

/// Reads and parses a dataset file.
///
/// The format is inferred from the extension: `.jsonl` and `.ndjson`
/// (case-insensitive) select [`CaseFormat::JsonLines`], anything else
/// [`CaseFormat::Json`].
///
/// # Errors
///
/// [`Error::not_found`] when the file is missing, [`Error::config`] for other
/// IO failures, and the parse errors of [`parse_cases_str`].
pub async fn load_cases_from_path(path: impl AsRef<Path>) -> Result<Vec<Case>> {
    let path = path.as_ref();
    let format = match path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jsonl" | "ndjson") => CaseFormat::JsonLines,
        _ => CaseFormat::Json,
    };
    let content = tokio::fs::read_to_string(path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Error::not_found(format!("dataset file not found: {}", path.display()))
        } else {
            Error::config(format!(
                "failed to read dataset file {}: {error}",
                path.display()
            ))
        }
    })?;
    parse_cases_str(&content, format)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A minimal valid case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(id.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(id.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// A loader holding `count` cases with the ids `0..count`.
    fn loader(count: usize) -> CaseLoader {
        CaseLoader::new((0..count).map(|i| case(&i.to_string())))
    }

    /// The case ids of a loader, in order.
    fn ids(loader: &CaseLoader) -> Vec<String> {
        loader.iter().map(|c| c.case_id.clone()).collect()
    }

    #[test]
    fn new_tracks_len_and_emptiness() {
        let empty = CaseLoader::new(Vec::new());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.size(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.cases(), []);

        let populated = loader(3);
        assert_eq!(populated.len(), 3);
        assert_eq!(populated.size(), 3);
        assert!(!populated.is_empty());
        assert_eq!(ids(&populated), ["0", "1", "2"]);
    }

    #[test]
    fn get_reports_in_and_out_of_range_indexes() {
        let loader = loader(2);
        assert_eq!(loader.get(0).map(|c| c.case_id.as_str()), Some("0"));
        assert!(loader.get(2).is_none());
        assert!(loader.get(usize::MAX).is_none());
    }

    #[test]
    fn split_uses_the_floor_of_ratio_times_len_without_shuffling() {
        let (left, right) = loader(5).split(0.5).unwrap();
        assert_eq!(ids(&left), ["0", "1"]);
        assert_eq!(ids(&right), ["2", "3", "4"]);

        let (left, right) = loader(3).split(0.0).unwrap();
        assert!(left.is_empty());
        assert_eq!(right.size(), 3);

        let (left, right) = loader(3).split(1.0).unwrap();
        assert_eq!(left.size(), 3);
        assert!(right.is_empty());
    }

    #[test]
    fn split_rejects_invalid_ratios() {
        for ratio in [-0.1, 1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = loader(3).split(ratio).unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("ratio")),
                "unexpected error for {ratio}: {error}"
            );
        }
    }

    #[test]
    fn shuffle_is_deterministic_and_a_permutation() {
        let fixture = loader(7);
        assert_eq!(ids(&fixture.shuffle(42)), ids(&fixture.shuffle(42)));
        for seed in [0, 1, 2, 42, u64::MAX] {
            let mut shuffled = ids(&fixture.shuffle(seed));
            shuffled.sort();
            let mut expected: Vec<String> = (0..7).map(|i| i.to_string()).collect();
            expected.sort();
            assert_eq!(shuffled, expected, "seed {seed} keeps the multiset");
        }
        assert!(CaseLoader::default().shuffle(7).is_empty());
        assert_eq!(ids(&loader(1).shuffle(7)), ["0"]);
    }

    #[test]
    fn shuffle_cases_matches_the_loader_method() {
        let loader = loader(10);
        assert_eq!(
            shuffle_cases(loader.cases().to_vec(), 99),
            loader.shuffle(99).into_cases()
        );
    }

    #[test]
    fn iteration_by_value_and_by_reference_preserve_order() {
        let loader = loader(3);
        let by_reference: Vec<&str> = (&loader).into_iter().map(|c| c.case_id.as_str()).collect();
        assert_eq!(by_reference, ["0", "1", "2"]);

        let by_value: Vec<String> = loader.into_iter().map(|c| c.case_id).collect();
        assert_eq!(by_value, ["0", "1", "2"]);
    }

    #[test]
    fn serialization_round_trips_through_the_cases_key() {
        let loader = loader(3);
        let value = serde_json::to_value(&loader).unwrap();
        assert_eq!(value["cases"].as_array().map(Vec::len), Some(3));
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(1));
        let restored: CaseLoader = serde_json::from_value(value).unwrap();
        assert_eq!(restored, loader);
    }

    #[test]
    fn deserialization_defaults_an_empty_loader_and_ignores_unknown_keys() {
        let empty: CaseLoader = serde_json::from_str("{}").unwrap();
        assert!(empty.is_empty());

        let with_extra: CaseLoader = serde_json::from_value(serde_json::json!({
            "cases": [serde_json::to_value(case("0")).unwrap()],
            "unknown_key": {"nested": true},
        }))
        .unwrap();
        assert_eq!(ids(&with_extra), ["0"]);
    }

    #[test]
    fn json_parsing_accepts_the_three_shapes() {
        let array =
            parse_cases_json(r#"[{"inputs": {"q": "a"}, "label": {"a": "b"}, "case_id": "c1"}]"#)
                .unwrap();
        assert_eq!(array.len(), 1);

        let wrapped = parse_cases_json(
            r#"{"cases": [{"inputs": {"q": "a"}, "label": {"a": "b"}, "case_id": "c1"}]}"#,
        )
        .unwrap();
        assert_eq!(wrapped[0].case_id, "c1");

        let single = parse_cases_json(r#"{"inputs": {"q": "a"}, "label": {"a": "b"}}"#).unwrap();
        assert_eq!(single.len(), 1);
    }

    #[test]
    fn json_parsing_rejects_malformed_and_invalid_cases() {
        assert!(matches!(
            parse_cases_json("not json").unwrap_err(),
            Error::Validation(_)
        ));
        assert!(matches!(
            parse_cases_json("42").unwrap_err(),
            Error::Validation(_)
        ));
        assert!(matches!(
            parse_cases_json(r#"{"cases": 42}"#).unwrap_err(),
            Error::Validation(_)
        ));
        assert!(matches!(
            parse_cases_json(r#"[{"inputs": {}, "label": {"a": "b"}}]"#).unwrap_err(),
            Error::Validation(_)
        ));
    }

    #[test]
    fn jsonl_parsing_skips_blank_and_comment_lines() {
        let content = "\n# a comment\n{\"inputs\": {\"q\": \"a\"}, \"label\": {\"a\": \"b\"}}\n\n   \n{\"inputs\": {\"q\": \"c\"}, \"label\": {\"a\": \"d\"}}\n";
        let cases = parse_cases_jsonl(content).unwrap();
        assert_eq!(cases.len(), 2);
    }

    #[test]
    fn jsonl_parsing_names_the_bad_line() {
        let content = "{\"inputs\": {\"q\": \"a\"}, \"label\": {\"a\": \"b\"}}\nnope\n";
        let error = parse_cases_jsonl(content).unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("line 2")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn parse_cases_str_dispatches_on_the_format() {
        let json = r#"{"inputs": {"q": "a"}, "label": {"a": "b"}}"#;
        assert_eq!(parse_cases_str(json, CaseFormat::Json).unwrap().len(), 1);
        assert_eq!(
            parse_cases_str(json, CaseFormat::JsonLines).unwrap().len(),
            1
        );
        assert_eq!(CaseFormat::default(), CaseFormat::Json);
    }
}
