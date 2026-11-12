//! Integration tests for the `dataset` module.
//!
//! Exercises JSON/JSONL parsing, async file loading, deterministic
//! shuffle/split semantics, and the serde/validation contract of `Case` and
//! `EvaluatedCase`. Everything is offline and deterministic.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use openjiuwen_agent_evolving::dataset::*;
use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::Value;

/// A one-entry map used for `inputs`/`label` fixtures.
fn map(key: &str, value: &str) -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    map.insert(key.to_owned(), Value::String(value.to_owned()));
    map
}

/// A valid case with a fixed identifier.
fn case(id: &str) -> Case {
    Case::new(map("question", id), map("answer", id))
        .unwrap()
        .with_case_id(id)
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
fn json_array_object_and_single_case_all_parse() {
    let array = parse_cases_json(
        r#"[
            {"inputs": {"q": "a"}, "label": {"a": "b"}, "case_id": "c1"},
            {"inputs": {"q": "c"}, "label": {"a": "d"}, "case_id": "c2"}
        ]"#,
    )
    .unwrap();
    assert_eq!(array.len(), 2);
    assert_eq!(array[0].case_id, "c1");

    let wrapped = parse_cases_json(
        r#"{"cases": [{"inputs": {"q": "a"}, "label": {"a": "b"}, "case_id": "c1"}]}"#,
    )
    .unwrap();
    assert_eq!(wrapped.len(), 1);
    assert_eq!(wrapped[0].case_id, "c1");

    let single =
        parse_cases_json(r#"{"inputs": {"q": "a"}, "label": {"a": "b"}, "case_id": "c1"}"#)
            .unwrap();
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].label, map("a", "b"));
}

#[test]
fn invalid_json_and_invalid_cases_are_validation_errors() {
    for content in [
        "not json",
        "42",
        r#"{"cases": 42}"#,
        r#"[{"inputs": {}, "label": {"a": "b"}}]"#,
        r#"[{"inputs": {"q": "a"}, "label": {}}]"#,
    ] {
        let error = parse_cases_json(content).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "content: {content}");
        assert!(
            matches!(&error, Error::Validation(_)),
            "content: {content}, error: {error}"
        );
    }
}

#[test]
fn case_new_rejects_empty_inputs_and_label() {
    let error = Case::new(BTreeMap::new(), map("answer", "b")).unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("inputs")),
        "unexpected error: {error}"
    );

    let error = Case::new(map("question", "a"), BTreeMap::new()).unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("label")),
        "unexpected error: {error}"
    );
}

#[test]
fn parsed_case_without_id_gets_a_32_char_lowercase_hex_id() {
    let cases = parse_cases_json(r#"{"inputs": {"q": "a"}, "label": {"a": "b"}}"#).unwrap();
    let id = &cases[0].case_id;
    assert_eq!(id.len(), 32, "simple v4 uuid");
    assert!(
        id.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "not lowercase hex: {id}"
    );
    assert!(!id.contains('-'));
}

#[test]
fn jsonl_skips_blank_lines_and_comments() {
    let content = "\
# leading comment
{\"inputs\": {\"q\": \"a\"}, \"label\": {\"a\": \"b\"}}

{\"inputs\": {\"q\": \"c\"}, \"label\": {\"a\": \"d\"}}
   # indented comment
";
    let cases = parse_cases_jsonl(content).unwrap();
    assert_eq!(cases.len(), 2);
}

#[test]
fn jsonl_bad_line_is_a_validation_error() {
    let content = "{\"inputs\": {\"q\": \"a\"}, \"label\": {\"a\": \"b\"}}\nnope\n";
    let error = parse_cases_jsonl(content).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("line 2"), "error: {error}");
}

#[test]
fn parse_cases_str_dispatches_both_formats() {
    let json = r#"{"inputs": {"q": "a"}, "label": {"a": "b"}}"#;
    assert_eq!(parse_cases_str(json, CaseFormat::Json).unwrap().len(), 1);
    assert_eq!(
        parse_cases_str(json, CaseFormat::JsonLines).unwrap().len(),
        1
    );
    assert_eq!(CaseFormat::default(), CaseFormat::Json);
}

#[tokio::test]
async fn load_cases_from_path_reads_a_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cases.jsonl");
    std::fs::write(
        &path,
        "# comment\n{\"inputs\": {\"q\": \"a\"}, \"label\": {\"a\": \"b\"}, \"case_id\": \"x1\"}\n{\"inputs\": {\"q\": \"c\"}, \"label\": {\"a\": \"d\"}, \"case_id\": \"x2\"}\n",
    )
    .unwrap();

    let cases = load_cases_from_path(&path).await.unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].case_id, "x1");
    assert_eq!(cases[1].case_id, "x2");

    // `.ndjson` also selects JSONL.
    let ndjson = dir.path().join("CASES.NDJSON");
    std::fs::write(
        &ndjson,
        "{\"inputs\": {\"q\": \"a\"}, \"label\": {\"a\": \"b\"}}\n",
    )
    .unwrap();
    assert_eq!(load_cases_from_path(&ndjson).await.unwrap().len(), 1);
}

#[tokio::test]
async fn load_cases_from_path_reads_a_json_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cases.json");
    std::fs::write(
        &path,
        r#"{"cases": [{"inputs": {"q": "a"}, "label": {"a": "b"}}]}"#,
    )
    .unwrap();

    let cases = load_cases_from_path(&path).await.unwrap();
    assert_eq!(cases.len(), 1);
}

#[tokio::test]
async fn load_cases_from_path_reports_a_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let error = load_cases_from_path(dir.path().join("absent.json"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(matches!(&error, Error::NotFound(_)), "error: {error}");
}

#[test]
fn shuffle_is_deterministic_for_a_seed() {
    let loader = loader(20);
    let first = loader.shuffle(1234);
    let second = loader.shuffle(1234);
    assert_eq!(ids(&first), ids(&second));
    assert_eq!(first, second);
}

#[test]
fn different_seeds_produce_different_orders() {
    let loader = loader(20);
    let baseline = ids(&loader);
    let orders: Vec<Vec<String>> = [1_u64, 2, 3, 4]
        .into_iter()
        .map(|seed| ids(&loader.shuffle(seed)))
        .collect();
    assert!(
        orders.iter().any(|order| *order != baseline),
        "no seed reordered the cases"
    );
    assert!(
        orders.windows(2).any(|pair| pair[0] != pair[1]),
        "two seeds produced identical orders: {orders:?}"
    );

    let mut sorted = orders[0].clone();
    sorted.sort();
    let mut expected = baseline.clone();
    expected.sort();
    assert_eq!(sorted, expected, "the multiset of ids is preserved");
}

#[test]
fn shuffle_cases_matches_the_loader_method() {
    let loader = loader(20);
    let via_helper = shuffle_cases(loader.cases().to_vec(), 77);
    let via_method = loader.shuffle(77).into_cases();
    assert_eq!(via_helper, via_method);
}

#[test]
fn split_uses_a_positional_cut() {
    let (left, right) = loader(5).split(0.5).unwrap();
    assert_eq!(left.size(), 2);
    assert_eq!(right.size(), 3);
    assert_eq!(ids(&left), ["0", "1"]);
    assert_eq!(ids(&right), ["2", "3", "4"]);
}

#[test]
fn split_cases_is_positional_and_does_not_shuffle() {
    let cases = loader(5).into_cases();
    let (left, right) = split_cases(cases, 0.4).unwrap();
    let left_ids: Vec<String> = left.into_iter().map(|c| c.case_id).collect();
    let right_ids: Vec<String> = right.into_iter().map(|c| c.case_id).collect();
    assert_eq!(left_ids, ["0", "1"]);
    assert_eq!(right_ids, ["2", "3", "4"]);
}

#[test]
fn split_rejects_invalid_ratios() {
    for ratio in [1.5, -0.1, f64::NAN] {
        let error = loader(5).split(ratio).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "ratio: {ratio}");

        let error = split_cases(loader(5).into_cases(), ratio).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "ratio: {ratio}");
    }
}

#[test]
fn case_serde_round_trip_including_missing_tools() {
    let case = case("c1");
    let value = serde_json::to_value(&case).unwrap();
    assert_eq!(value["tools"], Value::Null);
    let restored: Case = serde_json::from_value(value).unwrap();
    assert_eq!(restored, case);
    assert!(restored.tools.is_none());
}

#[test]
fn evaluated_case_serde_round_trip_with_per_metric() {
    let mut per_metric = BTreeMap::new();
    per_metric.insert("exact".to_owned(), 1.0);
    let mut evaluated =
        EvaluatedCase::new(case("c1"), Some(map("answer", "a")), 0.5, "half").unwrap();
    evaluated.per_metric = Some(per_metric);

    let value = serde_json::to_value(&evaluated).unwrap();
    let object = value.as_object().unwrap();
    for key in ["case", "answer", "score", "reason", "per_metric"] {
        assert!(object.contains_key(key), "missing field {key}");
    }

    let restored: EvaluatedCase = serde_json::from_value(value).unwrap();
    assert_eq!(restored, evaluated);
}

#[test]
fn absent_answer_and_per_metric_serialize_as_null() {
    let evaluated = EvaluatedCase::new(case("c1"), None, 0.0, "").unwrap();
    let value = serde_json::to_value(&evaluated).unwrap();
    assert_eq!(value["answer"], Value::Null);
    assert_eq!(value["per_metric"], Value::Null);
}

#[test]
fn from_case_yields_a_zero_score_and_no_answer() {
    let evaluated = EvaluatedCase::from(case("c1"));
    assert!(evaluated.score.abs() < f64::EPSILON);
    assert!(evaluated.answer.is_none());
    assert!(evaluated.per_metric.is_none());
    assert_eq!(evaluated.reason, "");
    assert_eq!(evaluated.case_id(), "c1");
    assert!(evaluated.validate().is_ok());
}

#[test]
fn evaluated_case_clamps_and_rejects_scores() {
    let evaluated = EvaluatedCase::new(case("c1"), None, 1.5, "high").unwrap();
    assert!((evaluated.score - 1.0).abs() < f64::EPSILON);

    let evaluated = EvaluatedCase::new(case("c1"), None, -2.0, "low").unwrap();
    assert!(evaluated.score.abs() < f64::EPSILON);

    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = EvaluatedCase::new(case("c1"), None, score, "bad").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "score: {score}");
    }
}
