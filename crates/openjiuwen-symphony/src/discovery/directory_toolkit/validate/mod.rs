//! The typed `skill_index` request and its full validation pass.
//!
//! Ports the argument layer of Python `openjiuwen.symphony.discovery.directory_toolkit`:
//! `_validate_request`, `_queries`, `_category_paths`, `_apply_simple_defaults`,
//! `_output_budget`, `_enum`, `_positive`, `_boolean`, `_nonempty`, `_bounded_query`,
//! `_paths`, and `_identifiers`.
//!
//! # The request type is the wire shape
//!
//! Python's request is an untyped `dict[str, Any]` filled from
//! `_SKILL_INDEX_DEFAULTS`, so an unrecognized enum value reaches `_enum` and becomes a
//! field error. [`SkillIndexRequest`] keeps that property: the nine enum-valued slots are
//! [`String`]s ([`Operation`], [`ListView`](crate::discovery::directory_toolkit::render::ListView), [`SearchMatch`], [`SearchResult`],
//! [`ReadMode`], [`OutputMode`], and [`PipelineOperation`] are the parsed forms), and the
//! numeric slots are `Option<i64>`, so a zero or a negative number is representable and
//! reportable instead of failing deserialization. [`validate`] parses all of them into a
//! [`ValidatedRequest`], which carries the typed enums, the normalized strings and
//! numbers, and the normalized request.
//!
//! # Accumulation
//!
//! Python raises on the first problem. The Rust pass collects every issue into a
//! `Vec<RequestIssue>` and returns them together, each naming its field (`view`,
//! `pipeline[0].lines`, …). That is a deliberate improvement: a model that gets one
//! field wrong usually gets several wrong, and one round trip should surface them all.
//!
//! # Deviations from Python
//!
//! - Issues accumulate instead of raising, and each names exactly one field.
//! - Python's `query` (list) and `queries` (list of 2+) are one field here: [`SkillIndexRequest::query`]
//!   accepts 1–8 values, and the batch rules (a single `limit` pipeline stage,
//!   `output_mode=entries`, `per_query_limit <= 10`) apply when it holds more than one.
//! - A blank `category` becomes "no category", mirroring `_DirectoryFunction`, instead of
//!   raising `_nonempty`'s error.
//! - `limit` is validated here but "`limit` cannot be combined with advanced arguments"
//!   stays with the toolkit layer, which is the only layer that knows whether the call
//!   used the simple or the advanced shape.
//! - The read-requires-`skills` and `search`-requires-a-`query` checks run in this pass;
//!   Python raises them later, in `_read_skills` and `_search`.

pub mod limits;
pub mod pipeline;
pub mod request;
pub mod rules;

pub use limits::{
    output_budget, validate_category_paths, validate_identifiers, validate_paths, validate_queries,
    DEFAULT_RESULT_LIMIT, DEFAULT_SEARCH_PAGE_SIZE, MAX_IDENTIFIER_CHARS, MAX_LINES,
    MAX_OUTPUT_CHARS, MAX_PATHS, MAX_PATH_CHARS, MAX_PER_QUERY_LIMIT, MAX_QUERIES, MAX_QUERY_CHARS,
    MAX_RESULT_LIMIT, MAX_SEARCH_PAGE_SIZE, MIN_OUTPUT_CHARS, ROOT_PATH,
};
pub use pipeline::validate_pipeline;
pub use request::{
    Operation, OutputMode, PipelineOperation, PipelineStage, ReadMode, RequestIssue, SearchMatch,
    SearchResult, SkillIndexRequest, ValidatedRequest,
};
pub use rules::{apply_simple_defaults, validate};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::discovery::directory_toolkit::render::ListView;

    #[test]
    fn enums_parse_and_name_themselves() {
        assert_eq!(Operation::ALL.len(), 3);
        assert_eq!(Operation::parse("search"), Some(Operation::Search));
        assert_eq!(Operation::parse("SEARCH"), None);
        assert_eq!(Operation::Search.as_str(), "search");
        assert_eq!(Operation::Search.to_string(), "search");
        assert_eq!(SearchMatch::parse("path"), Some(SearchMatch::Path));
        assert_eq!(
            PipelineOperation::names(),
            ["limit", "slice", "filter", "count"]
        );
    }

    #[test]
    fn request_defaults_match_the_python_defaults() {
        let request = SkillIndexRequest::default();
        assert_eq!(request.operation, "list");
        assert_eq!(request.view, "names");
        assert_eq!(request.search_match, "content");
        assert_eq!(request.result, "files");
        assert!(request.case_insensitive);
        assert_eq!(request.read_mode, "full");
        assert_eq!(request.output_mode, "entries");
        let decoded: SkillIndexRequest =
            serde_json::from_str(r#"{"operation":"search","match":"name","future":1}"#).unwrap();
        assert_eq!(decoded.search_match, "name");
        assert!(decoded.case_insensitive);
    }

    #[test]
    fn a_valid_list_request_passes() {
        let request = SkillIndexRequest::new(Operation::List);
        let validated = validate(&request).unwrap();
        assert_eq!(validated.operation, Operation::List);
        assert_eq!(validated.view, ListView::Names);
        assert_eq!(validated.paths, ["/"]);
        assert_eq!(validated.page_offset, 0);
        assert!(validated.page_size.is_none());
    }

    #[test]
    fn issues_accumulate_and_name_their_fields() {
        let request = SkillIndexRequest {
            operation: "bogus".to_string(),
            view: "wide".to_string(),
            max_depth: Some(0),
            ..SkillIndexRequest::default()
        };
        let issues = validate(&request).unwrap_err();
        let fields: Vec<&str> = issues.iter().map(|issue| issue.field.as_str()).collect();
        assert!(fields.contains(&"operation"));
        assert!(fields.contains(&"view"));
        assert!(fields.contains(&"max_depth"));
        assert!(issues
            .iter()
            .any(|issue| issue.to_string() == "view: must be one of: names, details, tree"));
    }

    #[test]
    fn pipeline_rules_are_typed_errors_too() {
        assert_eq!(
            validate_pipeline(&[PipelineStage::limit(3)]).unwrap(),
            [PipelineStage::limit(3)]
        );
        let error = validate_pipeline(&[PipelineStage::limit(0)]).unwrap_err();
        assert_eq!(
            error.message(),
            "pipeline[0].lines: must be a positive integer"
        );
        let error = validate_pipeline(&[PipelineStage::slice(5, 4)]).unwrap_err();
        assert_eq!(
            error.message(),
            "pipeline[0].end_line: must be >= start_line"
        );
        let error = validate_pipeline(&[PipelineStage::limit(5_001)]).unwrap_err();
        assert_eq!(error.message(), "pipeline[0].lines: must not exceed 5000");
        let mut stage = PipelineStage::limit(2);
        stage.query = Some("pdf".to_string());
        let error = validate_pipeline(&[stage]).unwrap_err();
        assert_eq!(
            error.message(),
            "pipeline[0]: has fields not valid for limit"
        );
        let mut missing = PipelineStage::filter("x");
        missing.query = None;
        let error = validate_pipeline(&[missing]).unwrap_err();
        assert_eq!(error.message(), "pipeline[0].query: is required for filter");
        let mut bad = PipelineStage::count();
        bad.operation = "shuffle".to_string();
        let error = validate_pipeline(&[bad]).unwrap_err();
        assert!(error
            .message()
            .contains("pipeline[0].operation: must be one of"));
    }

    #[test]
    fn budgets_are_bounded() {
        assert_eq!(output_budget(None, false, 12_000).unwrap(), Some(12_000));
        assert_eq!(output_budget(Some(600), false, 12_000).unwrap(), Some(600));
        assert_eq!(output_budget(Some(600), true, 12_000).unwrap(), None);
        assert!(output_budget(Some(0), false, 12_000).is_err());
        assert!(output_budget(Some(511), false, 12_000).is_err());
        assert!(output_budget(Some(48_001), false, 12_000).is_err());
    }

    #[test]
    fn queries_paths_and_identifiers_are_normalized() {
        assert_eq!(
            validate_queries(&Some(vec![" pdf ".to_string(), "pdf".to_string()])).unwrap(),
            ["pdf"]
        );
        assert!(validate_queries(&None).is_err());
        assert!(validate_queries(&Some(vec!["  ".to_string()])).is_err());
        assert_eq!(validate_paths(&None, false).unwrap(), ["/"]);
        assert!(validate_paths(&None, true).is_err());
        assert_eq!(
            validate_paths(&Some(vec![" /a ".to_string(), "/a".to_string()]), false).unwrap(),
            ["/a"]
        );
        assert_eq!(
            validate_identifiers(&Some(vec![" alpha ".to_string()]), "skills", true).unwrap(),
            ["alpha"]
        );
        assert!(validate_identifiers(&None, "skills", true).is_err());
    }
}
