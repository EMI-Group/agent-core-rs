//! `${node.field}` reference parsing and resolution.
//!
//! Workflow data flow is textual: a component's `inputs_schema` is a JSON template
//! whose pure `"${node.field}"` strings resolve against workflow state before the
//! component runs. This module owns the path grammar and the resolver; it never fails
//! a run, because a reference to a missing value resolves to `null`.
//!
//! Path grammar (a superset of Python's `split_nested_path`):
//!
//! - dotted keys: `a.b.c`
//! - array indexes: `a[0]`, `a[-1]` (negative indexes count from the end)
//! - quoted keys: `a['key with spaces']`, `a["k"]`
//! - a numeric dotted segment: `a.0` indexes an array, and falls back to a key when
//!   the container is an object
//!
//! ```
//! use serde_json::json;
//! use openjiuwen_workflow::reference::{project, resolve_path};
//!
//! let source = json!({"start": {"items": [{"name": "a"}, {"name": "b"}]}});
//! assert_eq!(project(&source, "start.items[1].name"), json!("b"));
//! assert_eq!(project(&source, "start.missing"), json!(null));
//! ```
//!
//! **Array projection.** When the current value is an array and the next segment is a
//! non-numeric key, the key is projected over the elements: every element that
//! resolves contributes its value, in order, and elements that lack the key are
//! skipped. An empty array projects to an empty array. Python returns `null` there;
//! the projection is a documented Rust extension that lets buffered stream
//! composition read `"${producer.text}"` from a list of frames.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

/// One segment of a reference path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathSegment {
    /// An object key, or a numeric key that indexes an array.
    Key(String),
    /// A bracketed array index; negative indexes count from the end.
    Index(i64),
}

/// Whether `s` is a whole-string reference such as `"${a.b}"`.
///
/// A string that merely contains a reference (`"prefix ${a}"`) is a literal.
#[must_use]
pub fn is_ref_path(s: &str) -> bool {
    s.len() > 3 && s.starts_with("${") && s.ends_with('}')
}

/// Extracts the path of a whole-string reference: `"${a.b}"` becomes `Some("a.b")`.
///
/// Returns `None` for literals, and for strings that concatenate several references
/// (`"${a}${b}"`), which are treated as literals because they are not a single path.
#[must_use]
pub fn extract_origin_key(s: &str) -> Option<&str> {
    if !is_ref_path(s) {
        return None;
    }
    let inner = &s[2..s.len() - 1];
    if inner.contains('{') || inner.contains('}') {
        return None;
    }
    Some(inner)
}

/// Splits a path into [`PathSegment`]s.
///
/// An empty path yields no segments.
///
/// # Errors
///
/// Returns [`Error::Validation`] for malformed paths: an unterminated `[`, an empty
/// subscript, an unterminated or empty quoted key, or a bare subscript that is
/// neither an integer nor a quoted key.
pub fn split_path(path: &str) -> Result<Vec<PathSegment>> {
    let mut segments = Vec::new();
    if path.is_empty() {
        return Ok(segments);
    }
    let chars: Vec<char> = path.chars().collect();
    let mut key = String::new();
    let mut cursor = 0;
    while cursor < chars.len() {
        match chars[cursor] {
            '.' => {
                if !key.is_empty() {
                    segments.push(PathSegment::Key(std::mem::take(&mut key)));
                }
                cursor += 1;
            }
            '[' => {
                if !key.is_empty() {
                    segments.push(PathSegment::Key(std::mem::take(&mut key)));
                }
                let end = find_bracket_end(&chars, cursor, path)?;
                let inner: String = chars[cursor + 1..end].iter().collect();
                segments.push(parse_bracket(inner.trim(), path)?);
                cursor = end + 1;
            }
            c => {
                key.push(c);
                cursor += 1;
            }
        }
    }
    if !key.is_empty() {
        segments.push(PathSegment::Key(key));
    }
    Ok(segments)
}

/// Returns the index of the `]` matching the `[` at `start`, honouring quoted keys.
fn find_bracket_end(chars: &[char], start: usize, path: &str) -> Result<usize> {
    let mut quote: Option<char> = None;
    for (offset, ch) in chars[start + 1..].iter().enumerate() {
        if let Some(q) = quote {
            if *ch == q {
                quote = None;
            }
            continue;
        }
        match *ch {
            '\'' | '"' => quote = Some(*ch),
            ']' => return Ok(start + 1 + offset),
            _ => {}
        }
    }
    Err(Error::validation(format!(
        "malformed reference path `{path}`: unterminated `[`"
    )))
}

/// Parses the inside of a `[...]` subscript.
fn parse_bracket(inner: &str, path: &str) -> Result<PathSegment> {
    if inner.is_empty() {
        return Err(Error::validation(format!(
            "malformed reference path `{path}`: empty `[]` subscript"
        )));
    }
    let quote = inner.chars().next().filter(|c| *c == '\'' || *c == '"');
    if let Some(q) = quote {
        let Some(body) = inner[1..].strip_suffix(q) else {
            return Err(Error::validation(format!(
                "malformed reference path `{path}`: unterminated quoted key in `[{inner}]`"
            )));
        };
        if body.is_empty() {
            return Err(Error::validation(format!(
                "malformed reference path `{path}`: empty quoted key in `[{inner}]`"
            )));
        }
        if body.contains(q) {
            return Err(Error::validation(format!(
                "malformed reference path `{path}`: nested quote in `[{inner}]`"
            )));
        }
        return Ok(PathSegment::Key(body.to_string()));
    }
    if let Ok(index) = inner.parse::<i64>() {
        return Ok(PathSegment::Index(index));
    }
    Err(Error::validation(format!(
        "malformed reference path `{path}`: `[{inner}]` is neither an integer index nor a quoted key"
    )))
}

/// Resolves `path` against `source`, returning `None` when any step is missing.
///
/// A `null` root, a missing key, an out-of-range index, or a scalar in the middle of
/// the path all resolve to `None`; the function never errors. See the module docs for
/// the array-projection rule.
#[must_use]
pub fn resolve_path(source: &Value, path: &str) -> Option<Value> {
    let segments = split_path(path).ok()?;
    resolve_segments(source, &segments)
}

/// Resolves an already-split path; `Some(value.clone())` for an empty path.
fn resolve_segments(value: &Value, segments: &[PathSegment]) -> Option<Value> {
    let Some((segment, rest)) = segments.split_first() else {
        return Some(value.clone());
    };
    match segment {
        PathSegment::Index(index) => match value {
            Value::Array(items) => resolve_segments(index_at(items, *index)?, rest),
            _ => None,
        },
        PathSegment::Key(key) => match value {
            Value::Object(map) => resolve_segments(map.get(key)?, rest),
            Value::Array(items) => {
                if let Ok(index) = key.parse::<i64>() {
                    return resolve_segments(index_at(items, index)?, rest);
                }
                let mut projected = Vec::with_capacity(items.len());
                for item in items {
                    if let Some(resolved) = resolve_segments(item, segments) {
                        projected.push(resolved);
                    }
                }
                Some(Value::Array(projected))
            }
            _ => None,
        },
    }
}

/// Returns the element at `index`, counting negative indexes from the end.
fn index_at(items: &[Value], index: i64) -> Option<&Value> {
    if index < 0 {
        let len = i64::try_from(items.len()).ok()?;
        let offset = usize::try_from(len.checked_add(index)?).ok()?;
        items.get(offset)
    } else {
        items.get(usize::try_from(index).ok()?)
    }
}

/// Deeply resolves a JSON template against `source`.
///
/// A string that is exactly a `${...}` reference is replaced by the referenced value
/// (`null` when missing); every other string is a literal. Objects and arrays are
/// walked element-wise, and non-string scalars are returned unchanged.
#[must_use]
pub fn resolve_schema(schema: &Value, source: &Value) -> Value {
    match schema {
        Value::String(text) => match extract_origin_key(text) {
            Some(path) => project(source, path),
            None => schema.clone(),
        },
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| resolve_schema(item, source))
                .collect(),
        ),
        Value::Object(map) => {
            let resolved: Map<String, Value> = map
                .iter()
                .map(|(key, value)| (key.clone(), resolve_schema(value, source)))
                .collect();
            Value::Object(resolved)
        }
        scalar => scalar.clone(),
    }
}

/// Resolves `path` against `value`, mapping a missing path to [`Value::Null`].
#[must_use]
pub fn project(value: &Value, path: &str) -> Value {
    resolve_path(value, path).unwrap_or(Value::Null)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

    fn state() -> Value {
        json!({
            "query": "hello",
            "start": {
                "a": 1,
                "nested": {"b": [10, 20, 30]},
                "items": [{"name": "x", "score": 1}, {"name": "y"}, {"score": 3}],
                "empty": [],
                "a b": "spaced",
                "null_field": null
            }
        })
    }

    #[test]
    fn is_ref_path_matches_whole_strings_only() {
        assert!(is_ref_path("${a}"));
        assert!(is_ref_path("${a.b[0]}"));
        assert!(!is_ref_path("${"));
        assert!(!is_ref_path("${}"));
        assert!(!is_ref_path("a"));
        assert!(!is_ref_path("${a"));
        assert!(!is_ref_path("prefix ${a}"));
        assert!(!is_ref_path("${a} suffix"));
        assert!(!is_ref_path(""));
    }

    #[test]
    fn extract_origin_key_strips_delimiters() {
        assert_eq!(extract_origin_key("${start.a}"), Some("start.a"));
        assert_eq!(extract_origin_key("${a}"), Some("a"));
        assert_eq!(extract_origin_key("a"), None);
        assert_eq!(extract_origin_key("${a}${b}"), None);
        assert_eq!(extract_origin_key("${}"), None);
    }

    #[test]
    fn split_path_parses_dotted_keys_and_indexes() {
        assert_eq!(
            split_path("a.b[0].c").unwrap(),
            vec![
                PathSegment::Key("a".to_string()),
                PathSegment::Key("b".to_string()),
                PathSegment::Index(0),
                PathSegment::Key("c".to_string()),
            ]
        );
        assert_eq!(
            split_path("start.items[-1]").unwrap(),
            vec![
                PathSegment::Key("start".to_string()),
                PathSegment::Key("items".to_string()),
                PathSegment::Index(-1),
            ]
        );
        assert_eq!(split_path("").unwrap(), Vec::new());
        assert_eq!(
            split_path("a.0").unwrap(),
            vec![
                PathSegment::Key("a".to_string()),
                PathSegment::Key("0".to_string())
            ]
        );
    }

    #[test]
    fn split_path_parses_quoted_keys() {
        assert_eq!(
            split_path("a['a b']").unwrap(),
            vec![
                PathSegment::Key("a".to_string()),
                PathSegment::Key("a b".to_string())
            ]
        );
        assert_eq!(
            split_path("a[\"k\"]").unwrap(),
            vec![
                PathSegment::Key("a".to_string()),
                PathSegment::Key("k".to_string())
            ]
        );
        assert_eq!(
            split_path("a[ '0' ]").unwrap(),
            vec![
                PathSegment::Key("a".to_string()),
                PathSegment::Key("0".to_string())
            ]
        );
    }

    #[test]
    fn split_path_rejects_malformed_paths() {
        for path in [
            "a[", "a[]", "a[abc]", "a['k]", "a['']", "a[1.5]", "a['x'y']",
        ] {
            let error = split_path(path).unwrap_err();
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation,
                "path `{path}` should be rejected"
            );
        }
    }

    #[test]
    fn resolve_path_walks_nested_values() {
        let source = state();
        assert_eq!(resolve_path(&source, "query"), Some(json!("hello")));
        assert_eq!(resolve_path(&source, "start.a"), Some(json!(1)));
        assert_eq!(resolve_path(&source, "start.nested.b[1]"), Some(json!(20)));
        assert_eq!(resolve_path(&source, "start.nested.b[-1]"), Some(json!(30)));
        assert_eq!(resolve_path(&source, "start['a b']"), Some(json!("spaced")));
        assert_eq!(
            resolve_path(&source, "start[\"a b\"]"),
            Some(json!("spaced"))
        );
    }

    #[test]
    fn resolve_path_reads_numeric_dotted_segments_from_arrays() {
        let source = state();
        assert_eq!(resolve_path(&source, "start.nested.b.0"), Some(json!(10)));
        assert_eq!(resolve_path(&source, "start.nested.b.-1"), Some(json!(30)));
        assert_eq!(resolve_path(&source, "start.nested.b.5"), None);
        assert_eq!(resolve_path(&source, "start.nested.b.abc"), Some(json!([])));
    }

    #[test]
    fn resolve_path_returns_whole_nodes() {
        let source = state();
        assert_eq!(
            resolve_path(&source, "start"),
            Some(source["start"].clone())
        );
        assert_eq!(resolve_path(&source, ""), Some(source.clone()));
    }

    #[test]
    fn resolve_path_misses_are_none() {
        let source = state();
        assert_eq!(resolve_path(&source, "missing"), None);
        assert_eq!(resolve_path(&source, "start.missing"), None);
        assert_eq!(resolve_path(&source, "start.nested.b[9]"), None);
        assert_eq!(resolve_path(&source, "query.deeper"), None);
        assert_eq!(resolve_path(&Value::Null, "start.a"), None);
        assert_eq!(resolve_path(&source, "start["), None);
    }

    #[test]
    fn resolve_path_keeps_explicit_nulls() {
        let source = state();
        assert_eq!(resolve_path(&source, "start.null_field"), Some(Value::Null));
    }

    #[test]
    fn resolve_path_projects_fields_over_arrays() {
        let source = state();
        assert_eq!(
            resolve_path(&source, "start.items.name"),
            Some(json!(["x", "y"]))
        );
        assert_eq!(
            resolve_path(&source, "start.items.score"),
            Some(json!([1, 3]))
        );
        assert_eq!(
            resolve_path(&source, "start.items"),
            Some(source["start"]["items"].clone())
        );
        assert_eq!(resolve_path(&source, "start.empty.name"), Some(json!([])));
        assert_eq!(
            resolve_path(&source, "start.nested.b.name"),
            Some(json!([]))
        );
    }

    #[test]
    fn resolve_schema_resolves_references_only() {
        let source = state();
        let schema = json!({
            "literal": "plain text",
            "embedded": "prefix ${start.a}",
            "whole": "${start.a}",
            "missing": "${nope}",
            "nested": {"deep": "${start.nested.b[0]}", "keep": 7},
            "list": ["${query}", "literal", null, true],
        });
        let resolved = resolve_schema(&schema, &source);
        assert_eq!(resolved["literal"], json!("plain text"));
        assert_eq!(resolved["embedded"], json!("prefix ${start.a}"));
        assert_eq!(resolved["whole"], json!(1));
        assert_eq!(resolved["missing"], Value::Null);
        assert_eq!(resolved["nested"]["deep"], json!(10));
        assert_eq!(resolved["nested"]["keep"], json!(7));
        assert_eq!(resolved["list"], json!(["hello", "literal", null, true]));
    }

    #[test]
    fn resolve_schema_passes_scalars_through() {
        let source = state();
        for schema in [json!(null), json!(true), json!(3.5), json!(-2), json!([])] {
            assert_eq!(resolve_schema(&schema, &source), schema);
        }
    }

    #[test]
    fn project_maps_misses_to_null() {
        let source = state();
        assert_eq!(project(&source, "start.a"), json!(1));
        assert_eq!(project(&source, "start.missing"), Value::Null);
    }

    #[test]
    fn path_segment_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&PathSegment::Key("a".to_string())).unwrap(),
            "{\"key\":\"a\"}"
        );
        assert_eq!(
            serde_json::to_string(&PathSegment::Index(-1)).unwrap(),
            "{\"index\":-1}"
        );
        let parsed: PathSegment = serde_json::from_str("{\"key\":\"a\"}").unwrap();
        assert_eq!(parsed, PathSegment::Key("a".to_string()));
    }
}
