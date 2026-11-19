//! JSON helpers: deep merging, dotted-path access, flattening, and tolerant
//! message-content extraction.
//!
//! Paths address objects with dot-separated keys and arrays with bracketed
//! indices, e.g. `a.b[0].c`, `items[2]`, or a bare `a`.
//! Flattening follows Python `utils.dict_utils.flatten_dict`: list indices are
//! rendered as `[index]` and appended without a separating dot.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::constants::MAX_COLLECTION_SIZE;
use crate::error::{Error, Result};

/// Largest array index a JSON path may address.
///
/// Paths reach [`set_path`] from user- and model-supplied state keys
/// (`openjiuwen_session::state::set_value`, `openjiuwen_graph::state::set_path`),
/// and a bracketed index of `usize::MAX` would otherwise pad the array with that
/// many [`Value::Null`] entries. The cap is the shared
/// [`MAX_COLLECTION_SIZE`] (Python `MAX_COLLECTION_SIZE`), which leaves ample
/// headroom for any realistic state tree while bounding one padding operation to
/// roughly 3 MiB of `Value`s.
pub const MAX_JSON_INDEX: usize = MAX_COLLECTION_SIZE;

/// Maximum total number of array slots a single path may create by padding.
///
/// Every index segment pads up to its own index, so a crafted path such as
/// `a[99999][99999]...` could otherwise multiply [`MAX_JSON_INDEX`] by the number
/// of index segments. One path may address a very large index or chain several
/// moderately sized ones, but not both: the sum of `index + 1` over all index
/// segments is capped here (about 6 MiB of `Value`s at the limit).
const MAX_JSON_PATH_GROWTH: usize = MAX_JSON_INDEX * 2;

/// Deep-merges `overlay` into `base`.
///
/// When both values are objects, the merge is recursive and key-by-key;
/// otherwise `overlay` replaces `base`. Arrays are replaced, never concatenated.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_common::utils::json::merge;
///
/// let mut base = json!({"a": {"b": 1, "c": 2}});
/// merge(&mut base, json!({"a": {"c": 3}}));
/// assert_eq!(base, json!({"a": {"b": 1, "c": 3}}));
/// ```
pub fn merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(existing) => merge(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

/// Looks up a dotted path inside `value`.
///
/// An empty path returns `value` itself. Missing keys, out-of-range indices,
/// type mismatches, malformed paths, and indices above [`MAX_JSON_INDEX`] all
/// return `None`.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_common::utils::json::get_path;
///
/// let value = json!({"a": {"b": [{"c": 7}]}});
/// assert_eq!(get_path(&value, "a.b[0].c"), Some(&json!(7)));
/// assert_eq!(get_path(&value, "a.missing"), None);
/// ```
#[must_use]
pub fn get_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(value);
    }

    let mut current = value;
    for segment in parse_path(path).ok()? {
        current = match segment {
            PathSegment::Key(key) => current.as_object()?.get(&key)?,
            PathSegment::Index(index) => current.as_array()?.get(index)?,
        };
    }
    Some(current)
}

/// Sets `new_value` at a dotted path inside `value`, creating intermediates.
///
/// Object segments are created as needed; `[index]` segments create arrays,
/// padding them with [`Value::Null`] up to the requested index. Existing values
/// that have the wrong shape are replaced by the required container.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `path` is empty or malformed (empty
/// segment, unterminated index, non-numeric index, or trailing characters after
/// an index), when an index exceeds [`MAX_JSON_INDEX`], or when the path would
/// create more array slots in total than that limit allows. The check happens
/// before anything is written, so a rejected path leaves `value` untouched.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_common::utils::json::set_path;
///
/// let mut value = json!({});
/// set_path(&mut value, "a.b[1].c", json!(7)).unwrap();
/// assert_eq!(value, json!({"a": {"b": [null, {"c": 7}]}}));
/// ```
pub fn set_path(value: &mut Value, path: &str, new_value: Value) -> Result<()> {
    let segments = parse_path(path)?;
    set_segments(value, &segments, new_value)
}

/// Flattens nested objects and arrays into a map of dotted paths to leaves.
///
/// Leaves include falsy values (`0`, `false`, `""`). Empty objects and arrays
/// flatten to their own path with the empty container as the value.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_common::utils::json::flatten;
///
/// let flat = flatten(&json!({"a": {"b": 1}, "c": [true, {"d": ""}]}));
/// assert_eq!(flat.get("a.b"), Some(&json!(1)));
/// assert_eq!(flat.get("c[0]"), Some(&json!(true)));
/// assert_eq!(flat.get("c[1].d"), Some(&json!("")));
/// ```
#[must_use]
pub fn flatten(value: &Value) -> BTreeMap<String, Value> {
    let mut flattened = BTreeMap::new();
    flatten_into(value, "", &mut flattened);
    flattened
}

/// Extracts text from message `content` in any of the shapes the SDK accepts.
///
/// - a string is returned as-is;
/// - an array concatenates the text of its elements with newlines (elements may
///   be plain strings or objects with a string `text`/`content` key);
/// - an object is read from its `text` key, falling back to `content`;
/// - anything else yields `None`.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_common::utils::json::content_to_text;
///
/// let content = json!([{"type": "text", "text": "hello"}, "world"]);
/// assert_eq!(content_to_text(&content).as_deref(), Some("hello\nworld"));
/// ```
#[must_use]
pub fn content_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().filter_map(content_to_text).collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        Value::Object(map) => match map.get("text") {
            Some(Value::String(text)) => Some(text.clone()),
            _ => map.get("content").and_then(content_to_text),
        },
        Value::Null | Value::Bool(_) | Value::Number(_) => None,
    }
}

/// Returns `true` when `value` is a JSON object with no keys.
#[must_use]
pub fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(Map::is_empty)
}

/// One parsed component of a dotted path.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PathSegment {
    /// An object key.
    Key(String),
    /// An array index.
    Index(usize),
}

/// Parses `a.b[0].c` into `[Key("a"), Key("b"), Index(0), Key("c")]`.
fn parse_path(path: &str) -> Result<Vec<PathSegment>> {
    if path.is_empty() {
        return Err(Error::validation("JSON path must not be empty"));
    }

    let mut segments = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            return Err(Error::validation(format!(
                "invalid JSON path `{path}`: empty segment"
            )));
        }

        let (name, rest) = match part.find('[') {
            Some(open) => (&part[..open], &part[open..]),
            None => (part, ""),
        };
        if name.contains(']') {
            return Err(Error::validation(format!(
                "invalid JSON path `{path}`: unexpected `]`"
            )));
        }
        if !name.is_empty() {
            segments.push(PathSegment::Key(name.to_string()));
        }
        parse_indices(path, rest, &mut segments)?;
    }

    if segments.is_empty() {
        return Err(Error::validation(format!(
            "invalid JSON path `{path}`: no segments"
        )));
    }
    let growth = total_growth(&segments);
    if growth > MAX_JSON_PATH_GROWTH {
        return Err(Error::validation(format!(
            "invalid JSON path `{path}`: padding the array to these indices would create \
             {growth} entries, exceeding the maximum of {MAX_JSON_PATH_GROWTH}"
        )));
    }
    Ok(segments)
}

/// Sums the `index + 1` slot counts a path implies, saturating at `usize::MAX`.
fn total_growth(segments: &[PathSegment]) -> usize {
    segments
        .iter()
        .fold(0_usize, |total, segment| match segment {
            PathSegment::Key(_) => total,
            PathSegment::Index(index) => total.saturating_add(index.saturating_add(1)),
        })
}

/// Parses the `[n][m]...` suffix of a dotted path part.
fn parse_indices(path: &str, mut rest: &str, segments: &mut Vec<PathSegment>) -> Result<()> {
    while !rest.is_empty() {
        let Some(close) = rest.find(']') else {
            return Err(Error::validation(format!(
                "invalid JSON path `{path}`: unterminated index"
            )));
        };
        let digits = &rest[1..close];
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(Error::validation(format!(
                "invalid JSON path `{path}`: bad index `{digits}`"
            )));
        }
        let index = digits.parse::<usize>().map_err(|_| {
            Error::validation(format!(
                "invalid JSON path `{path}`: index `{digits}` out of range"
            ))
        })?;
        if index > MAX_JSON_INDEX {
            return Err(Error::validation(format!(
                "invalid JSON path `{path}`: array index {index} exceeds the maximum of \
                 {MAX_JSON_INDEX}"
            )));
        }
        segments.push(PathSegment::Index(index));

        rest = &rest[close + 1..];
        if !rest.is_empty() && !rest.starts_with('[') {
            return Err(Error::validation(format!(
                "invalid JSON path `{path}`: trailing characters after index"
            )));
        }
    }
    Ok(())
}

/// Writes `new_value` at `segments`, creating containers along the way.
///
/// Indices are validated by [`parse_path`] before this runs; the guards here only
/// make a future caller that bypasses the parser fail with a typed error instead
/// of allocating unboundedly.
fn set_segments(target: &mut Value, segments: &[PathSegment], new_value: Value) -> Result<()> {
    let Some((first, rest)) = segments.split_first() else {
        *target = new_value;
        return Ok(());
    };

    match first {
        PathSegment::Key(key) => {
            if !target.is_object() {
                *target = Value::Object(Map::new());
            }
            let Some(map) = target.as_object_mut() else {
                return Ok(());
            };
            let entry = map.entry(key.clone()).or_insert(Value::Null);
            set_segments(entry, rest, new_value)
        }
        PathSegment::Index(index) => {
            if *index > MAX_JSON_INDEX {
                return Err(Error::validation(format!(
                    "cannot address JSON array index {index}: the maximum is {MAX_JSON_INDEX}"
                )));
            }
            if !target.is_array() {
                *target = Value::Array(Vec::new());
            }
            let Some(items) = target.as_array_mut() else {
                return Ok(());
            };
            let needed = index.saturating_add(1);
            if needed > items.len() {
                // Reserve up front so an allocation failure is a typed error
                // rather than an abort, and so `resize` cannot reallocate.
                items.try_reserve(needed - items.len()).map_err(|err| {
                    Error::validation(format!("cannot pad JSON array to index {index}: {err}"))
                })?;
                items.resize(needed, Value::Null);
            }
            let Some(slot) = items.get_mut(*index) else {
                return Ok(());
            };
            set_segments(slot, rest, new_value)
        }
    }
}

/// Recursively collects leaves of `value` into `out`, prefixed by `prefix`.
fn flatten_into(value: &Value, prefix: &str, out: &mut BTreeMap<String, Value>) {
    match value {
        Value::Object(map) if !map.is_empty() => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_into(child, &path, out);
            }
        }
        Value::Array(items) if !items.is_empty() => {
            for (index, child) in items.iter().enumerate() {
                let path = format!("{prefix}[{index}]");
                flatten_into(child, &path, out);
            }
        }
        leaf => {
            out.insert(prefix.to_string(), leaf.clone());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    // port of tests/unit_tests/core/common/utils/test_dict_utils.py

    #[test]
    fn merge_is_recursive_for_objects() {
        let mut base = json!({"a": {"b": 1, "c": 2}, "d": 3});
        merge(&mut base, json!({"a": {"c": 20, "e": 30}, "f": {"g": 4}}));
        assert_eq!(
            base,
            json!({"a": {"b": 1, "c": 20, "e": 30}, "d": 3, "f": {"g": 4}})
        );
    }

    #[test]
    fn merge_replaces_arrays_and_scalars() {
        let mut base = json!({"a": [1, 2, 3]});
        merge(&mut base, json!({"a": [9]}));
        assert_eq!(base, json!({"a": [9]}));

        let mut base = json!({"a": {"b": 1}});
        merge(&mut base, json!({"a": 5}));
        assert_eq!(base, json!({"a": 5}));

        let mut base = json!("old");
        merge(&mut base, json!({"new": true}));
        assert_eq!(base, json!({"new": true}));
    }

    #[test]
    fn merge_into_null_creates_the_overlay() {
        let mut base = Value::Null;
        merge(&mut base, json!({"a": 1}));
        assert_eq!(base, json!({"a": 1}));
    }

    #[test]
    fn get_path_reads_objects_and_arrays() {
        let value = json!({"a": {"b": [{"c": 7}, 8]}, "items": [0, 1, 2]});

        assert_eq!(get_path(&value, ""), Some(&value));
        assert_eq!(get_path(&value, "a"), Some(&json!({"b": [{"c": 7}, 8]})));
        assert_eq!(get_path(&value, "a.b[0].c"), Some(&json!(7)));
        assert_eq!(get_path(&value, "a.b[1]"), Some(&json!(8)));
        assert_eq!(get_path(&value, "items[2]"), Some(&json!(2)));
    }

    #[test]
    fn get_path_returns_none_for_missing_or_malformed() {
        let value = json!({"a": {"b": [1]}, "s": "text"});

        assert_eq!(get_path(&value, "missing"), None);
        assert_eq!(get_path(&value, "a.b[5]"), None);
        assert_eq!(get_path(&value, "s[0]"), None);
        assert_eq!(get_path(&value, "a.b.c"), None);
        assert_eq!(get_path(&value, "a..b"), None);
        assert_eq!(get_path(&value, "a["), None);
        assert_eq!(get_path(&value, "a[x]"), None);
    }

    #[test]
    fn set_path_creates_intermediate_objects_and_arrays() {
        let mut value = json!({});
        set_path(&mut value, "a.b[1].c", json!(7)).unwrap();
        assert_eq!(value, json!({"a": {"b": [null, {"c": 7}]}}));

        let mut value = json!({"a": {"existing": true}});
        set_path(&mut value, "a.new", json!("v")).unwrap();
        assert_eq!(value, json!({"a": {"existing": true, "new": "v"}}));
    }

    #[test]
    fn set_path_replaces_wrong_shapes() {
        let mut value = json!({"a": 1, "b": {"c": 2}});
        set_path(&mut value, "a.x", json!(3)).unwrap();
        set_path(&mut value, "b[0]", json!(4)).unwrap();
        assert_eq!(value, json!({"a": {"x": 3}, "b": [4]}));
    }

    #[test]
    fn set_path_and_get_path_round_trip() {
        let mut value = Value::Null;
        set_path(&mut value, "root.items[0].name", json!("first")).unwrap();
        set_path(&mut value, "root.items[2].name", json!("third")).unwrap();
        set_path(&mut value, "root.count", json!(3)).unwrap();

        assert_eq!(
            get_path(&value, "root.items[0].name"),
            Some(&json!("first"))
        );
        assert_eq!(
            get_path(&value, "root.items[2].name"),
            Some(&json!("third"))
        );
        assert_eq!(get_path(&value, "root.count"), Some(&json!(3)));
        assert_eq!(get_path(&value, "root.items[1]"), Some(&Value::Null));
    }

    #[test]
    fn set_path_rejects_invalid_paths() {
        let mut value = json!({});
        for path in ["", ".a", "a.", "a..b", "a[", "a[]", "a[x]", "a[0]x", "a]"] {
            let error = set_path(&mut value, path, json!(1)).unwrap_err();
            assert_eq!(
                error.kind(),
                crate::error::ErrorKind::Validation,
                "path: `{path}`"
            );
        }
        assert_eq!(value, json!({}), "invalid paths must not mutate the input");
    }

    #[test]
    fn set_path_rejects_an_oversized_index_without_allocating() {
        let mut value = json!({});
        let error = set_path(&mut value, "a[1000000000]", json!(1)).unwrap_err();

        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        let message = error.to_string();
        assert!(message.contains("1000000000"), "message: {message}");
        assert!(
            message.contains(&MAX_JSON_INDEX.to_string()),
            "the message names the limit: {message}"
        );
        assert_eq!(
            value,
            json!({}),
            "a rejected path must not mutate the input"
        );

        // `usize::MAX` is rejected by the same guard rather than overflowing.
        let error = set_path(&mut value, &format!("a[{}]", usize::MAX), json!(1)).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert_eq!(value, json!({}));
    }

    #[test]
    fn set_path_rejects_indices_that_would_pad_the_whole_path_too_far() {
        // Each index is legal on its own; together they would pad two arrays past
        // the cumulative limit, so the path is rejected before any work is done.
        let path = format!("a[{MAX_JSON_INDEX}][{MAX_JSON_INDEX}]");
        let mut value = json!({});
        let error = set_path(&mut value, &path, json!(1)).unwrap_err();

        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains(&MAX_JSON_PATH_GROWTH.to_string()),
            "the message names the limit: {error}"
        );
        assert_eq!(
            value,
            json!({}),
            "a rejected path must not mutate the input"
        );
    }

    #[test]
    fn set_path_accepts_a_large_but_sane_index() {
        let mut value = json!({});
        set_path(&mut value, "items[1000]", json!("end")).unwrap();

        let items = value["items"].as_array().unwrap();
        assert_eq!(items.len(), 1001);
        assert!(items[..1000].iter().all(Value::is_null));
        assert_eq!(items[1000], json!("end"));

        // The documented cap itself is inclusive.
        let mut value = json!({});
        set_path(&mut value, &format!("at[{MAX_JSON_INDEX}]"), json!(1)).unwrap();
        let items = value["at"].as_array().unwrap();
        assert_eq!(items.len(), MAX_JSON_INDEX + 1);
        assert_eq!(items[MAX_JSON_INDEX], json!(1));
    }

    #[test]
    fn get_path_ignores_an_oversized_index() {
        let value = json!({"items": [1, 2]});
        assert_eq!(get_path(&value, "items[1000000000]"), None);
        assert_eq!(get_path(&value, "items[1]"), Some(&json!(2)));
    }

    #[test]
    fn flatten_collects_leaves_including_falsy_values() {
        let value = json!({
            "a": {"b": 1, "c": [0, false, ""]},
            "d": "text",
            "e": {},
            "f": [],
        });
        let flat = flatten(&value);

        assert_eq!(flat.get("a.b"), Some(&json!(1)));
        assert_eq!(flat.get("a.c[0]"), Some(&json!(0)));
        assert_eq!(flat.get("a.c[1]"), Some(&json!(false)));
        assert_eq!(flat.get("a.c[2]"), Some(&json!("")));
        assert_eq!(flat.get("d"), Some(&json!("text")));
        assert_eq!(flat.get("e"), Some(&json!({})));
        assert_eq!(flat.get("f"), Some(&json!([])));
        assert_eq!(flat.len(), 7);
    }

    #[test]
    fn flatten_handles_scalars_and_nested_arrays() {
        assert_eq!(
            flatten(&json!(5)),
            BTreeMap::from([(String::new(), json!(5))])
        );

        let flat = flatten(&json!({"a": [[1], {"b": 2}]}));
        assert_eq!(flat.get("a[0][0]"), Some(&json!(1)));
        assert_eq!(flat.get("a[1].b"), Some(&json!(2)));
    }

    #[test]
    fn content_to_text_accepts_strings_and_parts() {
        assert_eq!(content_to_text(&json!("hello")).as_deref(), Some("hello"));
        assert_eq!(
            content_to_text(&json!([{"type": "text", "text": "hello"}, "world"])).as_deref(),
            Some("hello\nworld")
        );
        assert_eq!(
            content_to_text(&json!([{"content": "a"}, {"content": [{"text": "b"}]}])).as_deref(),
            Some("a\nb")
        );
        assert_eq!(
            content_to_text(&json!({"text": "direct"})).as_deref(),
            Some("direct")
        );
        assert_eq!(
            content_to_text(&json!({"content": "nested"})).as_deref(),
            Some("nested")
        );
        assert_eq!(
            content_to_text(&json!({"content": [{"text": "inner"}]})).as_deref(),
            Some("inner")
        );
    }

    #[test]
    fn content_to_text_returns_none_for_unsupported_shapes() {
        assert_eq!(content_to_text(&json!(null)), None);
        assert_eq!(content_to_text(&json!(42)), None);
        assert_eq!(content_to_text(&json!([])), None);
        assert_eq!(content_to_text(&json!([1, true, null])), None);
        assert_eq!(content_to_text(&json!({})), None);
        assert_eq!(content_to_text(&json!({"text": 1})), None);
        assert_eq!(content_to_text(&json!({"text": null, "content": 2})), None);
    }

    #[test]
    fn is_empty_object_only_matches_empty_objects() {
        assert!(is_empty_object(&json!({})));
        assert!(!is_empty_object(&json!({"a": 1})));
        assert!(!is_empty_object(&json!([])));
        assert!(!is_empty_object(&json!(null)));
        assert!(!is_empty_object(&json!("")));
    }
}
