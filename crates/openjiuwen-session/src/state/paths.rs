//! Nested-path helpers for session state.
//!
//! Paths address objects with dot-separated keys and arrays with bracketed
//! indices (`a.b[0].c`), matching `openjiuwen/core/session/utils.py`. Lookup and
//! write delegate to [`openjiuwen_common::utils::json`]'s path implementation so
//! the two crates agree on path syntax. [`apply_patch`] ports Python's
//! `update_dict` merge semantics.

use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::json::{get_path, set_path};

/// One parsed component of a nested path.
enum Segment {
    /// An object key.
    Key(String),
    /// An array index.
    Index(usize),
}

/// Returns a reference to the value at `path` inside `value`.
///
/// An empty path returns `value` itself. Missing keys, out-of-range indices,
/// type mismatches, and malformed paths yield `None`.
#[must_use]
pub fn get_value<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    get_path(value, path)
}

/// Returns a deep copy of the value at `path`, or `None` when it is absent.
#[must_use]
pub fn get_value_cloned(value: &Value, path: &str) -> Option<Value> {
    get_path(value, path).cloned()
}

/// Returns the deep subtree rooted at `path`.
///
/// An empty path returns the whole value; a missing path returns
/// [`Value::Null`].
#[must_use]
pub fn get_by_prefix(value: &Value, path: &str) -> Value {
    get_value_cloned(value, path).unwrap_or(Value::Null)
}

/// Writes `new_value` at `path`, creating intermediate containers.
///
/// Delegates to [`openjiuwen_common::utils::json::set_path`], so object
/// segments are created as needed and `[index]` segments pad arrays with
/// [`Value::Null`].
///
/// # Errors
///
/// Returns [`Error::Validation`] when `path` is empty or malformed.
pub fn set_value(value: &mut Value, path: &str, new_value: Value) -> Result<()> {
    set_path(value, path, new_value)
}

/// Removes the value at `path`, returning whether something was removed.
///
/// Object keys are deleted from their parent object; array entries are removed
/// from their parent array (shifting later elements). A missing path, an
/// out-of-range index, or a malformed path returns `false` and leaves `value`
/// untouched.
#[must_use]
pub fn remove_value(value: &mut Value, path: &str) -> bool {
    let Some(segments) = parse_segments(path) else {
        return false;
    };
    remove_at(value, &segments)
}

/// Merges `patch` into `value` using Python `update_dict` semantics.
///
/// `patch` keys may themselves be dotted/bracketed paths (`a.b[0].c`), resolved
/// relative to `value`. For each key:
///
/// - a `null` value marks the path for deletion; deletions are applied after
///   all writes so a patch may write and then delete in the same call;
/// - an object value whose current target is also an object is merged
///   recursively (inner keys are again nested paths);
/// - anything else replaces the target wholesale with a deep copy. Arrays are
///   never element-merged.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `patch` is not a JSON object or when a
/// key is not a valid nested path.
pub fn apply_patch(value: &mut Value, patch: &Value) -> Result<()> {
    let Some(patch) = patch.as_object() else {
        return Err(Error::validation("state patch must be a JSON object"));
    };
    apply_object(value, patch)
}

/// Applies one object level of [`apply_patch`], recursing into object targets.
fn apply_object(value: &mut Value, patch: &Map<String, Value>) -> Result<()> {
    let mut removals: Vec<&str> = Vec::new();
    for (key, patch_value) in patch {
        if patch_value.is_null() {
            removals.push(key);
            continue;
        }

        let target_is_object = get_value(value, key).is_some_and(Value::is_object);
        if patch_value.is_object() && target_is_object {
            let mut nested = get_value(value, key).cloned().unwrap_or(Value::Null);
            if let Some(nested_patch) = patch_value.as_object() {
                apply_object(&mut nested, nested_patch)?;
            }
            set_value(value, key, nested)?;
        } else {
            set_value(value, key, patch_value.clone())?;
        }
    }

    for key in removals {
        let _removed = remove_value(value, key);
    }
    Ok(())
}

/// Returns `true` when `path` is a `${...}` reference with a non-empty body.
#[must_use]
pub fn is_ref_path(path: &str) -> bool {
    path.len() > 3 && path.starts_with("${") && path.ends_with('}')
}

/// Extracts the origin key from a `${...}` reference.
///
/// The first well-formed reference anywhere in `key` wins, mirroring Python's
/// `REF_PATTERN` search. Returns `None` for non-reference or malformed input
/// (for example `"start123.p2"` or `"${unclosed"`).
#[must_use]
pub fn extract_origin_key(key: &str) -> Option<String> {
    let mut search_from = 0;
    while let Some(offset) = key.get(search_from..)?.find("${") {
        let content_start = search_from + offset + 2;
        let rest = key.get(content_start..)?;
        match rest.find(['}', '{']) {
            Some(end) if rest.as_bytes().get(end) == Some(&b'}') => {
                return rest.get(..end).map(ToString::to_string);
            }
            Some(end) => search_from = content_start + end,
            None => return None,
        }
    }
    None
}

/// Parses a nested path into object keys and array indices.
fn parse_segments(path: &str) -> Option<Vec<Segment>> {
    if path.is_empty() {
        return None;
    }

    let mut segments = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            return None;
        }

        let (name, mut rest) = match part.find('[') {
            Some(open) => (&part[..open], &part[open..]),
            None => (part, ""),
        };
        if name.contains(']') {
            return None;
        }
        if !name.is_empty() {
            segments.push(Segment::Key(name.to_string()));
        }

        while !rest.is_empty() {
            let close = rest.find(']')?;
            let digits = rest.get(1..close)?;
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            segments.push(Segment::Index(digits.parse().ok()?));
            rest = rest.get(close + 1..)?;
            if !rest.is_empty() && !rest.starts_with('[') {
                return None;
            }
        }
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments)
    }
}

/// Removes the value addressed by `segments` from `target`.
fn remove_at(target: &mut Value, segments: &[Segment]) -> bool {
    let Some((last, parents)) = segments.split_last() else {
        return false;
    };

    let mut current = target;
    for segment in parents {
        current = match segment {
            Segment::Key(key) => match current.as_object_mut().and_then(|map| map.get_mut(key)) {
                Some(value) => value,
                None => return false,
            },
            Segment::Index(index) => match current
                .as_array_mut()
                .and_then(|items| items.get_mut(*index))
            {
                Some(value) => value,
                None => return false,
            },
        };
    }

    match last {
        Segment::Key(key) => current
            .as_object_mut()
            .and_then(|map| map.remove(key))
            .is_some(),
        Segment::Index(index) => {
            let Some(items) = current.as_array_mut() else {
                return false;
            };
            if *index < items.len() {
                items.remove(*index);
                true
            } else {
                false
            }
        }
    }
}
