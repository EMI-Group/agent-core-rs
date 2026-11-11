//! The shared filter language used by the document and graph storage backends.
//!
//! A filter is a JSON value and a document is any JSON value. [`matches`](fn@matches)
//! resolves field paths with [`openjiuwen_common::utils::json::get_path`], so
//! dotted keys (`meta.owner`) and array indices (`items[1].kind`) work the same
//! way in every backend.
//!
//! # Shape of a filter
//!
//! - `null` or `{}` matches everything; any other non-object filter matches
//!   nothing.
//! - A plain `field: condition` pair is implicit equality when `condition` is a
//!   scalar, array, or an object without `$`-prefixed keys.
//! - An object `condition` that contains at least one `$`-prefixed key is an
//!   operator condition: every entry must hold (logical AND). A non-operator
//!   entry inside such an object is an unknown operator, so the condition fails.
//! - Top-level `$and` / `$or` take an array of sub-filters (`$and` of an empty
//!   array is true, `$or` of an empty array is false); `$not` takes a sub-filter
//!   object. They combine with sibling field conditions using AND.
//!
//! A key literally named `$and`, `$or`, or `$not` at the top level is therefore
//! a combinator, not a field; every other top-level key is a field path, even
//! when it starts with `$` (a top-level `$eq` looks up a field called `$eq`).
//! Inside an operator object, `$`-prefixed keys are operators and other keys are
//! unknown operators, so a field whose value is an object containing `$`-keys
//! cannot be matched by plain equality. Field names are otherwise free-form.
//!
//! # Operators
//!
//! - `$eq` — JSON equality, with numeric awareness so `1 == 1.0`. A missing
//!   field never satisfies `$eq`, not even `$eq: null` (use `$eq` only on
//!   fields that exist; a literal `null` is matched with implicit equality on a
//!   field that is present and null).
//! - `$ne` — negation of `$eq`; a missing field satisfies `$ne`.
//! - `$gt`, `$gte`, `$lt`, `$lte` — numeric ordering when both sides are
//!   numbers, lexicographic (Rust `str` ordering) when both are strings, and
//!   false for every other combination (booleans, null, arrays, mixed types
//!   never order).
//! - `$in` — the expected value must be an array. A document array matches when
//!   it contains any candidate; any other document value matches when it equals
//!   a candidate. Missing fields never match.
//! - `$contains` — for an array document value, an array of expected values
//!   matches when any candidate is contained; for a string document value, a
//!   string expectation is a substring test. Every other shape is false.
//!
//! Unknown operators and malformed conditions (`$and` that is not an array,
//! `$in` without an array, `$contains` of a number, ...) make their condition
//! false; [`matches`](fn@matches) never panics, whatever the input shape.

use std::cmp::Ordering;

use serde_json::{Map, Value};

use openjiuwen_common::utils::json::get_path;

const AND: &str = "$and";
const OR: &str = "$or";
const NOT: &str = "$not";

/// Returns `true` when `document` satisfies `filter`.
///
/// A `null` or empty-object filter matches every document; a non-object filter
/// matches none. See the [module docs](self) for the supported operators and
/// their type rules.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_foundation::store::filter::matches;
///
/// let document = json!({"name": "Ada", "age": 36, "tags": ["rust", "storage"]});
///
/// assert!(matches(&json!({}), &document));
/// assert!(matches(&json!({"age": {"$gte": 18, "$lt": 65}}), &document));
/// assert!(matches(&json!({"tags": {"$contains": ["rust"]}}), &document));
/// assert!(matches(&json!({"name": {"$contains": "d"}}), &document));
/// assert!(!matches(&json!({"age": {"$gt": 40}}), &document));
/// assert!(!matches(&json!({"missing": null}), &document));
/// ```
pub fn matches(filter: &Value, document: &Value) -> bool {
    match filter {
        Value::Null => true,
        Value::Object(entries) => object_matches(entries, document),
        _ => false,
    }
}

/// Returns `true` when every filter in `filters` matches `document`.
///
/// An empty slice matches everything. Each filter is evaluated independently
/// with [`matches`](fn@matches), so combining filters is equivalent to `$and`.
///
/// ```
/// use serde_json::json;
/// use openjiuwen_foundation::store::filter::matches_all;
///
/// let document = json!({"age": 36, "name": "Ada"});
///
/// assert!(matches_all(&[], &document));
/// assert!(matches_all(&[json!({"age": 36}), json!({"name": "Ada"})], &document));
/// assert!(!matches_all(&[json!({"age": 36}), json!({"name": "Bob"})], &document));
/// ```
pub fn matches_all(filters: &[Value], document: &Value) -> bool {
    filters.iter().all(|filter| matches(filter, document))
}

/// Evaluates every entry of a filter object with AND.
fn object_matches(entries: &Map<String, Value>, document: &Value) -> bool {
    entries.iter().all(|(key, condition)| match key.as_str() {
        AND => and_matches(condition, document),
        OR => or_matches(condition, document),
        NOT => not_matches(condition, document),
        path => field_matches(document, path, condition),
    })
}

/// `$and`: every sub-filter must match; a non-array (or empty array) is handled
/// literally (`false` for non-array, `true` for empty).
fn and_matches(condition: &Value, document: &Value) -> bool {
    condition
        .as_array()
        .is_some_and(|filters| filters.iter().all(|filter| matches(filter, document)))
}

/// `$or`: any sub-filter must match; an empty array matches nothing.
fn or_matches(condition: &Value, document: &Value) -> bool {
    condition
        .as_array()
        .is_some_and(|filters| filters.iter().any(|filter| matches(filter, document)))
}

/// `$not`: negates a sub-filter object; a non-object condition is malformed.
fn not_matches(condition: &Value, document: &Value) -> bool {
    condition.is_object() && !matches(condition, document)
}

/// Resolves one field path and applies either an operator object or equality.
fn field_matches(document: &Value, path: &str, condition: &Value) -> bool {
    let found = get_path(document, path);
    if let Value::Object(operators) = condition {
        if operators.keys().any(|key| key.starts_with('$')) {
            return operators_match(found, operators);
        }
    }
    found.is_some_and(|value| equals(value, condition))
}

/// Applies every operator of a condition object with AND.
fn operators_match(found: Option<&Value>, operators: &Map<String, Value>) -> bool {
    operators
        .iter()
        .all(|(operator, expected)| apply_operator(found, operator, expected))
}

/// Applies a single operator; unknown or malformed operators are false.
fn apply_operator(found: Option<&Value>, operator: &str, expected: &Value) -> bool {
    match operator {
        "$eq" => found.is_some_and(|value| equals(value, expected)),
        "$ne" => found.is_none_or(|value| !equals(value, expected)),
        "$gt" => compare(found, expected) == Some(Ordering::Greater),
        "$gte" => matches!(
            compare(found, expected),
            Some(Ordering::Greater | Ordering::Equal)
        ),
        "$lt" => compare(found, expected) == Some(Ordering::Less),
        "$lte" => matches!(
            compare(found, expected),
            Some(Ordering::Less | Ordering::Equal)
        ),
        "$in" => in_matches(found, expected),
        "$contains" => contains_matches(found, expected),
        _ => false,
    }
}

/// Orders two values when they are both numbers or both strings.
fn compare(found: Option<&Value>, expected: &Value) -> Option<Ordering> {
    match (found?, expected) {
        (Value::Number(left), Value::Number(right)) => left.as_f64()?.partial_cmp(&right.as_f64()?),
        (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

/// `$in`: candidate membership for scalar or array document values.
fn in_matches(found: Option<&Value>, expected: &Value) -> bool {
    let Some(candidates) = expected.as_array() else {
        return false;
    };
    let Some(found) = found else {
        return false;
    };
    match found {
        Value::Array(items) => items
            .iter()
            .any(|item| candidates.iter().any(|candidate| equals(item, candidate))),
        value => candidates.iter().any(|candidate| equals(value, candidate)),
    }
}

/// `$contains`: array containment for arrays, substring for strings.
fn contains_matches(found: Option<&Value>, expected: &Value) -> bool {
    match (expected, found) {
        (Value::Array(needles), Some(Value::Array(items))) => items
            .iter()
            .any(|item| needles.iter().any(|needle| equals(item, needle))),
        (Value::String(needle), Some(Value::String(text))) => text.contains(needle.as_str()),
        _ => false,
    }
}

/// JSON equality with numeric awareness, so `1` equals `1.0`.
///
/// Numbers are compared as `f64`; integers beyond the exact `f64` range may
/// compare equal when they round to the same float.
fn equals(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left_number), Value::Number(right_number)) => {
            match (left_number.as_f64(), right_number.as_f64()) {
                (Some(left_float), Some(right_float)) => {
                    left_float.partial_cmp(&right_float) == Some(Ordering::Equal)
                }
                _ => left_number == right_number,
            }
        }
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::{matches, matches_all};

    fn document() -> Value {
        json!({
            "id": 1,
            "name": "Ada",
            "age": 36,
            "score": 1.0,
            "active": true,
            "tags": ["rust", "storage"],
            "meta": {"owner": "ada", "nested": {"depth": 2}},
            "items": [{"kind": "a", "value": 1}, {"kind": "b", "value": 2}],
            "nothing": null
        })
    }

    #[test]
    fn null_and_empty_filters_match_everything() {
        for document in [document(), json!(null), json!([1, 2, 3]), json!("scalar")] {
            assert!(matches(&Value::Null, &document), "null filter: {document}");
            assert!(matches(&json!({}), &document), "empty filter: {document}");
        }
    }

    #[test]
    fn non_object_filters_match_nothing() {
        for filter in [
            json!(1),
            json!("name"),
            json!(true),
            json!([]),
            json!([{"name": "Ada"}]),
        ] {
            assert!(!matches(&filter, &document()), "filter: {filter}");
        }
    }

    #[test]
    fn implicit_equality_covers_present_and_missing_fields() {
        assert!(matches(&json!({"name": "Ada", "age": 36}), &document()));
        assert!(!matches(&json!({"name": "Bob"}), &document()));
        assert!(
            !matches(&json!({"missing": null}), &document()),
            "a missing field never matches, not even against null"
        );
        assert!(matches(&json!({"nothing": null}), &document()));
        assert!(matches(&json!({"active": true}), &document()));
        assert!(matches(&json!({"tags": ["rust", "storage"]}), &document()));
        assert!(!matches(&json!({"tags": ["rust"]}), &document()));
    }

    #[test]
    fn dotted_paths_and_array_indices_are_resolved() {
        let value = document();
        assert!(matches(&json!({"meta.owner": "ada"}), &value));
        assert!(matches(&json!({"meta.nested.depth": 2}), &value));
        assert!(matches(&json!({"items[1].kind": "b"}), &value));
        assert!(matches(&json!({"tags[0]": "rust"}), &value));
        assert!(matches(&json!({"id": 1}), &value));
        assert!(!matches(&json!({"items[5].kind": "b"}), &value));
        assert!(!matches(&json!({"meta..depth": 2}), &value));
        assert!(!matches(&json!({"meta.owner": "bob"}), &value));
        let scalar = json!(7);
        assert!(!matches(&json!({"id": 1}), &scalar));
    }

    #[test]
    fn eq_operator_compares_numerically() {
        let value = document();
        assert!(matches(&json!({"age": {"$eq": 36}}), &value));
        assert!(matches(&json!({"score": {"$eq": 1}}), &value));
        assert!(matches(&json!({"age": {"$eq": 36.0}}), &value));
        assert!(!matches(&json!({"age": {"$eq": 37}}), &value));
        assert!(
            !matches(&json!({"age": {"$eq": "36"}}), &value),
            "string and number never compare equal"
        );
        assert!(!matches(&json!({"missing": {"$eq": null}}), &value));
        assert!(!matches(&json!({"age": {"$eq": null}}), &value));
        assert!(matches(&json!({"active": {"$eq": true}}), &value));
        assert!(matches(&json!({"nothing": {"$eq": null}}), &value));
        assert!(matches(
            &json!({"meta": {"$eq": {"owner": "ada", "nested": {"depth": 2}}}}),
            &value
        ));
    }

    #[test]
    fn ne_operator_negates_eq_and_matches_missing_fields() {
        let value = document();
        assert!(matches(&json!({"age": {"$ne": 37}}), &value));
        assert!(!matches(&json!({"age": {"$ne": 36}}), &value));
        assert!(!matches(&json!({"age": {"$ne": 36.0}}), &value));
        assert!(matches(&json!({"missing": {"$ne": 1}}), &value));
        assert!(matches(&json!({"missing": {"$ne": null}}), &value));
        assert!(!matches(&json!({"active": {"$ne": true}}), &value));
        assert!(!matches(&json!({"nothing": {"$ne": null}}), &value));
    }

    #[test]
    fn ordered_operators_compare_numbers_and_strings() {
        let value = document();
        assert!(matches(&json!({"age": {"$gt": 35}}), &value));
        assert!(!matches(&json!({"age": {"$gt": 36}}), &value));
        assert!(matches(&json!({"age": {"$gt": 35.5}}), &value));
        assert!(matches(&json!({"age": {"$gte": 36}}), &value));
        assert!(matches(&json!({"age": {"$gte": 36.0}}), &value));
        assert!(!matches(&json!({"age": {"$gte": 37}}), &value));
        assert!(matches(&json!({"age": {"$lt": 36.5}}), &value));
        assert!(!matches(&json!({"age": {"$lt": 36}}), &value));
        assert!(matches(&json!({"age": {"$lte": 36}}), &value));
        assert!(!matches(&json!({"age": {"$lte": 35}}), &value));

        assert!(matches(&json!({"name": {"$gt": "Abc"}}), &value));
        assert!(!matches(&json!({"name": {"$gt": "Ada"}}), &value));
        assert!(matches(&json!({"name": {"$lt": "Ada2"}}), &value));
        assert!(!matches(&json!({"name": {"$lt": "A"}}), &value));
        assert!(matches(&json!({"name": {"$gte": "Ada"}}), &value));
        assert!(matches(&json!({"name": {"$lte": "Ada"}}), &value));

        assert!(!matches(&json!({"score": {"$gt": "a"}}), &value));
        assert!(!matches(&json!({"active": {"$gt": false}}), &value));
        assert!(!matches(&json!({"nothing": {"$gte": null}}), &value));
        assert!(!matches(&json!({"missing": {"$lt": 10}}), &value));
        assert!(!matches(&json!({"tags": {"$gt": []}}), &value));
    }

    #[test]
    fn in_operator_matches_scalars_and_array_documents() {
        let value = document();
        assert!(matches(&json!({"age": {"$in": [20, 36, 50]}}), &value));
        assert!(!matches(&json!({"age": {"$in": [20, 50]}}), &value));
        assert!(matches(&json!({"score": {"$in": [1]}}), &value));
        assert!(matches(&json!({"tags": {"$in": ["storage"]}}), &value));
        assert!(!matches(&json!({"tags": {"$in": ["python"]}}), &value));
        assert!(!matches(&json!({"tags": {"$in": []}}), &value));
        assert!(
            matches(
                &json!({"items": {"$in": [{"kind": "b", "value": 2}]}}),
                &value
            ),
            "an array document value matches when it contains a candidate"
        );
        assert!(!matches(&json!({"age": {"$in": 36}}), &value));
        assert!(!matches(&json!({"age": {"$in": null}}), &value));
        assert!(!matches(&json!({"missing": {"$in": [1]}}), &value));
    }

    #[test]
    fn contains_operator_handles_arrays_and_strings() {
        let value = document();
        assert!(matches(&json!({"tags": {"$contains": ["rust"]}}), &value));
        assert!(!matches(
            &json!({"tags": {"$contains": ["python"]}}),
            &value
        ));
        assert!(matches(
            &json!({"tags": {"$contains": ["python", "rust"]}}),
            &value
        ));
        assert!(!matches(&json!({"tags": {"$contains": []}}), &value));
        assert!(matches(
            &json!({"items": {"$contains": [{"kind": "a", "value": 1}]}}),
            &value
        ));
        assert!(
            !matches(&json!({"tags": {"$contains": "rust"}}), &value),
            "a scalar expectation only substring-matches a string field"
        );
        assert!(matches(&json!({"name": {"$contains": "d"}}), &value));
        assert!(!matches(&json!({"name": {"$contains": "ada"}}), &value));
        assert!(!matches(&json!({"age": {"$contains": "3"}}), &value));
        assert!(!matches(&json!({"tags": {"$contains": 1}}), &value));
        assert!(!matches(&json!({"missing": {"$contains": "x"}}), &value));
        assert!(!matches(&json!({"nothing": {"$contains": "x"}}), &value));
    }

    #[test]
    fn numeric_equality_applies_inside_in_and_contains() {
        let value = json!({"numbers": [1, 2], "single": 2});
        assert!(matches(&json!({"numbers": {"$contains": [1.0]}}), &value));
        assert!(matches(&json!({"numbers": {"$in": [2.0]}}), &value));
        assert!(matches(&json!({"single": {"$in": [1.0, 2.0]}}), &value));
    }

    #[test]
    fn and_or_combinators_follow_empty_array_rules() {
        let value = document();
        assert!(matches(
            &json!({"$and": [{"age": 36}, {"name": "Ada"}]}),
            &value
        ));
        assert!(!matches(
            &json!({"$and": [{"age": 36}, {"name": "Bob"}]}),
            &value
        ));
        assert!(matches(&json!({"$and": []}), &value));
        assert!(matches(
            &json!({"$or": [{"age": 1}, {"name": "Ada"}]}),
            &value
        ));
        assert!(!matches(
            &json!({"$or": [{"age": 1}, {"name": "Bob"}]}),
            &value
        ));
        assert!(!matches(&json!({"$or": []}), &value));

        assert!(!matches(&json!({"$and": {"age": 36}}), &value));
        assert!(!matches(&json!({"$or": null}), &value));
        assert!(!matches(&json!({"$and": "age"}), &value));

        assert!(matches(
            &json!({"active": true, "$or": [{"age": 36}, {"age": 99}]}),
            &value
        ));
        assert!(!matches(
            &json!({"active": false, "$or": [{"age": 36}]}),
            &value
        ));
        assert!(matches(
            &json!({"$or": [{"$and": [{"age": 36}, {"active": true}]}, {"name": "Bob"}]}),
            &value
        ));
        assert!(matches(&json!({"$and": [null, {"age": 36}]}), &value));
    }

    #[test]
    fn not_combinator_negates_an_object_filter() {
        let value = document();
        assert!(matches(&json!({"$not": {"name": "Bob"}}), &value));
        assert!(!matches(&json!({"$not": {"name": "Ada"}}), &value));
        assert!(!matches(&json!({"$not": {}}), &value));
        assert!(!matches(&json!({"$not": "name"}), &value));
        assert!(!matches(&json!({"$not": null}), &value));
        assert!(matches(&json!({"$not": {"age": {"$gt": 40}}}), &value));
    }

    #[test]
    fn operator_objects_apply_every_operator_with_and() {
        let value = document();
        assert!(matches(&json!({"age": {"$gte": 18, "$lt": 65}}), &value));
        assert!(!matches(&json!({"age": {"$gte": 18, "$lt": 30}}), &value));
        assert!(matches(&json!({"age": {"$gte": 18, "$ne": 40}}), &value));
        assert!(
            !matches(&json!({"age": {"$gte": 18, "$bogus": 1}}), &value),
            "an unknown operator fails the whole condition"
        );
        assert!(
            !matches(&json!({"age": {"$gt": 10, "unexpected": 1}}), &value),
            "a non-operator key inside an operator object is unknown"
        );
    }

    #[test]
    fn unknown_and_malformed_operators_are_false() {
        let value = document();
        assert!(!matches(&json!({"age": {"$regex": "3"}}), &value));
        assert!(!matches(&json!({"age": {"$gt": null}}), &value));
        assert!(!matches(&json!({"age": {"$lt": null}}), &value));
        assert!(!matches(&json!({"age": {"$in": {"a": 1}}}), &value));
        assert!(!matches(&json!({"name": {"$contains": null}}), &value));
        assert!(!matches(&json!({"tags": {"$contains": true}}), &value));
        assert!(!matches(&json!({"meta": {"$owner": "ada"}}), &value));
        assert!(!matches(&json!({"age": {}}), &value));
        assert!(matches(&json!({"meta": {}}), &json!({"meta": {}})));
    }

    #[test]
    fn object_conditions_without_operators_are_plain_equality() {
        let value = document();
        assert!(matches(
            &json!({"meta": {"owner": "ada", "nested": {"depth": 2}}}),
            &value
        ));
        assert!(!matches(&json!({"meta": {"owner": "ada"}}), &value));
        assert!(matches(
            &json!({"items[0]": {"kind": "a", "value": 1}}),
            &value
        ));
    }

    #[test]
    fn never_panics_on_odd_shapes() {
        let value = json!([[[{"a": [1]}]]]);
        assert!(matches(&json!({}), &value));
        assert!(!matches(&json!({"a": 1}), &value));
        assert!(matches(&json!({"[0][0][0].a[0]": 1}), &value));

        let deep = json!({"a": {"a": {"a": {"a": {"a": {"a": 1}}}}}});
        assert!(matches(&json!({"a.a.a.a.a.a": 1}), &deep));
        assert!(matches(
            &json!({"$and": [{"$or": [{"a.a.a.a.a.a": {"$in": [1.0]}}]}]}),
            &deep
        ));

        assert!(!matches(&json!({"a": {"$eq": {"$eq": 1}}}), &value));
        assert!(!matches(&json!({"$not": {"$and": []}}), &value));
        assert!(matches(&json!({"$not": {"$or": []}}), &value));
        assert!(!matches(&json!({"": 1}), &value));
        assert!(
            matches(&json!({"": 1}), &json!(1)),
            "the empty path resolves to the document itself"
        );
        assert!(matches(&json!({"": value}), &value));
    }

    #[test]
    fn matches_all_requires_every_filter() {
        let value = document();
        assert!(matches_all(&[], &value));
        assert!(matches_all(
            &[json!({"age": 36}), json!({"name": "Ada"})],
            &value
        ));
        assert!(!matches_all(
            &[json!({"age": 36}), json!({"name": "Bob"})],
            &value
        ));
        assert!(!matches_all(&[json!(null), json!({"missing": 1})], &value));
        assert!(!matches_all(&[Value::Null, json!({"$or": []})], &value));
    }
}
