//! [`MetadataFilter`]: the Python `Dict[str, Any]` filter shape as a typed tree.
//!
//! Supports plain-object equality (`{"author": "alice"}`), per-field operators
//! (`{"year": {"$gte": 2024}}`), and the logical operators `$and`, `$or`, and
//! `$not`. Both directions use the canonical form: an object without `$` keys is
//! a conjunction of equalities, so `{"a": 1, "b": 2}` deserializes to
//! `And([Eq(a, 1), Eq(b, 2)])` and serializes back to the same object.

use std::cmp::Ordering;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use super::document::Metadata;

/// A metadata predicate evaluated against a [`Metadata`] map.
///
/// The `doc_id` and `chunk_id` pseudo-fields are resolved by the vector store,
/// not by [`MetadataFilter::matches`].
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataFilter {
    /// The field equals the value.
    Eq {
        /// Metadata field name.
        field: String,
        /// Expected value.
        value: Value,
    },
    /// The field does not equal the value (a missing field also satisfies this).
    Ne {
        /// Metadata field name.
        field: String,
        /// Rejected value.
        value: Value,
    },
    /// The field is numerically or lexicographically greater than the value.
    Gt {
        /// Metadata field name.
        field: String,
        /// Exclusive lower bound.
        value: Value,
    },
    /// The field is numerically or lexicographically greater than or equal to the value.
    Gte {
        /// Metadata field name.
        field: String,
        /// Inclusive lower bound.
        value: Value,
    },
    /// The field is numerically or lexicographically less than the value.
    Lt {
        /// Metadata field name.
        field: String,
        /// Exclusive upper bound.
        value: Value,
    },
    /// The field is numerically or lexicographically less than or equal to the value.
    Lte {
        /// Metadata field name.
        field: String,
        /// Inclusive upper bound.
        value: Value,
    },
    /// The field equals one of the values (arrays match if any element does).
    In {
        /// Metadata field name.
        field: String,
        /// Accepted values.
        values: Vec<Value>,
    },
    /// The field equals none of the values.
    Nin {
        /// Metadata field name.
        field: String,
        /// Rejected values.
        values: Vec<Value>,
    },
    /// The field is present in the metadata.
    Exists {
        /// Metadata field name.
        field: String,
    },
    /// An array field contains the value, or a string field contains the substring.
    Contains {
        /// Metadata field name.
        field: String,
        /// Searched element or substring.
        value: Value,
    },
    /// All children match (an empty list matches everything).
    And(Vec<MetadataFilter>),
    /// Any child matches (an empty list matches nothing).
    Or(Vec<MetadataFilter>),
    /// The child does not match.
    Not(Box<MetadataFilter>),
}

impl MetadataFilter {
    /// Builds an equality predicate.
    pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Eq {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Builds a inequality predicate.
    pub fn ne(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Ne {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Builds a membership predicate.
    pub fn in_values<I, V>(field: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<Value>,
    {
        Self::In {
            field: field.into(),
            values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// Builds a key-presence predicate.
    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists {
            field: field.into(),
        }
    }

    /// Builds a conjunction.
    pub fn and(filters: impl IntoIterator<Item = Self>) -> Self {
        Self::And(filters.into_iter().collect())
    }

    /// Builds a disjunction.
    pub fn or(filters: impl IntoIterator<Item = Self>) -> Self {
        Self::Or(filters.into_iter().collect())
    }

    /// Builds a negation.
    #[must_use]
    pub fn not(filter: impl Into<Box<Self>>) -> Self {
        Self::Not(filter.into())
    }

    /// Evaluates the predicate against `metadata`.
    ///
    /// Never panics: numeric comparisons use [`f64`], string comparisons are
    /// lexicographic, and comparing incompatible JSON types yields `false`.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Self::Eq { field, value } => metadata
                .get(field)
                .is_some_and(|found| json_eq(found, value)),
            Self::Ne { field, value } => !metadata
                .get(field)
                .is_some_and(|found| json_eq(found, value)),
            Self::Gt { field, value } => compare(metadata.get(field), value, Ordering::is_gt),
            Self::Gte { field, value } => compare(metadata.get(field), value, Ordering::is_ge),
            Self::Lt { field, value } => compare(metadata.get(field), value, Ordering::is_lt),
            Self::Lte { field, value } => compare(metadata.get(field), value, Ordering::is_le),
            Self::In { field, values } => metadata
                .get(field)
                .is_some_and(|found| value_in_values(found, values)),
            Self::Nin { field, values } => !metadata
                .get(field)
                .is_some_and(|found| value_in_values(found, values)),
            Self::Exists { field } => metadata.contains_key(field),
            Self::Contains { field, value } => metadata
                .get(field)
                .is_some_and(|found| contains_value(found, value)),
            Self::And(filters) => filters.iter().all(|filter| filter.matches(metadata)),
            Self::Or(filters) => filters.iter().any(|filter| filter.matches(metadata)),
            Self::Not(filter) => !filter.matches(metadata),
        }
    }

    /// Parses the Python JSON filter shape into a canonical tree.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value is not an object, an operator
    /// argument has the wrong type, or an operator is unknown.
    pub fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Object(map) => Self::from_object(map),
            other => Err(Error::validation(format!(
                "metadata filter must be a JSON object, got {other}"
            ))),
        }
    }

    fn from_object(map: &Map<String, Value>) -> Result<Self> {
        let mut filters = Vec::with_capacity(map.len());
        for (key, value) in map {
            match key.as_str() {
                "$and" | "$or" => {
                    let items = value.as_array().ok_or_else(|| {
                        Error::validation(format!("{key} requires an array of filter objects"))
                    })?;
                    let children = items
                        .iter()
                        .map(Self::from_value)
                        .collect::<Result<Vec<_>>>()?;
                    filters.push(if key == "$and" {
                        Self::And(children)
                    } else {
                        Self::Or(children)
                    });
                }
                "$not" => {
                    if !value.is_object() {
                        return Err(Error::validation("$not requires a filter object"));
                    }
                    filters.push(Self::Not(Box::new(Self::from_value(value)?)));
                }
                field if field.starts_with('$') => {
                    return Err(Error::validation(format!(
                        "unsupported metadata filter operator {field}"
                    )));
                }
                field => filters.push(Self::from_field(field, value)?),
            }
        }
        Ok(combine(filters))
    }

    fn from_field(field: &str, value: &Value) -> Result<Self> {
        let Some(operators) = value
            .as_object()
            .filter(|ops| ops.keys().any(|k| k.starts_with('$')))
        else {
            return Ok(Self::Eq {
                field: field.to_owned(),
                value: value.clone(),
            });
        };
        let mut filters = Vec::with_capacity(operators.len());
        for (operator, operand) in operators {
            filters.push(Self::from_operator(field, operator, operand)?);
        }
        Ok(combine(filters))
    }

    fn from_operator(field: &str, operator: &str, operand: &Value) -> Result<Self> {
        let value = || operand.clone();
        let field_name = || field.to_owned();
        match operator {
            "$eq" => Ok(Self::Eq {
                field: field_name(),
                value: value(),
            }),
            "$ne" => Ok(Self::Ne {
                field: field_name(),
                value: value(),
            }),
            "$gt" => Ok(Self::Gt {
                field: field_name(),
                value: value(),
            }),
            "$gte" => Ok(Self::Gte {
                field: field_name(),
                value: value(),
            }),
            "$lt" => Ok(Self::Lt {
                field: field_name(),
                value: value(),
            }),
            "$lte" => Ok(Self::Lte {
                field: field_name(),
                value: value(),
            }),
            "$in" | "$nin" => {
                let values = operand.as_array().ok_or_else(|| {
                    Error::validation(format!("{operator} requires an array of values"))
                })?;
                let values = values.clone();
                Ok(if operator == "$in" {
                    Self::In {
                        field: field_name(),
                        values,
                    }
                } else {
                    Self::Nin {
                        field: field_name(),
                        values,
                    }
                })
            }
            "$exists" => {
                let exists = operand
                    .as_bool()
                    .ok_or_else(|| Error::validation("$exists requires a boolean operand"))?;
                let exists_filter = Self::Exists {
                    field: field_name(),
                };
                Ok(if exists {
                    exists_filter
                } else {
                    Self::Not(Box::new(exists_filter))
                })
            }
            "$contains" => Ok(Self::Contains {
                field: field_name(),
                value: value(),
            }),
            other => Err(Error::validation(format!(
                "unsupported metadata filter operator {other}"
            ))),
        }
    }

    /// Renders the canonical JSON shape (the inverse of `from_value`).
    fn to_value(&self) -> Value {
        match self {
            Self::And(filters) => logical("$and", filters),
            Self::Or(filters) => logical("$or", filters),
            Self::Not(filter) => {
                let mut map = Map::new();
                map.insert("$not".to_owned(), filter.to_value());
                Value::Object(map)
            }
            Self::Eq { field, value } => {
                let mut map = Map::new();
                map.insert(field.clone(), value.clone());
                Value::Object(map)
            }
            Self::Ne { field, value } => operator(field, "$ne", value.clone()),
            Self::Gt { field, value } => operator(field, "$gt", value.clone()),
            Self::Gte { field, value } => operator(field, "$gte", value.clone()),
            Self::Lt { field, value } => operator(field, "$lt", value.clone()),
            Self::Lte { field, value } => operator(field, "$lte", value.clone()),
            Self::In { field, values } => operator(field, "$in", Value::Array(values.clone())),
            Self::Nin { field, values } => operator(field, "$nin", Value::Array(values.clone())),
            Self::Exists { field } => operator(field, "$exists", Value::Bool(true)),
            Self::Contains { field, value } => operator(field, "$contains", value.clone()),
        }
    }
}

/// Folds parsed filters into a single node.
fn combine(mut filters: Vec<MetadataFilter>) -> MetadataFilter {
    match filters.len() {
        0 => MetadataFilter::And(Vec::new()),
        1 => filters
            .pop()
            .unwrap_or_else(|| MetadataFilter::And(Vec::new())),
        _ => MetadataFilter::And(filters),
    }
}

/// Builds `{field: {operator: operand}}`.
fn operator(field: &str, operator: &str, operand: Value) -> Value {
    let mut inner = Map::new();
    inner.insert(operator.to_owned(), operand);
    let mut outer = Map::new();
    outer.insert(field.to_owned(), Value::Object(inner));
    Value::Object(outer)
}

/// Builds `{logical: [children]}`.
fn logical(logical: &str, filters: &[MetadataFilter]) -> Value {
    let mut map = Map::new();
    map.insert(
        logical.to_owned(),
        Value::Array(filters.iter().map(MetadataFilter::to_value).collect()),
    );
    Value::Object(map)
}

/// JSON equality with numeric coercion, so `1` and `1.0` compare equal.
fn json_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => a == b,
        },
        _ => left == right,
    }
}

/// Orders two compatible JSON scalars, or `None` for incomparable types.
fn ordering(left: &Value, right: &Value) -> Option<Ordering> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => a.as_f64()?.partial_cmp(&b.as_f64()?),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Applies an ordering predicate to a looked-up field value.
fn compare(found: Option<&Value>, bound: &Value, predicate: fn(Ordering) -> bool) -> bool {
    found
        .and_then(|found| ordering(found, bound))
        .is_some_and(predicate)
}

/// `$in`: the value equals one of `values`, or an array value holds one of them.
fn value_in_values(found: &Value, values: &[Value]) -> bool {
    if values.iter().any(|value| json_eq(found, value)) {
        return true;
    }
    found.as_array().is_some_and(|items| {
        items
            .iter()
            .any(|item| values.iter().any(|value| json_eq(item, value)))
    })
}

/// `$contains`: array membership or substring search.
fn contains_value(found: &Value, needle: &Value) -> bool {
    match found {
        Value::Array(items) => items.iter().any(|item| json_eq(item, needle)),
        Value::String(text) => needle.as_str().is_some_and(|needle| text.contains(needle)),
        _ => false,
    }
}

impl std::ops::Not for MetadataFilter {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

impl Serialize for MetadataFilter {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MetadataFilter {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(&value).map_err(D::Error::custom)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn metadata(value: &Value) -> Metadata {
        value.as_object().cloned().unwrap()
    }

    fn parse(value: Value) -> MetadataFilter {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn equality_and_inequality() {
        let meta = metadata(&json!({"author": "alice", "year": 2024}));
        assert!(MetadataFilter::eq("author", "alice").matches(&meta));
        assert!(!MetadataFilter::eq("author", "bob").matches(&meta));
        assert!(MetadataFilter::eq("year", 2024).matches(&meta));
        // Numeric coercion: 2024 and 2024.0 are the same value.
        assert!(MetadataFilter::eq("year", 2024.0).matches(&meta));
        assert!(MetadataFilter::ne("author", "bob").matches(&meta));
        assert!(!MetadataFilter::ne("author", "alice").matches(&meta));
        // A missing field satisfies `$ne` and fails `$eq`.
        assert!(MetadataFilter::ne("missing", "x").matches(&meta));
        assert!(!MetadataFilter::eq("missing", "x").matches(&meta));
    }

    #[test]
    fn comparison_operators() {
        let meta = metadata(&json!({"year": 2024, "name": "bob"}));
        assert!(parse(json!({"year": {"$gt": 2023}})).matches(&meta));
        assert!(!parse(json!({"year": {"$gt": 2024}})).matches(&meta));
        assert!(parse(json!({"year": {"$gte": 2024}})).matches(&meta));
        assert!(parse(json!({"year": {"$lt": 2025}})).matches(&meta));
        assert!(parse(json!({"year": {"$lte": 2024}})).matches(&meta));
        // Strings compare lexicographically.
        assert!(parse(json!({"name": {"$gt": "amy"}})).matches(&meta));
        assert!(!parse(json!({"name": {"$gt": "carl"}})).matches(&meta));
    }

    #[test]
    fn in_nin_exists_and_contains() {
        let meta = metadata(&json!({
            "year": 2024,
            "tags": ["rust", "python"],
            "title": "Hello world"
        }));
        assert!(parse(json!({"year": {"$in": [2023, 2024]}})).matches(&meta));
        assert!(!parse(json!({"year": {"$in": [2020]}})).matches(&meta));
        assert!(parse(json!({"year": {"$nin": [2020]}})).matches(&meta));
        assert!(!parse(json!({"year": {"$nin": [2024]}})).matches(&meta));
        // Arrays match `$in` through any of their elements.
        assert!(parse(json!({"tags": {"$in": ["python", "go"]}})).matches(&meta));
        assert!(parse(json!({"title": {"$exists": true}})).matches(&meta));
        assert!(!parse(json!({"missing": {"$exists": true}})).matches(&meta));
        assert!(parse(json!({"missing": {"$exists": false}})).matches(&meta));
        assert!(parse(json!({"tags": {"$contains": "rust"}})).matches(&meta));
        assert!(!parse(json!({"tags": {"$contains": "go"}})).matches(&meta));
        // `$contains` on a string is a substring search.
        assert!(parse(json!({"title": {"$contains": "lo wo"}})).matches(&meta));
    }

    #[test]
    fn logical_operators_deserialize_and_match() {
        let meta = metadata(&json!({"year": 2024, "author": "alice"}));
        let filter = parse(json!({
            "$or": [
                {"author": "bob"},
                {"$and": [{"year": {"$gte": 2020}}, {"year": {"$lt": 2030}}]}
            ]
        }));
        assert!(filter.matches(&meta));
        assert!(!filter.matches(&metadata(&json!({"year": 1999, "author": "alice"}))));

        let negated = parse(json!({"$not": {"author": "alice"}}));
        assert!(!negated.matches(&meta));
        assert!(negated.matches(&metadata(&json!({"author": "bob"}))));

        // `$not` with a non-object operand is rejected.
        let error = serde_json::from_value::<MetadataFilter>(json!({"$not": 3})).unwrap_err();
        assert!(error.to_string().contains("$not requires a filter object"));
    }

    #[test]
    fn empty_logical_semantics() {
        assert!(MetadataFilter::and([]).matches(&Metadata::new()));
        assert!(!MetadataFilter::or([]).matches(&Metadata::new()));
        assert!(parse(json!({})).matches(&Metadata::new()));
    }

    #[test]
    fn canonical_serde_round_trip() {
        let cases = [
            json!({"author": "alice"}),
            json!({"author": "alice", "year": 2024}),
            json!({"year": {"$gte": 2024}}),
            json!({"year": {"$gt": 1, "$lt": 9}}),
            json!({"year": {"$in": [2023, 2024]}}),
            json!({"tags": {"$contains": "rust"}}),
            json!({"$or": [{"a": 1}, {"b": {"$ne": 2}}]}),
            json!({"a": 1, "$and": [{"b": 2}, {"$not": {"c": 3}}]}),
            json!({"$and": []}),
        ];
        for case in cases {
            let filter: MetadataFilter = serde_json::from_value(case.clone()).unwrap();
            let text = serde_json::to_string(&filter).unwrap();
            let reparsed: MetadataFilter = serde_json::from_str(&text).unwrap();
            assert_eq!(reparsed, filter, "case {case}");
            assert_eq!(
                serde_json::to_string(&reparsed).unwrap(),
                text,
                "case {case}"
            );
        }

        // A mixed top-level object canonicalizes to a single conjunction.
        let mixed = parse(json!({"a": 1, "$or": [{"b": 2}]}));
        assert_eq!(
            serde_json::to_value(&mixed).unwrap(),
            json!({"$and": [{"a": 1}, {"$or": [{"b": 2}]}]})
        );
        // Multiple operators in one object are AND-ed.
        let range = parse(json!({"year": {"$gte": 2020, "$lt": 2030}}));
        assert_eq!(
            serde_json::to_value(&range).unwrap(),
            json!({"$and": [{"year": {"$gte": 2020}}, {"year": {"$lt": 2030}}]})
        );
    }

    #[test]
    fn mixed_types_never_match_and_never_panic() {
        let meta = metadata(&json!({"year": "2024", "flag": true, "nested": {"a": 1}}));
        assert!(!parse(json!({"year": {"$gt": 2000}})).matches(&meta));
        assert!(!parse(json!({"flag": {"$gte": 1}})).matches(&meta));
        assert!(!parse(json!({"nested": {"$contains": "a"}})).matches(&meta));
        assert!(!parse(json!({"year": {"$in": [2024, 2025]}})).matches(&meta));
        assert!(!parse(json!({"year": {"$contains": "99"}})).matches(&meta));
    }

    #[test]
    fn invalid_filters_are_validation_errors() {
        let error = serde_json::from_value::<MetadataFilter>(json!([1, 2])).unwrap_err();
        assert!(error.to_string().contains("must be a JSON object"));
        let error =
            serde_json::from_value::<MetadataFilter>(json!({"year": {"$in": "not an array"}}))
                .unwrap_err();
        assert!(error.to_string().contains("requires an array"));
        // A top-level `$` key that is not a logical operator is rejected.
        let error = serde_json::from_value::<MetadataFilter>(json!({"$in": [1]})).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported metadata filter operator"));
        let error =
            serde_json::from_value::<MetadataFilter>(json!({"year": {"$regex": "x"}})).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported metadata filter operator"));
        let error =
            serde_json::from_value::<MetadataFilter>(json!({"year": {"$gt": 1, "$exists": "yes"}}))
                .unwrap_err();
        assert!(error.to_string().contains("$exists requires a boolean"));
    }
}
