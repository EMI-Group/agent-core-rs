//! Translation of the shared filter language into Elasticsearch query DSL.
//!
//! [`crate::store::filter`] is the filter language of every storage backend;
//! [`translate`] compiles the subset Elasticsearch can express exactly into the
//! query object of a `_search`, `_count` or `_delete_by_query` request.
//!
//! `None` means "Elasticsearch cannot express this filter": the caller then
//! sends `{"match_all": {}}` and re-checks the returned documents with
//! [`matches`](crate::store::filter::matches). A constraint is never silently
//! dropped — a filter is either enforced by the service or checked locally.
//!
//! # What is expressible
//!
//! - `null` and `{}` (everything) become `{"match_all": {}}`, and top-level
//!   `$and` / `$or` / `$not` become the matching `bool` clauses.
//! - A plain `field: scalar` pair becomes a `term` query, and `field: [..]`
//!   becomes a `terms` query.
//! - The operators `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte` and `$in` map onto
//!   `term`, `bool.must_not`, `range` and `terms`.
//!
//! # What is not
//!
//! - Dotted field paths (`meta.owner`), array indices (`items[0]`) and wildcards
//!   in a field name: the index the adapters create has no such nested fields.
//! - `$contains`, which has no faithful Elasticsearch counterpart for the shared
//!   substring/array semantics.
//! - A `null` condition and object literals: a missing field never satisfies
//!   the shared implicit equality, which `term` cannot reproduce.
//! - Malformed shapes (an operator with a non-scalar operand, `$in` with an
//!   empty or non-scalar array, `$and`/`$or` that is not an array, ...).
//! - Array-valued document fields: the adapters store scalars, so implicit
//!   equality on an array condition is compiled to `terms` (any candidate) and
//!   the caller's local re-check decides the exact-array case.
//!
//! Every unsupported construct makes [`translate`] return `None`, never a
//! narrower query.

use serde_json::{json, Map, Value};

const AND: &str = "$and";
const OR: &str = "$or";
const NOT: &str = "$not";

/// Compiles `filters` into an Elasticsearch query, or `None` when it cannot be
/// expressed.
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use openjiuwen_foundation::store::external::elasticsearch::query::translate;
///
/// assert_eq!(translate(&json!({})), Some(json!({"match_all": {}})));
/// assert_eq!(
///     translate(&json!({"age": {"$gte": 18}})),
///     Some(json!({"range": {"age": {"gte": 18}}}))
/// );
/// assert_eq!(translate(&json!({"meta.owner": "ada"})), None);
/// ```
pub fn translate(filters: &Value) -> Option<Value> {
    match filters {
        Value::Null => Some(json!({"match_all": {}})),
        Value::Object(entries) => translate_object(entries),
        _ => None,
    }
}

/// Compiles one filter object: combinators become `bool` clauses, every other
/// key is a field condition.
fn translate_object(entries: &Map<String, Value>) -> Option<Value> {
    let mut must = Vec::new();
    let mut should = Vec::new();
    let mut must_not = Vec::new();
    for (key, condition) in entries {
        match key.as_str() {
            AND => must.extend(translate_all(condition)?),
            OR => {
                let clauses = translate_all(condition)?;
                if clauses.is_empty() {
                    // `$or` of an empty array matches nothing and has no positive
                    // clause; let the caller's local pass decide.
                    return None;
                }
                should.extend(clauses);
            }
            NOT => {
                if !condition.is_object() {
                    return None;
                }
                must_not.push(translate(condition)?);
            }
            field => must.push(field_clause(field, condition)?),
        }
    }
    combine(must, should, must_not)
}

/// Compiles an `$and`/`$or` operand: every sub-filter must translate.
fn translate_all(condition: &Value) -> Option<Vec<Value>> {
    condition
        .as_array()?
        .iter()
        .map(translate)
        .collect::<Option<Vec<Value>>>()
}

/// Assembles the collected clauses into one query.
///
/// A single clause is returned as-is when no other kind of clause is present
/// (except a lone `must_not`, which always needs its `bool` wrapper), and
/// everything else becomes a `bool` query with `must`, `should` +
/// `minimum_should_match: 1`, and `must_not`.
fn combine(must: Vec<Value>, should: Vec<Value>, must_not: Vec<Value>) -> Option<Value> {
    if must.is_empty() && should.is_empty() && must_not.is_empty() {
        return Some(json!({"match_all": {}}));
    }
    if must_not.is_empty() {
        match (must.len(), should.len()) {
            (1, 0) => return must.into_iter().next(),
            (0, 1) => return should.into_iter().next(),
            _ => {}
        }
    }
    let mut query = Map::new();
    if !must.is_empty() {
        query.insert("must".to_string(), Value::Array(must));
    }
    if !should.is_empty() {
        query.insert("should".to_string(), Value::Array(should));
        query.insert("minimum_should_match".to_string(), json!(1));
    }
    if !must_not.is_empty() {
        query.insert("must_not".to_string(), Value::Array(must_not));
    }
    Some(json!({"bool": query}))
}

/// Compiles one top-level field condition.
///
/// A `$`-bearing object is an operator condition; every other shape is implicit
/// equality.
fn field_clause(field: &str, condition: &Value) -> Option<Value> {
    if !is_expressible_field(field) {
        return None;
    }
    if let Value::Object(entries) = condition {
        if entries.keys().any(|key| key.starts_with('$')) {
            return operator_clause(field, entries);
        }
        // An object literal has no `term` equivalent that matches the shared
        // implicit-equality semantics.
        return None;
    }
    equality_clause(field, condition)
}

/// Returns `false` for field names the created indices cannot host.
fn is_expressible_field(field: &str) -> bool {
    !field.is_empty()
        && !field.starts_with('$')
        && !field
            .chars()
            .any(|character| matches!(character, '.' | '[' | ']' | '*'))
}

/// Compiles implicit equality: a scalar becomes `term`, an array `terms`.
fn equality_clause(field: &str, condition: &Value) -> Option<Value> {
    match condition {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => Some(term_query(field, condition)),
        Value::Array(items) => {
            if items.is_empty() || !items.iter().all(is_scalar) {
                return None;
            }
            Some(json!({"terms": { (field): items }}))
        }
        _ => None,
    }
}

/// Compiles an operator condition: every operator must translate, combined
/// with AND.
fn operator_clause(field: &str, operators: &Map<String, Value>) -> Option<Value> {
    let mut clauses = Vec::with_capacity(operators.len());
    for (operator, operand) in operators {
        clauses.push(operator_query(field, operator, operand)?);
    }
    and_clauses(clauses)
}

/// Compiles one operator of an operator condition.
fn operator_query(field: &str, operator: &str, operand: &Value) -> Option<Value> {
    match operator {
        "$eq" if is_scalar(operand) => Some(term_query(field, operand)),
        "$ne" if is_scalar(operand) => Some(json!({
            "bool": {"must_not": [term_query(field, operand)]},
        })),
        "$gt" | "$gte" | "$lt" | "$lte" => range_query(field, operator, operand),
        "$in" => in_query(field, operand),
        // `$contains` and every unknown operator have no faithful mapping.
        _ => None,
    }
}

/// Compiles `$gt`/`$gte`/`$lt`/`$lte` into a `range` clause.
fn range_query(field: &str, operator: &str, operand: &Value) -> Option<Value> {
    if !matches!(operand, Value::Number(_) | Value::String(_)) {
        return None;
    }
    let mut range = Map::new();
    range.insert(
        operator.trim_start_matches('$').to_string(),
        operand.clone(),
    );
    Some(json!({"range": { (field): range }}))
}

/// Compiles `$in` into a `terms` clause.
fn in_query(field: &str, operand: &Value) -> Option<Value> {
    let candidates = operand.as_array()?;
    if candidates.is_empty() || !candidates.iter().all(is_scalar) {
        return None;
    }
    Some(json!({"terms": { (field): candidates }}))
}

/// A `term` query on `field`.
fn term_query(field: &str, value: &Value) -> Value {
    json!({"term": { (field): value }})
}

/// Combines operator clauses with AND.
fn and_clauses(mut clauses: Vec<Value>) -> Option<Value> {
    if clauses.len() == 1 {
        return clauses.pop();
    }
    Some(json!({"bool": {"must": clauses}}))
}

/// Returns `true` for the JSON scalars `term`/`terms` accept.
fn is_scalar(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn everything_becomes_match_all() {
        assert_eq!(translate(&json!(null)), Some(json!({"match_all": {}})));
        assert_eq!(translate(&json!({})), Some(json!({"match_all": {}})));
    }

    #[test]
    fn implicit_equality_becomes_term_or_terms() {
        assert_eq!(
            translate(&json!({"name": "Ada"})),
            Some(json!({"term": {"name": "Ada"}}))
        );
        assert_eq!(
            translate(&json!({"age": 36})),
            Some(json!({"term": {"age": 36}}))
        );
        assert_eq!(
            translate(&json!({"active": true})),
            Some(json!({"term": {"active": true}}))
        );
        assert_eq!(
            translate(&json!({"tags": ["rust", "storage"]})),
            Some(json!({"terms": {"tags": ["rust", "storage"]}}))
        );
        assert_eq!(
            translate(&json!({"name": "Ada", "age": 36})),
            Some(json!({
                "bool": {"must": [{"term": {"name": "Ada"}}, {"term": {"age": 36}}]},
            }))
        );
    }

    #[test]
    fn and_or_not_become_bool_clauses() {
        assert_eq!(
            translate(&json!({"$and": [{"age": {"$gte": 18}}, {"age": {"$lt": 65}}]})),
            Some(json!({
                "bool": {
                    "must": [
                        {"range": {"age": {"gte": 18}}},
                        {"range": {"age": {"lt": 65}}},
                    ],
                },
            }))
        );
        assert_eq!(
            translate(&json!({"$or": [{"name": "Ada"}, {"name": "Bob"}]})),
            Some(json!({
                "bool": {
                    "should": [{"term": {"name": "Ada"}}, {"term": {"name": "Bob"}}],
                    "minimum_should_match": 1,
                },
            }))
        );
        assert_eq!(
            translate(&json!({"$not": {"name": "Ada"}})),
            Some(json!({"bool": {"must_not": [{"term": {"name": "Ada"}}]}}))
        );
        // Combinators combine with sibling field conditions using AND.
        assert_eq!(
            translate(&json!({"name": "Ada", "$or": [{"age": 36}, {"age": 37}]})),
            Some(json!({
                "bool": {
                    "must": [{"term": {"name": "Ada"}}],
                    "should": [{"term": {"age": 36}}, {"term": {"age": 37}}],
                    "minimum_should_match": 1,
                },
            }))
        );
        // An empty `$and` is true, so it leaves the rest of the filter alone.
        assert_eq!(
            translate(&json!({"$and": [], "name": "Ada"})),
            Some(json!({"term": {"name": "Ada"}}))
        );
        // A single `$or` clause needs no wrapper.
        assert_eq!(
            translate(&json!({"$or": [{"name": "Ada"}]})),
            Some(json!({"term": {"name": "Ada"}}))
        );
    }

    #[test]
    fn every_operator_has_a_query() {
        assert_eq!(
            translate(&json!({"name": {"$eq": "Ada"}})),
            Some(json!({"term": {"name": "Ada"}}))
        );
        assert_eq!(
            translate(&json!({"name": {"$ne": "Ada"}})),
            Some(json!({"bool": {"must_not": [{"term": {"name": "Ada"}}]}}))
        );
        assert_eq!(
            translate(&json!({"age": {"$gt": 18}})),
            Some(json!({"range": {"age": {"gt": 18}}}))
        );
        assert_eq!(
            translate(&json!({"age": {"$gte": 18}})),
            Some(json!({"range": {"age": {"gte": 18}}}))
        );
        assert_eq!(
            translate(&json!({"age": {"$lt": 65}})),
            Some(json!({"range": {"age": {"lt": 65}}}))
        );
        assert_eq!(
            translate(&json!({"name": {"$lte": "Zoe"}})),
            Some(json!({"range": {"name": {"lte": "Zoe"}}}))
        );
        assert_eq!(
            translate(&json!({"age": {"$in": [18, 21]}})),
            Some(json!({"terms": {"age": [18, 21]}}))
        );
        // Operators of one condition are ANDed.
        assert_eq!(
            translate(&json!({"age": {"$gte": 18, "$lt": 65}})),
            Some(json!({
                "bool": {"must": [{"range": {"age": {"gte": 18}}}, {"range": {"age": {"lt": 65}}}]},
            }))
        );
    }

    #[test]
    fn expressible_filters_never_lose_a_constraint() {
        // A field condition next to an operator condition stays in `must`.
        assert_eq!(
            translate(&json!({"name": "Ada", "age": {"$gt": 18}})),
            Some(json!({
                "bool": {
                    "must": [{"term": {"name": "Ada"}}, {"range": {"age": {"gt": 18}}}],
                },
            }))
        );
    }

    #[test]
    fn inexpressible_filters_return_none() {
        for filter in [
            // Non-object filters match nothing in the shared language.
            json!(1),
            json!("name"),
            json!(true),
            json!([]),
            json!([{"name": "Ada"}]),
            // Nested paths and wildcards are not fields of a created index.
            json!({"meta.owner": "ada"}),
            json!({"items[0].kind": "a"}),
            json!({"*": "ada"}),
            json!({"": "ada"}),
            // A field name starting with `$` is not a combinator here.
            json!({"$eq": "ada"}),
            // `$contains` has no faithful counterpart.
            json!({"name": {"$contains": "d"}}),
            json!({"tags": {"$contains": ["rust"]}}),
            // Unknown operators, and non-operator entries inside an operator object.
            json!({"age": {"$mod": [2, 0]}}),
            json!({"age": {"$gt": 18, "other": 1}}),
            // Null conditions and object literals cannot be expressed by `term`.
            json!({"nothing": null}),
            json!({"meta": {"owner": "ada"}}),
            // Malformed operands.
            json!({"age": {"$gt": null}}),
            json!({"age": {"$gt": true}}),
            json!({"age": {"$eq": [1, 2]}}),
            json!({"age": {"$in": []}}),
            json!({"age": {"$in": 18}}),
            json!({"age": {"$in": [{"nested": 1}]}}),
            json!({"tags": []}),
            // Malformed combinators.
            json!({"$and": {"name": "Ada"}}),
            json!({"$or": {"name": "Ada"}}),
            json!({"$or": []}),
            json!({"$not": "Ada"}),
            json!({"$and": [{"name": "Ada"}, {"meta.owner": "ada"}]}),
        ] {
            assert_eq!(translate(&filter), None, "filter: {filter}");
        }
    }
}
