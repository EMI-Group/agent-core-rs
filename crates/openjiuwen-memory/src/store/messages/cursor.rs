//! Keyset cursors and document-filter translation for message-history pagination.
//!
//! [`MessageCursor`] is an opaque, backend-agnostic position in one scope's history:
//! the `(timestamp, id)` tuple of the last row a page returned, encoded as
//! `mc1.{millis}.{checksum}.{id}`. Because the token carries the keyset values
//! themselves rather than an offset or a row number, it survives a store reopen and a
//! rebuild of the underlying collection.
//!
//! [`page_filter`] is the single translation from "one page of this scope" to the
//! foundation document-filter language: the scope clauses ([`scope_filter`]) AND the
//! caller's [`MemoryFilter`] clauses ([`message_filter`]) AND the cursor lower bound.
//! The filter language has no `$or`, so the tie-break between rows sharing the
//! cursor's timestamp happens in memory in the parent module; the filter only has to
//! be an inclusive `timestamp >= cursor.timestamp` bound.
//!
//! [`MemoryFilter`]: crate::record::MemoryFilter

use std::cmp::Ordering;

use chrono::{DateTime, TimeZone, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::map::Entry;
use serde_json::{Map, Value};

use crate::record::{MemoryFilter, MemoryScope};

use super::{
    fnv1a_64, StoredMessage, FIELD_ID, FIELD_SCOPE_ID, FIELD_SESSION_ID, FIELD_TIMESTAMP,
    FIELD_USER_ID,
};

/// Version tag of an encoded cursor token.
pub(crate) const CURSOR_VERSION: &str = "mc1";
/// The number of dot-separated fields in an encoded cursor token.
const CURSOR_FIELDS: usize = 4;
/// The operator a cursor lower bound uses (inclusive, so same-timestamp rows are read
/// and then tie-broken in memory).
const GTE: &str = "$gte";
/// The `$in` operator, used to express "nothing can match" for contradicting clauses.
const IN: &str = "$in";

/// A keyset position in one scope's message history.
///
/// The position is the `(timestamp, id)` tuple of the last row a page returned; a page
/// resumes strictly after it, so identical timestamps stay ordered by id without
/// duplicates or gaps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MessageCursor {
    /// Timestamp of the last row of the previous page.
    pub timestamp: DateTime<Utc>,
    /// Id of the last row of the previous page (the document id).
    pub id: String,
}

impl MessageCursor {
    /// Builds the cursor of the `(timestamp, id)` position.
    #[must_use]
    pub fn new(timestamp: DateTime<Utc>, id: impl Into<String>) -> Self {
        Self {
            timestamp,
            id: id.into(),
        }
    }

    /// Builds the cursor that resumes after `message`.
    #[must_use]
    pub(super) fn of(message: &StoredMessage) -> Self {
        Self::new(message.timestamp, message.id.clone())
    }

    /// Returns `true` when `message` sorts strictly after this position.
    #[must_use]
    pub(super) fn is_before(&self, message: &StoredMessage) -> bool {
        (message.timestamp, message.id.as_str()) > (self.timestamp, self.id.as_str())
    }

    /// Encodes the position as an opaque, versioned token.
    ///
    /// The shape is `mc1.{millis}.{checksum}.{id}`: the millisecond stamp is exact
    /// (chrono timestamps are millisecond-precision in practice and the id is the
    /// tie-break), and the checksum makes a tampered token detectable instead of
    /// silently repositioning the page.
    #[must_use]
    pub fn encode(&self) -> String {
        let millis = self.timestamp.timestamp_millis();
        format!(
            "{CURSOR_VERSION}.{millis}.{:016x}.{}",
            cursor_checksum(millis, &self.id),
            self.id
        )
    }

    /// Decodes a token produced by [`MessageCursor::encode`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the token for a garbage, truncated, or
    /// tampered cursor; a malformed token is never a panic and never a silent default.
    pub fn decode(token: &str) -> Result<Self> {
        let malformed = |reason: &str| {
            Error::validation(format!("malformed message cursor `{token}`: {reason}"))
        };
        let parts: Vec<&str> = token.splitn(CURSOR_FIELDS, '.').collect();
        if parts.len() != CURSOR_FIELDS {
            return Err(malformed("expected `mc1.{millis}.{checksum}.{id}`"));
        }
        if parts[0] != CURSOR_VERSION {
            return Err(malformed("unsupported cursor version"));
        }
        let millis: i64 = parts[1]
            .parse()
            .map_err(|_| malformed("the millisecond stamp is not an integer"))?;
        if parts[2].len() != 16 || !parts[2].bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(malformed(
                "the checksum is not a 16-digit hexadecimal value",
            ));
        }
        if parts[3].is_empty() {
            return Err(malformed("the message id is empty"));
        }
        let expected = format!("{:016x}", cursor_checksum(millis, parts[3]));
        if expected != parts[2].to_ascii_lowercase() {
            return Err(malformed("the checksum does not match the cursor contents"));
        }
        let timestamp = Utc
            .timestamp_millis_opt(millis)
            .single()
            .ok_or_else(|| malformed("the millisecond stamp is out of range"))?;
        Ok(Self::new(timestamp, parts[3]))
    }
}

/// One page of conversation history.
///
/// `items` are in ascending `(timestamp, id)` order, `next_cursor` is the token that
/// resumes after the last item (absent on the last page), and `has_more` says whether
/// the history continues past `items`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessagePage {
    /// The rows of this page, oldest first.
    pub items: Vec<StoredMessage>,
    /// Token resuming after the last item; `None` on the last page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Whether more rows exist after this page.
    pub has_more: bool,
}

/// Computes the 64-bit FNV-1a checksum of one cursor's contents.
fn cursor_checksum(millis: i64, id: &str) -> u64 {
    fnv1a_64(format!("{millis}.{id}").as_bytes())
}

/// The document filter selecting every message of `scope`.
///
/// `user_id` and `scope_id` always participate; `session_id` only when the scope is
/// session-bound, so a session-free scope reads every session's history.
pub(super) fn scope_filter(scope: &MemoryScope) -> Value {
    let mut filter = Map::new();
    filter.insert(
        FIELD_USER_ID.to_owned(),
        Value::String(scope.user_id.clone()),
    );
    filter.insert(
        FIELD_SCOPE_ID.to_owned(),
        Value::String(scope.scope_id.clone()),
    );
    if let Some(session_id) = &scope.session_id {
        filter.insert(
            FIELD_SESSION_ID.to_owned(),
            Value::String(session_id.clone()),
        );
    }
    Value::Object(filter)
}

/// Translates a [`MemoryFilter`] into the document filter language.
pub(super) fn message_filter(filter: &MemoryFilter) -> Value {
    let mut clauses = Map::new();
    if let Some(user_id) = &filter.user_id {
        clauses.insert(FIELD_USER_ID.to_owned(), Value::String(user_id.clone()));
    }
    if let Some(scope_id) = &filter.scope_id {
        clauses.insert(FIELD_SCOPE_ID.to_owned(), Value::String(scope_id.clone()));
    }
    if let Some(session_id) = &filter.session_id {
        clauses.insert(
            FIELD_SESSION_ID.to_owned(),
            Value::String(session_id.clone()),
        );
    }
    if let Some(agent_id) = &filter.agent_id {
        clauses.insert("agent_id".to_owned(), Value::String(agent_id.clone()));
    }
    if !filter.ids.is_empty() {
        let ids = filter
            .ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect();
        let mut in_clause = Map::new();
        in_clause.insert(IN.to_owned(), Value::Array(ids));
        clauses.insert(FIELD_ID.to_owned(), Value::Object(in_clause));
    }
    let mut timestamp = Map::new();
    if let Some(after) = filter.created_after {
        timestamp.insert("$gt".to_owned(), Value::String(after.to_rfc3339()));
    }
    if let Some(before) = filter.created_before {
        timestamp.insert("$lt".to_owned(), Value::String(before.to_rfc3339()));
    }
    if !timestamp.is_empty() {
        clauses.insert(FIELD_TIMESTAMP.to_owned(), Value::Object(timestamp));
    }
    Value::Object(clauses)
}

/// The document filter of one page: the scope clauses AND `filter` AND the cursor.
///
/// Every clause of every source survives: two conditions on the same field are `AND`ed
/// (equal conditions collapse, contradicting ones become an `$in` over an empty
/// candidate list, and range bounds keep the stricter value), so a caller's
/// [`MemoryFilter`] can never widen the page past its scope.
pub(super) fn page_filter(
    scope: &MemoryScope,
    filter: Option<&MemoryFilter>,
    cursor: Option<&MessageCursor>,
) -> Value {
    let mut clauses = match scope_filter(scope) {
        Value::Object(clauses) => clauses,
        _ => Map::new(),
    };
    if let Some(filter) = filter {
        if let Value::Object(extra) = message_filter(filter) {
            for (field, condition) in extra {
                merge_clause(&mut clauses, field, condition);
            }
        }
    }
    if let Some(cursor) = cursor {
        let mut bound = Map::new();
        bound.insert(GTE.to_owned(), Value::String(cursor.timestamp.to_rfc3339()));
        merge_clause(
            &mut clauses,
            FIELD_TIMESTAMP.to_owned(),
            Value::Object(bound),
        );
    }
    Value::Object(clauses)
}

/// ANDs `condition` into the `field` clause of `clauses`.
fn merge_clause(clauses: &mut Map<String, Value>, field: String, condition: Value) {
    match clauses.entry(field) {
        Entry::Vacant(slot) => {
            slot.insert(condition);
        }
        Entry::Occupied(mut slot) => {
            let merged = combine(slot.get(), &condition);
            slot.insert(merged);
        }
    }
}

/// The condition that holds only when both `current` and `incoming` hold.
fn combine(current: &Value, incoming: &Value) -> Value {
    if current == incoming {
        return current.clone();
    }
    match (current, incoming) {
        (Value::Object(left), Value::Object(right)) => {
            let mut merged = left.clone();
            for (operator, value) in right {
                match merged.get(operator) {
                    None => {
                        merged.insert(operator.clone(), value.clone());
                    }
                    Some(existing) => {
                        let stricter = stricter_bound(operator, existing, value);
                        merged.insert(operator.clone(), stricter);
                    }
                }
            }
            Value::Object(merged)
        }
        // Two different values on one field (or a value and an operator set) can
        // never both match: an `$in` over no candidate matches nothing.
        _ => match_nothing(),
    }
}

/// The stricter of two values of the same operator.
fn stricter_bound(operator: &str, current: &Value, incoming: &Value) -> Value {
    if current == incoming {
        return current.clone();
    }
    match operator {
        "$gt" | GTE => later(current, incoming),
        "$lt" | "$lte" => earlier(current, incoming),
        IN => intersect(current, incoming),
        _ => match_nothing(),
    }
}

/// The later of two bound values.
fn later(left: &Value, right: &Value) -> Value {
    match compare_bounds(left, right) {
        Ordering::Greater => left.clone(),
        _ => right.clone(),
    }
}

/// The earlier of two bound values.
fn earlier(left: &Value, right: &Value) -> Value {
    match compare_bounds(left, right) {
        Ordering::Less => left.clone(),
        _ => right.clone(),
    }
}

/// The candidates both `$in` lists accept (the operator's value, not a clause).
///
/// An empty list matches no document, which is exactly the meaning of two disjoint
/// candidate sets.
fn intersect(left: &Value, right: &Value) -> Value {
    let (Some(left), Some(right)) = (left.as_array(), right.as_array()) else {
        return Value::Array(Vec::new());
    };
    Value::Array(
        left.iter()
            .filter(|candidate| right.contains(candidate))
            .cloned()
            .collect(),
    )
}

/// Compares two bound values, reading RFC 3339 timestamps chronologically.
fn compare_bounds(left: &Value, right: &Value) -> Ordering {
    if let (Some(left), Some(right)) = (left.as_str(), right.as_str()) {
        if let (Ok(left), Ok(right)) = (
            DateTime::parse_from_rfc3339(left),
            DateTime::parse_from_rfc3339(right),
        ) {
            return left.cmp(&right);
        }
    }
    left.to_string().cmp(&right.to_string())
}

/// The condition no document satisfies.
fn match_nothing() -> Value {
    let mut clause = Map::new();
    clause.insert(IN.to_owned(), Value::Array(Vec::new()));
    Value::Object(clause)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws").session("s1")
    }

    fn at(offset: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + offset, 0)
            .single()
            .expect("fixed time")
    }

    #[test]
    fn cursor_round_trips_and_keeps_the_position() {
        let cursor = MessageCursor::new(at(30), "msg_abc_1700000030000");
        let token = cursor.encode();
        assert!(token.starts_with("mc1.1700000030000."), "{token}");
        assert_eq!(MessageCursor::decode(&token).expect("decode"), cursor);
        let message = StoredMessage {
            id: "msg_abc_1700000030000".to_owned(),
            timestamp: at(30),
            ..StoredMessage::default()
        };
        assert!(!cursor.is_before(&message));
        let next = StoredMessage {
            id: "msg_abd_1700000030000".to_owned(),
            ..message.clone()
        };
        assert!(cursor.is_before(&next));
        let later = StoredMessage {
            timestamp: at(31),
            ..message
        };
        assert!(cursor.is_before(&later));
    }

    #[test]
    fn garbage_cursors_are_validation_errors() {
        for token in [
            "nonsense",
            "",
            "mc1",
            "mc1.",
            "mc1.1700000030000",
            "mc2.1700000030000.0000000000000000.id",
            "mc1.notamillis.0000000000000000.id",
            "mc1.1700000030000.zzzzzzzzzzzzzzzz.id",
            "mc1.1700000030000.0123.id",
            "mc1.1700000030000.0000000000000000.",
        ] {
            let error = MessageCursor::decode(token).expect_err(token);
            assert!(matches!(error, Error::Validation(_)), "{token}: {error:?}");
            assert!(error.to_string().contains("message cursor"), "{token}");
        }
        let valid = MessageCursor::new(at(7), "msg_1_1700000007000").encode();
        let mut tampered: Vec<char> = valid.chars().collect();
        let last = tampered.len() - 1;
        tampered[last] = if tampered[last] == 'x' { 'y' } else { 'x' };
        let tampered: String = tampered.into_iter().collect();
        assert!(matches!(
            MessageCursor::decode(&tampered),
            Err(Error::Validation(_))
        ));
        let truncated = &valid[..valid.len() - 4];
        assert!(matches!(
            MessageCursor::decode(truncated),
            Err(Error::Validation(_))
        ));
    }

    #[test]
    fn page_filter_keeps_the_scope_and_the_caller_clauses() {
        let mut filter = MemoryFilter::for_scope(&MemoryScope::new("alice", "ws"));
        filter.ids = vec!["a".to_owned()];
        filter.created_after = Some(at(1));
        let value = page_filter(
            &scope(),
            Some(&filter),
            Some(&MessageCursor::new(at(5), "id")),
        );
        assert_eq!(
            value,
            json!({
                "user_id": "alice",
                "scope_id": "ws",
                "session_id": "s1",
                "id": {"$in": ["a"]},
                "timestamp": {
                    "$gt": at(1).to_rfc3339(),
                    "$gte": at(5).to_rfc3339(),
                },
            })
        );
    }

    #[test]
    fn contradicting_clauses_match_nothing_instead_of_widening() {
        let filter = MemoryFilter::for_scope(&MemoryScope::new("bob", "other"));
        let value = page_filter(&scope(), Some(&filter), None);
        assert_eq!(
            value,
            json!({
                "user_id": {"$in": []},
                "scope_id": {"$in": []},
                "session_id": "s1",
            })
        );
        let equal = MemoryFilter::for_scope(&scope());
        assert_eq!(
            page_filter(&scope(), Some(&equal), None),
            json!({"user_id": "alice", "scope_id": "ws", "session_id": "s1"})
        );
    }

    #[test]
    fn timestamp_bounds_compose_instead_of_relaxing() {
        let mut filter = MemoryFilter::for_scope(&scope());
        filter.created_after = Some(at(9));
        let value = page_filter(
            &scope(),
            Some(&filter),
            Some(&MessageCursor::new(at(5), "id")),
        );
        assert_eq!(
            value[FIELD_TIMESTAMP],
            json!({"$gt": at(9).to_rfc3339(), "$gte": at(5).to_rfc3339()}),
            "an inclusive cursor bound must not relax an exclusive caller bound"
        );
        let mut upper = MemoryFilter::for_scope(&scope());
        upper.created_before = Some(at(4));
        let value = page_filter(
            &scope(),
            Some(&upper),
            Some(&MessageCursor::new(at(5), "id")),
        );
        assert_eq!(
            value[FIELD_TIMESTAMP],
            json!({"$lt": at(4).to_rfc3339(), "$gte": at(5).to_rfc3339()})
        );
        assert_eq!(
            combine(
                &json!({"$gte": at(9).to_rfc3339()}),
                &json!({"$gte": at(5).to_rfc3339()})
            ),
            json!({"$gte": at(9).to_rfc3339()}),
            "the same operator keeps the stricter lower bound"
        );
        assert_eq!(
            combine(
                &json!({"$lt": at(3).to_rfc3339()}),
                &json!({"$lt": at(8).to_rfc3339()})
            ),
            json!({"$lt": at(3).to_rfc3339()}),
            "the same operator keeps the stricter upper bound"
        );
        assert_eq!(
            combine(&json!({"$in": ["a", "b"]}), &json!({"$in": ["b", "c"]})),
            json!({"$in": ["b"]})
        );
        assert_eq!(
            combine(&json!({"$in": ["a"]}), &json!({"$in": ["b"]})),
            match_nothing()
        );
    }
}
