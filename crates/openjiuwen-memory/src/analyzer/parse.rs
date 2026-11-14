//! Tolerant parsing of model answers.
//!
//! Models wrap JSON in Markdown fences, add prose, rename fields, and occasionally emit
//! an entry that cannot be used. The helpers here accept the common shapes and skip
//! unusable entries with a `tracing::warn!` instead of failing the whole batch: an
//! answer is only rejected when nothing usable is left.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::conflict::{ConflictKind, ConflictVerdict};
use crate::analyzer::fact::{truncate_chars, ExtractedFact};
use crate::record::{normalize_text, MemoryRecord, MemoryType};
use serde_json::{Map, Value};

/// Parses an extraction answer into facts.
///
/// Accepted shapes: an array of fact objects, an object holding a `facts`, `memories`,
/// or `items` array, or a single fact object. Field aliases are `content`/`text`/
/// `fact`, `type`/`kind`/`memory_type`, and `metadata`/`fields`. The result is `None`
/// when no entry is usable; a parsed kind the store cannot write becomes
/// [`MemoryType::SemanticMemory`], and an absent or non-numeric `importance` becomes
/// [`AnalyzerConfig::default_importance`].
#[must_use]
pub fn parse_fact_response(raw: &str, config: &AnalyzerConfig) -> Option<Vec<ExtractedFact>> {
    let value = payload(raw)?;
    let items = match &value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(map) => {
            let nested = ["facts", "memories", "items"]
                .iter()
                .find_map(|key| map.get(*key));
            match nested {
                Some(Value::Array(items)) => items.iter().collect(),
                Some(item @ Value::Object(_)) => vec![item],
                _ if map.contains_key("content") || map.contains_key("text") => vec![&value],
                _ => return None,
            }
        }
        _ => return None,
    };
    let facts: Vec<ExtractedFact> = items
        .into_iter()
        .filter_map(|item| parse_fact(item, config))
        .collect();
    if facts.is_empty() {
        None
    } else {
        Some(facts)
    }
}

/// Parses one fact object, tolerating alias field names.
fn parse_fact(value: &Value, config: &AnalyzerConfig) -> Option<ExtractedFact> {
    let map = value.as_object()?;
    let content = string_field(map, &["content", "text", "fact"])?.trim();
    if content.is_empty() {
        tracing::warn!(
            target: "openjiuwen_memory::analyzer",
            "skipping extracted fact with empty content"
        );
        return None;
    }
    let kind = match string_field(map, &["type", "kind", "memory_type"]) {
        Some(raw) => {
            let parsed: MemoryType = raw.trim().parse().unwrap_or_default();
            if parsed.is_searchable() {
                parsed
            } else {
                tracing::warn!(
                    target: "openjiuwen_memory::analyzer",
                    %raw,
                    "extracted fact kind is not writable; using semantic_memory"
                );
                MemoryType::SemanticMemory
            }
        }
        None => MemoryType::SemanticMemory,
    };
    let importance = map
        .get("importance")
        .and_then(clamped_importance)
        .unwrap_or(config.default_importance);
    Some(ExtractedFact {
        content: truncate_chars(content, config.max_content_chars),
        kind,
        importance,
        tags: string_list(map.get("tags")),
        source: string_field(map, &["source", "origin"])
            .map(str::trim)
            .filter(|source| !source.is_empty())
            .map(str::to_owned),
        metadata: metadata_field(map),
    })
}

/// Parses a conflict-classification answer into one verdict per candidate.
///
/// The accepted item shape is Python's
/// `{"info_id", "info_text", "result", "related_infos", "reason", "merged_content"}`,
/// as a root array or under a `verdicts`/`items`/`results` key. Items are matched to
/// candidates by id first and then by normalized text, so a model that echoes a new
/// memory's own id still lands on the right existing record. Verdicts keep the input
/// order of `candidates`; candidates the answer never mentions are
/// [`ConflictKind::Unrelated`]. The result is `None` when nothing is parseable or no
/// item names a supplied candidate.
#[must_use]
pub fn parse_conflict_response(
    raw: &str,
    candidates: &[MemoryRecord],
) -> Option<Vec<ConflictVerdict>> {
    if candidates.is_empty() {
        return None;
    }
    let value = payload(raw)?;
    let items = match &value {
        Value::Array(items) => items.iter().collect::<Vec<_>>(),
        Value::Object(map) => {
            let nested = ["verdicts", "items", "results"]
                .iter()
                .find_map(|key| map.get(*key));
            match nested {
                Some(Value::Array(items)) => items.iter().collect(),
                Some(item @ Value::Object(_)) => vec![item],
                _ if map.contains_key("result") => vec![&value],
                _ => return None,
            }
        }
        _ => return None,
    };
    let mut verdicts: Vec<Option<ConflictVerdict>> = vec![None; candidates.len()];
    let mut matched_any = false;
    for item in items {
        let Some((index, verdict)) = parse_verdict(item, candidates) else {
            continue;
        };
        if verdicts[index].is_some() {
            tracing::warn!(
                target: "openjiuwen_memory::analyzer",
                "skipping a second verdict for the same candidate"
            );
            continue;
        }
        verdicts[index] = Some(verdict);
        matched_any = true;
    }
    if !matched_any {
        return None;
    }
    Some(
        verdicts
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect(),
    )
}

/// Parses one verdict object into a candidate index and its verdict.
fn parse_verdict(value: &Value, candidates: &[MemoryRecord]) -> Option<(usize, ConflictVerdict)> {
    let map = value.as_object()?;
    let kind = parse_conflict_kind(string_field(map, &["result", "verdict", "status"])?)?;
    let info_id = string_field(map, &["info_id", "id"]).map(str::trim);
    let info_text = string_field(map, &["info_text", "text", "content"]).map(str::trim);
    let index = info_id
        .filter(|id| !id.is_empty())
        .and_then(|id| candidates.iter().position(|record| record.id == id))
        .or_else(|| {
            let normalized = normalize_text(info_text?);
            if normalized.is_empty() {
                return None;
            }
            candidates
                .iter()
                .position(|record| normalize_text(&record.content) == normalized)
        })?;
    let related = map.get("related_infos").and_then(Value::as_object);
    let related_target = related.and_then(|entries| {
        entries
            .keys()
            .find(|id| candidates.iter().any(|record| record.id == id.as_str()))
    });
    let target_id = match kind {
        ConflictKind::Unrelated => None,
        ConflictKind::Redundant | ConflictKind::Conflicting => {
            Some(related_target.map_or_else(|| candidates[index].id.clone(), Clone::clone))
        }
    };
    Some((
        index,
        ConflictVerdict {
            kind,
            target_id,
            reason: string_field(map, &["reason", "why"])
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .map(str::to_owned),
            merged_content: string_field(map, &["merged_content", "merged", "content"])
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map(str::to_owned),
            degraded: None,
        },
    ))
}

/// Parses a verdict result name, accepting Python's `none` for `unrelated`.
fn parse_conflict_kind(raw: &str) -> Option<ConflictKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "redundant" | "duplicate" => Some(ConflictKind::Redundant),
        "conflicting" | "conflict" | "contradiction" => Some(ConflictKind::Conflicting),
        "none" | "unrelated" | "coexist" | "distinct" => Some(ConflictKind::Unrelated),
        other => {
            tracing::warn!(
                target: "openjiuwen_memory::analyzer",
                result = other,
                "skipping a verdict with an unknown result"
            );
            None
        }
    }
}

/// Cuts the JSON payload out of a model answer.
///
/// The first fenced block wins when present (with a `json` language tag removed);
/// otherwise a payload is sliced out of surrounding prose.
pub(crate) fn payload(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let body = fenced_body(trimmed).unwrap_or(trimmed);
    let slice = json_slice(body)?;
    match serde_json::from_str(slice) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::warn!(
                target: "openjiuwen_memory::analyzer",
                %error,
                "model answer is not valid JSON"
            );
            None
        }
    }
}

/// Returns the content of the first fenced code block, minus a `json` language tag.
fn fenced_body(raw: &str) -> Option<&str> {
    let open = raw.find("```")?;
    let after_open = raw.get(open + 3..)?;
    let close = after_open.find("```")?;
    Some(strip_language_tag(after_open.get(..close)?))
}

/// Drops a leading `json`/`JSON` tag from a fenced block body.
fn strip_language_tag(body: &str) -> &str {
    let trimmed = body.trim_start();
    let Some(head) = trimmed.get(..4) else {
        return trimmed;
    };
    if !head.eq_ignore_ascii_case("json") {
        return trimmed;
    }
    let rest = trimmed.get(4..).unwrap_or_default();
    let followed_by_whitespace = match rest.chars().next() {
        Some(ch) => ch.is_whitespace(),
        None => true,
    };
    if followed_by_whitespace {
        rest.trim_start()
    } else {
        trimmed
    }
}

/// Cuts the outermost JSON payload out of surrounding prose.
fn json_slice(text: &str) -> Option<&str> {
    let object_start = text.find('{');
    let array_start = text.find('[');
    let (start, close) = match (object_start, array_start) {
        (Some(object), Some(array)) if array < object => (array, ']'),
        (Some(object), _) => (object, '}'),
        (None, Some(array)) => (array, ']'),
        (None, None) => return None,
    };
    let end = text.rfind(close)?;
    if end <= start {
        return None;
    }
    text.get(start..=end)
}

/// Returns the first present string field among `keys`, trimmed later by the caller.
fn string_field<'a>(map: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
}

/// Collects a tag list from an array of strings or a comma-separated string.
fn string_list(value: Option<&Value>) -> Vec<String> {
    let mut tags = Vec::new();
    match value {
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(tag) = item.as_str().map(str::trim).filter(|tag| !tag.is_empty()) {
                    tags.push(tag.to_owned());
                }
            }
        }
        Some(Value::String(raw)) => {
            for tag in raw.split(',') {
                let tag = tag.trim();
                if !tag.is_empty() {
                    tags.push(tag.to_owned());
                }
            }
        }
        _ => {}
    }
    tags.dedup();
    tags
}

/// Collects the `metadata`/`fields` object of a fact.
fn metadata_field(map: &Map<String, Value>) -> std::collections::BTreeMap<String, Value> {
    map.get("metadata")
        .or_else(|| map.get("fields"))
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Reads an `importance` number, rejecting non-finite values and clamping to `[0, 1]`.
fn clamped_importance(value: &Value) -> Option<f32> {
    let number = value.as_f64()?;
    if !number.is_finite() {
        return None;
    }
    Some(number.clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MemoryScope, NewMemory};

    fn config() -> AnalyzerConfig {
        AnalyzerConfig::default()
    }

    fn record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            content,
            MemoryType::SemanticMemory,
        )
        .with_id(id)
    }

    #[test]
    fn parses_a_plain_array_with_aliases() {
        let raw = r#"[
            {"content": "Alice prefers dark mode", "type": "user_profile", "importance": 0.8,
             "tags": ["preference", "ui"], "source": "conversation"},
            {"text": "the service uses postgres", "kind": "semantic_memory", "tags": "project, db"},
            {"fact": "Alice is on the platform team", "memory_type": "nonsense"}
        ]"#;
        let facts = parse_fact_response(raw, &config()).unwrap();
        assert_eq!(facts.len(), 3);
        assert_eq!(facts[0].content, "Alice prefers dark mode");
        assert_eq!(facts[0].kind, MemoryType::UserProfile);
        assert!((facts[0].importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(facts[0].tags, vec!["preference", "ui"]);
        assert_eq!(facts[0].source.as_deref(), Some("conversation"));
        assert_eq!(facts[1].tags, vec!["project", "db"]);
        assert_eq!(facts[1].kind, MemoryType::SemanticMemory);
        assert!((facts[1].importance - 0.5).abs() < f32::EPSILON);
        assert_eq!(facts[2].kind, MemoryType::SemanticMemory);
    }

    #[test]
    fn parses_fenced_and_prose_wrapped_answers() {
        let raw =
            "Sure! Here you go:\n```json\n{\"facts\": [{\"content\": \"I use tokio\"}]}\n```\n";
        let facts = parse_fact_response(raw, &config()).unwrap();
        assert_eq!(facts[0].content, "I use tokio");

        let raw = "prefix [{\"content\": \"I use tokio\"}] suffix";
        assert_eq!(
            parse_fact_response(raw, &config()).unwrap()[0].content,
            "I use tokio"
        );

        let raw = "```JSON\n{\"memories\": [{\"content\": \"I use tokio\"}]}\n```";
        assert!(parse_fact_response(raw, &config()).is_some());

        let raw = r#"{"items": [{"content": "I use tokio"}]}"#;
        assert!(parse_fact_response(raw, &config()).is_some());
    }

    #[test]
    fn rejects_unusable_fact_answers() {
        for raw in [
            "",
            "   ",
            "no json here",
            "{}",
            "[]",
            r#"{"facts": []}"#,
            "{not json}",
            r#"[{"content": "   "}]"#,
            r#"[{"importance": 0.4}]"#,
        ] {
            assert!(parse_fact_response(raw, &config()).is_none(), "{raw}");
        }
    }

    #[test]
    fn skips_malformed_entries_but_keeps_good_ones() {
        let raw = r#"[
            {"content": ""},
            {"content": "valid fact"},
            42,
            {"content": "another fact", "importance": "high"}
        ]"#;
        let facts = parse_fact_response(raw, &config()).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].content, "valid fact");
        assert!((facts[1].importance - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn clamps_importance_and_truncates_content() {
        let config = AnalyzerConfig {
            max_content_chars: 8,
            ..Default::default()
        };
        let raw = r#"[{"content": "0123456789", "importance": 9.0},
                     {"content": "abc", "importance": -2.0},
                     {"content": "def", "importance": null}]"#;
        let facts = parse_fact_response(raw, &config).unwrap();
        assert_eq!(facts[0].content, "01234567");
        assert!((facts[0].importance - 1.0).abs() < f32::EPSILON);
        assert!((facts[1].importance - 0.0).abs() < f32::EPSILON);
        assert!((facts[2].importance - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn non_writable_kinds_become_semantic_memories() {
        let raw = r#"[{"content": "port = 8080", "type": "variable"},
                     {"content": "unknown kind", "type": "bogus"}]"#;
        let facts = parse_fact_response(raw, &config()).unwrap();
        assert_eq!(facts[0].kind, MemoryType::SemanticMemory);
        assert_eq!(facts[1].kind, MemoryType::SemanticMemory);
        let raw = r#"[{"content": "cargo fmt is required", "type": "coding"}]"#;
        assert_eq!(
            parse_fact_response(raw, &config()).unwrap()[0].kind,
            MemoryType::Coding
        );
    }

    #[test]
    fn metadata_aliases_are_read() {
        let raw = r#"[{"content": "x", "metadata": {"a": 1}},
                     {"content": "y", "fields": {"b": 2}}]"#;
        let facts = parse_fact_response(raw, &config()).unwrap();
        assert_eq!(facts[0].metadata.get("a"), Some(&serde_json::json!(1)));
        assert_eq!(facts[1].metadata.get("b"), Some(&serde_json::json!(2)));
    }

    #[test]
    fn parses_conflict_verdicts_in_candidate_order() {
        let candidates = vec![
            record("id-1", "Alice works at Globex"),
            record("id-2", "The service uses postgres"),
            record("id-3", "Alice lives in Berlin"),
        ];
        let raw = r#"[
            {"info_id": "new-1", "info_text": "Alice lives in Berlin", "result": "redundant",
             "related_infos": {"id-3": "Alice lives in Berlin"}, "reason": "same fact"},
            {"info_id": "new-2", "info_text": "Alice works at Globex", "result": "conflicting",
             "related_infos": {"id-1": "Alice works at Globex"},
             "merged_content": "Alice works at Acme"}
        ]"#;
        let verdicts = parse_conflict_response(raw, &candidates).unwrap();
        assert_eq!(verdicts.len(), 3);
        assert_eq!(verdicts[0].kind, ConflictKind::Conflicting);
        assert_eq!(verdicts[0].target_id.as_deref(), Some("id-1"));
        assert_eq!(
            verdicts[0].merged_content.as_deref(),
            Some("Alice works at Acme")
        );
        assert_eq!(verdicts[1].kind, ConflictKind::Unrelated);
        assert!(verdicts[1].target_id.is_none());
        assert_eq!(verdicts[2].kind, ConflictKind::Redundant);
        assert_eq!(verdicts[2].target_id.as_deref(), Some("id-3"));
        assert_eq!(verdicts[2].reason.as_deref(), Some("same fact"));
    }

    #[test]
    fn conflict_verdicts_match_by_id_first() {
        let candidates = vec![record("id-1", "some text")];
        let raw = r#"[{"info_id": "id-1", "info_text": "different text", "result": "none"}]"#;
        let verdicts = parse_conflict_response(raw, &candidates).unwrap();
        assert_eq!(verdicts[0].kind, ConflictKind::Unrelated);
        assert!(verdicts[0].target_id.is_none());
    }

    #[test]
    fn conflict_verdicts_accept_a_wrapped_object() {
        let candidates = vec![record("id-1", "Alice works at Globex")];
        let raw =
            r#"{"verdicts": [{"info_text": "Alice works at Globex", "result": "redundant"}]}"#;
        let verdicts = parse_conflict_response(raw, &candidates).unwrap();
        assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
        assert_eq!(verdicts[0].target_id.as_deref(), Some("id-1"));
    }

    #[test]
    fn conflict_parsing_rejects_unusable_answers() {
        let candidates = vec![record("id-1", "Alice works at Globex")];
        for raw in ["", "nope", "{}", "[]", r#"[{"result": "redundant"}]"#] {
            assert!(parse_conflict_response(raw, &candidates).is_none(), "{raw}");
        }
        // An answer that names no supplied candidate is unusable.
        let raw = r#"[{"info_id": "other", "result": "redundant"}]"#;
        assert!(parse_conflict_response(raw, &candidates).is_none());
        // A candidate list of zero is unusable too.
        assert!(parse_conflict_response("[]", &[]).is_none());
    }

    #[test]
    fn conflict_parsing_ignores_unknown_results_and_duplicates() {
        let candidates = vec![
            record("id-1", "Alice works at Globex"),
            record("id-2", "The service uses postgres"),
        ];
        let raw = r#"[
            {"info_id": "id-1", "result": "maybe"},
            {"info_id": "id-1", "result": "redundant"},
            {"info_id": "id-1", "result": "conflicting"},
            {"info_id": "id-2", "result": "none"}
        ]"#;
        let verdicts = parse_conflict_response(raw, &candidates).unwrap();
        assert_eq!(verdicts[0].kind, ConflictKind::Redundant);
        assert_eq!(verdicts[1].kind, ConflictKind::Unrelated);
    }

    #[test]
    fn parsing_never_panics_on_hostile_input() {
        let candidates = vec![record("id-1", "text")];
        let huge = format!("[{{\"content\": \"{}\"}}]", "x".repeat(50_000));
        assert!(parse_fact_response(&huge, &config()).is_some());
        for raw in ["{", "}", "[[[[", "```json", "```\n```", "null", "\u{0}"] {
            let _ = parse_fact_response(raw, &config());
            let _ = parse_conflict_response(raw, &candidates);
        }
        let _ = parse_conflict_response(&huge, &candidates);
        let _ = parse_fact_response("", &config());
        let _: ExtractedFact = ExtractedFact::from(&NewMemory::semantic("x"));
    }
}
