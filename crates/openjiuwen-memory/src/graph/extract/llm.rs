//! Model-backed entity/relation extraction.
//!
//! [`LlmExtractor`] asks a [`Model`] for JSON in the shape pinned by
//! [`EXTRACTION_SYSTEM_PROMPT`] and parses the answer tolerantly, falling back to the
//! heuristic extractor when the model output stays unusable.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::{Map, Value};

use super::heuristic::HeuristicExtractor;
use super::{EntityExtractor, ExtractedEntity, ExtractedRelation, Extraction};
use crate::config::{ExtractionConfig, GraphMemoryConfig};
use crate::record::{CallOptions, MemoryScope};

/// System prompt for [`LlmExtractor`].
///
/// It pins the exact JSON shape the parser accepts and nothing else, so a compliant
/// model answer needs no repair pass.
pub const EXTRACTION_SYSTEM_PROMPT: &str = concat!(
    "Extract entities and relations from the user text. ",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"entities\":[{\"name\":\"...\",\"type\":\"...\",\"importance\":0.0}],",
    "\"relations\":[{\"from\":\"...\",\"to\":\"...\",\"type\":\"...\"}]}. ",
    "Use a short `type` label such as Person, Project, Concept, or Organization ",
    "for entities and a short verb-like label such as works_at for relations. ",
    "`importance` must be a number between 0.0 and 1.0. ",
    "Do not add prose, Markdown fences, or explanations."
);

/// Model-backed extractor with heuristic fallback.
///
/// One [`Model::invoke`] call per attempt with [`EXTRACTION_SYSTEM_PROMPT`]; the answer
/// is parsed tolerantly by `parse_extraction`. Failed calls and unusable answers are
/// retried, and once the retry budget is spent the extractor falls back to
/// [`HeuristicExtractor`] so graph memory keeps working without a healthy model.
pub struct LlmExtractor {
    /// Model used for extraction.
    model: Arc<dyn Model>,
    /// Number of retries after the first attempt (`ExtractionConfig::model_retries`).
    retries: u32,
    /// Graph limits; the fallback heuristic and entity cap come from here.
    config: GraphMemoryConfig,
}

impl LlmExtractor {
    /// Builds an extractor around `model` using the graph limits.
    ///
    /// The retry budget is `ExtractionConfig::default().model_retries`, keeping it in
    /// step with the message-extraction path instead of a magic number.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: &GraphMemoryConfig) -> Self {
        Self {
            model,
            retries: ExtractionConfig::default().model_retries,
            config: config.clone(),
        }
    }
}

#[async_trait]
impl EntityExtractor for LlmExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // trait signature returns `&str` tied to `self`
    fn name(&self) -> &str {
        "llm"
    }

    async fn extract(
        &self,
        text: &str,
        scope: &MemoryScope,
        options: &CallOptions,
    ) -> Result<Extraction> {
        options.check()?;
        let messages = vec![
            BaseMessage::system(EXTRACTION_SYSTEM_PROMPT),
            BaseMessage::user(text),
        ];
        let request = ModelRequestConfig::new(self.model.model_name().to_owned());
        let attempts = self.retries.saturating_add(1);
        for attempt in 1..=attempts {
            let outcome = options.run(self.model.invoke(&messages, &request)).await;
            let response = match outcome {
                Ok(Ok(response)) => response,
                Ok(Err(error)) | Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    tracing::warn!(
                        target: "openjiuwen_memory::graph::extract",
                        attempt,
                        %error,
                        "LLM entity extraction call failed; retrying"
                    );
                    continue;
                }
            };
            let raw = response
                .message()
                .map(|message| message.text().into_owned());
            if let Some(mut extraction) = raw.as_deref().and_then(parse_extraction) {
                if self.config.max_entities > 0 {
                    extraction.entities.truncate(self.config.max_entities);
                }
                return Ok(extraction);
            }
            tracing::warn!(
                target: "openjiuwen_memory::graph::extract",
                attempt,
                "LLM entity extraction returned unusable output; retrying"
            );
        }
        tracing::warn!(
            target: "openjiuwen_memory::graph::extract",
            attempts,
            "LLM entity extraction exhausted its retries; falling back to heuristics"
        );
        HeuristicExtractor::new(&self.config)
            .extract(text, scope, options)
            .await
    }
}

/// Parses a model answer into an [`Extraction`], returning `None` when unusable.
///
/// The parser accepts a fenced code block, an object or array under `entities`/
/// `relations`, alias field names, and surrounding prose around the JSON payload.
/// Blank names and incomplete or self-referential relations are skipped; an answer
/// with neither entities nor relations is rejected.
fn parse_extraction(raw: &str) -> Option<Extraction> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let body = fenced_body(trimmed).unwrap_or(trimmed);
    let slice = json_slice(body)?;
    let value: Value = serde_json::from_str(slice).ok()?;
    let (entities, relations) = match &value {
        Value::Array(items) => (parse_entities(items), Vec::new()),
        Value::Object(map) => {
            let entities = parse_entities_field(map.get("entities"));
            let relations = parse_relations_field(map.get("relations"));
            (entities, relations)
        }
        _ => return None,
    };
    if entities.is_empty() && relations.is_empty() {
        None
    } else {
        Some(Extraction {
            entities,
            relations,
        })
    }
}

/// Returns the content of the first fenced code block, minus a `json` language tag.
fn fenced_body(raw: &str) -> Option<&str> {
    let open = raw.find("```")?;
    let after_open = raw.get(open + 3..)?;
    let close = after_open.find("```")?;
    let body = after_open.get(..close)?;
    Some(strip_language_tag(body))
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
    let rest = trimmed.get(4..).unwrap_or("");
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

/// Cuts the JSON payload out of surrounding prose.
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

/// Expands an `entities`/`relations` value into items (array or single object).
fn value_items(value: Option<&Value>) -> Vec<&Value> {
    match value {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(item) if item.is_object() => vec![item],
        _ => Vec::new(),
    }
}

/// Parses an already-collected entity array.
fn parse_entities(items: &[Value]) -> Vec<ExtractedEntity> {
    items.iter().filter_map(parse_entity).collect()
}

/// Parses the `entities` field.
fn parse_entities_field(value: Option<&Value>) -> Vec<ExtractedEntity> {
    value_items(value)
        .into_iter()
        .filter_map(parse_entity)
        .collect()
}

/// Parses the `relations` field.
fn parse_relations_field(value: Option<&Value>) -> Vec<ExtractedRelation> {
    value_items(value)
        .into_iter()
        .filter_map(parse_relation)
        .collect()
}

/// Parses one entity object, tolerating alias field names.
fn parse_entity(value: &Value) -> Option<ExtractedEntity> {
    let map = value.as_object()?;
    let name = string_field(map, &["name", "entity", "text"])?.trim();
    if name.is_empty() {
        return None;
    }
    let entity_type = string_field(map, &["type", "entity_type"])
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .unwrap_or("Concept");
    Some(ExtractedEntity {
        name: name.to_owned(),
        entity_type: entity_type.to_owned(),
        importance: map.get("importance").and_then(clamped_importance),
    })
}

/// Parses one relation object, tolerating alias field names and skipping self-links.
fn parse_relation(value: &Value) -> Option<ExtractedRelation> {
    let map = value.as_object()?;
    let from = string_field(map, &["from", "source", "from_entity"])?.trim();
    let to = string_field(map, &["to", "target", "to_entity"])?.trim();
    let relation_type = string_field(map, &["type", "relation_type"])?.trim();
    if from.is_empty() || to.is_empty() || relation_type.is_empty() {
        return None;
    }
    if from.eq_ignore_ascii_case(to) {
        return None;
    }
    Some(ExtractedRelation {
        from: from.to_owned(),
        to: to.to_owned(),
        relation_type: relation_type.to_owned(),
    })
}

/// Returns the first present string field among `keys`.
fn string_field<'a>(map: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
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

    use openjiuwen_common::error::Error;

    use openjiuwen_test_support::ScriptedModel;

    use tokio_util::sync::CancellationToken;

    fn scope() -> MemoryScope {
        MemoryScope::new("user", "scope")
    }

    #[test]
    fn parse_extraction_reads_plain_json_objects() {
        let raw = r#"{
            "entities": [
                {"name": "Alice", "type": "Person", "importance": 0.9},
                {"name": "Acme", "type": "Organization"}
            ],
            "relations": [{"from": "Alice", "to": "Acme", "type": "works_at"}]
        }"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(
            extraction.entities,
            vec![
                ExtractedEntity {
                    name: "Alice".into(),
                    entity_type: "Person".into(),
                    importance: Some(0.9),
                },
                ExtractedEntity {
                    name: "Acme".into(),
                    entity_type: "Organization".into(),
                    importance: None,
                },
            ]
        );
        assert_eq!(
            extraction.relations,
            vec![ExtractedRelation {
                from: "Alice".into(),
                to: "Acme".into(),
                relation_type: "works_at".into(),
            }]
        );
    }

    #[test]
    fn parse_extraction_reads_fenced_blocks() {
        let raw = "Here you go:\n```json\n{\"entities\": [{\"name\": \"Bob\"}]}\n```\n";
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities.len(), 1);
        assert_eq!(extraction.entities[0].name, "Bob");
        assert_eq!(extraction.entities[0].entity_type, "Concept");

        let raw = "```\n[{\"name\": \"Bob\"}]\n```";
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities[0].name, "Bob");
    }

    #[test]
    fn parse_extraction_reads_root_arrays_and_single_objects() {
        let extraction = parse_extraction(r#"[{"name": "Bob"}, {"entity": "Acme"}]"#).unwrap();
        assert_eq!(extraction.entities.len(), 2);
        assert_eq!(extraction.relations, Vec::<ExtractedRelation>::new());

        let raw = r#"{
            "entities": {"name": "Acme", "entity_type": "Organization", "importance": 2.0},
            "relations": {"source": "Bob", "target": "Acme", "relation_type": "works_at"}
        }"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities[0].entity_type, "Organization");
        assert_eq!(extraction.entities[0].importance, Some(1.0));
        assert_eq!(extraction.relations[0].from, "Bob");
        assert_eq!(extraction.relations[0].relation_type, "works_at");
    }

    #[test]
    fn parse_extraction_rejects_unusable_output() {
        assert!(parse_extraction("").is_none());
        assert!(parse_extraction("no json here").is_none());
        assert!(parse_extraction("{}").is_none());
        assert!(parse_extraction(r#"{"entities": []}"#).is_none());
        assert!(parse_extraction("{not json}").is_none());
    }

    #[test]
    fn parse_extraction_skips_blank_names_and_clamps_importance() {
        let raw = r#"{"entities": [
            {"name": "   "},
            {"name": "Bob", "importance": -3.0},
            {"name": "Acme", "importance": 4.0}
        ]}"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities.len(), 2);
        assert_eq!(extraction.entities[0].importance, Some(0.0));
        assert_eq!(extraction.entities[1].importance, Some(1.0));

        let raw = r#"{"entities": [{"name": "Bob", "importance": "high"}]}"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(extraction.entities[0].importance, None);
    }

    #[test]
    fn parse_extraction_skips_self_and_incomplete_relations() {
        let raw = r#"{
            "entities": [{"name": "Bob"}],
            "relations": [
                {"from": "Bob", "to": "Bob", "type": "knows"},
                {"from": "Bob", "to": "", "type": "knows"},
                {"from": "Bob", "to": "Acme"},
                {"from": "Bob", "to": "Acme", "type": "works_at"}
            ]
        }"#;
        let extraction = parse_extraction(raw).unwrap();
        assert_eq!(
            extraction.relations,
            vec![ExtractedRelation {
                from: "Bob".into(),
                to: "Acme".into(),
                relation_type: "works_at".into(),
            }]
        );
    }

    #[tokio::test]
    async fn llm_extractor_parses_scripted_json() {
        let model = ScriptedModel::builder()
            .respond_text(
                r#"{"entities":[{"name":"Alice","type":"Person","importance":0.9}],
                    "relations":[{"from":"Alice","to":"Acme","type":"works_at"}]}"#,
            )
            .build();
        let handle = model.clone();
        let extractor = LlmExtractor::new(Arc::new(model), &GraphMemoryConfig::default());
        let extraction = extractor
            .extract("Alice works at Acme", &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(
            extraction.entities,
            vec![ExtractedEntity {
                name: "Alice".into(),
                entity_type: "Person".into(),
                importance: Some(0.9),
            }]
        );
        assert_eq!(extraction.relations.len(), 1);
        assert_eq!(handle.call_count(), 1);
        assert_eq!(handle.remaining(), 0);
    }

    #[tokio::test]
    async fn llm_extractor_accepts_fenced_output_and_caps_entities() {
        let model = ScriptedModel::builder()
            .respond_text("```json\n{\"entities\":[{\"name\":\"Alice\"},{\"name\":\"Bob\"}]}\n```")
            .build();
        let config = GraphMemoryConfig {
            max_entities: 1,
            ..Default::default()
        };
        let extractor = LlmExtractor::new(Arc::new(model), &config);
        let extraction = extractor
            .extract("Alice and Bob", &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(extraction.entities.len(), 1);
        assert_eq!(extraction.entities[0].name, "Alice");
    }

    #[tokio::test]
    async fn llm_extractor_falls_back_to_heuristics_after_retries() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let config = GraphMemoryConfig::default();
        let extractor = LlmExtractor::new(Arc::new(model), &config);
        let text = "I heard Bob works at Acme";
        let extraction = extractor
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap();
        let expected = HeuristicExtractor::new(&config)
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(extraction, expected);
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn llm_extractor_returns_cancelled_without_calling_the_model() {
        let model = ScriptedModel::text(r#"{"entities":[{"name":"Alice"}]}"#);
        let handle = model.clone();
        let extractor = LlmExtractor::new(Arc::new(model), &GraphMemoryConfig::default());
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = CallOptions::new().with_cancellation(cancellation);
        let error = extractor
            .extract("Alice", &scope(), &options)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn llm_extractor_sends_the_system_prompt_and_user_text() {
        let model = ScriptedModel::builder()
            .respond_text(r#"{"entities":[{"name":"Alice"}]}"#)
            .build();
        let handle = model.clone();
        let extractor = LlmExtractor::new(Arc::new(model), &GraphMemoryConfig::default());
        let text = "Alice prefers dark mode";
        extractor
            .extract(text, &scope(), &CallOptions::new())
            .await
            .unwrap();
        let call = handle.last_call().expect("the model was called");
        let system = call.messages.first().expect("a system message");
        assert!(system.text().contains("entities"));
        assert_eq!(call.last_user_text().as_deref(), Some(text));
    }
}
