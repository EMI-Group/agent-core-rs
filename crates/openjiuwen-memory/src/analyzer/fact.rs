//! Analyzer input/output types and the [`FactExtractor`] trait.
//!
//! [`ExtractionInput`] is the provider-neutral conversation snapshot an extractor
//! consumes; [`ExtractedFact`] is the write-ready fact it produces and
//! [`ExtractionResult`] wraps a batch with its provenance and any documented
//! degradation. Everything is plain `serde` data so a batch can be logged, asserted in
//! tests, or replayed later.
use crate::record::{CallOptions, MemoryType, NewMemory};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Importance used when a fact does not carry a usable one.
const DEFAULT_IMPORTANCE: f32 = 0.5;

/// Truncates `text` to at most `max_chars` characters on a character boundary.
///
/// Slicing by characters (not bytes) means arbitrary UTF-8 input — including CJK text
/// and emoji — can never be cut mid-codepoint.
pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((boundary, _)) => text.get(..boundary).unwrap_or_default().to_owned(),
        None => text.to_owned(),
    }
}

/// One conversation message handed to an extractor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConversationTurn {
    /// Author role, using the provider role names (`user`, `assistant`, `system`).
    pub role: String,
    /// The message text.
    pub text: String,
}

impl ConversationTurn {
    /// Builds a turn from a role and its text.
    #[must_use]
    pub fn new(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            text: text.into(),
        }
    }

    /// Returns `true` when the text holds no non-whitespace character.
    #[must_use]
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// The conversation snapshot an extractor analyzes.
///
/// `turns` are the messages being analyzed (newest last); `history` is the bounded
/// preceding context, rendered before `turns` and never extracted from on its own.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractionInput {
    /// The messages to extract facts from.
    pub turns: Vec<ConversationTurn>,
    /// Earlier messages provided as context.
    pub history: Vec<ConversationTurn>,
}

impl ExtractionInput {
    /// Builds an input from provider messages, treating every message as a current turn.
    #[must_use]
    pub fn from_messages(messages: &[BaseMessage]) -> Self {
        Self {
            turns: messages.iter().map(turn_from_message).collect(),
            history: Vec::new(),
        }
    }

    /// Builds an input with an explicit history prefix.
    #[must_use]
    pub fn with_history_messages(mut self, messages: &[BaseMessage]) -> Self {
        self.history = messages.iter().map(turn_from_message).collect();
        self
    }

    /// Builds an input from one user message.
    #[must_use]
    pub fn from_text(text: impl Into<String>) -> Self {
        let mut input = Self::default();
        input.push_user(text);
        input
    }

    /// Appends one turn.
    pub fn push_turn(&mut self, role: impl Into<String>, text: impl Into<String>) {
        self.turns.push(ConversationTurn::new(role, text));
    }

    /// Appends one `user` turn.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.push_turn("user", text);
    }

    /// Returns `true` when no current turn and no history turn carries text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.iter().all(ConversationTurn::is_blank)
            && self.history.iter().all(ConversationTurn::is_blank)
    }

    /// Returns the number of current turns (history is not counted).
    #[must_use]
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Validates that every turn names a role.
    ///
    /// An empty or whitespace-only role is `Error::Validation`; an empty text is not an
    /// error (extractors skip blank turns).
    pub fn validate(&self) -> Result<()> {
        let turns = self.turns.iter().chain(self.history.iter());
        for (index, turn) in turns.enumerate() {
            if turn.role.trim().is_empty() {
                return Err(Error::validation(format!(
                    "conversation turn {index} has an empty `role`"
                )));
            }
        }
        Ok(())
    }

    /// Returns a copy whose turn texts are truncated to `max_chars` characters each.
    ///
    /// Truncation is char-boundary safe, so the copy is always valid UTF-8.
    #[must_use]
    pub fn bounded(&self, max_chars: usize) -> Self {
        let bounded = |turn: &ConversationTurn| ConversationTurn {
            role: turn.role.clone(),
            text: truncate_chars(&turn.text, max_chars),
        };
        Self {
            turns: self.turns.iter().map(bounded).collect(),
            history: self.history.iter().map(bounded).collect(),
        }
    }
}

/// Converts one provider message into a [`ConversationTurn`].
fn turn_from_message(message: &BaseMessage) -> ConversationTurn {
    ConversationTurn {
        role: message.role.as_str().to_owned(),
        text: message.text().into_owned(),
    }
}

/// One memory fact produced by an extractor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedFact {
    /// The memory text; must contain a non-whitespace character.
    pub content: String,
    /// The memory kind; must be writable (see [`ExtractedFact::validate`]).
    pub kind: MemoryType,
    /// Importance in `0.0..=1.0`.
    pub importance: f32,
    /// Free-form labels such as `preference`, `identity`, or `explicit`.
    pub tags: Vec<String>,
    /// Optional provenance marker.
    pub source: Option<String>,
    /// Structured fields stored with the record.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ExtractedFact {
    /// An empty semantic fact at the default importance.
    fn default() -> Self {
        Self {
            content: String::new(),
            kind: MemoryType::SemanticMemory,
            importance: DEFAULT_IMPORTANCE,
            tags: Vec::new(),
            source: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl ExtractedFact {
    /// Builds a fact from its content and kind.
    #[must_use]
    pub fn new(content: impl Into<String>, kind: MemoryType) -> Self {
        Self {
            content: content.into(),
            kind,
            ..Self::default()
        }
    }

    /// Builds a [`MemoryType::SemanticMemory`] fact.
    #[must_use]
    pub fn semantic(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::SemanticMemory)
    }

    /// Builds a [`MemoryType::UserProfile`] fact.
    #[must_use]
    pub fn user_profile(content: impl Into<String>) -> Self {
        Self::new(content, MemoryType::UserProfile)
    }

    /// Returns the fact with `kind` replaced.
    #[must_use]
    pub fn with_kind(mut self, kind: MemoryType) -> Self {
        self.kind = kind;
        self
    }

    /// Returns the fact with `importance` replaced.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    /// Returns the fact with one tag appended.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Returns the fact with `tags` replaced.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Returns the fact with `source` replaced.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Returns the fact with one metadata field inserted (incoming value wins).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Validates the fact.
    ///
    /// Empty/whitespace-only content is rejected, as is a kind the memory store cannot
    /// write ([`MemoryType::Variable`] and [`MemoryType::Unknown`]); both are
    /// `Error::Validation`.
    pub fn validate(&self) -> Result<()> {
        if self.content.trim().is_empty() {
            return Err(Error::validation(
                "extracted fact `content` must not be empty or whitespace",
            ));
        }
        if !self.kind.is_searchable() {
            return Err(Error::validation(format!(
                "extracted fact `kind` must be writable (got {})",
                self.kind.as_str()
            )));
        }
        Ok(())
    }

    /// Converts the fact into a [`NewMemory`] write input.
    ///
    /// The importance is clamped into `0.0..=1.0` (non-finite becomes the default), and
    /// a non-writable kind ([`MemoryType::Variable`]/[`MemoryType::Unknown`]) becomes
    /// [`MemoryType::SemanticMemory`], so the conversion cannot produce an input the
    /// engine rejects on kind grounds.
    #[must_use]
    pub fn to_new_memory(&self) -> NewMemory {
        let importance = if self.importance.is_finite() {
            self.importance.clamp(0.0, 1.0)
        } else {
            DEFAULT_IMPORTANCE
        };
        let kind = if self.kind.is_searchable() {
            self.kind
        } else {
            MemoryType::SemanticMemory
        };
        NewMemory {
            content: self.content.clone(),
            kind,
            importance: Some(importance),
            metadata: self.metadata.clone(),
            source: self.source.clone(),
            timestamp: None,
        }
    }
}

impl From<&NewMemory> for ExtractedFact {
    /// Views a write input as a fact, keeping content, kind, importance, source, and
    /// metadata.
    fn from(memory: &NewMemory) -> Self {
        Self {
            content: memory.content.clone(),
            kind: memory.kind,
            importance: memory.importance.unwrap_or(DEFAULT_IMPORTANCE),
            tags: Vec::new(),
            source: memory.source.clone(),
            metadata: memory.metadata.clone(),
        }
    }
}

/// The outcome of one extraction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractionResult {
    /// The facts produced, in extractor order.
    pub facts: Vec<ExtractedFact>,
    /// Why a deterministic fallback ran, when one did.
    pub degraded: Option<String>,
    /// Name of the extractor that actually produced `facts` (`rule`/`llm`).
    pub extractor: String,
}

impl ExtractionResult {
    /// Builds a successful result.
    #[must_use]
    pub fn new(facts: Vec<ExtractedFact>, extractor: impl Into<String>) -> Self {
        Self {
            facts,
            degraded: None,
            extractor: extractor.into(),
        }
    }

    /// Builds an empty, non-degraded result.
    #[must_use]
    pub fn empty(extractor: impl Into<String>) -> Self {
        Self::new(Vec::new(), extractor)
    }

    /// Returns `true` when a fallback produced this result.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded.is_some()
    }
}

/// Turns a conversation snapshot into memory facts.
///
/// Implementations are `Send + Sync` so an analyzer can hold one behind an `Arc` and
/// serve many concurrent calls. Extraction never writes storage: the caller decides
/// what to persist (see [`crate::analyzer::MemoryAnalyzer::consolidate`]).
#[async_trait]
pub trait FactExtractor: Send + Sync {
    /// The extractor name reported on [`ExtractionResult::extractor`].
    fn name(&self) -> &str;

    /// Extracts facts from `input`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when `options` is already cancelled,
    /// [`Error::Validation`] for an input without roles, and the model provider's error
    /// when a model-backed extractor cannot recover.
    async fn extract(
        &self,
        input: &ExtractionInput,
        options: &CallOptions,
    ) -> Result<ExtractionResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::prompt::Role;
    use serde_json::json;

    #[test]
    fn truncate_chars_is_boundary_safe() {
        assert_eq!(truncate_chars("abcdef", 3), "abc");
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abc", 9), "abc");
        assert_eq!(truncate_chars("", 0), "");
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("日本語テキスト", 3), "日本語");
        assert_eq!(truncate_chars("日本語", 0), "");
    }

    #[test]
    fn turns_and_inputs_round_trip_through_json() {
        let turn = ConversationTurn::new("user", "my name is Alice");
        let json = serde_json::to_value(&turn).unwrap();
        assert_eq!(json["role"], json!("user"));
        assert_eq!(json["text"], json!("my name is Alice"));
        let restored: ConversationTurn = serde_json::from_value(json).unwrap();
        assert_eq!(restored, turn);

        let input = ExtractionInput::from_text("hello there");
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["turns"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["history"].as_array().map(Vec::len), Some(0));
        let restored: ExtractionInput = serde_json::from_value(json).unwrap();
        assert_eq!(restored, input);
        let tolerant: ExtractionInput =
            serde_json::from_value(json!({"turns": [], "unknown": true})).unwrap();
        assert_eq!(tolerant.len(), 0);
    }

    #[test]
    fn from_messages_reads_roles_and_text() {
        let messages = vec![
            BaseMessage::system("be brief"),
            BaseMessage::user("my name is Alice"),
        ];
        let input = ExtractionInput::from_messages(&messages);
        assert_eq!(input.len(), 2);
        assert_eq!(input.turns[0].role, Role::System.as_str());
        assert_eq!(input.turns[1].role, "user");
        assert_eq!(input.turns[1].text, "my name is Alice");
        assert_eq!(input.history.len(), 0);
        assert_ne!(input.len(), 0);
        assert!(input.validate().is_ok());

        let with_history =
            ExtractionInput::from_text("now").with_history_messages(&[BaseMessage::user("hi")]);
        assert_eq!(with_history.history.len(), 1);
        assert_eq!(with_history.history[0].text, "hi");
    }

    #[test]
    fn push_and_empty_helpers() {
        let mut input = ExtractionInput::default();
        assert!(input.is_empty(), "a fresh input is empty");
        assert_eq!(input.len(), 0);
        input.push_user("  ");
        assert!(input.is_empty(), "a blank turn keeps the input empty");
        assert_eq!(input.len(), 1);
        input.push_turn("assistant", "a fact");
        assert!(!input.is_empty(), "a turn with text is not empty");
        assert_eq!(input.len(), 2);
        assert_eq!(input.turns[1].role, "assistant");
    }

    #[test]
    fn validate_rejects_blank_roles() {
        let mut input = ExtractionInput::from_text("hello");
        input.turns[0].role = "   ".to_owned();
        let error = input.validate().unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("role"));

        let mut input = ExtractionInput::from_text("hello");
        input.history.push(ConversationTurn::new("", "old"));
        assert!(input.validate().is_err());
    }

    #[test]
    fn bounded_truncates_every_turn() {
        let input = ExtractionInput {
            turns: vec![ConversationTurn::new("user", "abcdefgh")],
            history: vec![ConversationTurn::new("user", "日本語テキスト")],
        };
        let bounded = input.bounded(4);
        assert_eq!(bounded.turns[0].text, "abcd");
        assert_eq!(bounded.history[0].text, "日本語テ");
        assert_eq!(bounded.turns[0].role, "user");
        assert_eq!(input.turns[0].text.chars().count(), 8);
    }

    #[test]
    fn fact_defaults_builders_and_validation() {
        let fact = ExtractedFact::default();
        assert_eq!(fact.content, "");
        assert_eq!(fact.kind, MemoryType::SemanticMemory);
        assert!((fact.importance - 0.5).abs() < f32::EPSILON);

        let fact = ExtractedFact::user_profile("Alice is a developer")
            .with_importance(0.8)
            .with_tag("identity")
            .with_tags(vec!["identity".to_owned(), "explicit".to_owned()])
            .with_source("conversation")
            .with_metadata("key", json!("value"));
        assert_eq!(fact.kind, MemoryType::UserProfile);
        assert!((fact.importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(fact.tags, vec!["identity", "explicit"]);
        assert_eq!(fact.source.as_deref(), Some("conversation"));
        assert_eq!(fact.metadata.get("key"), Some(&json!("value")));
        assert!(fact.validate().is_ok());

        assert!(ExtractedFact::semantic("   ").validate().is_err());
        for kind in [MemoryType::Variable, MemoryType::Unknown] {
            let error = ExtractedFact::new("x", kind).validate().unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert!(error.to_string().contains("kind"));
        }
    }

    #[test]
    fn to_new_memory_clamps_and_rewrites_kinds() {
        let fact = ExtractedFact::semantic("uses postgres")
            .with_importance(4.0)
            .with_source("conversation");
        let memory = fact.to_new_memory();
        assert_eq!(memory.importance, Some(1.0));
        assert_eq!(memory.content, "uses postgres");
        assert_eq!(memory.kind, MemoryType::SemanticMemory);
        assert_eq!(memory.source.as_deref(), Some("conversation"));

        let fact = ExtractedFact::new("x", MemoryType::Unknown).with_importance(f32::NAN);
        let memory = fact.to_new_memory();
        assert_eq!(memory.kind, MemoryType::SemanticMemory);
        assert!((memory.importance.unwrap_or_default() - 0.5).abs() < f32::EPSILON);

        let fact = ExtractedFact::new("var", MemoryType::Variable);
        assert_eq!(fact.to_new_memory().kind, MemoryType::SemanticMemory);
        assert_eq!(
            ExtractedFact::new("fact", MemoryType::Coding)
                .to_new_memory()
                .kind,
            MemoryType::Coding
        );
    }

    #[test]
    fn from_new_memory_keeps_the_fields() {
        let memory = NewMemory::user_profile("Alice lives in Berlin")
            .with_importance(0.7)
            .with_source("chat")
            .with_metadata("lang", json!("en"));
        let fact = ExtractedFact::from(&memory);
        assert_eq!(fact.content, memory.content);
        assert_eq!(fact.kind, MemoryType::UserProfile);
        assert!((fact.importance - 0.7).abs() < f32::EPSILON);
        assert_eq!(fact.source.as_deref(), Some("chat"));
        assert_eq!(fact.metadata.get("lang"), Some(&json!("en")));
        assert_eq!(fact.tags.len(), 0);

        let fact = ExtractedFact::from(&NewMemory::semantic("x"));
        assert!((fact.importance - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn fact_and_result_serde_round_trip() {
        let fact = ExtractedFact::semantic("the service uses postgres")
            .with_importance(0.6)
            .with_tag("project");
        let json = serde_json::to_value(&fact).unwrap();
        for key in [
            "content",
            "kind",
            "importance",
            "tags",
            "source",
            "metadata",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["kind"], json!("semantic_memory"));
        let restored: ExtractedFact = serde_json::from_value(json).unwrap();
        assert_eq!(restored, fact);
        let tolerant: ExtractedFact = serde_json::from_value(json!({
            "content": "x",
            "kind": "nonsense",
            "unknown_field": 1,
        }))
        .unwrap();
        assert_eq!(tolerant.content, "x");
        assert_eq!(tolerant.kind, MemoryType::Unknown);
        assert!((tolerant.importance - 0.5).abs() < f32::EPSILON);

        let result = ExtractionResult::new(vec![fact], "llm");
        assert!(!result.is_degraded());
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["extractor"], json!("llm"));
        assert!(json["degraded"].is_null());
        let restored: ExtractionResult = serde_json::from_value(json).unwrap();
        assert_eq!(restored, result);
        assert_eq!(ExtractionResult::empty("rule").facts.len(), 0);
        let degraded = ExtractionResult {
            degraded: Some("reason".to_owned()),
            ..ExtractionResult::empty("rule")
        };
        assert!(degraded.is_degraded());
        let tolerant: ExtractionResult =
            serde_json::from_value(json!({"facts": [], "unknown": 2})).unwrap();
        assert_eq!(tolerant.extractor, "");
        assert!(!tolerant.is_degraded());
    }
}
