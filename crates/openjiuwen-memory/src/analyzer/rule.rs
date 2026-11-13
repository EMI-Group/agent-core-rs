//! Deterministic, model-free fact extraction.
//!
//! [`RuleExtractor`] is the always-available baseline: it segments the conversation into
//! lines and sentences and applies a fixed rule inventory, so memory extraction works
//! offline and every result is reproducible. It is also the fallback of
//! [`crate::analyzer::LlmExtractor`].
//!
//! # Rule inventory
//!
//! Rules are tried in this order; the first match wins and the fact keeps the rule's
//! kind, tag, and importance (all importance tiers come from [`AnalyzerConfig`]):
//!
//! 1. **explicit directive** — the sentence starts with `remember`, `note that`,
//!    `keep in mind`, `don't forget`, `memorize`, or `record` (optionally after
//!    `please`/`also`/`and`). The fact content is the remainder after the marker, the
//!    tag `explicit` is added, and the importance is the larger of
//!    `directive_importance` and the tier the remainder would earn on its own.
//! 2. **identity** — `my name/age/email/job title/... is`, `I am a/an`, `I work at/for`,
//!    `I live in`, `my favourite/preferred ...` → [`MemoryType::UserProfile`], tag
//!    `identity`.
//! 3. **preference** — `I/we prefer|like|love|enjoy|hate|dislike|avoid|want|need` (and
//!    the `don't like` / `would prefer` forms) → [`MemoryType::UserProfile`], tag
//!    `preference`.
//! 4. **project/technology** — `we/I use`, `the project/repo/service uses|runs on|...`,
//!    `I'm building|working on|migrating` → [`MemoryType::SemanticMemory`], tag
//!    `project`.
//! 5. **variable assignment** — `snake_case_key = value`, `key -> value`, `set key to
//!    value` → [`MemoryType::SemanticMemory`] with `variable`/`value` metadata and the
//!    tag `variable`. Python's variable-store write path is deferred, so these become
//!    ordinary searchable facts instead of [`MemoryType::Variable`] records.
//! 6. **generic statement** — at least four words plus a copula (`is`, `are`, `was`,
//!    `were`, `has`, `have`, `contains`, `supports`, `requires`) →
//!    [`MemoryType::SemanticMemory`], tag `statement`.
//!
//! Sentences shorter than three characters are ignored, the content of a fact is
//! truncated to `max_content_chars`, turns shorter than `min_message_len` are skipped,
//! every turn is bounded to `max_input_chars`, and facts below `min_importance` are
//! dropped. Duplicates within one batch (equal [`crate::record::normalize_text`]) are
//! merged into the first occurrence: tags are unioned and the larger importance wins.
//! At most `max_facts` facts are returned.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::fact::{
    truncate_chars, ExtractedFact, ExtractionInput, ExtractionResult, FactExtractor,
};
use crate::record::{normalize_text, CallOptions, MemoryType};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::LazyLock;

/// Matches a Markdown bullet or numbered list prefix at the start of a line.
static LIST_PREFIX: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"^(?:[-*+•]\s+|\d+[.)]\s+)"));
/// Matches a leading explicit-directive marker.
static DIRECTIVE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)^(?:please\s+|also\s+|and\s+|just\s+)*(?:remember|note\s+that|keep\s+in\s+mind|don'?t\s+forget|memorize|memorise|record|make\s+a\s+note\s+that)\b[\s:,，\-—]*",
    )
});
/// Matches the identity attribute forms (`my name is ...`).
static IDENTITY_ATTRIBUTE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)^(?:my|our)\s+(?:name|age|e-?mail|job\s*title|job|role|title|company|employer|team|phone|address|city|country|timezone|time\s*zone|birthday|birthdate|username|handle|website|homepage|location|language|preferred\s+\w+|favourite\s+\w+|favorite\s+\w+)\b",
    )
});
/// Matches `I am a/an ...`, `I'm a ...`, `we are a ...`.
static IDENTITY_SELF: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"(?i)^(?:i|we)(?:\s+am|\s+are|'m|'re)\s+(?:a|an|\d)"));
/// Matches `I work at/for ...`.
static IDENTITY_WORK: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"(?i)^(?:i|we)\s+work\s+(?:at|for|in|with)\b"));
/// Matches `I live in ...`.
static IDENTITY_LIVE: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"(?i)^(?:i|we)\s+live\s+in\b"));
/// Matches the preference verbs.
static PREFERENCE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)^(?:i|we)\s+(?:(?:really|also|always|never|just|do)\s+)*(?:prefer|like|love|enjoy|hate|dislike|avoid|want|need)\b",
    )
});
/// Matches negated preferences (`I don't like ...`).
static PREFERENCE_NEGATED: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(r"(?i)^(?:i|we)\s+(?:don'?t|do\s+not)\s+(?:like|love|enjoy|want|need|use)\b")
});
/// Matches conditional preferences (`I'd prefer ...`).
static PREFERENCE_CONDITIONAL: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"(?i)^(?:i|we)(?:'d|\s+would)\s+(?:prefer|like|love|rather)\b"));
/// Matches `we/I use ...`.
static PROJECT_USE: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"(?i)^(?:we|i|they)\s+use\b"));
/// Matches `<article> project/service uses ...`.
static PROJECT_SUBJECT: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)^(?:the|our|this|my)\s+(?:project|repo|repository|service|app|application|codebase|library|package|system|backend|frontend|pipeline|stack|tool|api)\s+(?:uses|runs\s+on|depends\s+on|is\s+built\s+with|requires|targets)\b",
    )
});
/// Matches `I'm building ...`, `we are migrating ...`.
static PROJECT_WORK: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)^(?:i|we)(?:\s+am|\s+are|'m|'re)\s+(?:building|working\s+on|migrating|refactoring)\b",
    )
});
/// Matches a generic declarative copula.
static COPULA: LazyLock<Option<Regex>> =
    LazyLock::new(|| compile(r"(?i)\b(?:is|are|was|were|has|have|contains|supports|requires)\b"));

/// Compiles one analyzer pattern, disabling the rule instead of panicking on failure.
///
/// All patterns in this module are literal constants exercised by tests, so a compile
/// failure means a programming error; it is logged and the rule simply never fires,
/// which keeps arbitrary user input from ever reaching a panic.
pub(crate) fn compile(pattern: &str) -> Option<Regex> {
    let compiled = Regex::new(pattern);
    if let Err(error) = &compiled {
        tracing::error!(
            target: "openjiuwen_memory::analyzer",
            %pattern,
            %error,
            "invalid analyzer pattern; rule disabled"
        );
    }
    compiled.ok()
}

/// Returns whether `pattern` matches `text` (a disabled pattern never matches).
pub(crate) fn matches(pattern: Option<&Regex>, text: &str) -> bool {
    pattern.is_some_and(|regex| regex.is_match(text))
}

/// Returns the byte offset just past a leading `pattern` match in `text`.
fn prefix_len(pattern: Option<&Regex>, text: &str) -> Option<usize> {
    pattern
        .and_then(|regex| regex.find(text))
        .map(|found| found.end())
}

/// Returns `true` for the sentence terminators the splitter recognizes.
fn is_terminator(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | ';' | '。' | '！' | '？' | '；')
}

/// Returns `true` for characters trimmed from both ends of a sentence.
fn is_sentence_filler(ch: char) -> bool {
    ch.is_whitespace() || is_terminator(ch) || matches!(ch, ',' | '，' | '、' | ':' | '：')
}

/// Strips a leading Markdown bullet or numbered list prefix.
fn strip_list_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();
    match LIST_PREFIX.as_ref().and_then(|regex| regex.find(trimmed)) {
        Some(found) => trimmed.get(found.end()..).unwrap_or_default(),
        None => trimmed,
    }
}

/// Splits a turn into trimmed sentences.
///
/// Lines are split first, list prefixes are removed, and a line is then split after
/// `.`, `!`, `?`, `;`, `。`, `！`, `？`, `；` **only** when the terminator is followed by
/// whitespace or ends the line; `src/main.rs` therefore stays one sentence. Sentences
/// shorter than three characters after trimming are dropped.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    for line in text.split('\n') {
        let line = strip_list_prefix(line);
        if line.trim().is_empty() {
            continue;
        }
        let mut start = 0;
        let mut chars = line.char_indices().peekable();
        while let Some((index, ch)) = chars.next() {
            if !is_terminator(ch) {
                continue;
            }
            let at_boundary = match chars.peek() {
                Some((_, next)) => next.is_whitespace(),
                None => true,
            };
            if !at_boundary {
                continue;
            }
            let end = index + ch.len_utf8();
            push_sentence(&mut sentences, line.get(start..end).unwrap_or_default());
            start = end;
        }
        push_sentence(&mut sentences, line.get(start..).unwrap_or_default());
    }
    sentences
}

/// Trims one candidate sentence and keeps it when it is long enough.
fn push_sentence(sentences: &mut Vec<String>, candidate: &str) {
    let sentence = candidate.trim_matches(is_sentence_filler).trim().to_owned();
    if sentence.chars().count() >= 3 {
        sentences.push(sentence);
    }
}

/// A classified sentence before it becomes an [`ExtractedFact`].
struct Draft {
    /// The memory kind the rule assigns.
    kind: MemoryType,
    /// The importance tier the rule assigns.
    importance: f32,
    /// The rule's tag.
    tag: &'static str,
    /// Extra structured fields the rule contributes.
    metadata: BTreeMap<String, Value>,
}

impl Draft {
    /// Builds a draft with no metadata.
    fn new(kind: MemoryType, importance: f32, tag: &'static str) -> Self {
        Self {
            kind,
            importance,
            tag,
            metadata: BTreeMap::new(),
        }
    }
}

/// Extracts facts with fixed rules and no model.
pub struct RuleExtractor {
    /// The analyzer settings that gate and tune every rule.
    config: AnalyzerConfig,
}

impl RuleExtractor {
    /// Builds a rule extractor with `config`.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Classifies one sentence into a fact.
    fn fact_from_sentence(&self, sentence: &str) -> Option<ExtractedFact> {
        let sentence = sentence.trim_matches(is_sentence_filler).trim();
        if sentence.chars().count() < 3 {
            return None;
        }
        if let Some(end) = prefix_len(DIRECTIVE.as_ref(), sentence) {
            let remainder = sentence.get(end..).unwrap_or_default();
            let remainder = remainder.trim_matches(is_sentence_filler).trim();
            if remainder.chars().count() >= 3 {
                let draft = classify_remainder(remainder, &self.config).unwrap_or_else(|| {
                    Draft::new(
                        MemoryType::SemanticMemory,
                        self.config.statement_importance,
                        "statement",
                    )
                });
                let importance = max_importance(self.config.directive_importance, draft.importance);
                return Some(
                    self.build_fact(remainder, draft.kind, importance, &draft.metadata)
                        .with_tag("explicit")
                        .with_tag(draft.tag),
                );
            }
        }
        let draft = classify(sentence, &self.config)?;
        Some(
            self.build_fact(sentence, draft.kind, draft.importance, &draft.metadata)
                .with_tag(draft.tag),
        )
    }

    /// Builds a fact whose content is `content` truncated to the configured limit.
    fn build_fact(
        &self,
        content: &str,
        kind: MemoryType,
        importance: f32,
        metadata: &BTreeMap<String, Value>,
    ) -> ExtractedFact {
        ExtractedFact {
            content: truncate_chars(content, self.config.max_content_chars),
            kind,
            importance,
            tags: Vec::new(),
            source: None,
            metadata: metadata.clone(),
        }
    }
}

#[async_trait]
impl FactExtractor for RuleExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "rule"
    }

    async fn extract(
        &self,
        input: &ExtractionInput,
        options: &CallOptions,
    ) -> Result<ExtractionResult> {
        options.check()?;
        if !self.config.enabled {
            return Ok(ExtractionResult::empty(self.name()));
        }
        input.validate()?;
        let bounded = input.bounded(self.config.max_input_chars);
        let mut facts: Vec<ExtractedFact> = Vec::new();
        for turn in &bounded.turns {
            if turn.text.trim().chars().count() < self.config.min_message_len {
                continue;
            }
            for sentence in split_sentences(&turn.text) {
                if let Some(fact) = self.fact_from_sentence(&sentence) {
                    push_deduped(&mut facts, fact);
                }
            }
        }
        facts.retain(|fact| {
            let importance = if fact.importance.is_finite() {
                fact.importance
            } else {
                0.0
            };
            importance >= self.config.min_importance
        });
        facts.truncate(self.config.max_facts);
        Ok(ExtractionResult::new(facts, self.name()))
    }
}

/// Returns the larger of two importances, ignoring non-finite values.
fn max_importance(left: f32, right: f32) -> f32 {
    match (left.is_finite(), right.is_finite()) {
        (true, true) => left.max(right),
        (true, false) => left,
        (false, true) => right,
        (false, false) => 0.0,
    }
}

/// Adds `fact` to `facts`, merging it into an equal-normalized existing fact.
fn push_deduped(facts: &mut Vec<ExtractedFact>, fact: ExtractedFact) {
    let normalized = normalize_text(&fact.content);
    if normalized.is_empty() {
        return;
    }
    if let Some(existing) = facts
        .iter_mut()
        .find(|existing| normalize_text(&existing.content) == normalized)
    {
        existing.importance = max_importance(existing.importance, fact.importance);
        for tag in fact.tags {
            if !existing.tags.contains(&tag) {
                existing.tags.push(tag);
            }
        }
        return;
    }
    facts.push(fact);
}

/// Applies the non-directive rules to a directive remainder.
///
/// A marker is often followed by `that`/`to` (`remember that I prefer tea`); the
/// remainder is classified as written first, then with that lead-in removed, so the
/// directive keeps the tier of the fact it introduces.
fn classify_remainder(remainder: &str, config: &AnalyzerConfig) -> Option<Draft> {
    if let Some(draft) = classify(remainder, config) {
        return Some(draft);
    }
    let lowered = remainder.to_ascii_lowercase();
    for lead_in in ["that ", "to "] {
        let rest = lowered
            .strip_prefix(lead_in)
            .and_then(|_| remainder.get(lead_in.len()..));
        if let Some(draft) = rest.and_then(|rest| classify(rest, config)) {
            return Some(draft);
        }
    }
    None
}

/// Applies the non-directive rules to one sentence.
fn classify(sentence: &str, config: &AnalyzerConfig) -> Option<Draft> {
    if is_identity(sentence) {
        return Some(Draft::new(
            MemoryType::UserProfile,
            config.identity_importance,
            "identity",
        ));
    }
    if is_preference(sentence) {
        return Some(Draft::new(
            MemoryType::UserProfile,
            config.preference_importance,
            "preference",
        ));
    }
    if is_project(sentence) {
        return Some(Draft::new(
            MemoryType::SemanticMemory,
            config.project_importance,
            "project",
        ));
    }
    if let Some((key, value)) = parse_assignment(sentence) {
        let mut draft = Draft::new(
            MemoryType::SemanticMemory,
            config.statement_importance,
            "variable",
        );
        draft.metadata.insert("variable".to_owned(), json!(key));
        draft.metadata.insert("value".to_owned(), json!(value));
        return Some(draft);
    }
    if is_generic_statement(sentence) {
        return Some(Draft::new(
            MemoryType::SemanticMemory,
            config.statement_importance,
            "statement",
        ));
    }
    None
}

/// Returns `true` for the identity forms of rule 2.
fn is_identity(sentence: &str) -> bool {
    matches(IDENTITY_ATTRIBUTE.as_ref(), sentence)
        || matches(IDENTITY_SELF.as_ref(), sentence)
        || matches(IDENTITY_WORK.as_ref(), sentence)
        || matches(IDENTITY_LIVE.as_ref(), sentence)
}

/// Returns `true` for the preference forms of rule 3.
fn is_preference(sentence: &str) -> bool {
    matches(PREFERENCE.as_ref(), sentence)
        || matches(PREFERENCE_NEGATED.as_ref(), sentence)
        || matches(PREFERENCE_CONDITIONAL.as_ref(), sentence)
}

/// Returns `true` for the project/technology forms of rule 4.
fn is_project(sentence: &str) -> bool {
    matches(PROJECT_USE.as_ref(), sentence)
        || matches(PROJECT_SUBJECT.as_ref(), sentence)
        || matches(PROJECT_WORK.as_ref(), sentence)
}

/// Returns `true` when the sentence is a long enough declarative statement.
fn is_generic_statement(sentence: &str) -> bool {
    sentence.split_whitespace().count() >= 4 && matches(COPULA.as_ref(), sentence)
}

/// Parses a variable assignment of the forms `key = value`, `key -> value`, and
/// `set key to value`.
///
/// The key must be 2–64 characters, start with an ASCII letter, and otherwise contain
/// only ASCII letters, digits, and underscores; the value must be non-empty. The
/// delimiter search is case-insensitive but byte offsets are preserved (lowercasing
/// only rewrites ASCII letters in place), so the returned key keeps its original case.
fn parse_assignment(sentence: &str) -> Option<(String, String)> {
    let lowered = sentence.to_ascii_lowercase();
    if let Some(rest) = lowered.strip_prefix("set ") {
        if let Some(offset) = rest.find(" to ") {
            let key_end = "set ".len() + offset;
            let key = sentence.get("set ".len()..key_end)?;
            let value = sentence.get(key_end + " to ".len()..)?;
            return assignment_parts(key, value);
        }
    }
    for delimiter in ["->", ":=", "="] {
        if let Some(index) = lowered.find(delimiter) {
            let key = sentence.get(..index)?;
            let value = sentence.get(index + delimiter.len()..)?;
            return assignment_parts(key, value);
        }
    }
    None
}

/// Validates the two sides of an assignment and returns them trimmed.
fn assignment_parts(key: &str, value: &str) -> Option<(String, String)> {
    let key = key.trim();
    let value = value.trim_matches(is_sentence_filler).trim();
    let valid_key = (2..=64).contains(&key.chars().count())
        && key
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic())
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
    if !valid_key || value.is_empty() {
        return None;
    }
    Some((key.to_owned(), value.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    fn extractor(config: AnalyzerConfig) -> RuleExtractor {
        RuleExtractor::new(config)
    }

    async fn facts(text: &str) -> Vec<ExtractedFact> {
        extract(text, AnalyzerConfig::default()).await
    }

    async fn extract(text: &str, config: AnalyzerConfig) -> Vec<ExtractedFact> {
        extractor(config)
            .extract(&ExtractionInput::from_text(text), &CallOptions::new())
            .await
            .unwrap()
            .facts
    }

    fn contents(facts: &[ExtractedFact]) -> Vec<&str> {
        facts.iter().map(|fact| fact.content.as_str()).collect()
    }

    #[test]
    fn sentences_split_on_terminators_but_not_filenames() {
        let sentences =
            split_sentences("src/main.rs is the entry point. It uses tokio!\n- npm run build");
        assert_eq!(
            sentences,
            vec![
                "src/main.rs is the entry point",
                "It uses tokio",
                "npm run build",
            ]
        );
        assert_eq!(
            split_sentences("First question? Second one; third one."),
            vec!["First question", "Second one", "third one"]
        );
        // A terminator inside a word never splits.
        assert_eq!(
            split_sentences("v1.2.3 is released"),
            vec!["v1.2.3 is released"]
        );
        // Short fragments are dropped.
        assert_eq!(split_sentences("hi. ok.").len(), 0);
        assert_eq!(split_sentences("   \n\n ").len(), 0);
        // CJK text is not mangled and stays intact without whitespace after the mark.
        assert_eq!(
            split_sentences("我喜欢深色模式。它也支持浅色模式"),
            vec!["我喜欢深色模式。它也支持浅色模式"]
        );
    }

    #[tokio::test]
    async fn extracts_identity_preference_and_project_facts() {
        let extracted = facts(
            "my name is Alice.\n\
             I prefer dark mode.\n\
             The service uses Postgres.",
        )
        .await;
        assert_eq!(
            contents(&extracted),
            vec![
                "my name is Alice",
                "I prefer dark mode",
                "The service uses Postgres"
            ]
        );
        assert_eq!(extracted[0].kind, MemoryType::UserProfile);
        assert_eq!(extracted[0].tags, vec!["identity"]);
        assert!((extracted[0].importance - 0.85).abs() < f32::EPSILON);
        assert_eq!(extracted[1].kind, MemoryType::UserProfile);
        assert_eq!(extracted[1].tags, vec!["preference"]);
        assert!((extracted[1].importance - 0.75).abs() < f32::EPSILON);
        assert_eq!(extracted[2].kind, MemoryType::SemanticMemory);
        assert_eq!(extracted[2].tags, vec!["project"]);
        assert!((extracted[2].importance - 0.6).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn extracts_explicit_directive_facts_with_the_remainder_as_content() {
        let extracted = facts("Remember that I prefer tea over coffee.").await;
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].content, "that I prefer tea over coffee");
        assert_eq!(extracted[0].kind, MemoryType::UserProfile);
        assert_eq!(extracted[0].tags, vec!["explicit", "preference"]);
        assert!((extracted[0].importance - 0.9).abs() < f32::EPSILON);

        let extracted = facts("please keep in mind the deploy window is Friday").await;
        assert_eq!(extracted[0].content, "the deploy window is Friday");
        assert_eq!(extracted[0].tags, vec!["explicit", "statement"]);
        assert!((extracted[0].importance - 0.9).abs() < f32::EPSILON);

        // A marker with nothing useful behind it is not a fact on its own.
        assert_eq!(facts("Remember.").await.len(), 0);
    }

    #[tokio::test]
    async fn extracts_variable_facts_with_metadata() {
        let extracted = facts("db_host = 10.0.0.4\nset retry_limit to 3").await;
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0].content, "db_host = 10.0.0.4");
        assert_eq!(extracted[0].kind, MemoryType::SemanticMemory);
        assert_eq!(extracted[0].tags, vec!["variable"]);
        assert_eq!(
            extracted[0].metadata.get("variable"),
            Some(&json!("db_host"))
        );
        assert_eq!(extracted[0].metadata.get("value"), Some(&json!("10.0.0.4")));
        assert_eq!(
            extracted[1].metadata.get("variable"),
            Some(&json!("retry_limit"))
        );
        assert_eq!(extracted[1].metadata.get("value"), Some(&json!("3")));

        // The assignment forms that are not variable facts stay statements.
        let extracted = facts("a = b").await;
        assert_eq!(extracted.len(), 0);
        let extracted = facts("the value = 1").await;
        assert_eq!(extracted.len(), 0);
    }

    #[tokio::test]
    async fn extracts_generic_statements() {
        let extracted = facts("The build pipeline has three stages").await;
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].content, "The build pipeline has three stages");
        assert_eq!(extracted[0].tags, vec!["statement"]);
        assert_eq!(extracted[0].kind, MemoryType::SemanticMemory);
        assert!((extracted[0].importance - 0.4).abs() < f32::EPSILON);

        // Too short and without a copula: no fact.
        assert_eq!(facts("the build fails").await.len(), 0);
        assert_eq!(facts("good morning everyone").await.len(), 0);
    }

    #[tokio::test]
    async fn skips_turns_below_min_message_len() {
        let config = AnalyzerConfig {
            min_message_len: 100,
            ..Default::default()
        };
        assert_eq!(extract("I prefer dark mode", config).await.len(), 0);
    }

    #[tokio::test]
    async fn dedupes_a_batch_and_unions_tags() {
        let extracted = facts("I prefer dark mode.\nI prefer dark mode.").await;
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].tags, vec!["preference"]);

        let extracted = facts("I prefer dark mode.\nRemember I prefer dark mode.").await;
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].content, "I prefer dark mode");
        assert_eq!(extracted[0].tags, vec!["preference", "explicit"]);
        assert!((extracted[0].importance - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn caps_the_number_of_facts() {
        let text = (0..20)
            .map(|index| format!("I prefer option number {index}."))
            .collect::<Vec<_>>()
            .join("\n");
        let config = AnalyzerConfig {
            max_facts: 3,
            ..Default::default()
        };
        let extracted = extract(&text, config).await;
        assert_eq!(extracted.len(), 3);
        assert_eq!(extracted[0].content, "I prefer option number 0");
    }

    #[tokio::test]
    async fn drops_facts_below_min_importance() {
        let config = AnalyzerConfig {
            min_importance: 0.7,
            ..Default::default()
        };
        let extracted = extract("The build pipeline has three stages", config).await;
        assert_eq!(extracted.len(), 0);
    }

    #[tokio::test]
    async fn truncates_content_and_bounds_huge_input() {
        let long = format!("my name is {}", "a".repeat(5_000));
        let config = AnalyzerConfig {
            max_content_chars: 32,
            ..Default::default()
        };
        let extracted = extract(&long, config).await;
        assert_eq!(extracted[0].content.chars().count(), 32);

        let huge = "I prefer dark mode.\n".repeat(50_000);
        let config = AnalyzerConfig {
            max_input_chars: 64,
            ..Default::default()
        };
        let extracted = extract(&huge, config).await;
        assert!(extracted.len() <= 8);
        assert!(extracted.iter().all(|fact| fact.content.len() <= 64));
    }

    #[tokio::test]
    async fn disabled_config_yields_an_empty_undegraded_result() {
        let config = AnalyzerConfig {
            enabled: false,
            ..Default::default()
        };
        let result = extractor(config)
            .extract(
                &ExtractionInput::from_text("my name is Alice"),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(result.facts.len(), 0);
        assert!(!result.is_degraded());
        assert_eq!(result.extractor, "rule");
    }

    #[tokio::test]
    async fn cancelled_options_stop_before_any_work() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = CallOptions::new().with_cancellation(cancellation);
        let error = extractor(AnalyzerConfig::default())
            .extract(&ExtractionInput::from_text("my name is Alice"), &options)
            .await
            .unwrap_err();
        assert!(matches!(error, openjiuwen_common::error::Error::Cancelled));
    }

    #[tokio::test]
    async fn blank_roles_are_rejected() {
        let input = ExtractionInput {
            turns: vec![crate::analyzer::fact::ConversationTurn::new("  ", "hi")],
            history: Vec::new(),
        };
        let error = extractor(AnalyzerConfig::default())
            .extract(&input, &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
    }

    #[tokio::test]
    async fn adversarial_input_never_panics() {
        for text in [
            "",
            "   ",
            "...",
            "=",
            "-> value",
            "set to",
            "I prefer",
            "my name is",
            &"a".repeat(10_000),
            "日本語 。 テスト 、 ok",
            "remember remember remember x",
        ] {
            let _ = extract(text, AnalyzerConfig::default()).await;
        }
    }

    #[test]
    fn rule_extractor_exposes_its_config() {
        let config = AnalyzerConfig {
            max_facts: 2,
            ..Default::default()
        };
        let extractor = RuleExtractor::new(config.clone());
        assert_eq!(extractor.config(), &config);
        assert_eq!(FactExtractor::name(&extractor), "rule");
    }
}
