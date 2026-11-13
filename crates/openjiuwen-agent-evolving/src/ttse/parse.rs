//! Parsing of tip candidates out of model or markdown output.
//!
//! Ports the offline grammar of Python `openjiuwen.agent_evolving.ttse.tip_parse`:
//! a JSON array or `{"tips": [...]}` payload, markdown bullets and numbered
//! lines are all accepted, and a malformed payload falls back to line parsing
//! instead of failing.
//!
//! The parser is deliberately forgiving and total: [`TipParser::parse`] never
//! panics and never returns an error, it just yields the tips it could read.
//! Every accepted rule is classified (unless the payload carried an explicit
//! category), keyworded, deduplicated and capped, so a caller can store the
//! result straight into a [`TipBank`](crate::ttse::TipBank).

use serde_json::Value;

use crate::ttse::bank::{jaccard, tokenize};
use crate::ttse::classify::TipClassifier;
use crate::ttse::config::TtseConfig;
use crate::ttse::models::{truncate_chars, TipCategory, TipRecord};

/// The default longest rule kept by the parser, in characters.
pub const DEFAULT_MAX_RULE_CHARS: usize = 240;

/// The default maximum number of tips one parse yields.
const DEFAULT_MAX_TIPS: usize = 8;
/// Token Jaccard similarity at or above which a candidate is a duplicate.
const DEDUPE_THRESHOLD: f64 = 0.88;
/// Maximum number of derived keywords attached to one rule.
const MAX_KEYWORDS: usize = 32;
/// The JSON keys a candidate's rule text may live under, in priority order.
const RULE_KEYS: [&str; 3] = ["rule", "text", "tip"];
/// The JSON keys a candidate's explicit category may live under, in priority order.
const CATEGORY_KEYS: [&str; 2] = ["category", "type"];

/// One parsed candidate before classification and record construction.
struct Candidate {
    /// The raw rule text; trimmed and truncated during finalization.
    rule: String,
    /// The explicit JSON category, when the payload carried a recognised one.
    category: Option<TipCategory>,
    /// The explicit JSON keywords, when the payload carried any.
    keywords: Option<Vec<String>>,
}

/// Turns free-form text into deduplicated, classified tip records.
///
/// The parser reads a fenced JSON block when the text has one, then any JSON
/// array / `{"tips": [...]}` / single-object payload, and finally falls back to
/// one candidate per markdown or plain line. Rules are truncated to
/// [`max_rule_chars`](Self::max_rule_chars), near-identical rules are dropped,
/// and at most [`max_tips`](Self::max_tips) tips are returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TipParser {
    /// Maximum number of tips one parse yields.
    max_tips: usize,
    /// Maximum number of characters kept per rule.
    max_rule_chars: usize,
}

impl Default for TipParser {
    fn default() -> Self {
        Self {
            max_tips: DEFAULT_MAX_TIPS,
            max_rule_chars: DEFAULT_MAX_RULE_CHARS,
        }
    }
}

impl TipParser {
    /// Builds a parser, clamping both limits to at least `1`.
    #[must_use]
    pub fn new(max_tips: usize, max_rule_chars: usize) -> Self {
        Self {
            max_tips: max_tips.max(1),
            max_rule_chars: max_rule_chars.max(1),
        }
    }

    /// Builds a parser from the TTSE tunables.
    ///
    /// Uses [`TtseConfig::max_tips_per_observation`] for the tip cap and
    /// [`DEFAULT_MAX_RULE_CHARS`] for the per-rule bound.
    #[must_use]
    pub fn from_config(config: &TtseConfig) -> Self {
        Self::new(config.max_tips_per_observation, DEFAULT_MAX_RULE_CHARS)
    }

    /// The maximum number of tips one parse yields.
    #[must_use]
    pub fn max_tips(&self) -> usize {
        self.max_tips
    }

    /// The maximum number of characters kept per rule.
    #[must_use]
    pub fn max_rule_chars(&self) -> usize {
        self.max_rule_chars
    }

    /// Extracts the tips from `text`.
    ///
    /// JSON is tried first (a bare array, `{"tips": [...]}`, or a single
    /// object); when that yields no usable item the text is read line by line.
    /// The result is deterministic: identical input yields identical records,
    /// with the first occurrence of a near-duplicate rule kept.
    #[must_use]
    pub fn parse(&self, text: &str) -> Vec<TipRecord> {
        let body = fenced_block(text).unwrap_or_else(|| text.trim().to_owned());
        let mut candidates = match serde_json::from_str::<Value>(body.trim()) {
            Ok(value) => candidates_from_json(&value),
            Err(_) => Vec::new(),
        };
        if candidates.is_empty() {
            candidates = candidates_from_lines(&body);
        }
        self.finish(candidates)
    }

    /// Truncates, classifies, keywords, deduplicates and caps the candidates.
    fn finish(&self, candidates: Vec<Candidate>) -> Vec<TipRecord> {
        let classifier = TipClassifier::default();
        let mut tips: Vec<TipRecord> = Vec::new();
        let mut kept_tokens: Vec<Vec<String>> = Vec::new();
        for candidate in candidates {
            if tips.len() >= self.max_tips {
                break;
            }
            let rule = truncate_chars(candidate.rule.trim(), self.max_rule_chars);
            let rule = rule.trim();
            if rule.is_empty() {
                continue;
            }
            let tokens = tokenize(rule);
            let duplicate = kept_tokens
                .iter()
                .any(|kept| jaccard(kept, &tokens) >= DEDUPE_THRESHOLD);
            if duplicate {
                continue;
            }
            let category = candidate
                .category
                .unwrap_or_else(|| classifier.classify(rule));
            let keywords = candidate.keywords.unwrap_or_else(|| {
                let mut derived = tokens.clone();
                derived.truncate(MAX_KEYWORDS);
                derived
            });
            let Ok(tip) = TipRecord::new(rule, category) else {
                continue;
            };
            let tip = tip.with_keywords(keywords);
            if tips.iter().any(|kept| kept.tip_id == tip.tip_id) {
                continue;
            }
            tips.push(tip);
            kept_tokens.push(tokens);
        }
        tips
    }
}

/// The body of the first fenced block in `text`, when there is one.
///
/// The opening and closing fence lines are dropped. An unclosed fence still
/// yields the lines that followed it, so a truncated model answer is parsed
/// rather than ignored.
fn fenced_block(text: &str) -> Option<String> {
    let mut inside = false;
    let mut body: Vec<&str> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if inside {
                return Some(body.join("\n"));
            }
            inside = true;
            continue;
        }
        if inside {
            body.push(line);
        }
    }
    if inside && !body.is_empty() {
        return Some(body.join("\n"));
    }
    None
}

/// Reads the candidates a JSON payload carries.
fn candidates_from_json(value: &Value) -> Vec<Candidate> {
    match value {
        Value::Array(items) => items.iter().filter_map(candidate_from_json).collect(),
        Value::Object(map) => match map.get("tips") {
            Some(Value::Array(items)) => items.iter().filter_map(candidate_from_json).collect(),
            Some(_) => Vec::new(),
            None => candidate_from_json(value).into_iter().collect(),
        },
        _ => Vec::new(),
    }
}

/// Reads one JSON item: a rule string or an object with rule/category/keywords.
fn candidate_from_json(value: &Value) -> Option<Candidate> {
    match value {
        Value::String(text) => Some(Candidate {
            rule: text.clone(),
            category: None,
            keywords: None,
        }),
        Value::Object(map) => {
            let rule = RULE_KEYS.iter().find_map(|key| {
                map.get(*key)
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                    .map(str::to_owned)
            })?;
            let category = CATEGORY_KEYS.iter().find_map(|key| {
                map.get(*key)
                    .and_then(Value::as_str)
                    .and_then(TipCategory::parse)
            });
            let keywords = map
                .get("keywords")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|keyword| !keyword.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .filter(|keywords| !keywords.is_empty());
            Some(Candidate {
                rule,
                category,
                keywords,
            })
        }
        _ => None,
    }
}

/// Reads one candidate per non-skippable line.
fn candidates_from_lines(text: &str) -> Vec<Candidate> {
    text.lines().filter_map(candidate_from_line).collect()
}

/// Reads one line, returning `None` for blanks, fences, comments and headers.
fn candidate_from_line(line: &str) -> Option<Candidate> {
    let line = line.trim();
    if line.is_empty() || line.starts_with("```") || line.starts_with('#') || is_header_line(line) {
        return None;
    }
    let line = strip_bullet(line);
    let line = strip_tag(line);
    let (line, category) = strip_category_group(line);
    let rule = line.trim();
    if rule.is_empty() {
        return None;
    }
    Some(Candidate {
        rule: rule.to_owned(),
        category,
        keywords: None,
    })
}

/// Whether the line is a short header such as `Tips:` rather than a rule.
fn is_header_line(line: &str) -> bool {
    let Some(stem) = line.strip_suffix(':') else {
        return false;
    };
    !stem.is_empty() && !stem.contains(':') && stem.split_whitespace().count() <= 3
}

/// Strips a leading markdown bullet or ordered prefix followed by whitespace.
fn strip_bullet(line: &str) -> &str {
    let mut chars = line.chars();
    let Some(first) = chars.next() else {
        return line;
    };
    if matches!(first, '-' | '*' | '+' | '•') {
        let rest = chars.as_str();
        if rest.chars().next().is_some_and(char::is_whitespace) {
            return rest.trim_start();
        }
        return line;
    }
    let digits_end = line
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(line.len());
    if digits_end == 0 || digits_end >= line.len() {
        return line;
    }
    if !matches!(line.as_bytes()[digits_end], b'.' | b')') {
        return line;
    }
    let rest = &line[digits_end + 1..];
    if rest.chars().next().is_some_and(char::is_whitespace) {
        return rest.trim_start();
    }
    line
}

/// Strips a leading `TIP:`/`FACT:` tag, case-insensitively.
fn strip_tag(line: &str) -> &str {
    let lowered = line.to_ascii_lowercase();
    for tag in ["tip:", "fact:"] {
        if lowered.starts_with(tag) {
            return line[tag.len()..].trim_start();
        }
    }
    line
}

/// Strips a leading `[category]` or `(category)` group when it is recognised.
///
/// Unrecognised groups are kept verbatim, so `[1]` or `(see below)` stay part of
/// the rule instead of being silently dropped.
fn strip_category_group(line: &str) -> (&str, Option<TipCategory>) {
    let Some(open) = line.chars().next() else {
        return (line, None);
    };
    let close = match open {
        '[' => ']',
        '(' => ')',
        _ => return (line, None),
    };
    let Some(end) = line.find(close) else {
        return (line, None);
    };
    let inner = &line[open.len_utf8()..end];
    match TipCategory::parse(inner) {
        Some(category) => (line[end + close.len_utf8()..].trim_start(), Some(category)),
        None => (line, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(tips: &[TipRecord]) -> Vec<&str> {
        tips.iter().map(|tip| tip.rule.as_str()).collect()
    }

    #[test]
    fn json_array_mixes_strings_and_objects() {
        let text = r#"[
            "use rg to search the workspace",
            {"rule": "Confirm before deleting files", "category": "pitfall",
             "keywords": ["Delete", " confirm ", ""]},
            {"text": "exported by text"},
            {"tip": "exported by tip"},
            {"rule": "   "},
            {"category": "tool"},
            42
        ]"#;
        let tips = TipParser::default().parse(text);
        assert_eq!(
            rules(&tips),
            vec![
                "use rg to search the workspace",
                "Confirm before deleting files",
                "exported by text",
                "exported by tip",
            ]
        );
        assert_eq!(tips[0].category, TipCategory::Environment);
        assert_eq!(tips[1].category, TipCategory::Pitfall);
        assert_eq!(tips[1].keywords, vec!["confirm", "delete"]);
        assert!(tips.iter().all(|tip| tip.source.is_empty()));
        assert!(tips.iter().all(|tip| tip.created_tick == 0));
    }

    #[test]
    fn json_tips_object_and_single_object() {
        let parser = TipParser::default();
        let wrapped = parser.parse(r#"{"tips": ["run cargo fmt before committing"]}"#);
        assert_eq!(rules(&wrapped), vec!["run cargo fmt before committing"]);

        let single = parser.parse(r#"{"rule": "ask the user to confirm", "type": "comm"}"#);
        assert_eq!(rules(&single), vec!["ask the user to confirm"]);
        assert_eq!(single[0].category, TipCategory::Communication);

        assert_eq!(parser.parse("").len(), 0);
        assert_eq!(parser.parse("   \n\n  ").len(), 0);
    }

    #[test]
    fn fenced_json_block_is_preferred_over_prose() {
        let text = "Here is what we learned:\n\n```json\n[\n  {\"rule\": \"first run the tests\", \"category\": \"workflow\"}\n]\n```\n\n- this bullet is outside the fence\n";
        let tips = TipParser::default().parse(text);
        assert_eq!(rules(&tips), vec!["first run the tests"]);
        assert_eq!(tips[0].category, TipCategory::Workflow);

        let unlabelled = TipParser::default().parse("```\n- run cargo test before committing\n```");
        assert_eq!(rules(&unlabelled), vec!["run cargo test before committing"]);
    }

    #[test]
    fn markdown_bullets_with_category_groups() {
        let text = "Tips:\n\n- [tool] always run the cargo command before reporting\n* (pitfall) never delete the lock file\n+ [nope] keep this group verbatim\n• explain the change to the user\n";
        let tips = TipParser::default().parse(text);
        assert_eq!(
            rules(&tips),
            vec![
                "always run the cargo command before reporting",
                "never delete the lock file",
                "[nope] keep this group verbatim",
                "explain the change to the user",
            ]
        );
        assert_eq!(tips[0].category, TipCategory::Tool);
        assert_eq!(tips[1].category, TipCategory::Pitfall);
        // `[nope]` is not a category, so the text is kept and classified.
        assert_eq!(tips[2].category, TipCategory::Other);
        assert_eq!(tips[3].category, TipCategory::Communication);
    }

    #[test]
    fn numbered_and_tagged_lines() {
        let text = "3. first run cargo fmt\n12) then run the tests in order\n1.no space so it is not a marker\nTIP: avoid the timeout error\nfact: the workspace path comes from config\n# a comment\n";
        let tips = TipParser::default().parse(text);
        assert_eq!(
            rules(&tips),
            vec![
                "first run cargo fmt",
                "then run the tests in order",
                "1.no space so it is not a marker",
                "avoid the timeout error",
                "the workspace path comes from config",
            ]
        );
        assert_eq!(tips[0].category, TipCategory::Workflow);
        assert_eq!(tips[3].category, TipCategory::Pitfall);
        assert_eq!(tips[4].category, TipCategory::Environment);
    }

    #[test]
    fn plain_lines_are_classified_and_keyworded() {
        let tips = TipParser::default()
            .parse("confirm the plan with the user\nuse the shell script helper\n");
        assert_eq!(
            rules(&tips),
            vec![
                "confirm the plan with the user",
                "use the shell script helper"
            ]
        );
        assert_eq!(tips[0].category, TipCategory::Communication);
        assert_eq!(tips[1].category, TipCategory::Tool);
        assert_eq!(tips[0].keywords, tokenize("confirm the plan with the user"));
        assert!(tips[0].keywords.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn malformed_json_falls_back_to_lines() {
        let text = "{\"rule\": \"use rg to search\"\n- run cargo fmt before committing\n- run cargo test before committing\n";
        let tips = TipParser::default().parse(text);
        let extracted = rules(&tips);
        assert!(extracted.contains(&"run cargo fmt before committing"));
        assert!(extracted.contains(&"run cargo test before committing"));
        assert!(tips.iter().all(|tip| !tip.rule.trim().is_empty()));
    }

    #[test]
    fn dedupe_cap_truncation_and_config_limits() {
        let base = "always run cargo fmt and cargo test before git commit";
        let similar = format!("{base} first");
        let text = format!(
            "- {base}\n- {similar}\n- check the workspace path\n- ask the user to confirm\n"
        );
        let parser = TipParser::new(2, DEFAULT_MAX_RULE_CHARS);
        let tips = parser.parse(&text);
        assert_eq!(rules(&tips), vec![base, "check the workspace path"]);

        let truncated =
            TipParser::new(4, 10).parse("this is a rather long tip that must be shortened");
        assert_eq!(truncated.len(), 1);
        assert_eq!(truncated[0].rule.chars().count(), 10);
        assert!(truncated[0].rule.ends_with('…'));

        let cjk = TipParser::new(4, 6).parse("日本語のテキストを長くする");
        assert_eq!(cjk[0].rule.chars().count(), 6);
        assert!(cjk[0].rule.ends_with('…'));

        let with_tips = TtseConfig {
            max_tips_per_observation: 3,
            ..TtseConfig::default()
        };
        let configured = TipParser::from_config(&with_tips);
        assert_eq!(configured.max_tips(), 3);
        assert_eq!(configured.max_rule_chars(), DEFAULT_MAX_RULE_CHARS);
        assert_eq!(TipParser::default().max_tips(), DEFAULT_MAX_TIPS);
        assert_eq!(TipParser::new(0, 0).max_tips(), 1);
        assert_eq!(TipParser::new(0, 0).max_rule_chars(), 1);
    }
}
