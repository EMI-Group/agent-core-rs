//! One fact per Markdown file: the `---` front-matter block plus the fact body.
//!
//! A [`FactDocument`] is the on-disk form of a single [`CodingFact`]: a `---`-fenced
//! block of flat `key: value` lines followed by the Markdown body. Ports the
//! front-matter model of Python `core/memory/lite/frontmatter.py` (`name`, `description`,
//! `type`, the timestamps) and adds `tags` (a JSON array) and `path` (present only when
//! the fact has a source path), so a fact survives a round trip through a per-fact file
//! as faithfully as through the inline [`MarkdownMemory`](crate::lite::markdown) form.
//!
//! ```text
//! ---
//! name: build
//! description: How to build the workspace
//! type: project
//! tags: ["build"]
//! path: docs/BUILD.md
//! updated_at: 2026-11-13T07:00:00.000000000Z
//! ---
//!
//! cargo build
//! ```
//!
//! # Tolerance
//!
//! Parsing is total for free-form input except for one structural error: a document whose
//! front matter is opened with `---` and never closed is
//! [`Error::Validation`]. Everything else is
//! tolerated — a document without front matter is all body, blank and non-`key: value`
//! lines inside the block are skipped, unknown keys are kept in the block (and rendered
//! back), and the body keeps its blank lines and Markdown.
//!
//! Keys are matched exactly and rendered with the spelling they were stored under, so an
//! edited key (`description` → `Description`) is preserved but no longer interpreted.
use crate::lite::fact::{CodingFact, CodingFactType};
use crate::lite::markdown::FrontMatter;
use crate::record::MemoryScope;
use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::error::{Error, Result};
use std::str::FromStr;
/// Front-matter key holding the fact name.
const KEY_NAME: &str = "name";
/// Front-matter key holding the one-line description.
const KEY_DESCRIPTION: &str = "description";
/// Front-matter key holding the fact-type name.
const KEY_TYPE: &str = "type";
/// Front-matter key holding the tag list as a JSON array.
const KEY_TAGS: &str = "tags";
/// Front-matter key holding the optional source path.
const KEY_PATH: &str = "path";
/// Front-matter key holding the last modification time.
const KEY_UPDATED_AT: &str = "updated_at";
/// The `---` front-matter fence.
const FENCE: &str = "---";
/// One fact as its own Markdown file: the front matter plus the body.
///
/// [`FactDocument::from_fact`] and [`FactDocument::to_fact`] are the lossless pair: every
/// field of a [`CodingFact`] other than the scope (the file holds one scope, which the
/// caller supplies) survives the round trip, and [`FactDocument::render`] is the shape
/// [`FactDocument::parse`] round-trips for a document that came from `from_fact`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FactDocument {
    /// The `---` key/value block; empty when the document has none.
    pub front_matter: FrontMatter,
    /// The Markdown body of the fact (the fact content).
    pub body: String,
}
impl FactDocument {
    /// Builds an empty document: no front matter and no body.
    ///
    /// Its [`FactDocument::render`] is the empty string, which parses back to the same
    /// document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Parses a fact file.
    ///
    /// Text without a leading `---` fence is all body. Leading blank lines before the
    /// fence are ignored, blank and unparsable lines inside the block are skipped, and
    /// the body is trimmed of leading and trailing blank lines. CRLF and lone CR are
    /// normalized to LF.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when
    /// the document opens a front-matter block with `---` and never closes it.
    pub fn parse(text: &str) -> Result<Self> {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let lines: Vec<&str> = normalized.split('\n').collect();
        let mut index = 0;
        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }
        if index >= lines.len() || lines[index].trim_end() != FENCE {
            return Ok(Self {
                front_matter: FrontMatter::new(),
                body: trim_blank_edges(&lines),
            });
        }
        let content_start = index + 1;
        let Some(fence) = (content_start..lines.len()).find(|line| lines[*line].trim() == FENCE)
        else {
            return Err(Error::validation(
                "fact document front matter starts with `---` but is never closed",
            ));
        };
        let mut front_matter = FrontMatter::new();
        for line in &lines[content_start..fence] {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Some((key, value)) = trimmed.split_once(':') else {
                continue;
            };
            if key.trim().is_empty() {
                continue;
            }
            let _previous = front_matter.insert(key.trim(), value.trim());
        }
        Ok(Self {
            front_matter,
            body: trim_blank_edges(&lines[fence + 1..]),
        })
    }
    /// Renders the document: the `---` block, a blank line, then the body.
    ///
    /// Output is LF with exactly one trailing newline; the body is trimmed of leading
    /// and trailing blank lines, embedded newlines in a front-matter value are folded to
    /// spaces (a fact file keeps one `key: value` per line), and an empty front matter
    /// renders no fence block at all, so an empty document renders as the empty string.
    #[must_use]
    pub fn render(&self) -> String {
        let mut rendered = String::new();
        if !self.front_matter.is_empty() {
            rendered.push_str(FENCE);
            rendered.push('\n');
            for (key, value) in self.front_matter.iter() {
                rendered.push_str(key);
                rendered.push_str(": ");
                rendered.push_str(&single_line(value));
                rendered.push('\n');
            }
            rendered.push_str(FENCE);
            rendered.push('\n');
        }
        let body = trim_blank_edges(&[self.body.as_str()]);
        if !body.is_empty() {
            if !rendered.is_empty() {
                rendered.push('\n');
            }
            rendered.push_str(&body);
            rendered.push('\n');
        }
        rendered
    }
    /// Encodes `fact` as a document.
    ///
    /// The canonical keys are written in a fixed order — `name`, `description`, `type`,
    /// `tags` (a JSON array), `path` (only when the fact has one), `updated_at` (RFC
    /// 3339 with nanosecond precision, so the timestamp round-trips exactly) — and the
    /// body is the fact content.
    #[must_use]
    pub fn from_fact(fact: &CodingFact) -> Self {
        let mut front_matter = FrontMatter::new();
        let _previous = front_matter.insert(KEY_NAME, single_line(&fact.name));
        let _previous = front_matter.insert(KEY_DESCRIPTION, single_line(&fact.description));
        let _previous = front_matter.insert(KEY_TYPE, fact.fact_type.as_str());
        let tags = serde_json::to_string(&fact.tags).unwrap_or_else(|_error| "[]".to_owned());
        let _previous = front_matter.insert(KEY_TAGS, tags);
        if let Some(path) = &fact.path {
            let _previous = front_matter.insert(KEY_PATH, single_line(path));
        }
        let _previous = front_matter.insert(
            KEY_UPDATED_AT,
            fact.updated_at.to_rfc3339_opts(SecondsFormat::Nanos, true),
        );
        Self {
            front_matter,
            body: fact.content.clone(),
        }
    }
    /// Decodes the document as a [`CodingFact`] of `scope`.
    ///
    /// Missing `description`, `tags`, `path`, and `updated_at` decode to their defaults
    /// (empty, empty, absent, and the Unix epoch, never "now"); an unknown fact-type name
    /// is rejected rather than guessed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when
    /// `name` is missing or blank, when `type` is missing, and when `type` is not one of
    /// `user`, `feedback`, `project`, or `reference`.
    pub fn to_fact(&self, scope: &MemoryScope) -> Result<CodingFact> {
        let name = self.name().ok_or_else(|| {
            Error::validation("fact document is missing a non-blank `name` front-matter field")
        })?;
        let raw_type = self.front_matter.get(KEY_TYPE).ok_or_else(|| {
            Error::validation(format!(
                "fact document `{name}` is missing the `type` front-matter field"
            ))
        })?;
        let fact_type = CodingFactType::from_str(raw_type)?;
        let mut fact = CodingFact::new(scope, name, self.body.clone()).with_type(fact_type);
        self.description()
            .unwrap_or_default()
            .clone_into(&mut fact.description);
        fact.tags = self.tags();
        fact.path = self.path().map(str::to_owned);
        fact.updated_at = self.updated_at().unwrap_or_default();
        Ok(fact)
    }
    /// The fact name (`name`), when present and non-blank.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.front_matter
            .get(KEY_NAME)
            .map(str::trim)
            .filter(|name| !name.is_empty())
    }
    /// The fact type (`type`), when present and one of the known names.
    #[must_use]
    pub fn fact_type(&self) -> Option<CodingFactType> {
        self.front_matter
            .get(KEY_TYPE)
            .and_then(|raw| CodingFactType::from_str(raw).ok())
    }
    /// The one-line description (`description`), when present.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.front_matter.get(KEY_DESCRIPTION)
    }
    /// The tag list (`tags`).
    ///
    /// A JSON array is the canonical form; a comma-separated value is accepted too, so a
    /// hand-written `tags: build, ci` behaves like `tags: ["build", "ci"]`. Empty and
    /// whitespace-only tags are dropped, and a missing or unusable value yields no tags.
    #[must_use]
    pub fn tags(&self) -> Vec<String> {
        let Some(raw) = self.front_matter.get(KEY_TAGS) else {
            return Vec::new();
        };
        let parsed = serde_json::from_str::<Vec<String>>(raw)
            .unwrap_or_else(|_error| raw.split(',').map(str::to_owned).collect());
        parsed
            .into_iter()
            .map(|tag| tag.trim().to_owned())
            .filter(|tag| !tag.is_empty())
            .collect()
    }
    /// The source path (`path`), when present and non-blank.
    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.front_matter
            .get(KEY_PATH)
            .map(str::trim)
            .filter(|path| !path.is_empty())
    }
    /// The last modification time (`updated_at`), when present and RFC 3339.
    #[must_use]
    pub fn updated_at(&self) -> Option<DateTime<Utc>> {
        self.front_matter
            .get(KEY_UPDATED_AT)
            .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
            .map(|timestamp| timestamp.with_timezone(&Utc))
    }
}
/// Folds embedded newlines into spaces, so a front-matter value stays one line.
fn single_line(value: &str) -> String {
    if !value.contains(['\n', '\r']) {
        return value.trim().to_owned();
    }
    value
        .split(['\n', '\r'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<&str>>()
        .join(" ")
}
/// Trims leading and trailing blank lines and joins the rest with `\n`.
fn trim_blank_edges<S: AsRef<str>>(lines: &[S]) -> String {
    let Some(first) = lines
        .iter()
        .position(|line| !line.as_ref().trim().is_empty())
    else {
        return String::new();
    };
    let last = lines
        .iter()
        .rposition(|line| !line.as_ref().trim().is_empty())
        .unwrap_or(first);
    lines[first..=last]
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<&str>>()
        .join("\n")
}
#[cfg(test)]
mod tests {
    use super::*;
    /// A fact with every optional field set and single-line metadata.
    fn sample_fact() -> CodingFact {
        let mut fact = CodingFact::new(
            &MemoryScope::new("alice", "workspace"),
            "build",
            "cargo build\ncargo test",
        )
        .with_type(CodingFactType::Project)
        .with_description("How to build the workspace")
        .with_tags(vec!["build".to_owned(), "ci".to_owned()])
        .with_path("docs/BUILD.md");
        fact.updated_at = DateTime::from_timestamp(1_700_000_000, 123_456_789).unwrap_or_default();
        fact
    }
    #[test]
    fn a_new_document_renders_empty_and_parses_back() {
        let document = FactDocument::new();
        assert_eq!(document.render(), "");
        assert_eq!(FactDocument::parse("").unwrap_or_default(), document);
    }
    #[test]
    fn a_fact_round_trips_through_the_document() {
        let fact = sample_fact();
        let document = FactDocument::from_fact(&fact);
        let decoded = document
            .to_fact(&fact.scope)
            .expect("the sample fact is decodable");
        assert_eq!(decoded, fact);
        let reparsed = FactDocument::parse(&document.render()).expect("rendered output parses");
        assert_eq!(reparsed, document);
        assert_eq!(reparsed.render(), document.render());
    }
    #[test]
    fn the_canonical_key_order_is_fixed() {
        let rendered = FactDocument::from_fact(&sample_fact()).render();
        let keys: Vec<&str> = rendered
            .lines()
            .filter_map(|line| line.split_once(':').map(|(key, _value)| key))
            .filter(|key| *key != FENCE)
            .collect();
        assert_eq!(
            keys,
            vec!["name", "description", "type", "tags", "path", "updated_at"]
        );
    }
    #[test]
    fn a_fact_without_a_path_or_tags_renders_no_path_key() {
        let fact = CodingFact::new(&MemoryScope::default(), "plain", "body");
        let document = FactDocument::from_fact(&fact);
        assert_eq!(document.path(), None);
        assert_eq!(document.tags(), Vec::<String>::new());
        assert!(!document.render().contains("path:"));
        assert!(document.render().contains("tags: []"));
        assert_eq!(document.to_fact(&MemoryScope::default()).ok(), Some(fact));
    }
    #[test]
    fn rendering_ends_with_exactly_one_newline() {
        let rendered = FactDocument::from_fact(&sample_fact()).render();
        assert!(rendered.ends_with('\n'));
        assert!(!rendered.ends_with("\n\n"));
        let mut empty_body = sample_fact();
        empty_body.content = String::new();
        let rendered = FactDocument::from_fact(&empty_body).render();
        assert!(rendered.ends_with('\n'));
        assert!(!rendered.ends_with("\n\n"));
    }
    #[test]
    fn the_body_keeps_inner_blank_lines_and_markdown() {
        let mut fact = sample_fact();
        fact.content = "# Heading\n\n## Another\n### Entry\n- item".to_owned();
        let rendered = FactDocument::from_fact(&fact).render();
        let decoded = FactDocument::parse(&rendered)
            .expect("a body with headings parses")
            .to_fact(&fact.scope)
            .expect("the fact decodes");
        assert_eq!(decoded.content, fact.content);
    }
    #[test]
    fn a_document_without_front_matter_is_all_body() {
        let document = FactDocument::parse("just a body\nsecond line").expect("tolerant");
        assert!(document.front_matter.is_empty());
        assert_eq!(document.body, "just a body\nsecond line");
        assert_eq!(document.render(), "just a body\nsecond line\n");
    }
    #[test]
    fn an_unterminated_front_matter_is_a_validation_error() {
        let error = FactDocument::parse("---\nname: build\ntype: project\n")
            .expect_err("an unclosed block is invalid");
        assert!(error.to_string().contains("never closed"), "got {error}");
    }
    #[test]
    fn front_matter_tolerates_blank_hostile_and_unknown_lines() {
        let text =
            "---\n\nname: build\nnot a pair\ntype: project\nwiki: https://x/y:z\n---\n\nbody";
        let document = FactDocument::parse(text).expect("hostile input parses");
        assert_eq!(document.name(), Some("build"));
        assert_eq!(document.fact_type(), Some(CodingFactType::Project));
        assert_eq!(
            document.front_matter.get("wiki"),
            Some("https://x/y:z"),
            "an unknown key is kept"
        );
        assert_eq!(document.body, "body");
        assert!(document.render().contains("wiki: https://x/y:z"));
    }
    #[test]
    fn crlf_input_is_normalized() {
        let document =
            FactDocument::parse("---\r\nname: build\r\ntype: project\r\n---\r\n\r\nbody")
                .expect("CRLF parses");
        assert_eq!(document.name(), Some("build"));
        assert_eq!(document.body, "body");
        assert!(!document.render().contains('\r'));
    }
    #[test]
    fn tags_accept_json_and_comma_separated_values() {
        let mut document = FactDocument::new();
        let _previous = document.front_matter.insert(KEY_TAGS, r#"["a", "b "]"#);
        assert_eq!(document.tags(), vec!["a".to_owned(), "b".to_owned()]);
        let _previous = document.front_matter.insert(KEY_TAGS, "a, , b");
        assert_eq!(document.tags(), vec!["a".to_owned(), "b".to_owned()]);
        let _previous = document.front_matter.insert(KEY_TAGS, "");
        assert_eq!(document.tags(), Vec::<String>::new());
        let _previous = document.front_matter.remove(KEY_TAGS);
        assert_eq!(document.tags(), Vec::<String>::new());
    }
    #[test]
    fn a_blank_or_missing_name_is_a_validation_error() {
        let mut document = FactDocument::new();
        let _previous = document.front_matter.insert(KEY_TYPE, "project");
        let error = document
            .to_fact(&MemoryScope::default())
            .expect_err("a missing name is invalid");
        assert!(error.to_string().contains("name"), "got {error}");
        let _previous = document.front_matter.insert(KEY_NAME, "   ");
        let error = document
            .to_fact(&MemoryScope::default())
            .expect_err("a blank name is invalid");
        assert!(error.to_string().contains("name"), "got {error}");
    }
    #[test]
    fn a_missing_or_unknown_type_is_a_validation_error() {
        let mut document = FactDocument::new();
        let _previous = document.front_matter.insert(KEY_NAME, "build");
        let error = document
            .to_fact(&MemoryScope::default())
            .expect_err("a missing type is invalid");
        assert!(error.to_string().contains("type"), "got {error}");
        let _previous = document.front_matter.insert(KEY_TYPE, "bogus");
        let error = document
            .to_fact(&MemoryScope::default())
            .expect_err("an unknown type is invalid");
        assert!(error.to_string().contains("bogus"), "got {error}");
    }
    #[test]
    fn optional_fields_default_when_missing() {
        let text = "---\nname: bare\ntype: feedback\n---\n\nbody text";
        let fact = FactDocument::parse(text)
            .expect("the minimal document parses")
            .to_fact(&MemoryScope::new("bob", "ws"))
            .expect("the minimal fact decodes");
        assert_eq!(fact.name, "bare");
        assert_eq!(fact.fact_type, CodingFactType::Feedback);
        assert_eq!(fact.content, "body text");
        assert_eq!(fact.description, "");
        assert_eq!(fact.tags, Vec::<String>::new());
        assert_eq!(fact.path, None);
        assert_eq!(fact.updated_at, DateTime::<Utc>::default());
        assert_eq!(fact.scope, MemoryScope::new("bob", "ws"));
    }
    #[test]
    fn a_body_line_that_looks_like_a_fence_stays_in_the_body() {
        let text = "---\nname: build\ntype: project\n---\n\nfirst\n---\nlast";
        let fact = FactDocument::parse(text)
            .expect("the document parses")
            .to_fact(&MemoryScope::default())
            .expect("the fact decodes");
        assert_eq!(fact.content, "first\n---\nlast");
    }
    #[test]
    fn an_empty_body_round_trips() {
        let mut fact = sample_fact();
        fact.content = String::new();
        let document = FactDocument::from_fact(&fact);
        let decoded = document
            .to_fact(&fact.scope)
            .expect("an empty body is legal");
        assert_eq!(decoded.content, "");
        assert_eq!(FactDocument::parse(&document.render()).ok(), Some(document));
    }
    #[test]
    fn an_unknown_front_matter_key_survives_a_render() {
        let text = "---\nname: build\ntype: project\nowner: alice\n---\n\nbody";
        let document = FactDocument::parse(text).expect("parses");
        assert_eq!(document.front_matter.get("owner"), Some("alice"));
        assert!(document.render().contains("owner: alice"));
    }
    #[test]
    fn updated_at_uses_nanosecond_precision_and_round_trips() {
        let fact = sample_fact();
        let rendered = FactDocument::from_fact(&fact).render();
        assert!(
            rendered.contains("updated_at: 2023-11-14T22:13:20.123456789Z"),
            "got {rendered}"
        );
        let decoded = FactDocument::parse(&rendered)
            .expect("parses")
            .to_fact(&fact.scope)
            .expect("decodes");
        assert_eq!(decoded.updated_at, fact.updated_at);
    }
    #[test]
    fn a_timestamp_without_an_offset_is_ignored() {
        let text = "---\nname: build\ntype: project\nupdated_at: 2023-11-14\n---\n\nbody";
        let document = FactDocument::parse(text).expect("parses");
        assert_eq!(document.updated_at(), None);
        assert_eq!(
            document
                .to_fact(&MemoryScope::default())
                .map(|fact| fact.updated_at)
                .ok(),
            Some(DateTime::<Utc>::default())
        );
    }
    #[test]
    fn accessors_read_the_interpreted_fields() {
        let document = FactDocument::from_fact(&sample_fact());
        assert_eq!(document.name(), Some("build"));
        assert_eq!(document.description(), Some("How to build the workspace"));
        assert_eq!(document.fact_type(), Some(CodingFactType::Project));
        assert_eq!(document.path(), Some("docs/BUILD.md"));
        assert_eq!(document.tags(), vec!["build".to_owned(), "ci".to_owned()]);
        assert!(document.updated_at().is_some());
    }
}
