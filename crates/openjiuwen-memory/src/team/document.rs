//! The `TEAM_MEMORY.md` document format: render team records to Markdown and parse them back.
//!
//! Python's reference file (`agent_teams/memory/shared_memory.py`) is a flat list of
//! `### [decision] Title` entries with a body and nothing else, and the leader prompt only
//! injects its first [`TEAM_MEMORY_MAX_READ_LINES`] lines. This module keeps that entry
//! syntax and extends it losslessly: front matter carries the team id, and every entry
//! carries a machine-readable payload comment so ids, timestamps, kinds, tags, metadata,
//! visibility, importance, provenance, and expiry survive a round trip.
//!
//! # Document shape
//!
//! ```text
//! ---
//! team_memory: openjiuwen-team-memory/v1
//! team_id: review-team
//! generated_at: 2026-01-02T03:04:05Z
//! entries: 1
//! ---
//!
//! # Team memory — review-team
//!
//! ## decision
//!
//! ### [decision] Storage choice
//! <!-- team-memory: {"kind":"decision","title":"Storage choice","importance":0.8,...} -->
//! *id: 018f1d2e3f4a5b6c7d8e9001 · member: alice · importance: 0.80 · updated: 2026-01-02*
//!
//! Use SQLite for the shared store.
//! ```
//!
//! The front matter is simple `key: value` lines (not YAML): `team_memory` names the
//! format, `team_id` is the owning team, `generated_at` is the render time, and `entries`
//! is the number of entries. Unknown keys are ignored and a missing front matter is
//! tolerated on parse (the entry payloads are still honoured).
//!
//! Sections are the four Python extractor tags — `decision`, `member`, `lesson` and
//! `context` — in that fixed order, followed by `custom` for caller-defined kinds; an
//! empty section is omitted. Each entry renders a `### [tag] <title>` heading (the title,
//! or the first non-blank content line), the payload comment, a short human metadata line,
//! a blank line, and the content verbatim with trailing whitespace trimmed.
//!
//! # Parsing
//!
//! - When a payload comment is present it is the source of truth: the record is rebuilt
//!   from it and the heading/body are only the human view. Body lines that start with `#`
//!   or `\` are rendered with one extra leading backslash and unescaped on parse, so a
//!   content line can never be mistaken for a section, an entry heading, or the H1 — which
//!   is what makes multi-line and Markdown-shaped content round-trip unchanged.
//! - A document without any payload — a Python-authored `TEAM_MEMORY.md` — is parsed
//!   structurally: every `### ` line starts an entry, the kind comes from the leading
//!   `[tag]` (or the enclosing `##` section, else `shared_fact`), the title from the
//!   heading text, and the content from the entry body. Such a document is read verbatim
//!   (no unescaping), so a hand-written body line that starts with `## ` is treated as a
//!   section heading. `parse` then needs a team id: the front-matter `team_id`, or the
//!   caller's fallback.
//! - A payload-less entry gets a deterministic id derived from the team id, kind, title,
//!   and content (`stable_id`), importance `0.5`, source `TEAM_MEMORY.md`,
//!   `team` visibility, and `created_at = updated_at = generated_at` (the front-matter
//!   time, or the Unix epoch when there is no front matter). Re-importing the same text is
//!   therefore idempotent: the same ids are upserted, and the store keeps their
//!   `created_at`.
//! - Unknown sections and `###` headings without a payload and without a body are skipped
//!   with a `tracing` warning; a bad `team_id` line or an unparseable payload is
//!   [`Error::Validation`] naming the 1-based line.
//!
//! Because an HTML comment cannot contain `--`, a metadata/title value containing the
//! literal sequence `-->` cannot be represented; such a record renders a payload the
//! parser rejects rather than silently dropping fields.
use crate::record::CallOptions;
use crate::team::model::{
    TeamMemoryFilter, TeamMemoryKind, TeamMemoryRecord, TeamMemoryScope, TeamVisibility,
    DEFAULT_IMPORTANCE,
};
use crate::team::store::TeamMemory;
use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};

/// The file name Python's shared-memory manager reads from `{team_home}/team-memory/`.
pub const TEAM_MEMORY_FILE: &str = "TEAM_MEMORY.md";

/// Number of leading document lines injected into a prompt (Python's read cap).
pub const TEAM_MEMORY_MAX_READ_LINES: usize = 200;

/// The front-matter `team_memory` value this module writes and understands.
pub const TEAM_MEMORY_FORMAT: &str = "openjiuwen-team-memory/v1";

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// Line that opens and closes the front matter.
const FRONT_MATTER_DELIMITER: &str = "---";
/// Prefix of an entry's machine-readable payload comment.
const PAYLOAD_PREFIX: &str = "<!-- team-memory:";
/// Suffix of an entry's machine-readable payload comment.
const PAYLOAD_SUFFIX: &str = "-->";
/// Prefix of the human metadata line directly under a payload.
const METADATA_PREFIX: &str = "*id:";
/// The section every caller-defined kind renders under.
const CUSTOM_SECTION: &str = "custom";
/// Section order: the four Python extractor tags, then caller-defined kinds.
const SECTION_ORDER: [&str; 5] = ["decision", "member", "lesson", "context", CUSTOM_SECTION];
/// Prefix of a payload-less entry's deterministic id.
const DERIVED_ID_PREFIX: &str = "md-";
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Byte hashed between the parts of a derived id, so part boundaries matter.
const PART_SEPARATOR: u8 = 0x1f;

/// A parsed or renderable `TEAM_MEMORY.md` document.
///
/// `entries` are always kept in the documented render order (section order, then
/// `updated_at` descending, then id ascending), so [`TeamMemoryDocument::render`] is
/// deterministic for a given set of records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemoryDocument {
    /// The team that owns every entry.
    pub team_id: String,
    /// When the document was rendered; also the timestamps of payload-less entries.
    pub generated_at: DateTime<Utc>,
    /// The entries, in render order.
    pub entries: Vec<TeamMemoryRecord>,
}

impl TeamMemoryDocument {
    /// Builds a document and sorts `entries` into the documented render order.
    ///
    /// The sort is stable and total: section order (`decision`, `member`, `lesson`,
    /// `context`, `custom`), then `updated_at` descending, then id ascending, then
    /// content ascending.
    #[must_use]
    pub fn from_records(
        team_id: impl Into<String>,
        mut entries: Vec<TeamMemoryRecord>,
        generated_at: DateTime<Utc>,
    ) -> Self {
        sort_records(&mut entries);
        Self {
            team_id: team_id.into(),
            generated_at,
            entries,
        }
    }

    /// Renders the document to Markdown, terminated by exactly one newline.
    ///
    /// An empty document renders the front matter and the H1 only.
    #[must_use]
    pub fn render(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push(FRONT_MATTER_DELIMITER.to_owned());
        lines.push(format!("team_memory: {TEAM_MEMORY_FORMAT}"));
        lines.push(format!("team_id: {}", self.team_id));
        lines.push(format!(
            "generated_at: {}",
            format_timestamp(self.generated_at)
        ));
        lines.push(format!("entries: {}", self.entries.len()));
        lines.push(FRONT_MATTER_DELIMITER.to_owned());
        lines.push(String::new());
        lines.push(format!("# Team memory \u{2014} {}", self.team_id));
        lines.push(String::new());
        let mut sorted = self.entries.clone();
        sort_records(&mut sorted);
        for section in SECTION_ORDER {
            let group: Vec<&TeamMemoryRecord> = sorted
                .iter()
                .filter(|record| section_tag(&record.kind) == section)
                .collect();
            if group.is_empty() {
                continue;
            }
            lines.push(format!("## {section}"));
            lines.push(String::new());
            for record in group {
                render_entry(&mut lines, record);
            }
        }
        while lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        let mut rendered = lines.join("\n");
        rendered.push('\n');
        rendered
    }

    /// Parses a document, taking the team id from the front matter or `fallback_team_id`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when neither source yields a valid team id, when a
    /// front-matter `team_id` line is invalid, when a payload comment is not valid JSON,
    /// or when a restored record fails validation; the message names the 1-based line.
    pub fn parse(content: &str, fallback_team_id: Option<&str>) -> Result<Self> {
        let normalized = normalize_newlines(content);
        let lines: Vec<&str> = normalized.split('\n').collect();
        let (front_matter, body_start) = match parse_front_matter(&lines)? {
            Some(parsed) => (Some(parsed.front), parsed.body_start),
            None => (None, 0),
        };
        let team_id = resolve_team_id(front_matter.as_ref(), fallback_team_id)?;
        let generated_at = front_matter
            .as_ref()
            .and_then(|front| front.generated_at)
            .unwrap_or(DateTime::UNIX_EPOCH);
        let mut entries: Vec<TeamMemoryRecord> = Vec::new();
        let mut current: Option<EntryBuilder> = None;
        let mut section: Option<String> = None;
        let mut index = body_start;
        while index < lines.len() {
            let raw = lines[index];
            let line = raw.trim_end();
            let number = index + 1;
            if let Some(heading) = line.strip_prefix("### ") {
                flush_entry(
                    &mut entries,
                    current.take(),
                    &team_id,
                    generated_at,
                    section.as_deref(),
                )?;
                current = Some(EntryBuilder::new(heading, number));
            } else if let Some(heading) = line.strip_prefix("## ") {
                flush_entry(
                    &mut entries,
                    current.take(),
                    &team_id,
                    generated_at,
                    section.as_deref(),
                )?;
                section = Some(normalize_tag(heading));
            } else if line.starts_with("# ") || line == "#" {
                flush_entry(
                    &mut entries,
                    current.take(),
                    &team_id,
                    generated_at,
                    section.as_deref(),
                )?;
            } else if line.starts_with(PAYLOAD_PREFIX) {
                if let Some(open) = current.as_mut().filter(|open| open.payload.is_none()) {
                    open.payload = Some(parse_payload(line, number)?);
                } else if let Some(open) = current.as_mut() {
                    open.body.push(unescape_body_line(raw));
                } else {
                    warn!(
                        target: LOG_TARGET,
                        line = number,
                        "skipping team memory markdown payload without an entry heading"
                    );
                }
            } else if let Some(open) = current.as_mut() {
                if open.payload.is_some() {
                    if open.body.is_empty() && !open.metadata_seen && is_metadata_line(line) {
                        open.metadata_seen = true;
                        index += 1;
                        continue;
                    }
                    if open.body.is_empty() && !open.separator_consumed && line.trim().is_empty() {
                        open.separator_consumed = true;
                        index += 1;
                        continue;
                    }
                    open.separator_consumed = true;
                    open.body.push(unescape_body_line(raw));
                } else {
                    open.body.push(raw.to_owned());
                }
            }
            index += 1;
        }
        flush_entry(
            &mut entries,
            current.take(),
            &team_id,
            generated_at,
            section.as_deref(),
        )?;
        Ok(Self {
            team_id,
            generated_at,
            entries,
        })
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the document has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl TeamMemory {
    /// Renders the records matching `filter` as a `TEAM_MEMORY.md` document.
    ///
    /// The team id comes from `filter.team_id` (the only id a filter requires) and
    /// [`Utc::now`] is the document's `generated_at`. The document is not truncated: use
    /// [`truncate_for_prompt`] for the Python-sized prompt view.
    pub async fn export_markdown(&self, filter: &TeamMemoryFilter) -> Result<String> {
        let records = self.list(filter).await?;
        debug!(
            target: LOG_TARGET,
            team_id = %filter.team_id,
            entries = records.len(),
            "exporting team memory markdown"
        );
        Ok(TeamMemoryDocument::from_records(filter.team_id.clone(), records, Utc::now()).render())
    }

    /// Parses `markdown` and upserts every entry into this facade's store, in file order.
    ///
    /// Returns the stored records. The document must carry its own front-matter
    /// `team_id`; use [`TeamMemoryDocument::parse`] with a fallback to import a
    /// Python-authored file that has none. Re-importing the same document is idempotent:
    /// identical ids are upserted, the count stays stable, and `created_at` is preserved by
    /// the store. A triggered cancellation token writes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] when the token is already triggered, [`Error::Validation`] for
    /// a malformed document (naming the 1-based line), and the store's error otherwise.
    pub async fn import_markdown(
        &self,
        markdown: &str,
        call: &CallOptions,
    ) -> Result<Vec<TeamMemoryRecord>> {
        call.check()?;
        let document = TeamMemoryDocument::parse(markdown, None)?;
        call.check()?;
        let mut stored = Vec::with_capacity(document.entries.len());
        for entry in &document.entries {
            let saved = call.run(self.store().upsert(entry)).await??;
            call.check()?;
            stored.push(saved);
        }
        debug!(
            target: LOG_TARGET,
            team_id = %document.team_id,
            imported = stored.len(),
            "imported team memory markdown"
        );
        Ok(stored)
    }
}

/// Reads the `team_id` out of a document's front matter, without parsing the entries.
#[must_use]
pub fn document_team_id(content: &str) -> Option<String> {
    let normalized = normalize_newlines(content);
    let lines: Vec<&str> = normalized.split('\n').collect();
    parse_front_matter(&lines)
        .ok()
        .flatten()
        .and_then(|parsed| parsed.front.team_id)
}

/// Keeps the first [`TEAM_MEMORY_MAX_READ_LINES`] lines of a document, trimmed.
///
/// This mirrors Python's read cap for prompt injection: the file on disk keeps every
/// entry, while only the head of the document reaches the model.
#[must_use]
pub fn truncate_for_prompt(content: &str) -> String {
    content
        .lines()
        .take(TEAM_MEMORY_MAX_READ_LINES)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

/// The front-matter fields this module understands.
struct FrontMatter {
    /// The validated owning team id, when the file declares one.
    team_id: Option<String>,
    /// The render time, when the file declares a parseable one.
    generated_at: Option<DateTime<Utc>>,
}

/// A parsed front matter plus the index of the first line after it.
struct ParsedFrontMatter {
    /// The parsed keys.
    front: FrontMatter,
    /// Index of the first body line.
    body_start: usize,
}

/// Parses the optional front matter starting at the first non-blank line.
///
/// Returns `None` when the document has no (terminated) front matter, in which case the
/// whole content is body. Every line number in an error is 1-based.
fn parse_front_matter(lines: &[&str]) -> Result<Option<ParsedFrontMatter>> {
    let Some(start) = lines.iter().position(|line| !line.trim().is_empty()) else {
        return Ok(None);
    };
    if lines[start].trim_end() != FRONT_MATTER_DELIMITER {
        return Ok(None);
    }
    let Some(end) =
        (start + 1..lines.len()).find(|index| lines[*index].trim_end() == FRONT_MATTER_DELIMITER)
    else {
        warn!(
            target: LOG_TARGET,
            "ignoring team memory markdown front matter that is not closed by `---`"
        );
        return Ok(None);
    };
    let mut front = FrontMatter {
        team_id: None,
        generated_at: None,
    };
    for (offset, raw) in lines[start + 1..end].iter().enumerate() {
        let number = start + offset + 2;
        let line = raw.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            warn!(
                target: LOG_TARGET,
                line = number,
                "ignoring team memory markdown front matter line without `key: value`"
            );
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        match key.as_str() {
            "team_id" => {
                validate_team_id(value).map_err(|error| with_line_number(number, error))?;
                front.team_id = Some(value.to_owned());
            }
            "generated_at" => match DateTime::parse_from_rfc3339(value) {
                Ok(parsed) => front.generated_at = Some(parsed.with_timezone(&Utc)),
                Err(error) => warn!(
                    target: LOG_TARGET,
                    line = number,
                    %error,
                    "ignoring unparseable team memory markdown `generated_at`"
                ),
            },
            "team_memory" if value != TEAM_MEMORY_FORMAT => {
                warn!(
                    target: LOG_TARGET,
                    line = number,
                    format = value,
                    "parsing an unrecognized team memory markdown format"
                );
            }
            _ => {}
        }
    }
    Ok(Some(ParsedFrontMatter {
        front,
        body_start: end + 1,
    }))
}

/// Resolves the owning team id from the front matter or the caller's fallback.
fn resolve_team_id(front: Option<&FrontMatter>, fallback: Option<&str>) -> Result<String> {
    if let Some(team_id) = front.and_then(|front| front.team_id.as_deref()) {
        return Ok(team_id.to_owned());
    }
    let Some(fallback) = fallback else {
        return Err(Error::validation(
            "team memory markdown has no front-matter `team_id`; a fallback team id is required",
        ));
    };
    validate_team_id(fallback)?;
    Ok(fallback.to_owned())
}

/// Validates a team id with the shared scope rules.
fn validate_team_id(team_id: &str) -> Result<()> {
    TeamMemoryScope::new(team_id).validate()
}

/// One entry while it is still being parsed.
struct EntryBuilder {
    /// Heading text after `### `.
    heading: String,
    /// 1-based line of the heading, named by validation errors.
    line: usize,
    /// The machine-readable payload, when the entry has one.
    payload: Option<Payload>,
    /// The human body lines, verbatim.
    body: Vec<String>,
    /// Whether the human metadata line was already skipped.
    metadata_seen: bool,
    /// Whether the blank separator line after the metadata was already skipped.
    separator_consumed: bool,
}

impl EntryBuilder {
    /// Opens an entry at `line`.
    fn new(heading: &str, line: usize) -> Self {
        Self {
            heading: heading.trim().to_owned(),
            line,
            payload: None,
            body: Vec::new(),
            metadata_seen: false,
            separator_consumed: false,
        }
    }

    /// Turns the builder into a record, or `None` when the entry has no content.
    fn into_record(
        self,
        team_id: &str,
        generated_at: DateTime<Utc>,
        section: Option<&str>,
    ) -> Result<Option<TeamMemoryRecord>> {
        if let Some(payload) = self.payload {
            let content = self.body.join("\n").trim_end().to_owned();
            if content.trim().is_empty() {
                warn!(
                    target: LOG_TARGET,
                    line = self.line,
                    "skipping team memory markdown entry without content"
                );
                return Ok(None);
            }
            let record = payload.into_record(team_id, content, generated_at);
            record
                .validate()
                .map_err(|error| with_line_number(self.line, error))?;
            return Ok(Some(record));
        }
        let (tag, heading_text) = heading_parts(&self.heading);
        let kind = tag.map_or_else(
            || section.map_or(TeamMemoryKind::SharedFact, kind_from_tag),
            kind_from_tag,
        );
        let content = trim_body(&self.body);
        if content.is_empty() {
            warn!(
                target: LOG_TARGET,
                line = self.line,
                "skipping team memory markdown entry without a body"
            );
            return Ok(None);
        }
        let title = (!heading_text.is_empty()).then(|| heading_text.to_owned());
        let id = stable_id(team_id, &kind, title.as_deref(), &content);
        let base = TeamMemoryRecord::new(&TeamMemoryScope::new(team_id), content, kind);
        let record = TeamMemoryRecord {
            id,
            title,
            created_at: generated_at,
            updated_at: generated_at,
            source: Some(TEAM_MEMORY_FILE.to_owned()),
            ..base
        };
        record
            .validate()
            .map_err(|error| with_line_number(self.line, error))?;
        Ok(Some(record))
    }
}

/// Flushes the open entry into `entries`.
fn flush_entry(
    entries: &mut Vec<TeamMemoryRecord>,
    builder: Option<EntryBuilder>,
    team_id: &str,
    generated_at: DateTime<Utc>,
    section: Option<&str>,
) -> Result<()> {
    let Some(builder) = builder else {
        return Ok(());
    };
    if let Some(record) = builder.into_record(team_id, generated_at, section)? {
        entries.push(record);
    }
    Ok(())
}

/// The machine-readable record carried by an entry's payload comment.
///
/// Every field is optional on parse, so a hand-written payload may carry a subset; a field
/// that is absent falls back to the documented default (the document's `generated_at`, the
/// default importance, an empty metadata object, the default visibility).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct Payload {
    /// The record id; a derived id fills an absent one.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    /// The authoring member id.
    #[serde(skip_serializing_if = "Option::is_none")]
    member_id: Option<String>,
    /// The coordination session id.
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    /// The record kind.
    kind: TeamMemoryKind,
    /// The record title, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    /// Relative importance.
    #[serde(skip_serializing_if = "Option::is_none")]
    importance: Option<f32>,
    /// Normalized tags.
    tags: Vec<String>,
    /// Structured metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
    /// Provenance marker.
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    /// Who may read the record.
    visibility: TeamVisibility,
    /// Creation time.
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<DateTime<Utc>>,
    /// Last modification time.
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_at: Option<DateTime<Utc>>,
    /// Optional expiry.
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<DateTime<Utc>>,
}

impl Payload {
    /// Extracts the payload of `record`; everything but `content`/`team_id` is carried.
    fn from_record(record: &TeamMemoryRecord) -> Self {
        Self {
            id: Some(record.id.clone()),
            member_id: record.member_id.clone(),
            session_id: record.session_id.clone(),
            kind: record.kind.clone(),
            title: record.title.clone(),
            importance: Some(record.importance),
            tags: record.tags.clone(),
            metadata: Some(record.metadata.clone()),
            source: record.source.clone(),
            visibility: record.visibility.clone(),
            created_at: Some(record.created_at),
            updated_at: Some(record.updated_at),
            expires_at: record.expires_at,
        }
    }

    /// Rebuilds a record from the payload and the human body.
    fn into_record(
        self,
        team_id: &str,
        content: String,
        generated_at: DateTime<Utc>,
    ) -> TeamMemoryRecord {
        let created_at = self.created_at.unwrap_or(generated_at);
        let id = self
            .id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| stable_id(team_id, &self.kind, self.title.as_deref(), &content));
        TeamMemoryRecord {
            id,
            team_id: team_id.to_owned(),
            member_id: self.member_id,
            session_id: self.session_id,
            kind: self.kind,
            content,
            title: self.title,
            importance: self.importance.unwrap_or(DEFAULT_IMPORTANCE),
            tags: self.tags,
            metadata: self
                .metadata
                .unwrap_or_else(|| Value::Object(serde_json::Map::new())),
            created_at,
            updated_at: self.updated_at.unwrap_or(created_at),
            source: self.source,
            visibility: self.visibility,
            expires_at: self.expires_at,
        }
    }
}

/// Renders one entry: heading, payload, metadata, blank separator, content, blank separator.
fn render_entry(lines: &mut Vec<String>, record: &TeamMemoryRecord) {
    lines.push(format!(
        "### [{}] {}",
        section_tag(&record.kind),
        heading_text(record)
    ));
    let payload = Payload::from_record(record);
    let json = serde_json::to_string(&payload).unwrap_or_else(|error| {
        warn!(
            target: LOG_TARGET,
            %error,
            id = %record.id,
            "team memory markdown payload could not be encoded"
        );
        "{}".to_owned()
    });
    lines.push(format!("{PAYLOAD_PREFIX} {json} {PAYLOAD_SUFFIX}"));
    lines.push(human_metadata(record));
    lines.push(String::new());
    for line in record.content.trim_end().lines() {
        lines.push(escape_body_line(line));
    }
    lines.push(String::new());
}

/// The heading text of an entry: the title, else the first non-blank content line.
fn heading_text(record: &TeamMemoryRecord) -> String {
    if let Some(title) = record.title.as_deref() {
        let trimmed = title.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    record
        .content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_owned()
}

/// The deterministic human metadata line under a payload.
fn human_metadata(record: &TeamMemoryRecord) -> String {
    let mut parts = vec![format!("id: {}", record.id)];
    parts.push(format!(
        "member: {}",
        record.member_id.as_deref().unwrap_or("team")
    ));
    parts.push(format!("importance: {:.2}", record.importance));
    if !record.tags.is_empty() {
        parts.push(format!("tags: {}", record.tags.join(", ")));
    }
    parts.push(format!("updated: {}", record.updated_at.format("%Y-%m-%d")));
    format!("*{}*", parts.join(" \u{00b7} "))
}

/// Splits a legacy heading into its optional `[tag]` and its remaining text.
fn heading_parts(heading: &str) -> (Option<&str>, &str) {
    if let Some(rest) = heading.strip_prefix('[') {
        if let Some((tag, tail)) = rest.split_once(']') {
            let tag = tag.trim();
            if !tag.is_empty() {
                return (Some(tag), tail.trim());
            }
        }
    }
    (None, heading.trim())
}

/// Maps a `[tag]` (or a section name) onto a kind.
///
/// `custom` is a section name, not a kind name: without a payload the original
/// caller-defined kind is unknown, so it falls back to [`TeamMemoryKind::SharedFact`].
fn kind_from_tag(tag: &str) -> TeamMemoryKind {
    if tag.trim().eq_ignore_ascii_case(CUSTOM_SECTION) {
        return TeamMemoryKind::SharedFact;
    }
    TeamMemoryKind::parse(tag)
}

/// The section a kind renders under: the extractor tag, or `custom`.
fn section_tag(kind: &TeamMemoryKind) -> &str {
    if matches!(kind, TeamMemoryKind::Custom(_)) {
        CUSTOM_SECTION
    } else {
        kind.extractor_tag()
    }
}

/// The position of a kind's section in [`SECTION_ORDER`].
fn kind_rank(kind: &TeamMemoryKind) -> usize {
    let tag = section_tag(kind);
    SECTION_ORDER
        .iter()
        .position(|candidate| *candidate == tag)
        .unwrap_or(SECTION_ORDER.len() - 1)
}

/// Sorts records into the documented render order.
fn sort_records(records: &mut [TeamMemoryRecord]) {
    records.sort_by(|left, right| {
        kind_rank(&left.kind)
            .cmp(&kind_rank(&right.kind))
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| left.id.cmp(&right.id))
            .then_with(|| left.content.cmp(&right.content))
    });
}

/// A deterministic id for a payload-less entry (FNV-1a 64-bit, 16 lowercase hex chars).
///
/// The id depends only on the team id, the kind, the title, and the content, so importing
/// the same Python-authored text twice produces the same ids and an upsert changes nothing.
fn stable_id(team_id: &str, kind: &TeamMemoryKind, title: Option<&str>, content: &str) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for part in [team_id, kind.as_str(), title.unwrap_or_default(), content] {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(PART_SEPARATOR);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{DERIVED_ID_PREFIX}{hash:016x}")
}

/// Parses the payload comment at `line` (1-based `number`).
fn parse_payload(line: &str, number: usize) -> Result<Payload> {
    let json = line
        .strip_prefix(PAYLOAD_PREFIX)
        .and_then(|rest| rest.strip_suffix(PAYLOAD_SUFFIX))
        .map(str::trim)
        .ok_or_else(|| {
            with_line_number(
                number,
                Error::validation("team memory markdown payload is not closed by `-->`"),
            )
        })?;
    serde_json::from_str(json).map_err(|error| {
        with_line_number(
            number,
            Error::validation(format!(
                "team memory markdown payload is not a valid record: {error}"
            )),
        )
    })
}

/// Returns `true` for the human metadata line directly under a payload.
fn is_metadata_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with(METADATA_PREFIX)
        && trimmed.ends_with('*')
        && trimmed.len() > METADATA_PREFIX.len()
}

/// Normalizes the section name of a `## ` heading.
fn normalize_tag(heading: &str) -> String {
    heading.trim().to_ascii_lowercase()
}

/// Joins body lines, dropping leading and trailing blank lines (payload-less entries).
fn trim_body(lines: &[String]) -> String {
    let Some(first) = lines.iter().position(|line| !line.trim().is_empty()) else {
        return String::new();
    };
    let last = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .unwrap_or(first);
    lines[first..=last].join("\n").trim_end().to_owned()
}

/// Escapes a body line that could otherwise read as a heading or an escape.
fn escape_body_line(line: &str) -> String {
    if line.starts_with('#') || line.starts_with('\\') {
        format!("\\{line}")
    } else {
        line.to_owned()
    }
}

/// Removes the single leading backslash [`escape_body_line`] added, if any.
fn unescape_body_line(line: &str) -> String {
    line.strip_prefix('\\')
        .map_or_else(|| line.to_owned(), str::to_owned)
}

/// Normalizes `\r\n` and lone `\r` to `\n`.
fn normalize_newlines(content: &str) -> String {
    content.replace("\r\n", "\n").replace('\r', "\n")
}

/// Formats a timestamp as RFC 3339 UTC with `Z` and no unnecessary fraction digits.
fn format_timestamp(timestamp: DateTime<Utc>) -> String {
    timestamp.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// Prefixes a validation error with the 1-based line it came from.
fn with_line_number(number: usize, error: Error) -> Error {
    match error {
        Error::Validation(message) => {
            Error::validation(format!("team memory markdown line {number}: {message}"))
        }
        other => other,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::team::model::NewTeamMemory;
    use chrono::TimeZone;
    use openjiuwen_common::error::Error;
    use tokio_util::sync::CancellationToken;

    fn ts(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).single().unwrap()
    }

    fn record(
        id: &str,
        kind: TeamMemoryKind,
        title: Option<&str>,
        content: &str,
        updated_at: i64,
    ) -> TeamMemoryRecord {
        TeamMemoryRecord {
            id: id.to_owned(),
            title: title.map(str::to_owned),
            updated_at: ts(updated_at),
            created_at: ts(1_699_000_000),
            ..TeamMemoryRecord::new(&TeamMemoryScope::new("review-team"), content, kind)
        }
    }

    /// Eight records: two decisions with tied `updated_at`, one of each remaining builtin
    /// kind, and a caller-defined kind, exercising every visibility and awkward content.
    fn sample_document() -> TeamMemoryDocument {
        let mut first = record(
            "018f1d2e3f4a5b6c7d8e9001",
            TeamMemoryKind::Decision,
            Some("Storage choice"),
            "使用 SQLite 作为共享存储。\n## 权衡\n### 不是 PostgreSQL\nfast enough",
            1_700_000_300,
        );
        first.member_id = Some("alice".to_owned());
        first.session_id = Some("s-1".to_owned());
        first.importance = 0.8;
        first.tags = vec!["storage".to_owned(), "sqlite".to_owned()];
        first.metadata = serde_json::json!({
            "issue": 42,
            "votes": {"alice": 1, "bob": 2},
            "nested": [1, 2, 3]
        });
        first.visibility = TeamVisibility::members(["alice", "bob"]);
        first.source = Some("leader-extraction".to_owned());
        first.expires_at = Some(ts(1_800_000_000));
        let mut second = record(
            "018f1d2e3f4a5b6c7d8e9000",
            TeamMemoryKind::Decision,
            Some("Skip PostgreSQL"),
            "Operational overhead is too high.",
            1_700_000_300,
        );
        second.member_id = Some("alice".to_owned());
        let mut profile = record(
            "018f1d2e3f4a5b6c7d8e9002",
            TeamMemoryKind::MemberProfile,
            Some("Backend owner"),
            "Bob owns the storage layer.",
            1_700_000_200,
        );
        profile.member_id = Some("bob".to_owned());
        profile.visibility = TeamVisibility::Private;
        let mut lesson = record(
            "018f1d2e3f4a5b6c7d8e9003",
            TeamMemoryKind::HandoffNote,
            None,
            "Align interfaces before parallel work.",
            1_700_000_100,
        );
        lesson.importance = 0.65;
        let fact = record(
            "018f1d2e3f4a5b6c7d8e9004",
            TeamMemoryKind::SharedFact,
            None,
            "  indented first line\nsecond line",
            1_700_000_050,
        );
        let mut task = record(
            "018f1d2e3f4a5b6c7d8e9005",
            TeamMemoryKind::Task,
            Some("Wire the migration"),
            "Run the SQLite migration before the release.",
            1_700_000_400,
        );
        task.importance = 0.9;
        task.tags = vec!["release".to_owned()];
        let artifact = record(
            "018f1d2e3f4a5b6c7d8e9006",
            TeamMemoryKind::Artifact,
            Some("docs/design.md"),
            "Design doc for the shared store.",
            1_700_000_250,
        );
        let mut custom = record(
            "018f1d2e3f4a5b6c7d8e9007",
            TeamMemoryKind::Custom("review_rule".to_owned()),
            Some("Review rule"),
            "Two approvals for schema changes.",
            1_700_000_050,
        );
        custom.metadata = serde_json::json!({"threshold": 2});
        TeamMemoryDocument::from_records(
            "review-team",
            vec![first, second, profile, lesson, fact, task, artifact, custom],
            ts(1_700_010_000),
        )
    }

    fn assert_same_records(actual: &[TeamMemoryRecord], expected: &[TeamMemoryRecord]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                serde_json::to_value(left).unwrap(),
                serde_json::to_value(right).unwrap(),
                "record {index} ({} / {}) differs",
                left.id,
                right.id
            );
        }
    }

    #[test]
    fn export_then_parse_round_trips_every_field() {
        let document = sample_document();
        let markdown = document.render();
        let parsed = TeamMemoryDocument::parse(&markdown, None).unwrap();
        assert_eq!(parsed.team_id, "review-team");
        assert_eq!(parsed.generated_at, document.generated_at);
        assert_same_records(&parsed.entries, &document.entries);
        let by_id = |id: &str| {
            parsed
                .entries
                .iter()
                .find(|entry| entry.id == id)
                .unwrap()
                .clone()
        };
        let first = by_id("018f1d2e3f4a5b6c7d8e9001");
        assert_eq!(first.kind, TeamMemoryKind::Decision);
        assert_eq!(
            first.content,
            "使用 SQLite 作为共享存储。\n## 权衡\n### 不是 PostgreSQL\nfast enough"
        );
        assert_eq!(first.tags, ["storage", "sqlite"]);
        assert_eq!(first.metadata["votes"]["bob"], serde_json::json!(2));
        assert_eq!(first.visibility, TeamVisibility::members(["alice", "bob"]));
        assert_eq!(first.expires_at, Some(ts(1_800_000_000)));
        assert_eq!(
            first.created_at, parsed.entries[0].created_at,
            "payload timestamps are restored"
        );
        assert_eq!(first.created_at, ts(1_699_000_000));
        assert_eq!(
            by_id("018f1d2e3f4a5b6c7d8e9007").kind,
            TeamMemoryKind::Custom("review_rule".to_owned())
        );
        assert_eq!(
            by_id("018f1d2e3f4a5b6c7d8e9002").visibility,
            TeamVisibility::Private
        );
        assert!(
            markdown.contains("\\## 权衡\n\\### 不是 PostgreSQL"),
            "content headings are escaped so they cannot be read as structure"
        );
    }

    #[test]
    fn render_has_the_documented_shape_and_order() {
        let markdown = sample_document().render();
        let lines: Vec<&str> = markdown.lines().collect();
        assert_eq!(lines[0], "---");
        assert_eq!(lines[1], "team_memory: openjiuwen-team-memory/v1");
        assert_eq!(lines[2], "team_id: review-team");
        assert_eq!(lines[3], "generated_at: 2023-11-15T01:00:00Z");
        assert_eq!(lines[4], "entries: 8");
        assert_eq!(lines[5], "---");
        assert!(markdown.contains("# Team memory \u{2014} review-team"));
        let sections: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|line| line.starts_with("## "))
            .collect();
        assert_eq!(
            sections,
            [
                "## decision",
                "## member",
                "## lesson",
                "## context",
                "## custom"
            ]
        );
        let headings: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|line| line.starts_with("### "))
            .collect();
        assert_eq!(
            headings,
            [
                "### [decision] Skip PostgreSQL",
                "### [decision] Storage choice",
                "### [member] Backend owner",
                "### [lesson] Align interfaces before parallel work.",
                "### [context] Wire the migration",
                "### [context] docs/design.md",
                "### [context] indented first line",
                "### [custom] Review rule",
            ]
        );
        let payload_line = lines
            .iter()
            .copied()
            .find(|line| line.starts_with(PAYLOAD_PREFIX))
            .unwrap();
        let json = payload_line
            .trim_start_matches(PAYLOAD_PREFIX)
            .trim_end_matches(PAYLOAD_SUFFIX)
            .trim();
        let value: Value = serde_json::from_str(json).unwrap();
        assert_eq!(value["id"], serde_json::json!("018f1d2e3f4a5b6c7d8e9000"));
        assert_eq!(value["kind"], serde_json::json!("decision"));
        assert!(value.get("content").is_none(), "content stays in the body");
        assert!(
            value.get("team_id").is_none(),
            "team id stays in the front matter"
        );
        assert!(lines
            .iter()
            .any(|line| line.starts_with("*id: 018f1d2e3f4a5b6c7d8e9001 · member: alice · importance: 0.80 · tags: storage, sqlite · updated: 2023-11-14*")));
        assert!(markdown.ends_with("Two approvals for schema changes.\n"));
        assert!(sample_document().render().contains("entries: 8"));
    }

    #[test]
    fn render_parse_render_is_byte_identical() {
        let markdown = sample_document().render();
        let reparsed = TeamMemoryDocument::parse(&markdown, None).unwrap();
        assert_eq!(reparsed.render(), markdown);
        assert_eq!(TeamMemoryDocument::parse("", Some("empty-team")).unwrap().render(), "---\nteam_memory: openjiuwen-team-memory/v1\nteam_id: empty-team\ngenerated_at: 1970-01-01T00:00:00Z\nentries: 0\n---\n\n# Team memory \u{2014} empty-team\n");
    }

    #[test]
    fn sorts_ties_by_id_and_kind_in_the_fixed_section_order() {
        let document = sample_document();
        let decision_ids: Vec<&str> = document
            .entries
            .iter()
            .filter(|entry| entry.kind == TeamMemoryKind::Decision)
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(
            decision_ids,
            ["018f1d2e3f4a5b6c7d8e9000", "018f1d2e3f4a5b6c7d8e9001"]
        );
        let ranks: Vec<usize> = document
            .entries
            .iter()
            .map(|entry| kind_rank(&entry.kind))
            .collect();
        let mut sorted = ranks.clone();
        sorted.sort_unstable();
        assert_eq!(ranks, sorted);
    }

    #[test]
    fn python_authored_document_parses_with_a_fallback_team_id() {
        let text = "### [decision] 选择方案 A\n原因是更简单。\n### [lesson] 并行任务需要先对齐接口\n上次因为没对齐导致返工 2 天。\n### [member] Alice 负责存储\n她维护索引与迁移。\n";
        let document = TeamMemoryDocument::parse(text, Some("review-team")).unwrap();
        assert_eq!(document.team_id, "review-team");
        assert_eq!(document.generated_at, DateTime::UNIX_EPOCH);
        assert_eq!(document.len(), 3);
        let decision = &document.entries[0];
        assert_eq!(decision.kind, TeamMemoryKind::Decision);
        assert_eq!(decision.title.as_deref(), Some("选择方案 A"));
        assert_eq!(decision.content, "原因是更简单。");
        assert_eq!(decision.importance, DEFAULT_IMPORTANCE);
        assert_eq!(decision.source.as_deref(), Some("TEAM_MEMORY.md"));
        assert_eq!(decision.visibility, TeamVisibility::Team);
        assert_eq!(decision.created_at, DateTime::UNIX_EPOCH);
        assert_eq!(decision.updated_at, DateTime::UNIX_EPOCH);
        assert_eq!(decision.member_id, None);
        assert!(decision.id.starts_with("md-"));
        assert_eq!(decision.id.len(), 19);
        assert_eq!(document.entries[1].kind, TeamMemoryKind::HandoffNote);
        assert_eq!(document.entries[2].kind, TeamMemoryKind::MemberProfile);
        assert_eq!(document.entries[2].title.as_deref(), Some("Alice 负责存储"));
        let again = TeamMemoryDocument::parse(text, Some("review-team")).unwrap();
        let ids: Vec<&str> = document.entries.iter().map(|e| e.id.as_str()).collect();
        let again_ids: Vec<&str> = again.entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, again_ids);
        let other_team = TeamMemoryDocument::parse(text, Some("other-team")).unwrap();
        assert_ne!(other_team.entries[0].id, document.entries[0].id);
        assert!(TeamMemoryDocument::parse(text, None)
            .unwrap_err()
            .to_string()
            .contains("team_id"));
    }

    #[test]
    fn section_tags_supply_the_kind_when_a_heading_has_none() {
        let text = "## lesson\n\n### Align interfaces\n\nBody.\n\n## mystery\n\n### Loose heading\n\nMore.\n";
        let document = TeamMemoryDocument::parse(text, Some("review-team")).unwrap();
        assert_eq!(document.len(), 2);
        assert_eq!(document.entries[0].kind, TeamMemoryKind::HandoffNote);
        assert_eq!(
            document.entries[0].title.as_deref(),
            Some("Align interfaces")
        );
        assert_eq!(
            document.entries[1].kind,
            TeamMemoryKind::Custom("mystery".to_owned())
        );
    }

    #[test]
    fn unknown_sections_and_empty_bodies_are_skipped() {
        let text =
            "## mystery\n\n### [decision] Kept\n\nbody\n\n### [context] Empty\n\n### Stray\n";
        let document = TeamMemoryDocument::parse(text, Some("review-team")).unwrap();
        assert_eq!(document.len(), 1);
        assert_eq!(document.entries[0].title.as_deref(), Some("Kept"));
        assert_eq!(document.entries[0].content, "body");
    }

    #[test]
    fn bad_front_matter_and_bad_payloads_name_their_line() {
        let bad_team = "---\nteam_memory: openjiuwen-team-memory/v1\nteam_id: bad/team\ngenerated_at: 2026-01-02T03:04:05Z\nentries: 0\n---\n";
        let error = TeamMemoryDocument::parse(bad_team, None).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("line 3"), "{error}");
        assert!(error.to_string().contains("team_id"), "{error}");
        let bad_payload = "---\nteam_id: review-team\n---\n\n## decision\n\n### [decision] X\n<!-- team-memory: {not json} -->\n\nbody\n";
        let error = TeamMemoryDocument::parse(bad_payload, None).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("line 8"), "{error}");
        let unterminated_payload = "---\nteam_id: review-team\n---\n\n### [decision] X\n<!-- team-memory: {\"kind\":\"decision\"}\n\nbody\n";
        let error = TeamMemoryDocument::parse(unterminated_payload, None).unwrap_err();
        assert!(error.to_string().contains("line 6"), "{error}");
        assert!(TeamMemoryDocument::parse("no front matter and no fallback", None).is_err());
        assert_eq!(
            TeamMemoryDocument::parse("no front matter and no fallback", Some("t"))
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn crlf_blank_padding_unknown_keys_and_format_drift_are_tolerated() {
        let text = "\r\n---\r\nunknown: 1\r\nteam_memory: v99\r\nteam_id: review-team\r\n\r\n---\r\n\r\n# Team memory\r\n\r\n## decision\r\n\r\n### [decision] Kept\r\n<!-- team-memory: {\"kind\":\"decision\",\"title\":\"Kept\",\"unknown_field\":true} -->\r\n*id: whatever*\r\n\r\nline one\r\nline two\r\n\r\n";
        let document = TeamMemoryDocument::parse(text, None).unwrap();
        assert_eq!(document.team_id, "review-team");
        assert_eq!(document.len(), 1);
        assert_eq!(document.entries[0].content, "line one\nline two");
        assert_eq!(document.entries[0].title.as_deref(), Some("Kept"));
        assert_eq!(document.entries[0].importance, DEFAULT_IMPORTANCE);
        assert_eq!(document.entries[0].metadata, serde_json::json!({}));
        assert_eq!(document_team_id(text).as_deref(), Some("review-team"));
        assert_eq!(document_team_id("### [decision] x\nbody\n"), None);
    }

    #[test]
    fn a_payload_without_an_id_gets_the_derived_one() {
        let text = "---\nteam_id: review-team\ngenerated_at: 2026-01-02T03:04:05Z\n---\n\n## decision\n\n### [decision] Kept\n<!-- team-memory: {\"kind\":\"decision\",\"title\":\"Kept\"} -->\n\nBody.\n";
        let document = TeamMemoryDocument::parse(text, None).unwrap();
        let entry = &document.entries[0];
        assert_eq!(
            entry.id,
            stable_id(
                "review-team",
                &TeamMemoryKind::Decision,
                Some("Kept"),
                "Body."
            )
        );
        assert_eq!(entry.created_at, ts(1_767_323_045));
        assert_eq!(entry.updated_at, entry.created_at);
    }

    #[test]
    fn truncate_for_prompt_keeps_the_first_two_hundred_lines() {
        use std::fmt::Write as _;
        let mut body = String::new();
        for index in 1..=260 {
            writeln!(body, "line {index}").unwrap();
        }
        let truncated = truncate_for_prompt(&body);
        assert_eq!(truncated.lines().count(), TEAM_MEMORY_MAX_READ_LINES);
        assert!(truncated.starts_with("line 1\n"));
        assert!(truncated.ends_with("line 200"));
        assert_eq!(truncate_for_prompt("  \n\n only \n\n"), "only");
        let document = sample_document();
        assert_eq!(
            truncate_for_prompt(&document.render()),
            document.render().trim()
        );
    }

    async fn seeded_memory() -> TeamMemory {
        let memory = TeamMemory::in_memory();
        let call = CallOptions::new();
        memory
            .remember(
                NewTeamMemory::decision("review-team", "Use SQLite for the shared store.")
                    .with_member("alice")
                    .with_title("Storage choice")
                    .with_tag("storage")
                    .with_visibility(TeamVisibility::members(["alice", "bob"])),
                &call,
            )
            .await
            .unwrap();
        memory
            .remember(
                NewTeamMemory::handoff_note("review-team", "Align interfaces first.")
                    .with_importance(0.65),
                &call,
            )
            .await
            .unwrap();
        memory
            .remember(
                NewTeamMemory::shared_fact("review-team", "Python 3.11 is the target runtime."),
                &call,
            )
            .await
            .unwrap();
        memory
    }

    #[tokio::test]
    async fn facade_export_then_import_round_trips_and_is_idempotent() {
        let source = seeded_memory().await;
        let filter = TeamMemoryFilter::for_team("review-team");
        let markdown = source.export_markdown(&filter).await.unwrap();
        assert!(markdown.contains("team_id: review-team"));
        let target = TeamMemory::in_memory();
        let imported = target
            .import_markdown(&markdown, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(imported.len(), 3);
        let source_ids: Vec<String> = source
            .list(&filter)
            .await
            .unwrap()
            .into_iter()
            .map(|record| record.id)
            .collect();
        let imported_ids: Vec<String> = imported.iter().map(|record| record.id.clone()).collect();
        let mut sorted_source = source_ids.clone();
        sorted_source.sort();
        let mut sorted_imported = imported_ids;
        sorted_imported.sort();
        assert_eq!(sorted_imported, sorted_source);
        assert_same_records(
            &target.list(&filter).await.unwrap(),
            &source.list(&filter).await.unwrap(),
        );
        let again = target
            .import_markdown(&markdown, &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(again.len(), 3);
        let listed = target.list(&filter).await.unwrap();
        assert_eq!(listed.len(), 3);
        for record in &listed {
            let first = imported.iter().find(|entry| entry.id == record.id).unwrap();
            assert_eq!(record.created_at, first.created_at);
        }
    }

    #[tokio::test]
    async fn cancelled_import_writes_nothing_and_empty_export_renders() {
        let memory = TeamMemory::in_memory();
        let markdown = memory
            .export_markdown(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap();
        assert!(markdown.contains("entries: 0"));
        let token = CancellationToken::new();
        token.cancel();
        let call = CallOptions::new().with_cancellation(token);
        let error = memory
            .import_markdown(
                "---\nteam_id: review-team\n---\n\n### [decision] X\n\nbody\n",
                &call,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(
            memory
                .list(&TeamMemoryFilter::for_team("review-team"))
                .await
                .unwrap()
                .len(),
            0
        );
        let error = memory
            .import_markdown("### [decision] X\n\nbody\n", &CallOptions::new())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("team_id"), "{error}");
    }

    #[tokio::test]
    async fn exporting_one_filter_only_renders_matching_records() {
        let memory = seeded_memory().await;
        let filter = TeamMemoryFilter::for_team("review-team").member("alice");
        let markdown = memory.export_markdown(&filter).await.unwrap();
        let document = TeamMemoryDocument::parse(&markdown, None).unwrap();
        assert_eq!(document.len(), 1);
        assert_eq!(document.entries[0].member_id.as_deref(), Some("alice"));
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_export_import_survives_a_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let source = TeamMemory::sqlite(directory.path().join("source.sqlite3"))
            .await
            .unwrap();
        let call = CallOptions::new();
        source
            .remember(
                NewTeamMemory::decision("review-team", "Use SQLite.").with_title("Storage"),
                &call,
            )
            .await
            .unwrap();
        source
            .remember(
                NewTeamMemory::artifact("review-team", "docs/design.md"),
                &call,
            )
            .await
            .unwrap();
        let filter = TeamMemoryFilter::for_team("review-team");
        let markdown = source.export_markdown(&filter).await.unwrap();
        let copy_path = directory.path().join("copy.sqlite3");
        {
            let target = TeamMemory::sqlite(&copy_path).await.unwrap();
            let imported = target.import_markdown(&markdown, &call).await.unwrap();
            assert_eq!(imported.len(), 2);
        }
        let reopened = TeamMemory::sqlite(&copy_path).await.unwrap();
        let records = reopened.list(&filter).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_same_records(&records, &source.list(&filter).await.unwrap());
    }
}
