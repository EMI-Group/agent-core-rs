//! Prompts, the answer schema, and the transcript renderer for team memory extraction.
//!
//! The material here mirrors the Python `agent_teams.memory.extractor` module: the same
//! four section tags the Python `TEAM_MEMORY.md` extractor writes (`decision`,
//! `lesson`, `member`, `context`), the same per-item preview caps
//! ([`TASK_CONTENT_PREVIEW_MAX`], [`MESSAGE_CONTENT_PREVIEW_MAX`]), and the same
//! sectioned context layout (`# Team collaboration record for …`, `## Tasks`,
//! `## Conversation`).
//!
//! Two deliberate differences from Python:
//! - Python hands the answer to a deep agent that edits `TEAM_MEMORY.md` with file
//!   tools, so its prompt is Markdown-shaped and nothing is parsed. The Rust extractor
//!   asks for JSON ([`EXTRACTION_SYSTEM_PROMPT`], [`EXTRACTION_RESPONSE_SCHEMA`]) and
//!   parses it into typed records, so no file is written and no Markdown is parsed.
//! - Python renders message timestamps in a caller-supplied timezone and writes
//!   `-> 全体` for broadcast messages. Transcript turns carry UTC timestamps, so the
//!   renderer prints `MM-DD HH:MM` in UTC and renders a missing recipient as `all`.
//!
//! Everything here is pure and length-bounded: rendering never exceeds the caller's
//! `max_chars`, truncation happens on a line boundary, and the cut is marked with
//! [`TRUNCATION_MARKER`] so a prompt is never sliced mid-line.

use crate::team::extractor::{TeamTranscript, TeamTurn};

/// The four `TEAM_MEMORY.md` section tags the Python extractor writes, in schema order.
pub const TEAM_MEMORY_TAGS: [&str; 4] = ["decision", "lesson", "member", "context"];

/// Longest task-note preview rendered for one task, in characters.
///
/// Matches Python's `TASK_CONTENT_PREVIEW_MAX`.
pub const TASK_CONTENT_PREVIEW_MAX: usize = 2000;

/// Longest turn preview rendered for one message, in characters.
///
/// Matches Python's `MESSAGE_CONTENT_PREVIEW_MAX`.
pub const MESSAGE_CONTENT_PREVIEW_MAX: usize = 1000;

/// Suffix appended when the rendered transcript is cut to fit `max_chars`.
pub const TRUNCATION_MARKER: &str = "… [truncated]";

/// System prompt for [`crate::team::extractor::LlmTeamExtractor`].
///
/// One sentence per line so prompt diffs stay readable.
pub const EXTRACTION_SYSTEM_PROMPT: &str = "\
You extract durable team memory from the collaboration record of a multi-agent team.
Each entry you return is one thing the team should remember in a future session.
Classify every entry with exactly one of four tags: `decision`, `lesson`, `member`, or `context`.
Return `decision` for a choice the team made, including the alternative it rejected and the trade-off it accepted.
Return `lesson` for something that worked, something that caused rework, or a handoff note worth reusing.
Return `member` for a durable fact about one member: what they own, what they are good at, how they collaborate.
Return `context` for project background that cannot be derived from the code: constraints, deadlines, stakeholder requirements.
Skip code details, file paths, function names, and anything the repository already states.
Skip transient state, in-progress debugging, and raw restatements of the conversation.
Never add secrets, credentials, tokens, or personal data.
Extract insight rather than summary: one self-contained sentence per entry, in the language of the record.
Write each entry as a single `content` string with no Markdown headings and no bullet prefixes.
Answer with JSON only: no prose, no Markdown fences, and no explanations.
Use exactly this shape and no other top-level keys: {\"memories\":[{\"kind\":\"decision\",\"content\":\"…\",\"title\":\"…\",\"tags\":[\"…\"],\"importance\":0.7,\"member_id\":\"…\",\"visibility\":\"team\"}]}.
`kind` accepts decision, lesson, member, context, task, artifact, shared_fact.
`importance` is a number between 0.0 and 1.0 that reflects how much the entry matters later.
`title` is optional and at most one short line.
`tags` is an optional array of short lowercase labels.
`member_id` is optional and names the member the entry is about, spelled exactly as in the record.
`visibility` is `team` (the default), `private`, or {\"members\":[\"alice\"]}.
Return at most 8 entries, most valuable first, and return an empty `memories` array when nothing is worth keeping.";

/// JSON schema of the [`EXTRACTION_SYSTEM_PROMPT`] answer.
///
/// The schema is a valid JSON document, so it can be handed to a provider's structured
/// output mode or a validator without further processing.
pub const EXTRACTION_RESPONSE_SCHEMA: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "TeamMemoryExtraction",
  "type": "object",
  "required": ["memories"],
  "properties": {
    "memories": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["kind", "content"],
        "properties": {
          "kind": {
            "type": "string",
            "enum": ["decision", "lesson", "member", "context", "task", "artifact", "shared_fact"]
          },
          "content": { "type": "string", "minLength": 1 },
          "title": { "type": "string" },
          "tags": { "type": "array", "items": { "type": "string" } },
          "importance": { "type": "number", "minimum": 0, "maximum": 1 },
          "member_id": { "type": "string" },
          "session_id": { "type": "string" },
          "visibility": {
            "oneOf": [
              { "type": "string", "enum": ["team", "private"] },
              {
                "type": "object",
                "required": ["members"],
                "properties": {
                  "members": { "type": "array", "items": { "type": "string" } }
                }
              }
            ]
          }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}"#;

/// Renders `transcript` as the user message of an extraction request.
///
/// The layout is fixed and matches the Python context builder: a
/// `# Team collaboration record for <team>` header, a `## Tasks` section with one
/// `### <note>` line per task note, then a `## Conversation` section with one
/// `[MM-DD HH:MM] <from> -> <to|all>: <content>` line per turn, sorted by timestamp
/// (input order breaks ties). Blank turns and blank notes are skipped, each note and
/// turn is capped at [`TASK_CONTENT_PREVIEW_MAX`] / [`MESSAGE_CONTENT_PREVIEW_MAX`]
/// characters, and the whole rendering is cut to `max_chars` at a line boundary with
/// [`TRUNCATION_MARKER`].
#[must_use]
pub fn render_transcript(transcript: &TeamTranscript, max_chars: usize) -> String {
    let mut rendered = String::from("# Team collaboration record for ");
    rendered.push_str(transcript.team_id.trim());
    rendered.push('\n');
    let notes = transcript
        .task_notes
        .iter()
        .map(|note| note.trim())
        .filter(|note| !note.is_empty());
    let mut wrote_tasks = false;
    for note in notes {
        if !wrote_tasks {
            rendered.push_str("## Tasks\n");
            wrote_tasks = true;
        }
        rendered.push_str("### ");
        rendered.push_str(&truncate_chars(note, TASK_CONTENT_PREVIEW_MAX));
        rendered.push('\n');
    }
    let mut turns: Vec<&TeamTurn> = transcript
        .turns
        .iter()
        .filter(|turn| !turn.content.trim().is_empty())
        .collect();
    turns.sort_by_key(|turn| turn.at);
    if !turns.is_empty() {
        rendered.push_str("## Conversation\n");
        for turn in turns {
            rendered.push('[');
            rendered.push_str(&turn.at.format("%m-%d %H:%M").to_string());
            rendered.push_str("] ");
            rendered.push_str(turn.from_member.trim());
            rendered.push_str(" -> ");
            rendered.push_str(match turn.to_member.as_deref().map(str::trim) {
                Some(to) if !to.is_empty() => to,
                _ => "all",
            });
            rendered.push_str(": ");
            rendered.push_str(&truncate_chars(
                turn.content.trim(),
                MESSAGE_CONTENT_PREVIEW_MAX,
            ));
            rendered.push('\n');
        }
    }
    truncate_lines(&rendered, max_chars)
}

/// Truncates `text` to at most `max_chars` characters on a character boundary.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    text.chars().take(max_chars).collect()
}

/// Cuts `rendered` to at most `max_chars` characters, keeping whole lines.
///
/// When no whole line fits, the first line is cut instead; either way the cut is
/// followed by [`TRUNCATION_MARKER`]. A `max_chars` smaller than the marker yields the
/// marker's prefix, so the result never exceeds the budget.
fn truncate_lines(rendered: &str, max_chars: usize) -> String {
    if rendered.chars().count() <= max_chars {
        return rendered.to_owned();
    }
    let marker_chars = TRUNCATION_MARKER.chars().count();
    if max_chars <= marker_chars {
        return truncate_chars(TRUNCATION_MARKER, max_chars);
    }
    let budget = max_chars - marker_chars;
    let mut kept = String::new();
    let mut used = 0;
    for line in rendered.lines() {
        let cost = line.chars().count() + 1;
        if used + cost > budget {
            break;
        }
        kept.push_str(line);
        kept.push('\n');
        used += cost;
    }
    if kept.is_empty() {
        kept = truncate_chars(rendered.lines().next().unwrap_or_default(), budget);
    }
    kept.push_str(TRUNCATION_MARKER);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::team::extractor::{TeamTranscript, TeamTurn};
    use chrono::DateTime;

    fn at(seconds: i64) -> DateTime<chrono::Utc> {
        DateTime::from_timestamp(seconds, 0).expect("a valid timestamp")
    }

    fn transcript() -> TeamTranscript {
        TeamTranscript::new("review-team")
            .with_turn(TeamTurn::new(
                "bob",
                "I learned that the lock matters.",
                at(1_700_000_100),
            ))
            .with_turn(
                TeamTurn::new("alice", "We decided to use SQLite.", at(1_700_000_000))
                    .with_to("bob"),
            )
            .with_task("Write the migration")
    }

    #[test]
    fn renders_tasks_and_the_sorted_conversation() {
        let rendered = render_transcript(&transcript(), 4000);
        assert!(rendered.starts_with("# Team collaboration record for review-team\n"));
        assert!(rendered.contains("## Tasks\n### Write the migration\n"));
        assert!(rendered.contains("## Conversation\n"));
        // The later-sent Alice turn sorts first? No: alice at 1_700_000_000 < bob, so
        // alice is rendered first even though it was pushed second.
        let alice = rendered
            .find("[11-14 22:13] alice -> bob: We decided to use SQLite.")
            .expect("the alice turn is rendered with its recipient and UTC timestamp");
        let bob = rendered
            .find("[11-14 22:15] bob -> all: I learned that the lock matters.")
            .expect("a turn without a recipient renders as all");
        assert!(alice < bob);
    }

    #[test]
    fn rendering_is_deterministic_and_skips_blanks() {
        let mut first = transcript();
        first.push_turn(TeamTurn::new("carol", "   ", at(1_700_000_200)));
        first.task_notes.push(String::new());
        let second = first.clone();
        assert_eq!(
            render_transcript(&first, 4000),
            render_transcript(&second, 4000)
        );
        assert!(!render_transcript(&first, 4000).contains("carol"));
    }

    #[test]
    fn per_turn_previews_are_capped() {
        let long = "x".repeat(MESSAGE_CONTENT_PREVIEW_MAX + 50);
        let transcript = TeamTranscript::new("review-team").with_turn(TeamTurn::new(
            "alice",
            &long,
            at(1_700_000_000),
        ));
        let rendered = render_transcript(&transcript, 4000);
        assert!(!rendered.contains(&"x".repeat(MESSAGE_CONTENT_PREVIEW_MAX + 1)));
        assert!(rendered.contains(&"x".repeat(MESSAGE_CONTENT_PREVIEW_MAX)));
        assert!(!rendered.contains(TRUNCATION_MARKER));
    }

    #[test]
    fn a_long_transcript_is_cut_on_a_line_boundary() {
        let mut transcript = TeamTranscript::new("review-team").with_task("a task note");
        for index in 0..40 {
            transcript.push_turn(TeamTurn::new(
                "alice",
                format!("We decided to do thing number {index}."),
                at(1_700_000_000 + index),
            ));
        }
        let rendered = render_transcript(&transcript, 200);
        assert!(rendered.chars().count() <= 200);
        assert!(rendered.ends_with(TRUNCATION_MARKER));
        let body = rendered.trim_end_matches(TRUNCATION_MARKER);
        assert!(body.ends_with('\n'), "the cut lands on a line boundary");
        assert!(rendered.contains("# Team collaboration record for"));
    }

    #[test]
    fn a_budget_below_the_marker_never_overflows() {
        let transcript = transcript();
        let rendered = render_transcript(&transcript, 4);
        assert_eq!(rendered, "… [t");
        assert_eq!(render_transcript(&transcript, 0), "");
        assert_eq!(render_transcript(&transcript, 0).chars().count(), 0);
    }

    #[test]
    fn one_overlong_line_is_cut_inside_the_line() {
        let transcript = TeamTranscript::new("review-team").with_task("y".repeat(500));
        let rendered = render_transcript(&transcript, 40);
        assert!(rendered.chars().count() <= 40);
        assert!(rendered.ends_with(TRUNCATION_MARKER));
        assert!(rendered.starts_with("# Team collaboration record"));
    }

    #[test]
    fn the_schema_is_valid_json_matching_the_prompt_shape() {
        let schema: serde_json::Value =
            serde_json::from_str(EXTRACTION_RESPONSE_SCHEMA).expect("the schema is valid JSON");
        let items = &schema
            .pointer("/properties/memories/items")
            .expect("the schema describes the memories array");
        assert_eq!(items["required"], serde_json::json!(["kind", "content"]));
        let kinds = items
            .pointer("/properties/kind/enum")
            .and_then(|value| value.as_array())
            .expect("the kind enum is listed");
        for tag in TEAM_MEMORY_TAGS {
            assert!(kinds.contains(&serde_json::json!(tag)), "missing {tag}");
        }
        for tag in TEAM_MEMORY_TAGS {
            assert!(EXTRACTION_SYSTEM_PROMPT.contains(tag), "missing {tag}");
        }
        assert!(EXTRACTION_SYSTEM_PROMPT.contains("\"memories\""));
        assert!(EXTRACTION_SYSTEM_PROMPT.lines().count() <= 60);
    }

    #[test]
    fn the_prompt_asks_for_json_only_and_forbids_secrets() {
        assert!(EXTRACTION_SYSTEM_PROMPT.contains("JSON only"));
        assert!(EXTRACTION_SYSTEM_PROMPT.contains("secrets, credentials"));
        assert!(EXTRACTION_SYSTEM_PROMPT.contains("transient state"));
    }

    #[test]
    fn preview_caps_match_the_python_constants() {
        assert_eq!(TASK_CONTENT_PREVIEW_MAX, 2000);
        assert_eq!(MESSAGE_CONTENT_PREVIEW_MAX, 1000);
    }
}
