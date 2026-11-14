//! Unit tests for the Markdown document model and its front matter.
//!
//! The suite pins the render/parse contract: the exact bytes of a fresh document,
//! byte-for-byte round trips, canonical section order, and the tolerant parse of
//! hostile, hand-written, or partially malformed input. It also covers the in-memory
//! mutators (`upsert_fact`, `remove`, scope rebinding, `with_front_matter`, and the
//! title accessors). Everything runs offline and deterministically: timestamps are
//! fixed constants and nothing touches the filesystem.

use super::*;
use chrono::{DateTime, TimeZone, Utc};

use crate::lite::fact::{CodingFact, CodingFactType};
use crate::lite::markdown::front_matter::{FORMAT_VERSION, SCHEMA_MARKER};
use crate::record::MemoryScope;

fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

fn fixed_time() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 11, 13, 7, 0, 0).unwrap()
}

fn fact(name: &str, content: &str) -> CodingFact {
    let mut fact = CodingFact::new(&scope(), name, content);
    fact.updated_at = fixed_time();
    fact
}

fn sample_document() -> MarkdownMemory {
    let mut document = MarkdownMemory::new().with_scope(scope());
    let _ = document.upsert_fact(
        &fact("build", "cargo build")
            .with_description("How to build")
            .with_tags(vec!["build".to_owned()])
            .with_path("docs/BUILD.md"),
    );
    let _ = document.upsert_fact(&fact("tone", "be concise").with_type(CodingFactType::Feedback));
    let _ = document.upsert_fact(&fact("alice", "the user").with_type(CodingFactType::User));
    document
}

#[test]
fn a_fresh_document_holds_the_schema_and_title() {
    let document = MarkdownMemory::new();
    assert_eq!(document.title(), Some("Memory"));
    assert_eq!(document.front_matter().get("schema"), Some(SCHEMA_MARKER));
    assert_eq!(document.front_matter().get("version"), Some(FORMAT_VERSION));
    assert!(document.is_empty());
    assert_eq!(document.facts(), Vec::<CodingFact>::new());
    assert_eq!(
        document.render(),
        "---\nschema: openjiuwen-memory-lite\nversion: 1\n---\n\n# Memory\n"
    );
}

#[test]
fn render_then_parse_preserves_every_fact() {
    let document = sample_document();
    let rendered = document.render();
    let parsed = MarkdownMemory::parse(&rendered)
        .unwrap()
        .with_scope(scope());
    assert_eq!(parsed.facts(), document.facts());
    assert_eq!(parsed.render(), rendered);
    let build = parsed.fact("build").unwrap();
    assert_eq!(build.description, "How to build");
    assert_eq!(build.tags, vec!["build".to_owned()]);
    assert_eq!(build.path.as_deref(), Some("docs/BUILD.md"));
    assert_eq!(build.updated_at, fixed_time());
}

#[test]
fn render_is_idempotent_under_parse() {
    let rendered = sample_document().render();
    let once = MarkdownMemory::parse(&rendered).unwrap().render();
    let twice = MarkdownMemory::parse(&once).unwrap().render();
    assert_eq!(rendered, once);
    assert_eq!(once, twice);
}

#[test]
fn sections_follow_the_canonical_order() {
    let rendered = sample_document().render();
    let user = rendered.find("## User").unwrap();
    let feedback = rendered.find("## Feedback").unwrap();
    let project = rendered.find("## Project").unwrap();
    assert!(user < feedback, "{rendered}");
    assert!(feedback < project, "{rendered}");
}

#[test]
fn a_blank_file_parses_to_an_empty_document() {
    for text in ["", "\n", "   \n\n\t\n"] {
        let document = MarkdownMemory::parse(text).unwrap();
        assert!(document.is_empty());
        assert_eq!(document.facts(), Vec::<CodingFact>::new());
        // Rendering an empty document still emits the canonical header.
        assert!(document.render().ends_with("# Memory\n"));
    }
}

#[test]
fn crlf_input_parses_and_renders_lf() {
    let text = "---\r\nschema: x\r\n---\r\n\r\n# Memory\r\n\r\n## Project\r\n\r\n\
                ### build\r\n\r\n- description: d\r\n- tags: [\"t\"]\r\n\r\nbody line\r\n";
    let document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
    assert_eq!(document.front_matter().get("schema"), Some("x"));
    let build = document.fact("build").unwrap();
    assert_eq!(build.content, "body line");
    assert!(!document.render().contains('\r'));
}

#[test]
fn unknown_front_matter_keys_and_raw_lines_are_preserved() {
    let text = "---\nschema: x\nnot a pair\ncustom: keep me\n---\n\n# Memory\n";
    let document = MarkdownMemory::parse(text).unwrap();
    assert_eq!(document.front_matter().get("custom"), Some("keep me"));
    let rendered = document.render();
    assert!(rendered.contains("not a pair"), "{rendered}");
    assert!(rendered.contains("custom: keep me"), "{rendered}");
    let again = MarkdownMemory::parse(&rendered).unwrap();
    assert_eq!(again.front_matter().get("custom"), Some("keep me"));
}

#[test]
fn unknown_sections_survive_a_round_trip() {
    let text = concat!(
        "# Memory\n\n",
        "## Project\n\n### build\n\n- description: d\n\nbody\n\n",
        "## Notes\n\nfree-form text\nwith two lines\n\n",
        "### not-a-fact\n\n- loose: yes\n",
    );
    let document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
    assert_eq!(document.facts().len(), 1);
    let rendered = document.render();
    assert!(rendered.contains("## Notes"), "{rendered}");
    assert!(
        rendered.contains("free-form text\nwith two lines"),
        "{rendered}"
    );
    assert!(rendered.contains("### not-a-fact"), "{rendered}");
    assert_eq!(MarkdownMemory::parse(&rendered).unwrap().render(), rendered);
}

#[test]
fn stray_text_inside_a_fact_section_is_kept() {
    let text = "# Memory\n\n## Project\n\nintro line\n\n### build\n\n- description: d\n\nbody\n";
    let document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
    assert_eq!(document.facts().len(), 1);
    let rendered = document.render();
    assert!(rendered.contains("intro line"), "{rendered}");
    assert_eq!(MarkdownMemory::parse(&rendered).unwrap().render(), rendered);
}

#[test]
fn an_unterminated_front_matter_is_a_validation_error() {
    let error = MarkdownMemory::parse("---\nschema: x\n").unwrap_err();
    assert!(matches!(
        error,
        openjiuwen_common::error::Error::Validation(_)
    ));
    assert!(error.to_string().contains("front matter"));
}

#[test]
fn facts_upsert_by_name_and_move_between_sections() {
    let mut document = MarkdownMemory::new().with_scope(scope());
    assert!(!document.upsert_fact(&fact("build", "first")));
    assert!(document.upsert_fact(&fact("build", "second")));
    assert_eq!(document.facts().len(), 1);
    assert_eq!(document.fact("build").unwrap().content, "second");

    let promoted = fact("build", "second").with_type(CodingFactType::Feedback);
    assert!(document.upsert_fact(&promoted));
    let rendered = document.render();
    assert!(rendered.contains("## Feedback"), "{rendered}");
    assert!(!rendered.contains("## Project"), "{rendered}");
    assert_eq!(
        document.fact("build").unwrap().fact_type,
        CodingFactType::Feedback
    );
}

#[test]
fn removing_a_fact_is_reported_and_drops_its_section() {
    let mut document = sample_document();
    assert!(document.remove("build"));
    assert!(!document.remove("build"));
    assert!(document.fact("build").is_none());
    assert_eq!(document.facts().len(), 2);
    assert!(!document.render().contains("### build"));
}

#[test]
fn the_scope_is_applied_to_every_fact() {
    let document = sample_document();
    for fact in document.facts() {
        assert_eq!(fact.scope, scope());
    }
    let text = document.render();
    let parsed = MarkdownMemory::parse(&text).unwrap().with_scope(scope());
    assert_eq!(parsed.facts()[0].scope, scope());
    assert_eq!(parsed.scope(), &scope());
}

#[test]
fn bodies_keep_heading_lookalikes_and_blank_lines() {
    let mut document = MarkdownMemory::new().with_scope(scope());
    let body = "intro\n\n## not a section\n\n### also not\n\ntail";
    let _ = document.upsert_fact(&fact("multi", body));
    let rendered = document.render();
    assert!(rendered.contains("\\## not a section"), "{rendered}");
    assert!(rendered.contains("\\### also not"), "{rendered}");
    let parsed = MarkdownMemory::parse(&rendered)
        .unwrap()
        .with_scope(scope());
    assert_eq!(parsed.fact("multi").unwrap().content, body);
    assert_eq!(parsed.facts().len(), 1);
}

#[test]
fn attribute_values_are_single_line() {
    let mut document = MarkdownMemory::new().with_scope(scope());
    let mut review = fact("review", "body").with_description("line one\nline two");
    review.path = Some("docs/\nREF.md".to_owned());
    let _ = document.upsert_fact(&review);
    let parsed = MarkdownMemory::parse(&document.render()).unwrap();
    let parsed = parsed.with_scope(scope());
    let restored = parsed.fact("review").unwrap();
    assert_eq!(restored.description, "line one line two");
    assert_eq!(restored.path.as_deref(), Some("docs/ REF.md"));
}

#[test]
fn tags_accept_json_and_comma_separated_forms() {
    let text = concat!(
        "# Memory\n\n## Project\n\n### a\n\n- description: d\n- tags: [\"one\", \"two\"]\n\nbody\n",
        "\n### b\n\n- description: d\n- tags: one, two\n\nbody\n",
        "\n### c\n\n- description: d\n- tags: not json, , three\n\nbody\n",
    );
    let document = MarkdownMemory::parse(text).unwrap();
    assert_eq!(document.fact("a").unwrap().tags, vec!["one", "two"]);
    assert_eq!(document.fact("b").unwrap().tags, vec!["one", "two"]);
    assert_eq!(document.fact("c").unwrap().tags, vec!["not json", "three"]);
}

#[test]
fn a_missing_or_bad_timestamp_decodes_to_the_epoch() {
    let text = "# Memory\n\n## Project\n\n### a\n\n- description: d\n\nbody\n\n\
                ### b\n\n- description: d\n- updated_at: yesterday\n\nbody\n";
    let document = MarkdownMemory::parse(text).unwrap();
    assert_eq!(
        document.fact("a").unwrap().updated_at,
        DateTime::<Utc>::default()
    );
    assert_eq!(
        document.fact("b").unwrap().updated_at,
        DateTime::<Utc>::default()
    );
}

#[test]
fn unknown_entry_attributes_survive_an_upsert() {
    let text = "# Memory\n\n## Project\n\n### build\n\n- description: d\n- owner: alice\n\nbody\n";
    let mut document = MarkdownMemory::parse(text).unwrap().with_scope(scope());
    let _ = document.upsert_fact(&fact("build", "new body").with_description("e"));
    let rendered = document.render();
    assert!(rendered.contains("- owner: alice"), "{rendered}");
    let parsed = MarkdownMemory::parse(&rendered).unwrap();
    assert_eq!(parsed.fact("build").unwrap().content, "new body");
}

#[test]
fn front_matter_is_ordered_and_keeps_positions() {
    let mut front_matter = FrontMatter::new();
    assert_eq!(front_matter.insert("b", "1"), None);
    assert_eq!(front_matter.insert("a", "2"), None);
    assert_eq!(front_matter.insert("b", "3"), Some("1".to_owned()));
    let keys: Vec<&str> = front_matter.iter().map(|(key, _)| key).collect();
    assert_eq!(keys, vec!["b", "a"]);
    assert_eq!(front_matter.get("b"), Some("3"));
    assert_eq!(front_matter.len(), 2);
    assert_eq!(front_matter.remove("b"), Some("3".to_owned()));
    assert!(!front_matter.is_empty());
    assert_eq!(front_matter.remove("missing"), None);
    assert!(FrontMatter::new().is_empty());
}

#[test]
fn with_front_matter_adds_a_key_without_reordering() {
    let document = MarkdownMemory::new()
        .with_scope(scope())
        .with_front_matter("owner", "alice");
    let rendered = document.render();
    let schema = rendered.find("schema:").unwrap();
    let owner = rendered.find("owner: alice").unwrap();
    assert!(schema < owner, "{rendered}");
    let parsed = MarkdownMemory::parse(&rendered).unwrap();
    assert_eq!(parsed.front_matter().get("owner"), Some("alice"));
}

#[test]
fn a_document_without_a_title_renders_no_title_line() {
    let document = MarkdownMemory::parse("## Project\n\n### a\n\n- description: d\n\nbody\n")
        .unwrap()
        .with_scope(scope());
    assert!(document.title().is_none());
    let rendered = document.render();
    assert!(!rendered.contains("# Memory"), "{rendered}");
    assert!(rendered.starts_with("## Project"), "{rendered}");
    assert_eq!(document.facts().len(), 1);
}

#[test]
fn empty_sections_render_nothing() {
    let mut document = MarkdownMemory::new().with_scope(scope());
    let _ = document.upsert_fact(&fact("build", "body"));
    let _ = document.remove("build");
    let rendered = document.render();
    assert!(!rendered.contains("## Project"), "{rendered}");
    assert!(rendered.ends_with("# Memory\n"), "{rendered}");
}

#[test]
fn a_fact_with_an_empty_body_round_trips() {
    let mut document = MarkdownMemory::new().with_scope(scope());
    let _ = document.upsert_fact(&fact("empty", ""));
    let rendered = document.render();
    let parsed = MarkdownMemory::parse(&rendered)
        .unwrap()
        .with_scope(scope());
    assert_eq!(parsed.fact("empty").unwrap().content, "");
}

#[test]
fn titles_can_be_replaced_without_touching_the_rest() {
    let document = MarkdownMemory::new()
        .with_scope(scope())
        .with_front_matter("owner", "alice")
        .with_title("Team Memory");
    assert_eq!(document.title(), Some("Team Memory"));
    let rendered = document.render();
    assert!(rendered.contains("\n# Team Memory\n"), "{rendered}");
    assert!(!rendered.contains("# Memory"), "{rendered}");
    assert_eq!(
        document.front_matter().get("schema"),
        Some(SCHEMA_MARKER),
        "the front matter is untouched"
    );
    assert_eq!(document.scope(), &scope(), "the scope is untouched");
    let reparsed = MarkdownMemory::parse(&rendered).unwrap();
    assert_eq!(reparsed.title(), Some("Team Memory"));

    let mut document = sample_document();
    document.set_title("Renamed");
    assert_eq!(document.title(), Some("Renamed"));
    assert_eq!(document.facts().len(), 3, "the facts are untouched");
    assert!(document.render().contains("# Renamed\n"));

    let mut document = sample_document();
    document.set_title("");
    assert_eq!(document.title(), Some(""));
    let rendered = document.render();
    assert!(
        !rendered.lines().any(|line| line.starts_with("# ")),
        "an empty title renders no title line: {rendered}"
    );
    assert_eq!(document.facts().len(), 3, "the facts are untouched");
}

#[test]
fn parsing_never_panics_on_hostile_input() {
    for text in [
        "---",
        "---\n---",
        "#",
        "##",
        "###",
        "### \n\n- : \n\n",
        "## Project\n### ###\n- description\n",
        "\u{feff}# Memory\n",
    ] {
        let document = MarkdownMemory::parse(text).unwrap_or_else(|error| {
            panic!("unexpected error for {text:?}: {error}");
        });
        let _rendered = document.render();
    }
}
