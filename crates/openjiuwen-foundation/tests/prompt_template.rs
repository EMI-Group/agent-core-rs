#![allow(clippy::unwrap_used, clippy::expect_used)]
// ports tests/unit_tests/core/foundation/prompt/ behaviors

//! End-to-end rendering matrix for [`PromptTemplate`].
//!
//! Every assertion goes through the public module paths, so these tests pin the
//! contract other crates rely on: the placeholder grammar (delimiters, trimming,
//! literals), dotted-path resolution, non-string value rendering, single-pass
//! substitution, message-list rendering that leaves non-text parts alone, and the
//! YAML/JSON round trips.

use std::collections::BTreeMap;
use std::path::PathBuf;

use openjiuwen_foundation::prompt::content::{ContentPart, ImageUrl, MessageContent};
use openjiuwen_foundation::prompt::message::BaseMessage;
use openjiuwen_foundation::prompt::role::Role;
use openjiuwen_foundation::prompt::template::{PromptTemplate, TemplateContent};
use openjiuwen_foundation::prompt::tool_call::ToolCall;
use serde_json::{json, Value};

/// Builds a values map from `(key, value)` pairs.
fn values(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// The rendered text of a text-only template.
fn text_of(template: &PromptTemplate) -> &str {
    match &template.content {
        TemplateContent::Text(text) => text,
        TemplateContent::Messages(_) => panic!("expected text content, found a message list"),
    }
}

/// An image content part at `url`.
fn image(url: &str) -> ContentPart {
    ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: url.to_owned(),
            detail: None,
        },
    }
}

/// A path under the system temp directory that no other test uses.
fn unique_temp_path(stem: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{stem}-{}-{nanos}.yaml", std::process::id()))
}

#[test]
fn complete_fill_and_two_stage_partial_fill() {
    let template = PromptTemplate::new("Hello {{name}}, you are {{age}} years old.");

    // Stage 1: only `name` is known; the unknown slot survives verbatim.
    let partial = template.format(&values(&[("name", json!("Bob"))]));
    assert_eq!(text_of(&partial), "Hello Bob, you are {{age}} years old.");

    // Stage 2: filling the rest from the partially formatted template completes it.
    let complete = partial.format(&values(&[("age", json!(42))]));
    assert_eq!(text_of(&complete), "Hello Bob, you are 42 years old.");

    // Filling everything in one call gives the same result.
    let all_at_once = template.format(&values(&[("name", json!("Bob")), ("age", json!(42))]));
    assert_eq!(all_at_once, complete);

    // The original template is untouched by either pass.
    assert_eq!(
        text_of(&template),
        "Hello {{name}}, you are {{age}} years old."
    );
}

#[test]
fn extra_keys_are_ignored() {
    let template = PromptTemplate::new("only {{a}}");
    let formatted = template.format(&values(&[
        ("a", json!(1)),
        ("b", json!(2)),
        ("c", json!(3)),
    ]));
    assert_eq!(text_of(&formatted), "only 1");
}

#[test]
fn custom_delimiters_angle_and_shell_style() {
    let angle = PromptTemplate::new("hi <<name>>, bye <<name>>").with_placeholders("<<", ">>");
    assert_eq!(
        text_of(&angle.format(&values(&[("name", json!("bob"))]))),
        "hi bob, bye bob"
    );
    assert_eq!(angle.placeholders(), vec!["name"]);

    let shell = PromptTemplate::new("${HOME}$/bin: ${USER}$").with_placeholders("${", "}$");
    assert_eq!(
        text_of(&shell.format(&values(&[("HOME", json!("/root")), ("USER", json!("bob"))]))),
        "/root/bin: bob"
    );
    assert_eq!(shell.input_keys(), vec!["HOME", "USER"]);
    // The default `{{ }}` pair is inert under custom delimiters.
    assert_eq!(
        text_of(&shell.format(&values(&[("{{HOME}}", json!("x"))]))),
        "${HOME}$/bin: ${USER}$"
    );
}

#[test]
fn whitespace_inside_delimiters_is_trimmed() {
    let template = PromptTemplate::new("|{{ name }}|{{\tother\n}}|");
    assert_eq!(template.placeholders(), vec!["name", "other"]);
    assert_eq!(template.input_keys(), vec!["name", "other"]);
    assert_eq!(
        text_of(&template.format(&values(&[("name", json!("bob")), ("other", json!(1))]))),
        "|bob|1|"
    );
    // The trimmed name carries no spaces, so a padded map key misses.
    assert_eq!(
        text_of(&template.format(&values(&[(" name ", json!("bob"))]))),
        "|{{ name }}|{{\tother\n}}|"
    );
}

#[test]
fn empty_and_single_brace_placeholders_stay_literal() {
    let empty = PromptTemplate::new("a{{}}b{{ }}c");
    assert_eq!(empty.placeholders(), Vec::<String>::new());
    assert_eq!(
        text_of(&empty.format(&values(&[("", json!("x"))]))),
        "a{{}}b{{ }}c"
    );

    let single = PromptTemplate::new("{name} and {a.b}");
    assert_eq!(single.placeholders(), Vec::<String>::new());
    assert_eq!(
        text_of(&single.format(&values(&[("name", json!("bob")), ("a", json!({"b": 1}))]))),
        "{name} and {a.b}"
    );
}

#[test]
fn dotted_paths_resolve_through_nested_objects() {
    let template = PromptTemplate::new("{{user.name}} ({{user.address.city}})");
    assert_eq!(
        template.placeholders(),
        vec!["user.name", "user.address.city"]
    );
    assert_eq!(template.input_keys(), vec!["user"]);

    let filled = template.format(&values(&[(
        "user",
        json!({"name": "bob", "address": {"city": "Paris"}}),
    )]));
    assert_eq!(text_of(&filled), "bob (Paris)");

    // A missing leaf keeps the placeholder.
    let missing_leaf = template.format(&values(&[("user", json!({"name": "bob", "address": {}}))]));
    assert_eq!(text_of(&missing_leaf), "bob ({{user.address.city}})");

    // A missing intermediate keeps the placeholder.
    let missing_intermediate = template.format(&values(&[("user", json!({"name": "bob"}))]));
    assert_eq!(
        text_of(&missing_intermediate),
        "bob ({{user.address.city}})"
    );

    // A missing head keeps the placeholder.
    let missing_head = template.format(&values(&[("other", json!({"name": "bob"}))]));
    assert_eq!(
        text_of(&missing_head),
        "{{user.name}} ({{user.address.city}})"
    );
}

#[test]
fn null_leaves_keep_the_placeholder_and_never_render_null() {
    let template = PromptTemplate::new("[{{a}}][{{user.name}}]");
    let filled = template.format(&values(&[
        ("a", Value::Null),
        ("user", json!({"name": null})),
    ]));
    assert_eq!(text_of(&filled), "[{{a}}][{{user.name}}]");
    assert!(!text_of(&filled).contains("null"));
}

#[test]
fn inserted_values_are_not_re_expanded() {
    let template = PromptTemplate::new("{{a}} and {{b}}");
    // `a` looks like a placeholder but is emitted literally; `b` still resolves.
    let filled = template.format(&values(&[("a", json!("{{b}}")), ("b", json!("BOOM"))]));
    assert_eq!(text_of(&filled), "{{b}} and BOOM");

    let nested = PromptTemplate::new("{{a}}").format_one("a", "{{a}}");
    assert_eq!(text_of(&nested), "{{a}}");
}

#[test]
fn repeated_slots_and_non_string_values_render_as_documented() {
    let template = PromptTemplate::new("{{x}}-{{x}}-{{x}}");
    assert_eq!(
        text_of(&template.format(&values(&[("x", json!("a"))]))),
        "a-a-a"
    );

    let typed = PromptTemplate::new("{{n}}|{{f}}|{{t}}|{{obj}}|{{arr}}|{{empty}}|{{str}}");
    let filled = typed.format(&values(&[
        ("n", json!(42)),
        ("f", json!(-1.5)),
        ("t", json!(true)),
        ("obj", json!({"k": 1, "s": "v"})),
        ("arr", json!([1, "two"])),
        ("empty", json!("")),
        ("str", json!("raw \"quoted\" text")),
    ]));
    assert_eq!(
        text_of(&filled),
        r#"42|-1.5|true|{"k":1,"s":"v"}|[1,"two"]||raw "quoted" text"#
    );
}

#[test]
fn message_list_rendering_touches_only_text_parts() {
    let call = ToolCall::new("get_weather")
        .with_id("call_1")
        .with_arguments(json!({"city": "Paris"}));
    let original = vec![
        BaseMessage::system("You are {{persona}}."),
        BaseMessage::user(vec![
            ContentPart::Text {
                text: "Describe {{topic}}".to_owned(),
            },
            image("https://example.test/{{topic}}.png"),
            ContentPart::Other(json!({"type": "video", "url": "{{topic}}", "n": 1})),
        ]),
        BaseMessage::assistant("")
            .with_tool_calls(vec![call.clone()])
            .with_reasoning("thinking about {{topic}}"),
        BaseMessage::tool("call_1", "sunny {{topic}}")
            .with_name("get_weather")
            .with_metadata("trace", json!("{{topic}}")),
    ];
    let template = PromptTemplate::named("agent", original.clone());
    assert_eq!(
        template.placeholders(),
        vec!["persona", "topic"],
        "image, video, metadata, and reasoning payloads are not scanned"
    );

    let filled = template.format(&values(&[
        ("persona", json!("a pirate")),
        ("topic", json!("weather")),
    ]));
    let TemplateContent::Messages(messages) = &filled.content else {
        panic!("expected message content");
    };
    assert_eq!(messages.len(), 4);

    // Role and text are rendered.
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), "You are a pirate.");
    assert_eq!(messages[1].role, Role::User);

    // Only the text part changes; the image and vendor blocks are identical.
    assert_eq!(
        messages[1].content,
        MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Describe weather".to_owned(),
            },
            image("https://example.test/{{topic}}.png"),
            ContentPart::Other(json!({"type": "video", "url": "{{topic}}", "n": 1})),
        ])
    );

    // Assistant tool calls and reasoning are preserved verbatim.
    assert_eq!(messages[2].role, Role::Assistant);
    assert_eq!(messages[2].tool_calls, vec![call]);
    assert_eq!(
        messages[2].reasoning_content.as_deref(),
        Some("thinking about {{topic}}")
    );

    // Tool messages keep the call id, the name, and their metadata.
    assert_eq!(messages[3].role, Role::Tool);
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(messages[3].name.as_deref(), Some("get_weather"));
    assert_eq!(messages[3].metadata.get("trace"), Some(&json!("{{topic}}")));
    assert_eq!(messages[3].text(), "sunny weather");

    // Name and delimiters survive rendering; the original template is unmodified.
    assert_eq!(filled.name, "agent");
    assert_eq!(template.content, TemplateContent::Messages(original));
}

#[test]
fn to_messages_covers_text_list_and_empty_content() {
    // Text content becomes one user message.
    let text = PromptTemplate::new("hello {{name}}");
    let messages = text.to_messages();
    assert_eq!(messages, vec![BaseMessage::user("hello {{name}}")]);
    assert_eq!(messages[0].role, Role::User);

    // Empty text yields no messages.
    assert_eq!(
        PromptTemplate::new("").to_messages(),
        Vec::<BaseMessage>::new()
    );
    assert_eq!(
        PromptTemplate::default().to_messages(),
        Vec::<BaseMessage>::new()
    );

    // A message list is deep-copied.
    let original = vec![BaseMessage::system("s"), BaseMessage::tool("c", "t")];
    let template = PromptTemplate::from_messages(original.clone());
    let mut copy = template.to_messages();
    assert_eq!(copy, original);
    copy[0] = BaseMessage::system("mutated");
    copy[1].content = MessageContent::Text("mutated".to_owned());
    assert_eq!(
        template.to_messages(),
        original,
        "mutating the result must not reach the template"
    );
}

#[test]
fn yaml_parses_mappings_and_loads_files() {
    let yaml = "name: greet\n\
                content: \"hi <<name>>\"\n\
                placeholder_prefix: \"<<\"\n\
                placeholder_suffix: \">>\"\n";
    let template = PromptTemplate::from_yaml_str(yaml).unwrap();
    assert_eq!(template.name, "greet");
    assert_eq!(template.placeholder_prefix, "<<");
    assert_eq!(template.placeholder_suffix, ">>");
    assert_eq!(text_of(&template), "hi <<name>>");
    assert_eq!(
        text_of(&template.format_one("name", "bob")),
        "hi bob",
        "parsed delimiters are used for rendering"
    );

    // A bare string document is treated as the content of a default template.
    let bare = PromptTemplate::from_yaml_str("hello {{x}}").unwrap();
    assert_eq!(bare, PromptTemplate::new("hello {{x}}"));

    // `load_yaml` reads the same document from disk.
    let path = unique_temp_path("prompt-template");
    std::fs::write(&path, yaml).unwrap();
    let loaded = PromptTemplate::load_yaml(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded, template);
    assert!(!path.exists());
}

#[test]
fn serde_round_trip_keeps_the_python_field_names() {
    let text = PromptTemplate::named("greet", "hi {{name}}");
    let value: Value = serde_json::to_value(&text).unwrap();
    let object = value.as_object().unwrap();
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "content",
            "name",
            "placeholder_prefix",
            "placeholder_suffix"
        ]
    );
    assert_eq!(value["content"], json!("hi {{name}}"));
    assert_eq!(
        serde_json::from_value::<PromptTemplate>(value).unwrap(),
        text
    );

    let list = PromptTemplate::from_messages(vec![
        BaseMessage::system("be {{style}}"),
        BaseMessage::user("hi"),
    ]);
    let value: Value = serde_json::to_value(&list).unwrap();
    assert!(value["content"].is_array());
    assert_eq!(value["content"][0]["role"], json!("system"));
    assert_eq!(
        serde_json::from_value::<PromptTemplate>(value).unwrap(),
        list
    );

    // Missing fields fall back to the braces-and-empty defaults.
    let parsed: PromptTemplate = serde_json::from_value(json!({"content": "x"})).unwrap();
    assert_eq!(parsed, PromptTemplate::new("x"));
    assert_eq!(parsed.placeholder_prefix, "{{");
    assert_eq!(parsed.placeholder_suffix, "}}");
}
