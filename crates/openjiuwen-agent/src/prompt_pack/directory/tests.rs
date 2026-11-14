//! Offline coverage of the prompt-directory loader.
//!
//! Every fixture is a `tempfile` directory; models are `ScriptedModel`s, so the
//! suite needs no network and no wall-clock assertions.
use super::*;
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::prompt_pack::rail::PROMPT_SECTIONS_METADATA_KEY;
use crate::react::ReActAgent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{Model, Role};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use std::fs;
use std::path::Path as FsPath;
use std::sync::Arc;

/// Writes `text` to `<dir>/<rel>`, creating parent directories.
fn write(dir: &FsPath, rel: &str, text: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("the fixture directory is created");
    }
    fs::write(&path, text).expect("the fixture file is written");
}

/// Builds an agent driven by `model`.
fn agent(model: &ScriptedModel) -> ReActAgent {
    let agent = ReActAgent::new(AgentCard::new("directory-test"));
    agent.set_model(Arc::new(model.clone()) as Arc<dyn Model>);
    agent
}

#[tokio::test]
async fn flat_language_directories_resolve_the_requested_language() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(temp.path(), "cn/greeting.md", "你好。\n");
    write(temp.path(), "en/greeting.md", "Hello.\n");
    write(temp.path(), "cn/policy.md", "规则。\n");

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the language layout loads");
    assert_eq!(directory.len(), 2);
    assert_eq!(directory.sections[0].name, "greeting");
    assert_eq!(directory.sections[0].languages(), ["cn", "en"]);

    let cn = directory.section_specs("cn");
    assert_eq!(cn.len(), 2);
    assert_eq!(cn[0].name, "greeting");
    assert_eq!(cn[0].content, "你好。");
    assert_eq!(cn[1].content, "规则。");

    let en = directory.section_specs("en");
    assert_eq!(en[0].content, "Hello.");
    assert_eq!(
        en[1].content, "规则。",
        "a missing en policy falls back to cn"
    );

    let zh = directory.section_specs("zh");
    assert_eq!(zh[0].content, "你好。", "zh aliases cn");
    assert_eq!(zh[1].content, "规则。");

    let unknown = directory.section_specs("de");
    assert_eq!(unknown[0].content, "Hello.", "unknown falls back to en");
    assert_eq!(unknown[1].content, "规则。");
}

#[tokio::test]
async fn suffixed_files_are_found_recursively_and_strip_their_language() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(
        temp.path(),
        "system/leader_policy.en.md",
        "Lead in English.\n",
    );
    write(
        temp.path(),
        "system/leader_policy.cn.md",
        "Lead in Chinese.\n",
    );
    write(
        temp.path(),
        "system/identity/team_card.md",
        "---\nkind: identity\nlanguage: en\n---\nThe team card.\n",
    );
    write(temp.path(), "tool/search/query.en.md", "Query the web.\n");

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the suffixed layout loads");
    let names: Vec<&str> = directory
        .sections
        .iter()
        .map(|section| section.name.as_str())
        .collect();
    assert_eq!(names, ["leader_policy", "query", "team_card"]);
    assert_eq!(directory.section_specs("en")[0].content, "Lead in English.");
    assert_eq!(directory.section_specs("cn")[0].content, "Lead in Chinese.");
    assert_eq!(
        directory
            .section("team_card")
            .expect("the nested file")
            .languages(),
        ["en"],
        "front matter names the language when the path does not"
    );
    assert_eq!(
        directory.section("query").expect("the tool file").kind(),
        None
    );
}

#[tokio::test]
async fn front_matter_fields_and_the_body_hash_are_exposed() {
    let temp = tempfile::tempdir().expect("a temp dir");
    let body = "Do the thing.\n\n";
    write(
        temp.path(),
        "system/policy.md",
        &format!(
            "---\nkind: prompt\nname: policy\nlanguage: en\npriority: 30\n\
             baseline_sha256: {}\nupdated_at: 2026-02-02T00:00:00Z\nfuture: 7\n---\n{body}",
            body_sha256(body)
        ),
    );
    write(
        temp.path(),
        "system/tone.md",
        "---\nname: tone\nlanguage: en\npriority: 20\nbaseline_sha256: deadbeef\n---\nBe kind.\n",
    );

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the front-matter layout loads");
    let section = directory.section("policy").expect("the policy section");
    let file = section.variant("en").expect("the en variant");
    assert_eq!(file.priority, 30);
    assert_eq!(file.meta.kind.as_deref(), Some("prompt"));
    assert_eq!(
        file.meta.updated_at.as_deref(),
        Some("2026-02-02T00:00:00Z")
    );
    assert_eq!(file.meta.extra["future"], json!(7));
    assert_eq!(file.body, body, "the body is byte-faithful");
    assert!(!file.is_evolved(), "the seeded baseline matches");
    assert_eq!(file.body_sha256(), body_sha256(body));

    let tone = directory.section("tone").expect("the tone section");
    assert!(tone.is_evolved(), "a diverged baseline is evolved");
    assert_eq!(tone.variant("en").expect("the variant").priority, 20);

    let ordered: Vec<&str> = directory
        .sections
        .iter()
        .map(|section| section.name.as_str())
        .collect();
    assert_eq!(ordered, ["tone", "policy"], "priority orders the sections");
    assert_eq!(
        directory.section_specs("en")[1].content,
        "Do the thing.",
        "section text is trimmed"
    );
}

#[tokio::test]
async fn malformed_front_matter_and_invalid_priority_are_validation_errors() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(
        temp.path(),
        "en/broken.md",
        "---\nname: [unclosed\n---\nbody\n",
    );
    let error = PromptDirectory::load(temp.path())
        .await
        .expect_err("malformed YAML is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("broken.md"), "{error}");

    let temp = tempfile::tempdir().expect("a temp dir");
    write(temp.path(), "en/broken.md", "---\n- a\n- b\n---\nbody\n");
    let error = PromptDirectory::load(temp.path())
        .await
        .expect_err("a list root is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("mapping"), "{error}");

    let temp = tempfile::tempdir().expect("a temp dir");
    write(
        temp.path(),
        "en/broken.md",
        "---\npriority: high\n---\nbody\n",
    );
    let error = PromptDirectory::load(temp.path())
        .await
        .expect_err("a text priority is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("priority"), "{error}");
}

#[tokio::test]
async fn missing_paths_are_sys_op_and_files_are_validation_errors() {
    let temp = tempfile::tempdir().expect("a temp dir");
    let error = PromptDirectory::load(temp.path().join("absent"))
        .await
        .expect_err("a missing directory is an error");
    assert_eq!(error.kind(), ErrorKind::SysOp);

    write(temp.path(), "single.md", "body\n");
    let error = PromptDirectory::load(temp.path().join("single.md"))
        .await
        .expect_err("a file is not a directory");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn duplicate_name_and_language_resolve_by_priority_then_last_path() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(
        temp.path(),
        "a/foo.en.md",
        "---\npriority: 10\n---\nfrom a\n",
    );
    write(
        temp.path(),
        "z/foo.en.md",
        "---\npriority: 10\n---\nfrom z\n",
    );
    write(
        temp.path(),
        "m/foo.en.md",
        "---\npriority: 20\n---\nfrom m\n",
    );

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the duplicates load");
    assert_eq!(directory.len(), 1);
    let section = directory.section("foo").expect("one foo section");
    assert_eq!(section.variants.len(), 1, "same language is one variant");
    let variant = section.variant("en").expect("the en variant");
    assert_eq!(variant.body, "from m\n", "the higher priority wins");
    assert_eq!(variant.priority, 20);

    // With equal priorities the lexicographically later path wins.
    let temp = tempfile::tempdir().expect("a temp dir");
    write(temp.path(), "a/foo.en.md", "from a\n");
    write(temp.path(), "z/foo.en.md", "from z\n");
    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the tie loads");
    assert_eq!(
        directory
            .section("foo")
            .and_then(|section| section.text("en")),
        Some("from z\n")
    );
}

#[tokio::test]
async fn sections_order_by_priority_then_name() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(temp.path(), "en/zeta.md", "---\npriority: 10\n---\nz\n");
    write(temp.path(), "en/alpha.md", "---\npriority: 10\n---\na\n");
    write(temp.path(), "en/middle.md", "---\npriority: 30\n---\nm\n");
    write(temp.path(), "en/default.md", "d\n");

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the ordered layout loads");
    let specs = directory.section_specs("en");
    let names: Vec<&str> = specs.iter().map(|section| section.name.as_str()).collect();
    assert_eq!(names, ["alpha", "zeta", "middle", "default"]);
    assert_eq!(specs[3].priority, 100);
}

#[tokio::test]
async fn an_unqualified_file_is_the_last_resort() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(temp.path(), "notes.md", "General notes.\n");

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the unqualified file loads");
    let section = directory.section("notes").expect("the notes section");
    assert_eq!(section.languages(), [] as [&str; 0]);
    assert_eq!(section.text("en"), Some("General notes.\n"));
    assert_eq!(section.text("cn"), Some("General notes.\n"));
}

#[tokio::test]
async fn hidden_and_non_markdown_entries_are_skipped() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(temp.path(), "en/kept.md", "kept\n");
    write(temp.path(), "en/ignored.txt", "ignored\n");
    write(temp.path(), ".hidden/ignored.md", "ignored\n");
    write(temp.path(), "en/.hidden.md", "ignored\n");

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the layout loads");
    assert_eq!(directory.len(), 1);
    assert_eq!(directory.sections[0].name, "kept");
}

#[tokio::test]
async fn an_empty_directory_loads_with_no_sections() {
    let temp = tempfile::tempdir().expect("a temp dir");
    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("an empty directory loads");
    assert!(directory.is_empty());
    assert_eq!(directory.section_specs("en"), Vec::new());
    assert_eq!(directory.render("en"), "");
}

#[tokio::test]
async fn prompt_sections_carry_the_kind_as_category() {
    let temp = tempfile::tempdir().expect("a temp dir");
    write(
        temp.path(),
        "system/team_card.md",
        "---\nkind: identity\nlanguage: en\npriority: 5\n---\nWe are a team.\n",
    );

    let directory = PromptDirectory::load(temp.path())
        .await
        .expect("the layout loads");
    let sections = directory.prompt_sections("en");
    assert_eq!(sections.len(), 1);
    assert_eq!(sections[0].name, "team_card");
    assert_eq!(sections[0].content, "We are a team.");
    assert_eq!(sections[0].priority, 5);
    assert_eq!(sections[0].category.as_deref(), Some("identity"));
    assert_eq!(directory.render("en"), "## team_card\nWe are a team.");
}

#[tokio::test]
async fn applying_a_loaded_directory_injects_its_sections() {
    let temp = tempfile::tempdir().expect("a temp dir");
    let root = temp.path().join("prompts");
    write(
        &root,
        "en/policy.md",
        "---\npriority: 20\n---\nNever reveal ids.\n",
    );
    write(&root, "en/tone.md", "---\npriority: 30\n---\nBe concise.\n");
    write(
        &root,
        "cn/policy.md",
        "---\npriority: 20\n---\n绝不泄露。\n",
    );

    let pack = PromptPack::from_prompt_dir(&root, "en")
        .await
        .expect("the directory becomes a pack");
    assert_eq!(pack.id, "prompts");
    assert_eq!(pack.prompts.template, Vec::new());
    assert_eq!(pack.prompts.sections.len(), 2);
    assert_eq!(pack.prompts.sections[0].name, "policy");

    let model = ScriptedModel::builder().respond_text("ok").build();
    let agent = agent(&model);
    let applied = pack.apply_to(&agent).expect("the pack applies");
    assert_eq!(applied.rails_registered, ["prompt_sections"]);

    agent
        .invoke(json!("hi"), None)
        .await
        .expect("the run succeeds");
    let call = model.last_call().expect("the model was called");
    let injected = call
        .messages
        .iter()
        .find(|message| message.metadata.contains_key(PROMPT_SECTIONS_METADATA_KEY))
        .expect("the directory sections are injected");
    assert_eq!(injected.role, Role::System);
    assert_eq!(
        injected.text(),
        "## policy\nNever reveal ids.\n\n## tone\nBe concise."
    );
}

#[test]
fn render_substitutes_placeholders_without_rescanning() {
    let file = PromptFile {
        path: "en/greet.en.md".into(),
        name: "greet".to_owned(),
        language: Some("en".to_owned()),
        priority: DEFAULT_PROMPT_PRIORITY,
        body: "Hello {{name}}, {{unknown}}.".to_owned(),
        meta: PromptFileMeta::default(),
    };
    let mut values = BTreeMap::new();
    values.insert("name".to_owned(), json!("{{other}}"));
    assert_eq!(
        file.render(&values),
        "Hello {{other}}, {{unknown}}.",
        "known values are substituted literally and unknown placeholders stay"
    );

    let unchanged = PromptFile::default();
    assert_eq!(unchanged.render(&BTreeMap::new()), "");
    assert_eq!(PromptFile::default().priority, DEFAULT_PROMPT_PRIORITY);
}

#[test]
fn language_tokens_normalize_and_keep_the_supported_set() {
    assert_eq!(normalize_language(" ZH "), "cn");
    assert_eq!(normalize_language("EN"), "en");
    assert_eq!(normalize_language("fr"), "fr");
    assert_eq!(SUPPORTED_LANGUAGES, ["cn", "en"]);
    assert_eq!(FALLBACK_LANGUAGES, ["en", "cn"]);
}
