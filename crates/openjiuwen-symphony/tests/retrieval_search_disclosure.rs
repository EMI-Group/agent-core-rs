//! Behavior tests for the disclosure rendering layer of
//! `openjiuwen_symphony::retrieval::search::render`: subtree exposure, code assignment,
//! the rendered candidate block, the prompt bank and the reply-parsing cascade.
//!
//! Everything here goes through the crate's public API only, with hand-built
//! [`RetrieverNode`] fixtures (no index build, no temp directory, no model). Every assertion
//! is deterministic: identical inputs must produce byte-identical fragments, prompts and
//! fingerprints, and no assertion depends on the wall clock, the network or hash-map
//! iteration order.
//!
//! The unit-level coverage (per-helper edge cases) lives beside the modules in
//! `src/retrieval/search/render/{expose,prompt,parse}.rs`; this file pins the end-to-end
//! shapes a caller of the renderer depends on.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp, clippy::assert_is_empty)]
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::retrieval::search::flow::{
    ChatMessage, CurrentSubtree, SelectableTarget, SelectionProtocol,
};
use openjiuwen_symphony::retrieval::search::{
    build_disclosure_messages, build_disclosure_prompt_parts, build_exposed_fragment,
    disclosure_system_prompt, exposure_item_count, fragmentation_fingerprint, parse_selected_codes,
    DefaultSubtreeRenderer, DisclosureConfig, SubtreeRenderer,
};
use openjiuwen_symphony::retrieval::types::{RetrieverItem, RetrieverNode, ROOT_NODE_ID};

/// A small, realistic catalogue subtree:
///
/// ```text
/// ROOT 能力树
/// ├── office-docs 办公文档   (摘要, 翻译)
/// └── weather 天气           (天气查询)
/// ```
fn catalogue() -> RetrieverNode {
    RetrieverNode::new(ROOT_NODE_ID, "能力树").with_children(vec![
        RetrieverNode::new("office-docs", "办公文档")
            .with_description("Office documents")
            .with_items(vec![
                RetrieverItem::new("skills.summary", "skills.summary")
                    .with_label("摘要")
                    .with_description("把长文档压缩成三点"),
                RetrieverItem::new("skills.translate", "skills.translate")
                    .with_label("翻译")
                    .with_description("在中文和英文之间翻译"),
            ]),
        RetrieverNode::new("weather", "天气").with_items(vec![RetrieverItem::new(
            "skills.weather",
            "skills.weather",
        )
        .with_label("天气查询")
        .with_description("查询实时天气")]),
    ])
}

fn branch_path() -> Vec<String> {
    vec![ROOT_NODE_ID.to_owned()]
}

fn compact_config(count: usize) -> DisclosureConfig {
    let book = ["AA", "AB", "AC", "AD", "AE"]
        .iter()
        .take(count)
        .map(|code| (*code).to_owned())
        .collect();
    DisclosureConfig::default()
        .with_compact_codes_enabled(true)
        .with_compact_codebook(book)
}

fn build(config: &DisclosureConfig) -> openjiuwen_symphony::retrieval::search::ExposedFragment {
    build_exposed_fragment(&catalogue(), &branch_path(), config, &exposure_item_count).unwrap()
}

#[test]
fn a_nested_tree_exposes_in_order_with_pinned_codes_and_text() {
    let fragment = build(&DisclosureConfig::default());
    assert_eq!(
        fragment.candidate_codes,
        vec![
            "Summary".to_owned(),
            "Translate".to_owned(),
            "Weather".to_owned()
        ]
    );
    assert_eq!(
        fragment.rendered_tree,
        "- Category 能力树\n  \
         - Category 办公文档: Office documents\n    \
         - Candidate Summary: 把长文档压缩成三点\n    \
         - Candidate Translate: 在中文和英文之间翻译\n  \
         - Category 天气\n    \
         - Candidate Weather: 查询实时天气"
    );
    // Mixed code widths leave the width at 0.
    assert_eq!(fragment.code_width, 0);
    assert!(!fragment.compact_codes_enabled);
    assert!(!fragment.flat_list_mode);
    assert!(!fragment.truncated);
    assert_eq!(
        fragment.rendered_chars,
        fragment.rendered_tree.chars().count()
    );

    let summary = fragment.resolution("Summary").unwrap();
    assert_eq!(summary.canonical_id, "skills.summary");
    assert!(summary.is_terminal);
    assert_eq!(summary.label, "摘要");
    assert_eq!(
        summary.branch_path,
        vec![ROOT_NODE_ID.to_owned(), "office-docs".to_owned()]
    );
    assert_eq!(summary.node_id, None);
    assert_eq!(
        summary.item.as_ref().map(|item| item.item_id.as_str()),
        Some("skills.summary")
    );

    // The exposed view mirrors the rendered text: categories are not selectable.
    assert!(!fragment.root.is_selectable);
    assert_eq!(fragment.root.children.len(), 2);
    assert!(!fragment.root.children[0].is_selectable);
    assert!(fragment.root.children[0].children[0].is_selectable);
    assert_eq!(fragment.selectable_count(), 3);
}

#[test]
fn a_compact_code_book_is_positional_and_a_short_book_is_a_validation_error() {
    let compact = build(&compact_config(3));
    assert_eq!(
        compact.candidate_codes,
        vec!["AA".to_owned(), "AB".to_owned(), "AC".to_owned()]
    );
    assert_eq!(compact.code_width, 2);
    assert!(compact.compact_codes_enabled);
    assert!(compact
        .rendered_tree
        .contains("- Summary [id: AA]: 把长文档压缩成三点"));
    assert!(compact
        .rendered_tree
        .contains("- Weather [id: AC]: 查询实时天气"));

    let error = build_exposed_fragment(
        &catalogue(),
        &branch_path(),
        &compact_config(2),
        &exposure_item_count,
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("provides 2 codes, but 3 selectable"),
        "{}",
        error.message()
    );

    // Without a code book the positional numbers are the codes.
    let numeric = build(&DisclosureConfig::default().with_compact_codes_enabled(true));
    assert_eq!(
        numeric.candidate_codes,
        vec!["1".to_owned(), "2".to_owned(), "3".to_owned()]
    );
    assert_eq!(numeric.code_width, 1);
}

#[test]
fn five_reply_shapes_recover_the_same_selection() {
    let fragment = build(&compact_config(3));
    let expected = vec!["skills.summary".to_owned(), "skills.weather".to_owned()]
        .into_iter()
        .map(|canonical| {
            fragment
                .resolutions()
                .into_iter()
                .find(|resolution| resolution.canonical_id == canonical)
                .cloned()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let replies = [
        // 1. A bare code list, one per line.
        "AA\nAC",
        // 2. Copied candidate lines.
        "- Summary [id: AA]: 把长文档压缩成三点\n- Weather [id: AC]: 查询实时天气",
        // 3. JSON objects.
        "{\"id\": \"AA\", \"why\": \"summary\"}\n{\"id\": \"AC\"}",
        // 4. `[id: …]` handles without the surrounding line.
        "[id: AA]\n[id: AC]",
        // 5. A comma-separated list.
        "AA, AC",
    ];
    for reply in replies {
        let parsed = parse_selected_codes(&fragment, reply);
        assert_eq!(parsed, expected, "reply: {reply:?}");
    }

    // The names-mode fragment answers to names, labels, positions and tokens too.
    let named = build(&DisclosureConfig::default());
    for reply in [
        "Summary\nWeather",
        "summary\nWEATHER",
        "摘要\n天气查询",
        "1\n3",
        "Summary, Weather",
    ] {
        let parsed = parse_selected_codes(&named, reply);
        assert_eq!(
            parsed
                .iter()
                .map(|resolution| resolution.canonical_id.as_str())
                .collect::<Vec<&str>>(),
            vec!["skills.summary", "skills.weather"],
            "reply: {reply:?}"
        );
    }

    // Garbage never panics and never invents a selection.
    for garbage in ["", "   ", "no idea", "{\"id\": \"ZZ\"}", "!!"] {
        assert!(parse_selected_codes(&fragment, garbage).is_empty());
    }
}

#[test]
fn the_prompt_has_two_messages_and_names_every_candidate_once() {
    let fragment = build(&compact_config(3));
    let messages = build_disclosure_messages(
        &fragment,
        &[ChatMessage::user(
            "From user: 帮我总结这份文档
",
        )],
        Some(2),
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].role, "user");
    // The `top_k == 2` preamble opens with the goal block; the general one adds the
    // `/no_think` directive before it.
    assert!(messages[0].content.starts_with("\n# 背景与目标："));
    assert!(messages[0]
        .content
        .contains("默认输出 2 个 Candidate skill 节点 id"));
    assert_eq!(messages[0].content.matches("# 候选列表").count(), 1);
    assert_eq!(messages[0].content.matches("<CANDIDATE_TREE>\n").count(), 1);
    assert_eq!(messages[0].content.matches("</CANDIDATE_TREE>").count(), 1);
    for (code, label) in [("AA", "Summary"), ("AB", "Translate"), ("AC", "Weather")] {
        assert_eq!(
            messages[0]
                .content
                .matches(&format!("{label} [id: {code}]"))
                .count(),
            1,
            "code {code}"
        );
    }
    // The candidate text lives in the system message only.
    assert!(!messages[1].content.contains("[id:"));
    assert!(!messages[1].content.contains("候选列表"));
    assert!(messages[1].content.starts_with("<USER_REQUEST>\n[req:"));
    assert!(messages[1].content.ends_with("</USER_REQUEST>"));
    // The `From <speaker>:` prefix is stripped from the query.
    assert!(messages[1].content.contains("帮我总结这份文档"));
    assert!(!messages[1].content.contains("From user:"));

    // The prefix variant caches the system message and the empty request prefix.
    let parts = build_disclosure_prompt_parts(&fragment, &[], Some(2));
    assert_eq!(parts.prefix_messages[0], messages[0]);
    assert_eq!(parts.prefix_messages[1].content, "<USER_REQUEST>\n");
    assert_eq!(parts.full_messages.len(), 2);
    assert!(parts.suffix_text.starts_with("[req:"));
    assert_eq!(parts.prefix_token_hash.len(), 64);
    assert_eq!(parts.cache_id.len(), 32);
    // A different top_k is a different prompt identity.
    let other = build_disclosure_prompt_parts(&fragment, &[], Some(3));
    assert_ne!(other.prefix_token_hash, parts.prefix_token_hash);
    assert_ne!(other.cache_id, parts.cache_id);
    // The system prompt and the builder agree.
    assert_eq!(
        disclosure_system_prompt(&fragment, Some(2)),
        messages[0].content
    );
}

#[test]
fn a_multibyte_budget_truncates_on_character_boundaries() {
    let full = build(&DisclosureConfig::default());
    let budget = 40;
    let truncated = build(&DisclosureConfig::default().with_max_chars(budget));
    assert!(truncated.truncated);
    assert!(truncated.rendered_chars <= budget);
    assert_eq!(
        truncated.rendered_chars,
        truncated.rendered_tree.chars().count()
    );
    assert!(truncated.rendered_tree.ends_with("..."));
    assert!(!truncated.rendered_tree.contains('\u{FFFD}'));
    // The kept head is a prefix of the untruncated rendering.
    let head = truncated.rendered_tree.trim_end_matches("...");
    assert!(full.rendered_tree.starts_with(head), "{head:?}");
    // Codes and resolutions survive a text budget: the model still answers with them.
    assert_eq!(truncated.candidate_codes, full.candidate_codes);
    assert_eq!(truncated.code_to_resolution, full.code_to_resolution);
    // The fingerprint covers the whole exposure, not the truncated text.
    assert_eq!(truncated.fragment_fingerprint, full.fragment_fingerprint);

    // A CJK label cut in the middle is impossible: every prefix is char-safe.
    for budget in [1_usize, 2, 5, 7, 11, 23] {
        let cut = build(&DisclosureConfig::default().with_max_chars(budget));
        assert!(cut.rendered_chars <= budget);
        assert!(!cut.rendered_tree.contains('\u{FFFD}'));
        assert_eq!(cut.rendered_tree.chars().count(), cut.rendered_chars);
    }
}

#[test]
fn identical_reruns_are_byte_identical() {
    for config in [
        DisclosureConfig::default(),
        compact_config(3),
        DisclosureConfig::default().with_flatten_full_tree_in_prompt(true),
        compact_config(3).with_flatten_full_tree_in_prompt(true),
        DisclosureConfig::default().with_max_chars(30),
    ] {
        let first = build(&config);
        let second = build(&config);
        assert_eq!(first.rendered_tree, second.rendered_tree);
        assert_eq!(first, second);
        assert_eq!(first.fragment_fingerprint, second.fragment_fingerprint);

        let queries = [ChatMessage::user("总结一下"), ChatMessage::user("换个话题")];
        let first_prompt = build_disclosure_messages(&first, &queries, Some(2));
        let second_prompt = build_disclosure_messages(&second, &queries, Some(2));
        assert_eq!(first_prompt, second_prompt);
        assert_eq!(
            build_disclosure_prompt_parts(&first, &queries, Some(2)),
            build_disclosure_prompt_parts(&second, &queries, Some(2))
        );
        assert_eq!(
            parse_selected_codes(&first, "AA\nAC"),
            parse_selected_codes(&second, "AA\nAC")
        );
    }
}

#[test]
fn the_warmup_renderer_returns_the_fragment_without_messages() {
    let tree = catalogue();
    let fragment = build(&DisclosureConfig::default());
    let protocol = SelectionProtocol {
        compact_codes_enabled: fragment.compact_codes_enabled,
        candidate_codes: fragment.candidate_codes.clone(),
        code_width: fragment.code_width,
        abstain_token: openjiuwen_symphony::retrieval::search::flow::ABSTAIN_TOKEN.to_owned(),
    };
    let subtree = CurrentSubtree {
        cursor: openjiuwen_symphony::retrieval::search::flow::SearchCursor::new(&tree, 2),
        fragment: fragment.clone(),
        selectable_targets: fragment
            .resolutions()
            .into_iter()
            .cloned()
            .map(SelectableTarget::new)
            .collect(),
    };

    let warmup = DefaultSubtreeRenderer::without_messages()
        .render_subtree(&subtree, &[ChatMessage::user("总结一下")], &protocol)
        .unwrap();
    assert!(warmup.messages.is_empty());
    assert_eq!(warmup.fragment, fragment);
    assert_eq!(warmup.fragment.candidate_codes.len(), 3);
    assert_eq!(warmup.protocol, protocol);

    let full = DefaultSubtreeRenderer::new()
        .render_subtree(&subtree, &[ChatMessage::user("总结一下")], &protocol)
        .unwrap();
    assert_eq!(full.messages.len(), 2);
    assert_eq!(full.fragment, fragment);
    // The renderer never parses; the caller does that separately.
    assert_eq!(parse_selected_codes(&full.fragment, "Summary").len(), 1);
    // `top_k == 2` selects the tuned two-skill preamble.
    assert!(full.messages[0]
        .content
        .contains("默认输出 2 个 Candidate skill 节点 id"));
    let mut wider = subtree.clone();
    wider.cursor.top_k = 3;
    let relaxed = DefaultSubtreeRenderer::new()
        .render_subtree(&wider, &[ChatMessage::user("总结一下")], &protocol)
        .unwrap();
    assert!(relaxed.messages[0]
        .content
        .contains("总共输出至多 3 个 Candidate skill 节点 id"));
}

#[test]
fn flat_mode_sorts_candidates_and_emits_fixed_json_fields() {
    let config = compact_config(3).with_flatten_full_tree_in_prompt(true);
    let fragment = build(&config);
    assert!(fragment.flat_list_mode);
    assert!(fragment.compact_codes_enabled);
    // Alphabetical by selectable identity: summary, translate, weather.
    assert_eq!(
        fragment.candidate_codes,
        vec!["AA".to_owned(), "AB".to_owned(), "AC".to_owned()]
    );
    assert_eq!(
        fragment
            .resolutions()
            .into_iter()
            .map(|resolution| resolution.canonical_id.as_str())
            .collect::<Vec<&str>>(),
        vec!["skills.summary", "skills.translate", "skills.weather"]
    );
    // The flat view is a root with one selectable child per candidate.
    assert_eq!(fragment.root.children.len(), 3);
    assert!(fragment.root.children.iter().all(|node| node.is_selectable));
    assert!(!fragment.rendered_tree.contains("Category"));

    let parsed: serde_json::Value = serde_json::from_str(&fragment.rendered_tree).unwrap();
    let array = parsed.as_array().unwrap();
    assert_eq!(array.len(), 3);
    for candidate in array {
        let object = candidate.as_object().unwrap();
        assert_eq!(
            object.keys().collect::<Vec<&String>>(),
            vec!["category", "name", "raw_name", "description", "id"]
        );
    }
    let first = array[0].as_object().unwrap();
    assert_eq!(first["category"], "办公文档");
    assert_eq!(first["name"], "摘要");
    assert_eq!(first["raw_name"], "Summary");
    assert_eq!(first["description"], "把长文档压缩成三点");
    assert_eq!(first["id"], "AA");
    assert_eq!(array[2]["category"], "weather");
    assert_eq!(array[2]["id"], "AC");

    // Names mode renders a plain option list instead of JSON.
    let named = build(&DisclosureConfig::default().with_flatten_full_tree_in_prompt(true));
    assert_eq!(
        named.rendered_tree,
        "- Summary: 把长文档压缩成三点\n\
         - Translate: 在中文和英文之间翻译\n\
         - Weather: 查询实时天气"
    );
}

#[test]
fn the_fingerprint_is_stable_and_sensitive_to_the_exposure() {
    let fragment = build(&DisclosureConfig::default());
    assert_eq!(fragment.fragment_fingerprint.len(), 16);
    assert!(fragment
        .fragment_fingerprint
        .chars()
        .all(|character| character.is_ascii_hexdigit()));
    assert_eq!(
        fragmentation_fingerprint(&fragment.root, &fragment.candidate_codes),
        fragment.fragment_fingerprint
    );
    assert_eq!(
        fragmentation_fingerprint(
            &build(&DisclosureConfig::default()).root,
            &fragment.candidate_codes
        ),
        fragment.fragment_fingerprint
    );

    // A changed label changes the fingerprint.
    let mut relabelled = fragment.root.clone();
    relabelled.children[1].label = "天气服务".to_owned();
    assert_ne!(
        fragmentation_fingerprint(&relabelled, &fragment.candidate_codes),
        fragment.fragment_fingerprint
    );
    // A changed code does too.
    let mut recoded = fragment.candidate_codes.clone();
    recoded[0] = "Summary2".to_owned();
    assert_ne!(
        fragmentation_fingerprint(&fragment.root, &recoded),
        fragment.fragment_fingerprint
    );
    // So does a different configuration.
    assert_ne!(
        build(&compact_config(3)).fragment_fingerprint,
        fragment.fragment_fingerprint
    );
    // A flat exposure is its own fragment, so its fingerprint differs as well.
    assert_ne!(
        build(&DisclosureConfig::default().with_flatten_full_tree_in_prompt(true))
            .fragment_fingerprint,
        fragment.fragment_fingerprint
    );
}
