//! Unit tests for `compact_template`: split/direction compaction with a total byte budget.

use openjiuwen_common::error::ErrorKind;

use super::*;

/// A transcript with one system message (stripped before the split) and six non-system messages: a
/// user request that names two files, an assistant decision, a fat tool result, an assistant error
/// report, a second fat tool result and a user feedback line.
///
/// The two fat tool results keep both sides of a split at 3 heavier than the summary line, so the
/// summary survives the "never grow" guard in both directions.
fn conversation() -> Vec<TranscriptMessage> {
    vec![
        TranscriptMessage::system("You are the parent member."),
        TranscriptMessage::user("read src/parser.rs and docs/PARITY.md"),
        TranscriptMessage::assistant("I decided to use a trait seam. Approach noted."),
        TranscriptMessage::tool("read_file", "a".repeat(400)),
        TranscriptMessage::assistant("error: parse failed on src/parser.rs:12"),
        TranscriptMessage::tool("bash", "b".repeat(400)),
        TranscriptMessage::user("please use tabs"),
    ]
}

/// The same conversation without its system message.
fn conversation_without_system() -> Vec<TranscriptMessage> {
    conversation()
        .into_iter()
        .filter(|message| message.role != CompactRole::System)
        .collect()
}

/// The contents of `messages`, in order.
fn contents(messages: &[TranscriptMessage]) -> Vec<&str> {
    messages
        .iter()
        .map(|message| message.content.as_str())
        .collect()
}

/// Total byte weight of `messages`.
fn weight(messages: &[TranscriptMessage]) -> usize {
    messages.iter().map(TranscriptMessage::byte_len).sum()
}

// ---------------------------------------------------------------------------------------------
// Split and direction
// ---------------------------------------------------------------------------------------------

#[test]
fn before_compresses_the_head_and_keeps_the_tail_verbatim() {
    let messages = conversation();
    let fat_tool = "b".repeat(400);
    let result = compact_with_template(&messages, &CompactTemplate::new(3)).unwrap();

    assert_eq!(result.direction, CompactDirection::Before);
    assert_eq!(result.compressed_side, CompactionSide::Head);
    assert_eq!(result.split_at, 3);
    assert_eq!(result.requested_split_at, 3);
    assert_eq!(result.compressed_messages, 3);
    assert_eq!(result.stripped_system_messages, 1);
    assert_eq!(result.kept_messages, 3);
    assert_eq!(result.squeezed_messages, 0);
    assert_eq!(result.truncated_tool_results, 0);
    assert_eq!(result.summary_omitted, None);
    assert!(!result.is_lossless());
    assert_eq!(
        contents(&result.messages),
        vec![
            result.summary.as_deref().unwrap(),
            "error: parse failed on src/parser.rs:12",
            fat_tool.as_str(),
            "please use tabs",
        ]
    );
    assert_eq!(result.messages[0].role, CompactRole::System);
    assert_eq!(result.compacted_bytes, weight(&result.messages));
}

#[test]
fn after_compresses_the_tail_and_keeps_the_head_verbatim() {
    let messages = conversation();
    let fat_tool = "a".repeat(400);
    let result = compact_with_template(&messages, &CompactTemplate::new(3).after()).unwrap();

    assert_eq!(result.direction, CompactDirection::After);
    assert_eq!(result.compressed_side, CompactionSide::Tail);
    assert_eq!(result.requested_split_at, 3);
    assert_eq!(
        result.split_at, 3,
        "the split lands on a non-tool message, so the head does not widen"
    );
    assert_eq!(result.compressed_messages, 3);
    assert_eq!(result.kept_messages, 3);
    assert_eq!(
        contents(&result.messages),
        vec![
            "read src/parser.rs and docs/PARITY.md",
            "I decided to use a trait seam. Approach noted.",
            fat_tool.as_str(),
            result.summary.as_deref().unwrap(),
        ]
    );
    assert_eq!(result.messages.last().unwrap().role, CompactRole::System);
}

#[test]
fn after_extends_the_head_through_trailing_tool_results() {
    let messages = vec![
        TranscriptMessage::user("head"),
        TranscriptMessage::tool("read_file", "a"),
        TranscriptMessage::tool("read_file", "b"),
        TranscriptMessage::user("t".repeat(300)),
    ];

    let extended = compact_with_template(&messages, &CompactTemplate::new(1).after()).unwrap();
    assert_eq!(extended.split_at, 3);
    assert_eq!(extended.kept_messages, 3);
    assert_eq!(extended.compressed_messages, 1);
    assert_eq!(
        contents(&extended.messages),
        vec!["head", "a", "b", extended.summary.as_deref().unwrap()]
    );

    let narrow = compact_with_template(
        &messages,
        &CompactTemplate::new(1).after().without_tool_extension(),
    )
    .unwrap();
    assert_eq!(narrow.split_at, 1);
    assert_eq!(narrow.kept_messages, 1);
    assert_eq!(narrow.compressed_messages, 3);
    assert_eq!(
        contents(&narrow.messages),
        vec!["head", narrow.summary.as_deref().unwrap()]
    );
}

#[test]
fn an_empty_compressed_side_is_lossless() {
    let messages = vec![
        TranscriptMessage::user("head"),
        TranscriptMessage::tool("read_file", "result"),
    ];
    let result = compact_with_template(&messages, &CompactTemplate::new(1).after()).unwrap();

    assert_eq!(result.compressed_messages, 0);
    assert_eq!(result.stripped_system_messages, 0);
    assert_eq!(result.summary, None);
    assert_eq!(result.summary_omitted, None);
    assert!(result.is_lossless());
    assert_eq!(result.messages, messages);
    assert_eq!(result.text(), "[user] head\n[tool:read_file] result");
}

#[test]
fn system_messages_are_stripped_before_the_split() {
    let messages = vec![
        TranscriptMessage::user("head one"),
        TranscriptMessage::system("identity"),
        TranscriptMessage::assistant("x".repeat(400)),
        TranscriptMessage::user("tail one"),
    ];

    // Three non-system messages, so the split is measured against 3, not against the raw length 4.
    let result = compact_with_template(&messages, &CompactTemplate::new(2)).unwrap();
    assert_eq!(result.stripped_system_messages, 1);
    assert_eq!(result.compressed_messages, 2);
    assert_eq!(result.kept_messages, 1);
    assert_eq!(result.messages.len(), 2);
    assert!(result.messages[0].content.starts_with(SUMMARY_PREFIX));
    assert_eq!(result.messages[1].content, "tail one");

    let error = compact_with_template(&messages, &CompactTemplate::new(3)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn keeping_system_messages_puts_them_first() {
    let messages = conversation();

    let before =
        compact_with_template(&messages, &CompactTemplate::new(3).keeping_system()).unwrap();
    assert_eq!(before.messages[0].role, CompactRole::System);
    assert_eq!(before.messages[0].content, "You are the parent member.");
    assert_eq!(before.messages[1].content, before.summary.clone().unwrap());
    assert_eq!(before.messages.len(), 5);

    let after = compact_with_template(&messages, &CompactTemplate::new(3).after().keeping_system())
        .unwrap();
    assert_eq!(after.messages[0].content, "You are the parent member.");
    assert!(!after.messages[0].content.starts_with(SUMMARY_PREFIX));
    assert_eq!(
        after.messages.last().unwrap().content,
        after.summary.clone().unwrap()
    );
}

#[test]
fn invalid_splits_are_typed_errors() {
    let messages = conversation();
    // Six non-system messages: 0 is empty-head, 6 and 7 are empty-tail / out of range.
    for split in [0, 6, 7, 99] {
        let error = compact_with_template(&messages, &CompactTemplate::new(split)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "split {split}");
        assert!(error.to_string().contains("split_at"));
    }

    let error = compact_with_template(&[TranscriptMessage::user("only")], &CompactTemplate::new(1))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    assert!(CompactTemplate::default().validate(5).is_err());
    assert!(CompactTemplate::new(1).validate(2).is_ok());
    assert!(CompactTemplate::new(2).validate(2).is_err());
    assert!(CompactTemplate::new(1).validate(1).is_err());
}

#[test]
fn direction_parsing_and_side_names() {
    assert_eq!(CompactDirection::default(), CompactDirection::Before);
    assert_eq!(
        CompactDirection::parse("before").unwrap(),
        CompactDirection::Before
    );
    assert_eq!(
        CompactDirection::parse("after").unwrap(),
        CompactDirection::After
    );
    let error = CompactDirection::parse("sideways").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("compact direction"));

    assert_eq!(CompactDirection::Before.as_str(), "before");
    assert_eq!(CompactDirection::After.as_str(), "after");
    assert_eq!(CompactDirection::Before.compresses(), CompactionSide::Head);
    assert_eq!(CompactDirection::After.compresses(), CompactionSide::Tail);
    assert_eq!(CompactionSide::default(), CompactionSide::Head);
    assert_eq!(CompactionSide::Head.as_str(), "head");
    assert_eq!(CompactionSide::Tail.as_str(), "tail");

    assert_eq!(SummaryOmission::Regressive.as_str(), "regressive");
    assert_eq!(SummaryOmission::Budget.as_str(), "budget");
    assert_eq!(BudgetTier::Summary.as_str(), "summary");
    assert_eq!(BudgetTier::KeptSide.as_str(), "kept_side");
    assert_eq!(BudgetTier::ToolResults.as_str(), "tool_results");
    assert_eq!(SummaryFact::Paths.as_str(), "paths");
    assert_eq!(SummaryFact::Decisions.as_str(), "decisions");
    assert_eq!(SummaryFact::Errors.as_str(), "errors");
    assert_eq!(SummaryFact::Feedback.as_str(), "feedback");
}

// ---------------------------------------------------------------------------------------------
// The deterministic summary
// ---------------------------------------------------------------------------------------------

#[test]
fn summary_reports_the_compressed_side_split_and_retention_facts() {
    let messages = conversation();

    let before = compact_with_template(&messages, &CompactTemplate::new(3)).unwrap();
    assert_eq!(
        before.summary.as_deref(),
        Some(
            "[compacted] head@3/6: 3 messages compressed (1 user, 1 assistant, 1 tool); \
             paths: src/parser.rs, docs/PARITY.md; \
             decisions: I decided to use a trait seam. Approach noted.; \
             feedback: read src/parser.rs and docs/PARITY.md"
        )
    );

    let after = compact_with_template(&messages, &CompactTemplate::new(3).after()).unwrap();
    assert_eq!(
        after.summary.as_deref(),
        Some(
            "[compacted] tail@3/6: 3 messages compressed (1 user, 1 assistant, 1 tool); \
             paths: src/parser.rs; \
             errors: error: parse failed on src/parser.rs:12; \
             feedback: please use tabs"
        )
    );
}

#[test]
fn summary_caps_excerpts_and_can_report_counts_only() {
    let messages = conversation();

    let capped = compact_with_template(
        &messages,
        &CompactTemplate::new(3).with_retention(SummaryRetention::new().with_max_facts(1)),
    )
    .unwrap();
    assert_eq!(
        capped.summary.as_deref(),
        Some(
            "[compacted] head@3/6: 3 messages compressed (1 user, 1 assistant, 1 tool); \
             paths: src/parser.rs (+1); \
             decisions: I decided to use a trait seam. Approach noted.; \
             feedback: read src/parser.rs and docs/PARITY.md"
        )
    );

    let counted = compact_with_template(
        &messages,
        &CompactTemplate::new(3).with_retention(SummaryRetention::new().with_max_facts(0)),
    )
    .unwrap();
    assert_eq!(
        counted.summary.as_deref(),
        Some(
            "[compacted] head@3/6: 3 messages compressed (1 user, 1 assistant, 1 tool); \
             paths: 2; decisions: 1; feedback: 1"
        )
    );

    let excerpted = compact_with_template(
        &messages,
        &CompactTemplate::new(3).with_retention(SummaryRetention::new().with_excerpt_chars(10)),
    )
    .unwrap();
    assert_eq!(
        excerpted.summary.as_deref(),
        Some(
            "[compacted] head@3/6: 3 messages compressed (1 user, 1 assistant, 1 tool); \
             paths: src/parser.rs, docs/PARITY.md; \
             decisions: I decided …; \
             feedback: read src/p…"
        )
    );
}

#[test]
fn retention_controls_which_facts_the_summary_reports() {
    let messages = conversation();

    let no_paths = compact_with_template(
        &messages,
        &CompactTemplate::new(3)
            .with_retention(SummaryRetention::new().without(SummaryFact::Paths)),
    )
    .unwrap();
    let line = no_paths.summary.as_deref().unwrap();
    assert!(!line.contains("paths:"));
    assert!(line.contains("decisions:"));
    assert!(line.contains("feedback:"));

    let none = compact_with_template(
        &messages,
        &CompactTemplate::new(3).with_retention(SummaryRetention::new().with_facts([])),
    )
    .unwrap();
    assert_eq!(
        none.summary.as_deref(),
        Some("[compacted] head@3/6: 3 messages compressed (1 user, 1 assistant, 1 tool)")
    );
    assert!(SummaryRetention::new().retains(SummaryFact::Errors));
    assert!(!SummaryRetention::new()
        .without(SummaryFact::Errors)
        .retains(SummaryFact::Errors));
    assert_eq!(
        SummaryRetention::new()
            .with_excerpt_chars(4)
            .max_excerpt_chars,
        4
    );
}

// ---------------------------------------------------------------------------------------------
// Monotonicity and the total byte budget
// ---------------------------------------------------------------------------------------------

#[test]
fn summary_is_omitted_when_it_would_grow_the_transcript() {
    let messages = vec![
        TranscriptMessage::user("src/a.rs"),
        TranscriptMessage::assistant("kept"),
    ];
    let result = compact_with_template(&messages, &CompactTemplate::new(1)).unwrap();

    assert_eq!(result.summary, None);
    assert_eq!(result.summary_omitted, Some(SummaryOmission::Regressive));
    assert_eq!(result.compressed_messages, 1);
    assert_eq!(result.kept_messages, 1);
    assert_eq!(contents(&result.messages), vec!["kept"]);
    assert_eq!(result.compacted_bytes, weight(&result.messages));
    assert!(result.compacted_bytes <= result.original_bytes);
}

#[test]
fn budget_omits_the_summary_when_the_template_frees_it_first() {
    // A fat compressed side keeps the summary inside the monotonicity guard, so only the budget can
    // remove it. Listing `summary` last makes it the first tier the budget reduces.
    let messages = vec![
        TranscriptMessage::user("h".repeat(400)),
        TranscriptMessage::assistant("kept one"),
    ];
    let unbudgeted = compact_with_template(&messages, &CompactTemplate::new(1)).unwrap();
    assert!(unbudgeted.summary.is_some());

    let budget = unbudgeted.compacted_bytes - 1;
    let squeezed = compact_with_template(
        &messages,
        &CompactTemplate::new(1)
            .with_budget(budget)
            .with_priority(vec![
                BudgetTier::KeptSide,
                BudgetTier::ToolResults,
                BudgetTier::Summary,
            ]),
    )
    .unwrap();

    assert_eq!(squeezed.summary, None);
    assert_eq!(squeezed.summary_omitted, Some(SummaryOmission::Budget));
    assert_eq!(squeezed.kept_messages, 1);
    assert_eq!(
        squeezed.squeezed_messages, 0,
        "an omitted summary is reported by summary_omitted, not squeezed_messages"
    );
    assert_eq!(contents(&squeezed.messages), vec!["kept one"]);
    assert!(squeezed.compacted_bytes <= budget);
}

#[test]
fn budget_drops_tool_results_before_kept_messages() {
    let messages = vec![
        TranscriptMessage::user("h".repeat(400)),
        TranscriptMessage::assistant("a1"),
        TranscriptMessage::tool("bash", "t".repeat(50)),
        TranscriptMessage::assistant("a2"),
    ];
    let full = compact_with_template(&messages, &CompactTemplate::new(1)).unwrap();
    let tool_bytes = messages[2].byte_len();
    let budget = full.compacted_bytes - tool_bytes;

    let squeezed =
        compact_with_template(&messages, &CompactTemplate::new(1).with_budget(budget)).unwrap();
    assert_eq!(squeezed.squeezed_messages, 1);
    assert_eq!(squeezed.kept_messages, 2);
    assert!(squeezed.summary.is_some(), "the summary is protected first");
    assert_eq!(
        contents(&squeezed.messages),
        vec![squeezed.summary.as_deref().unwrap(), "a1", "a2"]
    );
    assert!(squeezed.compacted_bytes <= budget);
}

#[test]
fn budget_keeps_the_messages_nearest_the_split() {
    // `before`: the kept side is the tail, so its first message is nearest the split.
    let messages = vec![
        TranscriptMessage::user("h".repeat(400)),
        TranscriptMessage::assistant("near"),
        TranscriptMessage::tool("bash", "t".repeat(50)),
        TranscriptMessage::assistant("f".repeat(100)),
    ];
    let full = compact_with_template(&messages, &CompactTemplate::new(1)).unwrap();
    let budget = messages[1].byte_len() + full.messages[0].byte_len() + 5;

    let squeezed =
        compact_with_template(&messages, &CompactTemplate::new(1).with_budget(budget)).unwrap();
    assert!(squeezed.summary.is_some());
    assert_eq!(
        contents(&squeezed.messages),
        vec![squeezed.summary.as_deref().unwrap(), "near"]
    );
    assert_eq!(
        squeezed.squeezed_messages, 2,
        "the tool result and the far message"
    );
    assert!(squeezed.compacted_bytes <= budget);

    // `after`: the kept side is the head, so its last message is nearest the split.
    let messages = vec![
        TranscriptMessage::assistant("f".repeat(100)),
        TranscriptMessage::tool("bash", "t".repeat(50)),
        TranscriptMessage::assistant("near"),
        TranscriptMessage::user("h".repeat(400)),
    ];
    let full = compact_with_template(&messages, &CompactTemplate::new(3).after()).unwrap();
    let summary_bytes = full.messages.last().unwrap().byte_len();
    let budget = messages[2].byte_len() + summary_bytes + 5;
    let squeezed = compact_with_template(
        &messages,
        &CompactTemplate::new(3).after().with_budget(budget),
    )
    .unwrap();
    assert_eq!(
        contents(&squeezed.messages),
        vec!["near", squeezed.summary.as_deref().unwrap()]
    );
    assert!(squeezed.compacted_bytes <= budget);
}

#[test]
fn custom_budget_priority_changes_the_squeeze_order() {
    // `after` with a trailing tool result: the head widens to the whole transcript, so there is no
    // summary to complicate the accounting.
    let messages = vec![
        TranscriptMessage::user("a1"),
        TranscriptMessage::tool("bash", "t".repeat(50)),
    ];
    let full = compact_with_template(&messages, &CompactTemplate::new(1).after()).unwrap();
    assert_eq!(full.compressed_messages, 0);
    assert_eq!(full.kept_messages, 2);
    let budget = messages[0].byte_len();

    let tools_first = compact_with_template(
        &messages,
        &CompactTemplate::new(1).after().with_budget(budget),
    )
    .unwrap();
    assert_eq!(contents(&tools_first.messages), vec!["a1"]);
    assert_eq!(tools_first.squeezed_messages, 1);

    let kept_side_first = compact_with_template(
        &messages,
        &CompactTemplate::new(1)
            .after()
            .with_budget(budget)
            .with_priority(vec![
                BudgetTier::Summary,
                BudgetTier::ToolResults,
                BudgetTier::KeptSide,
            ]),
    )
    .unwrap();
    assert_eq!(kept_side_first.messages, Vec::new());
    assert_eq!(kept_side_first.squeezed_messages, 2);
}

#[test]
fn budget_priority_must_be_non_empty_and_unique() {
    let messages = conversation();
    let empty = compact_with_template(
        &messages,
        &CompactTemplate::new(2).with_priority(Vec::new()),
    )
    .unwrap_err();
    assert_eq!(empty.kind(), ErrorKind::Validation);
    assert!(empty.to_string().contains("budget_priority"));

    let duplicate = compact_with_template(
        &messages,
        &CompactTemplate::new(2).with_priority(vec![BudgetTier::Summary, BudgetTier::Summary]),
    )
    .unwrap_err();
    assert_eq!(duplicate.kind(), ErrorKind::Validation);
    assert!(duplicate.to_string().contains("repeats tier 'summary'"));
}

#[test]
fn tool_results_are_truncated_at_the_template_budget() {
    let messages = vec![
        TranscriptMessage::user("head"),
        TranscriptMessage::tool("bash", "é".repeat(100)),
    ];
    let result =
        compact_with_template(&messages, &CompactTemplate::new(1).with_tool_budget(10)).unwrap();

    assert_eq!(result.truncated_tool_results, 1);
    let tool = result.messages.last().unwrap();
    assert_eq!(tool.role, CompactRole::Tool);
    assert!(tool.content.ends_with(DEFAULT_TRUNCATED_SUFFIX));
    assert!(tool.content.len() <= 10 + DEFAULT_TRUNCATED_SUFFIX.len());
    assert!(tool.content.len() < 200, "cut on a char boundary");

    // The compressed side is never truncated: only kept tool results are.
    let compressed_tool = vec![
        TranscriptMessage::tool("bash", "é".repeat(100)),
        TranscriptMessage::user("tail"),
    ];
    let untouched = compact_with_template(
        &compressed_tool,
        &CompactTemplate::new(1).with_tool_budget(4),
    )
    .unwrap();
    assert_eq!(untouched.truncated_tool_results, 0);
}

#[test]
fn compaction_never_grows_the_transcript_or_the_budget() {
    let mut transcripts = vec![
        conversation(),
        conversation_without_system(),
        vec![
            TranscriptMessage::tool("bash", "é".repeat(300)),
            TranscriptMessage::user("tail"),
        ],
    ];
    let templates = vec![
        CompactTemplate::new(1),
        CompactTemplate::new(1).after(),
        CompactTemplate::new(2).keeping_system(),
        CompactTemplate::new(2).after().keeping_system(),
        CompactTemplate::new(1).with_budget(0),
        CompactTemplate::new(1).with_budget(1),
        CompactTemplate::new(1).with_budget(40),
        CompactTemplate::new(2)
            .after()
            .keeping_system()
            .with_budget(64),
        CompactTemplate::new(1).with_budget(1_000_000),
        CompactTemplate::new(1)
            .with_priority(vec![
                BudgetTier::KeptSide,
                BudgetTier::ToolResults,
                BudgetTier::Summary,
            ])
            .with_budget(20),
    ];

    for messages in &mut transcripts {
        let original = weight(messages);
        for template in &templates {
            let Ok(result) = compact_with_template(messages, template) else {
                continue;
            };
            assert!(
                result.compacted_bytes <= result.original_bytes,
                "{template:?} grew {} bytes into {}",
                result.original_bytes,
                result.compacted_bytes
            );
            assert_eq!(result.original_bytes, original);
            if let Some(budget) = template.total_budget_bytes {
                assert!(
                    result.compacted_bytes <= budget.min(original),
                    "{template:?} exceeded its budget"
                );
            }
            assert_eq!(result.compacted_bytes, weight(&result.messages));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------------------------

#[test]
fn template_round_trips_through_json_and_tolerates_unknown_fields() {
    let template = CompactTemplate::new(3)
        .after()
        .keeping_system()
        .with_budget(4_096)
        .with_priority(vec![
            BudgetTier::Summary,
            BudgetTier::ToolResults,
            BudgetTier::KeptSide,
        ])
        .with_tool_budget(64)
        .without_tool_extension()
        .with_retention(
            SummaryRetention::new()
                .with_facts(vec![SummaryFact::Paths])
                .with_max_facts(2)
                .with_excerpt_chars(16),
        );

    let wire = serde_json::to_value(&template).unwrap();
    assert_eq!(
        wire,
        serde_json::json!({
            "split_at": 3,
            "direction": "after",
            "keep_system": true,
            "total_budget_bytes": 4096,
            "budget_priority": ["summary", "tool_results", "kept_side"],
            "max_tool_result_bytes": 64,
            "truncated_suffix": "… [truncated]",
            "extend_through_tool_results": false,
            "retention": {"facts": ["paths"], "max_facts": 2, "max_excerpt_chars": 16},
        })
    );
    let back: CompactTemplate = serde_json::from_value(wire).unwrap();
    assert_eq!(back, template);

    let tolerant: CompactTemplate =
        serde_json::from_value(serde_json::json!({"split_at": 2, "unknown": true})).unwrap();
    assert_eq!(tolerant, CompactTemplate::new(2));
    assert_eq!(CompactTemplate::default(), CompactTemplate::new(0));

    let result = compact_with_template(&conversation(), &CompactTemplate::new(2)).unwrap();
    let wire = serde_json::to_value(&result).unwrap();
    let rebuilt: TemplateCompaction = serde_json::from_value(wire).unwrap();
    assert_eq!(rebuilt, result);
    assert_eq!(TemplateCompaction::default().messages, Vec::new());
    assert_eq!(TemplateCompaction::default().summary, None);
    assert_eq!(TemplateCompaction::default().original_bytes, 0);
}
