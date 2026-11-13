//! Deterministic extractors: evaluation, conversation, review and team feedback
//! to [`EvolutionSignal`]s.
//!
//! Ports the offline adapters of Python `openjiuwen.agent_evolving.signal`
//! (`from_eval.py`, `from_conv.py`, `review_feedback.py`, `team.py`) onto the
//! Rust signal model. Every extractor is a pure function of its inputs:
//!
//! - output order follows input order, so repeated calls with identical inputs
//!   return identical values;
//! - every call is capped at [`MAX_SIGNALS_PER_INPUT`] signals, stopping as soon
//!   as the cap is reached;
//! - evidence is bounded — conversation, review, team and failure-fallback
//!   excerpts are all cut to [`MAX_EXCERPT_CHARS`], and evaluation excerpts are
//!   built from maps truncated to 80 characters each and then cut to
//!   [`MAX_EXCERPT_CHARS`];
//! - nothing calls a model, reads the environment or looks at a clock.
//!
//! Negative feedback is recognised by the closed marker list
//! [`NEGATIVE_MARKERS`] (English and Chinese), and the target layer by
//! `infer_target`'s fixed keyword priority.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use openjiuwen_core::foundation::prompt::{BaseMessage, Role};
use serde_json::Value;

use crate::dataset::EvaluatedCase;
use crate::signal::types::{truncate_excerpt, EvolutionSignal, EvolutionTarget, MAX_EXCERPT_CHARS};

/// The per-call cap on produced signals; extraction stops once it is reached.
pub const MAX_SIGNALS_PER_INPUT: usize = 64;

/// The substrings that mark feedback as negative, matched case-insensitively.
///
/// English and Chinese markers are kept in one list because a conversation mixes
/// them freely; matching is a plain lowercase substring test, so a marker may
/// appear anywhere in a segment.
pub const NEGATIVE_MARKERS: &[&str] = &[
    "wrong",
    "incorrect",
    "not correct",
    "failed",
    "doesn't work",
    "does not work",
    "please fix",
    "fix this",
    "错误",
    "不对",
    "失败",
    "修复",
    "重试",
];

/// The characters a feedback text is segmented on.
const SEGMENT_SEPARATORS: &[char] = &['\n', '\r', '.', '!', '?', '。', '！', '？', ';', '；'];

/// The length each serialized map is cut to before it enters an evaluation
/// excerpt.
const MAP_EXCERPT_CHARS: usize = 80;

/// Whether `text` contains any [`NEGATIVE_MARKERS`] entry.
fn contains_negative_marker(text: &str) -> bool {
    let lowered = text.to_lowercase();
    NEGATIVE_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

/// Guesses the target layer from keyword presence, in a fixed priority order.
///
/// Priority: tool keywords (`tool`, `工具`), then memory (`memory`, `记忆`), then
/// script/code (`script`, `脚本`, `code`, `代码`), then instruction/prompt
/// (`instruction`, `prompt`, `指令`, `提示词`); anything else targets the body.
fn infer_target(text: &str) -> EvolutionTarget {
    let lowered = text.to_lowercase();
    if lowered.contains("tool") || lowered.contains("工具") {
        EvolutionTarget::Tool
    } else if lowered.contains("memory") || lowered.contains("记忆") {
        EvolutionTarget::Memory
    } else if ["script", "脚本", "code", "代码"]
        .iter()
        .any(|keyword| lowered.contains(keyword))
    {
        EvolutionTarget::Script
    } else if ["instruction", "prompt", "指令", "提示词"]
        .iter()
        .any(|keyword| lowered.contains(keyword))
    {
        EvolutionTarget::Instruction
    } else {
        EvolutionTarget::Body
    }
}

/// Splits feedback into trimmed, non-empty segments on sentence separators.
fn feedback_segments(text: &str) -> Vec<&str> {
    text.split(SEGMENT_SEPARATORS)
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect()
}

/// The target for an evaluated case, derived from its label then input keys.
///
/// Within one key set the priority is `tool`/`tools`/`tool_name`, then `memory`,
/// then `skill`/`skill_name`, then `instruction`/`instructions`; anything else
/// targets the body. Label keys are consulted before input keys.
fn target_from_case(case: &EvaluatedCase) -> EvolutionTarget {
    let priority: [(&[&str], EvolutionTarget); 4] = [
        (&["tool", "tools", "tool_name"], EvolutionTarget::Tool),
        (&["memory"], EvolutionTarget::Memory),
        (&["skill", "skill_name"], EvolutionTarget::Skill),
        (
            &["instruction", "instructions"],
            EvolutionTarget::Instruction,
        ),
    ];
    for keys in [case.label(), case.inputs()] {
        for (names, target) in priority {
            if keys
                .keys()
                .any(|key| names.contains(&key.to_lowercase().as_str()))
            {
                return target;
            }
        }
    }
    EvolutionTarget::Body
}

/// Serializes a map, falling back to an empty string rather than panicking.
fn serialize_map(map: &BTreeMap<String, Value>) -> String {
    serde_json::to_string(map).unwrap_or_default()
}

/// The single-signal failure helper behind review and team extraction.
fn failure_signal(excerpt: &str, target: EvolutionTarget) -> Option<EvolutionSignal> {
    let excerpt = truncate_excerpt(excerpt, MAX_EXCERPT_CHARS);
    let signal = EvolutionSignal::new("execution_failure", target, excerpt).ok()?;
    signal.with_confidence(0.6).ok()
}

/// Derives one signal per evaluated case scoring below `threshold`.
///
/// A case produces `"low_score"` when its score is `0.0` and `"evaluated"`
/// otherwise, targeting the section `"Troubleshooting"` with source
/// `"evaluator"`. Confidence is `1.0 - score` clamped into `0.0..=1.0` and the
/// case is skipped when that value is not finite. The context carries
/// `case_id`, `score`, `label`, `prediction` (null when the case has no answer)
/// and `reason`. At most [`MAX_SIGNALS_PER_INPUT`] signals are returned.
#[must_use]
pub fn signals_from_evaluated_cases(
    cases: &[EvaluatedCase],
    threshold: f64,
) -> Vec<EvolutionSignal> {
    let mut signals = Vec::new();
    for case in cases {
        if signals.len() >= MAX_SIGNALS_PER_INPUT {
            break;
        }
        if case.score.partial_cmp(&threshold) != Some(Ordering::Less) {
            continue;
        }
        let confidence = (1.0 - case.score).clamp(0.0, 1.0);
        if !confidence.is_finite() {
            continue;
        }
        let prediction_text = case
            .answer
            .as_ref()
            .map_or_else(|| "none".to_owned(), serialize_map);
        let label_text = serialize_map(case.label());
        let excerpt = format!(
            "case {} scored {:.2}; prediction {}; label {}",
            case.case_id(),
            case.score,
            truncate_excerpt(&prediction_text, MAP_EXCERPT_CHARS),
            truncate_excerpt(&label_text, MAP_EXCERPT_CHARS),
        );
        let signal_type = if case.score == 0.0 {
            "low_score"
        } else {
            "evaluated"
        };
        let Ok(signal) = EvolutionSignal::new(
            signal_type,
            target_from_case(case),
            truncate_excerpt(&excerpt, MAX_EXCERPT_CHARS),
        ) else {
            continue;
        };
        let Ok(signal) = signal.with_confidence(confidence) else {
            continue;
        };
        let signal = signal
            .with_section("Troubleshooting")
            .with_source("evaluator")
            .with_context(evaluation_context(case));
        signals.push(signal);
    }
    signals
}

/// The provenance map attached to an evaluation signal.
fn evaluation_context(case: &EvaluatedCase) -> BTreeMap<String, Value> {
    let mut context = BTreeMap::new();
    context.insert(
        "case_id".to_owned(),
        Value::String(case.case_id().to_owned()),
    );
    context.insert(
        "score".to_owned(),
        serde_json::Number::from_f64(case.score).map_or(Value::Null, Value::Number),
    );
    context.insert(
        "label".to_owned(),
        Value::Object(case.label().clone().into_iter().collect()),
    );
    context.insert(
        "prediction".to_owned(),
        case.answer.as_ref().map_or(Value::Null, |answer| {
            Value::Object(answer.clone().into_iter().collect())
        }),
    );
    context.insert("reason".to_owned(), Value::String(case.reason.clone()));
    context
}

/// Derives one signal per user or tool message carrying negative feedback.
///
/// Only [`Role::User`] (kind `"negative_feedback"`, section `"Instructions"`)
/// and [`Role::Tool`] (kind `"execution_failure"`, section `"Troubleshooting"`)
/// messages are considered, at most one signal per message and at most
/// [`MAX_SIGNALS_PER_INPUT`] per call. The target comes from `infer_target`,
/// confidence is `0.6`, and the excerpt is the trimmed message text cut to
/// [`MAX_EXCERPT_CHARS`]. `skill_name` is attached when supplied.
#[must_use]
pub fn signals_from_conversation(
    messages: &[BaseMessage],
    skill_name: Option<&str>,
) -> Vec<EvolutionSignal> {
    let mut signals = Vec::new();
    for message in messages {
        if signals.len() >= MAX_SIGNALS_PER_INPUT {
            break;
        }
        let (signal_type, section) = match &message.role {
            Role::User => ("negative_feedback", "Instructions"),
            Role::Tool => ("execution_failure", "Troubleshooting"),
            _ => continue,
        };
        let text = message.text();
        if !contains_negative_marker(&text) {
            continue;
        }
        let excerpt = truncate_excerpt(text.trim(), MAX_EXCERPT_CHARS);
        if excerpt.is_empty() {
            continue;
        }
        let Ok(signal) = EvolutionSignal::new(signal_type, infer_target(&text), excerpt) else {
            continue;
        };
        let Ok(signal) = signal.with_confidence(0.6) else {
            continue;
        };
        let mut signal = signal.with_section(section).with_source("conversation");
        if let Some(skill) = skill_name {
            signal = signal.with_skill(skill);
        }
        signals.push(signal);
    }
    signals
}

/// Derives one signal per review segment carrying negative feedback.
///
/// Segments come from `feedback_segments`; each matching segment produces the
/// failure signal of [`signals_from_failure`] retyped as `"review_feedback"`
/// with section `"Review"`, source `"review"` and confidence `0.7`.
/// `skill_name` and the input cap apply as in the other extractors.
#[must_use]
pub fn signals_from_review_feedback(text: &str, skill_name: Option<&str>) -> Vec<EvolutionSignal> {
    let mut signals = Vec::new();
    for segment in feedback_segments(text) {
        if signals.len() >= MAX_SIGNALS_PER_INPUT {
            break;
        }
        if !contains_negative_marker(segment) {
            continue;
        }
        let Some(mut signal) = signals_from_failure(segment, infer_target(segment))
            .into_iter()
            .next()
        else {
            continue;
        };
        "review_feedback".clone_into(&mut signal.signal_type);
        let Ok(signal) = signal.with_confidence(0.7) else {
            continue;
        };
        let mut signal = signal.with_section("Review").with_source("review");
        if let Some(skill) = skill_name {
            signal = signal.with_skill(skill);
        }
        signals.push(signal);
    }
    signals
}

/// Derives one signal per team-feedback segment carrying negative feedback.
///
/// Same segmentation and marker rule as [`signals_from_review_feedback`], with
/// kind `"team_feedback"`, section `"Collaboration"`, source `"team"`,
/// confidence `0.7`, the reporting `agent_id` recorded as the skill name and as
/// an `agent_id` context entry. The input cap applies.
#[must_use]
pub fn signals_from_team_feedback(agent_id: &str, text: &str) -> Vec<EvolutionSignal> {
    let mut signals = Vec::new();
    for segment in feedback_segments(text) {
        if signals.len() >= MAX_SIGNALS_PER_INPUT {
            break;
        }
        if !contains_negative_marker(segment) {
            continue;
        }
        let Some(mut signal) = signals_from_failure(segment, infer_target(segment))
            .into_iter()
            .next()
        else {
            continue;
        };
        "team_feedback".clone_into(&mut signal.signal_type);
        let Ok(signal) = signal.with_confidence(0.7) else {
            continue;
        };
        let mut context = signal.context.clone();
        context.insert("agent_id".to_owned(), Value::String(agent_id.to_owned()));
        let signal = signal
            .with_section("Collaboration")
            .with_source("team")
            .with_skill(agent_id)
            .with_context(context);
        signals.push(signal);
    }
    signals
}

/// Turns free-form failure text into `"execution_failure"` signals for `target`.
///
/// Every feedback segment containing a [`NEGATIVE_MARKERS`] entry produces one
/// signal (section `"Troubleshooting"`, confidence `0.6`, no source). When no
/// segment matches but the trimmed text is non-empty, exactly one signal is
/// produced from that whole text. Segments that cannot form a signal are
/// skipped, never emitted empty, and at most [`MAX_SIGNALS_PER_INPUT`] signals
/// are returned.
#[must_use]
pub fn signals_from_failure(text: &str, target: EvolutionTarget) -> Vec<EvolutionSignal> {
    let mut signals = Vec::new();
    let mut matched = false;
    for segment in feedback_segments(text) {
        if signals.len() >= MAX_SIGNALS_PER_INPUT {
            break;
        }
        if !contains_negative_marker(segment) {
            continue;
        }
        matched = true;
        if let Some(signal) = failure_signal(segment, target) {
            signals.push(signal.with_section("Troubleshooting"));
        }
    }
    if !matched && signals.len() < MAX_SIGNALS_PER_INPUT {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if let Some(signal) = failure_signal(trimmed, target) {
                signals.push(signal.with_section("Troubleshooting"));
            }
        }
    }
    signals
}
