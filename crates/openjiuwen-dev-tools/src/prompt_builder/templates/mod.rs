//! The bundled template banks and the message constructors that wrap them.
//!
//! Ports the module-level prompt tables of
//! `openjiuwen/dev_tools/prompt_builder/builder/prompt_zh.py` and `prompt_en.py`
//! together with Python's `template_map` / `select_template` lookup.
//!
//! # Layout
//!
//! - [`zh_cn`] — the `zh-CN` bank, the default.
//! - [`en_us`] — the `en-US` bank.
//!
//! Both modules expose their texts as `pub const` items with the Python constant
//! names and collect them into a [`PromptBank`]. The texts are copied verbatim,
//! so the two banks differ only in language, never in shape: each text keeps the
//! Python triple-quoted literal's leading and trailing newline and its
//! `{{placeholder}}` markers.
//!
//! # Placeholders
//!
//! A template text is plain data. [`system_template`] and [`user_template`] wrap
//! it in a single message, and the `{{name}}` slots are filled later by
//! `PromptTemplate::format` (the caller supplies the values). Nothing in this
//! module renders anything by itself, so a builder can inspect a template's
//! placeholders before deciding what to fill.

pub mod en_us;
pub mod zh_cn;

use openjiuwen_foundation::prompt::{BaseMessage, PromptTemplate};

use super::language::Language;

/// One complete set of prompt-builder template texts.
///
/// The fields mirror the Python constant names one for one, so `bank(...).field`
/// reads as the constant it aliases. Every field is a `'static` reference into
/// the owning bank, so a [`PromptBank`] is a `Copy` view with no allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptBank {
    /// Meta-template system message for general (no-tools) prompt building.
    pub meta_general_system: &'static str,
    /// Meta-template user message carrying `{{instruction}}` for general building.
    pub meta_general_user: &'static str,
    /// Meta-template system message for planning (tool-aware) prompt building.
    pub meta_plan_system: &'static str,
    /// Meta-template user message carrying `{{instruction}}` and `{{tools}}`.
    pub meta_plan_user: &'static str,
    /// Feedback step 1: value judgement and feedback normalization.
    pub feedback_intent: &'static str,
    /// Feedback mode `general`: rewrite the whole prompt.
    pub feedback_general: &'static str,
    /// Feedback mode `select`: rewrite one selected fragment.
    pub feedback_select: &'static str,
    /// Feedback mode `insert`: generate content for the insert marker.
    pub feedback_insert: &'static str,
    /// Bad-case step 1: analyze the counterexamples.
    pub bad_case_analyze: &'static str,
    /// Bad-case step 2: repair the prompt from the aggregated feedback.
    pub bad_case_optimize: &'static str,
    /// Plain-text rendering of one evaluated case.
    pub format_bad_case: &'static str,
}

/// Returns the template bank for `language`.
///
/// Python's `select_template` resolves the same two dictionaries; the fallback to
/// `zh-CN` for an unknown tag happens earlier here, in
/// [`Language`], which rejects unknown tags instead of
/// defaulting.
#[must_use]
pub fn bank(language: Language) -> &'static PromptBank {
    match language {
        Language::ZhCn => &zh_cn::BANK,
        Language::EnUs => &en_us::BANK,
    }
}

/// Wraps `text` in a template holding exactly one system message.
///
/// Mirrors the Python `PromptTemplate(content=[SystemMessage(content=...)])` form
/// used by the two meta-template system prompts.
#[must_use]
pub fn system_template(text: &str) -> PromptTemplate {
    PromptTemplate::new(vec![BaseMessage::system(text)])
}

/// Wraps `text` in a template holding exactly one user message.
///
/// Mirrors the Python `PromptTemplate(content=[UserMessage(content=...)])` form
/// used by every other single-message template.
#[must_use]
pub fn user_template(text: &str) -> PromptTemplate {
    PromptTemplate::new(vec![BaseMessage::user(text)])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use openjiuwen_foundation::prompt::Role;
    use serde_json::Value;

    use super::*;

    /// Every bank field, in declaration order, paired with its text.
    fn fields(bank: &PromptBank) -> [(&'static str, &'static str); 11] {
        [
            ("meta_general_system", bank.meta_general_system),
            ("meta_general_user", bank.meta_general_user),
            ("meta_plan_system", bank.meta_plan_system),
            ("meta_plan_user", bank.meta_plan_user),
            ("feedback_intent", bank.feedback_intent),
            ("feedback_general", bank.feedback_general),
            ("feedback_select", bank.feedback_select),
            ("feedback_insert", bank.feedback_insert),
            ("bad_case_analyze", bank.bad_case_analyze),
            ("bad_case_optimize", bank.bad_case_optimize),
            ("format_bad_case", bank.format_bad_case),
        ]
    }

    /// The `{{name}}` slots each field must contain.
    ///
    /// `meta_general_system` has none in `zh-CN`; the `en-US` text shows an
    /// example marker (`{{this}}`) inside its prose, which the loop below picks up
    /// from the template itself rather than from this table.
    fn required_keys(field: &str) -> &'static [&'static str] {
        match field {
            "meta_general_system" | "meta_plan_system" => &[],
            "meta_general_user" => &["instruction"],
            "meta_plan_user" => &["instruction", "tools"],
            "feedback_intent" => &["original_prompt", "feedbacks"],
            "feedback_general" | "feedback_insert" => &["original_prompt", "suggestion"],
            "feedback_select" => &["original_prompt", "pending_optimized_prompt", "suggestion"],
            "bad_case_analyze" => &["original_prompt", "bad_cases"],
            "bad_case_optimize" => &["original_prompt", "feedback"],
            "format_bad_case" => &["question", "label", "answer", "reason"],
            other => panic!("unknown bank field {other}"),
        }
    }

    #[test]
    fn bank_selects_the_language() {
        assert_eq!(
            bank(Language::ZhCn).meta_general_system,
            zh_cn::PROMPT_BUILD_GENERAL_META_SYSTEM_TEMPLATE
        );
        assert_eq!(
            bank(Language::EnUs).meta_general_system,
            en_us::PROMPT_BUILD_GENERAL_META_SYSTEM_TEMPLATE
        );
        assert_ne!(bank(Language::ZhCn), bank(Language::EnUs));
    }

    #[test]
    fn every_text_of_both_banks_is_non_empty() {
        for language in [Language::ZhCn, Language::EnUs] {
            for (field, text) in fields(bank(language)) {
                assert!(!text.is_empty(), "{language} {field} must not be empty");
                assert!(
                    text.starts_with('\n') && text.ends_with('\n'),
                    "{language} {field} must keep the Python literal's newlines"
                );
            }
        }
    }

    #[test]
    fn the_two_banks_differ_except_for_the_case_format() {
        let zh = fields(bank(Language::ZhCn));
        let en = fields(bank(Language::EnUs));
        for ((zh_field, zh_text), (en_field, en_text)) in zh.iter().zip(en.iter()) {
            assert_eq!(zh_field, en_field);
            if *zh_field == "format_bad_case" {
                // The case rendering is language-neutral in Python: same text.
                assert_eq!(zh_text, en_text, "{zh_field} is shared by both banks");
            } else {
                assert_ne!(
                    zh_text, en_text,
                    "{zh_field} must be translated in the en-US bank"
                );
            }
        }
    }

    #[test]
    fn every_text_contains_its_expected_placeholders() {
        for language in [Language::ZhCn, Language::EnUs] {
            for (field, text) in fields(bank(language)) {
                for key in required_keys(field) {
                    let marker = format!("{{{{{key}}}}}");
                    assert!(
                        text.contains(&marker),
                        "{language} {field} must contain {marker}"
                    );
                }
            }
        }
    }

    #[test]
    fn formatting_with_every_placeholder_leaves_no_slot() {
        for language in [Language::ZhCn, Language::EnUs] {
            for (field, text) in fields(bank(language)) {
                let template = user_template(text);
                let mut names = template.placeholders();
                names.sort();

                // The slots the text declares must be exactly the documented ones.
                let mut expected: Vec<String> = required_keys(field)
                    .iter()
                    .map(|key| (*key).to_owned())
                    .collect();
                if language == Language::EnUs && field == "meta_general_system" {
                    // The English prose illustrates a marker inline: `like {{this}}`.
                    expected.push("this".to_owned());
                }
                expected.sort();
                assert_eq!(
                    names, expected,
                    "{language} {field} declares unexpected placeholders"
                );

                let values: BTreeMap<String, Value> = names
                    .iter()
                    .map(|name| (name.clone(), Value::String(format!("<{name}>"))))
                    .collect();
                let rendered_text = template.format(&values).to_messages()[0]
                    .text()
                    .into_owned();
                assert!(
                    !rendered_text.contains("{{"),
                    "{language} {field} left an unrendered slot: {rendered_text}"
                );
                if names.is_empty() {
                    assert_eq!(rendered_text, text, "{field} has nothing to render");
                } else {
                    assert_ne!(rendered_text, text, "{field} must change when rendered");
                }
            }
        }
    }

    #[test]
    fn system_and_user_templates_hold_one_message() {
        let system = system_template(zh_cn::PROMPT_BUILD_GENERAL_META_SYSTEM_TEMPLATE);
        let messages = system.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            messages[0].text(),
            zh_cn::PROMPT_BUILD_GENERAL_META_SYSTEM_TEMPLATE
        );

        let user = user_template(zh_cn::PROMPT_BUILD_GENERAL_META_USER_TEMPLATE);
        let messages = user.to_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(
            messages[0].text(),
            zh_cn::PROMPT_BUILD_GENERAL_META_USER_TEMPLATE
        );

        // An arbitrary text is passed through unchanged, newlines included.
        let text = "\nplain {{instruction}}\n";
        assert_eq!(system_template(text).to_messages()[0].text(), text);
        assert_eq!(user_template(text).to_messages()[0].text(), text);
    }

    #[test]
    fn the_case_format_ends_with_the_record_separator() {
        for language in [Language::ZhCn, Language::EnUs] {
            let text = bank(language).format_bad_case;
            // Python writes `=== \n`, i.e. the separator plus a trailing space.
            assert!(text.ends_with("=== \n"), "{language} separator");
            assert!(text.trim_end().ends_with("==="), "{language} trim_end");
        }
    }
}
