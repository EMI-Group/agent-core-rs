//! [`TextualParameter`]: a prompt template an optimizer rewrites.
//!
//! Python's `TextualParameter` (`openjiuwen/dev_tools/tune/optimizer/base.py`)
//! wraps a mutable `LLMCall` and records the textual gradients and candidate
//! values an optimizer staged for it. The Rust port has no `LLMCall`, so the
//! parameter *is* its template: `template` holds the text an optimizer rewrites,
//! [`TextualParameter::render`] fills it in from a placeholder map, and a
//! [`ParameterBinding`](super::ParameterBinding) applies a candidate to it.
//!
//! Because the parameter owns plain text rather than a handle to a live call, a
//! binding can be applied repeatedly and validated without mutating shared
//! state: nothing here has interior mutability, and every staged gradient or
//! candidate travels through a `with_*` builder that returns a new value.
use crate::tune::optimizer::placeholders;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// A named prompt template plus the gradients and candidates staged for it.
///
/// Python: `TextualParameter` in `openjiuwen.dev_tools.tune.optimizer.base`. The
/// JSON field names (`name`, `template`, `description`, `gradients`,
/// `candidates`) are stable, deserialization tolerates unknown fields, and every
/// field defaults to empty so a partial payload still parses.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TextualParameter {
    /// The parameter name, used as the key of a binding's parameter map.
    pub name: String,
    /// The prompt text this parameter currently holds.
    pub template: String,
    /// Free-form human context for the parameter, empty by default.
    pub description: String,
    /// Textual gradients staged per sub-parameter name.
    ///
    /// Python's `gradients` holds the single rewritten value an optimizer's
    /// `update()` applies; keeping it keyed by name lets one parameter carry a
    /// gradient for several fields at once.
    pub gradients: BTreeMap<String, String>,
    /// Every candidate value an optimizer considered, keyed by sub-parameter name.
    ///
    /// Python's `candidates` keeps the full pool (for example every prompt a
    /// multi-candidate optimizer generated), while `gradients` keeps only the
    /// winner, so a caller can evaluate more than the chosen value.
    pub candidates: BTreeMap<String, Vec<String>>,
}
impl TextualParameter {
    /// Creates a parameter from a name and its template.
    ///
    /// The constructor is infallible, matching Python's; call
    /// [`validate`](Self::validate) (or build a
    /// [`ParameterBinding`](super::ParameterBinding), which validates) to reject
    /// a blank name or template.
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
            ..Self::default()
        }
    }
    /// Replaces the human-readable description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
    /// Stages one textual gradient under `name`, replacing any previous value.
    #[must_use]
    pub fn with_gradient(mut self, name: impl Into<String>, gradient: impl Into<String>) -> Self {
        self.gradients.insert(name.into(), gradient.into());
        self
    }
    /// The staged gradient for `name`, when one was recorded.
    #[must_use]
    pub fn gradient(&self, name: &str) -> Option<&str> {
        self.gradients.get(name).map(String::as_str)
    }
    /// Records every candidate an optimizer considered for `name`.
    #[must_use]
    pub fn with_candidates(mut self, name: impl Into<String>, candidates: Vec<String>) -> Self {
        self.candidates.insert(name.into(), candidates);
        self
    }
    /// The candidates recorded for `name`; empty when none were recorded.
    #[must_use]
    pub fn candidates(&self, name: &str) -> &[String] {
        self.candidates.get(name).map_or(&[], Vec::as_slice)
    }
    /// The `{{name}}` placeholders the template declares, sorted.
    ///
    /// Delegates to [`placeholders`] so a binding and an optimizer agree on what
    /// a placeholder is, including the rule that a nested `{{` makes the outer
    /// pair malformed and is skipped rather than swallowed.
    #[must_use]
    pub fn placeholder_names(&self) -> Vec<String> {
        placeholders(&self.template).into_iter().collect()
    }
    /// Rejects a blank name or a blank template.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the offending field (`name` or `template`).
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation("TextualParameter.name must not be empty"));
        }
        if self.template.trim().is_empty() {
            return Err(Error::validation(format!(
                "TextualParameter '{}' template must not be empty",
                self.name
            )));
        }
        Ok(())
    }
    /// Renders the template, substituting every declared placeholder found in
    /// `values`.
    ///
    /// A placeholder with no entry in `values` is left verbatim (`{{name}}`), so
    /// a partially filled map renders a partially filled prompt instead of
    /// dropping text; a value is rendered bare when it is a JSON string and as
    /// compact JSON otherwise. Malformed brace pairs are preserved exactly as
    /// written.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the template is blank (there is nothing to
    /// render). The parameter name is not needed to render and is not checked
    /// here; use [`validate`](Self::validate) for that.
    pub fn render(&self, values: &BTreeMap<String, Value>) -> Result<String> {
        if self.template.trim().is_empty() {
            return Err(Error::validation(format!(
                "TextualParameter '{}' template must not be empty",
                self.name
            )));
        }
        Ok(render_template(&self.template, values))
    }
}
/// Substitutes `{{name}}` occurrences of `template` with the matching value.
///
/// A malformed opener (a `{{` that closes only after another `{{`) is emitted
/// verbatim and scanning resumes just past it, so unusual brace text survives a
/// render unchanged.
fn render_template(template: &str, values: &BTreeMap<String, Value>) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        rendered.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            rendered.push_str(&rest[start..]);
            return rendered;
        };
        if after[..end].contains("{{") {
            // Malformed outer pair: keep the opener and resume after it, so the
            // whole brace run round-trips instead of being dropped.
            rendered.push_str("{{");
            rest = after;
            continue;
        }
        let name = after[..end].trim();
        let whole = &rest[start..start + 2 + end + 2];
        match values.get(name) {
            Some(value) if !name.is_empty() && !name.contains('\n') => {
                rendered.push_str(&rendered_value(value));
            }
            _ => rendered.push_str(whole),
        }
        rest = &after[end + 2..];
    }
    rendered.push_str(rest);
    rendered
}
/// Renders one JSON value as prompt text: strings bare, everything else compact.
fn rendered_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn values(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect()
    }

    #[test]
    fn new_parameter_starts_empty_apart_from_name_and_template() {
        let parameter = TextualParameter::new("system", "Answer {{question}}.");
        assert_eq!(parameter.name, "system");
        assert_eq!(parameter.template, "Answer {{question}}.");
        assert_eq!(parameter.description, "");
        assert!(parameter.gradients.is_empty());
        assert!(parameter.candidates.is_empty());
        assert_eq!(parameter.placeholder_names(), vec!["question".to_owned()]);
    }

    #[test]
    fn placeholder_names_are_sorted_and_deduplicated() {
        let parameter = TextualParameter::new(
            "p",
            "{{ question }} then {{language}} then {{question}} and {{ language }}",
        );
        assert_eq!(
            parameter.placeholder_names(),
            vec!["language".to_owned(), "question".to_owned()]
        );
        assert_eq!(
            TextualParameter::new("p", "no placeholders").placeholder_names(),
            Vec::<String>::new()
        );
        assert_eq!(
            TextualParameter::new("p", "{{unclosed").placeholder_names(),
            Vec::<String>::new(),
            "an unclosed opener declares nothing"
        );
    }

    #[test]
    fn validate_rejects_blank_name_and_template() {
        let error = TextualParameter::new("  ", "Answer.")
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("name"));

        let error = TextualParameter::new("system", " \n ")
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("template"));
        assert!(error.to_string().contains("system"));

        TextualParameter::new("system", "Answer.")
            .validate()
            .unwrap();
    }

    #[test]
    fn render_substitutes_present_values_and_keeps_unknown_placeholders() {
        let parameter =
            TextualParameter::new("system", "Question: {{question}}\nAnswer in {{language}}.");
        let rendered = parameter
            .render(&values(&[("question", json!("2 + 2?"))]))
            .unwrap();
        assert_eq!(rendered, "Question: 2 + 2?\nAnswer in {{language}}.");
        assert_eq!(
            parameter
                .render(&values(&[
                    ("question", json!("2 + 2?")),
                    ("language", json!("French")),
                ]))
                .unwrap(),
            "Question: 2 + 2?\nAnswer in French."
        );
    }

    #[test]
    fn render_accepts_spaced_placeholders_and_renders_non_strings_compactly() {
        let parameter = TextualParameter::new("p", "{{ count }} items: {{items}} / {{flag}}");
        let rendered = parameter
            .render(&values(&[
                ("count", json!(3)),
                ("items", json!(["a", "b"])),
                ("flag", Value::Null),
            ]))
            .unwrap();
        assert_eq!(rendered, "3 items: [\"a\",\"b\"] / null");
    }

    #[test]
    fn render_keeps_malformed_braces_verbatim() {
        let parameter = TextualParameter::new("p", "x {{a{{b}} y");

        assert_eq!(
            parameter
                .render(&values(&[("b", json!("B")), ("a", json!("A"))]))
                .unwrap(),
            "x {{aB y",
            "the inner pair is a real placeholder; the malformed opener stays"
        );
        assert_eq!(parameter.render(&BTreeMap::new()).unwrap(), "x {{a{{b}} y");
        assert_eq!(
            TextualParameter::new("p", "tail {{unclosed")
                .render(&BTreeMap::new())
                .unwrap(),
            "tail {{unclosed"
        );
        assert_eq!(
            TextualParameter::new("p", "{{}} and {{ }}")
                .render(&BTreeMap::new())
                .unwrap(),
            "{{}} and {{ }}",
            "an empty placeholder name is not substituted"
        );
    }

    #[test]
    fn render_round_trips_through_the_placeholder_skeleton() {
        let parameter = TextualParameter::new("p", "A {{question}} B {{language}} C");
        let skeleton: BTreeMap<String, Value> = parameter
            .placeholder_names()
            .into_iter()
            .map(|name| {
                let placeholder = format!("{{{{{name}}}}}");
                (name, Value::String(placeholder))
            })
            .collect();
        assert_eq!(parameter.render(&skeleton).unwrap(), parameter.template);
    }

    #[test]
    fn render_rejects_a_blank_template() {
        let error = TextualParameter::new("p", "")
            .render(&BTreeMap::new())
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("template"));
    }

    #[test]
    fn gradients_and_candidates_are_staged_per_name() {
        let parameter = TextualParameter::new("system", "Answer {{question}}.")
            .with_description("system prompt")
            .with_gradient("instruction", "be terse")
            .with_gradient("format", "one line")
            .with_candidates(
                "instruction",
                vec!["be terse".to_owned(), "be kind".to_owned()],
            )
            .with_candidates("format", Vec::new());

        assert_eq!(parameter.description, "system prompt");
        assert_eq!(parameter.gradient("instruction"), Some("be terse"));
        assert_eq!(parameter.gradient("format"), Some("one line"));
        assert_eq!(parameter.gradient("missing"), None);
        assert_eq!(
            parameter.candidates("instruction"),
            &["be terse".to_owned(), "be kind".to_owned()][..]
        );
        assert_eq!(parameter.candidates("format"), &[] as &[String]);
        assert_eq!(parameter.candidates("missing"), &[] as &[String]);

        let replaced = parameter.with_gradient("instruction", "be terse");
        assert_eq!(
            replaced.gradients.len(),
            2,
            "a second gradient replaces, not appends"
        );
    }

    #[test]
    fn serde_round_trip_keeps_every_field() {
        let parameter = TextualParameter::new("system", "Answer {{question}}.")
            .with_description("d")
            .with_gradient("instruction", "g")
            .with_candidates("instruction", vec!["a".to_owned()]);
        let encoded = serde_json::to_value(&parameter).unwrap();
        assert_eq!(encoded["name"], json!("system"));
        assert_eq!(encoded["template"], json!("Answer {{question}}."));
        assert_eq!(encoded["description"], json!("d"));
        assert_eq!(encoded["gradients"]["instruction"], json!("g"));
        assert_eq!(encoded["candidates"]["instruction"], json!(["a"]));

        let decoded: TextualParameter = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, parameter);
    }

    #[test]
    fn serde_defaults_and_unknown_fields_are_tolerated() {
        let decoded: TextualParameter =
            serde_json::from_value(json!({"name": "system", "unknown": [1, 2]})).unwrap();
        assert_eq!(decoded.name, "system");
        assert_eq!(decoded.template, "");
        assert_eq!(decoded.description, "");
        assert!(decoded.gradients.is_empty());
        assert!(decoded.candidates.is_empty());

        let decoded: TextualParameter = serde_json::from_value(json!({})).unwrap();
        assert_eq!(decoded, TextualParameter::default());
    }
}
