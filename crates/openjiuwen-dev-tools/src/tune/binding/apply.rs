//! [`ParameterBinding`]: applies an optimizer's winning prompt to a parameter.
//!
//! Python's binding mutates an agent's `LLMCall` in place (`BaseOptimizer::update`
//! writes the staged gradient back into the live call). The Rust port has no
//! `LLMCall` and no interior mutability, so a binding is a *function*: it
//! validates a candidate prompt against the parameter's declared placeholders and
//! returns a [`ParameterUpdate`] describing the change instead of performing it.
//! The caller decides whether to store the new prompt, which is what lets one
//! binding be applied repeatedly and validated without side effects.
use super::parameter::TextualParameter;
use crate::tune::optimizer::{placeholders, OptimizationResult};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// One validated parameter change: the old prompt, the new prompt, and the
/// placeholders that survived the rewrite.
///
/// Serializes with `snake_case` field names, ignores unknown fields, and defaults
/// every field, so a partial payload still parses.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ParameterUpdate {
    /// The name of the [`TextualParameter`] this update applies to.
    pub parameter_name: String,
    /// The prompt the parameter held before the update.
    pub previous_prompt: String,
    /// The candidate prompt that passed validation, stored verbatim.
    pub new_prompt: String,
    /// The template's declared placeholders, sorted; validation guarantees each
    /// one still appears in [`new_prompt`](Self::new_prompt).
    pub preserved_placeholders: Vec<String>,
    /// What the optimizer reported for this prompt, when it reported anything.
    ///
    /// A binding holds no baseline, so there is no delta to compute: this carries
    /// the winning *absolute* score and stays `None` unless that score is positive
    /// and finite (see [`ParameterBinding::apply_result`]).
    pub gain: Option<f64>,
}
impl ParameterUpdate {
    /// Whether the candidate actually differs from the previous prompt.
    ///
    /// A `false` answer means the optimizer handed back the prompt the parameter
    /// already held, so storing the update would change nothing.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.previous_prompt != self.new_prompt
    }
}
/// A [`TextualParameter`] plus the rules for accepting a new prompt for it.
///
/// The binding owns the template, so validation is a pure comparison between the
/// template's placeholders and the candidate's: a candidate that drops, renames,
/// or invents a placeholder is rejected with [`Error::validation`] rather than
/// producing a prompt a downstream parameter map could not fill.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterBinding {
    /// The parameter this binding guards.
    parameter: TextualParameter,
}
impl ParameterBinding {
    /// Wraps a parameter.
    ///
    /// Infallible, like the parameter constructor: the first
    /// [`validate`](Self::validate)/[`apply`](Self::apply) call reports a blank
    /// name or template.
    pub fn new(parameter: TextualParameter) -> Self {
        Self { parameter }
    }
    /// Builds a binding from a name and its template.
    pub fn from_template(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self::new(TextualParameter::new(name, template))
    }
    /// The bound parameter.
    #[must_use]
    pub fn parameter(&self) -> &TextualParameter {
        &self.parameter
    }
    /// Replaces the parameter's description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.parameter = self.parameter.with_description(description);
        self
    }
    /// Validates `candidate_prompt` against the parameter's declared placeholders.
    ///
    /// Checks, in order: the parameter itself (blank name/template), the candidate
    /// (blank), the placeholders the candidate dropped, and the placeholders the
    /// candidate introduced. A *new* placeholder is rejected too, because nothing
    /// downstream would have a value for it.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the parameter and, for the placeholder rules,
    /// the offending names sorted and rendered as `{{name}}`.
    pub fn validate(&self, candidate_prompt: &str) -> Result<()> {
        self.parameter.validate()?;
        let name = &self.parameter.name;
        if candidate_prompt.trim().is_empty() {
            return Err(Error::validation(format!(
                "candidate prompt for parameter '{name}' must not be empty"
            )));
        }
        let declared = placeholders(&self.parameter.template);
        let candidate = placeholders(candidate_prompt);
        let missing = rendered_names(declared.difference(&candidate));
        if !missing.is_empty() {
            return Err(Error::validation(format!(
                "candidate prompt for parameter '{name}' dropped declared placeholders: {missing}"
            )));
        }
        let added = rendered_names(candidate.difference(&declared));
        if !added.is_empty() {
            return Err(Error::validation(format!(
                "candidate prompt for parameter '{name}' introduced undeclared placeholders: {added}"
            )));
        }
        Ok(())
    }
    /// Validates `candidate_prompt` and returns the change it would make.
    ///
    /// Nothing is mutated: the parameter keeps its template, and the caller
    /// decides whether to adopt [`ParameterUpdate::new_prompt`].
    ///
    /// # Errors
    ///
    /// Whatever [`validate`](Self::validate) reports.
    pub fn apply(&self, candidate_prompt: &str) -> Result<ParameterUpdate> {
        self.validate(candidate_prompt)?;
        Ok(ParameterUpdate {
            parameter_name: self.parameter.name.clone(),
            previous_prompt: self.parameter.template.clone(),
            new_prompt: candidate_prompt.to_owned(),
            // Validation passed, so every declared placeholder survives.
            preserved_placeholders: self.parameter.placeholder_names(),
            gain: None,
        })
    }
    /// Applies an optimizer's winning prompt and reports the score as a gain.
    ///
    /// `OptimizationResult` reports an absolute score, never a delta, and a
    /// binding holds no baseline of its own; so `gain` is
    /// `Some(result.score)` only when that score is finite **and** strictly
    /// positive. A zero, negative, or non-finite score reports `None` rather than
    /// a fabricated improvement. A winning prompt that fails placeholder
    /// validation is an error, not a silent no-op.
    ///
    /// # Errors
    ///
    /// Whatever [`apply`](Self::apply) reports for `result.best_prompt`.
    pub fn apply_result(&self, result: &OptimizationResult) -> Result<ParameterUpdate> {
        let mut update = self.apply(&result.best_prompt)?;
        update.gain = if result.score.is_finite() && result.score > 0.0 {
            Some(result.score)
        } else {
            None
        };
        Ok(update)
    }
    /// A skeleton value map covering exactly the template's declared placeholders.
    ///
    /// Each value is the literal placeholder text, so
    /// `parameter.render(&binding.prompt_values())` reproduces the template
    /// unchanged; callers overwrite the entries they have real values for.
    #[must_use]
    pub fn prompt_values(&self) -> BTreeMap<String, Value> {
        self.parameter
            .placeholder_names()
            .into_iter()
            .map(|name| {
                let placeholder = format!("{{{{{name}}}}}");
                (name, Value::String(placeholder))
            })
            .collect()
    }
}
/// Renders placeholder names sorted as `{{name}}`, comma-separated.
fn rendered_names<'a>(names: impl Iterator<Item = &'a String>) -> String {
    names
        .map(|name| format!("{{{{{name}}}}}"))
        .collect::<Vec<String>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::optimizer::OptimizationResult;

    fn template_binding() -> ParameterBinding {
        ParameterBinding::from_template(
            "system",
            "You are concise.\nQuestion: {{question}}\nAnswer in {{language}}.",
        )
    }

    fn result(best_prompt: &str, score: f64) -> OptimizationResult {
        OptimizationResult {
            best_prompt: best_prompt.to_owned(),
            score,
            iterations: 1,
            converged: false,
            convergence_reason: None,
            token_usage: BTreeMap::new(),
            run_id: "run".to_owned(),
            records: Vec::new(),
            train_score: None,
            val_score: None,
            selected_on: None,
        }
    }

    #[test]
    fn constructors_keep_name_and_template() {
        let binding = template_binding();
        assert_eq!(binding.parameter().name, "system");
        assert_eq!(
            binding.parameter().template,
            template_binding().parameter().template
        );

        let binding =
            ParameterBinding::new(TextualParameter::new("p", "A {{x}}")).with_description("d");
        assert_eq!(binding.parameter().description, "d");
        assert_eq!(
            binding.parameter().placeholder_names(),
            vec!["x".to_owned()]
        );
    }

    #[test]
    fn validate_accepts_a_rewrite_that_keeps_every_placeholder() {
        let binding = template_binding();
        assert!(binding
            .validate("Be brief.\nQuestion: {{question}}\nAnswer in {{language}} only.")
            .is_ok());
        assert!(
            binding
                .validate(binding.parameter().template.as_str())
                .is_ok(),
            "identical prompt is valid (and change-free)"
        );
    }

    #[test]
    fn validate_rejects_a_dropped_placeholder_and_sorts_the_names() {
        let binding = ParameterBinding::from_template("p", "{{zebra}} {{apple}} {{mango}}");
        let error = binding.validate("nothing here").unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        let message = error.to_string();
        assert!(message.contains("parameter 'p'"), "{message}");
        assert!(
            message.contains("{{apple}}, {{mango}}, {{zebra}}"),
            "missing names are sorted: {message}"
        );

        let error = binding.validate("{{apple}}").unwrap_err();
        assert!(error.to_string().contains("{{mango}}, {{zebra}}"));
    }

    #[test]
    fn validate_rejects_an_ambiguous_rename_as_missing_plus_added() {
        let binding = ParameterBinding::from_template("p", "{{question}}");
        let error = binding.validate("{{query}}").unwrap_err();
        assert!(
            error.to_string().contains("{{question}}"),
            "the dropped name is reported first: {error}"
        );

        let error = ParameterBinding::from_template("p", "{{}}")
            .validate("{{query}}")
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("introduced undeclared placeholders: {{query}}"),
            "{error}"
        );
    }

    #[test]
    fn validate_rejects_a_new_placeholder() {
        let binding = ParameterBinding::from_template("p", "Answer {{question}}.");
        let error = binding
            .validate("Answer {{question}} in {{language}}.")
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(
            error
                .to_string()
                .contains("introduced undeclared placeholders: {{language}}"),
            "{error}"
        );
    }

    #[test]
    fn validate_rejects_a_blank_candidate_and_an_invalid_parameter() {
        let error = template_binding().validate("   \n").unwrap_err();
        assert!(error.to_string().contains("must not be empty"));
        assert!(error.to_string().contains("system"));

        let error = ParameterBinding::from_template("", "{{x}}")
            .validate("{{x}}")
            .unwrap_err();
        assert!(error.to_string().contains("TextualParameter.name"));

        let error = ParameterBinding::from_template("p", "")
            .validate("anything")
            .unwrap_err();
        assert!(error.to_string().contains("TextualParameter 'p' template"));
    }

    #[test]
    fn apply_returns_the_update_without_mutating_the_parameter() {
        let binding = template_binding();
        let update = binding
            .apply("Be brief.\nQuestion: {{question}}\nAnswer in {{language}}.")
            .unwrap();
        assert_eq!(update.parameter_name, "system");
        assert_eq!(update.previous_prompt, binding.parameter().template);
        assert_eq!(
            update.new_prompt,
            "Be brief.\nQuestion: {{question}}\nAnswer in {{language}}."
        );
        assert_eq!(
            update.preserved_placeholders,
            vec!["language".to_owned(), "question".to_owned()]
        );
        assert_eq!(update.gain, None);
        assert!(update.changed());
        assert_eq!(
            binding.parameter().template,
            template_binding().parameter().template,
            "apply never mutates the template"
        );

        let unchanged = binding
            .apply(binding.parameter().template.as_str())
            .unwrap();
        assert!(!unchanged.changed());
        assert_eq!(
            unchanged.preserved_placeholders,
            vec!["language".to_owned(), "question".to_owned()],
            "every placeholder always survives a validated apply"
        );

        // Applying twice is fine: the binding is a pure function of its template.
        assert_eq!(
            binding
                .apply(binding.parameter().template.as_str())
                .unwrap(),
            unchanged
        );
    }

    #[test]
    fn apply_rejects_a_failing_candidate() {
        let error = template_binding()
            .apply("no placeholders at all")
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
    }

    #[test]
    fn apply_result_reports_a_positive_finite_score_as_gain() {
        let binding = template_binding();
        let update = binding
            .apply_result(&result("Be brief: {{question}} / {{language}}", 0.8))
            .unwrap();
        assert_eq!(update.gain, Some(0.8));
        assert!(update.changed());
        assert_eq!(
            update.preserved_placeholders,
            vec!["language".to_owned(), "question".to_owned()]
        );

        assert_eq!(
            binding
                .apply_result(&result("{{question}} {{language}}", 0.0))
                .unwrap()
                .gain,
            None,
            "a zero score is not a gain"
        );
        for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            assert_eq!(
                binding
                    .apply_result(&result("{{question}} {{language}}", score))
                    .unwrap()
                    .gain,
                None,
                "score {score} reports no gain"
            );
        }
    }

    #[test]
    fn apply_result_rejects_a_best_prompt_that_drops_a_placeholder() {
        let error = template_binding()
            .apply_result(&result("Be brief.", 0.9))
            .unwrap_err();
        assert!(error.to_string().contains("dropped declared placeholders"));
    }

    #[test]
    fn prompt_values_is_a_renderable_skeleton() {
        let binding = template_binding();
        let values = binding.prompt_values();
        assert_eq!(
            values.keys().cloned().collect::<Vec<String>>(),
            vec!["language".to_owned(), "question".to_owned()]
        );
        assert_eq!(
            binding.parameter().render(&values).unwrap(),
            binding.parameter().template
        );

        let mut filled = values.clone();
        filled.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
        assert_eq!(
            binding.parameter().render(&filled).unwrap(),
            "You are concise.\nQuestion: 2 + 2?\nAnswer in {{language}}."
        );
    }

    #[test]
    fn parameter_update_round_trips_through_json() {
        let update = ParameterUpdate {
            parameter_name: "system".to_owned(),
            previous_prompt: "a {{x}}".to_owned(),
            new_prompt: "b {{x}}".to_owned(),
            preserved_placeholders: vec!["x".to_owned()],
            gain: Some(0.5),
        };
        let encoded = serde_json::to_value(&update).unwrap();
        assert_eq!(encoded["parameter_name"], serde_json::json!("system"));
        assert_eq!(encoded["previous_prompt"], serde_json::json!("a {{x}}"));
        assert_eq!(encoded["preserved_placeholders"], serde_json::json!(["x"]));
        assert_eq!(encoded["gain"], serde_json::json!(0.5));

        let decoded: ParameterUpdate = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, update);
        assert!(decoded.changed());

        let decoded: ParameterUpdate =
            serde_json::from_value(serde_json::json!({"new_prompt": "b", "extra": 1})).unwrap();
        assert_eq!(decoded.new_prompt, "b");
        assert_eq!(decoded.parameter_name, "");
        assert_eq!(decoded.previous_prompt, "");
        assert_eq!(decoded.preserved_placeholders, Vec::<String>::new());
        assert_eq!(decoded.gain, None);
    }
}
