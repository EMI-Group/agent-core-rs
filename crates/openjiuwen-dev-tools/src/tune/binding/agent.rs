//! [`BoundAgent`]: the named parameter set one tuning run writes back into.
//!
//! Python's binding layer exists to mutate an agent's `LLMCall` parameters
//! (`BaseOptimizer::bind_parameter` keys them by name, `update()` writes each
//! staged gradient back). The Rust port has no live call to mutate, so the
//! agent-side adapter is a value: a name-keyed set of [`ParameterBinding`]s plus
//! the two operations a tuning driver needs — validate a candidate prompt map
//! against every binding, and hand the caller the resulting updates.
//!
//! Nothing here duplicates the single-parameter rules: [`BoundAgent::validate_all`]
//! delegates each parameter to [`ParameterBinding::apply`], and
//! [`BoundAgent::apply`] is the same operation under the name a caller reaching for
//! "apply the optimizer's output" expects.
use super::apply::{ParameterBinding, ParameterUpdate};
use openjiuwen_common::error::{Error, Result};
use std::collections::BTreeMap;
/// A named set of parameter bindings, kept in name order.
///
/// Names are matched exactly: only an all-whitespace name is rejected, and a
/// trailing space makes two names distinct. Parameters are stored sorted by name,
/// so [`parameters`](Self::parameters) and every update list are deterministic.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundAgent {
    /// The bound parameters, sorted by name.
    parameters: Vec<ParameterBinding>,
}
impl BoundAgent {
    /// Builds the set, rejecting a blank or duplicated parameter name.
    ///
    /// An empty set is accepted: it validates to an empty update list, because
    /// there is nothing to bind.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the offending parameter when a name is blank
    /// or bound twice.
    pub fn new(parameters: Vec<ParameterBinding>) -> Result<Self> {
        let mut sorted: Vec<ParameterBinding> = Vec::with_capacity(parameters.len());
        for binding in parameters {
            let name = binding.parameter().name.clone();
            if name.trim().is_empty() {
                return Err(Error::validation(
                    "BoundAgent parameter names must not be empty",
                ));
            }
            if sorted.iter().any(|bound| bound.parameter().name == name) {
                return Err(Error::validation(format!(
                    "BoundAgent binds parameter '{name}' twice"
                )));
            }
            sorted.push(binding);
        }
        sorted.sort_by(|left, right| left.parameter().name.cmp(&right.parameter().name));
        Ok(Self { parameters: sorted })
    }
    /// The bound parameters, sorted by name.
    #[must_use]
    pub fn parameters(&self) -> &[ParameterBinding] {
        &self.parameters
    }
    /// The binding for `name`, matched exactly.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&ParameterBinding> {
        self.parameters
            .iter()
            .find(|binding| binding.parameter().name == name)
    }
    /// Validates a candidate prompt for every bound parameter, in name order.
    ///
    /// Each declared parameter must have an entry in `candidate_prompts`, and each
    /// candidate must survive its binding's placeholder validation. Candidate
    /// entries for parameters this agent does not declare are ignored, so one
    /// candidate map can be shared by several agents (a supervisor and its
    /// sub-agents, say) without each of them owning every prompt.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the parameter when a candidate is missing;
    /// otherwise whatever the first failing binding reports.
    pub fn validate_all(
        &self,
        candidate_prompts: &BTreeMap<String, String>,
    ) -> Result<Vec<ParameterUpdate>> {
        let mut updates = Vec::with_capacity(self.parameters.len());
        for binding in &self.parameters {
            let name = &binding.parameter().name;
            let Some(candidate) = candidate_prompts.get(name) else {
                return Err(Error::validation(format!(
                    "BoundAgent has no candidate prompt for bound parameter '{name}'"
                )));
            };
            updates.push(binding.apply(candidate)?);
        }
        Ok(updates)
    }
    /// Validates and applies every candidate prompt, returning the updates.
    ///
    /// An alias for [`validate_all`](Self::validate_all) named for the operation a
    /// tuning driver performs; it mutates nothing, so the caller stores the new
    /// prompts it accepts.
    ///
    /// # Errors
    ///
    /// Whatever [`validate_all`](Self::validate_all) reports.
    pub fn apply(
        &self,
        candidate_prompts: &BTreeMap<String, String>,
    ) -> Result<Vec<ParameterUpdate>> {
        self.validate_all(candidate_prompts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(name: &str, template: &str) -> ParameterBinding {
        ParameterBinding::from_template(name, template)
    }

    fn candidates(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(name, prompt)| ((*name).to_owned(), (*prompt).to_owned()))
            .collect()
    }

    fn agent() -> BoundAgent {
        BoundAgent::new(vec![
            binding("user", "Question: {{question}}"),
            binding("system", "Answer in {{language}} about {{question}}."),
        ])
        .unwrap()
    }

    #[test]
    fn new_sorts_parameters_by_name() {
        let agent = agent();
        assert_eq!(
            agent
                .parameters()
                .iter()
                .map(|binding| binding.parameter().name.as_str())
                .collect::<Vec<&str>>(),
            vec!["system", "user"]
        );
        assert_eq!(
            agent.parameter("system").unwrap().parameter().name,
            "system"
        );
        assert!(agent.parameter("missing").is_none());
        assert!(agent.parameter("System").is_none(), "names match exactly");
    }

    #[test]
    fn new_rejects_blank_names() {
        let error =
            BoundAgent::new(vec![binding("system", "{{x}}"), binding("   ", "{{y}}")]).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("must not be empty"));
    }

    #[test]
    fn new_rejects_a_duplicate_name() {
        let error = BoundAgent::new(vec![binding("system", "{{x}}"), binding("system", "{{y}}")])
            .unwrap_err();
        assert!(error.to_string().contains("binds parameter 'system' twice"));
    }

    #[test]
    fn new_keeps_whitespace_distinct_and_accepts_an_empty_set() {
        let agent = BoundAgent::new(vec![
            binding("system", "{{x}}"),
            binding(" system", "{{y}}"),
        ])
        .unwrap();
        assert_eq!(agent.parameters().len(), 2);

        let empty = BoundAgent::default();
        assert_eq!(empty.parameters(), &[] as &[ParameterBinding]);
        assert_eq!(
            empty.validate_all(&BTreeMap::new()).unwrap(),
            Vec::<ParameterUpdate>::new()
        );
    }

    #[test]
    fn validate_all_returns_updates_in_name_order() {
        let agent = agent();
        let updates = agent
            .validate_all(&candidates(&[
                ("user", "Ask: {{question}}"),
                (
                    "system",
                    "Be brief. Answer in {{language}} about {{question}}.",
                ),
            ]))
            .unwrap();
        assert_eq!(
            updates
                .iter()
                .map(|update| update.parameter_name.as_str())
                .collect::<Vec<&str>>(),
            vec!["system", "user"]
        );
        assert!(updates.iter().all(ParameterUpdate::changed));
        assert_eq!(
            updates[0].preserved_placeholders,
            vec!["language".to_owned(), "question".to_owned()]
        );
        assert_eq!(
            updates[1].preserved_placeholders,
            vec!["question".to_owned()]
        );
        assert_eq!(updates[0].gain, None, "no optimizer result was applied");
        assert_eq!(
            updates[0].previous_prompt,
            "Answer in {{language}} about {{question}}."
        );
        assert_eq!(
            updates[0].new_prompt,
            "Be brief. Answer in {{language}} about {{question}}."
        );
    }

    #[test]
    fn validate_all_requires_a_candidate_for_every_parameter() {
        let agent = agent();
        let error = agent
            .validate_all(&candidates(&[("user", "Ask: {{question}}")]))
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("no candidate prompt for bound parameter 'system'"));
    }

    #[test]
    fn validate_all_ignores_candidates_for_undeclared_parameters() {
        let agent = agent();
        let updates = agent
            .validate_all(&candidates(&[
                ("system", "{{question}} in {{language}}"),
                ("user", "{{question}}"),
                ("subagent", "{{anything}}"),
            ]))
            .unwrap();
        assert_eq!(updates.len(), 2);
        assert!(updates
            .iter()
            .all(|update| update.parameter_name != "subagent"));
    }

    #[test]
    fn validate_all_propagates_a_failing_candidate() {
        let agent = agent();
        let error = agent
            .validate_all(&candidates(&[
                ("system", "Be brief."),
                ("user", "{{question}}"),
            ]))
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        let message = error.to_string();
        assert!(message.contains("parameter 'system'"), "{message}");
        assert!(message.contains("{{language}}"), "{message}");

        let error = agent
            .validate_all(&candidates(&[
                ("system", "{{question}} {{language}} {{extra}}"),
                ("user", "{{question}}"),
            ]))
            .unwrap_err();
        assert!(error.to_string().contains("{{extra}}"));
    }

    #[test]
    fn apply_is_the_validate_then_apply_alias() {
        let agent = agent();
        let prompts = candidates(&[
            ("system", "{{question}} / {{language}}"),
            ("user", "Question: {{question}}"),
        ]);
        let applied = agent.apply(&prompts).unwrap();
        assert_eq!(applied, agent.validate_all(&prompts).unwrap());
        assert!(!applied[1].changed(), "the user prompt was already current");
        assert!(applied[0].changed());

        let unchanged = agent
            .apply(&candidates(&[
                (
                    "system",
                    agent
                        .parameter("system")
                        .unwrap()
                        .parameter()
                        .template
                        .as_str(),
                ),
                (
                    "user",
                    agent
                        .parameter("user")
                        .unwrap()
                        .parameter()
                        .template
                        .as_str(),
                ),
            ]))
            .unwrap();
        assert!(unchanged.iter().all(|update| !update.changed()));
    }
}
