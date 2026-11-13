//! `SkillCallOperator`: skill-selection parameters of a skill call.
//!
//! Python's `openjiuwen/core/operator/skill_call/base.py` is a preview-only
//! *skill-experience* handle. The Rust counterpart keeps the same identity and
//! callback surface but tunes the selection side instead: which skills the
//! agent may advertise (`selected_skills`) and how many it may show
//! (`max_skills`). Experience staging stays with the caller's lifecycle
//! manager, as in Python.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};

use super::base::{ApplyResult, Operator, ParameterCallback, TunableKind, TunableSpec};
use crate::skills::Skill;

/// Default cap on selected skills.
pub const DEFAULT_MAX_SKILLS: usize = 5;

/// Largest `max_skills` a tuned operator may request.
pub const MAX_MAX_SKILLS: usize = 100;

/// Tunable skill-selection parameters of one skill call.
pub struct SkillCallOperator {
    operator_id: String,
    selected_skills: Vec<String>,
    max_skills: usize,
    on_parameter_updated: Option<ParameterCallback>,
}

impl SkillCallOperator {
    /// Creates a handle that considers every skill and shows up to
    /// [`DEFAULT_MAX_SKILLS`].
    #[must_use]
    pub fn new(operator_id: impl Into<String>) -> Self {
        Self {
            operator_id: operator_id.into(),
            selected_skills: Vec::new(),
            max_skills: DEFAULT_MAX_SKILLS,
            on_parameter_updated: None,
        }
    }

    /// Restricts selection to `names`; an empty list means every skill.
    #[must_use]
    pub fn with_selected_skills(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.selected_skills = names.into_iter().collect();
        self
    }

    /// Sets the show cap.
    #[must_use]
    pub fn with_max_skills(mut self, max_skills: usize) -> Self {
        self.max_skills = max_skills.min(MAX_MAX_SKILLS);
        self
    }

    /// Installs the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// The curated skill names; empty means every skill.
    #[must_use]
    pub fn selected_skills(&self) -> &[String] {
        &self.selected_skills
    }

    /// The show cap.
    #[must_use]
    pub const fn max_skills(&self) -> usize {
        self.max_skills
    }

    /// Selects the skills relevant to `query`, capped by `max_skills`.
    ///
    /// Candidates are the curated skills (or every skill when the list is
    /// empty). Skills matching `query` through [`Skill::matches`] are kept; when
    /// none matches, every candidate is kept so an unrelated query cannot hide
    /// the whole library. The result keeps input order and is truncated to
    /// `max_skills` (zero selects nothing).
    #[must_use]
    pub fn select<'a>(&self, skills: &'a [Skill], query: &str) -> Vec<&'a Skill> {
        if self.max_skills == 0 {
            return Vec::new();
        }
        let candidates: Vec<&Skill> = if self.selected_skills.is_empty() {
            skills.iter().collect()
        } else {
            skills
                .iter()
                .filter(|skill| {
                    self.selected_skills
                        .iter()
                        .any(|name| name.trim().eq_ignore_ascii_case(skill.name.trim()))
                })
                .collect()
        };
        let matched: Vec<&Skill> = candidates
            .iter()
            .copied()
            .filter(|skill| skill.matches(query))
            .collect();
        let chosen = if matched.is_empty() {
            candidates
        } else {
            matched
        };
        chosen.into_iter().take(self.max_skills).collect()
    }

    /// Parses and stores one field, returning its canonical JSON value.
    fn store(&mut self, target: &str, value: &Value) -> Result<Value> {
        match target {
            "selected_skills" => {
                self.selected_skills = parse_names(value)?;
                Ok(json!(self.selected_skills))
            }
            "max_skills" => {
                let max_skills = serde_json::from_value::<usize>(value.clone()).map_err(|_| {
                    Error::validation("tunable 'max_skills' expects a non-negative integer")
                })?;
                self.max_skills = max_skills.min(MAX_MAX_SKILLS);
                Ok(json!(self.max_skills))
            }
            other => Err(Error::validation(format!("unknown target '{other}'"))),
        }
    }

    /// Fires the callback for `target`.
    fn fire(&self, target: &str, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(target, value);
        }
    }
}

/// Parses a skill-name list: one name or a list of names.
fn parse_names(value: &Value) -> Result<Vec<String>> {
    match value {
        Value::String(name) => Ok(vec![name.clone()]),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| Error::validation("skill names must be text"))
            })
            .collect(),
        _ => Err(Error::validation(
            "tunable 'selected_skills' expects text or a list of text",
        )),
    }
}

#[async_trait]
impl Operator for SkillCallOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        vec![
            TunableSpec::new("selected_skills", TunableKind::TextList)
                .with_description("curated skill names; empty selects every skill"),
            TunableSpec::new("max_skills", TunableKind::Number)
                .with_description("maximum number of skills to expose"),
        ]
    }

    fn get_state(&self) -> Value {
        json!({
            "selected_skills": self.selected_skills,
            "max_skills": self.max_skills,
        })
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        let stored = match self.store(target, &value) {
            Ok(stored) => stored,
            Err(error) => return Ok(ApplyResult::skipped(target, error.to_string())),
        };
        self.fire(target, &stored);
        Ok(ApplyResult::applied(target))
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        let object = state
            .as_object()
            .ok_or_else(|| Error::validation("operator state must be a JSON object"))?;
        for target in ["selected_skills", "max_skills"] {
            let Some(value) = object.get(target) else {
                continue;
            };
            let stored = self.store(target, value)?;
            self.fire(target, &stored);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::assert_is_empty
    )]

    use super::*;
    use std::sync::Arc;

    fn skills() -> Vec<Skill> {
        vec![
            Skill::new("alpha", "Alpha skill").with_keywords(["alpha"]),
            Skill::new("beta", "Beta skill").with_keywords(["beta"]),
            Skill::new("gamma", "Gamma skill").with_keywords(["gamma"]),
        ]
    }

    type Recorded = Arc<parking_lot::Mutex<Vec<(String, Value)>>>;

    fn recording_callback() -> (ParameterCallback, Recorded) {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: ParameterCallback = Arc::new(move |target: &str, value: &Value| {
            sink.lock().push((target.to_owned(), value.clone()));
        });
        (callback, log)
    }

    #[test]
    fn tunables_and_defaults() {
        let operator = SkillCallOperator::new("agent/skill_call");
        assert_eq!(operator.max_skills(), DEFAULT_MAX_SKILLS);
        assert!(operator.selected_skills().is_empty());
        let names: Vec<String> = operator
            .get_tunables()
            .into_iter()
            .map(|spec| spec.name)
            .collect();
        assert_eq!(names, ["selected_skills", "max_skills"]);
    }

    #[test]
    fn selection_matches_and_falls_back() {
        let operator = SkillCallOperator::new("op");
        let skills = skills();
        let matched = operator.select(&skills, "beta");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "beta");
        let fallback = operator.select(&skills, "nothing matches");
        assert_eq!(fallback.len(), 3);
        let empty = operator.select(&skills, "");
        assert_eq!(empty.len(), 3);
    }

    #[test]
    fn curated_list_restricts_candidates_and_max_skills_bounds() {
        let operator = SkillCallOperator::new("op")
            .with_selected_skills(["alpha".to_owned(), "beta".to_owned()]);
        let skills = skills();
        let selected = operator.select(&skills, "gamma");
        assert_eq!(
            selected
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "beta"]
        );
        let bounded = SkillCallOperator::new("op").with_max_skills(1);
        assert_eq!(bounded.select(&skills, "").len(), 1);
        let none = SkillCallOperator::new("op").with_max_skills(0);
        assert!(none.select(&skills, "alpha").is_empty());
    }

    #[test]
    fn set_parameter_updates_and_fires() {
        let (callback, log) = recording_callback();
        let mut operator = SkillCallOperator::new("op").with_callback(callback);
        operator
            .set_parameter("selected_skills", json!(["alpha", "beta"]))
            .unwrap();
        assert_eq!(operator.selected_skills(), ["alpha", "beta"]);
        operator
            .set_parameter("selected_skills", json!("gamma"))
            .unwrap();
        assert_eq!(operator.selected_skills(), ["gamma"]);
        operator.set_parameter("max_skills", json!(10_000)).unwrap();
        assert_eq!(operator.max_skills(), MAX_MAX_SKILLS);
        let targets: Vec<String> = log
            .lock()
            .iter()
            .map(|(target, _)| target.clone())
            .collect();
        assert_eq!(
            targets,
            ["selected_skills", "selected_skills", "max_skills"]
        );
        assert!(!operator.set_parameter("unknown", json!(1)).unwrap().applied);
        assert!(
            !operator
                .set_parameter("selected_skills", json!([1]))
                .unwrap()
                .applied
        );
    }

    #[test]
    fn state_round_trips() {
        let mut operator = SkillCallOperator::new("op");
        operator
            .set_parameter("selected_skills", json!(["a"]))
            .unwrap();
        operator.set_parameter("max_skills", json!(2)).unwrap();
        let state = operator.get_state();
        let mut restored = SkillCallOperator::new("op");
        restored.load_state(state.clone()).unwrap();
        assert_eq!(restored.get_state(), state);
    }
}
