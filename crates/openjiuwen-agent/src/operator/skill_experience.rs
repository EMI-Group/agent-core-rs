//! Skill-experience preview operator (`openjiuwen/core/operator/skill_call`).
//!
//! Ports `SkillExperienceOperator` from
//! `openjiuwen/core/operator/skill_call/base.py`: a preview-only parameter
//! handle that owns only the `updates_generated -> local_apply_completed`
//! stage of the experience lifecycle. Pending approval belongs to the caller's
//! [`PendingChangeStore`](super::staging::PendingChangeStore) and persistence to
//! the caller's [`SkillExperienceStore`](super::staging::SkillExperienceStore)
//! sink, so this operator never mutates state and never writes a record.
//!
//! The handle exposes exactly one tunable, [`EXPERIENCES_TARGET`], through two
//! entry points:
//!
//! - [`Operator::set_parameter`] keeps Python's direct-compatibility
//!   notification path: the record payload is normalised into a list, the
//!   parameter callback fires once, and the result reports a pending change
//!   without staging anything.
//! - [`PreviewableOperator::preview_update`] — also reached through
//!   [`Operator::apply_update`] — validates an append/merge pending-change
//!   update and returns the records a caller may stage.
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde_json::{json, Value};

use super::base::{
    ApplyResult, Operator, ParameterCallback, PreviewableOperator, TunableKind, TunableSpec,
    UpdateEffect, UpdateMode, UpdateValue,
};
use super::protocol::{EXPERIENCES_TARGET, LOCAL_APPLY_COMPLETED};

/// Metadata key naming the skill an experience record set belongs to.
const SKILL_NAME_KEY: &str = "skill_name";

/// Path of the experience payload inside the skill document.
const EXPERIENCES_PATH: &str = "content";

/// Preview-only parameter handle for one skill's experience records.
///
/// Identity follows Python: `operator_id` is `skill_experience_<skill_name>`.
pub struct SkillExperienceOperator {
    skill_name: String,
    operator_id: String,
    on_parameter_updated: Option<ParameterCallback>,
}

impl SkillExperienceOperator {
    /// Creates a handle for `skill_name`, optionally wired to `callback`.
    #[must_use]
    pub fn new(skill_name: impl Into<String>, callback: Option<ParameterCallback>) -> Self {
        let skill_name = skill_name.into();
        let operator_id = format!("skill_experience_{skill_name}");
        Self {
            skill_name,
            operator_id,
            on_parameter_updated: callback,
        }
    }

    /// Installs (or replaces) the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// The skill whose experiences this handle stages.
    #[must_use]
    pub fn skill_name(&self) -> &str {
        &self.skill_name
    }

    /// Metadata copied onto every produced result.
    fn result_metadata(&self, update: &UpdateValue) -> serde_json::Map<String, Value> {
        let mut metadata = serde_json::Map::new();
        for (key, value) in &update.metadata {
            metadata.insert(key.clone(), value.clone());
        }
        metadata.insert(SKILL_NAME_KEY.to_owned(), json!(self.skill_name));
        metadata
    }

    /// Fires the callback for `target`.
    fn fire(&self, target: &str, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(target, value);
        }
    }
}

/// Wraps a payload into a record list, mirroring Python's `isinstance` check.
///
/// A JSON array is taken as-is; every other value — including `null` — becomes
/// a one-element list.
fn records_from(payload: &Value) -> Vec<Value> {
    match payload {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    }
}

#[async_trait]
impl Operator for SkillExperienceOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        vec![
            TunableSpec::new(EXPERIENCES_TARGET, TunableKind::SkillExperience)
                .with_description("skill-experience records staged for approval")
                .with_path(EXPERIENCES_PATH)
                .with_constraint(json!({"type": "record"})),
        ]
    }

    fn get_state(&self) -> Value {
        json!({})
    }

    /// Notifies consumers on the direct-compatibility path, without staging.
    ///
    /// # Errors
    ///
    /// Never fails: unsupported targets and `null` payloads return a
    /// non-applied [`ApplyResult`] naming the offending input.
    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        if target != EXPERIENCES_TARGET {
            return Ok(ApplyResult::skipped(
                target,
                format!("unsupported target for SkillExperienceOperator: {target}"),
            ));
        }
        if value.is_null() {
            return Ok(ApplyResult::skipped(
                target,
                "skill-experience payload must not be null",
            ));
        }
        let records = records_from(&value);
        self.fire(EXPERIENCES_TARGET, &Value::Array(records.clone()));
        Ok(ApplyResult::applied(EXPERIENCES_TARGET)
            .with_operator_id(self.operator_id())
            .with_effect(UpdateEffect::PendingChange)
            .with_value(value)
            .with_records(records)
            .with_metadata_entry(SKILL_NAME_KEY, json!(self.skill_name)))
    }

    fn load_state(&mut self, _state: Value) -> Result<()> {
        Ok(())
    }

    /// Routes to [`PreviewableOperator::preview_update`]; state never changes.
    ///
    /// # Errors
    ///
    /// Never fails for a malformed update: a rejection is an `Ok` result.
    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult> {
        self.preview_update(target, &update)
    }
}

impl PreviewableOperator for SkillExperienceOperator {
    /// Returns the local preview for `update` without mutating the handle.
    ///
    /// # Errors
    ///
    /// Never fails for a malformed update: rejections are `Ok` results carrying
    /// [`ApplyResult::applied`] `= false` and an explanation in
    /// [`ApplyResult::errors`].
    fn preview_update(&self, target: &str, update: &UpdateValue) -> Result<ApplyResult> {
        if target != EXPERIENCES_TARGET {
            return Ok(ApplyResult::rejection(
                self.operator_id(),
                target,
                update,
                vec![format!(
                    "unsupported target for SkillExperienceOperator: {target}"
                )],
            ));
        }
        if !update.is_pending_change()
            || !matches!(update.mode, UpdateMode::Append | UpdateMode::Merge)
        {
            return Ok(ApplyResult::rejection(
                self.operator_id(),
                target,
                update,
                vec![format!(
                    "unsupported update mode/effect for SkillExperienceOperator: {}/{}",
                    update.mode(),
                    update.effect.as_str()
                )],
            ));
        }
        let records = records_from(&update.payload);
        let mut result = ApplyResult::applied(EXPERIENCES_TARGET)
            .with_operator_id(self.operator_id())
            .with_mode(update.mode)
            .with_effect(update.effect)
            .with_value(update.payload.clone())
            .with_records(records)
            .with_lifecycle_stage(LOCAL_APPLY_COMPLETED);
        result.applied = !result.records.is_empty();
        result.change_type.clone_from(&update.change_type);
        for (key, value) in self.result_metadata(update) {
            result.metadata.insert(key, value);
        }
        Ok(result)
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

    type Recorded = Arc<parking_lot::Mutex<Vec<(String, Value)>>>;

    fn recording_callback() -> (ParameterCallback, Recorded) {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: ParameterCallback = Arc::new(move |target: &str, value: &Value| {
            sink.lock().push((target.to_owned(), value.clone()));
        });
        (callback, log)
    }

    fn pending_update(payload: Value, mode: UpdateMode) -> UpdateValue {
        UpdateValue::new(payload, mode).with_effect(UpdateEffect::PendingChange)
    }

    #[test]
    fn operator_identity_tunable_and_state() {
        let operator = SkillExperienceOperator::new("alpha", None);
        assert_eq!(operator.operator_id(), "skill_experience_alpha");
        assert_eq!(operator.skill_name(), "alpha");
        assert_eq!(operator.get_state(), json!({}));

        let tunables = operator.get_tunables();
        assert_eq!(tunables.len(), 1);
        let spec = &tunables[0];
        assert_eq!(spec.name, EXPERIENCES_TARGET);
        assert_eq!(spec.kind, TunableKind::SkillExperience);
        assert_eq!(spec.kind.as_str(), "skill_experience");
        assert_eq!(spec.path, "content");
        assert_eq!(spec.constraint, Some(json!({"type": "record"})));
        assert!(!spec.frozen);

        let mut operator = SkillExperienceOperator::new("alpha", None);
        operator
            .load_state(json!({"experiences": [{"text": "ignored"}]}))
            .unwrap();
        assert_eq!(operator.get_state(), json!({}));
    }

    #[test]
    fn preview_rejects_unsupported_target() {
        let operator = SkillExperienceOperator::new("alpha", None);
        let update = pending_update(json!([{"text": "x"}]), UpdateMode::Append)
            .with_metadata_entry("source", json!("online"));
        let result = operator.preview_update("other", &update).unwrap();

        assert!(!result.applied);
        assert!(!result.ok());
        assert_eq!(result.operator_id, "skill_experience_alpha");
        assert_eq!(result.target, "other");
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("other"));
        assert!(result.records.is_empty());
        assert!(result.lifecycle_stage.is_none());
        assert_eq!(result.metadata.get("source"), Some(&json!("online")));
    }

    #[test]
    fn preview_rejects_state_effect_and_replace_mode() {
        let operator = SkillExperienceOperator::new("alpha", None);

        let state_effect = UpdateValue::append(json!([{"text": "x"}]));
        let result = operator
            .preview_update(EXPERIENCES_TARGET, &state_effect)
            .unwrap();
        assert!(!result.applied);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("append/state"));

        let replace = pending_update(json!([{"text": "x"}]), UpdateMode::Replace);
        let result = operator
            .preview_update(EXPERIENCES_TARGET, &replace)
            .unwrap();
        assert!(!result.applied);
        assert!(result.errors[0].contains("replace/pending_change"));
    }

    #[test]
    fn preview_accepts_append_and_merge() {
        for mode in [UpdateMode::Append, UpdateMode::Merge] {
            let operator = SkillExperienceOperator::new("alpha", None);
            let update = pending_update(json!([{"text": "x"}, {"text": "y"}]), mode)
                .with_change_type("experience_entry")
                .with_metadata_entry("source", json!("offline"));
            let result = operator
                .preview_update(EXPERIENCES_TARGET, &update)
                .unwrap();

            assert!(result.ok());
            assert!(result.applied);
            assert_eq!(result.mode, mode);
            assert_eq!(result.effect, UpdateEffect::PendingChange);
            assert_eq!(result.value, json!([{"text": "x"}, {"text": "y"}]));
            assert_eq!(
                result.records,
                vec![json!({"text": "x"}), json!({"text": "y"})]
            );
            assert_eq!(result.change_type.as_deref(), Some("experience_entry"));
            assert_eq!(
                result.lifecycle_stage.as_deref(),
                Some(LOCAL_APPLY_COMPLETED)
            );
            assert_eq!(result.metadata.get("skill_name"), Some(&json!("alpha")));
            assert_eq!(result.metadata.get("source"), Some(&json!("offline")));
            assert!(result.errors.is_empty());
        }
    }

    #[test]
    fn preview_wraps_single_payload_and_empty_list() {
        let operator = SkillExperienceOperator::new("alpha", None);

        let single = operator
            .preview_update(
                EXPERIENCES_TARGET,
                &pending_update(json!({"text": "x"}), UpdateMode::Append),
            )
            .unwrap();
        assert!(single.applied);
        assert_eq!(single.records, vec![json!({"text": "x"})]);

        let empty = operator
            .preview_update(
                EXPERIENCES_TARGET,
                &pending_update(json!([]), UpdateMode::Append),
            )
            .unwrap();
        assert!(!empty.applied);
        assert!(empty.errors.is_empty());
        assert!(empty.records.is_empty());
        assert_eq!(
            empty.lifecycle_stage.as_deref(),
            Some(LOCAL_APPLY_COMPLETED)
        );

        // Python wraps any non-list payload, so `null` becomes `[null]` and
        // reports `applied` because the wrapped list is non-empty. This is a
        // divergence in spirit only: the Python `isinstance(list)` check
        // behaves identically.
        let null = operator
            .preview_update(
                EXPERIENCES_TARGET,
                &pending_update(Value::Null, UpdateMode::Append),
            )
            .unwrap();
        assert!(null.applied);
        assert_eq!(null.records, vec![Value::Null]);
    }

    #[test]
    fn set_parameter_notifies_once_with_wrapped_list() {
        let (callback, log) = recording_callback();
        let mut operator = SkillExperienceOperator::new("alpha", Some(callback));

        let result = operator
            .set_parameter(EXPERIENCES_TARGET, json!({"text": "x"}))
            .unwrap();
        assert!(result.applied);
        assert!(result.ok());
        assert_eq!(result.operator_id, "skill_experience_alpha");
        assert_eq!(result.effect, UpdateEffect::PendingChange);
        assert_eq!(result.value, json!({"text": "x"}));
        assert_eq!(result.records, vec![json!({"text": "x"})]);
        assert_eq!(result.metadata.get("skill_name"), Some(&json!("alpha")));
        assert_eq!(
            *log.lock(),
            vec![(EXPERIENCES_TARGET.to_owned(), json!([{"text": "x"}]))]
        );
        assert_eq!(operator.get_state(), json!({}));

        // A list payload is passed through unchanged and still fires once.
        let result = operator
            .set_parameter(EXPERIENCES_TARGET, json!([{"text": "a"}, {"text": "b"}]))
            .unwrap();
        assert!(result.applied);
        assert_eq!(log.lock().len(), 2);
        assert_eq!(
            log.lock()[1],
            (
                EXPERIENCES_TARGET.to_owned(),
                json!([{"text": "a"}, {"text": "b"}])
            )
        );

        // `null` and unknown targets are ignored: no callback, not applied.
        let null = operator
            .set_parameter(EXPERIENCES_TARGET, Value::Null)
            .unwrap();
        assert!(!null.applied);
        assert!(null.reason.contains("null"));
        let wrong = operator
            .set_parameter("other", json!([{"text": "x"}]))
            .unwrap();
        assert!(!wrong.applied);
        assert!(wrong.reason.contains("other"));
        assert_eq!(log.lock().len(), 2);
    }

    #[test]
    fn apply_update_routes_to_preview_without_mutating() {
        let (callback, log) = recording_callback();
        let mut operator = SkillExperienceOperator::new("alpha", Some(callback));
        let update = pending_update(json!([{"text": "x"}]), UpdateMode::Merge)
            .with_change_type("skill_experience_entry");

        let expected = operator
            .preview_update(EXPERIENCES_TARGET, &update)
            .unwrap();
        let result = operator
            .apply_update(EXPERIENCES_TARGET, update.clone())
            .unwrap();
        assert_eq!(result, expected);
        assert!(result.ok());
        assert_eq!(operator.get_state(), json!({}));
        assert!(log.lock().is_empty(), "preview must not notify consumers");

        let rejected = operator
            .apply_update("other", pending_update(json!([]), UpdateMode::Append))
            .unwrap();
        assert!(!rejected.applied);
        assert_eq!(operator.get_state(), json!({}));
    }
}
