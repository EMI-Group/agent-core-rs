//! Writing a resolved [`ReasoningPlan`] into a request body.
//!
//! Ports `apply_reasoning_plan` from `openjiuwen/core/foundation/llm/reasoning.py`.
//! The Python helper fills an SDK-kwargs dict and nests vendor extensions under an
//! `extra_body` key, which the `OpenAI` SDK merges into the wire body itself. This
//! crate builds the final body directly, so both parameter sets are applied at the
//! top level and the caller's keys are never lost.
use super::ReasoningPlan;
use serde_json::{Map, Value};
/// Writes `plan` into `body`, keeping every key the caller already set.
///
/// - `sdk_params` are inserted only when the body does not already carry a value
///   under that key.
/// - `extra_body` is merged one level deep: when both the existing value and the
///   plan's value are objects their keys are merged (the plan's keys win), so a
///   vendor object such as `thinking` keeps unrelated caller fields.
/// - Non-object bodies and empty plans are a no-op, so a body built without a
///   reasoning intent stays byte-identical.
///
/// Note that a caller who wants to *override* the plan can pass the same keys
/// through `ModelRequestConfig::extra`, which the OpenAI-compatible request
/// builder merges after this function runs.
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::{
///     apply_reasoning_plan, ReasoningPlan,
/// };
/// use serde_json::{json, Map, Value};
///
/// let mut plan = ReasoningPlan::new();
/// plan.extra_body
///     .insert("thinking".to_owned(), json!({"type": "enabled"}));
///
/// let mut body = json!({"model": "m", "thinking": {"type": "disabled", "extra": 1}});
/// apply_reasoning_plan(&mut body, &plan);
/// assert_eq!(body["thinking"], json!({"type": "enabled", "extra": 1}));
/// assert_eq!(body["model"], json!("m"));
///
/// // An empty plan never touches the body.
/// let mut untouched = json!({"model": "m"});
/// apply_reasoning_plan(&mut untouched, &ReasoningPlan::new());
/// assert_eq!(untouched, json!({"model": "m"}));
/// ```
pub fn apply_reasoning_plan(body: &mut Value, plan: &ReasoningPlan) {
    apply_reasoning_plan_with(body, plan, false);
}
/// Writes `plan` into `body`, optionally letting the plan overwrite caller keys.
///
/// With `override_existing` set, `sdk_params` replace a value the body already
/// carried under the same key. `extra_body` merging is unchanged: the plan's keys
/// always win, because they are the encoded reasoning intent.
///
/// Non-object bodies and empty plans are a no-op, exactly like
/// [`apply_reasoning_plan`].
pub fn apply_reasoning_plan_with(body: &mut Value, plan: &ReasoningPlan, override_existing: bool) {
    if plan.is_empty() {
        return;
    }
    let Value::Object(object) = body else {
        return;
    };
    for (key, value) in &plan.sdk_params {
        if override_existing || !object.contains_key(key) {
            object.insert(key.clone(), value.clone());
        }
    }
    for (key, value) in &plan.extra_body {
        apply_extra_body(object, key, value);
    }
}
/// Merges one vendor extension into the body, one object level deep.
fn apply_extra_body(object: &mut Map<String, Value>, key: &str, value: &Value) {
    match (object.get_mut(key), value) {
        (Some(Value::Object(existing)), Value::Object(incoming)) => {
            for (nested_key, nested_value) in incoming {
                existing.insert(nested_key.clone(), nested_value.clone());
            }
        }
        _ => {
            object.insert(key.to_owned(), value.clone());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::super::{ReasoningConfig, ReasoningMode};
    use super::*;
    use serde_json::json;
    fn plan(sdk: &[(&str, Value)], extra: &[(&str, Value)]) -> ReasoningPlan {
        ReasoningPlan {
            sdk_params: sdk
                .iter()
                .map(|(key, value)| ((*key).to_owned(), value.clone()))
                .collect(),
            extra_body: extra
                .iter()
                .map(|(key, value)| ((*key).to_owned(), value.clone()))
                .collect(),
            warnings: Vec::new(),
        }
    }
    #[test]
    fn empty_plan_leaves_the_body_untouched() {
        let mut body = json!({"model": "m", "messages": []});
        let before = body.clone();
        apply_reasoning_plan(&mut body, &ReasoningPlan::new());
        assert_eq!(body, before);
        apply_reasoning_plan_with(&mut body, &ReasoningPlan::new(), true);
        assert_eq!(body, before);
    }
    #[test]
    fn sdk_params_fill_missing_keys_only() {
        let plan = plan(&[("reasoning_effort", json!("high"))], &[]);
        let mut body = json!({"model": "m", "reasoning_effort": "low"});
        apply_reasoning_plan(&mut body, &plan);
        assert_eq!(body["reasoning_effort"], json!("low"));
        let mut body = json!({"model": "m"});
        apply_reasoning_plan(&mut body, &plan);
        assert_eq!(body["reasoning_effort"], json!("high"));
    }
    #[test]
    fn override_existing_replaces_sdk_params() {
        let plan = plan(&[("reasoning_effort", json!("high"))], &[]);
        let mut body = json!({"model": "m", "reasoning_effort": "low"});
        apply_reasoning_plan_with(&mut body, &plan, true);
        assert_eq!(body["reasoning_effort"], json!("high"));
    }
    #[test]
    fn extra_body_merges_one_object_level() {
        let plan = plan(
            &[],
            &[(
                "thinking",
                json!({"type": "enabled", "budget_tokens": 2048}),
            )],
        );
        let mut body = json!({"model": "m", "thinking": {"type": "disabled", "caller": true}});
        apply_reasoning_plan(&mut body, &plan);
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048, "caller": true})
        );
        assert_eq!(body["model"], json!("m"));
    }
    #[test]
    fn extra_body_never_clobbers_other_top_level_keys() {
        let plan = plan(&[], &[("enable_thinking", json!(true))]);
        let mut body = json!({
            "model": "m",
            "temperature": 0.2,
            "stop": ["x"],
            "extra_body": {"caller": 1}
        });
        apply_reasoning_plan(&mut body, &plan);
        assert_eq!(body["enable_thinking"], json!(true));
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(body["stop"], json!(["x"]));
        assert_eq!(body["extra_body"], json!({"caller": 1}));
    }
    #[test]
    fn scalar_existing_values_are_replaced_by_plan_objects() {
        let plan = plan(&[], &[("reasoning", json!({"effort": "high"}))]);
        let mut body = json!({"model": "m", "reasoning": "caller-value"});
        apply_reasoning_plan(&mut body, &plan);
        assert_eq!(body["reasoning"], json!({"effort": "high"}));
    }
    #[test]
    fn non_object_bodies_are_a_no_op() {
        let plan = plan(
            &[("reasoning_effort", json!("high"))],
            &[("enable_thinking", json!(true))],
        );
        for mut body in [json!(null), json!("text"), json!([1, 2]), json!(7)] {
            let before = body.clone();
            apply_reasoning_plan(&mut body, &plan);
            assert_eq!(body, before);
        }
    }
    #[test]
    fn applying_a_resolved_plan_keeps_caller_keys() {
        let intent = ReasoningConfig::new()
            .with_mode(ReasoningMode::Enabled)
            .with_effort("high");
        let resolved = super::super::resolve_reasoning_plan("openai", None, "gpt-5.1", &intent);
        let mut body = json!({"model": "gpt-5.1", "stream": false, "temperature": 0.7});
        apply_reasoning_plan(&mut body, &resolved);
        assert_eq!(body["reasoning_effort"], json!("high"));
        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["stream"], json!(false));
    }
}
