//! Provider-neutral reasoning ("thinking") intent and its per-provider wire encoding.
//!
//! Ports `openjiuwen/core/foundation/llm/reasoning_profiles.py` (the declarative
//! provider/model capability table) and `.../reasoning.py` (the intent model, the
//! resolver, and the wire encoders).
//!
//! The layer has three stages, each independently usable:
//!
//! 1. **Intent** — [`ReasoningConfig`] is what a caller asks for in neutral terms:
//!    a [`ReasoningMode`], an optional labelled `effort`, and an optional numeric
//!    `budget_tokens`.
//! 2. **Resolution** — [`resolve_reasoning_plan`] looks the provider/model up in a
//!    static [`ReasoningProfile`] table and encodes the intent
//!    into a [`ReasoningPlan`]: the wire parameters the provider expects plus any
//!    warnings about intent the model family cannot express.
//! 3. **Application** — [`apply_reasoning_plan`] writes a plan into a request body
//!    without clobbering caller-supplied keys.
//!
//! Wire parameter names stay exactly what the vendors expect (`reasoning_effort`,
//! `thinking`, `enable_thinking`, `output_config`, `reasoning_split`,
//! `reasoning`) — this module never invents a neutral wire shape.
//!
//! ```
//! use openjiuwen_foundation::llm::reasoning::{
//!     apply_reasoning_plan, resolve_reasoning_plan, ReasoningConfig, ReasoningMode,
//! };
//! use serde_json::json;
//!
//! let intent = ReasoningConfig::new()
//!     .with_mode(ReasoningMode::Enabled)
//!     .with_effort("high");
//! let plan = resolve_reasoning_plan("openai", None, "gpt-5.1", &intent);
//! assert_eq!(plan.sdk_params["reasoning_effort"], json!("high"));
//!
//! let mut body = json!({"model": "gpt-5.1"});
//! apply_reasoning_plan(&mut body, &plan);
//! assert_eq!(body["reasoning_effort"], json!("high"));
//! ```
//!
//! Two deliberate differences from the Python module are documented where they
//! matter: [`apply_reasoning_plan`] merges the plan's extra-body parameters at the
//! **top level** of the request body (the Python version nests them under
//! `extra_body` because the `OpenAI` SDK does that merging itself), and an intent
//! that asks for nothing (`mode: auto` with no `effort` and no `budget_tokens`)
//! always resolves to an empty plan.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
mod apply;
mod plan;
mod profiles;
#[cfg(test)]
mod tests;
mod vllm;
pub use apply::{apply_reasoning_plan, apply_reasoning_plan_with};
pub use plan::{
    resolve_chat_completion_plan_with_endpoint, resolve_reasoning_plan, resolve_reasoning_plan_for,
    resolve_reasoning_plan_with_endpoint,
};
pub use profiles::{
    find_reasoning_profile, matches_pattern, provider_key_for, provider_key_for_endpoint,
    reasoning_capability, reasoning_capability_for, reasoning_capability_for_endpoint,
    standard_reasoning_capability, MODEL_REASONING_FALLBACKS, REASONING_PROFILES,
};
/// Provider-neutral reasoning mode.
///
/// `Auto` means "no explicit intent": the resolver only sends parameters a
/// provider requires unconditionally. `Enabled` and `Disabled` ask for thinking
/// to be switched on or off, which some families cannot express.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Let the provider keep its default; only explicit `effort`/`budget_tokens`
    /// produce parameters.
    #[default]
    Auto,
    /// Ask for reasoning/thinking to be enabled.
    Enabled,
    /// Ask for reasoning/thinking to be disabled.
    Disabled,
}
/// Provider-neutral reasoning/thinking request intent.
///
/// Mirrors the Python `ReasoningConfig`: every field defaults, so a partial JSON
/// object is a valid intent. Unknown fields are tolerated by the `serde` impls,
/// but [`ReasoningConfig::from_value`] treats an object with unknown keys as a
/// legacy raw payload rather than an intent (see its documentation).
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::{ReasoningConfig, ReasoningMode};
///
/// let intent = ReasoningConfig::new()
///     .with_mode(ReasoningMode::Enabled)
///     .with_effort("medium")
///     .with_budget_tokens(4096);
/// assert_eq!(intent.effort.as_deref(), Some("medium"));
/// assert_eq!(intent.budget_tokens, Some(4096));
/// assert!(ReasoningConfig::disabled().is_disabled());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReasoningConfig {
    /// What the caller wants reasoning to do.
    pub mode: ReasoningMode,
    /// Labelled intensity such as `"low"`, `"medium"`, `"high"`, `"xhigh"`, or
    /// `"max"`. The accepted vocabulary is per model family; use
    /// [`reasoning_capability`] to discover it.
    pub effort: Option<String>,
    /// Explicit thinking budget in tokens, for families that expose one instead
    /// of (or in addition to) a labelled effort.
    pub budget_tokens: Option<u32>,
}
impl ReasoningConfig {
    /// Creates an `auto` intent with no effort and no budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Overrides [`mode`](Self::mode).
    #[must_use]
    pub fn with_mode(mut self, mode: ReasoningMode) -> Self {
        self.mode = mode;
        self
    }
    /// Overrides [`effort`](Self::effort).
    #[must_use]
    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = Some(effort.into());
        self
    }
    /// Overrides [`budget_tokens`](Self::budget_tokens).
    #[must_use]
    pub fn with_budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }
    /// An intent that asks for reasoning to be switched off.
    #[must_use]
    pub fn disabled() -> Self {
        Self::default().with_mode(ReasoningMode::Disabled)
    }
    /// Whether this intent asks for nothing at all (`auto`, no effort, no budget).
    ///
    /// Such an intent never produces wire parameters, which keeps bodies
    /// byte-identical for callers that never opted into reasoning.
    #[must_use]
    pub fn is_no_intent(&self) -> bool {
        self.mode == ReasoningMode::Auto && self.effort.is_none() && self.budget_tokens.is_none()
    }
    /// Whether the intent disables reasoning, either through [`mode`](Self::mode)
    /// or through an `"off"`/`"none"` effort.
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.mode == ReasoningMode::Disabled || self.has_disabled_effort()
    }
    /// Whether [`effort`](Self::effort) is the `"off"`/`"none"` disable marker.
    #[must_use]
    pub fn has_disabled_effort(&self) -> bool {
        self.effort.as_deref().is_some_and(is_disabled_effort)
    }
    /// The effort to encode, or `None` when it is absent or a disable marker.
    #[must_use]
    pub fn effective_effort(&self) -> Option<&str> {
        self.effort
            .as_deref()
            .map(str::trim)
            .filter(|effort| !effort.is_empty() && !is_disabled_effort(effort))
    }
    /// Parses an optional intent out of a JSON value.
    ///
    /// Accepted shapes, in order:
    ///
    /// - `null` → `Ok(None)` (no intent).
    /// - an object whose keys are all from `mode`, `effort`, `budget_tokens` →
    ///   the parsed intent; an invalid `mode` or a negative/non-integer
    ///   `budget_tokens` is [`Error::Validation`].
    /// - an object with any other key → `Ok(None)`: the caller is using the
    ///   legacy raw reasoning payload, which the provider client forwards
    ///   verbatim instead of encoding it.
    /// - `true`/`false` → an `enabled`/`disabled` intent.
    /// - a string → the string as the effort; `"on"`/`"enabled"`, `"auto"`, and
    ///   `"off"`/`"none"`/`"disabled"` are recognized as shorthands for the mode
    ///   and produce no effort.
    /// - any other JSON shape (number, array) → `Ok(None)`.
    ///
    /// This mirrors the Python `_coerce_reasoning_config` helper, with the
    /// boolean and string shorthands as a convenience on top.
    pub fn from_value(value: &Value) -> Result<Option<Self>> {
        match value {
            Value::Null | Value::Number(_) | Value::Array(_) => Ok(None),
            Value::Bool(enabled) => Ok(Some(Self::new().with_mode(if *enabled {
                ReasoningMode::Enabled
            } else {
                ReasoningMode::Disabled
            }))),
            Value::String(shorthand) => Ok(Self::from_shorthand(shorthand)),
            Value::Object(object) => {
                if !object.keys().all(|key| is_intent_key(key)) {
                    return Ok(None);
                }
                serde_json::from_value(value.clone())
                    .map(Some)
                    .map_err(|error| {
                        Error::validation(format!("invalid reasoning intent: {error}"))
                    })
            }
        }
    }
    /// Interprets a string shorthand: a mode keyword or a labelled effort.
    fn from_shorthand(shorthand: &str) -> Option<Self> {
        let value = shorthand.trim().to_lowercase();
        match value.as_str() {
            "" => None,
            "auto" => Some(Self::new()),
            "on" | "enabled" => Some(Self::new().with_mode(ReasoningMode::Enabled)),
            off if is_disabled_effort(off) => Some(Self::disabled()),
            _ => Some(
                Self::new()
                    .with_mode(ReasoningMode::Enabled)
                    .with_effort(value),
            ),
        }
    }
}
/// The keys a JSON object may carry and still be read as a [`ReasoningConfig`].
const INTENT_KEYS: [&str; 3] = ["mode", "effort", "budget_tokens"];
/// Whether `key` belongs to the intent schema.
fn is_intent_key(key: &str) -> bool {
    INTENT_KEYS.contains(&key)
}
/// Effort spellings that mean "no reasoning".
const DISABLED_EFFORTS: [&str; 2] = ["off", "none"];
/// Whether `effort` is the `"off"`/`"none"` disable marker (case-insensitive).
#[must_use]
pub fn is_disabled_effort(effort: &str) -> bool {
    DISABLED_EFFORTS.contains(&effort.trim().to_lowercase().as_str())
}
/// Normalizes an effort spelling for capability comparison: trimmed, lowercased,
/// with `"none"` folded onto `"off"`.
#[must_use]
pub fn normalize_effort(effort: &str) -> String {
    let value = effort.trim().to_lowercase();
    if value == "none" {
        "off".to_owned()
    } else {
        value
    }
}
/// User-selectable reasoning values for one model family, plus a recommendation.
///
/// The `Default` implementation is the protocol default used when no profile
/// matches: `["off", "low", "medium", "high"]` (the OpenAI-compatible family
/// default) with no recommendation, matching
/// [`standard_reasoning_capability`]`(ReasoningProtocol::ChatCompletions)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningCapability {
    /// Accepted effort spellings, always lower case and always including `"off"`
    /// when the family can be switched off.
    #[serde(default)]
    pub options: Vec<String>,
    /// The level a UI should preselect, when the vendor documents one.
    #[serde(default)]
    pub recommended: Option<String>,
}
impl Default for ReasoningCapability {
    fn default() -> Self {
        Self::standard(ReasoningProtocol::ChatCompletions)
    }
}
impl ReasoningCapability {
    /// The protocol default: no profile matched this provider/model.
    #[must_use]
    pub fn standard(protocol: ReasoningProtocol) -> Self {
        let options: &[&str] = match protocol {
            ReasoningProtocol::AnthropicMessages => &["off", "low", "medium", "high", "max"],
            ReasoningProtocol::ChatCompletions | ReasoningProtocol::Responses => {
                &["off", "low", "medium", "high"]
            }
        };
        Self {
            options: options.iter().map(|option| (*option).to_owned()).collect(),
            recommended: None,
        }
    }
    /// Whether `effort` is one of [`options`](Self::options).
    ///
    /// The comparison is case-insensitive and folds `"none"` onto `"off"`.
    #[must_use]
    pub fn supports(&self, effort: &str) -> bool {
        let wanted = normalize_effort(effort);
        self.options
            .iter()
            .any(|option| normalize_effort(option) == wanted)
    }
}
/// Compile-time shape of a [`ReasoningCapability`], usable in the static profile
/// table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapabilitySpec {
    /// Accepted effort spellings.
    pub options: &'static [&'static str],
    /// The level a UI should preselect.
    pub recommended: Option<&'static str>,
}
impl CapabilitySpec {
    /// Builds a spec; `const` so profiles live in read-only static memory.
    #[must_use]
    pub const fn new(options: &'static [&'static str], recommended: Option<&'static str>) -> Self {
        Self {
            options,
            recommended,
        }
    }
    /// Whether this spec offers any user-selectable reasoning control.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }
    /// Whether `effort` is one of [`options`](Self::options) (case-insensitive,
    /// `"none"` folds onto `"off"`).
    #[must_use]
    pub fn supports(&self, effort: &str) -> bool {
        let wanted = normalize_effort(effort);
        self.options
            .iter()
            .any(|option| normalize_effort(option) == wanted)
    }
    /// Renders the owned capability handed to callers.
    #[must_use]
    pub fn to_capability(self) -> ReasoningCapability {
        ReasoningCapability {
            options: self
                .options
                .iter()
                .map(|option| (*option).to_owned())
                .collect(),
            recommended: self.recommended.map(str::to_owned),
        }
    }
    /// The documented options as one comma-separated string, for warnings.
    #[must_use]
    pub fn options_list(&self) -> String {
        self.options.join(", ")
    }
}
/// Wire protocol a reasoning plan is encoded for.
///
/// Mirrors the Python `LLMApiMode`: `chat_completions` for OpenAI-compatible
/// chat bodies, `responses` for the `OpenAI` Responses API, and
/// `anthropic_messages` for the Anthropic Messages API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningProtocol {
    /// OpenAI-compatible `POST /chat/completions`.
    #[default]
    ChatCompletions,
    /// The `OpenAI` Responses API.
    Responses,
    /// Anthropic `POST /v1/messages`.
    AnthropicMessages,
}
/// One provider-scoped model family and the encoders that speak to it.
///
/// Entries are data, not code: the resolver picks the first entry whose provider
/// matches and whose [`model_patterns`](Self::model_patterns) glob-match the model
/// id, then dispatches on [`wire`](Self::wire) (OpenAI-compatible) or
/// [`anthropic_wire`](Self::anthropic_wire) (Anthropic Messages).
///
/// The Python dataclass carries `openai_wire`/`anthropic_wire`; this port keeps
/// the same pair under `wire`/`anthropic_wire` and adds
/// [`wire_for`](Self::wire_for) / [`capability_for`](Self::capability_for) so the
/// protocol is a parameter rather than a branch at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReasoningProfile {
    /// Provider key this entry applies to, or `"*"` for the cross-provider model
    /// fallbacks.
    pub provider: &'static str,
    /// Case-insensitive model globs (`*` and `?`) this entry covers.
    pub model_patterns: &'static [&'static str],
    /// Capability offered over OpenAI-compatible protocols.
    pub capability: CapabilitySpec,
    /// Capability offered over the Anthropic Messages protocol, when it differs.
    pub anthropic_capability: Option<CapabilitySpec>,
    /// Encoder name used for OpenAI-compatible protocols.
    pub wire: &'static str,
    /// Encoder name used for the Anthropic Messages protocol.
    pub anthropic_wire: &'static str,
}
impl ReasoningProfile {
    /// Builds a profile; `const` so the whole table lives in read-only memory.
    #[must_use]
    pub const fn new(
        provider: &'static str,
        model_patterns: &'static [&'static str],
        capability: CapabilitySpec,
        wire: &'static str,
        anthropic_wire: &'static str,
    ) -> Self {
        Self {
            provider,
            model_patterns,
            capability,
            anthropic_capability: None,
            wire,
            anthropic_wire,
        }
    }
    /// Attaches a capability that differs over the Anthropic Messages protocol.
    #[must_use]
    pub const fn with_anthropic_capability(mut self, capability: CapabilitySpec) -> Self {
        self.anthropic_capability = Some(capability);
        self
    }
    /// The encoder name for `protocol` (Anthropic Messages uses
    /// [`anthropic_wire`](Self::anthropic_wire), everything else
    /// [`wire`](Self::wire)).
    #[must_use]
    pub fn wire_for(&self, protocol: ReasoningProtocol) -> &'static str {
        match protocol {
            ReasoningProtocol::AnthropicMessages => self.anthropic_wire,
            ReasoningProtocol::ChatCompletions | ReasoningProtocol::Responses => self.wire,
        }
    }
    /// The capability offered over `protocol`.
    #[must_use]
    pub fn capability_for(&self, protocol: ReasoningProtocol) -> ReasoningCapability {
        match protocol {
            ReasoningProtocol::AnthropicMessages => self.anthropic_capability,
            ReasoningProtocol::ChatCompletions | ReasoningProtocol::Responses => None,
        }
        .unwrap_or(self.capability)
        .to_capability()
    }
    /// The compile-time capability offered over `protocol`.
    #[must_use]
    pub fn capability_spec_for(&self, protocol: ReasoningProtocol) -> CapabilitySpec {
        match protocol {
            ReasoningProtocol::AnthropicMessages => self.anthropic_capability,
            ReasoningProtocol::ChatCompletions | ReasoningProtocol::Responses => None,
        }
        .unwrap_or(self.capability)
    }
}
/// Provider wire parameters produced by resolving one [`ReasoningConfig`].
///
/// `sdk_params` are top-level request-body fields; `extra_body` are vendor
/// extensions that belong in the body too (see [`apply_reasoning_plan`]).
/// `warnings` explain intent the model family cannot express; an empty plan with
/// no warnings means "nothing to send".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReasoningPlan {
    /// Top-level request-body parameters, for example `reasoning_effort`.
    pub sdk_params: Map<String, Value>,
    /// Vendor extension parameters, for example `thinking` or `enable_thinking`.
    pub extra_body: Map<String, Value>,
    /// Human-readable diagnostics about ignored intent.
    pub warnings: Vec<String>,
}
impl ReasoningPlan {
    /// An empty plan: nothing to send and nothing to warn about.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Whether the plan carries no wire parameters.
    ///
    /// A plan with warnings only is empty by this definition; the warnings still
    /// describe why nothing was sent.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sdk_params.is_empty() && self.extra_body.is_empty()
    }
    /// Whether resolution produced any warning.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
    /// The `sdk_params` keys, used by callers that log the applied plan.
    #[must_use]
    pub fn sdk_param_names(&self) -> Vec<&str> {
        self.sdk_params.keys().map(String::as_str).collect()
    }
    /// The `extra_body` keys, used by callers that log the applied plan.
    #[must_use]
    pub fn extra_body_names(&self) -> Vec<&str> {
        self.extra_body.keys().map(String::as_str).collect()
    }
}
#[cfg(test)]
mod inline_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn reasoning_mode_serde_uses_snake_case() {
        assert_eq!(
            serde_json::to_value(ReasoningMode::Auto).unwrap(),
            json!("auto")
        );
        assert_eq!(
            serde_json::to_value(ReasoningMode::Enabled).unwrap(),
            json!("enabled")
        );
        assert_eq!(
            serde_json::from_value::<ReasoningMode>(json!("disabled")).unwrap(),
            ReasoningMode::Disabled
        );
        assert_eq!(ReasoningMode::default(), ReasoningMode::Auto);
    }
    #[test]
    fn reasoning_config_deserializes_partial_and_tolerates_unknown_fields() {
        let intent: ReasoningConfig =
            serde_json::from_value(json!({"effort": "high", "vendor_note": 1})).unwrap();
        assert_eq!(intent.mode, ReasoningMode::Auto);
        assert_eq!(intent.effort.as_deref(), Some("high"));
        let empty: ReasoningConfig = serde_json::from_value(json!({})).unwrap();
        assert!(empty.is_no_intent());
    }
    #[test]
    fn from_value_reads_intent_objects() {
        let intent =
            ReasoningConfig::from_value(&json!({"mode": "enabled", "budget_tokens": 4096}))
                .unwrap()
                .expect("an intent object parses");
        assert_eq!(intent.mode, ReasoningMode::Enabled);
        assert_eq!(intent.budget_tokens, Some(4096));

        assert!(ReasoningConfig::from_value(&json!(null)).unwrap().is_none());
        assert!(ReasoningConfig::from_value(&json!(7)).unwrap().is_none());
        assert!(ReasoningConfig::from_value(&json!("")).unwrap().is_none());
    }
    #[test]
    fn from_value_leaves_legacy_payloads_alone() {
        // An object with a key outside the intent schema is a legacy raw payload.
        assert!(
            ReasoningConfig::from_value(&json!({"effort": "high", "enabled": true}))
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn from_value_reads_shorthands() {
        assert_eq!(
            ReasoningConfig::from_value(&json!(true)).unwrap(),
            Some(ReasoningConfig::new().with_mode(ReasoningMode::Enabled))
        );
        assert_eq!(
            ReasoningConfig::from_value(&json!(false)).unwrap(),
            Some(ReasoningConfig::disabled())
        );
        assert_eq!(
            ReasoningConfig::from_value(&json!("High")).unwrap(),
            Some(
                ReasoningConfig::new()
                    .with_mode(ReasoningMode::Enabled)
                    .with_effort("high")
            )
        );
        assert_eq!(
            ReasoningConfig::from_value(&json!("off")).unwrap(),
            Some(ReasoningConfig::disabled())
        );
        assert_eq!(
            ReasoningConfig::from_value(&json!("auto")).unwrap(),
            Some(ReasoningConfig::new())
        );
        assert_eq!(
            ReasoningConfig::from_value(&json!("enabled")).unwrap(),
            Some(ReasoningConfig::new().with_mode(ReasoningMode::Enabled))
        );
    }
    #[test]
    fn from_value_rejects_invalid_intent_objects() {
        let error = ReasoningConfig::from_value(&json!({"mode": "sometimes"})).unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        assert!(ReasoningConfig::from_value(&json!({"budget_tokens": -1})).is_err());
    }
    #[test]
    fn disabled_effort_helpers_are_case_insensitive() {
        assert!(is_disabled_effort(" OFF "));
        assert!(is_disabled_effort("none"));
        assert!(!is_disabled_effort("low"));
        let intent = ReasoningConfig::new().with_effort("Off");
        assert!(intent.is_disabled());
        assert_eq!(intent.effective_effort(), None);
        assert!(!intent.is_no_intent());
    }
    #[test]
    fn capability_default_is_the_openai_standard_set() {
        let capability = ReasoningCapability::default();
        assert_eq!(capability.options, ["off", "low", "medium", "high"]);
        assert_eq!(capability.recommended, None);
        assert!(capability.supports("HIGH"));
        assert!(capability.supports("none"));
        assert!(!capability.supports("max"));
        assert_eq!(
            ReasoningCapability::standard(ReasoningProtocol::AnthropicMessages).options,
            ["off", "low", "medium", "high", "max"]
        );
    }
    #[test]
    fn capability_spec_renders_and_compares() {
        let spec = CapabilitySpec::new(&["off", "high"], Some("high"));
        assert!(!spec.is_empty());
        assert!(spec.supports("None"));
        assert!(!spec.supports("low"));
        assert_eq!(spec.options_list(), "off, high");
        let capability = spec.to_capability();
        assert_eq!(capability.options, ["off", "high"]);
        assert_eq!(capability.recommended.as_deref(), Some("high"));
        assert!(CapabilitySpec::new(&[], None).is_empty());
    }
    #[test]
    fn plan_emptiness_ignores_warnings() {
        let mut plan = ReasoningPlan::new();
        assert!(plan.is_empty());
        assert!(!plan.has_warnings());
        plan.sdk_params
            .insert("reasoning_effort".to_owned(), json!("high"));
        assert!(!plan.is_empty());
        assert_eq!(plan.sdk_param_names(), ["reasoning_effort"]);
        plan.extra_body
            .insert("thinking".to_owned(), json!({"type": "enabled"}));
        assert_eq!(plan.extra_body_names(), ["thinking"]);
        let warnings = ReasoningPlan {
            warnings: vec!["ignored".to_owned()],
            ..ReasoningPlan::new()
        };
        assert!(warnings.is_empty());
        assert!(warnings.has_warnings());
    }
    #[test]
    fn plan_serde_round_trips() {
        let plan = ReasoningPlan {
            sdk_params: Map::from_iter([("reasoning_effort".to_owned(), json!("low"))]),
            extra_body: Map::from_iter([("enable_thinking".to_owned(), json!(true))]),
            warnings: vec!["note".to_owned()],
        };
        let value = serde_json::to_value(&plan).unwrap();
        let parsed: ReasoningPlan = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, plan);
    }
}
