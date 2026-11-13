//! Built-in `questioner` workflow component: a human-in-the-loop question/answer node.
//!
//! Ports `openjiuwen/core/workflow/components/llm/questioner_comp.py` (openJiuwen
//! v0.1.20) onto [`ComponentContext`]'s interaction seam. The component runs a small
//! state machine that lives in the session's component lane under `questioner_state`:
//!
//! - `start`: the first round. A `question_content` template is rendered from the node
//!   inputs, or key fields are extracted from the conversation with a model, which may
//!   produce a follow-up question;
//! - `user_interact`: a resumed round. The seeded answer becomes the new query, the
//!   fields are extracted again, and the round either asks once more or ends;
//! - `end`: the extracted fields (plus `user_response` and `question`) are returned.
//!
//! A round that still has an open question returns the interrupt error raised by
//! [`ComponentContext::interact`], so the run halts as `INPUT_REQUIRED` and
//! `Workflow::resume` re-runs the component with the seeded answers. The interaction id
//! is the component id, because the question payload is the rendered question string.
//!
//! # Deviations from Python
//!
//! - **No chat-history window injection.** `with_chat_history` and
//!   `chat_history_max_rounds` are carried for configuration parity, but the extraction
//!   conversation always contains only the current query as the user turn; there is no
//!   context-engine window on [`ComponentContext`] to read.
//! - **No `QuestionerExecutable`/composable split and no `model_id` Runner lookup.** The
//!   model is injected directly through [`QuestionerConfig::with_model`]; Python resolves
//!   `model_id` through `Runner.resource_mgr`.
//! - **State lives in the session component lane** under `questioner_state` (Python's
//!   node-session state), and a completed round clears it so the component can be
//!   re-entered by a later run.
//! - Templates other than `question_content` render a non-string value as its JSON text
//!   instead of Python's `repr`; strings, numbers, booleans and `null` render exactly
//!   like `str()`.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{Model, ModelRequestConfig};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;

mod config;
mod prompts;

pub use config::{FieldInfo, QuestionerConfig};
use prompts::{extraction_messages, format_continue_ask, format_template};

/// The session component-lane key holding the questioner state.
const QUESTIONER_STATE_KEY: &str = "questioner_state";
/// The only supported `response_type` (Python `ResponseType.ReplyDirectly`).
const RESPONSE_TYPE_REPLY_DIRECTLY: &str = "reply_directly";

/// The Python status code of [`input_param_error`].
pub const COMPONENT_QUESTIONER_INPUT_PARAM_ERROR: u32 = 101_070;
/// The Python status code of [`config_error`].
pub const COMPONENT_QUESTIONER_CONFIG_ERROR: u32 = 101_071;
/// The Python status code of [`input_invalid`].
pub const COMPONENT_QUESTIONER_INPUT_INVALID: u32 = 101_072;
/// The Python status code of [`state_init_failed`].
pub const COMPONENT_QUESTIONER_STATE_INIT_FAILED: u32 = 101_073;
/// The Python status code of [`runtime_error`].
pub const COMPONENT_QUESTIONER_RUNTIME_ERROR: u32 = 101_074;
/// The Python status code of [`invoke_call_failed`].
pub const COMPONENT_QUESTIONER_INVOKE_CALL_FAILED: u32 = 101_075;
/// The Python status code of [`execution_process_error`].
pub const COMPONENT_QUESTIONER_EXECUTION_PROCESS_ERROR: u32 = 101_076;

/// Renders a Python `"[<code>] <message>"` workflow error.
fn coded(code: u32, message: impl Into<String>) -> Error {
    Error::workflow(format!("[{code}] {}", message.into()))
}

/// The questioner inputs are malformed (Python `COMPONENT_QUESTIONER_INPUT_PARAM_ERROR`).
pub fn input_param_error(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_INPUT_PARAM_ERROR,
        format!("component questioner_input parameter error, reason: {error_msg}"),
    )
}

/// The questioner configuration is invalid (Python `COMPONENT_QUESTIONER_CONFIG_ERROR`).
pub fn config_error(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_CONFIG_ERROR,
        format!("component questioner config error, reason: {error_msg}"),
    )
}

/// The questioner input cannot produce a question (Python `COMPONENT_QUESTIONER_INPUT_INVALID`).
pub fn input_invalid(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_INPUT_INVALID,
        format!("component questioner_input is invalid, reason: {error_msg}"),
    )
}

/// The persisted questioner state could not be used
/// (Python `COMPONENT_QUESTIONER_STATE_INIT_FAILED`).
pub fn state_init_failed(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_STATE_INIT_FAILED,
        format!("component questioner_state initialization failed, reason: {error_msg}"),
    )
}

/// The questioner gave up at run time (Python `COMPONENT_QUESTIONER_RUNTIME_ERROR`).
pub fn runtime_error(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_RUNTIME_ERROR,
        format!("component questioner runtime error, reason: {error_msg}"),
    )
}

/// The extraction model call failed (Python `COMPONENT_QUESTIONER_INVOKE_CALL_FAILED`).
pub fn invoke_call_failed(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_INVOKE_CALL_FAILED,
        format!("component questioner_invoke call failed, reason: {error_msg}"),
    )
}

/// The extraction answer could not be processed
/// (Python `COMPONENT_QUESTIONER_EXECUTION_PROCESS_ERROR`).
pub fn execution_process_error(error_msg: impl fmt::Display) -> Error {
    coded(
        COMPONENT_QUESTIONER_EXECUTION_PROCESS_ERROR,
        format!("component questioner_execution process error, reason: {error_msg}"),
    )
}

/// The built-in `questioner` component.
///
/// The component is immutable during a run: every mutable value lives in the session's
/// component lane, so one instance serves concurrent runs. See the module documentation
/// for the state machine.
pub struct QuestionerComponent {
    config: QuestionerConfig,
}

impl fmt::Debug for QuestionerComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuestionerComponent")
            .field("config", &self.config)
            .finish()
    }
}

impl QuestionerComponent {
    /// Builds a component from `config`, validating it like Python's `QuestionerExecutable`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::Workflow`] carrying code `101071` for an unsupported
    /// `response_type`, an empty field list while extraction is enabled, an empty field
    /// name, or `max_response == 0`; code `101072` when there is neither a
    /// `question_content` template nor an extractable field list.
    pub fn new(config: QuestionerConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    /// The configuration this component was built from.
    #[must_use]
    pub fn config(&self) -> &QuestionerConfig {
        &self.config
    }

    /// Whether the configuration carries a non-empty question template.
    fn has_question_content(&self) -> bool {
        !self.config.question_content.is_empty()
    }

    /// Whether more configured fields still have to be extracted (Python `_need_extract_fields`).
    ///
    /// Field *names* are compared, never counts: a hallucinated key must not satisfy the
    /// check while a required field is still missing.
    fn need_extract_fields(&self, state: &QuestionerState) -> bool {
        self.config.extract_fields_from_response
            && self
                .config
                .field_names
                .iter()
                .any(|field| !state.extracted_key_fields.contains_key(&field.field_name))
    }

    /// Reads the persisted state; a missing or unreadable value starts a fresh round.
    async fn load_state(&self, ctx: &ComponentContext) -> Result<QuestionerState> {
        let Some(raw) = ctx.get_component_state(QUESTIONER_STATE_KEY).await? else {
            return Ok(QuestionerState::default());
        };
        match serde_json::from_value(raw) {
            Ok(state) => Ok(state),
            Err(error) => {
                tracing::debug!(error = %error, "questioner state is unreadable; starting a fresh round");
                Ok(QuestionerState::default())
            }
        }
    }

    /// Persists `state` in the component lane.
    async fn save_state(&self, ctx: &ComponentContext, state: &QuestionerState) -> Result<()> {
        ctx.update_component_state(serde_json::json!({ QUESTIONER_STATE_KEY: state }))
            .await
    }

    /// Clears the persisted state so a later run starts from `start`.
    ///
    /// The lane deep-merges, so a `null` value is what removes a stale key.
    async fn reset_state(&self, ctx: &ComponentContext) -> Result<()> {
        ctx.update_component_state(serde_json::json!({
            QUESTIONER_STATE_KEY: {
                "response_num": 0,
                "user_response": null,
                "question": null,
                "extracted_key_fields": null,
                "status": Status::Start,
            }
        }))
        .await
    }

    /// Runs one `start` round (Python `_handle_start_state`).
    async fn handle_start(
        &self,
        query: &Value,
        inputs: &Value,
        state: &mut QuestionerState,
    ) -> Result<OutputCache> {
        let mut output = OutputCache::default();
        if self.has_question_content() {
            let fields = user_fields(inputs);
            output.question = format_template(&self.config.question_content, &fields);
            state.question.clone_from(&output.question);
            state.status = Status::UserInteract;
            return Ok(output);
        }
        if self.need_extract_fields(state) {
            if self.run_extraction(query, state, &mut output).await? {
                state.question.clone_from(&output.question);
                state.status = Status::UserInteract;
            } else {
                state.status = Status::End;
            }
            return Ok(output);
        }
        Err(input_invalid(
            "question_content is empty and no extractable fields are configured",
        ))
    }

    /// Runs one resumed round (Python `_handle_user_interact_state`).
    async fn handle_user_interact(
        &self,
        ctx: &ComponentContext,
        state: &mut QuestionerState,
    ) -> Result<OutputCache> {
        // Python keeps the last answer of the round.
        let mut latest = None;
        while let Some(answer) = ctx.seeded_answer(ctx.component_id()) {
            latest = Some(answer);
        }
        let answer = match latest {
            Some(answer) => answer,
            // No feedback was queued: ask the stored question again, which halts the run
            // again (or delivers an answer that raced in).
            None => ctx.interact(Value::String(state.question.clone())).await?,
        };
        state.response_num = state.response_num.saturating_add(1);
        let mut output = OutputCache {
            user_response: answer.clone(),
            question: state.question.clone(),
            key_fields: Map::new(),
        };
        if self.has_question_content() && !self.need_extract_fields(state) {
            state.status = Status::End;
            return Ok(output);
        }
        if self.need_extract_fields(state) {
            if self.run_extraction(&answer, state, &mut output).await? {
                state.question.clone_from(&output.question);
                state.status = Status::UserInteract;
            } else {
                state.status = Status::End;
            }
            return Ok(output);
        }
        Err(input_invalid(
            "question_content is empty and no extractable fields are configured",
        ))
    }

    /// Extracts the configured fields, fills defaults, and decides whether to ask again.
    async fn run_extraction(
        &self,
        query: &Value,
        state: &mut QuestionerState,
        output: &mut OutputCache,
    ) -> Result<bool> {
        for (name, value) in self.invoke_for_extraction(query).await? {
            output.key_fields.insert(name.clone(), value.clone());
            state.extracted_key_fields.insert(name, value);
        }
        self.apply_defaults(state, output);
        self.check_continue_ask(state, output)
    }

    /// Calls the model and parses the extracted fields (Python `_invoke_llm_for_extraction`).
    async fn invoke_for_extraction(&self, query: &Value) -> Result<Map<String, Value>> {
        let model = self
            .config
            .model
            .as_ref()
            .ok_or_else(|| input_invalid("questioner field extraction requires a model"))?;
        let messages = extraction_messages(&self.config, query);
        let response = model
            .invoke(&messages, &ModelRequestConfig::default())
            .await
            .map_err(|error| {
                invoke_call_failed(format!("failed to invoke llm for extraction: {error}"))
            })?;
        let text = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        self.parse_extraction(&text)
    }

    /// Parses a model answer into the configured, type-converted fields.
    ///
    /// A non-JSON answer yields no fields (Python parity); a valid JSON value that is not
    /// an object is an execution error.
    fn parse_extraction(&self, text: &str) -> Result<Map<String, Value>> {
        let Ok(parsed) = serde_json::from_str::<Value>(&strip_code_fences(text)) else {
            tracing::debug!("questioner extraction answer is not JSON; treating it as empty");
            return Ok(Map::new());
        };
        let Some(object) = parsed.as_object() else {
            return Err(execution_process_error(
                "failed to parse json from llm response",
            ));
        };
        let mut result = Map::new();
        for (name, value) in object {
            if !is_valid_value(value) {
                continue;
            }
            let Some(field) = self
                .config
                .field_names
                .iter()
                .find(|field| field.field_name == *name)
            else {
                // Only configured fields may enter the result or the state.
                continue;
            };
            let Some(converted) = convert_type(value, &field.field_type) else {
                // A failed conversion is treated as "not extracted".
                continue;
            };
            // Python keeps only truthy values (`if v and k in configured_names`).
            if is_truthy(&converted) {
                result.insert(name.clone(), converted);
            }
        }
        Ok(result)
    }

    /// Adds each field's non-empty default to the output and the state
    /// (Python `_update_param_default_value`).
    fn apply_defaults(&self, state: &mut QuestionerState, output: &mut OutputCache) {
        for field in &self.config.field_names {
            let default = &field.default_value;
            if is_truthy(default) && !state.extracted_key_fields.contains_key(&field.field_name) {
                output
                    .key_fields
                    .insert(field.field_name.clone(), default.clone());
                state
                    .extracted_key_fields
                    .insert(field.field_name.clone(), default.clone());
            }
        }
    }

    /// Decides whether another question is needed (Python `_check_if_continue_ask`).
    ///
    /// # Errors
    ///
    /// Returns a `101074` error when required fields are still missing after
    /// `max_response` answers.
    fn check_continue_ask(
        &self,
        state: &QuestionerState,
        output: &mut OutputCache,
    ) -> Result<bool> {
        let missing: Vec<&FieldInfo> = self
            .config
            .field_names
            .iter()
            .filter(|field| {
                field.required && !state.extracted_key_fields.contains_key(&field.field_name)
            })
            .collect();
        let mut continue_ask = false;
        if !missing.is_empty() {
            if state.response_num >= u64::from(self.config.max_response) {
                return Err(runtime_error(
                    " max_response reached before all required fields were extracted",
                ));
            }
            output.question = format_continue_ask(&missing, &self.config.accept_language);
            continue_ask = true;
        }
        if continue_ask {
            output.key_fields.clear();
        } else {
            output.key_fields.clone_from(&state.extracted_key_fields);
        }
        Ok(continue_ask)
    }
}

#[async_trait::async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for QuestionerComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("questioner", "questioner")
            .with_description("asks the user for information and collects key fields")
            .with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "questioner"
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let query = validate_inputs(&inputs)?;
        let loaded = self.load_state(ctx).await?;
        // Python's `is_undergoing_interaction` check: only an in-progress interaction
        // resumes; every other state starts a fresh round (the START event leaves a
        // `start` state at `start` and a `user_interact` state at `user_interact`).
        let mut state = if loaded.status == Status::UserInteract {
            loaded
        } else {
            QuestionerState::default()
        };
        let output = match state.status {
            Status::Start => self.handle_start(&query, &inputs, &mut state).await?,
            Status::UserInteract => self.handle_user_interact(ctx, &mut state).await?,
            Status::End => end_state_output(&state),
        };
        if state.status == Status::UserInteract {
            self.save_state(ctx, &state).await?;
            // Python's `session.interact(invoke_result["question"])` tail: a seeded answer
            // is dropped here because the resumed round consumes it.
            ctx.interact(Value::String(output.question.clone())).await?;
        } else {
            // Clear the state when the component completes normally, so the component can
            // be re-entered by a later run (Python's reentrancy reset).
            self.reset_state(ctx).await?;
        }
        Ok(output.to_value())
    }
}

/// The round status (Python `ExecutionStatus`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    /// The first round, before anything was asked.
    #[default]
    Start,
    /// A question is waiting for an answer.
    UserInteract,
    /// The round collected everything it needs.
    End,
}

/// The persisted questioner state (Python `QuestionerState`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct QuestionerState {
    /// How many answers this round consumed.
    response_num: u64,
    /// The last answer the component saw.
    user_response: Value,
    /// The question currently asked.
    question: String,
    /// The fields extracted so far, by field name.
    extracted_key_fields: Map<String, Value>,
    /// Where the round stands.
    status: Status,
}

impl Default for QuestionerState {
    fn default() -> Self {
        Self {
            response_num: 0,
            user_response: Value::String(String::new()),
            question: String::new(),
            extracted_key_fields: Map::new(),
            status: Status::Start,
        }
    }
}

/// The values a round produced (Python `OutputCache`).
#[derive(Debug, Clone)]
struct OutputCache {
    /// The answer of the round, when one was consumed.
    user_response: Value,
    /// The question the round asked.
    question: String,
    /// The fields the round collected.
    key_fields: Map<String, Value>,
}

impl Default for OutputCache {
    fn default() -> Self {
        Self {
            user_response: Value::String(String::new()),
            question: String::new(),
            key_fields: Map::new(),
        }
    }
}

impl OutputCache {
    /// Projects the round into the component's output (Python `format_questioner_output`).
    ///
    /// Empty `user_response`/`question` values are omitted, like `model_dump(exclude_defaults=True)`.
    fn to_value(&self) -> Value {
        let mut map = self.key_fields.clone();
        if !is_empty_output_value(&self.user_response) {
            map.insert("user_response".to_owned(), self.user_response.clone());
        }
        if !self.question.is_empty() {
            map.insert("question".to_owned(), Value::String(self.question.clone()));
        }
        Value::Object(map)
    }
}

/// The output of a round that already finished (Python `_handle_end_state`).
fn end_state_output(state: &QuestionerState) -> OutputCache {
    OutputCache {
        user_response: state.user_response.clone(),
        question: state.question.clone(),
        key_fields: state.extracted_key_fields.clone(),
    }
}

/// Validates the configuration like Python's `QuestionerExecutable`.
fn validate_config(config: &QuestionerConfig) -> Result<()> {
    if config.response_type != RESPONSE_TYPE_REPLY_DIRECTLY {
        return Err(config_error(format!(
            "response type {} is invalid",
            config.response_type
        )));
    }
    if config.extract_fields_from_response && config.field_names.is_empty() {
        return Err(config_error("extracted key fields cannot be empty"));
    }
    for field in &config.field_names {
        if field.field_name.is_empty() {
            return Err(config_error("extracted key field name cannot be empty"));
        }
    }
    if config.max_response == 0 {
        return Err(config_error("max response must be greater than 0"));
    }
    if config.question_content.is_empty() && !config.extract_fields_from_response {
        return Err(input_invalid(
            "question_content is empty and no extractable fields are configured",
        ));
    }
    Ok(())
}

/// Validates the inputs and returns the query (Python `QuestionerUtils.validate_inputs`).
fn validate_inputs(inputs: &Value) -> Result<Value> {
    let Some(object) = inputs.as_object() else {
        return Err(input_param_error("inputs must be a JSON object"));
    };
    match object.get("query") {
        None | Some(Value::Null) => Ok(Value::String(String::new())),
        Some(value @ (Value::String(_) | Value::Object(_))) => Ok(value.clone()),
        Some(_) => Err(input_param_error(
            "query must be a string, an object, or null",
        )),
    }
}

/// The node inputs without the `query` key (Python `model_dump(exclude={"query"})`).
fn user_fields(inputs: &Value) -> Map<String, Value> {
    let mut fields = inputs.as_object().cloned().unwrap_or_default();
    fields.remove("query");
    fields
}

/// Renders one value like Python's `str()`.
///
/// Strings, numbers, booleans and `null` match Python exactly; arrays and objects use
/// their JSON text instead of Python's `repr` (see the module docs).
fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Null => "None".to_owned(),
        Value::Number(number) => number.to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Python truthiness, used for defaults and for keeping extracted values.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Whether an output value is omitted from the round output
/// (Python's `exclude_defaults` on `""`, `null`, `{}` and `[]`).
fn is_empty_output_value(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        Value::Bool(_) | Value::Number(_) => false,
    }
}

/// Whether the model returned a usable value (Python `QuestionerUtils.is_valid_value`).
fn is_valid_value(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(text) => {
            let trimmed = text.trim().to_lowercase();
            !trimmed.is_empty() && trimmed != "null" && trimmed != "none"
        }
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
        Value::Bool(_) | Value::Number(_) => true,
    }
}

/// Converts `value` to the field's declared type (Python `validate_and_convert_type`).
///
/// `None` means the conversion failed, so the field counts as not extracted.
fn convert_type(value: &Value, expected_type: &str) -> Option<Value> {
    match expected_type {
        "integer" => convert_integer(value),
        "number" => convert_number(value),
        "boolean" => convert_boolean(value),
        // `string` and every unknown type fall back to the string form.
        _ => Some(Value::String(python_str(value))),
    }
}

/// Converts a JSON value to an integer, rejecting booleans and fractional numbers.
#[allow(clippy::float_cmp)] // whole-number detection is exact, like Python's `value == int(value)`
fn convert_integer(value: &Value) -> Option<Value> {
    match value {
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                return Some(Value::from(int));
            }
            let float = number.as_f64()?;
            if float.fract() == 0.0 && float.abs() < 9.0e18 {
                #[allow(clippy::cast_possible_truncation)]
                let int = float as i64;
                return Some(Value::from(int));
            }
            None
        }
        Value::String(text) => text.trim().parse::<i64>().ok().map(Value::from),
        // Booleans would be treated as 0/1, and the remaining kinds are not integers.
        Value::Bool(_) | Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Converts a JSON value to a float, rejecting booleans.
fn convert_number(value: &Value) -> Option<Value> {
    let number = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(text) => text.trim().parse::<f64>().ok()?,
        Value::Bool(_) | Value::Null | Value::Array(_) | Value::Object(_) => return None,
    };
    serde_json::Number::from_f64(number).map(Value::Number)
}

/// Converts a JSON value to a boolean; only `bool` and `"true"`/`"false"` are accepted.
fn convert_boolean(value: &Value) -> Option<Value> {
    match value {
        Value::Bool(value) => Some(Value::Bool(*value)),
        Value::String(text) => match text.trim().to_lowercase().as_str() {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        Value::Null | Value::Number(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Strips a Markdown/`'''` code fence from a model answer (Python's two `re.sub` passes).
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    let without_open = strip_open_fence(trimmed).trim_end();
    strip_close_fence(without_open).trim().to_owned()
}

/// Strips a leading ```` ```json ```` / ```` ``` ```` / `'''json` / `'''` fence.
fn strip_open_fence(text: &str) -> &str {
    for fence in ["```json", "```", "'''json", "'''"] {
        if let Some(prefix) = text.get(..fence.len()) {
            if prefix.eq_ignore_ascii_case(fence) {
                return &text[fence.len()..];
            }
        }
    }
    text
}

/// Strips a trailing ```` ``` ```` / `'''` fence.
fn strip_close_fence(text: &str) -> &str {
    for fence in ["```", "'''"] {
        if let Some(suffix) = text.get(text.len().saturating_sub(fence.len())..) {
            if suffix == fence {
                return &text[..text.len() - fence.len()];
            }
        }
    }
    text
}
