//! Local Python-style status codes for the `intent_detection` component.
//!
//! These are the component's own codes (`101050`–`101062`, Python's reserved
//! intent-detection block) and the builders that emit them as
//! `Error::workflow("[<code>] <message>")`. They stay local because `crate::error`
//! is the shared code table.

use openjiuwen_common::error::Error;

/// Input parameters are unusable (Python `COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR`).
pub const COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR: u32 = 101_050;
/// The model could not be created (Python `COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED`).
///
/// Reserved: the model is injected, so this port never builds one.
pub const COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED: u32 = 101_051;
/// The model call failed (Python `COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED`).
pub const COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED: u32 = 101_052;
/// A category display name is blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID: u32 = 101_053;
/// A category display name repeats another one (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE: u32 = 101_054;
/// `accept_language` is neither `"zh"` nor `"en"` (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID: u32 = 101_055;
/// A `keywords` key is not a category display name (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN: u32 = 101_056;
/// A `keywords` entry is blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_KEYWORD_INVALID: u32 = 101_057;
/// An `example_content` entry is blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID: u32 = 101_058;
/// `user_prompt` is present but blank (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_PROMPT_INVALID: u32 = 101_059;
/// The configuration is otherwise unusable (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_CONFIG_ERROR: u32 = 101_060;
/// The history configuration is unusable (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR: u32 = 101_061;
/// An [`IntentRouter`](super::IntentRouter) is unusable (Rust-only validation path).
pub const COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR: u32 = 101_062;

/// The input parameters are unusable (`101050`).
pub fn intent_detection_input_param_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR,
        format!("component intent_detection_input parameter error, reason: {reason}"),
    )
}

/// The model could not be created (`101051`).
///
/// Reserved: this port takes an already-built model, so nothing calls it.
pub fn intent_detection_llm_init_failed(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED,
        format!("component intent_detection_llm initialization failed, reason: {reason}"),
    )
}

/// The model call failed (`101052`).
pub fn intent_detection_invoke_call_failed(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED,
        format!("component intent_detection_invoke call failed, reason: {reason}"),
    )
}

/// A category display name is blank (`101053`).
pub fn intent_detection_category_name_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID,
        format!("component intent_detection_category_name parameter error, reason: {reason}"),
    )
}

/// A category display name repeats another one (`101054`).
pub fn intent_detection_category_name_duplicate(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE,
        format!("component intent_detection_category_name_list parameter error, reason: {reason}"),
    )
}

/// `accept_language` is unsupported (`101055`).
pub fn intent_detection_accept_language_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID,
        format!("component intent_detection_accept_language parameter error, reason: {reason}"),
    )
}

/// A `keywords` key is not a category display name (`101056`).
pub fn intent_detection_keyword_category_unknown(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN,
        format!("component intent_detection_keywords parameter error, reason: {reason}"),
    )
}

/// A `keywords` entry is blank (`101057`).
pub fn intent_detection_keyword_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_KEYWORD_INVALID,
        format!("component intent_detection_keywords entry parameter error, reason: {reason}"),
    )
}

/// An `example_content` entry is blank (`101058`).
pub fn intent_detection_example_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID,
        format!("component intent_detection_example_content parameter error, reason: {reason}"),
    )
}

/// `user_prompt` is present but blank (`101059`).
pub fn intent_detection_prompt_invalid(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_PROMPT_INVALID,
        format!("component intent_detection_user_prompt parameter error, reason: {reason}"),
    )
}

/// The configuration is otherwise unusable (`101060`).
pub fn intent_detection_config_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_CONFIG_ERROR,
        format!("component intent_detection config error, reason: {reason}"),
    )
}

/// The history configuration is unusable (`101061`).
pub fn intent_detection_history_config_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR,
        format!("component intent_detection_chat_history configuration error, reason: {reason}"),
    )
}

/// An [`IntentRouter`](super::IntentRouter) is unusable (`101062`).
pub fn intent_detection_router_config_error(reason: &str) -> Error {
    coded(
        COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR,
        format!("component intent_detection_router configuration error, reason: {reason}"),
    )
}

/// Renders one Python-shaped coded workflow failure.
fn coded(code: u32, message: impl Into<String>) -> Error {
    Error::workflow(format!("[{code}] {}", message.into()))
}
