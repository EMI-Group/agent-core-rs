//! Process invocation of the Claude Code transport: argv, `--settings` and environment.
//!
//! See the `harness` module docs for the transport contract these helpers implement.

use std::collections::BTreeMap;

use openjiuwen_harness_protocol::{HarnessContext, JsonValue};

use crate::vendor_cli::PermissionPolicy;

use super::frames::compact_json;
use super::{
    ClaudeCodeHarnessConfig, ClaudeModelConfig, ClaudePermissionMode, ClaudeSystemPromptMode,
};

/// Environment variable injecting an external endpoint base URL.
const ANTHROPIC_BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";

/// Environment variable injecting an external endpoint authentication token.
const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";

/// Builds the CLI argv for one cycle.
pub(super) fn build_args(
    config: &ClaudeCodeHarnessConfig,
    context: &HarnessContext,
    policy: PermissionPolicy,
    resume: Option<&str>,
    session_id: &str,
) -> Vec<String> {
    let mut args: Vec<String> = [
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    .iter()
    .map(|arg| (*arg).to_owned())
    .collect();
    if config.include_partial_messages {
        args.push("--include-partial-messages".to_owned());
    }
    let permission_mode = if policy.interactive() {
        "default"
    } else {
        permission_mode_wire(config.permission_mode)
    };
    args.push("--permission-mode".to_owned());
    args.push(permission_mode.to_owned());
    if policy.interactive() {
        args.push("--permission-prompt-tool".to_owned());
        args.push("stdio".to_owned());
    }
    if let Some(model) = config.model.as_ref() {
        push_flag(&mut args, "--model", model.model.as_deref());
        push_flag(&mut args, "--effort", model.effort.as_deref());
    }
    if let Some(resume) = resume {
        args.push("--resume".to_owned());
        args.push(resume.to_owned());
    } else {
        args.push("--session-id".to_owned());
        args.push(session_id.to_owned());
    }
    if let Some(max_turns) = config.max_turns {
        args.push("--max-turns".to_owned());
        args.push(max_turns.to_string());
    }
    for dir in &config.add_dirs {
        args.push("--add-dir".to_owned());
        args.push(dir.to_string_lossy().into_owned());
    }
    if !context.system_prompt.is_empty() {
        match config.system_prompt_mode {
            ClaudeSystemPromptMode::Append => args.push("--append-system-prompt".to_owned()),
            ClaudeSystemPromptMode::Replace => args.push("--system-prompt".to_owned()),
        }
        args.push(context.system_prompt.clone());
    }
    if let Some(settings) = settings_json(config) {
        args.push("--settings".to_owned());
        args.push(settings);
    }
    args
}

/// Pushes `flag value` when `value` is set.
fn push_flag(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(flag.to_owned());
        args.push(value.to_owned());
    }
}

/// Returns the wire value of a permission mode.
fn permission_mode_wire(mode: ClaudePermissionMode) -> &'static str {
    match mode {
        ClaudePermissionMode::Default => "default",
        ClaudePermissionMode::AcceptEdits => "acceptEdits",
        ClaudePermissionMode::Plan => "plan",
        ClaudePermissionMode::BypassPermissions => "bypassPermissions",
        ClaudePermissionMode::DontAsk => "dontAsk",
        ClaudePermissionMode::Auto => "auto",
    }
}

/// Builds the compact `--settings` JSON, when anything must be injected.
fn settings_json(config: &ClaudeCodeHarnessConfig) -> Option<String> {
    let mut generated = BTreeMap::new();
    generated.extend(config.settings_env.clone());
    if let Some(ClaudeModelConfig {
        api_base, api_key, ..
    }) = config.model.as_ref()
    {
        if let Some(api_base) = api_base {
            generated.insert(ANTHROPIC_BASE_URL_ENV.to_owned(), api_base.clone());
        }
        if let Some(api_key) = api_key {
            generated.insert(
                ANTHROPIC_AUTH_TOKEN_ENV.to_owned(),
                api_key.expose_secret().to_owned(),
            );
        }
    }
    let mut object = config
        .settings
        .as_ref()
        .and_then(JsonValue::as_object)
        .cloned()
        .unwrap_or_default();
    if !generated.is_empty() {
        let explicit_env = object
            .get("env")
            .and_then(JsonValue::as_object)
            .cloned()
            .unwrap_or_default();
        let mut env = generated;
        env.extend(
            explicit_env
                .into_iter()
                .filter_map(|(key, value)| value.as_str().map(|value| (key, value.to_owned()))),
        );
        object.insert(
            "env".to_owned(),
            JsonValue::Object(
                env.into_iter()
                    .map(|(key, value)| (key, JsonValue::String(value)))
                    .collect(),
            ),
        );
    }
    if object.is_empty() {
        return None;
    }
    Some(compact_json(&JsonValue::Object(object)))
}

/// Merges the inherited environment, `config.env` and `context.env`.
pub(super) fn build_env(
    config: &ClaudeCodeHarnessConfig,
    context: &HarnessContext,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.extend(config.env.clone());
    env.extend(context.env.clone());
    env
}
