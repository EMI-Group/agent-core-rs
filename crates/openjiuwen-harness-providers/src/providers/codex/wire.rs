//! Process invocation and app-server payloads of the Codex transport.
//!
//! See the `harness` module docs for the transport contract these helpers implement: the argv
//! (`codex app-server --listen stdio://` plus `-c key=value` overrides), the environment, and the
//! `thread/start` / `thread/resume` option object.

use std::collections::BTreeMap;
use std::path::Path;

use openjiuwen_harness_protocol::{
    json_object_to_value, HarnessContext, HostCapability, JsonObject, JsonValue, McpServerConfig,
    McpTransport,
};
use serde_json::json;

use super::{CodexHarnessConfig, CodexModelConfig};

/// Environment variable the external endpoint's API key is injected through.
///
/// Codex reads it because the generated `model_providers.<key>.env_key` override names it.
pub(super) const CODEX_API_KEY_ENV: &str = "OPENJIUWEN_CODEX_API_KEY";

/// Table key used when an endpoint names no provider of its own.
const DEFAULT_PROVIDER_KEY: &str = "openjiuwen";

/// Codex feature flag exposing the experimental `request_user_input` tool in default mode.
const USER_INPUT_FEATURE_OVERRIDE: &str = "features.default_mode_request_user_input=true";

/// `approvalPolicy` of a thread whose approvals are answered by the host.
const APPROVAL_POLICY_ON_REQUEST: &str = "on-request";

/// `approvalPolicy` of a thread that never asks for approval.
const APPROVAL_POLICY_NEVER: &str = "never";

/// `sandbox` mode of a thread Codex may modify its workspace in.
const SANDBOX_WORKSPACE_WRITE: &str = "workspace-write";

/// `sandbox` mode of a thread that runs without a sandbox.
const SANDBOX_FULL_ACCESS: &str = "danger-full-access";

/// Builds the CLI argv of one start/stop cycle.
///
/// The first three arguments select the JSON-RPC app-server on stdio; every later pair is one
/// Codex configuration override, with the value JSON-encoded so Codex's TOML parser reads it as a
/// string (or, for `args`, an array) instead of a bare token.
pub(super) fn build_args(config: &CodexHarnessConfig, context: &HarnessContext) -> Vec<String> {
    let mut args: Vec<String> = ["app-server", "--listen", "stdio://"]
        .iter()
        .map(|arg| (*arg).to_owned())
        .collect();
    for override_value in build_overrides(config, context) {
        args.push("-c".to_owned());
        args.push(override_value);
    }
    args
}

/// Builds the ordered `-c` overrides of one cycle.
fn build_overrides(config: &CodexHarnessConfig, context: &HarnessContext) -> Vec<String> {
    let mut overrides = Vec::new();
    if let Some(model) = config.model.as_ref().filter(|model| has_endpoint(model)) {
        overrides.extend(endpoint_overrides(model));
    }
    if context
        .host_capabilities
        .contains(&HostCapability::UserInput)
    {
        overrides.push(USER_INPUT_FEATURE_OVERRIDE.to_owned());
    }
    for server in &context.mcp_servers {
        overrides.extend(mcp_overrides(server));
    }
    overrides.extend(config.config_overrides.iter().cloned());
    overrides
}

/// Renders the `model_provider` / `model_providers.*` overrides of an external endpoint.
///
/// The provider table key decays a `-` into `_` because Codex reads the segment as a TOML bare
/// key; the label itself defaults to `openjiuwen` when the endpoint names none.
fn endpoint_overrides(model: &CodexModelConfig) -> Vec<String> {
    let provider = model
        .provider
        .as_deref()
        .filter(|provider| !provider.is_empty())
        .unwrap_or(DEFAULT_PROVIDER_KEY);
    let key = provider.replace('-', "_");
    let mut overrides = vec![format!("model_provider={}", json_string(provider))];
    overrides.push(format!(
        "model_providers.{key}.name={}",
        json_string(provider)
    ));
    if let Some(api_base) = model.api_base.as_deref() {
        overrides.push(format!(
            "model_providers.{key}.base_url={}",
            json_string(api_base)
        ));
    }
    if model.api_key.is_some() {
        overrides.push(format!(
            "model_providers.{key}.env_key={}",
            json_string(CODEX_API_KEY_ENV)
        ));
    }
    overrides
}

/// Renders the `mcp_servers.<key>.*` overrides of one host MCP server.
///
/// A stdio server is spawned by Codex from `command` (plus `args`); an HTTP server is reached
/// through `url`. In-process servers have no app-server representation: they are skipped with a
/// warning instead of failing the whole cycle.
fn mcp_overrides(server: &McpServerConfig) -> Vec<String> {
    let key = server.name.replace('-', "_");
    match server.transport {
        McpTransport::Stdio => {
            let Some((command, args)) = server.command.split_first() else {
                tracing::warn!(
                    server = %server.name,
                    "the Codex app-server needs a command for a stdio MCP server; skipping it"
                );
                return Vec::new();
            };
            let mut overrides = vec![format!(
                "mcp_servers.{key}.command={}",
                json_string(command)
            )];
            if !args.is_empty() {
                overrides.push(format!(
                    "mcp_servers.{key}.args={}",
                    json_string_array(args)
                ));
            }
            overrides
        }
        McpTransport::Http => match server.url.as_deref() {
            Some(url) => vec![format!("mcp_servers.{key}.url={}", json_string(url))],
            None => {
                tracing::warn!(
                    server = %server.name,
                    "the Codex app-server needs a url for an HTTP MCP server; skipping it"
                );
                Vec::new()
            }
        },
        McpTransport::InProcess => {
            tracing::warn!(
                server = %server.name,
                "the Codex app-server cannot mount an in-process MCP server; skipping it"
            );
            Vec::new()
        }
    }
}

/// Merges `config.env` and `context.env` and injects the endpoint API key.
///
/// The inherited process environment is applied by the spawn configuration, not here; the three
/// layers land in the order `config.env`, `context.env`, [`CODEX_API_KEY_ENV`], so the endpoint's
/// own key always wins over a stale one carried by the host context.
pub(super) fn build_env(
    config: &CodexHarnessConfig,
    context: &HarnessContext,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.extend(config.env.clone());
    env.extend(context.env.clone());
    if let Some(key) = config
        .model
        .as_ref()
        .and_then(|model| model.api_key.as_ref())
    {
        env.insert(CODEX_API_KEY_ENV.to_owned(), key.expose_secret().to_owned());
    }
    env
}

/// Builds the option object of `thread/start` and `thread/resume`.
///
/// `cwd` is the resolved working directory, `resume` the thread id a `thread/resume` targets.
pub(super) fn thread_params(
    config: &CodexHarnessConfig,
    context: &HarnessContext,
    cwd: Option<&Path>,
    resume: Option<&str>,
) -> JsonObject {
    let mut params = JsonObject::new();
    if let Some(thread_id) = resume {
        params.insert(
            "threadId".to_owned(),
            JsonValue::String(thread_id.to_owned()),
        );
    }
    params.insert("ephemeral".to_owned(), json!(false));
    if let Some(cwd) = cwd {
        params.insert(
            "cwd".to_owned(),
            JsonValue::String(cwd.to_string_lossy().into_owned()),
        );
    }
    if let Some(model) = config.model.as_ref() {
        if let Some(name) = model.model.as_deref() {
            params.insert("model".to_owned(), JsonValue::String(name.to_owned()));
        }
        if let Some(provider) = model.provider.as_deref() {
            params.insert(
                "modelProvider".to_owned(),
                JsonValue::String(provider.to_owned()),
            );
        }
    }
    if !context.system_prompt.is_empty() {
        params.insert(
            "developerInstructions".to_owned(),
            JsonValue::String(context.system_prompt.clone()),
        );
    }
    let bypass = bypasses_approvals(config);
    params.insert(
        "approvalPolicy".to_owned(),
        JsonValue::String(
            if bypass {
                APPROVAL_POLICY_NEVER
            } else {
                APPROVAL_POLICY_ON_REQUEST
            }
            .to_owned(),
        ),
    );
    params.insert(
        "sandbox".to_owned(),
        JsonValue::String(
            if bypass {
                SANDBOX_FULL_ACCESS
            } else {
                SANDBOX_WORKSPACE_WRITE
            }
            .to_owned(),
        ),
    );
    let mut thread_config = config.thread_config.clone();
    if let Some(effort) = config
        .model
        .as_ref()
        .and_then(|model| model.effort.as_deref())
    {
        thread_config.insert(
            "model_reasoning_effort".to_owned(),
            JsonValue::String(effort.to_owned()),
        );
    }
    params.insert(
        "config".to_owned(),
        JsonValue::Object(thread_config.into_iter().collect()),
    );
    params
}

/// Returns whether `model` targets an endpoint of its own instead of Codex's own login.
///
/// An endpoint is described by a provider label, a base URL or an API key; the key alone counts
/// because it is only meaningful together with an endpoint.
fn has_endpoint(model: &CodexModelConfig) -> bool {
    model.is_external() || model.api_key.is_some()
}

/// Returns whether the thread runs without Codex's own approval and sandbox machinery.
///
/// An external endpoint forces it: Codex's auto-review approval reviewer uses a built-in model
/// that cannot be redirected to another provider, so every review request against an endpoint
/// would fail. The host's own policy still governs the approvals the adapter answers.
fn bypasses_approvals(config: &CodexHarnessConfig) -> bool {
    config.bypass_approvals_and_sandbox || config.model.as_ref().is_some_and(has_endpoint)
}

/// Renders `value` as a JSON string, including its quotes.
fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Renders `values` as a JSON array of strings.
fn json_string_array(values: &[String]) -> String {
    serde_json::to_string(values).unwrap_or_default()
}

/// Returns the JSON form of an object, for the request `params` member.
///
/// Kept next to the builders so call sites never convert by hand.
pub(super) fn params_value(params: &JsonObject) -> JsonValue {
    json_object_to_value(params)
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use openjiuwen_harness_protocol::{HarnessContext, HostCapability, McpServerConfig};
    use serde_json::json;

    use super::{build_args, build_env, thread_params, CODEX_API_KEY_ENV};
    use crate::providers::codex::{CodexHarnessConfig, CodexModelConfig};

    /// Decodes one MCP server configuration.
    fn server(value: serde_json::Value) -> McpServerConfig {
        serde_json::from_value(value).expect("a valid MCP server configuration")
    }

    /// Context with no capabilities and no MCP servers.
    fn context() -> HarnessContext {
        HarnessContext::new("codex", "agent-1", "host-session", "be helpful")
    }

    /// Config with the given endpoint overrides.
    fn config_with(model: CodexModelConfig) -> CodexHarnessConfig {
        CodexHarnessConfig {
            model: Some(model),
            ..CodexHarnessConfig::default()
        }
    }

    #[test]
    fn the_argv_selects_the_stdio_app_server() {
        let args = build_args(&CodexHarnessConfig::default(), &context());
        assert_eq!(
            args[..3],
            [
                "app-server".to_owned(),
                "--listen".to_owned(),
                "stdio://".to_owned()
            ]
        );
        assert_eq!(args.len(), 3, "a bare configuration adds no override");
    }

    #[test]
    fn an_endpoint_renders_the_provider_table() {
        let config = config_with(CodexModelConfig {
            model: Some("gpt-5".to_owned()),
            provider: Some("my-endpoint".to_owned()),
            api_base: Some("https://example.test/v1".to_owned()),
            api_key: Some(openjiuwen_common::security::SecretString::new("sk-test")),
            effort: None,
        });
        let args = build_args(&config, &context());
        // `args[..3]` is the app-server selector; from then on `-c` and its value alternate.
        let overrides: Vec<&str> = args[3..].chunks(2).map(|pair| pair[1].as_str()).collect();
        assert_eq!(
            overrides,
            [
                r#"model_provider="my-endpoint""#,
                r#"model_providers.my_endpoint.name="my-endpoint""#,
                r#"model_providers.my_endpoint.base_url="https://example.test/v1""#,
                r#"model_providers.my_endpoint.env_key="OPENJIUWEN_CODEX_API_KEY""#,
            ]
        );
    }

    #[test]
    fn an_endpoint_without_a_provider_falls_back_to_openjiuwen() {
        let config = config_with(CodexModelConfig {
            api_base: Some("https://example.test/v1".to_owned()),
            ..CodexModelConfig::default()
        });
        let args = build_args(&config, &context());
        assert!(args.contains(&r#"model_provider="openjiuwen""#.to_owned()));
        assert!(args.contains(&r#"model_providers.openjiuwen.name="openjiuwen""#.to_owned()));
    }

    #[test]
    fn a_host_user_input_capability_enables_the_feature_flag() {
        let context = context().with_host_capabilities(BTreeSet::from([HostCapability::UserInput]));
        let args = build_args(&CodexHarnessConfig::default(), &context);
        assert!(args.contains(&"features.default_mode_request_user_input=true".to_owned()));
    }

    #[test]
    fn mcp_servers_render_command_args_and_url() {
        let context = context().with_mcp_servers(vec![
            server(json!({
                "name": "local-tools",
                "transport": "stdio",
                "command": ["node", "server.js"],
            })),
            server(json!({
                "name": "remote",
                "transport": "http",
                "url": "https://example.test/mcp",
            })),
        ]);
        let args = build_args(&CodexHarnessConfig::default(), &context);
        assert!(args.contains(&r#"mcp_servers.local_tools.command="node""#.to_owned()));
        assert!(args.contains(&r#"mcp_servers.local_tools.args=["server.js"]"#.to_owned()));
        assert!(args.contains(&r#"mcp_servers.remote.url="https://example.test/mcp""#.to_owned()));
    }

    #[test]
    fn configured_overrides_are_appended_verbatim() {
        let config = CodexHarnessConfig {
            config_overrides: vec!["model_reasoning_summary=\"concise\"".to_owned()],
            ..CodexHarnessConfig::default()
        };
        let args = build_args(&config, &context());
        assert_eq!(
            args.last().map(String::as_str),
            Some("model_reasoning_summary=\"concise\"")
        );
        assert_eq!(args[args.len() - 2], "-c");
    }

    #[test]
    fn the_environment_layers_context_and_the_endpoint_key() {
        let config = config_with(CodexModelConfig {
            api_key: Some(openjiuwen_common::security::SecretString::new("sk-test")),
            ..CodexModelConfig::default()
        });
        let mut context = context();
        context
            .env
            .insert("SHARED".to_owned(), "context".to_owned());
        let env = build_env(&config, &context);
        assert_eq!(env.get("SHARED").map(String::as_str), Some("context"));
        assert_eq!(
            env.get(CODEX_API_KEY_ENV).map(String::as_str),
            Some("sk-test")
        );
    }

    #[test]
    fn thread_params_carry_the_identity_and_the_host_policy() {
        let config = config_with(CodexModelConfig {
            model: Some("gpt-5".to_owned()),
            effort: Some("high".to_owned()),
            ..CodexModelConfig::default()
        });
        let params = thread_params(
            &config,
            &context(),
            Some(&PathBuf::from("/work")),
            Some("th-9"),
        );
        assert_eq!(params.get("threadId"), Some(&json!("th-9")));
        assert_eq!(params.get("ephemeral"), Some(&json!(false)));
        assert_eq!(params.get("cwd"), Some(&json!("/work")));
        assert_eq!(params.get("model"), Some(&json!("gpt-5")));
        assert_eq!(
            params.get("developerInstructions"),
            Some(&json!("be helpful"))
        );
        assert_eq!(params.get("approvalPolicy"), Some(&json!("on-request")));
        assert_eq!(params.get("sandbox"), Some(&json!("workspace-write")));
        let thread_config = params
            .get("config")
            .and_then(serde_json::Value::as_object)
            .expect("an object");
        assert_eq!(
            thread_config.get("model_reasoning_effort"),
            Some(&json!("high"))
        );
        assert_eq!(
            thread_config.get("model_reasoning_summary"),
            Some(&json!("detailed"))
        );
    }

    #[test]
    fn an_external_endpoint_bypasses_the_codex_approval_machinery() {
        let config = config_with(CodexModelConfig {
            api_base: Some("https://example.test/v1".to_owned()),
            ..CodexModelConfig::default()
        });
        let params = thread_params(&config, &context(), None, None);
        assert_eq!(params.get("approvalPolicy"), Some(&json!("never")));
        assert_eq!(params.get("sandbox"), Some(&json!("danger-full-access")));
        assert!(!params.contains_key("threadId"));
        assert!(!params.contains_key("cwd"));
    }
}
