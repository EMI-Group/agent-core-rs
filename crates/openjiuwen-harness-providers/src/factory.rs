//! The provider registry: name resolution, harness creation, model-configuration gates, and the
//! manifest-free harness-context builder.
//!
//! Python keeps a module-level `dict[str, HarnessProvider]` in
//! `openjiuwen.harness_providers.factory` and probes each entry with `isinstance` at call time.
//! Rust registers one [`ProviderSpec`] per name instead (see [`crate::provider`]) and keeps the
//! built-in table in [`crate::providers`], so resolution is a two-step lookup: the declared names
//! are served by the compile-time table, everything else by this crate's process-global registry.
//!
//! # Resolution order
//!
//! 1. [`crate::providers::builtin_provider`] — every declared name (`native`, `native_v2`,
//!    `claudecode`, `codex`, `dsh`) resolves here, including the reasons a build cannot serve it.
//!    Built-ins always win: a registration can never shadow them, and
//!    [`register_harness_provider`] rejects built-in names outright.
//! 2. The process-global registry, in insertion order.
//! 3. [`Error::NotFound`] naming every available provider, in Python's message wording.
//!
//! # Registry state
//!
//! The registry is process-global (Python's module dict), so tests must use unique provider names
//! instead of resetting it. It is a [`std::sync::RwLock`] over an [`IndexMap`] — insertion order is
//! part of the public contract of [`available_providers`] and [`provider_descriptors`] — guarded by
//! a [`std::sync::OnceLock`]. Poisoned locks are recovered (a provider factory that panics while
//! registering must not brick the registry for the rest of the process), and the lock is never held
//! across an `.await` (there are none in this module) or while a provider factory runs.
//!
//! # Deviations from Python
//!
//! - Python's `create_harness` and `build_harness_context` take an agent-template manifest; this
//!   module takes the provider name plus explicit [`HarnessContextOptions`] because the Rust
//!   harness crate has no manifest loader.
//! - [`build_harness_context`] silently *drops* MCP servers for the `native` provider (see its
//!   documentation) instead of raising.
//! - Provider descriptors distinguish [`Error::NotFound`] (unknown name), [`Error::Unsupported`]
//!   (declared but not implemented) and [`Error::Validation`] (invalid input) where Python raises a
//!   single `ValueError`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use indexmap::IndexMap;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness::protocol::{HARNESS_NAME, PROVIDER_VERSION};
use openjiuwen_harness_protocol::{
    HarnessCapability, HarnessCheckpoint, HarnessCheckpointSink, HarnessContext,
    HarnessInteractionHandler, HarnessProtocol, HostCapability, JsonObject, McpServerConfig,
    ModelChoice, ResumePolicy, ToolGateway,
};

use crate::error::providers_error_to_common;
use crate::provider::{is_builtin_provider, provider_names, ProviderDescriptor, ProviderSpec};
use crate::providers;

/// Python's `HARNESS_PROVIDER_ENTRY_POINT_GROUP`: the entry-point group provider distributions
/// publish under.
///
/// Rust has no `importlib.metadata` analogue, so the constant is exposed for callers that translate
/// a Python manifest; providers are registered in-process or compiled in.
pub const HARNESS_PROVIDER_ENTRY_POINT_GROUP: &str = "openjiuwen.harness_providers";

/// Every registry name this crate declares, in Python `PROVIDER_NAMES` order.
///
/// These are lookup keys, not harness identities: the card a provider reports may have a different
/// name (`native` serves `openjiuwen-native`).
pub const PROVIDER_NAMES: [&str; 5] = [
    provider_names::NATIVE,
    provider_names::NATIVE_V2,
    provider_names::CLAUDE_CODE,
    provider_names::CODEX,
    provider_names::DSH,
];

/// Card name of the `native_v2` provider (hosted by `openjiuwen-agent-teams`).
const NATIVE_V2_CARD_NAME: &str = "native_v2";
/// Implementation version declared for `native_v2`.
const NATIVE_V2_VERSION: &str = "0.2.0";
/// External prerequisite of `native_v2` when this build has the provider compiled out.
#[cfg(not(feature = "native-v2"))]
const NATIVE_V2_REQUIRES: &str = "openjiuwen-agent-teams";
/// External prerequisite reported for `native_v2`: none once the feature is compiled in.
#[cfg(feature = "native-v2")]
const NATIVE_V2_REQUIREMENT: Option<&str> = None;
/// External prerequisite reported for `native_v2` in a build without the feature.
#[cfg(not(feature = "native-v2"))]
const NATIVE_V2_REQUIREMENT: Option<&str> = Some(NATIVE_V2_REQUIRES);
/// Card name of the Claude Code provider.
const CLAUDE_CODE_CARD_NAME: &str = "claude-code";
/// Card name of the Codex provider.
const CODEX_CARD_NAME: &str = "codex";
/// Card name of the `DeepSeek` harness provider.
const DSH_CARD_NAME: &str = "deepseek-harness";
/// Implementation version declared for the Claude Code and Codex adapters.
const VENDOR_VERSION: &str = "0.1.0";
/// Implementation version declared for the DSH adapter.
const DSH_VERSION: &str = "0.3.0";
/// External prerequisite of the vendor providers when this build has them compiled out.
#[cfg(not(feature = "vendor"))]
const VENDOR_REQUIRES: &str = "rebuild with the `vendor` feature";
/// External prerequisite reported for the vendor providers: none once the feature is compiled in.
#[cfg(feature = "vendor")]
const VENDOR_REQUIREMENT: Option<&str> = None;
/// External prerequisite reported for the vendor providers in a build without the feature.
#[cfg(not(feature = "vendor"))]
const VENDOR_REQUIREMENT: Option<&str> = Some(VENDOR_REQUIRES);
/// Whether this build compiles the vendor providers in.
#[cfg(feature = "vendor")]
const VENDOR_IMPLEMENTED: bool = true;
/// Whether this build compiles the vendor providers in.
#[cfg(not(feature = "vendor"))]
const VENDOR_IMPLEMENTED: bool = false;

/// Creates the provider registered under a name.
///
/// The factory is invoked on every [`resolve_provider`] / [`create_harness`] call and must be
/// cheap. It may fail — a host that registers a lazy adapter around an optional SDK reports the
/// missing dependency through the returned error instead of panicking.
pub type ProviderFactory = Arc<dyn Fn() -> Result<Arc<dyn ProviderSpec>> + Send + Sync>;

/// Returns the process-global registry, initializing it on first use.
fn registry() -> &'static RwLock<IndexMap<String, ProviderFactory>> {
    static REGISTRY: OnceLock<RwLock<IndexMap<String, ProviderFactory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(IndexMap::new()))
}

/// Takes the registry read lock, recovering from a poisoned mutex.
fn read_registry() -> RwLockReadGuard<'static, IndexMap<String, ProviderFactory>> {
    registry().read().unwrap_or_else(PoisonError::into_inner)
}

/// Takes the registry write lock, recovering from a poisoned mutex.
fn write_registry() -> RwLockWriteGuard<'static, IndexMap<String, ProviderFactory>> {
    registry().write().unwrap_or_else(PoisonError::into_inner)
}

/// Registers `factory` under `name`.
///
/// Re-registering a name that is already registered **replaces** the factory in place, keeping the
/// original position in [`available_providers`] (Python dict semantics).
///
/// # Errors
///
/// Returns [`Error::Validation`] when `name` is empty or is one of the built-in provider names: the
/// built-in table always wins in [`resolve_provider`], so such a registration could never take
/// effect.
pub fn register_harness_provider(name: &str, factory: ProviderFactory) -> Result<()> {
    if name.is_empty() || is_builtin_provider(name) {
        return Err(Error::validation(format!(
            "cannot register harness provider '{name}': empty or built-in name"
        )));
    }
    write_registry().insert(name.to_owned(), factory);
    tracing::debug!(provider = name, "registered harness provider");
    Ok(())
}

/// Removes the provider registered under `name`, if any.
///
/// Unknown names — including the built-in names, which are never in the registry — are ignored, so
/// this never fails and never panics.
pub fn unregister_harness_provider(name: &str) {
    let removed = write_registry().shift_remove(name);
    if removed.is_some() {
        tracing::debug!(provider = name, "unregistered harness provider");
    }
}

/// Returns every resolvable provider name: the declared names first, then the registered ones in
/// insertion order, without duplicates.
#[must_use]
pub fn available_providers() -> Vec<String> {
    let mut names: Vec<String> = PROVIDER_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    let registry = read_registry();
    for name in registry.keys() {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    names
}

/// Resolves `name` to the provider that serves it.
///
/// The built-in table is consulted first and its result is returned verbatim, including the
/// `Unsupported` error of a declared-but-deferred provider. Only names outside the table reach the
/// registry; a registered factory is invoked after the registry lock is released.
///
/// # Errors
///
/// - [`Error::NotFound`] when nothing serves `name`; the message lists [`available_providers`] in
///   Python's wording.
/// - [`Error::Validation`] when the registered factory for `name` fails.
/// - Whatever the built-in table reports for a declared name (for example [`Error::Unsupported`]
///   for a feature this build was compiled without).
pub fn resolve_provider(name: &str) -> Result<Arc<dyn ProviderSpec>> {
    if let Some(provider) = providers::builtin_provider(name) {
        return provider;
    }
    // Clone the factory out of the lock: the factory is caller code and must never run under the
    // registry lock.
    let factory = read_registry().get(name).cloned();
    let Some(factory) = factory else {
        return Err(Error::not_found(format!(
            "unknown harness provider '{name}'; expected one of {}",
            available_providers().join(", ")
        )));
    };
    factory().map_err(|error| {
        Error::validation(format!(
            "harness provider '{name}' could not be instantiated: {error}"
        ))
    })
}

/// Creates an **unstarted** harness through the provider registered as `provider`.
///
/// `config` is the provider's own create mapping and is forwarded verbatim; starting the harness is
/// a separate [`HarnessProtocol::start`] call.
///
/// # Errors
///
/// Returns the [`resolve_provider`] error when `provider` cannot be resolved, and the provider's
/// translated [`HarnessError`](openjiuwen_harness_protocol::HarnessError) when `config` is invalid.
pub fn create_harness(provider: &str, config: JsonObject) -> Result<Box<dyn HarnessProtocol>> {
    let spec = resolve_provider(provider)?;
    spec.create(config).map_err(providers_error_to_common)
}

/// Maps a host model choice onto the create fragment of the `provider` model hook.
///
/// The capability gates run before the provider sees the choice, in Python's order: a provider that
/// does not configure models at all, then a fallback model without
/// [`HarnessCapability::AuthFallback`], then an endpoint-served model without
/// [`HarnessCapability::EndpointModels`].
///
/// # Errors
///
/// Returns [`Error::Validation`] for a failed gate or an unresolvable provider, and the translated
/// provider error when the hook rejects `choice`.
pub fn provider_model_config(
    provider: &str,
    choice: &ModelChoice,
    fallback: Option<&ModelChoice>,
) -> Result<JsonObject> {
    let spec = resolve_provider(provider)?;
    if !spec.configures_models() {
        return Err(Error::validation(format!(
            "harness provider '{provider}' does not configure models"
        )));
    }
    let card = spec.card();
    if fallback.is_some() && !card.supports(HarnessCapability::AuthFallback) {
        return Err(Error::validation(format!(
            "harness provider '{provider}' has no authentication fallback"
        )));
    }
    let uses_endpoint = choice.endpoint.is_some()
        || fallback
            .and_then(|fallback| fallback.endpoint.as_ref())
            .is_some();
    if uses_endpoint && !card.supports(HarnessCapability::EndpointModels) {
        return Err(Error::validation(format!(
            "harness provider '{provider}' does not take endpoint models"
        )));
    }
    spec.model_config(choice, fallback)
        .map_err(providers_error_to_common)
}

/// Describes every provider this build can name: the declared names first, then the registered ones
/// in insertion order.
///
/// A registered factory that fails to build its provider is skipped instead of failing the whole
/// listing, so one broken adapter does not hide the rest of the registry. The declared descriptors
/// are static metadata: the three vendor entries are `implemented` exactly when this build has the
/// `vendor` feature compiled in (`create` really constructs their subprocess harnesses), and
/// otherwise report the feature as their `requires` entry. Feature flags never rewrite the other
/// declared names: a build compiled without `native` still lists it as implemented here while
/// [`resolve_provider`] reports the missing feature.
#[must_use]
pub fn provider_descriptors() -> Vec<ProviderDescriptor> {
    let mut descriptors: Vec<ProviderDescriptor> = PROVIDER_NAMES
        .iter()
        .filter_map(|name| builtin_descriptor(name))
        .collect();
    let factories: Vec<(String, ProviderFactory)> = read_registry()
        .iter()
        .map(|(name, factory)| (name.clone(), Arc::clone(factory)))
        .collect();
    for (name, factory) in factories {
        if let Ok(spec) = factory() {
            descriptors.push(registered_descriptor(&name, spec.as_ref()));
        }
    }
    descriptors
}

/// Describes the provider registered as `name`.
///
/// Declared names are answered from static metadata (no feature probe, no construction); registered
/// names invoke their factory.
///
/// # Errors
///
/// Returns the same [`Error::NotFound`] as [`resolve_provider`] for an unknown name, and
/// [`Error::Validation`] when the registered factory fails.
pub fn describe_provider(name: &str) -> Result<ProviderDescriptor> {
    if let Some(descriptor) = builtin_descriptor(name) {
        return Ok(descriptor);
    }
    let spec = resolve_provider(name)?;
    Ok(registered_descriptor(name, spec.as_ref()))
}

/// Returns the static descriptor of a declared provider name, or [`None`] for any other name.
fn builtin_descriptor(name: &str) -> Option<ProviderDescriptor> {
    let descriptor = match name {
        provider_names::NATIVE => ProviderDescriptor {
            name: name.to_owned(),
            card_name: HARNESS_NAME.to_owned(),
            implementation_version: PROVIDER_VERSION.to_owned(),
            builtin: true,
            implemented: true,
            configures_models: true,
            requires: None,
        },
        provider_names::NATIVE_V2 => ProviderDescriptor {
            name: name.to_owned(),
            card_name: NATIVE_V2_CARD_NAME.to_owned(),
            implementation_version: NATIVE_V2_VERSION.to_owned(),
            builtin: true,
            implemented: true,
            configures_models: false,
            requires: NATIVE_V2_REQUIREMENT.map(str::to_owned),
        },
        provider_names::CLAUDE_CODE => ProviderDescriptor {
            name: name.to_owned(),
            card_name: CLAUDE_CODE_CARD_NAME.to_owned(),
            implementation_version: VENDOR_VERSION.to_owned(),
            builtin: true,
            implemented: VENDOR_IMPLEMENTED,
            configures_models: true,
            requires: VENDOR_REQUIREMENT.map(str::to_owned),
        },
        provider_names::CODEX => ProviderDescriptor {
            name: name.to_owned(),
            card_name: CODEX_CARD_NAME.to_owned(),
            implementation_version: VENDOR_VERSION.to_owned(),
            builtin: true,
            implemented: VENDOR_IMPLEMENTED,
            configures_models: true,
            requires: VENDOR_REQUIREMENT.map(str::to_owned),
        },
        provider_names::DSH => ProviderDescriptor {
            name: name.to_owned(),
            card_name: DSH_CARD_NAME.to_owned(),
            implementation_version: DSH_VERSION.to_owned(),
            builtin: true,
            implemented: VENDOR_IMPLEMENTED,
            configures_models: true,
            requires: VENDOR_REQUIREMENT.map(str::to_owned),
        },
        _ => return None,
    };
    Some(descriptor)
}

/// Builds the descriptor of a resolved, non-declared provider.
fn registered_descriptor(name: &str, spec: &dyn ProviderSpec) -> ProviderDescriptor {
    let card = spec.card();
    ProviderDescriptor {
        name: name.to_owned(),
        card_name: card.name.clone(),
        implementation_version: card.implementation_version.clone(),
        builtin: false,
        implemented: true,
        configures_models: spec.configures_models(),
        requires: None,
    }
}

/// Host input for [`build_harness_context`], mirroring the fields of [`HarnessContext`] a host can
/// decide on before the provider is known.
///
/// Every field maps one-to-one onto its [`HarnessContext`] counterpart; the identity fields are
/// optional because they default to the provider's card name when the host has no opinion.
/// [`env`](Self::env) and [`metadata`](Self::metadata) may hold secrets, so the [`Debug`]
/// implementation prints their sizes, never their contents.
#[derive(Clone, Default)]
pub struct HarnessContextOptions {
    /// Host session the agent runs in; checkpoints are scoped to it.
    pub host_session_id: String,
    /// Stable agent identifier; defaults to the resolved agent name.
    pub agent_id: Option<String>,
    /// Human-readable agent name; defaults to the provider's card name.
    pub agent_name: Option<String>,
    /// System prompt the harness should treat as the agent's instructions.
    pub system_prompt: String,
    /// Working directory for the harness, when the host pins one.
    pub cwd: Option<String>,
    /// Environment variables for the harness; may contain secrets.
    pub env: BTreeMap<String, String>,
    /// Host services the harness may use.
    pub host_capabilities: BTreeSet<HostCapability>,
    /// How the harness should use [`checkpoint`](Self::checkpoint).
    pub resume_policy: ResumePolicy,
    /// Checkpoint the harness may resume from.
    pub checkpoint: Option<HarnessCheckpoint>,
    /// Host service the harness persists checkpoints into.
    pub checkpoint_sink: Option<Arc<dyn HarnessCheckpointSink>>,
    /// Host interaction handler answering awaited provider requests.
    pub interactions: Option<Arc<dyn HarnessInteractionHandler>>,
    /// Host tool gateway used for native tool calls.
    pub tools: Option<Arc<dyn ToolGateway>>,
    /// MCP servers the harness should connect to; dropped for the `native` provider.
    pub mcp_servers: Vec<McpServerConfig>,
    /// Free-form host metadata; may contain secrets and is never logged.
    pub metadata: JsonObject,
}

impl HarnessContextOptions {
    /// Builds options from the two mandatory fields, with everything else defaulted.
    ///
    /// The agent name and identifier are left [`None`] so [`build_harness_context`] can derive them
    /// from the resolved provider's card.
    #[must_use]
    pub fn new(host_session_id: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        Self {
            host_session_id: host_session_id.into(),
            system_prompt: system_prompt.into(),
            ..Self::default()
        }
    }
}

impl fmt::Debug for HarnessContextOptions {
    /// Renders the scalars, the presence of every optional host service, and the sizes of the
    /// secret-bearing collections.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HarnessContextOptions")
            .field("host_session_id", &self.host_session_id)
            .field("agent_id", &self.agent_id)
            .field("agent_name", &self.agent_name)
            .field("system_prompt", &self.system_prompt)
            .field("cwd", &self.cwd)
            .field("env", &format_args!("<{} variables>", self.env.len()))
            .field("host_capabilities", &self.host_capabilities)
            .field("resume_policy", &self.resume_policy)
            .field("checkpoint", &self.checkpoint.is_some())
            .field("checkpoint_sink", &self.checkpoint_sink.is_some())
            .field("interactions", &self.interactions.is_some())
            .field("tools", &self.tools.is_some())
            .field("mcp_servers", &self.mcp_servers.len())
            .field("metadata", &format_args!("<{} keys>", self.metadata.len()))
            .finish()
    }
}

/// Builds the [`HarnessContext`] a harness is started with from provider-independent options.
///
/// This is the manifest-free Rust counterpart of Python's `build_harness_context`: the provider is
/// resolved only to read its card, so the identity defaults stay provider-specific
/// (`agent_name` ← [`HarnessContextOptions::agent_name`] or the card name, `agent_id` ←
/// [`HarnessContextOptions::agent_id`] or the resolved agent name).
///
/// Capabilities are injected rather than demanded: a [`HarnessContextOptions::checkpoint_sink`]
/// adds [`HostCapability::CheckpointSink`], and non-empty MCP servers add
/// [`HostCapability::McpServers`]. Hooks and telemetry are left unset — this builder has no
/// options for them; set [`HarnessContext::hooks`] / [`HarnessContext::telemetry`] afterwards.
///
/// # The `native` provider and MCP servers
///
/// The native harness refuses MCP servers supplied through the context — its tools come from the
/// `DeepAgent` it wraps — so the servers of [`HarnessContextOptions::mcp_servers`] are **dropped**
/// and [`HostCapability::McpServers`] is not injected for `native`. Python raises instead
/// (documented deviation): dropping keeps one options value reusable across providers.
///
/// # Errors
///
/// Returns the [`resolve_provider`] error when `provider` cannot be resolved.
pub fn build_harness_context(
    provider: &str,
    options: HarnessContextOptions,
) -> Result<HarnessContext> {
    let spec = resolve_provider(provider)?;
    let card_name = spec.card().name.clone();
    let agent_name = options.agent_name.unwrap_or(card_name);
    let agent_id = options.agent_id.unwrap_or_else(|| agent_name.clone());
    let mut context = HarnessContext::new(
        agent_name,
        agent_id,
        options.host_session_id,
        options.system_prompt,
    );
    context.cwd = options.cwd;
    context.env = options.env;
    context.host_capabilities = options.host_capabilities;
    context.resume_policy = options.resume_policy;
    context.checkpoint = options.checkpoint;
    context.checkpoint_sink = options.checkpoint_sink;
    context.interactions = options.interactions;
    context.tools = options.tools;
    context.metadata = options.metadata;
    if context.checkpoint_sink.is_some() {
        context
            .host_capabilities
            .insert(HostCapability::CheckpointSink);
    }
    if provider == provider_names::NATIVE {
        // The native harness owns its tools; context MCP servers are not accepted (see above).
    } else if !options.mcp_servers.is_empty() {
        context.host_capabilities.insert(HostCapability::McpServers);
        context.mcp_servers = options.mcp_servers;
    }
    Ok(context)
}

#[cfg(test)]
mod tests;
