//! The `AgentArts` memory-service adapter (`agentarts`) — SDK-only.
//!
//! Unlike the other four vendors this adapter never speaks raw HTTP. The Python adapter
//! imports `agentarts.sdk.memory.MemoryClient` and drives the vendor SDK, which owns its
//! own connection handling; no documented REST contract exists to fall back on, so there
//! is no [`WireSpec`](super::super::rest::WireSpec) for this vendor and an `endpoint` in
//! the configuration cannot make the handle operational.
//!
//! Every operation therefore returns the same typed
//! [`openjiuwen_common::error::Error::Unsupported`] naming the missing SDK, whether or not
//! an `endpoint` is configured, mirroring the Python package's lazy client acquisition (a
//! missing extra surfaces at first use, not at import).
use std::fmt;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use super::super::provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
use super::super::registry::{ProviderConfig, ProviderKind};
use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

/// Registry id of the adapter (the Python provider id).
const ID: &str = "agentarts";
/// Human-readable vendor name used in messages.
const DISPLAY: &str = "AgentArts";
/// The dependency every operation names.
const MISSING: &str = "the `agentarts-sdk` package (`MemoryClient`) and its HTTP transport";

/// The SDK-only backend of the `AgentArts` adapter.
///
/// It keeps the configuration (so `config()` and the redacting `Debug` can report it) and
/// nothing else: there is no transport to construct and no wire format to describe.
struct SdkOnlyProvider {
    /// The configuration the handle was built from.
    config: ProviderConfig,
}

impl SdkOnlyProvider {
    /// Builds the helper for `config`; never fails.
    fn new(config: &ProviderConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// The kind this backend implements.
    fn kind() -> ProviderKind {
        ProviderKind::AgentArts
    }

    /// The single typed error every operation returns.
    fn error() -> Error {
        Error::unsupported(format!(
            "the {DISPLAY} memory provider requires {MISSING}, which this build does not provide"
        ))
    }
}

#[async_trait]
impl MemoryProvider for SdkOnlyProvider {
    fn name(&self) -> &str {
        ID
    }

    /// A handle with no transport declares no capability at all.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    async fn initialize(&self) -> Result<()> {
        Err(Self::error())
    }

    async fn add(
        &self,
        _scope: &MemoryScope,
        _memories: Vec<NewMemory>,
        _options: &AddOptions,
    ) -> Result<AddOutcome> {
        Err(Self::error())
    }

    async fn search(
        &self,
        _scope: &MemoryScope,
        _query: &str,
        _options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        Err(Self::error())
    }

    async fn get(&self, _scope: &MemoryScope, _id: &str) -> Result<Option<MemoryRecord>> {
        Err(Self::error())
    }

    async fn update(
        &self,
        _scope: &MemoryScope,
        _id: &str,
        _update: MemoryUpdate,
        _options: &CallOptions,
    ) -> Result<MemoryRecord> {
        Err(Self::error())
    }

    async fn delete(&self, _scope: &MemoryScope, _id: &str) -> Result<bool> {
        Err(Self::error())
    }

    async fn list(&self, _filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        Err(Self::error())
    }

    async fn clear(&self, _filter: &MemoryFilter) -> Result<usize> {
        Err(Self::error())
    }

    async fn health(&self) -> Result<ProviderHealth> {
        Err(Self::error())
    }
}

/// Handle for the `AgentArts` memory service (`agentarts`).
///
/// Construction never fails: the vendor SDK is required at
/// [`MemoryProvider::initialize`], and configuring an `endpoint` changes nothing because
/// this adapter has no HTTP path.
pub struct AgentArtsMemoryProvider {
    /// The SDK-only backend.
    inner: SdkOnlyProvider,
}

impl AgentArtsMemoryProvider {
    /// Builds the adapter handle for `config`.
    #[must_use]
    pub fn new(config: &ProviderConfig) -> Self {
        Self {
            inner: SdkOnlyProvider::new(config),
        }
    }

    /// The configuration this handle was built from.
    #[must_use]
    pub fn config(&self) -> &ProviderConfig {
        &self.inner.config
    }

    /// The kind this adapter implements.
    #[must_use]
    pub fn kind(&self) -> ProviderKind {
        SdkOnlyProvider::kind()
    }
}

impl fmt::Debug for AgentArtsMemoryProvider {
    /// Prints the adapter identity, the missing dependency, and the configured endpoint —
    /// never the configuration or its credential.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentArtsMemoryProvider")
            .field("name", &ID)
            .field("missing", &MISSING)
            .field("endpoint", &self.inner.config.endpoint)
            .finish_non_exhaustive()
    }
}

implement_vendor_provider!(AgentArtsMemoryProvider, inner;);
