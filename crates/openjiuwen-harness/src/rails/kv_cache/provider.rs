//! The provider seam of the KV-cache lifecycle bridge.
//!
//! Python's `Session.prepare_kvc/suspend_kvc/release_kvc` delegate to a
//! KV-cache runtime that owns provider-level prefetch/offload/evict calls. The
//! Rust kernel has no such runtime yet, so this module defines the narrow seam
//! the lifecycle needs and ships a typed [`UnsupportedKvCacheBackend`] for every
//! backend that is not ported: a deployment must fail loudly instead of
//! silently dropping affinity.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
/// Builds the fixed, stable message of an unsupported backend.
fn unsupported_message(backend: &str) -> String {
    format!(
        "kv_cache: `{backend}` provider backend is not implemented; KV-cache prefetch/offload \
         and distributed (Redis/cluster) backends are deferred"
    )
}
/// The provider operations the subagent lifecycle drives.
///
/// One object per backend (see [`UnsupportedKvCacheBackend`]); the lifecycle
/// owns the policy and identity bookkeeping, the implementation owns the remote
/// cache identity. Every operation is idempotent at this seam: releasing an
/// already released identity must succeed rather than report a missing cache.
#[async_trait]
pub trait KvCacheProviderRegistry: Send + Sync {
    /// Name of the backend, used in diagnostics.
    fn backend(&self) -> &str;
    /// Prepares (prefetches) the cache identity of a freshly started subagent.
    ///
    /// # Errors
    ///
    /// A provider/transport failure.
    async fn prepare(&self, cache_id: &str) -> Result<()>;
    /// Suspends (offloads) a resumable cache identity of a finished subagent.
    ///
    /// # Errors
    ///
    /// A provider/transport failure.
    async fn suspend(&self, cache_id: &str) -> Result<()>;
    /// Releases (evicts) a cache identity that will not be resumed.
    ///
    /// # Errors
    ///
    /// A provider/transport failure.
    async fn release(&self, cache_id: &str) -> Result<()>;
}
/// A backend whose operations are not implemented in this port.
///
/// Every method returns [`openjiuwen_common::error::ErrorKind::Unsupported`]
/// with a message naming the backend, so wiring an unported provider is a
/// visible, typed failure in the delegation path rather than a silent no-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedKvCacheBackend {
    /// Backend name reported by [`KvCacheProviderRegistry::backend`].
    backend: String,
}
impl UnsupportedKvCacheBackend {
    /// Builds the stub for `backend`.
    #[must_use]
    pub fn new(backend: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
        }
    }
    /// The backend name this stub stands for.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        &self.backend
    }
    /// The message every operation returns.
    fn unsupported(&self) -> Error {
        Error::unsupported(unsupported_message(&self.backend))
    }
}
#[async_trait]
impl KvCacheProviderRegistry for UnsupportedKvCacheBackend {
    fn backend(&self) -> &str {
        self.backend_name()
    }
    async fn prepare(&self, _cache_id: &str) -> Result<()> {
        Err(self.unsupported())
    }
    async fn suspend(&self, _cache_id: &str) -> Result<()> {
        Err(self.unsupported())
    }
    async fn release(&self, _cache_id: &str) -> Result<()> {
        Err(self.unsupported())
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    #[test]
    fn names_the_backend() {
        let backend = UnsupportedKvCacheBackend::new("redis-cluster");
        assert_eq!(backend.backend(), "redis-cluster");
        assert_eq!(backend.backend_name(), "redis-cluster");
    }
    #[tokio::test]
    async fn every_operation_reports_the_documented_unsupported_message() {
        let backend = UnsupportedKvCacheBackend::new("vllm");
        let expected = "kv_cache: `vllm` provider backend is not implemented; KV-cache \
                        prefetch/offload and distributed (Redis/cluster) backends are deferred";
        for error in [
            backend.prepare("cache-1").await.unwrap_err(),
            backend.suspend("cache-1").await.unwrap_err(),
            backend.release("cache-1").await.unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert_eq!(error.message(), expected);
        }
    }
}
