//! Deferred Redis key-value backend.
//!
//! Ports `openjiuwen/extensions/store/kv/redis_store.py` (`RedisStore`), which
//! wraps the `redis.asyncio` `Redis`/`RedisCluster` clients. This crate keeps
//! the Python configuration surface but adds no Redis client dependency: the
//! backend is deferred, so [`RedisStoreConfig`] is validated and every
//! [`KvStore`] operation answers
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported).
//!
//! # Deviation from Python
//!
//! Python's `delete_by_prefix(prefix, batch_size=500)` and
//! `batch_delete(keys, batch_size=500)` chunk their work into batches of 500
//! keys. The kernel [`KvStore`] trait has no batch-size parameter, so the
//! deferred methods take the whole prefix or key set; the 500-key chunking is
//! documented here as the intended behavior of a future client-backed
//! implementation rather than modelled in the signature.
//!
//! Python counterpart tests:
//! `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py`
//! exercises Redis through the checkpointer; the Python tree has no dedicated
//! `RedisStore` unit test.
#![allow(clippy::module_name_repetitions)] // `Redis*` names are fixed by the contract.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{KvPipeline, KvStore, KvValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
/// The accepted connection-scheme prefixes.
///
/// `rediss://` is listed separately because it does not start with `redis://`.
const ACCEPTED_URL_SCHEMES: [&str; 4] = [
    "redis://",
    "rediss://",
    "redis+cluster://",
    "rediss+cluster://",
];
/// The message every deferred operation answers with.
const UNSUPPORTED: &str =
    "redis key-value store backend is deferred; use the in-memory, file, or sqlite backend";
/// Configuration for a [`RedisStore`].
///
/// Mirrors the Python constructor's inputs: a `url` or explicit
/// `connection_args`, plus an optional forced `cluster_mode`. When
/// `cluster_mode` is unset, [`RedisStore::new`] infers it from a `+cluster`
/// URL scheme.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct RedisStoreConfig {
    /// Connection URL; one of `redis://`, `rediss://`, `redis+cluster://`, or
    /// `rediss+cluster://`.
    pub url: Option<String>,
    /// Whether to use a cluster client; inferred from the URL when `None`.
    pub cluster_mode: Option<bool>,
    /// Extra connection parameters, used when no `url` is supplied.
    pub connection_args: BTreeMap<String, Value>,
}
/// A validated but deferred Redis key-value store.
///
/// Construction validates the configuration; every [`KvStore`] operation then
/// returns [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported)
/// naming the supported alternatives.
#[derive(Debug, Clone)]
pub struct RedisStore {
    config: RedisStoreConfig,
}
impl RedisStore {
    /// Validates `config` and wraps it in a store.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when neither a `url` nor a non-empty `connection_args` is present, or
    /// when the URL does not use an accepted scheme.
    pub fn new(mut config: RedisStoreConfig) -> Result<Self> {
        normalize(&mut config)?;
        Ok(Self { config })
    }
    /// Validates `config` and then reports that the backend is deferred.
    ///
    /// The async signature is fixed by the contract so a future client-backed
    /// implementation keeps the call sites unchanged.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`RedisStore::new`], followed by
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// for a valid configuration.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn connect(config: RedisStoreConfig) -> Result<Arc<dyn KvStore>> {
        Self::new(config)?;
        Err(unsupported())
    }
    /// The normalized configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &RedisStoreConfig {
        &self.config
    }
}
/// Normalizes `config` in place, inferring `cluster_mode` and validating the URL.
fn normalize(config: &mut RedisStoreConfig) -> Result<()> {
    let url = config
        .url
        .take()
        .map(|url| url.trim().to_string())
        .filter(|url| !url.is_empty());
    if let Some(url) = url.as_deref() {
        validate_url(url)?;
    }
    if url.is_none() && config.connection_args.is_empty() {
        return Err(Error::validation(
            "redis store config requires a `url` or non-empty `connection_args`",
        ));
    }
    if config.cluster_mode.is_none() {
        config.cluster_mode = Some(url.as_deref().is_some_and(|url| url.contains("+cluster")));
    }
    config.url = url;
    Ok(())
}
/// Rejects a URL whose scheme is not a Redis scheme.
fn validate_url(url: &str) -> Result<()> {
    if ACCEPTED_URL_SCHEMES
        .iter()
        .any(|scheme| url.starts_with(scheme))
    {
        Ok(())
    } else {
        Err(Error::validation(format!(
            "redis url `{url}` must start with one of: redis://, rediss://, redis+cluster://, rediss+cluster://"
        )))
    }
}
/// Builds the deferral error every operation returns.
fn unsupported() -> Error {
    Error::unsupported(UNSUPPORTED)
}
/// The no-op pipeline a deferred [`RedisStore`] hands out.
///
/// Queued operations are dropped and [`KvPipeline::execute`] answers the same
/// [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) as the
/// store, so a caller cannot mistake the deferral for an empty result.
struct RedisPipeline<'a> {
    store: PhantomData<&'a RedisStore>,
}
impl<'a> RedisPipeline<'a> {
    /// Binds an empty pipeline to `store`.
    fn new(_store: &'a RedisStore) -> Self {
        Self { store: PhantomData }
    }
}
#[async_trait]
impl KvPipeline for RedisPipeline<'_> {
    fn set(&mut self, _key: &str, _value: KvValue) {}
    fn get(&mut self, _key: &str) {}
    fn exists(&mut self, _key: &str) {}
    async fn execute(&mut self) -> Result<Vec<Option<KvValue>>> {
        Err(unsupported())
    }
}
#[async_trait]
impl KvStore for RedisStore {
    async fn set(&self, _key: &str, _value: KvValue) -> Result<()> {
        Err(unsupported())
    }
    async fn get(&self, _key: &str) -> Result<Option<KvValue>> {
        Err(unsupported())
    }
    async fn exists(&self, _key: &str) -> Result<bool> {
        Err(unsupported())
    }
    async fn delete(&self, _key: &str) -> Result<()> {
        Err(unsupported())
    }
    async fn exclusive_set(
        &self,
        _key: &str,
        _value: KvValue,
        _expiry: Option<Duration>,
    ) -> Result<bool> {
        Err(unsupported())
    }
    async fn get_by_prefix(&self, _prefix: &str) -> Result<BTreeMap<String, KvValue>> {
        Err(unsupported())
    }
    async fn delete_by_prefix(&self, _prefix: &str) -> Result<usize> {
        Err(unsupported())
    }
    async fn mget(&self, _keys: &[String]) -> Result<Vec<Option<KvValue>>> {
        Err(unsupported())
    }
    async fn batch_delete(&self, _keys: &[String]) -> Result<usize> {
        Err(unsupported())
    }
    fn pipeline(&self) -> Box<dyn KvPipeline + '_> {
        Box::new(RedisPipeline::new(self))
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    /// A configuration carrying only `url`.
    fn url_config(url: &str) -> RedisStoreConfig {
        RedisStoreConfig {
            url: Some(url.to_string()),
            ..RedisStoreConfig::default()
        }
    }
    /// Asserts that a deferred operation fails with the Redis message.
    fn assert_unsupported<T>(result: Result<T>) {
        let error = result.err().expect("a deferred operation must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().to_ascii_lowercase().contains("redis"));
    }
    #[test]
    fn every_accepted_scheme_builds_a_store() {
        for scheme in ACCEPTED_URL_SCHEMES {
            let url = format!("{scheme}localhost:6379");
            let store = RedisStore::new(url_config(&url)).unwrap();
            assert_eq!(store.config().url.as_deref(), Some(url.as_str()));
        }
    }
    #[test]
    fn invalid_scheme_is_rejected() {
        let error = RedisStore::new(url_config("http://localhost:6379")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("redis://"));
    }
    #[test]
    fn config_needs_a_url_or_connection_args() {
        let error = RedisStore::new(RedisStoreConfig::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let config = RedisStoreConfig {
            connection_args: BTreeMap::from([("host".to_string(), json!("localhost"))]),
            ..RedisStoreConfig::default()
        };
        let store = RedisStore::new(config).unwrap();
        assert_eq!(store.config().url, None);
        assert_eq!(store.config().cluster_mode, Some(false));
    }
    #[test]
    fn cluster_mode_is_inferred_from_the_url() {
        let standalone = RedisStore::new(url_config("redis://localhost:6379")).unwrap();
        assert_eq!(standalone.config().cluster_mode, Some(false));
        let tls_standalone = RedisStore::new(url_config("rediss://localhost:6379")).unwrap();
        assert_eq!(tls_standalone.config().cluster_mode, Some(false));
        let cluster = RedisStore::new(url_config("redis+cluster://localhost:6379")).unwrap();
        assert_eq!(cluster.config().cluster_mode, Some(true));
        let tls_cluster = RedisStore::new(url_config("rediss+cluster://localhost:6379")).unwrap();
        assert_eq!(tls_cluster.config().cluster_mode, Some(true));
        let explicit = RedisStore::new(RedisStoreConfig {
            cluster_mode: Some(false),
            ..url_config("redis+cluster://localhost:6379")
        })
        .unwrap();
        assert_eq!(explicit.config().cluster_mode, Some(false));
    }
    #[test]
    fn config_serde_round_trips() {
        let config = RedisStoreConfig {
            url: Some("redis://localhost:6379".to_string()),
            cluster_mode: Some(false),
            connection_args: BTreeMap::from([("db".to_string(), json!(1))]),
        };
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(
            encoded,
            json!({
                "url": "redis://localhost:6379",
                "cluster_mode": false,
                "connection_args": {"db": 1},
            })
        );
        let decoded: RedisStoreConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, config);
    }
    #[tokio::test]
    async fn connect_reports_the_deferral() {
        let error = RedisStore::connect(url_config("redis://localhost:6379"))
            .await
            .err()
            .expect("a deferred backend must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().to_ascii_lowercase().contains("redis"));
    }
    #[tokio::test]
    async fn every_store_operation_is_unsupported() {
        let store = RedisStore::new(url_config("redis://localhost:6379")).unwrap();
        assert_unsupported(store.set("key", KvValue::text("value")).await);
        assert_unsupported(store.get("key").await);
        assert_unsupported(store.exists("key").await);
        assert_unsupported(store.delete("key").await);
        assert_unsupported(
            store
                .exclusive_set("key", KvValue::text("value"), None)
                .await,
        );
        assert_unsupported(store.get_by_prefix("pre").await);
        assert_unsupported(store.delete_by_prefix("pre").await);
        let keys = vec!["a".to_string(), "b".to_string()];
        assert_unsupported(store.mget(&keys).await);
        assert_unsupported(store.batch_delete(&keys).await);
        let mut pipeline = store.pipeline();
        pipeline.set("key", KvValue::text("value"));
        pipeline.get("key");
        pipeline.exists("key");
        assert_unsupported(pipeline.execute().await);
    }
}
