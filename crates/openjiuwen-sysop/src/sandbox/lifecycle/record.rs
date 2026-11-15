//! The per-container bookkeeping entry of the sandbox lifecycle.
//!
//! Python parity: `SandboxRecord` (`sandbox/gateway/sandbox_store.py`).
use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::sandbox::config::{SandboxEndpoint, SandboxLauncherConfig};
use crate::sandbox::launcher::SandboxStatus;
use crate::util::sha256_hex;

/// The bookkeeping entry of one sandbox container.
///
/// Python parity: `SandboxRecord` (`sandbox/gateway/sandbox_store.py`). Python keys the
/// record by isolation key inside the store; this struct carries the key too, so a record
/// is meaningful on its own.
///
/// [`Self::container_config_hash`] identifies the launcher configuration the container was
/// created with and [`Self::metadata`] carries the labels that configuration derives; both
/// are refreshed whenever the lifecycle hands the record out again (see
/// [`refresh_container_config`]). Python additionally caches provider instances per hash;
/// this rewrite rebuilds providers from the resolved endpoint, so only the bookkeeping is
/// kept.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxRecord {
    /// Isolation key the record is stored under.
    pub isolation_key: String,
    /// Runtime-assigned sandbox id, when the launcher reports one.
    pub sandbox_id: Option<String>,
    /// Endpoint handed to the providers.
    pub endpoint: SandboxEndpoint,
    /// Current lifecycle state.
    pub status: SandboxStatus,
    /// Launcher type that created the container.
    pub launcher_type: String,
    /// Creation time, in seconds since the Unix epoch.
    pub created_epoch_secs: u64,
    /// Last time the endpoint was handed out, in seconds since the Unix epoch.
    pub last_used_epoch_secs: u64,
    /// SHA-256 hex digest of the launcher configuration the container was created with.
    ///
    /// Empty for a record that never saw a configuration (a hand-built default record). A
    /// changed hash means the container is reused under a different configuration, which
    /// would invalidate Python's provider cache.
    pub container_config_hash: String,
    /// Container/config labels derived from the launcher configuration.
    ///
    /// See [`container_metadata`] for the derivation rule. Merged, never replaced, so a
    /// label that a later configuration drops survives on the record.
    pub metadata: BTreeMap<String, String>,
}

/// The SHA-256 hex digest of `config`'s canonical JSON.
///
/// `serde_json` writes struct fields in declaration order and the configuration's maps are
/// [`BTreeMap`]s, so equal configurations always encode to the same bytes regardless of how
/// they were built.
#[must_use]
pub fn container_config_hash(config: &SandboxLauncherConfig) -> String {
    match serde_json::to_vec(config) {
        Ok(canonical) => sha256_hex(&canonical),
        // Every field is a string, an enum, an `Option<u64>` or a map of `Value`, so the
        // encoding cannot fail; hashing a marker anyway keeps the field a 64-character hex
        // digest instead of a value two different configurations could share.
        Err(_) => sha256_hex(format!("unencodable:{}", config.launcher_type).as_bytes()),
    }
}

/// The labels a launcher configuration contributes to [`SandboxRecord::metadata`].
///
/// Deterministic: `launcher_type` and `sandbox_type` are taken from the configuration, plus
/// every `extra_params` entry whose value is a JSON string (a non-string value has no label
/// representation and is left out). The two fixed keys win over an `extra_params` entry of
/// the same name.
#[must_use]
pub fn container_metadata(config: &SandboxLauncherConfig) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    for (key, value) in &config.extra_params {
        if let Some(value) = value.as_str() {
            metadata.insert(key.clone(), value.to_string());
        }
    }
    metadata.insert("launcher_type".to_string(), config.launcher_type.clone());
    metadata.insert("sandbox_type".to_string(), config.sandbox_type.clone());
    metadata
}

/// Merges the hash and labels of `config` into `record`.
///
/// A changed hash is logged at debug (the container is reused under a configuration it was
/// not created with, which would invalidate a provider cache keyed by the hash) and then
/// stored. Freshly derived labels are merged into [`SandboxRecord::metadata`] with new
/// values winning; labels the new configuration does not mention are preserved. Callers run
/// this only after a launcher call succeeded, so a failure never costs a record its
/// bookkeeping.
pub fn refresh_container_config(record: &mut SandboxRecord, config: &SandboxLauncherConfig) {
    let hash = container_config_hash(config);
    if record.container_config_hash != hash {
        tracing::debug!(
            isolation_key = %record.isolation_key,
            launcher_type = %config.launcher_type,
            previous_container_config_hash = %record.container_config_hash,
            container_config_hash = %hash,
            "sandbox launcher configuration changed; a provider cache keyed by the hash would \
             be invalidated"
        );
        record.container_config_hash = hash;
    }
    record.metadata.extend(container_metadata(config));
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::sandbox::config::OnStop;

    fn config(launcher_type: &str, sandbox_type: &str) -> SandboxLauncherConfig {
        SandboxLauncherConfig {
            launcher_type: launcher_type.to_string(),
            sandbox_type: sandbox_type.to_string(),
            ..SandboxLauncherConfig::default()
        }
    }

    #[test]
    fn hash_is_canonical_and_configuration_sensitive() {
        let base = config("docker", "aio");
        let same = config("docker", "aio");
        assert_eq!(container_config_hash(&base), container_config_hash(&same));
        assert_eq!(container_config_hash(&base).len(), 64);

        let mut other_type = base.clone();
        other_type.sandbox_type = "jiuwenbox".to_string();
        assert_ne!(
            container_config_hash(&base),
            container_config_hash(&other_type)
        );

        let mut other_ttl = base.clone();
        other_ttl.idle_ttl_seconds = Some(60);
        assert_ne!(
            container_config_hash(&base),
            container_config_hash(&other_ttl)
        );

        assert_ne!(
            container_config_hash(&base),
            container_config_hash(&SandboxLauncherConfig::default())
        );
    }

    #[test]
    fn extra_params_do_not_change_the_hash_by_insertion_order() {
        let mut first = config("docker", "aio");
        first
            .extra_params
            .insert("image".to_string(), json!("alpine"));
        first.extra_params.insert("cpu".to_string(), json!(2));
        let mut second = config("docker", "aio");
        second.extra_params.insert("cpu".to_string(), json!(2));
        second
            .extra_params
            .insert("image".to_string(), json!("alpine"));
        assert_eq!(
            container_config_hash(&first),
            container_config_hash(&second),
            "BTreeMap ordering makes the encoding canonical"
        );
    }

    #[test]
    fn metadata_carries_the_fixed_labels_and_string_extra_params() {
        let mut config = config("docker", "aio");
        config
            .extra_params
            .insert("image".to_string(), json!("alpine"));
        config.extra_params.insert("cpu".to_string(), json!(2));
        config
            .extra_params
            .insert("sandbox_type".to_string(), json!("spoofed"));
        let metadata = container_metadata(&config);
        assert_eq!(metadata["launcher_type"], "docker");
        assert_eq!(metadata["sandbox_type"], "aio", "the fixed key wins");
        assert_eq!(metadata["image"], "alpine");
        assert!(
            !metadata.contains_key("cpu"),
            "a non-string extra param has no label representation"
        );
        assert_eq!(metadata.len(), 3);
    }

    #[test]
    fn refresh_replaces_the_hash_and_merges_labels() {
        let mut record = SandboxRecord {
            isolation_key: "session_docker_aio_s1".to_string(),
            container_config_hash: container_config_hash(&config("docker", "aio")),
            metadata: BTreeMap::from([
                ("launcher_type".to_string(), "docker".to_string()),
                ("sandbox_type".to_string(), "aio".to_string()),
                ("stale".to_string(), "kept".to_string()),
            ]),
            ..SandboxRecord::default()
        };

        // An unrelated label survives, and an unchanged configuration keeps the hash.
        let before = record.clone();
        refresh_container_config(&mut record, &config("docker", "aio"));
        assert_eq!(record, before);

        let mut changed = config("docker", "jiuwenbox");
        changed
            .extra_params
            .insert("image".to_string(), json!("alpine"));
        refresh_container_config(&mut record, &changed);
        assert_eq!(
            record.container_config_hash,
            container_config_hash(&changed)
        );
        assert_eq!(record.metadata["sandbox_type"], "jiuwenbox");
        assert_eq!(record.metadata["image"], "alpine");
        assert_eq!(
            record.metadata["stale"], "kept",
            "labels the new configuration does not mention are preserved"
        );

        let mut on_stop = changed.clone();
        on_stop.on_stop = OnStop::Pause;
        refresh_container_config(&mut record, &on_stop);
        assert_eq!(
            record.container_config_hash,
            container_config_hash(&on_stop),
            "`on_stop` is part of the hashed configuration"
        );
    }

    #[test]
    fn a_default_record_round_trips_and_tolerates_unknown_fields() {
        let record = SandboxRecord::default();
        assert_eq!(record.container_config_hash, "");
        assert!(record.metadata.is_empty());

        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["container_config_hash"], "");
        assert_eq!(json["metadata"], json!({}));

        let back: SandboxRecord =
            serde_json::from_value(json!({"isolation_key": "x", "container_config_hash": "abc"}))
                .unwrap();
        assert_eq!(back.container_config_hash, "abc");
        assert!(back.metadata.is_empty());
    }
}
