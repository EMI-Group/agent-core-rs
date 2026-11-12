//! Transport configuration carriers and the [`create_messager`] factory.
//!
//! Python parity source: `openjiuwen/agent_teams/messager/base.py` (`MessagerTransportConfig`,
//! `MessagerPeerConfig`, the `create_messager` factory) and `pyzmq_backend.py`'s config record.
//! The carriers exist so a spec can round-trip through JSON unchanged even when this port cannot
//! build the named backend: [`create_messager`] rejects the deferred kinds with an
//! [`openjiuwen_common::error::Error::Unsupported`] that keeps the Python capability name.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::{InProcessMessager, Messager};
use crate::schema::TransportSpec;

/// Transport kind of the in-process messager (Python `"inprocess"`).
const INPROCESS_KIND: &str = "inprocess";
/// Transport kind of the deferred `ZeroMQ` backend (Python `"pyzmq"`).
const PYZMQ_KIND: &str = "pyzmq";
/// Transport kind of the deferred WebSocket relay (Python `"hybrid"`).
const HYBRID_KIND: &str = "hybrid";

/// Transport selection of a team, as carried by a spec.
///
/// Mirrors Python's `MessagerTransportConfig`: `kind` round-trips under the Rust key `"kind"` and
/// accepts the Python keys `"type"` and `"backend"` as aliases when deserializing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessagerTransportConfig {
    /// Backend key (`inprocess`, `pyzmq`, `hybrid`).
    #[serde(alias = "type", alias = "backend")]
    pub kind: String,
    /// Backend-specific parameters, never inspected by this crate.
    pub params: BTreeMap<String, Value>,
}

impl MessagerTransportConfig {
    /// Builds a config for `kind` with no parameters.
    #[must_use]
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            params: BTreeMap::new(),
        }
    }

    /// Builds the in-process (default) config.
    #[must_use]
    pub fn inprocess() -> Self {
        Self::new(INPROCESS_KIND)
    }

    /// Copies the kind and parameters of a team [`TransportSpec`].
    #[must_use]
    pub fn from_spec(spec: &TransportSpec) -> Self {
        Self {
            kind: spec.kind.clone(),
            params: spec.params.clone(),
        }
    }
}

/// Address of one peer of a non-in-process transport (Python `MessagerPeerConfig`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessagerPeerConfig {
    /// Member name the address belongs to.
    pub member_name: String,
    /// Backend-specific endpoint (for example `tcp://127.0.0.1:5555`).
    pub address: String,
}

/// Configuration of the in-process messager (Python `InProcessMessagerConfig`).
///
/// `peers` is carried for parity with the Python record; the in-process transport addresses
/// members by name and ignores it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InProcessMessagerConfig {
    /// Member name stamped into events published with an empty `sender_id`.
    pub sender: String,
    /// Peer address book of a distributed deployment (unused in process).
    pub peers: Vec<MessagerPeerConfig>,
}

/// Configuration of the deferred `ZeroMQ` messager (Python `PyZmqMessagerConfig`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PyZmqMessagerConfig {
    /// Team this messager would serve.
    pub team_name: String,
    /// Node identity inside the team (defaults to the member name in Python).
    pub node_id: Option<String>,
    /// Direct (point-to-point) endpoint.
    pub direct_addr: Option<String>,
    /// Publish endpoint of the pub/sub pair.
    pub pubsub_publish_addr: Option<String>,
    /// Subscribe endpoint of the pub/sub pair.
    pub pubsub_subscribe_addr: Option<String>,
    /// Per-request timeout in seconds.
    pub request_timeout_secs: f64,
    /// Peer address book.
    pub peers: Vec<MessagerPeerConfig>,
}

/// Builds the messager selected by `spec`.
///
/// | `spec.kind` | Result |
/// | --- | --- |
/// | `"inprocess"` | a live [`InProcessMessager`] with the default (empty) sender |
/// | `"pyzmq"` | [`Error::Unsupported`] naming the deferred `pyzmq` capability |
/// | `"hybrid"` | [`Error::Validation`] — `hybrid` was never a factory kind in Python |
/// | anything else | [`Error::Validation`] naming the unknown transport |
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error`] for every kind this port cannot build; see the
/// table above.
pub fn create_messager(spec: &TransportSpec) -> Result<Arc<dyn Messager>> {
    match spec.kind.as_str() {
        INPROCESS_KIND => Ok(Arc::new(InProcessMessager::new())),
        PYZMQ_KIND => Err(Error::unsupported(
            "transport 'pyzmq' is not implemented; use 'inprocess'",
        )),
        HYBRID_KIND => Err(Error::validation(
            "transport 'hybrid' is not supported by create_messager",
        )),
        kind => Err(Error::validation(format!("unknown transport '{kind}'"))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Runs the factory and returns the rejection, panicking if it built instead.
    fn rejection(spec: &TransportSpec) -> Error {
        match create_messager(spec) {
            Ok(_) => panic!("expected the factory to reject transport '{}'", spec.kind),
            Err(err) => err,
        }
    }

    #[test]
    fn factory_builds_the_in_process_transport() {
        let messager = create_messager(&TransportSpec::inprocess()).unwrap();
        assert_eq!(messager.name(), "inprocess");
        assert!(messager.is_started());
        assert_eq!(
            messager.subscribers(crate::schema::TeamTopic::Team).len(),
            0
        );
    }

    #[test]
    fn factory_rejects_pyzmq_as_unsupported() {
        let err = rejection(&TransportSpec {
            kind: "pyzmq".to_owned(),
            params: BTreeMap::new(),
        });
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert_eq!(
            err.to_string(),
            "unsupported: transport 'pyzmq' is not implemented; use 'inprocess'"
        );
    }

    #[test]
    fn factory_rejects_hybrid_as_validation() {
        let err = rejection(&TransportSpec {
            kind: "hybrid".to_owned(),
            params: BTreeMap::new(),
        });
        assert!(matches!(err, Error::Validation(_)), "{err:?}");
        assert_eq!(
            err.to_string(),
            "validation: transport 'hybrid' is not supported by create_messager"
        );
    }

    #[test]
    fn factory_rejects_an_unknown_transport() {
        let err = rejection(&TransportSpec {
            kind: "carrier-pigeon".to_owned(),
            params: BTreeMap::new(),
        });
        assert!(matches!(err, Error::Validation(_)), "{err:?}");
        assert_eq!(
            err.to_string(),
            "validation: unknown transport 'carrier-pigeon'"
        );
    }

    #[test]
    fn transport_config_round_trips_through_json() {
        let config = MessagerTransportConfig {
            kind: "pyzmq".to_owned(),
            params: BTreeMap::from([("request_timeout_secs".to_owned(), json!(2.5))]),
        };
        let text = serde_json::to_string(&config).unwrap();
        assert_eq!(
            text,
            r#"{"kind":"pyzmq","params":{"request_timeout_secs":2.5}}"#
        );
        let back: MessagerTransportConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn transport_config_accepts_the_python_kind_aliases() {
        for key in ["kind", "type", "backend"] {
            let config: MessagerTransportConfig =
                serde_json::from_value(json!({key: "inprocess", "unknown": 1})).unwrap();
            assert_eq!(config.kind, "inprocess", "{key}");
            assert!(config.params.is_empty(), "{key}");
        }
        let empty: MessagerTransportConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, MessagerTransportConfig::default());
        assert_eq!(
            MessagerTransportConfig::inprocess(),
            MessagerTransportConfig::new("inprocess")
        );
    }

    #[test]
    fn from_spec_preserves_kind_and_params() {
        let spec: TransportSpec = serde_json::from_value(json!({
            "type": "pyzmq",
            "params": {"team_name": "alpha", "request_timeout_secs": 5.0},
        }))
        .unwrap();
        let config = MessagerTransportConfig::from_spec(&spec);
        assert_eq!(config.kind, "pyzmq");
        assert_eq!(
            config.params,
            BTreeMap::from([
                ("team_name".to_owned(), json!("alpha")),
                ("request_timeout_secs".to_owned(), json!(5.0)),
            ])
        );
    }

    #[test]
    fn peer_and_inprocess_configs_round_trip_and_tolerate_unknown_fields() {
        let config = InProcessMessagerConfig {
            sender: "leader".to_owned(),
            peers: vec![MessagerPeerConfig {
                member_name: "bob".to_owned(),
                address: "tcp://127.0.0.1:7001".to_owned(),
            }],
        };
        let text = serde_json::to_string(&config).unwrap();
        assert_eq!(
            text,
            r#"{"sender":"leader","peers":[{"member_name":"bob","address":"tcp://127.0.0.1:7001"}]}"#
        );
        let back: InProcessMessagerConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, config);

        let sparse: InProcessMessagerConfig =
            serde_json::from_value(json!({"nope": true})).unwrap();
        assert_eq!(sparse, InProcessMessagerConfig::default());
    }

    #[test]
    fn pyzmq_config_round_trips_with_defaults() {
        let config = PyZmqMessagerConfig {
            team_name: "alpha".to_owned(),
            node_id: Some("leader".to_owned()),
            direct_addr: Some("tcp://127.0.0.1:7001".to_owned()),
            pubsub_publish_addr: Some("tcp://127.0.0.1:7002".to_owned()),
            pubsub_subscribe_addr: Some("tcp://127.0.0.1:7003".to_owned()),
            request_timeout_secs: 2.5,
            peers: Vec::new(),
        };
        let back: PyZmqMessagerConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(back, config);

        let default = PyZmqMessagerConfig::default();
        assert_eq!(default.team_name, "");
        assert_eq!(default.node_id, None);
        assert!((default.request_timeout_secs - 0.0).abs() < f64::EPSILON);
        assert!(default.peers.is_empty());
    }
}
