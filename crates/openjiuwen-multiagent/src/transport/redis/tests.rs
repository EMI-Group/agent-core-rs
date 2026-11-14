//! Unit tests of the Redis transport construction and configuration wiring.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// A handler every construction-only test needs.
struct NoopHandler;

#[async_trait]
impl MessageHandler for NoopHandler {
    async fn handle_p2p(&self, _envelope: &MessageEnvelope) -> Result<Value> {
        Ok(Value::Null)
    }

    async fn handle_pubsub(&self, _envelope: &MessageEnvelope, _subscriber: &str) -> Result<()> {
        Ok(())
    }
}

/// Builds a transport over `config`.
fn transport(config: RedisTransportConfig) -> Arc<RedisTransport> {
    RedisTransport::new(config, "team", "team", Arc::new(NoopHandler))
}

#[test]
fn try_new_rejects_a_non_single_topology() {
    for topology in [RedisTopology::Cluster, RedisTopology::Sentinel] {
        let config = RedisTransportConfig::new().with_topology(topology);
        let error =
            RedisTransport::try_new(config, "team", "team", Arc::new(NoopHandler)).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
    }
}

#[tokio::test]
async fn start_rejects_a_non_single_topology_before_connecting() {
    // The address is unreachable: reaching the guard before the handshake is
    // what makes this a topology error rather than a connection error.
    let config = RedisTransportConfig::new()
        .with_addr("127.0.0.1:1")
        .with_topology(RedisTopology::Sentinel);
    let error = transport(config).start().await.unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Unsupported
    );
    assert!(error.message().contains("Sentinel"), "{}", error.message());
}

#[test]
fn consumer_groups_are_derived_or_overridden() {
    let derived = transport(RedisTransportConfig::new());
    assert_eq!(derived.p2p_consumer_group(), "team:team:p2p:cg");
    assert_eq!(derived.pubsub_consumer_group(), "team:team:pubsub:cg");

    let overridden = transport(RedisTransportConfig::new().with_consumer_group("crew"));
    assert_eq!(overridden.p2p_consumer_group(), "crew");
    assert_eq!(overridden.pubsub_consumer_group(), "crew");
}

#[test]
fn consumer_names_are_distinct_instances() {
    let first = transport(RedisTransportConfig::new());
    let second = transport(RedisTransportConfig::new());
    assert_eq!(first.consumer_name().len(), 36);
    assert_ne!(first.consumer_name(), second.consumer_name());
}

#[test]
fn encode_envelope_keeps_routing_metadata() {
    let envelope = MessageEnvelope::p2p(serde_json::json!({"n": 1}), "a", "b").with_session("s1");
    let encoded = encode_envelope(&envelope).unwrap();
    let decoded: MessageEnvelope = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, envelope);
}
