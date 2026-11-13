//! External memory-provider SPI integration tests.
//!
//! Drives the `openjiuwen_memory::external` surface as it is wired into the
//! `LongTermMemory` facade: the `provider()` view over the shared engine, capability
//! and health reporting, scope isolation, the vendor stubs' typed `Unsupported`
//! behaviour, credential redaction in `ProviderConfig`, registry semantics
//! (defaults, replacement, `select`, custom kinds and factories), and cancellation.
//!
//! Every test builds its own in-memory facade and its own registry, so nothing needs
//! the network, a credential, an external service, or the wall clock.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty
)]

use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::REDACTED;
use openjiuwen_memory::config::MemoryEngineConfig;
use openjiuwen_memory::engine::MemoryEngine;
use openjiuwen_memory::external::{
    EngineProvider, LocalProviderFactory, Mem0MemoryProvider, MemoryProvider, ProviderCapabilities,
    ProviderConfig, ProviderFactory, ProviderKind, ProviderRegistry,
};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryScope, MemoryUpdate, NewMemory, SearchMode,
    SearchOptions,
};
use openjiuwen_retrieval::embedding::HashEmbedding;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The custom provider id used by the unknown-kind test.
const ACME: &str = "acme_memory";

/// The vendor stubs and the SDK/transport token each error message must name.
const VENDORS: [(ProviderKind, &str, &str); 5] = [
    (ProviderKind::AgentArts, "agentarts", "agentarts-sdk"),
    (ProviderKind::Jiuwen, "jiuwen_memory", "jiuwen_memory"),
    (ProviderKind::LakeBase, "lakebase", "httpx.AsyncClient"),
    (ProviderKind::Mem0, "mem0", "mem0ai"),
    (
        ProviderKind::OpenViking,
        "openviking",
        "OpenViking API client",
    ),
];

/// A factory for a kind this crate does not know, proving the `Custom` escape hatch.
struct AcmeFactory {
    /// The engine every built provider shares with the facade.
    engine: Arc<MemoryEngine>,
}

impl ProviderFactory for AcmeFactory {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Custom(ACME.to_owned())
    }

    fn build(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>> {
        if config.kind != self.kind() {
            return Err(Error::validation(format!(
                "the acme factory cannot build kind `{}`",
                config.kind
            )));
        }
        Ok(Arc::new(EngineProvider::with_name(
            Arc::clone(&self.engine),
            config.provider_name(),
        )))
    }
}

/// Builds an empty in-memory facade.
async fn facade() -> Arc<LongTermMemory> {
    LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("in-memory facade")
}

/// The scope most tests write under.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// A provider over `engine` that reports `name`.
fn named_provider(engine: &Arc<MemoryEngine>, name: &str) -> Arc<dyn MemoryProvider> {
    Arc::new(EngineProvider::with_name(Arc::clone(engine), name))
}

/// Keyword search options that cannot filter a candidate out on its blended score.
fn any_hit() -> SearchOptions {
    SearchOptions::default().with_threshold(0.0)
}

/// Asserts that `error` is a typed `Unsupported` naming `missing` and `id`.
fn assert_missing(error: &Error, id: &str, missing: &str) {
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains(missing),
        "`{message}` must name `{missing}`"
    );
    assert!(
        message.contains("does not provide"),
        "`{message}` must state the dependency is absent"
    );
    // The vendor appears under its display name (`LakeBase`, `mem0`, …).
    assert!(
        message.to_lowercase().contains(&id.to_lowercase()),
        "`{message}` must name the vendor `{id}`"
    );
}

/// Runs every SPI operation of a vendor stub and asserts each one is `Unsupported`.
async fn assert_vendor_surface_unsupported(provider: &dyn MemoryProvider, id: &str, missing: &str) {
    let scope = MemoryScope::new("alice", "ws");

    let error = provider.initialize().await.expect_err("initialize");
    assert_missing(&error, id, missing);

    let error = provider
        .add(
            &scope,
            vec![NewMemory::semantic("a vendor write")],
            &AddOptions::default(),
        )
        .await
        .expect_err("add");
    assert_missing(&error, id, missing);

    let error = provider
        .search(&scope, "a vendor query", &SearchOptions::default())
        .await
        .expect_err("search");
    assert_missing(&error, id, missing);

    let error = provider
        .get(&scope, "0123456789abcdef01234567")
        .await
        .expect_err("get");
    assert_missing(&error, id, missing);

    let error = provider
        .update(
            &scope,
            "0123456789abcdef01234567",
            MemoryUpdate::content("a vendor update"),
            &CallOptions::new(),
        )
        .await
        .expect_err("update");
    assert_missing(&error, id, missing);

    let error = provider
        .delete(&scope, "0123456789abcdef01234567")
        .await
        .expect_err("delete");
    assert_missing(&error, id, missing);

    let error = provider
        .list(&MemoryFilter::default())
        .await
        .expect_err("list");
    assert_missing(&error, id, missing);

    let error = provider
        .clear(&MemoryFilter::default())
        .await
        .expect_err("clear");
    assert_missing(&error, id, missing);

    let error = provider.health().await.expect_err("health");
    assert_missing(&error, id, missing);
}

#[tokio::test]
async fn provider_view_round_trips_crud_and_shares_the_engine() {
    let ltm = facade().await;
    let scope = scope();
    let provider = ltm.provider();
    assert_eq!(provider.name(), "local");
    provider.initialize().await.expect("initialize");
    // Initialization is idempotent.
    provider.initialize().await.expect("initialize again");

    let outcome = provider
        .add(
            &scope,
            vec![NewMemory::semantic(
                "alice prefers dark mode for every editor",
            )],
            &AddOptions::default(),
        )
        .await
        .expect("provider add");
    assert_eq!(outcome.added.len(), 1);
    let id = outcome.added[0].id.clone();

    // The provider is a view of the facade's engine, not a copy of its state.
    let via_engine = ltm
        .engine()
        .get(&scope, &id)
        .await
        .expect("engine get")
        .expect("record through the engine");
    assert_eq!(via_engine.id, id);
    assert_eq!(ltm.list(&scope, &[]).await.expect("facade list").len(), 1);

    let hits = provider
        .search(&scope, "dark mode editor", &any_hit())
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, id);
    assert_eq!(
        hits[0].record.content,
        "alice prefers dark mode for every editor"
    );

    let stored = provider
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("record through the provider");
    assert_eq!(stored.content, "alice prefers dark mode for every editor");

    let updated = provider
        .update(
            &scope,
            &id,
            MemoryUpdate::content("alice switched to the light theme"),
            &CallOptions::new(),
        )
        .await
        .expect("update");
    assert_eq!(updated.id, id);
    assert_eq!(updated.content, "alice switched to the light theme");
    let read_back = provider
        .get(&scope, &id)
        .await
        .expect("get updated")
        .expect("record");
    assert_eq!(read_back.content, "alice switched to the light theme");
    let hits = provider
        .search(&scope, "light theme", &any_hit())
        .await
        .expect("search updated");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "alice switched to the light theme");
    assert_eq!(
        ltm.get(&scope, &id)
            .await
            .expect("facade get")
            .expect("record through the facade")
            .content,
        "alice switched to the light theme"
    );

    assert!(provider.delete(&scope, &id).await.expect("delete"));
    assert!(!provider.delete(&scope, &id).await.expect("second delete"));
    assert!(provider
        .get(&scope, &id)
        .await
        .expect("get deleted")
        .is_none());
    assert!(ltm
        .engine()
        .get(&scope, &id)
        .await
        .expect("engine get deleted")
        .is_none());
    assert!(ltm.list(&scope, &[]).await.expect("facade list").is_empty());
    assert!(provider
        .list(&MemoryFilter::default())
        .await
        .expect("provider list")
        .is_empty());
}

#[tokio::test]
async fn capabilities_and_health_track_the_embedding_model() {
    let keyword_only = facade().await;
    let provider = keyword_only.provider();
    let caps = provider.capabilities();
    assert!(!caps.semantic_search, "no embedding model is attached");
    assert!(caps.keyword_search);
    assert!(!caps.graph);
    assert!(!caps.variables);
    assert!(!caps.message_history);
    assert!(!caps.pagination);
    assert!(caps.max_batch.is_none());

    // The declared capability matches the behavior of the adapter.
    let error = provider
        .search(
            &scope(),
            "dark mode",
            &SearchOptions::default().with_mode(SearchMode::Semantic),
        )
        .await
        .expect_err("semantic recall without an embedding model");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");

    let health = provider.health().await.expect("health");
    assert!(health.healthy, "{health:?}");
    assert_eq!(
        health.detail.as_deref(),
        Some("in-process engine reachable")
    );

    let semantic_ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .embedding(Arc::new(HashEmbedding::new(8)))
        .build()
        .expect("semantic facade");
    let semantic_provider = semantic_ltm.provider();
    let caps = semantic_provider.capabilities();
    assert!(caps.semantic_search, "an embedding model enables semantics");
    assert!(caps.keyword_search);
    assert!(semantic_provider.health().await.expect("health").healthy);
}

#[tokio::test]
async fn provider_keeps_scopes_isolated() {
    let ltm = facade().await;
    let provider = ltm.provider();
    let alice = MemoryScope::new("alice", "ws");
    let bob = MemoryScope::new("bob", "ws");

    provider
        .add(
            &alice,
            vec![NewMemory::semantic("alice prefers dark mode")],
            &AddOptions::default(),
        )
        .await
        .expect("alice add");
    provider
        .add(
            &bob,
            vec![NewMemory::semantic("bob prefers light mode")],
            &AddOptions::default(),
        )
        .await
        .expect("bob add");

    let hits = provider
        .search(&alice, "mode", &any_hit())
        .await
        .expect("alice search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.scope, alice);
    let hits = provider
        .search(&bob, "mode", &any_hit())
        .await
        .expect("bob search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.scope, bob);

    assert_eq!(
        provider
            .list(&MemoryFilter::for_scope(&alice))
            .await
            .expect("alice list")
            .len(),
        1
    );
    assert_eq!(
        provider
            .list(&MemoryFilter::for_scope(&bob))
            .await
            .expect("bob list")
            .len(),
        1
    );

    // A third user sees neither side, in recall or in listing.
    let carol = MemoryScope::new("carol", "ws");
    assert!(provider
        .search(&carol, "mode", &any_hit())
        .await
        .expect("carol search")
        .is_empty());
    assert!(provider
        .list(&MemoryFilter::for_scope(&carol))
        .await
        .expect("carol list")
        .is_empty());
}

#[tokio::test]
async fn vendor_stubs_report_unsupported_and_keep_their_identity() {
    let registry = ProviderRegistry::with_vendor_stubs();
    for (kind, id, missing) in VENDORS {
        let provider = registry
            .build(&ProviderConfig::new(kind))
            .unwrap_or_else(|error| panic!("{id} must build: {error}"));
        assert_eq!(provider.name(), id);
        assert_eq!(
            provider.capabilities(),
            ProviderCapabilities::none(),
            "{id} declares no capability"
        );
        assert_vendor_surface_unsupported(provider.as_ref(), id, missing).await;
    }

    // The concrete handle is debuggable without leaking its credential.
    let config = ProviderConfig::new(ProviderKind::Mem0).with_api_key("sk-vendor-secret");
    let handle = Mem0MemoryProvider::new(&config);
    assert_eq!(handle.name(), "mem0");
    assert_eq!(handle.kind(), ProviderKind::Mem0);
    let debug = format!("{handle:?}");
    assert!(debug.contains("mem0"), "{debug}");
    assert!(!debug.contains("sk-vendor-secret"), "{debug}");
}

#[tokio::test]
async fn provider_config_debug_redacts_credentials_and_sensitive_options() {
    let config = ProviderConfig::new(ProviderKind::Mem0)
        .with_endpoint("https://api.mem0.invalid")
        .with_api_key("sk-super-secret")
        .with_option("user_id", json!("alice"))
        .with_option("password", json!("hunter2-secret"));

    let debug = format!("{config:?}");
    assert!(debug.contains(REDACTED), "{debug}");
    assert!(!debug.contains("sk-super-secret"), "{debug}");
    assert!(!debug.contains("hunter2-secret"), "{debug}");
    // Non-sensitive fields stay readable.
    assert!(debug.contains("alice"), "{debug}");
    assert!(debug.contains("https://api.mem0.invalid"), "{debug}");

    // Only the rendering is redacted: the value keeps its credential for use.
    assert_eq!(config.api_key.as_deref(), Some("sk-super-secret"));
    assert_eq!(config.option("password"), Some(&json!("hunter2-secret")));

    let redacted = config.redacted();
    assert_eq!(redacted.api_key.as_deref(), Some(REDACTED));
    assert_eq!(redacted.endpoint, config.endpoint);
    let debug = format!("{redacted:?}");
    assert!(debug.contains(REDACTED), "{debug}");
    assert!(!debug.contains("sk-super-secret"), "{debug}");
    assert!(!debug.contains("hunter2-secret"), "{debug}");

    // Serialization deliberately keeps the credential so a config round-trips.
    let serialized = serde_json::to_value(&config).expect("serialize");
    assert_eq!(serialized["api_key"], json!("sk-super-secret"));
}

#[tokio::test]
async fn registry_last_registration_wins_and_select_sets_the_first_default() {
    let ltm = facade().await;
    let engine = ltm.engine();

    let registry = ProviderRegistry::new();
    assert!(registry
        .register("zeta", named_provider(engine, "v1"))
        .is_none());
    assert_eq!(registry.default_provider().expect("default").name(), "v1");
    assert!(registry
        .register("alpha", named_provider(engine, "v2"))
        .is_none());
    // Names come back sorted, and the first registration stays the default.
    assert_eq!(registry.names(), vec!["alpha", "zeta"]);
    assert_eq!(registry.default_provider().expect("default").name(), "v1");

    // Re-registering a name replaces the instance; the default key now resolves to it.
    let previous = registry
        .register("zeta", named_provider(engine, "v3"))
        .expect("duplicate registration returns the previous instance");
    assert_eq!(previous.name(), "v1");
    assert_eq!(registry.get("zeta").expect("zeta").name(), "v3");
    assert_eq!(registry.default_provider().expect("default").name(), "v3");
    assert!(registry.get("absent").is_none());
    let error = registry.set_default("absent").expect_err("not registered");
    assert!(matches!(error, Error::NotFound(_)), "{error:?}");

    // `select` on a registry with no default registers the built provider under
    // `provider_name` and makes it the default.
    let selectable = ProviderRegistry::new();
    selectable.register_factory(Arc::new(LocalProviderFactory::new(Arc::clone(engine))));
    let selected = selectable
        .select(&ProviderConfig::new(ProviderKind::Local))
        .expect("select local");
    assert_eq!(selected.name(), "local");
    assert_eq!(selectable.names(), vec!["local"]);
    assert_eq!(
        selectable.default_provider().expect("default").name(),
        "local"
    );

    // A later `select` registers the new instance but leaves the existing default in
    // place, matching the documented first-registration rule; `set_default` moves it.
    let second = selectable
        .select(&ProviderConfig::new(ProviderKind::Local).with_name("secondary"))
        .expect("select named");
    assert_eq!(second.name(), "secondary");
    assert_eq!(selectable.names(), vec!["local", "secondary"]);
    assert_eq!(
        selectable.default_provider().expect("default").name(),
        "local"
    );
    selectable.set_default("secondary").expect("registered");
    assert_eq!(
        selectable.default_provider().expect("default").name(),
        "secondary"
    );
}

#[tokio::test]
async fn unknown_kinds_deserialize_to_custom_and_a_custom_factory_can_be_selected() {
    let kind: ProviderKind = serde_json::from_value(json!(ACME)).expect("deserialize");
    assert_eq!(kind, ProviderKind::Custom(ACME.to_owned()));
    assert_eq!(kind.as_str(), ACME);
    // Known ids keep their Python spelling on the way in.
    assert_eq!(ProviderKind::from_name(" mem0 "), ProviderKind::Mem0);

    let ltm = facade().await;
    let registry = ProviderRegistry::new();
    assert!(registry
        .register_factory(Arc::new(AcmeFactory {
            engine: Arc::clone(ltm.engine()),
        }))
        .is_none());
    let provider = registry
        .select(&ProviderConfig::new(kind).with_name("acme"))
        .expect("custom select");
    assert_eq!(provider.name(), "acme");
    assert_eq!(registry.names(), vec!["acme"]);
    assert_eq!(registry.default_provider().expect("default").name(), "acme");

    // The custom provider is a working engine view, not a stub.
    provider
        .add(
            &scope(),
            vec![NewMemory::semantic(
                "the acme memory backend is the local engine",
            )],
            &AddOptions::default(),
        )
        .await
        .expect("add through the custom provider");
    assert_eq!(ltm.list(&scope(), &[]).await.expect("list").len(), 1);
    let hits = provider
        .search(&scope(), "acme backend", &any_hit())
        .await
        .expect("search through the custom provider");
    assert_eq!(hits.len(), 1);
}

#[tokio::test]
async fn cancelled_provider_calls_leave_nothing_written() {
    let ltm = facade().await;
    let provider = ltm.provider();
    let scope = scope();
    let token = CancellationToken::new();
    token.cancel();

    let error = provider
        .add(
            &scope,
            vec![NewMemory::semantic("a write that must not land")],
            &AddOptions::default().with_cancellation(token.clone()),
        )
        .await
        .expect_err("cancelled add");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert!(ltm.list(&scope, &[]).await.expect("list").is_empty());
    assert!(provider
        .list(&MemoryFilter::default())
        .await
        .expect("provider list")
        .is_empty());

    let error = provider
        .search(
            &scope,
            "a write that must not land",
            &any_hit().with_cancellation(token),
        )
        .await
        .expect_err("cancelled search");
    assert!(matches!(error, Error::Cancelled), "{error:?}");

    // The engine is still usable: cancellation changed no state.
    provider
        .add(
            &scope,
            vec![NewMemory::semantic("a live write")],
            &AddOptions::default(),
        )
        .await
        .expect("live add");
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}
