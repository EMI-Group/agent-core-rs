//! Behaviour tests for [`CapabilityRegistry`]: CRUD, validation, deterministic
//! ordering, write-through persistence, and catalog search.
//!
//! Everything is offline and deterministic: the in-memory backend is process-local and
//! the file backend uses a `tempfile::TempDir`, so no test touches the network or a
//! shared location. Search internals are covered by `tests/search.rs`; here the point
//! is the registry integration (snapshot, ordering, filtering).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::sync::Arc;

use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityKey};
use openjiuwen_symphony::registry::CapabilityRegistry;
use openjiuwen_symphony::search::SearchRequest;
use openjiuwen_symphony::store::{AssetStore, KvAssetStore};

/// Builds a minimally complete descriptor.
fn descriptor(
    capability_type: &str,
    id: &str,
    name: &str,
    description: &str,
) -> CapabilityDescriptor {
    CapabilityDescriptor::new(capability_type, id, name)
        .with_description(description)
        .with_source("builtin")
        .with_available(true)
}

/// The catalog used by the search integration tests.
fn catalog() -> Vec<CapabilityDescriptor> {
    vec![
        descriptor("tool", "sum", "Sum Numbers", "adds two numbers together"),
        descriptor(
            "tool",
            "translate",
            "Translate Text",
            "translates text between languages",
        ),
        descriptor(
            "workflow",
            "weather",
            "Weather Forecast",
            "fetches the forecast for a city",
        ),
    ]
}

/// The qualified ids of `descriptors`, in order.
fn qualified_ids(descriptors: &[CapabilityDescriptor]) -> Vec<String> {
    descriptors
        .iter()
        .map(CapabilityDescriptor::qualified_id)
        .collect()
}

/// The qualified ids currently persisted in `store`, in order.
async fn stored_ids(store: &KvAssetStore) -> Vec<String> {
    qualified_ids(&store.list().await.expect("list store"))
}

// ---------------------------------------------------------------------------
// Construction and store wiring
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_new_registry_is_empty_and_has_no_store() {
    let registry = CapabilityRegistry::new();
    assert_eq!(registry.len(), 0);
    assert!(registry.is_empty());
    assert!(registry.list().await.is_empty());
    assert!(registry.store().is_none());
    assert_eq!(registry.reload().await.expect("reload"), 0);
}

#[tokio::test]
async fn with_store_starts_empty_until_reloaded() {
    let store = Arc::new(KvAssetStore::in_memory());
    store
        .put(&descriptor("tool", "sum", "Sum", "adds"))
        .await
        .expect("seed store");

    let registry = CapabilityRegistry::with_store(store);
    assert!(registry.store().is_some());
    assert!(
        registry.is_empty(),
        "with_store must not hydrate on its own"
    );

    assert_eq!(registry.reload().await.expect("reload"), 1);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.list().await[0].capability_id, "sum");
}

#[tokio::test]
async fn load_hydrates_the_catalog_from_the_store() {
    let store = Arc::new(KvAssetStore::in_memory());
    let mut descriptors = catalog();
    for descriptor in &descriptors {
        store.put(descriptor).await.expect("seed");
    }
    descriptors.sort_by_key(CapabilityDescriptor::qualified_id);

    let registry = CapabilityRegistry::load(store).await.expect("load");
    assert_eq!(registry.len(), descriptors.len());
    assert_eq!(
        qualified_ids(&registry.list().await),
        qualified_ids(&descriptors)
    );
}

#[tokio::test]
async fn reload_replaces_entries_deleted_from_the_store() {
    let store = Arc::new(KvAssetStore::in_memory());
    let registry = CapabilityRegistry::with_store(store.clone());
    registry
        .register(descriptor("tool", "sum", "Sum", "adds"))
        .await
        .expect("register");

    // An out-of-band delete must disappear on the next reload.
    store
        .remove(&CapabilityKey::new("tool", "sum"))
        .await
        .expect("store remove");
    assert_eq!(registry.len(), 1, "memory is not consulted for reload");
    assert_eq!(registry.reload().await.expect("reload"), 0);
    assert!(registry.is_empty());
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn register_get_contains_and_remove_round_trip() {
    let registry = CapabilityRegistry::new();
    let sum = descriptor("tool", "sum", "Sum", "adds");
    let key = CapabilityKey::new("tool", "sum");

    registry.register(sum.clone()).await.expect("register");
    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());
    assert!(registry.contains(&key).await);
    assert_eq!(registry.get(&key).await, Some(sum));
    assert!(registry
        .get(&CapabilityKey::new("tool", "missing"))
        .await
        .is_none());
    assert!(
        !registry
            .contains(&CapabilityKey::new("tool", "missing"))
            .await
    );

    assert!(registry.remove(&key).await.expect("remove"));
    assert_eq!(registry.len(), 0);
    assert!(registry.get(&key).await.is_none());
}

#[tokio::test]
async fn re_registering_replaces_the_descriptor() {
    let registry = CapabilityRegistry::new();
    let key = CapabilityKey::new("tool", "sum");
    registry
        .register(descriptor("tool", "sum", "Sum", "adds two numbers"))
        .await
        .expect("register v1");
    registry
        .register(
            descriptor("tool", "sum", "Sum", "adds two numbers")
                .with_description("adds any numbers")
                .with_available(false),
        )
        .await
        .expect("register v2");

    assert_eq!(registry.len(), 1, "same key must not duplicate");
    let stored = registry.get(&key).await.expect("present");
    assert_eq!(stored.description, "adds any numbers");
    assert!(!stored.available);
}

#[tokio::test]
async fn remove_of_a_missing_key_returns_false() {
    let registry = CapabilityRegistry::new();
    assert!(!registry
        .remove(&CapabilityKey::new("tool", "sum"))
        .await
        .expect("remove"));
    assert!(registry.is_empty());
}

#[tokio::test]
async fn clear_empties_the_catalog_and_the_store() {
    let store = Arc::new(KvAssetStore::in_memory());
    let registry = CapabilityRegistry::with_store(store.clone());
    registry
        .register_all(catalog())
        .await
        .expect("register all");
    assert_eq!(registry.len(), 3);

    registry.clear().await.expect("clear");
    assert!(registry.is_empty());
    assert!(stored_ids(&store).await.is_empty());
}

#[tokio::test]
async fn register_all_returns_the_registered_count() {
    let registry = CapabilityRegistry::new();
    let count = registry
        .register_all(catalog())
        .await
        .expect("register all");
    assert_eq!(count, 3);
    assert_eq!(registry.len(), 3);

    assert_eq!(
        registry
            .register_all(Vec::new())
            .await
            .expect("empty batch"),
        0
    );
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn validation_failures_name_the_field_and_leave_catalog_and_store_empty() {
    let store = Arc::new(KvAssetStore::in_memory());
    let registry = CapabilityRegistry::with_store(store.clone());

    let cases = [
        (descriptor("tool", "  ", "Sum", "adds"), "capability_id"),
        (descriptor(" ", "sum", "Sum", "adds"), "capability_type"),
        (descriptor("tool", "sum", "   ", "adds"), "name"),
    ];
    for (bad, field) in cases {
        let error = registry.register(bad).await.expect_err("must reject");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains(field),
            "error `{}` must name `{field}`",
            error.message()
        );
    }

    assert!(registry.is_empty(), "nothing may reach the catalog");
    assert!(
        stored_ids(&store).await.is_empty(),
        "nothing may reach the store"
    );
}

#[tokio::test]
async fn register_all_validates_every_descriptor_before_writing_anything() {
    let store = Arc::new(KvAssetStore::in_memory());
    let registry = CapabilityRegistry::with_store(store.clone());

    let batch = vec![
        descriptor("tool", "sum", "Sum", "adds"),
        descriptor("tool", "", "Broken", "no id"),
    ];
    let error = registry.register_all(batch).await.expect_err("must reject");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("capability_id"));

    assert!(
        registry.is_empty(),
        "a rejected batch must not be partially applied"
    );
    assert!(
        stored_ids(&store).await.is_empty(),
        "a rejected batch must not persist anything"
    );
}

// ---------------------------------------------------------------------------
// Deterministic ordering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_is_sorted_by_qualified_id_regardless_of_insertion_order() {
    let registry = CapabilityRegistry::new();
    let unsorted = [
        descriptor("workflow", "zeta", "Zeta", "last"),
        descriptor("tool", "beta", "Beta", "second"),
        descriptor("tool", "alpha", "Alpha", "first"),
        descriptor("agent", "gamma", "Gamma", "third"),
    ];
    for descriptor in unsorted {
        registry.register(descriptor).await.expect("register");
    }

    assert_eq!(
        qualified_ids(&registry.list().await),
        ["agent:gamma", "tool:alpha", "tool:beta", "workflow:zeta"]
    );
}

// ---------------------------------------------------------------------------
// Write-through persistence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn register_and_remove_write_through_to_the_store() {
    let store = Arc::new(KvAssetStore::in_memory());
    let registry = CapabilityRegistry::with_store(store.clone());
    let sum = descriptor("tool", "sum", "Sum", "adds");

    registry.register(sum.clone()).await.expect("register");
    assert_eq!(stored_ids(&store).await, ["tool:sum"]);
    assert_eq!(
        store
            .get(&sum.key())
            .await
            .expect("store get")
            .expect("persisted"),
        sum
    );

    assert!(registry
        .remove(&CapabilityKey::new("tool", "sum"))
        .await
        .expect("remove"));
    assert!(stored_ids(&store).await.is_empty());
}

#[tokio::test]
async fn json_file_store_survives_registry_reconstruction() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sum = descriptor("tool", "sum", "Sum", "adds two numbers");
    let translate = descriptor(
        "tool",
        "translate",
        "Translate Text",
        "translates text between languages",
    );

    {
        let store = Arc::new(KvAssetStore::json_file(dir.path()).await.expect("open"));
        let registry = CapabilityRegistry::load(store).await.expect("load");
        registry.register(sum.clone()).await.expect("register sum");
        registry
            .register(translate.clone())
            .await
            .expect("register translate");
    }

    let reopened = Arc::new(KvAssetStore::json_file(dir.path()).await.expect("reopen"));
    let hydrated = CapabilityRegistry::load(reopened.clone())
        .await
        .expect("hydrate");
    assert_eq!(
        qualified_ids(&hydrated.list().await),
        ["tool:sum", "tool:translate"]
    );
    assert_eq!(hydrated.get(&sum.key()).await, Some(sum.clone()));
    assert_eq!(
        hydrated.get(&translate.key()).await,
        Some(translate.clone())
    );

    // Removing through the registry must delete the persisted entry too.
    hydrated.remove(&sum.key()).await.expect("remove persisted");
    let reopened_again = Arc::new(KvAssetStore::json_file(dir.path()).await.expect("reopen 2"));
    assert_eq!(stored_ids(&reopened_again).await, ["tool:translate"]);
}

// ---------------------------------------------------------------------------
// Search integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_ranks_a_text_match_first() {
    let registry = CapabilityRegistry::new();
    registry
        .register_all(catalog())
        .await
        .expect("register all");

    let hits = registry
        .search(&SearchRequest::for_text("adds numbers"))
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].descriptor.qualified_id(), "tool:sum");
    assert!(hits[0].score > 0.0);
}

#[tokio::test]
async fn filter_only_search_scores_zero_and_orders_by_qualified_id() {
    let registry = CapabilityRegistry::new();
    registry
        .register_all(catalog())
        .await
        .expect("register all");

    let hits = registry
        .search(&SearchRequest::default())
        .await
        .expect("search");
    assert_eq!(
        hits.iter()
            .map(|hit| hit.descriptor.qualified_id())
            .collect::<Vec<_>>(),
        ["tool:sum", "tool:translate", "workflow:weather"]
    );
    assert!(hits.iter().all(|hit| hit.score == 0.0));

    let filtered = registry
        .search(&SearchRequest {
            capability_types: vec!["workflow".to_string()],
            ..SearchRequest::default()
        })
        .await
        .expect("filtered search");
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].descriptor.qualified_id(), "workflow:weather");
}

#[tokio::test]
async fn search_on_an_empty_registry_returns_nothing() {
    let registry = CapabilityRegistry::new();
    assert!(registry
        .search(&SearchRequest::for_text("anything"))
        .await
        .expect("search")
        .is_empty());
    assert!(registry
        .search(&SearchRequest::default())
        .await
        .expect("search")
        .is_empty());
}
