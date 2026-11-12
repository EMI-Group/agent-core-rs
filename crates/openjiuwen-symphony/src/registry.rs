//! The in-memory capability registry with write-through persistence and search.
//!
//! Python's symphony engine keeps no catalog of its own: the host supplies a
//! `CapabilityProvider` and the engine works on whatever snapshot it is handed. The
//! Rust port needs a durable catalog with CRUD and search, so [`CapabilityRegistry`]
//! is the authoritative in-memory map of [`CapabilityDescriptor`]s, optionally backed
//! by an [`AssetStore`] that every write is forwarded to.
//!
//! # Concurrency
//!
//! The map lives behind a [`std::sync::RwLock`]. The guard is only ever held for map
//! mutations and snapshot clones — never across an `.await` — so an async caller can
//! never block the runtime's reactor on the catalog. Store I/O happens outside the
//! lock: writes persist first and only then become visible in memory, so a store
//! failure leaves the in-memory catalog untouched.
//!
//! A poisoned lock is recovered with [`std::sync::PoisonError::into_inner`] instead of
//! panicking: every critical section is a single map operation, so the map itself is
//! still coherent, and a panicking sibling task must not take the whole catalog down.
//!
//! # Why the in-memory accessors are `async`
//!
//! The contract fixes `get`, `contains`, `list`, and `search` as `async` so a
//! future backend-aware implementation can await without a breaking signature change.
//! Clippy suggests dropping `async` from the paths that currently only read memory;
//! the public contract forbids that, so the lint is allowed module-wide.
#![allow(clippy::unused_async, clippy::unused_async_trait_impl)]

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::error::Result;
use crate::model::{CapabilityDescriptor, CapabilityKey};
use crate::search::{search_descriptors, SearchHit, SearchRequest};
use crate::store::AssetStore;

/// The in-memory catalog: keys map to descriptors, ordered by key.
type Entries = BTreeMap<CapabilityKey, CapabilityDescriptor>;

/// The authoritative capability catalog, optionally write-through to an [`AssetStore`].
///
/// Reads are served from memory; every mutation is first applied to the optional
/// store and then to the map, so persistence failures never desynchronize the
/// in-memory catalog (see [`Self::register`] and [`Self::remove`]).
///
/// `Clone` is intentionally not implemented: the store is a shared trait object, and
/// cloning a registry would silently create a second writable view of one catalog.
pub struct CapabilityRegistry {
    /// The authoritative descriptors, keyed by [`CapabilityKey`].
    entries: RwLock<Entries>,
    /// The optional persistence backend; writes are forwarded to it.
    store: Option<Arc<dyn AssetStore>>,
}

impl CapabilityRegistry {
    /// Builds an empty registry with no persistent store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Entries::new()),
            store: None,
        }
    }

    /// Builds an empty registry that writes through to `store`.
    ///
    /// The in-memory map starts empty; call [`Self::reload`] to hydrate it from
    /// previously persisted descriptors, or use [`Self::load`] to hydrate immediately.
    #[must_use]
    pub fn with_store(store: Arc<dyn AssetStore>) -> Self {
        Self {
            entries: RwLock::new(Entries::new()),
            store: Some(store),
        }
    }

    /// Builds a registry over `store` and hydrates it from the store's contents.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the store cannot be listed.
    pub async fn load(store: Arc<dyn AssetStore>) -> Result<Self> {
        let registry = Self::with_store(store);
        registry.reload().await?;
        Ok(registry)
    }

    /// Replaces the in-memory catalog with the store's contents and returns the count.
    ///
    /// Without a store this is a no-op that reports the current catalog size. With a
    /// store, entries that disappeared from the backend disappear from memory too.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the store cannot be listed.
    pub async fn reload(&self) -> Result<usize> {
        let Some(store) = &self.store else {
            return Ok(self.len());
        };
        let descriptors = store.list().await?;
        let mut replacement = Entries::new();
        for descriptor in descriptors {
            replacement.insert(descriptor.key(), descriptor);
        }
        let count = replacement.len();
        *self.write_entries() = replacement;
        Ok(count)
    }

    /// The store this registry writes through to, if any.
    #[must_use]
    pub fn store(&self) -> Option<&Arc<dyn AssetStore>> {
        self.store.as_ref()
    }

    /// The number of descriptors in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read_entries().len()
    }

    /// Reports whether the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read_entries().is_empty()
    }

    /// Validates and inserts `descriptor`, replacing any descriptor with the same key.
    ///
    /// The descriptor is validated first, then persisted, then inserted into memory, so
    /// neither a validation error nor a store failure mutates the in-memory catalog.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] when the descriptor's identity fields
    /// are blank, or the backend's error when the write fails.
    pub async fn register(&self, descriptor: CapabilityDescriptor) -> Result<()> {
        descriptor.validate()?;
        if let Some(store) = &self.store {
            store.put(&descriptor).await?;
        }
        self.write_entries().insert(descriptor.key(), descriptor);
        Ok(())
    }

    /// Registers every descriptor in `descriptors` and returns how many were inserted.
    ///
    /// Every descriptor is validated before anything is written, so a bad descriptor
    /// rejects the whole batch without touching the store or the catalog. A store
    /// failure part-way through can leave earlier descriptors persisted, but the
    /// in-memory catalog is not updated at all in that case.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] for the first invalid descriptor, or
    /// the backend's error when a write fails.
    pub async fn register_all(&self, descriptors: Vec<CapabilityDescriptor>) -> Result<usize> {
        for descriptor in &descriptors {
            descriptor.validate()?;
        }
        if let Some(store) = &self.store {
            for descriptor in &descriptors {
                store.put(descriptor).await?;
            }
        }
        let count = descriptors.len();
        let mut entries = self.write_entries();
        for descriptor in descriptors {
            entries.insert(descriptor.key(), descriptor);
        }
        Ok(count)
    }

    /// Returns a copy of the descriptor stored under `key`, or `None`.
    pub async fn get(&self, key: &CapabilityKey) -> Option<CapabilityDescriptor> {
        self.read_entries().get(key).cloned()
    }

    /// Reports whether `key` is present in the catalog.
    pub async fn contains(&self, key: &CapabilityKey) -> bool {
        self.read_entries().contains_key(key)
    }

    /// Removes `key` from the store and the catalog.
    ///
    /// Returns `true` only when the key was present in the in-memory catalog; a key that
    /// exists only in the store still gets deleted and reports `false`.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the delete fails.
    pub async fn remove(&self, key: &CapabilityKey) -> Result<bool> {
        if let Some(store) = &self.store {
            store.remove(key).await?;
        }
        Ok(self.write_entries().remove(key).is_some())
    }

    /// Removes every descriptor from the store and the catalog.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the store cannot be cleared.
    pub async fn clear(&self) -> Result<()> {
        if let Some(store) = &self.store {
            store.clear().await?;
        }
        self.write_entries().clear();
        Ok(())
    }

    /// Lists every descriptor, sorted by qualified id.
    pub async fn list(&self) -> Vec<CapabilityDescriptor> {
        self.snapshot()
    }

    /// Searches the catalog; see [`search_descriptors`] for the ranking rules.
    ///
    /// The catalog is snapshotted (and the lock released) before ranking, so a slow
    /// search never blocks concurrent registrations.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Validation`] when the request's `metadata_filter`
    /// is not a valid Mongo-style predicate.
    pub async fn search(&self, request: &SearchRequest) -> Result<Vec<SearchHit>> {
        let snapshot = self.snapshot();
        search_descriptors(&snapshot, request)
    }

    /// Locks the catalog for reading, recovering from a poisoned lock.
    fn read_entries(&self) -> RwLockReadGuard<'_, Entries> {
        self.entries.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the catalog for writing, recovering from a poisoned lock.
    fn write_entries(&self) -> RwLockWriteGuard<'_, Entries> {
        self.entries.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Clones the catalog's descriptors, sorted by qualified id, with the lock released.
    fn snapshot(&self) -> Vec<CapabilityDescriptor> {
        let entries = self.read_entries();
        let mut descriptors: Vec<CapabilityDescriptor> = entries.values().cloned().collect();
        drop(entries);
        descriptors.sort_by_key(CapabilityDescriptor::qualified_id);
        descriptors
    }
}

impl Default for CapabilityRegistry {
    /// Equivalent to [`CapabilityRegistry::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CapabilityRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self.read_entries();
        formatter
            .debug_struct("CapabilityRegistry")
            .field("entries", &*entries)
            .field("persistent", &self.store.is_some())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn descriptor(capability_type: &str, id: &str, name: &str) -> CapabilityDescriptor {
        CapabilityDescriptor::new(capability_type, id, name).with_description("does something")
    }

    #[tokio::test]
    async fn debug_reports_the_entries_and_store_presence() {
        let registry = CapabilityRegistry::new();
        registry
            .register(descriptor("tool", "sum", "Sum"))
            .await
            .expect("register");
        let rendered = format!("{registry:?}");
        assert!(rendered.contains("CapabilityRegistry"));
        assert!(rendered.contains("persistent: false"));
        assert!(rendered.contains("Sum"));
    }

    #[tokio::test]
    async fn reload_without_a_store_is_a_no_op() {
        let registry = CapabilityRegistry::new();
        registry
            .register(descriptor("tool", "sum", "Sum"))
            .await
            .expect("register");
        assert_eq!(registry.reload().await.expect("reload"), 1);
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn default_matches_new() {
        let registry = CapabilityRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.store().is_none());
    }
}
