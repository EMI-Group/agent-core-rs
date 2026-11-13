//! The generic typed registry behind [`ResourceMgr`](super::ResourceMgr).
//!
//! One [`Registry`] stores `<card, provider>` entries keyed on the card id:
//! the card is the serializable identity (`id`, `name`) and the provider is the
//! factory invoked on every lookup, so a registered resource can build per-run
//! state instead of being pinned to one live instance.
//!
//! Python parity: `ResourceMgr` in
//! `openjiuwen/core/runner/resources_manager/resource_manager.py` keys every
//! resource manager on `card.id` and calls the provider on
//! `get_agent`/`get_workflow`. The Rust port keeps that contract and adds the
//! semantics hosts actually need: an explicit [`RegisterPolicy`] in place of
//! Python's `refresh`/`skip_if_exists` keyword soup, card-vs-name lookup, and a
//! monotonically increasing [`Registry::revision`] for cache invalidation.
//!
//! Interior mutability is a `parking_lot::RwLock`: the lock guards only
//! synchronous map/vector updates and card clones, and a provider is always
//! taken out of the guard and invoked **after** the lock is released.

use std::collections::HashMap;
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::ToolCard;
use openjiuwen_multiagent::TeamCard;
use openjiuwen_sysop::SysOperationCard;
use openjiuwen_workflow::WorkflowCard;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{already_registered, invalid_id};

/// The card-like identity a [`Registry`] keys on.
///
/// Implemented for [`ToolCard`], [`AgentCard`], [`WorkflowCard`],
/// [`TeamCard`], and [`SysOperationCard`]. A card is cheap to clone and
/// immutable after construction; the registry stores a clone and never mutates
/// it.
pub trait ResourceCard: Clone + Send + Sync + 'static {
    /// Returns the unique identifier the registry keys on.
    fn id(&self) -> &str;

    /// Returns the human-readable name (`by_name` lookups scan these).
    fn name(&self) -> &str;
}

impl ResourceCard for ToolCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ResourceCard for AgentCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ResourceCard for WorkflowCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ResourceCard for TeamCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ResourceCard for SysOperationCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Factory invoked once per lookup, returning the resource to hand out.
///
/// Python parity: `ResourceMgr` calls `provider()` on every `get_*`, so a
/// provider may return a fresh instance (and is expected to for agents,
/// workflows, and teams). The closure is `Send + Sync` because the registry is
/// shared across the runner's tasks.
pub type Provider<T> = Arc<dyn Fn() -> T + Send + Sync>;

/// What to do when a card id is already registered.
///
/// Python parity: `Fail` is the default (`RESOURCE_ADD_ERROR`), `Keep` mirrors
/// `skip_if_exists=True`, and `Replace` mirrors `refresh=True`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterPolicy {
    /// Reject the registration with [`openjiuwen_common::error::Error::Validation`]
    /// and leave the existing entry untouched.
    #[default]
    Fail,
    /// Keep the existing entry and report the registration as not replaced.
    Keep,
    /// Swap in the new card and provider, reporting `replaced: true`.
    Replace,
}

/// The outcome of one discovery-registered resource.
///
/// `name` is the card name that ended up in the registry (the existing one
/// under [`RegisterPolicy::Keep`], the new one otherwise).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Registration {
    /// The card id that was registered (or already present).
    pub id: String,
    /// The card name that ended up in the registry.
    pub name: String,
    /// Whether an existing entry was replaced by this call.
    pub replaced: bool,
}

/// One stored resource: the serializable identity plus its factory.
struct Entry<C, T> {
    card: C,
    provider: Provider<T>,
}

/// The lock-guarded state of a [`Registry`].
struct Inner<C, T> {
    /// Entries in registration order (`cards()` preserves it).
    entries: Vec<Entry<C, T>>,
    /// Id -> position in `entries`.
    index: HashMap<String, usize>,
    /// Mutation counter; `+1` on every successful mutation.
    revision: u64,
}

/// A typed registry of `<card, provider>` entries keyed on the card id.
///
/// Registration order is preserved (a `Vec`), and removal compacts it so
/// `cards()` keeps the relative order of the survivors. The revision counter
/// starts at zero and increments on every successful mutation (fresh
/// registration, replacement, removal, and a non-empty clear); a rejected
/// duplicate and a no-op [`Registry::clear`] do not change it.
pub struct Registry<C: ResourceCard, T> {
    /// The human noun used in error messages (`"tool"`, `"agent"`, …).
    kind: &'static str,
    inner: RwLock<Inner<C, T>>,
}

impl<C: ResourceCard, T> Registry<C, T> {
    /// Builds an empty registry whose errors name the resource `"resource"`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_kind("resource")
    }

    /// Builds an empty registry whose error messages use `kind` as the noun.
    ///
    /// [`ResourceMgr`](super::ResourceMgr) passes `"tool"`, `"agent"`,
    /// `"workflow"`, and `"team"` so a duplicate reads
    /// `agent 'assistant' is already registered`.
    #[must_use]
    pub fn with_kind(kind: &'static str) -> Self {
        Self {
            kind,
            inner: RwLock::new(Inner {
                entries: Vec::new(),
                index: HashMap::new(),
                revision: 0,
            }),
        }
    }

    /// Registers `card` with `provider` under `policy`.
    ///
    /// The id comes from `card.id()`; an empty or whitespace-only id fails with
    /// [`openjiuwen_common::error::ErrorKind::Validation`] and nothing is
    /// stored. A duplicate id follows `policy`: `Fail` keeps the first entry and
    /// returns an error, `Keep` keeps it and returns `replaced: false`, and
    /// `Replace` swaps both card and provider and returns `replaced: true`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an invalid id and,
    /// under [`RegisterPolicy::Fail`], for a duplicate id.
    pub fn register(
        &self,
        card: C,
        provider: Provider<T>,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        let id = card.id().to_owned();
        if id.trim().is_empty() {
            return Err(invalid_id(self.kind));
        }

        let mut inner = self.inner.write();
        if let Some(position) = inner.index.get(&id).copied() {
            return match policy {
                RegisterPolicy::Fail => Err(already_registered(self.kind, &id)),
                RegisterPolicy::Keep => Ok(Registration {
                    name: inner
                        .entries
                        .get(position)
                        .map_or_else(String::new, |entry| entry.card.name().to_owned()),
                    id,
                    replaced: false,
                }),
                RegisterPolicy::Replace => {
                    let name = card.name().to_owned();
                    inner.entries[position] = Entry { card, provider };
                    inner.revision = inner.revision.saturating_add(1);
                    Ok(Registration {
                        id,
                        name,
                        replaced: true,
                    })
                }
            };
        }

        let name = card.name().to_owned();
        let position = inner.entries.len();
        inner.entries.push(Entry { card, provider });
        inner.index.insert(id.clone(), position);
        inner.revision = inner.revision.saturating_add(1);
        Ok(Registration {
            id,
            name,
            replaced: false,
        })
    }

    /// Returns the resource for `id` by invoking its provider.
    ///
    /// The provider runs **after** the registry lock is released, so a provider
    /// may itself touch the registry without deadlocking. A missing id returns
    /// `None` and invokes nothing.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<T> {
        let provider = {
            let inner = self.inner.read();
            inner
                .index
                .get(id)
                .and_then(|position| inner.entries.get(*position))
                .map(|entry| Arc::clone(&entry.provider))
        }?;
        Some(provider())
    }

    /// Returns a clone of the card registered under `id`.
    #[must_use]
    pub fn card(&self, id: &str) -> Option<C> {
        let inner = self.inner.read();
        inner
            .index
            .get(id)
            .and_then(|position| inner.entries.get(*position))
            .map(|entry| entry.card.clone())
    }

    /// Returns the id of the first card registered under `name`.
    ///
    /// Names are not unique: `by_name` is a scan in registration order and the
    /// first match wins. Callers that need uniqueness use ids.
    #[must_use]
    pub fn id_by_name(&self, name: &str) -> Option<String> {
        let inner = self.inner.read();
        inner
            .entries
            .iter()
            .find(|entry| entry.card.name() == name)
            .map(|entry| entry.card.id().to_owned())
    }

    /// Returns the resource whose card name is `name` (first registration wins).
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<T> {
        self.id_by_name(name).and_then(|id| self.get(&id))
    }

    /// Removes `id` and returns its card.
    ///
    /// The relative order of the remaining cards is preserved. A missing id
    /// returns `None` and leaves the revision untouched.
    #[must_use]
    pub fn remove(&self, id: &str) -> Option<C> {
        let mut inner = self.inner.write();
        // Reborrow to a plain `&mut Inner` so the `entries` and `index` fields
        // can be borrowed disjointly while the shifted ids are reindexed.
        let inner: &mut Inner<C, T> = &mut inner;
        let position = inner.index.remove(id)?;
        let entry = inner.entries.remove(position);
        for (offset, shifted) in inner.entries.iter().enumerate().skip(position) {
            inner.index.insert(shifted.card.id().to_owned(), offset);
        }
        inner.revision = inner.revision.saturating_add(1);
        Some(entry.card)
    }

    /// Returns whether `id` is registered.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.inner.read().index.contains_key(id)
    }

    /// Returns clones of the registered cards in registration order.
    #[must_use]
    pub fn cards(&self) -> Vec<C> {
        self.inner
            .read()
            .entries
            .iter()
            .map(|entry| entry.card.clone())
            .collect()
    }

    /// Returns the registered ids in registration order.
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        self.inner
            .read()
            .entries
            .iter()
            .map(|entry| entry.card.id().to_owned())
            .collect()
    }

    /// Returns the number of registered resources.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    /// Returns whether the registry holds no resources.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.read().entries.is_empty()
    }

    /// Removes every entry and returns how many were removed.
    ///
    /// A non-empty clear increments the revision by one; clearing an empty
    /// registry is a no-op.
    pub fn clear(&self) -> usize {
        let mut inner = self.inner.write();
        let removed = Self::drain(&mut inner);
        if removed > 0 {
            inner.revision = inner.revision.saturating_add(1);
        }
        removed
    }

    /// Returns the mutation counter of this registry.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.inner.read().revision
    }

    /// Removes every entry and resets the revision counter to zero.
    ///
    /// Used by [`ResourceMgr::release`](super::ResourceMgr::release), which
    /// Python's `Runner::stop` calls to return the process to a pristine
    /// registry state.
    pub(crate) fn reset(&self) -> usize {
        let mut inner = self.inner.write();
        let removed = Self::drain(&mut inner);
        inner.revision = 0;
        removed
    }

    /// Clears the entries and the id index, returning how many were removed.
    fn drain(inner: &mut Inner<C, T>) -> usize {
        let removed = inner.entries.len();
        inner.entries.clear();
        inner.index.clear();
        removed
    }
}

impl<C: ResourceCard, T> Default for Registry<C, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: ResourceCard, T> std::fmt::Debug for Registry<C, T> {
    /// Prints the kind, ids in registration order, and the revision.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        let ids: Vec<&str> = inner.entries.iter().map(|entry| entry.card.id()).collect();
        formatter
            .debug_struct("Registry")
            .field("kind", &self.kind)
            .field("ids", &ids)
            .field("revision", &inner.revision)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The minimal card shape the generic machinery needs.
    #[derive(Clone, Debug, PartialEq)]
    struct TestCard {
        id: String,
        name: String,
    }

    impl TestCard {
        fn new(id: &str, name: &str) -> Self {
            Self {
                id: id.to_owned(),
                name: name.to_owned(),
            }
        }
    }

    impl ResourceCard for TestCard {
        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    /// A provider returning a fixed label.
    fn label(value: &'static str) -> Provider<String> {
        Arc::new(move || value.to_owned())
    }

    /// A provider counting how many times it ran and returning the count.
    fn counting(calls: &Arc<AtomicUsize>) -> Provider<String> {
        let calls = Arc::clone(calls);
        Arc::new(move || {
            calls.fetch_add(1, Ordering::SeqCst);
            "built".to_owned()
        })
    }

    fn registry() -> Registry<TestCard, String> {
        Registry::with_kind("widget")
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn fail_policy_keeps_the_first_and_reports_validation() {
        let registry = registry();
        let first = registry
            .register(
                TestCard::new("w1", "one"),
                label("first"),
                RegisterPolicy::Fail,
            )
            .unwrap();
        assert_eq!(
            first,
            Registration {
                id: "w1".to_owned(),
                name: "one".to_owned(),
                replaced: false,
            }
        );

        let error = registry
            .register(
                TestCard::new("w1", "duplicate"),
                label("second"),
                RegisterPolicy::Fail,
            )
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: widget 'w1' is already registered"
        );

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get("w1").as_deref(), Some("first"));
        assert_eq!(registry.card("w1").unwrap().name, "one");
        assert_eq!(
            registry.revision(),
            1,
            "a rejected duplicate is not a mutation"
        );
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn keep_policy_leaves_the_original_entry() {
        let registry = registry();
        registry
            .register(
                TestCard::new("w1", "one"),
                label("first"),
                RegisterPolicy::Fail,
            )
            .unwrap();

        let kept = registry
            .register(
                TestCard::new("w1", "ignored"),
                label("second"),
                RegisterPolicy::Keep,
            )
            .unwrap();
        assert_eq!(
            kept,
            Registration {
                id: "w1".to_owned(),
                name: "one".to_owned(),
                replaced: false,
            }
        );
        assert_eq!(registry.get("w1").as_deref(), Some("first"));
        assert_eq!(registry.card("w1").unwrap().name, "one");
        assert_eq!(registry.revision(), 1);
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn replace_policy_swaps_the_card_and_the_provider() {
        let registry = registry();
        registry
            .register(
                TestCard::new("w1", "one"),
                label("first"),
                RegisterPolicy::Fail,
            )
            .unwrap();

        let replaced = registry
            .register(
                TestCard::new("w1", "two"),
                label("second"),
                RegisterPolicy::Replace,
            )
            .unwrap();
        assert_eq!(
            replaced,
            Registration {
                id: "w1".to_owned(),
                name: "two".to_owned(),
                replaced: true,
            }
        );
        assert_eq!(registry.get("w1").as_deref(), Some("second"));
        assert_eq!(registry.card("w1").unwrap().name, "two");
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.revision(), 2);
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn the_provider_runs_on_every_lookup_and_never_for_a_missing_id() {
        let registry = registry();
        let calls = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                TestCard::new("w1", "one"),
                counting(&calls),
                RegisterPolicy::Fail,
            )
            .unwrap();

        assert!(registry.get("missing").is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        assert_eq!(registry.get("w1").as_deref(), Some("built"));
        assert_eq!(registry.get("w1").as_deref(), Some("built"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn lookups_by_name_scan_in_registration_order() {
        let registry = registry();
        registry
            .register(
                TestCard::new("w1", "shared"),
                label("first"),
                RegisterPolicy::Fail,
            )
            .unwrap();
        registry
            .register(
                TestCard::new("w2", "other"),
                label("second"),
                RegisterPolicy::Fail,
            )
            .unwrap();
        registry
            .register(
                TestCard::new("w3", "shared"),
                label("third"),
                RegisterPolicy::Fail,
            )
            .unwrap();

        assert_eq!(registry.id_by_name("shared").as_deref(), Some("w1"));
        assert_eq!(registry.get_by_name("shared").as_deref(), Some("first"));
        assert_eq!(registry.get_by_name("absent"), None);
        assert_eq!(registry.get_by_name("other").as_deref(), Some("second"));
    }

    #[test]
    fn cards_and_ids_preserve_registration_order() {
        let registry = registry();
        for (id, name) in [("a", "alpha"), ("b", "beta"), ("c", "gamma")] {
            registry
                .register(TestCard::new(id, name), label(name), RegisterPolicy::Fail)
                .unwrap();
        }

        let names: Vec<String> = registry.cards().into_iter().map(|card| card.name).collect();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
        assert_eq!(registry.ids(), vec!["a", "b", "c"]);
        assert!(registry.contains("b"));
        assert!(!registry.contains("z"));
        assert_eq!(registry.len(), 3);
        assert!(!registry.is_empty());
    }

    #[test]
    fn removal_returns_the_card_and_keeps_the_remaining_order() {
        let registry = registry();
        for (id, name) in [("a", "alpha"), ("b", "beta"), ("c", "gamma")] {
            registry
                .register(TestCard::new(id, name), label(name), RegisterPolicy::Fail)
                .unwrap();
        }

        let removed = registry.remove("b").unwrap();
        assert_eq!(removed, TestCard::new("b", "beta"));
        assert!(registry.get("b").is_none());
        assert!(
            registry.get("c").is_some(),
            "the shifted index still resolves"
        );
        assert_eq!(registry.ids(), vec!["a", "c"]);
        assert_eq!(registry.revision(), 4);

        assert!(registry.remove("b").is_none());
        assert_eq!(
            registry.revision(),
            4,
            "removing a missing id is not a mutation"
        );
    }

    #[test]
    fn clear_reports_the_count_and_counts_as_one_mutation() {
        let registry = registry();
        registry
            .register(
                TestCard::new("a", "alpha"),
                label("a"),
                RegisterPolicy::Fail,
            )
            .unwrap();
        registry
            .register(TestCard::new("b", "beta"), label("b"), RegisterPolicy::Fail)
            .unwrap();
        assert_eq!(registry.revision(), 2);

        assert_eq!(registry.clear(), 2);
        assert!(registry.is_empty());
        assert_eq!(registry.revision(), 3);

        assert_eq!(registry.clear(), 0);
        assert_eq!(
            registry.revision(),
            3,
            "clearing an empty registry is a no-op"
        );
    }

    #[test]
    fn reset_clears_and_zeroes_the_revision() {
        let registry = registry();
        registry
            .register(
                TestCard::new("a", "alpha"),
                label("a"),
                RegisterPolicy::Fail,
            )
            .unwrap();
        assert_eq!(registry.reset(), 1);
        assert!(registry.is_empty());
        assert_eq!(registry.revision(), 0);
    }

    #[test]
    fn empty_and_whitespace_ids_are_rejected() {
        let registry = registry();
        for id in ["", "   ", "\t\n"] {
            let error = registry
                .register(
                    TestCard::new(id, "ghost"),
                    label("never"),
                    RegisterPolicy::Fail,
                )
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: widget id cannot be empty or whitespace only"
            );
        }
        assert!(registry.is_empty());
        assert_eq!(registry.revision(), 0);
    }

    #[test]
    fn debug_output_lists_ids_and_the_revision() {
        let registry = registry();
        registry
            .register(
                TestCard::new("a", "alpha"),
                label("a"),
                RegisterPolicy::Fail,
            )
            .unwrap();
        let rendered = format!("{registry:?}");
        assert!(rendered.contains("kind: \"widget\""), "{rendered}");
        assert!(rendered.contains("\"a\""), "{rendered}");
        assert!(rendered.contains("revision: 1"), "{rendered}");
    }
}
