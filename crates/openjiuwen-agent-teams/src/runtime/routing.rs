//! Member-time model selection: the pool, the allocator and the effective member spec.
//!
//! [`TeamModelRouter`] hands one pool entry to each member. It composes
//! [`ModelPool`](crate::models::ModelPool) (validated, ordered endpoints) with the
//! [`ModelAllocator`], which hands entries out in declaration order for `round_robin` and by name
//! for `by_model_name`; a team without a pool keeps the allocator's by-name echo.
//! Per-request routing ([`crate::models::ModelRouter`]) stays with the provider layer, because a
//! member's endpoint is decided once per member, not once per request.
//!
//! The invariant is **one allocation per member**, reused by the roster row, the harness spec and the
//! KV-cache identity: [`TeamModelRouter::select_for`] memoizes the allocator's decision under the
//! member name, so the leader's `MemberRecord::model_ref` and the model its harness is built for can
//! never disagree, and spawning a member twice never rotates it onto a different endpoint.
//!
//! The selection is not just recorded — [`TeamModelRouter::effective_spec_for`] rewrites the member
//! [`DeepAgentSpec`](crate::schema::DeepAgentSpec)'s `model` field with the chosen entry's name and
//! provider, so the harness factory really builds a client for that endpoint, while the roster row
//! records the same choice as `MemberRecord::model_ref`.
use std::collections::BTreeMap;

use openjiuwen_common::error::Result;
use parking_lot::RwLock;
use tracing::debug;

use crate::models::ModelPool;
use crate::schema::{
    DeepAgentSpec, ModelPoolEntry, ModelPoolStrategy, ModelRef, TeamAgentSpec, TeamRole,
};

use super::allocator::{ModelAllocator, TeamModelAllocation};

/// One member's model choice: the persisted reference plus the pool entry behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSelection {
    /// `(model_name, index)` recorded on the member and stamped on the request.
    pub allocation: TeamModelAllocation,
    /// The pool entry the router picked, when the team declares a pool.
    pub entry: Option<ModelPoolEntry>,
}
impl ModelSelection {
    /// Returns the effective model name: the entry when pooled, else the allocation's name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.entry
            .as_ref()
            .map_or(self.allocation.model_name.as_str(), |entry| {
                entry.model_name.as_str()
            })
    }
}

/// Model-pool routing for one running team.
///
/// Built once at team build from [`TeamAgentSpec::model_pool`] and
/// [`TeamAgentSpec::model_pool_strategy`]; shared by the build path (leader + roster rows) and the
/// spawn path (teammates materialised later). The allocator is the only cursor:
/// [`select_for`](TeamModelRouter::select_for) allocates once per member and memoizes the result in
/// the per-member `selections` map, so every later lookup for that member (roster row, harness spec,
/// KV identity) returns the same endpoint.
pub struct TeamModelRouter {
    /// Hands the next pool entry out; the single source of the round-robin rotation.
    allocator: ModelAllocator,
    /// The validated endpoints the allocator indexes into.
    pool: ModelPool,
    /// Per-member memo of the allocation, so one member is allocated exactly once.
    selections: RwLock<BTreeMap<String, ModelSelection>>,
}
impl std::fmt::Debug for TeamModelRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamModelRouter")
            .field("strategy", &self.allocator.strategy())
            .field("pool_len", &self.pool.len())
            .field("allocated", &self.selections.read().len())
            .finish_non_exhaustive()
    }
}
impl TeamModelRouter {
    /// Builds the router for `entries` under `strategy`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) when the pool is invalid (blank or duplicate model name).
    pub fn new(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Result<Self> {
        let pool = ModelPool::from_entries(entries)?;
        Ok(Self {
            allocator: ModelAllocator::new(entries, strategy),
            pool,
            selections: RwLock::new(BTreeMap::new()),
        })
    }
    /// Returns a router over an empty pool: the no-pool compatibility path.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            allocator: ModelAllocator::new(&[], ModelPoolStrategy::RoundRobin),
            pool: ModelPool::empty(),
            selections: RwLock::new(BTreeMap::new()),
        }
    }
    /// Builds the router from a team spec's declared pool.
    ///
    /// # Errors
    ///
    /// Same as [`TeamModelRouter::new`].
    pub fn for_spec(spec: &TeamAgentSpec) -> Result<Self> {
        Self::new(&spec.model_pool, spec.model_pool_strategy)
    }
    /// Returns the underlying allocator (the no-pool compatibility path).
    #[must_use]
    pub fn allocator(&self) -> &ModelAllocator {
        &self.allocator
    }
    /// Returns the model pool.
    #[must_use]
    pub fn pool(&self) -> &ModelPool {
        &self.pool
    }
    /// Returns `true` when the team declares no model pool.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
    /// Returns the declared strategy.
    #[must_use]
    pub fn strategy(&self) -> ModelPoolStrategy {
        self.allocator.strategy()
    }
    /// Records one member's model reference, falling back to the allocator when unrouted.
    ///
    /// Prefer [`TeamModelRouter::model_ref_for`]: this non-memoized form allocates a fresh slot on
    /// every call, so it must not be used twice for the same member.
    #[must_use]
    pub fn model_ref(&self, requested: Option<&str>, role: TeamRole) -> Option<String> {
        match self.select(requested, role) {
            Ok(selection) => Some(selection.allocation.model_name).filter(|name| !name.is_empty()),
            Err(error) => {
                debug!(%error, "no model entry could be selected for member");
                None
            }
        }
    }
    /// Records `member_name`'s model reference, allocating it at most once.
    ///
    /// Mirrors [`TeamModelRouter::model_ref`] over the memoized
    /// [`TeamModelRouter::select_for`].
    #[must_use]
    pub fn model_ref_for(
        &self,
        member_name: &str,
        requested: Option<&str>,
        role: TeamRole,
    ) -> Option<String> {
        match self.select_for(member_name, requested, role) {
            Ok(selection) => Some(selection.allocation.model_name).filter(|name| !name.is_empty()),
            Err(error) => {
                debug!(%error, member = member_name, "no model entry could be selected for member");
                None
            }
        }
    }
    /// Selects the model for one member.
    ///
    /// With a pool this is exactly [`ModelAllocator::allocate`] followed by the pool lookup of the
    /// allocated index, so `round_robin` really rotates the cursor on every call and `by_model_name`
    /// serves the requested entry (falling back to the first). Without a pool this is the
    /// allocator's by-name echo, or [`Error::config`](openjiuwen_common::error::Error::config) when
    /// nothing is requested.
    ///
    /// `role` is accepted for API parity only: the Python allocator never varies an allocation by
    /// role, so this port does not either.
    ///
    /// # Errors
    ///
    /// Propagates the allocator's configuration error.
    pub fn select(&self, requested: Option<&str>, _role: TeamRole) -> Result<ModelSelection> {
        let allocation = self.allocator.allocate(requested)?;
        if self.pool.is_empty() {
            return Ok(ModelSelection {
                allocation,
                entry: None,
            });
        }
        let entry = self.pool.get(allocation.index).cloned();
        Ok(ModelSelection { allocation, entry })
    }
    /// Returns `member_name`'s memoized selection, allocating it on first use.
    ///
    /// This is the member-time entry point: the first caller (the roster row, the harness spec or
    /// the KV identity) allocates, and every later caller for that member reuses the same endpoint.
    /// A team without a pool and without a requested name resolves to the "nothing routed"
    /// selection rather than an error, matching the pre-router behaviour.
    ///
    /// # Errors
    ///
    /// Same as [`TeamModelRouter::select`] on the first allocation for the member.
    pub fn select_for(
        &self,
        member_name: &str,
        requested: Option<&str>,
        role: TeamRole,
    ) -> Result<ModelSelection> {
        {
            let cached = self.selections.read().get(member_name).cloned();
            if let Some(selection) = cached {
                return Ok(selection);
            }
        }
        let selection = if self.pool.is_empty() && requested.is_none() {
            Self::unrouted_selection()
        } else {
            self.select(requested, role)?
        };
        self.selections
            .write()
            .insert(member_name.to_owned(), selection.clone());
        Ok(selection)
    }
    /// Returns the "nothing routed" selection used when a team declares no pool at all.
    fn unrouted_selection() -> ModelSelection {
        ModelSelection {
            allocation: TeamModelAllocation {
                model_name: String::new(),
                index: 0,
            },
            entry: None,
        }
    }
    /// Rewrites a member spec's model identity and returns the persisted allocation.
    ///
    /// When a pool entry was chosen, `spec.model` names that entry (model name + provider) and
    /// keeps the spec's own `params`; otherwise the spec is returned unchanged apart from the
    /// recorded allocation. The pool therefore selects a real endpoint through the same field the
    /// native factory reads.
    ///
    /// Prefer [`TeamModelRouter::effective_spec_for`]: this non-memoized form allocates a fresh slot
    /// on every call, so it must not be used twice for the same member.
    ///
    /// # Errors
    ///
    /// Same as [`TeamModelRouter::select`].
    pub fn effective_spec(
        &self,
        spec: &DeepAgentSpec,
        requested: Option<&str>,
        role: TeamRole,
    ) -> Result<(DeepAgentSpec, ModelSelection)> {
        if self.pool.is_empty() && requested.is_none() {
            // No pool and no pinned name: the member keeps whatever the factory resolves
            // (environment / resolver default), exactly as before the router existed.
            return Ok((spec.clone(), Self::unrouted_selection()));
        }
        let selection = self.select(requested, role)?;
        let effective = Self::apply_selection(spec, &selection);
        Ok((effective, selection))
    }
    /// Rewrites `spec` for `member_name`, allocating that member at most once.
    ///
    /// The harness spec and the roster reference are therefore derived from one allocation.
    ///
    /// # Errors
    ///
    /// Same as [`TeamModelRouter::select_for`].
    pub fn effective_spec_for(
        &self,
        member_name: &str,
        spec: &DeepAgentSpec,
        requested: Option<&str>,
        role: TeamRole,
    ) -> Result<(DeepAgentSpec, ModelSelection)> {
        let selection = self.select_for(member_name, requested, role)?;
        let effective = Self::apply_selection(spec, &selection);
        Ok((effective, selection))
    }

    /// Returns `spec` with its model identity replaced by the selected pool entry.
    ///
    /// An unrouted selection (`entry == None`) leaves the spec untouched; a pooled one names the
    /// entry and keeps the spec's own request `params`.
    fn apply_selection(spec: &DeepAgentSpec, selection: &ModelSelection) -> DeepAgentSpec {
        let Some(entry) = selection.entry.as_ref() else {
            return spec.clone();
        };
        let mut effective = spec.clone();
        let params = effective
            .model
            .as_ref()
            .map(|model| model.params.clone())
            .unwrap_or_default();
        effective.model = Some(ModelRef {
            name: None,
            model_name: Some(entry.model_name.clone()),
            provider: Some(entry.api_provider.clone()).filter(|provider| !provider.is_empty()),
            params,
        });
        effective
    }
}
