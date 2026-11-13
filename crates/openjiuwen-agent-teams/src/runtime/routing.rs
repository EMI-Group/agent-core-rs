//! Pool + router selection of the model a member runs on.
//!
//! [`TeamModelRouter`] composes the two model-layer primitives the crate already ships: a
//! [`ModelPool`] (validated, ordered endpoints) and a [`ModelRouter`] (strategy-driven choice per
//! request). It also keeps the [`ModelAllocator`] so a team without a pool behaves exactly as
//! before: by-name echo, round-robin, and the persisted `(model_name, index)` reference.
//!
//! The selection is not just recorded — [`TeamModelRouter::effective_spec`] rewrites the member
//! [`DeepAgentSpec`]'s `model` field with the chosen pool entry's name and provider, so the
//! harness factory actually builds a client for that endpoint. That closes the gap where a pool
//! entry only reached `MemberRecord::model_ref`.
use openjiuwen_common::error::Result;
use tracing::debug;

use crate::models::{ModelPool, ModelRouter, RequestHint, RouterDecision, RouterStrategy};
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
/// spawn path (teammates materialised later). [`ModelRouter`] keeps its own cursor, so the router
/// and the allocator advance in lockstep for round-robin strategies.
pub struct TeamModelRouter {
    allocator: ModelAllocator,
    pool: ModelPool,
    router: std::sync::Arc<ModelRouter>,
}
impl std::fmt::Debug for TeamModelRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeamModelRouter")
            .field("strategy", &self.allocator.strategy())
            .field("pool_len", &self.pool.len())
            .finish_non_exhaustive()
    }
}
impl TeamModelRouter {
    /// Builds the router for `entries` under `strategy`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pool is invalid (blank or duplicate model name).
    pub fn new(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Result<Self> {
        let pool = ModelPool::from_entries(entries)?;
        let router = ModelRouter::for_pool(entries, strategy)?;
        Ok(Self {
            allocator: ModelAllocator::new(entries, strategy),
            pool,
            router: std::sync::Arc::new(router),
        })
    }
    /// Returns a router over an empty pool: the no-pool compatibility path.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            allocator: ModelAllocator::new(&[], ModelPoolStrategy::RoundRobin),
            pool: ModelPool::empty(),
            router: std::sync::Arc::new(ModelRouter::new(
                ModelPool::empty(),
                RouterStrategy::RoundRobin,
            )),
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
    /// Selects the model for one member.
    ///
    /// With a pool, the router decides first and the result is mirrored into a
    /// [`TeamModelAllocation`] (so the persisted reference and the pool index agree). Without a
    /// pool this is exactly [`ModelAllocator::allocate`]: by-name echo or
    /// [`Error::config`] when nothing is requested.
    ///
    /// # Errors
    ///
    /// Propagates the allocator's configuration error and a router failure that has no fallback.
    pub fn select(&self, requested: Option<&str>, role: TeamRole) -> Result<ModelSelection> {
        if self.pool.is_empty() {
            let allocation = self.allocator.allocate(requested)?;
            return Ok(ModelSelection {
                allocation,
                entry: None,
            });
        }
        let hint = requested.map_or_else(
            || RequestHint::for_role(role),
            |name| RequestHint::for_role(role).with_model_name(name),
        );
        match self.router.decide(&hint) {
            Ok(decision) => Ok(Self::selection_from(decision)),
            Err(error) => {
                debug!(%error, "model router refused; falling back to the allocator");
                let allocation = self.allocator.allocate(requested)?;
                Ok(ModelSelection {
                    allocation,
                    entry: None,
                })
            }
        }
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
        let Some(entry) = selection.entry.as_ref() else {
            return Ok((spec.clone(), selection));
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
        Ok((effective, selection))
    }

    /// Mirrors a router decision into the persisted allocation shape.
    fn selection_from(decision: RouterDecision) -> ModelSelection {
        ModelSelection {
            allocation: TeamModelAllocation {
                model_name: decision.entry.model_name.clone(),
                index: decision.index,
            },
            entry: Some(decision.entry),
        }
    }
}
