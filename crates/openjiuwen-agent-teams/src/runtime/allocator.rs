//! Model-pool allocation for team members.
//!
//! Python parity source: `openjiuwen/agent_teams/models/allocator.py`
//! (`RoundRobinModelAllocator`, `ByModelNameAllocator`, `Allocation`).
//!
//! A team may declare a pool of interchangeable LLM endpoints
//! ([`TeamAgentSpec::model_pool`](crate::schema::TeamAgentSpec::model_pool)). The allocator hands
//! one endpoint to each member so concurrent members spread over the pool instead of saturating a
//! single endpoint. The strategy is selected by
//! [`ModelPoolStrategy`](crate::schema::ModelPoolStrategy):
//!
//! * [`ModelPoolStrategy::RoundRobin`](crate::schema::ModelPoolStrategy::RoundRobin) rotates through
//!   every entry in pool order, ignoring the requested model name. Always allocates.
//! * [`ModelPoolStrategy::ByModelName`](crate::schema::ModelPoolStrategy::ByModelName) picks the
//!   entry whose `model_name` the member asked for,
//!   falling back to the first entry when the pool does not serve that name.
//! * [`ModelPoolStrategy::IntelliRouter`](crate::schema::ModelPoolStrategy::IntelliRouter) is
//!   modelled on the wire but never allocates: that strategy needs the `IntelliRouter` client-side
//!   reliable router and its route-group service, which this port does not implement, so every
//!   allocation is a typed [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported)
//!   refusal.
//!
//! Python builds no allocator when the pool is empty and lets every member resolve its per-agent
//! model. Here the allocator is always constructed and an empty pool reports the caller's requested
//! name verbatim, or a [`Error::Config`](openjiuwen_common::error::Error::Config) when there is
//! neither a pool nor a name to fall back to.
//!
//! ```
//! use openjiuwen_agent_teams::runtime::allocator::ModelAllocator;
//! use openjiuwen_agent_teams::schema::{ModelPoolEntry, ModelPoolStrategy};
//!
//! let pool = vec![
//!     ModelPoolEntry::new("gpt-4o", "openai", "https://api.openai.com/v1", "sk-a"),
//!     ModelPoolEntry::new("claude", "anthropic", "https://api.anthropic.com", "sk-b"),
//! ];
//! let allocator = ModelAllocator::new(&pool, ModelPoolStrategy::ByModelName);
//! let picked = allocator.allocate(Some("claude")).expect("claude is pooled");
//! assert_eq!((picked.model_name.as_str(), picked.index), ("claude", 1));
//! ```
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::schema::{ModelPoolEntry, ModelPoolStrategy};

/// Error a member gets when neither a pool entry nor a model name is available.
const NO_MODEL_MESSAGE: &str =
    "no model configured for member; set model_pool or a member model name";

/// Refusal for the `intelli_router` strategy, whose router client this port does not carry.
const INTELLI_ROUTER_MESSAGE: &str = "the intelli_router model-pool strategy requires the IntelliRouter client-side reliable router and its route-group service, which this port does not implement";

/// One allocation result: the model name a member runs on and its position in the pool.
///
/// Python's `Allocation` carries the pool entry itself plus a group index used to persist a DB
/// reference. The port keeps the two values that survive a restart — `(model_name, index)` — and
/// rehydrates the live endpoint (credentials, URL, request knobs) from the in-session pool, so a
/// pool update reaches every member on its next resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TeamModelAllocation {
    /// Model name the member should run.
    pub model_name: String,
    /// Position of the allocated entry in the pool, in pool order.
    pub index: usize,
}

impl TeamModelAllocation {
    /// Builds an allocation record.
    #[must_use]
    pub fn new(model_name: impl Into<String>, index: usize) -> Self {
        Self {
            model_name: model_name.into(),
            index,
        }
    }
}

/// Assigns pool entries to members (Python `ModelAllocator` protocol).
///
/// The allocator is immutable data plus a cursor: clones are independent (a clone resumes from the
/// cursor value it copied), and every allocation stores the cursor before returning so two
/// sequential allocations over the same instance advance.
///
/// `ModelPoolEntry::api_key` is a `SecretString`, so cloning a pool entry copies the credential
/// without it ever appearing in `Debug` output.
pub struct ModelAllocator {
    /// The pool this allocator distributes, in declaration order.
    entries: Vec<ModelPoolEntry>,
    /// How an entry is picked from [`ModelAllocator::entries`].
    strategy: ModelPoolStrategy,
    /// Round-robin cursor: the index of the entry the next allocation hands out.
    cursor: Mutex<usize>,
}

impl ModelAllocator {
    /// Builds an allocator over `entries` using `strategy`.
    ///
    /// The pool is copied so later edits to the caller's vector do not disturb a live team; the
    /// round-robin cursor starts at the first entry.
    #[must_use]
    pub fn new(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Self {
        Self {
            entries: entries.to_vec(),
            strategy,
            cursor: Mutex::new(0),
        }
    }

    /// Returns `true` when the pool has no entries, in which case no endpoint can be handed out.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of pooled endpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the strategy this allocator applies.
    #[must_use]
    pub fn strategy(&self) -> ModelPoolStrategy {
        self.strategy
    }

    /// Returns the pool this allocator distributes.
    #[must_use]
    pub fn entries(&self) -> &[ModelPoolEntry] {
        &self.entries
    }

    /// Allocates the endpoint a member should run on.
    ///
    /// `model_name` is the member's requested model (`LeaderSpec.model_name`,
    /// `TeamMemberSpec.model_name`, or the `spawn_member` argument). It is a hint: round-robin
    /// ignores it, by-name uses it as the lookup key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the strategy is
    /// [`ModelPoolStrategy::IntelliRouter`]: that strategy needs the `IntelliRouter` client-side
    /// reliable router and its route-group service, which this port does not implement, so the
    /// refusal is typed and fires before the empty-pool echo and before any round-robin/by-name
    /// dispatch.
    ///
    /// Returns [`Error::Config`] when the pool is empty and
    /// no model name was requested — there is nothing to allocate and nothing to echo back.
    pub fn allocate(&self, model_name: Option<&str>) -> Result<TeamModelAllocation> {
        match self.strategy {
            // Refused first, so the unported strategy can never fall through to the empty-pool echo
            // or to a round-robin/by-name dispatch.
            ModelPoolStrategy::IntelliRouter => Err(Error::unsupported(INTELLI_ROUTER_MESSAGE)),
            ModelPoolStrategy::RoundRobin | ModelPoolStrategy::ByModelName
                if self.entries.is_empty() =>
            {
                match model_name {
                    // Without a pool the member keeps the model its own spec declares.
                    Some(name) => Ok(TeamModelAllocation::new(name, 0)),
                    None => Err(Error::config(NO_MODEL_MESSAGE)),
                }
            }
            ModelPoolStrategy::RoundRobin => Ok(self.allocate_round_robin()),
            ModelPoolStrategy::ByModelName => Ok(self.allocate_by_name(model_name)),
        }
    }

    /// Snapshot of the allocator cursor for session persistence.
    ///
    /// The shape is
    /// `{"strategy": "round_robin"|"by_model_name"|"intelli_router", "next_index": n}`, where
    /// `next_index` is the pool position the next allocation hands out.
    #[must_use]
    pub fn state_dict(&self) -> Value {
        json!({
            "strategy": strategy_wire(self.strategy),
            "next_index": self.next_index(),
        })
    }

    /// Restores an allocator cursor from [`ModelAllocator::state_dict`] output.
    ///
    /// The pool itself is never persisted — it is rebuilt from the spec — so the restored allocator
    /// is empty and carries only the strategy and cursor. A known strategy wire value (including
    /// `"intelli_router"`) restores faithfully, an unknown string restores as
    /// [`ModelPoolStrategy::RoundRobin`] with one warning — so a state dict written by a newer
    /// runtime still restores — and an absent strategy restores as
    /// [`ModelPoolStrategy::RoundRobin`] silently, which is the Python default rather than an
    /// anomaly. A missing or malformed `next_index` restores as `0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `value` is
    /// not a JSON object.
    pub fn from_state_dict(value: &Value) -> Result<Self> {
        let Some(object) = value.as_object() else {
            return Err(Error::validation(format!(
                "model allocator state must be a JSON object, got {}",
                json_type_name(value)
            )));
        };
        let strategy = strategy_from_state(object.get("strategy"));
        let next_index = if let Some(index) = object.get("next_index").and_then(Value::as_u64) {
            usize::try_from(index).unwrap_or(0)
        } else {
            if object.contains_key("next_index") {
                tracing::warn!(
                    state = %value,
                    "model allocator state has a non-numeric next_index; restoring the cursor at 0"
                );
            }
            0
        };
        Ok(Self {
            entries: Vec::new(),
            strategy,
            cursor: Mutex::new(next_index),
        })
    }

    /// Returns the pool position the next allocation will hand out.
    fn next_index(&self) -> usize {
        *self.cursor.lock()
    }

    /// Advances the cursor and returns the entry at the current position.
    ///
    /// The modulo is guarded rather than assumed: `allocate` rejects the empty pool first, but a
    /// cursor operation must never divide by zero even if that check moves.
    fn allocate_round_robin(&self) -> TeamModelAllocation {
        let mut cursor = self.cursor.lock();
        let len = self.entries.len();
        let index = cursor.checked_rem(len).unwrap_or(0);
        *cursor = index.wrapping_add(1).checked_rem(len).unwrap_or(0);
        self.allocation_at(index)
    }

    /// Returns the pool entry matching `model_name`, else the first entry.
    ///
    /// A name the pool does not serve is not an error: the allocation reports the requested name
    /// while running the first endpoint, so the member keeps its declared model in the record.
    fn allocate_by_name(&self, model_name: Option<&str>) -> TeamModelAllocation {
        match model_name {
            None => self.allocation_at(0),
            Some(name) => match self
                .entries
                .iter()
                .position(|entry| entry.model_name == name)
            {
                Some(index) => TeamModelAllocation::new(name, index),
                None => TeamModelAllocation::new(name, 0),
            },
        }
    }

    /// Builds the allocation for pool position `index`.
    ///
    /// Every position comes from the pool itself (index `0` or a modulo-wrapped cursor), so the
    /// lookup succeeds; the fallback keeps the function total instead of panicking on an impossible
    /// position.
    fn allocation_at(&self, index: usize) -> TeamModelAllocation {
        match self.entries.get(index) {
            Some(entry) => TeamModelAllocation::new(entry.model_name.clone(), index),
            None => TeamModelAllocation::new(String::new(), index),
        }
    }
}

impl Clone for ModelAllocator {
    /// Copies the pool, the strategy and the current cursor.
    ///
    /// A clone resumes from the cursor value it copied; it never shares the rotation state with the
    /// original, so two clones of one allocator hand out the same sequence independently.
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            strategy: self.strategy,
            cursor: Mutex::new(self.next_index()),
        }
    }
}

impl fmt::Debug for ModelAllocator {
    /// Prints the strategy, pool size and cursor; pool entries stay out of the output so a
    /// credential can never reach a log line through `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelAllocator")
            .field("strategy", &strategy_wire(self.strategy))
            .field("entries", &self.entries.len())
            .field("cursor", &self.next_index())
            .finish()
    }
}

impl Default for ModelAllocator {
    /// An empty round-robin allocator, the state of a team that declares no pool.
    fn default() -> Self {
        Self::new(&[], ModelPoolStrategy::default())
    }
}

/// Returns the wire value of `strategy` (`"round_robin"` / `"by_model_name"` / `"intelli_router"`).
const fn strategy_wire(strategy: ModelPoolStrategy) -> &'static str {
    match strategy {
        ModelPoolStrategy::RoundRobin => "round_robin",
        ModelPoolStrategy::ByModelName => "by_model_name",
        ModelPoolStrategy::IntelliRouter => "intelli_router",
    }
}

/// Parses a persisted strategy value.
///
/// Three-way rule: a known wire value restores faithfully, a present but unknown string restores
/// [`ModelPoolStrategy::RoundRobin`] with one warning (so a state dict written by a newer runtime
/// still restores), and an absent or null strategy restores [`ModelPoolStrategy::RoundRobin`]
/// silently because that is the Python default, not an anomaly.
fn strategy_from_state(value: Option<&Value>) -> ModelPoolStrategy {
    match value {
        None | Some(Value::Null) => ModelPoolStrategy::RoundRobin,
        Some(Value::String(raw)) => match raw.as_str() {
            "round_robin" => ModelPoolStrategy::RoundRobin,
            "by_model_name" => ModelPoolStrategy::ByModelName,
            "intelli_router" => ModelPoolStrategy::IntelliRouter,
            unknown => {
                tracing::warn!(
                    strategy = unknown,
                    "unknown model-pool strategy in allocator state; restoring round_robin"
                );
                ModelPoolStrategy::RoundRobin
            }
        },
        Some(other) => {
            tracing::warn!(
                strategy = %other,
                "non-string model-pool strategy in allocator state; restoring round_robin"
            );
            ModelPoolStrategy::RoundRobin
        }
    }
}

/// Names the JSON type of `value` for error messages.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Builds `n` distinct pool entries named `m0`, `m1`, …
    fn pool(n: usize) -> Vec<ModelPoolEntry> {
        (0..n)
            .map(|i| {
                ModelPoolEntry::new(
                    format!("m{i}"),
                    "openai",
                    format!("https://endpoint-{i}.example/v1"),
                    format!("sk-{i}"),
                )
            })
            .collect()
    }

    #[test]
    fn round_robin_cycles_and_persists_the_cursor() {
        let entries = pool(2);
        let allocator = ModelAllocator::new(&entries, ModelPoolStrategy::RoundRobin);
        assert_eq!(allocator.len(), 2);
        assert!(!allocator.is_empty());
        assert_eq!(allocator.strategy(), ModelPoolStrategy::RoundRobin);
        assert_eq!(allocator.entries(), entries.as_slice());
        assert_eq!(allocator.state_dict()["next_index"], json!(0));

        let first = allocator.allocate(None).expect("non-empty pool");
        assert_eq!(first, TeamModelAllocation::new("m0", 0));
        assert_eq!(allocator.state_dict()["next_index"], json!(1));

        let second = allocator.allocate(None).expect("non-empty pool");
        assert_eq!(second, TeamModelAllocation::new("m1", 1));
        assert_eq!(allocator.state_dict()["next_index"], json!(0));

        // Wraps back to the first endpoint.
        let third = allocator.allocate(None).expect("non-empty pool");
        assert_eq!(third, TeamModelAllocation::new("m0", 0));
        assert_eq!(allocator.state_dict()["next_index"], json!(1));

        // The requested name is ignored: round-robin is name-agnostic.
        let hinted = allocator.allocate(Some("m1")).expect("non-empty pool");
        assert_eq!(hinted, TeamModelAllocation::new("m1", 1));
    }

    #[test]
    fn round_robin_clone_resumes_from_the_copied_cursor() {
        let entries = pool(3);
        let allocator = ModelAllocator::new(&entries, ModelPoolStrategy::RoundRobin);
        let _ = allocator.allocate(None).unwrap();
        let clone = allocator.clone();
        assert_eq!(
            clone.allocate(None).unwrap(),
            TeamModelAllocation::new("m1", 1)
        );
        // The clone advanced independently of the original.
        assert_eq!(allocator.state_dict()["next_index"], json!(1));
        assert_eq!(clone.state_dict()["next_index"], json!(2));
    }

    #[test]
    fn by_name_hits_misses_and_defaults() {
        let entries = pool(3);
        let allocator = ModelAllocator::new(&entries, ModelPoolStrategy::ByModelName);

        let hit = allocator.allocate(Some("m2")).expect("pooled name");
        assert_eq!(hit, TeamModelAllocation::new("m2", 2));

        let miss = allocator.allocate(Some("gemini")).expect("miss falls back");
        assert_eq!(miss, TeamModelAllocation::new("gemini", 0));

        let defaulted = allocator.allocate(None).expect("no hint");
        assert_eq!(defaulted, TeamModelAllocation::new("m0", 0));

        // By-name never advances the rotation cursor.
        assert_eq!(allocator.state_dict()["next_index"], json!(0));
    }

    #[test]
    fn by_name_reports_the_first_entry_of_a_repeated_name() {
        let mut entries = pool(1);
        entries.push(ModelPoolEntry::new(
            "m0",
            "openai",
            "https://other.example/v1",
            "sk-dup",
        ));
        let allocator = ModelAllocator::new(&entries, ModelPoolStrategy::ByModelName);
        assert_eq!(
            allocator.allocate(Some("m0")).unwrap(),
            TeamModelAllocation::new("m0", 0)
        );
    }

    #[test]
    fn empty_pool_echoes_the_requested_name() {
        for strategy in [
            ModelPoolStrategy::RoundRobin,
            ModelPoolStrategy::ByModelName,
        ] {
            let allocator = ModelAllocator::new(&[], strategy);
            assert!(allocator.is_empty());
            assert_eq!(allocator.len(), 0);
            let allocation = allocator.allocate(Some("gpt-4o")).expect("echoed name");
            assert_eq!(allocation, TeamModelAllocation::new("gpt-4o", 0));
            // The echo never moves the cursor.
            assert_eq!(allocator.state_dict()["next_index"], json!(0));
        }
    }

    #[test]
    fn empty_pool_without_a_name_is_a_config_error() {
        let allocator = ModelAllocator::default();
        assert!(allocator.is_empty());
        let error = allocator.allocate(None).expect_err("no model anywhere");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), format!("config: {NO_MODEL_MESSAGE}"));

        // An empty echoed name is still an echo, not an error.
        assert_eq!(
            allocator.allocate(Some("")).unwrap(),
            TeamModelAllocation::new("", 0)
        );
    }

    #[test]
    fn state_dict_shape_matches_python() {
        let allocator = ModelAllocator::new(&pool(2), ModelPoolStrategy::RoundRobin);
        assert_eq!(
            allocator.state_dict(),
            json!({"strategy": "round_robin", "next_index": 0})
        );
        let by_name = ModelAllocator::new(&pool(1), ModelPoolStrategy::ByModelName);
        assert_eq!(
            by_name.state_dict(),
            json!({"strategy": "by_model_name", "next_index": 0})
        );
    }

    #[test]
    fn state_dict_round_trips_through_from_state_dict() {
        let allocator = ModelAllocator::new(&pool(3), ModelPoolStrategy::RoundRobin);
        let _ = allocator.allocate(None).unwrap();
        let _ = allocator.allocate(None).unwrap();
        let state = allocator.state_dict();
        assert_eq!(state["next_index"], json!(2));

        let restored = ModelAllocator::from_state_dict(&state).expect("object state");
        assert!(restored.is_empty(), "the pool is rebuilt from the spec");
        assert_eq!(restored.strategy(), ModelPoolStrategy::RoundRobin);
        assert_eq!(restored.state_dict(), state);

        // A restored allocator keeps allocating from the restored cursor once it owns a pool.
        let rebuilt = ModelAllocator {
            entries: pool(3),
            ..restored
        };
        assert_eq!(
            rebuilt.allocate(None).unwrap(),
            TeamModelAllocation::new("m2", 2)
        );

        let by_name = ModelAllocator::new(&pool(1), ModelPoolStrategy::ByModelName);
        let restored = ModelAllocator::from_state_dict(&by_name.state_dict()).unwrap();
        assert_eq!(restored.strategy(), ModelPoolStrategy::ByModelName);
    }

    #[test]
    fn from_state_dict_defaults_missing_fields() {
        let restored = ModelAllocator::from_state_dict(&json!({})).expect("object");
        assert_eq!(restored.strategy(), ModelPoolStrategy::RoundRobin);
        assert_eq!(restored.state_dict()["next_index"], json!(0));

        let restored = ModelAllocator::from_state_dict(&json!({"next_index": "7"})).unwrap();
        assert_eq!(restored.state_dict()["next_index"], json!(0));

        let restored = ModelAllocator::from_state_dict(&json!({"next_index": 4})).unwrap();
        assert_eq!(
            restored.state_dict(),
            json!({"strategy": "round_robin", "next_index": 4})
        );

        // A known `intelli_router` value is restored faithfully rather than degraded to round-robin,
        // so the deferred strategy stays a typed refusal instead of a silent fallback.
        let restored =
            ModelAllocator::from_state_dict(&json!({"strategy": "intelli_router"})).unwrap();
        assert_eq!(restored.strategy(), ModelPoolStrategy::IntelliRouter);
        let error = restored
            .allocate(None)
            .expect_err("intelli_router is unported");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("intelli_router"), "{error}");
    }

    #[test]
    fn a_populated_intelli_router_allocator_refuses_every_allocation() {
        let allocator = ModelAllocator::new(&pool(3), ModelPoolStrategy::IntelliRouter);
        for requested in [None, Some("m1")] {
            let error = allocator
                .allocate(requested)
                .expect_err("intelli_router is unported");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error.to_string().contains("intelli_router"), "{error}");
        }
        // The refusal never advances the cursor or echoes the requested name.
        assert_eq!(allocator.state_dict()["next_index"], json!(0));
    }

    #[test]
    fn intelli_router_state_dict_round_trips() {
        let mut allocator = ModelAllocator::new(&pool(3), ModelPoolStrategy::IntelliRouter);
        // An IntelliRouter allocator never allocates, so pin a cursor to prove the value round-trips.
        allocator.cursor = Mutex::new(2);
        let state = allocator.state_dict();
        assert_eq!(
            state,
            json!({"strategy": "intelli_router", "next_index": 2})
        );

        let restored = ModelAllocator::from_state_dict(&state).expect("object state");
        assert!(restored.is_empty(), "the pool is rebuilt from the spec");
        assert_eq!(restored.strategy(), ModelPoolStrategy::IntelliRouter);
        assert_eq!(restored.state_dict(), state);
    }

    #[test]
    fn from_state_dict_tolerates_unknown_and_absent_strategies() {
        let restored = ModelAllocator::from_state_dict(&json!({"strategy": "bogus"})).unwrap();
        assert_eq!(restored.strategy(), ModelPoolStrategy::RoundRobin);

        let restored =
            ModelAllocator::from_state_dict(&json!({"strategy": "by_model_name"})).unwrap();
        assert_eq!(restored.strategy(), ModelPoolStrategy::ByModelName);

        for state in [json!({"strategy": null}), json!({})] {
            let restored = ModelAllocator::from_state_dict(&state).unwrap();
            assert_eq!(
                restored.strategy(),
                ModelPoolStrategy::RoundRobin,
                "state {state}"
            );
        }
    }

    #[test]
    fn intelli_router_refuses_before_the_empty_pool_echo() {
        let allocator = ModelAllocator::new(&[], ModelPoolStrategy::IntelliRouter);
        assert!(allocator.is_empty());

        // Neither the requested-name echo nor the empty-pool config error: a typed refusal.
        let error = allocator
            .allocate(Some("gpt-4o"))
            .expect_err("an empty IntelliRouter pool is still refused");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("intelli_router"), "{error}");

        let error = allocator
            .allocate(None)
            .expect_err("an empty IntelliRouter pool is still refused");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("intelli_router"), "{error}");
    }

    #[test]
    fn from_state_dict_rejects_a_non_object() {
        for (value, expected) in [
            (json!(null), "null"),
            (json!(3), "a number"),
            (json!("round_robin"), "a string"),
            (json!([0]), "an array"),
        ] {
            let error = ModelAllocator::from_state_dict(&value).expect_err("non-object state");
            assert_eq!(error.kind(), ErrorKind::Validation, "value {value}");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn allocation_serde_is_snake_case() {
        let allocation = TeamModelAllocation::new("gpt-4o", 3);
        let value = serde_json::to_value(&allocation).unwrap();
        assert_eq!(value, json!({"model_name": "gpt-4o", "index": 3}));
        let parsed: TeamModelAllocation = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, allocation);
    }

    #[test]
    fn debug_reports_the_cursor_without_the_pool() {
        let allocator = ModelAllocator::new(&pool(2), ModelPoolStrategy::RoundRobin);
        let _ = allocator.allocate(None).unwrap();
        let debug = format!("{allocator:?}");
        assert!(debug.contains("round_robin"), "{debug}");
        assert!(debug.contains("entries: 2"), "{debug}");
        assert!(debug.contains("cursor: 1"), "{debug}");
        assert!(!debug.contains("sk-0"), "{debug}");
    }
}
