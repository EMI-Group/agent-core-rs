//! Deferred agentdescent evolution surfaces.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::not_implemented;

/// The surface name used by [`EvolutionEngine`]'s errors.
const SURFACE: &str = "agentdescent evolution engine";

/// Candidate-selection policy of the deferred evolution engine.
///
/// Mirrors Python's `agentdescent` policy constants. The enum is real data (it
/// parses, formats, and round-trips through serde); only the engine that would
/// consume it is deferred.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionPolicy {
    /// Keep every evaluated candidate in the population.
    #[default]
    Full,
    /// Keep only candidates that pass the guardrails.
    Guarded,
    /// Keep candidates plus the reflections that produced them.
    Reflective,
}

impl EvolutionPolicy {
    /// Returns the `snake_case` wire name of the policy.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Guarded => "guarded",
            Self::Reflective => "reflective",
        }
    }
}

impl fmt::Display for EvolutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EvolutionPolicy {
    type Err = Error;

    /// Parses the `snake_case` wire name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for any name other than `full`, `guarded`,
    /// or `reflective`.
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "full" => Ok(Self::Full),
            "guarded" => Ok(Self::Guarded),
            "reflective" => Ok(Self::Reflective),
            other => Err(Error::validation(format!(
                "unknown evolution policy `{other}`: expected one of full, guarded, reflective"
            ))),
        }
    }
}

/// Distributed evolutionary-search engine — **deferred**.
///
/// Mirrors Python `openjiuwen.rsi.artifact_rsi.program_opt`'s dependence on the
/// `agentdescent` `EvolutionEngine` (`evolve`, `ledger`, `aggregator`,
/// `staleness`). Those internals are pinned to `agentdescent==0.4.6` and carry
/// distributed, deep-RL training machinery with no maintained Rust equivalent,
/// so this is an API-shaped stub listed in the crate `CONTEXT.md`
/// `## Deferred Scope` table.
///
/// The documented constructor always fails, so a caller cannot build one of
/// these by accident; [`Self::evolve`] is an associated function that reports
/// the same error without needing a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvolutionEngine;

impl EvolutionEngine {
    /// Mirrors `EvolutionEngine.__init__`; always reports that the engine is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] (`agentdescent evolution engine`).
    pub fn new() -> Result<Self> {
        Err(not_implemented(
            SURFACE,
            "needs the `agentdescent==0.4.6` evolve/ledger/aggregator internals and \
             distributed training, which have no maintained Rust equivalent",
        ))
    }

    /// Mirrors `EvolutionEngine.evolve`; always reports that the engine is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] (`agentdescent evolution engine`).
    pub fn evolve(_request: Value) -> Result<Value> {
        Err(not_implemented(
            SURFACE,
            "needs the `agentdescent==0.4.6` evolve/ledger/aggregator internals and \
             distributed training, which have no maintained Rust equivalent",
        ))
    }
}
