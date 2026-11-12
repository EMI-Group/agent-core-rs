//! The machine-readable counterpart of the crate's "Deferred Scope" table.
//!
//! Python `openjiuwen.agent_evolving` is a large area whose trainer, evaluator and
//! optimizer packages (plus the online-RL service) depend on a Python ML stack
//! (`torch`, `transformers`, `verl`, `ray`, Redis, vLLM). This crate ports the
//! portable, offline core and turns every deferred path into an explicit
//! [`openjiuwen_common::error::Error::Unsupported`] carrying the same list the
//! documentation and tests use:
//!
//! ```
//! use openjiuwen_agent_evolving::deferred::{deferred_features, unsupported};
//!
//! assert!(!deferred_features().is_empty());
//! assert!(deferred_features()
//!     .iter()
//!     .any(|feature| feature.area.contains("RL trainer")));
//! assert_eq!(
//!     unsupported("reward").to_string(),
//!     "unsupported: reward is deferred; see docs/PARITY.md"
//! );
//! ```
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Error;

/// One Python `openjiuwen.agent_evolving` area that is not ported.
///
/// Both fields are stable, human-readable `&'static str`s so the table can be
/// serialized into diagnostics, CLI output, and `/v1/...` error envelopes without
/// allocation.
///
/// Deserialization resolves the payload's `area` against [`DEFERRED_FEATURES`] so
/// the result borrows the canonical static strings; the canonical reason always
/// wins and an unknown area is a deserialization error (there is no owned data to
/// borrow it from).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeferredFeature {
    /// Short, stable name of the deferred area.
    pub area: &'static str,
    /// Why the area is deferred (the dependency or external system it needs).
    pub reason: &'static str,
}

/// Owned shape of a [`DeferredFeature`]; the deserialization entry point.
#[derive(Deserialize)]
struct OwnedDeferredFeature {
    area: String,
    #[serde(default)]
    reason: String,
}

impl<'de> Deserialize<'de> for DeferredFeature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let owned = OwnedDeferredFeature::deserialize(deserializer)?;
        Self::try_from(owned).map_err(serde::de::Error::custom)
    }
}

impl TryFrom<OwnedDeferredFeature> for DeferredFeature {
    type Error = String;

    fn try_from(owned: OwnedDeferredFeature) -> Result<Self, Self::Error> {
        DEFERRED_FEATURES
            .iter()
            .find(|feature| feature.area == owned.area)
            .copied()
            .ok_or_else(|| {
                format!(
                    "unknown deferred area: {} (reason: {})",
                    owned.area, owned.reason
                )
            })
    }
}

/// Every deferred area, in the order of the crate-level "Deferred Scope" table.
pub const DEFERRED_FEATURES: &[DeferredFeature] = &[
    DeferredFeature {
        area: "RL trainer (PPO/verl/ray)",
        reason: "Python-only distributed ML stack (torch, transformers, verl, ray)",
    },
    DeferredFeature {
        area: "online RL service state machine",
        reason: "needs Redis task queues, vLLM/verl executors and gateway integration",
    },
    DeferredFeature {
        area: "LLM-as-judge / default model evaluator",
        reason: "requires a live model and is non-deterministic",
    },
    DeferredFeature {
        area: "prompt/textual-gradient optimizers",
        reason: "research code tied to Python LLM flows",
    },
    DeferredFeature {
        area: "evaluator_pipeline",
        reason: "external Docker + git harness protocol",
    },
    DeferredFeature {
        area: "reward, attribution, experience, sharing, signal, trajectory and ttse subsystems",
        reason: "large, near-leaf subsystems with vendor/browser dependencies",
    },
    DeferredFeature {
        area: "evolution prompt/tool asset packages",
        reason: "LLM-flow prompt text and the evolution toolset are not ported",
    },
    DeferredFeature {
        area: "agent_rl proxy/judge/verification gateway",
        reason: "vendor HTTP services and authentication",
    },
    DeferredFeature {
        area: "RL YAML configuration ecosystem",
        reason: "Python training-stack configuration (verl/ray configs)",
    },
    DeferredFeature {
        area: "full Python agent_evolving unit suite",
        reason: "representative offline tests only",
    },
];

/// Returns [`DEFERRED_FEATURES`], the list of areas this crate does not implement.
#[must_use]
pub fn deferred_features() -> &'static [DeferredFeature] {
    DEFERRED_FEATURES
}

/// Builds the error a deferred code path returns.
///
/// `area` names the missing capability (for example `"reward"` or
/// `"OfflineRlOptimizer::train"`); the message always points at `docs/PARITY.md`.
pub fn unsupported(area: &str) -> Error {
    Error::unsupported(format!("{area} is deferred; see docs/PARITY.md"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_covers_every_documented_area() {
        let areas: Vec<&str> = deferred_features().iter().map(|f| f.area).collect();
        assert_eq!(areas.len(), 10);
        assert!(areas.contains(&"RL trainer (PPO/verl/ray)"));
        assert!(areas.contains(&"online RL service state machine"));
        assert!(areas.contains(&"evaluator_pipeline"));
        assert!(areas.contains(&"RL YAML configuration ecosystem"));
        assert!(areas.contains(&"full Python agent_evolving unit suite"));
    }

    #[test]
    fn entries_are_non_empty() {
        for feature in deferred_features() {
            assert_ne!(feature.area.trim(), "");
            assert_ne!(feature.reason.trim(), "");
        }
    }

    #[test]
    fn unsupported_names_the_area_and_the_doc_pointer() {
        let error = unsupported("reward");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert_eq!(error.message(), "reward is deferred; see docs/PARITY.md");
    }

    #[test]
    fn table_round_trips_through_json() {
        let json = serde_json::to_string(deferred_features()).expect("serialize");
        let decoded: Vec<DeferredFeature> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.as_slice(), deferred_features());
    }

    #[test]
    fn deserialization_resolves_the_canonical_entry() {
        let decoded: DeferredFeature = serde_json::from_str(
            r#"{"area":"evaluator_pipeline","reason":"ignored","unknown_field":true}"#,
        )
        .expect("deserialize");
        assert_eq!(decoded.area, "evaluator_pipeline");
        assert_eq!(decoded.reason, "external Docker + git harness protocol");
    }

    #[test]
    fn deserialization_rejects_an_unknown_area() {
        let error =
            serde_json::from_str::<DeferredFeature>(r#"{"area":"teleporter"}"#).unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown deferred area: teleporter"));
    }
}
