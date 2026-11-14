//! The team-memory observability bridge, attached where the facade is built.
//!
//! [`TeamMemoryWiring`](crate::memory::TeamMemoryWiring) builds the team's
//! [`TeamMemory`] facade; this module is the one place that decides whether that facade carries an
//! observer, and which one. It chains the crate's own bridge ([`memory_observer`] →
//! [`MemorySpanObserver`]) through
//! [`TeamMemory::with_observer`](openjiuwen_memory::team::TeamMemory::with_observer), which
//! *consumes* the facade, so the observer has to be known here rather than added later.
//!
//! # What a traced wiring produces
//!
//! Every observable team-memory operation (`search`, `write`, `forget`, `clear`, `prune`,
//! `extract`) becomes one `memory.{operation}` span plus one structured `tracing` event at target
//! `openjiuwen_agent_teams::memory`. On the `otel` build the observer also appends one
//! `SpanRecord` per operation to the [`OtelSpanSink`] the configuration carries, which keeps the
//! whole trace inspectable in process.
//!
//! # The trace caveat
//!
//! The bridge mints its trace from the team name
//! ([`TeamSpan::root_for`](crate::observability::TeamSpan::root_for)), so a wiring's `memory.*`
//! spans form a *sibling* trace of the runtime's `team.{team_name}` trace rather than children of
//! it: the wiring never sees the runtime's [`TeamSpan`](crate::observability::TeamSpan) root.
//! Joining the two needs a `MemorySpanObserver::from_root(TeamSpan)` constructor on the
//! observability node; until that exists, a host correlates the two traces by the `team.{name}`
//! attributes both roots carry.
//!
//! # Deferrals
//!
//! Real OTLP export and the `opentelemetry` SDK are not part of this crate: they belong to
//! `openjiuwen-extensions`, which sits in the same dependency layer and therefore cannot be
//! depended on from here. That is why the `otel` feature only fills an in-memory sink; a host that
//! wants OTLP forwards these `tracing` events (and, on `otel`, drains the sink) through the
//! extensions layer it already runs.
//!
//! # Cost
//!
//! Attaching is purely additive and infallible: the store's results, ordering and cancellation
//! behaviour are unchanged, the observer never awaits, and its only lock is the sink's short one.
//! `trace: false` or a disabled configuration attaches nothing at all.

use std::sync::Arc;

use openjiuwen_memory::team::{TeamMemory, TeamMemoryObserver};
use tracing::debug;

use crate::memory::config::{MemoryConfig, MemoryStoreConfig};
use crate::observability::memory::memory_observer;
use crate::observability::memory::MemorySpanObserver;

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_agent_teams::memory";

/// The span sink type [`MemoryConfig::otel_sink`] carries: the crate's in-memory `otel` sink.
#[cfg(feature = "otel")]
pub use crate::observability::otel::OtelSpanSink;

/// The span-sink slot of [`MemoryConfig`] in a build without the `otel` feature.
///
/// The real sink is `crate::observability::otel::OtelSpanSink`, which only exists on the `otel`
/// build. Without the feature this uninhabited stand-in keeps [`MemoryConfig`] one shape on every
/// build while making a sink impossible to construct, so
/// [`MemoryConfig::otel_sink`] is always `None` and nothing can be silently dropped.
#[cfg(not(feature = "otel"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtelSpanSink {}

/// Builds the facade `config` selects, with the observer it asks for already attached.
///
/// Answers the facade and, when tracing is on, the very observer the facade holds — so a caller
/// can keep the handle without reaching through the engine. Every store selection funnels through
/// here, which is what makes "a wiring is either traced or not" a single decision.
pub fn facade(
    team_id: &str,
    config: &MemoryConfig,
) -> (TeamMemory, Option<Arc<dyn TeamMemoryObserver>>) {
    let store = match &config.store {
        MemoryStoreConfig::Memory => TeamMemory::in_memory(),
        // This crate does not enable `openjiuwen-memory`'s `sqlite` feature, so a `File` store
        // keeps the in-memory facade and persists through `TEAM_MEMORY.md` instead.
        MemoryStoreConfig::File { .. } => {
            debug!(
                target: LOG_TARGET,
                team = team_id,
                "the file team memory store keeps the in-memory facade; the document is the \
                 durable image of the store"
            );
            TeamMemory::in_memory()
        }
    };
    let observer = observer_for(team_id, config);
    let memory = match observer.as_ref() {
        Some(observer) => store.with_observer(Arc::clone(observer)),
        None => store,
    };
    (memory, observer)
}

/// Builds the observer `config` asks for, or `None` when nothing should be observed.
///
/// A disabled configuration or `trace: false` answers `None`; otherwise the observer roots a fresh
/// `team.{team_id}` trace and, on the `otel` build, writes to [`MemoryConfig::otel_sink`] when one
/// is configured.
#[must_use]
pub fn observer_for(team_id: &str, config: &MemoryConfig) -> Option<Arc<dyn TeamMemoryObserver>> {
    if !config.enabled {
        return None;
    }
    if !config.trace {
        if config.otel_sink.is_some() {
            debug!(
                target: LOG_TARGET,
                team = team_id,
                "team memory tracing is off; the configured span sink stays untouched"
            );
        }
        return None;
    }
    Some(Arc::new(build_observer(team_id, config)))
}

/// Builds the concrete bridge, with the in-memory sink on the `otel` build.
// `config` is only read when the `otel` feature supplies a sink.
#[cfg_attr(not(feature = "otel"), allow(unused_variables))]
fn build_observer(team_id: &str, config: &MemoryConfig) -> MemorySpanObserver {
    #[cfg(feature = "otel")]
    if let Some(sink) = config.otel_sink.as_ref() {
        return memory_observer(team_id).with_sink(Arc::clone(sink));
    }
    memory_observer(team_id)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::config::MemoryExtractorKind;

    #[test]
    fn the_default_configuration_is_traced() {
        let config = MemoryConfig::default();
        assert!(config.trace);
        assert!(config.otel_sink.is_none());
        assert!(observer_for("team-1", &config).is_some());
    }

    #[test]
    fn trace_false_attaches_nothing() {
        let config = MemoryConfig {
            trace: false,
            ..MemoryConfig::default()
        };
        assert!(observer_for("team-1", &config).is_none());
    }

    #[test]
    fn a_disabled_configuration_attaches_nothing_even_when_traced() {
        let config = MemoryConfig {
            enabled: false,
            ..MemoryConfig::default()
        };
        assert!(config.trace);
        assert!(observer_for("team-1", &config).is_none());
    }

    #[test]
    fn a_traced_facade_holds_the_very_observer_that_was_returned() {
        let (memory, observer) = facade("team-1", &MemoryConfig::default());
        let observer = observer.expect("the default configuration is traced");
        let held = memory.observer().expect("the facade holds the observer");
        assert!(Arc::ptr_eq(held, &observer));
    }

    #[test]
    fn an_untraced_facade_holds_no_observer() {
        for config in [
            MemoryConfig {
                trace: false,
                ..MemoryConfig::default()
            },
            MemoryConfig {
                enabled: false,
                ..MemoryConfig::default()
            },
        ] {
            let (memory, observer) = facade("team-1", &config);
            assert!(observer.is_none());
            assert!(memory.observer().is_none());
        }
    }

    #[test]
    fn the_configuration_does_not_change_the_stored_facade_kind() {
        for store in [
            MemoryStoreConfig::Memory,
            MemoryStoreConfig::File { dir: None },
        ] {
            let config = MemoryConfig {
                store,
                extractor: MemoryExtractorKind::Rule,
                ..MemoryConfig::default()
            };
            let (memory, observer) = facade("team-1", &config);
            assert!(observer.is_some());
            assert!(memory.observer().is_some());
        }
    }

    #[test]
    fn each_facade_mints_its_own_trace() {
        let (_, first) = facade("team-1", &MemoryConfig::default());
        let (_, second) = facade("team-1", &MemoryConfig::default());
        let first = first.expect("traced");
        let second = second.expect("traced");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "no observer is shared through a global registry"
        );
    }

    #[cfg(not(feature = "otel"))]
    #[test]
    fn the_sink_slot_exists_and_stays_empty_without_the_feature() {
        // The field is part of every build, so the same code compiles either way; without the
        // feature the sink type is uninhabited and the slot can only ever be `None`.
        let config = MemoryConfig::default();
        assert!(config.otel_sink.is_none());
        assert!(observer_for("team-1", &config).is_some());
    }

    #[cfg(feature = "otel")]
    #[test]
    fn a_configured_sink_is_stored_on_the_observer() {
        let sink = Arc::new(OtelSpanSink::new(4));
        let config = MemoryConfig {
            otel_sink: Some(Arc::clone(&sink)),
            ..MemoryConfig::default()
        };
        let observer = observer_for("team-1", &config).expect("traced");
        observer.observe(&openjiuwen_memory::team::TeamMemoryEvent::new(
            openjiuwen_memory::team::TeamMemoryOperation::Write,
            "team-1",
        ));
        assert_eq!(sink.len(), 1);
    }

    #[cfg(feature = "otel")]
    #[test]
    fn an_untraced_configuration_leaves_the_sink_untouched() {
        let sink = Arc::new(OtelSpanSink::new(4));
        let config = MemoryConfig {
            trace: false,
            otel_sink: Some(Arc::clone(&sink)),
            ..MemoryConfig::default()
        };
        assert!(observer_for("team-1", &config).is_none());
        assert!(sink.is_empty());
    }
}
