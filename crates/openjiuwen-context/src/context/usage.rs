//! The prepared window's `context.usage` block.
//!
//! The snapshot describes one logical request, so it is derived from the
//! assembled window and its per-role statistics rather than from the stored
//! history; an analyzer without a window limit is copied with the resolved one.
//!
//! The copy keeps the analyzer's own counter and identity: those are the only
//! pairing whose provenance the analyzer can vouch for, and substituting the
//! context's counter would republish an approximate count under the analyzer's
//! native identity.
use std::sync::Arc;

use crate::policy::ContextWindowSnapshot;
use crate::stats::ContextStats;
use crate::usage::{AnalysisInput, ContextUsageAnalyzer, ContextUsageSnapshot};

use super::ConversationContext;

impl ConversationContext {
    /// The analyzer that fills [`PreparedContext::usage`], when configured.
    ///
    /// [`PreparedContext::usage`]: crate::stats::PreparedContext::usage
    #[must_use]
    pub fn usage_analyzer(&self) -> Option<&Arc<ContextUsageAnalyzer>> {
        self.usage_analyzer.as_ref()
    }

    /// Wraps `analyzer` with the resolved window limit when it has none.
    ///
    /// Only the limit is substituted: the analyzer keeps counting with its own
    /// counter, so the published identity always describes the counter that ran.
    /// A per-context `token_counter` override therefore affects the processors
    /// and the window, not the provenance of the usage snapshot.
    fn analyzer_with_limit(&self, analyzer: &ContextUsageAnalyzer) -> Option<ContextUsageAnalyzer> {
        if analyzer.context_window_limit().is_some() {
            return None;
        }
        let limit = u64::try_from(self.resolved_window().tokens).unwrap_or(u64::MAX);
        Some(analyzer.with_derived_limit(limit))
    }

    /// Measures `input` and reconciles it with the provider total.
    pub(crate) fn analyze_window(
        &self,
        analyzer: &ContextUsageAnalyzer,
        input: &AnalysisInput<'_>,
    ) -> ContextUsageSnapshot {
        match self.analyzer_with_limit(analyzer) {
            Some(derived) => derived.analyze(input),
            None => analyzer.analyze(input),
        }
    }

    /// Freezes the local measurement of `input` into a report.
    pub(crate) fn build_window_report(
        &self,
        analyzer: &ContextUsageAnalyzer,
        input: &AnalysisInput<'_>,
    ) -> crate::usage::ContextWindowTokenReport {
        match self.analyzer_with_limit(analyzer) {
            Some(derived) => derived.build_report(input),
            None => analyzer.build_report(input),
        }
    }

    /// Builds the `context.usage` block of a prepared window, when configured.
    pub(super) fn usage_snapshot_for(
        &self,
        window: &ContextWindowSnapshot,
        stats: &ContextStats,
    ) -> Option<ContextUsageSnapshot> {
        let analyzer = self.usage_analyzer.as_ref()?;
        let mut input = AnalysisInput::new("", &self.session_id);
        input.context_id = Some(&self.context_id);
        input.system_messages = Some(&window.system_messages);
        input.context_messages = &window.context_messages;
        input.tools = &window.tools;
        input.context_stats = Some(stats);
        Some(self.analyze_window(analyzer, &input))
    }
}
