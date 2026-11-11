//! Per-run execution options: limits, cancellation, resume, and event
//! delivery.

use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::event::GraphEvent;

/// Tunables for one [`crate::CompiledGraph::run`] invocation.
///
/// Every field is optional so callers only override what they need; unset
/// values fall back to the compile options and then to the engine defaults.
#[derive(Debug, Clone)]
pub struct ExecutionOptions {
    /// Maximum number of supersteps; `None` uses the compile option, then
    /// [`crate::DEFAULT_MAX_SUPERSTEPS`].
    pub max_supersteps: Option<usize>,
    /// Per-attempt node timeout in milliseconds; `None` uses the compile
    /// option, then the node's own option.
    pub node_timeout_ms: Option<u64>,
    /// Cancellation token checked before every superstep and raced against
    /// every node attempt.
    pub cancellation: CancellationToken,
    /// Restore the stored snapshot for this session and continue it; the input
    /// value is ignored.
    pub resume: bool,
    /// Optional sink receiving every execution event in order.
    pub events: Option<UnboundedSender<GraphEvent>>,
    /// Mirror every event to the session stream as a `graph_event` trace frame.
    pub write_session_trace: bool,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            max_supersteps: None,
            node_timeout_ms: None,
            cancellation: CancellationToken::new(),
            resume: false,
            events: None,
            write_session_trace: true,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_permissive() {
        let options = ExecutionOptions::default();
        assert!(options.max_supersteps.is_none());
        assert!(options.node_timeout_ms.is_none());
        assert!(!options.resume);
        assert!(options.events.is_none());
        assert!(options.write_session_trace);
        assert!(!options.cancellation.is_cancelled());
    }

    #[test]
    fn tokens_are_independent_per_default() {
        let first = ExecutionOptions::default();
        first.cancellation.cancel();
        let second = ExecutionOptions::default();
        assert!(first.cancellation.is_cancelled());
        assert!(!second.cancellation.is_cancelled());
    }
}
