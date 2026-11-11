//! Per-call options: cancellation/timeout, add behaviour, and search knobs.
//!
//! [`CallOptions`] carries runtime state (a [`CancellationToken`]) and therefore has no
//! `serde` implementation; the other types here are data and serialize normally.
use super::kind::MemoryType;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
/// The dedupe behaviour for a write.
///
/// This is the canonical definition of the type; `config::DedupePolicy` is the
/// configuration-facing re-export of it, so the engine can compare the config default
/// with a per-call override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DedupePolicy {
    /// Fold the incoming memory into the existing record (the default).
    #[default]
    Merge,
    /// Drop the incoming memory and report it as skipped.
    Skip,
    /// Replace the existing record's content with the incoming memory.
    Replace,
    /// Store the incoming memory even when a duplicate exists.
    Add,
}
/// Cancellation and deadline state attached to one memory call.
///
/// The default is "never cancelled, no deadline". [`CallOptions::run`] races the work
/// against cancellation and the optional timeout; [`CallOptions::check`] is the cheap
/// synchronous probe for loops that do not await.
#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    /// Token that cancels the call when triggered.
    pub cancellation: CancellationToken,
    /// Optional wall-clock deadline applied by [`CallOptions::run`].
    pub timeout: Option<Duration>,
}
impl CallOptions {
    /// Builds options that never cancel and never time out.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the options bound to `cancellation`.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }
    /// Returns the options with `timeout` as the deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    /// Returns `Err(Error::Cancelled)` when the token has been triggered.
    pub fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
    /// Runs `future`, racing it against cancellation and the optional timeout.
    ///
    /// Cancellation is checked first (`biased`), so a triggered token always yields
    /// `Error::Cancelled` even when the work would also have completed. An elapsed
    /// deadline yields `Error::Timeout`; otherwise the future's own output is returned.
    pub async fn run<F: Future>(&self, future: F) -> Result<F::Output> {
        let cancellation = self.cancellation.cancelled();
        tokio::pin!(cancellation);
        if let Some(limit) = self.timeout {
            let attempt = tokio::time::timeout(limit, future);
            tokio::pin!(attempt);
            tokio::select! {
                biased;
                () = &mut cancellation => Err(Error::Cancelled),
                result = &mut attempt => match result {
                    Ok(output) => Ok(output),
                    Err(_elapsed) => Err(Error::Timeout(format!(
                        "memory call exceeded its {limit:?} deadline"
                    ))),
                },
            }
        } else {
            tokio::pin!(future);
            tokio::select! {
                biased;
                () = &mut cancellation => Err(Error::Cancelled),
                output = &mut future => Ok(output),
            }
        }
    }
}
/// How recall scores candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Blend vector similarity and keyword relevance (the default).
    #[default]
    Hybrid,
    /// Use vector similarity only; unsupported without an embedding model.
    Semantic,
    /// Use keyword (BM25) relevance only.
    Keyword,
}
/// Options for a write.
///
/// The default inherits the engine's dedupe policy, embeds the content, and never
/// cancels.
#[derive(Debug, Clone, Default)]
pub struct AddOptions {
    /// Overrides the engine's dedupe policy for this write when set.
    pub dedupe: Option<DedupePolicy>,
    /// Skips embedding generation and vector indexing for this write.
    pub skip_embedding: bool,
    /// Cancellation and deadline state.
    pub call: CallOptions,
}
impl AddOptions {
    /// Returns the options with an explicit dedupe policy.
    #[must_use]
    pub fn with_dedupe(mut self, dedupe: DedupePolicy) -> Self {
        self.dedupe = Some(dedupe);
        self
    }
    /// Returns the options with embedding generation skipped.
    #[must_use]
    pub fn skip_embedding(mut self) -> Self {
        self.skip_embedding = true;
        self
    }
    /// Returns the options bound to `cancellation`.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.call.cancellation = cancellation;
        self
    }
}
/// Options for a recall.
///
/// Defaults mirror `MemoryEngineConfig`: `top_k = 5`, `threshold = 0.3`, no kind
/// restriction (the engine substitutes its searchable kinds), hybrid mode, and no
/// token budget.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of hits returned.
    pub top_k: usize,
    /// Minimum blended score a hit must reach.
    pub threshold: f32,
    /// Kinds to search; empty means the engine's searchable kinds.
    pub kinds: Vec<MemoryType>,
    /// Scoring mode.
    pub mode: SearchMode,
    /// Token budget for the returned hits; empty means the engine/config default.
    pub token_budget: Option<usize>,
    /// Cancellation and deadline state.
    pub call: CallOptions,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            threshold: 0.3,
            kinds: Vec::new(),
            mode: SearchMode::default(),
            token_budget: None,
            call: CallOptions::new(),
        }
    }
}
impl SearchOptions {
    /// Returns the options with `top_k` hits requested.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
    /// Returns the options with `threshold` as the minimum score.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
    /// Returns the options restricted to `kinds`.
    #[must_use]
    pub fn with_kinds(mut self, kinds: Vec<MemoryType>) -> Self {
        self.kinds = kinds;
        self
    }
    /// Returns the options with `mode` as the scoring mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }
    /// Returns the options with a token budget for the returned hits.
    #[must_use]
    pub fn with_token_budget(mut self, token_budget: usize) -> Self {
        self.token_budget = Some(token_budget);
        self
    }
    /// Returns the options bound to `cancellation`.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.call.cancellation = cancellation;
        self
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn call_options_default_is_live_and_undeadlined() {
        let options = CallOptions::default();
        assert!(!options.cancellation.is_cancelled());
        assert!(options.timeout.is_none());
        assert!(options.check().is_ok());
    }
    #[test]
    fn check_reports_cancellation() {
        let token = CancellationToken::new();
        let options = CallOptions::new().with_cancellation(token.clone());
        assert!(options.check().is_ok());
        token.cancel();
        assert!(matches!(options.check(), Err(Error::Cancelled)));
    }
    #[tokio::test]
    async fn run_returns_the_future_output() {
        let options = CallOptions::new();
        assert_eq!(options.run(async { 7_u32 }).await.unwrap(), 7);
    }
    #[tokio::test]
    async fn run_aborts_on_cancellation() {
        let token = CancellationToken::new();
        token.cancel();
        let options = CallOptions::new().with_cancellation(token);
        let result = options
            .run(tokio::time::sleep(Duration::from_secs(1)))
            .await;
        assert!(matches!(result, Err(Error::Cancelled)));
    }
    #[tokio::test]
    async fn run_times_out() {
        let options = CallOptions::new().with_timeout(Duration::from_millis(10));
        let result = options
            .run(tokio::time::sleep(Duration::from_secs(30)))
            .await;
        assert!(matches!(result, Err(Error::Timeout(_))));
    }
    #[tokio::test]
    async fn cancellation_wins_over_a_finished_timeout_branch() {
        let token = CancellationToken::new();
        token.cancel();
        let options = CallOptions::new()
            .with_cancellation(token)
            .with_timeout(Duration::from_secs(30));
        let result = options.run(async { 1 }).await;
        assert!(matches!(result, Err(Error::Cancelled)));
    }
    #[test]
    fn add_options_defaults_and_builders() {
        let options = AddOptions::default();
        assert!(options.dedupe.is_none());
        assert!(!options.skip_embedding);
        assert!(options.call.check().is_ok());
        let token = CancellationToken::new();
        let options = AddOptions::default()
            .with_dedupe(DedupePolicy::Add)
            .skip_embedding()
            .with_cancellation(token.clone());
        assert_eq!(options.dedupe, Some(DedupePolicy::Add));
        assert!(options.skip_embedding);
        token.cancel();
        assert!(matches!(options.call.check(), Err(Error::Cancelled)));
    }
    #[test]
    fn search_options_defaults_and_builders() {
        let options = SearchOptions::default();
        assert_eq!(options.top_k, 5);
        assert!((options.threshold - 0.3).abs() < f32::EPSILON);
        assert_eq!(options.kinds, Vec::new());
        assert_eq!(options.mode, SearchMode::Hybrid);
        assert!(options.token_budget.is_none());
        let token = CancellationToken::new();
        let options = SearchOptions::default()
            .with_top_k(10)
            .with_threshold(0.5)
            .with_kinds(vec![MemoryType::Coding])
            .with_mode(SearchMode::Keyword)
            .with_token_budget(256)
            .with_cancellation(token);
        assert_eq!(options.top_k, 10);
        assert!((options.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(options.kinds, vec![MemoryType::Coding]);
        assert_eq!(options.mode, SearchMode::Keyword);
        assert_eq!(options.token_budget, Some(256));
    }
    #[test]
    fn search_mode_serde_uses_snake_case() {
        assert_eq!(
            serde_json::to_value(SearchMode::Hybrid).unwrap(),
            serde_json::json!("hybrid")
        );
        assert_eq!(
            serde_json::to_value(SearchMode::Semantic).unwrap(),
            serde_json::json!("semantic")
        );
        assert_eq!(
            serde_json::to_value(SearchMode::Keyword).unwrap(),
            serde_json::json!("keyword")
        );
        assert_eq!(
            serde_json::from_str::<SearchMode>("\"keyword\"").unwrap(),
            SearchMode::Keyword
        );
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }
    #[test]
    fn dedupe_policy_serde_and_default() {
        assert_eq!(DedupePolicy::default(), DedupePolicy::Merge);
        for policy in [
            DedupePolicy::Merge,
            DedupePolicy::Skip,
            DedupePolicy::Replace,
            DedupePolicy::Add,
        ] {
            let json = serde_json::to_value(policy).unwrap();
            assert_eq!(
                serde_json::from_value::<DedupePolicy>(json).unwrap(),
                policy
            );
        }
        assert_eq!(
            serde_json::to_value(DedupePolicy::Skip).unwrap(),
            serde_json::json!("skip")
        );
    }
}
