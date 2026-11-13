//! `MemoryCallOperator`: memory-recall parameters plus the real recall path.
//!
//! Ports `openjiuwen/core/operator/memory_call/base.py`, extended with the
//! retrieval knobs an evolving agent tunes: `top_k`, `threshold`,
//! `memory_scope_id`, and `enabled`. Python's handle only carries
//! `enabled`/`max_retries` and never executes memory operations; the Rust port
//! adds [`MemoryCallOperator::recall`] behind the [`MemoryRecall`] seam so the
//! tuned parameters can drive a real [`LongTermMemory`] recall without this
//! crate mutating the memory crate.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::{LongTermMemory, MemoryScope, SearchOptions};
use serde_json::{json, Value};

use super::base::{ApplyResult, Operator, ParameterCallback, TunableKind, TunableSpec};

/// Default number of recalled records.
pub const DEFAULT_TOP_K: usize = 5;

/// Default minimum blended score of a recalled record.
pub const DEFAULT_THRESHOLD: f32 = 0.3;

/// Largest `top_k` a tuned operator may request.
pub const MAX_TOP_K: usize = 100;

/// Minimal recall seam the memory operator drives.
///
/// [`LongTermMemory`] implements it; a test or a vendor memory backend can
/// implement it too without this crate depending on that backend.
#[async_trait]
pub trait MemoryRecall: Send + Sync {
    /// Recalls the record texts of `scope_id` for `query`.
    ///
    /// # Errors
    ///
    /// Returns the backend's memory error.
    async fn recall_texts(
        &self,
        scope_id: &str,
        query: &str,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<String>>;
}

#[async_trait]
impl<T: MemoryRecall + ?Sized> MemoryRecall for std::sync::Arc<T> {
    async fn recall_texts(
        &self,
        scope_id: &str,
        query: &str,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<String>> {
        (**self)
            .recall_texts(scope_id, query, top_k, threshold)
            .await
    }
}

#[async_trait]
impl MemoryRecall for LongTermMemory {
    async fn recall_texts(
        &self,
        scope_id: &str,
        query: &str,
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<String>> {
        let scope_id = if scope_id.trim().is_empty() {
            openjiuwen_memory::DEFAULT_ID
        } else {
            scope_id
        };
        let scope = MemoryScope::new(openjiuwen_memory::DEFAULT_ID, scope_id);
        let options = SearchOptions::default()
            .with_top_k(top_k)
            .with_threshold(threshold);
        let hits = self.recall(&scope, query, &options).await?;
        Ok(hits.into_iter().map(|hit| hit.record.content).collect())
    }
}

/// Tunable memory-recall parameters of one memory call.
pub struct MemoryCallOperator {
    operator_id: String,
    top_k: usize,
    threshold: f32,
    memory_scope_id: String,
    enabled: bool,
    on_parameter_updated: Option<ParameterCallback>,
}

impl MemoryCallOperator {
    /// Creates an enabled handle with the default scope and scoring knobs.
    #[must_use]
    pub fn new(operator_id: impl Into<String>) -> Self {
        Self {
            operator_id: operator_id.into(),
            top_k: DEFAULT_TOP_K,
            threshold: DEFAULT_THRESHOLD,
            memory_scope_id: String::new(),
            enabled: true,
            on_parameter_updated: None,
        }
    }

    /// Installs the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// Number of records to recall.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// Minimum blended score of a recalled record.
    #[must_use]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// The memory scope id; an empty value means the default scope.
    #[must_use]
    pub fn memory_scope_id(&self) -> &str {
        &self.memory_scope_id
    }

    /// Whether recall runs at all.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Recalls `query` through `memory` with the tuned parameters.
    ///
    /// A disabled operator returns an empty result without touching the
    /// backend.
    ///
    /// # Errors
    ///
    /// Returns the backend's error from [`MemoryRecall::recall_texts`].
    pub async fn recall(&self, memory: &dyn MemoryRecall, query: &str) -> Result<Vec<String>> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        memory
            .recall_texts(&self.memory_scope_id, query, self.top_k, self.threshold)
            .await
    }

    /// Parses and stores one field, returning its canonical JSON value.
    fn store(&mut self, target: &str, value: &Value) -> Result<Value> {
        match target {
            "top_k" => {
                let top_k = serde_json::from_value::<usize>(value.clone()).map_err(|_| {
                    Error::validation("tunable 'top_k' expects a non-negative integer")
                })?;
                self.top_k = top_k.min(MAX_TOP_K);
                Ok(json!(self.top_k))
            }
            "threshold" => {
                let threshold = serde_json::from_value::<f32>(value.clone())
                    .map_err(|_| Error::validation("tunable 'threshold' expects a number"))?;
                self.threshold = threshold.clamp(0.0, 1.0);
                Ok(json!(f64::from(self.threshold)))
            }
            "memory_scope_id" => {
                let value = value
                    .as_str()
                    .ok_or_else(|| Error::validation("tunable 'memory_scope_id' expects text"))?;
                value.clone_into(&mut self.memory_scope_id);
                Ok(Value::String(self.memory_scope_id.clone()))
            }
            "enabled" => {
                let value = value
                    .as_bool()
                    .ok_or_else(|| Error::validation("tunable 'enabled' expects a boolean"))?;
                self.enabled = value;
                Ok(Value::Bool(self.enabled))
            }
            other => Err(Error::validation(format!("unknown target '{other}'"))),
        }
    }

    /// Fires the callback for `target`.
    fn fire(&self, target: &str, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(target, value);
        }
    }
}

#[async_trait]
impl Operator for MemoryCallOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        vec![
            TunableSpec::new("top_k", TunableKind::Number)
                .with_description("number of memory records to recall"),
            TunableSpec::new("threshold", TunableKind::Number)
                .with_description("minimum blended score in 0.0..=1.0"),
            TunableSpec::new("memory_scope_id", TunableKind::Text)
                .with_description("memory scope id; empty selects the default scope"),
            TunableSpec::new("enabled", TunableKind::Bool)
                .with_description("whether memory recall runs"),
        ]
    }

    fn get_state(&self) -> Value {
        json!({
            "top_k": self.top_k,
            "threshold": f64::from(self.threshold),
            "memory_scope_id": self.memory_scope_id,
            "enabled": self.enabled,
        })
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        let stored = match self.store(target, &value) {
            Ok(stored) => stored,
            Err(error) => return Ok(ApplyResult::skipped(target, error.to_string())),
        };
        self.fire(target, &stored);
        Ok(ApplyResult::applied(target))
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        let object = state
            .as_object()
            .ok_or_else(|| Error::validation("operator state must be a JSON object"))?;
        for target in ["top_k", "threshold", "memory_scope_id", "enabled"] {
            let Some(value) = object.get(target) else {
                continue;
            };
            let stored = self.store(target, value)?;
            self.fire(target, &stored);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::assert_is_empty
    )]

    use super::*;
    use openjiuwen_memory::MemoryEngineConfig;
    use std::sync::Arc;

    #[derive(Default)]
    struct FakeMemory {
        hits: Vec<String>,
        calls: parking_lot::Mutex<Vec<(String, String, usize, f32)>>,
    }

    #[async_trait]
    impl MemoryRecall for FakeMemory {
        async fn recall_texts(
            &self,
            scope_id: &str,
            query: &str,
            top_k: usize,
            threshold: f32,
        ) -> Result<Vec<String>> {
            self.calls
                .lock()
                .push((scope_id.to_owned(), query.to_owned(), top_k, threshold));
            Ok(self.hits.clone())
        }
    }

    type Recorded = Arc<parking_lot::Mutex<Vec<(String, Value)>>>;

    fn recording_callback() -> (ParameterCallback, Recorded) {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: ParameterCallback = Arc::new(move |target: &str, value: &Value| {
            sink.lock().push((target.to_owned(), value.clone()));
        });
        (callback, log)
    }

    #[test]
    fn tunables_and_defaults() {
        let operator = MemoryCallOperator::new("agent/memory_call");
        assert_eq!(operator.top_k(), DEFAULT_TOP_K);
        assert_eq!(operator.threshold(), DEFAULT_THRESHOLD);
        assert!(operator.enabled());
        assert_eq!(operator.memory_scope_id(), "");
        let names: Vec<String> = operator
            .get_tunables()
            .into_iter()
            .map(|spec| spec.name)
            .collect();
        assert_eq!(names, ["top_k", "threshold", "memory_scope_id", "enabled"]);
    }

    #[tokio::test]
    async fn disabled_operator_returns_nothing_and_skips_backend() {
        let mut operator = MemoryCallOperator::new("op");
        operator.set_parameter("enabled", json!(false)).unwrap();
        let memory = FakeMemory::default();
        let hits = operator.recall(&memory, "q").await.unwrap();
        assert!(hits.is_empty());
        assert!(memory.calls.lock().is_empty());
    }

    #[tokio::test]
    async fn recall_forwards_the_tuned_parameters() {
        let mut operator = MemoryCallOperator::new("op");
        operator.set_parameter("top_k", json!(2)).unwrap();
        operator.set_parameter("threshold", json!(0.75)).unwrap();
        operator
            .set_parameter("memory_scope_id", json!("workspace-1"))
            .unwrap();
        let memory = FakeMemory {
            hits: vec!["a".to_owned(), "b".to_owned()],
            ..FakeMemory::default()
        };
        let hits = operator.recall(&memory, "dark mode").await.unwrap();
        assert_eq!(hits, ["a", "b"]);
        assert_eq!(
            memory.calls.lock().as_slice(),
            [("workspace-1".to_owned(), "dark mode".to_owned(), 2, 0.75)]
        );
    }

    #[test]
    fn set_parameter_clamps_and_fires() {
        let (callback, log) = recording_callback();
        let mut operator = MemoryCallOperator::new("op").with_callback(callback);
        operator.set_parameter("top_k", json!(10_000)).unwrap();
        assert_eq!(operator.top_k(), MAX_TOP_K);
        operator.set_parameter("threshold", json!(-1.0)).unwrap();
        assert_eq!(operator.threshold(), 0.0);
        operator.set_parameter("threshold", json!(2.0)).unwrap();
        assert_eq!(operator.threshold(), 1.0);
        let targets: Vec<String> = log
            .lock()
            .iter()
            .map(|(target, _)| target.clone())
            .collect();
        assert_eq!(targets, ["top_k", "threshold", "threshold"]);
        assert!(!operator.set_parameter("unknown", json!(1)).unwrap().applied);
        assert!(
            !operator
                .set_parameter("enabled", json!("yes"))
                .unwrap()
                .applied
        );
    }

    #[test]
    fn state_round_trips_and_load_fires() {
        let mut operator = MemoryCallOperator::new("op");
        operator.set_parameter("top_k", json!(3)).unwrap();
        operator.set_parameter("enabled", json!(false)).unwrap();
        let state = operator.get_state();
        let (callback, log) = recording_callback();
        let mut restored = MemoryCallOperator::new("op").with_callback(callback);
        restored.load_state(state.clone()).unwrap();
        assert_eq!(restored.get_state(), state);
        assert!(!restored.enabled());
        assert_eq!(log.lock().len(), 4);
    }

    #[tokio::test]
    async fn long_term_memory_implements_the_seam() {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .unwrap();
        let operator = MemoryCallOperator::new("op");
        let hits = operator.recall(&memory, "anything").await.unwrap();
        assert!(hits.is_empty());
    }
}
