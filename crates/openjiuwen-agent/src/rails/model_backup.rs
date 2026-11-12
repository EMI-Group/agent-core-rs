//! The model failover rail: falls back to the next configured model when a call
//! fails.
//!
//! Python parity: Python's model-backup rail, expressed on the kernel
//! [`AgentRail`] seam.
//!
//! Intended behaviour (hooked in a later round): on `OnModelException` the rail
//! advances its cursor to the next backup model, installs it with
//! [`RailContext::set_model_override`], and calls
//! [`RailContext::request_retry`] with `None` so the loop retries immediately.
//! Once every backup has been used the context is left untouched and the
//! original error propagates.
use super::{AgentRail, RailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::Model;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
/// Tries the configured models in order when the primary model fails.
pub struct ModelBackupRail {
    models: Vec<Arc<dyn Model>>,
    next: AtomicUsize,
    name: String,
}
impl ModelBackupRail {
    /// Builds a rail over `models`, tried in order.
    #[must_use]
    pub fn new(models: Vec<Arc<dyn Model>>) -> Self {
        Self {
            models,
            next: AtomicUsize::new(0),
            name: "model_backup".to_owned(),
        }
    }
    /// Returns the backup models.
    #[must_use]
    pub fn models(&self) -> &[Arc<dyn Model>] {
        &self.models
    }
    /// Returns the number of backup models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }
    /// Whether no backup model is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}
impl Default for ModelBackupRail {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}
impl fmt::Debug for ModelBackupRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelBackupRail")
            .field("name", &self.name)
            .field("models", &self.models.len())
            .field("next", &self.next.load(Ordering::Relaxed))
            .finish()
    }
}
#[async_trait]
impl AgentRail for ModelBackupRail {
    fn name(&self) -> &str {
        &self.name
    }
    /// TODO: when a backup is left, advance `next`, call `set_model_override`
    /// with it, and `request_retry(None)`; otherwise leave the context as is.
    async fn on_model_exception(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rails::RailEvent;
    use openjiuwen_test_support::ScriptedModel;

    fn model(name: &str) -> Arc<dyn Model> {
        Arc::new(ScriptedModel::builder().model_name(name).build())
    }

    #[test]
    fn empty_by_default() {
        let rail = ModelBackupRail::default();
        assert_eq!(rail.name(), "model_backup");
        assert!(rail.is_empty());
        assert_eq!(rail.len(), 0);
        assert_eq!(rail.models().len(), 0);
    }
    #[test]
    fn keeps_the_configured_models_in_order() {
        let rail = ModelBackupRail::new(vec![model("first"), model("second")]);
        assert_eq!(rail.len(), 2);
        assert!(!rail.is_empty());
        assert_eq!(rail.models()[0].model_name(), "first");
        assert_eq!(rail.models()[1].model_name(), "second");
    }
    #[tokio::test]
    async fn hook_is_a_placeholder_that_leaves_the_context_untouched() {
        let rail = ModelBackupRail::new(vec![model("backup")]);
        let mut ctx = RailContext::new(RailEvent::OnModelException);
        rail.on_model_exception(&mut ctx).await.unwrap();
        assert!(ctx.model_override.is_none());
        assert!(ctx.take_retry().is_none());
    }
    #[test]
    fn debug_counts_models_and_shows_the_cursor() {
        let rail = ModelBackupRail::new(vec![model("a")]);
        let rendered = format!("{rail:?}");
        assert!(rendered.contains("models: 1"));
        assert!(rendered.contains("next: 0"));
    }
}
