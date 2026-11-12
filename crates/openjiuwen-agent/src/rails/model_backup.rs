//! The model failover rail: falls back to the next configured model when a call
//! fails.
//!
//! Python parity: Python's model-backup rail, expressed on the kernel
//! [`AgentRail`] seam.
//!
//! On `OnModelException` the rail advances its cursor to the next backup model,
//! installs it with [`RailContext::set_model_override`], and calls
//! [`RailContext::request_retry`] with `None` so the loop retries immediately.
//! Once every backup has been used the context is left untouched and the
//! original error propagates.
use super::{AgentRail, RailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::Model;
use parking_lot::Mutex;
use std::fmt;
use std::sync::Arc;
/// Tries the configured models in order when the primary model fails.
pub struct ModelBackupRail {
    models: Vec<Arc<dyn Model>>,
    next: Mutex<usize>,
    name: String,
}
impl ModelBackupRail {
    /// Builds a rail over `models`, tried in order.
    #[must_use]
    pub fn new(models: Vec<Arc<dyn Model>>) -> Self {
        Self {
            models,
            next: Mutex::new(0),
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
    /// Takes the next backup model, advancing the cursor; `None` once exhausted.
    fn take_next(&self) -> Option<Arc<dyn Model>> {
        let mut next = self.next.lock();
        let model = self.models.get(*next).cloned()?;
        *next += 1;
        Some(model)
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
            .field("next", &*self.next.lock())
            .finish()
    }
}
#[async_trait]
impl AgentRail for ModelBackupRail {
    fn name(&self) -> &str {
        &self.name
    }
    /// Installs the next unused backup model and requests an immediate retry.
    /// Leaves the context untouched once every backup has been consumed.
    async fn on_model_exception(&self, ctx: &mut RailContext) -> Result<()> {
        let Some(model) = self.take_next() else {
            return Ok(());
        };
        ctx.set_model_override(model);
        ctx.request_retry(None);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::rails::{RailEvent, RetryRequest};
    use openjiuwen_test_support::ScriptedModel;

    fn model(name: &str) -> Arc<dyn Model> {
        Arc::new(ScriptedModel::builder().model_name(name).build())
    }

    fn context() -> RailContext {
        RailContext::new(RailEvent::OnModelException)
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
    async fn uses_backups_in_order_and_requests_an_immediate_retry() {
        let rail = ModelBackupRail::new(vec![model("first"), model("second")]);

        let mut ctx = context();
        rail.on_model_exception(&mut ctx).await.unwrap();
        let installed = ctx.model_override.as_ref().unwrap();
        assert_eq!(installed.model_name(), "first");
        assert_eq!(ctx.take_retry(), Some(RetryRequest::immediate()));

        let mut ctx = context();
        rail.on_model_exception(&mut ctx).await.unwrap();
        let installed = ctx.model_override.as_ref().unwrap();
        assert_eq!(installed.model_name(), "second");
        assert_eq!(ctx.take_retry(), Some(RetryRequest::immediate()));
    }
    #[tokio::test]
    async fn leaves_the_context_untouched_once_exhausted() {
        let rail = ModelBackupRail::new(vec![model("first"), model("second")]);
        rail.on_model_exception(&mut context()).await.unwrap();
        rail.on_model_exception(&mut context()).await.unwrap();

        let mut ctx = context();
        rail.on_model_exception(&mut ctx).await.unwrap();
        assert!(ctx.model_override.is_none());
        assert!(ctx.take_retry().is_none());
    }
    #[tokio::test]
    async fn a_single_backup_is_used_once() {
        let rail = ModelBackupRail::new(vec![model("only")]);

        let mut ctx = context();
        rail.on_model_exception(&mut ctx).await.unwrap();
        assert_eq!(ctx.model_override.as_ref().unwrap().model_name(), "only");
        assert_eq!(ctx.take_retry(), Some(RetryRequest::immediate()));

        let mut ctx = context();
        rail.on_model_exception(&mut ctx).await.unwrap();
        assert!(ctx.model_override.is_none());
        assert!(ctx.take_retry().is_none());
    }
    #[tokio::test]
    async fn an_empty_rail_never_overrides() {
        let rail = ModelBackupRail::default();
        let mut ctx = context();
        rail.on_model_exception(&mut ctx).await.unwrap();
        assert!(ctx.model_override.is_none());
        assert!(ctx.take_retry().is_none());
    }
    #[tokio::test]
    async fn debug_reports_the_advancing_cursor() {
        let rail = ModelBackupRail::new(vec![model("a"), model("b")]);
        assert!(format!("{rail:?}").contains("next: 0"));
        rail.on_model_exception(&mut context()).await.unwrap();
        let rendered = format!("{rail:?}");
        assert!(rendered.contains("models: 2"));
        assert!(rendered.contains("next: 1"));
    }
}
