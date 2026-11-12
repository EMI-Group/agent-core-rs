//! The subagent seam and its prompt rail.
//!
//! Ports the prompt half of `openjiuwen/harness/rails/subagent/subagent_rail.py`
//! and the provider-neutral part of `subagents/subagent_runtime`: this module
//! defines *what* a subagent runtime must offer, not *how* it runs. The runtime
//! itself lives in `crate::subagents`; keeping the seam here lets the rail
//! publish the persistent-subagent instructions without importing the runtime.
//!
//! The rail never spawns anything itself — it only publishes
//! [`crate::prompts::subagent_section`] before the model call and exposes the
//! [`SubAgentSpawner`] that the task tool and upper layers drive.
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::prompts::{self, PromptSection};

use super::{publish_prompt_section, rail_priority, HarnessRail, HarnessRailContext};

/// A request to start one subagent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SpawnRequest {
    /// The task the subagent should perform.
    pub prompt: String,
    /// Preset (role) the runtime should instantiate.
    pub preset: Option<String>,
    /// Caller-chosen agent id; the runtime assigns one when absent.
    pub agent_id: Option<String>,
    /// Task id of the parent, for transcript grouping.
    pub parent_task_id: Option<String>,
}

impl SpawnRequest {
    /// A request running `prompt`.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Self::default()
        }
    }

    /// Returns this request with `preset` as its preset.
    #[must_use]
    pub fn with_preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Returns this request with `agent_id` as its agent id.
    #[must_use]
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Returns this request with `parent_task_id` as its parent task.
    #[must_use]
    pub fn with_parent_task_id(mut self, parent_task_id: impl Into<String>) -> Self {
        self.parent_task_id = Some(parent_task_id.into());
        self
    }
}

/// The handle a runtime returns after accepting a [`SpawnRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SubAgentHandle {
    /// Runtime-assigned id, used by every later operation.
    pub id: String,
    /// Preset the subagent runs, when one was requested.
    pub preset: Option<String>,
}

impl SubAgentHandle {
    /// Builds a handle from its id and optional preset.
    #[must_use]
    pub fn new(id: impl Into<String>, preset: Option<String>) -> Self {
        Self {
            id: id.into(),
            preset,
        }
    }
}

/// Lifecycle state of one subagent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubAgentState {
    /// Accepted and still working.
    Running,
    /// Finished successfully; its output is available.
    Completed,
    /// Finished with an error.
    Failed,
    /// Shut down by the parent.
    Closed,
}

/// A snapshot of one subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SubAgentStatus {
    /// Runtime-assigned id.
    pub id: String,
    /// Current lifecycle state.
    pub state: SubAgentState,
    /// Final output, once it completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Failure message, when it failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SubAgentStatus {
    /// Builds a running status without output or error.
    #[must_use]
    pub fn running(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            state: SubAgentState::Running,
            output: None,
            error: None,
        }
    }

    /// Whether the subagent reached a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            SubAgentState::Completed | SubAgentState::Failed | SubAgentState::Closed
        )
    }
}

/// The runtime seam a harness subagent needs.
///
/// Every method is object-safe (no generics, no `impl Trait`), so the rail and
/// the task tool can hold an `Arc<dyn SubAgentSpawner>`.
#[async_trait::async_trait]
pub trait SubAgentSpawner: Send + Sync {
    /// Starts a subagent.
    ///
    /// # Errors
    ///
    /// The runtime's concurrency, preset, or quota failure.
    async fn spawn(&self, request: SpawnRequest) -> Result<SubAgentHandle>;

    /// Waits for `id` to finish, or for `timeout` to expire and returns its
    /// current (possibly still running) status.
    ///
    /// # Errors
    ///
    /// An unknown id or a runtime failure.
    async fn wait(&self, id: &str, timeout: Option<Duration>) -> Result<SubAgentStatus>;

    /// Lists every subagent the runtime knows about.
    ///
    /// # Errors
    ///
    /// A runtime failure while collecting the snapshots.
    async fn list(&self) -> Result<Vec<SubAgentStatus>>;

    /// Delivers `input` to a running subagent.
    ///
    /// # Errors
    ///
    /// An unknown or no-longer-running id.
    async fn send_input(&self, id: &str, input: String) -> Result<()>;

    /// Shuts a subagent down.
    ///
    /// # Errors
    ///
    /// An unknown id or a runtime failure during shutdown.
    async fn close(&self, id: &str) -> Result<()>;
}

/// Publishes the persistent-subagent instructions and exposes the runtime seam.
///
/// Dispatch priority is [`rail_priority::SUBAGENT`] (95), the slot of the
/// Python subagent rail.
pub struct SubagentRail {
    spawner: Option<Arc<dyn SubAgentSpawner>>,
}

impl SubagentRail {
    /// Builds a rail without a runtime seam.
    #[must_use]
    pub fn new() -> Self {
        Self { spawner: None }
    }

    /// Returns this rail with `spawner` as its runtime seam.
    #[must_use]
    pub fn with_spawner(mut self, spawner: Arc<dyn SubAgentSpawner>) -> Self {
        self.spawner = Some(spawner);
        self
    }

    /// The runtime seam, when one is wired.
    #[must_use]
    pub fn spawner(&self) -> Option<&Arc<dyn SubAgentSpawner>> {
        self.spawner.as_ref()
    }
}

impl Default for SubagentRail {
    /// Equivalent to [`SubagentRail::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SubagentRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubagentRail")
            .field("spawner", &self.spawner.is_some())
            .finish()
    }
}

#[async_trait::async_trait]
impl HarnessRail for SubagentRail {
    fn name(&self) -> &'static str {
        "subagent"
    }

    fn priority(&self) -> i32 {
        rail_priority::SUBAGENT
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        publish_prompt_section(ctx, &prompts::subagent_section());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Mutex;

    use openjiuwen_agent::RailEvent;

    use super::*;
    use crate::prompts::{prompt_sections, SectionName};

    /// An in-memory [`SubAgentSpawner`] that records every call.
    #[derive(Default)]
    struct FakeSpawner {
        calls: Mutex<Vec<String>>,
    }

    impl FakeSpawner {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl SubAgentSpawner for FakeSpawner {
        async fn spawn(&self, request: SpawnRequest) -> Result<SubAgentHandle> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("spawn:{}", request.prompt));
            Ok(SubAgentHandle::new(
                "agent-1",
                request.preset.clone(),
            ))
        }

        async fn wait(&self, id: &str, timeout: Option<Duration>) -> Result<SubAgentStatus> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("wait:{id}:{}", timeout.is_some()));
            Ok(SubAgentStatus {
                id: id.to_string(),
                state: SubAgentState::Completed,
                output: Some("done".to_string()),
                error: None,
            })
        }

        async fn list(&self) -> Result<Vec<SubAgentStatus>> {
            self.calls.lock().unwrap().push("list".to_string());
            Ok(vec![SubAgentStatus::running("agent-2")])
        }

        async fn send_input(&self, id: &str, input: String) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("send:{id}:{input}"));
            Ok(())
        }

        async fn close(&self, id: &str) -> Result<()> {
            self.calls.lock().unwrap().push(format!("close:{id}"));
            Ok(())
        }
    }

    #[test]
    fn request_builders_round_trip() {
        let request = SpawnRequest::new("do work")
            .with_preset("explorer")
            .with_agent_id("a-1")
            .with_parent_task_id("t-9");
        assert_eq!(request.prompt, "do work");
        assert_eq!(request.preset.as_deref(), Some("explorer"));
        assert_eq!(request.agent_id.as_deref(), Some("a-1"));
        assert_eq!(request.parent_task_id.as_deref(), Some("t-9"));

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["prompt"], "do work");
        assert_eq!(json["preset"], "explorer");
    }

    #[test]
    fn status_terminal_states() {
        assert!(!SubAgentStatus::running("a").is_terminal());
        for state in [SubAgentState::Completed, SubAgentState::Failed] {
            assert!(SubAgentStatus {
                id: "a".to_string(),
                state,
                output: None,
                error: None,
            }
            .is_terminal());
        }
        assert!(SubAgentStatus {
            id: "a".to_string(),
            state: SubAgentState::Closed,
            output: None,
            error: None,
        }
        .is_terminal());
    }

    #[test]
    fn names_and_priorities() {
        let rail = SubagentRail::new();
        assert_eq!(rail.name(), "subagent");
        assert_eq!(rail.priority(), rail_priority::SUBAGENT);
        assert_eq!(rail.priority(), 95);
        assert!(rail.spawner().is_none());
        assert_eq!(format!("{rail:?}"), "SubagentRail { spawner: false }");
    }

    #[tokio::test]
    async fn before_model_call_publishes_the_subagent_section() {
        let rail = SubagentRail::new();
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        rail.before_model_call(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, SectionName::SUBAGENT_TOOLS);
    }

    #[tokio::test]
    async fn the_whole_seam_is_reachable_through_the_trait_object() {
        let fake = Arc::new(FakeSpawner::default());
        let spawner: Arc<dyn SubAgentSpawner> = fake.clone();
        let rail = SubagentRail::new().with_spawner(spawner.clone());
        let seam = rail.spawner().unwrap().clone();
        assert!(Arc::ptr_eq(&seam, &spawner));

        let handle = seam
            .spawn(SpawnRequest::new("dig").with_preset("explorer"))
            .await
            .unwrap();
        assert_eq!(handle.id, "agent-1");
        assert_eq!(handle.preset.as_deref(), Some("explorer"));

        let status = seam
            .wait("agent-1", Some(Duration::from_millis(5)))
            .await
            .unwrap();
        assert_eq!(status.state, SubAgentState::Completed);
        assert!(status.is_terminal());
        assert_eq!(status.output.as_deref(), Some("done"));

        let listed = seam.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert!(!listed[0].is_terminal());

        seam.send_input("agent-1", "more".to_string()).await.unwrap();
        seam.close("agent-1").await.unwrap();

        assert_eq!(
            fake.calls(),
            vec![
                "spawn:dig".to_string(),
                "wait:agent-1:true".to_string(),
                "list".to_string(),
                "send:agent-1:more".to_string(),
                "close:agent-1".to_string(),
            ]
        );
    }
}
