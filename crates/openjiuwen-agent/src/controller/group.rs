//! Group routing seam: point-to-point sends and subscription publishes.
//!
//! Ports Python's `BaseController.set_group`/`send_to_agent`/`publish` from
//! `openjiuwen/core/controller/legacy/controller.py`. A controller whose agent
//! has been added to an agent group holds a reference to the group's controller
//! and forwards sends and publishes to it; a controller that is not part of a
//! group therefore fails with a typed configuration error instead of routing
//! nowhere.
//!
//! This module owns only the seam and the typed errors: [`GroupRouter`] is the
//! handle a group controller implements (or that a host adapts onto its own
//! group type), and [`send_without_group`]/[`publish_without_group`] build the
//! Python-shaped failures. The controller surfaces themselves
//! (`BaseController::set_group`, `clear_group`, `group`, `send_to_agent`,
//! `publish`) live in [`mod@super::base`] and delegate here, so the routing
//! behaviour of a group is never re-implemented in this file.
use super::schema::Event;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;

/// Tail of every "not part of a group" message (Python's `RuntimeError` text).
pub const NOT_IN_GROUP: &str = "Agent is not part of a group with a controller.";

/// Routes events through the group a controller's agent belongs to.
///
/// Implementations are shared as `Arc<dyn GroupRouter>`; the controller keeps
/// one (see [`require_group`]) and forwards [`send_to_agent`](Self::send_to_agent)
/// and [`publish`](Self::publish) to it. A router is group-wide, so it sees the
/// caller's [`Event`] and session unchanged and reports results in the caller's
/// shape: one value for a point-to-point send, one value per subscriber for a
/// publish.
#[async_trait]
pub trait GroupRouter: Send + Sync {
    /// Returns the group id, used in diagnostics; `None` when the router is
    /// anonymous.
    fn group_id(&self) -> Option<&str> {
        None
    }

    /// Sends `event` to `agent_id` and returns that agent's result.
    ///
    /// # Errors
    ///
    /// Returns an error when the target agent is unknown to the group or when
    /// the routed turn fails.
    async fn send_to_agent(
        &self,
        agent_id: &str,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Value>;

    /// Publishes `event` to every subscriber of the group and returns their
    /// results in subscription order.
    ///
    /// # Errors
    ///
    /// Returns an error when the group cannot dispatch the publish (no
    /// subscribers, a closed transport); a failing subscriber is reported by the
    /// implementation, either as a value or as an error.
    async fn publish(&self, event: &Event, session: Option<Arc<dyn Session>>)
        -> Result<Vec<Value>>;
}

/// The controller method that needs a group router.
///
/// Carried into [`require_group`] so the failure names the method the caller
/// tried to use, matching Python's messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupMethod<'a> {
    /// `send_to_agent(agent_id, ..)`, carrying the target agent id.
    SendToAgent(&'a str),
    /// `publish(..)`.
    Publish,
}

impl<'a> GroupMethod<'a> {
    /// Returns the Python method name, without its arguments.
    #[must_use]
    pub const fn method_name(self) -> &'static str {
        match self {
            Self::SendToAgent(_) => "send_to_agent",
            Self::Publish => "publish",
        }
    }

    /// Returns the target agent id of [`GroupMethod::SendToAgent`], `None` for
    /// [`GroupMethod::Publish`].
    #[must_use]
    pub const fn agent_id(self) -> Option<&'a str> {
        match self {
            Self::SendToAgent(agent_id) => Some(agent_id),
            Self::Publish => None,
        }
    }
}

impl fmt::Display for GroupMethod<'_> {
    /// Renders the call shape, e.g. `send_to_agent('a1')` or `publish()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SendToAgent(agent_id) => write!(f, "send_to_agent('{agent_id}')"),
            Self::Publish => f.write_str("publish()"),
        }
    }
}

/// Builds the typed error a controller raises when `method` is unavailable.
///
/// `owner` is the controller or agent name Python interpolates
/// (`self.__class__.__name__`); the message keeps Python's shape, e.g.
/// `"ctl: Cannot send_to_agent('a1'). Agent is not part of a group with a
/// controller."`.
pub fn without_group_error(owner: &str, method: GroupMethod<'_>) -> Error {
    Error::config(format!("{owner}: Cannot {method}. {NOT_IN_GROUP}"))
}

/// Builds the error of a `send_to_agent` attempted outside a group.
pub fn send_without_group(owner: &str, agent_id: &str) -> Error {
    without_group_error(owner, GroupMethod::SendToAgent(agent_id))
}

/// Builds the error of a `publish` attempted outside a group.
pub fn publish_without_group(owner: &str) -> Error {
    without_group_error(owner, GroupMethod::Publish)
}

/// Returns the group router or the typed "not part of a group" error.
///
/// # Errors
///
/// Returns [`Error::config`] (via [`without_group_error`]) when no router is
/// installed, so a controller can write
/// `let router = require_group(self.group.as_ref(), owner, GroupMethod::Publish)?;`
/// and keep one error shape for both routing methods.
pub fn require_group<'a>(
    group: Option<&'a Arc<dyn GroupRouter>>,
    owner: &str,
    method: GroupMethod<'_>,
) -> Result<&'a Arc<dyn GroupRouter>> {
    group.ok_or_else(|| without_group_error(owner, method))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::sessions::AgentSession;
    use parking_lot::Mutex;
    use serde_json::json;

    /// Records every forwarded call and echoes the agent id back.
    #[derive(Default)]
    struct RecordingRouter {
        group_id: Option<&'static str>,
        sent: Mutex<Vec<(String, String, Option<String>)>>,
        published: Mutex<Vec<(String, Option<String>)>>,
    }

    #[async_trait]
    impl GroupRouter for RecordingRouter {
        fn group_id(&self) -> Option<&str> {
            self.group_id
        }

        async fn send_to_agent(
            &self,
            agent_id: &str,
            event: &Event,
            session: Option<Arc<dyn Session>>,
        ) -> Result<Value> {
            self.sent.lock().push((
                agent_id.to_owned(),
                event.event_id.clone(),
                session.as_deref().map(|session| session.id().to_owned()),
            ));
            Ok(json!({"target": agent_id, "event": event.event_id}))
        }

        async fn publish(
            &self,
            event: &Event,
            session: Option<Arc<dyn Session>>,
        ) -> Result<Vec<Value>> {
            self.published.lock().push((
                event.event_id.clone(),
                session.as_deref().map(|session| session.id().to_owned()),
            ));
            Ok(vec![json!({"subscriber": 1}), json!({"subscriber": 2})])
        }
    }

    #[tokio::test]
    async fn forwarding_passes_the_event_and_session_unchanged() {
        let recorder = Arc::new(RecordingRouter {
            group_id: Some("group-1"),
            ..RecordingRouter::default()
        });
        let router: Arc<dyn GroupRouter> = recorder.clone();
        assert_eq!(router.group_id(), Some("group-1"));

        let event = Event::user_input("hello");
        let session: Option<Arc<dyn Session>> = Some(Arc::new(AgentSession::new("s1")));
        let value = router
            .send_to_agent("agent-1", &event, session.clone())
            .await
            .expect("stub router always succeeds");
        assert_eq!(value, json!({"target": "agent-1", "event": event.event_id}));

        let values = router
            .publish(&event, session)
            .await
            .expect("stub router always succeeds");
        assert_eq!(values.len(), 2);

        assert_eq!(
            recorder.sent.lock().as_slice(),
            [(
                "agent-1".to_owned(),
                event.event_id.clone(),
                Some("s1".to_owned())
            )]
        );
        assert_eq!(
            recorder.published.lock().as_slice(),
            [(event.event_id.clone(), Some("s1".to_owned()))]
        );
    }

    #[tokio::test]
    async fn anonymous_router_reports_no_group_id() {
        let router = RecordingRouter::default();
        assert_eq!(router.group_id(), None);
        let event = Event::user_input("hi");
        let values = router
            .publish(&event, None)
            .await
            .expect("stub router always succeeds");
        assert_eq!(values.len(), 2);
        assert_eq!(
            router.published.lock().as_slice(),
            [(event.event_id.clone(), None)]
        );
    }

    #[tokio::test]
    async fn require_group_returns_the_installed_router() {
        let router: Arc<dyn GroupRouter> = Arc::new(RecordingRouter::default());
        let installed = Some(&router);
        let resolved =
            require_group(installed, "ctl", GroupMethod::Publish).expect("a router is installed");
        assert!(Arc::ptr_eq(resolved, &router));
    }

    #[test]
    fn require_group_without_a_router_is_a_config_error() {
        let missing: Option<&Arc<dyn GroupRouter>> = None;
        let Err(error) = require_group(missing, "ctl", GroupMethod::SendToAgent("a1")) else {
            panic!("no router is installed");
        };
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "ctl: Cannot send_to_agent('a1'). Agent is not part of a group with a controller."
        );

        let Err(error) = require_group(missing, "ctl", GroupMethod::Publish) else {
            panic!("no router is installed");
        };
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "ctl: Cannot publish(). Agent is not part of a group with a controller."
        );
    }

    #[test]
    fn error_helpers_match_the_python_messages() {
        let send = send_without_group("MyController", "a1");
        assert_eq!(send.kind(), ErrorKind::Config);
        let text = send.message().into_owned();
        assert!(text.contains("MyController"));
        assert!(text.contains("send_to_agent"));
        assert!(text.contains("'a1'"));
        assert!(text.ends_with(NOT_IN_GROUP));

        let publish = publish_without_group("MyController");
        assert_eq!(publish.kind(), ErrorKind::Config);
        let text = publish.message().into_owned();
        assert!(text.contains("MyController"));
        assert!(text.contains("publish()"));
        assert!(text.ends_with(NOT_IN_GROUP));
    }

    #[test]
    fn group_method_reports_its_shape() {
        let send = GroupMethod::SendToAgent("a1");
        assert_eq!(send.method_name(), "send_to_agent");
        assert_eq!(send.agent_id(), Some("a1"));
        assert_eq!(send.to_string(), "send_to_agent('a1')");

        let publish = GroupMethod::Publish;
        assert_eq!(publish.method_name(), "publish");
        assert_eq!(publish.agent_id(), None);
        assert_eq!(publish.to_string(), "publish()");
    }
}
