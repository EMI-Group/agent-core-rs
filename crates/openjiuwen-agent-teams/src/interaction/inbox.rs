//! Human-agent inbox: a dumb FIFO for already-parsed human-agent input.
//!
//! Python parity source: `openjiuwen/agent_teams/interaction/human_agent_inbox.py` (v0.1.20).
//! The inbox never parses a body — [`parse_interact_str`](crate::interaction::parse_interact_str)
//! has already turned `@member` / `@all` / `$name` syntax into typed payloads. It only mints an
//! id for each message, keeps the queue in arrival order and refuses to work on a team that has
//! no human-agent channel enabled.
//!
//! Python raises `HumanAgentNotEnabledError` from `_resolve_sender`; the Rust port surfaces the
//! same condition through [`HumanAgentInbox::require_enabled`], which returns
//! [`Error::NotFound`] with the message `human agent is not enabled for this team`. No separate
//! error type is needed because the crate-wide [`Error`] is the only error vocabulary.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_teams::interaction::HumanAgentInbox;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let inbox = HumanAgentInbox::new();
//! assert!(inbox.require_enabled().is_err());
//!
//! inbox.set_enabled(true);
//! let message = inbox.send("on my way", None, Some("alice"));
//! let id = inbox.push(message).await.unwrap();
//! assert!(id.starts_with("msg_"));
//! let drained = inbox.drain().await;
//! assert_eq!(drained.len(), 1);
//! assert_eq!(drained[0].0, id);
//! # }
//! ```
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;

use super::payload::HumanAgentMessage;
use crate::constants::HUMAN_AGENT_MEMBER_NAME;

/// Error message returned when a team has no human-agent channel.
pub const HUMAN_AGENT_NOT_ENABLED: &str = "human agent is not enabled for this team";

/// FIFO holding human-agent input until the avatar's run loop consumes it.
///
/// Cheap to share (`Send + Sync`). The inbox starts **disabled** because team HITL is opt-in
/// (`TeamAgentSpec::enable_hitt` defaults to `false`); a runtime that registered a human-agent
/// member calls [`HumanAgentInbox::set_enabled`] before delivering input.
#[derive(Debug)]
pub struct HumanAgentInbox {
    /// Whether the human-agent channel is enabled for this team.
    enabled: AtomicBool,
    /// Queued `(message id, message)` pairs, oldest first.
    queue: Mutex<VecDeque<(String, HumanAgentMessage)>>,
}

impl Default for HumanAgentInbox {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanAgentInbox {
    /// Builds a disabled, empty inbox.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Enqueues one message under a freshly minted `msg_<uuid hex>` id.
    ///
    /// The inbox does not check enablement here: callers gate with
    /// [`HumanAgentInbox::require_enabled`] before parsing and delivering input, matching
    /// Python's `_dispatch_payload`, which rejects before the inbox sees the payload.
    ///
    /// # Errors
    ///
    /// Reserved for future validation; this implementation never fails.
    // The async signature is the contract (Python's inbox is async and the run loop awaits it);
    // the queue itself is lock-guarded and needs no `.await`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn push(&self, msg: HumanAgentMessage) -> Result<String> {
        let id = format!("msg_{}", uuid_hex());
        self.queue.lock().push_back((id.clone(), msg));
        Ok(id)
    }

    /// Removes every queued message, preserving FIFO order.
    // Async for symmetry with [`HumanAgentInbox::push`]; the queue is drained under a short
    // lock and needs no `.await`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn drain(&self) -> Vec<(String, HumanAgentMessage)> {
        self.queue.lock().drain(..).collect()
    }

    /// Builds the payload for one human-agent message.
    ///
    /// `sender` is the member name of the human agent speaking; when omitted the reserved
    /// `human_agent` name is assumed. `to` selects the route the runtime applies later: `None`
    /// drives the avatar, a broadcast token reaches everyone, and any other value is a
    /// point-to-point target. The inbox stores the typed payload and does not interpret `to`
    /// beyond echoing it.
    #[must_use]
    pub fn send(
        &self,
        body: impl Into<String>,
        to: Option<&str>,
        sender: Option<&str>,
    ) -> HumanAgentMessage {
        HumanAgentMessage {
            human_name: sender.unwrap_or(HUMAN_AGENT_MEMBER_NAME).to_owned(),
            target: to.map(str::to_owned),
            body: body.into(),
            sender: sender.map(str::to_owned),
        }
    }

    /// Whether the human-agent channel is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enables or disables the human-agent channel.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns `Ok(())` when the channel is enabled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] carrying [`HUMAN_AGENT_NOT_ENABLED`] when the team has no
    /// enabled human-agent channel. This is the Rust equivalent of Python's
    /// `HumanAgentNotEnabledError`.
    pub fn require_enabled(&self) -> Result<()> {
        if self.is_enabled() {
            Ok(())
        } else {
            Err(Error::not_found(HUMAN_AGENT_NOT_ENABLED))
        }
    }

    /// Number of queued messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn message(body: &str) -> HumanAgentMessage {
        HumanAgentMessage {
            human_name: HUMAN_AGENT_MEMBER_NAME.to_owned(),
            target: None,
            body: body.to_owned(),
            sender: None,
        }
    }

    #[tokio::test]
    async fn push_assigns_msg_ids_and_drain_is_fifo() {
        let inbox = HumanAgentInbox::new();
        let first = inbox.push(message("first")).await.unwrap();
        let second = inbox.push(message("second")).await.unwrap();

        assert!(first.starts_with("msg_"));
        assert!(second.starts_with("msg_"));
        assert_ne!(first, second);
        assert_eq!(first.len(), "msg_".len() + 32);
        assert_eq!(inbox.len(), 2);

        let drained = inbox.drain().await;
        assert_eq!(
            drained,
            vec![(first, message("first")), (second, message("second"))]
        );
        assert!(inbox.is_empty());
        assert_eq!(inbox.drain().await.len(), 0);
    }

    #[tokio::test]
    async fn send_resolves_the_default_sender_name() {
        let inbox = HumanAgentInbox::new();

        let default = inbox.send("hello", None, None);
        assert_eq!(default.human_name, HUMAN_AGENT_MEMBER_NAME);
        assert_eq!(default.sender, None);
        assert_eq!(default.target, None);

        let explicit = inbox.send("hello", Some("*"), Some("alice"));
        assert_eq!(explicit.human_name, "alice");
        assert_eq!(explicit.sender.as_deref(), Some("alice"));
        assert_eq!(explicit.target.as_deref(), Some("*"));
        assert_eq!(explicit.body, "hello");
    }

    #[test]
    fn inbox_is_disabled_until_enabled() {
        let inbox = HumanAgentInbox::new();
        assert!(!inbox.is_enabled());

        let error = inbox.require_enabled().unwrap_err();
        assert!(matches!(error, Error::NotFound(_)));
        assert_eq!(
            error.to_string(),
            format!("not_found: {HUMAN_AGENT_NOT_ENABLED}")
        );

        inbox.set_enabled(true);
        assert!(inbox.is_enabled());
        assert!(inbox.require_enabled().is_ok());

        inbox.set_enabled(false);
        assert!(matches!(inbox.require_enabled(), Err(Error::NotFound(_))));
    }

    #[test]
    fn default_inbox_is_disabled() {
        assert!(!HumanAgentInbox::default().is_enabled());
    }
}
