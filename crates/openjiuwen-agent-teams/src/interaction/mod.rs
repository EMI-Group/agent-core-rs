//! External interaction layer: typed payloads, routing grammar, the run gate, human-agent
//! inboxes and the tool-approval bridge.
//!
//! Python parity source: `openjiuwen/agent_teams/interaction/` plus
//! `openjiuwen/agent_teams/runtime/gate.py` and `openjiuwen/agent_teams/rails/tool_approval_rail.py`
//! (v0.1.20). This module is the single seam between an external caller and the running team:
//!
//! | Concern | Entry point |
//! | --- | --- |
//! | Typed input from every channel | [`InteractPayload`] |
//! | Free-form text → typed route | [`parse_interact_str`] |
//! | Roster resolution of `@` mentions | [`resolve_targets`] |
//! | Admission control for one run cycle | [`InteractGate`] |
//! | Human-agent input queue | [`HumanAgentInbox`] |
//! | Leader decision on an interrupted tool call | [`ToolApprovalRequest`], [`parse_confirm`] |
//! | Uniform delivery outcome | [`DeliverResult`] |
//!
//! The runtime owns parsing-and-dispatch; nothing below this layer re-parses a body. Routing is
//! pure syntax plus roster membership, so it is independently testable and needs no session.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_teams::interaction::{
//!     parse_interact_str, DeliverResult, GodViewMessage, InteractGate, InteractPayload, Route,
//! };
//!
//! let roster = vec!["dev-1".to_owned()];
//! let route = parse_interact_str("# status?", &roster);
//! assert_eq!(route, Route::GodView { body: "status?".to_owned() });
//!
//! let gate = InteractGate::new();
//! let payload = InteractPayload::GodView(GodViewMessage {
//!     body: "status?".to_owned(),
//!     sender: None,
//! });
//! assert!(gate.dispatch(payload, true).ok);
//! assert_eq!(
//!     gate.dispatch(InteractPayload::GodView(GodViewMessage::default()), true).message_id,
//!     None
//! );
//! assert_eq!(DeliverResult::not_active().reason.as_deref(), Some("not_active"));
//! ```

pub mod approval;
pub mod gate;
pub mod inbox;
pub mod payload;
pub mod router;

pub use approval::{approval_result_event, parse_confirm, ConfirmPayload, ToolApprovalRequest};
pub use gate::InteractGate;
pub use inbox::{HumanAgentInbox, HUMAN_AGENT_NOT_ENABLED};
pub use payload::{
    DeliverResult, ExternalTeamEvent, GodViewMessage, GroupChatMessage, HumanAgentMessage,
    HumanAgentToolCall, InteractPayload, OperatorMessage,
};
pub use router::{parse_interact_str, resolve_targets, Route};
