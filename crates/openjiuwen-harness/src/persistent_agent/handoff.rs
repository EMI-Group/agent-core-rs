//! Handoff: transfer a persistent agent's state to a successor execution.
//!
//! A finished or paused execution hands its successor the three things the successor cannot
//! rebuild on its own — the persisted session record, the cursor it resumes the journal from and
//! the model window it starts from — plus the successor's next goal and the files the ending
//! listed for sharing. [`HandoffTransfer`] carries that batch and
//! [`HandoffTransfer::validate`] proves the parts belong together before a transport sees them.
//!
//! [`HandoffTarget`] is the seam the host implements. [`LocalHandoff`] parks the transfer in
//! process and lets the successor consume it exactly once; [`RemoteHandoff`] and
//! [`DistributedHandoff`] are deliberate typed refusals ([`HandoffTransport::reason`]) instead of
//! silent no-ops. [`hand_off`] is the validating entry point a caller uses.
//!
//! The module never calls a model, never runs the agent loop and never owns host queueing or
//! delivery: it validates, records and hands over.
//!
//! See `CONTEXT.md` in the parent directory for the API contract.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::identity::{ExecutionIdentity, SessionBinding};
use super::session::{ContinuationCursor, PersistentSessionRecord};
use super::share_file::SharedFileEntry;
use super::util::is_valid_identifier;
use super::window::WindowAssembly;
use super::Clock;

/// The transport a [`HandoffTarget`] receives transfers over.
///
/// Only [`HandoffTransport::InProcess`] is implemented in this build; the other two name what a
/// distributed deployment would need and are refused by [`HandoffTransport::reason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffTransport {
    /// The successor runs in the same process as the predecessor.
    InProcess,
    /// The successor runs behind a remote endpoint.
    Remote,
    /// The successor runs on another node of a distributed deployment.
    Distributed,
}

impl HandoffTransport {
    /// Returns the persisted snake-case spelling of the transport.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProcess => "in_process",
            Self::Remote => "remote",
            Self::Distributed => "distributed",
        }
    }

    /// Returns why this transport cannot accept a transfer in this build.
    ///
    /// [`HandoffTransport::InProcess`] is implemented, so its reason is empty; the other two
    /// return the exact message their `accept` reports as [`Error::Unsupported`].
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::InProcess => "",
            Self::Remote => {
                "persistent agent handoff over a remote transport is not implemented; use an \
                 in-process HandoffTarget"
            }
            Self::Distributed => {
                "persistent agent handoff over a distributed transport is not implemented; use \
                 an in-process HandoffTarget"
            }
        }
    }
}

/// The state one execution hands to its successor.
///
/// The transfer is self-describing data only: the source execution, the target session, the
/// persisted record, the continuation cursor, the window the successor starts from, the
/// successor's next goal and the files the ending listed for sharing. [`Self::validate`] proves
/// the parts belong together; nothing here reads a file or calls a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffTransfer {
    /// Execution that hands the state over.
    pub from: ExecutionIdentity,
    /// Session the successor continues.
    pub to: SessionBinding,
    /// Persisted session record the successor loads.
    pub record: PersistentSessionRecord,
    /// Where the successor picks up in the journal.
    pub cursor: ContinuationCursor,
    /// Model window the successor starts from.
    pub window: WindowAssembly,
    /// Goal the successor is asked to pursue.
    pub next_goal: String,
    /// Files the predecessor listed for sharing, in listing order.
    pub shared_files: Vec<SharedFileEntry>,
}

impl HandoffTransfer {
    /// Validates that the parts of the transfer belong together.
    ///
    /// The source execution and the target binding must name the same `(agent, session)`, the
    /// record must belong to the target session, a continuation cursor's recovery reference must
    /// be an identifier, and every shared file must be a usable listing (see
    /// [`SharedFileEntry::validate_input`]).
    ///
    /// # Errors
    /// `Error::Validation` naming the first rule that fails; a shared-file rejection is the
    /// listing's own error, unchanged.
    pub fn validate(&self) -> Result<()> {
        if self.from.session_binding() != self.to {
            return Err(Error::validation(
                "handoff source execution and target binding must name the same agent and session",
            ));
        }
        if self.record.agent_id != self.to.agent_id || self.record.session_id != self.to.session_id
        {
            return Err(Error::validation(
                "handoff record must belong to the target session",
            ));
        }
        if let Some(recovery_ref) = self.cursor.recovery_ref.as_deref() {
            if !is_valid_identifier(recovery_ref) {
                return Err(Error::validation(
                    "handoff continuation cursor recovery reference must be an identifier",
                ));
            }
        }
        for file in &self.shared_files {
            file.validate_input()?;
        }
        Ok(())
    }
}

/// What a successor holds after accepting a transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffReceipt {
    /// Session the successor continues.
    pub successor_session: SessionBinding,
    /// Number of messages the successor took over from the transferred record.
    pub accepted_entries: usize,
    /// RFC 3339 timestamp of the acceptance, read from the target's injected clock.
    pub transferred_at: String,
    /// Execution id of the successor, when the target already assigned one.
    pub successor_execution_id: Option<String>,
}

/// The successor side of a handoff: accepts the state another execution hands over.
#[async_trait]
pub trait HandoffTarget: Send + Sync {
    /// Returns the transport this target receives transfers over.
    fn transport(&self) -> HandoffTransport;

    /// Accepts `transfer` and returns the receipt describing what the successor now holds.
    ///
    /// # Errors
    /// `Error::Validation` when the transfer is inconsistent, and [`Error::Unsupported`] when the
    /// transport is not implemented in this build.
    async fn accept(&self, transfer: HandoffTransfer) -> Result<HandoffReceipt>;
}

/// The in-process [`HandoffTarget`]: parks the transfer until the successor takes it.
///
/// The successor consumes the parked state exactly once with [`Self::take`], so a transfer is
/// never applied twice; a newer accepted transfer replaces a parked one that was never taken.
#[derive(Debug)]
pub struct LocalHandoff {
    /// Clock the acceptance timestamp is read from.
    pub clock: Arc<dyn Clock>,
    /// The transfer the successor has not taken yet.
    pub slot: Mutex<Option<HandoffTransfer>>,
}

impl LocalHandoff {
    /// Creates an empty in-process target reading its timestamps from `clock`.
    #[must_use]
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            slot: Mutex::new(None),
        }
    }

    /// Takes the parked transfer, leaving the slot empty.
    ///
    /// Returns `None` when nothing is parked or the successor already took the state.
    #[must_use]
    pub fn take(&self) -> Option<HandoffTransfer> {
        self.slot.lock().take()
    }
}

#[async_trait]
impl HandoffTarget for LocalHandoff {
    fn transport(&self) -> HandoffTransport {
        HandoffTransport::InProcess
    }

    async fn accept(&self, transfer: HandoffTransfer) -> Result<HandoffReceipt> {
        transfer.validate()?;
        let receipt = HandoffReceipt {
            successor_session: transfer.to.clone(),
            accepted_entries: transfer.record.messages.len(),
            transferred_at: self.clock.now_rfc3339(),
            successor_execution_id: None,
        };
        *self.slot.lock() = Some(transfer);
        Ok(receipt)
    }
}

/// The remote [`HandoffTarget`]: a deliberate refusal in this build.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemoteHandoff;

#[async_trait]
impl HandoffTarget for RemoteHandoff {
    fn transport(&self) -> HandoffTransport {
        HandoffTransport::Remote
    }

    async fn accept(&self, _transfer: HandoffTransfer) -> Result<HandoffReceipt> {
        Err(Error::unsupported(HandoffTransport::Remote.reason()))
    }
}

/// The distributed [`HandoffTarget`]: a deliberate refusal in this build.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DistributedHandoff;

#[async_trait]
impl HandoffTarget for DistributedHandoff {
    fn transport(&self) -> HandoffTransport {
        HandoffTransport::Distributed
    }

    async fn accept(&self, _transfer: HandoffTransfer) -> Result<HandoffReceipt> {
        Err(Error::unsupported(HandoffTransport::Distributed.reason()))
    }
}

/// Validates `transfer` and hands it to `target`.
///
/// The transfer is validated here so no transport has to repeat the consistency rules, and a
/// target that refuses (for example [`RemoteHandoff`]) therefore never sees an inconsistent batch.
///
/// # Errors
/// `Error::Validation` when `transfer` is inconsistent (see [`HandoffTransfer::validate`]) and
/// whatever `target.accept` returns, in particular [`Error::Unsupported`] for a transport this
/// build does not implement.
pub async fn hand_off(
    target: &dyn HandoffTarget,
    transfer: HandoffTransfer,
) -> Result<HandoffReceipt> {
    transfer.validate()?;
    target.accept(transfer).await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::prompt::message::BaseMessage;

    use super::{
        hand_off, DistributedHandoff, HandoffReceipt, HandoffTarget, HandoffTransfer,
        HandoffTransport, LocalHandoff, RemoteHandoff,
    };
    use crate::persistent_agent::identity::{ExecutionIdentity, SessionBinding};
    use crate::persistent_agent::session::{
        ContinuationCursor, PersistentSessionRecord, RecordEnding,
    };
    use crate::persistent_agent::share_file::{ShareMode, SharedFileEntry};
    use crate::persistent_agent::window::{Capacity, WindowAssembler, WindowAssembly};
    use crate::persistent_agent::{Clock, ManualClock};

    /// Frozen instant the injected clock reports: `2023-11-14T22:13:20.000Z`.
    const NOW_MS: i64 = 1_700_000_000_000;

    /// The binding every fixture transfer targets.
    fn binding() -> SessionBinding {
        SessionBinding::new("agent", "session").unwrap()
    }

    /// A different, well-formed binding, used to break a fixture on purpose.
    fn other_binding() -> SessionBinding {
        SessionBinding::new("agent", "other").unwrap()
    }

    /// The source execution of every fixture transfer.
    fn identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
    }

    /// A resumable record of the target session holding `messages` messages.
    fn record(messages: usize) -> PersistentSessionRecord {
        let mut record = PersistentSessionRecord::new(&binding())
            .unwrap()
            .with_ending(RecordEnding {
                outcome: "completed".to_string(),
                request_id: "request".to_string(),
                execution_id: "execution".to_string(),
                spec_fingerprint: None,
                reply_target: None,
                output_refs: Vec::new(),
            })
            .unwrap();
        record.messages = (0..messages)
            .map(|index| BaseMessage::user(format!("message {index}")))
            .collect();
        record
    }

    /// The empty window the fixture transfer hands over.
    fn window() -> WindowAssembly {
        WindowAssembler::new(Capacity::default()).assemble(&[])
    }

    /// A valid transfer of a three-message record.
    fn transfer() -> HandoffTransfer {
        HandoffTransfer {
            from: identity(),
            to: binding(),
            record: record(3),
            cursor: ContinuationCursor::default(),
            window: window(),
            next_goal: "summarise the run".to_string(),
            shared_files: Vec::new(),
        }
    }

    /// A manual clock frozen at [`NOW_MS`].
    fn clock() -> Arc<dyn Clock> {
        Arc::new(ManualClock::new(NOW_MS))
    }

    #[tokio::test]
    async fn local_handoff_parks_the_transfer_until_the_successor_takes_it() {
        let target = LocalHandoff::new(clock());
        let expected = transfer();
        let receipt = target.accept(expected.clone()).await.unwrap();
        assert_eq!(target.transport(), HandoffTransport::InProcess);
        assert_eq!(receipt.successor_session, binding());
        assert_eq!(receipt.transferred_at, "2023-11-14T22:13:20.000Z");
        assert_eq!(receipt.successor_execution_id, None);
        assert_eq!(target.take().as_ref(), Some(&expected));
        assert_eq!(target.take(), None);
    }

    #[tokio::test]
    async fn local_handoff_receipt_counts_the_transferred_messages() {
        let target = LocalHandoff::new(clock());
        assert_eq!(target.accept(transfer()).await.unwrap().accepted_entries, 3);

        let mut empty = transfer();
        empty.record.messages.clear();
        assert_eq!(target.accept(empty).await.unwrap().accepted_entries, 0);
    }

    #[tokio::test]
    async fn a_newer_transfer_replaces_the_parked_one() {
        let target = LocalHandoff::new(clock());
        target.accept(transfer()).await.unwrap();
        let mut newer = transfer();
        newer.next_goal = "ship the report".to_string();
        target.accept(newer.clone()).await.unwrap();
        assert_eq!(target.take(), Some(newer));
        assert_eq!(target.take(), None);
    }

    #[tokio::test]
    async fn hand_off_validates_before_the_target_sees_the_transfer() {
        let local = LocalHandoff::new(clock());
        let invalid = HandoffTransfer {
            to: other_binding(),
            ..transfer()
        };
        let error = hand_off(&local, invalid).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(local.take(), None);

        let receipt = hand_off(&local, transfer()).await.unwrap();
        assert_eq!(receipt.accepted_entries, 3);
        assert_eq!(local.take(), Some(transfer()));
    }

    #[test]
    fn validation_rejects_a_target_binding_mismatch() {
        let mismatched = HandoffTransfer {
            to: other_binding(),
            ..transfer()
        };
        let error = mismatched.validate().unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("must name the same agent and session"));
    }

    #[test]
    fn validation_rejects_a_record_from_another_session() {
        let mut mismatched = transfer();
        mismatched.record.session_id = "other".to_string();
        let error = mismatched.validate().unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error
            .to_string()
            .contains("handoff record must belong to the target session"));

        let mut mismatched = transfer();
        mismatched.record.agent_id = "other".to_string();
        assert!(mismatched
            .validate()
            .unwrap_err()
            .to_string()
            .contains("handoff record must belong to the target session"));
    }

    #[test]
    fn validation_checks_the_recovery_reference() {
        let mut continued = transfer();
        continued.cursor.recovery_ref = Some("recovery/1".to_string());
        continued.validate().unwrap();

        for unusable in ["", "not an identifier", "line\nbreak"] {
            let mut broken = transfer();
            broken.cursor.recovery_ref = Some(unusable.to_string());
            let error = broken.validate().unwrap_err();
            assert!(matches!(error, Error::Validation(_)));
            assert!(error
                .to_string()
                .contains("recovery reference must be an identifier"));
        }
    }

    #[test]
    fn validation_rejects_an_unusable_shared_file() {
        let mut shared = transfer();
        shared.shared_files = vec![SharedFileEntry {
            path: "../escape/report.md".to_string(),
            share_mode: ShareMode::Upload,
            caption: None,
            sha256: None,
            size: None,
        }];
        let error = shared.validate().unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("logical relative path"));

        let mut blank_caption = transfer();
        blank_caption.shared_files = vec![SharedFileEntry {
            path: "out/report.md".to_string(),
            share_mode: ShareMode::Attachment,
            caption: Some("   ".to_string()),
            sha256: None,
            size: None,
        }];
        assert!(blank_caption
            .validate()
            .unwrap_err()
            .to_string()
            .contains("caption"));

        let mut listed = transfer();
        listed.shared_files = vec![SharedFileEntry {
            path: "out/report.md".to_string(),
            share_mode: ShareMode::Attachment,
            caption: Some("the finished report".to_string()),
            sha256: None,
            size: None,
        }];
        listed.validate().unwrap();
    }

    #[tokio::test]
    async fn remote_and_distributed_transports_are_typed_refusals() {
        let remote = RemoteHandoff;
        assert_eq!(remote.transport(), HandoffTransport::Remote);
        match remote.accept(transfer()).await.unwrap_err() {
            Error::Unsupported(reason) => assert_eq!(
                reason,
                "persistent agent handoff over a remote transport is not implemented; use an \
                 in-process HandoffTarget"
            ),
            other => panic!("unexpected error: {other}"),
        }

        let distributed = DistributedHandoff;
        assert_eq!(distributed.transport(), HandoffTransport::Distributed);
        match hand_off(&distributed, transfer()).await.unwrap_err() {
            Error::Unsupported(reason) => assert_eq!(
                reason,
                "persistent agent handoff over a distributed transport is not implemented; use \
                 an in-process HandoffTarget"
            ),
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(
            Error::unsupported(distributed.transport().reason()).to_string(),
            "unsupported: persistent agent handoff over a distributed transport is not \
             implemented; use an in-process HandoffTarget"
        );
    }

    #[test]
    fn transports_use_snake_case_and_explain_themselves() {
        assert_eq!(
            serde_json::to_string(&HandoffTransport::InProcess).unwrap(),
            "\"in_process\""
        );
        assert_eq!(
            serde_json::to_string(&HandoffTransport::Remote).unwrap(),
            "\"remote\""
        );
        assert_eq!(
            serde_json::to_string(&HandoffTransport::Distributed).unwrap(),
            "\"distributed\""
        );
        assert_eq!(
            serde_json::from_str::<HandoffTransport>("\"distributed\"").unwrap(),
            HandoffTransport::Distributed
        );
        assert_eq!(HandoffTransport::InProcess.as_str(), "in_process");
        assert_eq!(HandoffTransport::Remote.as_str(), "remote");
        assert_eq!(HandoffTransport::Distributed.as_str(), "distributed");
        assert_eq!(HandoffTransport::InProcess.reason(), "");
        assert!(HandoffTransport::Remote
            .reason()
            .starts_with("persistent agent handoff over a remote transport"));
        assert!(HandoffTransport::Distributed
            .reason()
            .contains("use an in-process HandoffTarget"));
        let copied = HandoffTransport::Remote;
        assert_eq!(copied, HandoffTransport::Remote);
    }

    #[test]
    fn transfers_and_receipts_round_trip_through_json() {
        let mut shared = transfer();
        shared.shared_files = vec![SharedFileEntry {
            path: "out/report.md".to_string(),
            share_mode: ShareMode::Attachment,
            caption: Some("the finished report".to_string()),
            sha256: None,
            size: None,
        }];
        let decoded: HandoffTransfer =
            serde_json::from_str(&serde_json::to_string(&shared).unwrap()).unwrap();
        assert_eq!(decoded, shared);
        assert_eq!(decoded.window, window());
        assert_eq!(decoded.cursor, ContinuationCursor::default());

        let receipt = HandoffReceipt {
            successor_session: binding(),
            accepted_entries: 7,
            transferred_at: "2023-11-14T22:13:20.000Z".to_string(),
            successor_execution_id: Some("next-execution".to_string()),
        };
        let encoded = serde_json::to_value(&receipt).unwrap();
        assert_eq!(encoded["accepted_entries"], serde_json::json!(7));
        assert_eq!(encoded["successor_session"]["agent_id"], "agent");
        assert_eq!(
            serde_json::from_value::<HandoffReceipt>(encoded).unwrap(),
            receipt
        );
    }
}
