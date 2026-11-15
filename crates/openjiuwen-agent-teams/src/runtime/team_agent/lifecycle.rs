//! Team lifecycle and run entry points: start, stop, invoke, stream, delivery and snapshots.
//!
//! The coordination loop itself lives in `super::super::leader`; this module only owns the
//! handle-side lifecycle of one team and the routing of external payloads into it.
use std::sync::atomic::Ordering;
use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_agent::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_session::Session;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use super::super::initial_input::parse_initial_input;
use super::super::leader;
use super::super::memory_bind;
use super::super::tool_dispatch;
use super::TeamAgent;
use crate::constants::USER_PSEUDO_MEMBER_NAME;
use crate::interaction::{DeliverResult, HumanAgentMessage, HumanAgentToolCall, InteractPayload};
use crate::member::MemberSpawner;
use crate::schema::{MemberLifetime, TeamRole, TeamSnapshot};

impl TeamAgent {
    /// Starts the team: the transport, the leader runtime and the interaction gate.
    ///
    /// Idempotent: a second call while started is a no-op. A failed start resets the started flag
    /// and propagates the error, so a caller may retry.
    ///
    /// # Errors
    ///
    /// Propagates transport start failures and the leader runtime's own start error (missing
    /// model, bad checkpoint). Starting a team whose leader row is already terminal fails the
    /// member lifecycle (`member is not in unstarted state`).
    pub async fn start(&self) -> Result<()> {
        if self.inner.started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        if let Err(error) = self.run_start().await {
            self.inner.started.store(false, Ordering::SeqCst);
            return Err(error);
        }
        Ok(())
    }

    /// Performs the actual start work, in order, so [`Self::start`] can roll the flag back.
    async fn run_start(&self) -> Result<()> {
        self.inner.backend.messager().start().await?;
        // Team memory is advisory for the run: the session is installed before the leader takes its
        // first turn, and a rejected session is logged instead of failing the start.
        memory_bind::session_started(
            self.inner.backend.memory().as_ref(),
            self.team_name(),
            self.session_id(),
        )
        .await;
        self.inner.member.start(None).await?;
        self.start_attach().await?;
        self.inner.signals.team_started(Some(self.session_id()));
        debug!(
            team = self.team_name(),
            member = self.member_name(),
            "team agent started"
        );
        Ok(())
    }

    /// Stops the team: cancels the run, stops the leader runtime and closes the gate.
    ///
    /// Idempotent and never fails: teardown problems are logged so a caller's cleanup path cannot
    /// be blocked by them. The member's row settles on the `shutdown` retire target, and every
    /// workspace still retained by the team (the leader's plus every remaining member's, including
    /// rows already shut down) is released exactly once.
    /// Team memory is closed out last: every still-live member gets its handoff note, then the
    /// buffered turns are flushed and `TEAM_MEMORY.md` is exported when the configuration exports on
    /// stop. Each of those is best-effort, so a memory failure never keeps the harness alive.
    ///
    /// # Errors
    ///
    /// This implementation returns `Ok(())` even when teardown partially fails.
    pub async fn stop(&self) -> Result<()> {
        self.inner.started.store(false, Ordering::SeqCst);
        self.inner.cancel.lock().cancel();
        let attach_host = self.inner.attach_host.lock().take();
        if let Some(host) = attach_host {
            host.shutdown().await;
        }
        self.leave_live_members().await;
        if let Some(runtime) = self.inner.member.runtime() {
            if let Err(error) = runtime.stop().await {
                warn!(team = self.team_name(), error = %error, "failed to stop the leader runtime");
            }
        }
        if let Err(error) = self
            .inner
            .member
            .shutdown(MemberLifetime::Shutdown, true)
            .await
        {
            warn!(team = self.team_name(), error = %error, "failed to settle the leader row");
        }
        if let Err(error) = self.inner.backend.messager().stop().await {
            warn!(team = self.team_name(), error = %error, "failed to stop the team transport");
        }
        self.inner.gate.close();
        if let Err(error) = self.inner.spawner.despawn(self.member_name()).await {
            debug!(team = self.team_name(), error = %error, "leader has no child process to reap");
        }
        let released = self.inner.kv_cache.release_team(self.session_id()).await;
        debug!(
            team = self.team_name(),
            released, "kv-cache affinity entries released"
        );
        self.release_all_workspaces().await;
        memory_bind::finish(self.inner.backend.memory().as_ref(), self.team_name()).await;
        self.inner.signals.team_finished();
        debug!(team = self.team_name(), "team agent stopped");
        Ok(())
    }

    /// Runs the team for one caller request and returns the leader's answer.
    ///
    /// The request is parsed once ([`parse_initial_input`]): routable `$`/`@` channels become
    /// deliveries that never reach the leader, everything else becomes the leader query.
    ///
    /// # Errors
    ///
    /// Propagates start failures, leader runtime failures and store failures from the run loop.
    pub async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let input = parse_initial_input(&inputs, &self.roster());
        leader::run(
            self.clone(),
            input.leader_query,
            session,
            input.deliveries,
            None,
        )
        .await
    }

    /// Streams the same run as [`Self::invoke`], frame by frame.
    ///
    /// The run is spawned into an unbounded channel, so frames are yielded as they are produced;
    /// a run error becomes the last stream item. Dropping the stream does not cancel the run —
    /// call [`Self::stop`] for that.
    // The caller hands the request in as one owned value; the stream owns it from then on.
    #[allow(clippy::needless_pass_by_value)]
    pub fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> futures::stream::BoxStream<'static, Result<AgentChunk>> {
        let input = parse_initial_input(&inputs, &self.roster());
        let agent = self.clone();
        let (tx, mut rx) = mpsc::unbounded_channel::<Result<AgentChunk>>();
        let run_sink = tx.clone();
        // The task owns the run: the stream only forwards frames, so a slow consumer never stalls
        // the coordination loop.
        tokio::spawn(async move {
            if let Err(error) = leader::run(
                agent,
                input.leader_query,
                session,
                input.deliveries,
                Some(run_sink),
            )
            .await
            {
                let _ = tx.send(Err(error));
            }
        });
        async_stream::stream! {
            while let Some(item) = rx.recv().await {
                yield item;
            }
        }
        .boxed()
    }

    /// Routes one external payload when the team is active.
    ///
    /// Python parity: `TeamAgent.interact`. The gate decides: closed, not active, or admitted.
    // The async signature is the seam (a future gate may await admission); the gate itself is a
    // lock-guarded queue.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn interact(&self, payload: InteractPayload) -> DeliverResult {
        self.inner.gate.dispatch(payload, self.is_started())
    }

    /// Delivers one payload into the team without touching the leader runtime.
    ///
    /// Routing (pinned by `src/runtime/CONTEXT.md`):
    ///
    /// * [`InteractPayload::HumanAgent`] — the inbox (when HITL is enabled) plus one persisted
    ///   mailbox row; the delivered message id is returned.
    /// * [`InteractPayload::Operator`] / [`InteractPayload::GroupChat`] /
    ///   [`InteractPayload::GodView`] — the interaction gate, then one persisted row addressed to
    ///   the leader.
    /// * [`InteractPayload::HumanAgentToolCall`] — the interaction gate, then one relayed team tool
    ///   call executed under the passive human's identity; the answer carries `{ok, output, data}`
    ///   and, on a refusal, one of the stable tokens `human_agent_not_enabled`,
    ///   `unknown_human_agent:<name>`, `tool_passthrough_avatar_not_supported`,
    ///   `unknown_tool:<tool>`, `invalid_arguments:<tool>` or `tool_failed:<tool>`.
    /// * [`InteractPayload::ExternalTeam`] — rejected as `unsupported:external_team_transport`:
    ///   this build ships no distributed transport.
    pub async fn deliver(&self, payload: InteractPayload) -> DeliverResult {
        match payload {
            InteractPayload::HumanAgent(message) => self.deliver_human_agent(message).await,
            InteractPayload::Operator(_)
            | InteractPayload::GroupChat(_)
            | InteractPayload::GodView(_) => self.deliver_leader_bound(payload).await,
            InteractPayload::HumanAgentToolCall(call) => self.deliver_tool_call(call).await,
            InteractPayload::ExternalTeam(_) => {
                DeliverResult::unsupported("external_team_transport")
            }
        }
    }

    /// Persists a human-agent message (and queues it on the inbox when HITL is on).
    async fn deliver_human_agent(&self, message: HumanAgentMessage) -> DeliverResult {
        if self.inner.inbox.is_enabled() {
            if let Err(error) = self.inner.inbox.push(message.clone()).await {
                warn!(team = self.team_name(), error = %error, "failed to queue a human-agent message");
            }
        } else {
            debug!(
                team = self.team_name(),
                human = %message.human_name,
                "human agent is not enabled; persisting the message only"
            );
        }
        let sender = message.sender.as_deref().unwrap_or(USER_PSEUDO_MEMBER_NAME);
        match self
            .inner
            .backend
            .send_message(sender, message.target.as_deref(), &message.body)
            .await
        {
            Ok(record) => DeliverResult::ok(Some(record.message_id)),
            Err(error) => DeliverResult::failed(error.to_string()),
        }
    }

    /// Admits a leader-bound payload through the gate and forwards it to the leader mailbox.
    async fn deliver_leader_bound(&self, payload: InteractPayload) -> DeliverResult {
        let admitted = self.inner.gate.dispatch(payload.clone(), self.is_started());
        if !admitted.ok {
            return admitted;
        }
        let (sender, body) = match &payload {
            InteractPayload::GodView(message) => (message.sender.clone(), message.body.clone()),
            InteractPayload::Operator(message) => (message.sender.clone(), message.body.clone()),
            InteractPayload::GroupChat(message) => (message.sender.clone(), message.body.clone()),
            _ => (None, String::new()),
        };
        let from = sender.as_deref().unwrap_or(USER_PSEUDO_MEMBER_NAME);
        match self
            .inner
            .backend
            .send_message(from, Some(self.leader_member_name()), &body)
            .await
        {
            Ok(_) => DeliverResult::ok(None),
            Err(error) => DeliverResult::failed(format!(
                "send_failed:{}: {error}",
                self.leader_member_name()
            )),
        }
    }

    /// Relays one team tool call on behalf of a passive human member.
    ///
    /// Python parity: `TeamAgent._dispatch_tool` in
    /// `openjiuwen/agent_teams/runtime/manager.py`. A passive human has no avatar, so the runtime
    /// executes the call verbatim under `call.human_name`'s identity: assignee locks, reviewer
    /// guards and sender-bound message rows all follow from that name.
    ///
    /// The call is admitted through the same gate as every other leader-bound payload (so an
    /// inactive team answers `not_active` and a closing one `gate_closed`), then checked in order:
    /// the human-agent channel must be enabled, `human_name` must be on the roster, that member must
    /// be a [`TeamRole::PassiveHuman`] (an avatar keeps its own governed LLM tool loop), and the
    /// tool name must be one of the [`tool_dispatch`] team tools. Only then is the call dispatched.
    ///
    /// A successful call answers `{ok: true, output, data}`: `output` is the model-facing text and
    /// `data` the raw tool payload, so an external protocol can show the human the effect
    /// synchronously. A failure answers `ok: false` with one stable token and no `output`/`data`.
    async fn deliver_tool_call(&self, call: HumanAgentToolCall) -> DeliverResult {
        let admitted = self.inner.gate.dispatch(
            InteractPayload::HumanAgentToolCall(call.clone()),
            self.is_started(),
        );
        if !admitted.ok {
            return admitted;
        }
        if !self.inner.inbox.is_enabled() {
            return DeliverResult::failed("human_agent_not_enabled");
        }
        let Some(member) = self.inner.backend.member(&call.human_name) else {
            return DeliverResult::failed(format!("unknown_human_agent:{}", call.human_name));
        };
        if member.role() != TeamRole::PassiveHuman {
            return DeliverResult::failed("tool_passthrough_avatar_not_supported");
        }
        // Checked before dispatching, so a board refusal can never be mistaken for an unknown tool.
        if !tool_dispatch::is_supported_tool(&call.tool_name) {
            return DeliverResult::failed(format!("unknown_tool:{}", call.tool_name));
        }
        match tool_dispatch::dispatch_tool(
            &self.inner.backend,
            &call.human_name,
            &call.tool_name,
            &call.tool_args,
        )
        .await
        {
            Ok(value) => DeliverResult {
                ok: true,
                message_id: None,
                reason: None,
                output: tool_dispatch::render_tool_output(&value),
                data: value,
            },
            Err(error) => {
                // The token stays stable and short; the human-readable detail goes to the log.
                if error.kind() == ErrorKind::Validation {
                    debug!(
                        team = self.team_name(),
                        human = %call.human_name,
                        tool = %call.tool_name,
                        error = %error,
                        "a relayed tool call carried invalid arguments"
                    );
                    DeliverResult::failed(format!("invalid_arguments:{}", call.tool_name))
                } else {
                    warn!(
                        team = self.team_name(),
                        human = %call.human_name,
                        tool = %call.tool_name,
                        error = %error,
                        "a relayed tool call failed"
                    );
                    DeliverResult::failed(format!("tool_failed:{}", call.tool_name))
                }
            }
        }
    }

    /// Returns the roster projection with the team state and its task board.
    ///
    /// A caller that reads the snapshot's directory sees a current `TEAM_MEMORY.md`: the document
    /// is exported best-effort here (a failure is logged), so a snapshot never fails on memory and
    /// its shape is unchanged.
    ///
    /// # Errors
    ///
    /// Propagates store failures while reading the board.
    pub async fn snapshot(&self) -> TeamSnapshot {
        let mut snapshot = self
            .inner
            .backend
            .snapshot_with_tasks()
            .await
            .unwrap_or_else(|error| {
                warn!(team = self.team_name(), error = %error, "failed to read the task board");
                self.inner.backend.snapshot()
            });
        snapshot.state = self.state();
        memory_bind::export_document(self.inner.backend.memory().as_ref(), self.team_name()).await;
        snapshot
    }

    /// Hands a handoff note to team memory for every roster row that is still live.
    ///
    /// "Live" is [`memory_bind::is_live`]: a row that started at least once and has not departed
    /// yet. A row that never started never joined, so it has nothing to leave.
    async fn leave_live_members(&self) {
        let memory = self.inner.backend.memory();
        let Some(wiring) = memory.as_ref() else {
            return;
        };
        for member in self.inner.backend.members() {
            if !memory_bind::is_live(member.status()) {
                continue;
            }
            memory_bind::member_left(Some(wiring), self.team_name(), member.member_name()).await;
        }
    }
}
