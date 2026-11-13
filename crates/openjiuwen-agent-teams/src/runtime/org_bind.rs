//! The organisation layer bound to one running team.
//!
//! A team participates in an organisation when its spec carries
//! `metadata["organization"] = {"spec": OrganizationSpec, "rules": DelegationRules}`. The build
//! path parses that block (see [`super::spec_meta`]) and stores an [`OrgBinding`] on the backend:
//! the chart, the delegation router, the cross-team task pool and the org mailbox sharing one
//! store.
//!
//! Wiring rules:
//!
//! * every [`OrgTaskEvent`] the pool publishes is forwarded as
//!   `EventMessage { event_type: "org_task", .. }` on [`TeamTopic::Task`], so the same bus that
//!   carries board events also carries org events; forwarding is fire-and-forget (a failed publish
//!   is logged, never propagated into the pool);
//! * the leader loop drains [`OrgMailbox::inbox`] each tick and acknowledges the messages only
//!   after a leader turn was successfully sent;
//! * an org-level delegation is checked against [`DelegationRouter`] before the pool is asked, so a
//!   refused hop never mutates the cross-team board.
//!
//! With no organisation block every entry point here is absent (`Option<Arc<OrgBinding>>`), which
//! keeps the plain in-process team untouched.
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use tracing::warn;

use crate::messager::Messager;
use crate::org::{
    DelegationRequest, DelegationRouter, DelegationRules, OrgChart, OrgMailbox, OrgMessage,
    OrgTask, OrgTaskEvent, OrgTaskEventSink, OrgTaskOpResult, OrgTaskPool, OrgTaskSpec,
    OrganizationSpec,
};
use crate::schema::{EventPayload, TeamTopic};
use crate::storage::TeamStore;

use super::backend::TeamBackend;
use super::spec_meta::OrganizationBlock;

/// Wire event type used for organisation task events on the team bus.
pub const ORG_TASK_EVENT: &str = "org_task";

/// The organisation half of a running team: chart, router, task pool and mailbox.
pub struct OrgBinding {
    organization_id: String,
    chart: Arc<OrgChart>,
    router: Arc<DelegationRouter>,
    pool: Arc<OrgTaskPool>,
    mailbox: Arc<OrgMailbox>,
}
impl std::fmt::Debug for OrgBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrgBinding")
            .field("organization_id", &self.organization_id)
            .finish()
    }
}
impl OrgBinding {
    /// Builds the binding for `block`, optionally forwarding pool events to `sink`.
    ///
    /// # Errors
    ///
    /// Propagates an invalid chart, an invalid organisation id and store construction failures.
    pub fn build(
        block: &OrganizationBlock,
        store: Arc<dyn TeamStore>,
        sink: Option<OrgTaskEventSink>,
    ) -> Result<Self> {
        let spec = block.spec.clone();
        let chart = Arc::new(OrgChart::new(spec)?);
        let router = Arc::new(DelegationRouter::new(
            Arc::clone(&chart),
            block.rules.clone(),
        ));
        let mut pool = OrgTaskPool::new(store.clone(), Arc::clone(&router))?;
        if let Some(sink) = sink {
            pool = pool.with_event_sink(sink);
        }
        let organization_id = chart.organization_id().to_owned();
        let mailbox = Arc::new(OrgMailbox::new(store, &organization_id)?);
        Ok(Self {
            organization_id,
            chart,
            router,
            pool: Arc::new(pool),
            mailbox,
        })
    }
    /// Returns the organisation id.
    #[must_use]
    pub fn organization_id(&self) -> &str {
        &self.organization_id
    }
    /// Returns the chart.
    #[must_use]
    pub fn chart(&self) -> &Arc<OrgChart> {
        &self.chart
    }
    /// Returns the delegation router.
    #[must_use]
    pub fn router(&self) -> &Arc<DelegationRouter> {
        &self.router
    }
    /// Returns the cross-team task pool.
    #[must_use]
    pub fn pool(&self) -> &Arc<OrgTaskPool> {
        &self.pool
    }
    /// Returns the org mailbox.
    #[must_use]
    pub fn mailbox(&self) -> &Arc<OrgMailbox> {
        &self.mailbox
    }
    /// Checks one delegation hop against the router (no board mutation).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] carrying the router's refusal message.
    pub fn check_delegation(&self, request: &DelegationRequest) -> Result<()> {
        self.router.check(request)
    }
    /// Creates a cross-team task owned by `owner_team`.
    ///
    /// # Errors
    ///
    /// Propagates the pool's validation and store failures.
    pub async fn create_task(&self, spec: OrgTaskSpec) -> Result<crate::org::OrgTaskMutation> {
        self.pool.create(spec).await
    }
    /// Looks up one org task.
    ///
    /// # Errors
    ///
    /// Propagates store failures.
    pub async fn task(&self, task_id: &str) -> Result<Option<OrgTask>> {
        self.pool.get(task_id).await
    }
    /// Delegates `task_id` after checking the router.
    ///
    /// `from_team`/`from_member` identify the delegating member; `depth` is the task's hop count
    /// and the target's open-task count is read from the pool so budget rules see the real load.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the router refuses, [`Error::not_found`] for an unknown
    /// task, and the pool's refusal as [`Error::Validation`] when it rejects the hop.
    pub async fn delegate(
        &self,
        task_id: &str,
        from_team: &str,
        from_member: &str,
        to_team: &str,
        to_member: Option<&str>,
    ) -> Result<OrgTaskOpResult> {
        let task = self
            .pool
            .get(task_id)
            .await?
            .ok_or_else(|| Error::not_found(format!("org task '{task_id}' not found")))?;
        let open = self.pool.open_task_count(to_team).await?;
        let request = DelegationRequest {
            from_team: from_team.to_owned(),
            from_member: from_member.to_owned(),
            to_team: to_team.to_owned(),
            to_member: to_member.map(str::to_owned),
            depth: task.depth,
            open_tasks_in_target: Some(open),
        };
        self.router.check(&request)?;
        let result = self
            .pool
            .delegate(task_id, from_member, to_team, to_member)
            .await?;
        if !result.is_ok() {
            return Err(Error::validation(
                result
                    .reason()
                    .unwrap_or("organisation refused the delegation")
                    .to_owned(),
            ));
        }
        Ok(result)
    }
    /// Drains the unread org mail for `team`.
    ///
    /// # Errors
    ///
    /// Propagates store failures.
    pub async fn unread(&self, team: &str) -> Result<Vec<OrgMessage>> {
        self.mailbox.inbox(team).await
    }
    /// Acknowledges delivered org messages for `team`.
    ///
    /// # Errors
    ///
    /// Propagates store failures.
    pub async fn acknowledge(&self, team: &str, ids: &[String]) -> Result<usize> {
        self.mailbox.acknowledge(team, ids).await
    }
    /// Sends an org message from `team` to another team.
    ///
    /// # Errors
    ///
    /// Propagates mailbox validation and store failures.
    pub async fn send_mail(
        &self,
        from_team: &str,
        to_team: &str,
        subject: &str,
        body: &str,
    ) -> Result<OrgMessage> {
        self.mailbox.send(from_team, to_team, subject, body).await
    }
    /// Broadcasts an org message from `team`.
    ///
    /// # Errors
    ///
    /// Propagates mailbox validation and store failures.
    pub async fn broadcast_mail(
        &self,
        from_team: &str,
        subject: &str,
        body: &str,
    ) -> Result<OrgMessage> {
        self.mailbox.broadcast(from_team, subject, body).await
    }
    /// Returns the chart's teams as `(team_name, role)` pairs, sorted by name.
    #[must_use]
    pub fn teams(&self) -> Vec<(String, crate::org::OrgRole)> {
        self.chart
            .teams()
            .into_iter()
            .map(|team| (team.team_name.clone(), team.role))
            .collect()
    }
    /// Returns the rules the router applies.
    #[must_use]
    pub fn rules(&self) -> &DelegationRules {
        self.router.rules()
    }
    /// Returns the chart spec.
    #[must_use]
    pub fn spec(&self) -> &OrganizationSpec {
        self.chart.spec()
    }
}
/// Builds the pool sink that forwards org task events onto the team's task topic.
///
/// The forwarding task is spawned on the current runtime; outside a runtime the event is dropped
/// with a warning (the pool keeps working, only the bus copy is lost).
#[must_use]
pub fn org_task_sink(backend: Arc<TeamBackend>) -> OrgTaskEventSink {
    Arc::new(move |event: OrgTaskEvent| {
        let payload = match serde_json::to_value(&event) {
            Ok(payload) => payload,
            Err(error) => {
                warn!(%error, "dropping unserializable org task event");
                return;
            }
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            warn!("org task event dropped: no tokio runtime to publish on");
            return;
        };
        let backend = Arc::clone(&backend);
        handle.spawn(async move {
            let payload = EventPayload::Unknown {
                event_type: ORG_TASK_EVENT.to_owned(),
                payload,
            };
            if let Err(error) = backend.post_event(TeamTopic::Task, payload).await {
                warn!(%error, "failed to publish org task event");
            }
        });
    })
}
/// Serializes an org task for a tool/attach response.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the task cannot be serialized (never for a decoded value).
pub fn task_value(task: &OrgTask) -> Result<Value> {
    serde_json::to_value(task).map_err(|error| Error::validation(error.to_string()))
}
/// Publishes a raw org event through `messager`, bypassing the backend (used by tests/hosts).
///
/// # Errors
///
/// Propagates the transport failure.
pub async fn publish_org_event(
    messager: &Arc<dyn Messager>,
    team_name: &str,
    event: &OrgTaskEvent,
) -> Result<()> {
    let payload = serde_json::to_value(event)
        .map_err(|error| Error::validation(format!("org event: {error}")))?;
    let message = crate::schema::EventMessage {
        event_type: ORG_TASK_EVENT.to_owned(),
        payload,
        sender_id: team_name.to_owned(),
    };
    messager.publish(TeamTopic::Task, message).await
}
