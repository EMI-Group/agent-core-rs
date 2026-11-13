//! Organisation operations exposed on the handle: the binding, task creation and delegation.
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use super::super::org_bind::OrgBinding;
use super::TeamAgent;
use crate::org::{OrgTaskOpResult, OrgTaskSpec};

impl TeamAgent {
    /// Returns the organisation binding, when the spec configures one.
    #[must_use]
    pub fn org(&self) -> Option<Arc<OrgBinding>> {
        self.inner.backend.org()
    }

    /// Creates a cross-team organisation task owned by this team.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when no organisation is configured and the pool's own errors otherwise.
    pub async fn create_org_task(&self, spec: OrgTaskSpec) -> Result<crate::org::OrgTaskMutation> {
        let org = self
            .org()
            .ok_or_else(|| Error::config("team has no organisation configured"))?;
        let mutation = org.create_task(spec).await?;
        self.inner.signals.delegation(
            self.leader_member_name(),
            "",
            &mutation.task.task_id,
            "created",
        );
        Ok(mutation)
    }

    /// Delegates one org task after consulting the delegation router.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] without an organisation, [`Error::Validation`] when the router refuses and
    /// the pool's own errors otherwise.
    pub async fn delegate_org_task(
        &self,
        task_id: &str,
        to_team: &str,
        to_member: Option<&str>,
    ) -> Result<OrgTaskOpResult> {
        let org = self
            .org()
            .ok_or_else(|| Error::config("team has no organisation configured"))?;
        let result = org
            .delegate(
                task_id,
                self.team_name(),
                self.leader_member_name(),
                to_team,
                to_member,
            )
            .await?;
        self.inner.signals.delegation(
            self.leader_member_name(),
            to_member.unwrap_or(to_team),
            task_id,
            "allowed",
        );
        Ok(result)
    }
}
