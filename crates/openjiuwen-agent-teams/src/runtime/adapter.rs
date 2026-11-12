//! The `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` seams over [`TeamAgent`].
//!
//! # Why this file exists
//!
//! A [`TeamAgent`] *is* the team, and two foreign traits must see it as their kind of object:
//!
//! * [`openjiuwen_agent::Agent`] — the seam `openjiuwen-runner` and every composite agent drive:
//!   one card, one [`invoke`](openjiuwen_agent::Agent::invoke), one
//!   [`stream`](openjiuwen_agent::Agent::stream).
//! * [`openjiuwen_multiagent::Team`] — the seam `Runner::run_agent_team` and the multiagent
//!   strategies drive: a team card, a membership registry, mail (send / publish / subscribe) and a
//!   team stream.
//!
//! Both implementations live directly on [`TeamAgent`]; there is no wrapper type, because a wrapper
//! would have to forward two dozen methods and would hide the identity `team_id() == team_name`.
//!
//! # Documented deviations from the multiagent contract
//!
//! * `Team::invoke` / `Team::stream` carry an `Option<Arc<AgentTeamSession>>`, while
//!   [`openjiuwen_agent::Agent`] carries `Option<Arc<dyn Session>>`. `AgentTeamSession` is not a
//!   `dyn Session`, so the multiagent session is *not* threaded through: the harness session
//!   created at build owns the cycle, and the inherent [`TeamAgent::invoke`] is called with `None`.
//!   Python's `TeamAgent.invoke` likewise ignores the session argument.
//! * `Team::add_agent` / `Team::add_member` are synchronous in the multiagent seam, so their roster
//!   row write cannot be awaited. The member joins the in-memory roster immediately and the store
//!   write is spawned onto the current runtime; a failed write is logged at `warn`, never hidden.
//! * The `provider` / `options` arguments are ignored: a member's brain always comes from the
//!   injected [`MemberHarnessFactory`](crate::harness::MemberHarnessFactory), never from a
//!   multiagent provider, and only an `unstarted` roster row is created.
//! * The multiagent bus is not used — the team's own [`Messager`](crate::messager::Messager)
//!   carries its envelope — so `Team::subscribe` installs an observing sink that logs at `trace`
//!   and drops the event, and `Team::send` / `Team::publish` route through the team's mailbox and
//!   bus instead of `TeamRuntime`.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_multiagent::{
    AgentProvider, MemberOptions, Team, TeamCard, TeamConfig, TeamRuntime,
};
use openjiuwen_session::AgentTeamSession;
use serde_json::{json, Value};
use tracing::{debug, trace, warn};

use super::team_agent::TeamAgent;
use crate::member::TeamMember;
use crate::messager::EventSink;
use crate::schema::{EventMessage, EventPayload, TeamRole, TeamTopic};
use crate::storage::MemberRecord;

/// Strategy kind reported by [`Team::kind`]; the Python `agent_teams` organisation.
const TEAM_KIND: &str = "agent_teams";

/// Event type [`Team::publish`] stamps onto a multiagent message republished on the team bus.
const PUBLISH_EVENT_TYPE: &str = "publish";

/// Message of the empty-card-name validation.
const EMPTY_CARD_NAME: &str = "agent card name must not be empty";

/// The [`openjiuwen_agent::Agent`] seam: one card, one `invoke`, one `stream`.
///
/// Every method delegates to the inherent [`TeamAgent`] method of the same name (the inherent
/// methods win in method-call syntax, so the delegation is written explicitly to rule out
/// recursion).
#[async_trait]
impl openjiuwen_agent::Agent for TeamAgent {
    fn card(&self) -> &AgentCard {
        TeamAgent::card(self)
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn openjiuwen_session::Session>>,
    ) -> Result<openjiuwen_agent::AgentOutput> {
        TeamAgent::invoke(self, inputs, session).await
    }

    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn openjiuwen_session::Session>>,
    ) -> BoxStream<'static, Result<openjiuwen_agent::AgentChunk>> {
        TeamAgent::stream(self, inputs, session)
    }
}

/// The [`openjiuwen_multiagent::Team`] seam over the running team.
///
/// The team card, the membership registry and the [`TeamRuntime`] are projections of the team
/// itself: `team_id()` is the team name, every current member contributes one
/// `<team>_<member>` card, and `runtime()` is the session/registry seam created at build. Mail is
/// *not* routed through [`TeamRuntime`] — the team's own mailbox and bus carry its envelope (see
/// the module docs).
#[async_trait]
impl Team for TeamAgent {
    /// Returns the team card: id and name are the team name, one `<team>_<member>` card per member.
    fn card(&self) -> TeamCard {
        TeamAgent::team_card(self)
    }

    /// Returns the multiagent defaults; the team's own limits live in
    /// [`TeamAgentSpec`](crate::schema::TeamAgentSpec) and [`TeamSpec`](crate::schema::TeamSpec).
    fn config(&self) -> TeamConfig {
        TeamConfig::default()
    }

    /// Returns the team name, which is both the team id and the card name.
    fn team_id(&self) -> &str {
        TeamAgent::team_name(self)
    }

    /// Returns the strategy kind of this team: `agent_teams`.
    fn kind(&self) -> &str {
        TEAM_KIND
    }

    /// Returns the `openjiuwen_multiagent` runtime created at build (a session/registry seam).
    fn runtime(&self) -> Arc<TeamRuntime> {
        TeamAgent::team_runtime(self)
    }

    /// Adds one `unstarted` roster member with the default member options.
    ///
    /// The provider is ignored (documented in the module docs); a duplicate name is a no-op.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty card name, and [`Error::Config`] when no
    /// tokio runtime is available to persist the roster row.
    fn add_agent(&self, card: AgentCard, provider: AgentProvider) -> Result<()> {
        debug!(
            team = self.team_name(),
            member = %card.name,
            "adding a multiagent member; its brain comes from the harness factory"
        );
        drop(provider);
        insert_roster_member(self, card)
    }

    /// Adds one `unstarted` roster member with explicit wiring options.
    ///
    /// Same as [`add_agent`](Team::add_agent): the provider and the options are ignored, because
    /// the team's own dispatch decides where a member sits.
    ///
    /// # Errors
    ///
    /// Same as [`add_agent`](Team::add_agent).
    fn add_member(
        &self,
        card: AgentCard,
        provider: AgentProvider,
        options: MemberOptions,
    ) -> Result<()> {
        debug!(
            team = self.team_name(),
            member = %card.name,
            parent = options.parent_agent_id.as_deref().unwrap_or_default(),
            "adding a multiagent member; provider and options are ignored"
        );
        drop(provider);
        insert_roster_member(self, card)
    }

    /// Drops a member from the live roster; an unknown member is a no-op.
    ///
    /// The store row is left untouched: retirement, not removal, settles a member lifecycle.
    ///
    /// # Errors
    ///
    /// This implementation never fails.
    fn remove_agent(&self, agent: &str) -> Result<()> {
        if self.backend().remove_member(agent).is_none() {
            debug!(
                team = self.team_name(),
                member = agent,
                "member is not on the roster; nothing to remove"
            );
        }
        Ok(())
    }

    /// Returns the card of a current member, named `<team>_<member>`; `None` when it is not one.
    fn get_agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        self.backend().member(agent_id)?;
        Some(AgentCard::new(format!("{}_{agent_id}", self.team_name())))
    }

    /// Returns the number of live roster members, the leader included.
    fn agent_count(&self) -> usize {
        self.backend().len()
    }

    /// Returns the live member names, sorted.
    fn list_agents(&self) -> Vec<String> {
        self.backend().member_names()
    }

    /// Validates that the team can run: the leader row exists and has a runtime attached.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when the leader row or its runtime is missing (a team
    /// always builds with both, so this only fires on a corrupted roster).
    fn check_ready(&self) -> Result<()> {
        let backend = self.backend();
        match backend.member(self.leader_member_name()) {
            Some(member) if member.runtime().is_some() => Ok(()),
            _ => Err(Error::config(format!(
                "team '{}' has no leader runtime",
                self.team_name()
            ))),
        }
    }

    /// Delivers one multiagent message into the team's mailbox and returns its ids.
    ///
    /// The body is the message itself when it is a string, else its `content` / `query` / `text`
    /// field, else the compact JSON. Routing is the team's own: one direct mailbox row addressed to
    /// `recipient`, published on the message topic, which is how a member reads its mail.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an over-long body or an empty target, and whatever the store or
    /// the transport fails with.
    // The multiagent session and timeout are part of the foreign signature and unused here.
    async fn send(
        &self,
        message: Value,
        recipient: &str,
        sender: &str,
        _session_id: Option<&str>,
        _timeout: Option<Duration>,
    ) -> Result<Value> {
        let text = message_text(&message);
        let record = self
            .backend()
            .send_message(sender, Some(recipient), &text)
            .await?;
        Ok(json!({"message_id": record.message_id, "recipient": recipient}))
    }

    /// Republishes one multiagent message on the team bus.
    ///
    /// `topic_id` maps to the team's topic category (`team` / `task` / anything else is `message`)
    /// and the message is published as a `publish` event; the event type is discarded.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the payload cannot be serialized, and whatever the transport
    /// fails with.
    async fn publish(
        &self,
        message: Value,
        topic_id: &str,
        _sender: &str,
        _session_id: Option<&str>,
    ) -> Result<()> {
        self.backend()
            .post_event(
                topic_of(topic_id),
                EventPayload::Unknown {
                    event_type: PUBLISH_EVENT_TYPE.to_owned(),
                    payload: message,
                },
            )
            .await?;
        Ok(())
    }

    /// Subscribes a member to one team topic with an observing sink.
    ///
    /// The sink logs the envelope at `trace` and drops it: this seam exists so a multiagent
    /// strategy can observe the team, while real routing stays on the team's own bus.
    ///
    /// # Errors
    ///
    /// Returns whatever the transport fails with.
    async fn subscribe(&self, agent_id: &str, topic: &str) -> Result<()> {
        self.backend()
            .messager()
            .subscribe(agent_id, &[topic_of(topic)], observing_sink())
    }

    /// Removes every subscription of a member; an unknown member is a no-op.
    ///
    /// # Errors
    ///
    /// Returns whatever the transport fails with.
    async fn unsubscribe(&self, agent_id: &str, _topic: &str) -> Result<()> {
        self.backend().messager().unsubscribe(agent_id)
    }

    /// Starts the team (transport, leader runtime and gate). Idempotent.
    ///
    /// # Errors
    ///
    /// Propagates start failures; see [`TeamAgent::start`].
    async fn start(&self) -> Result<()> {
        TeamAgent::start(self).await
    }

    /// Stops the team (cancels the run, stops the leader and closes the gate). Idempotent.
    ///
    /// # Errors
    ///
    /// This implementation never fails; see [`TeamAgent::stop`].
    async fn stop(&self) -> Result<()> {
        TeamAgent::stop(self).await
    }

    /// Runs one team invocation and returns the serialized [`AgentOutput`](openjiuwen_agent::AgentOutput).
    ///
    /// The multiagent session is not forwarded (documented in the module docs).
    ///
    /// # Errors
    ///
    /// Propagates the run's errors and [`Error::Validation`] when the output cannot be serialized.
    async fn invoke(&self, input: Value, _session: Option<Arc<AgentTeamSession>>) -> Result<Value> {
        let output = TeamAgent::invoke(self, input, None).await?;
        serde_json::to_value(output).map_err(|error| {
            Error::validation(format!("failed to serialize the team output: {error}"))
        })
    }

    /// Runs one team invocation as a stream of serialized [`AgentChunk`](openjiuwen_agent::AgentChunk)
    /// frames.
    ///
    /// A frame that cannot be serialized becomes the stream's error item.
    fn stream(
        &self,
        input: Value,
        _session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        TeamAgent::stream(self, input, None)
            .map(|item| {
                item.and_then(|chunk| {
                    serde_json::to_value(chunk).map_err(|error| {
                        Error::validation(format!(
                            "failed to serialize a team stream frame: {error}"
                        ))
                    })
                })
            })
            .boxed()
    }
}

/// Returns the team topic category of one multiagent topic id.
fn topic_of(topic_id: &str) -> TeamTopic {
    match topic_id {
        "team" => TeamTopic::Team,
        "task" => TeamTopic::Task,
        _ => TeamTopic::Message,
    }
}

/// Returns the text of one multiagent message.
///
/// A string message is its own text; an object uses its `content`, `query` or `text` field; every
/// other shape falls back to the compact JSON, so nothing is lost on the way to the mailbox.
fn message_text(message: &Value) -> String {
    if let Some(text) = message.as_str() {
        return text.to_owned();
    }
    for field in ["content", "query", "text"] {
        if let Some(text) = message.get(field).and_then(Value::as_str) {
            return text.to_owned();
        }
    }
    message.to_string()
}

/// Builds the observing sink a [`Team::subscribe`] call installs.
fn observing_sink() -> EventSink {
    Arc::new(|event: EventMessage| {
        trace!(
            event_type = event.event_type.as_str(),
            sender = event.sender_id.as_str(),
            "multiagent subscription observed a team event"
        );
    })
}

/// Registers one `unstarted` roster member for `card` and persists its row.
///
/// A duplicate name is a no-op. The row write is spawned because the multiagent seam is
/// synchronous; the in-memory roster is authoritative in the meantime, and a failed write is
/// logged so the divergence is visible.
fn insert_roster_member(team: &TeamAgent, card: AgentCard) -> Result<()> {
    let AgentCard {
        name, description, ..
    } = card;
    if name.is_empty() {
        return Err(Error::validation(EMPTY_CARD_NAME));
    }
    let backend = team.backend();
    if backend.member(&name).is_some() {
        debug!(
            team = team.team_name(),
            member = %name,
            "member is already on the roster; the add is a no-op"
        );
        return Ok(());
    }
    let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
        Error::config("adding a team member requires a tokio runtime to persist its roster row")
    })?;
    let record = MemberRecord {
        display_name: name.clone(),
        desc: description,
        ..MemberRecord::new(team.team_name(), &name, TeamRole::Teammate)
    };
    let store = backend.store();
    let member = Arc::new(TeamMember::new(Arc::clone(&store), record.clone()));
    let team_name = team.team_name().to_owned();
    runtime.spawn(async move {
        if let Err(error) = store.insert_member(record).await {
            warn!(
                team = %team_name,
                member = %name,
                error = %error,
                "failed to persist the new member row"
            );
        }
    });
    backend.add_member(member);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt;
    use openjiuwen_multiagent::Team;
    use serde_json::json;

    use super::{message_text, topic_of, TeamAgent};
    use crate::runtime::team_agent::test_support::{blueprint, built};
    use crate::schema::{MemberStatus, TeamTopic, TEAM_COMPLETED};
    use crate::storage::MemberRecord;

    /// Upper bound for any single wait in these tests.
    const BOUND: Duration = Duration::from_secs(5);

    /// Builds the standard leader handle: team `alpha`, leader `boss`, two teammates.
    async fn team_with_two_teammates() -> std::sync::Arc<TeamAgent> {
        let (team, _factory, _workspace) =
            built(&blueprint("alpha", "boss", &["dev-1", "dev-2"])).await;
        team
    }

    /// Builds an [`AgentProvider`] the adapter must ignore.
    fn provider(team: &std::sync::Arc<TeamAgent>) -> openjiuwen_multiagent::AgentProvider {
        let agent: std::sync::Arc<dyn openjiuwen_agent::Agent> = team.clone();
        std::sync::Arc::new(move || {
            openjiuwen_multiagent::TeamAgent::new(std::sync::Arc::clone(&agent))
        })
    }

    /// Waits (bounded) for the spawned roster-row write to land in the store.
    async fn persisted_row(team: &std::sync::Arc<TeamAgent>, member: &str) -> MemberRecord {
        let store = team.backend().store();
        for _ in 0..500 {
            if let Some(record) = store.member(team.team_name(), member).await.unwrap() {
                return record;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("the roster row of '{member}' was not persisted");
    }

    #[tokio::test]
    async fn identity_membership_and_readiness_project_the_team() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        assert_eq!(Team::team_id(agent), "alpha");
        assert_eq!(Team::team_id(agent), Team::card(agent).name);
        assert_eq!(Team::kind(agent), "agent_teams");
        assert_eq!(Team::agent_count(agent), 3, "the leader plus two teammates");
        assert_eq!(Team::list_agents(agent), vec!["boss", "dev-1", "dev-2"]);
        assert_eq!(
            Team::card(agent).agent_cards.len(),
            3,
            "one member card each"
        );
        assert_eq!(
            Team::card(agent).agent_cards[0].name,
            "alpha_boss",
            "member cards are named <team>_<member>"
        );
        assert_eq!(Team::config(agent).max_agents, 10, "multiagent defaults");
        assert!(std::sync::Arc::ptr_eq(
            &Team::runtime(agent),
            &agent.team_runtime()
        ));
        Team::check_ready(agent).unwrap();
    }

    #[tokio::test]
    async fn get_agent_card_answers_for_members_only() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        let card = Team::get_agent_card(agent, "dev-1").unwrap();
        assert_eq!(card.name, "alpha_dev-1");
        assert!(Team::get_agent_card(agent, "ghost").is_none());
    }

    #[tokio::test]
    async fn add_agent_registers_an_unstarted_row_and_is_idempotent() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        let card = openjiuwen_agent::AgentCard::new("dev-3").with_description("the third one");
        Team::add_agent(agent, card.clone(), provider(&team)).unwrap();

        let member = team.backend().member("dev-3").unwrap();
        assert_eq!(member.status(), MemberStatus::Unstarted);
        assert_eq!(member.role(), crate::schema::TeamRole::Teammate);
        assert_eq!(member.record().display_name, "dev-3");
        assert_eq!(member.record().desc, "the third one");
        assert!(member.runtime().is_none(), "no brain until it is spawned");
        assert_eq!(Team::agent_count(agent), 4);
        assert_eq!(
            Team::list_agents(agent),
            vec!["boss", "dev-1", "dev-2", "dev-3"]
        );

        let row = persisted_row(&team, "dev-3").await;
        assert_eq!(row.team_name, "alpha");
        assert_eq!(row.status, MemberStatus::Unstarted);
        assert_eq!(row.desc, "the third one");

        // Idempotent: a second add keeps the original row and returns `Ok`.
        Team::add_agent(agent, card, provider(&team)).unwrap();
        assert_eq!(Team::agent_count(agent), 4);
        assert!(team.backend().member("dev-3").is_some());
    }

    #[tokio::test]
    async fn add_member_accepts_options_but_ignores_them() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        Team::add_member(
            agent,
            openjiuwen_agent::AgentCard::new("dev-4"),
            provider(&team),
            openjiuwen_multiagent::MemberOptions {
                parent_agent_id: Some("boss".to_owned()),
            },
        )
        .unwrap();

        assert_eq!(
            team.backend().member("dev-4").unwrap().status(),
            MemberStatus::Unstarted
        );
        assert_eq!(Team::agent_count(agent), 4);
    }

    #[tokio::test]
    async fn add_agent_rejects_an_empty_card_name() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        let error = Team::add_agent(agent, openjiuwen_agent::AgentCard::new(""), provider(&team))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(
            error
                .to_string()
                .contains("agent card name must not be empty"),
            "{error}"
        );
        assert_eq!(Team::agent_count(agent), 3, "nothing was registered");
    }

    #[tokio::test]
    async fn remove_agent_drops_the_roster_member_and_ignores_unknown_names() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        Team::remove_agent(agent, "dev-1").unwrap();
        assert!(team.backend().member("dev-1").is_none());
        assert_eq!(Team::agent_count(agent), 2);
        assert!(
            !Team::list_agents(agent).contains(&"dev-1".to_owned()),
            "the removed member leaves the registry"
        );

        Team::remove_agent(agent, "ghost").unwrap();
        assert_eq!(Team::agent_count(agent), 2);
    }

    #[tokio::test]
    async fn send_persists_one_direct_mailbox_row() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        let receipt = Team::send(agent, json!("status?"), "dev-1", "boss", None, None)
            .await
            .unwrap();
        assert_eq!(receipt["recipient"], "dev-1");
        assert!(
            receipt["message_id"]
                .as_str()
                .is_some_and(|id| !id.is_empty()),
            "{receipt:?}"
        );

        let mailbox = team.backend().unread("dev-1").await.unwrap();
        assert_eq!(mailbox.len(), 1, "{mailbox:?}");
        assert_eq!(mailbox[0].content, "status?");
        assert_eq!(mailbox[0].from_member_name, "boss");
        assert!(!mailbox[0].broadcast, "a recipient keeps it direct");

        // An object message uses its `content` field as the body.
        Team::send(
            agent,
            json!({"content": "an object body"}),
            "dev-2",
            "boss",
            None,
            None,
        )
        .await
        .unwrap();
        let mailbox = team.backend().unread("dev-2").await.unwrap();
        assert_eq!(mailbox[0].content, "an object body");
    }

    #[tokio::test]
    async fn publish_and_subscribe_use_the_team_bus() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;
        let backend = team.backend();
        let messager = backend.messager();

        Team::publish(agent, json!({"n": 1}), "task", "boss", None)
            .await
            .unwrap();

        assert!(
            !messager
                .subscribers(TeamTopic::Task)
                .contains(&"dev-1".to_owned()),
            "the multiagent subscription is not implied by publish"
        );
        Team::subscribe(agent, "dev-1", "task").await.unwrap();
        assert!(messager
            .subscribers(TeamTopic::Task)
            .contains(&"dev-1".to_owned()));

        Team::unsubscribe(agent, "dev-1", "task").await.unwrap();
        assert!(
            !messager
                .subscribers(TeamTopic::Task)
                .contains(&"dev-1".to_owned()),
            "unsubscribe drops every topic of the member"
        );
        assert!(team.backend().member("dev-1").is_some(), "the roster stays");
    }

    #[test]
    fn helpers_map_topics_and_extract_text() {
        assert_eq!(topic_of("team"), TeamTopic::Team);
        assert_eq!(topic_of("task"), TeamTopic::Task);
        assert_eq!(topic_of("anything-else"), TeamTopic::Message);

        assert_eq!(message_text(&json!("hello")), "hello");
        assert_eq!(message_text(&json!({"content": "body"})), "body");
        assert_eq!(message_text(&json!({"query": "question"})), "question");
        assert_eq!(message_text(&json!({"text": "words"})), "words");
        assert_eq!(message_text(&json!({"other": 1})), "{\"other\":1}");
        assert_eq!(message_text(&json!(7)), "7");
    }

    #[tokio::test]
    async fn team_invoke_serializes_the_leader_answer() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        let value = Team::invoke(agent, json!("hello"), None).await.unwrap();
        assert_eq!(value["output"], "done: boss");
        assert_eq!(value["result_type"], "answer");
        assert!(agent.is_started());
    }

    #[tokio::test]
    async fn team_stream_yields_serialized_frames_ending_with_the_marker() {
        let team = team_with_two_teammates().await;
        let agent: &TeamAgent = &team;

        let frames = tokio::time::timeout(BOUND, async {
            let mut frames = Vec::new();
            let mut stream = Team::stream(agent, json!("hello"), None);
            while let Some(item) = stream.next().await {
                frames.push(item);
            }
            frames
        })
        .await
        .expect("the stream did not end within the bound");

        assert!(
            frames.len() >= 2,
            "an answer frame and the marker: {frames:?}"
        );
        assert!(frames.iter().all(Result::is_ok), "{frames:?}");
        let last = frames.last().unwrap().as_ref().unwrap();
        assert_eq!(last["type"], TEAM_COMPLETED);
        let first = frames.first().unwrap().as_ref().unwrap();
        assert_eq!(first["payload"]["source_member"], "boss");
    }

    #[tokio::test]
    async fn the_agent_seam_delegates_card_invoke_and_stream() {
        let team = team_with_two_teammates().await;
        let agent: &dyn openjiuwen_agent::Agent = team.as_ref();

        assert_eq!(agent.card().name, "alpha_boss");
        let output = agent.invoke(json!("hello"), None).await.unwrap();
        assert_eq!(output.output, "done: boss");

        let frames = tokio::time::timeout(BOUND, async {
            let mut frames = Vec::new();
            let mut stream = agent.stream(json!("hello"), None);
            while let Some(item) = stream.next().await {
                frames.push(item);
            }
            frames
        })
        .await
        .expect("the stream did not end within the bound");
        assert!(frames.iter().all(Result::is_ok), "{frames:?}");
        let kinds: Vec<String> = frames.into_iter().map(|item| item.unwrap().kind).collect();
        assert_eq!(kinds.first().map(String::as_str), Some("llm_output"));
        assert_eq!(kinds.last().map(String::as_str), Some(TEAM_COMPLETED));
    }
}
