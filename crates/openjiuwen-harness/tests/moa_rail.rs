//! End-to-end suite of the mixture-of-agents surface through the crate root.
//!
//! The module's own `tests.rs` drives `crate::moa::…` paths; this file is the host view — every
//! entry point is named through `openjiuwen_harness::{…}`, which is what a consumer of the flat
//! Python-shaped package sees. The scenarios are the two the rail exists for:
//!
//! - two scripted proposers are consulted concurrently, their proposals are structurally ranked,
//!   and the rail publishes the `moa` prompt section into the shared rail context while recording
//!   one bounded history entry;
//! - a batch in which every proposer fails is a typed `Error::Llm` (never a silent empty
//!   outcome), both through the free `consult` function and through [`MoaRail::consult`];
//! - an unusable configuration is refused with the Python `__post_init__` message.
//!
//! Offline and deterministic: `ScriptedModel` handles stand in for every model, a
//! [`CancellationToken`] nobody fires stands in for cancellation, and nothing reads the wall
//! clock.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use openjiuwen_agent::RailEvent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{BaseMessage, Model};
use openjiuwen_harness::rails::prompt_sections;
use openjiuwen_harness::{
    consult, rank_with_config, HarnessRail, HarnessRailContext, MoaConfig, MoaRail, MoaSnapshot,
    MOA_OUTCOME_EXTRA_KEY, MOA_PROMPT_SECTION, MOA_RAIL_NAME, MOA_STATE_EXTRA_KEY,
};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// The problem every fixture asks the references about.
const PROBLEM: &str = "fix the failing build in src/main.rs";

/// Wraps a scripted model as an injected reference handle.
fn reference(model: ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model)
}

/// Builds the two-proposer configuration of the fixture.
fn two_proposers() -> (MoaConfig, ScriptedModel, ScriptedModel) {
    let detailed = ScriptedModel::text(
        "1. Read src/main.rs.\n2. Fix the failing build.\n- verify with cargo test",
    );
    let terse = ScriptedModel::text("Rebuild the project.");
    let config = MoaConfig::new(vec![reference(detailed.clone()), reference(terse.clone())]);
    (config, detailed, terse)
}

/// Two proposers are consulted, ranked, and both really receive the request.
#[tokio::test]
async fn two_scripted_proposers_are_consulted_and_ranked() -> Result<()> {
    let (config, detailed, terse) = two_proposers();
    let cancel = CancellationToken::new();

    let batch = consult(&config, &[BaseMessage::user(PROBLEM)], &cancel).await?;
    ensure!(
        batch.total_models == 2 && batch.success_count == 2,
        "both references must answer: {batch:?}"
    );
    ensure!(
        batch.sources.len() == 2 && batch.sources[0].index == 1 && batch.sources[1].index == 2,
        "the batch must keep the configured order: {:?}",
        batch.sources
    );
    ensure!(
        batch.failures.is_empty(),
        "a successful batch records no proposer failure"
    );

    // Each proposer was asked exactly once, and the request carried the problem.
    ensure!(
        detailed.call_count() == 1 && terse.call_count() == 1,
        "each reference must be asked once: {} / {}",
        detailed.call_count(),
        terse.call_count()
    );
    let prompt = detailed
        .last_call()
        .ok_or_else(|| anyhow!("the first proposer must record its call"))?;
    ensure!(
        prompt
            .messages
            .iter()
            .any(|message| message.text().contains(PROBLEM)),
        "the consultation request must carry the problem: {:?}",
        prompt.messages
    );

    let ranking = rank_with_config(&config, PROBLEM, &batch.sources).await?;
    ensure!(
        ranking.len() == batch.sources.len(),
        "every retained proposal must be ranked: {ranking:?}"
    );
    for entry in &ranking {
        ensure!(
            (0.0..=1.0).contains(&entry.score),
            "a score must stay in 0..=1, got {}",
            entry.score
        );
        ensure!(
            !entry.reasons.is_empty(),
            "a ranked proposal must explain itself: {entry:?}"
        );
    }
    for pair in ranking.windows(2) {
        let order = pair[0].score.total_cmp(&pair[1].score);
        ensure!(
            order.is_gt() || (order.is_eq() && pair[0].index < pair[1].index),
            "the ranking must descend by score then index: {ranking:?}"
        );
    }
    ensure!(
        ranking[0].index == 1,
        "the structured proposal must outrank the terse one: {ranking:?}"
    );
    Ok(())
}

/// The rail publishes its section into the shared context and records the consultation.
#[tokio::test]
async fn rail_publishes_its_section_and_records_the_outcome() -> Result<()> {
    let (config, _detailed, _terse) = two_proposers();
    let rail = MoaRail::new(config)?;
    ensure!(
        rail.name() == MOA_RAIL_NAME && MOA_PROMPT_SECTION == "moa",
        "the rail must be the `moa` rail"
    );

    let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeInvoke);
    ctx.base.messages = vec![BaseMessage::user(PROBLEM)];
    // `before_invoke` is what installs the per-invocation state and publishes the section.
    rail.before_invoke(&mut ctx).await?;

    // The section travels through the documented prompt-section channel in `ctx.extra`.
    let channel = ctx
        .extra()
        .get("harness.prompt_sections")
        .ok_or_else(|| anyhow!("before_invoke must publish into the section channel"))?;
    ensure!(
        channel.is_object(),
        "the section channel must be a JSON object: {channel}"
    );
    let sections = prompt_sections(&ctx);
    ensure!(
        sections.len() == 1 && sections[0].name == MOA_PROMPT_SECTION,
        "exactly the MoA section must be published: {sections:?}"
    );
    ensure!(
        sections[0].render("en").contains("reference")
            && sections[0].render("en") != sections[0].render("cn"),
        "the section must be bilingual: {}",
        sections[0].render("en")
    );
    let installed = ctx
        .extra()
        .get(MOA_STATE_EXTRA_KEY)
        .ok_or_else(|| anyhow!("the rail must install its invocation state"))?;
    let snapshot: MoaSnapshot = serde_json::from_value(installed.clone())?;
    ensure!(
        snapshot.reference_calls == 0
            && snapshot.verifier_calls == 0
            && snapshot.latest_attempt.is_none(),
        "a fresh invocation state starts unspent: {snapshot:?}"
    );

    let outcome = rail
        .consult(&ctx.base.messages, &CancellationToken::new())
        .await?;
    ensure!(
        outcome.success_count == 2 && outcome.ranking.len() == 2,
        "the rail consultation must rank both proposals: {outcome:?}"
    );
    ensure!(
        outcome
            .aggregate
            .as_deref()
            .is_some_and(|answer| answer.contains("reference") || answer.contains("Integrated")),
        "a labelled aggregate must be produced: {:?}",
        outcome.aggregate
    );
    ensure!(
        rail.last_outcome().as_ref() == Some(&outcome),
        "the rail must remember its last outcome"
    );
    ensure!(
        rail.history().len() == 1 && rail.history()[0].problem == PROBLEM,
        "one bounded history record must be appended: {:?}",
        rail.history()
    );

    // The next model call publishes the outcome for the host, and the invocation retires the
    // per-invocation state.
    ctx.base.event = RailEvent::BeforeModelCall;
    rail.before_model_call(&mut ctx).await?;
    let published: Value = ctx
        .extra()
        .get(MOA_OUTCOME_EXTRA_KEY)
        .cloned()
        .ok_or_else(|| anyhow!("the outcome must be published to the context"))?;
    ensure!(
        published.get("success_count") == Some(&serde_json::json!(2)),
        "the published outcome must be the recorded one: {published}"
    );
    ctx.base.event = RailEvent::AfterInvoke;
    rail.after_invoke(&mut ctx).await?;
    ensure!(
        rail.snapshot().is_none() && !ctx.extra().contains_key(MOA_STATE_EXTRA_KEY),
        "after_invoke must retire the invocation state"
    );
    Ok(())
}

/// Builds a configuration whose two references both fail.
///
/// A fresh model per configuration is required: a `ScriptedModel` falls back to its default
/// answer once its script is spent, so a reused handle would stop failing.
fn failing_proposers() -> MoaConfig {
    MoaConfig::new(vec![
        reference(ScriptedModel::new([ScriptedTurn::fail("first boom")])),
        reference(ScriptedModel::new([ScriptedTurn::fail("second boom")])),
    ])
}

/// A batch in which every proposer fails is an `Error::Llm`, never an empty outcome.
#[tokio::test]
async fn all_proposers_failing_is_a_typed_error() -> Result<()> {
    let config = failing_proposers();
    let messages = [BaseMessage::user(PROBLEM)];

    let error = consult(&config, &messages, &CancellationToken::new())
        .await
        .err()
        .ok_or_else(|| anyhow!("a fully failed batch must not succeed"))?;
    ensure!(
        error.kind() == ErrorKind::Llm,
        "an all-fail batch must be an LLM error: {error:?}"
    );
    ensure!(
        error.to_string().contains("All reference models failed"),
        "the message must name the batch failure: {error}"
    );
    ensure!(
        error.to_string().contains("first boom"),
        "the message must carry the first proposer detail: {error}"
    );

    // The rail reports the same typed refusal instead of storing a partial outcome.
    let rail = MoaRail::new(failing_proposers())?;
    let rail_error = rail
        .consult(&messages, &CancellationToken::new())
        .await
        .err()
        .ok_or_else(|| anyhow!("the rail must propagate the proposer failure"))?;
    ensure!(
        rail_error.kind() == ErrorKind::Llm,
        "unexpected rail error: {rail_error:?}"
    );
    ensure!(
        rail.last_outcome().is_none() && rail.history().is_empty(),
        "a failed consultation must record nothing"
    );
    Ok(())
}

/// An unusable configuration is refused with the Python `__post_init__` message.
#[test]
fn a_configuration_without_references_is_refused() -> Result<()> {
    let error = MoaRail::new(MoaConfig::new(Vec::new()))
        .err()
        .ok_or_else(|| anyhow!("a reference-less configuration must be refused"))?;
    ensure!(
        error.kind() == ErrorKind::Config,
        "unexpected error kind: {error:?}"
    );
    ensure!(
        error
            .to_string()
            .contains("MoA requires at least one reference model"),
        "the message must mirror Python: {error}"
    );
    Ok(())
}
