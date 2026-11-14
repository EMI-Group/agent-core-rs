#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! KV-cache seam tests for the implemented `ReAct` loop.
//!
//! `openjiuwen-agent`'s `kv_cache` module implements prefix affinity and the
//! provider-action runtime; these tests pin the wiring that makes it reachable
//! from the agent facade:
//!
//! - `ReActAgent::enable_kv_cache` installs a hook from a runtime config, and
//!   with the default (disabled) affinity the hook is inert: a run records
//!   nothing and its output is unchanged;
//! - with affinity enabled, a run records the request prefix, so the runtime's
//!   own `lookup` afterwards hits and the entry carries the provider's input
//!   tokens;
//! - a hook built with `with_action_runtime` prefetches the request scope and
//!   releases its inference lease, without any failure reaching the output;
//! - a deferred backend (`Redis`, `Distributed`) surfaces its typed
//!   `Error::Unsupported` naming the backend and leaves an installed hook (and,
//!   therefore, its recorded prefixes) untouched;
//! - a run with no hook installed behaves exactly as before;
//! - the controller and KV-cache facade types are reachable from the crate root.
//!
//! Everything is offline and deterministic: models are `ScriptedModel`s,
//! sessions and runtimes are in-process, and no assertion depends on wall-clock
//! time.
mod common;
use openjiuwen_agent::kv_cache::{KVCacheBackend, KvCacheAction};
use openjiuwen_agent::{
    Agent, AgentCard, BaseController, Controller, ControllerConfig, GroupRouter,
    InProcessKVCacheRuntime, InProcessKvCacheActionProvider, KVCacheAffinityConfig, KVCacheEntry,
    KVCacheIdentity, KVCacheModelCallHook, KVCacheRuntimeConfig, KVCacheRuntimeProtocol,
    KvCacheActionProvider, KvCacheActionRuntime, KvCacheAdmission, KvCacheInferenceLease,
    ReActAgent, ReActAgentConfig, TaskManager, TaskQueue, TaskSchedulerLoop, TaskSchedulerState,
};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{Model, Usage};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;
use std::sync::Arc;
/// A runtime config whose affinity is switched on: the default config keeps
/// affinity disabled, so its hook records nothing.
fn enabled() -> KVCacheRuntimeConfig {
    KVCacheRuntimeConfig::default().with_affinity(KVCacheAffinityConfig::enabled())
}
/// Hands a clone of `model` to the agent while the test keeps the original for
/// inspecting the recorded calls.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}
/// Rebuilds the identity the loop used for its last model call.
///
/// The loop records the whole window with the request's model name and the
/// model-visible tool cards (none in these fixtures), which is exactly what
/// `KVCacheModelCallHook::identity` derives from a recorded call.
fn last_call_identity(hook: &KVCacheModelCallHook, model: &ScriptedModel) -> KVCacheIdentity {
    let call = model.last_call().expect("the run called the model");
    hook.identity(
        common::SESSION_ID,
        None,
        &call.request.model_name,
        &call.messages,
        &[],
    )
}

#[tokio::test]
async fn enable_kv_cache_installs_an_inert_hook_with_the_default_config() {
    let model = ScriptedModel::builder().respond_text("hello back").build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    agent
        .enable_kv_cache(KVCacheRuntimeConfig::default())
        .expect("the in-process backend always builds");
    let hook = agent
        .kv_cache_hook()
        .expect("enable_kv_cache installs a hook");
    assert!(!hook.capabilities().affinity);
    assert!(!hook.capabilities().offload);

    let output = agent
        .invoke(json!("hi"), Some(common::session(common::SESSION_ID)))
        .await
        .expect("the run answers");
    assert_eq!(output.output, "hello back");
    // Disabled affinity is a full bypass: nothing is recorded.
    assert_eq!(hook.stats().await.unwrap().entries, 0);
}

#[tokio::test]
async fn enabled_affinity_records_the_request_prefix_and_its_tokens() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::text("hello back").with_usage(Usage::new(2, 3)))
        .build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    agent
        .enable_kv_cache(enabled())
        .expect("the in-process backend always builds");
    let hook = agent
        .kv_cache_hook()
        .expect("enable_kv_cache installs a hook");
    assert!(hook.capabilities().affinity);

    let output = agent
        .invoke(json!("hi"), Some(common::session(common::SESSION_ID)))
        .await
        .expect("the run answers");
    assert_eq!(output.output, "hello back");

    // The loop recorded the request it sent, so the runtime now holds it and a
    // later lookup of the same request is a hit with the provider's token count.
    let identity = last_call_identity(&hook, &model);
    let lookup = hook.runtime().lookup(&identity).await.unwrap();
    assert!(lookup.is_hit());
    let entry = lookup.entry.expect("a hit carries its entry");
    assert_eq!(entry.token_estimate, 2);
    assert_eq!(entry.prefix_len, identity.message_count);
    assert_eq!(hook.stats().await.unwrap().entries, 1);
}

#[tokio::test]
async fn an_attached_action_runtime_prefetches_and_releases_its_lease() {
    let model = ScriptedModel::builder().respond_text("hello back").build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    let provider = Arc::new(InProcessKvCacheActionProvider::default());
    let hook = Arc::new(
        KVCacheModelCallHook::new(
            enabled().build_runtime().unwrap(),
            KVCacheAffinityConfig::enabled(),
        )
        .with_action_runtime(Arc::new(KvCacheActionRuntime::new(
            provider.clone(),
            KVCacheAffinityConfig::enabled(),
        ))),
    );
    agent.set_kv_cache_hook(Some(Arc::clone(&hook)));
    assert!(hook.capabilities().offload);

    let output = agent
        .invoke(json!("hi"), Some(common::session(common::SESSION_ID)))
        .await
        .expect("a provider-backed run answers");
    // Provider-side actions never change the run's output.
    assert_eq!(output.output, "hello back");

    // The first call opens a fresh scope, so the runtime prefetched it.
    assert_eq!(provider.calls(KvCacheAction::Prefetch), 1);
    // The loop released the inference lease it was granted.
    let action = hook.action_runtime().expect("the runtime is attached");
    let identity = last_call_identity(&hook, &model);
    assert_eq!(action.active_inference(&identity), 0);
    assert!(
        action.binding(&identity).is_some(),
        "the loop registered the request scope it admitted"
    );
}

#[test]
fn a_deferred_backend_surfaces_a_typed_error_and_keeps_the_installed_hook() {
    let agent = ReActAgent::new(AgentCard::new("react"));
    agent.enable_kv_cache(enabled()).unwrap();
    let installed = agent
        .kv_cache_hook()
        .expect("the in-process hook is installed");

    for (backend, name) in [
        (KVCacheBackend::Redis, "redis"),
        (KVCacheBackend::Distributed, "distributed"),
    ] {
        let error = agent
            .enable_kv_cache(KVCacheRuntimeConfig::default().with_backend(backend))
            .expect_err("a deferred backend cannot build a runtime");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.message().contains(name),
            "the message names the backend: {error}"
        );
        // The runtime is built before anything is stored, so the failing call
        // left the previously installed hook (and its prefixes) in place.
        let current = agent.kv_cache_hook().expect("the old hook survives");
        assert!(Arc::ptr_eq(&installed, &current));
    }

    agent.clear_kv_cache();
    assert!(agent.kv_cache_hook().is_none());
}

#[tokio::test]
async fn a_run_without_a_hook_is_unchanged() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("echo", json!({"value": "hi"})))
        .respond_text("final answer")
        .build();
    let log = common::log();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    agent.add_tool(common::recording_tool("echo", &log));
    assert!(agent.kv_cache_hook().is_none());

    let output = agent
        .invoke(json!("hi"), Some(common::session(common::SESSION_ID)))
        .await
        .expect("the tool round answers");
    assert_eq!(output.output, "final answer");
    assert_eq!(output.observations.len(), 1);
    assert_eq!(output.observations[0].content, "tool-output");
    assert_eq!(log.lock().len(), 1);
}

#[test]
fn controller_and_kv_cache_types_are_reachable_from_the_crate_root() {
    // The facade types a host needs without deep module paths. Naming them in
    // type position is the assertion: an unexported name fails to compile here.
    let _: Option<Arc<dyn Controller>> = None;
    let _: Option<BaseController> = None;
    let _: Option<Arc<dyn GroupRouter>> = None;
    let _: Option<TaskSchedulerLoop> = None;
    let _: Option<TaskSchedulerState> = None;
    let _: Option<Arc<dyn KvCacheActionProvider>> = None;
    let _: Option<Arc<dyn KVCacheRuntimeProtocol>> = None;
    let _: Option<KvCacheInferenceLease> = None;

    let config = ControllerConfig::new();
    assert_eq!(config.default_task_type, "agent");
    assert!(TaskManager::new().is_empty());
    assert!(TaskQueue::new().is_empty());

    let identity = KVCacheIdentity::new("s1", "hash", "gpt-test").with_parent("s1");
    let entry = KVCacheEntry::new(identity.clone(), 1).with_token_estimate(9);
    assert_eq!(entry.token_estimate, 9);
    assert_eq!(KvCacheAdmission::Open, KvCacheAdmission::Open);
    assert_ne!(KvCacheAdmission::Open, KvCacheAdmission::Terminal);

    let runtime = KVCacheRuntimeConfig::default()
        .with_affinity(KVCacheAffinityConfig::default())
        .build_runtime()
        .unwrap();
    let hook = KVCacheModelCallHook::new(runtime, KVCacheAffinityConfig::default());
    assert!(!hook.capabilities().affinity);
    let _in_process: Arc<dyn KVCacheRuntimeProtocol> = Arc::new(InProcessKVCacheRuntime::new(2));
    let action = KvCacheActionRuntime::new(
        Arc::new(InProcessKvCacheActionProvider::default()),
        KVCacheAffinityConfig::default(),
    );
    assert!(!action.fail_open(&identity));
}
