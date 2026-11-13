#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Cross-module integration suite for the runner's resource managers.
//!
//! Every case builds its own `ResourceMgr::new()` and drives only the public
//! `openjiuwen_runner::resources` surface: the tag facade over the four run-able
//! registries, the model factory registry, prompt templates, system operations,
//! and the cross-manager wiring (`is_pristine`, `release`, the `Debug` dump).
//!
//! Nothing here touches the network, the filesystem, or the wall clock. The MCP
//! connection lifecycle lives in `managers_mcp.rs` (it needs a transport fake);
//! this suite only registers MCP servers, which never connects.
//!
//! The suites here sit one level above the in-module unit tests: they prove the
//! managers cooperate (tags over a real registry, `release` resetting everything
//! at once) rather than re-deriving the per-module semantics.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::{BaseMessage, PromptTemplate};
use openjiuwen_foundation::tool::mcp::McpServerConfig;
use openjiuwen_foundation::tool::{FunctionTool, Tool, ToolCard, ToolContext, ToolHandler};
use openjiuwen_runner::resources::{
    ModelCard, RegisterPolicy, ResourceKind, ResourceMgr, Tag, TagMatchStrategy, TagUpdateStrategy,
    GLOBAL_TAG,
};
use openjiuwen_session::Session;
use openjiuwen_sysop::{SysOperation, SysOperationCard};
use serde_json::{json, Value};

// ---------------------------------------------------------------- fixtures --

/// An agent that answers with its own card id.
struct FakeAgent {
    card: AgentCard,
}

#[async_trait]
impl Agent for FakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        Ok(AgentOutput {
            output: format!("answer:{}", self.card.id),
            ..AgentOutput::default()
        })
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }
}

/// An echoing tool whose card carries `id` / `name`.
fn echo_tool(id: &str, name: &str) -> Arc<dyn Tool> {
    let card = ToolCard::new(name, "echoes its inputs", json!({})).with_id(id);
    let handler: ToolHandler = Arc::new(
        |inputs, _ctx: ToolContext| -> BoxFuture<'static, Result<Value>> {
            Box::pin(async move { Ok(inputs) })
        },
    );
    FunctionTool::new(card, handler).boxed()
}

/// Builds an agent card with an explicit id and name.
fn agent_card(id: &str, name: &str) -> AgentCard {
    AgentCard::default().with_id(id).with_name(name)
}

/// An agent provider minting a fresh [`FakeAgent`] per lookup.
fn agent_provider(card: AgentCard) -> Arc<dyn Fn() -> Arc<dyn Agent> + Send + Sync> {
    Arc::new(move || Arc::new(FakeAgent { card: card.clone() }) as Arc<dyn Agent>)
}

/// Registers one agent whose card id and name are both `id`.
fn register_agent(manager: &ResourceMgr, id: &str) {
    let card = agent_card(id, id);
    manager
        .register_agent(card.clone(), agent_provider(card))
        .unwrap();
}

/// A local-mode system operation (the local backends need no network).
fn sys_op(id: &str) -> SysOperation {
    SysOperation::new(SysOperationCard::new(id).with_name("local"))
        .expect("a local sys operation builds offline")
}

/// A model that carries a name so assertions can tell instances apart.
struct FakeModel {
    provider: &'static str,
    model_name: &'static str,
}

/// A [`FakeModel`] reporting `name` as its model name.
fn fake_model(name: &'static str) -> FakeModel {
    FakeModel {
        provider: "test",
        model_name: name,
    }
}

#[async_trait]
impl Model for FakeModel {
    fn provider(&self) -> &str {
        self.provider
    }

    fn model_name(&self) -> &str {
        self.model_name
    }

    async fn invoke(
        &self,
        _messages: &[BaseMessage],
        _request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        Err(Error::unsupported(
            "the fake model serves the managers suite only",
        ))
    }

    fn stream(
        &self,
        _messages: Vec<BaseMessage>,
        _request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        Box::pin(futures::stream::empty())
    }
}

/// Builds the tags of `values` in order.
fn tags(values: &[&str]) -> Vec<Tag> {
    values.iter().map(|value| Tag::new(*value)).collect()
}

/// Builds owned ids from `values`.
fn ids(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

// -------------------------------------------------------------------- tags --

#[test]
fn tags_attach_to_a_registered_tool_and_agent() {
    let manager = ResourceMgr::new();
    manager.register_tool(echo_tool("t1", "echo")).unwrap();
    register_agent(&manager, "a1");

    assert_eq!(
        manager
            .tag(ResourceKind::Tool, "t1", tags(&["a", "b"]))
            .unwrap(),
        tags(&["a", "b"])
    );
    assert_eq!(manager.tags_of(ResourceKind::Tool, "t1"), tags(&["a", "b"]));
    assert!(manager
        .tags()
        .has_resource_tag(ResourceKind::Tool, "t1", &Tag::new("a")));
    assert!(!manager
        .tags()
        .has_resource_tag(ResourceKind::Tool, "t1", &Tag::new("b2")));
    assert_eq!(
        manager
            .tags()
            .resources_with_tag(ResourceKind::Tool, &Tag::new("a")),
        ids(&["t1"])
    );

    // The kind partition: the same tag on an agent stays in its own partition.
    assert_eq!(
        manager
            .tag(ResourceKind::Agent, "a1", tags(&["a"]))
            .unwrap(),
        tags(&["a"])
    );
    assert_eq!(
        manager
            .tags()
            .resources_with_tag(ResourceKind::Tool, &Tag::new("a")),
        ids(&["t1"]),
        "the agent association is invisible to the tool partition"
    );
    assert_eq!(
        manager
            .tags()
            .resources_with_tag(ResourceKind::Agent, &Tag::new("a")),
        ids(&["a1"])
    );
    assert_eq!(manager.tags().list_tags(), tags(&["a", "b"]));
    assert_eq!(manager.tag_revision(), 2);
    assert_eq!(manager.tag_revision(), manager.tags().revision());
}

#[test]
fn any_matches_the_union_and_all_the_intersection() {
    let manager = ResourceMgr::new();
    manager.register_tool(echo_tool("t1", "echo")).unwrap();
    manager.register_tool(echo_tool("t2", "other")).unwrap();
    manager
        .tag(ResourceKind::Tool, "t1", tags(&["a", "b"]))
        .unwrap();
    manager
        .tag(ResourceKind::Tool, "t2", tags(&["b", "c"]))
        .unwrap();

    assert_eq!(
        manager
            .resources_by_tags(
                ResourceKind::Tool,
                &tags(&["a", "c"]),
                TagMatchStrategy::Any
            )
            .unwrap(),
        ids(&["t1", "t2"])
    );
    assert_eq!(
        manager
            .resources_by_tags(
                ResourceKind::Tool,
                &tags(&["a", "b"]),
                TagMatchStrategy::All
            )
            .unwrap(),
        ids(&["t1"])
    );
    assert_eq!(
        manager
            .resources_by_tags(
                ResourceKind::Tool,
                &tags(&["a", "c"]),
                TagMatchStrategy::All
            )
            .unwrap(),
        Vec::<String>::new(),
        "no tool carries both"
    );
    assert_eq!(
        manager
            .resources_by_tags(ResourceKind::Tool, &tags(&["b"]), TagMatchStrategy::All)
            .unwrap(),
        ids(&["t1", "t2"])
    );
    // A tag nobody carries contributes nothing rather than erroring.
    assert_eq!(
        manager
            .resources_by_tags(ResourceKind::Tool, &tags(&["ghost"]), TagMatchStrategy::Any)
            .unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn the_facade_rejects_unknown_ids_and_empty_tag_lists() {
    let manager = ResourceMgr::new();
    manager.register_tool(echo_tool("t1", "echo")).unwrap();

    let error = manager
        .tag(ResourceKind::Tool, "ghost", tags(&["a"]))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: tool 'ghost' is not registered"
    );

    let error = manager
        .update_tags(
            ResourceKind::Agent,
            "ghost",
            tags(&["a"]),
            TagUpdateStrategy::Merge,
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: agent 'ghost' is not registered"
    );

    // An empty list is a validation failure once the id resolves.
    let error = manager
        .tag(ResourceKind::Tool, "t1", Vec::<Tag>::new())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: at least one tag is required"
    );
    let error = manager
        .resources_by_tags(ResourceKind::Tool, &[], TagMatchStrategy::Any)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "validation: at least one tag is required"
    );

    assert_eq!(manager.tag_revision(), 0, "no rejected call mutated");
    assert!(manager.tags().is_empty());
}

#[test]
fn merge_replace_global_and_removal_end_states() {
    let manager = ResourceMgr::new();
    manager.register_tool(echo_tool("t1", "echo")).unwrap();
    register_agent(&manager, "a1");

    manager
        .tag(ResourceKind::Tool, "t1", tags(&["a", "b"]))
        .unwrap();
    // Merge unions, Replace swaps the whole set.
    assert_eq!(
        manager
            .update_tags(
                ResourceKind::Tool,
                "t1",
                tags(&["b", "c"]),
                TagUpdateStrategy::Merge,
            )
            .unwrap(),
        tags(&["a", "b", "c"])
    );
    assert_eq!(
        manager
            .update_tags(
                ResourceKind::Tool,
                "t1",
                tags(&["z"]),
                TagUpdateStrategy::Replace,
            )
            .unwrap(),
        tags(&["z"])
    );
    assert!(
        !manager.tags().has_tag(&Tag::new("a")),
        "a replaced tag is garbage-collected"
    );

    // GLOBAL is exclusive and drops the resource's other tags.
    assert_eq!(
        manager
            .tag(ResourceKind::Tool, "t1", [Tag::global()])
            .unwrap(),
        vec![Tag::global()]
    );
    assert!(!manager.tags().has_tag(&Tag::new("z")));
    // Merging into a global resource adds nothing; Replace swaps global out.
    assert_eq!(
        manager
            .update_tags(
                ResourceKind::Tool,
                "t1",
                tags(&["chat"]),
                TagUpdateStrategy::Merge,
            )
            .unwrap(),
        vec![Tag::global()]
    );
    assert_eq!(
        manager
            .update_tags(
                ResourceKind::Tool,
                "t1",
                tags(&["chat"]),
                TagUpdateStrategy::Replace,
            )
            .unwrap(),
        tags(&["chat"])
    );
    assert!(!manager.tags().has_tag(&Tag::global()));

    // untag clears one resource; remove_tag strips a tag from every resource.
    assert_eq!(manager.untag(ResourceKind::Tool, "t1"), tags(&["chat"]));
    assert_eq!(manager.untag(ResourceKind::Tool, "t1"), Vec::<Tag>::new());
    manager
        .tag(ResourceKind::Tool, "t1", [Tag::global()])
        .unwrap();
    manager
        .tag(ResourceKind::Agent, "a1", [Tag::global()])
        .unwrap();
    assert_eq!(manager.remove_tag(&Tag::global()), ids(&["a1", "t1"]));
    assert!(
        !manager.tags().has_tag(&Tag::global()),
        "GLOBAL is collected like any other tag"
    );
    assert_eq!(GLOBAL_TAG, "__global__");
}

#[test]
fn removing_or_clearing_a_resource_drops_its_tags() {
    let manager = ResourceMgr::new();
    manager.register_tool(echo_tool("t1", "echo")).unwrap();
    manager.register_tool(echo_tool("t2", "other")).unwrap();
    register_agent(&manager, "a1");
    manager.tag(ResourceKind::Tool, "t1", tags(&["x"])).unwrap();
    manager.tag(ResourceKind::Tool, "t2", tags(&["y"])).unwrap();
    manager
        .tag(ResourceKind::Agent, "a1", tags(&["z"]))
        .unwrap();
    let before = manager.tag_revision();
    assert_eq!(before, 3);

    assert_eq!(manager.remove_agent("a1").unwrap().id, "a1");
    assert_eq!(
        manager.tags_of(ResourceKind::Agent, "a1"),
        Vec::<Tag>::new()
    );
    assert!(manager.tag_revision() > before, "the cleanup is counted");
    assert!(!manager.tags().has_tag(&Tag::new("z")));

    assert_eq!(manager.clear_tools(), 2);
    assert_eq!(
        manager.tags().resources(ResourceKind::Tool),
        Vec::<String>::new()
    );
    assert_eq!(manager.tags().list_tags(), Vec::<Tag>::new());

    // release empties the index and zeroes its counter.
    manager
        .tags()
        .tag(ResourceKind::Tool, "ghost", tags(&["keep"]))
        .unwrap();
    assert!(!manager.tags().is_empty());
    manager.release();
    assert!(manager.tags().is_empty());
    assert_eq!(manager.tag_revision(), 0);
    assert!(manager.is_pristine());
}

#[test]
fn is_pristine_is_stricter_than_is_empty_for_tag_only_state() {
    let manager = ResourceMgr::new();
    assert!(manager.is_empty() && manager.is_pristine());

    // A tag on an id no registry holds: `is_empty` looks at the four run-able
    // registries only, `is_pristine` at every sub-manager.
    manager
        .tags()
        .tag(ResourceKind::Tool, "t1", tags(&["a"]))
        .unwrap();
    assert!(manager.is_empty());
    assert!(!manager.is_pristine());
}

// ------------------------------------------------------------------ models --

#[test]
fn a_registered_model_resolves_with_its_card() {
    let manager = ResourceMgr::new();
    let model = Arc::new(fake_model("gpt"));
    manager
        .models()
        .register_model("gpt", Arc::clone(&model) as Arc<dyn Model>)
        .unwrap();

    let resolved = manager.models().model("gpt").expect("registered");
    assert!(
        Arc::ptr_eq(&resolved, &(Arc::clone(&model) as Arc<dyn Model>)),
        "a live registration hands back the same instance"
    );
    assert_eq!(resolved.model_name(), "gpt");
    assert_eq!(manager.models().model_ids(), ids(&["gpt"]));
    assert_eq!(
        manager.models().models(),
        vec![ModelCard::new("gpt")],
        "registration order and the id-as-name card"
    );
    assert_eq!(manager.models().len(), 1);
    assert!(!manager.models().is_empty());
    assert_eq!(
        manager.models().model_card("gpt"),
        Some(ModelCard::new("gpt"))
    );

    assert_eq!(
        manager.models().remove_model("gpt"),
        Some(ModelCard::new("gpt"))
    );
    assert!(manager.models().model("gpt").is_none());
    assert!(manager.models().remove_model("gpt").is_none());
    assert_eq!(manager.models().revision(), 2);
}

#[test]
fn a_model_factory_runs_once_per_lookup() {
    let manager = ResourceMgr::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let provider: Arc<dyn Fn() -> Arc<dyn Model> + Send + Sync> = Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Arc::new(fake_model("gpt")) as Arc<dyn Model>
    });
    manager
        .models()
        .register_model_with("gpt", provider, RegisterPolicy::Fail)
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "registration invokes nothing"
    );

    let first = manager.models().model("gpt").expect("registered");
    let second = manager.models().model("gpt").expect("registered");
    assert_eq!(calls.load(Ordering::SeqCst), 2, "one call per lookup");
    assert!(
        !Arc::ptr_eq(&first, &second),
        "each lookup gets a fresh model"
    );

    assert!(manager.models().model("missing").is_none());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a missing id invokes nothing"
    );
}

#[test]
fn duplicate_model_ids_follow_the_policy_and_revision_counts() {
    let manager = ResourceMgr::new();
    manager
        .models()
        .register_model("gpt", Arc::new(fake_model("first")) as Arc<dyn Model>)
        .unwrap();
    assert_eq!(manager.models().revision(), 1);

    let error = manager
        .models()
        .register_model("gpt", Arc::new(fake_model("second")) as Arc<dyn Model>)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: model 'gpt' is already registered"
    );
    assert_eq!(
        manager.models().revision(),
        1,
        "the rejection did not mutate"
    );

    let registration = manager
        .models()
        .register_model_with(
            "gpt",
            Arc::new(|| Arc::new(fake_model("second")) as Arc<dyn Model>),
            RegisterPolicy::Replace,
        )
        .unwrap();
    assert!(registration.replaced);
    assert_eq!(
        manager.models().model("gpt").expect("swapped").model_name(),
        "second"
    );
    assert_eq!(manager.models().len(), 1);
    assert_eq!(manager.models().revision(), 2);
}

// ----------------------------------------------------------------- prompts --

#[test]
fn prompts_store_look_up_and_overwrite() {
    let manager = ResourceMgr::new();
    manager
        .prompts()
        .add_prompt("greeting", PromptTemplate::new("hello {{name}}"))
        .unwrap();
    assert_eq!(
        manager.prompts().prompt("greeting"),
        Some(PromptTemplate::new("hello {{name}}"))
    );
    assert!(manager.prompts().prompt("missing").is_none());
    assert_eq!(manager.prompts().revision(), 1);

    // Re-adding the same id overwrites silently and counts as a mutation.
    manager
        .prompts()
        .add_prompt("greeting", PromptTemplate::new("hi {{name}}"))
        .unwrap();
    assert_eq!(
        manager.prompts().prompt("greeting"),
        Some(PromptTemplate::new("hi {{name}}"))
    );
    assert_eq!(manager.prompts().len(), 1);
    assert_eq!(manager.prompts().revision(), 2);

    manager
        .prompts()
        .add_prompts(vec![
            ("first".to_owned(), PromptTemplate::new("one")),
            ("second".to_owned(), PromptTemplate::new("two")),
        ])
        .unwrap();
    assert_eq!(
        manager.prompts().prompt_ids(),
        ids(&["greeting", "first", "second"])
    );
    assert_eq!(manager.prompts().len(), 3);
    assert!(!manager.prompts().is_empty());
}

#[test]
fn prompt_removal_validation_and_serde() {
    let manager = ResourceMgr::new();
    manager
        .prompts()
        .add_prompt("greeting", PromptTemplate::new("hello"))
        .unwrap();

    assert_eq!(
        manager.prompts().remove_prompt("greeting"),
        Some(PromptTemplate::new("hello"))
    );
    assert!(manager.prompts().prompt("greeting").is_none());
    assert_eq!(manager.prompts().remove_prompt("greeting"), None);

    for blank in ["", "   ", "\t\n"] {
        let error = manager
            .prompts()
            .add_prompt(blank, PromptTemplate::new("ghost"))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: prompt id cannot be empty or whitespace only"
        );
        // The reads take a plain `&str` and never validate: a blank id simply
        // finds nothing (see `PromptMgr::prompt` / `PromptMgr::remove_prompt`).
        assert!(manager.prompts().prompt(blank).is_none());
        assert!(manager.prompts().remove_prompt(blank).is_none());
    }
    assert!(manager.prompts().is_empty());

    let template = PromptTemplate::named("greeting", "hello {{name}}");
    let value = serde_json::to_value(&template).unwrap();
    let restored: PromptTemplate = serde_json::from_value(value).unwrap();
    assert_eq!(restored, template);
}

// ---------------------------------------------------------- sys operations --

#[test]
fn sys_operations_register_and_expose_their_tool_specs() {
    let manager = ResourceMgr::new();
    manager.sysops().register(sys_op("sys_op")).unwrap();

    let service = manager.sysops().get("sys_op").expect("registered");
    assert_eq!(service.card().id, "sys_op");
    assert_eq!(manager.sysops().card("sys_op").unwrap().name, "local");
    assert_eq!(manager.sysops().ids(), ids(&["sys_op"]));
    assert_eq!(manager.sysops().len(), 1);
    assert!(!manager.sysops().is_empty());

    let specs = manager.sysops().tool_specs("sys_op").unwrap();
    assert_eq!(specs.len(), 13);
    assert_eq!(specs[0].tool_id, "sys_op.fs.read_file");

    let error = manager.sysops().register(sys_op("sys_op")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: sys_operation 'sys_op' is already registered"
    );
    assert_eq!(manager.sysops().revision(), 1);
}

#[test]
fn removing_a_sys_operation_drops_lookup_and_tool_specs() {
    let manager = ResourceMgr::new();
    manager.sysops().register(sys_op("sys_op")).unwrap();

    assert_eq!(manager.sysops().remove("sys_op").unwrap().id, "sys_op");
    assert!(manager.sysops().get("sys_op").is_none());
    assert!(manager.sysops().is_empty());

    let error = manager.sysops().tool_specs("sys_op").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: sys_operation 'sys_op' is not registered"
    );
}

// ---------------------------------------------------------- wiring/lifecycle --

#[test]
fn every_manager_reports_through_is_pristine_and_release() {
    let manager = ResourceMgr::new();
    assert!(manager.is_empty() && manager.is_pristine());

    manager
        .tags()
        .tag(ResourceKind::Tool, "t1", tags(&["a"]))
        .unwrap();
    manager
        .models()
        .register_model("gpt", Arc::new(fake_model("gpt")) as Arc<dyn Model>)
        .unwrap();
    manager
        .prompts()
        .add_prompt("greeting", PromptTemplate::new("hello"))
        .unwrap();
    manager.sysops().register(sys_op("sys_op")).unwrap();
    manager
        .mcp_servers()
        .register(McpServerConfig::stdio("srv", "cmd").with_server_id("srv-1"))
        .unwrap();

    // Nothing lives in the four run-able registries, so `is_empty` holds while
    // the sub-managers make the manager impure.
    assert!(manager.is_empty());
    assert!(!manager.is_pristine());

    manager.release();

    assert!(manager.is_empty() && manager.is_pristine());
    for len in [
        manager.tags().len(),
        manager.models().len(),
        manager.prompts().len(),
        manager.sysops().len(),
        manager.mcp_servers().len(),
    ] {
        assert_eq!(len, 0);
    }
    for revision in [
        manager.tag_revision(),
        manager.models().revision(),
        manager.prompts().revision(),
        manager.sysops().revision(),
        manager.mcp_servers().revision(),
        manager.tool_revision(),
        manager.agent_revision(),
        manager.workflow_revision(),
        manager.team_revision(),
    ] {
        assert_eq!(revision, 0);
    }
}

#[test]
fn the_debug_dump_summarises_every_manager_without_auth_material() {
    let manager = ResourceMgr::new();
    manager.register_tool(echo_tool("t1", "echo")).unwrap();
    register_agent(&manager, "a1");
    manager
        .tag(ResourceKind::Tool, "t1", tags(&["prod"]))
        .unwrap();
    manager
        .models()
        .register_model("gpt", Arc::new(fake_model("gpt")) as Arc<dyn Model>)
        .unwrap();
    manager
        .prompts()
        .add_prompt("greeting", PromptTemplate::new("hello"))
        .unwrap();
    manager.sysops().register(sys_op("sys_op")).unwrap();
    manager
        .mcp_servers()
        .register(
            McpServerConfig::stdio("srv", "cmd")
                .with_server_id("srv-1")
                .with_auth_header("authorization", "Bearer super-secret"),
        )
        .unwrap();

    let rendered = format!("{manager:?}");
    for field in [
        "tools",
        "agents",
        "workflows",
        "teams",
        "tags",
        "models",
        "prompts",
        "sysops",
        "mcp_servers",
    ] {
        assert!(rendered.contains(field), "missing {field}: {rendered}");
    }
    assert!(rendered.contains("srv-1"), "{rendered}");
    assert!(rendered.contains("Registered"), "{rendered}");
    assert!(
        !rendered.contains("super-secret"),
        "the auth header value must never be rendered: {rendered}"
    );
    assert!(
        !rendered.contains("auth_header"),
        "the auth header names must never be rendered: {rendered}"
    );
}
