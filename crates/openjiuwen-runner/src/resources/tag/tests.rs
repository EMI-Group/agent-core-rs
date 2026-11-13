//! Unit tests for the tag index ([`TagMgr`]) and the [`ResourceMgr`] tag facade.
//!
//! The suite covers the index on its own (the `TagMgr` cases) and the facade
//! layered on top of it (the `the_facade_*` cases), which additionally proves
//! that the resource manager rejects ids it has not registered.

use super::*;
use crate::resources::ResourceMgr;
use futures::future::BoxFuture;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::tool::mcp::McpServerConfig;
use openjiuwen_foundation::tool::{FunctionTool, ToolCard, ToolContext, ToolHandler};
use serde_json::{json, Value};
use std::sync::Arc;
use ResourceKind::{Agent, McpServer, Team, Tool, Workflow};
use TagMatchStrategy::{All, Any};
use TagUpdateStrategy::{Merge, Replace};
fn tag(value: &str) -> Tag {
    Tag::new(value)
}

fn tags(values: &[&str]) -> Vec<Tag> {
    values.iter().map(|value| tag(value)).collect()
}

fn ids(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn mgr() -> TagMgr {
    TagMgr::new()
}

fn tagged(manager: &TagMgr, kind: ResourceKind, id: &str, values: &[&str]) -> Vec<Tag> {
    manager.tag(kind, id, tags(values)).unwrap()
}

fn updated(
    manager: &TagMgr,
    kind: ResourceKind,
    id: &str,
    values: &[&str],
    strategy: TagUpdateStrategy,
) -> Vec<Tag> {
    manager
        .update_tags(kind, id, tags(values), strategy)
        .unwrap()
}

fn found(
    manager: &TagMgr,
    kind: ResourceKind,
    values: &[&str],
    strategy: TagMatchStrategy,
) -> Vec<String> {
    manager
        .resources_by_tags(kind, &tags(values), strategy)
        .unwrap()
}

fn removed(
    manager: &TagMgr,
    kind: ResourceKind,
    id: &str,
    values: &[&str],
    skip: bool,
) -> Vec<Tag> {
    manager.remove_tags(kind, id, &tags(values), skip).unwrap()
}

#[test]
fn tag_serializes_as_a_bare_json_string() {
    assert_eq!(serde_json::to_value(tag("chat")).unwrap(), json!("chat"));
    let parsed: Tag = serde_json::from_value(json!("chat")).unwrap();
    assert_eq!(parsed, tag("chat"));
    let global: Tag = serde_json::from_value(json!(GLOBAL_TAG)).unwrap();
    assert_eq!(global, Tag::global());
}

#[test]
fn resource_kind_round_trips_through_serde() {
    let expected = [
        (Tool, "tool"),
        (Agent, "agent"),
        (Workflow, "workflow"),
        (Team, "team"),
        (McpServer, "mcp_server"),
    ];
    assert_eq!(
        ResourceKind::ALL.map(|kind| kind.as_str()),
        expected.map(|(_, name)| name)
    );
    for (kind, name) in expected {
        assert_eq!(serde_json::to_value(kind).unwrap(), json!(name));
        let parsed: ResourceKind = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(parsed, kind);
        assert_eq!(kind.as_str(), name);
        assert_eq!(kind.to_string(), name);
    }
}

#[test]
fn strategies_round_trip_through_serde() {
    for (strategy, name) in [(Any, "any"), (All, "all")] {
        assert_eq!(serde_json::to_value(strategy).unwrap(), json!(name));
        let parsed: TagMatchStrategy = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(parsed, strategy);
    }
    for (strategy, name) in [(Merge, "merge"), (Replace, "replace")] {
        assert_eq!(serde_json::to_value(strategy).unwrap(), json!(name));
        let parsed: TagUpdateStrategy = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(parsed, strategy);
    }
}

#[test]
fn tag_conversions_and_display_expose_the_bare_value() {
    let borrowed: Tag = "chat".into();
    let owned: Tag = String::from("chat").into();
    assert_eq!(borrowed, owned);
    assert_eq!(borrowed, Tag::new("chat"));
    assert_eq!(borrowed.as_str(), "chat");
    assert_eq!(borrowed.to_string(), "chat");
    assert_eq!(&*borrowed, "chat");
    assert!(!borrowed.is_global());

    let mut sorted = vec![tag("b"), Tag::global(), tag("a")];
    sorted.sort();
    assert_eq!(sorted, vec![Tag::global(), tag("a"), tag("b")]);

    let global = Tag::global();
    assert!(global.is_global());
    assert_eq!(global.as_str(), GLOBAL_TAG);
    assert_eq!(global.to_string(), GLOBAL_TAG);
}

#[test]
fn tagging_records_and_reads_back_the_tags() {
    let manager = mgr();
    let result = tagged(&manager, Agent, "a1", &["chat", "beta"]);
    assert_eq!(result, tags(&["chat", "beta"]));
    assert_eq!(manager.tags_of(Agent, "a1"), result);
    assert!(manager.has_resource_tag(Agent, "a1", &tag("chat")));
    assert!(!manager.has_resource_tag(Agent, "a1", &tag("gamma")));
    assert!(manager.has_tag(&tag("chat")));
    assert!(!manager.has_tag(&tag("gamma")));
    // `list_tags` is sorted, not registration-ordered.
    assert_eq!(manager.list_tags(), tags(&["beta", "chat"]));
    assert_eq!((manager.len(), manager.is_empty()), (2, false));
    assert_eq!(manager.revision(), 1);
    assert_eq!(manager.resources(Agent), ids(&["a1"]));

    // The kind is the index partition, not part of the tag.
    assert_eq!(manager.tags_of(Tool, "a1"), Vec::new());
    assert!(!manager.has_resource_tag(Tool, "a1", &tag("chat")));
    assert_eq!(manager.resources(Tool), Vec::<String>::new());
}

#[test]
fn duplicate_and_already_present_tags_are_recorded_once() {
    let manager = mgr();
    assert_eq!(
        tagged(&manager, Tool, "t1", &["x", "x", "y"]),
        tags(&["x", "y"])
    );
    assert_eq!(manager.revision(), 1);

    assert_eq!(
        tagged(&manager, Tool, "t1", &["x"]),
        tags(&["x", "y"]),
        "a known tag is neither duplicated nor reordered"
    );
    assert_eq!(manager.revision(), 1, "re-tagging is not a mutation");
    assert_eq!(manager.len(), 2);
}

#[test]
fn the_global_tag_drops_every_other_tag_of_the_resource() {
    let manager = mgr();
    tagged(&manager, Agent, "a1", &["chat", "beta"]);
    tagged(&manager, Agent, "a2", &["chat"]);

    assert_eq!(
        manager.tag(Agent, "a1", [Tag::global()]).unwrap(),
        vec![Tag::global()]
    );
    assert_eq!(manager.tags_of(Agent, "a1"), vec![Tag::global()]);
    assert!(
        !manager.has_tag(&tag("beta")),
        "the orphaned tag is collected"
    );
    assert!(manager.has_tag(&tag("chat")), "a2 still carries it");
    assert!(manager.has_resource_tag(Agent, "a1", &Tag::global()));
    assert_eq!(
        manager.resources_with_tag(Agent, &Tag::global()),
        ids(&["a1"])
    );

    // A list that also names GLOBAL still ends up global-only.
    assert_eq!(
        manager
            .tag(Agent, "a2", tags(&["other", GLOBAL_TAG]))
            .unwrap(),
        vec![Tag::global()]
    );
    assert!(!manager.has_tag(&tag("other")));
}

#[test]
fn adding_tags_to_a_global_resource_adds_nothing() {
    let manager = mgr();
    manager.tag(Tool, "t1", [Tag::global()]).unwrap();
    let revision = manager.revision();

    assert_eq!(tagged(&manager, Tool, "t1", &["chat"]), vec![Tag::global()]);
    assert_eq!(
        updated(&manager, Tool, "t1", &["chat"], Merge),
        vec![Tag::global()]
    );
    assert_eq!(manager.tags_of(Tool, "t1"), vec![Tag::global()]);
    assert!(!manager.has_tag(&tag("chat")));
    assert_eq!(manager.revision(), revision, "both calls are no-ops");
}

#[test]
fn replace_on_a_global_resource_swaps_the_global_tag_out() {
    let manager = mgr();
    manager.tag(Tool, "t1", [Tag::global()]).unwrap();

    assert_eq!(
        updated(&manager, Tool, "t1", &["chat"], Replace),
        tags(&["chat"])
    );
    assert_eq!(manager.tags_of(Tool, "t1"), tags(&["chat"]));
    assert!(!manager.has_tag(&Tag::global()));
}

#[test]
fn merge_unions_and_replace_swaps_the_tag_lists() {
    let manager = mgr();
    tagged(&manager, Agent, "a1", &["a", "b"]);

    assert_eq!(
        updated(&manager, Agent, "a1", &["b", "c"], Merge),
        tags(&["a", "b", "c"])
    );
    assert_eq!(
        updated(&manager, Agent, "a1", &["z"], Replace),
        tags(&["z"])
    );
    assert_eq!(manager.tags_of(Agent, "a1"), tags(&["z"]));
    assert_eq!(
        manager.list_tags(),
        tags(&["z"]),
        "a, b, and c are collected"
    );
    assert_eq!(manager.revision(), 3);
}

#[test]
fn update_tags_creates_the_entry_for_an_untagged_resource() {
    let manager = mgr();
    assert_eq!(
        updated(&manager, Workflow, "w1", &["x"], Replace),
        tags(&["x"])
    );
    assert_eq!(
        updated(&manager, Team, "team1", &["y", "x"], Merge),
        tags(&["y", "x"])
    );
    assert_eq!(manager.tags_of(Workflow, "w1"), tags(&["x"]));
    assert_eq!(manager.tags_of(Team, "team1"), tags(&["y", "x"]));
    assert_eq!(manager.revision(), 2);
}

#[test]
fn any_matches_the_union_and_all_matches_the_intersection() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a", "b"]);
    tagged(&manager, Tool, "t2", &["b", "c"]);
    tagged(&manager, Tool, "t3", &["d"]);

    assert_eq!(found(&manager, Tool, &["a", "c"], Any), ids(&["t1", "t2"]));
    assert_eq!(found(&manager, Tool, &["a", "b"], All), ids(&["t1"]));
    // A tag only some resources carry narrows ALL to nothing.
    assert_eq!(
        found(&manager, Tool, &["a", "d"], All),
        Vec::<String>::new()
    );
    // A single-tag ALL query is the carrier list.
    assert_eq!(found(&manager, Tool, &["b"], All), ids(&["t1", "t2"]));
    // Unknown tags are skipped rather than rejected.
    assert_eq!(found(&manager, Tool, &["ghost"], Any), Vec::<String>::new());
    assert_eq!(
        found(&manager, Tool, &["a", "ghost"], All),
        Vec::<String>::new()
    );
}

#[test]
fn resource_lookups_are_sorted_and_partitioned_by_kind() {
    let manager = mgr();
    tagged(&manager, Tool, "t2", &["x"]);
    tagged(&manager, Agent, "t2", &["x"]);
    tagged(&manager, Tool, "t1", &["y"]);

    assert_eq!(manager.resources_with_tag(Tool, &tag("x")), ids(&["t2"]));
    assert_eq!(manager.resources_with_tag(Agent, &tag("x")), ids(&["t2"]));
    assert_eq!(manager.resources(Tool), ids(&["t1", "t2"]));
    assert_eq!(manager.resources(Agent), ids(&["t2"]));

    // Removing one kind's association leaves the other intact.
    assert_eq!(manager.remove_resource(Tool, "t2"), tags(&["x"]));
    assert!(manager.has_tag(&tag("x")));
    assert_eq!(
        manager.resources_with_tag(Tool, &tag("x")),
        Vec::<String>::new()
    );
    assert_eq!(manager.tags_of(Agent, "t2"), tags(&["x"]));
}

#[test]
fn a_tag_is_collected_when_its_last_resource_loses_it() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a", "b"]);

    assert_eq!(removed(&manager, Tool, "t1", &["a"], false), tags(&["b"]));
    assert!(!manager.has_tag(&tag("a")));
    assert!(manager.has_tag(&tag("b")));

    assert_eq!(removed(&manager, Tool, "t1", &["b"], false), Vec::new());
    assert!(!manager.has_tag(&tag("b")));
    assert!(manager.is_empty());
    assert_eq!((manager.len(), manager.list_tags().len()), (0, 0));
    assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
    assert_eq!(manager.revision(), 3);
}

#[test]
fn untag_removes_every_tag_of_one_resource() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a", "b"]);
    tagged(&manager, Tool, "t2", &["a"]);

    assert_eq!(manager.untag(Tool, "t1"), tags(&["a", "b"]));
    assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
    assert!(manager.has_tag(&tag("a")), "t2 still carries it");
    assert!(!manager.has_tag(&tag("b")));

    assert_eq!(manager.untag(Tool, "t1"), Vec::new());
    assert_eq!(manager.untag(Tool, "ghost"), Vec::new());
    assert_eq!(
        manager.revision(),
        3,
        "an unknown resource is not a mutation"
    );
}

#[test]
fn remove_tag_strips_the_tag_from_every_resource() {
    let manager = mgr();
    tagged(&manager, Tool, "t2", &["x", "y"]);
    tagged(&manager, Agent, "t1", &["x"]);

    assert_eq!(manager.remove_tag(&tag("x")), ids(&["t1", "t2"]));
    assert!(!manager.has_tag(&tag("x")));
    assert_eq!(manager.tags_of(Tool, "t2"), tags(&["y"]));
    assert_eq!(manager.tags_of(Agent, "t1"), Vec::new());

    let revision = manager.revision();
    assert_eq!(manager.remove_tag(&tag("x")), Vec::<String>::new());
    assert_eq!(manager.remove_tag(&tag("ghost")), Vec::<String>::new());
    assert_eq!(
        manager.revision(),
        revision,
        "unknown tags are not mutations"
    );
}

#[test]
fn remove_resource_cleans_up_every_tag_of_the_resource() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a", "b"]);
    tagged(&manager, McpServer, "t1", &["a"]);

    assert_eq!(manager.remove_resource(Tool, "t1"), tags(&["a", "b"]));
    assert!(!manager.has_resource_tag(Tool, "t1", &tag("a")));
    assert!(
        manager.has_tag(&tag("a")),
        "the MCP server still carries it"
    );
    assert!(!manager.has_tag(&tag("b")));
    assert_eq!(manager.remove_resource(Tool, "t1"), Vec::new());
}

#[test]
fn remove_tags_reports_missing_tags_and_leaves_the_index_alone() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a"]);

    let error = manager
        .remove_tags(Tool, "t1", &tags(&["a", "b"]), false)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: tool 't1' does not have tag 'b'"
    );
    assert_eq!(manager.tags_of(Tool, "t1"), tags(&["a"]));
    assert_eq!(
        manager.revision(),
        1,
        "a rejected removal is not a mutation"
    );
}

#[test]
fn remove_tags_skips_missing_tags_when_asked() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a", "b"]);

    assert_eq!(
        removed(&manager, Tool, "t1", &["b", "ghost"], true),
        tags(&["a"])
    );
    assert_eq!(manager.revision(), 2);
    assert_eq!(
        removed(&manager, Tool, "t1", &["ghost"], true),
        tags(&["a"]),
        "removing only absent tags is a no-op"
    );
    assert_eq!(manager.revision(), 2);
}

#[test]
fn removing_tags_from_an_unknown_resource_is_not_found_or_a_no_op() {
    let manager = mgr();

    let error = manager
        .remove_tags(Tool, "ghost", &tags(&["a"]), false)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: tool 'ghost' is not registered"
    );
    assert_eq!(removed(&manager, Tool, "ghost", &["a"], true), Vec::new());
    assert_eq!(manager.revision(), 0);
    assert_eq!(manager.tags_of(Tool, "ghost"), Vec::new());
}

#[test]
fn empty_ids_tags_and_lists_are_rejected() {
    let manager = mgr();
    let no_tags = "validation: at least one tag is required";

    assert_eq!(
        manager.tag(Tool, "t1", Vec::new()).unwrap_err().to_string(),
        no_tags
    );
    assert_eq!(
        manager
            .update_tags(Tool, "t1", Vec::new(), Merge)
            .unwrap_err()
            .to_string(),
        no_tags
    );
    assert_eq!(
        manager
            .resources_by_tags(Tool, &[], Any)
            .unwrap_err()
            .to_string(),
        no_tags
    );

    for blank in ["", "   ", "\t\n"] {
        let error = manager.tag(Tool, "t1", [tag(blank)]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: tag cannot be empty or whitespace only"
        );
        assert!(manager
            .remove_tags(Tool, "t1", &[tag(blank)], true)
            .is_err());
    }

    for blank in ["", "  "] {
        let error = manager.tag(Tool, blank, tags(&["a"])).unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: resource id cannot be empty or whitespace only"
        );
        assert_eq!(
            manager
                .remove_tags(Tool, blank, &tags(&["a"]), true)
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }

    assert!(manager.is_empty());
    assert_eq!(
        manager.revision(),
        0,
        "validation failures are not mutations"
    );
}

#[test]
fn revision_counts_mutations_and_reset_zeroes_it() {
    let manager = mgr();
    assert_eq!(manager.revision(), 0);

    tagged(&manager, Tool, "t1", &["a"]);
    tagged(&manager, Tool, "t1", &["a"]);
    assert_eq!(manager.revision(), 1, "the second tag is a no-op");
    tagged(&manager, Tool, "t2", &["a"]);
    assert_eq!(manager.revision(), 2);
    updated(&manager, Tool, "t1", &["a", "b"], Merge);
    assert_eq!(manager.revision(), 3);
    removed(&manager, Tool, "t2", &["a"], false);
    assert_eq!(manager.revision(), 4);
    assert_eq!(manager.remove_resource(Tool, "t1"), tags(&["a", "b"]));
    assert_eq!(manager.revision(), 5);

    tagged(&manager, Tool, "t1", &["a", "b"]);
    assert_eq!(manager.reset(), 2);
    assert_eq!(manager.revision(), 0);
    assert!(manager.is_empty());
    assert_eq!(manager.list_tags(), Vec::new());
}

#[test]
fn clear_removes_the_index_as_one_mutation() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a", "b"]);
    tagged(&manager, Agent, "a1", &["c"]);
    assert_eq!(manager.revision(), 2);

    assert_eq!(manager.clear(), 3);
    assert!(manager.is_empty());
    assert_eq!(manager.revision(), 3);
    assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
    assert_eq!(manager.resources(Agent), Vec::<String>::new());

    assert_eq!(manager.clear(), 0);
    assert_eq!(manager.revision(), 3, "clearing an empty index is a no-op");
}

#[test]
fn debug_output_reports_counts_tags_and_revision() {
    let manager = mgr();
    tagged(&manager, Tool, "t1", &["a"]);
    tagged(&manager, Agent, "a1", &["b"]);

    let rendered = format!("{manager:?}");
    assert!(rendered.contains("TagMgr"), "{rendered}");
    assert!(rendered.contains("(\"agent\", 1)"), "{rendered}");
    assert!(rendered.contains("(\"tool\", 1)"), "{rendered}");
    assert!(rendered.contains("tags: 2"), "{rendered}");
    assert!(rendered.contains("revision: 2"), "{rendered}");
    // The dump never lists tag values or ids.
    assert!(!rendered.contains("\"a\""), "{rendered}");
    assert!(!rendered.contains("t1"), "{rendered}");
}

#[test]
fn default_builds_an_empty_index() {
    let manager = TagMgr::default();
    assert!(manager.is_empty());
    assert_eq!((manager.len(), manager.revision()), (0, 0));
    assert_eq!(manager.list_tags(), Vec::new());
}

// ----------------------------------------------------------- tag facade --

/// An echoing tool with card id `id`, registered through the facade cases.
fn facade_tool(id: &str) -> Arc<dyn openjiuwen_foundation::tool::Tool> {
    let card = ToolCard::new("echo", "echoes its inputs", json!({})).with_id(id);
    let handler: ToolHandler = Arc::new(
        |inputs, _ctx: ToolContext| -> BoxFuture<'static, Result<Value>> {
            Box::pin(async move { Ok(inputs) })
        },
    );
    FunctionTool::new(card, handler).boxed()
}

#[test]
fn the_facade_tags_a_registered_tool() {
    let manager = ResourceMgr::new();
    manager.register_tool(facade_tool("t1")).unwrap();

    assert_eq!(
        manager.tag(Tool, "t1", [tag("prod")]).unwrap(),
        tags(&["prod"])
    );
    assert_eq!(manager.tags_of(Tool, "t1"), tags(&["prod"]));
    assert_eq!(
        manager
            .resources_by_tags(Tool, &tags(&["prod"]), Any)
            .unwrap(),
        ids(&["t1"])
    );
    // The merge strategy is reachable through the facade too.
    assert_eq!(
        manager
            .update_tags(Tool, "t1", tags(&["edge"]), Merge)
            .unwrap(),
        tags(&["prod", "edge"])
    );
    assert_eq!(
        manager
            .resources_by_tags(Tool, &tags(&["prod", "edge"]), All)
            .unwrap(),
        ids(&["t1"])
    );
    // Every mutation is revision-counted on the shared index.
    assert_eq!(manager.tag_revision(), 2);
    assert_eq!(manager.tag_revision(), manager.tags().revision());
}

#[test]
fn the_facade_rejects_an_unknown_resource() {
    let manager = ResourceMgr::new();

    let error = manager.tag(Agent, "ghost", [tag("x")]).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: agent 'ghost' is not registered"
    );

    let error = manager
        .update_tags(Agent, "ghost", tags(&["x"]), Replace)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: agent 'ghost' is not registered"
    );

    // A failed tag/update leaves the index untouched.
    assert_eq!(manager.tag_revision(), 0);
    assert!(manager.tags().is_empty());
    assert_eq!(manager.tags_of(Agent, "ghost"), Vec::new());

    // `resources_by_tags` takes no resource id, so there is nothing for the
    // facade to reject: a tag with no carriers is simply an empty result.
    assert_eq!(
        manager
            .resources_by_tags(Agent, &tags(&["x"]), Any)
            .unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(manager.tag_revision(), 0);
    assert!(!manager.tags().has_tag(&tag("x")));

    // A validation failure still surfaces through the facade, after the
    // existence check has passed.
    let error = manager.tag(Agent, "ghost", Vec::new()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[test]
fn the_facade_covers_the_mcp_server_kind() {
    let manager = ResourceMgr::new();
    let config = McpServerConfig::stdio("srv", "cmd");
    let server_id = config.server_id.clone();
    manager.mcp_servers().register(config).unwrap();

    assert_eq!(
        manager.tag(McpServer, &server_id, [tag("prod")]).unwrap(),
        tags(&["prod"])
    );
    assert_eq!(manager.tags_of(McpServer, &server_id), tags(&["prod"]));
    assert_eq!(
        manager
            .resources_by_tags(McpServer, &tags(&["prod"]), Any)
            .unwrap(),
        ids(&[server_id.as_str()])
    );

    // The MCP kind validates against the MCP-server manager, not the four
    // run-able registries.
    let error = manager
        .tag(McpServer, "missing-server", [tag("prod")])
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: mcp_server 'missing-server' is not registered"
    );
}

#[test]
fn the_facade_untags_and_removes_tags() {
    let manager = ResourceMgr::new();
    manager.register_tool(facade_tool("t1")).unwrap();
    manager.register_tool(facade_tool("t2")).unwrap();
    manager.tag(Tool, "t1", tags(&["prod", "edge"])).unwrap();
    manager.tag(Tool, "t2", [tag("prod")]).unwrap();
    assert_eq!(manager.tag_revision(), 2);

    assert_eq!(manager.untag(Tool, "t1"), tags(&["prod", "edge"]));
    assert_eq!(manager.tag_revision(), 3);
    assert_eq!(manager.tags_of(Tool, "t1"), Vec::new());
    // Untagging an untagged resource is a no-op.
    assert_eq!(manager.untag(Tool, "t1"), Vec::new());
    assert_eq!(manager.tag_revision(), 3);

    assert_eq!(manager.remove_tag(&tag("prod")), ids(&["t2"]));
    assert_eq!(manager.tag_revision(), 4);
    assert_eq!(manager.tags_of(Tool, "t2"), Vec::new());
    assert!(!manager.tags().has_tag(&tag("prod")), "garbage-collected");
    // Removing an unknown tag is a no-op as well.
    assert_eq!(manager.remove_tag(&tag("absent")), Vec::<String>::new());
    assert_eq!(manager.tag_revision(), 4);
}

#[test]
fn the_facade_tags_of_an_unknown_resource_is_empty() {
    let manager = ResourceMgr::new();

    assert_eq!(manager.tags_of(Tool, "ghost"), Vec::new());
    assert_eq!(manager.tags_of(Agent, ""), Vec::new());
    assert_eq!(manager.tag_revision(), 0);
    assert!(manager.tags().is_empty());
}
