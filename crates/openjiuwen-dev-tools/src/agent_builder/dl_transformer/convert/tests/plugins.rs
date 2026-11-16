use super::super::*;
use super::support::*;
use serde_json::json;

/// A plugin document whose single plugin node declares one input, one output,
/// and `configs`.
fn plugin_document(configs: Vec<(&str, Value)>) -> DlDocument {
    let mut plugin = with_next(
        with_outputs(
            with_inputs(
                node("plugin", "Plugin", "call tool"),
                vec![input("query", Some("String"), json!("${start.query}"))],
            ),
            vec![output("result", Some("String"), Some("tool result"))],
        ),
        "end",
    );
    for (key, value) in configs {
        plugin = with_config(plugin, key, value);
    }
    DlDocument {
        nodes: vec![plugin, node("end", "End", "exit")],
    }
}

/// Builds a plugin resource with every field spelled out.
fn resource(
    tool_id: &str,
    tool_name: &str,
    plugin_id: &str,
    plugin_name: &str,
    plugin_version: &str,
) -> PluginResource {
    PluginResource {
        tool_id: tool_id.to_string(),
        tool_name: tool_name.to_string(),
        plugin_id: plugin_id.to_string(),
        plugin_name: plugin_name.to_string(),
        plugin_version: plugin_version.to_string(),
    }
}

/// The platform plugin node of `document`.
fn plugin_node(document: &DlDocument, resources: &[PluginResource]) -> Value {
    let workflow = DlTransformer::new()
        .transform_with_resources(document, resources)
        .expect("document transforms");
    let value = workflow.to_value().expect("document serializes");
    value["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .find(|node| node["id"] == json!("plugin"))
        .expect("plugin node is present")
        .clone()
}

/// The `plugin_param` object of the transformed plugin node.
fn plugin_param(document: &DlDocument, resources: &[PluginResource]) -> Value {
    plugin_node(document, resources)["data"]["inputs"]["plugin_param"].clone()
}

/// The expected plugin `data` for `plugin_param`.
fn plugin_data(plugin_param: Value) -> Value {
    json!({
        "title": "call tool",
        "inputs": {
            "input_parameters": {
                "query": {
                    "type": "ref",
                    "content": ["start", "query"],
                    "extra": {"index": 0},
                },
            },
            "plugin_param": plugin_param,
        },
        "outputs": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "String",
                    "description": "tool result",
                    "extra": {"index": 1},
                },
                "error_code": {"type": "integer", "extra": {"index": 1}},
                "error_message": {"type": "string", "extra": {"index": 2}},
                "data": {"type": "object", "extra": {"index": 3}, "properties": {}},
            },
            "required": ["error_code", "error_message", "data"],
        },
    })
}

#[test]
fn plugin_node_without_resource_uses_its_configs() {
    let document = plugin_document(vec![
        ("tool_id", json!("tool-1")),
        ("tool_name", json!("search")),
        ("plugin_id", json!("plugin-9")),
        ("plugin_name", json!("Search")),
        ("plugin_version", json!("2.0.0")),
    ]);
    let parameters = json!({
        "toolID": "tool-1",
        "toolName": "search",
        "pluginID": "plugin-9",
        "pluginName": "Search",
        "pluginVersion": "2.0.0",
    });

    assert_eq!(plugin_param(&document, &[]), parameters);
    assert_eq!(
        plugin_node(&document, &[])["data"],
        plugin_data(parameters.clone())
    );
    assert_eq!(
        plugin_node(&document, &[])["data"]["inputs"]["plugin_param"],
        parameters
    );
}

#[test]
fn plugin_node_without_resource_or_configs_renders_empty_defaults() {
    let document = plugin_document(Vec::new());
    let node = plugin_node(&document, &[]);

    assert_eq!(
        node["data"]["inputs"]["plugin_param"],
        json!({
            "toolID": "",
            "toolName": "",
            "pluginID": "",
            "pluginName": "",
            "pluginVersion": "draft",
        })
    );
    assert_eq!(
        node["data"]["outputs"],
        json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "String",
                    "description": "tool result",
                    "extra": {"index": 1},
                },
                "error_code": {"type": "integer", "extra": {"index": 1}},
                "error_message": {"type": "string", "extra": {"index": 2}},
                "data": {"type": "object", "extra": {"index": 3}, "properties": {}},
            },
            "required": ["error_code", "error_message", "data"],
        })
    );
}

#[test]
fn plugin_node_copies_every_matched_resource_field() {
    let document = plugin_document(vec![
        ("tool_id", json!("tool-1")),
        ("tool_name", json!("config tool")),
        ("plugin_id", json!("config-plugin")),
        ("plugin_name", json!("Config plugin")),
        ("plugin_version", json!("0.1")),
    ]);
    let resources = [resource("tool-1", "search", "plugin-9", "Search", "1.0.0")];

    assert_eq!(
        plugin_node(&document, &resources)["data"],
        plugin_data(json!({
            "toolID": "tool-1",
            "toolName": "search",
            "pluginID": "plugin-9",
            "pluginName": "Search",
            "pluginVersion": "1.0.0",
        }))
    );
}

#[test]
fn plugin_blank_resource_fields_fall_through_to_configs() {
    let document = plugin_document(vec![
        ("tool_id", json!("tool-1")),
        ("tool_name", json!("search")),
        ("plugin_id", json!("plugin-9")),
        ("plugin_name", json!("Search")),
        ("plugin_version", json!("2.0.0")),
    ]);
    let resources = [resource("tool-1", "  ", "", "", " ")];

    assert_eq!(
        plugin_param(&document, &resources),
        json!({
            "toolID": "tool-1",
            "toolName": "search",
            "pluginID": "plugin-9",
            "pluginName": "Search",
            "pluginVersion": "2.0.0",
        })
    );
}

/// The rendered `pluginVersion` of `document` against `resources`.
fn version_of(document: &DlDocument, resources: &[PluginResource]) -> Value {
    plugin_param(document, resources)["pluginVersion"].clone()
}

#[test]
fn plugin_version_prefers_resource_then_config_then_draft() {
    let configured = plugin_document(vec![
        ("tool_id", json!("tool-1")),
        ("plugin_version", json!("2.0.0")),
    ]);
    let unconfigured = plugin_document(vec![("tool_id", json!("tool-1"))]);

    assert_eq!(
        version_of(&configured, &[resource("tool-1", "", "", "", "1.0.0")]),
        json!("1.0.0")
    );
    assert_eq!(
        version_of(&configured, &[resource("tool-1", "", "", "", "")]),
        json!("2.0.0")
    );
    assert_eq!(version_of(&configured, &[]), json!("2.0.0"));
    assert_eq!(version_of(&unconfigured, &[]), json!("draft"));
    assert_eq!(
        version_of(&unconfigured, &[resource("tool-1", "", "", "", "")]),
        json!("draft")
    );
}

#[test]
fn plugin_node_rejects_non_string_config_values() {
    let error_for = |document: &DlDocument, resources: &[PluginResource]| {
        DlTransformer::new()
            .transform_with_resources(document, resources)
            .expect_err("conversion fails")
            .to_string()
    };

    let numeric_tool = plugin_document(vec![("tool_id", json!(5))]);
    assert!(
        error_for(&numeric_tool, &[]).contains("node plugin: configs.tool_id must be a string"),
        "{}",
        error_for(&numeric_tool, &[])
    );

    let numeric_version = plugin_document(vec![
        ("tool_id", json!("tool-1")),
        ("plugin_version", json!({"version": "1.0.0"})),
    ]);
    assert!(
        error_for(&numeric_version, &[resource("tool-1", "", "", "", "1.0.0")])
            .contains("node plugin: configs.plugin_version must be a string"),
        "{}",
        error_for(&numeric_version, &[resource("tool-1", "", "", "", "1.0.0")])
    );

    let numeric_name = plugin_document(vec![
        ("tool_id", json!("tool-1")),
        ("tool_name", json!(["search"])),
    ]);
    assert!(
        error_for(&numeric_name, &[]).contains("node plugin: configs.tool_name must be a string"),
        "{}",
        error_for(&numeric_name, &[])
    );
}

#[test]
fn plugin_output_defaults_are_unchanged() {
    let document = plugin_document(vec![("tool_id", json!("tool-1"))]);
    let outputs = plugin_node(&document, &[])["data"]["outputs"].clone();

    assert_eq!(
        outputs["required"],
        json!(["error_code", "error_message", "data"])
    );
    assert_eq!(
        outputs["properties"]["result"],
        json!({"type": "String", "description": "tool result", "extra": {"index": 1}})
    );
    assert_eq!(
        outputs["properties"]["error_code"],
        json!({"type": "integer", "extra": {"index": 1}})
    );
    assert_eq!(
        outputs["properties"]["error_message"],
        json!({"type": "string", "extra": {"index": 2}})
    );
    assert_eq!(
        outputs["properties"]["data"],
        json!({"type": "object", "extra": {"index": 3}, "properties": {}})
    );
}
