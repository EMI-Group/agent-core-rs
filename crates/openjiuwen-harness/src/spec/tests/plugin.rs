//! Plugin package discovery, manifest validation and aggregation.

use crate::spec::{load_plugins, PluginManifest, PluginRegistry, PLUGIN_MANIFEST_FILENAME};
use crate::tools::BuiltinTool;
use openjiuwen_common::error::ErrorKind;
use tempfile::TempDir;

use super::support::{plugins_dir, temp_workspace, write};

/// Writes `manifest` into `<root>/<name>/plugin.yaml` and returns the package directory.
fn write_plugin(temp: &TempDir, name: &str, manifest: &str) -> std::path::PathBuf {
    let directory = plugins_dir(temp.path()).join(name);
    write(&directory.join(PLUGIN_MANIFEST_FILENAME), manifest);
    directory
}

#[test]
fn missing_plugins_directory_is_an_empty_registry() {
    let (_temp, workspace) = temp_workspace();
    let registry = PluginRegistry::load(&workspace).expect("load succeeds");
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    let resolved = load_plugins(&workspace).expect("aggregate succeeds");
    assert!(resolved.is_empty());
    assert_eq!(resolved.tools().len(), 0);
    assert_eq!(resolved.specs().len(), 0);
}

#[test]
fn loads_and_resolves_a_full_plugin() {
    let (temp, workspace) = temp_workspace();
    let directory = write_plugin(
        &temp,
        "acme",
        "name: acme\nversion: '0.1.0'\ndescription: analysis helpers\ntools: [read_file, web_search]\nrails: [core.security, skills]\nskills: [skills]\nspec: agent.yaml\nmetadata:\n  owner: platform\n",
    );
    std::fs::create_dir_all(directory.join("skills")).expect("create skill root");
    write(
        &directory.join("agent.yaml"),
        "name: acme-agent\ndescription: plugin template\ntools: [read_file]\n",
    );

    let registry = PluginRegistry::load(&workspace).expect("load succeeds");
    assert_eq!(registry.len(), 1);
    let plugin = registry.get("acme").expect("plugin is registered");
    assert_eq!(plugin.manifest.version.as_deref(), Some("0.1.0"));
    assert_eq!(plugin.manifest.description, "analysis helpers");
    assert_eq!(
        plugin.manifest.metadata.get("owner"),
        Some(&serde_json::json!("platform"))
    );
    assert_eq!(plugin.directory, directory);
    assert_eq!(
        plugin.tools,
        [BuiltinTool::ReadFile, BuiltinTool::WebSearch]
    );
    assert_eq!(plugin.rails, ["security", "skills"]);
    assert_eq!(plugin.skill_roots, [directory.join("skills")]);
    let spec = plugin.spec.as_ref().expect("template is parsed");
    assert_eq!(spec.name, "acme-agent");
    assert_eq!(spec.source_path, Some(directory.join("agent.yaml")));
}

#[test]
fn directories_without_a_manifest_are_skipped() {
    let (temp, workspace) = temp_workspace();
    std::fs::create_dir_all(plugins_dir(temp.path()).join("not-a-plugin"))
        .expect("create unrelated directory");
    write_plugin(&temp, "real", "name: real\n");
    let registry = PluginRegistry::load(&workspace).expect("load succeeds");
    assert_eq!(registry.len(), 1);
    assert!(registry.get("real").is_some());
}

#[test]
fn a_missing_name_falls_back_to_the_directory_name() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "fallback", "description: no name\n");
    let registry = PluginRegistry::load(&workspace).expect("load succeeds");
    let plugin = registry.get("fallback").expect("directory name is used");
    assert_eq!(plugin.manifest.name, "fallback");
}

#[test]
fn duplicate_plugin_names_report_both_manifests() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "first", "name: same\n");
    write_plugin(&temp, "second", "name: same\n");
    let error = PluginRegistry::load(&workspace).expect_err("duplicate names are rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(
        message.contains("duplicate plugin name 'same'"),
        "{message}"
    );
    assert!(message.contains("first"), "{message}");
    assert!(message.contains("second"), "{message}");
}

#[test]
fn unknown_tool_names_the_tool_and_the_manifest() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "bad-tool", "name: bad-tool\ntools: [teleport]\n");
    let error = PluginRegistry::load(&workspace).expect_err("unknown tool is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("teleport"), "{message}");
    assert!(message.contains("bad-tool"), "{message}");
    assert!(message.contains("web_search"), "{message}");
}

#[test]
fn unknown_rail_is_rejected() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "bad-rail", "name: bad-rail\nrails: [teleport]\n");
    let error = PluginRegistry::load(&workspace).expect_err("unknown rail is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("teleport"));
}

#[test]
fn unsafe_skill_roots_are_rejected() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "escape", "name: escape\nskills: ['../../outside']\n");
    let error = PluginRegistry::load(&workspace).expect_err("traversal is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("../../outside"), "{message}");
    assert!(message.contains("package-relative"), "{message}");
}

#[test]
fn missing_skill_directories_are_rejected() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "empty", "name: empty\nskills: [skills]\n");
    let error = PluginRegistry::load(&workspace).expect_err("missing root is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("is not a directory"));
}

#[test]
fn missing_spec_files_are_rejected() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "no-spec", "name: no-spec\nspec: agent.yaml\n");
    let error = PluginRegistry::load(&workspace).expect_err("missing template is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("does not exist"));
}

#[test]
fn unsafe_spec_paths_are_rejected() {
    let (temp, workspace) = temp_workspace();
    write_plugin(&temp, "escape", "name: escape\nspec: ../agent.yaml\n");
    let error = PluginRegistry::load(&workspace).expect_err("traversal is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("package-relative"));
}

#[test]
fn json_manifests_are_accepted_directly() {
    // Discovery looks for the documented `plugin.yaml` name; a JSON manifest is
    // still a valid manifest when loaded explicitly.
    let temp = TempDir::new().expect("temp dir");
    let workspace = crate::workspace::Workspace::for_language(temp.path(), "en");
    let directory = plugins_dir(temp.path()).join("json-plugin");
    write(
        &directory.join("plugin.json"),
        "{\"name\": \"json-plugin\", \"version\": \"1.0.0\", \"tools\": [\"read_file\"]}",
    );
    let registry = PluginRegistry::load(&workspace).expect("load succeeds");
    assert!(
        registry.is_empty(),
        "discovery only mounts plugin.yaml packages"
    );

    let manifest = PluginManifest::from_path(directory.join("plugin.json")).expect("manifest");
    assert_eq!(manifest.version.as_deref(), Some("1.0.0"));
    assert_eq!(manifest.tools, ["read_file"]);
    assert_eq!(
        manifest.label(),
        format!(
            "plugin 'json-plugin' ({})",
            directory.join("plugin.json").display()
        )
    );
}

#[test]
fn unsupported_manifest_extensions_are_rejected() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("plugin.toml");
    write(&path, "name = 'x'");
    let error = PluginManifest::from_path(&path).expect_err("toml is unsupported");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("supported: yaml, yml, json"));
}

#[test]
fn blank_version_is_rejected() {
    let temp = TempDir::new().expect("temp dir");
    let path = temp.path().join("plugin.yaml");
    write(&path, "name: blank\nversion: '  '\n");
    let error = PluginManifest::from_path(&path).expect_err("blank version is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("version"));
}

#[test]
fn aggregate_deduplicates_tools_rails_and_roots() {
    let (temp, workspace) = temp_workspace();
    for name in ["alpha", "beta"] {
        let directory = write_plugin(
            &temp,
            name,
            &format!("name: {name}\ntools: [read_file, web_search]\nrails: [security]\nskills: [skills]\n"),
        );
        std::fs::create_dir_all(directory.join("skills")).expect("create skill root");
    }
    let registry = PluginRegistry::load(&workspace).expect("load succeeds");
    let resolved = registry.resolve();
    assert_eq!(resolved.len(), 2);
    assert_eq!(
        resolved
            .plugins
            .iter()
            .map(|plugin| plugin.manifest.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert_eq!(
        resolved.tools(),
        [BuiltinTool::ReadFile, BuiltinTool::WebSearch]
    );
    assert_eq!(resolved.rails(), ["security"]);
    assert_eq!(
        resolved.skill_roots(),
        [
            plugins_dir(temp.path()).join("alpha").join("skills"),
            plugins_dir(temp.path()).join("beta").join("skills"),
        ]
    );
    assert_eq!(resolved.specs().len(), 0);
}
