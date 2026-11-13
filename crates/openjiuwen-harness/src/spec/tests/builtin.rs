//! The embedded built-in catalog: it parses, validates, and every template
//! resolves against a workspace.

use crate::spec::{builtin_specs, resolve_with_source, BUILTIN_RULES_YAML};

use super::support::temp_workspace;

#[test]
fn embedded_catalog_parses_and_is_cached() {
    assert!(BUILTIN_RULES_YAML.contains("name: deepagent"));
    let first = builtin_specs().expect("built-in catalog parses");
    let second = builtin_specs().expect("cached catalog parses");
    assert_eq!(first, second);
    assert_eq!(
        first
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        ["deepagent", "explorer", "reviewer"]
    );
    assert_eq!(first[0].id, "builtin.deepagent");
    assert!(first[0].task_loop);
    assert!(first[0]
        .subagents
        .iter()
        .any(|subagent| subagent.name == "explore_agent"));
}

#[test]
fn every_builtin_template_resolves() {
    let (_temp, workspace) = temp_workspace();
    let specs = builtin_specs().expect("built-in catalog parses");
    for spec in specs {
        let resolved = resolve_with_source(&spec, &workspace)
            .unwrap_or_else(|error| panic!("{} must resolve: {error}", spec.name));
        assert!(!resolved.tools.is_empty(), "{} declares tools", spec.name);
        assert!(
            !resolved.skills.is_empty(),
            "{} declares a skill root",
            spec.name
        );
        assert!(
            resolved.skills.iter().all(|skill| skill.enabled),
            "{} enables its skill roots",
            spec.name
        );
        assert!(
            resolved.rails.iter().any(|rail| rail == "security"),
            "{} keeps the security rail",
            spec.name
        );
        for subagent in &resolved.subagents {
            assert!(
                subagent.system_prompt.is_some(),
                "{}'s sub-agent has a prompt",
                spec.name
            );
        }
    }
}

#[test]
fn builtin_skill_roots_are_anchored_at_the_workspace() {
    let (_temp, workspace) = temp_workspace();
    let specs = builtin_specs().expect("built-in catalog parses");
    let resolved = resolve_with_source(&specs[0], &workspace).expect("resolve");
    assert_eq!(
        resolved.skill_roots[0],
        workspace.root_path.join(".openjiuwen").join("skills")
    );
}
