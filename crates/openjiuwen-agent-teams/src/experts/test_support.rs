//! Test-only fixtures for the `experts` unit tests.
//!
//! Kept out of the public API: the module is compiled only for `cargo test`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::agent_group::{load_agent_group, LoadedAgentGroupPackage, LEADER_MEMBER};

pub(crate) fn write_json(path: &Path, payload: &Value) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, serde_json::to_string_pretty(payload).unwrap()).unwrap();
}

pub(crate) fn write_text(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, text).unwrap();
}

pub(crate) fn update_manifest(path: &Path, mutate: impl FnOnce(&mut Map<String, Value>)) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut value: Value = serde_json::from_str(&text).unwrap();
    mutate(value.as_object_mut().unwrap());
    std::fs::write(path, serde_json::to_string(&value).unwrap()).unwrap();
}

/// Builds `root/group` with a leader plus `members`; every member has a persona `main.md`.
pub(crate) fn minimal_group_dir(root: &Path, members: &[&str]) -> PathBuf {
    let group = root.join("group");
    let mut roster = vec![LEADER_MEMBER];
    roster.extend_from_slice(members);
    write_json(
        &group.join("manifest.json"),
        &json!({
            "name": "group",
            "package_type": "agent_group",
            "agents": roster,
            "description": {"cn": "测试团", "en": "Test Group"},
            "display_name": {"cn": "甲", "en": "Alpha"},
            "version": "2.0.0",
            "instruction": "  work together  ",
        }),
    );
    for member in std::iter::once(LEADER_MEMBER).chain(members.iter().copied()) {
        let mut manifest = json!({
            "package_type": "agent_template",
            "name": format!("{member} display"),
            "description": format!("{member} description"),
            "persona": {"dir": "./persona"},
        });
        if member == "member1" {
            manifest["runtime"] = json!({
                "provider_name": "codex",
                "provider_version": "0.1.0",
                "config": {"mode": "all"},
            });
        }
        write_json(
            &group.join("agents").join(member).join("manifest.json"),
            &manifest,
        );
        write_text(
            &group
                .join("agents")
                .join(member)
                .join("persona")
                .join("main.md"),
            &format!("# {member}"),
        );
    }
    write_text(
        &group.join("agents").join(LEADER_MEMBER).join("AGENT.md"),
        "# leader rules\n",
    );
    group
}

/// Loads the default two-member fixture used by cross-module tests.
pub(crate) fn minimal_group_package(root: &Path) -> LoadedAgentGroupPackage {
    let group = minimal_group_dir(root, &["member1", "member2"]);
    load_agent_group(group).unwrap()
}
