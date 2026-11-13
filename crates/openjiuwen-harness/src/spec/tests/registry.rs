//! Discovery precedence, ordering and duplicate reporting of `SpecRegistry`.

use crate::spec::{DeepAgentSpec, SpecFormat, SpecRegistry, SpecSource};
use openjiuwen_common::error::ErrorKind;
use tempfile::TempDir;

use super::support::{agents_dir, minimal_spec, temp_workspace, write};

#[test]
fn discovery_orders_workspace_then_home_then_builtin() {
    let (temp, workspace) = temp_workspace();
    write(
        &agents_dir(temp.path()).join("alpha.yaml"),
        &minimal_spec("alpha"),
    );
    let home = TempDir::new().expect("home dir");
    write(
        &agents_dir(home.path()).join("beta.yaml"),
        &minimal_spec("beta"),
    );

    let registry = SpecRegistry::discover(
        &workspace,
        &[home.path().to_path_buf(), temp.path().to_path_buf()],
    )
    .expect("discovery succeeds");

    let names = registry.names();
    assert_eq!(&names[..2], ["alpha", "beta"]);
    assert_eq!(
        registry.entry("alpha").map(|entry| entry.source),
        Some(SpecSource::Workspace)
    );
    assert_eq!(
        registry.entry("beta").map(|entry| entry.source),
        Some(SpecSource::Home)
    );
    let builtin = registry
        .entry("deepagent")
        .expect("built-in template is discovered");
    assert_eq!(builtin.source, SpecSource::Builtin);
    assert!(builtin.path.is_none());
    // Built-ins keep their declaration order after the file templates.
    assert_eq!(&names[2..], ["deepagent", "explorer", "reviewer"]);
    assert!(!registry.is_empty());
    assert!(registry.get("missing").is_none());
}

#[test]
fn workspace_template_shadows_a_builtin() {
    let (temp, workspace) = temp_workspace();
    write(
        &agents_dir(temp.path()).join("deepagent.yaml"),
        "name: deepagent\ndescription: workspace override\ntools: [read_file]\n",
    );
    let registry = SpecRegistry::discover(&workspace, &[]).expect("discovery succeeds");
    let entry = registry.entry("deepagent").expect("deepagent resolves");
    assert_eq!(entry.source, SpecSource::Workspace);
    assert_eq!(entry.spec.description, "workspace override");
    // The built-in counterpart is not appended a second time.
    assert_eq!(
        registry
            .names()
            .iter()
            .filter(|name| **name == "deepagent")
            .count(),
        1
    );
}

#[test]
fn duplicate_file_names_report_winner_and_loser() {
    let (temp, workspace) = temp_workspace();
    let winner = agents_dir(temp.path()).join("dup.yaml");
    write(&winner, &minimal_spec("dup"));
    let home = TempDir::new().expect("home dir");
    let loser = agents_dir(home.path()).join("dup.yaml");
    write(&loser, &minimal_spec("dup"));

    let error = SpecRegistry::discover(&workspace, &[home.path().to_path_buf()])
        .expect_err("duplicate names are rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(
        message.contains("duplicate agent spec name 'dup'"),
        "{message}"
    );
    assert!(message.contains("dup.yaml"), "{message}");
    assert!(message.contains("wins"), "{message}");
    // Both paths are present: the workspace winner first, the home loser second.
    let winner_index = message
        .find(&winner.display().to_string())
        .expect("winner path is reported");
    let loser_index = message
        .find(&loser.display().to_string())
        .expect("loser path is reported");
    assert!(winner_index < loser_index, "{message}");
}

#[test]
fn duplicate_names_in_one_directory_are_rejected() {
    let (temp, workspace) = temp_workspace();
    write(
        &agents_dir(temp.path()).join("one.yaml"),
        &minimal_spec("same"),
    );
    write(
        &agents_dir(temp.path()).join("two.yaml"),
        &minimal_spec("same"),
    );
    let error = SpecRegistry::discover(&workspace, &[])
        .expect_err("same-directory duplicates are rejected");
    assert!(error.message().contains("duplicate agent spec name 'same'"));
}

#[test]
fn missing_directories_still_yield_the_builtin_catalog() {
    let (_temp, workspace) = temp_workspace();
    let registry = SpecRegistry::discover(&workspace, &[]).expect("discovery succeeds");
    assert_eq!(registry.names(), ["deepagent", "explorer", "reviewer"]);
    assert_eq!(registry.len(), 3);

    // A non-existent home root behaves the same way.
    let registry = SpecRegistry::discover(
        &workspace,
        &[std::path::PathBuf::from("/definitely/not/a/real/home")],
    )
    .expect("missing home is skipped");
    assert_eq!(registry.len(), 3);
}

#[test]
fn from_specs_rejects_duplicates_and_invalid_entries() {
    let first = DeepAgentSpec::parse(SpecFormat::Yaml, "name: same\n").expect("parse");
    let second = first.clone();
    let error = SpecRegistry::from_specs(vec![first, second]).expect_err("duplicate is rejected");
    assert!(error.message().contains("duplicate agent spec name 'same'"));

    let invalid =
        DeepAgentSpec::parse(SpecFormat::Yaml, "name: invalid\ntools: [nope]\n").expect("parse");
    let error = SpecRegistry::from_specs(vec![invalid]).expect_err("invalid spec is rejected");
    assert!(error.message().contains("nope"));
}

#[test]
fn registry_resolves_a_discovered_template() {
    let (temp, workspace) = temp_workspace();
    write(
        &agents_dir(temp.path()).join("solver.yaml"),
        "name: solver\ndescription: solves\ntools: [read_file]\n",
    );
    let registry = SpecRegistry::discover(&workspace, &[]).expect("discovery");
    let resolved = registry.resolve("solver", &workspace).expect("resolve");
    assert_eq!(resolved.card.name, "solver");
    assert_eq!(resolved.tools.len(), 1);
    let error = registry
        .resolve("missing", &workspace)
        .expect_err("unknown template");
    assert_eq!(error.kind(), ErrorKind::NotFound);
}
