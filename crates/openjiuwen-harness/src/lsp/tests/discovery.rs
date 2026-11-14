//! Discovery tests: extension matching, priority ordering, language resolution, marker-based
//! root inference (including the Cargo `[workspace]` rule), workspace containment and the
//! built-in definition table. Everything runs in temp directories; no server is started.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

use crate::lsp::config::LspServerConfig;
use crate::lsp::discovery::{builtin_server_configs, DiscoveryIndex, ResolvedServer, ServerKey};

/// Builds one server definition.
fn server(name: &str, extension: &str, priority: u32) -> LspServerConfig {
    LspServerConfig::new(format!("{name}-bin"))
        .with_name(name)
        .with_language_id(name)
        .with_extensions([extension])
        .with_priority(priority)
}

/// Builds an index over `servers` rooted at `root`.
fn index(root: &Path, servers: Vec<LspServerConfig>) -> DiscoveryIndex {
    DiscoveryIndex::new(root.to_path_buf(), servers)
}

/// Writes `content` to `path`, creating parent directories.
fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn extension_matching_ignores_case_and_a_leading_dot() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let index = index(
        root,
        vec![server("upper", "RS", 10), server("python", ".Py", 10)],
    );

    let rust = index
        .resolve(&root.join("src/main.Rs"))
        .expect("rs matches");
    assert_eq!(rust.key.name, "upper");
    assert_eq!(rust.config.name, "upper");

    let python = index.resolve(&root.join("tool.py")).expect("py matches");
    assert_eq!(python.key.name, "python");

    // No extension, unknown extension: nothing resolves, the caller decides the fallback.
    assert!(index.resolve(&root.join("Makefile")).is_none());
    assert!(index.resolve(&root.join("notes.txt")).is_none());
    assert_eq!(index.candidates_for_extension(".RS").len(), 1);
    assert_eq!(index.candidates_for_extension("").len(), 0);
}

#[test]
fn priority_then_registration_order_decides() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let ordered = index(
        root,
        vec![server("late", "rs", 20), server("early", "rs", 10)],
    );
    assert_eq!(
        ordered.resolve(&root.join("a.rs")).unwrap().key.name,
        "early"
    );
    assert_eq!(
        ordered
            .candidates_for_extension("rs")
            .iter()
            .map(|config| config.name.as_str())
            .collect::<Vec<_>>(),
        vec!["early", "late"]
    );

    let tied = index(
        root,
        vec![server("first", "rs", 5), server("second", "rs", 5)],
    );
    assert_eq!(tied.resolve(&root.join("a.rs")).unwrap().key.name, "first");
}

#[test]
fn resolve_by_language_prefers_the_configured_root() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let scoped = root.join("crate-a");
    fs::create_dir_all(&scoped).unwrap();
    let servers = vec![
        LspServerConfig::new("rust-analyzer")
            .with_name("rust")
            .with_language_id("rust")
            .with_extensions([".rs"])
            .with_root(&scoped),
        server("python", "py", 10),
    ];
    let index = index(root, servers);

    let rust = index.resolve_by_language("rust").expect("language match");
    assert_eq!(rust.root(), scoped.as_path());
    assert_eq!(rust.language_id(), "rust");
    assert_eq!(rust.key.name, "rust");

    let python = index.resolve_by_language("python").expect("language match");
    assert_eq!(
        python.root(),
        root,
        "no configured root falls back to the workspace"
    );
    assert!(index.resolve_by_language("cobol").is_none());
}

#[test]
fn nearest_marker_wins_over_the_workspace_root() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let nested = root.join("sub");
    write(&root.join("pyproject.toml"), "");
    write(&nested.join("pyproject.toml"), "");
    let config = LspServerConfig::new("pyright-langserver")
        .with_name("pyright")
        .with_language_id("python")
        .with_extensions([".py"])
        .with_root_markers(["pyproject.toml"]);
    let index = index(root, vec![config]);

    let resolved: ResolvedServer = index
        .resolve(&nested.join("deep/pkg/mod.py"))
        .expect("pyright resolves");
    assert_eq!(resolved.root(), nested.as_path());
    assert_eq!(resolved.key.root, nested);
}

#[test]
fn missing_markers_fall_back_to_configured_then_workspace_root() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let scoped = root.join("project");
    fs::create_dir_all(&scoped).unwrap();

    let with_root = LspServerConfig::new("pyright")
        .with_name("pyright")
        .with_language_id("python")
        .with_extensions([".py"])
        .with_root(&scoped)
        .with_root_markers(["pyproject.toml"]);
    let plain = LspServerConfig::new("lint")
        .with_name("lint")
        .with_language_id("python")
        .with_extensions([".py"]);

    let rooted = index(root, vec![with_root]);
    assert_eq!(
        rooted.resolve(&root.join("a.py")).unwrap().root(),
        scoped.as_path()
    );

    // A configured root outside the workspace is ignored.
    let outside = index(
        root,
        vec![LspServerConfig::new("x")
            .with_name("outside")
            .with_extensions([".py"])
            .with_root("/elsewhere")],
    );
    assert_eq!(outside.resolve(&root.join("a.py")).unwrap().root(), root);

    let plain_index = index(root, vec![plain]);
    assert_eq!(
        plain_index.resolve(&root.join("a.py")).unwrap().root(),
        root
    );

    // Relative paths resolve against the workspace root.
    assert!(plain_index.resolve(Path::new("a.py")).is_some());
}

#[test]
fn paths_outside_the_workspace_do_not_resolve() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let index = index(root, vec![server("rust", "rs", 10)]);
    assert!(
        index.resolve(root).is_none(),
        "a directory has no extension"
    );
    assert!(index.resolve(&root.join("../elsewhere/a.rs")).is_none());
    assert!(index.resolve(Path::new("/usr/lib/a.rs")).is_none());
}

#[test]
fn rust_walks_up_to_the_cargo_workspace_root() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let nested = root.join("nested");
    write(
        &root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"nested\"]\n",
    );
    write(&nested.join("Cargo.toml"), "[package]\nname = \"nested\"\n");
    let config = LspServerConfig::new("rust-analyzer")
        .with_name("rust")
        .with_language_id("rust")
        .with_extensions([".rs"])
        .with_root_markers(["Cargo.toml"]);
    let index = index(root, vec![config]);

    let resolved = index
        .resolve(&nested.join("src/lib.rs"))
        .expect("rust resolves");
    assert_eq!(
        resolved.root(),
        root,
        "the nearest Cargo.toml without [workspace] defers to the workspace manifest"
    );

    // A nested manifest that is itself a workspace stops the walk.
    write(&nested.join("Cargo.toml"), "[workspace]\nmembers = []\n");
    let resolved = index.resolve(&nested.join("src/lib.rs")).unwrap();
    assert_eq!(resolved.root(), nested.as_path());
}

#[test]
fn rust_without_any_workspace_manifest_keeps_the_nearest_root() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let nested = root.join("crate");
    write(&nested.join("Cargo.toml"), "[package]\nname = \"crate\"\n");
    let index = index(
        root,
        vec![LspServerConfig::new("rust-analyzer")
            .with_name("rust")
            .with_language_id("rust")
            .with_extensions([".rs"])
            .with_root_markers(["Cargo.toml"])],
    );
    let resolved = index.resolve(&nested.join("src/main.rs")).unwrap();
    assert_eq!(resolved.root(), nested.as_path());
}

#[test]
fn builtin_configs_mirror_the_python_definitions() {
    let configs = builtin_server_configs();
    let names: Vec<&str> = configs.iter().map(|config| config.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["pyright", "rust", "typescript", "gopls", "jdtls"]
    );

    let pyright = &configs[0];
    assert_eq!(pyright.command, "pyright-langserver");
    assert_eq!(pyright.args, vec!["--stdio"]);
    assert_eq!(pyright.language_id, "python");
    assert_eq!(pyright.extensions, vec![".py", ".pyi"]);
    assert!(pyright.root_markers.contains(&"pyproject.toml".to_owned()));
    assert!(pyright
        .root_markers
        .contains(&"pyrightconfig.json".to_owned()));
    assert_eq!(pyright.priority, 10);

    let rust = &configs[1];
    assert_eq!(rust.command, "rust-analyzer");
    assert_eq!(rust.args.len(), 0);
    assert_eq!(rust.extensions, vec![".rs"]);
    assert_eq!(rust.root_markers, vec!["Cargo.toml"]);

    let typescript = &configs[2];
    assert_eq!(typescript.command, "typescript-language-server");
    assert_eq!(typescript.args, vec!["--stdio"]);
    assert!(typescript.extensions.contains(&".tsx".to_owned()));
    assert!(typescript.extensions.contains(&".cjs".to_owned()));
    assert_eq!(typescript.language_id, "typescript");

    let gopls = &configs[3];
    assert_eq!(gopls.command, "gopls");
    assert_eq!(gopls.language_id, "go");
    assert_eq!(gopls.startup_timeout, Duration::from_secs(60));
    assert_eq!(gopls.root_markers, vec!["go.work", "go.mod", "go.sum"]);

    let jdtls = &configs[4];
    assert_eq!(jdtls.command, "jdtls");
    assert_eq!(jdtls.language_id, "java");
    assert!(jdtls.root_markers.contains(&".project".to_owned()));

    assert!(configs.iter().all(|config| config.priority == 10));
}

#[test]
fn builtin_configs_route_common_files() {
    let workspace = TempDir::new().unwrap();
    let root = workspace.path();
    let index = index(root, builtin_server_configs());
    assert_eq!(
        index.resolve(&root.join("a.py")).unwrap().key.name,
        "pyright"
    );
    assert_eq!(index.resolve(&root.join("a.rs")).unwrap().key.name, "rust");
    assert_eq!(
        index.resolve(&root.join("web/a.tsx")).unwrap().key.name,
        "typescript"
    );
    assert_eq!(
        index.resolve(&root.join("cmd/main.go")).unwrap().key.name,
        "gopls"
    );
    assert_eq!(
        index.resolve(&root.join("Main.java")).unwrap().key.name,
        "jdtls"
    );
    assert!(index.resolve(&root.join("a.rb")).is_none());
}

#[test]
fn server_keys_round_trip_and_order_by_name_then_root() {
    let key = ServerKey {
        name: "rust".to_owned(),
        root: PathBuf::from("/workspace"),
    };
    let value = serde_json::to_value(&key).unwrap();
    assert_eq!(
        value,
        serde_json::json!({ "name": "rust", "root": "/workspace" })
    );
    assert_eq!(serde_json::from_value::<ServerKey>(value).unwrap(), key);

    let earlier = ServerKey {
        name: "go".to_owned(),
        root: PathBuf::from("/z"),
    };
    let later = ServerKey {
        name: "rust".to_owned(),
        root: PathBuf::from("/a"),
    };
    assert!(earlier < later);
}
