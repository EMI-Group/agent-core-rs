//! Integration tests for the temporary-directory and environment fixtures.
//!
//! The fixtures replace the two pytest mechanisms the Python test-suite relied
//! on: `tmp_path` (an owned temporary directory tree) and `monkeypatch.setenv` /
//! `monkeypatch.delenv` (environment variables that are put back afterwards).
//! The unit tests beside `src/fixtures/` pin each helper in isolation; this file
//! proves the drop-scoped behaviour end to end from outside the crate, including
//! through [`OpenJiuwenHome`], which is the consumer that turns
//! `OPENJIUWEN_HOME` into a runtime directory.
//!
//! **Environment discipline**: the process environment is global state, so every
//! test that reads or mutates it holds a fixture guard for its whole body. The
//! guards serialize through one process-global lock, which means a test never
//! observes another test's temporary values and the restore assertions after a
//! guard is dropped cannot be raced. Consequently no test here calls bare
//! `std::env::set_var` / `std::env::remove_var`.
//!
//! All tests are synchronous (the fixtures are synchronous), offline, and touch
//! only the system temporary directory and the process environment.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use openjiuwen_common::config::{env, OpenJiuwenHome};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{EnvGuard, TempHomeGuard, TempWorkspace, ISOLATED_ENV_VARS};
use serde_json::json;

/// Python parity: pytest removes `tmp_path` after the test; `TempWorkspace`
/// owns the `tempfile::TempDir` and removes the tree on `Drop`.
#[test]
fn temp_workspace_drop_removes_the_tree() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let root = workspace.root().to_path_buf();

    let written = workspace
        .write("workspace/nested/deep/file.txt", "payload")
        .expect("write nested file");
    assert!(root.is_dir(), "root {} exists", root.display());
    assert_eq!(written, root.join("workspace/nested/deep/file.txt"));
    assert_eq!(
        workspace
            .read("workspace/nested/deep/file.txt")
            .expect("read"),
        "payload"
    );

    drop(workspace);

    assert!(
        !root.exists(),
        "root {} still exists after Drop",
        root.display()
    );
}

/// Layout, file I/O, `files()` ordering, and the two JSON error kinds.
#[test]
fn temp_workspace_layout_and_io_round_trip() {
    let workspace = TempWorkspace::new().expect("temp workspace");
    let root = workspace.root().to_path_buf();

    // The initial layout is `home/`, `workspace/`, `logs/`, all inside the root.
    for (dir, name) in [
        (workspace.home(), "home"),
        (workspace.workspace(), "workspace"),
        (workspace.logs(), "logs"),
    ] {
        assert!(dir.is_dir(), "{} is a directory", dir.display());
        assert_eq!(dir, root.join(name));
        assert!(dir.starts_with(&root));
    }

    // Plain text round-trips through a nested relative path.
    let written = workspace
        .write("workspace/src/notes.md", "first line\n")
        .expect("write");
    assert_eq!(written, root.join("workspace/src/notes.md"));
    assert_eq!(
        workspace.read("workspace/src/notes.md").expect("read"),
        "first line\n"
    );
    assert!(workspace.exists("workspace/src/notes.md"));
    assert!(!workspace.exists("workspace/missing.md"));

    // JSON round-trips through the same tree.
    let state = json!({"agent": {"name": "demo"}, "turns": 3});
    let state_path = workspace
        .write_json("home/state.json", &state)
        .expect("write_json");
    assert_eq!(state_path, root.join("home/state.json"));
    assert_eq!(
        workspace.read_json("home/state.json").expect("read_json"),
        state
    );

    // `files()` is relative to the root, sorted, and never yields directories.
    workspace.create_dir("workspace/empty").expect("create_dir");
    assert_eq!(
        workspace.files(),
        vec![
            PathBuf::from("home/state.json"),
            PathBuf::from("workspace/src/notes.md"),
        ]
    );

    // A missing document is a filesystem problem: `ErrorKind::SysOp`.
    let missing = workspace
        .read_json("home/absent.json")
        .expect_err("missing file");
    assert_eq!(missing.kind(), ErrorKind::SysOp);
    assert!(missing.to_string().contains("home/absent.json"));

    // A malformed document is a validation problem: `ErrorKind::Validation`.
    workspace
        .write("home/broken.json", "{not json")
        .expect("write malformed");
    let malformed = workspace
        .read_json("home/broken.json")
        .expect_err("malformed document");
    assert_eq!(malformed.kind(), ErrorKind::Validation);
    assert!(malformed.to_string().contains("home/broken.json"));

    // Overwriting the same path with a valid document recovers it, proving the
    // two failures above came from the document and not from a sticky state.
    workspace
        .write("home/broken.json", "{\"ok\": true}")
        .expect("overwrite");
    assert_eq!(
        workspace.read_json("home/broken.json").expect("read_json"),
        json!({"ok": true})
    );
}

/// Python parity: `monkeypatch.setenv("OPENJIUWEN_HOME", str(tmp_path / "home"))`
/// — the variable points at the temporary home for the guard's lifetime and is
/// put back afterwards, and the consumer (`OpenJiuwenHome::resolve`) sees it.
#[test]
fn temp_home_guard_drives_openjiuwen_home_resolution_and_restores() {
    // The outer guard owns the environment lock for the whole test and saves the
    // real process value, so the capture below cannot race with a parallel test.
    let outer = EnvGuard::new(&[env::OPENJIUWEN_HOME]);
    let previous = std::env::var_os(env::OPENJIUWEN_HOME);
    assert_eq!(previous, None, "EnvGuard::new clears the names it manages");

    // Case 1: the variable was unset, so it is unset again after Drop.
    {
        let guard = TempHomeGuard::new().expect("temp home");
        let home = guard.home().to_path_buf();
        let root = guard.root().to_path_buf();

        assert_eq!(
            std::env::var(env::OPENJIUWEN_HOME).expect("OPENJIUWEN_HOME is set"),
            home.display().to_string()
        );
        assert_eq!(OpenJiuwenHome::resolve().root(), home.as_path());

        drop(guard);

        assert_eq!(
            std::env::var_os(env::OPENJIUWEN_HOME),
            previous,
            "an unset previous value is restored by removing the variable"
        );
        assert!(
            !root.exists(),
            "temporary home {} was removed",
            root.display()
        );
    }

    // Case 2: the variable had a value, so that exact value comes back.
    outer.set(env::OPENJIUWEN_HOME, "/previous/openjiuwen/home");
    let previous = std::env::var_os(env::OPENJIUWEN_HOME);
    {
        let guard = TempHomeGuard::new().expect("temp home");
        assert_eq!(
            OpenJiuwenHome::resolve().root(),
            guard.home(),
            "the guard shadows the previous value while it is alive"
        );
    }
    assert_eq!(
        std::env::var_os(env::OPENJIUWEN_HOME),
        previous,
        "a set previous value is restored verbatim"
    );
}

/// `TempHomeGuard::isolated` clears the model/runtime environment, not just
/// `OPENJIUWEN_HOME`.
#[test]
fn isolated_temp_home_clears_the_model_environment() {
    // The outer guard stands in for the developer's shell: it owns the lock and
    // remembers the real process values of the two sentinels.
    let outer = EnvGuard::new(&[env::OPENJIUWEN_API_KEY, env::MODEL_NAME]);
    outer.set(env::OPENJIUWEN_API_KEY, "sk-sentinel");
    outer.set(env::MODEL_NAME, "sentinel-model");

    {
        let guard = TempHomeGuard::isolated().expect("temp home");

        for name in ISOLATED_ENV_VARS {
            if *name == env::OPENJIUWEN_HOME {
                assert_eq!(
                    std::env::var(name).ok(),
                    Some(guard.home().display().to_string()),
                    "{name} points at the guard home"
                );
            } else {
                assert_eq!(
                    std::env::var_os(name),
                    None,
                    "{name} is cleared by TempHomeGuard::isolated"
                );
            }
        }

        // The sentinels set before the guard existed are gone while it lives.
        assert!(std::env::var(env::OPENJIUWEN_API_KEY).is_err());
        assert!(std::env::var(env::MODEL_NAME).is_err());
    }

    // ... and the outer guard's values are back afterwards.
    assert_eq!(
        outer.get(env::OPENJIUWEN_API_KEY).as_deref(),
        Some("sk-sentinel")
    );
    assert_eq!(
        outer.get(env::MODEL_NAME).as_deref(),
        Some("sentinel-model")
    );
}

/// `EnvGuard` clears on construction, tracks `set`/`remove`/`get`, and restores
/// both the "was set" and the "was unset" cases on `Drop`.
#[test]
fn env_guard_restores_the_previous_value_and_tracks_removals() {
    let outer = EnvGuard::new(&[
        env::MODEL_PROVIDER,
        env::API_BASE,
        env::SSRF_PROTECT_ENABLED,
    ]);
    outer.set(env::MODEL_PROVIDER, "process-provider");
    outer.set(env::API_BASE, "https://process.test/v1");
    // `SSRF_PROTECT_ENABLED` is deliberately left unset.

    {
        let guard = EnvGuard::new(&[
            env::MODEL_PROVIDER,
            env::API_BASE,
            env::SSRF_PROTECT_ENABLED,
        ]);

        // Construction cleared everything it manages.
        assert_eq!(guard.get(env::MODEL_PROVIDER), None);
        assert_eq!(guard.get(env::API_BASE), None);
        assert_eq!(guard.get(env::SSRF_PROTECT_ENABLED), None);

        guard.set(env::MODEL_PROVIDER, "test-provider");
        assert_eq!(
            guard.get(env::MODEL_PROVIDER).as_deref(),
            Some("test-provider")
        );
        assert_eq!(
            std::env::var(env::MODEL_PROVIDER).ok().as_deref(),
            Some("test-provider")
        );

        guard.set(env::SSRF_PROTECT_ENABLED, "true");
        assert_eq!(
            guard.get(env::SSRF_PROTECT_ENABLED).as_deref(),
            Some("true")
        );

        // `remove` makes the variable unset for the guard's lifetime.
        guard.set(env::API_BASE, "https://test.test/v1");
        guard.remove(env::API_BASE);
        assert_eq!(guard.get(env::API_BASE), None);
        assert!(std::env::var_os(env::API_BASE).is_none());
    }

    // `Drop` restored exactly what existed when the guard was created.
    assert_eq!(
        outer.get(env::MODEL_PROVIDER).as_deref(),
        Some("process-provider")
    );
    assert_eq!(
        outer.get(env::API_BASE).as_deref(),
        Some("https://process.test/v1")
    );
    assert_eq!(
        outer.get(env::SSRF_PROTECT_ENABLED),
        None,
        "a variable that was unset when the guard was created stays unset"
    );
}

/// `EnvGuard::hold(ISOLATED_ENV_VARS)` clears every isolated name on
/// construction and leaves `set`/`get` usable while held.
#[test]
fn hold_clears_every_isolated_variable_while_held() {
    // Nesting is supported on one thread: the outer guard owns the process lock
    // and remembers the real values, the inner guard is the one under test.
    let outer = EnvGuard::new(ISOLATED_ENV_VARS);
    outer.set(env::MODEL_NAME, "shell-model");
    outer.set(env::OPENJIUWEN_HOME, "/shell/openjiuwen/home");

    {
        let held = EnvGuard::hold(ISOLATED_ENV_VARS);

        for name in ISOLATED_ENV_VARS {
            assert_eq!(std::env::var_os(name), None, "{name} cleared by hold");
        }

        held.set(env::MODEL_NAME, "held-model");
        assert_eq!(held.get(env::MODEL_NAME).as_deref(), Some("held-model"));
    }

    assert_eq!(outer.get(env::MODEL_NAME).as_deref(), Some("shell-model"));
    assert_eq!(
        outer.get(env::OPENJIUWEN_HOME).as_deref(),
        Some("/shell/openjiuwen/home")
    );
}

/// The isolated set is exactly the eleven documented model/runtime knobs.
#[test]
fn isolated_env_vars_lists_exactly_the_documented_names() {
    // This test only reads a constant; it never touches the environment.
    let documented = [
        env::OPENJIUWEN_HOME,
        env::OPENJIUWEN_API_KEY,
        env::API_KEY,
        env::API_BASE,
        env::MODEL_PROVIDER,
        env::MODEL_NAME,
        env::OPENAI_API_KEY,
        env::OPENAI_BASE_URL,
        env::SSRF_PROTECT_ENABLED,
        env::OPENJIUWEN_LOG_LEVEL,
        env::OPENJIUWEN_LOG_FORMAT,
    ];

    assert_eq!(ISOLATED_ENV_VARS.len(), 11, "{ISOLATED_ENV_VARS:?}");
    assert_eq!(documented.len(), 11);
    for expected in documented {
        assert!(
            ISOLATED_ENV_VARS.contains(&expected),
            "{expected} is isolated"
        );
    }

    // No duplicates: a repeated name would silently shrink the cleared set.
    let mut unique = ISOLATED_ENV_VARS.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        ISOLATED_ENV_VARS.len(),
        "{ISOLATED_ENV_VARS:?}"
    );
}

/// The restore performed by `Drop` is process-wide, so a second thread observes
/// it. `EnvGuard` is `!Send`, so the helper thread only reads the value.
#[test]
fn restoring_a_guard_is_visible_to_another_thread() {
    let outer = EnvGuard::new(&[env::OPENJIUWEN_HOME]);
    outer.set(env::OPENJIUWEN_HOME, "/thread/outer/home");

    let (signal_tx, signal_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        // Wait until the nested guard below has been dropped.
        signal_rx
            .recv_timeout(Duration::from_millis(200))
            .expect("nested guard dropped");
        std::env::var(env::OPENJIUWEN_HOME).ok()
    });

    {
        let inner = EnvGuard::new(&[env::OPENJIUWEN_HOME]);
        assert_eq!(inner.get(env::OPENJIUWEN_HOME), None);
        inner.set(env::OPENJIUWEN_HOME, "/thread/inner/home");
    }

    signal_tx.send(()).expect("signal reader");
    let seen = reader.join().expect("reader thread");

    assert_eq!(
        seen.as_deref(),
        Some("/thread/outer/home"),
        "the nested guard was restored to the outer guard's value"
    );
    assert_eq!(
        outer.get(env::OPENJIUWEN_HOME).as_deref(),
        Some("/thread/outer/home")
    );
}

/// `TempHomeGuard::set_var` mutates a managed name for the guard's lifetime and
/// the previous value returns on `Drop`.
#[test]
fn temp_home_guard_set_var_is_visible_and_restored() {
    // The outer guard establishes the process value these assertions restore to.
    let outer = EnvGuard::new(&[env::MODEL_NAME]);
    outer.set(env::MODEL_NAME, "process-model");

    {
        let guard = TempHomeGuard::isolated().expect("temp home");
        assert_eq!(
            std::env::var(env::MODEL_NAME).ok(),
            None,
            "`isolated` cleared the inherited model name"
        );

        guard.set_var(env::MODEL_NAME, "guard-model");
        assert_eq!(
            std::env::var(env::MODEL_NAME).ok().as_deref(),
            Some("guard-model")
        );
    }

    assert_eq!(outer.get(env::MODEL_NAME).as_deref(), Some("process-model"));
}
