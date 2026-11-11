//! Process-environment isolation guards.

use std::cell::Cell;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use openjiuwen_common::config::env;

use super::TempWorkspace;

/// The one process-global lock that serializes every environment mutation.
///
/// The lock exists because the process environment is global state: two tests
/// that clear and restore the same variable concurrently would observe each
/// other's writes. Every [`EnvGuard`] (and therefore every [`TempHomeGuard`])
/// takes this lock, so env-mutating tests serialize even when cargo runs them on
/// separate threads.
static ENV_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// How many environment guards the current thread holds.
    ///
    /// The outermost guard owns [`ENV_LOCK`]; nested guards on the same thread
    /// only bump the depth, which keeps them from deadlocking on a
    /// non-reentrant mutex.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// The variables [`TempHomeGuard::isolated`] clears and restores.
///
/// These are the model/runtime knobs whose process values would otherwise leak
/// from the developer's shell into a test: the `openJiuwen` home and API key,
/// the shared `API_KEY`/`API_BASE` and `OPENAI_*` fallbacks, the model
/// selection, the SSRF switch, and the log settings.
pub const ISOLATED_ENV_VARS: &[&str] = &[
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

/// An RAII guard that clears environment variables and restores them on drop.
///
/// Construction saves the current value of every listed name and then removes
/// it, so assertions never depend on the host environment. Dropping the guard
/// puts each name back exactly as it was (set when it had a value, removed
/// otherwise), which keeps the process clean even when an assertion panics.
///
/// The guard serializes against every other guard through one process-global
/// lock. It is reentrant on the thread that owns it: creating a nested guard on
/// the same thread does not deadlock, and the outermost guard keeps the lock
/// until it is dropped.
///
/// # Contract
///
/// - Only the names passed to [`EnvGuard::new`] (or [`EnvGuard::hold`]) are
///   saved and restored. Names mutated through [`EnvGuard::set`] or
///   [`EnvGuard::remove`] but not listed at construction are left at their
///   final value when the guard drops; list every variable the test touches.
/// - The value is `!Send` because it owns the lock: keep it on the test's
///   runtime thread and never move it into a spawned task.
/// - The lock is poisoned only if a test panics with the guard alive; poisoning
///   is recovered (`PoisonError::into_inner`) rather than propagated, so one
///   failing test does not abort the rest of the binary.
pub struct EnvGuard {
    /// Held only by the outermost guard on a thread.
    lock: Option<MutexGuard<'static, ()>>,
    /// Saved values, restored in reverse order on drop.
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvGuard {
    /// Locks the environment, saves and clears every name in `names`.
    ///
    /// Each name must be a `'static` string — the environment names of this
    /// crate are constants, so this is not a restriction in practice, and it
    /// lets the restore list outlive any borrowed input.
    pub fn new(names: &[&'static str]) -> Self {
        Self::with_names(names)
    }

    /// Same as [`EnvGuard::new`], taking an explicit static slice.
    ///
    /// Useful for passing [`ISOLATED_ENV_VARS`] without a borrow of a local.
    pub fn hold(names: &'static [&'static str]) -> Self {
        Self::new(names)
    }

    /// Sets `name` for the lifetime of the guard (whether or not it was listed
    /// at construction; see the type-level contract).
    pub fn set(&self, name: &str, value: impl AsRef<OsStr>) {
        std::env::set_var(name, value);
    }

    /// Removes `name` for the lifetime of the guard.
    pub fn remove(&self, name: &str) {
        std::env::remove_var(name);
    }

    /// Reads `name` as UTF-8, or `None` when it is unset or not valid UTF-8.
    pub fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    /// Shared construction path.
    fn with_names(names: &[&'static str]) -> Self {
        let lock = acquire();
        let saved = names
            .iter()
            .map(|name| (*name, std::env::var_os(name)))
            .collect();
        for name in names {
            std::env::remove_var(name);
        }
        Self { lock, saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        release(self.lock.take());
    }
}

/// Takes [`ENV_LOCK`] when this is the outermost guard on the thread.
fn acquire() -> Option<MutexGuard<'static, ()>> {
    let depth = DEPTH.with(Cell::get);
    let lock = if depth == 0 {
        Some(ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner))
    } else {
        None
    };
    DEPTH.with(|slot| slot.set(depth + 1));
    lock
}

/// Drops the depth and, for the outermost guard, releases the lock.
fn release(lock: Option<MutexGuard<'static, ()>>) {
    DEPTH.with(|slot| slot.set(slot.get().saturating_sub(1)));
    drop(lock);
}

/// A temporary home directory plus the `OPENJIUWEN_HOME` isolation around it.
///
/// This is the Rust counterpart of
/// `monkeypatch.setenv("OPENJIUWEN_HOME", str(tmp_path / "home"))`: construction
/// creates a [`TempWorkspace`], points `OPENJIUWEN_HOME` at `<root>/home`, and
/// drop restores the previous value and removes the directory. The directory is
/// created eagerly, so code that resolves `OPENJIUWEN_HOME` during construction
/// of the test subject already finds it.
pub struct TempHomeGuard {
    /// Keeps the temporary tree alive for the guard's lifetime.
    workspace: TempWorkspace,
    /// `<root>/home`, returned by [`TempHomeGuard::home`].
    home: PathBuf,
    /// Holds the environment lock and restores `OPENJIUWEN_HOME`.
    env: EnvGuard,
}

impl TempHomeGuard {
    /// Creates a temporary home and points `OPENJIUWEN_HOME` at it.
    ///
    /// Only `OPENJIUWEN_HOME` is saved, cleared, set, and restored; every other
    /// variable keeps its process value. Use [`TempHomeGuard::isolated`] when
    /// the test must not see the host's model or logging configuration.
    pub fn new() -> io::Result<Self> {
        Self::with_names(&[env::OPENJIUWEN_HOME])
    }

    /// Creates a temporary home and additionally clears [`ISOLATED_ENV_VARS`].
    ///
    /// The model, API-key, SSRF, and logging variables are cleared before
    /// `OPENJIUWEN_HOME` is set, so the guard's environment contains no state
    /// inherited from the developer's shell.
    pub fn isolated() -> io::Result<Self> {
        Self::with_names(ISOLATED_ENV_VARS)
    }

    /// The temporary root that owns `home/`, `workspace/`, and `logs/`.
    pub fn root(&self) -> &Path {
        self.workspace.root()
    }

    /// The directory `OPENJIUWEN_HOME` points at (`<root>/home`).
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// Sets `name` for the lifetime of the guard.
    ///
    /// The name is not restored unless it is in [`ISOLATED_ENV_VARS`] (or is
    /// `OPENJIUWEN_HOME`); a test that mutates other variables should wrap it in
    /// its own [`EnvGuard`].
    pub fn set_var(&self, name: &str, value: impl AsRef<OsStr>) {
        self.env.set(name, value);
    }

    /// Shared construction path: workspace, environment lock, `OPENJIUWEN_HOME`.
    fn with_names(names: &[&'static str]) -> io::Result<Self> {
        let workspace = TempWorkspace::new()?;
        let home = workspace.home();
        let guard = EnvGuard::new(names);
        guard.set(env::OPENJIUWEN_HOME, &home);
        Ok(Self {
            workspace,
            home,
            env: guard,
        })
    }
}

impl fmt::Debug for TempHomeGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TempHomeGuard")
            .field("root", &self.root())
            .field("home", &self.home)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn guard_clears_a_set_value_and_restores_it() {
        let outer = EnvGuard::new(&[env::MODEL_NAME]);
        outer.set(env::MODEL_NAME, "outer-model");

        {
            let inner = EnvGuard::new(&[env::MODEL_NAME]);
            assert_eq!(inner.get(env::MODEL_NAME), None);

            inner.set(env::MODEL_NAME, "inner-model");
            assert_eq!(inner.get(env::MODEL_NAME).as_deref(), Some("inner-model"));
        }

        assert_eq!(outer.get(env::MODEL_NAME).as_deref(), Some("outer-model"));
    }

    #[test]
    fn guard_restores_an_unset_value_by_removing_it() {
        let outer = EnvGuard::new(&[env::OPENJIUWEN_LOG_FORMAT]);
        assert_eq!(outer.get(env::OPENJIUWEN_LOG_FORMAT), None);

        {
            let inner = EnvGuard::new(&[env::OPENJIUWEN_LOG_FORMAT]);
            inner.set(env::OPENJIUWEN_LOG_FORMAT, "json");
            assert_eq!(
                inner.get(env::OPENJIUWEN_LOG_FORMAT).as_deref(),
                Some("json")
            );
        }

        assert_eq!(outer.get(env::OPENJIUWEN_LOG_FORMAT), None);
    }

    #[test]
    fn nested_guards_do_not_deadlock_and_keep_the_lock_managed() {
        let outer = EnvGuard::new(&[env::OPENJIUWEN_LOG_LEVEL]);
        outer.set(env::OPENJIUWEN_LOG_LEVEL, "info");

        {
            let inner = EnvGuard::new(&[env::OPENJIUWEN_LOG_LEVEL]);
            assert_eq!(inner.get(env::OPENJIUWEN_LOG_LEVEL), None);
            inner.set(env::OPENJIUWEN_LOG_LEVEL, "debug");
        }

        // The inner guard dropped without releasing the lock: the outer guard
        // still owns it and can keep mutating.
        outer.set(env::OPENJIUWEN_LOG_LEVEL, "warn");
        assert_eq!(
            outer.get(env::OPENJIUWEN_LOG_LEVEL).as_deref(),
            Some("warn")
        );
    }

    #[test]
    fn hold_accepts_a_static_slice() {
        let guard = EnvGuard::hold(ISOLATED_ENV_VARS);
        assert_eq!(guard.get(env::OPENAI_API_KEY), None);
        guard.set(env::OPENAI_API_KEY, "sk-test");
        assert_eq!(guard.get(env::OPENAI_API_KEY).as_deref(), Some("sk-test"));
        guard.remove(env::OPENAI_API_KEY);
        assert_eq!(guard.get(env::OPENAI_API_KEY), None);
    }

    #[test]
    fn guards_serialize_across_threads() {
        /// Two managed names, so the child's guard is a real guard (the lock is
        /// taken on construction, not on the first `set`).
        const NAMES: &[&str] = &[env::MODEL_NAME, env::OPENJIUWEN_LOG_LEVEL];

        let main_guard = EnvGuard::new(NAMES);
        main_guard.set(env::MODEL_NAME, "main-thread");

        let (tx, rx) = mpsc::channel();
        let child = std::thread::spawn(move || {
            // Sent *before* constructing the guard, so the main thread knows the
            // child reached (and is about to block on) the acquisition.
            tx.send("starting").expect("signal start");

            // Blocked here while the main guard owns the process-global lock.
            let guard = EnvGuard::new(NAMES);
            assert_eq!(
                guard.get(env::MODEL_NAME),
                None,
                "the child's guard cleared the names itself"
            );
            guard.set(env::OPENJIUWEN_LOG_LEVEL, "child");
            tx.send("acquired").expect("signal acquisition");

            // `EnvGuard` is `!Send`, so it must be dropped on this thread.
            drop(guard);
        });

        assert_eq!(
            rx.recv_timeout(Duration::from_millis(100)),
            Ok("starting"),
            "the child started and is about to block"
        );
        // Mutual exclusion: the main guard is still alive, so the child cannot
        // get past `EnvGuard::new` and nothing may arrive on the channel. Were
        // the global mutex gone, "acquired" (or a panic) would land here.
        assert_eq!(
            rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "the child acquired the lock while the main guard was alive"
        );

        drop(main_guard);

        // Releasing the lock lets the blocked child finish its construction and
        // report, which also proves the lock is released on drop.
        assert_eq!(
            rx.recv_timeout(Duration::from_millis(1000)),
            Ok("acquired"),
            "the child acquired the lock once the main guard dropped"
        );
        child.join().expect("the child thread finishes");
    }

    #[test]
    fn temp_home_guard_points_home_at_the_temp_root() {
        let guard = TempHomeGuard::new().expect("temp home");

        assert_eq!(guard.home(), guard.root().join("home"));
        assert!(guard.home().is_dir());
        assert_eq!(
            std::env::var(env::OPENJIUWEN_HOME).ok(),
            Some(guard.home().display().to_string())
        );
    }

    #[test]
    fn temp_home_guard_restores_the_previous_home() {
        let outer = EnvGuard::new(&[env::OPENJIUWEN_HOME]);
        outer.set(env::OPENJIUWEN_HOME, "/previous/openjiuwen/home");

        let root = {
            let guard = TempHomeGuard::new().expect("temp home");
            assert_eq!(
                std::env::var(env::OPENJIUWEN_HOME).ok(),
                Some(guard.home().display().to_string())
            );
            let root = guard.root().to_path_buf();
            drop(guard);
            root
        };

        assert!(!root.exists());
        assert_eq!(
            outer.get(env::OPENJIUWEN_HOME).as_deref(),
            Some("/previous/openjiuwen/home")
        );
    }

    #[test]
    fn isolated_clears_every_isolated_variable() {
        let outer = EnvGuard::new(ISOLATED_ENV_VARS);
        outer.set(env::OPENAI_API_KEY, "sk-from-shell");
        outer.set(env::MODEL_NAME, "from-shell");

        {
            let guard = TempHomeGuard::isolated().expect("temp home");

            assert_eq!(std::env::var(env::OPENAI_API_KEY).ok(), None);
            assert_eq!(std::env::var(env::MODEL_NAME).ok(), None);
            assert_eq!(
                std::env::var(env::OPENJIUWEN_HOME).ok(),
                Some(guard.home().display().to_string())
            );

            guard.set_var(env::MODEL_NAME, "isolated-model");
            assert_eq!(
                std::env::var(env::MODEL_NAME).ok().as_deref(),
                Some("isolated-model")
            );
        }

        assert_eq!(
            outer.get(env::OPENAI_API_KEY).as_deref(),
            Some("sk-from-shell")
        );
        assert_eq!(outer.get(env::MODEL_NAME).as_deref(), Some("from-shell"));
    }
}
