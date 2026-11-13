//! A POSIX `sh` child process that speaks the JSON-lines member protocol.
//!
//! Every fixture is a small script in its own [`TempDir`], with a file name that is unique within
//! the test binary; tests point a process spawner at [`FakeChild::path`] (or
//! [`FakeChild::program`]) and drive the child through stdin/stdout.
//!
//! # Executing the script: `sh <script>`, never `<script>`
//!
//! `execve` refuses a file that *any* process still holds open for writing (`ETXTBSY`, rendered as
//! `Text file busy (os error 26)`).
//! [`FakeChild`] flushes and closes its descriptor before it returns, but the test binary's other
//! threads fork constantly: a thread that forks inside the fixture's write window hands the
//! still-open descriptor to its child, which keeps it open until its own `exec`.
//! Executing a just-written script therefore races every other spawn in the same test binary.
//!
//! Suites that build their own command line run the script through an interpreter
//! ([`FakeChild::sh_command`]): `sh` is a file nobody writes, so its `exec` always succeeds and the
//! race disappears.
//! The SSH suite cannot do that — the SSH spawner renders its own argv and the script stands in for
//! `ssh` itself — so it keeps the direct exec and goes through [`spawn_with_retry`], which repeats a
//! transport-level `ETXTBSY` refusal a few times with a short backoff.
//! The retry lives here, in the fixture: production code never retries a spawn.
//!
//! # `serve` payload extraction
//!
//! The `serve` child extracts the `send` content with `sed` rather than echoing the whole line, so
//! `{"type":"chunk","chunk":{"delta":"hello"}}` reports the content the leader sent.
//! The extraction is a pure string split at `"content":"` … `","immediate":`, which keeps the
//! JSON-escaped text intact and stays deterministic (no shell evaluation of the payload).

use std::fs;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use tempfile::TempDir;

use super::spawn_config::StdioCommand;

/// How often [`spawn_with_retry`] attempts one spawn before it surfaces the error.
pub(crate) const SPAWN_ATTEMPTS: usize = 3;

/// Backoff between those attempts.
///
/// Long enough for a sibling process that inherited the script's write descriptor to reach its own
/// `exec` (or exit), short enough to stay invisible in a green run.
pub(crate) const SPAWN_RETRY_BACKOFF: Duration = Duration::from_millis(20);

/// Keeps every fixture's script name unique within the test binary.
///
/// Test-only global state: an atomic counter, read once per fixture, that exists so two fixtures
/// can never share a path even if their temp directories collided.
static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A temporary member child process script.
pub(crate) struct FakeChild {
    /// Keeps the script alive for the lifetime of the fixture.
    _dir: TempDir,
    /// Absolute path of the script.
    path: PathBuf,
}

impl FakeChild {
    /// Writes a script with the given `sh` body (shebang + `set -u` prepended, mode 0755).
    ///
    /// The script is written once here, fully, and its descriptor is closed before the fixture is
    /// returned; see the module docs for why that alone does not make a direct exec race-free.
    pub(crate) fn with_body(body: &str) -> Self {
        let dir = TempDir::new().expect("temp dir for member child");
        let sequence = FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = dir.path().join(format!("member-child-{sequence}.sh"));
        write_script(&path, &format!("#!/bin/sh\nset -u\n{body}\n"))
            .expect("write member child script");
        Self { _dir: dir, path }
    }

    /// A script that never reads stdin and never exits, for timeout/kill tests.
    pub(crate) fn silent() -> Self {
        Self::with_body("exec sleep 600")
    }

    /// A fully protocol-conforming member: answers start/send/abort/pause/resume/stop.
    pub(crate) fn serve() -> Self {
        Self::with_body(
            r#"while IFS= read -r line; do
  case "$line" in
    *'"type":"start"'*)
      printf '{"type":"ready","session_id":"child-session"}\n'
      printf '{"type":"state","state":"idle"}\n'
      ;;
    *'"type":"send"'*)
      content=$(printf '%s' "$line" | sed -n 's/.*"content":"//;s/","immediate":[a-z]*}$//p')
      printf '{"type":"round","outcome":"started"}\n'
      printf '{"type":"chunk","chunk":{"delta":"%s"}}\n' "$content"
      printf '{"type":"round","outcome":"finished"}\n'
      ;;
    *'"type":"abort"'*)
      printf '{"type":"round","outcome":"aborted"}\n'
      ;;
    *'"type":"pause"'*)
      printf '{"type":"state","state":"paused"}\n'
      ;;
    *'"type":"resume"'*)
      printf '{"type":"state","state":"running"}\n'
      ;;
    *'"type":"stop"'*)
      exit 0
      ;;
  esac
done
exit 0"#,
        )
    }

    /// A member that answers `start` with an error frame and then exits `3`.
    pub(crate) fn failing_start() -> Self {
        Self::with_body(
            r#"read -r _line || true
printf '{"type":"error","message":"start refused"}\n'
exit 3"#,
        )
    }

    /// The script path, suitable for spawning directly.
    ///
    /// Prefer [`Self::sh_command`] (or `sh <path>` in a hand-built command): executing a
    /// just-written script directly can fail with `ETXTBSY` — see the module docs.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// The script path as a `String`, suitable for `StdioCommand::new`/program overrides.
    pub(crate) fn program(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    /// The race-free `sh <script>` launch command for this child.
    ///
    /// `sh` is a binary no test writes, so this command can never hit `ETXTBSY`; prefer it over
    /// pointing a program override at [`Self::program`].
    pub(crate) fn sh_command(&self) -> StdioCommand {
        StdioCommand::new("sh").with_arg(self.program())
    }
}

/// Runs `attempt` up to [`SPAWN_ATTEMPTS`] times, retrying only an `ETXTBSY` transport refusal.
///
/// A retry is safe for every spawner in this node: each registers a runtime only after a successful
/// launch, so a refused attempt leaves no state behind.
/// Every other error — a missing program, a bad spec, a child that exits during the handshake — is
/// returned to the caller immediately.
pub(crate) async fn spawn_with_retry<T, F, Fut>(mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut tries = 1;
    loop {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(error) if tries < SPAWN_ATTEMPTS && is_text_file_busy(&error) => {
                tries += 1;
                tokio::time::sleep(SPAWN_RETRY_BACKOFF).await;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Reports whether `error` is the operating system's `ETXTBSY` refusal to execute a script.
///
/// `std::io::Error` renders `ETXTBSY` as `Text file busy` on every unix this crate targets, and
/// [`super::stdio`] wraps that text in a [`ErrorKind::Transport`] error, so the check is
/// kind-first and then wording.
pub(crate) fn is_text_file_busy(error: &Error) -> bool {
    error.kind() == ErrorKind::Transport && error.to_string().contains("Text file busy")
}

/// Writes `contents` to `path`, flushes it, syncs it and closes the descriptor before returning.
///
/// The file is created executable (0755, before the umask) so the mode is final when the last write
/// lands; [`make_executable`] then pins the exact bits.
fn write_script(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o755);
    }
    let mut file = options.open(path)?;
    file.write_all(contents.as_bytes())?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    make_executable(path)
}

/// Sets the executable bit (0755) on `path`.
fn make_executable(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// The error [`super::super::stdio`] produces when `execve` refuses a busy script.
    fn text_file_busy() -> Error {
        Error::transport(
            "failed to start member process '/tmp/member-child-0.sh': \
             Text file busy (os error 26)",
        )
    }

    #[test]
    fn a_fixture_writes_a_complete_script_with_a_unique_name() {
        let first = FakeChild::with_body("exit 0");
        let second = FakeChild::with_body("exit 0");

        assert_ne!(
            first.path(),
            second.path(),
            "fixtures must not share a path"
        );
        assert_eq!(
            fs::read_to_string(first.path()).expect("read the script"),
            "#!/bin/sh\nset -u\nexit 0\n"
        );
        assert_eq!(first.program(), first.path().to_string_lossy());
        assert_eq!(first.sh_command().program(), "sh");
        assert_eq!(first.sh_command().args().to_vec(), vec![first.program()]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(first.path())
                .expect("script metadata")
                .permissions()
                .mode();
            assert_ne!(mode & 0o111, 0, "the script must be executable: {mode:o}");
        }
    }

    #[test]
    fn a_text_file_busy_error_is_recognised_and_nothing_else_is() {
        assert!(is_text_file_busy(&text_file_busy()));
        assert!(!is_text_file_busy(&Error::config("missing program")));
        assert!(!is_text_file_busy(&Error::transport(
            "member process 'x' stdin is unavailable"
        )));
    }

    #[tokio::test]
    async fn a_text_file_busy_spawn_is_retried() {
        let attempts = Cell::new(0);
        let outcome: Result<u8> = spawn_with_retry(|| {
            attempts.set(attempts.get() + 1);
            let attempt = attempts.get();
            async move {
                if attempt == 1 {
                    Err(text_file_busy())
                } else {
                    Ok(7)
                }
            }
        })
        .await;

        assert_eq!(outcome.expect("the second attempt succeeds"), 7);
        assert_eq!(attempts.get(), 2);
    }

    #[tokio::test]
    async fn a_non_etxtbsy_error_is_not_retried() {
        let attempts = Cell::new(0);
        let outcome: Result<u8> = spawn_with_retry(|| {
            attempts.set(attempts.get() + 1);
            async move { Err(Error::config("missing program")) }
        })
        .await;

        assert_eq!(
            outcome.expect_err("a config error is final").kind(),
            ErrorKind::Config
        );
        assert_eq!(attempts.get(), 1);
    }

    #[tokio::test]
    async fn the_retry_is_bounded() {
        let attempts = Cell::new(0);
        let outcome: Result<u8> = spawn_with_retry(|| {
            attempts.set(attempts.get() + 1);
            async move { Err(text_file_busy()) }
        })
        .await;

        assert!(outcome.is_err(), "a permanent ETXTBSY must surface");
        assert_eq!(attempts.get(), SPAWN_ATTEMPTS);
    }
}
