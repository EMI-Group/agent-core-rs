//! A POSIX `sh` child process that speaks the JSON-lines member protocol.
//!
//! Every fixture is a small executable script in a [`TempDir`]; tests point a process spawner at
//! [`FakeChild::path`] (or [`FakeChild::program`]) and drive it through stdin/stdout.
//!
//! The `serve` child extracts the `send` content with `sed` rather than echoing the whole line, so
//! `{"type":"chunk","chunk":{"delta":"hello"}}` reports the content the leader sent.
//! The extraction is a pure string split at `"content":"` … `","immediate":`, which keeps the
//! JSON-escaped text intact and stays deterministic (no shell evaluation of the payload).

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

/// A temporary member child process script.
pub(crate) struct FakeChild {
    /// Keeps the script alive for the lifetime of the fixture.
    _dir: TempDir,
    /// Absolute path of the executable script.
    path: PathBuf,
}

impl FakeChild {
    /// Writes a script with the given `sh` body (shebang + `set -u` prepended, mode 0755).
    pub(crate) fn with_body(body: &str) -> Self {
        let dir = TempDir::new().expect("temp dir for member child");
        let path = dir.path().join("member-child.sh");
        fs::write(&path, format!("#!/bin/sh\nset -u\n{body}\n"))
            .expect("write member child script");
        make_executable(&path).expect("mark member child script executable");
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
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// The script path as a `String`, suitable for `StdioCommand::new`/program overrides.
    pub(crate) fn program(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }
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
