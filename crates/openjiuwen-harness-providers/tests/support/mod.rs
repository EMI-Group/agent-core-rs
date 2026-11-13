//! Shared fixtures for the crate's integration suites.
//!
//! [`FakeCli`] writes a real POSIX shell script that behaves like a vendor CLI: it reads stdin line
//! by line, records every request in a transcript and answers with scripted NDJSON replies. The
//! suites therefore exercise the true subprocess path (pipes, flushing, EOF, signals) without a
//! network, a vendor SDK or a wall-clock dependency.

#![allow(dead_code)] // a suite includes the whole module and uses a subset of the helpers
#![allow(unreachable_pub)] // the fixture is `pub` for `mod support;` inclusion, never a real library
#![allow(clippy::unwrap_used, clippy::expect_used)] // fixtures assert by panicking

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Environment variable the fake CLI receives and logs into.
pub const LOG_ENV: &str = "FAKE_CLI_LOG";

/// A fake vendor CLI: an executable shell script plus its stdin transcript.
pub struct FakeCli {
    /// Owns the script, the transcript and the reply file.
    dir: tempfile::TempDir,
    /// Path of the executable script.
    program: PathBuf,
    /// Path of the transcript the script appends stdin lines to.
    log: PathBuf,
}

impl FakeCli {
    /// Writes an executable `#!/bin/sh` script whose body is `body`, with `$FAKE_CLI_LOG` exported
    /// to the transcript path.
    ///
    /// The body may use the exported variable to record its own diagnostics; stdin is connected as
    /// usual, so a body that never reads it blocks until the harness closes it.
    pub fn script(body: &str) -> Self {
        let cli = Self::new();
        let script = format!(
            "#!/bin/sh\n{LOG_ENV}={}\nexport {LOG_ENV}\n{body}\n",
            quote_shell(&cli.log)
        );
        cli.write_program(&script);
        cli
    }

    /// Writes an executable script that appends every stdin line to the transcript and prints the
    /// next reply line, in order; it exits once stdin reaches EOF.
    ///
    /// Replies are emitted verbatim (they are NDJSON), so malformed replies stay malformed.
    pub fn replies(replies: &[&str]) -> Self {
        let cli = Self::new();
        let replies_path = cli.dir.path().join("replies.ndjson");
        let mut encoded = String::new();
        for reply in replies {
            encoded.push_str(reply);
            encoded.push('\n');
        }
        std::fs::write(&replies_path, encoded).expect("write the scripted replies");
        let script = format!(
            "#!/bin/sh\n{LOG_ENV}={log}\nexport {LOG_ENV}\nexec 3< {replies}\n\
             while IFS= read -r request; do\n\
             \x20 printf '%s\\n' \"$request\" >> \"${LOG_ENV}\"\n\
             \x20 if IFS= read -r reply <&3; then printf '%s\\n' \"$reply\"; fi\n\
             done\n",
            log = quote_shell(&cli.log),
            replies = quote_shell(&replies_path),
        );
        cli.write_program(&script);
        cli
    }

    /// Returns the executable path handed to the transport.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.program
    }

    /// Returns the stdin lines the script recorded, oldest first.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        match std::fs::read_to_string(&self.log) {
            Ok(text) => text.lines().map(str::to_owned).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Returns the environment the script expects (the transcript path).
    #[must_use]
    pub fn env(&self) -> BTreeMap<String, String> {
        BTreeMap::from([(LOG_ENV.to_owned(), self.log.to_string_lossy().into_owned())])
    }

    /// Creates the temporary directory layout.
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let program = dir.path().join("fake-cli");
        let log = dir.path().join("transcript.log");
        Self { dir, program, log }
    }

    /// Writes `script` as the executable program.
    ///
    /// The script is staged under a sibling name in the same directory, made executable there, and
    /// only then renamed onto the final path, so a concurrent `exec` can never observe a
    /// half-written script (`rename` is atomic within one filesystem).
    ///
    /// The bytes are piped to a helper `cat` instead of `std::fs::write` on purpose: an executable
    /// materialized from a descriptor opened in *this* process intermittently fails to run with
    /// `ETXTBSY` ("Text file busy"), because a fake CLI forked by another test thread copies the
    /// whole descriptor table — including the open write descriptor of the script being exec'd. The
    /// tests deliberately spawn processes from many threads at once, so the helper keeps every
    /// write descriptor of an executable out of this process entirely.
    fn write_program(&self, script: &str) {
        let staged = self.dir.path().join("fake-cli.staged");
        let mut helper = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("cat > {}", quote_shell(&staged)))
            .stdin(std::process::Stdio::piped())
            .spawn()
            .expect("start the helper that writes the fake CLI script");
        {
            let mut sink = helper.stdin.take().expect("piped helper stdin");
            sink.write_all(script.as_bytes())
                .expect("write the fake CLI script");
        }
        assert!(
            helper.wait().expect("wait for the helper").success(),
            "the fake CLI script must be written"
        );
        make_executable(&staged);
        std::fs::rename(&staged, &self.program).expect("install the fake CLI script");
    }
}

/// Quotes `value` for a POSIX shell, escaping embedded single quotes.
fn quote_shell(value: &Path) -> String {
    let text = value.to_string_lossy();
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// Marks `path` as executable by its owner.
#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = std::fs::metadata(path)
        .expect("stat the fake CLI script")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).expect("chmod the fake CLI script");
}

/// Marks `path` as executable by its owner (no mode bits exist off Unix).
#[cfg(not(unix))]
fn make_executable(_path: &Path) {}
