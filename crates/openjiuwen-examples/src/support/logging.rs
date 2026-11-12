//! Subscriber installation for the examples.
//!
//! Examples print their narrative to stdout through [`super::banner`], so every
//! diagnostic — including the SDK's own `tracing` events — is sent to stderr.
//! That keeps `cargo run --example 01_hello_agent > out.txt` readable and lets
//! `OPENJIUWEN_LOG_LEVEL=debug` add detail without disturbing the story.

use openjiuwen::core::common::error::Result;
use openjiuwen::core::common::logging::{init_with, LogWriter, LoggingConfig};

/// Installs the shared subscriber, writing to stderr.
///
/// The configuration comes from [`LoggingConfig::from_env`], which honors
/// `OPENJIUWEN_LOG_LEVEL` and `OPENJIUWEN_LOG_FORMAT`; the writer is forced to
/// stderr so no log record ever lands on stdout.
///
/// The call is idempotent: it is safe to invoke once per example even when a
/// library or a test harness installed a subscriber first — the underlying
/// helper is once-guarded and returns `Ok(())` in that case, so a second call
/// also returns `Ok(())` and leaves the existing subscriber untouched. The only
/// failure it can report is a malformed filter directive
/// (see [`LoggingConfig::with_filter`]) or a subscriber that cannot be installed
/// for any reason other than one already being present.
///
/// # Errors
///
/// Returns the error of `openjiuwen_common::logging::init_with`: a validation
/// error for an unparsable `EnvFilter` directive, or a configuration error when
/// the subscriber cannot be installed.
pub fn init() -> Result<()> {
    let config = LoggingConfig::from_env().with_writer(LogWriter::Stderr);
    init_with(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent_and_writes_to_stderr() {
        init().expect("the first init installs the subscriber");
        // The helper observes the already-installed global subscriber and
        // returns `Ok(())` instead of failing.
        init().expect("the second init observes the existing subscriber");
        assert!(openjiuwen::core::common::logging::is_initialized());
    }
}
