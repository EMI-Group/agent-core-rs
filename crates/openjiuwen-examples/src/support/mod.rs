//! Shared setup for the runnable examples.
//!
//! Every example starts with [`setup`], which loads an optional `.env`, installs
//! the stderr subscriber, prints the banner, and reports whether the run is
//! offline (the default) or live. The three child modules expose the pieces so
//! they can be used and tested on their own.

pub mod banner;
pub mod logging;
pub mod mode;

use openjiuwen::core::common::error::Result;

pub use self::mode::RunMode;

use self::mode::Evaluation;

/// Prepares an example to run and returns the detected [`RunMode`].
///
/// In order, this:
///
/// 1. loads `.env` from the current directory and its parents when present,
///    ignoring a missing file — a developer convenience, never a requirement;
/// 2. installs the stderr subscriber through [`logging::init`], which is
///    idempotent;
/// 3. prints the `openjiuwen examples` title and the run mode through
///    [`banner`], and, when live mode was requested but its prerequisites are
///    missing, a note naming the absent variables;
/// 4. returns the mode, which is [`RunMode::Offline`] in that degraded case so
///    the example keeps running against its scripted fakes.
///
/// # Errors
///
/// Returns the error of [`logging::init`] when the subscriber cannot be
/// installed; everything else about the environment is handled by degrading to
/// offline, never by failing.
pub fn setup() -> Result<RunMode> {
    // A missing `.env` is the normal case (CI, a fresh checkout), so the result
    // is deliberately ignored.
    let _ = dotenvy::dotenv();

    logging::init()?;

    let Evaluation { mode, missing } = mode::detect();

    banner::title("openjiuwen examples");
    banner::note(&format!("run mode: {}", mode.describe()));
    if !missing.is_empty() {
        banner::note(&format!(
            "{} is set but {} is missing; continuing offline",
            mode::LIVE_FLAG,
            missing.join(", ")
        ));
    }

    Ok(mode)
}
