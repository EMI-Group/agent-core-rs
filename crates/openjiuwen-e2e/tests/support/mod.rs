//! Shared helpers for the `openjiuwen-e2e` scenario binaries.
//!
//! Each `tests/*.rs` file is its own crate, so a helper used by only one
//! scenario would warn here; the allow below keeps the seam stable as suites
//! are added or removed.
//!
//! This module deliberately stays tiny: it holds only the cross-scenario
//! *timing and stream* contract. Fixtures for models, workspaces, sessions and
//! fake runtimes live in `openjiuwen-test-support` or next to the scenario that
//! needs them, so no scenario pays for another scenario's helpers.
#![allow(dead_code, unreachable_pub, clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::pin::pin;
use std::time::Duration;

use futures::Stream;
use futures::StreamExt;

/// Upper bound for any single await in a scenario.
///
/// A deadline only proves that a hang becomes a failure: no scenario uses a
/// sleep or a timeout to *order* events, it waits for the event itself.
pub const BOUND: Duration = Duration::from_secs(10);

/// Awaits `future`, panicking with `what` if it outlives [`BOUND`].
pub async fn within<F: Future>(what: &str, future: F) -> F::Output {
    match tokio::time::timeout(BOUND, future).await {
        Ok(value) => value,
        Err(elapsed) => panic!("{what} did not complete within {BOUND:?}: {elapsed}"),
    }
}

/// Collects at most `max` items from `stream`, ending early when it closes.
///
/// Every read is bounded by [`BOUND`], so a stream that stalls mid-scenario
/// fails instead of hanging. Scenarios assert on the collected frames rather
/// than sleeping and hoping for a count.
pub async fn collect_bounded<S, T>(stream: S, max: usize) -> Vec<T>
where
    S: Stream<Item = T>,
{
    let mut out = Vec::new();
    let mut stream = pin!(stream);
    while out.len() < max {
        match tokio::time::timeout(BOUND, stream.next()).await {
            Ok(Some(item)) => out.push(item),
            Ok(None) => break,
            Err(elapsed) => panic!("stream produced fewer than {max} items within {BOUND:?}: {elapsed}"),
        }
    }
    out
}
