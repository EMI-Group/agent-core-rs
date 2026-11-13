//! Dedicated-thread HTTP transport for OTLP/JSON.
//!
//! `opentelemetry_sdk` 0.31 runs `SpanExporter::export` from its batch
//! processor on a plain `std::thread` driven by `futures_executor::block_on`,
//! where no tokio reactor is installed — building or driving a `reqwest`
//! request there panics with "there is no reactor running". The transport
//! therefore owns one `std` thread that builds a current-thread tokio runtime
//! and the HTTP client, and runs every POST on that runtime.
//!
//! Jobs travel over an unbounded tokio channel: the sender is
//! `Send + Sync + Clone`, and the worker drains the channel with
//! `blocking_recv` (legal there because the thread sits outside any runtime
//! context). Each job carries a pre-serialized body plus a oneshot sender, so
//! `export` can await the outcome under any executor.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use reqwest::header::{HeaderMap, CONTENT_TYPE};
use tokio::sync::mpsc as job_channel;
use tokio::sync::oneshot;
use tracing::{debug, warn};

/// `tracing` target for transport failures.
const LOG_TARGET: &str = "openjiuwen_extensions::otel::otlp";

/// Thread name of the worker, visible in thread listings and panic output.
pub(super) const WORKER_THREAD_NAME: &str = "openjiuwen-otlp-http";

/// How long the worker lets its runtime drain after the last job.
const RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(100);

/// Maximum number of response-body characters quoted in an error message.
const MAX_ERROR_BODY: usize = 512;

/// One export job handed to the worker thread.
#[derive(Debug)]
pub(super) struct Job {
    /// The pre-serialized OTLP/JSON request body.
    pub(super) body: String,
    /// Where the worker reports the outcome of the POST.
    pub(super) reply: oneshot::Sender<Result<(), String>>,
}

/// Sending half of the job channel.
pub(super) type JobSender = job_channel::UnboundedSender<Job>;

/// Handle to the exporter's worker thread.
///
/// Dropping the handle without [`Worker::shutdown`] detaches the thread; it
/// exits as soon as the job channel closes, so an exporter dropped without a
/// shutdown never leaks a running thread.
pub(super) struct Worker {
    /// The worker's completion signal; `None` once joined or detached.
    done: Option<Receiver<()>>,
    /// The worker thread; `None` once joined.
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    /// Starts the worker and returns it with the job sender.
    ///
    /// # Errors
    ///
    /// The [`std::io::Error`] of the failed thread spawn (for example a
    /// resource limit).
    pub(super) fn spawn(
        url: String,
        headers: HeaderMap,
        timeout: Duration,
    ) -> std::io::Result<(Self, JobSender)> {
        let (jobs, mut incoming) = job_channel::unbounded_channel::<Job>();
        let (done, done_rx) = mpsc::channel::<()>();
        let thread = thread::Builder::new()
            .name(WORKER_THREAD_NAME.to_owned())
            .spawn(move || {
                run(&url, &headers, timeout, &mut incoming);
                // The receiver is dropped with this closure, which also closes
                // every queued reply channel.
                drop(incoming);
                let _ = done.send(());
            })?;
        Ok((
            Self {
                done: Some(done_rx),
                thread: Some(thread),
            },
            jobs,
        ))
    }

    /// Waits for the worker to finish, detaching it once `timeout` expires.
    ///
    /// The caller is expected to have dropped every [`JobSender`] first: the
    /// worker exits after the last job, so the wait is bounded by the client
    /// timeout of the request in flight plus the configured budget.
    pub(super) fn shutdown(&mut self, timeout: Duration) {
        if let Some(done) = self.done.take() {
            match done.recv_timeout(timeout) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    warn!(
                        target: LOG_TARGET,
                        timeout = ?timeout,
                        "the OTLP HTTP worker did not stop within the shutdown timeout; detaching it"
                    );
                    return;
                }
            }
        }
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                warn!(target: LOG_TARGET, "the OTLP HTTP worker thread panicked");
            }
        }
    }
}

/// The worker loop: own a runtime, own a client, serve jobs until the channel
/// closes.
fn run(
    url: &str,
    headers: &HeaderMap,
    timeout: Duration,
    jobs: &mut job_channel::UnboundedReceiver<Job>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            warn!(
                target: LOG_TARGET,
                %error,
                "failed to build the OTLP HTTP worker runtime; span export is disabled"
            );
            return;
        }
    };
    let client = match reqwest::Client::builder().timeout(timeout).build() {
        Ok(client) => client,
        Err(error) => {
            warn!(
                target: LOG_TARGET,
                %error,
                "failed to build the OTLP HTTP client; span export is disabled"
            );
            return;
        }
    };
    while let Some(job) = jobs.blocking_recv() {
        let outcome = runtime.block_on(post(&client, url, headers, &job.body));
        // A closed reply channel only means the exporter was dropped while the
        // request was in flight; the span is already lost either way.
        if job.reply.send(outcome).is_err() {
            debug!(
                target: LOG_TARGET,
                "dropped an OTLP export result because the caller is gone"
            );
        }
    }
    drop(client);
    runtime.shutdown_timeout(RUNTIME_SHUTDOWN_TIMEOUT);
}

/// Posts one OTLP/JSON body and maps the response to an error message.
async fn post(
    client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    body: &str,
) -> Result<(), String> {
    let request = client
        .post(url)
        .headers(headers.clone())
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_owned());
    let response = request.send().await.map_err(|error| describe(&error))?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    // The status is the signal; a body that cannot be read must not mask it.
    let body = response.text().await.unwrap_or_default();
    Err(format!(
        "OTLP HTTP export rejected with status {status}: {}",
        truncate_body(&body)
    ))
}

/// Describes a transport failure, naming the phase when it is known.
fn describe(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        format!("OTLP HTTP export timed out: {error}")
    } else if error.is_connect() {
        format!("OTLP HTTP export could not connect: {error}")
    } else {
        format!("OTLP HTTP export failed: {error}")
    }
}

/// Caps a response body at [`MAX_ERROR_BODY`] characters.
fn truncate_body(body: &str) -> String {
    let mut characters = body.chars();
    let mut kept: String = characters.by_ref().take(MAX_ERROR_BODY).collect();
    if characters.next().is_some() {
        kept.push('…');
    }
    kept
}
