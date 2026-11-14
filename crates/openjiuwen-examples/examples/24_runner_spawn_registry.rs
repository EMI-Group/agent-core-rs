//! `24_runner_spawn_registry` — the runner's spawn registry: two live backends side
//! by side, routing, health, and the typed deferred path.
//!
//! A [`SpawnerRegistry`] is the multi-backend form of the spawn seam. It hosts one
//! [`AgentSpawner`] per [`SpawnBackend`] and resolves a [`SpawnSpec`] to one of them:
//! the spec's `backend` hint first, then the registry default, then a sole registered
//! backend — and never a silent fallback across backends. This example registers two
//! live backends and drives both through one `Runner`:
//!
//! 1. **in-process** — an [`InProcessHandler`] bound to this process; it has no child,
//!    reports no pid, and its `health()` probe is a pure read of its own liveness;
//! 2. **external command** — an [`ExternalCommandSpawner`] that re-executes *this same
//!    binary* with [`CHILD_ENV`] set, so the demo ships its own child and the parent
//!    exercises the real one-JSON-object-per-line protocol over pipes.
//!
//! The frame exchange is real, not simulated: the parent writes the bootstrap frame
//! (`input` carrying the spawn spec), then one `input` frame per invocation; the child
//! answers `output` frames, `health_check_response` frames, and a final `shutdown_ack`.
//! That is why `main` decides child mode **before** anything else — in child mode
//! stdout is the protocol pipe, so it must stay clean of banners and logs.
//!
//! Dispatch goes through `Runner::spawn_agent` / `Runner::spawn_agent_with`, because a
//! registry alone does not fill `Runner::active_spawns`: the runner admits each
//! successful spawn as a [`RunKind::Spawn`] run and retires it when something observes
//! the end — a shutdown, a kill, a health probe that finds the child gone, or a drop of
//! the handle. The example therefore shuts both children down and asserts the active
//! list empties, so no child process leaks.
//!
//! Offline by default: loopback-free, the only transport is a pair of pipes to a child
//! that is this binary. There is **no Python counterpart** for the registry as a whole;
//! the seam ports `openjiuwen/core/runner/spawn/`.
//!
//! [`AgentSpawner`]: openjiuwen::core::runner::AgentSpawner
//! [`InProcessHandler`]: openjiuwen::core::runner::InProcessHandler
//! [`ExternalCommandSpawner`]: openjiuwen::core::runner::ExternalCommandSpawner

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen::core::common::error::Result as SdkResult;
use openjiuwen::core::runner::{
    ExternalCommandSpawner, InProcessHandler, InProcessSpawner, RunStatus, Runner, SpawnBackend,
    SpawnHealth, SpawnMessage, SpawnMessageKind, SpawnSpec, SpawnedAgent, SpawnerRegistry,
};
use openjiuwen::core::single_agent::output::{AgentChunk, AgentOutput};
use serde_json::{json, Value};

use common::banner;

/// Environment marker that turns this binary into the NDJSON child.
///
/// The parent sets it in the child's environment only, through
/// `ExternalCommandSpawner::with_env`, so a normal run is never in child mode.
const CHILD_ENV: &str = "OPENJIUWEN_SPAWN_DEMO_CHILD";

/// Prefix the in-process handler adds to its answers.
const IN_PROCESS_REPLY: &str = "in-process echo";

/// Prefix the external child adds to its answers.
const CHILD_REPLY: &str = "external child echo";

/// Session the in-process spawn binds to.
const IN_PROCESS_SESSION: &str = "demo-session";

/// Session the external child binds to.
const CHILD_SESSION: &str = "child-session";

/// The question both agents are asked, so the two answers differ only by backend.
const QUESTION: &str = "which backend answered this?";

/// Deadline for one request/reply exchange with the child.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Deadline for one on-demand health probe of the child.
const HEALTH_TIMEOUT: Duration = Duration::from_millis(300);

/// Consecutive failed probes that would kill the child, were the loop enabled.
const HEALTH_FAILURE_LIMIT: u32 = 3;

/// Deadline for the external child's graceful exit.
const CHILD_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Deadline for the in-process handler to stop.
const IN_PROCESS_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Child mode: this binary speaking the spawn protocol on stdin/stdout.
// ---------------------------------------------------------------------------

/// Runs the NDJSON child until the parent shuts it down or closes the pipe.
///
/// Blocking `std::io` on purpose: the example's `tokio` has no `io-std` feature, and
/// a dedicated process has nothing else to do while it waits for a frame. Stdout
/// carries frames only; diagnostics go to stderr, which the parent drains into its
/// debug log.
fn run_child() -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line.context("reading a frame from the parent failed")?;
        if line.trim().is_empty() {
            continue;
        }
        let frame = match SpawnMessage::from_line(&line) {
            Ok(frame) => frame,
            Err(error) => {
                // A malformed line is the parent's bug, not a reason to die: the
                // framing layer on the parent side skips noise, and so does the child.
                eprintln!("24 child: ignoring an unreadable frame ({error})");
                continue;
            }
        };

        match frame.kind {
            // An invocation answers with one terminal `output` frame.
            SpawnMessageKind::Input => {
                if let Some(inputs) = frame.payload.get("inputs") {
                    let answer = AgentOutput::answer(format!("{CHILD_REPLY}: {inputs}"));
                    write_child_frame(&mut out, SpawnMessageKind::Output, answer.to_value())?;
                } else {
                    // The first `input` is the bootstrap frame (`{"spawn_spec": …}`); the
                    // child takes note of it and replies nothing, so the parent's spawn
                    // returns without waiting for a frame.
                    let class = frame
                        .payload
                        .pointer("/spawn_spec/agent_class")
                        .and_then(Value::as_str)
                        .unwrap_or("<unnamed>");
                    eprintln!("24 child: bootstrapped as class '{class}'");
                }
            }
            SpawnMessageKind::HealthCheck => {
                write_child_frame(
                    &mut out,
                    SpawnMessageKind::HealthCheckResponse,
                    json!({ "healthy": true }),
                )?;
            }
            SpawnMessageKind::Shutdown => {
                // Acknowledge first, then leave: the parent waits for exactly this
                // frame (or for EOF) before reaping the child.
                write_child_frame(&mut out, SpawnMessageKind::ShutdownAck, Value::Null)?;
                return Ok(());
            }
            other => eprintln!("24 child: ignoring a '{other:?}' frame"),
        }
    }

    // EOF without a `shutdown`: the parent closed the pipe, which is a clean exit.
    Ok(())
}

/// Writes one frame to the protocol pipe and flushes it.
///
/// The flush is what makes the parent's read progress; a buffered frame would look
/// like a hung exchange.
fn write_child_frame(out: &mut impl Write, kind: SpawnMessageKind, payload: Value) -> Result<()> {
    let line = SpawnMessage::new(kind, payload).to_line()?;
    out.write_all(line.as_bytes())
        .context("writing a frame to the parent failed")?;
    out.flush()
        .context("flushing a frame to the parent failed")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The in-process half of the demo.
// ---------------------------------------------------------------------------

/// An agent bound to this process: it echoes its inputs and has no transport.
struct EchoHandler {
    /// Whether the handler still serves calls; `shutdown` clears it.
    alive: AtomicBool,
}

impl EchoHandler {
    /// A live handler as the shared handle the in-process spawner hosts.
    fn shared() -> Arc<dyn InProcessHandler> {
        Arc::new(Self {
            alive: AtomicBool::new(true),
        })
    }
}

#[async_trait]
impl InProcessHandler for EchoHandler {
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    async fn send(&self, inputs: Value) -> SdkResult<AgentOutput> {
        Ok(AgentOutput::answer(format!("{IN_PROCESS_REPLY}: {inputs}")))
    }

    fn stream(&self, inputs: Value) -> BoxStream<'static, SdkResult<AgentChunk>> {
        let chunk = AgentChunk::llm_output(0, format!("{IN_PROCESS_REPLY}: {inputs}"));
        Box::pin(futures::stream::iter([Ok(chunk)]))
    }

    async fn shutdown(&self) -> SdkResult<()> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parent mode.
// ---------------------------------------------------------------------------

/// Child mode is decided first: in child mode stdout is the protocol pipe.
fn main() -> Result<()> {
    if std::env::var(CHILD_ENV).is_ok_and(|value| value == "1") {
        return run_child();
    }

    // A plain runtime rather than `#[tokio::main]`, because the child branch above
    // must run without a runtime and without writing a single narrative line.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("cannot build the example's tokio runtime")?;
    runtime.block_on(parent())
}

/// The parent demo: build the registry, spawn both children, probe, and clean up.
async fn parent() -> Result<()> {
    let mode = common::setup()?;
    banner::title("24 runner spawn registry");

    if mode.is_live() {
        banner::note(
            "live mode is set, but this example is offline by construction: the child is this \
             same binary",
        );
    }
    banner::note(&format!(
        "run mode: {} (neither backend needs a model: both answer without one)",
        mode.describe()
    ));
    banner::note(
        "two live backends: an in-process handler bound to this process, and an external command \
         that re-executes this binary as its JSON-per-line child",
    );

    let registry = build_registry()?;
    narrate_routing(&registry);

    banner::step("installing the registry on a runner");
    let runner = Runner::new();
    runner.set_spawner_registry(Arc::clone(&registry));
    banner::note(&format!(
        "spawn_backends(): {} (registry installed: {})",
        join_backends(&runner.spawn_backends()),
        runner.spawner_registry().is_some(),
    ));

    let (in_process, child) = spawn_both(&runner).await?;
    let run_ids = probe_and_list(&runner, in_process.as_ref(), child.as_ref()).await?;
    report_deferred(&runner).await?;
    stop_children(&runner, child.as_ref(), in_process.as_ref()).await?;
    report_terminal_outcomes(&runner, &run_ids)?;
    what_this_showed();

    Ok(())
}

/// Builds the registry: both live backends, the in-process default, and the checks a
/// host runs after wiring it.
fn build_registry() -> Result<Arc<SpawnerRegistry>> {
    banner::step("an empty registry defers every spawn");
    let registry = Arc::new(SpawnerRegistry::new());
    banner::note(&format!(
        "new(): len={} is_empty={} default={}",
        registry.len(),
        registry.is_empty(),
        registry
            .default_backend()
            .map_or_else(|| "<none>".to_owned(), |backend| backend.to_string()),
    ));

    banner::step("registering the two live backends");
    registry.register_in_process(InProcessSpawner::from_handler(EchoHandler::shared()));

    let exe = std::env::current_exe().context("cannot locate this example's own binary")?;
    let external = ExternalCommandSpawner::new(exe.to_string_lossy().into_owned())
        .with_env(CHILD_ENV, "1")
        .with_request_timeout(REQUEST_TIMEOUT)
        // A zero interval disables the background probe task; `health()` still probes
        // on demand, which is what this example does.
        .with_health_check(Duration::ZERO, HEALTH_TIMEOUT, HEALTH_FAILURE_LIMIT);
    // `resolve` is the same validation a spawn performs, so a bad path fails here
    // instead of in the middle of the demo.
    let program = external
        .resolve()
        .context("the external child program is not executable")?;
    banner::note(&format!("external program: {}", program.display()));
    banner::note(
        "health-check interval 0 s: no background probe, so every probe below is requested by \
         this example",
    );
    registry.register_external_command(external);

    registry
        .set_default(SpawnBackend::InProcess)
        .context("the in-process backend cannot become the default")?;
    ensure!(
        registry.len() == 2,
        "expected two registered backends, found {}",
        registry.len()
    );
    banner::note(&format!(
        "backends(): {} (default {})",
        join_backends(&registry.backends()),
        registry
            .default_backend()
            .map_or_else(|| "<none>".to_owned(), |backend| backend.to_string()),
    ));
    banner::note(&format!(
        "contains: in_process={} external_command={} remote_http={}",
        registry.contains(SpawnBackend::InProcess),
        registry.contains(SpawnBackend::ExternalCommand),
        registry.contains(SpawnBackend::RemoteHttp),
    ));
    Ok(registry)
}

/// Prints how the registry resolves three specs: no hint, a registered hint, and an
/// unregistered hint that must not fall back to another backend.
fn narrate_routing(registry: &SpawnerRegistry) {
    banner::step("routing a spec: hint, then default, then the sole backend");
    let (default_backend, default_spawner) = registry.route(&SpawnSpec::default());
    banner::note(&format!("no hint -> {default_backend}"));
    banner::note(&format!("describe(): {}", default_spawner.describe()));

    let hinted = SpawnSpec::default().with_backend(SpawnBackend::ExternalCommand);
    let (hinted_backend, _) = registry.route(&hinted);
    banner::note(&format!("hint external_command -> {hinted_backend}"));

    let unrouted = SpawnSpec::default().with_backend(SpawnBackend::RemoteHttp);
    let (unrouted_backend, unrouted_spawner) = registry.route(&unrouted);
    banner::note(&format!(
        "hint remote_http (not registered) -> {unrouted_backend}, no fallback to another backend"
    ));
    banner::note(&format!("describe(): {}", unrouted_spawner.describe()));

    // A default the registry cannot honour is refused rather than stored.
    match registry.set_default(SpawnBackend::RemoteHttp) {
        Ok(()) => banner::note("unexpected: an unregistered backend became the default"),
        Err(error) => banner::note(&format!(
            "set_default(remote_http) refused: {error} (kind {})",
            error.kind()
        )),
    }
}

/// Spawns both children through the runner and exercises each one once.
async fn spawn_both(runner: &Runner) -> Result<(Arc<dyn SpawnedAgent>, Arc<dyn SpawnedAgent>)> {
    banner::step("spawning the in-process agent");
    let in_process = runner
        .spawn_agent(
            SpawnSpec::class_agent("demo", "Echo")
                .with_backend(SpawnBackend::InProcess)
                .with_session_id(IN_PROCESS_SESSION),
        )
        .await
        .context("the in-process backend refused the spawn")?;
    banner::note(&format!(
        "handle {} backend={} pid_is_set={}",
        in_process.id(),
        in_process.backend(),
        in_process.pid().is_some(),
    ));
    let local = in_process
        .send(json!({ "query": QUESTION }))
        .await
        .context("the in-process agent failed")?;
    banner::note(&format!("in-process answer: {}", local.output));

    banner::step("spawning the external child");
    let child = runner
        .spawn_agent_with(
            SpawnBackend::ExternalCommand,
            SpawnSpec::class_agent("demo", "Child").with_session_id(CHILD_SESSION),
        )
        .await
        .context("the external-command backend refused the spawn")?;
    banner::note(&format!(
        "handle {} backend={} pid_is_set={}",
        child.id(),
        child.backend(),
        child.pid().is_some(),
    ));

    banner::step("one frame round trip with the child");
    let request = json!({ "query": QUESTION });
    banner::note(&format!("parent writes: input {{\"inputs\": {request}}}"));
    let reply = child
        .send(request)
        .await
        .context("the child did not answer the input frame")?;
    banner::note(&format!(
        "child answers: one output frame carrying {} byte(s) of text",
        reply.output.len()
    ));
    banner::answer(&reply.output);

    Ok((in_process, child))
}

/// Probes both children, prints their health snapshots, then the active spawn table.
async fn probe_and_list(
    runner: &Runner,
    in_process: &dyn SpawnedAgent,
    child: &dyn SpawnedAgent,
) -> Result<Vec<String>> {
    banner::step("health: one pure read, one real probe");
    let local_probe = in_process
        .health()
        .await
        .context("the in-process probe failed")?;
    print_health("in-process", local_probe, &in_process.health_snapshot());
    let child_probe = child.health().await.context("the child probe failed")?;
    print_health("external", child_probe, &child.health_snapshot());
    banner::note(
        "the child's probe wrote a health_check frame and read the health_check_response frame",
    );

    banner::step("active spawns");
    let active = runner.active_spawns();
    for status in &active {
        print_spawn_row(status);
    }
    ensure!(
        active.len() == 2,
        "expected two active spawns, found {}",
        active.len()
    );
    Ok(active.iter().map(|status| status.run_id.clone()).collect())
}

/// Shows the typed deferred path: an unregistered backend fails, and admits no run.
async fn report_deferred(runner: &Runner) -> Result<()> {
    banner::step("the deferred backend");
    match runner
        .spawn_agent_with(SpawnBackend::RemoteMq, SpawnSpec::default())
        .await
    {
        Ok(unexpected) => {
            banner::note("unexpected: the deferred backend produced a child; stopping it");
            unexpected.force_kill().await?;
        }
        Err(error) => banner::note(&format!(
            "spawn_agent_with(remote_mq) failed: {error} (kind {})",
            error.kind()
        )),
    }
    ensure!(
        runner.active_spawns().len() == 2,
        "a failed spawn must not be admitted as an active run"
    );
    Ok(())
}

/// Stops both children and proves the active table emptied, so nothing leaks.
async fn stop_children(
    runner: &Runner,
    child: &dyn SpawnedAgent,
    in_process: &dyn SpawnedAgent,
) -> Result<()> {
    banner::step("stopping both children");
    let child_stopped = child
        .shutdown(CHILD_SHUTDOWN_TIMEOUT)
        .await
        .context("the external child refused to stop")?;
    banner::note(&format!("external child graceful exit: {child_stopped}"));
    let local_stopped = in_process
        .shutdown(IN_PROCESS_SHUTDOWN_TIMEOUT)
        .await
        .context("the in-process agent refused to stop")?;
    banner::note(&format!("in-process agent stopped: {local_stopped}"));

    // `active_spawns` is an observation, so the list only empties once the shutdowns
    // above reported the end of each run; an idle dead child would still be listed.
    let remaining = runner.active_spawns();
    ensure!(
        remaining.is_empty(),
        "{} spawn(s) stayed active after shutdown",
        remaining.len()
    );
    banner::note("active_spawns() is empty: neither child leaked");
    Ok(())
}

/// Prints how each finished spawn is retained by the runner's monitor.
fn report_terminal_outcomes(runner: &Runner, run_ids: &[String]) -> Result<()> {
    banner::step("terminal outcomes");
    for run_id in run_ids {
        let status = runner
            .monitor()
            .snapshot(run_id)
            .with_context(|| format!("no retained status for run {run_id}"))?;
        banner::note(&format!(
            "run {run_id} -> {:?} {}",
            status.phase, status.detail
        ));
    }
    Ok(())
}

/// The closing summary: the four behaviours this example demonstrated.
fn what_this_showed() {
    banner::step("what this showed");
    banner::note(
        "a registry hosts several backends and routes a spec by its hint, then the default, then a \
         sole backend",
    );
    banner::note(
        "an in-process handler has no pid and its probe is a pure read; an external command is a \
         real child and its probe is a frame",
    );
    banner::note(
        "the runner admits every spawned child as a spawn run, so active_spawns() lists it until a \
         shutdown, kill, drop, or probe observes the end",
    );
    banner::note(
        "an unregistered backend resolves to a deferred spawner whose spawn fails with a typed \
         Unsupported error naming the missing transport",
    );
    banner::note(
        "run with OPENJIUWEN_LOG_LEVEL=debug to see the child's own stderr diagnostics in the \
         parent's log",
    );
}

/// Joins backend names for one narrative line.
fn join_backends(backends: &[SpawnBackend]) -> String {
    backends
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Prints one health snapshot beside the probe result it belongs to.
fn print_health(label: &str, probe: bool, health: &SpawnHealth) {
    banner::note(&format!(
        "{label}: probe={probe} backend={} alive={} healthy={} pid={} last_error={}",
        health.backend,
        health.alive,
        health.healthy,
        if health.pid.is_some() { "Some" } else { "None" },
        health.last_error.as_deref().unwrap_or("<none>"),
    ));
}

/// Prints one row of `Runner::active_spawns`.
///
/// The detail is a caller-facing JSON object; the pid is rendered as presence only,
/// because the numeric pid is ephemeral and differs between runs.
fn print_spawn_row(status: &RunStatus) {
    let backend = status
        .detail
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("<none>");
    let pid =
        status.detail.get("pid").map_or(
            "None",
            |value| {
                if value.is_null() {
                    "None"
                } else {
                    "Some"
                }
            },
        );
    banner::note(&format!(
        "run {} kind={:?} phase={:?} backend={backend} pid={pid} session={}",
        status.run_id,
        status.kind,
        status.phase,
        status.session_id.as_deref().unwrap_or("<none>"),
    ));
}
