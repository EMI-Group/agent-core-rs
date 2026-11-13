# evaluator_pipeline

## Intent
Task-in-container evaluation: prepare a git workspace, build and run a Docker container, execute a test command, score its output, and report per task.
Ports Python `openjiuwen.agent_evolving.evaluator.evaluator_pipeline` (`models.py`, `config.py`, `docker_env.py`, workspace preparation and the driver) over an injectable command seam.

## API Surface
- `CommandRunner` (async trait, `Send + Sync + Debug`) — `async fn run(&self, spec: &CommandSpec, cancel: &CancellationToken) -> Result<ExecResult>` and `async fn which(&self, program: &str) -> Result<Option<PathBuf>>`; `find_program(program) -> Option<PathBuf>` is a free helper, not a trait method.
- `ProcessCommandRunner` (real `tokio::process` runner with timeout and merged output capture) and `FakeCommandRunner` (scripted, offline): `new`, `push_result(program, arg_prefix, ExecResult)`, `with_program_missing`, `record_only`, `invocations`, `command_lines`, `is_program_missing`.
- `FakeCommandRunner::which` answers `/fake/bin/<program>` for every program not declared missing; an unscripted command is `Error::validation` unless `record_only` is set, in which case it succeeds with `ExecResult::ok("")`.
- `CommandSpec` / `ExecResult` — what a command was and what it produced (`stdout`, `stderr`, `returncode`, `timed_out`).
- `DockerEnvironment` / `container_name` / `CONTAINER_NAME_PREFIX` / `DEFAULT_TIMEOUT_MS` — build, start, exec, copy and stop with typed lifecycle errors.
- `WorkspacePreparer` / `GitWorkspaceSpec` / `PreparedWorkspace` / `PATCH_FILE_NAME` — clone (optionally shallow, optionally with submodules), fetch, checkout, patch application and final-commit resolution.
- `ScoreSpec` (`ExitCode`, `Json`, `TestSummary`) / `parse_score` / `ScoredRun` / `TestCounts` / `parse_last_json` / `parse_test_counts` — pure scoring from an `ExecResult`.
- `EvaluationTask` / `TaskReport` / `sanitize_task_id` / `MAX_TASK_ID_SLUG_CHARS` — the evaluated task and its report.
- `EvaluationPipeline` / `PipelineConfig` / `PipelineSummary` / `MIN_ATTEMPTS`/`MAX_ATTEMPTS` — the driver and its batch aggregation.

## Constraints
- Every external side effect goes through `CommandRunner`, so the whole pipeline is testable offline with `FakeCommandRunner` and no test spawns `docker` or `git`.
- A whole-task offline run needs a `record_only` fake plus exactly two scripted replies — `docker run` (stdout must be a non-empty container id) and `docker exec` — because a task without `git_url` still probes `git rev-parse HEAD` and every unscripted `docker cp`/`docker rm` must succeed.
- To exercise the absent-docker path instead, mark `docker` missing with `with_program_missing("docker")`; `ensure_available` then fails before any command is recorded.
- A missing binary maps to `Error::Unsupported` naming the program; the pipeline additionally reports `Error::Unsupported` naming the missing `docker` binary before it starts.
- One task runs: validate task and config, verify `docker` on `PATH`, build the image when a Dockerfile is set, prepare the workspace, start the container and copy the workspace in, execute the test command up to `effective_max_attempts` times, parse the score, then stop the container unless `keep_container` is set — with a cancellation check between every step.
- No clocks and no randomness: the workspace directory and container name derive from the task id, and a report contains only the command result and the attempt count.
- `DockerEnvironment` methods take no cancellation token; the pipeline checks cancellation between operations.

## Design Decisions
- The command seam is the port's main structural difference from Python, whose Docker environment and adapters call `subprocess` directly and are therefore only testable against a real daemon.
- The container starts with `sleep infinity` and bind-mounts the workspace at the container workdir; locale probing, runtime-selected keep-alives and encoding diagnostics from Python are dropped.
- Agent adapters (`base.py`), the LLM evolution loop, skill management and mirror-rewriting image builds are not ported: they need live agents, a model gateway or network access.
- `ScoreSpec` is data instead of one parser per benchmark adapter, so a caller chooses exit-code, JSON or test-summary scoring without new code.
- Task ids are sanitised into a bounded slug (`MAX_TASK_ID_SLUG_CHARS` = 64) before they become workspace paths or container names.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./command.rs` → `CommandRunner`, `ProcessCommandRunner`, `FakeCommandRunner`, `find_program`
- `./docker.rs` → `DockerEnvironment`, `container_name`
- `./git.rs` → `WorkspacePreparer`, `PreparedWorkspace`
- `./models.rs` → `CommandSpec`, `ExecResult`, `GitWorkspaceSpec`, `EvaluationTask`, `TaskReport`
- `./scoring.rs` → `ScoreSpec`, `parse_score`, `ScoredRun`, `TestCounts`
- `./pipeline.rs` → `EvaluationPipeline`, `PipelineConfig`, `PipelineSummary`
