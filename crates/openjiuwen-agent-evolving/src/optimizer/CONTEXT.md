# optimizer

## Intent
Candidate generation and search: the optimizer seam, a deterministic grid baseline, best-so-far tracking, the LLM retry policy, the research (textual-gradient) optimizers, and evolve checkpoint persistence.
Ports the portable parts of Python `openjiuwen.agent_evolving.optimizer` and its checkpointing layer.

## API Surface
### Search
- `Candidate` — `id`, `parameters`, `score`, `generation`; `state()` projects parameters to the JSON object a state-restoring target expects; grid candidates are `cand-{index}` at generation `0`.
- `Optimizer` (async trait) — `name`, `propose(limit)`, `observe(candidate, score)` (default no-op), `best() -> Option<&Candidate>` (default `None`).
- `ParameterSpace` — ordered `BTreeMap<String, Vec<Value>>`; `grid(limit)` enumerates the Cartesian product deterministically (last dimension fastest) capped at `MAX_MATERIALIZED_CANDIDATES` (100 000).
- `GridSearchOptimizer` — deterministic grid baseline that proposes successive chunks and tracks its own best.
- `BestSoFar` / `select_best` — score-maximising best-so-far tracking shared by the trainer and optimizers.
- `LlmInvokePolicy` — pure-data retry policy (`attempt_timeout_secs`, `total_budget_secs`, `max_attempts`, `backoff_base_secs`, `retry_empty_response`).

### Research optimizers (re-exported from `research`)
- `TextualParameter`, `OperatorState`, `OperatorKind`, `ProposalRecord`, `TextualStrategy`, `TextualGradientOptimizer`, `parse_tagged`.
- `InstructionOptimizer`, `ToolOptimizer`, `MemoryOptimizer`, `SkillExperienceOptimizer`, `MetisOptimizer`, `TtseOptimizer` plus their strategies and the constants `CANDIDATE_META_KEY`, `DEFAULT_CLUSTER_THRESHOLD`, `DEFAULT_MAX_PROPOSALS`, `DEFAULT_SECTION_BUDGET`, `DEFAULT_TIP_CAPACITY`, `IMPROVED_TAG`, `MAX_GRADIENT_CHARS`, `PARAMETER_SEPARATOR`.

### Checkpoints
- `EvolveCheckpoint` — Python JSON keys `version`, `run_id`, `step{epoch,batch}`, `best{best_score}`, `seed`, `operators_state`, `updater_state`, `searcher_state`, `last_metrics`; helper types `CheckpointStep`, `CheckpointBest`.
- `CheckpointStore` (async trait) — `save_checkpoint`, `load_checkpoint`, `load_state_dict`; `FileCheckpointStore` (`None` base dir is a silent no-op, atomic temp+rename writes, default `latest.json`), `MemoryCheckpointStore`.
- `CheckpointManager` — `should_save`, `build_checkpoint`, `restore`; `DefaultCheckpointManager`, `CheckpointSnapshot`, `RestoredState`.

## Constraints
- Checkpoint JSON field names and nesting match Python exactly; stores tolerate a missing file by returning `None`.
- File I/O is async (`tokio::fs`) and atomic (temp + rename); no lock is held across an `.await`.
- `should_save` is `(improved && save_on_improve) || epoch % max(save_every_n_epochs, 1) == 0`, so epoch `0` always saves.
- `DefaultCheckpointManager::new()` takes a fresh UUID-hex run id (`with_run_id` overrides it); report and checkpoint contents otherwise never read a clock.
- Every research optimizer is deterministic without a model and falls back to the deterministic rewrite when a model answer cannot be parsed.
- `GridSearchOptimizer` never materializes more than `MAX_MATERIALIZED_CANDIDATES` points and never calls a model.

## Design Decisions
- `EvolveCheckpoint` is a plain serde struct rather than a dataclass mirror; all nine Python fields are required on read, with `CheckpointStep`/`CheckpointBest` giving the nested objects a type.
- `GridSearchOptimizer` materializes lazily in chunks and tracks its own best, so the trainer runs with no ML dependency.
- `RestoredState` carries `operators_state` so the trainer can restore a target snapshot directly, keyed by target id.
- The research optimizers share one engine plus per-domain strategies (`TextualGradientOptimizer<S>`), which keeps backward/propose/observe uniform across instructions, tools, memory, skills, Metis sections and TTSE tips.
- `LlmInvokePolicy` is pure data so retry behaviour can be asserted without sleeping or calling a provider.

## Routing Table
- `./mod.rs` → `Candidate`, `Optimizer`, `ParameterSpace`, `GridSearchOptimizer`, `BestSoFar`, `select_best`, `LlmInvokePolicy`
- `./checkpoint.rs` → `EvolveCheckpoint`, stores, `CheckpointManager`, `DefaultCheckpointManager`
- `./research/` → textual-gradient engine, parameters, and the six domain optimizers (`./research/CONTEXT.md`)
