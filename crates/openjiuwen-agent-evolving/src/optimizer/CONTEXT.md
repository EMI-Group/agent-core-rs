# optimizer

## Intent

Candidate generation, best-so-far selection, and evolve checkpoint persistence.
Ports the portable parts of Python `openjiuwen.agent_evolving.optimizer` (optimizer seam, candidates) and `checkpointing` (`EvolveCheckpoint`, `FileCheckpointStore`, `DefaultCheckpointManager`), replacing ML prompt optimisers with a deterministic grid baseline.

## API Surface

- `Candidate` — `id`, `parameters`, `score`, `generation`; `state()` projects parameters to JSON.
- `Optimizer` — async trait: `name`, `propose(limit)`, `observe`, `best`.
- `ParameterSpace`, `GridSearchOptimizer` — deterministic Cartesian grid baseline.
- `BestSoFar`, `select_best` — score-maximising best-so-far tracking.
- `LlmInvokePolicy` — pure-data retry policy with Python defaults.
- `EvolveCheckpoint` — Python JSON keys `version`, `run_id`, `step`, `best`, `seed`, `operators_state`, `updater_state`, `searcher_state`, `last_metrics`.
- `CheckpointStore` — async save/load/load_state_dict; `FileCheckpointStore` (`None` base dir = no-op, atomic writes), `MemoryCheckpointStore`.
- `CheckpointManager` — `should_save`, `build_checkpoint`, `restore`; `DefaultCheckpointManager`, `CheckpointSnapshot`, `RestoredState`.

## Constraints

- Checkpoint JSON field names and nesting match Python exactly; stores tolerate missing files by returning `None`.
- File IO is async (`tokio::fs`) and atomic (temp + rename); locks are never held across `.await`.
- `should_save` is `(improved && save_on_improve) || epoch % max(save_every_n_epochs, 1) == 0`, so epoch `0` always saves.

## Design Decisions

- `EvolveCheckpoint` is a plain serde struct rather than a dataclass mirror; all nine Python fields are required on read.
- `GridSearchOptimizer` materializes at most 100,000 grid points, produces candidates lazily in chunks, and tracks its own best, so the trainer can run without any ML dependency.
- `RestoredState` carries `operators_state` so the trainer can restore a target snapshot directly.

## Routing Table

- `./mod.rs` → `Candidate`, `Optimizer`, `ParameterSpace`, `GridSearchOptimizer`, `BestSoFar`, `LlmInvokePolicy`
- `./checkpoint.rs` → `EvolveCheckpoint`, stores, `CheckpointManager`, `DefaultCheckpointManager`
