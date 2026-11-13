# attribution

## Intent
Turning a run's outcome into per-node credit: the diagnostic-signal records, the dependency graph, backward credit assignment, deterministic keyword detection, and the reward-stage adapter.
Ports the offline part of Python `openjiuwen.agent_evolving.attribution`: `model.py`, `nodes.py`, `closure.py`, `keyword_detect.py` and `reward_stage/`.
Every algorithm here runs without a model; the LLM diagnosis agent and its prompt packs are not ported.

## API Surface
- `AttributionNode`, `AttributionEdge`, `AttributionGraph` — caller-supplied step records, "the `from` step produced the input the `to` step consumed" edges, and the traversal surface (`parents`, `children`, `roots`, `leaves`, `topological_order`, `closure`, `ancestors`, `subgraph`).
- `CreditAssigner` / `CreditRank` / `CreditReport` — `new`, `with_decay`, `with_normalise`, `with_max_depth`, `assign(graph, rewards)`; the report exposes `credit(node_id)` and `top(n)`.
- `DiagnosticSignal` / `SignalSource` / `SIGNAL_TIERS` / `reward_tier` — the Python-shaped diagnosis record, its provenance vocabulary, and the `perfect`/`fair`/`weak`/`uncertain` bands.
- `KeywordAttributor` / `KeywordRule` / `KeywordMatch` — deterministic keyword-to-target matching that keeps the matched keyword, weight and byte `position`.
- `RewardStageAdapter` — turns a `RewardSummary` into one diagnostic signal per sink step and folds scored signals back into per-node rewards.

## Constraints
- Deterministic: node order is insertion order, edge order is insertion order, and the topological sort always emits the earliest-inserted ready node.
- Cycles are legal in the stored graph (they occur in real runs); only `topological_order` reports them.
- Graph operations return `Error::validation` for unknown nodes, duplicate ids, and out-of-range parameters instead of panicking.
- `DiagnosticSignal` ids are content-stable FNV-1a digests (`diag-{16 hex}`) over the source spelling and text.

## Design Decisions
- The wall-clock `ts_ns` field is deliberately not carried over; callers that need a timestamp put it in `DiagnosticSignal::metadata`, and nothing in the crate orders signals by time.
- The sha256 digest is replaced by a local FNV-1a hash so the crate needs no crypto dependency; ids are stable but not byte-identical to Python's.
- `SignalSource` widens Python's two keyword sources into `user_feedback`/`review_feedback` and `conversation` so a host can say what it observed, with `other` as the tolerant catch-all.
- Credit assignment walks outward from the rewarded sink nodes to their ancestors and gives each cause a decaying share; Python defers this to an LLM deep agent, so the deterministic arithmetic replaces it.
- The reward stage depends on `crate::reward` types rather than duplicating dimension bookkeeping.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./graph.rs` → `AttributionNode`, `AttributionEdge`, `AttributionGraph` and traversal
- `./credit.rs` → `CreditAssigner`, `CreditRank`, `CreditReport`
- `./model.rs` → `DiagnosticSignal`, `SignalSource`, `SIGNAL_TIERS`, `reward_tier`
- `./keyword.rs` → `KeywordAttributor`, `KeywordRule`, `KeywordMatch`
- `./reward_stage.rs` → `RewardStageAdapter`
