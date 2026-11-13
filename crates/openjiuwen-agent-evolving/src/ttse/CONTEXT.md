# ttse

## Intent
Test-time skill experience: turning failures into reusable tips, banking and ranking them, consulting them into a prompt, and periodically merging/pruning the bank.
Ports the offline part of Python `openjiuwen.agent_evolving.ttse`: `config.py`, the tip/inference schema, `stores.py`/`index.py`, `classify.py`, `success.py`, `tip_parse.py`, `consult.py`/`render.py` and the deterministic parts of `induction.py` and `dream.py`.

## API Surface
- `TtseConfig` — Python-shaped serde tunables (`store_path`, `capacity`, `dedup_threshold`, `inject_enabled`, `evolve_enabled`, `success_threshold`, `detect_min_tool_calls`, the `consult_*` limits and the `dream_*` knobs) with `validate`.
- `TipRecord` — the tip schema: `tip_id`, `rule`, `category`, `keywords`, `weight`, `hits`/`successes`/`failures`, `source`, `created_tick`/`updated_tick`; `TipCategory` is `tool`, `workflow`, `pitfall`, `environment`, `communication`, `other` with a fixed render order; `TipEvidence` records the trajectory a tip was induced from.
- `tip_id_for(category, rule)` — content-addressed tip id, `{16 lowercase hex}` of an FNV-1a hash over `"{category}:{normalized rule}"`.
- `TipBank` / `TipBankStats` / `AddOutcome` — `add` (dedupe by similarity), `observe`, `top`, `by_category`, `prune`, `decay`, `increment_tick`, `stats`; the tick counter is the bank's only notion of time.
- `TipClassifier` / `ClassificationRule` — ordered keyword voting over a category table.
- `SuccessDetector` / `SuccessAssessment` — decides whether a run counts as a successful experience from its recorded outcome, tool activity and error steps.
- `TipParser` / `DEFAULT_MAX_RULE_CHARS` — total, never-failing parsing of JSON arrays, `{"tips": [...]}`, markdown bullets and numbered lines.
- `TipInducer` — mines failed trajectories and evolution signals for candidate tips, offline or through an injected `Model`.
- `Dreamer` / `DreamReport` — connected-component clustering and merging of near-duplicate tips plus stale-tip pruning.
- `TtseEngine` — `new`, `with_model`, `observe`, `consult`, `dream`, `stats`, `save`, `load`, `load_with_config`; the end-to-end pipeline with atomic JSON persistence.
- `consult` / `render_tips` / `ConsultResult` — selection and bounded, category-grouped rendering for prompt injection.

## Constraints
- Clock-free: the bank tick counter is the only time source and callers drive it, so identical inputs replay identically.
- Tip identity is content-addressed, never random, so bank files and checkpoints are reproducible.
- Similarity is token-set Jaccard (ASCII runs plus one token per CJK character), which is deterministic and hash-order independent.
- `TipParser::parse` never panics and never returns an error; a malformed payload falls back to line parsing.
- Persistence is atomic (`<path>.tmp` + rename) and stores the tips and tick only; engine-side counters such as the consultation count restart at `0` after a load.
- Successful trajectories never produce a tip; only failures and error-carrying `unknown` runs are mined.

## Design Decisions
- Retrieval, dedupe, pruner similarity and dream clustering are token-set Jaccard over `bank::tokenize`/`bank::jaccard`; there is no embedding-provider trait, so the whole pipeline runs offline.
- The only model seam is `TipInducer::with_model` (wrapped by `TtseEngine::with_model`, `Arc<dyn Model>`); `Dreamer` has no model hook, so LLM-driven merging stays unported (`deferred.rs` area `TTSE embedding retrieval and LLM dreaming`).
- Connected-component clustering (rather than iterative pairwise merging) makes dreaming order-independent and idempotent.
- Rendering groups by `TipCategory::order` and ranks by effective score inside a group, so identical inputs always produce identical prompt text.
- Selection does not mutate observation counters; hit bookkeeping belongs to `TtseEngine`, which is the only component holding `&mut TipBank`.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./config.rs` → `TtseConfig` and its bounds
- `./models.rs` → `TipRecord`, `TipCategory`, `tip_id_for`, result records
- `./bank.rs` → `TipBank`, dedupe/ranking/pruning, `tokenize`/`jaccard`
- `./classify.rs` → `TipClassifier`, `ClassificationRule`
- `./success.rs` → `SuccessDetector`, `SuccessAssessment`
- `./parse.rs` → `TipParser`
- `./induce.rs` → `TipInducer`
- `./dream.rs` → `Dreamer`
- `./consult.rs` → `consult`, `render_tips`
- `./engine.rs` → `TtseEngine` and its persistence
