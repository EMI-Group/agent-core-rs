# ttse

## Intent
Test-time skill experience: turning failures into reusable tips, banking and ranking them, consulting them into a prompt, and periodically merging/pruning the bank.
Ports the offline part of Python `openjiuwen.agent_evolving.ttse`: `config.py`, the tip/inference schema, `stores.py`/`index.py`, `classify.py`, `success.py`, `tip_parse.py`, `consult.py`/`render.py`, `prompts.py` and the deterministic plus model-backed parts of `induction.py` and `dream.py`.
Retrieval is available in two flavours: lexical (token overlap over the bank) and embedding-backed (cosine over an `EmbeddingIndex`), fused by Reciprocal Rank Fusion.

## API Surface
- `TtseConfig` — Python-shaped serde tunables (`store_path`, `capacity`, `dedup_threshold`, `inject_enabled`, `evolve_enabled`, `success_threshold`, `detect_min_tool_calls`, the `consult_*` limits and the `dream_*` knobs) with `validate`.
- `TipRecord` — the tip schema: `tip_id`, `rule`, `category`, `keywords`, `weight`, `hits`/`successes`/`failures`, `source`, `created_tick`/`updated_tick`; `TipCategory` is `tool`, `workflow`, `pitfall`, `environment`, `communication`, `other` with a fixed render order; `TipEvidence` records the trajectory a tip was induced from.
- `tip_id_for(category, rule)` — content-addressed tip id, `{16 lowercase hex}` of an FNV-1a hash over `"{category}:{normalized rule}"`.
- `TipBank` / `TipBankStats` / `AddOutcome` — `add` (dedupe by similarity), `observe`, `top`, `by_category`, `prune`, `decay`, `increment_tick`, `stats`; the tick counter is the bank's only notion of time.
- `TipClassifier` / `ClassificationRule` — ordered keyword voting over a category table.
- `SuccessDetector` / `SuccessAssessment` — decides whether a run counts as a successful experience from its recorded outcome, tool activity and error steps.
- `TipParser` / `DEFAULT_MAX_RULE_CHARS` — total, never-failing parsing of JSON arrays, `{"tips": [...]}`, markdown bullets and numbered lines.
- `TipInducer` — mines failed trajectories and evolution signals for candidate tips, offline or through an injected `Model`.
- `EmbeddingIndex` / `IndexMatch` — vectors keyed by tip id over an `Arc<dyn Embedding>`: `new`, `with_path`, `open`, `sync` (cold fill and stale re-embed, batched), `upsert`, `remove`, `clear`, `query`, `save`, `save_default`, `len`/`is_empty`/`contains`/`vector`/`path`/`dimension`/`embedding_model`.
- `cosine(a, b)` — prefix-limited cosine similarity, `0.0` for an empty slice or a zero norm.
- `RetrieveMode` / `RankedTip` / `rank` / `RRF_K` — the `hybrid | embed | lexical | dump` decision table over the bank and an optional index; a ranked hit carries the fused `score` plus the one-based `lexical_rank`/`embed_rank` that produced it.
- `consult_async` — ranks, maps back to `TipRecord`s and renders with `render_tips`, honouring `consult_limit()` and `consult_max_chars`.
- `Dreamer` / `DreamReport` — connected-component clustering and merging of near-duplicate tips plus stale-tip pruning; `with_model`, `has_model`, `with_max_llm_merges`, `max_llm_merges` and `dream_with_model` add the LLM merge/purge passes.
- `TtseEngine` — `new`, `with_model`, `with_embedding`, `with_embedding_index`, `embedding_index`, `observe`, `consult`, `consult_async`, `consult_with`, `dream`, `dream_async`, `sync_index`, `stats`, `save`, `save_with_index`, `load`, `load_with_config`.
- `consult` / `render_tips` / `ConsultResult` — the synchronous lexical selection and bounded, category-grouped rendering for prompt injection.

## Constraints
- Clock-free: the bank tick counter is the only time source and callers drive it, so identical inputs replay identically.
- Tip identity is content-addressed, never random, so bank files and checkpoints are reproducible.
- Lexical similarity is token-set Jaccard (ASCII runs plus one token per CJK character), which is deterministic and hash-order independent.
- Embedding similarity is cosine over caller-supplied vectors; the index never embeds on its own, so a deterministic embedder such as `retrieval::embedding::HashEmbedding` keeps the whole pipeline offline.
- The index stores `model_name` and `dimension` beside the vectors and rejects a mismatch on `open` with `Error::validation`; vectors are never truncated or padded.
- `TipParser::parse` never panics and never returns an error; a malformed payload falls back to line parsing.
- A malformed dream verdict leaves the cluster distinct instead of erroring, and merges never cross a `TipCategory`.
- Persistence is atomic (`<path>.tmp` + rename) and stores the tips and tick only; engine-side counters such as the consultation count restart at `0` after a load.
- Successful trajectories never produce a tip; only failures and error-carrying `unknown` runs are mined.

## Design Decisions
- The embedding seam is `openjiuwen_core::retrieval::embedding::Embedding`, so no new dependency is needed and `HashEmbedding` gives deterministic offline vectors; the index is a plain `BTreeMap<String, Vec<f32>>` with an atomic JSON document `{"version":1,"model_name":...,"dimension":...,"vectors":{...}}`.
- `EmbeddingIndex::sync` embeds only tips whose vector is missing or whose rule text changed (the text fingerprint lives beside the vector) and drops vectors of removed tips, batched at `EMBED_BATCH_SIZE = 32`.
- `rank` reproduces Python's `rank_track` decision table: `dump` and a pool no larger than the limit short-circuit to bank order, `embed` falls back to lexical (and `lexical` to embedding) when its preferred branch is empty, and `hybrid` fuses only when both branches matched.
- Fusion is Reciprocal Rank Fusion with `RRF_K = 60`, equal weights and `score += 1 / (rrf_k + rank)` over one-based ranks; ties break by ascending `tip_id`, so the ranking is stable.
- Retrieval never silently returns nothing: any empty ranking falls back to the dump order, and `consult` ranks with the synchronous lexical path so callers without an index keep working.
- Dedupe, pruner similarity and the no-index dream clustering stay token-set Jaccard over `bank::tokenize`/`bank::jaccard`; with an index, dreaming soft-clusters by cosine instead.
- `Dreamer::with_model` is additive: the model is optional, `Dreamer::dream` stays the deterministic path, and `dream_with_model` delegates to it verbatim when no model is installed.
- Model-backed dreaming runs prune → merge → purge: unused tips past `dream_ttl_ticks` are pruned first, clusters are merged under `dream_merge_threshold` with `dream_cluster_min_size`, and surviving tips are form-checked for malformed or generic rules.
- Merge prompts number the cluster rules and demand `THINKING` / `REASON` / `VERDICT: MERGE|KEEP_DISTINCT|REWRITE` / `CANONICAL` / `MERGE_INDICES` / `KEEP_INDICES`; the parser is tolerant, an unparseable answer keeps every member, and at most `DEFAULT_MAX_LLM_MERGES = 15` prompts run per pass.
- A merge rewrites the representative through `TipBank::merge_cluster` and then stores the canonical rule (recomputing `tip_id_for`) with `source = "dream"`; a canonical the record model rejects leaves the cluster distinct.
- The purge pass asks for `INDEX: i | VERDICT: KEEP|PURGE | REASON: ...` in batches; when the answer is unusable a deterministic fallback purges rules that are empty, too short or rejected by `TipParser`.
- Form checks are memoised in the `Dreamer` (`BTreeSet` of tip ids) rather than persisted on `TipRecord`, so the record schema stays Python-compatible.
- `TtseEngine::dream_async` syncs the index, dreams with the model and the index, then syncs again so merged ids are re-embedded before the next consultation.
- `TtseEngine::save_with_index` writes the bank to the requested path and the index to the path it was opened from, or to `<path>.embedding.json` when it has none.
- Connected-component clustering (rather than iterative pairwise merging) makes dreaming order-independent and idempotent.
- Rendering groups by `TipCategory::order` and ranks by effective score inside a group, so identical inputs always produce identical prompt text.
- Selection does not mutate observation counters; hit bookkeeping belongs to `TtseEngine`, which is the only component holding `&mut TipBank`.

## Routing Table
- `./mod.rs` → re-export surface and module documentation
- `./config.rs` → `TtseConfig` and its bounds
- `./models.rs` → `TipRecord`, `TipCategory`, `tip_id_for`, result records
- `./bank.rs` → `TipBank`, dedupe/ranking/pruning, `tokenize`/`jaccard`
- `./index.rs` → `EmbeddingIndex`, `cosine`, vector persistence
- `./retrieve.rs` → `RetrieveMode`, `rank`, RRF fusion, `consult_async`
- `./classify.rs` → `TipClassifier`, `ClassificationRule`
- `./success.rs` → `SuccessDetector`, `SuccessAssessment`
- `./parse.rs` → `TipParser`
- `./induce.rs` → `TipInducer`
- `./dream.rs` → `Dreamer`, `DreamReport` and the deterministic clustering
- `./dream_llm.rs` → model-backed merge/purge prompts, verdict parsing and fallbacks
- `./consult.rs` → `consult`, `render_tips`
- `./engine.rs` → `TtseEngine`, its embedding hooks and persistence
