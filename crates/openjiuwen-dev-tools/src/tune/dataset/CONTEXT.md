# openjiuwen-dev-tools / tune / dataset

## Intent

The sampling, splitting, and batch-scheduling layer that sits above `tune::loader`'s `CaseLoader`: seeded samplers that pick a subset of a case list, a tag that records which partition a case set came from, and an epoch/batch plan a trainer can iterate.
Every item here is a pure function of `(cases, configuration, seed)` — no clock, no environment, no global generator, no I/O — so a plan or a sample is reproducible byte for byte.
Python reference: `openjiuwen/dev_tools/tune/dataset/case_loader.py`, which only ships `CaseLoader`; the samplers and the scheduler are the Rust port's own shape for the batch/sampling layer Python leaves to callers.

## API Surface

- `CaseSampler` — the object-safe `Send + Sync` sampling seam: `fn sample(&self, cases: &[Case]) -> Result<Vec<Case>>`; the count lives on the implementation, never on the call.
  Contract: a count above `cases.len()` clamps to the list length, `count == 0` is valid and empty, the same sampler over the same cases returns the same cases in the same order, and the result is a subsequence of the input multiset (clones, never invented or edited cases).
  Implementations are expected to derive `Debug`; the samplers here never fail, but the seam stays fallible (validation errors, or I/O for a future store-backed sampler).
- `SequentialSampler` — `new(count)`, `count()`, `with_count(count)`; clones the first `count` cases in input order.
- `RandomSampler` — `new(seed)` (count defaults to every case), `seed()`, `count() -> Option<usize>`, `with_count(count)`; seeds the crate's `Rng64` Fisher-Yates over the whole index list and takes the first `count`.
- `StratifiedSampler` — `new(seed)` (count defaults to every case), `seed()`, `count() -> Option<usize>`, `with_count(count)`, `with_stratum_key(impl Fn(&Case) -> String + Send + Sync + 'static)` (a plain `fn` pointer works), manual `Debug` reporting `stratum_key: "label" | "custom"`, and `strata(cases) -> BTreeMap<String, Vec<usize>>` (stratum key → input indices) for inspection.
  Default stratum key: the case's `label` as comma-separated `"key"=value` entries in ascending key order, keys JSON-quoted, values rendered as compact JSON with object keys sorted recursively.
  Allocation: largest remainder over integer arithmetic (`floor(count * n_i / total)` seats, leftover seats to the largest `(count * n_i) mod total`), ties by earlier stratum index (hence the lexicographically smaller key), never more than a stratum's size, exact total `min(count, cases.len())`.
  Order: strata visited in sorted key order for both allocation and emission; each stratum shuffled with `Rng64` seeded `seed.wrapping_add((position as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))`, then the allocated prefix is emitted.
- `CaseSplit` — `Train`/`Val`/`Unspecified` (`as_str`, `Display`, `Ord` for ordered maps); serializes as `"train"`/`"val"`/`"unspecified"`.
- `SplitDataset` — `split` tag plus cases: `new`, `len`, `is_empty`, `cases`, `into_cases`, `get`, `iter`, `tag`, `split(ratio)`, `sample(&dyn CaseSampler)` and `batches(&BatchScheduler)` (thin delegations to the two seams), `IntoIterator` by value and by reference, permissive `serde(default)` round-trip.
- `BatchScheduler` — `batch_size`, `epochs`, `shuffle_seed: Option<u64>`, `drop_last`; `new(batch_size)` rejects `0` with `Error::validation` naming `batch_size`, `Default` is the `new(1)` configuration, `with_epochs`, `with_shuffle(seed)`, `with_drop_last`, `without_drop_last`, readers `batch_size()`/`epochs()`/`shuffle_seed()`/`drop_last()`, `batches_per_epoch(len)` (`len / batch_size` when dropping, else `len.div_ceil(batch_size)`, `0` for a zero batch size), and `plan(cases) -> Result<Vec<Epoch>>`.
  `plan` rejects a zero `batch_size` (reachable only through deserialization) and zero `epochs`; per epoch it numbers the indices, shuffles them with `Rng64` seeded `seed.wrapping_add((epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))` when a seed is set, and chunks them, dropping only the trailing partial chunk when `drop_last` is set.
- `Epoch` — `index` (0-based) plus `batches`; `len`/`is_empty`; `Default` is the empty epoch (needed by the permissive container-level `serde(default)`).
- `Batch` — `epoch`, `index` (0-based within the epoch), `step` (global 1-based), `cases` (cloned in batch order), `indices` (input positions, so `cases[i] == input[indices[i]]`); `len`/`is_empty`; `Default` is the empty batch.
- `DEFAULT_EPOCHS = 1` — the epoch count of `BatchScheduler::new` and `BatchScheduler::default`.

## Constraints

- No new dependencies: shuffling goes through the crate-private `tune::prng::Rng64` (xorshift64*), and no sampler or scheduler carries its own generator.
- Every sampler and scheduler is `Send + Sync`-friendly and object safe where relevant (`&dyn CaseSampler`), so a trainer can hold one behind a trait object.
- Errors are `openjiuwen_common::error::{Error, Result}`: `Error::validation` for a zero `batch_size` (in `new` and defensively in `plan`) and a zero `epochs` (in `plan`); nothing here returns `Error::Unsupported`, and every other count is clamped rather than rejected.
- Samplers never reorder beyond their own shuffle: `SequentialSampler` preserves the input order exactly, `RandomSampler` returns a prefix of a permutation, and `StratifiedSampler` emits grouped strata in sorted key order.
- Functions returning `Result` (`plan`, `sample`, `batches`) carry no `#[must_use]` attribute — `Result` is already `#[must_use]`, and clippy's `double_must_use` rejects the redundancy; the non-fallible readers and helpers are `#[must_use]`.
- Allocator arithmetic is done in `u128` so a large count cannot overflow the intermediate product, and `leftover` seats are redistributed when a stratum fills up.
- Rustdoc on every item (`missing_docs`), files ≲1000 lines with their test suites inline, no `unsafe`, no printing.

## Design Decisions

- Counts live on the sampler rather than on `sample`, so one sampler value is reusable across case lists and the trainer's configuration stays declarative; `RandomSampler`/`StratifiedSampler` model "all cases" as `count: Option<usize> = None` instead of a sentinel `usize::MAX`.
- Clamping at the seam (not rejection) mirrors Python's permissive dataset handling while the zero case stays a valid, empty sample.
- Stratification keys off the rendered label by default because `Case` has no class column; rendering is canonical (sorted keys recursively) so two equal labels built in different orders group together, and keys are JSON-quoted so a `=` or `,` in a key cannot be confused with a separator.
- Largest-remainder allocation is deterministic and exact: proportional shares round down, the remainder goes to the largest fractional parts, ties break on the sorted stratum order, and a full stratum hands its seat to the next one with capacity — so the sample length is always `min(count, cases.len())`, never short and never long.
- Per-stratum and per-epoch seeds are derived by adding `index * 0x9E37_79B9_7F4A_7C15` (wrapping) to the base seed, so one seed makes a whole plan reproducible while different strata/epochs still get different permutations.
- The batch plan is data (`Vec<Epoch>`), not an iterator: `plan` is re-runnable, comparable, JSON-serializable and inspectable before a training step, and equal plans compare equal.
- `Batch` carries both `cases` and `indices` so a training loop can report positions back to the caller without searching the batch contents, and the derived `Default` on `Batch`/`Epoch` exists only so the requested container-level `serde(default)` can fill omitted keys.

## Test Strategy

- `sampler.rs` unit tests cover sequential prefix/clamping/readers/`Debug`, random determinism, seed-zero remapping, multiset preservation, clamping, zero count and empty input, the default stratum-key rendering (nested object key sorting, multi-entry labels, arrays and scalars), label grouping and sorted key order, proportional allocation with a largest-remainder winner, the never-exceeds-a-stratum and exact-length invariants across counts `0..=5`, determinism and per-seed stability, custom `fn`-pointer stride keys, `strata` index reporting, the `allocate` helper on pinned vectors (including ties, clamping and empty input), and usability through `&dyn CaseSampler`/`Box<dyn CaseSampler>`.
- `batch.rs` unit tests cover batch counts with and without `drop_last` (10 cases at size 3 → 4 or 3), an exact divisor, fewer cases than one batch, empty case lists, multi-epoch layout and per-epoch batch indices, global 1..N `step` numbering, `indices`/`cases` agreement with the input, unshuffled order across every epoch, shuffled reproducibility plus per-epoch and per-seed difference and per-epoch permutation, shuffled batch layout, `new(0)` rejection, `plan` rejection of zero epochs and of a deserialized zero batch size, builder readers, scheduler JSON round-trip (all four keys, `null` seed, defaults, unknown keys), plan/epoch JSON round-trip, and `Default`.
- `split.rs` adds one delegation test: `sample` returns exactly what the sampler returns over `cases()` (sequential and random), `batches` equals `scheduler.plan(cases())`, and a scheduler that cannot plan propagates its validation error.
- Everything is offline and deterministic: fixed case ids, fixed seeds, no clock, no environment, no network.

## Routing Table

- `./mod.rs` → module docs and the re-export surface (`Batch`, `BatchScheduler`, `Epoch`, `DEFAULT_EPOCHS`, `CaseSampler`, `RandomSampler`, `SequentialSampler`, `StratifiedSampler`, `CaseSplit`, `SplitDataset`)
- `./split.rs` → `CaseSplit`, `SplitDataset`, the split/tag helpers and the sampler/scheduler delegations
- `./sampler.rs` → `CaseSampler`, `SequentialSampler`, `RandomSampler`, `StratifiedSampler`, default stratum-key rendering, largest-remainder allocation
- `./batch.rs` → `BatchScheduler`, `Epoch`, `Batch`, `DEFAULT_EPOCHS`, the epoch/batch planning rules
- `../prng.rs` → the shared crate-private `Rng64` generator (outside this node — read-only, escalate writes to the `tune` root)
- `../base.rs` → `Case` (sibling — read-only)
