//! Write modes (`create`/`append`/`merge`/`replace`/`skip`) and the bounded
//! conflict-retry checker.
//!
//! Ports Python's `WriteMode` from `core/memory/lite/conflict_types.py` and extends it with
//! the two modes the local store needs for its own Markdown source of truth.
//!
//! # Modes
//!
//! - `create` — write only when no same-named fact exists; an existing fact is a conflict.
//! - `append` — add the incoming body to the existing fact's body, keeping both.
//! - `skip` — leave an existing fact untouched and report the skip explicitly as an outcome,
//!   never as a silent success.
//! - `merge` — extension: union the attributes (description, tags, path) and take the longer
//!   body, preserving the existing fact's name and creation time.
//! - `replace` — extension: overwrite the fact wholesale, keeping its name and creation time.
//!
//! The default is the conflict-detecting `create`, so an unintended overwrite is impossible
//! without the caller asking for it.
//!
//! # Conflict checker
//!
//! The checker wraps a write attempt: it reads the current snapshot of the target fact,
//! decides the outcome from the mode and the snapshot, and retries only when the snapshot
//! changed underneath the attempt (a concurrent writer won the race). Retries are bounded by a
//! configured attempt count, spaced by exponential backoff that is computed from an injected
//! clock so tests never sleep on wall time.
//!
//! # Exhaustion is an error
//!
//! When the attempt budget is exhausted the call returns a typed conflict error naming the
//! fact and the number of attempts. It never degrades into a silent skip, never returns a
//! success for a write that did not happen, and never leaves a partially written fact — each
//! attempt either commits under the local file lock or changes nothing.
