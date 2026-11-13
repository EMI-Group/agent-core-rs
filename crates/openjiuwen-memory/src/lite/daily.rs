//! `daily_memory/YYYY-MM-DD.md` rollups with a stable id and deterministic content
//! ordering.
//!
//! Ports the deferred `daily_memory` rollups of Python `core/memory/lite/`: one Markdown
//! document per calendar day, aggregating that day's coding facts and session lines into a
//! compact, prompt-facing summary.
//!
//! # Layout
//!
//! ```text
//! <root>/daily_memory/YYYY-MM-DD.md
//! ```
//!
//! The file name is the day in UTC `YYYY-MM-DD` form, so the layout is self-describing and
//! sortable. The document carries front matter with its days and a stable id — the id is
//! derived from the day, not from the content, so it survives re-aggregation — followed by
//! one section per source (facts, sessions) and one entry per contributing item.
//!
//! # Deterministic ordering
//!
//! Entries are ordered by a fixed rule rather than by discovery order or filesystem
//! timestamps: facts by name ascending, session lines by session id then line number
//! ascending, and each bullet is rendered from the same fields every time. Aggregating the
//! same inputs twice therefore produces byte-identical documents, which makes the rollup
//! diffable and safe to commit.
//!
//! # Idempotence
//!
//! A rerun over an unchanged day rewrites nothing and reports the existing document; running
//! over a day whose inputs changed rewrites the document in place, preserving its id. Days are
//! independent, so a retried or partially completed sweep cannot corrupt an already written
//! day.
//!
//! # Constraints
//!
//! Rollups are derived state: they may be deleted and regenerated from the facts and session
//! logs at any time. Nothing here starts a timer or a background task — a sweep is an explicit
//! caller request — and a missing day of inputs produces no document rather than an empty one.
