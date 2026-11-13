//! The TTSE engine: observe → induce → consult → dream, with persistence.
//!
//! Wires the tip bank, the inducer, the dreamer and the consultation renderer
//! into one offline pipeline. Ticks are monotonic counters owned by the bank, so
//! no clock is read and repeated runs over identical inputs are identical.

/// Offline TTSE pipeline over trajectories and evolution signals.
pub struct TtseEngine;
