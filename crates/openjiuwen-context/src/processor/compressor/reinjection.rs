//! State-snapshot reinjection into a compacted window.
//!
//! Ports `processor/forked/compressor/reinjection/`: named state sections
//! (skills, task status, plan mode, team policy) contributed by a host-provided
//! snapshot builder and re-injected after a compaction, with a character
//! budget so the section can never outgrow the window it was trimmed for.
