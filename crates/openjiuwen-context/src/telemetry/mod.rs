//! Compression telemetry: `context.compression_state` records.
//!
//! Ports `schema/context_state.py`, `context/processor_state_recorder.py` and
//! `context/compression_scope.py`: the compression-state value objects, the
//! recorder that builds and emits them per processor pass, and the
//! request-purpose correlation scope for processor model calls.
