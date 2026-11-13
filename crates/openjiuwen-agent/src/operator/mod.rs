//! Tunable operators: parameter proxies for LLM calls, tool calls, memory calls,
//! and context evolution.
//!
//! Ports `openjiuwen/core/operator/`: every operator exposes typed tunables, a
//! parameter set/get surface, and a serializable state so an external optimiser
//! can read, update, and restore the live call path.
