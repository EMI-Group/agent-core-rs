//! Provider-specific context-window policy and KV-cache diffing.
//!
//! Ports `context/context_utils.py`'s window resolution (explicit override,
//! model map, built-in table, OpenRouter cache, default) and the
//! `ContextWindowChange` / `detect_context_window_change` primitives used to
//! reason about prompt-prefix invalidation.
