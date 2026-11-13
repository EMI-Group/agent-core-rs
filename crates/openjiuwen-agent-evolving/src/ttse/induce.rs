//! Tip induction from trajectories and evolution signals.
//!
//! Ports the portable part of Python `openjiuwen.agent_evolving.ttse.induction`:
//! failure evidence and explicit signals become candidate tips. The offline
//! path is deterministic; an injected [`Model`](openjiuwen_core::foundation::llm::Model)
//! enables a prompt-driven path that falls back to the deterministic one.

/// Produces candidate tips from trajectories and signals.
pub struct TipInducer;
