//! Chat-agent adapters that bind a tuning optimizer to an agent-like runner.
//!
//! Ports `openjiuwen/dev_tools/tune/chat_agent/` (`chat_config.py`,
//! `chat_agent.py`) without Python's runtime: a [`ChatConfig`](self) carries the
//! model and prompt settings a chat agent is built from, and the agent adapter
//! implements the tuning seams so the same optimizer runs against a real model
//! instead of the deterministic evaluator.
