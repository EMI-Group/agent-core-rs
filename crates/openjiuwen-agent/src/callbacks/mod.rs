//! Agent callback framework: lifecycle events, typed payloads, ordered handlers.
//!
//! Ports `openjiuwen/core/single_agent/agent_callback_manager.py`: a per-agent
//! registry of callbacks that fire on invocation start, ReAct step start and
//! end, tool-call start and end, and invocation end.
