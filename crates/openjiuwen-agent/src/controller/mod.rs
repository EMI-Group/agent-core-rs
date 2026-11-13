//! Controller stack: routing events to agents and workflows.
//!
//! Ports `openjiuwen/core/controller/`: an event queue feeds intent recognition,
//! the recognizer produces an intent, the task scheduler picks a registered
//! task executor (a controller agent, an LLM router, or a workflow), and the
//! controller returns the executor's output.
