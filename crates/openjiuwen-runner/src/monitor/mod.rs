//! Status and event monitoring for the runs the runner is executing.
//!
//! Keeps a snapshot per run and publishes transition events on a broadcast
//! channel so a host can show live progress without polling the session.
