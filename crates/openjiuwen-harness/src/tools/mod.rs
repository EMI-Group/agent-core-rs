//! Model-facing coding tools, the tool registry, and shared tool helpers.
//!
//! Ports `openjiuwen/harness/tools/`: the shell, filesystem, todo, ask-user and memory
//! tools exposed to the model, plus their shared payload/path helpers and the registry.
mod ask_user;
pub mod descriptions;
mod memory;
mod read;
pub mod read_state;
mod search;
mod shell;
pub mod support;
#[cfg(test)]
mod tests;
mod todo;
mod write;
