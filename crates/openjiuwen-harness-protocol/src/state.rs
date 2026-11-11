//! Harness lifecycle state.

use serde::{Deserialize, Serialize};

/// Lifecycle state of a harness handle.
///
/// Wire values are `idle`, `running`, `pausing`, `paused`, and `terminated`.
///
/// This is shared vocabulary with `openjiuwen-agent-teams`, which re-exports this type — never
/// define a second copy.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HarnessState {
    /// The harness exists but no turn is in flight.
    Idle,
    /// A turn is executing.
    Running,
    /// A pause was requested and is being honoured.
    Pausing,
    /// The run is suspended and may be resumed.
    Paused,
    /// The run finished and cannot be resumed.
    Terminated,
}
