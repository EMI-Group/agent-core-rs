//! `DeepAgent` facade, builder, factory, session state persistence, and round execution.

mod prompt_rail;
mod round;
mod state_store;

pub use prompt_rail::{PromptRail, PromptState};
pub use round::{normalize_inputs, RoundExecutorHandle, RoundState};
pub use state_store::{clear_state, load_state, save_state, SESSION_STATE_KEY};
