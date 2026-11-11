//! Memory kinds, scopes, records, inputs, filters, options, and result types.
//!
//! Everything in this module is plain `serde` data or a pure helper: no storage, no
//! embedding, and no async except [`CallOptions::run`], which races a future against
//! cancellation and an optional deadline. JSON names stay compatible with the Python
//! `MemoryDoc`/`MemResult` wire shapes.
pub mod filter;
pub mod input;
pub mod kind;
pub mod options;
#[allow(clippy::module_inception)] // the data module is named after the Python model
pub mod record;
pub mod results;
pub mod scope;
pub use filter::*;
pub use input::*;
pub use kind::*;
pub use options::*;
pub use record::*;
pub use results::*;
pub use scope::*;
