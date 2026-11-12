//! `PUCT` tree search over candidate programs.
//!
//! Python parity: `openjiuwen.rsi.artifact_rsi.program_opt` (v0.1.20). The Rust
//! port keeps the portable half — the configuration record and the flat tree with
//! its selection rule — and leaves everything that needs a model client, a
//! sandbox, git or subprocesses to the caller (`deferred::runtime`,
//! `deferred::agentdescent`).
//!
//! - [`config`] — [`ProgramOptConfig`], [`ReplyFormat`] and the worker bounds.
//! - [`tree`] — [`Program`], [`Node`], [`PuctTree`] and its exact `PUCT` formula.
//!
//! ```
//! use openjiuwen_rsi::artifact_rsi::program_opt::{Program, ProgramOptConfig, PuctTree};
//!
//! let mut tree = ProgramOptConfig::default().to_tree().expect("valid config");
//! let root = tree
//!     .add_node(None, Program { program_id: "seed".to_owned(), ..Program::default() })
//!     .expect("empty tree accepts a root");
//! tree.record_score(root, Some(0.1)).expect("root exists");
//! assert_eq!(tree.best().map(|node| node.program.program_id.as_str()), Some("seed"));
//! ```
pub mod config;
pub mod tree;
pub use config::{ProgramOptConfig, ReplyFormat, DEFAULT_WORKERS, MAX_WORKERS};
pub use tree::{Node, Program, PuctTree};
