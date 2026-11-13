//! In-crate `Redis` client (`RESP2`/`RESP3`) shared by the Redis adapters.
//!
//! `openjiuwen-extensions` cannot depend on a Redis driver: the workspace builds
//! offline and no such crate is vendored. The client in this module speaks the
//! Redis serialization protocol directly over [`tokio::net::TcpStream`], covering
//! the single-node command surface the Redis checkpointer and key-value store use
//! (`GET`/`SET`/`EXISTS`/`DEL`/`UNLINK`/`MGET`/`SCAN`/`EXPIRE`, sets, sorted sets,
//! and pipelined batches).
//!
//! Supported transport: `redis://` and `redis+cluster://` (single-node connect, with
//! [`ConnectionOptions::cluster_mode`] recorded). `rediss://` needs a TLS stack the
//! workspace does not vendor, and cluster slot routing is not implemented; both
//! return [`openjiuwen_common::error::Error::Unsupported`] with a reason.
//!
//! ```no_run
//! use openjiuwen_extensions::redis::{Client, Reply};
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let client = Client::from_url("redis://127.0.0.1:6379/0").await?;
//! client.command_str(&["SET", "key", "value"]).await?;
//! assert_eq!(
//!     client.command_str(&["GET", "key"]).await?,
//!     Reply::Bulk(b"value".to_vec())
//! );
//! # Ok(())
//! # }
//! ```
pub mod client;
pub mod codec;
#[cfg(test)]
pub(crate) mod testsupport;
pub use client::{Client, ConnectionOptions};
pub use codec::Reply;
/// Builds an owned argument vector for [`Client::command`] from byte slices.
///
/// This is a convenience for callers that build commands dynamically; static
/// commands usually go through [`Client::command_str`] instead.
#[must_use]
pub fn command(args: &[&[u8]]) -> Vec<Vec<u8>> {
    args.iter().map(|arg| arg.to_vec()).collect()
}
