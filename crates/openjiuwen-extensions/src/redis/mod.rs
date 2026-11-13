//! In-crate Redis client (RESP2/RESP3) shared by the Redis adapters.
//!
//! `openjiuwen-extensions` cannot depend on a Redis driver: the workspace builds
//! offline and no such crate is vendored. The client in this module speaks the
//! Redis serialization protocol directly over [`tokio::net::TcpStream`], which
//! covers the single-node command surface the checkpointer and the key-value
//! store need (`GET`/`SET`/`EXISTS`/`DEL`/`UNLINK`/`MGET`/`SCAN`/`EXPIRE` and
//! pipelined batches).
//!
//! `redis://` is the supported transport; `rediss://` needs a TLS stack that the
//! workspace does not vendor, and cluster slot routing is not implemented, so
//! both return [`openjiuwen_common::error::Error::Unsupported`] with a reason.
