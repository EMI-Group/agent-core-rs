//! The runner's in-process message queue: pub/sub plus request/reply over
//! per-topic FIFO channels.
//!
//! Ports `openjiuwen/core/runner/message_queue_base.py` (`QueueMessage`,
//! `MessageQueueBase`, `SubscriptionBase`, `AsyncMessageHandler`) and the
//! behavioural core of `message_queue_inmemory.py` /
//! `dunner/dmessage_queue/message_queue_fake.py`, without the distributed wire
//! (`DmqRequestMessage` / `DmqResponseMessage`, reply topics).
//!
//! [`MessageQueue`] is the seam a broker backend implements:
//! [`create_message_queue`] builds [`InMemoryMessageQueue`] for
//! [`MessageQueueKind::InMemory`] and answers `Error::Unsupported` for the
//! deferred `pulsar` / `zeromq` backends that `openjiuwen-extensions` will own.
//! [`build_message_queue`] is the infallible counterpart: a deferred transport
//! yields [`DeferredMessageQueue`], a typed placeholder whose every operation is
//! that same `Error::Unsupported`, so a runner can never serve an in-memory queue
//! under another backend's name.
//!
//! # Example
//!
//! ```
//! use openjiuwen_runner::mq::{InMemoryMessageQueue, MessageHandler, MessageQueue, QueueMessage};
//! use openjiuwen_common::error::Result;
//! use serde_json::{json, Value};
//! use std::sync::Arc;
//!
//! struct Doubler;
//!
//! #[async_trait::async_trait]
//! impl MessageHandler for Doubler {
//!     async fn handle(&self, message: QueueMessage) -> Result<Value> {
//!         Ok(json!(message.payload.as_i64().unwrap_or_default() * 2))
//!     }
//! }
//!
//! # async fn example() -> Result<()> {
//! let queue = InMemoryMessageQueue::with_capacity(16);
//! queue.start().await?;
//! queue.add_handler("math", Arc::new(Doubler)).await?;
//! let reply = queue.request("math", QueueMessage::new(json!(21)), None).await?;
//! assert_eq!(reply, json!(42));
//! queue.stop().await?;
//! # Ok(())
//! # }
//! ```

mod counters;
pub mod memory;
pub mod message;
pub mod queue;

pub use crate::config::{MessageQueueConfig, MessageQueueKind};
pub use memory::InMemoryMessageQueue;
pub use message::{MessageHandler, QueueMessage, SUCCESS};
pub use queue::{
    build_message_queue, create_message_queue, DeferredMessageQueue, MessageQueue, QueueStats,
};
