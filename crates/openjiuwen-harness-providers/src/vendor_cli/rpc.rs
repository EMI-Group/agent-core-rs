//! JSON-RPC 2.0 client over newline-delimited JSON stdio.
//!
//! Some vendor CLIs (the Codex app-server, the `DeepSeek` Harness runtime) speak JSON-RPC 2.0 with one
//! JSON message per line instead of the flat NDJSON envelopes the Claude Code adapter translates.
//! [`JsonRpcClient`] is that transport: it writes requests, notifications and answers, and it reads
//! responses, notifications and server-to-client requests off the same pipe.
//!
//! # Wire rules
//!
//! - a **request** is `{"id": <u64>, "method": <string>, "params": <value>}`; ids are minted from
//!   `1` and [`JsonRpcClient::request`] correlates the answer by id, so two outstanding requests may
//!   be answered in any order;
//! - a **notification** is `{"method": <string>, "params": <value>}` — the id member is omitted;
//! - a **response** is `{"id": <value>, "result": <value>}` or
//!   `{"id": <value>, "error": {"code": <number>, "message": <string>}}`; a peer error becomes
//!   `{label} JSON-RPC error for '{method}': {message} (code {code})`;
//! - a **server-to-client request** carries both `id` and `method`; it is delivered through
//!   [`JsonRpcClient::next_incoming`] and must be answered with
//!   [`respond`](JsonRpcClient::respond) or [`respond_error`](JsonRpcClient::respond_error) using
//!   the same `id` value, which is echoed verbatim (peer ids are arbitrary JSON, not just numbers).
//!
//! # Lifecycle
//!
//! [`JsonRpcClient::spawn`] starts exactly one reader task. That task is the only writer too: writes
//! are queued in the client and flushed by the reader between reads, so a caller never blocks on a
//! peer that is still answering an earlier request. A read is never abandoned mid-line (a cancelled
//! buffered read would lose bytes), so the queue is flushed whenever the reader is not blocked on
//! the stream; an answer to a server request is therefore written as soon as the application
//! supplies it. A transport that offers a [`writer`](RpcTransport::writer) bypasses the queue: its
//! writes go straight to the peer, which is what a command like `turn/interrupt` needs while the
//! turn loop is parked on a silent stream.
//!
//! When stdout reaches end of file the client fails every pending request with
//! `the {label} CLI exited before answering '{method}'` (plus the stderr tail when the peer wrote
//! one), marks itself closed, and closes the incoming channel so [`next_incoming`] returns [`None`].
//! Writing after end of file or [`close`] fails with
//! `the {label} CLI closed its JSON-RPC stream`. A malformed line is logged and skipped; only a run
//! of `MAX_CONSECUTIVE_READ_ERRORS` unreadable lines (or a transport error that keeps repeating)
//! declares the stream unusable, so a single garbled line never kills a live session.
//!
//! [`next_incoming`]: JsonRpcClient::next_incoming
//! [`close`]: JsonRpcClient::close

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{HarnessError, HarnessResult, JsonObject, JsonValue};
use serde_json::json;
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex, Notify};
use tokio::task::JoinHandle;

use super::child::{ChildProcess, LineWriter};
use super::jsonl::encode_line;
use super::lock;

/// How long [`JsonRpcClient::close`] waits for the aborted reader task to be dropped.
const READER_STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Number of consecutive unreadable lines tolerated before the stream is declared unusable.
///
/// A malformed line is skipped (a vendor CLI occasionally logs a diagnostic onto stdout), but a
/// stream that never yields a decodable message again — a broken pipe, a binary protocol — must not
/// keep the client spinning.
const MAX_CONSECUTIVE_READ_ERRORS: usize = 32;

/// Transport a JSON-RPC client writes to and reads from.
///
/// Implemented by [`ChildProcess`] for a real CLI; an adapter may supply its own transport (an
/// in-memory pipe, a socket wrapper) as long as the message-per-line contract holds.
#[async_trait]
pub trait RpcTransport: Send {
    /// Writes `line` as one newline-terminated message.
    ///
    /// # Errors
    ///
    /// Returns the transport's own error when the peer is gone; the client treats it as fatal.
    async fn write_line(&mut self, line: &str) -> HarnessResult<()>;

    /// Reads the next message, or [`None`] once the peer closed the stream.
    ///
    /// # Errors
    ///
    /// Returns the transport's own error when the line could not be read or decoded; the client
    /// logs it and skips the line.
    async fn read_json(&mut self) -> HarnessResult<Option<JsonValue>>;

    /// Returns the diagnostic tail captured from the peer's stderr.
    #[must_use]
    fn stderr_tail(&self) -> String;
    /// Returns a writer that can send a line while a read is still pending on this transport.
    ///
    /// The default is [`None`]: such a transport queues every write for the reader task to flush
    /// between reads, which is enough for a strictly request/response protocol. A transport whose
    /// peer can stay silent while the application owes it a command (the codex app-server during an
    /// idle turn) overrides this with a writer independent of the read path.
    fn writer(&self) -> Option<std::sync::Arc<dyn LineWriter>> {
        None
    }
}

#[async_trait]
impl RpcTransport for ChildProcess {
    async fn write_line(&mut self, line: &str) -> HarnessResult<()> {
        ChildProcess::write_line(self, line).await
    }

    fn writer(&self) -> Option<std::sync::Arc<dyn LineWriter>> {
        Some(std::sync::Arc::new(ChildProcess::writer(self)))
    }

    async fn read_json(&mut self) -> HarnessResult<Option<JsonValue>> {
        ChildProcess::read_json(self).await
    }

    fn stderr_tail(&self) -> String {
        ChildProcess::stderr_tail(self)
    }
}

/// Returns the JSON value a request minted by [`JsonRpcClient`] carries in its `id` member.
///
/// Client ids are JSON numbers counting from `1`. Ids minted by the peer may be any JSON value and
/// are echoed verbatim, so this helper only describes the client's own side of the correlation.
#[must_use]
pub fn request_id_value(id: u64) -> JsonValue {
    JsonValue::from(id)
}

/// One message queued for the reader task to write.
struct Outbound {
    /// The encoded message, without the trailing newline.
    line: String,
    /// The pending request this write belongs to, so a write failure can fail exactly that request.
    id: Option<u64>,
}

/// One request waiting for its response.
struct PendingRequest {
    /// Method name, reported in the failure message.
    method: String,
    /// Waiter resolved with the response (or the failure).
    sender: oneshot::Sender<HarnessResult<JsonValue>>,
}

/// Why the client stopped serving pending requests.
enum Ending {
    /// The peer's stdout reached end of file.
    Exited,
    /// The JSON-RPC stream failed; carries the transport's message.
    Failed(String),
    /// The client was closed by its owner.
    Closed,
}

impl Ending {
    /// Renders the failure message of one pending request.
    fn message(&self, label: &str, method: &str) -> String {
        match self {
            Self::Exited => format!("the {label} CLI exited before answering '{method}'"),
            Self::Failed(reason) => format!("the {label} CLI JSON-RPC stream failed: {reason}"),
            Self::Closed => format!("the {label} CLI closed its JSON-RPC stream"),
        }
    }
}

/// State one reader task owns.
///
/// The reader never holds a strong handle to the client: it upgrades [`client`](Self::client) per
/// message, so dropping the last external handle lets `Drop` abort the task instead of keeping the
/// client (and the child process) alive through the task.
struct ReaderState {
    /// Weak handle back to the client.
    client: Weak<JsonRpcClient>,
    /// The transport, shared with the client so `stderr_tail` can read a live tail.
    transport: Arc<AsyncMutex<Option<Box<dyn RpcTransport>>>>,
    /// Messages waiting to be written, oldest first.
    outbound: Arc<Mutex<VecDeque<Outbound>>>,
    /// Wakes the reader when there is something to write or to read.
    demand: Arc<Notify>,
}

/// JSON-RPC 2.0 client over one newline-delimited stdio stream.
///
/// See the [module documentation](self) for the wire rules and the lifecycle.
pub struct JsonRpcClient {
    /// The transport; [`None`] once it was dropped on close.
    transport: Arc<AsyncMutex<Option<Box<dyn RpcTransport>>>>,
    /// Requests waiting for their response, keyed by client id.
    pending: Mutex<BTreeMap<u64, PendingRequest>>,
    /// Messages queued for the reader task, oldest first; unused once [`writer`] is present.
    ///
    /// [`writer`]: RpcTransport::writer
    outbound: Arc<Mutex<VecDeque<Outbound>>>,
    /// Immediate write path, when the transport offers one.
    writer: Option<std::sync::Arc<dyn LineWriter>>,
    /// Sender of the inbound notification/server-request queue; [`None`] once closed.
    incoming_sender: Mutex<Option<mpsc::UnboundedSender<JsonObject>>>,
    /// Receiver handed to [`next_incoming`](Self::next_incoming) one call at a time.
    incoming: AsyncMutex<Option<mpsc::UnboundedReceiver<JsonObject>>>,
    /// Number of callers parked in [`next_incoming`](Self::next_incoming).
    incoming_waiters: AtomicUsize,
    /// Wakes the reader task.
    demand: Arc<Notify>,
    /// Last id handed out; ids start at `1`.
    next_id: AtomicU64,
    /// Whether the stream ended or the client was closed.
    closed: AtomicBool,
    /// Unreadable lines seen since the last decodable message.
    read_errors: AtomicUsize,
    /// Last stderr tail observed by the reader, used when the transport is busy.
    last_stderr: Mutex<String>,
    /// The reader task, aborted by [`close`](Self::close) and by `Drop`.
    reader: Mutex<Option<JoinHandle<()>>>,
    /// Label used in error messages, for example `Codex`.
    label: &'static str,
}

impl JsonRpcClient {
    /// Starts the single reader task and returns the shared client.
    ///
    /// The task is owned by the client: [`close`](Self::close) aborts it after bounding, and
    /// dropping every client handle aborts it too, so no detached task outlives the client.
    ///
    /// # Panics
    ///
    /// Requires a running Tokio runtime, like every spawned task in this crate.
    #[must_use]
    pub fn spawn(transport: Box<dyn RpcTransport>, label: &'static str) -> Arc<Self> {
        let (incoming_sender, incoming) = mpsc::unbounded_channel();
        let writer = transport.writer();
        let transport = Arc::new(AsyncMutex::new(Some(transport)));
        let outbound = Arc::new(Mutex::new(VecDeque::new()));
        let demand = Arc::new(Notify::new());
        let client = Arc::new(Self {
            transport: Arc::clone(&transport),
            pending: Mutex::new(BTreeMap::new()),
            outbound: Arc::clone(&outbound),
            writer,
            incoming_sender: Mutex::new(Some(incoming_sender)),
            incoming: AsyncMutex::new(Some(incoming)),
            incoming_waiters: AtomicUsize::new(0),
            demand: Arc::clone(&demand),
            next_id: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            read_errors: AtomicUsize::new(0),
            last_stderr: Mutex::new(String::new()),
            reader: Mutex::new(None),
            label,
        });
        let state = ReaderState {
            client: Arc::downgrade(&client),
            transport,
            outbound,
            demand,
        };
        let handle = tokio::spawn(read_loop(state));
        *lock(&client.reader) = Some(handle);
        client
    }

    /// Sends `method` with `params` and resolves with its result.
    ///
    /// The response is correlated by id, so an answer to a later request may arrive first.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] with the peer's error text when the peer replied with an
    /// `error` object, with `the {label} CLI exited before answering '{method}'` when the stream
    /// ended, and with `the {label} CLI closed its JSON-RPC stream` when the client was closed.
    pub async fn request(&self, method: &str, params: JsonValue) -> HarnessResult<JsonValue> {
        let (id, receiver) = self.start_request(method, &params).await?;
        self.await_response(id, receiver).await
    }

    /// Sends a request and gives up after `timeout`.
    ///
    /// # Errors
    ///
    /// Propagates [`request`](Self::request), and returns [`HarnessError::Protocol`] with
    /// `{label} did not answer '{method}' within {secs}s` when the timeout fires first. The request
    /// is then forgotten: a late answer is logged and dropped.
    pub async fn request_with_timeout(
        &self,
        method: &str,
        params: JsonValue,
        timeout: Duration,
    ) -> HarnessResult<JsonValue> {
        let (id, receiver) = self.start_request(method, &params).await?;
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                lock(&self.pending).remove(&id);
                Err(self.closed_error())
            }
            Err(_) => {
                lock(&self.pending).remove(&id);
                Err(HarnessError::protocol(format!(
                    "{} did not answer '{method}' within {}s",
                    self.label,
                    timeout.as_secs_f64()
                )))
            }
        }
    }

    /// Sends a notification, which the peer does not answer.
    ///
    /// The message is queued for the reader task and written before the next read.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] when the client is already closed.
    pub async fn notify(&self, method: &str, params: JsonValue) -> HarnessResult<()> {
        self.enqueue(&json!({"method": method, "params": params}), None)
            .await
    }

    /// Returns the next inbound message that is *not* a response.
    ///
    /// Notifications and server-to-client requests are delivered in arrival order. Returns [`None`]
    /// once the stream ended or the client was closed and everything queued was consumed.
    pub async fn next_incoming(&self) -> Option<JsonObject> {
        let mut guard = self.incoming.lock().await;
        let receiver = guard.as_mut()?;
        match receiver.try_recv() {
            Ok(message) => return Some(message),
            Err(mpsc::error::TryRecvError::Disconnected) => return None,
            Err(mpsc::error::TryRecvError::Empty) => {}
        }
        self.incoming_waiters.fetch_add(1, Ordering::SeqCst);
        self.demand.notify_one();
        let message = receiver.recv().await;
        self.incoming_waiters.fetch_sub(1, Ordering::SeqCst);
        message
    }

    /// Answers a server-to-client request with a result.
    ///
    /// `id` must be the value the peer sent; it is echoed verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] when the client is already closed.
    // The write happens in the reader task, see [`notify`](Self::notify).
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn respond(&self, id: JsonValue, result: JsonValue) -> HarnessResult<()> {
        self.enqueue(&json!({"id": id, "result": result}), None)
            .await
    }

    /// Answers a server-to-client request with an error object.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] when the client is already closed.
    // The write happens in the reader task, see [`notify`](Self::notify).
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn respond_error(
        &self,
        id: JsonValue,
        code: i64,
        message: &str,
    ) -> HarnessResult<()> {
        self.enqueue(
            &json!({"id": id, "error": {"code": code, "message": message}}),
            None,
        )
        .await
    }

    /// Stops the reader task, fails every pending request and drops the transport.
    ///
    /// Idempotent: a second call returns immediately. Pending requests of an already ended stream
    /// keep the failure they were failed with.
    pub async fn close(&self) {
        let already = self.closed.swap(true, Ordering::SeqCst);
        if !already {
            self.demand.notify_one();
        }
        self.abort_reader().await;
        self.shutdown("", &Ending::Closed);
        self.drop_transport();
    }

    /// Returns whether the stream ended or the client was closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Returns the label used in error messages, for example `Codex`.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Returns the peer's stderr tail.
    ///
    /// The live tail is read from the transport when the reader task is not holding it; otherwise
    /// the last tail the reader observed is returned, so this never blocks.
    #[must_use]
    pub fn stderr_tail(&self) -> String {
        match self.transport.try_lock() {
            Ok(guard) => match guard.as_ref() {
                Some(transport) => {
                    let tail = transport.stderr_tail();
                    self.record_tail(&tail);
                    tail
                }
                None => lock(&self.last_stderr).clone(),
            },
            Err(_) => lock(&self.last_stderr).clone(),
        }
    }

    /// Mints the next request id, sends the request and returns its waiter.
    async fn start_request(
        &self,
        method: &str,
        params: &JsonValue,
    ) -> HarnessResult<(u64, oneshot::Receiver<HarnessResult<JsonValue>>)> {
        if self.is_closed() {
            return Err(self.closed_error());
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        let (sender, receiver) = oneshot::channel();
        lock(&self.pending).insert(
            id,
            PendingRequest {
                method: method.to_owned(),
                sender,
            },
        );
        let message = json!({"id": id, "method": method, "params": params});
        self.enqueue(&message, Some(id)).await?;
        Ok((id, receiver))
    }

    /// Waits for the response of request `id`.
    async fn await_response(
        &self,
        id: u64,
        receiver: oneshot::Receiver<HarnessResult<JsonValue>>,
    ) -> HarnessResult<JsonValue> {
        // The sender is dropped only when the pending map is emptied without an answer.
        let Ok(result) = receiver.await else {
            lock(&self.pending).remove(&id);
            return Err(self.closed_error());
        };
        result
    }

    /// Sends one encoded message, either immediately or through the reader's queue.
    async fn enqueue(&self, message: &JsonValue, id: Option<u64>) -> HarnessResult<()> {
        if self.is_closed() {
            return Err(self.closed_error());
        }
        let line = encode_line(message);
        if let Some(writer) = self.writer.as_ref() {
            if let Err(error) = writer.write_line(&line).await {
                if let Some(id) = id {
                    self.fail_request(id, error.message());
                }
                return Err(error);
            }
            // The reader may be parked because nothing was pending when it last looked; wake it so
            // it starts reading the answer this message is about to produce.
            self.demand.notify_one();
            return Ok(());
        }
        lock(&self.outbound).push_back(Outbound { line, id });
        self.demand.notify_one();
        Ok(())
    }

    /// Fails one pending request with `message`, when it is still pending.
    fn fail_request(&self, id: u64, message: &str) {
        if let Some(entry) = lock(&self.pending).remove(&id) {
            let _ = entry
                .sender
                .send(Err(HarnessError::protocol(message.to_owned())));
        }
    }

    /// Marks the client closed, fails every pending request and ends the incoming channel.
    fn shutdown(&self, tail: &str, ending: &Ending) {
        self.closed.store(true, Ordering::SeqCst);
        let pending = std::mem::take(&mut *lock(&self.pending));
        for entry in pending.into_values() {
            let message = ending.message(self.label, &entry.method);
            let message = if tail.is_empty() {
                message
            } else {
                format!("{message}\n{tail}")
            };
            let _ = entry.sender.send(Err(HarnessError::protocol(message)));
        }
        let sender = lock(&self.incoming_sender).take();
        drop(sender);
    }

    /// Routes one inbound message, returning whether the peer awaits an answer from the application.
    fn dispatch(&self, value: &JsonValue) -> bool {
        self.read_errors.store(0, Ordering::SeqCst);
        let Some(object) = value.as_object() else {
            tracing::warn!(
                label = self.label,
                "ignoring a JSON-RPC message that is not an object"
            );
            return false;
        };
        let message: JsonObject = object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !message.contains_key("method") {
            match message.get("id").cloned() {
                Some(id) => self.resolve_response(&id, &message),
                None => {
                    tracing::warn!(
                        label = self.label,
                        "ignoring a JSON-RPC message with neither id nor method"
                    );
                }
            }
            return false;
        }
        let answers_peer = message.contains_key("id");
        self.push_incoming(message);
        answers_peer
    }

    /// Resolves the pending request `id` answered.
    fn resolve_response(&self, id: &JsonValue, message: &JsonObject) {
        let Some(id) = id.as_u64() else {
            tracing::warn!(
                label = self.label,
                "ignoring a JSON-RPC response with a non-numeric id"
            );
            return;
        };
        let Some(entry) = lock(&self.pending).remove(&id) else {
            tracing::warn!(
                id,
                label = self.label,
                "ignoring a JSON-RPC response for an unknown id"
            );
            return;
        };
        let result = match message.get("error") {
            Some(error) if error.is_object() => {
                let detail = error
                    .get("message")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("unknown error");
                let code = error.get("code").and_then(JsonValue::as_i64).unwrap_or(0);
                Err(HarnessError::protocol(format!(
                    "{} JSON-RPC error for '{}': {detail} (code {code})",
                    self.label, entry.method
                )))
            }
            _ => Ok(message.get("result").cloned().unwrap_or(JsonValue::Null)),
        };
        let _ = entry.sender.send(result);
    }

    /// Hands an inbound notification or server request to the application.
    fn push_incoming(&self, message: JsonObject) {
        if let Some(sender) = lock(&self.incoming_sender).as_ref() {
            if sender.send(message).is_err() {
                tracing::debug!(
                    label = self.label,
                    "dropping an inbound JSON-RPC message: nothing is reading it"
                );
            }
        } else {
            tracing::debug!(
                label = self.label,
                "dropping an inbound JSON-RPC message: the client is closed"
            );
        }
    }

    /// Returns whether the peer is expected to send something.
    fn has_demand(&self) -> bool {
        !lock(&self.pending).is_empty() || self.incoming_waiters.load(Ordering::SeqCst) > 0
    }

    /// Records `tail` as the last observed stderr tail.
    fn record_tail(&self, tail: &str) {
        if !tail.is_empty() {
            tail.clone_into(&mut lock(&self.last_stderr));
        }
    }

    /// Counts one unreadable line, returning `false` once the stream is declared unusable.
    fn note_read_error(&self, error: &HarnessError) -> bool {
        let consecutive = self.read_errors.fetch_add(1, Ordering::SeqCst) + 1;
        tracing::warn!(
            label = self.label,
            %error,
            consecutive,
            "skipping an unreadable JSON-RPC line"
        );
        consecutive < MAX_CONSECUTIVE_READ_ERRORS
    }

    /// Returns the "client closed" error.
    fn closed_error(&self) -> HarnessError {
        HarnessError::protocol(format!("the {} CLI closed its JSON-RPC stream", self.label))
    }

    /// Aborts and joins the reader task, bounded so a wedged transport cannot hang a close.
    async fn abort_reader(&self) {
        let Some(handle) = lock(&self.reader).take() else {
            return;
        };
        handle.abort();
        if tokio::time::timeout(READER_STOP_TIMEOUT, handle)
            .await
            .is_err()
        {
            tracing::warn!(
                label = self.label,
                "the JSON-RPC reader task did not stop within the timeout"
            );
        }
    }

    /// Drops the transport, which closes the child process.
    fn drop_transport(&self) {
        if let Ok(mut guard) = self.transport.try_lock() {
            guard.take();
        }
    }
}

impl Drop for JsonRpcClient {
    fn drop(&mut self) {
        if let Some(handle) = lock(&self.reader).take() {
            handle.abort();
        }
    }
}

/// The reader task: flushes queued writes, then reads exactly one message per iteration.
async fn read_loop(state: ReaderState) {
    loop {
        if !flush_outbound(&state).await {
            return;
        }
        let Some(client) = state.client.upgrade() else {
            return;
        };
        if client.is_closed() {
            return;
        }
        let demand = client.has_demand();
        drop(client);
        if !demand {
            state.demand.notified().await;
            continue;
        }
        match read_message(&state).await {
            ReadOutcome::Message(value) => {
                let Some(client) = state.client.upgrade() else {
                    return;
                };
                let answers_peer = client.dispatch(&value);
                drop(client);
                if answers_peer {
                    // The application owes this peer an answer: park so the answer is queued and
                    // flushed before the reader blocks on the stream again.
                    state.demand.notified().await;
                }
            }
            ReadOutcome::Ended(tail) => {
                if let Some(client) = state.client.upgrade() {
                    client.record_tail(&tail);
                    client.shutdown(&tail, &Ending::Exited);
                }
                return;
            }
            ReadOutcome::Failed(error, tail) => {
                let Some(client) = state.client.upgrade() else {
                    return;
                };
                client.record_tail(&tail);
                if client.note_read_error(&error) {
                    continue;
                }
                tracing::error!(
                    label = client.label(),
                    %error,
                    "the JSON-RPC stream is unusable; closing the client"
                );
                client.shutdown(&tail, &Ending::Failed(error.message().to_owned()));
                return;
            }
            ReadOutcome::Gone => return,
        }
    }
}

/// Writes every queued message, returning `false` when the reader must stop.
async fn flush_outbound(state: &ReaderState) -> bool {
    loop {
        let queued = lock(&state.outbound).pop_front();
        let Some(outbound) = queued else {
            return true;
        };
        let mut guard = state.transport.lock().await;
        let Some(transport) = guard.as_mut() else {
            return false;
        };
        let written = transport.write_line(&outbound.line).await;
        let tail = match &written {
            Ok(()) => String::new(),
            Err(_) => transport.stderr_tail(),
        };
        drop(guard);
        let error = match written {
            Ok(()) => continue,
            Err(error) => error,
        };
        let Some(client) = state.client.upgrade() else {
            return false;
        };
        client.record_tail(&tail);
        tracing::warn!(label = client.label(), %error, "failed to write a JSON-RPC message");
        let message = if tail.is_empty() {
            error.message().to_owned()
        } else {
            format!("{}\n{tail}", error.message())
        };
        if let Some(id) = outbound.id {
            client.fail_request(id, &message);
        }
        client.shutdown(&tail, &Ending::Failed(error.message().to_owned()));
        return false;
    }
}

/// One read from the transport.
enum ReadOutcome {
    /// One inbound message.
    Message(JsonValue),
    /// The peer closed the stream; carries the diagnostic tail.
    Ended(String),
    /// The line could not be read or decoded; carries the error and the diagnostic tail.
    Failed(HarnessError, String),
    /// The transport is gone, so the reader has nothing left to do.
    Gone,
}

/// Reads exactly one message, holding the transport for the whole read.
async fn read_message(state: &ReaderState) -> ReadOutcome {
    let mut guard = state.transport.lock().await;
    let Some(transport) = guard.as_mut() else {
        return ReadOutcome::Gone;
    };
    let outcome = transport.read_json().await;
    let tail = match &outcome {
        Ok(Some(_)) => String::new(),
        Ok(None) | Err(_) => transport.stderr_tail(),
    };
    match outcome {
        Ok(Some(value)) => ReadOutcome::Message(value),
        Ok(None) => ReadOutcome::Ended(tail),
        Err(error) => ReadOutcome::Failed(error, tail),
    }
}

#[cfg(test)]
mod tests;
