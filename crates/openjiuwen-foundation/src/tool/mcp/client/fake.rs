//! Test doubles for the [`McpClient`] seam.
//!
//! Split out of `client.rs` to keep that file under the crate's size baseline.
//! The fakes are `#[cfg(test)]`-only: nothing here is part of the public API.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde_json::{Map, Value};
use tokio::sync::Mutex;

use super::{McpClient, McpRemoteTool};
use crate::tool::mcp::result::McpToolResult;

/// An in-memory [`McpClient`] for tests: records calls, returns scripted results.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct FakeMcpClient {
    server_id: String,
    server_name: String,
    include_image_content: bool,
    tools: Mutex<Vec<McpRemoteTool>>,
    replies: Mutex<HashMap<String, FakeReply>>,
    calls: Mutex<Vec<FakeCall>>,
    connected: AtomicBool,
    connect_count: AtomicUsize,
    resource_subscriptions: AtomicBool,
    subscription_failure: Mutex<Option<String>>,
    subscription_calls: Mutex<Vec<FakeSubscriptionCall>>,
}

/// One recorded [`FakeMcpClient`] call.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct FakeCall {
    /// Tool that was called.
    pub tool_name: String,
    /// Arguments passed through.
    pub arguments: Value,
    /// Per-call deadline.
    pub timeout: Option<Duration>,
    /// Whether image blocks were requested as multimodal data.
    pub include_image_content: bool,
}

/// The resource-subscription request one [`FakeMcpClient`] call stands for.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeSubscriptionAction {
    /// A `resources/subscribe` request.
    Subscribe,
    /// A `resources/unsubscribe` request.
    Unsubscribe,
}

/// One recorded [`FakeMcpClient`] resource-subscription call.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeSubscriptionCall {
    /// Which request the client attempted.
    pub action: FakeSubscriptionAction,
    /// Resource URI the request addressed.
    pub uri: String,
}

#[cfg(test)]
#[derive(Debug, Clone)]
enum FakeReply {
    Result(McpToolResult),
    Failure(String),
}

#[cfg(test)]
impl FakeMcpClient {
    /// Creates a fake client for `server_id`/`server_name` with no tools.
    #[must_use]
    pub fn new(server_id: impl Into<String>, server_name: impl Into<String>) -> Self {
        Self {
            server_id: server_id.into(),
            server_name: server_name.into(),
            include_image_content: false,
            tools: Mutex::new(Vec::new()),
            replies: Mutex::new(HashMap::new()),
            calls: Mutex::new(Vec::new()),
            connected: AtomicBool::new(false),
            connect_count: AtomicUsize::new(0),
            resource_subscriptions: AtomicBool::new(false),
            subscription_failure: Mutex::new(None),
            subscription_calls: Mutex::new(Vec::new()),
        }
    }

    /// Sets the tools this client advertises.
    pub async fn set_tools(&self, tools: Vec<McpRemoteTool>) {
        *self.tools.lock().await = tools;
    }

    /// Scripts a successful reply for `tool_name`.
    pub async fn set_result(&self, tool_name: &str, result: McpToolResult) {
        self.replies
            .lock()
            .await
            .insert(tool_name.to_owned(), FakeReply::Result(result));
    }

    /// Scripts a failure reply for `tool_name`.
    pub async fn set_failure(&self, tool_name: &str, message: impl Into<String>) {
        self.replies
            .lock()
            .await
            .insert(tool_name.to_owned(), FakeReply::Failure(message.into()));
    }

    /// Returns the recorded calls, oldest first.
    pub async fn calls(&self) -> Vec<FakeCall> {
        self.calls.lock().await.clone()
    }

    /// Returns how many times [`McpClient::connect`] established a session.
    #[must_use]
    pub fn connect_count(&self) -> usize {
        self.connect_count.load(Ordering::SeqCst)
    }

    /// Sets the negotiated `resources.subscribe` capability this client reports.
    pub fn set_resource_subscriptions_supported(&self, supported: bool) {
        self.resource_subscriptions
            .store(supported, Ordering::SeqCst);
    }

    /// Scripts a failure for every later subscription request.
    ///
    /// The message becomes [`Error::Mcp`], matching a server that rejects
    /// `resources/subscribe`.
    pub async fn set_subscription_failure(&self, message: impl Into<String>) {
        *self.subscription_failure.lock().await = Some(message.into());
    }

    /// Clears a scripted subscription failure.
    pub async fn clear_subscription_failure(&self) {
        *self.subscription_failure.lock().await = None;
    }

    /// Returns the recorded subscription requests, oldest first.
    pub async fn subscription_calls(&self) -> Vec<FakeSubscriptionCall> {
        self.subscription_calls.lock().await.clone()
    }

    /// Records one subscription attempt and returns the scripted failure, if any.
    ///
    /// The fake records every attempt, including one a real server would refuse
    /// for want of the negotiated capability; scripting
    /// [`Self::set_resource_subscriptions_supported`] only changes what
    /// [`McpClient::supports_resource_subscriptions`] reports.
    async fn record_subscription(&self, action: FakeSubscriptionAction, uri: &str) -> Result<()> {
        self.subscription_calls
            .lock()
            .await
            .push(FakeSubscriptionCall {
                action,
                uri: uri.to_owned(),
            });
        match self.subscription_failure.lock().await.clone() {
            Some(message) => Err(Error::mcp(message)),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[async_trait]
impl McpClient for FakeMcpClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn include_image_content(&self) -> bool {
        self.include_image_content
    }

    async fn connect(&self) -> Result<()> {
        if !self.connected.swap(true, Ordering::SeqCst) {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
        self.connect().await?;
        Ok(self.tools.lock().await.clone())
    }

    async fn subscribe_resource(&self, uri: &str) -> Result<()> {
        self.record_subscription(FakeSubscriptionAction::Subscribe, uri)
            .await
    }

    async fn unsubscribe_resource(&self, uri: &str) -> Result<()> {
        self.record_subscription(FakeSubscriptionAction::Unsubscribe, uri)
            .await
    }

    fn supports_resource_subscriptions(&self) -> bool {
        self.resource_subscriptions.load(Ordering::SeqCst)
    }

    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: Value,
        timeout: Option<Duration>,
        include_image_content: bool,
    ) -> Result<McpToolResult> {
        self.connect().await?;
        self.calls.lock().await.push(FakeCall {
            tool_name: tool_name.to_owned(),
            arguments,
            timeout,
            include_image_content,
        });
        match self.replies.lock().await.get(tool_name).cloned() {
            Some(FakeReply::Result(result)) => Ok(result),
            Some(FakeReply::Failure(message)) => Err(Error::mcp(message)),
            None => Ok(McpToolResult::success(Map::new())),
        }
    }
}
