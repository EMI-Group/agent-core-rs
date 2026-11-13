//! The attach socket of a live team: binding it on start and reporting its path.
//!
//! Serving lives in `super::super::attach_host`; this module only owns the handle-side binding.
use std::sync::Arc;

use openjiuwen_common::error::Result;
use tracing::{debug, warn};

use super::super::attach_host::{AttachHost, BackendAttachHandler};
use super::TeamAgent;

impl TeamAgent {
    /// Binds the attach server when the spec (or `OPENJIUWEN_TEAM_SOCKET`) names a socket.
    ///
    /// Bind failures are logged and ignored unless `attach.required` is set, in which case the
    /// error aborts the start (see `src/runtime/CONTEXT.md` §"Attach policy").
    pub(super) async fn start_attach(&self) -> Result<()> {
        let Some(config) = self.inner.attach.clone() else {
            return Ok(());
        };
        if self.inner.attach_host.lock().is_some() {
            return Ok(());
        }
        let handler: Arc<dyn crate::attach::AttachHandler> = Arc::new(BackendAttachHandler::new(
            Arc::clone(&self.inner.backend),
            self.leader_member_name(),
            self.cancellation_token(),
        ));
        match AttachHost::start(&config, handler).await {
            Ok(host) => {
                debug!(
                    team = self.team_name(),
                    socket = %host.path().display(),
                    "serving the team over the attach socket"
                );
                *self.inner.attach_host.lock() = Some(host);
            }
            Err(error) if config.required => return Err(error),
            Err(error) => {
                warn!(
                    team = self.team_name(),
                    socket = %config.socket.display(),
                    %error,
                    "attach socket could not be bound; continuing without it"
                );
            }
        }
        Ok(())
    }

    /// Returns the bind path of the live attach server, if one is serving.
    #[must_use]
    pub fn attach_socket(&self) -> Option<std::path::PathBuf> {
        self.inner
            .attach_host
            .lock()
            .as_ref()
            .map(|host| host.path().to_path_buf())
    }
}
