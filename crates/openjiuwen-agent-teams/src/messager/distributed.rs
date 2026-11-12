//! Feature-gated stub of the deferred `ZeroMQ` transport.
//!
//! Python parity source: `openjiuwen/agent_teams/messager/pyzmq_backend.py`. The backend is
//! documented as deferred in the crate `CONTEXT.md`; this type exists only so the seam is visible
//! in rustdoc and so a caller that names `pyzmq` gets the same diagnostic the factory returns.
//! There is deliberately no constructible stub instance: [`PyZmqMessager::new`] always fails, and
//! [`openjiuwen_common::error::Error::Unsupported`] keeps the Python capability name so the parity
//! table can match it.

use openjiuwen_common::error::{Error, Result};

use super::factory::PyZmqMessagerConfig;

/// The deferred `ZeroMQ` messager (never constructed in this port).
#[derive(Debug)]
pub struct PyZmqMessager;

impl PyZmqMessager {
    /// Reports that the `pyzmq` transport is not implemented.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] — the `pyzmq` capability is deferred; use the
    /// in-process transport instead.
    pub fn new(config: PyZmqMessagerConfig) -> Result<Self> {
        let team_name = config.team_name;
        tracing::debug!(
            team_name,
            "pyzmq transport requested; the capability is deferred"
        );
        Err(Error::unsupported(
            "transport 'pyzmq' is not implemented; use 'inprocess'",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_is_rejected_as_unsupported() {
        let Err(err) = PyZmqMessager::new(PyZmqMessagerConfig {
            team_name: "alpha".to_owned(),
            ..PyZmqMessagerConfig::default()
        }) else {
            panic!("expected pyzmq construction to be rejected");
        };
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert_eq!(
            err.to_string(),
            "unsupported: transport 'pyzmq' is not implemented; use 'inprocess'"
        );
    }
}
