//! TLS transport for `rediss://` URLs, built on `tokio-rustls`.
//!
//! The trust roots come from [`webpki_roots::TLS_SERVER_ROOTS`], so server
//! certificates are verified against the public web PKI with the host name (or IP
//! address) taken from the connection URL. Certificate and host-name verification
//! are never disabled: there is no insecure escape hatch.
//!
//! `ring` is the only crypto provider compiled into this workspace (no `aws-lc-rs`
//! offline), so the configuration installs `ring` as the process default when no
//! other provider has claimed it and always builds with the `ring` provider
//! explicitly, which keeps a host application that installed `aws-lc-rs` working.
//!
//! A handshake failure, including a peer that is not speaking TLS at all, is
//! reported as [`openjiuwen_common::error::ErrorKind::Transport`] naming TLS; a
//! handshake that exceeds the connection timeout is
//! [`openjiuwen_common::error::ErrorKind::Timeout`].
use openjiuwen_common::error::{Error, Result};
use std::result::Result as StdResult;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

/// Wraps `stream` in a TLS session authenticated against the web PKI.
pub(super) async fn handshake(
    stream: TcpStream,
    host: &str,
    port: u16,
    timeout: Duration,
) -> Result<TlsStream<TcpStream>> {
    let config = client_config()?;
    let server_name = ServerName::try_from(host.to_owned()).map_err(|error| {
        Error::validation(format!(
            "redis TLS server name '{host}' is not a valid DNS name or IP address: {error}"
        ))
    })?;
    let connector = TlsConnector::from(Arc::clone(config));
    tokio::time::timeout(timeout, connector.connect(server_name, stream))
        .await
        .map_err(|_| {
            Error::timeout(format!(
                "redis TLS handshake with {host}:{port} timed out after {timeout:?}"
            ))
        })?
        .map_err(|error| {
            Error::transport(format!(
                "redis TLS handshake with {host}:{port} failed: {error}"
            ))
        })
}

/// The process-wide client configuration (built once, including the trust roots).
fn client_config() -> Result<&'static Arc<ClientConfig>> {
    static CONFIG: OnceLock<StdResult<Arc<ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(build_client_config)
        .as_ref()
        .map_err(|reason| Error::transport(format!("redis TLS is unavailable: {reason}")))
}

/// Builds the client configuration, carrying rustls' own reason on failure.
fn build_client_config() -> StdResult<Arc<ClientConfig>, String> {
    let provider = tokio_rustls::rustls::crypto::ring::default_provider();
    // A host application may already have installed a provider; the client always
    // builds with `ring` explicitly, so an existing default is simply kept.
    let _ = provider.clone().install_default();
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let builder = ClientConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions()
        .map_err(|error| {
            format!(
                "rustls rejected the default protocol versions for the `ring` crypto provider: \
                 {error}"
            )
        })?;
    Ok(Arc::new(
        builder.with_root_certificates(roots).with_no_client_auth(),
    ))
}
