//! TLS support for the RESP2 client (`rediss://` URLs).
//!
//! `resp.rs` is a file module, so this sibling file needs an explicit
//! `#[path]` declaration there; it holds everything that turns a
//! [`rustls::ClientConfig`] into a connected TLS transport:
//!
//! * the trust store — the compiled-in [`webpki_roots`] roots plus the extra
//!   anchors named by `checkpoint::redis::RedisOptions::tls_ca_file`,
//! * the server name derived from the URL host (an IP literal stays an IP,
//!   anything else is a DNS name), and
//! * the handshake itself, whose failures are
//!   [`openjiuwen_common::error::ErrorKind::Session`] errors naming the host.
//!
//! No system certificate store and no OpenSSL are involved, and the crypto
//! provider is `ring` (the workspace pins `rustls` with `ring`, never
//! aws-lc-rs), so the backend stays offline and dependency-light.
//!
//! A CA file is either a PEM bundle of `CERTIFICATE` blocks or a single DER
//! certificate. Both are read without a PEM or base64 crate: the PEM markers
//! are scanned for directly and the payload is decoded by the small base64
//! decoder at the bottom of this module. A structurally broken file is
//! [`openjiuwen_common::error::ErrorKind::Validation`]; an unreadable file, or a
//! certificate rustls cannot use as a trust anchor, is
//! [`openjiuwen_common::error::ErrorKind::Session`].

use std::path::Path;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;
use tokio_rustls::TlsConnector;

/// The PEM markers a certificate block is delimited by.
const PEM_BEGIN: &[u8] = b"-----BEGIN CERTIFICATE-----";
/// The PEM marker that closes a certificate block.
const PEM_END: &[u8] = b"-----END CERTIFICATE-----";

/// Builds the client configuration for `host`: the webpki roots plus, when
/// `ca_file` is set, the extra roots that file holds.
///
/// `host` only labels failures, which keeps a broken trust store reportable
/// before the socket is dialed.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for a malformed
/// CA file or an empty one, and
/// [`openjiuwen_common::error::ErrorKind::Session`] when the file cannot be
/// read, when it holds bytes rustls cannot use as a trust anchor, or when the
/// client configuration itself cannot be built.
pub(crate) async fn client_config(host: &str, ca_file: Option<&Path>) -> Result<Arc<ClientConfig>> {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    if let Some(path) = ca_file {
        for certificate in read_ca_certificates(path).await? {
            roots.add(certificate).map_err(|error| {
                Error::session(format!(
                    "CA file {} holds a certificate that cannot be trusted for the redis server \
                     {host}: {error}",
                    path.display()
                ))
            })?;
        }
    }
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| {
            Error::session(format!(
                "failed to build the TLS client configuration for the redis server {host}: \
                 {error}"
            ))
        })?;
    Ok(Arc::new(
        builder.with_root_certificates(roots).with_no_client_auth(),
    ))
}

/// Completes the TLS handshake over `socket` with the server named by `host`.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Session`] when the handshake
/// fails — an untrusted or mismatched certificate, a server that speaks
/// something other than TLS, or a transport error — and
/// [`openjiuwen_common::error::ErrorKind::Validation`] when `host` is not a
/// usable server name.
pub(crate) async fn handshake(
    socket: TcpStream,
    host: &str,
    config: Arc<ClientConfig>,
) -> Result<TlsStream<TcpStream>> {
    let name = server_name(host)?;
    TlsConnector::from(config)
        .connect(name, socket)
        .await
        .map_err(|error| {
            Error::session(format!(
                "TLS handshake with the redis server at {host} failed: {error}"
            ))
        })
}

/// Maps a URL host onto a TLS server name: an IP literal stays an IP address,
/// everything else is a DNS name.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `host` is
/// neither an IP address nor a valid DNS name.
pub(crate) fn server_name(host: &str) -> Result<ServerName<'static>> {
    if let Ok(address) = host.parse::<std::net::IpAddr>() {
        return Ok(ServerName::IpAddress(address.into()));
    }
    ServerName::try_from(host.to_string())
        .map_err(|error| Error::validation(format!("invalid TLS server name {host:?}: {error}")))
}

/// Reads the extra trust anchors from `path`.
///
/// A file that contains a `CERTIFICATE` PEM block is decoded as a PEM bundle;
/// anything else is taken as one DER certificate.
async fn read_ca_certificates(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let bytes = tokio::fs::read(path).await.map_err(|error| {
        Error::session(format!(
            "failed to read the TLS CA file {}: {error}",
            path.display()
        ))
    })?;
    if find(&bytes, PEM_BEGIN).is_some() {
        return decode_pem_bundle(&bytes, path);
    }
    if bytes.is_empty() {
        return Err(Error::validation(format!(
            "the TLS CA file {} is empty",
            path.display()
        )));
    }
    Ok(vec![CertificateDer::from(bytes)])
}

/// Decodes every `CERTIFICATE` block of a PEM bundle.
fn decode_pem_bundle(bytes: &[u8], path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let mut certificates = Vec::new();
    let mut rest = bytes;
    while let Some(start) = find(rest, PEM_BEGIN) {
        let after_begin = &rest[start + PEM_BEGIN.len()..];
        let Some(end) = find(after_begin, PEM_END) else {
            return Err(Error::validation(format!(
                "the TLS CA file {} has a certificate block without an END line",
                path.display()
            )));
        };
        let decoded = base64_decode(&after_begin[..end]).ok_or_else(|| {
            Error::validation(format!(
                "the TLS CA file {} has a certificate block that is not valid base64",
                path.display()
            ))
        })?;
        if decoded.is_empty() {
            return Err(Error::validation(format!(
                "the TLS CA file {} has an empty certificate block",
                path.display()
            )));
        }
        certificates.push(CertificateDer::from(decoded));
        rest = &after_begin[end + PEM_END.len()..];
    }
    if certificates.is_empty() {
        return Err(Error::validation(format!(
            "the TLS CA file {} holds no certificates",
            path.display()
        )));
    }
    Ok(certificates)
}

/// Returns the index of the first `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Decodes standard base64 (RFC 4648, `+`/`/` alphabet) with optional `=`
/// padding, ignoring line breaks, spaces and tabs.
///
/// Returns `None` for anything that is not a whole number of base64 groups, or
/// that contains a character outside the alphabet.
fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut symbols = Vec::with_capacity(input.len());
    let mut padding = 0_usize;
    for &byte in input {
        match byte {
            b'\r' | b'\n' | b' ' | b'\t' => {}
            b'=' => {
                padding += 1;
                if padding > 2 {
                    return None;
                }
            }
            _ => {
                if padding > 0 {
                    return None;
                }
                symbols.push(base64_symbol(byte)?);
            }
        }
    }
    if symbols.len() % 4 == 1 || (symbols.len() + padding) % 4 != 0 {
        return None;
    }
    let mut decoded = Vec::with_capacity(symbols.len() * 3 / 4 + 2);
    for group in symbols.chunks(4) {
        let mut accumulator = 0_u32;
        for (index, symbol) in group.iter().enumerate() {
            accumulator |= u32::from(*symbol) << (18 - 6 * index);
        }
        for index in 0..group.len().saturating_sub(1) {
            decoded.push(((accumulator >> (16 - 8 * index)) & 0xff) as u8);
        }
    }
    Some(decoded)
}

/// Maps one base64 character onto its six-bit value.
fn base64_symbol(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::{base64_decode, decode_pem_bundle, find, server_name};
    use std::path::Path;

    /// Encodes bytes the way the PEM fixtures are encoded, so the decoder and
    /// the encoder can check each other.
    fn base64_encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut encoded = String::new();
        for group in bytes.chunks(3) {
            let mut accumulator = 0_u32;
            for (index, byte) in group.iter().enumerate() {
                accumulator |= u32::from(*byte) << (16 - 8 * index);
            }
            for index in 0..=group.len() {
                let symbol = (accumulator >> (18 - 6 * index)) & 0x3f;
                encoded.push(char::from(ALPHABET[symbol as usize]));
            }
            for _ in 0..(3 - group.len()) {
                encoded.push('=');
            }
        }
        encoded
    }

    #[test]
    fn decodes_base64_the_encoder_produces() {
        for payload in [
            b"".as_slice(),
            b"f".as_slice(),
            b"fo".as_slice(),
            b"foo".as_slice(),
            b"foob".as_slice(),
            b"fooba".as_slice(),
            b"foobar".as_slice(),
            &[0x00, 0xff, 0x10, 0x7f, 0x80][..],
        ] {
            let encoded = base64_encode(payload);
            assert_eq!(
                base64_decode(encoded.as_bytes()).unwrap(),
                payload.to_vec(),
                "{encoded}"
            );
        }
        // Line breaks and padding are tolerated the way PEM bundles use them.
        assert_eq!(
            base64_decode(b"Zm9v\r\nYmFy\n").unwrap(),
            b"foobar".to_vec()
        );
        // Broken input is rejected rather than silently truncated.
        for malformed in [
            b"Z".as_slice(),
            b"Zm9".as_slice(),
            b"Zm9v!".as_slice(),
            b"Z===".as_slice(),
            b"Zm=v".as_slice(),
        ] {
            assert_eq!(base64_decode(malformed), None, "{malformed:?}");
        }
    }

    #[test]
    fn decodes_pem_bundles_and_reports_broken_ones() {
        let certificate = b"\x30\x82\x01\x00binary".to_vec();
        let encoded = base64_encode(&certificate);
        let bundle = format!(
            "# a comment\n-----BEGIN CERTIFICATE-----\n{encoded}\n-----END CERTIFICATE-----\n\
             -----BEGIN CERTIFICATE-----\n{encoded}\n-----END CERTIFICATE-----\n"
        );
        let certificates =
            decode_pem_bundle(bundle.as_bytes(), Path::new("ca.pem")).expect("decode the bundle");
        assert_eq!(certificates.len(), 2);
        assert_eq!(
            certificates[0],
            rustls::pki_types::CertificateDer::from(certificate.clone())
        );

        let broken = [
            (
                "-----BEGIN CERTIFICATE-----\nnot base64!\n-----END CERTIFICATE-----\n",
                "base64",
            ),
            ("-----BEGIN CERTIFICATE-----\nZm9v\n", "without an END line"),
            (
                "-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n",
                "empty certificate block",
            ),
        ];
        for (content, needle) in broken {
            let error = decode_pem_bundle(content.as_bytes(), Path::new("broken.pem"))
                .expect_err("a broken bundle must be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation, "{content:?}");
            assert!(error.to_string().contains(needle), "{error}");
        }
    }

    #[test]
    fn finds_pem_markers() {
        assert_eq!(find(b"abcd", b"cd"), Some(2));
        assert_eq!(find(b"abcd", b"ce"), None);
        assert_eq!(find(b"abc", b"abcd"), None);
        assert_eq!(find(b"abc", b""), None);
    }

    #[test]
    fn derives_server_names_from_url_hosts() {
        let name = server_name("127.0.0.1").unwrap();
        assert!(matches!(name, rustls::pki_types::ServerName::IpAddress(_)));
        let name = server_name("::1").unwrap();
        assert!(matches!(name, rustls::pki_types::ServerName::IpAddress(_)));
        let name = server_name("localhost").unwrap();
        assert!(matches!(name, rustls::pki_types::ServerName::DnsName(_)));
        let error = server_name("not a host").expect_err("spaces are not a DNS name");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
