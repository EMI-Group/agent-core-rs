//! Authentication for the `http_request` component.
//!
//! Basic, bearer, and API-key authentication are resolved while the request is built.
//! Digest authentication is challenge-driven — [`apply_authentication`] only validates the
//! credentials and the component replays the request once a `401` arrives (see
//! [`super::digest`]). AWS SigV4 is signed in [`sign_aws`], called after the body bytes are
//! known, because the payload hash is part of the signature.
//!
//! # AWS credentials
//!
//! | Value | Source | Default |
//! | --- | --- | --- |
//! | access key | `authentication.username`, else `AWS_ACCESS_KEY_ID` | — (required) |
//! | secret key | `authentication.password`, else `AWS_SECRET_ACCESS_KEY` | — (required) |
//! | session token | `AWS_SESSION_TOKEN` | none |
//! | region | `AWS_REGION`, else `AWS_DEFAULT_REGION` | `us-east-1` |
//! | service | `AWS_SERVICE` | `execute-api` |
//!
//! `execute-api` is the default service because API Gateway is the service an HTTP
//! component most often signs for; `AWS_SERVICE` overrides it for any other AWS API.
//! A missing access/secret key pair is a typed input-parameter error naming the
//! environment variables, never an unsigned request.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;

use openjiuwen_common::error::Result;

use super::config::{HttpAuthConfig, HttpAuthIn, HttpAuthType};
use super::sigv4::{self, Credentials, SigningRequest};
use crate::error::component_tool_input_param_error;

/// Environment variable holding the AWS access key id.
pub(crate) const ACCESS_KEY_ENV: &str = "AWS_ACCESS_KEY_ID";

/// Environment variable holding the AWS secret access key.
pub(crate) const SECRET_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";

/// Environment variable holding an AWS session token.
pub(crate) const SESSION_TOKEN_ENV: &str = "AWS_SESSION_TOKEN";

/// Environment variable holding the AWS region.
pub(crate) const REGION_ENV: &str = "AWS_REGION";

/// Environment variable holding the AWS region fallback.
pub(crate) const DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";

/// Environment variable holding the AWS service name.
pub(crate) const SERVICE_ENV: &str = "AWS_SERVICE";

/// The region used when no region is configured.
pub(crate) const DEFAULT_REGION: &str = "us-east-1";

/// The service used when `AWS_SERVICE` is not set.
pub(crate) const DEFAULT_SERVICE: &str = "execute-api";

/// Applies the header/query-authentications that need no challenge.
///
/// Digest credentials are validated here (and used later by the replay); AWS signing is
/// deferred to [`sign_aws`].
pub(crate) fn apply_authentication(
    authentication: Option<&HttpAuthConfig>,
    headers: &mut BTreeMap<String, String>,
    query: &mut Vec<(String, String)>,
) -> Result<()> {
    let Some(auth) = authentication else {
        return Ok(());
    };
    match auth.auth_type {
        HttpAuthType::None | HttpAuthType::Aws => {}
        HttpAuthType::Basic => {
            // Python only applies basic auth when both values are truthy.
            if let (Some(username), Some(password)) = (
                auth.username.as_deref().filter(|value| !value.is_empty()),
                auth.password.as_deref().filter(|value| !value.is_empty()),
            ) {
                let encoded = BASE64_STANDARD.encode(format!("{username}:{password}"));
                let _ = headers.insert("Authorization".to_owned(), format!("Basic {encoded}"));
            }
        }
        HttpAuthType::Bearer => {
            if let Some(token) = auth.token.as_deref().filter(|value| !value.is_empty()) {
                let _ = headers.insert("Authorization".to_owned(), format!("Bearer {token}"));
            }
        }
        HttpAuthType::ApiKey => {
            if let Some(api_key) = auth.api_key.as_deref().filter(|value| !value.is_empty()) {
                match auth.in_location {
                    HttpAuthIn::Header => {
                        let _ = headers.insert(auth.name.clone(), api_key.to_owned());
                    }
                    HttpAuthIn::Query => query.push((auth.name.clone(), api_key.to_owned())),
                    // Python ignores the body location as well.
                    HttpAuthIn::Body => {}
                }
            }
        }
        HttpAuthType::Digest => {
            // The replay needs the pair; failing before the first send is clearer than
            // answering a challenge with an empty username.
            let _ = digest_credentials(auth)?;
        }
    }
    Ok(())
}

/// Whether `authentication` selects AWS SigV4.
pub(crate) fn is_aws(authentication: Option<&HttpAuthConfig>) -> bool {
    authentication.is_some_and(|auth| auth.auth_type == HttpAuthType::Aws)
}

/// The user name and password of a digest authentication, or a typed error.
pub(crate) fn digest_credentials(auth: &HttpAuthConfig) -> Result<(&str, &str)> {
    match (
        auth.username.as_deref().filter(|value| !value.is_empty()),
        auth.password.as_deref().filter(|value| !value.is_empty()),
    ) {
        (Some(username), Some(password)) => Ok((username, password)),
        _ => Err(component_tool_input_param_error(
            "digest authentication requires authentication.username and \
             authentication.password",
        )),
    }
}

/// Signs `headers` with AWS SigV4 and records the `x-amz-*` headers the signature covers.
pub(crate) fn sign_aws(
    authentication: Option<&HttpAuthConfig>,
    method: &str,
    url: &str,
    query: &[(String, String)],
    headers: &mut BTreeMap<String, String>,
    payload: &[u8],
) -> Result<()> {
    let credentials = aws_credentials(authentication)?;
    let sent: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let region = resolve_region(|name| std::env::var(name).ok());
    let service = resolve_service(|name| std::env::var(name).ok());
    let signed = sigv4::sign(
        &credentials,
        &SigningRequest {
            method,
            url,
            query,
            headers: &sent,
            payload,
            region: &region,
            service: &service,
            timestamp: now_seconds(),
        },
    );
    let _ = headers.insert("x-amz-date".to_owned(), signed.amz_date);
    let _ = headers.insert("x-amz-content-sha256".to_owned(), signed.payload_hash);
    if let Some(token) = signed.session_token {
        let _ = headers.insert("x-amz-security-token".to_owned(), token);
    }
    let _ = headers.insert("Authorization".to_owned(), signed.authorization);
    Ok(())
}

/// Resolves the AWS credentials from the configuration, falling back to the environment.
pub(crate) fn aws_credentials(authentication: Option<&HttpAuthConfig>) -> Result<Credentials> {
    resolve_aws_credentials(authentication, |name| std::env::var(name).ok())
}

/// Resolves the AWS credentials with an injectable environment lookup.
fn resolve_aws_credentials(
    authentication: Option<&HttpAuthConfig>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<Credentials> {
    let auth = authentication.filter(|auth| auth.auth_type == HttpAuthType::Aws);
    let configured = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    let access_key = configured(auth.and_then(|auth| auth.username.as_deref()))
        .or_else(|| configured(lookup(ACCESS_KEY_ENV).as_deref()));
    let secret_key = configured(auth.and_then(|auth| auth.password.as_deref()))
        .or_else(|| configured(lookup(SECRET_KEY_ENV).as_deref()));
    let (Some(access_key), Some(secret_key)) = (access_key, secret_key) else {
        return Err(component_tool_input_param_error(&format!(
            "aws authentication requires an access key and a secret key: set \
             authentication.username/authentication.password or the {ACCESS_KEY_ENV} and \
             {SECRET_KEY_ENV} environment variables"
        )));
    };
    Ok(Credentials {
        access_key,
        secret_key,
        session_token: configured(lookup(SESSION_TOKEN_ENV).as_deref()),
    })
}

/// Resolves the signature region from the environment lookup.
fn resolve_region(lookup: impl Fn(&str) -> Option<String>) -> String {
    [REGION_ENV, DEFAULT_REGION_ENV]
        .iter()
        .find_map(|name| lookup(name))
        .map(|region| region.trim().to_owned())
        .filter(|region| !region.is_empty())
        .unwrap_or_else(|| DEFAULT_REGION.to_owned())
}

/// Resolves the signature service from the environment lookup.
fn resolve_service(lookup: impl Fn(&str) -> Option<String>) -> String {
    lookup(SERVICE_ENV)
        .map(|service| service.trim().to_owned())
        .filter(|service| !service.is_empty())
        .unwrap_or_else(|| DEFAULT_SERVICE.to_owned())
}

/// The current Unix timestamp in seconds.
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A lookup over a fixed table, standing in for the process environment.
    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn authentication_places_credentials_like_python() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Basic,
            username: Some("user".to_owned()),
            password: Some("pass".to_owned()),
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        let mut query = Vec::new();
        apply_authentication(Some(&auth), &mut headers, &mut query).unwrap();
        assert_eq!(headers.get("Authorization").unwrap(), "Basic dXNlcjpwYXNz");

        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Bearer,
            token: Some("t".to_owned()),
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        apply_authentication(Some(&auth), &mut headers, &mut Vec::new()).unwrap();
        assert_eq!(headers.get("Authorization").unwrap(), "Bearer t");

        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::ApiKey,
            api_key: Some("k".to_owned()),
            name: "X-Key".to_owned(),
            in_location: HttpAuthIn::Query,
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        let mut query = Vec::new();
        apply_authentication(Some(&auth), &mut headers, &mut query).unwrap();
        assert_eq!(query, vec![("X-Key".to_owned(), "k".to_owned())]);
        assert!(headers.is_empty());

        // `in_location = body` is ignored, as in Python.
        let auth = HttpAuthConfig {
            in_location: HttpAuthIn::Body,
            ..auth
        };
        let mut query = Vec::new();
        apply_authentication(Some(&auth), &mut BTreeMap::new(), &mut query).unwrap();
        assert!(query.is_empty());
    }

    #[test]
    fn basic_and_bearer_authentication_skip_empty_secrets() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Basic,
            username: Some("user".to_owned()),
            password: None,
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        apply_authentication(Some(&auth), &mut headers, &mut Vec::new()).unwrap();
        assert!(headers.is_empty());

        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Bearer,
            token: Some(String::new()),
            ..HttpAuthConfig::default()
        };
        apply_authentication(Some(&auth), &mut headers, &mut Vec::new()).unwrap();
        assert!(headers.is_empty());
    }

    #[test]
    fn digest_authentication_requires_a_credential_pair() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Digest,
            ..HttpAuthConfig::default()
        };
        let error =
            apply_authentication(Some(&auth), &mut BTreeMap::new(), &mut Vec::new()).unwrap_err();
        assert!(error.message().contains("[102001]"), "{}", error.message());
        assert!(error.message().contains("authentication.username"));
        // With both values present nothing is applied: the replay carries the header.
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Digest,
            username: Some("u".to_owned()),
            password: Some("p".to_owned()),
            ..HttpAuthConfig::default()
        };
        let mut headers = BTreeMap::new();
        apply_authentication(Some(&auth), &mut headers, &mut Vec::new()).unwrap();
        assert!(headers.is_empty());
        assert_eq!(digest_credentials(&auth).unwrap(), ("u", "p"));
    }

    #[test]
    fn aws_credentials_prefer_the_configuration_then_the_environment() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Aws,
            username: Some("configured-key".to_owned()),
            password: Some("configured-secret".to_owned()),
            ..HttpAuthConfig::default()
        };
        let credentials = resolve_aws_credentials(
            Some(&auth),
            env(&[
                (ACCESS_KEY_ENV, "env-key"),
                (SECRET_KEY_ENV, "env-secret"),
                (SESSION_TOKEN_ENV, "env-token"),
            ]),
        )
        .unwrap();
        assert_eq!(credentials.access_key, "configured-key");
        assert_eq!(credentials.secret_key, "configured-secret");
        assert_eq!(credentials.session_token.as_deref(), Some("env-token"));

        let credentials = resolve_aws_credentials(Some(&auth), env(&[])).unwrap();
        assert_eq!(credentials.access_key, "configured-key");
        assert_eq!(credentials.session_token, None);

        // An `Aws` config with no values falls back to the environment.
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Aws,
            ..HttpAuthConfig::default()
        };
        let credentials = resolve_aws_credentials(
            Some(&auth),
            env(&[(ACCESS_KEY_ENV, "env-key"), (SECRET_KEY_ENV, "env-secret")]),
        )
        .unwrap();
        assert_eq!(credentials.access_key, "env-key");
        assert_eq!(credentials.secret_key, "env-secret");
    }

    #[test]
    fn missing_aws_credentials_name_the_environment_variables() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Aws,
            ..HttpAuthConfig::default()
        };
        let error = resolve_aws_credentials(Some(&auth), env(&[])).unwrap_err();
        assert!(error.message().contains("[102001]"), "{}", error.message());
        assert!(error.message().contains(ACCESS_KEY_ENV));
        assert!(error.message().contains(SECRET_KEY_ENV));
        // A lone access key is not a usable pair either.
        let error =
            resolve_aws_credentials(Some(&auth), env(&[(ACCESS_KEY_ENV, "key")])).unwrap_err();
        assert!(error.message().contains(SECRET_KEY_ENV));
    }

    #[test]
    fn the_region_and_service_have_documented_defaults() {
        assert_eq!(resolve_region(env(&[])), "us-east-1");
        assert_eq!(
            resolve_region(env(&[(DEFAULT_REGION_ENV, "eu-west-1")])),
            "eu-west-1"
        );
        assert_eq!(
            resolve_region(env(&[
                (REGION_ENV, "ap-south-1"),
                (DEFAULT_REGION_ENV, "eu-west-1")
            ])),
            "ap-south-1"
        );
        assert_eq!(resolve_service(env(&[])), "execute-api");
        assert_eq!(resolve_service(env(&[(SERVICE_ENV, "s3")])), "s3");
        // Blank values fall back to the defaults.
        assert_eq!(resolve_region(env(&[(REGION_ENV, "  ")])), "us-east-1");
        assert_eq!(resolve_service(env(&[(SERVICE_ENV, "")])), "execute-api");
    }

    #[test]
    fn signing_records_every_amz_header() {
        let auth = HttpAuthConfig {
            auth_type: HttpAuthType::Aws,
            username: Some("AKIDEXAMPLE".to_owned()),
            password: Some("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_owned()),
            ..HttpAuthConfig::default()
        };
        let mut headers =
            BTreeMap::from([("content-type".to_owned(), "application/json".to_owned())]);
        sign_aws(
            Some(&auth),
            "POST",
            "https://api.example.com/upload",
            &[],
            &mut headers,
            br#"{"a":1}"#,
        )
        .unwrap();
        assert!(headers.contains_key("x-amz-date"));
        assert_eq!(
            headers.get("x-amz-content-sha256").unwrap(),
            "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
        );
        let authorization = headers.get("Authorization").unwrap();
        assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
        assert!(
            authorization
                .contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date"),
            "unexpected authorization: {authorization}"
        );
        assert!(is_aws(Some(&auth)));
        assert!(!is_aws(None));
    }
}
