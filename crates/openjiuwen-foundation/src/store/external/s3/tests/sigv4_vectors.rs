//! Known-answer vectors for the in-crate Signature Version 4 signer.
//!
//! (a) the AWS `get-vanilla` vector of the `aws-sig-v4-test-suite`, (b) the
//! RFC 4231 HMAC-SHA256 vectors, (c) the S3 documentation `GET Object` example,
//! plus the fixed-clock request the store sends for a `PUT`.

use super::*;

#[test]
fn get_vanilla_known_answer_vector() {
    // AWS `aws-sig-v4-test-suite/get-vanilla`: the canonical request signs only
    // `host` and `x-amz-date`; `x-amz-content-sha256` is not part of it.
    let target = url("https://example.amazonaws.com/");
    let headers = vec![
        ("host".to_string(), "example.amazonaws.com".to_string()),
        ("x-amz-date".to_string(), "20150830T123600Z".to_string()),
    ];
    let result = sigv4::sign_request(
        &credentials(),
        &SigningInput {
            method: "GET",
            url: &target,
            query: &[],
            headers: &headers,
            payload_sha256_hex: EMPTY_PAYLOAD_SHA256,
            region: "us-east-1",
            service: "service",
            now: instant(2015, 8, 30, 12, 36, 0),
        },
    );

    assert_eq!(
        result.signature,
        "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
    );
    assert_eq!(
        result.canonical_request,
        concat!(
            "GET\n",
            "/\n",
            "\n",
            "host:example.amazonaws.com\n",
            "x-amz-date:20150830T123600Z\n",
            "\n",
            "host;x-amz-date\n",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
    );
    assert_eq!(
        result.string_to_sign,
        concat!(
            "AWS4-HMAC-SHA256\n",
            "20150830T123600Z\n",
            "20150830/us-east-1/service/aws4_request\n",
            "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63",
        )
    );
    assert_eq!(
        result.authorization,
        concat!(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, ",
            "SignedHeaders=host;x-amz-date, ",
            "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        )
    );

    let emitted = header_map(&result.headers);
    assert_eq!(
        emitted.get("x-amz-date").map(String::as_str),
        Some("20150830T123600Z")
    );
    assert_eq!(
        emitted.get("x-amz-content-sha256").map(String::as_str),
        Some(EMPTY_PAYLOAD_SHA256)
    );
    assert!(!emitted.contains_key("x-amz-security-token"));
}

#[test]
fn hmac_sha256_matches_the_rfc_4231_vectors() {
    let vectors: [(&[u8], &[u8], &str); 4] = [
        (
            &[0x0b; 20],
            b"Hi There",
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
        ),
        (
            b"Jefe",
            b"what do ya want for nothing?",
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
        ),
        (
            &[0xaa; 20],
            &[0xdd; 50],
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
        ),
        (
            &[0xaa; 131],
            b"Test Using Larger Than Block-Size Key - Hash Key First",
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
        ),
    ];

    for (key, data, expected) in vectors {
        assert_eq!(sigv4::hex_encode(&sigv4::hmac_sha256(key, data)), expected);
    }
}

#[test]
fn sha256_hex_hashes_the_empty_payload() {
    assert_eq!(sigv4::sha256_hex(b""), EMPTY_PAYLOAD_SHA256);
    assert_eq!(
        sigv4::sha256_hex(b"hello world"),
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );
}

#[test]
fn s3_documentation_get_object_vector() {
    // AWS S3 documentation, "Example: GET Object": the payload hash is signed
    // because the caller lists it in `headers`.
    let target = url("https://examplebucket.s3.amazonaws.com/test.txt");
    let headers = vec![
        (
            "host".to_string(),
            "examplebucket.s3.amazonaws.com".to_string(),
        ),
        ("range".to_string(), "bytes=0-9".to_string()),
        (
            "x-amz-content-sha256".to_string(),
            EMPTY_PAYLOAD_SHA256.to_string(),
        ),
        ("x-amz-date".to_string(), "20130524T000000Z".to_string()),
    ];
    let result = sigv4::sign_request(
        &Credentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ),
        &SigningInput {
            method: "GET",
            url: &target,
            query: &[],
            headers: &headers,
            payload_sha256_hex: EMPTY_PAYLOAD_SHA256,
            region: "us-east-1",
            service: "s3",
            now: instant(2013, 5, 24, 0, 0, 0),
        },
    );

    assert_eq!(
        result.signature,
        "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    );
    assert!(result
        .canonical_request
        .contains("host;range;x-amz-content-sha256;x-amz-date"));
}

#[test]
fn fixed_clock_put_matches_the_reference_signature() {
    // The shape the store sends for `put("bucket", "dir/a.txt", "hello world")`:
    // host, payload hash and date are all signed.
    let target = url("http://127.0.0.1:9000/bucket/dir/a.txt");
    let payload_hash = sigv4::sha256_hex(b"hello world");
    let headers = vec![
        ("host".to_string(), "127.0.0.1:9000".to_string()),
        ("x-amz-content-sha256".to_string(), payload_hash.clone()),
        ("x-amz-date".to_string(), "20240101T000000Z".to_string()),
    ];
    let result = sigv4::sign_request(
        &credentials(),
        &SigningInput {
            method: "PUT",
            url: &target,
            query: &[],
            headers: &headers,
            payload_sha256_hex: &payload_hash,
            region: REGION,
            service: "s3",
            now: instant(2024, 1, 1, 0, 0, 0),
        },
    );

    assert_eq!(
        result.signature,
        "643a9ec78b8a8b76f1af98f853b5ac0b704c2e4dd16344a92c313e9a668fabc4"
    );
    assert_eq!(
        result.canonical_request,
        concat!(
            "PUT\n",
            "/bucket/dir/a.txt\n",
            "\n",
            "host:127.0.0.1:9000\n",
            "x-amz-content-sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9\n",
            "x-amz-date:20240101T000000Z\n",
            "\n",
            "host;x-amz-content-sha256;x-amz-date\n",
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )
    );
}

#[test]
fn session_token_is_sent_and_signed() {
    let target = url("https://example.com/b");
    let result = sigv4::sign_request(
        &credentials().with_session_token("session-token"),
        &SigningInput {
            method: "GET",
            url: &target,
            query: &[],
            headers: &[],
            payload_sha256_hex: EMPTY_PAYLOAD_SHA256,
            region: REGION,
            service: "s3",
            now: instant(2024, 1, 1, 0, 0, 0),
        },
    );

    let emitted = header_map(&result.headers);
    assert_eq!(
        emitted.get("x-amz-security-token").map(String::as_str),
        Some("session-token")
    );
    assert!(result
        .canonical_request
        .contains("x-amz-security-token:session-token"));
    assert!(result
        .canonical_request
        .contains("host;x-amz-date;x-amz-security-token"));
    assert!(result
        .authorization
        .contains("SignedHeaders=host;x-amz-date;x-amz-security-token"));
}

#[test]
fn canonical_query_and_path_are_rfc3986_encoded() {
    let target = url("https://example.com/b/a%20b/%C3%A4/x.txt");
    let query = vec![
        ("b".to_string(), "two words".to_string()),
        ("a".to_string(), "z".to_string()),
        ("a".to_string(), "a".to_string()),
    ];
    let result = sigv4::sign_request(
        &credentials(),
        &SigningInput {
            method: "GET",
            url: &target,
            query: &query,
            headers: &[],
            payload_sha256_hex: EMPTY_PAYLOAD_SHA256,
            region: REGION,
            service: "s3",
            now: instant(2024, 1, 1, 0, 0, 0),
        },
    );

    let lines: Vec<&str> = result.canonical_request.lines().collect();
    assert_eq!(lines[0], "GET");
    // Segments are decoded once and re-encoded once: no `%2520`, no `+`.
    assert_eq!(lines[1], "/b/a%20b/%C3%A4/x.txt");
    assert_eq!(lines[2], "a=a&a=z&b=two%20words");
}

#[test]
fn credentials_redact_their_secrets() {
    let credentials = credentials().with_session_token("session-secret-xyz");
    let debug = format!("{credentials:?}");
    assert!(!debug.contains(SECRET_KEY));
    assert!(!debug.contains("session-secret-xyz"));
    assert!(!credentials.is_anonymous());

    let restored: Credentials = serde_json::from_str(
        r#"{"access_key_id":"a","secret_access_key":"b","session_token":"c"}"#,
    )
    .expect("credentials deserialize");
    assert_eq!(restored.access_key_id, "a");
    assert_eq!(restored.secret_access_key.expose_secret(), "b");
    assert_eq!(
        restored
            .session_token
            .as_ref()
            .map(openjiuwen_common::security::SecretString::expose_secret),
        Some("c")
    );
    assert!(Credentials::default().is_anonymous());
}
