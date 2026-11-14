//! Unit tests for the Azure Blob transport.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;

#[test]
fn blob_names_keep_separators_and_encode_every_segment() {
    assert_eq!(
        paths::encode_blob_name("graph/versions/v1/graph.json"),
        "graph/versions/v1/graph.json"
    );
    assert_eq!(paths::encode_blob_name("a b/c d.json"), "a%20b/c%20d.json");
    assert_eq!(
        paths::encode_blob_name("gráf/图.json"),
        "gr%C3%A1f/%E5%9B%BE.json"
    );
    assert_eq!(paths::encode_blob_name("keep-me_1.2~3"), "keep-me_1.2~3");
    assert_eq!(paths::encode_blob_name("a?b#c%d"), "a%3Fb%23c%25d");
    assert_eq!(paths::encode_blob_name(""), "");
}

#[test]
fn the_default_configuration_derives_the_endpoint() {
    let empty = paths::endpoint_parts(&AzureBlobConfig::default()).expect("derive the endpoint");
    assert_eq!(empty.0, AZURITE_ENDPOINT);
    assert_eq!(empty.1, "");

    let account = paths::endpoint_parts(&AzureBlobConfig::new("demoaccount")).expect("derive");
    assert_eq!(account.0, "https://demoaccount.blob.core.windows.net");
    assert_eq!(account.1, "");

    let explicit = paths::endpoint_parts(
        &AzureBlobConfig::new("demoaccount").with_endpoint("http://127.0.0.1:10000/"),
    )
    .expect("an explicit endpoint wins");
    assert_eq!(explicit.0, "http://127.0.0.1:10000");
    assert_eq!(explicit.1, "");
}

#[test]
fn an_endpoint_path_is_kept_as_a_prefix() {
    let (base, prefix) = paths::endpoint_parts(
        &AzureBlobConfig::default().with_endpoint("http://127.0.0.1:10000/devstoreaccount1/"),
    )
    .expect("a path is kept");
    assert_eq!(base, "http://127.0.0.1:10000/devstoreaccount1");
    assert_eq!(prefix, "/devstoreaccount1");

    let store = AzureBlobStore::new(
        AzureBlobConfig::default().with_endpoint("http://127.0.0.1:10000/devstoreaccount1"),
    )
    .expect("build a store");
    assert_eq!(store.root_path(), "/devstoreaccount1/");
    assert_eq!(
        store.container_path("artifacts").expect("path"),
        "/devstoreaccount1/artifacts"
    );
    assert_eq!(
        store.blob_path("artifacts", "graph/a.json").expect("path"),
        "/devstoreaccount1/artifacts/graph/a.json"
    );
}

#[test]
fn endpoints_are_validated() {
    assert_eq!(
        paths::validate_endpoint(" https://demo.blob.core.windows.net/ ").expect("valid"),
        "https://demo.blob.core.windows.net"
    );
    for endpoint in [
        "",
        "  ",
        "demo.blob.core.windows.net",
        "ftp://example.com",
        "http://",
    ] {
        let error = paths::validate_endpoint(endpoint).expect_err("must reject");
        assert_eq!(
            error.kind(),
            crate::error::ErrorKind::Config,
            "{endpoint:?}"
        );
        assert!(error.message().contains("endpoint"), "{endpoint:?}");
    }
}

#[test]
fn container_names_and_blob_keys_are_validated() {
    for container in ["", "  ", "a/b", "a?b", "a#b", "a%b", "a b"] {
        let error = paths::validate_container(container).expect_err("must reject");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
    assert!(paths::validate_container("artifacts").is_ok());

    for key in ["", "/leading", "with\u{7f}control"] {
        let error = paths::validate_key(key).expect_err("must reject");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
    assert!(paths::validate_key("graph/versions/v1/graph.json").is_ok());
}

#[test]
fn the_http_date_is_normalized_to_rfc3339() {
    assert_eq!(
        parse_http_date("Fri, 14 Nov 2026 08:49:00 GMT").as_deref(),
        Some("2026-11-14T08:49:00Z")
    );
    assert_eq!(
        parse_http_date("  Sun, 06 Nov 1994 08:49:37 GMT ").as_deref(),
        Some("1994-11-06T08:49:37Z")
    );
    assert_eq!(
        parse_http_date("06 Nov 1994 08:49:37 GMT").as_deref(),
        Some("1994-11-06T08:49:37Z"),
        "a date without a weekday is read too"
    );
    assert_eq!(
        parse_http_date("Fri, 14 Nov 2026 08:49:00").as_deref(),
        Some("2026-11-14T08:49:00Z"),
        "a missing zone is tolerated (the service always sends GMT)"
    );
    assert_eq!(
        parse_http_date("2024-01-02T03:04:05.678Z").as_deref(),
        Some("2024-01-02T03:04:05.678Z"),
        "an RFC 3339 timestamp passes through"
    );
    for malformed in [
        "",
        "  ",
        "Fri, 32 Nov 2026 08:49:00 GMT",
        "Fri, 14 Foo 2026 08:49:00 GMT",
        "Fri, 14 Nov 2026 24:49:00 GMT",
        "garbage",
    ] {
        assert_eq!(parse_http_date(malformed), None, "{malformed:?}");
    }
}

#[test]
fn the_sas_token_is_normalized() {
    assert_eq!(
        paths::normalize_sas_token("  ?sv=2021&sig=abc  ").expect("normalize"),
        "sv=2021&sig=abc"
    );
    assert_eq!(
        paths::normalize_sas_token("sv=2021&sig=abc").expect("normalize"),
        "sv=2021&sig=abc"
    );
    for blank in ["", "  ", "?"] {
        let error = paths::normalize_sas_token(blank).expect_err("must reject");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
}

#[test]
fn a_malformed_shared_key_is_a_validation_error() {
    let error =
        AzureBlobStore::new(AzureBlobConfig::default().with_shared_key("acc", "not base64!"))
            .expect_err("a non-base64 key must be rejected");
    assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    assert!(error.message().contains("key"), "{}", error.message());

    let error = AzureBlobStore::new(AzureBlobConfig::default().with_shared_key("  ", "a2V5"))
        .expect_err("a blank account must be rejected");
    assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    assert!(error.message().contains("account"), "{}", error.message());

    let error = AzureBlobStore::new(AzureBlobConfig::default().with_sas_token("  "))
        .expect_err("a blank SAS token must be rejected");
    assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    assert!(error.message().contains("token"), "{}", error.message());
}

#[test]
fn the_authorization_redacts_the_credentials() {
    let config = AzureBlobConfig::default()
        .with_shared_key("acc", "a2V5")
        .with_sas_token("sv=1&sig=secret");
    assert!(!format!("{config:?}").contains("a2V5"));
    assert!(!format!("{config:?}").contains("sig=secret"));

    let store = AzureBlobStore::new(AzureBlobConfig::default().with_shared_key("acc", "a2V5"))
        .expect("build a store");
    let debug = format!("{store:?}");
    assert!(!debug.contains("a2V5"), "{debug}");
    assert!(debug.contains("acc"), "{debug}");

    let json = serde_json::to_value(&config).expect("serialize");
    assert_eq!(json["auth"]["sas"]["token"], "sv=1&sig=secret");
}

#[test]
fn a_partial_document_keeps_every_default() {
    let config: AzureBlobConfig =
        serde_json::from_str(r#"{"account": "demo"}"#).expect("decode a partial document");
    assert_eq!(config.endpoint, "");
    assert_eq!(config.auth.as_str(), "anonymous");
    assert_eq!(config.timeout, Duration::from_secs(60));

    let full: AzureBlobConfig = serde_json::from_str(
        r#"{"account": "demo", "auth": {"shared_key": {"account": "demo", "key": "a2V5"}}}"#,
    )
    .expect("decode a full document");
    assert_eq!(full.auth.as_str(), "shared_key");
    assert_eq!(full.auth.account(), Some("demo"));
}

#[test]
fn the_authorization_names_every_mode() {
    assert_eq!(AzureBlobAuth::default().as_str(), "anonymous");
    assert_eq!(AzureBlobAuth::sas("t").as_str(), "sas");
    assert_eq!(AzureBlobAuth::shared_key("a", "k").as_str(), "shared_key");
    assert_eq!(AzureBlobAuth::default().account(), None);
    assert_eq!(AzureBlobAuth::sas("t").account(), None);
}

#[test]
fn client_credentials_are_a_typed_deferral() {
    let error = AzureBlobStore::from_client_credentials("tenant", "client", "secret")
        .expect_err("a client-credential flow must be refused");
    assert!(crate::error::is_unsupported(&error));
    assert!(error.message().contains("client"), "{}", error.message());
    assert!(error.message().contains("SAS"), "{}", error.message());
    assert!(!error.message().contains("secret"), "{}", error.message());
}

/// The string to sign of the Shared Key layout for a fixed request, byte for byte.
#[test]
fn the_shared_key_string_to_sign_matches_the_documented_layout() {
    let mut headers = BTreeMap::new();
    headers.insert("x-ms-version".to_string(), API_VERSION.to_string());
    headers.insert(
        "x-ms-date".to_string(),
        "Fri, 14 Nov 2026 08:49:00 GMT".to_string(),
    );
    headers.insert("x-ms-blob-type".to_string(), "BlockBlob".to_string());
    headers.insert("content-length".to_string(), "7".to_string());
    headers.insert("content-type".to_string(), "ignored".to_string());
    let query = vec![
        ("comp".to_string(), "list".to_string()),
        ("restype".to_string(), "container".to_string()),
    ];
    let string = signing::string_to_sign(
        Verb::Put,
        Some(7),
        &headers,
        &query,
        "/artifacts/graph/current.json",
        "demoaccount",
    );
    let expected = concat!(
        "PUT\n",
        "\n",  // Content-Encoding
        "\n",  // Content-Language
        "7\n", // Content-Length
        "\n",  // Content-MD5
        "\n",  // Content-Type
        "\n",  // Date
        "\n",  // If-Modified-Since
        "\n",  // If-Match
        "\n",  // If-None-Match
        "\n",  // If-Unmodified-Since
        "\n",  // Range
        "x-ms-blob-type:BlockBlob\n",
        "x-ms-date:Fri, 14 Nov 2026 08:49:00 GMT\n",
        "x-ms-version:2021-12-02\n",
        "/demoaccount/artifacts/graph/current.json\n",
        "comp:list\n",
        "restype:container",
    );
    assert_eq!(string, expected);
}

#[test]
fn an_empty_content_length_and_no_query_reduce_to_the_bare_layout() {
    let mut headers = BTreeMap::new();
    headers.insert("x-ms-version".to_string(), API_VERSION.to_string());
    headers.insert(
        "x-ms-date".to_string(),
        "Fri, 14 Nov 2026 08:49:00 GMT".to_string(),
    );
    let string = signing::string_to_sign(Verb::Get, None, &headers, &[], "/", "demoaccount");
    let expected = concat!(
        "GET\n\n\n\n\n\n\n\n\n\n\n\n", // twelve empty fields after the verb
        "x-ms-date:Fri, 14 Nov 2026 08:49:00 GMT\n",
        "x-ms-version:2021-12-02\n",
        "/demoaccount/",
    );
    assert_eq!(string, expected);
    assert!(
        !string.contains("\n\n/demoaccount/"),
        "the resource follows the last canonicalized header directly"
    );

    headers.insert("x-ms-meta-b".to_string(), "  two  ".to_string());
    headers.insert("x-ms-meta-a".to_string(), "one\ntwo".to_string());
    let string = signing::string_to_sign(Verb::Get, Some(0), &headers, &[], "/x", "acc");
    let canonical = string.split("/acc/x").next().unwrap_or_default();
    assert!(
        canonical.contains("x-ms-meta-a:onetwo\nx-ms-meta-b:two\n"),
        "x-ms headers are sorted, trimmed and stripped of line breaks: {canonical:?}"
    );
}

#[test]
fn an_unknown_entity_is_left_alone_and_known_ones_are_replaced() {
    assert_eq!(
        xml::unescape("a &amp; b &lt;c&gt; &quot;d&quot; &apos;e&apos;"),
        "a & b <c> \"d\" 'e'"
    );
    assert_eq!(xml::unescape("&unknown; &"), "&unknown; &");
}

#[test]
fn listing_entries_are_parsed_from_the_blob_xml() {
    let xml = concat!(
        "<EnumerationResults><Blobs>",
        "<Blob><Name>graph/versions/v1/graph.json</Name><Properties>",
        "<Content-Length>7</Content-Length>",
        "<Last-Modified>Fri, 14 Nov 2026 08:49:00 GMT</Last-Modified>",
        "</Properties></Blob>",
        "<Blob><Name>a&amp;b.json</Name><Properties>",
        "<Content-Length>3</Content-Length>",
        "</Properties></Blob>",
        "<Blob><Properties><Content-Length>1</Content-Length></Properties></Blob>",
        "</Blobs><NextMarker>1!76!</NextMarker></EnumerationResults>",
    );
    let mut entries = Vec::new();
    append_entries(&mut entries, xml);
    assert_eq!(entries.len(), 2, "a blob without a name is skipped");
    assert_eq!(entries[0].key, "graph/versions/v1/graph.json");
    assert_eq!(entries[0].size, 7);
    assert_eq!(entries[1].key, "a&b.json");
    assert_eq!(entries[1].size, 3);
    assert_eq!(
        xml::next_marker(xml, None).as_deref(),
        Some("1!76!"),
        "the page marker is read"
    );
    assert_eq!(
        xml::next_marker(xml, Some("1!76!")),
        None,
        "a repeated marker ends the pagination"
    );
    assert_eq!(
        xml::next_marker(
            "<EnumerationResults><NextMarker /></EnumerationResults>",
            None
        ),
        None,
        "an empty marker ends the pagination"
    );
}

#[test]
fn container_entries_are_parsed_from_the_listing_xml() {
    let xml = concat!(
        "<EnumerationResults><Containers>",
        "<Container><Name>alpha</Name><Properties><Last-Modified>",
        "Fri, 14 Nov 2026 08:49:00 GMT</Last-Modified></Properties></Container>",
        "<Container><Name>zeta</Name></Container>",
        "</Containers><NextMarker>2!88!</NextMarker></EnumerationResults>",
    );
    let names: Vec<String> = xml::tag_texts(xml, "Container")
        .into_iter()
        .filter_map(|block| xml::first_tag(&block, "Name"))
        .collect();
    assert_eq!(names, vec!["alpha".to_string(), "zeta".to_string()]);
}

#[test]
fn object_metadata_falls_back_for_a_missing_timestamp() {
    let with_date = meta("k", 4, parse_http_date("Fri, 14 Nov 2026 08:49:00 GMT"));
    assert_eq!(with_date.key, "k");
    assert_eq!(with_date.size, 4);
    // 2026-11-14T08:49:00Z
    assert_eq!(with_date.updated_at.timestamp(), 1_794_646_140);

    let before = now_utc();
    let bare = meta("k", 5, None);
    assert_eq!(bare.size, 5);
    assert!(bare.updated_at >= before);

    let malformed = meta("k", 6, parse_http_date("Fri, 99 Nov 2026 08:49:00 GMT"));
    assert!(malformed.updated_at >= before);
}
