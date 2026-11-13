use super::*;

/// A minimal `png` payload of exactly `size` bytes (at least 8).
fn png(size: usize) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.resize(size.max(bytes.len()), 0);
    bytes
}

/// A minimal `jpeg` payload of exactly `size` bytes (at least 3).
fn jpeg(size: usize) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8, 0xFF];
    bytes.resize(size.max(bytes.len()), 0);
    bytes
}

/// A candidate carrying a downloaded payload.
fn downloaded(url: &str, bytes: Vec<u8>) -> ImageCandidate {
    ImageCandidate::new(url, "").with_bytes(bytes)
}

/// A reference-only candidate without a payload.
fn reference(url: &str) -> ImageCandidate {
    ImageCandidate::new(url, "")
}

/// An anchor with `href` and `text`.
fn link(href: &str, text: &str) -> Link {
    Link {
        href: href.to_string(),
        text: text.to_string(),
    }
}

/// The URLs of `images`, in order.
fn urls(images: &[ImageCandidate]) -> Vec<&str> {
    images.iter().map(|image| image.url.as_str()).collect()
}

#[test]
fn sniff_image_detects_binary_formats() {
    assert_eq!(sniff_image(b"\x89PNG\r\n\x1a\npayload"), Some("png"));
    assert_eq!(sniff_image(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpeg"));
    assert_eq!(sniff_image(b"GIF89a...."), Some("gif"));
    assert_eq!(sniff_image(b"GIF87a...."), Some("gif"));
    assert_eq!(sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 "), Some("webp"));
}

#[test]
fn sniff_image_detects_svg_documents() {
    let declared = br#"<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://example.com"/>"#;
    let with_bom = b"\xEF\xBB\xBF<svg xmlns=\"http://example.com\"/>";
    assert_eq!(sniff_image(declared), Some("svg"));
    assert_eq!(sniff_image(with_bom), Some("svg"));
    assert_eq!(sniff_image(b"\n\n  <SVG width=\"10\"></SVG>"), Some("svg"));
    assert_eq!(sniff_image(b"<!-- note --><svg></svg>"), Some("svg"));
    assert_eq!(
        sniff_image(b"<?xml version=\"1.0\"?><!DOCTYPE svg><svg></svg>"),
        Some("svg")
    );
}

#[test]
fn sniff_image_rejects_other_payloads() {
    for payload in [
        b"".as_slice(),
        &[0xFF, 0xD8],
        b"\x89P",
        b"RIFF\x00\x00\x00\x00WAVEfmt ",
        b"RIFF\x00\x00\x00\x00WEB",
        b"hello world",
        b"<?xml version=\"1.0\"?><html></html>",
        b"<!-- unterminated <svg",
    ] {
        assert_eq!(sniff_image(payload), None, "{payload:?}");
    }
}

#[test]
fn icon_heuristics_by_dimension() {
    let url = "https://example.com/assets/chart.png";
    assert!(is_probable_icon(url, Some(2), Some(2)));
    assert!(is_probable_icon(url, Some(1), Some(1)));
    assert!(is_probable_icon(url, Some(2), Some(8)));
    assert!(is_probable_icon(url, Some(100), Some(1)));
    assert!(!is_probable_icon(url, Some(80), Some(80)));
    assert!(!is_probable_icon(url, None, None));
}

#[test]
fn icon_heuristics_by_url() {
    for url in [
        "https://example.com/spacer.gif",
        "https://example.com/1x1.png",
        "https://example.com/img/pixel.gif",
        "https://example.com/blank.gif",
        "https://example.com/transparent.gif",
        "https://example.com/SPACER.GIF",
    ] {
        assert!(is_probable_icon(url, None, None), "{url}");
    }
    assert!(!is_probable_icon(
        "https://example.com/images/pivot-chart.png",
        None,
        None
    ));
}

#[test]
fn filter_drops_non_http_urls() {
    let images = [
        downloaded("/relative.png", png(2048)),
        downloaded("ftp://example.com/image.png", png(2048)),
        downloaded("data:image/png;base64,AAAA", png(2048)),
        downloaded("https://example.com/ok.png", png(2048)),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        ["https://example.com/ok.png"]
    );
}

#[test]
fn filter_drops_icons() {
    let images = [
        downloaded("https://example.com/spacer.gif", png(2048)),
        downloaded("https://example.com/1x1.png", png(2048)),
        downloaded("https://example.com/screen.png", png(2048)),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        ["https://example.com/screen.png"]
    );
}

#[test]
fn filter_drops_unknown_binary_formats() {
    let images = [
        downloaded(
            "https://example.com/bitmap.bmp",
            b"BM\x00\x00\x00\x00".repeat(256),
        ),
        downloaded("https://example.com/text.bin", b"not an image".repeat(128)),
    ];
    assert_eq!(filter_images(&images), Vec::<ImageCandidate>::new());
}

#[test]
fn filter_respects_size_bounds() {
    let filter = ResourceFilter::default()
        .with_min_image_bytes(8)
        .with_max_image_bytes(16);
    let images = [
        downloaded("https://example.com/small.png", jpeg(3)),
        downloaded("https://example.com/min.png", png(8)),
        downloaded("https://example.com/max.png", png(16)),
        downloaded("https://example.com/big.png", png(17)),
    ];
    let kept = filter_images_with(&images, &filter);
    assert_eq!(
        urls(&kept),
        ["https://example.com/min.png", "https://example.com/max.png"]
    );
}

#[test]
fn filter_uses_default_size_floor() {
    let images = [
        downloaded("https://example.com/tiny.png", png(512)),
        downloaded("https://example.com/ok.png", png(2048)),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        ["https://example.com/ok.png"]
    );
}

#[test]
fn filter_keeps_valid_png_and_jpeg() {
    let images = [
        downloaded("https://example.com/a.png", png(2048)),
        downloaded("https://example.com/b.jpg", jpeg(4096)),
        reference("https://example.com/c.gif").with_content_type("image/gif"),
        reference("https://example.com/d.png").with_content_type("image/png"),
        reference("https://example.com/e.png"),
    ];
    assert_eq!(filter_images(&images).len(), 5);
}

#[test]
fn keep_policy_accepts_only_raster_formats() {
    for format in ["png", "jpeg", "gif", "webp"] {
        assert!(is_supported_image_format(format), "{format}");
    }
    for format in ["svg", "bmp", "tiff", "avif", ""] {
        assert!(!is_supported_image_format(format), "{format}");
    }
    assert_eq!(SUPPORTED_EXTS, ["png", "jpg", "jpeg", "webp", "gif"]);
    assert_eq!(
        SUPPORTED_MIMES,
        ["image/jpeg", "image/png", "image/gif", "image/webp"]
    );
}

#[test]
fn filter_rejects_svg_payloads_and_declarations() {
    let images = [
        // SVG bytes without a declaration: classified by `sniff_image`, then
        // rejected by the keep policy.
        downloaded(
            "https://example.com/drawing.svg",
            b"<?xml version=\"1.0\"?><svg xmlns=\"http://example.com\"/>".repeat(64),
        ),
        // SVG reference with an SVG media type.
        reference("https://example.com/logo.svg").with_content_type("image/svg+xml; charset=utf-8"),
        // A real PNG payload declared as SVG is rejected: Python's gate is
        // the declared media type, `SUPPORTED_MIMES` has no SVG entry.
        downloaded("https://example.com/renamed.png", png(2048)).with_content_type("image/svg+xml"),
        downloaded("https://example.com/bitmap.bmp", png(2048)).with_content_type("image/bmp"),
        // A supported raster still passes.
        downloaded("https://example.com/ok.png", png(2048)).with_content_type("image/png"),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        ["https://example.com/ok.png"]
    );
}

#[test]
fn filter_drops_non_image_content_types_without_bytes() {
    let images = [
        reference("https://example.com/page.html").with_content_type("text/html"),
        reference("https://example.com/other.png"),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        ["https://example.com/other.png"]
    );
}

#[test]
fn filter_falls_back_to_content_type_for_bytes() {
    let payload = b"definitely not sniffed".repeat(128);
    let images = [
        downloaded("https://example.com/a.png", payload.clone()).with_content_type("image/png"),
        downloaded("https://example.com/b.png", payload).with_content_type("image/bmp"),
    ];
    assert_eq!(urls(&filter_images(&images)), ["https://example.com/a.png"]);
}

#[test]
fn image_extension_prefers_the_url_suffix() {
    assert_eq!(image_extension("https://example.com/a.PNG", None), "png");
    assert_eq!(image_extension("https://example.com/a.jpeg", None), "jpeg");
    assert_eq!(
        image_extension("https://example.com/a.jpg", Some("image/png")),
        "jpg"
    );
    assert_eq!(
        image_extension("https://example.com/a.png?x=1", Some("image/gif")),
        "png"
    );
    assert_eq!(
        image_extension("https://example.com/a.webp#frag", None),
        "webp"
    );
}

#[test]
fn image_extension_falls_back_to_the_media_type_then_png() {
    assert_eq!(
        image_extension("https://example.com/picture", Some("image/webp")),
        "webp"
    );
    assert_eq!(
        image_extension(
            "https://example.com/picture.avif",
            Some("image/jpeg; charset=binary")
        ),
        "jpg"
    );
    assert_eq!(
        image_extension("https://example.com/picture.blob", Some("IMAGE/GIF")),
        "gif"
    );
    assert_eq!(image_extension("https://example.com/picture", None), "png");
    assert_eq!(
        image_extension("https://example.com/picture.svg", Some("image/svg+xml")),
        "png"
    );
    assert_eq!(image_extension("/relative/.webp", None), "png");
}

#[test]
fn payload_dedupe_key_is_the_sha256_hex_digest() {
    assert_eq!(
        DedupeKey::Content(sha256::hex_digest(b"abc")),
        DedupeKey::Content(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string()
        )
    );
    assert_eq!(sha256::hex_digest(b"").len(), 64);
}

#[test]
fn filter_dedupes_identical_bytes() {
    let payload = png(2048);
    let images = [
        downloaded("https://example.com/first.png", payload.clone()),
        downloaded("https://example.com/second.png", payload),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        ["https://example.com/first.png"]
    );
}

#[test]
fn filter_dedupes_normalized_urls() {
    let images = [
        reference("https://Example.com/chart/#top"),
        reference("https://example.com/chart/"),
        reference("https://example.com/chart"),
        reference("https://example.com/chart?page=2"),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        [
            "https://Example.com/chart/#top",
            "https://example.com/chart?page=2"
        ]
    );
}

#[test]
fn filter_preserves_first_seen_order() {
    let images = [
        downloaded("https://example.com/a.png", png(2048)),
        downloaded("https://example.com/icon/2x2.png", png(2048)),
        downloaded("https://example.com/b.jpg", jpeg(2048)),
        downloaded("https://example.com/a-copy.png", png(2048)),
        downloaded("https://example.com/c.gif", b"GIF89a".repeat(512)),
    ];
    assert_eq!(
        urls(&filter_images(&images)),
        [
            "https://example.com/a.png",
            "https://example.com/b.jpg",
            "https://example.com/c.gif"
        ]
    );
}

#[test]
fn filter_is_idempotent() {
    let images = [
        downloaded("https://example.com/a.png", png(2048)),
        downloaded("https://example.com/b.jpg", jpeg(2048)),
        reference("https://example.com/c.png"),
    ];
    let once = filter_images(&images);
    assert_eq!(filter_images(&once), once);
}

#[test]
fn dedupe_links_collapses_equivalent_hrefs() {
    let links = [
        link("https://Example.COM/a/#section", "one"),
        link("https://example.com/a/", "two"),
        link("https://example.com/a", "three"),
        link("https://example.com/a?page=2", "four"),
        link("https://example.com/a/?page=2", "five"),
    ];
    let kept = dedupe_links(&links);
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0], link("https://Example.COM/a/#section", "one"));
    assert_eq!(kept[1], link("https://example.com/a?page=2", "four"));
}

#[test]
fn dedupe_links_drops_empty_and_preserves_order() {
    let links = [
        link("", "empty"),
        link("   ", "blank"),
        link("#anchor", "fragment"),
        link("/guide", "relative"),
        link("/guide/", "relative slash"),
        link("https://example.com/later", "later"),
    ];
    let kept = dedupe_links(&links);
    assert_eq!(
        kept,
        [
            link("/guide", "relative"),
            link("https://example.com/later", "later")
        ]
    );
}

#[test]
fn slugify_matrix() {
    assert_eq!(slugify("Create a PivotChart!"), "create-a-pivotchart");
    assert_eq!(slugify("创建图表"), "");
    assert_eq!(slugify("!!!"), "");
    assert_eq!(slugify(""), "");
    assert_eq!(slugify("  --Hello   World--  "), "hello-world");
    assert_eq!(slugify("_foo_bar_"), "foo-bar");
    assert_eq!(slugify("123 guide"), "123-guide");
    assert_eq!(slugify("A/B.C"), "a-b-c");
}

#[test]
fn slugify_caps_length_without_trailing_separator() {
    let slug = slugify(&"word ".repeat(40));
    assert!(slug.len() <= 60, "{slug}");
    assert!(!slug.ends_with('-'), "{slug}");
    assert_eq!(slug.matches('-').count(), 11);
    let dense = slugify(&"a".repeat(200));
    assert_eq!(dense.len(), 60);
    assert!(!dense.ends_with('-'));
}

#[test]
fn slugify_output_is_a_valid_skill_name() {
    for input in ["Create a PivotChart!", "A/B.C", "123 guide", "foo__bar"] {
        let slug = slugify(input);
        assert_ne!(slug, "");
        assert!(slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
        assert!(slug
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric()));
    }
}

#[test]
fn content_hash_matches_fnv1a_vectors() {
    assert_eq!(content_hash(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(content_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(content_hash(b"foobar"), 0x8594_4171_f739_67e8);
    assert_eq!(content_hash(b"payload"), content_hash(b"payload"));
    assert_ne!(content_hash(b"one"), content_hash(b"two"));
}

#[test]
fn host_extraction() {
    assert_eq!(
        host("https://Example.COM:8443/a/b?c=1"),
        Some("example.com:8443".to_string())
    );
    assert_eq!(host("http://example.com"), Some("example.com".to_string()));
    assert_eq!(
        host("http://user:secret@Example.com/a"),
        Some("example.com".to_string())
    );
    assert_eq!(host("/relative/path"), None);
    assert_eq!(host("mailto:someone@example.com"), None);
    assert_eq!(host("ftp://example.com/file"), None);
    assert_eq!(host("http:///no-authority"), None);
    assert_eq!(host("not a url"), None);
    assert_eq!(host(""), None);
}

#[test]
fn same_host_comparison() {
    assert!(same_host("https://Example.com/a", "http://example.COM/b"));
    assert!(same_host(
        "http://example.com:8080/a",
        "http://example.com:8080/b"
    ));
    assert!(!same_host(
        "http://example.com:8080",
        "http://example.com:9090"
    ));
    assert!(!same_host("http://www.example.com", "http://example.com"));
    assert!(!same_host("/relative", "/relative"));
    assert!(!same_host("mailto:a@example.com", "http://example.com"));
}

#[test]
fn resource_filter_defaults_and_validate() {
    let filter = ResourceFilter::default();
    assert_eq!(filter.min_image_bytes, 1024);
    assert_eq!(filter.max_image_bytes, 8 * 1024 * 1024);
    assert!(!filter.allow_cross_host);
    assert!(filter.validate().is_ok());
    let error = ResourceFilter::default()
        .with_min_image_bytes(0)
        .validate()
        .unwrap_err();
    assert!(error.to_string().contains("min_image_bytes"), "{error}");
    let error = ResourceFilter::default()
        .with_max_image_bytes(0)
        .validate()
        .unwrap_err();
    assert!(error.to_string().contains("max_image_bytes"), "{error}");
    let inverted = ResourceFilter::default().with_min_image_bytes(4096);
    let error = inverted.with_max_image_bytes(1024).validate().unwrap_err();
    assert!(error.to_string().contains("4096"), "{error}");
    assert!(error.to_string().contains("1024"), "{error}");
}

#[test]
fn resource_filter_serde_defaults_and_round_trip() {
    let parsed: ResourceFilter = serde_json::from_str("{}").unwrap();
    assert_eq!(parsed, ResourceFilter::default());
    let filter = ResourceFilter::default()
        .with_min_image_bytes(16)
        .with_allow_cross_host(true);
    let json = serde_json::to_string(&filter).unwrap();
    assert_eq!(
        serde_json::from_str::<ResourceFilter>(&json).unwrap(),
        filter
    );
    let candidate = reference("https://example.com/a.png")
        .with_bytes(vec![1, 2, 3])
        .with_content_type("image/png");
    let json = serde_json::to_string(&candidate).unwrap();
    assert_eq!(
        serde_json::from_str::<ImageCandidate>(&json).unwrap(),
        candidate
    );
}
