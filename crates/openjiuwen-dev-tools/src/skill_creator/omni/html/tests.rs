//! Tests for the rule-based HTML extraction.
//!
//! Every case runs offline on inline HTML strings: no network, no files, and no
//! wall-clock dependence.

use super::*;

const PAGE: &str = r#"<!doctype html>
<html><head>
<title>  My   Page </title>
<meta name="description" content="A &amp; B page">
</head>
<body>
<h1>Welcome</h1>
<h2>Features &amp; More</h2>
<a href="/docs">Docs</a>
<a href="https://example.com/x?y=1#frag">External</a>
<img src="/img/a.png" alt="Alpha">
<img alt="Beta" data-src="/img/b.png">
<pre><code>let x = 1;
fn main() {}</code></pre>
<p>Inline <code>cargo test</code> runs.</p>
</body></html>
"#;

fn heading_pairs(document: &HtmlDocument) -> Vec<(u8, &str)> {
    document
        .headings
        .iter()
        .map(|h| (h.level, h.text.as_str()))
        .collect()
}

fn link_pairs(document: &HtmlDocument) -> Vec<(&str, &str)> {
    document
        .links
        .iter()
        .map(|l| (l.href.as_str(), l.text.as_str()))
        .collect()
}

fn image_pairs(document: &HtmlDocument) -> Vec<(&str, &str)> {
    document
        .images
        .iter()
        .map(|i| (i.src.as_str(), i.alt.as_str()))
        .collect()
}

fn check(base: &str, href: &str, expected: Option<&str>) {
    assert_eq!(
        resolve_url(base, href).as_deref(),
        expected,
        "{base} + {href}"
    );
}

#[test]
fn parse_extracts_every_collection() {
    let document = HtmlDocument::parse(PAGE, "https://example.com");
    assert_eq!(document.title, "My Page");
    assert_eq!(document.description, "A & B page");
    assert_eq!(
        heading_pairs(&document),
        vec![(1, "Welcome"), (2, "Features & More")]
    );
    assert_eq!(
        link_pairs(&document),
        vec![
            ("/docs", "Docs"),
            ("https://example.com/x?y=1#frag", "External")
        ]
    );
    assert_eq!(
        image_pairs(&document),
        vec![("/img/a.png", "Alpha"), ("/img/b.png", "Beta")]
    );
    assert_eq!(
        document.code_blocks,
        vec!["let x = 1;\nfn main() {}", "cargo test"]
    );
    assert!(document.text.contains("Welcome"));
    assert!(document.text.contains("Inline cargo test runs."));
    assert!(!document.text.contains('<'));
}

#[test]
fn parse_decodes_entities_exactly_once() {
    let html = "<p>&amp; &lt; &gt; &quot; &#39; &apos; &nbsp; &mdash; &ndash; \
                    &hellip; &copy; &#65; &#x42; &amp;lt; &unknown; &#xZZ; &#999999999;</p>";
    let text = HtmlDocument::parse(html, "").text;
    assert!(text.starts_with("& < > \" ' ' "), "{text}");
    assert!(text.contains('\u{2014}') && text.contains('\u{2013}'));
    assert!(text.contains('\u{2026}') && text.contains('\u{a9}') && text.contains("A B"));
    for literal in ["&lt;", "&unknown;", "&#xZZ;", "&#999999999;"] {
        assert!(text.contains(literal), "{literal} missing from {text}");
    }
    assert!(!text.contains("&amp;lt;"));
}

#[test]
fn parse_strips_raw_text_elements_and_comments() {
    let html = r#"<html><head><style>body { color: red; }</style>
<script>var secret = "leak";</script></head><body><!-- hidden comment -->
<noscript>enable js</noscript><svg><path d="M0 0"/></svg>
<h1>Visible</h1><a href="/x">Link</a><pre>keep</pre></body></html>"#;
    let document = HtmlDocument::parse(html, "");
    for leaked in [
        "color",
        "secret",
        "leak",
        "hidden comment",
        "enable js",
        "M0 0",
    ] {
        assert!(!document.text.contains(leaked), "{leaked} leaked into text");
    }
    assert_eq!(heading_pairs(&document), vec![(1, "Visible")]);
    assert_eq!(link_pairs(&document), vec![("/x", "Link")]);
    assert_eq!(document.code_blocks, vec!["keep"]);
}

#[test]
fn parse_is_attribute_order_independent() {
    let first = HtmlDocument::parse(
        r#"<meta name="description" content="First"><img src="/a.png" alt="A"><a href="/x">X</a>"#,
        "",
    );
    let second = HtmlDocument::parse(
        r#"<meta content="First" name="description"><img alt="A" src="/a.png"><a target="" href="/x">X</a>"#,
        "",
    );
    assert_eq!(first.description, "First");
    assert_eq!(first.description, second.description);
    assert_eq!(first.images, second.images);
    assert_eq!(first.links, second.links);
}

/// A page of many unclosed anchors used to rescan to the end of the document
/// for every one of them: 20 000 anchors cost 2·10⁸ token comparisons, so the
/// parse took seconds on a page a scrape stage then blocked its worker on.
#[test]
fn many_unterminated_anchors_stay_linear() {
    const COUNT: usize = 20_000;
    let html = "<a href=target>".repeat(COUNT);
    let started = std::time::Instant::now();
    let document = HtmlDocument::parse(&html, "");
    let elapsed = started.elapsed();

    assert_eq!(document.links.len(), COUNT);
    assert!(
        document
            .links
            .iter()
            .all(|link| link.href == "target" && link.text.is_empty()),
        "an unclosed anchor keeps its href and an empty text"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "extracting {COUNT} unclosed anchors took {elapsed:?}"
    );
}

#[test]
fn close_matching_pairs_the_innermost_open_tag() {
    // Nested same-name tags pair innermost-first, so the outer heading is
    // unclosed and only the inner one is recorded.
    let nested = HtmlDocument::parse("<h1>outer<h1>inner</h1> tail", "");
    assert_eq!(heading_pairs(&nested), vec![(1, "inner")]);
    // A stray close tag is ignored, and a following anchor still pairs.
    let stray = HtmlDocument::parse("</a><a href=x>ok</a>", "");
    assert_eq!(link_pairs(&stray), vec![("x", "ok")]);
    // Nested anchors: the inner one keeps its text, the unclosed outer one does not.
    let anchors = HtmlDocument::parse("<a href=x>first<a href=y>second</a>", "");
    assert_eq!(link_pairs(&anchors), vec![("x", ""), ("y", "second")]);
}

#[test]
fn parse_survives_unterminated_markup() {
    for html in ["<p>before</p><script>var x = 1;", "<a href=>Click</a>"] {
        let document = HtmlDocument::parse(html, "");
        assert_eq!(document.links.len(), 0, "{html}");
    }
    assert_eq!(
        HtmlDocument::parse("<p>before</p><script>var x = 1;", "").text,
        "before"
    );
    assert_eq!(HtmlDocument::parse("<a href=>Click</a>", "").text, "Click");
    let unquoted = HtmlDocument::parse("<a href=unquoted>Bare</a>", "");
    assert_eq!(link_pairs(&unquoted), vec![("unquoted", "Bare")]);
    let document = HtmlDocument::parse("<h1>One</h1><h2>Two", "");
    assert_eq!(heading_pairs(&document), vec![(1, "One")]);
    assert!(document.text.contains("Two"));
    assert_eq!(
        image_pairs(&HtmlDocument::parse("<img src='x>", "")),
        vec![("x>", "")]
    );
}

#[test]
fn parse_handles_empty_and_plain_text() {
    assert_eq!(
        HtmlDocument::parse("", "https://example.com"),
        HtmlDocument::default()
    );
    let document = HtmlDocument::parse("Just some plain text.", "");
    assert_eq!(document.text, "Just some plain text.");
    let collection_lengths = (
        document.title.len(),
        document.description.len(),
        document.headings.len(),
        document.links.len(),
        document.images.len(),
        document.code_blocks.len(),
    );
    assert_eq!(collection_lengths, (0, 0, 0, 0, 0, 0));
}

#[test]
fn text_is_collapsed_and_free_of_markup() {
    let html = r#"<div class="wrapper" data-id="42">
        <p>Hello
           <b>bold</b>   world</p>
        <br><p>Second</p></div>"#;
    let document = HtmlDocument::parse(html, "");
    assert_eq!(document.text, "Hello bold world Second");
    for leaked in ["wrapper", "42", "div", "class"] {
        assert!(!document.text.contains(leaked), "{leaked} leaked into text");
    }
}

#[test]
fn pre_wins_over_nested_code() {
    let html = "<pre><code>inner</code></pre><code>standalone</code>";
    assert_eq!(
        HtmlDocument::parse(html, "").code_blocks,
        vec!["inner", "standalone"]
    );
}

#[test]
fn parse_is_deterministic() {
    let first = HtmlDocument::parse(PAGE, "https://example.com");
    assert_eq!(first, HtmlDocument::parse(PAGE, "https://example.com"));
    let hrefs: Vec<&str> = first.links.iter().map(|link| link.href.as_str()).collect();
    assert_eq!(hrefs, vec!["/docs", "https://example.com/x?y=1#frag"]);
}

#[test]
fn html_document_round_trips_json() {
    let document = HtmlDocument::parse(PAGE, "https://example.com");
    let json = serde_json::to_string(&document).unwrap();
    let restored: HtmlDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(document, restored);
    let partial: HtmlDocument = serde_json::from_str(r#"{"title":"T","unknown":1}"#).unwrap();
    assert_eq!(partial.title, "T");
    assert_eq!(partial.headings.len(), 0);
}

#[test]
fn resolve_url_matrix() {
    let base = "https://example.com/dir/sub/page.html?x=1";
    // Absolute and protocol-relative targets keep their query and lose their fragment.
    check(
        base,
        "https://other.org/x?q=1#f",
        Some("https://other.org/x?q=1"),
    );
    check(
        "http://example.com/a",
        "//cdn.example.com/lib.js",
        Some("http://cdn.example.com/lib.js"),
    );
    check(
        "https://example.com",
        "http://plain.org/",
        Some("http://plain.org/"),
    );
    // Root-relative and relative paths resolve against the base directory.
    check(base, "/root/x", Some("https://example.com/root/x"));
    check(
        base,
        "img/a.png",
        Some("https://example.com/dir/sub/img/a.png"),
    );
    check(base, "./x", Some("https://example.com/dir/sub/x"));
    check(base, "sub/", Some("https://example.com/dir/sub/sub/"));
    check(
        "https://example.com",
        "a/b",
        Some("https://example.com/a/b"),
    );
    // Dot segments are normalized and never climb above the root.
    check(base, "../a/b", Some("https://example.com/dir/a/b"));
    check(base, "../../../x", Some("https://example.com/x"));
    check(
        base,
        "../../../../../../../x",
        Some("https://example.com/x"),
    );
    check(base, "/a/./b/../c", Some("https://example.com/a/c"));
    // Queries survive resolution; fragments do not.
    check(
        base,
        "next?q=1&r=2#frag",
        Some("https://example.com/dir/sub/next?q=1&r=2"),
    );
    check(
        base,
        "https://example.com/x#frag",
        Some("https://example.com/x"),
    );
    check(
        base,
        "?only=query",
        Some("https://example.com/dir/sub/page.html?only=query"),
    );
    // Empty, fragment-only and non-http(s) references are rejected, as are non-http bases.
    for href in [
        "",
        "   ",
        "#frag",
        "#",
        "mailto:a@b.c",
        "javascript:void(0)",
        "data:text/plain,hi",
        "tel:+123",
        "ftp://example.com/x",
    ] {
        check(base, href, None);
    }
    check("ftp://example.com/x", "a", None);
    check("example.com/x", "a", None);
    check("", "a", None);
}
