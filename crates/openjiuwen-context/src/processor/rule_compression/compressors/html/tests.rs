//! Extraction and compressor tests.
use super::*;
#[test]
fn noise_is_dropped_and_structure_is_kept() {
    let html = "<!doctype html><html><head><title>Payment guide</title>\
        <style>body { color: red; }</style><script>alert('x')</script></head>\
        <body><nav>menu menu</nav><h1>Payment guide</h1>\
        <p>Payments are <a href=\"https://example.test/pay\">settled</a> \
        when the error clears.</p><pre>let x = 1;</pre>\
        <ul><li>first</li><li>second</li></ul></body></html>";
    let result = HTMLExtractor::default().extract(html);
    assert_eq!(result.title.as_deref(), Some("Payment guide"));
    assert!(!result.extracted.contains("alert"));
    assert!(!result.extracted.contains("color: red"));
    assert!(!result.extracted.contains("menu menu"));
    assert!(result.extracted.starts_with("# Payment guide"));
    assert!(result
        .extracted
        .contains("[settled](https://example.test/pay)"));
    assert!(result.extracted.contains("```\nlet x = 1;\n```"));
    assert!(result.extracted.contains("- first"));
    assert_eq!(result.link_count, 1);
    assert_eq!(result.code_block_count, 1);
}
#[test]
fn entities_and_tables_are_decoded() {
    let html = "<table><tr><th>name</th><th>qty</th></tr>\
        <tr><td>a &amp; b</td><td>2</td></tr></table>";
    let result = HTMLExtractor::default().extract(html);
    assert!(result.extracted.contains("| name | qty |"));
    assert!(result.extracted.contains("| a & b | 2 |"));
}
#[test]
fn tiny_pages_are_declined_by_the_compressor() {
    let html = "<p>tiny</p>";
    let result = HtmlCompressor::default().compress(html, &RuleContext::new(100));
    assert!(!result.modified);
    assert!(result.details.is_some());
}

/// A page whose text lower-cases to a *different byte length*.
///
/// `İ` (U+0130, two bytes) lower-cases to `i̇` (three bytes) and `ẞ` (U+1E9E,
/// three bytes) to `ß` (two bytes), so a lower-cased copy of this document holds
/// every later offset one byte off. A tag scan that measured offsets there and
/// sliced the original used to land on the wrong bytes, or on no char boundary at
/// all and panic.
fn wide_lowercase_page() -> &'static str {
    "<html><head><title>ẞ guide</title>\
     <meta name=\"author\" content=\"ẞ İlker\">\
     <script>alert('İ')</script><style>p { color: red }</style></head>\
     <body><h1>İstanbul ẞ guide</h1>\
     <p>İstanbul is a city where ẞ, ﬁ and 🚀 appear beside 中文 and plain Latin \
     text, so the extractor keeps every one of them instead of slicing a \
     character in half. Read <a title=\"İstanbul ẞ 🚀\" \
     href=\"https://example.test/İstanbul\">the İstanbul guide</a> for the ferry \
     schedule, the market hours and the opening times of every museum.</p>\
     </body></html>"
}

#[test]
fn non_ascii_text_never_shifts_a_tag_offset() {
    let result = HTMLExtractor::default().extract(wide_lowercase_page());
    assert_eq!(result.title.as_deref(), Some("ẞ guide"));
    // `ẞ` in the title shrinks in a lower-cased copy, so this `<meta` was read at
    // the wrong offset before the scan moved onto the original bytes.
    assert_eq!(result.author.as_deref(), Some("ẞ İlker"));
    // (1) A shifted `</script>` offset used to delete around the subtree instead
    // of over it, leaving the script body behind.
    assert!(!result.extracted.contains("alert"));
    assert!(!result.extracted.contains("color: red"));
    // (2) Every non-ASCII run survives intact: `İ`, `ẞ`, an `ﬁ` ligature, CJK and
    // an astral-plane emoji.
    assert!(result
        .extracted
        .contains("İstanbul is a city where ẞ, ﬁ and 🚀"));
    assert!(result.extracted.contains("中文"));
    assert!(result
        .extracted
        .contains("[the İstanbul guide](https://example.test/İstanbul)"));
}

#[test]
fn a_case_shrinking_character_does_not_panic_the_tag_scan() {
    // `ẞ` sits immediately before the noise tag, so a lower-cased copy reports
    // `<script` one byte early: the start offset falls *inside* the three-byte
    // `ẞ`, and slicing the original there panics.
    let html = "<html><body><p>ẞ<script>alert('x')</script></p>\
        <p>İstanbul and 中文 text with 🚀 keeps this paragraph long enough for the \
        extractor to treat it as content instead of page chrome, which is the \
        whole point of the scoring ladder.</p></body></html>";
    let result = HTMLExtractor::default().extract(html);
    assert!(!result.extracted.contains("alert"));
    assert!(result.extracted.contains("İstanbul and 中文"));
}

#[test]
fn a_case_expanding_attribute_does_not_hide_the_href() {
    // `İ` in the `title` value grows in a lower-cased copy, so `href` used to be
    // found one byte late, `=` no longer matched, and the link lost its target.
    let html = "<html><body><p>İstanbul is a city where the ferry schedule, the \
        market hours and every museum opening time matter to a visitor. Read \
        <a title=\"İstanbul\" href=\"https://example.test/İ\">the guide</a> \
        before the trip so the itinerary covers the old town as well.</p>\
        </body></html>";
    let result = HTMLExtractor::default().extract(html);
    assert!(result
        .extracted
        .contains("[the guide](https://example.test/İ)"));
}

#[test]
fn the_compressor_handles_non_ascii_pages() {
    let result = HtmlCompressor::default().compress(wide_lowercase_page(), &RuleContext::new(100));
    assert!(result.modified);
    let compressed = result.content;
    assert!(compressed.contains("中文"));
    assert!(compressed.contains("🚀"));
    assert!(!compressed.contains("alert"));
}

#[test]
fn tag_and_attribute_matching_is_ascii_case_insensitive() {
    let html = "<HTML><HEAD><TITLE>Ünal's guide</TITLE>\
        <Script>alert('İ')</SCRIPT>\
        <META NAME=\"author\" CONTENT=\"Ünal ẞ\"></HEAD>\
        <BODY><H1>Ünal's guide</H1>\
        <P>The <A HREF=\"https://example.test/Ünal\">guide</A> explains every \
        step of the migration in enough detail that the extractor keeps the \
        whole paragraph instead of dropping it as page chrome, which is exactly \
        what the ladder would do to a short link menu.</P></BODY></HTML>";
    let result = HTMLExtractor::default().extract(html);
    assert_eq!(result.title.as_deref(), Some("Ünal's guide"));
    assert_eq!(result.author.as_deref(), Some("Ünal ẞ"));
    assert!(!result.extracted.contains("alert"));
    assert!(result
        .extracted
        .contains("[guide](https://example.test/Ünal)"));
}
