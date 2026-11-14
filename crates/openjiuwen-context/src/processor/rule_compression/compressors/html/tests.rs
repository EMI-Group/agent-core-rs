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
