//! Charset detection and lossy response-body decoding for `fetch_webpage`.
//!
//! Ports `openjiuwen/harness/tools/web/_decode.py`: the `Content-Type` charset parameter wins
//! over a `<meta charset>` sniff, a byte-order mark wins over both, and anything undecodable
//! is replaced instead of panicking. Only the UTF, Latin-1 and UTF-16 families are decoded
//! exactly; every other label falls back to lossy UTF-8 (Python decodes the full codec
//! registry, which Rust's standard library does not ship).

/// The charset label used when nothing else declares one.
pub const DEFAULT_CHARSET: &str = "utf-8";

/// How many leading body bytes the `<meta charset>` sniff inspects.
const META_SNIFF_BYTES: usize = 4096;

/// Detects the charset of a response body.
///
/// Precedence: byte-order mark, then the `charset` parameter of `content_type`, then a
/// `<meta charset>`/`<meta http-equiv="content-type">` declaration in the first 4 KiB, then
/// [`DEFAULT_CHARSET`]. The returned label is lowercased and trimmed.
#[must_use]
pub fn detect_charset(content_type: &str, body: &[u8]) -> String {
    if let Some(charset) = bom_charset(body) {
        return charset.to_owned();
    }
    if let Some(charset) = charset_from_content_type(content_type) {
        return charset;
    }
    if let Some(charset) = sniff_meta_charset(body) {
        return charset;
    }
    DEFAULT_CHARSET.to_owned()
}

/// Extracts the `charset` parameter of a `Content-Type` header value.
///
/// Handles both the quoted (`charset="utf-8"`) and the bare (`charset=utf-8`) form, and
/// ignores every other parameter.
#[must_use]
pub fn charset_from_content_type(content_type: &str) -> Option<String> {
    let lower = content_type.to_ascii_lowercase();
    let start = lower.find("charset")? + "charset".len();
    let rest = lower[start..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let quote = rest.chars().next().filter(|c| *c == '"' || *c == '\'');
    let value = match quote {
        Some(quote) => rest[1..].split(quote).next().unwrap_or(""),
        None => rest
            .split(|c: char| {
                c == ';' || c == '"' || c == '\'' || c == '>' || c == '<' || c.is_whitespace()
            })
            .next()
            .unwrap_or(""),
    };
    normalize_charset_label(value)
}

/// Sniffs a `<meta charset>` or `<meta http-equiv="content-type">` declaration.
///
/// The first 4 KiB are scanned as UTF-8 (lossily) so a Latin-1 page still yields a readable
/// declaration; the first declaration that carries a charset wins.
#[must_use]
pub fn sniff_meta_charset(body: &[u8]) -> Option<String> {
    let window = &body[..body.len().min(META_SNIFF_BYTES)];
    let text = String::from_utf8_lossy(window);
    let lower = text.to_ascii_lowercase();
    let mut offset = 0;
    while let Some(position) = lower[offset..].find("<meta") {
        let start = offset + position;
        let end = lower[start..]
            .find('>')
            .map_or(lower.len(), |end| start + end + 1);
        if let Some(charset) = charset_from_content_type(&lower[start..end]) {
            return Some(charset);
        }
        offset = end;
    }
    None
}

/// Decodes `body` with `charset`, replacing undecodable bytes.
///
/// `utf-8`/`ascii` (the default), `iso-8859-1`/`windows-1252` and the `utf-16` family are
/// decoded exactly; every other label decodes as lossy UTF-8. A leading byte-order mark is
/// stripped. The function never fails and never panics.
#[must_use]
pub fn decode_body(body: &[u8], charset: &str) -> String {
    let label = normalize_charset_label(charset).unwrap_or_else(|| DEFAULT_CHARSET.to_owned());
    match label.as_str() {
        "utf-8" | "utf8" | "us-ascii" | "ascii" => {
            let body = strip_utf8_bom(body);
            String::from_utf8_lossy(body).into_owned()
        }
        "iso-8859-1" | "iso8859-1" | "latin1" | "latin-1" | "windows-1252" | "cp1252" => {
            body.iter().map(|byte| char::from(*byte)).collect()
        }
        "utf-16" | "utf-16le" | "utf16" | "utf16le" => decode_utf16(body, false),
        "utf-16be" | "utf16be" => decode_utf16(body, true),
        _ => String::from_utf8_lossy(strip_utf8_bom(body)).into_owned(),
    }
}

/// Returns the charset implied by a leading byte-order mark.
#[must_use]
pub fn bom_charset(body: &[u8]) -> Option<&'static str> {
    if body.starts_with(&[0xef, 0xbb, 0xbf]) {
        Some("utf-8")
    } else if body.starts_with(&[0xff, 0xfe]) {
        Some("utf-16le")
    } else if body.starts_with(&[0xfe, 0xff]) {
        Some("utf-16be")
    } else {
        None
    }
}

/// Trims, lowercases and validates a charset label.
fn normalize_charset_label(value: &str) -> Option<String> {
    let label = value.trim().trim_matches(['"', '\'']).to_ascii_lowercase();
    if label.is_empty()
        || !label.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return None;
    }
    Some(label)
}

/// Removes a leading UTF-8 byte-order mark.
fn strip_utf8_bom(body: &[u8]) -> &[u8] {
    body.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(body)
}

/// Decodes UTF-16 in `big_endian` order, dropping a byte-order mark and a trailing odd byte.
fn decode_utf16(body: &[u8], big_endian: bool) -> String {
    let mut units: Vec<u16> = body
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    if units.first() == Some(&0xfeff) {
        units.remove(0);
    }
    String::from_utf16_lossy(&units)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn reads_charset_from_the_content_type() {
        assert_eq!(
            charset_from_content_type("text/html; charset=UTF-8").as_deref(),
            Some("utf-8")
        );
        assert_eq!(
            charset_from_content_type("text/plain; charset=\"ISO-8859-1\"").as_deref(),
            Some("iso-8859-1")
        );
        assert_eq!(
            charset_from_content_type("application/json;charset='gbk'").as_deref(),
            Some("gbk")
        );
        assert_eq!(charset_from_content_type("text/html"), None);
        assert_eq!(charset_from_content_type(""), None);
    }

    #[test]
    fn sniffs_meta_charset_declarations() {
        let meta = br#"<html><head><meta charset="ISO-8859-1"><title>t</title></head></html>"#;
        assert_eq!(sniff_meta_charset(meta).as_deref(), Some("iso-8859-1"));

        let http_equiv = br#"<meta http-equiv="Content-Type" content="text/html; charset=gbk">"#;
        assert_eq!(sniff_meta_charset(http_equiv).as_deref(), Some("gbk"));

        assert_eq!(sniff_meta_charset(b"<html><body>plain</body></html>"), None);
    }

    #[test]
    fn detection_precedence_is_bom_then_header_then_meta() {
        let bom = [0xef, 0xbb, 0xbf, b'h', b'i'];
        assert_eq!(
            detect_charset("text/html; charset=iso-8859-1", &bom),
            "utf-8"
        );

        let body = br#"<meta charset="iso-8859-1">"#;
        assert_eq!(detect_charset("text/html; charset=utf-8", body), "utf-8");
        assert_eq!(detect_charset("text/html", body), "iso-8859-1");
        assert_eq!(detect_charset("text/html", b"plain"), "utf-8");
    }

    #[test]
    fn decodes_utf8_lossily_and_strips_the_bom() {
        assert_eq!(decode_body(b"caf\xc3\xa9", "utf-8"), "café");
        assert_eq!(decode_body(b"\xef\xbb\xbfhi", "utf-8"), "hi");
        assert_eq!(decode_body(b"a\xffb", "utf-8"), "a\u{fffd}b");
        assert_eq!(decode_body(b"plain", ""), "plain");
    }

    #[test]
    fn decodes_latin1_and_windows_1252() {
        assert_eq!(decode_body(b"caf\xe9", "iso-8859-1"), "café");
        assert_eq!(decode_body(b"\xe9", "latin1"), "é");
        assert_eq!(decode_body(b"caf\xe9", "windows-1252"), "café");
    }

    #[test]
    fn decodes_utf16_in_both_orders() {
        let little = [0xff, 0xfe, b'h', 0x00, b'i', 0x00];
        assert_eq!(decode_body(&little, "utf-16le"), "hi");
        let big = [0x00, b'h', 0x00, b'i'];
        assert_eq!(decode_body(&big, "utf-16be"), "hi");
        assert_eq!(decode_body(b"h\x00i\x00", "utf-16"), "hi");
    }

    #[test]
    fn unknown_labels_fall_back_to_lossy_utf8() {
        assert_eq!(decode_body(b"caf\xc3\xa9", "gbk"), "café");
        assert_eq!(decode_body(b"caf\xe9", "shift_jis"), "caf\u{fffd}");
    }
}
