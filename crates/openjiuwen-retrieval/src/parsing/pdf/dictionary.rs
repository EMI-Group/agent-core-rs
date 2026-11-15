//! Stream-dictionary reader for `PDF` filters and predictor parameters.
//!
//! The parser never builds a full `PDF` object model: it scans `stream … endstream`
//! regions and looks at the dictionary bytes just before the `stream` keyword.
//! This module turns that window into the two things the decode pipeline needs —
//! the `/Filter` chain (one name or an array of names, decoded left to right) and
//! the `/DecodeParms` (or `/DP`) predictor dictionaries aligned with it.
//!
//! Everything here is tolerant by design: an unparsable value yields an empty
//! chain (the stream is then treated as unfiltered), a missing or `null` parameter
//! entry falls back to [`PredictorParams::default`], and an indirect reference
//! (`N 0 R`) is ignored so the default applies instead of a guessed number.

use super::filters::Filter;
use super::find_subslice;
use super::predictor::PredictorParams;
use super::{is_delimiter, is_regular, is_whitespace};

/// Reads the `/Filter` chain of a stream dictionary in decode order.
///
/// Returns an empty vector when the key is absent or its value is neither a name
/// nor an array of names (for example an indirect reference).
pub(crate) fn parse_filter_chain(window: &[u8]) -> Vec<Filter> {
    let Some(value) = value_after_key(window, b"/Filter") else {
        return Vec::new();
    };
    let value = skip_whitespace(value);
    match value.first() {
        Some(b'/') => {
            let name = read_name(value);
            if name.is_empty() {
                Vec::new()
            } else {
                vec![Filter::from_name(name)]
            }
        }
        Some(b'[') => filter_names_in_array(&value[1..])
            .iter()
            .map(|name| Filter::from_name(name))
            .collect(),
        _ => Vec::new(),
    }
}

/// Reads the `/DecodeParms` (or `/DP`) predictor parameters.
///
/// A single dictionary yields one entry; an array yields one entry per element,
/// with `None` for `null` and for anything unparsable. An absent key yields an
/// empty vector, so the caller falls back to [`PredictorParams::default`].
pub(crate) fn parse_decode_parms(window: &[u8]) -> Vec<Option<PredictorParams>> {
    let Some(value) =
        value_after_key(window, b"/DecodeParms").or_else(|| value_after_key(window, b"/DP"))
    else {
        return Vec::new();
    };
    let value = skip_whitespace(value);
    if value.starts_with(b"<<") {
        return vec![Some(parse_parameters(value))];
    }
    if value.first() == Some(&b'[') {
        return parse_parameters_array(&value[1..]);
    }
    Vec::new()
}

/// Returns the bytes after `key` when the key is a complete dictionary token.
fn value_after_key<'a>(window: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let found = find_key(window, key)?;
    Some(&window[found + key.len()..])
}

/// Returns the offset of `key` when it is delimited on both sides.
///
/// Both neighbours must be non-regular bytes, so `/Fl` never matches inside
/// `/FlateDecode` and `/Filter` never matches inside `/Filters`.
fn find_key(window: &[u8], key: &[u8]) -> Option<usize> {
    let mut cursor = 0usize;
    while let Some(found) = find_subslice(window, key, cursor) {
        let end = found + key.len();
        let starts_cleanly = found == 0 || !is_regular(window[found - 1]);
        let ends_cleanly = match window.get(end) {
            None => true,
            Some(byte) => !is_regular(*byte),
        };
        if starts_cleanly && ends_cleanly {
            return Some(found);
        }
        cursor = end;
    }
    None
}

/// Returns `slice` without leading `PDF` whitespace.
fn skip_whitespace(slice: &[u8]) -> &[u8] {
    let mut start = 0usize;
    while start < slice.len() && is_whitespace(slice[start]) {
        start += 1;
    }
    &slice[start..]
}

/// Reads a `/Name` body (the bytes after the slash) through the next delimiter.
fn read_name(slice: &[u8]) -> &str {
    let body = slice.get(1..).unwrap_or_default();
    let mut end = 0usize;
    while end < body.len() && is_regular(body[end]) {
        end += 1;
    }
    std::str::from_utf8(&body[..end]).unwrap_or("")
}

/// Collects the `/Name` entries of a `/Filter [ … ]` array, in order.
fn filter_names_in_array(slice: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0usize;
    while cursor < slice.len() {
        match slice[cursor] {
            b']' => break,
            b'/' => {
                let body = &slice[cursor + 1..];
                let mut end = 0usize;
                while end < body.len() && is_regular(body[end]) {
                    end += 1;
                }
                if let Ok(name) = std::str::from_utf8(&body[..end]) {
                    if !name.is_empty() {
                        names.push(name.to_owned());
                    }
                }
                cursor += 1 + end;
            }
            byte if is_whitespace(byte) || is_delimiter(byte) => cursor += 1,
            _ => {
                // A stray token (number, keyword) cannot name a filter.
                while cursor < slice.len()
                    && !is_whitespace(slice[cursor])
                    && !is_delimiter(slice[cursor])
                {
                    cursor += 1;
                }
                cursor += 1;
            }
        }
    }
    names
}

/// Parses `[ … ]` predictor parameters, one entry per array element.
fn parse_parameters_array(slice: &[u8]) -> Vec<Option<PredictorParams>> {
    let mut entries = Vec::new();
    let mut cursor = 0usize;
    while cursor < slice.len() {
        match slice[cursor] {
            b']' => break,
            b'<' if slice.get(cursor + 1) == Some(&b'<') => {
                let start = cursor + 2;
                let Some(relative_end) = find_subslice(&slice[start..], b">>", 0) else {
                    entries.push(None);
                    break;
                };
                let dictionary = &slice[start..start + relative_end];
                entries.push(Some(parse_parameters(dictionary)));
                cursor = start + relative_end + 2;
            }
            b'[' => {
                // A nested array is not a parameter dictionary.
                entries.push(None);
                cursor = skip_nested_array(slice, cursor);
            }
            byte if is_whitespace(byte) => cursor += 1,
            _ => {
                let Some((token, after)) = next_token(slice, cursor) else {
                    break;
                };
                cursor = after;
                // An indirect reference (`N 0 R`) contributes one ignored entry;
                // its remaining tokens are consumed here so later entries stay
                // aligned with the filter array.
                if token == b"R" {
                    continue;
                }
                if token.iter().all(u8::is_ascii_digit) {
                    if let Some(after) = indirect_reference(slice, cursor) {
                        entries.push(None);
                        cursor = after;
                        continue;
                    }
                }
                entries.push(None);
            }
        }
    }
    entries
}

/// Skips a nested `[ … ]` array and returns the offset after its `]`.
fn skip_nested_array(slice: &[u8], open: usize) -> usize {
    let mut depth = 0usize;
    let mut cursor = open;
    while cursor < slice.len() {
        match slice[cursor] {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return cursor + 1;
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    cursor
}

/// Parses one `<< … >>` parameter dictionary.
fn parse_parameters(dictionary: &[u8]) -> PredictorParams {
    let mut params = PredictorParams::default();
    if let Some(value) = integer_parameter(dictionary, b"/Predictor") {
        params.predictor = value;
    }
    if let Some(value) = integer_parameter(dictionary, b"/Columns") {
        params.columns = value as usize;
    }
    if let Some(value) = integer_parameter(dictionary, b"/Colors") {
        params.colors = value as usize;
    }
    if let Some(value) = integer_parameter(dictionary, b"/BitsPerComponent") {
        params.bits_per_component = value as usize;
    }
    params
}

/// Reads a non-negative integer parameter, ignoring indirect references.
fn integer_parameter(dictionary: &[u8], key: &[u8]) -> Option<u32> {
    let found = find_key(dictionary, key)?;
    let rest = &dictionary[found + key.len()..];
    let (token, after) = next_token(rest, 0)?;
    let value: u32 = std::str::from_utf8(token).ok()?.parse().ok()?;
    if indirect_reference(rest, after).is_some() {
        return None;
    }
    Some(value)
}

/// Recognises the `<object> <generation> R` tail of an indirect reference.
fn indirect_reference(slice: &[u8], cursor: usize) -> Option<usize> {
    let (generation, after_generation) = next_token(slice, cursor)?;
    if !generation.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let (keyword, after_keyword) = next_token(slice, after_generation)?;
    if keyword == b"R" {
        Some(after_keyword)
    } else {
        None
    }
}

/// Reads the next whitespace-separated, non-delimiter token.
fn next_token(slice: &[u8], cursor: usize) -> Option<(&[u8], usize)> {
    let mut start = cursor;
    while start < slice.len() && is_whitespace(slice[start]) {
        start += 1;
    }
    if start >= slice.len() || is_delimiter(slice[start]) {
        return None;
    }
    let mut end = start;
    while end < slice.len() && !is_whitespace(slice[end]) && !is_delimiter(slice[end]) {
        end += 1;
    }
    Some((&slice[start..end], end))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_single_filter_name_is_read() {
        assert_eq!(
            parse_filter_chain(b"<< /Filter /FlateDecode >>"),
            vec![Filter::Flate]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter/FlateDecode>>"),
            vec![Filter::Flate]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter /Ahx >>"),
            vec![Filter::Unknown("Ahx".to_owned())]
        );
    }

    #[test]
    fn abbreviations_are_resolved() {
        assert_eq!(
            parse_filter_chain(b"<< /Filter /AHx >>"),
            vec![Filter::AsciiHex]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter /A85 >>"),
            vec![Filter::Ascii85]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter /Fl >>"),
            vec![Filter::Flate]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter /RL >>"),
            vec![Filter::RunLength]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter /CCF >>"),
            vec![Filter::CcittFax]
        );
        assert_eq!(parse_filter_chain(b"<< /Filter /LZW >>"), vec![Filter::Lzw]);
        assert_eq!(
            parse_filter_chain(b"<< /Filter /AHx /Length 4 >>"),
            vec![Filter::AsciiHex]
        );
    }

    #[test]
    fn a_filter_array_keeps_its_decode_order() {
        assert_eq!(
            parse_filter_chain(b"<< /Filter [/ASCII85Decode /ASCIIHexDecode] >>"),
            vec![Filter::Ascii85, Filter::AsciiHex]
        );
        assert_eq!(
            parse_filter_chain(b"<< /Filter [/FlateDecode /DCTDecode] /Length 9 >>"),
            vec![Filter::Flate, Filter::Dct]
        );
        assert_eq!(
            parse_filter_chain(b"<</Filter[ /Fl /AHx ]>>"),
            vec![Filter::Flate, Filter::AsciiHex]
        );
        assert!(parse_filter_chain(b"<< /Filter [] >>").is_empty());
    }

    #[test]
    fn missing_and_unparsable_filters_yield_an_empty_chain() {
        assert!(parse_filter_chain(b"<< /Length 12 >>").is_empty());
        // `/Filter` alone has no value.
        assert!(parse_filter_chain(b"<< /Filter >>").is_empty());
        // `/Filters` and `/FlateDecode` are not the `/Filter` key.
        assert!(parse_filter_chain(b"<< /Filters 3 >>").is_empty());
        assert!(parse_filter_chain(b"<< /MyFilter /Fl >>").is_empty());
        // An indirect value is not resolved here.
        assert!(parse_filter_chain(b"<< /Filter 5 0 R >>").is_empty());
        assert!(parse_filter_chain(b"").is_empty());
        assert!(parse_filter_chain(b"garbage /Filter << >>").is_empty());
    }

    #[test]
    fn a_single_decode_parms_dictionary_is_read() {
        assert_eq!(
            parse_decode_parms(
                b"<< /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 8 >> >>"
            ),
            vec![Some(PredictorParams {
                predictor: 12,
                columns: 8,
                colors: 1,
                bits_per_component: 8,
            })]
        );
        assert_eq!(
            parse_decode_parms(
                b"<< /DP << /Predictor 2 /Columns 4 /Colors 3 /BitsPerComponent 16 >> >>"
            ),
            vec![Some(PredictorParams {
                predictor: 2,
                columns: 4,
                colors: 3,
                bits_per_component: 16,
            })]
        );
        // An empty dictionary keeps every default.
        assert_eq!(
            parse_decode_parms(b"<< /DecodeParms << >> >>"),
            vec![Some(PredictorParams::default())]
        );
        assert!(parse_decode_parms(b"<< /Length 4 >>").is_empty());
    }

    #[test]
    fn decode_parms_arrays_align_with_the_filter_array() {
        assert_eq!(
            parse_decode_parms(b"<< /DecodeParms [null << /Predictor 11 /Columns 8 >>] >>"),
            vec![
                None,
                Some(PredictorParams {
                    predictor: 11,
                    columns: 8,
                    colors: 1,
                    bits_per_component: 8,
                }),
            ]
        );
        assert_eq!(
            parse_decode_parms(b"<< /DP [<< /Predictor 2 >> << /Predictor 10 >>] >>"),
            vec![
                Some(PredictorParams {
                    predictor: 2,
                    ..PredictorParams::default()
                }),
                Some(PredictorParams {
                    predictor: 10,
                    ..PredictorParams::default()
                }),
            ]
        );
        assert_eq!(
            parse_decode_parms(b"<< /DP [<< /Predictor 10 >> null] >>"),
            vec![
                Some(PredictorParams {
                    predictor: 10,
                    ..PredictorParams::default()
                }),
                None,
            ]
        );
    }

    #[test]
    fn indirect_references_fall_back_to_the_defaults() {
        assert_eq!(
            parse_decode_parms(b"<< /DecodeParms << /Predictor 12 0 R /Columns 4 >> >>"),
            vec![Some(PredictorParams {
                predictor: 1,
                columns: 4,
                ..PredictorParams::default()
            })]
        );
        // An indirect entry inside the array contributes one ignored entry.
        assert_eq!(
            parse_decode_parms(b"<< /DP [5 0 R << /Predictor 11 >>] >>"),
            vec![
                None,
                Some(PredictorParams {
                    predictor: 11,
                    ..PredictorParams::default()
                }),
            ]
        );
    }

    #[test]
    fn a_balanced_dictionary_window_is_tolerated() {
        // Two dictionaries in the window: the first `/Filter` wins.
        let window = b"<< /Filter /AHx >> endobj 3 0 obj << /Filter /Fl /DecodeParms << /Predictor 10 /Columns 2 >> >>";
        assert_eq!(parse_filter_chain(window), vec![Filter::AsciiHex]);
        assert_eq!(
            parse_decode_parms(window),
            vec![Some(PredictorParams {
                predictor: 10,
                columns: 2,
                ..PredictorParams::default()
            })]
        );
        // A truncated dictionary keeps every default rather than guessing.
        assert_eq!(
            parse_decode_parms(b"<< /DecodeParms << /Predictor"),
            vec![Some(PredictorParams::default())]
        );
        assert!(parse_decode_parms(b"<< /DecodeParms 4 >>").is_empty());
    }
}
