//! Incremental Server-Sent-Events decoding shared by both model clients.
//!
//! The decoder accepts arbitrary byte chunks from a streaming HTTP response and
//! emits one [`SseEvent`] per dispatched event, so a multi-byte UTF-8 character
//! or a complete line may be split across [`SseDecoder::push`] calls without
//! corrupting the payload.
//!
//! The decoder implements the field grammar of the WHATWG event-stream format:
//! `\n` and `\r\n` line endings, multi-line `data:` fields joined with `\n`,
//! `event:` names, `:` comment lines, and blank-line dispatch. A trailing event
//! without a final blank line is flushed by [`SseDecoder::finish`]. The
//! `[DONE]` sentinel is transport-specific, so it is surfaced as ordinary data;
//! clients can test it with [`SseEvent::is_done`].
//!
//! ```
//! use openjiuwen_foundation::llm::sse::SseDecoder;
//!
//! let mut decoder = SseDecoder::new();
//! let events = decoder.push(b"event: delta\ndata: {\"text\":\"hi\"}\n\n");
//! assert_eq!(events.len(), 1);
//! assert_eq!(events[0].event.as_deref(), Some("delta"));
//! assert_eq!(events[0].data, "{\"text\":\"hi\"}");
//! assert!(decoder.finish().is_none());
//! ```

/// One Server-Sent-Events event.
///
/// `event` is the `event:` name when the server sent one; `None` means the
/// default event type. `data` is the joined payload of every `data:` line in the
/// event, without the trailing newline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field value, or `None` for the default event type.
    pub event: Option<String>,
    /// The joined `data:` field values.
    pub data: String,
}

impl SseEvent {
    /// Returns `true` when the payload is the `OpenAI` `[DONE]` sentinel.
    ///
    /// This is a convenience for OpenAI-compatible clients; the decoder never
    /// special-cases the sentinel and always surfaces it as data.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

/// Incremental decoder for a `text/event-stream` body.
///
/// Bytes are buffered until a line is complete, so callers can feed network
/// chunks of any size. Each decoded event is returned exactly once.
#[derive(Debug, Clone, Default)]
pub struct SseDecoder {
    /// Raw bytes received but not yet terminated by a `\n`.
    buffer: Vec<u8>,
    /// Joined `data:` values of the event being assembled.
    data: String,
    /// Whether at least one `data:` field was seen for the current event.
    data_seen: bool,
    /// `event:` name of the event being assembled.
    event: Option<String>,
}

impl SseDecoder {
    /// Creates an empty decoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one network chunk and returns every event completed by it.
    ///
    /// Incomplete lines (including partial UTF-8 sequences) stay buffered until
    /// the next call or until [`SseDecoder::finish`]. An empty chunk emits
    /// nothing.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);

        let mut events = Vec::new();
        let mut consumed = 0;
        while let Some(offset) = self.buffer[consumed..]
            .iter()
            .position(|&byte| byte == b'\n')
        {
            let line_end = consumed + offset;
            let line = decode_line(&self.buffer[consumed..line_end]);
            self.process_line(&line, &mut events);
            consumed = line_end + 1;
        }
        if consumed > 0 {
            self.buffer.drain(..consumed);
        }
        events
    }

    /// Flushes a trailing event that was not followed by a blank line.
    ///
    /// Returns `None` when the decoder holds no incomplete event. Calling
    /// `finish` more than once is safe; later calls return `None`.
    pub fn finish(&mut self) -> Option<SseEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let line = decode_line(&std::mem::take(&mut self.buffer));
            self.process_line(&line, &mut events);
        }
        if let Some(event) = self.dispatch() {
            events.push(event);
        }
        events.into_iter().next()
    }

    /// Applies one line of the event-stream grammar.
    fn process_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            if let Some(event) = self.dispatch() {
                events.push(event);
            }
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => {
                if self.data_seen {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.data_seen = true;
            }
            "event" => {
                self.event = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
            }
            _ => {}
        }
    }

    /// Emits the pending event, if it has data or an explicit event name.
    fn dispatch(&mut self) -> Option<SseEvent> {
        let data = std::mem::take(&mut self.data);
        self.data_seen = false;
        let event = self.event.take();
        if data.is_empty() && event.is_none() {
            return None;
        }
        Some(SseEvent { event, data })
    }
}

/// Decodes line bytes, dropping a trailing `\r` from a CRLF ending.
///
/// Invalid UTF-8 is replaced lossily rather than rejecting the stream, so a
/// malformed provider frame can never panic the client.
fn decode_line(line: &[u8]) -> String {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    String::from_utf8_lossy(line).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(data: &str) -> SseEvent {
        SseEvent {
            event: None,
            data: data.to_string(),
        }
    }

    fn assert_no_events(events: Vec<SseEvent>) {
        assert_eq!(events, Vec::new());
    }

    #[test]
    fn single_event() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: hello\n\n");
        assert_eq!(events, vec![event("hello")]);
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn single_event_with_crlf_line_endings() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: hello\r\n\r\n");
        assert_eq!(events, vec![event("hello")]);
    }

    #[test]
    fn multi_line_data_is_joined_with_newlines() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: line one\ndata: line two\ndata: line three\n\n");
        assert_eq!(events, vec![event("line one\nline two\nline three")]);
    }

    #[test]
    fn comment_lines_are_ignored() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b": keep-alive\ndata: payload\n\n");
        assert_eq!(events, vec![event("payload")]);
    }

    #[test]
    fn comment_only_event_is_not_emitted() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b": ping\n\n"));
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn event_field_is_reported() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"event: message_start\ndata: {\"ok\":true}\n\n");
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("message_start".to_string()),
                data: "{\"ok\":true}".to_string(),
            }]
        );
    }

    #[test]
    fn empty_event_name_resets_to_the_default_event() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b"event:\n\n"));
        let events = decoder.push(b"event:\ndata: payload\n\n");
        assert_eq!(events, vec![event("payload")]);
    }

    #[test]
    fn several_events_dispatch_from_one_push() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: a\n\ndata: b\n\ndata: c\n\n");
        assert_eq!(events, vec![event("a"), event("b"), event("c")]);
    }

    #[test]
    fn event_split_across_pushes_by_line() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b"data: hel"));
        assert_no_events(decoder.push(b"lo\n"));
        assert_eq!(decoder.push(b"\n"), vec![event("hello")]);
    }

    #[test]
    fn utf8_characters_split_across_one_byte_pushes() {
        let payload = "event: token\ndata: héllo 🌍\n\ndata: [DONE]\n\n";
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        for byte in payload.as_bytes() {
            events.extend(decoder.push(std::slice::from_ref(byte)));
        }
        assert!(decoder.finish().is_none());
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("token".to_string()),
                    data: "héllo 🌍".to_string(),
                },
                event("[DONE]"),
            ]
        );
    }

    #[test]
    fn crlf_split_across_pushes() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b"data: hello\r"));
        assert_no_events(decoder.push(b"\n\r"));
        assert_eq!(decoder.push(b"\n"), vec![event("hello")]);
    }

    #[test]
    fn trailing_event_is_flushed_by_finish() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b"data: tail"));
        assert_eq!(decoder.finish(), Some(event("tail")));
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn trailing_multiline_event_is_flushed_by_finish() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"event: done\ndata: first\ndata: second");
        assert_no_events(events);
        assert_eq!(
            decoder.finish(),
            Some(SseEvent {
                event: Some("done".to_string()),
                data: "first\nsecond".to_string(),
            })
        );
    }

    #[test]
    fn done_sentinel_is_surfaced_as_data() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
        assert!(events[0].is_done());
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn is_done_trims_surrounding_whitespace() {
        assert!(SseEvent {
            event: None,
            data: "  [DONE] \n".to_string(),
        }
        .is_done());
        assert!(!event("[DONE] trailing").is_done());
        assert!(!event("done").is_done());
        assert!(!event("").is_done());
    }

    #[test]
    fn empty_push_returns_nothing() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b""));
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn blank_lines_without_data_are_not_emitted() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b"\n\r\n\n"));
        assert!(decoder.finish().is_none());
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"id: 42\nretry: 1000\ndata: payload\n\n");
        assert_eq!(events, vec![event("payload")]);
    }

    #[test]
    fn field_without_colon_has_an_empty_value() {
        let mut decoder = SseDecoder::new();
        // An unknown field without a colon is ignored, and a lone `data` with no
        // value still counts as a (skipped) empty event.
        assert_no_events(decoder.push(b"retry\n\n"));
        assert_no_events(decoder.push(b"data\n\n"));
    }

    #[test]
    fn only_one_leading_space_is_stripped_from_values() {
        let mut decoder = SseDecoder::new();
        assert_eq!(
            decoder.push(b"data:  two spaces\n\n"),
            vec![event(" two spaces")]
        );
        assert_eq!(decoder.push(b"data:no space\n\n"), vec![event("no space")]);
    }

    #[test]
    fn empty_pushes_do_not_disturb_a_partial_event() {
        let mut decoder = SseDecoder::new();
        assert_no_events(decoder.push(b"data: partial\n"));
        assert_no_events(decoder.push(b""));
        assert_eq!(decoder.push(b"\n"), vec![event("partial")]);
    }

    #[test]
    fn invalid_utf8_is_replaced_without_panicking() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: \xff\xfe\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "\u{fffd}\u{fffd}");
    }

    #[test]
    fn finish_after_a_dispatched_event_returns_none() {
        let mut decoder = SseDecoder::new();
        assert_eq!(decoder.push(b"data: x\n\n"), vec![event("x")]);
        assert!(decoder.finish().is_none());
    }
}
