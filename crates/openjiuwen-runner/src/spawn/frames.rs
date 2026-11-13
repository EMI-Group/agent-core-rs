//! Bounded NDJSON framing over the spawn seam's stdio pipes.
//!
//! One frame is one line of UTF-8 JSON ([`SpawnMessage`], see
//! [`crate::spawn::protocol`]). [`read_frame`] reads a line with a hard
//! [`DEFAULT_MAX_FRAME_BYTES`] cap so a runaway child cannot make the parent
//! allocate without bound, and it tolerates the noise a real child prints on
//! stdout (Python parity: `deserialize_message_from_stream` skips
//! non-protocol output) — a line that is not JSON is dropped with a `trace!`,
//! while a line that *is* JSON but not a frame is a validation error, because
//! that means the child and the parent disagree about the protocol.
//!
//! Nothing here owns a pipe: these are pure helpers over
//! [`tokio::io::AsyncBufRead`] and byte slices, so they are exercised directly
//! by unit tests without spawning a process.

use crate::spawn::protocol::SpawnMessage;
use openjiuwen_common::error::{Error, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt};
use tracing::trace;

/// Default cap on one frame line: 8 MiB.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Longest prefix of a skipped line echoed into the `trace!` log.
const LOG_PREVIEW_CHARS: usize = 160;

/// `tracing` target for the spawn seam.
const TARGET: &str = "openjiuwen_runner::spawn";

/// Reads one frame, skipping non-JSON noise lines.
///
/// Returns `Ok(None)` at end of input. A trailing line without a terminating
/// newline is still parsed (a child that exits mid-line is common enough that
/// dropping its last frame would lose the exit reason).
///
/// # Errors
///
/// Returns [`Error::Validation`] when a line exceeds `max_frame_bytes`, is not
/// UTF-8, or is JSON that is not a [`SpawnMessage`]; returns
/// [`Error::Transport`] when the underlying reader fails.
///
/// A line that is not JSON at all is not an error: it is skipped (the child's
/// stray stdout), and reading continues with the next line.
pub async fn read_frame<R>(reader: &mut R, max_frame_bytes: usize) -> Result<Option<SpawnMessage>>
where
    R: AsyncBufRead + Unpin + Send,
{
    loop {
        let Some(line) = read_line_bounded(reader, max_frame_bytes).await? else {
            return Ok(None);
        };
        if let Some(frame) = parse_frame_line(&line, max_frame_bytes)? {
            return Ok(Some(frame));
        }
    }
}

/// Parses one raw line (newline optional).
///
/// Returns `Ok(None)` for a blank line or for non-JSON noise, which callers
/// skip.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the line exceeds `max_frame_bytes`, is not
/// valid UTF-8, or is valid JSON that does not describe a [`SpawnMessage`]
/// (unknown `kind`, missing field, non-object).
pub fn parse_frame_line(bytes: &[u8], max_frame_bytes: usize) -> Result<Option<SpawnMessage>> {
    if bytes.len() > max_frame_bytes {
        return Err(Error::validation(format!(
            "spawn frame line of {} bytes exceeds the {max_frame_bytes} byte cap",
            bytes.len()
        )));
    }
    let text = std::str::from_utf8(bytes).map_err(|error| {
        Error::validation(format!("spawn frame line is not valid UTF-8: {error}"))
    })?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if serde_json::from_str::<serde_json::Value>(trimmed).is_err() {
        trace!(
            target: TARGET,
            line = %log_preview(trimmed),
            "skipping a non-JSON line from the child's stdout"
        );
        return Ok(None);
    }
    SpawnMessage::from_line(trimmed).map(Some)
}

/// Serializes one frame to newline-terminated bytes.
///
/// # Errors
///
/// Returns the error [`SpawnMessage::to_line`] reports (unserializable frame, or
/// serialization that would break NDJSON framing).
pub fn frame_bytes(frame: &SpawnMessage) -> Result<Vec<u8>> {
    Ok(frame.to_line()?.into_bytes())
}

/// Reads one newline-terminated line, never buffering more than
/// `max_frame_bytes` plus the reader's own chunk.
///
/// Uses [`AsyncBufReadExt::fill_buf`] + [`AsyncBufReadExt::consume`] so the cap
/// holds even when the reader's buffer is much larger than the cap: only the
/// bytes actually consumed are copied into the accumulator.
///
/// Returns `Ok(None)` at EOF with nothing buffered; a partial final line is
/// returned as-is.
async fn read_line_bounded<R>(reader: &mut R, max_frame_bytes: usize) -> Result<Option<Vec<u8>>>
where
    R: AsyncBufRead + Unpin + Send,
{
    let mut line: Vec<u8> = Vec::new();
    loop {
        let chunk = reader.fill_buf().await.map_err(|error| {
            crate::error::transport(format!("reading from the child process failed: {error}"))
        })?;
        if chunk.is_empty() {
            return Ok(if line.is_empty() { None } else { Some(line) });
        }
        let available = chunk.len();
        let newline = chunk.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(available);
        if line.len().saturating_add(take) > max_frame_bytes {
            return Err(Error::validation(format!(
                "spawn frame line exceeds the {max_frame_bytes} byte cap"
            )));
        }
        line.extend_from_slice(&chunk[..take]);
        match newline {
            Some(index) => {
                reader.consume(index + 1);
                return Ok(Some(line));
            }
            None => reader.consume(available),
        }
    }
}

/// A short, char-boundary-safe prefix of `text` for the trace log.
fn log_preview(text: &str) -> String {
    let mut shown: String = text.chars().take(LOG_PREVIEW_CHARS).collect();
    if text.chars().count() > LOG_PREVIEW_CHARS {
        shown.push('…');
    }
    shown
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::spawn::protocol::SpawnMessageKind;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn frame(kind: SpawnMessageKind, payload: serde_json::Value) -> SpawnMessage {
        SpawnMessage::new(kind, payload)
    }

    #[test]
    fn default_cap_is_eight_mebibytes() {
        assert_eq!(DEFAULT_MAX_FRAME_BYTES, 8 * 1024 * 1024);
    }

    #[test]
    fn blank_lines_are_skippable_noise() {
        for blank in [&b"\n"[..], b"   \n", b"\r\n", b"\t"] {
            assert!(
                parse_frame_line(blank, DEFAULT_MAX_FRAME_BYTES)
                    .unwrap()
                    .is_none(),
                "blank line {blank:?} should be skipped"
            );
        }
    }

    #[test]
    fn non_json_lines_are_skippable_noise() {
        for noise in [
            &b"starting child...\n"[..],
            b"WARNING: something happened\n",
            b"{\"kind\": \"input\", \"payload\": \n",
        ] {
            assert!(
                parse_frame_line(noise, DEFAULT_MAX_FRAME_BYTES)
                    .unwrap()
                    .is_none(),
                "non-JSON line {noise:?} should be skipped"
            );
        }
    }

    #[test]
    fn valid_json_that_is_not_a_frame_is_a_validation_error() {
        for line in [
            &b"42\n"[..],
            b"\"text\"\n",
            b"[1, 2]\n",
            b"null\n",
            b"{\"kind\": \"teleport\"}\n",
            b"{\"kind\": \"input\"}\n",
        ] {
            let error = parse_frame_line(line, DEFAULT_MAX_FRAME_BYTES)
                .expect_err("valid JSON that is not a frame must be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation, "for line {line:?}");
        }
    }

    #[test]
    fn a_frame_line_round_trips_with_its_newline() {
        let message = frame(SpawnMessageKind::Output, json!({"output": "hi"}));
        let bytes = frame_bytes(&message).unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
        let parsed = parse_frame_line(&bytes, DEFAULT_MAX_FRAME_BYTES)
            .unwrap()
            .expect("a frame line parses");
        assert_eq!(parsed.kind, SpawnMessageKind::Output);
        assert_eq!(parsed.payload, json!({"output": "hi"}));
        assert_eq!(parsed.message_id, message.message_id);
    }

    #[test]
    fn an_oversized_line_is_a_validation_error() {
        let long = vec![b'a'; 65];
        let error = parse_frame_line(&long, 64).expect_err("must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("64 byte cap"),
            "unexpected message: {error}"
        );

        // Exactly at the cap is still accepted.
        let exact = frame_bytes(&frame(SpawnMessageKind::Done, serde_json::Value::Null)).unwrap();
        assert!(parse_frame_line(&exact, exact.len()).is_ok());
        assert!(parse_frame_line(&exact, exact.len() - 1).is_err());
    }

    #[test]
    fn non_utf8_bytes_are_a_validation_error() {
        let error = parse_frame_line(&[0xff, 0xfe, 0x00, b'\n'], DEFAULT_MAX_FRAME_BYTES)
            .expect_err("must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("UTF-8"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn read_frame_returns_none_at_eof() {
        let mut reader: &[u8] = b"";
        assert!(read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn read_frame_skips_noise_and_returns_the_next_frame() {
        let message = frame(SpawnMessageKind::StreamChunk, json!({"content": "x"}));
        let mut input = Vec::new();
        input.extend_from_slice(b"booting the child\n");
        input.extend_from_slice(b"\n");
        input.extend_from_slice(&frame_bytes(&message).unwrap());
        let mut reader: &[u8] = &input;
        let parsed = read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap()
            .expect("one frame");
        assert_eq!(parsed.kind, SpawnMessageKind::StreamChunk);
        assert_eq!(parsed.payload, json!({"content": "x"}));
        assert!(read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn read_frame_reads_multiple_frames_in_order() {
        let mut input = Vec::new();
        for index in 0..3 {
            let message = frame(SpawnMessageKind::StreamChunk, json!({"index": index}));
            input.extend_from_slice(&frame_bytes(&message).unwrap());
        }
        let mut reader: &[u8] = &input;
        for index in 0..3 {
            let parsed = read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap()
                .expect("a frame");
            assert_eq!(parsed.payload, json!({"index": index}));
        }
        assert!(read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn read_frame_accepts_an_unterminated_final_line() {
        let message = frame(SpawnMessageKind::Error, json!({"message": "boom"}));
        let mut bytes = frame_bytes(&message).unwrap();
        bytes.pop();
        let mut reader: &[u8] = &bytes;
        let parsed = read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap()
            .expect("the trailing line is still parsed");
        assert_eq!(parsed.kind, SpawnMessageKind::Error);
        assert_eq!(parsed.payload, json!({"message": "boom"}));
    }

    #[tokio::test]
    async fn read_frame_rejects_a_line_over_the_cap() {
        // One line longer than the cap, with a frame after it: the reader fails
        // instead of buffering the whole line.
        let mut input = vec![b'a'; 128];
        input.push(b'\n');
        input.extend_from_slice(&frame_bytes(&frame(SpawnMessageKind::Done, json!(null))).unwrap());
        let mut reader: &[u8] = &input;
        let error = read_frame(&mut reader, 64)
            .await
            .expect_err("oversized line must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("64 byte cap"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn read_frame_stops_at_a_bad_frame_after_noise() {
        let mut input = Vec::new();
        input.extend_from_slice(b"noise\n");
        input.extend_from_slice(b"{\"kind\": \"teleport\"}\n");
        let mut reader: &[u8] = &input;
        let error = read_frame(&mut reader, DEFAULT_MAX_FRAME_BYTES)
            .await
            .expect_err("a bad frame is not noise");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn log_preview_is_short_and_respects_char_boundaries() {
        let long = "é".repeat(LOG_PREVIEW_CHARS + 5);
        let shown = log_preview(&long);
        assert_eq!(shown.chars().count(), LOG_PREVIEW_CHARS + 1);
        assert!(shown.ends_with('…'));
        assert_eq!(log_preview("short"), "short");
    }
}
