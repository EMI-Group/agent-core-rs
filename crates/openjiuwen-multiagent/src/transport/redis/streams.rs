//! RESP2 helpers for the Redis Streams commands the transport uses.
//!
//! Every helper here owns one lock of the transport's shared [`RedisLink`] and
//! sends exactly one command, so callers can keep their own logic (dispatch
//! order, acknowledgements, backoff) out of the wire layer. The encoding is the
//! command set described in [`super`]: `XADD`, `XGROUP CREATE`, `XREADGROUP`,
//! `XREAD`, `XACK`, `XAUTOCLAIM` and `DEL`.
//!
//! Replies are decoded into [`StreamEntry`] values — an id plus its field map —
//! so a worker never has to walk RESP arrays itself. A malformed reply is a
//! transport error rather than a panic.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::external::redis::codec::RespValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use super::RedisLink;

/// One entry of a stream reply: its id and its field/value pairs.
#[derive(Debug)]
pub(super) struct StreamEntry {
    /// The entry id (`<ms>-<seq>`).
    pub(super) id: String,
    /// The entry's fields; both names and values are binary-safe.
    pub(super) fields: BTreeMap<String, Vec<u8>>,
}

impl StreamEntry {
    /// Returns the value of `field`, when the entry carries it.
    pub(super) fn field(&self, field: &str) -> Option<&[u8]> {
        self.fields.get(field).map(Vec::as_slice)
    }
}

/// Builds one stream key of the `{prefix}:{team}:{kind}:{id}` scheme.
///
/// The same scheme names all three key families, so the configuration documented
/// on the transport applies verbatim.
pub(super) fn stream_key(key_prefix: &str, team_id: &str, kind: &str, id: &str) -> String {
    format!("{key_prefix}:{team_id}:{kind}:{id}")
}

/// Derives the consumer-group name of one key family.
///
/// `override_name` (the configured `consumer_group`) wins when it is non-empty;
/// otherwise the group is `{prefix}:{team}:{kind}:cg`, which keeps one group per
/// key family and therefore per inbox class.
pub(super) fn group_name(
    key_prefix: &str,
    team_id: &str,
    kind: &str,
    override_name: Option<&str>,
) -> String {
    match override_name.filter(|name| !name.is_empty()) {
        Some(name) => name.to_owned(),
        None => format!("{key_prefix}:{team_id}:{kind}:cg"),
    }
}

/// Appends one entry with an auto-generated id and returns that id.
pub(super) async fn xadd(
    link: &AsyncMutex<RedisLink>,
    key: &str,
    fields: &[(&str, &[u8])],
) -> Result<String> {
    let mut args: Vec<&[u8]> = Vec::with_capacity(3 + fields.len() * 2);
    args.push(b"XADD");
    args.push(key.as_bytes());
    args.push(b"*");
    for (name, value) in fields {
        args.push(name.as_bytes());
        args.push(value);
    }
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::BulkString(Some(id)) => Ok(String::from_utf8_lossy(&id).into_owned()),
        RespValue::SimpleString(id) => Ok(id),
        other => Err(unexpected("XADD", &other)),
    }
}

/// Creates `group` on `key` at id `0`, tolerating an existing group.
///
/// `MKSTREAM` creates a missing key, so a worker may start before any producer.
/// A `BUSYGROUP` error means another process already created the group and is
/// success for this idempotent call.
pub(super) async fn xgroup_create(
    link: &AsyncMutex<RedisLink>,
    key: &str,
    group: &str,
) -> Result<()> {
    let args: [&[u8]; 6] = [
        b"XGROUP",
        b"CREATE",
        key.as_bytes(),
        group.as_bytes(),
        b"0",
        b"MKSTREAM",
    ];
    let mut link = link.lock().await;
    match link.command(&args).await {
        Ok(RespValue::SimpleString(_)) => Ok(()),
        Err(error) if error.message().contains("BUSYGROUP") => Ok(()),
        Ok(other) => Err(unexpected("XGROUP CREATE", &other)),
        Err(error) => Err(error),
    }
}

/// Reads up to `count` never-delivered entries through the `>` cursor.
///
/// The server hands every returned entry to `consumer` and records it in the
/// group's pending list, so an entry that is not acknowledged is reclaimable by
/// [`xautoclaim`].
pub(super) async fn xreadgroup(
    link: &AsyncMutex<RedisLink>,
    key: &str,
    group: &str,
    consumer: &str,
    count: usize,
) -> Result<Vec<StreamEntry>> {
    let count = count.to_string();
    let args: [&[u8]; 9] = [
        b"XREADGROUP",
        b"GROUP",
        group.as_bytes(),
        consumer.as_bytes(),
        b"COUNT",
        count.as_bytes(),
        b"STREAMS",
        key.as_bytes(),
        b">",
    ];
    let mut link = link.lock().await;
    let reply = link.command(&args).await?;
    parse_streams("XREADGROUP", reply)
}

/// Reads up to `count` entries from the start of the stream (`STREAMS key 0`).
pub(super) async fn xread(
    link: &AsyncMutex<RedisLink>,
    key: &str,
    count: usize,
) -> Result<Vec<StreamEntry>> {
    let count = count.to_string();
    let args: [&[u8]; 6] = [
        b"XREAD",
        b"COUNT",
        count.as_bytes(),
        b"STREAMS",
        key.as_bytes(),
        b"0",
    ];
    let mut link = link.lock().await;
    let reply = link.command(&args).await?;
    parse_streams("XREAD", reply)
}

/// Acknowledges one delivered entry.
pub(super) async fn xack(
    link: &AsyncMutex<RedisLink>,
    key: &str,
    group: &str,
    id: &str,
) -> Result<()> {
    let args: [&[u8]; 4] = [b"XACK", key.as_bytes(), group.as_bytes(), id.as_bytes()];
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::Integer(_) => Ok(()),
        other => Err(unexpected("XACK", &other)),
    }
}

/// Claims pending entries idle for at least `min_idle_ms`.
///
/// Returns the next cursor (the caller stops at `0-0`) plus the claimed entries,
/// whose ownership the server moved to `consumer`.
pub(super) async fn xautoclaim(
    link: &AsyncMutex<RedisLink>,
    key: &str,
    group: &str,
    consumer: &str,
    min_idle_ms: u64,
    start: &str,
    count: usize,
) -> Result<(String, Vec<StreamEntry>)> {
    let idle = min_idle_ms.to_string();
    let count = count.to_string();
    let args: [&[u8]; 8] = [
        b"XAUTOCLAIM",
        key.as_bytes(),
        group.as_bytes(),
        consumer.as_bytes(),
        idle.as_bytes(),
        start.as_bytes(),
        b"COUNT",
        count.as_bytes(),
    ];
    let mut link = link.lock().await;
    let reply = link.command(&args).await?;
    let RespValue::Array(Some(mut parts)) = reply else {
        return Err(unexpected("XAUTOCLAIM", &reply));
    };
    if parts.len() < 2 {
        return Err(unexpected("XAUTOCLAIM", &RespValue::Array(Some(parts))));
    }
    let entries = parse_entry_list("XAUTOCLAIM", parts.remove(1))?;
    let cursor = parts
        .first()
        .and_then(RespValue::as_string)
        .ok_or_else(|| Error::transport("redis transport: XAUTOCLAIM cursor was not a string"))?;
    Ok((cursor, entries))
}

/// Deletes `key` (and its stream), returning `true` when it existed.
pub(super) async fn del(link: &AsyncMutex<RedisLink>, key: &str) -> Result<bool> {
    let args: [&[u8]; 2] = [b"DEL", key.as_bytes()];
    let mut link = link.lock().await;
    match link.command(&args).await? {
        RespValue::Integer(removed) => Ok(removed > 0),
        other => Err(unexpected("DEL", &other)),
    }
}

/// The reply one recipient's worker appends to `{prefix}:{team}:reply:{id}`.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct ReplyFrame {
    /// The handler's reply value, when it succeeded.
    #[serde(default)]
    value: Option<Value>,
    /// The handler's error text, when it failed.
    #[serde(default)]
    error: Option<String>,
}

impl ReplyFrame {
    /// Builds the frame of a successful delivery.
    pub(super) fn from_value(value: Value) -> Self {
        Self {
            value: Some(value),
            error: None,
        }
    }

    /// Builds the frame of a failed delivery.
    pub(super) fn from_error(message: impl Into<String>) -> Self {
        Self {
            value: None,
            error: Some(message.into()),
        }
    }

    /// Serializes the frame for an `XADD frame <bytes>` field.
    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|error| {
            Error::transport(format!(
                "redis transport: cannot encode a reply frame: {error}"
            ))
        })
    }

    /// Turns the frame back into the waiting `send`'s result.
    pub(super) fn into_result(self) -> Result<Value> {
        match self.error {
            Some(message) => Err(Error::transport(format!(
                "redis transport: remote handler failed: {message}"
            ))),
            None => Ok(self.value.unwrap_or(Value::Null)),
        }
    }
}

/// Decodes a reply frame, logging and skipping a malformed one.
pub(super) fn decode_reply(payload: &[u8]) -> Option<ReplyFrame> {
    match serde_json::from_slice(payload) {
        Ok(frame) => Some(frame),
        Err(error) => {
            tracing::warn!(%error, "redis transport skipped a malformed reply frame");
            None
        }
    }
}

/// Decodes the `XREAD`/`XREADGROUP` shape: an array of `[key, entries]` pairs.
fn parse_streams(command: &str, reply: RespValue) -> Result<Vec<StreamEntry>> {
    let streams = match reply {
        RespValue::Array(Some(streams)) => streams,
        RespValue::Array(None) | RespValue::Null => return Ok(Vec::new()),
        other => return Err(unexpected(command, &other)),
    };
    let mut entries = Vec::new();
    for stream in streams {
        let RespValue::Array(Some(mut pair)) = stream else {
            return Err(malformed(command));
        };
        if pair.len() != 2 {
            return Err(malformed(command));
        }
        let stream_entries = pair.remove(1);
        entries.extend(parse_entry_list(command, stream_entries)?);
    }
    Ok(entries)
}

/// Decodes an array of `[id, [field, value, ...]]` entries.
fn parse_entry_list(command: &str, frame: RespValue) -> Result<Vec<StreamEntry>> {
    let frames = match frame {
        RespValue::Array(Some(frames)) => frames,
        RespValue::Array(None) | RespValue::Null => return Ok(Vec::new()),
        other => return Err(unexpected(command, &other)),
    };
    let mut entries = Vec::with_capacity(frames.len());
    for frame in frames {
        let RespValue::Array(Some(mut pair)) = frame else {
            return Err(malformed(command));
        };
        if pair.len() != 2 {
            return Err(malformed(command));
        }
        let fields_frame = pair.remove(1);
        let id = pair.remove(0).as_string().ok_or_else(|| {
            Error::transport("redis transport: a stream entry id was not a string")
        })?;
        let RespValue::Array(Some(fields)) = fields_frame else {
            return Err(malformed(command));
        };
        if fields.len() % 2 != 0 {
            return Err(Error::transport(
                "redis transport: a stream entry carried an odd number of field values",
            ));
        }
        let mut map = BTreeMap::new();
        let mut fields = fields.into_iter();
        while let (Some(name), Some(value)) = (fields.next(), fields.next()) {
            let Some(name) = name.as_string() else {
                continue;
            };
            let Some(value) = value.into_bytes() else {
                continue;
            };
            map.insert(name, value);
        }
        entries.push(StreamEntry { id, fields: map });
    }
    Ok(entries)
}

/// Builds the transport error used for a reply of the wrong shape.
fn unexpected(command: &str, frame: &RespValue) -> Error {
    Error::transport(format!(
        "redis transport: unexpected {command} reply: {frame:?}"
    ))
}

/// Builds the transport error used when a reply is an array of the wrong shape.
fn malformed(command: &str) -> Error {
    Error::transport(format!("redis transport: malformed {command} reply"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Builds the `[id, [field, value, ...]]` entry frame used by both replies.
    fn entry(id: &str, fields: &[(&str, &str)]) -> RespValue {
        let mut flat = Vec::new();
        for (name, value) in fields {
            flat.push(RespValue::SimpleString((*name).to_owned()));
            flat.push(RespValue::SimpleString((*value).to_owned()));
        }
        RespValue::Array(Some(vec![
            RespValue::SimpleString(id.to_owned()),
            RespValue::Array(Some(flat)),
        ]))
    }

    #[test]
    fn key_scheme_is_stable() {
        assert_eq!(
            stream_key("openjiuwen:team", "t1", "p2p", "agent"),
            "openjiuwen:team:t1:p2p:agent"
        );
        assert_eq!(
            stream_key("openjiuwen:team", "t1", "reply", "m1"),
            "openjiuwen:team:t1:reply:m1"
        );
        assert_eq!(
            stream_key("openjiuwen:team", "t1", "pubsub", "sub"),
            "openjiuwen:team:t1:pubsub:sub"
        );
    }

    #[test]
    fn group_names_are_derived_per_key_family() {
        assert_eq!(group_name("team", "t1", "p2p", None), "team:t1:p2p:cg");
        assert_eq!(
            group_name("team", "t1", "pubsub", None),
            "team:t1:pubsub:cg"
        );
        // One explicit group name covers every family; an empty one is ignored.
        assert_eq!(group_name("team", "t1", "p2p", Some("shared")), "shared");
        assert_eq!(group_name("team", "t1", "p2p", Some("")), "team:t1:p2p:cg");
    }

    #[test]
    fn reply_frames_round_trip() {
        let ok = ReplyFrame::from_value(serde_json::json!({"a": 1}));
        let encoded = ok.encode().unwrap();
        let decoded = decode_reply(&encoded).unwrap();
        assert_eq!(decoded.into_result().unwrap(), serde_json::json!({"a": 1}));

        let failed = ReplyFrame::from_error("boom");
        let decoded = decode_reply(&failed.encode().unwrap()).unwrap();
        let error = decoded.into_result().unwrap_err();
        assert_eq!(
            error.message(),
            "redis transport: remote handler failed: boom"
        );

        assert!(decode_reply(b"not json").is_none());
    }

    #[test]
    fn reads_decode_the_stream_reply_shape() {
        let reply = RespValue::Array(Some(vec![RespValue::Array(Some(vec![
            RespValue::SimpleString("key".to_owned()),
            RespValue::Array(Some(vec![
                entry("1-0", &[("envelope", "{\"n\":1}")]),
                entry("2-0", &[("envelope", "{\"n\":2}")]),
            ])),
        ]))]));
        let entries = parse_streams("XREAD", reply).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "1-0");
        assert_eq!(entries[0].field("envelope"), Some(&b"{\"n\":1}"[..]));
        assert_eq!(entries[1].id, "2-0");
    }

    #[test]
    fn empty_reads_decode_to_no_entries() {
        for reply in [
            RespValue::Array(None),
            RespValue::Null,
            RespValue::Array(Some(Vec::new())),
        ] {
            assert!(parse_streams("XREAD", reply).unwrap().is_empty());
        }
        assert!(parse_entry_list("XAUTOCLAIM", RespValue::Array(None))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn malformed_replies_are_transport_errors() {
        let error = parse_streams("XREAD", RespValue::Integer(1)).unwrap_err();
        assert!(
            error.message().contains("unexpected XREAD reply"),
            "{error}"
        );

        let error = parse_entry_list(
            "XAUTOCLAIM",
            RespValue::Array(Some(vec![RespValue::Array(Some(vec![
                RespValue::SimpleString("1-0".to_owned()),
                RespValue::SimpleString("not-a-fields-array".to_owned()),
            ]))])),
        )
        .unwrap_err();
        assert!(error.message().contains("XAUTOCLAIM"), "{error}");
    }
}
