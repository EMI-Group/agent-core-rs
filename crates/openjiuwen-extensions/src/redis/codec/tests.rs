#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]
use super::*;
use openjiuwen_common::error::ErrorKind;
fn parse(bytes: &[u8]) -> Reply {
    let (reply, consumed) = parse_reply(bytes)
        .expect("sample is valid RESP")
        .expect("sample is a complete reply");
    assert_eq!(consumed, bytes.len(), "sample must consume every byte");
    reply
}
#[test]
fn encode_empty_argument_list() {
    assert_eq!(encode_command(&[]), b"*0\r\n");
}
#[test]
fn encode_single_and_multi_argument_commands() {
    assert_eq!(encode_command(&[b"PING"]), b"*1\r\n$4\r\nPING\r\n".to_vec());
    assert_eq!(
        encode_command(&[b"GET", b"user:1"]),
        b"*2\r\n$3\r\nGET\r\n$6\r\nuser:1\r\n".to_vec()
    );
    // Binary arguments keep their exact bytes, including CRLF and NUL.
    let binary: &[u8] = b"a\x00\r\nb\xff";
    let encoded = encode_command(&[b"SET", b"k", binary]);
    let mut expected = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$6\r\n".to_vec();
    expected.extend_from_slice(binary);
    expected.extend_from_slice(b"\r\n");
    assert_eq!(encoded, expected);
}
#[test]
fn decode_resp2_scalars() {
    assert_eq!(parse(b"+OK\r\n"), Reply::Simple("OK".to_owned()));
    assert_eq!(
        parse(b"-ERR unknown\r\n"),
        Reply::Error("ERR unknown".to_owned())
    );
    assert_eq!(parse(b":42\r\n"), Reply::Integer(42));
    assert_eq!(parse(b":-7\r\n"), Reply::Integer(-7));
    assert_eq!(parse(b"$5\r\nhello\r\n"), Reply::Bulk(b"hello".to_vec()));
    assert_eq!(parse(b"$0\r\n\r\n"), Reply::Bulk(Vec::new()));
    assert_eq!(
        parse(b"$5\r\n\x00\r\n\xffz\r\n"),
        Reply::Bulk(b"\x00\r\n\xffz".to_vec())
    );
}
#[test]
fn decode_nulls_and_empty_aggregates() {
    assert_eq!(parse(b"$-1\r\n"), Reply::Null);
    assert_eq!(parse(b"*-1\r\n"), Reply::Null);
    assert_eq!(parse(b"*0\r\n"), Reply::Array(Vec::new()));
    assert_eq!(parse(b"~0\r\n"), Reply::Set(Vec::new()));
    assert_ne!(parse(b"*0\r\n"), Reply::Null);
}
#[test]
fn decode_nested_arrays_and_consumed_length() {
    let nested = b"*3\r\n$3\r\nfoo\r\n:7\r\n*2\r\n+OK\r\n$-1\r\n";
    let (reply, consumed) = parse_reply(nested).unwrap().unwrap();
    assert_eq!(consumed, nested.len());
    assert_eq!(
        reply,
        Reply::Array(vec![
            Reply::Bulk(b"foo".to_vec()),
            Reply::Integer(7),
            Reply::Array(vec![Reply::Simple("OK".to_owned()), Reply::Null]),
        ])
    );
}
#[test]
fn decode_resp3_types() {
    assert_eq!(parse(b"_\r\n"), Reply::Null);
    assert_eq!(parse(b"#t\r\n"), Reply::Boolean(true));
    assert_eq!(parse(b"#f\r\n"), Reply::Boolean(false));
    assert_eq!(parse(b",3.5\r\n"), Reply::Double(3.5));
    assert_eq!(parse(b",inf\r\n"), Reply::Double(f64::INFINITY));
    assert_eq!(parse(b"(42\r\n"), Reply::Integer(42));
    assert_eq!(
        parse(b"(123456789012345678901234567890\r\n"),
        Reply::Bulk(b"123456789012345678901234567890".to_vec())
    );
    assert_eq!(
        parse(b"=15\r\ntxt:Some string\r\n"),
        Reply::Bulk(b"Some string".to_vec())
    );
    assert_eq!(
        parse(b"~2\r\n+a\r\n+b\r\n"),
        Reply::Set(vec![
            Reply::Simple("a".to_owned()),
            Reply::Simple("b".to_owned())
        ])
    );
    assert_eq!(
        parse(b">1\r\n+x\r\n"),
        Reply::Push(vec![Reply::Simple("x".to_owned())])
    );
    assert_eq!(
        parse(b"%1\r\n+key\r\n:1\r\n"),
        Reply::Map(vec![(Reply::Simple("key".to_owned()), Reply::Integer(1))])
    );
}
#[test]
fn decode_moved_and_ask_stay_error_replies() {
    assert_eq!(
        parse(b"-MOVED 3999 127.0.0.1:7000\r\n"),
        Reply::Error("MOVED 3999 127.0.0.1:7000".to_owned())
    );
    assert_eq!(
        parse(b"-ASK 12 127.0.0.1:7001\r\n"),
        Reply::Error("ASK 12 127.0.0.1:7001".to_owned())
    );
}
#[test]
fn unsupported_resp3_constructs_are_typed() {
    let attribute = parse_reply(b"|1\r\n+x\r\n:1\r\n").unwrap_err();
    assert_eq!(attribute.kind(), ErrorKind::Unsupported);
    let streamed = parse_reply(b";4\r\nabcd\r\n").unwrap_err();
    assert_eq!(streamed.kind(), ErrorKind::Unsupported);
}
#[test]
fn every_truncation_of_a_complete_stream_is_incomplete() {
    let samples: &[&[u8]] = &[
        b"+OK\r\n",
        b"-ERR bad\r\n",
        b":-42\r\n",
        b"$5\r\nhello\r\n",
        b"$0\r\n\r\n",
        b"$-1\r\n",
        b"*-1\r\n",
        b"*0\r\n",
        b"*2\r\n$3\r\nfoo\r\n:7\r\n",
        b"%1\r\n+key\r\n#t\r\n",
        b"~2\r\n+a\r\n+b\r\n",
        b">1\r\n+x\r\n",
        b"=15\r\ntxt:Some string\r\n",
        b"_\r\n",
        b"#f\r\n",
        b",2.5\r\n",
        b"(42\r\n",
    ];
    for sample in samples {
        for length in 0..sample.len() {
            assert!(
                matches!(parse_reply(&sample[..length]), Ok(None)),
                "prefix of {sample:?} at {length} bytes must be incomplete"
            );
        }
    }
}
#[test]
fn malformed_framing_is_rejected() {
    let cases: &[&[u8]] = &[
        b"+OK\n\r\n",         // bare line feed inside a line
        b"+OK\rX",            // carriage return without line feed
        b":abc\r\n",          // non-numeric integer
        b"$3\r\nabcXY",       // bulk payload not terminated by CRLF
        b"$-2\r\n",           // negative bulk length other than -1
        b"*-2\r\n",           // negative aggregate length other than -1
        b"%-1\r\n",           // negative map length
        b"#x\r\n",            // invalid boolean
        b",not-a-double\r\n", // invalid double
        b"_x\r\n",            // null must be an empty line
        b"?raw\r\n",          // unknown type byte
    ];
    for case in cases {
        let error = parse_reply(case).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport, "case {case:?}");
    }
}
#[test]
fn garbage_input_never_panics() {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..2_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let length = (state >> 33) as usize % 64;
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            bytes.push((state >> 40) as u8);
        }
        let _ = parse_reply(&bytes);
    }
}
#[test]
fn accessors_read_strings_and_integers() {
    let bulk = Reply::Bulk(b"value".to_vec());
    assert_eq!(bulk.as_str().unwrap(), "value");
    assert_eq!(bulk.as_opt_str().unwrap(), Some("value"));
    assert_eq!(bulk.as_string_owned().unwrap(), "value");
    assert_eq!(
        bulk.as_opt_string_owned().unwrap().as_deref(),
        Some("value")
    );
    assert_eq!(Reply::Null.as_opt_str().unwrap(), None);
    assert_eq!(Reply::Null.as_opt_bytes().unwrap(), None);
    assert_eq!(Reply::Boolean(true).as_integer().unwrap(), 1);
    assert_eq!(Reply::Boolean(false).as_integer().unwrap(), 0);
    assert!(Reply::Boolean(true).as_bool().unwrap());
    assert_eq!(Reply::Null.as_opt_str().unwrap(), None);
    assert_eq!(
        Reply::Array(vec![Reply::Integer(1)])
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(Reply::Set(vec![Reply::Null]).as_array().unwrap().len(), 1);
    assert_eq!(
        Reply::Map(vec![(Reply::Simple("k".into()), Reply::Integer(1))])
            .as_map()
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn accessors_reject_type_mismatches() {
    assert_eq!(
        Reply::Integer(1).as_str().unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        Reply::Simple("x".into()).as_integer().unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        Reply::Bulk(vec![0xff]).as_str().unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        Reply::Null.as_array().unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        Reply::Array(Vec::new()).as_map().unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert!(Reply::Error("ERR".into()).is_error());
    assert!(Reply::Null.is_null());
}
