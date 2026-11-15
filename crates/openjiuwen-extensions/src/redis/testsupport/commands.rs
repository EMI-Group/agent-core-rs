//! Command execution for the fake server.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex as StdMutex, MutexGuard};
use std::time::Duration;
use tokio::time::Instant;

use super::support::{
    absolute_expiry_millis, absolute_expiry_seconds, count_to_i64, format_score, matching_keys,
    parse_f64, parse_i64, protocol_error, relative_expiry, syntax_error, ttl_reply, upper,
    wrong_arity, wrong_type,
};
use super::{ClusterMode, ConnState, Data, FakeOptions, ServerState};
use crate::redis::codec::Reply;

impl ServerState {
    pub(super) fn new(options: FakeOptions, addr: std::net::SocketAddr) -> Self {
        Self {
            options,
            addr,
            data: StdMutex::new(Data::default()),
            connections: AtomicUsize::new(0),
            auth_attempts: AtomicUsize::new(0),
            commands: StdMutex::new(Vec::new()),
            connection_commands: StdMutex::new(Vec::new()),
        }
    }
    pub(super) fn data(&self) -> MutexGuard<'_, Data> {
        self.data
            .lock()
            .expect("fake redis state mutex is not poisoned")
    }
    /// Executes one command, returning the reply to write back.
    #[allow(clippy::too_many_lines)]
    pub(super) fn execute(&self, args: &[Vec<u8>], connection: &mut ConnState) -> Reply {
        let Some(name_bytes) = args.first() else {
            return protocol_error("expected a non-empty command");
        };
        let name = upper(name_bytes);
        self.commands
            .lock()
            .expect("fake redis command log mutex is not poisoned")
            .push(name.clone());
        if let Ok(mut logs) = self.connection_commands.lock() {
            if let Some(log) = logs.get_mut(connection.index) {
                log.push(name.clone());
            }
        }
        if self.options.require_auth.is_some() && !connection.authed && name != "AUTH" {
            return Reply::Error("NOAUTH Authentication required.".to_owned());
        }
        if name == "CLUSTER" {
            return self.cluster_reply(args);
        }
        if let Some(redirect) = self.options.redirect.as_ref() {
            if !matches!(name.as_str(), "AUTH" | "SELECT" | "HELLO" | "ASKING") {
                return Reply::Error(format!(
                    "{} {} {}",
                    if redirect.ask { "ASK" } else { "MOVED" },
                    redirect.slot,
                    redirect.target
                ));
            }
        }
        let mut data = self.data();
        data.purge();
        match name.as_str() {
            "ASKING" => Reply::Simple("OK".to_owned()),
            "PING" => match args.len() {
                1 => Reply::Simple("PONG".to_owned()),
                2 => Reply::Bulk(args[1].clone()),
                _ => wrong_arity(&name),
            },
            "ECHO" => {
                if args.len() == 2 {
                    Reply::Bulk(args[1].clone())
                } else {
                    wrong_arity(&name)
                }
            }
            "AUTH" => self.authenticate(args, connection),
            "SELECT" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                let Some(index) = parse_i64(&args[1]) else {
                    return Reply::Error("ERR invalid DB index".to_owned());
                };
                connection.db = Some(index);
                Reply::Simple("OK".to_owned())
            }
            "GET" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.sets.contains_key(key) || data.zsets.contains_key(key) {
                    return wrong_type();
                }
                data.values
                    .get(key)
                    .cloned()
                    .map_or(Reply::Null, Reply::Bulk)
            }
            "SET" => Self::set(&mut data, args),
            "EXISTS" => {
                if args.len() < 2 {
                    return wrong_arity(&name);
                }
                Reply::Integer(count_to_i64(
                    args[1..].iter().filter(|key| data.exists(key)).count(),
                ))
            }
            "DEL" | "UNLINK" => {
                if args.len() < 2 {
                    return wrong_arity(&name);
                }
                Reply::Integer(count_to_i64(
                    args[1..].iter().filter(|key| data.remove(key)).count(),
                ))
            }
            "MGET" => {
                if args.len() < 2 {
                    return wrong_arity(&name);
                }
                Reply::Array(
                    args[1..]
                        .iter()
                        .map(|key| {
                            data.values
                                .get(key.as_slice())
                                .cloned()
                                .map_or(Reply::Null, Reply::Bulk)
                        })
                        .collect(),
                )
            }
            "TTL" | "PTTL" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                ttl_reply(&data, &args[1], name == "PTTL")
            }
            "EXPIRE" => {
                if args.len() != 3 {
                    return wrong_arity(&name);
                }
                let Some(seconds) = parse_i64(&args[2]) else {
                    return Reply::Error("ERR value is not an integer or out of range".to_owned());
                };
                let key = args[1].as_slice();
                if !data.exists(key) {
                    return Reply::Integer(0);
                }
                if seconds <= 0 {
                    data.remove(key);
                    return Reply::Integer(1);
                }
                let at = Instant::now() + Duration::from_secs(u64::try_from(seconds).unwrap_or(0));
                data.expiries.insert(args[1].clone(), at);
                Reply::Integer(1)
            }
            "SCAN" => {
                if args.len() < 2 {
                    return wrong_arity(&name);
                }
                let mut pattern: Option<String> = None;
                let mut index = 2;
                while index < args.len() {
                    let option = upper(&args[index]);
                    match option.as_str() {
                        "MATCH" => {
                            let Some(raw) = args.get(index + 1) else {
                                return syntax_error();
                            };
                            pattern = Some(String::from_utf8_lossy(raw).into_owned());
                            index += 2;
                        }
                        "COUNT" | "TYPE" => {
                            if args.get(index + 1).is_none() {
                                return syntax_error();
                            }
                            index += 2;
                        }
                        _ => return syntax_error(),
                    }
                }
                let keys = matching_keys(&data, pattern.as_deref())
                    .into_iter()
                    .map(Reply::Bulk)
                    .collect();
                Reply::Array(vec![Reply::Bulk(b"0".to_vec()), Reply::Array(keys)])
            }
            "KEYS" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                let pattern = String::from_utf8_lossy(&args[1]).into_owned();
                Reply::Array(
                    matching_keys(&data, Some(&pattern))
                        .into_iter()
                        .map(Reply::Bulk)
                        .collect(),
                )
            }
            "SADD" => {
                if args.len() < 3 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.zsets.contains_key(key) {
                    return wrong_type();
                }
                let set = data.sets.entry(args[1].clone()).or_default();
                let mut added = 0;
                for member in &args[2..] {
                    if set.insert(member.clone()) {
                        added += 1;
                    }
                }
                Reply::Integer(count_to_i64(added))
            }
            "SREM" => {
                if args.len() < 3 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.zsets.contains_key(key) {
                    return wrong_type();
                }
                let removed = data.sets.get_mut(key).map_or(0, |set| {
                    args[2..]
                        .iter()
                        .filter(|member| set.remove(member.as_slice()))
                        .count()
                });
                if data.sets.get(key).is_some_and(BTreeSet::is_empty) {
                    data.sets.remove(key);
                }
                Reply::Integer(count_to_i64(removed))
            }
            "SMEMBERS" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.zsets.contains_key(key) {
                    return wrong_type();
                }
                Reply::Array(data.sets.get(key).map_or_else(Vec::new, |set| {
                    set.iter().cloned().map(Reply::Bulk).collect()
                }))
            }
            "SISMEMBER" => {
                if args.len() != 3 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.zsets.contains_key(key) {
                    return wrong_type();
                }
                Reply::Integer(i64::from(
                    data.sets
                        .get(key)
                        .is_some_and(|set| set.contains(args[2].as_slice())),
                ))
            }
            "ZCARD" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.sets.contains_key(key) {
                    return wrong_type();
                }
                Reply::Integer(
                    data.zsets
                        .get(key)
                        .map_or(0, |zset| count_to_i64(zset.len())),
                )
            }
            "ZADD" => {
                if args.len() < 4 || (args.len() - 2) % 2 != 0 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.sets.contains_key(key) {
                    return wrong_type();
                }
                let mut added = 0;
                let mut index = 2;
                while index + 1 < args.len() {
                    let Some(score) = parse_f64(&args[index]) else {
                        return Reply::Error("ERR value is not a valid float".to_owned());
                    };
                    let member = args[index + 1].clone();
                    if data
                        .zsets
                        .entry(args[1].clone())
                        .or_default()
                        .insert(member, score)
                        .is_none()
                    {
                        added += 1;
                    }
                    index += 2;
                }
                Reply::Integer(count_to_i64(added))
            }
            "ZRANGE" => Self::zrange(&data, args),
            "ZREM" => {
                if args.len() < 3 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                if data.values.contains_key(key) || data.sets.contains_key(key) {
                    return wrong_type();
                }
                let removed = data.zsets.get_mut(key).map_or(0, |zset| {
                    args[2..]
                        .iter()
                        .filter(|member| zset.remove(member.as_slice()).is_some())
                        .count()
                });
                if data.zsets.get(key).is_some_and(BTreeMap::is_empty) {
                    data.zsets.remove(key);
                }
                Reply::Integer(count_to_i64(removed))
            }
            "FLUSHDB" => {
                data.values.clear();
                data.sets.clear();
                data.zsets.clear();
                data.expiries.clear();
                Reply::Simple("OK".to_owned())
            }
            "TYPE" => {
                if args.len() != 2 {
                    return wrong_arity(&name);
                }
                let key = args[1].as_slice();
                let kind = if data.values.contains_key(key) {
                    "string"
                } else if data.sets.contains_key(key) {
                    "set"
                } else if data.zsets.contains_key(key) {
                    "zset"
                } else {
                    "none"
                };
                Reply::Simple(kind.to_owned())
            }
            _ => Reply::Error(format!("ERR unknown command '{name}'")),
        }
    }
    /// Handles `SET` (including `NX`/`XX`/`EX`/`PX`/`EXAT`/`PXAT`/`KEEPTTL`).
    fn set(data: &mut Data, args: &[Vec<u8>]) -> Reply {
        if args.len() < 3 {
            return wrong_arity("SET");
        }
        let key = args[1].clone();
        let value = args[2].clone();
        let mut nx = false;
        let mut xx = false;
        let mut keep_ttl = false;
        let mut new_expiry: Option<Instant> = None;
        let mut index = 3;
        while index < args.len() {
            match upper(&args[index]).as_str() {
                "NX" => {
                    nx = true;
                    index += 1;
                }
                "XX" => {
                    xx = true;
                    index += 1;
                }
                "KEEPTTL" => {
                    keep_ttl = true;
                    index += 1;
                }
                "EX" | "PX" | "EXAT" | "PXAT" => {
                    let Some(raw) = args.get(index + 1) else {
                        return syntax_error();
                    };
                    let Some(amount) = parse_i64(raw) else {
                        return Reply::Error(
                            "ERR value is not an integer or out of range".to_owned(),
                        );
                    };
                    let option = upper(&args[index]);
                    new_expiry = Some(match option.as_str() {
                        "EX" => {
                            relative_expiry(Duration::from_secs(u64::try_from(amount).unwrap_or(0)))
                        }
                        "PX" => relative_expiry(Duration::from_millis(
                            u64::try_from(amount).unwrap_or(0),
                        )),
                        "EXAT" => absolute_expiry_seconds(amount),
                        _ => absolute_expiry_millis(amount),
                    });
                    index += 2;
                }
                _ => return syntax_error(),
            }
        }
        let exists = data.exists(&key);
        if (nx && exists) || (xx && !exists) {
            return Reply::Null;
        }
        data.sets.remove(&key);
        data.zsets.remove(&key);
        data.values.insert(key.clone(), value);
        if let Some(expiry) = new_expiry {
            data.expiries.insert(key, expiry);
        } else if !keep_ttl {
            // Plain `SET` drops any previous TTL; the kernel `KvStore` contract
            // depends on this.
            data.expiries.remove(&key);
        }
        Reply::Simple("OK".to_owned())
    }
    /// Handles `CLUSTER INFO` / `CLUSTER SLOTS` for the scripted cluster modes.
    ///
    /// A node without a [`ClusterMode`] answers like a standalone server: `INFO`
    /// reports `cluster_enabled:0` and `SLOTS` is an error.
    fn cluster_reply(&self, args: &[Vec<u8>]) -> Reply {
        let subcommand = args.get(1).map(|arg| upper(arg)).unwrap_or_default();
        match (subcommand.as_str(), self.options.cluster.as_ref()) {
            ("INFO", Some(_)) => Reply::Bulk(
                b"cluster_enabled:1\r\ncluster_state:ok\r\ncluster_slots_assigned:16384\r\n"
                    .to_vec(),
            ),
            ("INFO", None) => Reply::Bulk(b"cluster_enabled:0\r\ncluster_state:fail\r\n".to_vec()),
            ("SLOTS", Some(mode)) => {
                let (host, port) = match mode {
                    ClusterMode::SelfOwned => (self.addr.ip().to_string(), self.addr.port()),
                    ClusterMode::OwnedBy(host, port) => (host.clone(), *port),
                };
                Reply::Array(vec![Reply::Array(vec![
                    Reply::Integer(0),
                    Reply::Integer(16383),
                    Reply::Array(vec![
                        Reply::Bulk(host.into_bytes()),
                        Reply::Integer(i64::from(port)),
                    ]),
                ])])
            }
            ("SLOTS", None) => {
                Reply::Error("ERR This instance has cluster support disabled".to_owned())
            }
            _ => Reply::Error(format!("ERR Unknown CLUSTER subcommand '{subcommand}'")),
        }
    }
    /// Handles `ZRANGE key start stop [WITHSCORES]`.
    fn zrange(data: &Data, args: &[Vec<u8>]) -> Reply {
        if args.len() < 4 || args.len() > 5 {
            return wrong_arity("ZRANGE");
        }
        let with_scores = args
            .get(4)
            .is_some_and(|option| upper(option) == "WITHSCORES");
        if args.len() == 5 && !with_scores {
            return syntax_error();
        }
        let (Some(start), Some(stop)) = (parse_i64(&args[2]), parse_i64(&args[3])) else {
            return Reply::Error("ERR value is not an integer or out of range".to_owned());
        };
        let mut entries: Vec<(Vec<u8>, f64)> =
            data.zsets
                .get(args[1].as_slice())
                .map_or_else(Vec::new, |zset| {
                    zset.iter()
                        .map(|(member, score)| (member.clone(), *score))
                        .collect()
                });
        entries.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        let length = count_to_i64(entries.len());
        let resolve = |index: i64| if index < 0 { length + index } else { index };
        let start = resolve(start).max(0);
        let stop = resolve(stop).min(length - 1);
        if stop < start || start >= length {
            return Reply::Array(Vec::new());
        }
        let mut out = Vec::new();
        for (member, score) in
            &entries[usize::try_from(start).unwrap_or(0)..=usize::try_from(stop).unwrap_or(0)]
        {
            out.push(Reply::Bulk(member.clone()));
            if with_scores {
                out.push(Reply::Bulk(format_score(*score)));
            }
        }
        Reply::Array(out)
    }
    /// Handles `AUTH` for both the any-password default and the required mode.
    fn authenticate(&self, args: &[Vec<u8>], connection: &mut ConnState) -> Reply {
        self.auth_attempts.fetch_add(1, Ordering::Relaxed);
        let Some((expected_user, expected_password)) = self.options.require_auth.as_ref() else {
            return if args.len() == 2 || args.len() == 3 {
                connection.authed = true;
                Reply::Simple("OK".to_owned())
            } else {
                wrong_arity("AUTH")
            };
        };
        let (username, password) = match args.len() {
            2 => (None, &args[1]),
            3 => (Some(&args[1]), &args[2]),
            _ => return wrong_arity("AUTH"),
        };
        let user_matches = match (expected_user, username) {
            (None, None) => true,
            (Some(expected), Some(actual)) => expected.as_bytes() == actual.as_slice(),
            _ => false,
        };
        if user_matches && expected_password.as_bytes() == password.as_slice() {
            connection.authed = true;
            Reply::Simple("OK".to_owned())
        } else {
            Reply::Error("WRONGPASS invalid username-password pair or user is disabled.".to_owned())
        }
    }
}
