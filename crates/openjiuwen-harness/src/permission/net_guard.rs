//! Fetch-tool network policy (pipeline C).
//!
//! Ports `openjiuwen/harness/security/permission_engine/netguard/net_guard.py`
//! (v0.1.20): the [`NetGuardChecker`] that inspects the URL argument of a fetch
//! tool and can only ever **DENY**. It never allows and never asks — a network
//! guard cannot widen a pipeline-A decision, it can only narrow it. The engine
//! folds its verdict with [`crate::permission::engine::strictest`].
//!
//! The policy has two levels of configuration, both read from the already-parsed
//! [`PermissionsSection`]:
//!
//! 1. `net_guard.urls` — URL patterns (see
//!    [`crate::permission::patterns::match_net_pattern`]: literal prefix with an
//!    authority-equality check, full URL glob, or hostname wildcard) mapped to
//!    `allow`/`deny`. `ask` entries are dropped at build time, exactly like
//!    Python, because this guard has no ask state.
//! 2. `net_guard.defaults` — the decision for a URL that hits no entry. Only
//!    `deny` blocks; `allow` and `ask` both mean "no opinion". A matching `allow`
//!    entry therefore bypasses `defaults = deny`, while any matching `deny` entry
//!    wins over any `allow` entry.
//!
//! Everything here is pure and in-memory: no IO, no locks, no panics, and no
//! logging beyond what the engine reports. Rules are stored in `BTreeMap` key
//! order, which is deterministic; the Python reference used insertion order, so
//! when several deny rules match, the reported `matched_rule` is the first one in
//! sorted order instead of the first one in the YAML. The decision itself is
//! unaffected.

use std::collections::BTreeMap;

use serde_json::Value;

use super::model::{PermissionLevel, PermissionResult, PermissionsSection};
use super::patterns::match_net_pattern;
use super::rules::normalize_tool_name;

/// Fetch tools the guard applies to (canonical Python set).
///
/// Model-facing aliases (`fetch_webpage`, `web_fetch_webpage`) are folded onto
/// `mcp_fetch_webpage` by [`normalize_tool_name`] before the lookup, so listing
/// the canonical names is enough.
pub const FETCH_TOOLS: &[&str] = &["mcp_fetch_webpage", "fetch_webpage", "web_fetch_webpage"];

/// Argument keys that may carry the URL, in Python's precedence order.
///
/// The first key holding a non-blank string wins, so `{"url": …, "href": …}`
/// always evaluates `url`.
pub const URL_ARG_KEYS: &[&str] = &["url", "uri", "href", "webpage", "page_url", "target_url"];

/// First non-blank string among [`URL_ARG_KEYS`], backslashes normalized to `/`.
///
/// Returns `None` when `args` is not a JSON object or holds no usable URL value
/// (missing, blank, or non-string). Python parity: `extract_fetch_url`.
#[must_use]
pub fn extract_fetch_url(args: &Value) -> Option<String> {
    let object = args.as_object()?;
    URL_ARG_KEYS.iter().find_map(|key| {
        let raw = object.get(*key)?.as_str()?.trim();
        if raw.is_empty() {
            None
        } else {
            Some(raw.replace('\\', "/"))
        }
    })
}

/// Network deny/allow policy for fetch tools.
///
/// Built from an *enabled* `net_guard` section; a disabled or absent section
/// yields no checker at all, so the pipeline never pays for the URL parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetGuardChecker {
    /// Decision for URLs that hit no `urls` entry; only `deny` blocks.
    defaults: PermissionLevel,
    /// Compiled URL rules, `ask` entries already dropped.
    urls: BTreeMap<String, PermissionLevel>,
}

impl NetGuardChecker {
    /// Builds the checker, or `None` when `section.net_guard` is disabled.
    ///
    /// Entries whose level is [`PermissionLevel::Ask`] are dropped (Python parity:
    /// the net guard has no ask), as are blank patterns. Patterns are trimmed.
    #[must_use]
    pub fn build(section: &PermissionsSection) -> Option<Self> {
        let net = &section.net_guard;
        if !net.enabled {
            return None;
        }
        let urls = net
            .urls
            .iter()
            .filter(|(_, level)| **level != PermissionLevel::Ask)
            .filter_map(|(pattern, level)| {
                let pattern = pattern.trim();
                (!pattern.is_empty()).then(|| (pattern.to_owned(), *level))
            })
            .collect();
        Some(Self {
            defaults: net.defaults,
            urls,
        })
    }

    /// Evaluates the network policy for one tool call.
    ///
    /// Returns `None` when the guard has no opinion (non-fetch tool, a matching
    /// allow rule, no matching rule and `defaults` is not `deny`), and
    /// `Some(deny)` when the URL is blocked. It never returns `allow` or `ask`.
    #[must_use]
    pub fn evaluate(&self, tool_name: &str, args: &Value) -> Option<PermissionResult> {
        let tool = normalize_tool_name(tool_name.trim());
        if !FETCH_TOOLS.contains(&tool.as_str()) {
            return None;
        }
        let mut allowed = false;
        if let Some(url) = extract_fetch_url(args) {
            for (pattern, level) in &self.urls {
                if !match_net_pattern(&url, pattern) {
                    continue;
                }
                match level {
                    PermissionLevel::Deny => return Some(deny_url(pattern)),
                    PermissionLevel::Allow => allowed = true,
                    // Dropped by `build`; kept for exhaustiveness.
                    PermissionLevel::Ask => {}
                }
            }
        }
        if allowed {
            return None;
        }
        if self.defaults == PermissionLevel::Deny {
            return Some(
                PermissionResult::new(PermissionLevel::Deny)
                    .with_matched_rule("net_guard:defaults")
                    .with_reason("net_guard denied: defaults"),
            );
        }
        None
    }
}

/// The deny result of one matching URL rule (Python label and reason templates).
fn deny_url(pattern: &str) -> PermissionResult {
    PermissionResult::new(PermissionLevel::Deny)
        .with_matched_rule(format!("net_guard:url:{pattern}"))
        .with_reason(format!("net_guard denied: {pattern}"))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;
    use crate::permission::model::NetGuardSection;

    /// A permissions section carrying the given `net_guard` policy.
    fn section(
        enabled: bool,
        defaults: PermissionLevel,
        urls: &[(&str, PermissionLevel)],
    ) -> PermissionsSection {
        PermissionsSection {
            net_guard: NetGuardSection {
                enabled,
                defaults,
                urls: urls
                    .iter()
                    .map(|(pattern, level)| ((*pattern).to_owned(), *level))
                    .collect(),
            },
            ..PermissionsSection::default()
        }
    }

    /// A checker for an enabled `net_guard` with the given policy.
    fn checker(defaults: PermissionLevel, urls: &[(&str, PermissionLevel)]) -> NetGuardChecker {
        NetGuardChecker::build(&section(true, defaults, urls)).unwrap()
    }

    /// The URL argument the tests evaluate.
    fn fetch(url: &str) -> Value {
        json!({ "url": url })
    }

    // ----- build ----------------------------------------------------------

    #[test]
    fn build_requires_an_enabled_net_guard() {
        assert!(NetGuardChecker::build(&PermissionsSection::default()).is_none());
        let disabled = section(
            false,
            PermissionLevel::Deny,
            &[("https://blocked.example", PermissionLevel::Deny)],
        );
        assert!(NetGuardChecker::build(&disabled).is_none());
    }

    #[test]
    fn build_drops_ask_rules_and_trims_patterns() {
        let built = checker(
            PermissionLevel::Allow,
            &[
                (" https://blocked.example ", PermissionLevel::Deny),
                ("https://asked.example", PermissionLevel::Ask),
                ("   ", PermissionLevel::Deny),
            ],
        );
        let result = built
            .evaluate("fetch_webpage", &fetch("https://blocked.example/x"))
            .expect("a trimmed pattern still denies");
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("net_guard:url:https://blocked.example")
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://asked.example/x")),
            None
        );
    }

    // ----- tool gate ------------------------------------------------------

    #[test]
    fn non_fetch_tools_are_never_guarded() {
        let built = checker(PermissionLevel::Deny, &[]);
        for tool in ["mcp_free_search", "free_search", "bash", "read_file"] {
            assert_eq!(built.evaluate(tool, &fetch("https://example.com/")), None);
        }
    }

    #[test]
    fn fetch_tool_aliases_are_guarded() {
        let built = checker(PermissionLevel::Deny, &[]);
        for tool in ["mcp_fetch_webpage", "fetch_webpage", "web_fetch_webpage"] {
            let result = built
                .evaluate(tool, &fetch("https://example.com/"))
                .expect("defaults deny blocks every fetch tool");
            assert!(result.is_denied());
            assert_eq!(result.matched_rule.as_deref(), Some("net_guard:defaults"));
        }
    }

    // ----- defaults -------------------------------------------------------

    #[test]
    fn defaults_allow_without_a_hit_has_no_opinion() {
        let built = checker(PermissionLevel::Allow, &[]);
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://example.com/")),
            None
        );
    }

    #[test]
    fn defaults_deny_without_a_hit_denies() {
        let built = checker(PermissionLevel::Deny, &[]);
        let result = built
            .evaluate("fetch_webpage", &fetch("https://example.com/"))
            .expect("defaults deny must block");
        assert_eq!(result.permission, PermissionLevel::Deny);
        assert_eq!(result.matched_rule.as_deref(), Some("net_guard:defaults"));
        assert_eq!(result.reason.as_deref(), Some("net_guard denied: defaults"));
    }

    #[test]
    fn ask_defaults_behave_like_allow() {
        let built = checker(PermissionLevel::Ask, &[]);
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://example.com/")),
            None
        );
    }

    #[test]
    fn missing_url_falls_through_to_defaults() {
        let built = checker(PermissionLevel::Deny, &[]);
        let result = built
            .evaluate("fetch_webpage", &json!({}))
            .expect("a missing URL hits no pattern and defaults deny");
        assert_eq!(result.matched_rule.as_deref(), Some("net_guard:defaults"));
    }

    // ----- hostname rules -------------------------------------------------

    #[test]
    fn metadata_and_loopback_hosts_are_denied() {
        let built = checker(
            PermissionLevel::Allow,
            &[
                ("169.254.169.254", PermissionLevel::Deny),
                ("localhost", PermissionLevel::Deny),
                ("127.0.0.1", PermissionLevel::Deny),
            ],
        );
        let metadata = built
            .evaluate(
                "fetch_webpage",
                &fetch("http://169.254.169.254/latest/meta-data/"),
            )
            .expect("the metadata endpoint is denied");
        assert_eq!(
            metadata.matched_rule.as_deref(),
            Some("net_guard:url:169.254.169.254")
        );
        assert_eq!(
            metadata.reason.as_deref(),
            Some("net_guard denied: 169.254.169.254")
        );

        for url in [
            "http://localhost/",
            "http://localhost:80/x",
            "http://localhost:3000/api",
            "http://127.0.0.1/health",
            "http://127.0.0.1:443/health",
        ] {
            let result = built
                .evaluate("fetch_webpage", &fetch(url))
                .unwrap_or_else(|| panic!("{url} must be denied"));
            assert!(result.is_denied(), "{url}");
        }

        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://example.com/")),
            None
        );
    }

    #[test]
    fn hostname_glob_denies_subdomains_only() {
        let built = checker(
            PermissionLevel::Allow,
            &[("*.internal.example", PermissionLevel::Deny)],
        );
        let result = built
            .evaluate("fetch_webpage", &fetch("https://api.internal.example/v1"))
            .expect("a subdomain is denied");
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("net_guard:url:*.internal.example")
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://example.com/")),
            None
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://internal.example/")),
            None
        );
    }

    // ----- URL rules ------------------------------------------------------

    #[test]
    fn url_glob_rule_denies_matching_paths() {
        let built = checker(
            PermissionLevel::Allow,
            &[("https://evil.example/admin*", PermissionLevel::Deny)],
        );
        let result = built
            .evaluate("fetch_webpage", &fetch("https://evil.example/admin/users"))
            .expect("the admin glob denies");
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("net_guard:url:https://evil.example/admin*")
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://evil.example/other")),
            None
        );
    }

    #[test]
    fn prefix_rule_denies_paths_below_it() {
        let built = checker(
            PermissionLevel::Allow,
            &[("https://blocked.example/secret", PermissionLevel::Deny)],
        );
        let result = built
            .evaluate(
                "fetch_webpage",
                &fetch("https://blocked.example/secret/token"),
            )
            .expect("the prefix denies");
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("net_guard:url:https://blocked.example/secret")
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://blocked.example/public")),
            None
        );
    }

    #[test]
    fn allow_rule_bypasses_defaults_deny_but_not_lookalikes() {
        let built = checker(
            PermissionLevel::Deny,
            &[("https://example.com", PermissionLevel::Allow)],
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://example.com/path")),
            None
        );
        for url in [
            "https://example.com.evil.com/x",
            "https://example.com@evil.com/x",
            "https://example.com:8443/x",
        ] {
            let result = built
                .evaluate("fetch_webpage", &fetch(url))
                .unwrap_or_else(|| panic!("{url} must hit defaults deny"));
            assert_eq!(result.matched_rule.as_deref(), Some("net_guard:defaults"));
        }
    }

    #[test]
    fn deny_rule_beats_a_matching_allow_rule() {
        let built = checker(
            PermissionLevel::Allow,
            &[
                ("https://example.com", PermissionLevel::Allow),
                ("https://example.com/admin*", PermissionLevel::Deny),
            ],
        );
        let result = built
            .evaluate("fetch_webpage", &fetch("https://example.com/admin/users"))
            .expect("any matching deny wins");
        assert_eq!(result.permission, PermissionLevel::Deny);
        assert_eq!(
            result.matched_rule.as_deref(),
            Some("net_guard:url:https://example.com/admin*")
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://example.com/public")),
            None
        );
    }

    #[test]
    fn ask_url_entries_are_dropped() {
        let built = checker(
            PermissionLevel::Allow,
            &[("https://blocked.example", PermissionLevel::Ask)],
        );
        assert_eq!(
            built.evaluate("fetch_webpage", &fetch("https://blocked.example/x")),
            None
        );

        let strict = checker(
            PermissionLevel::Deny,
            &[("https://blocked.example", PermissionLevel::Ask)],
        );
        let result = strict
            .evaluate("fetch_webpage", &fetch("https://blocked.example/x"))
            .expect("the dropped ask entry leaves only defaults");
        assert_eq!(result.matched_rule.as_deref(), Some("net_guard:defaults"));
    }

    // ----- extract_fetch_url ----------------------------------------------

    #[test]
    fn extract_fetch_url_prefers_the_first_usable_key() {
        assert_eq!(
            extract_fetch_url(
                &json!({"url": " https://a.example/x ", "href": "https://b.example"})
            ),
            Some("https://a.example/x".to_owned())
        );
        assert_eq!(
            extract_fetch_url(&json!({"uri": "https://u.example/", "href": "https://h.example/"})),
            Some("https://u.example/".to_owned())
        );
        assert_eq!(
            extract_fetch_url(&json!({"url": "   ", "href": "https://b.example/ x"})),
            Some("https://b.example/ x".to_owned())
        );
        assert_eq!(
            extract_fetch_url(&json!({"url": 42, "href": "https://b.example"})),
            Some("https://b.example".to_owned())
        );
    }

    #[test]
    fn extract_fetch_url_handles_every_key_and_normalizes_backslashes() {
        for key in URL_ARG_KEYS {
            let args = json!({ *key: "https://example.com/x" });
            assert_eq!(
                extract_fetch_url(&args),
                Some("https://example.com/x".to_owned()),
                "{key}"
            );
        }
        assert_eq!(
            extract_fetch_url(&json!({"page_url": "https://example.com\\a\\b"})),
            Some("https://example.com/a/b".to_owned())
        );
    }

    #[test]
    fn extract_fetch_url_rejects_unusable_args() {
        assert_eq!(extract_fetch_url(&json!({})), None);
        assert_eq!(extract_fetch_url(&json!({"url": ""})), None);
        assert_eq!(extract_fetch_url(&json!({"url": null, "href": true})), None);
        assert_eq!(
            extract_fetch_url(&json!({"url": ["https://example.com/"]})),
            None
        );
        assert_eq!(extract_fetch_url(&json!("https://example.com/")), None);
        assert_eq!(extract_fetch_url(&json!(null)), None);
    }
}
