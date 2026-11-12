//! Integration tests for [`AutoHarnessConfig`]: Python defaults, YAML/JSON
//! parsing, `validate()` rejections, derived paths and the async load helpers.
//!
//! Python reference: `tests/unit_tests/rsi/test_config.py` and
//! `tests/unit_tests/rsi/auto_harness/test_compatibility.py` (v0.1.20).
//! Everything here is offline and deterministic: no network, no git, no clock.
// This is a separate integration-test crate, so the lib's `cfg(test)` allow
// block does not apply to it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use openjiuwen_rsi::{AutoHarnessConfig, ErrorKind, PipelinePreference};
use serde_json::json;

/// A configuration that passes [`AutoHarnessConfig::validate`].
fn valid() -> AutoHarnessConfig {
    AutoHarnessConfig {
        data_dir: PathBuf::from("/tmp/rsi-valid"),
        ..AutoHarnessConfig::default()
    }
}

/// Assert that `config.validate()` fails with [`ErrorKind::Validation`].
fn assert_invalid(config: &AutoHarnessConfig, context: &str) {
    let error = config.validate().expect_err(context);
    assert_eq!(error.kind(), ErrorKind::Validation, "{context}: {error}");
}

// ---------------------------------------------------------------------------
// Python defaults
// ---------------------------------------------------------------------------

/// Python [`AutoHarnessConfig`] defaults, field by field.
#[test]
fn defaults_match_python_reference() {
    let config = AutoHarnessConfig::default();

    assert_eq!(config.session_budget_secs, 900_000.0);
    assert_eq!(config.cost_limit_usd, 10.0);
    assert_eq!(config.task_timeout_secs, 300_000.0);
    assert_eq!(config.model_timeout_secs, 300_000.0);
    assert_eq!(config.max_tasks_per_session, 10);
    assert_eq!(config.max_iterations, 10);
    assert_eq!(config.pipeline_preference, PipelinePreference::Auto);
    assert_eq!(config.git_base_branch, "develop");
    assert_eq!(config.fix_phase1_max_retries, 10);
    assert_eq!(config.fix_phase2_max_retries, 9);
    assert_eq!(config.language, "cn");
    assert!((config.min_improvement - 0.05).abs() < 1e-9);
    assert_eq!(config.patience, 2);
    assert!(config.target_score.is_none());
    assert_eq!(
        config.repo_url,
        "https://gitcode.com/openJiuwen/agent-core.git"
    );
    assert_eq!(config.high_impact_prefixes, vec!["openjiuwen/core/"]);

    assert!(config.data_dir.as_os_str().is_empty());
    assert!(config.local_repo.as_os_str().is_empty());
    assert!(config.config_path.as_os_str().is_empty());
    assert!(config.experience_dir.as_os_str().is_empty());
    assert_eq!(config.skills_dirs, Vec::<PathBuf>::new());
    assert_eq!(config.optimization_goal, "");
    assert_eq!(config.immutable_files, Vec::<String>::new());
    assert!(!config.config_bootstrapped);
    assert!(config.suggested_local_repo.is_none());
    assert!(config.extra.is_object());
    assert!(config.extra.as_object().expect("extra object").is_empty());
}

/// `default_agent_iterations()` carries the ten Python stage budgets.
#[test]
fn default_agent_iterations_match_python() {
    let iterations = AutoHarnessConfig::default_agent_iterations();

    assert_eq!(iterations.len(), 10);
    let expected = [
        ("implement", 30_u32),
        ("assess", 30),
        ("plan", 15),
        ("select_pipeline", 10),
        ("eval", 10),
        ("pr_draft", 5),
        ("learnings", 5),
        ("explore_subagent", 20),
        ("browser_subagent", 20),
        ("merge_ext", 8),
    ];
    for (stage, budget) in expected {
        assert_eq!(iterations.get(stage), Some(&budget), "stage {stage}");
    }
    // The default struct already holds the same map.
    assert_eq!(AutoHarnessConfig::default().agent_iterations, iterations);
}

// ---------------------------------------------------------------------------
// Parsing: defaults overlay, unknown keys, malformed input
// ---------------------------------------------------------------------------

/// YAML overlays the Python defaults and keeps unknown keys in `extra`.
#[test]
fn yaml_parsing_overlays_defaults_and_keeps_unknown_keys() {
    let yaml = r"
data_dir: /tmp/rsi
local_repo: /tmp/repo
pipeline_preference: meta
max_iterations: 3
agent_iterations:
  implement: 42
future_python_key: 7
";
    let config = AutoHarnessConfig::from_yaml_str(yaml).expect("yaml parses");

    assert_eq!(config.data_dir, PathBuf::from("/tmp/rsi"));
    assert_eq!(config.local_repo, PathBuf::from("/tmp/repo"));
    assert_eq!(config.pipeline_preference, PipelinePreference::Meta);
    assert_eq!(config.max_iterations, 3);
    assert_eq!(config.agent_iterations.len(), 1);
    assert_eq!(config.agent_iterations.get("implement"), Some(&42));
    assert_eq!(config.extra["future_python_key"], json!(7));

    // Missing keys fall back to the Python defaults.
    assert_eq!(config.language, "cn");
    assert_eq!(config.cost_limit_usd, 10.0);
    assert_eq!(config.git_base_branch, "develop");
    assert_eq!(config.patience, 2);
}

/// JSON parsing behaves like YAML: overlay defaults, retain unknown keys.
#[test]
fn json_parsing_overlays_defaults_and_keeps_unknown_keys() {
    let config = AutoHarnessConfig::from_json_value(json!({
        "data_dir": "/tmp/rsi-json",
        "pipeline_preference": "extended",
        "session_budget_secs": 60.0,
        "future_python_key": 7,
        "another_future_key": {"nested": true},
    }))
    .expect("json parses");

    assert_eq!(config.data_dir, PathBuf::from("/tmp/rsi-json"));
    assert_eq!(config.pipeline_preference, PipelinePreference::Extended);
    assert_eq!(config.session_budget_secs, 60.0);
    assert_eq!(config.extra["future_python_key"], json!(7));
    assert_eq!(config.extra["another_future_key"], json!({"nested": true}));
    assert_eq!(config.language, "cn");
    assert_eq!(config.max_iterations, 10);
    assert!(config.target_score.is_none());
}

/// Bad enum values and malformed YAML map to `ErrorKind::Validation`.
#[test]
fn yaml_parsing_rejects_bad_enum_and_malformed_documents() {
    let bogus = AutoHarnessConfig::from_yaml_str("pipeline_preference: bogus")
        .expect_err("unknown pipeline preference is rejected");
    assert_eq!(bogus.kind(), ErrorKind::Validation, "{bogus}");

    let malformed =
        AutoHarnessConfig::from_yaml_str("data_dir: [unclosed").expect_err("malformed yaml");
    assert_eq!(malformed.kind(), ErrorKind::Validation, "{malformed}");
}

/// Wrongly typed JSON and unknown enum values map to `ErrorKind::Validation`.
#[test]
fn json_parsing_rejects_bad_shape_and_bad_enum() {
    let wrong_type = AutoHarnessConfig::from_json_value(json!({"max_iterations": "many"}))
        .expect_err("string is not a u32");
    assert_eq!(wrong_type.kind(), ErrorKind::Validation, "{wrong_type}");

    let bogus_enum = AutoHarnessConfig::from_json_value(json!({
        "pipeline_preference": "bogus",
    }))
    .expect_err("unknown pipeline preference is rejected");
    assert_eq!(bogus_enum.kind(), ErrorKind::Validation, "{bogus_enum}");
}

// ---------------------------------------------------------------------------
// validate()
// ---------------------------------------------------------------------------

/// `data_dir`, `repo_url` and `git_base_branch` must be present.
#[test]
fn validate_rejects_empty_paths_and_urls() {
    // The Python default config (empty data_dir) cannot drive a session.
    assert_invalid(
        &AutoHarnessConfig::default(),
        "default config has an empty data_dir",
    );

    let no_url = AutoHarnessConfig {
        repo_url: String::new(),
        ..valid()
    };
    assert_invalid(&no_url, "repo_url must not be empty");

    let no_branch = AutoHarnessConfig {
        git_base_branch: String::new(),
        ..valid()
    };
    assert_invalid(&no_branch, "git_base_branch must not be empty");

    let blank_branch = AutoHarnessConfig {
        git_base_branch: "   ".to_owned(),
        ..valid()
    };
    assert_invalid(&blank_branch, "whitespace git_base_branch is empty");
}

/// Every budget must be a strictly positive finite number.
#[test]
fn validate_rejects_non_positive_budgets() {
    let cases: [(&str, AutoHarnessConfig); 6] = [
        (
            "session_budget_secs = 0",
            AutoHarnessConfig {
                session_budget_secs: 0.0,
                ..valid()
            },
        ),
        (
            "session_budget_secs = NaN",
            AutoHarnessConfig {
                session_budget_secs: f64::NAN,
                ..valid()
            },
        ),
        (
            "cost_limit_usd = 0",
            AutoHarnessConfig {
                cost_limit_usd: 0.0,
                ..valid()
            },
        ),
        (
            "cost_limit_usd = -1",
            AutoHarnessConfig {
                cost_limit_usd: -1.0,
                ..valid()
            },
        ),
        (
            "task_timeout_secs = 0",
            AutoHarnessConfig {
                task_timeout_secs: 0.0,
                ..valid()
            },
        ),
        (
            "model_timeout_secs = INFINITY",
            AutoHarnessConfig {
                model_timeout_secs: f64::INFINITY,
                ..valid()
            },
        ),
    ];
    for (context, config) in cases {
        assert_invalid(&config, context);
    }
}

/// Counts and `patience` must be greater than zero.
#[test]
fn validate_rejects_zero_counts() {
    let cases: [(&str, AutoHarnessConfig); 3] = [
        (
            "max_tasks_per_session = 0",
            AutoHarnessConfig {
                max_tasks_per_session: 0,
                ..valid()
            },
        ),
        (
            "max_iterations = 0",
            AutoHarnessConfig {
                max_iterations: 0,
                ..valid()
            },
        ),
        (
            "patience = 0",
            AutoHarnessConfig {
                patience: 0,
                ..valid()
            },
        ),
    ];
    for (context, config) in cases {
        assert_invalid(&config, context);
    }
}

/// Score thresholds must be finite, and `min_improvement` non-negative.
#[test]
fn validate_rejects_invalid_scores() {
    let negative = AutoHarnessConfig {
        min_improvement: -0.1,
        ..valid()
    };
    assert_invalid(&negative, "min_improvement must be non-negative");

    let nan_improvement = AutoHarnessConfig {
        min_improvement: f64::NAN,
        ..valid()
    };
    assert_invalid(&nan_improvement, "min_improvement must be finite");

    let nan_target = AutoHarnessConfig {
        target_score: Some(f64::NAN),
        ..valid()
    };
    assert_invalid(&nan_target, "target_score must be finite");

    // A finite target is accepted.
    let finite_target = AutoHarnessConfig {
        target_score: Some(0.9),
        ..valid()
    };
    finite_target.validate().expect("finite target is valid");
}

/// `extra` must stay a JSON object so unknown keys can be flattened back in.
#[test]
fn validate_rejects_non_object_extra() {
    let config = AutoHarnessConfig {
        extra: serde_json::Value::Null,
        ..valid()
    };
    assert_invalid(&config, "extra must be a JSON object");
}

/// A fully specified configuration passes validation.
#[test]
fn validate_accepts_valid_config() {
    valid().validate().expect("valid config passes validation");

    let configured = AutoHarnessConfig {
        data_dir: PathBuf::from("/tmp/rsi-data"),
        experience_dir: PathBuf::from("/tmp/rsi-experience"),
        target_score: Some(0.8),
        ..valid()
    };
    configured.validate().expect("configured paths are valid");
}

// ---------------------------------------------------------------------------
// Derived paths
// ---------------------------------------------------------------------------

/// Derived directories live underneath `data_dir`.
#[test]
fn derived_paths_follow_data_dir() {
    let config = AutoHarnessConfig {
        data_dir: PathBuf::from("/tmp/rsi-data"),
        ..AutoHarnessConfig::default()
    };

    assert_eq!(
        config.resolved_experience_dir(),
        PathBuf::from("/tmp/rsi-data/experience")
    );
    assert_eq!(
        config.resolved_worktrees_dir(),
        PathBuf::from("/tmp/rsi-data/worktrees")
    );
    assert_eq!(
        config.resolved_runs_dir(),
        PathBuf::from("/tmp/rsi-data/runs")
    );
    assert_eq!(
        config.resolved_cache_repo_dir(),
        PathBuf::from("/tmp/rsi-data/cache/repo")
    );
}

/// An explicit `experience_dir` overrides the derived location.
#[test]
fn experience_dir_override_wins() {
    let config = AutoHarnessConfig {
        data_dir: PathBuf::from("/tmp/rsi-data"),
        experience_dir: PathBuf::from("/tmp/custom-experience"),
        ..AutoHarnessConfig::default()
    };
    assert_eq!(
        config.resolved_experience_dir(),
        PathBuf::from("/tmp/custom-experience")
    );
}

/// With an empty `data_dir` every path falls back to `.auto_harness/`.
#[test]
fn derived_paths_fall_back_to_auto_harness_dir() {
    let config = AutoHarnessConfig::default();

    assert_eq!(
        config.resolved_experience_dir(),
        PathBuf::from(".auto_harness/experience")
    );
    assert_eq!(
        config.resolved_worktrees_dir(),
        PathBuf::from(".auto_harness/worktrees")
    );
    assert_eq!(
        config.resolved_runs_dir(),
        PathBuf::from(".auto_harness/runs")
    );
    assert_eq!(
        config.resolved_cache_repo_dir(),
        PathBuf::from(".auto_harness/cache/repo")
    );
}

// ---------------------------------------------------------------------------
// Agent iteration resolution and serde round-trip
// ---------------------------------------------------------------------------

/// Configured stage budgets overlay the Python defaults instead of replacing.
#[test]
fn resolve_agent_iterations_overlays_python_defaults() {
    let config = AutoHarnessConfig {
        agent_iterations: BTreeMap::from([("implement".to_owned(), 42)]),
        ..AutoHarnessConfig::default()
    };
    let resolved = config.resolve_agent_iterations();

    assert_eq!(resolved.len(), 10);
    assert_eq!(resolved.get("implement"), Some(&42));
    assert_eq!(resolved.get("assess"), Some(&30));
    assert_eq!(resolved.get("merge_ext"), Some(&8));

    // A stage unknown to Python is added verbatim.
    let extra_stage = AutoHarnessConfig {
        agent_iterations: BTreeMap::from([("brand_new_stage".to_owned(), 3)]),
        ..AutoHarnessConfig::default()
    };
    let resolved = extra_stage.resolve_agent_iterations();
    assert_eq!(resolved.len(), 11);
    assert_eq!(resolved.get("brand_new_stage"), Some(&3));
}

/// A non-default configuration survives a JSON round-trip unchanged.
#[test]
fn serde_round_trip_preserves_custom_config() {
    let config = AutoHarnessConfig {
        data_dir: PathBuf::from("/tmp/rsi-roundtrip"),
        local_repo: PathBuf::from("/tmp/rsi-repo"),
        config_path: PathBuf::from("/tmp/rsi-roundtrip/auto_harness.yaml"),
        experience_dir: PathBuf::from("/tmp/rsi-roundtrip/exp"),
        skills_dirs: vec![PathBuf::from("/tmp/rsi-skills")],
        optimization_goal: "close the Python parity gap".to_owned(),
        pipeline_preference: PipelinePreference::Extended,
        max_iterations: 4,
        agent_iterations: BTreeMap::from([("implement".to_owned(), 42), ("assess".to_owned(), 12)]),
        immutable_files: vec!["openjiuwen/core/__init__.py".to_owned()],
        target_score: Some(0.9),
        suggested_local_repo: Some(PathBuf::from("/tmp/rsi-suggested")),
        config_bootstrapped: true,
        extra: json!({"future_python_key": 7}),
        ..AutoHarnessConfig::default()
    };

    let value = serde_json::to_value(&config).expect("config serialises");
    let restored: AutoHarnessConfig = serde_json::from_value(value).expect("config deserialises");

    assert_eq!(restored, config);
    assert_eq!(restored.extra["future_python_key"], json!(7));
    assert_eq!(restored.pipeline_preference, PipelinePreference::Extended);
    assert_eq!(restored.target_score, Some(0.9));
    assert_eq!(
        restored.suggested_local_repo,
        Some(PathBuf::from("/tmp/rsi-suggested"))
    );
}

// ---------------------------------------------------------------------------
// Async loading
// ---------------------------------------------------------------------------

/// `load` parses a YAML file and reports a missing file as `ErrorKind::Config`.
#[tokio::test]
async fn load_reads_yaml_and_reports_missing_file_as_config_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("auto_harness.yaml");
    tokio::fs::write(
        &path,
        "data_dir: /tmp/rsi-loaded\npipeline_preference: extended\nconfig_bootstrapped: true\n",
    )
    .await
    .expect("write config");

    let config = AutoHarnessConfig::load(&path).await.expect("load config");
    assert_eq!(config.data_dir, PathBuf::from("/tmp/rsi-loaded"));
    assert_eq!(config.pipeline_preference, PipelinePreference::Extended);
    assert!(config.config_bootstrapped);
    assert_eq!(config.language, "cn");

    let missing = dir.path().join("absent.yaml");
    let error = AutoHarnessConfig::load(&missing)
        .await
        .expect_err("missing file is an error");
    assert_eq!(error.kind(), ErrorKind::Config, "{error}");
}

/// `load_or_default` never errors: missing and invalid files yield defaults,
/// while a loaded file keeps `config_bootstrapped`.
#[tokio::test]
async fn load_or_default_falls_back_and_preserves_bootstrap_flag() {
    let dir = tempfile::tempdir().expect("tempdir");

    let missing = dir.path().join("absent.yaml");
    let config = AutoHarnessConfig::load_or_default(&missing).await;
    assert_eq!(config, AutoHarnessConfig::default());

    let invalid = dir.path().join("invalid.yaml");
    tokio::fs::write(&invalid, "data_dir: [unclosed")
        .await
        .expect("write invalid config");
    let config = AutoHarnessConfig::load_or_default(&invalid).await;
    assert_eq!(config, AutoHarnessConfig::default());
    assert!(!config.config_bootstrapped);

    let bootstrapped = dir.path().join("bootstrapped.yaml");
    tokio::fs::write(&bootstrapped, "config_bootstrapped: true\n")
        .await
        .expect("write bootstrapped config");
    let config = AutoHarnessConfig::load_or_default(&bootstrapped).await;
    assert!(config.config_bootstrapped);
    assert_eq!(config.data_dir, PathBuf::new());
}
