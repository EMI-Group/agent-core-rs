//! Offline integration suite for the `PUCT` program search.
//!
//! Python parity: `openjiuwen/rsi/artifact_rsi/program_opt/tree.py` (v0.1.20,
//! `Node` at line 53, `PuctTree` at line 89) and the `FlatPuct` selection rule
//! it delegates to (`agentdescent/selection.py` line 479), plus the config
//! values read by `puct_provider.py` / `puct_engine.py`.
//!
//! Every score below is hand-computed from the documented formula
//! `puct_i = rank_score_i + c_puct * P_i * sqrt(sum_visits) / (1 + visits_i)`,
//! so no test depends on a golden value produced by the implementation.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_rsi::artifact_rsi::program_opt::{
    Node, Program, ProgramOptConfig, PuctTree, ReplyFormat, DEFAULT_WORKERS, MAX_WORKERS,
};
use openjiuwen_rsi::{rank_score, ErrorKind};
use std::collections::BTreeMap;

/// A minimal candidate program carrying only an identity.
fn program(id: &str) -> Program {
    Program {
        program_id: id.to_owned(),
        ..Program::default()
    }
}

/// A synthetic node: identity and ancestry in the payload mirror the tree links.
fn node(
    index: usize,
    parent_index: Option<usize>,
    score: Option<f64>,
    num_visits: u64,
    promise: Option<f64>,
) -> Node {
    let iteration = u32::try_from(index).expect("small index");
    Node {
        index,
        parent_index,
        program: Program {
            program_id: format!("p{index}"),
            iteration,
            parent_id: parent_index.map(|parent| format!("p{parent}")),
            code: format!("print({index})"),
            ..Program::default()
        },
        score,
        num_visits,
        promise,
    }
}

/// The visit counts of every node, in node order.
fn visits(tree: &PuctTree) -> Vec<u64> {
    tree.nodes().iter().map(|entry| entry.num_visits).collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

/// Asserts that `validate` rejects every configuration and names the field.
fn assert_validate_rejects(cases: Vec<(ProgramOptConfig, &str)>) {
    for (config, field) in cases {
        let error = config.validate().expect_err("config should be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "{field}");
        assert!(
            error.to_string().contains(field),
            "message {error} should name {field}"
        );
    }
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch: {actual:?}");
    for (position, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "index {position}: expected {want}, got {got} (all: {actual:?})"
        );
    }
}

/// Python parity: `FlatPuct._rank_scores` + `select` (`selection.py` lines
/// 561-599) as used by `PuctTree.select_parent` (`tree.py` line 146), with
/// `c_puct = 1.0` and the upstream uniform prior.
///
/// Scores 0.5 / 0.2 / 0.9 give ascending stable ranks 1 / 0 / 2 out of N = 3,
/// so rank scores 0.5 / 0.0 / 1.0; visits 2 / 1 / 0 sum to 3.
#[test]
fn hand_computed_puct_scores_select_the_exploration_target() {
    let mut tree = PuctTree::from_nodes(
        1.0,
        0.0,
        None,
        vec![
            node(0, None, Some(0.5), 2, None),
            node(1, Some(0), Some(0.2), 1, None),
            node(2, Some(0), Some(0.9), 0, None),
        ],
    )
    .expect("valid tree");

    assert_vec_close(&tree.rank_scores(), &[0.5, 0.0, 1.0]);
    assert_vec_close(&tree.priors(), &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);

    let root_of_three = 3.0_f64.sqrt();
    let scores = tree.puct_scores();
    assert_vec_close(
        &scores,
        &[
            0.5 + root_of_three / 9.0, // 0.5 + sqrt(3)/3 / (1 + 2)
            root_of_three / 6.0,       // 0.0 + sqrt(3)/3 / (1 + 1)
            1.0 + root_of_three / 3.0, // 1.0 + sqrt(3)/3 / (1 + 0)
        ],
    );
    // visit counts for the exploration term, spelled out for the reader
    assert_eq!(visits(&tree), vec![2, 1, 0]);

    // The selection is the first maximum of the same vector: ties go to the
    // lowest index (Python's stable `argmax`).
    let mut first_maximum = 0_usize;
    for (index, score) in scores.iter().enumerate() {
        if *score > scores[first_maximum] {
            first_maximum = index;
        }
    }
    assert_eq!(first_maximum, 2, "scores: {scores:?}");
    assert_eq!(tree.select(), Some(first_maximum));

    // The visit is reserved at selection: the chosen node and every ancestor.
    assert_eq!(visits(&tree), vec![3, 1, 1]);
}

/// Python parity: the single-node `0.5` rank (`selection.py` lines 561-570) and
/// `None` ranked as `-inf` (an unscored candidate ranks last, never first).
#[test]
fn rank_score_edges_and_degenerate_trees() {
    assert_close(rank_score(0, 0), 0.5);
    assert_close(rank_score(7, 1), 0.5);
    assert_close(rank_score(0, 3), 0.0);
    assert_close(rank_score(1, 3), 0.5);
    assert_close(rank_score(2, 3), 1.0);

    let single = PuctTree::from_nodes(1.0, 0.0, None, vec![node(0, None, Some(0.3), 0, None)])
        .expect("valid tree");
    assert_vec_close(&single.rank_scores(), &[0.5]);
    assert_vec_close(&single.priors(), &[1.0]);
    assert_vec_close(&single.puct_scores(), &[0.5]);

    let mixed = PuctTree::from_nodes(
        1.0,
        0.0,
        None,
        vec![
            node(0, None, Some(0.5), 0, None),
            node(1, Some(0), None, 0, None),
        ],
    )
    .expect("valid tree");
    assert_vec_close(&mixed.rank_scores(), &[1.0, 0.0]);
    assert_vec_close(&mixed.priors(), &[0.5, 0.5]);
}

/// Python parity: `_backpropagate_locked` (`tree.py` line 140) — selecting a
/// node increments that node and every ancestor.
#[test]
fn select_backpropagates_every_ancestor() {
    let mut tree = PuctTree::from_nodes(
        1.0,
        0.0,
        None,
        vec![
            node(0, None, Some(1.0), 0, None),
            node(1, Some(0), Some(1.0), 0, None),
            node(2, Some(1), Some(1.0), 0, None),
        ],
    )
    .expect("valid tree");

    // Equal scores rank 0.0 / 0.5 / 1.0 and the total visit count is zero, so
    // the exploration term vanishes and the last-ranked node wins.
    assert_eq!(tree.select(), Some(2));
    assert_eq!(visits(&tree), vec![1, 1, 1]);

    // A second selection still favours the same branch: the whole chain is +1.
    assert_eq!(tree.select(), Some(2));
    assert_eq!(visits(&tree), vec![2, 2, 2]);

    // An out-of-range index is a no-op rather than a panic.
    tree.record_visits(99);
    assert_eq!(visits(&tree), vec![2, 2, 2]);

    let mut empty = PuctTree::default();
    assert_eq!(empty.select(), None);
    assert!(empty.is_empty());
}

/// Python parity: `PuctTree.add_node` (`tree.py` line 174) — an unknown parent
/// is rejected, and a reached `candidate_limit` is checked before the parent.
#[test]
fn add_node_rejects_unknown_parent_and_reached_limit() {
    let mut tree = PuctTree::with_root(program("seed"));
    let error = tree
        .add_node(Some(7), program("orphan"))
        .expect_err("unknown parent");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("parent node 7"),
        "message: {error}"
    );
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.nodes()[0].program.program_id, "seed");

    let config = ProgramOptConfig {
        candidate_limit: Some(2),
        ..ProgramOptConfig::default()
    };
    let mut limited = PuctTree::new(&config);
    assert!(!limited.candidate_limit_reached());
    assert_eq!(limited.add_node(None, program("seed")).expect("root"), 0);
    assert_eq!(
        limited.add_node(Some(0), program("child")).expect("child"),
        1
    );
    assert!(limited.candidate_limit_reached());

    let error = limited
        .add_node(Some(0), program("third"))
        .expect_err("limit reached");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("candidate limit 2"),
        "message: {error}"
    );

    // The limit is checked first, so a full tree reports it even when the
    // parent is also unknown.
    let error = limited
        .add_node(Some(99), program("third"))
        .expect_err("limit reached");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("candidate limit 2"),
        "message: {error}"
    );
    assert_eq!(limited.len(), 2);
}

/// Python parity: `PuctTree.add_node` ancestry and iteration labels — the
/// parent's `program_id` fills an absent `parent_id`, `None` attaches to the
/// root on a non-empty tree, and every expansion consumes one iteration label.
#[test]
fn add_node_links_ancestry_and_increments_iterations() {
    let mut tree = PuctTree::with_root(program("seed"));
    assert_eq!(tree.next_iteration(), 1);
    assert_eq!(tree.add_node(Some(0), program("child")).expect("child"), 1);
    assert_eq!(tree.next_iteration(), 2);
    assert_eq!(
        tree.add_node(Some(1), program("grandchild"))
            .expect("grandchild"),
        2
    );
    assert_eq!(tree.next_iteration(), 3);

    let child = &tree.nodes()[1];
    assert_eq!(child.parent_index, Some(0));
    assert_eq!(child.program.parent_id.as_deref(), Some("seed"));
    assert_eq!(child.program.iteration, 1);
    assert_eq!(child.score, None);
    assert_eq!(child.num_visits, 0);

    // A caller-supplied parent_id wins over the inferred one.
    let mut kept = program("kept");
    kept.parent_id = Some("explicit".to_owned());
    let index = tree.add_node(Some(0), kept).expect("kept");
    assert_eq!(
        tree.nodes()[index].program.parent_id.as_deref(),
        Some("explicit")
    );

    // `None` attaches to index 0 once the tree is non-empty (Python coerces it).
    let index = tree.add_node(None, program("attached")).expect("attached");
    assert_eq!(index, 4);
    assert_eq!(tree.nodes()[index].parent_index, Some(0));
    assert_eq!(
        tree.nodes()[index].program.parent_id.as_deref(),
        Some("seed")
    );
    assert_eq!(tree.nodes()[index].program.iteration, 4);
    assert_eq!(tree.next_iteration(), 5);
}

/// Python parity: `PuctTree.best` — highest score wins, ties go to the lowest
/// index, unscored nodes never beat a scored one, and an all-unscored tree
/// returns the root (the incumbent every candidate is compared against).
#[test]
fn best_prefers_highest_score_then_lowest_index() {
    let tree = PuctTree::from_nodes(
        1.0,
        0.0,
        None,
        vec![
            node(0, None, Some(0.5), 0, None),
            node(1, Some(0), Some(0.9), 0, None),
            node(2, Some(0), Some(0.9), 0, None),
            node(3, Some(0), None, 0, None),
        ],
    )
    .expect("valid tree");
    assert_eq!(tree.best().map(|entry| entry.index), Some(1));

    let unscored_tail = PuctTree::from_nodes(
        1.0,
        0.0,
        None,
        vec![
            node(0, None, None, 0, None),
            node(1, Some(0), Some(0.1), 0, None),
        ],
    )
    .expect("valid tree");
    assert_eq!(unscored_tail.best().map(|entry| entry.index), Some(1));

    let all_unscored = PuctTree::from_nodes(
        1.0,
        0.0,
        None,
        vec![
            node(0, None, None, 0, None),
            node(1, Some(0), None, 0, None),
        ],
    )
    .expect("valid tree");
    assert_eq!(all_unscored.best().map(|entry| entry.index), Some(0));

    assert_eq!(PuctTree::default().best(), None);

    let mut single = PuctTree::with_root(program("seed"));
    let error = single.record_score(3, Some(0.1)).expect_err("no such node");
    assert_eq!(error.kind(), ErrorKind::Validation);
    single.record_score(0, Some(0.25)).expect("root exists");
    assert_close(single.best().expect("root").score.expect("scored"), 0.25);
}

/// Python parity: the numeric knobs `puct_provider.py` / `puct_engine.py` read
/// (`c_puct`, `prior_exponent`, `workers`, budgets); `validate` is the single
/// rejection point, and a default config always validates.
#[test]
fn config_validate_rejects_invalid_numeric_values() {
    assert_eq!(MAX_WORKERS, 8);
    assert_eq!(DEFAULT_WORKERS, 1);
    assert!(ProgramOptConfig::default().validate().is_ok());

    let cases: Vec<(ProgramOptConfig, &str)> = vec![
        (
            ProgramOptConfig {
                workers: 0,
                ..ProgramOptConfig::default()
            },
            "workers",
        ),
        (
            ProgramOptConfig {
                workers: MAX_WORKERS + 1,
                ..ProgramOptConfig::default()
            },
            "workers",
        ),
        (
            ProgramOptConfig {
                c_puct: -0.1,
                ..ProgramOptConfig::default()
            },
            "c_puct",
        ),
        (
            ProgramOptConfig {
                c_puct: f64::NAN,
                ..ProgramOptConfig::default()
            },
            "c_puct",
        ),
        (
            ProgramOptConfig {
                c_puct: f64::INFINITY,
                ..ProgramOptConfig::default()
            },
            "c_puct",
        ),
        (
            ProgramOptConfig {
                prior_exponent: -1.0,
                ..ProgramOptConfig::default()
            },
            "prior_exponent",
        ),
        (
            ProgramOptConfig {
                max_expansions: 0,
                ..ProgramOptConfig::default()
            },
            "max_expansions",
        ),
        (
            ProgramOptConfig {
                max_depth: 0,
                ..ProgramOptConfig::default()
            },
            "max_depth",
        ),
        (
            ProgramOptConfig {
                candidate_timeout_secs: 0.0,
                ..ProgramOptConfig::default()
            },
            "candidate_timeout_secs",
        ),
        (
            ProgramOptConfig {
                candidate_timeout_secs: f64::NAN,
                ..ProgramOptConfig::default()
            },
            "candidate_timeout_secs",
        ),
    ];
    assert_validate_rejects(cases);
}

/// Python parity: the shape values the card carries (`entrypoint`,
/// `candidate_limit`); `to_tree` refuses a config `validate` would reject.
#[test]
fn config_validate_rejects_bad_shape_and_builds_trees() {
    let cases: Vec<(ProgramOptConfig, &str)> = vec![
        (
            ProgramOptConfig {
                entrypoint: "   ".to_owned(),
                ..ProgramOptConfig::default()
            },
            "entrypoint",
        ),
        (
            ProgramOptConfig {
                candidate_limit: Some(0),
                ..ProgramOptConfig::default()
            },
            "candidate_limit",
        ),
    ];
    assert_validate_rejects(cases);

    let tree = ProgramOptConfig::default().to_tree().expect("valid config");
    assert!(tree.is_empty());
    assert_close(tree.c_puct, 1.0);

    let invalid = ProgramOptConfig {
        max_depth: 0,
        ..ProgramOptConfig::default()
    };
    let error = invalid.to_tree().expect_err("invalid config");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

/// A scorecard and a populated tree both round-trip through JSON, tolerate
/// unknown fields, and default missing ones — the property that lets a
/// checkpoint resume a search without reusing an iteration label.
#[test]
fn serde_round_trips_config_and_populated_tree() {
    let config = ProgramOptConfig {
        c_puct: 2.5,
        prior_exponent: 2.0,
        candidate_limit: Some(9),
        workers: 3,
        entrypoint: "src/main.py".to_owned(),
        reply_format: ReplyFormat::Tagged,
        ..ProgramOptConfig::default()
    };
    let serialized = serde_json::to_value(&config).expect("serialize config");
    let restored: ProgramOptConfig =
        serde_json::from_value(serialized).expect("deserialize config");
    assert_eq!(restored, config);
    assert_eq!(restored.reply_format, ReplyFormat::Tagged);

    let tolerant: ProgramOptConfig = serde_json::from_value(serde_json::json!({
        "c_puct": 2.5,
        "unknown_field": {"nested": true},
    }))
    .expect("unknown fields are tolerated");
    assert_close(tolerant.c_puct, 2.5);
    assert_eq!(tolerant.workers, DEFAULT_WORKERS);
    assert_eq!(tolerant.entrypoint, "candidate.py");
    assert_eq!(tolerant.reply_format, ReplyFormat::Files);

    let root = Node {
        promise: Some(3.0),
        program: Program {
            program_id: "root".to_owned(),
            iteration: 3,
            code: "print('root')".to_owned(),
            change_summary: "seed".to_owned(),
            metrics: BTreeMap::from([("runtime".to_owned(), 0.5)]),
            ..Program::default()
        },
        ..node(0, None, Some(0.25), 2, None)
    };
    let child = Node {
        promise: None,
        program: Program {
            program_id: "child".to_owned(),
            iteration: 5,
            parent_id: Some("root".to_owned()),
            code: "print('child')".to_owned(),
            ..Program::default()
        },
        ..node(1, Some(0), Some(0.75), 1, None)
    };
    let tree = PuctTree::from_nodes(1.5, 2.0, Some(9), vec![root, child]).expect("valid tree");
    assert_eq!(tree.next_iteration(), 6);

    let serialized = serde_json::to_value(&tree).expect("serialize tree");
    let restored: PuctTree = serde_json::from_value(serialized).expect("deserialize tree");
    assert_eq!(restored, tree);
    assert_eq!(restored.nodes(), tree.nodes());
    assert_close(restored.c_puct, 1.5);
    assert_close(restored.prior_exponent, 2.0);
    assert_eq!(restored.candidate_limit, Some(9));
    assert_eq!(restored.next_iteration(), 6);

    let restored_root = &restored.nodes()[0];
    assert_close(restored_root.score.expect("scored"), 0.25);
    assert_close(restored_root.promise.expect("rated"), 3.0);
    assert_eq!(restored_root.num_visits, 2);
    assert_eq!(restored_root.program.iteration, 3);
    assert_close(restored_root.program.metrics["runtime"], 0.5);

    let restored_child = &restored.nodes()[1];
    assert_close(restored_child.score.expect("scored"), 0.75);
    assert_eq!(restored_child.promise, None);
    assert_eq!(restored_child.num_visits, 1);
    assert_eq!(restored_child.parent_index, Some(0));
    assert_eq!(restored_child.program.parent_id.as_deref(), Some("root"));
}
