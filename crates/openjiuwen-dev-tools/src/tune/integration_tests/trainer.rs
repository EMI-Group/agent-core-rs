//! Scenarios 2 and 3: split provenance and held-out isolation.
//!
//! A tagged train/validation pair driven through `Trainer::train_split`, the
//! provenance the result and the summary report, and the guarantee that a
//! held-out-only token never reaches a fitted candidate.

use super::support::*;
use crate::tune::*;

#[tokio::test]
async fn train_split_reports_train_and_val_provenance() {
    let loader = CaseLoader::new(nine_cases());
    let (train, val) = loader.split_tagged(0.3).expect("a valid split ratio");
    assert_eq!((train.len(), val.len()), (2, 7));

    let result = Trainer::new()
        .train_split(SEED_PROMPT, &train, &val)
        .await
        .expect("a valid tagged run");

    let train_score = result
        .train_score
        .expect("a tagged run reports its in-sample score");
    assert!(
        train_score.is_finite() && (0.0..=1.0).contains(&train_score),
        "the in-sample score is bounded: {train_score}"
    );
    assert_eq!(result.val_score, Some(result.score));
    assert_eq!(result.selected_on, Some(CaseSplit::Val));
    assert!(result.score.is_finite() && (0.0..=1.0).contains(&result.score));
    assert_ne!(result.best_prompt, "", "the winner is a real prompt");
    assert!(result.iterations >= 1, "the seed leaves room to improve");

    let repeated = Trainer::new()
        .train_split(SEED_PROMPT, &train, &val)
        .await
        .expect("a valid tagged run");
    assert_eq!(result, repeated, "two identical runs agree");

    // The same evaluation over the validation partition keeps its tag.
    let summary = Trainer::new()
        .evaluate_split(SEED_PROMPT, &val)
        .await
        .expect("a valid evaluation");
    assert_eq!(summary.split, CaseSplit::Val);
    assert_eq!(summary.results.len(), val.len());
    assert!(!summary.is_empty());
    let untagged = Trainer::new()
        .evaluate(SEED_PROMPT, val.cases())
        .await
        .expect("a valid evaluation");
    assert_close(summary.mean_score, untagged.0);
    assert_eq!(summary.results, untagged.1);
}

// ------------------------------------------------- 3. held-out isolation ----

#[tokio::test]
async fn the_val_only_token_never_reaches_a_candidate() {
    // The first three cases are the train split, the rest carry the held-out
    // token in every input and label.
    let mut cases: Vec<Case> = (1..=3)
        .map(|index| {
            case(
                &format!("c{index}"),
                &format!("question {index}"),
                &format!("answer {index}"),
            )
        })
        .collect();
    cases.extend((4..=9).map(|index| {
        case(
            &format!("c{index}"),
            &format!("question {index} {VAL_ONLY}"),
            &format!("answer {index} {VAL_ONLY}"),
        )
    }));

    let (train, val) = CaseLoader::new(cases)
        .split_tagged(1.0 / 3.0)
        .expect("a valid split ratio");
    assert_eq!((train.len(), val.len()), (3, 6));
    assert!(
        train.cases().iter().all(|entry| !mentions(entry, VAL_ONLY)),
        "no train case carries the held-out token"
    );
    assert!(
        val.cases().iter().all(|entry| mentions(entry, VAL_ONLY)),
        "every held-out case carries it"
    );

    let result = Trainer::new()
        .train_split(SEED_PROMPT, &train, &val)
        .await
        .expect("a valid tagged run");

    assert!(
        !result.best_prompt.contains(VAL_ONLY),
        "the winner quotes the held-out text: {}",
        result.best_prompt
    );
    // The trainer fits on the train cases only, so no candidate it reports may
    // quote the held-out text either (it reports the winner; a trainer keeps no
    // per-round records, so this loop is empty for the current implementation and
    // stays a guard for any future candidate reporting).
    for candidate in &result.candidate_prompts() {
        assert!(
            !candidate.contains(VAL_ONLY),
            "a fitted candidate quotes the held-out text: {candidate}"
        );
    }

    // The same invariant over a search that does record every candidate: the
    // held-out cases go in as hidden cases, and no candidate may quote them.
    let task_cases: Vec<PromptTaskCase> = train
        .cases()
        .iter()
        .map(|entry| PromptTaskCase::from_case(entry, false))
        .chain(
            val.cases()
                .iter()
                .map(|entry| PromptTaskCase::from_case(entry, true)),
        )
        .collect();
    let task = PromptTaskSpec::new("Answer the question.")
        .with_base_prompt("Answer the question.")
        .with_cases(task_cases);
    let search = PromptSearchOptimizer::new()
        .optimize_task(&task)
        .await
        .expect("a valid prompt-search task");
    assert!(
        !search.best_prompt.contains(VAL_ONLY),
        "the search winner quotes the held-out text: {}",
        search.best_prompt
    );
    let candidates = search.candidate_prompts();
    assert!(
        !candidates.is_empty(),
        "a prompt search records its candidates"
    );
    for candidate in &candidates {
        assert!(
            !candidate.contains(VAL_ONLY),
            "a search candidate quotes the held-out text: {candidate}"
        );
    }
}
