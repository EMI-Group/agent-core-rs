//! Scenario 1: the dataset pipeline a training loop is driven with.
//!
//! A tagged cut from the loader, the batch plan over the train partition with
//! and without `drop_last`, and the seeded random and stratified samplers.

use std::collections::BTreeMap;

use serde_json::Value;

use super::support::*;
use crate::tune::*;

#[test]
fn tagged_split_and_batch_plan_cover_the_train_dataset() {
    let loader = CaseLoader::new(nine_cases());
    let (train, val) = loader
        .split_tagged(1.0 / 3.0)
        .expect("a split ratio inside 0.0..=1.0");

    // `floor(9 * 1/3) = 3` cases on the train side, the other six held out.
    assert_eq!(train.split, CaseSplit::Train);
    assert_eq!(val.split, CaseSplit::Val);
    assert_eq!((train.len(), val.len()), (3, 6));
    assert_eq!(ids(train.cases()), ["c1", "c2", "c3"]);
    assert_eq!(ids(val.cases()), ["c4", "c5", "c6", "c7", "c8", "c9"]);

    // Batch size 2 over three train cases: two batches an epoch while the short
    // tail is kept, one batch when it is dropped.
    let kept = BatchScheduler::new(2)
        .expect("a non-zero batch size")
        .with_epochs(2);
    let plan = train.batches(&kept).expect("a plannable train split");
    assert_eq!(plan.len(), 2, "two epochs");
    assert_eq!(plan[0].index, 0);
    assert_eq!(plan[1].index, 1);
    assert_eq!(
        plan.iter().map(Epoch::len).collect::<Vec<usize>>(),
        vec![2, 2]
    );
    assert_eq!(plan[0].batches[0].len(), 2);
    assert_eq!(plan[0].batches[1].len(), 1, "the short tail is kept");
    let steps: Vec<usize> = plan
        .iter()
        .flat_map(|epoch| epoch.batches.iter().map(|batch| batch.step))
        .collect();
    assert_eq!(
        steps,
        vec![1, 2, 3, 4],
        "steps are numbered across the plan"
    );
    assert_eq!(
        train.batches(&kept).expect("the plan is re-runnable"),
        plan,
        "a repeated call plans identically"
    );
    assert_eq!(kept.batches_per_epoch(train.len()), 2);

    let dropped = BatchScheduler::new(2)
        .expect("a non-zero batch size")
        .with_epochs(2)
        .with_drop_last();
    let dropped_plan = train.batches(&dropped).expect("a plannable train split");
    assert_eq!(
        dropped_plan.iter().map(Epoch::len).collect::<Vec<usize>>(),
        vec![1, 1],
        "the partial tail batch is dropped"
    );
    let dropped_steps: Vec<usize> = dropped_plan
        .iter()
        .flat_map(|epoch| epoch.batches.iter().map(|batch| batch.step))
        .collect();
    assert_eq!(dropped_steps, vec![1, 2]);
    assert_eq!(dropped.batches_per_epoch(train.len()), 1);

    // Every batch reports positions that point back into the train case list.
    for epoch in &plan {
        for batch in &epoch.batches {
            assert_eq!(batch.epoch, epoch.index);
            assert_eq!(batch.indices.len(), batch.cases.len());
            for (position, index) in batch.indices.iter().enumerate() {
                assert_eq!(batch.cases[position], train.cases()[*index]);
            }
        }
    }
}

#[test]
fn samplers_are_seeded_and_multiset_preserving() {
    let cases = nine_cases();

    // One seed samples the same selection twice; the result is a subsequence of
    // the input multiset, never an invented or edited case.
    let random = RandomSampler::new(42).with_count(4);
    let first = random.sample(&cases).expect("the sampler never fails");
    let second = random.sample(&cases).expect("the sampler never fails");
    assert_eq!(first, second);
    assert_eq!(first.len(), 4);
    let mut remaining = ids(&cases);
    for sampled in &first {
        let position = remaining
            .iter()
            .position(|id| id == &sampled.case_id)
            .expect("a sampled case comes from the input");
        remaining.remove(position);
    }

    // A different seed samples a valid selection of the same size.
    let other = RandomSampler::new(7)
        .with_count(4)
        .sample(&cases)
        .expect("the sampler never fails");
    assert_eq!(other.len(), 4);
    let mut sampled_ids = ids(&other);
    sampled_ids.sort();
    let mut input_ids = ids(&cases);
    input_ids.sort();
    assert!(sampled_ids.iter().all(|id| input_ids.contains(id)));

    // Full-list sampling: the same seed permutes identically, another seed
    // reorders, and the multiset is preserved either way.
    let full = RandomSampler::new(42).sample(&cases).expect("never fails");
    assert_eq!(
        full,
        RandomSampler::new(42).sample(&cases).expect("never fails")
    );
    let full_other = RandomSampler::new(7).sample(&cases).expect("never fails");
    assert_ne!(full, full_other, "a different seed permutes differently");
    let mut permuted = ids(&full_other);
    permuted.sort();
    assert_eq!(permuted, input_ids, "a full sample is a permutation");

    // Stratified sampling allocates proportionally over the label strata.
    let grouped = stratified_cases();
    let stratified = StratifiedSampler::new(11).with_count(6);
    let sample = stratified
        .sample(&grouped)
        .expect("the sampler never fails");
    assert_eq!(
        sample,
        stratified
            .sample(&grouped)
            .expect("the sampler never fails"),
        "one seed samples the same selection twice"
    );
    assert_eq!(sample.len(), 6);
    let mut per_stratum: BTreeMap<String, usize> = BTreeMap::new();
    for entry in &sample {
        let label = entry
            .label
            .get("answer")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        *per_stratum.entry(label).or_default() += 1;
    }
    assert_eq!(
        per_stratum,
        BTreeMap::from([
            ("alpha".to_owned(), 2),
            ("beta".to_owned(), 2),
            ("gamma".to_owned(), 2),
        ]),
        "two of every three-case stratum"
    );
    let strata = stratified.strata(&grouped);
    assert_eq!(strata.len(), 3);
    assert!(strata.values().all(|indices| indices.len() == 3));
}
