//! [`Rng64`]: the one deterministic `xorshift64*` generator behind every seeded
//! shuffle in this module.
//!
//! [`CaseLoader::shuffle`](crate::tune::CaseLoader) orders a case list through
//! it, [`ParameterSearcher`](crate::tune::ParameterSearcher) samples parameter
//! candidates through it, and anything else that needs a reproducible
//! permutation should use it rather than growing a private copy.
//!
//! The algorithm is Marsaglia's xorshift64 shifted through a 64-bit multiply,
//! which is plenty for a reproducible permutation and keeps the crate free of
//! external random-number dependencies while the results stay bit-identical to
//! the generator the loader and the parameter searcher used to carry privately.

/// A deterministic `xorshift64*` generator used by the module's seeded shuffles.
///
/// The generator is not cryptographic and makes no randomness guarantees beyond
/// reproducibility: the same seed always produces the same word sequence, so the
/// same seed always produces the same permutation.
pub(crate) struct Rng64 {
    /// The generator state; never zero after construction.
    state: u64,
}

impl Rng64 {
    /// Seeds the generator; seed `0` is remapped to the golden-ratio constant
    /// because xorshift requires a non-zero state.
    #[must_use]
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// The next pseudo-random 64-bit word.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A pseudo-random index in `0..bound`, and `0` when `bound` is `0` or `1`.
    ///
    /// The modulo bias is irrelevant for shuffling a short list: the largest
    /// list here has a handful of entries, so the deviation from a uniform
    /// distribution is far below anything the surrounding search can observe,
    /// and staying off `rand` keeps the crate dependency-free.
    #[must_use]
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    /// Shuffles `items` in place with a Fisher-Yates pass.
    ///
    /// Every index `i` from the end is swapped with a uniformly drawn
    /// `j` in `0..=i`, so the same seed always yields the same permutation of
    /// the same input and every permutation is reachable. An empty or
    /// one-element slice is left untouched.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The golden-ratio constant seed `0` is remapped to.
    const GOLDEN_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    /// The first `count` words of a generator seeded with `seed`.
    fn words(seed: u64, count: usize) -> Vec<u64> {
        let mut rng = Rng64::new(seed);
        (0..count).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn the_same_seed_yields_the_same_words() {
        assert_eq!(words(42, 8), words(42, 8));
        assert_eq!(words(u64::MAX, 4), words(u64::MAX, 4));
        assert_ne!(words(42, 4), words(43, 4));
    }

    #[test]
    fn seed_zero_uses_the_golden_ratio_seed() {
        assert_eq!(words(0, 8), words(GOLDEN_SEED, 8));
        // Pinned words: the generator must not drift, because the documented
        // permutations of the loader and the parameter searcher depend on it.
        assert_eq!(
            words(GOLDEN_SEED, 4),
            [
                0x0D83_B3E2_9A21_487A,
                0x54C4_4C79_F1FE_9D67,
                0xA845_F342_007A_0E78,
                0x7D6E_0B87_8A79_4779
            ]
        );
        assert_eq!(
            words(42, 3),
            [
                0x56CE_4AB7_719B_A3A0,
                0xC841_EB53_EBBB_2DDA,
                0xCA46_6BE0_C998_0276
            ]
        );
    }

    #[test]
    fn below_stays_inside_its_bound() {
        let mut rng = Rng64::new(7);
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.below(1), 0);
        for bound in 2..=64 {
            for _ in 0..32 {
                assert!(rng.below(bound) < bound, "bound {bound}");
            }
        }
    }

    #[test]
    fn shuffle_is_deterministic_for_a_seed() {
        let original: Vec<u32> = (0..12).collect();
        let shuffled = |seed: u64| {
            let mut items = original.clone();
            Rng64::new(seed).shuffle(&mut items);
            items
        };
        assert_eq!(shuffled(1), shuffled(1));
        assert_eq!(shuffled(0), shuffled(GOLDEN_SEED));
        assert!(
            (0..8u64).any(|seed| shuffled(seed) != original),
            "at least one seed must reorder the input"
        );
    }

    #[test]
    fn shuffle_preserves_the_multiset() {
        let original: Vec<u32> = (0..9).collect();
        for seed in [0, 1, 2, 42, u64::MAX] {
            let mut items = original.clone();
            Rng64::new(seed).shuffle(&mut items);
            assert_eq!(items.len(), original.len());
            let mut sorted = items;
            sorted.sort_unstable();
            assert_eq!(sorted, original, "seed {seed} keeps the multiset");
        }
    }

    #[test]
    fn shuffle_leaves_tiny_slices_untouched() {
        let mut rng = Rng64::new(3);
        rng.shuffle::<u8>(&mut []);
        let mut single = [7u8];
        rng.shuffle(&mut single);
        assert_eq!(single, [7]);
    }
}
