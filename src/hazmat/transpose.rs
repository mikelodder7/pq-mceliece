/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Transposition of a 64x64 bit matrix held as 64 words.
//!
//! Row `i` of the matrix is word `i`; column `j` of that row is bit `j` of the word. The
//! transpose is computed by six recursive block exchanges, each swapping the off-diagonal
//! halves of blocks of decreasing size.

/// Masks for the six exchange rounds: `(low_half, high_half)` for block size `2^d`.
const MASKS: [(u64, u64); 6] = [
    (0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA),
    (0x3333_3333_3333_3333, 0xCCCC_CCCC_CCCC_CCCC),
    (0x0F0F_0F0F_0F0F_0F0F, 0xF0F0_F0F0_F0F0_F0F0),
    (0x00FF_00FF_00FF_00FF, 0xFF00_FF00_FF00_FF00),
    (0x0000_FFFF_0000_FFFF, 0xFFFF_0000_FFFF_0000),
    (0x0000_0000_FFFF_FFFF, 0xFFFF_FFFF_0000_0000),
];

/// Transpose a 64x64 bit matrix in place.
pub(crate) fn transpose_64x64(m: &mut [u64; 64]) {
    for d in (0..6).rev() {
        let s = 1usize << d;
        let (lo_mask, hi_mask) = MASKS[d];

        let mut i = 0;
        while i < 64 {
            for j in i..i + s {
                let x = (m[j] & lo_mask) | ((m[j + s] & lo_mask) << s);
                let y = ((m[j] & hi_mask) >> s) | (m[j + s] & hi_mask);
                m[j] = x;
                m[j + s] = y;
            }
            i += s * 2;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Transpose by moving one bit at a time, as an obviously-correct oracle.
    fn naive(m: &[u64; 64]) -> [u64; 64] {
        let mut out = [0u64; 64];
        for (row, word) in m.iter().enumerate() {
            for (col, dest) in out.iter_mut().enumerate() {
                *dest |= ((word >> col) & 1) << row;
            }
        }
        out
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn transpose_matches_the_bit_by_bit_definition() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..64 {
            let mut m = [0u64; 64];
            for word in m.iter_mut() {
                *word = rng.next();
            }
            let expected = naive(&m);
            transpose_64x64(&mut m);
            assert_eq!(m, expected);
        }
    }

    #[test]
    fn transpose_is_an_involution() {
        let mut rng = Rng(0x0BAD_C0DE_1234_5678);
        let mut m = [0u64; 64];
        for word in m.iter_mut() {
            *word = rng.next();
        }
        let original = m;
        transpose_64x64(&mut m);
        transpose_64x64(&mut m);
        assert_eq!(m, original);
    }

    #[test]
    fn identity_and_all_ones_are_fixed_points() {
        let mut identity = [0u64; 64];
        for (i, word) in identity.iter_mut().enumerate() {
            *word = 1u64 << i;
        }
        let expected = identity;
        transpose_64x64(&mut identity);
        assert_eq!(identity, expected);

        let mut ones = [u64::MAX; 64];
        transpose_64x64(&mut ones);
        assert_eq!(ones, [u64::MAX; 64]);
    }
}
