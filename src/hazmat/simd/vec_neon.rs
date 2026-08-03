/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! The AArch64 form of the bit-sliced field multiplication.
//!
//! # Why this exists
//!
//! `u128` is an integer type on AArch64 just as it is on x86-64: rustc lowers a `u128`
//! exclusive-or to two `eor`s on a pair of general-purpose registers, it never promotes the
//! planes to `v` registers on its own, and the twenty-six live planes of the convolution
//! overflow the thirty-one integer registers, so the compiled portable multiply spends more
//! instructions on spills than on arithmetic. Naming the register type directly holds each
//! plane in one `v` register and turns each plane operation into one instruction.
//!
//! `neon` is part of the AArch64 baseline, so like the `sse2` kernel this one needs no
//! runtime detection and no fallback: if the target is `aarch64`, these instructions exist.
//!
//! # Constant time
//!
//! `veorq_u64` and `vandq_u64` are fixed-latency, data-independent operations. The sequence
//! and its length depend only on the field's extension degree, a compile-time constant of the
//! parameter set. Nothing here branches on, or indexes with, a field element.

use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64};

use crate::hazmat::vec::MAX_BITS;

/// Reinterpret a bit-plane as a vector register.
///
/// `u128` and `uint64x2_t` are both sixteen bytes, and the operations below are bitwise, so
/// the lane interpretation is irrelevant.
#[inline(always)]
fn to_vector(value: u128) -> uint64x2_t {
    // SAFETY: identical size, and every bit pattern is valid for both types.
    unsafe { core::mem::transmute(value) }
}

/// The inverse of [`to_vector`].
#[inline(always)]
fn from_vector(value: uint64x2_t) -> u128 {
    // SAFETY: as above.
    unsafe { core::mem::transmute(value) }
}

/// `a ^ (b & c)`, the Karatsuba inner step.
///
/// With the `sha3` extension -- baseline on Apple silicon -- the step is one `BCAX`, whose
/// shape is `a ^ (b & !c)`; [`mul`] pre-complements the `c` side once so the convolution's
/// one hundred and thirty-four `AND`/`XOR` pairs become that many single instructions, the
/// same fusion `vpternlogq` gives the x86 kernel. Without the extension the plain two-op
/// form runs unchanged.
#[cfg(target_feature = "sha3")]
#[inline(always)]
fn and_xor(a: uint64x2_t, b: uint64x2_t, not_c: uint64x2_t) -> uint64x2_t {
    use core::arch::aarch64::vbcaxq_u64;
    // SAFETY: the `cfg` above guarantees the `sha3` target feature, and the instruction
    // does not touch memory.
    unsafe { vbcaxq_u64(a, b, not_c) }
}

/// `a ^ (b & c)`, the Karatsuba inner step.
#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
fn and_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    use core::arch::aarch64::vandq_u64;
    // SAFETY: `neon` is part of the AArch64 baseline, so both intrinsics are unconditionally
    // present on this target. Neither touches memory.
    unsafe { veorq_u64(a, vandq_u64(b, c)) }
}

/// Multiply two bit-sliced groups lane by lane, holding every plane in a vector register.
///
/// Same Karatsuba split and same reduction as the portable form in `vec.rs`; the difference is
/// only which register file the planes live in.
#[inline]
pub(crate) fn mul<const BITS: usize>(
    out: &mut [u128; MAX_BITS],
    a: &[u128; MAX_BITS],
    b: &[u128; MAX_BITS],
) {
    debug_assert!(BITS == 12 || BITS == 13);

    // SAFETY: every intrinsic reached from here is `neon`, which is part of the AArch64
    // baseline and so unconditionally present on this target. The `unsafe` is a formality of
    // the intrinsic signatures, not a real precondition. None of them touches memory.
    unsafe {
        // The split that balances the two halves: seven and six for degree thirteen, six and
        // six for degree twelve.
        let s = if BITS == 12 { 6 } else { 7 };
        let u = 6usize;

        let zero = vdupq_n_u64(0);
        let mut av = [zero; MAX_BITS];
        let mut bv = [zero; MAX_BITS];
        for i in 0..BITS {
            av[i] = to_vector(a[i]);
            bv[i] = to_vector(b[i]);
        }

        let mut a_sum = [zero; 7];
        let mut b_sum = [zero; 7];
        for i in 0..u {
            a_sum[i] = veorq_u64(av[i], av[i + s]);
            b_sum[i] = veorq_u64(bv[i], bv[i + s]);
        }
        if u < s {
            a_sum[u] = av[u];
            b_sum[u] = bv[u];
        }

        // `BCAX` takes its third operand complemented, so complement the whole `b` side once
        // -- twenty operations buying one op off each of the hundred-plus inner steps. There
        // is no 64-bit `MVN` intrinsic; an exclusive-or with ones is the same one instruction.
        #[cfg(target_feature = "sha3")]
        {
            let ones = vdupq_n_u64(u64::MAX);
            for plane in bv.iter_mut().take(BITS) {
                *plane = veorq_u64(*plane, ones);
            }
            for sum in b_sum.iter_mut().take(s) {
                *sum = veorq_u64(*sum, ones);
            }
        }

        let mut low = [zero; MAX_BITS];
        let mut middle = [zero; MAX_BITS];
        let mut high = [zero; MAX_BITS];
        for i in 0..s {
            for j in 0..s {
                low[i + j] = and_xor(low[i + j], av[i], bv[j]);
                middle[i + j] = and_xor(middle[i + j], a_sum[i], b_sum[j]);
            }
        }
        for i in 0..u {
            for j in 0..u {
                high[i + j] = and_xor(high[i + j], av[i + s], bv[j + s]);
            }
        }

        let mut product = [zero; 2 * MAX_BITS - 1];
        for i in 0..2 * s - 1 {
            product[i] = veorq_u64(product[i], low[i]);
            let combined = veorq_u64(veorq_u64(middle[i], low[i]), high[i]);
            product[i + s] = veorq_u64(product[i + s], combined);
        }
        for i in 0..2 * u - 1 {
            product[i + 2 * s] = veorq_u64(product[i + 2 * s], high[i]);
        }

        // Fold the top half down through the standardized sparse field polynomial, highest
        // degree first so each term is complete before it is folded: z^12 = z^3 + 1, and
        // z^13 = z^4 + z^3 + z + 1.
        if BITS == 12 {
            for d in (0..11).rev() {
                let top = product[12 + d];
                product[d + 3] = veorq_u64(product[d + 3], top);
                product[d] = veorq_u64(product[d], top);
            }
        } else {
            for d in (0..12).rev() {
                let top = product[13 + d];
                product[d + 4] = veorq_u64(product[d + 4], top);
                product[d + 3] = veorq_u64(product[d + 3], top);
                product[d + 1] = veorq_u64(product[d + 1], top);
                product[d] = veorq_u64(product[d], top);
            }
        }

        for i in 0..BITS {
            out[i] = from_vector(product[i]);
        }
    }
    for slot in out.iter_mut().skip(BITS) {
        *slot = 0;
    }
}

/// Square a bit-sliced group lane by lane, holding every plane in a vector register.
///
/// Squaring is `F_2`-linear in characteristic two, so this is a fixed exclusive-or pattern with
/// no multiplication at all — the same pattern the portable form uses, in the other register
/// file. `Tables::inv` reaches it a dozen times per inversion through the Itoh-Tsujii chain,
/// and the additive transform reaches it again.
#[inline]
pub(crate) fn sq<const BITS: usize>(out: &mut [u128; MAX_BITS], a: &[u128; MAX_BITS]) {
    debug_assert!(BITS == 12 || BITS == 13);

    // SAFETY: `neon` is part of the AArch64 baseline, so every intrinsic here is
    // unconditionally present on this target. None of them touches memory.
    unsafe {
        let zero = vdupq_n_u64(0);
        let mut v = [zero; MAX_BITS];
        for i in 0..BITS {
            v[i] = to_vector(a[i]);
        }
        let x = |i: usize, j: usize| veorq_u64(v[i], v[j]);

        let mut r = [zero; MAX_BITS];
        if BITS == 12 {
            r[0] = x(0, 6);
            r[1] = v[11];
            r[2] = x(1, 7);
            r[3] = v[6];
            r[4] = veorq_u64(x(2, 8), v[11]);
            r[5] = v[7];
            r[6] = x(3, 9);
            r[7] = v[8];
            r[8] = x(4, 10);
            r[9] = v[9];
            r[10] = x(5, 11);
            r[11] = v[10];
        } else {
            let t = x(11, 12);
            r[0] = x(0, 11);
            r[1] = veorq_u64(v[7], t);
            r[2] = x(1, 7);
            r[3] = veorq_u64(v[8], t);
            r[4] = veorq_u64(veorq_u64(x(2, 7), v[8]), t);
            r[5] = x(7, 9);
            r[6] = veorq_u64(veorq_u64(x(3, 8), v[9]), v[12]);
            r[7] = x(8, 10);
            r[8] = veorq_u64(x(4, 9), v[10]);
            r[9] = x(9, 11);
            r[10] = veorq_u64(x(5, 10), v[11]);
            r[11] = x(10, 12);
            r[12] = veorq_u64(v[6], t);
        }

        for i in 0..BITS {
            out[i] = from_vector(r[i]);
        }
    }
    for slot in out.iter_mut().skip(BITS) {
        *slot = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitMix(u64);

    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn word(&mut self) -> u128 {
            (u128::from(self.next()) << 64) | u128::from(self.next())
        }
    }

    /// The register form must agree with the portable one on every plane.
    fn matches_the_portable_multiply<const BITS: usize>(seed: u64) {
        let mut rng = SplitMix(seed);
        for _ in 0..2000 {
            let mut a = [0u128; MAX_BITS];
            let mut b = [0u128; MAX_BITS];
            for plane in 0..BITS {
                a[plane] = rng.word();
                b[plane] = rng.word();
            }

            let mut expected = [0u128; MAX_BITS];
            crate::hazmat::vec::portable_mul::<BITS>(&mut expected, &a, &b);

            let mut actual = [0u128; MAX_BITS];
            mul::<BITS>(&mut actual, &a, &b);

            assert_eq!(actual, expected);
        }
    }

    /// The register form of squaring must agree with the portable one on every plane.
    fn squaring_matches_the_portable_form<const BITS: usize>(seed: u64) {
        let mut rng = SplitMix(seed);
        for _ in 0..2000 {
            let mut a = [0u128; MAX_BITS];
            for slot in a.iter_mut().take(BITS) {
                *slot = rng.word();
            }

            // Squaring is multiplication by self, which is the independent statement of what
            // the exclusive-or pattern is supposed to compute -- checking against that rather
            // than against a copy of the same pattern is what makes this test able to fail.
            let mut expected = [0u128; MAX_BITS];
            crate::hazmat::vec::portable_mul::<BITS>(&mut expected, &a, &a);

            let mut actual = [0u128; MAX_BITS];
            sq::<BITS>(&mut actual, &a);
            assert_eq!(actual, expected);

            // The portable pattern is what the remaining targets run, so keep it exercised
            // here too.
            let mut portable = [0u128; MAX_BITS];
            crate::hazmat::vec::portable_sq::<BITS>(&mut portable, &a);
            assert_eq!(portable, expected);
        }
    }

    #[test]
    fn twelve_bit_squaring_matches() {
        squaring_matches_the_portable_form::<12>(0x2545_F491_4F6C_DD1D);
    }

    #[test]
    fn thirteen_bit_squaring_matches() {
        squaring_matches_the_portable_form::<13>(0xBF58_476D_1CE4_E5B9);
    }

    #[test]
    fn twelve_bit_field_matches() {
        matches_the_portable_multiply::<12>(0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn thirteen_bit_field_matches() {
        matches_the_portable_multiply::<13>(0x0FED_CBA9_8765_4321);
    }

    /// Zero and one are where a convolution is most likely to be wrong at the edges.
    #[test]
    fn zero_and_one_behave() {
        let mut rng = SplitMix(0xDEAD_BEEF_CAFE_F00D);
        let mut a = [0u128; MAX_BITS];
        for slot in a.iter_mut().take(13) {
            *slot = rng.word();
        }

        let zero = [0u128; MAX_BITS];
        let mut one = [0u128; MAX_BITS];
        one[0] = u128::MAX;

        let mut got = [0u128; MAX_BITS];
        mul::<13>(&mut got, &a, &zero);
        assert_eq!(got, [0u128; MAX_BITS]);

        mul::<13>(&mut got, &a, &one);
        assert_eq!(got, a);
    }
}
