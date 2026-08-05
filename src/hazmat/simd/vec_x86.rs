/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! The x86 form of the bit-sliced field multiplication.
//!
//! # Why this exists
//!
//! `vec.rs` picks `u128` as the bit-plane word for its width, but `u128` is an *integer* type:
//! on x86-64 values arrive in pairs of general-purpose registers and have to be assembled into
//! `xmm` before a vector operation and taken apart afterwards. Compiling the thirteen-plane
//! exclusive-or on its own emits fifteen `movaps`, six `movq`, four `movsd` and two `movlhps`
//! around only nine actual exclusive-ors — more data movement than arithmetic. AArch64 has the
//! same problem in a different costume — rustc keeps `u128` in general-register pairs there
//! too, and the convolution's live planes overflow the integer file into stack spills — which
//! is why `vec_neon` exists as this module's sibling.
//!
//! Naming the register type directly removes the round trip. `sse2` is part of the x86-64
//! baseline, so unlike every other kernel here this one needs no runtime detection and no
//! fallback: if the target is `x86_64`, these instructions exist.
//!
//! # Constant time
//!
//! `_mm_xor_si128` and `_mm_and_si128` are fixed-latency, data-independent operations. The
//! sequence and its length depend only on the field's extension degree, a compile-time constant of
//! the parameter set. Nothing here branches on, or indexes with, a field element.

use core::arch::x86_64::{__m128i, _mm_and_si128, _mm_setzero_si128, _mm_xor_si128};

use super::{Level, level};

use crate::hazmat::vec::MAX_BITS;

/// Reinterpret a bit-plane as a vector register.
///
/// `u128` and `__m128i` are both sixteen bytes with sixteen-byte alignment, and the operations
/// below are bitwise, so the lane interpretation is irrelevant.
#[inline(always)]
fn to_vector(value: u128) -> __m128i {
    // SAFETY: identical size and alignment, and every bit pattern is valid for both types.
    unsafe { core::mem::transmute(value) }
}

/// The inverse of [`to_vector`].
#[inline(always)]
fn from_vector(value: __m128i) -> u128 {
    // SAFETY: as above.
    unsafe { core::mem::transmute(value) }
}

/// `a ^ (b & c)` with plain `sse2`, which is the baseline form of the Karatsuba inner step.
#[inline(always)]
fn plain_and_xor(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    // SAFETY: `sse2` is part of the x86-64 baseline, so both intrinsics are unconditionally
    // present on this target. Neither touches memory.
    unsafe { _mm_xor_si128(a, _mm_and_si128(b, c)) }
}

/// The bit-sliced multiply, parameterized on how its inner step accumulates and on how planes
/// enter and leave the registers.
///
/// The instantiations differ in the accumulate operation and the register width and nothing
/// else, so the convolution and the reduction live here once rather than in each. A macro
/// rather than a function parameter because the fused instantiations carry
/// `#[target_feature]`, and a closure passed across that boundary would not inline.
macro_rules! karatsuba_multiply {
    ($accumulate:ident, $xor:ident, $zero:expr, $load:ident, $store:ident,
     $a:ident, $b:ident, $bits:ident) => {{
        // The split that balances the two halves: seven and six for degree thirteen, six and six
        // for degree twelve.
        let s = if $bits == 12 { 6 } else { 7 };
        let u = 6usize;

        let zero = $zero;
        let mut av = [zero; MAX_BITS];
        let mut bv = [zero; MAX_BITS];
        for i in 0..$bits {
            av[i] = $load!($a, i);
            bv[i] = $load!($b, i);
        }

        let mut a_sum = [zero; 7];
        let mut b_sum = [zero; 7];
        for i in 0..u {
            a_sum[i] = $xor(av[i], av[i + s]);
            b_sum[i] = $xor(bv[i], bv[i + s]);
        }
        if u < s {
            a_sum[u] = av[u];
            b_sum[u] = bv[u];
        }

        let mut low = [zero; MAX_BITS];
        let mut middle = [zero; MAX_BITS];
        let mut high = [zero; MAX_BITS];
        for i in 0..s {
            for j in 0..s {
                low[i + j] = $accumulate(low[i + j], av[i], bv[j]);
                middle[i + j] = $accumulate(middle[i + j], a_sum[i], b_sum[j]);
            }
        }
        for i in 0..u {
            for j in 0..u {
                high[i + j] = $accumulate(high[i + j], av[i + s], bv[j + s]);
            }
        }

        let mut product = [zero; 2 * MAX_BITS - 1];
        for i in 0..2 * s - 1 {
            product[i] = $xor(product[i], low[i]);
            let combined = $xor($xor(middle[i], low[i]), high[i]);
            product[i + s] = $xor(product[i + s], combined);
        }
        for i in 0..2 * u - 1 {
            product[i + 2 * s] = $xor(product[i + 2 * s], high[i]);
        }

        // Fold the top half down through the standardized sparse field polynomial, highest degree
        // first so each term is complete before it is folded: z^12 = z^3 + 1, and
        // z^13 = z^4 + z^3 + z + 1.
        if $bits == 12 {
            for d in (0..11).rev() {
                let top = product[12 + d];
                product[d + 3] = $xor(product[d + 3], top);
                product[d] = $xor(product[d], top);
            }
        } else {
            for d in (0..12).rev() {
                let top = product[13 + d];
                product[d + 4] = $xor(product[d + 4], top);
                product[d + 3] = $xor(product[d + 3], top);
                product[d + 1] = $xor(product[d + 1], top);
                product[d] = $xor(product[d], top);
            }
        }

        for i in 0..$bits {
            $store!(i, product[i]);
        }
        for i in $bits..MAX_BITS {
            $store!(i, zero);
        }
    }};
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

    macro_rules! load {
        ($src:ident, $i:expr) => {
            to_vector($src[$i])
        };
    }
    macro_rules! store {
        ($i:expr, $v:expr) => {
            out[$i] = from_vector($v)
        };
    }
    // SAFETY: every intrinsic reached from here is `sse2`, which is part of the x86-64 baseline
    // and so unconditionally present on this target. The `unsafe` is a formality of the intrinsic
    // signatures, not a real precondition. None of them touches memory.
    unsafe {
        karatsuba_multiply!(
            plain_and_xor,
            _mm_xor_si128,
            _mm_setzero_si128(),
            load,
            store,
            a,
            b,
            BITS
        );
    }
}

/// The same multiply with the inner step fused into one instruction.
///
/// `vpternlogq` computes `a ^ (b & c)` directly, so the convolution's one hundred and thirty-four
/// `AND`/`XOR` pairs become one hundred and thirty-four single operations. At 128-bit width that
/// needs `avx512vl` as well as `avx512f`.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512vl`.
#[target_feature(enable = "avx512f,avx512vl")]
pub(crate) unsafe fn mul_fused<const BITS: usize>(
    out: &mut [u128; MAX_BITS],
    a: &[u128; MAX_BITS],
    b: &[u128; MAX_BITS],
) {
    debug_assert!(BITS == 12 || BITS == 13);

    macro_rules! load {
        ($src:ident, $i:expr) => {
            to_vector($src[$i])
        };
    }
    macro_rules! store {
        ($i:expr, $v:expr) => {
            out[$i] = from_vector($v)
        };
    }
    // SAFETY: this function's own target features cover every intrinsic reached from here, and
    // none of them touches memory.
    unsafe {
        karatsuba_multiply!(
            fused_and_xor,
            _mm_xor_si128,
            _mm_setzero_si128(),
            load,
            store,
            a,
            b,
            BITS
        );
    }
}

/// Two independent multiplies in one pass, one per 128-bit half of each 256-bit register.
///
/// Berlekamp--Massey scales two polynomials by two independent constants at every step;
/// carrying both multiplies in one register file halves the convolution's instruction count
/// and lets every spill of the register-starved Karatsuba serve both at once.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512vl`.
#[cfg(any(feature = "decapsulate", test))]
#[target_feature(enable = "avx512f,avx512vl")]
pub(crate) unsafe fn mul_fused_pair<const BITS: usize>(
    out: (&mut [u128; MAX_BITS], &mut [u128; MAX_BITS]),
    a: (&[u128; MAX_BITS], &[u128; MAX_BITS]),
    b: (&[u128; MAX_BITS], &[u128; MAX_BITS]),
) {
    use core::arch::x86_64::_mm256_setzero_si256;

    debug_assert!(BITS == 12 || BITS == 13);

    let (out0, out1) = out;
    macro_rules! load {
        ($src:ident, $i:expr) => {
            pack_pair($src.0[$i], $src.1[$i])
        };
    }
    macro_rules! store {
        ($i:expr, $v:expr) => {{
            let value = $v;
            out0[$i] = low_half(value);
            out1[$i] = high_half(value);
        }};
    }
    // SAFETY: this function's own target features cover every intrinsic reached from here
    // (`avx512vl` implies the `avx2` forms), and none of them touches memory.
    unsafe {
        karatsuba_multiply!(
            fused_and_xor_pair,
            xor_pair,
            _mm256_setzero_si256(),
            load,
            store,
            a,
            b,
            BITS
        );
    }
}

/// Square a bit-sliced group lane by lane, holding every plane in a vector register.
///
/// Squaring is `F_2`-linear in characteristic two, so this is a fixed exclusive-or pattern with no
/// multiplication at all — the same pattern the portable form uses, in the other register file.
/// `Tables::inv` reaches it a dozen times per inversion through the Itoh-Tsujii chain, and the
/// additive transform reaches it again.
#[inline]
pub(crate) fn sq<const BITS: usize>(out: &mut [u128; MAX_BITS], a: &[u128; MAX_BITS]) {
    debug_assert!(BITS == 12 || BITS == 13);

    // SAFETY: `sse2` is part of the x86-64 baseline, so every intrinsic here is unconditionally
    // present on this target. None of them touches memory.
    unsafe {
        let zero = _mm_setzero_si128();
        let mut v = [zero; MAX_BITS];
        for i in 0..BITS {
            v[i] = to_vector(a[i]);
        }
        let x = |i: usize, j: usize| _mm_xor_si128(v[i], v[j]);

        let mut r = [zero; MAX_BITS];
        if BITS == 12 {
            r[0] = x(0, 6);
            r[1] = v[11];
            r[2] = x(1, 7);
            r[3] = v[6];
            r[4] = _mm_xor_si128(x(2, 8), v[11]);
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
            r[1] = _mm_xor_si128(v[7], t);
            r[2] = x(1, 7);
            r[3] = _mm_xor_si128(v[8], t);
            r[4] = _mm_xor_si128(_mm_xor_si128(x(2, 7), v[8]), t);
            r[5] = x(7, 9);
            r[6] = _mm_xor_si128(_mm_xor_si128(x(3, 8), v[9]), v[12]);
            r[7] = x(8, 10);
            r[8] = _mm_xor_si128(x(4, 9), v[10]);
            r[9] = x(9, 11);
            r[10] = _mm_xor_si128(x(5, 10), v[11]);
            r[11] = x(10, 12);
            r[12] = _mm_xor_si128(v[6], t);
        }

        for i in 0..BITS {
            out[i] = from_vector(r[i]);
        }
    }

    for slot in out.iter_mut().skip(BITS) {
        *slot = 0;
    }
}

/// Whether the fused form below is available.
///
/// Read once per `Tables`, not per multiply: at roughly two hundred cycles a multiply, an atomic
/// load per call would be a couple of percent for nothing.
#[inline]
pub(crate) fn has_fused_and_xor() -> bool {
    level() == Level::Avx512
}

/// `a ^ (b & c)` in one instruction.
///
/// `0x78` is the truth table of that expression. This is exactly the shape of the Karatsuba
/// inner step, which is why the fused form nearly halves the convolution: one hundred and
/// thirty-four `AND`/`XOR` pairs become one hundred and thirty-four single operations.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512vl`; the 128-bit width is what needs `vl`.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn fused_and_xor(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    use core::arch::x86_64::_mm_ternarylogic_epi64;
    _mm_ternarylogic_epi64::<0x78>(a, b, c)
}

/// The 256-bit form of [`fused_and_xor`], carrying two independent lanes.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512vl`; the 256-bit width is what needs `vl`.
#[cfg(any(feature = "decapsulate", test))]
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn fused_and_xor_pair(
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
    c: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::_mm256_ternarylogic_epi64;
    _mm256_ternarylogic_epi64::<0x78>(a, b, c)
}

/// The 256-bit exclusive-or, named so the shared Karatsuba body can take it as a parameter.
#[cfg(any(feature = "decapsulate", test))]
#[inline(always)]
fn xor_pair(
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    // SAFETY: only reached from `mul_fused_pair`, whose target features imply `avx2`. The
    // operation is fixed-latency and touches no memory.
    unsafe { core::arch::x86_64::_mm256_xor_si256(a, b) }
}

/// Two bit-planes side by side in one 256-bit register.
#[cfg(any(feature = "decapsulate", test))]
#[inline(always)]
fn pack_pair(low: u128, high: u128) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::{_mm256_inserti128_si256, _mm256_zextsi128_si256};
    // SAFETY: as in `xor_pair`; both intrinsics are register-only `avx`/`avx2` forms.
    unsafe { _mm256_inserti128_si256::<1>(_mm256_zextsi128_si256(to_vector(low)), to_vector(high)) }
}

/// The low half of a packed pair.
#[cfg(any(feature = "decapsulate", test))]
#[inline(always)]
fn low_half(value: core::arch::x86_64::__m256i) -> u128 {
    // SAFETY: as in `xor_pair`.
    unsafe { from_vector(core::arch::x86_64::_mm256_castsi256_si128(value)) }
}

/// The high half of a packed pair.
#[cfg(any(feature = "decapsulate", test))]
#[inline(always)]
fn high_half(value: core::arch::x86_64::__m256i) -> u128 {
    // SAFETY: as in `xor_pair`.
    unsafe { from_vector(core::arch::x86_64::_mm256_extracti128_si256::<1>(value)) }
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

    /// The paired fused multiply must agree with the portable multiply in both halves.
    fn pair_matches_the_portable_multiply<const BITS: usize>(seed: u64) {
        if !has_fused_and_xor() {
            return;
        }

        let mut rng = SplitMix(seed);
        for _ in 0..2000 {
            let mut a = [[0u128; MAX_BITS]; 2];
            let mut b = [[0u128; MAX_BITS]; 2];
            for side in 0..2 {
                for plane in 0..BITS {
                    a[side][plane] = rng.word();
                    b[side][plane] = rng.word();
                }
            }

            let mut expected = [[0u128; MAX_BITS]; 2];
            for side in 0..2 {
                crate::hazmat::vec::portable_mul::<BITS>(&mut expected[side], &a[side], &b[side]);
            }

            let mut actual = [[0u128; MAX_BITS]; 2];
            let (first, second) = actual.split_at_mut(1);
            // SAFETY: `has_fused_and_xor` reported the required target features.
            unsafe {
                mul_fused_pair::<BITS>(
                    (&mut first[0], &mut second[0]),
                    (&a[0], &a[1]),
                    (&b[0], &b[1]),
                );
            }

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn twelve_bit_pair_multiply_matches() {
        pair_matches_the_portable_multiply::<12>(0x0F0E_0D0C_0B0A_0908);
    }

    #[test]
    fn thirteen_bit_pair_multiply_matches() {
        pair_matches_the_portable_multiply::<13>(0x1357_9BDF_0246_8ACE);
    }

    /// The register form of squaring must agree with the portable one on every plane.
    fn squaring_matches_the_portable_form<const BITS: usize>(seed: u64) {
        let mut rng = SplitMix(seed);
        for _ in 0..2000 {
            let mut a = [0u128; MAX_BITS];
            for slot in a.iter_mut().take(BITS) {
                *slot = rng.word();
            }

            // Squaring is multiplication by self, which is the independent statement of what the
            // exclusive-or pattern is supposed to compute -- checking against that rather than
            // against a copy of the same pattern is what makes this test able to fail.
            let mut expected = [0u128; MAX_BITS];
            crate::hazmat::vec::portable_mul::<BITS>(&mut expected, &a, &a);

            let mut actual = [0u128; MAX_BITS];
            sq::<BITS>(&mut actual, &a);
            assert_eq!(actual, expected);

            // The portable pattern is what non-x86 targets run, so keep it exercised here too.
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

    /// The fused form must agree with the plain one on every plane.
    fn fused_matches_the_plain_form<const BITS: usize>(seed: u64) {
        if !has_fused_and_xor() {
            return;
        }
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
            // SAFETY: guarded by the detection check above.
            unsafe { mul_fused::<BITS>(&mut actual, &a, &b) };

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn twelve_bit_fused_matches() {
        fused_matches_the_plain_form::<12>(0x0BAD_C0DE_1234_5678);
    }

    #[test]
    fn thirteen_bit_fused_matches() {
        fused_matches_the_plain_form::<13>(0x7777_8888_9999_AAAA);
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
