/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Bit-sliced arithmetic over `F_q`.
//!
//! A slice holds 128 field elements as `m` words: word `i` carries bit `i` of every element, one
//! per lane. Arithmetic then becomes bitwise operations on whole words, so a single
//! multiplication instruction sequence serves the entire group at once.
//!
//! Reduction and squaring use the fixed formulas for the two standardized fields. Exposing
//! those sparse formulas directly lets the compiler keep the bit-planes in registers instead
//! of routing them through runtime index tables.
//!
//! # Why the decoder keeps polynomials bit-sliced
//!
//! Measured against the scalar multiplier on an Apple M-series core, with both loops given
//! enough independent work to run at throughput rather than latency:
//!
//! | multiplier | ns per element | versus scalar |
//! | ---------- | -------------- | ------------- |
//! | scalar, carry-less multiply | 2.83 | 1.00x |
//! | bit-sliced, 64-bit lanes | 1.71 | 1.66x |
//! | bit-sliced, 128-bit lanes | 0.58 | 4.88x |
//!
//! A hardware carry-less multiply is already good, so 64-bit bit-slicing on its own wins too
//! little to pay for converting in and out of this representation. The decoder therefore uses
//! wide lanes and keeps the Berlekamp--Massey polynomial bit-sliced throughout its update loop,
//! so conversion happens only at the boundary.

use super::field::Field;

/// The widest field this crate uses, which bounds every per-element array below.
pub(crate) const MAX_BITS: usize = 13;

/// The machine word backing one bit-plane.
///
/// The width is the whole point: a hardware carry-less multiply already runs at about
/// 2.8 ns per element, and 64-bit bit-slicing only reaches 1.7, which does not pay for the
/// representation. At 128 bits it reaches 0.58. LLVM lowers `u128` bitwise operations to the
/// target's vector registers, so this stays portable rather than needing intrinsics.
pub(crate) type Word = u128;

/// The number of field elements one bit-sliced group holds.
pub(crate) const LANES: usize = Word::BITS as usize;

/// A bit-sliced group of [`LANES`] field elements.
///
/// Only the first `F::BITS` words carry data; the rest stay zero so that a single fixed-size
/// array serves both field widths.
pub(crate) type Slice = [Word; MAX_BITS];

/// A narrow bit-sliced group used by the degree-64 Berlekamp--Massey specialization.
#[cfg(feature = "decapsulate")]
pub(crate) type Slice64 = [u64; MAX_BITS];

/// Bit-sliced arithmetic selected by field type.
pub(crate) struct Tables<F: Field> {
    field: core::marker::PhantomData<F>,
    /// Whether the fused `AND`/`XOR` multiply is available, decided once at construction so the
    /// hot path never pays for the check.
    #[cfg(target_arch = "x86_64")]
    fused: bool,
}

macro_rules! karatsuba_product {
    ($name:ident, $word:ty, $slice:ty, $split:expr, $upper:expr) => {
        #[inline(always)]
        fn $name(a: &$slice, b: &$slice) -> [$word; 2 * MAX_BITS - 1] {
            const SPLIT: usize = $split;
            const UPPER: usize = $upper;

            let mut a_sum: [$word; 7] = Default::default();
            let mut b_sum: [$word; 7] = Default::default();
            for i in 0..UPPER {
                a_sum[i] = a[i] ^ a[i + SPLIT];
                b_sum[i] = b[i] ^ b[i + SPLIT];
            }
            if UPPER < SPLIT {
                a_sum[UPPER] = a[UPPER];
                b_sum[UPPER] = b[UPPER];
            }

            let mut low: [$word; 13] = Default::default();
            let mut middle: [$word; 13] = Default::default();
            let mut high: [$word; 13] = Default::default();
            for i in 0..SPLIT {
                for j in 0..SPLIT {
                    low[i + j] ^= a[i] & b[j];
                    middle[i + j] ^= a_sum[i] & b_sum[j];
                }
            }
            for i in 0..UPPER {
                for j in 0..UPPER {
                    high[i + j] ^= a[i + SPLIT] & b[j + SPLIT];
                }
            }

            let mut product: [$word; 2 * MAX_BITS - 1] = Default::default();
            for i in 0..2 * SPLIT - 1 {
                product[i] ^= low[i];
                product[i + SPLIT] ^= middle[i] ^ low[i] ^ high[i];
            }
            for i in 0..2 * UPPER - 1 {
                product[i + 2 * SPLIT] ^= high[i];
            }
            product
        }
    };
}

#[cfg(any(not(target_arch = "x86_64"), test))]
karatsuba_product!(product12, Word, Slice, 6, 6);
#[cfg(any(not(target_arch = "x86_64"), test))]
karatsuba_product!(product13, Word, Slice, 7, 6);
#[cfg(feature = "decapsulate")]
karatsuba_product!(product12_64, u64, Slice64, 6, 6);
#[cfg(feature = "decapsulate")]
karatsuba_product!(product13_64, u64, Slice64, 7, 6);

/// Multiply two bit-sliced groups lane by lane, portably.
///
/// The schoolbook convolution gives a product of degree up to `2m - 2`; the top half folds back
/// through the standardized sparse field polynomial. Every operation is a word-wide `AND` or
/// `XOR`, so all [`LANES`] lanes advance together. This is the reference the x86 kernel is
/// checked against, and the implementation every non-x86 target runs.
#[cfg(any(not(target_arch = "x86_64"), test))]
pub(crate) fn portable_mul<const BITS: usize>(out: &mut Slice, a: &Slice, b: &Slice) {
    let mut product = if BITS == 12 {
        product12(a, b)
    } else {
        debug_assert_eq!(BITS, 13);
        product13(a, b)
    };

    // Fold from the top so that each term is complete before it is folded. These are the two
    // standardized field polynomials: z^12 = z^3 + 1 and z^13 = z^4 + z^3 + z + 1.
    if BITS == 12 {
        for d in (0..11).rev() {
            let high = product[12 + d];
            product[d + 3] ^= high;
            product[d] ^= high;
        }
    } else {
        for d in (0..12).rev() {
            let high = product[13 + d];
            product[d + 4] ^= high;
            product[d + 3] ^= high;
            product[d + 1] ^= high;
            product[d] ^= high;
        }
    }

    out[..BITS].copy_from_slice(&product[..BITS]);
    out[BITS..].fill(0);
}

/// Square a bit-sliced group lane by lane, portably.
///
/// Squaring is `F_2`-linear in characteristic two, so it is just a fixed exclusive-or pattern
/// across the words with no multiplication at all.
#[cfg(any(not(target_arch = "x86_64"), test))]
pub(crate) fn portable_sq<const BITS: usize>(out: &mut Slice, a: &Slice) {
    let mut result = [0; MAX_BITS];
    if BITS == 12 {
        result[0] = a[0] ^ a[6];
        result[1] = a[11];
        result[2] = a[1] ^ a[7];
        result[3] = a[6];
        result[4] = a[2] ^ a[8] ^ a[11];
        result[5] = a[7];
        result[6] = a[3] ^ a[9];
        result[7] = a[8];
        result[8] = a[4] ^ a[10];
        result[9] = a[9];
        result[10] = a[5] ^ a[11];
        result[11] = a[10];
    } else {
        debug_assert_eq!(BITS, 13);
        let t = a[11] ^ a[12];
        result[0] = a[0] ^ a[11];
        result[1] = a[7] ^ t;
        result[2] = a[1] ^ a[7];
        result[3] = a[8] ^ t;
        result[4] = a[2] ^ a[7] ^ a[8] ^ t;
        result[5] = a[7] ^ a[9];
        result[6] = a[3] ^ a[8] ^ a[9] ^ a[12];
        result[7] = a[8] ^ a[10];
        result[8] = a[4] ^ a[9] ^ a[10];
        result[9] = a[9] ^ a[11];
        result[10] = a[5] ^ a[10] ^ a[11];
        result[11] = a[10] ^ a[12];
        result[12] = a[6] ^ t;
    }
    *out = result;
}

impl<F: Field> Tables<F> {
    /// Select the field arithmetic at compile time.
    pub(crate) fn new() -> Self {
        Self {
            field: core::marker::PhantomData,
            #[cfg(target_arch = "x86_64")]
            fused: crate::hazmat::simd::vec_x86::has_fused_and_xor(),
        }
    }

    /// Multiply two bit-sliced groups lane by lane.
    ///
    /// The schoolbook convolution gives a product of degree up to `2m - 2`; the top half folds
    /// back through the standardized sparse field polynomial. Every operation is a word-wide
    /// `AND` or `XOR`, so
    /// all [`LANES`] lanes advance together.
    pub(crate) fn mul(&self, out: &mut Slice, a: &Slice, b: &Slice) {
        // `BITS` is threaded as a constant rather than read from `F` at run time. This is the
        // crate's hottest routine -- Berlekamp-Massey calls it three times per iteration for 2t
        // iterations, and the additive transform's butterflies call it again -- and letting the
        // field width fold away at compile time measured 16% off a whole decapsulation.
        #[cfg(target_arch = "x86_64")]
        {
            // On x86 the planes go in vector registers explicitly; see `simd::vec_x86` for why
            // `u128` does not get there on its own. The `sse2` form needs no detection -- it is
            // baseline -- while the fused form needs `avx512vl` and so is selected from the flag
            // this `Tables` recorded when it was built.
            use crate::hazmat::simd::vec_x86;
            if self.fused {
                // SAFETY: `fused` was set from a CPUID probe, so `avx512f` and `avx512vl` are
                // both present. `out` is a `&mut` borrow and cannot alias either operand.
                unsafe {
                    if F::BITS == 12 {
                        vec_x86::mul_fused::<12>(out, a, b);
                    } else {
                        debug_assert_eq!(F::BITS, 13);
                        vec_x86::mul_fused::<13>(out, a, b);
                    }
                }
            } else if F::BITS == 12 {
                vec_x86::mul::<12>(out, a, b);
            } else {
                debug_assert_eq!(F::BITS, 13);
                vec_x86::mul::<13>(out, a, b);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            if F::BITS == 12 {
                portable_mul::<12>(out, a, b);
            } else {
                debug_assert_eq!(F::BITS, 13);
                portable_mul::<13>(out, a, b);
            }
        }
    }

    /// Multiply two 64-lane bit-sliced groups lane by lane.
    ///
    /// Degree 64 has exactly one machine word of non-leading coefficients, so using a `u64`
    /// avoids issuing both halves of every `u128` operation for an otherwise empty half.
    #[cfg(feature = "decapsulate")]
    pub(crate) fn mul64(&self, out: &mut Slice64, a: &Slice64, b: &Slice64) {
        let bits = F::BITS;
        let mut product = if F::BITS == 12 {
            product12_64(a, b)
        } else {
            debug_assert_eq!(F::BITS, 13);
            product13_64(a, b)
        };

        if F::BITS == 12 {
            for d in (0..11).rev() {
                let high = product[12 + d];
                product[d + 3] ^= high;
                product[d] ^= high;
            }
        } else {
            debug_assert_eq!(F::BITS, 13);
            for d in (0..12).rev() {
                let high = product[13 + d];
                product[d + 4] ^= high;
                product[d + 3] ^= high;
                product[d + 1] ^= high;
                product[d] ^= high;
            }
        }

        out[..bits].copy_from_slice(&product[..bits]);
        out[bits..].fill(0);
    }

    /// Square a bit-sliced group lane by lane.
    ///
    /// Squaring is `F_2`-linear in characteristic two, so it is just a fixed exclusive-or
    /// pattern across the words with no multiplication at all.
    pub(crate) fn sq(&self, out: &mut Slice, a: &Slice) {
        #[cfg(target_arch = "x86_64")]
        if F::BITS == 12 {
            crate::hazmat::simd::vec_x86::sq::<12>(out, a);
        } else {
            debug_assert_eq!(F::BITS, 13);
            crate::hazmat::simd::vec_x86::sq::<13>(out, a);
        }

        #[cfg(not(target_arch = "x86_64"))]
        if F::BITS == 12 {
            portable_sq::<12>(out, a);
        } else {
            debug_assert_eq!(F::BITS, 13);
            portable_sq::<13>(out, a);
        }
    }

    /// Multiply a bit-sliced group by a single field element in every lane.
    ///
    /// The decoder has no use for this: its butterflies multiply by a different element per
    /// lane. It stays as a statement of the cheap case, and as a test of the tables.
    ///
    /// The scalar is a public constant here, so branching on its bits leaks nothing. This is
    /// the cheap case: no convolution, just the exclusive-or pattern of one linear map.
    #[cfg(test)]
    pub(crate) fn mul_by_scalar(&self, out: &mut Slice, a: &Slice, scalar: u16) {
        let bits = F::BITS;
        let mut result = [0; MAX_BITS];

        for (j, &word) in a.iter().take(bits).enumerate() {
            let image = F::mul(scalar, 1u16 << j);
            for (i, slot) in result.iter_mut().take(bits).enumerate() {
                if (image >> i) & 1 == 1 {
                    *slot ^= word;
                }
            }
        }

        *out = result;
    }
}

/// The number of index bits a lane position covers.
pub(crate) const LANE_BITS: usize = LANES.trailing_zeros() as usize;

/// For each lane-index bit, the lanes where it is set.
///
/// Used to build a subspace sum directly in bit-sliced form: the bits of a lane index select
/// which basis elements contribute, so each basis element contributes to exactly the lanes one
/// of these masks picks out. Evaluated at compile time, since building one costs a loop over
/// every lane and they are wanted in the innermost part of the transform.
pub(crate) const LANE_BIT_MASKS: [Word; LANE_BITS] = {
    let mut masks = [0; LANE_BITS];
    let mut p = 0;
    while p < LANE_BITS {
        masks[p] = lane_bit_mask(p);
        p += 1;
    }
    masks
};

const fn lane_bit_mask(p: usize) -> Word {
    let mut mask: Word = 0;
    let mut lane = 0;
    while lane < LANES {
        if (lane >> p) & 1 == 1 {
            mask |= 1 << lane;
        }
        lane += 1;
    }
    mask
}

impl<F: Field> Tables<F> {
    /// Compute `a^(2^k - 1)` in every lane.
    ///
    /// The Itoh-Tsujii chain: `a^(2^(2h) - 1)` is `a^(2^h - 1)` raised to `2^h` times itself,
    /// so the exponent's run of ones doubles for one multiplication and `h` squarings. An odd
    /// length costs one more of each.
    fn power_of_ones(&self, out: &mut Slice, a: &Slice, k: usize) {
        if k == 1 {
            *out = *a;
            return;
        }

        let half = k / 2;
        let mut lower = [0; MAX_BITS];
        self.power_of_ones(&mut lower, a, half);

        let mut shifted = lower;
        for _ in 0..half {
            let input = shifted;
            self.sq(&mut shifted, &input);
        }

        let mut doubled = [0; MAX_BITS];
        self.mul(&mut doubled, &shifted, &lower);

        if k.is_multiple_of(2) {
            *out = doubled;
        } else {
            let mut squared = [0; MAX_BITS];
            self.sq(&mut squared, &doubled);
            self.mul(out, &squared, a);
        }
    }

    /// Invert every lane, by raising to the power `q - 2`.
    ///
    /// Bit-slicing has no cheap way to run Montgomery's trick, which needs a running product
    /// across elements rather than within them. Exponentiation is the better trade here anyway,
    /// because squaring is `F_2`-linear and so costs only exclusive-ors: this is four
    /// multiplications and a dozen squarings for a whole group. Zero inverts to zero.
    pub(crate) fn inv(&self, out: &mut Slice, a: &Slice) {
        let mut ones = [0; MAX_BITS];
        self.power_of_ones(&mut ones, a, F::BITS - 1);
        self.sq(out, &ones);
    }
}

/// Pack [`LANES`] field elements into a bit-sliced group.
///
/// The decoder never needs this: the transform builds its leaves directly as bit-planes, and
/// the syndrome consumes them the same way, so no conversion happens on the hot path.
#[cfg(test)]
pub(crate) fn pack<F: Field>(out: &mut Slice, elements: &[u16]) {
    debug_assert_eq!(elements.len(), LANES);

    out.fill(0);
    for (lane, &element) in elements.iter().enumerate() {
        for (i, word) in out.iter_mut().take(F::BITS).enumerate() {
            *word |= Word::from((element >> i) & 1) << lane;
        }
    }
}

/// Unpack a bit-sliced group back into [`LANES`] field elements.
///
/// Like [`pack`], only the tests need this; the decoder stays bit-sliced throughout.
#[cfg(test)]
pub(crate) fn unpack<F: Field>(out: &mut [u16], slice: &Slice) {
    debug_assert_eq!(out.len(), LANES);

    for (lane, element) in out.iter_mut().enumerate() {
        let mut value = 0u16;
        for i in (0..F::BITS).rev() {
            value = (value << 1) | (((slice[i] >> lane) & 1) as u16);
        }
        *element = value;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn elements<F: Field>(&mut self) -> Vec<u16> {
            (0..LANES).map(|_| (self.next() as u16) & F::MASK).collect()
        }
    }

    fn packing_round_trips<F: Field>(seed: u64) {
        let mut rng = Rng(seed);
        for _ in 0..8 {
            let elements = rng.elements::<F>();
            let mut slice = [0; MAX_BITS];
            pack::<F>(&mut slice, &elements);

            let mut back = vec![0u16; LANES];
            unpack::<F>(&mut back, &slice);
            assert_eq!(back, elements);
        }
    }

    /// Every bit-sliced operation must agree with the scalar one in every lane.
    fn matches_the_scalar_field<F: Field>(seed: u64) {
        let mut rng = Rng(seed);
        let tables = Tables::<F>::new();

        for _ in 0..16 {
            let left = rng.elements::<F>();
            let right = rng.elements::<F>();

            let mut a = [0; MAX_BITS];
            let mut b = [0; MAX_BITS];
            pack::<F>(&mut a, &left);
            pack::<F>(&mut b, &right);

            let mut product = [0; MAX_BITS];
            tables.mul(&mut product, &a, &b);
            let mut got = vec![0u16; LANES];
            unpack::<F>(&mut got, &product);
            for (lane, &value) in got.iter().enumerate() {
                assert_eq!(value, F::mul(left[lane], right[lane]), "mul lane {lane}");
            }

            let mut squared = [0; MAX_BITS];
            tables.sq(&mut squared, &a);
            unpack::<F>(&mut got, &squared);
            for (lane, &value) in got.iter().enumerate() {
                assert_eq!(value, F::sq(left[lane]), "sq lane {lane}");
            }

            for scalar in [0u16, 1, 2, 0x0ABC & F::MASK, F::MASK] {
                let mut scaled = [0; MAX_BITS];
                tables.mul_by_scalar(&mut scaled, &a, scalar);
                unpack::<F>(&mut got, &scaled);
                for (lane, &value) in got.iter().enumerate() {
                    assert_eq!(
                        value,
                        F::mul(left[lane], scalar),
                        "scalar {scalar} lane {lane}"
                    );
                }
            }
        }
    }

    /// Inversion must agree with the scalar field, including at zero.
    fn inversion_matches_the_scalar_field<F: Field>(seed: u64) {
        let mut rng = Rng(seed);
        let tables = Tables::<F>::new();

        for round in 0..8 {
            let mut values = rng.elements::<F>();
            if round == 0 {
                // Zero must survive, since it appears wherever a lane is unused.
                values[0] = 0;
                values[LANES - 1] = 0;
            }

            let mut a = [0; MAX_BITS];
            pack::<F>(&mut a, &values);
            let mut inverted = [0; MAX_BITS];
            tables.inv(&mut inverted, &a);

            let mut got = vec![0u16; LANES];
            unpack::<F>(&mut got, &inverted);
            for (lane, &value) in got.iter().enumerate() {
                assert_eq!(value, F::inv(values[lane]), "inv lane {lane}");
            }
        }
    }

    /// The identities that make the representation usable at all.
    fn algebraic_identities_hold<F: Field>(seed: u64) {
        let mut rng = Rng(seed);
        let tables = Tables::<F>::new();

        let values = rng.elements::<F>();
        let mut a = [0; MAX_BITS];
        pack::<F>(&mut a, &values);

        // Squaring is multiplication by self.
        let mut squared = [0; MAX_BITS];
        let mut multiplied = [0; MAX_BITS];
        tables.sq(&mut squared, &a);
        tables.mul(&mut multiplied, &a, &a);
        assert_eq!(squared, multiplied);

        // Multiplying by one leaves a group alone, and by zero clears it.
        let mut scaled = [0; MAX_BITS];
        tables.mul_by_scalar(&mut scaled, &a, 1);
        assert_eq!(scaled, a);
        tables.mul_by_scalar(&mut scaled, &a, 0);
        assert_eq!(scaled, [0; MAX_BITS]);
    }

    /// The 64-lane multiply is only reached by the degree-64 Berlekamp--Massey, which the
    /// standardized parameter sets only ever instantiate over `GF(2^12)`, since `t = 64` occurs
    /// only at `m = 12`. Its `GF(2^13)` path is therefore unreachable in practice and would
    /// otherwise never be run at all, so check it here against the scalar field directly.
    ///
    /// The 64-lane kernel itself only exists when decapsulation is compiled.
    #[cfg(feature = "decapsulate")]
    fn mul64_matches_the_scalar_field<F: Field>(seed: u64) {
        const LANES_64: usize = u64::BITS as usize;

        let mut rng = Rng(seed);
        let tables = Tables::<F>::new();

        for _ in 0..16 {
            let left: Vec<u16> = (0..LANES_64)
                .map(|_| (rng.next() as u16) & F::MASK)
                .collect();
            let right: Vec<u16> = (0..LANES_64)
                .map(|_| (rng.next() as u16) & F::MASK)
                .collect();

            let mut a: Slice64 = [0; MAX_BITS];
            let mut b: Slice64 = [0; MAX_BITS];
            for (lane, (&x, &y)) in left.iter().zip(right.iter()).enumerate() {
                for plane in 0..F::BITS {
                    a[plane] |= u64::from((x >> plane) & 1) << lane;
                    b[plane] |= u64::from((y >> plane) & 1) << lane;
                }
            }

            let mut product: Slice64 = [0; MAX_BITS];
            tables.mul64(&mut product, &a, &b);

            for (lane, (&x, &y)) in left.iter().zip(right.iter()).enumerate() {
                let mut got = 0u16;
                for plane in (0..F::BITS).rev() {
                    got = (got << 1) | ((product[plane] >> lane) & 1) as u16;
                }
                assert_eq!(got, F::mul(x, y), "lane {lane}");
            }
        }
    }

    macro_rules! vec_tests {
        ($($feature:literal => $mod_name:ident, $field:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::field::*;

                    #[test]
                    fn packing_round_trip() {
                        packing_round_trips::<$field>($seed);
                    }

                    #[test]
                    fn agrees_with_scalar_arithmetic() {
                        matches_the_scalar_field::<$field>($seed ^ 1);
                    }

                    #[test]
                    fn inversion() {
                        inversion_matches_the_scalar_field::<$field>($seed ^ 3);
                    }

                    #[test]
                    fn identities() {
                        algebraic_identities_hold::<$field>($seed ^ 2);
                    }

                    #[test]
                    #[cfg(feature = "decapsulate")]
                    fn sixty_four_lane_multiply() {
                        mul64_matches_the_scalar_field::<$field>($seed ^ 5);
                    }
                }
            )+
        };
    }

    vec_tests! {
        "mceliece348864" => gf12, Gf12, 0x1234_5678_9ABC_DEF0;
        "mceliece8192128" => gf13, Gf13, 0x0FED_CBA9_8765_4321;
    }
}
