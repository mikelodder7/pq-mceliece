/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Bit-sliced arithmetic over `F_q`.
//!
//! A slice holds 64 field elements as `m` words: word `i` carries bit `i` of every element, one
//! per lane. Arithmetic then becomes bitwise operations on whole words, so a single
//! multiplication instruction sequence serves all 64 lanes at once.
//!
//! The reference implementations hardcode the squaring and reduction formulas for one field.
//! Here both are derived from the scalar [`Field`] at start-up: squaring and multiplication by
//! a fixed element are `F_2`-linear maps, so each is fully described by where it sends the
//! basis `1, z, ..., z^(m-1)`, which the scalar implementation can simply be asked for.
//!
//! # Why this is not wired into the decoder yet
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
//! little to pay for converting in and out of this representation. The gain comes from the
//! register width, and it scales close to linearly. Wiring this in is therefore worth doing
//! together with wide lanes and with the whole decoder kept in bit-sliced form, so that no
//! conversion happens on the hot path, rather than on its own.

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

/// Tables describing a field's `F_2`-linear maps, built once and reused.
pub(crate) struct Tables<F: Field> {
    /// `reduce[d]` is `z^(m + d)` reduced into the field, for folding a product's high half.
    reduce: [u16; MAX_BITS],
    /// `square[j]` is `z^(2j)` reduced into the field.
    square: [u16; MAX_BITS],
    field: core::marker::PhantomData<F>,
}

impl<F: Field> Tables<F> {
    /// Derive the tables by asking the scalar implementation where it sends each basis element.
    pub(crate) fn new() -> Self {
        let mut reduce = [0u16; MAX_BITS];
        let mut square = [0u16; MAX_BITS];

        // `z^(m + d)` is `z^d` times `z^m`, and `z^m` is what a one-bit overflow reduces to.
        let overflow = F::reduce(1u32 << F::BITS);
        for (d, slot) in reduce.iter_mut().take(F::BITS).enumerate() {
            *slot = F::mul(overflow, 1u16 << d);
        }

        for (j, slot) in square.iter_mut().take(F::BITS).enumerate() {
            *slot = F::sq(1u16 << j);
        }

        Self {
            reduce,
            square,
            field: core::marker::PhantomData,
        }
    }

    /// Multiply two bit-sliced groups lane by lane.
    ///
    /// The schoolbook convolution gives a product of degree up to `2m - 2`; the top half folds
    /// back in through the reduction table. Every operation is a word-wide `AND` or `XOR`, so
    /// all 64 lanes advance together.
    pub(crate) fn mul(&self, out: &mut Slice, a: &Slice, b: &Slice) {
        let bits = F::BITS;
        let mut product = [0; 2 * MAX_BITS - 1];

        for (i, &ai) in a.iter().take(bits).enumerate() {
            for (j, &bj) in b.iter().take(bits).enumerate() {
                product[i + j] ^= ai & bj;
            }
        }

        // Fold from the top so that each term is complete before it is folded.
        for d in (0..bits - 1).rev() {
            let high = product[bits + d];
            let image = self.reduce[d];
            for (i, slot) in product.iter_mut().take(bits).enumerate() {
                if (image >> i) & 1 == 1 {
                    *slot ^= high;
                }
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
        let bits = F::BITS;
        let mut result = [0; MAX_BITS];

        for (j, &word) in a.iter().take(bits).enumerate() {
            let image = self.square[j];
            for (i, slot) in result.iter_mut().take(bits).enumerate() {
                if (image >> i) & 1 == 1 {
                    *slot ^= word;
                }
            }
        }

        *out = result;
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
                }
            )+
        };
    }

    vec_tests! {
        "mceliece348864" => gf12, Gf12, 0x1234_5678_9ABC_DEF0;
        "mceliece8192128" => gf13, Gf13, 0x0FED_CBA9_8765_4321;
    }
}
