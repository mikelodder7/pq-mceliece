/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Arithmetic in the binary fields `GF(2^12)` and `GF(2^13)`.
//!
//! Classic McEliece represents `F_q` as `F_2[z]/f(z)`. Two field polynomials are used by the
//! standardized parameter sets:
//!
//! | `m` | `f(z)` | parameter sets |
//! | --- | ------ | -------------- |
//! | 12 | `z^12 + z^3 + 1` | `mceliece348864*` |
//! | 13 | `z^13 + z^4 + z^3 + z + 1` | every other parameter set |
//!
//! Every routine in this module is constant time with respect to its inputs: there are no
//! branches, no lookup tables, and no variable shifts driven by secret data.

/// A field element. Values are always reduced, so the top `16 - m` bits are zero.
pub type Elem = u16;

/// Arithmetic in `F_q = F_2[z]/f(z)`.
///
/// The two implementors, [`Gf12`] and [`Gf13`], are zero-sized marker types selected by a
/// parameter set through [`Params::Field`](crate::hazmat::Params::Field).
pub trait Field:
    Copy + Clone + core::fmt::Debug + Default + Eq + Ord + Send + Sync + 'static
{
    /// The extension degree `m`, so that `q = 2^m`.
    const BITS: usize;

    /// A mask covering the `m` low bits.
    const MASK: Elem = (1 << Self::BITS) - 1;

    /// Reduce a carry-less product modulo `f(z)`.
    ///
    /// The input holds a polynomial of degree at most `2m - 2`.
    fn reduce(product: u32) -> Elem;

    /// Multiply two field elements.
    fn mul(a: Elem, b: Elem) -> Elem;

    /// Square a field element.
    ///
    /// Squaring only appears in two places: the inversion chain over `GF(2^12)`, and the
    /// `g(alpha)^2` of the Goppa syndrome. `GF(2^13)` inverts through fused helpers instead,
    /// so a key-generation-only build over that field never needs this.
    #[cfg(any(
        feature = "decapsulate",
        feature = "mceliece348864",
        feature = "mceliece348864f"
    ))]
    fn sq(a: Elem) -> Elem;

    /// Invert a field element. `inv(0)` is defined to be `0`.
    fn inv(a: Elem) -> Elem;

    /// Compute `num / den`. `frac(0, num)` is defined to be `0`.
    ///
    /// Only Berlekamp-Massey divides, so this is only present with the `decapsulate` feature.
    #[cfg(feature = "decapsulate")]
    fn frac(den: Elem, num: Elem) -> Elem {
        Self::mul(Self::inv(den), num)
    }

    /// Reverse the low `m` bits of `a`.
    ///
    /// Field orderings are defined in terms of `sum_j pi(i)_j z^(m-1-j)`, i.e. the bits of the
    /// permutation image read in the opposite order, so this conversion appears wherever a
    /// permutation is turned into a support element. Decoding folds it into the evaluation
    /// basis of the transposed transform instead, so only key generation calls it directly.
    #[cfg(any(feature = "keygen", test))]
    fn bitrev(a: Elem) -> Elem {
        let mut a = a;
        a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8);
        a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4);
        a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2);
        a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1);
        a >> (16 - Self::BITS)
    }
}

/// Return `MASK` when `a == 0` and `0` otherwise.
///
/// The `MASK` return value (rather than `1`) lets callers use the result directly as an
/// `AND` mask for constant-time selection.
///
/// Decoding tests for zero by or-reducing bit-planes instead, so only key generation, which
/// searches for a pivot, needs this.
#[cfg(feature = "keygen")]
#[inline]
pub const fn is_zero_mask(a: Elem) -> Elem {
    // `a - 1` borrows into the high bits exactly when `a == 0`.
    let t = (a as u32).wrapping_sub(1);
    (t >> 19) as u16
}

/// Add two field elements. Addition in characteristic two is exclusive-or.
#[inline]
pub const fn add(a: Elem, b: Elem) -> Elem {
    a ^ b
}

/// Carry-less multiplication of two values of at most 13 bits.
///
/// Returns the 25-bit polynomial product. This is the only part of field multiplication that
/// benefits from hardware acceleration, so it is the only part with architecture-specific
/// implementations.
#[inline]
fn clmul(a: Elem, b: Elem) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: `pclmulqdq` and `sse2` are statically enabled for this build, so both
        // intrinsics are available. Neither reads memory; both operate on register values
        // materialized here.
        unsafe {
            use core::arch::x86_64::{_mm_clmulepi64_si128, _mm_cvtsi32_si128, _mm_cvtsi128_si32};
            let x = _mm_cvtsi32_si128(a as i32);
            let y = _mm_cvtsi32_si128(b as i32);
            _mm_cvtsi128_si32(_mm_clmulepi64_si128(x, y, 0x00)) as u32
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the `aes` target feature (which is what gates PMULL on AArch64) is
        // statically enabled for this build. `vmull_p64` is a pure register operation.
        unsafe {
            use core::arch::aarch64::vmull_p64;
            vmull_p64(a as u64, b as u64) as u32
        }
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    {
        // Portable constant-time fallback: a 13-term convolution. Multiplying by an
        // already-masked single-bit value is a shift that the optimizer keeps branch free.
        let x = a as u64;
        let y = b as u64;
        let mut acc = x * (y & 1);
        for i in 1..13 {
            acc ^= x * (y & (1 << i));
        }
        acc as u32
    }
}

/// Emit items only when a parameter set over `GF(2^12)` is enabled.
///
/// Only the `mceliece348864` pair uses `m = 12`, so a build without them would otherwise carry
/// [`Gf12`] as dead code.
macro_rules! cfg_gf12 {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "mceliece348864", feature = "mceliece348864f"))]
            $item
        )*
    };
}

cfg_gf12! {
/// `GF(2^12)` with `f(z) = z^12 + z^3 + 1`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gf12;

impl Field for Gf12 {
    const BITS: usize = 12;

    #[inline]
    fn reduce(product: u32) -> Elem {
        // z^12 = z^3 + 1, so a set bit at position k >= 12 folds down to k - 9 and k - 12.
        // Bits 14..=22 fold into 5..=13, then bits 12..=13 fold into 0..=4; two passes suffice.
        let mut x = product;
        let mut t = x & 0x007F_C000;
        x ^= t >> 9;
        x ^= t >> 12;

        t = x & 0x0000_3000;
        x ^= t >> 9;
        x ^= t >> 12;

        (x as u16) & Self::MASK
    }

    #[inline]
    fn mul(a: Elem, b: Elem) -> Elem {
        Self::reduce(clmul(a & Self::MASK, b & Self::MASK))
    }

    #[inline]
    fn sq(a: Elem) -> Elem {
        // Spread the 12 input bits into even positions, which is squaring in characteristic two.
        let mut x = (a & Self::MASK) as u32;
        x = (x | (x << 8)) & 0x00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333;
        x = (x | (x << 1)) & 0x5555_5555;
        Self::reduce(x)
    }

    #[inline]
    fn inv(a: Elem) -> Elem {
        // Itoh-Tsujii: a^-1 = a^(2^12 - 2), reached with an addition chain on the exponent
        // written in binary as 111111111110.
        let mut out = Self::sq(a);
        let tmp_11 = Self::mul(out, a);

        out = Self::sq(Self::sq(tmp_11));
        let tmp_1111 = Self::mul(out, tmp_11);

        out = Self::sq(Self::sq(Self::sq(Self::sq(tmp_1111))));
        out = Self::mul(out, tmp_1111);

        out = Self::sq(Self::sq(out));
        out = Self::mul(out, tmp_11);

        out = Self::sq(out);
        out = Self::mul(out, a);

        Self::sq(out)
    }
}

}

/// Emit items only when a parameter set over `GF(2^13)` is enabled.
///
/// Every standardized parameter set except the `mceliece348864` pair uses `m = 13`, so a build
/// that enables only those two would otherwise carry [`Gf13`] as dead code.
macro_rules! cfg_gf13 {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "mceliece460896",
                feature = "mceliece460896f",
                feature = "mceliece460896pc",
                feature = "mceliece460896pcf",
                feature = "mceliece6688128",
                feature = "mceliece6688128f",
                feature = "mceliece6688128pc",
                feature = "mceliece6688128pcf",
                feature = "mceliece6960119",
                feature = "mceliece6960119f",
                feature = "mceliece6960119pc",
                feature = "mceliece6960119pcf",
                feature = "mceliece8192128",
                feature = "mceliece8192128f",
                feature = "mceliece8192128pc",
                feature = "mceliece8192128pcf"
            ))]
            $item
        )*
    };
}

cfg_gf13! {
/// `GF(2^13)` with `f(z) = z^13 + z^4 + z^3 + z + 1`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gf13;

impl Gf13 {
    /// One fold of the reduction: bits selected by `mask` are folded down by 9, 10, 12 and 13.
    #[inline]
    const fn fold(x: u32, mask: u32) -> u32 {
        let t = x & mask;
        x ^ (t >> 9) ^ (t >> 10) ^ (t >> 12) ^ (t >> 13)
    }
}

impl Field for Gf13 {
    const BITS: usize = 13;

    #[inline]
    fn reduce(product: u32) -> Elem {
        // z^13 = z^4 + z^3 + z + 1, so a set bit at position k >= 13 folds down to
        // k - 9, k - 10, k - 12 and k - 13.
        let x = Self::fold(product, 0x01FF_0000);
        let x = Self::fold(x, 0x0000_E000);
        (x as u16) & Self::MASK
    }

    #[inline]
    fn mul(a: Elem, b: Elem) -> Elem {
        Self::reduce(clmul(a & Self::MASK, b & Self::MASK))
    }

    #[cfg(any(
    feature = "decapsulate",
    feature = "mceliece348864",
    feature = "mceliece348864f"
))]
    #[inline]
    fn sq(a: Elem) -> Elem {
        let mut x = (a & Self::MASK) as u32;
        x = (x | (x << 8)) & 0x00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333;
        x = (x | (x << 1)) & 0x5555_5555;
        Self::reduce(x)
    }

    #[inline]
    fn inv(a: Elem) -> Elem {
        Self::divide(a, 1)
    }

    #[cfg(feature = "decapsulate")]
    #[inline]
    fn frac(den: Elem, num: Elem) -> Elem {
        Self::divide(den, num)
    }
}

impl Gf13 {
    /// Compute `num / den`, which is also how inversion is reached.
    ///
    /// `a^-1 = a^(2^13 - 2)`, exponent `1111111111110` in binary, built from repeated
    /// square-and-multiply steps that share the `^1111` intermediate. Folding the numerator
    /// into the final step makes division cost no more than inversion.
    #[inline]
    fn divide(den: Elem, num: Elem) -> Elem {
        let tmp_11 = Self::sqmul(den, den);
        let tmp_1111 = Self::sq2mul(tmp_11, tmp_11);

        let mut out = Self::sq2(tmp_1111);
        out = Self::sq2mul(out, tmp_1111);
        out = Self::sq2(out);
        out = Self::sq2mul(out, tmp_1111);

        Self::sqmul(out, num)
    }

    /// Compute `a^4`, fusing two squarings into one reduction chain.
    #[inline]
    fn sq2(a: Elem) -> Elem {
        const SPREAD: [u64; 4] = [
            0x1111_1111_1111_1111,
            0x0303_0303_0303_0303,
            0x000F_000F_000F_000F,
            0x0000_00FF_0000_00FF,
        ];
        const FOLD: [u64; 4] = [
            0x0001_FF00_0000_0000,
            0x0000_00FF_8000_0000,
            0x0000_0000_7FC0_0000,
            0x0000_0000_003F_E000,
        ];

        let mut x = (a & Self::MASK) as u64;
        x = (x | (x << 24)) & SPREAD[3];
        x = (x | (x << 12)) & SPREAD[2];
        x = (x | (x << 6)) & SPREAD[1];
        x = (x | (x << 3)) & SPREAD[0];

        for mask in FOLD {
            let t = x & mask;
            x ^= (t >> 9) ^ (t >> 10) ^ (t >> 12) ^ (t >> 13);
        }

        (x as u16) & Self::MASK
    }

    /// Compute `a^2 * m` in a single reduction chain.
    #[inline]
    fn sqmul(a: Elem, m: Elem) -> Elem {
        const FOLD: [u64; 3] = [
            0x0000_001F_F000_0000,
            0x0000_0000_0FF8_0000,
            0x0000_0000_0007_E000,
        ];

        let mut t0 = (a & Self::MASK) as u64;
        let t1 = (m & Self::MASK) as u64;

        let mut x = (t1 << 6) * (t0 & (1 << 6));
        t0 ^= t0 << 7;

        x ^= t1 * (t0 & 0x0_4001);
        x ^= (t1 * (t0 & 0x0_8002)) << 1;
        x ^= (t1 * (t0 & 0x1_0004)) << 2;
        x ^= (t1 * (t0 & 0x2_0008)) << 3;
        x ^= (t1 * (t0 & 0x4_0010)) << 4;
        x ^= (t1 * (t0 & 0x8_0020)) << 5;

        for mask in FOLD {
            let t = x & mask;
            x ^= (t >> 9) ^ (t >> 10) ^ (t >> 12) ^ (t >> 13);
        }

        (x as u16) & Self::MASK
    }

    /// Compute `a^4 * m` in a single reduction chain.
    #[inline]
    fn sq2mul(a: Elem, m: Elem) -> Elem {
        const FOLD: [u64; 6] = [
            0x1FF0_0000_0000_0000,
            0x000F_F800_0000_0000,
            0x0000_07FC_0000_0000,
            0x0000_0003_FE00_0000,
            0x0000_0000_01FE_0000,
            0x0000_0000_0001_E000,
        ];

        let mut t0 = (a & Self::MASK) as u64;
        let t1 = (m & Self::MASK) as u64;

        let mut x = (t1 << 18) * (t0 & (1 << 6));
        t0 ^= t0 << 21;

        x ^= t1 * (t0 & 0x0_1000_0001);
        x ^= (t1 * (t0 & 0x0_2000_0002)) << 3;
        x ^= (t1 * (t0 & 0x0_4000_0004)) << 6;
        x ^= (t1 * (t0 & 0x0_8000_0008)) << 9;
        x ^= (t1 * (t0 & 0x1_0000_0010)) << 12;
        x ^= (t1 * (t0 & 0x2_0000_0020)) << 15;

        for mask in FOLD {
            let t = x & mask;
            x ^= (t >> 9) ^ (t >> 10) ^ (t >> 12) ^ (t >> 13);
        }

        (x as u16) & Self::MASK
    }
}

}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Schoolbook multiplication reduced by long division, used as an independent oracle for
    /// whichever field the caller names.
    fn reference_mul(a: u16, b: u16, m: usize, poly: u32) -> u16 {
        let mut product = 0u32;
        for i in 0..m {
            if (b >> i) & 1 == 1 {
                product ^= (a as u32) << i;
            }
        }
        for i in (m..2 * m).rev() {
            if (product >> i) & 1 == 1 {
                product ^= poly << (i - m);
            }
        }
        (product & ((1 << m) - 1)) as u16
    }

    /// Check a whole field against the oracle: multiplication, squaring, inversion, division
    /// and the bit reversal used by field orderings.
    fn field_agrees_with_long_division<F: Field>(poly: u32) {
        let count = 1u16 << F::BITS;
        let probes: [u16; 6] = [0, 1, 2, 3, 5, F::MASK];

        for a in 0..count {
            for &b in probes.iter() {
                assert_eq!(F::mul(a, b), reference_mul(a, b, F::BITS, poly));
                assert_eq!(F::mul(b, a), reference_mul(b, a, F::BITS, poly));
            }
            #[cfg(any(
                feature = "decapsulate",
                feature = "mceliece348864",
                feature = "mceliece348864f"
            ))]
            assert_eq!(F::sq(a), reference_mul(a, a, F::BITS, poly));
            assert_eq!(F::bitrev(F::bitrev(a)), a);
        }

        assert_eq!(F::inv(0), 0);
        for a in 1..count {
            assert_eq!(F::mul(a, F::inv(a)), 1, "inverse of {a}");
        }

        #[cfg(feature = "decapsulate")]
        {
            assert_eq!(F::frac(0, 7), 0);
            for a in 1..count {
                for &num in probes.iter() {
                    assert_eq!(F::mul(F::frac(a, num), a), num, "{num} / {a}");
                }
            }
        }
    }

    #[test]
    fn is_zero_mask_selects_all_or_nothing() {
        assert_eq!(is_zero_mask(0), 0x1FFF);
        for a in 1u16..=u16::MAX {
            assert_eq!(is_zero_mask(a), 0);
        }
    }

    #[test]
    fn addition_is_exclusive_or() {
        assert_eq!(add(0x0F0F, 0xF0F0), 0xFFFF);
        assert_eq!(add(0x1234, 0x1234), 0);
    }

    cfg_gf12! {
        #[test]
        fn gf12_matches_long_division() {
            const POLY12: u32 = (1 << 12) | (1 << 3) | 1;
            field_agrees_with_long_division::<Gf12>(POLY12);
        }

        #[test]
        fn gf12_bitrev_reverses_twelve_bits() {
            assert_eq!(Gf12::bitrev(0b1011_0111_0111_1011), 0b0000_1101_1110_1110);
            assert_eq!(Gf12::bitrev(0b0110_1010_0101_1011), 0b0000_1101_1010_0101);
        }
    }

    cfg_gf13! {
        #[test]
        fn gf13_matches_long_division() {
            const POLY13: u32 = (1 << 13) | (1 << 4) | (1 << 3) | (1 << 1) | 1;
            field_agrees_with_long_division::<Gf13>(POLY13);
        }

        #[test]
        fn gf13_bitrev_reverses_thirteen_bits() {
            assert_eq!(Gf13::bitrev(0b1011_0111_0111_1011), 0b0001_1011_1101_1101);
            assert_eq!(Gf13::bitrev(0b0110_1010_0101_1011), 0b0001_1011_0100_1010);
        }

        /// The fused helpers exist only as a faster route to the same values.
        #[cfg(feature = "decapsulate")]
        #[test]
        fn gf13_fused_products_agree_with_the_simple_forms() {
            for a in 0u16..8192 {
                assert_eq!(Gf13::sq2(a), Gf13::sq(Gf13::sq(a)));
                for m in [0u16, 1, 3, 4095, 8191] {
                    assert_eq!(Gf13::sqmul(a, m), Gf13::mul(Gf13::sq(a), m));
                    assert_eq!(Gf13::sq2mul(a, m), Gf13::mul(Gf13::sq2(a), m));
                }
            }
        }
    }
}
