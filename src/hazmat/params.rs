/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Compile-time parameter sets.
//!
//! Every Classic McEliece parameter set is a zero-sized type implementing [`Params`]. Unlike
//! implementations that bake one parameter set into the whole crate through cargo features,
//! all enabled parameter sets here coexist and can be used side by side.

#[cfg(any(feature = "keygen", feature = "decapsulate"))]
use super::field::Field;
#[cfg(all(
    any(feature = "keygen", feature = "decapsulate"),
    any(feature = "mceliece348864", feature = "mceliece348864f")
))]
use super::field::Gf12;
#[cfg(all(
    any(feature = "keygen", feature = "decapsulate"),
    any(
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
    )
))]
use super::field::Gf13;

/// Which standards a parameter set appears in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Standards {
    /// Present in the NIST round-4 Classic McEliece submission of 2022-10-23.
    pub nist: bool,
    /// Present in the ISO Classic McEliece standard published in June 2026.
    pub iso: bool,
}

/// A single reduction tap of the field polynomial `F(y)`.
///
/// Only present when an operation that does field arithmetic is enabled.
///
/// `F(y) = y^t + sum(coeff_i * y^(exponent_i))`, so reducing a term `y^(t+d)` folds the
/// coefficient into position `d + exponent`, scaled by `coeff`.
#[cfg(feature = "keygen")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolyTap {
    /// The exponent of `y` this tap contributes to.
    pub exponent: usize,
    /// The `F_q` coefficient of the tap. All taps are `1` except the constant term of the
    /// `mceliece348864` polynomial `y^64 + y^3 + y + z`, whose coefficient is `z`.
    pub coeff: u16,
}

#[cfg(feature = "keygen")]
impl PolyTap {
    /// A tap with coefficient one.
    const fn one(exponent: usize) -> Self {
        Self { exponent, coeff: 1 }
    }
}

/// The constants defining a Classic McEliece parameter set.
///
/// Implementors supply the primitive parameters; every derived size has a default expression
/// and should not be overridden.
pub trait Params:
    Copy + Clone + core::fmt::Debug + Default + Eq + Ord + core::hash::Hash + Send + Sync + 'static
{
    /// The field `F_q = F_2[z]/f(z)` used by this parameter set.
    ///
    /// Only present when an operation that does field arithmetic is enabled. Encapsulation
    /// needs the field's size but never its arithmetic, so an encapsulate-only build drops
    /// the whole arithmetic layer.
    #[cfg(any(feature = "keygen", feature = "decapsulate"))]
    type Field: Field;

    /// The extension degree `m`, so that `q = 2^m`.
    const M: usize;

    /// The lowercase parameter set name, e.g. `mceliece6960119f`.
    const NAME: &'static str;

    /// The code length `n`.
    const N: usize;

    /// The guaranteed error-correction capacity `t`, and the degree of the Goppa polynomial.
    const T: usize;

    /// The semi-systematic parameter `mu`. Zero for parameter sets without an `f` suffix.
    const MU: usize;

    /// The semi-systematic parameter `nu`. Zero for parameter sets without an `f` suffix.
    const NU: usize;

    /// Whether this is a plaintext-confirmation (`pc`) parameter set.
    const PC: bool;

    /// The nonzero low-order taps of the field polynomial `F(y)` of degree `t`.
    ///
    /// Only `Irreducible` works in `F_q[y]/F(y)`, so this is only present with `keygen`.
    #[cfg(feature = "keygen")]
    const POLY_TAPS: &'static [PolyTap];

    /// The NIST security category claimed for this parameter set.
    const CLAIMED_NIST_LEVEL: usize;

    /// The standards this parameter set appears in.
    const STANDARDS: Standards;

    /// The field size `q = 2^m`.
    const Q: usize = 1 << Self::M;

    /// A mask covering the `m` low bits of a field element.
    ///
    /// Only `FixedWeight` reduces raw randomness into the field this way.
    #[cfg(feature = "encapsulate")]
    const GFMASK: u16 = ((1u32 << Self::M) - 1) as u16;

    /// The code dimension `k = n - mt`.
    const K: usize = Self::N - Self::M * Self::T;

    /// The number of rows of the public key matrix `T`, equal to `mt`.
    const PK_NROWS: usize = Self::M * Self::T;

    /// The number of columns of the public key matrix `T`, equal to `k`.
    const PK_NCOLS: usize = Self::K;

    /// The byte length of one row of the public key matrix.
    const PK_ROW_BYTES: usize = Self::PK_NCOLS.div_ceil(8);

    /// The byte length of the syndrome `C0`.
    const SYND_BYTES: usize = Self::PK_NROWS.div_ceil(8);

    /// The byte length of an `n`-bit vector such as `e` or `s`.
    const N_BYTES: usize = Self::N.div_ceil(8);

    /// The byte length of the stored Goppa polynomial.
    const IRR_BYTES: usize = Self::T * 2;

    /// The byte length of the Beneš control bits encoding the field ordering.
    const COND_BYTES: usize = (1 << (Self::M - 4)) * (2 * Self::M - 1);

    /// The byte length of the stored column selections `c`.
    const COLUMN_SELECTION_BYTES: usize = 8;

    /// The byte offset of the Goppa polynomial within a private key.
    const IRR_OFFSET: usize = Self::SEED_LENGTH + Self::COLUMN_SELECTION_BYTES;

    /// The byte offset of the control bits within a private key.
    const COND_OFFSET: usize = Self::IRR_OFFSET + Self::IRR_BYTES;

    /// The byte offset of the rejection string `s` within a private key.
    const S_OFFSET: usize = Self::COND_OFFSET + Self::COND_BYTES;

    /// The byte length of an encapsulation (public) key.
    const PUBLIC_KEY_LENGTH: usize = Self::PK_NROWS * Self::PK_ROW_BYTES;

    /// The byte length of a decapsulation (private) key.
    const SECRET_KEY_LENGTH: usize = Self::S_OFFSET + Self::N_BYTES;

    /// The byte length of a ciphertext.
    const CIPHERTEXT_LENGTH: usize = Self::SYND_BYTES + if Self::PC { 32 } else { 0 };

    /// The byte length of a shared secret, `ceil(l/8)` with `l = 256`.
    const SHARED_SECRET_LENGTH: usize = 32;

    /// The byte length of the key generation seed `delta`, `ceil(l/8)` with `l = 256`.
    const SEED_LENGTH: usize = 32;

    /// The number of candidate indices drawn per `FixedWeight` attempt.
    ///
    /// The specification defines `tau` as `t` when `n = q`, `2t` when `q/2 <= n < q`, and so on.
    #[cfg(feature = "encapsulate")]
    const TAU: usize = if Self::N == Self::Q {
        Self::T
    } else {
        2 * Self::T
    };

    /// Whether the parameter set uses semi-systematic matrix reduction.
    ///
    /// Only key generation consults this, so it is only present with the `keygen` feature.
    #[cfg(feature = "keygen")]
    const SEMI_SYSTEMATIC: bool = Self::NU > Self::MU;

    /// Whether ciphertexts carry padding bits that must be checked on input.
    ///
    /// Only decapsulation consults this, so it is only present with the `decapsulate` feature.
    #[cfg(feature = "decapsulate")]
    const CIPHERTEXT_HAS_PADDING: bool = Self::PK_NROWS % 8 != 0;

    /// Whether public keys carry padding bits that must be checked on input.
    ///
    /// Only encapsulation validates an incoming public key.
    #[cfg(feature = "encapsulate")]
    const PUBLIC_KEY_HAS_PADDING: bool = Self::PK_NCOLS % 8 != 0;
}

/// `F(y) = y^64 + y^3 + y + z` for `mceliece348864`.
#[cfg(all(
    feature = "keygen",
    any(feature = "mceliece348864", feature = "mceliece348864f")
))]
const TAPS_348864: &[PolyTap] = &[
    PolyTap::one(3),
    PolyTap::one(1),
    PolyTap {
        exponent: 0,
        coeff: 2,
    },
];

/// `F(y) = y^96 + y^10 + y^9 + y^6 + 1` for `mceliece460896`.
#[cfg(all(
    feature = "keygen",
    any(
        feature = "mceliece460896",
        feature = "mceliece460896f",
        feature = "mceliece460896pc",
        feature = "mceliece460896pcf"
    )
))]
const TAPS_460896: &[PolyTap] = &[
    PolyTap::one(10),
    PolyTap::one(9),
    PolyTap::one(6),
    PolyTap::one(0),
];

/// `F(y) = y^119 + y^8 + 1` for `mceliece6960119`.
#[cfg(all(
    feature = "keygen",
    any(
        feature = "mceliece6960119",
        feature = "mceliece6960119f",
        feature = "mceliece6960119pc",
        feature = "mceliece6960119pcf"
    )
))]
const TAPS_6960119: &[PolyTap] = &[PolyTap::one(8), PolyTap::one(0)];

/// `F(y) = y^128 + y^7 + y^2 + y + 1` for `mceliece6688128` and `mceliece8192128`.
#[cfg(all(
    feature = "keygen",
    any(
        feature = "mceliece6688128",
        feature = "mceliece6688128f",
        feature = "mceliece6688128pc",
        feature = "mceliece6688128pcf",
        feature = "mceliece8192128",
        feature = "mceliece8192128f",
        feature = "mceliece8192128pc",
        feature = "mceliece8192128pcf"
    )
))]
const TAPS_128: &[PolyTap] = &[
    PolyTap::one(7),
    PolyTap::one(2),
    PolyTap::one(1),
    PolyTap::one(0),
];

macro_rules! define_params {
    (
        $(
            $(#[$meta:meta])*
            $feature:literal => $name:ident {
                field: $field:ty,
                m: $m:expr,
                n: $n:expr,
                t: $t:expr,
                taps: $taps:expr,
                level: $level:expr,
                semi_systematic: $semi:expr,
                pc: $pc:expr,
                nist: $nist:expr,
                iso: $iso:expr,
            }
        )+
    ) => {
        $(
            $(#[$meta])*
            #[cfg(feature = $feature)]
            #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub struct $name;

            #[cfg(feature = $feature)]
            impl Params for $name {
                #[cfg(any(feature = "keygen", feature = "decapsulate"))]
                type Field = $field;
                const M: usize = $m;
                const NAME: &'static str = $feature;
                const N: usize = $n;
                const T: usize = $t;
                const MU: usize = if $semi { 32 } else { 0 };
                const NU: usize = if $semi { 64 } else { 0 };
                const PC: bool = $pc;
                #[cfg(feature = "keygen")]
                const POLY_TAPS: &'static [PolyTap] = $taps;
                const CLAIMED_NIST_LEVEL: usize = $level;
                const STANDARDS: Standards = Standards { nist: $nist, iso: $iso };
            }
        )+
    };
}

define_params! {
    /// `mceliece348864`: `m = 12`, `n = 3488`, `t = 64`. NIST category 1. Not in the ISO standard.
    "mceliece348864" => McEliece348864 {
        field: Gf12, m: 12, n: 3488, t: 64, taps: TAPS_348864, level: 1,
        semi_systematic: false, pc: false, nist: true, iso: false,
    }
    /// `mceliece348864f`: `mceliece348864` with semi-systematic key generation.
    "mceliece348864f" => McEliece348864f {
        field: Gf12, m: 12, n: 3488, t: 64, taps: TAPS_348864, level: 1,
        semi_systematic: true, pc: false, nist: true, iso: false,
    }
    /// `mceliece460896`: `m = 13`, `n = 4608`, `t = 96`. NIST category 3.
    "mceliece460896" => McEliece460896 {
        field: Gf13, m: 13, n: 4608, t: 96, taps: TAPS_460896, level: 3,
        semi_systematic: false, pc: false, nist: true, iso: true,
    }
    /// `mceliece460896f`: `mceliece460896` with semi-systematic key generation.
    "mceliece460896f" => McEliece460896f {
        field: Gf13, m: 13, n: 4608, t: 96, taps: TAPS_460896, level: 3,
        semi_systematic: true, pc: false, nist: true, iso: true,
    }
    /// `mceliece460896pc`: `mceliece460896` with plaintext confirmation. ISO only.
    "mceliece460896pc" => McEliece460896pc {
        field: Gf13, m: 13, n: 4608, t: 96, taps: TAPS_460896, level: 3,
        semi_systematic: false, pc: true, nist: false, iso: true,
    }
    /// `mceliece460896pcf`: plaintext confirmation and semi-systematic key generation. ISO only.
    "mceliece460896pcf" => McEliece460896pcf {
        field: Gf13, m: 13, n: 4608, t: 96, taps: TAPS_460896, level: 3,
        semi_systematic: true, pc: true, nist: false, iso: true,
    }
    /// `mceliece6688128`: `m = 13`, `n = 6688`, `t = 128`. NIST category 5.
    "mceliece6688128" => McEliece6688128 {
        field: Gf13, m: 13, n: 6688, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: false, pc: false, nist: true, iso: true,
    }
    /// `mceliece6688128f`: `mceliece6688128` with semi-systematic key generation.
    "mceliece6688128f" => McEliece6688128f {
        field: Gf13, m: 13, n: 6688, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: true, pc: false, nist: true, iso: true,
    }
    /// `mceliece6688128pc`: `mceliece6688128` with plaintext confirmation. ISO only.
    "mceliece6688128pc" => McEliece6688128pc {
        field: Gf13, m: 13, n: 6688, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: false, pc: true, nist: false, iso: true,
    }
    /// `mceliece6688128pcf`: plaintext confirmation and semi-systematic key generation. ISO only.
    "mceliece6688128pcf" => McEliece6688128pcf {
        field: Gf13, m: 13, n: 6688, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: true, pc: true, nist: false, iso: true,
    }
    /// `mceliece6960119`: `m = 13`, `n = 6960`, `t = 119`. NIST category 5.
    "mceliece6960119" => McEliece6960119 {
        field: Gf13, m: 13, n: 6960, t: 119, taps: TAPS_6960119, level: 5,
        semi_systematic: false, pc: false, nist: true, iso: true,
    }
    /// `mceliece6960119f`: `mceliece6960119` with semi-systematic key generation.
    "mceliece6960119f" => McEliece6960119f {
        field: Gf13, m: 13, n: 6960, t: 119, taps: TAPS_6960119, level: 5,
        semi_systematic: true, pc: false, nist: true, iso: true,
    }
    /// `mceliece6960119pc`: `mceliece6960119` with plaintext confirmation. ISO only.
    "mceliece6960119pc" => McEliece6960119pc {
        field: Gf13, m: 13, n: 6960, t: 119, taps: TAPS_6960119, level: 5,
        semi_systematic: false, pc: true, nist: false, iso: true,
    }
    /// `mceliece6960119pcf`: plaintext confirmation and semi-systematic key generation. ISO only.
    "mceliece6960119pcf" => McEliece6960119pcf {
        field: Gf13, m: 13, n: 6960, t: 119, taps: TAPS_6960119, level: 5,
        semi_systematic: true, pc: true, nist: false, iso: true,
    }
    /// `mceliece8192128`: `m = 13`, `n = 8192`, `t = 128`. NIST category 5.
    "mceliece8192128" => McEliece8192128 {
        field: Gf13, m: 13, n: 8192, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: false, pc: false, nist: true, iso: true,
    }
    /// `mceliece8192128f`: `mceliece8192128` with semi-systematic key generation.
    "mceliece8192128f" => McEliece8192128f {
        field: Gf13, m: 13, n: 8192, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: true, pc: false, nist: true, iso: true,
    }
    /// `mceliece8192128pc`: `mceliece8192128` with plaintext confirmation. ISO only.
    "mceliece8192128pc" => McEliece8192128pc {
        field: Gf13, m: 13, n: 8192, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: false, pc: true, nist: false, iso: true,
    }
    /// `mceliece8192128pcf`: plaintext confirmation and semi-systematic key generation. ISO only.
    "mceliece8192128pcf" => McEliece8192128pcf {
        field: Gf13, m: 13, n: 8192, t: 128, taps: TAPS_128, level: 5,
        semi_systematic: true, pc: true, nist: false, iso: true,
    }
}

// The size table checks constants owned by all three operations.
#[cfg(all(
    test,
    feature = "keygen",
    feature = "encapsulate",
    feature = "decapsulate"
))]
mod tests {
    use super::*;

    /// The values published alongside the reference implementation, checked against what this
    /// module derives from `m`, `n` and `t` alone.
    struct Expected {
        public_key: usize,
        secret_key: usize,
        ciphertext: usize,
        tau: usize,
        semi_systematic: bool,
        plaintext_confirmation: bool,
        has_padding: bool,
    }

    fn check<P: Params>(expected: Expected) {
        assert_eq!(P::PUBLIC_KEY_LENGTH, expected.public_key, "{}", P::NAME);
        assert_eq!(P::SECRET_KEY_LENGTH, expected.secret_key, "{}", P::NAME);
        assert_eq!(P::CIPHERTEXT_LENGTH, expected.ciphertext, "{}", P::NAME);
        assert_eq!(P::TAU, expected.tau, "{}", P::NAME);
        assert_eq!(P::SEMI_SYSTEMATIC, expected.semi_systematic, "{}", P::NAME);
        assert_eq!(P::PC, expected.plaintext_confirmation, "{}", P::NAME);
        assert_eq!(
            P::CIPHERTEXT_HAS_PADDING,
            expected.has_padding,
            "{}",
            P::NAME
        );
        assert_eq!(
            P::PUBLIC_KEY_HAS_PADDING,
            expected.has_padding,
            "{}",
            P::NAME
        );

        // Structural invariants the specification requires of any parameter set.
        assert_eq!(P::SHARED_SECRET_LENGTH, 32, "{}", P::NAME);
        assert_eq!(P::N % 8, 0, "{} n is a whole number of bytes", P::NAME);
        assert!(P::M * P::T < P::N, "{} needs mt < n", P::NAME);
        assert!(P::N <= P::Q, "{} needs n <= q", P::NAME);
        assert!(P::NU >= P::MU, "{} needs nu >= mu", P::NAME);
        assert!(P::NU <= P::K + P::MU, "{} needs nu <= k + mu", P::NAME);
        assert!(P::POLY_TAPS.len() >= 2, "{} F(y) needs taps", P::NAME);
        assert!(
            P::POLY_TAPS.iter().all(|tap| tap.exponent < P::T),
            "{} taps must reduce",
            P::NAME
        );
    }

    macro_rules! size_tests {
        ($(
            $feature:literal => $name:ident, $params:ty {
                public_key: $pk:expr,
                secret_key: $sk:expr,
                ciphertext: $ct:expr,
                tau: $tau:expr,
                semi_systematic: $semi:expr,
                plaintext_confirmation: $pc:expr,
                has_padding: $padding:expr,
            }
        )+) => {
            $(
                #[test]
                #[cfg(feature = $feature)]
                fn $name() {
                    check::<$params>(Expected {
                        public_key: $pk,
                        secret_key: $sk,
                        ciphertext: $ct,
                        tau: $tau,
                        semi_systematic: $semi,
                        plaintext_confirmation: $pc,
                        has_padding: $padding,
                    });
                }
            )+
        };
    }

    size_tests! {
        "mceliece348864" => sizes_348864, McEliece348864 {
            public_key: 261120, secret_key: 6492, ciphertext: 96, tau: 128,
            semi_systematic: false, plaintext_confirmation: false, has_padding: false,
        }
        "mceliece348864f" => sizes_348864f, McEliece348864f {
            public_key: 261120, secret_key: 6492, ciphertext: 96, tau: 128,
            semi_systematic: true, plaintext_confirmation: false, has_padding: false,
        }
        "mceliece460896" => sizes_460896, McEliece460896 {
            public_key: 524160, secret_key: 13608, ciphertext: 156, tau: 192,
            semi_systematic: false, plaintext_confirmation: false, has_padding: false,
        }
        "mceliece460896pcf" => sizes_460896pcf, McEliece460896pcf {
            public_key: 524160, secret_key: 13608, ciphertext: 188, tau: 192,
            semi_systematic: true, plaintext_confirmation: true, has_padding: false,
        }
        "mceliece6688128" => sizes_6688128, McEliece6688128 {
            public_key: 1044992, secret_key: 13932, ciphertext: 208, tau: 256,
            semi_systematic: false, plaintext_confirmation: false, has_padding: false,
        }
        "mceliece6688128pc" => sizes_6688128pc, McEliece6688128pc {
            public_key: 1044992, secret_key: 13932, ciphertext: 240, tau: 256,
            semi_systematic: false, plaintext_confirmation: true, has_padding: false,
        }
        "mceliece6960119" => sizes_6960119, McEliece6960119 {
            public_key: 1047319, secret_key: 13948, ciphertext: 194, tau: 238,
            semi_systematic: false, plaintext_confirmation: false, has_padding: true,
        }
        "mceliece6960119pcf" => sizes_6960119pcf, McEliece6960119pcf {
            public_key: 1047319, secret_key: 13948, ciphertext: 226, tau: 238,
            semi_systematic: true, plaintext_confirmation: true, has_padding: true,
        }
        // n == q, so a single draw of t candidates always lands in range.
        "mceliece8192128" => sizes_8192128, McEliece8192128 {
            public_key: 1357824, secret_key: 14120, ciphertext: 208, tau: 128,
            semi_systematic: false, plaintext_confirmation: false, has_padding: false,
        }
        "mceliece8192128pcf" => sizes_8192128pcf, McEliece8192128pcf {
            public_key: 1357824, secret_key: 14120, ciphertext: 240, tau: 128,
            semi_systematic: true, plaintext_confirmation: true, has_padding: false,
        }
    }
}
