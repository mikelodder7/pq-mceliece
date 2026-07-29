/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Polynomial arithmetic over `F_q`.
//!
//! Two different polynomial rings appear in Classic McEliece:
//!
//! * `F_q[y]/F(y)`, the representation of `F_(q^t)` used by the `Irreducible` algorithm to
//!   turn `t` random field elements into a monic irreducible degree-`t` polynomial;
//! * `F_q[x]`, where the Goppa polynomial `g` lives and is evaluated at support elements.

#[cfg(feature = "keygen")]
use super::field::is_zero_mask;
use super::field::{Field, add};
use super::params::Params;

/// Multiply `a` by `b` in `F_q[y]/F(y)`, where both operands have `t` coefficients.
///
/// `scratch` must hold at least `2t - 1` elements; it is used for the unreduced product.
#[cfg(feature = "keygen")]
pub(crate) fn mul_mod_f<P: Params>(out: &mut [u16], a: &[u16], b: &[u16], scratch: &mut [u16]) {
    let t = P::T;
    debug_assert_eq!(a.len(), t);
    debug_assert_eq!(b.len(), t);

    let prod = &mut scratch[..2 * t - 1];
    prod.fill(0);

    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            prod[i + j] = add(prod[i + j], P::Field::mul(ai, bj));
        }
    }

    // Fold `y^(t+d)` back down using `F(y) = y^t + sum(coeff * y^exponent)`. Working from the
    // highest degree downwards means each term is fully formed before it is folded.
    for i in (t..2 * t - 1).rev() {
        let high = prod[i];
        for tap in P::POLY_TAPS {
            let contribution = if tap.coeff == 1 {
                high
            } else {
                P::Field::mul(high, tap.coeff)
            };
            prod[i - t + tap.exponent] = add(prod[i - t + tap.exponent], contribution);
        }
    }

    out[..t].copy_from_slice(&prod[..t]);
}

/// Compute the minimal polynomial of `beta = sum(f_i y^i)` over `F_q`.
///
/// Returns `false` when the minimal polynomial has degree below `t`, which is the `⊥` outcome
/// of the specification's `Irreducible` algorithm and makes key generation retry with a fresh
/// seed. The `t` returned coefficients are `g_0, ..., g_(t-1)`; the leading coefficient is an
/// implicit `1`.
#[cfg(feature = "keygen")]
pub(crate) fn minimal_polynomial<P: Params>(out: &mut [u16], f: &[u16]) -> bool {
    let t = P::T;
    debug_assert_eq!(out.len(), t);
    debug_assert_eq!(f.len(), t);

    // `mat[c]` holds the coefficient vector of `beta^c`, so solving the system for the
    // dependency among `1, beta, ..., beta^t` yields the minimal polynomial.
    let mut mat = vec![0u16; (t + 1) * t];
    let mut scratch = vec![0u16; 2 * t];

    mat[0] = 1;
    mat[t..2 * t].copy_from_slice(f);
    for c in 2..=t {
        let (lower, upper) = mat.split_at_mut(c * t);
        let previous = &lower[(c - 1) * t..c * t];
        mul_mod_f::<P>(&mut upper[..t], previous, f, &mut scratch);
    }

    // Gaussian elimination on the `t x (t+1)` system. `mat[c][j]` is coefficient `j` of
    // `beta^c`, so the elimination runs across `c` for each column `j`.
    let at = |c: usize, j: usize| c * t + j;

    for j in 0..t {
        // Constant-time pivot search: add every later column into column `j` while column `j`
        // is still zero.
        for k in j + 1..t {
            let mask = is_zero_mask(mat[at(j, j)]);
            for c in j..=t {
                mat[at(c, j)] ^= mat[at(c, k)] & mask;
            }
        }

        if mat[at(j, j)] == 0 {
            // A zero pivot means `beta` generates a proper subfield. The reference
            // implementation declassifies this test; it depends only on the seed, and the
            // caller responds by restarting with `delta'`.
            return false;
        }

        let inv = P::Field::inv(mat[at(j, j)]);
        for c in j..=t {
            mat[at(c, j)] = P::Field::mul(mat[at(c, j)], inv);
        }

        for k in 0..t {
            if k != j {
                let factor = mat[at(j, k)];
                for c in j..=t {
                    mat[at(c, k)] ^= P::Field::mul(mat[at(c, j)], factor);
                }
            }
        }
    }

    out.copy_from_slice(&mat[at(t, 0)..at(t, 0) + t]);
    true
}

/// Evaluate the monic degree-`t` polynomial `f` at `a`.
///
/// `f` holds `t + 1` coefficients in ascending order, with `f[t] = 1` for a Goppa polynomial.
///
/// Only key generation evaluates one point at a time; decoding uses the additive FFT to reach
/// every point at once.
#[cfg(feature = "keygen")]
#[inline]
pub(crate) fn eval<P: Params>(f: &[u16], a: u16) -> u16 {
    let mut r = f[P::T];
    for i in (0..P::T).rev() {
        r = add(P::Field::mul(r, a), f[i]);
    }
    r
}

/// How many independent evaluations to interleave.
///
/// Horner's rule is a serial chain of multiplications, so evaluating one point at a time runs
/// at the latency of the field multiplier rather than its throughput. Evaluating several
/// points in lockstep fills the pipeline; eight is enough to cover the latency of both the
/// carry-less-multiply and the portable multiplier without running out of registers.
pub(crate) const EVAL_LANES: usize = 8;

/// Evaluate `f` at every element of `support`, writing the results to `out`.
#[cfg(feature = "keygen")]
pub(crate) fn eval_many<P: Params>(out: &mut [u16], f: &[u16], support: &[u16]) {
    debug_assert_eq!(out.len(), support.len());

    let mut chunks = out.chunks_exact_mut(EVAL_LANES);
    let mut points = support.chunks_exact(EVAL_LANES);

    for (dest, args) in chunks.by_ref().zip(points.by_ref()) {
        let mut acc = [f[P::T]; EVAL_LANES];
        for i in (0..P::T).rev() {
            let coefficient = f[i];
            for (r, &a) in acc.iter_mut().zip(args.iter()) {
                *r = add(P::Field::mul(*r, a), coefficient);
            }
        }
        dest.copy_from_slice(&acc);
    }

    for (dest, &a) in chunks.into_remainder().iter_mut().zip(points.remainder()) {
        *dest = eval::<P>(f, a);
    }
}

#[cfg(all(test, feature = "keygen"))]
#[allow(clippy::unwrap_used)]
mod tests {
    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    use super::*;

    /// Multiply in `F_q[y]/F(y)` by explicit long division, as an oracle for `mul_mod_f`.
    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn reference_mul<P: Params>(a: &[u16], b: &[u16]) -> Vec<u16> {
        let t = P::T;
        let mut prod = vec![0u16; 2 * t - 1];
        for i in 0..t {
            for j in 0..t {
                prod[i + j] ^= P::Field::mul(a[i], b[j]);
            }
        }
        for i in (t..2 * t - 1).rev() {
            let high = prod[i];
            prod[i] = 0;
            for tap in P::POLY_TAPS {
                prod[i - t + tap.exponent] ^= P::Field::mul(high, tap.coeff);
            }
        }
        prod[..t].to_vec()
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    struct Rng(u64);

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn elem<F: Field>(&mut self) -> u16 {
            (self.next() as u16) & F::MASK
        }
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn random_element<P: Params>(rng: &mut Rng) -> Vec<u16> {
        (0..P::T).map(|_| rng.elem::<P::Field>()).collect()
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn multiplication_agrees_with_long_division<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let mut scratch = vec![0u16; 2 * P::T];
        for _ in 0..8 {
            let a = random_element::<P>(&mut rng);
            let b = random_element::<P>(&mut rng);
            let mut out = vec![0u16; P::T];
            mul_mod_f::<P>(&mut out, &a, &b, &mut scratch);
            assert_eq!(out, reference_mul::<P>(&a, &b), "{}", P::NAME);
        }
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn multiplication_is_commutative_and_has_an_identity<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let mut scratch = vec![0u16; 2 * P::T];

        let mut one = vec![0u16; P::T];
        one[0] = 1;

        for _ in 0..8 {
            let a = random_element::<P>(&mut rng);
            let b = random_element::<P>(&mut rng);

            let mut ab = vec![0u16; P::T];
            let mut ba = vec![0u16; P::T];
            mul_mod_f::<P>(&mut ab, &a, &b, &mut scratch);
            mul_mod_f::<P>(&mut ba, &b, &a, &mut scratch);
            assert_eq!(ab, ba, "{} commutativity", P::NAME);

            let mut a1 = vec![0u16; P::T];
            mul_mod_f::<P>(&mut a1, &a, &one, &mut scratch);
            assert_eq!(a1, a, "{} identity", P::NAME);
        }
    }

    /// `g(beta) = 0` in `F_q[y]/F(y)`, evaluated with Horner's rule in that ring.
    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn minimal_polynomial_annihilates_its_generator<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let mut scratch = vec![0u16; 2 * P::T];

        let mut found = 0;
        for _ in 0..4 {
            let beta = random_element::<P>(&mut rng);
            let mut g = vec![0u16; P::T];
            if !minimal_polynomial::<P>(&mut g, &beta) {
                continue;
            }
            found += 1;

            // Horner over `F_q[y]/F(y)` starting from the implicit leading coefficient 1.
            let mut acc = vec![0u16; P::T];
            acc[0] = 1;
            for i in (0..P::T).rev() {
                let mut next = vec![0u16; P::T];
                mul_mod_f::<P>(&mut next, &acc, &beta, &mut scratch);
                next[0] ^= g[i];
                acc = next;
            }
            assert!(
                acc.iter().all(|&c| c == 0),
                "{} g(beta) should vanish",
                P::NAME
            );
        }
        assert!(found > 0, "{} produced no irreducible polynomial", P::NAME);
    }

    macro_rules! poly_tests {
        ($($feature:literal => $mod_name:ident, $params:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

                    #[test]
                    fn multiplication_matches_long_division() {
                        multiplication_agrees_with_long_division::<$params>($seed);
                    }

                    #[test]
                    fn multiplication_is_a_commutative_monoid() {
                        multiplication_is_commutative_and_has_an_identity::<$params>($seed ^ 1);
                    }

                    #[test]
                    fn minimal_polynomial_vanishes_at_beta() {
                        minimal_polynomial_annihilates_its_generator::<$params>($seed ^ 2);
                    }
                }
            )+
        };
    }

    // One parameter set per distinct `(m, t, F(y))` combination.
    poly_tests! {
        "mceliece348864" => t64_m12, McEliece348864, 0x1234_5678_9ABC_DEF0;
        "mceliece460896" => t96_m13, McEliece460896, 0x0FED_CBA9_8765_4321;
        "mceliece6960119" => t119_m13, McEliece6960119, 0x2468_ACE0_1357_9BDF;
        "mceliece8192128" => t128_m13, McEliece8192128, 0x0BAD_C0DE_F00D_1234;
    }

    #[test]
    #[cfg(feature = "mceliece8192128")]
    fn evaluation_matches_direct_expansion() {
        use crate::hazmat::params::McEliece8192128;
        type P = McEliece8192128;

        let mut rng = Rng(0xFACE_B00C_5EED_1234);
        let f: Vec<u16> = (0..=P::T)
            .map(|i| {
                if i == P::T {
                    1
                } else {
                    rng.elem::<<P as Params>::Field>()
                }
            })
            .collect();

        for _ in 0..64 {
            let a = rng.elem::<<P as Params>::Field>();
            let mut expected = 0u16;
            let mut power = 1u16;
            for &c in f.iter() {
                expected ^= <P as Params>::Field::mul(c, power);
                power = <P as Params>::Field::mul(power, a);
            }
            assert_eq!(eval::<P>(&f, a), expected);
        }
    }
}
