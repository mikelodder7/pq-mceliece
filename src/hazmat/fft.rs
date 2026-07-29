/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! The Gao-Mateer additive FFT.
//!
//! Decoding evaluates two polynomials at every element of `F_q`: the Goppa polynomial, to build
//! the position weighting, and the error locator, to find its roots. Doing that one point at a
//! time costs `q * t` field multiplications. The additive FFT costs about `(q/2) * m` instead,
//! which for `mceliece8192128` is roughly eighteen times fewer.
//!
//! The algorithm is the one from Gao and Mateer, *Additive Fast Fourier Transforms Over Finite
//! Fields* (<https://www.math.clemson.edu/~sgao/papers/GM10.pdf>). Evaluating `f` over a
//! subspace `W` with basis `b_0, ..., b_(l-1)`:
//!
//! * rescale so that the last basis element becomes one, giving `g(x) = f(b_(l-1) x)`;
//! * rewrite `g` in powers of `tau(x) = x^2 + x`, so `g(x) = g0(tau) + x g1(tau)`;
//! * recurse on `g0` and `g1` over the image basis `tau(b_i)`, which has one fewer element;
//! * recombine, using that `tau(w) = tau(w + 1)` in characteristic two, so each pair of output
//!   points shares a sub-evaluation.
//!
//! Output index `j` holds the evaluation at `offset + sum_i bit_i(j) * b_i`. Calling
//! [`eval_all`] with the standard basis `1, z, z^2, ...` therefore yields `out[a] = f(a)` with
//! `a` read as a field element, which is the ordering the decoder wants.

use super::field::Field;
use super::params::Params;

/// Rewrite `g` in powers of `tau(x) = x^2 + x`, in place.
///
/// On return, the coefficient pair `(g[2i], g[2i + 1])` is the degree-one polynomial
/// multiplying `tau^i`, so that `g(x) = sum_i (g[2i] + g[2i + 1] x) tau(x)^i`.
///
/// The split uses `tau^(n/4) = x^(n/2) + x^(n/4)`, which holds because squaring is additive in
/// characteristic two. Dividing by it is therefore just a fold of the top half onto the middle,
/// after which the top half already holds the high digits and the halves recurse independently.
fn taylor(g: &mut [u16]) {
    let n = g.len();
    if n <= 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    let quarter = n / 4;
    for i in (n / 2..n).rev() {
        g[i - quarter] ^= g[i];
    }

    let (low, high) = g.split_at_mut(n / 2);
    taylor(low);
    taylor(high);
}

/// Evaluate `f` at every element of `F_q`, writing `out[a] = f(a)`.
///
/// The decoder uses [`eval_all_bitrev`]; this natural ordering is what the tests state
/// correctness against.
#[cfg(test)]
pub(crate) fn eval_all<P: Params>(out: &mut [u16], f: &[u16]) {
    // The standard basis makes the output index equal the field element it evaluates at.
    let basis: Vec<u16> = (0..P::M).map(|i| 1u16 << i).collect();
    evaluate::<P>(out, f, basis);
}

/// Evaluate `f` at every element of `F_q`, writing `out[k] = f(bitrev(k))`.
///
/// Reversing the basis reverses the bits of the output index, which is exactly the indexing the
/// decoder works in: a support element is `bitrev` of a permutation image, so laying the
/// evaluations out this way lets the Beneš network line them up against positions without any
/// secret-dependent lookup.
pub(crate) fn eval_all_bitrev<P: Params>(out: &mut [u16], f: &[u16]) {
    let basis: Vec<u16> = (0..P::M).map(|i| 1u16 << (P::M - 1 - i)).collect();
    evaluate::<P>(out, f, basis);
}

/// Evaluate `f` over all of `F_q`, with output index `j` holding the evaluation at
/// `sum_i bit_i(j) * basis[i]`.
///
/// The transform is written as two flat passes rather than as a recursion. Every node at a
/// given depth shares the same basis and the same zero offset, so all the per-node quantities
/// the recursive form recomputes are in fact per-level constants, and the coefficient blocks
/// can live side by side in one array instead of in separately allocated halves.
fn evaluate<P: Params>(out: &mut [u16], f: &[u16], mut basis: Vec<u16>) {
    debug_assert_eq!(out.len(), P::Q);

    let length = f.len().next_power_of_two();
    let levels = length.trailing_zeros() as usize;
    debug_assert!(levels <= P::M);

    let mut coefficients = vec![0u16; length];
    coefficients[..f.len()].copy_from_slice(f);

    // Coefficient pass: rescale, rewrite in powers of tau, and split even from odd digits.
    // After `levels` rounds each entry is the constant one leaf of the recursion would see.
    let mut scratch = vec![0u16; length];
    let mut gammas: Vec<Vec<u16>> = Vec::with_capacity(levels);

    for depth in 0..levels {
        let beta = basis[basis.len() - 1];
        let beta_inverse = P::Field::inv(beta);
        let block = length >> depth;
        let half = block / 2;

        for start in (0..length).step_by(block) {
            let segment = &mut coefficients[start..start + block];

            let mut power = 1u16;
            for coefficient in segment.iter_mut() {
                *coefficient = P::Field::mul(*coefficient, power);
                power = P::Field::mul(power, beta);
            }

            taylor(segment);

            for i in 0..half {
                scratch[i] = segment[2 * i];
                scratch[half + i] = segment[2 * i + 1];
            }
            segment.copy_from_slice(&scratch[..block]);
        }

        let gamma: Vec<u16> = basis[..basis.len() - 1]
            .iter()
            .map(|&b| P::Field::mul(b, beta_inverse))
            .collect();
        basis = gamma.iter().map(|&g| P::Field::sq(g) ^ g).collect();
        gammas.push(gamma);
    }

    // Each leaf constant takes the same value across its whole output block.
    let span = P::Q >> levels;
    for (leaf, &constant) in coefficients.iter().enumerate() {
        out[leaf * span..(leaf + 1) * span].fill(constant);
    }

    // Butterfly pass, deepest level first. `tau(w) = tau(w + 1)` in characteristic two, so the
    // two halves of a block share a sub-evaluation and differ only by the top basis element.
    let mut prefix = [0u16; 16];

    for depth in (0..levels).rev() {
        let gamma = &gammas[depth];
        let block = P::Q >> depth;
        let half = block / 2;
        debug_assert_eq!(half, 1 << gamma.len());

        // Walking `j` upwards, the selected basis elements change by a prefix exclusive-or:
        // counting from `j - 1` clears the trailing ones and sets one more bit. Keeping the
        // running sum in a local rather than a table keeps it in a register through the inner
        // loop, which is the hottest one in the transform. Both indices are loop counters,
        // never secret data.
        let mut running = 0u16;
        for (slot, &g) in prefix.iter_mut().zip(gamma.iter()) {
            running ^= g;
            *slot = running;
        }

        for start in (0..P::Q).step_by(block) {
            let (low, high) = out[start..start + block].split_at_mut(half);

            // The first point of every block selects no basis element at all.
            high[0] ^= low[0];

            let mut omega = 0u16;
            for j in 1..half {
                omega ^= prefix[j.trailing_zeros() as usize];
                let top = high[j];
                low[j] ^= P::Field::mul(omega, top);
                high[j] = low[j] ^ top;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Evaluate `f` at `point` by Horner's rule, as the independent oracle.
    fn eval_at<F: Field>(f: &[u16], point: u16) -> u16 {
        let mut acc = 0u16;
        for &coefficient in f.iter().rev() {
            acc = F::mul(acc, point) ^ coefficient;
        }
        acc
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

    /// Expand a tau-adic representation back to the monomial basis, as an oracle for `taylor`.
    fn from_tau_adic<F: Field>(digits: &[u16], point: u16) -> u16 {
        let tau = F::sq(point) ^ point;
        let mut acc = 0u16;
        for pair in digits.chunks_exact(2).rev() {
            acc = F::mul(acc, tau) ^ (pair[0] ^ F::mul(pair[1], point));
        }
        acc
    }

    fn taylor_preserves_the_polynomial<F: Field>(seed: u64) {
        let mut rng = Rng(seed);
        for &len in &[2usize, 4, 8, 16, 64, 256] {
            let f: Vec<u16> = (0..len).map(|_| (rng.next() as u16) & F::MASK).collect();
            let mut digits = f.clone();
            taylor(&mut digits);

            for _ in 0..32 {
                let point = (rng.next() as u16) & F::MASK;
                assert_eq!(
                    from_tau_adic::<F>(&digits, point),
                    eval_at::<F>(&f, point),
                    "length {len}"
                );
            }
        }
    }

    /// The transform must agree with Horner's rule at every single point.
    fn transform_matches_horner<P: Params>(seed: u64) {
        let mut rng = Rng(seed);

        for &degree in &[1usize, 2, 5, P::T, P::T + 1] {
            let f: Vec<u16> = (0..degree)
                .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
                .collect();

            let mut out = vec![0u16; P::Q];
            eval_all::<P>(&mut out, &f);

            for (a, &value) in out.iter().enumerate() {
                assert_eq!(
                    value,
                    eval_at::<P::Field>(&f, a as u16),
                    "{} degree {degree} point {a}",
                    P::NAME
                );
            }
        }
    }

    /// A monic degree-`t` polynomial is the shape the decoder actually evaluates.
    fn monic_locator_shape_matches_horner<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let mut f: Vec<u16> = (0..=P::T)
            .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
            .collect();
        f[P::T] = 1;

        let mut out = vec![0u16; P::Q];
        eval_all::<P>(&mut out, &f);

        for (a, &value) in out.iter().enumerate() {
            assert_eq!(value, eval_at::<P::Field>(&f, a as u16), "point {a}");
        }
    }

    /// The bit-reversed ordering must be exactly the natural one with the index bits flipped.
    fn bitrev_ordering_matches<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let f: Vec<u16> = (0..=P::T)
            .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
            .collect();

        let mut natural = vec![0u16; P::Q];
        let mut reversed = vec![0u16; P::Q];
        eval_all::<P>(&mut natural, &f);
        eval_all_bitrev::<P>(&mut reversed, &f);

        for (k, &value) in reversed.iter().enumerate() {
            let a = <P::Field as Field>::bitrev(k as u16) as usize;
            assert_eq!(value, natural[a], "{} index {k}", P::NAME);
        }
    }

    /// The zero polynomial and constants are the recursion's base cases.
    fn constants_evaluate_everywhere<P: Params>() {
        let mut out = vec![0u16; P::Q];

        eval_all::<P>(&mut out, &[0]);
        assert!(out.iter().all(|&v| v == 0));

        eval_all::<P>(&mut out, &[0x0ABC]);
        assert!(out.iter().all(|&v| v == 0x0ABC));

        // `x` evaluates to the point itself, which pins down the output ordering.
        eval_all::<P>(&mut out, &[0, 1]);
        for (a, &value) in out.iter().enumerate() {
            assert_eq!(value, a as u16, "identity at {a}");
        }
    }

    macro_rules! fft_tests {
        ($($feature:literal => $mod_name:ident, $params:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

                    #[test]
                    fn taylor_round_trips() {
                        taylor_preserves_the_polynomial::<<$params as Params>::Field>($seed);
                    }

                    #[test]
                    fn matches_horner_at_every_point() {
                        transform_matches_horner::<$params>($seed ^ 1);
                    }

                    #[test]
                    fn monic_locator_matches_horner() {
                        monic_locator_shape_matches_horner::<$params>($seed ^ 2);
                    }

                    #[test]
                    fn constants_and_identity() {
                        constants_evaluate_everywhere::<$params>();
                    }

                    #[test]
                    fn bitrev_matches_natural_order() {
                        bitrev_ordering_matches::<$params>($seed ^ 3);
                    }
                }
            )+
        };
    }

    // One parameter set per field width and per distinct `t`.
    fft_tests! {
        "mceliece348864" => set_348864, McEliece348864, 0x1234_5678_9ABC_DEF0;
        "mceliece460896" => set_460896, McEliece460896, 0x0FED_CBA9_8765_4321;
        "mceliece6960119" => set_6960119, McEliece6960119, 0x2468_ACE0_1357_9BDF;
        "mceliece8192128" => set_8192128, McEliece8192128, 0x0BAD_C0DE_F00D_1234;
    }
}
