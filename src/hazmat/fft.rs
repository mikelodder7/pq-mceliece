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

/// Evaluate `f` at `point` by Horner's rule.
fn eval_at<F: Field>(f: &[u16], point: u16) -> u16 {
    let mut acc = 0u16;
    for &coefficient in f.iter().rev() {
        acc = F::mul(acc, point) ^ coefficient;
    }
    acc
}

/// Evaluate `f` over the affine subspace `offset + span(basis)`.
///
/// `out` receives `2^basis.len()` values, with index `j` holding the evaluation at
/// `offset + sum_i bit_i(j) * basis[i]`. `f` must have a power-of-two number of coefficients.
fn transform<F: Field>(out: &mut [u16], f: &[u16], basis: &[u16], offset: u16) {
    debug_assert_eq!(out.len(), 1 << basis.len());
    debug_assert!(f.len().is_power_of_two());

    if basis.is_empty() {
        out[0] = eval_at::<F>(f, offset);
        return;
    }
    if f.len() == 1 {
        // A constant takes the same value everywhere, which prunes the recursion as soon as
        // the polynomial runs out of degree.
        out.fill(f[0]);
        return;
    }

    let levels = basis.len();
    let beta = basis[levels - 1];
    let beta_inv = F::inv(beta);

    // g(x) = f(beta * x), so that the top basis element becomes one.
    let mut g = vec![0u16; f.len()];
    let mut power = 1u16;
    for (slot, &coefficient) in g.iter_mut().zip(f.iter()) {
        *slot = F::mul(coefficient, power);
        power = F::mul(power, beta);
    }

    taylor(&mut g);
    let half = g.len() / 2;
    let mut even = vec![0u16; half];
    let mut odd = vec![0u16; half];
    for i in 0..half {
        even[i] = g[2 * i];
        odd[i] = g[2 * i + 1];
    }

    // The image basis: tau(gamma_i) for gamma_i = basis_i / beta.
    let gamma: Vec<u16> = basis[..levels - 1]
        .iter()
        .map(|&b| F::mul(b, beta_inv))
        .collect();
    let image: Vec<u16> = gamma.iter().map(|&g| F::sq(g) ^ g).collect();

    let shifted = F::mul(offset, beta_inv);
    let image_offset = F::sq(shifted) ^ shifted;

    let sub = 1usize << (levels - 1);
    let mut low = vec![0u16; sub];
    let mut high = vec![0u16; sub];
    transform::<F>(&mut low, &even, &image, image_offset);
    transform::<F>(&mut high, &odd, &image, image_offset);

    // `omega` walks the subspace in index order, maintained incrementally rather than rebuilt.
    // Counting from `j - 1` to `j` clears the trailing ones and sets the bit at
    // `j.trailing_zeros()`, so the delta is the prefix exclusive-or up to that bit. The bit
    // index depends only on the loop counter, never on secret data.
    let mut prefix = vec![0u16; gamma.len()];
    let mut running = 0u16;
    for (slot, &g) in prefix.iter_mut().zip(gamma.iter()) {
        running ^= g;
        *slot = running;
    }

    let mut omega = shifted;
    for j in 0..sub {
        if j != 0 {
            omega ^= prefix[j.trailing_zeros() as usize];
        }
        out[j] = low[j] ^ F::mul(omega, high[j]);
        out[j + sub] = out[j] ^ high[j];
    }
}

/// Evaluate `f` at every element of `F_q`, writing `out[a] = f(a)`.
///
/// `f` holds coefficients in ascending order and may be any length up to `q`. The decoder uses
/// [`eval_all_bitrev`] instead; this natural ordering is the one the tests state correctness
/// against.
#[cfg(test)]
pub(crate) fn eval_all<P: Params>(out: &mut [u16], f: &[u16]) {
    // The standard basis makes the output index equal the field element it evaluates at.
    let basis: Vec<u16> = (0..P::M).map(|i| 1u16 << i).collect();
    evaluate::<P>(out, f, &basis);
}

/// Evaluate `f` at every element of `F_q`, writing `out[k] = f(bitrev(k))`.
///
/// Reversing the basis reverses the bits of the output index, which is exactly the indexing the
/// decoder works in: a support element is `bitrev` of a permutation image, so laying the
/// evaluations out this way lets the Beneš network line them up against positions without any
/// secret-dependent lookup.
pub(crate) fn eval_all_bitrev<P: Params>(out: &mut [u16], f: &[u16]) {
    let basis: Vec<u16> = (0..P::M).map(|i| 1u16 << (P::M - 1 - i)).collect();
    evaluate::<P>(out, f, &basis);
}

fn evaluate<P: Params>(out: &mut [u16], f: &[u16], basis: &[u16]) {
    debug_assert_eq!(out.len(), P::Q);

    // The recursion halves the coefficient count at each level, so it wants a power of two.
    let mut padded = vec![0u16; f.len().next_power_of_two()];
    padded[..f.len()].copy_from_slice(f);

    transform::<P::Field>(out, &padded, basis, 0);
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
