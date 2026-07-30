/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! The Gao-Mateer additive FFT.
//!
//! Key generation and decoding evaluate polynomials at every element of `F_q`: key generation
//! evaluates the Goppa polynomial to build the parity-check matrix, while decoding evaluates it
//! for position weighting and evaluates the error locator to find its roots. Doing that one
//! point at a time costs `q * t` field multiplications. The additive FFT costs about `(q/2) * m`
//! instead, which for `mceliece8192128` is roughly eighteen times fewer.
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
//!
//! Key generation and the decoder run the bit-sliced forms, [`eval_all_bitrev_sliced`] and
//! [`eval_transpose_bitrev_sliced`]. The scalar forms remain as the reference the bit-sliced
//! ones are checked against, point by point and lane by lane.

use super::field::Field;
use super::params::Params;
use super::vec::{LANE_BIT_MASKS, LANE_BITS, LANES, MAX_BITS, Slice, Tables, Word};

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

/// The transpose of [`taylor`], as a linear map over `F_q`.
///
/// Transposing a product of elementary row additions reverses their order and swaps each
/// source with its destination, so the fold runs upwards and writes the other way round, and
/// the recursion happens before it rather than after.
#[cfg(any(feature = "decapsulate", test))]
fn taylor_transpose(g: &mut [u16]) {
    let n = g.len();
    if n <= 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    let quarter = n / 4;
    {
        let (low, high) = g.split_at_mut(n / 2);
        taylor_transpose(low);
        taylor_transpose(high);
    }

    for i in n / 2..n {
        g[i] ^= g[i - quarter];
    }
}

/// The per-level constants shared by the transform and its transpose.
///
/// Every node at a given depth sees the same basis and the same zero offset, so the scaling
/// factor and the normalized basis depend on the depth alone.
struct Levels {
    /// The basis element each depth rescales by.
    betas: [u16; MAX_BITS],
    /// The remaining basis at each depth, normalized so the rescaled element is one.
    gammas: [[u16; MAX_BITS]; MAX_BITS],
    /// Number of initialized transform levels.
    count: usize,
    /// Start of each depth's scalar rescaling powers.
    power_starts: [usize; MAX_BITS + 1],
    /// Powers of each level's public rescaling element.
    powers: Vec<u16>,
    /// Start of each depth's precomputed bit-sliced subspace sums.
    omega_starts: [usize; MAX_BITS + 1],
    /// Bit-sliced subspace sums, shared by every transform in one decode.
    omegas: Vec<Slice>,
}

enum LevelPlan {
    Shared(&'static Levels),
    Owned(Box<Levels>),
}

impl core::ops::Deref for LevelPlan {
    type Target = Levels;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Shared(levels) => levels,
            Self::Owned(levels) => levels,
        }
    }
}

fn levels_for<P: Params>(count: usize, basis: &[u16]) -> Levels {
    debug_assert!(count <= basis.len());
    debug_assert!(basis.len() <= MAX_BITS);
    let mut betas = [0u16; MAX_BITS];
    let mut gammas = [[0u16; MAX_BITS]; MAX_BITS];
    let mut current = [0u16; MAX_BITS];
    current[..basis.len()].copy_from_slice(basis);
    let mut current_len = basis.len();

    for depth in 0..count {
        let beta = current[current_len - 1];
        let beta_inverse = P::Field::inv(beta);
        let gamma_len = current_len - 1;
        for i in 0..gamma_len {
            gammas[depth][i] = P::Field::mul(current[i], beta_inverse);
        }
        for i in 0..gamma_len {
            let gamma = gammas[depth][i];
            current[i] = P::Field::sq(gamma) ^ gamma;
        }
        current_len = gamma_len;
        betas[depth] = beta;
    }

    Levels {
        betas,
        gammas,
        count,
        power_starts: [0; MAX_BITS + 1],
        powers: Vec::new(),
        omega_starts: [0; MAX_BITS + 1],
        omegas: Vec::new(),
    }
}

fn levels_with_powers<P: Params>(count: usize, basis: &[u16], length: usize) -> Levels {
    let mut levels = levels_for::<P>(count, basis);
    for depth in 0..count {
        levels.power_starts[depth] = levels.powers.len();
        let mut power = 1u16;
        for _ in 0..length >> depth {
            levels.powers.push(power);
            power = P::Field::mul(power, levels.betas[depth]);
        }
    }
    levels.power_starts[count] = levels.powers.len();
    levels
}

/// Fill `prefix` so that walking `j` upwards accumulates the basis elements its bits select.
///
/// Counting from `j - 1` to `j` clears the trailing ones and sets one further bit, so the delta
/// is the prefix exclusive-or up to `j.trailing_zeros()`.
#[cfg(test)]
fn subspace_prefix(prefix: &mut [u16; 16], gamma: &[u16]) {
    let mut running = 0u16;
    for (slot, &g) in prefix.iter_mut().zip(gamma.iter()) {
        running ^= g;
        *slot = running;
    }
}

/// The bit-reversed evaluation basis: index `k` stands for the field element `bitrev(k)`.
fn reversed_basis<P: Params>() -> [u16; MAX_BITS] {
    let mut basis = [0u16; MAX_BITS];
    for (i, slot) in basis.iter_mut().take(P::M).enumerate() {
        *slot = 1u16 << (P::M - 1 - i);
    }
    basis
}

/// Public constants and reusable coefficient storage shared by one decoding operation.
pub(crate) struct TransformWorkspace {
    levels: LevelPlan,
    storage: Vec<u16>,
}

impl TransformWorkspace {
    /// Allocate for the largest transform the decoder performs.
    pub(crate) fn new<P: Params>() -> Self {
        use std::sync::OnceLock;

        static GF12_LEVELS: OnceLock<Levels> = OnceLock::new();
        static GF13_LEVELS: OnceLock<Levels> = OnceLock::new();

        let length = (2 * P::T).next_power_of_two();
        let build = || {
            let basis = reversed_basis::<P>();
            let count = length.trailing_zeros() as usize;
            let mut levels = levels_with_powers::<P>(count, &basis[..P::M], length);
            for depth in 0..count {
                levels.omega_starts[depth] = levels.omegas.len();
                let gamma = &levels.gammas[depth][..P::M - depth - 1];
                let half = P::Q >> (depth + 1);
                let omega_count = if half >= LANES { half / LANES } else { 1 };
                for offset in 0..omega_count {
                    levels.omegas.push(omega_group(gamma, offset * LANES, P::M));
                }
            }
            levels.omega_starts[count] = levels.omegas.len();
            levels
        };
        // A user-defined `Params` implementation may also have 12 or 13 index bits. Reuse a
        // process-wide plan only when its field has the standardized modulus that produced the
        // cached constants; custom fields receive an owned plan.
        let standardized_field = P::Field::BITS == P::M
            && match P::M {
                12 => P::Field::reduce(1 << 12) == 0b1001,
                13 => P::Field::reduce(1 << 13) == 0b1_1011,
                _ => false,
            };
        let levels = match (P::M, length, standardized_field) {
            (12, 128, true) => LevelPlan::Shared(GF12_LEVELS.get_or_init(build)),
            (13, 256, true) => LevelPlan::Shared(GF13_LEVELS.get_or_init(build)),
            _ => LevelPlan::Owned(Box::new(build())),
        };
        Self {
            levels,
            storage: vec![0; 2 * length],
        }
    }
}

impl Drop for TransformWorkspace {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.storage.zeroize();
    }
}

/// Evaluate `f` at every element of `F_q`, writing `out[a] = f(a)`.
///
/// The decoder uses [`eval_all_bitrev`]; this natural ordering is what the tests state
/// correctness against.
#[cfg(test)]
pub(crate) fn eval_all<P: Params>(out: &mut [u16], f: &[u16]) {
    // The standard basis makes the output index equal the field element it evaluates at.
    let mut basis = [0u16; MAX_BITS];
    for (i, slot) in basis.iter_mut().take(P::M).enumerate() {
        *slot = 1u16 << i;
    }
    evaluate::<P>(out, f, &basis[..P::M]);
}

/// Evaluate `f` at every element of `F_q`, writing `out[k] = f(bitrev(k))`.
///
/// Reversing the basis reverses the bits of the output index, which is exactly the indexing the
/// decoder works in: a support element is `bitrev` of a permutation image, so laying the
/// evaluations out this way lets the Beneš network line them up against positions without any
/// secret-dependent lookup.
#[cfg(test)]
pub(crate) fn eval_all_bitrev<P: Params>(out: &mut [u16], f: &[u16]) {
    let basis = reversed_basis::<P>();
    evaluate::<P>(out, f, &basis[..P::M]);
}

/// Rescale, rewrite in powers of tau, and split even from odd digits, once per level.
///
/// After `levels` rounds each entry is the constant that one leaf of the recursion would see.
fn coefficient_pass<P: Params>(
    coefficients: &mut [u16],
    scratch: &mut [u16],
    plan: &Levels,
    levels: usize,
) {
    let length = coefficients.len();
    debug_assert_eq!(scratch.len(), length);

    for depth in 0..levels {
        let block = length >> depth;
        let half = block / 2;

        for start in (0..length).step_by(block) {
            let segment = &mut coefficients[start..start + block];

            let powers = &plan.powers[plan.power_starts[depth]..plan.power_starts[depth] + block];
            for (coefficient, &power) in segment.iter_mut().zip(powers.iter()) {
                *coefficient = P::Field::mul(*coefficient, power);
            }

            taylor(segment);

            for i in 0..half {
                scratch[i] = segment[2 * i];
                scratch[half + i] = segment[2 * i + 1];
            }
            segment.copy_from_slice(&scratch[..block]);
        }
    }
}

/// The transpose of [`coefficient_pass`]: the level order reverses, the even-odd split becomes
/// an interleave, and the Taylor step runs backwards. Rescaling is diagonal and self-transposed.
#[cfg(any(feature = "decapsulate", test))]
fn coefficient_pass_transpose<P: Params>(
    coefficients: &mut [u16],
    scratch: &mut [u16],
    plan: &Levels,
    levels: usize,
) {
    let length = coefficients.len();
    debug_assert_eq!(scratch.len(), length);

    for depth in (0..levels).rev() {
        let block = length >> depth;
        let half = block / 2;

        for start in (0..length).step_by(block) {
            let segment = &mut coefficients[start..start + block];

            for i in 0..half {
                scratch[2 * i] = segment[i];
                scratch[2 * i + 1] = segment[half + i];
            }
            segment.copy_from_slice(&scratch[..block]);

            taylor_transpose(segment);

            let powers = &plan.powers[plan.power_starts[depth]..plan.power_starts[depth] + block];
            for (coefficient, &power) in segment.iter_mut().zip(powers.iter()) {
                *coefficient = P::Field::mul(*coefficient, power);
            }
        }
    }
}

/// Evaluate `f` over all of `F_q`, with output index `j` holding the evaluation at
/// `sum_i bit_i(j) * basis[i]`.
///
/// The transform is written as two flat passes rather than as a recursion, since all the
/// per-node quantities a recursion would recompute are really per-level constants.
#[cfg(test)]
fn evaluate<P: Params>(out: &mut [u16], f: &[u16], basis: &[u16]) {
    debug_assert_eq!(out.len(), P::Q);

    let length = f.len().next_power_of_two();
    let levels = length.trailing_zeros() as usize;
    debug_assert!(levels <= P::M);
    let plan = levels_with_powers::<P>(levels, basis, length);

    let mut coefficients = vec![0u16; length];
    let mut scratch = vec![0u16; length];
    coefficients[..f.len()].copy_from_slice(f);
    coefficient_pass::<P>(&mut coefficients, &mut scratch, &plan, levels);

    // Each leaf constant takes the same value across its whole output block.
    let span = P::Q >> levels;
    for (leaf, &constant) in coefficients.iter().enumerate() {
        out[leaf * span..(leaf + 1) * span].fill(constant);
    }

    // Butterfly pass, deepest level first. `tau(w) = tau(w + 1)` in characteristic two, so the
    // two halves of a block share a sub-evaluation and differ only by the top basis element.
    let mut prefix = [0u16; 16];
    for depth in (0..levels).rev() {
        let block = P::Q >> depth;
        let half = block / 2;
        let gamma = &plan.gammas[depth][..P::M - depth - 1];
        debug_assert_eq!(half, 1 << gamma.len());
        subspace_prefix(&mut prefix, gamma);

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

/// Apply the transpose of [`eval_all_bitrev`], writing `out[j] = sum_k values[k] * bitrev(k)^j`.
///
/// That sum is exactly the Goppa syndrome, so decoding gets it for the price of one transform
/// instead of the `q * 2t` field multiplications the direct series costs.
///
/// Multipoint evaluation is a linear map, and its matrix is the transpose of the one the
/// syndrome needs. So the transpose is the same three passes run backwards, with each step
/// replaced by its own transpose: the butterfly's two-by-two matrix flipped, the broadcast that
/// fills a leaf block turned into a sum over it, the even-odd split turned back into an
/// interleave, and the Taylor step reversed. Rescaling is diagonal and transposes to itself.
#[cfg(test)]
pub(crate) fn eval_transpose_bitrev<P: Params>(out: &mut [u16], values: &[u16]) {
    debug_assert_eq!(values.len(), P::Q);

    let length = out.len().next_power_of_two();
    let levels = length.trailing_zeros() as usize;
    debug_assert!(levels <= P::M);
    let basis = reversed_basis::<P>();
    let plan = levels_with_powers::<P>(levels, &basis[..P::M], length);

    let mut work = values.to_vec();
    let mut scratch = vec![0u16; length];

    // Butterflies transposed: the level order reverses, and each two-by-two step becomes
    // `(low, high) -> (low + high, high + omega (low + high))`.
    let mut prefix = [0u16; 16];
    for depth in 0..levels {
        let block = P::Q >> depth;
        let half = block / 2;
        let gamma = &plan.gammas[depth][..P::M - depth - 1];
        debug_assert_eq!(half, 1 << gamma.len());
        subspace_prefix(&mut prefix, gamma);

        for start in (0..P::Q).step_by(block) {
            let (low, high) = work[start..start + block].split_at_mut(half);

            low[0] ^= high[0];

            let mut omega = 0u16;
            for j in 1..half {
                omega ^= prefix[j.trailing_zeros() as usize];
                let sum = low[j] ^ high[j];
                high[j] ^= P::Field::mul(omega, sum);
                low[j] = sum;
            }
        }
    }

    // Broadcasting a constant across a leaf block transposes into summing that block.
    let span = P::Q >> levels;
    let mut coefficients = vec![0u16; length];
    for (leaf, slot) in coefficients.iter_mut().enumerate() {
        let mut acc = 0u16;
        for &value in &work[leaf * span..(leaf + 1) * span] {
            acc ^= value;
        }
        *slot = acc;
    }

    coefficient_pass_transpose::<P>(&mut coefficients, &mut scratch, &plan, levels);
    out.copy_from_slice(&coefficients[..out.len()]);
}

/// The number of machine words a bit-sliced copy of all of `F_q` occupies.
pub(crate) const fn sliced_words<P: Params>() -> usize {
    (P::Q / LANES) * P::M
}

/// The subspace sum for a run of [`LANES`] consecutive indices, built directly in bit-sliced
/// form.
///
/// `omega[j]` is the sum of the basis elements the bits of `j` select. Bits below the lane
/// width vary within the group and contribute a fixed alternating mask; bits at or above it are
/// constant across the group. Building it this way avoids ever materializing and packing the
/// scalar values.
fn omega_group(gamma: &[u16], base: usize, bits: usize) -> Slice {
    let mut out = [0; MAX_BITS];

    for (p, &element) in gamma.iter().enumerate() {
        let pattern: Word = if p < LANE_BITS {
            LANE_BIT_MASKS[p]
        } else if (base >> p) & 1 == 1 {
            Word::MAX
        } else {
            continue;
        };

        for (i, slot) in out.iter_mut().take(bits).enumerate() {
            if (element >> i) & 1 == 1 {
                *slot ^= pattern;
            }
        }
    }

    out
}

/// Read one bit-sliced group.
#[inline]
fn load(data: &[Word], group: usize, bits: usize) -> Slice {
    let mut out = [0; MAX_BITS];
    out[..bits].copy_from_slice(&data[group * bits..group * bits + bits]);
    out
}

/// Write one bit-sliced group.
#[inline]
fn store(data: &mut [Word], group: usize, bits: usize, value: &Slice) {
    data[group * bits..group * bits + bits].copy_from_slice(&value[..bits]);
}

/// Evaluate `f` at every element of `F_q`, leaving the result bit-sliced.
///
/// The coefficient side stays scalar because it is only a few hundred values, but the `q`-sized
/// side never exists in scalar form at all: the leaves are built straight into bit-planes and
/// every butterfly runs on whole machine words.
pub(crate) fn eval_all_bitrev_sliced<P: Params>(
    out: &mut [Word],
    f: &[u16],
    tables: &Tables<P::Field>,
    workspace: &mut TransformWorkspace,
) {
    debug_assert_eq!(out.len(), sliced_words::<P>());

    let bits = P::M;
    let length = f.len().next_power_of_two();
    let levels = length.trailing_zeros() as usize;
    debug_assert!(levels <= P::M);
    let TransformWorkspace {
        levels: plan,
        storage,
    } = workspace;
    debug_assert!(plan.count >= levels);

    let workspace_length = storage.len() / 2;
    let (coefficients, scratch) = storage.split_at_mut(workspace_length);
    let coefficients = &mut coefficients[..length];
    let scratch = &mut scratch[..length];
    coefficients.fill(0);
    coefficients[..f.len()].copy_from_slice(f);
    coefficient_pass::<P>(coefficients, scratch, plan, levels);

    // Leaves, built directly as bit-planes. Each constant covers `span` consecutive lanes.
    let span = P::Q >> levels;
    let groups = P::Q / LANES;
    for group in 0..groups {
        let base = group * LANES;
        let mut words = [0; MAX_BITS];

        let mut lane = 0;
        while lane < LANES {
            let take = span.min(LANES - lane);
            let mask: Word = if take == LANES {
                Word::MAX
            } else {
                ((1 << take) - 1) << lane
            };
            let constant = coefficients[(base + lane) / span];
            for (i, slot) in words.iter_mut().take(bits).enumerate() {
                if (constant >> i) & 1 == 1 {
                    *slot |= mask;
                }
            }
            lane += take;
        }

        store(out, group, bits, &words);
    }

    let mut product = [0; MAX_BITS];
    for depth in (0..levels).rev() {
        let half = P::Q >> (depth + 1);

        if half >= LANES {
            // The paired elements land in different groups, so every operation is word wide.
            let half_groups = half / LANES;
            // The subspace sum depends on the offset within a block, not on which block, so it
            // is built once per offset rather than once per pair.
            for offset in 0..half_groups {
                let omega = &plan.omegas[plan.omega_starts[depth] + offset];
                for base_group in (0..groups).step_by(half_groups * 2) {
                    let low_group = base_group + offset;

                    let mut low = load(out, low_group, bits);
                    let high = load(out, low_group + half_groups, bits);

                    tables.mul(&mut product, &high, omega);
                    for i in 0..bits {
                        low[i] ^= product[i];
                    }
                    let mut new_high = [0; MAX_BITS];
                    for i in 0..bits {
                        new_high[i] = low[i] ^ high[i];
                    }

                    store(out, low_group, bits, &low);
                    store(out, low_group + half_groups, bits, &new_high);
                }
            }
        } else {
            // The pair sits inside one group, `half` lanes apart.
            let omega = &plan.omegas[plan.omega_starts[depth]];
            let keep = lane_run_mask(half);

            for group in 0..groups {
                let word = load(out, group, bits);
                let mut low = [0; MAX_BITS];
                let mut high = [0; MAX_BITS];
                for i in 0..bits {
                    low[i] = word[i] & keep;
                    high[i] = (word[i] >> half) & keep;
                }

                tables.mul(&mut product, &high, omega);
                let mut merged = [0; MAX_BITS];
                for i in 0..bits {
                    let new_low = low[i] ^ product[i];
                    let new_high = new_low ^ high[i];
                    merged[i] = new_low | (new_high << half);
                }

                store(out, group, bits, &merged);
            }
        }
    }
}

/// Apply the transpose of [`eval_all_bitrev_sliced`], reading bit-sliced values.
#[cfg(feature = "decapsulate")]
pub(crate) fn eval_transpose_bitrev_sliced<P: Params>(
    out: &mut [u16],
    work: &mut [Word],
    tables: &Tables<P::Field>,
    workspace: &mut TransformWorkspace,
) {
    debug_assert_eq!(work.len(), sliced_words::<P>());

    let bits = P::M;
    let length = out.len().next_power_of_two();
    let levels = length.trailing_zeros() as usize;
    debug_assert!(levels <= P::M);
    let TransformWorkspace {
        levels: plan,
        storage,
    } = workspace;
    debug_assert!(plan.count >= levels);
    let workspace_length = storage.len() / 2;
    let (coefficients, scratch) = storage.split_at_mut(workspace_length);

    let groups = P::Q / LANES;
    let mut product = [0; MAX_BITS];

    // Butterflies transposed: the level order reverses and each two-by-two step flips.
    for depth in 0..levels {
        let half = P::Q >> (depth + 1);

        if half >= LANES {
            let half_groups = half / LANES;
            for offset in 0..half_groups {
                let omega = &plan.omegas[plan.omega_starts[depth] + offset];
                for base_group in (0..groups).step_by(half_groups * 2) {
                    let low_group = base_group + offset;

                    let low = load(work, low_group, bits);
                    let mut high = load(work, low_group + half_groups, bits);

                    let mut sum = [0; MAX_BITS];
                    for i in 0..bits {
                        sum[i] = low[i] ^ high[i];
                    }
                    tables.mul(&mut product, &sum, omega);
                    for i in 0..bits {
                        high[i] ^= product[i];
                    }

                    store(work, low_group, bits, &sum);
                    store(work, low_group + half_groups, bits, &high);
                }
            }
        } else {
            let omega = &plan.omegas[plan.omega_starts[depth]];
            let keep = lane_run_mask(half);

            for group in 0..groups {
                let word = load(work, group, bits);
                let mut low = [0; MAX_BITS];
                let mut high = [0; MAX_BITS];
                for i in 0..bits {
                    low[i] = word[i] & keep;
                    high[i] = (word[i] >> half) & keep;
                }

                let mut sum = [0; MAX_BITS];
                for i in 0..bits {
                    sum[i] = low[i] ^ high[i];
                }
                tables.mul(&mut product, &sum, omega);

                let mut merged = [0; MAX_BITS];
                for i in 0..bits {
                    let new_high = high[i] ^ product[i];
                    merged[i] = sum[i] | (new_high << half);
                }

                store(work, group, bits, &merged);
            }
        }
    }

    // Broadcasting a constant across a leaf block transposes into summing that block, which in
    // bit-sliced form is the parity of each masked bit-plane.
    let span = P::Q >> levels;
    let coefficients = &mut coefficients[..length];
    let scratch = &mut scratch[..length];
    for (leaf, slot) in coefficients.iter_mut().enumerate() {
        let start = leaf * span;
        let mut value = 0u16;
        for i in 0..bits {
            let mut parity = 0u32;
            let mut covered = 0;
            while covered < span {
                let group = (start + covered) / LANES;
                let lane = (start + covered) % LANES;
                let take = (LANES - lane).min(span - covered);
                let mask: Word = if take == LANES {
                    Word::MAX
                } else {
                    ((1 << take) - 1) << lane
                };
                parity ^= (work[group * bits + i] & mask).count_ones();
                covered += take;
            }
            value |= ((parity & 1) as u16) << i;
        }
        *slot = value;
    }

    coefficient_pass_transpose::<P>(coefficients, scratch, plan, levels);
    out.copy_from_slice(&coefficients[..out.len()]);
}

/// A mask selecting the low `run` lanes of every `2 * run` lane block.
fn lane_run_mask(run: usize) -> Word {
    let mut mask: Word = 0;
    let mut base = 0;
    while base < LANES {
        mask |= ((1 << run) - 1) << base;
        base += run * 2;
    }
    mask
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

    /// The transpose must satisfy the defining adjoint identity `<Vc, w> = <c, transpose(V) w>`
    /// for every pair of vectors, which pins it down completely.
    fn transpose_is_adjoint_to_evaluation<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let width = 2 * P::T;

        for _ in 0..4 {
            let c: Vec<u16> = (0..width)
                .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
                .collect();
            let w: Vec<u16> = (0..P::Q)
                .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
                .collect();

            let mut evaluated = vec![0u16; P::Q];
            eval_all_bitrev::<P>(&mut evaluated, &c);
            let mut left = 0u16;
            for (v, x) in evaluated.iter().zip(w.iter()) {
                left ^= <P::Field as Field>::mul(*v, *x);
            }

            let mut transposed = vec![0u16; width];
            eval_transpose_bitrev::<P>(&mut transposed, &w);
            let mut right = 0u16;
            for (s, x) in transposed.iter().zip(c.iter()) {
                right ^= <P::Field as Field>::mul(*s, *x);
            }

            assert_eq!(left, right, "{} adjoint identity", P::NAME);
        }
    }

    /// And it must equal the syndrome sum written out directly.
    fn transpose_matches_the_syndrome_sum<P: Params>(seed: u64) {
        let mut rng = Rng(seed);
        let width = 2 * P::T;

        // A sparse input keeps the reference sum cheap while still covering the whole domain.
        let mut w = vec![0u16; P::Q];
        for _ in 0..64 {
            let k = (rng.next() as usize) % P::Q;
            w[k] = (rng.next() as u16) & <P::Field as Field>::MASK;
        }

        let mut expected = vec![0u16; width];
        for (k, &weight) in w.iter().enumerate() {
            if weight == 0 {
                continue;
            }
            let point = <P::Field as Field>::bitrev(k as u16);
            let mut power = 1u16;
            for slot in expected.iter_mut() {
                *slot ^= <P::Field as Field>::mul(weight, power);
                power = <P::Field as Field>::mul(power, point);
            }
        }

        let mut actual = vec![0u16; width];
        eval_transpose_bitrev::<P>(&mut actual, &w);
        assert_eq!(actual, expected, "{} syndrome sum", P::NAME);
    }

    /// The bit-sliced transforms must agree with the scalar ones exactly, in every lane.
    fn sliced_matches_scalar<P: Params>(seed: u64) {
        use crate::hazmat::vec::{Tables, unpack};

        let mut rng = Rng(seed);
        let tables = Tables::<P::Field>::new();
        let mut workspace = TransformWorkspace::new::<P>();
        let width = 2 * P::T;

        for degree in [1usize, P::T, P::T + 1] {
            let f: Vec<u16> = (0..degree)
                .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
                .collect();

            let mut expected = vec![0u16; P::Q];
            eval_all_bitrev::<P>(&mut expected, &f);

            let mut sliced = vec![0; sliced_words::<P>()];
            eval_all_bitrev_sliced::<P>(&mut sliced, &f, &tables, &mut workspace);

            let mut actual = vec![0u16; P::Q];
            for group in 0..P::Q / LANES {
                let words = load(&sliced, group, P::M);
                unpack::<P::Field>(&mut actual[group * LANES..(group + 1) * LANES], &words);
            }
            assert_eq!(actual, expected, "{} forward degree {degree}", P::NAME);
        }

        // And the transpose, fed the bit-sliced form of a random input. Only decapsulation
        // computes syndromes, so a build without it has no transposed transform to check.
        #[cfg(feature = "decapsulate")]
        for _ in 0..3 {
            let values: Vec<u16> = (0..P::Q)
                .map(|_| (rng.next() as u16) & <P::Field as Field>::MASK)
                .collect();

            let mut expected = vec![0u16; width];
            eval_transpose_bitrev::<P>(&mut expected, &values);

            let mut sliced = vec![0; sliced_words::<P>()];
            for group in 0..P::Q / LANES {
                let mut words = [0; MAX_BITS];
                crate::hazmat::vec::pack::<P::Field>(
                    &mut words,
                    &values[group * LANES..(group + 1) * LANES],
                );
                store(&mut sliced, group, P::M, &words);
            }

            let mut actual = vec![0u16; width];
            eval_transpose_bitrev_sliced::<P>(&mut actual, &mut sliced, &tables, &mut workspace);
            assert_eq!(actual, expected, "{} transpose", P::NAME);
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

                    #[test]
                    fn transpose_is_adjoint() {
                        transpose_is_adjoint_to_evaluation::<$params>($seed ^ 4);
                    }

                    #[test]
                    fn bit_sliced_matches_scalar() {
                        sliced_matches_scalar::<$params>($seed ^ 6);
                    }

                    #[test]
                    fn transpose_equals_the_syndrome_sum() {
                        transpose_matches_the_syndrome_sum::<$params>($seed ^ 5);
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
