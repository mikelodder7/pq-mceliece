/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Decapsulation: syndrome computation, Berlekamp-Massey decoding and implicit rejection.

use super::benes::{apply_benes_prepared, prepare_conds, prepared_cond_words};
use super::fft::{
    TransformWorkspace, eval_all_bitrev_sliced, eval_transpose_bitrev_sliced, sliced_words,
};
use super::field::Field;
#[cfg(test)]
use super::field::add;
use super::hash::{HASH_ENCAPSULATION, HASH_PLAINTEXT_CONFIRMATION, HASH_REJECTION, hash_32};
use super::params::Params;
use super::vec::{LANES, MAX_BITS, Slice, Slice64, Tables, Word};
use ctutils::{Choice, CtAssign};

/// Read the Goppa polynomial from a private key, appending its implicit leading one.
pub(crate) fn load_goppa<P: Params>(sk: &[u8], goppa: &mut [u16]) {
    debug_assert_eq!(goppa.len(), P::T + 1);

    for (i, slot) in goppa.iter_mut().take(P::T).enumerate() {
        let at = P::IRR_OFFSET + i * 2;
        *slot = u16::from_le_bytes([sk[at], sk[at + 1]]) & P::Field::MASK;
    }
    goppa[P::T] = 1;
}

/// The control bits encoding the field ordering.
fn control_bits<P: Params>(sk: &[u8]) -> &[u8] {
    &sk[P::COND_OFFSET..P::COND_OFFSET + P::COND_BYTES]
}

/// One bit per lane, set when the bit-sliced field element in that lane is nonzero.
fn nonzero_lanes(value: &Slice, planes: usize) -> Word {
    value[..planes]
        .iter()
        .copied()
        .fold(0, |nonzero, plane| nonzero | plane)
}

/// The position weighting `1 / g(alpha)^2` for every field element, bit-sliced.
///
/// `g` is irreducible of degree at least two, so it has no root anywhere in `F_q` and every
/// inverse below is well defined. Inverting in bit-sliced form costs four multiplications and a
/// dozen squarings per group of lanes, which beats a scalar batched inversion here because
/// squaring in this representation is only exclusive-ors.
fn scaling<P: Params>(
    out: &mut [Word],
    prefixes: &mut [Word],
    goppa: &[u16],
    tables: &Tables<P::Field>,
    workspace: &mut TransformWorkspace,
) {
    debug_assert_eq!(out.len(), sliced_words::<P>());
    debug_assert_eq!(prefixes.len(), sliced_words::<P>());

    eval_all_bitrev_sliced::<P>(out, goppa, tables, workspace);

    let planes = P::M;
    let groups = P::Q / LANES;

    let read = |src: &[Word], group: usize| -> Slice {
        let mut value: Slice = [0; MAX_BITS];
        value[..planes].copy_from_slice(&src[group * planes..group * planes + planes]);
        value
    };

    // Montgomery's trick: invert the product of every group once, then walk back to recover
    // the individual inverses with multiplications. An inversion costs about eleven
    // multiplications here, so trading sixty-four of them for one plus three per group is a
    // large saving. Generated keys have no zero evaluations because `g` is irreducible, but
    // imported private keys are only length checked. Replacing zero lanes with one in the
    // running product and masking their outputs preserves the field's `inv(0) = 0` semantics
    // for those malformed keys without introducing secret-dependent control flow.
    let mut running: Slice = [0; MAX_BITS];

    for group in 0..groups {
        let evaluated = read(out, group);
        let mut squared: Slice = [0; MAX_BITS];
        tables.sq(&mut squared, &evaluated);
        let mut nonzero = 0;
        for (slot, &plane) in out[group * planes..group * planes + planes]
            .iter_mut()
            .zip(squared.iter())
        {
            *slot = plane;
            nonzero |= plane;
        }

        let mut factor = squared;
        factor[0] |= !nonzero;
        if group == 0 {
            running = factor;
        } else {
            let mut product: Slice = [0; MAX_BITS];
            tables.mul(&mut product, &running, &factor);
            running = product;
        }
        prefixes[group * planes..group * planes + planes].copy_from_slice(&running[..planes]);
    }

    let mut acc: Slice = [0; MAX_BITS];
    tables.inv(&mut acc, &running);

    // `acc` is the inverse of the product of groups `0..=group`, so multiplying it by the
    // product of groups `0..group` leaves the inverse of `group` alone.
    for group in (1..groups).rev() {
        let squared = read(out, group);
        let prefix = read(prefixes, group - 1);
        let nonzero = nonzero_lanes(&squared, planes);

        let mut inverted: Slice = [0; MAX_BITS];
        tables.mul(&mut inverted, &acc, &prefix);
        for (slot, &plane) in out[group * planes..group * planes + planes]
            .iter_mut()
            .zip(inverted.iter())
        {
            *slot = plane & nonzero;
        }

        let mut factor = squared;
        factor[0] |= !nonzero;
        let mut next: Slice = [0; MAX_BITS];
        tables.mul(&mut next, &acc, &factor);
        acc = next;
    }
    let first_nonzero = out[..planes]
        .iter()
        .copied()
        .fold(0, |nonzero, plane| nonzero | plane);
    for (slot, &plane) in out[..planes].iter_mut().zip(acc.iter()) {
        *slot = plane & first_nonzero;
    }
}

/// Compute the length-`2t` syndrome from a bit-reversed-order received word.
///
/// Entry `j` is `sum_k scale[k] * bits[k] * bitrev(k)^j`, which is the transpose of multipoint
/// evaluation and so comes straight out of the transposed transform.
///
/// `weighted` is caller-supplied scratch, since decoding needs two syndromes.
fn syndrome<P: Params>(
    out: &mut [u16],
    scale: &[Word],
    bits: &[u8],
    weighted: &mut [Word],
    tables: &Tables<P::Field>,
    workspace: &mut TransformWorkspace,
) {
    debug_assert_eq!(out.len(), 2 * P::T);
    debug_assert_eq!(scale.len(), sliced_words::<P>());
    debug_assert_eq!(weighted.len(), sliced_words::<P>());

    // Selecting by the received bit is a mask over whole lanes: the bit vector already carries
    // one bit per element, which is one bit per lane.
    let planes = P::M;
    for group in 0..P::Q / LANES {
        let byte = group * (LANES / 8);
        let mut select: Word = 0;
        for i in 0..LANES / 8 {
            select |= Word::from(bits[byte + i]) << (i * 8);
        }
        for i in 0..planes {
            weighted[group * planes + i] = scale[group * planes + i] & select;
        }
    }

    eval_transpose_bitrev_sliced::<P>(out, weighted, tables, workspace);
}

/// Shift a bit-sliced coefficient window right and insert `element` at its high end.
fn shift_coefficients<P: Params>(values: &mut [Slice; 2], group_count: usize, element: u16) {
    if group_count == 1 {
        for (plane, value) in values[0].iter_mut().take(P::M).enumerate() {
            let inserted = Word::from((element >> plane) & 1) << (LANES - 1);
            *value = (*value >> 1) | inserted;
        }
    } else {
        let (low, high) = values.split_at_mut(1);
        for (plane, (low, high)) in low[0]
            .iter_mut()
            .zip(high[0].iter_mut())
            .take(P::M)
            .enumerate()
        {
            let inserted = Word::from((element >> plane) & 1) << (LANES - 1);
            *low = (*low >> 1) | (*high << (LANES - 1));
            *high = (*high >> 1) | inserted;
        }
    }
}

/// Degree-64 Berlekamp--Massey in one native 64-bit lane.
fn berlekamp_massey_64<P: Params>(out: &mut [u16], syndrome: &[u16], tables: &Tables<P::Field>) {
    const LANES_64: usize = u64::BITS as usize;

    debug_assert_eq!(P::T, LANES_64);
    debug_assert_eq!(out.len(), LANES_64 + 1);
    debug_assert_eq!(syndrome.len(), 2 * LANES_64);

    let mut current: Slice64 = [0; MAX_BITS];
    let mut previous: Slice64 = [0; MAX_BITS];
    let mut interval: Slice64 = [0; MAX_BITS];
    previous[0] = 1 << (LANES_64 - 1);

    let mut length = 0u16;
    let mut base = 1u16;
    let mut constant = 1u16;

    for (n, &sample) in syndrome.iter().enumerate() {
        let mut product = [0u64; MAX_BITS];
        tables.mul64(&mut product, &current, &interval);
        for (plane, value) in interval.iter_mut().take(P::M).enumerate() {
            let inserted = u64::from((sample >> plane) & 1) << (LANES_64 - 1);
            *value = (*value >> 1) | inserted;
        }

        let mut discrepancy = P::Field::mul(constant, sample);
        for plane in (0..P::M).rev() {
            discrepancy ^= ((product[plane].count_ones() & 1) as u16) << plane;
        }

        let next_constant = P::Field::mul(constant, base);
        let nonzero = discrepancy.wrapping_sub(1).wrapping_shr(15).wrapping_sub(1);
        let grow = (n as u16)
            .wrapping_sub(length.wrapping_mul(2))
            .wrapping_shr(15)
            .wrapping_sub(1)
            & nonzero;

        let mut discrepancy_broadcast = [0u64; MAX_BITS];
        let mut base_broadcast = [0u64; MAX_BITS];
        for plane in 0..P::M {
            discrepancy_broadcast[plane] = 0u64.wrapping_sub(u64::from((discrepancy >> plane) & 1));
            base_broadcast[plane] = 0u64.wrapping_sub(u64::from((base >> plane) & 1));
        }

        let mut scaled_previous = [0u64; MAX_BITS];
        let mut scaled_current = [0u64; MAX_BITS];
        tables.mul64(&mut scaled_previous, &discrepancy_broadcast, &previous);
        tables.mul64(&mut scaled_current, &base_broadcast, &current);

        let select = 0u64.wrapping_sub(u64::from(grow & 1));
        for plane in 0..P::M {
            previous[plane] = (previous[plane] & !select) | (current[plane] & select);
            current[plane] = scaled_previous[plane] ^ scaled_current[plane];
            let inserted = u64::from(((constant & grow) >> plane) & 1) << (LANES_64 - 1);
            previous[plane] = (previous[plane] >> 1) | inserted;
        }

        constant = next_constant;
        length = (length & !grow) | (((n as u16).wrapping_add(1).wrapping_sub(length)) & grow);
        base = (base & !grow) | (discrepancy & grow);
    }

    let inverse = P::Field::inv(constant);
    let mut inverse_broadcast = [0u64; MAX_BITS];
    for (plane, slot) in inverse_broadcast.iter_mut().take(P::M).enumerate() {
        *slot = 0u64.wrapping_sub(u64::from((inverse >> plane) & 1));
    }
    let mut normalized = [0u64; MAX_BITS];
    tables.mul64(&mut normalized, &current, &inverse_broadcast);

    for (coefficient, slot) in out[..LANES_64].iter_mut().zip(0..LANES_64) {
        let mut value = 0u16;
        for plane in (0..P::M).rev() {
            value = (value << 1) | (((normalized[plane] >> slot) & 1) as u16);
        }
        *coefficient = value;
    }
    out[LANES_64] = 1;
}

/// Degree-128 Berlekamp--Massey with the monic coefficient kept separately.
///
/// A 129-coefficient bit-sliced window would need two 128-lane groups even though the second
/// group carries only one useful coefficient. Keeping that coefficient in `constant` lets the
/// whole polynomial body advance in one group, matching the representation used by the
/// official optimized implementation.
fn berlekamp_massey_128<P: Params>(out: &mut [u16], syndrome: &[u16], tables: &Tables<P::Field>) {
    debug_assert_eq!(P::T, LANES);
    debug_assert_eq!(out.len(), LANES + 1);
    debug_assert_eq!(syndrome.len(), 2 * LANES);

    let mut current = [[0; MAX_BITS]; 2];
    let mut previous = [[0; MAX_BITS]; 2];
    let mut interval = [[0; MAX_BITS]; 2];
    previous[0][0] = 1 << (LANES - 1);

    let mut length = 0u16;
    let mut base = 1u16;
    let mut constant = 1u16;

    for (n, &sample) in syndrome.iter().enumerate() {
        let mut product = [0; MAX_BITS];
        tables.mul(&mut product, &current[0], &interval[0]);
        shift_coefficients::<P>(&mut interval, 1, sample);

        let mut discrepancy = P::Field::mul(constant, sample);
        for plane in (0..P::M).rev() {
            discrepancy ^= ((product[plane].count_ones() & 1) as u16) << plane;
        }

        let next_constant = P::Field::mul(constant, base);
        let nonzero = discrepancy.wrapping_sub(1).wrapping_shr(15).wrapping_sub(1);
        let grow = (n as u16)
            .wrapping_sub(length.wrapping_mul(2))
            .wrapping_shr(15)
            .wrapping_sub(1)
            & nonzero;

        let mut discrepancy_broadcast = [0; MAX_BITS];
        let mut base_broadcast = [0; MAX_BITS];
        for plane in 0..P::M {
            discrepancy_broadcast[plane] =
                0u128.wrapping_sub(Word::from((discrepancy >> plane) & 1));
            base_broadcast[plane] = 0u128.wrapping_sub(Word::from((base >> plane) & 1));
        }

        let mut scaled_previous = [0; MAX_BITS];
        let mut scaled_current = [0; MAX_BITS];
        tables.mul(&mut scaled_previous, &discrepancy_broadcast, &previous[0]);
        tables.mul(&mut scaled_current, &base_broadcast, &current[0]);

        let select = 0u128.wrapping_sub(Word::from(grow & 1));
        for plane in 0..P::M {
            previous[0][plane] = (previous[0][plane] & !select) | (current[0][plane] & select);
            current[0][plane] = scaled_previous[plane] ^ scaled_current[plane];
        }
        shift_coefficients::<P>(&mut previous, 1, constant & grow);

        constant = next_constant;
        length = (length & !grow) | (((n as u16).wrapping_add(1).wrapping_sub(length)) & grow);
        base = (base & !grow) | (discrepancy & grow);
    }

    let inverse = P::Field::inv(constant);
    let mut inverse_broadcast = [0; MAX_BITS];
    for (plane, slot) in inverse_broadcast.iter_mut().take(P::M).enumerate() {
        *slot = 0u128.wrapping_sub(Word::from((inverse >> plane) & 1));
    }
    let mut normalized = [0; MAX_BITS];
    tables.mul(&mut normalized, &current[0], &inverse_broadcast);

    for (coefficient, slot) in out[..LANES].iter_mut().zip(0..LANES) {
        let mut value = 0u16;
        for plane in (0..P::M).rev() {
            value = (value << 1) | (((normalized[plane] >> slot) & 1) as u16);
        }
        *coefficient = value;
    }
    out[LANES] = 1;
}

/// Berlekamp-Massey: recover the error locator polynomial from a syndrome.
///
/// This is the inversion-free, bit-sliced formulation used by the official optimized
/// implementation. Up to [`LANES`] polynomial coefficients advance through each word in
/// parallel. The returned polynomial may carry a nonzero scalar factor, which does not change
/// its roots. All masks, loop bounds and indices are independent of the syndrome.
fn berlekamp_massey<P: Params>(out: &mut [u16], syndrome: &[u16], tables: &Tables<P::Field>) {
    let t = P::T;
    debug_assert_eq!(out.len(), t + 1);
    debug_assert_eq!(syndrome.len(), 2 * t);
    debug_assert!(t <= LANES);

    if t == u64::BITS as usize {
        berlekamp_massey_64::<P>(out, syndrome, tables);
        return;
    }
    if t == LANES {
        berlekamp_massey_128::<P>(out, syndrome, tables);
        return;
    }

    let group_count = (t + 1).div_ceil(LANES);
    let capacity = group_count * LANES;
    let mut current = [[0; MAX_BITS]; 2];
    let mut previous = [[0; MAX_BITS]; 2];
    let mut interval = [[0; MAX_BITS]; 2];
    current[group_count - 1][0] = 1 << (LANES - 1);
    previous[group_count - 1][0] = 1 << (LANES - 2);

    let mut length = 0u16;
    let mut base = 1u16;

    for (n, &sample) in syndrome.iter().enumerate() {
        shift_coefficients::<P>(&mut interval, group_count, sample);

        let mut products = [[0; MAX_BITS]; 2];
        for group in 0..group_count {
            tables.mul(&mut products[group], &current[group], &interval[group]);
        }

        let mut discrepancy = 0u16;
        for plane in (0..P::M).rev() {
            let mut parity = 0u32;
            for product in products.iter().take(group_count) {
                parity ^= product[plane].count_ones();
            }
            discrepancy = (discrepancy << 1) | ((parity & 1) as u16);
        }

        // All ones when the discrepancy is nonzero.
        let nonzero = discrepancy.wrapping_sub(1).wrapping_shr(15).wrapping_sub(1);
        // All ones when `n >= 2 * length`, i.e. when the linear complexity must grow.
        let grow = (n as u16)
            .wrapping_sub(length.wrapping_mul(2))
            .wrapping_shr(15)
            .wrapping_sub(1)
            & nonzero;

        let mut discrepancy_broadcast = [0; MAX_BITS];
        let mut base_broadcast = [0; MAX_BITS];
        for plane in 0..P::M {
            discrepancy_broadcast[plane] =
                0u128.wrapping_sub(Word::from((discrepancy >> plane) & 1));
            base_broadcast[plane] = 0u128.wrapping_sub(Word::from((base >> plane) & 1));
        }

        let select = 0u128.wrapping_sub(Word::from(grow & 1));
        let mut next = [[0; MAX_BITS]; 2];
        for group in 0..group_count {
            let mut scaled_previous = [0; MAX_BITS];
            let mut scaled_current = [0; MAX_BITS];
            tables.mul(
                &mut scaled_previous,
                &discrepancy_broadcast,
                &previous[group],
            );
            tables.mul(&mut scaled_current, &base_broadcast, &current[group]);
            for plane in 0..P::M {
                previous[group][plane] =
                    (previous[group][plane] & !select) | (current[group][plane] & select);
                next[group][plane] = scaled_previous[plane] ^ scaled_current[plane];
            }
        }
        shift_coefficients::<P>(&mut previous, group_count, 0);
        current = next;

        length = (length & !grow) | (((n as u16).wrapping_add(1).wrapping_sub(length)) & grow);
        base = (base & !grow) | (discrepancy & grow);
    }

    let shift = capacity - (t + 1);
    for (i, slot) in out.iter_mut().enumerate() {
        let bit = i + shift;
        let group = bit / LANES;
        let lane = bit % LANES;
        let mut coefficient = 0u16;
        for plane in (0..P::M).rev() {
            coefficient = (coefficient << 1) | (((current[group][plane] >> lane) & 1) as u16);
        }
        *slot = coefficient;
    }
}

/// Scalar oracle for the bit-sliced formulation above.
#[cfg(test)]
fn berlekamp_massey_scalar<P: Params>(out: &mut [u16], syndrome: &[u16]) {
    let t = P::T;
    let mut current = vec![0u16; t + 1];
    let mut previous = vec![0u16; t + 1];
    let mut length = 0u16;
    let mut base = 1u16;

    current[0] = 1;
    previous[1] = 1;

    for n in 0..2 * t {
        let mut discrepancy = 0u16;
        for (&coefficient, &sample) in current.iter().zip(syndrome[..=n].iter().rev()) {
            discrepancy = add(discrepancy, P::Field::mul(coefficient, sample));
        }

        let nonzero = discrepancy.wrapping_sub(1).wrapping_shr(15).wrapping_sub(1);
        let grow = (n as u16)
            .wrapping_sub(length.wrapping_mul(2))
            .wrapping_shr(15)
            .wrapping_sub(1)
            & nonzero;

        let factor = P::Field::frac(base, discrepancy);
        for i in 0..=t {
            let saved = current[i];
            current[i] ^= P::Field::mul(factor, previous[i]) & nonzero;
            previous[i] = (previous[i] & !grow) | (saved & grow);
        }

        length = (length & !grow) | (((n as u16).wrapping_add(1).wrapping_sub(length)) & grow);
        base = (base & !grow) | (discrepancy & grow);

        for i in (1..=t).rev() {
            previous[i] = previous[i - 1];
        }
        previous[0] = 0;
    }

    for (i, slot) in out.iter_mut().enumerate() {
        *slot = current[t - i];
    }
}

/// The bit-sliced word type [`prepare`] writes, named so callers outside this module can
/// hold the material without reaching into the bit-sliced arithmetic layer.
pub(crate) type PreparedWord = Word;

/// The number of bit-sliced words [`prepare`] writes to its `scale` argument.
pub(crate) fn scale_words<P: Params>() -> usize {
    sliced_words::<P>()
}

/// The number of bytes [`prepare`] writes to its `valid` argument.
pub(crate) fn valid_bytes<P: Params>() -> usize {
    P::Q / 8
}

/// The number of words [`prepare`] writes to its `conds` argument.
pub(crate) fn cond_words<P: Params>() -> usize {
    prepared_cond_words::<P>()
}

/// Precompute the parts of decoding that depend only on the private key.
///
/// `scale` receives `1 / g(a)^2` for every field element, bit-sliced and in field-element
/// order, `valid` the mask of elements that carry a real code position, and `conds` the
/// network's control words in prepared per-stage form. None of them reads the ciphertext, so a
/// holder that decapsulates repeatedly under one key can do this once instead of once per
/// message. It is roughly a third of the cost of a decapsulation.
pub(crate) fn prepare<P: Params>(
    sk: &[u8],
    scale: &mut [Word],
    valid: &mut [u8],
    conds: &mut [u64],
) {
    debug_assert_eq!(sk.len(), P::SECRET_KEY_LENGTH);
    debug_assert_eq!(scale.len(), scale_words::<P>());
    debug_assert_eq!(valid.len(), valid_bytes::<P>());
    debug_assert_eq!(conds.len(), cond_words::<P>());

    let cond = control_bits::<P>(sk);
    prepare_conds::<P>(conds, cond);

    // Which field elements carry a real code position. Elements beyond the code length must
    // not contribute to the re-encryption syndrome even if the locator happens to vanish there.
    valid.fill(0);
    valid[..P::N_BYTES].fill(0xFF);
    apply_benes_prepared::<P>(valid, cond, conds, true);

    let mut goppa = vec![0u16; P::T + 1];
    load_goppa::<P>(sk, &mut goppa);

    let tables = Tables::<<P as Params>::Field>::new();
    let mut transform = TransformWorkspace::new::<P>();
    let mut weighted = vec![0; sliced_words::<P>()];
    scaling::<P>(scale, &mut weighted, &goppa, &tables, &mut transform);

    use zeroize::Zeroize;
    goppa.zeroize();
    weighted.zeroize();
}

/// `Decode`: recover the weight-`t` error vector `e` from the syndrome `C0`.
///
/// Returns an all-ones mask when decoding succeeded and an all-zeros mask otherwise. A mask
/// rather than a `bool` keeps the caller's selection of `e` against the rejection string `s`
/// free of any branch on secret data. On failure `e` holds an unspecified value that the
/// caller must discard in favor of `s`.
///
/// The work happens in field-element order rather than code-position order. Support element
/// `alpha_i` is `bitrev` of the permutation image of position `i`, so moving the received word
/// through the inverse Beneš network lines positions up against field elements without a single
/// secret-dependent index. That is what lets the additive FFT evaluate the Goppa polynomial and
/// the error locator at every element at once, instead of one support point at a time.
///
/// This form takes the key-derived material from [`prepare`] rather than deriving it.
pub(crate) fn decode_prepared<P: Params>(
    e: &mut [u8],
    sk: &[u8],
    c0: &[u8],
    scale: &[Word],
    valid: &[u8],
    conds: &[u64],
) -> u8 {
    debug_assert_eq!(e.len(), P::N_BYTES);
    debug_assert_eq!(c0.len(), P::SYND_BYTES);
    debug_assert_eq!(scale.len(), scale_words::<P>());
    debug_assert_eq!(valid.len(), valid_bytes::<P>());
    debug_assert_eq!(conds.len(), cond_words::<P>());

    let cond = control_bits::<P>(sk);
    let plane = P::Q / 8;

    // The received word: `C0` extended with zeros, then moved into field-element order.
    let mut received = vec![0u8; plane];
    received[..P::SYND_BYTES].copy_from_slice(c0);
    apply_benes_prepared::<P>(&mut received, cond, conds, true);

    // The weighting depends only on the key, so both syndromes below share it.
    let tables = Tables::<<P as Params>::Field>::new();
    let mut transform = TransformWorkspace::new::<P>();
    let mut weighted = vec![0; sliced_words::<P>()];

    let mut s = vec![0u16; 2 * P::T];
    syndrome::<P>(
        &mut s,
        scale,
        &received,
        &mut weighted,
        &tables,
        &mut transform,
    );

    let mut locator = vec![0u16; P::T + 1];
    berlekamp_massey::<P>(&mut locator, &s, &tables);

    let mut images = vec![0; sliced_words::<P>()];
    eval_all_bitrev_sliced::<P>(&mut images, &locator, &tables, &mut transform);

    // A locator root is a lane whose every bit-plane is zero, so the whole root search is an
    // or-reduction across the planes rather than a test per element.
    let planes = P::M;
    let mut error = vec![0u8; plane];
    let mut weight = 0u16;
    for group in 0..P::Q / LANES {
        let mut any: Word = 0;
        for i in 0..planes {
            any |= images[group * planes + i];
        }

        let byte = group * (LANES / 8);
        let mut allowed: Word = 0;
        for i in 0..LANES / 8 {
            allowed |= Word::from(valid[byte + i]) << (i * 8);
        }

        let roots = !any & allowed;
        weight += roots.count_ones() as u16;
        for i in 0..LANES / 8 {
            error[byte + i] = (roots >> (i * 8)) as u8;
        }
    }

    // The decoder is only trusted when the recovered `e` has weight exactly `t` and reproduces
    // the syndrome, which is the `wt(e) = t and C = He` condition of the specification.
    let mut recomputed = vec![0u16; 2 * P::T];
    syndrome::<P>(
        &mut recomputed,
        scale,
        &error,
        &mut weighted,
        &tables,
        &mut transform,
    );

    let mut check = weight ^ (P::T as u16);
    for (a, b) in s.iter().zip(recomputed.iter()) {
        check |= a ^ b;
    }

    // Back to code-position order.
    apply_benes_prepared::<P>(&mut error, cond, conds, false);
    e.copy_from_slice(&error[..P::N_BYTES]);

    use zeroize::Zeroize;
    weighted.zeroize();
    received.zeroize();
    s.zeroize();
    recomputed.zeroize();
    locator.zeroize();
    images.zeroize();
    error.zeroize();

    // `check == 0` exactly when decoding succeeded.
    0u8.wrapping_sub((check.wrapping_sub(1) >> 15) as u8)
}

/// `Decode`, deriving the key material it needs up front.
///
/// See [`decode_prepared`] for the algorithm and for reusing that material across messages.
/// Decapsulation derives the material at its own level, so only the tests want this shape,
/// and those tests generate a key pair and a ciphertext, so they need both of those
/// operations compiled in. This gate must track the test module's.
#[cfg(all(test, feature = "keygen", feature = "encapsulate"))]
pub(crate) fn decode<P: Params>(e: &mut [u8], sk: &[u8], c0: &[u8]) -> u8 {
    let mut scale = vec![0; scale_words::<P>()];
    let mut valid = vec![0u8; valid_bytes::<P>()];
    let mut conds = vec![0u64; cond_words::<P>()];
    prepare::<P>(sk, &mut scale, &mut valid, &mut conds);

    let keep = decode_prepared::<P>(e, sk, c0, &scale, &valid, &conds);

    use zeroize::Zeroize;
    scale.zeroize();
    valid.zeroize();
    conds.zeroize();

    keep
}

/// Whether every padding bit of a ciphertext is zero.
///
/// Ciphertexts have padding only when `mt` is not a multiple of eight, which among the
/// standardized parameter sets is true for `mceliece6960119` alone.
pub(crate) fn ciphertext_padding_is_zero<P: Params>(ciphertext: &[u8]) -> bool {
    if !P::CIPHERTEXT_HAS_PADDING {
        return true;
    }
    ciphertext[P::SYND_BYTES - 1] >> (P::PK_NROWS % 8) == 0
}

/// `Decap`: derive the session key for `ciphertext` under the private key `sk`.
///
/// Decoding failure is not reported to the caller. Instead the session key is derived from the
/// private key's rejection string `s`, so a malformed ciphertext yields a key that is
/// unpredictable to the sender but perfectly reproducible by this holder of `sk`. Returns
/// `false` only for the separate, public condition of nonzero ciphertext padding, and then
/// fills the session key with ones exactly as the reference implementation does.
pub(crate) fn decapsulate_prepared<P: Params>(
    session_key: &mut [u8],
    ciphertext: &[u8],
    sk: &[u8],
    scale: &[Word],
    valid: &[u8],
    conds: &[u64],
) -> bool {
    debug_assert_eq!(session_key.len(), P::SHARED_SECRET_LENGTH);
    debug_assert_eq!(ciphertext.len(), P::CIPHERTEXT_LENGTH);
    debug_assert_eq!(sk.len(), P::SECRET_KEY_LENGTH);

    let padding_ok = ciphertext_padding_is_zero::<P>(ciphertext);

    let mut decoded = vec![0u8; P::N_BYTES];
    let mut keep = decode_prepared::<P>(
        &mut decoded,
        sk,
        &ciphertext[..P::SYND_BYTES],
        scale,
        valid,
        conds,
    );

    // Implicit rejection: replace `e` with `s` when decoding failed.
    let rejection = &sk[P::S_OFFSET..P::S_OFFSET + P::N_BYTES];
    let mut e = decoded;
    e.as_mut_slice()
        .ct_assign(rejection, Choice::from_u8_lsb(!keep));

    if P::PC {
        // Plaintext confirmation: `C1` must match `Hash(2, e)`, otherwise fall back to `s`.
        let mut confirmation = [0u8; 32];
        hash_32(HASH_PLAINTEXT_CONFIRMATION, &e, &[], &mut confirmation);

        let mut differs = 0u8;
        for (a, b) in confirmation.iter().zip(ciphertext[P::SYND_BYTES..].iter()) {
            differs |= a ^ b;
        }
        let matches = 0u8.wrapping_sub(((differs as u32).wrapping_sub(1) >> 31) as u8);
        use zeroize::Zeroize;
        confirmation.zeroize();

        e.as_mut_slice()
            .ct_assign(rejection, Choice::from_u8_lsb(!matches));
        keep &= matches;
    }

    // `HASH_REJECTION` is 0 and `HASH_ENCAPSULATION` is 1, so masking selects the domain byte
    // without a branch.
    debug_assert_eq!((HASH_REJECTION, HASH_ENCAPSULATION), (0, 1));
    hash_32(HASH_ENCAPSULATION & keep, &e, ciphertext, session_key);

    use zeroize::Zeroize;
    e.zeroize();

    if !padding_ok {
        for byte in session_key.iter_mut() {
            *byte |= 0xFF;
        }
    }

    padding_ok
}

/// `Decap`, deriving the key material it needs up front.
///
/// See [`decapsulate_prepared`] for reusing that material across messages.
pub(crate) fn decapsulate<P: Params>(session_key: &mut [u8], ciphertext: &[u8], sk: &[u8]) -> bool {
    let mut scale = vec![0; scale_words::<P>()];
    let mut valid = vec![0u8; valid_bytes::<P>()];
    let mut conds = vec![0u64; cond_words::<P>()];
    prepare::<P>(sk, &mut scale, &mut valid, &mut conds);

    let ok = decapsulate_prepared::<P>(session_key, ciphertext, sk, &scale, &valid, &conds);

    use zeroize::Zeroize;
    scale.zeroize();
    valid.zeroize();
    conds.zeroize();

    ok
}

#[cfg(all(test, feature = "keygen", feature = "encapsulate"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::hazmat::encap::{encapsulate, fixed_weight};
    use crate::hazmat::keygen::seeded_keypair;
    use rand_core::SeedableRng;

    /// The two-group form of the coefficient shift carries between groups, and no standardized
    /// parameter set reaches it: `group_count` is `ceil((t + 1) / 128)`, which is one for every
    /// supported `t`. Check it directly so the branch is not merely unexecuted code.
    fn shift_carries_between_two_groups<P: Params>() {
        // Lane `i` of group `g` holds coefficient `g * LANES + i`, so the pair is one window of
        // `2 * LANES` coefficients. Shifting must move each down one place and admit `element`
        // at the top.
        // A fixed spread of distinct-looking values; the shift is not data dependent, so a
        // deterministic pattern exercises it as well as randomness would.
        let mix = |position: usize| ((position as u32).wrapping_mul(2_654_435_761) >> 9) as u16;
        let before: Vec<u16> = (0..2 * LANES).map(|p| mix(p) & P::Field::MASK).collect();
        let element = mix(4242) & P::Field::MASK;

        let mut values: [Slice; 2] = [[0; MAX_BITS]; 2];
        for (position, &coefficient) in before.iter().enumerate() {
            let (group, lane) = (position / LANES, position % LANES);
            for (plane, slot) in values[group].iter_mut().take(P::M).enumerate() {
                *slot |= Word::from((coefficient >> plane) & 1) << lane;
            }
        }

        shift_coefficients::<P>(&mut values, 2, element);

        for position in 0..2 * LANES {
            let (group, lane) = (position / LANES, position % LANES);
            let mut got = 0u16;
            for plane in (0..P::M).rev() {
                got = (got << 1) | ((values[group][plane] >> lane) & 1) as u16;
            }
            let expected = if position + 1 < 2 * LANES {
                before[position + 1]
            } else {
                element
            };
            assert_eq!(got, expected, "{} position {position}", P::NAME);
        }
    }

    /// A round trip must reproduce the sender's session key.
    fn round_trip<P: Params>(seed: u8) {
        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        let mut sk = vec![0u8; P::SECRET_KEY_LENGTH];
        seeded_keypair::<P>(&mut pk, &mut sk, &[seed; 32]);

        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed ^ 0x33; 32]);
        for _ in 0..2 {
            let mut ciphertext = vec![0u8; P::CIPHERTEXT_LENGTH];
            let mut sent = vec![0u8; P::SHARED_SECRET_LENGTH];
            assert!(encapsulate::<P>(&mut ciphertext, &mut sent, &pk, &mut rng));

            let mut received = vec![0u8; P::SHARED_SECRET_LENGTH];
            assert!(decapsulate::<P>(&mut received, &ciphertext, &sk));
            assert_eq!(sent, received, "{} round trip", P::NAME);
        }
    }

    /// A corrupted ciphertext must decapsulate to a key that is stable but different.
    fn implicit_rejection_is_deterministic<P: Params>(seed: u8) {
        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        let mut sk = vec![0u8; P::SECRET_KEY_LENGTH];
        seeded_keypair::<P>(&mut pk, &mut sk, &[seed; 32]);

        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed ^ 0x77; 32]);
        let mut ciphertext = vec![0u8; P::CIPHERTEXT_LENGTH];
        let mut sent = vec![0u8; P::SHARED_SECRET_LENGTH];
        assert!(encapsulate::<P>(&mut ciphertext, &mut sent, &pk, &mut rng));

        ciphertext[0] ^= 1;
        let mut first = vec![0u8; P::SHARED_SECRET_LENGTH];
        let mut second = vec![0u8; P::SHARED_SECRET_LENGTH];
        assert!(decapsulate::<P>(&mut first, &ciphertext, &sk));
        assert!(decapsulate::<P>(&mut second, &ciphertext, &sk));

        assert_eq!(first, second, "{} rejection is deterministic", P::NAME);
        assert_ne!(
            first,
            sent,
            "{} rejection differs from the real key",
            P::NAME
        );
    }

    /// Decoding an honestly generated syndrome must recover the exact error vector.
    fn decode_recovers_the_error_vector<P: Params>(seed: u8) {
        use crate::hazmat::encap::encode;

        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        let mut sk = vec![0u8; P::SECRET_KEY_LENGTH];
        seeded_keypair::<P>(&mut pk, &mut sk, &[seed; 32]);

        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed ^ 0x11; 32]);
        let mut e = vec![0u8; P::N_BYTES];
        fixed_weight::<P>(&mut e, &mut rng);

        let mut c0 = vec![0u8; P::SYND_BYTES];
        encode::<P>(&mut c0, &pk, &e);

        let mut recovered = vec![0u8; P::N_BYTES];
        assert_eq!(decode::<P>(&mut recovered, &sk, &c0), 0xFF);
        assert_eq!(recovered, e, "{} decode", P::NAME);
    }

    /// A syndrome that is not a valid codeword syndrome must be rejected.
    fn decode_rejects_garbage<P: Params>(seed: u8) {
        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        let mut sk = vec![0u8; P::SECRET_KEY_LENGTH];
        seeded_keypair::<P>(&mut pk, &mut sk, &[seed; 32]);

        // All ones is not the syndrome of a weight-`t` vector for any of these codes.
        let mut c0 = vec![0xFFu8; P::SYND_BYTES];
        if P::CIPHERTEXT_HAS_PADDING {
            c0[P::SYND_BYTES - 1] &= (1u8 << (P::PK_NROWS % 8)) - 1;
        }

        let mut recovered = vec![0u8; P::N_BYTES];
        assert_eq!(decode::<P>(&mut recovered, &sk, &c0), 0x00);
    }

    macro_rules! decap_tests {
        ($($feature:literal => $mod_name:ident, $params:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

                    #[test]
                    fn encapsulation_round_trips() {
                        round_trip::<$params>($seed);
                    }

                    #[test]
                    fn corrupted_ciphertexts_are_implicitly_rejected() {
                        implicit_rejection_is_deterministic::<$params>($seed);
                    }

                    #[test]
                    fn decoding_recovers_e() {
                        decode_recovers_the_error_vector::<$params>($seed);
                    }

                    #[test]
                    fn decoding_rejects_invalid_syndromes() {
                        decode_rejects_garbage::<$params>($seed);
                    }

                    #[test]
                    fn two_group_coefficient_shift() {
                        shift_carries_between_two_groups::<$params>();
                    }
                }
            )+
        };
    }

    decap_tests! {
        "mceliece348864" => set_348864, McEliece348864, 0x11;
        "mceliece348864f" => set_348864f, McEliece348864f, 0x12;
        "mceliece460896" => set_460896, McEliece460896, 0x13;
        "mceliece460896pc" => set_460896pc, McEliece460896pc, 0x14;
        "mceliece6688128f" => set_6688128f, McEliece6688128f, 0x15;
        "mceliece6688128pcf" => set_6688128pcf, McEliece6688128pcf, 0x16;
        "mceliece6960119" => set_6960119, McEliece6960119, 0x17;
        "mceliece6960119pcf" => set_6960119pcf, McEliece6960119pcf, 0x18;
        "mceliece8192128" => set_8192128, McEliece8192128, 0x19;
        "mceliece8192128pc" => set_8192128pc, McEliece8192128pc, 0x1A;
    }

    #[test]
    #[cfg(feature = "mceliece6960119")]
    fn ciphertext_padding_is_rejected() {
        use crate::hazmat::params::McEliece6960119;
        type P = McEliece6960119;

        let mut ciphertext = vec![0u8; P::CIPHERTEXT_LENGTH];
        assert!(ciphertext_padding_is_zero::<P>(&ciphertext));
        ciphertext[P::SYND_BYTES - 1] = 1 << (P::PK_NROWS % 8);
        assert!(!ciphertext_padding_is_zero::<P>(&ciphertext));
    }
}

#[cfg(all(test, feature = "mceliece8192128f"))]
mod scaling_tests {
    use super::*;
    use crate::hazmat::params::McEliece8192128f as P;

    #[test]
    fn batched_scaling_matches_direct_inversion_with_zero_evaluations() {
        let tables = Tables::<<P as Params>::Field>::new();
        let mut transform = TransformWorkspace::new::<P>();
        let planes = P::M;
        let mut goppa = vec![0u16; P::T + 1];
        goppa[1] = 1;

        let mut expected = vec![0; sliced_words::<P>()];
        eval_all_bitrev_sliced::<P>(&mut expected, &goppa, &tables, &mut transform);
        for group in 0..P::Q / LANES {
            let at = group * planes;
            let mut evaluated = [0; MAX_BITS];
            evaluated[..planes].copy_from_slice(&expected[at..at + planes]);
            let mut squared = [0; MAX_BITS];
            let mut inverted = [0; MAX_BITS];
            tables.sq(&mut squared, &evaluated);
            tables.inv(&mut inverted, &squared);
            expected[at..at + planes].copy_from_slice(&inverted[..planes]);
        }

        let mut actual = vec![0; sliced_words::<P>()];
        let mut prefixes = vec![0; sliced_words::<P>()];
        scaling::<P>(&mut actual, &mut prefixes, &goppa, &tables, &mut transform);
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod berlekamp_massey_tests {
    use super::*;

    fn matches_scalar<P: Params>(mut seed: u64) {
        let tables = Tables::<P::Field>::new();
        for _ in 0..8 {
            let syndrome: Vec<u16> = (0..2 * P::T)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed as u16) & P::Field::MASK
                })
                .collect();

            let mut expected = vec![0u16; P::T + 1];
            let mut actual = vec![0u16; P::T + 1];
            berlekamp_massey_scalar::<P>(&mut expected, &syndrome);
            berlekamp_massey::<P>(&mut actual, &syndrome, &tables);

            let pivot = expected
                .iter()
                .zip(actual.iter())
                .position(|(&a, &b)| a != 0 || b != 0);
            assert!(pivot.is_some(), "a locator is nonzero");
            let pivot = pivot.unwrap_or(0);
            assert_ne!(expected[pivot], 0);
            assert_ne!(actual[pivot], 0);
            for i in 0..=P::T {
                assert_eq!(
                    P::Field::mul(expected[i], actual[pivot]),
                    P::Field::mul(actual[i], expected[pivot]),
                    "{} coefficient {i}",
                    P::NAME
                );
            }
        }
    }

    macro_rules! bm_tests {
        ($($feature:literal => $name:ident, $params:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                #[test]
                fn $name() {
                    matches_scalar::<$params>($seed);
                }
            )+
        };
    }

    bm_tests! {
        "mceliece348864f" => degree_64, crate::hazmat::params::McEliece348864f, 0x64A5_0001;
        "mceliece460896pcf" => degree_96, crate::hazmat::params::McEliece460896pcf, 0x96A5_0002;
        "mceliece6960119f" => degree_119, crate::hazmat::params::McEliece6960119f, 0x119A_5003;
        "mceliece8192128f" => degree_128, crate::hazmat::params::McEliece8192128f, 0x128A_5004;
    }
}
