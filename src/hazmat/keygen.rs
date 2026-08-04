/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Key generation: `FieldOrdering`, `MatGen` and `SeededKeyGen`.

use shake::digest::{ExtendableOutput, Update};
use zeroize::{Zeroize, Zeroizing};

use super::controlbits::control_bits_from_permutation;
use super::fft::{TransformWorkspace, eval_all_bitrev_sliced, sliced_words};
use super::field::Field;
use super::matrix::BitMatrix;
use super::params::Params;
use super::poly::minimal_polynomial;
use super::sort::sort_u64;
use super::vec::{LANES, MAX_BITS, Slice, Tables, Word};
use ctutils::{Choice, CtAssign, CtSelect};

/// The domain separator prefixed to `delta` before expansion, as required by `PRG`.
const PRG_PREFIX: u8 = 64;

/// The sentinel stored in place of the column selections for non-`f` parameter sets.
const NO_PIVOTS: u64 = 0xFFFF_FFFF;

/// Count the trailing zeros of a nonzero input without branching.
fn trailing_zeros(input: u64) -> u32 {
    let mut seen = 0u32;
    let mut count = 0u32;
    for i in 0..64 {
        let bit = ((input >> i) & 1) as u32;
        seen |= bit;
        count += (seen ^ 1) & (bit ^ 1);
    }
    count
}

/// All ones when `x == y`, all zeros otherwise.
#[inline]
fn same_mask(x: u16, y: u16) -> u64 {
    let mask = ((x ^ y) as u64).wrapping_sub(1) >> 63;
    0u64.wrapping_sub(mask)
}

/// Evaluate `g` over the whole field and replace every value with its inverse.
///
/// The polynomial reaching `MatGen` is irreducible, so none of its evaluations is zero.
/// Montgomery's trick reduces one bit-sliced inversion per group to one inversion total plus
/// three multiplications per group. The output remains bit-sliced so the arithmetic and its
/// memory access pattern are independent of the polynomial.
fn inverse_evaluations<P: Params>(out: &mut [Word], goppa: &[u16]) {
    debug_assert_eq!(out.len(), sliced_words::<P>());
    debug_assert_eq!(goppa.len(), P::T);

    let mut monic = Zeroizing::new(vec![0u16; P::T + 1]);
    monic[..P::T].copy_from_slice(goppa);
    monic[P::T] = 1;

    let tables = Tables::<P::Field>::new();
    let mut workspace = TransformWorkspace::new::<P>();
    eval_all_bitrev_sliced::<P>(out, &monic, &tables, &mut workspace);

    let planes = P::M;
    let groups = P::Q / LANES;
    let mut prefixes = Zeroizing::new(vec![0; out.len()]);
    let read = |src: &[Word], group: usize| -> Slice {
        let mut value: Slice = [0; MAX_BITS];
        value[..planes].copy_from_slice(&src[group * planes..group * planes + planes]);
        value
    };

    let mut running = read(out, 0);
    prefixes[..planes].copy_from_slice(&running[..planes]);
    for group in 1..groups {
        let evaluated = read(out, group);
        let mut product: Slice = [0; MAX_BITS];
        tables.mul(&mut product, &running, &evaluated);
        running = product;
        prefixes[group * planes..group * planes + planes].copy_from_slice(&running[..planes]);
    }

    let mut acc: Slice = [0; MAX_BITS];
    tables.inv(&mut acc, &running);
    for group in (1..groups).rev() {
        let evaluated = read(out, group);
        let prefix = read(&prefixes, group - 1);

        let mut inverted: Slice = [0; MAX_BITS];
        tables.mul(&mut inverted, &acc, &prefix);
        out[group * planes..group * planes + planes].copy_from_slice(&inverted[..planes]);

        let mut next: Slice = [0; MAX_BITS];
        tables.mul(&mut next, &acc, &evaluated);
        acc = next;
    }
    out[..planes].copy_from_slice(&acc[..planes]);
}

/// Bring the last `mu` pivots into place for a semi-systematic parameter set.
///
/// Reducing `N` to `(mu, nu)`-semi-systematic form and then swapping column `i` with column
/// `c_i` is equivalent to choosing, up front, which `mu` of the `nu` candidate columns will
/// carry the final pivots, and permuting the support to match. Doing it up front is what makes
/// the `f` variants faster: it removes the restart that a singular submatrix would otherwise
/// force.
///
/// Returns `false` when the `32 x 64` window is not full rank, which makes key generation
/// retry with a fresh seed.
fn move_columns<P: Params>(mat: &mut BitMatrix, pi: &mut [i16], pivots: &mut u64) -> bool {
    let window = P::PK_NROWS - P::MU;
    let mut buf = Zeroizing::new([0u64; 64]);
    let mut pivot_column = Zeroizing::new([0u32; 32]);

    for (i, slot) in buf.iter_mut().take(P::MU).enumerate() {
        *slot = mat.read_window(window + i, window);
    }

    // Locate the pivot columns by eliminating within the window alone.
    *pivots = 0;
    for i in 0..P::MU {
        let mut t = buf[i];
        for &other in &buf[i + 1..P::MU] {
            t |= other;
        }
        if t == 0 {
            return false;
        }

        pivot_column[i] = trailing_zeros(t);
        let s = pivot_column[i] as usize;
        *pivots |= 1u64 << s;

        for j in i + 1..P::MU {
            let mask = ((buf[i] >> s) & 1).wrapping_sub(1);
            let sum = buf[i] ^ buf[j];
            buf[i].ct_assign(&sum, Choice::from_u64_lsb(mask));
        }
        for j in i + 1..P::MU {
            let mask = 0u64.wrapping_sub((buf[j] >> s) & 1);
            let sum = buf[j] ^ buf[i];
            buf[j].ct_assign(&sum, Choice::from_u64_lsb(mask));
        }
    }

    // Apply the same column swaps to the support ordering.
    for j in 0..P::MU {
        for k in j + 1..P::NU {
            let left = pi[window + j];
            let right = pi[window + k];
            let choice = Choice::from_u64_lsb(same_mask(k as u16, pivot_column[j] as u16));
            pi[window + j] = left.ct_select(&right, choice);
            pi[window + k] = right.ct_select(&left, choice);
        }
    }

    // Swap the columns themselves across every row.
    for i in 0..mat.rows() {
        let mut t = mat.read_window(i, window);
        for (j, &column) in pivot_column.iter().enumerate().take(P::MU) {
            let mut d = t >> j;
            d ^= t >> column;
            d &= 1;
            t ^= d << column;
            t ^= d << j;
        }
        mat.write_window(i, window, t);
    }

    true
}

/// Build the leading `mt x mt` block of the parity-check matrix and eliminate it, reporting
/// whether it is invertible -- which is exactly whether the full-width sweep would succeed.
///
/// The block build mirrors the full build below restricted to the first `mt` support
/// positions, on a copy of the inverse evaluations so the caller's remain untouched. `mt` is
/// not a multiple of eight for every parameter set, so the copy pads to a byte boundary with
/// zeros; the padding columns sit past every pivot and influence nothing.
fn square_block_is_invertible<P: Params>(inv: &[u16], support: &[u16]) -> bool {
    const PANEL: usize = 32;

    let padded = P::PK_NROWS.div_ceil(8) * 8;
    let mut block_inv = Zeroizing::new(vec![0u16; padded]);
    block_inv[..P::PK_NROWS].copy_from_slice(&inv[..P::PK_NROWS]);

    let mut block = BitMatrix::zeros(P::PK_NROWS, P::PK_NROWS);
    for i in 0..P::T {
        for j in (0..padded).step_by(8) {
            for k in 0..P::M {
                let mut b = 0u8;
                for offset in (0..8).rev() {
                    b = (b << 1) | (((block_inv[j + offset] >> k) & 1) as u8);
                }
                block.set_byte(i * P::M + k, j / 8, b);
            }
        }
        for (slot, &a) in block_inv.iter_mut().zip(support.iter()) {
            *slot = P::Field::mul(*slot, a);
        }
    }

    // The same forward sweep `mat_gen` runs, on the same schedule, minus the trailing block.
    let mut panelled = 0;
    {
        let mut shadow = Zeroizing::new(vec![0u64; block.rows()]);
        let mut scratch =
            crate::hazmat::matrix::PanelScratch::new(block.rows(), block.stride(), PANEL);
        while panelled + PANEL <= P::PK_NROWS {
            if !block.eliminate_panel::<PANEL>(panelled, &mut shadow, &mut scratch) {
                return false;
            }
            panelled += PANEL;
        }
    }
    for row in panelled..P::PK_NROWS {
        let first_word = row / 64;
        block.accumulate_pivot(row, row, first_word);
        if block.bit(row, row) == 0 {
            return false;
        }
        block.clear_below(row, row, first_word);
    }

    true
}

/// `MatGen`: build the parity-check matrix for `(g, alpha)` and reduce it to systematic form.
///
/// On success `pk` holds the `mt x k` matrix `T`, `pi` holds the (possibly column-swapped)
/// permutation defining the support, and `pivots` holds the column selections `c`.
///
/// Returns `false` for the `⊥` outcome, which makes key generation retry with a fresh seed.
fn mat_gen<P: Params>(
    pk: &mut [u8],
    goppa: &[u16],
    perm: &[u32],
    pi: &mut [i16],
    pivots: &mut u64,
) -> bool {
    debug_assert_eq!(pk.len(), P::PUBLIC_KEY_LENGTH);
    debug_assert_eq!(goppa.len(), P::T);
    debug_assert_eq!(perm.len(), P::Q);
    debug_assert_eq!(pi.len(), P::Q);

    // Evaluate `1 / g(bitrev(i))` in bit-sliced form first, then attach each inverse to the
    // corresponding `(a_i, i)` field-ordering entry. The permutation occupies the high bits,
    // so the same integer sorting network both orders the support and carries the inverse
    // evaluations into matching positions.
    let mut evaluations = Zeroizing::new(vec![0; sliced_words::<P>()]);
    inverse_evaluations::<P>(&mut evaluations, goppa);

    let mut buf = Zeroizing::new(vec![0u64; P::Q]);
    for (i, slot) in buf.iter_mut().enumerate() {
        let group = i / LANES;
        let lane = i % LANES;
        let mut inverse = 0u64;
        for plane in 0..P::M {
            inverse |= (((evaluations[group * P::M + plane] >> lane) & 1) as u64) << plane;
        }
        *slot = ((perm[i] as u64) << 31) | (inverse << 16) | (i as u64);
    }
    sort_u64(&mut buf);

    for i in 1..P::Q {
        if buf[i - 1] >> 31 == buf[i] >> 31 {
            // The `a_i` are not distinct.
            return false;
        }
    }
    for (i, slot) in pi.iter_mut().enumerate() {
        *slot = (buf[i] as u16 & P::Field::MASK) as i16;
    }

    // `alpha_i = bitrev(pi(i))`, using only the first `n` of the `q` support elements.
    let support = Zeroizing::new(
        pi[..P::N]
            .iter()
            .map(|&value| P::Field::bitrev(value as u16))
            .collect::<Vec<u16>>(),
    );

    // Row `i` of the Goppa parity-check matrix is `alpha_j^i / g(alpha_j)`.
    let mut inv = Zeroizing::new(vec![0u16; P::N]);
    for (slot, &packed) in inv.iter_mut().zip(buf.iter()) {
        *slot = ((packed >> 16) as u16) & P::Field::MASK;
    }

    // For the non-`f` sets, forward elimination succeeds with probability around 0.29, and a
    // failure is only discovered near the last pivot columns -- after the whole `n`-wide
    // matrix has been built and swept, ~85% of which updates the `k`-wide block that the
    // failed attempt then throws away. Whether column `j` can supply a pivot depends only on
    // column `j` as transformed by the pivots before it, all of which lie in the first `mt`
    // columns, so eliminating the square `mt x mt` block alone is an exact predictor of the
    // full sweep's outcome. Screen on that block first and pay the full-width build only for
    // seeds that will succeed. The `f` sets skip this: they succeed almost always, and their
    // column-moving detour straddles the block boundary.
    if !P::SEMI_SYSTEMATIC && !square_block_is_invertible::<P>(&inv, &support) {
        return false;
    }

    // Row `i * m + k` of the matrix is bit-plane `k` of `alpha_j^i / g(alpha_j)`, so building
    // the powers bit-sliced makes every row a straight copy of plane words and turns the
    // per-element Horner step into one bit-sliced multiply per group of lanes. Lanes past
    // `n` stay zero through every multiply, which keeps the rows' tail bits zero exactly as
    // the byte-at-a-time build left them.
    let mut mat = BitMatrix::zeros(P::PK_NROWS, P::N);
    let planes = P::M;
    let groups = P::N.div_ceil(LANES);
    let mut inv_sliced: Zeroizing<Vec<Word>> = Zeroizing::new(vec![0; groups * planes]);
    let mut support_sliced: Zeroizing<Vec<Word>> = Zeroizing::new(vec![0; groups * planes]);
    for (j, (&value, &alpha)) in inv.iter().zip(support.iter()).enumerate() {
        let group = j / LANES;
        let lane = j % LANES;
        for plane in 0..planes {
            inv_sliced[group * planes + plane] |= Word::from((value >> plane) & 1) << lane;
            support_sliced[group * planes + plane] |= Word::from((alpha >> plane) & 1) << lane;
        }
    }

    let tables = Tables::<P::Field>::new();
    let read = |src: &[Word], group: usize| -> Slice {
        let mut value: Slice = [0; MAX_BITS];
        value[..planes].copy_from_slice(&src[group * planes..group * planes + planes]);
        value
    };
    // Rows carry one padding word past the columns; it stays zero, so only the data words
    // are written.
    let data_words = P::N.div_ceil(64);
    for i in 0..P::T {
        for k in 0..planes {
            let row = mat.row_words_mut(i * planes + k);
            for (word, slot) in row[..data_words].iter_mut().enumerate() {
                let plane = inv_sliced[(word / 2) * planes + k];
                *slot = (plane >> (64 * (word % 2))) as u64;
            }
        }
        if i + 1 == P::T {
            break;
        }
        let mut product: Slice = [0; MAX_BITS];
        for group in 0..groups {
            let current = read(&inv_sliced, group);
            let alpha = read(&support_sliced, group);
            tables.mul(&mut product, &current, &alpha);
            inv_sliced[group * planes..group * planes + planes].copy_from_slice(&product[..planes]);
        }
    }

    // Reduce to systematic form, taking the semi-systematic detour at the last `mu` rows.
    //
    // The bulk of the forward sweep runs in blocked panels: sixteen consecutive pivots are
    // established together and then applied to every trailing row in a single streaming pass,
    // rather than streaming the whole matrix once per pivot. The rows this leaves behind are
    // bit-identical to the row-at-a-time schedule -- see `BitMatrix::eliminate_panel`.
    //
    // Panels stop short of the semi-systematic detour so that `move_columns` still sees the
    // matrix in exactly the state it expects, and the ragged tail is finished one pivot at a
    // time because a panel needs its whole width.
    const PANEL: usize = 32;
    let detour = if P::SEMI_SYSTEMATIC {
        P::PK_NROWS - P::MU
    } else {
        P::PK_NROWS
    };
    let mut panelled = 0;
    {
        let mut shadow = Zeroizing::new(vec![0u64; mat.rows()]);
        let mut scratch = crate::hazmat::matrix::PanelScratch::new(mat.rows(), mat.stride(), PANEL);
        while panelled + PANEL <= detour {
            if !mat.eliminate_panel::<PANEL>(panelled, &mut shadow, &mut scratch) {
                return false;
            }
            panelled += PANEL;
        }
    }

    for row in panelled..P::PK_NROWS {
        if P::SEMI_SYSTEMATIC
            && row == P::PK_NROWS - P::MU
            && !move_columns::<P>(&mut mat, pi, pivots)
        {
            return false;
        }

        // Columns before this pivot are already reduced: every row holds a single one in its
        // own pivot column and zeros elsewhere, so the words covering them contribute nothing
        // and can be skipped. The saving is bounded by the width of the identity block, `mt`
        // out of `n` columns, so it is worth about a tenth of the elimination on average.
        let first_word = row / 64;

        // Pull a nonzero pivot up from the rows below without branching on which one.
        mat.accumulate_pivot(row, row, first_word);

        if mat.bit(row, row) == 0 {
            // No pivot exists in this column: the matrix has no systematic form.
            return false;
        }

        // Forward elimination needs only the rows below the pivot. Deferring the rows above
        // leaves `move_columns` unchanged: its window consists of the current and later rows,
        // all of which have already had every earlier pivot cleared.
        mat.clear_below(row, row, first_word);
    }

    // Back-substitute in blocks. Once consecutive pivot rows reduce each other to an identity
    // block, all of their columns can be cleared from an earlier row in one pass. Sixteen rows
    // measured fastest for the degree-96 and degree-119 matrices; eight wins for the other
    // standardized shapes by keeping the source working set smaller.
    let mut end = P::PK_NROWS;
    if P::T == 96 || P::T == 119 {
        while end >= 16 {
            let start = end - 16;
            mat.clear_above_block::<16>(start);
            end = start;
        }
    } else {
        while end >= 8 {
            let start = end - 8;
            mat.clear_above_block::<8>(start);
            end = start;
        }
    }
    // Only the degree-119 shape leaves a short prefix (eleven rows). Processing it one pivot
    // at a time avoids a dynamic block width in either hot monomorphization.
    while end != 0 {
        let start = end - 1;
        mat.clear_above_block::<1>(start);
        end = start;
    }

    for i in 0..P::PK_NROWS {
        let row = &mut pk[i * P::PK_ROW_BYTES..(i + 1) * P::PK_ROW_BYTES];
        mat.extract_bits(i, P::PK_NROWS, P::PK_NCOLS, row);
    }

    true
}

/// `SeededKeyGen`: derive a key pair deterministically from the 32-byte seed `delta`.
///
/// Each of `Irreducible`, `FieldOrdering` and `MatGen` can reject its input; the specification
/// responds by replacing `delta` with the `delta'` drawn from the same expansion and starting
/// over, which is what the loop here does. Termination is probabilistic but overwhelmingly
/// fast: each attempt succeeds with probability around 29 percent for non-`f` parameter sets
/// and essentially 1 for `f` parameter sets.
pub(crate) fn seeded_keypair<P: Params>(pk: &mut [u8], sk: &mut [u8], seed: &[u8; 32]) {
    debug_assert_eq!(pk.len(), P::PUBLIC_KEY_LENGTH);
    debug_assert_eq!(sk.len(), P::SECRET_KEY_LENGTH);

    // `E = PRG(delta)`, laid out as `s || perm || f || delta'`.
    let s_len = P::N_BYTES;
    let perm_len = P::Q * 4;
    let f_len = P::T * 2;
    let expansion_len = s_len + perm_len + f_len + 32;

    let mut delta = Zeroizing::new(*seed);
    let mut expansion = Zeroizing::new(vec![0u8; expansion_len]);
    let mut f = Zeroizing::new(vec![0u16; P::T]);
    let mut goppa = Zeroizing::new(vec![0u16; P::T]);
    let mut perm = Zeroizing::new(vec![0u32; P::Q]);
    let mut pi = Zeroizing::new(vec![0i16; P::Q]);

    loop {
        let mut prg = shake::Shake256::default();
        prg.update(&[PRG_PREFIX]);
        prg.update(&delta[..]);
        prg.finalize_xof_into(&mut expansion);

        // The seed stored in the private key is the one that produced this attempt.
        sk[..32].copy_from_slice(&delta[..]);
        delta.copy_from_slice(&expansion[expansion_len - 32..]);

        for (i, slot) in f.iter_mut().enumerate() {
            let at = s_len + perm_len + i * 2;
            *slot = u16::from_le_bytes([expansion[at], expansion[at + 1]]) & P::Field::MASK;
        }
        if !minimal_polynomial::<P>(&mut goppa, &f) {
            continue;
        }

        for (i, slot) in perm.iter_mut().enumerate() {
            let at = s_len + i * 4;
            *slot = u32::from_le_bytes([
                expansion[at],
                expansion[at + 1],
                expansion[at + 2],
                expansion[at + 3],
            ]);
        }

        let mut pivots = NO_PIVOTS;
        if !mat_gen::<P>(pk, &goppa, &perm, &mut pi, &mut pivots) {
            continue;
        }

        if !control_bits_from_permutation(
            &mut sk[P::COND_OFFSET..P::COND_OFFSET + P::COND_BYTES],
            &pi,
            P::M,
        ) {
            continue;
        }

        for (i, &coefficient) in goppa.iter().enumerate() {
            let at = P::IRR_OFFSET + i * 2;
            sk[at..at + 2].copy_from_slice(&coefficient.to_le_bytes());
        }
        sk[P::S_OFFSET..P::S_OFFSET + s_len].copy_from_slice(&expansion[..s_len]);
        sk[32..40].copy_from_slice(&pivots.to_le_bytes());

        return;
    }
}

/// Recover the encapsulation key from a decapsulation key by rerunning `SeededKeyGen`.
///
/// Classic McEliece private keys store the seed `delta`, so the public key is recoverable, but
/// only at the cost of a full key generation. Callers that need the public key repeatedly
/// should keep it rather than call this.
pub(crate) fn public_key_from_secret_key<P: Params>(sk: &[u8], pk: &mut [u8]) {
    debug_assert_eq!(sk.len(), P::SECRET_KEY_LENGTH);
    debug_assert_eq!(pk.len(), P::PUBLIC_KEY_LENGTH);

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&sk[..32]);
    let mut regenerated = vec![0u8; P::SECRET_KEY_LENGTH];
    seeded_keypair::<P>(pk, &mut regenerated, &seed);

    seed.zeroize();
    regenerated.zeroize();
}

#[cfg(all(test, feature = "decapsulate"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::hazmat::poly::eval_many;

    #[test]
    fn trailing_zeros_matches_the_intrinsic() {
        assert_eq!(trailing_zeros(0), 64);
        for i in 0..64 {
            assert_eq!(trailing_zeros(1u64 << i), i as u32);
        }
        for v in [3u64, 12, 0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF, 180] {
            assert_eq!(trailing_zeros(v), v.trailing_zeros());
        }
    }

    #[test]
    fn same_mask_is_all_or_nothing() {
        assert_eq!(same_mask(0, 0), u64::MAX);
        assert_eq!(same_mask(1234, 1234), u64::MAX);
        assert_eq!(same_mask(0, 1), 0);
        assert_eq!(same_mask(0xFFFF, 0), 0);
    }

    /// The public key is the systematic part of a parity-check matrix for the Goppa code, so
    /// `H = (I | T)` must annihilate every codeword. Verify by checking that `H` applied to
    /// the columns reproduces the Goppa relation for the recovered support.
    fn keypair_is_internally_consistent<P: Params>(seed_byte: u8) {
        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        let mut sk = vec![0u8; P::SECRET_KEY_LENGTH];
        let seed = [seed_byte; 32];
        seeded_keypair::<P>(&mut pk, &mut sk, &seed);

        // The stored seed is the one that actually succeeded, which for non-`f` parameter
        // sets is usually a `delta'` from a later attempt. Regenerating from it must land on
        // the same key pair in a single attempt.
        let stored_seed: [u8; 32] = sk[..32].try_into().unwrap();
        let mut pk_again = vec![0u8; P::PUBLIC_KEY_LENGTH];
        let mut sk_again = vec![0u8; P::SECRET_KEY_LENGTH];
        seeded_keypair::<P>(&mut pk_again, &mut sk_again, &stored_seed);
        assert_eq!(pk, pk_again, "{} regenerated public key", P::NAME);
        assert_eq!(sk, sk_again, "{} regenerated private key", P::NAME);

        let expected_pivots = if P::SEMI_SYSTEMATIC {
            None
        } else {
            Some(NO_PIVOTS)
        };
        let stored = u64::from_le_bytes(sk[32..40].try_into().unwrap());
        if let Some(expected) = expected_pivots {
            assert_eq!(stored, expected, "{} pivot sentinel", P::NAME);
        } else {
            assert_eq!(stored.count_ones(), P::MU as u32, "{} pivot count", P::NAME);
        }

        // Regenerating from the stored seed must reproduce the same key pair exactly.
        let mut pk2 = vec![0u8; P::PUBLIC_KEY_LENGTH];
        public_key_from_secret_key::<P>(&sk, &mut pk2);
        assert_eq!(pk, pk2, "{} public key is recoverable", P::NAME);

        // The support and Goppa polynomial recovered from the private key must be usable:
        // the support elements are distinct and none of them is a root of `g`.
        let mut goppa = vec![0u16; P::T + 1];
        let mut support = vec![0u16; P::N];
        crate::hazmat::decap::load_goppa::<P>(&sk, &mut goppa);
        crate::hazmat::benes::support_gen::<P>(
            &mut support,
            &sk[P::COND_OFFSET..P::COND_OFFSET + P::COND_BYTES],
        );

        let mut seen = vec![false; P::Q];
        for &a in support.iter() {
            assert!(!seen[a as usize], "{} support repeats {a}", P::NAME);
            seen[a as usize] = true;
        }

        let mut values = vec![0u16; P::N];
        eval_many::<P>(&mut values, &goppa, &support);
        assert!(
            values.iter().all(|&v| v != 0),
            "{} support hits a root of g",
            P::NAME
        );
    }

    /// `MatGen` must report the `⊥` outcome when the semi-systematic window is rank deficient,
    /// so that key generation restarts instead of emitting a key with no systematic form. A
    /// window of zeros is the extreme case: no column can supply a pivot.
    ///
    /// Only the `f` variants take this path; the others have no window to move.
    fn move_columns_rejects_a_rank_deficient_window<P: Params>() {
        if !P::SEMI_SYSTEMATIC {
            return;
        }

        let mut mat = BitMatrix::zeros(P::PK_NROWS, P::N);
        let mut pi = vec![0i16; P::Q];
        let mut pivots = 0u64;
        assert!(
            !move_columns::<P>(&mut mat, &mut pi, &mut pivots),
            "{} all-zero window",
            P::NAME
        );

        // A window whose rows all repeat one column is equally deficient once that column is
        // consumed, and it exercises the loop rather than failing on the first iteration.
        let window = P::PK_NROWS - P::MU;
        for i in 0..P::MU {
            mat.write_window(window + i, window, 1);
        }
        assert!(
            !move_columns::<P>(&mut mat, &mut pi, &mut pivots),
            "{} repeated column",
            P::NAME
        );
    }

    macro_rules! keygen_tests {
        ($($feature:literal => $mod_name:ident, $params:ty;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

                    #[test]
                    fn rejects_rank_deficient_pivot_window() {
                        move_columns_rejects_a_rank_deficient_window::<$params>();
                    }

                    #[test]
                    fn keypair_is_consistent() {
                        keypair_is_internally_consistent::<$params>(0x5A);
                    }
                }
            )+
        };
    }

    keygen_tests! {
        "mceliece348864" => set_348864, McEliece348864;
        "mceliece348864f" => set_348864f, McEliece348864f;
        "mceliece460896" => set_460896, McEliece460896;
        "mceliece460896f" => set_460896f, McEliece460896f;
        "mceliece6960119" => set_6960119, McEliece6960119;
        "mceliece6960119f" => set_6960119f, McEliece6960119f;
        "mceliece8192128" => set_8192128, McEliece8192128;
        "mceliece8192128f" => set_8192128f, McEliece8192128f;
    }
}
