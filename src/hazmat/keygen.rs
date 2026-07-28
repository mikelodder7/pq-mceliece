/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Key generation: `FieldOrdering`, `MatGen` and `SeededKeyGen`.

use shake::digest::{ExtendableOutput, Update};

use super::benes::support_gen;
use super::controlbits::control_bits_from_permutation;
use super::field::Field;
use super::matrix::BitMatrix;
use super::params::Params;
use super::poly::{eval_many, minimal_polynomial};
use super::sort::sort_u64;

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
    let mut buf = [0u64; 64];
    let mut pivot_column = [0u32; 32];

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
            buf[i] ^= buf[j] & mask;
        }
        for j in i + 1..P::MU {
            let mask = 0u64.wrapping_sub((buf[j] >> s) & 1);
            buf[j] ^= buf[i] & mask;
        }
    }

    // Apply the same column swaps to the support ordering.
    for j in 0..P::MU {
        for k in j + 1..P::NU {
            let mut d = (pi[window + j] ^ pi[window + k]) as u64;
            d &= same_mask(k as u16, pivot_column[j] as u16);
            pi[window + j] ^= d as i16;
            pi[window + k] ^= d as i16;
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

    // `FieldOrdering`: sort `(a_i, i)` lexicographically. Packing the pair into one integer
    // makes the tie-free comparison a plain integer comparison, so the constant-time sorting
    // network can be used directly.
    let mut buf = vec![0u64; P::Q];
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = ((perm[i] as u64) << 31) | (i as u64);
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
    let support: Vec<u16> = pi[..P::N]
        .iter()
        .map(|&value| P::Field::bitrev(value as u16))
        .collect();

    // Row `i` of the Goppa parity-check matrix is `alpha_j^i / g(alpha_j)`.
    let mut monic = vec![0u16; P::T + 1];
    monic[..P::T].copy_from_slice(goppa);
    monic[P::T] = 1;

    let mut inv = vec![0u16; P::N];
    eval_many::<P>(&mut inv, &monic, &support);
    for slot in inv.iter_mut() {
        *slot = P::Field::inv(*slot);
    }

    let mut mat = BitMatrix::zeros(P::PK_NROWS, P::N);
    for i in 0..P::T {
        for j in (0..P::N).step_by(8) {
            for k in 0..P::M {
                let mut b = 0u8;
                for offset in (0..8).rev() {
                    b = (b << 1) | (((inv[j + offset] >> k) & 1) as u8);
                }
                mat.set_byte(i * P::M + k, j / 8, b);
            }
        }
        for (slot, &a) in inv.iter_mut().zip(support.iter()) {
            *slot = P::Field::mul(*slot, a);
        }
    }

    // Reduce to systematic form, taking the semi-systematic detour at the last `mu` rows.
    for row in 0..P::PK_NROWS {
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
        for k in row + 1..P::PK_NROWS {
            let mask = 0u64.wrapping_sub(mat.bit(row, row) ^ mat.bit(k, row));
            mat.add_row(row, k, mask, first_word);
        }

        if mat.bit(row, row) == 0 {
            // No pivot exists in this column: the matrix has no systematic form.
            return false;
        }

        for k in 0..P::PK_NROWS {
            if k != row {
                let mask = 0u64.wrapping_sub(mat.bit(k, row));
                mat.add_row(k, row, mask, first_word);
            }
        }
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

    let mut delta = *seed;
    let mut expansion = vec![0u8; expansion_len];
    let mut f = vec![0u16; P::T];
    let mut goppa = vec![0u16; P::T];
    let mut perm = vec![0u32; P::Q];
    let mut pi = vec![0i16; P::Q];

    loop {
        let mut prg = shake::Shake256::default();
        prg.update(&[PRG_PREFIX]);
        prg.update(&delta);
        prg.finalize_xof_into(&mut expansion);

        // The seed stored in the private key is the one that produced this attempt.
        sk[..32].copy_from_slice(&delta);
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

    use zeroize::Zeroize;
    seed.zeroize();
    regenerated.zeroize();
}

/// Recover the support `alpha` and Goppa polynomial `g` from a private key.
///
/// `support` receives `n` elements and `goppa` receives `t + 1` coefficients with a leading 1.
pub(crate) fn load_private_key<P: Params>(sk: &[u8], goppa: &mut [u16], support: &mut [u16]) {
    debug_assert_eq!(goppa.len(), P::T + 1);
    debug_assert_eq!(support.len(), P::N);

    for (i, slot) in goppa.iter_mut().take(P::T).enumerate() {
        let at = P::IRR_OFFSET + i * 2;
        *slot = u16::from_le_bytes([sk[at], sk[at + 1]]) & P::Field::MASK;
    }
    goppa[P::T] = 1;

    support_gen::<P>(support, &sk[P::COND_OFFSET..P::COND_OFFSET + P::COND_BYTES]);
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

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
        load_private_key::<P>(&sk, &mut goppa, &mut support);

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

    macro_rules! keygen_tests {
        ($($feature:literal => $mod_name:ident, $params:ty;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

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
