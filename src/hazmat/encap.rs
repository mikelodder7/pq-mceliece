/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Encapsulation: `FixedWeight`, `Encode` and the session-key hash.

use rand_core::CryptoRng;
use zeroize::Zeroizing;

use super::hash::{HASH_ENCAPSULATION, HASH_PLAINTEXT_CONFIRMATION, hash_32};
use super::params::Params;
use super::sort::sort_u16;

/// All ones when `x == y`, all zeros otherwise.
#[inline]
fn same_mask(x: u16, y: u16) -> u64 {
    let mask = ((x ^ y) as u32).wrapping_sub(1) >> 31;
    0u64.wrapping_sub(mask as u64)
}

/// Whether every padding bit of an encapsulation key is zero.
///
/// Public keys have padding only when `k` is not a multiple of eight, which among the
/// standardized parameter sets is true for `mceliece6960119` alone. Rejecting nonzero padding
/// is the *narrowly decoded* behavior the specification describes, and is what the reference
/// implementation does.
pub(crate) fn public_key_padding_is_zero<P: Params>(pk: &[u8]) -> bool {
    if !P::PUBLIC_KEY_HAS_PADDING {
        return true;
    }
    let mut acc = 0u8;
    for i in 0..P::PK_NROWS {
        acc |= pk[i * P::PK_ROW_BYTES + P::PK_ROW_BYTES - 1];
    }
    acc >> (P::PK_NCOLS % 8) == 0
}

/// `FixedWeight`: sample a weight-`t` error vector of length `n`.
///
/// Each attempt draws `sigma1 * tau` bits, keeps the low `m` bits of each `sigma1`-bit group,
/// and takes the first `t` values below `n`. Attempts that yield too few in-range values, or
/// that repeat an index, are discarded outright rather than patched, because any correction
/// would bias the distribution.
pub(crate) fn fixed_weight<P: Params>(e: &mut [u8], mut rng: impl CryptoRng) {
    debug_assert_eq!(e.len(), P::N_BYTES);

    let mut bytes = Zeroizing::new(vec![0u8; P::TAU * 2]);
    let mut indices = Zeroizing::new(vec![0u16; P::T]);

    loop {
        rng.fill_bytes(&mut bytes);

        let mut count = 0;
        for pair in bytes.as_chunks::<2>().0 {
            if count == P::T {
                break;
            }
            let candidate = u16::from_le_bytes(*pair) & P::GFMASK;
            if (candidate as usize) < P::N {
                indices[count] = candidate;
                count += 1;
            }
        }
        if count < P::T {
            continue;
        }

        // A data-oblivious sorting network brings duplicates together in O(t log^2 t)
        // branch-free comparisons instead of comparing all O(t^2) pairs.
        sort_u16(&mut indices[..P::T]);
        let mut repeated = 0u16;
        for i in 1..P::T {
            let equal = ((indices[i] ^ indices[i - 1]) as u32).wrapping_sub(1) >> 31;
            repeated |= equal as u16;
        }
        if repeated == 0 {
            break;
        }
    }

    // Setting the chosen bits has to touch every position for every index, or which byte an
    // index landed in would leak. Comparing a whole word at a time rather than a byte at a
    // time does that in an eighth of the comparisons.
    for (w, chunk) in e.chunks_mut(8).enumerate() {
        let mut word = 0u64;
        for &index in &indices[..P::T] {
            let bit = 1u64 << (index & 63);
            word |= bit & same_mask(w as u16, index >> 6);
        }
        let bytes = word.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
}

/// `Encode`: compute the syndrome `C0 = He` for `H = (I | T)`.
///
/// Because `H` starts with an identity block, bit `i` of the syndrome is bit `i` of `e` plus
/// the inner product of row `i` of `T` with the trailing `k` bits of `e`. Extracting those `k`
/// bits once and reducing each row with word-wide `AND` and a population count avoids
/// materializing a full-length row per output bit.
pub(crate) fn encode<P: Params>(syndrome: &mut [u8], pk: &[u8], e: &[u8]) {
    debug_assert_eq!(syndrome.len(), P::SYND_BYTES);
    debug_assert_eq!(pk.len(), P::PUBLIC_KEY_LENGTH);
    debug_assert_eq!(e.len(), P::N_BYTES);

    // The trailing `k` bits of `e`, aligned to the start of a word so it lines up with a
    // public key row. Bits past `k` in the final word stay zero.
    let words = P::PK_ROW_BYTES.div_ceil(8);
    let mut tail = Zeroizing::new(vec![0u64; words]);
    let shift = P::PK_NROWS % 8;
    let first = P::PK_NROWS / 8;
    for (w, word) in tail.iter_mut().enumerate() {
        let mut packed = 0u64;
        for b in 0..8 {
            let at = first + w * 8 + b;
            let low = if at < P::N_BYTES { e[at] } else { 0 };
            let byte = if shift == 0 {
                low
            } else {
                let high = if at + 1 < P::N_BYTES { e[at + 1] } else { 0 };
                (low >> shift) | (high << (8 - shift))
            };
            packed |= u64::from(byte) << (b * 8);
        }
        *word = packed;
    }
    let spare = words * 64 - P::PK_NCOLS;
    if spare > 0 {
        tail[words - 1] &= u64::MAX >> spare;
    }

    syndrome.copy_from_slice(&e[..P::SYND_BYTES]);
    if P::PK_NROWS % 8 != 0 {
        syndrome[P::SYND_BYTES - 1] &= (1u8 << (P::PK_NROWS % 8)) - 1;
    }
    // Indexed rather than chunked: the row length is an associated const of a type parameter,
    // which cannot appear as a const-generic argument, and the row number is wanted anyway.
    debug_assert_eq!(pk.len(), P::PK_NROWS * P::PK_ROW_BYTES);

    // The reduction streams the whole megabyte-class key once and is bound by memory, not
    // arithmetic, so what matters is how many loads are in flight. One row at a time is one
    // serial stream; four rows with independent accumulators keep four streams open, which
    // is where sustained bandwidth comes from on every out-of-order core.
    let mut i = 0;
    while i + 4 <= P::PK_NROWS {
        let (g0, rest0) = pk[i * P::PK_ROW_BYTES..(i + 1) * P::PK_ROW_BYTES].as_chunks::<8>();
        let (g1, rest1) = pk[(i + 1) * P::PK_ROW_BYTES..(i + 2) * P::PK_ROW_BYTES].as_chunks::<8>();
        let (g2, rest2) = pk[(i + 2) * P::PK_ROW_BYTES..(i + 3) * P::PK_ROW_BYTES].as_chunks::<8>();
        let (g3, rest3) = pk[(i + 3) * P::PK_ROW_BYTES..(i + 4) * P::PK_ROW_BYTES].as_chunks::<8>();

        let mut acc = [0u64; 4];
        for (w, mask) in tail.iter().take(g0.len()).enumerate() {
            acc[0] ^= u64::from_le_bytes(g0[w]) & mask;
            acc[1] ^= u64::from_le_bytes(g1[w]) & mask;
            acc[2] ^= u64::from_le_bytes(g2[w]) & mask;
            acc[3] ^= u64::from_le_bytes(g3[w]) & mask;
        }
        if !rest0.is_empty() {
            let mask = tail[words - 1];
            for (slot, rest) in acc.iter_mut().zip([rest0, rest1, rest2, rest3]) {
                let mut buf = [0u8; 8];
                buf[..rest.len()].copy_from_slice(rest);
                *slot ^= u64::from_le_bytes(buf) & mask;
            }
        }

        for (r, &value) in acc.iter().enumerate() {
            let bit = value.count_ones() as u8 & 1;
            syndrome[(i + r) / 8] ^= bit << ((i + r) % 8);
        }
        i += 4;
    }

    while i < P::PK_NROWS {
        let row = &pk[i * P::PK_ROW_BYTES..(i + 1) * P::PK_ROW_BYTES];
        // Each whole word of the row is a fixed eight-byte copy, which compiles to one load.
        // Deriving the length per word instead, as a `min` against what is left of the row,
        // turns every one of them into a variable-length copy and costs an order of magnitude.
        let mut acc = 0u64;
        let (groups, rest) = row.as_chunks::<8>();
        for (group, mask) in groups.iter().zip(tail.iter()) {
            acc ^= u64::from_le_bytes(*group) & mask;
        }

        // A row whose length is not a multiple of eight leaves a short tail, which pairs with
        // the last mask word. The length is a parameter set constant, never secret.
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            acc ^= u64::from_le_bytes(buf) & tail[words - 1];
        }

        let bit = acc.count_ones() as u8 & 1;
        syndrome[i / 8] ^= bit << (i % 8);
        i += 1;
    }
}

/// `Encap`: produce a ciphertext and session key for the encapsulation key `pk`.
///
/// Returns `false` when `pk` carries nonzero padding bits, in which case both outputs are
/// zeroed. The check and the zeroing mirror the reference implementation so that the
/// rejection path is indistinguishable in timing from the accepting one.
pub(crate) fn encapsulate<P: Params>(
    ciphertext: &mut [u8],
    session_key: &mut [u8],
    pk: &[u8],
    rng: impl CryptoRng,
) -> bool {
    debug_assert_eq!(ciphertext.len(), P::CIPHERTEXT_LENGTH);
    debug_assert_eq!(session_key.len(), P::SHARED_SECRET_LENGTH);

    let padding_ok = public_key_padding_is_zero::<P>(pk);

    let mut e = Zeroizing::new(vec![0u8; P::N_BYTES]);
    fixed_weight::<P>(&mut e, rng);

    let (c0, c1) = ciphertext.split_at_mut(P::SYND_BYTES);
    encode::<P>(c0, pk, &e);

    if P::PC {
        hash_32(HASH_PLAINTEXT_CONFIRMATION, &e, &[], c1);
    }

    hash_32(HASH_ENCAPSULATION, &e, ciphertext, session_key);

    let mask = if padding_ok { 0xFFu8 } else { 0x00 };
    for byte in ciphertext.iter_mut() {
        *byte &= mask;
    }
    for byte in session_key.iter_mut() {
        *byte &= mask;
    }

    padding_ok
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6688128",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    use rand_core::{Rng, SeedableRng};

    #[test]
    fn same_mask_is_all_or_nothing() {
        assert_eq!(same_mask(0, 0), u64::MAX);
        assert_eq!(same_mask(0xFFFF, 0xFFFF), u64::MAX);
        assert_eq!(same_mask(1, 0), 0);
        assert_eq!(same_mask(0, 0xFFFF), 0);
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6688128",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn weight(e: &[u8]) -> usize {
        e.iter().map(|b| b.count_ones() as usize).sum()
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6688128",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn fixed_weight_is_well_formed<P: Params>(seed: u8) {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed; 32]);
        for _ in 0..4 {
            let mut e = vec![0u8; P::N_BYTES];
            fixed_weight::<P>(&mut e, &mut rng);
            assert_eq!(weight(&e), P::T, "{} weight", P::NAME);
            // `n` is a multiple of eight for every parameter set, so no bit can land
            // outside the code length.
            assert_eq!(e.len() * 8, P::N);
        }
    }

    /// `Encode` must agree with the definition `C = He` built from `H = (I | T)` directly.
    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6688128",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn encode_matches_the_definition<P: Params>(seed: u8) {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed; 32]);

        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        rng.fill_bytes(&mut pk);
        if P::PUBLIC_KEY_HAS_PADDING {
            // Keep the padding bits zero so the reference computation is well defined.
            for i in 0..P::PK_NROWS {
                let last = i * P::PK_ROW_BYTES + P::PK_ROW_BYTES - 1;
                pk[last] &= (1u8 << (P::PK_NCOLS % 8)) - 1;
            }
        }

        let mut e = vec![0u8; P::N_BYTES];
        fixed_weight::<P>(&mut e, &mut rng);

        let mut expected = vec![0u8; P::SYND_BYTES];
        for i in 0..P::PK_NROWS {
            let row = &pk[i * P::PK_ROW_BYTES..(i + 1) * P::PK_ROW_BYTES];
            let mut bit = (e[i / 8] >> (i % 8)) & 1;
            for column in 0..P::PK_NCOLS {
                let h = (row[column / 8] >> (column % 8)) & 1;
                let position = P::PK_NROWS + column;
                let value = (e[position / 8] >> (position % 8)) & 1;
                bit ^= h & value;
            }
            expected[i / 8] |= bit << (i % 8);
        }

        let mut actual = vec![0u8; P::SYND_BYTES];
        encode::<P>(&mut actual, &pk, &e);
        assert_eq!(actual, expected, "{}", P::NAME);
    }

    #[cfg(any(
        feature = "mceliece348864",
        feature = "mceliece460896",
        feature = "mceliece6688128",
        feature = "mceliece6960119",
        feature = "mceliece8192128"
    ))]
    fn padding_check_rejects_dirty_keys<P: Params>() {
        let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
        assert!(public_key_padding_is_zero::<P>(&pk));

        if P::PUBLIC_KEY_HAS_PADDING {
            let last = P::PK_ROW_BYTES - 1;
            pk[last] = 1 << (P::PK_NCOLS % 8);
            assert!(!public_key_padding_is_zero::<P>(&pk));
        }
    }

    macro_rules! encap_tests {
        ($($feature:literal => $mod_name:ident, $params:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

                    #[test]
                    fn fixed_weight_has_weight_t() {
                        fixed_weight_is_well_formed::<$params>($seed);
                    }

                    #[test]
                    fn encode_is_h_times_e() {
                        encode_matches_the_definition::<$params>($seed ^ 0x5A);
                    }

                    #[test]
                    fn padding_is_checked() {
                        padding_check_rejects_dirty_keys::<$params>();
                    }
                }
            )+
        };
    }

    encap_tests! {
        "mceliece348864" => set_348864, McEliece348864, 1;
        "mceliece460896" => set_460896, McEliece460896, 2;
        "mceliece6688128" => set_6688128, McEliece6688128, 3;
        "mceliece6960119" => set_6960119, McEliece6960119, 4;
        "mceliece8192128" => set_8192128, McEliece8192128, 5;
    }

    #[test]
    #[cfg(feature = "mceliece8192128")]
    fn a_full_length_code_needs_no_rejection() {
        use crate::hazmat::params::McEliece8192128;
        // `n == q` means every drawn index is in range, so `tau` collapses to `t`.
        assert_eq!(McEliece8192128::TAU, McEliece8192128::T);
        assert_eq!(McEliece8192128::N, 1 << McEliece8192128::M);
    }
}
