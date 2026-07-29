/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Decapsulation: syndrome computation, Berlekamp-Massey decoding and implicit rejection.

use super::benes::apply_benes;
use super::fft::{eval_all_bitrev, eval_transpose_bitrev};
use super::field::{Field, add, is_zero_mask};
use super::hash::{HASH_ENCAPSULATION, HASH_PLAINTEXT_CONFIRMATION, HASH_REJECTION, hash_32};
use super::params::Params;

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

/// The position weighting `1 / g(alpha)^2` for every field element, in bit-reversed order.
///
/// `g` is irreducible of degree at least two, so it has no root anywhere in `F_q` and every
/// inverse below is well defined.
fn scaling<P: Params>(out: &mut [u16], goppa: &[u16]) {
    debug_assert_eq!(out.len(), P::Q);

    eval_all_bitrev::<P>(out, goppa);
    for slot in out.iter_mut() {
        *slot = P::Field::inv(P::Field::sq(*slot));
    }
}

/// Compute the length-`2t` syndrome from a bit-reversed-order received word.
///
/// Entry `j` is `sum_k scale[k] * bits[k] * bitrev(k)^j`. Written out, that is one geometric
/// series per field element and costs `q * 2t` field multiplications; it is instead the
/// transpose of multipoint evaluation, so the transposed additive FFT produces it in about
/// `(q/2) * log2(2t)` multiplications.
///
/// `weighted` is caller-supplied scratch of `q` elements, since decoding needs two syndromes.
fn syndrome<P: Params>(out: &mut [u16], scale: &[u16], bits: &[u8], weighted: &mut [u16]) {
    debug_assert_eq!(out.len(), 2 * P::T);
    debug_assert_eq!(scale.len(), P::Q);
    debug_assert_eq!(weighted.len(), P::Q);

    // Folding the received bit into the weighting is what makes the sum linear in the input,
    // and therefore expressible as the transpose.
    for (k, slot) in weighted.iter_mut().enumerate() {
        let bit = ((bits[k / 8] >> (k % 8)) & 1) as u16;
        *slot = scale[k] & 0u16.wrapping_sub(bit);
    }

    eval_transpose_bitrev::<P>(out, weighted);
}

/// Berlekamp-Massey: recover the error locator polynomial from a syndrome.
///
/// The returned polynomial has `t + 1` coefficients in ascending order. The whole routine is
/// branch free: the two update decisions are folded into masks so that neither the current
/// linear complexity nor the discrepancy leaks through control flow.
fn berlekamp_massey<P: Params>(out: &mut [u16], syndrome: &[u16]) {
    let t = P::T;
    debug_assert_eq!(out.len(), t + 1);
    debug_assert_eq!(syndrome.len(), 2 * t);

    let mut current = vec![0u16; t + 1];
    let mut previous = vec![0u16; t + 1];
    let mut saved = vec![0u16; t + 1];
    let mut length = 0u16;
    let mut base = 1u16;

    current[0] = 1;
    previous[1] = 1;

    for n in 0..2 * t {
        let mut discrepancy = 0u16;
        for i in 0..=n.min(t) {
            discrepancy = add(discrepancy, P::Field::mul(current[i], syndrome[n - i]));
        }

        // All ones when the discrepancy is nonzero.
        let nonzero = discrepancy.wrapping_sub(1).wrapping_shr(15).wrapping_sub(1);
        // All ones when `n >= 2 * length`, i.e. when the linear complexity must grow.
        let grow = (n as u16)
            .wrapping_sub(length.wrapping_mul(2))
            .wrapping_shr(15)
            .wrapping_sub(1)
            & nonzero;

        saved.copy_from_slice(&current);

        let factor = P::Field::frac(base, discrepancy);
        for i in 0..=t {
            current[i] ^= P::Field::mul(factor, previous[i]) & nonzero;
        }

        length = (length & !grow) | (((n as u16).wrapping_add(1).wrapping_sub(length)) & grow);
        for i in 0..=t {
            previous[i] = (previous[i] & !grow) | (saved[i] & grow);
        }
        base = (base & !grow) | (discrepancy & grow);

        // Multiply the previous candidate by `x`.
        for i in (1..=t).rev() {
            previous[i] = previous[i - 1];
        }
        previous[0] = 0;
    }

    for (i, slot) in out.iter_mut().enumerate() {
        *slot = current[t - i];
    }
}

/// `Decode`: recover the weight-`t` error vector `e` from the syndrome `C0`.
///
/// Returns an all-ones mask when decoding succeeded and an all-zeros mask otherwise. A mask
/// rather than a `bool` keeps the caller's selection of `e` against the rejection string `s`
/// free of any branch on secret data. On failure `e` holds an unspecified value that the
/// caller must discard in favour of `s`.
///
/// The work happens in field-element order rather than code-position order. Support element
/// `alpha_i` is `bitrev` of the permutation image of position `i`, so moving the received word
/// through the inverse Beneš network lines positions up against field elements without a single
/// secret-dependent index. That is what lets the additive FFT evaluate the Goppa polynomial and
/// the error locator at every element at once, instead of one support point at a time.
pub(crate) fn decode<P: Params>(e: &mut [u8], sk: &[u8], c0: &[u8]) -> u8 {
    debug_assert_eq!(e.len(), P::N_BYTES);
    debug_assert_eq!(c0.len(), P::SYND_BYTES);

    let cond = control_bits::<P>(sk);
    let plane = P::Q / 8;

    let mut goppa = vec![0u16; P::T + 1];
    load_goppa::<P>(sk, &mut goppa);

    // The received word: `C0` extended with zeros, then moved into field-element order.
    let mut received = vec![0u8; plane];
    received[..P::SYND_BYTES].copy_from_slice(c0);
    apply_benes::<P>(&mut received, cond, true);

    // Which field elements carry a real code position. Elements beyond the code length must
    // not contribute to the re-encryption syndrome even if the locator happens to vanish there.
    let mut valid = vec![0u8; plane];
    valid[..P::N_BYTES].fill(0xFF);
    apply_benes::<P>(&mut valid, cond, true);

    // The weighting depends only on the key, so both syndromes below share it.
    let mut scale = vec![0u16; P::Q];
    scaling::<P>(&mut scale, &goppa);

    let mut weighted = vec![0u16; P::Q];
    let mut s = vec![0u16; 2 * P::T];
    syndrome::<P>(&mut s, &scale, &received, &mut weighted);

    let mut locator = vec![0u16; P::T + 1];
    berlekamp_massey::<P>(&mut locator, &s);

    let mut images = vec![0u16; P::Q];
    eval_all_bitrev::<P>(&mut images, &locator);

    let mut error = vec![0u8; plane];
    let mut weight = 0u16;
    for (k, &image) in images.iter().enumerate() {
        let is_root = (is_zero_mask(image) & 1) as u8 & ((valid[k / 8] >> (k % 8)) & 1);
        error[k / 8] |= is_root << (k % 8);
        weight += is_root as u16;
    }

    // The decoder is only trusted when the recovered `e` has weight exactly `t` and reproduces
    // the syndrome, which is the `wt(e) = t and C = He` condition of the specification.
    let mut recomputed = vec![0u16; 2 * P::T];
    syndrome::<P>(&mut recomputed, &scale, &error, &mut weighted);

    let mut check = weight ^ (P::T as u16);
    for (a, b) in s.iter().zip(recomputed.iter()) {
        check |= a ^ b;
    }

    // Back to code-position order.
    apply_benes::<P>(&mut error, cond, false);
    e.copy_from_slice(&error[..P::N_BYTES]);

    use zeroize::Zeroize;
    goppa.zeroize();
    weighted.zeroize();
    received.zeroize();
    valid.zeroize();
    scale.zeroize();
    s.zeroize();
    recomputed.zeroize();
    locator.zeroize();
    images.zeroize();
    error.zeroize();

    // `check == 0` exactly when decoding succeeded.
    0u8.wrapping_sub((check.wrapping_sub(1) >> 15) as u8)
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
pub(crate) fn decapsulate<P: Params>(session_key: &mut [u8], ciphertext: &[u8], sk: &[u8]) -> bool {
    debug_assert_eq!(session_key.len(), P::SHARED_SECRET_LENGTH);
    debug_assert_eq!(ciphertext.len(), P::CIPHERTEXT_LENGTH);
    debug_assert_eq!(sk.len(), P::SECRET_KEY_LENGTH);

    let padding_ok = ciphertext_padding_is_zero::<P>(ciphertext);

    let mut decoded = vec![0u8; P::N_BYTES];
    let mut keep = decode::<P>(&mut decoded, sk, &ciphertext[..P::SYND_BYTES]);

    // Implicit rejection: replace `e` with `s` when decoding failed.
    let rejection = &sk[P::S_OFFSET..P::S_OFFSET + P::N_BYTES];
    let mut e = vec![0u8; P::N_BYTES];
    for i in 0..P::N_BYTES {
        e[i] = (decoded[i] & keep) | (rejection[i] & !keep);
    }

    if P::PC {
        // Plaintext confirmation: `C1` must match `Hash(2, e)`, otherwise fall back to `s`.
        let mut confirmation = [0u8; 32];
        hash_32(HASH_PLAINTEXT_CONFIRMATION, &e, &[], &mut confirmation);

        let mut differs = 0u8;
        for (a, b) in confirmation.iter().zip(ciphertext[P::SYND_BYTES..].iter()) {
            differs |= a ^ b;
        }
        let matches = 0u8.wrapping_sub(((differs as u32).wrapping_sub(1) >> 31) as u8);

        for i in 0..P::N_BYTES {
            e[i] = (e[i] & matches) | (rejection[i] & !matches);
        }
        keep &= matches;
    }

    // `HASH_REJECTION` is 0 and `HASH_ENCAPSULATION` is 1, so masking selects the domain byte
    // without a branch.
    debug_assert_eq!((HASH_REJECTION, HASH_ENCAPSULATION), (0, 1));
    hash_32(HASH_ENCAPSULATION & keep, &e, ciphertext, session_key);

    use zeroize::Zeroize;
    decoded.zeroize();
    e.zeroize();

    if !padding_ok {
        for byte in session_key.iter_mut() {
            *byte |= 0xFF;
        }
    }

    padding_ok
}

#[cfg(all(test, feature = "keygen", feature = "encapsulate"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::hazmat::encap::{encapsulate, fixed_weight};
    use crate::hazmat::keygen::seeded_keypair;
    use rand_core::SeedableRng;

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
