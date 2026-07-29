/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! The `Hash` function and its input encodings.
//!
//! `Hash(x)` is the first `l = 256` bits of `SHAKE256(x)`. Every hash input begins with a
//! domain byte, and the pseudorandom generator used by key generation begins with byte 64, so
//! the two uses cannot collide.

use shake::digest::{ExtendableOutput, Update};

/// Domain byte for `Hash(0, e, C)`, the implicit-rejection session key.
///
/// Only decapsulation ever produces this input.
#[cfg(feature = "decapsulate")]
pub(crate) const HASH_REJECTION: u8 = 0;

/// Domain byte for `Hash(1, e, C)`, the session key of a successful decode.
pub(crate) const HASH_ENCAPSULATION: u8 = 1;

/// Domain byte for `Hash(2, e)`, the plaintext confirmation of a `pc` parameter set.
pub(crate) const HASH_PLAINTEXT_CONFIRMATION: u8 = 2;

/// Compute `Hash(prefix, vector, ciphertext)` into `out`.
///
/// Pass an empty `ciphertext` for the two-argument form `Hash(2, e)`.
pub(crate) fn hash_32(prefix: u8, vector: &[u8], ciphertext: &[u8], out: &mut [u8]) {
    let mut hasher = shake::Shake256::default();
    hasher.update(&[prefix]);
    hasher.update(vector);
    hasher.update(ciphertext);
    hasher.finalize_xof_into(out);
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_shake256_of_the_concatenation() {
        let mut direct = [0u8; 32];
        let mut hasher = shake::Shake256::default();
        hasher.update(&[1u8, 2, 3, 4, 5, 6]);
        hasher.finalize_xof_into(&mut direct);

        let mut split = [0u8; 32];
        hash_32(1, &[2, 3, 4], &[5, 6], &mut split);
        assert_eq!(direct, split);
    }

    #[cfg(feature = "decapsulate")]
    #[test]
    fn the_domain_bytes_separate_the_three_inputs() {
        let vector = [7u8; 16];
        let ciphertext = [9u8; 8];

        let mut rejection = [0u8; 32];
        let mut encapsulation = [0u8; 32];
        let mut confirmation = [0u8; 32];
        hash_32(HASH_REJECTION, &vector, &ciphertext, &mut rejection);
        hash_32(HASH_ENCAPSULATION, &vector, &ciphertext, &mut encapsulation);
        hash_32(HASH_PLAINTEXT_CONFIRMATION, &vector, &[], &mut confirmation);

        assert_ne!(rejection, encapsulation);
        assert_ne!(rejection, confirmation);
        assert_ne!(encapsulation, confirmation);
        assert_eq!(
            (
                HASH_REJECTION,
                HASH_ENCAPSULATION,
                HASH_PLAINTEXT_CONFIRMATION
            ),
            (0, 1, 2)
        );
    }

    /// The empty-ciphertext form must equal hashing the two pieces alone, so that
    /// `Hash(2, e)` is not accidentally domain-separated a second time.
    #[test]
    fn an_empty_ciphertext_contributes_nothing() {
        let mut with_empty = [0u8; 32];
        hash_32(2, &[1, 2, 3], &[], &mut with_empty);

        let mut direct = [0u8; 32];
        let mut hasher = shake::Shake256::default();
        hasher.update(&[2u8, 1, 2, 3]);
        hasher.finalize_xof_into(&mut direct);

        assert_eq!(with_empty, direct);
    }
}
