/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
use thiserror::Error;

/// The errors that can occur when using Classic McEliece.
#[derive(Error, Copy, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The decapsulation key length is invalid.
    #[error("Invalid decapsulation key length: {0}")]
    InvalidDecapsulationKeyLength(usize),
    /// The encapsulation key length is invalid.
    #[error("Invalid encapsulation key length: {0}")]
    InvalidEncapsulationKeyLength(usize),
    /// The ciphertext length is invalid.
    #[error("Invalid ciphertext length: {0}")]
    InvalidCiphertextLength(usize),
    /// The shared secret length is invalid.
    #[error("Invalid shared secret length: {0}")]
    InvalidSharedSecretLength(usize),
    /// The key-generation seed length is invalid.
    #[error("Invalid key-generation seed length: {0}")]
    InvalidSeedLength(usize),
    /// An encapsulation key has nonzero padding bits.
    ///
    /// Only parameter sets whose code dimension `k` is not a multiple of eight can produce
    /// this, which among the standardized sets means `mceliece6960119` and its variants.
    #[error("Encapsulation key has nonzero padding bits")]
    EncapsulationKeyPadding,
    /// A ciphertext has nonzero padding bits.
    ///
    /// Only parameter sets whose `mt` is not a multiple of eight can produce this, which
    /// among the standardized sets means `mceliece6960119` and its variants.
    #[error("Ciphertext has nonzero padding bits")]
    CiphertextPadding,
    /// A key or ciphertext belongs to a different algorithm.
    #[error("Algorithm mismatch")]
    AlgorithmMismatch,
    /// The algorithm name or identifier is not recognized, or its feature is not enabled.
    #[error("Unsupported algorithm")]
    UnsupportedAlgorithm,
}

/// The result type for Classic McEliece.
pub type McElieceResult<T> = Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_renders_a_distinct_message() {
        let errors = [
            Error::InvalidDecapsulationKeyLength(1),
            Error::InvalidEncapsulationKeyLength(2),
            Error::InvalidCiphertextLength(3),
            Error::InvalidSharedSecretLength(4),
            Error::InvalidSeedLength(5),
            Error::EncapsulationKeyPadding,
            Error::CiphertextPadding,
            Error::AlgorithmMismatch,
            Error::UnsupportedAlgorithm,
        ];
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        for (i, a) in messages.iter().enumerate() {
            assert!(!a.is_empty());
            for b in messages.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
        assert_eq!(
            Error::InvalidCiphertextLength(7).to_string(),
            "Invalid ciphertext length: 7"
        );
    }
}
