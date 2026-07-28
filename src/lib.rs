/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! A pure Rust implementation of the Classic McEliece key encapsulation mechanism.
//!
//! Every enabled parameter set is available at once. Unlike implementations that bake a single
//! parameter set into the crate through cargo features, choosing between `mceliece348864` and
//! `mceliece8192128pcf` here is a runtime decision.
//!
//! ## Usage
//!
//! The straightforward path is [`Algorithm`], which carries the parameter set as a value:
//!
//! ```
//! # #[cfg(feature = "mceliece348864f")] {
//! use pq_mceliece::Algorithm;
//! use rand::rngs::SysRng;
//! use rand_core::UnwrapErr;
//!
//! let alg = Algorithm::McEliece348864f;
//! let (ek, dk) = alg.generate_keypair(UnwrapErr(SysRng));
//! let (ct, sent) = alg.encapsulate(&ek, UnwrapErr(SysRng)).unwrap();
//! let received = alg.decapsulate(&dk, &ct).unwrap();
//!
//! assert_eq!(sent, received);
//! # }
//! ```
//!
//! Key generation is deterministic in a 32-byte seed, so a key pair can be reproduced from
//! backed-up seed material:
//!
//! ```
//! use pq_mceliece::Algorithm;
//!
//! // Any enabled parameter set works; `default` picks the first one.
//! let alg = Algorithm::default();
//! let seed = [0x42u8; 32];
//! let first = alg.generate_keypair_from_seed(seed).unwrap();
//! let second = alg.generate_keypair_from_seed(seed).unwrap();
//! assert_eq!(first, second);
//! ```
//!
//! When the parameter set is known at compile time, the [`hazmat`] layer offers the same
//! operations with the parameter set in the type, so a key from one parameter set cannot be
//! passed to another:
//!
//! ```
//! # #[cfg(all(feature = "mceliece348864", feature = "hazmat"))] {
//! use pq_mceliece::hazmat::{Kem, McEliece348864};
//! use rand::rngs::SysRng;
//! use rand_core::UnwrapErr;
//!
//! let (ek, dk) = McEliece348864::generate_keypair(UnwrapErr(SysRng));
//! let (ct, sent) = McEliece348864::encapsulate(&ek, UnwrapErr(SysRng)).unwrap();
//! let received = McEliece348864::decapsulate(&dk, &ct).unwrap();
//!
//! assert_eq!(sent, received);
//! # }
//! ```
//!
//! Implementations of the [`kem`](https://docs.rs/kem) crate traits live in the
//! [`kem`](crate::kem) module.
//!
//! ## Standards
//!
//! | parameter sets | NIST round 4 | ISO |
//! | -------------- | ------------ | --- |
//! | `mceliece348864`, `mceliece348864f` | yes | no |
//! | `mceliece460896`, `mceliece6688128`, `mceliece6960119`, `mceliece8192128`, and their `f` variants | yes | yes |
//! | the `pc` and `pcf` variants of those four sizes | no | yes |
//!
//! The `pc` parameter sets add *plaintext confirmation*: the ciphertext carries an extra
//! 32-byte hash of the error vector, which decapsulation checks before deriving the session
//! key. They are part of the ISO standard published in June 2026 and are not in the NIST
//! submission.
//!
//! Implementation choices follow the *narrowly decoded* reading of the specification:
//! encapsulation keys and ciphertexts with nonzero padding bits are rejected rather than
//! ignored. This only ever applies to `mceliece6960119`, the one standardized parameter set
//! whose `mt` and `k` are not multiples of eight.
//!
//! ## Conformance
//!
//! Every NIST parameter set is checked bit for bit against the published `kat_kem.rsp` known
//! answer tests. See `CONFORMANCE.md` for what is verified and how.
//!
//! ## Features
//!
//! Each parameter set has its own feature, and the `nist`, `iso` and `pc` groups enable the
//! corresponding sets. The `serde` feature adds serialization for every value type, and
//! `hazmat` makes the low-level layer public.
#![warn(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused,
    clippy::mod_module_files
)]
#![deny(clippy::unwrap_used)]

#[cfg(not(any(
    feature = "mceliece348864",
    feature = "mceliece348864f",
    feature = "mceliece460896",
    feature = "mceliece460896f",
    feature = "mceliece460896pc",
    feature = "mceliece460896pcf",
    feature = "mceliece6688128",
    feature = "mceliece6688128f",
    feature = "mceliece6688128pc",
    feature = "mceliece6688128pcf",
    feature = "mceliece6960119",
    feature = "mceliece6960119f",
    feature = "mceliece6960119pc",
    feature = "mceliece6960119pcf",
    feature = "mceliece8192128",
    feature = "mceliece8192128f",
    feature = "mceliece8192128pc",
    feature = "mceliece8192128pcf",
)))]
compile_error!("no Classic McEliece parameter set feature is enabled");

mod error;
pub use error::*;

pub mod kem;

#[cfg(feature = "hazmat")]
pub mod hazmat;
#[cfg(not(feature = "hazmat"))]
mod hazmat;

use ctutils::{Choice, CtEq};
use hazmat::{Kem, Params};
use rand_core::CryptoRng;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Run `$body` with `$p` bound to the parameter set of `$alg`.
///
/// Every dynamic operation funnels through this, so adding a parameter set means adding one
/// arm here and one entry to [`Algorithm`].
macro_rules! with_params {
    ($alg:expr, $p:ident, $body:block) => {
        match $alg {
            #[cfg(feature = "mceliece348864")]
            Algorithm::McEliece348864 => {
                type $p = hazmat::McEliece348864;
                $body
            }
            #[cfg(feature = "mceliece348864f")]
            Algorithm::McEliece348864f => {
                type $p = hazmat::McEliece348864f;
                $body
            }
            #[cfg(feature = "mceliece460896")]
            Algorithm::McEliece460896 => {
                type $p = hazmat::McEliece460896;
                $body
            }
            #[cfg(feature = "mceliece460896f")]
            Algorithm::McEliece460896f => {
                type $p = hazmat::McEliece460896f;
                $body
            }
            #[cfg(feature = "mceliece460896pc")]
            Algorithm::McEliece460896pc => {
                type $p = hazmat::McEliece460896pc;
                $body
            }
            #[cfg(feature = "mceliece460896pcf")]
            Algorithm::McEliece460896pcf => {
                type $p = hazmat::McEliece460896pcf;
                $body
            }
            #[cfg(feature = "mceliece6688128")]
            Algorithm::McEliece6688128 => {
                type $p = hazmat::McEliece6688128;
                $body
            }
            #[cfg(feature = "mceliece6688128f")]
            Algorithm::McEliece6688128f => {
                type $p = hazmat::McEliece6688128f;
                $body
            }
            #[cfg(feature = "mceliece6688128pc")]
            Algorithm::McEliece6688128pc => {
                type $p = hazmat::McEliece6688128pc;
                $body
            }
            #[cfg(feature = "mceliece6688128pcf")]
            Algorithm::McEliece6688128pcf => {
                type $p = hazmat::McEliece6688128pcf;
                $body
            }
            #[cfg(feature = "mceliece6960119")]
            Algorithm::McEliece6960119 => {
                type $p = hazmat::McEliece6960119;
                $body
            }
            #[cfg(feature = "mceliece6960119f")]
            Algorithm::McEliece6960119f => {
                type $p = hazmat::McEliece6960119f;
                $body
            }
            #[cfg(feature = "mceliece6960119pc")]
            Algorithm::McEliece6960119pc => {
                type $p = hazmat::McEliece6960119pc;
                $body
            }
            #[cfg(feature = "mceliece6960119pcf")]
            Algorithm::McEliece6960119pcf => {
                type $p = hazmat::McEliece6960119pcf;
                $body
            }
            #[cfg(feature = "mceliece8192128")]
            Algorithm::McEliece8192128 => {
                type $p = hazmat::McEliece8192128;
                $body
            }
            #[cfg(feature = "mceliece8192128f")]
            Algorithm::McEliece8192128f => {
                type $p = hazmat::McEliece8192128f;
                $body
            }
            #[cfg(feature = "mceliece8192128pc")]
            Algorithm::McEliece8192128pc => {
                type $p = hazmat::McEliece8192128pc;
                $body
            }
            #[cfg(feature = "mceliece8192128pcf")]
            Algorithm::McEliece8192128pcf => {
                type $p = hazmat::McEliece8192128pcf;
                $body
            }
        }
    };
}

/// Declare the [`Algorithm`] enum along with its name and identifier mappings.
macro_rules! declare_algorithms {
    ($(
        $(#[$meta:meta])*
        $feature:literal => $variant:ident, $id:literal;
    )+) => {
        /// A Classic McEliece parameter set, selected at run time.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
        #[non_exhaustive]
        pub enum Algorithm {
            $(
                $(#[$meta])*
                #[cfg(feature = $feature)]
                $variant,
            )+
        }

        impl Algorithm {
            /// Every parameter set enabled in this build, in a stable order.
            pub const fn enabled_algorithms() -> &'static [Algorithm] {
                &[
                    $(
                        #[cfg(feature = $feature)]
                        Self::$variant,
                    )+
                ]
            }
        }

        impl From<Algorithm> for u8 {
            fn from(algorithm: Algorithm) -> u8 {
                match algorithm {
                    $(
                        #[cfg(feature = $feature)]
                        Algorithm::$variant => $id,
                    )+
                }
            }
        }

        impl TryFrom<u8> for Algorithm {
            type Error = Error;

            fn try_from(value: u8) -> Result<Self, Self::Error> {
                match value {
                    $(
                        #[cfg(feature = $feature)]
                        $id => Ok(Algorithm::$variant),
                    )+
                    _ => Err(Error::UnsupportedAlgorithm),
                }
            }
        }

        impl core::str::FromStr for Algorithm {
            type Err = Error;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $(
                        #[cfg(feature = $feature)]
                        $feature => Ok(Algorithm::$variant),
                    )+
                    _ => Err(Error::UnsupportedAlgorithm),
                }
            }
        }
    };
}

declare_algorithms! {
    /// `mceliece348864`: NIST category 1. In the NIST submission only.
    "mceliece348864" => McEliece348864, 1;
    /// `mceliece348864f`: `mceliece348864` with faster semi-systematic key generation.
    "mceliece348864f" => McEliece348864f, 2;
    /// `mceliece460896`: NIST category 3.
    "mceliece460896" => McEliece460896, 3;
    /// `mceliece460896f`: `mceliece460896` with faster semi-systematic key generation.
    "mceliece460896f" => McEliece460896f, 4;
    /// `mceliece460896pc`: `mceliece460896` with plaintext confirmation. ISO only.
    "mceliece460896pc" => McEliece460896pc, 5;
    /// `mceliece460896pcf`: plaintext confirmation and semi-systematic key generation.
    "mceliece460896pcf" => McEliece460896pcf, 6;
    /// `mceliece6688128`: NIST category 5.
    "mceliece6688128" => McEliece6688128, 7;
    /// `mceliece6688128f`: `mceliece6688128` with faster semi-systematic key generation.
    "mceliece6688128f" => McEliece6688128f, 8;
    /// `mceliece6688128pc`: `mceliece6688128` with plaintext confirmation. ISO only.
    "mceliece6688128pc" => McEliece6688128pc, 9;
    /// `mceliece6688128pcf`: plaintext confirmation and semi-systematic key generation.
    "mceliece6688128pcf" => McEliece6688128pcf, 10;
    /// `mceliece6960119`: NIST category 5.
    "mceliece6960119" => McEliece6960119, 11;
    /// `mceliece6960119f`: `mceliece6960119` with faster semi-systematic key generation.
    "mceliece6960119f" => McEliece6960119f, 12;
    /// `mceliece6960119pc`: `mceliece6960119` with plaintext confirmation. ISO only.
    "mceliece6960119pc" => McEliece6960119pc, 13;
    /// `mceliece6960119pcf`: plaintext confirmation and semi-systematic key generation.
    "mceliece6960119pcf" => McEliece6960119pcf, 14;
    /// `mceliece8192128`: NIST category 5.
    "mceliece8192128" => McEliece8192128, 15;
    /// `mceliece8192128f`: `mceliece8192128` with faster semi-systematic key generation.
    "mceliece8192128f" => McEliece8192128f, 16;
    /// `mceliece8192128pc`: `mceliece8192128` with plaintext confirmation. ISO only.
    "mceliece8192128pc" => McEliece8192128pc, 17;
    /// `mceliece8192128pcf`: plaintext confirmation and semi-systematic key generation.
    "mceliece8192128pcf" => McEliece8192128pcf, 18;
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::enabled_algorithms()[0]
    }
}

impl core::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

impl CtEq for Algorithm {
    fn ct_eq(&self, other: &Self) -> Choice {
        u8::from(*self).ct_eq(&u8::from(*other))
    }
}

macro_rules! widening_conversions {
    ($($ty:ty),+) => {
        $(
            impl From<Algorithm> for $ty {
                fn from(algorithm: Algorithm) -> Self {
                    u8::from(algorithm) as $ty
                }
            }

            impl TryFrom<$ty> for Algorithm {
                type Error = Error;

                fn try_from(value: $ty) -> Result<Self, Self::Error> {
                    u8::try_from(value)
                        .map_err(|_| Error::UnsupportedAlgorithm)?
                        .try_into()
                }
            }
        )+
    };
}

widening_conversions!(u16, u32, u64, u128, usize);

/// The parameters underlying an [`Algorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AlgorithmParams {
    /// The parameter set name.
    pub name: &'static str,
    /// The extension degree `m`, so that the field is `GF(2^m)`.
    pub m: usize,
    /// The code length `n`.
    pub n: usize,
    /// The number of correctable errors `t`, and the degree of the Goppa polynomial.
    pub t: usize,
    /// The code dimension `k = n - mt`.
    pub k: usize,
    /// The semi-systematic parameters `(mu, nu)`. `(0, 0)` for non-`f` parameter sets.
    pub semi_systematic: (usize, usize),
    /// Whether the parameter set uses plaintext confirmation.
    pub plaintext_confirmation: bool,
    /// The claimed NIST security category.
    pub claimed_nist_level: usize,
    /// Whether the parameter set is in the NIST round-4 submission.
    pub in_nist_submission: bool,
    /// Whether the parameter set is in the ISO standard.
    pub in_iso_standard: bool,
    /// The byte length of an encapsulation key.
    pub encapsulation_key_length: usize,
    /// The byte length of a decapsulation key.
    pub decapsulation_key_length: usize,
    /// The byte length of a ciphertext.
    pub ciphertext_length: usize,
    /// The byte length of a shared secret.
    pub shared_secret_length: usize,
    /// The byte length of a key-generation seed.
    pub seed_length: usize,
}

macro_rules! serde_impl {
    ($name:ident, $from_method:ident) => {
        #[cfg(feature = "serde")]
        impl serde::Serialize for $name {
            fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                if s.is_human_readable() {
                    use serde::ser::SerializeStruct;

                    let mut map = s.serialize_struct(stringify!($name), 2)?;
                    map.serialize_field("algorithm", self.algorithm.name())?;
                    map.serialize_field("value", &hex::encode(&self.value))?;
                    map.end()
                } else {
                    let mut seq = Vec::with_capacity(self.value.len() + 1);
                    seq.push(u8::from(self.algorithm));
                    seq.extend_from_slice(&self.value);
                    s.serialize_bytes(&seq)
                }
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(d: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                if d.is_human_readable() {
                    #[derive(serde::Deserialize)]
                    #[serde(field_identifier, rename_all = "snake_case")]
                    enum Field {
                        Algorithm,
                        Value,
                    }

                    struct StructVisitor;

                    impl<'de> serde::de::Visitor<'de> for StructVisitor {
                        type Value = $name;

                        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                            write!(f, "a struct with an algorithm and a hex value")
                        }

                        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                        where
                            A: serde::de::MapAccess<'de>,
                        {
                            let mut algorithm = Option::<Algorithm>::None;
                            let mut value = Option::<String>::None;
                            while let Some(key) = map.next_key()? {
                                match key {
                                    Field::Algorithm => {
                                        if algorithm.is_some() {
                                            return Err(serde::de::Error::duplicate_field(
                                                "algorithm",
                                            ));
                                        }
                                        algorithm = Some(map.next_value()?);
                                    }
                                    Field::Value => {
                                        if value.is_some() {
                                            return Err(serde::de::Error::duplicate_field("value"));
                                        }
                                        value = Some(map.next_value()?);
                                    }
                                }
                            }

                            let algorithm = algorithm
                                .ok_or_else(|| serde::de::Error::missing_field("algorithm"))?;
                            let value =
                                value.ok_or_else(|| serde::de::Error::missing_field("value"))?;
                            let value = hex::decode(&value).map_err(serde::de::Error::custom)?;
                            algorithm
                                .$from_method(&value)
                                .map_err(serde::de::Error::custom)
                        }
                    }

                    const FIELDS: &[&str] = &["algorithm", "value"];
                    d.deserialize_struct(stringify!($name), FIELDS, StructVisitor)
                } else {
                    struct BytesVisitor;

                    impl<'de> serde::de::Visitor<'de> for BytesVisitor {
                        type Value = $name;

                        fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                            write!(f, "a byte sequence")
                        }

                        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                        where
                            E: serde::de::Error,
                        {
                            let (tag, value) = v.split_first().ok_or_else(|| {
                                serde::de::Error::custom(Error::UnsupportedAlgorithm)
                            })?;
                            let algorithm =
                                Algorithm::try_from(*tag).map_err(serde::de::Error::custom)?;
                            algorithm
                                .$from_method(value)
                                .map_err(serde::de::Error::custom)
                        }
                    }

                    d.deserialize_bytes(BytesVisitor)
                }
            }
        }
    };
}

macro_rules! value_type {
    (
        $(#[$meta:meta])*
        $name:ident, $from_method:ident, secret = $secret:literal
    ) => {
        $(#[$meta])*
        #[derive(Clone, Default)]
        pub struct $name {
            pub(crate) algorithm: Algorithm,
            pub(crate) value: Vec<u8>,
        }

        impl $name {
            /// The parameter set this value belongs to.
            pub fn algorithm(&self) -> Algorithm {
                self.algorithm
            }

            /// The raw bytes.
            pub fn value(&self) -> &[u8] {
                &self.value
            }

            /// Parse bytes as this value type for `algorithm`.
            pub fn from_bytes<B: AsRef<[u8]>>(
                algorithm: Algorithm,
                value: B,
            ) -> McElieceResult<Self> {
                algorithm.$from_method(value.as_ref())
            }
        }

        impl AsRef<[u8]> for $name {
            fn as_ref(&self) -> &[u8] {
                &self.value
            }
        }

        impl CtEq for $name {
            fn ct_eq(&self, other: &Self) -> Choice {
                self.algorithm.ct_eq(&other.algorithm)
                    & self.value.as_slice().ct_eq(other.value.as_slice())
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                bool::from(self.ct_eq(other))
            }
        }

        impl Eq for $name {}

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut out = f.debug_struct(stringify!($name));
                out.field("algorithm", &self.algorithm);
                if $secret {
                    out.finish_non_exhaustive()
                } else {
                    out.field("value", &self.value).finish()
                }
            }
        }

        serde_impl!($name, $from_method);
    };
}

value_type! {
    /// A Classic McEliece encapsulation (public) key.
    EncapsulationKey, encapsulation_key_from_bytes, secret = false
}

value_type! {
    /// A Classic McEliece decapsulation (private) key.
    DecapsulationKey, decapsulation_key_from_bytes, secret = true
}

value_type! {
    /// A Classic McEliece ciphertext.
    Ciphertext, ciphertext_from_bytes, secret = false
}

value_type! {
    /// A Classic McEliece shared secret.
    SharedSecret, shared_secret_from_bytes, secret = true
}

macro_rules! zeroize_on_drop {
    ($name:ident) => {
        impl Zeroize for $name {
            fn zeroize(&mut self) {
                self.value.zeroize();
            }
        }

        impl ZeroizeOnDrop for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                self.zeroize();
            }
        }
    };
}

zeroize_on_drop!(DecapsulationKey);
zeroize_on_drop!(SharedSecret);

impl EncapsulationKey {
    /// Encapsulate to this key, producing a ciphertext and the shared secret it transports.
    pub fn encapsulate(&self, rng: impl CryptoRng) -> McElieceResult<(Ciphertext, SharedSecret)> {
        self.algorithm.encapsulate(self, rng)
    }
}

impl From<&DecapsulationKey> for EncapsulationKey {
    /// Recover the encapsulation key by rerunning key generation from the stored seed.
    ///
    /// This costs a full key generation.
    fn from(secret: &DecapsulationKey) -> Self {
        secret
            .algorithm
            .encapsulation_key_from_decapsulation_key(secret)
    }
}

impl DecapsulationKey {
    /// Recover the shared secret from a ciphertext.
    ///
    /// See [`Algorithm::decapsulate`] for what happens when the ciphertext does not decode.
    pub fn decapsulate(&self, ciphertext: &Ciphertext) -> McElieceResult<SharedSecret> {
        self.algorithm.decapsulate(self, ciphertext)
    }
}

impl Algorithm {
    /// The lowercase parameter set name, e.g. `mceliece6960119f`.
    pub const fn name(&self) -> &'static str {
        with_params!(self, P, { P::NAME })
    }

    /// The parameters underlying this parameter set.
    pub const fn params(&self) -> AlgorithmParams {
        with_params!(self, P, {
            AlgorithmParams {
                name: P::NAME,
                m: P::M,
                n: P::N,
                t: P::T,
                k: P::K,
                semi_systematic: (P::MU, P::NU),
                plaintext_confirmation: P::PC,
                claimed_nist_level: P::CLAIMED_NIST_LEVEL,
                in_nist_submission: P::STANDARDS.nist,
                in_iso_standard: P::STANDARDS.iso,
                encapsulation_key_length: P::PUBLIC_KEY_LENGTH,
                decapsulation_key_length: P::SECRET_KEY_LENGTH,
                ciphertext_length: P::CIPHERTEXT_LENGTH,
                shared_secret_length: P::SHARED_SECRET_LENGTH,
                seed_length: P::SEED_LENGTH,
            }
        })
    }

    /// Generate a key pair, drawing a fresh seed from `rng`.
    pub fn generate_keypair(&self, rng: impl CryptoRng) -> (EncapsulationKey, DecapsulationKey) {
        with_params!(self, P, {
            let (ek, dk) = <P as Kem>::generate_keypair(rng);
            (
                EncapsulationKey {
                    algorithm: *self,
                    value: ek.into_vec(),
                },
                DecapsulationKey {
                    algorithm: *self,
                    value: dk.into_vec(),
                },
            )
        })
    }

    /// Generate a key pair deterministically from a 32-byte seed.
    ///
    /// The seed must come from a cryptographically secure source and is exactly as sensitive
    /// as the resulting private key.
    pub fn generate_keypair_from_seed<B: AsRef<[u8]>>(
        &self,
        seed: B,
    ) -> McElieceResult<(EncapsulationKey, DecapsulationKey)> {
        with_params!(self, P, {
            let (ek, dk) = <P as Kem>::generate_keypair_from_seed(seed.as_ref())?;
            Ok((
                EncapsulationKey {
                    algorithm: *self,
                    value: ek.into_vec(),
                },
                DecapsulationKey {
                    algorithm: *self,
                    value: dk.into_vec(),
                },
            ))
        })
    }

    /// Encapsulate to `key`, producing a ciphertext and the shared secret it transports.
    pub fn encapsulate(
        &self,
        key: &EncapsulationKey,
        rng: impl CryptoRng,
    ) -> McElieceResult<(Ciphertext, SharedSecret)> {
        if key.algorithm != *self {
            return Err(Error::AlgorithmMismatch);
        }
        with_params!(self, P, {
            let typed = hazmat::EncapsulationKey::<P>::from_slice(&key.value)?;
            let (ct, ss) = <P as Kem>::encapsulate(&typed, rng)?;
            Ok((
                Ciphertext {
                    algorithm: *self,
                    value: ct.into_vec(),
                },
                SharedSecret {
                    algorithm: *self,
                    value: ss.into_vec(),
                },
            ))
        })
    }

    /// Recover the shared secret from `ciphertext`.
    ///
    /// A ciphertext that does not decode is not reported as an error. Classic McEliece rejects
    /// implicitly: the returned secret is derived from the private key's rejection string, so
    /// it is unpredictable without the private key and identical every time the same bad
    /// ciphertext is presented.
    pub fn decapsulate(
        &self,
        key: &DecapsulationKey,
        ciphertext: &Ciphertext,
    ) -> McElieceResult<SharedSecret> {
        if key.algorithm != *self || ciphertext.algorithm != *self {
            return Err(Error::AlgorithmMismatch);
        }
        with_params!(self, P, {
            let typed_key = hazmat::DecapsulationKey::<P>::from_slice(&key.value)?;
            let typed_ct = hazmat::Ciphertext::<P>::from_slice(&ciphertext.value)?;
            let ss = <P as Kem>::decapsulate(&typed_key, &typed_ct)?;
            Ok(SharedSecret {
                algorithm: *self,
                value: ss.into_vec(),
            })
        })
    }

    /// Recover an encapsulation key from a decapsulation key.
    pub fn encapsulation_key_from_decapsulation_key(
        &self,
        key: &DecapsulationKey,
    ) -> EncapsulationKey {
        with_params!(self, P, {
            match hazmat::DecapsulationKey::<P>::from_slice(&key.value) {
                Ok(typed) => EncapsulationKey {
                    algorithm: *self,
                    value: hazmat::EncapsulationKey::<P>::from(&typed).into_vec(),
                },
                // A `DecapsulationKey` is only constructible through a length-checked
                // constructor, so this arm means the value belongs to another parameter set.
                Err(_) => EncapsulationKey::default(),
            }
        })
    }

    /// Parse bytes as an encapsulation key for this parameter set.
    pub fn encapsulation_key_from_bytes(&self, value: &[u8]) -> McElieceResult<EncapsulationKey> {
        with_params!(self, P, {
            hazmat::EncapsulationKey::<P>::from_slice(value).map(|key| EncapsulationKey {
                algorithm: *self,
                value: key.into_vec(),
            })
        })
    }

    /// Parse bytes as a decapsulation key for this parameter set.
    pub fn decapsulation_key_from_bytes(&self, value: &[u8]) -> McElieceResult<DecapsulationKey> {
        with_params!(self, P, {
            hazmat::DecapsulationKey::<P>::from_slice(value).map(|key| DecapsulationKey {
                algorithm: *self,
                value: key.into_vec(),
            })
        })
    }

    /// Parse bytes as a ciphertext for this parameter set.
    pub fn ciphertext_from_bytes(&self, value: &[u8]) -> McElieceResult<Ciphertext> {
        with_params!(self, P, {
            hazmat::Ciphertext::<P>::from_slice(value).map(|ct| Ciphertext {
                algorithm: *self,
                value: ct.into_vec(),
            })
        })
    }

    /// Parse bytes as a shared secret for this parameter set.
    pub fn shared_secret_from_bytes(&self, value: &[u8]) -> McElieceResult<SharedSecret> {
        with_params!(self, P, {
            hazmat::SharedSecret::<P>::from_slice(value).map(|ss| SharedSecret {
                algorithm: *self,
                value: ss.into_vec(),
            })
        })
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Algorithm {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if s.is_human_readable() {
            s.serialize_str(self.name())
        } else {
            s.serialize_u8(u8::from(*self))
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Algorithm {
    fn deserialize<D>(d: D) -> Result<Algorithm, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if d.is_human_readable() {
            let name = <String as serde::Deserialize>::deserialize(d)?;
            name.parse().map_err(serde::de::Error::custom)
        } else {
            let tag = <u8 as serde::Deserialize>::deserialize(d)?;
            tag.try_into().map_err(serde::de::Error::custom)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn every_algorithm_round_trips_through_its_public_conversions() {
        for &algorithm in Algorithm::enabled_algorithms() {
            let params = algorithm.params();
            assert_eq!(params.name, algorithm.name());
            assert_eq!(algorithm.to_string(), params.name);
            assert_eq!(params.name.parse::<Algorithm>().unwrap(), algorithm);

            let id = u8::from(algorithm);
            assert_eq!(Algorithm::try_from(id).unwrap(), algorithm);
            assert_eq!(
                Algorithm::try_from(u16::from(algorithm)).unwrap(),
                algorithm
            );
            assert_eq!(
                Algorithm::try_from(u32::from(algorithm)).unwrap(),
                algorithm
            );
            assert_eq!(
                Algorithm::try_from(u64::from(algorithm)).unwrap(),
                algorithm
            );
            assert_eq!(
                Algorithm::try_from(usize::from(algorithm)).unwrap(),
                algorithm
            );

            assert_eq!(params.k, params.n - params.m * params.t);
            assert_eq!(params.shared_secret_length, 32);
            assert_eq!(params.seed_length, 32);
            assert!(params.in_nist_submission || params.in_iso_standard);
            // Plaintext confirmation is exactly the 32-byte extension of the ciphertext.
            let syndrome_bytes = (params.m * params.t).div_ceil(8);
            assert_eq!(
                params.ciphertext_length,
                syndrome_bytes + if params.plaintext_confirmation { 32 } else { 0 }
            );
        }

        assert!(Algorithm::try_from(0u8).is_err());
        assert!(Algorithm::try_from(200u8).is_err());
        assert!(Algorithm::try_from(300u16).is_err());
        assert!("not-an-algorithm".parse::<Algorithm>().is_err());
    }

    #[test]
    fn the_standards_table_matches_the_published_lists() {
        for &algorithm in Algorithm::enabled_algorithms() {
            let params = algorithm.params();
            // The ISO standard covers every size except 3488, and adds the pc variants.
            assert_eq!(params.in_iso_standard, params.n != 3488, "{}", params.name);
            // The NIST submission covers every non-pc set.
            assert_eq!(
                params.in_nist_submission, !params.plaintext_confirmation,
                "{}",
                params.name
            );
        }
    }

    #[test]
    fn value_types_validate_lengths_and_hide_secrets() {
        for &algorithm in Algorithm::enabled_algorithms() {
            let params = algorithm.params();

            let ek =
                EncapsulationKey::from_bytes(algorithm, vec![0; params.encapsulation_key_length])
                    .unwrap();
            let dk =
                DecapsulationKey::from_bytes(algorithm, vec![0; params.decapsulation_key_length])
                    .unwrap();
            let ct = Ciphertext::from_bytes(algorithm, vec![0; params.ciphertext_length]).unwrap();
            let ss =
                SharedSecret::from_bytes(algorithm, vec![0; params.shared_secret_length]).unwrap();

            assert_eq!(ek.algorithm(), algorithm);
            assert_eq!(ek.value(), ek.as_ref());
            assert_eq!(dk.algorithm(), algorithm);
            assert_eq!(ct.algorithm(), algorithm);
            assert_eq!(ss.algorithm(), algorithm);

            assert!(EncapsulationKey::from_bytes(algorithm, []).is_err());
            assert!(DecapsulationKey::from_bytes(algorithm, []).is_err());
            assert!(Ciphertext::from_bytes(algorithm, []).is_err());
            assert!(SharedSecret::from_bytes(algorithm, []).is_err());

            assert!(!format!("{dk:?}").contains("value"));
            assert!(!format!("{ss:?}").contains("value"));
            assert!(format!("{ek:?}").contains("value"));
        }
    }

    #[test]
    fn the_dynamic_api_rejects_mismatched_algorithms() {
        let alg = Algorithm::enabled_algorithms()[0];
        let Some(&other) = Algorithm::enabled_algorithms().iter().find(|&&a| a != alg) else {
            // A build with a single parameter set has no mismatch to reject.
            return;
        };

        let mut rng = rand_chacha::ChaCha8Rng::from_seed([5u8; 32]);
        let (ek, dk) = alg.generate_keypair(&mut rng);
        let (ct, _) = alg.encapsulate(&ek, &mut rng).unwrap();

        assert_eq!(
            other.encapsulate(&ek, &mut rng).unwrap_err(),
            Error::AlgorithmMismatch
        );
        assert_eq!(
            other.decapsulate(&dk, &ct).unwrap_err(),
            Error::AlgorithmMismatch
        );

        let foreign_ct =
            Ciphertext::from_bytes(other, vec![0; other.params().ciphertext_length]).unwrap();
        assert_eq!(
            dk.decapsulate(&foreign_ct).unwrap_err(),
            Error::AlgorithmMismatch
        );
    }

    #[test]
    fn a_round_trip_recovers_the_shared_secret_and_rejection_is_stable() {
        let alg = Algorithm::enabled_algorithms()[0];
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([13u8; 32]);
        let (ek, dk) = alg.generate_keypair(&mut rng);

        let (ct, sent) = ek.encapsulate(&mut rng).unwrap();
        assert_eq!(dk.decapsulate(&ct).unwrap(), sent);
        assert_eq!(EncapsulationKey::from(&dk), ek);

        let mut tampered = ct.clone();
        tampered.value[0] ^= 1;
        let first = dk.decapsulate(&tampered).unwrap();
        let second = dk.decapsulate(&tampered).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, sent);
    }

    #[test]
    fn deterministic_generation_validates_the_seed_length() {
        for &algorithm in Algorithm::enabled_algorithms() {
            let seed = vec![0xA5; algorithm.params().seed_length];
            let first = algorithm.generate_keypair_from_seed(&seed).unwrap();
            let second = algorithm.generate_keypair_from_seed(&seed).unwrap();
            assert_eq!(first, second);

            assert_eq!(
                algorithm
                    .generate_keypair_from_seed(&seed[..seed.len() - 1])
                    .unwrap_err(),
                Error::InvalidSeedLength(31)
            );
            let mut long = seed;
            long.push(0);
            assert_eq!(
                algorithm.generate_keypair_from_seed(long).unwrap_err(),
                Error::InvalidSeedLength(33)
            );
        }
    }

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        macro_rules! serde_round_trip {
            ($name:ident, $ser:path, $de:path) => {
                #[test]
                fn $name() {
                    let alg = Algorithm::enabled_algorithms()[0];
                    let mut rng = rand_chacha::ChaCha8Rng::from_seed([3u8; 32]);
                    let (ek, dk) = alg.generate_keypair(&mut rng);
                    let (ct, ss) = alg.encapsulate(&ek, &mut rng).unwrap();

                    let encoded = $ser(&ek).unwrap();
                    let decoded: EncapsulationKey = $de(&encoded).unwrap();
                    assert_eq!(decoded, ek);

                    let encoded = $ser(&dk).unwrap();
                    let decoded: DecapsulationKey = $de(&encoded).unwrap();
                    assert_eq!(decoded, dk);

                    let encoded = $ser(&ct).unwrap();
                    let decoded: Ciphertext = $de(&encoded).unwrap();
                    assert_eq!(decoded, ct);

                    let encoded = $ser(&ss).unwrap();
                    let decoded: SharedSecret = $de(&encoded).unwrap();
                    assert_eq!(decoded, ss);
                }
            };
        }

        serde_round_trip!(json, serde_json::to_string, serde_json::from_str);
        serde_round_trip!(toml, toml::to_string, toml::from_str);
        serde_round_trip!(yaml, serde_yaml::to_string, serde_yaml::from_str);
        serde_round_trip!(bare, serde_bare::to_vec, serde_bare::from_slice);
        serde_round_trip!(cbor, serde_cbor::to_vec, serde_cbor::from_slice);
        serde_round_trip!(postcard, postcard::to_stdvec, postcard::from_bytes);
    }
}
