/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
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
//! # #[cfg(all(
//! #     feature = "mceliece348864f",
//! #     feature = "keygen",
//! #     feature = "encapsulate",
//! #     feature = "decapsulate"
//! # ))] {
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
//! # #[cfg(feature = "keygen")] {
//! use pq_mceliece::Algorithm;
//!
//! // Any enabled parameter set works; `default` picks the first one.
//! let alg = Algorithm::default();
//! let seed = [0x42u8; 32];
//! let first = alg.generate_keypair_from_seed(seed).unwrap();
//! let second = alg.generate_keypair_from_seed(seed).unwrap();
//! assert_eq!(first, second);
//! # }
//! ```
//!
//! When the parameter set is known at compile time, the [`hazmat`] layer offers the same
//! operations with the parameter set in the type, so a key from one parameter set cannot be
//! passed to another:
//!
//! ```
//! # #[cfg(all(
//! #     feature = "mceliece348864",
//! #     feature = "hazmat",
//! #     feature = "keygen",
//! #     feature = "encapsulate",
//! #     feature = "decapsulate"
//! # ))] {
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
//! Implementations of the [`kem`](https://docs.rs/kem) crate traits live in the [`kem`] module.
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
//! Parameter sets and operations are selected independently.
//!
//! Each parameter set has its own feature, and the `nist`, `iso` and `pc` groups enable the
//! corresponding sets.
//!
//! The three operations are `keygen`, `encapsulate` and `decapsulate`. At least one must be
//! enabled, and a build only carries the code and dependencies the enabled ones reach.
//! Encapsulation is much the lightest: `Encode` is pure bit manipulation over the public key,
//! so an encapsulate-only build contains no field arithmetic, no polynomial arithmetic, no
//! matrix reduction, no sorting networks and no Beneš network.
//!
//! | feature | effect |
//! | ------- | ------ |
//! | `keygen` | `KeyGen` and `SeededKeyGen`, and recovering a public key from a private one |
//! | `encapsulate` | `Encap` |
//! | `decapsulate` | `Decap` |
//! | `kem` | the [`kem`](https://docs.rs/kem) crate traits; implies all three operations |
//! | `serde` | serialization for every value type |
//! | `hazmat` | makes the low-level layer public |
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

#[cfg(not(any(feature = "keygen", feature = "encapsulate", feature = "decapsulate")))]
compile_error!(
    "no Classic McEliece operation feature is enabled; \
     select at least one of `keygen`, `encapsulate` or `decapsulate`"
);

/// Compiles the README's examples as doctests so they cannot drift from the API.
///
/// Only exists under `cargo test --doc`, so it adds nothing to the rendered documentation or
/// to a normal build. Gated on everything the examples name, so that a build without those
/// features skips them rather than failing on code it was never going to compile. That keeps
/// the feature gating in here instead of scattered through the README, where hidden doctest
/// lines would still be visible to anyone reading it on the repository page.
#[cfg(all(
    doctest,
    feature = "hazmat",
    feature = "keygen",
    feature = "encapsulate",
    feature = "decapsulate",
    feature = "mceliece348864f",
    feature = "mceliece6960119f",
    feature = "mceliece8192128f"
))]
#[doc = include_str!("../README.md")]
pub struct ReadmeExamples;

mod error;
pub use error::*;

#[cfg(feature = "kem")]
pub mod kem;

#[cfg(feature = "hazmat")]
pub mod hazmat;
#[cfg(not(feature = "hazmat"))]
mod hazmat;

use ctutils::{Choice, CtEq};
#[cfg(feature = "keygen")]
use hazmat::Kem;
use hazmat::Params;
#[cfg(any(feature = "keygen", feature = "encapsulate"))]
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
        $name:ident, $from_method:ident, secret = $secret:literal, length = $length:ident
    ) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name {
            pub(crate) algorithm: Algorithm,
            pub(crate) value: Vec<u8>,
        }

        impl Default for $name {
            /// An all-zero value of the correct length for [`Algorithm::default`].
            ///
            /// A derived `Default` would produce an empty value that violates the length
            /// invariant every constructor enforces, so this builds one that upholds it.
            fn default() -> Self {
                let algorithm = Algorithm::default();
                let value = with_params!(&algorithm, P, { vec![0u8; P::$length] });
                Self { algorithm, value }
            }
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
    EncapsulationKey, encapsulation_key_from_bytes, secret = false, length = PUBLIC_KEY_LENGTH
}

value_type! {
    /// A Classic McEliece decapsulation (private) key.
    DecapsulationKey, decapsulation_key_from_bytes, secret = true, length = SECRET_KEY_LENGTH
}

value_type! {
    /// A Classic McEliece ciphertext.
    Ciphertext, ciphertext_from_bytes, secret = false, length = CIPHERTEXT_LENGTH
}

value_type! {
    /// A Classic McEliece shared secret.
    SharedSecret, shared_secret_from_bytes, secret = true, length = SHARED_SECRET_LENGTH
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

#[cfg(feature = "encapsulate")]
impl EncapsulationKey {
    /// Whether every padding bit is zero.
    ///
    /// [`encapsulate`](Self::encapsulate) rejects a key that fails this check. Parameter sets
    /// whose `k` is a multiple of eight have no padding, so this is always `true` for them.
    pub fn padding_is_zero(&self) -> bool {
        with_params!(&self.algorithm, P, {
            match hazmat::EncapsulationKey::<P>::from_slice(&self.value) {
                Ok(key) => key.padding_is_zero(),
                Err(_) => false,
            }
        })
    }

    /// Encapsulate to this key, producing a ciphertext and the shared secret it transports.
    pub fn encapsulate(&self, rng: impl CryptoRng) -> McElieceResult<(Ciphertext, SharedSecret)> {
        self.algorithm.encapsulate(self, rng)
    }
}

#[cfg(feature = "keygen")]
impl From<&DecapsulationKey> for EncapsulationKey {
    /// Recover the encapsulation key by rerunning key generation from the stored seed.
    ///
    /// This costs a full key generation.
    fn from(secret: &DecapsulationKey) -> Self {
        with_params!(&secret.algorithm, P, {
            match hazmat::DecapsulationKey::<P>::from_slice(&secret.value) {
                Ok(typed) => EncapsulationKey {
                    algorithm: secret.algorithm,
                    value: hazmat::EncapsulationKey::<P>::from(&typed).into_vec(),
                },
                // Every constructor, including `Default`, upholds the length invariant for
                // the key's own parameter set, so this arm is unreachable; produce the
                // default rather than panic.
                Err(_) => EncapsulationKey::default(),
            }
        })
    }
}

#[cfg(feature = "decapsulate")]
impl Ciphertext {
    /// Whether every padding bit is zero.
    ///
    /// [`DecapsulationKey::decapsulate`] rejects a ciphertext that fails this check. Parameter
    /// sets whose `mt` is a multiple of eight have no padding, so this is always `true` for
    /// them.
    pub fn padding_is_zero(&self) -> bool {
        with_params!(&self.algorithm, P, {
            match hazmat::Ciphertext::<P>::from_slice(&self.value) {
                Ok(ct) => ct.padding_is_zero(),
                Err(_) => false,
            }
        })
    }
}

impl DecapsulationKey {
    /// The 32-byte seed this key was generated from.
    ///
    /// Knowing the seed is equivalent to knowing the whole private key, so treat it with the
    /// same care. Storing the seed instead of the key trades a full key generation for a
    /// factor of several hundred in storage.
    pub fn seed(&self) -> &[u8] {
        with_params!(&self.algorithm, P, { hazmat::seed_of::<P>(&self.value) })
    }
}

#[cfg(feature = "decapsulate")]
impl DecapsulationKey {
    /// Recover the shared secret from a ciphertext.
    ///
    /// See [`Algorithm::decapsulate`] for what happens when the ciphertext does not decode.
    pub fn decapsulate(&self, ciphertext: &Ciphertext) -> McElieceResult<SharedSecret> {
        self.algorithm.decapsulate(self, ciphertext)
    }

    /// Precompute the key-derived part of decoding, for repeated decapsulation.
    ///
    /// About a third of a decapsulation depends only on the private key, not on the message.
    /// Doing it once here rather than once per ciphertext is worth roughly a third off each
    /// subsequent [`PreparedDecapsulationKey::decapsulate`], at the cost of holding the derived
    /// material alongside the key. Decapsulating a single message is not worth preparing for.
    pub fn prepare(&self) -> PreparedDecapsulationKey {
        let (scale, valid) = with_params!(&self.algorithm, P, {
            let mut scale = vec![0; hazmat::decap::scale_words::<P>()];
            let mut valid = vec![0u8; hazmat::decap::valid_bytes::<P>()];
            hazmat::decap::prepare::<P>(&self.value, &mut scale, &mut valid);
            (scale, valid)
        });
        PreparedDecapsulationKey {
            key: self.clone(),
            scale,
            valid,
        }
    }
}

/// A [`DecapsulationKey`] with the message-independent part of decoding already computed.
///
/// Produced by [`DecapsulationKey::prepare`]. The derived material is a function of the private
/// key alone and is as sensitive as the key itself, so it is zeroized on drop and never
/// serialized: store and transmit the [`DecapsulationKey`] and prepare again on the far side.
#[cfg(feature = "decapsulate")]
pub struct PreparedDecapsulationKey {
    key: DecapsulationKey,
    scale: Vec<hazmat::decap::PreparedWord>,
    valid: Vec<u8>,
}

#[cfg(feature = "decapsulate")]
impl PreparedDecapsulationKey {
    /// The parameter set this key belongs to.
    pub fn algorithm(&self) -> Algorithm {
        self.key.algorithm
    }

    /// The key this was prepared from.
    pub fn key(&self) -> &DecapsulationKey {
        &self.key
    }

    /// Recover the shared secret from a ciphertext.
    ///
    /// Identical in result to [`DecapsulationKey::decapsulate`], including which ciphertexts are
    /// rejected and the session key produced for those that fail to decode.
    pub fn decapsulate(&self, ciphertext: &Ciphertext) -> McElieceResult<SharedSecret> {
        let algorithm = self.key.algorithm;
        if ciphertext.algorithm != algorithm {
            return Err(Error::AlgorithmMismatch);
        }
        with_params!(&algorithm, P, {
            if ciphertext.value.len() != P::CIPHERTEXT_LENGTH {
                return Err(Error::InvalidCiphertextLength(ciphertext.value.len()));
            }
            let mut ss = vec![0u8; P::SHARED_SECRET_LENGTH];
            if !hazmat::decap::decapsulate_prepared::<P>(
                &mut ss,
                &ciphertext.value,
                &self.key.value,
                &self.scale,
                &self.valid,
            ) {
                return Err(Error::CiphertextPadding);
            }
            Ok(SharedSecret {
                algorithm,
                value: ss,
            })
        })
    }
}

#[cfg(feature = "decapsulate")]
impl Drop for PreparedDecapsulationKey {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.scale.zeroize();
        self.valid.zeroize();
    }
}

#[cfg(feature = "decapsulate")]
impl core::fmt::Debug for PreparedDecapsulationKey {
    /// Deliberately opaque: the derived material is as sensitive as the private key.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreparedDecapsulationKey")
            .field("algorithm", &self.key.algorithm)
            .finish_non_exhaustive()
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
    #[cfg(feature = "keygen")]
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
    #[cfg(feature = "keygen")]
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
    #[cfg(feature = "encapsulate")]
    pub fn encapsulate(
        &self,
        key: &EncapsulationKey,
        rng: impl CryptoRng,
    ) -> McElieceResult<(Ciphertext, SharedSecret)> {
        if key.algorithm != *self {
            return Err(Error::AlgorithmMismatch);
        }
        with_params!(self, P, {
            if key.value.len() != P::PUBLIC_KEY_LENGTH {
                return Err(Error::InvalidEncapsulationKeyLength(key.value.len()));
            }
            let mut ct = vec![0u8; P::CIPHERTEXT_LENGTH];
            let mut ss = vec![0u8; P::SHARED_SECRET_LENGTH];
            if !hazmat::encap::encapsulate::<P>(&mut ct, &mut ss, &key.value, rng) {
                return Err(Error::EncapsulationKeyPadding);
            }
            Ok((
                Ciphertext {
                    algorithm: *self,
                    value: ct,
                },
                SharedSecret {
                    algorithm: *self,
                    value: ss,
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
    #[cfg(feature = "decapsulate")]
    pub fn decapsulate(
        &self,
        key: &DecapsulationKey,
        ciphertext: &Ciphertext,
    ) -> McElieceResult<SharedSecret> {
        if key.algorithm != *self || ciphertext.algorithm != *self {
            return Err(Error::AlgorithmMismatch);
        }
        with_params!(self, P, {
            if key.value.len() != P::SECRET_KEY_LENGTH {
                return Err(Error::InvalidDecapsulationKeyLength(key.value.len()));
            }
            if ciphertext.value.len() != P::CIPHERTEXT_LENGTH {
                return Err(Error::InvalidCiphertextLength(ciphertext.value.len()));
            }
            let mut ss = vec![0u8; P::SHARED_SECRET_LENGTH];
            if !hazmat::decap::decapsulate::<P>(&mut ss, &ciphertext.value, &key.value) {
                return Err(Error::CiphertextPadding);
            }
            Ok(SharedSecret {
                algorithm: *self,
                value: ss,
            })
        })
    }

    /// Recover an encapsulation key from a decapsulation key.
    ///
    /// The key must belong to this parameter set. Parameter sets within a size family share
    /// their private-key length, so without this check a key from a sibling set would rerun
    /// key generation under the wrong variant and yield a public key that does not match the
    /// private key.
    #[cfg(feature = "keygen")]
    pub fn encapsulation_key_from_decapsulation_key(
        &self,
        key: &DecapsulationKey,
    ) -> McElieceResult<EncapsulationKey> {
        if key.algorithm != *self {
            return Err(Error::AlgorithmMismatch);
        }
        Ok(EncapsulationKey::from(key))
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

#[cfg(all(
    test,
    feature = "keygen",
    feature = "encapsulate",
    feature = "decapsulate"
))]
#[allow(clippy::unwrap_used)]
mod tests {

    /// A prepared key must be indistinguishable from the key it came from, including on the
    /// implicit-rejection path where a corrupted ciphertext still yields a reproducible secret.
    #[cfg(all(feature = "keygen", feature = "encapsulate", feature = "decapsulate"))]
    #[test]
    fn prepared_decapsulation_matches_the_plain_path() {
        use rand_core::{SeedableRng, UnwrapErr};

        for &algorithm in Algorithm::enabled_algorithms() {
            let mut rng = UnwrapErr(rand_chacha::ChaCha8Rng::from_seed([29u8; 32]));
            let (ek, dk) = algorithm.generate_keypair(&mut rng);
            let prepared = dk.prepare();
            assert_eq!(prepared.algorithm(), algorithm);
            assert_eq!(prepared.key(), &dk);

            let (ct, sent) = algorithm
                .encapsulate(&ek, &mut rng)
                .expect("a freshly generated key encapsulates");
            let direct = dk.decapsulate(&ct).expect("the ciphertext is well formed");
            let via_prepared = prepared.decapsulate(&ct).expect("likewise");
            assert_eq!(direct, sent, "{}", algorithm.name());
            assert_eq!(via_prepared, direct, "{} prepared", algorithm.name());

            // Implicit rejection: flipping a syndrome bit must fail to decode, and both paths
            // must still agree on the substitute secret.
            let mut damaged = ct.clone();
            damaged.value[0] ^= 1;
            let direct = dk.decapsulate(&damaged);
            let via_prepared = prepared.decapsulate(&damaged);
            assert_eq!(direct, via_prepared, "{} rejected", algorithm.name());
            if let Ok(secret) = direct {
                assert_ne!(secret, sent, "{} rejection", algorithm.name());
            }

            // A ciphertext from another parameter set must be refused, not decoded.
            let other = Algorithm::enabled_algorithms()
                .iter()
                .find(|&&a| a != algorithm)
                .copied();
            if let Some(other) = other {
                let (other_ek, _) = other.generate_keypair(&mut rng);
                let (other_ct, _) = other
                    .encapsulate(&other_ek, &mut rng)
                    .expect("a freshly generated key encapsulates");
                assert!(matches!(
                    prepared.decapsulate(&other_ct),
                    Err(Error::AlgorithmMismatch)
                ));
            }
        }
    }
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
    fn the_dynamic_api_rejects_default_values_without_panicking() {
        // `Default` values are well-formed all-zero values of the correct length, so every
        // operation completes without panicking: encapsulating to the zero key succeeds, and
        // decapsulating with the zero key degrades to implicit rejection. `seed` and
        // `prepare` used to panic on the old empty-value `Default`; they must not regress.
        let algorithm = Algorithm::default();
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([6u8; 32]);
        let (ct, _ss) = algorithm
            .encapsulate(&EncapsulationKey::default(), &mut rng)
            .unwrap();

        let key = DecapsulationKey::default();
        assert_eq!(key.seed(), [0u8; 32]);
        let plain = algorithm.decapsulate(&key, &ct).unwrap();
        let prepared = key.prepare();
        assert_eq!(prepared.decapsulate(&ct).unwrap(), plain);
        assert!(algorithm.decapsulate(&key, &Ciphertext::default()).is_ok());

        // Hand-assembled wrong-length values still produce errors rather than panics.
        let empty = Ciphertext {
            algorithm,
            value: Vec::new(),
        };
        assert_eq!(
            algorithm.decapsulate(&key, &empty).unwrap_err(),
            Error::InvalidCiphertextLength(0)
        );
    }

    #[test]
    #[cfg(all(
        feature = "keygen",
        feature = "mceliece348864",
        feature = "mceliece348864f"
    ))]
    fn public_key_recovery_rejects_a_key_from_a_sibling_parameter_set() {
        // The `f` and non-`f` variants share their private-key length, so without the
        // algorithm check this would rerun key generation under the wrong variant and
        // silently return a public key that does not match the private key.
        let algorithm = Algorithm::McEliece348864;
        let sibling = DecapsulationKey::from_bytes(
            Algorithm::McEliece348864f,
            vec![0; Algorithm::McEliece348864f.params().decapsulation_key_length],
        )
        .unwrap();
        assert_eq!(
            algorithm
                .encapsulation_key_from_decapsulation_key(&sibling)
                .unwrap_err(),
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

    /// The padding accessors are how a caller inspects the *narrowly decoded* rejection before
    /// it happens, and `mceliece6960119` is the one standardized set that has padding at all.
    #[cfg(all(feature = "keygen", feature = "encapsulate"))]
    #[test]
    fn padding_accessors_report_both_outcomes() {
        for &algorithm in Algorithm::enabled_algorithms() {
            let mut rng = rand_chacha::ChaCha8Rng::from_seed([23u8; 32]);
            let (ek, _) = algorithm.generate_keypair(&mut rng);
            let (ct, _) = algorithm.encapsulate(&ek, &mut rng).unwrap();

            assert!(ek.padding_is_zero(), "{} fresh key", algorithm.name());
            assert!(
                ct.padding_is_zero(),
                "{} fresh ciphertext",
                algorithm.name()
            );

            // Setting every padding bit is only detectable where padding exists.
            let has_padding = algorithm.params().k % 8 != 0;
            let mut damaged = ek.clone();
            let last = damaged.value.len() - 1;
            damaged.value[last] = 0xFF;
            assert_eq!(
                damaged.padding_is_zero(),
                !has_padding,
                "{} damaged key",
                algorithm.name()
            );

            // The padding sits at the end of the syndrome, which for a `pc` set is not the end
            // of the ciphertext: the plaintext-confirmation hash follows it.
            let params = algorithm.params();
            let syndrome_bytes = (params.m * params.t).div_ceil(8);
            let ciphertext_has_padding = params.m * params.t % 8 != 0;
            let mut damaged = ct.clone();
            damaged.value[syndrome_bytes - 1] = 0xFF;
            assert_eq!(
                damaged.padding_is_zero(),
                !ciphertext_has_padding,
                "{} damaged ciphertext",
                algorithm.name()
            );

            // A value of the wrong length belongs to no parameter set and cannot be inspected.
            let mut truncated = ek;
            truncated.value.truncate(1);
            assert!(
                !truncated.padding_is_zero(),
                "{} short key",
                algorithm.name()
            );
            let mut truncated = ct;
            truncated.value.truncate(1);
            assert!(
                !truncated.padding_is_zero(),
                "{} short ciphertext",
                algorithm.name()
            );
        }
    }

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        /// Deserialization is the crate's exposure to input it did not produce, so the paths
        /// that reject malformed encodings need exercising as much as the round trip does.
        #[cfg(all(feature = "keygen", feature = "encapsulate"))]
        #[test]
        fn deserialization_rejects_malformed_encodings() {
            let alg = Algorithm::enabled_algorithms()[0];
            let mut rng = rand_chacha::ChaCha8Rng::from_seed([17u8; 32]);
            let (ek, _) = alg.generate_keypair(&mut rng);
            let encoded = serde_json::to_string(&ek).unwrap();

            // A repeated field must be refused rather than silently taking one of the two.
            let doubled_algorithm = encoded.replacen('{', r#"{"algorithm":"mceliece348864","#, 1);
            assert!(serde_json::from_str::<EncapsulationKey>(&doubled_algorithm).is_err());
            let doubled_value = encoded.replacen('{', r#"{"value":"00","#, 1);
            assert!(serde_json::from_str::<EncapsulationKey>(&doubled_value).is_err());

            // So must a missing field, an unknown one, and a value that is not a hex string.
            assert!(serde_json::from_str::<EncapsulationKey>(r#"{"value":"00"}"#).is_err());
            assert!(
                serde_json::from_str::<EncapsulationKey>(&encoded.replacen(
                    '{',
                    r#"{"nope":1,"#,
                    1
                ))
                .is_err()
            );
            assert!(
                serde_json::from_str::<EncapsulationKey>(
                    r#"{"algorithm":"mceliece348864","value":"zz"}"#
                )
                .is_err()
            );

            // A shape that is not a map at all, and a value of the wrong length for its set.
            assert!(serde_json::from_str::<EncapsulationKey>("[]").is_err());
            assert!(
                serde_json::from_str::<EncapsulationKey>(
                    r#"{"algorithm":"mceliece348864","value":"00ff"}"#
                )
                .is_err()
            );
        }

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
