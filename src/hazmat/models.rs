/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Statically typed keys, ciphertexts and shared secrets, and the [`Kem`] trait over them.
//!
//! Each type is parameterized by the parameter set it belongs to, so mixing values from two
//! parameter sets is a compile error rather than a runtime check. The dynamic [`Algorithm`]
//! API in the crate root wraps these when the parameter set is only known at run time.
//!
//! [`Algorithm`]: crate::Algorithm

use core::marker::PhantomData;

use ctutils::{Choice, CtEq};
#[cfg(any(feature = "keygen", all(feature = "hazmat", feature = "encapsulate")))]
use rand_core::CryptoRng;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(feature = "decapsulate")]
use super::decap::ciphertext_padding_is_zero;
#[cfg(all(feature = "decapsulate", any(feature = "hazmat", feature = "kem")))]
use super::decap::decapsulate;
#[cfg(all(feature = "encapsulate", any(feature = "hazmat", feature = "kem")))]
use super::encap::encapsulate;
#[cfg(feature = "encapsulate")]
use super::encap::public_key_padding_is_zero;
#[cfg(feature = "keygen")]
use super::keygen::{public_key_from_secret_key, seeded_keypair};
use super::params::Params;
#[cfg_attr(
    not(any(feature = "keygen", feature = "encapsulate", feature = "decapsulate")),
    allow(unused_imports)
)]
use crate::Error;
use crate::McElieceResult;

macro_rules! byte_value {
    (
        $(#[$meta:meta])*
        $name:ident, $length:ident, $error:ident,
        secret = $secret:literal, constructed_when = $constructed:meta
    ) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name<P: Params> {
            pub(crate) bytes: Vec<u8>,
            pub(crate) params: PhantomData<P>,
        }

        impl<P: Params> $name<P> {
            /// The byte length of this value for parameter set `P`.
            pub const LENGTH: usize = P::$length;

            /// Parse a value from bytes, checking the length.
            ///
            /// The length is the only thing validated here. A key assembled from arbitrary
            /// bytes is structurally unconstrained; using one degrades safely (decapsulation
            /// falls back to implicit rejection) but produces no useful result.
            pub fn from_slice(bytes: &[u8]) -> McElieceResult<Self> {
                if bytes.len() != Self::LENGTH {
                    return Err(Error::$error(bytes.len()));
                }
                Ok(Self {
                    bytes: bytes.to_vec(),
                    params: PhantomData,
                })
            }

            /// Consume the value and return its bytes.
            ///
            /// The bytes are moved out rather than copied, so a megabyte-scale encapsulation
            /// key does not get duplicated on the way to the caller.
            ///
            /// For secret types this moves the bytes out of the zeroize-on-drop wrapper: the
            /// caller now owns a plain `Vec` and is responsible for scrubbing it.
            pub fn into_vec(mut self) -> Vec<u8> {
                core::mem::take(&mut self.bytes)
            }

            #[cfg($constructed)]
            pub(crate) fn zeroed() -> Self {
                Self {
                    bytes: vec![0u8; P::$length],
                    params: PhantomData,
                }
            }
        }

        impl<P: Params> AsRef<[u8]> for $name<P> {
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }

        impl<P: Params> CtEq for $name<P> {
            fn ct_eq(&self, other: &Self) -> Choice {
                self.bytes.as_slice().ct_eq(other.bytes.as_slice())
            }
        }

        impl<P: Params> PartialEq for $name<P> {
            fn eq(&self, other: &Self) -> bool {
                bool::from(self.ct_eq(other))
            }
        }

        impl<P: Params> Eq for $name<P> {}

        impl<P: Params> core::fmt::Debug for $name<P> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut out = f.debug_struct(stringify!($name));
                out.field("algorithm", &P::NAME);
                if $secret {
                    out.finish_non_exhaustive()
                } else {
                    out.field("bytes", &self.bytes).finish()
                }
            }
        }
    };
}

byte_value! {
    /// A Classic McEliece encapsulation (public) key, the matrix `T`.
    EncapsulationKey, PUBLIC_KEY_LENGTH, InvalidEncapsulationKeyLength, secret = false, constructed_when = feature = "keygen"
}

byte_value! {
    /// A Classic McEliece decapsulation (private) key, `(delta, c, g, alpha, s)`.
    DecapsulationKey, SECRET_KEY_LENGTH, InvalidDecapsulationKeyLength, secret = true, constructed_when = feature = "keygen"
}

byte_value! {
    /// A Classic McEliece ciphertext.
    Ciphertext, CIPHERTEXT_LENGTH, InvalidCiphertextLength, secret = false, constructed_when = all(feature = "encapsulate", any(feature = "hazmat", feature = "kem"))
}

byte_value! {
    /// A Classic McEliece shared secret.
    SharedSecret, SHARED_SECRET_LENGTH, InvalidSharedSecretLength, secret = true, constructed_when = any(all(feature = "encapsulate", any(feature = "hazmat", feature = "kem")), all(feature = "decapsulate", any(feature = "hazmat", feature = "kem")))
}

macro_rules! zeroize_on_drop {
    ($name:ident) => {
        impl<P: Params> Zeroize for $name<P> {
            fn zeroize(&mut self) {
                self.bytes.zeroize();
            }
        }

        impl<P: Params> ZeroizeOnDrop for $name<P> {}

        impl<P: Params> Drop for $name<P> {
            fn drop(&mut self) {
                self.zeroize();
            }
        }
    };
}

zeroize_on_drop!(DecapsulationKey);
zeroize_on_drop!(SharedSecret);

#[cfg(feature = "encapsulate")]
impl<P: Params> EncapsulationKey<P> {
    /// Whether every padding bit is zero.
    ///
    /// A key that fails this check is rejected by [`Kem::encapsulate`]. Parameter sets whose
    /// `k` is a multiple of eight have no padding, so this is always `true` for them.
    pub fn padding_is_zero(&self) -> bool {
        public_key_padding_is_zero::<P>(&self.bytes)
    }
}

/// The seed `delta` stored at the front of a private key.
///
/// Shared by the typed and dynamic APIs so that the layout is described in exactly one place.
pub(crate) fn seed_of<P: Params>(private_key: &[u8]) -> &[u8] {
    &private_key[..P::SEED_LENGTH]
}

// The dynamic API reads the seed through `seed_of`, so this accessor only exists where the
// typed value itself is reachable.
#[cfg(feature = "hazmat")]
impl<P: Params> DecapsulationKey<P> {
    /// The 32-byte seed `delta` this key was generated from.
    ///
    /// Knowing `delta` is equivalent to knowing the whole private key, so treat it with the
    /// same care.
    pub fn seed(&self) -> &[u8] {
        seed_of::<P>(&self.bytes)
    }
}

#[cfg(feature = "keygen")]
impl<P: Params> From<&DecapsulationKey<P>> for EncapsulationKey<P> {
    /// Recover the encapsulation key by rerunning key generation from the stored seed.
    ///
    /// This costs a full key generation. Keep the encapsulation key around instead if you
    /// need it more than once.
    fn from(secret: &DecapsulationKey<P>) -> Self {
        let mut public = Self::zeroed();
        public_key_from_secret_key::<P>(&secret.bytes, &mut public.bytes);
        public
    }
}

#[cfg(feature = "keygen")]
pub(crate) fn generate_keypair_from_seed_array<P: Params>(
    seed: &[u8; 32],
) -> (EncapsulationKey<P>, DecapsulationKey<P>) {
    let mut public = EncapsulationKey::<P>::zeroed();
    let mut secret = DecapsulationKey::<P>::zeroed();
    seeded_keypair::<P>(&mut public.bytes, &mut secret.bytes, seed);
    (public, secret)
}

#[cfg(feature = "decapsulate")]
impl<P: Params> Ciphertext<P> {
    /// Whether every padding bit is zero.
    ///
    /// A ciphertext that fails this check is rejected by [`Kem::decapsulate`].
    pub fn padding_is_zero(&self) -> bool {
        ciphertext_padding_is_zero::<P>(&self.bytes)
    }
}

/// The Classic McEliece key encapsulation mechanism for one parameter set.
///
/// This trait is blanket implemented for every type implementing [`Params`], so the parameter
/// set marker itself is the entry point: `McEliece6960119f::generate_keypair(rng)`.
#[cfg(any(feature = "keygen", feature = "hazmat", feature = "kem"))]
pub trait Kem: Params {
    /// `SeededKeyGen`: derive a key pair deterministically from a 32-byte seed.
    ///
    /// Requires the `keygen` feature.
    ///
    /// The same seed always yields the same key pair, which makes key generation reproducible
    /// for testing and for deriving a long-term key from an existing secret. The seed must be
    /// generated with a cryptographically secure source; it is exactly as sensitive as the
    /// private key.
    #[cfg(feature = "keygen")]
    fn generate_keypair_from_seed(
        seed: &[u8],
    ) -> McElieceResult<(EncapsulationKey<Self>, DecapsulationKey<Self>)> {
        if seed.len() != Self::SEED_LENGTH {
            return Err(Error::InvalidSeedLength(seed.len()));
        }
        let mut delta = [0u8; 32];
        delta.copy_from_slice(seed);

        let pair = generate_keypair_from_seed_array::<Self>(&delta);

        delta.zeroize();
        Ok(pair)
    }

    /// `KeyGen`: draw a fresh seed from `rng` and generate a key pair from it.
    ///
    /// Requires the `keygen` feature.
    #[cfg(feature = "keygen")]
    fn generate_keypair(
        mut rng: impl CryptoRng,
    ) -> (EncapsulationKey<Self>, DecapsulationKey<Self>) {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);

        let mut public = EncapsulationKey::<Self>::zeroed();
        let mut secret = DecapsulationKey::<Self>::zeroed();
        seeded_keypair::<Self>(&mut public.bytes, &mut secret.bytes, &seed);

        seed.zeroize();
        (public, secret)
    }

    /// `Encap`: produce a ciphertext and the shared secret it transports.
    ///
    /// Returns [`Error::EncapsulationKeyPadding`] when `key` has nonzero padding bits.
    ///
    /// Requires the `encapsulate` feature.
    #[cfg(all(feature = "encapsulate", any(feature = "hazmat", feature = "kem")))]
    fn encapsulate(
        key: &EncapsulationKey<Self>,
        rng: impl CryptoRng,
    ) -> McElieceResult<(Ciphertext<Self>, SharedSecret<Self>)> {
        let mut ciphertext = Ciphertext::<Self>::zeroed();
        let mut shared = SharedSecret::<Self>::zeroed();

        if !encapsulate::<Self>(&mut ciphertext.bytes, &mut shared.bytes, &key.bytes, rng) {
            return Err(Error::EncapsulationKeyPadding);
        }
        Ok((ciphertext, shared))
    }

    /// `Decap`: recover the shared secret from a ciphertext.
    ///
    /// A ciphertext that does not decode does not produce an error. Classic McEliece rejects
    /// implicitly: the returned secret is derived from the private key's rejection string, so
    /// it is unpredictable to anyone without the private key and identical every time the same
    /// bad ciphertext is presented. Distinguishing the two cases from the outside is exactly
    /// what the construction is designed to prevent.
    ///
    /// The one condition that is reported is [`Error::CiphertextPadding`], which depends only
    /// on public data.
    ///
    /// Requires the `decapsulate` feature.
    #[cfg(all(feature = "decapsulate", any(feature = "hazmat", feature = "kem")))]
    fn decapsulate(
        key: &DecapsulationKey<Self>,
        ciphertext: &Ciphertext<Self>,
    ) -> McElieceResult<SharedSecret<Self>> {
        let mut shared = SharedSecret::<Self>::zeroed();
        if !decapsulate::<Self>(&mut shared.bytes, &ciphertext.bytes, &key.bytes) {
            return Err(Error::CiphertextPadding);
        }
        Ok(shared)
    }
}

#[cfg(any(feature = "keygen", feature = "hazmat", feature = "kem"))]
impl<P: Params> Kem for P {}

#[cfg(all(
    test,
    feature = "keygen",
    feature = "encapsulate",
    feature = "decapsulate"
))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[cfg(any(feature = "hazmat", feature = "kem"))]
    use rand_core::SeedableRng;

    /// The typed constructors and `Kem` methods only exist when the low-level layer or the
    /// `kem` traits are exposed, so these need more than the three operation features.
    #[cfg(any(feature = "hazmat", feature = "kem"))]
    fn typed_api_round_trips<P: Params>(seed: u8) {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed; 32]);
        let (ek, dk) = P::generate_keypair(&mut rng);

        assert_eq!(ek.as_ref().len(), P::PUBLIC_KEY_LENGTH);
        assert_eq!(dk.as_ref().len(), P::SECRET_KEY_LENGTH);
        assert!(ek.padding_is_zero());

        let (ct, sent) = P::encapsulate(&ek, &mut rng).unwrap();
        assert_eq!(ct.as_ref().len(), P::CIPHERTEXT_LENGTH);
        assert!(ct.padding_is_zero());

        let received = P::decapsulate(&dk, &ct).unwrap();
        assert_eq!(sent, received);

        // Reparsing through bytes must preserve everything.
        let ek2 = EncapsulationKey::<P>::from_slice(ek.as_ref()).unwrap();
        let dk2 = DecapsulationKey::<P>::from_slice(dk.as_ref()).unwrap();
        let ct2 = Ciphertext::<P>::from_slice(ct.as_ref()).unwrap();
        assert_eq!(ek, ek2);
        assert_eq!(dk, dk2);
        assert_eq!(ct, ct2);
        assert_eq!(P::decapsulate(&dk2, &ct2).unwrap(), sent);

        // The public key is recoverable from the private key.
        assert_eq!(EncapsulationKey::<P>::from(&dk), ek);
        #[cfg(feature = "hazmat")]
        assert_eq!(dk.seed().len(), 32);

        // Wrong lengths are rejected.
        assert!(EncapsulationKey::<P>::from_slice(&[]).is_err());
        assert!(DecapsulationKey::<P>::from_slice(&[]).is_err());
        assert!(Ciphertext::<P>::from_slice(&[]).is_err());
        assert!(SharedSecret::<P>::from_slice(&[]).is_err());
        assert!(P::generate_keypair_from_seed(&[0u8; 31]).is_err());
        assert!(P::generate_keypair_from_seed(&[0u8; 33]).is_err());

        // Secrets must not print their contents.
        assert!(!format!("{dk:?}").contains("bytes"));
        assert!(!format!("{sent:?}").contains("bytes"));
    }

    fn deterministic_generation_is_reproducible<P: Params>() {
        let seed = [0xC3u8; 32];
        let (ek1, dk1) = P::generate_keypair_from_seed(&seed).unwrap();
        let (ek2, dk2) = P::generate_keypair_from_seed(&seed).unwrap();
        assert_eq!(ek1, ek2);
        assert_eq!(dk1, dk2);
    }

    #[cfg(any(feature = "hazmat", feature = "kem"))]
    fn padding_is_reported<P: Params>() {
        if !P::PUBLIC_KEY_HAS_PADDING {
            return;
        }
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([9u8; 32]);
        let mut bad = vec![0u8; P::PUBLIC_KEY_LENGTH];
        bad[P::PK_ROW_BYTES - 1] = 1 << (P::PK_NCOLS % 8);
        let ek = EncapsulationKey::<P>::from_slice(&bad).unwrap();
        assert!(!ek.padding_is_zero());
        assert_eq!(
            P::encapsulate(&ek, &mut rng).unwrap_err(),
            Error::EncapsulationKeyPadding
        );

        let mut ct_bytes = vec![0u8; P::CIPHERTEXT_LENGTH];
        ct_bytes[P::SYND_BYTES - 1] = 1 << (P::PK_NROWS % 8);
        let ct = Ciphertext::<P>::from_slice(&ct_bytes).unwrap();
        assert!(!ct.padding_is_zero());
        let (_, dk) = P::generate_keypair(&mut rng);
        assert_eq!(
            P::decapsulate(&dk, &ct).unwrap_err(),
            Error::CiphertextPadding
        );
    }

    macro_rules! model_tests {
        ($($feature:literal => $mod_name:ident, $params:ty, $seed:expr;)+) => {
            $(
                #[cfg(feature = $feature)]
                mod $mod_name {
                    use super::*;
                    use crate::hazmat::params::*;

                    #[test]
                    #[cfg(any(feature = "hazmat", feature = "kem"))]
                    fn round_trips() {
                        typed_api_round_trips::<$params>($seed);
                    }

                    #[test]
                    fn seeded_generation_is_deterministic() {
                        deterministic_generation_is_reproducible::<$params>();
                    }

                    #[test]
                    #[cfg(any(feature = "hazmat", feature = "kem"))]
                    fn padding_errors_surface() {
                        padding_is_reported::<$params>();
                    }
                }
            )+
        };
    }

    model_tests! {
        "mceliece348864" => set_348864, McEliece348864, 0x21;
        "mceliece460896pc" => set_460896pc, McEliece460896pc, 0x22;
        "mceliece6960119f" => set_6960119f, McEliece6960119f, 0x23;
        "mceliece8192128pcf" => set_8192128pcf, McEliece8192128pcf, 0x24;
    }
}
