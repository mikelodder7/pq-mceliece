/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Implementations of the traits from the [`kem`] crate.
//!
//! These wrap the [`hazmat`] layer in the fixed-size array types that the `kem`
//! traits use, so Classic McEliece can be dropped into generic code alongside other KEMs.
//!
//! # Stack usage
//!
//! Classic McEliece encapsulation keys are far larger than the values these traits were shaped
//! for, and [`KeyExport::to_bytes`] returns one **by value**:
//!
//! | parameter set | encapsulation key |
//! | ------------- | ----------------- |
//! | `mceliece348864` | 261,120 bytes |
//! | `mceliece460896` | 524,160 bytes |
//! | `mceliece6688128` | 1,044,992 bytes |
//! | `mceliece6960119` | 1,047,319 bytes |
//! | `mceliece8192128` | 1,357,824 bytes |
//!
//! The signature comes from the trait, so a caller writing `let bytes = key.to_bytes();` holds
//! a megabyte-scale local no matter what this crate does. On Rust's default 2 MiB thread stack
//! that overflows and aborts for the larger sets in an unoptimized build, where the copies a
//! release build elides are all materialized. Importing through [`TryKeyInit::new`] needs the
//! same array to exist somewhere.
//!
//! Encapsulation and decapsulation are unaffected: ciphertexts and shared keys are a few
//! hundred bytes, and the key types themselves are heap-backed handles of a few dozen bytes.
//! Only key import and export move the large arrays.
//!
//! If that is a problem, use [`Algorithm`](crate::Algorithm) or the [`hazmat`] layer instead:
//! both keep key material on the heap and hand out slices. Otherwise, give the thread a stack
//! that comfortably exceeds twice the key size.

use hybrid_array::{Array, ArraySize};
use kem::{
    Ciphertext, Decapsulate, Decapsulator, Encapsulate, Generate, InvalidKey, Key, KeyExport,
    KeySizeUser, SharedKey, TryKeyInit,
};
use rand_core::{CryptoRng, TryCryptoRng};
use zeroize::Zeroizing;

use crate::hazmat;

fn array_from_slice<U: ArraySize>(bytes: &[u8]) -> Array<u8, U> {
    let mut array = Array::default();
    array.copy_from_slice(bytes);
    array
}

/// Compile-time sizes used by the [`kem`] trait implementations.
#[doc(hidden)]
pub trait KemSizes:
    hazmat::Kem + Copy + Clone + core::fmt::Debug + Eq + Ord + Send + Sync + 'static
{
    /// Encapsulation key size.
    type EncapsulationKeySize: ArraySize;
    /// Decapsulation key size.
    type DecapsulationKeySize: ArraySize;
    /// Ciphertext size.
    type CiphertextSize: ArraySize;
    /// Shared key size.
    type SharedKeySize: ArraySize;
}

/// A Classic McEliece encapsulation key for use with the [`kem`] traits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncapsulationKey<K: KemSizes>(hazmat::EncapsulationKey<K>);

/// A Classic McEliece decapsulation key for use with the [`kem`] traits.
///
/// The encapsulation key is kept alongside the private key because [`Decapsulator`] exposes
/// it, and recovering it from the private key alone costs a full key generation.
pub struct DecapsulationKey<K: KemSizes> {
    key: hazmat::DecapsulationKey<K>,
    encapsulation_key: EncapsulationKey<K>,
}

impl<K: KemSizes> core::fmt::Debug for DecapsulationKey<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecapsulationKey")
            .field("algorithm", &K::NAME)
            .finish_non_exhaustive()
    }
}

impl<K: KemSizes> KeySizeUser for EncapsulationKey<K> {
    type KeySize = K::EncapsulationKeySize;
}

impl<K: KemSizes> TryKeyInit for EncapsulationKey<K> {
    fn new(key: &Key<Self>) -> Result<Self, InvalidKey> {
        hazmat::EncapsulationKey::from_slice(key)
            .map(Self)
            .map_err(|_| InvalidKey)
    }
}

impl<K: KemSizes> KeyExport for EncapsulationKey<K> {
    fn to_bytes(&self) -> Key<Self> {
        array_from_slice(self.0.as_ref())
    }
}

impl<K: KemSizes> KeySizeUser for DecapsulationKey<K> {
    type KeySize = K::DecapsulationKeySize;
}

impl<K: KemSizes> TryKeyInit for DecapsulationKey<K> {
    /// Import a private key, rerunning key generation to recover the matching public key.
    ///
    /// This is as expensive as generating a fresh key pair.
    fn new(key: &Key<Self>) -> Result<Self, InvalidKey> {
        let key = hazmat::DecapsulationKey::from_slice(key).map_err(|_| InvalidKey)?;
        let encapsulation_key = EncapsulationKey((&key).into());
        Ok(Self {
            key,
            encapsulation_key,
        })
    }
}

impl<K: KemSizes> KeyExport for DecapsulationKey<K> {
    fn to_bytes(&self) -> Key<Self> {
        array_from_slice(self.key.as_ref())
    }
}

impl<K: KemSizes> Generate for DecapsulationKey<K> {
    fn try_generate_from_rng<R: TryCryptoRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        let mut seed = Zeroizing::new([0u8; 32]);
        rng.try_fill_bytes(seed.as_mut())?;
        let (ek, dk) = hazmat::generate_keypair_from_seed_array::<K>(&seed);
        Ok(Self {
            key: dk,
            encapsulation_key: EncapsulationKey(ek),
        })
    }
}

impl<K> Decapsulator for DecapsulationKey<K>
where
    K: KemSizes + kem::Kem<EncapsulationKey = EncapsulationKey<K>>,
{
    type Kem = K;

    fn encapsulation_key(&self) -> &EncapsulationKey<K> {
        &self.encapsulation_key
    }
}

impl<K> Decapsulate for DecapsulationKey<K>
where
    K: KemSizes + kem::Kem<EncapsulationKey = EncapsulationKey<K>>,
{
    /// Decapsulation never fails observably.
    ///
    /// A ciphertext that does not decode yields a shared key derived from the private key's
    /// rejection string, and one whose padding bits are nonzero yields an all-ones key, both
    /// of which are what the reference implementation produces.
    fn decapsulate(&self, ct: &Ciphertext<K>) -> SharedKey<K> {
        let Ok(ct) = hazmat::Ciphertext::<K>::from_slice(ct.as_slice()) else {
            return SharedKey::<K>::default();
        };
        let shared = match <K as hazmat::Kem>::decapsulate(&self.key, &ct) {
            Ok(shared) => shared.into_vec(),
            Err(_) => vec![0xFFu8; K::SHARED_SECRET_LENGTH],
        };
        array_from_slice(&shared)
    }
}

impl<K> Encapsulate for EncapsulationKey<K>
where
    K: KemSizes + kem::Kem,
{
    type Kem = K;

    /// Encapsulate to this key.
    ///
    /// A key whose padding bits are nonzero produces an all-zero ciphertext and shared key,
    /// which is what the reference implementation produces.
    fn encapsulate_with_rng<R>(&self, rng: &mut R) -> (Ciphertext<K>, SharedKey<K>)
    where
        R: CryptoRng + ?Sized,
    {
        match <K as hazmat::Kem>::encapsulate(&self.0, rng) {
            Ok((ct, shared)) => (
                array_from_slice(ct.as_ref()),
                array_from_slice(shared.as_ref()),
            ),
            Err(_) => (Array::default(), Array::default()),
        }
    }
}

macro_rules! impl_kem {
    ($($feature:literal => $params:ident, $ek:ident, $dk:ident, $ct:ident, $ss:ident;)+) => {
        $(
            #[cfg(feature = $feature)]
            impl KemSizes for hazmat::$params {
                type EncapsulationKeySize = hybrid_array::sizes::$ek;
                type DecapsulationKeySize = hybrid_array::sizes::$dk;
                type CiphertextSize = hybrid_array::sizes::$ct;
                type SharedKeySize = hybrid_array::sizes::$ss;
            }

            #[cfg(feature = $feature)]
            impl kem::Kem for hazmat::$params {
                type DecapsulationKey = DecapsulationKey<Self>;
                type EncapsulationKey = EncapsulationKey<Self>;
                type SharedKeySize = hybrid_array::sizes::$ss;
                type CiphertextSize = hybrid_array::sizes::$ct;
            }
        )+
    };
}

impl_kem! {
    "mceliece348864" => McEliece348864, U261120, U6492, U96, U32;
    "mceliece348864f" => McEliece348864f, U261120, U6492, U96, U32;
    "mceliece460896" => McEliece460896, U524160, U13608, U156, U32;
    "mceliece460896f" => McEliece460896f, U524160, U13608, U156, U32;
    "mceliece460896pc" => McEliece460896pc, U524160, U13608, U188, U32;
    "mceliece460896pcf" => McEliece460896pcf, U524160, U13608, U188, U32;
    "mceliece6688128" => McEliece6688128, U1044992, U13932, U208, U32;
    "mceliece6688128f" => McEliece6688128f, U1044992, U13932, U208, U32;
    "mceliece6688128pc" => McEliece6688128pc, U1044992, U13932, U240, U32;
    "mceliece6688128pcf" => McEliece6688128pcf, U1044992, U13932, U240, U32;
    "mceliece6960119" => McEliece6960119, U1047319, U13948, U194, U32;
    "mceliece6960119f" => McEliece6960119f, U1047319, U13948, U194, U32;
    "mceliece6960119pc" => McEliece6960119pc, U1047319, U13948, U226, U32;
    "mceliece6960119pcf" => McEliece6960119pcf, U1047319, U13948, U226, U32;
    "mceliece8192128" => McEliece8192128, U1357824, U14120, U208, U32;
    "mceliece8192128f" => McEliece8192128f, U1357824, U14120, U208, U32;
    "mceliece8192128pc" => McEliece8192128pc, U1357824, U14120, U240, U32;
    "mceliece8192128pcf" => McEliece8192128pcf, U1357824, U14120, U240, U32;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    fn traits_round_trip<K>(seed: u8)
    where
        K: KemSizes
            + kem::Kem<EncapsulationKey = EncapsulationKey<K>, DecapsulationKey = DecapsulationKey<K>>,
    {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed([seed; 32]);
        let (dk, ek) = K::generate_keypair_from_rng(&mut rng);

        // The private key must not print its contents.
        assert!(!format!("{dk:?}").contains(&hex::encode(dk.key.as_ref())));

        let (ct, sent) = ek.encapsulate_with_rng(&mut rng);
        let received = dk.decapsulate(&ct);
        assert_eq!(sent, received);

        // Round tripping through the byte-array representation preserves both keys.
        let ek_bytes = ek.to_bytes();
        let imported = EncapsulationKey::<K>::new(&ek_bytes).unwrap();
        assert_eq!(imported, ek);
        assert_eq!(dk.encapsulation_key(), &ek);

        let dk_bytes = dk.to_bytes();
        let imported = DecapsulationKey::<K>::new(&dk_bytes).unwrap();
        assert_eq!(imported.decapsulate(&ct), sent);
        assert_eq!(imported.encapsulation_key(), &ek);
    }

    fn traits_round_trip_with_large_stack<K>(seed: u8)
    where
        K: KemSizes
            + kem::Kem<EncapsulationKey = EncapsulationKey<K>, DecapsulationKey = DecapsulationKey<K>>,
    {
        // This round trip exports and reimports both keys, and `KeyExport::to_bytes` hands
        // back a megabyte-scale array by value for the larger parameter sets. See the module
        // documentation's stack usage section: the default thread stack cannot hold one, so
        // the test supplies its own rather than pretending the requirement is not there.
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || traits_round_trip::<K>(seed))
            .unwrap()
            .join()
            .unwrap();
    }

    /// The `kem` traits cannot report failure, so they answer a rejected value with a fixed
    /// substitute instead. `mceliece6960119` is the one standardized set whose `mt` and `k` are
    /// not multiples of eight, so it is the only one where padding bits exist to corrupt.
    #[cfg(feature = "mceliece6960119")]
    #[test]
    fn rejected_values_yield_the_documented_substitutes() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                use kem::Kem as _;
                type K = hazmat::McEliece6960119;

                let mut rng = rand_chacha::ChaCha8Rng::from_seed([41u8; 32]);
                let (dk, ek) = K::generate_keypair_from_rng(&mut rng);
                let (ct, sent) = ek.encapsulate_with_rng(&mut rng);

                // Setting the ciphertext's padding bits must give an all-ones shared key.
                let mut damaged = ct;
                let last = damaged.len() - 1;
                damaged[last] = 0xFF;
                assert_ne!(dk.decapsulate(&damaged), sent);
                assert!(dk.decapsulate(&damaged).iter().all(|&b| b == 0xFF));

                // Setting the encapsulation key's padding bits must give an all-zero pair.
                let mut key_bytes = ek.to_bytes();
                let last = key_bytes.len() - 1;
                key_bytes[last] = 0xFF;
                let damaged_key = EncapsulationKey::<K>::new(&key_bytes).unwrap();
                let (ct, shared) = damaged_key.encapsulate_with_rng(&mut rng);
                assert!(ct.iter().all(|&b| b == 0));
                assert!(shared.iter().all(|&b| b == 0));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// The module documentation promises that only key import and export move the large
    /// arrays, and that encapsulation and decapsulation stay clear of them. Assert it on an
    /// ordinary thread: no large-stack wrapper here, deliberately. The largest parameter set
    /// has a 1.3 MB encapsulation key, so if any of these paths ever starts passing one by
    /// value this aborts on the default stack rather than quietly regressing.
    #[cfg(feature = "mceliece8192128")]
    #[test]
    fn hot_path_does_not_need_a_large_stack() {
        use kem::Kem as _;
        type K = hazmat::McEliece8192128;

        let mut rng = rand_chacha::ChaCha8Rng::from_seed([53u8; 32]);
        let (dk, ek) = K::generate_keypair_from_rng(&mut rng);
        let (ct, sent) = ek.encapsulate_with_rng(&mut rng);
        assert_eq!(dk.decapsulate(&ct), sent);

        // The handles themselves stay small; the key bytes live on the heap.
        assert!(core::mem::size_of::<EncapsulationKey<K>>() <= 64);
        assert!(core::mem::size_of::<DecapsulationKey<K>>() <= 128);
    }

    macro_rules! kem_trait_tests {
        ($($feature:literal => $name:ident, $params:ident, $seed:expr;)+) => {
            $(
                #[test]
                #[cfg(feature = $feature)]
                fn $name() {
                    traits_round_trip_with_large_stack::<hazmat::$params>($seed);
                }
            )+
        };
    }

    // One per distinct size class, including a pc set and the padded 6960119 set.
    kem_trait_tests! {
        "mceliece348864" => round_trip_348864, McEliece348864, 0x31;
        "mceliece460896pcf" => round_trip_460896pcf, McEliece460896pcf, 0x32;
        "mceliece6960119f" => round_trip_6960119f, McEliece6960119f, 0x33;
    }
}
