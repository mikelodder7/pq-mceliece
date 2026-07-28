/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Verification against the official known-answer tests.
//!
//! The Classic McEliece team publishes `kat_kem.rsp` files for the ten NIST parameter sets,
//! each holding ten `(seed, pk, sk, ct, ss)` records. Those files total roughly 160 MB, so
//! rather than vendoring them this module regenerates every record and compares a SHAKE256
//! digest of the concatenated fields against the digest of the published file.
//!
//! Reproducing a record requires the deterministic random bit generator the NIST test harness
//! uses, an AES-256 counter-mode DRBG, since the harness seeds it from the record's `seed` and
//! then draws key generation and encapsulation randomness from it.

use aes::{
    Aes256Enc, Block,
    cipher::{BlockCipherEncrypt, KeyInit},
};
use rand_core::{Rng, TryCryptoRng, TryRng};
use shake::digest::{ExtendableOutput, Update};

use super::decap::decapsulate;
use super::encap::encapsulate;
use super::keygen::seeded_keypair;
use super::params::Params;

/// The AES-256 counter-mode DRBG specified for the NIST post-quantum test harness.
struct NistDrbg {
    key: [u8; 32],
    counter: [u8; 16],
}

impl NistDrbg {
    /// `randombytes_init(entropy_input, NULL, 256)`.
    fn new(entropy: &[u8; 48]) -> Self {
        let mut drbg = Self {
            key: [0u8; 32],
            counter: [0u8; 16],
        };
        drbg.update(Some(entropy));
        drbg
    }

    fn increment(&mut self) {
        for byte in self.counter.iter_mut().rev() {
            *byte = byte.wrapping_add(1);
            if *byte != 0 {
                break;
            }
        }
    }

    fn update(&mut self, provided: Option<&[u8; 48]>) {
        let enc = Aes256Enc::new_from_slice(&self.key).expect("a 32-byte AES-256 key");
        let mut temp = [0u8; 48];
        let mut input = Block::default();
        let mut output = Block::default();

        for chunk in temp.chunks_exact_mut(16) {
            self.increment();
            input.copy_from_slice(&self.counter);
            enc.encrypt_block_b2b(&input, &mut output);
            chunk.copy_from_slice(&output);
        }
        if let Some(provided) = provided {
            for (t, p) in temp.iter_mut().zip(provided.iter()) {
                *t ^= p;
            }
        }

        self.key.copy_from_slice(&temp[..32]);
        self.counter.copy_from_slice(&temp[32..]);
    }
}

impl TryRng for NistDrbg {
    type Error = core::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut bytes = [0u8; 4];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut bytes = [0u8; 8];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        let enc = Aes256Enc::new_from_slice(&self.key).expect("a 32-byte AES-256 key");
        let mut input = Block::default();
        let mut output = Block::default();

        for chunk in dest.chunks_mut(16) {
            self.increment();
            input.copy_from_slice(&self.counter);
            enc.encrypt_block_b2b(&input, &mut output);
            chunk.copy_from_slice(&output[..chunk.len()]);
        }

        self.update(None);
        Ok(())
    }
}

impl TryCryptoRng for NistDrbg {}

/// Regenerate the published KAT records and return a SHAKE256 digest over their fields.
fn kat_digest<P: Params>(vectors: usize) -> [u8; 32] {
    // The harness starts from `entropy_input = 0, 1, ..., 47` and draws one 48-byte seed per
    // record before generating any keys.
    let mut entropy = [0u8; 48];
    for (i, byte) in entropy.iter_mut().enumerate() {
        *byte = i as u8;
    }
    let mut harness = NistDrbg::new(&entropy);

    let mut seeds = Vec::with_capacity(vectors);
    for _ in 0..vectors {
        let mut seed = [0u8; 48];
        harness.fill_bytes(&mut seed);
        seeds.push(seed);
    }

    let mut hasher = shake::Shake256::default();
    let mut pk = vec![0u8; P::PUBLIC_KEY_LENGTH];
    let mut sk = vec![0u8; P::SECRET_KEY_LENGTH];
    let mut ct = vec![0u8; P::CIPHERTEXT_LENGTH];
    let mut ss = vec![0u8; P::SHARED_SECRET_LENGTH];
    let mut received = vec![0u8; P::SHARED_SECRET_LENGTH];

    for seed in seeds.iter() {
        let mut rng = NistDrbg::new(seed);

        // `crypto_kem_keypair` draws the 32-byte `delta` and then runs `SeededKeyGen`.
        let mut delta = [0u8; 32];
        rng.fill_bytes(&mut delta);
        seeded_keypair::<P>(&mut pk, &mut sk, &delta);

        assert!(
            encapsulate::<P>(&mut ct, &mut ss, &pk, &mut rng),
            "{} encapsulation rejected its own public key",
            P::NAME
        );
        assert!(
            decapsulate::<P>(&mut received, &ct, &sk),
            "{} decapsulation rejected its own ciphertext",
            P::NAME
        );
        assert_eq!(ss, received, "{} round trip", P::NAME);

        hasher.update(seed);
        hasher.update(&pk);
        hasher.update(&sk);
        hasher.update(&ct);
        hasher.update(&ss);
    }

    let mut digest = [0u8; 32];
    hasher.finalize_xof_into(&mut digest);
    digest
}

/// Check a parameter set against the digest of its published `kat_kem.rsp`.
fn check<P: Params>(expected_hex: &str) {
    let digest = kat_digest::<P>(10);
    assert_eq!(
        hex::encode(digest),
        expected_hex,
        "{} does not match the published known-answer tests",
        P::NAME
    );
}

macro_rules! kat_tests {
    ($($feature:literal => $name:ident, $params:ty, $digest:literal;)+) => {
        $(
            #[test]
            #[cfg(feature = $feature)]
            fn $name() {
                use crate::hazmat::params::*;
                check::<$params>($digest);
            }
        )+
    };
}

// SHAKE256 digests over the concatenation of `seed || pk || sk || ct || ss` for each of the
// ten records in the corresponding `kat_kem.rsp` from the 2022-10-23 NIST submission package
// (<https://classic.mceliece.org/nist/mceliece-kat-20221023.tar.gz>).
kat_tests! {
    "mceliece348864" => kat_348864, McEliece348864,
        "89ec7bb2d7e802f306f8625b4f61def7d01605e0a9a5ba2b1ccac5ef59ddcf2d";
    "mceliece348864f" => kat_348864f, McEliece348864f,
        "ec12f491895bbbbb8ea8979fae7bd5db2ee60d788cc73065778d747e47013492";
    "mceliece460896" => kat_460896, McEliece460896,
        "b72a85d124d8af64ca506aae583598b477739945b25bfcc09a1416b92ef668bc";
    "mceliece460896f" => kat_460896f, McEliece460896f,
        "f884c5abe465098ab55215ac0db9acf59d05feb1dd2a079daa758610eb416282";
    "mceliece6688128" => kat_6688128, McEliece6688128,
        "5ae56d1156ab3a8e20603686a3f7b66278537df55e0f4e754010e1959aa16e1d";
    "mceliece6688128f" => kat_6688128f, McEliece6688128f,
        "ab4ffcdbd3d4a27b7b81913594a7f94524a2e380fb2a8cf86a4c49c838b8edb4";
    "mceliece6960119" => kat_6960119, McEliece6960119,
        "a98fe7f6f064282345fcf80b0effb352653a35d8001591ea0ddd24e5aa8a4eb1";
    "mceliece6960119f" => kat_6960119f, McEliece6960119f,
        "bf52910654069993aed89159167b458227bc2bec40ea087f1e39dcf687f54f45";
    "mceliece8192128" => kat_8192128, McEliece8192128,
        "b05cca9448925e535bcf91ae5234984f128a76668d35044998b8e02c785983c7";
    "mceliece8192128f" => kat_8192128f, McEliece8192128f,
        "62d7cdfd521164dd4989347e4133fb6f65882d4e64945e1caf0a5168208a63d6";
}

#[test]
fn the_drbg_matches_the_nist_reference_output() {
    // The first 48 bytes drawn after `randombytes_init(0..48)` are the seed of record 0 in
    // every published `kat_kem.rsp`.
    let mut entropy = [0u8; 48];
    for (i, byte) in entropy.iter_mut().enumerate() {
        *byte = i as u8;
    }
    let mut drbg = NistDrbg::new(&entropy);

    let mut seed = [0u8; 48];
    drbg.fill_bytes(&mut seed);
    assert_eq!(
        hex::encode_upper(seed),
        "061550234D158C5EC95595FE04EF7A25767F2E24CC2BC479D09D86DC9ABCFDE7\
         056A8C266F9EF97ED08541DBD2E1FFA1"
    );
}
