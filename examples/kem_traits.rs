/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Classic McEliece through the [`kem`](https://docs.rs/kem) crate traits, in code that is
//! generic over the parameter set and would work just as well over any other KEM.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example kem_traits
//! ```

use pq_mceliece::kem::{
    Decapsulate, DecapsulationKey, Decapsulator, Encapsulate, EncapsulationKey, Kem, KemSizes,
    KeyExport, McEliece460896f, McEliece6960119f, TryKeyInit,
};
use rand::SeedableRng;
use rand::rngs::{StdRng, SysRng};
use rand_core::CryptoRng;

/// Establish a shared secret and hand back the sizes involved.
///
/// Nothing here names Classic McEliece: the same function compiles against any KEM whose key
/// types implement the `kem` traits.
fn round_trip<K>(mut rng: impl CryptoRng) -> (usize, usize)
where
    K: KemSizes
        + Kem<EncapsulationKey = EncapsulationKey<K>, DecapsulationKey = DecapsulationKey<K>>,
{
    let (dk, ek) = K::generate_keypair_from_rng(&mut rng);
    let (ct, sent) = ek.encapsulate_with_rng(&mut rng);
    let received = dk.decapsulate(&ct);
    assert_eq!(sent, received);
    (ct.len(), received.len())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // A cryptographically secure generator, seeded once from the operating system.
    let mut rng = StdRng::try_from_rng(&mut SysRng)?;

    for (name, (ct, ss)) in [
        ("McEliece460896f", round_trip::<McEliece460896f>(&mut rng)),
        ("McEliece6960119f", round_trip::<McEliece6960119f>(&mut rng)),
    ] {
        println!("{name}: ciphertext {ct} bytes, shared secret {ss} bytes");
    }

    // Exporting or importing a key moves the whole megabyte-scale array by value, which
    // outgrows Rust's default 2 MiB thread stack. Give the thread a real stack for that;
    // encapsulation and decapsulation above need no such treatment. See the `kem` module
    // documentation's stack-usage section.
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let mut rng = StdRng::try_from_rng(&mut SysRng)?;
            let (dk, ek) = McEliece6960119f::generate_keypair_from_rng(&mut rng);
            let exported = ek.to_bytes();
            let imported = EncapsulationKey::<McEliece6960119f>::new(&exported)?;
            assert_eq!(&imported, dk.encapsulation_key());
            println!(
                "McEliece6960119f: exported and reimported {} bytes",
                exported.len()
            );
            Ok(())
        })?
        .join()
        .map_err(|_| "the export thread panicked")??;
    Ok(())
}
