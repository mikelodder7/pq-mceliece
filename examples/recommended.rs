/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! The shortest path to a sound choice: use [`Algorithm::RECOMMENDED`].
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example recommended
//! ```

use pq_mceliece::Algorithm;
use rand::SeedableRng;
use rand::rngs::{StdRng, SysRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A cryptographically secure generator, seeded once from the operating system.
    let mut rng = StdRng::try_from_rng(&mut SysRng)?;

    // McEliece6960119f: an mceliece6* size, which the Classic McEliece team recommends for
    // long-term security, with the faster semi-systematic key generation.
    let alg = Algorithm::RECOMMENDED;

    let (ek, dk) = alg.generate_keypair(&mut rng);
    let (ct, sent) = alg.encapsulate(&ek, &mut rng)?;
    let received = alg.decapsulate(&dk, &ct)?;
    assert_eq!(sent, received);

    println!(
        "{alg:?}: public key {} bytes, ciphertext {} bytes, shared secret {} bytes",
        ek.value().len(),
        ct.value().len(),
        received.value().len(),
    );
    Ok(())
}
