/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Round-trip every enabled parameter set: generate a key pair, encapsulate a shared secret
//! to the public key, decapsulate it with the private key, and confirm both sides agree.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example round_trip
//! ```

use pq_mceliece::Algorithm;
use rand::SeedableRng;
use rand::rngs::{StdRng, SysRng};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A cryptographically secure generator, seeded once from the operating system.
    let mut rng = StdRng::try_from_rng(&mut SysRng)?;

    for &alg in Algorithm::enabled_algorithms() {
        let start = Instant::now();
        let (ek, dk) = alg.generate_keypair(&mut rng);
        let keygen = start.elapsed();

        let (ct, sent) = alg.encapsulate(&ek, &mut rng)?;
        let received = alg.decapsulate(&dk, &ct)?;
        assert_eq!(sent, received);

        println!(
            "{alg:?}: ok (keygen {keygen:.2?}, ciphertext {} bytes)",
            ct.as_ref().len()
        );
    }
    Ok(())
}
