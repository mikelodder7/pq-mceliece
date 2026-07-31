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
use rand::rngs::SysRng;
use rand_core::UnwrapErr;
use std::time::Instant;

fn main() -> Result<(), pq_mceliece::Error> {
    for &alg in Algorithm::enabled_algorithms() {
        let start = Instant::now();
        let (ek, dk) = alg.generate_keypair(UnwrapErr(SysRng));
        let keygen = start.elapsed();

        let (ct, sent) = alg.encapsulate(&ek, UnwrapErr(SysRng))?;
        let received = alg.decapsulate(&dk, &ct)?;
        assert_eq!(sent, received);

        println!(
            "{alg:?}: ok (keygen {keygen:.2?}, ciphertext {} bytes)",
            ct.as_ref().len()
        );
    }
    Ok(())
}
