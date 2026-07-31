/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Statistical constant-time tests, in the style of `dudect`.
//!
//! Reading the source and classifying every branch establishes that no *written* branch depends on
//! a secret. It cannot establish that the compiled program has no data-dependent timing: a compiler
//! may introduce a branch where the source has none, a library routine may be variable time, and a
//! microarchitecture may treat two operand values differently. This measures the program that
//! actually runs.
//!
//! # The method
//!
//! From Reparaz, Balasch and Verbauwhede, *Dude, is my code constant time?* (DATE 2017). Time two
//! input classes, interleaved so drift and thermal effects hit both equally:
//!
//! * **fixed** — the same input every time;
//! * **random** — a fresh input every time.
//!
//! If the implementation is data oblivious the two timing distributions are the same distribution,
//! and Welch's t-statistic between them stays small. If it is not, the statistic grows without
//! bound as samples accumulate. The conventional threshold is `|t| > 4.5`, which is roughly a
//! `1e-5` false-positive rate for a single comparison.
//!
//! Timing distributions have a long right tail from interrupts and scheduling, which inflates the
//! variance and hides real leaks. `dudect` handles this by also computing the statistic over
//! progressively cropped samples; a leak that the uncropped test misses usually shows in a cropped
//! one. The verdict here is the largest `|t|` over all crops.
//!
//! # What a pass means
//!
//! A pass is evidence, not proof: it says no leak was detected at this sample count, on this
//! machine, against this compiler. Absence of evidence at `n` samples does not rule out a leak that
//! needs `10n`. A *failure*, by contrast, is close to conclusive — the statistic does not grow
//! without a cause.
//!
//! # Running
//!
//! Ignored by default: these take tens of seconds each and are statistical rather than
//! deterministic, so they do not belong in a plain `cargo test`.
//!
//! ```text
//! cargo test --release --test constant_time -- --ignored --nocapture
//! ```
//!
//! Run them on an otherwise idle machine. Competing load raises the variance and makes a real leak
//! harder to see, which fails safe for the attacker rather than for you.

#![cfg(all(
    feature = "mceliece8192128",
    feature = "keygen",
    feature = "encapsulate",
    feature = "decapsulate"
))]

use std::hint::black_box;
use std::time::Instant;

use pq_mceliece::Algorithm;
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};

/// The conventional `dudect` decision threshold.
const THRESHOLD: f64 = 4.5;

/// Samples per class, overridable so a borderline result can be re-run deeper.
///
/// The distinction that matters is not whether `|t|` clears the threshold at one sample count but
/// whether it *grows* with the count. A real leak accumulates evidence and the statistic rises
/// roughly as the square root of the samples; noise does not. Re-run with
/// `PQ_MCELIECE_DUDECT_SAMPLES=200000` before trusting any result that lands near 4.5.
fn samples(default: usize) -> usize {
    std::env::var("PQ_MCELIECE_DUDECT_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Percentiles the samples are cropped at before re-testing, as fractions of the sorted timings.
///
/// The long right tail of a timing distribution is scheduler noise, not signal. Cropping it makes
/// a small difference in the bulk of the distribution visible.
const CROPS: [f64; 8] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];

/// Which input class a measurement belongs to.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Class {
    Fixed,
    Random,
}

/// Welch's t-statistic between two samples, or `None` when either is too small to have a variance.
fn welch_t(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() < 2 || right.len() < 2 {
        return None;
    }
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let (mean_left, mean_right) = (mean(left), mean(right));
    let variance = |xs: &[f64], m: f64| {
        xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
    };
    let pooled = variance(left, mean_left) / left.len() as f64
        + variance(right, mean_right) / right.len() as f64;
    if pooled <= 0.0 {
        return None;
    }
    Some((mean_left - mean_right) / pooled.sqrt())
}

/// The largest `|t|` over the uncropped samples and every cropped view of them.
fn max_abs_t(fixed: &mut [f64], random: &mut [f64]) -> f64 {
    fixed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    random.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut worst: f64 = 0.0;
    for crop in CROPS {
        let take_fixed = ((fixed.len() as f64) * crop) as usize;
        let take_random = ((random.len() as f64) * crop) as usize;
        if let Some(t) = welch_t(&fixed[..take_fixed], &random[..take_random]) {
            worst = worst.max(t.abs());
        }
    }
    worst
}

/// Interleave two input classes, time `operation` on each, and report the worst `|t|`.
///
/// `prepare` builds the input for a class; only the class distinction should reach the timing.
fn dudect<T>(
    label: &str,
    samples: usize,
    mut prepare: impl FnMut(Class, &mut ChaCha8Rng) -> T,
    mut operation: impl FnMut(&T),
) -> f64 {
    let mut rng = ChaCha8Rng::from_seed([0x5Au8; 32]);
    let mut fixed = Vec::with_capacity(samples);
    let mut random = Vec::with_capacity(samples);

    // Warm caches and branch predictors so the first measurements are not systematically slow.
    {
        let warm = prepare(Class::Fixed, &mut rng);
        for _ in 0..64 {
            operation(&warm);
        }
    }

    for _ in 0..samples {
        // Alternate by coin flip rather than in strict rotation, so any periodic disturbance on
        // the machine cannot line up with one class.
        let class = if rng.next_u32() & 1 == 0 {
            Class::Fixed
        } else {
            Class::Random
        };
        let input = prepare(class, &mut rng);

        let start = Instant::now();
        operation(black_box(&input));
        let elapsed = start.elapsed().as_nanos() as f64;

        match class {
            Class::Fixed => fixed.push(elapsed),
            Class::Random => random.push(elapsed),
        }
    }

    let t = max_abs_t(&mut fixed, &mut random);
    println!(
        "  {label:<44} max |t| = {t:>7.2}   ({} fixed, {} random)  {}",
        fixed.len(),
        random.len(),
        if t < THRESHOLD {
            "ok"
        } else {
            "above threshold"
        }
    );
    t
}

/// Decapsulation must not reveal, by timing, whether the ciphertext decodes.
///
/// This is the property implicit rejection exists to provide: a ciphertext that fails to decode
/// returns a session key derived from the private key's rejection secret, and it must do so in the
/// same time a successful decode takes. A timing difference here is directly exploitable — it is an
/// oracle for whether a chosen ciphertext is a valid encoding, which is the classical entry point
/// to a reaction attack on a code-based KEM.
#[test]
#[ignore]
fn decapsulation_does_not_leak_whether_the_ciphertext_decodes() {
    let algorithm = Algorithm::McEliece8192128;
    let mut setup = ChaCha8Rng::from_seed([7u8; 32]);
    let (ek, dk) = algorithm.generate_keypair(&mut setup);
    let (valid, _) = algorithm
        .encapsulate(&ek, &mut setup)
        .expect("a freshly generated key encapsulates");
    let prepared = dk.prepare();

    let ciphertext_bytes = valid.value().to_vec();
    let t = dudect(
        "decapsulate: valid vs corrupted ciphertext",
        samples(20_000),
        |class, rng| match class {
            Class::Fixed => valid.clone(),
            Class::Random => {
                // Corrupt one byte, so the word almost certainly fails to decode. The length and
                // every public structure stay identical; only decodability changes.
                let mut bytes = ciphertext_bytes.clone();
                let at = (rng.next_u32() as usize) % bytes.len();
                bytes[at] ^= 1 << (rng.next_u32() % 8);
                algorithm
                    .ciphertext_from_bytes(&bytes)
                    .unwrap_or_else(|_| valid.clone())
            }
        },
        |ciphertext| {
            let _ = black_box(prepared.decapsulate(ciphertext));
        },
    );

    assert!(
        t < THRESHOLD,
        "decapsulation timing depends on whether the ciphertext decodes (max |t| = {t:.2})"
    );
}

/// Decapsulation must not leak the private key through timing.
///
/// One key held fixed against a stream of fresh keys, decapsulating a ciphertext each time. A
/// difference means some part of the decoder runs for a length that depends on key material.
#[test]
#[ignore]
fn decapsulation_does_not_leak_the_private_key() {
    let algorithm = Algorithm::McEliece8192128;
    let mut setup = ChaCha8Rng::from_seed([11u8; 32]);
    let (fixed_ek, fixed_dk) = algorithm.generate_keypair(&mut setup);
    let (fixed_ct, _) = algorithm
        .encapsulate(&fixed_ek, &mut setup)
        .expect("a freshly generated key encapsulates");

    // Generating a key per sample would dominate the measurement, so draw from a small pool of
    // pre-generated keys instead. The pool is larger than any per-key caching effect.
    let pool: Vec<_> = (0..16)
        .map(|i| {
            let mut rng = ChaCha8Rng::from_seed([0x20 + i as u8; 32]);
            let (ek, dk) = algorithm.generate_keypair(&mut rng);
            let (ct, _) = algorithm
                .encapsulate(&ek, &mut rng)
                .expect("a freshly generated key encapsulates");
            (dk, ct)
        })
        .collect();

    let t = dudect(
        "decapsulate: fixed key vs varying keys",
        samples(4_000),
        |class, rng| match class {
            Class::Fixed => (fixed_dk.clone(), fixed_ct.clone()),
            Class::Random => pool[(rng.next_u32() as usize) % pool.len()].clone(),
        },
        |(dk, ct)| {
            let _ = black_box(algorithm.decapsulate(dk, ct));
        },
    );

    assert!(
        t < THRESHOLD,
        "decapsulation timing depends on the private key (max |t| = {t:.2})"
    );
}

/// Decapsulation must not leak which error vector a valid ciphertext carries.
///
/// The strongest of these tests. Every input decodes successfully, under one key, so neither the
/// success path nor the key varies — only *which* weight-`t` error vector the decoder recovers.
/// A timing difference here would mean the decoder's running time depends on where the errors fell,
/// which is the secret the whole construction protects.
#[test]
#[ignore]
fn decapsulation_does_not_leak_which_error_vector_decodes() {
    let algorithm = Algorithm::McEliece8192128;
    let mut setup = ChaCha8Rng::from_seed([23u8; 32]);
    let (ek, dk) = algorithm.generate_keypair(&mut setup);
    let prepared = dk.prepare();

    let fixed = algorithm
        .encapsulate(&ek, &mut setup)
        .expect("a freshly generated key encapsulates")
        .0;
    // A pool of distinct valid ciphertexts under the same key. Each is a few hundred bytes, so
    // unlike the public key they all stay resident and the comparison is not a cache measurement.
    let pool: Vec<_> = (0..64)
        .map(|_| {
            algorithm
                .encapsulate(&ek, &mut setup)
                .expect("a freshly generated key encapsulates")
                .0
        })
        .collect();

    let t = dudect(
        "decapsulate: fixed vs varying valid ciphertexts",
        samples(20_000),
        |class, rng| match class {
            Class::Fixed => fixed.clone(),
            Class::Random => pool[(rng.next_u32() as usize) % pool.len()].clone(),
        },
        |ciphertext| {
            let _ = black_box(prepared.decapsulate(ciphertext));
        },
    );

    assert!(
        t < THRESHOLD,
        "decapsulation timing depends on which error vector decodes (max |t| = {t:.2})"
    );
}

/// Encapsulation's running time varies by specification, and this records by how much.
///
/// Reports rather than asserts, because the variability is specified rather than accidental.
/// `FixedWeight` draws candidate indices and restarts if it cannot find `t` distinct ones in range,
/// so the number of restarts depends on the randomness consumed. A fixed stream restarts the same
/// number of times every call; random streams give a distribution. The statistic is consequently
/// enormous — around `|t| = 180` here — and carries no security information, because the restarts
/// depend on *rejected* candidates while the ciphertext commits only to the accepted vector.
///
/// It is kept as a measurement so that the number is on the record and a future change to the
/// sampler can be compared against it. The security-relevant surface is decapsulation, where the
/// private key lives and where an attacker has chosen-ciphertext power; encapsulation consumes only
/// public data and fresh randomness an attacker neither controls nor observes.
///
/// An earlier version of this test varied the *public key* instead and reported `|t| = 14.5`. That
/// was the cache, not a side channel: a `mceliece8192128` public key is 1.3 MB, so a pool of
/// sixteen evicts everything while a fixed key stays resident.
#[test]
#[ignore]
fn encapsulation_timing_variability_is_recorded() {
    let algorithm = Algorithm::McEliece8192128;
    let mut setup = ChaCha8Rng::from_seed([13u8; 32]);
    let (ek, _) = algorithm.generate_keypair(&mut setup);

    let t = dudect(
        "encapsulate: fixed randomness vs varying (informational)",
        samples(20_000),
        |class, rng| match class {
            Class::Fixed => ChaCha8Rng::from_seed([0x99u8; 32]),
            Class::Random => {
                let mut seed = [0u8; 32];
                let (words, _) = seed.as_chunks_mut::<4>();
                for word in words {
                    *word = rng.next_u32().to_le_bytes();
                }
                ChaCha8Rng::from_seed(seed)
            }
        },
        |stream| {
            let mut draw = stream.clone();
            let _ = black_box(algorithm.encapsulate(&ek, &mut draw));
        },
    );

    println!("    (informational only: FixedWeight restarts are specified, |t| = {t:.2})");
}
