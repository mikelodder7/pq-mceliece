/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Benchmarks for every enabled Classic McEliece parameter set.
//!
//! Key generation is by far the most expensive operation and is what the `f` variants exist to
//! speed up, so it is measured separately from encapsulation and decapsulation.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pq_mceliece::Algorithm;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use std::hint::black_box;

fn keypair(c: &mut Criterion) {
    let mut group = c.benchmark_group("keypair");
    group.sample_size(10);

    for &algorithm in Algorithm::enabled_algorithms() {
        let mut rng = ChaCha8Rng::from_seed([7u8; 32]);
        group.bench_with_input(
            BenchmarkId::from_parameter(algorithm.name()),
            &algorithm,
            |b, &algorithm| {
                b.iter(|| black_box(algorithm.generate_keypair(&mut rng)));
            },
        );
    }
    group.finish();
}

fn encapsulate(c: &mut Criterion) {
    let mut group = c.benchmark_group("encapsulate");

    for &algorithm in Algorithm::enabled_algorithms() {
        let mut rng = ChaCha8Rng::from_seed([11u8; 32]);
        let (ek, _) = algorithm.generate_keypair(&mut rng);
        group.bench_with_input(
            BenchmarkId::from_parameter(algorithm.name()),
            &ek,
            |b, ek| {
                b.iter(|| black_box(algorithm.encapsulate(ek, &mut rng)));
            },
        );
    }
    group.finish();
}

fn decapsulate(c: &mut Criterion) {
    let mut group = c.benchmark_group("decapsulate");

    for &algorithm in Algorithm::enabled_algorithms() {
        let mut rng = ChaCha8Rng::from_seed([13u8; 32]);
        let (ek, dk) = algorithm.generate_keypair(&mut rng);
        let (ct, _) = algorithm
            .encapsulate(&ek, &mut rng)
            .expect("a freshly generated key encapsulates");
        group.bench_with_input(
            BenchmarkId::from_parameter(algorithm.name()),
            &(dk, ct),
            |b, (dk, ct)| {
                b.iter(|| black_box(algorithm.decapsulate(dk, ct)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, keypair, encapsulate, decapsulate);
criterion_main!(benches);
