# Cross-implementation benchmarks

This standalone Criterion harness compares the public KEM APIs of three crates from crates.io:

- `pq-mceliece` (the current checkout): pure Rust with architecture-specific SIMD kernels.
- `classic-mceliece-rust` 3.1.0: an independent safe, pure-Rust implementation.
- `pqcrypto-classicmceliece` 0.2.1, labelled `PQClean` in results: Rust wrappers around PQClean's
  C implementation. Its default features select AVX2 at run time on supported x86 machines and
  the portable C implementation elsewhere.

The common surface is the ten NIST round-4 parameter sets. ISO plaintext-confirmation (`pc`)
sets are omitted because neither comparison crate implements them. Each benchmark build selects
one parameter set because `classic-mceliece-rust` intentionally allows only one.

Run the recommended parameter set:

```sh
cargo bench --manifest-path benches/comparison/Cargo.toml
```

Run another set (the default feature must be disabled):

```sh
cargo bench \
  --manifest-path benches/comparison/Cargo.toml \
  --no-default-features \
  --features mceliece348864f
```

Run all ten common sets:

```sh
benches/comparison/run-all.sh
```

The harness measures key-pair generation, encapsulation, and decapsulation. Setup and round-trip
validation happen outside timed regions. Key generation includes the allocations and key-value
construction performed by each crate's normal heap-capable public API; encapsulation and
decapsulation likewise include construction of their returned ciphertext/shared-secret values.
This reflects caller-visible API cost, not isolated internal kernels.

Randomness is necessarily implementation-specific: both Rust implementations receive a seeded
ChaCha8 RNG compatible with the `rand_core` version they expose, while PQClean calls the operating
system RNG through its C API. The seed is fixed for repeatability but the generator advances
between samples. Do not compare results from different machines or build flags as though they
were from the same experiment.

Criterion uses a 1 second warm-up and 3 second measurement window. Key generation uses the
minimum supported sample size of 10 because it is expensive. Results are written beneath
`benches/comparison/target/criterion/`.

See [RESULTS.md](RESULTS.md) for one complete run and its machine/build details.

