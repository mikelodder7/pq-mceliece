# Comparison results

Measured 2026-08-05 on an Apple M2 Max MacBook Pro (12 cores: 8 performance + 4 efficiency,
64 GB RAM), `aarch64-apple-darwin`, Darwin 25.5.0, rustc 1.97.1. The command was
`benches/comparison/run-all.sh`, with the normal Cargo bench profile and no `RUSTFLAGS`.
PQClean therefore used its portable C implementation; its AVX2 implementation is available only
on supported x86 machines.

The compared versions were `classic-mceliece-rust` 3.1.0 and
`pqcrypto-classicmceliece` 0.2.1 against the current `pq-mceliece` 0.3.0 checkout. Criterion used
a 1 second warm-up and 3 second requested measurement window. Key generation used 10 samples;
encapsulation and decapsulation used 100. Criterion lengthened the window where an operation was
too slow to collect that many samples. Values below are Criterion's regression estimate when
available and its arithmetic mean otherwise, rounded for readability.

## Test parameters

The ordinary and `f` variants have identical KEM dimensions and key/ciphertext sizes. The `f`
variant changes key generation to the faster semi-systematic form.

| Parameter family | NIST level | `m` | `n` | `t` | Public key | Private key | Ciphertext |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 348864 / 348864f | 1 | 12 | 3,488 | 64 | 261,120 B | 6,492 B | 96 B |
| 460896 / 460896f | 3 | 13 | 4,608 | 96 | 524,160 B | 13,608 B | 156 B |
| 6688128 / 6688128f | 5 | 13 | 6,688 | 128 | 1,044,992 B | 13,932 B | 208 B |
| 6960119 / 6960119f | 5 | 13 | 6,960 | 119 | 1,047,319 B | 13,948 B | 194 B |
| 8192128 / 8192128f | 5 | 13 | 8,192 | 128 | 1,357,824 B | 14,120 B | 208 B |

All shared secrets are 32 bytes.

## Key-pair generation

| Parameter set | `pq-mceliece` | `classic-mceliece-rust` | PQClean |
| --- | ---: | ---: | ---: |
| mceliece348864 | 15.89 ms | 207.1 ms | 175.2 ms |
| mceliece348864f | 9.99 ms | 73.86 ms | 66.96 ms |
| mceliece460896 | 47.99 ms | 755.0 ms | 628.9 ms |
| mceliece460896f | 28.38 ms | 237.6 ms | 224.5 ms |
| mceliece6688128 | 102.7 ms | 1,336 ms | 1,416 ms |
| mceliece6688128f | 54.86 ms | 521.4 ms | 487.3 ms |
| mceliece6960119 | 84.52 ms | 1,823 ms | 1,043 ms |
| mceliece6960119f | 47.99 ms | 464.6 ms | 444.5 ms |
| mceliece8192128 | 106.9 ms | 2,380 ms | 2,082 ms |
| mceliece8192128f | 64.09 ms | 612.6 ms | 596.8 ms |

Non-`f` key generation uses rejection sampling and had wide 95% confidence intervals in this
short run. For example, PQClean `mceliece6688128` was 0.69–2.33 seconds. The `f` results were much
more stable.

## Encapsulation

| Parameter set | `pq-mceliece` | `classic-mceliece-rust` | PQClean |
| --- | ---: | ---: | ---: |
| mceliece348864 | 7.981 µs | 14.26 µs | 18.54 µs |
| mceliece348864f | 8.112 µs | 14.26 µs | 18.58 µs |
| mceliece460896 | 14.70 µs | 25.07 µs | 39.90 µs |
| mceliece460896f | 14.67 µs | 24.93 µs | 39.68 µs |
| mceliece6688128 | 30.32 µs | 55.05 µs | 109.0 µs |
| mceliece6688128f | 29.80 µs | 54.85 µs | 110.0 µs |
| mceliece6960119 | 30.10 µs | 77.74 µs | 111.6 µs |
| mceliece6960119f | 30.03 µs | 80.16 µs | 110.5 µs |
| mceliece8192128 | 28.85 µs | 53.22 µs | 73.15 µs |
| mceliece8192128f | 27.96 µs | 50.78 µs | 70.38 µs |

## Decapsulation

| Parameter set | `pq-mceliece` | `classic-mceliece-rust` | PQClean |
| --- | ---: | ---: | ---: |
| mceliece348864 | 0.0689 ms | 5.308 ms | 17.13 ms |
| mceliece348864f | 0.0689 ms | 5.114 ms | 17.20 ms |
| mceliece460896 | 0.1359 ms | 38.94 ms | 40.92 ms |
| mceliece460896f | 0.1354 ms | 39.22 ms | 40.76 ms |
| mceliece6688128 | 0.1614 ms | 75.24 ms | 78.62 ms |
| mceliece6688128f | 0.1617 ms | 75.21 ms | 78.62 ms |
| mceliece6960119 | 0.1402 ms | 72.57 ms | 76.20 ms |
| mceliece6960119f | 0.1398 ms | 72.64 ms | 76.00 ms |
| mceliece8192128 | 0.1696 ms | 96.28 ms | 99.40 ms |
| mceliece8192128f | 0.1615 ms | 91.82 ms | 96.22 ms |

These are end-to-end public-API timings, including each API's returned-value construction and
allocation behavior. The Rust implementations receive seeded ChaCha8 RNGs compatible with their
respective `rand_core` versions; PQClean's C API obtains randomness from the operating system.
