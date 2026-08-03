# Comparison results

Measured 2026-08-03 on an Apple M2 Max MacBook Pro (12 cores: 8 performance + 4 efficiency,
64 GB RAM), `aarch64-apple-darwin`, Darwin 25.5.0, rustc 1.97.1. The command was
`benches/comparison/run-all.sh`, with the normal Cargo bench profile and no `RUSTFLAGS`.
PQClean therefore used its portable C implementation; its AVX2 implementation is available only
on supported x86 machines.

The compared versions were `classic-mceliece-rust` 3.1.0 and
`pqcrypto-classicmceliece` 0.2.1 against the current `pq-mceliece` 0.2.1 checkout. Criterion used
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
| mceliece348864 | 27.34 ms | 206.8 ms | 210.4 ms |
| mceliece348864f | 13.23 ms | 74.61 ms | 67.43 ms |
| mceliece460896 | 80.99 ms | 764.0 ms | 661.5 ms |
| mceliece460896f | 40.67 ms | 255.7 ms | 227.3 ms |
| mceliece6688128 | 214.5 ms | 1,314 ms | 1,637 ms |
| mceliece6688128f | 70.56 ms | 509.9 ms | 488.8 ms |
| mceliece6960119 | 174.8 ms | 1,833 ms | 1,326 ms |
| mceliece6960119f | 64.37 ms | 467.4 ms | 454.6 ms |
| mceliece8192128 | 176.7 ms | 2,507 ms | 1,452 ms |
| mceliece8192128f | 83.23 ms | 624.5 ms | 605.7 ms |

Non-`f` key generation uses rejection sampling and had wide 95% confidence intervals in this
short run. For example, PQClean `mceliece6688128` was 0.82–2.55 seconds. The `f` results were much
more stable.

## Encapsulation

| Parameter set | `pq-mceliece` | `classic-mceliece-rust` | PQClean |
| --- | ---: | ---: | ---: |
| mceliece348864 | 6.838 µs | 14.42 µs | 18.65 µs |
| mceliece348864f | 6.771 µs | 14.50 µs | 18.43 µs |
| mceliece460896 | 15.14 µs | 25.27 µs | 40.09 µs |
| mceliece460896f | 15.56 µs | 26.12 µs | 40.22 µs |
| mceliece6688128 | 26.32 µs | 54.31 µs | 100.5 µs |
| mceliece6688128f | 26.64 µs | 54.62 µs | 95.45 µs |
| mceliece6960119 | 27.30 µs | 77.66 µs | 111.0 µs |
| mceliece6960119f | 27.29 µs | 77.90 µs | 111.3 µs |
| mceliece8192128 | 31.96 µs | 53.13 µs | 72.59 µs |
| mceliece8192128f | 30.67 µs | 51.26 µs | 71.19 µs |

## Decapsulation

| Parameter set | `pq-mceliece` | `classic-mceliece-rust` | PQClean |
| --- | ---: | ---: | ---: |
| mceliece348864 | 0.0862 ms | 5.237 ms | 16.85 ms |
| mceliece348864f | 0.0862 ms | 5.064 ms | 17.04 ms |
| mceliece460896 | 0.2355 ms | 38.55 ms | 41.05 ms |
| mceliece460896f | 0.2347 ms | 38.41 ms | 41.00 ms |
| mceliece6688128 | 0.2779 ms | 73.83 ms | 77.37 ms |
| mceliece6688128f | 0.2767 ms | 73.85 ms | 77.40 ms |
| mceliece6960119 | 0.2456 ms | 71.39 ms | 74.93 ms |
| mceliece6960119f | 0.2468 ms | 72.45 ms | 78.98 ms |
| mceliece8192128 | 0.2861 ms | 96.19 ms | 96.20 ms |
| mceliece8192128f | 0.2792 ms | 91.45 ms | 97.24 ms |

These are end-to-end public-API timings, including each API's returned-value construction and
allocation behavior. The Rust implementations receive seeded ChaCha8 RNGs compatible with their
respective `rand_core` versions; PQClean's C API obtains randomness from the operating system.
