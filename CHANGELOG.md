# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-31

Initial release.

### Added

- All eighteen standardized Classic McEliece parameter sets: the ten from the NIST round-4
  submission and the sixteen from the ISO standard (June 2026), including the ISO-only
  plaintext-confirmation (`pc`) variants.
- Runtime parameter-set selection through `Algorithm`, and a compile-time-typed `hazmat`
  layer for builds that know their parameter set.
- Key generation (plain and seeded/deterministic), encapsulation, and decapsulation as
  independently selectable cargo features.
- Implementations of the [`kem`](https://docs.rs/kem) crate traits behind the `kem` feature.
- Precomputed decapsulation via `DecapsulationKey::prepare` for holders that decapsulate
  many ciphertexts under one key.
- `serde` support for every value type.
- Bit-for-bit known-answer-test verification against the published NIST `kat_kem.rsp`
  vectors, and structural verification for the ISO-only `pc` sets (see `CONFORMANCE.md`).
- Constant-time implementation with vector kernels for x86-64 (AVX2/AVX-512, runtime
  selected) and AArch64 (NEON), each checked against its scalar twin, plus a `dudect`
  statistical timing test suite.

[0.1.0]: https://github.com/mikelodder7/pq-mceliece/releases/tag/v0.1.0
