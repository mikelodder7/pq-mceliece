# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-01

### Changed

- **Breaking:** `Algorithm::encapsulation_key_from_decapsulation_key` returns `Result` and
  rejects a key from a different parameter set with `Error::AlgorithmMismatch` instead of
  silently deriving a mismatched public key.
- `Default` for the value types now produces a correct-length all-zero value instead of an
  empty one, so `DecapsulationKey::default().seed()` and `.prepare()` no longer panic.
- Examples and documentation construct a `StdRng` seeded once from the operating system
  instead of wrapping `SysRng` in `UnwrapErr`.
- Declared `rust-version = "1.88"`.

### Added

- `Algorithm::RECOMMENDED` (`mceliece6960119f`), with documentation on choosing a parameter
  set and why `mceliece348864` is for research rather than production.
- The `kem` module now re-exports the `kem` crate's traits and the parameter-set marker
  types, so the trait implementations are usable without the `hazmat` feature or a direct
  dependency on the `kem` crate.
- Examples: `round_trip` covers every enabled parameter set, `recommended` shows the
  shortest sound configuration, and `kem_traits` shows KEM-generic code and large-key
  export.
- Constant-time hardening from the pre-release audit: the FFT leaf construction and the
  control-bits recursion now use mask arithmetic where they previously used a branch and
  `Ord::min` on secret-derived values. Known-answer tests are unchanged.

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

[0.2.0]: https://github.com/mikelodder7/pq-mceliece/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mikelodder7/pq-mceliece/releases/tag/v0.1.0
