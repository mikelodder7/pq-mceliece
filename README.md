# pq-mceliece

A pure Rust implementation of the Classic McEliece post-quantum key encapsulation mechanism.

[![crates.io](https://img.shields.io/crates/v/pq-mceliece.svg)](https://crates.io/crates/pq-mceliece)
[![docs.rs](https://docs.rs/pq-mceliece/badge.svg)](https://docs.rs/pq-mceliece)

All eighteen standardized parameter sets are available at once. Choosing between
`mceliece348864` and `mceliece8192128pcf` is a runtime decision, not a compile-time one.

## Usage

```rust
use pq_mceliece::Algorithm;
use rand::rngs::SysRng;
use rand_core::UnwrapErr;

let alg = Algorithm::McEliece6960119f;
let (ek, dk) = alg.generate_keypair(UnwrapErr(SysRng));
let (ct, sent) = alg.encapsulate(&ek, UnwrapErr(SysRng))?;
let received = alg.decapsulate(&dk, &ct)?;

assert_eq!(sent, received);
# Ok::<(), pq_mceliece::Error>(())
```

Key generation is deterministic in a 32-byte seed, so a key pair can be rebuilt from backed-up
seed material rather than from a megabyte of stored public key:

```rust
use pq_mceliece::Algorithm;

let alg = Algorithm::McEliece348864f;
let (ek, dk) = alg.generate_keypair_from_seed([0x42u8; 32])?;
# Ok::<(), pq_mceliece::Error>(())
```

When the parameter set is known at compile time, the `hazmat` layer puts it in the type, so a
key from one parameter set cannot be passed to another:

```rust,ignore
use pq_mceliece::hazmat::{Kem, McEliece8192128f};

let (ek, dk) = McEliece8192128f::generate_keypair(rng);
let (ct, sent) = McEliece8192128f::encapsulate(&ek, rng)?;
let received = McEliece8192128f::decapsulate(&dk, &ct)?;
```

The `kem` module implements the [`kem`](https://docs.rs/kem) crate traits for every parameter
set, so Classic McEliece can be used in generic code alongside other KEMs.

## Parameter sets

| parameter set | NIST level | public key | private key | ciphertext | NIST | ISO |
| ------------- | ---------- | ---------- | ----------- | ---------- | ---- | --- |
| `mceliece348864`, `f`      | 1 | 261 120 B   | 6 492 B  | 96 B  | yes | no  |
| `mceliece460896`, `f`      | 3 | 524 160 B   | 13 608 B | 156 B | yes | yes |
| `mceliece460896pc`, `pcf`  | 3 | 524 160 B   | 13 608 B | 188 B | no  | yes |
| `mceliece6688128`, `f`     | 5 | 1 044 992 B | 13 932 B | 208 B | yes | yes |
| `mceliece6688128pc`, `pcf` | 5 | 1 044 992 B | 13 932 B | 240 B | no  | yes |
| `mceliece6960119`, `f`     | 5 | 1 047 319 B | 13 948 B | 194 B | yes | yes |
| `mceliece6960119pc`, `pcf` | 5 | 1 047 319 B | 13 948 B | 226 B | no  | yes |
| `mceliece8192128`, `f`     | 5 | 1 357 824 B | 14 120 B | 208 B | yes | yes |
| `mceliece8192128pc`, `pcf` | 5 | 1 357 824 B | 14 120 B | 240 B | no  | yes |

Shared secrets are 32 bytes everywhere.

- The **`f`** suffix means semi-systematic key generation, which is faster and produces keys
  drawn from the same distribution. Prefer it.
- The **`pc`** suffix means plaintext confirmation, which adds a 32-byte hash of the error
  vector to the ciphertext. These sets are in the ISO standard only.
- The Classic McEliece team recommends the `mceliece6*` sizes for long-term security.

## Features

| feature | effect |
| ------- | ------ |
| `nist` | the ten parameter sets in the NIST round-4 submission |
| `iso` | the sixteen parameter sets in the ISO standard |
| `pc` | the eight plaintext-confirmation parameter sets |
| `mceliece...` | one parameter set each, for building only what you use |
| `serde` | serialization for every value type |
| `hazmat` | makes the low-level, parameter-set-typed layer public |

`default = ["nist", "iso", "serde"]`.

## Performance

Classic McEliece has small ciphertexts, fast encapsulation, and large keys with expensive key
generation. Generate keys rarely and keep them.

To let the field arithmetic use carry-less multiply instructions, build with
`-C target-cpu=native`, or add to `.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-feature=+pclmulqdq"]   # x86-64
```

On AArch64 the equivalent instruction is enabled by default on most targets.

Run `cargo bench` for numbers on your own machine.

## Conformance

Every NIST parameter set is verified bit for bit against the published `kat_kem.rsp` known
answer tests. See [CONFORMANCE.md](CONFORMANCE.md) for the full picture, including what is
verified for the ISO-only `pc` sets and the constant-time properties this crate does and does
not claim.

## Security

This crate has not been audited. See [SECURITY.md](SECURITY.md) for how to report a problem.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgements

The Classic McEliece specification, reference implementation and known-answer tests are the
work of the Classic McEliece team: Martin R. Albrecht, Daniel J. Bernstein, Tung Chou, Carlos
Cid, Jan Gilcher, Tanja Lange, Varun Maram, Ingo von Maurich, Rafael Misoczki, Ruben
Niederhagen, Kenneth G. Paterson, Edoardo Persichetti, Christiane Peters, Peter Schwabe,
Nicolas Sendrier, Jakub Szefer, Cen Jung Tjhai, Martin Tomlinson and Wen Wang.
