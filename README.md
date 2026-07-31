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

```rust
use pq_mceliece::hazmat::{Kem, McEliece8192128f};
# use rand::rngs::SysRng;
# use rand_core::UnwrapErr;

let (ek, dk) = McEliece8192128f::generate_keypair(UnwrapErr(SysRng));
let (ct, sent) = McEliece8192128f::encapsulate(&ek, UnwrapErr(SysRng))?;
let received = McEliece8192128f::decapsulate(&dk, &ct)?;

assert_eq!(sent, received);
# Ok::<(), pq_mceliece::Error>(())
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

Parameter sets and operations are selected independently.

| feature | effect |
| ------- | ------ |
| `nist` | the ten parameter sets in the NIST round-4 submission |
| `iso` | the sixteen parameter sets in the ISO standard |
| `pc` | the eight plaintext-confirmation parameter sets |
| `mceliece...` | one parameter set each, for building only what you use |
| `keygen` | `KeyGen` and `SeededKeyGen` |
| `encapsulate` | `Encap` |
| `decapsulate` | `Decap` |
| `kem` | the [`kem`](https://docs.rs/kem) crate traits; implies all three operations. Those traits pass keys as fixed-size arrays by value, and a Classic McEliece encapsulation key runs to 1.3 MB, so exporting or importing one needs a thread stack well above the 2 MiB default. Encapsulation and decapsulation are unaffected. See the `kem` module documentation. |
| `serde` | serialization for every value type |
| `hazmat` | makes the low-level, parameter-set-typed layer public |

`default = ["nist", "iso", "serde", "keygen", "encapsulate", "decapsulate", "kem"]`.

At least one operation must be enabled. A build only carries the code and dependencies its
enabled operations reach, which matters most for constrained targets:

- **Encapsulate-only** — a client encrypting to a server's public key. `Encode` is pure bit
  manipulation, so this build has no field arithmetic, no polynomial arithmetic, no matrix
  reduction, no sorting networks and no Beneš network.
- **Decapsulate-only** — a server unwrapping with a provisioned key. Drops `rand_core`
  entirely, since decapsulation consumes no randomness.

Dropping the `kem` feature also drops `hybrid-array`'s `extra-sizes`, which is what generates
the type-level constants up to `U1357824`, so it is a noticeable compile-time saving.

## Performance

Classic McEliece has small ciphertexts, fast encapsulation, and large keys with expensive key
generation. Generate keys rarely and keep them.

### Build flags

**Build with the defaults.** A stock `cargo build --release` already reaches the vector kernels:
on x86-64 they are selected at run time from CPUID, so no `RUSTFLAGS` and no `.cargo/config.toml`
entry are needed. AArch64 uses NEON unconditionally.

Two flags are worth *not* setting, both measured on a Zen 5 part:

- **`-C target-feature=+pclmulqdq`** makes key generation about 5% slower. The hardware
  carry-less multiply needs the operands moved into a vector register and the result moved back,
  and for the 13-bit field elements this crate multiplies, that round trip costs more latency
  than the portable convolution costs throughput.
- **`-C target-cpu=native`** was measured as a regression for decapsulation. Given AVX-512, LLVM
  vectorizes the decoder's tight branch-free loops into code slower than what it replaced. The
  kernels this crate ships are hand-written for the places where vectorization actually pays,
  and they do not depend on the flag.

If you benchmark a flag and it wins on your hardware, use it — but measure rather than assume,
and measure with the machine otherwise idle.

### Decapsulating repeatedly under one key

Roughly a third of a decapsulation depends only on the private key rather than on the message.
A holder that decapsulates many ciphertexts under one key can do that part once:

```rust
use pq_mceliece::Algorithm;
use rand::rngs::SysRng;
use rand_core::UnwrapErr;

let alg = Algorithm::McEliece8192128f;
let (ek, dk) = alg.generate_keypair(UnwrapErr(SysRng));
let prepared = dk.prepare();

for _ in 0..10 {
    let (ct, sent) = alg.encapsulate(&ek, UnwrapErr(SysRng)).unwrap();
    assert_eq!(prepared.decapsulate(&ct).unwrap(), sent);
}
```

The result is identical to `dk.decapsulate(&ct)`, including for ciphertexts that fail to decode.
The precomputed material is as sensitive as the private key: it is zeroized on drop and is
deliberately not serializable, so store the `DecapsulationKey` and prepare again after loading
it. Preparing to decapsulate a single message is not worth it.

Run `cargo bench` for numbers on your own machine; the `decapsulate` and `decapsulate_prepared`
groups measure the two paths.

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
