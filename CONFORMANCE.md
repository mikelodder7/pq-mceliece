# Conformance

## What this crate implements

Classic McEliece, as specified in:

- **NIST round 4**, the [2022-10-23 submission package](https://classic.mceliece.org/nist.html)
  (`mceliece-spec-20221023.pdf` plus `mceliece-pc-20221023.pdf` for the `pc` variants).
- **ISO**, standardized June 2026. The ISO standard is compatible with the official
  specification from the Classic McEliece team.

### Parameter sets

| parameter set | `m` | `n` | `t` | `(mu, nu)` | pc | NIST | ISO |
| ------------- | --- | --- | --- | ---------- | -- | ---- | --- |
| `mceliece348864`      | 12 | 3488 | 64  | (0, 0)   | no  | yes | no  |
| `mceliece348864f`     | 12 | 3488 | 64  | (32, 64) | no  | yes | no  |
| `mceliece460896`      | 13 | 4608 | 96  | (0, 0)   | no  | yes | yes |
| `mceliece460896f`     | 13 | 4608 | 96  | (32, 64) | no  | yes | yes |
| `mceliece460896pc`    | 13 | 4608 | 96  | (0, 0)   | yes | no  | yes |
| `mceliece460896pcf`   | 13 | 4608 | 96  | (32, 64) | yes | no  | yes |
| `mceliece6688128`     | 13 | 6688 | 128 | (0, 0)   | no  | yes | yes |
| `mceliece6688128f`    | 13 | 6688 | 128 | (32, 64) | no  | yes | yes |
| `mceliece6688128pc`   | 13 | 6688 | 128 | (0, 0)   | yes | no  | yes |
| `mceliece6688128pcf`  | 13 | 6688 | 128 | (32, 64) | yes | no  | yes |
| `mceliece6960119`     | 13 | 6960 | 119 | (0, 0)   | no  | yes | yes |
| `mceliece6960119f`    | 13 | 6960 | 119 | (32, 64) | no  | yes | yes |
| `mceliece6960119pc`   | 13 | 6960 | 119 | (0, 0)   | yes | no  | yes |
| `mceliece6960119pcf`  | 13 | 6960 | 119 | (32, 64) | yes | no  | yes |
| `mceliece8192128`     | 13 | 8192 | 128 | (0, 0)   | no  | yes | yes |
| `mceliece8192128f`    | 13 | 8192 | 128 | (32, 64) | no  | yes | yes |
| `mceliece8192128pc`   | 13 | 8192 | 128 | (0, 0)   | yes | no  | yes |
| `mceliece8192128pcf`  | 13 | 8192 | 128 | (32, 64) | yes | no  | yes |

These eighteen sets are exactly the union of the ten in the NIST submission and the sixteen in
the ISO standard. `mceliece348864` and `mceliece348864f` are NIST-only; the eight `pc` and
`pcf` sets are ISO-only.

### Symmetric-cryptography parameters

Every set uses the values the standards fix for all selected parameter sets:

- `l = 256`, so `Hash` and the session key are 32 bytes.
- `Hash(x)` is the first 256 bits of `SHAKE256(x)`.
- `sigma1 = 16`, `sigma2 = 32`.
- `PRG(delta)` is the first `n + sigma2 * q + sigma1 * t + l` bits of `SHAKE256(64, delta)`.

## What is verified

### Known-answer tests

`src/hazmat/kat.rs` regenerates all ten records of the published `kat_kem.rsp` for each of the
ten NIST parameter sets and compares a SHAKE256 digest over the concatenated `seed`, `pk`,
`sk`, `ct` and `ss` fields against the digest of the published file. This exercises the whole
pipeline bit for bit, including the AES-256-CTR DRBG that the NIST harness seeds key generation
and encapsulation from.

The `kat_kem.rsp` files total roughly 160 MB and are therefore not vendored. To recheck the
committed digests against the originals:

```sh
curl -O https://classic.mceliece.org/nist/mceliece-kat-20221023.tar.gz
tar xzf mceliece-kat-20221023.tar.gz
```

then hash `seed || pk || sk || ct || ss` over the ten records of each file with SHAKE256.

The ISO-only `pc` parameter sets have no published known-answer tests. They are verified
structurally instead: their non-`pc` counterparts are KAT-verified, and `pc` changes only the
ciphertext framing and the session-key hash, both of which are tested directly.

### Property tests

Beyond the known-answer tests, each layer is checked against an independent definition rather
than against itself:

- **Field arithmetic** against schoolbook multiplication with explicit long division, over
  every element of `GF(2^12)` and `GF(2^13)`.
- **Polynomial arithmetic** in `F_q[y]/F(y)` against long division, plus the defining property
  that the generated minimal polynomial annihilates its generator.
- **Beneš network** against directly applying the permutation to a bit vector, in both
  directions, for `m = 12` and `m = 13`.
- **Control bits** by replaying the generated network over the identity permutation.
- **Sorting networks** against the standard library sort, at many lengths.
- **Precomputed decapsulation** by requiring a prepared key to agree with the key it came from
  on every parameter set, both for ciphertexts that decode and for ones that are implicitly
  rejected, where the substituted session key must still match.
- **`Encode`** against the definition `C = He` with `H = (I | T)` built element by element.
- **`Decode`** by recovering the exact error vector from an honestly generated syndrome, and by
  rejecting syndromes that do not correspond to a weight-`t` vector.
- **Key generation** by confirming the recovered support is a set of distinct field elements
  none of which is a root of the Goppa polynomial, and that regenerating from the stored seed
  reproduces the key pair exactly.

## Implementation choices

### Narrowly decoded

The specification describes two readings for parameter sets whose bit lengths are not multiples
of eight. *Simply Decoded* ignores padding bits on input; *Narrowly Decoded* rejects inputs
whose padding bits are nonzero. This crate is Narrowly Decoded, matching the reference
implementation.

Among the standardized sets this only ever applies to `mceliece6960119` and its variants, where
`mt = 1547` and `k = 5413` are both odd. Every other set has no padding bits at all, so the
distinction does not arise.

The `Algorithm` and `hazmat` APIs report this as `Error::EncapsulationKeyPadding` and
`Error::CiphertextPadding`. The `kem` crate trait implementations cannot return an error, so
they reproduce the reference behavior instead: an all-zero ciphertext and shared key for a
padded encapsulation key, and an all-ones shared key for a padded ciphertext.

### Implicit rejection

A ciphertext that does not decode is not reported. Decapsulation derives the session key from
the private key's rejection string `s`, so the result is unpredictable to anyone without the
private key and identical every time the same bad ciphertext is presented. This is what makes
the KEM CCA-secure, and distinguishing the two cases from the outside is exactly what the
construction prevents.

### Constant-time behavior

All operations on secret data are branch free and index free with respect to that data. There
are no lookup tables, no data-dependent shift amounts, and no early exits. Selections use
arithmetic masks throughout, including the two decision points in Berlekamp-Massey, the
substitution of `s` for a failed decode, and the plaintext-confirmation comparison.

Three conditions do influence control flow, and all three depend only on public data or on the
key generation seed, matching the reference implementation's use of `crypto_declassify`:

- Key generation restarts when `Irreducible`, `FieldOrdering` or `MatGen` rejects its input.
  These depend on the seed alone, before any key exists.
- `FixedWeight` restarts when a draw yields too few in-range indices or a repeat. This depends
  on fresh randomness that is discarded.
- Padding checks on encapsulation keys and ciphertexts, which are public values.

Precomputing the message-independent part of decapsulation does not change any of this. The
precomputation runs the same masked operations the decoder previously ran inline, and the
decoding that follows differs only in reading that material rather than deriving it. The
material is a function of the decapsulation key alone and is treated as equally sensitive: it
is zeroized on drop, kept out of the `Debug` output, and cannot be serialized.

This crate has not been audited, and no formal constant-time verification has been run against
compiled output.

This source-level claim does not cover physical power or electromagnetic observation, template
attacks, or fault injection. Published Classic McEliece attacks have targeted syndrome/FFT
computation, Berlekamp--Massey, Goppa-polynomial loading, and Gaussian elimination under those
physical threat models. Deployments exposed to those attackers need separately evaluated
masking, fault detection, and platform-specific countermeasures; this crate does not currently
provide them.
