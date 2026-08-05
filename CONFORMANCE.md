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

These eighteen sets are exactly the union of the ten in the NIST submission (Classic McEliece
was not selected as a NIST final standard) and the sixteen in the ISO standard.
`mceliece348864` and `mceliece348864f` are NIST-only; the eight `pc` and `pcf` sets are
ISO-only.

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

Three conditions do influence control flow, and each depends only on public data, the key
generation seed, or randomness that is discarded, matching the reference implementation's use
of `crypto_declassify`:

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

This crate has not been independently audited, and no formal constant-time verification has been run against
compiled output.

This source-level claim does not cover physical power or electromagnetic observation, template
attacks, or fault injection. Published Classic McEliece attacks have targeted syndrome/FFT
computation, Berlekamp-Massey, Goppa-polynomial loading, and Gaussian elimination under those
physical threat models. Deployments exposed to those attackers need separately evaluated
masking, fault detection, and platform-specific countermeasures; this crate does not currently
provide them.

## Architecture-specific code

The algorithm is architecture independent; only the inner loops of a few data-oblivious
primitives are not. Every vector kernel has a scalar twin, and the two are checked against each
other rather than only against the KAT vectors.

| Primitive | AArch64 | x86-64 | Portable |
| --------- | ------- | ------ | -------- |
| Bit-matrix row XOR: 1, 8 and 16 wide, paired-destination, and panel-fold forms | inline assembly and NEON intrinsics | `vpternlogq` on AVX-512, `vpand`/`vpxor` on AVX2 | blocked word loops with the same one-pass schedule |
| Bit-sliced field multiply and square | NEON; the inner step fuses to `BCAX` where the `sha3` extension is available | SSE2 baseline; fused single and paired 256-bit forms via `vpternlogq` on AVX-512 | `u128` Karatsuba |
| Sorting-network comparators, contiguous short-stride stages, and lockstep chain batches | NEON `SMIN`/`SMAX` with `EXT`/`REV64` alignment and `BSL` blends | `pminsd`/`pmaxsd` tiers on SSE4.1 and AVX2, selected at run time | scalar arithmetic mask |
| Carry-less multiply | `vmull_p64` when `aes` is enabled | portable convolution (measured: faster than `pclmulqdq` here) | 13-term convolution |
| 64x64 bit transpose | shared delta-form word code | shared | shared |

Selection on x86-64 is a run-time decision made once per process from CPUID, cached in an
atomic, and never data dependent. Setting `PQ_MCELIECE_DISABLE_SIMD` to a non-empty value
forces the scalar tier on x86-64 — the row kernels, the sorting-network kernels, and the
fused and paired multiplies (the baseline SSE2 field multiply and the AArch64 NEON kernels
are compile-time and unconditional); CI runs the test suite under both settings and the
results must be identical, and the differential unit tests additionally check every vector
kernel against its scalar twin inside one binary.

The portable carry-less multiply is a fixed 13-term convolution built on integer
multiplication. On mainstream CPUs integer multiply latency is operand independent; on the
few embedded cores where it is not (early-terminating multipliers), the constant-time claim
would need re-evaluation for that platform.

### Constant time under vectorization

A vector kernel is only admissible if its instruction sequence and its memory access pattern
depend on public values alone — lengths, strides, and parameter-set constants. The audit
section below spells out what that allows and forbids; the design consequence worth stating
here is that it costs performance. Masked row operations touch every row whether their
condition bit is set or not, and the blocked forward elimination keeps a shadow word per row
rather than using the faster Method of Four Russians, because both shortcuts would trade a
timing channel for the speed.

### Constant-time audit of the vectorized paths

Every vector kernel — the x86 tiers and the AArch64 assembly and intrinsics alike — was
audited against the property the data-oblivious design exists to provide: **no branch, no
memory address, and no shift count may depend on a secret.** What that allows and forbids,
concretely:

- A secret bit becomes an all-zeros or all-ones mask by wrapping negation and reaches only the
  operand of an `AND`. It never reaches a comparison, a jump, or an index.
- The masked row operations read and write every row whether the condition is zero or one.
  Skipping the untaken rows would be faster and would leak which pivots were found where.
- Loop bounds come from lengths, strides, panel widths and the field's extension degree. All are
  public: they are properties of the parameter set, not of the key.
- The AVX-512 mask is broadcast into a general vector lane rather than an opmask register, so no
  argument about predication timing is needed.
- No lookup table is indexed by key material anywhere. In particular the blocked forward
  elimination deliberately avoids the Method of Four Russians, whose precomputed-combination
  table would be indexed by secret matrix bits.

**One regression was found by this audit and fixed.** An early version of the blocked forward
elimination skipped a trailing row whose conditions were all zero — `if any == 0 { continue }`.
Those conditions derive from secret matrix bits, so the skip was a timing oracle for how the
pivots fell. It is removed; the row is applied unconditionally and masked. Removing it cost nothing
measurable, because the conditions are roughly balanced and the skip almost never fired.

**Declassified, as in the reference implementation:** `MatGen` returning `⊥` is observable. Key
generation responds by restarting with a fresh seed, which is the specification's own behavior,
and the vectorized code declassifies exactly the same predicate the scalar code did.

**Statistical verification.** `tests/constant_time.rs` implements the `dudect` method (Reparaz,
Balasch and Verbauwhede, DATE 2017): time two input classes interleaved, and compare them with
Welch's t-statistic over progressively cropped samples. The `tests/` directory is not shipped
in the crates.io package, so run this from a repository checkout:

```text
cargo test --release --test constant_time -- --ignored --nocapture
```

Measured on an idle machine at 200,000 samples per class, `mceliece8192128`:

| property under test | max \|t\| | verdict |
| --- | --- | --- |
| decapsulation: valid vs corrupted ciphertext | 1.75 | no leak detected |
| decapsulation: fixed vs varying valid ciphertexts | 4.44 (2.42 at 400k) | no leak detected |
| decapsulation: fixed vs varying private keys | 1.29 | no leak detected |

The decision threshold is `|t| = 4.5`. What matters more than any single number is the trend: a
real effect makes the statistic *grow* with sample count, roughly as the square root. All three
decapsulation figures fall as samples increase, which is the signature of noise rather than
signal.

The suite contains a positive control that confirms the method detects real effects.
Encapsulation's `FixedWeight` sampler restarts when it cannot draw `t` distinct in-range indices,
so its running time genuinely varies with the randomness consumed; that test reports `|t|` rising
111 → 166 → 183 → 462 as samples increase. It is recorded as informational rather than asserted,
because the restarts depend on *rejected* candidates while the ciphertext commits only to the
accepted vector, so no secret is revealed. The specification defines the sampler this way and the
reference implementation behaves identically.

**What this audit is and is not.** It is a source-level classification of every branch in the
changed code, disassembly spot checks confirming the multiply compiles to straight-line vector
code, and the statistical tests above. It is not machine-checked. A compiler is entitled to
introduce a branch where the source has none; the disassembly checks cover the hot kernels but
not every path. Independent verification with a constant-time analysis tool is worth doing
before this is relied on in an adversarial setting.
