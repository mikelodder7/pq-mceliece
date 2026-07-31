/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Application of an in-place Beneš network.
//!
//! A field ordering is stored in a private key as the `(2m-1)2^(m-1)` control bits of a Beneš
//! network rather than as `q` field elements. Recovering the support means running that network
//! over the `q` bit-planes of the identity ordering.
//!
//! The implementation strategy is the one from Tung Chou's *McBits Revisited*
//! (<https://eprint.iacr.org/2017/793.pdf>): the `q` bits being permuted are held as 64-bit
//! words, so stages whose stride is at least 64 act directly on whole words. Stages with a
//! smaller stride are handled by transposing the bit matrix first, which turns the low bits of
//! a position index into word indices.

#[cfg(all(test, feature = "keygen"))]
use super::field::Field;
use super::params::Params;
use super::transpose::transpose_64x64;

/// Read a little-endian `u64` from a byte slice at `offset`.
#[inline]
fn load_u64(src: &[u8], offset: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&src[offset..offset + 8]);
    u64::from_le_bytes(buf)
}

/// Read a little-endian `u32` from a byte slice at `offset`, widened to `u64`.
#[inline]
fn load_u32(src: &[u8], offset: usize) -> u64 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&src[offset..offset + 4]);
    u32::from_le_bytes(buf) as u64
}

/// One stage of conditional swaps at stride `1 << lgs` over `data`.
///
/// Control words are consumed sequentially from `bits`, one per swapped pair of words.
#[inline]
fn layer(data: &mut [u64], bits: &[u64], lgs: usize) {
    let s = 1usize << lgs;
    let mut index = 0;
    let mut i = 0;
    while i < data.len() {
        for j in i..i + s {
            let d = (data[j] ^ data[j + s]) & bits[index];
            index += 1;
            data[j] ^= d;
            data[j + s] ^= d;
        }
        i += s * 2;
    }
}

/// One stage of conditional swaps applied to two interleaved halves.
///
/// The two halves share a single control-bit stream, alternating between them, which is how
/// the `m = 13` network addresses its 8192 bits with 64-bit words.
#[inline]
fn layer_interleaved(lo: &mut [u64; 64], hi: &mut [u64; 64], bits: &[u64], lgs: usize) {
    let s = 1usize << lgs;
    let mut index = 0;
    let mut i = 0;
    while i < 64 {
        for j in i..i + s {
            let d = (lo[j] ^ lo[j + s]) & bits[index];
            index += 1;
            lo[j] ^= d;
            lo[j + s] ^= d;

            let d = (hi[j] ^ hi[j + s]) & bits[index];
            index += 1;
            hi[j] ^= d;
            hi[j + s] ^= d;
        }
        i += s * 2;
    }
}

/// Walks the control-bit stream forwards for the network and backwards for its inverse.
struct CondReader {
    offset: usize,
    /// Bytes consumed per stage.
    stage: usize,
    reverse: bool,
}

impl CondReader {
    /// `stage` is the number of control bytes each stage consumes, and `stages` the total
    /// number of stages, so the reverse direction can start at the final stage.
    fn new(stage: usize, stages: usize, reverse: bool) -> Self {
        let offset = if reverse { (stages - 1) * stage } else { 0 };
        Self {
            offset,
            stage,
            reverse,
        }
    }

    /// The byte offset of the current stage, then advance to the next one.
    fn take(&mut self) -> usize {
        let at = self.offset;
        if self.reverse {
            // The final `take` of a run would underflow; the value is never read again.
            self.offset = self.offset.saturating_sub(self.stage);
        } else {
            self.offset += self.stage;
        }
        at
    }
}

/// Apply the `m = 12` network to 4096 bits held as 512 bytes.
fn apply_benes_12(r: &mut [u8], bits: &[u8], reverse: bool) {
    const STAGE: usize = 256;
    const STAGES: usize = 2 * 12 - 1;

    let mut bs = [0u64; 64];
    let mut cond = [0u64; 64];
    let mut reader = CondReader::new(STAGE, STAGES, reverse);

    for (i, word) in bs.iter_mut().enumerate() {
        *word = load_u64(r, i * 8);
    }

    // Stages 0..=5: stride below 64, so work on the transposed matrix. The control bits are
    // stored one 32-bit group per row and must be transposed the same way.
    transpose_64x64(&mut bs);
    for lgs in 0..6 {
        let at = reader.take();
        for (i, word) in cond.iter_mut().enumerate() {
            *word = load_u32(bits, at + i * 4);
        }
        transpose_64x64(&mut cond);
        layer(&mut bs, &cond, lgs);
    }
    transpose_64x64(&mut bs);

    // Stages 6..=16: stride of at least 64, acting directly on words.
    for lgs in (0..6).chain((0..5).rev()) {
        let at = reader.take();
        for (i, word) in cond.iter_mut().take(32).enumerate() {
            *word = load_u64(bits, at + i * 8);
        }
        layer(&mut bs, &cond, lgs);
    }

    // Stages 17..=22: mirror of the first block.
    transpose_64x64(&mut bs);
    for lgs in (0..6).rev() {
        let at = reader.take();
        for (i, word) in cond.iter_mut().enumerate() {
            *word = load_u32(bits, at + i * 4);
        }
        transpose_64x64(&mut cond);
        layer(&mut bs, &cond, lgs);
    }
    transpose_64x64(&mut bs);

    for (i, word) in bs.iter().enumerate() {
        r[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
}

/// Apply the `m = 13` network to 8192 bits held as 1024 bytes.
fn apply_benes_13(r: &mut [u8], bits: &[u8], reverse: bool) {
    const STAGE: usize = 512;
    const STAGES: usize = 2 * 13 - 1;

    // The 8192 bits are viewed as 64 rows of 128 bits. `lo`/`hi` hold the two 64-bit halves
    // of each row; `flat` holds the transposed form as one contiguous 128-word array so the
    // outer stages can address it with a single stride.
    let mut lo = [0u64; 64];
    let mut hi = [0u64; 64];
    let mut flat = [0u64; 128];
    let mut cond_v = [0u64; 64];
    let mut cond_h = [0u64; 64];
    let mut reader = CondReader::new(STAGE, STAGES, reverse);

    for i in 0..64 {
        lo[i] = load_u64(r, i * 16);
        hi[i] = load_u64(r, i * 16 + 8);
    }

    /// Transpose both halves from row form into `flat`.
    macro_rules! to_flat {
        () => {
            transpose_64x64(&mut lo);
            transpose_64x64(&mut hi);
            flat[..64].copy_from_slice(&lo);
            flat[64..].copy_from_slice(&hi);
        };
    }

    /// Transpose `flat` back into row form.
    macro_rules! from_flat {
        () => {
            lo.copy_from_slice(&flat[..64]);
            hi.copy_from_slice(&flat[64..]);
            transpose_64x64(&mut lo);
            transpose_64x64(&mut hi);
        };
    }

    to_flat!();
    for lgs in 0..=6 {
        let at = reader.take();
        for (i, word) in cond_v.iter_mut().enumerate() {
            *word = load_u64(bits, at + i * 8);
        }
        cond_h.copy_from_slice(&cond_v);
        transpose_64x64(&mut cond_h);
        layer(&mut flat, &cond_h, lgs);
    }
    from_flat!();

    for lgs in (0..=5).chain((0..=4).rev()) {
        let at = reader.take();
        for (i, word) in cond_v.iter_mut().enumerate() {
            *word = load_u64(bits, at + i * 8);
        }
        layer_interleaved(&mut lo, &mut hi, &cond_v, lgs);
    }

    to_flat!();
    for lgs in (0..=6).rev() {
        let at = reader.take();
        for (i, word) in cond_v.iter_mut().enumerate() {
            *word = load_u64(bits, at + i * 8);
        }
        cond_h.copy_from_slice(&cond_v);
        transpose_64x64(&mut cond_h);
        layer(&mut flat, &cond_h, lgs);
    }
    from_flat!();

    for i in 0..64 {
        r[i * 16..i * 16 + 8].copy_from_slice(&lo[i].to_le_bytes());
        r[i * 16 + 8..i * 16 + 16].copy_from_slice(&hi[i].to_le_bytes());
    }
}

/// Apply the Beneš network described by `bits` to the `q`-bit vector `r`.
///
/// `r` must be `q / 8` bytes and `bits` must be [`Params::COND_BYTES`] bytes. Setting
/// `reverse` applies the inverse permutation.
pub(crate) fn apply_benes<P: Params>(r: &mut [u8], bits: &[u8], reverse: bool) {
    debug_assert_eq!(r.len(), P::Q / 8);
    debug_assert_eq!(bits.len(), P::COND_BYTES);

    match P::M {
        12 => apply_benes_12(r, bits, reverse),
        _ => apply_benes_13(r, bits, reverse),
    }
}

/// Recover the support `(alpha_0, ..., alpha_{n-1})` from stored control bits.
///
/// Decoding no longer needs this: it works in field-element order, where the support is the
/// identity. It remains as a direct statement of what the control bits mean, which the key
/// generation tests check against.
///
/// The network is applied to the `m` bit-planes of the identity ordering; reading the planes
/// back column-wise reassembles one field element per position.
#[cfg(all(test, feature = "keygen"))]
pub(crate) fn support_gen<P: Params>(support: &mut [u16], cond: &[u8]) {
    debug_assert_eq!(support.len(), P::N);
    debug_assert_eq!(cond.len(), P::COND_BYTES);

    let plane_len = P::Q / 8;
    let mut planes = vec![0u8; P::M * plane_len];

    for i in 0..P::Q {
        let a = P::Field::bitrev(i as u16);
        for j in 0..P::M {
            planes[j * plane_len + i / 8] |= (((a >> j) & 1) as u8) << (i % 8);
        }
    }

    for j in 0..P::M {
        apply_benes::<P>(&mut planes[j * plane_len..(j + 1) * plane_len], cond, false);
    }

    for (i, out) in support.iter_mut().enumerate() {
        let mut value = 0u16;
        for j in (0..P::M).rev() {
            value <<= 1;
            value |= ((planes[j * plane_len + i / 8] >> (i % 8)) & 1) as u16;
        }
        *out = value;
    }
}

// The round-trip tests build control bits, which only key generation compiles.
#[cfg(all(test, feature = "keygen"))]
#[allow(clippy::unwrap_used)]
mod tests {
    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    use super::*;
    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    use crate::hazmat::controlbits::control_bits_from_permutation;

    // Every helper below feeds the two round-trip tests, which need a concrete parameter set
    // of each field width.
    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    struct Rng(u64);

    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Build a pseudo-random permutation of `0..n` by Fisher-Yates.
    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    fn permutation(n: usize, seed: u64) -> Vec<i16> {
        let mut rng = Rng(seed);
        let mut pi: Vec<i16> = (0..n as i16).collect();
        for i in (1..n).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            pi.swap(i, j);
        }
        pi
    }

    /// Apply a permutation directly to a bit vector, as an oracle for the network.
    ///
    /// The control bits produced for `pi` drive a network that gathers: output position `i`
    /// receives the input at position `pi[i]`.
    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    fn permute_bits(bits: &[u8], pi: &[i16]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len()];
        for (i, &source) in pi.iter().enumerate() {
            let s = source as usize;
            let bit = (bits[s / 8] >> (s % 8)) & 1;
            out[i / 8] |= bit << (i % 8);
        }
        out
    }

    #[cfg(any(feature = "mceliece348864", feature = "mceliece8192128"))]
    fn round_trip<P: Params>(seed: u64) {
        let pi = permutation(P::Q, seed);
        let mut cond = vec![0u8; P::COND_BYTES];
        assert!(control_bits_from_permutation(&mut cond, &pi, P::M));

        let mut rng = Rng(seed ^ 0xA5A5_A5A5);
        let mut data = vec![0u8; P::Q / 8];
        for byte in data.iter_mut() {
            *byte = rng.next() as u8;
        }

        let expected = permute_bits(&data, &pi);
        let mut forward = data.clone();
        apply_benes::<P>(&mut forward, &cond, false);
        assert_eq!(forward, expected, "{} forward network", P::NAME);

        let mut back = forward;
        apply_benes::<P>(&mut back, &cond, true);
        assert_eq!(back, data, "{} inverse network", P::NAME);
    }

    #[test]
    #[cfg(feature = "mceliece348864")]
    fn network_realizes_the_permutation_for_m12() {
        use crate::hazmat::params::McEliece348864;
        round_trip::<McEliece348864>(0x1111_2222_3333_4444);
        round_trip::<McEliece348864>(0x5555_6666_7777_8888);
    }

    #[test]
    #[cfg(feature = "mceliece8192128")]
    fn network_realizes_the_permutation_for_m13() {
        use crate::hazmat::params::McEliece8192128;
        round_trip::<McEliece8192128>(0x0F1E_2D3C_4B5A_6978);
        round_trip::<McEliece8192128>(0xFEDC_BA98_7654_3210);
    }

    #[test]
    #[cfg(feature = "mceliece8192128")]
    fn support_matches_the_permutation_it_encodes() {
        use crate::hazmat::params::McEliece8192128;
        type P = McEliece8192128;

        let pi = permutation(P::Q, 0x2468_ACE0_1357_9BDF);
        let mut cond = vec![0u8; P::COND_BYTES];
        assert!(control_bits_from_permutation(&mut cond, &pi, P::M));

        let mut support = vec![0u16; P::N];
        support_gen::<P>(&mut support, &cond);

        // The plane for bit `j` starts out holding bit `j` of `bitrev(index)`, so after the
        // gathering network position `i` holds `bitrev(pi[i])`.
        for (i, &value) in support.iter().enumerate() {
            let expected = <P as Params>::Field::bitrev(pi[i] as u16);
            assert_eq!(value, expected, "support position {i}");
        }
    }
}
