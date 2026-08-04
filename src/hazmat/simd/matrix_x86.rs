/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! x86 vector kernels for the bit-matrix elimination that dominates key generation.
//!
//! These are the x86 counterparts of the AArch64 assembly in [`matrix`](crate::hazmat::matrix).
//! Every kernel computes the same primitive:
//!
//! ```text
//! destination[i] ^= source[i] & mask
//! ```
//!
//! where `mask` is `0` or `u64::MAX`, formed by negating a secret matrix bit. Nothing branches
//! on `mask`; it only ever reaches the operand of an `AND`.
//!
//! # Why the kernels are blocked
//!
//! Widening the inner loop is the smaller half of the win. For `mceliece8192128` the matrix is
//! 1664 rows of 129 words, about 1.7 MiB, so elimination is bound by how many times that
//! region streams past the caches rather than by arithmetic. Folding eight source rows into one
//! destination in a single pass, or one source row into eight destinations, divides the number
//! of streaming passes by eight. That is the same reason the AArch64 path has these three
//! shapes, and it is the part an autovectorizer cannot do for you: it can widen a loop body,
//! but it will not restructure a loop nest.
//!
//! # Why `vpternlogq`
//!
//! `dst ^ (src & mask)` is a three-input bitwise function, so AVX-512 computes it in one
//! instruction with immediate `0x78`. Writing the mask into a general-purpose lane rather than
//! into an AVX-512 `k` register is deliberate: predication timing would otherwise need a
//! separate constant-time argument per microarchitecture, and the ternary-logic form is at
//! least as fast anyway.

use super::{Level, level};

/// Words processed per iteration of the 512-bit kernels.
const AVX512_LANES: usize = 8;
/// Words processed per iteration of the 256-bit kernels.
const AVX2_LANES: usize = 4;

/// Turn a secret 0/1 condition into an all-zeros or all-ones mask.
///
/// Wrapping negation, so no branch and no comparison instruction appears.
#[inline(always)]
fn mask_of(condition: u64) -> u64 {
    0u64.wrapping_sub(condition)
}

/// Expand `conditions` into masks, preserving order.
#[inline(always)]
fn masks_of<const N: usize>(conditions: &[u64; N]) -> [u64; N] {
    let mut masks = [0u64; N];
    for (slot, &condition) in masks.iter_mut().zip(conditions.iter()) {
        *slot = mask_of(condition);
    }
    masks
}

// ---------------------------------------------------------------------------------------------
// One source row into one destination row.
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f")]
unsafe fn xor_row_avx512(destination: *mut u64, source: *const u64, len: usize, mask: u64) {
    use core::arch::x86_64::{
        _mm512_loadu_epi64, _mm512_set1_epi64, _mm512_storeu_epi64, _mm512_ternarylogic_epi64,
    };

    // SAFETY: the caller guarantees `len` readable words at `source` and `len` writable words
    // at `destination`, and that the two regions do not overlap. Loads and stores are the
    // unaligned forms, so no alignment precondition applies.
    unsafe {
        let broadcast = _mm512_set1_epi64(mask as i64);
        let mut i = 0;
        while i + AVX512_LANES <= len {
            let d = _mm512_loadu_epi64(destination.add(i) as *const i64);
            let s = _mm512_loadu_epi64(source.add(i) as *const i64);
            // 0x78 is the truth table of `a ^ (b & c)`.
            let out = _mm512_ternarylogic_epi64::<0x78>(d, s, broadcast);
            _mm512_storeu_epi64(destination.add(i) as *mut i64, out);
            i += AVX512_LANES;
        }
        while i < len {
            *destination.add(i) ^= *source.add(i) & mask;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn xor_row_avx2(destination: *mut u64, source: *const u64, len: usize, mask: u64) {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_storeu_si256,
        _mm256_xor_si256,
    };

    // SAFETY: same contract as the 512-bit kernel above.
    unsafe {
        let broadcast = _mm256_set1_epi64x(mask as i64);
        let mut i = 0;
        while i + AVX2_LANES <= len {
            let d = _mm256_loadu_si256(destination.add(i) as *const __m256i);
            let s = _mm256_loadu_si256(source.add(i) as *const __m256i);
            let out = _mm256_xor_si256(d, _mm256_and_si256(s, broadcast));
            _mm256_storeu_si256(destination.add(i) as *mut __m256i, out);
            i += AVX2_LANES;
        }
        while i < len {
            *destination.add(i) ^= *source.add(i) & mask;
            i += 1;
        }
    }
}

/// XOR `source` into `destination` when `condition` is one.
///
/// The lengths and addresses are public; `condition` is a secret matrix bit and reaches only
/// the operand of an `AND`.
#[inline(never)]
pub(crate) fn xor_row_if(destination: &mut [u64], source: &[u64], condition: u64) {
    debug_assert_eq!(destination.len(), source.len());

    let len = destination.len();
    let mask = mask_of(condition);
    let dst = destination.as_mut_ptr();
    let src = source.as_ptr();

    match level() {
        // SAFETY: the level was detected from CPUID, so the target features the kernel names
        // are present. `destination` and `source` are distinct borrows of the same length, so
        // both cover `len` valid words and cannot overlap.
        Level::Avx512 => unsafe { xor_row_avx512(dst, src, len, mask) },
        // SAFETY: as above.
        Level::Avx2 => unsafe { xor_row_avx2(dst, src, len, mask) },
        Level::Scalar => {
            for (destination, &source) in destination.iter_mut().zip(source.iter()) {
                *destination ^= source & mask;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// One source row into `N` destination rows.
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f")]
unsafe fn xor_rows_avx512<const N: usize>(
    destinations: *mut u64,
    source: *const u64,
    stride: usize,
    len: usize,
    masks: &[u64; N],
) {
    use core::arch::x86_64::{
        _mm512_loadu_epi64, _mm512_set1_epi64, _mm512_storeu_epi64, _mm512_ternarylogic_epi64,
    };

    // SAFETY: the caller guarantees `N` destination rows of `stride` words starting at
    // `destinations`, each with `len` valid words from the given offset, one disjoint source
    // row with the same, and that no destination aliases the source.
    unsafe {
        let mut broadcasts = [_mm512_set1_epi64(0); N];
        for (slot, &mask) in broadcasts.iter_mut().zip(masks.iter()) {
            *slot = _mm512_set1_epi64(mask as i64);
        }

        let mut i = 0;
        while i + AVX512_LANES <= len {
            // The source chunk is loaded once and reused across every destination, which is
            // the whole point of the blocked shape.
            let s = _mm512_loadu_epi64(source.add(i) as *const i64);
            for (row, &broadcast) in broadcasts.iter().enumerate() {
                let at = destinations.add(row * stride + i);
                let d = _mm512_loadu_epi64(at as *const i64);
                _mm512_storeu_epi64(
                    at as *mut i64,
                    _mm512_ternarylogic_epi64::<0x78>(d, s, broadcast),
                );
            }
            i += AVX512_LANES;
        }
        while i < len {
            let s = *source.add(i);
            for (row, &mask) in masks.iter().enumerate() {
                *destinations.add(row * stride + i) ^= s & mask;
            }
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn xor_rows_avx2<const N: usize>(
    destinations: *mut u64,
    source: *const u64,
    stride: usize,
    len: usize,
    masks: &[u64; N],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    // SAFETY: same contract as the 512-bit kernel above.
    unsafe {
        let mut broadcasts = [_mm256_setzero_si256(); N];
        for (slot, &mask) in broadcasts.iter_mut().zip(masks.iter()) {
            *slot = _mm256_set1_epi64x(mask as i64);
        }

        let mut i = 0;
        while i + AVX2_LANES <= len {
            let s = _mm256_loadu_si256(source.add(i) as *const __m256i);
            for (row, &broadcast) in broadcasts.iter().enumerate() {
                let at = destinations.add(row * stride + i);
                let d = _mm256_loadu_si256(at as *const __m256i);
                let out = _mm256_xor_si256(d, _mm256_and_si256(s, broadcast));
                _mm256_storeu_si256(at as *mut __m256i, out);
            }
            i += AVX2_LANES;
        }
        while i < len {
            let s = *source.add(i);
            for (row, &mask) in masks.iter().enumerate() {
                *destinations.add(row * stride + i) ^= s & mask;
            }
            i += 1;
        }
    }
}

/// XOR one source row into `N` destination rows, each gated by its own condition.
///
/// `destinations` holds `N` complete rows of `stride` words; `source` is a disjoint row of the
/// same width. Only words `first..stride` are touched, which is how the caller skips the
/// columns elimination has already reduced to zero in both rows.
#[inline(never)]
pub(crate) fn xor_rows_if<const N: usize>(
    destinations: &mut [u64],
    source: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; N],
) {
    debug_assert_eq!(destinations.len(), N * stride);
    debug_assert_eq!(source.len(), stride);
    debug_assert!(first <= stride);

    let masks = masks_of(conditions);
    let len = stride - first;
    if len == 0 {
        return;
    }

    // SAFETY: shared by all three arms. `destinations` is exactly `N * stride` words, so row
    // `row` starts at `row * stride` and `first + len == stride` words are in bounds for each.
    // `source` is `stride` words and `first + len == stride` is in bounds for it too. The two
    // slices come from disjoint mutable and shared borrows, so they cannot alias. The detected
    // level guarantees the named target features exist.
    unsafe {
        let dst = destinations.as_mut_ptr().add(first);
        let src = source.as_ptr().add(first);
        match level() {
            Level::Avx512 => xor_rows_avx512::<N>(dst, src, stride, len, &masks),
            Level::Avx2 => xor_rows_avx2::<N>(dst, src, stride, len, &masks),
            Level::Scalar => {
                for (row, &mask) in masks.iter().enumerate() {
                    for i in 0..len {
                        *dst.add(row * stride + i) ^= *src.add(i) & mask;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// `N` source rows into one destination row.
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f")]
unsafe fn xor_sources_avx512<const N: usize>(
    destination: *mut u64,
    sources: *const u64,
    stride: usize,
    len: usize,
    masks: &[u64; N],
) {
    use core::arch::x86_64::{
        _mm512_loadu_epi64, _mm512_set1_epi64, _mm512_storeu_epi64, _mm512_ternarylogic_epi64,
        _mm512_xor_si512,
    };

    // SAFETY: the caller guarantees one destination row and `N` disjoint source rows, each of
    // `stride` words with `len` valid from the given offset.
    unsafe {
        let mut broadcasts = [_mm512_set1_epi64(0); N];
        for (slot, &mask) in broadcasts.iter_mut().zip(masks.iter()) {
            *slot = _mm512_set1_epi64(mask as i64);
        }

        let mut i = 0;
        while i + AVX512_LANES <= len {
            // The destination chunk is loaded and stored once for all `N` sources, so the
            // destination streams past the cache once instead of `N` times. The fold runs as
            // two independent accumulator chains joined at the store: one chain would make
            // every ternary-logic op wait on the previous one, and the sources arrive from
            // cache faster than that.
            let mut even = _mm512_loadu_epi64(destination.add(i) as *const i64);
            let mut odd = _mm512_set1_epi64(0);
            let (pairs, tail) = broadcasts.as_chunks::<2>();
            for (pair_index, pair) in pairs.iter().enumerate() {
                let row = 2 * pair_index;
                let s0 = _mm512_loadu_epi64(sources.add(row * stride + i) as *const i64);
                let s1 = _mm512_loadu_epi64(sources.add((row + 1) * stride + i) as *const i64);
                even = _mm512_ternarylogic_epi64::<0x78>(even, s0, pair[0]);
                odd = _mm512_ternarylogic_epi64::<0x78>(odd, s1, pair[1]);
            }
            if let Some(&last) = tail.first() {
                let s = _mm512_loadu_epi64(sources.add((N - 1) * stride + i) as *const i64);
                even = _mm512_ternarylogic_epi64::<0x78>(even, s, last);
            }
            _mm512_storeu_epi64(destination.add(i) as *mut i64, _mm512_xor_si512(even, odd));
            i += AVX512_LANES;
        }
        while i < len {
            let mut acc = *destination.add(i);
            for (row, &mask) in masks.iter().enumerate() {
                acc ^= *sources.add(row * stride + i) & mask;
            }
            *destination.add(i) = acc;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn xor_sources_avx2<const N: usize>(
    destination: *mut u64,
    sources: *const u64,
    stride: usize,
    len: usize,
    masks: &[u64; N],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    // SAFETY: same contract as the 512-bit kernel above.
    unsafe {
        let mut broadcasts = [_mm256_setzero_si256(); N];
        for (slot, &mask) in broadcasts.iter_mut().zip(masks.iter()) {
            *slot = _mm256_set1_epi64x(mask as i64);
        }

        let mut i = 0;
        while i + AVX2_LANES <= len {
            // Two accumulator chains joined at the store, for the same reason as the 512-bit
            // kernel above.
            let mut even = _mm256_loadu_si256(destination.add(i) as *const __m256i);
            let mut odd = _mm256_setzero_si256();
            let (pairs, tail) = broadcasts.as_chunks::<2>();
            for (pair_index, pair) in pairs.iter().enumerate() {
                let row = 2 * pair_index;
                let s0 = _mm256_loadu_si256(sources.add(row * stride + i) as *const __m256i);
                let s1 = _mm256_loadu_si256(sources.add((row + 1) * stride + i) as *const __m256i);
                even = _mm256_xor_si256(even, _mm256_and_si256(s0, pair[0]));
                odd = _mm256_xor_si256(odd, _mm256_and_si256(s1, pair[1]));
            }
            if let Some(&last) = tail.first() {
                let s = _mm256_loadu_si256(sources.add((N - 1) * stride + i) as *const __m256i);
                even = _mm256_xor_si256(even, _mm256_and_si256(s, last));
            }
            _mm256_storeu_si256(
                destination.add(i) as *mut __m256i,
                _mm256_xor_si256(even, odd),
            );
            i += AVX2_LANES;
        }
        while i < len {
            let mut acc = *destination.add(i);
            for (row, &mask) in masks.iter().enumerate() {
                acc ^= *sources.add(row * stride + i) & mask;
            }
            *destination.add(i) = acc;
            i += 1;
        }
    }
}

/// XOR `N` masked source rows into one destination row.
///
/// `sources` holds `N` complete rows of `stride` words; `destination` is a disjoint row of the
/// same width. Only words `first..stride` are touched.
#[inline(never)]
pub(crate) fn xor_sources_if<const N: usize>(
    destination: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; N],
) {
    debug_assert_eq!(destination.len(), stride);
    debug_assert_eq!(sources.len(), N * stride);
    debug_assert!(first <= stride);

    let masks = masks_of(conditions);
    let len = stride - first;
    if len == 0 {
        return;
    }

    // SAFETY: mirrors `xor_rows_if` — `sources` is exactly `N * stride` words so every row's
    // `first + len == stride` words are in bounds, `destination` is `stride` words, the two
    // come from disjoint borrows, and the detected level guarantees the target features.
    unsafe {
        let dst = destination.as_mut_ptr().add(first);
        let src = sources.as_ptr().add(first);
        match level() {
            Level::Avx512 => xor_sources_avx512::<N>(dst, src, stride, len, &masks),
            Level::Avx2 => xor_sources_avx2::<N>(dst, src, stride, len, &masks),
            Level::Scalar => {
                for i in 0..len {
                    let mut acc = *dst.add(i);
                    for (row, &mask) in masks.iter().enumerate() {
                        acc ^= *src.add(row * stride + i) & mask;
                    }
                    *dst.add(i) = acc;
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// The definition every kernel must agree with, written the obvious way.
    fn reference_rows<const N: usize>(
        destinations: &mut [u64],
        source: &[u64],
        stride: usize,
        first: usize,
        conditions: &[u64; N],
    ) {
        for (row, &condition) in conditions.iter().enumerate() {
            let mask = 0u64.wrapping_sub(condition);
            for i in first..stride {
                destinations[row * stride + i] ^= source[i] & mask;
            }
        }
    }

    fn reference_sources<const N: usize>(
        destination: &mut [u64],
        sources: &[u64],
        stride: usize,
        first: usize,
        conditions: &[u64; N],
    ) {
        for (row, &condition) in conditions.iter().enumerate() {
            let mask = 0u64.wrapping_sub(condition);
            for i in first..stride {
                destination[i] ^= sources[row * stride + i] & mask;
            }
        }
    }

    /// Strides that exercise every tail: exact multiples of eight and four, and every residue
    /// in between, plus the real widths of the standardized parameter sets.
    const STRIDES: [usize; 12] = [1, 2, 3, 4, 5, 7, 8, 9, 16, 55, 110, 129];

    #[test]
    fn single_row_matches_the_definition() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for stride in STRIDES {
            for first in [0, stride / 2, stride] {
                for condition in [0u64, 1] {
                    let source: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let original: Vec<u64> = (0..stride).map(|_| rng.next()).collect();

                    let mut expected = original.clone();
                    reference_rows(&mut expected, &source, stride, first, &[condition]);

                    let mut actual = original;
                    xor_row_if(&mut actual[first..], &source[first..], condition);

                    assert_eq!(actual, expected, "stride {stride} first {first}");
                }
            }
        }
    }

    #[test]
    fn eight_destinations_match_the_definition() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for stride in STRIDES {
            for first in [0, stride / 2, stride] {
                let source: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let original: Vec<u64> = (0..8 * stride).map(|_| rng.next()).collect();
                let mut conditions = [0u64; 8];
                for slot in conditions.iter_mut() {
                    *slot = rng.next() & 1;
                }

                let mut expected = original.clone();
                reference_rows(&mut expected, &source, stride, first, &conditions);

                let mut actual = original;
                xor_rows_if::<8>(&mut actual, &source, stride, first, &conditions);

                assert_eq!(actual, expected, "stride {stride} first {first}");
            }
        }
    }

    #[test]
    fn eight_sources_match_the_definition() {
        let mut rng = Rng(0xBF58_476D_1CE4_E5B9);
        for stride in STRIDES {
            for first in [0, stride / 2, stride] {
                let sources: Vec<u64> = (0..8 * stride).map(|_| rng.next()).collect();
                let original: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let mut conditions = [0u64; 8];
                for slot in conditions.iter_mut() {
                    *slot = rng.next() & 1;
                }

                let mut expected = original.clone();
                reference_sources(&mut expected, &sources, stride, first, &conditions);

                let mut actual = original;
                xor_sources_if::<8>(&mut actual, &sources, stride, first, &conditions);

                assert_eq!(actual, expected, "stride {stride} first {first}");
            }
        }
    }

    #[test]
    fn sixteen_sources_match_the_definition() {
        let mut rng = Rng(0x94D0_49BB_1331_11EB);
        for stride in STRIDES {
            for first in [0, stride / 2, stride] {
                let sources: Vec<u64> = (0..16 * stride).map(|_| rng.next()).collect();
                let original: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let mut conditions = [0u64; 16];
                for slot in conditions.iter_mut() {
                    *slot = rng.next() & 1;
                }

                let mut expected = original.clone();
                reference_sources(&mut expected, &sources, stride, first, &conditions);

                let mut actual = original;
                xor_sources_if::<16>(&mut actual, &sources, stride, first, &conditions);

                assert_eq!(actual, expected, "stride {stride} first {first}");
            }
        }
    }

    /// A zero condition must leave the destination untouched, whatever the source holds.
    #[test]
    fn zero_conditions_are_a_no_op() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let stride = 129;
        let sources: Vec<u64> = (0..16 * stride).map(|_| rng.next()).collect();
        let original: Vec<u64> = (0..stride).map(|_| rng.next()).collect();

        let mut actual = original.clone();
        xor_sources_if::<16>(&mut actual, &sources, stride, 0, &[0u64; 16]);
        assert_eq!(actual, original);
    }
}
