/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! A dense bit matrix over `F_2` stored as 64-bit words.
//!
//! The reference implementation eliminates a row at a time byte by byte. Storing each row as
//! `u64` words instead makes the inner loop eight times shorter and lets the compiler use the
//! machine's widest vector registers for it, which matters because reducing `N` to systematic
//! form dominates key generation.
//!
//! Bit `b` of a row lives in bit `b % 64` of word `b / 64`, matching the little-endian byte
//! order the specification uses for bit vectors. Each row carries one spare word past the end
//! so that a 64-bit read at any bit offset within the row never runs off the end.

/// A `rows x columns` matrix over `F_2`.
pub(crate) struct BitMatrix {
    rows: usize,
    /// Words per row, including the trailing guard word.
    stride: usize,
    data: Vec<u64>,
}

/// XOR `source` into `destination` when `condition` is one.
///
/// The lengths and addresses are public. `condition` is a secret matrix bit and must influence
/// only predicated arithmetic, never control flow.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn xor_row_if(destination: &mut [u64], source: &[u64], condition: u64) {
    use core::arch::asm;

    debug_assert_eq!(destination.len(), source.len());

    // SAFETY: both pointers cover `remaining` readable words, `destination` is writable, and
    // Rust's mutable/shared borrows make the regions disjoint. The loop handles two words only
    // while at least two remain, then handles at most one tail word. Its branches depend solely
    // on the public length; the secret condition is used only to form a mask.
    unsafe {
        asm!(
            "mov x9, {destination}",
            "mov x10, {source}",
            "mov x11, {remaining}",
            "neg x12, {condition}",
            "dup v0.2d, x12",
            "cmp x11, #2",
            "b.lo 3f",
            "2:",
            "ldr q1, [x10], #16",
            "ldr q2, [x9]",
            "and v1.16b, v1.16b, v0.16b",
            "eor v2.16b, v2.16b, v1.16b",
            "str q2, [x9], #16",
            "sub x11, x11, #2",
            "cmp x11, #2",
            "b.hs 2b",
            "3:",
            "cbz x11, 4f",
            "ldr x13, [x10]",
            "ldr x14, [x9]",
            "and x13, x13, x12",
            "eor x14, x14, x13",
            "str x14, [x9]",
            "4:",
            condition = in(reg) condition,
            destination = in(reg) destination.as_mut_ptr(),
            source = in(reg) source.as_ptr(),
            remaining = in(reg) source.len(),
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            options(nostack)
        );
    }
}

/// On x86 the same primitive lives in the SIMD module, which is where every `core::arch`
/// intrinsic in the crate is quarantined so its constant-time argument can be audited on its
/// own. It dispatches to a 512-bit, 256-bit or scalar body at runtime.
#[cfg(target_arch = "x86_64")]
use super::simd::matrix_x86::xor_row_if;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(never)]
fn xor_row_if(destination: &mut [u64], source: &[u64], condition: u64) {
    use ctutils::{Choice, CtSelect};

    debug_assert_eq!(destination.len(), source.len());
    let choice = Choice::from_u64_lsb(condition);
    for (destination, &source) in destination.iter_mut().zip(source.iter()) {
        let sum = *destination ^ source;
        *destination = destination.ct_select(&sum, choice);
    }
}

// The blocked kernels fold several rows into one streaming pass. The blocked shape is a
// memory-traffic optimization rather than a width one: for the larger parameter sets the
// matrix does not fit in L2, so what matters is how many times it streams past the caches,
// and folding eight or sixteen rows into one pass divides that by as much. Every target gets
// the shape; which kernel implements it is per-architecture, with a plain-word form for the
// rest.

/// The portable form of the destination-blocked kernel: one masked pass of the source over
/// each of `COUNT` cached destination rows.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn xor_rows_portable<const COUNT: usize>(
    destinations: &mut [u64],
    source: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; COUNT],
) {
    debug_assert!(COUNT.is_multiple_of(4));
    let mut rows = destinations.chunks_exact_mut(stride);
    for quad in 0..COUNT / 4 {
        // Four destination rows advance together: the source word is loaded once for all
        // four, and four independent store streams stay in flight.
        let (Some(r0), Some(r1), Some(r2), Some(r3)) =
            (rows.next(), rows.next(), rows.next(), rows.next())
        else {
            return;
        };
        let masks: [u64; 4] = [
            0u64.wrapping_sub(conditions[4 * quad]),
            0u64.wrapping_sub(conditions[4 * quad + 1]),
            0u64.wrapping_sub(conditions[4 * quad + 2]),
            0u64.wrapping_sub(conditions[4 * quad + 3]),
        ];
        for ((((t0, t1), t2), t3), &word) in r0[first..]
            .iter_mut()
            .zip(r1[first..].iter_mut())
            .zip(r2[first..].iter_mut())
            .zip(r3[first..].iter_mut())
            .zip(source[first..].iter())
        {
            *t0 ^= word & masks[0];
            *t1 ^= word & masks[1];
            *t2 ^= word & masks[2];
            *t3 ^= word & masks[3];
        }
    }
}

/// The portable form of the source-blocked kernel: `COUNT` masked source rows folded into one
/// destination row that stays cached across the whole fold.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn xor_sources_portable<const COUNT: usize>(
    destination: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; COUNT],
) {
    debug_assert!(COUNT.is_multiple_of(4));
    for quad in 0..COUNT / 4 {
        // Four masked source rows fold into the destination per pass: a quarter of the
        // destination read-and-write passes, with four load streams in flight.
        let row = |r: usize| &sources[r * stride..(r + 1) * stride][first..];
        let (r0, r1, r2, r3) = (
            row(4 * quad),
            row(4 * quad + 1),
            row(4 * quad + 2),
            row(4 * quad + 3),
        );
        let masks: [u64; 4] = [
            0u64.wrapping_sub(conditions[4 * quad]),
            0u64.wrapping_sub(conditions[4 * quad + 1]),
            0u64.wrapping_sub(conditions[4 * quad + 2]),
            0u64.wrapping_sub(conditions[4 * quad + 3]),
        ];
        for ((((target, &a), &b), &c), &d) in destination[first..]
            .iter_mut()
            .zip(r0.iter())
            .zip(r1.iter())
            .zip(r2.iter())
            .zip(r3.iter())
        {
            *target ^= (a & masks[0]) ^ (b & masks[1]) ^ (c & masks[2]) ^ (d & masks[3]);
        }
    }
}

/// XOR one source row into eight destinations, each gated by its own condition.
#[inline(always)]
fn xor_eight_rows(
    destinations: &mut [u64],
    source: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; 8],
) {
    #[cfg(target_arch = "aarch64")]
    xor_eight_rows_if_compact(destinations, source, stride, first, *conditions);
    #[cfg(target_arch = "x86_64")]
    super::simd::matrix_x86::xor_rows_if::<8>(destinations, source, stride, first, conditions);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    xor_rows_portable::<8>(destinations, source, stride, first, conditions);
}

/// XOR eight masked source rows into one destination.
#[inline(always)]
fn xor_eight_sources(
    destination: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; 8],
) {
    #[cfg(target_arch = "aarch64")]
    xor_eight_sources_if_compact(destination, sources, stride, first, *conditions);
    #[cfg(target_arch = "x86_64")]
    super::simd::matrix_x86::xor_sources_if::<8>(destination, sources, stride, first, conditions);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    xor_sources_portable::<8>(destination, sources, stride, first, conditions);
}

/// XOR sixteen masked source rows into one destination.
#[inline(always)]
fn xor_sixteen_sources(
    destination: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; 16],
) {
    #[cfg(target_arch = "aarch64")]
    xor_sixteen_sources_if_compact(destination, sources, stride, first, *conditions);
    #[cfg(target_arch = "x86_64")]
    super::simd::matrix_x86::xor_sources_if::<16>(destination, sources, stride, first, conditions);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    xor_sources_portable::<16>(destination, sources, stride, first, conditions);
}

/// XOR sixteen masked source rows into each of two consecutive destination rows.
///
/// On x86 the pair form loads every source chunk once for both destinations; elsewhere it is
/// exactly two single-destination passes, in row order.
#[inline(always)]
fn xor_sixteen_sources_pair(
    destinations: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: &[[u64; 16]; 2],
) {
    #[cfg(target_arch = "x86_64")]
    super::simd::matrix_x86::xor_sources_pair_if::<16>(
        destinations,
        sources,
        stride,
        first,
        conditions,
    );
    #[cfg(not(target_arch = "x86_64"))]
    {
        let (first_row, second_row) = destinations.split_at_mut(stride);
        xor_sixteen_sources(first_row, sources, stride, first, &conditions[0]);
        xor_sixteen_sources(second_row, sources, stride, first, &conditions[1]);
    }
}

#[cfg(target_arch = "aarch64")]
macro_rules! compact_xor_row_pair {
    ($mask:literal) => {
        concat!(
            "ldp q2, q3, [x13]\n",
            "and v4.16b, v0.16b, ",
            $mask,
            ".16b\n",
            "and v5.16b, v1.16b, ",
            $mask,
            ".16b\n",
            "eor v2.16b, v2.16b, v4.16b\n",
            "eor v3.16b, v3.16b, v5.16b\n",
            "stp q2, q3, [x13]\n",
            "add x13, x13, x12"
        )
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! compact_xor_row_one {
    ($mask:literal) => {
        concat!(
            "ldr q1, [x13]\n",
            "and v2.16b, v0.16b, ",
            $mask,
            ".16b\n",
            "eor v1.16b, v1.16b, v2.16b\n",
            "str q1, [x13]\n",
            "add x13, x13, x12"
        )
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! compact_xor_row_scalar {
    ($mask:literal) => {
        concat!(
            "fmov x14, ",
            $mask,
            "\n",
            "and x14, x8, x14\n",
            "ldr x10, [x13]\n",
            "eor x10, x10, x14\n",
            "str x10, [x13]\n",
            "add x13, x13, x12"
        )
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! compact_xor_source_pair {
    ($mask:literal) => {
        concat!(
            "ldp q2, q3, [x13]\n",
            "and v2.16b, v2.16b, ",
            $mask,
            ".16b\n",
            "and v3.16b, v3.16b, ",
            $mask,
            ".16b\n",
            "eor v0.16b, v0.16b, v2.16b\n",
            "eor v1.16b, v1.16b, v3.16b\n",
            "add x13, x13, x12"
        )
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! compact_xor_source_one {
    ($mask:literal) => {
        concat!(
            "ldr q1, [x13]\n",
            "and v1.16b, v1.16b, ",
            $mask,
            ".16b\n",
            "eor v0.16b, v0.16b, v1.16b\n",
            "add x13, x13, x12"
        )
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! compact_xor_source_scalar {
    ($mask:literal) => {
        concat!(
            "fmov x14, ",
            $mask,
            "\n",
            "ldr x10, [x13]\n",
            "and x10, x10, x14\n",
            "eor x8, x8, x10\n",
            "add x13, x13, x12"
        )
    };
}

/// XOR one source into eight destination rows using only caller-saved registers.
///
/// Streaming one destination at a time uses fewer live vectors than keeping all eight rows in
/// registers. That avoids saving and restoring the ABI's callee-saved SIMD registers around
/// every short elimination pass.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn xor_eight_rows_if_compact(
    destinations: &mut [u64],
    source: &[u64],
    stride: usize,
    first: usize,
    conditions: [u64; 8],
) {
    use core::arch::asm;

    debug_assert_eq!(destinations.len(), 8 * stride);
    debug_assert_eq!(source.len(), stride);

    // SAFETY: the destination contains eight complete disjoint rows and the source is a ninth
    // disjoint row. Adding `first` leaves `stride - first` valid words in every row. The main
    // vector loop advances four words at a time; vector and scalar tails handle two and one.
    // Secret conditions are used only to form masks. The assembly clobbers exclusively
    // caller-saved integer and SIMD registers, so it needs no hidden save/restore traffic.
    unsafe {
        asm!(
            "ldr x8, [{conditions}, #0]",
            "neg x8, x8",
            "dup v16.2d, x8",
            "ldr x8, [{conditions}, #8]",
            "neg x8, x8",
            "dup v17.2d, x8",
            "ldr x8, [{conditions}, #16]",
            "neg x8, x8",
            "dup v18.2d, x8",
            "ldr x8, [{conditions}, #24]",
            "neg x8, x8",
            "dup v19.2d, x8",
            "ldr x8, [{conditions}, #32]",
            "neg x8, x8",
            "dup v20.2d, x8",
            "ldr x8, [{conditions}, #40]",
            "neg x8, x8",
            "dup v21.2d, x8",
            "ldr x8, [{conditions}, #48]",
            "neg x8, x8",
            "dup v22.2d, x8",
            "ldr x8, [{conditions}, #56]",
            "neg x8, x8",
            "dup v23.2d, x8",
            "mov x9, {destinations}",
            "mov x10, {source}",
            "mov x11, {remaining}",
            "mov x12, {stride_bytes}",
            "cmp x11, #4",
            "b.lo 3f",
            "2:",
            "ldp q0, q1, [x10], #32",
            "mov x13, x9",
            compact_xor_row_pair!("v16"),
            compact_xor_row_pair!("v17"),
            compact_xor_row_pair!("v18"),
            compact_xor_row_pair!("v19"),
            compact_xor_row_pair!("v20"),
            compact_xor_row_pair!("v21"),
            compact_xor_row_pair!("v22"),
            compact_xor_row_pair!("v23"),
            "add x9, x9, #32",
            "sub x11, x11, #4",
            "cmp x11, #4",
            "b.hs 2b",
            "3:",
            "cmp x11, #2",
            "b.lo 4f",
            "ldr q0, [x10], #16",
            "mov x13, x9",
            compact_xor_row_one!("v16"),
            compact_xor_row_one!("v17"),
            compact_xor_row_one!("v18"),
            compact_xor_row_one!("v19"),
            compact_xor_row_one!("v20"),
            compact_xor_row_one!("v21"),
            compact_xor_row_one!("v22"),
            compact_xor_row_one!("v23"),
            "add x9, x9, #16",
            "sub x11, x11, #2",
            "4:",
            "cbz x11, 5f",
            "ldr x8, [x10]",
            "mov x13, x9",
            compact_xor_row_scalar!("d16"),
            compact_xor_row_scalar!("d17"),
            compact_xor_row_scalar!("d18"),
            compact_xor_row_scalar!("d19"),
            compact_xor_row_scalar!("d20"),
            compact_xor_row_scalar!("d21"),
            compact_xor_row_scalar!("d22"),
            compact_xor_row_scalar!("d23"),
            "5:",
            conditions = in(reg) conditions.as_ptr(),
            destinations = in(reg) destinations.as_mut_ptr().add(first),
            source = in(reg) source.as_ptr().add(first),
            remaining = in(reg) stride - first,
            stride_bytes = in(reg) stride * core::mem::size_of::<u64>(),
            out("x8") _,
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            options(nostack)
        );
    }
}

/// XOR eight masked source rows into one destination using only caller-saved registers.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn xor_eight_sources_if_compact(
    destination: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: [u64; 8],
) {
    use core::arch::asm;

    debug_assert_eq!(destination.len(), stride);
    debug_assert_eq!(sources.len(), 8 * stride);

    // SAFETY: `destination` and the eight complete source rows are disjoint. Adding `first`
    // leaves `stride - first` valid words in each. Main and tail loops process four, two and
    // one word, and every branch depends only on that public count. Conditions affect only
    // fixed mask-and-XOR instructions, using caller-saved registers throughout.
    unsafe {
        asm!(
            "ldr x8, [{conditions}, #0]",
            "neg x8, x8",
            "dup v16.2d, x8",
            "ldr x8, [{conditions}, #8]",
            "neg x8, x8",
            "dup v17.2d, x8",
            "ldr x8, [{conditions}, #16]",
            "neg x8, x8",
            "dup v18.2d, x8",
            "ldr x8, [{conditions}, #24]",
            "neg x8, x8",
            "dup v19.2d, x8",
            "ldr x8, [{conditions}, #32]",
            "neg x8, x8",
            "dup v20.2d, x8",
            "ldr x8, [{conditions}, #40]",
            "neg x8, x8",
            "dup v21.2d, x8",
            "ldr x8, [{conditions}, #48]",
            "neg x8, x8",
            "dup v22.2d, x8",
            "ldr x8, [{conditions}, #56]",
            "neg x8, x8",
            "dup v23.2d, x8",
            "mov x9, {destination}",
            "mov x10, {sources}",
            "mov x11, {remaining}",
            "mov x12, {stride_bytes}",
            "cmp x11, #4",
            "b.lo 3f",
            "2:",
            "ldp q0, q1, [x9]",
            "mov x13, x10",
            compact_xor_source_pair!("v16"),
            compact_xor_source_pair!("v17"),
            compact_xor_source_pair!("v18"),
            compact_xor_source_pair!("v19"),
            compact_xor_source_pair!("v20"),
            compact_xor_source_pair!("v21"),
            compact_xor_source_pair!("v22"),
            compact_xor_source_pair!("v23"),
            "stp q0, q1, [x9], #32",
            "add x10, x10, #32",
            "sub x11, x11, #4",
            "cmp x11, #4",
            "b.hs 2b",
            "3:",
            "cmp x11, #2",
            "b.lo 4f",
            "ldr q0, [x9]",
            "mov x13, x10",
            compact_xor_source_one!("v16"),
            compact_xor_source_one!("v17"),
            compact_xor_source_one!("v18"),
            compact_xor_source_one!("v19"),
            compact_xor_source_one!("v20"),
            compact_xor_source_one!("v21"),
            compact_xor_source_one!("v22"),
            compact_xor_source_one!("v23"),
            "str q0, [x9], #16",
            "add x10, x10, #16",
            "sub x11, x11, #2",
            "4:",
            "cbz x11, 5f",
            "ldr x8, [x9]",
            "mov x13, x10",
            compact_xor_source_scalar!("d16"),
            compact_xor_source_scalar!("d17"),
            compact_xor_source_scalar!("d18"),
            compact_xor_source_scalar!("d19"),
            compact_xor_source_scalar!("d20"),
            compact_xor_source_scalar!("d21"),
            compact_xor_source_scalar!("d22"),
            compact_xor_source_scalar!("d23"),
            "str x8, [x9]",
            "5:",
            conditions = in(reg) conditions.as_ptr(),
            destination = in(reg) destination.as_mut_ptr().add(first),
            sources = in(reg) sources.as_ptr().add(first),
            remaining = in(reg) stride - first,
            stride_bytes = in(reg) stride * core::mem::size_of::<u64>(),
            out("x8") _,
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            options(nostack)
        );
    }
}

/// XOR sixteen masked source rows into one destination using caller-saved registers.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn xor_sixteen_sources_if_compact(
    destination: &mut [u64],
    sources: &[u64],
    stride: usize,
    first: usize,
    conditions: [u64; 16],
) {
    use core::arch::asm;

    debug_assert_eq!(destination.len(), stride);
    debug_assert_eq!(sources.len(), 16 * stride);

    // SAFETY: `destination` and the sixteen complete source rows are disjoint. Adding `first`
    // leaves `stride - first` valid words in each. Main and tail loops process four, two and
    // one word, and every branch depends only on that public count. The sixteen conditions
    // occupy the caller-saved `v16..v31` registers and affect only fixed mask-and-XOR
    // instructions.
    unsafe {
        asm!(
            "ldr x8, [{conditions}, #0]",
            "neg x8, x8",
            "dup v16.2d, x8",
            "ldr x8, [{conditions}, #8]",
            "neg x8, x8",
            "dup v17.2d, x8",
            "ldr x8, [{conditions}, #16]",
            "neg x8, x8",
            "dup v18.2d, x8",
            "ldr x8, [{conditions}, #24]",
            "neg x8, x8",
            "dup v19.2d, x8",
            "ldr x8, [{conditions}, #32]",
            "neg x8, x8",
            "dup v20.2d, x8",
            "ldr x8, [{conditions}, #40]",
            "neg x8, x8",
            "dup v21.2d, x8",
            "ldr x8, [{conditions}, #48]",
            "neg x8, x8",
            "dup v22.2d, x8",
            "ldr x8, [{conditions}, #56]",
            "neg x8, x8",
            "dup v23.2d, x8",
            "ldr x8, [{conditions}, #64]",
            "neg x8, x8",
            "dup v24.2d, x8",
            "ldr x8, [{conditions}, #72]",
            "neg x8, x8",
            "dup v25.2d, x8",
            "ldr x8, [{conditions}, #80]",
            "neg x8, x8",
            "dup v26.2d, x8",
            "ldr x8, [{conditions}, #88]",
            "neg x8, x8",
            "dup v27.2d, x8",
            "ldr x8, [{conditions}, #96]",
            "neg x8, x8",
            "dup v28.2d, x8",
            "ldr x8, [{conditions}, #104]",
            "neg x8, x8",
            "dup v29.2d, x8",
            "ldr x8, [{conditions}, #112]",
            "neg x8, x8",
            "dup v30.2d, x8",
            "ldr x8, [{conditions}, #120]",
            "neg x8, x8",
            "dup v31.2d, x8",
            "mov x9, {destination}",
            "mov x10, {sources}",
            "mov x11, {remaining}",
            "mov x12, {stride_bytes}",
            "cmp x11, #4",
            "b.lo 3f",
            "2:",
            "ldp q0, q1, [x9]",
            "mov x13, x10",
            compact_xor_source_pair!("v16"),
            compact_xor_source_pair!("v17"),
            compact_xor_source_pair!("v18"),
            compact_xor_source_pair!("v19"),
            compact_xor_source_pair!("v20"),
            compact_xor_source_pair!("v21"),
            compact_xor_source_pair!("v22"),
            compact_xor_source_pair!("v23"),
            compact_xor_source_pair!("v24"),
            compact_xor_source_pair!("v25"),
            compact_xor_source_pair!("v26"),
            compact_xor_source_pair!("v27"),
            compact_xor_source_pair!("v28"),
            compact_xor_source_pair!("v29"),
            compact_xor_source_pair!("v30"),
            compact_xor_source_pair!("v31"),
            "stp q0, q1, [x9], #32",
            "add x10, x10, #32",
            "sub x11, x11, #4",
            "cmp x11, #4",
            "b.hs 2b",
            "3:",
            "cmp x11, #2",
            "b.lo 4f",
            "ldr q0, [x9]",
            "mov x13, x10",
            compact_xor_source_one!("v16"),
            compact_xor_source_one!("v17"),
            compact_xor_source_one!("v18"),
            compact_xor_source_one!("v19"),
            compact_xor_source_one!("v20"),
            compact_xor_source_one!("v21"),
            compact_xor_source_one!("v22"),
            compact_xor_source_one!("v23"),
            compact_xor_source_one!("v24"),
            compact_xor_source_one!("v25"),
            compact_xor_source_one!("v26"),
            compact_xor_source_one!("v27"),
            compact_xor_source_one!("v28"),
            compact_xor_source_one!("v29"),
            compact_xor_source_one!("v30"),
            compact_xor_source_one!("v31"),
            "str q0, [x9], #16",
            "add x10, x10, #16",
            "sub x11, x11, #2",
            "4:",
            "cbz x11, 5f",
            "ldr x8, [x9]",
            "mov x13, x10",
            compact_xor_source_scalar!("d16"),
            compact_xor_source_scalar!("d17"),
            compact_xor_source_scalar!("d18"),
            compact_xor_source_scalar!("d19"),
            compact_xor_source_scalar!("d20"),
            compact_xor_source_scalar!("d21"),
            compact_xor_source_scalar!("d22"),
            compact_xor_source_scalar!("d23"),
            compact_xor_source_scalar!("d24"),
            compact_xor_source_scalar!("d25"),
            compact_xor_source_scalar!("d26"),
            compact_xor_source_scalar!("d27"),
            compact_xor_source_scalar!("d28"),
            compact_xor_source_scalar!("d29"),
            compact_xor_source_scalar!("d30"),
            compact_xor_source_scalar!("d31"),
            "str x8, [x9]",
            "5:",
            conditions = in(reg) conditions.as_ptr(),
            destination = in(reg) destination.as_mut_ptr().add(first),
            sources = in(reg) sources.as_ptr().add(first),
            remaining = in(reg) stride - first,
            stride_bytes = in(reg) stride * core::mem::size_of::<u64>(),
            out("x8") _,
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            out("v24") _,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack)
        );
    }
}

#[inline]
fn accumulate_rows(
    target: &mut [u64],
    sources: &[u64],
    stride: usize,
    pivot_word: usize,
    pivot_shift: usize,
    first: usize,
) {
    let mut target_bit = (target[pivot_word] >> pivot_shift) & 1;

    let grouped = sources.len() / (8 * stride) * (8 * stride);
    let (groups, remainder) = sources.split_at(grouped);
    for group in groups.chunks_exact(8 * stride) {
        let source_bits = [
            (group[pivot_word] >> pivot_shift) & 1,
            (group[stride + pivot_word] >> pivot_shift) & 1,
            (group[2 * stride + pivot_word] >> pivot_shift) & 1,
            (group[3 * stride + pivot_word] >> pivot_shift) & 1,
            (group[4 * stride + pivot_word] >> pivot_shift) & 1,
            (group[5 * stride + pivot_word] >> pivot_shift) & 1,
            (group[6 * stride + pivot_word] >> pivot_shift) & 1,
            (group[7 * stride + pivot_word] >> pivot_shift) & 1,
        ];
        let mut conditions = [0u64; 8];
        for i in 0..8 {
            conditions[i] = target_bit ^ source_bits[i];
            target_bit |= source_bits[i];
        }
        xor_eight_sources(target, group, stride, first, &conditions);
    }
    for source in remainder.chunks_exact(stride) {
        let source_bit = (source[pivot_word] >> pivot_shift) & 1;
        xor_row_if(
            &mut target[first..],
            &source[first..],
            target_bit ^ source_bit,
        );
        target_bit |= source_bit;
    }
}

#[inline]
fn clear_rows(
    targets: &mut [u64],
    source: &[u64],
    stride: usize,
    pivot_word: usize,
    pivot_shift: usize,
    first: usize,
) {
    let grouped = targets.len() / (8 * stride) * (8 * stride);
    let (groups, remainder) = targets.split_at_mut(grouped);
    for group in groups.chunks_exact_mut(8 * stride) {
        let conditions = [
            (group[pivot_word] >> pivot_shift) & 1,
            (group[stride + pivot_word] >> pivot_shift) & 1,
            (group[2 * stride + pivot_word] >> pivot_shift) & 1,
            (group[3 * stride + pivot_word] >> pivot_shift) & 1,
            (group[4 * stride + pivot_word] >> pivot_shift) & 1,
            (group[5 * stride + pivot_word] >> pivot_shift) & 1,
            (group[6 * stride + pivot_word] >> pivot_shift) & 1,
            (group[7 * stride + pivot_word] >> pivot_shift) & 1,
        ];
        xor_eight_rows(group, source, stride, first, &conditions);
    }
    for target in remainder.chunks_exact_mut(stride) {
        let condition = (target[pivot_word] >> pivot_shift) & 1;
        xor_row_if(&mut target[first..], &source[first..], condition);
    }
}

/// What the narrow phase of a panel decides, to be replayed wide.
///
/// Each panel row's final content is
/// `origin(i) ^ sum_k folds[i][k] * origin(start + k) ^ sum_j c(i, j) * origin(j)`,
/// where the trailing coefficient `c(i, j)` is the parity of `carried[i] & direct[j]`. Factoring
/// the trailing term through `carried` is what lets one mask per panel row stand in for a
/// coefficient against every row below it.
struct PanelPlan<const COUNT: usize> {
    /// Per panel row, a mask over the original panel rows folded into it.
    folds: [u64; 32],
    /// Per panel row, a mask over the per-pivot trailing accumulations it carries.
    carried: [u64; 32],
}

/// Row-sized working buffers the panel elimination reuses across panels.
///
/// Allocated once per key generation rather than per panel: at `mt = 1664` these are a few tens of
/// kilobytes and stay resident in cache for the whole reduction.
pub(crate) struct PanelScratch {
    /// Each row's panel bits as they entered the current panel.
    pub(crate) origin: Vec<u64>,
    /// Per trailing row, which pivots' accumulations took it.
    pub(crate) direct: Vec<u64>,
    /// Copies of the panel rows as they entered, so a panel row can be its own source.
    pub(crate) saved: Vec<u64>,
}

impl PanelScratch {
    /// Size the buffers for a matrix of `rows` rows and `stride` words, with panels of `count`.
    pub(crate) fn new(rows: usize, stride: usize, count: usize) -> Self {
        Self {
            origin: vec![0; rows],
            direct: vec![0; rows],
            saved: vec![0; count * stride],
        }
    }
}

impl Drop for PanelScratch {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.origin.zeroize();
        self.direct.zeroize();
        self.saved.zeroize();
    }
}

impl BitMatrix {
    /// The words per row, including the trailing guard word.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Allocate a zeroed matrix with `rows` rows and at least `columns` columns.
    pub(crate) fn zeros(rows: usize, columns: usize) -> Self {
        let stride = columns.div_ceil(64) + 1;
        Self {
            rows,
            stride,
            data: vec![0u64; rows * stride],
        }
    }

    /// Store `value` as byte `index` of `row`.
    ///
    /// The library builds rows from bit-sliced planes; only the tests still assemble
    /// matrices byte by byte.
    #[cfg(test)]
    #[inline]
    pub(crate) fn set_byte(&mut self, row: usize, index: usize, value: u8) {
        let word = row * self.stride + index / 8;
        let shift = (index % 8) * 8;
        self.data[word] = (self.data[word] & !(0xFFu64 << shift)) | ((value as u64) << shift);
    }

    /// The words of row `row`, for callers that assemble whole rows at once.
    #[inline]
    pub(crate) fn row_words_mut(&mut self, row: usize) -> &mut [u64] {
        &mut self.data[row * self.stride..(row + 1) * self.stride]
    }

    /// Read bit `bit` of `row` as 0 or 1.
    #[inline]
    pub(crate) fn bit(&self, row: usize, bit: usize) -> u64 {
        (self.data[row * self.stride + bit / 64] >> (bit % 64)) & 1
    }

    /// Read 64 bits of `row` starting at bit offset `bit`.
    #[inline]
    pub(crate) fn read_window(&self, row: usize, bit: usize) -> u64 {
        let base = row * self.stride + bit / 64;
        let shift = bit % 64;
        if shift == 0 {
            self.data[base]
        } else {
            (self.data[base] >> shift) | (self.data[base + 1] << (64 - shift))
        }
    }

    /// Overwrite the 64 bits of `row` starting at bit offset `bit`, leaving neighbors intact.
    #[inline]
    pub(crate) fn write_window(&mut self, row: usize, bit: usize, value: u64) {
        let base = row * self.stride + bit / 64;
        let shift = bit % 64;
        if shift == 0 {
            self.data[base] = value;
        } else {
            let low_mask = (1u64 << shift) - 1;
            self.data[base] = (self.data[base] & low_mask) | (value << shift);
            // The window covers only the low `shift` bits of the next word, so everything
            // above them must survive.
            let high_mask = !((1u64 << shift) - 1);
            self.data[base + 1] = (self.data[base + 1] & high_mask) | (value >> (64 - shift));
        }
    }

    /// Add `source` into `target` from word `first` onwards, gated by `mask`.
    ///
    /// This is the inner loop of Gaussian elimination and the reason rows are word aligned.
    /// `first` lets the caller skip the leading words that elimination has already reduced to
    /// zero in both rows; it depends only on how far the elimination has progressed, never on
    /// the contents of the matrix, so skipping them is not a data-dependent shortcut.
    #[cfg(test)]
    #[inline]
    pub(crate) fn add_row(&mut self, target: usize, source: usize, mask: u64, first: usize) {
        debug_assert_ne!(target, source);
        let stride = self.stride;
        let (target_words, source_words) = if target < source {
            let (head, tail) = self.data.split_at_mut(source * stride);
            (
                &mut head[target * stride + first..target * stride + stride],
                &tail[first..stride],
            )
        } else {
            let (head, tail) = self.data.split_at_mut(target * stride);
            (
                &mut tail[first..stride],
                &head[source * stride + first..source * stride + stride],
            )
        };
        for (dest, &src) in target_words.iter_mut().zip(source_words.iter()) {
            *dest ^= src & mask;
        }
    }

    /// Fold every row below `row` into it until its `pivot` bit is one, using masked additions.
    ///
    /// The loop bounds and memory locations depend only on the public matrix dimensions and
    /// elimination position. Keeping the target row borrowed once avoids splitting and
    /// rechecking the matrix allocation for every candidate pivot row.
    #[inline]
    pub(crate) fn accumulate_pivot(&mut self, row: usize, pivot: usize, first: usize) {
        let stride = self.stride;
        let row_start = row * stride;
        let (_, target_and_after) = self.data.split_at_mut(row_start);
        let (target, after) = target_and_after.split_at_mut(stride);
        let pivot_word = pivot / 64;
        let pivot_shift = pivot % 64;

        accumulate_rows(target, after, stride, pivot_word, pivot_shift, first);
    }

    /// Clear `pivot` from every row below `row`, using `row` as the source.
    ///
    /// Forward elimination only needs the rows below the pivot in echelon form. Rows above it
    /// are reduced later in blocks, which lets eight pivot rows share one destination pass.
    #[inline]
    pub(crate) fn clear_below(&mut self, row: usize, pivot: usize, first: usize) {
        let stride = self.stride;
        let row_start = row * stride;
        let (_, source_and_after) = self.data.split_at_mut(row_start);
        let (source, after) = source_and_after.split_at_mut(stride);
        let pivot_word = pivot / 64;
        let pivot_shift = pivot % 64;

        clear_rows(after, source, stride, pivot_word, pivot_shift, first);
    }

    /// Back-substitute a block of at most sixteen consecutive pivot rows.
    ///
    /// The pivot rows first reduce each other to an identity block. On AArch64, a full block
    /// then clears all sixteen pivot columns from each earlier row in one streaming pass. The
    /// sequence of rows, loads and stores depends only on the public matrix dimensions.
    #[inline]
    pub(crate) fn clear_above_block<const COUNT: usize>(&mut self, start: usize) {
        let count = COUNT;
        debug_assert!(count > 0 && count <= 16);
        debug_assert!(start + count <= self.rows);

        let stride = self.stride;
        let first = start / 64;

        // Make the pivot rows an identity block. Forward elimination already cleared every
        // earlier pivot from each later row, so only entries above the diagonal remain.
        for source_row in (start..start + count).rev() {
            let source_start = source_row * stride;
            let (before_source, source_and_after) = self.data.split_at_mut(source_start);
            let source = &source_and_after[..stride];
            let targets = &mut before_source[start * stride..source_start];
            clear_rows(
                targets,
                source,
                stride,
                source_row / 64,
                source_row % 64,
                first,
            );
        }

        let block_start = start * stride;
        let (before, block_and_after) = self.data.split_at_mut(block_start);
        let block = &block_and_after[..count * stride];

        if count == 16 {
            let mut pairs = before.chunks_exact_mut(2 * stride);
            for pair in pairs.by_ref() {
                let mut conditions = [[0u64; 16]; 2];
                for (side, side_conditions) in conditions.iter_mut().enumerate() {
                    let row = &pair[side * stride..(side + 1) * stride];
                    for (offset, condition) in side_conditions.iter_mut().enumerate() {
                        let pivot = start + offset;
                        *condition = (row[pivot / 64] >> (pivot % 64)) & 1;
                    }
                }
                xor_sixteen_sources_pair(pair, block, stride, first, &conditions);
            }
            for target in pairs.into_remainder().chunks_exact_mut(stride) {
                let mut conditions = [0u64; 16];
                for (offset, condition) in conditions.iter_mut().enumerate() {
                    let pivot = start + offset;
                    *condition = (target[pivot / 64] >> (pivot % 64)) & 1;
                }
                xor_sixteen_sources(target, block, stride, first, &conditions);
            }
            return;
        }

        if count == 8 {
            for target in before.chunks_exact_mut(stride) {
                let conditions = [
                    (target[start / 64] >> (start % 64)) & 1,
                    (target[(start + 1) / 64] >> ((start + 1) % 64)) & 1,
                    (target[(start + 2) / 64] >> ((start + 2) % 64)) & 1,
                    (target[(start + 3) / 64] >> ((start + 3) % 64)) & 1,
                    (target[(start + 4) / 64] >> ((start + 4) % 64)) & 1,
                    (target[(start + 5) / 64] >> ((start + 5) % 64)) & 1,
                    (target[(start + 6) / 64] >> ((start + 6) % 64)) & 1,
                    (target[(start + 7) / 64] >> ((start + 7) % 64)) & 1,
                ];
                xor_eight_sources(target, block, stride, first, &conditions);
            }
            return;
        }

        // The one short block possible when the row count is not a multiple of sixteen. The
        // fixed source order is still data oblivious.
        for (offset, source) in block.chunks_exact(stride).enumerate().rev() {
            clear_rows(
                before,
                source,
                stride,
                (start + offset) / 64,
                (start + offset) % 64,
                first,
            );
        }
    }

    /// Clear `pivot` from every row except `row`, using `row` as the source.
    ///
    /// Borrowing the source once lets the two contiguous target ranges stream through memory
    /// without re-splitting the allocation for every row.
    #[cfg(test)]
    #[inline]
    pub(crate) fn clear_column(&mut self, row: usize, pivot: usize, first: usize) {
        let stride = self.stride;
        let row_start = row * stride;
        let (before, source_and_after) = self.data.split_at_mut(row_start);
        let (source, after) = source_and_after.split_at_mut(stride);
        let pivot_word = pivot / 64;
        let pivot_shift = pivot % 64;

        clear_rows(before, source, stride, pivot_word, pivot_shift, first);
        clear_rows(after, source, stride, pivot_word, pivot_shift, first);
    }

    /// Extract `count` bits of `row` starting at `start`, packed little-endian into `out`.
    ///
    /// Any bits past `count` in the final byte are cleared, which is what makes the padding of
    /// `mceliece6960119` public keys well defined.
    pub(crate) fn extract_bits(&self, row: usize, start: usize, count: usize, out: &mut [u8]) {
        debug_assert_eq!(out.len(), count.div_ceil(8));
        let full_words = count / 64;
        for w in 0..full_words {
            let value = self.read_window(row, start + w * 64);
            out[w * 8..w * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }

        let remaining = count % 64;
        if remaining != 0 {
            let value = self.read_window(row, start + full_words * 64) & ((1u64 << remaining) - 1);
            let bytes = value.to_le_bytes();
            let tail = &mut out[full_words * 8..];
            tail.copy_from_slice(&bytes[..tail.len()]);
        }
    }

    /// The number of rows.
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Decide a whole panel's elimination on one `u64` per row, recording what to replay wide.
    ///
    /// This is the half that makes the panel worth having. The previous shape interleaved wide row
    /// operations with the pivot decisions and so streamed every row below the panel once per
    /// pivot, thirty-two times per panel; deciding narrowly first lets the replay stream them once.
    ///
    /// `origin` keeps each row's panel bits as they entered, because a trailing row's matrix copy
    /// is untouched until the replay and the pivot rows accumulate *those* values rather than the
    /// reduced ones. `shadow` is the reduced view the decisions read.
    ///
    /// Returns `None` for the `⊥` outcome, when some pivot column has no one below it.
    fn plan_panel<const COUNT: usize>(
        &mut self,
        start: usize,
        shadow: &mut [u64],
        scratch: &mut PanelScratch,
    ) -> Option<PanelPlan<COUNT>> {
        let stride = self.stride;
        let word = start / 64;
        let shift = start % 64;
        let panel_mask = if COUNT >= 64 {
            u64::MAX
        } else {
            (1u64 << COUNT) - 1
        };
        let mut plan = PanelPlan::<COUNT> {
            folds: [0u64; 32],
            carried: [0u64; 32],
        };

        // Every decision below is taken on one `u64` per row. The wide row operations are not
        // interleaved with them any more: they are replayed afterwards, once, from the masks this
        // loop records. That is the whole point -- the previous shape streamed every row below the
        // panel once per pivot, thirty-two times per panel, and this streams them once.
        //
        // `origin` keeps each row's panel bits as they entered the panel, because the matrix copy
        // of a trailing row is untouched until the replay and the pivot rows accumulate *those*
        // values, not the reduced ones. `shadow` is the reduced view the pivot decisions read.
        for (row, slot) in shadow.iter_mut().enumerate().skip(start) {
            *slot = (self.data[row * stride + word] >> shift) & panel_mask;
        }
        scratch.origin[start..self.rows].copy_from_slice(&shadow[start..self.rows]);
        scratch.direct[start + COUNT..self.rows].fill(0);

        // How each panel row is built out of the rows that entered the panel:
        //   final(i) = origin(i) ^ sum_k plan.folds[i][k] * origin(start + k)
        //                        ^ sum_j (parity of direct[i] & direct_of(j)) * origin(j)
        // `plan.folds[i]` is a mask over the original panel rows; `plan.carried[i]` is a mask over the
        // per-pivot trailing accumulations, which is what lets the trailing contribution be
        // factored instead of tracked per row.
        //
        // Folding row `k` into row `i` transfers row `k`'s whole content, which is its own
        // original *plus* whatever it has absorbed -- hence `plan.folds[k] ^ (1 << k)` rather than
        // `plan.folds[k]`. Dropping that bit silently loses one original row per fold.
        let mut raw = [0u64; 32];
        raw[..COUNT].copy_from_slice(&shadow[start..start + COUNT]);

        for i in 0..COUNT {
            let pivot_row = start + i;

            // Clear the earlier panel columns from this row. Sequential: each step changes the
            // bit the next one reads.
            for k in 0..i {
                let mask = 0u64.wrapping_sub((raw[i] >> k) & 1);
                raw[i] ^= raw[k] & mask;
                plan.folds[i] ^= (plan.folds[k] ^ (1 << k)) & mask;
                plan.carried[i] ^= plan.carried[k] & mask;
            }

            // Pull a pivot up from the rows below. Panel rows come first, in row order, then the
            // trailing rows -- the same order the row-at-a-time schedule visits them in.
            let mut target_bit = (shadow[pivot_row] >> i) & 1;
            for k in i + 1..COUNT {
                let source_bit = (shadow[start + k] >> i) & 1;
                let mask = 0u64.wrapping_sub(target_bit ^ source_bit);
                raw[i] ^= raw[k] & mask;
                plan.folds[i] ^= (plan.folds[k] ^ (1 << k)) & mask;
                plan.carried[i] ^= plan.carried[k] & mask;
                shadow[pivot_row] ^= shadow[start + k] & mask;
                target_bit |= source_bit;
            }
            for j in start + COUNT..self.rows {
                let source_bit = (shadow[j] >> i) & 1;
                let condition = target_bit ^ source_bit;
                let mask = 0u64.wrapping_sub(condition);
                raw[i] ^= scratch.origin[j] & mask;
                scratch.direct[j] |= condition << i;
                shadow[pivot_row] ^= shadow[j] & mask;
                target_bit |= source_bit;
            }
            // This row now carries its own trailing accumulation.
            plan.carried[i] ^= 1 << i;

            if target_bit == 0 {
                // No pivot exists in this column: the matrix has no systematic form.
                return None;
            }

            // Cancel the earlier panel columns the unreduced sources reintroduced.
            for k in 0..i {
                let mask = 0u64.wrapping_sub((raw[i] >> k) & 1);
                raw[i] ^= raw[k] & mask;
                plan.folds[i] ^= (plan.folds[k] ^ (1 << k)) & mask;
                plan.carried[i] ^= plan.carried[k] & mask;
            }

            // Keep the established pivots an identity block, so the trailing pass can read its
            // conditions straight out of the matrix.
            for k in 0..i {
                let mask = 0u64.wrapping_sub((raw[k] >> i) & 1);
                raw[k] ^= raw[i] & mask;
                plan.folds[k] ^= (plan.folds[i] ^ (1 << i)) & mask;
                plan.carried[k] ^= plan.carried[i] & mask;
                shadow[start + k] ^= shadow[pivot_row] & mask;
            }

            // Narrow-clear this pivot from every row below, so the next pivot decides on reduced
            // bits. One word per row.
            let pivot_shadow = shadow[pivot_row];
            for slot in shadow.iter_mut().skip(pivot_row + 1) {
                let mask = 0u64.wrapping_sub((*slot >> i) & 1);
                *slot ^= pivot_shadow & mask;
            }
        }

        Some(plan)
    }

    /// Forward-eliminate `COUNT` consecutive pivots starting at `start`, in one blocked panel.
    ///
    /// # Why a panel at all
    ///
    /// Row-at-a-time forward elimination streams the whole matrix past the caches once per pivot,
    /// which for the larger parameter sets is where nearly all of key generation's time goes. The
    /// back-substitution already avoids this by applying sixteen pivots per pass; this brings the
    /// forward sweep to the same shape. The trailing rows are touched once per *panel* instead of
    /// once per *pivot*.
    ///
    /// # Why a shadow instead of a table
    ///
    /// The classic Method of Four Russians would precompute every combination of the panel's
    /// pivot rows and index it by the trailing row's bits. Those bits are secret, so the lookup
    /// would be a secret-dependent memory access — a cache-timing side channel. Instead this
    /// keeps a `shadow` word per row holding only that row's `COUNT` panel-column bits, reduced
    /// by the pivots established so far, and drives every decision from it with masked XORs. The
    /// shadow is one `u64` per row (13 KiB at `mt = 1664`), so it stays in L1 and the narrow
    /// bookkeeping is free next to the wide row operations.
    ///
    /// # Why the result is unchanged
    ///
    /// Reduction is linear, so accumulating an unreduced source row and reducing afterwards
    /// gives the same vector as reducing first and accumulating after. Every pivot decision is
    /// taken on the reduced bits held in the shadow, so the sequence of conditions matches the
    /// row-at-a-time schedule exactly.
    ///
    /// The rows *below* the pivots come out bit-identical to that schedule, which is what the
    /// semi-systematic parameter sets depend on: a trailing row ends with zeros in every pivot
    /// column, and its coset holds exactly one such vector. The pivot rows themselves do not —
    /// this leaves them an identity block where the row-at-a-time schedule leaves echelon form —
    /// but back-substitution takes both to the same unique reduced row echelon form, so the
    /// public key is unchanged either way.
    ///
    /// `shadow` must have one entry per row and is (re)initialized here. `start` must be a
    /// multiple of 64 or otherwise leave `COUNT` bits inside a single word. Returns `false` when
    /// some pivot column has no one below it, which is the `⊥` outcome key generation retries on.
    pub(crate) fn eliminate_panel<const COUNT: usize>(
        &mut self,
        start: usize,
        shadow: &mut [u64],
        scratch: &mut PanelScratch,
    ) -> bool {
        debug_assert!(COUNT > 0 && COUNT <= 32);
        debug_assert!(start + COUNT <= self.rows);
        debug_assert_eq!(shadow.len(), self.rows);
        // The panel's columns must not straddle a word boundary, or one shadow word could not
        // hold them.
        debug_assert!(start % 64 <= 64 - COUNT);

        let stride = self.stride;
        let first = start / 64;
        let word = start / 64;
        let shift = start % 64;
        let panel_mask = if COUNT >= 64 {
            u64::MAX
        } else {
            (1u64 << COUNT) - 1
        };

        let Some(plan) = self.plan_panel::<COUNT>(start, shadow, scratch) else {
            // No pivot exists in some column: the matrix has no systematic form.
            return false;
        };
        let (folds, carried) = (plan.folds, plan.carried);

        // Replay, wide. Both correction terms accumulate onto the panel rows as they entered, so
        // the originals have to be copied aside *before* either term is applied -- a panel row is
        // a source for the panel term and a destination for both.
        let block_start = start * stride;
        scratch.saved[..COUNT * stride]
            .copy_from_slice(&self.data[block_start..block_start + COUNT * stride]);

        // First the trailing rows, once. Row `j` belongs in panel row `i` when an odd number of
        // the accumulations `i` carries actually took `j`, which is the parity of one `AND`.
        // Every trailing row is applied to every panel row unconditionally, masked. An earlier
        // version skipped rows whose conditions were all zero, which is a branch on secret
        // matrix bits and therefore a timing oracle for how the pivots fell. The skip is not
        // available here at any price.
        {
            let trailing_start = (start + COUNT) * stride;
            let (before, rest) = self.data.split_at_mut(trailing_start);
            let block = &mut before[block_start..block_start + COUNT * stride];

            if cfg!(target_arch = "x86_64") && COUNT.is_multiple_of(8) {
                // Eight panel rows at a time ride in registers while the whole trailing
                // stream folds past them; the parity conditions are recomputed from
                // `carried` inside the kernel.
                #[cfg(target_arch = "x86_64")]
                {
                    let trailing_rows = rest.len() / stride;
                    let taken = &scratch.direct[start + COUNT..start + COUNT + trailing_rows];
                    for (group, rows) in block.chunks_exact_mut(8 * stride).enumerate() {
                        let mut group_carried = [0u64; 8];
                        group_carried.copy_from_slice(&carried[8 * group..8 * group + 8]);
                        super::simd::matrix_x86::fold_trailing_if(
                            rows,
                            rest,
                            stride,
                            first,
                            &group_carried,
                            taken,
                        );
                    }
                }
            } else {
                for (offset, source) in rest.chunks_exact(stride).enumerate() {
                    let taken = scratch.direct[start + COUNT + offset];
                    let mut conditions = [0u64; 32];
                    for (i, condition) in conditions.iter_mut().enumerate().take(COUNT) {
                        *condition = u64::from((carried[i] & taken).count_ones() & 1);
                    }
                    apply_to_panel::<COUNT>(block, source, stride, first, &conditions);
                }
            }
        }

        // Then the panel rows against copies of what they entered with, because here a panel row
        // is both a source and a destination.
        {
            for (i, &row_folds) in folds.iter().enumerate().take(COUNT) {
                let (target_start, target_end) =
                    (block_start + i * stride, block_start + (i + 1) * stride);
                for k in 0..COUNT {
                    let condition = (row_folds >> k) & 1;
                    let target = &mut self.data[target_start..target_end];
                    let source = &scratch.saved[k * stride..(k + 1) * stride];
                    xor_row_if(&mut target[first..], &source[first..], condition);
                }
            }
        }

        // One streaming pass over the trailing rows, applying all `COUNT` pivots at once. The
        // panel rows are an identity block in these columns, so each trailing row's own bits are
        // the conditions, and XORing them clears the panel columns.
        let block_start = start * stride;
        let trailing_start = (start + COUNT) * stride;
        let (before, rest) = self.data.split_at_mut(trailing_start);
        let block = &before[block_start..block_start + COUNT * stride];

        let mut pairs = rest.chunks_exact_mut(2 * stride);
        for pair in pairs.by_ref() {
            let mut conditions = [[0u64; 16]; 2];
            let mut conditions_high = [[0u64; 16]; 2];
            for ((low, high), row) in conditions
                .iter_mut()
                .zip(conditions_high.iter_mut())
                .zip(pair.chunks_exact(stride))
            {
                let bits = (row[word] >> shift) & panel_mask;
                for (offset, condition) in low.iter_mut().enumerate() {
                    *condition = (bits >> offset) & 1;
                }
                for (offset, condition) in high.iter_mut().enumerate() {
                    *condition = (bits >> (offset + 16)) & 1;
                }
            }
            apply_panel_pair::<COUNT>(pair, block, stride, first, &conditions, &conditions_high);
        }
        for target in pairs.into_remainder().chunks_exact_mut(stride) {
            let bits = (target[word] >> shift) & panel_mask;
            let mut conditions = [0u64; 16];
            let mut conditions_high = [0u64; 16];
            for (offset, condition) in conditions.iter_mut().enumerate() {
                *condition = (bits >> offset) & 1;
            }
            for (offset, condition) in conditions_high.iter_mut().enumerate() {
                *condition = (bits >> (offset + 16)) & 1;
            }
            apply_panel::<COUNT>(target, block, stride, first, &conditions, &conditions_high);
        }

        true
    }
}

/// XOR one trailing row into the `COUNT` panel rows, each gated by its own condition.
///
/// The mirror of [`apply_panel`]: there the panel is the source, here it is the destination. The
/// panel rows are contiguous and disjoint from the source, which is what the blocked kernel needs.
#[inline]
fn apply_to_panel<const COUNT: usize>(
    destinations: &mut [u64],
    source: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; 32],
) {
    debug_assert_eq!(destinations.len(), COUNT * stride);

    let mut done = 0;
    while done + 8 <= COUNT {
        let mut group = [0u64; 8];
        group.copy_from_slice(&conditions[done..done + 8]);
        let rows = &mut destinations[done * stride..(done + 8) * stride];
        xor_eight_rows(rows, source, stride, first, &group);
        done += 8;
    }
    for (offset, target) in destinations[done * stride..]
        .chunks_exact_mut(stride)
        .enumerate()
    {
        xor_row_if(
            &mut target[first..],
            &source[first..],
            conditions[done + offset],
        );
    }
}

/// XOR the `COUNT` masked panel rows into one trailing row.
///
/// Dispatches to the sixteen-wide kernel when the panel is full and the target has one, since
/// that is the shape the standardized parameter sets always take.
#[inline]
fn apply_panel<const COUNT: usize>(
    target: &mut [u64],
    block: &[u64],
    stride: usize,
    first: usize,
    conditions: &[u64; 16],
    conditions_high: &[u64; 16],
) {
    if COUNT == 16 {
        xor_sixteen_sources(target, block, stride, first, conditions);
        return;
    }
    if COUNT == 32 {
        let (low, high) = block.split_at(16 * stride);
        let mut upper = [0u64; 16];
        upper.copy_from_slice(&conditions_high[..16]);
        xor_sixteen_sources(target, low, stride, first, conditions);
        xor_sixteen_sources(target, high, stride, first, &upper);
        return;
    }

    for (offset, source) in block.chunks_exact(stride).enumerate().take(COUNT) {
        let condition = if offset < 16 {
            conditions[offset]
        } else {
            conditions_high[offset - 16]
        };
        xor_row_if(&mut target[first..], &source[first..], condition);
    }
}

/// The pair form of [`apply_panel`]: two consecutive trailing rows advance through one pass
/// of the panel's source chunks, halving the source traffic that bounds the fold.
#[inline]
fn apply_panel_pair<const COUNT: usize>(
    targets: &mut [u64],
    block: &[u64],
    stride: usize,
    first: usize,
    conditions: &[[u64; 16]; 2],
    conditions_high: &[[u64; 16]; 2],
) {
    if COUNT == 16 {
        xor_sixteen_sources_pair(targets, block, stride, first, conditions);
        return;
    }
    if COUNT == 32 {
        let (low, high) = block.split_at(16 * stride);
        xor_sixteen_sources_pair(targets, low, stride, first, conditions);
        xor_sixteen_sources_pair(targets, high, stride, first, conditions_high);
        return;
    }

    let (first_row, second_row) = targets.split_at_mut(stride);
    apply_panel::<COUNT>(
        first_row,
        block,
        stride,
        first,
        &conditions[0],
        &conditions_high[0],
    );
    apply_panel::<COUNT>(
        second_row,
        block,
        stride,
        first,
        &conditions[1],
        &conditions_high[1],
    );
}

impl Drop for BitMatrix {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.data.zeroize();
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

    /// Fill a matrix row from a byte vector and return the bytes for comparison.
    fn fill_row(m: &mut BitMatrix, row: usize, bytes: &[u8]) {
        for (i, &b) in bytes.iter().enumerate() {
            m.set_byte(row, i, b);
        }
    }

    #[test]
    fn bytes_round_trip_through_bit_and_window_accessors() {
        let mut rng = Rng(0x51ED_2701_ABCD_9876);
        let columns = 6960;
        let bytes: Vec<u8> = (0..columns / 8).map(|_| rng.next() as u8).collect();

        let mut m = BitMatrix::zeros(2, columns);
        fill_row(&mut m, 0, &bytes);

        for bit in 0..columns {
            let expected = ((bytes[bit / 8] >> (bit % 8)) & 1) as u64;
            assert_eq!(m.bit(0, bit), expected, "bit {bit}");
        }

        // Unaligned windows must agree with the bits they cover.
        for start in [0usize, 1, 7, 63, 64, 1515, 1547, 4096, columns - 64] {
            let window = m.read_window(0, start);
            for k in 0..64 {
                assert_eq!(
                    (window >> k) & 1,
                    m.bit(0, start + k),
                    "window {start} offset {k}"
                );
            }
        }
    }

    #[test]
    fn write_window_replaces_exactly_sixty_four_bits() {
        let mut rng = Rng(0x0192_8374_6555_1122);
        let columns = 4096;
        let original: Vec<u8> = (0..columns / 8).map(|_| rng.next() as u8).collect();

        for start in [0usize, 3, 63, 64, 100, 1515, columns - 64] {
            let mut m = BitMatrix::zeros(1, columns);
            fill_row(&mut m, 0, &original);

            let value = rng.next();
            m.write_window(0, start, value);

            for bit in 0..columns {
                let expected = if bit >= start && bit < start + 64 {
                    (value >> (bit - start)) & 1
                } else {
                    ((original[bit / 8] >> (bit % 8)) & 1) as u64
                };
                assert_eq!(m.bit(0, bit), expected, "start {start} bit {bit}");
            }
        }
    }

    #[test]
    fn add_row_is_masked_and_works_in_both_directions() {
        let mut rng = Rng(0xABCD_0123_4567_89EF);
        let columns = 1024;
        let a: Vec<u8> = (0..columns / 8).map(|_| rng.next() as u8).collect();
        let b: Vec<u8> = (0..columns / 8).map(|_| rng.next() as u8).collect();

        for (target, source) in [(0usize, 1usize), (1, 0)] {
            let mut m = BitMatrix::zeros(2, columns);
            fill_row(&mut m, 0, &a);
            fill_row(&mut m, 1, &b);

            // A zero mask must leave the matrix untouched.
            let before: Vec<u64> = (0..columns).map(|b| m.bit(target, b)).collect();
            m.add_row(target, source, 0, 0);
            let after: Vec<u64> = (0..columns).map(|b| m.bit(target, b)).collect();
            assert_eq!(before, after);

            m.add_row(target, source, u64::MAX, 0);
            let rows = [&a, &b];
            for bit in 0..columns {
                let t = (rows[target][bit / 8] >> (bit % 8)) & 1;
                let s = (rows[source][bit / 8] >> (bit % 8)) & 1;
                assert_eq!(m.bit(target, bit), (t ^ s) as u64, "bit {bit}");
            }
        }
    }

    #[test]
    fn pivot_sweeps_match_individual_row_additions() {
        let mut rng = Rng(0xF00D_CAFE_1357_2468);
        let rows = 17;
        let columns = 320;
        let bytes: Vec<Vec<u8>> = (0..rows)
            .map(|_| (0..columns / 8).map(|_| rng.next() as u8).collect())
            .collect();

        for row in [0usize, 5, rows - 1] {
            let pivot = 67 + row;
            let first = pivot / 64;
            let mut expected = BitMatrix::zeros(rows, columns);
            let mut actual = BitMatrix::zeros(rows, columns);
            for (i, value) in bytes.iter().enumerate() {
                fill_row(&mut expected, i, value);
                fill_row(&mut actual, i, value);
            }

            for source in row + 1..rows {
                let mask =
                    0u64.wrapping_sub(expected.bit(row, pivot) ^ expected.bit(source, pivot));
                expected.add_row(row, source, mask, first);
            }
            actual.accumulate_pivot(row, pivot, first);

            for target in 0..rows {
                for bit in 0..columns {
                    assert_eq!(
                        actual.bit(target, bit),
                        expected.bit(target, bit),
                        "accumulate row {row} target {target} bit {bit}"
                    );
                }
            }

            for target in 0..rows {
                if target != row {
                    let mask = 0u64.wrapping_sub(expected.bit(target, pivot));
                    expected.add_row(target, row, mask, first);
                }
            }
            actual.clear_column(row, pivot, first);

            for target in 0..rows {
                for bit in 0..columns {
                    assert_eq!(
                        actual.bit(target, bit),
                        expected.bit(target, bit),
                        "clear row {row} target {target} bit {bit}"
                    );
                }
            }
        }
    }

    /// `clear_below` must agree with clearing the pivot one row at a time.
    ///
    /// The existing sweep test covers `accumulate_pivot` and `clear_column`; `clear_below` is the
    /// third of the trio and the one forward elimination actually calls.
    #[test]
    fn clear_below_matches_individual_row_additions() {
        let mut rng = Rng(0x51ED_2701_ABCD_9876);
        let rows = 17;
        let columns = 320;
        let bytes: Vec<Vec<u8>> = (0..rows)
            .map(|_| (0..columns / 8).map(|_| rng.next() as u8).collect())
            .collect();

        for row in [0usize, 5, rows - 2] {
            for &pivot in &[row, 8, 63, 64, 67 + row] {
                let first = pivot / 64;
                let mut expected = BitMatrix::zeros(rows, columns);
                let mut actual = BitMatrix::zeros(rows, columns);
                for (i, value) in bytes.iter().enumerate() {
                    fill_row(&mut expected, i, value);
                    fill_row(&mut actual, i, value);
                }

                for target in row + 1..rows {
                    let mask = 0u64.wrapping_sub(expected.bit(target, pivot));
                    expected.add_row(target, row, mask, first);
                }
                actual.clear_below(row, pivot, first);

                for target in 0..rows {
                    for bit in 0..columns {
                        assert_eq!(
                            actual.bit(target, bit),
                            expected.bit(target, bit),
                            "row {row} pivot {pivot} target {target} bit {bit}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn extract_bits_packs_and_clears_padding() {
        let mut rng = Rng(0x7777_8888_9999_AAAA);
        let columns = 6960;
        let bytes: Vec<u8> = (0..columns / 8).map(|_| rng.next() as u8).collect();
        let mut m = BitMatrix::zeros(1, columns);
        fill_row(&mut m, 0, &bytes);

        // 5413 bits starting at bit 1547 is exactly the mceliece6960119 public key row.
        let (start, count) = (1547usize, 5413usize);
        let mut out = vec![0xFFu8; count.div_ceil(8)];
        m.extract_bits(0, start, count, &mut out);

        for i in 0..count {
            let expected = ((out[i / 8] >> (i % 8)) & 1) as u64;
            assert_eq!(m.bit(0, start + i), expected, "bit {i}");
        }
        // Bits above `count` in the last byte must be zero, not leftover 0xFF.
        assert_eq!(out[count / 8] >> (count % 8), 0);
    }

    #[test]
    fn extract_bits_handles_whole_word_counts() {
        let mut rng = Rng(0x1010_2020_3030_4040);
        let columns = 512;
        let bytes: Vec<u8> = (0..columns / 8).map(|_| rng.next() as u8).collect();
        let mut m = BitMatrix::zeros(1, columns);
        fill_row(&mut m, 0, &bytes);

        let mut out = vec![0u8; 16];
        m.extract_bits(0, 128, 128, &mut out);
        assert_eq!(out.as_slice(), &bytes[16..32]);
    }

    #[test]
    fn rows_reports_the_configured_height() {
        let m = BitMatrix::zeros(37, 100);
        assert_eq!(m.rows(), 37);
    }

    /// A splitmix64 generator, for the tests that care about linear independence.
    ///
    /// The xorshift generator the other tests use is `GF(2)`-linear: every bit it produces is a
    /// linear function of the same 64 seed bits, so a matrix filled from it has built-in column
    /// dependencies and no elimination can find pivots past them. That is invisible to a test
    /// that only compares two implementations of the same operation, and fatal to one that needs
    /// a full-rank matrix. The multiplications here are what make this one nonlinear over
    /// `GF(2)`.
    struct SplitMix(u64);

    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Fill a matrix with pseudo-random bits and return an independent copy of the same content.
    fn random_pair(rows: usize, columns: usize, seed: u64) -> (BitMatrix, BitMatrix) {
        let mut rng = SplitMix(seed);
        let mut a = BitMatrix::zeros(rows, columns);
        let mut b = BitMatrix::zeros(rows, columns);
        for row in 0..rows {
            for byte in 0..columns / 8 {
                let value = rng.next() as u8;
                a.set_byte(row, byte, value);
                b.set_byte(row, byte, value);
            }
        }
        (a, b)
    }

    /// Forward-eliminate `count` pivots the original way, one row at a time.
    fn eliminate_row_at_a_time(mat: &mut BitMatrix, count: usize) -> bool {
        for row in 0..count {
            let first = row / 64;
            mat.accumulate_pivot(row, row, first);
            if mat.bit(row, row) == 0 {
                return false;
            }
            mat.clear_below(row, row, first);
        }
        true
    }

    fn assert_same_matrix(actual: &BitMatrix, expected: &BitMatrix, columns: usize, label: &str) {
        for row in 0..expected.rows() {
            for bit in 0..columns {
                assert_eq!(
                    actual.bit(row, bit),
                    expected.bit(row, bit),
                    "{label}: row {row} bit {bit}"
                );
            }
        }
    }

    /// Reduce a matrix to systematic form the original way: forward elimination one row at a
    /// time, then back-substitution in blocks.
    fn reduce_row_at_a_time(mat: &mut BitMatrix, count: usize) -> bool {
        if !eliminate_row_at_a_time(mat, count) {
            return false;
        }
        let mut end = count;
        while end != 0 {
            let start = end - 1;
            mat.clear_above_block::<1>(start);
            end = start;
        }
        true
    }

    /// The rows *below* the pivots must be identical under either schedule.
    ///
    /// This is the invariant the semi-systematic parameter sets depend on: `move_columns` reads
    /// the contents of specific rows below the pivots processed so far, so those contents must
    /// not depend on how the pivots above them were scheduled. They do not, and the reason is
    /// forced: after elimination a trailing row has zeros in every pivot column, and the only
    /// vector in its coset with that property is unique, because the pivot rows' projection onto
    /// the pivot columns is invertible.
    ///
    /// The pivot rows themselves are *not* identical — this panel leaves them as an identity
    /// block in the panel columns, where the row-at-a-time schedule leaves them in echelon form.
    /// Both reach the same reduced row echelon form once back-substitution runs, which is what
    /// `reaches_the_same_systematic_form` below checks.
    #[test]
    fn panel_elimination_matches_row_at_a_time_below_the_pivots() {
        for &(rows, columns, seed) in &[
            (128usize, 256usize, 0x1234_5678_9ABC_DEF0u64),
            (128, 320, 0x0FED_CBA9_8765_4321),
            (192, 512, 0xDEAD_BEEF_CAFE_F00D),
            (256, 448, 0x9E37_79B9_7F4A_7C15),
        ] {
            for &panel in &[16usize, 32] {
                // Eliminate half the rows, so every pivot has rows below it to draw from.
                let count = rows / 2;
                if count % panel != 0 {
                    continue;
                }
                let (mut expected, mut actual) = random_pair(rows, columns, seed ^ panel as u64);

                assert!(
                    eliminate_row_at_a_time(&mut expected, count),
                    "reference elimination failed for {rows}x{columns}"
                );

                let mut shadow = vec![0u64; rows];
                let mut scratch = PanelScratch::new(rows, actual.stride(), panel);
                let mut start = 0;
                while start < count {
                    let ok = if panel == 16 {
                        actual.eliminate_panel::<16>(start, &mut shadow, &mut scratch)
                    } else {
                        actual.eliminate_panel::<32>(start, &mut shadow, &mut scratch)
                    };
                    assert!(ok, "panel elimination failed at {start}");
                    start += panel;
                }

                for row in count..rows {
                    for bit in 0..columns {
                        assert_eq!(
                            actual.bit(row, bit),
                            expected.bit(row, bit),
                            "{rows}x{columns} panel {panel}: trailing row {row} bit {bit}"
                        );
                    }
                }
            }
        }
    }

    /// Both schedules must reach the same systematic form, every bit of it.
    ///
    /// The reduced row echelon form of a fixed row space with fixed pivot columns is unique, so
    /// this is the invariant that makes the public key independent of the elimination schedule.
    #[test]
    fn panel_elimination_reaches_the_same_systematic_form() {
        for &(rows, columns, seed) in &[
            (128usize, 256usize, 0x2545_F491_4F6C_DD1Du64),
            (128, 320, 0xBF58_476D_1CE4_E5B9),
            (192, 512, 0x94D0_49BB_1331_11EB),
        ] {
            for &panel in &[16usize, 32] {
                let count = rows / 2;
                if count % panel != 0 {
                    continue;
                }
                let (mut expected, mut actual) = random_pair(rows, columns, seed ^ panel as u64);

                assert!(reduce_row_at_a_time(&mut expected, count));

                let mut shadow = vec![0u64; rows];
                let mut scratch = PanelScratch::new(rows, actual.stride(), panel);
                let mut start = 0;
                while start < count {
                    let ok = if panel == 16 {
                        actual.eliminate_panel::<16>(start, &mut shadow, &mut scratch)
                    } else {
                        actual.eliminate_panel::<32>(start, &mut shadow, &mut scratch)
                    };
                    assert!(ok);
                    start += panel;
                }
                let mut end = count;
                while end != 0 {
                    let start = end - 1;
                    actual.clear_above_block::<1>(start);
                    end = start;
                }

                assert_same_matrix(
                    &actual,
                    &expected,
                    columns,
                    &format!("{rows}x{columns} panel {panel} systematic form"),
                );
            }
        }
    }

    /// A panel must report the same rejection the row-at-a-time schedule does.
    ///
    /// A matrix whose first column is entirely zero has no systematic form, which key generation
    /// answers by retrying with a fresh seed. Both schedules must agree on that.
    #[test]
    fn panel_elimination_rejects_a_singular_matrix() {
        let (mut expected, mut actual) = random_pair(64, 256, 0x0BAD_C0DE_1234_5678);
        for row in 0..64 {
            // Clear column zero in every row, so no pivot can ever be found for it.
            let word = row * expected.stride;
            expected.data[word] &= !1;
            actual.data[word] &= !1;
        }

        assert!(!eliminate_row_at_a_time(&mut expected, 64));

        let mut shadow = vec![0u64; 64];
        let mut scratch = PanelScratch::new(64, actual.stride(), 16);
        assert!(!actual.eliminate_panel::<16>(0, &mut shadow, &mut scratch));
    }

    /// Partial panels are what the ragged tail of a real parameter set runs into.
    #[test]
    fn a_panel_narrower_than_sixteen_still_matches() {
        let (mut expected, mut actual) = random_pair(64, 320, 0x7777_8888_9999_AAAA);
        assert!(eliminate_row_at_a_time(&mut expected, 8));

        let mut shadow = vec![0u64; 64];
        let mut scratch = PanelScratch::new(64, actual.stride(), 8);
        assert!(actual.eliminate_panel::<8>(0, &mut shadow, &mut scratch));

        for row in 8..64 {
            for bit in 0..320 {
                assert_eq!(
                    actual.bit(row, bit),
                    expected.bit(row, bit),
                    "8-wide panel: trailing row {row} bit {bit}"
                );
            }
        }
    }
}

/// What the systematic-form reduction costs on its own.
///
/// Ignored by default; it measures rather than asserts. The shape is `mceliece8192128`'s:
/// `mt = 1664` rows of `n = 8192` columns. Companion to `controlbits::timing`.
#[cfg(all(test, feature = "keygen"))]
mod timing {
    use super::*;
    use std::time::Instant;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    #[ignore]
    fn timing_of_the_systematic_reduction() {
        const ROWS: usize = 1664;
        const COLS: usize = 8192;

        let mut rng = Rng(0xF00D_CAFE_1357_2468);
        let (mut forward, mut back) = (0.0f64, 0.0f64);
        let start = Instant::now();
        const RUNS: usize = 5;
        for _ in 0..RUNS {
            let mut mat = BitMatrix::zeros(ROWS, COLS);
            for row in 0..ROWS {
                for byte in 0..COLS / 8 {
                    mat.set_byte(row, byte, rng.next() as u8);
                }
            }

            // The same schedule `mat_gen` runs, timed in its two halves: forward elimination
            // row by row, then back-substitution in blocks of sixteen.
            let t0 = Instant::now();
            for row in 0..ROWS {
                let first_word = row / 64;
                mat.accumulate_pivot(row, row, first_word);
                mat.clear_below(row, row, first_word);
            }
            forward += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut end = ROWS;
            while end >= 16 {
                let start_row = end - 16;
                mat.clear_above_block::<16>(start_row);
                end = start_row;
            }
            back += t1.elapsed().as_secs_f64();
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / RUNS as f64;
        let fwd = forward * 1e6 / RUNS as f64;
        let bck = back * 1e6 / RUNS as f64;
        println!("\n  systematic reduction (1664x8192) {us:>10.1} us");
        println!(
            "    forward elimination (row at a time) {fwd:>8.1} us  ({:.0}%)",
            fwd / us * 100.0
        );
        println!(
            "    back-substitution (16-pivot blocks) {bck:>8.1} us  ({:.0}%)\n",
            bck / us * 100.0
        );
    }
}
