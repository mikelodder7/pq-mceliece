/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Constant-time sorting networks.
//!
//! Both sorts are data-oblivious merge-exchange networks in the style of
//! [djbsort](https://sorting.cr.yp.to/): the sequence of comparisons depends only on the
//! length, and each comparison is a branch-free conditional exchange.

/// Order packed nonnegative index pairs without branching.
///
/// Control-bit generation keeps every operand between zero and `0x3fff_ffff`, so their
/// difference cannot overflow an `i32`. This avoids the widened comparison needed by a general
/// signed sorter.
#[inline]
#[cfg(feature = "keygen")]
pub(crate) const fn minmax_packed_i32(a: i32, b: i32) -> (i32, i32) {
    let mask = a.wrapping_sub(b) >> 31;
    ((a & mask) | (b & !mask), (a & !mask) | (b & mask))
}

/// Order a pair of unsigned 16-bit integers without branching.
#[inline]
#[cfg(feature = "encapsulate")]
const fn minmax_u16(a: u16, b: u16) -> (u16, u16) {
    let mask = (((a as i32) - (b as i32)) >> 31) as u16;
    ((a & mask) | (b & !mask), (a & !mask) | (b & mask))
}

/// Order a pair of unsigned integers without branching.
///
/// Same idea as the signed case: widening past the operands' width makes the difference exact,
/// so its sign bit is the comparison.
#[inline]
#[cfg(feature = "keygen")]
const fn minmax_u64(a: u64, b: u64) -> (u64, u64) {
    let mask = (((a as i128) - (b as i128)) >> 127) as u64;
    ((a & mask) | (b & !mask), (a & !mask) | (b & mask))
}

/// How many independent chains the strided stage advances together.
///
/// Four measured fastest; two is close, and eight or more starts spilling.
const CHAIN_WIDTH: usize = 4;

/// The NEON comparator below handles exactly one vector's worth of lanes per call, so retuning
/// [`CHAIN_WIDTH`] without widening it would silently leave the rest of each chain unsorted on
/// AArch64 and nowhere else. Fail the build instead.
#[cfg(all(feature = "keygen", target_arch = "aarch64"))]
const _: () = assert!(CHAIN_WIDTH == 4);

#[cfg(all(feature = "keygen", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn minmax_packed_i32x4(left: *mut i32, right: *mut i32) {
    use core::arch::aarch64::{vld1q_s32, vmaxq_s32, vminq_s32, vst1q_s32};

    // SAFETY: the caller provides two disjoint, writable four-element regions. AArch64 NEON
    // loads and stores permit unaligned pointers, and every AArch64 target has NEON.
    unsafe {
        let a = vld1q_s32(left);
        let b = vld1q_s32(right);
        vst1q_s32(left, vminq_s32(a, b));
        vst1q_s32(right, vmaxq_s32(a, b));
    }
}

/// The x86 counterparts of the NEON comparator. `pminsd`/`pmaxsd` need SSE4.1 at minimum, which
/// is above the `x86_64` baseline, so unlike AArch64 these cannot be reached at compile time;
/// the sorter picks a whole-network kernel from [`super::simd::level`] at runtime instead.
/// The network keeps both widths in a cascade: the small strides carry the most chain levels,
/// so a network that let `p = 4` fall to scalar code measured slower on the whole key
/// generation, not faster.
#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn minmax_packed_i32x4_x86(left: *mut i32, right: *mut i32) {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_max_epi32, _mm_min_epi32, _mm_storeu_si128,
    };

    // SAFETY: the caller provides two disjoint, writable four-element regions and a CPU with
    // `sse4.1`. Loads and stores are the unaligned forms, so no alignment precondition applies.
    unsafe {
        let a = _mm_loadu_si128(left as *const __m128i);
        let b = _mm_loadu_si128(right as *const __m128i);
        _mm_storeu_si128(left as *mut __m128i, _mm_min_epi32(a, b));
        _mm_storeu_si128(right as *mut __m128i, _mm_max_epi32(a, b));
    }
}

#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn minmax_packed_i32x8(left: *mut i32, right: *mut i32) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_max_epi32, _mm256_min_epi32, _mm256_storeu_si256,
    };

    // SAFETY: the caller provides two disjoint, writable eight-element regions and a CPU with
    // `avx2`. Loads and stores are the unaligned forms, so no alignment precondition applies.
    unsafe {
        let a = _mm256_loadu_si256(left as *const __m256i);
        let b = _mm256_loadu_si256(right as *const __m256i);
        _mm256_storeu_si256(left as *mut __m256i, _mm256_min_epi32(a, b));
        _mm256_storeu_si256(right as *mut __m256i, _mm256_max_epi32(a, b));
    }
}

#[cfg(feature = "keygen")]
macro_rules! batch_minmax_packed_i32 {
    ($left:expr, $right:expr) => {{
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `left` has exactly `CHAIN_WIDTH` elements, `right` is the disjoint
            // `CHAIN_WIDTH`-element sorting window, and `CHAIN_WIDTH` is four.
            unsafe {
                minmax_packed_i32x4($left.as_mut_ptr(), $right.as_mut_ptr());
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (a, b) in $left.iter_mut().zip($right.iter_mut()) {
                let (lo, hi) = minmax_packed_i32(*a, *b);
                *a = lo;
                *b = hi;
            }
        }
    }};
}

#[cfg(feature = "encapsulate")]
macro_rules! batch_minmax_u16 {
    ($left:expr, $right:expr) => {
        for (a, b) in $left.iter_mut().zip($right.iter_mut()) {
            let (lo, hi) = minmax_u16(*a, *b);
            *a = lo;
            *b = hi;
        }
    };
}

#[cfg(feature = "keygen")]
macro_rules! batch_minmax_u64 {
    ($left:expr, $right:expr) => {
        for (a, b) in $left.iter_mut().zip($right.iter_mut()) {
            let (lo, hi) = minmax_u64(*a, *b);
            *a = lo;
            *b = hi;
        }
    };
}

macro_rules! sorter {
    ($(#[$meta:meta])* $name:ident, $ty:ty, $minmax:ident, $batch_minmax:ident) => {
        $(#[$meta])*
        pub(crate) fn $name(x: &mut [$ty]) {
            let n = x.len();
            if n < 2 {
                return;
            }

            let mut top = 1;
            while top < n - top {
                top += top;
            }

            let mut p = top;
            while p > 0 {
                // The indices this stage touches are those with one particular bit clear, and
                // since the stride is a power of two they come in contiguous runs. Walking the
                // runs, rather than testing every index, leaves the inner loop branch free and
                // over two provably disjoint slices, which is what lets it vectorize.
                let mut base = 0;
                while base < n - p {
                    let run = p.min(n - p - base);
                    let (left, right) = x[base..base + p + run].split_at_mut(p);
                    for (a, b) in left[..run].iter_mut().zip(right[..run].iter_mut()) {
                        let (lo, hi) = $minmax(*a, *b);
                        *a = lo;
                        *b = hi;
                    }
                    base += p * 2;
                }

                // This stage carries a value down a chain of strides for each index. The
                // chain stays in a register, which is why the loops are not swapped here:
                // hoisting the chain outward and sweeping runs vectorizes the inner step but
                // spills the carried value to memory, and measured slower at every width
                // tried. Neighboring indices are independent, though, so a fixed number of
                // them advance together. The width has to be a compile-time constant: with a
                // runtime length the carried values spill to memory and the gain disappears.
                //
                // `index` deliberately carries across the whole `q` loop instead of restarting
                // at zero. Each index belongs to exactly one `q` -- the largest one it fits
                // under -- and since `n - q` only grows as `q` shrinks, sweeping from where the
                // previous `q` stopped visits every index once. Restarting would repeat the
                // earlier indices under every smaller stride, which still sorts but costs
                // nearly four times the comparisons at these lengths.
                let mut index = 0;
                let mut q = top;
                while q > p {
                    let limit = n - q;
                    while index < limit {
                        // Indices with this bit clear come in contiguous runs of `p` starting
                        // at multiples of `2 * p`, so the run holding `index` ends here.
                        let block_end = (index | (p - 1)) + 1;
                        if index & p != 0 {
                            index = block_end;
                            continue;
                        }
                        let end = block_end.min(limit);
                        let run = end - index;

                        let mut offset = 0;
                        while offset + CHAIN_WIDTH <= run {
                            let i = index + offset;
                            let mut carry = [<$ty>::default(); CHAIN_WIDTH];
                            carry.copy_from_slice(&x[i + p..i + p + CHAIN_WIDTH]);

                            let mut r = q;
                            while r > p {
                                let window = &mut x[i + r..i + r + CHAIN_WIDTH];
                                $batch_minmax!(&mut carry, window);
                                r >>= 1;
                            }

                            x[i + p..i + p + CHAIN_WIDTH].copy_from_slice(&carry);
                            offset += CHAIN_WIDTH;
                        }

                        for i in index + offset..end {
                            let mut a = x[i + p];
                            let mut r = q;
                            while r > p {
                                let (lo, hi) = $minmax(a, x[i + r]);
                                a = lo;
                                x[i + r] = hi;
                                r >>= 1;
                            }
                            x[i + p] = a;
                        }

                        index = end;
                    }
                    q >>= 1;
                }
                p >>= 1;
            }
        }
    };
}

sorter! {
    #[cfg(all(feature = "keygen", test))]
    /// Sort packed nonnegative index pairs in ascending order in constant time.
    sort_packed_i32,
    i32,
    minmax_packed_i32,
    batch_minmax_packed_i32
}

/// Sort a power-of-two slice of packed control-bit values.
///
/// Every invocation made by the control-bit recursion has this shape. Knowing that the runs
/// divide the input exactly removes the ragged-tail bounds and minimum calculations from the
/// generic sorter without changing its compare-exchange schedule.
#[cfg(feature = "keygen")]
pub(crate) fn sort_packed_i32_power_of_two(x: &mut [i32]) {
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::{Level, level};
        // Both vector tiers run the eight-lane network: a sixteen-lane AVX-512 form measured
        // 1.9% slower for whole key generation on Zen 5, where the 512-bit datapath is
        // double-pumped and the sixty-four-byte windows split cache lines.
        if level() >= Level::Avx2 {
            // SAFETY: the level was detected from CPUID, so the target features the kernel
            // names are present.
            return unsafe { sort_packed_i32_pow2_avx2(x) };
        }
    }
    sort_packed_i32_power_of_two_portable(x)
}

/// One vector of the contiguous stage at stride one or two: each `2p` block's halves compare
/// in place, with an in-register swap standing in for the second stream.
///
/// The NEON twin of `minmax_adjacent_x86`: `REV64` faces stride-one partners, `EXT` by two
/// lanes faces stride-two partners, and a bit-select keeps each maximum in its high lane.
/// Both strides divide the four-lane vector, so whole blocks are always covered.
#[cfg(all(feature = "keygen", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn minmax_adjacent_neon(values: *mut i32, p: usize) {
    use core::arch::aarch64::{
        vbslq_s32, vextq_s32, vld1q_s32, vld1q_u32, vmaxq_s32, vminq_s32, vrev64q_s32, vst1q_s32,
    };

    debug_assert!(p == 1 || p == 2);

    // SAFETY: the caller provides four writable elements, and `neon` is part of the AArch64
    // baseline. The load and store forms permit unaligned pointers.
    unsafe {
        let v = vld1q_s32(values);
        let (partner, keep_max) = if p == 1 {
            (vrev64q_s32(v), [0u32, u32::MAX, 0, u32::MAX])
        } else {
            (vextq_s32::<2>(v, v), [0u32, 0, u32::MAX, u32::MAX])
        };
        let lo = vminq_s32(v, partner);
        let hi = vmaxq_s32(v, partner);
        vst1q_s32(values, vbslq_s32(vld1q_u32(keep_max.as_ptr()), hi, lo));
    }
}

/// Advance eight strided chains of the `p = 1` or `p = 2` stage through one whole `r`
/// cascade, sixteen elements at a time.
///
/// The NEON twin of `small_stride_chain_x86`, with one simplification: both strides'
/// residue classes fall inside a single four-lane register, so the window alignment is an
/// `EXT` against zero within each register rather than a cross-register permute. Carries sit
/// in lanes `p..2p-1` modulo `2p` and stay resident across the cascade; windows load at
/// `base + r`, shift up `p` lanes to face them, and the maxima blend back into the window
/// lanes only, leaving the interleaved foreign lanes untouched.
///
/// Running the cascades in lockstep over descending `r` preserves the scalar per-chain order
/// exactly; the argument is the `small_stride_chain_x86` doc comment's. Same-`r` accesses of
/// different chains touch distinct words, so the register order within one step is free. The
/// carries fold back through a reload-and-blend because head lanes double as other chains'
/// window words while the cascade is in flight.
#[cfg(all(feature = "keygen", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn small_stride_chain_neon(base: *mut i32, q: usize, p: usize) {
    use core::arch::aarch64::{
        int32x4_t, vbslq_s32, vdupq_n_s32, vextq_s32, vld1q_s32, vld1q_u32, vmaxq_s32, vminq_s32,
        vst1q_s32,
    };

    debug_assert!(p == 1 || p == 2);

    // SAFETY: the caller guarantees sixteen writable elements at `base` and sixteen
    // readable, writable elements at every `base + r` the cascade visits; `neon` is part of
    // the AArch64 baseline and the load and store forms permit unaligned pointers.
    unsafe {
        let carry_lanes = if p == 1 {
            [0u32, u32::MAX, 0, u32::MAX]
        } else {
            [0u32, 0, u32::MAX, u32::MAX]
        };
        let keep = vld1q_u32(carry_lanes.as_ptr());
        let zero = vdupq_n_s32(0);

        let mut carries: [int32x4_t; 4] = [
            vld1q_s32(base),
            vld1q_s32(base.add(4)),
            vld1q_s32(base.add(8)),
            vld1q_s32(base.add(12)),
        ];

        let mut r = q;
        while r > p {
            for (reg, carry) in carries.iter_mut().enumerate() {
                let window = base.add(r + 4 * reg);
                let w = vld1q_s32(window);
                let aligned = if p == 1 {
                    vextq_s32::<3>(zero, w)
                } else {
                    vextq_s32::<2>(zero, w)
                };
                let lo = vminq_s32(*carry, aligned);
                let hi = vmaxq_s32(*carry, aligned);
                *carry = vbslq_s32(keep, lo, *carry);
                let back = if p == 1 {
                    vextq_s32::<1>(hi, zero)
                } else {
                    vextq_s32::<2>(hi, zero)
                };
                vst1q_s32(window, vbslq_s32(keep, w, back));
            }
            r >>= 1;
        }

        for (reg, carry) in carries.iter().enumerate() {
            let head = base.add(4 * reg);
            vst1q_s32(head, vbslq_s32(keep, *carry, vld1q_s32(head)));
        }
    }
}

/// The compile-time-dispatched form of [`sort_packed_i32_power_of_two`]: NEON comparators on
/// AArch64, the arithmetic-mask idiom everywhere else.
#[cfg(feature = "keygen")]
fn sort_packed_i32_power_of_two_portable(x: &mut [i32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    if n < 2 {
        return;
    }

    let top = n >> 1;
    let mut p = top;
    while p > 0 {
        // At strides one and two, whole `2p` blocks fit inside one vector, so the stage runs
        // as an in-register swap-compare-blend per four lanes; the scalar walk below covers
        // inputs shorter than a vector.
        #[cfg(target_arch = "aarch64")]
        let contiguous_done = p <= 2 && {
            let mut k = 0;
            while k + 4 <= n {
                // SAFETY: `k + 4 <= n` keeps the four elements in bounds.
                unsafe {
                    if p == 1 {
                        minmax_adjacent_neon(x.as_mut_ptr().add(k), 1);
                    } else {
                        minmax_adjacent_neon(x.as_mut_ptr().add(k), 2);
                    }
                }
                k += 4;
            }
            while k < n {
                let (left, right) = x[k..k + 2 * p].split_at_mut(p);
                for (a, b) in left.iter_mut().zip(right.iter_mut()) {
                    let (lo, hi) = minmax_packed_i32(*a, *b);
                    *a = lo;
                    *b = hi;
                }
                k += 2 * p;
            }
            true
        };
        #[cfg(not(target_arch = "aarch64"))]
        let contiguous_done = false;

        let mut base = 0;
        while !contiguous_done && base < n {
            let (left, right) = x[base..base + 2 * p].split_at_mut(p);
            // The two halves are contiguous and disjoint, which is exactly the comparator's
            // shape, so the vector form applies directly. `p` is a power of two, so the
            // vector loop is exact for `p >= 4` and the scalar loop below covers `p < 4`
            // whole; LLVM does not recognize `minmax_packed_i32`'s arithmetic mask as a
            // min/max, so without this the stage compiles to three times the operations.
            #[cfg(target_arch = "aarch64")]
            let mut k = 0;
            #[cfg(not(target_arch = "aarch64"))]
            let k = 0;
            #[cfg(target_arch = "aarch64")]
            while k + CHAIN_WIDTH <= p {
                // SAFETY: `left` and `right` are disjoint `p`-element slices and
                // `k + CHAIN_WIDTH <= p`, so both four-element regions are in bounds.
                unsafe {
                    minmax_packed_i32x4(left.as_mut_ptr().add(k), right.as_mut_ptr().add(k));
                }
                k += CHAIN_WIDTH;
            }
            for (a, b) in left[k..].iter_mut().zip(right[k..].iter_mut()) {
                let (lo, hi) = minmax_packed_i32(*a, *b);
                *a = lo;
                *b = hi;
            }
            base += 2 * p;
        }

        let mut index = 0;
        let mut q = top;
        while q > p {
            let limit = n - q;

            // Strides one and two never reach the four-wide chain loop below, so their
            // chains advance eight at a time in lockstep instead; see
            // [`small_stride_chain_neon`] for why the order is preserved. `index` stays a
            // multiple of `2p` across levels, which is the alignment the batch's residue
            // classes assume.
            #[cfg(target_arch = "aarch64")]
            if p <= 2 {
                while index + 16 <= limit {
                    // SAFETY: `index + 16 <= limit = n - q` keeps the sixteen carry words
                    // in bounds and every window below `index + q + 15 <= n - 1`.
                    unsafe {
                        if p == 1 {
                            small_stride_chain_neon(x.as_mut_ptr().add(index), q, 1);
                        } else {
                            small_stride_chain_neon(x.as_mut_ptr().add(index), q, 2);
                        }
                    }
                    index += 16;
                }
            }

            while index < limit {
                let block_end = (index | (p - 1)) + 1;
                if index & p != 0 {
                    index = block_end;
                    continue;
                }

                let mut offset = 0;
                while offset + CHAIN_WIDTH <= p {
                    let i = index + offset;
                    let mut carry = [0i32; CHAIN_WIDTH];
                    carry.copy_from_slice(&x[i + p..i + p + CHAIN_WIDTH]);

                    let mut r = q;
                    while r > p {
                        // SAFETY: `n`, `q`, `p` are powers of two with `2p <= q`, so
                        // `n - q` is a multiple of `2p` and the largest run start with bit
                        // `p` clear below `limit = n - q` is `n - q - 2p`. With
                        // `offset <= p - CHAIN_WIDTH` and `r <= q` the window ends at
                        // `i + r + CHAIN_WIDTH <= n - p`. The compiler cannot see this, and
                        // the range check it otherwise emits costs a third of the loop.
                        let window = unsafe { x.get_unchecked_mut(i + r..i + r + CHAIN_WIDTH) };
                        batch_minmax_packed_i32!(&mut carry, window);
                        r >>= 1;
                    }

                    x[i + p..i + p + CHAIN_WIDTH].copy_from_slice(&carry);
                    offset += CHAIN_WIDTH;
                }

                for i in index + offset..block_end {
                    let chain = &mut x[i..=i + q];
                    let mut a = chain[p];
                    let mut r = q;
                    while r > p {
                        let (lo, hi) = minmax_packed_i32(a, chain[r]);
                        a = lo;
                        chain[r] = hi;
                        r >>= 1;
                    }
                    chain[p] = a;
                }

                index = block_end;
            }
            q >>= 1;
        }
        p >>= 1;
    }
}

/// Advance eight strided chains of the `p = $p` stage through one whole `r` cascade.
///
/// At strides below four the runs are too short for the width cascade and the chain stage was
/// measured spending half the whole sort in scalar compare-exchanges. Sixteen consecutive
/// elements hold eight independent chains: their carries sit in lanes `p..2p-1` modulo `2p`
/// of `x[index..index + 16]` and each step's windows in lanes `0..p-1` modulo `2p` of
/// `x[index + r..]`, so one in-register shift by `p` lanes aligns a window load with the
/// carries, and blends confine every write to its own residue class.
///
/// Running the eight cascades in lockstep over descending `r` preserves the scalar
/// per-chain order exactly: any two chains that touch the same word do so with the later
/// access at the smaller stride, and descending `r` replays those accesses in that order.
/// The carries fold back through a reload-and-blend at the end, because head lanes double as
/// other chains' window words while the cascade is in flight.
#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
macro_rules! small_stride_chain_x86 {
    ($(#[$meta:meta])* $name:ident, $p:literal, $up:expr, $down:expr,
     $carry_blend:literal, $window_blend:literal) => {
        $(#[$meta])*
        #[target_feature(enable = "avx2")]
        unsafe fn $name(base: *mut i32, q: usize) {
            use core::arch::x86_64::{
                __m256i, _mm256_blend_epi32, _mm256_loadu_si256, _mm256_max_epi32,
                _mm256_min_epi32, _mm256_permutevar8x32_epi32, _mm256_setr_epi32,
                _mm256_storeu_si256,
            };

            // SAFETY: the caller guarantees sixteen writable elements at `base` and sixteen
            // readable, writable elements at every `base + r` the cascade visits. Unaligned
            // forms throughout, so no alignment precondition applies.
            unsafe {
                let up = $up;
                let down = $down;
                let mut carry0 = _mm256_loadu_si256(base as *const __m256i);
                let mut carry1 = _mm256_loadu_si256(base.add(8) as *const __m256i);

                let mut r = q;
                while r > $p {
                    let window0 = base.add(r);
                    let window1 = base.add(r + 8);
                    let w0 = _mm256_loadu_si256(window0 as *const __m256i);
                    let w1 = _mm256_loadu_si256(window1 as *const __m256i);
                    let aligned0 = _mm256_permutevar8x32_epi32(w0, up);
                    let aligned1 = _mm256_permutevar8x32_epi32(w1, up);

                    let lo0 = _mm256_min_epi32(carry0, aligned0);
                    let lo1 = _mm256_min_epi32(carry1, aligned1);
                    let hi0 = _mm256_max_epi32(carry0, aligned0);
                    let hi1 = _mm256_max_epi32(carry1, aligned1);
                    carry0 = _mm256_blend_epi32::<$carry_blend>(carry0, lo0);
                    carry1 = _mm256_blend_epi32::<$carry_blend>(carry1, lo1);

                    let back0 = _mm256_permutevar8x32_epi32(hi0, down);
                    let back1 = _mm256_permutevar8x32_epi32(hi1, down);
                    _mm256_storeu_si256(
                        window0 as *mut __m256i,
                        _mm256_blend_epi32::<$window_blend>(w0, back0),
                    );
                    _mm256_storeu_si256(
                        window1 as *mut __m256i,
                        _mm256_blend_epi32::<$window_blend>(w1, back1),
                    );
                    r >>= 1;
                }

                let head0 = _mm256_loadu_si256(base as *const __m256i);
                let head1 = _mm256_loadu_si256(base.add(8) as *const __m256i);
                _mm256_storeu_si256(
                    base as *mut __m256i,
                    _mm256_blend_epi32::<$carry_blend>(head0, carry0),
                );
                _mm256_storeu_si256(
                    base.add(8) as *mut __m256i,
                    _mm256_blend_epi32::<$carry_blend>(head1, carry1),
                );
            }
        }
    };
}

#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
small_stride_chain_x86! {
    /// Eight lockstep chains of the stride-one stage: carries in odd lanes, windows in even.
    chain_batch_stride1_x86, 1,
    _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6),
    _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7),
    0b1010_1010, 0b0101_0101
}

#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
small_stride_chain_x86! {
    /// Eight lockstep chains of the stride-two stage: carries in lanes two and three of each
    /// four, windows in lanes zero and one.
    chain_batch_stride2_x86, 2,
    _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5),
    _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 7, 7),
    0b1100_1100, 0b0011_0011
}

/// One vector of the contiguous stage at stride one or two: each `2p` block's halves compare
/// in place, with an in-register swap standing in for the second stream.
///
/// `SWAP` permutes each 128-bit half so every lane faces its partner; `KEEP_MAX` selects
/// which lanes keep the maximum. Both strides divide the vector width, so whole blocks are
/// always covered.
#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn minmax_adjacent_x86<const SWAP: i32, const KEEP_MAX: i32>(values: *mut i32) {
    use core::arch::x86_64::{
        __m256i, _mm256_blend_epi32, _mm256_loadu_si256, _mm256_max_epi32, _mm256_min_epi32,
        _mm256_shuffle_epi32, _mm256_storeu_si256,
    };

    // SAFETY: the caller provides eight writable elements and a CPU with `avx2`. Loads and
    // stores are the unaligned forms, so no alignment precondition applies.
    unsafe {
        let v = _mm256_loadu_si256(values as *const __m256i);
        let swapped = _mm256_shuffle_epi32::<SWAP>(v);
        let lo = _mm256_min_epi32(v, swapped);
        let hi = _mm256_max_epi32(v, swapped);
        _mm256_storeu_si256(
            values as *mut __m256i,
            _mm256_blend_epi32::<KEEP_MAX>(lo, hi),
        );
    }
}

/// The x86 vector form of the power-of-two sorter.
///
/// The body is the portable sorter with the comparator width lifted to a cascade of vector
/// widths: each stage runs the widest comparator its stride admits, so `p >= W` strides
/// advance a whole top-width vector of chains per step and the smaller strides — which carry
/// the most chain levels — still run their own vector widths instead of falling to scalar
/// code. The compare-exchange schedule is identical at every width; only the grouping of
/// independent comparisons changes.
#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
macro_rules! sort_packed_i32_pow2_x86 {
    ($(#[$meta:meta])* $name:ident, $feature:literal, $(($width:expr, $minmax_vec:ident)),+) => {
        $(#[$meta])*
        #[target_feature(enable = $feature)]
        unsafe fn $name(x: &mut [i32]) {
            let n = x.len();
            debug_assert!(n.is_power_of_two());
            if n < 2 {
                return;
            }

            let top = n >> 1;
            let mut p = top;
            while p > 0 {
                if p <= 2 {
                    // At strides one and two, whole `2p` blocks fit inside one vector, so
                    // the stage runs as an in-register swap-compare-blend per eight lanes;
                    // the scalar loop covers inputs shorter than a vector.
                    let mut k = 0;
                    while k + 8 <= n {
                        // SAFETY: `k + 8 <= n` keeps the eight elements in bounds and the
                        // dispatched level guarantees `avx2`.
                        unsafe {
                            if p == 1 {
                                minmax_adjacent_x86::<0b10_11_00_01, 0b1010_1010>(
                                    x.as_mut_ptr().add(k),
                                );
                            } else {
                                minmax_adjacent_x86::<0b01_00_11_10, 0b1100_1100>(
                                    x.as_mut_ptr().add(k),
                                );
                            }
                        }
                        k += 8;
                    }
                    while k < n {
                        let (left, right) = x[k..k + 2 * p].split_at_mut(p);
                        for (a, b) in left.iter_mut().zip(right.iter_mut()) {
                            let (lo, hi) = minmax_packed_i32(*a, *b);
                            *a = lo;
                            *b = hi;
                        }
                        k += 2 * p;
                    }
                } else {
                    let mut base = 0;
                    while base < n {
                        let (left, right) = x[base..base + 2 * p].split_at_mut(p);
                        // `p` and the widths are powers of two, so exactly one cascade
                        // level runs and it covers `p` whole.
                        let mut k = 0;
                        $(
                            while k + $width <= p {
                                // SAFETY: `left` and `right` are disjoint `p`-element
                                // slices and `k + $width <= p`, so both regions are in
                                // bounds. The detected level's target features cover the
                                // comparator's.
                                unsafe {
                                    $minmax_vec(
                                        left.as_mut_ptr().add(k),
                                        right.as_mut_ptr().add(k),
                                    );
                                }
                                k += $width;
                            }
                        )+
                        for (a, b) in left[k..].iter_mut().zip(right[k..].iter_mut()) {
                            let (lo, hi) = minmax_packed_i32(*a, *b);
                            *a = lo;
                            *b = hi;
                        }
                        base += 2 * p;
                    }
                }

                let mut index = 0;
                let mut q = top;
                while q > p {
                    let limit = n - q;
                    // Strides one and two never reach the width cascade below, so their
                    // chains advance eight at a time in lockstep instead; see
                    // [`small_stride_chain_x86`] for why the order is preserved. `index`
                    // stays a multiple of `2p` across levels, which is the alignment the
                    // batch's residue classes assume.
                    if p <= 2 {
                        while index + 16 <= limit {
                            // SAFETY: `index + 16 <= limit = n - q` keeps the sixteen
                            // carry words in bounds and every window below
                            // `index + q + 15 <= n - 1`; the dispatched level guarantees
                            // `avx2`.
                            unsafe {
                                if p == 1 {
                                    chain_batch_stride1_x86(x.as_mut_ptr().add(index), q);
                                } else {
                                    chain_batch_stride2_x86(x.as_mut_ptr().add(index), q);
                                }
                            }
                            index += 16;
                        }
                    }
                    while index < limit {
                        let block_end = (index | (p - 1)) + 1;
                        if index & p != 0 {
                            index = block_end;
                            continue;
                        }

                        let mut offset = 0;
                        $(
                            while offset + $width <= p {
                                let i = index + offset;
                                let mut carry = [0i32; $width];
                                carry.copy_from_slice(&x[i + p..i + p + $width]);

                                let mut r = q;
                                while r > p {
                                    // SAFETY: `n`, `q`, `p` are powers of two with `2p <= q`,
                                    // so `n - q` is a multiple of `2p` and the largest run
                                    // start with bit `p` clear below `limit = n - q` is
                                    // `n - q - 2p`. With `offset <= p - $width` and `r <= q`
                                    // the window ends at `i + r + $width <= n - p`. The
                                    // compiler cannot see this, and the range check it
                                    // otherwise emits costs a third of the loop.
                                    let window =
                                        unsafe { x.get_unchecked_mut(i + r..i + r + $width) };
                                    // SAFETY: `carry` and `window` are disjoint
                                    // `$width`-element regions and the detected level's
                                    // target features cover the comparator's.
                                    unsafe {
                                        $minmax_vec(carry.as_mut_ptr(), window.as_mut_ptr());
                                    }
                                    r >>= 1;
                                }

                                x[i + p..i + p + $width].copy_from_slice(&carry);
                                offset += $width;
                            }
                        )+

                        for i in index + offset..block_end {
                            let chain = &mut x[i..=i + q];
                            let mut a = chain[p];
                            let mut r = q;
                            while r > p {
                                let (lo, hi) = minmax_packed_i32(a, chain[r]);
                                a = lo;
                                chain[r] = hi;
                                r >>= 1;
                            }
                            chain[p] = a;
                        }

                        index = block_end;
                    }
                    q >>= 1;
                }
                p >>= 1;
            }
        }
    };
}

#[cfg(all(feature = "keygen", target_arch = "x86_64"))]
sort_packed_i32_pow2_x86! {
    /// The AVX2 form of [`sort_packed_i32_power_of_two`]: eight chains per step, four for the
    /// `p = 4` stride. `avx2` transitively enables the cascade's `sse4.1`.
    sort_packed_i32_pow2_avx2, "avx2",
    (8, minmax_packed_i32x8), (4, minmax_packed_i32x4_x86)
}

sorter! {
    #[cfg(feature = "encapsulate")]
    /// Sort a slice of `u16` in ascending order in constant time.
    sort_u16, u16, minmax_u16, batch_minmax_u16
}

sorter! {
    #[cfg(feature = "keygen")]
    /// Sort a slice of `u64` in ascending order in constant time.
    sort_u64, u64, minmax_u64, batch_minmax_u64
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A tiny xorshift generator; tests need reproducible pseudo-randomness, not cryptography.
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
    #[cfg(feature = "keygen")]
    fn packed_minmax_orders_control_bit_values() {
        let mut rng = Rng(0x6A09_E667_F3BC_C909);
        for _ in 0..10_000 {
            let a = (rng.next() as i32) & 0x3fff_ffff;
            let b = (rng.next() as i32) & 0x3fff_ffff;
            let (lo, hi) = minmax_packed_i32(a, b);
            assert_eq!((lo, hi), (a.min(b), a.max(b)));
        }
    }

    #[test]
    #[cfg(feature = "encapsulate")]
    fn minmax_orders_u16_pairs_including_extremes() {
        assert_eq!(minmax_u16(45, 17), (17, 45));
        assert_eq!(minmax_u16(u16::MAX, 0), (0, u16::MAX));
        assert_eq!(minmax_u16(7, 7), (7, 7));

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..10_000 {
            let a = rng.next() as u16;
            let b = rng.next() as u16;
            let (lo, hi) = minmax_u16(a, b);
            assert_eq!((lo, hi), (a.min(b), a.max(b)));
        }
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn minmax_orders_u64_pairs_including_extremes() {
        assert_eq!(minmax_u64(45, 17), (17, 45));
        assert_eq!(minmax_u64(u64::MAX, 0), (0, u64::MAX));
        assert_eq!(minmax_u64(7, 7), (7, 7));

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..10_000 {
            let a = rng.next();
            let b = rng.next();
            let (lo, hi) = minmax_u64(a, b);
            assert_eq!((lo, hi), (a.min(b), a.max(b)));
        }
    }

    /// Every short length, so no combination of stride, run boundary and ragged tail goes
    /// untried, plus a few large ones and the shortest parameter-set support size.
    const LENGTHS: [usize; 4] = [1000, 1024, 1025, 3488];

    #[test]
    #[cfg(feature = "keygen")]
    fn packed_sorting_matches_the_standard_library_for_many_lengths() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        for len in (0usize..300).chain(LENGTHS) {
            let packed: Vec<i32> = (0..len)
                .map(|_| (rng.next() as i32) & 0x3fff_ffff)
                .collect();
            let mut actual = packed.clone();
            sort_packed_i32(&mut actual);
            let mut expected = packed;
            expected.sort_unstable();
            assert_eq!(actual, expected, "sort_packed_i32 length {len}");

            let unsigned: Vec<u64> = (0..len).map(|_| rng.next()).collect();
            let mut actual = unsigned.clone();
            sort_u64(&mut actual);
            let mut expected = unsigned;
            expected.sort_unstable();
            assert_eq!(actual, expected, "sort_u64 length {len}");
        }
    }

    #[test]
    #[cfg(feature = "encapsulate")]
    fn u16_sorting_matches_the_standard_library_for_many_lengths() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        for len in (0usize..300).chain(LENGTHS) {
            let values: Vec<u16> = (0..len).map(|_| rng.next() as u16).collect();
            let mut actual = values.clone();
            sort_u16(&mut actual);
            let mut expected = values;
            expected.sort_unstable();
            assert_eq!(actual, expected, "sort_u16 length {len}");
        }
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn sorting_handles_duplicates_and_already_sorted_input() {
        let mut values = vec![5i32; 40];
        values.extend(0..40);
        let mut expected = values.clone();
        expected.sort_unstable();
        sort_packed_i32(&mut values);
        assert_eq!(values, expected);

        let mut values: Vec<u64> = (0..64).rev().collect();
        sort_u64(&mut values);
        assert_eq!(values, (0..64).collect::<Vec<u64>>());
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn power_of_two_sort_matches_the_generic_sort() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for power in 0..=13 {
            let values: Vec<i32> = (0..1usize << power)
                .map(|_| (rng.next() as i32) & 0x3fff_ffff)
                .collect();
            let mut actual = values.clone();
            sort_packed_i32_power_of_two(&mut actual);
            let mut expected = values;
            sort_packed_i32(&mut expected);
            assert_eq!(actual, expected, "power {power}");
        }
    }

    /// The vector network is called directly, not through
    /// [`level`](super::super::simd::level), so it is exercised even where detection would
    /// pick another path.
    #[test]
    #[cfg(all(feature = "keygen", target_arch = "x86_64"))]
    fn x86_tier_sorts_match_the_portable_sort() {
        let mut rng = Rng(0x8000_0000_B21D_C581);
        for power in 0..=13 {
            let values: Vec<i32> = (0..1usize << power)
                .map(|_| (rng.next() as i32) & 0x3fff_ffff)
                .collect();
            let mut expected = values.clone();
            sort_packed_i32_power_of_two_portable(&mut expected);

            if is_x86_feature_detected!("avx2") {
                let mut actual = values.clone();
                // SAFETY: the detected feature covers the kernel's requirement.
                unsafe { sort_packed_i32_pow2_avx2(&mut actual) };
                assert_eq!(actual, expected, "avx2 power {power}");
            }
        }
    }
}
