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
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    if n < 2 {
        return;
    }

    let top = n >> 1;
    let mut p = top;
    while p > 0 {
        let mut base = 0;
        while base < n {
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
}
