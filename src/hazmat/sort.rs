/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Constant-time sorting networks.
//!
//! Both sorts are data-oblivious merge-exchange networks in the style of
//! [djbsort](https://sorting.cr.yp.to/): the sequence of comparisons depends only on the
//! length, and each comparison is a branch-free conditional exchange.

/// Order a pair of signed integers without branching.
///
/// Widening to 64 bits first means the difference cannot overflow, so its sign bit answers the
/// comparison directly and the rest is a masked select. That is shorter and has a far tighter
/// dependency chain than doing the same with 32-bit borrow propagation, and it stays branchless
/// by construction rather than relying on the compiler to choose a conditional move.
#[inline]
const fn minmax_i32(a: i32, b: i32) -> (i32, i32) {
    let mask = (((a as i64) - (b as i64)) >> 63) as i32;
    ((a & mask) | (b & !mask), (a & !mask) | (b & mask))
}

/// Order a pair of unsigned integers without branching.
///
/// Same idea as the signed case: widening past the operands' width makes the difference exact,
/// so its sign bit is the comparison.
#[inline]
const fn minmax_u64(a: u64, b: u64) -> (u64, u64) {
    let mask = (((a as i128) - (b as i128)) >> 127) as u64;
    ((a & mask) | (b & !mask), (a & !mask) | (b & mask))
}

/// How many independent chains the strided stage advances together.
///
/// Four measured fastest; two is close, and eight or more starts spilling.
const CHAIN_WIDTH: usize = 4;

macro_rules! sorter {
    ($(#[$meta:meta])* $name:ident, $ty:ty, $minmax:ident) => {
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
                // tried.
                // This stage carries a value down a chain of strides. The chain is serial, but
                // neighbouring indices are independent, so a fixed number of them advance
                // together. The width has to be a compile-time constant: with a runtime length
                // the carried values spill to memory and the whole gain disappears.
                let mut q = top;
                while q > p {
                    let mut base = 0;
                    while base < n - q {
                        let run = p.min(n - q - base);

                        let mut offset = 0;
                        while offset + CHAIN_WIDTH <= run {
                            let i = base + offset;
                            let mut carry = [<$ty>::default(); CHAIN_WIDTH];
                            carry.copy_from_slice(&x[i + p..i + p + CHAIN_WIDTH]);

                            let mut r = q;
                            while r > p {
                                let window = &mut x[i + r..i + r + CHAIN_WIDTH];
                                for (a, b) in carry.iter_mut().zip(window.iter_mut()) {
                                    let (lo, hi) = $minmax(*a, *b);
                                    *a = lo;
                                    *b = hi;
                                }
                                r >>= 1;
                            }

                            x[i + p..i + p + CHAIN_WIDTH].copy_from_slice(&carry);
                            offset += CHAIN_WIDTH;
                        }

                        for i in base + offset..base + run {
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

                        base += p * 2;
                    }
                    q >>= 1;
                }
                p >>= 1;
            }
        }
    };
}

sorter! {
    /// Sort a slice of `i32` in ascending order in constant time.
    sort_i32, i32, minmax_i32
}

sorter! {
    /// Sort a slice of `u64` in ascending order in constant time.
    sort_u64, u64, minmax_u64
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
    fn minmax_orders_signed_pairs_including_extremes() {
        assert_eq!(minmax_i32(45, -17), (-17, 45));
        assert_eq!(minmax_i32(i32::MAX, i32::MIN), (i32::MIN, i32::MAX));
        assert_eq!(minmax_i32(7, 7), (7, 7));

        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for _ in 0..10_000 {
            let a = rng.next() as i32;
            let b = rng.next() as i32;
            let (lo, hi) = minmax_i32(a, b);
            assert_eq!((lo, hi), (a.min(b), a.max(b)));
        }
    }

    #[test]
    fn minmax_orders_unsigned_pairs_including_extremes() {
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

    #[test]
    fn sorting_matches_the_standard_library_for_many_lengths() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        // Lengths that are powers of two, and lengths that are not, exercise both the
        // fast path and the ragged tail of the network.
        for len in [
            0usize, 1, 2, 3, 5, 8, 16, 17, 63, 64, 100, 128, 255, 256, 1000,
        ] {
            let signed: Vec<i32> = (0..len).map(|_| rng.next() as i32).collect();
            let mut actual = signed.clone();
            sort_i32(&mut actual);
            let mut expected = signed;
            expected.sort_unstable();
            assert_eq!(actual, expected, "sort_i32 length {len}");

            let unsigned: Vec<u64> = (0..len).map(|_| rng.next()).collect();
            let mut actual = unsigned.clone();
            sort_u64(&mut actual);
            let mut expected = unsigned;
            expected.sort_unstable();
            assert_eq!(actual, expected, "sort_u64 length {len}");
        }
    }

    #[test]
    fn sorting_handles_duplicates_and_already_sorted_input() {
        let mut values = vec![5i32; 40];
        values.extend(0..40);
        let mut expected = values.clone();
        expected.sort_unstable();
        sort_i32(&mut values);
        assert_eq!(values, expected);

        let mut values: Vec<u64> = (0..64).rev().collect();
        sort_u64(&mut values);
        assert_eq!(values, (0..64).collect::<Vec<u64>>());
    }
}
