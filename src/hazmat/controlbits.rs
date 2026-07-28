/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
*/
//! Conversion of a permutation into Beneš network control bits.
//!
//! This is the Nassimi-Sahni algorithm in the form Bernstein specifies for conformance in
//! *Verified fast formulas for control bits for permutation networks*
//! (<https://cr.yp.to/papers/controlbits-20200923.pdf>). A permutation admits many control-bit
//! vectors; the standards require this particular one, so the routine here reproduces the
//! reference recursion exactly rather than choosing any valid encoding.

use super::sort::sort_i32;

/// Apply one stride-`2^s` layer of conditional swaps to `p`, driven by packed control bits.
///
/// `start` is the bit position of the layer's first control bit within `cb`. Tracking the
/// offset in bits rather than bytes keeps the routine correct for `n < 16`, where a whole
/// layer occupies less than one byte.
fn layer_i16(p: &mut [i16], cb: &[u8], start: usize, s: usize) {
    let n = p.len();
    let stride = 1usize << s;
    let mut index = start;

    let mut i = 0;
    while i < n {
        for j in i..i + stride {
            let mask = 0i16.wrapping_sub(((cb[index >> 3] >> (index & 7)) & 1) as i16);
            index += 1;
            let d = (p[j] ^ p[j + stride]) & mask;
            p[j] ^= d;
            p[j + stride] ^= d;
        }
        i += stride * 2;
    }
}

/// One recursion step, writing `(2w-1)n/2` control bits at positions `pos`, `pos + step`, ...
///
/// `temp` is scratch space of at least `2n` elements, reused by every level of the recursion.
/// `out` must be zeroed by the caller: bits are written with exclusive-or.
fn cbrecursion(
    out: &mut [u8],
    mut pos: usize,
    step: usize,
    pi: &[i16],
    w: usize,
    n: usize,
    temp: &mut [i32],
) {
    if w == 1 {
        out[pos >> 3] ^= (pi[0] as u8) << (pos & 7);
        return;
    }

    // `q` carries the two half-size permutations into the recursive calls. The reference
    // implementation aliases it into `temp`; keeping it separate costs `n` 16-bit values per
    // live recursion level and removes the aliasing entirely.
    let mut q = vec![0i16; n];

    {
        let (a, b) = temp.split_at_mut(n);
        let a = &mut a[..n];
        let b = &mut b[..n];

        for x in 0..n {
            a[x] = (((pi[x] ^ 1) as i32) << 16) | (pi[x ^ 1] as i32);
        }
        sort_i32(a); // a = (id << 16) + pibar

        for x in 0..n {
            let px = a[x] & 0xffff;
            let cx = px.min(x as i32);
            b[x] = (px << 16) | cx;
        }
        // b = (p << 16) + c

        for (x, slot) in a.iter_mut().enumerate() {
            *slot = (*slot << 16) | (x as i32);
        }
        sort_i32(a); // a = (id << 16) + pibar^-1

        for x in 0..n {
            a[x] = (a[x] << 16) + (b[x] >> 16);
        }
        sort_i32(a); // a = (id << 16) + pibar^2

        if w <= 10 {
            // Ten bits are enough to hold an index, so `p` and `c` share one 20-bit word.
            for x in 0..n {
                b[x] = ((a[x] & 0xffff) << 10) | (b[x] & 0x3ff);
            }

            for _ in 1..w - 1 {
                // b = (p << 10) + c
                for (x, slot) in a.iter_mut().enumerate() {
                    *slot = ((b[x] & !0x3ff) << 6) | (x as i32);
                }
                sort_i32(a); // a = (id << 16) + p^-1

                for x in 0..n {
                    a[x] = (a[x] << 20) | b[x];
                }
                sort_i32(a); // a = (id << 20) + (pp << 10) + cp

                for x in 0..n {
                    let ppcpx = a[x] & 0xfffff;
                    let ppcx = (a[x] & 0xffc00) | (b[x] & 0x3ff);
                    b[x] = ppcx.min(ppcpx);
                }
            }
            for slot in b.iter_mut() {
                *slot &= 0x3ff;
            }
        } else {
            for x in 0..n {
                b[x] = (a[x] << 16) | (b[x] & 0xffff);
            }

            for i in 1..w - 1 {
                // b = (p << 16) + c
                for (x, slot) in a.iter_mut().enumerate() {
                    *slot = (b[x] & !0xffff) | (x as i32);
                }
                sort_i32(a); // a = (id << 16) + p^-1

                for x in 0..n {
                    a[x] = (a[x] << 16) | (b[x] & 0xffff);
                }
                // a = (p^-1 << 16) + c

                if i < w - 2 {
                    for x in 0..n {
                        b[x] = (a[x] & !0xffff) | (b[x] >> 16);
                    }
                    sort_i32(b); // b = (id << 16) + p^-2
                    for x in 0..n {
                        b[x] = (b[x] << 16) | (a[x] & 0xffff);
                    }
                    // b = (p^-2 << 16) + c
                }

                sort_i32(a); // a = (id << 16) + cp
                for x in 0..n {
                    let cpx = (b[x] & !0xffff) | (a[x] & 0xffff);
                    b[x] = b[x].min(cpx);
                }
            }
            for slot in b.iter_mut() {
                *slot &= 0xffff;
            }
        }

        for (x, slot) in a.iter_mut().enumerate() {
            *slot = ((pi[x] as i32) << 16) + (x as i32);
        }
        sort_i32(a); // a = (id << 16) + pi^-1

        // The first output layer: `f`, one bit per pair.
        for j in 0..n / 2 {
            let x = 2 * j;
            let fj = b[x] & 1;
            let fx = x as i32 + fj;
            let fx1 = fx ^ 1;

            out[pos >> 3] ^= (fj << (pos & 7)) as u8;
            pos += step;

            b[x] = (a[x] << 16) | fx;
            b[x + 1] = (a[x + 1] << 16) | fx1;
        }
        sort_i32(b); // b = (id << 16) + F(pi)

        // Skip past the bits that the recursive calls will fill in.
        pos += (2 * w - 3) * step * (n / 2);

        // The last output layer: `l`, one bit per pair.
        for k in 0..n / 2 {
            let y = 2 * k;
            let lk = b[y] & 1;
            let ly = y as i32 + lk;
            let ly1 = ly ^ 1;

            out[pos >> 3] ^= (lk << (pos & 7)) as u8;
            pos += step;

            a[y] = (ly << 16) | (b[y] & 0xffff);
            a[y + 1] = (ly1 << 16) | (b[y + 1] & 0xffff);
        }
        sort_i32(a); // a = (id << 16) + M

        pos -= (2 * w - 2) * step * (n / 2);

        for j in 0..n / 2 {
            q[j] = ((a[2 * j] & 0xffff) >> 1) as i16;
            q[j + n / 2] = ((a[2 * j + 1] & 0xffff) >> 1) as i16;
        }
    }

    let (lower, upper) = q.split_at(n / 2);
    cbrecursion(out, pos, step * 2, lower, w - 1, n / 2, temp);
    cbrecursion(out, pos + step, step * 2, upper, w - 1, n / 2, temp);
}

/// Compute the `(2w-1)2^(w-1)` control bits realizing the permutation `pi` of `0..2^w`.
///
/// Returns whether the generated bits were verified to reproduce `pi`. A `false` result means
/// the input was not a permutation; for a genuine permutation the check always succeeds, so
/// callers can treat `false` as a key generation failure.
pub(crate) fn control_bits_from_permutation(out: &mut [u8], pi: &[i16], w: usize) -> bool {
    let n = 1usize << w;
    debug_assert_eq!(pi.len(), n);
    debug_assert_eq!(out.len(), ((2 * w - 1) * (n / 2)).div_ceil(8));

    out.fill(0);
    let mut temp = vec![0i32; 2 * n];
    cbrecursion(out, 0, 1, pi, w, n, &mut temp);

    // Replay the network over the identity and confirm it lands on `pi`.
    let mut probe: Vec<i16> = (0..n as i16).collect();
    let mut offset = 0;
    for s in (0..w).chain((0..w - 1).rev()) {
        layer_i16(&mut probe, out, offset, s);
        offset += n / 2;
    }

    let mut diff = 0i16;
    for (expected, actual) in pi.iter().zip(probe.iter()) {
        diff |= expected ^ actual;
    }
    diff == 0
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

    fn permutation(n: usize, seed: u64) -> Vec<i16> {
        let mut rng = Rng(seed);
        let mut pi: Vec<i16> = (0..n as i16).collect();
        for i in (1..n).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            pi.swap(i, j);
        }
        pi
    }

    fn check(w: usize, seed: u64) {
        let n = 1usize << w;
        let pi = permutation(n, seed);
        let mut out = vec![0u8; ((2 * w - 1) * (n / 2)).div_ceil(8)];
        assert!(control_bits_from_permutation(&mut out, &pi, w), "w = {w}");
    }

    #[test]
    fn control_bits_reproduce_the_permutation() {
        // The self-check inside `control_bits_from_permutation` is the assertion; running it
        // across widths covers both the `w <= 10` and `w > 10` code paths of the recursion.
        for w in 1..=11 {
            check(w, 0x1357_9BDF_2468_ACE0 ^ (w as u64));
        }
    }

    #[test]
    fn control_bits_reproduce_the_permutation_at_field_widths() {
        check(12, 0xCAFE_BABE_DEAD_BEEF);
        check(13, 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn the_identity_permutation_yields_all_zero_control_bits() {
        for w in 1..=8 {
            let n = 1usize << w;
            let pi: Vec<i16> = (0..n as i16).collect();
            let mut out = vec![0u8; ((2 * w - 1) * (n / 2)).div_ceil(8)];
            assert!(control_bits_from_permutation(&mut out, &pi, w));
            assert!(out.iter().all(|&b| b == 0), "w = {w}");
        }
    }

    #[test]
    fn a_non_permutation_is_rejected() {
        let w = 4;
        let n = 1usize << w;
        let mut pi: Vec<i16> = (0..n as i16).collect();
        pi[3] = 2; // now 2 appears twice and 3 never
        let mut out = vec![0u8; ((2 * w - 1) * (n / 2)).div_ceil(8)];
        assert!(!control_bits_from_permutation(&mut out, &pi, w));
    }

    #[test]
    fn a_single_swap_sets_exactly_one_bit() {
        // For w = 1 the network is one stage of one conditional swap.
        let mut out = vec![0u8; 1];
        assert!(control_bits_from_permutation(&mut out, &[1, 0], 1));
        assert_eq!(out[0] & 1, 1);

        assert!(control_bits_from_permutation(&mut out, &[0, 1], 1));
        assert_eq!(out[0] & 1, 0);
    }
}
