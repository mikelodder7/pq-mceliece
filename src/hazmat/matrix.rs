/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
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

impl BitMatrix {
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
    #[inline]
    pub(crate) fn set_byte(&mut self, row: usize, index: usize, value: u8) {
        let word = row * self.stride + index / 8;
        let shift = (index % 8) * 8;
        self.data[word] = (self.data[word] & !(0xFFu64 << shift)) | ((value as u64) << shift);
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

    /// Overwrite the 64 bits of `row` starting at bit offset `bit`, leaving neighbours intact.
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
}
