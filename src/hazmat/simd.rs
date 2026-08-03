/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! Runtime CPU feature detection.
//!
//! The crate ships one scalar implementation of every kernel and, on some targets, one or more
//! vector implementations of the same kernel. On x86 which one runs is a runtime decision, so
//! a stock `cargo build --release` reaches the fast path without the caller having to set
//! `RUSTFLAGS`. On AArch64 the kernels use only baseline `neon`, so they are selected at
//! compile time and no detection exists at all.
//!
//! On x86, detection happens once and is cached in an atomic. The cached value is a *level*
//! rather than a set of flags: kernels are written for a whole instruction-set tier, and a
//! tier is only reported once every feature that tier's kernels use is present.
//!
//! # Constant time
//!
//! The selected level depends on the host CPU, never on key material, so branching on it leaks
//! nothing. Within a level, every kernel is data oblivious: secrets reach only the operands of
//! fixed arithmetic, never a branch, a loop bound, or an address.

/// The x86 vector kernels for the bit-matrix elimination.
///
/// All `core::arch` intrinsics in the crate live under this module, so the constant-time
/// argument for a kernel can be audited without reading the algorithm that calls it.
#[cfg(all(target_arch = "x86_64", feature = "keygen"))]
pub(crate) mod matrix_x86;

/// The instruction-set tier the current host supports.
///
/// Ordered by capability, so a kernel can test `level() >= Level::Avx2`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    /// No vector kernels; the portable path runs.
    #[default]
    Scalar = 0,
    /// 256-bit integer vectors.
    Avx2 = 1,
    /// 512-bit integer vectors, including `vpternlogq`.
    Avx512 = 2,
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod detect {
    use super::Level;
    use core::sync::atomic::{AtomicU8, Ordering};

    /// Sentinel meaning detection has not run yet. Real levels are stored as `level as u8 + 1`
    /// so that zero, the atomic's initial value, is unambiguous.
    const UNKNOWN: u8 = 0;

    static CACHED: AtomicU8 = AtomicU8::new(UNKNOWN);

    /// Read the environment override, if the platform has one to read.
    ///
    /// Setting `PQ_MCELIECE_DISABLE_SIMD` to any non-empty value forces the scalar row
    /// kernels and disables the fused multiply. The variable is read once, when the first
    /// operation runs, and the answer is cached for the life of the process. CI runs the
    /// test suite under both settings and the results must be identical.
    fn disabled_by_environment() -> bool {
        std::env::var_os("PQ_MCELIECE_DISABLE_SIMD").is_some_and(|value| !value.is_empty())
    }

    fn probe() -> Level {
        if disabled_by_environment() {
            return Level::Scalar;
        }

        // The 512-bit kernels use `vpternlogq` and 64-bit masked lanes, which need `avx512f`
        // for the core operations and `avx512vl`/`avx512bw` for the narrower element widths
        // the sorting comparators reach for. Requiring all three keeps one tier boundary
        // rather than three overlapping ones.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512bw")
        {
            return Level::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return Level::Avx2;
        }
        Level::Scalar
    }

    /// The instruction-set tier for this process, detected once and cached.
    #[inline]
    pub(crate) fn level() -> Level {
        let cached = CACHED.load(Ordering::Relaxed);
        if cached != UNKNOWN {
            return match cached - 1 {
                2 => Level::Avx512,
                1 => Level::Avx2,
                _ => Level::Scalar,
            };
        }

        let detected = probe();
        // A racing thread computes the same answer from the same CPU, so the last writer wins
        // harmlessly and no synchronization beyond the atomic itself is needed.
        CACHED.store(detected as u8 + 1, Ordering::Relaxed);
        detected
    }
}

/// The x86 form of the bit-sliced field multiply. Needs no detection: `sse2` is baseline.
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "keygen", feature = "decapsulate")
))]
pub(crate) mod vec_x86;

/// The AArch64 form of the bit-sliced field multiply. Needs no detection: `neon` is baseline.
#[cfg(all(
    target_arch = "aarch64",
    any(feature = "keygen", feature = "decapsulate")
))]
pub(crate) mod vec_neon;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) use detect::level;

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "x86")))]
mod tests {
    use super::*;

    #[test]
    fn detection_is_stable_across_calls() {
        let first = level();
        for _ in 0..1000 {
            assert_eq!(level(), first);
        }
    }

    #[test]
    fn levels_are_ordered_by_capability() {
        assert!(Level::Scalar < Level::Avx2);
        assert!(Level::Avx2 < Level::Avx512);
    }
}
