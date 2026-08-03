/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0 OR MIT
*/
//! ⚠️ Low-level "hazmat" Classic McEliece API.
//!
//! # ☢️️ WARNING: HAZARDOUS API ☢️
//!
//! This module exposes the parameter sets as types, so keys, ciphertexts and shared secrets
//! carry their parameter set in the type system rather than in a runtime tag. It is the same
//! algorithm the [`Algorithm`](crate::Algorithm) API runs; the difference is only where
//! mismatches are caught.
//!
//! The building blocks are:
//!
//! * [`Params`], the constants defining a parameter set, implemented by the eighteen marker
//!   types [`McEliece348864`] through [`McEliece8192128pcf`];
//! * [`Field`], the arithmetic of `F_q`, implemented by [`Gf12`] and [`Gf13`];
//! * [`Kem`], the key encapsulation mechanism itself, blanket implemented for every
//!   [`Params`];
//! * [`EncapsulationKey`], [`DecapsulationKey`], [`Ciphertext`] and [`SharedSecret`], the
//!   parameterized value types.
//!
//! Which [`Kem`] methods exist depends on the `keygen`, `encapsulate` and `decapsulate`
//! features.

// Field arithmetic is only reachable from key generation and decapsulation; encapsulation
// needs the field's size, which lives on `Params`, but never its arithmetic.
#[cfg(any(feature = "keygen", feature = "decapsulate"))]
mod field;
// `Hash` is the session-key function; key generation uses SHAKE directly for its PRG and
// never touches it.
#[cfg(any(feature = "encapsulate", feature = "decapsulate"))]
mod hash;
mod models;
mod params;

// Supporting machinery, compiled only for the operations that reach it.
#[cfg(feature = "decapsulate")]
mod benes;
#[cfg(feature = "keygen")]
mod controlbits;
#[cfg(any(feature = "keygen", feature = "decapsulate"))]
mod fft;
#[cfg(feature = "keygen")]
mod matrix;
#[cfg(feature = "keygen")]
mod poly;
// Runtime CPU feature detection and the vector kernels it selects between. Compiled only where
// something consumes it: the x86 bit-matrix elimination and the x86/AArch64 bit-sliced field
// multiplies; widen this as kernels for the other hot spots land.
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    any(feature = "keygen", feature = "decapsulate")
))]
mod simd;
#[cfg(any(feature = "keygen", feature = "encapsulate"))]
mod sort;
#[cfg(feature = "decapsulate")]
mod transpose;
#[cfg(any(feature = "keygen", feature = "decapsulate"))]
mod vec;

#[cfg(feature = "decapsulate")]
pub(crate) mod decap;
#[cfg(feature = "encapsulate")]
pub(crate) mod encap;
#[cfg(feature = "keygen")]
mod keygen;

#[cfg(all(
    test,
    feature = "keygen",
    feature = "encapsulate",
    feature = "decapsulate"
))]
mod kat;

// The field types are only reachable when this module is public and something needs them.
#[cfg(all(feature = "hazmat", any(feature = "keygen", feature = "decapsulate")))]
pub use field::*;
pub use models::*;
pub use params::*;
