/*
    Copyright Michael Lodder. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
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

// Supporting machinery, compiled only for the operations that reach it. Encapsulation needs
// none of it: `Encode` is pure bit manipulation over the public key.
#[cfg(feature = "decapsulate")]
mod benes;
#[cfg(feature = "keygen")]
mod controlbits;
#[cfg(feature = "decapsulate")]
mod fft;
#[cfg(feature = "keygen")]
mod matrix;
#[cfg(feature = "keygen")]
mod poly;
#[cfg(feature = "keygen")]
mod sort;
#[cfg(feature = "decapsulate")]
mod transpose;

#[cfg(feature = "decapsulate")]
mod decap;
#[cfg(feature = "encapsulate")]
mod encap;
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
