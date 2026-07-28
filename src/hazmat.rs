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

mod benes;
mod controlbits;
mod decap;
mod encap;
mod field;
mod hash;
mod keygen;
mod matrix;
mod models;
mod params;
mod poly;
mod sort;
mod transpose;

#[cfg(test)]
mod kat;

// The field types are only reachable when this module is public.
#[cfg(feature = "hazmat")]
pub use field::*;
pub use models::*;
pub use params::*;
