#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unstable_features,
        unused_import_braces, unused_qualifications)]

//! This crate implements common Asymmetric numeral systems coding algorithms
//!
//!
//!
//! # Quickstart
//! ## Encoding
//! ```ignore
//! use rans::ANSCoder;
//! use std::fs::File;
//! use std::io::Read;
//!
//! // Read test file
//! let mut f = File::open("book1").unwrap();
//! f.read_to_end(&mut data).unwrap();
//!
//! // Compute probablities for every token in the document
//! let mut probs = vec![0; 256];
//! for c in data.iter() {
//!     probs[*c as usize] += 1;
//! }
//! let mut ans = ANSCoder::new_static(&probs);
//!
//! for symbol in data.iter() {
//!   ans.encode_symbol(*symbol);
//! }
//! let encoded = ans.get_encoded();
//! ```
//! ## Decoding
//!
//! ```ignore
//! use rans::ANSDecoder;
//! // Construct decoder with the same stats object as the encoder
//! let mut decoder = ANSDecoder::new(encoded);
//! decoder.stats = ans.stats;
//!
//! let mut decoded_data = vec![];
//! let length_decoded = data.len();
//! for _ in 0..length_decoded {
//!   decoded_data.push(decoder.decode_symbol().unwrap())
//! }
//! decoded_data = decoded_data.into_iter().rev().collect();
//! ```



/// A good module
mod coder;

pub use crate::coder::{ANSCoder, ANSDecoder};
