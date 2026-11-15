//! `PDF` stream predictors (`/DecodeParms`) applied after a filter.
//!
//! A `FlateDecode` (or any other) stream may be pre-processed by a predictor that
//! stores each row as a delta against its predecessor: predictor `2` is the `TIFF`
//! horizontal differencing scheme, `10`..`15` are the PNG row filters, and `1`
//! (the default) is a passthrough. Predictors are not compression — they run after
//! the filter chain, which is why [`apply`] takes already-decoded bytes.
//!
//! - [`Error::Unsupported`] — a predictor value, bit depth, or column/colour count
//!   this crate does not implement (reported at document level by the parser).
//! - [`Error::Retrieval`] — the decoded bytes do not end on a row boundary, or a
//!   PNG row names an unknown filter type.

use openjiuwen_common::error::{Error, Result};

/// A `/DecodeParms` dictionary, reduced to the predictor fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PredictorParams {
    /// `/Predictor` (default `1`, no predictor).
    pub predictor: u32,
    /// `/Columns`: samples per row (default `1`).
    pub columns: usize,
    /// `/Colors`: interleaved colour components (default `1`).
    pub colors: usize,
    /// `/BitsPerComponent` (default `8`).
    pub bits_per_component: usize,
}

impl Default for PredictorParams {
    fn default() -> Self {
        Self {
            predictor: 1,
            columns: 1,
            colors: 1,
            bits_per_component: 8,
        }
    }
}

impl PredictorParams {
    /// Returns `true` when the predictor leaves the bytes untouched.
    pub(crate) fn is_identity(&self) -> bool {
        self.predictor <= 1
    }

    /// Rejects a predictor this crate cannot apply.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] naming the predictor and the offending parameter.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.is_identity() {
            return Ok(());
        }
        if self.columns == 0 || self.colors == 0 {
            return Err(Error::Unsupported(format!(
                "PDF predictor {} with /Columns {} and /Colors {} has no samples to reconstruct",
                self.predictor, self.columns, self.colors
            )));
        }
        match self.predictor {
            2 => {
                if matches!(self.bits_per_component, 1 | 2 | 4 | 8 | 16) {
                    Ok(())
                } else {
                    Err(Error::Unsupported(format!(
                        "PDF TIFF predictor 2 with /BitsPerComponent {} is not supported",
                        self.bits_per_component
                    )))
                }
            }
            10..=15 => {
                if self.bits_per_component == 8 {
                    Ok(())
                } else {
                    Err(Error::Unsupported(format!(
                        "PDF PNG predictor {} with /BitsPerComponent {} is not supported",
                        self.predictor, self.bits_per_component
                    )))
                }
            }
            other => Err(Error::Unsupported(format!(
                "PDF predictor {other} is not supported"
            ))),
        }
    }
}

/// Undoes a predictor on already-filtered `data`, bounded by `max_output` bytes.
///
/// # Errors
///
/// [`PredictorParams::validate`] errors, [`Error::Retrieval`] for data that ends
/// mid-row or names an unknown PNG row filter, and [`Error::Validation`] when the
/// reconstructed bytes would exceed `max_output`.
pub(crate) fn apply(data: &[u8], params: &PredictorParams, max_output: usize) -> Result<Vec<u8>> {
    params.validate()?;
    if params.is_identity() {
        if data.len() > max_output {
            return Err(Error::Validation(format!(
                "PDF stream exceeds the {max_output}-byte output limit"
            )));
        }
        return Ok(data.to_vec());
    }
    match params.predictor {
        2 => apply_tiff(data, params, max_output),
        _ => apply_png(data, params, max_output),
    }
}

/// Undoes the `TIFF` horizontal-differencing predictor (`2`).
fn apply_tiff(data: &[u8], params: &PredictorParams, max_output: usize) -> Result<Vec<u8>> {
    let bits = params.bits_per_component;
    let samples_per_row = row_geometry(params)?;
    let row_bytes = samples_per_row.saturating_mul(bits).div_ceil(8);
    if row_bytes == 0 {
        return Ok(Vec::new());
    }
    if data.len() % row_bytes != 0 {
        return Err(Error::Retrieval(
            "TIFF predictor data ends in the middle of a row".to_owned(),
        ));
    }
    if data.len() > max_output {
        return Err(Error::Validation(format!(
            "PDF stream exceeds the {max_output}-byte output limit"
        )));
    }
    let mut out = Vec::with_capacity(data.len());
    for row in data.chunks_exact(row_bytes) {
        let mut samples = unpack_samples(row, samples_per_row, bits);
        for index in params.colors..samples.len() {
            let modulus = sample_modulus(bits);
            samples[index] = (samples[index] + samples[index - params.colors]) % modulus;
        }
        pack_samples(&samples, row_bytes, bits, &mut out);
    }
    Ok(out)
}

/// Undoes the PNG row predictors (`10`..`15`): each row starts with its type.
fn apply_png(data: &[u8], params: &PredictorParams, max_output: usize) -> Result<Vec<u8>> {
    let row_bytes = row_geometry(params)?;
    let bpp = params.colors;
    let mut out = Vec::with_capacity(data.len());
    let mut previous = vec![0u8; row_bytes];
    let mut cursor = 0usize;
    while cursor < data.len() {
        let filter_type = data[cursor];
        cursor += 1;
        let Some(row) = data.get(cursor..cursor + row_bytes) else {
            return Err(Error::Retrieval(
                "PNG predictor data ends in the middle of a row".to_owned(),
            ));
        };
        cursor += row_bytes;
        let mut current = row.to_vec();
        match filter_type {
            // None.
            0 => {}
            // Sub: the byte `bpp` positions to the left.
            1 => {
                for index in bpp..row_bytes {
                    current[index] = current[index].wrapping_add(current[index - bpp]);
                }
            }
            // Up: the byte above.
            2 => {
                for index in 0..row_bytes {
                    current[index] = current[index].wrapping_add(previous[index]);
                }
            }
            // Average: half of left plus above.
            3 => {
                for index in 0..row_bytes {
                    let left = if index >= bpp {
                        u16::from(current[index - bpp])
                    } else {
                        0
                    };
                    let above = u16::from(previous[index]);
                    current[index] = current[index].wrapping_add(u16::midpoint(left, above) as u8);
                }
            }
            // Paeth: the closest of left, above, and above-left.
            4 => {
                for index in 0..row_bytes {
                    let left = if index >= bpp {
                        current[index - bpp]
                    } else {
                        0
                    };
                    let above = previous[index];
                    let upper_left = if index >= bpp {
                        previous[index - bpp]
                    } else {
                        0
                    };
                    current[index] = current[index].wrapping_add(paeth(left, above, upper_left));
                }
            }
            other => {
                return Err(Error::Retrieval(format!(
                    "unknown PNG predictor row filter {other}"
                )));
            }
        }
        let new_len = out.len() + row_bytes;
        if new_len > max_output {
            return Err(Error::Validation(format!(
                "PDF stream exceeds the {max_output}-byte output limit"
            )));
        }
        out.extend_from_slice(&current);
        previous = current;
    }
    Ok(out)
}

/// Returns the number of samples in one row (`/Columns` × `/Colors`).
fn row_geometry(params: &PredictorParams) -> Result<usize> {
    params.columns.checked_mul(params.colors).ok_or_else(|| {
        Error::Unsupported(format!(
            "PDF predictor {} with /Columns {} and /Colors {} describes an oversized row",
            params.predictor, params.columns, params.colors
        ))
    })
}

/// Returns the PNG Paeth predictor of `left`, `above`, and `upper_left`.
fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let left = i32::from(left);
    let above = i32::from(above);
    let upper_left = i32::from(upper_left);
    let estimate = left + above - upper_left;
    let distance_left = (estimate - left).abs();
    let distance_above = (estimate - above).abs();
    let distance_upper_left = (estimate - upper_left).abs();
    if distance_left <= distance_above && distance_left <= distance_upper_left {
        left as u8
    } else if distance_above <= distance_upper_left {
        above as u8
    } else {
        upper_left as u8
    }
}

/// Returns `2^bits`, the modulus one sample is reconstructed in.
///
/// [`PredictorParams::validate`] restricts `bits` to `1`, `2`, `4`, `8`, or `16`,
/// so the shift cannot overflow.
fn sample_modulus(bits: usize) -> usize {
    1usize << bits
}

/// Unpacks `count` big-endian, MSB-first samples of `bits` width from `row`.
fn unpack_samples(row: &[u8], count: usize, bits: usize) -> Vec<usize> {
    match bits {
        8 => row
            .iter()
            .take(count)
            .map(|&byte| usize::from(byte))
            .collect(),
        16 => row
            .chunks_exact(2)
            .take(count)
            .map(|pair| usize::from(u16::from_be_bytes([pair[0], pair[1]])))
            .collect(),
        _ => {
            let mask = (1usize << bits) - 1;
            let mut samples = Vec::with_capacity(count);
            let mut bit = 0usize;
            for _ in 0..count {
                let mut value = 0usize;
                for _ in 0..bits {
                    let byte = row.get(bit / 8).copied().unwrap_or(0);
                    let shift = 7 - (bit % 8);
                    value = (value << 1) | usize::from((byte >> shift) & 1);
                    bit += 1;
                }
                samples.push(value & mask);
            }
            samples
        }
    }
}

/// Packs reconstructed samples back into big-endian, MSB-first row bytes.
fn pack_samples(samples: &[usize], row_bytes: usize, bits: usize, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + row_bytes, 0);
    let row = &mut out[start..];
    match bits {
        8 => {
            for (slot, &sample) in row.iter_mut().zip(samples.iter()) {
                *slot = sample as u8;
            }
        }
        16 => {
            for (pair, &sample) in row.chunks_exact_mut(2).zip(samples.iter()) {
                pair.copy_from_slice(&(sample as u16).to_be_bytes());
            }
        }
        _ => {
            let mask = (1usize << bits) - 1;
            let mut bit = 0usize;
            for &sample in samples {
                let value = sample & mask;
                for shift in (0..bits).rev() {
                    if (value >> shift) & 1 == 1 {
                        row[bit / 8] |= 1 << (7 - (bit % 8));
                    }
                    bit += 1;
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn defaults_are_the_pdf_defaults_and_identity() {
        let params = PredictorParams::default();
        assert_eq!(params.predictor, 1);
        assert_eq!(params.columns, 1);
        assert_eq!(params.colors, 1);
        assert_eq!(params.bits_per_component, 8);
        assert!(params.is_identity());
        assert!(params.validate().is_ok());
    }

    #[test]
    fn predictor_one_is_a_passthrough() {
        let params = PredictorParams::default();
        assert_eq!(
            apply(b"BT (text) Tj ET", &params, 1024).unwrap(),
            b"BT (text) Tj ET"
        );
        let error = apply(b"0123456789", &params, 4).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn png_sub_rows_reconstruct_the_first_row_both_ways() {
        let params = PredictorParams {
            predictor: 12,
            columns: 4,
            colors: 1,
            bits_per_component: 8,
        };
        // Row 0 with PNG filter type 2 (Up) is stored verbatim.
        assert_eq!(
            apply(&[2, b'a', b'b', b'c', b'd'], &params, 64).unwrap(),
            b"abcd"
        );
        // Two Up rows: the second adds the first.
        let encoded = [2, 1, 2, 3, 4, 2, 1, 1, 1, 1];
        assert_eq!(
            apply(&encoded, &params, 64).unwrap(),
            vec![1, 2, 3, 4, 2, 3, 4, 5]
        );
        // Sub (1) adds the byte `bpp` positions to the left.
        let encoded = [1, 1, 1, 1, 1];
        assert_eq!(apply(&encoded, &params, 64).unwrap(), vec![1, 2, 3, 4]);
        // None (0) keeps the row as stored.
        assert_eq!(
            apply(&[0, 9, 8, 7, 6], &params, 64).unwrap(),
            vec![9, 8, 7, 6]
        );
    }

    #[test]
    fn png_average_and_paeth_rows_reconstruct() {
        let params = PredictorParams {
            predictor: 11,
            columns: 3,
            colors: 1,
            bits_per_component: 8,
        };
        // Average over a zero previous row halves the left byte.
        let encoded = [3, 0, 10, 10];
        assert_eq!(apply(&encoded, &params, 64).unwrap(), vec![0, 10, 15]);
        // Paeth over a zero previous row picks the left byte.
        let encoded = [4, 0, 5, 5];
        assert_eq!(apply(&encoded, &params, 64).unwrap(), vec![0, 5, 10]);
        // Paeth picks `above` when the zero row makes it the closest value.
        let mut encoded = vec![0, 7, 7, 7];
        encoded.extend_from_slice(&[4, 1, 1, 1]);
        let decoded = apply(&encoded, &params, 64).unwrap();
        assert_eq!(decoded, vec![7, 7, 7, 8, 9, 10]);
    }

    #[test]
    fn tiff_predictor_reconstructs_8_bit_rows() {
        let params = PredictorParams {
            predictor: 2,
            columns: 3,
            colors: 1,
            bits_per_component: 8,
        };
        // Row 0: 10, 1, 1 -> 10, 11, 12. Row 1 resets to its own first sample.
        let decoded = apply(&[10, 1, 1, 20, 1, 1], &params, 64).unwrap();
        assert_eq!(decoded, vec![10, 11, 12, 20, 21, 22]);
    }

    #[test]
    fn tiff_predictor_reconstructs_16_bit_rows() {
        let params = PredictorParams {
            predictor: 2,
            columns: 2,
            colors: 1,
            bits_per_component: 16,
        };
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&100u16.to_be_bytes());
        encoded.extend_from_slice(&5u16.to_be_bytes());
        let decoded = apply(&encoded, &params, 64).unwrap();
        assert_eq!(decoded, vec![0, 100, 0, 105]);
    }

    #[test]
    fn tiff_predictor_reconstructs_4_bit_rows() {
        let params = PredictorParams {
            predictor: 2,
            columns: 4,
            colors: 1,
            bits_per_component: 4,
        };
        // Samples 1, 1, 1, 1 in one byte -> 1, 2, 3, 4.
        assert_eq!(apply(&[0x11, 0x11], &params, 64).unwrap(), vec![0x12, 0x34]);
        // The 4-bit addition wraps inside the nibble.
        assert_eq!(apply(&[0xF1, 0x10], &params, 64).unwrap(), vec![0xF0, 0x11]);
    }

    #[test]
    fn tiff_colour_components_differ_independently() {
        let params = PredictorParams {
            predictor: 2,
            columns: 2,
            colors: 2,
            bits_per_component: 8,
        };
        // Pairs (10, 20), (1, 2) -> (10, 20), (11, 22).
        assert_eq!(
            apply(&[10, 20, 1, 2], &params, 64).unwrap(),
            vec![10, 20, 11, 22]
        );
    }

    #[test]
    fn unsupported_predictors_and_bit_depths_are_reported() {
        let png_with_16_bits = PredictorParams {
            predictor: 12,
            columns: 8,
            colors: 1,
            bits_per_component: 16,
        };
        let error = png_with_16_bits.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("predictor 12"), "{error}");
        assert!(error.to_string().contains("BitsPerComponent 16"), "{error}");
        assert_eq!(
            apply(&[0], &png_with_16_bits, 64).unwrap_err().kind(),
            ErrorKind::Unsupported
        );

        let unknown = PredictorParams {
            predictor: 3,
            columns: 1,
            colors: 1,
            bits_per_component: 8,
        };
        let error = unknown.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("predictor 3"), "{error}");

        let twelve_bit_tiff = PredictorParams {
            predictor: 2,
            columns: 1,
            colors: 1,
            bits_per_component: 12,
        };
        let error = twelve_bit_tiff.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("BitsPerComponent 12"), "{error}");

        let no_columns = PredictorParams {
            predictor: 2,
            columns: 0,
            colors: 1,
            bits_per_component: 8,
        };
        let error = no_columns.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("no samples"), "{error}");
    }

    #[test]
    fn truncated_and_unknown_rows_are_retrieval_errors() {
        let params = PredictorParams {
            predictor: 11,
            columns: 4,
            colors: 1,
            bits_per_component: 8,
        };
        let error = apply(&[2, b'a', b'b'], &params, 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.to_string().contains("middle of a row"), "{error}");

        let error = apply(&[2, b'a', b'b', b'c', b'd', 9, 1, 2, 3, 4], &params, 64).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.to_string().contains("row filter 9"), "{error}");

        let tiff = PredictorParams {
            predictor: 2,
            columns: 4,
            colors: 1,
            bits_per_component: 8,
        };
        assert_eq!(
            apply(&[1, 2, 3], &tiff, 64).unwrap_err().kind(),
            ErrorKind::Retrieval
        );
    }

    #[test]
    fn output_is_bounded() {
        let params = PredictorParams {
            predictor: 11,
            columns: 4,
            colors: 1,
            bits_per_component: 8,
        };
        let error = apply(&[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], &params, 7).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let tiff = PredictorParams {
            predictor: 2,
            columns: 4,
            colors: 1,
            bits_per_component: 8,
        };
        assert_eq!(
            apply(&[1, 2, 3, 4], &tiff, 2).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
}
