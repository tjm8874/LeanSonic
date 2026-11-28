pub fn mxscale_load_benchmark(
    conditioned: bool,
    codes: &[u8],
    scales: &[u8],
    count: usize,
) -> Result<Vec<u8>, String> {
    let matrix = if conditioned {
        ExpertMatrix::MxScale {
            bytes: codes,
            scales: ExpertScaleSource::RansV1(scales),
            count,
        }
    } else {
        ExpertMatrix::Packed {
            bytes: codes,
            count,
            coding: ExpertCoding::Byte,
        }
    };
    matrix.device_codebook()
}
