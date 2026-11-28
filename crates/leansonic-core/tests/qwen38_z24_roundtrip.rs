use leansonic_core::qwen38_gpu::{decode_z24_sb_pair_ref, pack_z24_sb_slice};

#[inline]
fn bf16_to_f32_bits(b: u16) -> u32 {
    (b as u32) << 16
}

#[test]
fn z24_roundtrips_every_valid_15_exponent_window_and_signed_zeros_at_0ulp() {
    // Test across representative base exponents (including Qwen3.8 actual `[115..123]` and `[119..127]`).
    for base_exp in [1u8, 64, 115, 119, 240] {
        let mut sc_u16 = Vec::new();
        let mut bi_u16 = Vec::new();
        // Include `+0.0` (`0x0000`) and `-0.0` (`0x8000`)
        sc_u16.push(0x0000u16);
        sc_u16.push(0x8000u16);
        bi_u16.push(0x8000u16);
        bi_u16.push(0x0000u16);

        // Exhaustively include BOTH signs × ALL 15 exponents `[base_exp .. base_exp + 14]` × ALL 128 mantissas!
        for sign in [0u16, 1u16] {
            for de in 0..15u8 {
                let exp = (base_exp + de) as u16;
                for mant in 0..128u16 {
                    let bits = (sign << 15) | (exp << 7) | mant;
                    sc_u16.push(bits);
                    bi_u16.push(bits ^ 0x8000u16);
                }
            }
        }
        while sc_u16.len() % 4 != 0 {
            sc_u16.push(0x0000);
            bi_u16.push(0x0000);
        }

        let sc_bytes: Vec<u8> = sc_u16.iter().flat_map(|u| u.to_le_bytes()).collect();
        let bi_bytes: Vec<u8> = bi_u16.iter().flat_map(|u| u.to_le_bytes()).collect();
        let mut lo_u32 = vec![0u32; sc_u16.len() / 2];
        let mut hi_u32 = vec![0u32; sc_u16.len() / 4];

        let (base_sc_m1, base_bi_m1) =
            pack_z24_sb_slice(&sc_bytes, &bi_bytes, &mut lo_u32, &mut hi_u32)
                .expect("15-exponent window + signed zeros must pack losslessly");

        for i in 0..sc_u16.len() {
            let (sc_dec, bi_dec) =
                decode_z24_sb_pair_ref(&lo_u32, &hi_u32, i, base_sc_m1, base_bi_m1);
            assert_eq!(
                sc_dec.to_bits(),
                bf16_to_f32_bits(sc_u16[i]),
                "0-ULP mismatch on scale idx={i} base_exp={base_exp}"
            );
            assert_eq!(
                bi_dec.to_bits(),
                bf16_to_f32_bits(bi_u16[i]),
                "0-ULP mismatch on bias idx={i} base_exp={base_exp}"
            );
        }
    }
}

#[test]
fn z24_rejects_16_exponent_span_subnormals_and_nonfinite() {
    let mut lo = [0u32; 2];
    let mut hi = [0u32; 1];

    // 1. Span of 16 exponents (`100` and `115`: `115 - 100 = 15 > 14`) -> None
    let wide_sc: Vec<u8> = [100u16 << 7, 115u16 << 7, 100u16 << 7, 100u16 << 7]
        .iter()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let ok_bi: Vec<u8> = [120u16 << 7; 4]
        .iter()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    assert!(pack_z24_sb_slice(&wide_sc, &ok_bi, &mut lo, &mut hi).is_none());

    // 2. Subnormal (`exp = 0, mant = 1`) -> None
    let sub_sc: Vec<u8> = [0x0001u16, 120u16 << 7, 120u16 << 7, 120u16 << 7]
        .iter()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    assert!(pack_z24_sb_slice(&sub_sc, &ok_bi, &mut lo, &mut hi).is_none());

    // 3. Inf / NaN (`exp = 255`) -> None
    let inf_sc: Vec<u8> = [0x7F80u16, 120u16 << 7, 120u16 << 7, 120u16 << 7]
        .iter()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    assert!(pack_z24_sb_slice(&inf_sc, &ok_bi, &mut lo, &mut hi).is_none());
}
