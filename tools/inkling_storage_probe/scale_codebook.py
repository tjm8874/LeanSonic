"""Refine a finite E4M3 codebook using the existing exponent of each 32-weight block."""
import numpy as np

from choose_codebooks import count_weights, magnitude


def fit(counts, magnitudes):
    counts = np.asarray(counts, dtype=np.int64)
    active = np.flatnonzero(counts.sum(axis=1))
    freq = counts[active]
    levels = np.array([magnitude(i & 127) for i in range(256)])
    baseline = np.array([
        (i & 128) | magnitudes[i & 127] if (i & 127) < 127 and magnitudes[i & 127] else 0
        for i in range(256)
    ], dtype=np.uint8)
    output_freq = np.bincount(baseline, weights=freq.sum(axis=0), minlength=256)
    centers = np.flatnonzero(output_freq)
    # Model the same signed byte stream that the four-lane rANS frame encodes.
    bits = -np.log2(output_freq[centers] / freq.sum())
    base_bits = np.dot(output_freq[centers], bits)
    square_scale = np.exp2(2.0 * (active - 127))
    power = np.sum(square_scale[:, None] * freq * levels**2)
    if power <= 0:
        raise ValueError("plane has no signal")
    distance = square_scale[:, None, None] * (levels[None, :, None] - levels[centers])**2
    # One unit of rate is one bit per source weight. One unit of distortion is full signal power.
    distance *= freq.sum() / power
    source = np.arange(256)[:, None]
    allowed = ((source & 128) == (centers & 128)) | (centers == 0)
    allowed &= ((source & 127) != 0) | (centers == 0)
    distance[:, ~allowed] = np.inf
    if not allowed.any(axis=1).all():
        raise ValueError("codebook lacks a compatible finite reconstruction")

    def solve(penalty):
        idx = (distance + penalty * bits).argmin(axis=2)
        return centers[idx], np.sum(freq * bits[idx])

    high = 1.0
    while solve(high)[1] > base_bits + 0.01:
        high *= 2.0
        if high > 1e12:
            raise ValueError("could not meet baseline cross-entropy budget")
    low = 0.0
    for _ in range(42):
        mid = (low + high) / 2
        if solve(mid)[1] <= base_bits + 0.01:
            high = mid
        else:
            low = mid
    table, fitted_bits = solve(high)
    def error(codes):
        return float(np.sum(square_scale[:,None] * freq * (levels - levels[codes])**2) / power)
    base_error = error(baseline)
    if error(table) > base_error:
        table = np.broadcast_to(baseline, freq.shape)
        fitted_bits = base_bits
    lut = np.zeros((256,256), dtype=np.uint8)
    lut[active] = table
    return lut, dict(original_relative_mse=base_error, fitted_relative_mse=error(table),
                     original_entropy_bits=float(base_bits), fitted_cross_entropy_bits=float(fitted_bits),
                     changed_codes=int(np.sum(freq * (table != np.arange(256)))),
                     codes=int(freq.sum()))


def transform(shard, meta, start, key, magnitudes, output):
    counts = count_weights(shard, meta, start, key)
    lut, receipt = fit(counts, magnitudes)
    lo, hi = meta[key]["data_offsets"]
    first, last = meta[key + "_scale"]["data_offsets"]
    observed = np.zeros(256, dtype=np.int64)
    with shard.open("rb") as codes, shard.open("rb") as scales, output.open("xb") as out:
        codes.seek(start+lo)
        scales.seek(start+first)
        for begin in range(0, last-first, 1<<16):
            n = min(1<<16, last-first-begin)
            exponent = np.frombuffer(scales.read(n),dtype=np.uint8).reshape(n,1)
            src = np.frombuffer(codes.read(n*32),dtype=np.uint8).reshape(n,32)
            encoded = lut[exponent,src]
            out.write(encoded.tobytes())
            observed += np.bincount(encoded.ravel(),minlength=256)
    assert output.stat().st_size == hi-lo
    expected = np.bincount(lut.ravel(),weights=counts.ravel(),minlength=256)
    np.testing.assert_array_equal(observed, expected)
    return receipt
