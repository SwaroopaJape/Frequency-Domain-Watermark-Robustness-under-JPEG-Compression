import numpy as np
from .base import Watermarker

# haar filter coefficient
# scaling / low-pass
_LO = np.array([1.0, 1.0]) / np.sqrt(2)   
# wavelet / high-pass
_HI = np.array([1.0, -1.0]) / np.sqrt(2)  

# haar 1d
def _haar_1d(signal: np.ndarray, filt: np.ndarray) -> np.ndarray:
    n = (len(signal) // 2) * 2         
    s = signal[:n]
    out = np.zeros(n // 2, dtype=np.float64)
    for k in range(n // 2):
        out[k] = filt[0] * s[2 * k] + filt[1] * s[2 * k + 1]
    return out

# inverse haar 1d
def _ihaar_1d(low: np.ndarray, filt_lo: np.ndarray, filt_hi: np.ndarray, high: np.ndarray) -> np.ndarray:
    n = len(low)
    out = np.zeros(2 * n, dtype=np.float64)
    for k in range(n):
        out[2 * k]     += filt_lo[0] * low[k] + filt_hi[0] * high[k]
        out[2 * k + 1] += filt_lo[1] * low[k] + filt_hi[1] * high[k]
    return out

# one level 2D haar dwt
def dwt2(image: np.ndarray) -> dict:
    img = image.astype(np.float64)
    # padding
    h, w = img.shape
    hp = h + (h % 2)
    wp = w + (w % 2)
    padded = np.zeros((hp, wp), dtype=np.float64)
    padded[:h, :w] = img

    # rows
    L_rows = np.apply_along_axis(_haar_1d, 1, padded, _LO)
    H_rows = np.apply_along_axis(_haar_1d, 1, padded, _HI)

    # columns
    LL = np.apply_along_axis(_haar_1d, 0, L_rows, _LO)
    LH = np.apply_along_axis(_haar_1d, 0, L_rows, _HI)
    HL = np.apply_along_axis(_haar_1d, 0, H_rows, _LO)
    HH = np.apply_along_axis(_haar_1d, 0, H_rows, _HI)

    return {"LL": LL, "LH": LH, "HL": HL, "HH": HH, "_shape": (h, w)}

# inverse 2d haar dwt
def idwt2(subbands: dict) -> np.ndarray:
    LL, LH, HL, HH = subbands["LL"], subbands["LH"], subbands["HL"], subbands["HH"]
    orig_h, orig_w = subbands["_shape"]

    # reconstruction rows
    def inv_cols(lo, hi):
        n_rows = lo.shape[0]
        n_cols = lo.shape[1]
        out = np.zeros((2 * n_rows, n_cols), dtype=np.float64)
        for c in range(n_cols):
            out[:, c] = _ihaar_1d(lo[:, c], _LO, _HI, hi[:, c])
        return out

    L_rows = inv_cols(LL, LH)
    H_rows = inv_cols(HL, HH)

    def inv_rows(lo, hi):
        n_rows = lo.shape[0]
        n_cols = lo.shape[1]
        out = np.zeros((n_rows, 2 * n_cols), dtype=np.float64)
        for r in range(n_rows):
            out[r, :] = _ihaar_1d(lo[r, :], _LO, _HI, hi[r, :])
        return out

    reconstructed = inv_rows(L_rows, H_rows)
    return reconstructed[:orig_h, :orig_w]


# derived from abstract class
class DWTWatermarker(Watermarker):

    SUBBANDS = ("LL", "LH", "HL", "HH")

    # override constructer
    def __init__(self, subband: str = "LH"):
        if subband not in self.SUBBANDS:
            raise ValueError(f"subband must be one of {self.SUBBANDS}")
        self.subband = subband

    # helper to calculate capacity
    def _capacity(self, image: np.ndarray) -> int:
        h, w = image.shape
        return (h // 2) * (w // 2)

    # override embed function
    def embed(self, image: np.ndarray, watermark: np.ndarray, alpha: float = 0.1,) -> np.ndarray:
        n_bits = len(watermark)
        cap = self._capacity(image)
        if n_bits > cap:
            raise ValueError(
                f"Watermark length {n_bits} exceeds DWT-{self.subband} "
                f"capacity {cap}."
            )

        subbands = dwt2(image)
        sb = subbands[self.subband].flatten()
        sb[:n_bits] += alpha * watermark
        subbands[self.subband] = sb.reshape(subbands[self.subband].shape)

        reconstructed = idwt2(subbands)
        return np.clip(reconstructed, 0, 255).astype(np.float32)

    # override extract function (non-blind)
    def extract(self, watermarked: np.ndarray, original: np.ndarray, n_bits: int, alpha: float = 0.1,) -> np.ndarray:
        sb_wm = dwt2(watermarked)[self.subband].flatten()
        sb_or = dwt2(original)[self.subband].flatten()
        extracted = (sb_wm[:n_bits] - sb_or[:n_bits]) / (alpha + 1e-12)
        return extracted.astype(np.float32)

    @property
    def name(self) -> str:
        return f"DWT-{self.subband}"
