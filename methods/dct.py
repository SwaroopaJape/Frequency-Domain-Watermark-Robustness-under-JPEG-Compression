import numpy as np
from .base import Watermarker
from scipy.fft import dct, idct

# mid-frequency DCT positions within an 8×8 block (zigzag order indices 10-20)
_MID_FREQ_COORDS = [
    (1, 2), (2, 1), (3, 0), (2, 2), (1, 3),
    (0, 4), (1, 4), (2, 3), (3, 2), (4, 1),
    (5, 0), (4, 2), (3, 3),
]

# 2D DCT using scipy.fft
def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


# 2D iDCT using scipy.fft
def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")

# derived from abstraact class
class DCTWatermarker(Watermarker):

    # overide constructer
    def __init__(self, coords: list | None = None):
        self.coords = coords or _MID_FREQ_COORDS
        self.n_per_block = len(self.coords)

    # helper to calculate capacity
    def _block_capacity(self, image: np.ndarray) -> int:
        h, w = image.shape
        n_blocks_h = h // 8
        n_blocks_w = w // 8
        return n_blocks_h * n_blocks_w * self.n_per_block

    # override embed function 
    def embed(self, image: np.ndarray, watermark: np.ndarray,alpha: float = 0.1,) -> np.ndarray:
        h, w = image.shape
        n_bits = len(watermark)

        if n_bits > self._block_capacity(image):
            raise ValueError(
                f"Watermark length {n_bits} exceeds DCT capacity "
                f"{self._block_capacity(image)} for this image size."
            )

        result = image.copy().astype(np.float64)
        bit_idx = 0

        for i in range(0, h - 7, 8):
            if bit_idx >= n_bits:
                break
            for j in range(0, w - 7, 8):
                if bit_idx >= n_bits:
                    break
                block = result[i : i + 8, j : j + 8]
                dct_block = _dct2(block)
                for r, c in self.coords:
                    if bit_idx >= n_bits:
                        break
                    dct_block[r, c] += alpha * watermark[bit_idx]
                    bit_idx += 1
                result[i : i + 8, j : j + 8] = _idct2(dct_block)

        return np.clip(result, 0, 255).astype(np.float32)

    # override extract function 
    def extract(self, watermarked: np.ndarray,original: np.ndarray,n_bits: int,alpha: float = 0.1,) -> np.ndarray:
        
        h, w = watermarked.shape
        extracted = []

        for i in range(0, h - 7, 8):
            if len(extracted) >= n_bits:
                break
            for j in range(0, w - 7, 8):
                if len(extracted) >= n_bits:
                    break
                blk_wm = _dct2(watermarked[i : i + 8, j : j + 8].astype(np.float64))
                blk_or = _dct2(original[i : i + 8, j : j + 8].astype(np.float64))
                for r, c in self.coords:
                    if len(extracted) >= n_bits:
                        break
                    extracted.append((blk_wm[r, c] - blk_or[r, c]) / (alpha + 1e-12))

        return np.array(extracted[:n_bits], dtype=np.float32)

    @property
    def name(self) -> str:
        return "DCT"
