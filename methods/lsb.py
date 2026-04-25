import numpy as np
from .base import Watermarker

# derived from abstract class
class LSBWatermarker(Watermarker):

    def embed(self, image: np.ndarray, watermark: np.ndarray, alpha: float = 0.1,) -> np.ndarray:
    
        img_uint8 = np.clip(np.round(image), 0, 255).astype(np.uint8)
        flat = img_uint8.flatten()

        n_bits = len(watermark)
        if n_bits > flat.size:
            raise ValueError(
                f"Watermark length {n_bits} exceeds image capacity {flat.size}."
            )

        bits = ((watermark + 1) / 2).astype(np.uint8)

        flat[:n_bits] = (flat[:n_bits] & 0xFE) | bits

        watermarked = flat.reshape(img_uint8.shape).astype(np.float32)
        return watermarked

    def extract(self, watermarked: np.ndarray, original: np.ndarray, n_bits: int, alpha: float = 0.1,) -> np.ndarray:
        wm_uint8 = np.clip(np.round(watermarked), 0, 255).astype(np.uint8).flatten()
        or_uint8 = np.clip(np.round(original), 0, 255).astype(np.uint8).flatten()

        wm_lsb = (wm_uint8[:n_bits] & 1).astype(np.float32)

        extracted = 2.0 * wm_lsb - 1.0
        return extracted

    @property
    def name(self) -> str:
        return "LSB"
