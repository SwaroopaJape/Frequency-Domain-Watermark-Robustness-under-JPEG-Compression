import io
import numpy as np
from PIL import Image

# apply jpeg compression and decompression to grayscale float32 image, i.e. jpeg attack
def jpeg_attack(image: np.ndarray, quality: int = 50) -> np.ndarray:

    if quality < 1 or quality > 100:
        raise ValueError(f"JPEG quality must be 1–100, got {quality}.")

    uint8_img = np.clip(np.round(image), 0, 255).astype(np.uint8)

    if uint8_img.ndim == 2:
        pil_img = Image.fromarray(uint8_img, mode="L")
    elif uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
        pil_img = Image.fromarray(uint8_img, mode="RGB")
    else:
        raise ValueError(f"Unexpected image shape: {uint8_img.shape}")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)

    decoded = np.array(Image.open(buf), dtype=np.float32)
    return decoded

# for import convinience
def apply_attack(image: np.ndarray, qf: int) -> np.ndarray:
    return jpeg_attack(image, quality=qf)
