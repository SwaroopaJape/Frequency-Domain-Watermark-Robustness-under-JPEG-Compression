import numpy as np
from pathlib import Path
import cv2

# create a simple cross / checkerboard synthetic logo
def _synthetic_logo(size: int = 32) -> np.ndarray:
    logo = np.zeros((size, size), dtype=np.float32)
    mid = size // 2
    logo[mid - 2 : mid + 2, :] = 1.0
    logo[:, mid - 2 : mid + 2] = 1.0
    logo[:4, :4] = 1.0
    logo[-4:, :4] = 1.0
    logo[:4, -4:] = 1.0
    logo[-4:, -4:] = 1.0
    return logo

# generate a 32x32 float32 array with values in +/- 1
def generate(size: int = 32, path: str | None = None) -> np.ndarray:
    if path and Path(path).exists():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size))
        logo = (img > img.mean()).astype(np.float32)
    else:
        logo = _synthetic_logo(size)

    logo = np.where(logo > 0.5, 1.0, -1.0).astype(np.float32)
    return logo


def flatten(logo: np.ndarray) -> np.ndarray:
    return logo.flatten()
