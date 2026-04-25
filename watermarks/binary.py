import numpy as np

# generate a 1d binary array of +/- 1 values (length n_bits)
def generate(n_bits: int = 1024, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    return (2 * bits - 1).astype(np.float32)


# reshape flat watermark to 2-D image patch (crop / pad as needed)
def to_image(w: np.ndarray, shape: tuple) -> np.ndarray:
    h, c = shape
    total = h * c
    if w.size >= total:
        return w[:total].reshape(h, c)
    padded = np.zeros(total, dtype=w.dtype)
    padded[: w.size] = w
    return padded.reshape(h, c)
