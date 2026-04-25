import numpy as np
from scipy.ndimage import uniform_filter

# image quality analysis
def psnr(original: np.ndarray, watermarked: np.ndarray, max_val: float = 255.0) -> float:
    # peak Signal-to-Noise Ratio
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def ssim(original: np.ndarray, watermarked: np.ndarray, max_val: float = 255.0, win_size: int = 7,) -> float:
    # structural Similarity Index

    img1 = original.astype(np.float64)
    img2 = watermarked.astype(np.float64)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = uniform_filter(img1, size=win_size)
    mu2 = uniform_filter(img2, size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size) - mu2_sq
    sigma12   = uniform_filter(img1 * img2, size=win_size) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(np.mean(ssim_map))


# watermark quality analysis
def ncc(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    # normalized Cross-Correlation
    
    w1 = original_wm.astype(np.float64).flatten()
    w2 = extracted_wm.astype(np.float64).flatten()

    n = min(len(w1), len(w2))
    w1, w2 = w1[:n], w2[:n]

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return float(np.dot(w1, w2) / (norm1 * norm2))


def ber(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    # Bit Error Rate
    w1 = (original_wm.flatten() >= 0).astype(np.uint8)
    w2 = (extracted_wm.flatten() >= 0).astype(np.uint8)

    n = min(len(w1), len(w2))
    if n == 0:
        return 0.5
    errors = np.sum(w1[:n] != w2[:n])
    return float(errors / n)


# convenience wrapper
def all_metrics(original_img: np.ndarray, watermarked_img: np.ndarray, original_wm: np.ndarray, extracted_wm: np.ndarray,) -> dict:
    return {
        "PSNR": psnr(original_img, watermarked_img),
        "SSIM": ssim(original_img, watermarked_img),
        "NCC":  ncc(original_wm, extracted_wm),
        "BER":  ber(original_wm, extracted_wm),
    }
