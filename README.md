# Frequency-Domain Watermark Robustness under JPEG Compression

**CS611 — Digital Image Processing with Python | IIT Goa**

A framework for measuring how well spatial and frequency-domain digital watermarks survive JPEG compression at varying quality factors (QF). Three embedding methods (LSB, DCT, DWT), three watermark types (binary, logo, text), and six embedding strengths (α) are swept across eight JPEG quality levels, with results measured via PSNR, SSIM, NCC, and BER.

---

## Project Structure

```
├── watermarks/
│   ├── binary.py       # Random ±1 bit pattern (seed-controlled)
│   ├── logo.py         # 32×32 synthetic or custom image watermark
│   └── text.py         # UTF-8 string → ±1 bit array (with decoder)
├── methods/
│   ├── base.py         # Abstract Watermarker: embed / extract interface
│   ├── lsb.py          # Spatial LSB — baseline / lower bound
│   ├── dct.py          # 8×8 block DCT, mid-frequency embedding
│   └── dwt.py          # 2-D Haar DWT, all four subbands (LL/LH/HL/HH)
├── metrics.py          # PSNR, SSIM, NCC, BER
├── attack.py           # In-memory JPEG encode→decode at a given QF
├── analyze.py          # Batch sweep → results.csv
├── plot.py             # Heatmaps and line plots from results.csv
└── generate_results.py # Downloads skimage test images, runs analyze.py
```

---

## Quickstart

**Install dependencies**
```bash
pip install numpy scipy scikit-image pillow matplotlib opencv-python
# or, if using uv:
uv sync
```

**Run the sweep on standard test images**
```bash
python generate_results.py
```
This pulls `camera`, `coins`, `moon`, `brick`, and `astronaut` from `skimage.data`, saves them temporarily, and runs the full sweep via `analyze.py`. Progress is printed live. Output: `results.csv`.

**Run on your own images**
```bash
python analyze.py --image path/to/image.png --out results.csv
```
Images can be colour — they are converted to grayscale internally.

**Generate plots**
```bash
python plot.py 
```
Produces comparitive plots for each method (LSB, DCT, DWT-LL, DWT-HL, DWT-HH)

## Methods

### LSB (Spatial domain)
Replaces the least significant bit of the first *N* pixels with watermark bits. Imperceptible (PSNR ≈ 59 dB) but completely destroyed by any JPEG compression, since JPEG does not preserve exact pixel values. Serves as the baseline / lower bound.

### DCT (Frequency domain)
Divides the image into non-overlapping 8×8 blocks (same partition as JPEG) and additively embeds watermark bits into 13 mid-frequency DCT coefficients per block.

```
F'[u,v] = F[u,v] + α · wᵢ
```

Mid-frequencies are chosen because low frequencies (DC) produce visible artifacts and high frequencies are zeroed by JPEG quantisation. This method is moderately robust.

### DWT (Frequency domain)
Applies a one-level 2-D Haar wavelet transform, embedding additively into a chosen subband:

| Subband | Content | Robustness | Imperceptibility |
|---------|---------|------------|-----------------|
| LL | Approximation (smooth) | ✅✅ High | ❌ Lower |
| **LH** | **Horizontal edges** | **✅✅ Good** | **✅✅ Good** |
| HL | Vertical edges | ✅ Moderate | ✅ Good |
| HH | Diagonal / noise | ❌ Fragile | ✅✅ Very high |

**LH is recommended** — best robustness/imperceptibility trade-off.

Extraction is non-blind in all methods (original image is available):
```
W' = (Iw − I) / α
```

---

## Sweep Parameters

| Parameter | Values |
|-----------|--------|
| Embedding strength α | 0.01, 0.05, 0.1, 0.3, 0.5, 1.0 |
| JPEG quality factor | 5, 10, 20, 30, 50, 70, 90, 100 |
| Watermark types | binary, logo, text |
| Methods | LSB, DCT, DWT-LL, DWT-LH, DWT-HL, DWT-HH |

---

## Metrics

| Metric | Measures | Perfect value |
|--------|----------|---------------|
| **PSNR** | Image quality after embedding | ∞ (higher = better, ≥40 dB = imperceptible) |
| **SSIM** | Structural similarity after embedding | 1.0 |
| **NCC** | Watermark recovery accuracy | 1.0 |
| **BER** | Fraction of bits incorrectly recovered | 0.0 (0.5 = random / no information) |

---

## Key Finding

Frequency-domain methods (DCT, DWT) survive JPEG compression significantly better than spatial LSB because JPEG itself operates in the DCT domain — its quantisation table is designed to preserve low and mid frequencies. At α=1.0 and QF=90, DCT achieves NCC≈0.88 while LSB collapses to NCC≈0 at any QF below 100.

---

## Dependencies

- `numpy`, `scipy` — core numerics and DWT (Haar, pure NumPy — no PyWavelets required)
- `scikit-image` — test images and SSIM reference
- `pillow` — in-memory JPEG encode/decode
- `opencv-python` — image I/O
- `matplotlib` — plots and heatmaps
