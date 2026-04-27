# Frequency-Domain Watermark Robustness under JPEG Compression

**CS611 — Digital Image Processing with Python | IIT Goa**

A framework for measuring how well spatial and frequency-domain digital watermarks survive JPEG compression at varying quality factors (QF). Three embedding methods (LSB, DCT, DWT), three watermark types (binary, logo, text), and six embedding strengths (α) are swept across eight JPEG quality levels, with results measured via PSNR, SSIM, NCC, and BER.

---

## Project Structure

```
├── watermarks/
│   ├── binary.py       # Random ±1 bit pattern 
│   ├── logo.py         # 32×32 image watermark
│   └── text.py         # UTF-8 string -> ±1 bit array (with decoder)
├── methods/
│   ├── base.py         # Abstract Watermarker: embed / extract 
│   ├── lsb.py          # Spatial LSB — baseline / lower bound
│   ├── dct.py          # 8×8 block DCT, mid-frequency embedding
│   └── dwt.py          # 2-D Haar DWT, all four subbands (LL/LH/HL/HH)
├── metrics.py          # PSNR, SSIM, NCC, BER
├── attack.py           # In-memory JPEG encode→decode at a given QF
├── analyze.py          # for a given image runs analysis using all metrics over all combinations of methods and qf values
├── plot.py             # plots for results.csv analysis
└── generate_results.py # Downloads skimage test images, runs analyze.py
```

---

## Quickstart

**Install dependencies**
```bash
pip install numpy scipy scikit-image pillow matplotlib opencv-python

```

**Run the sweep on standard test images**
```bash
python generate_results.py
```
This pulls `camera`, `coins`, `moon`, `brick`, and `astronaut` from `skimage.data`, saves them temporarily, and runs the full sweep via `analyze.py`. Progress is printed live. Output: `results.csv`.

**Run on your own images**
```bash
python analyze.py <path_to_image>
```
Images can be colour — they are converted to grayscale internally.

**Generate plots**
```bash
python plot.py 
```
Produces comparitive plots.

## Dependencies

- `numpy`, `scipy` — core numerics and DWT (Haar, pure NumPy — no PyWavelets required)
- `scikit-image` — test images and SSIM reference
- `pillow` — in-memory JPEG encode/decode
- `opencv-python` — image I/O
- `matplotlib` — plots and heatmaps
