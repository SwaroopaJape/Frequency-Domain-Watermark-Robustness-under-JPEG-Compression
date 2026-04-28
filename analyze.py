import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from watermarks import binary as wm_binary, logo as wm_logo, text as wm_text
from methods import LSBWatermarker, DCTWatermarker, DWTWatermarker
from metrics import psnr, ssim, ncc, ber
from attack import jpeg_attack

# iterator parametors
ALPHA_SWEEP = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
QF_SWEEP    = [5, 10, 20, 30, 50, 70, 90, 100]

WM_N_BITS = None
WM_TEXT   = "CS611-IIT-GOA-WATERMARK" * 20

# capacity calculation
def _method_capacity(method, img: np.ndarray) -> int:
    h, w = img.shape[:2]
    if method.name == "LSB":
        return h * w
    elif method.name == "DCT":
        return (h // 8) * (w // 8) * 13
    else:  
        return (h // 2) * (w // 2)

def build_watermarks(capacity_bits: int) -> dict:
    n = max(1024, capacity_bits // 4)   # 25% of method capacity
    wm_bin = wm_binary(n_bits=n)
    wm_lg  = wm_logo(size=64).flatten()[:n]   # 64 instead of 32 -> 4096 bits available
    wm_txt = wm_text(WM_TEXT)[:n]
    return {"binary": wm_bin, "logo": wm_lg, "text": wm_txt}

# a single set of parameters on a single img
def evaluate_single_condition(img, method, watermark, alpha, qf, n_bits):
    # embed
    try:
        wm_img = method.embed(img, watermark, alpha=alpha)
    except Exception as e:
        print(f"EMBED FAIL [{method.name} α={alpha}]: {e}", file=sys.stderr)
        return None

    p_embed = psnr(img, wm_img)
    s_embed = ssim(img, wm_img)

    # attack
    attacked = jpeg_attack(wm_img, quality=qf)

    # extract
    try:
        ext_wm = method.extract(attacked, img, n_bits=n_bits, alpha=alpha)
    except Exception as e:
        print(f"EXTRACT FAIL [{method.name} qf={qf}]: {e}", file=sys.stderr)
        return None

    # metrics
    n_val = ncc(watermark, ext_wm)
    b_val = ber(watermark, ext_wm)
    p_atk = psnr(img, attacked)
    s_atk = ssim(img, attacked)

    return (p_embed, s_embed, p_atk, s_atk, n_val, b_val)

# load img and run all loops for the img
def process_image(image_path: str):
    img_path_obj = Path(image_path)
    if not img_path_obj.exists():
        print(f"File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(img_path_obj), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    img = img.astype(np.float32)
    img_name = img_path_obj.stem

    METHODS = [
        LSBWatermarker(),
        DCTWatermarker(),
        DWTWatermarker("LL"),
        DWTWatermarker("LH"),
        DWTWatermarker("HL"),
        DWTWatermarker("HH"),
    ]

    print("image,method,watermark_type,alpha,qf,PSNR_embed,SSIM_embed,PSNR_attack,SSIM_attack,NCC,BER")

    for method in METHODS:
        cap = _method_capacity(method, img)
        watermarks = build_watermarks(cap)

        for wm_name, watermark in watermarks.items():
            n_bits = len(watermark)

            for alpha in ALPHA_SWEEP:
                # skiping LSB alpha sweep (alpha is irrelevant for LSB)
                if method.name == "LSB" and alpha != ALPHA_SWEEP[0]:
                    continue

                for qf in QF_SWEEP:
                    metrics = evaluate_single_condition(
                        img, method, watermark, alpha, qf, n_bits
                    )
                    
                    if metrics:
                        p_emb, s_emb, p_atk, s_atk, n_val, b_val = metrics
                        print(f"{img_name},{method.name},{wm_name},{alpha},{qf},{p_emb:.4f},{s_emb:.6f},{p_atk:.4f},{s_atk:.6f},{n_val:.6f},{b_val:.6f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <path_to_image>", file=sys.stderr)
        sys.exit(1)
    
    target_image = sys.argv[1]
    process_image(target_image)

if __name__ == "__main__":
    main()
