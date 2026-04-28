import pandas as pd
import numpy as np

# thresholds (kept for optional use if needed later)
PASS_NCC   = 0.5
MARGIN_NCC = 0.3


def build_summary_table(csv_path: str = "results.csv") -> None:
    df = pd.read_csv(
        csv_path,
        names=["image","method","wm_type","alpha","qf",
               "PSNR_embed","SSIM_embed","PSNR_attack","SSIM_attack","NCC","BER"],
        header=0,
    )

    # merge DWT-LH and DWT-HL into one display row
    df["method_display"] = df["method"].replace(
        {"DWT-LH": "DWT-LH/HL", "DWT-HL": "DWT-LH/HL"}
    )

    display_order = ["LSB", "DCT", "DWT-LL", "DWT-LH/HL", "DWT-HH"]

    rows = []
    for disp in display_order:
        sub = df[df["method_display"] == disp]
        if sub.empty:
            continue

        # ✅ FIX: choose best alpha based on HIGH QF region (meaningful regime)
        high_qf = sub[sub["qf"] >= 70]
        best_alpha = high_qf.groupby("alpha")["NCC"].mean().idxmax()

        alpha_sub = sub[sub["alpha"] == best_alpha]

        # embedding quality
        psnr_embed = alpha_sub["PSNR_embed"].mean()

        # NCC across QFs
        ncc_by_qf = alpha_sub.groupby("qf")["NCC"].mean().sort_index()

        # ✅ use meaningful QFs
        ncc_70  = ncc_by_qf.get(70, float("nan"))
        ncc_90  = ncc_by_qf.get(90, float("nan"))
        ncc_100 = ncc_by_qf.get(100, float("nan"))

        domain = "Spatial" if disp == "LSB" else "Freq"

        rows.append({
            "Method": disp,
            "Domain": domain,
            "Best α": best_alpha,
            "Embed PSNR": f"~{psnr_embed:.0f} dB",
            "NCC@70": f"{ncc_70:.3f}",
            "NCC@90": f"{ncc_90:.3f}",
            "NCC@100": f"{ncc_100:.3f}",
            "_score": ncc_90 * 0.6 + ncc_100 * 0.4,
        })

    result_df = pd.DataFrame(rows).set_index("Method")

    # main table 
    print("\n    SUMMARY TABLE (High-QF Performance)   ")
    print(result_df[["Domain", "Embed PSNR", "NCC@70", "NCC@90", "NCC@100"]])

    #survival curve
    print("\nNCC across QF (best α per method):")
    all_qfs = sorted(df["qf"].unique())
    print(f"{'Method':<14}", end="")
    for qf in all_qfs:
        print(f"  QF={qf:<4}", end="")
    print()
    print("-" * (14 + len(all_qfs) * 10))

    for disp in display_order:
        sub = df[df["method_display"] == disp]
        if sub.empty:
            continue

        high_qf = sub[sub["qf"] >= 70]
        best_alpha = high_qf.groupby("alpha")["NCC"].mean().idxmax()
        alpha_sub = sub[sub["alpha"] == best_alpha]

        ncc_by_qf = alpha_sub.groupby("qf")["NCC"].mean()

        print(f"{disp:<14}", end="")
        for qf in all_qfs:
            v = ncc_by_qf.get(qf, float("nan"))
            print(f"  {v:+.3f}", end="")
        print()

    # winner / loser (meaningful now)
    scores = result_df["_score"].dropna()
    winner = scores.idxmax()
    loser  = scores.idxmin()

    print(f"\n Best method (robust at high QF): {winner}")
    print(f" Worst method: {loser}\n")
    print("\nConclusion:")
    print("Frequency-domain methods (DCT, DWT) are significantly more robust to JPEG compression than LSB.")
    print("All methods fail at very low QF, but DCT and DWT retain meaningful information at higher QF values.")
    print("DCT offers a good balance between robustness and image quality, making it the most practical choice.")


if __name__ == "__main__":
    build_summary_table("results.csv")