import os
import csv
import matplotlib.pyplot as plt

# file paths
CSV_FILE = "results.csv"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

# ploting specifics
METHOD_COLORS = {"LSB": "#e63946", "DCT": "#2196F3", "DWT-LL": "#4CAF50", "DWT-LH": "#FF9800", "DWT-HL": "#9C27B0", "DWT-HH": "#795548"}
METHOD_MARKERS = {"LSB": "o", "DCT": "s", "DWT-LL": "^", "DWT-LH": "D", "DWT-HL": "v", "DWT-HH": "P"}
ALPHA_COLORS = {0.01: "#1a1a2e", 0.05: "#16213e", 0.1: "#0f3460", 0.3: "#533483", 0.5: "#e94560", 1.0: "#f5a623"}
WTYPE_COLORS = {"binary": "#e63946", "logo": "#2196F3", "text": "#4CAF50"}
WTYPE_MARKERS = {"binary": "o", "logo": "s", "text": "^"}
WTYPE_TITLES = {"binary": "Binary Noise", "logo": "Logo (32×32)", "text": "Text Bits"}

METRICS = [("NCC", "NCC ↑"), ("BER", "BER ↓"), ("PSNR_attack", "PSNR Attack ↑"), ("SSIM_attack", "SSIM Attack ↑")]

# load data
rows = list(csv.DictReader(open(CSV_FILE)))
for r in rows:
    r["alpha"] = float(r["alpha"])
    r["qf"] = float(r["qf"])
    for m in ("PSNR_embed", "SSIM_embed", "PSNR_attack", "SSIM_attack", "NCC", "BER"):
        r[m] = float(r[m])

IMAGES = sorted(set(r["image"] for r in rows))
METHODS = sorted(set(r["method"] for r in rows))
WTYPES = sorted(set(r["watermark_type"] for r in rows))
ALPHAS = sorted(set(r["alpha"] for r in rows))
QFS = sorted(set(r["qf"] for r in rows))

def get_avg(subset, metric):
    return sum(r[metric] for r in subset) / len(subset) if subset else float("nan")

def format_ax(ax):
    ax.tick_params(labelsize=7)
    ax.set_xlim(min(QFS) - 2, max(QFS) + 2)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

# per image robustness
for img in IMAGES + ["average"]:
    print(f"Plotting image: {img}")
    subset = rows if img == "average" else [r for r in rows if r["image"] == img]
    
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    fig.suptitle(f"Watermark Robustness vs JPEG QF  —  Image = {img.capitalize()}", fontsize=13, fontweight="bold")

    for col_idx, wtype in enumerate(WTYPES):
        w_subset = [r for r in subset if r["watermark_type"] == wtype]
        for row_idx, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            for method in METHODS:
                m_subset = [r for r in w_subset if r["method"] == method]
                ys = [get_avg([r for r in m_subset if r["qf"] == qf], metric) for qf in QFS]
                ax.plot(QFS, ys, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                        markersize=3.5, linewidth=1.4, label=method)
            
            if row_idx == 0: ax.set_title(WTYPE_TITLES[wtype], fontsize=9, fontweight="bold")
            if col_idx == 0: ax.set_ylabel(ylabel, fontsize=8)
            if row_idx == 3: ax.set_xlabel("Quality Factor (QF)", fontsize=8)
            format_ax(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), title="Method")
    fig.savefig(os.path.join(OUT_DIR, f"robustness_{img}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# alpha vs NCC
print("Plotting: alpha effect on NCC")
fig2, axes2 = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
fig2.suptitle("Effect of Embedding Strength α on NCC", fontsize=12, fontweight="bold")

for ax, method in zip(axes2.flatten(), METHODS):
    m_subset = [r for r in rows if r["method"] == method]
    for alpha in ALPHAS:
        a_subset = [r for r in m_subset if r["alpha"] == alpha]
        ys = [get_avg([r for r in a_subset if r["qf"] == qf], "NCC") for qf in QFS]
        ax.plot(QFS, ys, color=ALPHA_COLORS[alpha], marker="o", markersize=3, label=f"α={alpha}")

    ax.set_title(method, fontsize=10, fontweight="bold")
    ax.set_xlabel("QF", fontsize=8)
    ax.set_ylabel("NCC ↑", fontsize=8)
    format_ax(ax)

handles, labels = axes2.flatten()[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), title="Embedding Strength (α)")
fig2.savefig(os.path.join(OUT_DIR, "alpha_effect_NCC.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)

# method and payload type plots
print("Plotting: method * payload summary")
fig3, axes3 = plt.subplots(4, 6, figsize=(20, 11), constrained_layout=True)
fig3.suptitle("Method * Payload Type Robustness Summary", fontsize=12, fontweight="bold")

for col_idx, method in enumerate(METHODS):
    m_subset = [r for r in rows if r["method"] == method]
    for row_idx, (metric, ylabel) in enumerate(METRICS):
        ax = axes3[row_idx][col_idx]
        for wtype in WTYPES:
            w_subset = [r for r in m_subset if r["watermark_type"] == wtype]
            ys = [get_avg([r for r in w_subset if r["qf"] == qf], metric) for qf in QFS]
            ax.plot(QFS, ys, color=WTYPE_COLORS[wtype], marker=WTYPE_MARKERS[wtype],
                    markersize=3, linewidth=1.4, label=WTYPE_TITLES[wtype])

        if row_idx == 0: ax.set_title(method, fontsize=9, fontweight="bold")
        if col_idx == 0: ax.set_ylabel(ylabel, fontsize=8)
        if row_idx == 3: ax.set_xlabel("QF", fontsize=8)
        format_ax(ax)

handles, labels = axes3[0][0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), title="Payload / Watermark Type")
fig3.savefig(os.path.join(OUT_DIR, "method_payload_summary.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)

# alpha vs BER plots
print("Plotting: alpha effect on BER per image and payload")
fig4, axes4 = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
fig4.suptitle("Effect of α on BER per Image & Payload Type", fontsize=12, fontweight="bold")

for row_idx, img in enumerate(IMAGES):
    i_subset = [r for r in rows if r["image"] == img]
    for col_idx, wtype in enumerate(WTYPES):
        ax = axes4[row_idx][col_idx]
        w_subset = [r for r in i_subset if r["watermark_type"] == wtype]
        for alpha in ALPHAS:
            a_subset = [r for r in w_subset if r["alpha"] == alpha]
            ys = [get_avg([r for r in a_subset if r["qf"] == qf], "BER") for qf in QFS]
            ax.plot(QFS, ys, color=ALPHA_COLORS[alpha], marker="o", markersize=3, label=f"α={alpha}")

        if row_idx == 0: ax.set_title(WTYPE_TITLES[wtype], fontsize=9, fontweight="bold")
        if col_idx == 0: ax.set_ylabel(f"{img.capitalize()}\nBER ↓", fontsize=8)
        if row_idx == 3: ax.set_xlabel("QF", fontsize=8)
        format_ax(ax)

handles, labels = axes4[0][0].get_legend_handles_labels()
fig4.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), title="Embedding Strength (α)")
fig4.savefig(os.path.join(OUT_DIR, "alpha_BER_per_image_payload.png"), dpi=150, bbox_inches="tight")
plt.close(fig4)

print(f"\nAll plots saved to: {os.path.abspath(OUT_DIR)}")
