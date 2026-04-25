import pandas as pd
import matplotlib.pyplot as plt

def main():
    # data extraction and cleaning
    df = pd.read_csv("results.csv") 
    df = df.dropna()

    # extracting methodwise PSNR and ploting comparison
    methods = df['method'].unique()
    print(f"Data loaded successfully for methods: {methods}")

    plt.figure(figsize=(8, 5))

    for m in methods:
        subdata = df[df['method'] == m].groupby('alpha')['PSNR_embed'].mean()
        plt.plot(subdata.index, subdata.values, marker = 'o', label = m)

    plt.title("Image Quality vs. Alphas")
    plt.xlabel("Alpha")
    plt.ylabel("Average PSNR (dB)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    plt.figure(figsize=(8, 5))

    # extracting NCC and ploting comparison 
    for m in methods:
        subset = df[df['method'] == m].groupby('qf')['NCC'].mean()
        plt.plot(subset.index, subset.values, marker='s', label=m)

    plt.title("Watermark Robustness to JPEG Compression")
    plt.xlabel("JPEG Quality Factor")
    plt.ylabel("Average NCC")
    plt.xlim(0, 105) 
    plt.ylim(0, 1.1) 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    plt.figure(figsize=(6, 4))

    # ploting bar graph of BER
    wm_types = df['watermark_type'].unique()
    ber_means = [df[df['watermark_type'] == w]['BER'].mean() for w in wm_types]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.bar(wm_types, ber_means, color=colors[:len(wm_types)], edgecolor='black')
    plt.title("Extraction Errors by Payload Type")
    plt.xlabel("Watermark Type")
    plt.ylabel("Average Bit Error Rate (BER)")

    for i, v in enumerate(ber_means):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')

    plt.show()

if __name__ == "__main__":
    main()