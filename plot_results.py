import pandas as pd
import matplotlib.pyplot as plt
import os

print("Starting graph generation...")

df = pd.read_csv("final_comparison.csv")

print("CSV loaded successfully!")
print(df)

# Generate WER graphs
for condition in ["Clean", "Noisy"]:
    data = df[df["Condition"] == condition]

    plt.figure(figsize=(10, 5))
    plt.bar(data["Model"], data["WER (%)"])
    plt.title(f"WER Comparison - {condition} Audio")
    plt.xlabel("Model")
    plt.ylabel("Word Error Rate (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    filename = f"wer_{condition.lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print("Saved:", os.path.abspath(filename))

# Generate processing-time graphs
for condition in ["Clean", "Noisy"]:
    data = df[df["Condition"] == condition]

    plt.figure(figsize=(10, 5))
    plt.bar(data["Model"], data["Average Time (seconds)"])
    plt.title(f"Processing Time - {condition} Audio")
    plt.xlabel("Model")
    plt.ylabel("Average Time (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    filename = f"time_{condition.lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print("Saved:", os.path.abspath(filename))

print("All four graphs generated successfully!")