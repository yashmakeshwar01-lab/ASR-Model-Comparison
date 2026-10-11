import pandas as pd

data = [
    {
        "Model": "Faster-Whisper Tiny",
        "Condition": "Clean",
        "WER (%)": 13.65,
        "Average Time (seconds)": 1.014
    },
    {
        "Model": "Faster-Whisper Tiny",
        "Condition": "Noisy",
        "WER (%)": 34.26,
        "Average Time (seconds)": 1.344
    },
    {
        "Model": "Wav2Vec2 Base",
        "Condition": "Clean",
        "WER (%)": 5.04,
        "Average Time (seconds)": 1.638
    },
    {
        "Model": "Wav2Vec2 Base",
        "Condition": "Noisy",
        "WER (%)": 32.35,
        "Average Time (seconds)": 1.513
    },
    {
        "Model": "Faster-Whisper Small",
        "Condition": "Clean",
        "WER (%)": 7.91,
        "Average Time (seconds)": 6.780
    },
    {
        "Model": "Faster-Whisper Small",
        "Condition": "Noisy",
        "WER (%)": 14.70,
        "Average Time (seconds)": 6.791
    }
]

df = pd.DataFrame(data)

df.to_csv("final_comparison.csv", index=False)

print("\n===== FINAL MODEL COMPARISON =====")
print(df.to_string(index=False))

print("\nSaved successfully to final_comparison.csv")