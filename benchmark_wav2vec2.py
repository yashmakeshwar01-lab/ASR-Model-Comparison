from datasets import load_from_disk
from transformers import pipeline
from jiwer import wer
from io import BytesIO
import soundfile as sf
import numpy as np
import pandas as pd
import time
import re
import string
import torch

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading dataset...")
dataset = load_from_disk("librispeech_data")

print("Loading Wav2Vec2 model...")
model_name = "facebook/wav2vec2-base-960h"

asr = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    device=-1
)

results = []
references = []
predictions = []

total_start = time.time()

for index, sample in enumerate(dataset):
    audio_info = sample["audio"]

    if audio_info["bytes"] is not None:
        audio_data, sample_rate = sf.read(
            BytesIO(audio_info["bytes"]),
            dtype="float32"
        )
    elif audio_info["path"]:
        audio_data, sample_rate = sf.read(
            audio_info["path"],
            dtype="float32"
        )
    else:
        print(f"Skipping sample {index + 1}: no audio data")
        continue

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=16000
        )
        sample_rate = 16000

    start = time.time()

    output = asr({
        "raw": audio_data,
        "sampling_rate": sample_rate
    })

    prediction = output["text"]
    elapsed = time.time() - start

    reference = sample["text"]

    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)

    references.append(normalized_reference)
    predictions.append(normalized_prediction)

    sample_wer = wer(
        normalized_reference,
        normalized_prediction
    )

    results.append({
        "sample": index + 1,
        "reference": reference,
        "prediction": prediction,
        "wer_percent": round(sample_wer * 100, 2),
        "processing_time_seconds": round(elapsed, 3)
    })

    print(
        f"Sample {index + 1}/{len(dataset)} | "
        f"WER: {sample_wer * 100:.2f}% | "
        f"Time: {elapsed:.2f}s"
    )

total_time = time.time() - total_start

overall_wer = wer(references, predictions)

df = pd.DataFrame(results)
df.to_csv("wav2vec2_results.csv", index=False)

print("\n===== WAV2VEC2 BENCHMARK SUMMARY =====")
print("Samples evaluated:", len(results))
print("Overall WER:", round(overall_wer * 100, 2), "%")
print(
    "Average transcription time:",
    round(df["processing_time_seconds"].mean(), 3),
    "seconds"
)
print("Total elapsed time:", round(total_time, 2), "seconds")
print("Results saved to wav2vec2_results.csv")