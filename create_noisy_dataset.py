from datasets import load_from_disk
import numpy as np
import soundfile as sf
from io import BytesIO
import os

print("Loading clean dataset...")
dataset = load_from_disk("librispeech_data")

output_folder = "noisy_audio"
os.makedirs(output_folder, exist_ok=True)

noise_level_db = 10

for index, sample in enumerate(dataset):
    audio_info = sample["audio"]

    if audio_info["bytes"] is not None:
        audio, sample_rate = sf.read(
            BytesIO(audio_info["bytes"]),
            dtype="float32"
        )
    elif audio_info["path"]:
        audio, sample_rate = sf.read(
            audio_info["path"],
            dtype="float32"
        )
    else:
        print(f"Skipping sample {index + 1}")
        continue

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    rng = np.random.default_rng(seed=index)
    noise = rng.normal(0, 1, len(audio)).astype(np.float32)

    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)

    if signal_power > 0 and noise_power > 0:
        scale = np.sqrt(
            signal_power /
            (noise_power * (10 ** (noise_level_db / 10)))
        )
        noisy_audio = audio + noise * scale
    else:
        noisy_audio = audio.copy()

    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

    output_path = os.path.join(
        output_folder,
        f"sample_{index + 1:03d}.wav"
    )

    sf.write(output_path, noisy_audio, sample_rate)

    print(f"Created noisy sample {index + 1}/{len(dataset)}")

print("\nNoisy dataset created successfully!")
print("Folder:", output_folder)
print("Noise level: 10 dB SNR")