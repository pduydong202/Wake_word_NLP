import time
import queue
import numpy as np
import sounddevice as sd

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


PACKAGE_PATH = "hey_mixi_siamese_wav2vec2_package.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.5
STEP_SECONDS = 0.5

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SIZE = int(SAMPLE_RATE * STEP_SECONDS)

# Tăng ngưỡng để khó kích hoạt hơn local
OVERRIDE_THRESHOLD = 0.9

# Cần nhiều window liên tiếp vượt ngưỡng hơn
REQUIRED_HITS = 1

# Sau khi nhận wake word thì kết thúc luôn
print("Device:", DEVICE)


class SiameseWav2Vec2(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        embedding_dim=128,
        freeze_encoder=True
    ):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, wav):
        outputs = self.encoder(wav)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)

        emb = self.projector(pooled)
        emb = F.normalize(emb, p=2, dim=1)

        return emb


print("Loading package...")
package = torch.load(PACKAGE_PATH, map_location=DEVICE)

model_name = package.get("model_name", "facebook/wav2vec2-base")
embedding_dim = package.get("embedding_dim", 128)

# Dùng threshold trong package để tham khảo, nhưng override bằng threshold cao hơn
package_threshold = float(package["threshold"])
threshold = OVERRIDE_THRESHOLD

prototype = package["prototype"].to(DEVICE)

model = SiameseWav2Vec2(
    model_name=model_name,
    embedding_dim=embedding_dim,
    freeze_encoder=True
).to(DEVICE)

model.load_state_dict(package["model_state_dict"])
model.eval()

print("Loaded model:", model_name)
print("Package threshold:", package_threshold)
print("Using threshold:", threshold)
print("Required hits:", REQUIRED_HITS)
print("Sample rate:", SAMPLE_RATE)


def normalize_audio(wav_np):
    wav = torch.tensor(wav_np, dtype=torch.float32)

    if wav.ndim > 1:
        wav = wav.mean(dim=1)

    max_abs = wav.abs().max()
    if max_abs > 0:
        wav = wav / max_abs

    if wav.shape[0] > WINDOW_SIZE:
        wav = wav[-WINDOW_SIZE:]
    elif wav.shape[0] < WINDOW_SIZE:
        pad_len = WINDOW_SIZE - wav.shape[0]
        wav = F.pad(wav, (pad_len, 0))

    return wav


@torch.no_grad()
def predict_window(audio_window):
    wav = normalize_audio(audio_window)
    wav = wav.unsqueeze(0).to(DEVICE)

    emb = model.forward_once(wav)

    sim = F.cosine_similarity(
        emb,
        prototype.unsqueeze(0)
    ).item()

    return sim


audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def main():
    print()
    print("Đang nghe...")
    print("Nói: hey mixi")
    print("Chương trình sẽ tự kết thúc khi nhận được wake word.")
    print()

    audio_buffer = np.zeros((0,), dtype=np.float32)
    hit_count = 0

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=STEP_SIZE,
        callback=audio_callback
    ):
        while True:
            block = audio_queue.get()
            block = block.squeeze()

            audio_buffer = np.concatenate([audio_buffer, block])

            if len(audio_buffer) > WINDOW_SIZE:
                audio_buffer = audio_buffer[-WINDOW_SIZE:]

            if len(audio_buffer) < WINDOW_SIZE:
                continue

            score = predict_window(audio_buffer)

            if score >= threshold:
                hit_count += 1
            else:
                hit_count = 0

            print(
                f"Similarity: {score:.3f} | threshold: {threshold:.3f} | hits: {hit_count}/{REQUIRED_HITS}",
                end="\r"
            )

            if hit_count >= REQUIRED_HITS:
                print()
                print("Xin chào, Mixi đây")
                break


if __name__ == "__main__":
    main()