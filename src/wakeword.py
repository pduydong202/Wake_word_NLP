import queue
import numpy as np
import sounddevice as sd

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class MultiHeadAttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=4, attention_hidden=128):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, num_heads),
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [batch, time, hidden]
        attention_mask: [batch, input_length]
        """

        scores = self.attention(hidden_states)
        # [batch, time, heads]

        if attention_mask is not None:
            mask = F.interpolate(
                attention_mask.unsqueeze(1).float(),
                size=hidden_states.shape[1],
                mode="nearest",
            ).squeeze(1)

            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)
        # [batch, time, heads]

        mean = torch.einsum("bth,btd->bhd", weights, hidden_states)
        # [batch, heads, hidden]

        diff = hidden_states.unsqueeze(1) - mean.unsqueeze(2)
        # [batch, heads, time, hidden]

        weights_h = weights.permute(0, 2, 1).unsqueeze(-1)
        # [batch, heads, time, 1]

        var = torch.sum(weights_h * diff ** 2, dim=2)
        std = torch.sqrt(var.clamp(min=1e-9))

        pooled = torch.cat([mean, std], dim=-1)
        # [batch, heads, hidden * 2]

        pooled = pooled.reshape(hidden_states.size(0), -1)
        # [batch, heads * hidden * 2]

        return pooled


class WakeWordWav2Vec2(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        num_labels=2,
        num_heads=4,
    ):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        hidden_size = self.wav2vec2.config.hidden_size

        self.pooling = MultiHeadAttentiveStatsPooling(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_hidden=128,
        )

        pooling_dim = hidden_size * 2 * num_heads

        self.classifier = nn.Sequential(
            nn.LayerNorm(pooling_dim),
            nn.Linear(pooling_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state
        pooled = self.pooling(hidden_states, attention_mask)
        logits = self.classifier(pooled)

        return logits


class WakeWordDetector:
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device

        self.sample_rate = int(config.sample_rate)
        self.window_size = int(config.sample_rate * config.wake_window_seconds)
        self.step_size = int(config.sample_rate * config.wake_step_seconds)

        self.audio_queue = queue.Queue()

        self.model, self.package_threshold = self._load_model()

    def _load_model(self):
        print("[WakeWord] Loading classifier package...")

        package = torch.load(
            self.config.wake_package_path,
            map_location=self.device,
            weights_only=False,
        )

        model_name = package.get("model_name", "facebook/wav2vec2-base")
        num_heads = int(package.get("num_heads", 4))
        package_threshold = float(package.get("threshold", self.config.wake_threshold))

        model = WakeWordWav2Vec2(
            model_name=model_name,
            num_labels=2,
            num_heads=num_heads,
        ).to(self.device)

        model.load_state_dict(package["model_state_dict"])
        model.eval()

        print("[WakeWord] Loaded:", model_name)
        print("[WakeWord] Package threshold:", package_threshold)
        print("[WakeWord] Using threshold:", self.config.wake_threshold)
        print("[WakeWord] Required hits:", self.config.wake_required_hits)

        return model, package_threshold

    def _resample_if_needed(
        self,
        audio: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        """
        Resample đơn giản bằng numpy để tránh phải cài thêm scipy/librosa.
        Frontend nên gửi 16000Hz, nhưng hàm này giúp an toàn hơn nếu lệch sample rate.
        """

        if source_sample_rate == target_sample_rate:
            return audio

        if len(audio) == 0:
            return audio

        duration = len(audio) / float(source_sample_rate)

        old_indices = np.linspace(0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * target_sample_rate)
        new_indices = np.linspace(0, duration, num=new_length, endpoint=False)

        resampled = np.interp(new_indices, old_indices, audio)

        return resampled.astype(np.float32)

    def _prepare_numpy_audio(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """
        Chuẩn hóa audio từ frontend hoặc sounddevice:
        - ép về numpy float32
        - mono
        - range [-1, 1]
        - resample về config.sample_rate nếu cần
        """

        if audio is None:
            return np.zeros((0,), dtype=np.float32)

        audio = np.asarray(audio)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        # Nếu frontend gửi int16 trực tiếp thì đưa về [-1, 1]
        if np.max(np.abs(audio)) > 1.5:
            audio = audio / 32768.0

        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            audio = self._resample_if_needed(
                audio=audio,
                source_sample_rate=int(sample_rate),
                target_sample_rate=self.sample_rate,
            )

        return audio

    def _normalize_audio(self, wav_np: np.ndarray) -> torch.Tensor:
        wav = torch.tensor(wav_np, dtype=torch.float32)

        if wav.ndim > 1:
            wav = wav.mean(dim=1)

        max_abs = wav.abs().max()
        if max_abs > 0:
            wav = wav / max_abs

        if wav.shape[0] > self.window_size:
            wav = wav[-self.window_size:]
        elif wav.shape[0] < self.window_size:
            pad_len = self.window_size - wav.shape[0]
            wav = F.pad(wav, (pad_len, 0))

        return wav

    @torch.no_grad()
    def predict_score(self, audio_window: np.ndarray) -> float:
        """
        Hàm cũ: dự đoán score từ một đoạn audio đã đúng sample rate.
        Dùng cho terminal/local.
        """

        wav = self._normalize_audio(audio_window)
        wav = wav.unsqueeze(0).to(self.device)

        attention_mask = torch.ones_like(wav, dtype=torch.long).to(self.device)

        logits = self.model(
            input_values=wav,
            attention_mask=attention_mask,
        )

        prob = torch.softmax(logits, dim=-1)[:, 1].item()

        return float(prob)

    @torch.no_grad()
    def predict_array(self, audio_window: np.ndarray, sample_rate: int | None = None) -> float:
        """
        Hàm mới: dùng cho web realtime.

        pipeline.py sẽ gọi:
            self.wake_detector.predict_array(audio_window, sample_rate)

        audio_window:
            numpy array float32 hoặc int16

        sample_rate:
            sample rate từ frontend, thường là 16000
        """

        audio_window = self._prepare_numpy_audio(
            audio=audio_window,
            sample_rate=sample_rate,
        )

        if len(audio_window) == 0:
            return 0.0

        return self.predict_score(audio_window)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("[WakeWord] Audio status:", status)

        self.audio_queue.put(indata.copy())

    def listen_until_wake(self) -> float:
        """
        Hàm cũ: nghe trực tiếp từ mic máy tính bằng sounddevice.
        Dùng cho terminal demo.
        """

        print()
        print("[WakeWord] Đang nghe wake word...")
        print("[WakeWord] Hãy nói: hey mixi")
        print()

        audio_buffer = np.zeros((0,), dtype=np.float32)
        hit_count = 0

        # Clear queue cũ để tránh dính audio từ lần trước
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        with sd.InputStream(
            channels=1,
            samplerate=self.config.sample_rate,
            blocksize=self.step_size,
            callback=self._audio_callback,
        ):
            while True:
                block = self.audio_queue.get()
                block = block.squeeze()

                block = self._prepare_numpy_audio(
                    audio=block,
                    sample_rate=self.config.sample_rate,
                )

                audio_buffer = np.concatenate([audio_buffer, block])

                if len(audio_buffer) > self.window_size:
                    audio_buffer = audio_buffer[-self.window_size:]

                if len(audio_buffer) < self.window_size:
                    continue

                score = self.predict_score(audio_buffer)

                if score >= self.config.wake_threshold:
                    hit_count += 1
                else:
                    hit_count = 0

                status_text = (
                    f"[WakeWord] prob={score:.4f} | "
                    f"threshold={self.config.wake_threshold:.3f} | "
                    f"hits={hit_count}/{self.config.wake_required_hits}"
                )

                print("\r" + status_text + " " * 30, end="", flush=True)

                if hit_count >= self.config.wake_required_hits:
                    print()
                    print("Xin chào, Mixi đây")
                    return score