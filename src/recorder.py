import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch


class CommandRecorder:
    def __init__(self, config):
        self.config = config

        self.sample_rate = getattr(config, "sample_rate", 16000)

        # Silero VAD ở 16kHz dùng chunk 512 samples ~ 32ms
        self.chunk_duration = getattr(config, "vad_chunk_duration", 0.032)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        self.start_timeout = getattr(config, "start_timeout", 3.0)
        self.silence_duration = getattr(config, "silence_duration", 0.8)
        self.max_record_duration = getattr(config, "max_record_duration", 6.0)

        # Nên để 0.55 nếu mic hơi nhiễu
        self.vad_threshold = getattr(config, "vad_threshold", 0.55)

        self.pre_speech_seconds = getattr(config, "pre_speech_seconds", 0.35)
        self.post_speech_seconds = getattr(config, "post_speech_seconds", 0.20)

        self.pre_speech_chunks = max(
            1,
            int(self.pre_speech_seconds / self.chunk_duration)
        )

        self.post_speech_chunks = max(
            1,
            int(self.post_speech_seconds / self.chunk_duration)
        )

        self.silence_chunks_required = max(
            1,
            int(self.silence_duration / self.chunk_duration)
        )

        print("[Silero VAD] Loading...")

        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        self.model.eval()

        print("[Silero VAD] Loaded.")
        print(f"[Recorder] chunk_size={self.chunk_size}")
        print(f"[Recorder] vad_threshold={self.vad_threshold}")
        print(f"[Recorder] silence_chunks_required={self.silence_chunks_required}")

    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        audio_chunk: numpy float32, shape = [chunk_size]
        sample_rate: 16000
        """

        audio_chunk = audio_chunk.astype(np.float32)

        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / 32768.0

        tensor = torch.from_numpy(audio_chunk)

        with torch.no_grad():
            speech_prob = self.model(tensor, self.sample_rate).item()

        return speech_prob
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        speech_prob = self.get_speech_prob(audio_chunk)
        return speech_prob >= self.vad_threshold

    def record_until_silence(self):
        print("[Recorder] Đang chờ câu lệnh...")

        # Rất nên reset state trước mỗi lượt ghi câu lệnh
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

        audio_buffer = []
        pre_buffer = deque(maxlen=self.pre_speech_chunks)

        speech_started = False
        silence_count = 0
        post_count = 0

        wait_start_time = time.time()
        record_start_time = None

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(self.chunk_size)
                chunk = chunk.flatten()

                now = time.time()

                speech_prob = self.get_speech_prob(chunk)
                speech = speech_prob >= self.vad_threshold

                # Debug nhẹ, nếu thấy log này toàn 0.6 - 0.9 khi im lặng
                # thì tức là threshold thấp hoặc mic quá nhiễu
                # print(f"[VAD] prob={speech_prob:.3f} | speech={speech}")

                if not speech_started:
                    pre_buffer.append(chunk)

                    if speech:
                        print(f"[Recorder] Bắt đầu nhận giọng nói. prob={speech_prob:.3f}")

                        speech_started = True
                        record_start_time = now
                        silence_count = 0

                        # Thêm đoạn âm thanh trước khi VAD phát hiện nói
                        audio_buffer.extend(list(pre_buffer))
                        pre_buffer.clear()

                    else:
                        if now - wait_start_time > self.start_timeout:
                            print("[Recorder] Không nghe thấy câu lệnh.")
                            return None

                else:
                    audio_buffer.append(chunk)

                    if speech:
                        silence_count = 0
                        post_count = 0
                    else:
                        silence_count += 1

                    if silence_count >= self.silence_chunks_required:
                        print("[Recorder] Kết thúc câu lệnh.")

                        # Giữ thêm một đoạn nhỏ sau khi im lặng
                        while post_count < self.post_speech_chunks:
                            extra_chunk, _ = stream.read(self.chunk_size)
                            extra_chunk = extra_chunk.flatten()
                            audio_buffer.append(extra_chunk)
                            post_count += 1

                        break

                    if now - record_start_time >= self.max_record_duration:
                        print("[Recorder] Đạt giới hạn thời gian ghi.")
                        break

        if len(audio_buffer) == 0:
            return None

        audio = np.concatenate(audio_buffer).astype(np.float32)

        # Normalize nhẹ trước khi đưa vào STT
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs

        print(f"[Recorder] Audio length: {len(audio) / self.sample_rate:.2f}s")

        return audio