import time
import torch
import numpy as np

from src.wakeword import WakeWordDetector
from src.recorder import CommandRecorder
from src.stt import PhoWhisperSTT
from src.intent import PhoBERTIntentClassifier
from src.speaker import Speaker


class VoiceAssistantPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[System] Device:", self.device)

        self.wake_detector = WakeWordDetector(config, self.device)
        self.recorder = CommandRecorder(config)
        self.stt = PhoWhisperSTT(config, self.device)
        self.intent_classifier = PhoBERTIntentClassifier(config, self.device)

        # Speaker sẽ random 1 file trong assets/sounds/
        self.speaker = Speaker()

        # ==========================================================
        # STATE CHO WEB REALTIME STREAM
        # ==========================================================
        self.state = "WAIT_WAKE"

        self.audio_buffer = np.array([], dtype=np.float32)
        self.command_buffer = np.array([], dtype=np.float32)

        self.last_status_time = 0.0
        self.last_wake_score = 0.0

        # Sliding window + required hits cho web
        self.stream_hit_count = 0
        self.last_wake_check_time = 0.0

        # State VAD cho web stream sau wake word
        self.stream_speech_started = False
        self.stream_wait_start_time = None
        self.stream_record_start_time = None
        self.stream_last_speech_time = None

        # Buffer để gom đúng kích thước chunk cho Silero VAD
        self.stream_vad_buffer = np.array([], dtype=np.float32)

        # Config chung
        self.sample_rate = self._get_config_value("sample_rate", 16000)
        self.wake_threshold = self._get_config_value("wake_threshold", 0.9)

        # Cửa sổ wake word, ví dụ 1.5s hoặc 2.0s
        self.wake_window_seconds = self._get_config_value(
            "wake_window_seconds",
            1.5,
        )

        # Bước trượt cửa sổ.
        # Ví dụ 0.25s nghĩa là cứ 0.25 giây check wake word một lần.
        self.wake_step_seconds = self._get_config_value(
            "wake_step_seconds",
            0.25,
        )

        # Giữ tối đa bao nhiêu giây audio khi đang chờ wake word
        self.max_wait_seconds = self._get_config_value(
            "max_wait_seconds",
            5.0,
        )

        # Giữ lại để tương thích config cũ, nhưng stream command sẽ dùng VAD
        self.command_seconds = self._get_config_value(
            "command_seconds",
            4.0,
        )

        # Bao lâu thì gửi status về frontend một lần
        self.status_interval = self._get_config_value(
            "status_interval",
            0.5,
        )

        # Config VAD command recording
        self.start_timeout = self._get_config_value("start_timeout", 3.0)
        self.silence_duration = self._get_config_value("silence_duration", 0.8)
        self.max_record_duration = self._get_config_value("max_record_duration", 5.0)
        self.command_min_seconds = self._get_config_value("command_min_seconds", 0.0)

        self.vad_chunk_duration = self._get_config_value("vad_chunk_duration", 0.032)
        self.vad_chunk_size = int(self.sample_rate * self.vad_chunk_duration)

        print("[Stream Config] wake_window_seconds:", self.wake_window_seconds)
        print("[Stream Config] wake_step_seconds:", self.wake_step_seconds)
        print("[Stream Config] wake_threshold:", self.wake_threshold)
        print("[Stream Config] wake_required_hits:", self.config.wake_required_hits)
        print("[Stream Config] command_seconds:", self.command_seconds)
        print("[Stream Config] start_timeout:", self.start_timeout)
        print("[Stream Config] silence_duration:", self.silence_duration)
        print("[Stream Config] max_record_duration:", self.max_record_duration)
        print("[Stream Config] vad_chunk_size:", self.vad_chunk_size)

    def _get_config_value(self, name, default):
        return getattr(self.config, name, default)

    def _empty_stream_result(self, status):
        return {
            "status": status,
            "wake_score": round(float(self.last_wake_score), 4),
            "text": "",
            "processed_text": "",
            "intent": "",
            "confidence": 0.0,
            "all_probs": {},
        }

    def _stream_vad_is_speech(self, audio_chunk):
        """
        Dùng lại Silero VAD trong CommandRecorder cho web stream.

        Lý do cần buffer riêng:
        - Frontend có thể gửi audio chunk không đúng bằng vad_chunk_size.
        - recorder.is_speech() nên nhận chunk đều, ví dụ 512 samples ở 16kHz.
        """

        if audio_chunk is None or len(audio_chunk) == 0:
            return False

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        self.stream_vad_buffer = np.concatenate(
            [self.stream_vad_buffer, audio_chunk]
        )

        if len(self.stream_vad_buffer) < self.vad_chunk_size:
            return False

        speech_detected = False

        while len(self.stream_vad_buffer) >= self.vad_chunk_size:
            vad_chunk = self.stream_vad_buffer[: self.vad_chunk_size]
            self.stream_vad_buffer = self.stream_vad_buffer[self.vad_chunk_size :]

            if self.recorder.is_speech(vad_chunk):
                speech_detected = True

        return speech_detected

    # ==========================================================
    # MODE 1: TERMINAL / LOCAL MIC
    # ==========================================================
    def run_once(self):
        try:
            # 1. Nghe wake word
            wake_score = self.wake_detector.listen_until_wake()

            print()
            print(f"[System] Wake word detected. score={wake_score:.3f}")

            # 2. Trợ lý nói sau khi được wake word
            self.speaker.play_random()

            print("[System] Hãy nói câu lệnh...")

            # 3. VAD ghi câu lệnh
            command_audio = self.recorder.record_until_silence()

            if command_audio is None:
                print("[System] Không nghe thấy câu lệnh sau wake word.")

                return {
                    "status": "Không nghe thấy câu lệnh sau wake word.",
                    "wake_score": round(wake_score, 4),
                    "text": "",
                    "processed_text": "",
                    "intent": "",
                    "confidence": 0.0,
                    "all_probs": {},
                }

            print()
            print(f"[Recorder] Đã ghi được {len(command_audio)} samples.")

            # 4. Speech-to-Text
            print()
            print("[PhoWhisper] Đang chuyển giọng nói thành văn bản...")
            text = self.stt.transcribe(command_audio)

            print()
            print("[PhoWhisper] Text:", text)

            if not text or not text.strip():
                print("[System] Không nhận được văn bản từ PhoWhisper.")

                return {
                    "status": "Không nhận được văn bản từ PhoWhisper.",
                    "wake_score": round(wake_score, 4),
                    "text": "",
                    "processed_text": "",
                    "intent": "",
                    "confidence": 0.0,
                    "all_probs": {},
                }

            # 5. Intent classification
            print()
            print("[PhoBERT] Đang tách từ và phân loại ý định...")
            intent_result = self.intent_classifier.predict(text)

            result = {
                "status": "Hoàn thành xử lý câu lệnh.",
                "wake_score": round(wake_score, 4),
                "text": intent_result["raw_text"],
                "processed_text": intent_result["processed_text"],
                "intent": intent_result["label"],
                "confidence": round(intent_result["confidence"], 4),
                "all_probs": intent_result["all_probs"],
            }

            print()
            print("================ KẾT QUẢ ================")
            print("Wake word score :", result["wake_score"])
            print("STT raw text    :", result["text"])
            print("Segmented text  :", result["processed_text"])
            print("Intent label    :", result["intent"])
            print("Confidence      :", result["confidence"])
            print("All probs       :", result["all_probs"])
            print("=========================================")

            return result

        except Exception as e:
            print()
            print("[System] Có lỗi trong một lượt xử lý:", str(e))

            return {
                "status": f"Lỗi: {str(e)}",
                "wake_score": 0.0,
                "text": "",
                "processed_text": "",
                "intent": "",
                "confidence": 0.0,
                "all_probs": {},
            }

    # ==========================================================
    # MODE 2: WEB REALTIME STREAM
    # Frontend gửi audio chunk liên tục qua WebSocket.
    # app.py gọi hàm này mỗi khi nhận được audio chunk.
    # ==========================================================
    def run_stream_chunk(self, audio_chunk, sample_rate=16000):
        try:
            if audio_chunk is None or len(audio_chunk) == 0:
                return None

            # Đảm bảo audio là float32 mono [-1, 1]
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)

            if np.max(np.abs(audio_chunk)) > 1.0:
                audio_chunk = audio_chunk / 32768.0

            if self.state == "WAIT_WAKE":
                return self._handle_wait_wake(audio_chunk, sample_rate)

            if self.state == "LISTEN_COMMAND":
                return self._handle_listen_command(audio_chunk, sample_rate)

            self._reset_stream_state()
            return None

        except Exception as e:
            print()
            print("[Stream] Lỗi:", str(e))

            self._reset_stream_state()

            return {
                "status": f"Lỗi stream: {str(e)}",
                "wake_score": 0.0,
                "text": "",
                "processed_text": "",
                "intent": "",
                "confidence": 0.0,
                "all_probs": {},
            }

    def _handle_wait_wake(self, audio_chunk, sample_rate):
        """
        Trạng thái WAIT_WAKE:
        - Gom audio chunk từ frontend vào audio_buffer
        - Giữ vài giây gần nhất
        - Cứ wake_step_seconds thì lấy wake_window_seconds cuối để predict
        - Cần wake_required_hits lần liên tiếp vượt threshold mới kích hoạt
        """

        # 1. Gom audio vào buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # 2. Chỉ giữ vài giây gần nhất để tránh buffer quá lớn
        max_wait_samples = int(sample_rate * self.max_wait_seconds)
        if len(self.audio_buffer) > max_wait_samples:
            self.audio_buffer = self.audio_buffer[-max_wait_samples:]

        # 3. Chưa đủ audio cho một cửa sổ wake word
        wake_window_samples = int(sample_rate * self.wake_window_seconds)
        if len(self.audio_buffer) < wake_window_samples:
            return None

        now = time.time()

        # 4. Sliding step:
        # Không predict mọi chunk, mà cứ wake_step_seconds mới predict một lần
        if now - self.last_wake_check_time < self.wake_step_seconds:
            return None

        self.last_wake_check_time = now

        # 5. Lấy cửa sổ audio gần nhất để predict
        audio_window = self.audio_buffer[-wake_window_samples:]

        wake_score = self.wake_detector.predict_array(
            audio_window,
            sample_rate,
        )

        self.last_wake_score = float(wake_score)

        # 6. Required hits giống bản terminal
        if wake_score >= self.wake_threshold:
            self.stream_hit_count += 1
        else:
            self.stream_hit_count = 0

        print(
            f"\r[Stream Wake] prob={wake_score:.4f} | "
            f"threshold={self.wake_threshold:.3f} | "
            f"hits={self.stream_hit_count}/{self.config.wake_required_hits}",
            end="",
            flush=True,
        )

        # 7. Nếu đủ số lần hit liên tiếp thì kích hoạt wake word
        if self.stream_hit_count >= self.config.wake_required_hits:
            print()
            print(f"[Stream] Wake word detected. score={wake_score:.3f}")

            self.state = "LISTEN_COMMAND"

            # Reset buffer câu lệnh để chỉ lấy audio sau wake word
            self.command_buffer = np.array([], dtype=np.float32)

            # Reset state VAD câu lệnh
            self.stream_speech_started = False
            self.stream_wait_start_time = time.time()
            self.stream_record_start_time = None
            self.stream_last_speech_time = None
            self.stream_vad_buffer = np.array([], dtype=np.float32)

            # Reset hit count sau khi đã kích hoạt
            self.stream_hit_count = 0

            return {
                "status": (
                    f"Đã phát hiện wake word. "
                    f"Score={wake_score:.3f}. Hãy nói câu lệnh..."
                ),
                "wake_score": round(float(wake_score), 4),
                "text": "",
                "processed_text": "",
                "intent": "",
                "confidence": 0.0,
                "all_probs": {},
            }

        # 8. Gửi status về frontend nhưng không spam quá nhiều
        if now - self.last_status_time >= self.status_interval:
            self.last_status_time = now

            return {
                "status": (
                    f"Đang chờ wake word... "
                    f"hits={self.stream_hit_count}/{self.config.wake_required_hits}"
                ),
                "wake_score": round(float(wake_score), 4),
                "text": "",
                "processed_text": "",
                "intent": "",
                "confidence": 0.0,
                "all_probs": {},
            }

        return None

    def _handle_listen_command(self, audio_chunk, sample_rate):
        """
        Trạng thái LISTEN_COMMAND:
        - Sau khi wake word được kích hoạt, dùng VAD để chờ câu lệnh.
        - Nếu chưa có speech thật trong start_timeout thì bỏ qua, không gọi STT.
        - Nếu đã có speech thì gom audio.
        - Dừng khi silence_duration hoặc max_record_duration.
        - Sau đó giữ nguyên logic STT -> Intent như cũ.
        """

        now = time.time()

        if self.stream_wait_start_time is None:
            self.stream_wait_start_time = now

        speech = self._stream_vad_is_speech(audio_chunk)

        # ======================================================
        # 1. Chưa bắt đầu có speech sau wake word
        # ======================================================
        if not self.stream_speech_started:
            if speech:
                print("[Stream Recorder] Bắt đầu nhận giọng nói...")

                self.stream_speech_started = True
                self.stream_record_start_time = now
                self.stream_last_speech_time = now

                self.command_buffer = np.concatenate(
                    [self.command_buffer, audio_chunk]
                )

                return {
                    "status": "Đang nghe câu lệnh...",
                    "wake_score": round(float(self.last_wake_score), 4),
                    "text": "",
                    "processed_text": "",
                    "intent": "",
                    "confidence": 0.0,
                    "all_probs": {},
                }

            if now - self.stream_wait_start_time > self.start_timeout:
                print("[Stream Recorder] Không nghe thấy câu lệnh.")

                result = self._empty_stream_result(
                    "Không nghe thấy câu lệnh sau wake word. Quay lại chờ wake word."
                )

                self._reset_stream_state()
                return result

            if now - self.last_status_time >= self.status_interval:
                self.last_status_time = now

                return {
                    "status": "Đang chờ câu lệnh...",
                    "wake_score": round(float(self.last_wake_score), 4),
                    "text": "",
                    "processed_text": "",
                    "intent": "",
                    "confidence": 0.0,
                    "all_probs": {},
                }

            return None

        # ======================================================
        # 2. Đã có speech thì mới gom audio
        # ======================================================
        self.command_buffer = np.concatenate([self.command_buffer, audio_chunk])

        if speech:
            self.stream_last_speech_time = now

        current_seconds = len(self.command_buffer) / sample_rate

        should_stop = False

        if self.stream_last_speech_time is not None:
            if now - self.stream_last_speech_time >= self.silence_duration:
                should_stop = True
                print("[Stream Recorder] Kết thúc câu lệnh.")

        if self.stream_record_start_time is not None:
            if now - self.stream_record_start_time >= self.max_record_duration:
                should_stop = True
                print("[Stream Recorder] Đạt giới hạn thời gian ghi.")

        if current_seconds < self.command_min_seconds:
            should_stop = False

        if not should_stop:
            if now - self.last_status_time >= self.status_interval:
                self.last_status_time = now

                return {
                    "status": f"Đang nghe câu lệnh... {current_seconds:.1f}s",
                    "wake_score": round(float(self.last_wake_score), 4),
                    "text": "",
                    "processed_text": "",
                    "intent": "",
                    "confidence": 0.0,
                    "all_probs": {},
                }

            return None

        if len(self.command_buffer) == 0:
            result = self._empty_stream_result(
                "Không nghe thấy câu lệnh sau wake word. Quay lại chờ wake word."
            )

            self._reset_stream_state()
            return result

        print()
        print(f"[Stream] Đã thu câu lệnh: {len(self.command_buffer)} samples.")

        # Normalize nhẹ giống recorder.py trước khi đưa vào STT
        command_audio = self.command_buffer.astype(np.float32)

        max_abs = np.max(np.abs(command_audio))
        if max_abs > 0:
            command_audio = command_audio / max_abs

        # ======================================================
        # 3. Speech-to-Text
        # ======================================================
        print("[PhoWhisper] Đang chuyển giọng nói thành văn bản...")
        text = self.stt.transcribe(command_audio)

        print("[PhoWhisper] Text:", text)

        if not text or not text.strip():
            self._reset_stream_state()

            return {
                "status": (
                    "Không nhận được văn bản từ PhoWhisper. "
                    "Quay lại chờ wake word."
                ),
                "wake_score": round(float(self.last_wake_score), 4),
                "text": "",
                "processed_text": "",
                "intent": "",
                "confidence": 0.0,
                "all_probs": {},
            }

        # ======================================================
        # 4. Intent classification
        # ======================================================
        print("[PhoBERT] Đang tách từ và phân loại ý định...")
        intent_result = self.intent_classifier.predict(text)

        result = {
            "status": "Hoàn thành xử lý câu lệnh. Đang chờ wake word tiếp theo.",
            "wake_score": round(float(self.last_wake_score), 4),
            "text": intent_result["raw_text"],
            "processed_text": intent_result["processed_text"],
            "intent": intent_result["label"],
            "confidence": round(intent_result["confidence"], 4),
            "all_probs": intent_result["all_probs"],
        }

        print()
        print("================ STREAM RESULT ================")
        print("Wake word score :", result["wake_score"])
        print("STT raw text    :", result["text"])
        print("Segmented text  :", result["processed_text"])
        print("Intent label    :", result["intent"])
        print("Confidence      :", result["confidence"])
        print("All probs       :", result["all_probs"])
        print("===============================================")

        # 5. Reset để quay lại chờ wake word tiếp theo
        self._reset_stream_state()

        return result

    def _reset_stream_state(self):
        """
        Reset trạng thái stream sau khi xử lý xong/lỗi.
        """

        self.state = "WAIT_WAKE"

        self.audio_buffer = np.array([], dtype=np.float32)
        self.command_buffer = np.array([], dtype=np.float32)

        self.last_status_time = 0.0
        self.last_wake_score = 0.0

        self.stream_hit_count = 0
        self.last_wake_check_time = 0.0

        self.stream_speech_started = False
        self.stream_wait_start_time = None
        self.stream_record_start_time = None
        self.stream_last_speech_time = None
        self.stream_vad_buffer = np.array([], dtype=np.float32)