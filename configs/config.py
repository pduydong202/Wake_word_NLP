from dataclasses import dataclass


@dataclass
class AppConfig:
    # =========================
    # Audio config
    # =========================
    sample_rate: int = 16000

    # =========================
    # Wake word config
    # =========================
    wake_package_path: str = "models/wakeword/wake_word_wav2vec2_final.pt"

    # Cửa sổ wake-word nên dài hơn độ dài câu "hey mixi" một chút
    wake_window_seconds: float = 2.0

    # Step càng nhỏ càng realtime nhưng tốn xử lý hơn
    wake_step_seconds: float = 0.4

    # Nếu hay nhận nhầm thì tăng 0.92 - 0.95
    wake_threshold: float = 0.90

    # 4 hits với step 0.4s nghĩa là cần ổn định khoảng 1.6s
    wake_required_hits: int = 3

    # =========================
    # Command recording config - fallback RMS
    # =========================
    command_max_seconds: float = 6.0
    command_min_seconds: float = 0.8

    # Nếu người dùng im lặng chừng này giây thì dừng ghi
    silence_seconds: float = 1.0

    # Ngưỡng im lặng cho RMS fallback
    rms_silence_threshold: float = 0.010

    # =========================
    # VAD command recording config
    # =========================
    use_vad: bool = True

    # 32ms là ổn cho realtime
    vad_chunk_duration: float = 0.032

    # Nếu hay cắt mất đầu câu, giảm xuống 0.35 - 0.45
    # Nếu hay bắt tiếng ồn là giọng nói, tăng lên 0.55 - 0.65
    vad_threshold: float = 0.5

    # Chờ người dùng bắt đầu nói sau wake-word
    start_timeout: float = 4

    # Im lặng bao lâu thì dừng ghi
    silence_duration: float = 0.8

    # Giới hạn thời lượng câu lệnh
    max_record_duration: float = 10

    # Giữ lại một đoạn âm thanh trước khi VAD phát hiện nói
    # Rất quan trọng để tránh mất chữ đầu: "bật", "mở", "tắt"
    pre_speech_seconds: float = 0.25

    # Giữ thêm một đoạn sau khi im lặng để câu không bị cụt
    post_speech_seconds: float = 0.25

    # =========================
    # PhoWhisper config
    # =========================
    phowhisper_model_name: str = "vinai/PhoWhisper-medium"

    # Có thể thêm nếu code STT hỗ trợ
    phowhisper_language: str = "vi"
    phowhisper_task: str = "transcribe"

    # =========================
    # PhoBERT intent config
    # =========================
    phobert_model_dir: str = "models/phobert_intent_model"

    use_word_segmentation: bool = True

    # Câu lệnh ngắn thì 64 là đủ, nhanh hơn 128 một chút
    max_text_length: int = 64
