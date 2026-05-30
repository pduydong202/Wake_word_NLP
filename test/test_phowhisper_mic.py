import torch
import sounddevice as sd
import scipy.io.wavfile as wav
import soundfile as sf
from transformers import pipeline

MODEL_NAME = "vinai/PhoWhisper-base"

SAMPLE_RATE = 16000
DURATION = 5
OUTPUT_FILE = "mic_input.wav"


def record_audio():
    print(f"\nĐang ghi âm {DURATION} giây...")
    print("Nói thử: bật quạt lên / mở nhạc đi / thời tiết hôm nay thế nào")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio)
    print("Đã ghi âm xong.")

    return OUTPUT_FILE


def main():
    device = 0 if torch.cuda.is_available() else -1

    print("Đang load model...")
    print("CUDA available:", torch.cuda.is_available())

    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device
    )

    print("Load model xong.")

    while True:
        input("\nNhấn Enter để ghi âm, hoặc Ctrl+C để thoát...")

        audio_path = record_audio()

        print("Đang nhận dạng...")

        audio_array, sr = sf.read(audio_path)

        result = pipe({
            "array": audio_array,
            "sampling_rate": sr
        })

        print("\nKết quả STT:")
        print(result["text"])


if __name__ == "__main__":
    main()