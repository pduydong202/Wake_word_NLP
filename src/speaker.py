import os
import random
import sounddevice as sd
import soundfile as sf


class Speaker:
    def __init__(self, sound_dir="assets/sounds"):
        self.sound_dir = sound_dir
        self.supported_exts = (".wav", ".flac", ".ogg")

    def play_file(self, path: str):
        try:
            if not os.path.exists(path):
                print(f"[Speaker] File không tồn tại: {path}")
                return

            audio, sr = sf.read(path, dtype="float32")

            sd.play(audio, sr)
            sd.wait()

        except Exception as e:
            print("[Speaker] Không phát được âm thanh:", e)

    def play_random(self):
        try:
            if not os.path.exists(self.sound_dir):
                print(f"[Speaker] Folder không tồn tại: {self.sound_dir}")
                return

            files = [
                file for file in os.listdir(self.sound_dir)
                if file.lower().endswith(self.supported_exts)
            ]

            if not files:
                print(f"[Speaker] Không có file âm thanh trong folder: {self.sound_dir}")
                return

            random_file = random.choice(files)
            path = os.path.join(self.sound_dir, random_file)

            print(f"[Speaker] Playing: {path}")
            self.play_file(path)

        except Exception as e:
            print("[Speaker] Lỗi khi chọn file random:", e)