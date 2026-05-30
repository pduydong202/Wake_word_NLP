from pathlib import Path
import shutil

import librosa
import soundfile as sf
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Gain
)

# =========================
# CONFIG
# =========================

SAMPLE_RATE = 16000

ROOT_IN = Path(r"D:\dataset_wake_word_split")
ROOT_OUT = Path(r"D:\dataset_wake_word_final")

AUDIO_EXTS = {".wav"}

NUM_AUG_POSITIVE = 10
NUM_AUG_GAN_GIONG = 6
NUM_AUG_BINH_THUONG = 3


def get_augmenter(mode):
    if mode == "positive":
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.012, p=0.45),
            TimeStretch(min_rate=0.92, max_rate=1.08, p=0.35),
            PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.35),
            Shift(min_shift=-0.15, max_shift=0.15, p=0.45),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.50),
        ])

    if mode == "gan_giong":
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.35),
            TimeStretch(min_rate=0.96, max_rate=1.04, p=0.25),
            PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.25),
            Shift(min_shift=-0.10, max_shift=0.10, p=0.35),
            Gain(min_gain_db=-4, max_gain_db=4, p=0.45),
        ])

    if mode == "binh_thuong":
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.020, p=0.50),
            TimeStretch(min_rate=0.90, max_rate=1.10, p=0.30),
            PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.20),
            Shift(min_shift=-0.20, max_shift=0.20, p=0.45),
            Gain(min_gain_db=-8, max_gain_db=6, p=0.50),
        ])

    raise ValueError("mode phải là: positive, gan_giong hoặc binh_thuong")


def list_audio_files(input_dir: Path):
    if not input_dir.exists():
        print(f"Không tìm thấy thư mục: {input_dir}")
        return []

    return sorted([
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    ])


def load_audio(path: Path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def augment_folder(
    input_dir: Path,
    output_dir: Path,
    prefix: str,
    mode: str,
    num_aug: int
):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(input_dir)
    augmenter = get_augmenter(mode)

    print(f"\nAugment: {input_dir}")
    print(f"Số file gốc: {len(files)}")
    print(f"Mỗi file tạo thêm: {num_aug}")

    total = 0

    for idx, input_path in enumerate(files, start=1):
        try:
            audio = load_audio(input_path)
        except Exception as e:
            print(f"Lỗi đọc {input_path.name}: {e}")
            continue

        # Lưu bản gốc vào dataset final
        out_original = output_dir / f"{prefix}_{idx:05d}_orig.wav"
        sf.write(out_original, audio, SAMPLE_RATE)
        total += 1

        # Lưu bản augment
        for i in range(num_aug):
            try:
                augmented = augmenter(samples=audio, sample_rate=SAMPLE_RATE)

                out_aug = output_dir / f"{prefix}_{idx:05d}_aug_{i:02d}.wav"
                sf.write(out_aug, augmented, SAMPLE_RATE)
                total += 1

            except Exception as e:
                print(f"Lỗi augment {input_path.name}: {e}")

        if idx % 50 == 0:
            print(f"Đã xử lý {idx}/{len(files)}")

    print(f"Xong: {total} file -> {output_dir}")


def copy_folder(
    input_dir: Path,
    output_dir: Path,
    prefix: str
):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(input_dir)

    print(f"\nCopy không augment: {input_dir}")
    print(f"Số file: {len(files)}")

    for idx, input_path in enumerate(files, start=1):
        out_path = output_dir / f"{prefix}_{idx:05d}.wav"
        shutil.copy2(input_path, out_path)

        if idx % 100 == 0:
            print(f"Đã copy {idx}/{len(files)}")

    print(f"Xong copy -> {output_dir}")


def main():
    print("Bắt đầu tạo dataset final...")
    print(f"Input : {ROOT_IN}")
    print(f"Output: {ROOT_OUT}")

    # =========================
    # AUGMENT TRAIN
    # =========================

    augment_folder(
        input_dir=ROOT_IN / "train" / "positive",
        output_dir=ROOT_OUT / "train" / "positive",
        prefix="positive",
        mode="positive",
        num_aug=NUM_AUG_POSITIVE
    )

    augment_folder(
        input_dir=ROOT_IN / "train" / "negative" / "gan_giong",
        output_dir=ROOT_OUT / "train" / "negative" / "gan_giong",
        prefix="negative_gan_giong",
        mode="gan_giong",
        num_aug=NUM_AUG_GAN_GIONG
    )

    augment_folder(
        input_dir=ROOT_IN / "train" / "negative" / "binh_thuong",
        output_dir=ROOT_OUT / "train" / "negative" / "binh_thuong",
        prefix="negative_binh_thuong",
        mode="binh_thuong",
        num_aug=NUM_AUG_BINH_THUONG
    )

    # =========================
    # COPY VAL/TEST, KHÔNG AUGMENT
    # =========================

    for split in ["val", "test"]:
        copy_folder(
            input_dir=ROOT_IN / split / "positive",
            output_dir=ROOT_OUT / split / "positive",
            prefix="positive"
        )

        copy_folder(
            input_dir=ROOT_IN / split / "negative" / "gan_giong",
            output_dir=ROOT_OUT / split / "negative" / "gan_giong",
            prefix="negative_gan_giong"
        )

        copy_folder(
            input_dir=ROOT_IN / split / "negative" / "binh_thuong",
            output_dir=ROOT_OUT / split / "negative" / "binh_thuong",
            prefix="negative_binh_thuong"
        )

    print("\nHoàn tất.")
    print(f"Dataset cuối nằm ở: {ROOT_OUT}")


if __name__ == "__main__":
    main()