import os
import random
import subprocess
import json

def get_video_duration(input_path):
    """
    Lấy thời lượng video bằng ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        input_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    return float(info["format"]["duration"])


def split_mp4_random_segments(
    input_mp4,
    output_dir,
    num_segments=300,
    min_duration=1.0,
    max_duration=3.0
):
    os.makedirs(output_dir, exist_ok=True)

    total_duration = get_video_duration(input_mp4)

    if total_duration < max_duration:
        raise ValueError("Video quá ngắn để cắt đoạn 1–3 giây.")

    base_name = os.path.splitext(os.path.basename(input_mp4))[0]

    for i in range(num_segments):
        seg_duration = random.uniform(min_duration, max_duration)
        start_time = random.uniform(0, total_duration - seg_duration)

        output_path = os.path.join(
            output_dir,
            f"{base_name}_neg_{i:04d}_{seg_duration:.2f}s.mp4"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", input_mp4,
            "-t", str(seg_duration),
            "-c", "copy",
            output_path
        ]

        subprocess.run(cmd, check=True)

        print(f"Saved: {output_path}")

    print(f"Hoàn tất. Đã tạo {num_segments} file MP4 trong: {output_dir}")


split_mp4_random_segments(
    input_mp4="1.mp3",
    output_dir="negative_mp4_segments",
    num_segments=600,
    min_duration=1.0,
    max_duration=3.0
)