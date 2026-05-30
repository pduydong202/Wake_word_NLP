import json
import os
from collections import Counter

OUTPUT_FILE = "intent_raw.jsonl"

LABELS = {
    "1": "device_control",
    "2": "weather_query",
    "3": "entertainment"
}


def show_labels():
    print("\nChọn nhãn:")
    for key, label in LABELS.items():
        print(f"{key}. {label}")


def save_sample(text, label):
    sample = {
        "text": text.strip(),
        "label": label
    }

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_counts():
    counter = Counter()

    if not os.path.exists(OUTPUT_FILE):
        return counter

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                counter[item["label"]] += 1
            except Exception:
                pass

    return counter


def show_counts():
    counter = load_counts()
    print("\nSố lượng hiện tại:")
    for label in LABELS.values():
        print(f"- {label}: {counter[label]}")
    print(f"Tổng: {sum(counter.values())}")


def main():
    print("=== TOOL NHẬP DATASET INTENT CLASSIFICATION ===")
    print(f"Dữ liệu sẽ được lưu vào: {OUTPUT_FILE}")
    print("Gõ 'q' để thoát.")
    print("Gõ 'count' để xem số lượng từng nhãn.\n")

    while True:
        text = input("Nhập câu lệnh: ").strip()

        if text.lower() == "q":
            print("Đã thoát.")
            break

        if text.lower() == "count":
            show_counts()
            continue

        if not text:
            print("Câu rỗng, nhập lại.")
            continue

        show_labels()
        label_choice = input("Nhập số nhãn: ").strip()

        if label_choice not in LABELS:
            print("Nhãn không hợp lệ, bỏ qua câu này.")
            continue

        label = LABELS[label_choice]
        save_sample(text, label)

        print(f"Đã lưu: {text} -> {label}")
        show_counts()


if __name__ == "__main__":
    main()