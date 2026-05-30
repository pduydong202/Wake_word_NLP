import gradio as gr
import threading
import time

from configs.config import AppConfig
from src.pipeline import VoiceAssistantPipeline


assistant = None
running = False
worker_thread = None

latest_result = {
    "status": "Chưa khởi động.",
    "wake_score": 0.0,
    "text": "",
    "processed_text": "",
    "intent": "",
    "confidence": 0.0,
    "all_probs": {},
}

logs = []


def add_log(message: str):
    global logs
    timestamp = time.strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {message}")

    if len(logs) > 30:
        logs = logs[-30:]


def load_assistant():
    global assistant

    if assistant is None:
        add_log("Đang load model...")
        config = AppConfig()
        assistant = VoiceAssistantPipeline(config)
        add_log("Load model xong.")

    return assistant


def assistant_loop():
    global running, latest_result

    app = load_assistant()

    add_log("Assistant đã bắt đầu nghe wake word.")

    while running:
        add_log("Đang chờ wake word: hey mixi")

        result = app.run_once()

        if result is not None:
            latest_result = result

            add_log(f"Wake score: {result['wake_score']}")
            add_log(f"STT: {result['text']}")
            add_log(f"Intent: {result['intent']} | Confidence: {result['confidence']}")

        add_log("Quay lại trạng thái nghe wake word.")

    add_log("Assistant đã dừng.")


def start_assistant():
    global running, worker_thread

    if running:
        return get_ui_state("Assistant đang chạy rồi.")

    running = True
    worker_thread = threading.Thread(target=assistant_loop, daemon=True)
    worker_thread.start()

    return get_ui_state("Đã bật assistant.")


def stop_assistant():
    global running

    running = False

    return get_ui_state("Đã yêu cầu dừng assistant.")


def get_ui_state(status_override=None):
    status = status_override if status_override else latest_result["status"]

    probs = latest_result.get("all_probs", {})

    probs_text = ""
    if probs:
        probs_text = "\n".join(
            [f"{label}: {prob:.4f}" for label, prob in probs.items()]
        )

    log_text = "\n".join(logs[-30:])

    return (
        status,
        latest_result["wake_score"],
        latest_result["text"],
        latest_result["processed_text"],
        latest_result["intent"],
        latest_result["confidence"],
        probs_text,
        log_text,
    )


def refresh_ui():
    return get_ui_state()


with gr.Blocks(title="Vietnamese Wake Word Voice Assistant") as demo:
    gr.Markdown(
        """
        # Vietnamese Wake Word Voice Assistant

        Demo pipeline:

        **Wake word → VAD → PhoWhisper STT → PhoBERT Intent Classification**

        Cách dùng:

        1. Bấm **Start Assistant**
        2. Nói **"hey mixi"**
        3. Sau khi hệ thống kích hoạt, nói câu lệnh như **"bật quạt"**, **"mở nhạc"**, **"thời tiết hôm nay thế nào"**
        4. Xem kết quả STT và intent trên giao diện
        """
    )

    with gr.Row():
        start_btn = gr.Button("Start Assistant", variant="primary")
        stop_btn = gr.Button("Stop Assistant", variant="stop")
        refresh_btn = gr.Button("Refresh UI")

    status_box = gr.Textbox(label="Trạng thái", value="Chưa khởi động.", interactive=False)

    with gr.Row():
        wake_score_box = gr.Number(label="Wake word score", value=0.0)
        confidence_box = gr.Number(label="Intent confidence", value=0.0)

    raw_text_box = gr.Textbox(label="STT raw text", interactive=False)
    processed_text_box = gr.Textbox(label="Segmented / processed text", interactive=False)
    intent_box = gr.Textbox(label="Intent label", interactive=False)

    probs_box = gr.Textbox(label="All probabilities", lines=4, interactive=False)
    logs_box = gr.Textbox(label="Logs", lines=15, interactive=False)

    outputs = [
        status_box,
        wake_score_box,
        raw_text_box,
        processed_text_box,
        intent_box,
        confidence_box,
        probs_box,
        logs_box,
    ]

    start_btn.click(fn=start_assistant, outputs=outputs)
    stop_btn.click(fn=stop_assistant, outputs=outputs)
    refresh_btn.click(fn=refresh_ui, outputs=outputs)

    timer = gr.Timer(1.0)
    timer.tick(fn=refresh_ui, outputs=outputs)


if __name__ == "__main__":
    demo.launch()