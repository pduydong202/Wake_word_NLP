import { useRef, useState } from "react";
import { startAudioStream } from "../utils/audioStream";

export default function VoiceAssistant() {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Chưa bắt đầu");
  const [wakeScore, setWakeScore] = useState("");

  const [lastResult, setLastResult] = useState({
    text: "",
    processedText: "",
    intent: "",
    confidence: "",
  });

  const socketRef = useRef(null);
  const audioControllerRef = useRef(null);
  const lastWakeSoundTimeRef = useRef(0);

  const cleanStatus = (rawStatus = "") => {
    const status = rawStatus.toLowerCase();

    if (status.includes("đang chờ wake word")) {
      return "Đang chờ wake word: hey mixi";
    }

    if (status.includes("đã phát hiện wake word")) {
      return "Đã phát hiện wake word. Hãy nói câu lệnh...";
    }

    if (status.includes("đang nghe câu lệnh")) {
      return "Đang nghe câu lệnh...";
    }

    if (status.includes("hoàn thành xử lý")) {
      return "Xử lý xong. Đang chờ wake word tiếp theo.";
    }

    if (status.includes("không nhận được văn bản")) {
      return "Không nhận diện được câu lệnh. Đang chờ lại wake word.";
    }

    if (status.includes("backend đã kết nối")) {
      return "Backend đã kết nối. Đang chờ wake word...";
    }

    return rawStatus || "Đang chờ...";
  };

  const playWakeSound = () => {
    const now = Date.now();

    // Chặn phát trùng nếu backend gửi nhiều message gần nhau
    if (now - lastWakeSoundTimeRef.current < 1500) {
      return;
    }

    lastWakeSoundTimeRef.current = now;

    const sounds = [
      "/sounds/wakeup_1.wav",
      "/sounds/wakeup_2.wav",
      "/sounds/wakeup_3.wav",
    ];

    const randomSound = sounds[Math.floor(Math.random() * sounds.length)];
    const audio = new Audio(randomSound);

    audio.volume = 1.0;

    audio.play().catch((error) => {
      console.log("Không phát được âm thanh:", error);
    });
  };

  const start = async () => {
    if (running) return;

    const socket = new WebSocket("ws://127.0.0.1:8000/ws/audio");
    socketRef.current = socket;

    socket.onopen = async () => {
      try {
        setRunning(true);
        setStatus("Đang xin quyền microphone...");

        audioControllerRef.current = await startAudioStream(socket);

        setStatus("Đang chờ wake word: hey mixi");
      } catch (error) {
        setStatus("Không thể truy cập microphone");
        setRunning(false);
        socket.close();
      }
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.status) {
        const cleanedStatus = cleanStatus(data.status);
        setStatus(cleanedStatus);

        if (data.status.toLowerCase().includes("đã phát hiện wake word")) {
          playWakeSound();
        }
      }

      if (data.wake_score !== undefined && data.wake_score !== "") {
        setWakeScore(data.wake_score);
      }

      const hasFinalResult =
        data.text &&
        data.intent &&
        data.confidence !== undefined &&
        data.confidence !== "";

      if (hasFinalResult) {
        setLastResult({
          text: data.text || "",
          processedText: data.processed_text || "",
          intent: data.intent || "",
          confidence: data.confidence || "",
        });
      }
    };

    socket.onerror = () => {
      setStatus("Lỗi kết nối backend");
    };

    socket.onclose = () => {
      setRunning(false);
      setStatus("Đã dừng nghe");
    };
  };

  const stop = () => {
    audioControllerRef.current?.stop();
    socketRef.current?.close();

    setRunning(false);
    setStatus("Đã dừng nghe");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-cyan-950 text-white flex items-center justify-center px-6 py-10">
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
        <div className="flex flex-col justify-center">
          <p className="text-cyan-300 font-semibold mb-3">
            Vietnamese Wake Word System
          </p>

          <h1 className="text-5xl font-black leading-tight mb-5">
            Hey Mixi <br />
            Voice Assistant
          </h1>

          <p className="text-slate-300 text-lg mb-8">
            Hệ thống nhận diện wake word, chuyển giọng nói thành văn bản và phân loại ý định tiếng Việt theo thời gian thực.
          </p>

          <div className="flex gap-4">
            <button
              onClick={start}
              disabled={running}
              className="px-6 py-4 rounded-2xl bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-700 font-bold transition shadow-lg shadow-cyan-500/20"
            >
              Bắt đầu nghe
            </button>

            <button
              onClick={stop}
              disabled={!running}
              className="px-6 py-4 rounded-2xl bg-red-500 hover:bg-red-400 disabled:bg-slate-700 font-bold transition"
            >
              Dừng
            </button>
          </div>
        </div>

        <div className="bg-white/10 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl">
          <div className="flex justify-center mb-8">
            <div
              className={`w-40 h-40 rounded-full flex items-center justify-center text-6xl ${
                running
                  ? "bg-cyan-500 animate-pulse shadow-2xl shadow-cyan-500/40"
                  : "bg-slate-700"
              }`}
            >
              🎙️
            </div>
          </div>

          <div className="space-y-4">
            <InfoCard title="Trạng thái hiện tại" value={status} />

            <InfoCard
              title="Wake score gần nhất"
              value={wakeScore !== "" ? wakeScore : "..."}
            />

            <div className="bg-slate-950/70 border border-cyan-400/20 rounded-2xl p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-cyan-300 text-sm font-semibold">
                  Kết quả gần nhất
                </p>

                <span className="text-xs px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-400/20">
                  Latest
                </span>
              </div>

              <div className="space-y-4">
                <ResultItem
                  title="Văn bản nhận diện"
                  value={lastResult.text || "Chưa có kết quả"}
                />

                <ResultItem
                  title="Văn bản sau xử lý"
                  value={lastResult.processedText || "..."}
                />

                <ResultItem
                  title="Intent"
                  value={lastResult.intent || "..."}
                  highlight
                />

                <ResultItem
                  title="Confidence"
                  value={
                    lastResult.confidence !== ""
                      ? Number(lastResult.confidence).toFixed(4)
                      : "..."
                  }
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoCard({ title, value }) {
  return (
    <div className="bg-slate-950/60 border border-white/10 rounded-2xl p-5">
      <p className="text-slate-400 text-sm mb-1">{title}</p>
      <p className="text-xl font-semibold text-white">{value}</p>
    </div>
  );
}

function ResultItem({ title, value, highlight }) {
  return (
    <div>
      <p className="text-slate-500 text-sm mb-1">{title}</p>
      <p
        className={`text-lg font-semibold ${
          highlight ? "text-cyan-300" : "text-white"
        }`}
      >
        {value}
      </p>
    </div>
  );
}