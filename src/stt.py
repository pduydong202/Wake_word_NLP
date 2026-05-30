from transformers import pipeline


class PhoWhisperSTT:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        print("[PhoWhisper] Loading model:", config.phowhisper_model_name)

        pipeline_device = 0 if device.type == "cuda" else -1

        self.asr = pipeline(
            task="automatic-speech-recognition",
            model=config.phowhisper_model_name,
            device=pipeline_device,
            model_kwargs={
                "use_safetensors": False
            }
        )

        print("[PhoWhisper] Loaded.")

    def transcribe(self, audio) -> str:
        result = self.asr(
            {
                "array": audio,
                "sampling_rate": self.config.sample_rate,
            },
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
            },
        )

        text = result.get("text", "").strip()
        return text