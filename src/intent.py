import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize


class PhoBERTIntentClassifier:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        print("[PhoBERT] Loading model:", config.phobert_model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(config.phobert_model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.phobert_model_dir
        ).to(device)

        self.model.eval()
        self.id2label = self.model.config.id2label

        print("[PhoBERT] Loaded.")
        print("[PhoBERT] Labels:", self.id2label)
        print("[PhoBERT] Word segmentation: enabled")

    def preprocess(self, text: str) -> str:
        text = text.strip().lower()

        if self.config.use_word_segmentation:
            text = word_tokenize(text, format="text")

        return text

    @torch.no_grad()
    def predict(self, text: str):
        processed_text = self.preprocess(text)

        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_text_length,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

        label = self.id2label.get(pred_id, str(pred_id))

        all_probs = {
            self.id2label.get(i, str(i)): float(probs[i].item())
            for i in range(len(probs))
        }

        return {
            "raw_text": text,
            "processed_text": processed_text,
            "label": label,
            "label_id": pred_id,
            "confidence": confidence,
            "all_probs": all_probs,
        }