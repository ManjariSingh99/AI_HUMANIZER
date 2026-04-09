from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class RewriteEngine:
    def __init__(self, detector):
        self.detector = detector
        self.AI_THRESHOLD = 0.65
        self.MIN_WORDS = 10

        # Load model once
        self.device = "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model.to(self.device)

    def process(self, text):
        # Step 1: basic validation
        if len(text.split()) < self.MIN_WORDS:
            return self._no_rewrite(text, "Text too short", None)

        # Step 2: detect AI score
        detection = self.detector.detect(text)
        ai_score = detection.get("ai_score")

        if ai_score is None:
            return self._no_rewrite(text, "Low confidence detection", detection)

        # Step 3: decision logic
        if ai_score < self.AI_THRESHOLD:
            return self._no_rewrite(text, "Already human-like", detection)

        # Step 4: rewrite
        rewritten_text = self._rewrite(text)

        return {
            "detection": detection,
            "final_text": rewritten_text,
            "rewritten": rewritten_text != text,
            "reason": "Rewritten due to high AI probability"
        }

    def _rewrite(self, text):
        prompt = f"paraphrase: {text}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _no_rewrite(self, text, reason, detection):
        return {
            "detection": detection,
            "final_text": text,
            "rewritten": False,
            "reason": reason
        }