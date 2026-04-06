import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import Counter
import numpy as np
import re


class AIDetector:
    _model = None
    _tokenizer = None

    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if AIDetector._model is None:
            AIDetector._tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
            AIDetector._model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

        self.tokenizer = AIDetector._tokenizer
        self.model = AIDetector._model

    def calculate_perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        return torch.exp(loss).item()

    def normalize_perplexity(self, ppl):
        return min(ppl / 100, 1)

    def repetition_score(self, text):
        words = text.lower().split()
        counts = Counter(words)

        total_repeated = sum(count - 1 for count in counts.values() if count > 1)
        return total_repeated / len(words)

    def sentence_variation(self, text):
        sentences = re.split(r'[.!?]', text)
        lengths = [len(s.split()) for s in sentences if len(s.strip()) > 0]

        if len(lengths) < 2:
            return 0

        return np.std(lengths)

    def detect(self, text):
        if len(text.split()) < 20:
            return {
                "perplexity": None,
                "repetition_score": None,
                "sentence_variation": None,
                "ai_score": None,
                "ai_probability": None,
                "label": "Low confidence (text too short)",
                "confidence": "Low confidence"
            }

        ppl = self.calculate_perplexity(text)
        rep = self.repetition_score(text)
        var = self.sentence_variation(text)

        norm_ppl = self.normalize_perplexity(ppl)
        norm_rep = min(rep * 2, 1)
        norm_var = 1 / (var + 1)

        ai_score = (
            (1 - norm_ppl) * 0.5 +
            norm_rep * 0.3 +
            norm_var * 0.2
        )

        # Override rules
        if ppl < 12:
            label = "Highly Likely AI-generated"
        elif ppl > 60:
            label = "Highly Likely Human-written"
        else:
            if ai_score > 0.6:
                label = "Likely AI-generated"
            elif ai_score < 0.4:
                label = "Likely Human-written"
            else:
                label = "Possibly AI-generated (Low confidence)"

        # Confidence calculation
        if ai_score > 0.7 or ai_score < 0.3:
            confidence = "High confidence"
        elif 0.4 < ai_score < 0.6:
            confidence = "Low confidence — ambiguous text"
        else:
            confidence = "Moderate confidence"

        return {
            "perplexity": round(ppl, 2),
            "repetition_score": round(rep, 2),
            "sentence_variation": round(var, 2),
            "ai_score": round(ai_score, 2),
            "ai_probability": round(ai_score * 100, 2),
            "label": label,
            "confidence": confidence
        }