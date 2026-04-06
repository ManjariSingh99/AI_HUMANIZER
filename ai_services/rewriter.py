import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast
from difflib import SequenceMatcher


class TextRewriter:
    _model = None
    _tokenizer = None
    _initialized = False

    def __init__(self, model_name="Vamsi/T5_Paraphrase_Paws"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not TextRewriter._initialized:
            print("Loading T5 model safely...")

            try:
                # Try fast tokenizer first
                try:
                    tokenizer = T5TokenizerFast.from_pretrained(model_name)
                except Exception:
                    print("Fast tokenizer failed, falling back...")
                    tokenizer = T5Tokenizer.from_pretrained(model_name)

                model = T5ForConditionalGeneration.from_pretrained(model_name)
                model = model.to(self.device)

                TextRewriter._tokenizer = tokenizer
                TextRewriter._model = model
                TextRewriter._initialized = True

                print("✅ Model loaded successfully")

            except Exception as e:
                print("❌ Model loading failed:", str(e))
                raise RuntimeError("Failed to load T5 model.")

        self.tokenizer = TextRewriter._tokenizer
        self.model = TextRewriter._model

    # ---------------------------
    # 🔥 MAIN REWRITE FUNCTION
    # ---------------------------
    def rewrite(self, text, strategy=None):
        if len(text.split()) < 10:
            return {
                "rewritten_text": text,
                "rewrite_changed": False,
                "quality_score": 1.0
            }

        sentences = self.split_sentences(text)
        rewritten_sentences = []
        similarities = []

        for sentence in sentences:
            if len(sentence.split()) < 5:
                rewritten_sentences.append(sentence)
                continue

            rewritten = self._rewrite_single(sentence, strategy)

            similarity = SequenceMatcher(
                None,
                sentence.lower(),
                rewritten.lower()
            ).ratio()

            similarities.append(similarity)
            rewritten_sentences.append(rewritten)

        final_text = " ".join(rewritten_sentences).strip()

        return {
            "rewritten_text": final_text,
            "rewrite_changed": True,
            "quality_score": sum(similarities) / len(similarities) if similarities else 1.0
        }

    # ---------------------------
    # 🔥 SINGLE SENTENCE REWRITE
    # ---------------------------
    def _rewrite_single(self, text, strategy=None):
        strategies = {
            "paraphrase": f"paraphrase: {text} </s>",
            "casual": f"paraphrase: make it more natural and human-like: {text} </s>",
            "simplify": f"paraphrase: simplify: {text} </s>",
            "expand": f"paraphrase: add slight detail: {text} </s>"
        }

        strategy_list = ["paraphrase", "casual", "simplify", "expand"]

        best_output = text
        best_score = 1.0  # lower = better (less similar)

        for attempt in range(4):
            current_strategy = strategy_list[attempt % len(strategy_list)]
            prompt = strategies[current_strategy]

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(self.device)

            if attempt < 3:
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(len(text.split()) + 40, 128),
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3
                )
            else:
                # fallback deterministic
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(len(text.split()) + 40, 128),
                    num_beams=5,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )

            rewritten_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            rewritten_text = self.clean_output(rewritten_text)

            if self.is_valid_output(text, rewritten_text):
                similarity = SequenceMatcher(
                    None,
                    text.lower(),
                    rewritten_text.lower()
                ).ratio()

                # 🔥 Keep BEST (most different but valid)
                if similarity < best_score:
                    best_score = similarity
                    best_output = rewritten_text

        return best_output

    # ---------------------------
    # 🔥 VALIDATION
    # ---------------------------
    def is_valid_output(self, original, rewritten):
        if not rewritten:
            return False

        original = original.strip()
        rewritten = rewritten.strip()
        rewritten_lower = rewritten.lower()

        # Invalid outputs
        if rewritten_lower in ["", "false", "true", "none", "null"]:
            return False

        # Reject instruction leakage
        if any(rewritten_lower.startswith(word) for word in ["rewrite", "paraphrase", "rephrase"]):
            return False

        # Structural change check
        if original.split() == rewritten.split():
            return False

        original_lower = original.lower()

        similarity = SequenceMatcher(
            None,
            original_lower,
            rewritten_lower
        ).ratio()

        original_keywords = set(original_lower.split())
        rewritten_keywords = set(rewritten_lower.split())

        if len(original_keywords) == 0:
            return False

        overlap = len(original_keywords & rewritten_keywords) / len(original_keywords)

        # Semantic guard
        if overlap < 0.3 and similarity < 0.5:
            return False

        # Similarity constraints
        if similarity > 0.7:
            return False

        if similarity < 0.4:
            return False

        # Length constraints
        if len(rewritten.split()) < len(original.split()) * 0.6:
            return False

        if len(rewritten.split()) > len(original.split()) * 1.8:
            return False

        return True

    # ---------------------------
    # 🔥 CLEAN OUTPUT
    # ---------------------------
    def clean_output(self, text):
        text = text.strip()
        text = " ".join(text.split())

        bad_starts = [
            "paraphrase",
            "rewrite",
            "rephrase",
            "this text",
            "the text"
        ]

        lower_text = text.lower()

        for phrase in bad_starts:
            if lower_text.startswith(phrase):
                if ":" in text:
                    text = text.split(":", 1)[-1].strip()
                else:
                    text = text[len(phrase):].strip()

        # Remove leading junk chars
        text = text.lstrip(" ,.-:")

        return text

    # ---------------------------
    # 🔥 SPLIT SENTENCES
    # ---------------------------
    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]


# ---------------------------
# 🔥 SINGLETON INSTANCE
# ---------------------------
rewriter_instance = TextRewriter()