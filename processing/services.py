import re
from ai_services.detector import AIDetector
from processing.rewrite_engine import RewriteEngine


class TextProcessingService:
    def __init__(self, text):
        self.text = text
        self.detector = AIDetector()
        self.engine = RewriteEngine(self.detector)

    def process(self):
        cleaned_text = self._clean_text(self.text)

        # 🔥 Use engine (decision + rewrite)
        result = self.engine.process(cleaned_text)

        return {
            "original_text": self.text,
            "detection": result.get("detection"),
            "final_text": result.get("final_text"),
            "rewritten": result.get("rewritten"),
            "reason": result.get("reason")
        }

    # ---------------------------
    # 🔥 CLEAN TEXT
    # ---------------------------
    def _clean_text(self, text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces
        return text