from ai_services.rewriter import rewriter_instance

class RewriteEngine:
    def __init__(self, detector):
        self.detector = detector
        self.rewriter = rewriter_instance
        self.AI_THRESHOLD = 0.65
        self.MIN_WORDS = 10

    def process(self, text):
        if len(text.split()) < self.MIN_WORDS:
            return self._no_rewrite(text, "Text too short", None)

        detection = self.detector.detect(text)
        ai_score = detection.get("ai_score")

        if ai_score is None:
            return self._no_rewrite(text, "Low confidence detection", detection)

        if ai_score < self.AI_THRESHOLD:
            return self._no_rewrite(text, "Already human-like", detection)

        rewrite_result = self.rewriter.rewrite(text)

        return {
            "detection": detection,
            "final_text": rewrite_result["rewritten_text"],
            "rewritten": rewrite_result["rewrite_changed"],
            "reason": "Rewritten due to high AI probability"
        }

    def _no_rewrite(self, text, reason, detection):
        return {
            "detection": detection,
            "final_text": text,
            "rewritten": False,
            "reason": reason
        }