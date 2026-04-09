import gradio as gr

from ai_engine.detector import AIDetector
from ai_engine.rewrite_engine import RewriteEngine

# Initialize once (IMPORTANT for performance)
detector = AIDetector()
rewriter = RewriteEngine(detector)


def process_text(text):
    # Input validation
    if not text or not text.strip():
        return "Enter text", "", "", ""

    try:
        # Run full pipeline (detect + decision + rewrite)
        result = rewriter.process(text)

        detection = result.get("detection") or {}

        ai_score = detection.get("ai_score", 0)
        perplexity = detection.get("perplexity", 0)
        repetition = detection.get("repetition", 0)

        final_text = result.get("final_text", "")
        reason = result.get("reason", "")

        return (
            f"{round(ai_score, 3)} ({reason})",
            round(perplexity, 3),
            round(repetition, 3),
            final_text
        )

    except Exception as e:
        return "Error", "Error", "Error", str(e)


# Gradio UI
interface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your text here..."
    ),
    outputs=[
        gr.Text(label="AI Score (with decision)"),
        gr.Number(label="Perplexity"),
        gr.Number(label="Repetition"),
        gr.Textbox(label="Final Output Text")
    ],
    title="AI Content Humanizer",
    description="Detect and intelligently rewrite AI-generated text"
)


if __name__ == "__main__":
    interface.launch(share=True)