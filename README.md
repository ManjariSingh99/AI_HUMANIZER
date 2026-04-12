# 🤖 AI Content Humanization System

A production-oriented AI system designed to **detect and humanize AI-generated text** using NLP techniques and transformer-based models.

---

## 🚀 Overview

This project focuses on solving a real-world problem:

> Detect whether content is AI-generated and intelligently rewrite it to appear more human-like.

It combines:
- AI detection (perplexity, repetition, sentence variation)
- Transformer-based rewriting (T5)
- Decision-based rewrite engine

---

## 🧠 Core Features

### 🔍 AI Detection Engine
- Perplexity-based scoring (GPT-2)
- Repetition analysis
- Sentence structure variation
- Combined AI probability scoring

### ✍️ Rewrite Engine
- T5-based paraphrasing
- Threshold-based decision system
- Conditional rewriting (only when needed)

### ⚙️ Backend Architecture
- Django-based modular backend
- Service-layer separation
- Scalable processing pipeline

---

## 🏗️ Tech Stack

- **Backend:** Django
- **ML/NLP:** Transformers (Hugging Face), PyTorch
- **Models:** GPT-2 (detection), T5 (rewriting)
- **Language:** Python


---

## ⚡ How It Works

1. Input text
2. Detection engine calculates:
   - AI score
   - Perplexity
   - Repetition
3. Rewrite engine decides:
   - Rewrite or not
4. If needed → T5 generates improved version

---

## 🧪 Current Status

- ✅ Phase 1–4 Completed
- 🚧 Phase 5 (Intelligent Rewrite Optimization) in progress

---

## 🌐 Live Demo

👉 https://huggingface.co/spaces/ManjariSingh99/ai-humanizer

---

## 📌 Future Improvements

- Iterative rewrite optimization
- Strategy switching (formal, casual, etc.)
- Better scoring models
- API layer (FastAPI)

---

## 👨‍💻 Author

Built as part of a learning journey in:
- Backend engineering
- NLP systems
- AI application architecture

---

## ⭐ Note

This is an evolving project — actively being improved with better models, logic, and system design.
