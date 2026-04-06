# AI Content Humanization & Quality Enhancement System

## Description
AI_HUMANIZER is a Django-based application that detects AI-generated content and rewrites it to produce **more human-like, readable, and natural text**.  
It includes features like AI content detection, iterative rewriting, and real-time scoring for quality enhancement.

## Tech Stack
- **Backend:** Django  
- **AI Models:** HuggingFace Transformers (T5)  
- **Task Queue:** Celery + Redis (planned)  
- **Database:** SQLite (for demo)  
- **Frontend:** Django Templates  

## Key Features
- AI content detection (perplexity, repetition, AI score)  
- Iterative humanization of AI-generated text  
- Live scoring to ensure quality improvement  
- Clean and professional UI  

## Demo
![Home Page Screenshot](screenshots/home.png)  
*Add screenshots in a folder `screenshots/` in your repo.*

## Installation & Setup (Local)
```bash
# Clone the repo
git clone https://github.com/<your-username>/AI_HUMANIZER.git
cd AI_HUMANIZER

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env)
SECRET_KEY=<your-secret-key>
DEBUG=True

# Run migrations
python manage.py migrate

# Run the server
python manage.py runserver
