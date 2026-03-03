# Voice-Based Personalized Shopping Assistant

An AI-powered voice shopping assistant that uses **Machine Learning** and **Natural Language Processing** to recommend products from an Amazon product catalog based on voice commands.

## Features

- **Voice Input** — Speak your shopping queries using your browser's microphone
- **Google Speech Recognition API** — Converts speech to text in real-time
- **ML Category Classifier (LinearSVC)** — Trained on 490K+ products with 92% accuracy
- **Query Understanding** — Extracts colors, gender, sizes from natural language
- **TF-IDF + Cosine Similarity** — Ranks products with attribute-boosted scoring
- **Modern Glassmorphism UI** — Premium responsive web interface

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| ML/NLP | Scikit-learn (TF-IDF, LinearSVC, Cosine Similarity) |
| Speech | Google Speech Recognition API |
| Frontend | HTML5, CSS3, JavaScript |
| Audio | Web Audio API (PCM WAV encoding) |
| Dataset | Amazon Products (490K+ items) |

## Architecture

```
voice-shopping-assistant/
├── app/
│   ├── app.py                 # Flask backend server
│   ├── templates/
│   │   └── index.html         # Frontend UI
│   └── static/
│       ├── css/style.css      # Glassmorphism styling
│       └── js/main.js         # Audio recording & UI logic
├── src/
│   ├── nlp_engine.py          # ML-powered recommendation engine
│   └── speech_agent.py        # Google Speech API integration
├── data/
│   └── Amazon-Products.csv    # Dataset (download separately)
├── requirements.txt
└── README.md
```

## ML Pipeline (3 Stages)

### Stage 1: Query Understanding
Parses natural language to extract structured attributes:
```
"yellow shirt for boys" → colors: [yellow], gender: [boys], product: [shirt]
```

### Stage 2: Category Classifier (LinearSVC)
- Supervised ML model trained on product names → categories
- 92% accuracy across 20 product categories
- Predicts which category matches user intent

### Stage 3: Attribute-Boosted TF-IDF Similarity
- Color matches: **2.5x score boost**
- Gender matches: **1.5x score boost**
- Product term matches: incremental boost
- Cosine similarity ranking within predicted category

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/voice-shopping-assistant.git
cd voice-shopping-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `Amazon-Products.csv` from [Kaggle](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset) and place it in the `data/` folder.

### 4. Run the application
```bash
python app/app.py
```

### 5. Open in browser
Navigate to `http://127.0.0.1:5000`

## Usage

1. Click the **microphone button** and speak your shopping query
2. Or use the **text search** at the bottom of the page
3. View AI-recommended products with match scores

### Example Queries
- "yellow shirt"
- "red shoes for men"
- "black watch"
- "blue dress for women"
- "running shoes"
- "headphones"

## Screenshots

### Voice Search
Search by speaking into your microphone with real-time transcription.

### Product Recommendations
AI-ranked product cards with match scores, images, prices, and ratings.

## License

This project is for educational purposes.
