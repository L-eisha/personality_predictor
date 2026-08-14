# Personality Predictor

An NLP-powered web application that predicts Big Five (OCEAN) personality traits from CVs and resumes.

## Overview

Personality Predictor analyzes the language in your CV or resume to score your personality across the Big Five dimensions:

- **Openness** — creativity, curiosity, intellectual exploration
- **Conscientiousness** — organization, reliability, attention to detail
- **Extraversion** — leadership, communication, social confidence
- **Agreeableness** — empathy, cooperation, supportiveness
- **Neuroticism (Stability)** — emotional resilience, composure under pressure

The tool uses weighted keyword matching with word-boundary detection and sigmoid normalization to produce reliable trait scores from 5 to 95.

## Features

- **PDF and TXT upload** — extract text from uploaded CVs
- **Text paste** — paste CV content directly
- **Demo CV** — test with a built-in sample resume
- **Weighted keyword analysis** — 50+ keywords per trait with importance weights (1.0 / 1.5 / 2.0)
- **Word-boundary matching** — prevents false positives from substring matches
- **Sigmoid scoring** — smooth normalization for realistic score distribution
- **Confidence indicator** — rates analysis reliability based on text length and score spread
- **Career recommendations** — suggests 5 career paths based on dominant trait
- **CV section detection** — automatically identifies Summary, Experience, Education, Skills, and more
- **Interactive charts** — radar chart and horizontal bar chart (Plotly)
- **JSON report download** — export full analysis results
- **Dark theme UI** — glass-morphism design with animations

## Project Structure

```
personality_predictor/
├── app.py                    # Streamlit web application
├── personality_predictor.py  # CLI version for terminal use
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/L-eisha/personality_predictor.git
cd personality_predictor
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

### Run from Command Line

```bash
python personality_predictor.py path/to/resume.pdf
```

Run without arguments for demo mode:

```bash
python personality_predictor.py
```

## Tech Stack

- **Streamlit** — web interface
- **NLTK** — tokenization and stopword removal
- **pdfminer.six** — PDF text extraction
- **Plotly** — interactive data visualization

## How It Works

1. **Text extraction** — reads PDF or TXT content
2. **Preprocessing** — lowercases, removes special characters, tokenizes, filters stopwords
3. **Trait scoring** — matches keywords using regex word boundaries, applies importance weights, normalizes with sigmoid function
4. **Confidence** — evaluates reliability based on word count and trait score spread
5. **Career mapping** — recommends careers aligned with the dominant personality trait
6. **Report** — displays results with charts, cards, and downloadable JSON

## Author

**Leisha Choudhary** — [@L-eisha](https://github.com/L-eisha)
