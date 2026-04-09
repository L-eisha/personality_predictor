
import os
import re
import json
import pdfminer.high_level
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

TRAIT_KEYWORDS = {
    "Openness": [
        "creative", "innovative", "imaginative", "curious", "artistic",
        "inventive", "visionary", "design", "research", "explore",
        "novel", "experiment", "idea", "strategy", "brainstorm",
        "diverse", "flexible", "open-minded", "concept", "prototype"
    ],
    "Conscientiousness": [
        "organized", "detail-oriented", "responsible", "reliable",
        "punctual", "planned", "systematic", "thorough", "disciplined",
        "goal-oriented", "efficient", "deadline", "accurate", "structured",
        "managed", "documented", "tracked", "scheduled", "completed", "delivered"
    ],
    "Extraversion": [
        "leadership", "team", "communication", "presentation", "public speaking",
        "networking", "collaborated", "mentored", "coordinated", "negotiated",
        "motivated", "enthusiastic", "outgoing", "sociable", "client",
        "stakeholder", "event", "drove", "facilitated", "led"
    ],
    "Agreeableness": [
        "supportive", "empathy", "cooperative", "helpful", "patient",
        "kind", "collaborative", "volunteer", "community", "resolved",
        "assisted", "trained", "guided", "feedback", "inclusive",
        "listened", "understood", "caring", "trustworthy", "mediated"
    ],
    "Neuroticism (Stability)": [
        "calm", "composed", "resilient", "adaptable", "stable",
        "confident", "focused", "stress", "pressure", "challenge",
        "overcame", "handled", "managed crisis", "problem-solving",
        "recovered", "balanced", "consistent", "steady", "reliable", "persistence"
    ]
}

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    text = pdfminer.high_level.extract_text(pdf_path)
    return text.lower()

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().lower()

def preprocess(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z\s\-]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(filtered)

def score_traits(text: str) -> dict:
    scores = {}
    words = text.split()
    total_words = len(words) if len(words) > 0 else 1
    for trait, keywords in TRAIT_KEYWORDS.items():
        count = sum(text.count(kw) for kw in keywords)
        raw_score = (count / total_words) * 1000
        scores[trait] = min(round(raw_score, 2), 100.0)
    return scores

def score_to_label(score: float) -> str:
    if score >= 5.0:
        return "High ✅"
    elif score >= 2.0:
        return "Moderate 🟡"
    else:
        return "Low 🔴"

def extract_sections(text: str) -> dict:
    sections = {}
    patterns = {
        "Skills":      r'(skills?)(.*?)(experience|education|project|certif|$)',
        "Experience":  r'(experience|work history)(.*?)(education|skills?|project|certif|$)',
        "Education":   r'(education|academic)(.*?)(skills?|experience|project|certif|$)',
        "Projects":    r'(projects?)(.*?)(skills?|experience|education|certif|$)',
    }
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = match.group(2).strip()[:300]
        else:
            sections[section] = "Not found"
    return sections

def generate_report(pdf_path: str):
    print("\n" + "="*60)
    print("   🧠 PERSONALITY PREDICTION SYSTEM — CV ANALYSIS")
    print("="*60)

    ext = os.path.splitext(pdf_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(pdf_path)
    elif ext == ".txt":
        raw_text = extract_text_from_txt(pdf_path)
    else:
        print("❌ Unsupported file type. Please use .pdf or .txt")
        return

    clean_text = preprocess(raw_text)
    scores = score_traits(clean_text)

    print(f"\n📄 File Analyzed : {os.path.basename(pdf_path)}")
    print(f"📝 Word Count    : {len(clean_text.split())} (after cleaning)\n")

    print("─"*60)
    print(f"{'TRAIT':<30} {'SCORE':>8}   {'LEVEL'}")
    print("─"*60)
    for trait, score in scores.items():
        label = score_to_label(score)
        bar = "█" * int(score * 2) + "░" * (20 - int(score * 2))
        bar = bar[:20]
        print(f"{trait:<30} {score:>6.2f}   {label}")
        print(f"  [{bar}]")
    print("─"*60)

    dominant = max(scores, key=scores.get)
    print(f"\n🏆 Dominant Trait : {dominant}")

    summaries = {
        "Openness": "This candidate is creative and innovative. Great for R&D, design, or strategy roles.",
        "Conscientiousness": "Highly organized and reliable. Excellent fit for operations, finance, or project management.",
        "Extraversion": "Strong communicator and team leader. Well-suited for sales, HR, or client-facing roles.",
        "Agreeableness": "Empathetic and cooperative. Ideal for support, training, or collaborative environments.",
        "Neuroticism (Stability)": "Calm under pressure and resilient. Good fit for high-stakes or fast-paced roles."
    }
    print(f"📋 Summary        : {summaries.get(dominant, 'N/A')}")

    print("\n" + "─"*60)
    print("📂 CV SECTION PREVIEW")
    print("─"*60)
    sections = extract_sections(raw_text)
    for section, content in sections.items():
        preview = content[:120].replace('\n', ' ') if content != "Not found" else "Not found"
        print(f"\n[{section}]\n  {preview}...")

    report = {
        "file": os.path.basename(pdf_path),
        "scores": scores,
        "dominant_trait": dominant,
        "summary": summaries.get(dominant, "N/A")
    }
    with open("personality_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "="*60)
    print(f"✅ Report saved to: personality_report.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("\n📌 Usage: python personality_predictor.py <path_to_cv.pdf>")
        print("🔁 Running in DEMO MODE with sample CV text...\n")
        sample_cv = """
        John Doe - Software Engineer
        Skills: Python, Machine Learning, Data Analysis, Communication, Leadership
        Experience: Led a team of 5 engineers to deliver a creative AI solution on time.
        Managed project deadlines, coordinated with stakeholders, mentored junior developers.
        Volunteered at community coding bootcamp. Calm under pressure, detail-oriented.
        Organized sprint planning, documented technical specs thoroughly.
        Explored novel deep learning architectures and published research findings.
        """
        with open("sample_cv.txt", "w") as f:
            f.write(sample_cv)
        generate_report("sample_cv.txt")
    else:
        generate_report(sys.argv[1])