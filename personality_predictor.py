import os
import re
import json
import math
import pdfminer.high_level
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

TRAIT_KEYWORDS = {
    "Openness": {
        "creative": 2.0, "innovative": 2.0, "imaginative": 2.0, "visionary": 2.0,
        "inventive": 1.5, "original": 1.5, "artistic": 1.5, "curious": 1.5,
        "research": 1.5, "explore": 1.5, "experiment": 1.5, "discover": 1.5,
        "investigate": 1.0, "analyze": 1.0, "study": 1.0, "learn": 1.0,
        "design": 1.0, "prototype": 1.5, "brainstorm": 1.5, "ideation": 1.5,
        "concept": 1.0, "architecture": 1.0, "blueprint": 1.0,
        "strategy": 1.0, "strategic": 1.0, "vision": 1.0, "roadmap": 1.0,
        "initiative": 1.0, "pioneered": 2.0, "spearheaded": 1.5,
        "diverse": 1.0, "flexible": 1.0, "versatile": 1.0, "adaptable": 1.0,
        "open-minded": 2.0, "interdisciplinary": 1.5, "cross-functional": 1.0,
        "novel": 1.5, "unconventional": 1.5, "cutting-edge": 1.5,
        "state-of-the-art": 1.5, "emerging": 1.0, "disruptive": 1.5,
        "transformative": 1.5, "reimagined": 2.0, "revolutionized": 2.0,
        "solution": 1.0, "solved": 1.0, "devised": 1.5, "crafted": 1.0,
        "invented": 2.0, "conceived": 1.5, "envisioned": 1.5,
        "innovation": 2.0, "creativity": 2.0
    },
    "Conscientiousness": {
        "organized": 2.0, "systematic": 2.0, "structured": 1.5, "methodical": 2.0,
        "meticulous": 2.0, "orderly": 1.5, "categorized": 1.0,
        "detail-oriented": 2.0, "thorough": 1.5, "accurate": 1.5, "precise": 1.5,
        "rigorous": 1.5, "careful": 1.0, "quality": 1.0, "audit": 1.0,
        "responsible": 1.5, "reliable": 1.5, "dependable": 1.5, "accountable": 1.5,
        "committed": 1.0, "dedicated": 1.0, "diligent": 2.0, "conscientious": 2.0,
        "planned": 1.5, "scheduled": 1.0, "punctual": 1.5, "deadline": 1.5,
        "on-time": 1.5, "timely": 1.0, "milestone": 1.0, "timeline": 1.0,
        "goal-oriented": 2.0, "achieved": 1.0, "accomplished": 1.0, "completed": 1.0,
        "delivered": 1.5, "exceeded": 1.5, "surpassed": 1.5, "attained": 1.0,
        "documented": 1.5, "tracked": 1.0, "monitored": 1.0, "reported": 1.0,
        "managed": 1.0, "maintained": 1.0, "implemented": 1.0, "executed": 1.0,
        "efficient": 1.5, "optimized": 1.5, "streamlined": 1.5, "automated": 1.0,
        "disciplined": 2.0, "compliant": 1.0, "standardized": 1.0,
        "consistency": 1.5, "persistent": 1.5
    },
    "Extraversion": {
        "leadership": 2.0, "led": 1.5, "directed": 1.0, "managed team": 2.0,
        "supervised": 1.0, "headed": 1.5, "captained": 1.5,
        "communication": 2.0, "presented": 1.5, "presentation": 1.5,
        "public speaking": 2.0, "articulate": 1.5, "eloquent": 1.5,
        "persuasive": 1.5, "influential": 1.5, "spokesperson": 2.0,
        "team": 1.0, "collaborated": 1.0, "coordinated": 1.0,
        "cross-team": 1.5, "interdepartmental": 1.0,
        "networking": 2.0, "sociable": 2.0, "outgoing": 2.0,
        "charismatic": 2.0, "personable": 1.5, "approachable": 1.0,
        "relationship": 1.0, "rapport": 1.5, "engaged": 1.0,
        "client": 1.0, "stakeholder": 1.0, "customer": 1.0,
        "client-facing": 1.5, "negotiated": 1.5, "pitched": 1.5,
        "mentored": 1.5, "coached": 1.5, "motivated": 1.5,
        "enthusiastic": 2.0, "energetic": 1.5, "dynamic": 1.0,
        "proactive": 1.5, "drove": 1.0, "facilitated": 1.0,
        "championed": 1.5, "advocated": 1.0, "promoted": 1.0,
        "organized event": 1.5, "hosted": 1.5, "conference": 1.0,
        "workshop": 1.0, "seminar": 1.0
    },
    "Agreeableness": {
        "empathy": 2.0, "empathetic": 2.0, "compassionate": 2.0,
        "caring": 1.5, "understanding": 1.0, "sensitive": 1.0,
        "considerate": 1.5, "thoughtful": 1.0, "kind": 1.5,
        "cooperative": 2.0, "collaborative": 1.5, "team player": 2.0,
        "supportive": 1.5, "helpful": 1.5, "assisted": 1.0,
        "trustworthy": 2.0, "honest": 1.0, "integrity": 1.0,
        "diplomatic": 1.5, "tactful": 1.5, "harmonious": 1.5,
        "respectful": 1.0, "courteous": 1.0, "polite": 1.0,
        "resolved": 1.0, "mediated": 2.0, "negotiated peace": 2.0,
        "de-escalated": 2.0, "reconciled": 1.5,
        "volunteer": 2.0, "community": 1.0, "charity": 1.5,
        "trained": 1.0, "guided": 1.0,
        "nurtured": 1.5, "developed talent": 1.5,
        "patient": 1.5, "listened": 1.5, "receptive": 1.5,
        "feedback": 1.0, "inclusive": 1.5, "welcoming": 1.0,
        "accommodating": 1.0, "tolerant": 1.0, "forgiving": 1.5,
        "generous": 1.5, "altruistic": 2.0, "selfless": 2.0
    },
    "Neuroticism (Stability)": {
        "calm": 2.0, "composed": 2.0, "poised": 1.5, "collected": 1.5,
        "unflappable": 2.0, "level-headed": 2.0, "even-tempered": 2.0,
        "resilient": 2.0, "persevered": 1.5, "endured": 1.0,
        "overcame": 1.5, "bounced back": 2.0, "recovered": 1.0,
        "withstood": 1.5, "tenacious": 1.5,
        "adaptable": 1.5, "adjusted": 1.0, "pivoted": 1.5,
        "agile": 1.0,
        "confident": 1.5, "self-assured": 2.0, "assertive": 1.0,
        "focused": 1.0, "determined": 1.5, "steadfast": 2.0,
        "stress": 1.0, "pressure": 1.0, "high-pressure": 1.5,
        "managed crisis": 2.0, "crisis management": 2.0,
        "problem-solving": 1.5, "troubleshooting": 1.0,
        "balanced": 1.5, "consistent": 1.0, "steady": 1.5,
        "stable": 1.5, "reliable": 1.0, "dependable": 1.0,
        "grounded": 1.5, "mature": 1.0, "rational": 1.5,
        "logical": 1.0, "objective": 1.0, "pragmatic": 1.0,
        "patience": 1.5, "persistence": 1.5, "grit": 2.0,
        "fortitude": 2.0, "courage": 1.5, "brave": 1.5
    }
}

CAREER_MAP = {
    ("Openness", "Conscientiousness"): [
        "Research Scientist", "Data Analyst", "UX Designer",
        "Product Manager", "Technical Writer"
    ],
    ("Openness", "Extraversion"): [
        "Marketing Director", "Creative Director", "Entrepreneur",
        "Brand Strategist", "Public Relations Manager"
    ],
    ("Openness", "Agreeableness"): [
        "Art Therapist", "Social Worker", "Counselor",
        "Community Developer", "Non-Profit Director"
    ],
    ("Openness", "Neuroticism (Stability)"): [
        "Innovation Consultant", "Strategic Planner", "Futurist",
        "Risk Analyst", "Scenario Planner"
    ],
    ("Conscientiousness", "Extraversion"): [
        "Project Manager", "Operations Director", "Business Analyst",
        "Management Consultant", "Team Lead"
    ],
    ("Conscientiousness", "Agreeableness"): [
        "Healthcare Administrator", "HR Manager", "Quality Assurance",
        "Compliance Officer", "Training Coordinator"
    ],
    ("Conscientiousness", "Neuroticism (Stability)"): [
        "Financial Analyst", "Accountant", "Systems Administrator",
        "Database Administrator", "Auditor"
    ],
    ("Extraversion", "Agreeableness"): [
        "Sales Manager", "Customer Success Manager", "Recruiter",
        "Event Planner", "Public Speaker"
    ],
    ("Extraversion", "Neuroticism (Stability)"): [
        "Crisis Manager", "Emergency Responder", "Military Officer",
        "Broadcast Journalist", "Trial Lawyer"
    ],
    ("Agreeableness", "Neuroticism (Stability)"): [
        "Nurse", "Therapist", "Mediator",
        "School Counselor", "Palliative Care Specialist"
    ]
}

SUMMARIES = {
    "Openness": "Creative and innovative thinker. Excels in R&D, design, strategy, and roles requiring imagination.",
    "Conscientiousness": "Highly organized and reliable. Excellent for operations, finance, project management, and detail-oriented work.",
    "Extraversion": "Strong communicator and natural leader. Well-suited for sales, HR, management, and client-facing roles.",
    "Agreeableness": "Empathetic and cooperative team player. Ideal for support, training, healthcare, and collaborative environments.",
    "Neuroticism (Stability)": "Calm under pressure and emotionally resilient. Great for high-stakes, crisis management, and fast-paced roles."
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
    """Score traits using word-boundary matching and weighted keywords."""
    scores = {}
    words = text.split()
    total_words = len(words) if words else 1

    for trait, keyword_weights in TRAIT_KEYWORDS.items():
        weighted_count = 0.0
        for keyword, weight in keyword_weights.items():
            if " " in keyword:
                count = text.count(keyword)
            else:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
            weighted_count += count * weight

        raw = (weighted_count / total_words) * 100
        scaled = 100 * (1 - math.exp(-raw * 0.8))
        scores[trait] = round(max(0, min(scaled, 100)), 2)

    return scores


def compute_confidence(text: str, scores: dict) -> dict:
    """Compute analysis confidence based on text length and trait coverage."""
    word_count = len(text.split())
    total_score = sum(scores.values())

    if word_count < 50:
        level = "Low"
        detail = "Very short text. Results may not be reliable."
    elif word_count < 150:
        level = "Moderate"
        detail = "Short text. Consider providing a more detailed CV."
    elif word_count < 300:
        level = "Good"
        detail = "Adequate text length for personality analysis."
    else:
        level = "High"
        detail = "Sufficient text for reliable personality analysis."

    if total_score < 10:
        level = "Low"
        detail += " Few personality indicators detected."

    return {
        "level": level,
        "detail": detail,
        "word_count": word_count,
        "total_indicators": round(total_score, 1)
    }


def get_career_recommendations(scores: dict, top_n: int = 5) -> list:
    """Recommend careers based on top two personality traits."""
    sorted_traits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_two = [sorted_traits[0][0], sorted_traits[1][0]]

    key = tuple(sorted(top_two))
    careers = CAREER_MAP.get(key, [])

    if not careers:
        key_rev = tuple(reversed(list(key)))
        careers = CAREER_MAP.get(key_rev, [])

    if not careers:
        fallback = {
            "Openness": ["Designer", "Researcher", "Strategist", "Innovation Lead", "Creative Director"],
            "Conscientiousness": ["Project Manager", "Analyst", "Accountant", "Quality Engineer", "Operations Manager"],
            "Extraversion": ["Sales Manager", "HR Director", "Public Speaker", "Recruiter", "Account Executive"],
            "Agreeableness": ["Counselor", "Teacher", "Social Worker", "Nurse", "HR Specialist"],
            "Neuroticism (Stability)": ["Crisis Manager", "Security Analyst", "Air Traffic Controller", "Surgeon", "Firefighter"]
        }
        careers = fallback.get(top_two[0], ["Versatile Professional"])

    return careers[:top_n]


def score_to_label(score: float) -> str:
    if score >= 60.0:
        return "High"
    elif score >= 30.0:
        return "Moderate"
    else:
        return "Low"


def extract_sections(text: str) -> dict:
    sections = {}
    patterns = {
        "Skills":     r'(skills?)(.*?)(experience|education|project|certif|$)',
        "Experience": r'(experience|work history)(.*?)(education|skills?|project|certif|$)',
        "Education":  r'(education|academic)(.*?)(skills?|experience|project|certif|$)',
        "Projects":   r'(projects?)(.*?)(skills?|experience|education|certif|$)',
    }
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = match.group(2).strip()[:300]
        else:
            sections[section] = "Not found"
    return sections


def generate_report(pdf_path: str):
    print("\n" + "=" * 60)
    print("   PERSONALITY PREDICTION SYSTEM - CV ANALYSIS")
    print("=" * 60)

    ext = os.path.splitext(pdf_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(pdf_path)
    elif ext == ".txt":
        raw_text = extract_text_from_txt(pdf_path)
    else:
        print("Unsupported file type. Please use .pdf or .txt")
        return

    clean_text = preprocess(raw_text)
    scores = score_traits(clean_text)
    confidence = compute_confidence(clean_text, scores)
    careers = get_career_recommendations(scores)

    print(f"\nFile Analyzed : {os.path.basename(pdf_path)}")
    print(f"Word Count    : {len(clean_text.split())} (after cleaning)")
    print(f"Confidence    : {confidence['level']} - {confidence['detail']}\n")

    print("-" * 60)
    print(f"{'TRAIT':<30} {'SCORE':>8}   LEVEL")
    print("-" * 60)
    for trait, score in scores.items():
        label = score_to_label(score)
        filled = int(score / 5)
        bar = "#" * filled + "." * (20 - filled)
        print(f"{trait:<30} {score:>6.2f}   {label}")
        print(f"  [{bar}]")
    print("-" * 60)

    dominant = max(scores, key=scores.get)
    print(f"\nDominant Trait : {dominant}")
    print(f"Summary        : {SUMMARIES.get(dominant, 'N/A')}")

    print("\nCareer Recommendations:")
    for i, career in enumerate(careers, 1):
        print(f"  {i}. {career}")

    print("\n" + "-" * 60)
    print("CV SECTION PREVIEW")
    print("-" * 60)
    sections = extract_sections(raw_text)
    for section, content in sections.items():
        preview = content[:120].replace("\n", " ") if content != "Not found" else "Not found"
        print(f"\n[{section}]\n  {preview}...")

    report = {
        "file": os.path.basename(pdf_path),
        "scores": scores,
        "dominant_trait": dominant,
        "summary": SUMMARIES.get(dominant, "N/A"),
        "confidence": confidence,
        "career_recommendations": careers
    }
    with open("personality_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 60)
    print("Report saved to: personality_report.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("\nUsage: python personality_predictor.py <path_to_cv.pdf>")
        print("Running in DEMO MODE with sample CV text...\n")
        sample_cv = """
        Leisha Choudhary - AI/ML and Data Science
        Summary: B.Tech student specializing in Computer Science with a strong passion for AI/ML.
        Experienced in building and deploying real-world AI applications using Python and Streamlit.
        Skills: Python, Machine Learning, NLP, Data Analysis, Streamlit, REST APIs, SQL, Communication
        Experience: AI Researcher Intern at Be Bright 1, working on Generative AI and Data Science research.
        Head of CodeGeeks Society, leading technical activities, mentoring junior members, coordinating events.
        Managed society operations with organized planning and collaborative team leadership.
        Built innovative NLP-based personality prediction system with creative design approach.
        Developed responsive frontend websites with meticulous attention to detail and quality.
        Calm and resilient under deadline pressure, adaptable to fast-paced environments.
        """
        with open("sample_cv.txt", "w") as f:
            f.write(sample_cv)
        generate_report("sample_cv.txt")
    else:
        generate_report(sys.argv[1])
