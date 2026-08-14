import streamlit as st
import os, re, json, time, math, tempfile
import nltk
import pdfminer.high_level
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.graph_objects as go

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(
    page_title="Personality Predictor | HEX Softwares",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600;700&display=swap');
:root{--bg:#0a0a0f;--bg2:#12121a;--card:rgba(255,255,255,0.03);--bdr:rgba(255,255,255,0.06);--t1:#e8e8ed;--t2:#8b8b9e;--t3:#5a5a6e;--pur:#8b5cf6;--blu:#3b82f6;--grn:#10b981;--amb:#f59e0b;--red:#ef4444;--cyn:#06b6d4;--rsm:8px;--rmd:12px;--rlg:16px}
.stApp{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--t1)}
@keyframes fadeIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.hero-section{text-align:center;padding:3rem 1rem 2rem;position:relative}
.hero-section::before{content:'';position:absolute;top:-50%;left:50%;transform:translateX(-50%);width:600px;height:600px;background:radial-gradient(circle,rgba(139,92,246,0.08)0%,transparent 70%);pointer-events:none}
.hero-badge{display:inline-flex;padding:6px 16px;border:1px solid rgba(139,92,246,0.3);border-radius:100px;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--pur);background:rgba(139,92,246,0.08);margin-bottom:1.5rem}
.hero-title{font-size:2.75rem;font-weight:800;letter-spacing:-.02em;line-height:1.15;margin-bottom:1rem;color:var(--t1);position:relative;z-index:1}
.hero-title .gtext{background:linear-gradient(135deg,var(--pur),var(--blu),var(--cyn));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:1.05rem;color:var(--t2);max-width:600px;margin:0 auto;line-height:1.6}
.glass-card{background:var(--card);border:1px solid var(--bdr);border-radius:var(--rlg);padding:2rem;backdrop-filter:blur(20px);animation:fadeIn .5s ease-out;transition:border-color .3s}
.glass-card:hover{border-color:rgba(255,255,255,0.1)}
.slbl{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--pur);margin-bottom:.5rem}
.sttl{font-size:1.5rem;font-weight:700;color:var(--t1);margin-bottom:.5rem}
.sdsc{font-size:.85rem;color:var(--t2);margin-bottom:1.5rem}
.stat-box{background:var(--card);border:1px solid var(--bdr);border-radius:var(--rmd);padding:1.25rem;text-align:center;transition:transform .3s;animation:fadeIn .5s ease-out}
.stat-box:hover{transform:translateY(-2px)}
.stat-val{font-family:'Fira Code',monospace;font-size:1.75rem;font-weight:700;color:var(--pur)}
.stat-lbl{font-size:.7rem;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-top:.25rem}
.tcard{background:var(--card);border:1px solid var(--bdr);border-radius:var(--rmd);padding:1.25rem;transition:transform .3s;animation:fadeIn .5s ease-out}
.tcard:hover{transform:translateY(-2px)}
.thdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:.75rem}
.tico{width:36px;height:36px;border-radius:var(--rsm);display:flex;align-items:center;justify-content:center}
.tnam{font-size:.8rem;font-weight:600;color:var(--t1);margin-top:.5rem}
.tscr{font-family:'Fira Code',monospace;font-size:1.5rem;font-weight:700}
.tbar-bg{width:100%;height:4px;background:rgba(255,255,255,0.06);border-radius:100px;margin-top:.75rem;overflow:hidden}
.tbar-fill{height:100%;border-radius:100px;transition:width 1s ease-out}
.tlvl{font-size:.65rem;font-weight:500;text-transform:uppercase;letter-spacing:.05em;margin-top:.5rem}
.dom-card{background:linear-gradient(135deg,rgba(139,92,246,0.1),rgba(59,130,246,0.1));border:1px solid rgba(139,92,246,0.2);border-radius:var(--rlg);padding:2rem;text-align:center;animation:fadeIn .5s ease-out}
.dom-lbl{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--pur);margin-bottom:.5rem}
.dom-trait{font-size:1.75rem;font-weight:800;background:linear-gradient(135deg,var(--pur),var(--blu));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}
.dom-desc{font-size:.85rem;color:var(--t2);max-width:500px;margin:0 auto;line-height:1.5}
.cc{background:var(--card);border:1px solid var(--bdr);border-radius:var(--rlg);padding:2rem;backdrop-filter:blur(20px);animation:fadeIn .5s ease-out}
.ci{display:flex;align-items:center;gap:12px;background:rgba(255,255,255,0.02);border:1px solid var(--bdr);border-radius:var(--rsm);padding:.85rem;transition:all .3s;margin-bottom:.5rem}
.ci:hover{border-color:rgba(139,92,246,0.3);transform:translateX(4px)}
.cnum{font-family:'Fira Code',monospace;font-size:.8rem;font-weight:600;color:var(--pur);background:rgba(139,92,246,0.1);border-radius:6px;padding:4px 10px}
.cname{font-size:.92rem;color:var(--t1);font-weight:500}
.conf-lvl{font-family:'Fira Code',monospace;font-weight:700;font-size:.85rem;display:inline-block;padding:4px 14px;border-radius:100px}
.conf-lvl.high{background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);color:var(--grn)}
.conf-lvl.good{background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);color:var(--blu)}
.conf-lvl.moderate{background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);color:var(--amb)}
.conf-lvl.low{background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);color:var(--red)}
.conf-dtl{font-size:.7rem;color:var(--t3);margin-top:.5rem}
.sp{background:var(--card);border:1px solid var(--bdr);border-radius:var(--rsm);padding:1rem;margin-bottom:.75rem}
.sp-t{font-size:.75rem;font-weight:600;color:var(--pur);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}
.sp-c{font-size:.8rem;color:var(--t2);line-height:1.5}
.footer{text-align:center;padding:3rem 1rem;color:var(--t3);font-size:.75rem}
.stButton>button{background:linear-gradient(135deg,var(--pur),var(--blu))!important;color:white!important;border:none!important;border-radius:var(--rmd)!important;padding:.6rem 2rem!important;font-weight:600!important;transition:all .3s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 25px rgba(139,92,246,0.3)!important}
.stDownloadButton>button{background:var(--card)!important;border:1px solid var(--bdr)!important;color:var(--t1)!important;border-radius:var(--rmd)!important;transition:all .3s!important}
.stDownloadButton>button:hover{border-color:rgba(139,92,246,0.3)!important}
.stTabs [data-baseweb="tab-list"]{gap:0;background:transparent}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--t2)!important;border:none!important;padding:.5rem 1rem!important}
.stTabs [aria-selected="true"]{color:var(--pur)!important;border-bottom:2px solid var(--pur)!important}
.stTextArea textarea{background:var(--bg2)!important;color:var(--t1)!important;border:1px solid var(--bdr)!important;font-family:'Fira Code',monospace!important;border-radius:var(--rmd)!important}
.stTextArea textarea:focus{border-color:var(--pur)!important;box-shadow:0 0 0 2px rgba(139,92,246,0.2)!important}
</style>
""", unsafe_allow_html=True)

TRAIT_KEYWORDS = {
    "Openness": {
        "creative": 2.0, "innovative": 2.0, "visionary": 2.0, "imaginative": 2.0,
        "curious": 1.5, "open-minded": 1.5, "artistic": 1.5, "inventive": 1.5,
        "original": 1.5, "resourceful": 1.5, "unconventional": 1.5, "exploratory": 1.5,
        "adaptable": 1.0, "versatile": 1.0, "flexible": 1.0, "experimental": 1.0,
        "aesthetic": 1.0, "philosophical": 1.0, "insightful": 1.0, "perceptive": 1.0,
        "brainstorm": 1.0, "ideation": 1.5, "design": 1.0, "prototype": 1.5,
        "research": 1.0, "exploration": 1.0, "novel": 1.5, "pioneering": 2.0,
        "transform": 1.0, "disruptive": 1.5, "conceptual": 1.0,
        "abstract": 1.0, "theoretical": 1.0, "intellectual": 1.0, "cultured": 1.0,
        "sophisticated": 1.0, "eclectic": 1.5, "avant-garde": 2.0, "progressive": 1.0,
        "reform": 1.0, "modernize": 1.0, "reimagine": 1.5, "envision": 1.5,
        "inspire": 1.0, "intuitive": 1.0, "divergent": 1.5, "synthesis": 1.0,
        "holistic": 1.0, "interdisciplinary": 1.5
    },
    "Conscientiousness": {
        "organized": 2.0, "meticulous": 2.0, "disciplined": 2.0, "systematic": 2.0,
        "efficient": 1.5, "thorough": 1.5, "reliable": 1.5, "punctual": 1.5,
        "methodical": 1.5, "structured": 1.5, "detail-oriented": 2.0, "precise": 1.5,
        "diligent": 1.5, "responsible": 1.0, "accountable": 1.0, "consistent": 1.0,
        "planning": 1.0, "scheduling": 1.0, "deadline": 1.0, "quality": 1.0,
        "accuracy": 1.5, "compliance": 1.0, "audit": 1.0, "documentation": 1.0,
        "process": 1.0, "procedure": 1.0, "protocol": 1.0, "standard": 1.0,
        "optimization": 1.5, "productivity": 1.0, "milestone": 1.0, "deliverable": 1.0,
        "budget": 1.0, "timeline": 1.0, "governance": 1.5, "oversight": 1.0,
        "vigilant": 1.5, "scrupulous": 2.0, "industrious": 1.5, "tenacious": 1.5,
        "persistent": 1.0, "dedicated": 1.0, "committed": 1.0, "focused": 1.0,
        "analytical": 1.0, "logical": 1.0, "rigorous": 1.5, "pragmatic": 1.0,
        "strategic": 1.0, "goal-oriented": 1.5
    },
    "Extraversion": {
        "leadership": 2.0, "charismatic": 2.0, "outgoing": 2.0, "sociable": 2.0,
        "collaborative": 1.5, "persuasive": 1.5, "influential": 1.5, "dynamic": 1.5,
        "energetic": 1.5, "enthusiastic": 1.5, "assertive": 1.5, "confident": 1.5,
        "networking": 1.5, "public-speaking": 2.0, "presentation": 1.0, "communication": 1.0,
        "teamwork": 1.0, "mentoring": 1.0, "coaching": 1.0, "facilitation": 1.0,
        "negotiation": 1.5, "stakeholder": 1.0, "client-facing": 1.5, "engagement": 1.0,
        "advocacy": 1.0, "ambassador": 1.5, "spokesperson": 2.0, "moderator": 1.0,
        "host": 1.0, "emcee": 1.5, "rallying": 1.5, "motivating": 1.5,
        "inspiring": 1.0, "galvanizing": 2.0, "mobilizing": 1.5, "activating": 1.0,
        "connecting": 1.0, "relationship": 1.0, "partnership": 1.0, "alliance": 1.0,
        "outreach": 1.0, "community": 1.0, "evangelist": 1.5, "champion": 1.5,
        "proactive": 1.0, "initiative": 1.0, "driven": 1.0, "ambitious": 1.0,
        "competitive": 1.0, "bold": 1.5
    },
    "Agreeableness": {
        "empathetic": 2.0, "compassionate": 2.0, "supportive": 2.0, "nurturing": 2.0,
        "cooperative": 1.5, "diplomatic": 1.5, "considerate": 1.5, "patient": 1.5,
        "understanding": 1.5, "tolerant": 1.5, "harmonious": 1.5, "altruistic": 2.0,
        "helpful": 1.0, "kind": 1.0, "generous": 1.0, "caring": 1.0,
        "respectful": 1.0, "courteous": 1.0, "tactful": 1.0, "gracious": 1.0,
        "mediation": 1.5, "conflict-resolution": 2.0, "consensus": 1.5, "inclusion": 1.0,
        "diversity": 1.0, "equity": 1.0, "welfare": 1.0, "well-being": 1.0,
        "volunteer": 1.5, "charity": 1.5, "humanitarian": 2.0, "philanthropy": 2.0,
        "mentor": 1.0, "counsel": 1.0, "guide": 1.0, "assist": 1.0,
        "accommodate": 1.0, "collaborate": 1.0, "trustworthy": 1.5, "loyal": 1.0,
        "sincere": 1.0, "genuine": 1.0, "warm": 1.0, "friendly": 1.0,
        "approachable": 1.0, "amiable": 1.5, "benevolent": 2.0, "selfless": 2.0,
        "forgiving": 1.5, "gentle": 1.0
    },
    "Neuroticism (Stability)": {
        "resilient": 2.0, "composed": 2.0, "steady": 2.0, "calm": 2.0,
        "stress-management": 2.0, "pressure": 1.5, "crisis": 1.5, "emergency": 1.5,
        "high-stakes": 2.0, "critical": 1.0, "urgent": 1.0, "intense": 1.0,
        "demanding": 1.0, "challenging": 1.0, "adversity": 1.5, "obstacle": 1.0,
        "risk": 1.0, "uncertainty": 1.0, "volatile": 1.5, "turbulent": 1.5,
        "unpredictable": 1.0, "fast-paced": 1.0, "high-pressure": 2.0, "deadline-driven": 1.5,
        "multitasking": 1.0, "prioritization": 1.0, "triage": 1.5, "rapid-response": 2.0,
        "contingency": 1.5, "backup": 1.0, "recovery": 1.0, "mitigation": 1.5,
        "stability": 1.0, "endurance": 1.5, "stamina": 1.5, "fortitude": 2.0,
        "perseverance": 1.5, "grit": 2.0, "determination": 1.0, "resolve": 1.5,
        "unflappable": 2.0, "imperturbable": 2.0, "level-headed": 2.0, "even-keeled": 2.0,
        "grounded": 1.5, "centered": 1.0, "balanced": 1.0, "poised": 1.5,
        "measured": 1.0, "controlled": 1.0
    }
}

CAREER_MAP = {
    "Openness": ["UX/UI Designer", "Creative Director", "Research Scientist", "Innovation Consultant",
                 "Product Designer", "Data Scientist", "Architect", "Content Strategist",
                 "R&D Engineer", "Entrepreneur"],
    "Conscientiousness": ["Project Manager", "Financial Analyst", "Quality Assurance Engineer",
                          "Operations Manager", "Compliance Officer", "Data Analyst",
                          "Accountant", "Supply Chain Manager", "Systems Administrator", "Auditor"],
    "Extraversion": ["Sales Manager", "Marketing Director", "Public Relations Specialist",
                     "Business Development Manager", "Team Lead", "Event Coordinator",
                     "Recruiter", "Customer Success Manager", "Brand Ambassador", "Trainer"],
    "Agreeableness": ["Human Resources Manager", "Social Worker", "Counselor",
                      "Nonprofit Director", "Teacher", "Mediator", "Community Manager",
                      "Patient Advocate", "Therapist", "Nurse"],
    "Neuroticism (Stability)": ["Crisis Manager", "Emergency Responder", "Air Traffic Controller",
                                 "Surgeon", "Military Officer", "Cybersecurity Analyst",
                                 "Risk Manager", "Pilot", "Paramedic", "Detective"]
}

TRAIT_COLORS = {
    "Openness": "#8b5cf6",
    "Conscientiousness": "#3b82f6",
    "Extraversion": "#10b981",
    "Agreeableness": "#f59e0b",
    "Neuroticism (Stability)": "#06b6d4"
}

TRAIT_ICONS = {
    "Openness": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "Conscientiousness": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 12l2 2 4-4"/></svg>',
    "Extraversion": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "Agreeableness": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>',
    "Neuroticism (Stability)": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
}

SUMMARIES = {
    "Openness": "Highly creative and open to new experiences. Thrives in environments encouraging innovation and unconventional thinking.",
    "Conscientiousness": "Exceptionally organized and detail-oriented. Excels in structured environments with clear goals and quality standards.",
    "Extraversion": "Natural leader and communicator. Energized by social interaction, team collaboration, and public-facing roles.",
    "Agreeableness": "Empathetic and cooperative team player. Values harmony, trust, and supportive relationships in the workplace.",
    "Neuroticism (Stability)": "Emotionally stable and resilient under pressure. Maintains composure in high-stress situations and adapts well to change."
}


def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        text = pdfminer.high_level.extract_text(tmp_path)
    finally:
        os.unlink(tmp_path)
    return text


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="replace")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s\-]", " ", text)
    tokens = word_tokenize(text)
    stop = set(stopwords.words("english"))
    return [t for t in tokens if t not in stop and len(t) > 2]


def score_traits(text):
    tokens = preprocess(text)
    raw = {}
    for trait, kw in TRAIT_KEYWORDS.items():
        s = 0
        for word, weight in kw.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            s += len(re.findall(pattern, text.lower())) * weight
        raw[trait] = s
    mx = max(raw.values()) if max(raw.values()) > 0 else 1
    scores = {}
    for t, r in raw.items():
        n = r / mx
        sig = 100 / (1 + math.exp(-5 * (n - 0.5)))
        scores[t] = round(max(5, min(95, sig)), 1)
    return scores


def compute_confidence(scores, word_count):
    spread = max(scores.values()) - min(scores.values())
    if word_count >= 200 and spread >= 20:
        return ("High", "Strong signal from a detailed CV")
    elif word_count >= 100 and spread >= 10:
        return ("Good", "Adequate data for reliable prediction")
    elif word_count >= 50:
        return ("Moderate", "Limited text; results are approximate")
    else:
        return ("Low", "Very short text; consider adding more detail")


def get_career_recommendations(scores, top_n=5):
    dominant = max(scores, key=scores.get)
    return CAREER_MAP.get(dominant, [])[:top_n]


def score_to_label(score):
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Moderate"
    else:
        return "Low"


def extract_sections(text):
    patterns = {
        "Summary": r"(?i)\b(summary|profile|objective|about)\b",
        "Experience": r"(?i)\b(experience|employment|work\s*history)\b",
        "Education": r"(?i)\b(education|academic|qualification|degree)\b",
        "Skills": r"(?i)\b(skills|competencies|expertise|proficiencies)\b",
        "Certifications": r"(?i)\b(certifications?|licenses?)\b",
        "Projects": r"(?i)\b(projects?|portfolio)\b",
        "Awards": r"(?i)\b(awards?|honors?|achievements?)\b",
    }
    lines = text.split("\n")
    sections = {}
    cur = None
    buf = []
    for line in lines:
        matched = False
        for name, pat in patterns.items():
            if re.search(pat, line) and len(line.strip()) < 80:
                if cur and buf:
                    sections[cur] = "\n".join(buf).strip()
                cur = name
                buf = []
                matched = True
                break
        if not matched and cur:
            buf.append(line)
    if cur and buf:
        sections[cur] = "\n".join(buf).strip()
    return sections


def radar_chart(scores):
    traits = list(scores.keys())
    vals = list(scores.values())
    traits_c = traits + [traits[0]]
    vals_c = vals + [vals[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals_c, theta=traits_c, fill='toself',
        fillcolor='rgba(139,92,246,0.15)',
        line=dict(color='#8b5cf6', width=2),
        marker=dict(size=6, color='#8b5cf6')
    ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100],
                tickfont=dict(size=10, color='#5a5a6e', family='Fira Code'),
                gridcolor='rgba(255,255,255,0.06)'),
            angularaxis=dict(
                tickfont=dict(size=11, color='#8b8b9e', family='Inter'),
                gridcolor='rgba(255,255,255,0.06)')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, showlegend=False, margin=dict(l=60, r=60, t=40, b=40))
    return fig


def bar_chart(scores):
    traits = list(scores.keys())
    vals = list(scores.values())
    colors = [TRAIT_COLORS[t] for t in traits]
    fig = go.Figure(data=go.Bar(
        y=traits, x=vals, orientation='h',
        marker=dict(color=colors),
        text=[f"{v}%" for v in vals], textposition='outside',
        textfont=dict(family='Fira Code', size=12, color='#e8e8ed')
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        xaxis=dict(range=[0,110], showgrid=True,
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(color='#5a5a6e', family='Fira Code'), zeroline=False),
        yaxis=dict(tickfont=dict(color='#8b8b9e', family='Inter', size=12),
            autorange='reversed'),
        margin=dict(l=150, r=60, t=20, b=20), showlegend=False)
    return fig


# ── Hero ──
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">AI-Powered Analysis</div>
    <div class="hero-title">
        <span class="gtext">Personality Predictor</span><br>from Your CV
    </div>
    <div class="hero-sub">
        Upload your CV or paste text to receive a Big Five personality analysis
        with career recommendations, powered by NLP.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ──
st.markdown("""
<div class="glass-card" style="margin-bottom:2rem">
    <div class="slbl">Input</div>
    <div class="sttl">Provide Your CV</div>
    <div class="sdsc">Upload a PDF/TXT file or paste your CV text directly below.</div>
</div>
""", unsafe_allow_html=True)

DEMO_CV = """SUMMARY
Creative and innovative software engineer with a visionary approach. Curious and open-minded professional who thrives in collaborative, fast-paced environments. Known for being organized, meticulous, and detail-oriented with a disciplined work ethic.

EXPERIENCE

Senior Software Engineer | TechCorp Innovation Labs | 2020 - Present
- Led a dynamic team of 12 engineers, demonstrating strong leadership and persuasive communication
- Pioneered an innovative machine learning pipeline, showcasing creative research capabilities
- Managed project timelines with thorough planning, structured documentation, and reliable delivery
- Mentored junior developers with empathetic, supportive, and patient guidance
- Maintained composure and resilience during high-pressure product launches and crisis situations
- Drove stakeholder engagement through enthusiastic presentations and confident public-speaking

Software Developer | DataDriven Solutions | 2017 - 2020
- Designed and prototyped novel data visualization tools with an artistic, intuitive interface
- Implemented systematic quality assurance processes with rigorous compliance standards
- Built cooperative partnerships across departments through diplomatic, considerate collaboration
- Handled demanding deadlines with steady prioritization and calm multitasking
- Championed community outreach programs, volunteering as a compassionate technology mentor

EDUCATION
Master of Science in Computer Science | Stanford University | 2017
Bachelor of Science in Software Engineering | MIT | 2015

SKILLS
Python, JavaScript, Machine Learning, Data Analysis, Strategic Planning, Agile Methodology,
Public Speaking, Team Leadership, Conflict Resolution, Risk Management, Brainstorming,
Prototyping, Documentation, Quality Assurance, Stakeholder Engagement

CERTIFICATIONS
- AWS Certified Solutions Architect
- Certified ScrumMaster (CSM)
- Google Professional Data Engineer

AWARDS
- Innovation Champion Award, TechCorp (2022)
- Humanitarian Technology Grant Recipient (2021)
- Outstanding Volunteer Service Award (2019)
"""

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    tab_upload, tab_paste = st.tabs(["Upload CV", "Paste Text"])
    with tab_upload:
        uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"], label_visibility="collapsed")
        use_demo = st.checkbox("Use demo CV for testing")
    with tab_paste:
        pasted_text = st.text_area("Paste your CV text here", height=250,
                                   placeholder="Paste your CV or resume text here...",
                                   label_visibility="collapsed")
    analyze_btn = st.button("Analyze Personality", use_container_width=True)

# ── Analysis ──
if analyze_btn:
    raw_text = None
    if use_demo:
        raw_text = DEMO_CV
    elif uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = extract_text_from_txt(uploaded_file)
    elif pasted_text and pasted_text.strip():
        raw_text = pasted_text.strip()

    if not raw_text or len(raw_text.strip()) < 20:
        st.error("Please provide at least 20 characters of text.")
    else:
        with st.status("Analyzing your CV...", expanded=True) as status:
            st.write("Extracting text content...")
            time.sleep(0.5)
            st.write("Preprocessing and tokenizing...")
            time.sleep(0.5)
            scores = score_traits(raw_text)
            st.write("Scoring Big Five personality traits...")
            time.sleep(0.5)
            word_count = len(raw_text.split())
            dominant = max(scores, key=scores.get)
            top_score = scores[dominant]
            confidence = compute_confidence(scores, word_count)
            careers = get_career_recommendations(scores)
            sections = extract_sections(raw_text)
            st.write("Generating recommendations...")
            time.sleep(0.3)
            status.update(label="Analysis complete!", state="complete", expanded=False)

        st.markdown("---")

        st.markdown('<div style="text-align:center"><div class="slbl">Overview</div><div class="sttl">Analysis Results</div></div>', unsafe_allow_html=True)
        st.write("")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{word_count}</div><div class="stat-lbl">Words Analyzed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-box"><div class="stat-val">5</div><div class="stat-lbl">Traits Scored</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{top_score}%</div><div class="stat-lbl">Top Score</div></div>', unsafe_allow_html=True)
        with c4:
            conf_cls = confidence[0].lower()
            st.markdown(f'<div class="stat-box"><div class="conf-lvl {conf_cls}">{confidence[0]}</div><div class="conf-dtl">{confidence[1]}</div></div>', unsafe_allow_html=True)

        st.write("")

        st.markdown('<div style="text-align:center"><div class="slbl">Visualization</div><div class="sttl">Trait Profile</div></div>', unsafe_allow_html=True)
        st.write("")

        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(scores), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(bar_chart(scores), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")

        st.markdown('<div style="text-align:center"><div class="slbl">Breakdown</div><div class="sttl">Individual Traits</div></div>', unsafe_allow_html=True)
        st.write("")

        tcols = st.columns(5)
        for i, (trait, score) in enumerate(scores.items()):
            color = TRAIT_COLORS[trait]
            icon = TRAIT_ICONS[trait]
            level = score_to_label(score)
            with tcols[i]:
                html = f'<div class="tcard"><div class="thdr"><div class="tico" style="background:{color}15">{icon}</div><div class="tscr" style="color:{color}">{score}%</div></div><div class="tnam">{trait}</div><div class="tbar-bg"><div class="tbar-fill" style="width:{score}%;background:{color}"></div></div><div class="tlvl" style="color:{color}">{level}</div></div>'
                st.markdown(html, unsafe_allow_html=True)

        st.write("")

        dom_html = f'<div class="dom-card"><div class="dom-lbl">Dominant Trait</div><div class="dom-trait">{dominant}</div><div class="dom-desc">{SUMMARIES[dominant]}</div></div>'
        st.markdown(dom_html, unsafe_allow_html=True)

        st.write("")

        st.markdown('<div style="text-align:center"><div class="slbl">Recommendations</div><div class="sttl">Career Paths</div></div>', unsafe_allow_html=True)
        st.write("")

        career_html = '<div class="cc">'
        for idx, career in enumerate(careers, 1):
            career_html += f'<div class="ci"><div class="cnum">{idx:02d}</div><div class="cname">{career}</div></div>'
        career_html += '</div>'
        st.markdown(career_html, unsafe_allow_html=True)

        st.write("")

        if sections:
            st.markdown('<div style="text-align:center"><div class="slbl">CV Structure</div><div class="sttl">Detected Sections</div></div>', unsafe_allow_html=True)
            st.write("")
            for sn, sc in sections.items():
                prev = sc[:300] + ("..." if len(sc) > 300 else "")
                st.markdown(f'<div class="sp"><div class="sp-t">{sn}</div><div class="sp-c">{prev}</div></div>', unsafe_allow_html=True)

        st.write("")

        report = {
            "scores": scores,
            "dominant_trait": dominant,
            "summary": SUMMARIES[dominant],
            "confidence": {"level": confidence[0], "detail": confidence[1]},
            "career_recommendations": careers,
            "sections_detected": list(sections.keys()) if sections else [],
            "word_count": word_count
        }
        st.download_button(
            label="Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="personality_report.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown('<div class="footer">Built with &hearts; by HEX Softwares &middot; Powered by NLP & Big Five Model</div>', unsafe_allow_html=True)
