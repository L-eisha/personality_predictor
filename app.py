

import streamlit as st
import os, re, json, time, tempfile
import nltk
import pdfminer.high_level
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.graph_objects as go

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Personality Predictor | HEX Softwares",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0e1117; color: #e0e0e0; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #2a3550;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4facfe, #a78bfa, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-sub { color: #8892a4; font-size: 1rem; }
.hero-badge {
    display: inline-block;
    background: #1e2d4a;
    color: #4facfe;
    border: 1px solid #2a4a7f;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 1rem;
}
.stat-box {
    background: #161b27;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #2a3550;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #4facfe; }
.stat-label { font-size: 0.78rem; color: #8892a4; margin-top: 2px; }

.trait-card {
    background: #161b27;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    border: 1px solid #2a3550;
}
.trait-name { font-size: 0.95rem; font-weight: 600; color: #e0e0e0; margin-bottom: 6px; }
.trait-label { font-size: 0.8rem; font-weight: 500; float: right; }

.dominant-card {
    background: linear-gradient(135deg, #1a1f2e, #0f3460);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    border: 1px solid #2a4a7f;
    margin: 1rem 0;
}
.dominant-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; color: #8892a4; margin-bottom: 0.5rem; }
.dominant-trait { font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; }
.dominant-summary { font-size: 0.9rem; color: #8892a4; max-width: 400px; margin: 0 auto; }

.section-card {
    background: #161b27;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    border-left: 3px solid #4facfe;
}
.section-title-text { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #4facfe; margin-bottom: 4px; }
.section-content { font-size: 0.85rem; color: #8892a4; line-height: 1.6; }

.stButton > button {
    background: linear-gradient(90deg, #4facfe, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stDownloadButton > button {
    background: #161b27 !important;
    color: #4facfe !important;
    border: 1px solid #2a4a7f !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}
hr { border-color: #2a3550 !important; }
[data-testid="stFileUploader"] { background: #161b27; border-radius: 12px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
TRAIT_KEYWORDS = {
    "Openness": [
        "creative","innovative","imaginative","curious","artistic",
        "inventive","visionary","design","research","explore",
        "novel","experiment","idea","strategy","brainstorm",
        "diverse","flexible","open-minded","concept","prototype"
    ],
    "Conscientiousness": [
        "organized","detail-oriented","responsible","reliable",
        "punctual","planned","systematic","thorough","disciplined",
        "goal-oriented","efficient","deadline","accurate","structured",
        "managed","documented","tracked","scheduled","completed","delivered"
    ],
    "Extraversion": [
        "leadership","team","communication","presentation","public speaking",
        "networking","collaborated","mentored","coordinated","negotiated",
        "motivated","enthusiastic","outgoing","sociable","client",
        "stakeholder","event","drove","facilitated","led"
    ],
    "Agreeableness": [
        "supportive","empathy","cooperative","helpful","patient",
        "kind","collaborative","volunteer","community","resolved",
        "assisted","trained","guided","feedback","inclusive",
        "listened","understood","caring","trustworthy","mediated"
    ],
    "Neuroticism (Stability)": [
        "calm","composed","resilient","adaptable","stable",
        "confident","focused","stress","pressure","challenge",
        "overcame","handled","managed crisis","problem-solving",
        "recovered","balanced","consistent","steady","reliable","persistence"
    ]
}

TRAIT_COLORS = {
    "Openness":               "#a78bfa",
    "Conscientiousness":      "#4facfe",
    "Extraversion":           "#f093fb",
    "Agreeableness":          "#43e97b",
    "Neuroticism (Stability)":"#fa8231"
}

TRAIT_EMOJI = {
    "Openness":               "💡",
    "Conscientiousness":      "📋",
    "Extraversion":           "🎤",
    "Agreeableness":          "🤝",
    "Neuroticism (Stability)":"🧘"
}

SUMMARIES = {
    "Openness":               "Creative and innovative. Great for R&D, design, or strategy roles.",
    "Conscientiousness":      "Highly organized and reliable. Excellent for operations, finance, or project management.",
    "Extraversion":           "Strong communicator and team leader. Well-suited for sales, HR, or client-facing roles.",
    "Agreeableness":          "Empathetic and cooperative. Ideal for support, training, or collaborative environments.",
    "Neuroticism (Stability)":"Calm under pressure and resilient. Good fit for high-stakes or fast-paced roles."
}

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def extract_text_from_pdf(path):
    return pdfminer.high_level.extract_text(path).lower()

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()

def preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z\s\-]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w not in stop_words and len(w) > 2])

def score_traits(text):
    scores = {}
    total = len(text.split()) or 1
    for trait, kws in TRAIT_KEYWORDS.items():
        count = sum(text.count(kw) for kw in kws)
        scores[trait] = min(round((count / total) * 1000, 2), 100.0)
    return scores

def score_to_label(score):
    if score >= 5.0:   return "High ✅"
    elif score >= 2.0: return "Moderate 🟡"
    else:              return "Low 🔴"

def extract_sections(text):
    sections = {}
    patterns = {
        "Skills":     r'(skills?)(.*?)(experience|education|project|certif|$)',
        "Experience": r'(experience|work history)(.*?)(education|skills?|project|certif|$)',
        "Education":  r'(education|academic)(.*?)(skills?|experience|project|certif|$)',
        "Projects":   r'(projects?)(.*?)(skills?|experience|education|certif|$)',
    }
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        sections[section] = match.group(2).strip()[:300] if match else "Not found"
    return sections

def radar_chart(scores):
    traits  = list(scores.keys())
    values  = list(scores.values()) + [list(scores.values())[0]]
    labels  = [t.split(" ")[0] for t in traits] + [list(scores.keys())[0].split(" ")[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill='toself',
        fillcolor='rgba(79,172,254,0.15)',
        line=dict(color='#4facfe', width=2),
        marker=dict(size=6, color='#4facfe')
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#161b27',
            radialaxis=dict(visible=True, range=[0,100], gridcolor='#2a3550', tickfont=dict(color='#8892a4', size=10)),
            angularaxis=dict(gridcolor='#2a3550', tickfont=dict(color='#e0e0e0', size=11))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20,b=20,l=40,r=40),
        showlegend=False, height=320
    )
    return fig

def bar_chart(scores):
    traits = [t.split(" ")[0] for t in scores.keys()]
    values = list(scores.values())
    colors = list(TRAIT_COLORS.values())
    fig = go.Figure(go.Bar(
        x=traits, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in values],
        textposition='outside',
        textfont=dict(color='#e0e0e0', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='#2a3550', tickfont=dict(size=11)),
        yaxis=dict(gridcolor='#2a3550', range=[0,115]),
        margin=dict(t=30,b=10,l=10,r=10),
        height=280
    )
    return fig

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 Personality Predictor</div>
    <div class="hero-sub">Upload a CV and predict personality using the Big Five (OCEAN) model</div>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
col_up, col_demo = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader("📂 Upload CV (PDF or TXT)", type=["pdf","txt"])
with col_demo:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.checkbox("🔁 Use Demo CV")

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔍  Analyze Personality", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if analyze:
    raw_text = ""

    if use_demo:
        raw_text = """
        john doe software engineer
        skills python machine learning data analysis communication leadership
        experience led team engineers deliver creative ai solution time
        managed project deadlines coordinated stakeholders mentored junior developers
        volunteered community coding bootcamp calm pressure detail oriented
        organized sprint planning documented technical specs thoroughly
        explored novel deep learning architectures published research findings
        resilient adaptable confident focused innovative visionary
        """.lower()
        st.info("✦ Running with demo CV sample")

    elif uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            raw_text = extract_text_from_pdf(tmp_path) if uploaded_file.name.endswith(".pdf") else extract_text_from_txt(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        st.warning("⚠️ Please upload a CV or enable Demo Mode.")
        st.stop()

    # animated loader
    with st.status("🧠 Analyzing your CV...", expanded=True) as status:
        st.write("📄 Reading document...")
        time.sleep(0.6)
        st.write("🧹 Cleaning and processing text...")
        clean_text = preprocess(raw_text)
        time.sleep(0.6)
        st.write("🔍 Scoring personality traits...")
        scores = score_traits(clean_text)
        time.sleep(0.6)
        st.write("📊 Building your report...")
        dominant = max(scores, key=scores.get)
        sections = extract_sections(raw_text)
        time.sleep(0.4)
        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    st.divider()

    # stats row
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(clean_text.split())}</div><div class="stat-label">Words Analyzed</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat-box"><div class="stat-num">5</div><div class="stat-label">Traits Scored</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{max(scores.values()):.0f}</div><div class="stat-label">Top Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("##### 📡 Radar Chart")
        st.plotly_chart(radar_chart(scores), use_container_width=True, config={"displayModeBar": False})
    with ch2:
        st.markdown("##### 📊 Bar Chart")
        st.plotly_chart(bar_chart(scores), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # trait cards
    st.markdown("##### 🎯 Trait Breakdown")
    for trait, score in scores.items():
        color = TRAIT_COLORS[trait]
        emoji = TRAIT_EMOJI[trait]
        label = score_to_label(score)
        bar_filled = int(score / 5)
        bar_empty  = 20 - bar_filled
        bar_html   = f"<span style='color:{color}'>{'█' * bar_filled}</span><span style='color:#2a3550'>{'░' * bar_empty}</span>"
        st.markdown(f"""
        <div class="trait-card">
            <div class="trait-name">{emoji} {trait}
                <span class="trait-label" style="color:{color}">{score:.1f} — {label}</span>
            </div>
            <div style="font-family:monospace;font-size:0.85rem">[{bar_html}]</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # dominant trait
    dom_color = TRAIT_COLORS[dominant]
    dom_emoji = TRAIT_EMOJI[dominant]
    st.markdown(f"""
    <div class="dominant-card">
        <div class="dominant-label">🏆 Dominant Personality Trait</div>
        <div class="dominant-trait" style="color:{dom_color}">{dom_emoji} {dominant}</div>
        <div class="dominant-summary">{SUMMARIES[dominant]}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # cv section preview
    st.markdown("##### 📂 CV Section Preview")
    for section, content in sections.items():
        preview = content[:200].replace('\n', ' ') if content != "Not found" else "Not found"
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title-text">{section}</div>
            <div class="section-content">{preview}...</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # download
    report = {"scores": scores, "dominant_trait": dominant, "summary": SUMMARIES[dominant]}
    st.download_button(
        label="⬇️  Download Full Report (JSON)",
        data=json.dumps(report, indent=4),
        file_name="personality_report.json",
        mime="application/json",
        use_container_width=True
    )
