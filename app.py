
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


st.set_page_config(
    page_title="Personality Predictor | HEX Softwares",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

/* ── Tokens ── */
:root {
  --bg-base:       #0D0D0D;
  --bg-surface:    #161B27;
  --bg-card:       rgba(22, 27, 39, 0.75);
  --bg-glass:      rgba(255, 255, 255, 0.06);
  --border-subtle: rgba(255, 255, 255, 0.10);
  --border-card:   rgba(255, 255, 255, 0.14);
  --text-primary:  #F1F5F9;
  --text-secondary:#94A3B8;
  --text-muted:    #64748B;
  --accent-blue:   #3B82F6;
  --accent-cyan:   #06B6D4;
  --accent-purple: #A78BFA;
  --accent-pink:   #EC4899;
  --accent-green:  #10B981;
  --accent-amber:  #F59E0B;
  --accent-orange: #F97316;
  --radius-card:   16px;
  --radius-btn:    10px;
  --shadow-card:   0 4px 24px rgba(0,0,0,0.4);
  --blur-glass:    blur(18px);
  --transition:    all 0.2s ease;
}

/* ── Base ── */
* { box-sizing: border-box; }
html, body, [data-testid="stApp"] { background: var(--bg-base) !important; color: var(--text-primary); font-family: 'Fira Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 3px; }

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, #0D0D0D 0%, #111827 50%, #0F172A 100%);
  border: 1px solid var(--border-subtle);
  border-radius: 20px;
  padding: 3rem 2.5rem;
  text-align: center;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(59,130,246,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: 'Fira Code', monospace;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--accent-blue);
  margin-bottom: 0.75rem;
}
.hero-title {
  font-size: 2.6rem;
  font-weight: 700;
  line-height: 1.2;
  background: linear-gradient(90deg, #60A5FA, #A78BFA, #EC4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.75rem;
}
.hero-sub { color: var(--text-secondary); font-size: 1rem; line-height: 1.6; max-width: 520px; margin: 0 auto; }
.hero-badge {
  display: inline-block;
  font-family: 'Fira Code', monospace;
  font-size: 0.72rem;
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  color: var(--accent-cyan);
  border-radius: 20px;
  padding: 4px 14px;
  margin-top: 1.25rem;
}

/* ── Glass Card ── */
.glass-card {
  background: var(--bg-card);
  backdrop-filter: var(--blur-glass);
  -webkit-backdrop-filter: var(--blur-glass);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-card);
  padding: 1.5rem;
  box-shadow: var(--shadow-card);
  transition: var(--transition);
}
.glass-card:hover { border-color: rgba(255,255,255,0.22); }

/* ── Stat Box ── */
.stat-box {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 1.25rem;
  text-align: center;
  backdrop-filter: var(--blur-glass);
  transition: var(--transition);
}
.stat-box:hover { border-color: var(--accent-blue); }
.stat-num {
  font-family: 'Fira Code', monospace;
  font-size: 1.75rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.2;
}
.stat-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

/* ── Trait Card ── */
.trait-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-card);
  padding: 1.2rem 1.4rem;
  margin-bottom: 0.85rem;
  backdrop-filter: var(--blur-glass);
  transition: border-color 0.2s ease;
}
.trait-card:hover { border-color: rgba(255,255,255,0.25); }
.trait-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.trait-name {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 8px;
}
.trait-label {
  font-family: 'Fira Code', monospace;
  font-size: 0.78rem;
  font-weight: 500;
  padding: 3px 10px;
  border-radius: 20px;
  border: 1px solid;
}
.trait-label.High   { color: #10B981; border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.1); }
.trait-label.Moderate { color: #F59E0B; border-color: rgba(245,158,11,0.4); background: rgba(245,158,11,0.1); }
.trait-label.Low    { color: #EF4444; border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.1); }

/* ── Dominant Card ── */
.dominant-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(59,130,246,0.08));
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 18px;
  padding: 2rem;
  text-align: center;
  margin: 1rem 0;
  position: relative;
  overflow: hidden;
}
.dominant-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(59,130,246,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.dominant-eyebrow {
  font-family: 'Fira Code', monospace;
  font-size: 0.7rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}
.dominant-trait {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}
.dominant-summary { font-size: 0.9rem; color: var(--text-secondary); max-width: 420px; margin: 0 auto; line-height: 1.6; }

/* ── Section Card ── */
.section-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-left: 3px solid var(--accent-blue);
  border-radius: 0 var(--radius-card) var(--radius-card) 0;
  padding: 1rem 1.2rem;
  margin-bottom: 0.75rem;
  backdrop-filter: var(--blur-glass);
}
.section-title-text {
  font-family: 'Fira Code', monospace;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--accent-blue);
  margin-bottom: 6px;
}
.section-content { font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6; }

/* ── Buttons ── */
.stButton > button, [data-testid="stButton"] > button {
  background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-btn) !important;
  font-family: 'Fira Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 0.6rem 1.5rem !important;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 16px rgba(59,130,246,0.35);
}
.stButton > button:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(59,130,246,0.45);
}
.stDownloadButton > button, [data-testid="stDownloadButton"] > button {
  background: var(--bg-glass) !important;
  color: var(--accent-blue) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: var(--radius-btn) !important;
  font-family: 'Fira Sans', sans-serif !important;
  font-weight: 500 !important;
  cursor: pointer;
  transition: var(--transition);
}
.stDownloadButton > button:hover {
  border-color: var(--accent-blue) !important;
  background: rgba(59,130,246,0.1) !important;
}

/* ── Divider ── */
[data-testid="stDivider"] { border-color: var(--border-subtle) !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg-card);
  border: 1px dashed var(--border-card) !important;
  border-radius: var(--radius-card) !important;
  padding: 1.5rem;
  transition: var(--transition);
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-blue) !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color: var(--text-secondary); font-size: 0.9rem; cursor: pointer; }
.stCheckbox { display: flex; align-items: center; gap: 8px; }

/* ── Chart Containers ── */
.chart-container {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-card);
  padding: 1.25rem;
  backdrop-filter: var(--blur-glass);
}
.chart-title {
  font-family: 'Fira Code', monospace;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-muted);
  margin-bottom: 1rem;
}

/* ── Animations ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.hero, .glass-card, .stat-box, .trait-card, .dominant-card, .section-card {
  animation: fadeInUp 0.4s ease forwards;
}

/* ── Reduced Motion ── */
@media (prefers-reduced-motion: reduce) {
  .hero, .glass-card, .stat-box, .trait-card, .dominant-card, .section-card { animation: none; }
  * { transition: none !important; }
}
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
    "Openness":               "#A78BFA",
    "Conscientiousness":      "#3B82F6",
    "Extraversion":           "#EC4899",
    "Agreeableness":          "#10B981",
    "Neuroticism (Stability)":"#F97316"
}

TRAIT_ICONS = {
    "Openness":               """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/><path d="M16.5 7.5a9 9 0 0 1 3 3"/></svg>""",
    "Conscientiousness":      """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>""",
    "Extraversion":           """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>""",
    "Agreeableness":          """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>""",
    "Neuroticism (Stability)":"""<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>"""
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
    if score >= 5.0:   return ("High", "High")
    elif score >= 2.0: return ("Moderate", "Moderate")
    else:              return ("Low", "Low")

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
    colors  = [TRAIT_COLORS[t] for t in traits] + [TRAIT_COLORS[traits[0]]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels,
        fill='toself',
        fillcolor='rgba(59,130,246,0.12)',
        line=dict(color='#3B82F6', width=2),
        marker=dict(size=6, color='#3B82F6')
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(13,13,13,0)',
            radialaxis=dict(
                visible=True, range=[0,100],
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='#64748B', size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='#94A3B8', size=11)
            )
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
    colors = [TRAIT_COLORS[t] for t in scores.keys()]

    fig = go.Figure(go.Bar(
        x=traits, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"<b>{v:.1f}</b>" for v in values],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94A3B8', family='Fira Sans'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(color='#94A3B8', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            range=[0,115],
            tickfont=dict(color='#64748B', size=10)
        ),
        margin=dict(t=30,b=10,l=10,r=10),
        height=280
    )
    return fig

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">HEX Softwares — AI Resume Analysis</div>
    <div class="hero-title">Personality Predictor</div>
    <div class="hero-sub">
        Upload a CV and discover personality traits using the Big Five (OCEAN) model.
        Powered by NLP keyword analysis.
    </div>
    <div class="hero-badge">Big Five Model · OCEAN Framework</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
col_upload, col_demo = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload CV (PDF or TXT)", type=["pdf","txt"])
with col_demo:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.checkbox("Use Demo CV")

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("Analyze Personality", use_container_width=True, type="primary")

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
        with st.container():
            st.info("Running with demo CV sample")

    elif uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            raw_text = extract_text_from_pdf(tmp_path) if uploaded_file.name.endswith(".pdf") else extract_text_from_txt(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        st.warning("Please upload a CV or enable Demo Mode.")
        st.stop()

    with st.status("Analyzing CV...", expanded=True) as status:
        status.write("Reading document...")
        time.sleep(0.5)
        status.write("Preprocessing text...")
        clean_text = preprocess(raw_text)
        time.sleep(0.5)
        status.write("Scoring personality traits...")
        scores = score_traits(clean_text)
        time.sleep(0.5)
        status.write("Building report...")
        dominant = max(scores, key=scores.get)
        sections = extract_sections(raw_text)
        time.sleep(0.3)
        status.update(label="Analysis complete!", state="complete", expanded=False)

    st.divider()

    # ── Stats Row ──
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class="stat-box"><div class="stat-num">{len(clean_text.split())}</div><div class="stat-label">Words Analyzed</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="stat-box"><div class="stat-num">5</div><div class="stat-label">Traits Scored</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="stat-box"><div class="stat-num">{max(scores.values()):.0f}</div><div class="stat-label">Top Score</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("""<div class="chart-container"><div class="chart-title">Radar Chart</div>""", unsafe_allow_html=True)
        st.plotly_chart(radar_chart(scores), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with ch2:
        st.markdown("""<div class="chart-container"><div class="chart-title">Trait Scores</div>""", unsafe_allow_html=True)
        st.plotly_chart(bar_chart(scores), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Trait Cards ──
    st.markdown("#### Trait Breakdown")
    for trait, score in scores.items():
        color = TRAIT_COLORS[trait]
        icon_svg = TRAIT_ICONS[trait]
        level, css_cls = score_to_label(score)
        filled = int(score / 5)
        empty  = 20 - filled
        bar_html = f"<span style='color:{color}'>{'█' * filled}</span><span style='color:#1E293B'>{'░' * empty}</span>"

        st.markdown(f"""
        <div class="trait-card">
            <div class="trait-header">
                <div class="trait-name" style="color:{color}">
                    {icon_svg}
                    {trait}
                </div>
                <span class="trait-label {css_cls}">{level} &nbsp;{score:.1f}</span>
            </div>
            <div style="font-family:'Fira Code',monospace;font-size:0.8rem;letter-spacing:1px">[{bar_html}]</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Dominant Trait ──
    dom_color = TRAIT_COLORS[dominant]
    dom_icon  = TRAIT_ICONS[dominant]
    st.markdown(f"""
    <div class="dominant-card">
        <div class="dominant-eyebrow">Dominant Personality Trait</div>
        <div class="dominant-trait" style="color:{dom_color}">{dom_icon}&nbsp; {dominant}</div>
        <div class="dominant-summary">{SUMMARIES[dominant]}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── CV Sections ──
    st.markdown("#### CV Section Preview")
    for section, content in sections.items():
        preview = content[:200].replace('\n', ' ') if content != "Not found" else "Not found"
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title-text">{section}</div>
            <div class="section-content">{preview}...</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Download ──
    report = {"scores": scores, "dominant_trait": dominant, "summary": SUMMARIES[dominant]}
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(report, indent=4),
        file_name="personality_report.json",
        mime="application/json",
        use_container_width=True
    )
