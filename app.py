import os, datetime, html, tempfile, streamlit as st
import markdown as md_lib
from backend.ingestion import ingest_directory, DEFAULT_SAMPLE_DIR, load_file
from backend.vectorstore import VectorStoreManager
from backend.workflow import WorkflowPipeline
from backend.evaluator import SystemEvaluator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="bloom. Your academic companion", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

# ── Aggressive CSS reset + bloom. design system ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: #F5DABF !important; color: #111827 !important; }
.stApp { background: #F5DABF !important; }
.stApp > header { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }
.block-container { padding: 20px 28px 80px 28px !important; max-width: 1400px !important; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* ── FORCE ALL BASE TEXT DARK (safe, no SVG/icon bleed) ── */
[class*="stMarkdown"] p { color: #111827 !important; margin-bottom: 4px; }
[class*="stMarkdown"] li { color: #111827 !important; }
[class*="stMarkdown"] td, [class*="stMarkdown"] th { color: #111827 !important; }
[class*="stMarkdown"] h1, [class*="stMarkdown"] h2,
[class*="stMarkdown"] h3, [class*="stMarkdown"] h4 { color: #111827 !important; }
.stMarkdown, .stMarkdown > div { color: #111827 !important; }
/* NOTE: spans are intentionally excluded to avoid SVG arrow icon overlap */

/* ── SELECTBOX / DROPDOWN ── */
.stSelectbox > div > div { color: #111827 !important; background: #FFFFFF !important; }
.stSelectbox [data-baseweb="select"] > div { color: #111827 !important; background: #FFFFFF !important; }
/* Force selected value text visible */
.stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] { color: #111827 !important; }
.stSelectbox span { color: #111827 !important; }
.stSelectbox [role="combobox"] { color: #111827 !important; }
.stSelectbox [data-baseweb="select"] div[class] { color: #111827 !important; }
/* The value text container */
[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="select"] > div > div { color: #111827 !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="option"] { color: #111827 !important; background: #FFFFFF !important; }
[role="option"]:hover { background: #F9EEE9 !important; color: #6C151E !important; }
[aria-selected="true"] { background: #F9EEE9 !important; color: #6C151E !important; }

/* ── NUMBER / DATE INPUTS ── */
.stNumberInput > div > div > input { color: #111827 !important; background: #FFFFFF !important; }
.stNumberInput button { color: #374151 !important; background: #F3F4F6 !important; border: 1px solid #D1D5DB !important; }
.stNumberInput button svg { fill: #374151 !important; }
.stDateInput input { color: #111827 !important; background: #FFFFFF !important; }

/* ── CAPTION TEXT ── */
.stCaption, .stCaption p, [data-testid="stCaption"] { color: #5C3A3E !important; font-size: 0.78rem !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid #E4E8EF !important; }
[data-testid="stTabs"] [role="tab"] { color: #5C3A3E !important; font-weight: 600 !important; font-size: 0.85rem !important; padding: 8px 16px !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #6C151E !important; border-bottom: 2px solid #6C151E !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: #374151 !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader { background: #F9FAFB !important; border-radius: 8px !important; font-size: 0.85rem !important; color: #374151 !important; font-weight: 600 !important; border: 1px solid #E4E8EF !important; }
.streamlit-expanderHeader:hover { background: #F9EEE9 !important; color: #6C151E !important; }
.streamlit-expanderContent { background: #FFFFFF !important; border: 1px solid #E4E8EF !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 12px !important; color: #111827 !important; }
.streamlit-expanderContent p, .streamlit-expanderContent li { color: #111827 !important; }
/* stExpander new selectors — avoid touching SVG/arrow spans */
[data-testid="stExpander"] details { background: #FFFFFF !important; border: 1px solid #E4E8EF !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stExpander"] details > div { color: #111827 !important; }
[data-testid="stExpander"] details summary { display: flex !important; align-items: center !important; }
/* Only color the text node inside summary, not the SVG arrow */
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary > div { color: #374151 !important; font-weight: 600 !important; font-size: 0.85rem !important; }
[data-testid="stExpander"] details summary:hover p,
[data-testid="stExpander"] details summary:hover > div { color: #6C151E !important; }
/* Hide any text that leaks out of expander SVG arrows */
[data-testid="stExpander"] svg text { display: none !important; }
.streamlit-expanderHeader svg { flex-shrink: 0 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] label { color: #374151 !important; font-weight: 600 !important; font-size: 0.85rem !important; }
[data-testid="stFileUploader"] section { border: 2px dashed #D4A8AD !important; background: #FBF5F0 !important; border-radius: 10px !important; }
[data-testid="stFileUploader"] section p, [data-testid="stFileUploader"] section span { color: #5C3A3E !important; }

/* ── CODE BLOCKS ── */
.stCode code, [data-testid="stCode"] code, .stCode pre, [data-testid="stCode"] pre { color: #E2E8F0 !important; background: #1E293B !important; font-size: 0.82rem !important; }
code { background: #F1F5F9 !important; color: #6C151E !important; border-radius: 4px !important; padding: 1px 5px !important; }
pre code { background: transparent !important; color: #E2E8F0 !important; padding: 0 !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] p { color: #111827 !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #6C151E !important; }
[data-testid="stSpinner"] p { color: #374151 !important; }

/* ── NAVBAR ── */
.nav { display:flex; align-items:center; justify-content:space-between; background:#FFFFFF; border:1px solid #E4E8EF; border-radius:14px; padding:12px 22px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,.04); }
.nav-left { display:flex; align-items:center; gap:14px; }
.nav-logo-icon { width:36px; height:36px; background:#F9EEE9; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; color:#6C151E !important; font-weight:800; border:1px solid #D4A8AD; }
.nav-logo-text { font-size:1.45rem; font-weight:800; color:#111827 !important; letter-spacing:-0.5px; line-height:1; }
.nav-logo-text span { color:#6C151E !important; }
.nav-tagline { font-size:0.82rem; color:#5C3A3E !important; font-weight:500; border-left:1px solid #E4E8EF; padding-left:14px; margin-left:2px; }
.nav-right { display:flex; align-items:center; gap:14px; font-size:0.82rem; color:#5C3A3E !important; font-weight:500; }
.nav-status { display:flex; align-items:center; gap:6px; }
.nav-status-dot { width:7px; height:7px; border-radius:50%; background:#6C151E; display:inline-block; }
.nav-help { width:28px; height:28px; border-radius:50%; border:1.5px solid #D1D5DB; display:flex; align-items:center; justify-content:center; color:#5C3A3E !important; font-size:0.78rem; font-weight:700; cursor:pointer; }
.nav-avatar { width:32px; height:32px; border-radius:50%; background:#F9EEE9; border:2px solid #D4A8AD; display:flex; align-items:center; justify-content:center; color:#6C151E !important; font-size:0.82rem; font-weight:800; }

/* ── HERO ── */
.hero { margin-bottom:18px; padding:0 2px; }
.hero-eyebrow { font-size:0.68rem; font-weight:800; letter-spacing:2.5px; color:#6C151E !important; text-transform:uppercase; margin-bottom:7px; }
.hero-h1 { font-size:2.1rem; font-weight:700; color:#111827 !important; letter-spacing:-0.6px; line-height:1.2; margin-bottom:6px; }
.hero-serif { font-family:'DM Serif Display', Georgia, serif !important; font-style:italic; font-weight:400; color:#6C151E !important; }
.hero-sub { font-size:0.9rem; color:#5C3A3E !important; font-weight:400; }

/* ── CARD ── */
.card { background:#FFFFFF; border:1px solid #E4E8EF; border-radius:16px; padding:20px 22px; box-shadow:0 1px 4px rgba(0,0,0,.03); }
.card-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #F3F4F6; }
.card-header-left { display:flex; align-items:center; gap:12px; }
.card-icon { width:36px; height:36px; border-radius:9px; background:#F9EEE9; border:1px solid #D4A8AD; display:flex; align-items:center; justify-content:center; font-size:1rem; }
.card-title { font-size:0.97rem; font-weight:700; color:#111827 !important; margin-bottom:1px; }
.card-subtitle { font-size:0.78rem; color:#5C3A3E !important; font-weight:400; }

/* ── ACTION CARDS ── */
.action-card { display:flex; align-items:center; justify-content:space-between; background:#FFFFFF; border:1px solid #E4E8EF; border-radius:11px; padding:14px 16px; margin-bottom:9px; cursor:pointer; transition:all .18s; text-decoration:none; }
.action-card:hover { border-color:#6C151E; background:#FBF5F0; box-shadow:0 3px 10px rgba(108,21,30,.08); }
.action-card-left { display:flex; align-items:center; gap:13px; }
.action-card-icon { width:34px; height:34px; border-radius:8px; background:#F3F4F6; display:flex; align-items:center; justify-content:center; font-size:1rem; }
.action-card-title { font-size:0.88rem; font-weight:600; color:#111827 !important; margin-bottom:1px; }
.action-card-desc { font-size:0.75rem; color:#5C3A3E !important; font-weight:400; }
.action-card-arrow { font-size:1rem; color:#9C7A7E !important; font-weight:500; }

/* ── CHAT MESSAGES ── */
.msg-wrap-user { display:flex; justify-content:flex-end; margin-bottom:10px; }
.msg-wrap-asst { display:flex; justify-content:flex-start; margin-bottom:10px; }
.msg-user { background:#6C151E; color:#FFFFFF !important; border-radius:16px 16px 4px 16px; padding:11px 16px; font-size:0.9rem; max-width:82%; line-height:1.5; }
.msg-asst-outer { max-width:92%; }
.msg-asst-header { display:flex; align-items:center; gap:7px; margin-bottom:6px; }
.msg-asst-avatar { width:28px; height:28px; border-radius:50%; background:#F9EEE9; border:1px solid #D4A8AD; display:flex; align-items:center; justify-content:center; font-size:0.75rem; color:#6C151E !important; font-weight:800; flex-shrink:0; }
.msg-asst-name { font-size:0.72rem; font-weight:700; color:#6C151E !important; }
.msg-asst-role { font-size:0.68rem; color:#5C3A3E !important; margin-left:2px; }
.msg-asst-body {
    background:#FFFFFF;
    border:1px solid #E4E8EF;
    border-radius:4px 16px 16px 16px;
    padding:16px 18px;
    font-size:0.9rem;
    color:#111827 !important;
    line-height:1.7;
    word-break:break-word;
    overflow-wrap:anywhere;
}
.msg-asst-body h1,.msg-asst-body h2 { font-size:1.05rem !important; font-weight:700 !important; color:#111827 !important; margin:12px 0 6px 0 !important; padding-bottom:4px; border-bottom:1px solid #F3F4F6; }
.msg-asst-body h3 { font-size:0.95rem !important; font-weight:700 !important; color:#6C151E !important; margin:10px 0 5px 0 !important; }
.msg-asst-body h4 { font-size:0.88rem !important; font-weight:700 !important; color:#374151 !important; margin:8px 0 4px 0 !important; }
.msg-asst-body ul { padding-left:20px !important; margin:6px 0 8px 0 !important; }
.msg-asst-body ol { padding-left:20px !important; margin:6px 0 8px 0 !important; }
.msg-asst-body li { margin-bottom:4px !important; color:#111827 !important; line-height:1.6 !important; }
.msg-asst-body p { margin-bottom:8px !important; color:#111827 !important; line-height:1.7 !important; }
.msg-asst-body p:last-child { margin-bottom:0 !important; }
.msg-asst-body strong { color:#111827 !important; font-weight:700 !important; }
.msg-asst-body em { color:#374151 !important; font-style:italic !important; }
.msg-asst-body a { color:#6C151E !important; text-decoration:underline !important; }
.msg-asst-body hr { border:none !important; border-top:1px solid #E4E8EF !important; margin:10px 0 !important; }
.msg-asst-body blockquote { border-left:3px solid #6C151E !important; padding:4px 10px !important; color:#5C3A3E !important; font-style:italic !important; margin:6px 0 !important; background:#FBF5F0 !important; border-radius:0 6px 6px 0 !important; }
.msg-asst-body table { border-collapse:collapse !important; width:100% !important; font-size:0.83rem !important; margin:8px 0 !important; }
.msg-asst-body td { border:1px solid #E4E8EF !important; padding:7px 10px !important; color:#111827 !important; }
.msg-asst-body th { border:1px solid #D4A8AD !important; padding:7px 10px !important; background:#F9EEE9 !important; font-weight:700 !important; color:#6C151E !important; }
.msg-asst-body code { background:#F1F5F9 !important; color:#6C151E !important; border-radius:4px !important; padding:2px 6px !important; font-size:0.83em !important; font-family:'SFMono-Regular',Consolas,monospace !important; }
.msg-asst-body pre { background:#1E293B !important; border-radius:8px !important; padding:12px 16px !important; margin:10px 0 !important; overflow-x:auto !important; }
.msg-asst-body pre code { background:transparent !important; color:#E2E8F0 !important; padding:0 !important; font-size:0.82em !important; }
/* Source citation line at bottom of answer */
.msg-asst-body p em { font-size:0.78rem !important; color:#5C3A3E !important; }

/* ── EMPTY STATE ── */
.empty-center { text-align:center; padding:28px 16px 20px; }
.empty-eyebrow { font-size:0.66rem; font-weight:800; letter-spacing:2px; color:#6C151E !important; text-transform:uppercase; margin-bottom:5px; }
.empty-title { font-size:1.35rem; font-weight:700; color:#111827 !important; letter-spacing:-0.3px; }
.empty-hint { font-size:0.77rem; color:#5C3A3E !important; margin-top:14px; }

/* ── STUDY PLANNER FORM ── */
.plan-section-title { font-size:0.72rem; font-weight:700; letter-spacing:0.8px; color:#5C3A3E !important; text-transform:uppercase; margin-bottom:6px; margin-top:14px; }
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius:9px !important; border:1px solid #D1D5DB !important;
    background:#FFFFFF !important; color:#111827 !important;
    font-size:0.88rem !important; padding:9px 12px !important;
    box-shadow:none !important;
}
.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within,
.stDateInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color:#6C151E !important;
    box-shadow:0 0 0 3px rgba(108,21,30,.12) !important;
}
.stTextInput input::placeholder, .stDateInput input::placeholder, .stNumberInput input::placeholder { color:#9C7A7E !important; }
label[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] { font-size:0.82rem !important; font-weight:600 !important; color:#374151 !important; margin-bottom:4px !important; }

/* ── BUTTONS ── */
.stButton > button[kind="primary"] {
    background:#6C151E !important; color:#FFFFFF !important;
    font-weight:700 !important; font-size:0.88rem !important;
    border-radius:10px !important; border:none !important;
    padding:11px 20px !important; width:100% !important;
    box-shadow:0 2px 8px rgba(108,21,30,.28) !important;
    transition:all .18s ease !important; letter-spacing:0.1px !important;
}
.stButton > button:hover { background:#5A1019 !important; box-shadow:0 4px 14px rgba(108,21,30,.38) !important; transform:translateY(-1px) !important; }

/* ── New Chat button secondary ── */
.stButton > button[kind="secondary"] {
    text-align: left !important;
    background:transparent !important; color:#374151 !important;
    border:1px solid #D1D5DB !important; font-weight:600 !important;
    font-size:0.82rem !important; box-shadow:none !important;
    padding:7px 12px !important;
}
.stButton > button[kind="secondary"]:hover { color:#6C151E !important; border-color:#6C151E !important; background:#FBF5F0 !important; transform:none !important; }

/* ── PLAN CARDS ── */
.plan-day { background:#FFFFFF; border:1px solid #E4E8EF; border-left:4px solid #6C151E; border-radius:9px; padding:10px 13px; margin-bottom:8px; }
.plan-day-wknd { border-left-color:#0F3D3A; background:#F3F8F7; }
.plan-day-date { font-size:0.68rem; font-weight:800; color:#6C151E !important; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:2px; }
.plan-day-wknd .plan-day-date { color:#0F3D3A !important; }
.plan-day-subject { font-size:0.86rem; font-weight:600; color:#111827 !important; margin-bottom:2px; }
.plan-day-meta { font-size:0.74rem; color:#5C3A3E !important; }

/* ── BADGES ── */
.badge { display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; background:#F3F4F6; color:#374151 !important; border:1px solid #D1D5DB; }
.badge-green { background:#ECFDF5; color:#065F46 !important; border-color:#6EE7B7; }

/* ── METRICS ── */
[data-testid="stMetric"] { background:#F9EEE9; border:1px solid #D4A8AD; border-radius:10px; padding:10px 14px; }
[data-testid="stMetric"] label { font-size:0.72rem !important; color:#374151 !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.5px !important; }
[data-testid="stMetricValue"] { font-size:1.4rem !important; font-weight:800 !important; color:#111827 !important; }
[data-testid="stMetricDelta"] { font-size:0.82rem !important; font-weight:700 !important; }

/* ── DATE INPUT WHITE BACKGROUND ── */
.stDateInput > div > div > input { background:#FFFFFF !important; color:#1A0A0C !important; }
[data-baseweb="input"] input { background:#FFFFFF !important; color:#1A0A0C !important; }
[data-baseweb="base-input"] { background:#FFFFFF !important; }

/* ── DIVIDER ── */
div[data-testid="stChatInput"] { background:#FFFFFF !important; border-radius:12px !important; border:1.5px solid #D4A8AD !important; box-shadow:0 2px 10px rgba(108,21,30,.06) !important; }
div[data-testid="stChatInput"] textarea { font-size:0.88rem !important; color:#1A0A0C !important; background:#FFFFFF !important; }
div[data-testid="stChatInput"] textarea::placeholder { color:#9C7A7E !important; }
div[data-testid="stChatInput"]:focus-within { border-color:#6C151E !important; box-shadow:0 0 0 3px rgba(108,21,30,.1), 0 2px 10px rgba(108,21,30,.06) !important; }

/* ── DIVIDER ── */
hr { border:none !important; border-top:1px solid #D4A8AD !important; margin:12px 0 !important; }

/* ── FOOTER ── */
.page-footer { display:flex; justify-content:space-between; align-items:center; font-size:0.7rem; color:#5C3A3E !important; padding:8px 2px 0; margin-top:6px; border-top:1px solid #D4A8AD; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:transparent; } ::-webkit-scrollbar-thumb { background:#D4A8AD; border-radius:99px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = WorkflowPipeline()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = SystemEvaluator()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_study_plan" not in st.session_state:
    st.session_state.current_study_plan = None
if "vector_db_ready" not in st.session_state:
    vs = VectorStoreManager()
    if vs.vector_store is None:
        ingest_directory(DEFAULT_SAMPLE_DIR)
    st.session_state.vector_db_ready = True
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ── Branch data ────────────────────────────────────────────────────────────────
BRANCHES = {
    "Electronics & Communication (EC / ECE)": ["Basic Electronics (26ECE101)","Applied Digital Logic Design (26ECE111)","Analog & Digital Communication (25EC205)","Microcontrollers & Embedded Controllers (25EC206)","Signals and Systems (25EC203)","Digital Signal Processing (25EC302)"],
    "Computer Science & Engineering (CSE)": ["Data Structures (25CSE201)","OOP with Java (25CSE202)","Computer Architecture (25CSE211)","Database Management Systems (25CSE205)","Operating Systems (25CSE206)","Computer Networks (CS3001)"],
    "Information Science & Engineering (ISE)": ["Data Communication & Networking (IS3001-1)","Ethical Hacking & Defense (IS3002-1)","Machine Learning Foundations (IS2002-1)","Unix Shell & System Programming (IS1602-1)","Web Technologies (IS2504-1)","OS Fundamentals (IS3101-1)"],
    "AI & Data Science (AI & DS)": ["Mathematical Foundations (25MAT204)","Essentials of Data Science (25AID111)","Foundations of Machine Learning (25AID202)","Big Data Analytics (25AID203)","Deep Learning (25AID301)"],
    "AI & Machine Learning (AI & ML)": ["Principles of AI (25AIM101)","Machine Learning Algorithms (25AIM201)","Deep Learning & Neural Networks (25AIM301)","Computer Vision (25AIM302)"],
    "Cyber Security (CYB)": ["Fundamentals of Cyber Security (IS1101-1)","Applied Cryptography & PKI (CY2001)","Operating System Security (CS2006)","Network Security (CY3001)"],
    "Computer & Communication (CCE)": ["Digital Signal Processing (CC1001)","Analog & Digital Communication (25ECE101)","Computer Networks & Protocols (CC2001)","Embedded Systems (CC2002)"],
    "Robotics & AI (RAI)": ["Kinematics & Dynamics of Robots (RI1001)","Sensors & Actuators (RI1002)","Robot Control Systems (RI2001)","Autonomous Navigation (RI3002)"],
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def send_message(prompt: str):
    st.session_state.chat_history.append({"role":"user","content":prompt})
    with st.spinner("bloom. is thinking..."):
        result = st.session_state.pipeline.process_user_turn(
            query=prompt, history=st.session_state.chat_history,
            current_study_plan=st.session_state.current_study_plan)
    ans = result.get("final_answer") or "Sorry, I could not find an answer right now."
    new_plan = result.get("study_plan")
    if new_plan:
        st.session_state.current_study_plan = new_plan
    st.session_state.chat_history.append({
        "role":"assistant","content":ans,
        "intent":result.get("intent","document_qa"),
        "sources":result.get("retrieved_chunks",[])
    })

# ── Process pending prompt BEFORE rendering ────────────────────────────────────
if st.session_state.pending_prompt:
    _p = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    send_message(_p)

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav">
  <div class="nav-left">
    <div class="nav-logo-icon">✦</div>
    <div class="nav-logo-text">bloom<span>.</span></div>
    <div class="nav-tagline">Your academic companion</div>
  </div>
  <div class="nav-right">
    <div class="nav-status"><div class="nav-status-dot"></div> NMAMIT Academic AI · Always ready</div>
      </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
hero_l, hero_r = st.columns([0.78, 0.22])
with hero_l:
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">A little clarity. A lot of possibility.</div>
      <h1 class="hero-h1">Make room for your next <span class="hero-serif">bright idea.</span></h1>
      <p class="hero-sub">Your questions, your goals, your pace. Let's figure it out together.</p>
    </div>
    """, unsafe_allow_html=True)
with hero_r:
    st.markdown("""
    <div style="display:flex;justify-content:flex-end;align-items:flex-start;padding-top:6px;">
      <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#5C3A3E;font-weight:500;background:#FFFFFF;border:1px solid #E4E8EF;border-radius:99px;padding:6px 14px;">🍃 Made for your student journey</div>
    </div>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# MAIN TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_chat, col_plan = st.columns([0.60, 0.40], gap="large")

# ─── LEFT COLUMN: CHAT ────────────────────────────────────────────────────────
with col_chat:
    # Card header row
    hd1, hd2 = st.columns([0.75, 0.25])
    with hd1:
        st.markdown("""
        <div class="card-header" style="border:none;padding-bottom:10px;margin-bottom:10px;">
          <div class="card-header-left">
            <div class="card-icon">✦</div>
            <div>
              <div class="card-title">Your learning space</div>
              <div class="card-subtitle">Ask freely. Learn confidently.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with hd2:
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("+ New session", key="new_btn"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Empty state ────────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-center">
          <svg width="96" height="72" viewBox="0 0 120 90" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:14px;opacity:0.85;">
            <ellipse cx="60" cy="80" rx="40" ry="6" fill="#E8EDFF" opacity="0.7"/>
            <path d="M60 18C44 7 20 8 10 13V68C20 64 44 63 60 73C76 63 100 64 110 68V13C100 8 76 7 60 18Z" fill="#F9EEE9" stroke="#6C151E" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M60 20V72" stroke="#6C151E" stroke-width="2.5" stroke-linecap="round"/>
            <path d="M24 28C33 25 45 25 52 28" stroke="#C27A80" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M24 38C33 35 45 35 52 38" stroke="#C27A80" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M24 48C33 45 45 45 52 48" stroke="#E8C5C9" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M68 28C75 25 87 25 96 28" stroke="#C27A80" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M68 38C75 35 87 35 96 38" stroke="#C27A80" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M68 48C75 45 87 45 96 48" stroke="#E8C5C9" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M52 18V44L56 40L60 44V20" fill="#6C151E" opacity="0.8"/>
            <circle cx="108" cy="14" r="3" fill="#C27A80"/>
            <circle cx="14" cy="58" r="2.5" fill="#E8C5C9"/>
            <path d="M102 32L104 36L108 38L104 40L102 44L100 40L96 38L100 36L102 32Z" fill="#C27A80"/>
          </svg>
          <div class="empty-eyebrow">Curious minds welcome</div>
          <div class="empty-title">A good question is a start.</div>
        </div>
        """, unsafe_allow_html=True)

        # Action cards as Streamlit buttons styled to look like cards
        for key, icon, title, desc, prompt in [
            ("q1","📖","CSE Syllabus overview","What's covered in CSE this semester?","Summarize the CSE branch syllabus subjects and modules"),
            ("q2","📅","Build my study plan","Exam prep made structured","Create a 2-hour daily study plan for EC branch subjects"),
            ("q3","📋","Attendance & exam rules","Know your academic rights","What is the attendance requirement and what happens below 75%?"),
            ("q4","🧮","Attendance calculator","How many classes can I miss?","I attended 45 out of 60 classes. How many more do I need for 75%?"),
        ]:
            col_btn = st.container()
            with col_btn:
                st.markdown('<div class="card-btn">', unsafe_allow_html=True)
                if st.button(f"{icon}  {title}  ›\n{desc}", key=f"btn_{key}", use_container_width=True,
                             help=f"Ask: {prompt}"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="empty-hint">🍃 No question is too small.</div>', unsafe_allow_html=True)

    # ── Chat history ───────────────────────────────────────────────────────────
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                safe_content = html.escape(msg["content"])
                st.markdown(
                    '<div class="msg-wrap-user"><div class="msg-user">' + safe_content + '</div></div>',
                    unsafe_allow_html=True
                )
            else:
                # Clean raw answer text before rendering
                raw_content = msg["content"]
                # Remove any leftover literal \n sequences (from stringified message objects)
                import re as _re
                # Replace literal \\n with actual newline if present
                if "\\n" in raw_content:
                    raw_content = raw_content.replace("\\n", "\n")
                # Strip any [System]: / [Human]: / [AI]: prefixes leaking through
                raw_content = _re.sub(r"^\[(System|Human|AI|Assistant)\]:\s*", "", raw_content, flags=_re.MULTILINE)
                rendered_html = md_lib.markdown(
                    raw_content,
                    extensions=["tables", "fenced_code", "nl2br"]
                )
                intent_lbl = msg.get("intent","")
                intent_badge = ""
                if intent_lbl == "create_study_plan":
                    intent_badge = '<span style="font-size:0.65rem;background:#F9EEE9;color:#6C151E;border-radius:4px;padding:1px 6px;font-weight:700;margin-left:6px;">STUDY PLAN</span>'
                elif intent_lbl == "document_qa":
                    intent_badge = '<span style="font-size:0.65rem;background:#ECFDF5;color:#059669;border-radius:4px;padding:1px 6px;font-weight:700;margin-left:6px;">GROUNDED</span>'
                elif intent_lbl == "modify_study_plan":
                    intent_badge = '<span style="font-size:0.65rem;background:#FEF3C7;color:#92400E;border-radius:4px;padding:1px 6px;font-weight:700;margin-left:6px;">MODIFIED</span>'
                elif intent_lbl == "calculator":
                    intent_badge = '<span style="font-size:0.65rem;background:#FFF0F6;color:#C2255C;border-radius:4px;padding:1px 6px;font-weight:700;margin-left:6px;">CALCULATOR</span>'
                elif intent_lbl == "summarize":
                    intent_badge = '<span style="font-size:0.65rem;background:#F3F0FF;color:#0F3D3A;border-radius:4px;padding:1px 6px;font-weight:700;margin-left:6px;">SUMMARY</span>'
                bubble_html = (
                    '<div class="msg-wrap-asst">'
                    '<div class="msg-asst-outer">'
                    '<div class="msg-asst-header">'
                    '<div class="msg-asst-avatar">✦</div>'
                    '<div>'
                    '<span class="msg-asst-name">bloom.</span>'
                    '<span class="msg-asst-role">· NMAMIT Academic AI</span>'
                    + intent_badge +
                    '</div>'
                    '</div>'
                    '<div class="msg-asst-body">' + rendered_html + '</div>'
                    '</div>'
                    '</div>'
                )
                st.markdown(bubble_html, unsafe_allow_html=True)
                if msg.get("sources"):
                    src_list = msg["sources"][:3]
                    with st.expander(f"📎 {len(src_list)} source(s) · click to inspect", expanded=False):
                        for i, ch in enumerate(src_list):
                            score_pct = round(ch.get("score", 0) * 100, 1)
                            dtype = ch.get("doc_type", "").replace("_", " ").title()
                            st.markdown(f"""
<div style="background:#F9FAFB;border:1px solid #E4E8EF;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="font-size:0.7rem;background:#F9EEE9;color:#6C151E;border-radius:4px;padding:2px 7px;font-weight:700;">Source {i+1}</span>
    <span style="font-size:0.78rem;font-weight:700;color:#111827;">{ch["source"]}</span>
    <span style="font-size:0.7rem;color:#5C3A3E;margin-left:auto;">{dtype}</span>
  </div>
  <div style="font-size:0.75rem;color:#5C3A3E;line-height:1.5;font-family:monospace;background:#FFFFFF;border-radius:5px;padding:8px 10px;border:1px solid #F3F4F6;max-height:120px;overflow:hidden;">
    {html.escape(ch["content"][:250]).replace(chr(10), "<br>")}{"..." if len(ch["content"]) > 250 else ""}
  </div>
</div>
                            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-footer">
      <span>✦ bloom. · NMAMIT Academic AI</span>
      <span>Always verify important academic information officially.</span>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.chat_input("Ask about syllabus, attendance, exams, calculate classes, or create a study plan...")
    if user_input and user_input.strip():
        send_message(user_input.strip())
        st.rerun()


# ─── RIGHT COLUMN: STUDY PLANNER ─────────────────────────────────────────────
with col_plan:

    # ── Setup card ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-header" style="border:none;padding-bottom:10px;margin-bottom:10px;">
      <div class="card-header-left">
        <div class="card-icon">📅</div>
        <div>
          <div class="card-title">Study plan setup</div>
          <div class="card-subtitle">A plan that works around you.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="plan-section-title">📋 Your subjects</div>', unsafe_allow_html=True)
    chosen_branch = st.selectbox("Branch", options=list(BRANCHES.keys()), key="branch_sel", label_visibility="collapsed")
    subjects_val = ", ".join(BRANCHES[chosen_branch][:3])
    subjects_input = st.text_input("Subjects", value=subjects_val, placeholder="e.g. Mathematics, Data Structures", key="subj_input", label_visibility="collapsed")
    st.caption("Separate each subject with a comma.")

    st.markdown('<div class="plan-section-title">⏱ Available study time</div>', unsafe_allow_html=True)
    col_wk, col_we = st.columns(2)
    with col_wk:
        st.markdown('<div style="font-size:0.7rem;font-weight:700;color:#5C3A3E;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Weekday hrs/day</div>', unsafe_allow_html=True)
        study_hours = st.number_input("Weekday", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="hrs_input", label_visibility="collapsed")
    with col_we:
        st.markdown('<div style="font-size:0.7rem;font-weight:700;color:#5C3A3E;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Weekend hrs/day</div>', unsafe_allow_html=True)
        weekend_hours = st.number_input("Weekend", min_value=0.5, max_value=12.0, value=4.0, step=0.5, key="wknd_input", label_visibility="collapsed")

    st.markdown('<div class="plan-section-title">📅 Exam date</div>', unsafe_allow_html=True)
    exam_date = st.date_input("Exam date", value=datetime.date.today()+datetime.timedelta(days=14),
        min_value=datetime.date.today()+datetime.timedelta(days=1), key="exam_date_sel", label_visibility="collapsed")

    gen = st.button("✦  Generate study plan", key="gen_btn", use_container_width=True)
    if gen:
        subs = [s.strip() for s in subjects_input.split(",") if s.strip()] or ["Data Structures","Operating Systems"]
        inputs = {"subjects":subs,"weekday_hours":study_hours,"weekend_hours":weekend_hours,"exam_date":exam_date.strftime("%Y-%m-%d")}
        with st.spinner("Crafting your study plan..."):
            res = st.session_state.pipeline.process_user_turn(
                query=f"Create a study plan for {', '.join(subs)} on {exam_date}", history=[], planner_inputs=inputs)
        if res.get("study_plan"):
            st.session_state.current_study_plan = res["study_plan"]
            st.session_state.chat_history.append({"role":"assistant","content":res.get("final_answer","Your study plan is ready!"),"intent":"create_study_plan","sources":[]})
            st.rerun()

    st.markdown("""
    <div style="text-align:center;font-size:0.74rem;color:#5C3A3E;margin:8px 0 18px;">A little planning today. A calmer tomorrow.</div>
    """, unsafe_allow_html=True)

    # ── Active plan card ───────────────────────────────────────────────────────
    has_plan = bool(st.session_state.current_study_plan)
    badge_cls = "badge-green" if has_plan else "badge"
    badge_txt = "Active" if has_plan else "Not created yet"

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;padding-top:14px;border-top:1px solid #F3F4F6;">
      <div style="display:flex;align-items:center;gap:9px;">
        <span style="font-size:1rem;">📖</span>
        <span style="font-size:0.97rem;font-weight:700;color:#111827;">Your current plan</span>
      </div>
      <div class="{badge_cls}">{badge_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    if not has_plan:
        st.markdown("""
        <div style="background:#F9FAFB;border:1px solid #E4E8EF;border-radius:13px;padding:22px 18px;text-align:center;">
          <div style="font-size:2rem;margin-bottom:8px;">📅</div>
          <div style="font-size:0.93rem;font-weight:700;color:#111827;margin-bottom:5px;">Your next chapter starts here.</div>
          <div style="font-size:0.79rem;color:#9C7A7E;line-height:1.5;">Add a few details above, and I'll turn your goals into small, doable steps.</div>
          <div style="font-size:0.72rem;color:#9C7A7E;margin-top:12px;">✦ Built for you. Ready when you are.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        plan = st.session_state.current_study_plan
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Days Remaining", f"{plan.get('days_remaining',0)}")
        with m2:
            st.metric("Study Hours", f"{plan.get('total_available_hours',0)} hrs")
        st.caption(f"**Exam Date** · `{plan.get('exam_date','N/A')}`")
        items = plan.get("allocated_schedule",[]) or plan.get("daily_breakdown",[])
        for d in items[:14]:
            wknd = d.get("is_weekend",False)
            cls = "plan-day plan-day-wknd" if wknd else "plan-day"
            st.markdown(f"""
            <div class="{cls}">
              <div class="plan-day-date">{d.get('date','')} · {d.get('day_name','')[:3].upper()}</div>
              <div class="plan-day-subject">{d.get('allocated_subject','')}</div>
              <div class="plan-day-meta">{d.get('time_slot','')} · {d.get('available_hours','')} hrs</div>
            </div>
            """, unsafe_allow_html=True)
        if len(items) > 7:
            st.caption(f"Showing {min(14,len(items))} of {len(items)} days · Use the chat to see more or modify")

    st.markdown("""
    <div style="margin-top:16px;padding:13px 15px;background:#FFFFFF;border:1px solid #E4E8EF;border-radius:11px;">
      <div style="font-size:0.83rem;font-weight:700;color:#111827;margin-bottom:3px;">🍃 Progress, not perfection.</div>
      <div style="font-size:0.77rem;color:#5C3A3E;line-height:1.5;">Small steps today make a difference tomorrow. You've got this.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DEV TOOLS (collapsed)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("⚙️ Developer Tools & RAG Evaluation", expanded=False):
    t1, t2, t3 = st.tabs(["RAG vs Plain LLM", "Full Benchmark", "Knowledge Base"])
    with t1:
        st.markdown("""<div style="font-size:0.83rem;color:#5C3A3E;margin-bottom:12px;">
        Compare how the <strong>RAG pipeline</strong> (with retrieved document context) performs vs a 
        <strong>plain LLM</strong> (no document access). The RAG answer is grounded in official NITTE documents.
        </div>""", unsafe_allow_html=True)
        eq = st.text_input("Enter a question to compare:", value="What is the minimum attendance required?", key="dev_compare_q")
        if st.button("▶ Run RAG vs Plain LLM Comparison", key="dev_compare_btn", use_container_width=True):
            with st.spinner("Running both pipelines..."):
                r = st.session_state.evaluator.compare_rag_vs_plain(eq)
            c1, c2 = st.columns(2)
            with c1:
                rag_score = r["rag"]["groundedness_score"]
                score_color = "#059669" if rag_score >= 0.8 else "#D97706"
                st.markdown(f"""<div style="background:#ECFDF5;border:1px solid #6EE7B7;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
                  <div style="font-size:0.78rem;font-weight:800;color:#065F46;margin-bottom:4px;">✅ GROUNDED RAG ANSWER</div>
                  <div style="font-size:0.7rem;color:#047857;">Score: {rag_score} · Latency: {r["rag"]["latency_sec"]}s · Sources: {len(r["rag"].get("sources_retrieved", []))}</div>
                </div>""", unsafe_allow_html=True)
                import markdown as _md
                rag_html = _md.markdown(r["rag"]["answer"], extensions=["tables","nl2br"])
                st.markdown(f'<div style="font-size:0.85rem;color:#111827;line-height:1.6;">{rag_html}</div>', unsafe_allow_html=True)
                if r["rag"].get("sources_retrieved"):
                    st.caption(f"📎 Sources used: {', '.join(r['rag']['sources_retrieved'])}")
            with c2:
                st.markdown(f"""<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
                  <div style="font-size:0.78rem;font-weight:800;color:#92400E;margin-bottom:4px;">⚠️ PLAIN LLM (No Document Access)</div>
                  <div style="font-size:0.7rem;color:#B45309;">Latency: {r["plain_llm"]["latency_sec"]}s · Not grounded in NITTE docs</div>
                </div>""", unsafe_allow_html=True)
                plain_html = _md.markdown(r["plain_llm"]["answer"], extensions=["tables","nl2br"])
                st.markdown(f'<div style="font-size:0.85rem;color:#374151;line-height:1.6;">{plain_html}</div>', unsafe_allow_html=True)
                st.caption(f"⚠️ {r['plain_llm'].get('note', 'Generic answer — not from official documents')}")
    with t2:
        st.markdown("Run the full 5-category PRD benchmark suite to evaluate system accuracy and grounding.")
        if st.button("▶ Run Full Benchmark Suite (5 categories)", key="dev_benchmark_btn"):
            with st.spinner("Running full benchmark — this may take 30–60 seconds..."):
                bench_results = st.session_state.evaluator.run_full_benchmark()
            passed = sum(1 for r in bench_results if r["status"] == "PASSED")
            total = len(bench_results)
            st.metric("Benchmark Pass Rate", f"{passed}/{total}", delta=f"{round(passed/total*100)}%")
            for r in bench_results:
                is_pass = r["status"] == "PASSED"
                status_color = "#059669" if is_pass else "#D97706"
                bg_color = "#ECFDF5" if is_pass else "#FFF7ED"
                border_color = "#6EE7B7" if is_pass else "#FED7AA"
                status_text = "PASSED" if is_pass else "FAILED"
                status_dot = "#10B981" if is_pass else "#F59E0B"
                score_str = str(r["groundedness_score"])
                sources_str = ", ".join(r["sources"]) if r["sources"] else "None"
                ans_preview = r["final_answer"][:300].replace("<", "&lt;").replace(">", "&gt;")
                import html as _html_lib
                q_safe = _html_lib.escape(r["query"])
                st.markdown(
                    f'''<div style="background:{bg_color};border:1px solid {border_color};border-radius:10px;padding:14px 16px;margin-bottom:10px;">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="width:10px;height:10px;border-radius:50%;background:{status_dot};display:inline-block;flex-shrink:0;"></span>
    <span style="font-size:0.82rem;font-weight:700;color:#111827;">{r["category"]}</span>
    <code style="font-size:0.7rem;background:#E5E7EB;color:#374151;padding:1px 5px;border-radius:3px;">{r["test_id"]}</code>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="font-size:0.7rem;font-weight:700;color:{status_color};background:white;border:1px solid {border_color};border-radius:4px;padding:1px 7px;">{status_text}</span>
    <span style="font-size:0.7rem;color:#5C3A3E;">Score: <strong>{score_str}</strong></span>
  </div>
</div>
<div style="font-size:0.78rem;color:#374151;margin-bottom:6px;"><strong>Query:</strong> {q_safe}</div>
<div style="font-size:0.75rem;color:#5C3A3E;margin-bottom:4px;"><strong>Intent:</strong> {r["intent"]} &nbsp;|&nbsp; <strong>Latency:</strong> {r["latency_sec"]}s &nbsp;|&nbsp; <strong>Sources:</strong> {sources_str}</div>
<div style="font-size:0.78rem;color:#374151;background:#FFFFFF;border-radius:6px;padding:8px 10px;border:1px solid rgba(0,0,0,0.06);margin-top:6px;line-height:1.5;">{ans_preview}{"..." if len(r["final_answer"]) > 300 else ""}</div>
</div>''',
                    unsafe_allow_html=True
                )
    with t3:
        st.markdown("**📂 Upload new documents** to extend the knowledge base:")
        uploaded = st.file_uploader(
            "Upload PDF, TXT or Markdown files",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        if uploaded:
            if st.button("📥 Ingest uploaded files", key="ingest_uploads_btn"):
                ingested_count = 0
                for uf in uploaded:
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=DEFAULT_SAMPLE_DIR) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    # Rename to original filename in sample_documents
                    target_path = os.path.join(DEFAULT_SAMPLE_DIR, uf.name)
                    os.replace(tmp_path, target_path)
                    docs = load_file(target_path)
                    if docs:
                        vs = VectorStoreManager()
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                        chunks = splitter.split_documents(docs)
                        for idx, c in enumerate(chunks):
                            c.metadata["chunk_id"] = f"{uf.name}_c{idx}"
                        vs.add_documents(chunks)
                        ingested_count += len(chunks)
                st.success(f"Ingested {ingested_count} chunks from {len(uploaded)} file(s) into the knowledge base!")
                st.rerun()
        st.divider()
        st.markdown("**📚 Current knowledge base files:**")
        files = sorted(os.listdir(DEFAULT_SAMPLE_DIR)) if os.path.exists(DEFAULT_SAMPLE_DIR) else []
        for f in files:
            fpath = os.path.join(DEFAULT_SAMPLE_DIR, f)
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            st.markdown(f"• `{f}` — {size_kb} KB")
        if st.button("🔄 Re-index entire database", key="reindex_btn"):
            with st.spinner("Re-indexing all documents..."):
                n = ingest_directory(DEFAULT_SAMPLE_DIR)
            st.success(f"Re-indexed {n} chunks from all documents!")