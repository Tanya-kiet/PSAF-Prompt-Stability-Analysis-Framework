"""
streamlit_app.py — PSAF Professional Dashboard (Refined)
Prompt Stability Analysis Framework

Architecture:
  • All Groq calls happen ONLY when user clicks "Run Experiment"
  • Results cached in session_state and on disk (.psaf_cache/)
  • SentenceTransformer loaded once at module level
  • Tab switching never triggers new API calls
"""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from backend import (
    QuestionResult,
    FailedProviderResult,
    compute_category_stats,
    run_experiment,
    run_all_providers,
    MultiProviderResult,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSAF · Prompt Stability Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════
   DESIGN TOKENS — Light Professional
═══════════════════════════════════════ */
:root {
  --bg:           #F8FAFC;
  --surface:      #FFFFFF;
  --card:         #FFFFFF;
  --card-alt:     #F1F5F9;
  --border:       #E2E8F0;
  --border-focus: #93C5FD;
  --accent:       #2563EB;
  --accent-light: #EFF6FF;
  --accent-mid:   #DBEAFE;
  --purple:       #7C3AED;
  --purple-light: #F5F3FF;
  --teal:         #0D9488;
  --teal-light:   #F0FDFA;
  --orange:       #EA580C;
  --orange-light: #FFF7ED;
  --success:      #16A34A;
  --success-light:#F0FDF4;
  --warning:      #D97706;
  --warning-light:#FFFBEB;
  --error:        #DC2626;
  --error-light:  #FEF2F2;
  --text-1:       #0F172A;
  --text-2:       #475569;
  --text-3:       #64748B;
  --text-4:       #94A3B8;
  --text-5:       #CBD5E1;
  --text-section: #1E293B;
  --radius-s:     6px;
  --radius-m:     10px;
  --radius-l:     12px;
  --radius-xl:    16px;
  --shadow-s:     0 1px 2px rgba(0,0,0,0.05);
  --shadow-m:     0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

/* ═══════════════════════════════════════
   BASE
═══════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background: var(--bg) !important; color: var(--text-1); }
* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding-top: 0.75rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px;
}

/* ═══════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ═══════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Collapse the large reserved space above the logo (req #1) — keep just
   enough height for the native collapse-arrow control to remain usable */
section[data-testid="stSidebarHeader"] {
  height: 2.25rem !important;
  min-height: 2.25rem !important;
  padding: 0 !important;
}
section[data-testid="stSidebarUserContent"] {
  padding-top: 0.25rem !important;
}

/* Tighter, more intentional vertical rhythm inside the sidebar only */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.35rem !important;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] {
  margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stCheckbox {
  margin-bottom: 0.15rem !important;
}

/* Logo */
.sb-logo {
  padding: 1.25rem 1.25rem 1rem;
  border-bottom: 1px solid var(--border);
}
.sb-logo-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.02em;
}
.sb-logo-sub {
  font-size: 0.8rem;
  color: var(--text-3);
  margin-top: 0.2rem;
  line-height: 1.4;
}

/* Sidebar section heading */
.sb-section {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-2);
  padding: 0.45rem 1.25rem 0.4rem;
  display: block;
}

/* Sidebar card */
.sb-card {
  padding: 0 1.25rem;
  margin-bottom: 0.25rem;
}

/* sb-label: backward compat alias */
.sb-label {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-2);
  padding: 0.45rem 1.25rem 0.4rem;
  display: block;
}

/* ═══════════════════════════════════════
   PROVIDER STATUS
═══════════════════════════════════════ */
.provider-row {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid var(--border);
}
.provider-row:last-child { border-bottom: none; }
.provider-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-top: 0.35rem;
}
.dot-ok  { background: var(--success); }
.dot-err { background: var(--error); }
.provider-info { flex: 1; min-width: 0; }
.provider-name {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--text-1);
  line-height: 1.3;
}
.provider-meta {
  font-size: 0.78rem;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
  margin-top: 0.12rem;
}
.provider-status-tag {
  font-size: 0.75rem;
  font-weight: 400;
  color: var(--text-3);
  margin-top: 0.06rem;
}

/* ═══════════════════════════════════════
   BUTTONS
═══════════════════════════════════════ */
.stButton > button {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: var(--radius-m) !important;
  font-weight: 600 !important;
  font-size: 16px !important;
  padding: 0.55rem 1.25rem !important;
  width: 100% !important;
  height: 2.65rem !important;
  letter-spacing: 0.01em !important;
  transition: background 0.15s, transform 0.1s !important;
  box-shadow: var(--shadow-s) !important;
}
.stButton > button:hover {
  background: #1D4ED8 !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active {
  background: #1E40AF !important;
  transform: translateY(0) !important;
}
.stButton > button:disabled {
  background: var(--card-alt) !important;
  color: var(--text-4) !important;
  border: 1px solid var(--border) !important;
  opacity: 1 !important;
  box-shadow: none !important;
}

/* Secondary */
button[kind="secondary"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-2) !important;
  border-radius: var(--radius-m) !important;
}
button[kind="secondary"]:hover {
  background: var(--card-alt) !important;
  color: var(--text-1) !important;
}

/* ═══════════════════════════════════════
   FORM CONTROLS
═══════════════════════════════════════ */
.stSelectbox label,
.stSlider label,
.stCheckbox label span,
.stMultiSelect label {
  font-size: 15px !important;
  font-weight: 500 !important;
  color: var(--text-2) !important;
  margin-bottom: 0.3rem !important;
}
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-m) !important;
  color: var(--text-1) !important;
  font-size: 15px !important;
  box-shadow: var(--shadow-s) !important;
}
.stSelectbox > div > div:hover,
.stSelectbox > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"] {
  background: var(--accent) !important;
}

/* Checkbox */
[data-testid="stCheckbox"] input:checked + span {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* ═══════════════════════════════════════
   TABS
═══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--text-3) !important;
  font-size: 18px !important;
  font-weight: 500 !important;
  padding: 0.55rem 1.1rem !important;
  border-radius: 0 !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.12s !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-1) !important;
  background: var(--card-alt) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  font-weight: 600 !important;
  background: transparent !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ═══════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════ */
div[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-l) !important;
  padding: 1.1rem 1.25rem !important;
  box-shadow: var(--shadow-s) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
  color: var(--text-3) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--text-1) !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  letter-spacing: -0.025em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDeltaIcon-Up"],
div[data-testid="stMetric"] [data-testid="stMetricDeltaIcon-Down"] {
  display: none !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: var(--text-3) !important;
  font-size: 0.82rem !important;
}

/* ═══════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════ */
.stProgress > div > div > div {
  background: var(--accent) !important;
  border-radius: 99px !important;
}
.stProgress > div > div {
  background: var(--border) !important;
  border-radius: 99px !important;
  height: 3px !important;
}

/* ═══════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════ */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] details > summary,
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-l) !important;
  color: var(--text-1) !important;
  font-size: 15px !important;
  font-weight: 500 !important;
}
div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] details > summary * {
  color: var(--text-1) !important;
}
div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] details > summary:hover,
.streamlit-expanderHeader:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-light) !important;
}
div[data-testid="stExpander"] summary:hover *,
div[data-testid="stExpander"] details > summary:hover * {
  color: var(--accent) !important;
}
div[data-testid="stExpander"] details,
div[data-testid="stExpander"],
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-l) !important;
}

/* ═══════════════════════════════════════
   DATAFRAME
═══════════════════════════════════════ */
.stDataFrame {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-l) !important;
  box-shadow: var(--shadow-s) !important;
}
.stDataFrame th {
  background: var(--card-alt) !important;
  color: var(--text-3) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
}

/* ═══════════════════════════════════════
   DIVIDER
═══════════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }
section[data-testid="stSidebar"] hr { margin: 0.6rem 0 !important; }

/* ═══════════════════════════════════════
   STICKY HEADER
═══════════════════════════════════════ */
.psaf-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(248,250,252,0.95);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 0.65rem 0;
  margin-bottom: 1.25rem;
}
.psaf-header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.psaf-header-left { display: flex; align-items: center; gap: 0.85rem; }
.psaf-logo-icon {
  width: 40px; height: 40px;
  background: var(--accent-light);
  border: 1px solid var(--accent-mid);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem;
}
.psaf-title {
  font-size: 30px;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.02em;
  line-height: 1.25;
}
.psaf-subtitle { font-size: 16px; color: var(--text-3); margin-top: 0.15rem; }

/* ═══════════════════════════════════════
   PSI SCORE
═══════════════════════════════════════ */
.psi-score-large {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.5rem;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.03em;
}
.psi-high   { color: var(--success); }
.psi-medium { color: var(--warning); }
.psi-low    { color: var(--error); }

/* ═══════════════════════════════════════
   SECTION LABEL
═══════════════════════════════════════ */
.section-label {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: none;
  color: var(--text-section);
  padding-bottom: 0.65rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
  display: block;
}

/* ═══════════════════════════════════════
   STAGE PROGRESS
═══════════════════════════════════════ */
.stage-row {
  display: flex; align-items: center; gap: 0.65rem;
  padding: 0.45rem 0.1rem;
  font-size: 0.9rem;
  color: var(--text-3);
  border-bottom: 1px solid var(--border);
}
.stage-row:last-child { border-bottom: none; }
.stage-row.done   { color: var(--success); }
.stage-row.active { color: var(--text-1); font-weight: 500; }
.stage-icon { width: 16px; text-align: center; font-size: 0.9rem; flex-shrink: 0; }

/* ═══════════════════════════════════════
   INFO PILL
═══════════════════════════════════════ */
.info-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 0.32rem 0.9rem;
  font-size: 0.86rem;
  color: var(--text-2);
  margin-bottom: 1rem;
  background: var(--surface);
  box-shadow: var(--shadow-s);
}

/* ═══════════════════════════════════════
   VARIATION CARDS
═══════════════════════════════════════ */
.var-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-l);
  padding: 1rem 1.1rem;
  margin-bottom: 0.5rem;
  box-shadow: var(--shadow-s);
  transition: border-color 0.15s;
}
.var-card:hover { border-color: var(--accent); }
.var-card.original {
  background: var(--accent-light);
  border-color: var(--accent-mid);
}
.var-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  margin-bottom: 0.45rem;
}
.tag-orig { color: var(--accent); }
.tag-var  { color: var(--text-3); }
.var-text { color: var(--text-1); font-size: 15px; line-height: 1.65; }

/* ═══════════════════════════════════════
   RESPONSE DISPLAY
═══════════════════════════════════════ */
.resp-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 0.45rem;
}

/* ═══════════════════════════════════════
   EMPTY / WELCOME STATE
═══════════════════════════════════════ */
.empty-hero {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center;
  padding: 2.5rem 1.5rem 1.5rem;
  min-height: 50vh;
}
.empty-visual {
  font-size: 1.75rem;
  margin-bottom: 1rem;
  opacity: 0.6;
}
.empty-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.025em;
  margin-bottom: 0.45rem;
}
.empty-sub {
  font-size: 15px;
  color: var(--text-3);
  max-width: 420px;
  line-height: 1.65;
  margin-bottom: 1.75rem;
}
.steps-row {
  display: flex;
  border: 1px solid var(--border);
  border-radius: var(--radius-l);
  overflow: hidden;
  margin-bottom: 1.5rem;
  width: 100%;
  max-width: 580px;
  box-shadow: var(--shadow-s);
}
.step-item {
  flex: 1;
  padding: 0.9rem 1rem;
  border-right: 1px solid var(--border);
  background: var(--surface);
  text-align: left;
  transition: background 0.15s;
}
.step-item:hover { background: var(--card-alt); }
.step-item:last-child { border-right: none; }
.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem; color: var(--accent); font-weight: 600;
  letter-spacing: 0.04em; margin-bottom: 0.3rem;
}
.step-label { font-size: 0.88rem; color: var(--text-1); font-weight: 600; }
.step-desc  { font-size: 0.78rem; color: var(--text-3); margin-top: 0.2rem; }

/* ═══════════════════════════════════════
   CATEGORY STAT CARDS
═══════════════════════════════════════ */
.cat-stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-l);
  padding: 1rem 1.1rem;
  box-shadow: var(--shadow-s);
  transition: border-color 0.15s;
}
.cat-stat-card:hover { border-color: var(--accent); }
.cat-stat-name {
  font-size: 0.74rem; color: var(--text-3); text-transform: uppercase;
  letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.4rem;
}
.cat-stat-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.9rem; font-weight: 700; line-height: 1;
  margin-bottom: 0.25rem; letter-spacing: -0.025em;
}
.cat-stat-meta { font-size: 0.78rem; color: var(--text-3); }

/* ═══════════════════════════════════════
   RANK BADGE
═══════════════════════════════════════ */
.rank-pill {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; font-weight: 600;
  padding: 0.12rem 0.5rem; border-radius: 99px;
  margin-left: 0.35rem;
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid var(--accent-mid);
}

/* ═══════════════════════════════════════
   SUCCESS TOAST
═══════════════════════════════════════ */
.success-toast {
  background: var(--success-light);
  border: 1px solid #BBF7D0;
  border-radius: var(--radius-l);
  padding: 0.75rem 1rem;
  display: flex; align-items: center; gap: 0.55rem;
  font-size: 15px; color: var(--success);
  box-shadow: var(--shadow-s);
}

/* ═══════════════════════════════════════
   SKELETON
═══════════════════════════════════════ */
.skeleton-block {
  background: linear-gradient(90deg, var(--border) 25%, var(--card-alt) 50%, var(--border) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: var(--radius-s);
  margin-bottom: 0.5rem;
}
@keyframes shimmer {
  0%   { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* ═══════════════════════════════════════
   STATUS BADGES (kept for compat)
═══════════════════════════════════════ */
.status-badge { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "result": None,
    "all_results": {},
    "last_run_time": None,
    "last_run_calls": 0,
    "last_run_elapsed": None,
    "last_run_provider": None,
    "running": False,
    "execution_mode": "fast",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
def psi_class(score: float) -> str:
    return "psi-high" if score >= 75 else "psi-medium" if score >= 50 else "psi-low"

def psi_label(score: float) -> str:
    if score >= 75: return "Highly Stable"
    if score >= 50: return "Moderately Stable"
    return "Unstable"

def bar_color(score: float) -> str:
    if score >= 75: return "#16A34A"
    if score >= 50: return "#D97706"
    return "#DC2626"

def cat_color(score: float) -> str:
    return bar_color(score)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-title">🧬 PSAF</div>
      <div class="sb-logo-sub">Prompt Stability Analysis Framework</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Provider status ───────────────────────────────────────────────────────
    api_ok = bool(config.GROQ_API_KEY)

    import os as _os
    _openai_key_ok = bool(_os.getenv("OPENAI_API_KEY", ""))
    if not _openai_key_ok:
        try:
            _openai_key_ok = bool(st.secrets.get("OPENAI_API_KEY", ""))
        except Exception:
            pass

    groq_dot = "dot-ok"  if api_ok           else "dot-err"
    groq_tag = "Connected" if api_ok         else "No API key"
    oai_dot  = "dot-ok"  if _openai_key_ok   else "dot-err"
    oai_tag  = "Connected" if _openai_key_ok  else "No API key"

    st.markdown('<span class="sb-section">Providers</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin:0 1.25rem 0.75rem;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden;">
      <div class="provider-row">
        <div class="provider-dot {groq_dot}"></div>
        <div class="provider-info">
          <div class="provider-name">Groq</div>
          <div class="provider-status-tag">{groq_tag}</div>
          <div class="provider-meta">llama-3.1-8b-instant</div>
        </div>
      </div>
      <div class="provider-row" style="border-bottom:none;">
        <div class="provider-dot {oai_dot}"></div>
        <div class="provider-info">
          <div class="provider-name">OpenAI</div>
          <div class="provider-status-tag">{oai_tag}</div>
          <div class="provider-meta">gpt-4o-mini · Research only</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Experiment configuration ───────────────────────────────────────────────
    st.markdown('<span class="sb-section">Experiment</span>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        execution_mode = st.selectbox(
            "Mode",
            options=["fast", "research"],
            format_func=lambda x: {
                "fast":     "⚡ Fast — Groq only",
                "research": "🔬 Research — Groq + OpenAI",
            }.get(x, x),
            index=["fast", "research"].index(st.session_state["execution_mode"]),
            help=(
                "Fast: Groq only — instant real-time PSI.\n"
                "Research: Groq + OpenAI — cross-provider PSI comparison."
            ),
        )
        st.session_state["execution_mode"] = execution_mode

        categories = list(config.PROMPT_CATEGORIES.keys())
        selected_category = st.selectbox("Category", categories)
        prompts = config.PROMPT_CATEGORIES[selected_category]
        selected_prompt = st.selectbox("Question", prompts)
        n_variations = st.slider(
            "Variations",
            2, config.MAX_VARIATIONS, 3,
            help="Number of paraphrased versions to generate.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Run controls ──────────────────────────────────────────────────────────
    st.markdown('<span class="sb-section">Run</span>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="sb-card" style="padding-bottom:0.75rem;">', unsafe_allow_html=True)
        force_rerun = st.checkbox(
            "Bypass cache",
            help="Ignore cached results and re-call the API.",
        )
        run_clicked = st.button("▶  Run Experiment", disabled=not api_ok, use_container_width=True)
        if not api_ok:
            st.caption("⚠ Set GROQ_API_KEY to enable.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Session summary ───────────────────────────────────────────────────────
    n_cached = len(st.session_state.all_results)
    if n_cached > 0:
        st.divider()
        st.markdown('<span class="sb-section">Session</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:0 1.25rem 0.5rem;">
          <div style="font-size:0.88rem;color:#64748B;">
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
            color:#0F172A;">{n_cached}</span>
            &nbsp;question{"s" if n_cached != 1 else ""} cached
          </div>
          {"<div style='font-size:0.78rem;color:#64748B;margin-top:0.2rem;'>Last run: " + st.session_state.last_run_time + "</div>" if st.session_state.last_run_time else ""}
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑  Clear session", key="clear_session", use_container_width=True):
            st.session_state.all_results = {}
            st.session_state.result = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
STAGES = [
    ("Generating variations", "🔀"),
    ("Collecting LLM responses", "🤖"),
    ("Computing embeddings", "📐"),
    ("Calculating PSI score", "🎯"),
    ("Saving to cache", "💾"),
]

if run_clicked:
    stage_states = ["pending"] * len(STAGES)

    def render_stages():
        html = '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:0.75rem 1rem;">'
        for i, (label, icon) in enumerate(STAGES):
            state = stage_states[i]
            if state == "done":
                cls = "done"; ic = "✓"
            elif state == "active":
                cls = "active"; ic = "⟳"
            else:
                cls = ""; ic = icon
            html += f'<div class="stage-row {cls}"><span class="stage-icon">{ic}</span>{label}</div>'
        html += "</div>"
        return html

    messages_log = []

    with st.status("Running experiment…", expanded=True) as status_box:
        progress_bar = st.progress(0)
        stage_html_placeholder = st.empty()
        log_placeholder = st.empty()
        stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)

        def progress_cb(msg: str):
            messages_log.append(msg)
            pct = min(int(len(messages_log) / (n_variations + 4) * 100), 95)
            progress_bar.progress(pct)

            if "Generating" in msg and stage_states[0] == "pending":
                stage_states[0] = "active"
            elif "response" in msg.lower() and stage_states[0] == "active":
                stage_states[0] = "done"; stage_states[1] = "active"
            elif "response" in msg.lower() and stage_states[1] == "active":
                pass
            elif "PSI" in msg or "Comput" in msg:
                stage_states[1] = "done"; stage_states[2] = "active"
            elif "cached" in msg.lower() or "cache" in msg.lower():
                stage_states[2] = "done"; stage_states[3] = "done"; stage_states[4] = "active"

            stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)
            log_placeholder.markdown(
                f"<div style='font-size:0.85rem;color:#64748B;margin-top:0.4rem;'>"
                f"› {messages_log[-1]}</div>",
                unsafe_allow_html=True
            )

        t0 = time.time()

        # ── Execution Mode is the SOLE controller of provider selection ───────
        # fast     → Groq only (single provider, instant)
        # research → Groq + OpenAI (multi-provider comparison via run_all_providers)
        _execution_mode = st.session_state["execution_mode"]

        if _execution_mode == "research":
            # ── RESEARCH MODE: best-effort multi-model execution ──────────────
            # run_all_providers() only raises if EVERY provider fails.
            # Partial success (e.g. Groq OK, OpenAI timeout) returns a
            # MultiProviderResult with FailedProviderResult entries for failures.
            try:
                multi_result = run_all_providers(
                    category=selected_category,
                    prompt=selected_prompt,
                    n_variations=n_variations,
                    force_rerun=force_rerun,
                    progress_cb=progress_cb,
                    mode="research",
                )
            except Exception as _exc:
                # Only reached when ALL providers failed simultaneously.
                progress_bar.progress(0)
                status_box.update(label="Experiment failed", state="error", expanded=True)
                st.error(
                    f"❌ All providers failed — no results available.\n\n{_exc}\n\n"
                    "Check both GROQ_API_KEY and OPENAI_API_KEY."
                )
                st.stop()
            elapsed = time.time() - t0

            for i in range(len(STAGES)):
                stage_states[i] = "done"
            stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)
            progress_bar.progress(100)
            log_placeholder.empty()

            key = f"{selected_category}||{selected_prompt}"
            st.session_state.all_results[key] = multi_result

            # Surface first SUCCESSFUL provider's QuestionResult for single-result panels
            first_provider = next(iter(multi_result.successful_results))
            st.session_state.result = multi_result.successful_results[first_provider]
            st.session_state.last_run_time = time.strftime("%H:%M:%S")
            st.session_state.last_run_calls = (n_variations + 2) * len(multi_result.successful_results)
            st.session_state.last_run_elapsed = elapsed
            st.session_state.last_run_provider = " + ".join(
                p.upper() for p in multi_result.successful_results
            )

            # Build status summary — show SUCCESS / FAILED per provider
            success_parts = [
                f"<span style='color:#15803D;'>✓ {p.upper()}: {r.psi_score:.1f}</span>"
                for p, r in multi_result.successful_results.items()
            ]
            failed_parts = [
                f"<span style='color:#DC2626;'>✗ {p.upper()}: failed</span>"
                for p in multi_result.failed_results
            ]
            status_parts = "  &nbsp;|&nbsp;  ".join(success_parts + failed_parts)

            # Show partial-success warning if any provider failed
            any_failed = bool(multi_result.failed_results)
            toast_bg     = "#F0FDF4" if not any_failed else "#FFFBEB"
            toast_border = "#BBF7D0" if not any_failed else "#FDE68A"
            toast_icon   = "✓" if not any_failed else "⚠"

            st.markdown(f"""
            <div style="background:{toast_bg};border:1px solid {toast_border};
            border-radius:10px;padding:0.75rem 1rem;font-size:15px;">
              <span style="color:{'#15803D' if not any_failed else '#B45309'};">
                {toast_icon} &nbsp;Research Mode complete in {elapsed:.1f}s
              </span>
              <br/><span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;">
                {status_parts}
              </span>
            </div>
            """, unsafe_allow_html=True)

            status_box.update(
                label=f"Research Mode complete in {elapsed:.1f}s",
                state="error" if not multi_result.successful_results else "complete",
                expanded=False,
            )

        else:
            # ── FAST MODE: Groq only — provider_name is always "groq" ─────────
            # No user override possible. mode="fast" in run_experiment also enforces this.
            try:
                result = run_experiment(
                    category=selected_category,
                    prompt=selected_prompt,
                    n_variations=n_variations,
                    force_rerun=force_rerun,
                    progress_cb=progress_cb,
                    provider_name="groq",   # always groq in fast mode
                    mode="fast",
                )
            except Exception as _exc:
                progress_bar.progress(0)
                status_box.update(label="Experiment failed", state="error", expanded=True)
                st.error(
                    f"❌ Experiment failed: {_exc}\n\n"
                    "Check your GROQ_API_KEY. No silent fallback occurs — "
                    "errors surface here immediately."
                )
                st.stop()
            elapsed = time.time() - t0

            for i in range(len(STAGES)):
                stage_states[i] = "done"
            stage_html_placeholder.markdown(render_stages(), unsafe_allow_html=True)
            progress_bar.progress(100)
            log_placeholder.empty()

            key = f"{selected_category}||{selected_prompt}"
            st.session_state.all_results[key] = result
            st.session_state.result = result
            st.session_state.last_run_time = time.strftime("%H:%M:%S")
            st.session_state.last_run_calls = n_variations + 2
            st.session_state.last_run_elapsed = elapsed
            st.session_state.last_run_provider = "GROQ"

            st.markdown(f"""
            <div class="success-toast">
              ✓ &nbsp;Fast Mode complete in {elapsed:.1f}s —
              PSI Score: <strong style="font-family:'JetBrains Mono',monospace;">{result.psi_score:.1f}</strong>
            </div>
            """, unsafe_allow_html=True)

            status_box.update(
                label=f"Fast Mode complete in {elapsed:.1f}s — PSI {result.psi_score:.1f}",
                state="complete",
                expanded=False,
            )

    time.sleep(0.6)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + KEY METRICS
# ══════════════════════════════════════════════════════════════════════════════
result: QuestionResult | None = st.session_state.result

st.markdown(f"""
<div class="psaf-header">
  <div class="psaf-header-inner">
    <div class="psaf-header-left">
      <div class="psaf-logo-icon">🧬</div>
      <div>
        <div class="psaf-title">Prompt Stability Analysis Framework</div>
        <div class="psaf-subtitle">Research Dashboard</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if not result:
    st.markdown("""
    <div class="empty-hero">
      <div class="empty-visual">🧬</div>
      <div class="empty-title">Ready to analyze</div>
      <div class="empty-sub">
        Select a category and question in the sidebar, then click
        <strong style="color:#64748B;font-weight:500;">Run Experiment</strong>
        to measure prompt stability across paraphrased variations.
      </div>
      <div class="steps-row">
        <div class="step-item">
          <div class="step-num">01</div>
          <div class="step-label">Category + Question</div>
          <div class="step-desc">Choose from the sidebar</div>
        </div>
        <div class="step-item">
          <div class="step-num">02</div>
          <div class="step-label">Run</div>
          <div class="step-desc">Groq generates variations</div>
        </div>
        <div class="step-item">
          <div class="step-num">03</div>
          <div class="step-label">PSI Score</div>
          <div class="step-desc">Explore results across tabs</div>
        </div>
      </div>
      <div style="font-size:0.82rem;color:#64748B;">
        Results are cached — re-running the same question costs zero API calls.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════

# ── Key metrics row ────────────────────────────────────────────────────────
with st.container(border=True):
    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
    _psi_delta = None
    _psi_color = "normal"
    _m1.metric(
        "PSI Score",
        f"{result.psi_score:.1f}",
        delta=psi_label(result.psi_score),
        delta_color="off",
    )
    _m2.metric("Semantic Similarity", f"{result.semantic_similarity:.3f}")
    _m3.metric("Keyword Consistency", f"{result.keyword_consistency:.3f}")
    _m4.metric("Length Consistency",  f"{result.length_consistency:.3f}")
    _m5.metric("Variations", len(result.variations) - 1)

    st.divider()

    # ── Run metadata row — execution time, active provider, total responses ──
    _active_mr = st.session_state.all_results.get(f"{selected_category}||{selected_prompt}")
    _total_responses = len(result.variations)
    if isinstance(_active_mr, MultiProviderResult):
        _total_responses = sum(
            len(r.variations) for r in _active_mr.successful_results.values()
        )

    _r1, _r2, _r3 = st.columns(3)
    _r1.metric(
        "Execution Time",
        f"{st.session_state.last_run_elapsed:.1f}s" if st.session_state.last_run_elapsed is not None else "—",
    )
    _r2.metric("Active Provider", st.session_state.last_run_provider or "—")
    _r3.metric("Total Responses", _total_responses)

st.divider()

tabs = st.tabs([
    "📝  Prompt Variations",
    "💬  LLM Responses",
    "📊  Similarity Analysis",
    "🎯  PSI Breakdown",
    "📋  Category Comparison",
    "🔬  Comparison Dashboard",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Prompt Variations
# ───────────────────────────────────────────────────────────────────────────
with tabs[0]:
    col_info, col_q = st.columns([2, 3])
    with col_info:
        st.markdown(f"""
        <div class="info-pill">
          🔀 &nbsp;{len(result.variations) - 1} paraphrase{"s" if len(result.variations) != 2 else ""} generated
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:15px;color:#64748B;line-height:1.65;">
          Each variation rephrases the original question with different wording
          while preserving its meaning. The LLM receives each independently.
        </div>
        """, unsafe_allow_html=True)
    with col_q:
        st.markdown(f"""
        <div style="background:rgba(37,99,235,0.04);border:1px solid rgba(37,99,235,0.2);border-radius:10px;
        padding:0.875rem 1rem;">
          <div style="font-size:0.74rem;font-weight:700;letter-spacing:.06em;
          text-transform:uppercase;color:#2563EB;margin-bottom:.4rem;
          font-family:'JetBrains Mono',monospace;">Category</div>
          <div style="font-size:15px;color:#2563EB;font-weight:500;">{result.category}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    for i, vr in enumerate(result.variations):
        is_orig = i == 0
        tag_cls = "tag-orig" if is_orig else "tag-var"
        card_cls = "var-card original" if is_orig else "var-card"
        label = "ORIGINAL" if is_orig else f"VARIATION {i}"
        st.markdown(f"""
        <div class="{card_cls}">
          <div class="var-tag {tag_cls}">{label}</div>
          <div class="var-text">{vr.variation}</div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — LLM Responses
# ───────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("""
    <div class="info-pill">
      💬 &nbsp;Each variation was sent to Groq independently
    </div>
    """, unsafe_allow_html=True)

    for i, vr in enumerate(result.variations):
        label = "ORIGINAL" if i == 0 else f"VARIATION {i}"
        short = vr.variation[:70] + ("…" if len(vr.variation) > 70 else "")
        with st.expander(f"{label} — {short}", expanded=(i == 0)):
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
              <div class="resp-label">Prompt sent to LLM</div>
              <div style="background:rgba(37,99,235,0.04);border:1px solid rgba(37,99,235,0.2);border-radius:10px;
              padding:0.75rem;font-size:15px;color:#2563EB;line-height:1.55;">
                {vr.variation}
              </div>
            </div>
            <div>
              <div class="resp-label">LLM Response</div>
              <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
              padding:0.875rem;font-size:15px;color:#0F172A;line-height:1.7;white-space:pre-wrap;">
                {vr.response if vr.response else "*(no response received)*"}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Similarity Analysis
# ───────────────────────────────────────────────────────────────────────────
with tabs[2]:
    mat = np.array(result.similarity_matrix)
    n = mat.shape[0]
    labels = ["Original"] + [f"Var {i}" for i in range(1, n)]

    col_heat, col_bar = st.columns([1, 1])
    with col_heat:
        st.markdown("""
        <div class="info-pill" style="margin-bottom:0.75rem;">
          📊 &nbsp;Pairwise cosine similarity between responses
        </div>
        """, unsafe_allow_html=True)
        fig_heat = go.Figure(data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale=[
                [0.0,  "#FEF2F2"],
                [0.4,  "#FEF9C3"],
                [0.75, "#DCFCE7"],
                [1.0,  "#BBF7D0"],
            ],
            zmin=0, zmax=1,
            text=[[f"{mat[i, j]:.3f}" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
            textfont={"size": 10, "family": "JetBrains Mono", "color": "#0F172A"},
            showscale=True,
            colorbar=dict(
                tickcolor="#64748B", tickfont=dict(color="#64748B", size=10),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            ),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        fig_heat.update_xaxes(tickfont=dict(size=10, color="#64748B"), gridcolor="#E2E8F0")
        fig_heat.update_yaxes(tickfont=dict(size=10, color="#64748B"), gridcolor="#E2E8F0")
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_bar:
        if n > 1:
            sim_to_orig = [mat[0, j] for j in range(1, n)]
            bar_labels = [f"Variation {i}" for i in range(1, n)]
            st.markdown("""
            <div class="info-pill" style="margin-bottom:0.75rem;">
              🎯 &nbsp;Similarity to original response
            </div>
            """, unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=bar_labels,
                y=sim_to_orig,
                marker_color=[bar_color(s * 100) for s in sim_to_orig],
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{s:.3f}" for s in sim_to_orig],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color="#64748B"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
                yaxis=dict(
                  range=[0, 1.18],
                  gridcolor="#E2E8F0",
                  tickfont=dict(size=10),
                  title=dict(
                    text="Cosine similarity",
                    font=dict(size=11, color="#64748B")
                  )
                ),
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=10)),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Mean similarity metric — guarded against empty upper-triangle (single-response / partial data)
    idx_upper = np.triu_indices_from(mat, k=1)
    vals = mat[idx_upper]
    if vals.size == 0:
        mean_sim = 0.0
        min_sim  = 0.0
        max_sim  = 0.0
    else:
        mean_sim = float(np.mean(vals))
        min_sim  = float(np.min(vals))
        max_sim  = float(np.max(vals))
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Pairwise Similarity", f"{mean_sim:.3f}")
    m2.metric("Min Pairwise Similarity",  f"{min_sim:.3f}")
    m3.metric("Max Pairwise Similarity",  f"{max_sim:.3f}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — PSI Breakdown
# ───────────────────────────────────────────────────────────────────────────
with tabs[3]:
    # Key component metrics row
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("PSI Score", f"{result.psi_score:.1f}", delta=psi_label(result.psi_score), delta_color="off")
    _c2.metric("Semantic (50%)", f"{result.semantic_similarity:.3f}")
    _c3.metric("Keyword (30%)", f"{result.keyword_consistency:.3f}")
    _c4.metric("Length (20%)", f"{result.length_consistency:.3f}")

    st.divider()

    # Top score card
    score_col, info_col = st.columns([1, 2])
    with score_col:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
        padding:1.5rem;text-align:center;">
          <div style="font-size:0.78rem;font-weight:700;letter-spacing:.08em;
          text-transform:uppercase;color:#64748B;margin-bottom:0.75rem;">
            Prompt Stability Index
          </div>
          <div class="psi-score-large {psi_class(result.psi_score)}">
            {result.psi_score:.1f}
          </div>
          <div style="font-size:0.85rem;color:#64748B;margin-top:.25rem;">out of 100</div>
          <div style="margin-top:0.75rem;font-size:0.92rem;font-weight:500;
          color:{'#15803D' if result.psi_score >= 75 else '#B45309' if result.psi_score >= 50 else '#DC2626'};">
            {psi_label(result.psi_score)}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
        padding:1.25rem;font-size:15px;color:#64748B;line-height:1.75;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;
          background:#FFFFFF;border:1px solid #E2E8F0;border-radius:6px;
          padding:0.5rem 0.75rem;color:#2563EB;margin-bottom:0.75rem;">
            PSI = 100 × (0.50·S + 0.30·K + 0.20·L)
          </div>
          <table style="width:100%;font-size:15px;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #E2E8F0;">
              <td style="padding:0.4rem 0;color:#64748B;width:24px;">S</td>
              <td style="padding:0.4rem 0.5rem;color:#0F172A;font-weight:500;">Semantic similarity</td>
              <td style="padding:0.4rem 0;color:#2563EB;font-family:'JetBrains Mono',monospace;text-align:right;">50%</td>
              <td style="padding:0.4rem 0 0.4rem 1rem;color:#0F172A;font-family:'JetBrains Mono',monospace;text-align:right;">{result.semantic_similarity:.3f}</td>
            </tr>
            <tr style="border-bottom:1px solid #E2E8F0;">
              <td style="padding:0.4rem 0;color:#64748B;">K</td>
              <td style="padding:0.4rem 0.5rem;color:#0F172A;font-weight:500;">Keyword consistency</td>
              <td style="padding:0.4rem 0;color:#2563EB;font-family:'JetBrains Mono',monospace;text-align:right;">30%</td>
              <td style="padding:0.4rem 0 0.4rem 1rem;color:#0F172A;font-family:'JetBrains Mono',monospace;text-align:right;">{result.keyword_consistency:.3f}</td>
            </tr>
            <tr>
              <td style="padding:0.4rem 0;color:#64748B;">L</td>
              <td style="padding:0.4rem 0.5rem;color:#0F172A;font-weight:500;">Length consistency</td>
              <td style="padding:0.4rem 0;color:#2563EB;font-family:'JetBrains Mono',monospace;text-align:right;">20%</td>
              <td style="padding:0.4rem 0 0.4rem 1rem;color:#0F172A;font-family:'JetBrains Mono',monospace;text-align:right;">{result.length_consistency:.3f}</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Component chart
    st.markdown('<span class="section-label">Component Scores vs Weighted Contribution</span>',
                unsafe_allow_html=True)

    comps = {
        "Semantic (S)": (result.semantic_similarity, 0.50),
        "Keyword (K)": (result.keyword_consistency, 0.30),
        "Length (L)": (result.length_consistency, 0.20),
    }
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Raw score",
        x=list(comps.keys()),
        y=[v[0] for v in comps.values()],
        marker_color="#BFDBFE",
        marker_line_color="#2563EB",
        marker_line_width=1,
        text=[f"{v[0]:.3f}" for v in comps.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#64748B"),
    ))
    fig_comp.add_trace(go.Bar(
        name="Weighted contribution",
        x=list(comps.keys()),
        y=[v[0] * v[1] for v in comps.values()],
        marker_color="rgba(22,163,74,0.10)",
        marker_line_color="#16A34A",
        marker_line_width=1,
        text=[f"{v[0]*v[1]:.3f}" for v in comps.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#64748B"),
    ))
    fig_comp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
        yaxis=dict(
          range=[0, 1.2],
          gridcolor="#E2E8F0",
          tickfont=dict(size=10),
          title=dict(
          text="Score (0–1)",
          font=dict(size=11, color="#64748B")
          )
        ),
        xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color="#0F172A")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#64748B")),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Stability scale
    st.markdown('<span class="section-label">Stability Scale</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;gap:0;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden;">
      <div style="flex:1;padding:0.875rem 1rem;
        background:{'#F0FDF4' if result.psi_score >= 75 else '#FFFFFF'};
        border-right:1px solid #E2E8F0;">
        <div style="font-size:0.78rem;font-weight:700;letter-spacing:.06em;
        text-transform:uppercase;color:#15803D;margin-bottom:.3rem;">Highly Stable</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;color:#15803D;">75 – 100</div>
        <div style="font-size:0.85rem;color:#64748B;margin-top:.3rem;">Wording barely affects answer</div>
      </div>
      <div style="flex:1;padding:0.875rem 1rem;
        background:{'#FFFBEB' if 50 <= result.psi_score < 75 else '#FFFFFF'};
        border-right:1px solid #E2E8F0;">
        <div style="font-size:0.78rem;font-weight:700;letter-spacing:.06em;
        text-transform:uppercase;color:#B45309;margin-bottom:.3rem;">Moderately Stable</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;color:#B45309;">50 – 74</div>
        <div style="font-size:0.85rem;color:#64748B;margin-top:.3rem;">Some drift across paraphrases</div>
      </div>
      <div style="flex:1;padding:0.875rem 1rem;
        background:{'#FEF2F2' if result.psi_score < 50 else '#FFFFFF'};">
        <div style="font-size:0.78rem;font-weight:700;letter-spacing:.06em;
        text-transform:uppercase;color:#DC2626;margin-bottom:.3rem;">Unstable</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;color:#DC2626;">0 – 49</div>
        <div style="font-size:0.85rem;color:#64748B;margin-top:.3rem;">Wording significantly changes answer</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — Category Comparison
# ───────────────────────────────────────────────────────────────────────────
with tabs[4]:
    all_results = st.session_state.all_results

    if len(all_results) < 2:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:#64748B;">
          <div style="font-size:2rem;margin-bottom:0.75rem;">📋</div>
          <div style="font-size:1rem;color:#64748B;margin-bottom:.5rem;">
            Run experiments across multiple questions to compare categories here.
          </div>
          <div style="font-size:0.85rem;">
            Currently showing 1 result. Run at least 2 to enable comparison.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Table — flatten MultiProviderResult → QuestionResult so both fast and
        # research mode results are displayed correctly.
        rows = []
        for key, r in all_results.items():
            if isinstance(r, MultiProviderResult):
                # Research mode: show one row per provider
                for prov, qr in r.successful_results.items():
                    rows.append({
                        "Category": qr.category,
                        "Question": qr.original_prompt[:55] + ("…" if len(qr.original_prompt) > 55 else ""),
                        "Provider": prov.capitalize(),
                        "PSI Score": round(qr.psi_score, 1),
                        "Semantic": round(qr.semantic_similarity, 3),
                        "Keyword": round(qr.keyword_consistency, 3),
                        "Length": round(qr.length_consistency, 3),
                        "Stability": psi_label(qr.psi_score),
                        "Variations": len(qr.variations) - 1,
                    })
            else:
                rows.append({
                    "Category": r.category,
                    "Question": r.original_prompt[:55] + ("…" if len(r.original_prompt) > 55 else ""),
                    "Provider": "Groq",
                    "PSI Score": round(r.psi_score, 1),
                    "Semantic": round(r.semantic_similarity, 3),
                    "Keyword": round(r.keyword_consistency, 3),
                    "Length": round(r.length_consistency, 3),
                    "Stability": psi_label(r.psi_score),
                    "Variations": len(r.variations) - 1,
                })
        rows.sort(key=lambda x: x["PSI Score"], reverse=True)

        with st.container(border=True):
            st.dataframe(
                rows,
                use_container_width=True,
                column_config={
                    "PSI Score": st.column_config.ProgressColumn(
                        "PSI Score", min_value=0, max_value=100, format="%.1f"
                    ),
                    "Semantic": st.column_config.NumberColumn(format="%.3f"),
                    "Keyword":  st.column_config.NumberColumn(format="%.3f"),
                    "Length":   st.column_config.NumberColumn(format="%.3f"),
                },
                hide_index=True,
            )
        # compute_category_stats expects QuestionResult objects; passing
        # MultiProviderResult (which has no .psi_score or .category) crashes.
        _flat_for_stats: list[QuestionResult] = []
        for _v in all_results.values():
            if isinstance(_v, MultiProviderResult):
                _flat_for_stats.extend(_v.successful_results.values())
            else:
                _flat_for_stats.append(_v)
        cat_stats = compute_category_stats(_flat_for_stats)

        if len(cat_stats) >= 2:
            st.divider()
            st.markdown('<span class="section-label">Per-Category Averages</span>', unsafe_allow_html=True)

            cats_sorted = sorted(cat_stats.items(), key=lambda x: x[1]["avg"], reverse=True)
            stat_cols = st.columns(len(cats_sorted))
            for col, (cat, stats) in zip(stat_cols, cats_sorted):
                with col:
                    color = cat_color(stats["avg"])
                    st.markdown(f"""
                    <div class="cat-stat-card">
                      <div class="cat-stat-name">{cat.split()[0]}</div>
                      <div class="cat-stat-score" style="color:{color};">{stats['avg']:.1f}</div>
                      <span class="rank-pill">#{stats['rank']}</span>
                      <div class="cat-stat-meta" style="margin-top:0.5rem;">
                        {stats['count']} question{"s" if stats["count"] != 1 else ""} ·
                        Max {stats['max']:.1f} · Min {stats['min']:.1f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Chart
            avgs = [s["avg"] for _, s in cats_sorted]
            cat_names = [c.split()[0] for c, _ in cats_sorted]
            fig_cat = go.Figure(go.Bar(
                x=cat_names,
                y=avgs,
                text=[f"{a:.1f}" for a in avgs],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#64748B"),
                marker_color=[bar_color(a) for a in avgs],
                marker_line_color="rgba(0,0,0,0)",
            ))
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
                yaxis=dict(range=[0, 115], gridcolor="#E2E8F0",
                           tickfont=dict(size=10), title="Avg PSI",
                           titlefont=dict(size=11, color="#64748B")),
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color="#0F172A")),
                margin=dict(l=10, r=10, t=30, b=10),
                height=260,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 6 — Comparison Dashboard (Phase 6) + Export System (Phase 7)
# ───────────────────────────────────────────────────────────────────────────
with tabs[5]:

    # ── Helper: collect all MultiProviderResult entries from session ─────────
    def _get_multi_results() -> list[MultiProviderResult]:
        """Return all MultiProviderResult objects stored in session_state.all_results."""
        out = []
        for v in st.session_state.all_results.values():
            if isinstance(v, MultiProviderResult):
                out.append(v)
        return out

    def _avg_response_words(res) -> float:
        """Safely compute average response-word count from a QuestionResult.
        VariationResult has .variation (prompt text) and .response (LLM reply).
        Falls back gracefully for plain strings (backward-compatible).
        """
        if not res.variations:
            return 0.0
        def _words(v) -> int:
            if isinstance(v, str):
                return len(v.split())
            if hasattr(v, "response"):
                return len(v.response.split())
            if hasattr(v, "variation"):
                return len(v.variation.split())
            return len(str(v).split())
        return sum(_words(v) for v in res.variations) / len(res.variations)

    multi_runs = _get_multi_results()

    # ── Empty state ──────────────────────────────────────────────────────────
    if not multi_runs:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#64748B;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">🔬</div>
          <div style="font-size:1rem;font-weight:600;color:#64748B;margin-bottom:0.5rem;">
            No comparison data yet
          </div>
          <div style="font-size:0.92rem;color:#64748B;max-width:380px;margin:0 auto;">
            Switch to <strong style="color:#0F172A;">🧠 Research Mode</strong> in the sidebar
            and click <strong style="color:#0F172A;">▶ Run Experiment</strong>
            to generate multi-provider PSI comparison results.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Graceful disabled export buttons ────────────────────────────────
        st.divider()
        st.markdown('<span class="section-label">Export Results</span>', unsafe_allow_html=True)
        st.info("No data available for export — run an All Models experiment first.")
        exp_cols = st.columns(3)
        with exp_cols[0]:
            st.download_button("⬇  Download CSV", data="", file_name="psaf_comparison.csv",
                               mime="text/csv", disabled=True, use_container_width=True)
        with exp_cols[1]:
            st.download_button("⬇  Download JSON", data="", file_name="psaf_comparison.json",
                               mime="application/json", disabled=True, use_container_width=True)
        with exp_cols[2]:
            st.download_button("⬇  Download Report (TXT)", data="",
                               file_name="psaf_report.txt", mime="text/plain",
                               disabled=True, use_container_width=True)

    else:
        # Use the most-recent multi run as the "active" comparison
        active = multi_runs[-1]

        # ── Prompt header ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
                    padding:1rem 1.25rem;margin-bottom:1rem;">
          <div style="font-size:0.78rem;font-weight:700;letter-spacing:.08em;
                      text-transform:uppercase;color:#64748B;margin-bottom:0.35rem;">
            Compared Prompt
          </div>
          <div style="font-size:1rem;color:#0F172A;font-weight:500;">
            {active.prompt}
          </div>
          <div style="font-size:0.85rem;color:#64748B;margin-top:0.35rem;">
            Category: {active.category}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Build ranked list ─────────────────────────────────────────────────
        ranked = sorted(
            active.successful_results.items(),
            key=lambda kv: kv[1].psi_score,
            reverse=True,
        )

        def _stability_label(score: float) -> str:
            if score >= 75:
                return "Highly Stable"
            if score >= 50:
                return "Moderately Stable"
            return "Unstable"

        def _medal(rank: int) -> str:
            return ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"

        # ── Show failed provider alerts (partial-success mode) ────────────────
        failed_providers = active.failed_results
        if failed_providers:
            for fp_name, fp in failed_providers.items():
                st.markdown(f"""
                <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;
                            padding:0.75rem 1rem;margin-bottom:0.5rem;font-size:15px;">
                  <span style="color:#DC2626;font-weight:600;">
                    ✗ {fp_name.upper()} — Provider failed (partial results shown)
                  </span><br/>
                  <span style="color:#64748B;font-size:0.85rem;font-family:'JetBrains Mono',monospace;">
                    {fp.error}
                  </span>
                </div>
                """, unsafe_allow_html=True)

        # ── Medal ranking row ─────────────────────────────────────────────────
        st.markdown('<span class="section-label">Provider Ranking</span>', unsafe_allow_html=True)
        medal_cols = st.columns(len(ranked))
        for idx, (prov, res) in enumerate(ranked):
            with medal_cols[idx]:
                stability = _stability_label(res.psi_score)
                color = "#16A34A" if res.psi_score >= 75 else ("#D97706" if res.psi_score >= 50 else "#DC2626")
                text_color = "#15803D" if res.psi_score >= 75 else ("#B45309" if res.psi_score >= 50 else "#DC2626")
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
                            padding:1.1rem;text-align:center;">
                  <div style="font-size:2rem;margin-bottom:0.4rem;">{_medal(idx)}</div>
                  <div style="font-size:0.95rem;font-weight:600;color:#0F172A;
                              text-transform:capitalize;">{prov}</div>
                  <div style="font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;
                              color:{color};margin:0.4rem 0;">{res.psi_score:.1f}</div>
                  <div style="font-size:0.82rem;color:{text_color};">{stability}</div>
                </div>
                """, unsafe_allow_html=True)


        # ── Comparison table ──────────────────────────────────────────────────
        st.markdown('<span class="section-label">Provider Comparison Table</span>', unsafe_allow_html=True)
        table_rows = []
        for idx, (prov, res) in enumerate(ranked):
            avg_words = (
                _avg_response_words(res)
                if res.variations else 0
            )
            table_rows.append({
                "Rank":     f"#{idx + 1}",
                "Provider": prov.capitalize(),
                "PSI Score": round(res.psi_score, 1),
                "Semantic":  round(res.semantic_similarity, 3),
                "Keyword":   round(res.keyword_consistency, 3),
                "Length":    round(res.length_consistency, 3),
                "Avg Words": round(avg_words, 1),
                "Stability": _stability_label(res.psi_score),
            })

        with st.container(border=True):
            st.dataframe(
                table_rows,
                use_container_width=True,
                column_config={
                    "PSI Score": st.column_config.ProgressColumn(
                        "PSI Score", min_value=0, max_value=100, format="%.1f"
                    ),
                    "Semantic": st.column_config.NumberColumn(format="%.3f"),
                    "Keyword":  st.column_config.NumberColumn(format="%.3f"),
                    "Length":   st.column_config.NumberColumn(format="%.3f"),
                },
                hide_index=True,
            )


        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown('<span class="section-label">Visual Comparison</span>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        provider_labels = [p.capitalize() for p, _ in ranked]
        psi_values      = [r.psi_score for _, r in ranked]
        bar_colors_psi  = [
            "#16A34A" if v >= 75 else ("#D97706" if v >= 50 else "#DC2626")
            for v in psi_values
        ]

        with ch1:
            fig_psi = go.Figure(go.Bar(
                x=provider_labels,
                y=psi_values,
                marker_color=bar_colors_psi,
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{v:.1f}" for v in psi_values],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#64748B"),
            ))
            fig_psi.update_layout(
                title=dict(text="PSI Score by Provider", font=dict(size=12, color="#64748B")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
                yaxis=dict(range=[0, 115], gridcolor="#E2E8F0",
                           tickfont=dict(size=10), title="PSI Score",
                           titlefont=dict(size=11, color="#64748B")),
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color="#0F172A")),
                margin=dict(l=10, r=10, t=35, b=10),
                height=280,
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        with ch2:
            avg_words_list = []
            for _, res in ranked:
                avg_words_list.append(
                    _avg_response_words(res)
                    if res.variations else 0
                )
            fig_words = go.Figure(go.Bar(
                x=provider_labels,
                y=avg_words_list,
                marker_color="#BFDBFE",
                marker_line_color="#2563EB",
                marker_line_width=1,
                text=[f"{w:.0f}" for w in avg_words_list],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#64748B"),
            ))
            fig_words.update_layout(
                title=dict(text="Avg Response Length (words)", font=dict(size=12, color="#64748B")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
                yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=10),
                           title="Avg Words", titlefont=dict(size=11, color="#64748B")),
                xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color="#0F172A")),
                margin=dict(l=10, r=10, t=35, b=10),
                height=280,
            )
            st.plotly_chart(fig_words, use_container_width=True)

        # ── PSI component breakdown chart ─────────────────────────────────────
        st.markdown('<span class="section-label">PSI Component Breakdown</span>', unsafe_allow_html=True)
        fig_comp = go.Figure()
        comp_defs = [
            ("Semantic Similarity", [r.semantic_similarity for _, r in ranked], "#BFDBFE", "#2563EB"),
            ("Keyword Consistency", [r.keyword_consistency for _, r in ranked], "#DCFCE7", "#16A34A"),
            ("Length Consistency",  [r.length_consistency  for _, r in ranked], "rgba(210,153,34,0.12)", "#D97706"),
        ]
        for comp_name, vals, fill, line in comp_defs:
            fig_comp.add_trace(go.Bar(
                name=comp_name,
                x=provider_labels,
                y=vals,
                marker_color=fill,
                marker_line_color=line,
                marker_line_width=1,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=9, color="#64748B"),
            ))
        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#64748B", family="Inter, system-ui, sans-serif"),
            yaxis=dict(range=[0, 1.25], gridcolor="#E2E8F0", tickfont=dict(size=10),
                       title="Score (0–1)", titlefont=dict(size=11, color="#64748B")),
            xaxis=dict(gridcolor="#E2E8F0", tickfont=dict(size=11, color="#0F172A")),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#64748B")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── All compared prompts (when ≥2 multi-runs exist) ───────────────────
        if len(multi_runs) >= 2:
            st.markdown('<span class="section-label">All Compared Prompts</span>', unsafe_allow_html=True)
            hist_rows = []
            for mr in multi_runs:
                mr_ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
                best_prov, best_res = mr_ranked[0]
                hist_rows.append({
                    "Prompt":     mr.prompt[:60] + ("…" if len(mr.prompt) > 60 else ""),
                    "Category":   mr.category,
                    "Best Model": best_prov.capitalize(),
                    "Best PSI":   round(best_res.psi_score, 1),
                    **{p.capitalize() + " PSI": round(r.psi_score, 1) for p, r in mr_ranked},
                })
            hist_rows.sort(key=lambda x: x["Best PSI"], reverse=True)
            with st.container(border=True):
                st.dataframe(hist_rows, use_container_width=True, hide_index=True)

        # ── PHASE 7 — Export Section ──────────────────────────────────────────
        st.divider()
        st.markdown('<span class="section-label">Export Results</span>', unsafe_allow_html=True)

        # ── Build export payloads ────────────────────────────────────────────

        # 1. CSV — comparison table for the active multi run
        def _build_csv(mr: MultiProviderResult) -> str:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["Rank", "Provider", "PSI Score", "Semantic Similarity",
                             "Keyword Consistency", "Length Consistency",
                             "Avg Response Words", "Stability", "Prompt", "Category"])
            _ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
            for rank_idx, (prov, res) in enumerate(_ranked):
                avg_w = (
                    _avg_response_words(res)
                    if res.variations else 0
                )
                writer.writerow([
                    rank_idx + 1,
                    prov.capitalize(),
                    round(res.psi_score, 2),
                    round(res.semantic_similarity, 4),
                    round(res.keyword_consistency, 4),
                    round(res.length_consistency, 4),
                    round(avg_w, 1),
                    _stability_label(res.psi_score),
                    mr.prompt,
                    mr.category,
                ])
            return buf.getvalue()

        # 2. JSON — full structured results
        def _build_json(mr: MultiProviderResult) -> str:
            _ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
            payload = {
                "prompt":    mr.prompt,
                "category":  mr.category,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "results": {
                    prov: {
                        "psi_score":           round(res.psi_score, 4),
                        "semantic_similarity":  round(res.semantic_similarity, 4),
                        "keyword_consistency":  round(res.keyword_consistency, 4),
                        "length_consistency":   round(res.length_consistency, 4),
                        "stability":            _stability_label(res.psi_score),
                        "num_variations":       len(res.variations),
                        "response_length_avg_words": round(
                            _avg_response_words(res)
                            if res.variations else 0, 1
                        ),
                    }
                    for prov, res in mr.successful_results.items()
                },
                "ranking": [
                    {"rank": i + 1, "provider": prov, "psi_score": round(res.psi_score, 4)}
                    for i, (prov, res) in enumerate(_ranked)
                ],
            }
            return json.dumps(payload, indent=2)

        # 3. TXT — human-readable ranking report
        def _build_txt(mr: MultiProviderResult) -> str:
            _ranked = sorted(mr.successful_results.items(), key=lambda kv: kv[1].psi_score, reverse=True)
            lines = [
                "=" * 52,
                "  PSAF Comparison Report",
                "  Prompt Stability Analysis Framework",
                "=" * 52,
                "",
                f"Prompt   : {mr.prompt}",
                f"Category : {mr.category}",
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                "",
                "-" * 52,
                "Ranking",
                "-" * 52,
            ]
            medals = ["🥇 Best Model", "🥈 Second", "🥉 Third"]
            for i, (prov, res) in enumerate(_ranked):
                label = medals[i] if i < len(medals) else f"#{i+1}"
                lines.append(f"  {label}: {prov.capitalize()}")
            lines += ["", "-" * 52, "Scores", "-" * 52]
            for prov, res in _ranked:
                lines.append(f"  {prov.capitalize():<12} PSI: {res.psi_score:6.2f}  "
                             f"({_stability_label(res.psi_score)})")
            lines += ["", "-" * 52, "Component Detail", "-" * 52]
            for prov, res in _ranked:
                lines.append(f"  {prov.capitalize()}")
                lines.append(f"    Semantic Similarity : {res.semantic_similarity:.4f}  (weight 50%)")
                lines.append(f"    Keyword Consistency : {res.keyword_consistency:.4f}  (weight 30%)")
                lines.append(f"    Length Consistency  : {res.length_consistency:.4f}  (weight 20%)")
            lines += ["", "=" * 52, "  End of Report", "=" * 52, ""]
            return "\n".join(lines)

        # ── Render three download buttons ────────────────────────────────────
        csv_data  = _build_csv(active)
        json_data = _build_json(active)
        txt_data  = _build_txt(active)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        exp_c1, exp_c2, exp_c3 = st.columns(3)

        with exp_c1:
            st.download_button(
                label="⬇  Download CSV",
                data=csv_data,
                file_name=f"psaf_comparison_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Comparison table with PSI scores, components, and ranking.",
            )

        with exp_c2:
            st.download_button(
                label="⬇  Download JSON",
                data=json_data,
                file_name=f"psaf_comparison_{ts}.json",
                mime="application/json",
                use_container_width=True,
                help="Full structured results including all PSI components and ranking.",
            )

        with exp_c3:
            st.download_button(
                label="⬇  Download Report (TXT)",
                data=txt_data,
                file_name=f"psaf_report_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Human-readable ranking report with scores and stability labels.",
            )

        st.markdown(
            "<div style='font-size:0.85rem;color:#64748B;margin-top:0.4rem;'>"
            "Exports reflect the most recent All Models run shown above. "
            "Re-run an experiment to refresh.</div>",
            unsafe_allow_html=True,
        )
