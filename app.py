"""
AegisGraph Sentinel 2.0 - Streamlit Web Application
Real-time Fraud Detection Interface
"""
# Updated: May 17, 2026

import logging
logger = logging.getLogger(__name__)
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
import requests
import json
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from datetime import timezone
import time
import os
import random
import numpy as np
import networkx as nx

# Lazy loading heavy visualization and graph modules implemented inline where possible
# Page configuration
st.set_page_config(
    page_title="AegisGraph Sentinel 2.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MAX_BATCH_UPLOAD_BYTES = 5 * 1024 * 1024
BATCH_PREVIEW_ROWS = 10
BATCH_CHUNK_SIZE = 50
BATCH_MAX_ROWS = 500
COMMAND_CENTER_REFRESH_KEY = "command_center_live_refresh"
COMMAND_CENTER_IO_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _cache_data(ttl: int):
    """Return a cache decorator that degrades gracefully on older Streamlit builds."""
    cache_data = getattr(st, "cache_data", None)
    if cache_data is None:
        def passthrough(fn):
            return fn
        return passthrough
    return cache_data(ttl=ttl)


def _accessible_status(emoji: str, label: str) -> str:
    """Return a visual status with an adjacent plain-text equivalent."""
    return f"{emoji} {label} ({label})"


def _build_batch_transaction(row, index: int) -> dict:
    """Normalize one CSV row into the fraud-check request payload."""
    return {
        "transaction_id": str(row.get("transaction_id", f"TXN_{index}")),
        "source_account": str(row.get("source_account", "unknown")),
        "target_account": str(row.get("target_account", "unknown")),
        "amount": float(row.get("amount", 0)),
        "currency": str(row.get("currency", "INR")),
        "mode": str(row.get("mode", "UPI")),
        "timestamp": str(row.get("timestamp", datetime.now(timezone.utc).isoformat() + "Z")),
    }


def _estimate_csv_rows(uploaded_file) -> int:
    """Count CSV rows in bounded chunks without materializing the file."""
    uploaded_file.seek(0)
    total_rows = 0
    for chunk in pd.read_csv(uploaded_file, chunksize=BATCH_CHUNK_SIZE):
        total_rows += len(chunk)
        if total_rows >= BATCH_MAX_ROWS:
            break
    uploaded_file.seek(0)
    return total_rows


def _schedule_live_refresh(interval_ms: int = 1500) -> None:
    """Request a non-blocking dashboard refresh when the helper is available."""
    if st_autorefresh is not None:
        st_autorefresh(interval=interval_ms, key=COMMAND_CENTER_REFRESH_KEY)


@_cache_data(ttl=20)
def _fetch_health_snapshot(api_url: str) -> dict:
    response = requests.get(f"{api_url}/health", timeout=2)
    return response.json() if response.status_code == 200 else {}


@_cache_data(ttl=5)
def _fetch_stats_snapshot(api_url: str) -> dict:
    response = requests.get(f"{api_url}/stats", timeout=5)
    return response.json() if response.status_code == 200 else {}


def _build_live_event(api_url: str, txn: dict) -> dict | None:
    """Execute one live fraud check off the UI thread and shape the dashboard payload."""
    try:
        start_t = time.time()
        resp = requests.post(f"{api_url}/api/v1/fraud/check", json=txn, timeout=2)
        latency = int((time.time() - start_t) * 1000)
        if resp.status_code != 200:
            return None

        result = resp.json()
        return {
            "time": datetime.now().strftime("%H:%M:%S"),
            "id": txn["transaction_id"],
            "amount": txn["amount"],
            "decision": result.get("decision", "ALLOW"),
            "risk": result.get("risk_score", 0.0),
            "latency": latency,
            "explanation": result.get("explanation", ""),
            "breakdown": result.get("breakdown", {}),
        }
    except Exception:
        return None


def _advance_timed_state(
    state_key: str,
    timestamp_key: str,
    interval_seconds: float,
    max_steps: int,
    loop: bool = True,
) -> bool:
    """Advance a session-state animation step when enough time has elapsed."""
    if max_steps <= 0:
        return False

    now = datetime.now(timezone.utc)
    last_tick = st.session_state.get(timestamp_key)
    if last_tick is None:
        st.session_state[timestamp_key] = now
        return False

    elapsed = (now - last_tick).total_seconds()
    if elapsed < interval_seconds:
        return False

    current_step = int(st.session_state.get(state_key, 0))
    steps_to_advance = max(1, int(elapsed // interval_seconds))

    if loop:
        next_step = (current_step + steps_to_advance) % max_steps
    else:
        next_step = min(current_step + steps_to_advance, max_steps - 1)

    st.session_state[state_key] = next_step
    st.session_state[timestamp_key] = now
    return next_step != current_step

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #2dd4bf 0%, #0f766e 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        margin-bottom: 2px;
        letter-spacing: -0.04em;
        text-shadow: 0 10px 30px rgba(045, 212,191, 0.15);
    }
    
    /* Sleek Glassmorphism Metric Cards */
    [data-testid="stMetric"], .metric-card {
        background: rgba(22, 27, 48, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    [data-testid="stMetric"]:hover, .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(45, 212, 191, 0.45) !important;
        box-shadow: 0 15px 40px 0 rgba(56, 189, 248, 0.2) !important;
        background: rgba(22, 27, 48, 0.65) !important;
    }
    
    /* Modern Streamlit Alerts styling */
    .stAlert {
        background: rgba(22, 27, 48, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 5px solid #0f766e !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.3) !important;
    }
     /* Micro-animations and active elements */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e 0%, #0284c7 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
    }
    
    /* Alert Center Styling */
    .alert-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    @media (prefers-reduced-motion: no-preference) {
        .alert-card {
            animation: slideIn 0.3s ease-out forwards;
            transition: transform 0.2s;
        }
        .alert-card:hover {
            transform: translateX(5px);
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
    }
    .severity-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .severity-Low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .severity-Medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .severity-High { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid #f97316; }
    .severity-Critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    @media (prefers-reduced-motion: no-preference) {
        .severity-Critical { animation: pulse 2s infinite; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
    }
    .alert-time { font-family: monospace; color: #94a3b8; font-size: 0.85rem; }
    .alert-title { font-weight: 600; color: #f1f5f9; margin: 0 12px; flex-grow: 1; }
    
    button[kind="primary"]:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 25px rgba(056, 189,248, 0.5) !important;
        background: linear-gradient(135deg, #14b8a6 0%, #f59e0b 100%) !important;
    }
    
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(16, 185, 129, 0.2) !important;
        border-color: #10b981 !important;
        transform: scale(1.03);
    }
    
    /* Navigation Sidebar Enhancements */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    

    /* Interactive icon navigation */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 12px !important;
        padding: 8px 10px !important;
        margin: 6px 0 !important;
        background: rgba(15, 23, 42, 0.36) !important;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        transform: translateX(4px);
        border-color: rgba(45, 212, 191, 0.55) !important;
        background: rgba(20, 184, 166, 0.12) !important;
        cursor: pointer;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:focus-within,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) {
        outline: 3px solid rgba(45, 212, 191, 0.95) !important;
        outline-offset: 2px !important;
        border-color: rgba(45, 212, 191, 0.9) !important;
        background: rgba(20, 184, 166, 0.16) !important;
        box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.18) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
        border-color: rgba(45, 212, 191, 0.9) !important;
        background: linear-gradient(135deg, rgba(8, 47, 73, 0.95) 0%, rgba(15, 118, 110, 0.35) 100%) !important;
        box-shadow: inset 0 0 0 1px rgba(45, 212, 191, 0.35), 0 8px 24px rgba(15, 118, 110, 0.2) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] p {
        color: #e6fffb !important;
        font-weight: 800 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #0f766e;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛡️ AegisGraph Sentinel 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #94a3b8; font-weight: 500;">Real-Time Cross-Channel Mule Account Detection & Neutralization</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=100)
    st.title("Navigation")
    page = st.radio("Select Page", [
        "🧭 Command Center",
        "💳 Transaction Scan",
        "📁 Batch Triage",
        "📊 Risk Analytics",
        "🕸️ Network Graph Explorer",
        "⌨️ Behavioral Biometrics",
        "🧪 Innovation Lab",
        "ℹ️ System Brief"
    ])
    
    st.markdown("---")
    
    # Innovation sub-menu (conditional)
    if page == "🧪 Innovation Lab":
        innovation_page = st.radio("Innovation Module", [
            "🍯 Honeypot Escrow",
            "📞 Voice Stress Analysis",
            "🎯 Predictive Mule Scoring",
            "⌨️ Keystroke Stress Detection",
            "🧠 Aegis-Oracle Explainer",
            "⛓️ Blockchain Evidence"
        ])
    else:
        innovation_page = None
    
    st.markdown("---")
    
    # API Status Check
    try:
        health = _fetch_health_snapshot(API_URL)
        if health:
            st.success(_accessible_status("✅", "API Online"))
            st.metric("Uptime", f"{int(health.get('uptime_seconds', 0))}s")
            mode = "🎭 DEMO MODE" if not health.get('model_loaded', False) else "🚀 PRODUCTION"
            mode_label = "Demo Mode" if "DEMO" in mode else "Production Mode"
            st.info(f"{mode} ({mode_label})")
        else:
            st.error(_accessible_status("⚠️", "API Issue"))
    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(_accessible_status("❌", "API Offline"))
        st.warning("Start API: `python -m uvicorn src.api.main:app --reload`")

# Page: Dashboard

# Modularized Page Imports
from src.ui.pages.command_center import render_command_center
from src.ui.pages.transaction_scan import render_transaction_scan
from src.ui.pages.batch_triage import render_batch_triage
from src.ui.pages.risk_analytics import render_risk_analytics
from src.ui.pages.innovation_lab import render_innovation_lab
from src.ui.pages.network_graph import render_network_graph
from src.ui.pages.behavioral_biometrics import render_behavioral_biometrics
from src.ui.pages.system_brief import render_system_brief

# Helpers map
helpers = {
    '_fetch_health_snapshot': _fetch_health_snapshot,
    '_fetch_stats_snapshot': _fetch_stats_snapshot,
    '_schedule_live_refresh': _schedule_live_refresh,
    '_build_live_event': _build_live_event,
    '_accessible_status': _accessible_status,
    '_build_batch_transaction': _build_batch_transaction,
    '_estimate_csv_rows': _estimate_csv_rows,
    '_advance_timed_state': _advance_timed_state,
}
st_globals = {
    'API_URL': API_URL,
    'COMMAND_CENTER_IO_EXECUTOR': COMMAND_CENTER_IO_EXECUTOR,
    'BATCH_PREVIEW_ROWS': BATCH_PREVIEW_ROWS,
    'BATCH_CHUNK_SIZE': BATCH_CHUNK_SIZE,
    'BATCH_MAX_ROWS': BATCH_MAX_ROWS,
    'MAX_BATCH_UPLOAD_BYTES': MAX_BATCH_UPLOAD_BYTES,
}

# Page Routing
if page == "🧭 Command Center":
    render_command_center(helpers, st_globals)
elif page == "💳 Transaction Scan":
    render_transaction_scan(helpers, st_globals)
elif page == "📁 Batch Triage":
    render_batch_triage(helpers, st_globals)
elif page == "📊 Risk Analytics":
    render_risk_analytics(helpers, st_globals)
elif page == "🕸️ Network Graph Explorer":
    render_network_graph(helpers, st_globals)
elif page == "⌨️ Behavioral Biometrics":
    render_behavioral_biometrics(helpers, st_globals)
elif page == "🧪 Innovation Lab":
    render_innovation_lab(innovation_page, helpers, st_globals)
elif page == "ℹ️ System Brief":
    render_system_brief(helpers, st_globals)
