import streamlit as st
import logging
import requests
import json
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time
import os
import random
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)
from src.ui.error_boundary import with_error_boundary


@with_error_boundary('🧭 Command Center')
def render_command_center(helpers, st_globals):
    # Unpack globals
    API_URL = st_globals.get('API_URL')
    COMMAND_CENTER_IO_EXECUTOR = st_globals.get('COMMAND_CENTER_IO_EXECUTOR')
    _fetch_health_snapshot = helpers.get('_fetch_health_snapshot')
    _fetch_stats_snapshot = helpers.get('_fetch_stats_snapshot')
    _schedule_live_refresh = helpers.get('_schedule_live_refresh')
    _build_live_event = helpers.get('_build_live_event')
    _accessible_status = helpers.get('_accessible_status')
    _build_batch_transaction = helpers.get('_build_batch_transaction')
    _estimate_csv_rows = helpers.get('_estimate_csv_rows')
    _advance_timed_state = helpers.get('_advance_timed_state')
    BATCH_PREVIEW_ROWS = st_globals.get('BATCH_PREVIEW_ROWS', 10)
    BATCH_CHUNK_SIZE = st_globals.get('BATCH_CHUNK_SIZE', 50)
    BATCH_MAX_ROWS = st_globals.get('BATCH_MAX_ROWS', 500)
    MAX_BATCH_UPLOAD_BYTES = st_globals.get('MAX_BATCH_UPLOAD_BYTES', 5 * 1024 * 1024)

    st.header("🧭 Real-Time Command Center")
    
    # Live Mode Toggle
    live_mode = st.toggle("🔴 Enable Live Event Stream", value=False, key="live_mode_toggle")
    
    if 'live_events' not in st.session_state:
        st.session_state.live_events = []
    if 'live_event_future' not in st.session_state:
        st.session_state.live_event_future = None
    if 'live_event_txn' not in st.session_state:
        st.session_state.live_event_txn = None
    
    try:
        health = _fetch_health_snapshot(API_URL)
    except Exception:
        health = {}
    try:
        stats = _fetch_stats_snapshot(API_URL)
    except Exception:
        stats = {}
        
    # Generate a live event if active
    if live_mode:
        _schedule_live_refresh()
        live_event_future = st.session_state.live_event_future
        if live_event_future is not None and live_event_future.done():
            event = live_event_future.result()
            if event is not None:
                st.session_state.live_events.insert(0, event)
                st.session_state.live_events = st.session_state.live_events[:15]
            st.session_state.live_event_future = None
            st.session_state.live_event_txn = None
    
        if st.session_state.live_event_future is None:
            accounts = ["ACC" + str(random.randint(1000, 9999)), "mule_acc_001", "ACC" + str(random.randint(1000, 9999))]
            txn = {
                "transaction_id": f"LIVE_{int(time.time()*1000)}",
                "source_account": random.choice(accounts),
                "target_account": random.choice(accounts),
                "amount": float(random.choice([500, 2500, 50000, 150000, 300000])),
                "currency": "INR",
                "mode": random.choice(["UPI", "IMPS"]),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
            }
            st.session_state.live_event_txn = txn
            st.session_state.live_event_future = COMMAND_CENTER_IO_EXECUTOR.submit(_build_live_event, API_URL, txn)
    
    # Metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    total_reqs = stats.get('total_requests', len(st.session_state.live_events))
    flagged = stats.get('decisions', {}).get('REVIEW', 0) + stats.get('decisions', {}).get('BLOCK', 0)
    flag_rate = (flagged / max(total_reqs, 1)) * 100
    
    with m_col1:
        st.metric("Total Checks", total_reqs, delta="Live")
    with m_col2:
        st.metric("Flagged", flagged, delta=f"{flag_rate:.1f}%")
    with m_col3:
        recent_lat = st.session_state.live_events[0]['latency'] if st.session_state.live_events else stats.get('avg_processing_time_ms', 0)
        st.metric("Avg Response", f"{recent_lat:.1f}ms", delta="Fast")
    with m_col4:
        uptime_hours = stats.get('uptime_seconds', 0) / 3600
        st.metric("Uptime", f"{uptime_hours:.1f}h", delta="Stable")
        
    st.markdown("---")
    
    # Realtime Visualizations
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("📡 Live Fraud Event Stream")
        if not st.session_state.live_events:
            st.info("Live stream inactive. Toggle above to begin simulating transactions.")
        else:
            # Lazy load pandas for dataframe operations if not globally imported
            # (pandas is kept global if used extensively, but for this PR we demonstrate lazy loading concepts)
            df_events = pd.DataFrame(st.session_state.live_events)
            display_df = df_events[['time', 'id', 'amount', 'decision', 'risk', 'latency']].copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.0f}")
            display_df['risk'] = display_df['risk'].apply(lambda x: f"{x:.1%}")
            display_df['latency'] = display_df['latency'].apply(lambda x: f"{x}ms")
            
            def highlight_decision(val):
                if val == 'BLOCK':
                    return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;'
                elif val == 'REVIEW':
                    return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: bold;'
                return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981;'
            
            # Note: Pandas Styler applymap was deprecated in Pandas 2.1.0, replaced by map
            st.dataframe(
                display_df.style.map(highlight_decision, subset=['decision']),
                use_container_width=True,
                height=400
            )
            
            st.subheader("📈 Realtime Risk Density (Heatmap)")
            if len(st.session_state.live_events) > 2:
                # Lazy load plotly
                import plotly.express as px
                fig = px.area(df_events[::-1], x='time', y='risk', color='decision',
                             color_discrete_map={'ALLOW': '#10b981', 'REVIEW': '#f59e0b', 'BLOCK': '#ef4444'},
                             title='Live Risk Timeline')
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
     
    with col_side:
        st.subheader("🧠 Aegis-Oracle Explainability")
        if st.session_state.live_events:
            latest = st.session_state.live_events[0]
            
            # Lazy load plotly graph_objects
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest['risk'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Risk Level", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': '#10b981'},
                        {'range': [40, 70], 'color': '#f59e0b'},
                        {'range': [70, 100], 'color': '#ef4444'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("**Decision Breakdown**")
            bd = latest.get('breakdown', {})
            st.progress(float(min(bd.get('graph', 0), 1.0)), text=f"Graph Anomaly ({bd.get('graph',0):.1%})")
            st.progress(float(min(bd.get('velocity', 0), 1.0)), text=f"Velocity Risk ({bd.get('velocity',0):.1%})")
            st.progress(float(min(bd.get('behavior', 0), 1.0)), text=f"Behavioral Stress ({bd.get('behavior',0):.1%})")
            st.progress(float(min(bd.get('entropy', 0), 1.0)), text=f"Entropy Risk ({bd.get('entropy',0):.1%})")
            
            if latest['decision'] == 'BLOCK':
                st.error(latest['explanation'])
            elif latest['decision'] == 'REVIEW':
                st.warning(latest['explanation'])
            else:
                st.success(latest['explanation'])
        else:
            st.write("Awaiting transactions...")
    
    # Page: Single Transaction Check
