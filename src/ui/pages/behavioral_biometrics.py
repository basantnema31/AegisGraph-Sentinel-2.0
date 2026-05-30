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


@with_error_boundary('⌨️ Behavioral Biometrics')
def render_behavioral_biometrics(helpers, st_globals):
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

    st.header("⌨️ Keystroke Dynamics & Biometric Analysis")
    st.markdown("Visualizing typing cadence anomalies to detect hesitation, duress, or bot-driven inputs.")
    
    # Local imports consolidated globally
    
    # Session state for demo selection
    demo_type = st.radio("Select Session Profile", ["🟢 Normal User", "🔴 Under Duress (Hesitation Spikes)", "🤖 Automated Bot (Zero Variance)"], horizontal=True)
    
    st.markdown("---")
    
    # Generate mock data
    num_keys = 50
    x_axis = list(range(1, num_keys + 1))
    
    if "Normal" in demo_type:
        hold_times = np.random.normal(loc=120, scale=20, size=num_keys).clip(60, 200)
        flight_times = np.random.normal(loc=180, scale=30, size=num_keys).clip(100, 300)
        risk_score = 0.15
        explanation = "Typing rhythm is consistent with historical baseline. Variance is natural."
    elif "Duress" in demo_type:
        hold_times = np.random.normal(loc=140, scale=30, size=num_keys).clip(60, 250)
        flight_times = np.random.normal(loc=200, scale=40, size=num_keys).clip(100, 400)
        # Inject hesitation spikes
        spike_indices = [12, 13, 14, 35, 36]
        for idx in spike_indices:
            flight_times[idx] = random.randint(800, 1500)
            hold_times[idx] = random.randint(300, 500)
        risk_score = 0.88
        explanation = "🚨 HIGH DURESS DETECTED: Significant hesitation spikes observed. User may be receiving dictated instructions or under coercion."
    else: # Bot
        hold_times = np.random.normal(loc=50, scale=1, size=num_keys)
        flight_times = np.random.normal(loc=80, scale=1, size=num_keys)
        risk_score = 0.99
        explanation = "🚨 BOT DETECTED: Unnatural zero-variance mechanical cadence. Typing speed exceeds human limits."
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create dual-axis plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Flight Times (Line + Markers)
        fig.add_trace(
            go.Scatter(x=x_axis, y=flight_times, name="Flight Time (ms)",
                       mode='lines+markers', line=dict(color='#00d2ff', width=2),
                       marker=dict(size=6, color='#00d2ff')),
            secondary_y=False,
        )
        
        # Hold Times (Bar)
        fig.add_trace(
            go.Bar(x=x_axis, y=hold_times, name="Hold Time (ms)", opacity=0.7, marker_color='#ff007f'),
            secondary_y=True,
        )
        
        # Add highlight regions for spikes
        if "Duress" in demo_type:
            fig.add_vrect(x0=11.5, x1=15.5, fillcolor="red", opacity=0.1, layer="below", line_width=0)
            fig.add_vrect(x0=34.5, x1=37.5, fillcolor="red", opacity=0.1, layer="below", line_width=0)
            fig.add_annotation(x=13.5, y=1400, text="Hesitation Anomaly", showarrow=True, arrowhead=1)
    
        fig.update_layout(
            title="Session Keystroke Cadence (Hold vs. Flight Times)",
            xaxis_title="Keystroke Sequence Index",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Flight Time (ms) - Time between keys", secondary_y=False, showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(title_text="Hold Time (ms) - Key press duration", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Analysis")
        
        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': "Biometric Risk"},
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
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.metric("Avg Hold Time", f"{np.mean(hold_times):.1f} ms")
        st.metric("Avg Flight Time", f"{np.mean(flight_times):.1f} ms")
        st.metric("Cadence Variance", f"{np.var(flight_times):.1f}")
        
        if risk_score > 0.7:
            st.error(explanation)
        else:
            st.success(explanation)
