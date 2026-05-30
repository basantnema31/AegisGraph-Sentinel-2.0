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


@with_error_boundary('💳 Transaction Scan')
def render_transaction_scan(helpers, st_globals):
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

    st.header("💳 Single Transaction Fraud Check")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            txn_id = st.text_input("Transaction ID", value=f"TXN{int(time.time())}")
            source_account = st.text_input("Source Account", value="ACC_SOURCE_001")
            target_account = st.text_input("Target Account", value="ACC_TARGET_001")
            amount = st.number_input("Amount (₹)", min_value=0.01, value=10000.0, step=100.0)
            
        with col2:
            st.subheader("Additional Information")
            currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP"])
            mode = st.selectbox("Transaction Mode", ["UPI", "IMPS", "NEFT", "RTGS", "Card", "Wallet"])
            device_id = st.text_input("Device ID (Optional)", value="")
            location = st.text_input("Location (Optional)", value="")
        
        st.markdown("---")
        
        # Biometrics (Optional)
        with st.expander("🔑 Add Behavioral Biometrics (Optional)"):
            use_biometrics = st.checkbox("Include keystroke dynamics")
            if use_biometrics:
                st.info("Simulated biometrics will be added")
        
        submit = st.form_submit_button("🔎 Check Transaction", use_container_width=True)
        
        if submit:
            with st.spinner("🔄 Analyzing transaction..."):
                # Prepare request
                transaction = {
                    "transaction_id": txn_id,
                    "source_account": source_account,
                    "target_account": target_account,
                    "amount": float(amount),
                    "currency": currency,
                    "mode": mode,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
                }
                
                if device_id:
                    transaction["device_id"] = device_id
                if location:
                    transaction["location"] = location
                
                # Make API call
                try:
                    response = requests.post(f"{API_URL}/api/v1/fraud/check", json=transaction, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success(_accessible_status("✅", "Analysis Complete"))
                        
                        # Results Display
                        st.markdown("---")
                        st.subheader("📋 Analysis Results")
                        
                        # Top Metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            risk = result['risk_score']
                            st.metric("Risk Score", f"{risk:.3f}", delta=f"{(risk-0.5):.3f}")
                        with metric_cols[1]:
                            decision = result['decision']
                            emoji = "🟢" if decision == "ALLOW" else "🟡" if decision == "REVIEW" else "🔴"
                            st.metric("Decision", f"{emoji} {decision} ({decision.title()} decision)")
                        with metric_cols[2]:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with metric_cols[3]:
                            st.metric("Processing Time", f"{result['processing_time_ms']:.1f}ms")
                        
                        # Risk Breakdown
                        st.markdown("---")
                        st.subheader("📊 Risk Component Breakdown")
                        
                        breakdown = result['breakdown']
                        df = pd.DataFrame({
                            'Component': ['Graph Risk', 'Velocity Risk', 'Behavioral Risk', 'Entropy Risk'],
                            'Score': [breakdown['graph'], breakdown['velocity'], breakdown['behavior'], breakdown['entropy']]
                        })
                        
                        col_chart, col_table = st.columns([2, 1])
                        
                        with col_chart:
                            fig = px.bar(df, x='Component', y='Score', 
                                        title='Risk Factors',
                                        color='Score',
                                        color_continuous_scale='RdYlGn_r')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_table:
                            st.dataframe(df.style.background_gradient(cmap='RdYlGn_r', subset=['Score']), 
                                       use_container_width=True, height=400)
                        
                        # Explanation
                        st.markdown("---")
                        st.subheader("💡 Explanation")
                        
                        if decision == "BLOCK":
                            st.error(result['explanation'])
                        elif decision == "REVIEW":
                            st.warning(result['explanation'])
                        else:
                            st.success(result['explanation'])
                        
                        st.info(f"🚨 **Recommended Action:** {result['recommended_action']}")
                        
                        # JSON Response
                        with st.expander("🧾 View Full JSON Response"):
                            st.json(result)
                    
                    else:
                        st.error(f"Error: {response.status_code}")
                        st.json(response.json())
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure the API server is running: `python -m uvicorn src.api.main:app --reload`")
    
    # Page: Batch Processing
