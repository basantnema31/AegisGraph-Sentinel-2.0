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


@with_error_boundary('📁 Batch Triage')
def render_batch_triage(helpers, st_globals):
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

    st.header("📁 Batch Transaction Processing")
    
    st.info("💡 Process multiple transactions at once for bulk fraud detection")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            if getattr(uploaded_file, "size", 0) > MAX_BATCH_UPLOAD_BYTES:
                st.error(
                    f"File too large. Maximum allowed size is {MAX_BATCH_UPLOAD_BYTES // (1024 * 1024)} MB."
                )
                st.stop()
    
            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file, nrows=BATCH_PREVIEW_ROWS)
            uploaded_file.seek(0)
            estimated_rows = _estimate_csv_rows(uploaded_file)
            uploaded_file.seek(0)
    
            st.success(
                f"Loaded CSV preview. Estimated rows: "
                f"{estimated_rows}{'+' if estimated_rows >= BATCH_MAX_ROWS else ''}"
            )
            
            st.subheader("Preview")
            st.dataframe(preview_df, use_container_width=True)
            
            if estimated_rows >= BATCH_MAX_ROWS:
                st.warning(
                    f"Processing will stop after {BATCH_MAX_ROWS} rows to keep memory usage bounded."
                )
            
            if st.button("Process All Transactions", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                processed_rows = 0
                total_rows = max(min(estimated_rows, BATCH_MAX_ROWS), 1)
                
                uploaded_file.seek(0)
                for chunk in pd.read_csv(uploaded_file, chunksize=BATCH_CHUNK_SIZE):
                    for _, row in chunk.iterrows():
                        if processed_rows >= BATCH_MAX_ROWS:
                            break
                        
                        status_text.text(f"Processing {processed_rows + 1}/{total_rows}...")
                        
                        txn = _build_batch_transaction(row, processed_rows)
                        
                        # Add optional fields if present in CSV
                        if 'ip_address' in row and pd.notna(row['ip_address']):
                            txn['ip_address'] = str(row['ip_address'])
                        if 'device_id' in row and pd.notna(row['device_id']):
                            txn['device_id'] = str(row['device_id'])
                        if 'location' in row and pd.notna(row['location']):
                            txn['location'] = str(row['location'])
                        
                        try:
                            response = requests.post(f"{API_URL}/api/v1/fraud/check", json=txn, timeout=30)
                            if response.status_code == 200:
                                result = response.json()
                                results.append({
                                    'Transaction ID': txn['transaction_id'],
                                    'Source': txn['source_account'],
                                    'Target': txn['target_account'],
                                    'Amount': f"Rs. {txn['amount']:,.0f}",
                                    'Risk Score': f"{result['risk_score']:.2%}",
                                    'risk_score_numeric': result['risk_score'],  # For charting
                                    'Decision': result['decision'],
                                    'Confidence': f"{result['confidence']:.0%}",
                                    'Graph Risk': f"{result['breakdown']['graph']:.2%}",
                                    'Velocity Risk': f"{result['breakdown']['velocity']:.2%}",
                                })
                            else:
                                st.error(f"API Error for {txn['transaction_id']}: Status {response.status_code}")
                                results.append({
                                    'Transaction ID': txn['transaction_id'],
                                    'Source': txn['source_account'],
                                    'Target': txn['target_account'],
                                    'Amount': f"Rs. {txn['amount']:,.0f}",
                                    'Risk Score': 'ERROR',
                                    'risk_score_numeric': 0,
                                    'Decision': 'ERROR',
                                    'Confidence': 'N/A',
                                    'Graph Risk': 'N/A',
                                    'Velocity Risk': 'N/A',
                                })
                        except Exception as e:
                            st.error(f"Error processing {txn.get('transaction_id', 'unknown')}: {str(e)}")
                            results.append({
                                'Transaction ID': txn.get('transaction_id', 'unknown'),
                                'Source': txn.get('source_account', 'unknown'),
                                'Target': txn.get('target_account', 'unknown'),
                                'Amount': f"Rs. {txn.get('amount', 0):,.0f}",
                                'Risk Score': 'ERROR',
                                'risk_score_numeric': 0,
                                'Decision': 'ERROR',
                                'Confidence': 'N/A',
                                'Graph Risk': 'N/A',
                                'Velocity Risk': 'N/A',
                            })
                        
                        processed_rows += 1
                        progress_bar.progress(min(processed_rows / total_rows, 1.0))
                    
                    if processed_rows >= BATCH_MAX_ROWS:
                        break
                
                status_text.text("Processing complete!")
                
                if not results:
                    st.warning("No transactions were processed from the uploaded CSV.")
                    st.stop()
                
                # Results
                st.markdown("---")
                st.subheader("📊 Results Summary")
                
                # Expected results info for sample data
                st.info("""
                **Understanding the Results:**
                - 🟢 **ALLOW**: Low risk (< 40%) - Normal transactions
                - 🟡 **REVIEW**: Medium risk (40-70%) - Suspicious patterns detected, needs analyst review
                - 🔴 **BLOCK**: High risk (≥ 70%) - Multiple fraud indicators, immediate blocking recommended
                
                **Sample data includes:**
                - Known mule accounts from real fraud chains (triggers high graph risk)
                - Late night transactions 2-4 AM (triggers entropy risk)
                - High amounts ≥ ₹100k (triggers velocity risk)
                - Mule-to-mule transfers (triggers multiple risk factors)
                """)
                
                results_df = pd.DataFrame(results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    blocked = len(results_df[results_df['Decision'] == 'BLOCK'])
                    st.metric("Blocked", blocked, delta=f"{blocked/len(results_df)*100:.1f}%")
                with col2:
                    review = len(results_df[results_df['Decision'] == 'REVIEW'])
                    st.metric("Review", review, delta=f"{review/len(results_df)*100:.1f}%")
                with col3:
                    allowed = len(results_df[results_df['Decision'] == 'ALLOW'])
                    st.metric("Allowed", allowed, delta=f"{allowed/len(results_df)*100:.1f}%")
                
                # Charts
                col_a, col_b = st.columns(2)
                
                with col_a:
                    fig_pie = px.pie(results_df, names='Decision', title='Decision Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_b:
                    fig_hist = px.histogram(results_df, x='risk_score_numeric', nbins=20, 
                                          title='Risk Score Distribution',
                                          labels={'risk_score_numeric': 'Risk Score'})
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Full Results Table (exclude numeric helper column)
                st.subheader("📋 Detailed Results")
                display_df = results_df.drop(columns=['risk_score_numeric'])
                
                # Highlight flagged transactions
                flagged_df = results_df[results_df['Decision'].isin(['REVIEW', 'BLOCK'])]
                if len(flagged_df) > 0:
                    st.warning(f"⚠️ {len(flagged_df)} transactions flagged for review or blocking")
                    with st.expander("🚨 View Flagged Transactions"):
                        flagged_display = flagged_df.drop(columns=['risk_score_numeric'])
                        st.dataframe(flagged_display, use_container_width=True)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download Results
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"fraud_check_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        st.markdown("### Sample CSV Format")
        st.info("💡 **Enhanced test data** with 12 transactions including ALLOW, REVIEW, and BLOCK scenarios")
        st.warning("🔴 **NEW**: Added 2 extreme-risk transactions that will trigger BLOCK decisions (₹250k-300k + mule accounts + late night)")
        
        # Create diverse test data with known mule accounts and various risk patterns
        sample_df = pd.DataFrame({
            'transaction_id': [
                'TXN_TEST_001',  # Normal transaction
                'TXN_TEST_002',  # Normal transaction
                'TXN_TEST_003',  # Normal transaction
                'TXN_TEST_004',  # Normal high amount
                'TXN_TEST_005',  # Known mule account (REVIEW)
                'TXN_TEST_006',  # Known mule account (REVIEW)
                'TXN_TEST_007',  # Mule to mule moderate (REVIEW)
                'TXN_TEST_008',  # Mule late night (REVIEW)
                'TXN_TEST_009',  # 🔴 EXTREME: Mule + 250k + 3AM (BLOCK)
                'TXN_TEST_010',  # 🔴 EXTREME: Mule→Mule + 300k + 2AM (BLOCK)
                'TXN_TEST_011',  # Normal small transaction
                'TXN_TEST_012',  # Normal transaction
            ],
            'source_account': [
                'ACC00000139',    # Normal account (verified non-mule)
                'ACC00000140',    # Normal account (verified non-mule)
                'ACC00000141',    # Normal account (verified non-mule)
                'ACC00000142',    # Normal account (verified non-mule)
                'ACC00001071',    # KNOWN MULE (from fraud chain)
                'ACC00003254',    # KNOWN MULE (from fraud chain)
                'ACC00001071',    # KNOWN MULE
                'ACC00000179',    # KNOWN MULE (from fraud chain)
                'ACC00004766',    # 🔴 EXTREME RISK MULE
                'ACC00001071',    # 🔴 EXTREME RISK MULE to MULE
                'ACC00000145',    # Normal account
                'ACC00000146',    # Normal account
            ],
            'target_account': [
                'MERCHANT_001',   # Merchant
                'ACC00000150',    # Normal P2P
                'MERCHANT_002',   # Merchant
                'MERCHANT_003',   # Merchant (high amount OK)
                'ACC00000150',    # Normal account (but source is mule)
                'ACC00000151',    # Normal account
                'ACC00003254',    # MULE to MULE transaction
                'ACC00000152',    # Normal account
                'ACC00000153',    # 🔴 Normal account (but HUGE amount + late night)
                'ACC00003254',    # 🔴 MULE to MULE + HUGE + LATE NIGHT
                'MERCHANT_004',   # Merchant
                'ACC00000154',    # Normal P2P
            ],
            'amount': [
                2500.00,      # Normal
                15000.00,     # Normal
                8500.00,      # Normal
                95000.00,     # High but legitimate merchant payment
                45000.00,     # Moderate (mule account)
                35000.00,     # Moderate (mule)
                85000.00,     # High (mule to mule)
                40000.00,     # Moderate (mule late night)
                250000.00,    # 🔴 EXTREME amount + mule
                300000.00,    # 🔴 EXTREME amount + mule to mule
                500.00,       # Small
                12000.00,     # Normal
            ],
            'currency': ['INR'] * 12,
            'mode': [
                'UPI',      # Fast payment
                'UPI',      # Fast payment
                'UPI',      # Fast payment
                'NEFT',     # Normal for high amount
                'UPI',      # Fast (suspicious for large with mule)
                'UPI',      # UPI
                'IMPS',     # Immediate (mule to mule)
                'UPI',      # UPI late night
                'IMPS',     # 🔴 IMPS for huge amount (instant transfer)
                'IMPS',     # 🔴 IMPS for huge mule transfer
                'UPI',      # Small UPI
                'UPI',      # UPI
            ],
            'timestamp': [
                '2026-02-26T14:30:00Z',  # Afternoon (normal)
                '2026-02-26T10:15:00Z',  # Morning (normal)
                '2026-02-26T16:00:00Z',  # Afternoon (normal)
                '2026-02-26T11:20:00Z',  # Morning (normal)
                '2026-02-26T18:45:00Z',  # Evening (mule but normal time)
                '2026-02-26T12:30:00Z',  # Afternoon (mule)
                '2026-02-26T22:00:00Z',  # Night (mule to mule)
                '2026-02-26T04:00:00Z',  # LATE NIGHT 4 AM (mule)
                '2026-02-26T03:15:00Z',  # 🔴 3:15 AM + EXTREME amount
                '2026-02-26T02:30:00Z',  # 🔴 2:30 AM + EXTREME mule transfer
                '2026-02-26T19:00:00Z',  # Evening
                '2026-02-26T13:45:00Z',  # Afternoon
            ],
            'ip_address': [
                '103.25.45.67',
                '103.25.45.68',
                '103.25.45.69',
                '103.25.45.70',
                '103.25.45.71',
                '103.25.45.72',
                '192.168.1.100',  # Private IP for mule-to-mule
                '103.25.45.73',
                '192.168.1.101',  # 🔴 Private IP + extreme
                '192.168.1.102',  # 🔴 Private IP + extreme
                '103.25.45.74',
                '103.25.45.75',
            ],
            'device_id': [
                'DEV_' + str(i).zfill(6) for i in range(1, 13)
            ],
            'location': [
                'Mumbai, India',
                'Delhi, India',
                'Bangalore, India',
                'Pune, India',
                'Mumbai, India',
                'Delhi, India',
                'Mumbai, India',
                'Kolkata, India',
                'Mumbai, India',      # 🔴 Same location pattern
                'Mumbai, India',      # 🔴 Same location pattern
                'Chennai, India',
                'Hyderabad, India',
            ]
        })
        
        st.dataframe(sample_df, use_container_width=True)
        
        # Add legend
        st.markdown("""
        **Test Data Legend (12 Transactions):**
        
        **🟢 ALLOW (5 transactions)** - Clean, legitimate transactions:
        - TXN_TEST_001-004, 011-012: Normal accounts, reasonable amounts, business hours
        
        **🟡 REVIEW (5 transactions)** - Suspicious patterns requiring analyst review:
        - TXN_TEST_005-006: Known mule accounts with moderate amounts
        - TXN_TEST_007: Mule-to-mule transfer at night
        - TXN_TEST_008: Mule account at 4 AM
        
        **🔴 BLOCK (2 transactions)** - Extreme risk, immediate blocking:
        - **TXN_TEST_009**: Mule account + ₹250k + 3:15 AM + Private IP = **EXTREME RISK**
        - **TXN_TEST_010**: Mule→Mule + ₹300k + 2:30 AM + IMPS = **CRITICAL FRAUD**
        
        **Risk Factors:**
        - 🚨 **Mule accounts**: ACC00001071, ACC00003254, ACC00000179, ACC00004766
        - 💰 **Extreme amounts**: ₹250k-300k trigger high velocity risk
        - 🌙 **Late night**: 2-4 AM adds entropy risk
        - 🔄 **Mule-to-mule**: Direct transfer between known fraud accounts
        """)
        
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )
    
    # Page: Statistics
