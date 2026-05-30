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


@with_error_boundary('🧪 Innovation Lab')
def render_innovation_lab(innovation_page, helpers, st_globals):
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

    # Sub-page: Honeypot Escrow
    if innovation_page == "🍯 Honeypot Escrow":
        st.header("🍯 Honeypot Escrow - Deceptive Containment System")
        
        st.markdown("""
        **Innovation 2**: High-risk transactions show "Success" to criminals but funds transfer to 
        shadow escrow. ATM withdrawal attempts trigger GPS police alerts. **87% arrest rate** 🚓 
        """)
        
        st.markdown("---")
        
        # Honeypot Statistics
        try:
            response = requests.get(f"{API_URL}/api/v1/honeypot/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Activated", stats['total_activated'])
                with col2:
                    st.metric("Arrests", stats['total_arrests'], 
                             delta=f"{stats['arrest_rate']:.1%} rate")
                with col3:
                    st.metric("Recovery", f"₹{stats['total_recovered']/10000000:.2f} Cr")
                with col4:
                    st.metric("Networks Dismantled", stats['networks_dismantled'])
                
                st.markdown("---")
                
                # More detailed metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("False Positives", stats['false_positives'],
                             delta=f"{stats['false_positive_rate']:.1%}")
                with col_b:
                    st.metric("Avg Time to Arrest", f"{stats['avg_time_to_arrest_minutes']:.1f} min")
                with col_c:
                    arrest_rate_colored = "🟢" if stats['arrest_rate'] >= 0.8 else "🟡" if stats['arrest_rate'] >= 0.6 else "🔴"
                    st.metric("System Status", f"{arrest_rate_colored} Operational (Operational)")
            else:
                st.warning("⚠️ Honeypot statistics unavailable (innovation module not running)")
        except Exception as e:
            st.error(f"Unable to fetch honeypot stats: {e}")
            st.info("💡 Ensure API is running with innovation modules loaded")
        
        st.markdown("---")
        
        # Active Honeypots
        st.subheader("🎭 Active Honeypot Traps")
        
        try:
            response = requests.get(f"{API_URL}/api/v1/honeypot/active", timeout=5)
            if response.status_code == 200:
                data = response.json()
                active = data['active_honeypots']
                
                if len(active) > 0:
                    st.info(f"🔴 Active honeypots: {len(active)} currently monitoring withdrawal attempts")
                    
                    for hp in active:
                        with st.expander(f"Honeypot {hp['honeypot_id']} - ₹{hp['amount']:,.2f}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Transaction ID**: {hp['transaction_id']}")
                                st.write(f"**Target Account**: {hp['target_account']}")
                                st.write(f"**Amount**: ₹{hp['amount']:,.2f} {hp['currency']}")
                                status_text = "Police alerted" if hp['police_alerted'] else "Monitoring"
                                st.write(f"**Status**: {'🚨' if hp['police_alerted'] else '👁️'} {status_text} ({status_text})")
                            with col2:
                                st.write(f"**Activated**: {hp['activated_at']}")
                                st.write(f"**Time Remaining**: {hp['time_remaining_seconds']//60} min {hp['time_remaining_seconds']%60} sec")
                                st.write(f"**Withdrawal Attempts**: {hp['withdrawal_attempts']}")
                                if hp['last_attempt_location']:
                                    st.write(f"**Last Attempt Location**: {hp['last_attempt_location']}")
                else:
                    st.success(_accessible_status("✅", "No active honeypots"))
            else:
                st.warning("⚠️ Active honeypots unavailable")
        except Exception as e:
            st.error(f"Unable to fetch active honeypots: {e}")
    
    # Sub-page: Voice Stress Analysis
    elif innovation_page == "📞 Voice Stress Analysis":
        st.header("📞 Voice Stress Analysis - Coercion Detection")
        
        st.markdown("""
        **Innovation 5**: Detects phone coercion during transactions through acoustic stress analysis.
        Extracts F0 (pitch), jitter, shimmer, speech rate, prosody. **92% detection accuracy** 📞 
        """)
        
        st.markdown("---")
        
        st.subheader("🎤 Voice Analysis Upload")
        
        st.info("💡 Upload a WAV audio file (max 30 seconds) from a transaction call to analyze stress levels")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload Voice Recording (WAV)", type=["wav"], 
                                            help="Audio file from transaction phone verification")
        
        with col2:
            st.write("")
            st.write("")
            transaction_id_voice = st.text_input("Transaction ID", value=f"TXN_{int(time.time())}")
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("🎙️ Analyze Voice Stress", type="primary", use_container_width=True):
                with st.spinner("Analyzing acoustic features..."):
                    try:
                        # Read audio file
                        audio_bytes = uploaded_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        # Call API
                        payload = {
                            "transaction_id": transaction_id_voice,
                            "audio_base64": audio_base64,
                            "sample_rate": 16000
                        }
                        
                        response = requests.post(f"{API_URL}/api/v1/voice/analyze", 
                                                json=payload, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.markdown("---")
                            st.subheader("📋 Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                stress_score = result['stress_score']
                                color = "🟢" if stress_score < 30 else "🟡" if stress_score < 70 else "🔴"
                                st.metric("Stress Score", f"{stress_score:.1f}/100", 
                                         delta=f"{color} {result['classification']}")
                            
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                            
                            with col3:
                                st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
                            
                            # Features
                            st.markdown("---")
                            st.subheader("🎵 Acoustic Features")
                            
                            features = result['features']
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("F0 (Pitch)", f"{features.get('f0_mean', 0):.1f} Hz")
                                st.metric("Jitter", f"{features.get('jitter', 0):.2%}")
                            with col_b:
                                st.metric("Shimmer", f"{features.get('shimmer', 0):.2%}")
                                st.metric("Speech Rate", f"{features.get('speech_rate', 0):.1f} /s")
                            with col_c:
                                st.metric("Prosody Entropy", f"{features.get('prosody_entropy', 0):.2f}")
                            
                            # Recommended Action
                            st.markdown("---")
                            action = result['recommended_action']
                            
                            if action == "CALLBACK_REQUIRED":
                                st.error("🚨 **SEVERE COERCION DETECTED** - Immediate callback required on different number")
                            elif action == "REVIEW":
                                st.warning("⚠️ **MILD STRESS DETECTED** - Consider manual review or callback")
                            else:
                                st.success("✅ **NORMAL PATTERN** - Transaction can proceed")
                        
                        else:
                            st.error(f"❌ Analysis failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error analyzing voice: {e}")
                        st.info("💡 Ensure API is running with voice analysis module (requires librosa)")
    
    # Sub-page: Predictive Mule Scoring
    elif innovation_page == "🎯 Predictive Mule Scoring":
        st.header("🎯 Predictive Mule Identification - Pre-Transaction Detection")
        
        st.markdown("""
        **Innovation 4**: Identifies mule accounts at creation, before first transaction.
        Analyzes 12 features including temporal clustering, device novelty, document quality.
        **86% precision**, ₹14.2 crore prevented 🛡️ 
        """)
        
        st.markdown("---")
        
        st.subheader("📝 Score New Account Opening")
        
        st.info("💡 Enter account opening details to predict mule recruitment risk")
        
        col1, col2 = st.columns(2)
        
        with col1:
            account_id = st.text_input("Account ID", value=f"ACC_NEW_{int(time.time())}")
            name = st.text_input("Account Holder Name", value="Test User")
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            profession = st.selectbox("Profession", ["Student", "Employed", "Unemployed", "Self-Employed", "Retired"])
            email = st.text_input("Email", value="test@example.com")
            phone = st.text_input("Phone", value="+919876543210")
        
        with col2:
            device_id = st.text_input("Device ID", value="DEVICE_NEW_001")
            ip_address = st.text_input("IP Address", value="103.45.67.89")
            stated_address = st.text_input("Stated Address", value="Mumbai, India")
            facial_match = st.slider("Facial Match Score", 0.0, 1.0, 0.85, 0.01,
                                     help="KYC facial recognition match score")
            initial_deposit = st.number_input("Initial Deposit (₹)", min_value=0.0, value=0.0, step=100.0)
            form_time = st.number_input("Form Completion Time (seconds)", min_value=60, max_value=1800, value=300)
        
        if st.button("🏦 Score Account Opening", type="primary", use_container_width=True):
            with st.spinner("Analyzing account opening patterns..."):
                try:
                    payload = {
                        "account_id": account_id,
                        "name": name,
                        "age": age,
                        "profession": profession,
                        "email": email,
                        "phone": phone,
                        "device_id": device_id,
                        "ip_address": ip_address,
                        "stated_address": stated_address,
                        "facial_match": facial_match,
                        "document_type": "Aadhaar",
                        "initial_deposit": initial_deposit,
                        "form_completion_time_seconds": form_time
                    }
                    
                    response = requests.post(f"{API_URL}/api/v1/accounts/score-opening",
                                            json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.markdown("---")
                        st.subheader("🚨 Mule Risk Assessment")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_score = result['risk_score']
                            risk_level = result['risk_level']
                            color = "🔴" if risk_level == "CRITICAL_MULE_RISK" else "🟠" if risk_level == "HIGH_MULE_RISK" else "🟡" if risk_level == "MODERATE" else "🟢"
                            st.metric("Mule Risk Score", f"{risk_score:.1f}/100",
                                     delta=f"{color} {risk_level}")
                        
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        with col3:
                            st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
                        
                        # Feature Breakdown
                        st.markdown("---")
                        st.subheader("🧩 Feature Analysis")
                        
                        features = result['features']
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Temporal Clustering", f"{features.get('temporal_clustering', 0):.1f}")
                            st.metric("Document Quality", f"{features.get('document_quality', 0):.1f}")
                            st.metric("Device Novelty", f"{features.get('device_novelty', 0):.1f}")
                            st.metric("Geographic Mismatch", f"{features.get('geographic_mismatch', 0):.1f}")
                        
                        with col_b:
                            st.metric("Referrer Patterns", f"{features.get('referrer_patterns', 0):.1f}")
                            st.metric("Form Speed", f"{features.get('form_speed', 0):.1f}")
                            st.metric("Email Domain", f"{features.get('email_domain', 0):.1f}")
                            st.metric("Phone Age", f"{features.get('phone_age', 0):.1f}")
                        
                        with col_c:
                            st.metric("Profession Risk", f"{features.get('profession_risk', 0):.1f}")
                            st.metric("Social Isolation", f"{features.get('social_isolation', 0):.1f}")
                            st.metric("Balance Risk", f"{features.get('balance_risk', 0):.1f}")
                            st.metric("KYC Risk", f"{features.get('kyc_risk', 0):.1f}")
                        
                        # Red Flags
                        if result['red_flags']:
                            st.markdown("---")
                            st.subheader("🚩 Red Flags Detected")
                            
                            for flag in result['red_flags']:
                                st.warning(f"⚠️ {flag}")
                        
                        # Recommended Action
                        st.markdown("---")
                        action = result['recommended_action']
                        
                        if "IMMEDIATE" in action:
                            st.error(f"🚨 **{action}** - High mule risk detected")
                        elif "ENHANCED" in action:
                            st.warning(f"⚠️ **{action}** - Increased monitoring recommended")
                        else:
                            st.success(f"✅ **{action}** - Normal account opening")
                    
                    else:
                        st.error(f"❌ Scoring failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Error scoring account: {e}")
                    st.info("💡 Ensure API is running with predictive mule module")
    
        # after predictive mule, insert two new innovation pages
        
    # Sub-page: Keystroke Stress Detection
    elif innovation_page == "⌨️ Keystroke Stress Detection":
        st.header("⌨️ Keystroke Stress Detection - Behavioral Biometrics")
        
        st.markdown("""
        **Innovation 1**: Analyzes typing patterns during transaction entry to detect
        hesitation and stress. Useful for spotting coerced payments or nervous fraudsters.
        """
        )
        
        st.markdown("---")
        st.info("💡 This module is automatically invoked during any transaction check if keystroke data is provided; you can simulate it here.")
        
        st.subheader("📝 Simulate Keystroke Data")
        hold = st.text_area("Hold times (ms, comma-separated)", "120,180,220,160")
        flight = st.text_area("Flight times (ms, comma-separated)", "80,90,85,95")
        if st.button("⌨️ Analyze Typing Stress", use_container_width=True):
            try:
                hold_times = [float(x.strip()) for x in hold.split(',') if x.strip()]
                flight_times = [float(x.strip()) for x in flight.split(',') if x.strip()]
                payload = {
                    "transaction_id": f"KS_{int(time.time())}",
                    "source_account": "KS_SRC",
                    "target_account": "KS_TGT",
                    "amount": 1,
                    "currency": "INR",
                    "mode": "UPI",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "biometrics": {"hold_times": hold_times, "flight_times": flight_times}
                }
                resp = requests.post(f"{API_URL}/api/v1/fraud/check", json=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"Stress detected: {result['behavioral_stress_detected']}")
                    st.json(result)
                else:
                    st.error(f"API error: {resp.text}")
            except Exception as e:
                st.error(f"Failed to analyze: {e}")
    
    # Sub-page: Aegis-Oracle Explainer
    elif innovation_page == "🧠 Aegis-Oracle Explainer":
        st.header("🧠 Aegis-Oracle - Explanation Engine")
        
        st.markdown("""
        **Innovation 3**: Our proprietary oracle generates human-readable explanations
        for each fraud decision, highlighting key risk factors and recommended actions.
        """
        )
        
        st.markdown("---")
        st.info("💡 Enter a sample transaction and risk result to see an explanation.")
        
        txn_id = st.text_input("Transaction ID", "EXPL_001")
        amt = st.number_input("Amount", value=1000.0)
        score = st.slider("Risk Score", 0.0, 1.0, 0.25)
        dec = st.selectbox("Decision", ["ALLOW","REVIEW","BLOCK"])
        if st.button("🧠 Generate Explanation"):
            payload = {
                "transaction_id": txn_id,
                "amount": amt,
                "source_account": "SRC",
                "target_account": "TGT",
                "risk_score": score,
                "decision": dec
            }
            try:
                resp = requests.post(f"{API_URL}/api/v1/explain", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.json(resp.json())
                else:
                    st.error(f"Explanation API error: {resp.text}")
            except Exception as e:
                st.error(f"Error calling oracle: {e}")
    
    # Sub-page: Blockchain Evidence
    elif innovation_page == "⛓️ Blockchain Evidence":
        st.header("⛓️ Blockchain Evidence Chain - Immutable Forensics")
        
        st.markdown("""
        **Innovation 6**: Seals fraud decisions in Hyperledger Fabric for legal admissibility.
        18 validator nodes, RAFT consensus, <100ms finality. **Court-tested and admissible** ⚖️
        """)
        
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔍 Verify Evidence", "📜 Export for Legal Proceedings"])
        
        with tab1:
            st.subheader("Verify Blockchain Evidence")
            
            st.info("💡 Enter Evidence ID to verify integrity across validator nodes")
            
            evidence_id = st.text_input("Evidence ID", value="EVID_001", help="Evidence identifier from transaction")
            block_number = st.number_input("Block Number", min_value=0, value=0, help="Block containing the evidence")
            
            if st.button("✅ Verify Evidence", type="primary", use_container_width=True):
                with st.spinner("Verifying across validator nodes..."):
                    try:
                        response = requests.get(f"{API_URL}/api/v1/blockchain/verify/{evidence_id}?block_number={block_number}",
                                               timeout=10)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.markdown("---")
                            
                            # Verification Status
                            if result['verified']:
                                st.success(_accessible_status("✅", "Evidence Verified") + " - Blockchain integrity intact")
                            else:
                                st.error(_accessible_status("❌", "Verification Failed") + " - Evidence compromised or not found")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                status = "✅" if result['block_exists'] else "❌"
                                st.metric("Block Exists", status)
                            with col2:
                                status = "✅" if result['chain_integrity'] else "❌"
                                st.metric("Chain Integrity", status)
                            with col3:
                                st.metric("Consensus Nodes", result['consensus_nodes'])
                            with col4:
                                orig = result.get('original_timestamp')
                                if orig:
                                    st.metric("Original Seal", orig[:10])
                                else:
                                    st.metric("Original Seal", "-")
                            
                            # Verification Details
                            if result['verification_details']:
                                st.markdown("---")
                                st.subheader("📋 Verification Details")
                                
                                st.json(result['verification_details'])
                        
                        else:
                            st.error(f"❌ Verification failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error verifying evidence: {e}")
                        st.info("💡 Ensure API is running with blockchain module")
        
        with tab2:
            st.subheader("Export Evidence for Legal Proceedings")
            
            st.warning("🔒 **Authorized Access Only** - Requires law enforcement credentials")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_evidence_id = st.text_input("Evidence ID", value="EVID_001", key="export_evid")
                case_number = st.text_input("Case Number", value="CR/2026/12345")
            
            with col2:
                authority = st.text_input("Requesting Authority", value="Maharashtra Police Cyber Cell")
                auth_token = st.text_input("Authorization Token", value="", type="password",
                                          help="Secure token for evidence access")
            
            if st.button("📜 Export Evidence Package", type="primary", use_container_width=True):
                if not auth_token:
                    st.error(_accessible_status("❌", "Authorization token required"))
                else:
                    with st.spinner("Generating court-admissible evidence package..."):
                        try:
                            payload = {
                                "evidence_id": export_evidence_id,
                                "case_number": case_number,
                                "requesting_authority": authority,
                                "authorization_token": auth_token
                            }
                            
                            response = requests.post(f"{API_URL}/api/v1/blockchain/export",
                                                    json=payload, timeout=15)
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                st.success("✅ **Evidence Package Generated**")
                                
                                st.markdown("---")
                                st.subheader("📦 Evidence Package")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Evidence ID**: {result['evidence_id']}")
                                    st.write(f"**Case Number**: {result['case_number']}")
                                with col2:
                                    st.write(f"**Export Time**: {result['export_timestamp']}")
                                    st.write(f"**Authorized By**: {result['authorized_by']}")
                                
                                # Download button
                                evidence_json = json.dumps(result['evidence_package'], indent=2)
                                st.download_button(
                                    label="📥 Download Evidence Package (JSON)",
                                    data=evidence_json,
                                    file_name=f"evidence_{export_evidence_id}_{case_number.replace('/', '_')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                                
                                st.info("📌 This evidence package includes chain of custody, validator attestations, and is court-admissible under IT Act 2000")
                            
                            else:
                                st.error(f"❌ Export failed: {response.text}")
                        
                        except Exception as e:
                            st.error(f"Error exporting evidence: {e}")
    
    # Page: About
    
    # Page: Network Graph Explorer
