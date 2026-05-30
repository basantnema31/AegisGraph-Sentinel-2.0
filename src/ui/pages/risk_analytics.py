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


@with_error_boundary('📊 Risk Analytics')
def render_risk_analytics(helpers, st_globals):
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

    st.header("📊 Enterprise Operations & Risk Analytics")
    st.markdown("Centralized SOC command console for real-time infrastructure and decision monitoring.")
    st.markdown("---")
    
    # Local imports consolidated globally
    
    # Check API Status
    stats = _fetch_stats_snapshot(API_URL)
    health = _fetch_health_snapshot(API_URL)
    
    # Extract metrics
    total_requests = stats.get('total_requests', 0)
    decisions = stats.get('decisions', {})
    flagged = decisions.get('REVIEW', 0) + decisions.get('BLOCK', 0)
    avg_time = stats.get('avg_processing_time_ms', 45.0)
    uptime_sec = health.get('uptime_seconds', 0)
    
    # Interactive Live Toggle
    is_live = st.toggle("🔴 Enable Live Infrastructure Monitoring (Auto-Refresh)", value=False, key="risk_analytics_live_toggle")
    
    # Initialize session state for live tracking
    if 'risk_latency_history' not in st.session_state:
        st.session_state.risk_latency_history = list(np.random.normal(loc=avg_time, scale=5.0, size=20).clip(10, 200))
    if 'risk_tps_history' not in st.session_state:
        st.session_state.risk_tps_history = list(np.random.normal(loc=12.5, scale=2.5, size=20).clip(1, 100))
    if 'risk_time_history' not in st.session_state:
        st.session_state.risk_time_history = list(pd.date_range(end=pd.Timestamp.now(), periods=20, freq='2S'))
    if 'fraud_risk_score_history' not in st.session_state:
        st.session_state.fraud_risk_score_history = list(np.random.normal(loc=30, scale=5.0, size=20).clip(0, 100))
        
    if is_live:
        # Generate new data point
        new_latency = np.random.normal(loc=avg_time, scale=6.0)
        new_tps = np.random.normal(loc=15.0 if flagged > 0 else 10.0, scale=3.0)
        
        # Calculate new risk score based on real-time alerts
        new_risk = np.random.normal(loc=30, scale=5.0)
        if 'realtime_alerts' in st.session_state and len(st.session_state.realtime_alerts) > 0:
            active_alerts_for_risk = [a for a in st.session_state.realtime_alerts if a.get('status', 'Active') != 'Resolved']
            if active_alerts_for_risk:
                latest_alert = active_alerts_for_risk[0]
                # Link to alert severity if fresh (within 3 seconds)
                time_diff = (pd.Timestamp.now() - latest_alert['time']).total_seconds()
                if time_diff < 3:
                    if latest_alert['severity'] == "Critical":
                        new_risk = np.random.normal(loc=95, scale=2.0)
                    elif latest_alert['severity'] == "High":
                        new_risk = np.random.normal(loc=82, scale=4.0)
                    elif latest_alert['severity'] == "Medium":
                        new_risk = np.random.normal(loc=65, scale=6.0)
        new_risk = max(0, min(100, new_risk))
        
        st.session_state.risk_latency_history.append(new_latency)
        st.session_state.risk_tps_history.append(new_tps)
        st.session_state.risk_time_history.append(pd.Timestamp.now())
        st.session_state.fraud_risk_score_history.append(new_risk)
        
        # Keep only last 20 using slice reassignment (faster and safer in Streamlit)
        if len(st.session_state.risk_latency_history) > 20:
            st.session_state.risk_latency_history = st.session_state.risk_latency_history[-20:]
            st.session_state.risk_tps_history = st.session_state.risk_tps_history[-20:]
            st.session_state.risk_time_history = st.session_state.risk_time_history[-20:]
            st.session_state.fraud_risk_score_history = st.session_state.fraud_risk_score_history[-20:]
    
    # Top metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests Checked", f"{total_requests:,}")
    with col2:
        fraud_rate = (flagged / max(total_requests, 1)) * 100
        st.metric("Fraud Detection Rate", f"{fraud_rate:.2f}%", delta="Normal" if fraud_rate < 15.0 else "Elevated Risk", delta_color="inverse")
    with col3:
        current_latency = st.session_state.risk_latency_history[-1]
        st.metric("Avg Latency (ms)", f"{current_latency:.2f} ms", delta=f"{current_latency - st.session_state.risk_latency_history[-2]:.2f} ms", delta_color="inverse")
    with col4:
        st.metric("SOC Node Status", "🟢 ACTIVE" if health.get('model_loaded', False) else "🎭 DEMO MODE")
    
    st.markdown("---")
    
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("⚡ Live System Throughput & Latency")
        
        # Create dual axis graph
        fig_lat = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lat.add_trace(
            go.Scatter(
                x=st.session_state.risk_time_history, 
                y=st.session_state.risk_latency_history, 
                name="Inference Latency (ms)", 
                mode='lines+markers', 
                line=dict(color='#ff9900', width=3),
                marker=dict(size=6)
            ),
            secondary_y=False,
        )
        fig_lat.add_trace(
            go.Scatter(
                x=st.session_state.risk_time_history, 
                y=st.session_state.risk_tps_history, 
                name="System TPS (Throughput)", 
                mode='lines+markers', 
                line=dict(color='#00ffcc', width=2, dash='dot'),
                marker=dict(size=4)
            ),
            secondary_y=True,
        )
        
        fig_lat.update_layout(
            height=280, 
            margin=dict(l=10, r=10, t=30, b=10), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_lat.update_yaxes(title_text="Latency (ms)", secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
        fig_lat.update_yaxes(title_text="TPS (Req/Sec)", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig_lat, use_container_width=True)
        
        def render_risk_and_threat_charts():
            risk_col, threat_col = st.columns([2, 1])
            
            with risk_col:
                st.subheader("🌐 Global Fraud Risk Index")
                st.markdown("**Real-time aggregate risk score driven by live alert severities**")
                
                # Build Risk Trend Chart
                risk_df = pd.DataFrame({
                    'Time': st.session_state.risk_time_history,
                    'RiskScore': st.session_state.fraud_risk_score_history
                })
                
                # Color line based on latest risk score
                current_risk = risk_df['RiskScore'].iloc[-1]
                if current_risk >= 80:
                    line_color = '#ef4444'
                    fill_color = 'rgba(239, 68, 68, 0.2)'
                elif current_risk >= 50:
                    line_color = '#f59e0b'
                    fill_color = 'rgba(245, 158, 11, 0.2)'
                else:
                    line_color = '#10b981'
                    fill_color = 'rgba(16, 185, 129, 0.2)'
                    
                fig_risk = go.Figure()
                
                # Gradient area fill
                fig_risk.add_trace(go.Scatter(
                    x=risk_df['Time'], 
                    y=risk_df['RiskScore'],
                    fill='tozeroy',
                    mode='lines',
                    line=dict(color=line_color, width=3),
                    fillcolor=fill_color,
                    name="Risk Index"
                ))
                
                # Add spike markers for high risk
                high_risk_points = risk_df[risk_df['RiskScore'] >= 80]
                if not high_risk_points.empty:
                    fig_risk.add_trace(go.Scatter(
                        x=high_risk_points['Time'],
                        y=high_risk_points['RiskScore'],
                        mode='markers',
                        marker=dict(color='#ef4444', size=10, symbol='diamond-open', line=dict(width=2, color='#ef4444')),
                        name="Critical Anomaly"
                    ))
                    
                fig_risk.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    showlegend=False,
                    yaxis=dict(range=[0, 100], title="Risk Score", gridcolor='rgba(255,255,255,0.1)'),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
            with threat_col:
                st.subheader("🍩 Threat Distribution")
                st.markdown("**Live severity breakdown**")
                
                if 'realtime_alerts' in st.session_state and len(st.session_state.realtime_alerts) > 0:
                    alerts = st.session_state.realtime_alerts
                    
                    if not alerts:
                        st.info("No active threats.")
                    else:
                        sev_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
                        for a in alerts:
                            sev_counts[a['severity']] += 1
                        
                        labels = list(sev_counts.keys())
                        values = list(sev_counts.values())
                        colors = ['#ef4444', '#f97316', '#f59e0b', '#10b981'] # Critical, High, Medium, Low
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=labels, 
                            values=values, 
                            hole=.6,
                            marker=dict(colors=colors, line=dict(color='#0b0f19', width=2)),
                            textinfo='none',
                            hoverinfo='label+percent'
                        )])
                        fig_pie.update_layout(
                            height=200,
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False,
                            annotations=[dict(text=f"{len(alerts)}", x=0.5, y=0.5, font_size=24, showarrow=False, font_color="white")]
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No active threats.")
                    
        # Call the modularized function
        render_risk_and_threat_charts()
        
        st.subheader("🧠 AI Decision Breakdown Engine")
        
        # Stacked Risk Bar selector for different high-profile alerts
        st.markdown("**Analyze Risk Distribution of Flagged Accounts**")
        selected_alert = st.selectbox(
            "Select Account to Inspect Anomaly Composition",
            ["ACC00004766 (High Risk - Money Laundering)", "ACC00001071 (Star Hub - Mule Network)", "ACC00003254 (Moderate Risk - Multi-hop Transfer)"]
        )
        
        if "4766" in selected_alert:
            categories = ['Graph Anomaly', 'Velocity Risk', 'Behavioral Stress', 'Entropy']
            values = [0.45, 0.35, 0.12, 0.08]
            desc = "Critical Graph and Velocity signals detected. Transaction occurred at 3:15 AM (Entropy)."
        elif "1071" in selected_alert:
            categories = ['Graph Anomaly', 'Velocity Risk', 'Behavioral Stress', 'Entropy']
            values = [0.70, 0.15, 0.05, 0.10]
            desc = "Severe Fan-in and Fan-out topology matches known Mule Ring template (Graph)."
        else:
            categories = ['Graph Anomaly', 'Velocity Risk', 'Behavioral Stress', 'Entropy']
            values = [0.25, 0.40, 0.20, 0.15]
            desc = "High frequency transfer from suspected IP address (Velocity)."
            
        fig_bar = px.bar(
            x=values, 
            y=['Anomaly Contribution'] * len(values), 
            orientation='h', 
            color=categories,
            color_discrete_sequence=['#ef4444', '#f59e0b', '#0ea5e9', '#8b5cf6'],
            labels={'x': 'Risk Component Contribution', 'y': ''}
        )
        fig_bar.update_layout(
            height=140, 
            margin=dict(l=10, r=10, t=10, b=10), 
            barmode='stack', 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_bar.update_xaxes(visible=True, showgrid=False)
        fig_bar.update_yaxes(visible=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(f"ℹ️ **Oracle Risk Breakdown**: {desc}")
        
    with col_side:
        st.subheader("🛡️ Analyst Assistant")
        
        # Determine card status based on selected alert
        if "4766" in selected_alert:
            st.error("🚨 **DECISION RECOMMENDATION: BLOCK**")
            urgency = "CRITICAL"
            action_code = "BLOCK"
            summary_text = "The transaction exhibits extreme graph anomalies matching known **Mule Network Hub** topologies. Coupled with velocity spikes and late-night execution, this indicates money laundering."
        elif "1071" in selected_alert:
            st.error("🚨 **DECISION RECOMMENDATION: BLOCK**")
            urgency = "CRITICAL"
            action_code = "BLOCK"
            summary_text = "Active multi-hop money distribution center. Direct connections to 12 victim nodes and 8 layering endpoints. Star pattern confirmed."
        else:
            st.warning("⚠️ **DECISION RECOMMENDATION: REVIEW**")
            urgency = "HIGH"
            action_code = "REVIEW"
            summary_text = "Transaction velocity exceeds baseline threshold by 340%. Behavioral biometrics show mild hesitation spikes. Risk score is elevated."
            
        st.markdown(f"**Urgency Level:** `{urgency}`")
        st.markdown(f"**AI Recommendation Summary:**\n{summary_text}")
        
        st.markdown("**Suggested Actions:**")
        if action_code == "BLOCK":
            st.info("🛑 **Freeze Account Immediately**")
            st.info("📋 **Escalate to Tier 2 Fraud Team**")
            st.info("⛓️ **Seal evidence in blockchain ledger**")
        else:
            st.info("📞 **Callback verification required**")
            st.info("👁️ **Add to high-frequency watchlist**")
            
        st.markdown("---")
        
        # Interactive actions
        action_btn_col1, action_btn_col2 = st.columns(2)
        
        # Action feedback in session state
        if 'action_taken' not in st.session_state:
            st.session_state.action_taken = None
            st.session_state.action_target = None
            
        with action_btn_col1:
            if st.button("🔒 Freeze Account", use_container_width=True, type="primary"):
                st.session_state.action_taken = "FREEZE"
                st.session_state.action_target = selected_alert.split(" ")[0]
        with action_btn_col2:
            if st.button("✅ Approve Clean", use_container_width=True):
                st.session_state.action_taken = "APPROVE"
                st.session_state.action_target = selected_alert.split(" ")[0]
                
        if st.session_state.action_taken:
            if st.session_state.action_taken == "FREEZE":
                st.success(f"🔒 **Account {st.session_state.action_target} frozen successfully.**\n\nNotification sent to core banking core ledger. Hyperledger Fabric evidence block sealed.")
            else:
                st.info(f"✅ **Account {st.session_state.action_target} approved.**\n\nWhitelist updated and alert resolved in console.")
            # Clear after display or on next run
            st.session_state.action_taken = None
    
        # Calculate dynamic metrics if alerts exist
        active_threats = 0
        critical_incidents = 0
        if 'realtime_alerts' in st.session_state:
            active_threats = len(st.session_state.realtime_alerts)
            critical_incidents = sum(1 for a in st.session_state.realtime_alerts if a['severity'] == 'Critical')
            
    st.markdown("---")
    
    # Fraud Heatmap Visualization
    st.subheader("🗺️ Global Threat Telemetry")
    st.markdown("**Real-time geolocation of active fraud vectors**")
    
    if 'realtime_alerts' in st.session_state and len(st.session_state.realtime_alerts) > 0:
        active_threats = st.session_state.realtime_alerts
        
        if active_threats:
            geo_df = pd.DataFrame(active_threats)
            
            size_map = {"Critical": 24, "High": 16, "Medium": 10, "Low": 6}
            color_map = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#f59e0b", "Low": "#10b981"}
            
            geo_df['size'] = geo_df['severity'].map(size_map)
            geo_df['color'] = geo_df['severity'].map(color_map)
            
            fig_geo = go.Figure(go.Scattergeo(
                lon = geo_df['lon'],
                lat = geo_df['lat'],
                text = geo_df['title'] + '<br>' + geo_df['severity'],
                mode = 'markers',
                marker = dict(
                    size = geo_df['size'],
                    color = geo_df['color'],
                    line = dict(width=1, color='rgba(255, 255, 255, 0.5)'),
                    opacity = 0.8
                )
            ))
            
            fig_geo.update_geos(
                projection_type="natural earth",
                showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.1)",
                showland=True, landcolor="rgba(30, 41, 59, 0.5)",
                showocean=True, oceancolor="rgba(15, 23, 42, 0.5)",
                showlakes=True, lakecolor="rgba(15, 23, 42, 0.5)",
                showcountries=True, countrycolor="rgba(255, 255, 255, 0.1)",
                bgcolor="rgba(0,0,0,0)"
            )
            
            fig_geo.update_layout(
                height=350,
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("No active geolocation threats detected.")
            
    st.markdown("---")
    
    export_col1, export_col2 = st.columns([4, 1])
    with export_col1:
        st.subheader("📈 Realtime Alert Analytics")
    with export_col2:
        if 'realtime_alerts' in st.session_state and st.session_state.realtime_alerts:
            csv_data = pd.DataFrame(st.session_state.realtime_alerts).to_csv(index=False)
            st.download_button(
                label="📥 Export Intel",
                data=csv_data,
                file_name=f"threat_intel_export_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("Active Threats", f"{active_threats if 'realtime_alerts' in st.session_state else 12}", "+2 this hour")
    with summary_col2:
        st.metric("Critical Incidents", f"{critical_incidents if 'realtime_alerts' in st.session_state else 3}", "Immediate Action Required" if critical_incidents > 0 else "Stable", delta_color="inverse")
    with summary_col3:
        st.metric("Avg Resolution Time", "4.2m", "-30s")
    with summary_col4:
        st.metric("False Positive Rate", "1.8%", "-0.2%")
        
    st.markdown("---")
    
    with st.expander("🚨 Realtime Fraud Alert Center", expanded=True):
    
        
        # Initialize alerts
        if 'realtime_alerts' not in st.session_state:
            st.session_state.realtime_alerts = [
                {"id": "AL-1001", "time": pd.Timestamp.now() - pd.Timedelta(seconds=45), "severity": "Critical", "title": "Mule Ring Topology Detected (Fan-out)", "category": "Graph", "lat": 51.50, "lon": -0.12, "status": "Active"},
                {"id": "AL-1002", "time": pd.Timestamp.now() - pd.Timedelta(seconds=120), "severity": "High", "title": "Velocity Spike on ACC00003254", "category": "Velocity", "lat": 40.71, "lon": -74.00, "status": "Active"},
                {"id": "AL-1003", "time": pd.Timestamp.now() - pd.Timedelta(minutes=5), "severity": "Medium", "title": "Hesitation Cadence Anomaly", "category": "Biometric", "lat": 35.67, "lon": 139.65, "status": "Active"},
                {"id": "AL-1004", "time": pd.Timestamp.now() - pd.Timedelta(minutes=15), "severity": "Low", "title": "Unusual Device Fingerprint", "category": "Device", "lat": 1.35, "lon": 103.81, "status": "Active"}
            ]
            
        # Simulate incoming alerts if live
        if is_live and np.random.random() > 0.6:
            severities = ["Low", "Medium", "High", "Critical"]
            probs = [0.4, 0.3, 0.2, 0.1]
            sev = np.random.choice(severities, p=probs)
            
            categories = {"Low": "Device", "Medium": "Biometric", "High": "Velocity", "Critical": "Graph"}
            titles = {
                "Low": ["New IP Address Login", "Unusual Browser User-Agent", "Minor Typo Correction Rate"],
                "Medium": ["Hesitation Cadence Anomaly", "High Flight Time (Keyboard)", "Location Jump (100km)"],
                "High": ["Velocity Spike on Account", "Multiple Rapid Transfers", "Known Bad Actor Interaction"],
                "Critical": ["Mule Ring Topology Detected", "Large Value Extraction", "Account Takeover Pattern"]
            }
            
            hubs = [(40.71, -74.00), (51.50, -0.12), (35.67, 139.65), (1.35, 103.81), (37.77, -122.41), (50.11, 8.68)]
            base_lat, base_lon = hubs[np.random.randint(0, len(hubs))]
            new_alert = {
                "id": f"AL-{int(time.time())}",
                "time": pd.Timestamp.now(),
                "severity": sev,
                "title": np.random.choice(titles[sev]),
                "category": categories[sev],
                "lat": base_lat + np.random.normal(0, 5.0),
                "lon": base_lon + np.random.normal(0, 5.0),
                "status": "Active"
            }
            # Prepend and keep max 50 alerts safely
            st.session_state.realtime_alerts = [new_alert] + st.session_state.realtime_alerts[:49]
                
        # Filters UI
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            # Safely try to use st.pills if available in this Streamlit version, else fallback to multiselect
            try:
                selected_severities = st.pills("Filter by Severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"], selection_mode="multi")
            except AttributeError:
                selected_severities = st.multiselect("Filter by Severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
        with filter_col2:
            search_term = st.text_input("🔍 Search Alerts", placeholder="e.g. Mule, Velocity...")
            
        # Filter alerts
        filtered_alerts = [
            a for a in st.session_state.realtime_alerts 
            if a["severity"] in (selected_severities or [])
            and (search_term.lower() in a["title"].lower() or search_term.lower() in a["category"].lower() or search_term.lower() in a["id"].lower())
        ]
        
        # Investigation Timeline
        if 'investigate_alert_id' in st.session_state and st.session_state.investigate_alert_id:
            inv_id = st.session_state.investigate_alert_id
            target_alert = next((a for a in st.session_state.realtime_alerts if a['id'] == inv_id), None)
            if target_alert:
                timeline_steps = [
                    (target_alert['time'] - pd.Timedelta(minutes=45), "Initial Login (Normal IP)"),
                    (target_alert['time'] - pd.Timedelta(minutes=10), "Security Settings Changed (2FA Disabled)"),
                    (target_alert['time'] - pd.Timedelta(minutes=2), "Large Transfer Initiated"),
                    (target_alert['time'], f"ALERT TRIGGERED: {target_alert['title']}"),
                ]
                if st.session_state.get('investigation_alert_id_active') != inv_id:
                    st.session_state.investigation_alert_id_active = inv_id
                    st.session_state.investigation_timeline_step = 0
                    st.session_state.investigation_timeline_last_tick = datetime.now(timezone.utc)
    
                _schedule_live_refresh(1000)
                _advance_timed_state(
                    'investigation_timeline_step',
                    'investigation_timeline_last_tick',
                    2.0,
                    len(timeline_steps),
                    loop=False,
                )
                current_step = int(st.session_state.get('investigation_timeline_step', 0))
    
                st.info(f"🔎 **Active Investigation:** {target_alert['title']} (`{inv_id}`)")
                st.progress((current_step + 1) / len(timeline_steps))
                timeline_col1, timeline_col2 = st.columns([3, 1])
                
                with timeline_col1:
                    for idx, (timestamp_value, label) in enumerate(timeline_steps):
                        time_label = timestamp_value.strftime('%H:%M:%S')
                        is_current = idx == current_step
                        is_complete = idx < current_step
    
                        if is_current:
                            st.info(f"**{time_label}** &nbsp; ▶ {label}")
                        elif is_complete:
                            st.markdown(f"**{time_label}** &nbsp; ✅ {label}")
                        else:
                            st.markdown(
                                f"<div style='opacity:0.4;'><strong>{time_label}</strong> &nbsp; {label}</div>",
                                unsafe_allow_html=True,
                            )
                with timeline_col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("Close Investigation", use_container_width=True, type="primary"):
                        st.session_state.investigate_alert_id = None
                        st.session_state.pop('investigation_alert_id_active', None)
                        st.session_state.pop('investigation_timeline_step', None)
                        st.session_state.pop('investigation_timeline_last_tick', None)
                        st.rerun()
                st.markdown("---")
    
        # Render Alerts
        st.markdown('<div style="max-height: 400px; overflow-y: auto; padding-right: 10px; margin-top: 15px;">', unsafe_allow_html=True)
        if not filtered_alerts:
            st.info("No alerts match the current filters.")
        else:
            for alert in filtered_alerts:
                time_str = alert["time"].strftime("%H:%M:%S")
                is_resolved = alert.get('status', 'Active') == 'Resolved'
                opacity = "0.5" if is_resolved else "1.0"
                status_badge = "Resolved" if is_resolved else alert['severity']
                
                alert_col, inv_col, btn_col = st.columns([4, 1, 1])
                with alert_col:
                    aria_label = (
                        f"{alert['severity']} alert {alert['title']} in {alert['category']} category, "
                        f"status {status_badge}, alert id {alert['id']}, time {time_str}"
                    )
                    html = f"""
                    <div class="alert-card" style="opacity: {opacity}; margin-bottom: 0;" role="article" aria-label="{aria_label}">
                        <span class="alert-time">{time_str}</span>
                        <span class="alert-title">[{alert['category']}] {alert['title']} <span style="color:#64748b; font-size:0.75rem; margin-left:8px;">#{alert['id']}</span></span>
                        <span class="severity-badge severity-{alert['severity']}" role="status" aria-live="polite" aria-label="Alert status {status_badge}">{status_badge}</span>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                with inv_col:
                    if st.button("Investigate", key=f"inv_{alert['id']}", use_container_width=True):
                        st.session_state.investigate_alert_id = alert['id']
                        st.rerun()
                with btn_col:
                    if not is_resolved:
                        if st.button("Resolve", key=f"resolve_{alert['id']}", use_container_width=True):
                            alert['status'] = 'Resolved'
                            st.rerun()
                    else:
                        st.button("Resolved", key=f"resolved_{alert['id']}", disabled=True, use_container_width=True)
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if is_live:
        _schedule_live_refresh(2000)
    
    # Page: Innovations
