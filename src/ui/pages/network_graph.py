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


@with_error_boundary('🕸️ Network Graph Explorer')
def render_network_graph(helpers, st_globals):
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

    st.header("🕸️ Real-Time Fraud Network Explorer")
    st.markdown("Visualizing multi-hop money laundering patterns, star hubs, and mule accounts.")
    st.markdown("---")
    
    # Local imports consolidated globally
    
    # Initialize graph (same structure for consistency)
    G = nx.DiGraph()
    central_hub = "ACC00001071"
    G.add_node(central_hub, type="Mule Hub (Level 1)", risk=0.95)
    
    # Fan in (victims to hub)
    for i in range(12):
        node_id = f"ACC_VICTIM_{i}"
        G.add_node(node_id, type="Victim", risk=0.1)
        G.add_edge(node_id, central_hub, amount=random.randint(5000, 50000))
        
    # Fan out (hub to layer mules)
    for i in range(8):
        node_id = f"ACC_LAYER_{i}"
        G.add_node(node_id, type="Mule Layer (Level 2)", risk=0.85)
        G.add_edge(central_hub, node_id, amount=random.randint(20000, 100000))
        
        # Further distribution
        for j in range(2):
            end_node = f"ACC_END_{i}_{j}"
            G.add_node(end_node, type="Withdrawal Node", risk=0.9)
            G.add_edge(node_id, end_node, amount=random.randint(10000, 40000))
    
    # Controls for active propagation tracking
    st.subheader("📡 Active Fraud Propagation Tracking")
    
    col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
    
    with col_c1:
        animate_propagation = st.toggle("🔴 Auto-Play Simulation", value=False, key="graph_animate_propagation")
    with col_c2:
        step_option = st.selectbox("Select Simulation Step", ["Step 1: Infiltration", "Step 2: Aggregation", "Step 3: Layering", "Step 4: Cashout"], key="simulation_step_selectbox")
    with col_c3:
        st.write("")
        st.write("")
        st.caption("Auto-Play automatically cycles through transaction lifecycle steps.")
        
    # Mapping selection/auto-play step to index
    step_indices = {
        "Step 1: Infiltration": 0,
        "Step 2: Aggregation": 1,
        "Step 3: Layering": 2,
        "Step 4: Cashout": 3
    }
    
    if 'prop_step' not in st.session_state:
        st.session_state.prop_step = 0
        
    if animate_propagation:
        if st.session_state.get('prop_animate_enabled') != True:
            st.session_state.prop_animate_enabled = True
            st.session_state.prop_last_advance_at = datetime.now(timezone.utc)
    
        _schedule_live_refresh(1000)
        _advance_timed_state('prop_step', 'prop_last_advance_at', 2.0, 4, loop=True)
        # Sync selectbox
        step_names = list(step_indices.keys())
        selected_step_name = step_names[st.session_state.prop_step]
    else:
        if st.session_state.get('prop_animate_enabled') != False:
            st.session_state.prop_animate_enabled = False
            st.session_state.prop_last_advance_at = datetime.now(timezone.utc)
        st.session_state.prop_step = step_indices[step_option]
        selected_step_name = step_option
    
    # Show progress bar & description of propagation
    st.progress((st.session_state.prop_step + 1) * 25)
    
    if st.session_state.prop_step == 0:
        st.info("🟢 **Propagation Phase 1: Victim Infiltration** — Victims' accounts are compromised or social engineered. Funds are being initiated into the laundering ring.")
    elif st.session_state.prop_step == 1:
        st.error("🚨 **Propagation Phase 2: Hub Aggregation** — Funds are aggregated at the central Mule Hub (`ACC00001071`) from multiple sources simultaneously to evade velocity checks.")
    elif st.session_state.prop_step == 2:
        st.warning("⚡ **Propagation Phase 3: Outbound Layering** — High velocity dispersion of funds from the primary hub to secondary Level 2 Layering accounts.")
    else:
        st.error("🛑 **Propagation Phase 4: Cashout Extraction** — Layered accounts distribute funds to withdrawal endpoints (ATM nodes/crypto gateways) for final extraction.")
    
    # Interactivity settings panel
    col_p1, col_p2, col_p3 = st.columns([1.5, 1, 1.5])
    with col_p1:
        search_query = st.text_input("🔍 Search Account ID (e.g. ACC_VICTIM_3)", value="", key="graph_search_box")
    with col_p2:
        physics_enabled = st.toggle("🔒 Dynamic Spring Physics", value=True, key="graph_physics_toggle")
    with col_p3:
        st.write("")
        st.caption("💡 **Tip**: Double-click any node to expand or collapse its transaction paths!")
    
    # === CONSTRUCT VIS.JS DATA ===
    vis_nodes = []
    for node in G.nodes():
        role = G.nodes[node]['type']
        risk = G.nodes[node]['risk']
        
        # Shapes & Sizes based on Role
        if role == 'Mule Hub (Level 1)':
            shape = 'star'
            size = 35
            group = 'hub'
            label = node
        elif 'Layer' in role:
            shape = 'dot'
            size = 20
            group = 'layer'
            label = ''
        elif role == 'Victim':
            shape = 'dot'
            size = 15
            group = 'victim'
            label = ''
        else:
            shape = 'diamond'
            size = 18
            group = 'cashout'
            label = ''
            
        # Color coding based on risk scores
        if risk >= 0.7:
            color = {
                'background': '#ef4444',
                'border': '#b91c1c',
                'highlight': {'background': '#f87171', 'border': '#dc2626'},
                'hover': {'background': '#f87171', 'border': '#dc2626'}
            }
        elif risk >= 0.4:
            color = {
                'background': '#f59e0b',
                'border': '#b45309',
                'highlight': {'background': '#fbbf24', 'border': '#d97706'},
                'hover': {'background': '#fbbf24', 'border': '#d97706'}
            }
        else:
            color = {
                'background': '#10b981',
                'border': '#047857',
                'highlight': {'background': '#34d399', 'border': '#059669'},
                'hover': {'background': '#34d399', 'border': '#059669'}
            }
            
        # Dynamic tooltip with risk statistics and node details
        title = f"""
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; padding: 10px; color: #f1f5f9; background: #0f172a; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); font-size: 13px;">
            <b style="color: #38bdf8; font-size: 14px;">{node}</b><br/>
            <b>Role:</b> {role}<br/>
            <b>Risk Score:</b> <span style="color: {'#ef4444' if risk >= 0.7 else '#f59e0b' if risk >= 0.4 else '#10b981'}">{risk:.2%}</span><br/>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.1);"/>
            <span style="font-size: 11px; color: #94a3b8;">Double-click to expand/collapse connections</span>
        </div>
        """
        
        # Hide connected nodes initially in static exploration mode to allow click-to-discover
        hidden = False
        if role != 'Mule Hub (Level 1)' and not animate_propagation and not search_query:
            hidden = True
            
        vis_nodes.append({
            'id': node,
            'label': label,
            'title': title,
            'shape': shape,
            'size': size,
            'color': color,
            'hidden': hidden,
            'group': group
        })
    
    vis_edges = []
    for edge in G.edges():
        u, v = edge
        amount = G[u][v].get('amount', 10000)
        
        is_active_edge = False
        if st.session_state.prop_step == 0 and G.nodes[u]['type'] == 'Victim':
            is_active_edge = True
        elif st.session_state.prop_step == 1 and u == central_hub:
            is_active_edge = True
        elif st.session_state.prop_step == 2 and G.nodes[u]['type'] == 'Mule Layer (Level 2)' and G.nodes[v]['type'] == 'Withdrawal Node':
            is_active_edge = True
        elif st.session_state.prop_step == 3:
            is_active_edge = True
            
        width = 3.0 if is_active_edge else 1.0
        color = {
            'color': '#ef4444' if is_active_edge else 'rgba(148, 163, 184, 0.4)',
            'highlight': '#ef4444',
            'hover': '#ef4444'
        }
        
        title = f"""
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; padding: 6px 10px; color: #f1f5f9; background: #1e293b; border-radius: 6px; font-size: 12px;">
            <b>Transfer Amount:</b> ₹{amount:,.2f}<br/>
            <b>Route:</b> {u} ➡️ {v}
        </div>
        """
        
        hidden = False
        if (G.nodes[u]['type'] != 'Mule Hub (Level 1)' and G.nodes[v]['type'] != 'Mule Hub (Level 1)') and not animate_propagation and not search_query:
            hidden = True
            
        vis_edges.append({
            'from': u,
            'to': v,
            'value': amount,
            'width': width,
            'color': color,
            'title': title,
            'hidden': hidden,
            'arrows': 'to'
        })
    
    # Render custom HTML5 vis.js graph Component
    import json
    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    physics_val = "true" if physics_enabled else "false"
    search_val = search_query.strip() if search_query.strip() else "None"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Network Explorer</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background-color: #0b0f19;
                overflow: hidden;
            }}
            #network-canvas {{
                width: 100%;
                height: 100%;
                border: none;
            }}
            div.vis-tooltip {{
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }}
        </style>
    </head>
    <body>
        <div id="network-canvas"></div>
        <script type="text/javascript">
            var nodesRaw = {nodes_json};
            var edgesRaw = {edges_json};
    
            // Convert raw HTML title strings into actual DOM elements so vis.js renders them as HTML
            nodesRaw.forEach(function(node) {{
                if (node.title) {{
                    var el = document.createElement("div");
                    el.innerHTML = node.title;
                    node.title = el;
                }}
            }});
    
            edgesRaw.forEach(function(edge) {{
                if (edge.title) {{
                    var el = document.createElement("div");
                    el.innerHTML = edge.title;
                    edge.title = el;
                }}
            }});
    
            var nodes = new vis.DataSet(nodesRaw);
            var edges = new vis.DataSet(edgesRaw);
    
            var container = document.getElementById('network-canvas');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
    
            var options = {{
                physics: {{
                    enabled: {physics_val},
                    stabilization: {{
                        enabled: true,
                        iterations: 100,
                        fit: true
                    }},
                    barnesHut: {{
                        gravitationalConstant: -2500,
                        centralGravity: 0.15,
                        springLength: 95,
                        springConstant: 0.04
                    }}
                }},
                interaction: {{
                    hover: true,
                    zoomView: true,
                    dragView: true,
                    navigationButtons: true
                }},
                nodes: {{
                    borderWidth: 2,
                    shadow: true,
                    font: {{
                        color: '#f1f5f9',
                        size: 13,
                        face: 'Plus Jakarta Sans, sans-serif'
                    }}
                }},
                edges: {{
                    shadow: true,
                    smooth: {{
                        type: 'continuous'
                    }}
                }}
            }};
    
            var network = new vis.Network(container, data, options);
    
            // Double-click dynamic expander
            network.on("doubleClick", function(params) {{
                if (params.nodes.length > 0) {{
                    var clickedNode = params.nodes[0];
                    var connectedNodes = network.getConnectedNodes(clickedNode);
                    var connectedEdges = network.getConnectedEdges(clickedNode);
    
                    var anyHidden = false;
                    connectedNodes.forEach(function(nodeId) {{
                        var n = nodes.get(nodeId);
                        if (n && n.hidden) {{
                            anyHidden = true;
                        }}
                    }});
    
                    connectedNodes.forEach(function(nodeId) {{
                        if (nodeId !== clickedNode) {{
                            var n = nodes.get(nodeId);
                            if (n) {{
                                var newLabel = n.group === 'victim' ? 'Victim Node' : n.group === 'layer' ? 'Layer Node' : 'Cashout Node';
                                nodes.update({{
                                    id: nodeId, 
                                    hidden: !anyHidden,
                                    label: !anyHidden ? newLabel : ''
                                }});
                            }}
                        }}
                    }});
    
                    connectedEdges.forEach(function(edgeId) {{
                        var e = edges.get(edgeId);
                        if (e) {{
                            edges.update({{id: edgeId, hidden: !anyHidden}});
                        }}
                    }});
                }}
            }});
    
            // Search focusing
            var searchId = "{search_val}";
            if (searchId && searchId !== "None") {{
                setTimeout(function() {{
                    var targetNode = nodes.get(searchId);
                    if (targetNode) {{
                        nodes.update({{id: searchId, hidden: false, label: searchId}});
                        network.selectNodes([searchId]);
                        network.focus(searchId, {{
                            scale: 1.6,
                            animation: {{
                                duration: 1200,
                                easingFunction: 'easeInOutQuad'
                            }}
                        }});
                    }}
                }}, 500);
            }}
        </script>
    </body>
    </html>
    """
    
    st.components.v1.html(html_content, height=600)
    
    st.markdown("""
    ### 📊 Topology Analytics
    *   🔴 **Red Glowing Nodes / Edges**: Represents the active transactions in the selected propagation simulation step.
    *   🟢 **Green Nodes**: Low risk accounts ($<40\\%$ risk).
    *   🟡 **Amber Nodes**: Medium risk review accounts ($40\\%-70\\%$ risk).
    *   🔴 **Solid Red Nodes**: High risk blocked accounts ($\\ge 70\\%$ risk).
    *   ⭐ **Star Shape**: Central Mule Hub (`ACC00001071`).
    *   🔷 **Diamond Shape**: Withdrawal endpoints (ATM nodes/crypto gateways).
    *   **Double-Click Interactive Mode**: Double-clicking any node dynamically expands or collapses its transactions with smooth spring physics.
    """)
    
    # Page: Behavioral Biometrics
