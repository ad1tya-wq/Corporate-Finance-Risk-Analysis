import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import json
import time
from langchain_core.messages import HumanMessage
from agent import app as agent_app

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentinel: Financial Risk Controller",
    page_icon="🛡️",
    layout="wide"  # Use the full screen width
)

# --- CUSTOM CSS (THE "ENTERPRISE" LOOK) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #f0f2f6; }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    .stChatMessage { background-color: #1f2937; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=50)
    st.title("Sentinel Protocol")
    st.caption("v1.0.4 | Connected to MySQL")
    st.markdown("---")
    # In app.py sidebar section

    if st.button("🔄 Reset System Memory"):
        # 1. Clear Chat History
        st.session_state.messages = []
        
        # 2. Delete the Memory Files (The "Hard Reset")
        files_to_delete = [
            "static/metrics.json", 
            "static/forecast_plot.png", 
            "static/active_policy.txt"
        ]
        
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)
                
        # 3. Rerun the app to refresh the UI
        st.rerun()

# --- TOP BANNER (LIVE METRICS) ---
st.title("🛡️ Corporate Financial Sentinel")

# Placeholder for metrics (we load them from the JSON file)
m1, m2, m3 = st.columns(3)

def load_metrics():
    if os.path.exists("static/metrics.json"):
        with open("static/metrics.json", "r") as f:
            return json.load(f)
    return None

metrics = load_metrics()
if metrics:
    # Color code the trend
    trend_color = "normal"
    if "RISK" in metrics['trend']:
        trend_color = "inverse" # Highlights it red in dark mode
    
    m1.metric("Risk Status", metrics['trend'], delta_color=trend_color)
    m1.markdown(f"**Status:** :red[{metrics['trend']}]" if "RISK" in metrics['trend'] else f"**Status:** :green[{metrics['trend']}]")
    m2.metric("Current Monthly Burn", f"${metrics['current_burn']:,.0f}")
    
    # Calculate delta
    delta = metrics['predicted_burn'] - metrics['current_burn']
    m3.metric("Projected Burn (90d)", f"${metrics['predicted_burn']:,.0f}", delta=f"${delta:,.0f}", delta_color="inverse")
else:
    m1.metric("Risk Status", "Waiting for analysis...")
    m2.metric("Current Monthly Burn", "--")
    m3.metric("Projected Burn (90d)", "--")

st.markdown("---")

# --- MAIN LAYOUT (CHAT + EVIDENCE) ---
col_chat, col_evidence = st.columns([2, 1])  # 2/3rds for Chat, 1/3rd for Graphs

# === RIGHT COLUMN: EVIDENCE LOCKER ===
with col_evidence:
    st.subheader("📊 Live Forecast")
    if os.path.exists("static/forecast_plot.png"):
        st.image("static/forecast_plot.png", caption="Prophet Model Projection", width="stretch")
    else:
        st.info("Run an analysis to generate the forecast plot.")

    st.subheader("📜 Active Policies")
    
    # Check if a policy file exists and if the trend is bad
    policy_file = "static/active_policy.txt"
    
    if metrics and "RISK" in metrics['trend']:
        if os.path.exists(policy_file):
            with open(policy_file, "r", encoding="utf-8") as f:
                policy_content = f.read()
            
            st.warning(f"⚠️ PROTOCOL ACTIVATED:\n\n{policy_content}")
        else:
            # Fallback if Agent hasn't called the tool yet
            st.warning("⚠️ High Risk detected. Waiting for Agent to retrieve policy details...")
            
    else:
        st.success("System Normal. No restrictive policies active.")
# === LEFT COLUMN: CHAT INTERFACE ===
with col_chat:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Old Messages
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    # Handle Input
    if user_input := st.chat_input("Command the Sentinel (e.g., 'Run full risk analysis')..."):
        # 1. Show User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 2. Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("AI Assistant is engaging agents..."):
                try:
                    # Run the Agent
                    inputs = {"messages": [HumanMessage(content=user_input)]}
                    result = agent_app.invoke(inputs)
                    bot_response = result['messages'][-1].content
                    
                    st.write(bot_response)
                    
                    # Save history
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                    # RERUN to update the top metrics immediately after analysis
                    time.sleep(1) # Tiny pause so the user sees the text first
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"System Error: {e}")