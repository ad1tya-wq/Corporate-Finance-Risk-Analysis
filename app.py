import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import streamlit as st
from langchain_core.messages import HumanMessage

from agent import app as agent_app

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentinel: Financial Risk Controller",
    page_icon="🛡️",
    layout="wide",
)

# --- CUSTOM CSS (THE "ENTERPRISE" LOOK) ---
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
# All per-run data (forecast metrics, plot, retrieved policy text) lives in
# st.session_state instead of static/*.json|png|txt on disk. That removes
# the file-based IPC race: state now comes straight back from the graph's
# invoke() result for this session, not a re-read of a shared file another
# session could be writing to.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "policy_chunks" not in st.session_state:
    st.session_state.policy_chunks = None

# --- SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=50)
    st.title("Sentinel Protocol")
    st.caption("v1.1.0 | Connected to MySQL")
    st.markdown("---")

    if st.button("🔄 Reset System Memory"):
        # Clears this session's view only. forecast_log in MySQL (the
        # durable audit trail) is untouched.
        st.session_state.messages = []
        st.session_state.forecast_result = None
        st.session_state.policy_chunks = None
        st.rerun()

# --- TOP BANNER (LIVE METRICS) ---
st.title("🛡️ Corporate Financial Sentinel")

m1, m2, m3 = st.columns(3)

metrics = st.session_state.forecast_result
if metrics and metrics.get("trend") not in (None, "ERROR"):
    trend = metrics["trend"]
    is_risk = "RISK" in trend or "CRITICAL" in trend
    trend_color = "inverse" if is_risk else "normal"

    m1.metric("Risk Status", trend, delta_color=trend_color)
    m1.markdown(f"**Status:** :red[{trend}]" if is_risk else f"**Status:** :green[{trend}]")
    m2.metric("Current Monthly Burn", f"${metrics['current_burn']:,.0f}")

    delta = metrics["predicted_burn"] - metrics["current_burn"]
    m3.metric("Projected Burn (90d)", f"${metrics['predicted_burn']:,.0f}", delta=f"${delta:,.0f}", delta_color="inverse")
else:
    m1.metric("Risk Status", "Waiting for analysis...")
    m2.metric("Current Monthly Burn", "--")
    m3.metric("Projected Burn (90d)", "--")

st.markdown("---")

# --- MAIN LAYOUT (CHAT + EVIDENCE) ---
col_chat, col_evidence = st.columns([2, 1])

# === RIGHT COLUMN: EVIDENCE LOCKER ===
with col_evidence:
    st.subheader("📊 Live Forecast")
    if metrics and metrics.get("plot_bytes"):
        st.image(metrics["plot_bytes"], caption="Prophet Model Projection", width="stretch")
    else:
        st.info("Run an analysis to generate the forecast plot.")

    st.subheader("📜 Active Policies")
    is_risk = bool(metrics) and ("RISK" in metrics.get("trend", "") or "CRITICAL" in metrics.get("trend", ""))

    if is_risk:
        if st.session_state.policy_chunks:
            policy_content = "\n\n---\n\n".join(st.session_state.policy_chunks)
            st.warning(f"⚠️ PROTOCOL ACTIVATED:\n\n{policy_content}")
        else:
            st.warning("⚠️ High Risk detected. Waiting for Agent to retrieve policy details...")
    else:
        st.success("System Normal. No restrictive policies active.")

# === LEFT COLUMN: CHAT INTERFACE ===
with col_chat:
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    if user_input := st.chat_input("Command the Sentinel (e.g., 'Run full risk analysis')..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Sentinel is analyzing..."):
                try:
                    inputs = {"messages": [HumanMessage(content=user_input)]}
                    result = agent_app.invoke(inputs)
                    bot_response = result["messages"][-1].content

                    if result.get("forecast_result"):
                        st.session_state.forecast_result = result["forecast_result"]
                    if result.get("policy_chunks"):
                        st.session_state.policy_chunks = result["policy_chunks"]

                    st.write(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})

                    time.sleep(1)  # let the user read the response before the rerun
                    st.rerun()

                except Exception as e:
                    st.error(f"System Error: {e}")
