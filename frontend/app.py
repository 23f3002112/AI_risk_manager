"""
Streamlit Dashboard — Track 02: AI Risk Manager
Interactive dashboard for the AI Risk Manager backend.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Risk Manager Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e2e;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Risk Manager Command Center")
st.markdown("Monitor order risk, interact with the review queue, and visualize cost dynamics in real-time.")

# Test connection
try:
    requests.get(f"{API_URL}/daily-summary")
except Exception as e:
    st.error(f"Could not connect to backend API at {API_URL}. Is FastAPI running?")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Review Queue", "💰 Cost Simulator", "🩺 Model Health"])

with tab1:
    st.header("Human Review Queue")
    st.markdown("Orders flagged by the AI for potential return-risk. Manual review required.")
    
    # Load queue
    queue_data = requests.get(f"{API_URL}/review-queue").json().get("queue", [])
    
    if not queue_data:
        st.success("✅ The review queue is currently empty. All flagged orders have been processed.")
    else:
        st.info(f"**{len(queue_data)} orders pending review.** Showing up to 10.")
        
        for item in queue_data[:10]:
            with st.container():
                st.markdown(f"### Order: `{item['order_id']}`")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Action Recommended:** `{item['action']}`")
                    st.markdown(f"**Reason:** {item['reason']}")
                    st.markdown("**Primary Risk Factors:**")
                    for exp in item["explanation"]:
                        st.markdown(f"- *{exp}*")
                with col2:
                    st.metric("Risk Score", f"{item['risk_score']:.2f}")
                    
                    if st.button("✅ Approve", key=f"app_{item['order_id']}", use_container_width=True):
                        requests.post(f"{API_URL}/review-queue/{item['order_id']}/resolve", json={"decision": "approve"})
                        st.rerun()
                    if st.button("🚫 Reject", key=f"rej_{item['order_id']}", use_container_width=True):
                        requests.post(f"{API_URL}/review-queue/{item['order_id']}/resolve", json={"decision": "reject"})
                        st.rerun()
                st.divider()
                
        if len(queue_data) > 10:
            st.markdown(f"*...and {len(queue_data) - 10} more orders in the queue.*")

with tab2:
    st.header("Live Cost Simulator")
    st.markdown("Adjust the assumed business costs to dynamically re-evaluate the optimal threshold.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        cost_fp = st.slider("Cost of False Positive (INR) - e.g., Customer insult, manual review cost", 
                            min_value=10, max_value=500, value=15, step=5)
    with col2:
        cost_fn = st.slider("Cost of False Negative (INR) - e.g., Return shipping, restocking", 
                            min_value=100, max_value=1000, value=180, step=10)
                            
    # Simulate costs
    sim_data = requests.post(f"{API_URL}/simulate-cost", json={"cost_fp": cost_fp, "cost_fn": cost_fn}).json().get("results", [])
    
    if sim_data:
        df_sim = pd.DataFrame(sim_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        # Use a modern color palette
        ax.plot(df_sim["threshold"], df_sim["total_cost_inr"], marker="o", color="#4C72B0", linewidth=2, label="Total Cost (INR)")
        
        # Find minimum cost
        min_idx = df_sim["total_cost_inr"].idxmin()
        best_thresh = df_sim.loc[min_idx, "threshold"]
        best_cost = df_sim.loc[min_idx, "total_cost_inr"]
        
        ax.plot(best_thresh, best_cost, marker="*", color="#C44E52", markersize=15, label=f"Optimal Threshold ({best_thresh})")
        
        ax.set_title("Total Cost vs. Risk Threshold", fontsize=14, pad=15)
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Total Cost (INR)", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
        st.markdown("### Detailed Threshold Breakdown")
        st.dataframe(df_sim.style.format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1": "{:.3f}",
            "total_cost_inr": "₹{:,.0f}",
            "baseline_cost_do_nothing_inr": "₹{:,.0f}",
            "net_savings_inr": "₹{:,.0f}"
        }), use_container_width=True)

with tab3:
    st.header("Model Health & Drift Status")
    st.markdown("Monitors calibration across different risk buckets to identify concept drift early.")
    
    drift_data = requests.get(f"{API_URL}/drift-status").json()
    
    is_drifted = drift_data["drift_detected"]
    if is_drifted:
        st.error("🚨 **DRIFT DETECTED**: Model calibration has degraded in specific score buckets. Retraining recommended.")
    else:
        st.success("✅ **HEALTHY**: Model is well-calibrated across all risk buckets.")
        
    st.markdown("---")
    st.markdown("### Calibration Table")
    df_cal = pd.DataFrame(drift_data["calibration_table"])
    
    # Highlight row where drift_flag is True
    st.dataframe(
        df_cal.style.apply(lambda x: ['background: rgba(255, 0, 0, 0.2)' if x['drift_flag'] else '' for i in x], axis=1)\
                    .format({"actual_return_rate": "{:.3f}", "expected_midpoint": "{:.3f}", "calibration_gap": "{:.3f}"}),
        use_container_width=True
    )
    
    reasons = drift_data.get("miscalibration_reasons", {})
    if reasons:
        st.markdown("### Root Cause Analysis (Feature Deviation)")
        st.markdown("The system automatically compares the feature distribution of the miscalibrated buckets against a healthy baseline to explain the drift.")
        for bucket, bucket_reasons in reasons.items():
            with st.expander(f"Drift Analysis for Bucket: {bucket}", expanded=True):
                for r in bucket_reasons:
                    st.markdown(f"- ⚠️ {r}")
