"""
app.py
~~~~~~~
CareAgent Streamlit Dashboard.
Shows live agent progress and provider scorecard.

Run with:
    streamlit run app.py
"""

import time
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareAgent — Provider Quality Intelligence",
    page_icon="🏥",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏥 CareAgent")
st.caption("Multi-Agent Provider Quality Scoring System")

# ── Sidebar — health ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        status_color = "🟢" if health["status"] == "healthy" else "🔴"
        st.markdown(f"{status_color} **{health['status'].title()}**")
        st.metric("Providers Loaded",  f"{health['providers_loaded']:,}")
        st.metric("Providers Scored",  f"{health['providers_scored']:,}")
        st.metric("Anomalies Flagged", f"{health['anomalies_flagged']:,}")
    except Exception:
        st.error("API not reachable. Start with: uvicorn careagent.api.main:app")

    st.divider()
    st.caption("CareAgent v0.1.0")

# ── Main — NPI input ───────────────────────────────────────────────────────────
st.subheader("Analyze a Provider")

col1, col2 = st.columns([3, 1])
with col1:
    npi = st.text_input(
        "Enter Provider NPI (10 digits)",
        placeholder="e.g. 1000153386",
        max_chars=10,
    )
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Pipeline execution ─────────────────────────────────────────────────────────
if run_btn and npi:
    if len(npi) != 10 or not npi.isdigit():
        st.error("NPI must be exactly 10 digits.")
    else:
        st.divider()
        st.subheader("Agent Pipeline")

        # Agent stepper UI
        agent_labels = {
            "data_cleaner": ("🧹", "Data Cleaner",    "Validates and imputes missing fields"),
            "statistical":  ("📊", "Statistical",     "Quality score vs national benchmarks"),
            "anomaly":      ("🔍", "Anomaly Detector","Isolation Forest cost/quality analysis"),
            "summarizer":   ("✍️",  "Summarizer",      "GPT-4o-mini performance narrative"),
            "reporter":     ("📋", "Reporter",        "Assembles final scorecard"),
        }

        # Show stepper with spinner
        placeholders = {}
        for key, (icon, name, desc) in agent_labels.items():
            placeholders[key] = st.empty()
            placeholders[key].markdown(
                f"⬜ **{icon} {name}** — {desc}"
            )

        result_placeholder = st.empty()

        with st.spinner("Running pipeline..."):
            start = time.time()
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"npi": npi},
                    timeout=60,
                )
                duration = time.time() - start

                if resp.status_code == 404:
                    st.error(f"NPI {npi} not found in database.")
                elif resp.status_code != 200:
                    st.error(f"Pipeline error: {resp.json().get('detail', 'Unknown error')}")
                else:
                    data = resp.json()
                    executed = data.get("agents_executed", [])
                    skipped  = data.get("agents_skipped",  [])

                    # Update stepper
                    for key, (icon, name, desc) in agent_labels.items():
                        if key in skipped:
                            placeholders[key].markdown(
                                f"⏭️ ~~**{icon} {name}**~~ — *skipped (not needed)*"
                            )
                        elif key in executed:
                            placeholders[key].markdown(
                                f"✅ **{icon} {name}** — {desc}"
                            )

                    # ── Scorecard ──────────────────────────────────────────────
                    st.divider()
                    st.subheader("Provider Scorecard")

                    # Recommendation badge
                    rec = data.get("network_recommendation", "review")
                    rec_color = {"include": "🟢", "review": "🟡", "exclude": "🔴"}.get(rec, "🟡")
                    st.markdown(f"### {rec_color} Network Recommendation: **{rec.upper()}**")

                    # Provider info
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Provider", data.get("provider_name", "N/A"))
                    c2.metric("Specialty", data.get("provider_type", "N/A"))
                    c3.metric("State", data.get("state", "N/A"))

                    st.divider()

                    # Quality metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Quality Score",
                        f"{data.get('quality_score', 0):.1f} / 100",
                    )
                    c2.metric(
                        "National Percentile",
                        f"{data.get('quality_percentile', 0):.1f}%",
                    )
                    c3.metric(
                        "Cost Efficiency Ratio",
                        f"{data.get('cost_efficiency_ratio', 0):.2f}x",
                        help="Submitted charge / Medicare payment. Lower = more efficient.",
                    )
                    c4.metric(
                        "Volume Percentile",
                        f"{data.get('volume_percentile', 0):.1f}%",
                    )

                    st.divider()

                    # Anomaly + data quality
                    c1, c2, c3 = st.columns(3)
                    anomaly_icon = "🚨 Yes" if data.get("is_anomaly") else "✅ No"
                    c1.metric("Anomaly Flagged", anomaly_icon)
                    c2.metric(
                        "Data Quality Score",
                        f"{data.get('data_quality_score', 0) * 100:.0f}%",
                    )
                    c3.metric(
                        "Fields Imputed",
                        data.get("fields_imputed", 0),
                    )

                    if data.get("is_anomaly") and data.get("anomaly_reason"):
                        st.warning(f"⚠️ **Anomaly reason:** {data['anomaly_reason']}")

                    st.divider()

                    # Narrative
                    st.subheader("Performance Narrative")
                    st.info(data.get("performance_narrative", "No narrative generated."))

                    st.divider()

                    # Pipeline metadata
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pipeline Duration", f"{data.get('pipeline_duration_seconds', 0):.3f}s")
                    c2.metric("Agents Executed", len(executed))
                    c3.metric("Agents Skipped", len(skipped))

                    st.caption(f"Run ID: {data.get('run_id')} | Generated: {data.get('generated_at')}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

elif run_btn and not npi:
    st.warning("Please enter a NPI before running.")

# ── Instructions ───────────────────────────────────────────────────────────────
if not run_btn:
    st.divider()
    st.markdown("""
    ### How to use
    1. Enter a 10-digit provider NPI in the field above
    2. Click **Run Analysis**
    3. Watch the 5 agents process the provider in real time
    4. Review the scorecard and network recommendation

    ### Sample NPIs to try
    | NPI | Provider | Specialty |
    |-----|----------|-----------|
    | 1000153386 | Davis | Orthopedic Surgery |
    | 1000367517 | Smith | Psychiatry |
    | 1001518755 | Thomas | Anesthesiology |
    """)
