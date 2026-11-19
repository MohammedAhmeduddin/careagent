"""
app.py
~~~~~~
CareAgent Streamlit dashboard: runs the agent pipeline for an NPI through
the API and shows the peer-comparison scorecard.

    streamlit run app.py                     # uses the deployed API
    CAREAGENT_API=http://localhost:8080 streamlit run app.py
"""

import os

import requests
import streamlit as st

API_BASE = os.environ.get(
    "CAREAGENT_API", "https://careagent-api-668260909878.us-central1.run.app"
).rstrip("/")

STATUS_BADGE = {
    "typical": ("🟢", "Typical for specialty"),
    "outlier": ("🟠", "Billing-pattern outlier"),
    "insufficient_peers": ("⚪", "Not compared: too few specialty peers"),
    "insufficient_data": ("⚪", "Not compared: missing data"),
}
AGENTS = {
    "data_cleaner": ("🧹", "Data Validator", "Required fields present and valid (no imputation)"),
    "statistical": ("📊", "Statistical", "Risk-adjusted comparison with same-specialty peers"),
    "anomaly": ("🔍", "Anomaly", "Outlier rule + Isolation Forest rank, with reasons"),
    "summarizer": ("✍️", "Summarizer", "LLM narrative, number-checked (template fallback)"),
    "reporter": ("📋", "Reporter", "Final scorecard"),
}

st.set_page_config(page_title="CareAgent: Medicare Billing Patterns", page_icon="🏥", layout="wide")
st.title("🏥 CareAgent")
st.caption("Multi-agent comparison of Medicare billing patterns with same-specialty peers "
           "(CMS 2024, New Jersey)")


@st.cache_data(ttl=300)
def get_json(path: str):
    return requests.get(f"{API_BASE}{path}", timeout=10).json()


with st.sidebar:
    st.header("System status")
    try:
        health = get_json("/health")
        st.markdown(f"{'🟢' if health['status'] == 'healthy' else '🔴'} **{health['status'].title()}**")
        st.metric("Providers loaded", f"{health['providers_loaded']:,}")
        st.metric("Compared with peers", f"{health['providers_compared']:,}")
        st.metric("Outliers flagged", f"{health['outliers_flagged']:,}")
        st.metric("Specialty peer groups", f"{health['peer_groups']:,}")
    except Exception:
        st.error("API not reachable.")
    try:
        examples = get_json("/").get("example_npis", {})
    except Exception:
        examples = {}
    if examples:
        st.divider()
        st.subheader("Example NPIs")
        for status, example in examples.items():
            if example:
                st.code(example, language=None)
                st.caption(STATUS_BADGE.get(status, ("", status))[1])

npi = st.text_input("Provider NPI (10 digits)", max_chars=10)
run = st.button("▶ Run analysis", type="primary")

if run:
    if len(npi) != 10 or not npi.isdigit():
        st.error("NPI must be exactly 10 digits.")
        st.stop()
    with st.spinner("Running the agent pipeline..."):
        try:
            resp = requests.post(f"{API_BASE}/analyze", json={"npi": npi}, timeout=90)
        except requests.exceptions.RequestException:
            st.error("Cannot reach the API.")
            st.stop()
    if resp.status_code == 404:
        st.error(f"NPI {npi} is not in the dataset (New Jersey, CMS 2024).")
        st.stop()
    if resp.status_code != 200:
        st.error(f"Pipeline error: {resp.json().get('detail', resp.status_code)}")
        st.stop()
    data = resp.json()

    st.subheader("Agent pipeline")
    for key, (icon, name, desc) in AGENTS.items():
        if key in data["agents_skipped"]:
            st.markdown(f"⏭️ ~~{icon} **{name}**~~ (skipped: nothing to compare)")
        elif key in data["agents_executed"]:
            st.markdown(f"✅ {icon} **{name}**: {desc}")

    st.divider()
    icon, label = STATUS_BADGE.get(data["billing_pattern_status"], ("⚪", data["billing_pattern_status"]))
    st.markdown(f"### {icon} {label}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Specialty", data.get("provider_type") or "N/A")
    c2.metric("Location", f"{data.get('city') or ''}, {data.get('state') or ''}")
    c3.metric("Specialty peers", f"{data.get('peer_group_size') or 0:,}")

    comparison = data.get("peer_comparison") or {}
    if any(v.get("ratio_to_peer_median") is not None for v in comparison.values()):
        st.subheader("Compared with the specialty median")
        cols = st.columns(len(comparison))
        for col, item in zip(cols, comparison.values()):
            col.metric(item["label"].capitalize(), f"{item['ratio_to_peer_median']:.2f}x",
                       help=f"Percentile {item['percentile']:.0f}, modified z {item['modified_z']:.2f}")
        if data.get("isolation_forest_percentile") is not None:
            st.caption(f"Isolation Forest unusualness within specialty: "
                       f"{data['isolation_forest_percentile']:.0f}th percentile")

    for reason in data.get("outlier_reasons") or []:
        st.warning(reason)

    st.subheader("Summary")
    st.info(data.get("performance_narrative") or "")
    st.caption(("Written by the language model and number-checked."
                if data.get("narrative_source") == "llm"
                else "Written from a template (model unavailable or failed the number check)."))
    st.caption(f"Run {data['run_id']} · {data['pipeline_duration_seconds']:.3f}s · "
               f"scorecard v{data.get('scorecard_version')}")
