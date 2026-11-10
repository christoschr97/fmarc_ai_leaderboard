import os
import statistics
import streamlit as st
from src.pdf_utils import get_paper_text
from src.agent import run_single_evaluation_pass

st.set_page_config(page_title="FMARC AI Leaderboard Evaluator", layout="wide")

st.title("FMARC AI Referee Leaderboard Evaluator")
st.markdown("Executes a 3-pass independent evaluation strictly enforcing absolute scientific scoring.")

uploaded_file = st.file_uploader("Upload a Finance Working Paper (PDF)", type="pdf")

if uploaded_file and st.button("Evaluate Paper"):
    # Guardrail: Verify API Key is present
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("Missing API Key: Please set the OPENROUTER_API_KEY environment variable before running.")
        st.stop()

    with st.spinner("Extracting text from PDF..."):
        manuscript_text = get_paper_text(uploaded_file)

    st.info("Running 3 independent evaluation passes to reduce idiosyncratic noise...")

    results = []
    progress_bar = st.progress(0)

    for i in range(3):
        pass_result = run_single_evaluation_pass(manuscript_text)
        if pass_result:
            results.append(pass_result)
        progress_bar.progress((i + 1) / 3)

    if not results:
        st.error("The agent failed to return any valid structured evaluations. Please try again.")
        st.stop()

    st.success(f"Completed {len(results)} valid evaluation pass(es). Aggregating scores...")

    # 1. Average the Numeric Dimensions
    avg_overall = round(statistics.mean([r["overall_score"] for r in results]))
    avg_significance = round(statistics.mean([r["significance"] for r in results]))
    avg_originality = round(statistics.mean([r["originality"] for r in results]))
    avg_correctness = round(statistics.mean([r["correctness"] for r in results]))
    avg_data = round(statistics.mean([r["data_and_methodology"] for r in results]))
    avg_exposition = round(statistics.mean([r["exposition"] for r in results]))

    # Select the qualitative feedback from the pass closest to the overall average
    rep = min(results, key=lambda x: abs(x["overall_score"] - avg_overall))

    # 2. Render Metrics UI
    st.header(f"Overall Scientific Quality: {avg_overall}/100")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Significance", avg_significance)
    col2.metric("Originality", avg_originality)
    col3.metric("Correctness", avg_correctness)
    col4.metric("Data & Method", avg_data)
    col5.metric("Exposition", avg_exposition)

    st.divider()

    # 3. Render Qualitative Feedback
    st.subheader("Referee Feedback")
    st.write(f"**Paper Type:** {rep['paper_type']}")
    st.write(f"**Main Limiting Reason:** {rep['main_limiting_reason']}")
    st.write(f"**Revision Burden:** {rep['revision_burden']}")
    st.write(f"**Confidence Level:** {rep['confidence_level']} — *{rep['confidence_rationale']}*")

    st.markdown("### Detailed Breakdowns")

    def render_semicolon_list(raw_str: str):
        items = [item.strip() for item in raw_str.split(";") if item.strip()]
        for item in items:
            st.markdown(f"- {item}")

    with st.expander("Strengths", expanded=True):
        render_semicolon_list(rep["strengths"])

    with st.expander("Decisive Weaknesses", expanded=True):
        render_semicolon_list(rep["decisive_weaknesses"])

    with st.expander("Major Fixable Weaknesses"):
        render_semicolon_list(rep["major_fixable_weaknesses"])

    with st.expander("Optional Improvements"):
        render_semicolon_list(rep["optional_improvements"])
