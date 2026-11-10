import io
import os
import statistics
import pandas as pd
import streamlit as st
from src.pdf_utils import get_paper_text
from src.agent import run_single_evaluation_pass

st.set_page_config(page_title="FMARC AI Leaderboard Evaluator", layout="wide")

st.title("FMARC AI Referee Leaderboard Evaluator")
st.markdown("Executes a 3-pass independent evaluation per paper, strictly enforcing absolute scientific scoring.")

uploaded_files = st.file_uploader("Upload Finance Working Papers (PDF)", type="pdf", accept_multiple_files=True)

SCORE_FIELDS = ["overall_score", "significance", "originality", "correctness", "data_and_methodology", "exposition"]
TEXT_FIELDS = ["paper_type", "main_limiting_reason", "revision_burden", "confidence_level", "confidence_rationale",
               "strengths", "decisive_weaknesses", "major_fixable_weaknesses", "optional_improvements"]

if uploaded_files and st.button("Evaluate Papers"):
    # Guardrail: Verify API Key is present
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("Missing API Key: Please set the OPENROUTER_API_KEY environment variable before running.")
        st.stop()

    st.info("Running 3 independent evaluation passes per paper to reduce idiosyncratic noise...")
    progress_bar = st.progress(0)
    total_passes = len(uploaded_files) * 3
    done_passes = 0
    rows = []

    for uploaded_file in uploaded_files:
        manuscript_text = get_paper_text(uploaded_file)

        results = []
        for _ in range(3):
            pass_result = run_single_evaluation_pass(manuscript_text)
            if pass_result:
                results.append(pass_result)
            done_passes += 1
            progress_bar.progress(done_passes / total_passes)

        if not results:
            st.warning(f"{uploaded_file.name}: the agent failed to return any valid evaluation. Skipped.")
            continue

        # Average the numeric dimensions; take the qualitative feedback from the pass closest to the overall average
        row = {"file": uploaded_file.name}
        row.update({k: round(statistics.mean([r[k] for r in results])) for k in SCORE_FIELDS})
        rep = min(results, key=lambda x: abs(x["overall_score"] - row["overall_score"]))
        row.update({k: rep[k] for k in TEXT_FIELDS})
        rows.append(row)

    if not rows:
        st.error("No paper produced a valid evaluation. Please try again.")
        st.stop()

    st.success(f"Evaluated {len(rows)} of {len(uploaded_files)} paper(s).")

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    st.download_button(
        "Download results as Excel",
        data=excel_buffer.getvalue(),
        file_name="fmarc_evaluations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click="ignore",  # don't rerun the app on download, which would wipe the results
    )
