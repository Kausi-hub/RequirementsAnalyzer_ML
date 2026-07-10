import streamlit as st
from datetime import datetime
import os

from requirement_engine import analyze_requirements

os.makedirs("logs", exist_ok=True)

st.set_page_config(
    page_title="Requirements Analyzer",
    layout="wide"
)

st.title("Requirements Analyzer")

st.write(
    "Analyze requirements for conflicts, ambiguity, "
    "performance constraints, and response time issues."
)

requirements_text = st.text_area(
    "Enter Requirements (one per line)",
    height=300
)

threshold = st.slider(
    "Satisfaction Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05
)

if st.button("Analyze Requirements"):

    requirements = [
        r.strip()
        for r in requirements_text.splitlines()
        if r.strip()
    ]

    if not requirements:
        st.error("Please enter at least one requirement.")
        st.stop()

    try:
        result = analyze_requirements(requirements)

        conversation_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        log_file = f"logs/conversation_{conversation_id}.md"

        with open(log_file, "w", encoding="utf-8") as f:

            f.write("# Requirement Analysis\n\n")

            for req in requirements:
                f.write(f"- {req}\n")

            f.write(
                f"\n## Satisfaction Score\n"
                f"{result['satisfaction']}\n\n"
            )

            for category, issues in result["issues"].items():
                f.write(f"\n### {category}\n")

                for issue in issues:
                    f.write(f"- {issue}\n")

        st.subheader("Analysis Results")

        st.metric(
            "Satisfaction Score",
            f"{result['satisfaction']:.2f}"
        )

        if result["satisfaction"] >= threshold:
            st.success("Requirements pass threshold.")
        else:
            st.warning("Requirements below threshold.")

        for category, issues in result["issues"].items():

            st.markdown(f"### {category}")

            if issues:
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.write("No issues detected.")

        st.markdown("---")
        st.subheader("Generated Report")

        with open(log_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        st.markdown(markdown_content)

        with open(log_file, "rb") as f:
            st.download_button(
                label="Download Report",
                data=f.read(),
                file_name=os.path.basename(log_file),
                mime="text/markdown"
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")