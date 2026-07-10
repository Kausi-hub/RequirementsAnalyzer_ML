import streamlit as st
import json
import os
from datetime import datetime

from requirement_engine import analyze_requirements

# ==========================================================
# Configuration
# ==========================================================

st.set_page_config(
    page_title="Requirements Analyzer",
    layout="wide"
)

os.makedirs("logs", exist_ok=True)

STATE_FILE = "last_state.json"

# ==========================================================
# Sample Requirements
# ==========================================================

SAMPLE_REQUIREMENTS = """REQ-MC-001: The motor controller shall start within 100 ms of receiving a START command.

REQ-MC-002: The motor controller shall stop within 50 ms of receiving a STOP command.

REQ-MC-003: The motor controller shall accelerate from 0 RPM to 3000 RPM within 100 ms.

REQ-MC-004: The motor controller shall limit acceleration to 10000 RPM/s.

REQ-MC-005: The motor controller shall maintain motor speed within ±2%.

REQ-MC-006: The motor controller shall enter low-power mode within 5 ms after shutdown.

REQ-MC-007: The motor controller shall continue transmitting status messages every 1 ms for 100 ms after shutdown.
"""

# ==========================================================
# Helper Functions
# ==========================================================

def load_saved_state():
    """Restore last state from disk."""

    if not os.path.exists(STATE_FILE):
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved_state = json.load(f)

        st.session_state.requirements_text = saved_state.get(
            "requirements_text", ""
        )

        st.session_state.analysis_result = saved_state.get(
            "analysis_result"
        )

        st.session_state.log_file = saved_state.get(
            "log_file"
        )

    except Exception as e:
        st.warning(f"Unable to load saved state: {e}")


def save_state():
    """Save current state to disk."""

    try:
        state = {
            "requirements_text":
                st.session_state.requirements_text,

            "analysis_result":
                st.session_state.analysis_result,

            "log_file":
                st.session_state.log_file
        }

        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    except Exception as e:
        st.warning(f"Unable to save state: {e}")


def clear_saved_state():
    """Reset memory and file."""

    st.session_state.requirements_text = ""
    st.session_state.analysis_result = None
    st.session_state.log_file = None

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# ==========================================================
# Session State Initialization
# ==========================================================

if "initialized" not in st.session_state:

    st.session_state.initialized = True

    st.session_state.requirements_text = ""
    st.session_state.analysis_result = None
    st.session_state.log_file = None

    load_saved_state()

# ==========================================================
# UI
# ==========================================================

st.title("Requirements Analyzer")

st.write(
    "Analyze requirements for conflicts, ambiguity, "
    "performance constraints, consistency, and response-time issues."
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Load Sample Requirements"):
        st.session_state.requirements_text = SAMPLE_REQUIREMENTS

with col2:
    if st.button("Clear Saved State"):
        clear_saved_state()
        st.rerun()

threshold = st.slider(
    "Satisfaction Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.05
)

requirements_text = st.text_area(
    "Enter Requirements (one per line)",
    key="requirements_text",
    height=300
)

# ==========================================================
# Analyze Button
# ==========================================================

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

        log_file = (
            f"logs/conversation_{conversation_id}.md"
        )

        with open(log_file, "w", encoding="utf-8") as f:

            f.write("# Requirement Analysis\n\n")

            f.write("## Requirements\n\n")

            for req in requirements:
                f.write(f"- {req}\n")

            f.write(
                f"\n## Satisfaction Score\n\n"
                f"{result['satisfaction']}\n"
            )

            for category, issues in result["issues"].items():

                f.write(f"\n## {category}\n\n")

                for issue in issues:
                    f.write(f"- {issue}\n")

        st.session_state.analysis_result = result
        st.session_state.log_file = log_file

        save_state()

    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ==========================================================
# Results Display
# ==========================================================

if st.session_state.analysis_result:

    result = st.session_state.analysis_result

    st.markdown("---")
    st.subheader("Analysis Results")

    st.metric(
        "Satisfaction Score",
        f"{result['satisfaction']:.2f}"
    )

    if result["satisfaction"] >= threshold:
        st.success(
            f"Requirements meet the threshold ({threshold:.2f})."
        )
    else:
        st.warning(
            f"Requirements are below the threshold ({threshold:.2f})."
        )

    for category, issues in result["issues"].items():

        st.markdown(f"### {category}")

        if issues:

            for issue in issues:
                st.write(f"• {issue}")

        else:
            st.write("No issues detected.")

# ==========================================================
# Report Display & Download
# ==========================================================

if (
    st.session_state.log_file
    and os.path.exists(st.session_state.log_file)
):

    st.markdown("---")
    st.subheader("Generated Report")

    with open(
        st.session_state.log_file,
        "r",
        encoding="utf-8"
    ) as f:

        report_content = f.read()

    st.markdown(report_content)

    with open(
        st.session_state.log_file,
        "rb"
    ) as f:

        st.download_button(
            label="Download Report",
            data=f.read(),
            file_name=os.path.basename(
                st.session_state.log_file
            ),
            mime="text/markdown"
        )