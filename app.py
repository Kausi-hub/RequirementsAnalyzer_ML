"""Main Streamlit entry point for AI-Assisted Requirement Analysis Tool."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from core.analyzer import ConflictAnalyzer
from core.ears_checker import EARSChecker
from core.graph_rag import GraphRAGEngine


load_dotenv()


def read_prompt(path: Path) -> str:
    """Loads a prompt template from disk."""

    return path.read_text(encoding="utf-8")


@st.cache_resource
def init_services() -> Dict[str, object]:
    """Initializes core app services once per Streamlit process."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    input_folder = os.getenv("INPUT_FOLDER_PATH", "input").strip()
    allow_fallback = os.getenv("ALLOW_LOCAL_EMBEDDING_FALLBACK", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")

    prompt_dir = Path("prompts")
    engine = GraphRAGEngine(
        api_key=api_key,
        model=model,
        input_folder_path=input_folder,
        prompts_path=str(prompt_dir),
        allow_local_embedding_fallback=allow_fallback,
    )
    conflict_analyzer = ConflictAnalyzer(engine, api_key=api_key, model=model)
    ears_checker = EARSChecker(api_key=api_key, model=model)

    prompts = {
        "conflict": read_prompt(prompt_dir / "conflict_system.txt"),
        "ears": read_prompt(prompt_dir / "ears_system.txt"),
        "qna": read_prompt(prompt_dir / "qna_system.txt"),
    }

    return {
        "engine": engine,
        "conflict_analyzer": conflict_analyzer,
        "ears_checker": ears_checker,
        "prompts": prompts,
        "input_folder": Path(input_folder),
    }


def render_mermaid(mermaid_code: str, height: int = 520) -> None:
    """Renders Mermaid graph inside Streamlit using Mermaid.js."""

    html = f"""
    <div class=\"mermaid\">\n{mermaid_code}\n</div>
    <script type=\"module\">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


def store_requirement(input_folder: Path, text: str) -> tuple[Path, bool]:
    """Persists requirement content in input folder using content hash dedup."""

    input_folder.mkdir(parents=True, exist_ok=True)
    clean_text = text.strip()
    content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:12]
    file_path = input_folder / f"requirement_{content_hash}.md"

    if file_path.exists():
        return file_path, False

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = f"<!-- stored_at: {stamp} -->\n{clean_text}\n"
    file_path.write_text(payload, encoding="utf-8")
    return file_path, True


def render_conflicts(conflicts: List[Dict[str, str]]) -> None:
    """Renders conflict results in table-like cards."""

    if not conflicts:
        st.success("No conflicts detected against the indexed requirement pool.")
        return

    st.error(f"Detected {len(conflicts)} potential conflict(s).")
    for idx, conflict in enumerate(conflicts, start=1):
        with st.container(border=True):
            st.markdown(f"**Conflict {idx}**")
            st.write(f"Requirement ID: {conflict.get('conflicting_requirement_id', 'unknown')}")
            st.write(f"Severity: {conflict.get('severity', 'unknown')}")
            st.write(f"Reason: {conflict.get('reason', 'No reason provided')}" )
            st.write(f"Citation: {conflict.get('citation', 'No citation provided')}" )


def main() -> None:
    """Runs the Streamlit application."""

    st.set_page_config(page_title="AI Requirement Analyzer (GraphRAG)", layout="wide")
    st.title("AI-Assisted Requirement Analysis Tool")
    st.caption("GraphRAG + FAISS + OpenAI for conflict detection, EARS validation, and Q&A")

    try:
        services = init_services()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    engine: GraphRAGEngine = services["engine"]  # type: ignore[assignment]
    conflict_analyzer: ConflictAnalyzer = services["conflict_analyzer"]  # type: ignore[assignment]
    ears_checker: EARSChecker = services["ears_checker"]  # type: ignore[assignment]
    prompts: Dict[str, str] = services["prompts"]  # type: ignore[assignment]
    input_folder: Path = services["input_folder"]  # type: ignore[assignment]

    st.sidebar.header("Repository Overview")
    st.sidebar.write(f"Indexed requirements: {len(engine.records)}")
    st.sidebar.write(f"Input folder: {input_folder}")

    if "last_requirement" not in st.session_state:
        st.session_state.last_requirement = ""
    if "last_conflict_result" not in st.session_state:
        st.session_state.last_conflict_result = None
    if "last_ears_result" not in st.session_state:
        st.session_state.last_ears_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tabs = st.tabs(
        [
            "New Requirement",
            "Conflict Report",
            "EARS Validation",
            "Requirement Q&A",
            "Relational Graph",
        ]
    )

    with tabs[0]:
        st.subheader("Submit New Requirement")
        uploaded_file = st.file_uploader(
            "Upload a requirement file (.txt or .md)",
            type=["txt", "md"],
        )
        manual_text = st.text_area("Or paste requirement text", height=180)

        new_requirement = ""
        source_label = "manual"

        if uploaded_file is not None:
            try:
                new_requirement = uploaded_file.read().decode("utf-8")
                source_label = uploaded_file.name
            except UnicodeDecodeError:
                st.error("Unable to decode uploaded file. Please use UTF-8 encoded text.")

        if manual_text.strip():
            new_requirement = manual_text.strip()
            source_label = "manual_input"

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Analyze Requirement", type="primary"):
                if not new_requirement.strip():
                    st.warning("Provide a requirement via upload or text input first.")
                else:
                    with st.spinner("Running conflict analysis and EARS validation..."):
                        conflict_result = conflict_analyzer.analyze(
                            new_requirement,
                            prompts["conflict"],
                        )
                        ears_result = ears_checker.validate(new_requirement, prompts["ears"])

                    st.session_state.last_requirement = new_requirement
                    st.session_state.last_conflict_result = conflict_result
                    st.session_state.last_ears_result = ears_result
                    st.success("Analysis complete. Check the other tabs for details.")

        with col_b:
            if st.button("Add Requirement to Knowledge Base"):
                if not new_requirement.strip():
                    st.warning("Nothing to add. Provide a requirement first.")
                else:
                    try:
                        result = engine.add_requirements_from_text(
                            new_requirement,
                            source=source_label,
                        )

                        if result["added_count"] == 0:
                            st.info(
                                "No new requirement was added. Uploaded content is already indexed "
                                "or failed to embed."
                            )
                        else:
                            file_path, created = store_requirement(input_folder, new_requirement)
                            if created:
                                st.success(
                                    "Added "
                                    f"{result['added_count']} requirement(s) "
                                    f"from {result['parsed_count']} parsed and stored "
                                    f"snapshot in {file_path.name}."
                                )
                            else:
                                st.success(
                                    "Added "
                                    f"{result['added_count']} requirement(s). "
                                    f"Snapshot already exists as {file_path.name}."
                                )

                        st.write(
                            "Ingestion summary: "
                            f"parsed={result['parsed_count']}, "
                            f"added={result['added_count']}, "
                            f"duplicates={result['duplicate_count']}, "
                            f"failed={result.get('failed_count', 0)}"
                        )

                        if result.get("errors"):
                            with st.expander("Ingestion errors"):
                                for err in result["errors"]:
                                    st.write(f"- {err}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to add requirement: {exc}")
                    else:
                        st.rerun()

    with tabs[1]:
        st.subheader("Conflict Report Dashboard")
        conflict_result = st.session_state.last_conflict_result
        if not conflict_result:
            st.info("Analyze a requirement in the New Requirement tab to view conflicts.")
        else:
            st.write(conflict_result.get("summary", "Conflict analysis summary unavailable."))
            render_conflicts(conflict_result.get("conflicts", []))
            with st.expander("Retrieved Context"):
                st.json(conflict_result.get("retrieved_context", []))

    with tabs[2]:
        st.subheader("EARS Validation")
        ears_result = st.session_state.last_ears_result
        if not ears_result:
            st.info("Analyze a requirement in the New Requirement tab to view EARS results.")
        else:
            if ears_result.get("is_valid"):
                st.success("Requirement matches EARS structure.")
            else:
                st.warning("Requirement is not EARS-compliant.")

            st.write(f"Matched pattern: {ears_result.get('matched_pattern', 'None')}")
            st.write("Issues:")
            for issue in ears_result.get("issues", []):
                st.write(f"- {issue}")

            st.text_area(
                "Suggested EARS-Compliant Rewrite",
                value=ears_result.get("suggestion", ""),
                height=150,
            )

    with tabs[3]:
        st.subheader("Interactive Requirement Q&A")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the current requirement pool...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving graph context and generating answer..."):
                    answer = engine.answer_question(question, prompts["qna"])
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with tabs[4]:
        st.subheader("Relational Requirement Graph (Mermaid)")
        query_focus = st.text_input("Optional focus query for graph filtering")
        mermaid_code = engine.build_mermaid_graph(
            focus_query=query_focus.strip() if query_focus.strip() else None,
            top_k=8,
        )
        st.code(mermaid_code, language="mermaid")
        render_mermaid(mermaid_code)


if __name__ == "__main__":
    main()
