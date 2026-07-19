# AI-Assisted Requirement Analysis Tool (GraphRAG)

This project is a production-oriented, modular Python application for requirement engineering workflows using GraphRAG (Graph Retrieval-Augmented Generation).

It provides:
- Requirement ingestion from a local `input/` directory (`.txt` and `.md`).
- Conflict detection against existing requirement context.
- EARS compliance validation with suggested rewrites.
- Natural language Q&A over the requirement pool.
- Dynamic relational graph visualization via Mermaid.js in Streamlit.

## Tech Stack
- UI: Streamlit
- Graph rendering: Mermaid.js (embedded in Streamlit)
- LLM + embeddings: OpenAI API
- Vector database: FAISS (local, open-source)
- Config: `.env` via `python-dotenv`

All dependencies are open-source except the OpenAI API endpoint, as required.

## Project Structure

```text
.
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── core/
│   ├── __init__.py
│   ├── graph_rag.py
│   ├── ears_checker.py
│   └── analyzer.py
└── prompts/
    ├── __init__.py
    ├── conflict_system.txt
    ├── ears_system.txt
    ├── entity_system.txt
    └── qna_system.txt
```

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create/update `.env` with:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
INPUT_FOLDER_PATH=input
```

## Usage

1. Ensure your requirement files exist in `input/` (optional at first run).
2. Launch the app:

```bash
streamlit run app.py
```

3. In the UI:
- Use **New Requirement** to upload or paste a new requirement.
- Run **Analyze Requirement** to generate conflict and EARS outputs.
- Use **Add Requirement to Knowledge Base** to persist and index the requirement.
- Review **Conflict Report** and **EARS Validation** tabs.
- Ask contextual questions in **Requirement Q&A**.
- Inspect graph topology in **Relational Graph**.

## GraphRAG Technical Summary

### Ingestion and Indexing
- On startup, the engine scans `input/` for `.txt` and `.md` files.
- It splits file content into requirement statements and deduplicates by text.
- For each requirement, it performs entity extraction (`systems`, `users`, `actions`, `constraints`, `relationships`).
- Requirement text embeddings are generated with OpenAI and normalized.
- Embeddings are indexed in local FAISS (`IndexFlatIP`) for cosine-style retrieval.

### Graph Construction
- Every requirement is represented as a requirement node.
- Extracted entities are represented as typed nodes.
- `contains` edges connect requirement nodes to entity nodes.
- Extracted inter-entity relationships are added as labeled edges.

### Hybrid Retrieval
- Query embedding -> FAISS nearest requirements.
- Graph neighbors/edges around retrieved requirement nodes are collected.
- Combined contextual payload is passed to LLM for grounded generation.

### Prompt Architecture
- System prompts are externalized in `prompts/` files.
- Every AI call uses strict separation:
  - System prompt loaded from file.
  - Runtime user prompt assembled with context placeholders/content.

### Reliability
- OpenAI chat and embedding calls handle timeout/network/API exceptions.
- File reading/writing uses guarded behavior and UTF-8 handling.
- Index and metadata are persisted locally under `vector_store/`.

## Security
- No API keys are hardcoded.
- `.env` is ignored by Git.
- Local requirement files in `input/` are ignored by Git.

## Notes
- If the OpenAI API is unavailable, conflict/EARS checks return fallback-safe outputs.
- You can clear local state by removing `vector_store/` and restarting.
