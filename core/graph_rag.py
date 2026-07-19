"""GraphRAG engine with OpenAI embeddings, FAISS retrieval, and graph context."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


@dataclass
class RequirementRecord:
    """Represents one requirement and its extracted graph context."""

    req_id: str
    source: str
    text: str
    text_hash: str
    source_req_id: Optional[str]
    entities: Dict[str, List[str]]
    relationships: List[Dict[str, str]]


class GraphRAGEngine:
    """Builds and queries a hybrid graph + vector retrieval system."""

    def __init__(
        self,
        api_key: str,
        model: str,
        input_folder_path: str,
        prompts_path: str,
        embedding_model: str = "text-embedding-3-small",
        storage_path: str = "vector_store",
        allow_local_embedding_fallback: bool = True,
    ) -> None:
        self.client = OpenAI(api_key=api_key, timeout=30.0)
        self.model = model
        self.embedding_model = embedding_model
        self.input_folder = Path(input_folder_path)
        self.prompts_path = Path(prompts_path)
        self.storage_path = Path(storage_path)
        self.allow_local_embedding_fallback = allow_local_embedding_fallback

        self.records: List[RequirementRecord] = []
        self.record_id_to_pos: Dict[str, int] = {}
        self.record_ids: List[str] = []
        self.graph_nodes: Dict[str, str] = {}
        self.graph_edges: List[Dict[str, str]] = []
        self.record_embeddings: List[List[float]] = []
        self.text_hashes: set[str] = set()
        self.embedding_dim: Optional[int] = None
        self.index: Optional[faiss.IndexFlatIP] = None

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.input_folder.mkdir(parents=True, exist_ok=True)

        self._load_persisted_state()
        self._ingest_input_folder()
        # Ensure local KB manifest exists even if no requirements were added yet.
        if not (self.storage_path / "records.json").exists():
            self._persist_state()

    def _read_prompt(self, file_name: str) -> str:
        path = self.prompts_path / file_name
        return path.read_text(encoding="utf-8")

    def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        return self._extract_json_object(content)

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def _embed_text(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text],
            )
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            if self.allow_local_embedding_fallback:
                fallback_dim = self.index.d if self.index is not None else (self.embedding_dim or 256)
                return self._fallback_embed_text(text, dims=fallback_dim)
            raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

        vector = np.array(response.data[0].embedding, dtype=np.float32)

        if self.index is not None and vector.shape[0] != self.index.d:
            if self.allow_local_embedding_fallback:
                return self._fallback_embed_text(text, dims=self.index.d)
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"got {vector.shape[0]}, index expects {self.index.d}."
            )

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def _fallback_embed_text(text: str, dims: int = 256) -> np.ndarray:
        """Build a deterministic local embedding when remote embedding is unavailable."""

        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if not tokens:
            tokens = ["empty"]

        vector = np.zeros(dims, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(8):
                start = i * 4
                value = int.from_bytes(digest[start:start + 4], byteorder="big", signed=False)
                idx = value % dims
                sign = 1.0 if ((value >> 1) & 1) == 1 else -1.0
                vector[idx] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _ensure_index(self, embedding_dim: int) -> None:
        if self.index is None:
            self.embedding_dim = embedding_dim
            self.index = faiss.IndexFlatIP(embedding_dim)

    def _heuristic_entity_fallback(self, text: str) -> Dict[str, List[str]]:
        systems = re.findall(r"\bThe\s+([A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)*)\b", text)
        users = re.findall(r"\b(user|operator|admin|client|customer|driver|passenger)s?\b", text, flags=re.IGNORECASE)
        actions = re.findall(r"\bshall\s+([a-z][^.,;]*)", text, flags=re.IGNORECASE)
        constraints = re.findall(
            r"\b(within\s+[^.,;]+|at least\s+[^.,;]+|at most\s+[^.,;]+|no later than\s+[^.,;]+)",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "systems": sorted(set(systems)),
            "users": sorted({u.lower() for u in users}),
            "actions": sorted({a.strip() for a in actions}),
            "constraints": sorted({c.strip() for c in constraints}),
            "relationships": [],
        }

    @staticmethod
    def _normalize_requirement_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip())
        return normalized

    def _text_hash(self, text: str) -> str:
        normalized = self._normalize_requirement_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def extract_entities(self, requirement_text: str) -> Dict[str, Any]:
        system_prompt = self._read_prompt("entity_system.txt").replace(
            "{{new_requirement}}", requirement_text
        )
        user_prompt = (
            "Extract entities from this requirement and return strict JSON.\n\n"
            f"Requirement:\n{requirement_text}"
        )

        try:
            extracted = self._chat_json(system_prompt, user_prompt)
        except RuntimeError:
            return self._heuristic_entity_fallback(requirement_text)

        required_keys = ["systems", "users", "actions", "constraints", "relationships"]
        if not all(key in extracted for key in required_keys):
            return self._heuristic_entity_fallback(requirement_text)
        return extracted

    def _register_graph_for_requirement(self, record: RequirementRecord) -> None:
        req_node = f"{record.req_id}|Requirement"
        self.graph_nodes[req_node] = "Requirement"

        for entity_type in ["systems", "users", "actions", "constraints"]:
            for value in record.entities.get(entity_type, []):
                entity = value.strip()
                if not entity:
                    continue
                node_id = f"{entity}|{entity_type[:-1].capitalize()}"
                self.graph_nodes[node_id] = entity_type[:-1].capitalize()
                self.graph_edges.append(
                    {
                        "source": req_node,
                        "target": node_id,
                        "relation": "contains",
                    }
                )

        for rel in record.relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "related_to").strip() or "related_to"
            if source and target:
                s_node = f"{source}|Entity"
                t_node = f"{target}|Entity"
                self.graph_nodes[s_node] = "Entity"
                self.graph_nodes[t_node] = "Entity"
                self.graph_edges.append(
                    {
                        "source": s_node,
                        "target": t_node,
                        "relation": relation,
                    }
                )

    @staticmethod
    def _split_requirements(file_text: str) -> List[str]:
        chunks = [line.strip() for line in file_text.splitlines() if line.strip()]
        return [chunk for chunk in chunks if len(chunk) > 10]

    @staticmethod
    def _extract_markdown_table_requirements(content: str) -> List[Tuple[Optional[str], str]]:
        requirements: List[Tuple[Optional[str], str]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue
            if ":---" in line or "Req ID" in line:
                continue

            cells = [part.strip() for part in line.strip("|").split("|")]
            if len(cells) < 3:
                continue

            req_id_cell = re.sub(r"\*\*", "", cells[0]).strip()
            req_id = req_id_cell if re.match(r"^REQ-[A-Za-z0-9-]+$", req_id_cell) else None
            requirement_text = cells[2]
            requirement_text = re.sub(r"\*\*", "", requirement_text).strip()

            if not requirement_text:
                continue

            if len(requirement_text) < 25:
                continue

            if not re.search(r"\bshall\b", requirement_text, flags=re.IGNORECASE):
                continue

            requirements.append((req_id, requirement_text))

        return requirements

    def _extract_requirements(self, content: str) -> List[Tuple[Optional[str], str]]:
        markdown_rows = self._extract_markdown_table_requirements(content)
        if markdown_rows:
            return markdown_rows

        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", content)
            if p.strip()
        ]
        candidates = []
        for para in paragraphs:
            if re.search(r"\bshall\b", para, flags=re.IGNORECASE) and len(para) >= 25:
                candidates.append((None, self._normalize_requirement_text(para)))

        if candidates:
            return candidates

        return [(None, self._normalize_requirement_text(chunk)) for chunk in self._split_requirements(content)]

    def _next_requirement_id(self) -> str:
        return f"REQ-{len(self.records) + 1:04d}"

    def add_requirement(
        self,
        text: str,
        source: str = "user",
        source_req_id: Optional[str] = None,
    ) -> Optional[RequirementRecord]:
        clean_text = self._normalize_requirement_text(text)
        if not clean_text:
            raise ValueError("Requirement text cannot be empty.")

        text_hash = self._text_hash(clean_text)
        if text_hash in self.text_hashes:
            return None

        entities = self.extract_entities(clean_text)
        record = RequirementRecord(
            req_id=self._next_requirement_id(),
            source=source,
            text=clean_text,
            text_hash=text_hash,
            source_req_id=source_req_id,
            entities={
                "systems": entities.get("systems", []),
                "users": entities.get("users", []),
                "actions": entities.get("actions", []),
                "constraints": entities.get("constraints", []),
            },
            relationships=entities.get("relationships", []),
        )

        embedding = self._embed_text(clean_text)

        if self.index is not None and embedding.shape[0] != self.index.d:
            if self.allow_local_embedding_fallback:
                embedding = self._fallback_embed_text(clean_text, dims=self.index.d)
            else:
                raise RuntimeError(
                    "Embedding dimension mismatch on add: "
                    f"got {embedding.shape[0]}, index expects {self.index.d}."
                )

        self._ensure_index(embedding.shape[0])
        assert self.index is not None

        self.index.add(np.expand_dims(embedding, axis=0))
        self.record_id_to_pos[record.req_id] = len(self.records)
        self.record_ids.append(record.req_id)
        self.records.append(record)
        self.record_embeddings.append(embedding.tolist())
        self.text_hashes.add(text_hash)
        self._register_graph_for_requirement(record)
        self._persist_state()

        return record

    def add_requirements_from_text(self, content: str, source: str = "user") -> Dict[str, Any]:
        parsed_requirements = self._extract_requirements(content)
        added_records: List[RequirementRecord] = []
        duplicate_count = 0
        failed_count = 0
        errors: List[str] = []

        for index, (source_req_id, req_text) in enumerate(parsed_requirements, start=1):
            try:
                record = self.add_requirement(
                    req_text,
                    source=source,
                    source_req_id=source_req_id,
                )
                if record is None:
                    duplicate_count += 1
                    continue
                added_records.append(record)
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                errors.append(f"item {index}: {exc}")

        return {
            "parsed_count": len(parsed_requirements),
            "added_count": len(added_records),
            "duplicate_count": duplicate_count,
            "failed_count": failed_count,
            "errors": errors,
            "records": added_records,
        }

    def _ingest_input_folder(self) -> None:
        for file_path in sorted(self.input_folder.glob("*")):
            if file_path.suffix.lower() not in {".txt", ".md"}:
                continue
            if file_path.name.startswith("."):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError:
                continue

            self.add_requirements_from_text(content, source=file_path.name)

    def _persist_state(self) -> None:
        meta_path = self.storage_path / "records.json"
        index_path = self.storage_path / "faiss.index"

        payload = {
            "records": [asdict(record) for record in self.records],
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "record_ids": self.record_ids,
            "record_embeddings": self.record_embeddings,
            "embedding_dim": self.embedding_dim,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if self.index is not None:
            faiss.write_index(self.index, str(index_path))

    def _load_persisted_state(self) -> None:
        meta_path = self.storage_path / "records.json"
        index_path = self.storage_path / "faiss.index"

        if not meta_path.exists():
            return

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        loaded_records = []
        for record in payload.get("records", []):
            if "text_hash" not in record:
                record["text_hash"] = self._text_hash(record.get("text", ""))
            loaded_records.append(RequirementRecord(**record))

        self.records = loaded_records
        self.graph_nodes = payload.get("graph_nodes", {})
        self.graph_edges = payload.get("graph_edges", [])
        self.record_ids = payload.get("record_ids", [record.req_id for record in self.records])
        self.record_embeddings = payload.get("record_embeddings", [])
        self.embedding_dim = payload.get("embedding_dim")
        self.record_id_to_pos = {record.req_id: idx for idx, record in enumerate(self.records)}
        self.text_hashes = {record.text_hash for record in self.records}

        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
            except RuntimeError:
                self.index = None

        if self.index is not None and not self.record_embeddings:
            try:
                total = self.index.ntotal
                self.record_embeddings = [
                    self.index.reconstruct(i).astype(np.float32).tolist()
                    for i in range(total)
                ]
            except RuntimeError:
                self.record_embeddings = []

        # Backfill source requirement IDs for old records if they are embedded in text.
        for record in self.records:
            if record.source_req_id:
                continue
            match = re.search(r"\b(REQ-[A-Za-z0-9-]+)\b", record.text)
            if match:
                record.source_req_id = match.group(1)

        # Rebuild FAISS from persisted embeddings when index file is missing/corrupt.
        if self.index is None and self.record_embeddings:
            vectors = np.array(self.record_embeddings, dtype=np.float32)
            if vectors.ndim == 2 and vectors.shape[1] > 0:
                self.embedding_dim = vectors.shape[1]
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(vectors)

        # Last-resort rebuild: if records exist but no index/embeddings, re-embed records.
        if self.index is None and self.records and not self.record_embeddings:
            rebuilt_vectors: List[np.ndarray] = []
            for record in self.records:
                try:
                    vector = self._embed_text(record.text)
                except RuntimeError:
                    rebuilt_vectors = []
                    break
                rebuilt_vectors.append(vector)

            if rebuilt_vectors:
                matrix = np.vstack(rebuilt_vectors)
                self.embedding_dim = matrix.shape[1]
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(matrix)
                self.record_embeddings = [vec.astype(np.float32).tolist() for vec in rebuilt_vectors]
                self._persist_state()

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        text_terms = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
        if not query_terms or not text_terms:
            return 0.0
        return float(len(query_terms & text_terms)) / float(len(query_terms))

    def _retrieve_lexical(self, query: str, top_k: int) -> Dict[str, Any]:
        scored = sorted(
            self.records,
            key=lambda rec: self._lexical_overlap_score(query, rec.text),
            reverse=True,
        )
        selected_records = [rec for rec in scored[:top_k] if self._lexical_overlap_score(query, rec.text) > 0]

        selected_req_nodes = {f"{record.req_id}|Requirement" for record in selected_records}
        selected_edges = [
            edge
            for edge in self.graph_edges
            if edge["source"] in selected_req_nodes or edge["target"] in selected_req_nodes
        ]

        neighbor_nodes = set()
        for edge in selected_edges:
            neighbor_nodes.add(edge["source"])
            neighbor_nodes.add(edge["target"])

        context_blocks = []
        for record in selected_records:
            source_req = record.source_req_id or record.req_id
            context_blocks.append(
                f"[{source_req}] from {record.source}: {record.text}"
            )

        context_text = "\n".join(context_blocks) if context_blocks else "No relevant requirements found."

        return {
            "records": selected_records,
            "neighbors": sorted(neighbor_nodes),
            "edges": selected_edges,
            "context_text": context_text,
        }

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.records:
            return {
                "records": [],
                "neighbors": [],
                "edges": [],
                "context_text": "No requirements available.",
            }

        if self.index is None:
            return self._retrieve_lexical(query, top_k)

        try:
            query_vector = self._embed_text(query)
        except RuntimeError:
            return self._retrieve_lexical(query, top_k)

        if query_vector.shape[0] != self.index.d:
            return self._retrieve_lexical(query, top_k)

        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)

        selected_records: List[RequirementRecord] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.record_ids):
                continue
            req_id = self.record_ids[idx]
            record_pos = self.record_id_to_pos.get(req_id)
            if record_pos is None:
                continue
            selected_records.append(self.records[record_pos])

        selected_req_nodes = {f"{record.req_id}|Requirement" for record in selected_records}
        selected_edges = [
            edge
            for edge in self.graph_edges
            if edge["source"] in selected_req_nodes or edge["target"] in selected_req_nodes
        ]

        neighbor_nodes = set()
        for edge in selected_edges:
            neighbor_nodes.add(edge["source"])
            neighbor_nodes.add(edge["target"])

        context_blocks = []
        for distance, record in zip(distances[0], selected_records):
            source_req = record.source_req_id or record.req_id
            context_blocks.append(
                f"[{source_req}] (score={distance:.4f}) from {record.source}: {record.text}"
            )

        context_text = "\n".join(context_blocks) if context_blocks else "No relevant requirements found."

        return {
            "records": selected_records,
            "neighbors": sorted(neighbor_nodes),
            "edges": selected_edges,
            "context_text": context_text,
        }

    def build_mermaid_graph(self, focus_query: Optional[str] = None, top_k: int = 8) -> str:
        if not self.records:
            return "flowchart LR\n    A[No requirements loaded]"

        if focus_query:
            retrieved = self.retrieve(focus_query, top_k=top_k)
            edge_list = retrieved["edges"]
            allowed_nodes = set(retrieved["neighbors"])
        else:
            edge_list = self.graph_edges
            allowed_nodes = set(self.graph_nodes.keys())

        if not edge_list:
            return "flowchart LR\n    A[Graph empty]"

        lines = ["flowchart LR"]

        for node in sorted(allowed_nodes):
            label, kind = node.split("|", maxsplit=1)
            safe_id = self._safe_node_id(node)
            lines.append(f"    {safe_id}[\"{label} ({kind})\"]")

        for edge in edge_list:
            src = self._safe_node_id(edge["source"])
            tgt = self._safe_node_id(edge["target"])
            rel = edge.get("relation", "related_to")
            lines.append(f"    {src} -->|{rel}| {tgt}")

        return "\n".join(lines)

    @staticmethod
    def _safe_node_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", value)

    def answer_question(self, question: str, qna_system_prompt: str) -> str:
        retrieved = self.retrieve(question, top_k=6)
        formatted_system_prompt = (
            qna_system_prompt.replace("{{existing_context}}", retrieved["context_text"])
            .replace("{{graph_edges}}", json.dumps(retrieved["edges"], indent=2))
            .replace("{{question}}", question)
        )
        user_prompt = (
            "Use the following GraphRAG context to answer the question.\n\n"
            f"Graph context:\n{retrieved['context_text']}\n\n"
            f"Linked edges:\n{json.dumps(retrieved['edges'], indent=2)}\n\n"
            f"Question:\n{question}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": formatted_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            return f"Unable to answer right now due to API error: {exc}"

        return response.choices[0].message.content or "No answer generated."
