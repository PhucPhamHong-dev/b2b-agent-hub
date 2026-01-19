from __future__ import annotations

"""Lightweight markdown knowledge store with chunking and keyword retrieval."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils import normalize_text


class KnowledgeStore:
    """Manage core/delta knowledge markdown and retrieve relevant chunks."""

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        self._base_dir = Path(__file__).resolve().parents[2]
        self._knowledge_dir = knowledge_dir or (self._base_dir / "knowledge")
        self._core_path = self._knowledge_dir / "knowledge_core.md"
        self._delta_path = self._knowledge_dir / "knowledge_delta.md"
        self._index_path = self._knowledge_dir / "md_index.json"
        self._index_cache: Optional[Dict[str, object]] = None
        self._index_mtime: Tuple[float, float] = (0.0, 0.0)

    def load_core_delta(self) -> Tuple[str, str]:
        """Purpose: Load core and delta markdown, creating templates if missing.
        Inputs/Outputs: No inputs; returns tuple(core_text, delta_text).
        Side Effects / State: Ensures knowledge directory and files exist.
        Dependencies: Uses filesystem paths under repo root.
        Failure Modes: File read/write errors propagate to caller.
        If Removed: Retrieval cannot access knowledge sources and returns empty.
        Testing Notes: Delete files and verify they are recreated with defaults.
        """
        self._ensure_files()
        core_text = self._core_path.read_text(encoding="utf-8")
        delta_text = self._delta_path.read_text(encoding="utf-8")
        return core_text, delta_text

    def build_or_load_index(self) -> Dict[str, object]:
        """Purpose: Build or load the knowledge index with chunk metadata.
        Inputs/Outputs: No inputs; returns an index dict with chunks and mtimes.
        Side Effects / State: Writes md_index.json when rebuilding.
        Dependencies: Uses chunk_markdown and load_core_delta.
        Failure Modes: JSON decode errors trigger a rebuild.
        If Removed: retrieve_topk must parse markdown on every call.
        Testing Notes: Touch core/delta and confirm index rebuilds.
        """
        core_text, delta_text = self.load_core_delta()
        core_mtime = self._core_path.stat().st_mtime
        delta_mtime = self._delta_path.stat().st_mtime

        if self._index_cache and self._index_mtime == (core_mtime, delta_mtime):
            return self._index_cache

        if self._index_path.exists():
            try:
                cached = json.loads(self._index_path.read_text(encoding="utf-8"))
                if (
                    isinstance(cached, dict)
                    and cached.get("core_mtime") == core_mtime
                    and cached.get("delta_mtime") == delta_mtime
                ):
                    self._index_cache = cached
                    self._index_mtime = (core_mtime, delta_mtime)
                    return cached
            except json.JSONDecodeError:
                pass

        chunks = []
        chunks.extend(self.chunk_markdown(core_text, source="core"))
        chunks.extend(self.chunk_markdown(delta_text, source="delta"))
        index = {
            "core_mtime": core_mtime,
            "delta_mtime": delta_mtime,
            "chunks": chunks,
        }

        self._write_index(index)
        self._index_cache = index
        self._index_mtime = (core_mtime, delta_mtime)
        return index

    def retrieve_topk(self, query: str, topk: int = 6) -> List[str]:
        """Purpose: Retrieve top-K knowledge chunks relevant to a query.
        Inputs/Outputs: Input is query string and topk; output is list of chunk texts.
        Side Effects / State: Loads or rebuilds the index as needed.
        Dependencies: Uses normalize_text and build_or_load_index.
        Failure Modes: Empty query or index returns empty list.
        If Removed: LLM prompts cannot be enriched with prior knowledge.
        Testing Notes: Query with known synonyms and verify matching chunks.
        """
        if not query or topk <= 0:
            return []
        if os.getenv("KNOWLEDGE_ENABLED", "1") == "0":
            return []

        index = self.build_or_load_index()
        chunks = index.get("chunks", []) if isinstance(index, dict) else []
        if not chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for chunk in chunks:
            content = chunk.get("content", "")
            title = chunk.get("title", "")
            section = chunk.get("section", "")
            source = chunk.get("source", "")
            score = _score_chunk(query_tokens, content, title, section)
            if source == "delta":
                score += 0.1
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [_format_chunk(chunk) for _, chunk in scored[:topk]]

    def chunk_markdown(self, md_text: str, source: str) -> List[Dict[str, str]]:
        """Purpose: Split markdown text into chunks based on headings.
        Inputs/Outputs: Inputs are markdown text and source label; output is chunk dicts.
        Side Effects / State: None.
        Dependencies: Uses _split_long_content for long sections.
        Failure Modes: Empty input returns empty list.
        If Removed: Retrieval cannot target specific knowledge sections.
        Testing Notes: Validate chunks from both ## and ### headings.
        """
        if not md_text:
            return []

        lines = md_text.splitlines()
        chunks: List[Dict[str, str]] = []
        section = ""
        title = ""
        buffer: List[str] = []

        def flush() -> None:
            nonlocal buffer
            content = "\n".join(buffer).strip()
            buffer = []
            if not content:
                return
            parts = _split_long_content(content)
            for part in parts:
                chunk_id = f"{source}-{len(chunks)}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "section": section,
                        "title": title or section,
                        "content": part,
                        "source": source,
                    }
                )

        for line in lines:
            if line.startswith("## "):
                flush()
                section = line[3:].strip()
                title = section
                continue
            if line.startswith("### "):
                flush()
                title = line[4:].strip()
                continue
            buffer.append(line)

        flush()
        return chunks

    def _ensure_files(self) -> None:
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        if not self._core_path.exists():
            self._core_path.write_text(_DEFAULT_CORE, encoding="utf-8")
        if not self._delta_path.exists():
            self._delta_path.write_text(_DEFAULT_DELTA, encoding="utf-8")
        if not self._index_path.exists():
            self._index_path.write_text(json.dumps(_EMPTY_INDEX), encoding="utf-8")

    def _write_index(self, index: Dict[str, object]) -> None:
        tmp_path = self._index_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(index, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(self._index_path)


def _tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [token for token in normalized.split() if token]


def _score_chunk(tokens: List[str], content: str, title: str, section: str) -> float:
    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0
    content_counts: Dict[str, int] = {}
    for token in content_tokens:
        content_counts[token] = content_counts.get(token, 0) + 1

    title_tokens = set(_tokenize(title))
    section_tokens = set(_tokenize(section))

    score = 0.0
    for token in tokens:
        score += content_counts.get(token, 0)
        if token in title_tokens:
            score += 2.0
        if token in section_tokens:
            score += 1.0
    return score


def _format_chunk(chunk: Dict[str, str]) -> str:
    section = chunk.get("section", "")
    title = chunk.get("title", "")
    source = chunk.get("source", "")
    header_parts = [part for part in (section, title) if part]
    header = " / ".join(dict.fromkeys(header_parts))
    prefix = f"[{source.upper()}] {header}".strip()
    content = chunk.get("content", "")
    return f"{prefix}\n{content}".strip()


def _split_long_content(content: str, max_words: int = 400) -> List[str]:
    words = content.split()
    if len(words) <= max_words:
        return [content]
    parts = []
    for idx in range(0, len(words), max_words):
        parts.append(" ".join(words[idx : idx + max_words]))
    return parts


_DEFAULT_CORE = (
    "# Knowledge Core\n\n"
    "## Purpose\n"
    "- [2026-01-16][RULE][high] Use the catalog as the only source of product facts (SKU/specs/images).\n"
    "- [2026-01-16][RULE][high] For technical intents, avoid internal handoff phrases unless a contact form is requested.\n\n"
    "## Synonyms\n"
    "- [2026-01-16][SYN][medium] \"than giu bec\" => TIP_BODY\n"
    "- [2026-01-16][SYN][medium] \"cach dien\" => INSULATOR\n"
    "- [2026-01-16][SYN][medium] \"chup khi\" => NOZZLE\n"
    "- [2026-01-16][SYN][medium] \"su phan phoi khi\" => ORIFICE\n"
)

_DEFAULT_DELTA = "# Knowledge Delta\n\n## CHANGELOG (APPEND ONLY)\n"

_EMPTY_INDEX = {"core_mtime": 0, "delta_mtime": 0, "chunks": []}
