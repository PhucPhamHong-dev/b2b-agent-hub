from __future__ import annotations

"""LLM-assisted knowledge updater with guardrails and append-only delta writes."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..gemini_client import GeminiClient
from ..prompt_loader import load_prompt
from ..resource_loader import ResourceItem, ResourceLoader, get_raw_value
from ..utils import normalize_text


class KnowledgeUpdater:
    """Propose and append new knowledge lines after each response."""

    def __init__(self, gemini: GeminiClient, resource_loader: ResourceLoader, prompts_dir: Path) -> None:
        self._gemini = gemini
        self._resource_loader = resource_loader
        self._prompts_dir = prompts_dir
        self._base_dir = Path(__file__).resolve().parents[2]
        self._knowledge_dir = self._base_dir / "knowledge"
        self._core_path = self._knowledge_dir / "knowledge_core.md"
        self._delta_path = self._knowledge_dir / "knowledge_delta.md"

    def update(self, context: object) -> int:
        """Purpose: Run extraction, gate, and append to delta knowledge file.
        Inputs/Outputs: Input is a context-like object; returns appended line count.
        Side Effects / State: Appends to knowledge_delta.md when entries pass gating.
        Dependencies: Uses GeminiClient, knowledge_extractor.txt, and ResourceLoader.
        Failure Modes: Returns 0 on LLM errors or gated-out entries.
        If Removed: Knowledge delta never grows and long-term memory stays static.
        Testing Notes: Simulate a chat turn and verify delta append.
        """
        if os.getenv("KNOWLEDGE_ENABLED", "1") == "0":
            return 0

        entries = self.propose_entries(context)
        if not entries:
            return 0

        catalog_items = getattr(context, "catalog_items", None)
        if not catalog_items:
            catalog_items, _ = self._resource_loader.load()

        context_text = _build_context_text(context)
        cleaned = self.memory_gate(entries, catalog_items, context_text)
        if not cleaned:
            return 0

        self.append_delta(cleaned)
        return len(cleaned)

    def propose_entries(self, context: object) -> List[str]:
        """Purpose: Ask the LLM to suggest reusable knowledge lines.
        Inputs/Outputs: Input is a context-like object; output is raw entry lines.
        Side Effects / State: Calls the Gemini model to generate text.
        Dependencies: Uses knowledge_extractor.txt and GeminiClient.
        Failure Modes: Returns empty list if prompt missing or LLM fails.
        If Removed: Delta knowledge cannot evolve with new rules.
        Testing Notes: Feed a simple context and verify bullet output format.
        """
        prompt_path = self._prompts_dir / "knowledge_extractor.txt"
        if not prompt_path.exists():
            return []

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        user_message = getattr(context, "user_message", "")
        answer_text = getattr(context, "answer_text", "")
        intent = getattr(context, "intent_label", "")
        anchor = _infer_anchor(context)
        route = getattr(context, "intent_label", "")

        prompt_template = load_prompt(prompt_path)
        prompt = (
            prompt_template.replace("<<DATE>>", date_str)
            .replace("<<USER_MESSAGE>>", str(user_message))
            .replace("<<ASSISTANT_ANSWER>>", str(answer_text))
            .replace("<<INTENT>>", str(intent))
            .replace("<<ANCHOR>>", str(anchor))
            .replace("<<ROUTE>>", str(route))
        )

        model_name = getattr(context, "model_flash", None)
        try:
            raw = self._gemini.generate_text(prompt, model=model_name, temperature=0.1)
        except Exception:
            return []

        lines = [line.strip() for line in (raw or "").splitlines() if line.strip().startswith("-")]
        max_lines = int(os.getenv("KNOWLEDGE_MAX_NEW_LINES", "5"))
        return lines[:max_lines]

    def memory_gate(
        self,
        entries: List[str],
        catalog_items: Sequence[ResourceItem],
        context_text: str,
    ) -> List[str]:
        """Purpose: Filter proposed entries with safety and catalog checks.
        Inputs/Outputs: Inputs are raw entries, catalog items, and context text;
            output is clean entries.
        Side Effects / State: Reads core/delta knowledge files for dedupe.
        Dependencies: Uses normalize_text and catalog SKU extraction.
        Failure Modes: Malformed entries are dropped silently.
        If Removed: Unsafe or duplicate lines could pollute long-term memory.
        Testing Notes: Verify blocked keywords and unknown SKUs are rejected.
        """
        if not entries:
            return []

        existing = self._read_existing()
        existing_signatures = _extract_existing_signatures(existing)
        sku_set = _collect_known_skus(catalog_items)
        cleaned: List[str] = []
        seen_signatures: set[str] = set()

        for line in entries:
            parsed = _parse_entry_line(line)
            if not parsed:
                continue
            date, tag, confidence, content = parsed
            if tag not in {"QA", "SYN", "RULE", "TEMPLATE"}:
                continue
            if confidence not in {"high", "medium", "low"}:
                continue
            tag = _auto_relabel_tag(tag, content)

            signature = _signature(content)
            if not signature:
                continue
            if _contains_blocked_terms(signature):
                continue
            if signature in existing_signatures:
                continue
            if signature in seen_signatures:
                continue
            if _mentions_specs_without_sku(signature):
                continue
            if not _is_mostly_vietnamese(content):
                continue
            if _is_generic_robot_hand_template(tag, signature):
                continue
            if _mentions_sku(content) and not _all_skus_known(content, sku_set):
                continue
            if _mentions_sku(content) and not _sku_in_context(content, context_text):
                continue
            if tag == "QA" and not _is_relevant_qa(content, context_text):
                continue

            rebuilt = f"- [{date}][{tag}][{confidence}] {content}"
            cleaned.append(rebuilt)
            seen_signatures.add(signature)
            if len(cleaned) >= int(os.getenv("KNOWLEDGE_MAX_NEW_LINES", "5")):
                break

        return cleaned

    def append_delta(self, entries: List[str]) -> None:
        """Purpose: Append gated entries to the delta knowledge file.
        Inputs/Outputs: Input is a list of entries; no return value.
        Side Effects / State: Writes knowledge_delta.md atomically.
        Dependencies: Uses filesystem paths under knowledge/.
        Failure Modes: File write errors propagate to caller.
        If Removed: Approved knowledge never persists beyond current run.
        Testing Notes: Append lines and verify they appear under CHANGELOG.
        """
        if not entries:
            return
        self._ensure_delta()
        content = self._delta_path.read_text(encoding="utf-8")
        if "## CHANGELOG (APPEND ONLY)" not in content:
            content = content.rstrip() + "\n\n## CHANGELOG (APPEND ONLY)\n"

        append_block = "\n".join(entries).strip()
        if not append_block:
            return

        new_text = content.rstrip() + "\n" + append_block + "\n"
        tmp_path = self._delta_path.with_suffix(".tmp")
        tmp_path.write_text(new_text, encoding="utf-8")
        tmp_path.replace(self._delta_path)

    def _ensure_delta(self) -> None:
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        if not self._delta_path.exists():
            self._delta_path.write_text("# Knowledge Delta\n\n## CHANGELOG (APPEND ONLY)\n", encoding="utf-8")

    def _read_existing(self) -> str:
        core_text = self._core_path.read_text(encoding="utf-8") if self._core_path.exists() else ""
        delta_text = self._delta_path.read_text(encoding="utf-8") if self._delta_path.exists() else ""
        return core_text + "\n" + delta_text


def _infer_anchor(context: object) -> str:
    items = getattr(context, "items", None) or []
    if items:
        item = items[0]
        code = getattr(item, "code", "") or ""
        name = getattr(item, "name", "") or ""
        return f"{name} {code}".strip()
    order_state = getattr(context, "order_state", {}) or {}
    return str(order_state.get("selected_sku") or "")


def _parse_entry_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    match = re.match(r"^-\s*\[(\d{4}-\d{2}-\d{2})\]\[([A-Z]+)\]\[(high|medium|low)\]\s+(.+)$", line)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4).strip()


def _contains_blocked_terms(content_norm: str) -> bool:
    blocked = [
        "bo luat",
        "ignore",
        "system prompt",
        "tiet lo",
        "agentx",
        "excel",
        "log noi bo",
        "prompt noi bo",
        "noi bo",
        "internal",
        "typically include",
        "distinguish between",
        "manual torch",
        "robot welding",
    ]
    return any(term in content_norm for term in blocked)


def _mentions_specs_without_sku(content_norm: str) -> bool:
    spec_terms = ["size", "dai", "ren", "mm", "amp", "350a", "500a"]
    if any(term in content_norm for term in spec_terms) and not re.search(r"\b\d{5,6}\b", content_norm):
        return True
    return False


def _mentions_sku(content: str) -> bool:
    return bool(re.search(r"\b\d{5,6}\b", content))


def _all_skus_known(content: str, sku_set: set[str]) -> bool:
    digits = re.findall(r"\b\d{5,6}\b", content)
    if not digits:
        return True
    return all(digit in sku_set for digit in digits)


def _build_context_text(context: object) -> str:
    user_message = str(getattr(context, "user_message", "") or "")
    intent = str(getattr(context, "intent_label", "") or "")
    anchor = _infer_anchor(context)
    return " ".join(part for part in [user_message, intent, anchor] if part).strip()


def _auto_relabel_tag(tag: str, content: str) -> str:
    if tag != "SYN":
        return tag
    content_norm = normalize_text(content)
    if _mentions_sku(content) or _mentions_specs_without_sku(content_norm) or _mentions_numeric_specs(content_norm):
        return "QA"
    return tag


def _mentions_numeric_specs(content_norm: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\b", content_norm))


def _extract_existing_signatures(existing_text: str) -> set[str]:
    signatures: set[str] = set()
    for line in existing_text.splitlines():
        parsed = _parse_entry_line(line.strip())
        if not parsed:
            continue
        _date, _tag, _conf, content = parsed
        sig = _signature(content)
        if sig:
            signatures.add(sig)
    return signatures


def _signature(content: str) -> str:
    return normalize_text(content)


def _is_relevant_qa(content: str, context_text: str) -> bool:
    context_norm = normalize_text(context_text)
    if not context_norm:
        return False
    content_norm = normalize_text(content)

    content_nums = _extract_numbers(content_norm)
    context_nums = _extract_numbers(context_norm)
    if content_nums and any(num not in context_nums for num in content_nums):
        return False

    content_tokens = {token for token in content_norm.split() if len(token) > 2}
    context_tokens = {token for token in context_norm.split() if len(token) > 2}
    if content_tokens and context_tokens and not (content_tokens & context_tokens):
        return False
    return True


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def _is_mostly_vietnamese(content: str, threshold: float = 0.6) -> bool:
    tokens = content.split()
    if not tokens:
        return False
    ascii_tokens = [t for t in tokens if all(ord(ch) < 128 for ch in t)]
    ratio_ascii = len(ascii_tokens) / max(len(tokens), 1)
    return ratio_ascii <= threshold


def _is_generic_robot_hand_template(tag: str, signature: str) -> bool:
    if tag not in {"TEMPLATE", "RULE"}:
        return False
    if "robot" in signature and "tay" in signature and not _mentions_sku(signature) and not _mentions_numeric_specs(signature):
        return True
    return False


def _sku_in_context(content: str, context_text: str) -> bool:
    digits = re.findall(r"\b\d{5,6}\b", content)
    if not digits:
        return True
    context_norm = normalize_text(context_text)
    return any(digit in context_norm for digit in digits)


def _collect_known_skus(items: Sequence[ResourceItem]) -> set[str]:
    sku_set: set[str] = set()
    tokin_keys = [
        "Mã Tokin (Tokin Part No.)",
        "Tokin Part No.",
        "Tokin Part No",
        "Mã Tokin",
        "SKU",
        "sku",
    ]
    for item in items:
        for value in (item.code, get_raw_value(item.raw, tokin_keys)):
            if not value:
                continue
            digits = _extract_digits(str(value))
            if len(digits) in {5, 6}:
                sku_set.add(digits)
    return sku_set


def _extract_digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())
