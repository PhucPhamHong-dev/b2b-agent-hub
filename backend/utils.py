import json
import re
import unicodedata
from typing import Any, Dict, Optional


def normalize_text(text: str) -> str:
    """Purpose: Normalize free-form text for stable matching in the pipeline.
    Inputs/Outputs: Input is a raw string; output is a lowercase ASCII-only string with
        diacritics removed and whitespace collapsed.
    Side Effects / State: None; pure function.
    Dependencies: Uses unicodedata and regex; called by intent, retrieval, and guards.
    Failure Modes: Returns an empty string when input is falsy; regex may over-strip
        non-ASCII symbols, which is intended for matching.
    If Removed: Matching and routing degrade or break (intent/rule checks miss), causing
        misroutes in the pipeline and poor retrieval.
    Testing Notes: Validate Vietnamese text is normalized (e.g., "bec" -> "bec") and
        punctuation/whitespace are collapsed.
    """
    # Normalize to lowercase and strip diacritics for consistent matching.
    if not text:
        return ""
    lowered = text.lower()
    lowered = lowered.replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^a-z0-9\s\-_/._]+", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_key(text: str) -> str:
    """Purpose: Produce a compact normalization key without spaces.
    Inputs/Outputs: Input is a raw string; output is normalized string with spaces removed.
    Side Effects / State: None; pure function.
    Dependencies: Calls normalize_text; used in key comparisons and lookups.
    Failure Modes: Returns empty string for falsy input; otherwise deterministic.
    If Removed: Callers lose stable keying and matching for map/set operations.
    Testing Notes: Ensure spaces are removed after normalization.
    """
    # Collapse normalization output into a compact key.
    return normalize_text(text).replace(" ", "")


def extract_json_block(text: str) -> Optional[str]:
    """Purpose: Extract the first JSON object block from an arbitrary string.
    Inputs/Outputs: Input is a raw string; output is JSON substring or None.
    Side Effects / State: None; pure function.
    Dependencies: None beyond built-ins; used by safe_json_loads.
    Failure Modes: Returns None if braces are missing or inverted.
    If Removed: Model outputs cannot be parsed safely, breaking intent parsing.
    Testing Notes: Provide strings with extra text before/after JSON and ensure extraction.
    """
    # Locate the outermost JSON braces to extract a parseable block.
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Purpose: Parse a JSON object from a model output string safely.
    Inputs/Outputs: Input is raw text; output is a dict or None if parsing fails.
    Side Effects / State: None; pure function.
    Dependencies: Uses extract_json_block and json.loads; called by intent parsing.
    Failure Modes: Returns None on JSONDecodeError or missing JSON block.
    If Removed: Intent parsing becomes brittle and crashes on malformed model output.
    Testing Notes: Validate valid JSON parses and malformed JSON returns None.
    """
    # Parse only the extracted JSON block to avoid non-JSON prefixes/suffixes.
    block = extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None
