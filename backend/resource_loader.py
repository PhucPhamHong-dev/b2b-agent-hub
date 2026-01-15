from __future__ import annotations

"""Resource loader and retrieval utilities for the AgentX catalog.

This module loads AgentX.json into ResourceItem objects and provides lightweight,
deterministic retrieval helpers used by the agent pipeline.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import normalize_key, normalize_text


CODE_KEYS = [
    "sku",
    "code",
    "ma san pham",
    "ma tokin (tokin part no.)",
    "tokin part no.",
    "ma tokin",
    "ma",
    "product code",
]
NAME_KEYS = [
    "name",
    "ten",
    "ten san pham",
    "ten tieng viet",
    "ten tieng anh",
    "product name",
]
DESC_KEYS = [
    "description",
    "mo ta",
    "chi tiet",
    "spec",
]
CATEGORY_KEYS = [
    "category",
    "danh muc",
    "loai",
    "nhom",
    "product category",
]
LINK_KEYS = [
    "link san pham",
    "product link",
    "url",
    "link",
]

EXCLUDED_MATCH_KEYS = [
    "gia",
    "price",
    "khuyenmai",
    "discount",
    "chietkhau",
]

TOKIN_KEYS = [
    "ma tokin (tokin part no.)",
    "tokin part no.",
    "ma tokin",
]
P_CODE_KEYS = [
    "ma p (p part no.)",
    "p part no.",
    "ma p",
]
D_CODE_KEYS = [
    "ma d (d part no.)",
    "d part no.",
    "ma d",
]

CATEGORY_KEYWORDS = {
    "TIP": ["tip", "bec han"],
    "TIP_BODY": ["tip body", "than giu bec", "tip holder", "body"],
    "NOZZLE": ["nozzle", "chup khi"],
    "INSULATOR": ["insulator", "cach dien"],
    "ORIFICE": ["orifice", "su phan phoi khi", "gas diffuser"],
}


@dataclass
class ResourceItem:
    """Normalized view of a catalog record with a raw backing dict."""
    code: str
    name: str
    description: str
    category: str
    link: str
    raw: Dict[str, Any]


@dataclass
class ResourceMeta:
    """Metadata describing the resource file version for logging."""
    file_name: str
    updated_at: str
    sha256: str


class ResourceLoader:
    def __init__(self, path: Path) -> None:
        """Purpose: Configure the loader with a resource file path.
        Inputs/Outputs: Input is a Path to AgentX.json; no return value.
        Side Effects / State: Stores the path for later load calls.
        Dependencies: None beyond Path usage.
        Failure Modes: None at init; load() handles read/parse errors.
        If Removed: Resource loading cannot be configured for the pipeline.
        Testing Notes: Instantiate with a temp path and call load().
        """
        # Store the resource file location for subsequent loads.
        self._path = path

    def load(self) -> Tuple[List[ResourceItem], ResourceMeta]:
        """Purpose: Load and normalize catalog data from the resource file.
        Inputs/Outputs: No inputs; returns a list of ResourceItem and ResourceMeta.
        Side Effects / State: Reads file contents and computes hash/mtime.
        Dependencies: Uses json, hashlib, and helper _get_first_value.
        Failure Modes: JSON decode errors raise exceptions to the caller.
        If Removed: Pipeline cannot retrieve catalog items or log metadata.
        Testing Notes: Use a known AgentX.json and validate item normalization.
        """
        # Read bytes for hashing and parse JSON into normalized items.
        raw_bytes = self._path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        updated_at = datetime.fromtimestamp(self._path.stat().st_mtime).isoformat()

        data = json.loads(raw_bytes.decode("utf-8-sig"))
        items: List[Dict[str, Any]]
        if isinstance(data, dict):
            items = data.get("items", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []

        resource_items: List[ResourceItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            code = _get_first_value(item, CODE_KEYS)
            name = _get_first_value(item, NAME_KEYS)
            description = _get_first_value(item, DESC_KEYS)
            category = _get_first_value(item, CATEGORY_KEYS)
            link = _get_first_value(item, LINK_KEYS)

            resource_items.append(
                ResourceItem(
                    code=str(code or "").strip(),
                    name=str(name or "").strip(),
                    description=str(description or "").strip(),
                    category=str(category or "").strip(),
                    link=str(link or "").strip(),
                    raw=item,
                )
            )

        meta = ResourceMeta(
            file_name=self._path.name,
            updated_at=updated_at,
            sha256=sha256,
        )
        return resource_items, meta


def _get_first_value(item: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """Purpose: Find the first matching field in a dict by key synonyms.
    Inputs/Outputs: Input is a raw dict and a list of candidate keys; returns value or None.
    Side Effects / State: None.
    Dependencies: Uses normalize_key and _has_value.
    Failure Modes: Returns None when no keys match or values are empty.
    If Removed: Field mapping for code/name/category/link fails in load().
    Testing Notes: Verify synonym keys resolve to the expected value.
    """
    # Normalize keys and search for exact or partial matches.
    normalized_map = {normalize_key(k): k for k in item.keys()}
    for key in keys:
        normalized = normalize_key(key)
        if normalized in normalized_map:
            value = item.get(normalized_map[normalized])
            if _has_value(value):
                return value
    for key in keys:
        normalized = normalize_key(key)
        for item_key, actual_key in normalized_map.items():
            if normalized in item_key:
                value = item.get(actual_key)
                if _has_value(value):
                    return value
    return None


def _has_value(value: Any) -> bool:
    """Purpose: Determine whether a value is present and non-empty.
    Inputs/Outputs: Input is any value; output is True if usable.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: None; simple checks only.
    If Removed: Empty strings/None may be treated as valid fields.
    Testing Notes: Check None, empty string, and non-empty values.
    """
    # Treat None or empty strings as missing values.
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _should_include_key(key: str) -> bool:
    """Purpose: Filter out keys that should not be included in retrieval blobs.
    Inputs/Outputs: Input is a key string; output is True if it should be included.
    Side Effects / State: None.
    Dependencies: Uses normalize_key and EXCLUDED_MATCH_KEYS.
    Failure Modes: Overly broad exclusions may reduce recall in retrieval.
    If Removed: Price/discount fields may pollute retrieval scoring.
    Testing Notes: Verify excluded keys are filtered and others pass.
    """
    # Exclude pricing-related fields from matching.
    normalized = normalize_key(key)
    return not any(exclude in normalized for exclude in EXCLUDED_MATCH_KEYS)


def _build_item_blob(item: ResourceItem) -> str:
    """Purpose: Build a normalized text blob for retrieval scoring.
    Inputs/Outputs: Input is a ResourceItem; output is a normalized string.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and _should_include_key.
    Failure Modes: Missing fields produce smaller blobs but no exceptions.
    If Removed: Retrieval loses a unified matching representation.
    Testing Notes: Ensure name/description/category and raw fields appear in blob.
    """
    # Combine core fields with allowed raw fields for matching.
    parts: List[str] = []
    for value in (item.name, item.description, item.category):
        if value:
            parts.append(str(value))
    for key, value in item.raw.items():
        if value is None:
            continue
        if not _should_include_key(str(key)):
            continue
        parts.append(str(value))
    return normalize_text(" ".join(parts))


def _extract_numbers(text: str) -> List[float]:
    """Purpose: Extract numeric values from text for numeric matching.
    Inputs/Outputs: Input is text; output is list of floats.
    Side Effects / State: None.
    Dependencies: Uses regex for numeric patterns.
    Failure Modes: Non-numeric text returns an empty list.
    If Removed: Numeric matching for size/length becomes less accurate.
    Testing Notes: Validate integers and decimals are parsed correctly.
    """
    # Parse integer and decimal tokens into floats.
    numbers: List[float] = []
    for match in re.findall(r"\d+(?:\.\d+)?", text or ""):
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers


def _numbers_match(query_numbers: List[float], item_numbers: List[float]) -> bool:
    """Purpose: Check whether all query numbers appear in item numbers.
    Inputs/Outputs: Inputs are numeric lists; output is True if all match.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Empty query numbers always match; empty item numbers never match.
    If Removed: Numeric precision matching in retrieval becomes inconsistent.
    Testing Notes: Validate exact match and mismatch scenarios.
    """
    # Require each query number to appear within tolerance.
    if not query_numbers:
        return True
    if not item_numbers:
        return False
    for q in query_numbers:
        if not any(abs(q - item) < 1e-6 for item in item_numbers):
            return False
    return True


def _is_numeric_token(token: str) -> bool:
    """Purpose: Identify whether a token is a numeric literal.
    Inputs/Outputs: Input is token string; output is True if numeric.
    Side Effects / State: None.
    Dependencies: Uses regex.
    Failure Modes: None; returns False on non-numeric tokens.
    If Removed: Token-based numeric handling may accept invalid inputs.
    Testing Notes: Check integers, decimals, and non-numeric tokens.
    """
    # Treat integer or decimal strings as numeric.
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def detect_category_from_text(text: str) -> Optional[str]:
    """Purpose: Detect a product category label from free text.
    Inputs/Outputs: Input is text; output is a category string or None.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and CATEGORY_KEYWORDS.
    Failure Modes: Returns None when no keyword matches.
    If Removed: Category inference for queries becomes unavailable.
    Testing Notes: Validate known keyword matches and negatives.
    """
    # Map keywords to normalized category labels.
    normalized = normalize_text(text)
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if normalize_text(keyword) in normalized:
                return category
    return None


def detect_categories_in_query(query: str) -> List[str]:
    """Purpose: Collect all category labels mentioned in a query.
    Inputs/Outputs: Input is query string; output is list of category labels.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and CATEGORY_KEYWORDS.
    Failure Modes: Returns empty list if no keywords are present.
    If Removed: Multi-category queries cannot be detected for filtering.
    Testing Notes: Queries containing multiple category keywords should return both.
    """
    # Capture every category keyword mentioned in the query.
    normalized = normalize_text(query)
    categories: List[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if normalize_text(keyword) in normalized:
                categories.append(category)
                break
    return categories


def is_listing_query(query: str) -> bool:
    """Purpose: Heuristic check for listing-style queries.
    Inputs/Outputs: Input is query string; output is True if listing intent detected.
    Side Effects / State: None.
    Dependencies: Uses normalize_text.
    Failure Modes: Keyword-only heuristic may miss some listings.
    If Removed: Retrieval may treat list requests as generic queries.
    Testing Notes: Validate list keywords trigger True.
    """
    # Detect common listing keywords in the query.
    normalized = normalize_text(query)
    keywords = ["liet ke", "danh sach", "list", "cac", "nhung", "tat ca", "full"]
    return any(keyword in normalized for keyword in keywords)


def is_compatibility_query(query: str) -> bool:
    """Purpose: Heuristic check for compatibility/equivalence queries.
    Inputs/Outputs: Input is query string; output is True if compatibility intent detected.
    Side Effects / State: None.
    Dependencies: Uses normalize_text.
    Failure Modes: Keyword-only heuristic may miss some phrasing.
    If Removed: Compatibility-related routing loses a simple signal.
    Testing Notes: Validate compatibility keywords trigger True.
    """
    # Detect compatibility keywords in the query text.
    normalized = normalize_text(query)
    keywords = ["tuong thich", "equivalent", "thay the", "compatible", "dung chung"]
    return any(keyword in normalized for keyword in keywords)


def get_raw_value(raw: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    """Purpose: Fetch a raw value from a dict using key synonyms.
    Inputs/Outputs: Input is raw dict and key list; output is the matched value or None.
    Side Effects / State: None.
    Dependencies: Uses _get_first_value.
    Failure Modes: Returns None when no matching key exists.
    If Removed: Downstream access to raw fields becomes inconsistent.
    Testing Notes: Ensure it returns the first available synonym value.
    """
    # Thin wrapper for synonym key lookup.
    return _get_first_value(raw, keys)


def retrieve_relevant_items(question: str, items: List[ResourceItem], limit: int = 3) -> List[ResourceItem]:
    """Purpose: Rank and return catalog items relevant to a free-text question.
    Inputs/Outputs: Inputs are a question string and item list; output is ranked items.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and get_raw_value for numeric hints.
    Failure Modes: Empty question returns empty list; scoring is heuristic.
    If Removed: Semantic-like retrieval fallback in the pipeline disappears.
    Testing Notes: Query with SKU/size/length and verify top matches.
    """
    # Score items by SKU match, numeric hints, and category keywords.
    q = question.strip().lower()
    if not q:
        return []

    q_norm = normalize_text(q)
    numbers = re.findall(r"\d+(?:\.\d+)?", q)
    scored: List[Tuple[int, ResourceItem]] = []

    for item in items:
        score = 0
        sku = str(item.code or "").lower()
        name = str(item.name or "").lower()

        size = get_raw_value(item.raw, ["Kích thước dây (Size mm)"])
        length = get_raw_value(item.raw, ["Tổng chiều dài (mm)"])
        size_str = str(size) if size is not None else ""
        length_str = str(length) if length is not None else ""

        if sku and q and q in sku:
            score += 5000

        matches = 0
        for num in numbers:
            num_float = str(float(num)) if num else ""
            if size_str == num or size_str == num_float:
                score += 2000
                matches += 1
            if length_str == num or length_str == num_float:
                score += 1500
                matches += 1
            if num and num in sku:
                score += 500

        if matches >= 2:
            score += 2000

        if "bec" in q_norm and ("bec" in normalize_text(name) or "tip" in normalize_text(name)):
            score += 500
        if "chup" in q_norm and ("chup" in normalize_text(name) or "nozzle" in normalize_text(name)):
            score += 500

        if score > 400:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def _is_robot_item(name: str, desc: str) -> bool:
    """Purpose: Detect robot items from name/description text.
    Inputs/Outputs: Input is name and description; output is True if robot keywords exist.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: False negatives if keywords are missing from text.
    If Removed: Simple robot/hand classification loses a helper.
    Testing Notes: Validate strings containing "robot" or "robotic".
    """
    # Check for robot keywords in combined text.
    combined = f"{name} {desc}"
    return "robot" in combined or "robotic" in combined
