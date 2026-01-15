from __future__ import annotations

import json
from pathlib import Path
from typing import Set


class IntentMemory:
    """Persisted registry of detected intents for audit and analysis."""

    def __init__(self, path: Path) -> None:
        """Purpose: Initialize intent memory and load prior intents from disk.
        Inputs/Outputs: Input is a Path; no return value.
        Side Effects / State: Loads intents into an in-memory set.
        Dependencies: Calls _load; uses JSON file on disk.
        Failure Modes: JSON decode errors are ignored, leaving an empty set.
        If Removed: Intent tracking stops and new intents are not recorded.
        Testing Notes: Ensure a new intent is persisted and reloaded.
        """
        # Keep the backing file path and hydrate cached intents.
        self._path = path
        self._intents: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """Purpose: Load intent list from the JSON file if it exists.
        Inputs/Outputs: Reads self._path; no return value.
        Side Effects / State: Populates self._intents.
        Dependencies: json.loads and Path.read_text.
        Failure Modes: Missing file or JSONDecodeError results in empty cache.
        If Removed: Existing intent history is never loaded on startup.
        Testing Notes: Validate behavior with missing and malformed files.
        """
        # Read and parse JSON intent list.
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        intents = data.get("intents", [])
        if isinstance(intents, list):
            self._intents = {str(intent) for intent in intents}

    def _persist(self) -> None:
        """Purpose: Write the intent set to disk.
        Inputs/Outputs: Writes a JSON file; no return value.
        Side Effects / State: Persists current intent set.
        Dependencies: json.dumps and Path.write_text.
        Failure Modes: IO errors will raise exceptions (not handled here).
        If Removed: New intents are not persisted across restarts.
        Testing Notes: Ensure file content matches the in-memory set.
        """
        # Persist intent set for audit and reuse.
        payload = {"intents": sorted(self._intents)}
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def record(self, intent: str) -> bool:
        """Purpose: Record a new intent if it has not been seen before.
        Inputs/Outputs: Input is intent string; output is True if recorded.
        Side Effects / State: Mutates self._intents and writes to disk.
        Dependencies: Uses _persist for durability.
        Failure Modes: IO errors on persist; duplicate intents return False.
        If Removed: Intent discovery logging stops working.
        Testing Notes: Record a duplicate and ensure it returns False.
        """
        # Add intent if new and persist the updated set.
        if intent in self._intents:
            return False
        self._intents.add(intent)
        self._persist()
        return True
