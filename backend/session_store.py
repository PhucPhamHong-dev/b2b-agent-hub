from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .models import ImageSpec, SessionSummary, StoredMessage


class SessionStore:
    """Session storage for chat history, summaries, and order state."""

    def __init__(self, path: Optional[Path] = None, max_sessions: Optional[int] = None) -> None:
        """Purpose: Initialize the session store and hydrate from disk if available.
        Inputs/Outputs: Inputs are an optional file path and max_sessions cap; no return.
        Side Effects / State: Loads and caches sessions/summaries/order_state in memory.
        Dependencies: Calls _load; relies on StoredMessage/SessionSummary models.
        Failure Modes: JSON decode errors are swallowed and leave empty caches.
        If Removed: App loses session persistence and history endpoints break.
        Testing Notes: Verify load on startup populates caches and respects max_sessions.
        """
        # Keep configuration and preload persisted sessions if present.
        self._path = path
        self._max_sessions = max_sessions
        self._sessions: Dict[str, List[StoredMessage]] = {}
        self._summaries: Dict[str, SessionSummary] = {}
        self._order_states: Dict[str, Dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        """Purpose: Load persisted session data from disk into memory.
        Inputs/Outputs: Reads from self._path; no return value.
        Side Effects / State: Populates _sessions, _summaries, _order_states caches.
        Dependencies: Uses json.loads and pydantic models for validation.
        Failure Modes: Missing file or JSONDecodeError results in an empty cache.
        If Removed: Previously stored sessions are never restored on startup.
        Testing Notes: Corrupt JSON should not crash; valid JSON should hydrate caches.
        """
        # Read and decode persisted JSON if present.
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        sessions = data.get("sessions", {})
        summaries = data.get("summaries", {})
        order_states = data.get("order_states", {})
        for session_id, messages in sessions.items():
            self._sessions[session_id] = [StoredMessage(**msg) for msg in messages]
        for session_id, summary in summaries.items():
            self._summaries[session_id] = SessionSummary(**summary)
        if isinstance(order_states, dict):
            self._order_states = {
                session_id: state for session_id, state in order_states.items() if isinstance(state, dict)
            }
        if self._prune_sessions():
            self._persist()

    def _persist(self) -> None:
        """Purpose: Persist in-memory sessions and summaries to disk.
        Inputs/Outputs: Writes to self._path; no return value.
        Side Effects / State: Writes a JSON file with sessions/summaries/order_states.
        Dependencies: Uses json.dumps and Path.write_text.
        Failure Modes: IO errors raise exceptions (not caught here).
        If Removed: Messages and order_state are never saved across restarts.
        Testing Notes: Ensure file is created/updated and JSON structure matches models.
        """
        # Serialize current caches to disk for persistence.
        if not self._path:
            return
        payload = {
            "sessions": {
                session_id: [msg.dict() for msg in messages]
                for session_id, messages in self._sessions.items()
            },
            "summaries": {session_id: summary.dict() for session_id, summary in self._summaries.items()},
            "order_states": self._order_states,
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        thinking_logs: Optional[List[dict]] = None,
        images: Optional[List[ImageSpec]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> None:
        """Purpose: Append a message to a session and update its summary metadata.
        Inputs/Outputs: Inputs include session_id, role, content, optional logs/images/meta.
        Side Effects / State: Mutates in-memory caches and persists to disk.
        Dependencies: Uses StoredMessage, SessionSummary, _prune_sessions, _persist.
        Failure Modes: Persist can raise IO errors; missing session is created implicitly.
        If Removed: Chat history is not recorded and UI session list becomes stale.
        Testing Notes: Add a message and verify summary title/updated_at and file output.
        """
        # Create a StoredMessage and keep session metadata in sync.
        timestamp = time.time()
        message = StoredMessage(
            role=role,
            content=content,
            timestamp=timestamp,
            thinking_logs=thinking_logs,
            images=images,
            meta=meta,
        )
        self._sessions.setdefault(session_id, []).append(message)

        if session_id not in self._summaries:
            title = content.strip().splitlines()[0][:48] or "New Chat"
            self._summaries[session_id] = SessionSummary(
                session_id=session_id,
                title=title,
                updated_at=timestamp,
            )
        else:
            self._summaries[session_id].updated_at = timestamp
        self._prune_sessions()
        self._persist()

    def list_sessions(self) -> List[SessionSummary]:
        """Purpose: Return session summaries sorted by most recent activity.
        Inputs/Outputs: No inputs; returns a list of SessionSummary instances.
        Side Effects / State: None.
        Dependencies: Uses in-memory _summaries cache.
        Failure Modes: None; returns empty list if no sessions.
        If Removed: UI cannot show the session sidebar list.
        Testing Notes: Ensure ordering by updated_at descending.
        """
        # Sort summaries by last update time.
        return sorted(self._summaries.values(), key=lambda s: s.updated_at, reverse=True)

    def get_messages(self, session_id: str) -> List[StoredMessage]:
        """Purpose: Retrieve all stored messages for a session.
        Inputs/Outputs: Input is session_id; output is a list of StoredMessage.
        Side Effects / State: None.
        Dependencies: Uses in-memory _sessions cache.
        Failure Modes: Missing session returns empty list.
        If Removed: Chat history endpoint fails to return messages.
        Testing Notes: Query a known session and verify message count/order.
        """
        # Return cached messages or an empty list for unknown sessions.
        return self._sessions.get(session_id, [])

    def ensure_session(self, session_id: str) -> None:
        """Purpose: Ensure a session exists with summary and order_state stubs.
        Inputs/Outputs: Input is session_id; no return value.
        Side Effects / State: Creates entries in caches and persists to disk.
        Dependencies: Uses SessionSummary and _persist.
        Failure Modes: Persist can raise IO errors; otherwise deterministic.
        If Removed: New sessions are not created and message appends may fail.
        Testing Notes: Ensure new session creates empty history and summary.
        """
        # Initialize empty session structures when missing.
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._summaries[session_id] = SessionSummary(
                session_id=session_id,
                title="New Chat",
                updated_at=time.time(),
            )
            self._order_states.setdefault(session_id, {})
            self._prune_sessions()
            self._persist()

    def get_order_state(self, session_id: str) -> Dict[str, object]:
        """Purpose: Fetch the stored order_state for a session.
        Inputs/Outputs: Input is session_id; output is a shallow copy of state dict.
        Side Effects / State: None.
        Dependencies: Uses in-memory _order_states cache.
        Failure Modes: Missing session returns empty dict.
        If Removed: Agent loses session-scoped state and follow-ups break.
        Testing Notes: Verify updates persist across set/get calls.
        """
        # Return a copy to prevent external mutation of cache.
        return dict(self._order_states.get(session_id, {}))

    def set_order_state(self, session_id: str, state: Dict[str, object]) -> None:
        """Purpose: Persist order_state for a session.
        Inputs/Outputs: Inputs are session_id and state dict; no return value.
        Side Effects / State: Mutates cache and writes to disk.
        Dependencies: Uses _persist.
        Failure Modes: Persist can raise IO errors.
        If Removed: Follow-up context is lost across turns and restarts.
        Testing Notes: Set state and verify it appears in persisted JSON.
        """
        # Update cache and flush to disk.
        self._order_states[session_id] = state
        self._persist()

    def _prune_sessions(self) -> bool:
        """Purpose: Enforce max_sessions by dropping oldest sessions.
        Inputs/Outputs: No inputs; returns True if any sessions were removed.
        Side Effects / State: Mutates _sessions/_summaries/_order_states caches.
        Dependencies: Uses _max_sessions and updated_at ordering.
        Failure Modes: None; no-op when max_sessions is unset or not exceeded.
        If Removed: Session list grows unbounded and file size increases.
        Testing Notes: Set a low max_sessions and verify pruning order.
        """
        # Remove least-recent sessions when above the configured cap.
        if not self._max_sessions or self._max_sessions <= 0:
            return False
        if len(self._summaries) <= self._max_sessions:
            return False

        sorted_summaries = sorted(self._summaries.values(), key=lambda s: s.updated_at, reverse=True)
        keep_ids = {summary.session_id for summary in sorted_summaries[: self._max_sessions]}
        removed = [session_id for session_id in list(self._summaries.keys()) if session_id not in keep_ids]
        for session_id in removed:
            self._summaries.pop(session_id, None)
            self._sessions.pop(session_id, None)
            self._order_states.pop(session_id, None)
        return bool(removed)
