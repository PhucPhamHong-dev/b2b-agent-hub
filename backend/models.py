from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for chat API."""
    session_id: Optional[str] = Field(default=None)
    message: str


class ImageSpec(BaseModel):
    """Rendered image placement specification for UI."""
    url: str
    after_paragraph_index: int


class ChatResponse(BaseModel):
    """Response payload returned by the chat API."""
    answer_text: str
    images: List[ImageSpec]
    thinking_logs: List[Dict[str, str]]
    session_id: str


class StoredMessage(BaseModel):
    """Persisted message record with optional logs and images."""
    role: str
    content: str
    timestamp: float
    thinking_logs: Optional[List[Dict[str, str]]] = None
    images: Optional[List[ImageSpec]] = None
    meta: Optional[Dict[str, Any]] = None


class SessionSummary(BaseModel):
    """Lightweight session summary for sidebar listing."""
    session_id: str
    title: str
    updated_at: float
