from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .agent_pipeline import SalesAssistantAgent
from .config import load_settings
from .gemini_client import GeminiClient
from .intent_memory import IntentMemory
from .models import ChatRequest, ChatResponse, ImageSpec
from .resource_loader import ResourceLoader
from .session_store import SessionStore

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BASE_DIR / ".." / "frontend").resolve()
DATA_DIR = (BASE_DIR / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

logging.getLogger("autoss").setLevel(log_level)

ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

app = FastAPI(title="Autoss Sales Assistant Demo")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

settings = load_settings()
resource_loader = ResourceLoader(settings.resources_path)
intent_memory = IntentMemory(DATA_DIR / "intent_memory.json")
session_store = SessionStore(DATA_DIR / "sessions.json", max_sessions=3)

gemini = GeminiClient(settings)
agent = SalesAssistantAgent(
    gemini=gemini,
    resource_loader=resource_loader,
    intent_memory=intent_memory,
    prompts_dir=settings.prompts_dir,
    max_images=settings.max_images,
    max_attempts=settings.max_attempts,
    model_flash=settings.gemini_model_flash,
    model_pro=settings.gemini_model_pro,
)


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Purpose: Serve the frontend entrypoint HTML file.
    Inputs/Outputs: No inputs; returns a FileResponse for index.html.
    Side Effects / State: None.
    Dependencies: Uses FRONTEND_DIR and FastAPI routing.
    Failure Modes: FileResponse raises if the file is missing.
    If Removed: Frontend UI cannot load from the root path.
    Testing Notes: Request "/" and verify HTML is returned.
    """
    # Serve the static index.html from the frontend directory.
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Purpose: Handle chat requests and run the agent pipeline.
    Inputs/Outputs: Input is ChatRequest; output is ChatResponse with answer/logs/images.
    Side Effects / State: Updates session history and order_state in SessionStore.
    Dependencies: Uses SalesAssistantAgent, SessionStore, and GeminiClient.
    Failure Modes: Exceptions from agent or storage propagate as 500 errors.
    If Removed: Core chat functionality is unavailable.
    Testing Notes: Send a sample message and verify response schema and persistence.
    """
    # Load session context and run the agent pipeline.
    session_id = request.session_id
    if session_id:
        session_store.ensure_session(session_id)
        history = [message.dict() for message in session_store.get_messages(session_id)]
        order_state = session_store.get_order_state(session_id)
    else:
        history = []
        order_state = {}
    context = agent.handle_message(session_id, request.message, chat_history=history, order_state=order_state)

    session_store.add_message(context.session_id, "user", request.message)
    images = [ImageSpec(**image) for image in context.images]
    session_store.add_message(
        context.session_id,
        "assistant",
        context.answer_text,
        thinking_logs=context.thinking_logs,
        images=images,
        meta={
            "context_codes": [item.code for item in context.display_items if item.code],
            "asked_form": context.asked_form,
            "reminded_contact": context.reminded_contact,
        },
    )
    session_store.set_order_state(context.session_id, context.order_state)
    return ChatResponse(
        answer_text=context.answer_text,
        images=images,
        thinking_logs=context.thinking_logs,
        session_id=context.session_id,
    )


@app.get("/api/sessions")
def list_sessions() -> List[dict]:
    """Purpose: Return session summaries for the UI sidebar.
    Inputs/Outputs: No inputs; output is a list of summary dicts.
    Side Effects / State: None.
    Dependencies: Uses SessionStore.list_sessions.
    Failure Modes: None; returns empty list if no sessions.
    If Removed: UI cannot display session history list.
    Testing Notes: Create multiple sessions and verify sorting.
    """
    # Serialize summaries for the frontend.
    return [summary.dict() for summary in session_store.list_sessions()]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """Purpose: Return all messages for a given session.
    Inputs/Outputs: Input is session_id; output is a dict with message list.
    Side Effects / State: None.
    Dependencies: Uses SessionStore.get_messages.
    Failure Modes: Unknown session returns empty message list.
    If Removed: Frontend cannot load a session transcript.
    Testing Notes: Request a known session and verify message payload.
    """
    # Serialize stored messages for the requested session.
    messages = session_store.get_messages(session_id)
    return {
        "session_id": session_id,
        "messages": [message.dict() for message in messages],
    }
