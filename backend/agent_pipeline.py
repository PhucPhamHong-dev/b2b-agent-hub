"""Autoss agent pipeline orchestration and deterministic guards.

Role:
    Implements the end-to-end routing, retrieval, guardrails, generation, and memory
    update flow for the sales assistant. It owns the PipelineContext contract and all
    step-level decisions used by the ADK runner.

Pipeline data contract (core fields passed across steps):
    - intent_label, intent_topic, next_action: routing outputs from intent detection.
    - intent_entities: structured slots (skus, product_group, amp, required_categories, etc.).
    - should_ask_type, should_show_form, should_render_products: guardrails for generation.
    - order_state.selected_sku/selected_group/quantity/contact: session-scoped state.
    - short_memory.last_anchor/last_results/pending_request: short-memory context.
    - display_items/items/related_items: retrieval results used for rendering.
    - is_asking_price/is_availability_query/is_info_only: high-level question flags.

Step contracts:
    Intent Detection:
        Reads user_message + short_memory; sets intent_label/topic/next_action/entities.
    Resource Retrieval:
        Uses intent/slots to populate items/related_items and logs match context.
    Context Guard:
        Applies business rules to set should_* flags and normalize order_state.
    Generation:
        Produces answer_text and images based on items + guard flags.
    Finalize:
        Persists order_state + short_memory updates for next turn.
"""

from __future__ import annotations

import json
import os
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .adk_runtime import AdkAgent, AdkStep
from .gemini_client import GeminiClient
from .intent_memory import IntentMemory
from .prompt_loader import load_prompt
from .resource_loader import ResourceItem, ResourceLoader, get_raw_value, retrieve_relevant_items
from .utils import normalize_text, safe_json_loads

logger = logging.getLogger("autoss.agent")

ASK_SELLING_SCOPE_PHRASES = [
    "ban gi",
    "ban cai gi",
    "cong ty ban ban gi",
    "shop ban gi",
    "kinh doanh gi",
]

SELLING_SCOPE_TEMPLATES = [
    (
        "Hiện tại Autoss chuyên phụ kiện cho súng hàn MIG/MAG Tokinarc (Nhật Bản), gồm: "
        "béc hàn, thân giữ béc, chụp khí, cách điện và sứ phân phối khí. "
    ),
    (
        "Autoss đang cung cấp phụ kiện Tokinarc cho MIG/MAG: béc hàn, thân giữ béc, chụp khí, "
        "cách điện, sứ phân phối khí."
    ),
    (
        "Danh mục bên em là phụ kiện MIG/MAG Tokinarc (Nhật Bản): béc hàn, thân giữ béc, chụp khí, "
        "cách điện, sứ phân phối khí."
    ),
]
SHORT_MEMORY_TTL_SEC = 15 * 60

PIPELINE_STEPS: List[Tuple[str, str]] = [
    ("Intent Detection", "Phân tích yêu cầu và linh kiện đi kèm..."),
    ("Data Retrieval", "Truy xuất thông tin linh kiện từ AgentX..."),
    ("Context Guard", "Kiểm tra trạng thái Tay/Robot và Thương mại..."),
    ("Final Logic", "Áp dụng quy tắc mặc định súng Tay..."),
]

DEFAULT_PRICE_REPLY = (
    "Dạ, Em sẽ ghi nhận nhu cầu và chuyển bộ phận phụ trách phản hồi chi tiết cho Anh/Chị."
)

ASK_TYPE_QUESTION = "Anh/Chị đang dùng súng hàn tay hay súng hàn robot ạ?"
DEFAULT_HAND_NOTE = (
    "Dạ, Em đang tư vấn theo bộ phụ kiện cho súng Tay MIG 350A thông dụng, "
    "nếu Anh/Chị dùng Robot hãy cho em biết để em đổi mã ạ."
)
TECHNICAL_INTENTS = {
    "PRODUCT_LOOKUP",
    "ACCESSORY_LOOKUP",
    "ACCESSORY_BUNDLE_LOOKUP",
    "LIST",
    "LIST_REQUEST",
    "SLOT_FILL_AMP",
}
TECHNICAL_CLOSING_OPTIONS = [
    "Anh/Chị muốn em liệt kê thêm linh kiện đi kèm cùng hệ để mình ráp đồng bộ không ạ?",
]

FORM_BLOCK = "🏢 Tên công ty\n👤 Người liên hệ\n📞 Số điện thoại (Zalo)"
REMINDER_LINE = (
    "Dạ, Anh/Chị cho em xin 3 thông tin: Tên công ty, Người liên hệ, Số điện thoại (Zalo). "
    "Em sẽ chuyển thông tin cho nhân viên phụ trách qua Zalo để hỗ trợ chi tiết ạ."
)
NO_RETAIL_REPLY = "Dạ bên em không bán lẻ 1 cái ạ. Anh/Chị cho em số lượng dự kiến và mã cần mua để em tư vấn đúng ạ."
REPEAT_BLOCK_REPLY = "Dạ em đã gửi danh sách trước đó rồi ạ. Anh/Chị cần em gửi lại không ạ?"
MISSING_IMAGE_NOTICE = (
    "Dạ em sẽ gửi link hình ảnh trên website cho Anh/Chị qua Zalo khi mình chốt thông tin giúp em ạ."
)
CODE_LOOKUP_NOT_FOUND_REPLY = "Dạ, Em sẽ ghi nhận và chuyển bộ phận phụ trách phản hồi cho Anh/Chị."
ASK_SKU_GROUP_REPLY = (
    "Dạ Anh/Chị cho em xin mã hoặc nhóm sản phẩm cần mua (béc hàn, chụp khí, thân giữ béc, cách điện, sứ phân phối khí) "
    "để em tư vấn đúng ạ."
)
AVAILABILITY_NEED_QTY_REPLY = "Dạ em đã ghi nhận nhu cầu. Anh/Chị cho em xin số lượng dự kiến để em hỗ trợ đúng ạ."

RELATED_QUERY_RE = re.compile(
    r"\b(di kem|phu kien|linh kien|kem theo|di cung|chup|chup khi|than giu|cach dien|su|orifice|insulator|body)\b",
    re.IGNORECASE,
)
BUNDLE_QUERY_RE = re.compile(
    r"\b(kem theo|di kem|phu kien di kem|phu kien kem theo)\b",
    re.IGNORECASE,
)
TYPE_ANSWER_RE = re.compile(r"\b(tay|robot|robotic|hand)\b", re.IGNORECASE)
PRICE_RE = re.compile(
    r"\b(gia|chiet khau|bao gia|ton kho|kho|co san|con hang|co hang|giao|vat)\b",
    re.IGNORECASE,
)
LISTING_RE = re.compile(r"\b(liet ke|danh sach|list|cac ma|nhung ma|ma nao)\b", re.IGNORECASE)
INFO_RE = re.compile(r"\b(xuat xu|nguon goc|vat lieu|chat lieu|ampe|ampere|amp)\b", re.IGNORECASE)
CLOSE_INTENT_RE = re.compile(
    r"\b(so luong|dat hang|don hang|bao gia|giao hang|xuat hoa don|xac nhan|lay hang|lay san pham|combo)\b",
    re.IGNORECASE,
)
CODE_RE = re.compile(r"\b(tokin(?:arc)?\s*\d+)\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\d{8,12}\b")
SINGLE_UNIT_RE = re.compile(r"\b(1|mot)\s*(cai|chiec|con|bo|cap)\b", re.IGNORECASE)
INFO_ONLY_RE = re.compile(
    r"\b(hang trung quoc|hang tq|xuat xu|nguon goc|hang nhat|nhat ban|chinh hang|tokinarc|"
    r"co phai tokinarc|hang that)\b",
    re.IGNORECASE,
)
PRODUCT_INFO_RE = re.compile(
    r"\b(thong tin san pham|thong so|cho xem|xem thong tin|xem thong so|xem san pham|xem hinh|hinh anh|anh san pham|link anh)\b",
    re.IGNORECASE,
)
REPEAT_REQUEST_RE = re.compile(r"\b(gui lai|nhac lai|cho xem lai|xem lai|gui lai danh sach|nhac lai danh sach)\b", re.IGNORECASE)
ACK_WORDS = {"ok", "oke", "okay", "okie", "duoc", "dc", "roi", "r", "vang", "va", "vay", "u", "uh", "da"}
BUY_INTENT_RE = re.compile(r"\b(mua|chot|chot don|chot mua|xac nhan mua)\b", re.IGNORECASE)
QUANTITY_RE = re.compile(r"\b(\d{1,6})\s*(cai|chiec|con|bo|cap|set|pcs|sp)\b", re.IGNORECASE)
SO_LUONG_RE = re.compile(r"\b(so luong|sl)\s*(\d{1,6})\b", re.IGNORECASE)
COMPATIBILITY_RE = re.compile(r"\b(tuong thich|compatible|equivalent|thay the|dung chung)\b", re.IGNORECASE)
D_CODE_RE = re.compile(r"\bU[0-9A-Z]{4,}\b", re.IGNORECASE)
P_CODE_RE = re.compile(r"\bP[0-9A-Z]{4,}\b", re.IGNORECASE)
NUM_CODE_RE = re.compile(r"\b\d{5,6}\b")
AMP_ANY_RE = re.compile(r"\b(\d{3})\s*a\b", re.IGNORECASE)
SIZE_RE = re.compile(r"\b(\d(?:\.\d)?)\b")
TIP_SIZE_LEN_RE = re.compile(r"\b(\d(?:\.\d)?)\s*[x×]\s*(\d{2,3})(?:\s*l)?\b", re.IGNORECASE)
LOOKUP_HINT_RE = re.compile(
    r"\b(ma|sku|code|part\s*no|la\s*ma\s*nao|ma\s*nao|ma\s*gi|xin\s*ma|tip\s*body\s*la)\b",
    re.IGNORECASE,
)
SELLING_VERB_RE = re.compile(r"\b(co\s*ban|ban\s*khong|ban\s*ko|ban\s*k)\b", re.IGNORECASE)
FOLLOWUP_CUE_RE = re.compile(r"\b(thi sao|the sao|sao)\b", re.IGNORECASE)
THREAD_RE = re.compile(r"\bM\d+(?:x\d+)?\b", re.IGNORECASE)
MATERIAL_RE = re.compile(r"\b(nhom|aluminum|aluminium|al)\b", re.IGNORECASE)

RELATED_CATEGORIES = {"TIPBODY", "INSULATOR", "ORIFICE", "NOZZLE"}
GROUP_SYNONYMS = {
    "TIP": ["bec han", "contact tip", "tip"],
    "TIP_BODY": ["than giu bec", "tip body", "holder"],
    "NOZZLE": ["chup khi", "nozzle"],
    "INSULATOR": ["cach dien", "insulator"],
    "ORIFICE": ["su phan phoi", "orifice", "diffuser"],
}
DEFAULT_BUNDLE_CATEGORIES = ["TIP_BODY", "INSULATOR", "NOZZLE", "ORIFICE"]
PART_SYNONYMS = {
    "TIP_BODY": ["than giu bec", "tip body", "holder"],
    "INSULATOR": ["cach dien", "insulator"],
    "NOZZLE": ["chup khi", "nozzle"],
    "ORIFICE": ["su phan phoi khi", "orifice", "diffuser"],
}
BUNDLE_HINT_WORDS = ["dong bo", "tron bo", "full bo", "kem ca bo", "combo", "di kem du bo"]
AFFIRM_TERMS = {
    "muon",
    "ok",
    "oke",
    "okay",
    "dong y",
    "nhat tri",
    "chap nhan",
    "co",
    "duoc",
    "yes",
}
NEGATE_TERMS = {
    "khong",
    "khong can",
    "khong muon",
    "de sau",
    "thoi",
    "huy",
    "cancel",
    "ko",
    "k",
}
ACCESSORY_INVITE_TERMS = {
    "linh kien di kem",
    "rap dong bo",
    "liet ke them",
    "di kem cung he",
}
BULK_QTY_KEYS = [
    "min_bulk_qty",
    "min_bulk",
    "min_qty",
    "min_order_qty",
    "min order qty",
    "bulk_qty",
    "bulk qty",
    "so luong toi thieu",
    "sl toi thieu",
]


@dataclass
class ImageCandidate:
    """Lightweight image candidate for post-processing insertion."""
    code: str
    name: str
    url: str


@dataclass
class IntentDecision:
    """Structured intent decision returned by rule or model parsing."""
    intent: str = "OTHER"
    buy_intent: bool = False
    info_only: bool = False
    topic: str = "product"
    entities: Dict[str, object] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    next_action: str = "ANSWER_ONLY"
    commercial_action: Dict[str, object] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable context passed through each pipeline step."""
    session_id: str
    user_message: str
    chat_history: List[dict]
    prompts_dir: Path
    max_images: int
    model_flash: str
    model_pro: str
    answer_text: str = ""
    order_state: Dict[str, object] = field(default_factory=dict)
    short_memory: Dict[str, object] = field(default_factory=dict)
    resolved_request: Dict[str, object] = field(default_factory=dict)
    intent_label: str = "OTHER"
    intent_topic: str = "product"
    next_action: str = "ANSWER_ONLY"
    intent_entities: Dict[str, object] = field(default_factory=dict)
    intent_missing: List[str] = field(default_factory=list)
    primary_code: str = ""
    buy_intent: bool = False
    request_contact: bool = False
    missing_quantity: bool = False
    missing_sku: bool = False
    missing_contact: bool = False
    missing_type: bool = False
    items: List[ResourceItem] = field(default_factory=list)
    related_items: List[ResourceItem] = field(default_factory=list)
    all_items: List[ResourceItem] = field(default_factory=list)
    catalog_items: List[ResourceItem] = field(default_factory=list)
    previous_codes: List[str] = field(default_factory=list)
    has_asked_type: bool = False
    has_answered_type: bool = False
    has_default_hand_note: bool = False
    has_asked_form: bool = False
    asked_form: bool = False
    reminded_contact: bool = False
    waiting_for_contact: bool = False
    reminder_count: int = 0
    should_remind_contact: bool = False
    has_contact_info: bool = False
    contact_received: bool = False
    should_ask_type: bool = False
    force_default_hand: bool = False
    should_show_form: bool = False
    is_asking_related: bool = False
    is_availability_query: bool = False
    is_asking_price: bool = False
    is_info_query: bool = False
    is_info_only: bool = False
    is_close_intent: bool = False
    is_single_unit: bool = False
    has_code_query: bool = False
    should_render_products: bool = False
    should_repeat_products: bool = False
    images: List[dict] = field(default_factory=list)
    thinking_logs: List[Dict[str, str]] = field(default_factory=list)
    display_items: List[ResourceItem] = field(default_factory=list)

    def log(self, event: str, detail: str, status: str = "success") -> None:
        """Purpose: Append a structured log entry for UI and debugging.
        Inputs/Outputs: Inputs are event, detail, status; no return value.
        Side Effects / State: Mutates thinking_logs list on the context.
        Dependencies: Used by pipeline step wrappers.
        Failure Modes: None; always appends.
        If Removed: UI loses step-by-step logs and debugging output.
        Testing Notes: Ensure entries appear in ChatResponse.thinking_logs.
        """
        # Store a normalized log entry for the frontend.
        self.thinking_logs.append(
            {
                "event": event,
                "step": event,
                "detail": detail,
                "status": status,
            }
        )


def mask_contact_value(value: object) -> str:
    """Purpose: Mask contact-like values for safe logging.
    Inputs/Outputs: Input is any value; output is a masked string with last digits only.
    Side Effects / State: None.
    Dependencies: Uses regex digit extraction.
    Failure Modes: Non-numeric inputs yield a generic mask.
    If Removed: Logs may expose sensitive contact data.
    Testing Notes: Verify outputs for short and long numeric strings.
    """
    # Keep only the last digits while hiding the rest.
    if value is None:
        return ""
    digits = re.findall(r"\d", str(value))
    if len(digits) < 4:
        return "***"
    return "***" + "".join(digits[-3:])


def sanitize_state_for_log(state: Dict[str, object]) -> Dict[str, object]:
    """Purpose: Return a log-safe copy of order state with masked contact.
    Inputs/Outputs: Input is a state dict; output is a sanitized shallow copy.
    Side Effects / State: None.
    Dependencies: Uses mask_contact_value.
    Failure Modes: None; returns empty dict for falsy input.
    If Removed: Logs may leak sensitive contact fields.
    Testing Notes: Ensure contact fields are masked in output.
    """
    # Clone state and mask contact value for logging.
    safe = dict(state) if state else {}
    if safe.get("contact"):
        safe["contact"] = mask_contact_value(safe["contact"])
    return safe


def default_short_memory() -> Dict[str, object]:
    """Purpose: Provide a default short-memory structure for sessions.
    Inputs/Outputs: No inputs; returns a dict with initialized slots.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: None; deterministic structure.
    If Removed: Short-memory normalization cannot initialize missing fields.
    Testing Notes: Validate required keys exist in the returned dict.
    """
    # Initialize the short-memory slots with empty defaults.
    return {
        "last_anchor": {"sku": "", "cat": "", "line_amp": "", "is_robot": None, "name": ""},
        "last_intent": "",
        "last_topic": "",
        "last_results": [],
        "pending_request": {"required_parts": [], "missing_fields": [], "done_parts": [], "todo_parts": []},
        "pending_action": {
            "action": "",
            "required_parts": [],
            "anchor_sku": "",
            "product_group": "",
            "constraints": {},
        },
        "last_user_constraints": {},
        "last_commercial_context": {"quantity": None, "contact_collected": False, "show_form": False},
    }


def normalize_short_memory(order_state: Dict[str, object]) -> Dict[str, object]:
    """Purpose: Normalize and TTL-expire short-memory in order_state.
    Inputs/Outputs: Input is order_state; output is a normalized short_memory dict.
    Side Effects / State: Resets memory when TTL has expired.
    Dependencies: Uses SHORT_MEMORY_TTL_SEC and default_short_memory.
    Failure Modes: None; invalid timestamps trigger reset.
    If Removed: Memory can become stale and break follow-up resolution.
    Testing Notes: Simulate expired timestamps and verify reset.
    """
    # Restore or reset short-memory based on TTL.
    mem = order_state.get("short_memory")
    ts = order_state.get("short_memory_ts")
    now = time.time()
    if not isinstance(mem, dict):
        mem = default_short_memory()
    if not isinstance(ts, (int, float)) or (now - ts) > SHORT_MEMORY_TTL_SEC:
        mem = default_short_memory()
    mem.setdefault("last_anchor", {"sku": "", "cat": "", "line_amp": "", "is_robot": None, "name": ""})
    mem.setdefault("last_results", [])
    mem.setdefault("pending_request", {"required_parts": [], "missing_fields": [], "done_parts": [], "todo_parts": []})
    mem.setdefault(
        "pending_action",
        {
            "action": "",
            "required_parts": [],
            "anchor_sku": "",
            "product_group": "",
            "constraints": {},
        },
    )
    mem.setdefault("last_user_constraints", {})
    mem.setdefault("last_commercial_context", {"quantity": None, "contact_collected": False, "show_form": False})
    return mem


def sanitize_short_memory_for_log(mem: Dict[str, object]) -> Dict[str, object]:
    """Purpose: Produce a compact, log-safe view of short memory.
    Inputs/Outputs: Input is short_memory dict; output is filtered summary dict.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: None; returns empty dict if input is falsy.
    If Removed: Debug logs become verbose and harder to read.
    Testing Notes: Ensure sensitive fields are omitted in output.
    """
    # Reduce memory to a stable, log-friendly subset.
    if not mem:
        return {}
    last_anchor = mem.get("last_anchor") or {}
    pending = mem.get("pending_request") or {}
    pending_action = mem.get("pending_action") or {}
    return {
        "last_anchor": {
            "sku": last_anchor.get("sku"),
            "cat": last_anchor.get("cat"),
            "line_amp": last_anchor.get("line_amp"),
            "is_robot": last_anchor.get("is_robot"),
        },
        "last_intent": mem.get("last_intent"),
        "last_topic": mem.get("last_topic"),
        "last_results": mem.get("last_results"),
        "pending_request": {
            "required_parts": pending.get("required_parts"),
            "missing_fields": pending.get("missing_fields"),
            "done_parts": pending.get("done_parts"),
            "todo_parts": pending.get("todo_parts"),
        },
        "pending_action": {
            "action": pending_action.get("action"),
            "required_parts": pending_action.get("required_parts"),
            "anchor_sku": pending_action.get("anchor_sku"),
            "product_group": pending_action.get("product_group"),
        },
        "last_user_constraints": mem.get("last_user_constraints"),
        "last_commercial_context": mem.get("last_commercial_context"),
    }


class SalesAssistantAgent:
    def __init__(
        self,
        gemini: GeminiClient,
        resource_loader: ResourceLoader,
        intent_memory: IntentMemory,
        prompts_dir: Path,
        max_images: int,
        max_attempts: int,
        model_flash: str,
        model_pro: str,
    ) -> None:
        """Purpose: Initialize the agent pipeline runner and dependencies.
        Inputs/Outputs: Inputs are Gemini client, resource loader, intent memory,
            prompt directory, limits, and model names; no return value.
        Side Effects / State: Constructs an AdkAgent with ordered steps.
        Dependencies: Uses AdkAgent/AdkStep and step methods on this class.
        Failure Modes: None at init; runtime errors occur within step functions.
        If Removed: The chat endpoint cannot construct the agent pipeline.
        Testing Notes: Instantiate with mocks and verify steps are registered.
        """
        # Store dependencies and build the ADK step runner.
        self._gemini = gemini
        self._resource_loader = resource_loader
        self._prompts_dir = prompts_dir
        self._max_images = max_images
        self._model_flash = model_flash
        self._model_pro = model_pro
        self._agent = AdkAgent(
            steps=[
                AdkStep("pipeline_logs", self._step_pipeline_logs),
                AdkStep("intent_detection", self._step_intent_detection),
                AdkStep("resource_retrieval", self._step_resource_retrieval),
                AdkStep("context_guard", self._step_context_guard),
                AdkStep("generation", self._step_generation),
                AdkStep("finalize", self._step_finalize),
            ]
        )

    def handle_message(
        self, session_id: Optional[str], user_message: str, chat_history: List[dict], order_state: Dict[str, object]
    ) -> PipelineContext:
        """Purpose: Run the full pipeline for a user message and return context.
        Inputs/Outputs: Inputs are session_id, message, history, and order_state; output
            is a populated PipelineContext.
        Side Effects / State: Mutates context and logs; does not persist state itself.
        Dependencies: Uses AdkAgent.run and PipelineContext normalization.
        Failure Modes: Exceptions in steps propagate to the caller.
        If Removed: Chat handler cannot execute pipeline logic.
        Testing Notes: Pass a simple message and verify context fields are set.
        """
        # Build initial context, normalize memory, and execute pipeline steps.
        context = PipelineContext(
            session_id=session_id or uuid.uuid4().hex,
            user_message=user_message,
            chat_history=chat_history,
            prompts_dir=self._prompts_dir,
            max_images=self._max_images,
            model_flash=self._model_flash,
            model_pro=self._model_pro,
            order_state=normalize_order_state(order_state),
        )
        context.short_memory = normalize_short_memory(context.order_state)
        logger.info("session=%s question=%s", context.session_id, user_message)
        self._agent.run(context)
        return context

    def _step_pipeline_logs(self, context: PipelineContext) -> None:
        """Purpose: Emit standard pipeline step logs for UI/debugging.
        Inputs/Outputs: Input is PipelineContext; no return value.
        Side Effects / State: Appends to thinking_logs and emits logger entries.
        Dependencies: Uses PIPELINE_STEPS and PipelineContext.log.
        Failure Modes: None; logs are best-effort.
        If Removed: Frontend reasoning logs lose step scaffolding.
        Testing Notes: Ensure logs are emitted for each pipeline step.
        """
        # Emit a pending and success log for each pipeline step.
        for step, detail in PIPELINE_STEPS:
            context.log(step, detail, status="pending")
            context.log(step, detail, status="success")
            logger.info("session=%s step=%s status=success", context.session_id, step)

    def _step_intent_detection(self, context: PipelineContext) -> None:
        """Purpose: Determine intent, slots, and routing decision for the message.
        Inputs/Outputs: Input is PipelineContext; mutates intent fields in context.
        Side Effects / State: Updates order_state and short_memory via resolved request.
        Dependencies: Uses rule-based checks, resolve_request_with_memory, and Gemini.
        Failure Modes: LLM errors fall back to empty output and rule parsing.
        If Removed: Downstream steps have no intent routing and will misbehave.
        Testing Notes: Validate rule-based paths and LLM parsing with sample queries.
        """
        # Resolve short-memory and apply rule-based intent guards before LLM.
        memory_before = sanitize_short_memory_for_log(context.short_memory)
        parsed_input = parse_user_input(context.user_message)
        resolved = resolve_request_with_memory(
            context.user_message,
            parsed_input,
            context.short_memory,
        )
        context.resolved_request = resolved
        apply_resolved_to_order_state(context.order_state, resolved)
        logger.info(
            "session=%s memory_before=%s resolved_request=%s",
            context.session_id,
            json.dumps(memory_before, ensure_ascii=True),
            json.dumps(resolved, ensure_ascii=True),
        )
        if is_ask_selling_scope(context.user_message):
            decision = IntentDecision(
                intent="ASK_SELLING_SCOPE",
                buy_intent=False,
                info_only=False,
                topic="commercial",
                entities={},
                missing=[],
                next_action="ANSWER_ONLY",
            )
            self._apply_intent_decision(context, decision)
            logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(decision.__dict__, ensure_ascii=True),
            )
            return
        if is_type_only_message(context.user_message):
            normalized = normalize_text(context.user_message)
            pending_action = context.short_memory.get("pending_action") or {}
            if pending_action.get("action"):
                entities: Dict[str, object] = {
                    "is_robot": "robot" in normalized,
                    "is_hand": "tay" in normalized or "hand" in normalized,
                }
                if pending_action.get("anchor_sku"):
                    entities["skus"] = [pending_action.get("anchor_sku")]
                if pending_action.get("product_group"):
                    entities["product_group"] = pending_action.get("product_group")
                required_parts = pending_action.get("required_parts") or []
                if required_parts:
                    entities["required_categories"] = required_parts
                    entities["bundle_hint"] = True
                for key, value in (pending_action.get("constraints") or {}).items():
                    if value:
                        entities[key] = value
                decision = IntentDecision(
                    intent=str(pending_action.get("action") or "ACCESSORY_BUNDLE_LOOKUP"),
                    buy_intent=False,
                    info_only=False,
                    topic="product",
                    entities=entities,
                    missing=[],
                    next_action="ANSWER_ONLY",
                )
                self._apply_intent_decision(context, decision)
                logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
                logger.debug(
                    "session=%s intent_decision=%s",
                    context.session_id,
                    json.dumps(decision.__dict__, ensure_ascii=True),
                )
                return
            decision = IntentDecision(
                intent="TYPE_SWITCH",
                buy_intent=False,
                info_only=False,
                topic="product",
                entities={
                    "is_robot": "robot" in normalized,
                    "is_hand": "tay" in normalized or "hand" in normalized,
                },
                missing=[],
                next_action="ANSWER_ONLY",
            )
            self._apply_intent_decision(context, decision)
            logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(decision.__dict__, ensure_ascii=True),
            )
            return
        if is_amp_only_message(context.user_message):
            normalized = normalize_text(context.user_message)
            amp_match = AMP_ANY_RE.search(normalized)
            amp_value = f"{amp_match.group(1)}A" if amp_match else ""
            pending_action = context.short_memory.get("pending_action") or {}
            pending_parts = pending_action.get("required_parts") or (
                (context.short_memory.get("pending_request") or {}).get("required_parts") or []
            )
            slot_target = pending_action.get("action") or ""
            if not slot_target and pending_parts:
                slot_target = "ACCESSORY_BUNDLE_LOOKUP"
            if not slot_target:
                slot_target = (
                    context.short_memory.get("last_intent")
                    or context.order_state.get("last_intent")
                    or ""
                )
            entities: Dict[str, object] = {
                "amp": amp_value,
                "slot_target_intent": slot_target,
            }
            if pending_action.get("anchor_sku"):
                entities["skus"] = [pending_action.get("anchor_sku")]
            if pending_parts:
                entities["required_categories"] = pending_parts
                entities["bundle_hint"] = True
            decision = IntentDecision(
                intent="SLOT_FILL_AMP",
                buy_intent=False,
                info_only=False,
                topic="product",
                entities=entities,
                missing=[],
                next_action="ANSWER_ONLY",
            )
            self._apply_intent_decision(context, decision)
            logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(decision.__dict__, ensure_ascii=True),
            )
            return
        if is_pure_quantity_message(context.user_message) or is_quantity_followup_message(context.user_message):
            quantity = parsed_input.get("quantity")
            if quantity is None:
                quantity = parse_pure_quantity_value(context.user_message)
            entities: Dict[str, object] = {}
            if quantity is not None:
                entities["quantity"] = quantity
            decision = IntentDecision(
                intent="QUANTITY_FOLLOWUP",
                buy_intent=True,
                info_only=False,
                topic="product",
                entities=entities,
                missing=[],
                next_action="ANSWER_ONLY",
            )
            self._apply_intent_decision(context, decision)
            logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(decision.__dict__, ensure_ascii=True),
            )
            return
        forced_decision = build_forced_decision(context, parsed_input, resolved)
        if forced_decision:
            self._apply_intent_decision(context, forced_decision)
            logger.info(
                "session=%s intent=%s action=%s forced=true",
                context.session_id,
                forced_decision.intent,
                forced_decision.next_action,
            )
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(forced_decision.__dict__, ensure_ascii=True),
            )
            return
        if is_tech_product_inquiry(context.user_message):
            constraints = extract_lookup_constraints(context.user_message)
            decision = IntentDecision(
                intent="PRODUCT_LOOKUP",
                buy_intent=False,
                info_only=False,
                topic="product",
                entities=constraints,
                missing=[],
                next_action="ANSWER_ONLY",
            )
            self._apply_intent_decision(context, decision)
            logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(decision.__dict__, ensure_ascii=True),
            )
            return
        if is_technical_lookup(context.user_message):
            constraints = extract_lookup_constraints(context.user_message)
            decision = IntentDecision(
                intent="PRODUCT_LOOKUP",
                buy_intent=False,
                info_only=False,
                topic="product",
                entities=constraints,
                missing=[],
                next_action="ANSWER_ONLY",
            )
            self._apply_intent_decision(context, decision)
            logger.info("session=%s intent=%s action=%s", context.session_id, decision.intent, decision.next_action)
            logger.debug(
                "session=%s intent_decision=%s",
                context.session_id,
                json.dumps(decision.__dict__, ensure_ascii=True),
            )
            return

        state = build_intent_state(context.chat_history, context.user_message, context.order_state)
        prompt_template = load_prompt(context.prompts_dir / "intent_detection.txt")
        state_json = json.dumps(state, ensure_ascii=False)
        prompt = prompt_template.replace("<<STATE_JSON>>", state_json).replace("<<MESSAGE>>", context.user_message)
        raw = ""
        try:
            raw = self._gemini.generate_text(prompt, model=context.model_flash, temperature=0.1)
        except Exception:
            raw = ""

        decision = parse_intent_output(raw, context.user_message, state, context.order_state)
        decision = merge_decision_with_resolved(decision, resolved)
        self._apply_intent_decision(context, decision)
        logger.info(
            "session=%s intent=%s buy_intent=%s info_only=%s topic=%s action=%s missing=%s",
            context.session_id,
            decision.intent,
            decision.buy_intent,
            decision.info_only,
            decision.topic,
            decision.next_action,
            decision.missing,
        )
        logger.debug(
            "session=%s intent_decision=%s",
            context.session_id,
            json.dumps(
                {
                    "intent": decision.intent,
                    "buy_intent": decision.buy_intent,
                    "info_only": decision.info_only,
                    "topic": decision.topic,
                    "next_action": decision.next_action,
                    "missing": decision.missing,
                    "entities": decision.entities,
                },
                ensure_ascii=True,
            ),
        )

    def _apply_intent_decision(self, context: PipelineContext, decision: IntentDecision) -> None:
        """Purpose: Apply an IntentDecision into the PipelineContext fields.
        Inputs/Outputs: Inputs are context and decision; mutates context in place.
        Side Effects / State: Updates flags used by retrieval, guards, and generation.
        Dependencies: Used by _step_intent_detection and rule-based paths.
        Failure Modes: None; assumes decision fields are well-formed.
        If Removed: Intent-derived flags remain unset and routing breaks.
        Testing Notes: Provide a sample decision and verify context flags.
        """
        # Map decision fields to context flags and derived booleans.
        entities = decision.entities or {}
        skus = entities.get("skus") or entities.get("sku") or []
        if isinstance(skus, str):
            skus = [skus]
        context.intent_label = decision.intent
        context.buy_intent = decision.buy_intent
        context.is_info_only = decision.info_only
        context.intent_topic = decision.topic
        context.next_action = decision.next_action
        context.intent_entities = entities
        context.intent_missing = decision.missing or []
        context.primary_code = str(entities.get("primary_code") or "")
        context.missing_sku = "sku" in context.intent_missing
        context.missing_quantity = "quantity" in context.intent_missing
        context.missing_contact = "contact" in context.intent_missing
        context.missing_type = "tay_robot" in context.intent_missing
        context.has_code_query = bool(skus) or bool(context.primary_code)
        commercial_action = decision.commercial_action or entities.get("commercial_action") or {}
        entities["commercial_action"] = commercial_action
        context.intent_entities = entities
        llm_collect = bool(commercial_action.get("collect_contact"))
        context.request_contact = (
            decision.next_action == "REQUEST_CONTACT_FORM"
            or (decision.next_action == "COMMERCIAL_NEUTRAL_REPLY" and decision.buy_intent and context.missing_contact)
            or llm_collect
        )
    def _step_resource_retrieval(self, context: PipelineContext) -> None:
        """Purpose: Retrieve catalog items based on intent and slot constraints.
        Inputs/Outputs: Input is PipelineContext; populates items/related_items/display_items.
        Side Effects / State: Reads resource file via ResourceLoader and logs matches.
        Dependencies: Uses exact_lookup_by_code, retrieve_relevant_items, and helper filters.
        Failure Modes: Missing files or JSON errors propagate from ResourceLoader.load.
        If Removed: Generation has no grounded items and will not render products.
        Testing Notes: Use known queries to verify exact lookup and bundle retrieval.
        """
        # Load catalog items and apply intent-specific retrieval logic.
        items, _meta = self._resource_loader.load()
        context.catalog_items = items
        matched: List[ResourceItem] = []
        normalized_msg = normalize_text(context.user_message)
        if context.intent_label == "SLOT_FILL_AMP":
            slot_target = (
                context.intent_entities.get("slot_target_intent")
                or context.order_state.get("last_intent")
                or context.short_memory.get("last_intent")
                or ""
            )
            if slot_target:
                context.intent_entities["slot_fill"] = "amp"
                context.intent_entities["slot_fill_target"] = slot_target
                context.intent_label = slot_target
                if context.short_memory.get("last_topic"):
                    context.intent_topic = context.short_memory.get("last_topic") or context.intent_topic
                if slot_target == "ACCESSORY_BUNDLE_LOOKUP":
                    pending_parts = (context.short_memory.get("pending_request") or {}).get("required_parts") or []
                    if pending_parts and not context.intent_entities.get("required_categories"):
                        context.intent_entities["required_categories"] = pending_parts
                        context.intent_entities["bundle_hint"] = True
            else:
                context.items = []
                logger.info(
                    "session=%s step=resource_retrieval route=slot_fill_no_target",
                    context.session_id,
                )
                return
        if is_type_only_message(context.user_message):
            last_intent = context.order_state.get("last_intent")
            if last_intent in {"LIST", "LIST_REQUEST"}:
                last_group = context.order_state.get("last_group")
                last_constraints = context.order_state.get("last_constraints") or {}
                target_amp = str(last_constraints.get("amp") or "").upper()
                mode = "ROBOT" if ("robot" in normalized_msg or "robotic" in normalized_msg) else "HAND"
                filtered: List[ResourceItem] = []
                for item in items:
                    if last_group and not item_matches_group(item, last_group):
                        continue
                    if target_amp:
                        amp_val = detect_amp_line(f"{item.name} {item.description}")
                        if amp_val and amp_val != target_amp:
                            continue
                    if mode and detect_item_type(item) != mode:
                        continue
                    filtered.append(item)
                context.items = filtered[:6]
                matched_codes = [item.code for item in context.items if item.code]
                logger.info(
                    "session=%s step=resource_retrieval matched=%d codes=%s",
                    context.session_id,
                    len(matched_codes),
                    matched_codes,
                )
                logger.debug(
                    "session=%s retrieval_route=list_type_switch group=%s amp=%s mode=%s codes=%s",
                    context.session_id,
                    last_group or "",
                    target_amp or "",
                    mode,
                    matched_codes,
                )
                return
            selected = context.order_state.get("selected_sku")
            cached = context.order_state.get("last_context_codes") or context.short_memory.get("last_results") or []
            codes = [selected] if selected else list(cached)
            context.items = match_items_by_codes(items, codes) if codes else []
            matched_codes = [item.code for item in context.items if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            logger.debug(
                "session=%s retrieval_route=type_only codes=%s",
                context.session_id,
                matched_codes,
            )
            return
        if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
            explicit_roles = context.intent_entities.get("required_categories") or []
            if not explicit_roles:
                explicit_roles = detect_bundle_required_categories(context.user_message)
            explicit_roles = [str(role).upper() for role in explicit_roles if role]
            bundle_hint = bool(context.intent_entities.get("bundle_hint")) or bool(BUNDLE_QUERY_RE.search(normalized_msg))
            expand_bundle = bool(context.intent_entities.get("expand_bundle"))

            codes, primary_code = extract_codes(context.user_message)
            anchor: Optional[ResourceItem] = None
            if primary_code:
                matched = exact_lookup_by_code(items, primary_code)
                if matched:
                    anchor = matched[0]
            if not anchor:
                selected = context.order_state.get("selected_sku")
                if selected:
                    matched = match_items_by_codes(items, [selected])
                    if matched:
                        anchor = matched[0]
            if not anchor:
                anchor_sku = (context.short_memory.get("last_anchor") or {}).get("sku")
                if anchor_sku:
                    matched = match_items_by_codes(items, [anchor_sku])
                    if matched:
                        anchor = matched[0]
            if not anchor:
                cached = context.order_state.get("last_context_codes") or context.short_memory.get("last_results") or []
                if cached:
                    matched = match_items_by_codes(items, cached)
                    if matched:
                        anchor = matched[0]

            context.items = [anchor] if anchor else []

            anchor_text = f"{anchor.name} {anchor.description}" if anchor else ""
            target_amp = str(context.intent_entities.get("amp") or "").upper()
            target_amp_source = "intent" if target_amp else ""
            if not target_amp:
                target_amp = detect_amp_line(anchor_text)
                if target_amp:
                    target_amp_source = "anchor"
            if not target_amp:
                last_constraints = context.order_state.get("last_constraints") or {}
                if last_constraints.get("amp"):
                    target_amp = str(last_constraints.get("amp")).upper()
                    target_amp_source = "state"
            target_system = str(context.intent_entities.get("system") or "")
            if not target_system:
                target_system = detect_system_tag(anchor_text)
            if not target_system:
                last_constraints = context.order_state.get("last_constraints") or {}
                target_system = last_constraints.get("system") or ""
            torch_type = context.order_state.get("hand_or_robot") or ""
            if context.order_state.get("hand_or_robot_source") == "ASSUMED_DEFAULT":
                torch_type = ""
            if not torch_type:
                torch_type = "HAND"

            anchor_group = detect_item_group(anchor) if anchor else ""
            product_group = context.intent_entities.get("product_group") or anchor_group
            logger.info(
                "session=%s bundle_query_text=%s",
                context.session_id,
                context.user_message,
            )
            logger.info(
                "session=%s bundle_filters amp=%s system=%s anchor_sku=%s product_group=%s",
                context.session_id,
                target_amp or "",
                target_system or "",
                anchor.code if anchor else "",
                product_group or "",
            )

            bundle_roles: List[str] = []
            if expand_bundle:
                bundle_roles = infer_bundle_roles_from_catalog(
                    items,
                    anchor,
                    target_amp,
                    target_system,
                    torch_type if context.order_state.get("hand_or_robot_source") != "ASSUMED_DEFAULT" else "",
                )

            required = merge_unique(explicit_roles, bundle_roles)
            if not required:
                pending = (context.short_memory.get("pending_request") or {}).get("required_parts") or []
                required = pending
            if not required and expand_bundle:
                required = list(DEFAULT_BUNDLE_CATEGORIES)

            bundle_items: List[ResourceItem] = []
            missing_groups: List[str] = []
            ambiguous_groups: List[str] = []
            for group in required:
                candidates = [item for item in items if item_matches_group(item, group)]
                top_entries = build_bundle_top_entries(candidates, target_amp, target_system, torch_type, limit=5)
                logger.info(
                    "session=%s bundle_topk group=%s results=%s",
                    context.session_id,
                    group,
                    json.dumps(top_entries, ensure_ascii=True),
                )
                if not target_amp and has_ambiguous_amp_by_sku(candidates):
                    ambiguous_groups.append(group)
                filtered: List[ResourceItem] = []
                for item in candidates:
                    amp_val = detect_amp_line(f"{item.name} {item.description}")
                    if target_amp and amp_val and amp_val != target_amp:
                        continue
                    system_val = detect_system_tag(f"{item.name} {item.description}")
                    if target_system and system_val and system_val != target_system:
                        continue
                    if torch_type and detect_item_type(item) != torch_type:
                        continue
                    filtered.append(item)
                filtered = dedupe_by_sku(filtered)
                if not filtered:
                    missing_groups.append(group)
                    continue
                bundle_items.extend(filtered[:2])

            if target_amp:
                context.intent_entities["amp"] = target_amp
            if target_system:
                context.intent_entities["system"] = target_system
            context.related_items = dedupe_by_sku(bundle_items)
            context.intent_entities["required_categories"] = required
            context.intent_entities["missing_categories"] = missing_groups
            context.intent_entities["ambiguous_categories"] = ambiguous_groups
            matched_codes = [item.code for item in context.related_items if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            logger.debug(
                "session=%s retrieval_route=accessory_bundle required=%s missing=%s ambiguous=%s anchor=%s amp_source=%s",
                context.session_id,
                required,
                missing_groups,
                ambiguous_groups,
                anchor.code if anchor else "",
                target_amp_source or "",
            )
            return
        if context.intent_label == "ACCESSORY_LOOKUP":
            codes, primary_code = extract_codes(context.user_message)
            matched = []
            if primary_code:
                matched = exact_lookup_by_code(items, primary_code)
            if not matched and codes:
                matched = match_items_by_codes(items, codes)
            if not matched:
                matched = retrieve_relevant_items(context.user_message, items)
            context.items = matched
            matched_codes = [item.code for item in matched if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            logger.debug(
                "session=%s retrieval_route=accessory_lookup codes=%s",
                context.session_id,
                matched_codes,
            )
            return
        if context.intent_label in {"LIST", "LIST_REQUEST"} or (
            context.intent_topic == "list" and LISTING_RE.search(normalized_msg)
        ):
            group = detect_product_group(normalized_msg)
            if not group:
                group = (
                    context.order_state.get("selected_group")
                    or (context.short_memory.get("last_anchor") or {}).get("cat")
                    or context.order_state.get("last_group")
                )
            target_amp = "350A" if "350" in normalized_msg else ("500A" if "500" in normalized_msg else "")
            if not target_amp:
                target_amp = str(context.intent_entities.get("amp") or "")
            if not target_amp:
                target_amp = str((context.short_memory.get("last_anchor") or {}).get("line_amp") or "")
            mode = "HAND"
            if "robot" in normalized_msg or "robotic" in normalized_msg:
                mode = "ROBOT"
            elif "tay" in normalized_msg or "hand" in normalized_msg:
                mode = "HAND"
            elif context.order_state.get("hand_or_robot") in {"ROBOT", "HAND"}:
                mode = str(context.order_state.get("hand_or_robot"))
            filtered: List[ResourceItem] = []
            for item in items:
                if group:
                    item_group = detect_product_group(
                        normalize_text(f"{item.name} {item.description} {item.category}")
                    )
                    if item_group != group:
                        continue
                if target_amp:
                    amp_val = detect_amp_line(f"{item.name} {item.description}")
                    if amp_val != target_amp:
                        continue
                if mode and detect_item_type(item) != mode:
                    continue
                filtered.append(item)
            context.items = filtered[:6]
            matched_codes = [item.code for item in context.items if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            logger.debug(
                "session=%s retrieval_route=list group=%s amp=%s codes=%s",
                context.session_id,
                group or "",
                target_amp,
                matched_codes,
            )
            return
        if context.intent_label == "PRODUCT_LOOKUP":
            group = str(context.intent_entities.get("product_group") or "")
            target_amp = str(context.intent_entities.get("amp") or "").upper()
            target_size = context.intent_entities.get("size")
            target_len = context.intent_entities.get("length")
            if isinstance(target_size, str):
                try:
                    target_size = float(target_size)
                except ValueError:
                    target_size = None
            if isinstance(target_len, str):
                try:
                    target_len = int(float(target_len))
                except ValueError:
                    target_len = None

            candidates = [item for item in items if group and item_matches_group(item, group)]

            filtered: List[ResourceItem] = []
            for item in candidates:
                amp = item_amp(item)
                if target_amp and amp and amp != target_amp:
                    continue
                size = item_size(item)
                if target_size is not None and size is not None and float(size) != float(target_size):
                    continue
                length = item_length(item)
                if target_len is not None and length is not None and int(length) != int(target_len):
                    continue
                filtered.append(item)

            context.items = filtered
            matched_codes = [item.code for item in filtered if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            return
        skip_query = is_pure_quantity_message(context.user_message) or is_quantity_followup_message(context.user_message)
        exact_only = False
        if context.intent_label == "CODE_LOOKUP":
            primary_code = context.primary_code or extract_codes(context.user_message)[1]
            context.primary_code = primary_code
            if primary_code:
                matched = exact_lookup_by_code(items, primary_code)
            context.previous_codes = []
            exact_only = True
        if skip_query:
            selected_sku = str(context.order_state.get("selected_sku") or "")
            if selected_sku:
                matched = match_items_by_codes(items, [selected_sku])
                context.previous_codes = [selected_sku]
            context.items = matched
            matched_codes = [item.code for item in matched if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            return
        if exact_only:
            context.items = matched
            matched_codes = [item.code for item in matched if item.code]
            logger.info(
                "session=%s step=resource_retrieval matched=%d codes=%s",
                context.session_id,
                len(matched_codes),
                matched_codes,
            )
            return
        if not matched and not skip_query:
            matched = retrieve_relevant_items(context.user_message, items)

        if not matched and context.chat_history:
            cached_codes = []
            for message in reversed(context.chat_history):
                meta = message.get("meta") or {}
                codes = meta.get("context_codes")
                if isinstance(codes, list) and codes:
                    cached_codes = codes
                    break
            if cached_codes:
                code_set = {normalize_text(str(code)) for code in cached_codes if code}
                matched = [item for item in items if normalize_text(item.code) in code_set]
                context.previous_codes = cached_codes

        if not context.previous_codes and context.chat_history:
            cached_codes = []
            for message in context.chat_history:
                if message.get("role") != "assistant":
                    continue
                meta = message.get("meta") or {}
                codes = meta.get("context_codes")
                if isinstance(codes, list) and codes:
                    cached_codes.extend(codes)
            context.previous_codes = cached_codes

        if not matched and context.chat_history:
            for message in reversed(context.chat_history):
                content = message.get("content", "")
                if not content:
                    continue
                past = retrieve_relevant_items(content, items)
                if past:
                    matched = past
                    break

        context.items = matched
        matched_codes = [item.code for item in matched if item.code]
        logger.info(
            "session=%s step=resource_retrieval matched=%d codes=%s",
            context.session_id,
            len(matched_codes),
            matched_codes,
        )

    def _step_context_guard(self, context: PipelineContext) -> None:
        """Purpose: Apply deterministic guardrails and derive rendering flags.
        Inputs/Outputs: Input is PipelineContext; mutates should_* flags and state.
        Side Effects / State: Updates order_state based on the current turn.
        Dependencies: Uses regex rules, update_order_state_from_turn, and intent fields.
        Failure Modes: None; logic is rule-based.
        If Removed: LLM output may violate business rules and routing becomes unstable.
        Testing Notes: Verify should_show_form/should_render_products across scenarios.
        """
        # Compute rule-based flags before generation.
        normalized_msg = normalize_text(context.user_message)
        context.is_asking_related = bool(RELATED_QUERY_RE.search(normalized_msg))
        context.is_availability_query = is_availability_query(context.user_message)
        context.is_single_unit = is_single_unit_request(context.user_message)
        context.has_code_query = context.has_code_query or bool(CODE_RE.search(context.user_message))
        context.is_info_only = context.is_info_only or is_info_only_query(context.user_message)
        if (
            LISTING_RE.search(normalized_msg)
            or CODE_RE.search(context.user_message)
            or RELATED_QUERY_RE.search(normalized_msg)
            or extract_quantity(normalized_msg) is not None
        ):
            context.is_info_only = False
        if context.is_availability_query:
            context.is_asking_related = False
        context.is_info_query = bool(INFO_RE.search(normalized_msg))
        context.is_asking_price = context.is_asking_price or bool(PRICE_RE.search(normalized_msg))
        context.is_close_intent = context.buy_intent or bool(CLOSE_INTENT_RE.search(normalized_msg)) or context.is_asking_price
        context.has_asked_type = any(
            msg.get("role") == "assistant"
            and "tay" in normalize_text(msg.get("content", ""))
            and "robot" in normalize_text(msg.get("content", ""))
            and (
                "hay" in normalize_text(msg.get("content", ""))
                or "hoac" in normalize_text(msg.get("content", ""))
            )
            for msg in context.chat_history
        )
        context.has_default_hand_note = any(
            DEFAULT_HAND_NOTE in (msg.get("content", "") or "")
            for msg in context.chat_history
            if msg.get("role") == "assistant"
        )
        (
            context.has_asked_form,
            context.reminder_count,
            context.contact_received,
            context.waiting_for_contact,
        ) = _get_contact_state(context.chat_history)
        context.has_asked_form = context.has_asked_form or bool(context.order_state.get("asked_contact_form"))
        context.has_contact_info = (
            context.contact_received or detect_contact_info(context.user_message) or bool(context.order_state.get("contact"))
        )
        context.has_answered_type = bool(TYPE_ANSWER_RE.search(normalized_msg))
        context.has_asked_type = context.has_asked_type or bool(context.order_state.get("asked_hand_robot"))
        context.should_ask_type = (
            not context.has_asked_type
            and not context.has_answered_type
            and not context.is_asking_related
            and not context.is_availability_query
            and not context.is_info_query
            and not context.is_info_only
        )
        context.force_default_hand = context.has_asked_type and not context.has_answered_type
        context.waiting_for_contact = context.waiting_for_contact and not context.has_contact_info
        context.should_show_form = (
            context.is_close_intent
            and not context.should_ask_type
            and not context.has_asked_form
            and not context.has_contact_info
            and not context.is_single_unit
        )
        context.should_remind_contact = (
            context.waiting_for_contact
            and context.is_close_intent
            and not context.should_ask_type
            and context.reminder_count < 1
            and not context.is_info_query
            and not context.is_info_only
        )

        context.should_repeat_products = bool(REPEAT_REQUEST_RE.search(normalized_msg))
        context.should_render_products = (
            not context.is_info_only
            and (
                context.is_asking_related
                or context.has_code_query
                or bool(LISTING_RE.search(normalized_msg))
                or bool(PRODUCT_INFO_RE.search(normalized_msg))
                or (context.is_close_intent and context.has_code_query)
            )
        )
        if context.is_availability_query:
            context.should_render_products = bool(context.items)

        if context.intent_label:
            context.is_info_only = context.is_info_only or context.intent_topic == "origin"
            if context.buy_intent:
                context.is_close_intent = True
            if context.intent_topic == "commercial":
                context.is_asking_price = True
            if context.intent_entities.get("skus"):
                context.has_code_query = True
            if context.next_action == "ASK_HAND_VS_ROBOT_ONCE":
                context.should_ask_type = not context.has_asked_type and not context.has_answered_type
            elif context.next_action in {"ANSWER_ONLY", "ASK_FOR_SKU_OR_GROUP", "REQUEST_CONTACT_FORM"}:
                if context.intent_label != "PRODUCT_AVAILABILITY":
                    context.should_ask_type = False
            if context.intent_label == "PRODUCT_AVAILABILITY" and context.missing_type:
                context.should_ask_type = not context.has_asked_type and not context.has_answered_type
            if context.request_contact:
                context.should_show_form = (
                    not context.has_asked_form
                    and not context.has_contact_info
                    and not context.is_single_unit
                )
            if context.next_action in {"ASK_FOR_SKU_OR_GROUP", "REQUEST_CONTACT_FORM", "COMMERCIAL_NEUTRAL_REPLY"}:
                context.should_render_products = False
            if (context.is_asking_price or context.is_availability_query or context.intent_topic == "commercial") and context.items:
                context.should_render_products = context.has_code_query or bool(context.order_state.get("selected_sku"))
            if context.is_info_only and context.intent_label not in {"LIST", "LIST_REQUEST"}:
                context.should_render_products = False
            if context.intent_label == "CODE_LOOKUP" and not COMPATIBILITY_RE.search(normalized_msg):
                context.is_asking_related = False

        if (
            not context.is_asking_related
            and context.intent_label != "CODE_LOOKUP"
            and context.items
            and ("bec" in normalized_msg or "tip" in normalized_msg)
        ):
            context.is_asking_related = True

        update_order_state_from_turn(context)
        if context.intent_label in {"LIST", "LIST_REQUEST"} and not context.order_state.get("hand_or_robot"):
            context.order_state["hand_or_robot"] = "HAND"
            if not context.order_state.get("hand_or_robot_source"):
                context.order_state["hand_or_robot_source"] = "ASSUMED_DEFAULT"
        commercial_action = context.intent_entities.get("commercial_action") or {}
        quantity_value = context.order_state.get("quantity") or context.intent_entities.get("quantity")
        bulk_threshold = get_bulk_qty_threshold(context.catalog_items)
        if bulk_threshold and isinstance(quantity_value, int) and quantity_value >= bulk_threshold:
            commercial_action["collect_contact"] = True
            commercial_action.setdefault("reason", "bulk_quantity_order")
            context.intent_entities["commercial_action"] = commercial_action
            context.request_contact = True
        has_selected = bool(context.order_state.get("selected_sku") or context.order_state.get("selected_group"))
        context.has_code_query = context.has_code_query or bool(context.order_state.get("selected_sku"))
        quantity_present = context.order_state.get("quantity") is not None
        contact_missing = not context.order_state.get("contact")
        if context.order_state.get("asked_contact_form") and contact_missing:
            context.waiting_for_contact = True
        if has_selected:
            context.missing_sku = False
        if quantity_present:
            context.missing_quantity = False
        if not contact_missing:
            context.missing_contact = False
        context.should_show_form = (
            has_selected
            and quantity_present
            and contact_missing
            and not context.is_single_unit
            and not context.is_info_only
            and (context.buy_intent or context.is_close_intent or context.request_contact)
        )
        if context.next_action == "ASK_FOR_SKU_OR_GROUP":
            context.should_show_form = False
        if context.is_asking_price or context.is_availability_query:
            context.should_show_form = False
        if (
            (is_pure_quantity_message(context.user_message) or is_quantity_followup_message(context.user_message))
            and has_selected
            and quantity_present
            and contact_missing
            and not context.is_single_unit
            and not context.is_info_only
        ):
            context.should_show_form = True
        context.should_remind_contact = (
            context.waiting_for_contact
            and (context.buy_intent or context.is_close_intent)
            and not context.should_ask_type
            and context.reminder_count < 1
            and not context.is_info_query
            and not context.is_info_only
        )
        if context.is_asking_price or context.is_availability_query:
            context.should_remind_contact = False
        if context.intent_label == "CODE_LOOKUP":
            context.should_render_products = bool(context.items)
            context.should_ask_type = False
        if context.is_asking_price or context.is_availability_query or context.intent_topic == "commercial":
            if context.items:
                context.should_ask_type = False
            else:
                context.should_ask_type = not context.has_asked_type and not context.has_answered_type
        if context.intent_label == "PRODUCT_LOOKUP":
            context.is_asking_price = False
            context.is_availability_query = False
            context.is_close_intent = False
            context.should_show_form = False
            context.should_remind_contact = False
            context.should_ask_type = False
            context.should_render_products = True
            context.is_asking_related = False
        if context.intent_label == "TYPE_SWITCH":
            context.is_asking_price = False
            context.is_availability_query = False
            context.is_close_intent = False
            context.should_show_form = False
            context.should_remind_contact = False
            context.should_ask_type = False
            context.should_render_products = True
            context.is_asking_related = False
        if context.intent_label == "ACCESSORY_LOOKUP":
            context.is_asking_price = False
            context.is_availability_query = False
            context.is_close_intent = False
            context.should_show_form = False
            context.should_remind_contact = False
            context.should_ask_type = False
            context.should_render_products = True
            context.is_asking_related = True
        if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
            context.is_asking_price = False
            context.is_availability_query = False
            context.is_close_intent = False
            context.should_show_form = False
            context.should_remind_contact = False
            context.should_ask_type = False
            context.should_render_products = True
            context.is_asking_related = True
        if context.intent_label in {"LIST", "LIST_REQUEST"}:
            context.should_ask_type = False

        if context.intent_label != "ACCESSORY_BUNDLE_LOOKUP":
            context.related_items = []
        if (
            context.items
            and context.is_asking_related
            and not context.is_availability_query
            and context.intent_label != "ACCESSORY_BUNDLE_LOOKUP"
        ):
            main_item = context.items[0]
            main_text = f"{main_item.name} {main_item.description}"
            target_amp = detect_amp_line(main_text)
            if not target_amp:
                last_constraints = context.order_state.get("last_constraints") or {}
                target_amp = last_constraints.get("amp") or ""
            target_system = detect_system_tag(main_text)

            related = []
            for item in context.catalog_items:
                cat_norm = _normalize_category(item.category)
                if cat_norm not in RELATED_CATEGORIES:
                    continue
                combined = f"{item.name} {item.description}"
                amp_val = detect_amp_line(combined)
                item_system = detect_system_tag(combined)
                if target_amp and amp_val and amp_val != target_amp:
                    continue
                if target_system and item_system != target_system:
                    continue
                if not target_system and item_system:
                    continue
                related.append(item)

            context.related_items = related[:6]

        context.all_items = _dedupe_items(context.items + context.related_items)
        context.display_items = context.all_items
        if context.previous_codes and not context.should_repeat_products:
            prev_set = {normalize_text(code) for code in context.previous_codes}
            context.display_items = [
                item for item in context.display_items if normalize_text(item.code) not in prev_set
            ]
        if not context.should_render_products:
            context.display_items = []

        if context.items:
            context.should_ask_type = False

        logger.info(
            "session=%s step=context_guard info_only=%s should_render=%s ask_type=%s show_form=%s remind=%s",
            context.session_id,
            context.is_info_only,
            context.should_render_products,
            context.should_ask_type,
            context.should_show_form,
            context.should_remind_contact,
        )
        logger.debug(
            "session=%s context_guard=%s",
            context.session_id,
            json.dumps(
                {
                    "intent": context.intent_label,
                    "topic": context.intent_topic,
                    "next_action": context.next_action,
                    "buy_intent": context.buy_intent,
                    "is_info_only": context.is_info_only,
                    "is_close_intent": context.is_close_intent,
                    "is_asking_price": context.is_asking_price,
                    "is_availability_query": context.is_availability_query,
                    "should_render_products": context.should_render_products,
                    "should_show_form": context.should_show_form,
                    "should_ask_type": context.should_ask_type,
                    "items_count": len(context.items),
                    "related_count": len(context.related_items),
                    "order_state": sanitize_state_for_log(context.order_state),
                },
                ensure_ascii=True,
            ),
        )

    def _step_generation(self, context: PipelineContext) -> None:
        """Purpose: Generate the final answer text and image placements.
        Inputs/Outputs: Input is PipelineContext; outputs answer_text and images.
        Side Effects / State: May call Gemini and update context display_items.
        Dependencies: Uses prompts, gemini client, and render/post-process helpers.
        Failure Modes: Gemini errors return a fallback apology message.
        If Removed: The API cannot produce responses.
        Testing Notes: Run sample queries for each route (lookup, bundle, list).
        """
        # Route through deterministic branches before invoking the LLM.
        normalized_msg = normalize_text(context.user_message)
        if is_ask_selling_scope(context.user_message):
            context.answer_text = get_selling_scope_response(context.order_state)
            logger.info("session=%s step=generation route=ask_selling_scope", context.session_id)
            return
        if context.is_single_unit:
            context.answer_text = NO_RETAIL_REPLY
            logger.info("session=%s step=generation route=no_retail", context.session_id)
            return
        if context.intent_label == "SLOT_FILL_AMP":
            context.answer_text = (
                "Dạ vâng ạ, Anh/Chị cho em xin mã Tokin hoặc quy cách đang hỏi để em đối chiếu đúng ạ."
            )
            context.images = []
            logger.info("session=%s step=generation route=slot_fill_need_anchor", context.session_id)
            return
        if is_type_only_message(context.user_message):
            normalized = normalize_text(context.user_message)
            mode = "ROBOT" if "robot" in normalized else "HAND"
            context.order_state["hand_or_robot"] = mode
            if context.items:
                filtered = [item for item in context.items if detect_item_type(item) == mode] or context.items
                last_constraints = context.order_state.get("last_constraints") or {}
                target_amp = last_constraints.get("amp") or detect_amp_line(
                    " ".join(f"{item.name} {item.description}" for item in context.items)
                )
                if target_amp:
                    amp_filtered = [
                        item
                        for item in filtered
                        if detect_amp_line(f"{item.name} {item.description}") == target_amp
                    ]
                    if amp_filtered:
                        filtered = amp_filtered

                lines = render_product_lookup_lines(filtered, limit=2)
                origin = "Xuất xứ: Tokinarc – Nhật Bản"
                note = (
                    "Dạ vâng ạ, em đã chuyển sang cấu hình **súng Robot**. "
                    "Em sẽ đối chiếu lại để chọn đúng mã phù hợp theo cổ súng/chuẩn Robot ạ."
                    if mode == "ROBOT"
                    else
                    "Dạ vâng ạ, em đang theo cấu hình **súng hàn tay** thông dụng. "
                    "Nếu Anh/Chị đổi sang Robot, em sẽ đối chiếu lại mã phù hợp ạ."
                )
                context.answer_text = "\n\n".join(
                    part for part in ["\n".join(lines).strip(), origin, note] if part
                ).strip()
                context.images = []
                logger.info("session=%s step=generation route=type_switch", context.session_id)
                return

            context.answer_text = (
                "Dạ vâng ạ, Anh/Chị cho em xin mã Tokin hoặc quy cách đang hỏi để em đối chiếu đúng bản Robot ạ."
                if mode == "ROBOT"
                else "Dạ vâng ạ, Anh/Chị cho em xin mã Tokin hoặc quy cách đang hỏi để em đối chiếu đúng bản tay ạ."
            )
            context.images = []
            logger.info("session=%s step=generation route=type_switch_no_cache", context.session_id)
            return
        if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
            anchor = context.items[0] if context.items else None
            required_groups = context.intent_entities.get("required_categories") or []
            missing_groups = context.intent_entities.get("missing_categories") or []
            ambiguous_groups = context.intent_entities.get("ambiguous_categories") or []
            if not anchor:
                context.answer_text = (
                    "Dạ vâng ạ, Anh/Chị cho em xin mã Tokin hoặc quy cách đang hỏi để em đối chiếu đúng linh kiện đi kèm ạ."
                )
                context.images = []
                logger.info("session=%s step=generation route=accessory_bundle_need_anchor", context.session_id)
                return
            if not required_groups:
                context.answer_text = (
                    "Dạ vâng ạ, Anh/Chị cần linh kiện đi kèm nào để em đối chiếu đúng ạ."
                )
                context.images = []
                logger.info("session=%s step=generation route=accessory_bundle_need_parts", context.session_id)
                return
            if anchor and (context.related_items or missing_groups):
                origin = "Xuất xứ: Tokinarc – Nhật Bản 🇯🇵"
                note = build_hand_robot_note([anchor])
                target_amp = str(context.intent_entities.get("amp") or "")
                target_system = str(context.intent_entities.get("system") or "")
                context.answer_text = render_accessory_lookup(
                    context.user_message,
                    anchor,
                    context.related_items,
                    origin,
                    note,
                    target_groups=required_groups,
                    missing_groups=missing_groups,
                    ambiguous_groups=ambiguous_groups,
                    target_amp=target_amp,
                    target_system=target_system,
                )
                context.images = []
                logger.info("session=%s step=generation route=accessory_bundle_rule", context.session_id)
                return
        is_commercial_or_availability = (
            context.is_asking_price or context.is_availability_query or context.intent_topic == "commercial"
        )
        if context.intent_label in {
            "PRODUCT_LOOKUP",
            "CODE_LOOKUP",
            "ACCESSORY_LOOKUP",
            "ACCESSORY_BUNDLE_LOOKUP",
            "LIST",
            "LIST_REQUEST",
        }:
            if context.intent_label == "PRODUCT_LOOKUP":
                context.display_items = context.items[:2] if context.items else []
                if not context.items:
                    context.should_render_products = False
            elif context.intent_label == "CODE_LOOKUP":
                context.display_items = context.items[:3] if context.items else []
                if not context.items:
                    context.should_render_products = False
            elif context.intent_label == "ACCESSORY_LOOKUP":
                context.display_items = context.related_items[:4] if context.related_items else []
                if not context.related_items:
                    context.should_render_products = False
            elif context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
                context.display_items = dedupe_by_sku(context.related_items) if context.related_items else []
                if not context.related_items:
                    context.should_render_products = False
            else:
                context.display_items = context.items[:4] if context.items else []
                if not context.display_items:
                    context.should_render_products = False
        if is_commercial_or_availability and not context.items and context.intent_label not in {
            "PRODUCT_LOOKUP",
            "CODE_LOOKUP",
            "ACCESSORY_LOOKUP",
            "LIST",
            "LIST_REQUEST",
        }:
            if context.should_ask_type:
                answer = f"{ASK_TYPE_QUESTION}\n\n{DEFAULT_PRICE_REPLY}"
            else:
                answer = DEFAULT_PRICE_REPLY
            context.answer_text = answer
            logger.info("session=%s step=generation route=commercial_no_match", context.session_id)
            return
        if context.is_info_only:
            context.answer_text = build_info_only_response(context.user_message)
            logger.info("session=%s step=generation route=info_only", context.session_id)
            return
        if is_pure_quantity_message(context.user_message) or is_quantity_followup_message(context.user_message):
            context.intent_entities["is_quantity_followup"] = True
            qstate = build_quantity_context_json(context)
            prompt_template = load_prompt(context.prompts_dir / "quantity_followup.txt")
            prompt = (
                prompt_template.replace("<<STATE_JSON>>", json.dumps(qstate, ensure_ascii=False))
                .replace("<<MESSAGE>>", context.user_message)
            )
            try:
                answer = self._gemini.generate_text(prompt, model=context.model_flash, temperature=0.1)
            except Exception:
                answer = ""
            answer = (answer or "").strip()
            if not answer:
                if qstate.get("selected_sku"):
                    answer = (
                        f"Dạ em đã ghi nhận số lượng {qstate.get('quantity')} cho {qstate.get('selected_sku')} ạ."
                    )
                elif qstate.get("quantity"):
                    answer = (
                        f"Dạ vâng ạ, mình cần {qstate.get('quantity')} cái mã nào để em ghi nhận đúng ạ?"
                    )
                else:
                    answer = "Dạ vâng ạ, mình cần số lượng và mã nào để em ghi nhận đúng ạ?"

            if context.should_show_form:
                answer = ensure_contains_form_and_tail(
                    answer,
                    qstate.get("form_block", FORM_BLOCK),
                    qstate.get("required_tail_sentence", ""),
                )
            else:
                answer = remove_form_block(answer)
                answer = remove_contact_reminder(answer)
            answer = insert_stock_line(
                answer,
                qstate.get("stock_line", ""),
                qstate.get("form_block", FORM_BLOCK),
            )
            context.answer_text = answer
            logger.info("session=%s step=generation route=quantity_followup_llm", context.session_id)
            return
        if (
            context.intent_label not in {"PRODUCT_AVAILABILITY", "CODE_LOOKUP"}
            and not is_commercial_or_availability
            and context.missing_quantity
            and (
                context.order_state.get("selected_sku") or context.order_state.get("selected_group") or context.has_code_query
            )
        ):
            target = context.order_state.get("selected_sku") or context.order_state.get("selected_group")
            if target:
                context.answer_text = f"Dạ Anh/Chị cho em xin số lượng dự kiến cho {target} ạ."
            else:
                context.answer_text = "Dạ Anh/Chị cho em xin số lượng dự kiến ạ."
            logger.info("session=%s step=generation route=ask_quantity", context.session_id)
            return
        if context.next_action == "ASK_FOR_SKU_OR_GROUP" and context.intent_label != "CODE_LOOKUP":
            context.answer_text = ASK_SKU_GROUP_REPLY
            logger.info("session=%s step=generation route=ask_sku_group", context.session_id)
            return
        if (
            context.next_action == "ASK_HAND_VS_ROBOT_ONCE"
            and context.intent_label != "CODE_LOOKUP"
            and context.should_ask_type
        ):
            context.answer_text = ASK_TYPE_QUESTION
            logger.info("session=%s step=generation route=ask_type", context.session_id)
            return
        if context.next_action == "COMMERCIAL_NEUTRAL_REPLY" and context.intent_label != "CODE_LOOKUP":
            cards = render_product_cards(context.display_items or context.items, limit=3, include_type_line=False)
            answer = cards or ""
            answer = ensure_neutral_sentence(answer)
            context.answer_text = answer
            logger.info("session=%s step=generation route=commercial_reply", context.session_id)
            return
        if context.next_action == "REQUEST_CONTACT_FORM" and context.intent_label != "CODE_LOOKUP":
            if context.should_show_form:
                context.answer_text = FORM_BLOCK
                logger.info("session=%s step=generation route=request_contact", context.session_id)
                return
            if context.missing_sku:
                context.answer_text = ASK_SKU_GROUP_REPLY
                logger.info("session=%s step=generation route=ask_sku_group", context.session_id)
                return
            if context.missing_quantity:
                context.answer_text = AVAILABILITY_NEED_QTY_REPLY
                logger.info("session=%s step=generation route=ask_quantity", context.session_id)
                return
        if context.is_info_query:
            context.answer_text = build_info_response(context)
            logger.info("session=%s step=generation route=info_query", context.session_id)
            return

        if context.should_render_products and not context.display_items and context.previous_codes:
            context.answer_text = REPEAT_BLOCK_REPLY
            logger.info("session=%s step=generation route=repeat_block", context.session_id)
            return

        if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
            product_data = build_bundle_product_data(context.related_items)
        else:
            product_data = build_product_data(context.display_items)
        base_item = context.items[0] if context.items else None
        base_item_line = ""
        if base_item:
            base_amp = detect_amp_line(f"{base_item.name} {base_item.description}")
            base_system = detect_system_tag(f"{base_item.name} {base_item.description}")
            base_item_line = (
                "BASE_ITEM: "
                f"SKU={base_item.code or ''}; "
                f"NAME={base_item.name or ''}; "
                f"CAT={base_item.category or ''}; "
                f"AMP={base_amp}; "
                f"SYSTEM={base_system}"
            )
        bundle_required = context.intent_entities.get("required_categories") or []
        bundle_missing = context.intent_entities.get("missing_categories") or []
        bundle_ambiguous = context.intent_entities.get("ambiguous_categories") or []
        bundle_lines = []
        if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
            if bundle_required:
                bundle_lines.append(f"BUNDLE_REQUIRED: {', '.join(bundle_required)}")
            if bundle_missing:
                bundle_lines.append(f"BUNDLE_MISSING: {', '.join(bundle_missing)}")
            if bundle_ambiguous:
                bundle_lines.append(f"BUNDLE_AMBIGUOUS: {', '.join(bundle_ambiguous)}")
        bundle_info = "\n".join(bundle_lines)
        status_lines = [
            f"- ĐÃ HỎI TAY/ROBOT TRƯỚC ĐÓ: {'RỒI' if context.has_asked_type else 'CHƯA'}",
            f"- CẦN HỎI TAY/ROBOT BÂY GIỜ: {'CÓ' if context.should_ask_type else 'KHÔNG'}",
            f"- MẶC ĐỊNH SÚNG TAY: {'CÓ' if context.force_default_hand else 'KHÔNG'}",
            f"- HIỆN FORM THÔNG TIN: {'CÓ' if context.should_show_form else 'KHÔNG'}",
        ]
        user_wants_specs = bool(PRODUCT_INFO_RE.search(normalized_msg) or INFO_RE.search(normalized_msg))

        code_type = detect_code_type(context.user_message, context.primary_code)
        user_prompt = (
            "DỮ LIỆU CỬA HÀNG:\n"
            f"{product_data}\n\n"
            f"{base_item_line}\n\n"
            f"{bundle_info}\n\n"
            f"INTENT: {context.intent_label}\n"
            f"CODE_TYPE: {code_type}\n"
            f"user_wants_specs: {str(user_wants_specs).lower()}\n"
            f"allow_show_products: {str(context.should_render_products).lower()}\n"
            f"neutral_commercial_sentence: {DEFAULT_PRICE_REPLY}\n"
            f"show_form: {str(context.should_show_form).lower()}\n\n"
            f'TIN NHẮN HIỆN TẠI: "{context.user_message}"\n'
            "TRẠNG THÁI:\n"
            f"{chr(10).join(status_lines)}\n\n"
            "YÊU CẦU:\n"
            "1. Trả lời ngay về linh kiện đi kèm nếu khách hỏi.\n"
            "2. Nếu MẶC ĐỊNH SÚNG TAY là CÓ, hãy tư vấn dòng súng Tay và ghi chú rõ như quy tắc 3.\n"
            f'3. Khi khách hỏi giá/kho, chỉ dùng câu: "{DEFAULT_PRICE_REPLY}"\n'
            '4. Chỉ xin thông tin liên hệ khi dòng "HIỆN FORM THÔNG TIN: CÓ".\n'
            f"5. Nếu không được yêu cầu hiển thị sản phẩm, KHÔNG liệt kê SKU, hình ảnh hay thông số."
        )

        history_contents = []
        for message in context.chat_history:
            content = message.get("content", "")
            if not content:
                continue
            role = "user" if message.get("role") == "user" else "model"
            history_contents.append({"role": role, "parts": [{"text": content}]})

        contents = history_contents + [{"role": "user", "parts": [{"text": user_prompt}]}]

        system_instruction = load_prompt(context.prompts_dir / "answer_generation.txt")
        try:
            answer = self._gemini.generate_content(
                contents,
                model=context.model_flash,
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=10024,
            )
        except Exception:
            context.answer_text = "Dạ kết nối gián đoạn, Anh/Chị nhắn lại giúp em nhé! 🙏"
            logger.info("session=%s step=generation route=exception", context.session_id)
            return

        answer = answer.strip() or "Dạ Em ghi nhận yêu cầu và sẽ phản hồi Anh/Chị ngay ạ."

        if not context.display_items:
            codes, _ = extract_codes(answer)
            if not codes:
                codes = extract_skus(answer)
            if codes:
                matched = match_items_by_codes(context.catalog_items, codes)
                if matched:
                    context.display_items = matched
                    context.should_render_products = True

        if context.intent_label == "CODE_LOOKUP":
            code_type = detect_code_type(context.user_message, context.primary_code)
            if code_type == "TOKIN":
                answer = enforce_tokin_code_wording(answer, context.primary_code)

        if is_commercial_or_availability:
            answer = remove_quantity_request(answer)
            answer = remove_form_block(answer)
            answer = remove_contact_reminder(answer)
            answer = remove_commercial_commitments(answer)
            answer = ensure_product_cards(answer, context.display_items, include_type_line=False)
            answer = ensure_neutral_sentence(answer)
        if context.intent_label == "CODE_LOOKUP" and context.intent_topic == "commercial":
            answer = append_line_if_missing(answer, DEFAULT_PRICE_REPLY, "ghi nhan nhu cau")
        if context.intent_label == "CODE_LOOKUP" and context.buy_intent and context.missing_quantity:
            target = context.order_state.get("selected_sku") or context.order_state.get("selected_group")
            answer = append_quantity_question(answer, target)

        if (
            context.force_default_hand
            and not context.has_default_hand_note
            and not context.is_availability_query
            and not context.is_asking_price
            and context.intent_topic != "commercial"
            and DEFAULT_HAND_NOTE not in answer
        ):
            answer = f"{answer.strip()}\n\n{DEFAULT_HAND_NOTE}"

        if context.should_ask_type and ASK_TYPE_QUESTION not in answer:
            answer = f"{answer.strip()}\n\n{ASK_TYPE_QUESTION}"
        if context.has_asked_type:
            answer = remove_type_question(answer)
        if context.intent_label in {"LIST", "LIST_REQUEST", "ACCESSORY_BUNDLE_LOOKUP"}:
            answer = remove_type_question(answer)
        if context.has_default_hand_note:
            answer = remove_default_hand_note(answer)

        if context.should_show_form:
            answer = append_form_if_missing(answer)
        else:
            answer = remove_form_block(answer)

        if context.should_remind_contact:
            answer = append_reminder_if_missing(answer)
        else:
            answer = remove_contact_reminder(answer)

        if (
            context.intent_label in TECHNICAL_INTENTS
            and not context.should_show_form
            and not is_commercial_or_availability
        ):
            answer = remove_handoff_phrases(answer)
            answer = ensure_technical_closing_line(answer, context)

        if context.should_render_products:
            answer = convert_raw_image_links_to_markdown(answer)
        if not context.should_render_products:
            answer = remove_product_lines(answer)
            answer = remove_markdown_images(answer)
            images = []
        else:
            if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
                images = []
            else:
                answer = prune_repeated_product_lines(
                    answer,
                    context.user_message,
                    context.previous_codes,
                    allow_repeat=context.should_repeat_products,
                )
                answer = dedupe_sku_lines(answer)
                answer, images = insert_images_after_mentions(answer, context.display_items, context.max_images)
                answer = insert_missing_image_notice(answer, context.display_items)
                if context.previous_codes and not context.display_items and not context.should_repeat_products:
                    answer = REPEAT_BLOCK_REPLY
        context.answer_text = answer
        context.images = images
        context.asked_form = has_form_block(answer)
        context.reminded_contact = has_contact_reminder(answer)

    def _step_finalize(self, context: PipelineContext) -> None:
        """Purpose: Persist per-session state and update short memory.
        Inputs/Outputs: Input is PipelineContext; mutates order_state in place.
        Side Effects / State: Updates order_state and short_memory fields.
        Dependencies: Uses update_short_memory_from_context and helper extractors.
        Failure Modes: None; pure state mutation and logging.
        If Removed: Follow-up context and contact gating will break.
        Testing Notes: Verify state updates after a turn and across sessions.
        """
        # Persist computed state for the next turn.
        state = normalize_order_state(context.order_state)
        if context.should_ask_type or (ASK_TYPE_QUESTION in (context.answer_text or "")):
            state["asked_hand_robot"] = True
        if context.asked_form:
            state["asked_contact_form"] = True
        memory_after = update_short_memory_from_context(context)
        state["short_memory"] = memory_after
        state["short_memory_ts"] = time.time()
        context.short_memory = memory_after
        if context.display_items or context.items:
            source_items = context.display_items or context.items
            state["last_intent"] = context.intent_label
            state["last_context_codes"] = [item.code for item in source_items if item.code][:4]
            state["last_group"] = context.intent_entities.get("product_group")
            constraints: Dict[str, object] = {}
            if context.intent_entities.get("amp"):
                constraints["amp"] = context.intent_entities.get("amp")
            if context.intent_entities.get("size") is not None:
                constraints["size"] = context.intent_entities.get("size")
            if context.intent_entities.get("length") is not None:
                constraints["length"] = context.intent_entities.get("length")
            system = detect_system_tag(" ".join(f"{item.name} {item.description}" for item in source_items))
            if system:
                constraints["system"] = system
            if context.intent_label in {"LIST", "LIST_REQUEST"}:
                inferred = extract_lookup_constraints(context.user_message)
                if not state.get("last_group") and inferred.get("product_group"):
                    state["last_group"] = inferred.get("product_group")
                if inferred.get("amp") and "amp" not in constraints:
                    constraints["amp"] = inferred.get("amp")
            if constraints:
                state["last_constraints"] = constraints
        context.order_state = state
        logger.info("session=%s answer=%s", context.session_id, context.answer_text)
        logger.info(
            "session=%s memory_after=%s",
            context.session_id,
            json.dumps(sanitize_short_memory_for_log(memory_after), ensure_ascii=True),
        )
        logger.debug(
            "session=%s order_state=%s",
            context.session_id,
            json.dumps(sanitize_state_for_log(context.order_state), ensure_ascii=True),
        )
        return


def is_ask_selling_scope(message: str) -> bool:
    """Purpose: Detect selling-scope questions that trigger the fixed reply route.
    Inputs/Outputs: Input is message string; output is True if scope phrases match.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and ASK_SELLING_SCOPE_PHRASES.
    Failure Modes: False negatives if phrasing deviates from phrase list.
    If Removed: Selling-scope questions fall through to generic intent detection.
    Testing Notes: Verify known "ban gi" variants return True.
    """
    # Compare normalized message with scope phrases.
    normalized = normalize_text(message)
    return any(phrase in normalized for phrase in ASK_SELLING_SCOPE_PHRASES)


def is_availability_query(message: str) -> bool:
    """Purpose: Detect availability-style queries like "co ban khong".
    Inputs/Outputs: Input is message string; output is True if phrase match found.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and static phrase list.
    Failure Modes: False negatives for uncommon phrasing.
    If Removed: Commercial guardrails may not trigger for availability questions.
    Testing Notes: Validate common availability phrases return True.
    """
    # Match normalized message against availability phrases.
    normalized = normalize_text(message)
    phrases = [
        "co ban khong",
        "ban khong",
        "con ban khong",
        "co ban ko",
        "ban ko",
        "co ban k",
        "ban k",
        "co cung cap khong",
        "co cung cap ko",
        "co cung cap k",
        "ben em co khong",
        "ben em co ko",
        "ben em co k",
        "co khong",
        "co ko",
        "co k",
    ]
    return any(phrase in normalized for phrase in phrases)


def is_single_unit_request(message: str) -> bool:
    """Purpose: Detect requests for a single unit to enforce no-retail rule.
    Inputs/Outputs: Input is message string; output is True if single-unit pattern matches.
    Side Effects / State: None.
    Dependencies: Uses SINGLE_UNIT_RE and normalize_text.
    Failure Modes: False negatives if phrasing is outside regex.
    If Removed: The "no retail 1 unit" guard will not trigger.
    Testing Notes: Validate "1 cai" and "mot cai" examples.
    """
    # Match normalized text against single-unit regex.
    normalized = normalize_text(message)
    return bool(SINGLE_UNIT_RE.search(normalized))


def is_acknowledgement(normalized: str) -> bool:
    """Purpose: Detect short acknowledgement replies (ok/duoc/roi).
    Inputs/Outputs: Input is normalized text; output is True if all tokens are ACK_WORDS.
    Side Effects / State: None.
    Dependencies: Uses ACK_WORDS set.
    Failure Modes: Returns False for longer or mixed-content messages.
    If Removed: Info-only detection will be less accurate.
    Testing Notes: Verify single-word and two-word acknowledgements.
    """
    # Treat short messages of known tokens as acknowledgements.
    words = normalized.split()
    if not words or len(words) > 3:
        return False
    return all(word in ACK_WORDS for word in words)


def is_info_only_query(message: str) -> bool:
    """Purpose: Determine if a message is informational-only (no product request).
    Inputs/Outputs: Input is raw message; output is True if info-only heuristics match.
    Side Effects / State: None.
    Dependencies: Uses normalize_text, regexes, and is_acknowledgement.
    Failure Modes: Heuristics may misclassify ambiguous messages.
    If Removed: Routing may treat info-only as product queries.
    Testing Notes: Test short origin/brand questions and acknowledgements.
    """
    # Apply info-only heuristics after normalization.
    normalized = normalize_text(message)
    if not normalized:
        return False
    if LISTING_RE.search(normalized) or PRODUCT_INFO_RE.search(normalized) or RELATED_QUERY_RE.search(normalized):
        return False
    if is_acknowledgement(normalized):
        return True
    word_count = len(normalized.split())
    if INFO_ONLY_RE.search(normalized) and word_count <= 8:
        return True
    return False


def build_info_only_response(message: str) -> str:
    """Purpose: Produce a short, compliant response for info-only questions.
    Inputs/Outputs: Input is raw message; output is a short response string.
    Side Effects / State: None.
    Dependencies: Uses normalize_text and is_acknowledgement.
    Failure Modes: None; defaults to a generic brand-origin answer.
    If Removed: Info-only queries may fall through to verbose product output.
    Testing Notes: Validate responses for origin/brand and acknowledgement cases.
    """
    # Return a concise scripted response based on recognized keywords.
    normalized = normalize_text(message)
    if "trung quoc" in normalized or "hang tq" in normalized:
        return "Dạ không ạ, bên em là Tokinarc Nhật Bản chính hãng."
    if "tokinarc" in normalized or "chinh hang" in normalized or "hang nhat" in normalized or "nhat ban" in normalized:
        return "Dạ, bên em là Tokinarc Nhật Bản chính hãng ạ."
    if "xuat xu" in normalized or "nguon goc" in normalized:
        return "Dạ, bên em là Tokinarc Nhật Bản ạ."
    if is_acknowledgement(normalized):
        return "Dạ vâng ạ."
    return "Dạ, bên em là Tokinarc Nhật Bản chính hãng ạ."


def get_selling_scope_response(state: Dict[str, object]) -> str:
    """Purpose: Rotate selling-scope templates to avoid repetitive replies.
    Inputs/Outputs: Input is order_state dict; output is selected template string.
    Side Effects / State: Updates state['selling_scope_variant'].
    Dependencies: Uses SELLING_SCOPE_TEMPLATES.
    Failure Modes: Returns empty string if templates list is empty.
    If Removed: Selling-scope replies will be static or missing.
    Testing Notes: Call twice and verify template rotation.
    """
    # Rotate the index stored in state to vary responses.
    if not SELLING_SCOPE_TEMPLATES:
        return ""
    last_index = state.get("selling_scope_variant")
    try:
        last_index = int(last_index)
    except (TypeError, ValueError):
        last_index = -1
    next_index = (last_index + 1) % len(SELLING_SCOPE_TEMPLATES)
    state["selling_scope_variant"] = next_index
    return SELLING_SCOPE_TEMPLATES[next_index]


def extract_skus(message: str) -> List[str]:
    """Purpose: Extract Tokin SKU strings from the message.
    Inputs/Outputs: Input is message string; output is a list of cleaned SKUs.
    Side Effects / State: None.
    Dependencies: Uses CODE_RE and extract_digits.
    Failure Modes: Returns empty list when no match.
    If Removed: SKU detection is weaker, affecting code lookup routing.
    Testing Notes: Validate "Tokin 002005" and "002005" cases.
    """
    # Extract code tokens and normalize to digits where possible.
    matches = CODE_RE.findall(message or "")
    skus: List[str] = []
    for match in matches:
        cleaned = extract_digits(match) or match
        if cleaned:
            skus.append(cleaned.strip())
    return skus


def extract_codes(message: str) -> Tuple[List[str], str]:
    """Purpose: Extract ordered codes (D, P, numeric) and primary code.
    Inputs/Outputs: Input is message string; output is (all_codes, primary_code).
    Side Effects / State: None.
    Dependencies: Uses D_CODE_RE, P_CODE_RE, NUM_CODE_RE.
    Failure Modes: Returns empty list and empty primary when no matches.
    If Removed: CODE_LOOKUP routing cannot identify explicit codes.
    Testing Notes: Validate U/P/number codes and order preference.
    """
    # Collect all codes in order of appearance and choose the first as primary.
    text = message or ""
    matches: List[Tuple[int, str]] = []
    for match in D_CODE_RE.finditer(text):
        matches.append((match.start(), match.group(0).strip()))
    for match in P_CODE_RE.finditer(text):
        matches.append((match.start(), match.group(0).strip()))
    for match in NUM_CODE_RE.finditer(text):
        matches.append((match.start(), match.group(0).strip()))
    if not matches:
        return [], ""
    matches.sort(key=lambda item: item[0])
    ordered = [code for _, code in matches]
    primary = ordered[0]
    return ordered, primary


def detect_code_type(message: str, primary_code: str) -> str:
    """Purpose: Classify codes as TOKIN or EXTERNAL for wording rules.
    Inputs/Outputs: Input is message and primary_code; output is "TOKIN" or "EXTERNAL".
    Side Effects / State: None.
    Dependencies: Uses normalize_text and regex checks.
    Failure Modes: Ambiguous codes default to EXTERNAL.
    If Removed: Code wording rules (equivalent vs direct) cannot be applied.
    Testing Notes: Validate U-codes and numeric Tokin codes.
    """
    # Infer code type from prefixes and message context.
    normalized = normalize_text(message)
    code = (primary_code or "").strip()
    if code.upper().startswith(("U", "P")):
        return "EXTERNAL"
    if re.search(r"\b(tokin|tokinarc)\b", normalized):
        return "TOKIN"
    if code and re.fullmatch(r"\d{5,6}", code):
        return "TOKIN"
    if re.search(r"\b\d{5,6}\b", normalized):
        return "TOKIN"
    return "EXTERNAL"


def extract_quantity(normalized: str) -> Optional[int]:
    """Purpose: Extract a numeric quantity from normalized text.
    Inputs/Outputs: Input is normalized string; output is int quantity or None.
    Side Effects / State: None.
    Dependencies: Uses SO_LUONG_RE and QUANTITY_RE.
    Failure Modes: Non-numeric quantities return None.
    If Removed: Quantity follow-up detection and lead capture logic degrade.
    Testing Notes: Validate "so luong 100" and "100 cai" patterns.
    """
    # Parse quantity from common patterns.
    match = SO_LUONG_RE.search(normalized)
    if match:
        try:
            return int(match.group(2))
        except ValueError:
            return None
    match = QUANTITY_RE.search(normalized)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def detect_product_group(normalized: str) -> Optional[str]:
    """Purpose: Map normalized text to a product group label.
    Inputs/Outputs: Input is normalized text; output is group code or None.
    Side Effects / State: None.
    Dependencies: Uses keyword checks for group labels.
    Failure Modes: Returns None if no keywords match.
    If Removed: Group-based retrieval and routing will be unreliable.
    Testing Notes: Validate each group keyword maps correctly.
    """
    # Prefer specific keywords before generic ones.
    if "than giu" in normalized or "tip body" in normalized:
        return "TIP_BODY"
    if "cach dien" in normalized or "insulator" in normalized:
        return "INSULATOR"
    if "su" in normalized or "orifice" in normalized or "diffuser" in normalized:
        return "ORIFICE"
    if "chup" in normalized or "nozzle" in normalized:
        return "NOZZLE"
    if "bec" in normalized or "contact tip" in normalized or "tip" in normalized:
        return "TIP"
    return None


def detect_bundle_required_categories(message: str) -> List[str]:
    """Purpose: Extract explicit accessory categories requested in the message.
    Inputs/Outputs: Inputs: message (str). Outputs: list[str] required categories.
    Side Effects / State: None; pure helper.
    Dependencies: extract_requested_parts; used in bundle intent resolution.
    Failure Modes: Returns empty list when no keywords; callers apply defaults.
    If Removed: Bundle routing loses explicit parts, leading to wrong retrieval sets.
    Testing Notes: "than giu bec va cach dien" should yield TIP_BODY + INSULATOR.
    """
    # Pull explicit parts from the message without expanding bundle.
    requested, _expand = extract_requested_parts(message)
    return requested


def is_accessory_bundle_query(message: str) -> bool:
    """Purpose:
    Decide whether the message is asking for bundled accessories vs a simple list.

    Inputs/Outputs:
    - Inputs: message (str).
    - Outputs: bool bundle-intent flag.

    Side Effects / State:
    - None.

    Dependencies:
    - extract_requested_parts, LISTING_RE, BUNDLE_QUERY_RE, normalize_text.

    Failure Modes:
    - Over-detection can route list requests into bundle flow.

    If Removed:
    - Bundle requests would be missed and routed to generic lookup.

    Testing Notes:
    - "liet ke chup khi 350" -> False.
    - "than giu bec va cach dien" -> True.
    """
    # Bundle requires explicit bundle hints or multiple requested parts.
    normalized = normalize_text(message)
    requested, expand = extract_requested_parts(message)
    is_list = bool(LISTING_RE.search(normalized))
    bundle_hint = bool(BUNDLE_QUERY_RE.search(normalized))
    if expand or bundle_hint:
        return True
    if len(requested) >= 2:
        return True
    if is_list and len(requested) == 1:
        return False
    return False


def extract_lookup_constraints(message: str) -> Dict[str, object]:
    """Purpose: Parse technical constraints (group, amp, size, length) from text.
    Inputs/Outputs: Inputs: message (str). Outputs: dict of parsed constraints.
    Side Effects / State: None.
    Dependencies: detect_product_group, AMP_ANY_RE, TIP_SIZE_LEN_RE, SIZE_RE.
    Failure Modes: Numeric casts may fail; returns None values instead of raising.
    If Removed: Technical lookup loses constraints and retrieval becomes noisy.
    Testing Notes: "0.8 x 45L" should yield size=0.8 and length=45.
    """
    # Extract size/length/amp hints for constrained retrieval.
    norm = normalize_text(message)
    group = detect_product_group(norm)

    amp = ""
    match = AMP_ANY_RE.search(norm)
    if match:
        amp = f"{match.group(1)}A"

    size: Optional[float] = None
    length: Optional[int] = None
    match = TIP_SIZE_LEN_RE.search(norm)
    if match:
        try:
            size = float(match.group(1))
        except ValueError:
            size = None
        try:
            length = int(match.group(2))
        except ValueError:
            length = None
    else:
        match = SIZE_RE.search(norm)
        if match:
            try:
                size = float(match.group(1))
            except ValueError:
                size = None

    return {"product_group": group, "amp": amp, "size": size, "length": length}


def extract_requested_parts(text: str) -> Tuple[List[str], bool]:
    """Purpose: Extract requested part roles and bundle expansion hint from text.
    Inputs/Outputs: Inputs: text (str). Outputs: (list[str], bool).
    Side Effects / State: None.
    Dependencies: PART_SYNONYMS, BUNDLE_HINT_WORDS, normalize_text.
    Failure Modes: Returns empty list/False when no keyword matches.
    If Removed: Bundle resolution will not know which parts to fetch explicitly.
    Testing Notes: "tip body va cach dien" returns (['TIP_BODY','INSULATOR'], False).
    """
    # Match part synonyms and bundle hint words in normalized text.
    normalized = normalize_text(text)
    requested: List[str] = []
    for role, words in PART_SYNONYMS.items():
        for word in words:
            if normalize_text(word) in normalized:
                requested.append(role)
                break
    expand_bundle = any(normalize_text(word) in normalized for word in BUNDLE_HINT_WORDS)
    return requested, expand_bundle


def parse_commercial_action(data: Dict[str, object]) -> Dict[str, object]:
    """Purpose: Normalize commercial_action payload from model output.
    Inputs/Outputs: Inputs: data (dict). Outputs: dict with collect_contact/fields/reason.
    Side Effects / State: None.
    Dependencies: None; defensive type checks only.
    Failure Modes: Non-dict structures return empty defaults without raising.
    If Removed: Lead-collection decisions from LLM cannot be consumed safely.
    Testing Notes: Missing commercial_action should return collect_contact False.
    """
    # Coerce optional LLM decision fields into a stable schema.
    action = data.get("commercial_action")
    if not isinstance(action, dict):
        action = {}
    collect_contact = bool(action.get("collect_contact")) if "collect_contact" in action else False
    fields = action.get("fields") if isinstance(action.get("fields"), list) else []
    reason = action.get("reason") if isinstance(action.get("reason"), str) else ""
    return {"collect_contact": collect_contact, "fields": fields, "reason": reason}


def get_bulk_qty_threshold(items: List[ResourceItem]) -> Optional[int]:
    """Purpose: Determine bulk-quantity threshold from env or catalog metadata.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: min threshold or None.
    Side Effects / State: None; reads environment variables.
    Dependencies: os.getenv, BULK_QTY_KEYS, normalize_text.
    Failure Modes: Non-numeric values are skipped; returns None if not found.
    If Removed: Quantity-based lead guardrails cannot infer thresholds.
    Testing Notes: Set BULK_QTY_THRESHOLD env and verify it overrides metadata.
    """
    # Prefer explicit env override before scanning item metadata.
    env_value = os.getenv("BULK_QTY_THRESHOLD")
    if env_value and str(env_value).isdigit():
        return int(env_value)
    values: List[int] = []
    for item in items:
        raw = item.raw or {}
        for key, value in raw.items():
            key_norm = normalize_text(str(key))
            if key_norm in (normalize_text(k) for k in BULK_QTY_KEYS) or (
                "min" in key_norm and ("qty" in key_norm or "so luong" in key_norm)
            ):
                try:
                    num = int(str(value).strip())
                except (TypeError, ValueError):
                    continue
                if num > 0:
                    values.append(num)
    return min(values) if values else None


def is_technical_lookup(message: str) -> bool:
    """Purpose: Heuristic gate for technical lookup (non-commercial) queries.
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: is_quantity_followup_message, regex flags, detect_product_group.
    Failure Modes: Returns False for mixed commercial signals; caller falls back to LLM.
    If Removed: Product lookup may be routed to commercial handling too often.
    Testing Notes: "bec 0.8x45l" should be True; "bao gia 0.8x45l" False.
    """
    # Block lookups for follow-ups or commercial intents.
    norm = normalize_text(message)
    if is_quantity_followup_message(message):
        return False
    if RELATED_QUERY_RE.search(norm) or is_accessory_bundle_query(message):
        return False
    group = detect_product_group(norm)
    if not group:
        return False
    if extract_codes(message)[0]:
        return False

    if PRICE_RE.search(norm) or is_availability_query(message) or QUANTITY_RE.search(norm) or BUY_INTENT_RE.search(norm):
        return False

    if LOOKUP_HINT_RE.search(norm):
        return True
    if len(norm.split()) <= 6:
        return True
    return False


def is_type_only_message(message: str) -> bool:
    """Purpose: Detect short messages that only specify hand/robot usage.
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: TYPE_ANSWER_RE, CODE_RE, LISTING_RE, PRICE_RE.
    Failure Modes: Returns False for long or mixed-content messages.
    If Removed: Type-switch follow-ups would not reuse cached context.
    Testing Notes: "robot" -> True; "robot 350A" -> False.
    """
    # Guard against false positives with short-message checks.
    n = normalize_text(message)
    if len(n.split()) > 6:
        return False
    return (
        bool(TYPE_ANSWER_RE.search(n))
        and not CODE_RE.search(n)
        and not LISTING_RE.search(n)
        and not PRICE_RE.search(n)
    )


def is_amp_only_message(message: str) -> bool:
    """Purpose: Detect amp-only follow-up messages like "350A" or "500A".
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: AMP_ANY_RE, extract_quantity, detect_product_group, CODE_RE.
    Failure Modes: Returns False if extra info is present.
    If Removed: Slot-fill for amp will be misrouted as new requests.
    Testing Notes: "350A" -> True; "350A cach dien" -> False.
    """
    # Treat short amp-only content as a slot-fill response.
    normalized = normalize_text(message)
    if not AMP_ANY_RE.search(normalized):
        return False
    if extract_quantity(normalized) is not None:
        return False
    if detect_product_group(normalized):
        return False
    if CODE_RE.search(message) or D_CODE_RE.search(message) or P_CODE_RE.search(message) or NUM_CODE_RE.search(message):
        return False
    if PRICE_RE.search(normalized) or LISTING_RE.search(normalized) or RELATED_QUERY_RE.search(normalized):
        return False
    if len(normalized.split()) <= 4:
        return True
    return bool(FOLLOWUP_CUE_RE.search(normalized) and len(normalized.split()) <= 6)


def is_quantity_followup_message(message: str) -> bool:
    """Purpose: Detect quantity-only follow-up messages (no new SKU/group).
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: extract_quantity, CODE_RE, LISTING_RE, PRICE_RE, RELATED_QUERY_RE.
    Failure Modes: Returns False if message introduces a new product group or code.
    If Removed: Quantity follow-ups will trigger fresh retrieval and drift.
    Testing Notes: "so luong 100 cai" -> True; "100 cai tok in 002005" -> False.
    """
    # Require quantity plus short, non-product-changing text.
    n = normalize_text(message)
    qty = extract_quantity(n)
    if qty is None:
        return False
    if CODE_RE.search(message) or LISTING_RE.search(n) or PRICE_RE.search(n) or RELATED_QUERY_RE.search(n):
        return False
    if detect_product_group(n):
        return False
    return len(n.split()) <= 8


def message_has_any_term(normalized: str, terms: Iterable[str]) -> bool:
    """Purpose: Check normalized text for any whole-word term in a term list.
    Inputs/Outputs: Inputs: normalized (str), terms (iterable[str]). Outputs: bool.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Returns False for empty inputs or empty term lists.
    If Removed: Affirmation/negation detection becomes brittle and noisy.
    Testing Notes: "muon" should match AFFIRM_TERMS; "khong muon" should match NEGATE_TERMS.
    """
    # Match full terms against a padded normalized string to avoid substrings.
    if not normalized or not terms:
        return False
    padded = f" {normalized} "
    for term in terms:
        if not term:
            continue
        if f" {term} " in padded:
            return True
    return False


def is_affirmation_message(message: str) -> bool:
    """Purpose: Detect short affirmation replies such as "muon" or "ok".
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text, AFFIRM_TERMS, and intent guard regexes.
    Failure Modes: Returns False for mixed-content or long messages.
    If Removed: Pending action follow-ups will be treated as new intents.
    Testing Notes: "muon" -> True; "co ban khong" -> False.
    """
    # Accept only very short, non-technical replies.
    normalized = normalize_text(message)
    if not normalized:
        return False
    if is_availability_query(message):
        return False
    if CODE_RE.search(message) or LISTING_RE.search(normalized) or PRICE_RE.search(normalized):
        return False
    if extract_quantity(normalized) is not None or AMP_ANY_RE.search(normalized):
        return False
    if detect_product_group(normalized) or RELATED_QUERY_RE.search(normalized):
        return False
    if len(normalized.split()) > 4:
        return False
    return message_has_any_term(normalized, AFFIRM_TERMS)


def is_negative_message(message: str) -> bool:
    """Purpose: Detect short negative replies such as "khong" or "thoi".
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text, NEGATE_TERMS, and intent guard regexes.
    Failure Modes: Returns False for long or mixed messages.
    If Removed: Pending actions may linger and misroute future messages.
    Testing Notes: "khong" -> True; "khong biet ma" -> False.
    """
    # Accept only very short, non-technical rejections.
    normalized = normalize_text(message)
    if not normalized:
        return False
    if is_availability_query(message):
        return False
    if CODE_RE.search(message) or LISTING_RE.search(normalized) or PRICE_RE.search(normalized):
        return False
    if extract_quantity(normalized) is not None or AMP_ANY_RE.search(normalized):
        return False
    if detect_product_group(normalized) or RELATED_QUERY_RE.search(normalized):
        return False
    if len(normalized.split()) > 4:
        return False
    return message_has_any_term(normalized, NEGATE_TERMS)


def detect_dialogue_act(message: str) -> str:
    """Purpose: Classify short replies into dialogue acts for follow-up handling.
    Inputs/Outputs: Inputs: message (str). Outputs: str label (AFFIRM/NEGATE/etc.).
    Side Effects / State: None.
    Dependencies: is_affirmation_message, is_negative_message, slot-fill helpers.
    Failure Modes: Returns "NEW_INTENT" when no act is detected.
    If Removed: Pending actions cannot be matched to terse follow-ups.
    Testing Notes: "muon" -> AFFIRM; "350A" -> SLOT_FILL_AMP; "ok" -> AFFIRM.
    """
    # Evaluate slot-fill acts before affirmation/negation.
    if is_amp_only_message(message):
        return "SLOT_FILL_AMP"
    if is_type_only_message(message):
        return "SLOT_FILL_TYPE"
    if is_pure_quantity_message(message) or is_quantity_followup_message(message):
        return "SLOT_FILL_QUANTITY"
    if is_affirmation_message(message):
        return "AFFIRM"
    if is_negative_message(message):
        return "NEGATE"
    return "NEW_INTENT"


def has_accessory_invite(answer: str) -> bool:
    """Purpose: Detect if a response invited the user to list accessories.
    Inputs/Outputs: Inputs: answer (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text and ACCESSORY_INVITE_TERMS.
    Failure Modes: Returns False for empty answers.
    If Removed: Pending action will not be created for "muon" follow-ups.
    Testing Notes: The default closing line should return True.
    """
    # Match accessory invitation keywords in the assistant answer.
    normalized = normalize_text(answer)
    return message_has_any_term(normalized, ACCESSORY_INVITE_TERMS)


def build_pending_action_from_context(context: PipelineContext) -> Dict[str, object]:
    """Purpose: Build a pending action payload from the current context.
    Inputs/Outputs: Inputs: context (PipelineContext). Outputs: dict for pending_action.
    Side Effects / State: None; caller persists into short memory.
    Dependencies: detect_item_group, find_item_by_selected_sku, DEFAULT_BUNDLE_CATEGORIES.
    Failure Modes: Returns empty dict when no anchor is available.
    If Removed: Short confirmations like "muon" will not map to bundle lookup.
    Testing Notes: After LIST, pending_action should include ACCESSORY_BUNDLE_LOOKUP.
    """
    # Use the best available anchor to drive a follow-up bundle lookup.
    anchor = None
    if context.display_items:
        anchor = context.display_items[0]
    elif context.items:
        anchor = context.items[0]
    else:
        selected = context.order_state.get("selected_sku")
        if selected:
            anchor = find_item_by_selected_sku(context.catalog_items, str(selected))
    if not anchor:
        return {}

    anchor_group = detect_item_group(anchor)
    required_parts = [part for part in DEFAULT_BUNDLE_CATEGORIES if part != anchor_group]
    constraints: Dict[str, object] = {}
    amp = context.intent_entities.get("amp") or (context.order_state.get("last_constraints") or {}).get("amp")
    system = context.intent_entities.get("system") or (context.order_state.get("last_constraints") or {}).get("system")
    if amp:
        constraints["amp"] = amp
    if system:
        constraints["system"] = system
    return {
        "action": "ACCESSORY_BUNDLE_LOOKUP",
        "required_parts": required_parts,
        "anchor_sku": anchor.code or "",
        "product_group": anchor_group,
        "constraints": constraints,
    }


def parse_user_input(message: str) -> Dict[str, object]:
    """Purpose: Parse raw user text into structured slots and constraints.
    Inputs/Outputs: Inputs: message (str). Outputs: dict of parsed slots.
    Side Effects / State: None; pure parser.
    Dependencies: extract_* helpers, detect_product_group, regex constants.
    Failure Modes: Missing patterns yield empty/None fields; callers apply fallbacks.
    If Removed: Memory resolution and intent forcing lose structured signal.
    Testing Notes: Include SKU + size + amp in message and verify slots populate.
    """
    # Centralized slot extraction for memory resolution.
    normalized = normalize_text(message)
    skus = extract_skus(message)
    codes, primary_code = extract_codes(message)
    if not skus and codes:
        digit_codes = [code for code in codes if code.isdigit()]
        if digit_codes:
            skus = digit_codes
    quantity = extract_quantity(normalized)
    amp = ""
    match = AMP_ANY_RE.search(normalized)
    if match:
        amp = f"{match.group(1)}A"
    is_robot = None
    if "robot" in normalized or "robotic" in normalized:
        is_robot = True
    if "tay" in normalized or "hand" in normalized:
        is_robot = False
    product_group = detect_product_group(normalized)
    required_parts, expand_bundle = extract_requested_parts(message)
    bundle_hint = bool(BUNDLE_QUERY_RE.search(normalized))
    constraints = extract_lookup_constraints(message)
    thread = None
    thread_match = THREAD_RE.search(message)
    if thread_match:
        thread = thread_match.group(0).upper()
    material = None
    if MATERIAL_RE.search(normalized):
        material = "ALUMINUM"
    system = detect_system_tag(normalized)
    if thread:
        constraints["thread"] = thread
    if material:
        constraints["material"] = material
    if system:
        constraints["system"] = system
    return {
        "normalized": normalized,
        "skus": skus,
        "codes": codes,
        "primary_code": primary_code,
        "quantity": quantity,
        "amp": amp,
        "is_robot": is_robot,
        "product_group": product_group,
        "required_parts": required_parts,
        "bundle_hint": bundle_hint,
        "expand_bundle": expand_bundle,
        "constraints": constraints,
    }


def resolve_request_with_memory(
    message: str,
    parsed: Dict[str, object],
    memory: Dict[str, object],
) -> Dict[str, object]:
    """Purpose: Merge parsed input with short memory to resolve anchor and slots.
    Inputs/Outputs: Inputs: message (str), parsed (dict), memory (dict). Outputs: dict.
    Side Effects / State: None; caller applies returned resolution to state.
    Dependencies: normalize_text, FOLLOWUP_CUE_RE, TECHNICAL_INTENTS, memory keys.
    Failure Modes: If memory missing, anchor fields remain empty; caller may ask for SKU.
    If Removed: Follow-up handling degrades and anchor-based retrieval drifts.
    Testing Notes: After a lookup, "500A thi sao" should keep the last anchor.
    """
    # Blend current message signals with recent anchor context.
    normalized = parsed.get("normalized") or normalize_text(message)
    words = normalized.split()
    last_anchor = memory.get("last_anchor") or {}
    last_results = memory.get("last_results") or []
    pending_action = memory.get("pending_action") or {}
    dialogue_act = detect_dialogue_act(message)
    clear_pending_action = False
    is_list = bool(LISTING_RE.search(normalized))
    followup = bool(FOLLOWUP_CUE_RE.search(normalized))
    followup = followup or bool(parsed.get("quantity")) or bool(parsed.get("amp"))
    followup = followup or parsed.get("is_robot") is not None
    followup = followup or bool(parsed.get("bundle_hint"))
    if len(words) <= 4 and last_anchor.get("sku"):
        followup = True

    anchor_sku = ""
    anchor_cat = ""
    anchor_name = ""
    anchor_used = False

    skus = parsed.get("skus") or []
    if skus:
        anchor_sku = skus[0]
        anchor_used = False
    elif followup and last_anchor.get("sku"):
        anchor_sku = str(last_anchor.get("sku"))
        anchor_cat = str(last_anchor.get("cat") or "")
        anchor_name = str(last_anchor.get("name") or "")
        anchor_used = True
    elif followup and len(last_results) == 1:
        anchor_sku = str(last_results[0])
        anchor_used = True

    if not anchor_cat and last_anchor.get("cat"):
        anchor_cat = str(last_anchor.get("cat") or "")
    if not anchor_name and last_anchor.get("name"):
        anchor_name = str(last_anchor.get("name") or "")

    product_group = parsed.get("product_group") or anchor_cat
    line_amp = parsed.get("amp") or (memory.get("last_user_constraints") or {}).get("amp") or ""
    is_robot = parsed.get("is_robot")
    if is_robot is None and last_anchor.get("is_robot") is not None:
        is_robot = last_anchor.get("is_robot")

    requested_parts = parsed.get("required_parts") or []
    required_parts = list(requested_parts)
    expand_bundle = bool(parsed.get("expand_bundle"))
    if not required_parts and parsed.get("bundle_hint"):
        pending = (memory.get("pending_request") or {}).get("required_parts") or []
        required_parts = pending
    if re.match(r"^con\b", normalized) and requested_parts:
        required_parts = list(requested_parts)
        expand_bundle = False
        if len(requested_parts) == 1:
            product_group = requested_parts[0]
    if "cach dien" in normalized or "insulator" in normalized:
        product_group = "INSULATOR"
        if "INSULATOR" not in required_parts:
            required_parts = merge_unique(required_parts, ["INSULATOR"])

    constraints = dict(memory.get("last_user_constraints") or {})
    for key, value in (parsed.get("constraints") or {}).items():
        if value is None or value == "":
            continue
        constraints[key] = value

    force_intent = ""
    last_intent = memory.get("last_intent") or ""
    if not is_list and (required_parts or expand_bundle or parsed.get("bundle_hint")) and anchor_sku:
        force_intent = "ACCESSORY_BUNDLE_LOOKUP"
    elif not is_list and line_amp and anchor_sku and last_intent in TECHNICAL_INTENTS:
        force_intent = last_intent
    elif not is_list and is_robot is not None and anchor_sku and last_intent in TECHNICAL_INTENTS:
        force_intent = last_intent
    if is_pure_quantity_message(message) or is_quantity_followup_message(message):
        force_intent = ""

    if pending_action and dialogue_act == "AFFIRM":
        pending_intent = pending_action.get("action") or ""
        if pending_intent:
            force_intent = pending_intent
            clear_pending_action = True
            if pending_action.get("anchor_sku"):
                anchor_sku = str(pending_action.get("anchor_sku") or "")
                anchor_used = True
            if pending_action.get("product_group") and not parsed.get("product_group"):
                product_group = str(pending_action.get("product_group") or "")
            pending_parts = pending_action.get("required_parts") or []
            if pending_parts:
                required_parts = list(pending_parts)
            for key, value in (pending_action.get("constraints") or {}).items():
                if value and key not in constraints:
                    constraints[key] = value
    elif pending_action and dialogue_act in {"SLOT_FILL_AMP", "SLOT_FILL_TYPE", "SLOT_FILL_QUANTITY"}:
        clear_pending_action = True
    elif pending_action and dialogue_act == "NEGATE":
        clear_pending_action = True
    elif pending_action and dialogue_act == "NEW_INTENT":
        new_intent_signal = bool(
            CODE_RE.search(message)
            or LISTING_RE.search(normalized)
            or PRICE_RE.search(normalized)
            or RELATED_QUERY_RE.search(normalized)
            or detect_product_group(normalized)
            or extract_quantity(normalized) is not None
            or AMP_ANY_RE.search(normalized)
            or is_availability_query(message)
        )
        if new_intent_signal:
            clear_pending_action = True

    return {
        "anchor_sku": anchor_sku,
        "anchor_cat": anchor_cat,
        "anchor_name": anchor_name,
        "anchor_used": anchor_used,
        "product_group": product_group,
        "line_amp": line_amp,
        "is_robot": is_robot,
        "required_parts": required_parts,
        "bundle_hint": bool(parsed.get("bundle_hint")),
        "expand_bundle": expand_bundle,
        "constraints": constraints,
        "force_intent": force_intent,
        "clear_pending_action": clear_pending_action,
    }


def apply_resolved_to_order_state(order_state: Dict[str, object], resolved: Dict[str, object]) -> None:
    """Purpose: Copy resolved anchor/group/type into order_state.
    Inputs/Outputs: Inputs: order_state (dict), resolved (dict). Outputs: None.
    Side Effects / State: Mutates order_state fields used across pipeline steps.
    Dependencies: None; simple assignment helper in intent detection.
    Failure Modes: No-op if resolved has no anchor fields.
    If Removed: Session state will not reflect resolved anchors, causing follow-up loss.
    Testing Notes: After resolve, order_state.selected_sku should match anchor_sku.
    """
    # Apply resolved anchor and type preference to session state.
    if resolved.get("anchor_sku"):
        order_state["selected_sku"] = resolved.get("anchor_sku")
    if resolved.get("product_group"):
        order_state["selected_group"] = resolved.get("product_group")
    if resolved.get("is_robot") is True:
        order_state["hand_or_robot"] = "ROBOT"
    elif resolved.get("is_robot") is False:
        order_state["hand_or_robot"] = "HAND"


def build_forced_decision(
    context: PipelineContext,
    parsed: Dict[str, object],
    resolved: Dict[str, object],
) -> Optional[IntentDecision]:
    """Purpose: Build an IntentDecision when resolve step forces an intent.
    Inputs/Outputs: Inputs: context, parsed (dict), resolved (dict). Outputs: IntentDecision or None.
    Side Effects / State: None; returns a new decision object.
    Dependencies: detect_buy_intent, detect_topic, resolved slots.
    Failure Modes: Returns None when no force_intent is present.
    If Removed: Forced bundle/slot-fill paths will be skipped, causing wrong routing.
    Testing Notes: For bundle follow-up, force_intent should produce ACCESSORY_BUNDLE_LOOKUP.
    """
    # Materialize a decision from resolved intent override.
    intent = resolved.get("force_intent")
    if not intent:
        return None
    normalized = parsed.get("normalized") or normalize_text(context.user_message)
    entities: Dict[str, object] = {
        "skus": [resolved.get("anchor_sku")] if resolved.get("anchor_sku") else [],
        "quantity": parsed.get("quantity"),
        "is_robot": resolved.get("is_robot"),
        "product_group": resolved.get("product_group"),
        "primary_code": parsed.get("primary_code"),
        "codes": parsed.get("codes") or [],
        "required_categories": resolved.get("required_parts") or [],
        "bundle_hint": bool(parsed.get("bundle_hint")),
        "expand_bundle": bool(resolved.get("expand_bundle")),
    }
    if resolved.get("is_robot") is True:
        entities["is_hand"] = False
    elif resolved.get("is_robot") is False:
        entities["is_hand"] = True
    constraints = resolved.get("constraints") or {}
    for key in ("amp", "size", "length", "thread", "material", "system"):
        if constraints.get(key):
            entities[key] = constraints.get(key)
    if resolved.get("line_amp") and not entities.get("amp"):
        entities["amp"] = resolved.get("line_amp")

    decision = IntentDecision(
        intent=intent,
        buy_intent=detect_buy_intent(normalized, parsed.get("quantity")),
        info_only=False,
        topic=detect_topic(normalized),
        entities=entities,
        missing=[],
        next_action="ANSWER_ONLY",
    )
    return decision


def merge_decision_with_resolved(decision: IntentDecision, resolved: Dict[str, object]) -> IntentDecision:
    """Purpose: Merge resolved slots into LLM intent decision.
    Inputs/Outputs: Inputs: decision (IntentDecision), resolved (dict). Outputs: IntentDecision.
    Side Effects / State: Mutates decision.entities and missing fields.
    Dependencies: resolved keys, detect_product_group slots, bundle flags.
    Failure Modes: If resolved empty, returns original decision unchanged.
    If Removed: LLM decisions will ignore memory anchors and slot hints.
    Testing Notes: With anchor_sku in resolved, decision.entities should include skus.
    """
    # Inject resolved anchor and constraints into decision entities.
    if not resolved:
        return decision
    entities = dict(decision.entities or {})
    if resolved.get("anchor_sku") and not entities.get("skus"):
        entities["skus"] = [resolved.get("anchor_sku")]
    if resolved.get("product_group") and not entities.get("product_group"):
        entities["product_group"] = resolved.get("product_group")
    if resolved.get("line_amp") and not entities.get("amp"):
        entities["amp"] = resolved.get("line_amp")
    if resolved.get("is_robot") is True:
        entities["is_robot"] = True
        entities["is_hand"] = False
    elif resolved.get("is_robot") is False:
        entities["is_robot"] = False
        entities["is_hand"] = True
    if resolved.get("required_parts"):
        entities["required_categories"] = resolved.get("required_parts")
    if resolved.get("expand_bundle") is True:
        entities["expand_bundle"] = True
    if "bundle_hint" not in entities and (resolved.get("required_parts") or resolved.get("bundle_hint")):
        entities["bundle_hint"] = True
    constraints = resolved.get("constraints") or {}
    for key in ("size", "length", "thread", "material", "system"):
        if constraints.get(key) and not entities.get(key):
            entities[key] = constraints.get(key)
    decision.entities = entities
    if resolved.get("anchor_sku") and "sku" in decision.missing:
        decision.missing = [item for item in decision.missing if item != "sku"]
    return decision


def has_technical_constraints(message: str) -> bool:
    """Purpose: Check if message contains explicit technical constraints.
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: TIP_SIZE_LEN_RE, AMP_ANY_RE, CODE_RE/D_CODE_RE/P_CODE_RE/NUM_CODE_RE.
    Failure Modes: Returns False when patterns are missing; caller uses other routing.
    If Removed: Technical inquiry detection loses key signals and may misroute.
    Testing Notes: "0.8 x 45L" and "Tokin 002005" should return True.
    """
    # Scan for size/amp/code signals that indicate a technical lookup.
    norm = normalize_text(message)
    if TIP_SIZE_LEN_RE.search(norm):
        return True
    if AMP_ANY_RE.search(norm):
        return True
    if CODE_RE.search(message) or D_CODE_RE.search(message) or P_CODE_RE.search(message) or NUM_CODE_RE.search(message):
        return True
    return False


def is_tech_product_inquiry(message: str) -> bool:
    """Purpose: Detect selling-verb questions that include technical constraints.
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: is_quantity_followup_message, detect_product_group, extract_codes.
    Failure Modes: Returns False for code-bearing or quantity-only messages.
    If Removed: "co ban ..." with specs may fall into commercial route.
    Testing Notes: "co ban bec 0.8x45l" should be True.
    """
    # Gate by selling verb plus explicit constraints.
    norm = normalize_text(message)
    if is_quantity_followup_message(message):
        return False
    group = detect_product_group(norm)
    if not group:
        return False
    if extract_codes(message)[0]:
        return False
    return bool(SELLING_VERB_RE.search(norm)) and has_technical_constraints(message)


def item_matches_group(item: ResourceItem, group: str) -> bool:
    """Purpose: Determine whether an item belongs to a target product group.
    Inputs/Outputs: Inputs: item (ResourceItem), group (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: _normalize_category, GROUP_SYNONYMS, normalize_text.
    Failure Modes: Returns False if category/name is missing or unmatched.
    If Removed: Group filtering in retrieval will drift across categories.
    Testing Notes: TIP_BODY group should match items with category "TIP BODY".
    """
    # Prefer normalized category match before synonym fallback.
    cat = _normalize_category(item.category or "")
    group_norm = group.replace("_", "").upper()
    if cat == group_norm:
        return True
    hay = normalize_text(f"{item.category} {item.name} {item.description}")
    return any(normalize_text(term) in hay for term in GROUP_SYNONYMS.get(group, []))


def item_amp(item: ResourceItem) -> str:
    """Purpose: Extract amp label from item text (e.g., 350A/500A).
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: amp string or "".
    Side Effects / State: None.
    Dependencies: AMP_ANY_RE, normalize_text.
    Failure Modes: Returns empty string if not found.
    If Removed: Amp-based filtering and scoring will be less accurate.
    Testing Notes: Item name containing "350A" should yield "350A".
    """
    # Parse amp value from item name/description.
    normalized = normalize_text(f"{item.name} {item.description}")
    match = AMP_ANY_RE.search(normalized)
    return f"{match.group(1)}A" if match else ""


def has_ambiguous_amp_by_sku(items: List[ResourceItem]) -> bool:
    """Purpose: Detect whether a SKU maps to multiple amp values.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: bool.
    Side Effects / State: None.
    Dependencies: sku_key_for_group, detect_amp_line.
    Failure Modes: Returns False when no amps are detected.
    If Removed: Bundle logic may incorrectly assume a single amp variant.
    Testing Notes: Same SKU with 350A and 500A should return True.
    """
    # Track amp variants per SKU key.
    sku_amps: Dict[str, set] = {}
    for item in items:
        key = sku_key_for_group(item)
        if not key:
            continue
        amp = detect_amp_line(f"{item.name} {item.description}")
        if not amp:
            continue
        sku_amps.setdefault(key, set()).add(amp)
    return any(len(amps) > 1 for amps in sku_amps.values())


def item_size(item: ResourceItem) -> Optional[float]:
    """Purpose: Extract numeric size from item raw data.
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: float size or None.
    Side Effects / State: None.
    Dependencies: get_raw_value.
    Failure Modes: Returns None on parse errors or missing fields.
    If Removed: Size-based lookup will not filter by wire size.
    Testing Notes: Size field "0.8" should return 0.8.
    """
    # Parse size from item raw fields defensively.
    value = get_raw_value(item.raw, ["Kích thước dây (Size mm)"])
    try:
        return float(value) if value not in (None, "") else None
    except Exception:
        return None


def item_length(item: ResourceItem) -> Optional[int]:
    """Purpose: Extract numeric length from item raw data.
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: int length or None.
    Side Effects / State: None.
    Dependencies: get_raw_value.
    Failure Modes: Returns None on parse errors or missing fields.
    If Removed: Length-based lookup will not filter by overall length.
    Testing Notes: Length field "45" should return 45.
    """
    # Parse length from item raw fields defensively.
    value = get_raw_value(item.raw, ["Tổng chiều dài (mm)"])
    try:
        return int(float(value)) if value not in (None, "") else None
    except Exception:
        return None


def normalize_code_value(value: Optional[str]) -> str:
    """Purpose: Normalize a code by removing whitespace and uppercasing.
    Inputs/Outputs: Inputs: value (str|None). Outputs: normalized code string.
    Side Effects / State: None.
    Dependencies: re.sub.
    Failure Modes: Returns empty string for falsy input.
    If Removed: Code matching across formats will be inconsistent.
    Testing Notes: "  U4167L00 " -> "U4167L00".
    """
    # Strip whitespace and enforce uppercase for stable comparisons.
    if not value:
        return ""
    return re.sub(r"\s+", "", str(value)).upper()


def extract_digits(text: str) -> str:
    """Purpose: Extract only digit characters from a string.
    Inputs/Outputs: Inputs: text (str). Outputs: digits-only string.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Returns empty string if no digits are present.
    If Removed: Code normalization and matching will be less robust.
    Testing Notes: "Tokin 002005" -> "002005".
    """
    # Keep digits only to compare numeric codes consistently.
    return "".join(ch for ch in text if ch.isdigit())


def exact_lookup_by_code(items: List[ResourceItem], code: str) -> List[ResourceItem]:
    """Purpose: Exact-match lookup for Tokin/P/D codes across catalog rows.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), code (str). Outputs: list matches.
    Side Effects / State: None.
    Dependencies: get_raw_value, normalize_code_value, extract_digits.
    Failure Modes: Returns empty list if no exact match found.
    If Removed: CODE_LOOKUP route will fall back to semantic retrieval, violating rules.
    Testing Notes: D-code "U4167L00" should match items by D Part No.
    """
    # Strict lookup path that bypasses semantic retrieval.
    if not code:
        return []
    code_clean = normalize_code_value(code)
    if not code_clean:
        return []

    d_keys = ["Mã D (D Part No.)", "D Part No.", "D Part No", "Mã D", "D Part No (D Part No.)"]
    p_keys = ["Mã P (P Part No.)", "P Part No.", "P Part No", "Mã P", "P Part No (P Part No.)"]
    tokin_keys = ["Mã Tokin (Tokin Part No.)", "Tokin Part No.", "Tokin Part No", "Mã Tokin", "SKU", "sku"]

    matched: List[ResourceItem] = []
    seen: set[str] = set()
    if code_clean.startswith("U"):
        for item in items:
            val = get_raw_value(item.raw, d_keys)
            if normalize_code_value(val) == code_clean:
                key = normalize_text(item.code or item.name)
                if key and key not in seen:
                    matched.append(item)
                    seen.add(key)
        return matched

    code_digits = extract_digits(code_clean)
    for item in items:
        item_key = normalize_text(item.code or item.name)
        if item_key in seen:
            continue

        sku_digits = extract_digits(str(item.code or ""))
        p_val = get_raw_value(item.raw, p_keys)
        d_val = get_raw_value(item.raw, d_keys)
        p_digits = extract_digits(str(p_val or ""))

        if code_clean.startswith("P"):
            if normalize_code_value(p_val) == code_clean or (code_digits and p_digits == code_digits):
                matched.append(item)
                seen.add(item_key)
                continue

        if code_digits and sku_digits == code_digits:
            matched.append(item)
            seen.add(item_key)
            continue
        if code_digits and p_digits == code_digits:
            matched.append(item)
            seen.add(item_key)
            continue
        if normalize_code_value(d_val) == code_clean:
            matched.append(item)
            seen.add(item_key)
            continue

        val = get_raw_value(item.raw, tokin_keys)
        if code_digits and extract_digits(str(val or "")) == code_digits:
            matched.append(item)
            seen.add(item_key)
    return matched


def detect_item_type(item: ResourceItem) -> str:
    """Purpose: Classify item usage as ROBOT or HAND by keyword.
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: "ROBOT" or "HAND".
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Defaults to HAND if keyword is absent.
    If Removed: Robot/hand filtering will be inaccurate.
    Testing Notes: Description containing "robot" yields ROBOT.
    """
    # Use normalized text to detect robot keyword.
    combined = normalize_text(f"{item.name} {item.description}")
    if "robot" in combined:
        return "ROBOT"
    return "HAND"


def detect_amp_line(text: str) -> str:
    """Purpose: Extract 350A/500A tags from text.
    Inputs/Outputs: Inputs: text (str). Outputs: amp string or "".
    Side Effects / State: None.
    Dependencies: normalize_text, regex literal for 350a/500a.
    Failure Modes: Returns empty string when no amp tag present.
    If Removed: Amp-based filtering and notes will degrade.
    Testing Notes: "350A" in text returns "350A".
    """
    # Match common amp markers for MIG lines.
    normalized = normalize_text(text)
    match = re.search(r"\b(350a|500a)\b", normalized)
    return match.group(1).upper() if match else ""


def infer_default_amp(items: List[ResourceItem]) -> str:
    """Purpose: Infer the most common amp across accessory categories.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: "350A", "500A", or "".
    Side Effects / State: None.
    Dependencies: _normalize_category, detect_amp_line.
    Failure Modes: Returns empty string when no amp is found in items.
    If Removed: Bundle fallback cannot pick a neutral amp baseline.
    Testing Notes: Catalog with more 350A entries should return "350A".
    """
    # Count amp occurrences across key accessory categories.
    counts = {"350A": 0, "500A": 0}
    for item in items:
        cat_norm = _normalize_category(item.category)
        if cat_norm not in {"TIPBODY", "INSULATOR", "NOZZLE", "ORIFICE"}:
            continue
        amp = detect_amp_line(f"{item.name} {item.description}")
        if amp in counts:
            counts[amp] += 1
    if max(counts.values()) > 0:
        return max(counts, key=counts.get)
    return ""


def merge_unique(primary: Optional[List[str]], secondary: Optional[List[str]]) -> List[str]:
    """Purpose: Merge two string lists while preserving order and uniqueness.
    Inputs/Outputs: Inputs: primary/secondary lists. Outputs: merged list.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Skips falsy values; returns empty list if both are empty.
    If Removed: Bundle role merging can duplicate or lose requested parts.
    Testing Notes: ["A","B"] + ["B","C"] -> ["A","B","C"].
    """
    # Preserve order while removing duplicates.
    seen = set()
    output: List[str] = []
    for value in (primary or []) + (secondary or []):
        if not value:
            continue
        key = str(value).upper()
        if key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def detect_item_group(item: ResourceItem) -> str:
    """Purpose: Determine the best-fit group label for an item.
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: group string or "".
    Side Effects / State: None.
    Dependencies: detect_product_group, item_matches_group, _normalize_category.
    Failure Modes: Returns empty string if no group can be derived.
    If Removed: Bundle role inference will not filter by category correctly.
    Testing Notes: Category "TIP BODY" should map to TIP_BODY.
    """
    # Resolve group from category/name/description signals.
    combined = normalize_text(f"{item.category} {item.name} {item.description}")
    group = detect_product_group(combined)
    if group:
        return group
    for group_key in GROUP_SYNONYMS:
        if item_matches_group(item, group_key):
            return group_key
    cat = _normalize_category(item.category or "")
    if cat in {"TIPBODY", "TIP_BODY"}:
        return "TIP_BODY"
    if cat == "INSULATOR":
        return "INSULATOR"
    if cat == "NOZZLE":
        return "NOZZLE"
    if cat in {"ORIFICE", "DIFFUSER"}:
        return "ORIFICE"
    if cat == "TIP":
        return "TIP"
    return ""


def infer_bundle_roles_from_catalog(
    items: List[ResourceItem],
    anchor: Optional[ResourceItem],
    target_amp: str,
    target_system: str,
    torch_type: str,
) -> List[str]:
    """Purpose: Infer accessory roles compatible with an anchor and constraints.
    Inputs/Outputs: Inputs: items, anchor, target_amp/system, torch_type. Outputs: list roles.
    Side Effects / State: None.
    Dependencies: detect_item_group, detect_amp_line, detect_system_tag, detect_item_type.
    Failure Modes: Returns empty list if no candidates pass constraints.
    If Removed: Bundle expansion will not suggest any related parts.
    Testing Notes: Anchor TIP should infer TIP_BODY/INSULATOR/NOZZLE when present.
    """
    # Filter candidate roles by amp/system/type compatibility.
    if not items:
        return []
    roles: set[str] = set()
    anchor_group = detect_item_group(anchor) if anchor else ""
    allowed = {"TIP_BODY", "INSULATOR", "NOZZLE", "ORIFICE"}
    for item in items:
        group = detect_item_group(item)
        if not group or group not in allowed:
            continue
        if anchor_group and group == anchor_group:
            continue
        if target_amp:
            amp_val = detect_amp_line(f"{item.name} {item.description}")
            if amp_val and amp_val != target_amp:
                continue
        if target_system:
            system_val = detect_system_tag(f"{item.name} {item.description}")
            if system_val and system_val != target_system:
                continue
        if torch_type and detect_item_type(item) != torch_type:
            continue
        roles.add(group)
    return sorted(roles)


def build_bundle_top_entries(
    items: List[ResourceItem],
    target_amp: str,
    target_system: str,
    torch_type: str,
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Purpose: Score and summarize bundle candidates for logging/diagnostics.
    Inputs/Outputs: Inputs: items, target_amp/system, torch_type, limit. Outputs: list of dicts.
    Side Effects / State: None.
    Dependencies: detect_amp_line, detect_system_tag, detect_item_type.
    Failure Modes: Returns empty list when no items are provided.
    If Removed: Retrieval logs lose top-k diagnostics for bundle debugging.
    Testing Notes: Ensure score ordering prefers matching amp/system/type.
    """
    # Build scored entries for logging and debugging bundle filters.
    entries: List[Dict[str, object]] = []
    for item in items:
        amp_val = detect_amp_line(f"{item.name} {item.description}")
        system_val = detect_system_tag(f"{item.name} {item.description}")
        item_type = detect_item_type(item)
        score = 0
        if target_amp and amp_val == target_amp:
            score += 3
        if target_system and system_val == target_system:
            score += 2
        if torch_type and item_type == torch_type:
            score += 1
        entry = {
            "code": item.code or "",
            "name": item.name or "",
            "cat": item.category or "",
            "amp": amp_val,
            "system": system_val,
            "type": item_type,
            "score": score,
        }
        entries.append(entry)
    entries.sort(key=lambda item: (item.get("score", 0), item.get("code", "")), reverse=True)
    return entries[:limit]


def detect_system_tag(text: str) -> str:
    """Purpose: Detect system tag token (N/D) from text.
    Inputs/Outputs: Inputs: text (str). Outputs: "N", "D", or "".
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns empty string if no token is found.
    If Removed: System-based filtering will lose a key constraint.
    Testing Notes: "he N" should return "N".
    """
    # Scan tokens for single-letter system markers.
    tokens = normalize_text(text).split()
    for token in tokens:
        if token in {"n", "d"}:
            return token.upper()
    return ""


def has_ambiguous_type(items: List[ResourceItem]) -> bool:
    """Purpose: Detect mixed robot/hand types in a result set.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: bool.
    Side Effects / State: None.
    Dependencies: detect_item_type.
    Failure Modes: Returns False when list is empty.
    If Removed: Ambiguity checks may not guard type conflicts.
    Testing Notes: Items with both ROBOT and HAND should return True.
    """
    # Identify if results include both ROBOT and HAND.
    types = {detect_item_type(item) for item in items if item}
    return "ROBOT" in types and "HAND" in types


def detect_topic(normalized: str) -> str:
    """Purpose: Map normalized message to high-level topic label.
    Inputs/Outputs: Inputs: normalized (str). Outputs: topic string.
    Side Effects / State: None.
    Dependencies: INFO_ONLY_RE, COMPATIBILITY_RE, LISTING_RE, PRICE_RE.
    Failure Modes: Defaults to "product" if no topic match.
    If Removed: Intent routing will lose topic hints for guards.
    Testing Notes: Text with "gia" should map to "commercial".
    """
    # Select a single topic label based on keyword rules.
    if INFO_ONLY_RE.search(normalized):
        return "origin"
    if COMPATIBILITY_RE.search(normalized):
        return "compatibility"
    if LISTING_RE.search(normalized):
        return "list"
    if PRICE_RE.search(normalized):
        return "commercial"
    return "product"


def detect_buy_intent(normalized: str, quantity: Optional[int]) -> bool:
    """Purpose: Determine whether the user expresses buying intent.
    Inputs/Outputs: Inputs: normalized (str), quantity (int|None). Outputs: bool.
    Side Effects / State: None.
    Dependencies: BUY_INTENT_RE, CLOSE_INTENT_RE.
    Failure Modes: Returns False for neutral messages.
    If Removed: Contact-form logic will not be triggered correctly.
    Testing Notes: "mua 10 cai" should return True.
    """
    # Treat explicit quantity or buy verbs as intent to purchase.
    if quantity:
        return True
    if BUY_INTENT_RE.search(normalized):
        return True
    return bool(CLOSE_INTENT_RE.search(normalized))


def build_intent_state(chat_history: List[dict], user_message: str, order_state: Dict[str, object]) -> Dict[str, object]:
    """Purpose: Build compact state for LLM intent detection prompt.
    Inputs/Outputs: Inputs: chat_history (list), user_message (str), order_state (dict).
        Outputs: dict with flags and selected slots.
    Side Effects / State: None; prepares data only.
    Dependencies: _get_contact_state, detect_contact_info, normalize_order_state.
    Failure Modes: Missing history yields default flags; caller still proceeds.
    If Removed: Intent prompt loses state context and degrades routing.
    Testing Notes: When form was asked, asked_form should be True.
    """
    # Summarize conversational state for the intent prompt.
    has_asked_type = any(
        msg.get("role") == "assistant"
        and "tay" in normalize_text(msg.get("content", ""))
        and "robot" in normalize_text(msg.get("content", ""))
        and (
            "hay" in normalize_text(msg.get("content", ""))
            or "hoac" in normalize_text(msg.get("content", ""))
        )
        for msg in chat_history
    )
    asked_form, reminder_count, contact_received, waiting_for_contact = _get_contact_state(chat_history)
    has_contact_info = contact_received or detect_contact_info(user_message)
    state = normalize_order_state(order_state)
    short_memory = normalize_short_memory(order_state)
    return {
        "has_asked_type": has_asked_type,
        "has_contact_info": has_contact_info,
        "waiting_for_contact": waiting_for_contact and not has_contact_info,
        "asked_form": asked_form,
        "reminder_count": reminder_count,
        "selected_sku": state.get("selected_sku"),
        "selected_group": state.get("selected_group"),
        "quantity": state.get("quantity"),
        "asked_hand_robot": state.get("asked_hand_robot"),
        "asked_contact_form": state.get("asked_contact_form"),
        "short_memory": sanitize_short_memory_for_log(short_memory),
    }


def parse_intent_output(raw: str, message: str, state: Dict[str, object], order_state: Dict[str, object]) -> IntentDecision:
    """Purpose: Parse LLM intent JSON and merge with rule-based signals.
    Inputs/Outputs: Inputs: raw (str), message (str), state (dict), order_state (dict).
        Outputs: IntentDecision.
    Side Effects / State: None; returns a new decision object.
    Dependencies: safe_json_loads, extract_* helpers, detect_* helpers.
    Failure Modes: Invalid JSON falls back to rule-based defaults.
    If Removed: LLM intent outputs cannot be interpreted, breaking routing.
    Testing Notes: Non-JSON output should still produce a valid decision.
    """
    # Combine LLM output with deterministic guards and slots.
    normalized = normalize_text(message)
    quantity = extract_quantity(normalized)
    codes, primary_code = extract_codes(message)
    requested_parts, expand_bundle = extract_requested_parts(message)
    is_accessory = bool(RELATED_QUERY_RE.search(normalized))
    is_bundle = is_accessory_bundle_query(message)
    is_list = bool(LISTING_RE.search(normalized))
    fallback = IntentDecision(
        intent="PRODUCT_AVAILABILITY" if is_availability_query(message) else "OTHER",
        buy_intent=detect_buy_intent(normalized, quantity),
        info_only=is_info_only_query(message),
        topic=detect_topic(normalized),
        entities={},
        missing=[],
        next_action="ANSWER_ONLY",
    )

    data = safe_json_loads(raw or "")
    if not isinstance(data, dict):
        data = {}

    intent = str(data.get("intent") or fallback.intent).strip().upper() or fallback.intent
    if is_availability_query(message):
        intent = "PRODUCT_AVAILABILITY"
    if is_amp_only_message(message):
        intent = "SLOT_FILL_AMP"
    if is_bundle:
        intent = "ACCESSORY_BUNDLE_LOOKUP"
    elif codes and is_accessory:
        intent = "ACCESSORY_LOOKUP"
    elif codes:
        intent = "CODE_LOOKUP"
    elif is_list:
        intent = "LIST"
    if is_quantity_followup_message(message) and not codes and not detect_product_group(normalized):
        intent = "QUANTITY_FOLLOWUP"
    buy_intent = bool(data.get("buy_intent")) if "buy_intent" in data else fallback.buy_intent
    info_only = is_info_only_query(message)
    topic = str(data.get("topic") or fallback.topic).strip().lower() or fallback.topic
    if topic not in {"product", "origin", "compatibility", "list", "commercial"}:
        topic = fallback.topic
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    if is_quantity_followup_message(message):
        topic = "product"
    if is_amp_only_message(message):
        topic = "product"
    if intent in {
        "LIST",
        "LIST_REQUEST",
        "PRODUCT_LOOKUP",
        "ACCESSORY_LOOKUP",
        "ACCESSORY_BUNDLE_LOOKUP",
        "CODE_LOOKUP",
        "PRODUCT_AVAILABILITY",
        "QUANTITY_FOLLOWUP",
        "SLOT_FILL_AMP",
    }:
        info_only = False
    if LISTING_RE.search(normalized) or CODE_RE.search(message) or RELATED_QUERY_RE.search(normalized) or quantity is not None:
        info_only = False

    skus = entities.get("skus") if isinstance(entities.get("skus"), list) else None
    if not skus:
        skus = extract_skus(message)
    skus = [str(code).strip() for code in skus if str(code).strip()]
    primary_code = str(entities.get("primary_code") or primary_code or "").strip()
    codes = entities.get("codes") if isinstance(entities.get("codes"), list) else codes
    codes = [str(code).strip() for code in codes if str(code).strip()]

    anchor_used = bool(data.get("anchor_used")) if "anchor_used" in data else False
    quantity_val = entities.get("quantity")
    if quantity_val is None and isinstance(data.get("quantity"), int):
        quantity_val = data.get("quantity")
    if isinstance(quantity_val, str) and quantity_val.isdigit():
        quantity_val = int(quantity_val)
    if not isinstance(quantity_val, int):
        quantity_val = quantity

    is_robot = bool(entities.get("is_robot")) if "is_robot" in entities else ("robot" in normalized)
    is_hand = bool(entities.get("is_hand")) if "is_hand" in entities else ("tay" in normalized)
    product_group = entities.get("product_group")
    if isinstance(product_group, str):
        product_group = product_group.strip().upper()
    if not product_group:
        product_group = detect_product_group(normalized)
    if "cach dien" in normalized or "insulator" in normalized:
        product_group = "INSULATOR"
        if "INSULATOR" not in requested_parts:
            requested_parts = merge_unique(requested_parts, ["INSULATOR"])

    amp = entities.get("amp")
    if isinstance(amp, str):
        amp = amp.strip().upper()
    if not amp:
        match = AMP_ANY_RE.search(normalized)
        if match:
            amp = f"{match.group(1)}A"
    thread = entities.get("thread")
    material = entities.get("material")
    system = entities.get("system")

    entities = {
        "skus": skus,
        "quantity": quantity_val,
        "is_robot": is_robot,
        "is_hand": is_hand,
        "product_group": product_group,
        "amp": amp,
        "thread": thread,
        "material": material,
        "system": system,
        "primary_code": primary_code,
        "codes": codes,
        "required_categories": requested_parts,
        "bundle_hint": bool(BUNDLE_QUERY_RE.search(normalized)),
        "expand_bundle": expand_bundle,
        "anchor_used": anchor_used,
    }

    missing = data.get("missing") if isinstance(data.get("missing"), list) else []
    missing = [str(item).strip().lower() for item in missing if str(item).strip()]
    missing = ["contact" if item in {"contact_info", "contactinfo", "phone"} else item for item in missing]
    if intent in {"LIST", "LIST_REQUEST", "ACCESSORY_BUNDLE_LOOKUP", "SLOT_FILL_AMP"} and "tay_robot" in missing:
        missing = [item for item in missing if item != "tay_robot"]

    commercial_action = parse_commercial_action(data)

    order_state = normalize_order_state(order_state)
    has_contact_info = bool(state.get("has_contact_info")) or bool(order_state.get("contact"))
    has_type_info = is_robot or is_hand or bool(state.get("has_asked_type"))
    has_selected = bool(order_state.get("selected_sku") or order_state.get("selected_group") or skus)

    if (buy_intent or topic == "commercial") and "quantity" not in missing and quantity_val is None:
        missing.append("quantity")
    if "sku" not in missing and not skus and not order_state.get("selected_sku"):
        missing.append("sku")
    if buy_intent and "contact" not in missing and not has_contact_info:
        missing.append("contact")
    if commercial_action.get("collect_contact") and "contact" not in missing and not has_contact_info:
        missing.append("contact")
    if not has_type_info and "tay_robot" not in missing:
        missing.append("tay_robot")
    if intent == "CODE_LOOKUP" and "sku" in missing:
        missing = [item for item in missing if item != "sku"]

    next_action = str(data.get("next_action") or fallback.next_action).strip().upper() or fallback.next_action
    if commercial_action.get("collect_contact") and not has_contact_info:
        next_action = "REQUEST_CONTACT_FORM"
    if is_quantity_followup_message(message) and has_selected and not has_contact_info:
        next_action = "REQUEST_CONTACT_FORM"
        if "contact" not in missing:
            missing.append("contact")

    decision = IntentDecision(
        intent=intent,
        buy_intent=buy_intent,
        info_only=info_only,
        topic=topic,
        entities=entities,
        missing=missing,
        next_action=next_action,
        commercial_action=commercial_action,
    )

    if is_pure_quantity_message(message) and has_selected:
        decision.buy_intent = True
        decision.info_only = False
        decision.entities["quantity"] = quantity_val or parse_pure_quantity_value(message) or extract_quantity(normalized)
        decision.missing = [item for item in decision.missing if item not in {"sku", "quantity"}]
    return apply_intent_rules(decision, normalized)


def apply_intent_rules(decision: IntentDecision, normalized_message: str) -> IntentDecision:
    """Purpose: Apply deterministic routing rules to an intent decision.
    Inputs/Outputs: Inputs: decision (IntentDecision), normalized_message (str).
        Outputs: IntentDecision with adjusted next_action/info_only.
    Side Effects / State: None; returns a new decision object.
    Dependencies: decision fields and rule constants.
    Failure Modes: Falls back to ANSWER_ONLY on invalid next_action.
    If Removed: Model output may violate hard routing constraints.
    Testing Notes: LIST intents must always keep ANSWER_ONLY action.
    """
    # Enforce hard rules that override model-proposed actions.
    allowed_actions = {
        "ANSWER_ONLY",
        "ASK_FOR_SKU_OR_GROUP",
        "ASK_HAND_VS_ROBOT_ONCE",
        "REQUEST_CONTACT_FORM",
        "COMMERCIAL_NEUTRAL_REPLY",
    }
    info_only = decision.info_only
    commercial_action = decision.commercial_action or {}
    if decision.intent in {
        "LIST",
        "LIST_REQUEST",
        "PRODUCT_LOOKUP",
        "ACCESSORY_LOOKUP",
        "ACCESSORY_BUNDLE_LOOKUP",
        "CODE_LOOKUP",
        "QUANTITY_FOLLOWUP",
        "SLOT_FILL_AMP",
    }:
        info_only = False
    next_action = decision.next_action if decision.next_action in allowed_actions else "ANSWER_ONLY"

    if info_only:
        next_action = "ANSWER_ONLY"
    elif decision.intent in {"LIST", "LIST_REQUEST"}:
        next_action = "ANSWER_ONLY"
    elif decision.intent in {"ACCESSORY_LOOKUP", "ACCESSORY_BUNDLE_LOOKUP"}:
        next_action = "ANSWER_ONLY"
    elif decision.intent == "SLOT_FILL_AMP":
        next_action = "ANSWER_ONLY"
    elif decision.intent == "QUANTITY_FOLLOWUP":
        next_action = "REQUEST_CONTACT_FORM" if commercial_action.get("collect_contact") else "ANSWER_ONLY"
    elif decision.intent == "PRODUCT_AVAILABILITY":
        next_action = "ANSWER_ONLY"
    elif decision.intent == "CODE_LOOKUP" and not decision.buy_intent:
        next_action = "ANSWER_ONLY"
    elif decision.buy_intent and "sku" in decision.missing:
        next_action = "ASK_FOR_SKU_OR_GROUP"
    elif decision.buy_intent and "contact" in decision.missing and decision.topic != "commercial":
        next_action = "REQUEST_CONTACT_FORM"
    elif decision.topic == "commercial":
        next_action = "COMMERCIAL_NEUTRAL_REPLY"
    elif "tay_robot" in decision.missing:
        next_action = "ASK_HAND_VS_ROBOT_ONCE"

    return IntentDecision(
        intent=decision.intent,
        buy_intent=decision.buy_intent,
        info_only=info_only,
        topic=decision.topic,
        entities=decision.entities,
        missing=decision.missing,
        next_action=next_action,
    )


def build_product_data(items: List[ResourceItem]) -> str:
    """Purpose: Build a structured text block from items for the LLM prompt.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: str block.
    Side Effects / State: None.
    Dependencies: get_raw_value; used by generation prompt assembly.
    Failure Modes: Returns empty string when items are empty.
    If Removed: LLM prompt loses grounded catalog context, increasing hallucinations.
    Testing Notes: Items should produce SKU/CAT/NAME/IMG/SPECS lines.
    """
    # Assemble a compact catalog context for the LLM.
    if not items:
        return ""
    blocks: List[str] = []
    for item in items:
        sku = item.code or "N/A"
        category = item.category or "Phụ kiện"
        name = item.name or "Sản phẩm Tokinarc"
        image = item.link or ""
        size = get_raw_value(item.raw, ["Kích thước dây (Size mm)"])
        length = get_raw_value(item.raw, ["Tổng chiều dài (mm)"])
        thread = get_raw_value(item.raw, ["Ren (Thread)"])
        size_text = str(size) if size not in (None, "") else "N/A"
        length_text = str(length) if length not in (None, "") else "N/A"
        thread_text = str(thread) if thread not in (None, "") else "N/A"

        blocks.append(
            "\n".join(
                [
                    f"SKU: {sku}",
                    f"CAT: {category}",
                    f"NAME: {name}",
                    f"IMG: {image}",
                    f"SPECS: Size {size_text}, Dài {length_text}, Ren {thread_text}",
                ]
            )
        )
    return "\n---\n".join(blocks)


def sku_key_for_group(item: ResourceItem) -> str:
    """Purpose: Build a stable key for grouping items by SKU and category.
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: key string.
    Side Effects / State: None.
    Dependencies: extract_digits, normalize_text, _normalize_category.
    Failure Modes: Returns empty string when no identifiers are present.
    If Removed: Grouping/deduplication will not preserve category context.
    Testing Notes: "Tokin 036001" with TIP BODY should yield a stable key.
    """
    # Combine numeric SKU and normalized category for grouping.
    code = item.code or ""
    digits = extract_digits(code)
    base = digits or normalize_text(code or item.name or "")
    category = _normalize_category(item.category or "")
    if category:
        return f"{base}|{category}"
    return base


def dedupe_by_sku(items: List[ResourceItem]) -> List[ResourceItem]:
    """Purpose: Deduplicate items by SKU+category key while keeping order.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: list[ResourceItem].
    Side Effects / State: None.
    Dependencies: sku_key_for_group.
    Failure Modes: Returns empty list when input is empty.
    If Removed: Bundle output may repeat identical SKUs.
    Testing Notes: Two items with same SKU+category should collapse to one.
    """
    # Drop duplicate SKU/category pairs while preserving order.
    seen: set[str] = set()
    result: List[ResourceItem] = []
    for item in items:
        key = sku_key_for_group(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def extract_variant_label(item: ResourceItem) -> str:
    """Purpose: Derive a variant label (A-D) from item text or image URL.
    Inputs/Outputs: Inputs: item (ResourceItem). Outputs: label string or "".
    Side Effects / State: None.
    Dependencies: re, item.name/description/link.
    Failure Modes: Returns empty string when no label is detected.
    If Removed: Variant grouping loses human-readable labels.
    Testing Notes: URL ending with "-A" should return "A".
    """
    # Look for A-D markers in text or URL.
    texts = [item.name or "", item.description or ""]
    for text in texts:
        match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    link = item.link or ""
    match = re.search(r"(?:-|_)([A-D])(?:\.|_|-)", link, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def build_bundle_product_data(items: List[ResourceItem]) -> str:
    """Purpose: Build grouped bundle data (with variants) for LLM prompts.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: str block.
    Side Effects / State: None.
    Dependencies: sku_key_for_group, extract_variant_label, get_raw_value.
    Failure Modes: Returns empty string when no items are provided.
    If Removed: Bundle prompt loses structured grouping and variant hints.
    Testing Notes: Multiple items with same SKU should yield VARIANTS lines.
    """
    # Group items by SKU/category and expose variant links.
    if not items:
        return ""
    grouped: Dict[str, List[ResourceItem]] = {}
    for item in items:
        key = sku_key_for_group(item)
        if not key:
            continue
        grouped.setdefault(key, []).append(item)

    blocks: List[str] = []
    for group_items in grouped.values():
        base = group_items[0]
        sku = base.code or "N/A"
        category = base.category or "Phu kien"
        name = base.name or "San pham Tokinarc"

        variant_parts: List[str] = []
        used_labels: set[str] = set()
        used_links: set[str] = set()
        for item in group_items:
            if not item.link:
                continue
            if item.link in used_links:
                continue
            label = extract_variant_label(item)
            if not label or label in used_labels:
                label = chr(ord("A") + len(used_labels))
            if label in used_labels:
                continue
            used_labels.add(label)
            used_links.add(item.link)
            variant_parts.append(f"{label}:{item.link}")

        size = get_raw_value(base.raw, ["Kich thuoc day (Size mm)", "Kích thước dây (Size mm)"])
        length = get_raw_value(base.raw, ["Tong chieu dai (mm)", "Tổng chiều dài (mm)"])
        thread = get_raw_value(base.raw, ["Ren (Thread)"])
        size_text = str(size) if size not in (None, "") else "N/A"
        length_text = str(length) if length not in (None, "") else "N/A"
        thread_text = str(thread) if thread not in (None, "") else "N/A"

        lines = [
            f"SKU: {sku}",
            f"CAT: {category}",
            f"NAME: {name}",
        ]
        if variant_parts:
            lines.append("VARIANTS: " + "; ".join(variant_parts))
        else:
            lines.append(f"IMG: {base.link or ''}")
        lines.append(f"SPECS: Size {size_text}, Dai {length_text}, Ren {thread_text}")
        blocks.append("\n".join(lines))

    return "\n---\n".join(blocks)


def build_product_card(item: ResourceItem, include_type_line: bool = True) -> str:
    """Purpose: Render a single product card in Markdown.
    Inputs/Outputs: Inputs: item (ResourceItem), include_type_line (bool). Outputs: str.
    Side Effects / State: None.
    Dependencies: get_raw_value, detect_amp_line, detect_system_tag, sanitize_alt_text.
    Failure Modes: Returns minimal card when fields are missing.
    If Removed: Product rendering will lose consistent card formatting.
    Testing Notes: Item with image should include Markdown image line.
    """
    # Compose a Markdown card with optional type line.
    sku = item.code or "N/A"
    name = item.name or "Sản phẩm Tokinarc"
    lines: List[str] = [f"**{name} ({sku})**"]

    if item.link:
        alt = sanitize_alt_text(name or sku)
        lines.append(f"![{alt}]({item.link})")

    spec_parts: List[str] = []
    size = get_raw_value(item.raw, ["Kích thước dây (Size mm)"])
    length = get_raw_value(item.raw, ["Tổng chiều dài (mm)"])
    thread = get_raw_value(item.raw, ["Ren (Thread)"])
    if size not in (None, ""):
        spec_parts.append(f"Size {size}")
    if length not in (None, ""):
        spec_parts.append(f"Dài {length}")
    if thread not in (None, ""):
        spec_parts.append(f"Ren {thread}")
    if spec_parts:
        lines.append("Thông số: " + ", ".join(spec_parts))

    if include_type_line:
        type_parts: List[str] = []
        if item.category:
            type_parts.append(f"Loại: {item.category}")
        amp = detect_amp_line(f"{item.name} {item.description}")
        system = detect_system_tag(f"{item.name} {item.description}")
        line_value = " ".join(part for part in [amp, system] if part).strip()
        if line_value:
            type_parts.append(f"Dòng: {line_value}")
        if type_parts:
            lines.append(" | ".join(type_parts))

    if not item.link:
        lines.append(MISSING_IMAGE_NOTICE)

    return "\n".join(lines).strip()


def render_product_cards(items: List[ResourceItem], limit: int = 3, include_type_line: bool = True) -> str:
    """Purpose: Render multiple product cards with a size limit.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), limit (int). Outputs: str.
    Side Effects / State: None.
    Dependencies: build_product_card.
    Failure Modes: Returns empty string when items are empty.
    If Removed: Card rendering in responses will be inconsistent.
    Testing Notes: limit=2 should render at most two cards.
    """
    # Render up to "limit" cards and join with blank lines.
    if not items:
        return ""
    cards = [build_product_card(item, include_type_line=include_type_line) for item in items[:limit]]
    return "\n\n".join(card for card in cards if card).strip()


def format_sku_display(code: str) -> str:
    """Purpose: Format SKU for display with Tokin prefix if numeric.
    Inputs/Outputs: Inputs: code (str). Outputs: display string.
    Side Effects / State: None.
    Dependencies: extract_digits.
    Failure Modes: Returns empty string for empty input.
    If Removed: Displayed SKU text may lose consistent formatting.
    Testing Notes: "004002" should format as "Tokin 004002".
    """
    # Normalize numeric SKU display for readability.
    if not code:
        return ""
    digits = extract_digits(code)
    if digits:
        return f"Tokin {digits}"
    return code.strip()


def build_hand_robot_note(items: List[ResourceItem]) -> str:
    """Purpose: Build a hand/robot note based on detected amp.
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: note string.
    Side Effects / State: None.
    Dependencies: detect_amp_line.
    Failure Modes: Uses generic scope when amp is missing.
    If Removed: Responses may omit required hand/robot note.
    Testing Notes: Items with 350A should mention MIG 350A scope.
    """
    # Prefer the first detected amp to build a scope note.
    amp = ""
    for item in items:
        amp = detect_amp_line(f"{item.name} {item.description}")
        if amp:
            break
    scope = f"MIG {amp}" if amp else "MIG thông dụng"
    return (
        "Dạ vâng ạ, hiện em đang tư vấn theo bộ phụ kiện cho súng hàn tay "
        f"{scope}. Nếu Anh/Chị dùng súng hàn robot, Anh/Chị báo giúp em để em "
        "đối chiếu và chọn đúng mã phù hợp ạ."
    )


def render_lookup_cards(items: List[ResourceItem], limit: int = 2) -> str:
    """Purpose: Render product lookup content with origin and hand/robot note.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), limit (int). Outputs: str.
    Side Effects / State: None.
    Dependencies: render_product_lookup_lines, build_hand_robot_note.
    Failure Modes: Returns fallback text when items are empty.
    If Removed: PRODUCT_LOOKUP responses become too sparse.
    Testing Notes: Non-empty items should include origin and note lines.
    """
    # Compose lookup output with origin and note.
    if not items:
        return CODE_LOOKUP_NOT_FOUND_REPLY

    output: List[str] = []
    output.extend(render_product_lookup_lines(items, limit=limit))
    output.append("")
    output.append("Xuất xứ: Tokinarc – Nhật Bản")
    output.append(build_hand_robot_note(items))

    return "\n".join(output).strip() or CODE_LOOKUP_NOT_FOUND_REPLY


def render_product_lookup_lines(items: List[ResourceItem], limit: int = 2) -> List[str]:
    """Purpose: Render bullet-style lines for product lookup.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), limit (int). Outputs: list[str].
    Side Effects / State: None.
    Dependencies: format_sku_display, sanitize_alt_text, get_raw_value.
    Failure Modes: Skips items without names; returns empty list.
    If Removed: Lookup formatting will lose consistent bullet layout.
    Testing Notes: Items should render bullet, image, and spec lines.
    """
    # Build a list of bullet lines for lookup responses.
    output: List[str] = []
    for item in items[:limit]:
        raw_sku = (item.code or "").strip()
        name = (item.name or "").strip()
        sku_display = format_sku_display(raw_sku)
        if not name:
            continue
        if sku_display:
            output.append(f"- **{name} ({sku_display})**")
        else:
            output.append(f"- **{name}**")

        if item.link:
            alt = sanitize_alt_text(name or sku_display or raw_sku)
            output.append(f"![{alt}]({item.link})")

        spec_parts: List[str] = []
        size = get_raw_value(item.raw, ["Kích thước dây (Size mm)"])
        length = get_raw_value(item.raw, ["Tổng chiều dài (mm)"])
        thread = get_raw_value(item.raw, ["Ren (Thread)"])
        if size not in (None, ""):
            spec_parts.append(f"Size {size}")
        if length not in (None, ""):
            spec_parts.append(f"Dài {length}")
        if thread not in (None, ""):
            spec_parts.append(f"Ren {thread}")
        if spec_parts:
            output.append("Thông số: " + ", ".join(spec_parts))

        output.append("")
    return output


def format_tokin(code: str) -> str:
    """Purpose: Format a Tokin code with prefix if digits are present.
    Inputs/Outputs: Inputs: code (str). Outputs: display string.
    Side Effects / State: None.
    Dependencies: extract_digits.
    Failure Modes: Returns stripped input when no digits are found.
    If Removed: Mapping sentences may show inconsistent code formats.
    Testing Notes: "Tokin 002005" should format to "Tokin 002005".
    """
    # Normalize code into Tokin-prefixed display.
    digits = extract_digits(code or "")
    if digits:
        return f"Tokin {digits}"
    return (code or "").strip()


def build_anchor_context(anchor: ResourceItem) -> str:
    """Purpose: Build an anchor context line for bundle responses.
    Inputs/Outputs: Inputs: anchor (ResourceItem). Outputs: str line.
    Side Effects / State: None.
    Dependencies: format_tokin, detect_amp_line, detect_system_tag.
    Failure Modes: Returns generic wording when fields are missing.
    If Removed: Bundle responses lose anchor context and mapping clarity.
    Testing Notes: Anchor with amp/system should include both tags.
    """
    # Summarize anchor item with optional tags.
    sku = format_tokin(anchor.code or "")
    name = " ".join((anchor.name or "").split()) or "Sản phẩm"
    combined = f"{anchor.name} {anchor.description}"
    amp = detect_amp_line(combined)
    system = detect_system_tag(combined)

    tags: List[str] = []
    if anchor.category:
        tags.append(anchor.category)
    if amp:
        tags.append(amp)
    if system:
        tags.append(f"hệ {system}")

    tag_text = f" ({', '.join(tags)})" if tags else ""
    return f"Dạ vâng ạ, **{name} ({sku})**{tag_text}."


def build_mapping_sentence(anchor: ResourceItem, target_group: str) -> str:
    """Purpose: Build a mapping sentence from anchor to a target group.
    Inputs/Outputs: Inputs: anchor (ResourceItem), target_group (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: detect_amp_line, detect_system_tag.
    Failure Modes: Falls back to generic mapping when tags are absent.
    If Removed: Bundle responses lose the mapping explanation.
    Testing Notes: With amp+system, sentence should mention both.
    """
    # Explain how the anchor maps to the requested accessory group.
    combined = f"{anchor.name} {anchor.description}"
    amp = detect_amp_line(combined)
    system = detect_system_tag(combined)

    group_vi = {
        "NOZZLE": "chụp khí",
        "TIP_BODY": "thân giữ béc",
        "INSULATOR": "cách điện",
        "ORIFICE": "sứ phân phối khí",
        "TIP": "béc hàn",
    }.get(target_group, "linh kiện")

    if amp and system:
        return f"→ Mã này dùng **{group_vi} hệ {system} cho dòng {amp}** tương ứng."
    if amp and not system:
        return f"→ Mã này thuộc dòng **{amp}**, em sẽ đối chiếu và ưu tiên **{group_vi} {amp}** phù hợp."
    if system and not amp:
        return f"→ Mã này thuộc **hệ {system}**, em sẽ đối chiếu và chọn **{group_vi} hệ {system}** phù hợp."
    return f"→ Em sẽ đối chiếu để chọn đúng **{group_vi}** tương thích theo thông tin của mã này ạ."


def render_code_lookup(item: ResourceItem, queried: str) -> str:
    """Purpose: Render a code-lookup response for a single matched item.
    Inputs/Outputs: Inputs: item (ResourceItem), queried (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: format_tokin, sanitize_alt_text, get_raw_value.
    Failure Modes: Returns minimal text if item fields are missing.
    If Removed: CODE_LOOKUP outputs lose consistent formatting.
    Testing Notes: D-code query should include mapping sentence and item line.
    """
    # Format mapping and item details for code lookup.
    sku = format_tokin(item.code)
    queried_clean = " ".join((queried or "").split())
    if queried_clean:
        line = f"Dạ vâng ạ, mã {queried_clean} tương đương {sku}."
    else:
        line = f"Dạ vâng ạ, mã này tương đương {sku}."

    name = " ".join((item.name or "").split())
    out = [line, f"• {name} – {sku}"]
    if item.link:
        alt = sanitize_alt_text(name or sku)
        out.append(f"![{alt}]({item.link})")

    spec_parts: List[str] = []
    size = get_raw_value(item.raw, ["Kích thước dây (Size mm)"])
    length = get_raw_value(item.raw, ["Tổng chiều dài (mm)"])
    thread = get_raw_value(item.raw, ["Ren (Thread)"])
    if size not in (None, ""):
        spec_parts.append(f"Size {size}")
    if length not in (None, ""):
        spec_parts.append(f"Dài {length}")
    if thread not in (None, ""):
        spec_parts.append(f"Ren {thread}")
    if spec_parts:
        out.append("Thông số: " + ", ".join(spec_parts))
    return "\n".join(out).strip()


def render_accessory_group_lines(label: str, items: List[ResourceItem], limit: int = 2) -> List[str]:
    """Purpose: Render labeled accessory group lines for output.
    Inputs/Outputs: Inputs: label (str), items (list[ResourceItem]), limit (int). Outputs: list[str].
    Side Effects / State: None.
    Dependencies: render_product_lookup_lines.
    Failure Modes: Returns empty list if items are empty.
    If Removed: Grouped accessory sections will be missing.
    Testing Notes: Label should appear before group item lines.
    """
    # Prepend group label and render item lines.
    if not items:
        return []
    lines = [f"{label}:"]
    lines.extend(render_product_lookup_lines(items, limit=limit))
    return lines


def desired_accessory_groups(message: str) -> List[str]:
    """Purpose: Determine desired accessory groups from message text.
    Inputs/Outputs: Inputs: message (str). Outputs: list[str] group order.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns default group list if no keywords found.
    If Removed: Accessory responses may not prioritize user-requested parts.
    Testing Notes: "cach dien" should return ["INSULATOR"].
    """
    # Use keyword matches to order accessory groups.
    normalized = normalize_text(message)
    groups = []
    if "than giu" in normalized or "tip body" in normalized:
        groups.append("TIP_BODY")
    if "cach dien" in normalized or "insulator" in normalized:
        groups.append("INSULATOR")
    if "chup" in normalized or "nozzle" in normalized:
        groups.append("NOZZLE")
    if "su" in normalized or "orifice" in normalized:
        groups.append("ORIFICE")
    if groups:
        return groups
    return ["TIP_BODY", "INSULATOR", "NOZZLE", "ORIFICE"]


def build_missing_bundle_question(target_amp: str, target_system: str, missing_text: str) -> str:
    """Purpose: Build a follow-up question when bundle parts are missing.
    Inputs/Outputs: Inputs: target_amp (str), target_system (str), missing_text (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Returns generic prompt when amp/system are unknown.
    If Removed: Bundle missing-part responses lose a focused follow-up question.
    Testing Notes: With amp known, should ask for system; without amp, ask for amp.
    """
    # Ask only for the next missing constraint in bundle flow.
    if not target_amp:
        return (
            f"Anh/Chị cho em xin dòng Ampe 350A hay 500A (và nếu có hệ N/D) để em lọc đúng {missing_text} ạ."
        )
    if not target_system:
        return f"Anh/Chị cho em xin hệ N/D đang dùng để em lọc đúng {missing_text} ạ."
    return (
        f"Anh/Chị cho em xin model cổ súng/torch hoặc gửi hình ảnh để em đối chiếu đúng {missing_text} ạ."
    )


def build_ambiguous_amp_question(ambiguous_text: str) -> str:
    """Purpose: Build a clarification question when amp variants are ambiguous.
    Inputs/Outputs: Inputs: ambiguous_text (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Returns a generic question when text is empty.
    If Removed: Ambiguous amp cases may return the wrong variant.
    Testing Notes: Should ask for 350A or 500A selection.
    """
    # Ask for amp selection when multiple variants exist.
    return (
        f"Hiện em thấy nhiều tùy chọn 350A/500A cho {ambiguous_text}. "
        "Anh/Chị cho em xin dòng Ampe đang dùng để em chọn đúng mã ạ."
    )


def render_accessory_lookup(
    message: str,
    anchor: ResourceItem,
    related_items: List[ResourceItem],
    origin: str,
    note: str,
    target_groups: Optional[List[str]] = None,
    missing_groups: Optional[List[str]] = None,
    ambiguous_groups: Optional[List[str]] = None,
    target_amp: str = "",
    target_system: str = "",
) -> str:
    """Purpose: Render accessory bundle response grouped by requested parts.
    Inputs/Outputs: Inputs: message, anchor, related_items, origin, note, target_groups, missing_groups.
        Outputs: str response.
    Side Effects / State: None.
    Dependencies: build_anchor_context, build_mapping_sentence, render_product_lookup_lines.
    Failure Modes: If related_items empty, returns a guided missing-part response.
    If Removed: ACCESSORY_BUNDLE_LOOKUP responses will lack structured grouping.
    Testing Notes: With TIP_BODY+INSULATOR, output should include both groups.
    """
    # Compose grouped accessory sections with missing-part prompts.
    lines: List[str] = []
    if anchor:
        lines.append(build_anchor_context(anchor))

    group_order = [group for group in (target_groups or desired_accessory_groups(message)) if group]
    label_map = {
        "TIP_BODY": "than giu bec",
        "INSULATOR": "cach dien",
        "NOZZLE": "chup khi",
        "ORIFICE": "su phan phoi khi",
    }

    missing = merge_unique(missing_groups or [], [])
    ambiguous = merge_unique(ambiguous_groups or [], [])
    for group in group_order:
        group_items = [item for item in related_items if _normalize_category(item.category) == group.replace("_", "")]
        group_items = dedupe_by_sku(group_items)
        if not group_items:
            missing = merge_unique(missing, [group])
            continue
        lines.append(build_mapping_sentence(anchor, group))
        label = label_map.get(group, "linh kien")
        lines.append(f"Gợi ý {label} Tokinarc phù hợp:")
        lines.extend(render_product_lookup_lines(group_items, limit=4))
        lines.append("")

    if missing:
        missing_labels = [label_map.get(group, group.lower()) for group in missing]
        missing_text = ", ".join(missing_labels)
        lines.append(
            f"Hiện em chưa thấy dữ liệu {missing_text} đi kèm theo mốc này trong danh mục em đang tra ạ."
        )
        lines.append(build_missing_bundle_question(target_amp, target_system, missing_text))
    if ambiguous:
        ambiguous_labels = [label_map.get(group, group.lower()) for group in ambiguous]
        ambiguous_text = ", ".join(ambiguous_labels)
        lines.append(build_ambiguous_amp_question(ambiguous_text))

    if origin:
        lines.append(origin)
    if note:
        lines.append(note)
    return "\n".join(part for part in lines if part).strip()


def find_item_by_selected_sku(items: List[ResourceItem], selected_sku: str) -> Optional[ResourceItem]:
    """Purpose: Find the first catalog item matching a selected SKU.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), selected_sku (str). Outputs: ResourceItem or None.
    Side Effects / State: None.
    Dependencies: normalize_text, extract_digits.
    Failure Modes: Returns None when no match is found.
    If Removed: Quantity follow-ups cannot anchor to the selected SKU.
    Testing Notes: Selected SKU "004002" should match item.code "Tokin 004002".
    """
    # Match by normalized code or numeric digits.
    if not items or not selected_sku:
        return None
    selected_norm = normalize_text(selected_sku)
    selected_digits = extract_digits(selected_sku)
    for item in items:
        if not item.code:
            continue
        if normalize_text(item.code) == selected_norm:
            return item
        if selected_digits and extract_digits(item.code) == selected_digits:
            return item
    return None


def extract_stock_quantity(item: Optional[ResourceItem]) -> Optional[int]:
    """Purpose: Parse a numeric stock value from the catalog row if present.
    Inputs/Outputs: Inputs: item (ResourceItem|None). Outputs: int or None.
    Side Effects / State: None.
    Dependencies: get_raw_value, re.
    Failure Modes: Returns None when the field is missing or not numeric.
    If Removed: Quantity follow-ups cannot add stock-status messaging.
    Testing Notes: A value like "120" or "120 cái" should parse to 120.
    """
    if not item or not item.raw:
        return None
    value = get_raw_value(item.raw, ["Đơn vị", "Don vi"])
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def build_stock_status_line(stock_qty: Optional[int]) -> str:
    """Purpose: Build a stock-availability line using the configured thresholds.
    Inputs/Outputs: Inputs: stock_qty (int|None). Outputs: str (may be empty).
    Side Effects / State: None.
    Dependencies: None.
    Failure Modes: Returns empty string when stock is missing or invalid.
    If Removed: Responses will not include the requested stock status line.
    Testing Notes: 120 -> "có sẵn"; 80 -> "còn 80 cái"; None -> "".
    """
    if stock_qty is None:
        return ""
    if stock_qty >= 100:
        return "Hiện hàng đang có sẵn ạ."
    if stock_qty > 0:
        return f"Hiện kho còn {stock_qty} cái, số lượng còn lại sẽ cập nhật trong báo giá ạ."
    return ""


def insert_stock_line(answer: str, stock_line: str, form_block: str) -> str:
    """Purpose: Insert a stock line before the contact form if available.
    Inputs/Outputs: Inputs: answer (str), stock_line (str), form_block (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns the original answer if inputs are empty.
    If Removed: Stock messaging may appear in the wrong position or be missing.
    Testing Notes: Stock line should appear before the form block when present.
    """
    if not answer or not stock_line:
        return answer
    if normalize_text(stock_line) in normalize_text(answer):
        return answer.strip()
    if form_block and form_block in answer:
        return answer.replace(form_block, f"{stock_line}\n\n{form_block}", 1).strip()
    return f"{answer.strip()}\n\n{stock_line}".strip()


def build_quantity_context_json(context: PipelineContext) -> Dict[str, object]:
    """Purpose: Build context JSON for quantity follow-up prompt.
    Inputs/Outputs: Inputs: context (PipelineContext). Outputs: dict payload.
    Side Effects / State: None.
    Dependencies: find_item_by_selected_sku, detect_amp_line, detect_system_tag.
    Failure Modes: Returns minimal payload when anchor item is missing.
    If Removed: Quantity follow-up prompt will lack product anchoring.
    Testing Notes: With selected SKU, anchor fields should be populated.
    """
    # Assemble quantity-followup context for LLM prompts.
    selected = str(context.order_state.get("selected_sku") or "")
    quantity = context.order_state.get("quantity")
    item = find_item_by_selected_sku(context.catalog_items, selected)
    anchor: Dict[str, object] = {}
    stock_qty = extract_stock_quantity(item)
    stock_line = build_stock_status_line(stock_qty)
    if item:
        combined = f"{item.name} {item.description}"
        anchor = {
            "sku": item.code,
            "name": item.name,
            "category": item.category,
            "amp": detect_amp_line(combined),
            "system": detect_system_tag(combined),
        }

    return {
        "mode": "QUANTITY_FOLLOWUP",
        "quantity": quantity,
        "selected_sku": selected,
        "anchor": anchor,
        "stock_qty": stock_qty,
        "stock_line": stock_line,
        "should_show_form": context.should_show_form,
        "missing_contact": not bool(context.order_state.get("contact")),
        "form_block": FORM_BLOCK,
        "required_tail_sentence": (
            "Em sẽ chuyển đầy đủ thông tin để bên em phản hồi phương án phù hợp cho Anh/Chị ạ."
        ),
    }


def ensure_contains_form_and_tail(answer: str, form_block: str, tail: str) -> str:
    """Purpose: Enforce required form block and tail sentence in a response.
    Inputs/Outputs: Inputs: answer (str), form_block (str), tail (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns original answer when form/tail are empty.
    If Removed: Quantity follow-up may omit required contact form.
    Testing Notes: Missing form should be appended exactly once.
    """
    # Append missing form and tail strings if needed.
    output = answer.strip()
    if form_block and normalize_text(form_block) not in normalize_text(output):
        output = f"{output}\n\n{form_block}".strip()
    if tail and normalize_text(tail) not in normalize_text(output):
        output = f"{output}\n\n{tail}".strip()
    return output.strip()


def update_short_memory_from_context(context: PipelineContext) -> Dict[str, object]:
    """Purpose: Update short-memory slots from the current pipeline context.
    Inputs/Outputs: Inputs: context (PipelineContext). Outputs: updated memory dict.
    Side Effects / State: None; caller persists to order_state.
    Dependencies: normalize_short_memory, detect_* helpers, find_item_by_selected_sku.
    Failure Modes: Falls back to prior memory when no anchor is available.
    If Removed: Follow-up resolution will lose anchors and pending bundle state.
    Testing Notes: After a lookup, last_anchor and last_results should update.
    """
    # Derive short memory slots from the current turn.
    memory = normalize_short_memory({"short_memory": context.short_memory, "short_memory_ts": time.time()})
    memory["last_intent"] = context.intent_label
    memory["last_topic"] = context.intent_topic

    results = [item.code for item in context.display_items if item.code]
    if results:
        memory["last_results"] = results

    anchor_item = None
    if context.items:
        anchor_item = context.items[0]
    elif context.display_items:
        anchor_item = context.display_items[0]
    else:
        selected = context.order_state.get("selected_sku")
        if selected:
            anchor_item = find_item_by_selected_sku(context.catalog_items, str(selected))

    if anchor_item:
        combined = f"{anchor_item.name} {anchor_item.description}"
        line_amp = detect_amp_line(combined)
        is_robot = True if detect_item_type(anchor_item) == "ROBOT" else False
        if context.order_state.get("hand_or_robot") == "ROBOT":
            is_robot = True
        elif context.order_state.get("hand_or_robot") == "HAND":
            is_robot = False
        anchor_cat = detect_product_group(normalize_text(f"{anchor_item.name} {anchor_item.category}")) or (
            anchor_item.category or ""
        )
        memory["last_anchor"] = {
            "sku": anchor_item.code or "",
            "cat": anchor_cat,
            "line_amp": line_amp,
            "is_robot": is_robot,
            "name": anchor_item.name or "",
        }

    if context.intent_label == "ACCESSORY_BUNDLE_LOOKUP":
        required_parts = context.intent_entities.get("required_categories") or []
        missing_parts = context.intent_entities.get("missing_categories") or []
        prev_pending = memory.get("pending_request") or {}
        prev_required = prev_pending.get("required_parts") or []
        prev_done = prev_pending.get("done_parts") or []
        if prev_required and (not required_parts or set(required_parts).issubset(set(prev_required))):
            base_required = list(prev_required)
        else:
            base_required = list(required_parts)
        done_current = [part for part in required_parts if part not in missing_parts]
        done_parts = merge_unique(prev_done, done_current)
        todo_parts = [part for part in base_required if part not in done_parts]
        missing_fields: List[str] = []
        if base_required and not context.intent_entities.get("amp"):
            missing_fields.append("AMP")
        if base_required and not context.intent_entities.get("system"):
            missing_fields.append("SYSTEM")
        memory["pending_request"] = {
            "required_parts": base_required,
            "missing_fields": missing_fields,
            "done_parts": done_parts,
            "todo_parts": todo_parts,
        }
    else:
        memory["pending_request"] = {"required_parts": [], "missing_fields": [], "done_parts": [], "todo_parts": []}

    pending_action = memory.get("pending_action") or {}
    if context.resolved_request.get("clear_pending_action"):
        pending_action = {}
    elif (
        context.intent_label in TECHNICAL_INTENTS
        and not context.should_show_form
        and not context.is_asking_price
        and not context.is_availability_query
        and has_accessory_invite(context.answer_text)
    ):
        new_pending = build_pending_action_from_context(context)
        if new_pending:
            pending_action = new_pending
    memory["pending_action"] = pending_action if pending_action else {}

    constraints = dict(memory.get("last_user_constraints") or {})
    for key in ("amp", "size", "length", "thread", "material", "system"):
        if context.intent_entities.get(key) is not None:
            if key == "amp" and not AMP_ANY_RE.search(normalize_text(context.user_message)):
                continue
            constraints[key] = context.intent_entities.get(key)
    memory["last_user_constraints"] = constraints

    memory["last_commercial_context"] = {
        "quantity": context.order_state.get("quantity"),
        "contact_collected": bool(context.order_state.get("contact")),
        "show_form": context.should_show_form,
    }
    memory["updated_at"] = time.time()
    return memory


def build_code_lookup_mapping(items: List[ResourceItem], primary_code: str) -> str:
    """Purpose: Build mapping lines between external codes and Tokinarc SKUs.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), primary_code (str). Outputs: str.
    Side Effects / State: None.
    Dependencies: get_raw_value, extract_digits.
    Failure Modes: Returns empty string when no mapping fields are present.
    If Removed: CODE_LOOKUP responses may miss explicit mapping sentences.
    Testing Notes: D-part value should appear in mapping line.
    """
    # Compose mapping sentences for code lookup responses.
    lines: List[str] = []
    seen: set[str] = set()
    for item in items:
        sku_digits = extract_digits(item.code or "")
        sku = sku_digits or (item.code or "").strip()
        d_code = get_raw_value(
            item.raw,
            ["Mã D (D Part No.)", "D Part No.", "D Part No", "Mã D", "D Part No (D Part No.)"],
        )
        mapped_code = str(d_code).strip() if d_code else primary_code
        if not mapped_code or not sku:
            continue
        line = f"Mã {mapped_code} tương đương Tokinarc {sku}."
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines).strip()


def has_form_block(answer: str) -> bool:
    """Purpose: Detect whether the answer already contains the contact form block.
    Inputs/Outputs: Inputs: answer (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns False on empty input.
    If Removed: Form insertion may duplicate or be skipped incorrectly.
    Testing Notes: Answer containing "Ten cong ty" should return True.
    """
    # Check for form keywords in normalized answer.
    normalized = normalize_text(answer)
    return "ten cong ty" in normalized and "nguoi lien he" in normalized and "zalo" in normalized


def append_form_if_missing(answer: str) -> str:
    """Purpose: Append the standard form block if it is missing.
    Inputs/Outputs: Inputs: answer (str). Outputs: str with form appended if needed.
    Side Effects / State: None.
    Dependencies: has_form_block.
    Failure Modes: Returns original answer if already contains form.
    If Removed: Contact form may not appear in lead collection flows.
    Testing Notes: Missing form should be appended once.
    """
    # Add form block only when not already present.
    if has_form_block(answer):
        return answer.strip()
    return f"{answer.strip()}\n\n{FORM_BLOCK}"


def has_contact_reminder(answer: str) -> bool:
    """Purpose: Detect whether the answer contains a short contact reminder.
    Inputs/Outputs: Inputs: answer (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns False on empty input.
    If Removed: Reminder logic may repeat or omit follow-up prompts.
    Testing Notes: A reminder sentence with "lien he" should return True.
    """
    # Detect reminder keywords in normalized answer.
    normalized = normalize_text(answer)
    return "cho em xin" in normalized and "lien he" in normalized


def is_contact_request_line(line: str, allow_form: bool = True) -> bool:
    """Purpose: Check whether a line requests contact information.
    Inputs/Outputs: Inputs: line (str), allow_form (bool). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns False for empty lines.
    If Removed: Contact-reminder filtering will not work reliably.
    Testing Notes: Line with "cho em xin lien he" should return True.
    """
    # Identify lines that request contact info or form fields.
    normalized = normalize_text(line)
    if "ten cong ty" in normalized or "nguoi lien he" in normalized:
        return True
    if allow_form and normalized.startswith("zalo"):
        return True
    if "cho em xin" in normalized and ("lien he" in normalized or "thong tin" in normalized):
        return True
    if "so dien thoai" in normalized and "cho em xin" in normalized:
        return True
    if "sdt" in normalized and "cho em xin" in normalized:
        return True
    if "zalo" in normalized and "cho em xin" in normalized:
        return True
    return False


def detect_contact_info(message: str) -> bool:
    """Purpose: Detect if user message includes contact info.
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: PHONE_RE, normalize_text.
    Failure Modes: Returns False for non-numeric contact formats.
    If Removed: System will keep asking for contact even after user provides it.
    Testing Notes: "zalo 0909xxxx" should return True.
    """
    # Look for phone-like digits or zalo markers.
    normalized = normalize_text(message)
    if not normalized:
        return False
    if PHONE_RE.search(normalized):
        return True
    if "zalo" in normalized and re.search(r"\d{4,}", normalized):
        return True
    if "sdt" in normalized and re.search(r"\d{4,}", normalized):
        return True
    return False


def match_items_by_codes(items: List[ResourceItem], codes: List[str]) -> List[ResourceItem]:
    """Purpose: Match catalog items by normalized code or digits.
    Inputs/Outputs: Inputs: items (list[ResourceItem]), codes (list[str]). Outputs: list matches.
    Side Effects / State: None.
    Dependencies: normalize_text, extract_digits.
    Failure Modes: Returns empty list if no code matches.
    If Removed: Follow-up retrieval cannot re-anchor by prior codes.
    Testing Notes: "002005" should match item.code "Tokin 002005".
    """
    # Match by normalized code or numeric digits.
    if not items or not codes:
        return []
    code_set = {normalize_text(code) for code in codes if code}
    digit_set = {extract_digits(code) for code in codes if extract_digits(code)}
    matched = []
    for item in items:
        item_code_norm = normalize_text(item.code)
        item_code_digits = extract_digits(item.code)
        if item_code_norm in code_set or (item_code_digits and item_code_digits in digit_set):
            matched.append(item)
    return matched


def _get_contact_state(chat_history: List[dict]) -> Tuple[bool, int, bool, bool]:
    """Purpose: Derive contact-related state from chat history.
    Inputs/Outputs: Inputs: chat_history (list[dict]). Outputs: tuple(asked_form, reminder_count, contact_received, waiting_for_contact).
    Side Effects / State: None.
    Dependencies: has_form_block, is_contact_request_line, detect_contact_info.
    Failure Modes: Returns all False/0 for empty history.
    If Removed: Reminder logic and contact gating will be inconsistent.
    Testing Notes: After assistant asks form and user replies with phone, contact_received=True.
    """
    # Walk history to detect form asks and contact replies.
    asked_form = False
    reminder_count = 0
    contact_received = False
    waiting_for_contact = False
    for message in chat_history:
        role = message.get("role")
        content = message.get("content", "")
        meta = message.get("meta") or {}
        if role == "assistant":
            if meta.get("asked_form") is True or has_form_block(content):
                asked_form = True
                reminder_count = 0
                contact_received = False
                waiting_for_contact = True
                continue
            if asked_form:
                if meta.get("reminded_contact") is True or is_contact_request_line(content, allow_form=False):
                    reminder_count += 1
        if role == "user" and asked_form:
            if detect_contact_info(content):
                contact_received = True
                waiting_for_contact = False
    return asked_form, reminder_count, contact_received, waiting_for_contact


def normalize_order_state(state: Dict[str, object]) -> Dict[str, object]:
    """Purpose: Normalize order_state keys and fill defaults.
    Inputs/Outputs: Inputs: state (dict). Outputs: normalized dict.
    Side Effects / State: None; returns a new dict.
    Dependencies: None.
    Failure Modes: Missing keys default to empty values.
    If Removed: Callers may crash on missing keys or inconsistent defaults.
    Testing Notes: Should always return keys like selected_sku and last_constraints.
    """
    # Provide a stable schema for order_state across steps.
    return {
        "selected_sku": state.get("selected_sku"),
        "selected_group": state.get("selected_group"),
        "quantity": state.get("quantity"),
        "hand_or_robot": state.get("hand_or_robot"),
        "hand_or_robot_source": state.get("hand_or_robot_source"),
        "contact": state.get("contact"),
        "last_intent": state.get("last_intent"),
        "last_context_codes": state.get("last_context_codes") or [],
        "last_group": state.get("last_group"),
        "last_constraints": state.get("last_constraints") or {},
        "short_memory": state.get("short_memory") or {},
        "short_memory_ts": state.get("short_memory_ts"),
        "selling_scope_variant": state.get("selling_scope_variant"),
        "asked_hand_robot": bool(state.get("asked_hand_robot")),
        "asked_contact_form": bool(state.get("asked_contact_form")),
    }


def is_pure_quantity_message(message: str) -> bool:
    """Purpose: Detect messages that are only a quantity.
    Inputs/Outputs: Inputs: message (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text, re.fullmatch.
    Failure Modes: Returns False for mixed-content messages.
    If Removed: Quantity-only follow-ups may trigger incorrect retrieval.
    Testing Notes: "100" and "100 cai" should return True.
    """
    # Identify minimal quantity-only patterns.
    normalized = normalize_text(message)
    if not normalized:
        return False
    if re.fullmatch(r"\d{1,6}", normalized):
        return True
    if re.fullmatch(r"(mot)(\s+(cai|chiec|con|bo|cap))?", normalized):
        return True
    if re.fullmatch(r"\d{1,6}\s*(cai|chiec|con|bo|cap)", normalized):
        return True
    return False


def parse_pure_quantity_value(message: str) -> Optional[int]:
    """Purpose: Parse integer quantity from a quantity-only message.
    Inputs/Outputs: Inputs: message (str). Outputs: int or None.
    Side Effects / State: None.
    Dependencies: normalize_text, re.fullmatch.
    Failure Modes: Returns None for non-quantity strings.
    If Removed: Quantity-only flow will not capture numeric value.
    Testing Notes: "mot cai" should return 1; "100" should return 100.
    """
    # Convert short quantity messages to an integer value.
    normalized = normalize_text(message)
    if not normalized:
        return None
    if re.fullmatch(r"\d{1,6}", normalized):
        return int(normalized)
    match = re.fullmatch(r"(\d{1,6})\s*(cai|chiec|con|bo|cap)", normalized)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"mot(\s+(cai|chiec|con|bo|cap))?", normalized):
        return 1
    return None


def extract_contact_value(message: str) -> Optional[str]:
    """Purpose: Extract a phone/Zalo-like contact value from text.
    Inputs/Outputs: Inputs: message (str). Outputs: contact string or None.
    Side Effects / State: None.
    Dependencies: PHONE_RE, normalize_text.
    Failure Modes: Returns None if no digits are found.
    If Removed: order_state.contact will not be captured from user replies.
    Testing Notes: "zalo 0987" should return "0987".
    """
    # Extract a numeric contact token from the message.
    normalized = normalize_text(message)
    match = PHONE_RE.search(normalized)
    if match:
        return match.group(0)
    if "zalo" in normalized and re.search(r"\d{4,}", normalized):
        return re.search(r"\d{4,}", normalized).group(0)
    if "sdt" in normalized and re.search(r"\d{4,}", normalized):
        return re.search(r"\d{4,}", normalized).group(0)
    return None


def update_order_state_from_turn(context: PipelineContext) -> None:
    """Purpose: Update order_state slots from the current turn signals.
    Inputs/Outputs: Inputs: context (PipelineContext). Outputs: None.
    Side Effects / State: Mutates context.order_state fields.
    Dependencies: extract_skus, extract_codes, detect_product_group, extract_quantity.
    Failure Modes: Leaves previous state when no new signals are present.
    If Removed: Session state will not track SKU/quantity/type/contact changes.
    Testing Notes: Mentioning "robot" should set hand_or_robot to ROBOT.
    """
    # Apply parsed signals to the session order_state.
    state = normalize_order_state(context.order_state)
    normalized = normalize_text(context.user_message)
    entities = context.intent_entities or {}

    skus = entities.get("skus") or entities.get("sku") or []
    if isinstance(skus, str):
        skus = [skus]
    if not skus:
        skus = extract_skus(context.user_message)
    if not skus:
        _, primary_code = extract_codes(context.user_message)
        if primary_code and not primary_code.upper().startswith("U"):
            skus = [primary_code]
    if skus:
        state["selected_sku"] = skus[0]
    elif context.items and context.intent_label == "CODE_LOOKUP":
        first_item = context.items[0]
        if first_item.code:
            state["selected_sku"] = first_item.code

    product_group = entities.get("product_group")
    if isinstance(product_group, str):
        product_group = product_group.strip().upper()
    if not product_group:
        product_group = detect_product_group(normalized)
    if product_group:
        state["selected_group"] = product_group

    quantity = entities.get("quantity")
    if isinstance(quantity, str) and quantity.isdigit():
        quantity = int(quantity)
    if quantity is None:
        quantity = extract_quantity(normalized)
    if quantity is None and is_pure_quantity_message(context.user_message):
        if state.get("selected_sku") or state.get("selected_group"):
            quantity = parse_pure_quantity_value(context.user_message)
    if quantity is not None:
        state["quantity"] = quantity

    if "robot" in normalized:
        state["hand_or_robot"] = "ROBOT"
        state["hand_or_robot_source"] = "USER"
    elif "tay" in normalized or "hand" in normalized:
        state["hand_or_robot"] = "HAND"
        state["hand_or_robot_source"] = "USER"

    contact = extract_contact_value(context.user_message)
    if contact:
        state["contact"] = contact

    context.order_state = state


def remove_form_block(answer: str) -> str:
    """Purpose: Remove contact-form lines from an answer string.
    Inputs/Outputs: Inputs: answer (str). Outputs: cleaned answer (str).
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns original answer if no form lines found.
    If Removed: Commercial guardrails may leave duplicate or disallowed form text.
    Testing Notes: Lines with "Ten cong ty" should be removed.
    """
    # Strip known form lines from the response.
    if not answer:
        return answer
    cleaned_lines = []
    for line in answer.splitlines():
        normalized = normalize_text(line)
        if (
            "ten cong ty" in normalized
            or "nguoi lien he" in normalized
            or ("zalo" in normalized and ("cho em xin" in normalized or normalized.startswith("zalo")))
            or ("so dien thoai" in normalized and ("cho em xin" in normalized or normalized.startswith("so dien thoai")))
            or ("sdt" in normalized and ("cho em xin" in normalized or normalized.startswith("sdt")))
            or ("cho em xin" in normalized and ("thong tin" in normalized or "lien he" in normalized))
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def append_reminder_if_missing(answer: str) -> str:
    """Purpose: Append a short contact reminder if it is missing.
    Inputs/Outputs: Inputs: answer (str). Outputs: answer with reminder appended if needed.
    Side Effects / State: None.
    Dependencies: has_contact_reminder.
    Failure Modes: Returns original answer if reminder already present.
    If Removed: Follow-up reminders may not be sent when required.
    Testing Notes: Missing reminder should be appended once.
    """
    # Add a single reminder line when absent.
    if has_contact_reminder(answer):
        return answer.strip()
    return f"{answer.strip()}\n\n{REMINDER_LINE}"


def remove_contact_reminder(answer: str) -> str:
    """Purpose: Remove short contact reminder lines from an answer.
    Inputs/Outputs: Inputs: answer (str). Outputs: cleaned answer (str).
    Side Effects / State: None.
    Dependencies: is_contact_request_line.
    Failure Modes: Returns original answer if no reminder lines found.
    If Removed: Responses may include reminders when not allowed.
    Testing Notes: Reminder lines should be stripped from commercial responses.
    """
    # Strip reminder lines without touching main content.
    if not answer:
        return answer
    cleaned_lines = []
    for line in answer.splitlines():
        if is_contact_request_line(line, allow_form=False):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def remove_handoff_phrases(answer: str) -> str:
    """Purpose: Remove internal handoff phrases from technical responses.
    Inputs/Outputs: Inputs: answer (str). Outputs: cleaned answer (str).
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns original answer if no handoff phrases found.
    If Removed: Technical intents may incorrectly include handoff language.
    Testing Notes: "chuyen bo phan" should be removed.
    """
    # Filter phrases that indicate internal handoff.
    if not answer:
        return answer
    blockers = [
        "ghi nhan nhu cau",
        "chuyen bo phan",
        "bo phan phu trach",
        "phan hoi sau",
        "ben em phan hoi",
        "em se phan hoi",
        "de kho kiem tra",
        "kho kiem tra",
    ]
    cleaned_lines = []
    for line in answer.splitlines():
        normalized = normalize_text(line)
        if any(phrase in normalized for phrase in blockers):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def has_technical_closing_line(answer: str) -> bool:
    """Purpose: Check if answer already contains a technical closing line.
    Inputs/Outputs: Inputs: answer (str). Outputs: bool.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns False for empty answers.
    If Removed: Closing-line insertion may duplicate or misfire.
    Testing Notes: Answer containing "linh kien di kem" should return True.
    """
    # Detect known closing keywords to avoid duplicates.
    normalized = normalize_text(answer)
    if not normalized:
        return False
    keywords = [
        "day thep",
        "day nhom",
        "link anh",
        "hinh anh",
        "linh kien di kem",
        "rap dong bo",
        "doi chieu lai he robot",
    ]
    return any(key in normalized for key in keywords)


def pick_technical_closing_line(context: PipelineContext) -> str:
    """Purpose: Pick a deterministic technical closing line option.
    Inputs/Outputs: Inputs: context (PipelineContext). Outputs: str closing line.
    Side Effects / State: None.
    Dependencies: TECHNICAL_CLOSING_OPTIONS, context.chat_history length.
    Failure Modes: Returns empty string when no options are configured.
    If Removed: Technical responses lose a consistent call-to-action line.
    Testing Notes: Different history lengths should cycle through options.
    """
    # Use history length to rotate through configured options.
    if not TECHNICAL_CLOSING_OPTIONS:
        return ""
    idx = len(context.chat_history) % len(TECHNICAL_CLOSING_OPTIONS)
    return TECHNICAL_CLOSING_OPTIONS[idx]


def ensure_technical_closing_line(answer: str, context: PipelineContext) -> str:
    """Purpose: Ensure a technical closing line exists in the answer.
    Inputs/Outputs: Inputs: answer (str), context (PipelineContext). Outputs: str.
    Side Effects / State: None.
    Dependencies: has_technical_closing_line, pick_technical_closing_line.
    Failure Modes: Returns original answer when closing already present.
    If Removed: Technical responses may end abruptly without a soft CTA.
    Testing Notes: Missing closing line should be appended once.
    """
    # Append a closing line when one is missing.
    if has_technical_closing_line(answer):
        return answer.strip()
    closing = pick_technical_closing_line(context)
    if not closing:
        return answer.strip()
    return f"{answer.strip()}\n\n{closing}".strip()


def build_info_response(context: PipelineContext) -> str:
    """Purpose: Build a short factual response for info-only queries.
    Inputs/Outputs: Inputs: context (PipelineContext). Outputs: response string.
    Side Effects / State: None.
    Dependencies: get_raw_value, detect_system_tag, detect_amp_line.
    Failure Modes: Falls back to a request for more info when data is missing.
    If Removed: Info queries will be routed through generic generation.
    Testing Notes: "amp" question should return the detected amp when available.
    """
    # Respond briefly using catalog data when available.
    normalized = normalize_text(context.user_message)
    items = context.items or context.all_items
    item = items[0] if items else None

    if "xuat xu" in normalized or "nguon goc" in normalized:
        reply = "Dạ hàng Tokinarc (Nhật Bản) ạ. Anh/Chị cần em hỗ trợ mã nào ạ?"
    elif "vat lieu" in normalized or "chat lieu" in normalized:
        material = None
        if item:
            material = get_raw_value(item.raw, ["Vật liệu", "vat lieu", "material"])
        if material:
            sku = item.code or ""
            reply = f"Dạ vật liệu của mã {sku} là {material} ạ. Anh/Chị cần em hỗ trợ mã nào ạ?".strip()
        else:
            reply = "Dạ Anh/Chị cho em xin mã cụ thể để em trả lời đúng vật liệu ạ."
    elif "ampe" in normalized or "ampere" in normalized or "amp" in normalized:
        amp = None
        if item:
            combined = f"{item.name} {item.description}".lower()
            match = re.search(r"\b\d{3}a\b", combined)
            if match:
                amp = match.group(0).upper()
        if amp:
            reply = f"Dạ dòng {amp} ạ."
        else:
            reply = "Dạ Anh/Chị cho em xin mã hoặc dòng Ampe đang dùng để em kiểm tra đúng ạ."
    else:
        reply = "Dạ Anh/Chị cho em xin thông tin cụ thể để em trả lời chính xác ạ."

    return reply.strip()


def _normalize_category(text: str) -> str:
    """Purpose: Normalize category strings for consistent comparisons.
    Inputs/Outputs: Inputs: text (str). Outputs: normalized uppercase string.
    Side Effects / State: None.
    Dependencies: normalize_text.
    Failure Modes: Returns empty string when input is empty.
    If Removed: Category matching will be inconsistent across variants.
    Testing Notes: "Tip Body" should normalize to "TIPBODY".
    """
    # Normalize category for reliable group matching.
    normalized = normalize_text(text)
    normalized = normalized.replace("_", "").replace(" ", "")
    return normalized.upper()


def _dedupe_items(items: List[ResourceItem]) -> List[ResourceItem]:
    """Purpose: Deduplicate items using a composite key (code+cat+amp+system+type).
    Inputs/Outputs: Inputs: items (list[ResourceItem]). Outputs: deduped list.
    Side Effects / State: None.
    Dependencies: detect_amp_line, detect_system_tag, detect_item_type, extract_digits.
    Failure Modes: Returns empty list when input is empty.
    If Removed: Duplicate SKUs may appear in responses and logs.
    Testing Notes: Two identical entries should collapse to one.
    """
    # Remove duplicates using a composite identity key.
    seen = set()
    result = []
    for item in items:
        amp = detect_amp_line(f"{item.name} {item.description}")
        system = detect_system_tag(f"{item.name} {item.description}")
        cat = _normalize_category(item.category or "")
        typ = detect_item_type(item)
        code_digits = extract_digits(item.code or item.name or "")
        key = f"{code_digits}|{cat}|{amp}|{system}|{typ}"
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def insert_images_after_mentions(
    answer_text: str,
    items: List[ResourceItem],
    max_images: int,
) -> Tuple[str, List[dict]]:
    """Purpose: Insert Markdown images after paragraphs mentioning items.
    Inputs/Outputs: Inputs: answer_text (str), items (list[ResourceItem]), max_images (int).
        Outputs: (updated_text, images_meta).
    Side Effects / State: None.
    Dependencies: remove_markdown_images, strip_image_placeholders, sanitize_alt_text.
    Failure Modes: Returns original text when no candidates have URLs.
    If Removed: Images may not appear near their referenced products.
    Testing Notes: Paragraph mentioning SKU should receive its image.
    """
    # Insert images inline only when the paragraph mentions the item.
    candidates = [
        ImageCandidate(code=item.code or "", name=item.name or "", url=item.link)
        for item in items
        if item.link
    ]
    if not candidates:
        return answer_text, []

    cleaned = remove_markdown_images(answer_text)
    cleaned = strip_image_placeholders(cleaned)
    paragraphs = cleaned.split("\n\n") if cleaned else [""]

    used_urls = set()
    output: List[str] = []
    images: List[dict] = []

    for idx, paragraph in enumerate(paragraphs):
        output.append(paragraph)
        if len(images) >= max_images:
            continue
        for candidate in candidates:
            if candidate.url in used_urls:
                continue
            if _paragraph_mentions(paragraph, candidate):
                alt = sanitize_alt_text(candidate.name or candidate.code or "Hình ảnh sản phẩm")
                output.append(f"![{alt}]({candidate.url})")
                images.append({"url": candidate.url, "after_paragraph_index": idx})
                used_urls.add(candidate.url)
                if len(images) >= max_images:
                    break

    return "\n\n".join(part for part in output if part is not None).strip(), images


def remove_type_question(answer: str) -> str:
    """Purpose:
    Remove any explicit hand-vs-robot question line to enforce one-time asking.

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: cleaned answer (str).

    Side Effects / State:
    - None; does not mutate state.

    Dependencies:
    - normalize_text; used in generation post-processing.

    Failure Modes:
    - May remove lines that contain both keywords even if not a question.

    If Removed:
    - The bot can repeat the hand/robot question and violate routing rules.

    Testing Notes:
    - Include answers with and without the question; verify only the question line is removed.
    """
    # Drop the hand/robot question line if present.
    if not answer:
        return answer
    cleaned_lines = []
    for line in answer.splitlines():
        normalized = normalize_text(line)
        if ("tay" in normalized and "robot" in normalized) and ("hay" in normalized or "hoac" in normalized):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def remove_default_hand_note(answer: str) -> str:
    """Purpose:
    Remove the default hand-note line when it should not appear.

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: cleaned answer (str).

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text; used in generation post-processing.

    Failure Modes:
    - May remove unrelated lines containing the same tokens.

    If Removed:
    - The default hand note can be duplicated across turns, breaking UX rules.

    Testing Notes:
    - Use answers with and without the default hand note; confirm removal is correct.
    """
    # Remove the default hand note when a later guard disables it.
    if not answer:
        return answer
    cleaned_lines = []
    for line in answer.splitlines():
        normalized = normalize_text(line)
        if "mig 350a" in normalized or "tu van theo bo phu kien" in normalized:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def dedupe_sku_lines(answer: str) -> str:
    """Purpose:
    Remove duplicate SKU lines to avoid repeating the same product in one answer.

    Inputs/Outputs:
    - Inputs: answer (str) containing SKU lines.
    - Outputs: answer with duplicate SKU lines removed.

    Side Effects / State:
    - None; pure string cleanup.

    Dependencies:
    - extract_codes, CODE_RE, normalize_text.

    Failure Modes:
    - Can drop legitimate lines if codes are mis-detected.

    If Removed:
    - Responses may show duplicate SKU lines, violating anti-repeat rules.

    Testing Notes:
    - Provide answers with repeated SKUs and ensure only one instance remains.
    """
    # Track seen codes to drop repeated SKU lines.
    if not answer:
        return answer
    seen = set()
    output: List[str] = []
    for line in answer.splitlines():
        codes_in_line, _ = extract_codes(line)
        if not codes_in_line:
            matches = CODE_RE.findall(line)
            codes_in_line = [extract_digits(match) or match for match in matches if match]
        if not codes_in_line:
            output.append(line)
            continue
        normalized_codes = [normalize_text(match) for match in codes_in_line]
        line_norm = line.strip().lower()
        is_sku_line = line_norm.startswith(("sku", "-", "*", "•", "1.", "2.", "3.")) or "sku" in line_norm
        if is_sku_line and any(code in seen for code in normalized_codes):
            continue
        for code in normalized_codes:
            seen.add(code)
        output.append(line)
    return "\n".join(output).strip()


def prune_repeated_product_lines(
    answer: str, message: str, previous_codes: List[str], allow_repeat: bool = False
) -> str:
    """Purpose:
    Remove product lines already shown in prior turns unless repeat is requested.

    Inputs/Outputs:
    - Inputs: answer (str), message (str), previous_codes (list), allow_repeat (bool).
    - Outputs: pruned answer (str).

    Side Effects / State:
    - None; used for output post-processing only.

    Dependencies:
    - extract_codes, CODE_RE, normalize_text, LISTING_RE.

    Failure Modes:
    - Can over-prune if codes are mis-extracted or list intent is mis-detected.

    If Removed:
    - The bot may re-list products every turn, violating anti-repeat rules.

    Testing Notes:
    - Include a history with previously shown SKUs; verify pruning respects allow_repeat.
    """
    # Skip pruning when repeats are allowed or no history is present.
    if not answer or not previous_codes or allow_repeat:
        return answer

    is_listing = bool(LISTING_RE.search(normalize_text(message)))
    asked_codes = {normalize_text(code) for code in extract_codes(message)[0]}
    if not asked_codes:
        asked_codes = {normalize_text(code) for code in CODE_RE.findall(message)}
    code_set = {normalize_text(code) for code in previous_codes}
    code_set.update({extract_digits(code) for code in previous_codes if extract_digits(code)})

    if not is_listing and asked_codes:
        return answer

    output: List[str] = []
    for line in answer.splitlines():
        matches = extract_codes(line)[0]
        if not matches:
            matches = CODE_RE.findall(line)
        if not matches:
            output.append(line)
            continue
        normalized_codes = {normalize_text(code) for code in matches}
        normalized_codes.update({extract_digits(code) for code in matches if extract_digits(code)})
        if is_listing:
            continue
        if normalized_codes.issubset(code_set) and not normalized_codes.intersection(asked_codes):
            continue
        output.append(line)
    cleaned = "\n".join(output).strip()
    if not cleaned:
        return REPEAT_BLOCK_REPLY
    return cleaned


def _paragraph_mentions(paragraph: str, candidate: ImageCandidate) -> bool:
    """Purpose:
    Decide whether a paragraph references a product so an image can be inserted nearby.

    Inputs/Outputs:
    - Inputs: paragraph (str), candidate (ImageCandidate).
    - Outputs: bool.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text, extract_digits.

    Failure Modes:
    - False positives if short codes appear in unrelated text.

    If Removed:
    - Image insertion may stop or become inconsistent, breaking UI expectations.

    Testing Notes:
    - Test paragraphs that include code, digits-only code, and name matches.
    """
    # Match by code, digits, or name mention.
    para_norm = normalize_text(paragraph)
    code_norm = normalize_text(candidate.code)
    name_norm = normalize_text(candidate.name)
    code_digits = extract_digits(candidate.code)
    if code_norm and code_norm in para_norm:
        return True
    if code_digits and code_digits in para_norm:
        return True
    if name_norm and name_norm in para_norm:
        return True
    return False


def strip_image_placeholders(text: str) -> str:
    """Purpose:
    Remove placeholder lines like "product image" before re-inserting real images.

    Inputs/Outputs:
    - Inputs: text (str).
    - Outputs: cleaned text (str).

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text.

    Failure Modes:
    - May remove legitimate content if it matches placeholder tokens.

    If Removed:
    - Placeholders may leak into user-visible output.

    Testing Notes:
    - Include placeholder-only lines and ensure they are removed.
    """
    # Drop common placeholder tokens used by LLMs.
    if not text:
        return text
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        normalized = normalize_text(line)
        if not normalized:
            cleaned_lines.append(line)
            continue
        if normalized in {"product image", "hinh anh san pham", "anh san pham", "image"}:
            continue
        if normalized.startswith("product image") or normalized.startswith("hinh anh") or normalized.startswith(
            "anh san pham"
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def remove_markdown_images(text: str) -> str:
    """Purpose:
    Strip markdown image tags to avoid duplicated images in post-processing.

    Inputs/Outputs:
    - Inputs: text (str).
    - Outputs: text (str) without markdown image syntax.

    Side Effects / State:
    - None.

    Dependencies:
    - re module for regex removal.

    Failure Modes:
    - Over-removal if non-image markdown matches the pattern.

    If Removed:
    - Images can be duplicated or placed incorrectly by later steps.

    Testing Notes:
    - Verify that only ![alt](url) patterns are removed.
    """
    # Remove markdown image tags before re-inserting curated images.
    return re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)


def sanitize_alt_text(text: str) -> str:
    """Purpose:
    Normalize text used as markdown image alt text.

    Inputs/Outputs:
    - Inputs: text (str).
    - Outputs: cleaned alt text (str).

    Side Effects / State:
    - None.

    Dependencies:
    - None beyond basic string ops.

    Failure Modes:
    - Empty input yields a generic fallback alt label.

    If Removed:
    - Alt text may include brackets or extra whitespace, degrading UI quality.

    Testing Notes:
    - Pass text with brackets and extra spaces; ensure output is compact.
    """
    # Strip brackets and collapse whitespace for alt text safety.
    cleaned = " ".join(text.replace("[", "").replace("]", "").split())
    return cleaned or "Hình ảnh sản phẩm"


def append_line_if_missing(answer: str, line: str, marker: str) -> str:
    """Purpose:
    Append a line if the marker token is not already present in the answer.

    Inputs/Outputs:
    - Inputs: answer (str), line (str), marker (str).
    - Outputs: answer (str) with the line appended if missing.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text.

    Failure Modes:
    - Marker collisions can prevent appending when expected.

    If Removed:
    - Required lines (e.g., quantity question) may be omitted from responses.

    Testing Notes:
    - Verify that repeated calls do not duplicate the line.
    """
    # Append only when the marker token is absent.
    normalized = normalize_text(answer)
    if marker in normalized:
        return answer.strip()
    return f"{answer.strip()}\n\n{line}"


def append_quantity_question(answer: str, target: Optional[str]) -> str:
    """Purpose:
    Add a quantity question, optionally scoped to a target SKU/group.

    Inputs/Outputs:
    - Inputs: answer (str), target (str or None).
    - Outputs: answer (str) with a quantity prompt.

    Side Effects / State:
    - None.

    Dependencies:
    - append_line_if_missing.

    Failure Modes:
    - None beyond possible duplicate text if marker is not stable.

    If Removed:
    - The bot will not request quantity in flows that require it.

    Testing Notes:
    - Test both with and without target; ensure marker prevents duplicates.
    """
    # Use a stable marker so the question is added only once.
    marker = "so luong"
    question = "Anh/Chị cho em xin số lượng dự kiến ạ."
    if target:
        question = f"Anh/Chị cho em xin số lượng dự kiến cho {target} ạ."
    return append_line_if_missing(answer, question, marker)


def remove_quantity_request(answer: str) -> str:
    """Purpose:
    Remove quantity request lines when they are not allowed (commercial guards).

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: answer (str) with quantity prompts removed.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text.

    Failure Modes:
    - May remove lines containing the same tokens in another context.

    If Removed:
    - The bot may ask quantity in commercial/availability replies, violating rules.

    Testing Notes:
    - Include lines with "so luong" and ensure they are removed.
    """
    # Strip lines that ask for quantity.
    if not answer:
        return answer
    cleaned_lines: List[str] = []
    for line in answer.splitlines():
        normalized = normalize_text(line)
        if "so luong" in normalized and ("cho em xin" in normalized or "du kien" in normalized):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def remove_commercial_commitments(answer: str) -> str:
    """Purpose:
    Remove phrases that imply stock or availability commitments.

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: sanitized answer (str).

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text.

    Failure Modes:
    - Over-removal when a sentence contains blocked tokens in a different meaning.

    If Removed:
    - Commercial/availability rules may be violated by assertive language.

    Testing Notes:
    - Ensure phrases like "co san" are removed while "khong ban" stays.
    """
    # Strip banned commitment phrases while preserving negative statements.
    if not answer:
        return answer
    blocked = [
        "ben em co ban",
        "ben em co san",
        "ben em co cung cap",
        "co ban",
        "con hang",
        "dang co",
        "co san",
        "co cung cap",
    ]
    cleaned_lines: List[str] = []
    for line in answer.splitlines():
        normalized = normalize_text(line)
        if any(phrase in normalized for phrase in blocked) and "khong ban" not in normalized:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def convert_raw_image_links_to_markdown(answer: str) -> str:
    """Purpose:
    Convert raw image URL lines into markdown image syntax for rendering.

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: answer (str) with markdown image tags.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text and regex for URL detection.

    Failure Modes:
    - May miss URLs if they are not on their own line or not prefixed by label.

    If Removed:
    - UI may show raw image URLs instead of images.

    Testing Notes:
    - Provide lines starting with "Anh"/"Image" and URLs; verify conversion.
    """
    # Convert labeled URL lines into markdown image syntax.
    if not answer:
        return answer
    output: List[str] = []
    for line in answer.splitlines():
        line_norm = normalize_text(line)
        match = re.search(r"(https?://\S+)", line)
        if match and (line_norm.startswith("anh") or line_norm.startswith("image")):
            url = match.group(1).rstrip(").,")
            output.append(f"![Hinh anh san pham]({url})")
        else:
            output.append(line)
    return "\n".join(output).strip()


def enforce_tokin_code_wording(answer: str, primary_code: str) -> str:
    """Purpose:
    Replace "equivalent" wording when the input code is already Tokin/Tokinarc.

    Inputs/Outputs:
    - Inputs: answer (str), primary_code (str).
    - Outputs: answer (str) with corrected wording.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text, format_tokin.

    Failure Modes:
    - If the answer already contains the corrected wording, it remains unchanged.

    If Removed:
    - Responses may incorrectly claim equivalence for native Tokin codes.

    Testing Notes:
    - Test with Tokin and non-Tokin inputs; verify phrasing is correct.
    """
    # Drop "equivalent" wording for Tokin-origin inputs.
    if not answer:
        return answer
    if "tuong duong" not in normalize_text(answer):
        return answer
    sku = format_tokin(primary_code or "")
    if not sku:
        return answer
    cleaned_lines = [line for line in answer.splitlines() if "tuong duong" not in normalize_text(line)]
    cleaned = "\n".join(line for line in cleaned_lines if line.strip()).strip()
    prefix = f"Dạ vâng ạ, mã Tokinarc {sku} bên Autoss đang phân phối ạ."
    if normalize_text(prefix) in normalize_text(cleaned):
        return cleaned or prefix
    if cleaned:
        return f"{prefix}\n\n{cleaned}".strip()
    return prefix


def answer_mentions_any_code(answer: str, items: List[ResourceItem]) -> bool:
    """Purpose:
    Check whether an answer already mentions any SKU from the provided items.

    Inputs/Outputs:
    - Inputs: answer (str), items (list[ResourceItem]).
    - Outputs: bool.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text, extract_digits.

    Failure Modes:
    - False negatives if codes are formatted differently than expected.

    If Removed:
    - The system may duplicate product cards unnecessarily.

    Testing Notes:
    - Verify detection for full codes and digits-only forms.
    """
    # Detect whether a SKU code already appears in the answer text.
    if not answer or not items:
        return False
    normalized = normalize_text(answer)
    for item in items:
        if not item.code:
            continue
        code_norm = normalize_text(item.code)
        code_digits = extract_digits(item.code)
        if (code_norm and code_norm in normalized) or (code_digits and code_digits in normalized):
            return True
    return False


def ensure_product_cards(answer: str, items: List[ResourceItem], include_type_line: bool = False) -> str:
    """Purpose:
    Ensure that product cards are present when required by policy.

    Inputs/Outputs:
    - Inputs: answer (str), items (list[ResourceItem]), include_type_line (bool).
    - Outputs: answer (str) with cards inserted if missing.

    Side Effects / State:
    - None.

    Dependencies:
    - render_product_cards, answer_mentions_any_code, DEFAULT_PRICE_REPLY.

    Failure Modes:
    - May add cards in the wrong place if the answer has unexpected format.

    If Removed:
    - Commercial/availability replies may miss mandatory product evidence.

    Testing Notes:
    - Check cases with and without existing SKU mentions.
    """
    # Inject product cards when the answer lacks explicit SKU mentions.
    if not items or answer_mentions_any_code(answer, items):
        return answer.strip()
    cards = render_product_cards(items, limit=3, include_type_line=include_type_line)
    if not cards:
        return answer.strip()
    if DEFAULT_PRICE_REPLY in answer:
        prefix, _, _ = answer.partition(DEFAULT_PRICE_REPLY)
        prefix = prefix.strip()
        if prefix:
            prefix = f"{prefix}\n\n{cards}"
        else:
            prefix = cards
        return f"{prefix}\n\n{DEFAULT_PRICE_REPLY}"
    if answer.strip():
        return f"{answer.strip()}\n\n{cards}"
    return cards


def ensure_neutral_sentence(answer: str) -> str:
    """Purpose:
    Append the mandatory neutral commercial sentence as the final line.

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: answer (str) with the neutral sentence appended.

    Side Effects / State:
    - None.

    Dependencies:
    - DEFAULT_PRICE_REPLY.

    Failure Modes:
    - None; always returns a string.

    If Removed:
    - Commercial/availability responses may miss the required closing line.

    Testing Notes:
    - Validate that the neutral sentence appears exactly once at the end.
    """
    # Enforce the standard neutral commercial sentence.
    if not answer:
        return DEFAULT_PRICE_REPLY
    cleaned = answer.replace(DEFAULT_PRICE_REPLY, "").strip()
    if cleaned:
        return f"{cleaned}\n\n{DEFAULT_PRICE_REPLY}"
    return DEFAULT_PRICE_REPLY


def remove_product_lines(answer: str) -> str:
    """Purpose:
    Remove product listing lines when rendering is disabled for a response.

    Inputs/Outputs:
    - Inputs: answer (str).
    - Outputs: answer (str) without product lines.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text, extract_codes, CODE_RE.

    Failure Modes:
    - May remove lines that resemble SKU lines but are not product listings.

    If Removed:
    - Product details could leak into info-only or guardrail replies.

    Testing Notes:
    - Provide responses with SKU lines and ensure they are removed.
    """
    # Remove SKU-style and product metadata lines from the answer.
    if not answer:
        return answer
    cleaned_lines = []
    for line in answer.splitlines():
        line_norm = normalize_text(line)
        if not line_norm:
            cleaned_lines.append(line)
            continue
        if any(token in line_norm for token in ("sku", "specs", "cat", "img")):
            continue
        codes_in_line, _ = extract_codes(line)
        if (CODE_RE.search(line) or codes_in_line) and line_norm.startswith(("sku", "-", "*", "1", "2", "3", "•")):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def insert_missing_image_notice(answer_text: str, items: List[ResourceItem]) -> str:
    """Purpose:
    Insert a notice when items lack image URLs, near their mentions.

    Inputs/Outputs:
    - Inputs: answer_text (str), items (list[ResourceItem]).
    - Outputs: answer_text (str) with notice lines inserted.

    Side Effects / State:
    - None.

    Dependencies:
    - normalize_text, MISSING_IMAGE_NOTICE.

    Failure Modes:
    - Notice may be placed in the wrong paragraph if mention matching fails.

    If Removed:
    - Users will not be told why images are missing for certain items.

    Testing Notes:
    - Use items without links and verify the notice appears once per item mention.
    """
    # Add a single notice per item without an image URL.
    if not answer_text or not items:
        return answer_text
    notice_norm = normalize_text(MISSING_IMAGE_NOTICE)
    if notice_norm in normalize_text(answer_text):
        return answer_text

    missing_items = [item for item in items if not item.link]
    if not missing_items:
        return answer_text

    paragraphs = answer_text.split("\n\n")
    inserted: set[str] = set()
    output: List[str] = []

    for paragraph in paragraphs:
        output.append(paragraph)
        para_norm = normalize_text(paragraph)
        if not para_norm:
            continue
        for item in missing_items:
            key = normalize_text(item.code or item.name)
            if not key or key in inserted:
                continue
            if key in para_norm:
                output.append(MISSING_IMAGE_NOTICE)
                inserted.add(key)
                break

    return "\n\n".join(part for part in output if part is not None).strip()
